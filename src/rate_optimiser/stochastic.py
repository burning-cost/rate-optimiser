"""
Stochastic rate optimisation via chance constraints (Branda 2013).

Replaces the deterministic loss ratio constraint with a chance constraint:

    P(portfolio LR <= target) >= alpha

Under the assumption that aggregate losses are approximately normally distributed
(central limit theorem; reasonable for large books), this reformulates to:

    E[LR] + z_alpha * sigma[LR] <= target

where z_alpha is the normal quantile at confidence level alpha, and sigma[LR]
is the standard deviation of the portfolio loss ratio given the rate strategy.

This formulation requires variance estimates for expected claims — available
from any GLM with an overdispersion parameter (Tweedie, negative binomial,
quasi-Poisson) or from a cell-level claim count model.

The module uses cvxpy if available for the convex reformulation. If cvxpy is
not installed, falls back to scipy SLSQP with the deterministic reformulation
(which is already convex in the relevant terms).

References
----------
Branda, M. (2013). "Optimization Approaches to Multiplicative Tariff of Rates."
ASTIN Colloquium, Hague.

Charnes, A. and Cooper, W. W. (1959). "Chance-Constrained Programming."
Management Science 6(1):73-79.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from rate_optimiser.data import PolicyData, FactorStructure
from rate_optimiser.demand import DemandModel
from rate_optimiser.optimiser import RateChangeOptimiser, OptimiserResult
from rate_optimiser.constraints import (
    Constraint,
    _compute_adjusted_premiums,
    _compute_renewal_probs,
)


def _check_cvxpy() -> bool:
    """Return True if cvxpy is importable."""
    try:
        import cvxpy  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class ClaimsVarianceModel:
    """
    Per-policy variance estimates for expected claims.

    In a Tweedie GLM, the variance of the claims amount Y_i is:

        Var(Y_i) = phi * mu_i^p

    where mu_i is the fitted mean (technical premium), phi is the dispersion
    parameter, and p is the Tweedie power parameter (typically 1.5 for combined
    frequency-severity, or 1.0 for Poisson frequency).

    For a portfolio, the aggregate loss variance is approximately:

        Var(L) = sum_i p_i * Var(Y_i) + sum_i p_i * (1-p_i) * mu_i^2

    The second term captures the variance from uncertainty in whether each
    policy actually renews (Bernoulli variance on the selection process).

    Parameters
    ----------
    mean_claims : np.ndarray
        Shape (n_policies,). Expected claims per policy (technical premium).
    variance_claims : np.ndarray
        Shape (n_policies,). Per-policy claims variance from the GLM.
    """

    mean_claims: np.ndarray
    variance_claims: np.ndarray

    @classmethod
    def from_tweedie(
        cls,
        mean_claims: np.ndarray,
        dispersion: float,
        power: float = 1.5,
    ) -> "ClaimsVarianceModel":
        """
        Construct from Tweedie GLM outputs.

        Parameters
        ----------
        mean_claims : np.ndarray
            Fitted means from the Tweedie GLM.
        dispersion : float
            Dispersion parameter phi (from GLM summary).
        power : float
            Tweedie power parameter. Default 1.5 (insurance compound model).
        """
        variance = dispersion * np.power(mean_claims, power)
        return cls(mean_claims=mean_claims, variance_claims=variance)

    @classmethod
    def from_overdispersed_poisson(
        cls,
        expected_counts: np.ndarray,
        mean_severity: np.ndarray,
        severity_variance: np.ndarray,
        overdispersion: float = 1.0,
    ) -> "ClaimsVarianceModel":
        """
        Construct from a frequency-severity decomposition.

        Var(aggregate) = expected_count * severity_variance
                       + mean_severity^2 * expected_count * overdispersion

        Parameters
        ----------
        expected_counts : np.ndarray
            Expected claim counts per policy.
        mean_severity : np.ndarray
            Expected severity per claim.
        severity_variance : np.ndarray
            Variance of severity per claim.
        overdispersion : float
            Overdispersion relative to Poisson. 1.0 = Poisson.
        """
        mean_claims = expected_counts * mean_severity
        var_claims = (
            expected_counts * severity_variance
            + mean_severity**2 * expected_counts * overdispersion
        )
        return cls(mean_claims=mean_claims, variance_claims=var_claims)


class ChanceConstrainedLRConstraint(Constraint):
    """
    Chance-constrained loss ratio: P(LR <= target) >= alpha.

    Under normal approximation, reformulates to:

        E[LR] + z_alpha * sigma[LR] <= target

    The sigma[LR] term requires per-policy claims variance from a GLM.

    Parameters
    ----------
    bound : float
        Maximum loss ratio at confidence level alpha.
    alpha : float
        Confidence level. E.g., 0.95 means the constraint must hold with
        95% probability. Higher alpha is more conservative.
    variance_model : ClaimsVarianceModel
        Per-policy claims variance estimates.
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        bound: float,
        alpha: float,
        variance_model: ClaimsVarianceModel,
        name: str = "chance_lr",
    ) -> None:
        if not 0.5 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0.5, 1.0), got {alpha}.")
        if not 0 < bound < 2:
            raise ValueError(f"bound={bound} is implausible. Expected (0, 2).")
        self.bound = bound
        self.alpha = alpha
        self.z_alpha = float(stats.norm.ppf(alpha))
        self.variance_model = variance_model
        self.name = name

    def evaluate(
        self,
        adjustments: np.ndarray,
        data: pd.DataFrame,
        factor_structure,
        demand_model,
    ) -> float:
        """
        Returns bound - (E[LR] + z_alpha * sigma[LR]).
        Positive when the chance constraint is satisfied.
        """
        e_lr, sigma_lr = self._compute_lr_moments(
            adjustments, data, factor_structure, demand_model
        )
        chance_lr = e_lr + self.z_alpha * sigma_lr
        return self.bound - chance_lr

    def _compute_lr_moments(
        self,
        adjustments: np.ndarray,
        data: pd.DataFrame,
        factor_structure,
        demand_model,
    ) -> tuple[float, float]:
        """
        Compute E[LR] and sigma[LR] at the given factor adjustments.

        E[LR] = sum_i(p_i * c_i) / sum_i(p_i * pi_i)

        For sigma[LR], we use the delta method on the aggregate:

            sigma[LR] ≈ sigma[L] / E[P]

        where L = aggregate losses and P = aggregate premium, and
        sigma^2[L] = sum_i p_i * Var(Y_i) + sum_i p_i*(1-p_i)*c_i^2
        """
        probs = _compute_renewal_probs(adjustments, data, factor_structure, demand_model)
        adjusted_premiums = _compute_adjusted_premiums(adjustments, data, factor_structure)
        claims = self.variance_model.mean_claims
        claim_var = self.variance_model.variance_claims

        expected_claims = np.dot(probs, claims)
        expected_premium = np.dot(probs, adjusted_premiums)

        if expected_premium < 1e-10:
            return 1.0, 0.0

        e_lr = expected_claims / expected_premium

        # Variance of aggregate losses (two components):
        # 1. Claims variance given retention: sum p_i * Var(Y_i)
        # 2. Retention uncertainty: sum p_i*(1-p_i) * c_i^2
        var_claims_given_retention = np.dot(probs, claim_var)
        var_retention_uncertainty = np.dot(probs * (1 - probs), claims**2)
        var_aggregate_losses = var_claims_given_retention + var_retention_uncertainty

        sigma_aggregate_losses = np.sqrt(max(var_aggregate_losses, 0.0))
        sigma_lr = sigma_aggregate_losses / expected_premium

        return e_lr, sigma_lr

    def to_scipy_dict(self, data, factor_structure, demand_model) -> dict:
        def fun(adj):
            return self.evaluate(adj, data, factor_structure, demand_model)

        return {"type": "ineq", "fun": fun}

    def __repr__(self) -> str:
        return (
            f"ChanceConstrainedLRConstraint("
            f"bound={self.bound}, alpha={self.alpha}, "
            f"z_alpha={self.z_alpha:.3f}, name='{self.name}')"
        )


class StochasticRateOptimiser(RateChangeOptimiser):
    """
    Rate optimiser with chance-constrained loss ratio.

    Extends RateChangeOptimiser to add a Branda-style chance constraint.
    The LR constraint becomes P(LR <= target) >= alpha rather than E[LR] <= target.

    Parameters
    ----------
    data : PolicyData
        Policy-level input data.
    demand : DemandModel
        Demand model.
    factor_structure : FactorStructure
        Tariff factor structure.
    variance_model : ClaimsVarianceModel
        Per-policy claims variance from GLM.
    lr_bound : float
        Loss ratio target.
    alpha : float
        Required probability that LR stays below the target. Default 0.95.
    objective : str
        Objective function type. See RateChangeOptimiser.

    Examples
    --------
    >>> variance_model = ClaimsVarianceModel.from_tweedie(
    ...     mean_claims=data.df["technical_premium"].values,
    ...     dispersion=1.2,
    ...     power=1.5,
    ... )
    >>> opt = StochasticRateOptimiser(
    ...     data=data, demand=demand, factor_structure=fs,
    ...     variance_model=variance_model,
    ...     lr_bound=0.72, alpha=0.95,
    ... )
    >>> result = opt.solve()
    """

    def __init__(
        self,
        data: PolicyData,
        demand: DemandModel,
        factor_structure: FactorStructure,
        variance_model: ClaimsVarianceModel,
        lr_bound: float = 0.72,
        alpha: float = 0.95,
        **kwargs,
    ) -> None:
        super().__init__(data=data, demand=demand, factor_structure=factor_structure, **kwargs)
        self._variance_model = variance_model
        self._lr_bound = lr_bound
        self._alpha = alpha

        chance_constraint = ChanceConstrainedLRConstraint(
            bound=lr_bound,
            alpha=alpha,
            variance_model=variance_model,
        )
        self.add_constraint(chance_constraint)

    @property
    def variance_model(self) -> ClaimsVarianceModel:
        return self._variance_model

    def __repr__(self) -> str:
        return (
            f"StochasticRateOptimiser("
            f"n_factors={self.n_factors}, "
            f"n_policies={self._data.n_policies}, "
            f"lr_bound={self._lr_bound}, "
            f"alpha={self._alpha})"
        )
