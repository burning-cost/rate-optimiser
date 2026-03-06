"""
Constraint classes for rate change optimisation.

Constraints are added to a RateChangeOptimiser via add_constraint(). Each
constraint class knows how to evaluate itself given factor adjustments and
data, and how to express itself in the format scipy.optimize.minimize expects.

scipy SLSQP uses the convention: inequality constraints are >= 0 (i.e., the
function value must be non-negative at a feasible point). We follow this
convention throughout.

The Lagrange multipliers (shadow prices) reported by SLSQP are extracted from
the OptimizeResult.v attribute. A binding constraint has a non-zero shadow price;
the magnitude tells you how much the objective would improve if the constraint
were relaxed by one unit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import polars as pl


class Constraint(ABC):
    """
    Base class for all rate optimisation constraints.

    Subclasses implement ``evaluate`` (scalar feasibility measure) and
    ``to_scipy_dict`` (the dict format expected by scipy.optimize.minimize).
    """

    name: str

    @abstractmethod
    def evaluate(
        self,
        adjustments: np.ndarray,
        data: pl.DataFrame,
        factor_structure,
        demand_model,
    ) -> float:
        """
        Evaluate the constraint at the given factor adjustments.

        Returns a scalar. The constraint is satisfied when this value >= 0
        (for inequality constraints) or == 0 (for equality constraints).

        Parameters
        ----------
        adjustments : np.ndarray
            Current factor adjustments (decision variables).
        data : pl.DataFrame
            Policy-level data.
        factor_structure : FactorStructure
            Tariff structure for computing adjusted premiums.
        demand_model : DemandModel
            Demand model for computing retention probabilities.
        """

    @abstractmethod
    def to_scipy_dict(
        self,
        data: pl.DataFrame,
        factor_structure,
        demand_model,
    ) -> dict:
        """Return a dict in scipy.optimize.minimize constraint format."""


class LossRatioConstraint(Constraint):
    """
    Upper bound on portfolio-level expected loss ratio.

    Enforces: E[LR] <= bound

    The expected loss ratio is computed as:

        E[LR] = sum_i(p_i * c_i) / sum_i(p_i * pi_i)

    where p_i is the demand-model probability, c_i is the technical premium
    (expected claims proxy), and pi_i is the adjusted premium.

    Parameters
    ----------
    bound : float
        Maximum allowable loss ratio. E.g., 0.72 for 72% LR target.
    name : str
        Constraint identifier used in shadow price reporting.
    """

    def __init__(self, bound: float, name: str = "loss_ratio_ub") -> None:
        if not 0 < bound < 2:
            raise ValueError(f"loss_ratio bound={bound} is implausible. Expected (0, 2).")
        self.bound = bound
        self.name = name

    def evaluate(
        self,
        adjustments: np.ndarray,
        data: pl.DataFrame,
        factor_structure,
        demand_model,
    ) -> float:
        """
        Returns bound - E[LR]. Positive when constraint is satisfied.
        """
        expected_lr = _compute_expected_lr(adjustments, data, factor_structure, demand_model)
        return self.bound - expected_lr

    def to_scipy_dict(self, data, factor_structure, demand_model) -> dict:
        def fun(adj):
            return self.evaluate(adj, data, factor_structure, demand_model)

        return {"type": "ineq", "fun": fun}

    def __repr__(self) -> str:
        return f"LossRatioConstraint(bound={self.bound}, name='{self.name}')"


class VolumeConstraint(Constraint):
    """
    Lower bound on expected policy count relative to the baseline expected count.

    Enforces: sum_i(p_i(adj)) / sum_i(p_i(current)) >= bound

    The denominator is the sum of renewal probabilities at current pricing
    (the ``renewal_prob`` column in PolicyData). This makes the constraint
    relative to what would be expected at the current rate level, not to the
    total policy count.

    A bound of 0.97 means: the optimised rates can cause at most a 3% reduction
    in expected renewals relative to the current pricing. If the current renewal
    rate is 72%, the constraint is that it should not fall below 72% * 0.97 = 69.8%.

    Parameters
    ----------
    bound : float
        Minimum acceptable volume ratio relative to current expected volume.
        E.g., 0.97 means no more than 3% loss of expected renewals.
    name : str
        Constraint identifier.
    """

    def __init__(self, bound: float, name: str = "volume_lb") -> None:
        if not 0 < bound <= 1:
            raise ValueError(
                f"volume bound={bound} is implausible. Expected (0, 1]."
            )
        self.bound = bound
        self.name = name

    def evaluate(
        self,
        adjustments: np.ndarray,
        data: pl.DataFrame,
        factor_structure,
        demand_model,
    ) -> float:
        """
        Returns E[volume_ratio] - bound. Positive when constraint is satisfied.
        """
        vol_ratio = _compute_volume_ratio(adjustments, data, factor_structure, demand_model)
        return vol_ratio - self.bound

    def to_scipy_dict(self, data, factor_structure, demand_model) -> dict:
        def fun(adj):
            return self.evaluate(adj, data, factor_structure, demand_model)

        return {"type": "ineq", "fun": fun}

    def __repr__(self) -> str:
        return f"VolumeConstraint(bound={self.bound}, name='{self.name}')"


class FactorBoundsConstraint(Constraint):
    """
    Bounds on individual factor adjustments.

    Expressed as a scipy ``bounds`` object (used in the ``bounds`` argument to
    scipy.optimize.minimize, not as constraints). However, the class also
    provides an ``evaluate`` method for feasibility checking.

    Parameters
    ----------
    lower : float or np.ndarray
        Minimum adjustment multiplier per factor. Scalar applies to all factors.
        E.g., 0.90 means factors can be reduced by at most 10%.
    upper : float or np.ndarray
        Maximum adjustment multiplier per factor. Scalar applies to all factors.
        E.g., 1.15 means factors can be increased by at most 15%.
    n_factors : int
        Number of rating factors (length of decision variable vector).
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
        n_factors: int,
        name: str = "factor_bounds",
    ) -> None:
        self.n_factors = n_factors
        self.name = name
        self.lower = np.broadcast_to(np.asarray(lower, dtype=float), (n_factors,)).copy()
        self.upper = np.broadcast_to(np.asarray(upper, dtype=float), (n_factors,)).copy()
        if (self.lower <= 0).any():
            raise ValueError("factor lower bounds must be strictly positive.")
        if (self.lower > self.upper).any():
            raise ValueError("factor lower bounds must be <= upper bounds.")

    def to_scipy_bounds(self) -> list[tuple[float, float]]:
        """Return as a list of (lower, upper) tuples for scipy.optimize.minimize bounds."""
        return list(zip(self.lower.tolist(), self.upper.tolist()))

    def evaluate(
        self,
        adjustments: np.ndarray,
        data=None,
        factor_structure=None,
        demand_model=None,
    ) -> float:
        """
        Returns the minimum slack across all factor bounds.
        Positive if all bounds are satisfied.
        """
        lower_slack = adjustments - self.lower
        upper_slack = self.upper - adjustments
        return float(min(lower_slack.min(), upper_slack.min()))

    def to_scipy_dict(self, data, factor_structure, demand_model) -> dict:
        # Factor bounds are handled via the bounds parameter, not as constraints.
        # This method exists for interface consistency.
        raise NotImplementedError(
            "FactorBoundsConstraint uses scipy bounds, not constraint dicts. "
            "Call to_scipy_bounds() instead."
        )

    def __repr__(self) -> str:
        return (
            f"FactorBoundsConstraint("
            f"lower={self.lower.tolist()}, "
            f"upper={self.upper.tolist()}, "
            f"name='{self.name}')"
        )


class ENBPConstraint(Constraint):
    """
    FCA PS21/5 equivalent new business pricing constraint.

    Enforces: renewal_premium_i <= NB_equivalent_premium_i for all renewal policies.

    The NB-equivalent premium is the premium a new business customer with
    identical risk characteristics would be quoted through the same channel.
    In a multiplicative tariff, the NB equivalent excludes renewal-only factors
    (e.g., tenure discounts, NCB).

    If the rate change vector is applied uniformly to both renewal and NB
    factors, this constraint is automatically satisfied. It binds only when
    renewal-specific factor adjustments diverge from NB adjustments.

    The constraint is implemented as a portfolio-level measure: the mean
    excess of renewal over NB-equivalent premium must be <= 0.

    Parameters
    ----------
    channels : list of str, optional
        Channels to apply the constraint to. If None, applies to all channels.
        FCA PS21/5 is channel-specific: the renewal premium must not exceed
        the NB equivalent *through the same channel*.
    name : str
        Constraint identifier.
    tolerance : float
        Small positive tolerance for numerical stability. Default 1e-6.
    """

    def __init__(
        self,
        channels: Optional[list[str]] = None,
        name: str = "enbp",
        tolerance: float = 1e-6,
    ) -> None:
        self.channels = channels
        self.name = name
        self.tolerance = tolerance

    def evaluate(
        self,
        adjustments: np.ndarray,
        data: pl.DataFrame,
        factor_structure,
        demand_model,
    ) -> float:
        """
        Returns the minimum of (NB_equiv - renewal_premium) across relevant policies.

        Positive means all renewal premiums are at or below NB equivalent (constraint satisfied).
        Negative means at least one renewal exceeds its NB equivalent (constraint violated).
        """
        excess = self._compute_excess(adjustments, data, factor_structure)
        if len(excess) == 0:
            return 0.0  # No renewal policies; constraint trivially satisfied.
        return float(-excess.max() + self.tolerance)  # satisfied when max excess <= 0

    def _compute_excess(
        self,
        adjustments: np.ndarray,
        data: pl.DataFrame,
        factor_structure,
    ) -> np.ndarray:
        """
        Per-policy (renewal_premium - NB_equivalent) for renewal policies.

        Positive values indicate a violation.
        """
        renewal_mask = data["renewal_flag"].to_numpy().astype(bool)
        if self.channels is not None:
            channel_mask = data["channel"].is_in(self.channels).to_numpy()
            policy_mask = renewal_mask & channel_mask
        else:
            policy_mask = renewal_mask

        if not policy_mask.any():
            return np.array([])

        current_premiums = data["current_premium"].to_numpy()

        # Renewal premium: apply all factor adjustments
        renewal_adj_product = np.prod(adjustments)  # all factors apply
        renewal_premiums = current_premiums[policy_mask] * renewal_adj_product

        # NB-equivalent premium: exclude renewal-only factor adjustments.
        # For renewal-only factors, revert to 1.0 (no adjustment — NB doesn't get
        # the renewal-specific discount).
        nb_adj = adjustments.copy()
        renewal_factor_indices = [
            i for i, f in enumerate(factor_structure.factor_names)
            if f in factor_structure.renewal_factor_names
        ]
        for idx in renewal_factor_indices:
            nb_adj[idx] = 1.0
        nb_adj_product = np.prod(nb_adj)
        nb_premiums = current_premiums[policy_mask] * nb_adj_product

        return renewal_premiums - nb_premiums

    def to_scipy_dict(self, data, factor_structure, demand_model) -> dict:
        def fun(adj):
            return self.evaluate(adj, data, factor_structure, demand_model)

        return {"type": "ineq", "fun": fun}

    def __repr__(self) -> str:
        return f"ENBPConstraint(channels={self.channels}, name='{self.name}')"


# ---------------------------------------------------------------------------
# Internal helper functions shared across constraints
# ---------------------------------------------------------------------------


def _compute_adjusted_premiums(
    adjustments: np.ndarray,
    data: pl.DataFrame,
    factor_structure,
) -> np.ndarray:
    """
    Compute adjusted premiums for all policies given factor adjustments.

    Adjustment is multiplicative: each policy's premium is scaled by
    the product of all factor adjustment multipliers.
    """
    adj_product = np.prod(adjustments)
    return data["current_premium"].to_numpy() * adj_product


def _compute_renewal_probs(
    adjustments: np.ndarray,
    data: pl.DataFrame,
    factor_structure,
    demand_model,
) -> np.ndarray:
    """
    Compute renewal probabilities at adjusted premiums.

    The price ratio is adjusted_premium / market_premium. If market_premium
    is not present in the data, falls back to technical_premium as proxy.
    """
    adjusted_premiums = _compute_adjusted_premiums(adjustments, data, factor_structure)

    if "market_premium" in data.columns:
        market = data["market_premium"].to_numpy()
    else:
        market = data["technical_premium"].to_numpy()

    price_ratio = adjusted_premiums / np.where(market > 0, market, adjusted_premiums)
    probs = demand_model.predict(price_ratio, policy_features=data)
    return np.clip(probs, 0.0, 1.0)


def _compute_expected_lr(
    adjustments: np.ndarray,
    data: pl.DataFrame,
    factor_structure,
    demand_model,
) -> float:
    """
    Expected portfolio loss ratio at the given factor adjustments.

        E[LR] = sum_i(p_i * c_i) / sum_i(p_i * pi_i)

    where p_i is renewal probability, c_i is technical premium (claims proxy),
    and pi_i is the adjusted premium.
    """
    probs = _compute_renewal_probs(adjustments, data, factor_structure, demand_model)
    adjusted_premiums = _compute_adjusted_premiums(adjustments, data, factor_structure)
    claims = data["technical_premium"].to_numpy()

    expected_claims = np.dot(probs, claims)
    expected_premium = np.dot(probs, adjusted_premiums)

    if expected_premium < 1e-10:
        return 1.0  # degenerate: return max LR to penalise

    return expected_claims / expected_premium


def _compute_volume_ratio(
    adjustments: np.ndarray,
    data: pl.DataFrame,
    factor_structure,
    demand_model,
) -> float:
    """
    Expected volume relative to the current expected count.

        E[vol_ratio] = sum_i(p_i(adj)) / sum_i(p_i(current))

    where p_i(current) is the renewal probability at current pricing,
    stored in the ``renewal_prob`` column of the data DataFrame.

    This is the semantically correct definition for a volume constraint:
    "retain at least X% of the renewals we would expect at current rates".
    It is independent of the absolute renewal rate and focuses on the change
    caused by the rate adjustment.
    """
    probs_new = _compute_renewal_probs(adjustments, data, factor_structure, demand_model)
    if "renewal_prob" in data.columns:
        baseline = float(data["renewal_prob"].to_numpy().sum())
    else:
        baseline = float(len(data))
    if baseline < 1e-10:
        return 1.0
    return float(probs_new.sum() / baseline)
