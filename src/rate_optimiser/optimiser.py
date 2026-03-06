"""
Core rate change optimiser.

RateChangeOptimiser wraps scipy.optimize.minimize (SLSQP) to find factor
adjustments that minimise premium dislocation while satisfying portfolio-level
constraints.

The objective is ||m - 1||² (sum of squared deviations of factor adjustments
from 1.0). This is the "minimum dislocation" criterion: find the smallest rate
change that achieves the required outcome. Alternative objectives (e.g., mean
absolute change, premium-weighted dislocation) are supported via the objective
parameter.

Shadow prices (Lagrange multipliers) are extracted from the SLSQP solution.
These are the key output for pricing teams: the shadow price on the LR
constraint tells you the marginal cost in volume terms of tightening the loss
ratio target by one percentage point.

References
----------
Branda, M. (2013). "Optimization Approaches to Multiplicative Tariff of Rates."
ASTIN Colloquium, Hague.

Guven, S. and McPhail, J. (2013). "Beyond the Cost Model: Understanding Price
Elasticity and Its Applications." CAS Spring Forum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Literal
import warnings

import numpy as np
import polars as pl
from scipy.optimize import minimize, OptimizeResult

from rate_optimiser.data import PolicyData, FactorStructure
from rate_optimiser.demand import DemandModel
from rate_optimiser.constraints import (
    Constraint,
    FactorBoundsConstraint,
    LossRatioConstraint,
    VolumeConstraint,
    ENBPConstraint,
    _compute_expected_lr,
    _compute_volume_ratio,
    _compute_renewal_probs,
)


ObjectiveType = Literal["min_dislocation", "min_weighted_dislocation", "min_abs_dislocation"]


@dataclass
class OptimiserResult:
    """
    Output from a single optimisation run.

    Attributes
    ----------
    factor_adjustments : dict[str, float]
        Optimal multiplicative adjustment for each rating factor. A value of
        1.05 means the factor's relativities have been scaled up by 5%.
    expected_lr : float
        Expected portfolio loss ratio at the optimal adjustments.
    expected_volume : float
        Expected volume ratio (fraction of current count expected to renew or
        convert) at optimal adjustments.
    shadow_prices : dict[str, float]
        Lagrange multipliers for each named constraint. A non-zero value indicates
        a binding constraint; the magnitude is the marginal improvement in the
        objective per unit relaxation of the constraint bound.
    success : bool
        Whether the solver found a feasible solution within convergence tolerance.
    message : str
        Solver status message.
    n_iterations : int
        Number of solver iterations.
    objective_value : float
        Value of the objective function at the solution.
    raw_result : OptimizeResult
        The raw scipy OptimizeResult for full diagnostics.
    """

    factor_adjustments: dict[str, float]
    expected_lr: float
    expected_volume: float
    shadow_prices: dict[str, float]
    success: bool
    message: str
    n_iterations: int
    objective_value: float
    raw_result: OptimizeResult

    def summary(self) -> str:
        """One-paragraph plain-text summary of the optimisation result."""
        adj_str = ", ".join(
            f"{k}: {v:+.3f}" for k, v in self.factor_adjustments.items()
        )
        shadow_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in self.shadow_prices.items()
        )
        status = "converged" if self.success else "did not converge"
        return (
            f"Optimisation {status} in {self.n_iterations} iterations.\n"
            f"Expected LR: {self.expected_lr:.4f}\n"
            f"Expected volume ratio: {self.expected_volume:.4f}\n"
            f"Factor adjustments: {adj_str}\n"
            f"Shadow prices: {shadow_str}\n"
            f"Objective: {self.objective_value:.6f}\n"
            f"Solver message: {self.message}"
        )


class RateChangeOptimiser:
    """
    Constrained rate change optimiser for a multiplicative tariff.

    Finds multiplicative factor adjustments (m_k) that minimise premium
    dislocation (||m - 1||²) subject to user-defined constraints on loss
    ratio, volume, factor movement bounds, and regulatory requirements.

    The problem solved is:

        minimise   sum_k (m_k - 1)²
        subject to portfolio_LR(m) <= LR_bound
                   volume_ratio(m) >= vol_bound
                   m_k in [lower_k, upper_k] for all k
                   ENBP constraints (if added)

    Parameters
    ----------
    data : PolicyData
        Policy-level input data.
    demand : DemandModel
        Demand model for computing renewal probabilities.
    factor_structure : FactorStructure
        Tariff factor structure.
    objective : str
        Objective function type. Options:

        - ``"min_dislocation"``: minimise ||m - 1||² (default)
        - ``"min_weighted_dislocation"``: premium-weighted sum of squared deviations
        - ``"min_abs_dislocation"``: minimise sum |m_k - 1| (non-smooth; use carefully)

    Examples
    --------
    >>> opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
    >>> opt.add_constraint(LossRatioConstraint(bound=0.72))
    >>> opt.add_constraint(VolumeConstraint(bound=0.97))
    >>> opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))
    >>> result = opt.solve()
    >>> print(result.summary())
    """

    def __init__(
        self,
        data: PolicyData,
        demand: DemandModel,
        factor_structure: FactorStructure,
        objective: ObjectiveType = "min_dislocation",
    ) -> None:
        self._data = data
        self._demand = demand
        self._fs = factor_structure
        self._objective = objective
        self._constraints: list[Constraint] = []
        self._bounds_constraint: Optional[FactorBoundsConstraint] = None

        data.validate_demand_outputs()

    @property
    def n_factors(self) -> int:
        return self._fs.n_factors

    def add_constraint(self, constraint: Constraint) -> "RateChangeOptimiser":
        """
        Add a constraint to the optimisation problem.

        Parameters
        ----------
        constraint : Constraint
            An instance of LossRatioConstraint, VolumeConstraint, ENBPConstraint,
            or FactorBoundsConstraint.

        Returns
        -------
        RateChangeOptimiser
            Self, for method chaining.
        """
        if isinstance(constraint, FactorBoundsConstraint):
            if constraint.n_factors != self.n_factors:
                raise ValueError(
                    f"FactorBoundsConstraint has n_factors={constraint.n_factors} "
                    f"but optimiser has {self.n_factors} factors."
                )
            self._bounds_constraint = constraint
        else:
            self._constraints.append(constraint)
        return self

    def remove_constraint(self, name: str) -> "RateChangeOptimiser":
        """
        Remove a named constraint.

        Parameters
        ----------
        name : str
            The ``name`` attribute of the constraint to remove.
        """
        before = len(self._constraints)
        self._constraints = [c for c in self._constraints if c.name != name]
        if len(self._constraints) == before:
            raise KeyError(f"No constraint named '{name}' found.")
        return self

    def replace_constraint(self, constraint: Constraint) -> "RateChangeOptimiser":
        """
        Replace an existing constraint with the same name.

        Useful when sweeping over constraint bounds (e.g., in EfficientFrontier).
        """
        if isinstance(constraint, FactorBoundsConstraint):
            self._bounds_constraint = constraint
            return self
        self._constraints = [c for c in self._constraints if c.name != constraint.name]
        self._constraints.append(constraint)
        return self

    def solve(
        self,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
    ) -> OptimiserResult:
        """
        Run the optimiser and return results.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial guess for factor adjustments. Defaults to all-ones (no change).
        tol : float
            Convergence tolerance for SLSQP. Default 1e-8.
        maxiter : int
            Maximum solver iterations. Default 500.

        Returns
        -------
        OptimiserResult
        """
        if x0 is None:
            x0 = self._fs.initial_adjustments()

        df = self._data.df

        scipy_constraints = [
            c.to_scipy_dict(df, self._fs, self._demand)
            for c in self._constraints
        ]

        bounds = None
        if self._bounds_constraint is not None:
            bounds = self._bounds_constraint.to_scipy_bounds()

        obj_fn = self._build_objective(df)

        raw = minimize(
            obj_fn,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": maxiter, "ftol": tol, "disp": False},
        )

        shadow_prices = self._extract_shadow_prices(raw)
        adj_dict = dict(zip(self._fs.factor_names, raw.x.tolist()))

        expected_lr = _compute_expected_lr(raw.x, df, self._fs, self._demand)
        expected_vol = _compute_volume_ratio(raw.x, df, self._fs, self._demand)

        if not raw.success and raw.status not in (0, 1, 2):
            warnings.warn(
                f"Optimiser did not converge: {raw.message}. "
                "Check that constraints are feasible and the demand model is well-behaved.",
                RuntimeWarning,
                stacklevel=2,
            )

        return OptimiserResult(
            factor_adjustments=adj_dict,
            expected_lr=expected_lr,
            expected_volume=expected_vol,
            shadow_prices=shadow_prices,
            success=raw.success,
            message=raw.message,
            n_iterations=raw.nit,
            objective_value=float(raw.fun),
            raw_result=raw,
        )

    def _build_objective(self, df: pl.DataFrame) -> Callable:
        """Build the objective function closure."""
        objective_type = self._objective

        if objective_type == "min_dislocation":
            def obj(adj: np.ndarray) -> float:
                return float(np.sum((adj - 1.0) ** 2))

        elif objective_type == "min_weighted_dislocation":
            total_premium = df["current_premium"].to_numpy().sum()
            factor_premiums = np.array([
                df["current_premium"].to_numpy().sum() / self.n_factors
                for _ in range(self.n_factors)
            ])

            def obj(adj: np.ndarray) -> float:
                return float(np.sum(factor_premiums * (adj - 1.0) ** 2)) / total_premium

        elif objective_type == "min_abs_dislocation":
            def obj(adj: np.ndarray) -> float:
                return float(np.sum(np.abs(adj - 1.0)))

        else:
            raise ValueError(f"Unknown objective: {objective_type!r}")

        return obj

    def _extract_shadow_prices(self, raw: OptimizeResult) -> dict[str, float]:
        """
        Extract Lagrange multipliers from the SLSQP result.

        SLSQP stores multipliers in raw.v (a list). The ordering matches the
        constraint list passed to minimize, plus the bound constraints.
        """
        shadow = {}
        v = getattr(raw, "v", None)
        if v is None or len(v) == 0:
            # Fall back: return zeros
            for c in self._constraints:
                shadow[c.name] = 0.0
            return shadow

        # v is [multipliers_for_ineq..., multipliers_for_bounds...]
        for i, c in enumerate(self._constraints):
            if i < len(v):
                shadow[c.name] = float(v[i])
            else:
                shadow[c.name] = 0.0

        return shadow

    def feasibility_report(self, adjustments: Optional[np.ndarray] = None) -> pl.DataFrame:
        """
        Evaluate all constraints at the given (or current) adjustments.

        Useful for diagnosing infeasible problems before running the solver.

        Parameters
        ----------
        adjustments : np.ndarray, optional
            Factor adjustments to evaluate. Defaults to all-ones (no change baseline).

        Returns
        -------
        pl.DataFrame
            One row per constraint with columns: name, value, satisfied.
        """
        if adjustments is None:
            adjustments = self._fs.initial_adjustments()

        df = self._data.df
        rows = []
        for c in self._constraints:
            val = c.evaluate(adjustments, df, self._fs, self._demand)
            rows.append({"constraint": c.name, "value": val, "satisfied": val >= 0})

        if self._bounds_constraint is not None:
            val = self._bounds_constraint.evaluate(adjustments)
            rows.append({
                "constraint": self._bounds_constraint.name,
                "value": val,
                "satisfied": val >= 0,
            })

        return pl.DataFrame(rows)

    def __repr__(self) -> str:
        constraint_names = [c.name for c in self._constraints]
        if self._bounds_constraint:
            constraint_names.append(self._bounds_constraint.name)
        return (
            f"RateChangeOptimiser("
            f"n_factors={self.n_factors}, "
            f"n_policies={self._data.n_policies}, "
            f"constraints={constraint_names}, "
            f"objective='{self._objective}')"
        )
