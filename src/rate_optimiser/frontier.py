"""
Efficient frontier tracer for the loss ratio / volume tradeoff.

This is the insurance analogue of the Markowitz efficient frontier. In portfolio
theory, you trace the frontier by solving the mean-variance problem for a range
of target returns and plotting risk against return. Here, you solve the rate
optimisation problem for a range of loss ratio targets and plot expected volume
against expected LR.

The frontier tells the pricing team the full set of achievable outcomes, not
just a single recommended strategy. The shadow price on the LR constraint at
each frontier point is the marginal cost (in volume) of a one-percentage-point
improvement in LR — a number that belongs in the conversation with commercial
directors.

See section 5.5 of the implementation blueprint for the algorithm design.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd

from rate_optimiser.optimiser import RateChangeOptimiser, OptimiserResult
from rate_optimiser.constraints import LossRatioConstraint


class EfficientFrontier:
    """
    Traces the efficient frontier of achievable (LR, volume) pairs.

    Sweeps over a range of loss ratio targets, solving the constrained
    optimisation at each point. The result is a DataFrame describing the
    Pareto-optimal set of rate strategies — strategies where you cannot
    improve LR without sacrificing volume.

    Parameters
    ----------
    optimiser : RateChangeOptimiser
        A configured optimiser instance. Should have all constraints set
        except the loss ratio bound, which EfficientFrontier controls.
        Any existing LossRatioConstraint named "loss_ratio_ub" will be
        replaced during the sweep.

    Examples
    --------
    >>> frontier = EfficientFrontier(opt)
    >>> df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)
    >>> frontier.plot()
    """

    def __init__(self, optimiser: RateChangeOptimiser) -> None:
        self._opt = optimiser
        self._results: list[dict] = []
        self._df: Optional[pd.DataFrame] = None

    def trace(
        self,
        lr_range: tuple[float, float] = (0.65, 0.80),
        n_points: int = 20,
        lr_constraint_name: str = "loss_ratio_ub",
        warm_start: bool = True,
        tol: float = 1e-8,
        maxiter: int = 500,
    ) -> pd.DataFrame:
        """
        Solve the optimisation for each LR target and collect results.

        The sweep runs from the most relaxed LR target to the tightest,
        using each solution as the starting point for the next (warm start).
        Infeasible points are recorded with ``feasible=False`` but do not
        abort the sweep.

        Parameters
        ----------
        lr_range : tuple[float, float]
            (min_lr, max_lr) range to sweep over. The frontier is traced from
            max to min (relaxed to tight).
        n_points : int
            Number of LR target values to evaluate.
        lr_constraint_name : str
            Name to use for the LR constraint in each solve. Results are
            keyed by this name in the shadow price column.
        warm_start : bool
            Use the previous solution as initial guess for the next solve.
            Usually improves convergence speed.
        tol : float
            Convergence tolerance passed to the solver.
        maxiter : int
            Maximum iterations per solve.

        Returns
        -------
        pd.DataFrame
            Columns: lr_target, expected_lr, expected_volume, feasible,
            objective_value, shadow_lr, shadow_volume, factor_adjustments.
            Each row is one point on the frontier.
        """
        lr_min, lr_max = lr_range
        if lr_min >= lr_max:
            raise ValueError(
                f"lr_range must have min < max, got ({lr_min}, {lr_max})."
            )
        lr_targets = np.linspace(lr_max, lr_min, n_points)

        x0 = None
        rows = []

        for lr_target in lr_targets:
            constraint = LossRatioConstraint(bound=float(lr_target), name=lr_constraint_name)
            self._opt.replace_constraint(constraint)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = self._opt.solve(x0=x0, tol=tol, maxiter=maxiter)

            if warm_start and result.success:
                x0 = np.array(list(result.factor_adjustments.values()))

            rows.append({
                "lr_target": lr_target,
                "expected_lr": result.expected_lr,
                "expected_volume": result.expected_volume,
                "feasible": result.success,
                "objective_value": result.objective_value,
                "shadow_lr": result.shadow_prices.get(lr_constraint_name, 0.0),
                "shadow_volume": result.shadow_prices.get("volume_lb", 0.0),
                "factor_adjustments": result.factor_adjustments.copy(),
                "n_iterations": result.n_iterations,
            })

        self._df = pd.DataFrame(rows)
        self._results = rows
        return self._df

    @property
    def frontier_df(self) -> pd.DataFrame:
        """The frontier DataFrame from the most recent trace() call."""
        if self._df is None:
            raise RuntimeError("Call trace() before accessing frontier_df.")
        return self._df

    def feasible_points(self) -> pd.DataFrame:
        """Subset of the frontier DataFrame with feasible=True."""
        return self.frontier_df[self.frontier_df["feasible"]].copy()

    def shadow_price_summary(self) -> pd.DataFrame:
        """
        Summary of shadow prices across the frontier.

        The shadow price on the LR constraint at each point is the marginal
        cost (in objective units) of tightening the LR target by one unit.
        Economically: how much additional dislocation does it cost to achieve
        one more percentage point of LR improvement?
        """
        df = self.feasible_points()
        return df[["lr_target", "expected_lr", "expected_volume", "shadow_lr", "shadow_volume"]].copy()

    def plot(
        self,
        ax=None,
        show_shadow_prices: bool = True,
        feasible_color: str = "#1f77b4",
        infeasible_color: str = "#d62728",
        annotate_shadow: bool = False,
    ):
        """
        Plot the efficient frontier.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. Creates a new figure if not supplied.
        show_shadow_prices : bool
            If True, annotate points with shadow price on LR constraint.
        feasible_color : str
            Colour for feasible frontier points.
        infeasible_color : str
            Colour for infeasible points.
        annotate_shadow : bool
            If True, annotate each feasible point with its shadow price value.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            )

        from rate_optimiser.plotting import plot_frontier

        return plot_frontier(
            self.frontier_df,
            ax=ax,
            show_shadow_prices=show_shadow_prices,
            feasible_color=feasible_color,
            infeasible_color=infeasible_color,
            annotate_shadow=annotate_shadow,
        )

    def __repr__(self) -> str:
        n_traced = len(self._results) if self._results else 0
        n_feasible = sum(1 for r in self._results if r.get("feasible", False))
        return (
            f"EfficientFrontier("
            f"n_points_traced={n_traced}, "
            f"n_feasible={n_feasible})"
        )
