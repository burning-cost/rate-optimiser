"""
Plotting utilities for rate optimisation outputs.

All functions return matplotlib Axes objects, so callers can further customise.
No inline plt.show() calls — that is the caller's responsibility.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def plot_frontier(
    frontier_df: pd.DataFrame,
    ax=None,
    show_shadow_prices: bool = True,
    feasible_color: str = "#1f77b4",
    infeasible_color: str = "#d62728",
    annotate_shadow: bool = False,
    xlabel: str = "Expected loss ratio",
    ylabel: str = "Expected volume ratio",
    title: str = "Efficient Frontier: LR vs. Volume",
) -> "matplotlib.axes.Axes":
    """
    Plot the efficient frontier from an EfficientFrontier.trace() result.

    Feasible points are plotted in blue; infeasible points (where the solver
    did not converge or constraints could not be satisfied) in red.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Output of EfficientFrontier.trace(). Must have columns: expected_lr,
        expected_volume, feasible.
    ax : matplotlib Axes, optional
        Axes to plot on. Creates a new figure if not supplied.
    show_shadow_prices : bool
        If True, plot a secondary colour-encoded scatter showing shadow_lr
        magnitude across frontier points.
    feasible_color : str
        Hex colour for feasible points.
    infeasible_color : str
        Hex colour for infeasible points.
    annotate_shadow : bool
        If True, annotate each feasible point with its shadow_lr value.
    xlabel, ylabel, title : str
        Axis labels and title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    feasible = frontier_df[frontier_df["feasible"]]
    infeasible = frontier_df[~frontier_df["feasible"]]

    if len(feasible) > 0:
        ax.plot(
            feasible["expected_lr"],
            feasible["expected_volume"],
            "-o",
            color=feasible_color,
            linewidth=2,
            markersize=6,
            label="Feasible",
            zorder=3,
        )

    if len(infeasible) > 0:
        ax.scatter(
            infeasible["expected_lr"],
            infeasible["expected_volume"],
            marker="x",
            color=infeasible_color,
            s=50,
            linewidths=1.5,
            label="Infeasible",
            zorder=2,
        )

    if annotate_shadow and "shadow_lr" in feasible.columns and len(feasible) > 0:
        for _, row in feasible.iterrows():
            ax.annotate(
                f"{row['shadow_lr']:.3f}",
                xy=(row["expected_lr"], row["expected_volume"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#555555",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Flip x-axis: lower LR is better (tighter target), shown on right
    if len(feasible) > 0:
        x_min, x_max = feasible["expected_lr"].min(), feasible["expected_lr"].max()
        margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.01
        ax.set_xlim(x_max + margin, x_min - margin)

    return ax


def plot_factor_adjustments(
    factor_adjustments: dict[str, float],
    ax=None,
    color: str = "#1f77b4",
    title: str = "Factor Adjustments",
) -> "matplotlib.axes.Axes":
    """
    Horizontal bar chart of factor adjustments relative to 1.0 (no change).

    Parameters
    ----------
    factor_adjustments : dict[str, float]
        Factor name to adjustment multiplier. Values above 1.0 are rate
        increases; below 1.0 are rate reductions.
    ax : matplotlib Axes, optional
    color : str
        Bar colour.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, len(factor_adjustments) * 0.5)))

    names = list(factor_adjustments.keys())
    values = [v - 1.0 for v in factor_adjustments.values()]
    colours = ["#d62728" if v > 0 else "#1f77b4" for v in values]

    bars = ax.barh(names, values, color=colours, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars, values):
        x = bar.get_width()
        ax.text(
            x + (0.001 if x >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1%}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )

    ax.set_xlabel("Adjustment relative to current (0 = no change)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    return ax


def plot_shadow_prices(
    frontier_df: pd.DataFrame,
    ax=None,
    title: str = "Shadow Price on LR Constraint",
) -> "matplotlib.axes.Axes":
    """
    Plot how the shadow price on the LR constraint varies across the frontier.

    The shadow price tells you: at this LR target, how much dislocation (in
    objective units) does one additional percentage point of LR improvement
    cost? A steeply rising shadow price signals a hard part of the frontier
    where further improvement is increasingly costly.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Output of EfficientFrontier.trace().
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    feasible = frontier_df[frontier_df["feasible"]].copy()

    if "shadow_lr" not in feasible.columns or len(feasible) == 0:
        ax.text(0.5, 0.5, "No shadow price data available.", transform=ax.transAxes, ha="center")
        return ax

    ax.plot(
        feasible["lr_target"],
        feasible["shadow_lr"].abs(),
        "-o",
        color="#ff7f0e",
        linewidth=2,
        markersize=5,
    )
    ax.set_xlabel("LR target")
    ax.set_ylabel("|Shadow price| on LR constraint")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax
