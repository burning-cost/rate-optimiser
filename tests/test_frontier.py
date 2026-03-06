"""Tests for EfficientFrontier."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rate_optimiser.frontier import EfficientFrontier
from rate_optimiser.constraints import FactorBoundsConstraint, VolumeConstraint
from rate_optimiser.optimiser import RateChangeOptimiser


@pytest.fixture
def frontier_optimiser(policy_data, demand_model, factor_structure):
    """Optimiser configured for frontier tracing."""
    opt = RateChangeOptimiser(data=policy_data, demand=demand_model, factor_structure=factor_structure)
    opt.add_constraint(VolumeConstraint(bound=0.90))
    opt.add_constraint(
        FactorBoundsConstraint(lower=0.80, upper=1.30, n_factors=factor_structure.n_factors)
    )
    return opt


class TestEfficientFrontier:
    def test_trace_returns_dataframe(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.70, 0.80), n_points=5)
        assert isinstance(df, pl.DataFrame)

    def test_trace_has_required_columns(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.70, 0.80), n_points=5)
        required = {"lr_target", "expected_lr", "expected_volume", "feasible", "shadow_lr"}
        assert required <= set(df.columns)

    def test_trace_has_correct_number_of_rows(self, frontier_optimiser):
        n = 6
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.70, 0.80), n_points=n)
        assert len(df) == n

    def test_lr_targets_span_range(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.70, 0.80), n_points=5)
        assert df["lr_target"].min() >= 0.70 - 1e-6
        assert df["lr_target"].max() <= 0.80 + 1e-6

    def test_feasible_points_at_relaxed_bounds(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.72, 0.80), n_points=5)
        feasible = df.filter(pl.col("feasible"))
        assert len(feasible) > 0, "Expected at least some feasible frontier points"

    def test_volume_decreases_as_lr_tightens(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.70, 0.80), n_points=8)
        feasible = df.filter(pl.col("feasible")).sort("lr_target", descending=True)
        if len(feasible) >= 3:
            # As LR target decreases (tighter), volume should tend to decrease
            vols = feasible["expected_volume"].to_numpy()
            # At least the general trend should be downward
            # (not strictly monotone due to solver noise)
            assert vols[0] >= vols[-1] - 0.05, (
                "Volume should not increase significantly as LR target tightens"
            )

    def test_shadow_lr_is_numeric(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.72, 0.80), n_points=5)
        assert df["shadow_lr"].dtype in [pl.Float64, pl.Float32]

    def test_frontier_df_property_raises_before_trace(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        with pytest.raises(RuntimeError, match="trace"):
            _ = ef.frontier_df

    def test_frontier_df_property_after_trace(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        ef.trace(lr_range=(0.72, 0.80), n_points=4)
        df = ef.frontier_df
        assert isinstance(df, pl.DataFrame)

    def test_feasible_points_method(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        ef.trace(lr_range=(0.72, 0.80), n_points=5)
        feasible = ef.feasible_points()
        assert feasible["feasible"].all()

    def test_shadow_price_summary_columns(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        ef.trace(lr_range=(0.72, 0.80), n_points=5)
        summary = ef.shadow_price_summary()
        assert "shadow_lr" in summary.columns
        assert "lr_target" in summary.columns
        assert "expected_volume" in summary.columns

    def test_invalid_lr_range_raises(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        with pytest.raises(ValueError, match="min < max"):
            ef.trace(lr_range=(0.80, 0.70))

    def test_repr_before_trace(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        r = repr(ef)
        assert "EfficientFrontier" in r
        assert "n_points_traced=0" in r

    def test_repr_after_trace(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        ef.trace(lr_range=(0.72, 0.80), n_points=4)
        r = repr(ef)
        assert "n_points_traced=4" in r

    def test_factor_adjustments_stored_per_point(self, frontier_optimiser):
        ef = EfficientFrontier(frontier_optimiser)
        df = ef.trace(lr_range=(0.72, 0.80), n_points=4)
        assert "factor_adjustments" in df.columns
        # Should be a dict in each row
        for adj in df["factor_adjustments"].to_list():
            assert isinstance(adj, dict)
