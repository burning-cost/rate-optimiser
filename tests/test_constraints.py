"""Tests for constraint classes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rate_optimiser.constraints import (
    LossRatioConstraint,
    VolumeConstraint,
    FactorBoundsConstraint,
    ENBPConstraint,
    _compute_expected_lr,
    _compute_volume_ratio,
    _compute_renewal_probs,
)


class TestLossRatioConstraint:
    def test_satisfied_at_identity_when_lr_is_low(self, policy_data, factor_structure, demand_model):
        # The synthetic data has LR ~0.75; constraint at 0.80 should be satisfied at identity
        adj = factor_structure.initial_adjustments()
        c = LossRatioConstraint(bound=0.80)
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        assert val >= 0, f"Expected constraint satisfied at 0.80 bound, got value={val:.4f}"

    def test_violated_at_identity_when_bound_is_tight(self, policy_data, factor_structure, demand_model):
        # With LR ~0.75, a bound of 0.60 should be violated at identity
        adj = factor_structure.initial_adjustments()
        c = LossRatioConstraint(bound=0.60)
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        assert val < 0, f"Expected constraint violated at 0.60 bound, got value={val:.4f}"

    def test_to_scipy_dict_format(self, policy_data, factor_structure, demand_model):
        c = LossRatioConstraint(bound=0.72)
        d = c.to_scipy_dict(policy_data.df, factor_structure, demand_model)
        assert d["type"] == "ineq"
        assert callable(d["fun"])
        adj = factor_structure.initial_adjustments()
        val = d["fun"](adj)
        assert isinstance(val, float)

    def test_implausible_bound_raises(self):
        with pytest.raises(ValueError, match="implausible"):
            LossRatioConstraint(bound=3.0)

    def test_repr(self):
        c = LossRatioConstraint(bound=0.72, name="my_lr")
        assert "LossRatioConstraint" in repr(c)
        assert "0.72" in repr(c)

    def test_lr_increases_with_higher_adjustments(self, policy_data, factor_structure, demand_model):
        # Increasing all factor adjustments raises premiums → lowers LR
        adj_low = np.ones(factor_structure.n_factors) * 0.90
        adj_high = np.ones(factor_structure.n_factors) * 1.15
        lr_low = _compute_expected_lr(adj_low, policy_data.df, factor_structure, demand_model)
        lr_high = _compute_expected_lr(adj_high, policy_data.df, factor_structure, demand_model)
        assert lr_high < lr_low, "Higher premiums should give lower LR"


class TestVolumeConstraint:
    def test_satisfied_at_identity(self, policy_data, factor_structure, demand_model):
        # vol_ratio = 1.0 at identity (demand model calibrated to stored renewal_prob)
        adj = factor_structure.initial_adjustments()
        c = VolumeConstraint(bound=0.95)
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        assert val >= 0, f"Expected vol_ratio >= 0.95 at identity, got {val:.4f}"

    def test_violated_when_prices_raised_significantly(self, policy_data, factor_structure, demand_model):
        # Raising prices sharply should drop vol_ratio below a reasonable bound
        adj = np.ones(factor_structure.n_factors) * 1.15  # +15% per factor
        c = VolumeConstraint(bound=1.0)
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        assert val < 0, f"Raising prices sharply should violate 100% volume retention, got {val:.4f}"

    def test_volume_ratio_is_one_at_identity(self, policy_data, factor_structure, demand_model):
        adj = factor_structure.initial_adjustments()
        vol = _compute_volume_ratio(adj, policy_data.df, factor_structure, demand_model)
        assert abs(vol - 1.0) < 1e-6, f"vol_ratio should be 1.0 at identity, got {vol:.6f}"

    def test_volume_decreases_with_higher_prices(self, policy_data, factor_structure, demand_model):
        # Identity gives vol_ratio=1.0; raising prices should reduce it
        adj_base = np.ones(factor_structure.n_factors)
        adj_up = np.ones(factor_structure.n_factors) * 1.10
        vol_base = _compute_volume_ratio(adj_base, policy_data.df, factor_structure, demand_model)
        vol_up = _compute_volume_ratio(adj_up, policy_data.df, factor_structure, demand_model)
        assert vol_up < vol_base, "Volume should fall when prices increase"

    def test_to_scipy_dict_format(self, policy_data, factor_structure, demand_model):
        c = VolumeConstraint(bound=0.95)
        d = c.to_scipy_dict(policy_data.df, factor_structure, demand_model)
        assert d["type"] == "ineq"
        assert callable(d["fun"])

    def test_implausible_bound_raises(self):
        with pytest.raises(ValueError, match="implausible"):
            VolumeConstraint(bound=1.5)

    def test_repr(self):
        c = VolumeConstraint(bound=0.97)
        assert "VolumeConstraint" in repr(c)


class TestFactorBoundsConstraint:
    def test_feasible_at_identity(self, factor_structure):
        n = factor_structure.n_factors
        c = FactorBoundsConstraint(lower=0.85, upper=1.20, n_factors=n)
        adj = np.ones(n)
        val = c.evaluate(adj)
        assert val > 0

    def test_infeasible_when_outside_bounds(self, factor_structure):
        n = factor_structure.n_factors
        c = FactorBoundsConstraint(lower=0.85, upper=1.20, n_factors=n)
        adj = np.ones(n) * 1.50  # above upper bound
        val = c.evaluate(adj)
        assert val < 0

    def test_broadcast_scalar_bounds(self, factor_structure):
        n = factor_structure.n_factors
        c = FactorBoundsConstraint(lower=0.9, upper=1.1, n_factors=n)
        assert len(c.lower) == n
        assert len(c.upper) == n

    def test_scipy_bounds_format(self, factor_structure):
        n = factor_structure.n_factors
        c = FactorBoundsConstraint(lower=0.9, upper=1.1, n_factors=n)
        bounds = c.to_scipy_bounds()
        assert len(bounds) == n
        for lo, hi in bounds:
            assert lo == pytest.approx(0.9)
            assert hi == pytest.approx(1.1)

    def test_nonpositive_lower_raises(self, factor_structure):
        with pytest.raises(ValueError, match="strictly positive"):
            FactorBoundsConstraint(lower=-0.5, upper=1.1, n_factors=factor_structure.n_factors)

    def test_inverted_bounds_raise(self, factor_structure):
        with pytest.raises(ValueError, match="lower bounds must be <= upper"):
            FactorBoundsConstraint(lower=1.2, upper=0.8, n_factors=factor_structure.n_factors)

    def test_wrong_n_factors_raises_in_optimiser(self, basic_optimiser):
        wrong = FactorBoundsConstraint(lower=0.9, upper=1.1, n_factors=99)
        with pytest.raises(ValueError, match="n_factors"):
            basic_optimiser.add_constraint(wrong)

    def test_to_scipy_dict_raises(self, factor_structure):
        n = factor_structure.n_factors
        c = FactorBoundsConstraint(lower=0.9, upper=1.1, n_factors=n)
        with pytest.raises(NotImplementedError):
            c.to_scipy_dict(None, None, None)

    def test_repr(self, factor_structure):
        n = factor_structure.n_factors
        c = FactorBoundsConstraint(lower=0.9, upper=1.1, n_factors=n)
        assert "FactorBoundsConstraint" in repr(c)


class TestENBPConstraint:
    def test_satisfied_at_identity_no_renewal_factor_adjustment(
        self, policy_data, factor_structure, demand_model
    ):
        adj = factor_structure.initial_adjustments()
        c = ENBPConstraint()
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        # At identity, tenure_discount factor = 1.0 for all; renewal = NB equivalent
        assert val >= 0, f"ENBP should be satisfied at identity, got {val}"

    def test_violated_when_renewal_factor_increases(
        self, policy_data, factor_structure, demand_model
    ):
        # Increase the renewal-only factor (tenure_discount index = 4)
        adj = factor_structure.initial_adjustments()
        tenure_idx = factor_structure.factor_names.index("factor_tenure_discount")
        adj[tenure_idx] = 1.30  # renewal price 30% above NB equivalent
        c = ENBPConstraint()
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        assert val < 0, f"ENBP should be violated when renewal factor > 1.0, got {val}"

    def test_satisfied_when_renewal_factor_decreases(
        self, policy_data, factor_structure, demand_model
    ):
        adj = factor_structure.initial_adjustments()
        tenure_idx = factor_structure.factor_names.index("factor_tenure_discount")
        adj[tenure_idx] = 0.85  # renewal price 15% below NB equivalent (allowed)
        c = ENBPConstraint()
        val = c.evaluate(adj, policy_data.df, factor_structure, demand_model)
        assert val >= 0, f"ENBP should be satisfied when renewal factor < 1.0, got {val}"

    def test_channel_filter(self, policy_data, factor_structure, demand_model):
        adj = factor_structure.initial_adjustments()
        tenure_idx = factor_structure.factor_names.index("factor_tenure_discount")
        adj[tenure_idx] = 1.30
        # PCW-only constraint
        c_pcw = ENBPConstraint(channels=["PCW"])
        c_direct = ENBPConstraint(channels=["direct"])
        val_pcw = c_pcw.evaluate(adj, policy_data.df, factor_structure, demand_model)
        val_direct = c_direct.evaluate(adj, policy_data.df, factor_structure, demand_model)
        # Both PCW and direct have renewals, so both should detect violation
        assert val_pcw < 0 or val_direct < 0

    def test_no_renewal_policies_returns_zero(self, raw_policy_df, factor_structure, demand_model):
        df = raw_policy_df.with_columns(pl.lit(False).alias("renewal_flag"))
        from rate_optimiser.data import PolicyData
        data = PolicyData(df)
        adj = factor_structure.initial_adjustments()
        c = ENBPConstraint()
        val = c.evaluate(adj, data.df, factor_structure, demand_model)
        assert val == pytest.approx(0.0, abs=0.01)

    def test_repr(self):
        c = ENBPConstraint(channels=["PCW"])
        assert "ENBPConstraint" in repr(c)
        assert "PCW" in repr(c)

    def test_to_scipy_dict_format(self, policy_data, factor_structure, demand_model):
        c = ENBPConstraint()
        d = c.to_scipy_dict(policy_data.df, factor_structure, demand_model)
        assert d["type"] == "ineq"
        assert callable(d["fun"])
