"""Tests for the stochastic (chance-constrained) rate optimisation module."""

from __future__ import annotations

import numpy as np
import pytest

from rate_optimiser.stochastic import (
    ClaimsVarianceModel,
    ChanceConstrainedLRConstraint,
    StochasticRateOptimiser,
)
from rate_optimiser.constraints import LossRatioConstraint, FactorBoundsConstraint
from rate_optimiser.optimiser import OptimiserResult


@pytest.fixture
def variance_model(raw_policy_df):
    mean_claims = raw_policy_df["technical_premium"].values
    return ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.2, power=1.5)


@pytest.fixture
def stochastic_optimiser(policy_data, demand_model, factor_structure, variance_model):
    opt = StochasticRateOptimiser(
        data=policy_data,
        demand=demand_model,
        factor_structure=factor_structure,
        variance_model=variance_model,
        lr_bound=0.73,
        alpha=0.90,
    )
    opt.add_constraint(
        FactorBoundsConstraint(lower=0.85, upper=1.20, n_factors=factor_structure.n_factors)
    )
    return opt


class TestClaimsVarianceModel:
    def test_from_tweedie_shapes(self, raw_policy_df):
        mean_claims = raw_policy_df["technical_premium"].values
        model = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.5)
        assert model.mean_claims.shape == mean_claims.shape
        assert model.variance_claims.shape == mean_claims.shape

    def test_from_tweedie_variance_positive(self, raw_policy_df):
        mean_claims = raw_policy_df["technical_premium"].values
        model = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.5)
        assert (model.variance_claims > 0).all()

    def test_from_tweedie_higher_dispersion_gives_higher_variance(self, raw_policy_df):
        mean_claims = raw_policy_df["technical_premium"].values
        m1 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.5)
        m2 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=2.0, power=1.5)
        assert (m2.variance_claims > m1.variance_claims).all()

    def test_from_overdispersed_poisson(self, raw_policy_df):
        n = len(raw_policy_df)
        counts = np.ones(n) * 0.1
        severity = raw_policy_df["technical_premium"].values / 0.1
        sev_var = severity ** 2 * 0.5
        model = ClaimsVarianceModel.from_overdispersed_poisson(
            expected_counts=counts,
            mean_severity=severity,
            severity_variance=sev_var,
            overdispersion=1.5,
        )
        assert model.mean_claims.shape == (n,)
        assert (model.variance_claims > 0).all()

    def test_tweedie_power_affects_variance_shape(self, raw_policy_df):
        mean_claims = raw_policy_df["technical_premium"].values
        # Higher power: variance scales more steeply with mean
        m1 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=1.0)
        m2 = ClaimsVarianceModel.from_tweedie(mean_claims, dispersion=1.0, power=2.0)
        # For means > 1, higher power gives higher variance
        high_mean_mask = mean_claims > 1
        if high_mean_mask.any():
            assert (m2.variance_claims[high_mean_mask] > m1.variance_claims[high_mean_mask]).all()


class TestChanceConstrainedLRConstraint:
    def test_more_conservative_than_deterministic(
        self, raw_policy_df, policy_data, factor_structure, demand_model, variance_model
    ):
        """Chance constraint at alpha=0.95 should be tighter than deterministic LR constraint."""
        adj = np.ones(factor_structure.n_factors) * 1.05  # slight rate increase

        det = LossRatioConstraint(bound=0.73)
        chance = ChanceConstrainedLRConstraint(
            bound=0.73, alpha=0.95, variance_model=variance_model
        )

        det_val = det.evaluate(adj, policy_data.df, factor_structure, demand_model)
        chance_val = chance.evaluate(adj, policy_data.df, factor_structure, demand_model)

        # Chance constraint should have smaller slack (more conservative)
        assert chance_val <= det_val, (
            f"Chance constraint (val={chance_val:.4f}) should be <= "
            f"deterministic (val={det_val:.4f})"
        )

    def test_invalid_alpha_raises(self, variance_model):
        with pytest.raises(ValueError, match="alpha"):
            ChanceConstrainedLRConstraint(bound=0.73, alpha=0.3, variance_model=variance_model)

    def test_invalid_bound_raises(self, variance_model):
        with pytest.raises(ValueError, match="implausible"):
            ChanceConstrainedLRConstraint(bound=5.0, alpha=0.95, variance_model=variance_model)

    def test_higher_alpha_is_more_conservative(
        self, policy_data, factor_structure, demand_model, variance_model
    ):
        adj = np.ones(factor_structure.n_factors) * 1.05

        c90 = ChanceConstrainedLRConstraint(bound=0.73, alpha=0.90, variance_model=variance_model)
        c99 = ChanceConstrainedLRConstraint(bound=0.73, alpha=0.99, variance_model=variance_model)

        val90 = c90.evaluate(adj, policy_data.df, factor_structure, demand_model)
        val99 = c99.evaluate(adj, policy_data.df, factor_structure, demand_model)

        assert val99 <= val90, "alpha=0.99 should be more conservative than alpha=0.90"

    def test_to_scipy_dict_format(self, policy_data, factor_structure, demand_model, variance_model):
        c = ChanceConstrainedLRConstraint(bound=0.73, alpha=0.90, variance_model=variance_model)
        d = c.to_scipy_dict(policy_data.df, factor_structure, demand_model)
        assert d["type"] == "ineq"
        assert callable(d["fun"])

    def test_repr(self, variance_model):
        c = ChanceConstrainedLRConstraint(bound=0.73, alpha=0.95, variance_model=variance_model)
        r = repr(c)
        assert "ChanceConstrainedLRConstraint" in r
        assert "0.73" in r
        assert "0.95" in r

    def test_z_alpha_positive_for_alpha_above_half(self, variance_model):
        c = ChanceConstrainedLRConstraint(bound=0.73, alpha=0.95, variance_model=variance_model)
        assert c.z_alpha > 0


class TestStochasticRateOptimiser:
    def test_solve_returns_result(self, stochastic_optimiser):
        result = stochastic_optimiser.solve()
        assert isinstance(result, OptimiserResult)

    def test_solve_finds_feasible_solution(self, stochastic_optimiser):
        result = stochastic_optimiser.solve()
        assert result.success, f"Stochastic optimiser failed: {result.message}"

    def test_stochastic_requires_higher_rate_than_deterministic(
        self, policy_data, demand_model, factor_structure, variance_model
    ):
        """With uncertainty, the solver should require a higher rate to achieve the same target."""
        def solve_det(lr_bound):
            from rate_optimiser.optimiser import RateChangeOptimiser
            opt = RateChangeOptimiser(data=policy_data, demand=demand_model, factor_structure=factor_structure)
            opt.add_constraint(LossRatioConstraint(bound=lr_bound))
            opt.add_constraint(
                FactorBoundsConstraint(lower=0.5, upper=2.0, n_factors=factor_structure.n_factors)
            )
            return opt.solve()

        def solve_stoc(lr_bound, alpha):
            opt = StochasticRateOptimiser(
                data=policy_data,
                demand=demand_model,
                factor_structure=factor_structure,
                variance_model=variance_model,
                lr_bound=lr_bound,
                alpha=alpha,
            )
            opt.add_constraint(
                FactorBoundsConstraint(lower=0.5, upper=2.0, n_factors=factor_structure.n_factors)
            )
            return opt.solve()

        r_det = solve_det(0.73)
        r_stoc = solve_stoc(0.73, 0.95)

        if r_det.success and r_stoc.success:
            det_total_adj = sum(r_det.factor_adjustments.values())
            stoc_total_adj = sum(r_stoc.factor_adjustments.values())
            assert stoc_total_adj >= det_total_adj - 0.1, (
                "Stochastic optimiser should require at least as much rate as deterministic"
            )

    def test_chance_constraint_in_shadow_prices(self, stochastic_optimiser):
        result = stochastic_optimiser.solve()
        assert "chance_lr" in result.shadow_prices

    def test_repr(self, stochastic_optimiser):
        r = repr(stochastic_optimiser)
        assert "StochasticRateOptimiser" in r

    def test_variance_model_property(self, stochastic_optimiser, variance_model):
        assert stochastic_optimiser.variance_model is variance_model
