"""Tests for PolicyData and FactorStructure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rate_optimiser.data import PolicyData, FactorStructure, REQUIRED_COLUMNS


class TestPolicyData:
    def test_loads_valid_dataframe(self, raw_policy_df):
        data = PolicyData(raw_policy_df)
        assert data.n_policies == len(raw_policy_df)

    def test_missing_required_column_raises(self, raw_policy_df):
        df = raw_policy_df.drop(columns=["technical_premium"])
        with pytest.raises(ValueError, match="missing required columns"):
            PolicyData(df)

    def test_empty_dataframe_raises(self, raw_policy_df):
        df = raw_policy_df.iloc[:0]
        with pytest.raises(ValueError, match="empty"):
            PolicyData(df)

    def test_renewal_flag_coerced_to_bool(self, raw_policy_df):
        df = raw_policy_df.copy()
        df["renewal_flag"] = df["renewal_flag"].astype(int)
        data = PolicyData(df)
        assert data.df["renewal_flag"].dtype == bool

    def test_n_renewals_correct(self, raw_policy_df):
        data = PolicyData(raw_policy_df)
        expected = int(raw_policy_df["renewal_flag"].sum())
        assert data.n_renewals == expected

    def test_channels_returns_sorted_list(self, policy_data):
        channels = policy_data.channels
        assert isinstance(channels, list)
        assert channels == sorted(channels)
        assert set(channels) <= {"PCW", "direct"}

    def test_renewal_subset_correct(self, policy_data):
        r = policy_data.renewal
        assert r["renewal_flag"].all()
        assert len(r) == policy_data.n_renewals

    def test_new_business_subset_correct(self, policy_data):
        nb = policy_data.new_business
        assert (~nb["renewal_flag"]).all()
        assert len(nb) == policy_data.n_policies - policy_data.n_renewals

    def test_current_loss_ratio_in_plausible_range(self, policy_data):
        lr = policy_data.current_loss_ratio()
        # Constructed to be ~0.75
        assert 0.50 < lr < 0.95, f"Unexpected LR: {lr}"

    def test_validate_demand_outputs_raises_if_missing(self, raw_policy_df):
        df = raw_policy_df.drop(columns=["renewal_prob"])
        data = PolicyData(df)
        with pytest.raises(ValueError, match="renewal_prob"):
            data.validate_demand_outputs()

    def test_validate_demand_outputs_raises_if_out_of_range(self, raw_policy_df):
        df = raw_policy_df.copy()
        df.loc[0, "renewal_prob"] = 1.5
        data = PolicyData(df)
        with pytest.raises(ValueError, match="outside"):
            data.validate_demand_outputs()

    def test_validate_demand_outputs_passes_valid_data(self, policy_data):
        policy_data.validate_demand_outputs()  # should not raise

    def test_repr_string(self, policy_data):
        r = repr(policy_data)
        assert "PolicyData" in r
        assert "n_policies" in r

    def test_from_csv_roundtrip(self, raw_policy_df, tmp_path):
        path = tmp_path / "policies.csv"
        raw_policy_df.to_csv(path, index=False)
        loaded = PolicyData.from_csv(path)
        assert loaded.n_policies == len(raw_policy_df)

    def test_dataframe_is_copied(self, raw_policy_df):
        data = PolicyData(raw_policy_df)
        data.df.loc[data.df.index[0], "current_premium"] = 99999
        assert raw_policy_df.loc[raw_policy_df.index[0], "current_premium"] != 99999


class TestFactorStructure:
    def test_creates_valid_structure(self, factor_structure):
        assert factor_structure.n_factors == 5

    def test_missing_factor_column_raises(self, raw_policy_df):
        with pytest.raises(ValueError, match="missing columns"):
            FactorStructure(
                factor_names=["factor_age_band", "nonexistent_factor"],
                factor_values=raw_policy_df[["factor_age_band"]],
            )

    def test_nonpositive_factor_values_raise(self, raw_policy_df, factor_names):
        df = raw_policy_df[factor_names].copy()
        df.loc[df.index[0], "factor_age_band"] = -0.5
        with pytest.raises(ValueError, match="strictly positive"):
            FactorStructure(factor_names=factor_names, factor_values=df)

    def test_invalid_renewal_factor_name_raises(self, raw_policy_df, factor_names):
        df = raw_policy_df[factor_names]
        with pytest.raises(ValueError, match="renewal_factor_name"):
            FactorStructure(
                factor_names=factor_names,
                factor_values=df,
                renewal_factor_names=["nonexistent"],
            )

    def test_initial_adjustments_are_ones(self, factor_structure):
        adj = factor_structure.initial_adjustments()
        np.testing.assert_array_equal(adj, np.ones(factor_structure.n_factors))

    def test_renewal_only_mask(self, factor_structure):
        mask = factor_structure.renewal_only_mask()
        # tenure_discount is the renewal-only factor; it should be False in mask
        # (mask is True for NON-renewal factors)
        tenure_idx = factor_structure.factor_names.index("factor_tenure_discount")
        assert mask[tenure_idx] == False
        assert mask.sum() == factor_structure.n_factors - 1

    def test_premium_ratio_at_identity_is_one(self, factor_structure):
        adj = np.ones(factor_structure.n_factors)
        ratio = factor_structure.premium_ratio(adj)
        assert abs(ratio.mean() - 1.0) < 1e-10

    def test_premium_ratio_uniform_uplift(self, factor_structure):
        adj = np.ones(factor_structure.n_factors) * 1.05
        ratio = factor_structure.premium_ratio(adj)
        # All factors at 1.05: product = 1.05^5
        expected = 1.05 ** factor_structure.n_factors
        assert abs(ratio.mean() - expected) < 1e-9

    def test_non_renewal_factor_names(self, factor_structure):
        non_renewal = factor_structure.non_renewal_factor_names
        assert "factor_tenure_discount" not in non_renewal
        assert len(non_renewal) == factor_structure.n_factors - 1

    def test_repr_string(self, factor_structure):
        r = repr(factor_structure)
        assert "FactorStructure" in r
        assert "n_factors" in r
