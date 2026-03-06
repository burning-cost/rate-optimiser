"""
Data layer: PolicyData and FactorStructure.

PolicyData holds the policy-level inputs the optimiser operates on. FactorStructure
describes the multiplicative tariff — which rating factors exist and which factor
level applies to each policy. The optimiser's decision variables are adjustments to
factor relativities within this structure.

Design note: this library consumes model outputs, not models. The columns
``technical_premium`` and ``renewal_prob`` are expected to be pre-computed by
the caller's GLM/GBM pipeline. We do not refit models here.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


# Minimum required columns in the policy DataFrame.
REQUIRED_COLUMNS = {
    "policy_id",
    "channel",
    "renewal_flag",
    "technical_premium",
    "current_premium",
}


@dataclass
class PolicyData:
    """
    Policy-level data container for the rate optimiser.

    Each row represents one in-force or renewal-offer policy. The optimiser
    needs technical prices (from your GLM/GBM), current premiums, renewal
    flags, and a demand model output (renewal/conversion probability).

    Parameters
    ----------
    df : pd.DataFrame
        Policy-level DataFrame. Required columns: ``policy_id``, ``channel``,
        ``renewal_flag`` (bool), ``technical_premium`` (float),
        ``current_premium`` (float). Optional but used when present:
        ``market_premium``, ``renewal_prob``, ``claims_count``,
        ``claims_variance``.
    """

    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                f"PolicyData is missing required columns: {sorted(missing)}. "
                f"Present columns: {sorted(self.df.columns.tolist())}"
            )
        if len(self.df) == 0:
            raise ValueError("PolicyData DataFrame cannot be empty.")
        self.df = self.df.copy()
        if not pd.api.types.is_bool_dtype(self.df["renewal_flag"]):
            self.df["renewal_flag"] = self.df["renewal_flag"].astype(bool)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PolicyData":
        """Load from a Parquet file. All required columns must be present."""
        df = pd.read_parquet(path)
        return cls(df)

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "PolicyData":
        """Load from a CSV file."""
        df = pd.read_csv(path, **kwargs)
        return cls(df)

    @property
    def n_policies(self) -> int:
        """Total policy count."""
        return len(self.df)

    @property
    def n_renewals(self) -> int:
        """Count of renewal policies."""
        return int(self.df["renewal_flag"].sum())

    @property
    def channels(self) -> list[str]:
        """Distinct channels present in the data."""
        return sorted(self.df["channel"].unique().tolist())

    @property
    def renewal(self) -> pd.DataFrame:
        """Subset of renewal policies."""
        return self.df[self.df["renewal_flag"]].copy()

    @property
    def new_business(self) -> pd.DataFrame:
        """Subset of new business policies."""
        return self.df[~self.df["renewal_flag"]].copy()

    def current_loss_ratio(self) -> float:
        """
        Portfolio-level loss ratio at current premiums.

        Computed as sum of technical premiums divided by sum of current
        premiums — i.e., using technical price as a proxy for expected claims.
        Actual loss ratio requires realised claims data.
        """
        return float(
            self.df["technical_premium"].sum() / self.df["current_premium"].sum()
        )

    def validate_demand_outputs(self) -> None:
        """
        Check that ``renewal_prob`` is present and well-formed.

        Raises
        ------
        ValueError
            If ``renewal_prob`` is absent or contains values outside [0, 1].
        """
        if "renewal_prob" not in self.df.columns:
            raise ValueError(
                "Column 'renewal_prob' is required for optimisation. "
                "Populate it from your demand model before calling the optimiser."
            )
        probs = self.df["renewal_prob"]
        if probs.isna().any():
            raise ValueError("Column 'renewal_prob' contains NaN values.")
        if (probs < 0).any() or (probs > 1).any():
            raise ValueError(
                "Column 'renewal_prob' contains values outside [0, 1]. "
                f"Range found: [{probs.min():.4f}, {probs.max():.4f}]"
            )

    def __repr__(self) -> str:
        return (
            f"PolicyData(n_policies={self.n_policies}, "
            f"n_renewals={self.n_renewals}, "
            f"channels={self.channels})"
        )


@dataclass
class FactorStructure:
    """
    Describes the multiplicative tariff structure.

    In a multiplicative tariff, the premium for policy i is:

        premium_i = base_rate × Π_k factor_k(x_ik)

    The optimiser's decision variables are multiplicative adjustments m_k
    applied to each factor's current relativities. A value of m_k = 1.05
    means factor k's relativities are all scaled up by 5%.

    Parameters
    ----------
    factor_names : sequence of str
        Names of the rating factors. These must match column names in the
        PolicyData DataFrame (after applying the ``factor_col_prefix``).
    factor_values : pd.DataFrame
        DataFrame with one row per policy (in the same order as PolicyData.df)
        and one column per factor containing the current relativity value for
        that policy. These are the multiplicative factor values, not the raw
        feature levels.
    factor_col_prefix : str
        Optional prefix stripped from column names when matching factors.
        Defaults to empty string (exact match).
    renewal_factor_names : sequence of str, optional
        Names of factors that apply only to renewal policies (e.g., tenure
        discounts, no-claims bonus). These are relevant to the ENBP constraint:
        the NB-equivalent price excludes renewal-only factors.
    """

    factor_names: list[str]
    factor_values: pd.DataFrame
    factor_col_prefix: str = ""
    renewal_factor_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.factor_names = list(self.factor_names)
        self.renewal_factor_names = list(self.renewal_factor_names)
        missing = set(self.factor_names) - set(self.factor_values.columns)
        if missing:
            raise ValueError(
                f"factor_values DataFrame is missing columns for factors: "
                f"{sorted(missing)}"
            )
        if (self.factor_values[self.factor_names] <= 0).any().any():
            raise ValueError(
                "All factor values must be strictly positive (multiplicative relativities)."
            )
        for rn in self.renewal_factor_names:
            if rn not in self.factor_names:
                raise ValueError(
                    f"renewal_factor_name '{rn}' is not in factor_names."
                )

    @property
    def n_factors(self) -> int:
        """Number of rating factors."""
        return len(self.factor_names)

    @property
    def non_renewal_factor_names(self) -> list[str]:
        """Factors that apply to both renewal and new business."""
        return [f for f in self.factor_names if f not in self.renewal_factor_names]

    def adjusted_premiums(
        self,
        current_premiums: np.ndarray,
        adjustments: np.ndarray,
        factor_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute premiums after applying multiplicative factor adjustments.

        Each policy's premium is scaled by the product of the adjustment
        multipliers for each factor. This is the core premium calculation
        used throughout the optimiser.

        Parameters
        ----------
        current_premiums : np.ndarray
            Shape (n_policies,). Current premiums before adjustment.
        adjustments : np.ndarray
            Shape (n_factors,). Multiplicative adjustment for each factor.
            Value of 1.0 means no change; 1.05 means +5%.
        factor_mask : np.ndarray, optional
            Boolean mask shape (n_factors,). If supplied, only the masked
            factors are included in the adjustment product. Used for ENBP
            calculation (exclude renewal-only factors).

        Returns
        -------
        np.ndarray
            Shape (n_policies,). Adjusted premiums.
        """
        factor_vals = self.factor_values[self.factor_names].values  # (n, k)
        adj = adjustments.copy()
        if factor_mask is not None:
            adj = np.where(factor_mask, adj, 1.0)
        # For each policy: premium * prod_k (adj_k / 1) where adj_k shifts the relativity
        # The adjustment m_k scales factor k's relativities uniformly.
        # New relativity for factor k on policy i = factor_vals[i,k] * adj[k]
        # But since current_premium already embeds current factor_vals, we multiply by
        # the ratio of new to old factor product:
        ratio = np.prod(adj, axis=0)  # scalar if all factors adjusted uniformly
        # Per-policy: multiply current_premium by prod_k(adj_k) (uniform shift)
        # This is correct for multiplicative tariff: if each factor k is scaled by m_k,
        # then for policy i the overall premium changes by prod_k(m_k) where m_k is the
        # specific adjustment for each factor that applies to that policy.
        # Since every policy carries all factors, the adjustment to policy i is prod_k(m_k).
        policy_adjustments = np.prod(factor_vals * adj[np.newaxis, :] / factor_vals, axis=1)
        return current_premiums * policy_adjustments

    def premium_ratio(
        self,
        adjustments: np.ndarray,
        factor_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the per-policy premium ratio (new / current) for given factor adjustments.

        Parameters
        ----------
        adjustments : np.ndarray
            Shape (n_factors,). Multiplicative adjustment per factor.
        factor_mask : np.ndarray, optional
            Boolean mask for factor inclusion.

        Returns
        -------
        np.ndarray
            Shape (n_policies,). Per-policy ratio of new to current premium.
        """
        adj = adjustments.copy()
        if factor_mask is not None:
            adj = np.where(factor_mask, adj, 1.0)
        # Each policy gets the product of all factor adjustments
        return np.full(len(self.factor_values), np.prod(adj))

    def renewal_only_mask(self) -> np.ndarray:
        """
        Boolean mask over factor_names: True for non-renewal factors.

        Used to compute the NB-equivalent premium by excluding renewal-specific
        factor adjustments.
        """
        return np.array(
            [f not in self.renewal_factor_names for f in self.factor_names],
            dtype=bool,
        )

    def initial_adjustments(self) -> np.ndarray:
        """Return an array of ones — the identity (no change) adjustment."""
        return np.ones(self.n_factors)

    def __repr__(self) -> str:
        return (
            f"FactorStructure(n_factors={self.n_factors}, "
            f"factor_names={self.factor_names}, "
            f"renewal_factors={self.renewal_factor_names})"
        )
