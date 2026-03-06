"""
Shared test fixtures for rate-optimiser tests.

Uses synthetic motor insurance data with 5 rating factors and a known logistic
demand model. The data is constructed so that:

- Base loss ratio at current premiums is ~0.75 (needs a rate lift to hit 0.72)
- Base renewal rate is ~78% at current premiums (demand model matches stored probs)
- Factors span plausible motor ranges (age band, NCB, vehicle group, region, tenure)
- Renewals and new business are split ~60/40

The demand model uses beta = -2.0 (price semi-elasticity), within the -1.5 to -3.0
range cited for UK motor PCW, and includes a tenure effect. Crucially, the demand
model used in the optimiser must produce the same probabilities as the stored
``renewal_prob`` column when called at the identity adjustment — otherwise the
volume ratio at the starting point is not 1.0 and constraint bounds become confusing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from rate_optimiser.data import PolicyData, FactorStructure
from rate_optimiser.demand import DemandModel
from rate_optimiser.constraints import (
    LossRatioConstraint,
    VolumeConstraint,
    FactorBoundsConstraint,
    ENBPConstraint,
)
from rate_optimiser.optimiser import RateChangeOptimiser


N_POLICIES = 120
RANDOM_SEED = 42
INTERCEPT = 1.0
PRICE_COEF = -2.0
TENURE_COEF = 0.05


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def raw_policy_df(rng) -> pl.DataFrame:
    """Synthetic motor policy DataFrame."""
    n = N_POLICIES

    # Rating factor relativities (multiplicative)
    age_band = rng.choice([0.8, 1.0, 1.2, 1.5, 2.0], size=n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    ncb = rng.choice([0.7, 0.8, 0.9, 1.0], size=n, p=[0.30, 0.30, 0.25, 0.15])
    vehicle_group = rng.choice([0.9, 1.0, 1.1, 1.3], size=n, p=[0.25, 0.35, 0.25, 0.15])
    region = rng.choice([0.85, 1.0, 1.1, 1.2], size=n, p=[0.20, 0.40, 0.25, 0.15])
    tenure = rng.integers(0, 10, size=n).astype(float)

    # Technical premium = base * factor product
    base_rate = 350.0
    factor_product = age_band * ncb * vehicle_group * region
    technical_premium = base_rate * factor_product * rng.uniform(0.95, 1.05, size=n)

    # Current premium: slightly above technical (underlying LR ~0.75)
    current_premium = technical_premium / 0.75 * rng.uniform(0.98, 1.02, size=n)

    # Market premium: roughly at technical price (competitive market)
    market_premium = technical_premium / 0.73 * rng.uniform(0.90, 1.10, size=n)

    # Renewal flag: 60% renewals
    renewal_flag = rng.random(size=n) < 0.60

    # Channel: PCW-heavy for renewals
    channel_options = ["PCW", "direct"]
    channel = np.where(
        renewal_flag,
        rng.choice(channel_options, size=n, p=[0.70, 0.30]),
        rng.choice(channel_options, size=n, p=[0.60, 0.40]),
    )

    # Demand model ground truth: logistic with beta=-2.0, tenure effect
    price_ratio = current_premium / market_premium
    logit = INTERCEPT + PRICE_COEF * np.log(price_ratio) + TENURE_COEF * tenure
    renewal_prob = expit(logit)

    df = pl.DataFrame({
        "policy_id": [f"POL{i:04d}" for i in range(n)],
        "channel": channel.tolist(),
        "renewal_flag": renewal_flag.tolist(),
        "technical_premium": technical_premium.tolist(),
        "current_premium": current_premium.tolist(),
        "market_premium": market_premium.tolist(),
        "renewal_prob": renewal_prob.tolist(),
        "tenure": tenure.tolist(),
        # Store factor relativities as columns
        "factor_age_band": age_band.tolist(),
        "factor_ncb": ncb.tolist(),
        "factor_vehicle_group": vehicle_group.tolist(),
        "factor_region": region.tolist(),
        "factor_tenure_discount": [1.0] * n,
    })

    return df


@pytest.fixture(scope="session")
def policy_data(raw_policy_df) -> PolicyData:
    return PolicyData(raw_policy_df)


@pytest.fixture(scope="session")
def factor_names() -> list[str]:
    return [
        "factor_age_band",
        "factor_ncb",
        "factor_vehicle_group",
        "factor_region",
        "factor_tenure_discount",
    ]


@pytest.fixture(scope="session")
def factor_structure(raw_policy_df, factor_names) -> FactorStructure:
    return FactorStructure(
        factor_names=factor_names,
        factor_values=raw_policy_df.select(factor_names),
        renewal_factor_names=["factor_tenure_discount"],
    )


@pytest.fixture(scope="session")
def demand_model() -> DemandModel:
    """
    Logistic demand model matching the data generation process.

    This model includes the tenure feature so that, at the identity adjustment,
    it produces probabilities that exactly match the stored ``renewal_prob``
    column. This ensures vol_ratio = 1.0 at the starting point.
    """
    intercept = INTERCEPT
    price_coef = PRICE_COEF
    tenure_coef = TENURE_COEF

    def _logistic(price_ratio: np.ndarray, features=None) -> np.ndarray:
        log_ratio = np.log(np.clip(price_ratio, 1e-6, 10.0))
        linear = intercept + price_coef * log_ratio
        if features is not None and "tenure" in features.columns:
            linear = linear + tenure_coef * features["tenure"].to_numpy()
        return expit(linear)

    return DemandModel(_logistic, feature_names=["tenure"])


@pytest.fixture
def basic_optimiser(policy_data, demand_model, factor_structure) -> RateChangeOptimiser:
    """Optimiser with standard LR + volume constraints."""
    opt = RateChangeOptimiser(
        data=policy_data,
        demand=demand_model,
        factor_structure=factor_structure,
    )
    opt.add_constraint(LossRatioConstraint(bound=0.72))
    opt.add_constraint(VolumeConstraint(bound=0.96))
    opt.add_constraint(
        FactorBoundsConstraint(lower=0.85, upper=1.20, n_factors=factor_structure.n_factors)
    )
    return opt


@pytest.fixture
def unconstrained_optimiser(policy_data, demand_model, factor_structure) -> RateChangeOptimiser:
    """Optimiser with only factor bounds — no LR or volume constraint."""
    opt = RateChangeOptimiser(
        data=policy_data,
        demand=demand_model,
        factor_structure=factor_structure,
    )
    opt.add_constraint(
        FactorBoundsConstraint(lower=0.85, upper=1.20, n_factors=factor_structure.n_factors)
    )
    return opt
