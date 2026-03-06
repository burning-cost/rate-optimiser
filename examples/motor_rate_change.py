"""
Motor rate change optimisation — worked example.

This script demonstrates end-to-end use of rate-optimiser on a synthetic UK
motor book. It mirrors the kind of rate review cycle a pricing team would run
quarterly: start with GLM outputs, add demand model predictions, set constraints,
solve, and trace the efficient frontier.

The synthetic data is deliberately simplified — five rating factors, 200 policies,
a clean logistic demand model. In production you would replace this with your
actual GLM and demand model outputs.

Run from the repo root:
    uv run python examples/motor_rate_change.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from rate_optimiser import (
    PolicyData,
    FactorStructure,
    DemandModel,
    RateChangeOptimiser,
    EfficientFrontier,
    LossRatioConstraint,
    VolumeConstraint,
    ENBPConstraint,
    FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams
from rate_optimiser.stochastic import ClaimsVarianceModel, StochasticRateOptimiser


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
#    In production: replace with your GLM output DataFrame.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(2026)
N = 200

age_relativity = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_relativity = rng.choice([0.70, 0.80, 0.90, 1.00], N, p=[0.30, 0.30, 0.25, 0.15])
vehicle_relativity = rng.choice([0.90, 1.00, 1.10, 1.30], N, p=[0.25, 0.35, 0.25, 0.15])
region_relativity = rng.choice([0.85, 1.00, 1.10, 1.20], N, p=[0.20, 0.40, 0.25, 0.15])
tenure = rng.integers(0, 10, N).astype(float)
tenure_discount = np.ones(N)  # renewal-only factor; currently neutral

base_rate = 350.0
technical_premium = (
    base_rate
    * age_relativity * ncb_relativity * vehicle_relativity * region_relativity
    * rng.uniform(0.97, 1.03, N)
)

# Current premium has LR of ~0.75: some pricing above technical
current_premium = technical_premium / 0.75 * rng.uniform(0.98, 1.02, N)

# Market premium: competitive, slightly below current
market_premium = technical_premium / 0.73 * rng.uniform(0.92, 1.08, N)

renewal_flag = rng.random(N) < 0.60
channel = np.where(renewal_flag,
                   rng.choice(["PCW", "direct"], N, p=[0.70, 0.30]),
                   rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]))

# Demand model: logistic with semi-elasticity = -2.0
price_ratio = current_premium / market_premium
logit_vals = 1.0 + (-2.0) * np.log(price_ratio) + 0.05 * tenure
renewal_prob = expit(logit_vals)

df = pd.DataFrame({
    "policy_id": [f"MTR{i:05d}" for i in range(N)],
    "channel": channel,
    "renewal_flag": renewal_flag,
    "technical_premium": technical_premium,
    "current_premium": current_premium,
    "market_premium": market_premium,
    "renewal_prob": renewal_prob,
    "tenure": tenure,
    "f_age": age_relativity,
    "f_ncb": ncb_relativity,
    "f_vehicle": vehicle_relativity,
    "f_region": region_relativity,
    "f_tenure_discount": tenure_discount,
})


# ---------------------------------------------------------------------------
# 2. Wrap in rate-optimiser data layer
# ---------------------------------------------------------------------------

data = PolicyData(df)
print(f"Portfolio: {data.n_policies} policies, {data.n_renewals} renewals")
print(f"Current LR (at face premiums): {data.current_loss_ratio():.3f}")
print(f"Channels: {data.channels}")


# ---------------------------------------------------------------------------
# 3. Factor structure
# ---------------------------------------------------------------------------

factor_names = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]
fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df[factor_names],
    renewal_factor_names=["f_tenure_discount"],  # renewal-only: subject to ENBP
)
print(f"\nFactor structure: {fs.n_factors} factors")
print(f"Renewal-only factors (ENBP relevant): {fs.renewal_factor_names}")


# ---------------------------------------------------------------------------
# 4. Demand model
# ---------------------------------------------------------------------------

params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.05)
demand = make_logistic_demand(params)

# Check elasticities are in expected range
sample_ratios = np.ones(10)
elasticities = demand.elasticity_at(sample_ratios)
print(f"\nPrice elasticity at market price: {elasticities.mean():.2f} "
      f"(expected ~-1.73 for beta=-2.0, p≈0.73)")


# ---------------------------------------------------------------------------
# 5. Constrained optimisation
# ---------------------------------------------------------------------------

opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt.add_constraint(LossRatioConstraint(bound=0.72))
opt.add_constraint(VolumeConstraint(bound=0.96))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

print("\nFeasibility at identity (no rate change):")
print(opt.feasibility_report().to_string(index=False))

result = opt.solve()
print(f"\n{result.summary()}")

print("\nFactor adjustments (deviation from 1.0 = no change):")
for factor, adj in result.factor_adjustments.items():
    direction = "UP" if adj > 1.0 else ("DOWN" if adj < 1.0 else "FLAT")
    print(f"  {factor:25s}: {adj:.4f}  ({(adj-1)*100:+.1f}%)  [{direction}]")


# ---------------------------------------------------------------------------
# 6. Efficient frontier trace
# ---------------------------------------------------------------------------

print("\nTracing efficient frontier (LR 0.69 → 0.78, 12 points)...")
frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.69, 0.78), n_points=12)

print("\nFrontier results (feasible points only):")
feasible = frontier.feasible_points()
print(
    feasible[["lr_target", "expected_lr", "expected_volume", "shadow_lr"]]
    .round(4)
    .to_string(index=False)
)

print("\nShadow price interpretation:")
if len(feasible) >= 2:
    # Shadow price on LR constraint: marginal cost of tightening LR target
    mid = feasible.iloc[len(feasible) // 2]
    print(
        f"  At LR target {mid['lr_target']:.3f}: shadow price = {mid['shadow_lr']:.4f}\n"
        f"  Interpretation: relaxing the LR target by 0.01 would save "
        f"  {abs(mid['shadow_lr']) * 0.01:.5f} units of objective (dislocation)."
    )


# ---------------------------------------------------------------------------
# 7. Stochastic (chance-constrained) optimisation
# ---------------------------------------------------------------------------

print("\n--- Stochastic rate optimisation (Branda approach) ---")

variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=data.df["technical_premium"].values,
    dispersion=1.2,
    power=1.5,
)

stoc_opt = StochasticRateOptimiser(
    data=data,
    demand=demand,
    factor_structure=fs,
    variance_model=variance_model,
    lr_bound=0.72,
    alpha=0.95,
)
stoc_opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
stoc_opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

stoc_result = stoc_opt.solve()
print(f"Stochastic result (alpha=0.95):\n{stoc_result.summary()}")

print("\nComparison — deterministic vs. stochastic LR target at 72%:")
print(f"  Deterministic expected LR: {result.expected_lr:.4f}")
print(f"  Stochastic expected LR:    {stoc_result.expected_lr:.4f}")
det_total = sum(result.factor_adjustments.values())
stoc_total = sum(stoc_result.factor_adjustments.values())
print(f"  Deterministic sum of adjustments: {det_total:.4f}")
print(f"  Stochastic sum of adjustments:    {stoc_total:.4f}")
print(
    "\nThe stochastic formulation requires a higher rate increase because it must "
    "hold the loss ratio below 72% with 95% probability, not just in expectation."
)

print("\nExample complete.")
