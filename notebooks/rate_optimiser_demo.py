# Databricks notebook source
# MAGIC %md
# MAGIC # rate-optimiser: Constrained Rate Change Optimisation
# MAGIC
# MAGIC The insurance analogue of Markowitz portfolio optimisation. A UK motor pricing
# MAGIC team wants to take +3.5% average rate on renewal. Before presenting to the
# MAGIC underwriting director, they need to answer three questions:
# MAGIC
# MAGIC 1. Which rating factors should move, and by how much?
# MAGIC 2. Does the strategy hit the LR target without breaching the volume budget?
# MAGIC 3. Is the renewal price compliant with FCA PS21/5 (no renewal exceeds NB equivalent)?
# MAGIC
# MAGIC This notebook solves all three, then traces the efficient frontier so the team
# MAGIC can see the full (LR, volume) tradeoff surface.
# MAGIC
# MAGIC ## What this demonstrates
# MAGIC
# MAGIC 1. Build synthetic UK motor renewal portfolio
# MAGIC 2. Feasibility check at current rates
# MAGIC 3. Solve for minimum-dislocation factor adjustments
# MAGIC 4. Shadow prices on constraints
# MAGIC 5. Trace the efficient frontier
# MAGIC 6. FCA PS21/5 ENBP constraint

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install rate-optimiser polars numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import polars as pl
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

print("rate-optimiser imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK motor renewal portfolio
# MAGIC
# MAGIC We generate 10,000 renewal policies with:
# MAGIC - Technical premium from a simple multiplicative tariff
# MAGIC - Current premium with some undercutting (current LR > target)
# MAGIC - Pre-computed renewal probability from a logistic demand model
# MAGIC
# MAGIC In production, technical premium comes from your GLM/GBM, and renewal_prob
# MAGIC comes from your fitted demand model. This library consumes those outputs — it
# MAGIC does not refit models.

# COMMAND ----------

rng = np.random.default_rng(42)
N = 10_000

# Rating factors (multiplicative relativities, already applied to base rate)
age_band = rng.choice([0.85, 0.95, 1.00, 1.08, 1.20], size=N, p=[0.15, 0.25, 0.30, 0.20, 0.10])
ncb = rng.choice([1.30, 1.15, 1.00, 0.90, 0.82, 0.75], size=N, p=[0.05, 0.10, 0.15, 0.20, 0.25, 0.25])
vehicle_group = rng.choice([0.90, 1.00, 1.10, 1.25], size=N, p=[0.20, 0.40, 0.30, 0.10])
region = rng.choice([0.92, 1.00, 1.08, 1.15], size=N, p=[0.25, 0.35, 0.25, 0.15])
tenure_discount = rng.choice([1.00, 0.97, 0.95, 0.93], size=N, p=[0.30, 0.25, 0.25, 0.20])
tenure_years = rng.integers(0, 10, N)

# Technical premium: base £400 × product of all factors
base_rate = 400.0
technical_premium = base_rate * age_band * ncb * vehicle_group * region * tenure_discount
technical_premium = technical_premium + rng.normal(0, 5, N)  # small noise

# Current premium: slightly below technical on average (current book is underpriced)
# This gives us a current LR of ~78%, target is 72%
current_premium = technical_premium * rng.uniform(0.88, 0.98, N)

# Channel: 60% PCW, 40% direct
channel = rng.choice(["PCW", "direct"], size=N, p=[0.60, 0.40])

# Demand model: logit(p_renew) = 1.0 - 2.0 * log(price_ratio) + 0.05 * tenure
log_price_ratio = np.log(current_premium / technical_premium)  # ~log(0.93) ≈ -0.075
renewal_prob = expit(1.0 - 2.0 * log_price_ratio + 0.05 * tenure_years)

df = pl.DataFrame({
    "policy_id": np.arange(N),
    "channel": channel,
    "renewal_flag": np.ones(N, dtype=bool),
    "technical_premium": technical_premium,
    "current_premium": current_premium,
    "market_premium": technical_premium,  # assume at-market technical pricing
    "renewal_prob": renewal_prob,
    "tenure": tenure_years.astype(float),
    # Factor columns
    "f_age_band": age_band,
    "f_ncb": ncb,
    "f_vehicle_group": vehicle_group,
    "f_region": region,
    "f_tenure_discount": tenure_discount,
})

data = PolicyData(df)

print(f"Portfolio: {data.n_policies:,} policies, {data.n_renewals:,} renewals")
print(f"Channels: {data.channels}")
print(f"Current LR (technical/current): {data.current_loss_ratio():.3f}")
print(f"Mean renewal probability: {df['renewal_prob'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Factor structure and demand model
# MAGIC
# MAGIC `FactorStructure` describes the multiplicative tariff. The decision variables
# MAGIC are uniform multiplicative adjustments `m_k` to each factor — so m_age_band=1.05
# MAGIC means every age band relativity is scaled up by 5%.
# MAGIC
# MAGIC `f_tenure_discount` is declared as renewal-only: it is excluded from the
# MAGIC NB-equivalent premium calculation for the PS21/5 ENBP constraint.

# COMMAND ----------

factor_names = ["f_age_band", "f_ncb", "f_vehicle_group", "f_region", "f_tenure_discount"]

fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df.select(factor_names),
    renewal_factor_names=["f_tenure_discount"],
)

print(fs)
print(f"\nNon-renewal factors (included in ENBP calculation): {fs.non_renewal_factor_names}")

# COMMAND ----------

# Demand model: logistic with price semi-elasticity -2.0
# In production, fit this from quote/bind data or price-test results
params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.05)
demand = make_logistic_demand(params)

# Check elasticity at current premiums
price_ratio = (df["current_premium"] / df["market_premium"]).to_numpy()
elasticity = demand.elasticity_at(price_ratio, policy_features=df)
print(f"Mean price elasticity: {elasticity.mean():.3f} (expect ~-1.5 to -2.5 for UK motor PCW)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feasibility check
# MAGIC
# MAGIC Before running the solver, check whether the target constraints are achievable
# MAGIC at the current rates. The current LR of ~0.78 is above the 0.72 target —
# MAGIC the book is underpriced and needs a rate increase.

# COMMAND ----------

opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt.add_constraint(LossRatioConstraint(bound=0.72))
opt.add_constraint(VolumeConstraint(bound=0.97))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

feasibility = opt.feasibility_report()
print(feasibility)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Solve for optimal factor adjustments

# COMMAND ----------

print("Solving for optimal factor adjustments...")
result = opt.solve()

print("\n" + "=" * 60)
print(result.summary())

# COMMAND ----------

# Factor adjustments
print("\nOptimal factor adjustments:")
for factor, adjustment in result.factor_adjustments.items():
    pct_change = (adjustment - 1.0) * 100
    print(f"  {factor:25s}: {adjustment:.4f}  ({pct_change:+.1f}%)")

print(f"\nExpected portfolio LR: {result.expected_lr:.4f}")
print(f"Expected volume ratio: {result.expected_volume:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Shadow prices
# MAGIC
# MAGIC The Lagrange multipliers tell you which constraints are binding and at what
# MAGIC cost. A non-zero shadow price on LR means tightening the LR target costs
# MAGIC additional volume. This is the number to put in front of a commercial director.

# COMMAND ----------

print("Shadow prices (Lagrange multipliers):")
for constraint_name, shadow_price in result.shadow_prices.items():
    status = "BINDING" if abs(shadow_price) > 1e-4 else "slack"
    print(f"  {constraint_name:30s}: {shadow_price:.6f}  [{status}]")

print()
if result.shadow_prices.get("loss_ratio_ub", 0) > 1e-4:
    sp = result.shadow_prices["loss_ratio_ub"]
    print(f"The LR constraint is binding. Shadow price = {sp:.4f}")
    print("Tightening the LR target by 1pp costs this much additional dislocation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Efficient frontier
# MAGIC
# MAGIC Rather than solving for a single rate strategy, trace the full Pareto frontier
# MAGIC of achievable (LR, volume) pairs. This is the Markowitz insight applied to
# MAGIC insurance pricing: the frontier shows all efficient rate strategies, and the
# MAGIC commercial team can choose where on the curve to operate.

# COMMAND ----------

print("Tracing efficient frontier (LR range 0.68 to 0.80)...")
frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.80), n_points=13)

print("\nEfficient frontier:")
display(frontier_df.select([
    "lr_target", "expected_lr", "expected_volume", "shadow_lr", "feasible"
]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Interpret the frontier
# MAGIC
# MAGIC The shadow price column shows where the frontier's knee is: the point where
# MAGIC further LR improvement costs disproportionate volume. A rising shadow price
# MAGIC signals you are approaching that knee.

# COMMAND ----------

feasible = frontier_df.filter(pl.col("feasible") == True)

print("Feasible frontier points:")
print(f"{'LR target':>12} {'Expected LR':>13} {'Volume ratio':>14} {'Shadow price':>14}")
print("-" * 60)
for row in feasible.iter_rows(named=True):
    lr_t = row["lr_target"]
    e_lr = row["expected_lr"]
    e_vol = row["expected_volume"]
    shadow = row["shadow_lr"]
    print(f"{lr_t:>12.3f} {e_lr:>13.4f} {e_vol:>14.4f} {shadow:>14.4f}")

print()
print("The knee of the frontier is where shadow_lr starts rising steeply.")
print("That is the point where further LR improvement costs disproportionate volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. FCA PS21/5 ENBP constraint
# MAGIC
# MAGIC The `ENBPConstraint` ensures no renewal price exceeds the NB equivalent through
# MAGIC the same channel. The `f_tenure_discount` factor was declared renewal-only in
# MAGIC `FactorStructure`, so it is excluded from the NB-equivalent calculation —
# MAGIC tenure discounts are permitted under PS21/5 as they are based on costs.

# COMMAND ----------

# Run the optimiser with ENBP vs without, and compare the shadow prices
opt_no_enbp = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt_no_enbp.add_constraint(LossRatioConstraint(bound=0.72))
opt_no_enbp.add_constraint(VolumeConstraint(bound=0.97))
opt_no_enbp.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

result_no_enbp = opt_no_enbp.solve()

print("With ENBP constraint:")
print(f"  Expected LR:     {result.expected_lr:.4f}")
print(f"  Expected volume: {result.expected_volume:.4f}")
print(f"  ENBP shadow:     {result.shadow_prices.get('enbp_PCW', 0):.6f}")

print("\nWithout ENBP constraint:")
print(f"  Expected LR:     {result_no_enbp.expected_lr:.4f}")
print(f"  Expected volume: {result_no_enbp.expected_volume:.4f}")

lr_cost = result.expected_lr - result_no_enbp.expected_lr
vol_cost = result.expected_volume - result_no_enbp.expected_volume
print(f"\nRegulatory compliance cost: {lr_cost:+.4f} LR, {vol_cost:+.4f} volume")
print("(The difference quantifies the cost of PS21/5 compliance.)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Scenario | Expected LR | Expected volume | Status |
# MAGIC |----------|------------|-----------------|--------|
# MAGIC | Current rates | ~0.78 | baseline | Underpriced |
# MAGIC | Optimised (with ENBP) | ~0.72 | ~0.97 | Compliant |
# MAGIC | Optimised (no ENBP) | ~0.72 | ~0.97 | Non-compliant |
# MAGIC
# MAGIC The solver finds the minimum-dislocation rate strategy that simultaneously:
# MAGIC - Hits the 72% LR target
# MAGIC - Limits volume loss to 3%
# MAGIC - Complies with FCA PS21/5 (no renewal above NB equivalent)
# MAGIC - Keeps factor adjustments within ±15% bounds
# MAGIC
# MAGIC The efficient frontier shows all (LR, volume) tradeoffs. Shadow prices identify
# MAGIC which constraints are binding and at what cost — directly relevant to
# MAGIC commercial and regulatory impact analysis.
