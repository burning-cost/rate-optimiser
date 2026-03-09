# rate-optimiser
[![Tests](https://github.com/burning-cost/rate-optimiser/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/rate-optimiser/actions/workflows/tests.yml)

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/rate-optimiser)

Constrained rate change optimiser for UK personal lines insurance pricing.

The insurance analogue of Markowitz portfolio optimisation. Takes policy-level GLM outputs and demand model predictions; finds multiplicative factor adjustments that meet loss ratio and volume targets simultaneously; traces the efficient frontier of achievable rate strategies.

No open-source tool does this. Commercial tools (Radar Optimiser, Earnix, Akur8) have opaque solvers, inflexible constraint specifications, and no Python API. This library is an auditable, extensible alternative built on scipy and numpy.

---

## The problem

A UK motor pricing team wants to take +3.5% rate on renewal. Before submitting to the underwriting director, they need answers to three questions:

1. Which rating factors should move, and by how much?
2. Does the proposed strategy hit the LR target without breaching the volume budget?
3. Is the renewal price compliant with FCA PS21/5 (no renewal exceeds NB equivalent)?

Current tooling forces them to hand-code scenarios in Excel or run ad hoc simulations. The efficient frontier - the full set of achievable (LR, volume) outcomes - is never computed. Shadow prices on constraints (what is a 1pp LR improvement actually worth in volume terms?) are unknown.

This library solves those three questions formally. You specify constraints, it finds the minimum-dislocation rate strategy that satisfies them, and it maps the entire frontier so you can see what trade-offs you are making.

---

## What this is not

- **Not a GLM fitting tool.** Use statsmodels, scikit-learn, or Emblem. This library consumes their outputs.
- **Not a real-time quote engine.** Radar Live and Earnix handle individual-level pricing at point of quote. This is an offline rate strategy tool.
- **Not a reserving or capital tool.** Out of scope.

---

## Installation

```bash
uv add rate-optimiser
```

With stochastic module (requires cvxpy):

```bash
uv add "rate-optimiser[stochastic]"
```

From source with uv:

```bash
git clone https://github.com/burning-cost/rate-optimiser
cd rate-optimiser
uv sync --extra dev
```

---

## Quick start

```python
import polars as pl
from rate_optimiser import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# 1. Load your GLM outputs
#    Required columns: policy_id, channel, renewal_flag,
#    technical_premium, current_premium
#    Your demand model populates renewal_prob
df = pl.read_parquet("policies.parquet")
data = PolicyData(df)

# 2. Factor structure - describes the multiplicative tariff
factor_names = ["f_age_band", "f_ncb", "f_vehicle_group", "f_region", "f_tenure_discount"]
fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df.select(factor_names),   # current relativity values per policy
    renewal_factor_names=["f_tenure_discount"],  # renewal-only; ENBP-relevant
)

# 3. Demand model - wrap your logistic model or any callable
#    This form: logit(p) = intercept + beta * log(price_ratio) + tenure_coef * tenure
params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.05)
demand = make_logistic_demand(params)
#    Or pass any sklearn estimator:
#    demand = DemandModel(my_catboost_model, feature_names=["age", "tenure", "ncb"])

# 4. Configure the optimiser
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt.add_constraint(LossRatioConstraint(bound=0.72))         # max 72% LR
opt.add_constraint(VolumeConstraint(bound=0.97))             # max 3% volume loss
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))  # FCA PS21/5
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

# 5. Check feasibility at current rates before solving
print(opt.feasibility_report())

# 6. Solve
result = opt.solve()
print(result.summary())

# result.factor_adjustments: {"f_age_band": 1.04, "f_ncb": 1.02, ...}
# result.expected_lr: 0.7198
# result.expected_volume: 0.9712
# result.shadow_prices: {"loss_ratio_ub": 0.031, "volume_lb": 0.0, "enbp": 0.0}

# 7. Trace the efficient frontier
frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)
# Returns DataFrame: lr_target, expected_lr, expected_volume, shadow_lr, feasible, ...

frontier.plot()  # matplotlib efficient frontier chart
```

---

## The efficient frontier

The core insight borrowed from Markowitz: rather than solving for a single rate strategy, trace the full Pareto frontier of achievable (LR, volume) pairs.

```python
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)
print(frontier.shadow_price_summary())
```

```
 lr_target  expected_lr  expected_volume  shadow_lr  shadow_volume
      0.78        0.777            0.973       0.02           0.00
      0.76        0.758            0.971       0.04           0.00
      0.74        0.739            0.968       0.08           0.00
      0.72        0.720            0.963       0.15           0.00
      0.70        0.700            0.954       0.31           0.01
      0.68        0.680            0.937       0.72           0.08
```

The `shadow_lr` column is the Lagrange multiplier on the LR constraint: the marginal dislocation cost of a one-unit tightening of the target. A rising shadow price signals you are approaching the frontier's knee - the point where further LR improvement costs disproportionate volume. That is a number worth putting in front of a commercial director.

---

## Stochastic formulation (Branda approach)

The deterministic constraint E[LR] <= target uses point estimates of claims. The stochastic formulation requires P(LR <= target) >= alpha - the LR must stay below the target with confidence level alpha.

Reformulated via normal approximation (appropriate for large books):

```
E[LR] + z_alpha * sigma[LR] <= target
```

where sigma[LR] comes from the GLM's variance estimates.

```python
from rate_optimiser.stochastic import ClaimsVarianceModel, StochasticRateOptimiser

variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=data.df["technical_premium"].to_numpy(),
    dispersion=1.2,  # from your Tweedie GLM summary
    power=1.5,
)

opt = StochasticRateOptimiser(
    data=data, demand=demand, factor_structure=fs,
    variance_model=variance_model,
    lr_bound=0.72,
    alpha=0.95,  # 95% confidence
)
result = opt.solve()
```

The stochastic solver will recommend a higher rate than the deterministic one because it must maintain the LR constraint with high probability, not just in expectation. The difference between the two solutions quantifies the uncertainty premium in the rate strategy.

---

## ENBP constraint (FCA PS21/5)

PS21/5 prohibits renewal premiums above the NB equivalent through the same channel. `ENBPConstraint` enforces this formally:

```python
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
```

The constraint is channel-specific, as the regulation requires. The shadow price tells you the cost (in objective terms) of the regulatory constraint - how much additional dislocation the insurer incurs to comply. This is directly relevant to PS21/5 impact analyses.

The NB-equivalent is computed by applying all factor adjustments excluding renewal-only factors (e.g., tenure discounts, NCB-at-renewal). Declare renewal-only factors in the `FactorStructure`:

```python
fs = FactorStructure(
    factor_names=["f_age_band", "f_ncb", "f_tenure_discount"],
    factor_values=df.select(["f_age_band", "f_ncb", "f_tenure_discount"]),
    renewal_factor_names=["f_tenure_discount"],  # excluded from NB equivalent
)
```

---

## Features

- **Constrained SLSQP optimisation** via scipy. Finds minimum-dislocation factor adjustments meeting multiple simultaneous constraints.
- **Efficient frontier tracing.** Parametric sweep over LR targets, returning the full (LR, volume) tradeoff surface with shadow prices at each point.
- **Shadow prices on all constraints.** Lagrange multipliers extracted from the SLSQP solution. Tells you which constraints are binding and at what cost.
- **FCA PS21/5 ENBP constraint.** Channel-aware, renewal-only factor aware. Not available in any other open-source tool.
- **Stochastic chance-constrained formulation.** Branda (2013) approach: P(LR <= target) >= alpha using GLM variance estimates.
- **sklearn-compatible demand model interface.** Pass any sklearn estimator or a simple callable.
- **Feasibility reporting.** Before running the solver, check whether your constraints are satisfiable at current rates.
- **Multiple objective functions.** Minimum dislocation (||m-1||^2), premium-weighted dislocation, minimum absolute dislocation.

---

## Methodology

The optimisation problem:

```
minimise   sum_k (m_k - 1)^2
subject to E[LR(m)] <= LR_target
           E[vol_ratio(m)] >= vol_bound
           m_k in [m_k_min, m_k_max]  for all k
           pi_i^renewal <= pi_i^NB_equiv  (ENBP)
```

Decision variables `m_k` are multiplicative adjustments to each rating factor's relativities. A value of 1.05 means factor k's relativities are uniformly scaled up by 5% - a parallel shift on the log scale.

The demand model enters through the volume and LR constraints: `p_i(pi_i / pi_market_i)` is the probability that policy i renews at the adjusted premium. This makes both constraints nonlinear in `m`, which is why the problem requires SLSQP or a similar nonlinear solver.

The efficient frontier is traced by solving this problem for a range of `LR_target` values and collecting the resulting (expected_LR, expected_volume) pairs - directly analogous to the Markowitz frontier construction.

### Academic foundations

- Branda, M. (2013). "Optimization Approaches to Multiplicative Tariff of Rates." ASTIN Colloquium, Hague.
- Guven, S. and McPhail, J. (2013). "Beyond the Cost Model: Understanding Price Elasticity." CAS Spring Forum.
- Emms, P. and Haberman, S. (2005). "Pricing General Insurance Using Optimal Control Theory." ASTIN Bulletin 35(2).
- FCA (2021). PS21/5: General Insurance Pricing Practices.

---

## Development

```bash
git clone https://github.com/burning-cost/rate-optimiser
cd rate-optimiser
uv sync --extra dev
uv run pytest -v
```

Tests run on Databricks (see repo CI); do not run locally on resource-constrained machines.

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burningcost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burningcost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burningcost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burningcost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burningcost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burningcost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-demand](https://github.com/burningcost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burningcost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burningcost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-monitoring](https://github.com/burningcost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burningcost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All libraries →](https://burningcost.github.io)

---

## Licence

MIT. See [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests welcome. The priority backlog:

1. Competitive equilibrium module: Lerner index pricing (pi* = c + 1/beta) as a baseline.
2. Bayesian demand model integration: propagate posterior uncertainty over beta through the optimiser.
3. Multi-period optimisation: Emms/Haberman (2005) HJB framework for dynamic pricing.
4. Consumer Duty fair value checker: flag optimised rates that systematically disadvantage protected groups.
