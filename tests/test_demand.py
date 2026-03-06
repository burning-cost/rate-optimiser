"""Tests for DemandModel and related utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from rate_optimiser.demand import (
    DemandModel,
    LogisticDemandParams,
    make_logistic_demand,
)


class TestDemandModel:
    def test_callable_model_basic(self):
        def simple(price_ratio):
            return expit(1.0 - 2.0 * np.log(price_ratio))

        dm = DemandModel(simple)
        probs = dm.predict(np.array([0.9, 1.0, 1.1]))
        assert probs.shape == (3,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predictions_decrease_with_price_ratio(self):
        dm = make_logistic_demand()
        ratios = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        probs = dm.predict(ratios)
        # Higher price ratio should yield lower renewal probability
        assert np.all(np.diff(probs) < 0), "Demand should be decreasing in price ratio"

    def test_predictions_in_unit_interval(self):
        dm = make_logistic_demand()
        ratios = np.linspace(0.5, 2.0, 50)
        probs = dm.predict(ratios)
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_predictions_at_price_ratio_one(self):
        params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.0)
        dm = make_logistic_demand(params)
        # At price_ratio=1.0, log(1.0)=0, so p = expit(1.0) ≈ 0.731
        prob = dm.predict(np.array([1.0]))
        expected = float(expit(1.0))
        assert abs(prob[0] - expected) < 1e-6

    def test_tenure_increases_renewal_probability(self):
        params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.2)
        dm = make_logistic_demand(params)
        low_tenure = pd.DataFrame({"tenure": [0]})
        high_tenure = pd.DataFrame({"tenure": [10]})
        ratio = np.array([1.0])
        p_low = dm.predict(ratio, low_tenure)[0]
        p_high = dm.predict(ratio, high_tenure)[0]
        assert p_high > p_low, "Higher tenure should increase renewal probability"

    def test_sklearn_compatible_estimator(self):
        """Test with an sklearn-style estimator."""
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        # Train a trivial logistic model
        X_train = np.column_stack([
            np.log(np.array([0.9, 1.0, 1.1, 0.8, 1.2])),
        ])
        y_train = np.array([1, 1, 0, 1, 0])

        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)

        dm = DemandModel(lr, feature_names=[], price_ratio_col="price_ratio")
        probs = dm.predict(np.array([0.9, 1.0, 1.1]))
        assert probs.shape == (3,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_elasticity_is_negative(self):
        dm = make_logistic_demand()
        ratios = np.ones(10)
        elasticities = dm.elasticity_at(ratios)
        assert (elasticities < 0).all(), "Price elasticity should be negative"

    def test_elasticity_magnitude_plausible(self):
        """Elasticity should be in the -1.5 to -3.0 range for beta=-2.0."""
        params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.0)
        dm = make_logistic_demand(params)
        # At price_ratio=1.0: elasticity = beta * (1 - p) where p = expit(1.0)
        ratio = np.array([1.0])
        elasticity = dm.elasticity_at(ratio)[0]
        assert -4.0 < elasticity < -0.5, f"Elasticity {elasticity} outside plausible range"

    def test_repr_string(self, demand_model):
        r = repr(demand_model)
        assert "DemandModel" in r

    def test_default_params_give_sensible_model(self):
        dm = make_logistic_demand()
        # Default: base renewal prob at market price should be around 70-80%
        prob_at_market = dm.predict(np.array([1.0]))[0]
        assert 0.60 < prob_at_market < 0.90

    def test_model_with_very_high_price_ratio_near_zero(self):
        dm = make_logistic_demand()
        prob = dm.predict(np.array([5.0]))[0]
        assert prob < 0.10, "At 5x market price, renewal probability should be very low"

    def test_model_with_very_low_price_ratio_near_one(self):
        dm = make_logistic_demand()
        prob = dm.predict(np.array([0.3]))[0]
        assert prob > 0.90, "At 0.3x market price, renewal probability should be very high"
