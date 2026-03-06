"""
Demand model wrapper.

The DemandModel wraps any callable or sklearn-compatible estimator that
predicts renewal/conversion probability as a function of the price ratio
(charged premium / market premium) and optional policy features.

Design principle: the library does not fit demand models. That is the
caller's job, using whatever data and modelling approach they have available.
This wrapper provides a standard interface so the optimiser can query
probabilities given any price ratio, without caring about the model internals.

The logistic form is the industry default (see Guven & McPhail 2013, CAS):

    logit(p_i) = α + β × log(π_i / π_market_i) + X_i @ γ

where β is the price semi-elasticity (typically negative, around -1.5 to -3.0
for UK motor PCW). This class supports both this parametric form and arbitrary
callables, so teams with XGBoost or neural-net demand models can plug them in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class SklearnEstimator(Protocol):
    """Protocol for sklearn-compatible classifiers."""

    def predict_proba(self, X) -> np.ndarray: ...


class DemandModel:
    """
    Wraps a renewal/conversion probability model for use in rate optimisation.

    The model is called during optimisation to re-evaluate demand probabilities
    at each candidate set of factor adjustments. It must be fast enough for
    repeated evaluation (hundreds of calls during SLSQP iteration).

    Parameters
    ----------
    callable_or_estimator : callable or sklearn estimator
        Either:
        - A callable with signature ``f(price_ratio: np.ndarray, **kwargs) -> np.ndarray``
          returning renewal probabilities in [0, 1], one per policy.
        - An sklearn-compatible estimator with a ``predict_proba`` method. In
          that case, ``feature_names`` must be supplied and the DataFrame passed
          to ``predict`` must contain those columns plus ``price_ratio``.
    feature_names : list of str, optional
        Feature columns from the policy DataFrame to pass to an sklearn estimator
        alongside the price ratio. Ignored if a raw callable is supplied.
    price_ratio_col : str
        Name to use for the price ratio feature when constructing the feature
        matrix for sklearn estimators. Default is ``"price_ratio"``.

    Examples
    --------
    Parametric logistic model (simplest case):

    >>> from scipy.special import expit
    >>> def logistic_demand(price_ratio, alpha=0.5, beta=-2.0):
    ...     return expit(alpha + beta * np.log(price_ratio))
    >>> demand = DemandModel(logistic_demand)

    Sklearn estimator:

    >>> demand = DemandModel(
    ...     my_logistic_regression,
    ...     feature_names=["age", "tenure", "ncb_years"],
    ... )
    """

    def __init__(
        self,
        callable_or_estimator,
        feature_names: Optional[list[str]] = None,
        price_ratio_col: str = "price_ratio",
    ) -> None:
        self._model = callable_or_estimator
        self._feature_names = feature_names or []
        self._price_ratio_col = price_ratio_col
        self._is_sklearn = isinstance(callable_or_estimator, SklearnEstimator)

    def predict(
        self,
        price_ratio: np.ndarray,
        policy_features: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Return renewal/conversion probabilities at the given price ratios.

        Parameters
        ----------
        price_ratio : np.ndarray
            Shape (n_policies,). Ratio of the candidate premium to market premium.
            Values > 1 mean the insurer is above market; < 1 below market.
        policy_features : pd.DataFrame, optional
            Policy-level features required by the model. Must be supplied when
            using an sklearn estimator with ``feature_names``. Not required for
            simple callables.

        Returns
        -------
        np.ndarray
            Shape (n_policies,). Predicted probabilities in [0, 1].
        """
        price_ratio = np.asarray(price_ratio, dtype=float)

        if self._is_sklearn:
            return self._predict_sklearn(price_ratio, policy_features)
        else:
            # Raw callable
            if policy_features is not None and self._feature_names:
                return self._model(price_ratio, policy_features[self._feature_names])
            return self._model(price_ratio)

    def _predict_sklearn(
        self,
        price_ratio: np.ndarray,
        policy_features: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """Build feature matrix and call predict_proba."""
        if policy_features is None and self._feature_names:
            raise ValueError(
                "policy_features must be supplied when using an sklearn estimator "
                f"with feature_names={self._feature_names}"
            )
        if policy_features is not None and self._feature_names:
            X = policy_features[self._feature_names].copy()
            X[self._price_ratio_col] = price_ratio
        else:
            X = pd.DataFrame({self._price_ratio_col: price_ratio})
        proba = self._model.predict_proba(X)
        # sklearn returns shape (n, 2); take column 1 (positive class = renews)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def elasticity_at(
        self,
        price_ratio: np.ndarray,
        policy_features: Optional[pd.DataFrame] = None,
        delta: float = 0.01,
    ) -> np.ndarray:
        """
        Numerical price elasticity: d(log p) / d(log price_ratio).

        Computed via central finite differences. Useful for diagnostics and
        for verifying the demand model is plausible before running optimisation.

        Parameters
        ----------
        price_ratio : np.ndarray
            Shape (n_policies,). Reference price ratios.
        policy_features : pd.DataFrame, optional
            Policy features (passed through to predict).
        delta : float
            Relative step size for finite differences. Default 0.01 (1%).

        Returns
        -------
        np.ndarray
            Shape (n_policies,). Elasticity values. Expect negative values
            (demand falls as price rises). UK motor PCW: typically -1.5 to -3.0.
        """
        p_up = self.predict(price_ratio * (1 + delta), policy_features)
        p_down = self.predict(price_ratio * (1 - delta), policy_features)
        p_base = self.predict(price_ratio, policy_features)
        # Avoid division by zero
        p_base_safe = np.where(p_base > 1e-10, p_base, 1e-10)
        dp = (p_up - p_down) / (2 * delta * price_ratio)
        return dp * price_ratio / p_base_safe

    def __repr__(self) -> str:
        model_type = type(self._model).__name__
        return (
            f"DemandModel(model_type={model_type}, "
            f"feature_names={self._feature_names})"
        )


@dataclass
class LogisticDemandParams:
    """
    Parameters for the canonical logistic demand model used in testing and examples.

    logit(p_i) = intercept + price_coef * log(price_ratio_i) + tenure_coef * tenure_i

    Parameters
    ----------
    intercept : float
        Log-odds of renewal at market price (price_ratio=1) for zero tenure.
        A value of 1.0 implies a ~73% base renewal probability.
    price_coef : float
        Semi-elasticity. Negative. Typical range: -1.5 to -3.0 for UK motor PCW.
    tenure_coef : float
        Effect of each additional year of tenure on log-odds of renewal.
        Positive (longer-tenured customers are stickier).
    """

    intercept: float = 1.0
    price_coef: float = -2.0
    tenure_coef: float = 0.05


def make_logistic_demand(params: Optional[LogisticDemandParams] = None) -> DemandModel:
    """
    Construct a logistic demand model with the given parameters.

    This is the industry-standard form. For production use, fit the parameters
    from price-test data or natural experiments in your quote/bind records.

    Parameters
    ----------
    params : LogisticDemandParams, optional
        Model parameters. Uses defaults if not supplied.

    Returns
    -------
    DemandModel
    """
    if params is None:
        params = LogisticDemandParams()

    from scipy.special import expit

    intercept = params.intercept
    price_coef = params.price_coef
    tenure_coef = params.tenure_coef

    def _logistic(price_ratio: np.ndarray, features: Optional[pd.DataFrame] = None) -> np.ndarray:
        log_ratio = np.log(np.clip(price_ratio, 1e-6, 10.0))
        linear = intercept + price_coef * log_ratio
        if features is not None and "tenure" in features.columns:
            linear = linear + tenure_coef * features["tenure"].values
        return expit(linear)

    return DemandModel(_logistic, feature_names=["tenure"])
