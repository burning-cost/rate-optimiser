"""
rate-optimiser: Constrained rate change optimiser for UK personal lines insurance.

The insurance analogue of Markowitz portfolio optimisation. Takes policy-level
technical prices and demand model outputs; finds factor adjustments that meet
loss ratio and volume targets; traces the efficient frontier of achievable outcomes.

Designed for pricing actuaries working in UK personal lines who want an auditable,
extensible alternative to the black-box solvers in commercial tools.
"""

from rate_optimiser.data import FactorStructure, PolicyData
from rate_optimiser.demand import DemandModel
from rate_optimiser.optimiser import OptimiserResult, RateChangeOptimiser
from rate_optimiser.frontier import EfficientFrontier
from rate_optimiser.constraints import (
    ENBPConstraint,
    FactorBoundsConstraint,
    LossRatioConstraint,
    VolumeConstraint,
)

__version__ = "0.2.0"
__all__ = [
    "PolicyData",
    "FactorStructure",
    "DemandModel",
    "RateChangeOptimiser",
    "OptimiserResult",
    "EfficientFrontier",
    "ENBPConstraint",
    "FactorBoundsConstraint",
    "LossRatioConstraint",
    "VolumeConstraint",
]
