"""Stage 2: Label Discipline and Quarantine Engine.

This module provides causal dynamic volatility estimation, asymmetric triple barrier labeling,
and the strict Quarantine Projection Operator Q to isolate target-generating artifacts from predictors.
"""

from src.labeling.volatility import CausalVolatilityEstimator
from src.labeling.triple_barrier import DynamicTripleBarrierLabeler
from src.labeling.quarantine import QuarantineOperator

__all__ = [
    "CausalVolatilityEstimator",
    "DynamicTripleBarrierLabeler",
    "QuarantineOperator",
]