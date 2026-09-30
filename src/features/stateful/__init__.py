"""Stateful Feature Generation Submodule.

Distribution-dependent components fit strictly on historical training partitions
and frozen prior to out-of-sample inference.
"""

from src.features.stateful.fracdiff import OptimalFractionalDifferentiator
from src.features.stateful.gmm_regime import GMMRegimeFilter

__all__ = [
    "OptimalFractionalDifferentiator",
    "GMMRegimeFilter",
]