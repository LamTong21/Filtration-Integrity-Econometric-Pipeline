"""Stage 4: Moving-Block Consensus Selection (MBCS) Engine.

Combines fast linear orthogonalization (VIF), non-linear hierarchical medoid clustering,
and dual causal lead-lag screening across sliding temporal blocks to extract robust signals.
"""

from src.selection.fast_vif import FastVIFPruner
from src.selection.hrp_medoids import HRPMedoidExtractor
from src.selection.causality import DualCausalScreen
from src.selection.mbcs_engine import MovingBlockConsensusSelector

__all__ = [
    "FastVIFPruner",
    "HRPMedoidExtractor",
    "DualCausalScreen",
    "MovingBlockConsensusSelector",
]