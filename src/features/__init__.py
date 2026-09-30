"""Stage 3: Dual Feature Engineering Router.

Bifurcates predictor generation into:
  - Stateless Pipeline: Pure backward rolling operators enforced by a Universal Shift-Lock.
  - Stateful Pipeline: Distribution-dependent models fit strictly on training partitions
    and frozen prior to out-of-sample inference.
"""

from src.features.shift_lock import GlobalShiftLock
from src.features.stateless.kinematics import KinematicDerivativeEngine
from src.features.stateless.shadow_geometry import ShadowGeometryEngine
from src.features.stateless.normalized_flow import NormalizedVolumeFlowEngine
from src.features.stateless.illiquidity import AmihudIlliquidityEngine
from src.features.stateful.fracdiff import OptimalFractionalDifferentiator
from src.features.stateful.gmm_regime import GMMRegimeFilter

__all__ = [
    "GlobalShiftLock",
    "KinematicDerivativeEngine",
    "ShadowGeometryEngine",
    "NormalizedVolumeFlowEngine",
    "AmihudIlliquidityEngine",
    "OptimalFractionalDifferentiator",
    "GMMRegimeFilter",
]