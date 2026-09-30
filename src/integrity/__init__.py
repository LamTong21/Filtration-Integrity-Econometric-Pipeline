"""Stage 1: Data Integrity and Topology Engine.

This module provides institutional data sanity checks, candle geometry audits,
forward-compounded total return synthesis (eliminating backward adjustment leakage),
and spatially decayed microstructure metrics.
"""

from src.integrity.audit import CandleGeometryAuditor
from src.integrity.return_index import ForwardReturnSynthesizer
from src.integrity.microstructure import MicrostructureTopologyEngine

__all__ = [
    "CandleGeometryAuditor",
    "ForwardReturnSynthesizer",
    "MicrostructureTopologyEngine",
]