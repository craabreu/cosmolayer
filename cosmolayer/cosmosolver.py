"""Compatibility shim.

Implementation lives in ``cosmolayer.cosmolayer.cosmosolver``.
"""

from .cosmolayer.cosmosolver import (
    NEWTON_RESIDUAL_TOLERANCE,
    NEWTON_STEP_TOLERANCE,
    CosmoSolver,
)

__all__ = [
    "CosmoSolver",
    "NEWTON_RESIDUAL_TOLERANCE",
    "NEWTON_STEP_TOLERANCE",
]
