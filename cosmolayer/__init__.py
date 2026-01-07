"""
Differentiable COSMO-Type Activity Coefficient Layer
"""

from ._version import __version__
from .cosmospace import CosmoSpace
from .sac import (
    Component,
    create_cosmo_sac_2002_matrix,
    create_cosmo_sac_2010_matrices,
)

__all__ = [
    "__version__",
    "Component",
    "CosmoSpace",
    "create_cosmo_sac_2002_matrix",
    "create_cosmo_sac_2010_matrices",
]
