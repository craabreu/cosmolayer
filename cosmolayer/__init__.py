"""
Differentiable COSMO-Type Activity Coefficient Layer
"""

from ._version import __version__
from .sac import Component, create_cosmo_sac_2002_matrix

__all__ = [
    "__version__",
    "Component",
    "create_cosmo_sac_2002_matrix",
]
