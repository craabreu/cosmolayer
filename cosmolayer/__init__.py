"""
Differentiable COSMO-Type Activity Coefficient Layer
"""

from ._version import __version__
from .sac import Component, create_lin_sandler_matrix

__all__ = [
    "__version__",
    "Component",
    "create_lin_sandler_matrix",
]
