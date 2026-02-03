"""
Differentiable COSMO-Type Activity Coefficient Layer
"""

from . import cosmosac
from ._version import __version__
from .cosmolayer import CosmoLayer
from .cosmospace import CosmoSpace

__all__ = [
    "__version__",
    "CosmoLayer",
    "CosmoSpace",
    "cosmosac",
]
