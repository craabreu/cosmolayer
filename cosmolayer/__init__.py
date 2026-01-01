"""
Differentiable COSMO-Type Activity Coefficient Layer
"""

from ._version import __version__
from .sac import Component, LinSandlerMatrix

__all__ = ["__version__", "Component", "LinSandlerMatrix"]
