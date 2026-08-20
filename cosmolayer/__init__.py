"""
Differentiable COSMO-Type Activity Coefficient Layer
"""

import importlib
from typing import TYPE_CHECKING

from . import store
from ._version import __version__

if TYPE_CHECKING:
    from . import cosmosac
    from .cosmodata import (
        MixtureDatapoint,
        MixtureInferenceDataset,
        MixtureTrainingDataset,
    )
    from .cosmolayer import CosmoLayer
    from .cosmolightning import LogGammaLightningModule
    from .cosmosolver import CosmoSolver

__all__ = [
    "__version__",
    "cosmosac",
    "store",
    "CosmoLayer",
    "LogGammaLightningModule",
    "CosmoSolver",
    "MixtureDatapoint",
    "MixtureInferenceDataset",
    "MixtureTrainingDataset",
]

# Submodules and attributes that pull in heavy optional dependencies (torch,
# lightning, rdkit-adjacent chains) are imported lazily, on first access,
# rather than eagerly at package-import time. This keeps `import cosmolayer`
# (and cheap subpackages like `cosmolayer.store`) fast.
_LAZY_SUBMODULES = frozenset({"cosmosac"})
_LAZY_ATTRS = {
    "CosmoLayer": ".cosmolayer",
    "LogGammaLightningModule": ".cosmolightning",
    "CosmoSolver": ".cosmosolver",
    "MixtureDatapoint": ".cosmodata",
    "MixtureInferenceDataset": ".cosmodata",
    "MixtureTrainingDataset": ".cosmodata",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | _LAZY_SUBMODULES)
