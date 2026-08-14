"""Layer implementation lives in the cosmolayer.cosmolayer subpackage.

Legacy import paths remain supported and resolve to the same objects.
"""

from cosmolayer import CosmoLayer as TopLevelCosmoLayer
from cosmolayer.cosmodata import MixtureDatapoint
from cosmolayer.cosmolayer import CosmoLayer
from cosmolayer.cosmolayer.cosmodata import MixtureDatapoint as InnerDatapoint
from cosmolayer.cosmolayer.cosmolightning import (
    LogGammaLightningModule as InnerLightning,
)
from cosmolayer.cosmolayer.cosmosolver import CosmoSolver as InnerSolver
from cosmolayer.cosmolayer.layer import CosmoLayer as LayerImpl
from cosmolayer.cosmolightning import LogGammaLightningModule
from cosmolayer.cosmosolver import CosmoSolver


def test_implementation_lives_in_layer_subpackage() -> None:
    assert LayerImpl.__module__ == "cosmolayer.cosmolayer.layer"
    assert InnerSolver.__module__ == "cosmolayer.cosmolayer.cosmosolver"
    assert InnerDatapoint.__module__ == "cosmolayer.cosmolayer.cosmodata"
    assert InnerLightning.__module__ == "cosmolayer.cosmolayer.cosmolightning"


def test_legacy_module_paths_reexport_subpackage_objects() -> None:
    assert CosmoLayer is LayerImpl
    assert TopLevelCosmoLayer is LayerImpl
    assert CosmoSolver is InnerSolver
    assert MixtureDatapoint is InnerDatapoint
    assert LogGammaLightningModule is InnerLightning
