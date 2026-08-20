import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .component import Component
    from .datapoint import CosmoSacMixtureDatapoint
    from .mixture import Mixture
    from .model import CosmoSac2002Model, CosmoSac2010Model, Model

__all__ = [
    "Component",
    "CosmoSac2002Model",
    "CosmoSac2010Model",
    "CosmoSacMixtureDatapoint",
    "Mixture",
    "Model",
]

# CosmoSacMixtureDatapoint pulls in cosmolayer.cosmodata, which imports torch;
# import it (and every other attribute here) lazily so that `import
# cosmolayer.cosmosac` alone -- or accessing only e.g. Component or Mixture --
# stays cheap.
_LAZY_ATTRS = {
    "Component": ".component",
    "CosmoSacMixtureDatapoint": ".datapoint",
    "Mixture": ".mixture",
    "CosmoSac2002Model": ".model",
    "CosmoSac2010Model": ".model",
    "Model": ".model",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
