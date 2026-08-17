"""Compatibility shim.

Implementation lives in ``cosmolayer.cosmolayer.cosmodata``.
"""

from .cosmolayer.cosmodata import (
    InputsType,
    MixtureDatapoint,
    MixtureInferenceDataset,
    MixtureTrainingDataset,
    NumpyArray1D,
    NumpyArray2D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
)

__all__ = [
    "InputsType",
    "MixtureDatapoint",
    "MixtureInferenceDataset",
    "MixtureTrainingDataset",
    "NumpyArray1D",
    "NumpyArray2D",
    "Tensor0D",
    "Tensor1D",
    "Tensor2D",
]
