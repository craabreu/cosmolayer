"""
.. module:: cosmolayer.cosmodata
   :synopsis: Data tensors for COSMO-SAC calculations.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import torch

NumpyArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.number]]
NumpyArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.number]]

Tensor0D: TypeAlias = torch.Tensor
Tensor1D: TypeAlias = torch.Tensor
Tensor2D: TypeAlias = torch.Tensor

InputsType: TypeAlias = tuple[Tensor0D, Tensor1D, Tensor1D, Tensor1D, Tensor2D]
TargetsType: TypeAlias = tuple[Tensor1D, Tensor2D]


@dataclass
class MixtureDatapoint:
    """Base dataclass for a mixture datapoint.

    Stores physical inputs (temperature, mole fractions, areas, volumes, and
    segment-type probabilities) and optional training targets. Shape metadata
    is computed and validated automatically on construction.

    Parameters
    ----------
    temperature
        Temperature.
    mole_fractions
        Mole fractions.
        Shape: ``(num_components,)``.
    areas
        Segment surface areas per component.
        Shape: ``(num_components,)``.
    volumes
        Molar volumes per component.
        Shape: ``(num_components,)``.
    probabilities : NumpyArray2D
        Sigma-profile probabilities.
        Shape: ``(num_components, num_segment_types)``.
    mixture_targets : NumpyArray1D
        Mixture-level training targets.
        Shape: ``(num_mixture_targets,)``.
    per_component_targets : NumpyArray2D
        Per-component training targets.
        Shape: ``(num_per_component_targets, num_components)``.

    Attributes
    ----------
    num_components : int
        Number of components.
    num_segment_types : int
        Number of segment-type probabilities.
    num_mixture_targets : int
        Number of mixture-level targets.
    num_per_component_targets : int
        Number of per-component target types.

    Raises
    ------
    ValueError
        If array shapes are inconsistent.
    """

    temperature: float
    mole_fractions: NumpyArray1D = field(repr=False)
    areas: NumpyArray1D = field(repr=False)
    volumes: NumpyArray1D = field(repr=False)
    probabilities: NumpyArray2D = field(repr=False)
    mixture_targets: NumpyArray1D = field(repr=False)
    per_component_targets: NumpyArray2D = field(repr=False)
    num_components: int = field(init=False)
    num_segment_types: int = field(init=False)
    num_mixture_targets: int = field(init=False)
    num_per_component_targets: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate array shapes and freeze stored numpy arrays.

        Raises
        ------
        ValueError
            If any stored array has an incompatible shape.
        """
        try:
            assert self.probabilities.ndim == 2  # noqa: PLR2004
            self.num_components, self.num_segment_types = self.probabilities.shape
            assert self.mole_fractions.shape == (self.num_components,)
            assert self.areas.shape == (self.num_components,)
            assert self.volumes.shape == (self.num_components,)
            assert self.mixture_targets.ndim == 1
            assert self.per_component_targets.ndim == 2  # noqa: PLR2004
            assert self.per_component_targets.shape[1] == self.num_components
        except AssertionError as e:
            raise ValueError("Invalid array shapes") from e
        self.num_mixture_targets = len(self.mixture_targets)
        self.num_per_component_targets = self.per_component_targets.shape[0]
        for array in (
            self.mole_fractions,
            self.areas,
            self.volumes,
            self.probabilities,
            self.mixture_targets,
            self.per_component_targets,
        ):
            array.flags.writeable = False

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the structural shape metadata for the datapoint.

        Returns
        -------
        tuple[int, int, int, int]
            Tuple containing the number of components, segment types,
            mixture-level targets, and per-component target types.
        """
        return (
            self.num_components,
            self.num_segment_types,
            self.num_mixture_targets,
            self.num_per_component_targets,
        )

    def get_inputs(self, dtype: torch.dtype) -> InputsType:
        """Convert physical inputs to torch tensors.

        Parameters
        ----------
        dtype : torch.dtype
            Data type used for all returned tensors.

        Returns
        -------
        InputsType
            Temperature, mole fractions, areas, volumes, and probabilities
            as torch tensors.
        """
        return (
            torch.tensor(self.temperature, dtype=dtype),
            torch.tensor(self.mole_fractions, dtype=dtype),
            torch.tensor(self.areas, dtype=dtype),
            torch.tensor(self.volumes, dtype=dtype),
            torch.tensor(self.probabilities, dtype=dtype),
        )

    def get_targets(self, dtype: torch.dtype) -> TargetsType:
        """Convert target arrays to torch tensors.

        Parameters
        ----------
        dtype : torch.dtype
            Data type used for all returned tensors.

        Returns
        -------
        TargetsType
            Mixture-level and per-component targets as torch tensors.
        """
        return (
            torch.tensor(self.mixture_targets, dtype=dtype),
            torch.tensor(self.per_component_targets, dtype=dtype),
        )


class MixtureDataset(torch.utils.data.Dataset[tuple[InputsType, TargetsType]]):
    """Torch dataset wrapper for shape-compatible mixture datapoints.

    Parameters
    ----------
    mixtures : Sequence[MixtureDatapoint]
        Datapoints to expose through the dataset interface. All datapoints
        must share the same structural shape.
    dtype : torch.dtype
        Data type used when converting datapoints to tensors.

    Raises
    ------
    ValueError
        If ``mixtures`` is empty or contains incompatible datapoint shapes.


    Examples
    --------
    >>> from cosmolayer.cosmodata import MixtureDataset, MixtureDatapoint
    >>> from cosmolayer.cosmosac import CosmoSac2002Model
    >>> from cosmolayer.cosmosac.datapoint import CosmoSacMixtureDatapoint
    >>> from importlib.resources import files
    >>> data = files("cosmolayer.data")
    >>> cosmo_files = [data / "C=C(N)O.cosmo", data / "NCCO.cosmo"]
    >>> mole_fractions = [0.5, 0.5]
    >>> temperature = 298.15
    >>> mixture_targets = [1.2]
    >>> component_targets = [[0.9, 1.1]]
    >>> dp = CosmoSacMixtureDatapoint(
    ...     cosmo_files,
    ...     mole_fractions,
    ...     temperature,
    ...     mixture_targets,
    ...     component_targets,
    ...     CosmoSac2002Model,
    ... )
    >>> dataset = MixtureDataset([dp], dtype=torch.float32)
    >>> len(dataset)
    1
    >>> inputs, targets = dataset[0]
    >>> len(inputs)
    5
    >>> len(targets)
    2
    """

    def __init__(
        self,
        mixtures: Sequence[MixtureDatapoint],
        dtype: torch.dtype,
    ):
        if len(mixtures) == 0:
            raise ValueError("MixtureDataset must contain at least one mixture")
        shape = mixtures[0].shape
        if any(mixture.shape != shape for mixture in mixtures[1:]):
            raise ValueError("All mixtures must have the same shape")
        self._mixtures = mixtures
        self._dtype = dtype

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return len(self._mixtures)

    def __getitem__(self, index: int) -> tuple[InputsType, TargetsType]:
        """Return one datapoint as input and target tensor tuples.

        Parameters
        ----------
        index : int
            Position of the datapoint in the dataset.

        Returns
        -------
        tuple[InputsType, TargetsType]
            Input tensors and target tensors for the selected datapoint.
        """
        mixture = self._mixtures[index]
        return mixture.get_inputs(self._dtype), mixture.get_targets(self._dtype)
