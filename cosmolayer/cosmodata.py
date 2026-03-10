"""
.. module:: cosmolayer.cosmodata
   :synopsis: Data tensors for COSMO-SAC calculations.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

import numpy as np
import torch

NumpyArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.number]]
NumpyArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.number]]

Tensor0D: TypeAlias = torch.Tensor
Tensor1D: TypeAlias = torch.Tensor
Tensor2D: TypeAlias = torch.Tensor

InputsType: TypeAlias = tuple[Tensor0D, Tensor1D, Tensor1D, Tensor1D, Tensor2D]

if TYPE_CHECKING:
    _DatasetItemT = TypeVar("_DatasetItemT")

    class _DatasetBase(Generic[_DatasetItemT]):
        """Mypy-only base to avoid torch stub variability in CI."""

else:
    _DatasetBase = torch.utils.data.Dataset


@dataclass
class MixtureDatapoint:
    """Base dataclass for a mixture datapoint.

    Stores physical inputs (temperature, mole fractions, areas, volumes, and
    segment-type probabilities) and optional training targets. Shape metadata
    is computed and validated automatically on construction.

    Parameters
    ----------
    temperature: float
        Temperature.
    mole_fractions: NumpyArray1D
        Mole fractions.
        Shape: ``(num_components,)``.
    areas: NumpyArray1D
        Segment surface areas per component.
        Shape: ``(num_components,)``.
    volumes: NumpyArray1D
        Molar volumes per component.
        Shape: ``(num_components,)``.
    probabilities : NumpyArray2D
        Sigma-profile probabilities.
        Shape: ``(num_components, num_segment_types)``.
    targets : NumpyArray1D
        Training targets.
        Shape: ``(num_targets,)``.

    Attributes
    ----------
    num_components : int
        Number of components.
    num_segment_types : int
        Number of segment-type probabilities.
    num_targets : int
        Number of training targets.

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
    targets: NumpyArray1D = field(repr=False)
    num_components: int = field(init=False)
    num_segment_types: int = field(init=False)
    num_targets: int = field(init=False)

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
            assert self.targets.ndim == 1
        except AssertionError as e:
            raise ValueError("Invalid array shapes") from e
        self.num_targets = len(self.targets)
        for array in (
            self.mole_fractions,
            self.areas,
            self.volumes,
            self.probabilities,
            self.targets,
        ):
            array.flags.writeable = False

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the structural shape metadata for the datapoint.

        Returns
        -------
        tuple[int, int, int]
            Tuple containing the number of components, segment types,
            and number of training targets.
        """
        return (
            self.num_components,
            self.num_segment_types,
            self.num_targets,
        )

    def get_inputs(self, dtype: torch.dtype = torch.float64) -> InputsType:
        """Convert physical inputs to torch tensors.

        Parameters
        ----------
        dtype : torch.dtype
            Data type used for all returned tensors. Default is ``torch.float64``.

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

    def get_targets(self, dtype: torch.dtype = torch.float64) -> Tensor1D:
        """Convert target arrays to torch tensors.

        Parameters
        ----------
        dtype : torch.dtype
            Data type used for all returned tensors. Default is ``torch.float64``.

        Returns
        -------
        Tensor1D
            Training targets as torch tensors.
        """
        return torch.tensor(self.targets, dtype=dtype)


class MixtureInferenceDataset(_DatasetBase[InputsType]):
    """Torch dataset wrapper for shape-compatible mixture datapoints in inference.

    Parameters
    ----------
    mixtures : Sequence[MixtureDatapoint]
        Datapoints to expose through the dataset interface. All datapoints
        must share the same input shape (number of components and segment types).
    dtype : torch.dtype
        Data type used when converting datapoints to tensors.

    Raises
    ------
    ValueError
        If ``mixtures`` is empty or contains incompatible input shapes.

    Examples
    --------
    >>> from cosmolayer.cosmodata import MixtureInferenceDataset, MixtureDatapoint
    >>> dp = MixtureDatapoint(
    ...     temperature=298.15,
    ...     mole_fractions=np.array([0.5, 0.5]),
    ...     areas=np.array([1.0, 2.0]),
    ...     volumes=np.array([1.0, 2.0]),
    ...     probabilities=np.array([[0.5, 0.5], [0.4, 0.6]]),
    ...     targets=np.array([]),
    ... )
    >>> dataset = MixtureInferenceDataset([dp], dtype=torch.float32)
    >>> inputs = dataset[0]
    >>> len(inputs)
    5
    """

    def __init__(
        self,
        mixtures: Sequence[MixtureDatapoint],
        dtype: torch.dtype,
    ):
        if len(mixtures) == 0:
            raise ValueError(
                "MixtureInferenceDataset must contain at least one mixture"
            )
        input_shape = mixtures[0].shape[:2]
        if any(mixture.shape[:2] != input_shape for mixture in mixtures[1:]):
            raise ValueError("All mixtures must have the same input shape")
        self._mixtures = mixtures
        self._dtype = dtype

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return len(self._mixtures)

    def __getitem__(self, index: int) -> InputsType:
        """Return one datapoint as input tensors.

        Parameters
        ----------
        index : int
            Position of the datapoint in the dataset.

        Returns
        -------
        InputsType
            Input tensors for the selected datapoint.
        """
        return self._mixtures[index].get_inputs(self._dtype)


class MixtureTrainingDataset(_DatasetBase[tuple[InputsType, Tensor1D]]):
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
    >>> from cosmolayer.cosmodata import MixtureTrainingDataset, MixtureDatapoint
    >>> from cosmolayer.cosmosac import CosmoSac2002Model
    >>> from cosmolayer.cosmosac.datapoint import CosmoSacMixtureDatapoint
    >>> from importlib.resources import files
    >>> data = files("cosmolayer.data")
    >>> cosmo_files = [data / "C=C(N)O.cosmo", data / "NCCO.cosmo"]
    >>> mole_fractions = [0.5, 0.5]
    >>> temperature = 298.15
    >>> targets = [1.2]
    >>> dp = CosmoSacMixtureDatapoint(
    ...     cosmo_files,
    ...     mole_fractions,
    ...     temperature,
    ...     targets,
    ...     CosmoSac2002Model,
    ... )
    >>> dataset = MixtureTrainingDataset([dp], dtype=torch.float32)
    >>> len(dataset)
    1
    >>> inputs, targets = dataset[0]
    >>> len(inputs)
    5
    >>> len(targets)
    1
    """

    def __init__(
        self,
        mixtures: Sequence[MixtureDatapoint],
        dtype: torch.dtype = torch.float64,
    ):
        if len(mixtures) == 0:
            raise ValueError("MixtureTrainingDataset must contain at least one mixture")
        shape = mixtures[0].shape
        if any(mixture.shape != shape for mixture in mixtures[1:]):
            raise ValueError("All mixtures must have the same shape")
        self._mixtures = mixtures
        self._dtype = dtype

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset."""
        return len(self._mixtures)

    def __getitem__(self, index: int) -> tuple[InputsType, Tensor1D]:
        """Return one datapoint as input and target tensor tuples.

        Parameters
        ----------
        index : int
            Position of the datapoint in the dataset.

        Returns
        -------
        tuple[InputsType, Tensor1D]
            Input tensors and target tensors for the selected datapoint.
        """
        mixture = self._mixtures[index]
        return mixture.get_inputs(self._dtype), mixture.get_targets(self._dtype)

    def to_inference_dataset(self) -> MixtureInferenceDataset:
        """Convert the training dataset to an inference dataset.

        Returns
        -------
        MixtureInferenceDataset
            An inference dataset with the same mixtures and dtype.
        """
        return MixtureInferenceDataset(self._mixtures, self._dtype)
