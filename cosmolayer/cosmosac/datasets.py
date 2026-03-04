"""
.. module:: cosmolayer.cosmosac.datasets
   :synopsis: Datasets for COSMO-SAC calculations.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import functools
import itertools
import os
import pathlib
from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .model import CosmoSac2010Model, Model

OutputType: TypeAlias = tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


@functools.cache
def _compute_component_properties(
    model: Model, file_path: str
) -> tuple[float, float, NDArray[np.float64]]:
    """Return ``(area, volume, probabilities)`` for a component, with LRU caching.

    Parameters
    ----------
    model : Model
        COSMO-SAC model used to parse the file and derive properties.
    file_path : str
        Absolute or relative path to the ``.cosmo`` file.  Passed as a plain
        string so that :func:`functools.lru_cache` can use it as a hashable
        key.

    Returns
    -------
    tuple[float, float, NDArray[np.float64]]
        A 3-tuple ``(area, volume, probabilities)`` where *area* is in Å²,
        *volume* in Å³, and *probabilities* is the normalised sigma-profile
        array with shape ``(51,)`` for merged profiles or ``(3, 51)`` for
        split NHB/OH/OT profiles.
    """
    component = model.create_component(pathlib.Path(file_path).read_text())
    return (
        component.area,
        component.volume,
        component.probabilities,
    )


class CosmoFileDataset(Dataset[OutputType]):
    """Dataset of COSMO files for COSMO-SAC calculations.

    Each sample in the dataset corresponds to one row of *mixture_dataframe*
    and yields the temperature, mole fractions, areas, volumes, and
    segment-type probabilities as arrays suitable for :meth:`CosmoLayer.forward`.

    Component properties are derived by parsing the corresponding ``.cosmo``
    files and are cached across samples via :func:`functools.lru_cache`, so
    repeated access to the same component is essentially free.

    Parameters
    ----------
    mixture_dataframe : pd.DataFrame
        DataFrame containing the mixture data with columns for the component
        names, their mole fractions, and optionally the temperature.  Column
        names are determined by *component_formatter* and
        *mole_fraction_formatter*.
    component_to_file_mapping : Mapping[str, str | os.PathLike[str]]
        Dictionary mapping component names to paths to ``.cosmo`` files.
        Paths are resolved relative to *prefix* (if provided).
    model : Model
        COSMO-SAC model used to compute molecular properties from the files.
    prefix : str | os.PathLike[str], optional
        Directory prepended to every value in *component_to_file_mapping*.
        Useful when the mapping stores bare filenames.  Defaults to ``""``
        (no prefix).
    temperature : str | float, optional
        Column name or fixed value of the temperature in Kelvin.  When a
        string is given it must name an existing column in
        *mixture_dataframe*.  When a float is given it is used as the
        temperature for every sample.  Defaults to ``"temperature"``.
    component_formatter : Callable[[int], str], optional
        Callable that maps a 0-based integer to the column name for that
        component.  Defaults to ``"component_{}".format``.
    mole_fraction_formatter : Callable[[int], str], optional
        Callable that maps a 0-based integer to the column name for that
        component's mole fraction.  Defaults to ``"mole_fraction_{}".format``.

    Raises
    ------
    ValueError
        If no component columns are found, if the number of mole-fraction
        columns does not match the number of component columns, or if
        *temperature* is a string that does not name a column in
        *mixture_dataframe*.

    Examples
    --------
    Constructing a binary-mixture dataset with a fixed temperature:

    >>> from importlib.resources import files
    >>> import pandas as pd
    >>> from cosmolayer.cosmosac.model import CosmoSac2002Model
    >>> from cosmolayer.cosmosac.datasets import CosmoFileDataset
    >>> data = files("cosmolayer.data")
    >>> df = pd.DataFrame({
    ...     "component_0": ["water", "water"],
    ...     "mole_fraction_0": [0.3, 0.7],
    ...     "component_1": ["fluoromethane", "fluoromethane"],
    ...     "mole_fraction_1": [0.7, 0.3],
    ... })
    >>> mapping = {
    ...     "water": str(data / "O.cosmo"),
    ...     "fluoromethane": str(data / "CF.cosmo"),
    ... }
    >>> ds = CosmoFileDataset(df, mapping, CosmoSac2002Model, temperature=298.15)
    >>> len(ds)
    2
    """

    def __init__(  # noqa: PLR0913
        self,
        mixture_dataframe: pd.DataFrame,
        component_to_file_mapping: Mapping[str, str | os.PathLike[str]],
        model: Model = CosmoSac2010Model,
        prefix: str | os.PathLike[str] = "",
        temperature: str | float = "temperature",
        component_formatter: Callable[[int], str] = "component_{}".format,
        mole_fraction_formatter: Callable[[int], str] = "mole_fraction_{}".format,
    ) -> None:
        num_components = CosmoFileDataset._count_components(
            mixture_dataframe, component_formatter
        )
        if num_components == 0:
            raise ValueError("No components found in the mixture dataframe")
        num_mole_fractions = CosmoFileDataset._count_components(
            mixture_dataframe, mole_fraction_formatter
        )
        if num_mole_fractions != num_components:
            raise ValueError(
                "Number of mole fractions does not match number of components"
            )
        temperature_column_provided = isinstance(temperature, str)
        if temperature_column_provided and temperature not in mixture_dataframe.columns:
            raise ValueError("Temperature column not found in the mixture dataframe")

        self._prefix = pathlib.Path(prefix)
        self._model = model
        self._mixture_dataframe = mixture_dataframe
        self._component_to_file_mapping = component_to_file_mapping
        self._temperature_column = temperature if temperature_column_provided else None
        self._temperature_value = None if temperature_column_provided else temperature
        self._component_columns = [
            component_formatter(i) for i in range(num_components)
        ]
        self._mole_fraction_columns = [
            mole_fraction_formatter(i) for i in range(num_components)
        ]

    @property
    def mixture_dataframe(self) -> pd.DataFrame:
        """The mixture data as a read-only pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame that was passed to the constructor, with columns for
            component names, mole fractions, and optionally temperature.

        Examples
        --------
        >>> from importlib.resources import files
        >>> import pandas as pd
        >>> from cosmolayer.cosmosac.model import CosmoSac2002Model
        >>> from cosmolayer.cosmosac.datasets import CosmoFileDataset
        >>> data = files("cosmolayer.data")
        >>> df = pd.DataFrame({
        ...     "component_0": ["water"],
        ...     "mole_fraction_0": [0.4],
        ...     "component_1": ["fluoromethane"],
        ...     "mole_fraction_1": [0.6],
        ... })
        >>> mapping = {
        ...     "water": str(data / "O.cosmo"),
        ...     "fluoromethane": str(data / "CF.cosmo"),
        ... }
        >>> ds = CosmoFileDataset(df, mapping, CosmoSac2002Model, temperature=298.15)
        >>> ds.mixture_dataframe is df
        True
        """
        return self._mixture_dataframe

    def __len__(self) -> int:
        """Return the number of samples (rows) in the dataset.

        Returns
        -------
        int
            Number of rows in the underlying mixture DataFrame.

        Examples
        --------
        >>> from importlib.resources import files
        >>> import pandas as pd
        >>> from cosmolayer.cosmosac.model import CosmoSac2002Model
        >>> from cosmolayer.cosmosac.datasets import CosmoFileDataset
        >>> data = files("cosmolayer.data")
        >>> df = pd.DataFrame({
        ...     "component_0": ["water", "water", "water"],
        ...     "mole_fraction_0": [0.2, 0.5, 0.8],
        ...     "component_1": ["fluoromethane"] * 3,
        ...     "mole_fraction_1": [0.8, 0.5, 0.2],
        ... })
        >>> mapping = {
        ...     "water": str(data / "O.cosmo"),
        ...     "fluoromethane": str(data / "CF.cosmo"),
        ... }
        >>> ds = CosmoFileDataset(df, mapping, CosmoSac2002Model, temperature=298.15)
        >>> len(ds)
        3
        """
        return len(self.mixture_dataframe)

    def __getitem__(self, idx: int) -> OutputType:
        """Return the sample at position *idx*.

        Return values are suitable for passing to :meth:`CosmoLayer.forward`
        as ``(temp, fracs, areas, volumes, probs)``.

        Parameters
        ----------
        idx : int
            Row index into the mixture DataFrame.

        Returns
        -------
        temperature : float
            Temperature in Kelvin for this sample.
        mole_fractions : NDArray[np.float64]
            Mole fractions of the components. Shape ``(n,)``, sums to 1.
        areas : NDArray[np.float64]
            Surface areas of the components in Å². Shape ``(n,)``.
        volumes : NDArray[np.float64]
            Volumes of the components in Å³. Shape ``(n,)``.
        probabilities : NDArray[np.float64]
            Normalised segment-type probability per component. Shape ``(n, m)``
            where *m* is 51 (merged) or 153 (split NHB/OH/OT).

        Raises
        ------
        KeyError
            If a component name in the row is not present in
            *component_to_file_mapping*.

        Examples
        --------
        >>> from importlib.resources import files
        >>> import pandas as pd
        >>> from cosmolayer.cosmosac.model import CosmoSac2002Model
        >>> from cosmolayer.cosmosac.datasets import CosmoFileDataset
        >>> data = files("cosmolayer.data")
        >>> df = pd.DataFrame({
        ...     "component_0": ["water"],
        ...     "mole_fraction_0": [0.4],
        ...     "component_1": ["fluoromethane"],
        ...     "mole_fraction_1": [0.6],
        ... })
        >>> mapping = {
        ...     "water": str(data / "O.cosmo"),
        ...     "fluoromethane": str(data / "CF.cosmo"),
        ... }
        >>> ds = CosmoFileDataset(df, mapping, CosmoSac2002Model, temperature=298.15)
        >>> T, fracs, areas, volumes, probs = ds[0]
        >>> T
        298.15
        >>> fracs.shape
        (2,)
        >>> areas.shape
        (2,)
        >>> probs.shape
        (2, 51)
        """
        temperature = (
            self._temperature_value
            if self._temperature_value is not None
            else self.mixture_dataframe.iloc[idx][self._temperature_column]
        )
        fracs_list = [
            float(self.mixture_dataframe.iloc[idx][column])
            for column in self._mole_fraction_columns
        ]
        components = [
            self.mixture_dataframe.iloc[idx][column]
            for column in self._component_columns
        ]
        file_paths = [
            str(self._prefix / self._component_to_file_mapping[component])
            for component in components
        ]
        properties = [
            _compute_component_properties(self._model, file_path)
            for file_path in file_paths
        ]
        T = float(temperature)
        mole_fractions = np.array(fracs_list, dtype=np.float64)
        areas = np.array([p[0] for p in properties], dtype=np.float64)
        volumes = np.array([p[1] for p in properties], dtype=np.float64)
        probabilities = np.stack([p[2] for p in properties], axis=0)
        return (T, mole_fractions, areas, volumes, probabilities)

    @staticmethod
    def _count_components(
        dataframe: pd.DataFrame, column_formatter: Callable[[int], str]
    ) -> int:
        """Count consecutive formatted columns starting from index 0."""
        return next(
            i for i in itertools.count() if column_formatter(i) not in dataframe.columns
        )
