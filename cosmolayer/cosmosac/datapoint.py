"""
.. module:: cosmolayer.cosmosac.datapoint
   :synopsis: Single datapoint for a COSMO-SAC mixture.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..cosmodata import MixtureDatapoint, NumpyArray1D
from .model import CosmoSac2010Model, Model


@functools.cache
def _get_component_data(
    cosmo_file_path: str,
    model: Model,
) -> tuple[float, float, NumpyArray1D]:
    """Load area, volume, and sigma-profile probabilities for one component.

    Results are cached by (path, model) so repeated loads of the same file
    with the same model are cheap.

    Parameters
    ----------
    cosmo_file_path : str
        Path to a COSMO output file (e.g. ``.cosmo``).
    model : :class:`~cosmolayer.cosmosac.model.Model`
        Model used to create the component and compute probabilities.

    Returns
    -------
    tuple[float, float, NumpyArray1D]
        (area, volume, probabilities) for the component.
    """
    with open(cosmo_file_path) as f:
        component = model.create_component(f.read())
    probabilities = component.probabilities
    probabilities.flags.writeable = False
    return component.area, component.volume, probabilities


class CosmoSacMixtureDatapoint(MixtureDatapoint):
    """Subclass of :class:`MixtureDatapoint` for COSMO-SAC mixtures.

    Parameters
    ----------
    cosmo_files : Sequence[os.PathLike[str]]
        Paths to COSMO files, one per component. Order must match
        ``mole_fractions`` and rows of ``component_targets``.
    mole_fractions : Sequence[float]
        Mole fractions for each component (should sum to 1).
    temperature : float
        Temperature in Kelvin.
    targets : Sequence[float] | None, optional
        Target values for the mixture (e.g. activity coefficients, excess
        properties). Length defines the number of training targets. If ``None``,
        no training targets are stored.
    model: :class:`~cosmolayer.cosmosac.model.Model`
        COSMO-SAC model used to load components and compute probabilities.

    Raises
    ------
    ValueError
        If the number of mole fractions does not match the number of COSMO files.

    Examples
    --------
    Build a binary mixture datapoint from packaged COSMO files and read inputs
    and targets:

    >>> from importlib.resources import files
    >>> from cosmolayer.cosmosac import CosmoSac2002Model
    >>> from cosmolayer.cosmosac.datapoint import CosmoSacMixtureDatapoint
    >>> data = files("cosmolayer.data")
    >>> cosmo_files = [data / "C=C(N)O.cosmo", data / "NCCO.cosmo"]
    >>> mole_fractions = [0.5, 0.5]
    >>> temperature = 298.15
    >>> targets = [1.2]
    >>> dp = CosmoSacMixtureDatapoint(
    ...     cosmo_files, mole_fractions, temperature,
    ...     targets, CosmoSac2002Model,
    ... )
    >>> dp.temperature
    298.15
    >>> dp.mole_fractions.shape
    (2,)
    >>> dp.areas.shape, dp.volumes.shape
    ((2,), (2,))
    >>> dp.probabilities.shape
    (2, 51)
    >>> dp.targets.shape
    (1,)
    >>> dp.num_components, dp.num_segment_types
    (2, 51)
    >>> dp.num_targets
    1
    """

    def __init__(  # noqa: PLR0913
        self,
        cosmo_files: Sequence[os.PathLike[str]],
        mole_fractions: Sequence[float],
        temperature: float,
        targets: Sequence[float] | None = None,
        model: Model = CosmoSac2010Model,
    ):
        """Build a mixture datapoint from COSMO files and optional targets.

        Parameters
        ----------
        cosmo_files : Sequence[os.PathLike[str]]
            Paths to COSMO files, one per component.
        mole_fractions : Sequence[float]
            Mole fractions for each component.
        temperature : float
            Temperature in Kelvin.
        targets : Sequence[float] | None, optional
            Optional training targets. If ``None``, no training targets are stored.
        model : Model, optional
            COSMO-SAC model used to parse component data. Default is
            :data:`CosmoSac2010Model`.
        """
        if targets is None:
            targets = []

        areas, volumes, probabilities = zip(
            *[_get_component_data(str(path), model) for path in cosmo_files],
            strict=True,
        )

        super().__init__(
            temperature=temperature,
            mole_fractions=np.array(mole_fractions),
            areas=np.array(areas),
            volumes=np.array(volumes),
            probabilities=np.stack(probabilities, axis=0),
            targets=np.array(targets),
        )

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        cosmo_files: Sequence[str | os.PathLike[str]],
        mole_fractions: Sequence[str | float],
        temperature: str | float,
        targets: Sequence[str | float],
        model: Model = CosmoSac2010Model,
    ) -> CosmoSacMixtureDatapoint:
        """Build a mixture datapoint from one row of a DataFrame (as a Series).

        This method is useful for building
        :class:`~cosmolayer.MixtureTrainingDataset`
        instances from a pandas DataFrame using :meth:`pandas.DataFrame.apply`.

        Column specifiers can be column names (strings), in which case values
        are taken from ``series[key]``, or literal numbers or paths (floats or
        os.PathLike), which are converted to strings and used as-is. This allows
        mixing DataFrame columns with fixed values (e.g. same solvent, same
        temperature, or same mole fractions for all datapoints).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from pathlib import Path
        >>> data = Path(str(files("cosmolayer") / "data"))
        >>> row = pd.Series(
        ...     {
        ...         "file_a": data / "C=C(N)O.cosmo",
        ...         "target_1": 1.2,
        ...     }
        ... )
        >>> point = CosmoSacMixtureDatapoint.from_series(
        ...     series=row,
        ...     cosmo_files=["file_a", data / "NCCO.cosmo"],
        ...     mole_fractions=[0.25, 0.75],
        ...     temperature=298.15,
        ...     targets=["target_1"],
        ... )
        >>> point.num_components, point.num_targets
        (2, 1)
        >>> point.mole_fractions.tolist()
        [0.25, 0.75]

        Parameters
        ----------
        series : pd.Series
            One row of a DataFrame (e.g. from ``df.iloc[i]`` or ``df.iterrows()``).
        cosmo_files : Sequence[str | pathlib.Path]
            For each component, either a column name (str) or a path to a COSMO
            file (pathlib.Path).
        mole_fractions : Sequence[str | float]
            For each component, either a column name (str) or a literal mole
            fraction (float). Values should sum to 1.
        temperature : str | float
            Column name for temperature in Kelvin, or a literal temperature.
        targets : Sequence[str | float]
            For each target, either a column name (str) or a literal value (float).
        model : Model, optional
            COSMO-SAC model used to load components. Default is
            :class:`~cosmolayer.cosmosac.model.CosmoSac2010Model`.

        Returns
        -------
        CosmoSacMixtureDatapoint
            A datapoint built from the series values.
        """

        return cls(
            cosmo_files=[
                series[cosmo_file] if isinstance(cosmo_file, str) else cosmo_file
                for cosmo_file in cosmo_files
            ],
            mole_fractions=[
                series[fraction] if isinstance(fraction, str) else fraction
                for fraction in mole_fractions
            ],
            temperature=(
                series[temperature] if isinstance(temperature, str) else temperature
            ),
            targets=[
                series[target] if isinstance(target, str) else target
                for target in targets
            ],
            model=model,
        )
