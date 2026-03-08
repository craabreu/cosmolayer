"""
.. module:: cosmolayer.cosmosac.datapoint
   :synopsis: Single datapoint for a COSMO-SAC mixture.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import functools
import os
from collections.abc import Sequence

import numpy as np

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
    targets : Sequence[float]
        Target values for the mixture (e.g. activity coefficients, excess
        properties). Length defines the number of training targets.
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
