"""
.. module:: cosmolayer.cosmosac.model
   :synopsis: Model configuration for COSMO-SAC calculations.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .component import Component
from .constants import (
    COSMO_SAC_2002_AREA_PER_SEGMENT,
    COSMO_SAC_2002_AVERAGING_RADIUS,
    COSMO_SAC_2002_EXPONENTS,
    COSMO_SAC_2002_F_DECAY,
    COSMO_SAC_2002_SIGMA_0,
    COSMO_SAC_2010_AREA_PER_SEGMENT,
    COSMO_SAC_2010_AVERAGING_RADIUS,
    COSMO_SAC_2010_EXPONENTS,
    COSMO_SAC_2010_F_DECAY,
    COSMO_SAC_2010_SIGMA_0,
)
from .interaction_matrices import (
    create_cosmo_sac_2002_matrix,
    create_cosmo_sac_2010_matrices,
)
from .mixture import Mixture


@dataclass(frozen=True)
class Model:
    r"""Immutable configuration for a COSMO-SAC model variant.

    Bundles all model-specific parameters into a single object, ensuring that
    components and mixtures are always created with consistent settings.  Use the
    pre-built :data:`CosmoSac2002Model` and :data:`CosmoSac2010Model` singletons
    for standard calculations, or construct a custom instance for research purposes.

    Parameters
    ----------
    min_sigma : float
        Minimum screening charge density in e/Å².
    max_sigma : float
        Maximum screening charge density in e/Å².
    num_points : int
        Number of discrete points in the sigma grid.
    area_per_segment : float
        Reference surface area of a single segment in Å².
    averaging_radius : float
        Effective radius for distance-weighted sigma averaging in Å.
    f_decay : float
        Decay factor for exponential distance weighting in the sigma averaging
        procedure.
    sigma_0 : float or None
        Standard deviation of the Gaussian probability of a segment to form a
        hydrogen bond in e/Å².  Set to ``None`` to disable hydrogen-bond
        splitting (all surface area is assigned to the NHB class).
    merge_profiles : bool
        Whether segment-group profiles (NHB, OH, OT) should be merged into a
        single distribution by default when computing probabilities.
    temperature_exponents : tuple[int, ...]
        Exponents applied to the temperature in the interaction energy expression.
        Must have the same length as the tuple returned by
        ``interaction_matrix_generator``.
    interaction_matrix_generator : Callable
        Function ``f(temperature) -> tuple[NDArray, ...]`` that produces the
        dimensionless interaction energy matrices ΔW/(RT) at a given temperature.

    Examples
    --------
    Using a pre-built model to create a component:

    >>> from importlib.resources import files
    >>> from cosmolayer.cosmosac.model import CosmoSac2002Model
    >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
    >>> component = CosmoSac2002Model.create_component(path.read_text())
    >>> component.area
    97.34554...

    Inspecting model parameters:

    >>> CosmoSac2002Model.area_per_segment
    7.5
    >>> CosmoSac2002Model.merge_profiles
    True
    >>> CosmoSac2002Model.temperature_exponents
    (1,)
    """

    min_sigma: float
    max_sigma: float
    num_points: int
    area_per_segment: float
    averaging_radius: float
    f_decay: float
    sigma_0: float | None
    merge_profiles: bool
    temperature_exponents: tuple[int, ...]
    interaction_matrix_generator: Callable[[float], tuple[NDArray[np.float64], ...]] = (
        field(repr=False)
    )

    def create_interaction_matrices(
        self, temperature: float
    ) -> tuple[NDArray[np.float64], ...]:
        """Create the interaction energy matrices at a given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        tuple[NDArray[np.float64], ...]
            Dimensionless interaction energy matrices ΔW/(RT).  The number of
            matrices matches the length of :attr:`temperature_exponents`.

        Examples
        --------
        >>> from cosmolayer.cosmosac import CosmoSac2002Model, CosmoSac2010Model

        COSMO-SAC 2002 produces a single interaction matrix:

        >>> matrices = CosmoSac2002Model.create_interaction_matrices(298.15)
        >>> len(matrices)
        1
        >>> matrix = matrices[0]
        >>> matrix.shape
        (51, 51)
        >>> print(matrix.min() < 0)  # H-bonding can be favorable (negative)
        True
        >>> print(matrix.max() > 0)  # Misfit interactions are unfavorable
        True

        Plotting the COSMO-SAC 2002 interaction matrix:

        .. plot::
            :context: close-figs

            >>> from cosmolayer.cosmosac import CosmoSac2002Model
            >>> from matplotlib import pyplot as plt
            >>> matrices = CosmoSac2002Model.create_interaction_matrices(298.15)
            >>> fig, ax = plt.subplots(figsize=(8, 6))
            >>> im = ax.imshow(matrices[0], cmap="Spectral")
            >>> _ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
            >>> fig.tight_layout()

        COSMO-SAC 2010 produces two matrices (one per temperature exponent):

        >>> matrices = CosmoSac2010Model.create_interaction_matrices(298.15)
        >>> len(matrices)
        2
        >>> all(m.shape == (153, 153) for m in matrices)
        True

        Plotting the COSMO-SAC 2010 interaction matrix:

        .. plot::
            :context: close-figs

            >>> from cosmolayer.cosmosac import CosmoSac2010Model
            >>> from matplotlib import pyplot as plt
            >>> matrices = CosmoSac2010Model.create_interaction_matrices(298.15)
            >>> fig, ax = plt.subplots(figsize=(8, 6))
            >>> im = ax.imshow(sum(matrices), cmap="Spectral")
            >>> _ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
            >>> fig.tight_layout()

        """
        return self.interaction_matrix_generator(temperature)

    def create_component(
        self,
        cosmo_string: str,
    ) -> Component:
        """Create a :class:`~cosmolayer.cosmosac.component.Component` consistent
        with this model.

        Parameters
        ----------
        cosmo_string : str
            Contents of a COSMO output file.

        Returns
        -------
        Component
            Molecular component configured with the model's parameters.

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.cosmosac.model import CosmoSac2010Model
        >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
        >>> component = CosmoSac2010Model.create_component(path.read_text())
        >>> component.area
        97.34554...
        >>> component.probabilities.shape
        (153,)
        """
        return Component(
            cosmo_string,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_points=self.num_points,
            averaging_radius=self.averaging_radius,
            f_decay=self.f_decay,
            sigma_0=self.sigma_0,
            merge_profiles=self.merge_profiles,
        )

    def create_mixture(
        self,
        components: dict[str, str],
    ) -> "Mixture":
        """Create a :class:`~cosmolayer.cosmosac.mixture.Mixture` consistent with
        this model.

        Parameters
        ----------
        components : dict[str, str]
            Dictionary mapping component names to COSMO strings (contents of
            COSMO output files).

        Returns
        -------
        Mixture
            Mixture configured with the model's parameters.

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.cosmosac.model import CosmoSac2002Model, CosmoSac2010Model

        Creating a mixture with the COSMO-SAC 2002 model:

        >>> source = files("cosmolayer.data")
        >>> components = {
        ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
        ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
        ... }
        >>> mixture = CosmoSac2002Model.create_mixture(components)
        >>> len(mixture)
        2
        >>> mixture.interaction_matrices(298.15)[0].shape
        (51, 51)

        Creating a mixture with the COSMO-SAC 2010 model:

        >>> mixture = CosmoSac2010Model.create_mixture(components)
        >>> len(mixture)
        2
        >>> matrices = mixture.interaction_matrices(298.15)
        >>> len(matrices)
        2
        >>> all(m.shape == (153, 153) for m in matrices)
        True
        """

        return Mixture(
            components,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_points=self.num_points,
            area_per_segment=self.area_per_segment,
            averaging_radius=self.averaging_radius,
            f_decay=self.f_decay,
            sigma_0=self.sigma_0,
            merge_profiles=self.merge_profiles,
            interaction_matrix_generator=self.interaction_matrix_generator,
            temperature_exponents=self.temperature_exponents,
        )

    @property
    def num_segment_types(self) -> int:
        """Number of segment types."""
        if self.merge_profiles:
            return self.num_points
        return 3 * self.num_points


CosmoSac2002Model = Model(
    min_sigma=-0.025,
    max_sigma=0.025,
    num_points=51,
    area_per_segment=COSMO_SAC_2002_AREA_PER_SEGMENT,
    averaging_radius=COSMO_SAC_2002_AVERAGING_RADIUS,
    f_decay=COSMO_SAC_2002_F_DECAY,
    sigma_0=COSMO_SAC_2002_SIGMA_0,
    merge_profiles=True,
    temperature_exponents=COSMO_SAC_2002_EXPONENTS,
    interaction_matrix_generator=lambda T: (create_cosmo_sac_2002_matrix(T),),
)

CosmoSac2010Model = Model(
    min_sigma=-0.025,
    max_sigma=0.025,
    num_points=51,
    area_per_segment=COSMO_SAC_2010_AREA_PER_SEGMENT,
    averaging_radius=COSMO_SAC_2010_AVERAGING_RADIUS,
    f_decay=COSMO_SAC_2010_F_DECAY,
    sigma_0=COSMO_SAC_2010_SIGMA_0,
    merge_profiles=False,
    temperature_exponents=COSMO_SAC_2010_EXPONENTS,
    interaction_matrix_generator=create_cosmo_sac_2010_matrices,
)
