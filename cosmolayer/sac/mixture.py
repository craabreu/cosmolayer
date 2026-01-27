import os
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .component import Component
from .interaction_matrices import (
    COSMO_SAC_2002_EXPONENTS,
    COSMO_SAC_2002_REFERENCE_AREA,
    COSMO_SAC_2010_EXPONENTS,
    COSMO_SAC_2010_REFERENCE_AREA,
    create_cosmo_sac_2002_matrix,
    create_cosmo_sac_2010_matrices,
)


class Mixture:
    """Mixture of molecular components for COSMO-SAC calculations.

    This class manages a collection of molecular components, each defined by
    a COSMO output file from quantum mechanical calculations.

    .. note::
        With all default parameters, this class is equivalent to
        :class:`CosmoSac2010Mixture`.

    Parameters
    ----------
    components : dict[str, str | os.PathLike]
        Dictionary mapping component names to paths of COSMO output files.
    min_sigma : float, optional
        Minimum screening charge density in e/Å². Default is -0.025 e/Å².
    max_sigma : float, optional
        Maximum screening charge density in e/Å². Default is 0.025 e/Å².
    num_points : int, optional
        Number of discrete points in the sigma profile. Default is 51.
    area_per_segment : float, optional
        Reference area in Å². Default is 7.25 Å².
    averaging_squared_radius : float, optional
        Effective squared radius for distance-weighted sigma averaging in Å².
        Default is (7.25 / π) Å².
    f_decay : float, optional
        Decay factor for exponential distance weighting. Default is 3.57.
    sigma_0 : float, optional
        Standard deviation of the Gaussian probability of a segment to form
        a hydrogen bond. Default is 0.007 e/Å².
    merge : bool, optional
        Whether to merge segment groups (NHB, OH, OT) into a single profile
        when calling get_probabilities(). Default is False.
    regularize : float, optional
        Minimum value for clipping probabilities. Default is 1e-10.
    interaction_matrix_generator : Callable, optional
        Function to generate the interaction matrix for the mixture at a given
        temperature. Default is :func:`create_cosmo_sac_2010_matrices` with default
        parameters.
    temperature_exponents : tuple[float, ...], optional
        Temperature exponents for the interaction matrices. Must be the same length as
        the tuple returned by ``interaction_matrix_generator``.
        Default is (1, 3).

    Raises
    ------
    ValueError
        If no components are provided.
    FileNotFoundError
        If any of the specified files do not exist.

    Examples
    --------
    >>> from importlib.resources import files
    >>> from cosmolayer.sac import Mixture
    >>> components = {
    ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
    ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
    ... }
    >>> mixture = Mixture(components)
    >>> len(mixture)
    2
    >>> mixture["1-aminoethenol"].get_area()
    97.34554...
    >>> mixture["2-aminoethanol"].get_area()
    103.51765...
    >>> mixture[0].get_area()
    97.34554...
    >>> mixture.get_component_names()
    ('1-aminoethenol', '2-aminoethanol')
    >>> areas = mixture.get_areas()
    >>> areas.shape
    (2,)
    >>> float(areas[0])
    97.34554...
    """

    def __init__(  # noqa: PLR0913
        self,
        components: dict[str, str | os.PathLike[str]],
        min_sigma: float = -0.025,  # e/A^2
        max_sigma: float = 0.025,  # e/A^2
        num_points: int = 51,
        area_per_segment: float = COSMO_SAC_2010_REFERENCE_AREA,  # Å²
        averaging_squared_radius: float = COSMO_SAC_2010_REFERENCE_AREA / np.pi,  # A^2
        f_decay: float = 3.57,
        sigma_0: float = 0.007,  # e/A^2
        merge: bool = False,
        regularize: float = 1e-10,
        interaction_matrix_generator: Callable[
            [float], tuple[NDArray[np.float64], ...]
        ] = lambda temperature: create_cosmo_sac_2010_matrices(temperature),
        temperature_exponents: tuple[float, ...] = COSMO_SAC_2010_EXPONENTS,
    ) -> None:
        if not components:
            raise ValueError("At least one component must be provided.")

        self._names = list(components.keys())
        self._components_dict = {
            name: Component(
                path,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_points=num_points,
                averaging_squared_radius=averaging_squared_radius,
                f_decay=f_decay,
                sigma_0=sigma_0,
            )
            for name, path in components.items()
        }
        self._area_per_segment = area_per_segment
        self._merge = merge
        self._regularize = regularize
        self._interaction_matrix_generator = interaction_matrix_generator
        self._temperature_exponents = temperature_exponents

    def __len__(self) -> int:
        """Return the number of components in the mixture."""
        return len(self._names)

    def __getitem__(self, key: str | int) -> Component:
        """Get a component by name or index.

        Parameters
        ----------
        key : str or int
            Component name or integer index.

        Returns
        -------
        Component
            The requested component.

        Raises
        ------
        KeyError
            If component name not found.
        IndexError
            If index out of range.
        """
        if isinstance(key, str):
            return self._components_dict[key]
        return self._components_dict[self._names[key]]

    def get_area_per_segment(self) -> float:
        """Get the area per segment for the mixture.

        Returns
        -------
        float
            Area per segment in Å².
        """
        return self._area_per_segment

    def get_component_names(self) -> tuple[str, ...]:
        """Get the names of all components in the mixture.

        Returns
        -------
        tuple[str, ...]
            Tuple of component names in the order they were provided.
        """
        return tuple(self._names)

    def get_areas(self) -> NDArray[np.float64]:
        """Get cavity surface areas for all components.

        Returns
        -------
        NDArray[np.float64]
            Array of cavity surface areas in Å² for each component.
            Shape: (n_components,).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Mixture
        >>> components = {
        ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
        ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
        ... }
        >>> mixture = Mixture(components)
        >>> areas = mixture.get_areas()
        >>> areas.shape
        (2,)
        """
        return np.array(
            [self._components_dict[name].get_area() for name in self._names]
        )

    def get_volumes(self) -> NDArray[np.float64]:
        """Get cavity volumes for all components.

        Returns
        -------
        NDArray[np.float64]
            Array of cavity volumes in Å³ for each component.
            Shape: (n_components,).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Mixture
        >>> components = {
        ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
        ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
        ... }
        >>> mixture = Mixture(components)
        >>> volumes = mixture.get_volumes()
        >>> volumes.shape
        (2,)
        """
        return np.array(
            [self._components_dict[name].get_volume() for name in self._names]
        )

    def get_probabilities(self) -> NDArray[np.float64]:
        """Get probabilities of segment types for all components.

        Returns
        -------
        NDArray[np.float64]
            Array of probabilities for each component.
            If merge=True: shape is (n_components, num_points).
            If merge=False: shape is (n_components, 3*num_points).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Mixture
        >>> import numpy as np
        >>> components = {
        ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
        ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
        ... }
        >>> mixture = Mixture(components, merge=True)
        >>> probabilities = mixture.get_probabilities()
        >>> probabilities.shape
        (2, 51)
        >>> bool(np.all(probabilities <= 1))
        True
        """
        return np.stack(
            [
                self._components_dict[name].get_probabilities(
                    self._merge, self._regularize
                )
                for name in self._names
            ]
        )

    def get_sigma_profiles(
        self, segment_class: str | None = None
    ) -> NDArray[np.float64]:
        """Get sigma profiles for all components.

        Parameters
        ----------
        segment_class : str, optional
            Segment class ("NHB", "OH", or "OT"). If None, returns total profiles.

        Returns
        -------
        NDArray[np.float64]
            Array of sigma profiles for each component.
            Shape: (n_components, num_points).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Mixture
        >>> components = {
        ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
        ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
        ... }
        >>> mixture = Mixture(components)
        >>> profiles = mixture.get_sigma_profiles()
        >>> profiles.shape
        (2, 51)
        """
        return np.stack(
            [
                self._components_dict[name].get_sigma_profile(segment_class)
                for name in self._names
            ]
        )

    def get_interaction_matrices(
        self, temperature: float
    ) -> tuple[NDArray[np.float64], ...]:
        """Get the COSMO-SAC interaction matrices for the mixture.

        Parameters
        ----------
        temperature : float
            The temperature in Kelvin at which the interaction matrices are computed.

        Returns
        -------
        tuple[NDArray[np.float64], ...]
            Tuple of interaction matrices, one for each segment type pair.
            Each matrix has shape (n_segment_types, n_segment_types).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Mixture
        >>> components = {
        ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
        ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
        ... }
        >>> mixture = Mixture(components)
        >>> matrices = mixture.get_interaction_matrices(298.15)
        >>> isinstance(matrices, tuple)
        True
        >>> len(matrices)
        2
        >>> all(isinstance(mat, np.ndarray) for mat in matrices)
        True
        >>> all(mat.shape == (153, 153) for mat in matrices)
        True
        """
        return self._interaction_matrix_generator(temperature)

    def get_temperature_exponents(self) -> tuple[float, ...]:
        """Get the temperature exponents for the interaction matrices.

        Returns
        -------
        tuple[float, ...]
            Tuple of temperature exponents.
        """
        return self._temperature_exponents


class CosmoSac2002Mixture(Mixture):
    """Mixture of molecular components for COSMO-SAC 2002 calculations.

    This class is pre-configured with COSMO-SAC 2002 model parameters:

    - averaging_squared_radius = (0.81764 Å)²
    - f_decay = 1.0
    - merge = True (single segment type distribution)
    - Interaction matrix from :func:`create_cosmo_sac_2002_matrix` with default
      parameters.
    - temperature_exponents = (1,)

    Parameters
    ----------
    components : dict[str, str | os.PathLike]
        Dictionary mapping component names to paths of COSMO output files.

    Examples
    --------
    >>> from importlib.resources import files
    >>> from cosmolayer.sac import CosmoSac2002Mixture
    >>> components = {
    ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
    ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
    ... }
    >>> mixture = CosmoSac2002Mixture(components)
    >>> len(mixture)
    2
    >>> probabilities = mixture.get_probabilities()
    >>> probabilities.shape  # merge=True, so shape is (n_components, num_points)
    (2, 51)
    >>> matrices = mixture.get_interaction_matrices(298.15)
    >>> len(matrices)  # COSMO-SAC 2002 returns single matrix (in tuple)
    1
    >>> matrices[0].shape
    (51, 51)
    """

    def __init__(self, components: dict[str, str | os.PathLike[str]]) -> None:
        super().__init__(
            components,
            area_per_segment=COSMO_SAC_2002_REFERENCE_AREA,
            averaging_squared_radius=0.81764**2,  # ≈ 0.6685 Å²
            f_decay=1.0,
            merge=True,
            interaction_matrix_generator=lambda temperature: (
                create_cosmo_sac_2002_matrix(temperature),
            ),
            temperature_exponents=COSMO_SAC_2002_EXPONENTS,
        )


class CosmoSac2010Mixture(Mixture):
    """Mixture of molecular components for COSMO-SAC 2010 calculations.

    This class is pre-configured with COSMO-SAC 2010 model parameters:

    - averaging_squared_radius = 7.25/π Å²
    - f_decay = 3.57
    - merge = False (separate segment type distributions for NHB, OH, OT)
    - Interaction matrices from :func:`create_cosmo_sac_2010_matrices` with default
      parameters.

    Parameters
    ----------
    components : dict[str, str | os.PathLike]
        Dictionary mapping component names to paths of COSMO output files.

    Examples
    --------
    >>> from importlib.resources import files
    >>> from cosmolayer.sac import CosmoSac2010Mixture
    >>> components = {
    ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
    ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
    ... }
    >>> mixture = CosmoSac2010Mixture(components)
    >>> len(mixture)
    2
    >>> probabilities = mixture.get_probabilities()
    >>> probabilities.shape  # merge=False, so shape is (n_components, 3*num_points)
    (2, 153)
    >>> matrices = mixture.get_interaction_matrices(298.15)
    >>> len(matrices)  # COSMO-SAC 2010 returns two matrices
    2
    >>> all(mat.shape == (153, 153) for mat in matrices)
    True
    """

    def __init__(self, components: dict[str, str | os.PathLike[str]]) -> None:
        super().__init__(components)
