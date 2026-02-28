from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .component import Component
from .constants import (
    COSMO_SAC_2010_AREA_PER_SEGMENT,
    COSMO_SAC_2010_AVERAGING_RADIUS,
    COSMO_SAC_2010_EXPONENTS,
    COSMO_SAC_2010_F_DECAY,
    COSMO_SAC_2010_SIGMA_0,
)
from .interaction_matrices import create_cosmo_sac_2010_matrices


class Mixture:
    """Mixture of molecular components for COSMO-SAC calculations.

    This class manages a collection of molecular components, each defined by
    a COSMO output file from quantum mechanical calculations.

    .. note::
        The default parameters correspond to the COSMO-SAC 2010 model :cite:`Bell2020`.

    Parameters
    ----------
    components : dict[str, str]
        Dictionary mapping component names to COSMO strings (i.e., contents of COSMO
        output files from quantum mechanical calculations).
    min_sigma : float, optional
        Minimum screening charge density in e/Å². Default is -0.025 e/Å².
    max_sigma : float, optional
        Maximum screening charge density in e/Å². Default is 0.025 e/Å².
    num_points : int, optional
        Number of discrete points in the sigma profile. Default is 51.
    area_per_segment : float, optional
        Reference area in Å². Default is 7.25 Å².
    averaging_radius : float, optional
        Effective radius for distance-weighted sigma averaging in Å.
        Default is √(7.25 / π) Å.
    f_decay : float, optional
        Decay factor for exponential distance weighting. Default is 3.57.
    sigma_0 : float or None, optional
        Standard deviation of the Gaussian probability of a segment to form
        a hydrogen bond in e/Å².  Set to ``None`` to disable hydrogen-bond
        splitting (all surface area is assigned to the NHB class).
        Default is 0.007 e/Å².
    merge_profiles : bool, optional
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
    >>> from cosmolayer.cosmosac import Mixture
    >>> source = files("cosmolayer.data")
    >>> components = {
    ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
    ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
    ... }
    >>> mixture = Mixture(components)
    >>> len(mixture)
    2
    >>> mixture["1-aminoethenol"].get_area()
    97.34554...
    >>> mixture["2-aminoethanol"].get_area()
    103.51765...
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
        components: dict[str, str],
        min_sigma: float = -0.025,  # e/Å²
        max_sigma: float = 0.025,  # e/Å²
        num_points: int = 51,
        area_per_segment: float = COSMO_SAC_2010_AREA_PER_SEGMENT,  # Å²
        averaging_radius: float = COSMO_SAC_2010_AVERAGING_RADIUS,  # Å
        f_decay: float = COSMO_SAC_2010_F_DECAY,
        sigma_0: float | None = COSMO_SAC_2010_SIGMA_0,  # e/Å²
        merge_profiles: bool = False,
        interaction_matrix_generator: Callable[
            [float], tuple[NDArray[np.float64], ...]
        ] = create_cosmo_sac_2010_matrices,
        temperature_exponents: tuple[float, ...] = COSMO_SAC_2010_EXPONENTS,
    ) -> None:
        if not components:
            raise ValueError("At least one component must be provided.")

        self._components: OrderedDict[str, Component] = OrderedDict()
        self._min_sigma = min_sigma
        self._max_sigma = max_sigma
        self._num_points = num_points
        self._area_per_segment = area_per_segment
        self._averaging_radius = averaging_radius
        self._f_decay = f_decay
        self._sigma_0 = sigma_0
        self._merge_profiles = merge_profiles
        self._interaction_matrix_generator = interaction_matrix_generator
        self._temperature_exponents = temperature_exponents

        for name, cosmo_string in components.items():
            self.add_component(name, cosmo_string)

    def __len__(self) -> int:
        """Return the number of components in the mixture."""
        return len(self._components)

    def __getitem__(self, name: str) -> Component:
        """Get a component by name or index.

        Parameters
        ----------
        name : str
            Component name.

        Returns
        -------
        Component
            The requested component.

        Raises
        ------
        KeyError
            If component name not found.
        """
        return self._components[name]

    def _create_component(self, cosmo_string: str) -> Component:
        return Component(
            cosmo_string,
            min_sigma=self._min_sigma,
            max_sigma=self._max_sigma,
            num_points=self._num_points,
            averaging_radius=self._averaging_radius,
            f_decay=self._f_decay,
            sigma_0=self._sigma_0,
            merge_profiles=self._merge_profiles,
        )

    def add_component(self, name: str, cosmo_string: str) -> None:
        """Add a component to the mixture.

        Parameters
        ----------
        name : str
            Component name.
        cosmo_string : str
            COSMO string.
        """
        self._components[name] = self._create_component(cosmo_string)

    def remove_component(self, name: str) -> None:
        """Remove a component from the mixture.

        Parameters
        ----------
        name : str
            Component name.
        """
        del self._components[name]

    def replace_component(
        self, old_name: str, new_name: str, cosmo_string: str
    ) -> None:
        """Replace a component in the mixture.

        The new name must not already exist in the mixture, unless it is the same as
        the old name. In this case, the component data is updated using the new COSMO
        string.

        Parameters
        ----------
        old_name : str
            Name of the component to replace.
        new_name : str
            Name of the new component.
        cosmo_string : str
            New component's COSMO string.

        Raises
        ------
        ValueError
            If the old name is not found in the mixture.
            If the new name already exists in the mixture and is not the same as the
            old name.
        """
        if old_name not in self._components:
            raise ValueError(f"Component {old_name} not found in mixture.")
        if new_name != old_name and new_name in self._components:
            raise ValueError(f"Component {new_name} already exists in mixture.")
        if new_name == old_name:
            self._components[new_name] = self._create_component(cosmo_string)
        else:
            old_components = self._components
            self._components = OrderedDict()
            for name, component in old_components.items():
                if name == old_name:
                    self._components[new_name] = self._create_component(cosmo_string)
                else:
                    self._components[name] = component

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
        return tuple(self._components.keys())

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
        >>> from cosmolayer.cosmosac import Mixture
        >>> source = files("cosmolayer.data")
        >>> components = {
        ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
        ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
        ... }
        >>> mixture = Mixture(components)
        >>> areas = mixture.get_areas()
        >>> areas.shape
        (2,)
        """
        return np.array(
            [component.get_area() for component in self._components.values()]
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
        >>> from cosmolayer.cosmosac import Mixture
        >>> source = files("cosmolayer.data")
        >>> components = {
        ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
        ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
        ... }
        >>> mixture = Mixture(components)
        >>> volumes = mixture.get_volumes()
        >>> volumes.shape
        (2,)
        """
        return np.array(
            [component.get_volume() for component in self._components.values()]
        )

    def get_probabilities(self, regularize: float = 1e-10) -> NDArray[np.float64]:
        """Get probabilities of segment types for all components.

        Parameters
        ----------
        regularize : float, optional
            Minimum value for clipping probabilities. Set to 0 to disable
            regularization. Default is 1e-10. If clipping occurs, the returned
            distribution is renormalized to sum to 1.

        Returns
        -------
        NDArray[np.float64]
            Array of probabilities for each component.
            If merge=True: shape is (n_components, num_points).
            If merge=False: shape is (n_components, 3*num_points).

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.cosmosac import Mixture
        >>> import numpy as np
        >>> source = files("cosmolayer.data")
        >>> components = {
        ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
        ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
        ... }
        >>> mixture = Mixture(components)
        >>> probabilities = mixture.get_probabilities()
        >>> probabilities.shape
        (2, 153)
        >>> bool(np.all(probabilities <= 1))
        True
        """
        return np.stack(
            [
                component.get_probabilities(regularize=regularize)
                for component in self._components.values()
            ],
            axis=0,
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
        >>> from cosmolayer.cosmosac import Mixture
        >>> source = files("cosmolayer.data")
        >>> components = {
        ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
        ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
        ... }
        >>> mixture = Mixture(components)
        >>> profiles = mixture.get_sigma_profiles()
        >>> profiles.shape
        (2, 51)
        """
        return np.stack(
            [
                component.get_sigma_profile(segment_class)
                for component in self._components.values()
            ],
            axis=0,
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
        >>> from cosmolayer.cosmosac import Mixture
        >>> source = files("cosmolayer.data")
        >>> components = {
        ...     "1-aminoethenol": (source / "C=C(N)O.cosmo").read_text(),
        ...     "2-aminoethanol": (source / "NCCO.cosmo").read_text(),
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
