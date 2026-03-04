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
        when accessing :attr:`probabilities` and :attr:`sigma_profiles`.
        Default is False.
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
    >>> mixture["1-aminoethenol"].area
    97.34554...
    >>> mixture["2-aminoethanol"].area
    103.51765...
    >>> mixture.component_names
    ('1-aminoethenol', '2-aminoethanol')
    >>> areas = mixture.areas
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
        """Get a component by name.

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
            Contents of a COSMO output file.
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
            Contents of the new component's COSMO output file.

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

    @property
    def area_per_segment(self) -> float:
        """Reference area per segment used by the COSMO-SAC model, in Å²."""
        return self._area_per_segment

    @property
    def merge_profiles(self) -> bool:
        """Whether segment groups (NHB, OH, OT) are merged for :attr:`sigma_profiles`
        and :attr:`probabilities`."""
        return self._merge_profiles

    @property
    def component_names(self) -> tuple[str, ...]:
        """Names of all components in the order they were provided."""
        return tuple(self._components.keys())

    @property
    def areas(self) -> NDArray[np.float64]:
        """Cavity surface areas for all components in Å². Shape: (n_components,)."""
        return np.array([component.area for component in self._components.values()])

    @property
    def volumes(self) -> NDArray[np.float64]:
        """Cavity volumes for all components in Å³. Shape: (n_components,)."""
        return np.array([component.volume for component in self._components.values()])

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Normalized segment-type probabilities for each component.

        Stack of each component's :attr:`Component.probabilities`. Shape is
        ``(n_components, num_points)`` if :attr:`merge_profiles` is True, else
        ``(n_components, 3*num_points)``.

        Returns
        -------
        np.ndarray
            Probabilities; each row sums to 1.0.
        """
        return np.stack(
            [component.probabilities for component in self._components.values()],
            axis=0,
        )

    @property
    def sigma_profiles(self) -> NDArray[np.float64]:
        """Per-component sigma profiles (surface area vs. charge density), in Å².

        Stack of each component's :attr:`Component.sigma_profile`. Shape is
        ``(n_components, num_points)`` when :attr:`merge_profiles` is True,
        ``(n_components, 3, num_points)`` when False (NHB, OH, OT).

        Returns
        -------
        np.ndarray
            Sigma profile array; last dimension is the sigma grid.
        """
        return np.stack(
            [component.sigma_profile for component in self._components.values()],
            axis=0,
        )

    def interaction_matrices(
        self, temperature: float
    ) -> tuple[NDArray[np.float64], ...]:
        """COSMO-SAC interaction matrices for the mixture at the given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in K; used to scale the matrices.

        Returns
        -------
        tuple of np.ndarray
            Matrices used in the COSMO-SAC activity coefficient calculation.
            Length and shapes match the generator (e.g. sigma–sigma and
            sigma'–sigma' for the 2010 model).
        """
        return self._interaction_matrix_generator(temperature)

    @property
    def temperature_exponents(self) -> tuple[float, ...]:
        """Exponents used to scale each interaction matrix with temperature.

        Each entry scales the corresponding matrix from
        :meth:`interaction_matrices` as T^exponent (e.g. 1 and 3 for the
        COSMO-SAC 2010 model).

        Returns
        -------
        tuple of float
            One exponent per interaction matrix.
        """
        return self._temperature_exponents
