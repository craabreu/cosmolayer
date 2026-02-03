import os

import numpy as np
import pandas as pd
import periodictable as pt
from numpy.typing import NDArray

from ..parser import parse_cosmo_file
from .segment_groups import NHB, OH, OT, SEGMENT_GROUPS

COVALENT_FACTOR = 1.3  # Same as in RDKit


class Component:
    r"""Molecular component for the COSMO-SAC activity coefficient model.

    Parameters
    ----------
    cosmo_file_path : str or os.PathLike
        Path to the COSMO output file from quantum mechanical calculations.

    Keyword Arguments
    -----------------
    min_sigma : float, optional
        Minimum screening charge density in e/Å². Default is -0.025 e/Å².
    max_sigma : float, optional
        Maximum screening charge density in e/Å². Default is 0.025 e/Å².
    num_points : int, optional
        Number of discrete points in the sigma profile. Default is 51.
    averaging_squared_radius : float, optional
        Effective squared radius for distance-weighted sigma averaging in Å².
        Default is (7.25 / π) Å² :cite:`Bell2020`.
    f_decay : float, optional
        Decay factor for exponential distance weighting in the sigma averaging
        procedure. Default is 3.57 :cite:`Bell2020`.
    sigma_0 : float, optional
        Standard deviation of the Gaussian probability of a segment to form a hydrogen
        bond. Default is 0.007 e/Å² :cite:`Bell2020`.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file contents are not in the expected format.
    ValueError
        If averaged charge densities fall outside the specified sigma range.

    Examples
    --------
    >>> import numpy as np
    >>> from importlib.resources import files
    >>> from cosmolayer.sac import Component
    >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
    >>> component = Component(path)
    >>> component.get_area()
    97.34554...
    >>> component.get_volume()
    80.07160...
    >>> sigma_profile = component.get_sigma_profile()
    >>> print(sum(sigma_profile))
    97.34554...
    >>> sigma_profiles = {
    ...     s: component.get_sigma_profile(s)
    ...     for s in ["NHB", "OH", "OT"]
    ... }
    >>> for s in ["NHB", "OH", "OT"]:
    ...     print(s, sum(sigma_profiles[s]))
    NHB 72.31802...
    OH 12.25732...
    OT 12.77019...

    Plotting the sigma profiles:

    .. plot::
        :context: close-figs

        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Component
        >>> from matplotlib import pyplot as plt
        >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
        >>> component = Component(path)
        >>> fig, ax = plt.subplots(figsize=(8, 4))
        >>> grid = component.get_sigma_grid()
        >>> for s in ["NHB", "OH", "OT"]:
        ...     _ = ax.plot(grid, component.get_sigma_profile(s), label=s)
        >>> _ = ax.plot(grid, component.get_sigma_profile(), label="Overall")
        >>> _ = ax.set_xlabel("Charge density (e/Å²)")
        >>> _ = ax.set_ylabel("Surface area contribution (Å²)")
        >>> _ = ax.legend()
        >>> fig.tight_layout()

    Plotting the segment-type probabilities:

    .. plot::
        :context: close-figs

        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Component
        >>> from matplotlib import pyplot as plt
        >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
        >>> component = Component(path)
        >>> fig, ax = plt.subplots(figsize=(8, 4))
        >>> p = component.get_probabilities()
        >>> _ = ax.bar(range(len(p)), p)
        >>> _ = ax.set_xlabel("Segment type index")
        >>> _ = ax.set_ylabel("Probability")
        >>> fig.tight_layout()
    """

    def __init__(  # noqa: PLR0913
        self,
        cosmo_file_path: str | os.PathLike[str],
        *,
        min_sigma: float = -0.025,  # e/A^2
        max_sigma: float = 0.025,  # e/A^2
        num_points: int = 51,
        averaging_squared_radius: float = 7.25 / np.pi,  # A^2
        f_decay: float = 3.57,
        sigma_0: float = 0.007,  # e/A^2
    ):
        self._min_sigma = min_sigma
        self._grid = np.linspace(min_sigma, max_sigma, num_points)
        self._bin_width = (max_sigma - min_sigma) / (num_points - 1)

        self._averaging_squared_radius = averaging_squared_radius
        self._f_decay = f_decay
        self._sigma_0 = sigma_0

        self._format, self._atom_data, self._segment_data, self._volume = (
            parse_cosmo_file(cosmo_file_path)
        )
        averaged_sigmas = self._average_sigmas()
        if (averaged_sigmas < min_sigma).any() or (averaged_sigmas > max_sigma).any():
            raise ValueError("Averaged charge densities out of range.")

        self._area = float(self._segment_data["area"].sum())

        self._sigma_profiles = self._compute_sigma_profiles(averaged_sigmas)

    @staticmethod
    def _get_covalent_radius(element: str) -> float:
        """Get scaled covalent radius for bond detection.

        Parameters
        ----------
        element : str
            Chemical element symbol.

        Returns
        -------
        float
            Covalent radius in Å, scaled by factor 1.3.
        """
        radius: float = float(pt.elements.symbol(element).covalent_radius)
        return COVALENT_FACTOR * radius

    def _get_hydrogen_bonding_classes(self) -> pd.Series:
        """Classify atoms into hydrogen bonding types (OH, OT, NHB).

        Determines bonds from interatomic distances and assigns hydrogen bonding
        classes: OH (O-H bonds), OT (N-H, F-H bonds or isolated N/F/O), and
        NHB (all other atoms).

        Returns
        -------
        pd.Series
            Hydrogen bonding class label for each atom.
        """
        df = self._atom_data.copy()
        coords = df[["x", "y", "z"]].values
        distances = np.sqrt(np.square(coords[:, None, :] - coords).sum(axis=-1))
        elements = df["element"]
        radii = elements.apply(self._get_covalent_radius).values
        bonds = np.nonzero(np.triu(distances < (radii[:, None] + radii[None, :]), k=1))
        hb_class = df["element"].apply(
            lambda element: OT if element in ["N", "F", "O"] else NHB
        )
        for i, j in zip(*bonds, strict=True):
            elements_ij = set(elements.iloc[[i, j]])
            if elements_ij in [{"O", "H"}, {"N", "H"}, {"F", "H"}]:
                hb_class.at[i] = hb_class.at[j] = OH if "O" in elements_ij else OT
        return hb_class

    def _average_sigmas(self) -> NDArray[np.float64]:
        """Apply distance-weighted averaging to segment charge densities.

        Smooths raw screening charge densities (σ = q/A) using exponentially
        decaying weights based on distances between segment centroids.

        Returns
        -------
        np.ndarray
            Averaged screening charge density for each segment in e/Å.
        """
        sigmas = self._segment_data["charge"].values / self._segment_data["area"].values
        coords = self._segment_data[["x", "y", "z"]].values
        squared_distances = np.square(coords[:, None, :] - coords).sum(axis=-1)
        squared_radii = self._segment_data["area"].values / np.pi

        sums = squared_radii + self._averaging_squared_radius
        prods = squared_radii * self._averaging_squared_radius
        weights = np.exp(-self._f_decay * squared_distances / sums) * prods / sums
        result: NDArray[np.float64] = np.sum(weights * sigmas, axis=1) / np.sum(
            weights, axis=1
        )
        return result

    def _compute_sigma_profile(
        self, averaged_sigmas: NDArray[np.float64], areas: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Bin segment areas by charge density using linear interpolation.

        Parameters
        ----------
        averaged_sigmas : np.ndarray
            Averaged screening charge densities in e/Å².
        areas : np.ndarray
            Surface areas in Å².

        Returns
        -------
        np.ndarray
            Sigma profile histogram. Shape: (num_points,).
        """
        profile = np.zeros_like(self._grid)
        for sigma, area in zip(averaged_sigmas, areas, strict=True):
            index = int((sigma - self._min_sigma) / self._bin_width)
            weight = (self._grid[index + 1] - sigma) / self._bin_width
            profile[index] += area * weight
            profile[index + 1] += area * (1.0 - weight)
        return profile

    def _compute_sigma_profiles(
        self, averaged_sigmas: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Compute sigma profiles separated by hydrogen bonding type.

        Classifies segments by H-bonding type (OH, OT, NHB) based on parent atom
        and sigma sign, then applies a Gaussian probability weighting function.

        Parameters
        ----------
        averaged_sigmas : np.ndarray
            Averaged screening charge densities for all segments in e/Å².

        Returns
        -------
        dict
            Dictionary with keys "NHB", "OH", "OT" and values as sigma profile
            arrays. Each profile has shape (num_points,).
        """
        atom_indices = self._segment_data["atom"] - 1
        element = atom_indices.map(self._atom_data["element"])
        is_hb_candidate = (element == "H") == (averaged_sigmas < 0.0)
        hb_class = atom_indices.map(self._get_hydrogen_bonding_classes())
        mask_oh = is_hb_candidate & (hb_class == OH)
        mask_ot = is_hb_candidate & (hb_class == OT)
        mask_nhb = ~(mask_oh | mask_ot)
        areas = self._segment_data["area"].values
        profile_oh = self._compute_sigma_profile(
            averaged_sigmas[mask_oh], areas[mask_oh]
        )
        profile_ot = self._compute_sigma_profile(
            averaged_sigmas[mask_ot], areas[mask_ot]
        )
        profile_nhb = self._compute_sigma_profile(
            averaged_sigmas[mask_nhb], areas[mask_nhb]
        )
        hb_probability = 1.0 - np.exp(-0.5 * (self._grid / self._sigma_0) ** 2)
        return {
            NHB: profile_nhb + (profile_oh + profile_ot) * (1.0 - hb_probability),
            OH: profile_oh * hb_probability,
            OT: profile_ot * hb_probability,
        }

    def get_area(self) -> float:
        """Get the cavity surface area of the molecule in Å².

        Returns
        -------
        float
            Cavity surface area in Å². This is the sum of the areas of all
            segments from the COSMO calculation.
        """
        return self._area

    def get_volume(self) -> float:
        """Get the cavity volume of the molecule in Å³.

        Returns
        -------
        float
            Cavity volume in Å³.
        """
        return self._volume

    def get_format(self) -> str:
        """Get the COSMO file format that was parsed.

        Returns
        -------
        str
            The detected file format. Either "TURBOMOLE" or "DMol-3".

        Examples
        --------
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Component
        >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
        >>> component = Component(path)
        >>> component.get_format()
        'TURBOMOLE'
        >>> path = files("cosmolayer.data") / "NCCO.cosmo"
        >>> component = Component(path)
        >>> component.get_format()
        'DMol-3'
        """
        return self._format

    def get_sigma_grid(self) -> NDArray[np.float64]:
        """Get the charge density grid in e/Å².

        Returns
        -------
        np.ndarray
            Charge density grid in e/Å².
        """
        return self._grid

    def get_sigma_profile(
        self, segment_class: str | None = None
    ) -> NDArray[np.float64]:
        """Get the sigma profile for a given segment class or the overall sigma profile.

        The segment classes are:
        - NHB: Non-hydrogen-bonding segments
        - OH: Segments associated with hydroxyl groups
        - OT: Segments associated with other hydrogen-bonding groups

        Parameters
        ----------
        segment_class : str, optional
            Segment class. If None, returns the total sigma profile.

        Returns
        -------
        np.ndarray
            Sigma profile for the given segment class or the overall sigma profile.
            Shape: (num_points,). Units: Å².
        """
        if segment_class:
            try:
                profile: NDArray[np.float64] = self._sigma_profiles[
                    segment_class.upper()
                ]
                return profile
            except KeyError as e:
                raise ValueError(f"Invalid segment class: {segment_class}") from e
        total_profile: NDArray[np.float64] = np.sum(
            list(self._sigma_profiles.values()), axis=0
        )
        return total_profile

    def get_probabilities(
        self, merge: bool = False, regularize: float = 1e-10
    ) -> NDArray[np.float64]:
        """Get the probabilities of segment types in the molecule.

        A segment type is defined by its hydrogen bonding class (NHB, OH, OT) and its
        averaged charge density.

        Parameters
        ----------
        merge : bool, optional
            Whether to merge the segment groups (NHB, OH, OT) into a single profile.
            Default is False.
        regularize : float, optional
            Minimum value for clipping probabilities. Set to 0 to disable
            regularization. Default is 1e-10. If clipping occurs, the returned
            distribution is renormalized to sum to 1.

        Returns
        -------
        np.ndarray
            Normalized distribution of segment groups.
            If merge=True: shape is (num_points,) - total sigma profile normalized.
            If merge=False: shape is (3*num_points,) - concatenated profiles
            normalized.

        Examples
        --------
        >>> import numpy as np
        >>> from importlib.resources import files
        >>> from cosmolayer.sac import Component
        >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
        >>> component = Component(path)
        >>> probabilities = component.get_probabilities(merge=True)
        >>> probabilities.shape
        (51,)
        >>> bool(np.all(probabilities <= 1))
        True
        >>> bool(np.isclose(probabilities.sum(), 1.0))
        True
        >>> probabilities_full = component.get_probabilities(merge=False)
        >>> probabilities_full.shape
        (153,)
        >>> bool(np.isclose(probabilities_full.sum(), 1.0))
        True
        """
        if regularize < 0:
            raise ValueError("Regularization value must be non-negative.")
        profiles = [self._sigma_profiles[segtype] for segtype in SEGMENT_GROUPS]
        summed = (
            np.sum(profiles, axis=0) if merge else np.concatenate(profiles)
        ) / self._area
        clipped = summed.clip(min=regularize)
        normalized: NDArray[np.float64] = clipped / clipped.sum()
        return normalized
