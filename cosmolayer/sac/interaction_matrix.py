"""
COSMO-SAC Interaction Matrix
"""

import numpy as np


class LinSandlerMatrix:
    """Interaction matrix for COSMO-SAC activity coefficient calculations.

    Computes the pairwise segment-segment interaction energies
    :math:`\\Delta W(\\sigma_i, \\sigma_j)` between surface segments with screening
    charge densities :math:`\\sigma_i` and :math:`\\sigma_j`.
    This matrix is used to calculate segment activity coefficients in the COSMO-SAC
    model :cite:`Bell2020`.

    Parameters
    ----------
    min_sigma : float, optional
        Minimum screening charge density in e/Å². Default is -0.025.
    max_sigma : float, optional
        Maximum screening charge density in e/Å². Default is 0.025.
    num_points : int, optional
        Number of discrete points in the sigma grid. Default is 51.
    sigma_hb : float, optional
        Hydrogen bonding cutoff parameter in e/Å². Defines the range for
        hydrogen bonding interactions. Default is 0.0084 :cite:`Bell2020`.
    alpha_prime : float, optional
        Misfit energy constant in (kcal/mol)/(e/Å²)². Controls the strength
        of electrostatic misfit interactions. Default is 16466.72 :cite:`Bell2020`.
    c_hb : float, optional
        Hydrogen bonding energy constant in (kcal/mol)/(e/Å²)². Controls the
        strength of hydrogen bonding interactions. Default is 85580.0 :cite:`Bell2020`.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = LinSandlerMatrix()
    >>> interaction_matrix = matrix.get_matrix()
    >>> interaction_matrix.shape
    (51, 51)
    >>> np.all(np.isfinite(interaction_matrix))
    True
    >>> interaction_matrix.min() < 0  # H-bonding can be favorable (negative)
    True
    >>> interaction_matrix.max() > 0  # Misfit interactions are unfavorable
    True
    """

    def __init__(
        self,
        min_sigma: float = -0.025,  # e/A^2
        max_sigma: float = 0.025,  # e/A^2
        num_points: int = 51,
        sigma_hb: float = 0.0084,  # e/A^2
        alpha_prime: float = 16466.72,  # (kcal/mol)/(e/A^2)^2
        c_hb: float = 85580.0,  # kcal A^4 / mol/e^2
    ):
        sigma = np.linspace(min_sigma, max_sigma, num_points)
        delta = (sigma - sigma_hb).clip(max=0) + (sigma + sigma_hb).clip(min=0)
        self._matrix = (alpha_prime / 2) * np.add.outer(
            sigma, sigma
        ) ** 2 + c_hb * np.outer(delta, delta).clip(max=0)

    def get_matrix(self) -> np.ndarray:
        """Get the pairwise segment interaction energy matrix.

        Returns
        -------
        np.ndarray
            Interaction energy matrix :math:`\\Delta W(\\sigma_i, \\sigma_j)` in
            kcal/mol. Shape: (num_points, num_points).
        """
        return self._matrix
