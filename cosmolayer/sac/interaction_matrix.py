"""
.. module:: cosmolayer.sac.interaction_matrix
   :synopsis: Create interaction matrix for COSMO-SAC activity coefficient calculations.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import numpy as np
from scipy import constants as spc

GAS_CONSTANT = spc.gas_constant / (spc.kilo * spc.calorie)  # kcal/(mol·K)


def create_cosmo_sac_2002_matrix(  # noqa: PLR0913
    reference_temperature: float = 298.15,  # K
    min_sigma: float = -0.025,
    max_sigma: float = 0.025,
    num_points: int = 51,
    sigma_hb: float = 0.0084,  # e/Å²
    alpha_prime: float = 16466.72,  # (kcal/mol)/(e/Å²)²
    c_hb: float = 85580.0,  # (kcal/mol)/(e/Å²)²
) -> np.ndarray:
    r"""Create an interaction matrix for the COSMO-SAC 2002 model :cite:`Bell2020`.

    Computes the pairwise interaction energies between surface segments with given
    screening charge densities, ΔW(σ,σ'), divided by the product RT₀, where R is the
    universal gas constant and T₀ is the reference temperature.

    Parameters
    ----------
    reference_temperature : float, optional
        Reference temperature in K. Default is 298.15.
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

    Returns
    -------
    np.ndarray
        Dimensionless interaction energy matrix ΔW(σ,σ') / (RT₀).
        Shape: (num_points, num_points).

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> matrix = create_cosmo_sac_2002_matrix()
    >>> matrix.shape
    (51, 51)
    >>> np.all(np.isfinite(matrix))
    True
    >>> matrix.min() < 0  # H-bonding can be favorable (negative)
    True
    >>> matrix.max() > 0  # Misfit interactions are unfavorable
    True

    Plotting the interaction matrix:

    >>> fig, ax = plt.subplots(figsize=(8, 6))  # doctest: +SKIP
    >>> im = ax.imshow(matrix, cmap="Spectral", origin="lower")  # doctest: +SKIP
    >>> fig.colorbar(im, ax=ax, label="ΔW/(RT₀)")  # doctest: +SKIP
    >>> fig.tight_layout()  # doctest: +SKIP

    .. plot::
        :context: close-figs

        >>> from cosmolayer.sac import create_cosmo_sac_2002_matrix
        >>> from matplotlib import pyplot as plt
        >>> matrix = create_cosmo_sac_2002_matrix()
        >>> fig, ax = plt.subplots(figsize=(8, 6))
        >>> im = ax.imshow(matrix, cmap="Spectral", origin="lower")
        >>> cbar = fig.colorbar(im, ax=ax, label="ΔW/(RT₀)")
        >>> fig.tight_layout()
    """

    grid = np.linspace(min_sigma, max_sigma, num_points)
    squared_sum_block = np.add.outer(grid, grid) ** 2
    delta = (grid - sigma_hb).clip(min=0) + (grid + sigma_hb).clip(max=0)
    hb_block = np.outer(delta, delta).clip(max=0)
    energy_matrix = (alpha_prime / 2) * squared_sum_block + c_hb * hb_block
    return energy_matrix / (GAS_CONSTANT * reference_temperature)
