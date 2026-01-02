"""
.. module:: cosmolayer.sac.interaction_matrix
   :synopsis: Create interaction matrix for COSMO-SAC activity coefficient calculations.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import numpy as np


def create_lin_sandler_matrix(
    min_sigma: float = -0.025,
    max_sigma: float = 0.025,
    num_points: int = 51,
    sigma_hb: float = 0.0084,
    alpha_prime: float = 16466.72,
    c_hb: float = 85580.0,
) -> np.ndarray:
    r"""Create an interaction matrix for COSMO-SAC activity coefficient calculations.

    Computes the pairwise segment-segment interaction energies
    :math:`\Delta W(\sigma, \sigma')` between surface segments with screening
    charge densities :math:`\sigma` and :math:`\sigma'`.
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

    Returns
    -------
    np.ndarray
        Interaction energy matrix :math:`\Delta W(\sigma_i, \sigma_j)` in
        kcal/mol. Shape: (num_points, num_points).

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> matrix = create_lin_sandler_matrix()
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
    >>> fig.colorbar(im, ax=ax, label="Energy (kcal/mol)")  # doctest: +SKIP
    >>> fig.tight_layout()  # doctest: +SKIP

    .. plot::
        :context: close-figs

        >>> from cosmolayer.sac import create_lin_sandler_matrix
        >>> from matplotlib import pyplot as plt
        >>> matrix = create_lin_sandler_matrix()
        >>> fig, ax = plt.subplots(figsize=(8, 6))
        >>> im = ax.imshow(matrix, cmap="Spectral", origin="lower")
        >>> cbar = fig.colorbar(im, ax=ax, label="Energy (kcal/mol)")
        >>> fig.tight_layout()
    """

    grid = np.linspace(min_sigma, max_sigma, num_points)
    delta = (grid - sigma_hb).clip(min=0) + (grid + sigma_hb).clip(max=0)
    return (alpha_prime / 2) * np.add.outer(grid, grid) ** 2 + c_hb * np.outer(
        delta, delta
    ).clip(max=0)
