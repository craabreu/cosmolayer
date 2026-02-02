"""
.. module:: cosmolayer.sac.interaction_matrices
   :synopsis: Create interaction matrices for COSMO-SAC calculations.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from .segment_groups import OH, OT, SEGMENT_GROUPS

COSMO_SAC_2002_EXPONENTS = (1,)
COSMO_SAC_2010_EXPONENTS = (1, 3)
COSMO_SAC_2002_REFERENCE_AREA = 7.5  # Å²
COSMO_SAC_2010_REFERENCE_AREA = 7.25  # Å²


def create_cosmo_sac_2002_matrix(  # noqa: PLR0913
    temperature: float,  # K
    *,
    min_sigma: float = -0.025,
    max_sigma: float = 0.025,
    num_points: int = 51,
    sigma_hb: float = 0.0084,  # e/Å²
    alpha_prime: float = 16466.72,  # (kcal/mol)/(e/Å²)²
    c_hb: float = 85580.0,  # (kcal/mol)/(e/Å²)²
    gas_constant: float = 0.001987,  # kcal/(mol·K)
) -> NDArray[np.float64]:
    r"""Create an interaction matrix for the COSMO-SAC 2002 model :cite:`Bell2020`.

    Computes the pairwise interaction energies between surface segments with given
    screening charge densities, ΔW(σ,σ'), divided by the product RT, where R is the
    universal gas constant and T is the temperature at which the interaction matrix
    is computed.

    Parameters
    ----------
    temperature : float
        The temperature in Kelvin at which the interaction matrix is computed.

    Keyword Arguments
    -----------------
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
    gas_constant : float, optional
        Universal gas constant in kcal/(mol·K). Default is 0.001987 :cite:`Bell2020`.

    Returns
    -------
    np.ndarray
        Dimensionless interaction energy matrix ΔW(σ,σ') / (RT).
        Shape: (num_points, num_points).

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> T = 298.15  # K
    >>> matrix = create_cosmo_sac_2002_matrix(T)
    >>> matrix.shape
    (51, 51)
    >>> print(np.all(np.isfinite(matrix)))
    True
    >>> print(matrix.min() < 0)  # H-bonding can be favorable (negative)
    True
    >>> print(matrix.max() > 0)  # Misfit interactions are unfavorable
    True

    Plotting the interaction matrix:

    .. plot::
        :context: close-figs

        >>> from cosmolayer.sac import create_cosmo_sac_2002_matrix
        >>> from matplotlib import pyplot as plt
        >>> matrix = create_cosmo_sac_2002_matrix(298.15)
        >>> fig, ax = plt.subplots(figsize=(8, 6))
        >>> im = ax.imshow(matrix, cmap="Spectral")
        >>> _ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
        >>> fig.tight_layout()
    """

    grid = np.linspace(min_sigma, max_sigma, num_points)
    squared_sum_block = np.add.outer(grid, grid) ** 2
    delta = (grid - sigma_hb).clip(min=0) + (grid + sigma_hb).clip(max=0)
    hb_block = np.outer(delta, delta).clip(max=0)
    energy_matrix = (alpha_prime / 2) * squared_sum_block + c_hb * hb_block
    result: NDArray[np.float64] = energy_matrix / (gas_constant * temperature)
    return result


def create_cosmo_sac_2010_matrices(  # noqa: PLR0913
    temperature: float,  # K
    *,
    min_sigma: float = -0.025,
    max_sigma: float = 0.025,
    num_points: int = 51,
    a_es: float = 6525.69,  # (kcal/mol)/(e/Å²)²
    b_es: float = 1.4859e8,  # (kcal/mol)K²/(e/Å²)²
    c_oh_oh: float = 4013.78,  # kcal·Å^4·mol⁻¹·e⁻²
    c_ot_ot: float = 932.31,  # kcal·Å^4·mol⁻¹·e⁻²
    c_oh_ot: float = 3016.43,  # kcal·Å^4·mol⁻¹·e⁻²
    gas_constant: float = 0.0019872043011606513,  # kcal/(mol·K)
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Create interaction matrices for the COSMO-SAC 2010 model :cite:`Bell2020`.

    Computes the electrostatic and hydrogen bonding parts of pairwise interaction
    energies between surface segments with given screening charge densities, ΔW(σ,σ'),
    divided by the product RT, where R is the universal gas constant and T is the
    temperature at which the interaction matrix is computed.

    Parameters
    ----------
    temperature : float
        The temperature in Kelvin at which the interaction matrix is computed.

    Keyword Arguments
    -----------------
    min_sigma : float, optional
        Minimum screening charge density in e/Å². Default is -0.025.
    max_sigma : float, optional
        Maximum screening charge density in e/Å². Default is 0.025.
    num_points : int, optional
        Number of discrete points in the sigma grid. Default is 51.
    a_es : float, optional
        Misfit energy constant in (kcal/mol)/(e/Å²)². Controls the strength
        of electrostatic misfit interactions. Default is 6525.69 :cite:`Bell2020`.
    b_es : float, optional
        Misfit energy constant in (kcal/mol)/(e/Å²)². Controls the strength
        of electrostatic misfit interactions. Default is 1.4859e8 :cite:`Bell2020`.
    c_oh_oh : float, optional
        Hydrogen bonding energy constant in (kcal/mol)/(e/Å²)². Controls the
        strength of hydrogen bonding interactions. Default is 4013.78 :cite:`Bell2020`.
    c_ot_ot : float, optional
        Hydrogen bonding energy constant in (kcal/mol)/(e/Å²)². Controls the
        strength of hydrogen bonding interactions. Default is 932.31 :cite:`Bell2020`.
    c_oh_ot : float, optional
        Hydrogen bonding energy constant in (kcal/mol)/(e/Å²)². Controls the
        strength of hydrogen bonding interactions. Default is 3016.43 :cite:`Bell2020`.
    gas_constant : float, optional
        Universal gas constant in kcal/(mol·K). Default is 0.0019872043011606513
        :cite:`Bell2020`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Dimensionless interaction energy matrices ΔW(σ,σ') / (RT).
        Shape: (num_points, num_points).

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> T = 298.15  # K
    >>> delta_w_a, delta_w_b = create_cosmo_sac_2010_matrices(T)

    Plotting the interaction matrix:

    .. plot::
        :context: close-figs

        >>> from cosmolayer.sac import create_cosmo_sac_2010_matrices
        >>> from matplotlib import pyplot as plt
        >>> delta_w_a, delta_w_b = create_cosmo_sac_2010_matrices(298.15)
        >>> fig, ax = plt.subplots(figsize=(8, 6))
        >>> im = ax.imshow(delta_w_a + delta_w_b, cmap="Spectral")
        >>> _ = fig.colorbar(im, ax=ax, label="ΔW/(RT)")
        >>> fig.tight_layout()
    """
    RT = gas_constant * temperature
    c_hb: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    c_hb[OH][OH] = c_oh_oh
    c_hb[OT][OT] = c_ot_ot
    c_hb[OH][OT] = c_hb[OT][OH] = c_oh_ot
    grid = np.linspace(min_sigma, max_sigma, num_points)

    es_block = np.add.outer(grid, grid) ** 2 / RT
    hb_block = (np.outer(grid, grid) < 0) * np.subtract.outer(grid, grid) ** 2 / RT

    es_matrix = np.block([[es_block] * 3] * 3)
    hb_matrix = np.block(
        [[c_hb[s][t] * hb_block for t in SEGMENT_GROUPS] for s in SEGMENT_GROUPS]
    )

    result_a: NDArray[np.float64] = a_es * es_matrix - hb_matrix
    result_b: NDArray[np.float64] = (b_es / temperature**2) * es_matrix
    return result_a, result_b
