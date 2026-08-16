"""The symmetric, evenly spaced grid a sigma profile is binned onto.

A sigma profile is a histogram of surface area over charge density
(sigma, in e/Å²). Profiles are always binned onto the ``SigmaGrid``
passed in.

Two operations are named separately because both are sometimes called
"shifting":

- *centering* subtracts a row's mean charge density from its segments
  **before** binning, so the profile has zero first moment. It changes
  the data, not the grid (``SigmaProfileTable.centered``).
- *translating* moves an already-binned row along a grid, used when
  un-centering atom profiles while summing them to molecule level.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

DEFAULT_MAX_ABS_SIGMA = 0.025
DEFAULT_NUM_POINTS = 51


@dataclass(frozen=True)
class SigmaGrid:
    """A symmetric, evenly spaced grid of sigma-profile points.

    Parameters
    ----------
    max_abs_sigma : float, optional
        Bounded sigma-profile value at each end of the grid, in e/Å², by
        default ``DEFAULT_MAX_ABS_SIGMA``.
    num_points : int, optional
        Number of grid points, by default ``DEFAULT_NUM_POINTS``. An odd
        count puts a point at sigma = 0; an even count straddles it.

    Examples
    --------
    >>> grid = SigmaGrid(max_abs_sigma=0.025, num_points=51)
    >>> round(grid.bin_width, 6)
    0.001
    >>> len(grid)
    51
    >>> grid.values[0], grid.values[-1]
    (np.float64(-0.025), np.float64(0.025))
    """

    max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA
    num_points: int = DEFAULT_NUM_POINTS

    @property
    def bin_width(self) -> float:
        """Width of one bin, in e/Å².

        Returns
        -------
        float
            ``2 * max_abs_sigma / (num_points - 1)``.
        """
        return (2.0 * self.max_abs_sigma) / (self.num_points - 1)

    @cached_property
    def values(self) -> NDArray[np.float64]:
        """Sigma value at every grid point, in e/Å².

        Returns
        -------
        np.ndarray, shape (num_points,)
            Evenly spaced values from ``-max_abs_sigma`` to
            ``max_abs_sigma``.
        """
        return np.linspace(-self.max_abs_sigma, self.max_abs_sigma, self.num_points)

    @classmethod
    def from_values(cls, values: NDArray[np.float64]) -> "SigmaGrid":
        """Reconstruct a :class:`SigmaGrid` from an existing array of grid
        values.

        Parameters
        ----------
        values : np.ndarray
            Evenly spaced, symmetric grid values, as returned by
            ``self.values``.

        Returns
        -------
        SigmaGrid
            A grid whose ``.values`` reproduce ``values``.

        Examples
        --------
        >>> grid = SigmaGrid(0.025, 51)
        >>> SigmaGrid.from_values(grid.values) == grid
        True
        """
        return cls(float(values[-1]), len(values))

    def __len__(self) -> int:
        return self.num_points


DEFAULT_SIGMA_GRID = SigmaGrid()
"""Default 51-point grid from -0.025 to 0.025 e/Å²."""
