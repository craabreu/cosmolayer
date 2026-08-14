"""The symmetric, evenly spaced grid a sigma profile is binned onto.

Every sigma profile in this package -- per-segment, per-atom, or
per-molecule -- is a histogram of surface area over charge density
(sigma, in e/Å²), binned linearly onto a :class:`SigmaGrid`. Two distinct
operations both get called "shifting" elsewhere in this domain, so this
package is deliberately careful to keep them apart:

- *centering* a profile: subtracting a row's own mean charge density from
  its segments before binning, so the resulting profile has zero first
  moment. This is a per-row transform of the *data*, tracked as a
  ``centered`` flag alongside a table of profiles (e.g.
  ``SigmaProfileTable.centered``).
- *translating* a grid: :meth:`SigmaGrid.for_centered_profiles` derives a
  grid with no point sitting exactly at sigma = 0, gaining one point when
  the original point count is odd. This is a transform of the *grid* --
  it does not touch any data, and does not by itself mean anything was
  centered (an even-point-count grid is its own result either way, so the
  grid alone can't record whether centering happened; hence the separate
  flag above).

A centered profile is binned onto the grid :meth:`SigmaGrid.for_centered_profiles`
returns, because a value can land exactly at zero once each row's own mean
has been removed, and no grid point should coincide with the "true zero" a
centered profile now represents as signed deviation. An uncentered profile
is binned directly onto its base grid.
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
        Number of grid points, by default ``DEFAULT_NUM_POINTS``. May be
        even (no point exactly at zero) or odd (a point exactly at zero).

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

    def for_centered_profiles(self) -> "SigmaGrid":
        """The grid a centered profile should be binned onto.

        Named for its purpose, not its mechanism, to avoid colliding with
        the unrelated ``centered`` flag tables of profiles carry (see the
        module docstring) -- this method never inspects or sets that flag,
        it only derives a grid variant.

        If ``num_points`` is even, returns ``self`` unchanged -- an even
        grid never has a point at exactly zero. If odd, returns a grid
        with one more point, extended by half a bin width on each side,
        which preserves ``bin_width`` exactly. Either way, the result
        never has a point at exactly zero.

        Returns
        -------
        SigmaGrid
            The grid variant with no point at exactly zero.

        Examples
        --------
        >>> centered = SigmaGrid(0.025, 51).for_centered_profiles()
        >>> centered.num_points, round(centered.max_abs_sigma, 6)
        (52, 0.0255)
        >>> SigmaGrid(0.025, 50).for_centered_profiles()
        SigmaGrid(max_abs_sigma=0.025, num_points=50)
        """
        if self.num_points % 2 == 0:
            return self
        half_bin = self.bin_width / 2.0
        return SigmaGrid(self.max_abs_sigma + half_bin, self.num_points + 1)

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
"""The default grid, usable as a shared immutable default argument value."""
