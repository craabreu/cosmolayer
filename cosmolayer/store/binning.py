"""Linear binning of segment- and atom-level data onto a ``SigmaGrid``.

Every accumulation function here uses the same two-tap linear
interpolation: a value is split between the two nearest grid points in
proportion to its distance from each, so mass is conserved exactly except
for values outside the grid's range, which are folded into the nearest
boundary point.
"""

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .grid import SigmaGrid

_Float = TypeVar("_Float", bound=np.floating[Any])


def row_indices_from_offsets(
    offsets: NDArray[np.int64], total_num_rows: int
) -> NDArray[np.int64]:
    """Expand a table of row-start offsets into a row index per element.

    Parameters
    ----------
    offsets : np.ndarray
        Start index of each row's elements within the element-level
        arrays. The last row's elements are assumed to run to
        ``total_num_rows``.
    total_num_rows : int
        Total number of elements described by ``offsets``.

    Returns
    -------
    np.ndarray
        Row index of every element, shape ``(total_num_rows,)``.

    Examples
    --------
    >>> row_indices_from_offsets(np.array([0, 2, 3]), 5)
    array([0, 0, 1, 2, 2])
    """
    return np.repeat(
        np.arange(len(offsets), dtype=np.int64),
        np.diff(np.append(offsets, total_num_rows)),
    )


def compute_per_atom_properties(
    properties: NDArray[_Float],
    atom_indices: NDArray[np.int64],
    total_num_atoms: int,
) -> NDArray[_Float]:
    """Sum a segment-level property into per-atom totals.

    Parameters
    ----------
    properties : np.ndarray
        Segment-level values to sum, e.g. charges or areas.
    atom_indices : np.ndarray
        Atom indices for the segments.
    total_num_atoms : int
        Total number of atoms in the dataset.

    Returns
    -------
    np.ndarray
        Per-atom sum of ``properties``, of shape ``(total_num_atoms,)``.
    """
    atom_properties = np.zeros(total_num_atoms, dtype=properties.dtype)
    np.add.at(atom_properties, atom_indices, properties)
    return atom_properties


def compute_per_molecule_properties(
    properties: NDArray[_Float], atom_offsets: NDArray[np.int64]
) -> NDArray[_Float]:
    """Compute the per-molecule sum of an atom-level property.

    Parameters
    ----------
    properties : np.ndarray
        Atom-level property values, of shape ``(total_num_atoms,)``.
    atom_offsets : np.ndarray
        Global index of each molecule's first atom (cumulative sum of
        ``num_atoms`` for preceding molecules) -- *not* the
        ``segment_offsets`` column of the ``molecules`` table, which
        indexes segment-level arrays instead.

    Returns
    -------
    np.ndarray
        Per-molecule sum of ``properties``, of shape ``(n_molecules,)``.
    """
    return np.add.reduceat(properties, atom_offsets)


@dataclass
class AtomProfileAccumulator:
    """The three per-atom arrays ``accumulate_atom_profiles`` accumulates
    into.

    Grouped into one object because every call site needs all three
    together, and because keeping them as one parameter (instead of three)
    keeps ``accumulate_atom_profiles``'s signature within this package's
    argument-count limit.

    Parameters
    ----------
    areas : np.ndarray
        Per-atom areas, shape ``(num_atoms,)``. Modified in place by
        ``accumulate_atom_profiles``.
    charges : np.ndarray
        Per-atom charges, shape ``(num_atoms,)``. Modified in place.
    profiles : np.ndarray
        Per-atom sigma profiles, shape ``(num_atoms, num_points)``.
        Modified in place.
    """

    areas: NDArray[np.float32]
    charges: NDArray[np.float32]
    profiles: NDArray[np.float32]


def accumulate_atom_profiles(
    accumulator: AtomProfileAccumulator,
    sigmas: NDArray[np.float64],
    areas: NDArray[np.float64],
    atom_indices: NDArray[np.int64],
    grid: SigmaGrid,
    centered: bool,
) -> None:
    """Accumulate a batch of segments' area into per-atom sigma profiles.

    Each segment's charge density is linearly interpolated between the
    two nearest grid points, and its area split between those points
    accordingly. Charge densities outside ``grid``'s range are folded
    into the nearest boundary point. Each atom's profile ends up summing
    to 1 (all-zero for atoms with no surface segments).

    Takes each segment's charge density directly, as ``sigmas``, rather
    than deriving it from ``charge / area`` -- pass raw (``charges /
    areas``) or an averaged density from ``average_sigmas_by_molecule``.
    ``accumulator.charges`` accumulates ``sigmas * areas``, the true net
    charge when ``sigmas`` is raw.

    Safe to call concurrently on disjoint segment ranges split on
    *molecule* boundaries: every segment of an atom must land in the same
    batch, since ``accumulator.areas``/``.charges`` are read back mid-call
    to normalize.

    Parameters
    ----------
    accumulator : AtomProfileAccumulator
        Per-atom areas, charges, and profiles to accumulate into. Modified
        in place.
    sigmas : np.ndarray
        Charge density of the surface segments in this batch, in e/Å² --
        raw or averaged, at the caller's choice. Not modified in place.
    areas : np.ndarray
        Areas of the surface segments in this batch.
    atom_indices : np.ndarray
        Global atom index associated with each segment in this batch.
    grid : SigmaGrid
        Grid to bin onto. Pass ``grid.centered()`` here (not ``grid``
        itself) when ``centered=True`` -- this function does not derive
        it.
    centered : bool
        Whether to center each atom's profile on its own mean charge
        density ``q_a / A_a`` before binning, so its first moment is
        zero.

    Returns
    -------
    None
        ``accumulator``'s arrays are updated in place.
    """
    atom_areas, atom_charges, sigma_profiles = (
        accumulator.areas,
        accumulator.charges,
        accumulator.profiles,
    )
    np.add.at(atom_areas, atom_indices, areas)
    np.add.at(atom_charges, atom_indices, sigmas * areas)

    num_points = len(grid)
    summed_areas = atom_areas[atom_indices]

    if centered:
        sigmas = sigmas - atom_charges[atom_indices] / summed_areas

    fractional_bins = (sigmas - (-grid.max_abs_sigma)) / grid.bin_width

    points_at_left = np.floor(fractional_bins).astype(int)
    points_at_right = points_at_left + 1

    normalized_areas = areas / summed_areas
    contributions_at_left = normalized_areas * (points_at_right - fractional_bins)
    contributions_at_right = normalized_areas * (fractional_bins - points_at_left)

    np.add.at(
        sigma_profiles,
        (atom_indices, points_at_left.clip(0, num_points - 1)),
        contributions_at_left,
    )
    np.add.at(
        sigma_profiles,
        (atom_indices, points_at_right.clip(0, num_points - 1)),
        contributions_at_right,
    )


@dataclass
class AtomTranslationBatch:
    """One batch of atoms' profiles, ready to be translated and summed
    onto a molecule grid by ``accumulate_translated_profiles``.

    Grouped into one object because these four values always travel
    together (they describe the same atoms), and because keeping them as
    one parameter keeps ``accumulate_translated_profiles``'s signature
    within this package's argument-count limit.

    Parameters
    ----------
    areas : np.ndarray
        Areas of the atoms in this batch, shape ``(n,)``.
    translations : np.ndarray
        Amount to translate each atom's profile by, in sigma units
        (positive values move mass toward the positive end of the grid).
        Pass zeros to accumulate profiles untranslated.
    profiles : np.ndarray
        Per-atom sigma profiles for the atoms in this batch, shape
        ``(n, len(grid))``.
    grid : SigmaGrid
        Grid ``profiles`` are binned on.
    """

    areas: NDArray[np.float32]
    translations: NDArray[np.float64]
    profiles: NDArray[np.float32]
    grid: SigmaGrid


def accumulate_translated_profiles(
    molecule_profiles: NDArray[np.float64],
    molecule_indices: NDArray[np.int64],
    batch: AtomTranslationBatch,
    molecule_grid: SigmaGrid,
) -> None:
    """Accumulate a batch of atoms' translated, area-weighted profiles
    into per-molecule sigma profiles.

    The molecule grid shares the atom grid's ``bin_width`` (callers derive
    it via ``SigmaGrid.centered`` on a common base grid), so a zero
    translation places atom column ``k`` at molecule column
    ``k + grid_offset``, where
    ``grid_offset = (len(molecule_grid) - len(batch.grid)) / 2`` -- an
    integer when the point counts share parity, a half-integer otherwise.
    Each row is redistributed with the same two-tap linear interpolation
    ``accumulate_atom_profiles`` uses for a single value, with
    out-of-range destinations folded into the nearest boundary column.

    Safe to call concurrently on disjoint atom ranges as long as no two
    calls share a molecule.

    Parameters
    ----------
    molecule_profiles : np.ndarray
        Per-molecule sigma profiles to accumulate into, of shape
        ``(num_molecules, len(molecule_grid))``. Modified in place.
    molecule_indices : np.ndarray
        Global molecule index associated with each atom in this batch.
    batch : AtomTranslationBatch
        The atoms' areas, translations, and profiles to accumulate.
    molecule_grid : SigmaGrid
        Grid to bin the output ``molecule_profiles`` onto.

    Returns
    -------
    None
        ``molecule_profiles`` is updated in place.
    """
    atom_areas, translations, atom_profiles, atom_grid = (
        batch.areas,
        batch.translations,
        batch.profiles,
        batch.grid,
    )
    bin_width = atom_grid.bin_width
    num_points = len(atom_grid)
    molecule_num_points = len(molecule_grid)
    grid_offset = (molecule_num_points - num_points) / 2

    fractional_translation = grid_offset + translations / bin_width
    points_translation = np.floor(fractional_translation).astype(np.int64)
    weight_right = (fractional_translation - points_translation)[:, None]

    contributions = atom_areas[:, None].astype(np.float64) * atom_profiles.astype(
        np.float64
    )
    points_at_left = np.arange(num_points)[None, :] + points_translation[:, None]
    points_at_right = points_at_left + 1

    np.add.at(
        molecule_profiles,
        (
            molecule_indices[:, None],
            points_at_left.clip(0, molecule_num_points - 1),
        ),
        contributions * (1.0 - weight_right),
    )
    np.add.at(
        molecule_profiles,
        (
            molecule_indices[:, None],
            points_at_right.clip(0, molecule_num_points - 1),
        ),
        contributions * weight_right,
    )
