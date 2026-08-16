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
    """Per-atom areas, charges, and profiles updated in place by
    ``accumulate_atom_profiles``.

    Parameters
    ----------
    areas : np.ndarray
        Per-atom areas, shape ``(num_atoms,)``.
    charges : np.ndarray
        Per-atom charges, shape ``(num_atoms,)``.
    profiles : np.ndarray
        Per-atom sigma profiles, shape ``(num_atoms, num_points)``.
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
    """Accumulate a batch of segments into per-atom sigma profiles.

    Each segment's charge density is linearly interpolated between the
    two nearest grid points, and its area split between those points.
    Values outside ``grid`` fold into the nearest boundary. Each atom's
    profile sums to 1 (all-zero if it has no surface segments).

    ``sigmas`` is charge density (raw ``charges / areas``, or averaged).
    ``accumulator.charges`` accumulates ``sigmas * areas``.

    Safe concurrently on disjoint ranges split on *molecule* boundaries:
    every segment of an atom must land in the same batch.

    Parameters
    ----------
    accumulator : AtomProfileAccumulator
        Per-atom arrays to update in place.
    sigmas : np.ndarray
        Segment charge density in this batch, in e/Å².
    areas : np.ndarray
        Segment areas in this batch.
    atom_indices : np.ndarray
        Global atom index of each segment in this batch.
    grid : SigmaGrid
        Grid to bin onto.
    centered : bool
        If True, center each atom's profile on its mean charge density
        before binning.
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
    """Atom profiles to translate and sum onto a molecule grid.

    Parameters
    ----------
    areas : np.ndarray
        Areas of the atoms in this batch, shape ``(n,)``.
    translations : np.ndarray
        Shift of each atom's profile, in sigma units (positive toward
        the positive end of the grid). Pass zeros to leave untranslated.
    profiles : np.ndarray
        Per-atom sigma profiles, shape ``(n, len(grid))``.
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
    """Accumulate translated, area-weighted atom profiles into molecules.

    Both grids must be symmetric about zero and share a ``bin_width``.
    Zero translation places atom column ``k`` at molecule column
    ``k + (len(molecule_grid) - len(batch.grid)) / 2``. Out-of-range
    destinations fold into the nearest boundary column.

    Safe concurrently on disjoint atom ranges that share no molecule.

    Parameters
    ----------
    molecule_profiles : np.ndarray
        Per-molecule profiles to update in place, shape
        ``(num_molecules, len(molecule_grid))``.
    molecule_indices : np.ndarray
        Global molecule index of each atom in this batch.
    batch : AtomTranslationBatch
        Atom areas, translations, and profiles to accumulate.
    molecule_grid : SigmaGrid
        Grid for the output ``molecule_profiles``.
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
