"""Area-weighted sigma profiles, at atom or molecule level."""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .binning import (
    AtomProfileAccumulator,
    AtomTranslationBatch,
    accumulate_atom_profiles,
    accumulate_translated_profiles,
    compute_per_molecule_properties,
    row_indices_from_offsets,
)
from .grid import DEFAULT_SIGMA_GRID, SigmaGrid
from .parallel import run_in_threads


@dataclass(frozen=True)
class SigmaProfileTable:
    """Area-weighted sigma profiles, at atom or molecule level.

    Build with ``from_segments`` (bin segment-level data) or ``aggregate``
    (atom-level table to molecule-level). Direct construction wraps
    already-computed arrays.

    Parameters
    ----------
    areas : np.ndarray
        Per-row area, shape ``(n,)``.
    charges : np.ndarray
        Per-row net (or smoothed equivalent) charge, shape ``(n,)``.
    profiles : np.ndarray
        Per-row area-fraction sigma profile, shape ``(n, len(grid))``.
    grid : SigmaGrid
        Grid these profiles are binned on.
    centered : bool
        Whether each row was centered on its mean charge density before
        binning (zero first moment). A property of the data, not the grid.
    atom_offsets : np.ndarray | None, optional
        Global index of each molecule's first atom. ``None`` (default)
        means molecule-level; when set, the table is atom-level and
        ``aggregate`` can reassemble it.
    """

    areas: NDArray[np.float32]
    charges: NDArray[np.float32]
    profiles: NDArray[np.float32]
    grid: SigmaGrid
    centered: bool
    atom_offsets: NDArray[np.int64] | None = None

    @property
    def level(self) -> Literal["atom", "molecule"]:
        """Whether each row of this table describes one atom or one
        molecule.

        Returns
        -------
        Literal["atom", "molecule"]
            ``"atom"`` if ``atom_offsets`` is set, else ``"molecule"``.
        """
        return "molecule" if self.atom_offsets is None else "atom"

    @property
    def sigma_values(self) -> NDArray[np.float64]:
        """Sigma value at every profile column, in e/Å².

        Returns
        -------
        np.ndarray, shape (len(grid),)
            ``grid.values``.
        """
        return self.grid.values

    @classmethod
    def from_segments(  # noqa: PLR0913
        cls,
        sigmas: NDArray[np.float64],
        areas: NDArray[np.float64],
        segment_offsets: NDArray[np.int64],
        *,
        atom_indices: NDArray[np.int64] | None = None,
        num_rows: int | None = None,
        grid: SigmaGrid = DEFAULT_SIGMA_GRID,
        centered: bool = False,
        num_threads: int | None = None,
    ) -> "SigmaProfileTable":
        """Bin segment-level data into per-atom or per-molecule sigma
        profiles.

        ``sigmas`` is each segment's charge density: pass raw
        ``charges / areas``, or an averaged density from
        ``average_sigmas_by_molecule``.

        Parameters
        ----------
        sigmas : np.ndarray
            Segment charge density, in e/Å² (raw or averaged).
        areas : np.ndarray
            Segment areas.
        segment_offsets : np.ndarray
            Start index of each molecule's segments. Must describe
            *exactly* the molecules present in the segment-level arrays;
            slice offsets and arrays together.
        atom_indices : np.ndarray | None, optional
            Global atom index of each segment. ``None`` (default) builds
            one profile per molecule. When given, builds one profile per
            atom; the result's ``atom_offsets`` is
            ``atom_indices[segment_offsets]``.
        num_rows : int | None, optional
            Number of profile rows. ``None`` (default) uses
            ``len(segment_offsets)`` (molecule level) or
            ``int(atom_indices.max()) + 1`` (atom level). Pass explicitly
            when subsetting.
        grid : SigmaGrid, optional
            Grid to bin onto, by default ``DEFAULT_SIGMA_GRID``.
        centered : bool, optional
            If True, center each profile on its mean charge density
            before binning.
        num_threads : int | None, optional
            Thread count. ``None`` (default) uses every CPU core.

        Returns
        -------
        SigmaProfileTable
            Atom-level if ``atom_indices`` is given, else molecule-level.
        """
        sigmas = np.asarray(sigmas)
        areas = np.asarray(areas)
        segment_offsets = np.asarray(segment_offsets)

        if atom_indices is None:
            row_indices = row_indices_from_offsets(segment_offsets, len(sigmas))
            num_rows = len(segment_offsets) if num_rows is None else num_rows
            atom_offsets = None
        else:
            row_indices = np.asarray(atom_indices)
            num_rows = int(np.max(row_indices)) + 1 if num_rows is None else num_rows
            atom_offsets = row_indices[segment_offsets].astype(np.int64)

        areas_out = np.zeros(num_rows, dtype=np.float32)
        charges_out = np.zeros(num_rows, dtype=np.float32)
        profiles_out = np.zeros((num_rows, len(grid)), dtype=np.float32)
        assert int(np.max(row_indices, initial=-1)) < num_rows, (
            "atom_indices/segment_offsets reference a row index >= num_rows; "
            "segment_offsets must describe exactly the molecules present in "
            "the segment-level arrays"
        )

        num_segs = len(sigmas)
        num_mols = len(segment_offsets)
        accumulator = AtomProfileAccumulator(areas_out, charges_out, profiles_out)

        def process_range(start_mol: int, stop_mol: int) -> None:
            start_seg = segment_offsets[start_mol]
            stop_seg = segment_offsets[stop_mol] if stop_mol < num_mols else num_segs
            accumulate_atom_profiles(
                accumulator,
                sigmas[start_seg:stop_seg],
                areas[start_seg:stop_seg],
                row_indices[start_seg:stop_seg],
                grid,
                centered,
            )

        run_in_threads(process_range, num_mols, num_threads=num_threads)

        return cls(
            areas_out,
            charges_out,
            profiles_out,
            grid,
            centered,
            atom_offsets=atom_offsets,
        )

    def aggregate(
        self,
        *,
        grid: SigmaGrid | None = None,
        normalize: bool = False,
        num_threads: int | None = None,
    ) -> "SigmaProfileTable":
        """Reassemble per-molecule profiles from these per-atom ones.

        Area-weighted sum of atom profiles onto a shared molecule axis.
        If this table is centered, each atom is translated back by its
        mean charge density first. The result is always uncentered.

        Parameters
        ----------
        grid : SigmaGrid | None, optional
            Grid for the molecule profiles. ``None`` (default) uses
            ``self.grid``. A different grid must share this table's
            ``bin_width``.
        normalize : bool, optional
            If True, divide each molecule's profile by its total area so
            it sums to 1.
        num_threads : int | None, optional
            Thread count. ``None`` (default) uses every CPU core.

        Returns
        -------
        SigmaProfileTable
            Molecule-level (``atom_offsets`` is None, ``centered`` False).

        Raises
        ------
        ValueError
            If this table is already molecule-level, or if ``grid`` does
            not share this table's ``bin_width``.
        """
        if self.atom_offsets is None:
            raise ValueError(
                "This SigmaProfileTable has no atom_offsets -- it is "
                "already at molecule level, so there is nothing to "
                "aggregate."
            )
        output_grid = self.grid if grid is None else grid
        if not math.isclose(output_grid.bin_width, self.grid.bin_width, rel_tol=1e-9):
            raise ValueError(
                f"Output grid bin width {output_grid.bin_width!r} does not "
                f"match this table's {self.grid.bin_width!r}. Aggregation "
                "aligns columns by point-count difference alone, which is "
                "only valid between symmetric grids of equal bin width."
            )

        num_atoms = len(self.areas)
        num_mols = len(self.atom_offsets)
        molecule_num_points = len(output_grid)

        molecule_indices = row_indices_from_offsets(self.atom_offsets, num_atoms)

        translations = np.zeros(num_atoms, dtype=np.float64)
        if self.centered:
            has_area = self.areas > 0
            translations[has_area] = self.charges[has_area] / self.areas[has_area]

        molecule_profiles = np.zeros((num_mols, molecule_num_points), dtype=np.float64)
        atom_offsets = self.atom_offsets

        def process_range(start_mol: int, stop_mol: int) -> None:
            start_atom = atom_offsets[start_mol]
            stop_atom = atom_offsets[stop_mol] if stop_mol < num_mols else num_atoms
            batch = AtomTranslationBatch(
                self.areas[start_atom:stop_atom],
                translations[start_atom:stop_atom],
                self.profiles[start_atom:stop_atom],
                self.grid,
            )
            accumulate_translated_profiles(
                molecule_profiles,
                molecule_indices[start_atom:stop_atom],
                batch,
                output_grid,
            )

        run_in_threads(process_range, num_mols, num_threads=num_threads)

        if normalize:
            molecule_profiles = molecule_profiles / molecule_profiles.sum(
                axis=1, keepdims=True
            )

        molecule_areas = compute_per_molecule_properties(self.areas, atom_offsets)
        molecule_charges = compute_per_molecule_properties(self.charges, atom_offsets)

        return replace(
            self,
            areas=molecule_areas,
            charges=molecule_charges,
            profiles=molecule_profiles.astype(np.float32),
            grid=output_grid,
            centered=False,
            atom_offsets=None,
        )
