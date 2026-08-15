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
    """A set of area-weighted sigma profiles, at atom or molecule level.

    Construct via ``from_segments`` (bins fresh segment-level data) or
    ``aggregate`` (reassembles an atom-level table into a molecule-level
    one). Ordinary ``__init__`` is for wrapping already-computed arrays,
    e.g. ones read back from disk.

    Parameters
    ----------
    areas : np.ndarray
        Per-row area, shape ``(n,)``.
    charges : np.ndarray
        Per-row net (or smoothed "equivalent") charge, shape ``(n,)``.
    profiles : np.ndarray
        Per-row area-fraction sigma profile, shape ``(n, len(grid))``.
    grid : SigmaGrid
        The grid these profiles are binned on -- exactly the one the
        caller asked for, whether or not they are centered.
    centered : bool
        Whether each row's profile was centered on its own mean charge
        density before binning, giving it zero first moment. A property of
        the *data*; it does not affect which grid is used.
    atom_offsets : np.ndarray | None, optional
        Global index of each molecule's first atom, by default None,
        meaning this table is already at molecule level. When given, this
        table is at atom level, and ``aggregate`` can reassemble it into a
        molecule-level table.
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

        Takes each segment's charge density directly, as ``sigmas`` --
        pass raw ``charges / areas``, or an averaged density from
        ``average_sigmas_by_molecule``.

        Parameters
        ----------
        sigmas : np.ndarray
            Charge density of the surface segments, in e/Å² -- raw or
            averaged, at the caller's choice (see above).
        areas : np.ndarray
            Areas of the surface segments.
        segment_offsets : np.ndarray
            Start index of each molecule's segments within the
            segment-level arrays. Must describe *exactly* the molecules
            present in ``areas`` / ``sigmas`` / ``atom_indices``: the last
            molecule's segments run to the end of those arrays. Slicing
            ``segment_offsets`` and the segment-level arrays for a subset
            must be done together, or leftover segments are silently
            attributed to the wrong molecule -- pass a matching,
            under-sized ``num_rows`` to turn that mistake into an
            ``AssertionError`` instead.
        atom_indices : np.ndarray | None, optional
            Global atom index associated with each segment. By default
            None, meaning build one profile *per molecule* instead of per
            atom -- segments are grouped straight by ``segment_offsets``
            (faking one monoatomic "atom" per molecule), and the result's
            ``atom_offsets`` stays None. When given, one profile is built
            *per atom* instead, grouped via ``atom_indices``, and the
            result's ``atom_offsets`` (each molecule's first atom index,
            so ``aggregate`` can later reassemble these into
            molecule-level profiles) is derived as
            ``atom_indices[segment_offsets]`` -- correct because segments
            are grouped by ascending atom index within a molecule (true of
            the COSMO file formats this package parses; verified against a
            real TURBOMOLE file).
        num_rows : int | None, optional
            Total number of profile rows to build. By default None,
            meaning ``len(segment_offsets)`` (molecule level) or
            ``int(atom_indices.max()) + 1`` (atom level) -- correct for a
            full store, but potentially wrong for an already-subsetted one
            (see the ``segment_offsets`` caveat above); pass it explicitly
            when subsetting.
        grid : SigmaGrid, optional
            Grid to bin profiles onto, by default ``DEFAULT_SIGMA_GRID``.
            Used exactly as given, so the result's ``profiles`` always has
            ``len(grid)`` columns regardless of ``centered``.
        centered : bool, optional
            Whether to center each profile on its own mean charge density
            before binning, by default False. Affects the binned values,
            not the grid they land on.
        num_threads : int | None, optional
            Number of threads to use, by default None, meaning every
            available CPU core.

        Returns
        -------
        SigmaProfileTable
            Atom-level if ``atom_indices`` is given, else molecule-level.
            ``charges`` is the true net charge when ``sigmas`` is raw, or
            a smoothed "equivalent charge" when it isn't.
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
            num_rows = int(row_indices.max()) + 1 if num_rows is None else num_rows
            atom_offsets = row_indices[segment_offsets].astype(np.int64)

        areas_out = np.zeros(num_rows, dtype=np.float32)
        charges_out = np.zeros(num_rows, dtype=np.float32)
        profiles_out = np.zeros((num_rows, len(grid)), dtype=np.float32)
        assert int(row_indices.max(initial=-1)) < num_rows, (
            "atom_indices/segment_offsets reference a row index >= num_rows; "
            "segment_offsets must describe exactly the molecules present in "
            "the segment-level arrays (see this method's docstring)"
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

        Area-weighted sum of atom profiles onto a shared molecule axis,
        with each atom's own mean charge density un-translated first when
        ``self.centered`` is True.

        The result is always uncentered (``centered=False``): un-translating
        returns each atom's mass to absolute sigma, so a centered atom table
        aggregates into an uncentered molecule one.

        Parameters
        ----------
        grid : SigmaGrid | None, optional
            Grid for the output molecule profiles, by default None,
            meaning ``self.grid`` (same axis as the atom profiles). A
            different grid must share this table's ``bin_width``: the
            column alignment is ``k -> k + (len(grid) - len(self.grid)) / 2``,
            which is only valid for two symmetric grids of equal bin width.
        normalize : bool, optional
            Whether to divide each molecule's profile by its total area
            so it sums to 1, by default False.
        num_threads : int | None, optional
            Number of threads to use, by default None, meaning every
            available CPU core.

        Returns
        -------
        SigmaProfileTable
            Molecule-level (``atom_offsets is None``, ``centered`` False),
            with ``profiles`` having ``len(grid)`` columns.

        Raises
        ------
        ValueError
            If ``self.atom_offsets`` is None (already at molecule level,
            nothing to aggregate), or if ``grid`` does not share this
            table's ``bin_width``.
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
