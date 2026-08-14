"""Area-weighted sigma profiles, at atom or molecule level."""

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
class BinningSpec:
    """Which grid a table's profiles are binned on, and whether each row
    is centered on its own mean charge density first.

    Grouped into one object (rather than two loose parameters) mainly to
    keep ``from_segments``'s own signature within this package's
    argument-count limit; the two fields are otherwise independent. Mirrors
    ``SigmaProfileTable``'s own ``grid``/``centered`` fields.

    Parameters
    ----------
    grid : SigmaGrid, optional
        Base grid to bin profiles onto, by default ``DEFAULT_SIGMA_GRID``.
    centered : bool, optional
        Whether to center each profile on its own mean charge density
        before binning, by default False.
    """

    grid: SigmaGrid = DEFAULT_SIGMA_GRID
    centered: bool = False


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
        Per-row area-fraction sigma profile, shape
        ``(n, len(binning_grid))``.
    grid : SigmaGrid
        The *base* (uncentered) grid this table's profiles are described
        against. Use ``binning_grid`` for the grid the profile columns are
        actually binned on.
    centered : bool
        Whether each row's profile has been centered on its own mean
        charge density before binning (zero first moment), and is
        therefore binned on ``grid.centered()`` rather than ``grid``
        itself.
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
    def binning_grid(self) -> SigmaGrid:
        """The grid ``profiles``'s columns are actually binned on.

        Returns
        -------
        SigmaGrid
            ``grid.centered()`` if ``centered``, else ``grid``.
        """
        return self.grid.centered() if self.centered else self.grid

    @property
    def sigma_values(self) -> NDArray[np.float64]:
        """Sigma value at every profile column, in e/Å².

        Returns
        -------
        np.ndarray, shape (len(binning_grid),)
            ``binning_grid.values``.
        """
        return self.binning_grid.values

    @classmethod
    def from_segments(  # noqa: PLR0913
        cls,
        sigmas: NDArray[np.float64],
        areas: NDArray[np.float64],
        segment_offsets: NDArray[np.int64],
        *,
        atom_indices: NDArray[np.int64] | None = None,
        num_rows: int | None = None,
        binning: BinningSpec | None = None,
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
        binning : BinningSpec | None, optional
            Which grid to bin onto and whether to center each profile
            first, by default None, meaning ``BinningSpec()``. With
            ``centered=True``, profiles are actually binned onto
            ``grid.centered()`` instead, and the result's ``binning_grid``
            reflects that.
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
        binning = BinningSpec() if binning is None else binning
        grid, centered = binning.grid, binning.centered

        if atom_indices is None:
            row_indices = row_indices_from_offsets(segment_offsets, len(sigmas))
            num_rows = len(segment_offsets) if num_rows is None else num_rows
            atom_offsets = None
        else:
            row_indices = np.asarray(atom_indices)
            num_rows = int(row_indices.max()) + 1 if num_rows is None else num_rows
            atom_offsets = row_indices[segment_offsets].astype(np.int64)

        binning_grid = grid.centered() if centered else grid

        areas_out = np.zeros(num_rows, dtype=np.float32)
        charges_out = np.zeros(num_rows, dtype=np.float32)
        profiles_out = np.zeros((num_rows, len(binning_grid)), dtype=np.float32)
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
                binning_grid,
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

        The result is always uncentered (``centered=False``) and binned on
        the base grid, since un-translating returns each atom's mass to
        absolute sigma. Aggregating a centered table therefore also
        regrids it from ``self.grid.centered()`` back onto ``self.grid``,
        one point narrower.

        Parameters
        ----------
        grid : SigmaGrid | None, optional
            Grid for the output molecule profiles, by default None,
            meaning ``self.grid``. Must share this table's ``bin_width``
            (``SigmaGrid.centered`` preserves it, so the default always
            does).
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
            If ``self.atom_offsets`` is None -- already at molecule
            level, nothing to aggregate.
        """
        if self.atom_offsets is None:
            raise ValueError(
                "This SigmaProfileTable has no atom_offsets -- it is "
                "already at molecule level, so there is nothing to "
                "aggregate."
            )
        output_grid = self.grid if grid is None else grid
        # Aggregation un-translates every atom profile by that atom's own
        # mean charge density, putting its mass back at absolute sigma. So
        # the molecule result is never centered, whatever ``self`` was, and
        # belongs on the base grid rather than a centered one. When ``self``
        # *is* centered its profiles sit on a grid with one extra point,
        # half a bin wider at each end, so atom column k lands on base
        # column k - 0.5; accumulate_translated_profiles handles that
        # half-integer offset with its usual two-tap split.
        output_binning_grid = output_grid

        num_atoms = len(self.areas)
        num_mols = len(self.atom_offsets)
        atom_binning_grid = self.binning_grid
        molecule_num_points = len(output_binning_grid)

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
                atom_binning_grid,
            )
            accumulate_translated_profiles(
                molecule_profiles,
                molecule_indices[start_atom:stop_atom],
                batch,
                output_binning_grid,
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
