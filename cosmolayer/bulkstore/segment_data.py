"""Parse COSMO files into flat segment-level arrays and derive atom and
molecule properties.

Turns a mapping of SMILES to ``.cosmo`` files into a compact,
memory-mappable on-disk store (``SegmentStore``): parallel segment-indexed
arrays of coordinates, charges, and areas, a
global atom index per segment, and a ``molecules.parquet`` table of
per-molecule offsets and cavity volume, plus any averaged sigmas
computed for it. Also provides
per-atom / per-molecule sigma profiles: distributions of surface area
fraction over charge density.

Running this module directly builds the store (if missing) and prints
summary statistics.
"""

import argparse
import json
import os
import pathlib
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import threadpoolctl
from rdkit import Chem
from tqdm.auto import tqdm

from cosmolayer import parser
from cosmolayer.cosmosac.constants import (
    COSMO_SAC_2002_AVERAGING_RADIUS,
    COSMO_SAC_2002_F_DECAY,
    COSMO_SAC_2010_AVERAGING_RADIUS,
    COSMO_SAC_2010_F_DECAY,
)

DATA_FILE: pathlib.Path = pathlib.Path("data.npy")
ATOM_INDICES_FILE: pathlib.Path = pathlib.Path("atom_indices.npy")
MOLECULES_FILE: pathlib.Path = pathlib.Path("molecules.parquet")
METADATA_FILE: pathlib.Path = pathlib.Path("metadata.json")

AVERAGING_SCHEMES: dict[str, tuple[float, float]] = {
    "cosmo-rs": (0.5, 1.0),
    "cosmo-sac-2002": (COSMO_SAC_2002_AVERAGING_RADIUS, COSMO_SAC_2002_F_DECAY),
    "cosmo-sac-2010": (COSMO_SAC_2010_AVERAGING_RADIUS, COSMO_SAC_2010_F_DECAY),
}

DEFAULT_MAX_ABS_SIGMA = 0.025
DEFAULT_NUM_POINTS = 51


class SegmentStore:
    """A segment-data store: segment arrays plus the per-molecule
    ``molecules_df`` table describing them, and any averaged sigmas
    computed for it.

    Construct directly only if you already have every component in hand
    (e.g. re-wrapping arrays); normal callers use ``load`` (read an
    existing store from disk, memory-mapped) or ``from_cosmo_files``
    (build a new one).

    Parameters
    ----------
    storage_dir : pathlib.Path
        Directory this store's files live in (or will be written to).
    data : np.ndarray
        ``(n_segs_total, 5)`` array, columns ``[x, y, z, charge, area]``.
    atom_indices : np.ndarray
        ``(n_segs_total,)`` global atom index of each segment.
    molecules_df : pd.DataFrame
        One row per molecule, with columns ``smiles``, ``segment_offsets``,
        ``atom_offsets``, ``num_atoms``, and ``volume``.
    metadata : dict
        ``num_molecules`` and ``num_cosmo_parse_failures`` for this store,
        plus a ``schemes`` entry (scheme name -> ``averaging_radius``,
        ``f_decay``) once any averaged sigmas have been computed for it
        (see ``_populate_averaged_sigmas``).
    averaged_sigmas : dict[str, np.ndarray]
        Scheme name -> ``(n_segs_total,)`` averaged charge density,
        computed automatically by ``from_cosmo_files`` for whatever
        schemes have been computed for this store so far. Empty if none
        have.

    Attributes
    ----------
    coords, charges, areas : np.ndarray
        Views into ``data``'s columns.
    """

    def __init__(
        self,
        storage_dir: pathlib.Path,
        data: np.ndarray,
        atom_indices: np.ndarray,
        molecules_df: pd.DataFrame,
        metadata: dict,
        averaged_sigmas: dict[str, np.ndarray],
    ):
        self.storage_dir = pathlib.Path(storage_dir)
        self.data = data
        self.atom_indices = atom_indices
        self.molecules_df = molecules_df
        self.metadata = metadata
        self.averaged_sigmas = averaged_sigmas
        self.coords = data[:, :3]
        self.charges = data[:, 3]
        self.areas = data[:, 4]

    @staticmethod
    def _reorder_molecule(mol: Chem.Mol) -> Chem.Mol:
        """Reorder a molecule's atoms by their AtomMapNum property.

        Sorts atoms into ascending AtomMapNum order, so atom ``i`` ends up
        carrying ``AtomMapNum == i`` (0-based mapping) or ``AtomMapNum ==
        i + 1`` (1-based). ``from_cosmo_files`` relies on this: it uses
        each segment's COSMO-file atom index directly as its global atom
        index, which only matches a molecule's atom-mapped SMILES once the
        atoms are in this order.

        Parameters
        ----------
        mol : Chem.Mol
            Molecule to reorder.

        Returns
        -------
        Chem.Mol
            Reordered molecule.

        Raises
        ------
        ValueError
            If the molecule has bad atom map numbers.
        """
        num_atoms = mol.GetNumAtoms()
        map_nums = {atom.GetAtomMapNum() for atom in mol.GetAtoms()}
        if map_nums == {0}:
            return mol

        if map_nums in (set(range(num_atoms)), set(range(1, num_atoms + 1))):
            new_order = sorted(
                range(num_atoms), key=lambda i: mol.GetAtomWithIdx(i).GetAtomMapNum()
            )
            return Chem.RenumberAtoms(mol, new_order)

        raise ValueError(
            "Bad atom map numbers: must be all 0, or a 0-based or 1-based "
            "permutation of the atom indices"
        )

    def _populate_averaged_sigmas(
        self,
        schemes: dict[str, tuple[float, float]] | None = None,
        num_threads: int = os.cpu_count(),
    ) -> dict[str, pathlib.Path]:
        """Compute averaged sigmas for this store, under every scheme in
        ``schemes``, writing one file per scheme and updating
        ``self.averaged_sigmas`` in place.

        Called automatically by ``from_cosmo_files``; not otherwise
        exposed, since a store's averaged sigmas are meant to be computed
        once, at build time.

        Calls ``_smooth_segment_sigmas`` once for every scheme, sharing the
        pairwise-distance computation across all of them (see
        ``_compute_averaged_sigmas`` for the measured saving over calling
        this once per scheme). Writes
        ``<name>.npy`` to ``self.storage_dir`` for every ``name`` in
        ``schemes``, each of shape ``(n_segs_total,)`` and dtype float32 --
        aligned index for index with this store's own ``data``/
        ``atom_indices``. Records every scheme's ``averaging_radius`` and
        ``f_decay`` in ``self.metadata["schemes"]``, merging into (not
        replacing) whatever was already recorded there, and rewrites
        ``METADATA_FILE`` with the updated ``self.metadata``.

        The freshly computed arrays are kept in memory (in
        ``self.averaged_sigmas``) rather than reloaded from the files just
        written -- callers that want the memory-mapped versions instead can
        reload with ``SegmentStore.load(self.storage_dir)``.

        Parameters
        ----------
        schemes : dict[str, tuple[float, float]] | None, optional
            Scheme name -> ``(averaging_radius, f_decay)``, by default
            None, meaning ``AVERAGING_SCHEMES`` (Klamt, COSMO-SAC 2002,
            COSMO-SAC 2010). The name becomes the output file's stem, so
            it must not collide with ``DATA_FILE``, ``ATOM_INDICES_FILE``,
            ``MOLECULES_FILE``, or ``METADATA_FILE`` -- not checked.
        num_threads : int, optional
            Number of threads to use, by default the number of available
            CPU cores.

        Returns
        -------
        dict[str, pathlib.Path]
            Scheme name -> path of the ``.npy`` file written for it, in
            the same order as ``schemes``.
        """
        if schemes is None:
            schemes = AVERAGING_SCHEMES
        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")

        scheme_names = list(schemes)
        averaged = self._smooth_segment_sigmas(
            np.asarray(self.coords),
            np.asarray(self.charges),
            np.asarray(self.areas),
            segment_offsets,
            [schemes[name] for name in scheme_names],
            num_threads=num_threads,
        )

        paths: dict[str, pathlib.Path] = {}
        for name, arr in zip(scheme_names, averaged, strict=True):
            f32_arr = arr.astype(np.float32)
            path = self.storage_dir / f"{name}.npy"
            np.save(path, f32_arr)
            self.averaged_sigmas[name] = f32_arr
            paths[name] = path

        self.metadata.setdefault("schemes", {}).update(
            {
                name: {"averaging_radius": r_av, "f_decay": f_decay}
                for name, (r_av, f_decay) in schemes.items()
            }
        )
        with open(self.storage_dir / METADATA_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return paths

    @staticmethod
    def _smooth_segment_sigmas(
        coords: np.ndarray,
        charges: np.ndarray,
        areas: np.ndarray,
        segment_offsets: np.ndarray,
        schemes: Sequence[tuple[float, float]],
        num_threads: int = os.cpu_count(),
    ) -> np.ndarray:
        """Apply one or more COSMO-SAC-style averaging schemes to every
        segment, in one threaded pass over the dataset.

        Calls ``_compute_averaged_sigmas`` once per molecule -- each thread
        handles a disjoint range of whole molecules, so this is safe with no
        locking. Output row ``i`` (``_smooth_segment_sigmas(...)[i]``) can be
        passed straight into ``compute_per_atom_sigma_profiles`` /
        ``compute_per_molecule_sigma_profiles`` as their ``sigmas`` argument,
        alongside the same ``areas``.

        Parameters
        ----------
        coords : np.ndarray
            Segment centroid coordinates, shape ``(n_segs_total, 3)``.
        charges : np.ndarray
            Segment charges, shape ``(n_segs_total,)``.
        areas : np.ndarray
            Segment areas, shape ``(n_segs_total,)``.
        segment_offsets : np.ndarray
            Start index of each molecule's segments within the segment-level
            arrays. Must describe *exactly* the molecules present in
            ``coords`` / ``charges`` / ``areas``: the last molecule's
            segments are assumed to run to the end of those arrays (see
            ``compute_per_atom_sigma_profiles`` for the subsetting caveat).
        schemes : Sequence[tuple[float, float]]
            ``(averaging_radius, f_decay)`` per scheme, in the order the
            result rows are returned -- see ``_compute_averaged_sigmas``.
        num_threads : int, optional
            Number of threads to use, by default the number of available
            CPU cores.

        Returns
        -------
        np.ndarray, shape (len(schemes), n_segs_total)
            Averaged charge density (sigma) for every segment under every
            scheme, in e/Å², row ``i`` matching ``schemes[i]``.
        """
        num_segs = len(charges)
        num_mols = len(segment_offsets)
        averaged_sigmas = np.empty((len(schemes), num_segs), dtype=np.float64)

        def process_range(start_mol: int, stop_mol: int) -> None:
            for mol in range(start_mol, stop_mol):
                start_seg = segment_offsets[mol]
                stop_seg = segment_offsets[mol + 1] if mol + 1 < num_mols else num_segs
                if stop_seg == start_seg:
                    continue
                averaged_sigmas[:, start_seg:stop_seg] = (
                    SegmentStore._compute_averaged_sigmas(
                        coords[start_seg:stop_seg],
                        charges[start_seg:stop_seg],
                        areas[start_seg:stop_seg],
                        schemes,
                    )
                )

        chunk_size = (num_mols + num_threads - 1) // num_threads
        with (
            threadpoolctl.threadpool_limits(limits=1),
            ThreadPoolExecutor(max_workers=num_threads) as executor,
        ):
            futures = []
            for thread_id in range(num_threads):
                start_mol = thread_id * chunk_size
                if start_mol >= num_mols:
                    continue
                stop_mol = min(start_mol + chunk_size, num_mols)
                futures.append(executor.submit(process_range, start_mol, stop_mol))
            for future in as_completed(futures):
                future.result()

        return averaged_sigmas

    @staticmethod
    def _compute_averaged_sigmas(
        coords: np.ndarray,
        charges: np.ndarray,
        areas: np.ndarray,
        schemes: Sequence[tuple[float, float]],
    ) -> np.ndarray:
        """Distance-weighted average of one molecule's segment charge
        densities, under one or more averaging schemes at once.

        Implements the COSMO-SAC segment-averaging procedure (Klamt;
        re-derived in Wang et al. 2007). For every segment ``m``, replaces
        its raw charge density ``sigma_m = q_m / A_m`` with a weighted
        average over every segment ``n`` in the same molecule, including
        itself::

            sigma_avg[m] = sum_n(sigma[n] * w[m, n]) / sum_n(w[m, n])
            w[m, n] = (r_n^2 * r_av^2 / (r_n^2 + r_av^2))
                      * exp(-f_decay * d_mn^2 / (r_n^2 + r_av^2))

        where ``r_n = sqrt(A_n / pi)`` is segment ``n``'s own effective
        radius and ``d_mn`` is the distance between segment centroids ``m``
        and ``n``. The weight uses the *neighbor* segment's radius ``r_n``,
        not the segment being averaged -- easy to get backwards, since it
        makes ``w`` asymmetric even though ``d_mn`` itself is symmetric. In
        COSMO-RS and COSMO-SAC, this averaged density *is* what "sigma"
        refers to, not an optional smoothing step on top of the raw one.

        Accepts a list of ``(averaging_radius, f_decay)`` pairs so multiple
        schemes can share one pairwise-distance computation and each
        scheme's result is returned as its own row. O(n^2) in the
        molecule's own segment count, computed densely (no distance
        cutoff), matching the reference implementation exactly. Always
        upcast to float64 internally regardless of input dtype, since the
        store's own arrays are float32 and the Gram-matrix distance
        expansion loses significant precision at that width.

        The caller must never pass segments from more than one molecule at
        once (see ``_smooth_segment_sigmas`` for the whole-dataset wrapper).

        Parameters
        ----------
        coords : np.ndarray
            Segment centroid coordinates for one molecule, shape
            ``(n_segs, 3)``.
        charges : np.ndarray
            Segment charges for the same molecule, shape ``(n_segs,)``.
        areas : np.ndarray
            Segment areas for the same molecule, shape ``(n_segs,)``.
        schemes : Sequence[tuple[float, float]]
            ``(averaging_radius, f_decay)`` per scheme, in the order the
            result rows are returned. ``averaging_radius`` is the effective
            averaging radius ``r_av``, in Å; ``f_decay`` is the exponential
            decay factor. COSMO-SAC 2010 uses ``(sqrt(7.25 / pi), 3.57)``;
            COSMO-SAC 2002 uses ``(0.8176300195, 1.0)``; Klamt's original
            COSMO-RS scheme uses ``(0.5, 1.0)``.

        Returns
        -------
        np.ndarray, shape (len(schemes), n_segs)
            Averaged charge density for each segment under each scheme, in
            e/Å², row ``i`` matching ``schemes[i]``.
        """
        coords = coords.astype(np.float64, copy=False)
        charges = charges.astype(np.float64, copy=False)
        areas = areas.astype(np.float64, copy=False)

        sigmas = charges / areas
        squared_norms = np.sum(np.square(coords), axis=1)
        squared_distances = (
            squared_norms[:, None] + squared_norms[None, :] - 2.0 * (coords @ coords.T)
        )
        np.clip(squared_distances, 0.0, None, out=squared_distances)
        squared_radii = areas / np.pi

        results = np.empty((len(schemes), len(charges)), dtype=np.float64)
        for i, (averaging_radius, f_decay) in enumerate(schemes):
            sums = squared_radii + averaging_radius**2
            prods = squared_radii * averaging_radius**2
            weights = np.exp(-f_decay * squared_distances / sums) * prods / sums
            results[i] = np.sum(weights * sigmas, axis=1) / np.sum(weights, axis=1)

        return results

    @staticmethod
    def _add_per_atom_area_contributions(
        atom_areas: np.ndarray,
        atom_charges: np.ndarray,
        sigma_profiles: np.ndarray,
        sigmas: np.ndarray,
        areas: np.ndarray,
        atom_indices: np.ndarray,
        max_abs_sigma: float,
        num_points: int,
        shift: bool,
    ) -> None:
        """Accumulate a batch of segments' area into per-atom sigma
        profiles.

        Each segment's charge density is linearly interpolated between the
        two nearest sigma-profile bins, and its area split between those
        bins accordingly. Charge densities outside ``[-max_abs_sigma,
        max_abs_sigma]`` are folded into the nearest boundary point. Each
        atom's profile ends up summing to 1 (all-zero for atoms with no
        surface segments).

        Takes each segment's charge density directly, as ``sigmas``, rather
        than deriving it from ``charge / area`` -- pass raw (``charges /
        areas``) or an averaged density from ``_smooth_segment_sigmas`` /
        ``averaged_sigmas``. ``atom_charges`` accumulates ``sigmas *
        areas``, the true net charge when ``sigmas`` is raw.

        Safe to call concurrently on disjoint segment ranges split on
        *molecule* boundaries (see ``compute_per_atom_sigma_profiles``):
        every segment of an atom must land in the same batch, since
        ``atom_areas``/``atom_charges`` are read back mid-call to
        normalize.

        Parameters
        ----------
        atom_areas : np.ndarray
            Per-atom areas to accumulate into, of shape ``(num_atoms,)``.
            Modified in place.
        atom_charges : np.ndarray
            Per-atom charges to accumulate into, of shape ``(num_atoms,)``.
            Modified in place.
        sigma_profiles : np.ndarray
            Per-atom sigma profiles to accumulate into, of shape
            ``(num_atoms, num_points)``. Modified in place.
        sigmas : np.ndarray
            Charge density of the surface segments in this batch, in e/Å²
            -- raw or averaged, at the caller's choice. Not modified in
            place.
        areas : np.ndarray
            Areas of the surface segments in this batch.
        atom_indices : np.ndarray
            Global atom index associated with each segment in this batch.
        max_abs_sigma : float
            Bounded sigma-profile value at each end of the range: the
            profile spans ``[-max_abs_sigma, max_abs_sigma]``.
        num_points : int
            Number of sigma-profile points. May be even (no point exactly
            at zero) or odd (a point exactly at zero).
        shift : bool
            Whether to shift each atom's profile onto its own mean charge
            density ``q_a / A_a``, centering it so its first moment is
            zero.

        Returns
        -------
        None
            ``atom_areas``, ``atom_charges``, and ``sigma_profiles`` are
            updated in place.
        """
        np.add.at(atom_areas, atom_indices, areas)
        np.add.at(atom_charges, atom_indices, sigmas * areas)

        min_sigma = -max_abs_sigma
        bin_width = sigma_bin_width(max_abs_sigma, num_points)

        summed_areas = atom_areas[atom_indices]

        if shift:
            sigmas = sigmas - atom_charges[atom_indices] / summed_areas

        fractional_bins = (sigmas - min_sigma) / bin_width

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

    @staticmethod
    def _add_shifted_profile_contributions(
        molecule_profiles: np.ndarray,
        molecule_indices: np.ndarray,
        atom_areas: np.ndarray,
        shifts: np.ndarray,
        sigma_profiles: np.ndarray,
        max_abs_sigma: float,
        num_points: int,
        molecule_num_points: int,
    ) -> None:
        """Accumulate a batch of atoms' shifted, area-weighted profiles
        into per-molecule sigma profiles.

        The molecule grid shares the atom grid's ``bin_width`` (see
        ``shifted_grid``), so a zero shift places atom column ``k`` at
        molecule column ``k + grid_offset``, where ``grid_offset =
        (molecule_num_points - num_points) / 2`` -- an integer when the
        point counts share parity, a half-integer otherwise. Each row is
        redistributed with the same two-tap linear interpolation
        ``_add_per_atom_area_contributions`` uses for a single value, with
        out-of-range destinations folded into the nearest boundary column.

        Safe to call concurrently on disjoint atom ranges (see
        ``BulkSigmaProfiles.aggregate``) as long as no two calls share a
        molecule.

        Parameters
        ----------
        molecule_profiles : np.ndarray
            Per-molecule sigma profiles to accumulate into, of shape
            ``(num_molecules, molecule_num_points)``. Modified in place.
        molecule_indices : np.ndarray
            Global molecule index associated with each atom in this batch.
        atom_areas : np.ndarray
            Areas of the atoms in this batch.
        shifts : np.ndarray
            Amount to shift each atom's profile by, in sigma units
            (positive shifts move mass toward the positive end of the
            grid). Pass zeros to accumulate profiles unshifted.
        sigma_profiles : np.ndarray
            Per-atom sigma profiles for the atoms in this batch, of shape
            ``(len(atom_areas), num_points)``.
        max_abs_sigma : float
            Bounded sigma-profile value at each end of the range: the
            per-atom profile spans ``[-max_abs_sigma, max_abs_sigma]``.
        num_points : int
            Number of per-atom sigma-profile points.
        molecule_num_points : int
            Number of per-molecule sigma-profile points.

        Returns
        -------
        None
            ``molecule_profiles`` is updated in place.
        """
        bin_width = sigma_bin_width(max_abs_sigma, num_points)
        grid_offset = (molecule_num_points - num_points) / 2

        fractional_shift = grid_offset + shifts / bin_width
        points_shift = np.floor(fractional_shift).astype(np.int64)
        weight_right = (fractional_shift - points_shift)[:, None]

        contributions = atom_areas[:, None].astype(np.float64) * sigma_profiles.astype(
            np.float64
        )
        points_at_left = np.arange(num_points)[None, :] + points_shift[:, None]
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

    @classmethod
    def load(cls, storage_dir: pathlib.Path | str) -> "SegmentStore":
        """Load an existing segment-data store from disk, memory-mapped.

        Parameters
        ----------
        storage_dir : pathlib.Path | str
            Directory holding a store built by ``from_cosmo_files`` (i.e.
            for which ``segment_data_exists`` is True).

        Returns
        -------
        SegmentStore
            ``data``, ``atom_indices``, and any ``averaged_sigmas`` arrays
            are memory-mapped (``mmap_mode="r"``).

        Raises
        ------
        FileNotFoundError
            If ``storage_dir`` doesn't hold a complete store.
        """
        storage_dir = pathlib.Path(storage_dir)
        if not segment_data_exists(storage_dir):
            raise FileNotFoundError(
                f"No segment-data store in {storage_dir} (missing one of "
                f"{DATA_FILE}, {ATOM_INDICES_FILE}, {MOLECULES_FILE}, "
                f"{METADATA_FILE}; see SegmentStore.from_cosmo_files)."
            )
        with open(storage_dir / METADATA_FILE) as f:
            metadata = json.load(f)
        data = np.load(storage_dir / DATA_FILE, mmap_mode="r")
        atom_indices = np.load(storage_dir / ATOM_INDICES_FILE, mmap_mode="r")
        molecules_df = pd.read_parquet(storage_dir / MOLECULES_FILE)

        scheme_names = sorted(metadata.get("schemes", {}))
        averaged_sigmas = {
            name: np.load(storage_dir / f"{name}.npy", mmap_mode="r")
            for name in scheme_names
        }

        return cls(
            storage_dir, data, atom_indices, molecules_df, metadata, averaged_sigmas
        )

    @classmethod
    def from_cosmo_files(
        cls,
        cosmo_files_dir: pathlib.Path,
        smiles_to_filename: dict[str, str],
        storage_dir: pathlib.Path,
        ignore_errors: bool = False,
        schemes: dict[str, tuple[float, float]] | None = None,
        num_threads: int = os.cpu_count(),
    ) -> "SegmentStore":
        """Parse COSMO files and persist their segment data to a directory.

        Writes ``data.npy`` (columns ``[x, y, z, charge, area]``),
        ``atom_indices.npy`` (global atom index per segment), a
        ``molecules.parquet`` table (columns ``smiles`` [RDKit canonical],
        ``segment_offsets``, ``atom_offsets``, ``num_atoms``, ``volume``), and
        a ``metadata.json`` (``num_molecules``, ``num_cosmo_parse_failures``).
        A molecule's atoms are numbered by ``segment_df["atom"]`` (the COSMO
        file's own 0-based order) directly, which is why each SMILES's atom
        count is checked against its COSMO file and, if atom-mapped, reordered
        via ``_reorder_molecule`` onto that same indexing.

        Also computes and writes averaged sigmas for the new store (see
        ``_populate_averaged_sigmas``), unless ``schemes`` is an empty
        dict -- this adds a ``schemes`` entry to ``metadata.json`` rather
        than writing it as a separate file.

        Parameters
        ----------
        cosmo_files_dir : pathlib.Path
            Directory containing the ``.cosmo`` files named by
            ``smiles_to_filename``'s values.
        smiles_to_filename : dict[str, str]
            SMILES string -> ``.cosmo`` filename (relative to
            ``cosmo_files_dir``), one entry per molecule to store.
        storage_dir : pathlib.Path
            Destination directory for the output files. Created if missing.
        ignore_errors : bool, optional
            If True, a molecule that fails to parse or validate is skipped
            (and counted in ``num_cosmo_parse_failures``) instead of raising.
            By default False.
        schemes : dict[str, tuple[float, float]] | None, optional
            Passed to ``_populate_averaged_sigmas``, by default None,
            meaning ``AVERAGING_SCHEMES``. Pass ``{}`` to skip averaging
            entirely.
        num_threads : int, optional
            Passed to ``_populate_averaged_sigmas``, by default the number
            of available CPU cores.

        Raises
        ------
        ValueError
            If a molecule's SMILES or COSMO file cannot be parsed and
            ``ignore_errors`` is False, or if no molecule could be stored.
        """
        data_chunks, atoms_chunks = [], []
        segment_offsets, segment_offset = [], 0
        atom_offsets, atom_offset = [], 0
        num_atoms = []
        volumes = []
        successful_molecules = []
        num_cosmo_parse_failures = 0

        for smi, filename in tqdm(
            smiles_to_filename.items(), desc="Processing COSMO files"
        ):
            try:
                _, atom_df, segment_df, volume = parser.parse_cosmo_file(
                    (cosmo_files_dir / filename).read_text()
                )
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"RDKit could not parse SMILES {smi!r}")
                if mol.GetNumAtoms() != len(atom_df):
                    raise ValueError(
                        f"SMILES {smi!r} has {mol.GetNumAtoms()} atoms, but "
                        f"{filename} has {len(atom_df)}"
                    )
                mol = cls._reorder_molecule(mol)
            except (ValueError, AssertionError) as e:
                if ignore_errors:
                    tqdm.write(f"Error parsing {smi}->{filename}: {e}")
                    num_cosmo_parse_failures += 1
                    continue
                else:
                    raise e

            data_chunks.append(
                segment_df[["x", "y", "z", "charge", "area"]].values.astype("float32")
            )
            atoms_chunks.append(segment_df["atom"].values.astype("int64") + atom_offset)

            segment_offsets.append(segment_offset)
            segment_offset += len(segment_df)
            atom_offsets.append(atom_offset)
            atom_offset += len(atom_df)
            num_atoms.append(len(atom_df))
            volumes.append(volume)
            successful_molecules.append((Chem.MolToSmiles(mol), mol))

        if not successful_molecules:
            raise ValueError("No COSMO files could be parsed successfully.")

        storage_dir = pathlib.Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        data = np.concatenate(data_chunks, axis=0)
        atom_indices = np.concatenate(atoms_chunks)
        np.save(storage_dir / DATA_FILE, data)
        np.save(storage_dir / ATOM_INDICES_FILE, atom_indices)

        molecules = pd.DataFrame(
            {
                "smiles": [smi for smi, _ in successful_molecules],
                "segment_offsets": np.array(segment_offsets, dtype="int64"),
                "atom_offsets": np.array(atom_offsets, dtype="int64"),
                "num_atoms": np.array(num_atoms, dtype="int64"),
                "volume": np.array(volumes, dtype="float64"),
            }
        )
        molecules.to_parquet(storage_dir / MOLECULES_FILE, index=False)

        metadata = {
            "num_molecules": len(successful_molecules),
            "num_cosmo_parse_failures": num_cosmo_parse_failures,
        }
        with open(storage_dir / METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        store = cls(storage_dir, data, atom_indices, molecules, metadata, {})
        if schemes is None or schemes:
            store._populate_averaged_sigmas(
                schemes=AVERAGING_SCHEMES if schemes is None else schemes,
                num_threads=num_threads,
            )
        return store

    def compute_atom_sigma_profiles(
        self,
        scheme: str | None = None,
        max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
        num_points: int = DEFAULT_NUM_POINTS,
        num_threads: int = os.cpu_count(),
        shift: bool = False,
    ) -> "BulkSigmaProfiles":
        """Compute this store's per-atom sigma profiles.

        A thin wrapper around ``BulkSigmaProfiles`` that supplies this
        store's own ``areas``/``atom_indices``/``segment_offsets``/
        ``num_atoms``, and resolves ``sigmas`` from ``scheme``.

        Parameters
        ----------
        scheme : str | None, optional
            Which charge density to bin: None (default) uses raw
            ``charges / areas``; a scheme name uses
            ``self.averaged_sigmas[scheme]`` (populated automatically by
            ``from_cosmo_files``).
        max_abs_sigma : float, optional
            Bounded sigma-profile value at each end of the unshifted
            range, by default ``DEFAULT_MAX_ABS_SIGMA``. With
            ``shift=True``, profiles are actually binned onto
            ``shifted_grid(max_abs_sigma, num_points)`` instead.
        num_points : int, optional
            Number of sigma-profile points of the unshifted grid, by
            default ``DEFAULT_NUM_POINTS``.
        num_threads : int, optional
            Number of threads to use, by default the number of available
            CPU cores.
        shift : bool, optional
            Whether to shift each atom's profile onto its own mean charge
            density, centering it, by default False.

        Returns
        -------
        BulkSigmaProfiles
            Atom-level (``atom_offset`` derived, so the result can be
            reassembled into molecule-level profiles via its own
            ``aggregate`` method). ``charges`` is the true net charge when
            ``scheme`` is None, or a smoothed "equivalent charge" when it
            isn't.

        Raises
        ------
        KeyError
            If ``scheme`` is given but not in ``self.averaged_sigmas``.
        """
        if scheme is None:
            sigmas = np.asarray(self.charges) / np.asarray(self.areas)
        else:
            if scheme not in self.averaged_sigmas:
                raise KeyError(
                    f"No averaged sigmas for scheme {scheme!r} in this "
                    f"store. Known schemes: {sorted(self.averaged_sigmas)}."
                )
            sigmas = np.asarray(self.averaged_sigmas[scheme])

        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")
        total_num_atoms = int(self.molecules_df["num_atoms"].sum())
        return BulkSigmaProfiles(
            np.asarray(self.areas),
            sigmas,
            segment_offsets,
            atom_indices=np.asarray(self.atom_indices),
            num_atoms=total_num_atoms,
            max_abs_sigma=max_abs_sigma,
            num_points=num_points,
            num_threads=num_threads,
            shift=shift,
        )


class BulkSigmaProfiles:
    """A set of area-weighted sigma profiles, at atom or molecule level.

    Computes profiles itself from raw segment-level data and keeps its
    own copies of everything needed to interpret and, if applicable,
    aggregate them -- independent of whatever ``SegmentStore`` the inputs
    came from.

    Parameters
    ----------
    areas : np.ndarray
        Areas of the surface segments, shape ``(n_segs,)``.
    sigmas : np.ndarray
        Charge density of the surface segments, in e/Å² -- raw or
        averaged, at the caller's choice.
    segment_offsets : np.ndarray
        Start index of each molecule's segments within the segment-level
        arrays. Must describe *exactly* the molecules present in
        ``areas`` / ``sigmas`` / ``atom_indices``: the last molecule's
        segments run to the end of those arrays.
    atom_indices : np.ndarray | None, optional
        Global atom index associated with each segment. By default None,
        meaning build one profile *per molecule* instead of per atom --
        segments are grouped straight by ``segment_offsets`` (faking one
        monoatomic "atom" per molecule), and ``self.atom_offset`` stays
        None. When given, one profile is built *per atom* instead,
        grouped via ``atom_indices``, and ``self.atom_offset`` (each
        molecule's first atom index, so ``aggregate`` can later
        reassemble these into molecule-level profiles) is derived as
        ``atom_indices[segment_offsets]`` -- correct because segments are
        grouped by ascending atom index within a molecule (true of the
        COSMO file formats this module parses; verified against a real
        TURBOMOLE file).
    num_atoms : int | None, optional
        Total number of atoms, i.e. the number of profile rows to build
        when ``atom_indices`` is given. By default None, meaning
        ``int(atom_indices.max()) + 1`` -- correct for a full store, but
        potentially wrong for an already-subsetted one (see
        ``compute_per_atom_sigma_profiles``'s subsetting caveat); pass it
        explicitly when subsetting.
    max_abs_sigma : float, optional
        Bounded sigma-profile value at each end of the unshifted range,
        by default ``DEFAULT_MAX_ABS_SIGMA``. With ``shift=True``,
        profiles are actually binned onto ``shifted_grid(max_abs_sigma,
        num_points)`` instead, and ``self.sigma_grid`` reflects that.
    num_points : int, optional
        Number of sigma-profile points of the unshifted grid, by default
        ``DEFAULT_NUM_POINTS``.
    num_threads : int, optional
        Number of threads to use, by default the number of available CPU
        cores.
    shift : bool, optional
        Whether to shift each profile onto its own mean charge density,
        centering it, by default False.

    Attributes
    ----------
    areas, charges : np.ndarray
        Per-row area and net (or smoothed "equivalent") charge, shape
        ``(n,)``.
    profiles : np.ndarray
        Per-row area-fraction sigma profile, shape ``(n, len(sigma_grid))``.
    sigma_grid : np.ndarray
        Sigma value at each profile column (see ``sigma_grid_points``).
    atom_offset : np.ndarray | None
        See ``atom_indices`` above.
    """

    def __init__(
        self,
        areas: np.ndarray,
        sigmas: np.ndarray,
        segment_offsets: np.ndarray,
        atom_indices: np.ndarray | None = None,
        num_atoms: int | None = None,
        max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
        num_points: int = DEFAULT_NUM_POINTS,
        num_threads: int = os.cpu_count(),
        shift: bool = False,
    ):
        areas = np.asarray(areas)
        sigmas = np.asarray(sigmas)
        segment_offsets = np.asarray(segment_offsets)

        if atom_indices is None:
            num_rows = len(segment_offsets)
            row_indices = np.repeat(
                np.arange(num_rows, dtype=np.int64),
                np.diff(np.append(segment_offsets, len(sigmas))),
            )
            atom_offset = None
        else:
            row_indices = np.asarray(atom_indices)
            num_rows = int(row_indices.max()) + 1 if num_atoms is None else num_atoms
            atom_offset = row_indices[segment_offsets].astype(np.int64)

        bin_max_abs_sigma, bin_num_points = (
            shifted_grid(max_abs_sigma, num_points)
            if shift
            else (max_abs_sigma, num_points)
        )

        self.areas, self.charges, self.profiles = compute_per_atom_sigma_profiles(
            sigmas,
            areas,
            row_indices,
            segment_offsets,
            num_rows,
            max_abs_sigma=max_abs_sigma,
            num_points=num_points,
            num_threads=num_threads,
            shift=shift,
        )
        self.sigma_grid = sigma_grid_points(bin_max_abs_sigma, bin_num_points)
        self.atom_offset = atom_offset

    @classmethod
    def _from_arrays(
        cls,
        areas: np.ndarray,
        charges: np.ndarray,
        profiles: np.ndarray,
        sigma_grid: np.ndarray,
        atom_offset: np.ndarray | None,
    ) -> "BulkSigmaProfiles":
        """Wrap already-computed profile arrays with no further binning.

        Used internally by ``aggregate`` to build its molecule-level
        result, since that reassembles existing profiles rather than
        binning fresh segment-level data. Ordinary construction should go
        through ``__init__`` instead.
        """
        self = cls.__new__(cls)
        self.areas = np.asarray(areas)
        self.charges = np.asarray(charges)
        self.profiles = np.asarray(profiles)
        self.sigma_grid = np.asarray(sigma_grid)
        self.atom_offset = None if atom_offset is None else np.asarray(atom_offset)
        return self

    def aggregate(
        self,
        shift: bool = True,
        output_sigma_grid: np.ndarray | None = None,
        num_threads: int = os.cpu_count(),
        normalize: bool = False,
    ) -> "BulkSigmaProfiles":
        """Reassemble per-molecule profiles from these per-atom ones.

        Mirrors ``SegmentStore._add_shifted_profile_contributions``'s
        reassembly (area-weighted sum of atom profiles onto a shared
        molecule axis, with each atom's own mean charge density
        un-shifted first when ``shift=True``) but reads the atom grid
        straight from ``self.sigma_grid`` instead of re-deriving it from
        separate max_abs_sigma/num_points arguments.

        Parameters
        ----------
        shift : bool, optional
            Whether these profiles are centered on each atom's own mean
            charge density and need un-shifting before summing, by
            default True. Pass the same value used to build ``self``.
        output_sigma_grid : np.ndarray | None, optional
            Sigma grid for the output molecule profiles, by default None,
            meaning ``self.sigma_grid`` (same grid, no change of extent
            or point count).
        num_threads : int, optional
            Number of threads to use, by default the number of available
            CPU cores.
        normalize : bool, optional
            Whether to divide each molecule's profile by its total area
            so it sums to 1, by default False.

        Returns
        -------
        BulkSigmaProfiles
            Molecule-level (``atom_offset is None``).

        Raises
        ------
        ValueError
            If ``self.atom_offset`` is None -- already at molecule level,
            nothing to aggregate.
        """
        if self.atom_offset is None:
            raise ValueError(
                "This BulkSigmaProfiles has no atom_offset -- it is "
                "already at molecule level, so there is nothing to "
                "aggregate."
            )
        if output_sigma_grid is None:
            output_sigma_grid = self.sigma_grid
        output_sigma_grid = np.asarray(output_sigma_grid)

        num_atoms = len(self.areas)
        num_mols = len(self.atom_offset)
        atom_num_points = len(self.sigma_grid)
        atom_max_abs_sigma = float(self.sigma_grid[-1])
        molecule_num_points = len(output_sigma_grid)

        molecule_indices = np.repeat(
            np.arange(num_mols), np.diff(np.append(self.atom_offset, num_atoms))
        )

        shifts = np.zeros(num_atoms, dtype=np.float64)
        if shift:
            has_area = self.areas > 0
            shifts[has_area] = self.charges[has_area] / self.areas[has_area]

        molecule_profiles = np.zeros((num_mols, molecule_num_points), dtype=np.float64)
        chunk_size = (num_mols + num_threads - 1) // num_threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                start_mol = thread_id * chunk_size
                if start_mol >= num_mols:
                    continue
                stop_mol = min(start_mol + chunk_size, num_mols)
                start_atom = self.atom_offset[start_mol]
                stop_atom = (
                    self.atom_offset[stop_mol] if stop_mol < num_mols else num_atoms
                )
                futures.append(
                    executor.submit(
                        SegmentStore._add_shifted_profile_contributions,
                        molecule_profiles,
                        molecule_indices[start_atom:stop_atom],
                        self.areas[start_atom:stop_atom],
                        shifts[start_atom:stop_atom],
                        self.profiles[start_atom:stop_atom],
                        atom_max_abs_sigma,
                        atom_num_points,
                        molecule_num_points,
                    )
                )
            for future in as_completed(futures):
                future.result()

        if normalize:
            molecule_profiles = molecule_profiles / molecule_profiles.sum(
                axis=1, keepdims=True
            )

        molecule_areas = compute_per_molecule_properties(self.areas, self.atom_offset)
        molecule_charges = compute_per_molecule_properties(
            self.charges, self.atom_offset
        )

        return BulkSigmaProfiles._from_arrays(
            molecule_areas,
            molecule_charges,
            molecule_profiles.astype(np.float32),
            output_sigma_grid,
            atom_offset=None,
        )


def sigma_bin_width(max_abs_sigma: float, num_points: int) -> float:
    """Bin width of a symmetric sigma grid spanning ``[-max_abs_sigma, max_abs_sigma]``.

    Parameters
    ----------
    max_abs_sigma : float
        Bounded sigma-profile value at each end of the range.
    num_points : int
        Number of grid points.

    Returns
    -------
    float
        ``2 * max_abs_sigma / (num_points - 1)``.
    """
    return (2 * max_abs_sigma) / (num_points - 1)


def sigma_grid_points(max_abs_sigma: float, num_points: int) -> np.ndarray:
    """Sigma values at every point of a symmetric sigma grid.

    Parameters
    ----------
    max_abs_sigma : float
        Bounded sigma-profile value at each end of the range.
    num_points : int
        Number of grid points.

    Returns
    -------
    np.ndarray, shape (num_points,)
        Evenly spaced values from ``-max_abs_sigma`` to ``max_abs_sigma``.
    """
    return np.linspace(-max_abs_sigma, max_abs_sigma, num_points)


def shifted_grid(max_abs_sigma: float, num_points: int) -> tuple[float, int]:
    """Grid to bin a shifted (centered) profile onto, given the unshifted
    grid's own parameters.

    Treats ``(max_abs_sigma, num_points)`` as describing the unshifted
    grid. If ``num_points`` is even, the shifted grid is identical. If
    odd, the shifted grid gains one point and extends by half a bin width
    on each side, preserving ``bin_width`` exactly -- so the shifted grid
    always ends up with an even point count, with no point sitting exactly
    at sigma = 0.

    Parameters
    ----------
    max_abs_sigma : float
        Unshifted grid's bounded sigma-profile value at each end of its
        range.
    num_points : int
        Unshifted grid's number of points.

    Returns
    -------
    tuple[float, int]
        ``(max_abs_sigma, num_points)`` for the shifted grid.
    """
    if num_points % 2 == 0:
        return max_abs_sigma, num_points
    bin_width = sigma_bin_width(max_abs_sigma, num_points)
    return max_abs_sigma + bin_width / 2, num_points + 1


def segment_data_exists(storage_dir: pathlib.Path | str) -> bool:
    """Check if the segment data exists in a directory.

    Parameters
    ----------
    storage_dir : pathlib.Path | str
        Directory containing the segment data.

    Returns
    -------
    bool
        True if the segment data exists, False otherwise.
    """
    storage_dir = pathlib.Path(storage_dir)
    return all(
        (storage_dir / file).exists()
        for file in [DATA_FILE, ATOM_INDICES_FILE, MOLECULES_FILE, METADATA_FILE]
    )


def compute_per_atom_properties(
    properties: np.ndarray, atom_indices: np.ndarray, total_num_atoms: int
) -> np.ndarray:
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
    properties: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Compute the per-molecule sum of a property.

    Parameters
    ----------
    properties : np.ndarray
        Property values, of shape ``(total_num_atoms,)``.
    offsets : np.ndarray
        Global index of each molecule's first atom (cumulative sum of
        ``num_atoms`` for preceding molecules) -- *not* the ``offsets``
        column of the ``molecules`` table, which indexes segment-level
        arrays instead.

    Returns
    -------
    np.ndarray
        Per-molecule sum of ``properties``, of shape ``(n_molecules,)``.
    """
    return np.add.reduceat(properties, offsets)


def compute_per_atom_sigma_profiles(
    sigmas: np.ndarray,
    areas: np.ndarray,
    atom_indices: np.ndarray,
    segment_offsets: np.ndarray,
    num_atoms: int,
    max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
    num_points: int = DEFAULT_NUM_POINTS,
    num_threads: int = os.cpu_count(),
    shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linearly interpolated per-atom sigma profiles.

    Takes each segment's charge density directly, as ``sigmas`` -- pass
    raw ``charges / areas``, or an averaged density from
    ``SegmentStore._smooth_segment_sigmas`` / ``SegmentStore.averaged_sigmas``
    (see ``SegmentStore._add_per_atom_area_contributions``).

    ``max_abs_sigma``/``num_points`` describe the unshifted grid. With
    ``shift=True``, profiles are actually binned onto
    ``shifted_grid(max_abs_sigma, num_points)`` instead -- so the returned
    ``sigma_profiles``'s column count is ``num_points`` when
    ``shift=False``, or the (possibly one point larger) shifted count when
    ``shift=True``.

    Parameters
    ----------
    sigmas : np.ndarray
        Charge density of the surface segments, in e/Å² -- raw or averaged,
        at the caller's choice (see above).
    areas : np.ndarray
        Areas of the surface segments.
    atom_indices : np.ndarray
        Global atom index associated with each segment.
    segment_offsets : np.ndarray
        Start index of each molecule's segments within the segment-level
        arrays. Must describe *exactly* the molecules present in ``sigmas``
        / ``areas`` / ``atom_indices``: the last molecule's segments run to
        the end of those arrays. Slicing ``segment_offsets`` and the
        segment-level arrays for a subset must be done together, or
        leftover segments are silently attributed to the wrong molecule --
        pass a matching, under-sized ``num_atoms`` to turn that mistake
        into an ``AssertionError`` instead.
    num_atoms : int
        Total number of atoms in the dataset.
    max_abs_sigma : float, optional
        Bounded sigma-profile value at each end of the unshifted range, by
        default 0.0255. See above for the shifted case.
    num_points : int, optional
        Number of sigma-profile points of the unshifted grid, by default
        52. See above for the shifted case.
    num_threads : int, optional
        Number of threads to use, by default the number of available CPU cores.
    shift : bool, optional
        Whether to shift each atom's profile onto its own mean charge
        density ``q_a / A_a``, centering it, by default False.

    Returns
    -------
    atom_areas : np.ndarray, shape (num_atoms,)
        Per-atom areas.
    atom_charges : np.ndarray, shape (num_atoms,)
        Per-atom charges -- the true net charge when ``sigmas`` is raw, or
        a smoothed "equivalent charge" when it isn't.
    sigma_profiles : np.ndarray, shape (num_atoms, num_points or shifted_grid(...)[1])
        Per-atom area-fraction profiles, each summing to 1 (all-zero for
        atoms with no surface segments). Charge densities outside the
        range are folded into the boundary columns. With ``shift=True``
        each profile also has zero first moment (up to folding error).
    """
    bin_max_abs_sigma, bin_num_points = (
        shifted_grid(max_abs_sigma, num_points)
        if shift
        else (max_abs_sigma, num_points)
    )
    atom_areas = np.zeros(num_atoms, dtype=np.float32)
    atom_charges = np.zeros(num_atoms, dtype=np.float32)
    sigma_profiles = np.zeros((num_atoms, bin_num_points), dtype=np.float32)
    assert int(atom_indices.max(initial=-1)) < num_atoms, (
        "atom_indices references an atom index >= num_atoms; segment_offsets "
        "must describe exactly the molecules present in the segment-level "
        "arrays (see this function's docstring)"
    )
    num_segs = len(sigmas)
    num_mols = len(segment_offsets)
    chunk_size = (num_mols + num_threads - 1) // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            start_mol = thread_id * chunk_size
            if start_mol >= num_mols:
                continue
            stop_mol = min(start_mol + chunk_size, num_mols)
            start_seg = segment_offsets[start_mol]
            stop_seg = segment_offsets[stop_mol] if stop_mol < num_mols else num_segs
            futures.append(
                executor.submit(
                    SegmentStore._add_per_atom_area_contributions,
                    atom_areas,
                    atom_charges,
                    sigma_profiles,
                    sigmas[start_seg:stop_seg],
                    areas[start_seg:stop_seg],
                    atom_indices[start_seg:stop_seg],
                    bin_max_abs_sigma,
                    bin_num_points,
                    shift,
                )
            )
        for future in as_completed(futures):
            future.result()
    return atom_areas, atom_charges, sigma_profiles


def compute_per_molecule_sigma_profiles(
    sigmas: np.ndarray,
    areas: np.ndarray,
    segment_offsets: np.ndarray,
    num_molecules: int,
    max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
    num_points: int = DEFAULT_NUM_POINTS,
    num_threads: int = os.cpu_count(),
    shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linearly interpolated per-molecule sigma profiles.

    A wrapper around ``compute_per_atom_sigma_profiles`` that computes
    per-molecule profiles by faking monoatomic molecules. Takes each
    segment's charge density directly, as ``sigmas`` (see
    ``compute_per_atom_sigma_profiles`` for raw vs. averaged).

    Parameters
    ----------
    sigmas: np.ndarray
        Charge density of the surface segments, in e/Å² -- raw or averaged,
        at the caller's choice (see above).
    areas: np.ndarray
        Areas of the surface segments.
    segment_offsets: np.ndarray
        Start index of each molecule's segments within the segment-level
        arrays. Must describe *exactly* the molecules present in ``sigmas``
        / ``areas``: the last molecule's segments run to the end of those
        arrays. Slicing ``segment_offsets`` and the segment-level arrays
        for a subset must be done together, or leftover segments are
        silently attributed to the wrong molecule -- pass a matching,
        under-sized ``num_molecules`` to turn that mistake into an
        ``AssertionError`` instead.
    num_molecules: int
        Total number of molecules in the dataset.
    max_abs_sigma: float, optional
        Bounded sigma-profile value at each end of the unshifted range, by
        default 0.0255. With ``shift=True``, profiles are actually binned
        onto ``shifted_grid(max_abs_sigma, num_points)`` instead (see
        ``compute_per_atom_sigma_profiles``).
    num_points: int, optional
        Number of sigma-profile points of the unshifted grid, by default
        52.
    num_threads: int
        Number of threads to use.
    shift: bool
        Whether to shift each molecule's profile onto its own mean charge
        density ``q_m / A_m``, centering it so its first moment is zero.

    Returns
    -------
    molecule_areas : np.ndarray, shape (num_molecules,)
        Per-molecule areas.
    molecule_charges : np.ndarray, shape (num_molecules,)
        Per-molecule charges -- the true net charge when ``sigmas`` is raw,
        or a smoothed "equivalent charge" when it isn't.
    molecule_sigma_profiles : np.ndarray, shape (num_molecules, num_points)
        Per-molecule area-fraction profiles, each summing to 1 (all-zero
        for molecules with no surface segments). Charge densities outside
        the range are folded into the boundary columns. With
        ``shift=True`` each profile also has zero first moment.
    """
    atom_indices = np.repeat(
        np.arange(num_molecules, dtype=np.int64),
        np.diff(np.append(segment_offsets, len(sigmas))),
    )

    return compute_per_atom_sigma_profiles(
        sigmas,
        areas,
        atom_indices,
        segment_offsets,
        num_molecules,
        max_abs_sigma=max_abs_sigma,
        num_points=num_points,
        num_threads=num_threads,
        shift=shift,
    )


def print_stats(
    title: str,
    properties: np.ndarray,
    quantiles: tuple[float, ...] = (0.01, 0.1, 0.5, 0.9, 0.99),
    value_format: str = ".6f",
) -> None:
    """Print min, max, mean, and quantiles of a property as a table.

    Parameters
    ----------
    title : str
        Label identifying the property, used as the table's column header.
    properties : np.ndarray
        Values to summarize.
    quantiles : tuple[float, ...], optional
        Quantiles (in ``[0, 1]``) to report, by default
        ``(0.01, 0.1, 0.5, 0.9, 0.99)``.
    value_format : str, optional
        Format spec for the reported values, by default ``".6f"``. Use an
        exponential format such as ``".3e"`` for quantities whose whole
        range sits far below the fixed-point resolution, which would
        otherwise print as a column of zeros.

    Returns
    -------
    None
        The statistics are printed directly; nothing is returned.
    """
    print(f"\n{'Statistic':<10} | {title:>20}")
    print("-" * 33)
    print(f"{'Count':<10} | {len(properties):>20}")
    print(f"{'Min':<10} | {properties.min():>20{value_format}}")
    for q in sorted(quantiles):
        q_percent = q * 100
        if int(q_percent) == q_percent:
            qtext = f"{q_percent:.0f}%"
        else:
            qtext = f"{q_percent:.1f}%"
        print(f"{qtext:<10} | {np.quantile(properties, q):>20{value_format}}")
    print(f"{'Max':<10} | {properties.max():>20{value_format}}")
    print(f"{'Mean':<10} | {properties.mean():>20{value_format}}")
    print()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description=(
            "Build the segment data store (if missing) and print summary "
            "statistics for atom- and molecule-level charges, areas, and "
            "sigma profiles."
        )
    )
    arg_parser.add_argument(
        "--storage-dir",
        type=str,
        help="The directory to store the segment data.",
        default=(
            pathlib.Path(__file__).resolve().parents[1]
            / "sigma-prediction"
            / "segment_data"
        ).as_posix(),
    )
    arg_parser.add_argument(
        "--cosmo-files-dir",
        type=str,
        default=None,
        help=(
            "Directory containing the .cosmo files referenced by "
            "--smiles-to-filename. Required only if --storage-dir doesn't "
            "already hold a built store."
        ),
    )
    arg_parser.add_argument(
        "--smiles-to-filename",
        type=str,
        default=None,
        help=(
            "Path to a JSON file mapping each SMILES string to its .cosmo "
            "filename (relative to --cosmo-files-dir); passed straight to "
            "SegmentStore.from_cosmo_files. Required only if --storage-dir "
            "doesn't already hold a built store."
        ),
    )
    arg_parser.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count(),
        help=(
            "Number of threads to use for every threaded step (segment "
            "averaging, sigma-profile binning). Defaults to the number of "
            "available CPU cores (os.cpu_count()) -- lower this on a shared "
            "machine to leave headroom for other users."
        ),
    )
    arg_parser.add_argument(
        "--sigma-scheme",
        type=str,
        default=None,
        help=(
            "Name of a COSMO-SAC averaging scheme (see AVERAGING_SCHEMES, "
            "e.g. 'cosmo-rs', 'cosmo-sac-2002', 'cosmo-sac-2010') to use for "
            "every statistic below, in place of raw charges / areas. "
            "Requires storage_dir's SegmentStore to already have that "
            "scheme in its averaged_sigmas (computed automatically by "
            "SegmentStore.from_cosmo_files, unless skipped). Default None: "
            "use raw sigma, unchanged."
        ),
    )
    args = arg_parser.parse_args()
    num_threads = args.num_threads
    sigma_scheme = args.sigma_scheme

    storage_dir = pathlib.Path(args.storage_dir)

    if not segment_data_exists(storage_dir):
        if args.cosmo_files_dir is None or args.smiles_to_filename is None:
            arg_parser.error(
                "--cosmo-files-dir and --smiles-to-filename are required "
                f"when --storage-dir ({storage_dir}) doesn't already hold "
                "a built store."
            )
        print("Storing segment data and averaged sigmas...")
        cosmo_files_dir = pathlib.Path(args.cosmo_files_dir)
        with open(args.smiles_to_filename) as f:
            smiles_to_filename = json.load(f)
        start_time = time.time()
        SegmentStore.from_cosmo_files(
            cosmo_files_dir, smiles_to_filename, storage_dir, num_threads=num_threads
        )
        elapsed_time = time.time() - start_time
        print(
            "Time to store segment data and averaged sigmas: "
            f"{elapsed_time:.2f} seconds"
        )
    else:
        print("Segment data already exists.")

    start_time = time.time()
    store = SegmentStore.load(storage_dir)
    coords, charges, areas, atom_indices, molecules_df = (
        store.coords,
        store.charges,
        store.areas,
        store.atom_indices,
        store.molecules_df,
    )

    segment_offsets = molecules_df["segment_offsets"].values.astype("int64")
    elapsed_time = time.time() - start_time
    print(f"Time to load segment data: {elapsed_time:.2f} seconds")

    total_num_atoms = molecules_df["num_atoms"].sum()

    start_time = time.time()
    atom_charges = compute_per_atom_properties(charges, atom_indices, total_num_atoms)
    atom_areas = compute_per_atom_properties(areas, atom_indices, total_num_atoms)
    atom_segment_counts = np.bincount(atom_indices, minlength=total_num_atoms)
    atom_charges_pos = compute_per_atom_properties(
        np.where(charges > 0, charges, 0.0), atom_indices, total_num_atoms
    )
    atom_charges_neg = compute_per_atom_properties(
        np.where(charges < 0, charges, 0.0), atom_indices, total_num_atoms
    )
    hidden_charge = np.minimum(atom_charges_pos, -atom_charges_neg)
    elapsed_time = time.time() - start_time
    print(f"Time to compute per-atom properties: {elapsed_time:.2f} seconds")
    print_stats("Atom charges", atom_charges)
    print_stats("Atom areas", atom_areas)
    print_stats("Atom segment counts", atom_segment_counts)
    print_stats("Hidden charge", hidden_charge)

    max_abs_sigma = DEFAULT_MAX_ABS_SIGMA
    num_points = DEFAULT_NUM_POINTS
    bin_width = sigma_bin_width(max_abs_sigma, num_points)
    if sigma_scheme is None:
        sigmas = charges / areas
    else:
        if sigma_scheme not in store.averaged_sigmas:
            arg_parser.error(
                f"No averaged sigmas for scheme {sigma_scheme!r} in "
                f"{storage_dir}. Known schemes: {sorted(store.averaged_sigmas)}."
            )
        print(f"Using averaged sigmas for scheme {sigma_scheme!r}...")
        sigmas = np.asarray(store.averaged_sigmas[sigma_scheme])
    start_time = time.time()
    atom_sigma_profiles = store.compute_atom_sigma_profiles(
        scheme=sigma_scheme,
        max_abs_sigma=max_abs_sigma,
        num_points=num_points,
        num_threads=num_threads,
        shift=True,
    )
    new_atom_areas = atom_sigma_profiles.areas
    new_atom_charges = atom_sigma_profiles.charges
    sigma_profiles = atom_sigma_profiles.profiles
    elapsed_time = time.time() - start_time

    print(f"Time to compute atom sigma profiles: {elapsed_time:.2f} seconds")

    assert np.allclose(atom_areas, new_atom_areas), "Atom areas do not match"
    if sigma_scheme is None:
        assert np.allclose(atom_charges, new_atom_charges, atol=1e-6), (
            "Atom charges do not match"
        )
    else:
        print(
            "Atom charges vs scheme-averaged equivalent charge, max abs "
            f"diff: {np.abs(atom_charges - new_atom_charges).max():.3e}"
        )

    has_area = new_atom_areas > 0
    print(f"Atoms with no surface segments: {(~has_area).sum()} of {total_num_atoms}")
    centered_profiles = sigma_profiles[has_area]
    assert np.allclose(centered_profiles.sum(axis=1), 1.0), (
        "Sigma profiles are not normalized"
    )
    assert not sigma_profiles[~has_area].any(), (
        "Atoms with no surface segments must have all-zero profiles"
    )

    shifted_sigma_grid = atom_sigma_profiles.sigma_grid
    shifted_num_points = len(shifted_sigma_grid)
    mass_below_zero = centered_profiles[:, : shifted_num_points // 2].sum(axis=1)
    mass_above_zero = centered_profiles[:, shifted_num_points // 2 :].sum(axis=1)
    print_stats("Mass below zero", mass_below_zero)
    print_stats("Mass above zero", mass_above_zero)

    leading_zeros = np.argmax(centered_profiles > 0, axis=1)
    trailing_zeros = np.argmax(centered_profiles[:, ::-1] > 0, axis=1)
    print_stats(
        "Leading zeros",
        leading_zeros,
        quantiles=(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999),
    )
    print_stats(
        "Trailing zeros",
        trailing_zeros,
        quantiles=(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999),
    )

    assert not np.any(shifted_sigma_grid == 0.0), (
        "Shifted atom sigma grid must not contain a zero point"
    )
    first_moments = centered_profiles.astype(np.float64) @ shifted_sigma_grid
    print_stats("First moments", first_moments, value_format=".3e")
    print(
        f"Max |first moment| of centered profiles: "
        f"{np.abs(first_moments).max() / bin_width:.2e} bin widths"
    )

    start_time = time.time()
    molecule_areas = compute_per_molecule_properties(areas, segment_offsets)
    molecule_charges = compute_per_molecule_properties(charges, segment_offsets)
    elapsed_time = time.time() - start_time
    print(f"Time to compute per-molecule properties: {elapsed_time:.2f} seconds")
    print_stats("Molecule areas", molecule_areas)
    print_stats("Molecule charges", molecule_charges)

    start_time = time.time()
    atom_offsets = molecules_df["atom_offsets"].values.astype("int64")
    molecule_areas = compute_per_molecule_properties(atom_areas, atom_offsets)
    molecule_charges = compute_per_molecule_properties(atom_charges, atom_offsets)
    elapsed_time = time.time() - start_time
    print(f"Time to compute per-molecule properties: {elapsed_time:.2f} seconds")
    print_stats("Molecule areas", molecule_areas)
    print_stats("Molecule charges", molecule_charges)

    start_time = time.time()
    molecule_sigma_profiles = atom_sigma_profiles.aggregate(
        shift=True, num_threads=num_threads
    )
    molecule_profiles = molecule_sigma_profiles.profiles
    elapsed_time = time.time() - start_time
    print(f"Time to compute molecule sigma profiles: {elapsed_time:.2f} seconds")

    mass_err = np.abs(molecule_profiles.sum(axis=1) / molecule_areas - 1)
    print(
        f"Molecule profile mass conservation, max relative error: {mass_err.max():.2e}"
    )
    assert mass_err.max() < 1e-4, "Molecule sigma profiles do not conserve area"
    assert (molecule_profiles >= 0).all(), "Molecule sigma profiles have negative bins"

    assert molecule_profiles.shape[1] == num_points, (
        "Molecule sigma profiles must be on the unshifted grid"
    )

    unshifted_atom_sigma_profiles = store.compute_atom_sigma_profiles(
        scheme=sigma_scheme,
        max_abs_sigma=max_abs_sigma,
        num_points=num_points,
        num_threads=num_threads,
        shift=False,
    )
    exact_profiles = unshifted_atom_sigma_profiles.aggregate(
        shift=False, num_threads=num_threads
    ).profiles
    start_time = time.time()
    direct_areas, _, direct_profiles = compute_per_molecule_sigma_profiles(
        sigmas,
        areas,
        segment_offsets,
        len(molecules_df),
        max_abs_sigma=max_abs_sigma,
        num_points=num_points,
        num_threads=num_threads,
        shift=False,
    )
    elapsed_time = time.time() - start_time
    print(f"Time to bin molecule sigma profiles directly: {elapsed_time:.2f} seconds")
    assert np.allclose(direct_areas, molecule_areas, rtol=1e-5), (
        "Directly binned molecule areas do not match the per-atom sums"
    )
    ground_truth = direct_areas[:, None].astype(np.float64) * direct_profiles
    assert np.allclose(exact_profiles, ground_truth, atol=1e-3), (
        "Unshifted molecule sigma profiles must match direct segment binning"
    )

    normalized_recon = molecule_profiles / molecule_profiles.sum(axis=1, keepdims=True)
    normalized_truth = ground_truth / ground_truth.sum(axis=1, keepdims=True)
    profile_w1 = np.abs(
        np.cumsum(normalized_truth, axis=1) - np.cumsum(normalized_recon, axis=1)
    ).sum(axis=1)
    print_stats(
        "Molecule profile W1 vs direct binning (bin widths)",
        profile_w1,
        value_format=".4f",
    )

    normalized_molecule_profiles = atom_sigma_profiles.aggregate(
        shift=True, num_threads=num_threads, normalize=True
    ).profiles
    assert np.allclose(normalized_molecule_profiles.sum(axis=1), 1.0), (
        "Normalized molecule sigma profiles do not sum to 1"
    )
