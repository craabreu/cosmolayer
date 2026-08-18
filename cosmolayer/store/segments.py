"""A segment-data store: on-disk arrays of COSMO segment coordinates,
charges, and areas, plus the per-molecule table describing them.

Build one from ``.cosmo`` files with ``SegmentStore.from_cosmo_files``, or
load an existing one with ``SegmentStore.load``. Both use a flat
``storage_dir``::

    <storage_dir>/data.npy          float32 (n_segs_total, 5): [x, y, z, charge, area]
    <storage_dir>/atom_indices.npy  int64   (n_segs_total,):   global atom index per
                                     segment
    <storage_dir>/molecules.parquet columns: smiles, segment_offsets, atom_offsets,
                                     num_atoms, volume, cluster_id, split (optional)
    <storage_dir>/metadata.json     {num_molecules, num_cosmo_parse_failures, schemes}
    <storage_dir>/<scheme>.npy      float32 (n_segs_total,): one per averaging scheme

``segment_offsets`` (the ``molecules_df`` column of that name) must describe
*exactly* the molecules present in the accompanying segment-level arrays:
the last molecule's segments run to the end of those arrays.
"""

import json
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem
from tqdm.auto import tqdm

from cosmolayer.parser import parse_cosmo_file

from .averaging import (
    AVERAGING_SCHEMES,
    AveragingScheme,
    average_sigmas_by_molecule,
)
from .clustering import ClusteringSpecs, FingerprintGenerator, butina_cluster
from .grid import DEFAULT_SIGMA_GRID, SigmaGrid
from .profiles import SigmaProfileTable
from .splitting import greedy_cluster_split
from .subsampling import apportion_counts, restrict_to_molecules

DATA_FILE = pathlib.Path("data.npy")
ATOM_INDICES_FILE = pathlib.Path("atom_indices.npy")
MOLECULES_FILE = pathlib.Path("molecules.parquet")
METADATA_FILE = pathlib.Path("metadata.json")

_STORE_FILES = (DATA_FILE, ATOM_INDICES_FILE, MOLECULES_FILE, METADATA_FILE)

# Averaged sigmas are written to "<name>.npy", so scheme names must not
# collide with the store's own .npy stems (data, atom_indices).
_RESERVED_SCHEME_NAMES = frozenset(
    f.stem for f in _STORE_FILES if f.suffix == DATA_FILE.suffix
)

# COSMO atom tables include explicit hydrogens; disable RDKit's default
# folding of terminal [H] so SMILES atoms match the COSMO file.
_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()
cast(Any, _SMILES_PARSER_PARAMS).removeHs = False


@dataclass
class StoreMetadata:
    """Typed view of a ``SegmentStore``'s ``metadata.json``.

    Parameters
    ----------
    num_molecules : int
        Number of molecules successfully stored.
    num_cosmo_parse_failures : int
        Number of molecules skipped because they failed to parse or
        validate (only possible with ``ignore_errors=True``, see
        ``SegmentStore.from_cosmo_files``).
    schemes : dict[str, AveragingScheme]
        Averaging schemes this store has computed sigmas for, keyed by
        scheme name.
    """

    num_molecules: int
    num_cosmo_parse_failures: int
    schemes: dict[str, AveragingScheme] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-compatible shape written to ``metadata.json``."""
        return {
            "num_molecules": self.num_molecules,
            "num_cosmo_parse_failures": self.num_cosmo_parse_failures,
            "schemes": {
                name: {
                    "averaging_radius": scheme.averaging_radius,
                    "f_decay": scheme.f_decay,
                }
                for name, scheme in self.schemes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoreMetadata":
        """Reconstruct from the dict produced by ``to_dict``.

        Parameters
        ----------
        data : dict
            As produced by ``to_dict``.
        """
        schemes = {
            name: AveragingScheme(name, params["averaging_radius"], params["f_decay"])
            for name, params in data.get("schemes", {}).items()
        }
        return cls(
            num_molecules=data["num_molecules"],
            num_cosmo_parse_failures=data["num_cosmo_parse_failures"],
            schemes=schemes,
        )


class SegmentStore:
    """On-disk COSMO segment arrays, the per-molecule table describing
    them, and any averaged sigmas computed for the store.

    Prefer ``load`` (read an existing store, memory-mapped) or
    ``from_cosmo_files`` (build a new one). Direct construction is for
    wrapping arrays already in hand.

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
    metadata : StoreMetadata
        Molecule and parse-failure counts, plus averaging schemes already
        computed for this store.
    averaged_sigmas : dict[str, np.ndarray]
        Scheme name to ``(n_segs_total,)`` averaged charge density. Empty
        if none have been computed.

    Attributes
    ----------
    coords, charges, areas : np.ndarray
        Views into ``data``'s columns.
    """

    def __init__(
        self,
        storage_dir: pathlib.Path,
        data: NDArray[np.float32],
        atom_indices: NDArray[np.int64],
        molecules_df: pd.DataFrame,
        metadata: StoreMetadata,
        averaged_sigmas: dict[str, NDArray[np.float32]],
    ) -> None:
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
        """Reorder atoms into ascending AtomMapNum order.

        After reordering, atom ``i`` has ``AtomMapNum == i`` (0-based) or
        ``AtomMapNum == i + 1`` (1-based), matching the COSMO file's
        0-based atom indices used as global atom indices.

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
            If the atom map numbers are not all 0, or a 0-based or
            1-based permutation of the atom indices.
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

    @classmethod
    def _parse_molecule(
        cls,
        cosmo_files_dir: pathlib.Path,
        filename: str,
        smi: str,
        fingerprint_generator: FingerprintGenerator,
    ) -> tuple[Chem.Mol, pd.DataFrame, pd.DataFrame, float, NDArray[np.int8]]:
        """Parse one ``.cosmo`` file/SMILES pair and fingerprint the
        molecule.

        Parameters
        ----------
        cosmo_files_dir : pathlib.Path
            Directory holding ``filename``.
        filename : str
            ``.cosmo`` filename, relative to ``cosmo_files_dir``.
        smi : str
            SMILES string for this molecule.
        fingerprint_generator : FingerprintGenerator
            Generator used to fingerprint the parsed molecule.

        Returns
        -------
        mol, atom_df, segment_df, volume, fingerprint
            The reordered molecule, its atom and segment tables, its
            volume, and its fingerprint.

        Raises
        ------
        ValueError
            If the SMILES can't be parsed, or its atom count doesn't
            match the ``.cosmo`` file's.
        """
        _, atom_df, segment_df, volume = parse_cosmo_file(
            (cosmo_files_dir / filename).read_text(encoding="utf-8", errors="replace")
        )
        mol = Chem.MolFromSmiles(smi, _SMILES_PARSER_PARAMS)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {smi!r}")
        if mol.GetNumAtoms() != len(atom_df):
            raise ValueError(
                f"SMILES {smi!r} has {mol.GetNumAtoms()} atoms, but "
                f"{filename} has {len(atom_df)}"
            )
        mol = cls._reorder_molecule(mol)
        fingerprint = fingerprint_generator.generate(mol)
        return mol, atom_df, segment_df, volume, fingerprint

    @staticmethod
    def _build_molecules_df(
        successful_molecules: list[str],
        segment_offsets: list[int],
        atom_offsets: list[int],
        num_atoms: list[int],
        volumes: list[float],
        cluster_ids: NDArray[np.int64],
    ) -> pd.DataFrame:
        """Assemble the per-molecule table written to ``molecules.parquet``."""
        return pd.DataFrame(
            {
                "smiles": successful_molecules,
                "segment_offsets": np.array(segment_offsets, dtype="int64"),
                "atom_offsets": np.array(atom_offsets, dtype="int64"),
                "num_atoms": np.array(num_atoms, dtype="int64"),
                "volume": np.array(volumes, dtype="float64"),
                "cluster_id": np.array(cluster_ids, dtype="int64"),
            }
        )

    def compute_averaged_sigmas(
        self,
        schemes: Sequence[AveragingScheme] | None = None,
        num_threads: int | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        """Compute averaged charge densities under each scheme, without
        writing to disk or mutating this store.

        Parameters
        ----------
        schemes : Sequence[AveragingScheme] | None, optional
            Schemes to apply. ``None`` (default) uses ``AVERAGING_SCHEMES``.
        num_threads : int | None, optional
            Thread count. ``None`` (default) uses every CPU core.

        Returns
        -------
        dict[str, np.ndarray]
            Scheme name to ``(n_segs_total,)`` float32 averaged charge
            density.

        Raises
        ------
        ValueError
            If a scheme's name would overwrite ``data.npy`` or
            ``atom_indices.npy`` on ``save``.
        """
        if schemes is None:
            schemes = AVERAGING_SCHEMES
        for scheme in schemes:
            if scheme.name in _RESERVED_SCHEME_NAMES:
                raise ValueError(
                    f"Averaging scheme name {scheme.name!r} collides with a "
                    f"reserved store filename ({sorted(_RESERVED_SCHEME_NAMES)})."
                )
        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")

        averaged = average_sigmas_by_molecule(
            np.asarray(self.coords, dtype=np.float64),
            np.asarray(self.charges, dtype=np.float64),
            np.asarray(self.areas, dtype=np.float64),
            segment_offsets,
            schemes,
            num_threads=num_threads,
        )
        return {
            scheme.name: arr.astype(np.float32)
            for scheme, arr in zip(schemes, averaged, strict=True)
        }

    def assign_splits(self, fractions: Mapping[str, float]) -> NDArray[np.str_]:
        """Partition this store's molecules into named splits (e.g.
        train/val/test) that respect ``cluster_id`` boundaries, and record
        the result as ``molecules_df["split"]``.

        Splitting is independent of ``ClusteringSpecs``: it's a cheap pass
        over the already-computed ``cluster_id`` column, so it can be
        called (and re-called, e.g. to try different fractions) on a
        freshly built or a loaded store, without recomputing fingerprints
        or clusters. Each call overwrites any previous ``split`` column.
        Call ``save`` afterward to persist the result.

        Parameters
        ----------
        fractions : Mapping[str, float]
            Target fraction per split name, e.g. ``{"train": 0.8, "val":
            0.1, "test": 0.1}`` or ``{"train": 0.8, "test": 0.2}``. Values
            must be positive and sum to 1.0.

        Returns
        -------
        np.ndarray, shape (n_molecules,)
            Split name assigned to each molecule, in ``molecules_df`` row
            order (the same array written to ``molecules_df["split"]``).

        Raises
        ------
        ValueError
            If ``fractions`` is empty, contains non-positive values, or
            doesn't sum to 1.0.
        """
        cluster_ids = self.molecules_df["cluster_id"].values.astype("int64")
        labels = greedy_cluster_split(cluster_ids, fractions)
        self.molecules_df["split"] = labels
        return labels

    def subsample(
        self, num_molecules: int, shuffle_seed: int | None = None
    ) -> "SegmentStore":
        """Return a new, unsaved store shrunk to ``num_molecules``, without
        moving any molecule across its existing ``split``.

        Requires ``molecules_df["split"]`` to already exist (see
        ``assign_splits``) -- split membership must be fixed *before*
        subsampling, since ``num_molecules`` is apportioned across the
        existing splits' own sizes and molecules are only ever dropped from
        within a split, never moved to another one. This is what keeps a
        test molecule from ending up in train/val just because the store
        got smaller.

        Parameters
        ----------
        num_molecules : int
            Total number of molecules to keep, summed across all splits.
            Must be positive and not exceed the current molecule count.
        shuffle_seed : int | None, optional
            If given, each split's share is a uniform random sample without
            replacement, drawn with this seed. ``None`` (default) instead
            keeps the first molecules in row order within each split, a
            deterministic choice.

        Returns
        -------
        SegmentStore
            A new store (see ``subsampling.restrict_to_molecules``) with
            the same ``storage_dir`` as this one -- pass a real directory
            to ``save`` to persist it.

        Raises
        ------
        ValueError
            If ``molecules_df`` has no ``split`` column, or if
            ``num_molecules`` is not positive or exceeds the current
            molecule count.
        """
        if "split" not in self.molecules_df.columns:
            raise ValueError(
                "molecules_df has no 'split' column; call assign_splits "
                "before subsample."
            )
        n_total = len(self.molecules_df)
        if num_molecules <= 0 or num_molecules > n_total:
            raise ValueError(
                f"num_molecules ({num_molecules}) must be positive and not "
                f"exceed the current molecule count ({n_total})."
            )

        split_labels = self.molecules_df["split"].to_numpy()
        split_names = list(dict.fromkeys(split_labels))
        members_by_split = [
            np.flatnonzero(split_labels == name) for name in split_names
        ]
        sizes = np.array([len(m) for m in members_by_split])
        target_counts = apportion_counts(sizes, num_molecules)

        rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        chosen_parts = []
        for members, raw_target in zip(members_by_split, target_counts, strict=True):
            target = int(raw_target)
            chosen_parts.append(
                members[:target]
                if rng is None
                else rng.choice(members, size=target, replace=False)
            )
        selected = np.sort(np.concatenate(chosen_parts))
        return restrict_to_molecules(self, selected)

    def save(self, storage_dir: pathlib.Path | str | None = None) -> None:
        """Write this store's arrays, table, metadata, and averaged
        sigmas to disk.

        ``metadata.json`` is written atomically (temp file + rename), so an
        interrupted save is never reported as complete by ``exists``.

        Parameters
        ----------
        storage_dir : pathlib.Path | str | None, optional
            Destination directory. ``None`` (default) uses
            ``self.storage_dir``. Created if missing.
        """
        storage_dir = (
            self.storage_dir if storage_dir is None else pathlib.Path(storage_dir)
        )
        storage_dir.mkdir(parents=True, exist_ok=True)

        np.save(storage_dir / DATA_FILE, self.data)
        np.save(storage_dir / ATOM_INDICES_FILE, self.atom_indices)
        self.molecules_df.to_parquet(storage_dir / MOLECULES_FILE, index=False)
        for name, arr in self.averaged_sigmas.items():
            np.save(storage_dir / f"{name}.npy", np.asarray(arr, dtype=np.float32))

        fd, tmp_path = tempfile.mkstemp(
            dir=storage_dir, prefix=".metadata-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            os.replace(tmp_path, storage_dir / METADATA_FILE)
        except BaseException:
            os.unlink(tmp_path)
            raise

    @classmethod
    def exists(cls, storage_dir: pathlib.Path | str) -> bool:
        """Return whether ``storage_dir`` holds a complete store.

        Requires the four fixed files and every scheme ``.npy`` listed in
        ``metadata.json``.

        Parameters
        ----------
        storage_dir : pathlib.Path | str
            Directory to check.

        Returns
        -------
        bool
            True if the store is complete.
        """
        storage_dir = pathlib.Path(storage_dir)
        if not all((storage_dir / f).exists() for f in _STORE_FILES):
            return False
        try:
            with open(storage_dir / METADATA_FILE) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        scheme_names = metadata.get("schemes", {})
        return all((storage_dir / f"{name}.npy").exists() for name in scheme_names)

    @classmethod
    def load(cls, storage_dir: pathlib.Path | str) -> "SegmentStore":
        """Load an existing segment-data store from disk, memory-mapped.

        Parameters
        ----------
        storage_dir : pathlib.Path | str
            Directory holding a store built by ``from_cosmo_files`` (i.e.
            for which ``exists`` is True).

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
        if not cls.exists(storage_dir):
            raise FileNotFoundError(
                f"No segment-data store in {storage_dir} (missing one of "
                f"{_STORE_FILES}, or a scheme .npy listed in {METADATA_FILE}; "
                "see SegmentStore.from_cosmo_files)."
            )
        with open(storage_dir / METADATA_FILE) as f:
            metadata = StoreMetadata.from_dict(json.load(f))
        data = np.load(storage_dir / DATA_FILE, mmap_mode="r")
        atom_indices = np.load(storage_dir / ATOM_INDICES_FILE, mmap_mode="r")
        molecules_df = pd.read_parquet(storage_dir / MOLECULES_FILE)

        averaged_sigmas = {
            name: np.load(storage_dir / f"{name}.npy", mmap_mode="r")
            for name in sorted(metadata.schemes)
        }

        return cls(
            storage_dir, data, atom_indices, molecules_df, metadata, averaged_sigmas
        )

    @classmethod
    def from_cosmo_files(  # noqa: PLR0913, PLR0917
        cls,
        cosmo_files_dir: pathlib.Path,
        smiles_to_filename: dict[str, str],
        storage_dir: pathlib.Path,
        ignore_errors: bool = False,
        schemes: Sequence[AveragingScheme] | None = None,
        clustering_specs: ClusteringSpecs | None = None,
        split_fractions: Mapping[str, float] | None = None,
        num_threads: int | None = None,
    ) -> "SegmentStore":
        """Parse COSMO files, build a store, and write it to ``storage_dir``.

        Atoms are numbered in the COSMO file's 0-based order. Each SMILES
        must have the same atom count as its COSMO file; atom-mapped SMILES
        are reordered onto that indexing. Averaged sigmas are computed and
        written unless ``schemes`` is an empty sequence.

        Parameters
        ----------
        cosmo_files_dir : pathlib.Path
            Directory containing the ``.cosmo`` files named by
            ``smiles_to_filename``'s values.
        smiles_to_filename : dict[str, str]
            SMILES string to ``.cosmo`` filename (relative to
            ``cosmo_files_dir``), one entry per molecule to store.
        storage_dir : pathlib.Path
            Destination directory for the output files. Created if
            missing.
        ignore_errors : bool, optional
            If True, skip molecules that fail to parse or validate and
            count them in ``metadata.num_cosmo_parse_failures``. Default
            False.
        schemes : Sequence[AveragingScheme] | None, optional
            Averaging schemes to compute. ``None`` (default) uses
            ``AVERAGING_SCHEMES``. Pass ``()`` to skip averaging.
        clustering_specs : ClusteringSpecs | None, optional
            Fingerprinting and Butina-clustering parameters, used to
            assign each molecule a ``cluster_id``. ``None`` (default)
            uses ``ClusteringSpecs()``.
        split_fractions : Mapping[str, float] | None, optional
            Target fraction per named split (e.g. ``{"train": 0.8, "val":
            0.1, "test": 0.1}``), assigned via ``assign_splits`` and
            written to ``molecules_df["split"]``. ``None`` (default) skips
            splitting; the ``split`` column is omitted. Can also be
            applied later, without rebuilding, via ``assign_splits`` on a
            loaded store.
        num_threads : int | None, optional
            Thread count for averaging. ``None`` (default) uses every CPU
            core.

        Returns
        -------
        SegmentStore
            The newly built and saved store.

        Raises
        ------
        ValueError
            If a molecule cannot be parsed and ``ignore_errors`` is False,
            if no molecule could be stored, or if a scheme name collides
            with a reserved store filename.
        """
        if clustering_specs is None:
            clustering_specs = ClusteringSpecs()

        data_chunks, atoms_chunks = [], []
        segment_offsets, segment_offset = [], 0
        atom_offsets, atom_offset = [], 0
        fingerprints = []
        num_atoms = []
        volumes = []
        successful_molecules = []
        num_cosmo_parse_failures = 0
        fingerprint_generator = FingerprintGenerator(clustering_specs)

        for smi, filename in tqdm(
            smiles_to_filename.items(), desc="Processing COSMO files"
        ):
            try:
                mol, atom_df, segment_df, volume, fingerprint = cls._parse_molecule(
                    cosmo_files_dir, filename, smi, fingerprint_generator
                )
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
            fingerprints.append(fingerprint)
            segment_offsets.append(segment_offset)
            segment_offset += len(segment_df)
            atom_offsets.append(atom_offset)
            atom_offset += len(atom_df)
            num_atoms.append(len(atom_df))
            volumes.append(volume)
            successful_molecules.append(Chem.MolToSmiles(mol))

        if not successful_molecules:
            raise ValueError("No COSMO files could be parsed successfully.")

        fingerprint_array = np.stack(fingerprints, axis=0)
        cluster_ids = butina_cluster(fingerprint_array, clustering_specs.cutoff)

        data = np.concatenate(data_chunks, axis=0)
        atom_indices = np.concatenate(atoms_chunks)
        molecules_df = cls._build_molecules_df(
            successful_molecules,
            segment_offsets,
            atom_offsets,
            num_atoms,
            volumes,
            cluster_ids,
        )
        metadata = StoreMetadata(
            num_molecules=len(successful_molecules),
            num_cosmo_parse_failures=num_cosmo_parse_failures,
        )

        store = cls(
            pathlib.Path(storage_dir), data, atom_indices, molecules_df, metadata, {}
        )
        if schemes is None or schemes:
            resolved_schemes = AVERAGING_SCHEMES if schemes is None else schemes
            store.averaged_sigmas = store.compute_averaged_sigmas(
                schemes=resolved_schemes, num_threads=num_threads
            )
            store.metadata.schemes.update({s.name: s for s in resolved_schemes})
        if split_fractions is not None:
            store.assign_splits(split_fractions)
        store.save()
        return store

    def sigmas(self, scheme: str | None = None) -> NDArray[np.float64]:
        """Resolve this store's segment charge density for a given
        scheme.

        Parameters
        ----------
        scheme : str | None, optional
            Which charge density to return: None (default) returns raw
            ``charges / areas``; a scheme name returns
            ``self.averaged_sigmas[scheme]`` (populated automatically by
            ``from_cosmo_files``).

        Returns
        -------
        np.ndarray, shape (n_segs_total,)
            Segment charge density, in e/Å².

        Raises
        ------
        KeyError
            If ``scheme`` is given but not in ``self.averaged_sigmas``.
        """
        if scheme is None:
            charges = np.asarray(self.charges, dtype=np.float64)
            areas = np.asarray(self.areas, dtype=np.float64)
            return charges / areas
        if scheme not in self.averaged_sigmas:
            raise KeyError(
                f"No averaged sigmas for scheme {scheme!r} in this store. "
                f"Known schemes: {sorted(self.averaged_sigmas)}."
            )
        return np.asarray(self.averaged_sigmas[scheme], dtype=np.float64)

    def compute_atom_sigma_profiles(
        self,
        scheme: str | None = None,
        grid: SigmaGrid = DEFAULT_SIGMA_GRID,
        num_threads: int | None = None,
        centered: bool = False,
    ) -> SigmaProfileTable:
        """Compute per-atom sigma profiles for this store.

        Parameters
        ----------
        scheme : str | None, optional
            Charge-density source. ``None`` (default) uses raw
            ``charges / areas``; a name selects ``averaged_sigmas[scheme]``.
        grid : SigmaGrid, optional
            Grid to bin onto, by default ``DEFAULT_SIGMA_GRID``.
        num_threads : int | None, optional
            Thread count. ``None`` (default) uses every CPU core.
        centered : bool, optional
            If True, center each atom's profile on its mean charge density.

        Returns
        -------
        SigmaProfileTable
            One row per atom.

        Raises
        ------
        KeyError
            If ``scheme`` is given but not in ``averaged_sigmas``.
        """
        sigmas = self.sigmas(scheme)
        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")
        total_num_atoms = int(self.molecules_df["num_atoms"].sum())
        return SigmaProfileTable.from_segments(
            sigmas,
            np.asarray(self.areas, dtype=np.float64),
            segment_offsets,
            atom_indices=np.asarray(self.atom_indices),
            num_rows=total_num_atoms,
            grid=grid,
            centered=centered,
            num_threads=num_threads,
        )

    def compute_molecule_sigma_profiles(
        self,
        scheme: str | None = None,
        grid: SigmaGrid = DEFAULT_SIGMA_GRID,
        num_threads: int | None = None,
        centered: bool = False,
    ) -> SigmaProfileTable:
        """Compute per-molecule sigma profiles from segment-level data.

        Bins segments directly to molecules, without an atom-level
        intermediate.

        Parameters
        ----------
        scheme : str | None, optional
            Charge-density source. ``None`` (default) uses raw
            ``charges / areas``; a name selects ``averaged_sigmas[scheme]``.
        grid : SigmaGrid, optional
            Grid to bin onto, by default ``DEFAULT_SIGMA_GRID``.
        num_threads : int | None, optional
            Thread count. ``None`` (default) uses every CPU core.
        centered : bool, optional
            If True, center each molecule's profile on its mean charge
            density.

        Returns
        -------
        SigmaProfileTable
            One row per molecule.
        """
        sigmas = self.sigmas(scheme)
        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")
        return SigmaProfileTable.from_segments(
            sigmas,
            np.asarray(self.areas, dtype=np.float64),
            segment_offsets,
            num_rows=len(self.molecules_df),
            grid=grid,
            centered=centered,
            num_threads=num_threads,
        )


__all__ = [
    "DATA_FILE",
    "ATOM_INDICES_FILE",
    "MOLECULES_FILE",
    "METADATA_FILE",
    "StoreMetadata",
    "SegmentStore",
]
