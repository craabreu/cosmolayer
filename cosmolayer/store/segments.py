"""A segment-data store: on-disk arrays of COSMO segment coordinates,
charges, and areas, plus the per-molecule table describing them.

Build one from ``.cosmo`` files with ``SegmentStore.from_cosmo_files``, or
load an existing one with ``SegmentStore.load``. Both work with a flat
``storage_dir`` holding these files, none of which change name or shape
under this package's revisions of the surrounding code::

    <storage_dir>/data.npy          float32 (n_segs_total, 5): [x, y, z, charge, area]
    <storage_dir>/atom_indices.npy  int64   (n_segs_total,):   global atom index per
                                     segment
    <storage_dir>/molecules.parquet columns: smiles, segment_offsets, atom_offsets,
                                     num_atoms, volume
    <storage_dir>/metadata.json     {num_molecules, num_cosmo_parse_failures, schemes}
    <storage_dir>/<scheme>.npy      float32 (n_segs_total,): one per averaging scheme

Every function taking ``segment_offsets`` (the ``molecules_df`` column of
that name) requires it to describe *exactly* the molecules present in the
accompanying segment-level arrays: the last molecule's segments are
assumed to run to the end of those arrays.
"""

import json
import os
import pathlib
import tempfile
from collections.abc import Sequence
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
from .grid import DEFAULT_SIGMA_GRID, SigmaGrid
from .profiles import SigmaProfileTable

DATA_FILE = pathlib.Path("data.npy")
ATOM_INDICES_FILE = pathlib.Path("atom_indices.npy")
MOLECULES_FILE = pathlib.Path("molecules.parquet")
METADATA_FILE = pathlib.Path("metadata.json")

_STORE_FILES = (DATA_FILE, ATOM_INDICES_FILE, MOLECULES_FILE, METADATA_FILE)

# A scheme's averaged sigmas are written to "<name>.npy" (see save()), so
# only the store's own *.npy files can actually collide with one -- not
# MOLECULES_FILE or METADATA_FILE, which have different suffixes and so
# can never share a name with a "<name>.npy" scheme output. Derived from
# _STORE_FILES itself rather than a second, hand-maintained set, so the
# two can't drift apart.
_RESERVED_SCHEME_NAMES = frozenset(
    f.stem for f in _STORE_FILES if f.suffix == DATA_FILE.suffix
)

# A .cosmo file's atom table always lists every atom explicitly, hydrogens
# included, and _reorder_molecule's atom-mapping contract requires each
# SMILES atom to correspond 1:1 with a COSMO atom -- so explicit hydrogens
# in a SMILES string must survive parsing as real atoms. RDKit's default
# parser silently folds ordinary terminal "[H]" atoms back into implicit
# valence on read (even when atom-mapped), which would desync the two atom
# orderings; removeHs=False disables that folding. SMILES with only
# implicit hydrogens (e.g. "CCO") are unaffected either way.
#
# Assigned through cast(Any, ...) rather than a plain attribute set:
# rdkit's generated stubs for this Boost.Python class have typed removeHs
# inconsistently across releases, and lint_env.yaml pins no rdkit
# version, so whatever mypy infers here can vary by CI run. cast(Any, ...)
# opts this one assignment out of that check unconditionally, instead of
# a `# type: ignore` that would itself be version-sensitive.
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
        """Serialize to the JSON-compatible shape written to
        ``metadata.json``.

        Returns
        -------
        dict
            ``{"num_molecules": ..., "num_cosmo_parse_failures": ...,
            "schemes": {name: {"averaging_radius": ..., "f_decay":
            ...}}}``.
        """
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
        """Reconstruct from the shape read back from ``metadata.json``.

        Parameters
        ----------
        data : dict
            As produced by ``to_dict``.

        Returns
        -------
        StoreMetadata
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
    metadata : StoreMetadata
        Molecule/failure counts for this store, plus whichever averaging
        schemes have been computed for it so far.
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
        data: NDArray[np.float32],
        atom_indices: NDArray[np.int64],
        molecules_df: pd.DataFrame,
        metadata: StoreMetadata,
        averaged_sigmas: dict[str, NDArray[np.float32]],
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

    def compute_averaged_sigmas(
        self,
        schemes: Sequence[AveragingScheme] | None = None,
        num_threads: int | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        """Compute averaged sigmas for this store under every scheme in
        ``schemes``.

        Pure: returns the computed arrays without writing anything or
        mutating ``self``. Called by ``from_cosmo_files``, which passes
        the result to ``save``. A store's averaged sigmas are meant to be
        computed once, at build time -- this is exposed mainly so
        ``from_cosmo_files`` (and tests) don't have to go through disk.

        Calls ``average_sigmas_by_molecule`` once, sharing the
        pairwise-distance computation across every scheme.

        Parameters
        ----------
        schemes : Sequence[AveragingScheme] | None, optional
            Schemes to average under, by default None, meaning
            ``AVERAGING_SCHEMES`` (Klamt, COSMO-SAC 2002, COSMO-SAC 2010).
        num_threads : int | None, optional
            Number of threads to use, by default None, meaning every
            available CPU core.

        Returns
        -------
        dict[str, np.ndarray]
            Scheme name -> ``(n_segs_total,)`` float32 averaged charge
            density, one entry per scheme in ``schemes``.

        Raises
        ------
        ValueError
            If a scheme's name collides with a reserved store filename
            stem (i.e. would overwrite ``DATA_FILE`` or
            ``ATOM_INDICES_FILE`` on ``save``).
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
            np.asarray(self.coords),
            np.asarray(self.charges),
            np.asarray(self.areas),
            segment_offsets,
            schemes,
            num_threads=num_threads,
        )
        return {
            scheme.name: arr.astype(np.float32)
            for scheme, arr in zip(schemes, averaged, strict=True)
        }

    def save(self, storage_dir: pathlib.Path | str | None = None) -> None:
        """Write this store's arrays, table, metadata, and averaged
        sigmas to disk.

        ``metadata.json`` is written atomically (temp file + rename), so a
        process interrupted mid-``save`` never leaves a store that
        ``exists()`` reports as complete but that ``load`` cannot actually
        read.

        Parameters
        ----------
        storage_dir : pathlib.Path | str | None, optional
            Destination directory, by default None, meaning
            ``self.storage_dir``. Created if missing.

        Returns
        -------
        None
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
        """Check whether a directory holds a complete segment-data store.

        Checks not just the four fixed files but every scheme ``.npy``
        listed in ``metadata.json``, so an interrupted build (metadata
        written before its scheme arrays) is correctly reported as
        incomplete.

        Parameters
        ----------
        storage_dir : pathlib.Path | str
            Directory to check.

        Returns
        -------
        bool
            True if the store is complete, False otherwise.
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
    def from_cosmo_files(
        cls,
        cosmo_files_dir: pathlib.Path,
        smiles_to_filename: dict[str, str],
        storage_dir: pathlib.Path,
        ignore_errors: bool = False,
        schemes: Sequence[AveragingScheme] | None = None,
        num_threads: int | None = None,
    ) -> "SegmentStore":
        """Parse COSMO files, build a store in memory, and persist it to
        ``storage_dir``.

        A molecule's atoms are numbered by the COSMO file's own 0-based
        order directly, which is why each SMILES's atom count is checked
        against its COSMO file and, if atom-mapped, reordered via
        ``_reorder_molecule`` onto that same indexing.

        Also computes and writes averaged sigmas for the new store (see
        ``compute_averaged_sigmas``), unless ``schemes`` is an empty
        sequence.

        Parameters
        ----------
        cosmo_files_dir : pathlib.Path
            Directory containing the ``.cosmo`` files named by
            ``smiles_to_filename``'s values.
        smiles_to_filename : dict[str, str]
            SMILES string -> ``.cosmo`` filename (relative to
            ``cosmo_files_dir``), one entry per molecule to store.
        storage_dir : pathlib.Path
            Destination directory for the output files. Created if
            missing.
        ignore_errors : bool, optional
            If True, a molecule that fails to parse or validate is skipped
            (and counted in ``metadata.num_cosmo_parse_failures``) instead
            of raising. By default False.
        schemes : Sequence[AveragingScheme] | None, optional
            Passed to ``compute_averaged_sigmas``, by default None,
            meaning ``AVERAGING_SCHEMES``. Pass ``()`` to skip averaging
            entirely.
        num_threads : int | None, optional
            Passed to ``compute_averaged_sigmas``, by default None,
            meaning every available CPU core.

        Returns
        -------
        SegmentStore
            The newly built and saved store.

        Raises
        ------
        ValueError
            If a molecule's SMILES or COSMO file cannot be parsed and
            ``ignore_errors`` is False, if no molecule could be stored, or
            if a scheme's name collides with a reserved store filename
            stem (see ``compute_averaged_sigmas``).
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
                _, atom_df, segment_df, volume = parse_cosmo_file(
                    (cosmo_files_dir / filename).read_text(
                        encoding="utf-8", errors="replace"
                    )
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
            successful_molecules.append(Chem.MolToSmiles(mol))

        if not successful_molecules:
            raise ValueError("No COSMO files could be parsed successfully.")

        data = np.concatenate(data_chunks, axis=0)
        atom_indices = np.concatenate(atoms_chunks)
        molecules_df = pd.DataFrame(
            {
                "smiles": successful_molecules,
                "segment_offsets": np.array(segment_offsets, dtype="int64"),
                "atom_offsets": np.array(atom_offsets, dtype="int64"),
                "num_atoms": np.array(num_atoms, dtype="int64"),
                "volume": np.array(volumes, dtype="float64"),
            }
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
        return np.asarray(self.averaged_sigmas[scheme])

    def compute_atom_sigma_profiles(
        self,
        scheme: str | None = None,
        grid: SigmaGrid = DEFAULT_SIGMA_GRID,
        num_threads: int | None = None,
        centered: bool = False,
    ) -> SigmaProfileTable:
        """Compute this store's per-atom sigma profiles.

        A thin wrapper around ``SigmaProfileTable.from_segments`` that
        supplies this store's own ``areas``/``atom_indices``/
        ``segment_offsets``, and resolves ``sigmas`` from ``scheme`` (see
        ``sigmas``).

        Parameters
        ----------
        scheme : str | None, optional
            Passed to ``sigmas``, by default None (raw charge density).
        grid : SigmaGrid, optional
            Base grid to bin profiles onto, by default ``DEFAULT_SIGMA_GRID``.
        num_threads : int | None, optional
            Number of threads to use, by default None, meaning every
            available CPU core.
        centered : bool, optional
            Whether to center each atom's profile on its own mean charge
            density, by default False.

        Returns
        -------
        SigmaProfileTable
            Atom-level.

        Raises
        ------
        KeyError
            If ``scheme`` is given but not in ``self.averaged_sigmas``.
        """
        sigmas = self.sigmas(scheme)
        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")
        total_num_atoms = int(self.molecules_df["num_atoms"].sum())
        return SigmaProfileTable.from_segments(
            sigmas,
            np.asarray(self.areas),
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
        """Compute this store's per-molecule sigma profiles directly from
        segment-level data.

        Equivalent to ``self.compute_atom_sigma_profiles(...).aggregate()``
        but skips the atom-level intermediate; mainly useful as a
        cross-check for that two-step path.

        Parameters
        ----------
        scheme : str | None, optional
            Passed to ``sigmas``, by default None (raw charge density).
        grid : SigmaGrid, optional
            Base grid to bin profiles onto, by default ``DEFAULT_SIGMA_GRID``.
        num_threads : int | None, optional
            Number of threads to use, by default None, meaning every
            available CPU core.
        centered : bool, optional
            Whether to center each molecule's profile on its own mean
            charge density, by default False.

        Returns
        -------
        SigmaProfileTable
            Molecule-level.
        """
        sigmas = self.sigmas(scheme)
        segment_offsets = self.molecules_df["segment_offsets"].values.astype("int64")
        return SigmaProfileTable.from_segments(
            sigmas,
            np.asarray(self.areas),
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
