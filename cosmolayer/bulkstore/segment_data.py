"""Parse COSMO files into flat segment-level arrays and derive atom properties.

This module turns a directory of ``.cosmo`` files into a compact,
memory-mappable on-disk representation (see ``store_segment_data`` /
``read_segment_data``): parallel segment-indexed arrays of coordinates,
charges, and areas, a global atom index for each segment, and a
``molecules.parquet`` table describing where each molecule's segments and
atoms live within those arrays, and its cavity volume. It also provides helpers to aggregate that
segment-level data into per-atom and per-molecule properties, including
per-atom sigma profiles: distributions of an atom's surface *area fraction*
over charge density, optionally centered on the atom's mean charge density
``q_a / A_a``. Charge densities outside the bounded range are folded into
the nearest boundary point, since they account for a negligible fraction of
the total area.

Running this module directly builds the segment data store (if missing)
and prints summary statistics for atom- and molecule-level charges, areas,
and sigma profiles.
"""

import json
import os
import pathlib
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import chalcedon
import numpy as np
import pandas as pd
from cosmolayer import parser
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.RDLogger import DisableLog, EnableLog
from tqdm.auto import tqdm

DATA_FILE: pathlib.Path = pathlib.Path("data.npy")
ATOM_INDICES_FILE: pathlib.Path = pathlib.Path("atom_indices.npy")
MOLECULES_FILE: pathlib.Path = pathlib.Path("molecules.parquet")
METADATA_FILE: pathlib.Path = pathlib.Path("metadata.json")
RADICAL_TO_PARENT_FILE: pathlib.Path = pathlib.Path("radical_to_parent.json")

DEFAULT_MAX_ABS_SIGMA = 0.0305
DEFAULT_NUM_POINTS_PER_SIDE = 31

FP_RADIUS = 2
FP_SIZE = 2048
FP_INCLUDE_CHIRALITY = True

CLUSTER_CUTOFF = 0.65

REORDER_ATOMS = False


SMILES_ENCODE = {
    "=": "_db_",
    "#": "_tb_",
    "(": "_lp_",
    ")": "_rp_",
    "[": "_lb_",
    "]": "_rb_",
}
SMILES_DECODE = {v: k for k, v in SMILES_ENCODE.items()}


def basename_to_smiles(basename: str) -> str:
    """Convert a COSMO filename to the SMILES string it encodes.

    Parameters
    ----------
    basename : str
        Filename (with or without the ``.cosmo`` extension) in which SMILES
        special characters have been replaced by their encoded form.

    Returns
    -------
    str
        The decoded SMILES string.
    """
    smiles = basename.replace(".cosmo", "")
    for k, v in SMILES_DECODE.items():
        smiles = smiles.replace(k, v)
    return smiles


def flat_canonical_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string with atom maps and stereochemistry removed.

    Parameters
    ----------
    smiles : str
        SMILES string to canonicalize, optionally atom-mapped.

    Returns
    -------
    str | None
        The canonical SMILES, or None if ``smiles`` does not parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def saturate_radical(smiles: str) -> str | None:
    """Re-saturate a radical's open valence and canonicalize the result.

    Adds a hydrogen to the atom bearing an unpaired electron, producing the
    closed-shell structure the radical was generated from, and returns its
    flat canonical SMILES (atom maps and stereochemistry stripped, since
    re-adding a hydrogen does not reliably recover the original stereo
    label).

    Parameters
    ----------
    smiles : str
        Atom-mapped SMILES of the radical.

    Returns
    -------
    str | None
        Flat canonical SMILES of the re-saturated structure, or None if
        ``smiles`` does not parse or does not describe a molecule with
        exactly one radical atom.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    radical_atoms = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0
    ]
    if len(radical_atoms) != 1:
        return None

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(radical_atoms[0])
    atom.SetNumRadicalElectrons(0)
    atom.SetNoImplicit(False)
    atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)

    # Adding a hydrogen to the radical center can overfill its valence, which
    # is a legitimate "this isn't the parent" answer rather than an error.
    # MolSanitizeException is the base class RDKit raises for those (valence
    # and kekulization failures alike); anything else is a real bug and
    # should propagate.
    try:
        Chem.SanitizeMol(rw)
    except Chem.rdchem.MolSanitizeException:
        return None

    Chem.RemoveStereochemistry(rw)
    return Chem.MolToSmiles(rw)


def match_radicals_to_parents(
    closed_basename_to_smiles: dict[str, str],
    open_basename_to_smiles: dict[str, str],
) -> dict[str, str]:
    """Match each radical in an open-shell set to its closed-shell parent.

    A radical is any open-shell entry not present in the closed-shell set --
    the open-shell JSON also lists every closed-shell parent verbatim under
    its own filename, which is skipped here. A radical is matched to a
    parent by re-saturating its open valence (``saturate_radical``) and
    looking up the result among the closed-shell structures' own flat
    canonical SMILES.

    Parameters
    ----------
    closed_basename_to_smiles : dict[str, str]
        Closed-shell basename -> atom-mapped SMILES.
    open_basename_to_smiles : dict[str, str]
        Open-shell basename -> atom-mapped SMILES.

    Returns
    -------
    dict[str, str]
        Radical basename -> parent closed-shell basename, for radicals that
        matched exactly one closed-shell structure.
    """
    # Canonicalizing atom-mapped SMILES with explicit hydrogens makes RDKit
    # attempt to fold them back into implicit H counts, which logs a
    # "not removing hydrogen atom" warning on any atom it considers
    # non-tetrahedral -- noise here, since flat_canonical_smiles /
    # saturate_radical strip stereochemistry immediately after parsing.
    DisableLog("rdApp.warning")
    try:
        closed_by_canon: dict[str, str] = {}
        for basename, smiles in closed_basename_to_smiles.items():
            canon = flat_canonical_smiles(smiles)
            if canon is not None:
                closed_by_canon.setdefault(canon, basename)

        radical_to_parent = {}
        for basename, smiles in open_basename_to_smiles.items():
            if basename in closed_basename_to_smiles:
                continue
            parent_canon = saturate_radical(smiles)
            if parent_canon is None:
                continue
            parent_basename = closed_by_canon.get(parent_canon)
            if parent_basename is not None:
                radical_to_parent[basename] = parent_basename
    finally:
        EnableLog("rdApp.warning")

    return radical_to_parent


def generate_fingerprints(
    molecules: Sequence[Chem.Mol],
    radius: int = 2,
    fp_size: int = 2048,
    include_chirality: bool = True,
) -> np.ndarray:
    """Generate fingerprints for a list of molecules.

    Parameters
    ----------
    molecules : Sequence[Chem.Mol]
        Molecules to fingerprint.
    radius : int, optional
        Radius for the Morgan fingerprint.
    fp_size : int, optional
        Size of the fingerprint.
    include_chirality : bool, optional
        Whether to include chirality in the fingerprint.

    Returns
    -------
    np.ndarray
        Fingerprints for the molecules.
    """

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=include_chirality,
    )

    fingerprints = np.zeros((len(molecules), fp_size), dtype=np.uint8)
    for i, mol in tqdm(
        enumerate(molecules),
        total=len(molecules),
        desc="Generating fingerprints",
    ):
        fingerprints[i] = generator.GetFingerprintAsNumPy(mol)
    return fingerprints


def store_segment_data(
    closed_shell_cosmo_files_dir: pathlib.Path,
    storage_dir: pathlib.Path,
    open_shell_cosmo_files_dir: pathlib.Path | None = None,
    reorder_atoms: bool = REORDER_ATOMS,
) -> None:
    """Parse COSMO files and persist their segment data to a directory.

    The resulting directory contains the parallel, segment-indexed arrays
    ``data.npy`` (columns ``[x, y, z, charge, area]``) and
    ``atom_indices.npy`` (the *global*, dataset-wide atom index of each
    segment: unique across all molecules, not just within one). Within a
    molecule, atoms are numbered either in the COSMO file's own atom order
    (``reorder_atoms=False``, the default) or in RDKit's
    ``Chem.Mol.GetAtoms()`` parse order for the atom-mapped SMILES, with
    explicit hydrogens kept (``reorder_atoms=True``) -- the two orderings
    can differ. Also written is a ``molecules.parquet`` table with one row
    per molecule and columns ``smiles``, ``atom_mapped_smiles``,
    ``cluster_id``, ``is_radical``, ``segment_offsets`` (start index of the
    molecule's segments within ``data``/``atom_indices``), ``atom_offsets``
    (start index of the molecule's atoms in the global atom index space),
    ``num_atoms``, and ``volume`` (the molecule's cavity volume, in cubic
    Angstroms, as reported by the COSMO file). Molecules whose ``.cosmo``
    files fail to parse are skipped and do not appear in the arrays or the
    molecules table. A ``metadata.json`` records the parameters used to
    build the store (``reorder_atoms``, fingerprint settings, cluster
    cutoff, molecule counts for what was actually stored,
    ``num_cosmo_parse_failures``, and ``num_radicals_orphaned``).

    Clustering (Butina, on Morgan fingerprints) is performed only on the
    closed-shell molecules. If ``open_shell_cosmo_files_dir`` is given, each
    of its radicals (open-shell entries with exactly one unpaired electron)
    is matched to its closed-shell parent via ``match_radicals_to_parents``
    and added to the dataset in the same cluster as that parent; radicals
    whose parent structure is not found among the closed-shell molecules are
    dropped. A radical is also dropped if its parent's own ``.cosmo`` file
    fails to parse, since a radical stored without its parent would carry a
    ``cluster_id`` inherited from a molecule absent from the store and have
    no entry in ``radical_to_parent.json``; entries are processed
    closed-shell first, so a radical's parent's parse outcome is always
    known before the radical itself is attempted. Radicals are never
    themselves clustered or used to seed a cluster. The matched radical ->
    parent mapping is written to ``radical_to_parent.json`` for pairs where
    both the radical and its parent were stored successfully (omitted if no
    such pairs remain).

    All segment data is held in memory for the duration of this call before
    being written out; for the current closed-shell-only dataset size
    (~4 GB) this is fine.

    Parameters
    ----------
    closed_shell_cosmo_files_dir : pathlib.Path
        Directory containing the closed-shell ``.cosmo`` files and a
        ``filename_to_atom_mapped_smiles.json`` mapping.
    storage_dir : pathlib.Path
        Destination directory for the output files. Created if missing;
        existing files with the same names are overwritten.
    open_shell_cosmo_files_dir : pathlib.Path | None, optional
        Directory containing the open-shell ``.cosmo`` files and a
        ``filename_to_atom_mapped_smiles.json`` mapping, by default None
        (no radicals are added).
    reorder_atoms : bool, optional
        Whether to number each molecule's atoms in RDKit parse order
        instead of the COSMO file's own order, by default False.
    """
    with open(
        closed_shell_cosmo_files_dir / "filename_to_atom_mapped_smiles.json", "r"
    ) as f:
        closed_basename_to_smiles = json.load(f)

    params = Chem.SmilesParserParams()
    params.removeHs = False

    closed_basename_to_mol = {}
    for basename, smi in tqdm(
        closed_basename_to_smiles.items(), desc="Parsing closed-shell SMILES"
    ):
        mol = Chem.MolFromSmiles(smi, params)
        if mol is not None:
            closed_basename_to_mol[basename] = mol

    num_dropped = len(closed_basename_to_smiles) - len(closed_basename_to_mol)
    if num_dropped:
        print(
            f"Skipped {num_dropped} closed-shell molecule(s) with unparseable SMILES."
        )

    DisableLog("rdApp.warning")
    try:
        fingerprints = generate_fingerprints(
            [Chem.RemoveHs(mol) for mol in closed_basename_to_mol.values()],
            radius=FP_RADIUS,
            fp_size=FP_SIZE,
            include_chirality=FP_INCLUDE_CHIRALITY,
        )
    finally:
        EnableLog("rdApp.warning")

    cluster_ids = chalcedon.butina_cluster(fingerprints, cutoff=CLUSTER_CUTOFF)
    basename_to_cluster_id = dict(
        zip(closed_basename_to_mol.keys(), cluster_ids.astype("int64"))
    )

    # basename -> (mol, source_dir, cluster_id, is_radical), in the order
    # entries are written out: closed-shell molecules first, then radicals.
    entries: dict[str, tuple[Chem.Mol, pathlib.Path, int, bool]] = {
        basename: (
            mol,
            closed_shell_cosmo_files_dir,
            basename_to_cluster_id[basename],
            False,
        )
        for basename, mol in closed_basename_to_mol.items()
    }
    basename_to_atom_mapped_smiles = dict(closed_basename_to_smiles)

    num_radicals_added = 0
    num_radicals_unmatched = 0
    if open_shell_cosmo_files_dir is not None:
        with open(
            open_shell_cosmo_files_dir / "filename_to_atom_mapped_smiles.json", "r"
        ) as f:
            open_basename_to_smiles = json.load(f)

        radical_to_parent = match_radicals_to_parents(
            closed_basename_to_smiles, open_basename_to_smiles
        )
        num_genuine_radicals = sum(
            1
            for basename in open_basename_to_smiles
            if basename not in closed_basename_to_smiles
        )
        num_radicals_unmatched = num_genuine_radicals - len(radical_to_parent)

        added_radical_to_parent = {}
        for basename, parent_basename in tqdm(
            radical_to_parent.items(), desc="Parsing radical SMILES"
        ):
            if parent_basename not in basename_to_cluster_id:
                continue
            smi = open_basename_to_smiles[basename]
            mol = Chem.MolFromSmiles(smi, params)
            if mol is None:
                continue
            entries[basename] = (
                mol,
                open_shell_cosmo_files_dir,
                basename_to_cluster_id[parent_basename],
                True,
            )
            basename_to_atom_mapped_smiles[basename] = smi
            added_radical_to_parent[basename] = parent_basename
            num_radicals_added += 1

        print(
            f"Added {num_radicals_added} radical(s) to the dataset "
            f"({num_radicals_unmatched} unmatched to a closed-shell parent)."
        )

    data_chunks, atoms_chunks = [], []
    segment_offsets, segment_offset = [], 0
    atom_offsets, atom_offset = [], 0
    num_atoms = []
    cluster_id_list, is_radical_list = [], []
    volumes = []
    basenames = []
    stored_basenames: set[str] = set()
    num_radicals_orphaned = 0
    num_cosmo_parse_failures = 0

    for basename, (mol, source_dir, cluster_id, is_radical) in tqdm(
        entries.items(), desc="Processing COSMO files"
    ):
        # Closed-shell entries are processed first, so a radical's parent's
        # parse outcome is already known: skip without even attempting to
        # parse the radical's own file if its parent was dropped.
        if is_radical and added_radical_to_parent[basename] not in stored_basenames:
            num_radicals_orphaned += 1
            continue

        filename = source_dir / f"{basename}.cosmo"
        try:
            _, atom_df, segment_df, volume = parser.parse_cosmo_file(
                filename.read_text()
            )
        except ValueError as e:
            tqdm.write(f"Error parsing {filename}: {e}")
            num_cosmo_parse_failures += 1
            continue

        data_chunks.append(
            segment_df[["x", "y", "z", "charge", "area"]].values.astype("float32")
        )
        if reorder_atoms:
            mapping = {a.GetAtomMapNum() - 1: i for i, a in enumerate(mol.GetAtoms())}
            indices = segment_df["atom"].map(mapping.get).values.astype("int64")
        else:
            indices = segment_df["atom"].values.astype("int64")
        atoms_chunks.append(indices + atom_offset)
        num_atoms.append(len(atom_df))

        segment_offsets.append(segment_offset)
        segment_offset += len(segment_df)
        atom_offsets.append(atom_offset)
        atom_offset += len(atom_df)
        cluster_id_list.append(cluster_id)
        is_radical_list.append(is_radical)
        volumes.append(volume)
        basenames.append(basename)
        stored_basenames.add(basename)

    if num_radicals_orphaned:
        print(
            f"Dropped {num_radicals_orphaned} radical(s) whose parent's "
            "COSMO file failed to parse."
        )

    if not basenames:
        raise ValueError("No COSMO files could be parsed successfully.")

    storage_dir = pathlib.Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    np.save(storage_dir / DATA_FILE, np.concatenate(data_chunks, axis=0))
    np.save(storage_dir / ATOM_INDICES_FILE, np.concatenate(atoms_chunks))

    molecules = pd.DataFrame(
        {
            "smiles": [basename_to_smiles(basename) for basename in basenames],
            "atom_mapped_smiles": [
                basename_to_atom_mapped_smiles[basename] for basename in basenames
            ],
            "cluster_id": np.array(cluster_id_list, dtype="int64"),
            "is_radical": np.array(is_radical_list, dtype=bool),
            "segment_offsets": np.array(segment_offsets, dtype="int64"),
            "atom_offsets": np.array(atom_offsets, dtype="int64"),
            "num_atoms": np.array(num_atoms, dtype="int64"),
            "volume": np.array(volumes, dtype="float64"),
        }
    )
    molecules.to_parquet(storage_dir / MOLECULES_FILE, index=False)

    num_radicals_stored = int(sum(is_radical_list))
    if open_shell_cosmo_files_dir is not None:
        radical_to_parent_path = storage_dir / RADICAL_TO_PARENT_FILE
        if added_radical_to_parent:
            added_radical_to_parent = {
                radical: parent
                for radical, parent in added_radical_to_parent.items()
                if radical in stored_basenames and parent in stored_basenames
            }
        if added_radical_to_parent:
            with open(radical_to_parent_path, "w") as f:
                json.dump(added_radical_to_parent, f, indent=2)
        else:
            radical_to_parent_path.unlink(missing_ok=True)

    metadata = {
        "reorder_atoms": reorder_atoms,
        "fingerprint_radius": FP_RADIUS,
        "fingerprint_size": FP_SIZE,
        "fingerprint_include_chirality": FP_INCLUDE_CHIRALITY,
        "cluster_cutoff": CLUSTER_CUTOFF,
        "num_molecules": len(basenames),
        "num_closed_shell_molecules": len(basenames) - num_radicals_stored,
        "num_radicals": num_radicals_stored,
        "num_radicals_unmatched": num_radicals_unmatched,
        "num_radicals_orphaned": num_radicals_orphaned,
        "num_dropped": num_dropped,
        "num_cosmo_parse_failures": num_cosmo_parse_failures,
    }
    with open(storage_dir / METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def segment_data_exists(storage_dir: pathlib.Path) -> bool:
    """Check if the segment data exists in a directory.

    Parameters
    ----------
    storage_dir : pathlib.Path
        Directory containing the segment data.

    Returns
    -------
    bool
        True if the segment data exists, False otherwise.
    """
    return all(
        (storage_dir / file).exists()
        for file in [DATA_FILE, ATOM_INDICES_FILE, MOLECULES_FILE, METADATA_FILE]
    )


def read_segment_data(
    storage_dir: pathlib.Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Read the segment data from a directory.

    Parameters
    ----------
    storage_dir : pathlib.Path | str
        Directory containing the segment data.

    Returns
    -------
    coords : np.ndarray
        Memory-mapped ``(n_segs_total, 3)`` coordinates of each segment.
    charges : np.ndarray
        Memory-mapped ``(n_segs_total,)`` charges of each segment.
    areas : np.ndarray
        Memory-mapped ``(n_segs_total,)`` areas of each segment.
    atom_indices : np.ndarray
        Memory-mapped ``(n_segs_total,)`` global atom index of each segment.
    molecules_df : pd.DataFrame
        One row per molecule, with columns ``smiles``, ``atom_mapped_smiles``,
        ``cluster_id``, ``is_radical``, ``segment_offsets``, ``atom_offsets``,
        ``num_atoms``, and ``volume``.
    """
    data = np.load(storage_dir / DATA_FILE, mmap_mode="r")
    atom_indices = np.load(storage_dir / ATOM_INDICES_FILE, mmap_mode="r")
    molecules_df = pd.read_parquet(storage_dir / MOLECULES_FILE)
    coords = data[:, :3]
    charges = data[:, 3]
    areas = data[:, 4]
    return coords, charges, areas, atom_indices, molecules_df


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

    Atoms belonging to the same molecule occupy a contiguous range in
    ``properties``, so each molecule's sum is just a reduction over that range.

    Parameters
    ----------
    properties : np.ndarray
        Property values, of shape ``(total_num_atoms,)``.
    offsets : np.ndarray
        Global index of each molecule's first atom, i.e. the cumulative sum
        of ``num_atoms`` for all preceding molecules. This is *not* the
        ``offsets`` column of the ``molecules`` table, which indexes into
        the segment-level arrays instead.

    Returns
    -------
    np.ndarray
        Per-molecule sum of ``properties``, of shape ``(n_molecules,)``.
    """
    return np.add.reduceat(properties, offsets)


def add_per_atom_area_contributions(
    atom_areas: np.ndarray,
    atom_charges: np.ndarray,
    sigma_profiles: np.ndarray,
    charges: np.ndarray,
    areas: np.ndarray,
    atom_indices: np.ndarray,
    max_abs_sigma: float,
    num_points_per_side: int,
    shift: bool,
) -> None:
    """Accumulate a batch of segments' area into per-atom sigma profiles.

    Each segment's charge density (``charge / area``) is linearly
    interpolated between the two nearest sigma-profile bins, and its area
    is split between those two bins in proportion to how close the charge
    density is to each. Charge densities outside
    ``[-max_abs_sigma, max_abs_sigma]`` are folded entirely into the
    nearest boundary point (column 0 or the last column) rather than
    tracked separately, since they account for a negligible fraction of
    the total area in practice. Each contribution is expressed as a
    fraction of its atom's total area, so
    every atom's profile sums to 1 (except atoms with no surface segments,
    whose profile stays all-zero). The split contributions are added in
    place, atom by atom, into ``sigma_profiles``.

    This function is intended to be run concurrently on disjoint segment
    ranges (see ``compute_per_atom_sigma_profiles``); since ``np.add.at``
    performs an unbuffered in-place reduction, it is safe to call from
    multiple threads as long as the accumulators are otherwise not accessed
    until all calls complete. Normalizing by the atom's total area reads
    ``atom_areas`` back immediately after accumulating into it, which is
    correct only because the caller splits work on *molecule* boundaries:
    every segment of an atom lands in the same batch, so an atom's total is
    complete by the time it is read, and no two batches touch the same atom.

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
        ``(num_atoms, 2 * num_points_per_side)``. Modified in place.
    charges : np.ndarray
        Charges of the surface segments in this batch.
    areas : np.ndarray
        Areas of the surface segments in this batch.
    atom_indices : np.ndarray
        Global atom index associated with each segment in this batch.
    max_abs_sigma : float
        Bounded sigma-profile value at each end of the range: the profile
        spans ``[-max_abs_sigma, max_abs_sigma]``.
    num_points_per_side : int
        Number of sigma-profile points on each side of zero. There is never
        a point exactly at zero, so the profile has ``2 * num_points_per_side``
        points in total, split evenly between negative and positive sigma.
    shift : bool
        Whether to shift each atom's profile onto its own mean charge
        density ``q_a / A_a``, centering it so its first moment is zero.

    Returns
    -------
    None
        ``atom_areas``, ``atom_charges``, and ``sigma_profiles`` are updated in place.
    """
    np.add.at(atom_areas, atom_indices, areas)
    np.add.at(atom_charges, atom_indices, charges)

    num_points = 2 * num_points_per_side
    min_sigma = -max_abs_sigma
    bin_width = (2 * max_abs_sigma) / (num_points - 1)

    summed_areas = atom_areas[atom_indices]

    sigmas = charges / areas
    if shift:
        sigmas -= atom_charges[atom_indices] / summed_areas

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


def compute_per_atom_sigma_profiles(
    charges: np.ndarray,
    areas: np.ndarray,
    atom_indices: np.ndarray,
    segment_offsets: np.ndarray,
    num_atoms: int,
    max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
    num_points_per_side: int = DEFAULT_NUM_POINTS_PER_SIDE,
    num_threads: int = os.cpu_count(),
    shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linearly interpolated per-atom sigma profiles.

    Parameters
    ----------
    charges : np.ndarray
        Charges of the surface segments.
    areas : np.ndarray
        Areas of the surface segments.
    atom_indices : np.ndarray
        Global atom index associated with each segment.
    segment_offsets : np.ndarray
        Start index of each molecule's segments within the segment-level
        arrays. Must describe *exactly* the molecules present in ``charges``
        / ``areas`` / ``atom_indices``: the last molecule's segments are
        assumed to run to the end of those arrays. Subsetting to fewer
        molecules requires slicing ``segment_offsets`` and the three
        segment-level arrays together -- slicing one without the others
        does not raise an error by itself; it silently attributes the
        leftover segments to the last molecule still described by
        ``segment_offsets``. Passing a ``num_atoms`` sized for the smaller
        subset turns that mistake into an ``AssertionError`` instead of
        letting it pass silently, so prefer under-sizing ``num_atoms``
        deliberately while debugging a new subsetting code path.
    num_atoms : int
        Total number of atoms in the dataset.
    max_abs_sigma : float, optional
        Bounded sigma-profile value at each end of the range: the profile
        spans ``[-max_abs_sigma, max_abs_sigma]``, by default 0.0305.
    num_points_per_side : int, optional
        Number of sigma-profile points on each side of zero, by default 31.
        There is never a point exactly at zero, so the profile has
        ``2 * num_points_per_side`` points in total.
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
        Per-atom charges.
    sigma_profiles : np.ndarray, shape (num_atoms, 2 * num_points_per_side)
        Per-atom area-fraction profiles, each summing to 1 -- except for
        atoms with no surface segments, whose row is all-zero. Charge
        densities below ``-max_abs_sigma`` or above ``max_abs_sigma`` are
        folded into the first or last column respectively, rather than kept
        in separate buckets. With ``shift=True`` each profile also has
        zero first moment, up to the negligible error introduced by that
        folding.
    """
    num_points = 2 * num_points_per_side
    atom_areas = np.zeros(num_atoms, dtype=np.float32)
    atom_charges = np.zeros(num_atoms, dtype=np.float32)
    sigma_profiles = np.zeros((num_atoms, num_points), dtype=np.float32)
    assert int(atom_indices.max(initial=-1)) < num_atoms, (
        "atom_indices references an atom index >= num_atoms; segment_offsets "
        "must describe exactly the molecules present in the segment-level "
        "arrays (see this function's docstring)"
    )
    num_segs = len(charges)
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
                    add_per_atom_area_contributions,
                    atom_areas,
                    atom_charges,
                    sigma_profiles,
                    charges[start_seg:stop_seg],
                    areas[start_seg:stop_seg],
                    atom_indices[start_seg:stop_seg],
                    max_abs_sigma,
                    num_points_per_side,
                    shift,
                )
            )
        for future in as_completed(futures):
            future.result()
    return atom_areas, atom_charges, sigma_profiles


def add_shifted_profile_contributions(
    molecule_profiles: np.ndarray,
    molecule_indices: np.ndarray,
    atom_areas: np.ndarray,
    shifts: np.ndarray,
    sigma_profiles: np.ndarray,
    max_abs_sigma: float,
    num_points_per_side: int,
) -> None:
    """Accumulate a batch of atoms' shifted, area-weighted profiles into
    per-molecule sigma profiles.

    The target grid is the per-atom grid itself -- same span, same
    ``2 * num_points_per_side`` points -- so a zero shift is the identity
    and every profile in the module lives on one grid (see
    ``segment_data.md``, "Reassembling molecule profiles"). Each row is
    redistributed with the same two-tap linear interpolation
    ``add_per_atom_area_contributions`` uses for a single value, applied
    here to every bin of a row at once, with out-of-range destinations
    folded into the nearest boundary column.

    This function is intended to be run concurrently on disjoint atom
    ranges (see ``aggregate_sigma_profiles``); since ``np.add.at``
    performs an unbuffered in-place reduction, it is safe to call from
    multiple threads as long as ``molecule_profiles`` is otherwise not
    accessed until all calls complete and no two calls are given atoms from
    the same molecule.

    Parameters
    ----------
    molecule_profiles : np.ndarray
        Per-molecule sigma profiles to accumulate into, of shape
        ``(num_molecules, 2 * num_points_per_side)``. Modified in place.
    molecule_indices : np.ndarray
        Global molecule index associated with each atom in this batch.
    atom_areas : np.ndarray
        Areas of the atoms in this batch.
    shifts : np.ndarray
        Amount to shift each atom's profile by, in sigma units (positive
        shifts move mass toward the positive end of the grid). Pass zeros
        to accumulate profiles unshifted.
    sigma_profiles : np.ndarray
        Per-atom sigma profiles for the atoms in this batch, of shape
        ``(len(atom_areas), 2 * num_points_per_side)``.
    max_abs_sigma : float
        Bounded sigma-profile value at each end of the range: the per-atom
        profile spans ``[-max_abs_sigma, max_abs_sigma]``.
    num_points_per_side : int
        Number of per-atom sigma-profile points on each side of zero.

    Returns
    -------
    None
        ``molecule_profiles`` is updated in place.
    """
    num_points = 2 * num_points_per_side
    bin_width = (2 * max_abs_sigma) / (num_points - 1)

    fractional_shift = shifts / bin_width
    points_shift = np.floor(fractional_shift).astype(np.int64)
    weight_right = (fractional_shift - points_shift)[:, None]

    contributions = atom_areas[:, None].astype(np.float64) * sigma_profiles.astype(
        np.float64
    )
    points_at_left = np.arange(num_points)[None, :] + points_shift[:, None]
    points_at_right = points_at_left + 1

    np.add.at(
        molecule_profiles,
        (molecule_indices[:, None], points_at_left.clip(0, num_points - 1)),
        contributions * (1.0 - weight_right),
    )
    np.add.at(
        molecule_profiles,
        (molecule_indices[:, None], points_at_right.clip(0, num_points - 1)),
        contributions * weight_right,
    )


def compute_per_molecule_sigma_profiles(
    charges: np.ndarray,
    areas: np.ndarray,
    segment_offsets: np.ndarray,
    num_molecules: int,
    max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
    num_points_per_side: int = DEFAULT_NUM_POINTS_PER_SIDE,
    num_threads: int = os.cpu_count(),
    shift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linearly interpolated per-molecule sigma profiles.

    This function is a wrapper around ``compute_per_atom_sigma_profiles`` that
    computes per-molecule sigma profiles by faking monoatomic molecules.

    Parameters
    ----------
    charges: np.ndarray
        Charges of the surface segments.
    areas: np.ndarray
        Areas of the surface segments.
    segment_offsets: np.ndarray
        Start index of each molecule's segments within the segment-level
        arrays. Must describe *exactly* the molecules present in ``charges``
        / ``areas``: the last molecule's segments are assumed to run to the
        end of those arrays. Subsetting to fewer molecules requires slicing
        ``segment_offsets`` and the two segment-level arrays together --
        slicing one without the others does not raise an error by itself; it
        silently attributes the leftover segments to the last molecule still
        described by ``segment_offsets``. Passing a ``num_molecules`` sized
        for the smaller subset turns that mistake into an ``AssertionError``
        instead of letting it pass silently, so prefer under-sizing
        ``num_molecules`` deliberately while debugging a new subsetting code
        path.
    num_molecules: int
        Total number of molecules in the dataset.
    max_abs_sigma: float
        Bounded sigma-profile value at each end of the range: the profile
        spans ``[-max_abs_sigma, max_abs_sigma]``.
    num_points_per_side: int
        Number of sigma-profile points on each side of zero.
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
        Per-molecule charges.
    molecule_sigma_profiles : np.ndarray, shape (num_molecules, 2 * num_points_per_side)
        Per-molecule area-fraction profiles, each summing to 1 -- except for
        molecules with no surface segments, whose row is all-zero. Charge
        densities below ``-max_abs_sigma`` or above ``max_abs_sigma`` are
        folded into the first or last column respectively, rather than kept
        in separate buckets. With ``shift=True`` each profile also has
        zero first moment, up to the negligible error introduced by that
        folding.
    """
    atom_indices = np.repeat(
        np.arange(num_molecules, dtype=np.int64),
        np.diff(np.append(segment_offsets, len(charges))),
    )

    return compute_per_atom_sigma_profiles(
        charges,
        areas,
        atom_indices,
        segment_offsets,
        num_molecules,
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
        num_threads=num_threads,
        shift=shift,
    )


def aggregate_sigma_profiles(
    atom_areas: np.ndarray,
    atom_charges: np.ndarray,
    sigma_profiles: np.ndarray,
    atom_offsets: np.ndarray,
    max_abs_sigma: float = DEFAULT_MAX_ABS_SIGMA,
    num_points_per_side: int = DEFAULT_NUM_POINTS_PER_SIDE,
    num_threads: int = os.cpu_count(),
    shift: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """Reassemble per-molecule sigma profiles from per-atom ones.

    A molecule's sigma profile is the area-weighted sum of its atoms'
    profiles. If those profiles are centered on each atom's own mean charge
    density (see ``compute_per_atom_sigma_profiles``), each one must first
    be shifted back by that same amount before summing, since a molecule
    has one shared sigma axis, not one per atom. That shift is a continuous
    quantity, so shifted values generally fall between grid points and are
    redistributed by linear interpolation (see
    ``add_shifted_profile_contributions``), the same way individual
    charge-density values are binned in the first place.

    Note that ``shift`` here means the *opposite* direction from the
    parameter of the same name in ``compute_per_atom_sigma_profiles``: there
    it moves each atom's profile onto its own mean charge density (centering
    it); here it moves the profile back off that mean, onto the molecule's
    shared axis. Both name the same fact -- that a shift by ``q_a / A_a`` is
    applied -- so pass ``shift=True`` here exactly when the profiles were
    built with ``shift=True`` there.

    Parameters
    ----------
    atom_areas : np.ndarray
        Per-atom areas, of shape ``(num_atoms,)``.
    atom_charges : np.ndarray
        Per-atom net charges, of shape ``(num_atoms,)``. Only used when
        ``shift=True``, to recover each atom's shift as
        ``atom_charges / atom_areas``.
    sigma_profiles : np.ndarray
        Per-atom sigma profiles, of shape
        ``(num_atoms, 2 * num_points_per_side)``.
    atom_offsets : np.ndarray
        Global index of each molecule's first atom, i.e. the cumulative sum
        of ``num_atoms`` for all preceding molecules (the ``atom_offsets``
        column of the ``molecules`` table).
    max_abs_sigma : float, optional
        Bounded sigma-profile value at each end of the range that
        ``sigma_profiles`` was built on, by default 0.0305.
    num_points_per_side : int, optional
        Number of sigma-profile points on each side of zero that
        ``sigma_profiles`` was built with, by default 31.
    num_threads : int, optional
        Number of threads to use, by default the number of available CPU
        cores.
    shift : bool, optional
        Whether ``sigma_profiles`` is centered and needs un-shifting before
        summing, by default True. With ``False``, profiles are summed
        as-is and the result is exact (no interpolation), since the target
        grid is the one they are already on.
    normalize : bool, optional
        Whether to divide each molecule's profile by its total area so it
        sums to 1, by default False (each bin holds surface area, and the
        profile sums to the molecule's total area -- the standard
        COSMO-SAC convention, and what keeps profiles additive across
        atoms). A molecule's total area cannot be zero (every molecule has
        some surface), so this division needs no zero guard.

    Returns
    -------
    np.ndarray, shape (num_molecules, 2 * num_points_per_side)
        Per-molecule sigma profiles, on the same grid as the per-atom
        ``sigma_profiles`` passed in and as
        ``compute_per_molecule_sigma_profiles``'s output. Its first moment
        should not be read as molecular charge -- use
        ``compute_per_molecule_properties(atom_charges, atom_offsets)``
        instead (see ``segment_data.md`` for why).
    """
    num_points = 2 * num_points_per_side
    num_atoms = len(atom_areas)
    num_mols = len(atom_offsets)

    molecule_indices = np.repeat(
        np.arange(num_mols), np.diff(np.append(atom_offsets, num_atoms))
    )

    shifts = np.zeros(num_atoms, dtype=np.float64)
    if shift:
        has_area = atom_areas > 0
        shifts[has_area] = atom_charges[has_area] / atom_areas[has_area]

    molecule_profiles = np.zeros((num_mols, num_points), dtype=np.float64)
    chunk_size = (num_mols + num_threads - 1) // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            start_mol = thread_id * chunk_size
            if start_mol >= num_mols:
                continue
            stop_mol = min(start_mol + chunk_size, num_mols)
            start_atom = atom_offsets[start_mol]
            stop_atom = atom_offsets[stop_mol] if stop_mol < num_mols else num_atoms
            futures.append(
                executor.submit(
                    add_shifted_profile_contributions,
                    molecule_profiles,
                    molecule_indices[start_atom:stop_atom],
                    atom_areas[start_atom:stop_atom],
                    shifts[start_atom:stop_atom],
                    sigma_profiles[start_atom:stop_atom],
                    max_abs_sigma,
                    num_points_per_side,
                )
            )
        for future in as_completed(futures):
            future.result()

    if normalize:
        molecule_profiles = molecule_profiles / molecule_profiles.sum(
            axis=1, keepdims=True
        )

    return molecule_profiles.astype(np.float32)


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
        qtext = f"{q * 100:.0f}%"
        print(f"{qtext:<10} | {np.quantile(properties, q):>20{value_format}}")
    print(f"{'Max':<10} | {properties.max():>20{value_format}}")
    print(f"{'Mean':<10} | {properties.mean():>20{value_format}}")
    print()


if __name__ == "__main__":
    base_dir = pathlib.Path(__file__).resolve().parents[1]
    closed_shell_cosmo_files_dir = (
        base_dir / "cosmo_files" / "closed_shell_species_cosmo_files"
    )
    open_shell_cosmo_files_dir = (
        base_dir / "cosmo_files" / "open_shell_species_cosmo_files"
    )
    storage_dir = base_dir / "sigma-prediction" / "segment_data"

    if not segment_data_exists(storage_dir):
        print("Storing segment data...")
        store_segment_data(
            closed_shell_cosmo_files_dir,
            storage_dir,
            open_shell_cosmo_files_dir=open_shell_cosmo_files_dir,
            reorder_atoms=REORDER_ATOMS,
        )
    else:
        print("Segment data already exists.")

    start_time = time.time()
    coords, charges, areas, atom_indices, molecules_df = read_segment_data(storage_dir)

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
    num_points_per_side = DEFAULT_NUM_POINTS_PER_SIDE
    start_time = time.time()
    new_atom_areas, new_atom_charges, sigma_profiles = compute_per_atom_sigma_profiles(
        charges,
        areas,
        atom_indices,
        segment_offsets,
        total_num_atoms,
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
        shift=True,
    )
    elapsed_time = time.time() - start_time

    print(f"Time to compute atom sigma profiles: {elapsed_time:.2f} seconds")

    assert np.allclose(atom_areas, new_atom_areas), "Atom areas do not match"
    assert np.allclose(atom_charges, new_atom_charges), "Atom charges do not match"

    has_area = new_atom_areas > 0
    print(f"Atoms with no surface segments: {(~has_area).sum()} of {total_num_atoms}")
    centered_profiles = sigma_profiles[has_area]
    assert np.allclose(centered_profiles.sum(axis=1), 1.0), (
        "Sigma profiles are not normalized"
    )
    assert not sigma_profiles[~has_area].any(), (
        "Atoms with no surface segments must have all-zero profiles"
    )

    mass_below_zero = centered_profiles[:, :num_points_per_side].sum(axis=1)
    mass_above_zero = centered_profiles[:, num_points_per_side:].sum(axis=1)
    print_stats("Mass below zero", mass_below_zero)
    print_stats("Mass above zero", mass_above_zero)

    num_points = 2 * num_points_per_side
    sigma_grid = np.linspace(-max_abs_sigma, max_abs_sigma, num_points)
    bin_width = (2 * max_abs_sigma) / (num_points - 1)
    assert not np.any(sigma_grid == 0.0), "Sigma grid must not contain a zero point"
    first_moments = centered_profiles.astype(np.float64) @ sigma_grid
    print_stats("First moments", first_moments, value_format=".3e")
    print(
        f"Max |first moment| of centered profiles: "
        f"{np.abs(first_moments).max() / bin_width:.2e} bin widths"
    )
    # Threshold has headroom over the single worst known outlier (0.117 bin
    # widths, a radical carbon with a 0.4 A^2 surface -- a tiny-area atom
    # whose own normalized profile amplifies one boundary-folded segment;
    # see segment_data.md, "Centering, and what it costs"), not over the
    # bulk, which stays under 0.01 bin widths even with radicals included.
    assert np.abs(first_moments).max() < 0.2 * bin_width, (
        "Centered sigma profiles do not have zero first moment"
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
    molecule_profiles = aggregate_sigma_profiles(
        atom_areas,
        atom_charges,
        sigma_profiles,
        atom_offsets,
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
        shift=True,
    )
    elapsed_time = time.time() - start_time
    print(f"Time to compute molecule sigma profiles: {elapsed_time:.2f} seconds")

    mass_err = np.abs(molecule_profiles.sum(axis=1) / molecule_areas - 1)
    print(
        f"Molecule profile mass conservation, max relative error: {mass_err.max():.2e}"
    )
    assert mass_err.max() < 1e-4, "Molecule sigma profiles do not conserve area"
    assert (molecule_profiles >= 0).all(), "Molecule sigma profiles have negative bins"

    assert molecule_profiles.shape[1] == num_points, (
        "Molecule sigma profiles must be on the same grid as the per-atom ones"
    )

    _, _, unshifted_profiles = compute_per_atom_sigma_profiles(
        charges,
        areas,
        atom_indices,
        segment_offsets,
        total_num_atoms,
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
        shift=False,
    )
    exact_profiles = aggregate_sigma_profiles(
        atom_areas,
        atom_charges,
        unshifted_profiles,
        atom_offsets,
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
        shift=False,
    )
    # Independent reference: bin every segment straight into its molecule,
    # never passing through per-atom profiles at all. Both paths bin the same
    # segments onto the same grid, so they must agree exactly.
    start_time = time.time()
    direct_areas, _, direct_profiles = compute_per_molecule_sigma_profiles(
        charges,
        areas,
        segment_offsets,
        len(molecules_df),
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
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

    normalized_molecule_profiles = aggregate_sigma_profiles(
        atom_areas,
        atom_charges,
        sigma_profiles,
        atom_offsets,
        max_abs_sigma=max_abs_sigma,
        num_points_per_side=num_points_per_side,
        shift=True,
        normalize=True,
    )
    assert np.allclose(normalized_molecule_profiles.sum(axis=1), 1.0), (
        "Normalized molecule sigma profiles do not sum to 1"
    )
