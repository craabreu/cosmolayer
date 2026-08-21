"""CLI: build a segment-data store (if missing) and report summary
statistics for atom- and molecule-level charges, areas, and sigma
profiles.

Run as::

    cosmostore --storage-dir DIR \\
        --cosmo-files-dir DIR --filenames-to-smiles FILE.json
"""

import argparse
import json
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .binning import compute_per_atom_properties, compute_per_molecule_properties
from .grid import SigmaGrid
from .profiles import SigmaProfileTable
from .reporting import print_stats
from .segments import SegmentStore

_T = TypeVar("_T")


def _timed(label: str, fn: Callable[[], _T]) -> _T:
    """Run ``fn``, print how long it took under ``label``, and return its
    result."""
    start_time = time.time()
    result = fn()
    print(f"Time to {label}: {time.time() - start_time:.2f} seconds")
    return result


def get_parser() -> argparse.ArgumentParser:
    """Return the argument parser for cosmostore (used by sphinx-argparse)."""
    arg_parser = argparse.ArgumentParser(
        prog="cosmostore",
        description=(
            "Build the segment data store (if missing) and print summary "
            "statistics for atom- and molecule-level charges, areas, and "
            "sigma profiles."
        ),
    )
    arg_parser.add_argument(
        "--storage-dir",
        type=str,
        required=True,
        help="The directory to store (or load) the segment data.",
    )
    arg_parser.add_argument(
        "--cosmo-files-dir",
        type=str,
        default=None,
        help=(
            "Directory containing the .cosmo files referenced by "
            "--filenames-to-smiles. Required only if --storage-dir doesn't "
            "already hold a built store."
        ),
    )
    arg_parser.add_argument(
        "--filenames-to-smiles",
        type=str,
        default=None,
        help=(
            "Path to a JSON file mapping each .cosmo filename (relative "
            "to --cosmo-files-dir) to that file's atom-mapped SMILES. "
            "Required only if --storage-dir doesn't already hold a built "
            "store."
        ),
    )
    arg_parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help=(
            "Number of threads to use for every threaded step (segment "
            "averaging, sigma-profile binning). Defaults to every "
            "available CPU core -- lower this on a shared machine to "
            "leave headroom for other users."
        ),
    )
    arg_parser.add_argument(
        "--sigma-scheme",
        type=str,
        default=None,
        help=(
            "Averaging scheme for every statistic (e.g. 'cosmo-rs', "
            "'cosmo-sac-2002', 'cosmo-sac-2010'). The store must already "
            "contain that scheme. Default: raw sigma."
        ),
    )
    arg_parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable tqdm progress bars. By default, bars are shown for "
            "COSMO parsing, averaging, clustering, and sigma-profile work."
        ),
    )
    arg_parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help=(
            "Skip .cosmo files that fail to parse or validate when "
            "building a store. By default those failures abort the run."
        ),
    )
    return arg_parser


def _ensure_store_built(
    args: argparse.Namespace,
    arg_parser: argparse.ArgumentParser,
    storage_dir: pathlib.Path,
) -> None:
    """Build the store at ``storage_dir`` if it doesn't already exist."""
    if SegmentStore.exists(storage_dir):
        print("Segment data already exists.")
        return
    if args.cosmo_files_dir is None or args.filenames_to_smiles is None:
        arg_parser.error(
            "--cosmo-files-dir and --filenames-to-smiles are required "
            f"when --storage-dir ({storage_dir}) doesn't already hold "
            "a built store."
        )
    print("Storing segment data and averaged sigmas...")
    cosmo_files_dir = pathlib.Path(args.cosmo_files_dir)
    with open(args.filenames_to_smiles) as f:
        filename_to_smiles = json.load(f)
    _timed(
        "store segment data and averaged sigmas",
        lambda: SegmentStore.from_cosmo_files(
            cosmo_files_dir,
            filename_to_smiles,
            storage_dir,
            ignore_errors=args.ignore_errors,
            num_threads=args.num_threads,
            progress=not args.no_progress,
        ),
    )


def _report_atom_stats(
    store: SegmentStore,
    sigma_scheme: str | None,
    num_threads: int | None,
    *,
    progress: bool,
) -> tuple[SigmaProfileTable, NDArray[np.float32], NDArray[np.float32]]:
    """Print per-atom statistics and return the atom profile table, areas,
    and charges."""
    total_num_atoms = int(store.molecules_df["num_atoms"].sum())
    atom_charges, atom_areas = _timed(
        "compute per-atom properties",
        lambda: (
            compute_per_atom_properties(
                np.asarray(store.charges), store.atom_indices, total_num_atoms
            ),
            compute_per_atom_properties(
                np.asarray(store.areas), store.atom_indices, total_num_atoms
            ),
        ),
    )
    atom_segment_counts = np.bincount(store.atom_indices, minlength=total_num_atoms)
    print_stats("Atom charges", atom_charges)
    print_stats("Atom areas", atom_areas)
    print_stats("Atom segment counts", atom_segment_counts)

    atom_sigma_profiles = _timed(
        "compute atom sigma profiles",
        lambda: store.compute_atom_sigma_profiles(
            scheme=sigma_scheme,
            grid=SigmaGrid(),
            num_threads=num_threads,
            centered=True,
            progress=progress,
        ),
    )
    has_area = atom_sigma_profiles.areas > 0
    print(f"Atoms with no surface segments: {(~has_area).sum()} of {total_num_atoms}")
    first_moments = (
        atom_sigma_profiles.profiles[has_area].astype(np.float64)
        @ atom_sigma_profiles.sigma_values
    )
    print_stats("Atom profile first moments", first_moments, value_format=".3e")

    return atom_sigma_profiles, atom_areas, atom_charges


def _report_molecule_stats(
    store: SegmentStore,
    atom_sigma_profiles: SigmaProfileTable,
    atom_areas: NDArray[np.float32],
    atom_charges: NDArray[np.float32],
    num_threads: int | None,
    *,
    progress: bool,
) -> None:
    """Print per-molecule statistics derived from the atom-level results."""
    atom_offsets = store.molecules_df["atom_offsets"].values.astype("int64")
    molecule_areas, molecule_charges = _timed(
        "compute per-molecule properties",
        lambda: (
            compute_per_molecule_properties(atom_areas, atom_offsets),
            compute_per_molecule_properties(atom_charges, atom_offsets),
        ),
    )
    print_stats("Molecule areas", molecule_areas)
    print_stats("Molecule charges", molecule_charges)

    molecule_sigma_profiles = _timed(
        "compute molecule sigma profiles",
        lambda: atom_sigma_profiles.aggregate(
            num_threads=num_threads, progress=progress
        ),
    )
    mass_err = np.abs(molecule_sigma_profiles.profiles.sum(axis=1) / molecule_areas - 1)
    print(
        f"Molecule profile mass conservation, max relative error: {mass_err.max():.2e}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``cosmostore``.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Arguments to parse, by default None, meaning ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    arg_parser = get_parser()
    args = arg_parser.parse_args(argv)
    storage_dir = pathlib.Path(args.storage_dir)

    _ensure_store_built(args, arg_parser, storage_dir)
    store = _timed("load segment data", lambda: SegmentStore.load(storage_dir))

    if args.sigma_scheme is not None and args.sigma_scheme not in store.averaged_sigmas:
        arg_parser.error(
            f"No averaged sigmas for scheme {args.sigma_scheme!r} in "
            f"{storage_dir}. Known schemes: {sorted(store.averaged_sigmas)}."
        )

    progress = not args.no_progress
    atom_sigma_profiles, atom_areas, atom_charges = _report_atom_stats(
        store, args.sigma_scheme, args.num_threads, progress=progress
    )
    _report_molecule_stats(
        store,
        atom_sigma_profiles,
        atom_areas,
        atom_charges,
        args.num_threads,
        progress=progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
