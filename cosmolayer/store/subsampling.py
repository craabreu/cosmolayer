"""Shrink a SegmentStore to a molecule subset without leaking splits.

``SegmentStore.subsample`` shrinks a store to ``num_molecules`` while
keeping every molecule's ``split`` assignment (see ``splitting.py``) fixed:
molecules are only ever dropped from within a split, never moved to
another one, so a test molecule can never end up in train/val just
because the dataset got smaller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .segments import SegmentStore


def apportion_counts(sizes: NDArray[np.int64], total: int) -> NDArray[np.int64]:
    """Apportion ``total`` items across buckets, proportional to ``sizes``.

    Each bucket's share is fixed via largest-remainder (Hamilton)
    apportionment: raw proportional shares are floored, then the leftover
    units are handed out one at a time, in order of largest fractional
    remainder, to whichever bucket still has room -- a bucket's own
    ``sizes[i]`` doubles as its capacity, so none is ever asked for more
    than it has.

    Parameters
    ----------
    sizes : np.ndarray, shape (k,)
        Bucket sizes (also used as capacities).
    total : int
        Total number of items to apportion; must not exceed ``sizes.sum()``.

    Returns
    -------
    np.ndarray, shape (k,)
        Non-negative integer counts, each clamped to its bucket's size,
        summing to ``total``.
    """
    n_total = int(sizes.sum())
    raw_counts = total * sizes / n_total
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        fractional = raw_counts - counts
        headroom = sizes - counts
        top_up_order = np.argsort(-fractional, kind="stable")
        for bucket in top_up_order:
            if remainder == 0:
                break
            if headroom[bucket] > 0:
                counts[bucket] += 1
                headroom[bucket] -= 1
                remainder -= 1
    return np.minimum(counts, sizes)


def restrict_to_molecules(
    store: SegmentStore, selected: NDArray[np.int64]
) -> SegmentStore:
    """Return a new, unsaved store restricted to ``selected`` molecules.

    Segment-indexed arrays (``data``, each ``averaged_sigmas`` scheme) carry
    no molecule identity of their own, so they're sliced by a segment mask
    derived from ``selected``. ``atom_indices`` does carry atom identity
    (global, dataset-wide), so it's additionally rebased onto a new,
    compacted index space covering only the kept molecules' atoms;
    ``atoms_df`` is sliced by the same atom mask, preserving row order.

    Parameters
    ----------
    store : SegmentStore
        Store to restrict.
    selected : np.ndarray, shape (k,)
        Ascending-sorted row indices into ``store.molecules_df`` to keep.

    Returns
    -------
    SegmentStore
        A new store with the same ``storage_dir`` as ``store`` (a
        placeholder -- pass a real directory to ``save`` to persist it),
        holding only ``selected``'s molecules, segments, and atoms.
    """
    # Deferred: segments.py imports this module at load time (for
    # SegmentStore.subsample), so importing SegmentStore back at module
    # level here would be circular.
    from .segments import SegmentStore, StoreMetadata  # noqa: PLC0415

    molecules_df = store.molecules_df
    n_mols_total = len(molecules_df)
    n_segs_total = len(store.data)
    full_segment_offsets = molecules_df["segment_offsets"].to_numpy().astype("int64")
    full_atom_offsets = molecules_df["atom_offsets"].to_numpy().astype("int64")
    full_num_atoms = molecules_df["num_atoms"].to_numpy().astype("int64")

    segment_counts = np.diff(np.append(full_segment_offsets, n_segs_total))
    segment_molecule = np.repeat(np.arange(n_mols_total), segment_counts)
    atom_molecule = np.repeat(np.arange(n_mols_total), full_num_atoms)

    is_selected = np.zeros(n_mols_total, dtype=bool)
    is_selected[selected] = True
    segment_mask = is_selected[segment_molecule]
    atom_mask = is_selected[atom_molecule]

    new_atom_offset_by_molecule = np.zeros(n_mols_total, dtype=np.int64)
    new_atom_offset_by_molecule[selected] = np.concatenate(
        [[0], np.cumsum(full_num_atoms[selected])[:-1]]
    )

    kept_molecule = segment_molecule[segment_mask]
    data = np.asarray(store.data)[segment_mask]
    atom_indices = (
        np.asarray(store.atom_indices)[segment_mask]
        - full_atom_offsets[kept_molecule]
        + new_atom_offset_by_molecule[kept_molecule]
    )
    atoms_df = store.atoms_df.iloc[atom_mask].reset_index(drop=True)
    averaged_sigmas = {
        name: np.asarray(arr)[segment_mask]
        for name, arr in store.averaged_sigmas.items()
    }

    new_segment_offsets = np.concatenate(
        [[0], np.cumsum(segment_counts[selected])[:-1]]
    )
    new_molecules_df = molecules_df.iloc[selected].copy()
    new_molecules_df["segment_offsets"] = new_segment_offsets
    new_molecules_df["atom_offsets"] = new_atom_offset_by_molecule[selected]

    metadata = StoreMetadata(
        num_molecules=len(selected),
        num_cosmo_parse_failures=0,
        schemes=dict(store.metadata.schemes),
    )
    return SegmentStore(
        store.storage_dir,
        data,
        atom_indices,
        new_molecules_df,
        atoms_df,
        metadata,
        averaged_sigmas,
    )
