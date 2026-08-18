"""Cluster-preserving train/val/test splitting of molecules.

Used by ``SegmentStore.assign_splits`` to partition stored molecules into
named splits (e.g. ``"train"``/``"val"``/``"test"``) without letting
near-duplicate structures -- as identified by ``cluster_id`` (see
``clustering.py``) -- leak across splits.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from cosmolayer.store._chalcedon.greedy_cluster_split import (
    greedy_cluster_split as _chalcedon_greedy_cluster_split,
)


def greedy_cluster_split(
    cluster_ids: NDArray[np.int64], fractions: Mapping[str, float]
) -> NDArray[np.str_]:
    """Assign each molecule a split name, keeping every cluster intact.

    Delegates to chalcedon's greedy LPT-scheduling split (vendored in
    ``cosmolayer.store._chalcedon``): clusters are assigned, largest first,
    to whichever split is currently furthest below its target fraction, so
    a whole cluster always lands in a single split.

    Parameters
    ----------
    cluster_ids : np.ndarray, shape (n,)
        Cluster id per molecule, as produced by ``clustering.butina_cluster``.
    fractions : Mapping[str, float]
        Target fraction per split name, e.g. ``{"train": 0.8, "val": 0.1,
        "test": 0.1}`` or ``{"train": 0.8, "test": 0.2}``. Values must be
        positive and sum to 1.0.

    Returns
    -------
    np.ndarray, shape (n,)
        Split name assigned to each molecule, in ``cluster_ids`` order.

    Raises
    ------
    ValueError
        If ``fractions`` is empty, contains non-positive values, or doesn't
        sum to 1.0 within 1e-6.
    """
    labels = np.empty(len(cluster_ids), dtype=object)
    splits = _chalcedon_greedy_cluster_split(cluster_ids, dict(fractions))
    for name, indices in splits.items():
        labels[indices] = name
    return labels.astype(str)
