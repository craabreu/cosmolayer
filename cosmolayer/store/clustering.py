"""Butina clustering of molecules by Morgan-fingerprint Tanimoto
similarity.

Used by ``SegmentStore.from_cosmo_files`` to assign each stored molecule a
``cluster_id`` (see the ``molecules.parquet`` column of that name), so
callers can build train/test splits or diversity subsets that don't leak
near-duplicate structures across a split.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm

from cosmolayer.store._chalcedon.butina_cluster import (
    butina_cluster as _chalcedon_butina_cluster,
)
from cosmolayer.store._chalcedon.tanimoto_similarity import TanimotoSimilarity


@dataclass(frozen=True)
class ClusteringSpecs:
    """Parameters for fingerprinting and Butina-clustering molecules.

    Parameters
    ----------
    cutoff : float
        Tanimoto distance threshold: molecules within ``cutoff`` of a
        cluster centroid join that cluster. Default 0.65.
    radius : int
        Morgan fingerprint radius. Default 2.
    fp_size : int
        Morgan fingerprint bit-vector length. Default 2048.
    include_chirality : bool
        Whether the fingerprint distinguishes stereoisomers. Default True.
    """

    cutoff: float = 0.65
    radius: int = 2
    fp_size: int = 2048
    include_chirality: bool = True


class FingerprintGenerator:
    """Generates Morgan fingerprints for molecules under a fixed
    ``ClusteringSpecs``.

    Parameters
    ----------
    specs : ClusteringSpecs
        Fingerprint parameters (radius, size, chirality).
    """

    def __init__(self, specs: ClusteringSpecs) -> None:
        self.specs = specs
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=specs.radius,
            fpSize=specs.fp_size,
            includeChirality=specs.include_chirality,
        )

    def generate(self, mol: Chem.Mol) -> NDArray[np.int8]:
        """Generate a molecule's fingerprint as a dense bit array.

        Parameters
        ----------
        mol : Chem.Mol
            Molecule to fingerprint.

        Returns
        -------
        np.ndarray, shape (fp_size,)
            Dense 0/1 bit array.
        """
        with rdBase.BlockLogs():
            fp = self._generator.GetFingerprintAsNumPy(mol).astype(np.int8)
        return cast(NDArray[np.int8], fp)


def butina_cluster(
    fingerprints: NDArray[np.int8], cutoff: float, *, progress: bool = False
) -> NDArray[np.int64]:
    """Butina-cluster molecules by fingerprint Tanimoto distance.

    Delegates to chalcedon's count-sort-assign Butina implementation
    (vendored in ``cosmolayer.store._chalcedon``), which produces
    the same partition as RDKit's reference implementation at typical
    cheminformatics cutoffs but scales substantially better with ``n``.
    Pairwise work is still effectively ``O(n**2)``, so this is intended
    for dataset-scale molecule counts rather than unbounded streaming.

    Parameters
    ----------
    fingerprints : np.ndarray, shape (n, fp_size)
        One fingerprint per molecule, as produced by
        ``FingerprintGenerator.generate``.
    cutoff : float
        Tanimoto distance threshold: molecules within ``cutoff`` of a
        cluster centroid join that cluster.
    progress : bool, optional
        If True, show chalcedon's tqdm bars (neighbor counting and
        cluster assignment) on stderr. Default False, so a library call
        stays quiet.

    Returns
    -------
    np.ndarray, shape (n,)
        Cluster id per molecule, in ``[0, num_clusters)``. Cluster ids are
        ordered by cluster formation order (largest neighbor lists first,
        per Butina's algorithm), not by input order.
    """
    n = fingerprints.shape[0]
    if n == 0:
        return np.empty(n, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    cluster_ids = _chalcedon_butina_cluster(
        fingerprints, cutoff=cutoff, progress=progress
    )
    return cluster_ids.astype(np.int64)


# Bound (batch, k) Tanimoto workspace so large clusters never allocate k×k.
_MEDOID_SCORE_MAX_CELLS = 4_000_000


def cluster_medoid_distances(
    fingerprints: NDArray[np.int8],
    cluster_ids: NDArray[np.int64],
    *,
    progress: bool = False,
) -> NDArray[np.float64]:
    """Tanimoto distance of each molecule to its cluster's Tanimoto medoid.

    The medoid of a cluster is the member that maximizes the sum of
    Tanimoto similarities to the other members (equivalently, minimizes
    the sum of Tanimoto distances). Ties break to the lowest row index.
    The medoid's own distance is exactly 0.

    Parameters
    ----------
    fingerprints : np.ndarray, shape (n, fp_size)
        One fingerprint per molecule, as produced by
        ``FingerprintGenerator.generate``.
    cluster_ids : np.ndarray, shape (n,)
        Cluster id per molecule, as produced by ``butina_cluster``.
    progress : bool, optional
        If True, show a tqdm bar over clusters. Default False.

    Returns
    -------
    np.ndarray, shape (n,)
        Tanimoto distance to that molecule's cluster medoid, ``float64``.

    Raises
    ------
    ValueError
        If ``fingerprints`` and ``cluster_ids`` have different lengths.
    """
    n = fingerprints.shape[0]
    if cluster_ids.shape[0] != n:
        raise ValueError(
            "fingerprints and cluster_ids must have the same length, "
            f"got {fingerprints.shape[0]} and {cluster_ids.shape[0]}."
        )
    distances = np.zeros(n, dtype=np.float64)
    if n == 0:
        return distances

    fps_all = np.asarray(fingerprints, dtype=np.float64)
    for cluster_id in tqdm(
        np.unique(cluster_ids),
        desc="Computing medoids",
        disable=not progress,
    ):
        members = np.flatnonzero(cluster_ids == cluster_id)
        k = int(members.shape[0])
        if k == 1:
            continue
        cluster_fps = fps_all[members]
        similarity = TanimotoSimilarity(cluster_fps, dtype="float64")
        batch = max(1, min(k, _MEDOID_SCORE_MAX_CELLS // k))
        scores = np.empty(k, dtype=np.float64)
        for start in range(0, k, batch):
            end = min(start + batch, k)
            scores[start:end] = similarity.chunk(start, end).sum(axis=1)
        local_medoid = int(np.argmax(scores))
        medoid_fp = cluster_fps[local_medoid]
        norms = np.einsum("ij,ij->i", cluster_fps, cluster_fps)
        dots = cluster_fps @ medoid_fp
        unions = norms + norms[local_medoid] - dots
        sims = np.divide(dots, unions, out=np.zeros_like(dots), where=unions > 0)
        dists = 1.0 - sims
        dists[local_medoid] = 0.0
        distances[members] = dists
    return distances
