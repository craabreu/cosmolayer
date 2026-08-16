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
from rdkit.ML.Cluster import Butina


@dataclass(frozen=True)
class ClusteringSpecs:
    """Parameters for fingerprinting and Butina-clustering molecules.

    Parameters
    ----------
    cutoff : float
        Tanimoto distance threshold: molecules within ``cutoff`` of a
        cluster centroid join that cluster. Default 0.5.
    radius : int
        Morgan fingerprint radius. Default 2.
    fp_size : int
        Morgan fingerprint bit-vector length. Default 2048.
    include_chirality : bool
        Whether the fingerprint distinguishes stereoisomers. Default True.
    """

    cutoff: float = 0.5
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

    def __init__(self, specs: ClusteringSpecs):
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


def _tanimoto_distances(fingerprints: NDArray[np.int8]) -> list[float]:
    """Condensed lower-triangle Tanimoto distance matrix, in the row-major
    ``i in range(n); j in range(i)`` order required by
    ``rdkit.ML.Cluster.Butina.ClusterData(..., isDistData=True)``.

    Parameters
    ----------
    fingerprints : np.ndarray, shape (n, fp_size)
        Dense 0/1 bit arrays, one row per molecule.

    Returns
    -------
    list[float]
        ``n * (n - 1) / 2`` pairwise Tanimoto distances.
    """
    fp = fingerprints.astype(np.float32)
    popcount = fp.sum(axis=1)
    intersection = fp @ fp.T
    union = popcount[:, None] + popcount[None, :] - intersection
    similarity = np.divide(
        intersection, union, out=np.ones_like(intersection), where=union > 0
    )
    distance = 1.0 - similarity
    rows, cols = np.tril_indices(fp.shape[0], k=-1)
    return cast(list[float], distance[rows, cols].tolist())


def butina_cluster(fingerprints: NDArray[np.int8], cutoff: float) -> NDArray[np.int64]:
    """Butina-cluster molecules by fingerprint Tanimoto distance.

    Pairwise distances are computed densely (``O(n**2)`` memory), as
    required by RDKit's Butina implementation, so this is intended for
    dataset-scale molecule counts rather than unbounded streaming.

    Parameters
    ----------
    fingerprints : np.ndarray, shape (n, fp_size)
        One fingerprint per molecule, as produced by
        ``FingerprintGenerator.generate``.
    cutoff : float
        Tanimoto distance threshold passed to
        ``rdkit.ML.Cluster.Butina.ClusterData``.

    Returns
    -------
    np.ndarray, shape (n,)
        Cluster id per molecule, in ``[0, num_clusters)``. Cluster ids are
        ordered by cluster formation order (largest neighbor lists first,
        per Butina's algorithm), not by input order.
    """
    n = fingerprints.shape[0]
    cluster_ids = np.empty(n, dtype=np.int64)
    if n == 0:
        return cluster_ids
    if n == 1:
        cluster_ids[0] = 0
        return cluster_ids

    distances = _tanimoto_distances(fingerprints)
    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    for cluster_id, members in enumerate(clusters):
        for member in members:
            cluster_ids[member] = cluster_id
    return cluster_ids
