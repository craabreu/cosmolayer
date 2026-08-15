"""COSMO-SAC-style distance-weighted averaging of segment charge density.

In COSMO-RS and COSMO-SAC, "sigma" refers to this averaged charge
density, not to the raw per-segment ``charge / area`` -- averaging is not
an optional smoothing step on top of "the real sigma", it *is* sigma.

Every function here that takes ``segment_offsets`` requires it to
describe *exactly* the molecules present in the accompanying segment-level
arrays: the last molecule's segments are assumed to run to the end of
those arrays. Slicing ``segment_offsets`` and the segment-level arrays for
a subset must always be done together, or leftover segments are silently
attributed to the wrong molecule.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cosmolayer.cosmosac.constants import (
    COSMO_SAC_2002_AVERAGING_RADIUS,
    COSMO_SAC_2002_F_DECAY,
    COSMO_SAC_2010_AVERAGING_RADIUS,
    COSMO_SAC_2010_F_DECAY,
)

from .parallel import run_in_threads


@dataclass(frozen=True)
class AveragingScheme:
    """One COSMO-SAC-style segment-averaging scheme.

    A pure value object describing the physics of a scheme -- it has no
    notion of storage, so it does not (and cannot, without an import
    cycle) validate its name against a ``SegmentStore``'s on-disk layout.
    ``SegmentStore`` becomes the stem of the ``<name>.npy`` file it writes
    this scheme's averaged sigmas to, and rejects a colliding name itself
    (see ``SegmentStore.compute_averaged_sigmas``).

    Parameters
    ----------
    name : str
        Scheme identifier.
    averaging_radius : float
        Effective averaging radius ``r_av``, in Å.
    f_decay : float
        Exponential decay factor.
    """

    name: str
    averaging_radius: float
    f_decay: float


COSMO_RS = AveragingScheme("cosmo-rs", averaging_radius=0.5, f_decay=1.0)
COSMO_SAC_2002 = AveragingScheme(
    "cosmo-sac-2002",
    averaging_radius=COSMO_SAC_2002_AVERAGING_RADIUS,
    f_decay=COSMO_SAC_2002_F_DECAY,
)
COSMO_SAC_2010 = AveragingScheme(
    "cosmo-sac-2010",
    averaging_radius=COSMO_SAC_2010_AVERAGING_RADIUS,
    f_decay=COSMO_SAC_2010_F_DECAY,
)

AVERAGING_SCHEMES: tuple[AveragingScheme, ...] = (
    COSMO_RS,
    COSMO_SAC_2002,
    COSMO_SAC_2010,
)
"""Every scheme this package knows about. Each carries its own ``name``,
so this is a plain sequence rather than a name-keyed mapping -- keying it
by name too would just be a second copy of information ``AveragingScheme``
already has."""


def average_sigmas(
    coords: NDArray[np.float64],
    charges: NDArray[np.float64],
    areas: NDArray[np.float64],
    schemes: Sequence[AveragingScheme],
) -> NDArray[np.float64]:
    """Distance-weighted average of one molecule's segment charge
    densities, under one or more averaging schemes at once.

    For every segment ``m``, replaces its raw charge density
    ``sigma_m = q_m / A_m`` with a weighted average over every segment
    ``n`` in the same molecule, including itself::

        sigma_avg[m] = sum_n(sigma[n] * w[m, n]) / sum_n(w[m, n])
        w[m, n] = (r_n^2 * r_av^2 / (r_n^2 + r_av^2))
                  * exp(-f_decay * d_mn^2 / (r_n^2 + r_av^2))

    where ``r_n = sqrt(A_n / pi)`` is segment ``n``'s own effective radius
    and ``d_mn`` is the distance between segment centroids ``m`` and
    ``n``. The weight uses the *neighbor* segment's radius ``r_n``, not
    the segment being averaged -- easy to get backwards, since it makes
    ``w`` asymmetric even though ``d_mn`` itself is symmetric.

    Accepts multiple schemes so they can share one pairwise-distance
    computation; each scheme's result is returned as its own row. O(n^2)
    in the molecule's own segment count, computed densely (no distance
    cutoff), matching the reference implementation exactly. Always upcast
    to float64 internally regardless of input dtype, since a store's own
    arrays are float32 and the Gram-matrix distance expansion loses
    significant precision at that width.

    The caller must never pass segments from more than one molecule at
    once (see ``average_sigmas_by_molecule`` for the whole-dataset
    wrapper).

    Parameters
    ----------
    coords : np.ndarray
        Segment centroid coordinates for one molecule, shape
        ``(n_segs, 3)``.
    charges : np.ndarray
        Segment charges for the same molecule, shape ``(n_segs,)``.
    areas : np.ndarray
        Segment areas for the same molecule, shape ``(n_segs,)``.
    schemes : Sequence[AveragingScheme]
        Schemes to average under, in the order the result rows are
        returned. COSMO-SAC 2010 uses ``r_av = sqrt(7.25 / pi)``,
        ``f_decay = 3.57``; COSMO-SAC 2002 uses ``r_av = 0.8176300195``,
        ``f_decay = 1.0``; Klamt's original COSMO-RS scheme uses
        ``r_av = 0.5``, ``f_decay = 1.0``.

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
    for i, scheme in enumerate(schemes):
        r_av_sq = scheme.averaging_radius**2
        sums = squared_radii + r_av_sq
        prods = squared_radii * r_av_sq
        weights = np.exp(-scheme.f_decay * squared_distances / sums) * prods / sums
        results[i] = np.sum(weights * sigmas, axis=1) / np.sum(weights, axis=1)

    return results


def average_sigmas_by_molecule(
    coords: NDArray[np.float64],
    charges: NDArray[np.float64],
    areas: NDArray[np.float64],
    segment_offsets: NDArray[np.int64],
    schemes: Sequence[AveragingScheme],
    num_threads: int | None = None,
) -> NDArray[np.float64]:
    """Apply one or more averaging schemes to every segment of a dataset,
    in one threaded pass.

    Calls ``average_sigmas`` once per molecule -- each thread handles a
    disjoint range of whole molecules (see ``run_in_threads``), so this is
    safe with no locking. Output row ``i``
    (``average_sigmas_by_molecule(...)[i]``) matches ``schemes[i]``, and
    can be passed as ``sigmas`` straight into
    ``SigmaProfileTable.from_segments``, alongside the same ``areas``.

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
        arrays (see module docstring for the exactness requirement).
    schemes : Sequence[AveragingScheme]
        Schemes to average under, in the order the result rows are
        returned.
    num_threads : int | None, optional
        Number of threads to use, by default None, meaning every
        available CPU core.

    Returns
    -------
    np.ndarray, shape (len(schemes), n_segs_total)
        Averaged charge density for every segment under every scheme, in
        e/Å², row ``i`` matching ``schemes[i]``.
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
            averaged_sigmas[:, start_seg:stop_seg] = average_sigmas(
                coords[start_seg:stop_seg],
                charges[start_seg:stop_seg],
                areas[start_seg:stop_seg],
                schemes,
            )

    run_in_threads(process_range, num_mols, num_threads=num_threads, limit_blas=True)
    return averaged_sigmas
