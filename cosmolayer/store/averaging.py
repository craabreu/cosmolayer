"""COSMO-SAC-style distance-weighted averaging of segment charge density.

In COSMO-RS and COSMO-SAC, "sigma" is this averaged charge density, not
the raw per-segment ``charge / area``.

``segment_offsets`` must describe *exactly* the molecules present in the
accompanying segment-level arrays: the last molecule's segments run to
the end of those arrays. Slice ``segment_offsets`` and the arrays
together, or leftover segments are attributed to the wrong molecule.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from cosmolayer.cosmosac.constants import (
    COSMO_SAC_2002_AVERAGING_RADIUS,
    COSMO_SAC_2002_F_DECAY,
    COSMO_SAC_2010_AVERAGING_RADIUS,
    COSMO_SAC_2010_F_DECAY,
)

from .parallel import run_in_threads


@dataclass(frozen=True)
class AveragingScheme:
    """Named averaging scheme (radius and decay) for segment charge
    densities.

    Parameters
    ----------
    name : str
        Identifier used as the ``<name>.npy`` stem when a store saves
        averaged sigmas. Must not collide with reserved store filenames
        (``data``, ``atom_indices``).
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
"""Built-in schemes: COSMO-RS, COSMO-SAC 2002, and COSMO-SAC 2010."""


def average_sigmas(
    coords: NDArray[np.float64],
    charges: NDArray[np.float64],
    areas: NDArray[np.float64],
    schemes: Sequence[AveragingScheme],
) -> NDArray[np.float64]:
    """Distance-weighted average of one molecule's segment charge
    densities, under one or more averaging schemes.

    For every segment ``m``, replaces its raw charge density
    ``sigma_m = q_m / A_m`` with a weighted average over every segment
    ``n`` in the same molecule, including itself::

        sigma_avg[m] = sum_n(sigma[n] * w[m, n]) / sum_n(w[m, n])
        w[m, n] = (r_n^2 * r_av^2 / (r_n^2 + r_av^2))
                  * exp(-f_decay * d_mn^2 / (r_n^2 + r_av^2))

    where ``r_n = sqrt(A_n / pi)`` is the *neighbor* segment's effective
    radius and ``d_mn`` is the distance between centroids, so ``w`` is
    asymmetric even though ``d_mn`` is not. Pass segments of a single
    molecule only; use ``average_sigmas_by_molecule`` for a dataset.

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
        Schemes to apply, in the order of the result rows.

    Returns
    -------
    np.ndarray, shape (len(schemes), n_segs)
        Averaged charge density for each segment under each scheme, in
        e/Å², row ``i`` matching ``schemes[i]``.
    """
    # float64: the Gram-matrix distance expansion loses precision in float32.
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


def average_sigmas_by_molecule(  # noqa: PLR0913
    coords: NDArray[np.float64],
    charges: NDArray[np.float64],
    areas: NDArray[np.float64],
    segment_offsets: NDArray[np.int64],
    schemes: Sequence[AveragingScheme],
    num_threads: int | None = None,
    *,
    progress: bool = False,
) -> NDArray[np.float64]:
    """Apply one or more averaging schemes to every molecule in a dataset.

    Each thread handles a disjoint range of whole molecules. Row ``i`` of
    the result matches ``schemes[i]`` and can be passed as ``sigmas`` to
    ``SigmaProfileTable.from_segments``.

    Parameters
    ----------
    coords : np.ndarray
        Segment centroid coordinates, shape ``(n_segs_total, 3)``.
    charges : np.ndarray
        Segment charges, shape ``(n_segs_total,)``.
    areas : np.ndarray
        Segment areas, shape ``(n_segs_total,)``.
    segment_offsets : np.ndarray
        Start index of each molecule's segments. Must describe exactly
        the molecules present in the segment-level arrays.
    schemes : Sequence[AveragingScheme]
        Schemes to apply, in the order of the result rows.
    num_threads : int | None, optional
        Thread count. ``None`` (default) uses every CPU core.
    progress : bool, optional
        If True, show a tqdm bar over molecules. Default False.

    Returns
    -------
    np.ndarray, shape (len(schemes), n_segs_total)
        Averaged charge density for every segment under every scheme, in
        e/Å², row ``i`` matching ``schemes[i]``.
    """
    num_segs = len(charges)
    num_mols = len(segment_offsets)
    averaged_sigmas = np.empty((len(schemes), num_segs), dtype=np.float64)

    with tqdm(
        total=num_mols, desc="Averaging sigmas", disable=not progress
    ) as progress_bar:

        def process_range(start_mol: int, stop_mol: int) -> None:
            for mol in range(start_mol, stop_mol):
                start_seg = segment_offsets[mol]
                stop_seg = segment_offsets[mol + 1] if mol + 1 < num_mols else num_segs
                if stop_seg != start_seg:
                    averaged_sigmas[:, start_seg:stop_seg] = average_sigmas(
                        coords[start_seg:stop_seg],
                        charges[start_seg:stop_seg],
                        areas[start_seg:stop_seg],
                        schemes,
                    )
                progress_bar.update(1)

        run_in_threads(
            process_range, num_mols, num_threads=num_threads, limit_blas=True
        )
    return averaged_sigmas
