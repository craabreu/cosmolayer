"""Build and query on-disk segment-level COSMO datasets.

This subpackage is an offline dataset-building tool -- it depends on
``rdkit``, ``tqdm``, and a parquet engine, which the rest of
``cosmolayer`` does not need, so it is not imported eagerly by
``cosmolayer``'s top-level ``__init__``. Import it explicitly::

    from cosmolayer.store import SegmentStore

Build a store from ``.cosmo`` files with ``SegmentStore.from_cosmo_files``,
then derive per-atom or per-molecule sigma profiles with
``SegmentStore.compute_atom_sigma_profiles`` /
``compute_molecule_sigma_profiles``, both returning a ``SigmaProfileTable``.
"""

from .averaging import (
    AVERAGING_SCHEMES,
    COSMO_RS,
    COSMO_SAC_2002,
    COSMO_SAC_2010,
    AveragingScheme,
)
from .grid import SigmaGrid
from .profiles import SigmaProfileTable
from .segments import SegmentStore, StoreMetadata

__all__ = [
    "AVERAGING_SCHEMES",
    "COSMO_RS",
    "COSMO_SAC_2002",
    "COSMO_SAC_2010",
    "AveragingScheme",
    "SegmentStore",
    "SigmaGrid",
    "SigmaProfileTable",
    "StoreMetadata",
]
