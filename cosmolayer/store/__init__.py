"""Build and query on-disk segment-level COSMO datasets.

An offline dataset-building tool: parse ``.cosmo`` files into an on-disk
segment-data store, then derive per-atom or per-molecule sigma profiles
from it::

    from cosmolayer.store import SegmentStore

    store = SegmentStore.from_cosmo_files(
        cosmo_files_dir, smiles_to_filename, storage_dir
    )
    profiles = store.compute_molecule_sigma_profiles()

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
