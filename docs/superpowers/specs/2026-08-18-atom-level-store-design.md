# Atom-level data in SegmentStore

Date: 2026-08-18

## Problem

`SegmentStore` parses each COSMO file's atom table (`ATOM_INFO_SCHEMA`:
`id: str`, `x/y/z: float`, `element: str`) via `_parse_molecule`, but keeps
only its row count (`num_atoms`) — the atom ids, coordinates, and elements
themselves are discarded once `from_cosmo_files` finishes with them. There
is currently no way to recover per-atom identity or geometry from a saved
store.

## Goals

- Persist per-atom `id`, `element`, and `x`/`y`/`z` coordinates as a
  first-class part of `SegmentStore`, addressable by the same *global atom
  index* that `atom_indices.npy` already uses to tie segments to atoms.
- Keep it consistent through every existing store operation that touches
  atom identity: `save`/`load`, `subsample` (via `restrict_to_molecules`),
  and `coarse_grain`.

## Non-goals

- No repositioning of coarse-grained pseudo-atoms (e.g. centroid
  averaging). A merged hydrogen's atom row is simply dropped; the
  surviving heavy atom's original `id`/`element`/coordinates are kept
  as-is. This mirrors today's `coarse_grain`, which already leaves
  segment positions (`data.npy`) untouched.

## Data model

A new `atoms_df` (`pandas.DataFrame`) becomes a required `SegmentStore`
field, alongside `data`, `atom_indices`, and `molecules_df`. Columns:

| column    | dtype    | source                          |
|-----------|----------|----------------------------------|
| `id`      | `str`    | `atom_df["id"]` (COSMO atom label, e.g. `"C1"`) |
| `element` | `str`    | `atom_df["element"]`            |
| `x`       | `float32`| `atom_df["x"]`                  |
| `y`       | `float32`| `atom_df["y"]`                  |
| `z`       | `float32`| `atom_df["z"]`                  |

Row order is the global atom index: row `atom_offsets[m] + j` is molecule
`m`'s local atom `j` (0-based) — the same index space `atom_indices.npy`
values already point into, and the same order `_parse_molecule`'s
`atom_df` is already in (COSMO file order == local atom index order,
per `_reorder_molecule`'s existing guarantee). `atoms_df` must have
exactly `molecules_df["num_atoms"].sum()` rows, one table for the whole
store (mirroring `molecules_df`, not a per-molecule split).

This follows the store's existing split by dtype homogeneity: per-segment
*numeric* data lives in `.npy` (`data.npy`); per-molecule *mixed-type*
data lives in `.parquet` (`molecules.parquet`). Atom data is mixed-type
(`id`/`element` are strings), so it follows the same convention as
`molecules_df`.

## On-disk format

New file: `<storage_dir>/atoms.parquet`, columns as above. Added to
`_STORE_FILES` alongside `MOLECULES_FILE`. Module docstring's file-format
table gets a new row:

```
<storage_dir>/atoms.parquet     columns: id, element, x, y, z (one row per atom,
                                 global atom index order)
```

## Component changes

- **`SegmentStore.__init__`**: gains an `atoms_df: pd.DataFrame` parameter,
  stored as `self.atoms_df`, positioned next to `molecules_df` in the
  signature and in every call site that constructs a `SegmentStore`.
- **`from_cosmo_files`**: for each successfully parsed molecule, append its
  `atom_df` (already returned by `_parse_molecule`) to an `atoms_chunks`
  list; after the loop, `pd.concat(atoms_chunks, ignore_index=True)` to
  build `self.atoms_df` in the same append order as `atom_offsets` are
  assigned (already global-atom-index order).
- **`save`**: writes `self.atoms_df` to `atoms.parquet` (same pattern as
  `molecules_df` → `molecules.parquet`).
- **`load`**: reads `atoms.parquet` into `atoms_df`, passed to the
  `SegmentStore` constructor.
- **`restrict_to_molecules`** (`subsampling.py`): given the existing
  per-molecule `selected` row indices, derive an atom-level boolean mask
  the same way the segment mask is derived from `segment_molecule` today
  — build an `atom_molecule` array (via `np.repeat` over
  `full_num_atoms`), mask by `is_selected[atom_molecule]`, and slice
  `atoms_df` by that mask (row order preserved, so it stays aligned with
  the rebased `atom_indices`).
- **`coarse_grain`**: `compute_atom_remap` already identifies, per
  molecule, which local atom indices survive vs. get merged into a
  neighbor. Build a per-molecule boolean "survives" mask (an atom index is
  kept iff it's a key that maps to itself, i.e. not merged away), assemble
  the global-index survivor mask across molecules in atom-offset order,
  and slice `atoms_df` by it — same drop-only, no-reposition, no-reorder
  approach already used for segments in this method.

## Error handling

`_parse_molecule` already raises `ValueError` when `atom_df`'s length
disagrees with `mol.GetNumAtoms()`. This work adds one more check in the
same place, once atom-map numbers are stamped: for each atom in `mol`
(now ordered by ascending `AtomMapNum`, i.e. COSMO/local-atom-index
order), its RDKit element symbol (`GetSymbol()`) must equal
`atom_df["element"]` at that same local index. This catches a SMILES/COSMO
mismatch that count-checking alone would miss — e.g. two atoms transposed,
or an element substituted — by confirming the reordered molecule and the
atom table agree atom-for-atom, not just in aggregate. Mismatch raises
`ValueError` naming the molecule, the mismatched local index, and both
elements.

## Testing

Extend `cosmolayer/tests/test_store.py`:

- `from_cosmo_files` / `save` / `load` round-trip: assert `atoms_df` has
  the right columns, row count (`== molecules_df["num_atoms"].sum()`), and
  that its `id`/`element`/coordinates for a known fixture molecule match
  the source `.cosmo` file's atom table.
- `subsample`: assert the restricted store's `atoms_df` contains exactly
  the atoms belonging to the kept molecules, correctly reindexed.
- `coarse_grain`: assert merged hydrogens' rows are absent from the
  result's `atoms_df` and survivors' `id`/`element`/coordinates are
  unchanged from the pre-coarse-grained store.
- `_parse_molecule`: a fixture where the SMILES's element sequence
  (by atom-map/local-index order) disagrees with the COSMO file's
  `atom_df["element"]` at some index raises `ValueError`.
