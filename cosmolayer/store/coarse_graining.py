"""Merge hydrogens into their heavy-atom neighbor, shrinking a SegmentStore
to a united-atom store.

``SegmentStore.coarse_grain`` builds a new store where every hydrogen
``Chem.RemoveHs`` would actually remove is merged into the heavy atom it's
bonded to: fewer atoms, same segments. Requires every molecule's stored
``smiles`` to carry atom-map numbers reflecting local (COSMO) atom index
(guaranteed for any store built after GH issue #43's fix) -- that's what
lets a removed hydrogen's segments be redirected to the right heavy-atom
neighbor.
"""

from rdkit import Chem


def compute_atom_remap(
    mapped_smiles: str,
) -> tuple[dict[int, int], frozenset[int], str]:
    """Compute one molecule's old-to-new local atom index map and its
    coarse-grained, re-mapped SMILES.

    Parses ``mapped_smiles`` with hydrogens kept, then calls
    ``Chem.RemoveHs`` -- which returns a *new* ``Mol``, not an in-place
    edit -- to find out which hydrogens RDKit's own rules actually remove.
    ``Chem.RemoveHs`` never reorders surviving atoms relative to each
    other, only deletes, so a surviving atom's position in the reduced
    molecule (``0, 1, 2, ...``) is exactly its rank among survivors in the
    original local-index order -- the new, compacted local index. A
    removed hydrogen's segments belong wherever its (single) heavy-atom
    neighbor ended up.

    A hydrogen ``Chem.RemoveHs`` conservatively keeps (a non-default
    isotope, or a neighbor with non-tetrahedral stereochemistry it can't
    safely represent without the explicit atom) is not merged: it survives
    as its own atom in the coarse-grained molecule, with its own new
    index, same as any heavy atom.

    Parameters
    ----------
    mapped_smiles : str
        Atom-mapped SMILES for one molecule, with map numbers a clean
        0-based or 1-based permutation of local atom index (as
        ``molecules_df["smiles"]`` stores it -- see GH issue #43).

    Returns
    -------
    new_local_index : dict[int, int]
        Every original local atom index mapped to its new, compacted
        local index -- both surviving atoms (mapped to their own new
        index) and merged hydrogens (mapped to their heavy-atom
        neighbor's new index) have an entry.
    survivors : frozenset[int]
        Original local atom indices that are still their own atom in the
        coarse-grained molecule (as opposed to a merged hydrogen) -- one
        per new, compacted local index.
    new_mapped_smiles : str
        The coarse-grained molecule's SMILES, atom-mapped with contiguous
        map numbers starting at the same base (0 or 1) as the input, in
        the same relative order as the surviving atoms originally had.

    Raises
    ------
    ValueError
        If ``mapped_smiles`` can't be parsed, or its atoms aren't a clean
        0-based or 1-based permutation of local atom index (including the
        unmapped case, where every atom's map number is 0).
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(mapped_smiles, params)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {mapped_smiles!r}")

    num_atoms = mol.GetNumAtoms()
    map_nums = {atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    if map_nums == set(range(num_atoms)):
        base = 0
    elif map_nums == set(range(1, num_atoms + 1)):
        base = 1
    else:
        raise ValueError(
            f"SMILES {mapped_smiles!r} has no usable atom-map numbers -- "
            "expected a 0-based or 1-based permutation of its atom count "
            f"({num_atoms}), got map numbers {sorted(map_nums)}."
        )

    reduced = Chem.RemoveHs(mol)

    new_local_index: dict[int, int] = {}
    survivors: set[int] = set()
    for new_idx, atom in enumerate(reduced.GetAtoms()):
        old_local = atom.GetAtomMapNum() - base
        new_local_index[old_local] = new_idx
        survivors.add(old_local)

    surviving_map_nums = {a.GetAtomMapNum() for a in reduced.GetAtoms()}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetAtomMapNum() in surviving_map_nums:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError(
                f"expected exactly one neighbor for a hydrogen RemoveHs "
                f"actually removed, got {len(neighbors)} (atom map num "
                f"{atom.GetAtomMapNum()}, smiles {mapped_smiles!r})"
            )
        old_local = atom.GetAtomMapNum() - base
        neighbor_old_local = neighbors[0].GetAtomMapNum() - base
        new_local_index[old_local] = new_local_index[neighbor_old_local]

    for atom in reduced.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + base)
    new_mapped_smiles = Chem.MolToSmiles(reduced)

    return new_local_index, frozenset(survivors), new_mapped_smiles
