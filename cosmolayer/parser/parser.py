"""
.. module:: cosmolayer.parser
   :synopsis: Parser for COSMO output files.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from types import ModuleType

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from . import chaos, dmol3, turbomole
from .utils import parse_table, parse_value


def get_atom_dataframe(module: ModuleType, file_contents: str) -> pd.DataFrame:
    df = parse_table(
        file_contents,
        module.ATOM_ROW_REGEX,
        module.ATOM_SECTION_REGEX,
        module.ATOM_INFO_SCHEMA,
    )
    for axis in "xyz":
        df[axis] *= module.ATOM_POSITION_CONVERSION_FACTOR
    return df


def get_segment_dataframe(module: ModuleType, file_contents: str) -> pd.DataFrame:
    df = parse_table(
        file_contents,
        module.SEGMENT_ROW_REGEX,
        module.SEGMENT_SECTION_REGEX,
        module.SEGMENT_INFO_SCHEMA,
    )
    for axis in "xyz":
        df[axis] *= module.SEGMENT_POSITION_CONVERSION_FACTOR
    df["atom"] -= 1
    return df


def get_volume(module: ModuleType, file_contents: str) -> float:
    return float(
        parse_value(file_contents, module.VOLUME_REGEX)
        * module.VOLUME_CONVERSION_FACTOR
    )


def get_atom_mapped_smiles(
    atoms_df: pd.DataFrame,
    charge: int = 0,
    print_errors: bool = True,
) -> str | None:
    """Derive an atom-mapped SMILES string from a molecule's 3D geometry.

    Bonds and stereochemistry are perceived from the atomic coordinates using
    RDKit. This is a best-effort chemistry step: it can fail for legitimate
    reasons (e.g. a geometry from which no valid bond order can be derived),
    in which case ``None`` is returned rather than raising.

    Parameters
    ----------
    atoms_df : pd.DataFrame
        DataFrame with ``element``, ``x``, ``y``, and ``z`` columns, one row
        per atom, in the order the resulting SMILES should be atom-mapped.
    charge : int, default=0
        Net molecular charge to assume when perceiving bond orders.
    print_errors : bool, default=True
        Whether to print a message to stdout when bond/stereo perception
        fails.

    Returns
    -------
    str | None
        The atom-mapped, canonical, isomeric SMILES string, or ``None`` if
        bonds or stereochemistry could not be determined from the geometry.
    """
    xyz_block = (
        f"{len(atoms_df)}"
        f"\n\n"
        f"{atoms_df[['element', 'x', 'y', 'z']].to_string(index=False, header=False)}"
    )
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        if print_errors:
            print("Error parsing molecule: could not build a molecule from geometry")
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.AssignStereochemistryFrom3D(mol)
    except (ValueError, RuntimeError, Chem.rdchem.MolSanitizeException) as e:
        if print_errors:
            print(f"Error parsing molecule: {e}")
        return None
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    return Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True, allHsExplicit=True
    )


def parse_cosmo_file(
    contents: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame, float]:
    """Parse the contents of a COSMO output file.

    This function reads the contents of a COSMO (Conductor-like Screening Model) output
    file and extracts atomic coordinates, segment information, and molecular volume.
    It automatically detects the file format (TURBOMOLE, DMol-3, or CHAOS) and uses
    the appropriate parser.

    Parameters
    ----------
    contents : str
        Contents of the COSMO output file to parse.

    Returns
    -------
    format : str
        The file format detected ("DMol-3", "TURBOMOLE", or "CHAOS").
    atom_df : pd.DataFrame
        DataFrame containing atomic information with columns:
        - id: atom identifier (str)
        - x, y, z: Cartesian coordinates in Angstroms (float)
        - element: chemical element symbol (str)
    segment_df : pd.DataFrame
        DataFrame containing segment information with columns:
        - atom: associated atom number (int)
        - x, y, z: segment coordinates in Angstroms (float)
        - charge: segment charge (float)
        - area: segment surface area (float)
    volume : float
        Molecular cavity volume in cubic Angstroms.

    Raises
    ------
    ValueError
        If the file format is not recognized or does not contain the required
        COSMO sections.
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    Parse a TURBOMOLE COSMO file:

    >>> from importlib.resources import files
    >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
    >>> contents = path.read_text(encoding="utf-8", errors="replace")
    >>> fmt, atoms, segments, volume = parse_cosmo_file(contents)
    >>> print(fmt)
    TURBOMOLE
    >>> atoms.tail(3)
       id       x       y       z element
    6  H3  0.6389 -1.8805 -0.1568       H
    7  H4  1.6297 -0.8236  0.6729       H
    8  H5  1.1806  1.3231 -0.4305       H
    >>> segments.tail(3)
         atom         x         y         z    charge      area
    468     8  1.003395  2.214518 -1.389667 -0.002498  0.193147
    469     8  1.068201  0.923523 -1.695803 -0.002131  0.130985
    470     8  2.133636  1.152865  0.489697 -0.001817  0.145681
    >>> volume
    80.07160...

    Parse a DMol-3 COSMO file:

    >>> path = files("cosmolayer.data") / "NCCO.cosmo"
    >>> contents = path.read_text(encoding="utf-8", errors="replace")
    >>> fmt, atoms, segments, volume = parse_cosmo_file(contents)
    >>> print(fmt)
    DMol-3
    >>> len(atoms)
    11
    >>> len(segments)
    429
    >>> volume
    86.10187...

    Parse a CHAOS dataset JSON record:

    >>> path = files("cosmolayer.data") / "chaos_sample.json"
    >>> contents = path.read_text(encoding="utf-8", errors="replace")
    >>> fmt, atoms, segments, volume = parse_cosmo_file(contents)
    >>> print(fmt)
    CHAOS
    >>> len(atoms)
    15
    >>> len(segments)
    1226
    >>> volume
    135.8705...
    """
    if dmol3.is_dmol3_format(contents):
        format = dmol3.FORMAT_NAME
        atoms_df = get_atom_dataframe(dmol3, contents)
        segments_df = get_segment_dataframe(dmol3, contents)
        volume = get_volume(dmol3, contents)
    elif turbomole.is_turbomole_format(contents):
        format = turbomole.FORMAT_NAME
        atoms_df = get_atom_dataframe(turbomole, contents)
        segments_df = get_segment_dataframe(turbomole, contents)
        volume = get_volume(turbomole, contents)
    elif (data := chaos.parse_record(contents)) is not None:
        format = chaos.FORMAT_NAME
        atoms_df = chaos.get_atom_dataframe(data)
        segments_df = chaos.get_segment_dataframe(data)
        volume = chaos.get_volume(data)
    else:
        raise ValueError(
            "Could not parse COSMO file contents. Supported formats: "
            f"{turbomole.FORMAT_NAME}, {dmol3.FORMAT_NAME}, {chaos.FORMAT_NAME}"
        )
    return format, atoms_df, segments_df, volume
