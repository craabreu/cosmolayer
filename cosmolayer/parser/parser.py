"""
.. module:: cosmolayer.parser
   :synopsis: Parser for COSMO output files.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from types import ModuleType

import pandas as pd

from . import dmol3, turbomole
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


def parse_cosmo_file(
    contents: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame, float]:
    """Parse the contents of a COSMO output file.

    This function reads the contents of a COSMO (Conductor-like Screening Model) output
    file and extracts atomic coordinates, segment information, and molecular volume.
    It automatically detects the file format (TURBOMOLE or DMol-3) and uses the
    appropriate parser.

    Parameters
    ----------
    contents : str
        Contents of the COSMO output file to parse.

    Returns
    -------
    format : str
        The file format detected ("DMol-3" or "TURBOMOLE").
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
    """
    module: ModuleType
    if "DMol3/COSMO Results" in contents:
        format = "DMol-3"
        module = dmol3
    elif "$segment_information" in contents and "$coord_car" in contents:
        format = "TURBOMOLE"
        module = turbomole
    else:
        raise ValueError(
            "Could not parse COSMO file contents. Supported formats: TURBOMOLE, DMol-3"
        )
    return (
        format,
        get_atom_dataframe(module, contents),
        get_segment_dataframe(module, contents),
        get_volume(module, contents),
    )
