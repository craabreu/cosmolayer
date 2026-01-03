"""
.. module:: cosmolayer.parser
   :synopsis: Parser for COSMO output files.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import os
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
        df[axis] *= module.ATOM_CONVERSION_FACTOR
    return df


def get_segment_dataframe(module: ModuleType, file_contents: str) -> pd.DataFrame:
    df = parse_table(
        file_contents,
        module.SEGMENT_ROW_REGEX,
        module.SEGMENT_SECTION_REGEX,
        module.SEGMENT_INFO_SCHEMA,
    )
    for axis in "xyz":
        df[axis] *= module.SEGMENT_CONVERSION_FACTOR
    return df


def get_volume(module: ModuleType, file_contents: str) -> float:
    return float(
        parse_value(file_contents, module.VOLUME_REGEX)
        * module.VOLUME_CONVERSION_FACTOR
    )


def parse_cosmo_file(
    path: str | os.PathLike[str],
) -> tuple[str, pd.DataFrame, pd.DataFrame, float]:
    """Parse a COSMO output file.

    This function reads a COSMO (Conductor-like Screening Model) output file
    and extracts atomic coordinates, segment information, and molecular volume.
    It automatically detects the file format (TURBOMOLE or DMol-3) and uses
    the appropriate parser.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the COSMO output file to parse.

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
    >>> fmt, atoms, segments, volume = parse_cosmo_file(path)
    >>> print(fmt)
    TURBOMOLE
    >>> atoms.tail(3)
       id         x         y         z element
    6  H3  0.338091 -0.995118 -0.082975       H
    7  H4  0.862400 -0.435830  0.356083       H
    8  H5  0.624747  0.700154 -0.227811       H
    >>> segments.tail(3)
         atom         x         y         z    charge      area
    468     9  1.003395  2.214518 -1.389667 -0.002498  0.193147
    469     9  1.068201  0.923523 -1.695803 -0.002131  0.130985
    470     9  2.133636  1.152865  0.489697 -0.001817  0.145681
    >>> volume
    80.07160...

    Parse a DMol-3 COSMO file:

    >>> path = files("cosmolayer.data") / "NCCO.cosmo"
    >>> fmt, atoms, segments, volume = parse_cosmo_file(path)
    >>> print(fmt)
    DMol-3
    >>> len(atoms)
    11
    >>> len(segments)
    429
    >>> volume
    86.10187...
    """
    with open(path, encoding="utf-8", errors="replace") as file:
        contents = file.read()

    module: ModuleType
    if "DMol3/COSMO Results" in contents:
        format = "DMol-3"
        module = dmol3
    elif "$segment_information" in contents and "$coord_car" in contents:
        format = "TURBOMOLE"
        module = turbomole
    else:
        raise ValueError(
            "Could not parse COSMO file. Supported formats: TURBOMOLE, DMol-3"
        )
    return (
        format,
        get_atom_dataframe(module, contents),
        get_segment_dataframe(module, contents),
        get_volume(module, contents),
    )
