"""DMol-3 COSMO file parser.

This module provides functions to parse COSMO output files from DMol-3.
"""

import re

import pandas as pd

from .utils import parse_table, parse_value

# Segment information pattern for DMol-3
SEGMENT_ROW_REGEX = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+"
    + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 3)  # x, y, z coordinates
    + r"\s+"
    + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 4)  # charge, area, charge/area, potential
    + r".*?$",
    re.MULTILINE,
)

SEGMENT_SECTION_REGEX = re.compile(
    r"Segment information:.*?n\s+atom\s+position.*?potential\s*\n+((?:"
    + SEGMENT_ROW_REGEX.pattern
    + r"(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

SEGMENT_INFO_SCHEMA = {
    "n": int,
    "atom": int,
    "x": float,
    "y": float,
    "z": float,
    "charge": float,
    "area": float,
    "charge_density": float,
    "potential": float,
}

# Atom information pattern for DMol-3 (.car format in the file)
ATOM_ROW_REGEX = re.compile(
    r"^([A-Z][A-Za-z0-9]*)\s+"  # atom id (e.g., N1, C1, H1)
    + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 3)  # x, y, z coordinates
    + r"\s+XXXX\s+\d+\s+\S+\s+([A-Z][a-z]?)\s+.*?$",  # element symbol
    re.MULTILINE,
)

ATOM_SECTION_REGEX = re.compile(
    r"Molecular car file\s*:.*?!DATE[^\n]*\n((?:"
    + ATOM_ROW_REGEX.pattern
    + r"(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

ATOM_INFO_SCHEMA = {"id": str, "x": float, "y": float, "z": float, "element": str}

# Volume pattern for DMol-3
VOLUME_REGEX = re.compile(r"Total volume of cavity \(A\*\*3\)\s*=\s*(\d+(?:\.\d+)?)")


def get_atom_dataframe(file_contents: str) -> pd.DataFrame:
    """Extract atom information from DMol-3 COSMO file.

    Parameters
    ----------
    file_contents : str
        Contents of the DMol-3 COSMO file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, x, y, z, element.
        Coordinates are in Angstroms (DMol-3 outputs are already in Angstroms).
    """
    return parse_table(
        file_contents,
        ATOM_ROW_REGEX,
        ATOM_SECTION_REGEX,
        ATOM_INFO_SCHEMA,
    )


def get_segment_dataframe(file_contents: str) -> pd.DataFrame:
    """Extract segment information from DMol-3 COSMO file.

    Parameters
    ----------
    file_contents : str
        Contents of the DMol-3 COSMO file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: n, atom, x, y, z, charge, area,
        charge_density, potential. Coordinates are in Angstroms
        (DMol-3 outputs are already in Angstroms).

    Notes
    -----
    The 'n' column (segment number) is dropped as it's redundant with the index.
    The 'charge_density' and 'potential' columns are also dropped to maintain
    compatibility with the TurboMole parser output.
    """
    df = parse_table(
        file_contents,
        SEGMENT_ROW_REGEX,
        SEGMENT_SECTION_REGEX,
        SEGMENT_INFO_SCHEMA,
    )
    # Drop columns to match TurboMole output format
    return df[["atom", "x", "y", "z", "charge", "area"]]


def get_volume(file_contents: str) -> float:
    """Extract molecular volume from DMol-3 COSMO file.

    Parameters
    ----------
    file_contents : str
        Contents of the DMol-3 COSMO file.

    Returns
    -------
    float
        Molecular cavity volume in cubic Angstroms.
        DMol-3 outputs are already in Angstroms^3, so no conversion is needed.
    """
    return parse_value(file_contents, VOLUME_REGEX)

