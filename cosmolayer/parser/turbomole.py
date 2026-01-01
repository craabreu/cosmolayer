import re

import pandas as pd

from .utils import parse_table, parse_value

SEGMENT_ROW_REGEX = re.compile(
    r"^\s*\d+\s+(\d+)\s+" + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 5) + r".*?$",
    re.MULTILINE,
)
SEGMENT_SECTION_REGEX = re.compile(
    rf"^\$segment_information\b.*?\n((?:{SEGMENT_ROW_REGEX.pattern}(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)
SEGMENT_INFO_SCHEMA = {
    "atom": int,
    "x": float,
    "y": float,
    "z": float,
    "charge": float,
    "area": float,
}

ATOM_ROW_REGEX = re.compile(
    r"^([A-Za-z0-9]+)\s+"
    + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 3)
    + r"(?:\s+\S+){3}\s*([A-Z][a-z]?)\s*(?:\S+)?",
    re.MULTILINE,
)
ATOM_SECTION_REGEX = re.compile(
    rf"\$coord_car\b.*?\n((?:{ATOM_ROW_REGEX.pattern}(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)
ATOM_INFO_SCHEMA = {"id": str, "x": float, "y": float, "z": float, "element": str}

VOLUME_REGEX = re.compile(r"volume=\s+(\d+(?:\.\d+)?)")
BOHR_TO_ANGSTROM = 0.52917721067


def get_atom_dataframe(file_contents: str) -> pd.DataFrame:
    df = parse_table(
        file_contents,
        ATOM_ROW_REGEX,
        ATOM_SECTION_REGEX,
        ATOM_INFO_SCHEMA,
    )
    for axis in "xyz":
        df[axis] *= BOHR_TO_ANGSTROM
    return df


def get_segment_dataframe(file_contents: str) -> pd.DataFrame:
    df = parse_table(
        file_contents,
        SEGMENT_ROW_REGEX,
        SEGMENT_SECTION_REGEX,
        SEGMENT_INFO_SCHEMA,
    )
    for axis in "xyz":
        df[axis] *= BOHR_TO_ANGSTROM
    return df


def get_volume(file_contents: str) -> float:
    return parse_value(file_contents, VOLUME_REGEX) * BOHR_TO_ANGSTROM**3
