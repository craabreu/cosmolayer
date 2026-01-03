import re

SEGMENT_ROW_REGEX = re.compile(
    r"^\s*\d+\s+(\d+)\s+"  # n (not used), atom
    + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 5)  # x, y, z, charge, area
    + r"\s+"
    + r"\s+".join([r"[+-]?\d+(?:\.\d+)?"] * 2)  # charge/area, potential (not used)
    + r".*?$",
    re.MULTILINE,
)

ATOM_ROW_REGEX = re.compile(
    r"^([A-Za-z0-9]+)\s+"  # atom id (e.g., N1, C1, H1)
    + r"\s+".join([r"([+-]?\d+(?:\.\d+)?)"] * 3)  # x, y, z coordinates
    + r"(?:\s+\S+){3}"
    + r"\s*([A-Z][a-z]?)"  # element symbol
    + r"\s*(?:\S+)?",
    re.MULTILINE,
)

SEGMENT_INFO_SCHEMA = {
    "atom": int,
    "x": float,
    "y": float,
    "z": float,
    "charge": float,
    "area": float,
}

ATOM_INFO_SCHEMA = {"id": str, "x": float, "y": float, "z": float, "element": str}

BOHR_TO_ANGSTROM = 0.52917721067
ATOM_CONVERSION_FACTOR = BOHR_TO_ANGSTROM
SEGMENT_CONVERSION_FACTOR = BOHR_TO_ANGSTROM
VOLUME_CONVERSION_FACTOR = BOHR_TO_ANGSTROM**3
