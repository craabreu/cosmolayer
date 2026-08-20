"""DMol-3 COSMO file parser.

This module provides functions to parse COSMO output files from DMol-3.
"""

import re

from .common import (  # noqa: F401
    ATOM_INFO_SCHEMA,
    ATOM_ROW_REGEX,
    SEGMENT_INFO_SCHEMA,
    SEGMENT_POSITION_CONVERSION_FACTOR,
    SEGMENT_ROW_REGEX,
)

FORMAT_NAME = "DMol-3"

SEGMENT_SECTION_REGEX = re.compile(
    r"Segment information:.*?n\s+atom\s+position.*?potential\s*\n+((?:"
    + SEGMENT_ROW_REGEX.pattern
    + r"(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

ATOM_SECTION_REGEX = re.compile(
    r"Molecular car file\s*:.*?!DATE[^\n]*\n((?:"
    + ATOM_ROW_REGEX.pattern
    + r"(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

VOLUME_REGEX = re.compile(r"Total volume of cavity \(A\*\*3\)\s*=\s*(\d+(?:\.\d+)?)")

ATOM_POSITION_CONVERSION_FACTOR = 1.0
VOLUME_CONVERSION_FACTOR = 1.0


def is_dmol3_format(contents: str) -> bool:
    """Check if the contents are a DMol-3 COSMO file.

    Parameters
    ----------
    contents : str
        Contents of a COSMO file.

    Returns
    -------
    bool
        True if the contents are a DMol-3 COSMO file, False otherwise.
    s"""
    return "DMol3/COSMO Results" in contents
