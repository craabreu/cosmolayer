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
TOTAL_AREA_REGEX = re.compile(
    r"Total surface area of cavity \(A\*\*2\)\s*=\s*(\d+(?:\.\d+)?)"
)
NUM_SEGMENTS_REGEX = re.compile(r"total number of segments:\s*(\d+)")

ATOM_POSITION_CONVERSION_FACTOR = 1.0
VOLUME_CONVERSION_FACTOR = 1.0
AREA_CONVERSION_FACTOR = 1.0
