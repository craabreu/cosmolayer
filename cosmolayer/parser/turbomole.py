import re

from .common import (  # noqa: F401
    ATOM_INFO_SCHEMA,
    ATOM_POSITION_CONVERSION_FACTOR,
    ATOM_ROW_REGEX,
    SEGMENT_INFO_SCHEMA,
    SEGMENT_POSITION_CONVERSION_FACTOR,
    SEGMENT_ROW_REGEX,
    VOLUME_CONVERSION_FACTOR,
)

SEGMENT_SECTION_REGEX = re.compile(
    rf"^\$segment_information\b.*?\n((?:{SEGMENT_ROW_REGEX.pattern}(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

ATOM_SECTION_REGEX = re.compile(
    rf"\$coord_car\b.*?\n((?:{ATOM_ROW_REGEX.pattern}(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

VOLUME_REGEX = re.compile(r"volume=\s+(\d+(?:\.\d+)?)")
