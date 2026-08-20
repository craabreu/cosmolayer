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

FORMAT_NAME = "TURBOMOLE"

SEGMENT_SECTION_REGEX = re.compile(
    rf"^\$segment_information\b.*?\n((?:{SEGMENT_ROW_REGEX.pattern}(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

ATOM_SECTION_REGEX = re.compile(
    rf"\$coord_car\b.*?\n((?:{ATOM_ROW_REGEX.pattern}(?:\n|$))+)",
    re.MULTILINE | re.DOTALL,
)

VOLUME_REGEX = re.compile(r"volume=\s+(\d+(?:\.\d+)?)")


def is_turbomole_format(contents: str) -> bool:
    """Check if the contents are a Turbomole COSMO file.

    Parameters
    ----------
    contents : str
        Contents of a COSMO file.

    Returns
    -------
    bool
        True if the contents are a Turbomole COSMO file, False otherwise.
    """
    return "$segment_information" in contents and "$coord_car" in contents
