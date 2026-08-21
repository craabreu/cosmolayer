"""CHAOS dataset JSON parser.

This module parses per-molecule JSON records from the CHAOS dataset
(arXiv:2511.19002) into the same shape produced by the DMol-3 and TURBOMOLE
text parsers, so they can be consumed interchangeably by
:func:`cosmolayer.parser.parser.parse_cosmo_file`.

Unlike the DMol-3/TURBOMOLE modules, CHAOS records are JSON rather than
fixed-column text, so this module reads them with :func:`json.loads` and
dict/list indexing instead of the regex-based machinery in
:mod:`cosmolayer.parser.common`.

Two CHAOS-specific unit quirks matter here and are handled below:

- ``solvation.SegmentList`` segment positions are reported in Bohr (atomic
  units), unlike ``structural.Coordinates``, which is already in Å.
- ``solvation.CavVolume`` (and ``solvation.CavArea``, unused here) are also
  reported in atomic units (Bohr\\ :sup:`3` and Bohr\\ :sup:`2`
  respectively), even though the per-atom/per-segment ``area``/``charge``
  fields elsewhere in the ``solvation`` block are already in Å²/e.
"""

import json

import pandas as pd

from .common import BOHR_TO_ANGSTROM

FORMAT_NAME = "CHAOS"

REQUIRED_TOP_LEVEL_KEYS = frozenset({"general", "structural", "solvation"})

ATOM_POSITION_CONVERSION_FACTOR = 1.0
SEGMENT_POSITION_CONVERSION_FACTOR = BOHR_TO_ANGSTROM
VOLUME_CONVERSION_FACTOR = BOHR_TO_ANGSTROM**3


def _has_required_fields(data: object) -> bool:
    """Check that a parsed JSON value has the shape a CHAOS record needs.

    Validates both the top-level keys (``general``, ``structural``,
    ``solvation``) and the nested fields (``general.AtomList``,
    ``structural.Coordinates``, ``solvation.SegmentList``,
    ``solvation.CavVolume``) that :func:`get_atom_dataframe`,
    :func:`get_segment_dataframe`, and :func:`get_volume` require. A value
    missing any of these is not recognized as CHAOS JSON, so callers fall
    through to raising ``ValueError`` instead of a raw
    ``KeyError``/``TypeError``.
    """
    if not (isinstance(data, dict) and REQUIRED_TOP_LEVEL_KEYS <= data.keys()):
        return False
    try:
        data["general"]["AtomList"]
        data["structural"]["Coordinates"]
        data["solvation"]["SegmentList"]
        data["solvation"]["CavVolume"]
    except (KeyError, TypeError):
        return False
    return True


def is_chaos_json(contents: str) -> bool:
    """Detect whether ``contents`` is a CHAOS dataset JSON record.

    Parameters
    ----------
    contents : str
        Candidate file contents.

    Returns
    -------
    bool
        True if ``contents`` parses as JSON and has the top-level and
        nested fields a CHAOS record needs (see :func:`_has_required_fields`).
    """
    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        return False
    return _has_required_fields(data)


def parse_record(contents: str) -> dict | None:
    """Parse ``contents`` as a CHAOS JSON record, if it looks like one.

    Parses ``contents`` exactly once and validates its shape in the same
    pass, so callers that need both the format-detection answer and the
    parsed record (e.g. :func:`cosmolayer.parser.parser.parse_cosmo_file`)
    avoid re-parsing the same JSON text once per field they read.

    Parameters
    ----------
    contents : str
        Candidate file contents.

    Returns
    -------
    dict or None
        The parsed record if ``contents`` is valid CHAOS JSON (see
        :func:`_has_required_fields`), else ``None``.
    """
    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        return None
    return data if _has_required_fields(data) else None


def get_atom_dataframe(data: dict) -> pd.DataFrame:
    """Parse per-atom data from a CHAOS JSON record.

    Combines ``general.AtomList`` (element symbols) with
    ``structural.Coordinates`` (Cartesian coordinates, already in Å) in
    ``general.AtomList`` order, which is the atom-numbering convention used
    throughout the rest of the record (including
    ``solvation.SegmentList``'s parent-atom index).

    Parameters
    ----------
    data : dict
        Contents of a CHAOS JSON file.

    Returns
    -------
    pd.DataFrame
        Columns: ``id`` (synthesized as ``f"{element}{index}"``), ``x``,
        ``y``, ``z`` (Å), ``element``.

    Raises
    ------
    ValueError
        If ``structural.Coordinates`` is JSON ``null``, or any per-atom
        entry (or component) is ``null``.
    """
    atom_list = data["general"]["AtomList"]
    coordinates = data["structural"]["Coordinates"]
    if coordinates is None or any(
        xyz is None or any(c is None for c in xyz) for xyz in coordinates
    ):
        raise ValueError(
            "CHAOS record has null structural.Coordinates; cannot build an atom table."
        )
    rows = [
        {
            "id": f"{atom['element']}{atom['index']}",
            "x": xyz[0] * ATOM_POSITION_CONVERSION_FACTOR,
            "y": xyz[1] * ATOM_POSITION_CONVERSION_FACTOR,
            "z": xyz[2] * ATOM_POSITION_CONVERSION_FACTOR,
            "element": atom["element"],
        }
        for atom, xyz in zip(atom_list, coordinates, strict=True)
    ]
    return pd.DataFrame(rows, columns=["id", "x", "y", "z", "element"])


def get_segment_dataframe(data: dict) -> pd.DataFrame:
    """Parse per-segment cavity data from a CHAOS JSON record.

    Each entry of ``solvation.SegmentList`` is
    ``[segment_index, parent_atom_index, x, y, z, charge, area, sigma,
    potential]``, both indices 1-based. ``x, y, z`` are in Bohr and are
    converted to Å here; ``charge`` (e) and ``area`` (Å²) need no
    conversion. ``sigma`` and ``potential`` are dropped, matching the
    columns produced by the DMol-3/TURBOMOLE parsers.

    Parameters
    ----------
    data : dict
        Contents of a CHAOS JSON file.

    Returns
    -------
    pd.DataFrame
        Columns: ``atom`` (0-based index into the atom dataframe from
        :func:`get_atom_dataframe`), ``x``, ``y``, ``z`` (Å), ``charge``
        (e), ``area`` (Å²).

    Raises
    ------
    ValueError
        If ``solvation.SegmentList`` is JSON ``null``, or any entry (or
        component) is ``null``.
    """
    segment_list = data["solvation"]["SegmentList"]
    if segment_list is None or any(
        entry is None or any(c is None for c in entry) for entry in segment_list
    ):
        raise ValueError(
            "CHAOS record has null solvation.SegmentList; cannot build a segment table."
        )
    rows = [
        {
            "atom": parent_atom_index - 1,
            "x": x * SEGMENT_POSITION_CONVERSION_FACTOR,
            "y": y * SEGMENT_POSITION_CONVERSION_FACTOR,
            "z": z * SEGMENT_POSITION_CONVERSION_FACTOR,
            "charge": charge,
            "area": area,
        }
        for (
            _segment_index,
            parent_atom_index,
            x,
            y,
            z,
            charge,
            area,
            _sigma,
            _potential,
        ) in segment_list
    ]
    return pd.DataFrame(rows, columns=["atom", "x", "y", "z", "charge", "area"])


def get_volume(data: dict) -> float:
    """Parse the cavity volume from a CHAOS JSON record.

    ``solvation.CavVolume`` is reported in Bohr³ and is converted to Å³
    here.

    Parameters
    ----------
    data : dict
        Contents of a CHAOS JSON file.

    Returns
    -------
    float
        Cavity volume in Å³.

    Raises
    ------
    ValueError
        If ``solvation.CavVolume`` is JSON ``null``.
    """
    cav_volume = data["solvation"]["CavVolume"]
    if cav_volume is None:
        raise ValueError(
            "CHAOS record has null solvation.CavVolume; cannot compute a cavity volume."
        )
    return float(cav_volume) * VOLUME_CONVERSION_FACTOR
