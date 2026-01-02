"""
.. module:: cosmolayer.parser
   :synopsis: Parser for COSMO output files.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import os

import pandas as pd

from . import turbomole


def parse_cosmo_file(
    path: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Parse a COSMO output file.

    This function reads a COSMO (Conductor-like Screening Model) output file
    and extracts atomic coordinates, segment information, and molecular volume.

    Note
    ----
    The current implementation supports TURBOMOLE format. Support for additional
    COSMO file formats may be added in future versions.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the COSMO output file to parse.

    Returns
    -------
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
    >>> from importlib.resources import files
    >>> path = files("cosmolayer.data") / "C=C(N)O.cosmo"
    >>> atoms, segments, volume = parse_cosmo_file(path)
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
    """
    with open(path, encoding="utf-8", errors="replace") as file:
        contents = file.read()

    if "$segment_information" in contents and "$coord_car" in contents:
        return (
            turbomole.get_atom_dataframe(contents),
            turbomole.get_segment_dataframe(contents),
            turbomole.get_volume(contents),
        )
    else:
        raise ValueError("Could not parse COSMO file.")
