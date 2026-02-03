"""Integration test for DMol-3 parser and Component class.

This test verifies that the DMol-3 parser correctly reads COSMO files
and that the Component class correctly calculates sigma profiles by
comparing against precalculated reference values.
"""

import json
import re
from importlib.resources import files

import numpy as np
import pytest

from cosmolayer.cosmosac import Component


def load_reference_sigma_profiles(
    sigma_file_path: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], float, float]:
    """Load precalculated sigma profiles from a .sigma file.

    Parameters
    ----------
    sigma_file_path : str
        Path to the .sigma file containing reference data.

    Returns
    -------
    sigma_grid : np.ndarray
        Array of sigma values (charge density grid points).
    profiles : dict[str, np.ndarray]
        Dictionary mapping segment group names to their sigma profile values.
    area : float
        Total molecular surface area parsed from the file's metadata.
    volume : float
        Total molecular volume parsed from the file's metadata.
    """

    with open(sigma_file_path) as f:
        lines = f.readlines()

    # Parse area and volume from the metadata on the first line
    meta_line = None
    for line in lines:
        if line.lstrip().startswith("# meta:"):
            meta_line = line
            break
    if meta_line is None:
        raise ValueError("No metadata line found in .sigma file")

    # Extract the JSON from '# meta: ...'
    meta_json_str = re.sub(r"^#\s*meta:\s*", "", meta_line.strip())
    meta = json.loads(meta_json_str)
    area = float(meta["area [A^2]"])
    volume = float(meta["volume [A^3]"])

    # Skip header lines and filter out comment lines
    data_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    # Parse the data - expecting 3 groups of 51 lines each (NHB, OH, OT)
    nhb_profile = []
    oh_profile = []
    ot_profile = []

    # First 51 lines: NHB
    for line in data_lines[:51]:
        _, psigma = map(float, line.split())
        nhb_profile.append(psigma)

    # Next 51 lines: OH
    for line in data_lines[51:102]:
        _, psigma = map(float, line.split())
        oh_profile.append(psigma)

    # Remaining lines: OT
    for line in data_lines[102:]:
        _, psigma = map(float, line.split())
        ot_profile.append(psigma)

    # Build sigma grid from first profile
    sigma_grid = np.array(
        [float(line.split()[0]) for line in data_lines[:51]], dtype=np.float64
    )

    profiles = {
        "NHB": np.array(nhb_profile, dtype=np.float64),
        "OH": np.array(oh_profile, dtype=np.float64),
        "OT": np.array(ot_profile, dtype=np.float64),
    }

    return sigma_grid, profiles, area, volume


def test_dmol3_parser_integration() -> None:
    """Test DMol-3 parser and Component class integration.

    This test verifies that:
    1. The DMol-3 parser correctly reads NCCO.cosmo
    2. The Component class correctly processes DMol-3 data
    3. Calculated sigma profiles match precalculated reference values
    """
    # Load the COSMO file using the Component class with default parameters
    cosmo_path = files("cosmolayer.data") / "NCCO.cosmo"
    component = Component(str(cosmo_path))

    # Load reference sigma profiles
    sigma_path = files("cosmolayer.data") / "NCCO.sigma3"
    expected_sigma_grid, expected_profiles, expected_area, expected_volume = (
        load_reference_sigma_profiles(str(sigma_path))
    )

    # Get calculated sigma grid
    calculated_sigma_grid = component.get_sigma_grid()

    # Verify sigma grid matches exactly
    np.testing.assert_allclose(
        calculated_sigma_grid,
        expected_sigma_grid,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Sigma grid does not match expected values",
    )

    # Get calculated sigma profiles for each segment group
    calculated_profiles = {
        group: component.get_sigma_profile(group) for group in ["NHB", "OH", "OT"]
    }

    # Verify basic properties match metadata from .sigma file
    assert component.get_area() == pytest.approx(expected_area, abs=1e-4)
    assert component.get_volume() == pytest.approx(expected_volume, abs=1e-5)

    # Verify that the calculated profiles have reasonable properties
    for group_name in ["NHB", "OH", "OT"]:
        calculated = calculated_profiles[group_name]
        expected = expected_profiles[group_name]

        # All values should be non-negative
        assert np.all(calculated >= 0).item(), (
            f"{group_name} profile should be non-negative"
        )

        # Profile should be finite
        assert np.all(np.isfinite(calculated)).item(), (
            f"{group_name} profile should be finite"
        )

        # Verify that the calculated profiles match the expected profiles
        np.testing.assert_allclose(
            calculated,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{group_name} profile does not match expected values",
        )
