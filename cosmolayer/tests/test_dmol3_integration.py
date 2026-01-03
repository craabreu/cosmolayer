"""Integration test for DMol-3 parser and Component class.

This test verifies that the DMol-3 parser correctly reads COSMO files
and that the Component class correctly calculates sigma profiles by
comparing against precalculated reference values.
"""

from importlib.resources import files

import numpy as np
import pytest

from cosmolayer.sac import Component


def load_reference_sigma_profiles(
    sigma_file_path: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    """
    with open(sigma_file_path) as f:
        lines = f.readlines()

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
        sigma, psigma = map(float, line.split())
        nhb_profile.append(psigma)

    # Next 51 lines: OH
    for line in data_lines[51:102]:
        sigma, psigma = map(float, line.split())
        oh_profile.append(psigma)

    # Remaining lines: OT
    for line in data_lines[102:]:
        sigma, psigma = map(float, line.split())
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

    return sigma_grid, profiles


def test_dmol3_parser_integration() -> None:
    """Test DMol-3 parser and Component class integration.

    This test verifies that:
    1. The DMol-3 parser correctly reads NCCO.cosmo
    2. The Component class correctly processes DMol-3 data
    3. Calculated sigma profiles match precalculated reference values
    """
    # Load the COSMO file using the Component class with default parameters
    cosmo_path = files("cosmolayer.data") / "NCCO.cosmo"
    component = Component(cosmo_path)  # type: ignore[arg-type]

    # Load reference sigma profiles
    sigma_path = files("cosmolayer.data") / "NCCO.sigma"
    expected_sigma_grid, expected_profiles = load_reference_sigma_profiles(
        str(sigma_path)
    )

    # Get calculated sigma profiles for each segment group
    calculated_profiles = {
        group: component.get_sigma_profile(group) for group in ["NHB", "OH", "OT"]
    }

    # Get calculated sigma grid
    calculated_sigma_grid = component.get_sigma_grid()

    # Check that all segment groups are present
    assert set(calculated_profiles.keys()) == set(expected_profiles.keys()), (
        f"Expected segment groups {expected_profiles.keys()}, "
        f"got {calculated_profiles.keys()}"
    )

    # Verify sigma grid matches exactly
    np.testing.assert_allclose(
        calculated_sigma_grid,
        expected_sigma_grid,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Sigma grid does not match expected values",
    )

    # Verify basic properties match metadata from .sigma file
    assert component.get_area() == pytest.approx(103.51756, abs=1e-4)
    assert component.get_volume() == pytest.approx(86.10187, abs=1e-5)

    # Compare each sigma profile with strict tolerances
    for group_name in ["NHB", "OH", "OT"]:
        calculated = calculated_profiles[group_name]
        expected = expected_profiles[group_name]

        np.testing.assert_allclose(
            calculated,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Sigma profile for {group_name} does not match reference values",
        )


def test_dmol3_component_properties() -> None:
    """Test that Component correctly extracts properties from DMol-3 files."""
    cosmo_path = files("cosmolayer.data") / "NCCO.cosmo"
    component = Component(cosmo_path)  # type: ignore[arg-type]

    # Check basic molecular properties
    area = component.get_area()
    volume = component.get_volume()

    assert area > 0, "Molecular surface area should be positive"
    assert volume > 0, "Molecular volume should be positive"
    assert area == pytest.approx(103.51756, abs=1e-4)
    assert volume == pytest.approx(86.10187, abs=1e-5)

    # Check that sigma profiles are properly normalized
    # The sum of all segment group profiles should equal the total area
    profiles = {
        group: component.get_sigma_profile(group) for group in ["NHB", "OH", "OT"]
    }
    total_profile_area = float(sum(np.sum(profile) for profile in profiles.values()))

    assert total_profile_area == pytest.approx(area, abs=1e-4), (
        f"Total profile area {total_profile_area} should match molecular area {area}"
    )


def test_dmol3_vs_turbomole_consistency() -> None:
    """Test that DMol-3 and TurboMole parsers produce consistent Component objects."""
    # Load both COSMO files
    dmol3_path = files("cosmolayer.data") / "NCCO.cosmo"
    turbomole_path = files("cosmolayer.data") / "C=C(N)O.cosmo"

    # Both use default parameters
    dmol3_component = Component(dmol3_path)  # type: ignore[arg-type]
    turbomole_component = Component(turbomole_path)  # type: ignore[arg-type]

    # Both should have the same segment groups
    segment_groups = ["NHB", "OH", "OT"]

    # Both should have the same number of grid points (even if ranges differ)
    assert len(dmol3_component.get_sigma_grid()) == len(
        turbomole_component.get_sigma_grid()
    ), "Both components should have the same number of sigma grid points"

    # Both should produce valid (finite) sigma profiles for all segment groups
    for group_name in segment_groups:
        dmol3_profile = dmol3_component.get_sigma_profile(group_name)
        turbomole_profile = turbomole_component.get_sigma_profile(group_name)

        assert np.all(np.isfinite(dmol3_profile)).item(), (
            f"DMol-3 {group_name} profile should be finite"
        )
        assert np.all(dmol3_profile >= 0).item(), (
            f"DMol-3 {group_name} profile should be non-negative"
        )

        assert np.all(np.isfinite(turbomole_profile)).item(), (
            f"TurboMole {group_name} profile should be finite"
        )
        assert np.all(turbomole_profile >= 0).item(), (
            f"TurboMole {group_name} profile should be non-negative"
        )
