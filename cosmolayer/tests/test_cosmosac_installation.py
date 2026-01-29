"""
Test that the COSMOSAC reference package (cCOSMO) is installed and importable.

This test verifies that the COSMOSAC package from
https://github.com/usnistgov/COSMOSAC can be successfully imported as cCOSMO,
ensuring that the installation scripts (install_cosmosac.sh and
install_cosmosac.ps1) work correctly across platforms.
"""

import importlib.util

import pytest


def test_cCOSMO_installation() -> None:
    """Verify cCOSMO package can be found by import machinery."""
    spec = importlib.util.find_spec("cCOSMO")
    if spec is None:
        pytest.fail("cCOSMO module is not installed or importable")
