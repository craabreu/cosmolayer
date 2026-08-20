"""Integration test for the CHAOS JSON parser and Component class.

Verifies that Component, fed a CHAOS dataset record through the CHAOS
parser, reproduces that same record's own precomputed sigma profiles
(solvation.Sigma_NHB / Sigma_OH / Sigma_OT) and cavity area/volume. CHAOS
computes its sigma profiles with the COSMO-SAC-dsp protocol
(:cite:`Bell2020`), which is also Component's default parameterization, so
this is expected to match to floating-point precision once units are
handled correctly -- it is not merely a smoke test.

solvation.Sigma_NHB/OH/OT are each individually normalized to sum to 1;
solvation.Fraction_PartSigmas gives the area fraction of each class, and
solvation.Norm_Sigma_total (== CavArea in Angstrom^2) is the total cavity
area. The area-weighted profile for class c is reconstructed as
Fraction_PartSigmas[c] * Norm_Sigma_total * Sigma_c.
"""

import json
from importlib.resources import files

import numpy as np
import pytest

from cosmolayer.cosmosac import Component
from cosmolayer.cosmosac.segment_groups import NHB, OH, OT, SEGMENT_GROUPS

BOHR_TO_ANGSTROM = 0.52917721067


@pytest.fixture
def chaos_json() -> str:
    path = files("cosmolayer.data") / "Cc1ccccc1F.json"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def chaos_solvation(chaos_json: str) -> dict:
    return json.loads(chaos_json)["solvation"]


def reconstruct_area_weighted_profiles(solvation: dict) -> dict[str, np.ndarray]:
    """Reconstruct area-weighted (Angstrom^2) sigma profiles from CHAOS's
    normalized-probability representation."""
    norm = solvation["Norm_Sigma_total"]
    fraction_nhb, fraction_oh, fraction_ot = solvation["Fraction_PartSigmas"]
    return {
        NHB: np.array(solvation["Sigma_NHB"]) * fraction_nhb * norm,
        OH: np.array(solvation["Sigma_OH"]) * fraction_oh * norm,
        OT: np.array(solvation["Sigma_OT"]) * fraction_ot * norm,
    }


def test_chaos_parser_component_matches_chaos_reference_sigma_profiles(
    chaos_json: str, chaos_solvation: dict
) -> None:
    component = Component(chaos_json, merge_profiles=False)

    reference_profiles = reconstruct_area_weighted_profiles(chaos_solvation)

    calculated_stacked = component.sigma_profile
    calculated_profiles = {
        group: calculated_stacked[i] for i, group in enumerate(SEGMENT_GROUPS)
    }

    for group in SEGMENT_GROUPS:
        np.testing.assert_allclose(
            calculated_profiles[group],
            reference_profiles[group],
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"{group} profile does not match CHAOS reference",
        )


def test_chaos_parser_component_matches_chaos_reference_area(
    chaos_json: str, chaos_solvation: dict
) -> None:
    component = Component(chaos_json)
    expected_area = chaos_solvation["CavArea"] * BOHR_TO_ANGSTROM**2
    assert component.area == pytest.approx(expected_area, rel=1e-4)


def test_chaos_parser_component_matches_chaos_reference_volume(
    chaos_json: str, chaos_solvation: dict
) -> None:
    component = Component(chaos_json)
    expected_volume = chaos_solvation["CavVolume"] * BOHR_TO_ANGSTROM**3
    assert component.volume == pytest.approx(expected_volume)


def test_chaos_parser_component_cosmo_format_is_chaos(chaos_json: str) -> None:
    component = Component(chaos_json)
    assert component.cosmo_format == "CHAOS"
