"""Unit tests for the CHAOS JSON parser."""

import copy
import json
from importlib.resources import files

import pandas as pd
import pytest

from cosmolayer.parser import chaos, parse_cosmo_file


@pytest.fixture
def chaos_json() -> str:
    path = files("cosmolayer.data") / "chaos_sample.json"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def chaos_dict(chaos_json: str) -> dict:
    return json.loads(chaos_json)


def test_is_chaos_json_true_for_chaos_record(chaos_json: str) -> None:
    assert chaos.is_chaos_json(chaos_json) is True


def test_is_chaos_json_false_for_non_json_text() -> None:
    assert chaos.is_chaos_json("not json at all") is False


def test_is_chaos_json_false_for_unrelated_json() -> None:
    assert chaos.is_chaos_json('{"foo": "bar"}') is False


def test_get_atom_dataframe_shape_and_columns(chaos_dict: dict) -> None:
    df = chaos.get_atom_dataframe(chaos_dict)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "x", "y", "z", "element"]
    assert len(df) == 15


def test_get_atom_dataframe_element_order_matches_atom_list(chaos_dict: dict) -> None:
    df = chaos.get_atom_dataframe(chaos_dict)
    expected_elements = [a["element"] for a in chaos_dict["general"]["AtomList"]]
    assert list(df["element"]) == expected_elements
    assert expected_elements[:8] == ["C", "C", "C", "C", "C", "C", "C", "F"]
    assert expected_elements[8:] == ["H"] * 7


def test_get_atom_dataframe_coordinates_match_structural_block(
    chaos_dict: dict,
) -> None:
    df = chaos.get_atom_dataframe(chaos_dict)
    expected_first = chaos_dict["structural"]["Coordinates"][0]
    expected_last = chaos_dict["structural"]["Coordinates"][-1]
    assert df.iloc[0][["x", "y", "z"]].tolist() == pytest.approx(expected_first)
    assert df.iloc[-1][["x", "y", "z"]].tolist() == pytest.approx(expected_last)


def test_get_atom_dataframe_ids_are_unique_strings(chaos_dict: dict) -> None:
    df = chaos.get_atom_dataframe(chaos_dict)
    assert df["id"].map(type).eq(str).all()
    assert df["id"].is_unique


def test_get_atom_dataframe_rejects_null_coordinates(chaos_dict: dict) -> None:
    data = copy.deepcopy(chaos_dict)
    data["structural"]["Coordinates"] = None
    with pytest.raises(ValueError, match="null"):
        chaos.get_atom_dataframe(data)


def test_get_atom_dataframe_rejects_null_coordinate_entry(chaos_dict: dict) -> None:
    data = copy.deepcopy(chaos_dict)
    data["structural"]["Coordinates"][0] = None
    with pytest.raises(ValueError, match="null"):
        chaos.get_atom_dataframe(data)


def test_get_segment_dataframe_shape_and_columns(chaos_dict: dict) -> None:
    df = chaos.get_segment_dataframe(chaos_dict)
    assert list(df.columns) == ["atom", "x", "y", "z", "charge", "area"]
    assert len(df) == 1226


def test_get_segment_dataframe_atom_index_is_zero_based(chaos_dict: dict) -> None:
    df = chaos.get_segment_dataframe(chaos_dict)
    n_atoms = len(chaos_dict["general"]["AtomList"])
    assert df["atom"].min() >= 0
    assert df["atom"].max() <= n_atoms - 1
    # first raw segment row has parent_atom_index == 1 (1-based) -> 0 here
    assert df["atom"].iloc[0] == 0
    # last raw segment row has parent_atom_index == 15 (1-based) -> 14 here
    assert df["atom"].iloc[-1] == 14


def test_get_segment_dataframe_positions_converted_from_bohr(chaos_dict: dict) -> None:
    df = chaos.get_segment_dataframe(chaos_dict)
    # raw SegmentList[0] = [1, 1, -3.158137715, -1.863936184, 3.195421904, ...]
    expected_x_angstrom = -3.158137715 * 0.52917721067
    expected_y_angstrom = -1.863936184 * 0.52917721067
    expected_z_angstrom = 3.195421904 * 0.52917721067
    row = df.iloc[0]
    assert row["x"] == pytest.approx(expected_x_angstrom)
    assert row["y"] == pytest.approx(expected_y_angstrom)
    assert row["z"] == pytest.approx(expected_z_angstrom)


def test_get_segment_dataframe_charge_and_area_unconverted(chaos_dict: dict) -> None:
    df = chaos.get_segment_dataframe(chaos_dict)
    # raw SegmentList[0] charge=-1.0343e-05, area=0.042547346 (already e / Å²)
    row = df.iloc[0]
    assert row["charge"] == pytest.approx(-1.0343e-05)
    assert row["area"] == pytest.approx(0.042547346)


def test_get_segment_dataframe_area_sum_matches_atom_cosmo_charge(
    chaos_dict: dict,
) -> None:
    df = chaos.get_segment_dataframe(chaos_dict)
    per_atom_area = df.groupby("atom")["area"].sum()
    atom_cosmo = chaos_dict["solvation"]["AtomCOSMOCharge"]
    for atom_idx, expected in enumerate(atom_cosmo):
        assert per_atom_area.loc[atom_idx] == pytest.approx(expected["area"], abs=1e-4)


def test_get_segment_dataframe_rejects_null_segment_list(chaos_dict: dict) -> None:
    data = copy.deepcopy(chaos_dict)
    data["solvation"]["SegmentList"] = None
    with pytest.raises(ValueError, match="null"):
        chaos.get_segment_dataframe(data)


def test_get_segment_dataframe_rejects_null_segment_entry(chaos_dict: dict) -> None:
    data = copy.deepcopy(chaos_dict)
    data["solvation"]["SegmentList"][0] = None
    with pytest.raises(ValueError, match="null"):
        chaos.get_segment_dataframe(data)


def test_get_segment_dataframe_rejects_null_segment_component(chaos_dict: dict) -> None:
    data = copy.deepcopy(chaos_dict)
    data["solvation"]["SegmentList"][0][2] = None
    with pytest.raises(ValueError, match="null"):
        chaos.get_segment_dataframe(data)


def test_get_volume_converts_bohr_cubed_to_angstrom_cubed(chaos_dict: dict) -> None:
    volume = chaos.get_volume(chaos_dict)
    # raw solvation.CavVolume = 916.9 (Bohr^3)
    expected = 916.9 * 0.52917721067**3
    assert volume == pytest.approx(expected)
    assert isinstance(volume, float)


def test_get_volume_rejects_null_cav_volume(chaos_dict: dict) -> None:
    data = copy.deepcopy(chaos_dict)
    data["solvation"]["CavVolume"] = None
    with pytest.raises(ValueError, match="null"):
        chaos.get_volume(data)


def test_parse_cosmo_file_detects_chaos_format(chaos_json: str) -> None:
    fmt, atom_df, segment_df, volume = parse_cosmo_file(chaos_json)
    assert fmt == "CHAOS"
    assert len(atom_df) == 15
    assert len(segment_df) == 1226
    assert volume == pytest.approx(916.9 * 0.52917721067**3)


def test_is_chaos_json_false_for_top_level_keys_missing_nested_fields() -> None:
    assert chaos.is_chaos_json('{"general":{},"structural":{},"solvation":{}}') is False


def test_parse_cosmo_file_raises_value_error_for_incomplete_chaos_record() -> None:
    with pytest.raises(ValueError, match="Could not parse COSMO file"):
        parse_cosmo_file('{"general":{},"structural":{},"solvation":{}}')


def test_parse_cosmo_file_still_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Could not parse COSMO file"):
        parse_cosmo_file("this is neither DMol-3, TURBOMOLE, nor CHAOS")
