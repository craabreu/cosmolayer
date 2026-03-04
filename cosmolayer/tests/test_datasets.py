"""Tests for CosmoFileDataset in cosmolayer.cosmosac.datasets."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cosmolayer.cosmolayer import CosmoLayer
from cosmolayer.cosmosac.datasets import CosmoFileDataset, _compute_component_properties
from cosmolayer.cosmosac.model import CosmoSac2002Model, CosmoSac2010Model

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_DATA = files("cosmolayer.data")


def _cosmo_path(name: str) -> str:
    """Return the string path to a bundled .cosmo file."""
    return str(_DATA / f"{name}.cosmo")


@pytest.fixture()
def binary_df() -> pd.DataFrame:
    """Two-row DataFrame for a water/fluoromethane binary mixture."""
    return pd.DataFrame(
        {
            "component_0": ["water", "water"],
            "mole_fraction_0": [0.3, 0.7],
            "component_1": ["fluoromethane", "fluoromethane"],
            "mole_fraction_1": [0.7, 0.3],
            "temperature": [298.15, 320.0],
        }
    )


@pytest.fixture()
def binary_mapping() -> dict[str, str]:
    """Component-name → file-path mapping for the binary mixture."""
    return {
        "water": _cosmo_path("O"),
        "fluoromethane": _cosmo_path("CF"),
    }


@pytest.fixture()
def binary_dataset(
    binary_df: pd.DataFrame, binary_mapping: dict[str, str]
) -> CosmoFileDataset:
    """CosmoFileDataset for the binary mixture using the 2002 model."""
    return CosmoFileDataset(binary_df, binary_mapping, CosmoSac2002Model)


# ---------------------------------------------------------------------------
# _count_components
# ---------------------------------------------------------------------------


class TestCountComponents:
    def test_returns_zero_when_no_matching_columns(self) -> None:
        df = pd.DataFrame({"other": [1, 2]})
        assert CosmoFileDataset._count_components(df, "component_{}".format) == 0

    def test_returns_one_when_single_column_present(self) -> None:
        df = pd.DataFrame({"component_0": [1], "other": [2]})
        assert CosmoFileDataset._count_components(df, "component_{}".format) == 1

    def test_returns_two_when_two_consecutive_columns_present(self) -> None:
        df = pd.DataFrame({"component_0": [], "component_1": [], "extra": []})
        assert CosmoFileDataset._count_components(df, "component_{}".format) == 2

    def test_stops_at_gap_in_sequence(self) -> None:
        # component_0 and component_2 are present but component_1 is absent
        df = pd.DataFrame({"component_0": [], "component_2": []})
        assert CosmoFileDataset._count_components(df, "component_{}".format) == 1

    def test_non_default_formatter(self) -> None:
        df = pd.DataFrame({"x_0": [], "x_1": [], "x_2": []})
        assert CosmoFileDataset._count_components(df, "x_{}".format) == 3


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_raises_when_no_component_columns(self) -> None:
        df = pd.DataFrame({"other": [1]})
        with pytest.raises(ValueError, match="No components found"):
            CosmoFileDataset(df, {}, CosmoSac2002Model, temperature=298.15)

    def test_raises_when_mole_fraction_count_mismatches_components(self) -> None:
        # 1 component column, 2 mole-fraction columns
        df = pd.DataFrame(
            {
                "component_0": ["water"],
                "mole_fraction_0": [0.4],
                "mole_fraction_1": [0.6],
            }
        )
        with pytest.raises(ValueError, match="Number of mole fractions"):
            CosmoFileDataset(
                df,
                {"water": _cosmo_path("O")},
                CosmoSac2002Model,
                temperature=298.15,
            )

    def test_raises_when_temperature_column_missing(self) -> None:
        df = pd.DataFrame({"component_0": ["water"], "mole_fraction_0": [1.0]})
        with pytest.raises(ValueError, match="Temperature column not found"):
            CosmoFileDataset(
                df,
                {"water": _cosmo_path("O")},
                CosmoSac2002Model,
                temperature="T",  # column "T" does not exist
            )

    def test_succeeds_with_fixed_temperature(self) -> None:
        df = pd.DataFrame({"component_0": ["water"], "mole_fraction_0": [1.0]})
        ds = CosmoFileDataset(
            df,
            {"water": _cosmo_path("O")},
            CosmoSac2002Model,
            temperature=298.15,
        )
        assert len(ds) == 1

    def test_succeeds_with_temperature_column(self) -> None:
        df = pd.DataFrame(
            {"component_0": ["water"], "mole_fraction_0": [1.0], "T": [298.15]}
        )
        ds = CosmoFileDataset(
            df,
            {"water": _cosmo_path("O")},
            CosmoSac2002Model,
            temperature="T",
        )
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# mixture_dataframe property
# ---------------------------------------------------------------------------


class TestMixtureDataframe:
    def test_returns_same_dataframe_object(
        self, binary_df: pd.DataFrame, binary_mapping: dict[str, str]
    ) -> None:
        ds = CosmoFileDataset(
            binary_df, binary_mapping, CosmoSac2002Model, temperature=298.15
        )
        assert ds.mixture_dataframe is binary_df

    def test_contains_expected_columns(self, binary_dataset: CosmoFileDataset) -> None:
        cols = set(binary_dataset.mixture_dataframe.columns)
        assert "component_0" in cols
        assert "mole_fraction_0" in cols
        assert "temperature" in cols


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    def test_matches_dataframe_row_count(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        assert len(binary_dataset) == 2

    def test_single_row_dataset(self, binary_mapping: dict[str, str]) -> None:
        df = pd.DataFrame(
            {
                "component_0": ["water"],
                "mole_fraction_0": [1.0],
            }
        )
        ds = CosmoFileDataset(df, binary_mapping, CosmoSac2002Model, temperature=300.0)
        assert len(ds) == 1

    def test_five_row_dataset(self, binary_mapping: dict[str, str]) -> None:
        df = pd.DataFrame(
            {
                "component_0": ["water"] * 5,
                "mole_fraction_0": [0.2, 0.4, 0.5, 0.6, 0.8],
                "component_1": ["fluoromethane"] * 5,
                "mole_fraction_1": [0.8, 0.6, 0.5, 0.4, 0.2],
            }
        )
        ds = CosmoFileDataset(df, binary_mapping, CosmoSac2002Model, temperature=298.15)
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# __getitem__ — temperature column
# ---------------------------------------------------------------------------


class TestGetitemWithTemperatureColumn:
    def test_temperature_read_from_column(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        T0, _, _, _, _ = binary_dataset[0]
        T1, _, _, _, _ = binary_dataset[1]
        assert T0 == pytest.approx(298.15)
        assert T1 == pytest.approx(320.0)

    def test_mole_fractions_match_dataframe(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        _, fracs, _, _, _ = binary_dataset[0]
        assert fracs[0] == pytest.approx(0.3)
        assert fracs[1] == pytest.approx(0.7)

    def test_return_array_shapes(self, binary_dataset: CosmoFileDataset) -> None:
        """Returned arrays have shapes (n,) and (n, m) for CosmoLayer.forward."""
        _, fracs, areas, volumes, probs = binary_dataset[0]
        n = 2
        assert fracs.shape == (n,)
        assert areas.shape == (n,)
        assert volumes.shape == (n,)
        assert probs.shape == (n, 51)

    def test_probabilities_shape_2002_model(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        """COSMO-SAC 2002 merges profiles → (n, 51)."""
        _, _, _, _, probs = binary_dataset[0]
        assert probs.shape == (2, 51)

    def test_probabilities_shape_2010_model(
        self, binary_df: pd.DataFrame, binary_mapping: dict[str, str]
    ) -> None:
        """COSMO-SAC 2010 keeps NHB/OH/OT split → (n, 153)."""
        ds = CosmoFileDataset(binary_df, binary_mapping, CosmoSac2010Model)
        _, _, _, _, probs = ds[0]
        assert probs.shape == (2, 153)

    def test_area_is_positive(self, binary_dataset: CosmoFileDataset) -> None:
        _, _, areas, _, _ = binary_dataset[0]
        assert np.all(areas > 0.0)

    def test_volume_is_positive(self, binary_dataset: CosmoFileDataset) -> None:
        _, _, _, volumes, _ = binary_dataset[0]
        assert np.all(volumes > 0.0)

    def test_second_row_different_temperature(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        T0, _, _, _, _ = binary_dataset[0]
        T1, _, _, _, _ = binary_dataset[1]
        assert T0 != T1


# ---------------------------------------------------------------------------
# __getitem__ — fixed temperature
# ---------------------------------------------------------------------------


class TestGetitemWithFixedTemperature:
    def test_fixed_temperature_propagated_to_all_samples(
        self, binary_mapping: dict[str, str]
    ) -> None:
        df = pd.DataFrame(
            {
                "component_0": ["water", "water"],
                "mole_fraction_0": [0.3, 0.7],
                "component_1": ["fluoromethane", "fluoromethane"],
                "mole_fraction_1": [0.7, 0.3],
            }
        )
        ds = CosmoFileDataset(df, binary_mapping, CosmoSac2002Model, temperature=310.0)
        T0, _, _, _, _ = ds[0]
        T1, _, _, _, _ = ds[1]
        assert T0 == pytest.approx(310.0)
        assert T1 == pytest.approx(310.0)

    def test_fixed_temperature_does_not_require_temperature_column(
        self, binary_mapping: dict[str, str]
    ) -> None:
        df = pd.DataFrame(
            {
                "component_0": ["water"],
                "mole_fraction_0": [1.0],
            }
        )
        ds = CosmoFileDataset(df, binary_mapping, CosmoSac2002Model, temperature=350.0)
        T, _, _, _, _ = ds[0]
        assert T == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# __getitem__ — prefix
# ---------------------------------------------------------------------------


class TestGetitemWithPrefix:
    def test_prefix_applied_to_relative_paths(self, tmp_path: Path) -> None:
        """Dataset constructed with prefix=tmp_path and bare filenames in mapping."""
        # Write bundled COSMO files into tmp_path under bare names
        for name, bundle_name in [("water.cosmo", "O"), ("cf.cosmo", "CF")]:
            content = (_DATA / f"{bundle_name}.cosmo").read_text()
            (tmp_path / name).write_text(content)

        df = pd.DataFrame(
            {
                "component_0": ["water"],
                "mole_fraction_0": [0.5],
                "component_1": ["fluoromethane"],
                "mole_fraction_1": [0.5],
            }
        )
        mapping = {"water": "water.cosmo", "fluoromethane": "cf.cosmo"}
        ds = CosmoFileDataset(
            df, mapping, CosmoSac2002Model, prefix=tmp_path, temperature=298.15
        )

        T, fracs, areas, volumes, probs = ds[0]
        assert T == pytest.approx(298.15)
        assert fracs.shape == (2,)
        assert np.all(areas > 0.0)
        assert probs.shape == (2, 51)

    def test_prefix_result_matches_absolute_path_dataset(
        self, tmp_path: Path, binary_mapping: dict[str, str]
    ) -> None:
        """Results with prefix+relative path equal results with absolute paths."""
        # Write the same files under tmp_path
        (tmp_path / "O.cosmo").write_text((_DATA / "O.cosmo").read_text())
        (tmp_path / "CF.cosmo").write_text((_DATA / "CF.cosmo").read_text())

        df = pd.DataFrame(
            {
                "component_0": ["water"],
                "mole_fraction_0": [0.4],
                "component_1": ["fluoromethane"],
                "mole_fraction_1": [0.6],
            }
        )

        ds_abs = CosmoFileDataset(
            df, binary_mapping, CosmoSac2002Model, temperature=298.15
        )
        ds_prefix = CosmoFileDataset(
            df,
            {"water": "O.cosmo", "fluoromethane": "CF.cosmo"},
            CosmoSac2002Model,
            prefix=tmp_path,
            temperature=298.15,
        )

        _, _, areas_abs, volumes_abs, probs_abs = ds_abs[0]
        _, _, areas_prefix, volumes_prefix, probs_prefix = ds_prefix[0]

        np.testing.assert_allclose(areas_abs, areas_prefix)
        np.testing.assert_allclose(volumes_abs, volumes_prefix)
        np.testing.assert_array_equal(probs_abs, probs_prefix)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    def test_repeated_access_returns_numerically_identical_arrays(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        """Repeated __getitem__ returns same values (cache hit)."""
        _, fracs1, areas1, vols1, probs1 = binary_dataset[0]
        _, fracs2, areas2, vols2, probs2 = binary_dataset[0]
        np.testing.assert_array_equal(fracs1, fracs2)
        np.testing.assert_array_equal(areas1, areas2)
        np.testing.assert_array_equal(vols1, vols2)
        np.testing.assert_array_equal(probs1, probs2)

    def test_different_samples_same_component_share_cache(
        self, binary_mapping: dict[str, str]
    ) -> None:
        """Same component in different rows yields same areas/volumes/probs (cache)."""
        df = pd.DataFrame(
            {
                "component_0": ["water", "water"],
                "mole_fraction_0": [0.3, 0.7],
                "component_1": ["fluoromethane", "fluoromethane"],
                "mole_fraction_1": [0.7, 0.3],
            }
        )
        ds = CosmoFileDataset(df, binary_mapping, CosmoSac2002Model, temperature=298.15)

        _, _, areas0, _, probs0 = ds[0]
        _, _, areas1, _, probs1 = ds[1]

        # Same component index → same cached properties
        np.testing.assert_array_equal(areas0, areas1)
        np.testing.assert_array_equal(probs0, probs1)

    def test_compute_component_properties_direct_cache_hit(self) -> None:
        path = _cosmo_path("O")
        result_a = _compute_component_properties(CosmoSac2002Model, path)
        result_b = _compute_component_properties(CosmoSac2002Model, path)
        assert result_a is result_b


# ---------------------------------------------------------------------------
# CosmoLayer integration
# ---------------------------------------------------------------------------


class TestCosmoLayerIntegration:
    """Dataset output is feedable to CosmoLayer.forward."""

    def test_dataset_sample_feedable_to_cosmo_layer(
        self, binary_dataset: CosmoFileDataset
    ) -> None:
        T, fracs, areas, volumes, probs = binary_dataset[0]
        mixture = CosmoSac2002Model.create_mixture(
            {
                "water": (_DATA / "O.cosmo").read_text(),
                "fluoromethane": (_DATA / "CF.cosmo").read_text(),
            }
        )
        layer = CosmoLayer(
            mixture.interaction_matrices(298.15),
            CosmoSac2002Model.temperature_exponents,
            mixture.area_per_segment,
            reference_temperature=298.15,
        )
        # Add batch dimension for forward: (..., n) or (..., n, m)
        temp = torch.tensor(T)
        fracs_t = torch.tensor(fracs, dtype=torch.float64).unsqueeze(0)
        areas_t = torch.tensor(areas, dtype=torch.float64).unsqueeze(0)
        volumes_t = torch.tensor(volumes, dtype=torch.float64).unsqueeze(0)
        probs_t = torch.tensor(probs, dtype=torch.float64).unsqueeze(0)
        ln_gamma = layer(temp, fracs_t, areas_t, volumes_t, probs_t)
        assert ln_gamma.shape == (1, 2)
        assert torch.isfinite(ln_gamma).all()
