"""Tests for Mixture add_component, remove_component, and replace_component."""

from importlib.resources import files

import numpy as np
import pytest

from cosmolayer.cosmosac import Mixture


def _read_cosmo(name: str) -> str:
    """Load COSMO string from package data."""
    path = files("cosmolayer") / "data" / f"{name}.cosmo"
    return path.read_text()


@pytest.fixture
def cosmo_strings() -> dict[str, str]:
    """COSMO file contents keyed by a short label (SMILES or name)."""
    return {
        "water": _read_cosmo("O"),
        "fluoromethane": _read_cosmo("CF"),
        "ethanolamine": _read_cosmo("NCCO"),
        "aminoethenol": _read_cosmo("C=C(N)O"),
    }


def test_add_component_adds_and_accessible(
    cosmo_strings: dict[str, str],
) -> None:
    """Adding a component increases len, updates names, and component is accessible."""
    mixture = Mixture({"water": cosmo_strings["water"]})
    assert len(mixture) == 1
    assert mixture.get_component_names() == ("water",)

    mixture.add_component("fluoromethane", cosmo_strings["fluoromethane"])
    assert len(mixture) == 2
    assert mixture.get_component_names() == ("water", "fluoromethane")

    areas = mixture.get_areas()
    assert areas.shape == (2,)
    np.testing.assert_allclose(
        mixture["water"].get_area(),
        areas[0],
    )
    np.testing.assert_allclose(
        mixture["fluoromethane"].get_area(),
        areas[1],
    )


def test_add_component_preserves_order(
    cosmo_strings: dict[str, str],
) -> None:
    """Components stay in insertion order after add_component."""
    mixture = Mixture(
        {
            "a": cosmo_strings["water"],
            "b": cosmo_strings["fluoromethane"],
        }
    )
    mixture.add_component("c", cosmo_strings["ethanolamine"])
    assert mixture.get_component_names() == ("a", "b", "c")


def test_remove_component_reduces_len_and_raises_for_removed(
    cosmo_strings: dict[str, str],
) -> None:
    """Removing a component decreases len and KeyError when accessing it."""
    mixture = Mixture(
        {
            "water": cosmo_strings["water"],
            "fluoromethane": cosmo_strings["fluoromethane"],
        }
    )
    assert len(mixture) == 2

    mixture.remove_component("water")
    assert len(mixture) == 1
    assert mixture.get_component_names() == ("fluoromethane",)
    with pytest.raises(KeyError, match="water"):
        _ = mixture["water"]
    assert mixture["fluoromethane"].get_area() > 0


def test_remove_component_raises_for_unknown_name(
    cosmo_strings: dict[str, str],
) -> None:
    """Removing a non-existent component raises KeyError."""
    mixture = Mixture({"water": cosmo_strings["water"]})
    with pytest.raises(KeyError, match="nonexistent"):
        mixture.remove_component("nonexistent")


def test_replace_component_swaps_component_preserves_order(
    cosmo_strings: dict[str, str],
) -> None:
    """Replacing a component changes name and COSMO data but preserves position."""
    mixture = Mixture(
        {
            "first": cosmo_strings["water"],
            "second": cosmo_strings["fluoromethane"],
            "third": cosmo_strings["ethanolamine"],
        }
    )
    area_second_before = mixture["second"].get_area()

    mixture.replace_component(
        "second",
        "replacement",
        cosmo_strings["aminoethenol"],
    )
    assert mixture.get_component_names() == ("first", "replacement", "third")
    assert mixture["replacement"].get_area() != area_second_before
    with pytest.raises(KeyError, match="second"):
        _ = mixture["second"]


def test_replace_component_same_name_new_cosmo(
    cosmo_strings: dict[str, str],
) -> None:
    """Replacing with the same name but new COSMO string updates the component."""
    mixture = Mixture(
        {
            "water": cosmo_strings["water"],
            "other": cosmo_strings["fluoromethane"],
        }
    )
    area_water_before = mixture["water"].get_area()

    mixture.replace_component("water", "water", cosmo_strings["ethanolamine"])
    assert mixture.get_component_names() == ("water", "other")
    assert mixture["water"].get_area() != area_water_before
    np.testing.assert_allclose(
        mixture["water"].get_area(),
        mixture.get_areas()[0],
    )


def test_replace_component_same_name_only_updates_data(
    cosmo_strings: dict[str, str],
) -> None:
    """Replacing with the same name (no rename) updates COSMO data."""
    mixture = Mixture(
        {
            "water": cosmo_strings["water"],
            "other": cosmo_strings["fluoromethane"],
        }
    )
    names_before = mixture.get_component_names()
    area_water_before = mixture["water"].get_area()

    mixture.replace_component("water", "water", cosmo_strings["ethanolamine"])

    assert mixture.get_component_names() == names_before
    assert mixture["water"].get_area() != area_water_before
    np.testing.assert_allclose(
        mixture["water"].get_area(),
        mixture.get_areas()[0],
    )


def test_replace_component_new_name_already_exists_raises(
    cosmo_strings: dict[str, str],
) -> None:
    """Replacing with a new name that already exists in the mixture raises."""
    mixture = Mixture(
        {
            "water": cosmo_strings["water"],
            "other": cosmo_strings["fluoromethane"],
        }
    )
    names_before = mixture.get_component_names()

    with pytest.raises(ValueError, match="Component other already exists in mixture"):
        mixture.replace_component(
            "water",
            "other",
            cosmo_strings["ethanolamine"],
        )
    assert mixture.get_component_names() == names_before


def test_replace_component_unknown_old_name_leaves_mixture_unchanged(
    cosmo_strings: dict[str, str],
) -> None:
    """Replacing a non-existent component raises error."""
    mixture = Mixture(
        {
            "water": cosmo_strings["water"],
            "other": cosmo_strings["fluoromethane"],
        }
    )

    with pytest.raises(ValueError, match="Component nonexistent not found in mixture"):
        mixture.replace_component(
            "nonexistent",
            "new",
            cosmo_strings["ethanolamine"],
        )


def test_add_remove_replace_roundtrip_get_areas_consistent(
    cosmo_strings: dict[str, str],
) -> None:
    """Build same mixture via constructor vs add/remove/replace; get_areas match."""
    direct = Mixture(
        {
            "water": cosmo_strings["water"],
            "fluoromethane": cosmo_strings["fluoromethane"],
        }
    )
    areas_direct = direct.get_areas()

    built = Mixture({"water": cosmo_strings["water"]})
    built.add_component("fluoromethane", cosmo_strings["fluoromethane"])
    areas_built = built.get_areas()
    np.testing.assert_allclose(areas_direct, areas_built)

    built.remove_component("fluoromethane")
    assert len(built) == 1
    np.testing.assert_allclose(built.get_areas()[0], areas_direct[0])

    built.add_component("fluoromethane", cosmo_strings["fluoromethane"])
    built.replace_component(
        "fluoromethane", "fluoromethane", cosmo_strings["fluoromethane"]
    )
    np.testing.assert_allclose(built.get_areas(), areas_direct)
