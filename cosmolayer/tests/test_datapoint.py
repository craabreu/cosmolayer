"""Tests for mixture datapoint helpers.

Includes MixtureDataset validation: rejection of empty mixture lists and of
datapoints with differing shapes (num_components, segment types, or targets).
"""

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmolayer.cosmodata import MixtureDatapoint, MixtureDataset
from cosmolayer.cosmosac import CosmoSac2002Model
from cosmolayer.cosmosac.datapoint import CosmoSacMixtureDatapoint


def _cosmo_files() -> list[Path]:
    """Return two packaged COSMO files for binary-mixture tests."""
    data = Path(str(files("cosmolayer") / "data"))
    return [data / "C=C(N)O.cosmo", data / "NCCO.cosmo"]


def _make_datapoint(
    num_targets: int = 1,
) -> MixtureDatapoint:
    """Build a simple 2-component datapoint for dataset tests."""
    return MixtureDatapoint(
        temperature=298.15,
        mole_fractions=np.array([0.5, 0.5]),
        areas=np.array([1.0, 2.0]),
        volumes=np.array([10.0, 20.0]),
        probabilities=np.array(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
            ]
        ),
        targets=np.arange(num_targets, dtype=float),
    )


def test_cosmosac_mixture_datapoint_allows_missing_targets() -> None:
    """Missing targets should produce empty target arrays with valid shapes."""
    datapoint = CosmoSacMixtureDatapoint(
        _cosmo_files(),
        [0.5, 0.5],
        298.15,
    )

    assert datapoint.targets.shape == (0,)
    assert datapoint.num_targets == 0


def test_cosmosac_mixture_datapoint_preserves_target_layout() -> None:
    """Explicit targets should preserve their original 1D and 2D layouts."""
    datapoint = CosmoSacMixtureDatapoint(
        _cosmo_files(),
        [0.25, 0.75],
        298.15,
        targets=[1.2, 3.4],
        model=CosmoSac2002Model,
    )

    assert datapoint.probabilities.shape == (2, 51)
    np.testing.assert_allclose(datapoint.targets, [1.2, 3.4])


def test_mixture_dataset_rejects_empty_mixtures() -> None:
    """Dataset construction should fail for an empty mixture list."""
    with pytest.raises(ValueError, match="at least one mixture"):
        MixtureDataset([], dtype=torch.float32)


def test_mixture_dataset_rejects_shape_mismatch() -> None:
    """Dataset construction should fail when datapoint shapes differ."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(num_targets=1),
        _make_datapoint(num_targets=2),
    ]
    with pytest.raises(ValueError, match="same shape"):
        MixtureDataset(mixtures, dtype=torch.float64)


def test_mixture_dataset_len_matches_number_of_datapoints() -> None:
    """Dataset length should match the number of provided datapoints."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(),
        _make_datapoint(),
    ]
    dataset = MixtureDataset(mixtures, dtype=torch.float32)
    assert len(dataset) == 2


def test_mixture_dataset_getitem_returns_typed_tensors() -> None:
    """Dataset indexing should return input/target tensors with requested dtype."""
    dataset = MixtureDataset([_make_datapoint()], dtype=torch.float64)
    inputs, targets = dataset[0]
    temperature, mole_fractions, areas, volumes, probabilities = inputs

    assert temperature.dtype == torch.float64
    assert mole_fractions.dtype == torch.float64
    assert areas.dtype == torch.float64
    assert volumes.dtype == torch.float64
    assert probabilities.dtype == torch.float64
    assert targets.dtype == torch.float64

    assert temperature.ndim == 0
    assert mole_fractions.shape == (2,)
    assert areas.shape == (2,)
    assert volumes.shape == (2,)
    assert probabilities.shape == (2, 3)
    assert targets.shape == (1,)


def test_cosmosac_mixture_datapoint_rejects_mole_fraction_length_mismatch() -> None:
    """Constructor should fail when mole fractions do not match component count."""
    with pytest.raises(ValueError, match="Invalid array shapes"):
        CosmoSacMixtureDatapoint(
            _cosmo_files(),
            [1.0],
            298.15,
        )
