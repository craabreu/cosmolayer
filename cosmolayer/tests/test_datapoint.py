"""Tests for mixture datapoint helpers.

Includes MixtureTrainingDataset validation: rejection of empty mixture lists and of
datapoints with differing shapes (num_components, segment types, or targets).
"""

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cosmolayer.cosmodata import (
    MixtureDatapoint,
    MixtureInferenceDataset,
    MixtureTrainingDataset,
)
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
        MixtureTrainingDataset([], dtype=torch.float32)


def test_mixture_dataset_rejects_shape_mismatch() -> None:
    """Dataset construction should fail when datapoint shapes differ."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(num_targets=1),
        _make_datapoint(num_targets=2),
    ]
    with pytest.raises(ValueError, match="same shape"):
        MixtureTrainingDataset(mixtures, dtype=torch.float64)


def test_mixture_dataset_len_matches_number_of_datapoints() -> None:
    """Dataset length should match the number of provided datapoints."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(),
        _make_datapoint(),
    ]
    dataset = MixtureTrainingDataset(mixtures, dtype=torch.float32)
    assert len(dataset) == 2


def test_mixture_dataset_getitem_returns_typed_tensors() -> None:
    """Dataset indexing should return input/target tensors with requested dtype."""
    dataset = MixtureTrainingDataset([_make_datapoint()], dtype=torch.float64)
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


def test_mixture_inference_dataset_rejects_empty_mixtures() -> None:
    """Inference dataset construction should fail for an empty mixture list."""
    with pytest.raises(ValueError, match="at least one mixture"):
        MixtureInferenceDataset([], dtype=torch.float32)


def test_mixture_inference_dataset_rejects_input_shape_mismatch() -> None:
    """Inference dataset construction should fail when input shapes differ."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(num_targets=1),
        MixtureDatapoint(
            temperature=298.15,
            mole_fractions=np.array([1.0]),
            areas=np.array([1.0]),
            volumes=np.array([10.0]),
            probabilities=np.array([[1.0, 0.0, 0.0]]),
            targets=np.array([0.0]),
        ),
    ]
    with pytest.raises(ValueError, match="same input shape"):
        MixtureInferenceDataset(mixtures, dtype=torch.float64)


def test_mixture_inference_dataset_allows_different_target_shapes() -> None:
    """Inference dataset should ignore target dimensions when validating shape."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(num_targets=1),
        _make_datapoint(num_targets=2),
    ]
    dataset = MixtureInferenceDataset(mixtures, dtype=torch.float32)
    assert len(dataset) == 2


def test_mixture_inference_dataset_getitem_returns_typed_inputs() -> None:
    """Inference dataset indexing should return only typed input tensors."""
    dataset = MixtureInferenceDataset([_make_datapoint()], dtype=torch.float64)
    temperature, mole_fractions, areas, volumes, probabilities = dataset[0]

    assert temperature.dtype == torch.float64
    assert mole_fractions.dtype == torch.float64
    assert areas.dtype == torch.float64
    assert volumes.dtype == torch.float64
    assert probabilities.dtype == torch.float64

    assert temperature.ndim == 0
    assert mole_fractions.shape == (2,)
    assert areas.shape == (2,)
    assert volumes.shape == (2,)
    assert probabilities.shape == (2, 3)


def test_to_inference_dataset_returns_inference_dataset() -> None:
    """Training dataset should convert to an inference dataset instance."""
    training = MixtureTrainingDataset([_make_datapoint()], dtype=torch.float32)

    inference = training.to_inference_dataset()

    assert isinstance(inference, MixtureInferenceDataset)
    assert len(inference) == len(training)


def test_to_inference_dataset_preserves_inputs_and_dtype() -> None:
    """Converted inference dataset should preserve tensor values and dtype."""
    mixtures: Sequence[MixtureDatapoint] = [
        _make_datapoint(),
        _make_datapoint(),
    ]
    training = MixtureTrainingDataset(mixtures, dtype=torch.float64)
    inference = training.to_inference_dataset()

    for index in range(len(training)):
        training_inputs, _ = training[index]
        inference_inputs = inference[index]
        for actual, expected in zip(inference_inputs, training_inputs, strict=True):
            assert actual.dtype == torch.float64
            torch.testing.assert_close(actual, expected)


def test_cosmosac_mixture_datapoint_from_series_with_column_keys() -> None:
    """from_series should resolve all values from Series column names."""
    cosmo_a, cosmo_b = _cosmo_files()
    series = pd.Series(
        {
            "file_a": str(cosmo_a),
            "file_b": str(cosmo_b),
            "x_a": 0.25,
            "x_b": 0.75,
            "temperature": 303.15,
            "target_1": 1.2,
            "target_2": 3.4,
        }
    )

    datapoint = CosmoSacMixtureDatapoint.from_series(
        series=series,
        cosmo_files=["file_a", "file_b"],
        mole_fractions=["x_a", "x_b"],
        temperature="temperature",
        targets=["target_1", "target_2"],
        model=CosmoSac2002Model,
    )
    expected = CosmoSacMixtureDatapoint(
        cosmo_files=[cosmo_a, cosmo_b],
        mole_fractions=[0.25, 0.75],
        temperature=303.15,
        targets=[1.2, 3.4],
        model=CosmoSac2002Model,
    )

    assert datapoint.temperature == expected.temperature
    np.testing.assert_allclose(datapoint.mole_fractions, expected.mole_fractions)
    np.testing.assert_allclose(datapoint.areas, expected.areas)
    np.testing.assert_allclose(datapoint.volumes, expected.volumes)
    np.testing.assert_allclose(datapoint.probabilities, expected.probabilities)
    np.testing.assert_allclose(datapoint.targets, expected.targets)


def test_cosmosac_mixture_datapoint_from_series_with_mixed_literals() -> None:
    """from_series should support mixing Series keys with literal inputs."""
    cosmo_a, cosmo_b = _cosmo_files()
    series = pd.Series({"x_a": 0.4, "target_1": 9.0})

    datapoint = CosmoSacMixtureDatapoint.from_series(
        series=series,
        cosmo_files=[cosmo_a, cosmo_b],
        mole_fractions=["x_a", 0.6],
        temperature=298.15,
        targets=["target_1", 7.5],
    )
    expected = CosmoSacMixtureDatapoint(
        cosmo_files=[cosmo_a, cosmo_b],
        mole_fractions=[0.4, 0.6],
        temperature=298.15,
        targets=[9.0, 7.5],
    )

    assert datapoint.temperature == expected.temperature
    np.testing.assert_allclose(datapoint.mole_fractions, expected.mole_fractions)
    np.testing.assert_allclose(datapoint.areas, expected.areas)
    np.testing.assert_allclose(datapoint.volumes, expected.volumes)
    np.testing.assert_allclose(datapoint.probabilities, expected.probabilities)
    np.testing.assert_allclose(datapoint.targets, expected.targets)
