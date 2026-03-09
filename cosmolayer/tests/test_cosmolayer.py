"""Unit and regression tests for the cosmolayer package."""

# Import package, test suite, and other packages as needed
import sys

import numpy as np
import torch

from cosmolayer import CosmoLayer


def test_cosmolayer_imported() -> None:
    """Sample test, will always pass so long as import statement worked"""
    assert "cosmolayer" in sys.modules


def test_cosmolayer_output_transform_applies_to_log_gamma() -> None:
    """Forward should apply output_transform to log-activity coefficients."""
    interaction = np.zeros((2, 2), dtype=np.float64)
    temp = torch.tensor(298.15, dtype=torch.float64)
    fracs = torch.tensor([0.2, 0.8], dtype=torch.float64)
    areas = torch.tensor([1.0, 2.0], dtype=torch.float64)
    volumes = torch.tensor([1.5, 0.8], dtype=torch.float64)
    probs = torch.tensor(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=torch.float64,
    )

    base_layer = CosmoLayer((interaction,), (1,), area_per_segment=1.0)
    transformed_layer = CosmoLayer(
        (interaction,),
        (1,),
        area_per_segment=1.0,
        output_transform=torch.exp,
    )

    log_gamma = base_layer.log_activity_coefficients(temp, fracs, areas, volumes, probs)
    expected = torch.exp(log_gamma)
    actual = transformed_layer(temp, fracs, areas, volumes, probs)

    torch.testing.assert_close(actual, expected)
