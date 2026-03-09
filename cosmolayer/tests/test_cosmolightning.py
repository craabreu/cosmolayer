"""Unit tests for the CosmoLightningModule wrapper."""

from typing import cast

import torch

from cosmolayer.cosmodata import InputsType, Tensor1D
from cosmolayer.cosmolayer import CosmoLayer
from cosmolayer.cosmolightning import CosmoLightningModule


class _DummyCosmoLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        temp: torch.Tensor,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        del temp, areas, volumes, probs
        return fracs * self.scale


def _make_batch() -> tuple[InputsType, Tensor1D]:
    temp = torch.tensor(300.0)
    fracs = torch.tensor([0.1, 0.3, 0.6])
    areas = torch.tensor([1.0, 1.0, 1.0])
    volumes = torch.tensor([1.0, 1.0, 1.0])
    probs = torch.tensor(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    targets = torch.tensor([0.2, 0.6, 1.2])
    return (temp, fracs, areas, volumes, probs), targets


def test_forward_returns_predictions() -> None:
    module = CosmoLightningModule(cosmo_layer=cast(CosmoLayer, _DummyCosmoLayer()))
    inputs, _ = _make_batch()

    predictions = module.forward(inputs)

    torch.testing.assert_close(predictions, torch.tensor([0.2, 0.6, 1.2]))


def test_configure_optimizers_returns_adam() -> None:
    module = CosmoLightningModule(cosmo_layer=cast(CosmoLayer, _DummyCosmoLayer()))
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


def test_training_step_uses_loss_function() -> None:
    module = CosmoLightningModule(cosmo_layer=cast(CosmoLayer, _DummyCosmoLayer()))
    batch = _make_batch()
    module.log = lambda *args, **kwargs: None

    loss = module.training_step(batch, batch_idx=0)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_validation_and_test_steps_run() -> None:
    module = CosmoLightningModule(cosmo_layer=cast(CosmoLayer, _DummyCosmoLayer()))
    batch = _make_batch()
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None

    val_loss = module.validation_step(batch, batch_idx=0)
    test_loss = module.test_step(batch, batch_idx=0)

    torch.testing.assert_close(val_loss, torch.tensor(0.0))
    torch.testing.assert_close(test_loss, torch.tensor(0.0))
