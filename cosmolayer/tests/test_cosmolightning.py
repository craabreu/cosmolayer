"""Unit tests for the CosmoLightningModule wrapper."""

from pathlib import Path
from typing import cast

import pytest
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


class _ScaleTransform(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 10.0 * x


class _AffineTransform(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


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
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    inputs, _ = _make_batch()

    predictions = module.forward(inputs)

    torch.testing.assert_close(predictions, torch.tensor([0.2, 0.6, 1.2]))


def test_configure_optimizers_returns_adam() -> None:
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


def test_training_step_uses_loss_function() -> None:
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    batch = _make_batch()
    module.log = lambda *args, **kwargs: None

    loss = module.training_step(batch, batch_idx=0)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_validation_and_test_steps_run() -> None:
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    batch = _make_batch()
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None

    val_loss = module.validation_step(batch, batch_idx=0)
    test_loss = module.test_step(batch, batch_idx=0)

    torch.testing.assert_close(val_loss, torch.tensor(0.0))
    torch.testing.assert_close(test_loss, torch.tensor(0.0))


def test_forward_applies_module_prediction_head() -> None:
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
        prediction_head=_ScaleTransform(),
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    inputs, _ = _make_batch()

    predictions = module.forward(inputs)

    torch.testing.assert_close(predictions, torch.tensor([2.0, 6.0, 12.0]))


def test_rejects_non_module_prediction_head() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        CosmoLightningModule(
            num_segment_types=4,
            temperature_exponents=(1,),
            area_per_segment=1.0,
            prediction_head=cast(torch.nn.Module, torch.exp),
        )


def test_rejects_unknown_loss_function() -> None:
    with pytest.raises(ValueError, match="Unsupported loss_function 'not_a_loss'."):
        CosmoLightningModule(
            num_segment_types=4,
            temperature_exponents=(1,),
            area_per_segment=1.0,
            loss_function="not_a_loss",
        )


def test_checkpoint_roundtrip_restores_predictions(tmp_path: Path) -> None:
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
        initialization=123,
    )
    inputs, _ = _make_batch()
    expected = module(inputs)

    ckpt_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        ckpt_path,
    )

    loaded = CosmoLightningModule.load_from_checkpoint(str(ckpt_path))
    actual = loaded(inputs)

    torch.testing.assert_close(actual, expected)


def test_checkpoint_roundtrip_restores_prediction_head(tmp_path: Path) -> None:
    module = CosmoLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
        prediction_head=_AffineTransform(scale=3.0),
        initialization=123,
    )
    inputs, _ = _make_batch()
    expected = module(inputs)

    ckpt_path = tmp_path / "model_with_transform.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="requires an prediction_head of class"):
        CosmoLightningModule.load_from_checkpoint(str(ckpt_path))

    with pytest.raises(ValueError, match="Output transform class mismatch"):
        CosmoLightningModule.load_from_checkpoint(
            str(ckpt_path),
            prediction_head=_ScaleTransform(),
        )

    loaded = CosmoLightningModule.load_from_checkpoint(
        str(ckpt_path),
        prediction_head=_AffineTransform(scale=0.0),
    )
    actual = loaded(inputs)

    assert isinstance(loaded.prediction_head, _AffineTransform)
    torch.testing.assert_close(loaded.prediction_head.scale, torch.tensor(3.0))
    torch.testing.assert_close(actual, expected)
