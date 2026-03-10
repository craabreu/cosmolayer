"""Unit tests for the LogGammaLightningModule wrapper."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

from cosmolayer.cosmodata import InputsType, Tensor1D
from cosmolayer.cosmolayer import CosmoLayer
from cosmolayer.cosmolightning import LogGammaLightningModule


class _DummyTrainer:
    def __init__(
        self,
        train_dataloader: list[tuple[object, torch.Tensor]],
        datamodule: object | None = None,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.datamodule = datamodule


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


class _ScaledLogGammaLightningModule(LogGammaLightningModule):
    def __init__(self, scale: float) -> None:
        super().__init__(
            num_segment_types=4,
            temperature_exponents=(1,),
            area_per_segment=1.0,
        )
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def predict_from_log_gamma(
        self,
        temperature: torch.Tensor,
        mole_fractions: torch.Tensor,
        log_gamma: torch.Tensor,
    ) -> torch.Tensor:
        del temperature, mole_fractions
        return self.scale * log_gamma


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
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    inputs, _ = _make_batch()

    predictions = module.forward(inputs)

    torch.testing.assert_close(predictions, torch.tensor([0.2, 0.6, 1.2]))


def test_configure_optimizers_returns_adam() -> None:
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


def test_training_step_uses_loss_function() -> None:
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    batch = _make_batch()
    module.log = lambda *args, **kwargs: None

    loss = module.training_step(batch, 0)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_validation_and_test_steps_run() -> None:
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    batch = _make_batch()
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None

    val_loss = module.validation_step(batch, 0)
    test_loss = module.test_step(batch, 0)

    torch.testing.assert_close(val_loss, torch.tensor(0.0))
    torch.testing.assert_close(test_loss, torch.tensor(0.0))


def test_test_step_normalizes_loss_when_enabled() -> None:
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
        normalize_targets=True,
    )
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    inputs, _ = _make_batch()
    targets = torch.zeros(3)
    module.target_mean.data = torch.zeros(3)
    module.target_std.data = torch.full((3,), 2.0)
    module.log_dict = lambda *args, **kwargs: None

    test_loss = module.test_step((inputs, targets), 0)

    torch.testing.assert_close(test_loss, torch.tensor(0.15333333333333335))


def test_forward_applies_predict_from_log_gamma_override() -> None:
    module = _ScaledLogGammaLightningModule(scale=10.0)
    module.cosmo_layer = cast(CosmoLayer, _DummyCosmoLayer())
    inputs, _ = _make_batch()

    predictions = module.forward(inputs)

    torch.testing.assert_close(predictions, torch.tensor([2.0, 6.0, 12.0]))


def test_rejects_unknown_loss_function() -> None:
    with pytest.raises(ValueError, match="Unsupported loss_function 'not_a_loss'."):
        LogGammaLightningModule(
            num_segment_types=4,
            temperature_exponents=(1,),
            area_per_segment=1.0,
            loss_function="not_a_loss",
        )


def test_compute_target_statistics_reduces_across_ranks(
    monkeypatch: MonkeyPatch,
) -> None:
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
        normalize_targets=True,
    )
    cast(Any, module)._trainer = _DummyTrainer(
        train_dataloader=[(None, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))]
    )

    calls = {"count": 0}

    def _fake_all_reduce(tensor: torch.Tensor, op: object | None = None) -> None:
        del op
        calls["count"] += 1
        tensor.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)

    module._compute_target_statistics()

    assert calls["count"] == 3
    torch.testing.assert_close(module.target_mean, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(
        module.target_std, torch.sqrt(torch.tensor([1.0, 1.0]) + 1e-8)
    )


def test_compute_target_statistics_rejects_empty_dataloader() -> None:
    module = LogGammaLightningModule(
        num_segment_types=4,
        temperature_exponents=(1,),
        area_per_segment=1.0,
        normalize_targets=True,
    )
    cast(Any, module)._trainer = _DummyTrainer(train_dataloader=[])

    with pytest.raises(
        ValueError, match="Training dataloader is empty; cannot normalize targets"
    ):
        module._compute_target_statistics()


def test_checkpoint_roundtrip_restores_predictions(tmp_path: Path) -> None:
    module = LogGammaLightningModule(
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

    loaded = LogGammaLightningModule.load_from_checkpoint(str(ckpt_path))
    actual = loaded(inputs)

    torch.testing.assert_close(actual, expected)


def test_checkpoint_roundtrip_restores_subclass_parameter(tmp_path: Path) -> None:
    module = _ScaledLogGammaLightningModule(scale=3.0)
    inputs, _ = _make_batch()
    expected = module(inputs)

    ckpt_path = tmp_path / "model_with_override.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        ckpt_path,
    )

    loaded = _ScaledLogGammaLightningModule.load_from_checkpoint(
        str(ckpt_path),
        scale=0.0,
    )
    actual = loaded(inputs)

    assert isinstance(loaded, _ScaledLogGammaLightningModule)
    torch.testing.assert_close(loaded.scale, torch.tensor(3.0))
    torch.testing.assert_close(actual, expected)
