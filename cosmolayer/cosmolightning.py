"""
.. module:: cosmolayer.cosmolightning
   :synopsis: PyTorch Lightning module for batched CosmoLayer training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from lightning import pytorch as pl
from numpy.typing import NDArray
from torch import distributed as td
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from .cosmodata import InputsType
from .cosmolayer import CosmoLayer
from .utils import is_loss_function

EPSILON = 1e-8


class LogGammaLightningModule(pl.LightningModule):
    """PyTorch Lightning module for batched training of a learnable
    :class:`~cosmolayer.CosmoLayer`.

    This class is the canonical high-level training interface for CosmoLayer.
    It constructs an internal :class:`~cosmolayer.CosmoLayer` with learnable
    interaction matrices and defines the optimization, training, validation,
    test, and prediction logic.

    The targets are the log-activity coefficients of the components. In order to
    handle other tasks, the user must subclass :class:`LogGammaLightningModule` and
    override the :meth:`~LogGammaLightningModule.predict_from_log_gamma` method. For
    instance:

    .. code-block:: python

        from scipy.constants import R

        class ExcessGibbsLightningModule(LogGammaLightningModule):
            def predict_from_log_gamma(self, T, x, log_gamma):
                return (R * T * (x * log_gamma).sum(dim=-1)).unsqueeze(-1)

    The module is batch-first throughout. All inputs must represent a minibatch
    of ``b`` datapoints, and the returned predictions must have leading
    dimension ``b``. Targets must have the same shape as the predictions.

    Parameters
    ----------
    num_segment_types : int
        Number of COSMO segment types.
    temperature_exponents : tuple[int, ...]
        Exponents defining the temperature dependence of the interaction
        matrices.
    area_per_segment : float
        Area associated with one segment.
    reference_temperature : float, optional
        Reference temperature used by :class:`CosmoLayer`.
        Default is ``298.15``.
    max_iter : int, optional
        Maximum number of internal fixed-point or iterative solver steps used
        by :class:`CosmoLayer`. Default is ``100``.
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is ``1e-3``.
    weight_decay : float, optional
        Weight decay for the Adam optimizer. Default is ``0.0``.
    loss_function : str, optional
        Loss function used in training, validation, and test steps. Must be a
        valid loss function from :mod:`torch.nn.functional`.
        Default is ``"mse_loss"``.
    initialization : Sequence[NDArray[np.float64]] | int, optional
        Initialization for the learnable interaction matrices.

        - If an ``int`` is provided, it is interpreted as the random seed used
          to sample one matrix per temperature exponent from a standard normal
          distribution.
        - If a sequence of NumPy arrays is provided, it must contain exactly
          one array per temperature exponent, and each array must have shape
          ``(num_segment_types, num_segment_types)``.

        Default is ``42``.

    Examples
    --------
    >>> import torch
    >>> from importlib.resources import files
    >>> import cosmolayer as cl
    >>> from cosmolayer import cosmosac
    >>> model = cosmosac.CosmoSac2010Model
    >>> module = LogGammaLightningModule(
    ...     num_segment_types=model.num_segment_types,
    ...     temperature_exponents=model.temperature_exponents,
    ...     area_per_segment=model.area_per_segment,
    ... )
    >>> solute_path = files("cosmolayer.data") / "NCCO.cosmo"
    >>> solvent_path = files("cosmolayer.data") / "O.cosmo"
    >>> datapoint = cosmosac.CosmoSacMixtureDatapoint(
    ...     cosmo_files=[solute_path, solvent_path],
    ...     mole_fractions=[0.2, 0.8],
    ...     temperature=298.15,
    ...     targets=[-0.2, 0.02],
    ...     model=model,
    ... )
    >>> single_inputs = datapoint.get_inputs()
    >>> batched_inputs = tuple(x.unsqueeze(0) for x in single_inputs)
    >>> preds = module(batched_inputs)
    >>> preds.shape
    torch.Size([1, 2])
    """

    def __init__(  # noqa: PLR0913
        self,
        num_segment_types: int,
        temperature_exponents: Sequence[int],
        area_per_segment: float,
        reference_temperature: float = 298.15,
        max_iter: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        normalize_targets: bool = False,
        loss_function: str = "mse_loss",
        initialization: Sequence[NDArray[np.float64]] | int = 42,
    ) -> None:
        super().__init__()

        if num_segment_types <= 0:
            raise ValueError("num_segment_types must be a positive integer")
        if len(temperature_exponents) == 0:
            raise ValueError("temperature_exponents must not be empty")
        if area_per_segment <= 0.0:
            raise ValueError("area_per_segment must be positive")
        if reference_temperature <= 0.0:
            raise ValueError("reference_temperature must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        loss_callable = getattr(F, loss_function, None)
        if not is_loss_function(loss_callable):
            raise ValueError(f"Unsupported loss_function '{loss_function}'.")

        self.save_hyperparameters(ignore=["initialization"])
        self.normalize_targets = normalize_targets
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_callable

        initial_matrices = self._build_initial_matrices(
            initialization=initialization,
            num_segment_types=num_segment_types,
            num_matrices=len(temperature_exponents),
        )

        self.cosmo_layer = CosmoLayer(
            interaction_matrices=initial_matrices,
            exponents=temperature_exponents,
            area_per_segment=area_per_segment,
            reference_temperature=reference_temperature,
            max_iter=max_iter,
            learn_matrices=True,
        )

        self.test_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_r2 = R2Score()

        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    @staticmethod
    def _build_initial_matrices(
        initialization: Sequence[NDArray[np.float64]] | int,
        num_segment_types: int,
        num_matrices: int,
    ) -> list[NDArray[np.float64]]:
        """Create and validate the initial interaction matrices."""
        if isinstance(initialization, int):
            rng = np.random.default_rng(initialization)
            return [
                rng.normal(size=(num_segment_types, num_segment_types))
                for _ in range(num_matrices)
            ]

        matrices = [np.asarray(matrix, dtype=np.float64) for matrix in initialization]

        if len(matrices) != num_matrices:
            raise ValueError(
                "initialization must contain exactly one matrix per temperature "
                f"exponent: expected {num_matrices}, got {len(matrices)}"
            )

        expected_shape = (num_segment_types, num_segment_types)
        for index, matrix in enumerate(matrices):
            if matrix.shape != expected_shape:
                raise ValueError(
                    "Each initialization matrix must have shape "
                    f"{expected_shape}; matrix {index} has shape {matrix.shape}"
                )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"Initialization matrix {index} contains non-finite values"
                )

        return matrices

    @staticmethod
    def _infer_batch_size(predictions: torch.Tensor, targets: torch.Tensor) -> int:
        """Infer the minibatch size from prediction and target tensors."""
        if predictions.ndim == 0 or targets.ndim == 0:
            raise ValueError(
                "Predictions and targets must be batched tensors with a leading "
                "batch dimension"
            )
        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have the same shape; "
                f"got {predictions.shape} and {targets.shape}"
            )
        return int(targets.shape[0])

    @torch.no_grad()
    def _compute_target_statistics(self) -> None:
        trainer = self.trainer
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            dataloader = trainer.datamodule.train_dataloader()
        else:
            dataloader = trainer.train_dataloader

        if dataloader is None:
            raise ValueError(
                "Training dataloader is unavailable; cannot normalize targets"
            )

        count = torch.tensor(0.0)
        target_sum: torch.Tensor | None = None
        target_sumsq: torch.Tensor | None = None

        for batch in dataloader:
            _, targets = batch
            targets = targets.detach()

            batch_count = torch.tensor(float(targets.shape[0]), device=targets.device)
            batch_sum = targets.sum(dim=0)
            batch_sumsq = (targets**2).sum(dim=0)

            if target_sum is None:
                count = count.to(targets.device)
                target_sum = torch.zeros_like(batch_sum)
                target_sumsq = torch.zeros_like(batch_sumsq)

            count = count + batch_count
            target_sum = target_sum + batch_sum
            target_sumsq = target_sumsq + batch_sumsq

        if target_sum is None or count.item() == 0:
            raise ValueError("Training dataloader is empty; cannot normalize targets")

        if td.is_available() and td.is_initialized():
            td.all_reduce(count, op=td.ReduceOp.SUM)
            td.all_reduce(target_sum, op=td.ReduceOp.SUM)
            td.all_reduce(target_sumsq, op=td.ReduceOp.SUM)

        if target_sum is None or target_sumsq is None:
            raise ValueError("Training dataloader is empty; cannot normalize targets")

        mean = target_sum / count
        variance = torch.clamp(target_sumsq / count - mean**2, min=0.0)
        std = torch.sqrt(variance + EPSILON)

        self.target_mean = mean.to(self.device)
        self.target_std = std.to(self.device)

    def forward(self, inputs: InputsType) -> torch.Tensor:
        """Compute predictions for a minibatch of datapoints.

        Parameters
        ----------
        inputs : InputsType
            Batched input tuple ``(temperature, mole_fractions, areas, volumes,
            probabilities)``. All tensors must be batch-first and represent the
            same minibatch of size ``b``.

        Returns
        -------
        torch.Tensor
            Batched predictions with leading dimension ``b``.
        """
        temperature, mole_fractions, areas, volumes, probabilities = inputs
        log_gamma: torch.Tensor = self.cosmo_layer(
            temperature, mole_fractions, areas, volumes, probabilities
        )
        return self.predict_from_log_gamma(temperature, mole_fractions, log_gamma)

    def predict_from_log_gamma(
        self,
        T: torch.Tensor,
        x: torch.Tensor,
        log_gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Convert log-activity coefficients to final predictions.

        Parameters
        ----------
        T : torch.Tensor
            Temperature in the same units as the reference temperature.
            Shape: (...,).
        x : torch.Tensor
            Mole fractions of the components. Must sum to 1.
            Shape: (..., num_components).
        log_gamma : torch.Tensor
            Logarithms of the activity coefficients.
            Shape: (..., num_components).

        Returns
        -------
        torch.Tensor
            Final predictions.
        """
        return log_gamma

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer used during training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def on_fit_start(self) -> None:
        if self.normalize_targets:
            self._compute_target_statistics()

    def training_step(
        self, batch: tuple[InputsType, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one training step on a minibatch.

        Parameters
        ----------
        batch : tuple[InputsType, torch.Tensor]
            Batched inputs and batched ground-truth targets. Targets must have
            the same shape as the model predictions, with leading dimension
            equal to the minibatch size.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss for the batch.
        """
        inputs, targets = batch
        predictions = self(inputs)
        batch_size = self._infer_batch_size(predictions, targets)
        if self.normalize_targets:
            target_mean = self.target_mean
            target_std = self.target_std
            targets = (targets - target_mean) / target_std
            predictions = (predictions - target_mean) / target_std
        loss: torch.Tensor = self.loss_function(predictions, targets)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(
        self, batch: tuple[InputsType, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one validation step on a minibatch.

        Parameters
        ----------
        batch : tuple[InputsType, torch.Tensor]
            Batched inputs and batched ground-truth targets. Targets must have
            the same shape as the model predictions, with leading dimension
            equal to the minibatch size.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Validation loss for the batch.
        """
        inputs, targets = batch
        predictions = self(inputs)
        batch_size = self._infer_batch_size(predictions, targets)
        if self.normalize_targets:
            target_mean = self.target_mean
            target_std = self.target_std
            targets = (targets - target_mean) / target_std
            predictions = (predictions - target_mean) / target_std
        loss: torch.Tensor = self.loss_function(predictions, targets)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        return loss

    def test_step(
        self, batch: tuple[InputsType, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one test step on a minibatch and update regression metrics.

        Parameters
        ----------
        batch : tuple[InputsType, torch.Tensor]
            Batched inputs and batched ground-truth targets. Targets must have
            the same shape as the model predictions, with leading dimension
            equal to the minibatch size.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Test loss for the batch.
        """
        inputs, targets = batch
        predictions = self(inputs)
        batch_size = self._infer_batch_size(predictions, targets)
        loss_predictions = predictions
        loss_targets = targets
        if self.normalize_targets:
            target_mean = self.target_mean
            target_std = self.target_std
            loss_targets = (targets - target_mean) / target_std
            loss_predictions = (predictions - target_mean) / target_std
        loss: torch.Tensor = self.loss_function(loss_predictions, loss_targets)

        self.test_mae.update(predictions, targets)
        self.test_rmse.update(predictions, targets)
        self.test_r2.update(predictions, targets)

        self.log_dict(
            {
                "test_loss": loss,
                "test_mae": self.test_mae,
                "test_rmse": self.test_rmse,
                "test_r2": self.test_r2,
            },
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
