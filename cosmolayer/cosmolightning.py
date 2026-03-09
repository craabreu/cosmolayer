"""
.. module:: cosmolayer.cosmolightning
   :synopsis: PyTorch Lightning wrapper for CosmoLayer training.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch
from lightning import pytorch as pl
from numpy.typing import NDArray
from torch.nn import functional as F
from torchmetrics.functional import r2_score

from .cosmodata import InputsType, Tensor1D
from .cosmolayer import CosmoLayer


class CosmoLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for training a :class:`~cosmolayer.CosmoLayer`.

    Parameters
    ----------
    cosmo_layer : CosmoLayer
        COSMO layer used to compute predictions.
    output_transform : Callable[[torch.Tensor], torch.Tensor], optional
        Function to transform the output of the layer from the logarithms of the
        activity coefficients to another tensor-valued quantity of interest.
        If ``None`` (default), the logarithms of the activity coefficients are
        returned.
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is ``1e-3``.
    weight_decay : float, optional
        Weight decay for the Adam optimizer. Default is ``0.0``.
    loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        Loss function used in training, validation, and test steps.
        Default is :func:`torch.nn.functional.mse_loss`.
    initialization : Sequence[NDArray[np.float64]] | int, optional

    Examples
    --------
    >>> import torch
    >>> from importlib.resources import files
    >>> import cosmolayer as cl
    >>> from cosmolayer import cosmosac
    >>> model = cosmosac.CosmoSac2010Model
    >>> cosmo_layer = cl.CosmoLayer(
    ...     interaction_matrices=model.create_interaction_matrices(298.15),
    ...     exponents=model.temperature_exponents,
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
    >>> module = CosmoLightningModule(
    ...    num_segment_types=model.num_segment_types,
    ...    temperature_exponents=model.temperature_exponents,
    ...    area_per_segment=model.area_per_segment,
    ... )
    >>> preds = module(datapoint.get_inputs())
    >>> preds.shape
    torch.Size([2])
    """

    def __init__(  # noqa: PLR0913
        self,
        num_segment_types: int,
        temperature_exponents: tuple[int, ...],
        area_per_segment: float,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        reference_temperature: float = 298.15,
        max_iter: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = F.mse_loss,
        initialization: Sequence[NDArray[np.float64]] | int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_function", "initialization"])
        self.num_segment_types = num_segment_types
        self.temperature_exponents = temperature_exponents
        self.area_per_segment = area_per_segment
        self.output_transform = output_transform
        self.reference_temperature = reference_temperature
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function

        initial_matrices: Sequence[NDArray[np.float64]]
        if isinstance(initialization, int):
            rng = np.random.default_rng(initialization)
            initial_matrices = [
                rng.normal(size=(num_segment_types, num_segment_types))
                for _ in range(len(temperature_exponents))
            ]
        else:
            initial_matrices = initialization
        self.cosmo_layer = CosmoLayer(
            interaction_matrices=initial_matrices,
            exponents=temperature_exponents,
            area_per_segment=area_per_segment,
            reference_temperature=reference_temperature,
            max_iter=max_iter,
        )

    def forward(self, inputs: InputsType) -> torch.Tensor:
        """Compute model predictions for one datapoint.

        Parameters
        ----------
        inputs : InputsType
            Input tuple ``(temperature, mole_fractions, areas, volumes,
            probabilities)``.

        Returns
        -------
        torch.Tensor
            Predicted target values.
        """
        log_gamma: torch.Tensor = self.cosmo_layer(*inputs)
        if self.output_transform is None:
            return log_gamma
        else:
            return self.output_transform(log_gamma)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer used during training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer over all module parameters.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def training_step(
        self, batch: tuple[InputsType, Tensor1D], batch_idx: int
    ) -> torch.Tensor:
        """Run one training step and log epoch-level training loss.

        Parameters
        ----------
        batch : tuple[InputsType, Tensor1D]
            Input tensors and ground-truth targets.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss for the batch.
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.loss_function(predictions, targets)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=targets.shape[0],
        )
        return loss

    def validation_step(
        self, batch: tuple[InputsType, Tensor1D], batch_idx: int
    ) -> torch.Tensor:
        """Run one validation step and log epoch-level validation loss.

        Parameters
        ----------
        batch : tuple[InputsType, Tensor1D]
            Input tensors and ground-truth targets.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Validation loss for the batch.
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.loss_function(predictions, targets)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=targets.shape[0],
        )
        return loss

    def test_step(
        self, batch: tuple[InputsType, Tensor1D], batch_idx: int
    ) -> torch.Tensor:
        """Run one test step and log standard regression metrics.

        Parameters
        ----------
        batch : tuple[InputsType, Tensor1D]
            Input tensors and ground-truth targets.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Test loss for the batch.
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.loss_function(predictions, targets)
        self.log_dict(
            {
                "test_loss": loss,
                "test_mae": F.l1_loss(predictions, targets),
                "test_rmse": F.mse_loss(predictions, targets).sqrt(),
                "test_r2": r2_score(predictions, targets),
            },
            on_step=False,
            on_epoch=True,
            batch_size=targets.shape[0],
        )
        return loss
