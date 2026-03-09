"""
.. module:: cosmolayer.cosmolightning
   :synopsis: PyTorch Lightning wrapper for CosmoLayer training.
"""

from collections.abc import Callable

import torch
from lightning import pytorch as pl
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
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is ``1e-3``.
    weight_decay : float, optional
        Weight decay for the Adam optimizer. Default is ``0.0``.
    loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        Loss function used in training, validation, and test steps.
        Default is :func:`torch.nn.functional.mse_loss`.

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
    >>> module = CosmoLightningModule(cosmo_layer=cosmo_layer)
    >>> preds = module(datapoint.get_inputs())
    >>> preds.tolist()
    [-0.208..., 0.018...]
    """

    def __init__(
        self,
        cosmo_layer: CosmoLayer,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss_function: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = F.mse_loss,
    ):
        super().__init__()
        self.cosmo_layer = cosmo_layer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.save_hyperparameters(ignore=["cosmo_layer", "loss_function"])

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
        predictions: torch.Tensor = self.cosmo_layer(*inputs)
        return predictions

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
