"""
.. module:: cosmolayer.utils
   :synopsis: Utility functions for the COSMO-related computations.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import inspect

import torch


def log_matmul_exp(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""Compute :math:`\log(\exp(A) \exp(B))` stably in log-space.

    Parameters
    ----------
    A : torch.Tensor
        Tensor of shape (..., M, K).
    B : torch.Tensor
        Tensor of shape (..., K, N).

    Returns
    -------
    torch.Tensor
        Tensor of shape (..., M, N).
    """
    if A.shape[-1] != B.shape[-2]:
        raise ValueError("Last dimension of A must match second-to-last dimension of B")
    return torch.logsumexp(A.unsqueeze(-1) + B.unsqueeze(-3), dim=-2)


def is_loss_function(func: object) -> bool:
    if not callable(func):
        return False

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    params = list(sig.parameters.values())

    if len(params) < 2:  # noqa: PLR2004
        return False

    return params[0].name == "input" and params[1].name == "target"
