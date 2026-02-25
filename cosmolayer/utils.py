"""
.. module:: cosmolayer.utils
   :synopsis: Utility functions for the COSMO-related computations.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

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
