"""
.. module:: cosmolayer.cosmospace
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd.function import FunctionCtx, NestedIOFunction

from .utils import log_matmul_exp


class CosmoSpace(torch.autograd.Function):
    r"""Implicit COSMOspace layer.

    Solves the following implicit equation for the activity coefficient vector
    :math:`\boldsymbol{\gamma}`, given the segment-type weight vector :math:`\mathbf{x}`
    (nonnegative, not necessarily normalized) and the reduced interaction energy matrix
    :math:`\hat{\mathbf{U}}_T = {(RT)}^{-1}\mathbf{U}`:

    .. math::

        \boldsymbol{\gamma} \odot (\mathbf{B} (\mathbf{x} \odot \boldsymbol{\gamma})) =
            s \mathbf{1},

    where :math:`\mathbf{B} = \exp(\hat{\mathbf{U}}_T)` is the matrix of Boltzmann
    factors and :math:`s = \mathbf{1} \cdot \mathbf{x}` is the sum of the segment-type
    weights. For a physically meaningful solution, :math:`s` must be equal to 1.

    Domain constraint: :math:`\mathbf{x} \geq \mathbf{0}` elementwise, with at least one
    strictly positive component.

    With :math:`s = 1`, the solution satisfies :math:`\min(\boldsymbol{\gamma}) > 0` and
    :math:`\mathbf{a}^T \mathbf{B} \mathbf{a} = 1`, where
    :math:`\mathbf{a} = \mathbf{x} \odot \boldsymbol{\gamma}` is the activity vector.

    Even though :math:`\hat{\mathbf{U}}_T` is usually symmetric, it is not assumed to be
    so.

    .. note::
        Supports batching, meaning that :math:`\mathbf{x}` and
        :math:`\hat{\mathbf{U}}_T` can have broadcastable leading dimensions, and
        all computations are vectorized along these dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Segment-type distribution vector.
        Shape: (..., num_segment_types).
    U_RT : torch.Tensor
        Reduced interaction energy matrix
        :math:`\hat{\mathbf{U}}_T = {(RT)}^{-1}\mathbf{U}`.
        Shape: (..., num_segment_types, num_segment_types).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    gamma : torch.Tensor
        The segment activity coefficient vector.
        Shape: (..., num_segment_types).

    Raises
    ------
    RuntimeError
        If the fixed-point solver does not converge within ``max_iter`` iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from cosmolayer.cosmosac import Component, CosmoSac2002Model
    >>> from importlib.resources import files
    >>> cosmo_strings = [
    ...     (files("cosmolayer.data") / f"{species}.cosmo").read_text()
    ...     for species in ("C=C(N)O", "NCCO")
    ... ]
    >>> probabilities = [
    ...     CosmoSac2002Model.create_component(cosmo_string).get_probabilities()
    ...     for cosmo_string in cosmo_strings
    ... ]
    >>> x = torch.stack(
    ...     [torch.tensor(p, dtype=torch.float32) for p in probabilities],
    ... ).requires_grad_(True)
    >>> U_RT = torch.tensor(
    ...     CosmoSac2002Model.create_interaction_matrices(298.15)[0],
    ...     dtype=torch.float32,
    ...     requires_grad=True,
    ... )
    >>> gamma = CosmoSpace.apply(x, U_RT)
    >>> gamma.log()
    tensor([[-4.7...e+00, -4.0...e+00, ... -1.4056e+01],
            [-2.1...e+01, -1.9...e+01, ... -5.3149e+00]], grad_fn=<LogBackward0>)
    >>> loss = (gamma ** 2).sum()
    >>> loss.backward()
    >>> x.grad
    tensor([[ 6.4...e+04,  1.1...e+04, ... -4.2...e+05],
            [-6.6...e+02, -6.3...e+02, ...  7.4...e+02]])
    """

    @staticmethod
    def _logspace_newton_solver(
        p: torch.Tensor,
        neg_U_RT: torch.Tensor,
        max_iter: int,
    ) -> torch.Tensor:
        tol = 1e-12 if p.dtype == torch.float64 else 1e-6
        with torch.no_grad():
            p = p.unsqueeze(-1)
            log_p = p.log() - p.sum(dim=-2, keepdim=True).log()
            log_gamma = -log_matmul_exp(neg_U_RT, log_p)
            for _ in range(max_iter):
                log_alpha = log_p + log_gamma
                log_B_alpha = log_matmul_exp(neg_U_RT, log_alpha)
                f = log_gamma + log_B_alpha
                J = torch.exp(neg_U_RT + log_alpha.transpose(-2, -1) - log_B_alpha)
                J.diagonal(dim1=-2, dim2=-1).add_(1)
                delta = torch.linalg.solve(J, -f)
                log_gamma += delta
                if delta.abs().max().item() < tol:
                    return log_gamma.exp().squeeze(-1)

            raise RuntimeError(
                f"log-gamma Newton solver did not converge in {max_iter} iterations"
            )

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int = 200,
    ) -> torch.Tensor:
        # Save shapes for correct gradient reductions in backward when broadcasting
        # happened in forward
        ctx_any: Any = ctx
        ctx_any.x_shape = tuple(x.shape)
        ctx_any.u_shape = tuple(U_RT.shape)

        neg_U_RT = -U_RT
        gamma = CosmoSpace._logspace_newton_solver(x, neg_U_RT, max_iter=max_iter)

        # Save tensors needed in backward
        ctx.save_for_backward(gamma, x, neg_U_RT)
        return gamma

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: NestedIOFunction,
        grad_gamma: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if grad_gamma is None:
            return None, None, None

        gamma, x, neg_U_RT = ctx.saved_tensors  # (..., m), (..., m), (..., m, m)
        B = torch.exp(neg_U_RT)

        # t = sum(x) (shape (..., 1))
        t = x.sum(dim=-1, keepdim=True)

        BT = B.transpose(-2, -1)

        # JT = (∂F/∂gamma)^T evaluated at solution:
        # For F(gamma) = gamma ⊙ (B (x ⊙ gamma)) - t*1:
        # JT = diag(t/gamma) + diag(x) B^T diag(gamma)
        JT = x.unsqueeze(-1) * BT * gamma.unsqueeze(-2)  # (..., m, m)
        JT.diagonal(dim1=-2, dim2=-1).add_(t * gamma.reciprocal())  # (..., m)

        # Solve JT v = dL/dgamma
        v = torch.linalg.solve(JT, grad_gamma.unsqueeze(-1)).squeeze(-1)  # (..., m)

        gv = (gamma * v).unsqueeze(-1)  # (..., m, 1)

        # grad_x
        term1 = gamma * (BT @ gv).squeeze(-1)  # (..., m)
        v_sum = v.sum(dim=-1, keepdim=True)  # (..., 1)
        grad_x = -term1 + v_sum  # (..., m)

        # grad_B
        grad_B = -(gv * (x * gamma).unsqueeze(-2))  # (..., m, m)

        # B = exp(-U_RT) => dB/dU_RT = -B
        grad_U_RT = -(B * grad_B)

        # Reduce gradients back to original (possibly broadcasted) input shapes
        ctx_any: Any = ctx
        grad_x = grad_x.sum_to_size(ctx_any.x_shape)
        grad_U_RT = grad_U_RT.sum_to_size(ctx_any.u_shape)

        return grad_x, grad_U_RT, None
