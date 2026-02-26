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

NEWTON_STEP_TOLERANCE = {torch.float32: 1e-5, torch.float64: 1e-10}


class CosmoSpace(torch.autograd.Function):
    r"""Implicit COSMOspace layer.

    Solves the COSMO self-consistent equations for the activity coefficient vector
    :math:`\boldsymbol{\gamma}`, given the nonnegative probability distribution vector
    :math:`\mathbf{p}` and the reduced interaction energy matrix
    :math:`\mathbf{U}/(RT)`:

    .. math::

        \boldsymbol{\gamma}\circ \left(
            \mathbf{B} ({\mathbf p} \circ \boldsymbol{\gamma})
        \right) = t \mathbf{1},

    where :math:`\mathbf{B} = \exp(-\mathbf{U}/(RT))` is the matrix of Boltzmann
    factors, :math:`t=\mathbf{1}^T \mathbf{p}` is the sum of the probabilities, and
    :math:`\circ` represents an elementwise product.

    The solution is strictly positive (:math:`\min(\boldsymbol{\gamma}) > 0`) and
    satisfies :math:`\boldsymbol{\gamma}^\mathsf{T} \mathbf{M} \boldsymbol{\gamma} = t`,
    where :math:`\mathbf{M} = \mathbf{B} \circ (\mathbf{p}\mathbf{p}^T)`.

    .. note::
        Supports batching, i.e., if :math:`\mathbf{p}` and :math:`\mathbf{U}/(RT)`
        can have broadcastable leading dimensions, all computations are performed
        in a single vectorized operation.

    Parameters
    ----------
    p : torch.Tensor
        Segment-type probability distribution vector. Must be nonnegative.
        Shape: (..., num_segment_types).
    U_RT : torch.Tensor
        Reduced interaction energy matrix :math:`\mathbf{U}/(RT)`.
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
    >>> p = torch.stack(
    ...     [torch.tensor(prob, dtype=torch.float32) for prob in probabilities],
    ... ).requires_grad_(True)
    >>> U_RT = torch.tensor(
    ...     CosmoSac2002Model.create_interaction_matrices(298.15)[0],
    ...     dtype=torch.float32,
    ...     requires_grad=True,
    ... )
    >>> gamma = CosmoSpace.apply(p, U_RT)
    >>> gamma.log()
    tensor([[-4.7...e+00, -4.0...e+00, ... -1.4056e+01],
            [-2.1...e+01, -1.9...e+01, ... -5.3149e+00]], grad_fn=<LogBackward0>)
    >>> loss = (gamma ** 2).sum()
    >>> loss.backward()
    >>> p.grad
    tensor([[ 6.4...e+04,  1.1...e+04, ... -4.2...e+05],
            [-6.6...e+02, -6.3...e+02, ...  7.4...e+02]])
    """

    @staticmethod
    def _logspace_newton_solver(
        p: torch.Tensor,  # (..., m)
        U_RT: torch.Tensor,  # (..., m, m)
        max_iter: int,
    ) -> torch.Tensor:
        tol = NEWTON_STEP_TOLERANCE[p.dtype]
        tiny = torch.finfo(p.dtype).tiny
        with torch.no_grad():
            log_t = p.sum(dim=-1, keepdim=True).log().unsqueeze(-1)  # (..., 1, 1)
            log_A = p.clamp_min(tiny).log().unsqueeze(-2) - U_RT  # (..., m, m)
            log_gamma = -torch.logsumexp(log_A, dim=-1, keepdim=True)  # (..., m, 1)
            for _ in range(max_iter):
                log_A_gamma = log_matmul_exp(log_A, log_gamma)  # (..., m, 1)
                f = log_gamma + log_A_gamma - log_t  # (..., m, 1)
                J = torch.exp(log_gamma.mT + log_A - log_A_gamma)  # (..., m, m)
                J.diagonal(dim1=-2, dim2=-1).add_(1)
                delta = torch.linalg.solve(J, -f)  # (..., m, 1)
                log_gamma += delta  # (..., m, 1)
                if delta.abs().max() < tol:
                    return log_gamma.exp().squeeze(-1)  # (..., m)
            raise RuntimeError(
                f"Newton solver did not converge in {max_iter} iterations"
            )

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        p: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int = 200,
    ) -> torch.Tensor:
        # Save shapes for correct gradient reductions in backward when broadcasting
        # happened in forward
        ctx_any: Any = ctx
        ctx_any.p_shape = tuple(p.shape)
        ctx_any.u_shape = tuple(U_RT.shape)

        gamma = CosmoSpace._logspace_newton_solver(p, U_RT, max_iter=max_iter)

        # Save tensors needed in backward
        ctx.save_for_backward(gamma, p, U_RT)
        return gamma

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: NestedIOFunction,
        grad_gamma: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if grad_gamma is None:
            return None, None, None

        gamma, p, U_RT = ctx.saved_tensors  # (..., m), (..., m), (..., m, m)
        B = torch.exp(-U_RT)  # (..., m, m)

        # t = sum(p)
        t = p.sum(dim=-1, keepdim=True)

        # JT = (∂F/∂gamma)^T evaluated at solution:
        # For F(gamma) = gamma ⊙ (B (p ⊙ gamma)) - t*1:
        # JT = diag(t/gamma) + diag(p) B^T diag(gamma)
        JT = p.unsqueeze(-1) * B.mT * gamma.unsqueeze(-2)  # (..., m, m)
        JT.diagonal(dim1=-2, dim2=-1).add_(t * gamma.reciprocal())  # (..., m)

        # Solve JT v = dL/dgamma
        v = torch.linalg.solve(JT, grad_gamma.unsqueeze(-1)).squeeze(-1)  # (..., m)

        gv = (gamma * v).unsqueeze(-1)  # (..., m, 1)

        # grad_p
        term1 = gamma * (B.mT @ gv).squeeze(-1)  # (..., m)
        v_sum = v.sum(dim=-1, keepdim=True)  # (..., 1)
        grad_p = -term1 + v_sum  # (..., m)

        # grad_B
        grad_B = -(gv * (p * gamma).unsqueeze(-2))  # (..., m, m)

        # B = exp(-U_RT) => dB/dU_RT = -B
        grad_U_RT = -(B * grad_B)  # (..., m, m)

        # Reduce gradients back to original (possibly broadcasted) input shapes
        ctx_any: Any = ctx
        grad_p = grad_p.sum_to_size(ctx_any.p_shape)  # (..., m)
        grad_U_RT = grad_U_RT.sum_to_size(ctx_any.u_shape)  # (..., m, m)

        return grad_p, grad_U_RT, None
