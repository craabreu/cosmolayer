"""
.. module:: cosmolayer.cosmosolver
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd.function import FunctionCtx, NestedIOFunction

from .utils import log_matmul_exp

NEWTON_STEP_TOLERANCE = {torch.float32: 1e-5, torch.float64: 1e-10}
NEWTON_RESIDUAL_TOLERANCE = {torch.float32: 1e-6, torch.float64: 1e-12}


class CosmoSolver(torch.autograd.Function):
    r"""COSMO self-consistent equation solver.

    Solves the COSMO self-consistent equations for the logarithm of the activity
    coefficient vector, :math:`\ln \boldsymbol{\gamma}`, given the nonnegative
    probability distribution vector :math:`\mathbf{p}` and the reduced interaction
    energy matrix :math:`\mathbf{U}/(RT)`.

    The self-consistent equations are:

    .. math::

        \boldsymbol{\gamma}\circ \left(
            \mathbf{B} ({\mathbf p} \circ \boldsymbol{\gamma})
        \right) = t \mathbf{1},

    where :math:`\mathbf{B} = \exp(-\mathbf{U}/(RT))` is the matrix of Boltzmann
    factors, :math:`t=\mathbf{1}^T \mathbf{p}` is the sum of the probabilities, and
    :math:`\circ` represents an elementwise product.

    The solution satisfies
    :math:`\boldsymbol{\gamma}^\mathsf{T} \mathbf{M} \boldsymbol{\gamma} = t`,
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
    log_gamma : torch.Tensor
        The logarithm of the segment activity coefficient vector.
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
    ...     CosmoSac2002Model.create_component(cosmo_string).probabilities
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
    >>> log_gamma, converged = CosmoSolver.apply(p, U_RT)
    >>> converged.all().item()
    True
    >>> log_gamma
    tensor([[-4.5...e+00, -4.0...e+00, ... -1.3...e+01],
            [-2.1...e+01, -1.9...e+01, ... -5.3...e+00]], grad_fn=<CosmoSolverBackward>)
    >>> loss = (2 * log_gamma).exp().sum()
    >>> loss.backward()
    >>> p.grad
    tensor([[ 2.1...e+02,  2.1...e+02, ... -7.4...e+05],
            [-6.6...e+02, -6.3...e+02, ...  7.4...e+02]])
    """

    @staticmethod
    def _logspace_newton_solver(
        p: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_tol = NEWTON_STEP_TOLERANCE[p.dtype]
        resid_tol = NEWTON_RESIDUAL_TOLERANCE[p.dtype]
        with torch.no_grad():
            log_t = p.sum(dim=-1, keepdim=True).log().unsqueeze(-1)
            log_A = p.log().unsqueeze(-2) - U_RT
            Id = torch.eye(log_A.shape[-1], dtype=log_A.dtype, device=log_A.device)
            log_gamma = -torch.logsumexp(log_A, dim=-1, keepdim=True) + 0.5 * log_t
            log_A_gamma = log_matmul_exp(log_A, log_gamma)
            f = log_gamma + log_A_gamma - log_t
            for _ in range(max_iter):
                J = torch.exp(log_gamma.mT + log_A - log_A_gamma) + Id
                delta = torch.linalg.solve(J, -f)
                log_gamma += delta
                log_A_gamma = log_matmul_exp(log_A, log_gamma)
                f = log_gamma + log_A_gamma - log_t
                delta_norm = delta.abs().amax(dim=(-2, -1))
                f_norm = f.abs().amax(dim=(-2, -1))
                converged = (delta_norm < step_tol) & (f_norm < resid_tol)
                if bool(converged.all()):
                    break
            return log_gamma, converged

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        p: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_any: Any = ctx
        ctx_any.p_shape = tuple(p.shape)
        ctx_any.u_shape = tuple(U_RT.shape)

        if max_iter <= 0:
            raise ValueError("Maximum number of iterations must be positive")

        invalid = (p < 0).any() | (p == 0).all(dim=-1).any()
        if bool(invalid):
            raise ValueError("Segment-type probabilities are invalid")

        log_gamma, converged = CosmoSolver._logspace_newton_solver(
            p, U_RT, max_iter=max_iter
        )
        ctx.save_for_backward(log_gamma, p, U_RT)

        return log_gamma.squeeze(-1), converged

    @staticmethod
    def backward(
        ctx: NestedIOFunction,
        grad_log_gamma: torch.Tensor | None,
        grad_converged: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if grad_log_gamma is None:
            return None, None, None

        log_gamma, p, U_RT = ctx.saved_tensors

        gamma = log_gamma.exp()
        B = torch.exp(-U_RT)

        t = p.sum(dim=-1, keepdim=True)

        # Rebuild log_A, log_A_gamma, and J (same as forward)
        log_A = p.log().unsqueeze(-2) - U_RT
        log_A_gamma = log_matmul_exp(log_A, log_gamma)
        Id = torch.eye(log_A.shape[-1], dtype=log_A.dtype, device=log_A.device)
        J = torch.exp(log_gamma.mT + log_A - log_A_gamma) + Id

        # Solve (∂F/∂log_gamma)^T v = dL/dlog_gamma
        v = torch.linalg.solve(J.mT, grad_log_gamma.unsqueeze(-1))

        # r = v / (A @ gamma)
        r = v / log_A_gamma.exp()

        # grad_p: -gamma * (B^T r) + (sum(v)/t)
        grad_p = -(gamma * (B.mT @ r)).squeeze(-1) + v.sum(dim=-2) / t

        # grad_U_RT: r_i * B_ij * (p_j * gamma_j)
        pg = p * gamma.squeeze(-1)
        grad_U_RT = r * B * pg.unsqueeze(-2)

        # Reduce to original shapes if broadcasting happened
        ctx_any: Any = ctx
        grad_p = grad_p.sum_to_size(ctx_any.p_shape)
        grad_U_RT = grad_U_RT.sum_to_size(ctx_any.u_shape)

        return grad_p, grad_U_RT, None
