"""
.. module:: cosmolayer.cosmospace
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd.function import FunctionCtx, NestedIOFunction


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
    >>> from cosmolayer.cosmosac import Component, create_cosmo_sac_2002_matrix
    >>> from importlib.resources import files
    >>> components = [
    ...     Component.from_file(files("cosmolayer.data") / f"{species}.cosmo")
    ...     for species in ("C=C(N)O", "NCCO")
    ... ]
    >>> probabilities = [
    ...     component.get_probabilities(merge=True)
    ...     for component in components
    ... ]
    >>> x = torch.stack(
    ...     [torch.tensor(p, dtype=torch.float32) for p in probabilities],
    ... ).requires_grad_(True)
    >>> U_RT = torch.tensor(
    ...     create_cosmo_sac_2002_matrix(298.15),
    ...     dtype=torch.float32,
    ...     requires_grad=True,
    ... )
    >>> gamma = CosmoSpace.apply(x, U_RT)
    >>> gamma.log()
    tensor([[ -5.2...,  -4.6...,  ... -13.3..., -14.5...],
            [-22.4..., -20.7...,  ... -4.8...,  -5.5...]], grad_fn=<LogBackward0>)
    >>> loss = (gamma ** 2).sum()
    >>> loss.backward()
    >>> x.grad
    tensor([[ 2.8...e+03,  6.0...e+02, ... -4.8...e+04],
            [-5.1...e+02, -5.0...e+02, ...  6.3...e+02]])
    """

    @staticmethod
    def _fixed_point_solver(
        x: torch.Tensor, B: torch.Tensor, max_iter: int
    ) -> torch.Tensor:
        tol = 10 * torch.finfo(x.dtype).eps
        with torch.no_grad():
            x = x.unsqueeze(-1)
            gamma = (B @ x).reciprocal()
            for _ in range(max_iter):
                gamma_prev = gamma
                a = x * gamma
                Ba = B @ a
                s = (a * Ba).sum(dim=-2, keepdim=True).sqrt()
                gamma = s / Ba
                if ((gamma - gamma_prev) / gamma).abs().max().item() < tol:
                    return gamma.squeeze(-1)
            raise RuntimeError(
                f"Fixed-point solver did not converge in {max_iter} iterations"
            )

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int = 1000,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Segment-type weights (not necessarily normalized).
            Shape: (..., num_segment_types).
        U_RT : torch.Tensor
            Reduced interaction energy matrix.
            Shape: (..., num_segment_types, num_segment_types).
        max_iter : int
            Maximum iterations for fixed-point solver.

        Returns
        -------
        gamma : torch.Tensor
            Segment activity coefficient vector.
            Shape: (..., num_segment_types).
        """
        # Save shapes for correct gradient reductions when broadcasting happened
        ctx_any: Any = ctx
        ctx_any.x_shape = tuple(x.shape)
        ctx_any.u_shape = tuple(U_RT.shape)

        B = torch.exp(-U_RT)
        gamma = CosmoSpace._fixed_point_solver(x, B, max_iter)

        # Save tensors needed in backward
        ctx.save_for_backward(gamma, x, B)
        return gamma

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: NestedIOFunction,
        grad_gamma: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        r"""
        Parameters
        ----------
        grad_gamma : torch.Tensor
            Gradient of the output scalar function with respect to the segment activity
            coefficient vector.
            Shape: (..., num_segment_types).

        Returns
        -------
        grad_x : torch.Tensor
            Gradient of the output scalar function with respect to the segment-type
            weights.
            Shape: (..., num_segment_types).
        grad_U_RT : torch.Tensor
            Gradient of the output scalar function with respect to the reduced
            interaction energy matrix.
            Shape: (..., num_segment_types, num_segment_types).
        None : NoneType
            Placeholder for the `max_iter` argument, which does not require a gradient.
        """
        if grad_gamma is None:
            return None, None, None

        gamma, x, B = ctx.saved_tensors  # (..., m), (..., m), (..., m, m)

        # t = sum(x) (shape (..., 1))
        t = x.sum(dim=-1, keepdim=True)

        BT = B.transpose(-2, -1)

        # JT = (∂F/∂gamma)^T evaluated at solution:
        # JT = diag(t/gamma) + diag(x) B^T diag(gamma)
        JT = x.unsqueeze(-1) * BT * gamma.unsqueeze(-2)  # (..., m, m)
        JT.diagonal(dim1=-2, dim2=-1).add_(t * gamma.reciprocal())  # (..., m)

        # Solve JT v = dL/dgamma
        v = torch.linalg.solve(JT, grad_gamma.unsqueeze(-1)).squeeze(-1)  # (..., m)

        gv = (gamma * v).unsqueeze(-1)  # (..., m, 1)

        # grad_x:
        # dF/dx has two parts:
        # 1) from Ba term:   gamma_i * B_{i j} * gamma_j  ->  -gamma ⊙ (B^T (gamma ⊙ v))
        # 2) from -sum(x):   -1 for each component -> +sum(v) broadcast
        term1 = gamma * (BT @ gv).squeeze(-1)  # (..., m)
        v_sum = v.sum(dim=-1, keepdim=True)  # (..., 1)
        grad_x = -term1 + v_sum  # (..., m)

        # grad_B:  - (gamma ⊙ v) ⊗ (x ⊙ gamma)
        grad_B = -(gv * (x * gamma).unsqueeze(-2))  # (..., m, m)

        # B = exp(-U_RT) => dB/dU_RT = -B
        grad_U_RT = -(B * grad_B)

        # Reduce gradients back to original (possibly broadcasted) input shapes
        ctx_any: Any = ctx
        grad_x = grad_x.sum_to_size(ctx_any.x_shape)
        grad_U_RT = grad_U_RT.sum_to_size(ctx_any.u_shape)

        return grad_x, grad_U_RT, None
