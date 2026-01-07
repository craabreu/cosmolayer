"""
.. module:: cosmolayer.cosmospace
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import torch
from torch.autograd.function import FunctionCtx, NestedIOFunction


class CosmoSpace(torch.autograd.Function):
    """
    Implicit COSMOspace layer.

    Solves the following implicit equation for the activity coefficient vector γ, given
    the (unnormalized, log-scale) segment-type distribution vector log(p) and the
    reduced interaction energy matrix U/RT:

        γ ⊙ (B (x ⊙ γ)) = 𝟙ₙ,

    where x = softmax(log(p)) is the normalized segment-type distribution vector and
    B = exp(-U/RT) is the Boltzmann-factor matrix.

    The solution satisfies min(γ) > 0 and aᵀBa = 1, where a = x ⊙ γ is the activity
    vector.

    Even though U/RT is usually symmetric, it is not assumed to be so.

    .. note::
        Supports batching, meaning that log(p) and U/RT can have broadcastable leading
        dimensions, and all computations are vectorized along these dimensions.

    Parameters
    ----------
    log_p : torch.Tensor
        Segment-type distribution vector in log-scale. Shape: (..., n).
    U_RT : torch.Tensor
        Reduced interaction energy matrix U/RT. Shape: (..., n, n).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    gamma : torch.Tensor
        The segment activity coefficient vector. Satisfies min(γ) > 0 and aᵀBa = 1,
        where a = x ⊙ γ, with x = softmax(log(p)). Shape: (..., n).

    Raises
    ------
    RuntimeError
        If the fixed-point solver does not converge within ``max_iter`` iterations.

    Examples
    --------
    >>> from cosmolayer import Component, create_cosmo_sac_2002_matrix
    >>> from importlib.resources import files
    >>> components = [
    ...     Component(files("cosmolayer.data") / f"{species}.cosmo")
    ...     for species in ("C=C(N)O", "NCCO")
    ... ]
    >>> log_distributions = [
    ...     component.get_log_probabilities(merge=True)
    ...     for component in components
    ... ]
    >>> log_p = torch.stack(
    ...     [torch.tensor(log_p, dtype=torch.float32) for log_p in log_distributions],
    ... ).requires_grad_(True)
    >>> U_RT = torch.tensor(
    ...     create_cosmo_sac_2002_matrix(298.15),
    ...     dtype=torch.float32,
    ...     requires_grad=True,
    ... )
    >>> gamma = CosmoSpace.apply(log_p, U_RT)
    >>> gamma.log()
    tensor([[ -5.2...,  -4.6...,  ... -13.3..., -14.5...],
            [-22.4..., -20.7...,  ... -4.8...,  -5.5...]], grad_fn=<LogBackward0>)
    >>> loss = (gamma ** 2).sum()
    >>> loss.backward()
    >>> log_p.grad
    tensor([[ 2.7...e-07,  6.0...e-08,  ... -4.8...e-06],
            [-5.1...e-08, -5.0...e-08,  ...  6.3...e-08]])
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
                gamma = s / Ba  # Enforces aᵀBa = 1 at each iteration
                if ((gamma - gamma_prev) / gamma).abs().max().item() < tol:
                    return gamma.squeeze(-1)
            raise RuntimeError(
                f"Fixed-point solver did not converge in {max_iter} iterations"
            )

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        log_p: torch.Tensor,
        U_RT: torch.Tensor,
        max_iter: int = 1000,
    ) -> torch.Tensor:
        """Forward pass of the COSMOspace layer.

        Solves the fixed-point equation for the activity coefficient vector γ.

        Parameters
        ----------
        ctx : FunctionCtx
            Context object for saving tensors needed in backward pass.
        log_p : torch.Tensor
            Log-probabilities of segment types. Shape: (..., n).
        U_RT : torch.Tensor
            Reduced interaction energy matrix. Shape: (..., n, n).
        max_iter : int, optional
            Maximum number of iterations for the fixed-point solver.

        Returns
        -------
        torch.Tensor
            Activity coefficient vector γ. Shape: (..., n).
        """
        x = torch.softmax(log_p, dim=-1)
        B = torch.exp(-U_RT)
        gamma = CosmoSpace._fixed_point_solver(x, B, max_iter)
        ctx.save_for_backward(gamma, x, B)
        return gamma

    @staticmethod
    def backward(
        ctx: NestedIOFunction,
        grad_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Backward pass of the COSMOspace layer.

        Computes gradients with respect to log_p and U_RT using implicit
        differentiation.

        Parameters
        ----------
        ctx : NestedIOFunction
            Context object containing saved tensors from forward pass.
        grad_gamma : torch.Tensor
            Gradient with respect to the output γ. Shape: (..., n).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, None]
            Gradients with respect to log_p, U_RT, and max_iter (None).
        """
        gamma, x, B = ctx.saved_tensors
        BT = B.transpose(-2, -1)
        JT = x.unsqueeze(-1) * BT * gamma.unsqueeze(-2)
        JT.diagonal(dim1=-2, dim2=-1).add_(gamma.reciprocal())
        v = torch.linalg.solve(JT, grad_gamma.unsqueeze(-1)).squeeze(-1)
        gv = (gamma * v).unsqueeze(-1)
        grad_x = -(gamma * (BT @ gv).squeeze(-1))
        grad_B = -(gv * (x * gamma).unsqueeze(-2))
        grad_U_RT = -(B * grad_B)
        grad_log_p = x * (grad_x - (grad_x * x).sum(dim=-1, keepdim=True))
        return grad_log_p, grad_U_RT, None
