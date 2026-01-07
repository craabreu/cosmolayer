"""
.. module:: cosmolayer.cosmospace
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import torch


def cosmospace(
    x: torch.Tensor,
    B: torch.Tensor,
    max_iter: int = 1000,
) -> tuple[torch.Tensor, int]:
    r"""Solves the self-consistent equation for the segment activities.

    Parameters
    ----------
    x : torch.Tensor
        The segment type distribution vector, assumed to satisfy x.sum(dim=-1) = 1.
        Shape: (..., n).
    B : torch.Tensor
        The interaction Boltzmann factor matrix, assummed to be symmetric with positive
        entries. Shape: (..., n, n).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    gamma : torch.Tensor
        The segment activity coefficient vector. Shape: (..., n).
    iterations : int
        The number of iterations taken to converge.

    Raises
    ------
    RuntimeError
        If the fixed-point solver does not converge in the maximum number of iterations.

    Examples
    --------
    >>> from cosmolayer import Component, create_cosmo_sac_2002_matrix
    >>> from importlib.resources import files
    >>> components = [
    ...     Component(files("cosmolayer.data") / f"{species}.cosmo")
    ...     for species in ("C=C(N)O", "NCCO")
    ... ]
    >>> distributions = [
    ...     component.get_segment_type_distribution(merged=True)
    ...     for component in components
    ... ]
    >>> P = torch.stack(
    ...     [torch.tensor(p, dtype=torch.float32) for p in distributions],
    ... )
    >>> U_RT = create_cosmo_sac_2002_matrix(298.15)
    >>> B = torch.exp(-torch.tensor(U_RT, dtype=torch.float32))
    >>> Gamma, iterations = cosmospace(P, B)
    >>> 80 < iterations < 90
    True
    >>> Gamma.log()
    tensor([[ -5.2...,  -4.6..., ... -13.3..., -14.4...],
            [-22.4..., -20.7..., ... -4.8...,  -5.5...]])
    >>> [(gamma.T @ (B * p) @ gamma).item() for gamma, p in zip(Gamma, P)]
    [51.000..., 51.000...]
    """
    tol = 10 * torch.finfo(x.dtype).eps
    x = x.unsqueeze(-1)
    gamma = (B @ x).reciprocal()
    for iterations in range(max_iter):
        gamma_prev = gamma
        a = x * gamma
        Ba = B @ a
        gamma = torch.sqrt((a * Ba).sum(dim=-2, keepdim=True)) / Ba
        if ((gamma - gamma_prev) / gamma).abs().max() < tol:
            return gamma.squeeze(-1), iterations + 1
    raise RuntimeError(f"Fixed-point solver did not converge in {max_iter} iterations")


class CosmoSpace(torch.autograd.Function):
    """
    Implicit COSMOspace layer.

    Solves the following implicit equation for the activity coefficient vector γ, given
    the segment-type fraction vector x and the interaction Boltzmann-factor matrix B:

        F(γ; x, B) = γ ⊙ (B a) - 𝟙ₙ = 𝟘ₙ

    where a = x ⊙ γ is the vector of activities.

    The user must make sure that min(B) > 0, min(x) ≥ 0, and xᵀ𝟙ₙ = sum(x) = 1. These
    conditions are assumed to be satisfied, as well as their implications for the
    solutions, namely min(γ) > 0 and aᵀBa = 1.

    Even though B is usually symmetric, there is no assumption about its symmetry.

    Parameters (apply inputs)
    -------------------------
    x : torch.Tensor
        Vector of segment type fractions. Shape (..., n). Must satisfy sum(x) = 1.
    B : torch.Tensor
        Interaction Boltzmann factor matrix. Shape (..., n, n).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    gamma : torch.Tensor
        The segment activity coefficient vector.
        Shape: (..., n).

    Examples
    --------
    >>> B = torch.tensor(
    ...     [
    ...         [1.0, 2.24, 0.45, 0.82, 0.30],
    ...         [2.24, 1.0, 0.67, 3.35, 0.92],
    ...         [0.45, 0.67, 1.0, 0.45, 0.30],
    ...         [0.82, 3.35, 0.45, 1.0, 0.67],
    ...         [0.30, 0.92, 0.30, 0.67, 1.0],
    ...     ],
    ...     dtype=torch.float32,
    ...     requires_grad=True,
    ... )
    >>> p = torch.full((5,), 0.2, dtype=torch.float32, requires_grad=True)
    >>> gamma = CosmoSpace.apply(p, B)
    >>> gamma
    tensor([1.1544, 0.5818, 1.5232, 0.9132, 1.4643], grad_fn=<CosmoSpaceBackward>)
    >>> gamma_ref = torch.tensor(
    ...     [1.2, 0.6, 1.5, 0.9, 1.5],
    ...     dtype=torch.float32,
    ... )
    >>> mse = ((gamma - gamma_ref) ** 2).mean()
    >>> mse.backward()
    >>> p.grad
    tensor([ 0.0160,  0.0175, -0.0088,  0.0161,  0.0273])
    >>> B.grad
    tensor([[ 4.7...e-03,  2.3...e-03,  6.2...e-03,  3.7...e-03,  6.0...e-03],
            [ 9.8...e-05,  4.9...e-05,  1.3...e-04,  7.8...e-05,  1.2...e-04],
            [-4.3...e-03, -2.1...e-03, -5.7...e-03, -3.4...e-03, -5.5...e-03],
            [-1.6...e-03, -8.5...e-04, -2.2...e-03, -1.3...e-03, -2.1...e-03],
            [ 5.2...e-03,  2.6...e-03,  6.8...e-03,  4.1...e-03,  6.6...e-03]])
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
                gamma = torch.sqrt((a * Ba).sum(dim=-2, keepdim=True)) / Ba
                if ((gamma - gamma_prev) / gamma).abs().max() < tol:
                    return gamma.squeeze(-1)
            raise RuntimeError(
                f"Fixed-point solver did not converge in {max_iter} iterations"
            )

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        B: torch.Tensor,
        max_iter: int = 1000,
    ) -> torch.Tensor:
        gamma = CosmoSpace._fixed_point_solver(x, B, max_iter)
        ctx.save_for_backward(gamma, B, x)
        return gamma

    @staticmethod
    def backward(
        ctx,
        grad_gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        gamma, B, x = ctx.saved_tensors
        JT = x.unsqueeze(-1) * B * gamma.unsqueeze(-2)
        JT.diagonal(dim1=-2, dim2=-1).add_(gamma.reciprocal())
        v = torch.linalg.solve(JT, grad_gamma.unsqueeze(-1)).squeeze(-1)
        gv = (gamma * v).unsqueeze(-1)
        grad_x = -(gamma * (B.transpose(-2, -1) @ gv).squeeze(-1))
        grad_B = -(gv * (x * gamma).unsqueeze(-2))
        return grad_x, grad_B, None
