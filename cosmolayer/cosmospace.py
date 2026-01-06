"""
.. module:: cosmolayer.cosmospace
   :synopsis: Solves the self-consistent equation for the segment activity coefficients.

.. functionauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import torch


def cosmospace(
    p: torch.Tensor,
    B: torch.Tensor,
    max_iter: int = 1000,
) -> tuple[torch.Tensor, int]:
    r"""Solves the self-consistent equation for the segment activity coefficients.

    Parameters
    ----------
    p : torch.Tensor
        The segment type distribution vector, assumed to satisfy p.sum(dim=-1) = 1.
        Shape: [..., n].
    B : torch.Tensor
        The interaction Boltzmann factor matrix, assummed to be symmetric with positive
        entries. Shape: [..., n, n].
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    lngamma : torch.Tensor
        The entrywise natural logarithm of the segment activity coefficient vector.
        Shape: [..., n].
    iterations : int
        The number of iterations taken to converge.

    Raises
    ------
    RuntimeError
        If the fixed-point iteration does not converge in the maximum number of
        iterations.

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
    >>> lngamma, iterations = cosmospace(P, B)
    >>> 80 < iterations < 90
    True
    >>> lngamma
    tensor([[ -5.2...,  -4.6..., ... -13.3..., -14.4...],
            [-22.4..., -20.7..., ... -4.8...,  -5.5...]])
    """
    tol = 10 * torch.finfo(p.dtype).eps
    p = p.unsqueeze(-1)
    gamma = (B @ p).reciprocal()
    for iterations in range(max_iter):
        z = p * gamma
        Bz = B @ z
        gamma0 = gamma
        gamma = (z * Bz).sum(dim=-2, keepdim=True).sqrt() / Bz
        if ((gamma - gamma0).abs() < tol * gamma).all():
            return gamma.squeeze(-1).log(), iterations + 1
    raise RuntimeError(f"Fixed-point solver did not converge in {max_iter} iterations")
