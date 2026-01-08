"""
.. module:: cosmolayer.cosmolayer
   :synopsis: Differentiable COSMO-type activity coefficient layer.
"""

from typing import cast

import torch


class CosmoLayer(torch.nn.Module):
    r"""Differentiable COSMO-type activity coefficient layer.

    The temperature-dependent interaction matrix is computed as:

    .. math::

        \frac{\mathbf{U}}{RT} = \sum_{n=1}^{N_{\rm matrices}} \left(
            \frac{T_{\rm ref}}{T}
        \right)^{\alpha_n} \frac{\mathbf{U}_n}{R T_{\rm ref}^{\alpha_n}},

    where :math:`T_{\rm ref}` is the reference temperature,
    :math:`\mathbf{U}_n/(R T_{\rm ref}^{\alpha_n})` is the n-th reduced
    interaction matrix, and :math:`\alpha_n` is the n-th temperature exponent.

    Parameters
    ----------
    interaction_matrices : tuple[torch.Tensor, ...]
        Reduced interaction energy matrices. Must be square matrices, all with the same
        shape.
    exponents : tuple[float, ...]
        Temperature exponents. Must be the same length as ``interaction_matrices``.
    reference_temperature : float, optional
        Reference temperature. Default is 298.15 K.
    learn_matrices : bool, optional
        Whether to learn all interaction matrices as trainable parameters.
        If True, all matrices are registered as Parameters. If False, all matrices
        are registered as buffers. Default is False.

    Examples
    --------
    >>> from importlib.resources import files
    >>> from cosmolayer import CosmoLayer
    >>> from cosmolayer.sac import CosmoSac2010Mixture
    >>> import torch
    >>> T_ref = 298.15  # K
    >>> components = {
    ...     "1-aminoethenol": files("cosmolayer.data") / "C=C(N)O.cosmo",
    ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
    ... }
    >>> mixture = CosmoSac2010Mixture(components)
    >>> interaction_matrices = mixture.get_interaction_matrices(T_ref)
    >>> exponents = mixture.get_temperature_exponents()
    >>> cosmo_layer = CosmoLayer(interaction_matrices, exponents)
    >>> cosmo_layer
    CosmoLayer(t_ref=298.15, exponents=[1, 3], num_types=153)
    """

    def __init__(
        self,
        interaction_matrices: tuple[torch.Tensor, ...],
        exponents: tuple[float, ...],
        *,
        reference_temperature: float = 298.15,
        learn_matrices: bool = False,
    ):
        super().__init__()

        num_matrices = len(interaction_matrices)
        if len(exponents) != num_matrices:
            raise ValueError(
                f"Number of exponents ({len(exponents)}) must match "
                f"number of interaction matrices ({num_matrices})"
            )

        self._interaction_matrices: list[torch.Tensor] = []

        shapes = {matrix.shape for matrix in interaction_matrices}
        if len(shapes) != 1:
            raise ValueError("All interaction matrices must have the same shape")
        rows, cols = shapes.pop()
        if rows != cols:
            raise ValueError("Interaction matrices must be square")
        self._num_types = rows

        for idx, input_matrix in enumerate(interaction_matrices, start=1):
            matrix = torch.as_tensor(input_matrix)
            name = f"interaction_matrix_{idx}"
            if learn_matrices:
                param = torch.nn.Parameter(matrix)
                self.register_parameter(name, param)
                self._interaction_matrices.append(param)
            else:
                self.register_buffer(name, matrix)
                self._interaction_matrices.append(getattr(self, name))

        self.register_buffer(
            "exponents",
            torch.as_tensor(exponents),
        )
        self.register_buffer(
            "reference_temperature",
            torch.as_tensor(reference_temperature),
        )

    def extra_repr(self) -> str:
        ref_temp = cast(torch.Tensor, self.reference_temperature).item()
        exp = cast(torch.Tensor, self.exponents).tolist()
        return f"t_ref={ref_temp:.2f}, exponents={exp}, num_types={self._num_types}"

    def residual_log_activity_coefficients(
        self,
        temperature: torch.Tensor,
        mole_fractions: torch.Tensor,
        areas: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logarithms of the residual activity coefficients.

        Parameters
        ----------
        temperature : torch.Tensor
            Temperature in the same units as the reference temperature. Shape: (...,).
        mole_fractions : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the components, all in the same units. Shape: (..., n).
        log_p : torch.Tensor
            Log-probabilities of segment types. Shape: (..., num_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the residual activity coefficients. Shape: (..., n).
        """
        raise NotImplementedError("Not implemented")

    def combinatorial_log_activity_coefficients(
        self,
        mole_fractions: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logarithms of the combinatorial activity coefficients.

        Parameters
        ----------
        mole_fractions : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the components, all in the same units. Shape: (..., n).
        volumes : torch.Tensor
            Volumes of the components, all in the same units. Shape: (..., n).

        Returns
        -------
        torch.Tensor
            Logarithms of the combinatorial activity coefficients. Shape: (..., n).
        """
        raise NotImplementedError("Not implemented")

    def forward(
        self,
        temperature: torch.Tensor,
        mole_fractions: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
        log_p: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the CosmoLayer.

        Parameters
        ----------
        temperature : torch.Tensor
            Temperature in the same units as the reference temperature. Shape: (...,).
        mole_fractions : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the components, all in the same units. Shape: (..., n).
        volumes : torch.Tensor
            Volumes of the components, all in the same units. Shape: (..., n).
        log_p : torch.Tensor
            Log-probabilities of segment types. Shape: (..., num_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the activity coefficients. Shape: (..., n).
        """
        log_gamma_r = self.residual_log_activity_coefficients(
            temperature, mole_fractions, areas, log_p
        )
        log_gamma_c = self.combinatorial_log_activity_coefficients(
            mole_fractions, areas, volumes
        )
        return log_gamma_r + log_gamma_c
