"""
.. module:: cosmolayer.cosmolayer
   :synopsis: Differentiable COSMO-type activity coefficient layer.
"""

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from cosmolayer.cosmospace import CosmoSpace

AREA_PER_CONTACT = 79.53  # Å²
COORDINATION_NUMBER = 10


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
    interaction_matrices : tuple[NDArray[np.float64], ...]
        Reduced interaction energy matrices. Must be square matrices, all with the same
        shape.
    exponents : tuple[float, ...]
        Temperature exponents. Must be the same length as ``interaction_matrices``.
    area_per_segment : float
        Surface area of one segment.
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
    >>> from cosmolayer.sac import CosmoSac2002Mixture
    >>> import torch
    >>> T_ref = 298.15  # K
    >>> components = {
    ...     "2-aminoethanol": files("cosmolayer.data") / "NCCO.cosmo",
    ...     "water": files("cosmolayer.data") / "O.cosmo",
    ... }
    >>> mixture = CosmoSac2002Mixture(components)
    >>> interaction_matrices = mixture.get_interaction_matrices(T_ref)
    >>> exponents = mixture.get_temperature_exponents()
    >>> area_per_segment = mixture.get_area_per_segment()
    >>> cosmo_layer = CosmoLayer(interaction_matrices, exponents, area_per_segment)
    >>> cosmo_layer
    CosmoLayer(t_ref=298.15, aps=7.50, exponents=[1], n_types=51)
    >>> x = torch.tensor([0.235, 0.765], requires_grad=True)
    >>> a = torch.tensor(mixture.get_areas())
    >>> v = torch.tensor(mixture.get_volumes())
    >>> ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
    >>> ln_gamma_c.tolist()
    [-0.27687..., -0.052266...]
    """

    def __init__(  # noqa: PLR0913
        self,
        interaction_matrices: tuple[NDArray[np.float64], ...],
        exponents: tuple[float, ...],
        area_per_segment: float,
        *,
        reference_temperature: float = 298.15,  # K
        learn_matrices: bool = False,
    ):
        super().__init__()

        num_matrices = len(interaction_matrices)
        if len(exponents) != num_matrices:
            raise ValueError(
                f"Number of exponents ({len(exponents)}) must match "
                f"number of interaction matrices ({num_matrices})"
            )

        self._num_matrices = num_matrices

        shapes = {matrix.shape for matrix in interaction_matrices}
        if len(shapes) != 1:
            raise ValueError("All interaction matrices must have the same shape")
        rows, cols = shapes.pop()
        if rows != cols:
            raise ValueError("Interaction matrices must be square")
        self._n_types = rows

        for idx, input_matrix in enumerate(interaction_matrices, start=1):
            matrix = torch.as_tensor(input_matrix)
            name = f"interaction_matrix_{idx}"
            if learn_matrices:
                param = torch.nn.Parameter(matrix)
                self.register_parameter(name, param)
            else:
                self.register_buffer(name, matrix)

        self.register_buffer(
            "exponents",
            torch.as_tensor(exponents),
        )
        self.register_buffer(
            "reference_temperature",
            torch.as_tensor(reference_temperature),
        )
        self.register_buffer(
            "area_per_segment",
            torch.as_tensor(area_per_segment),
        )
        self.register_buffer(
            "kappa",
            torch.as_tensor(COORDINATION_NUMBER / (2 * AREA_PER_CONTACT)),
        )

    def extra_repr(self) -> str:
        ref_temp = cast(torch.Tensor, self.reference_temperature).item()
        exp = cast(torch.Tensor, self.exponents).tolist()
        aps = cast(torch.Tensor, self.area_per_segment).item()
        return (
            f"t_ref={ref_temp:.2f}, "
            f"aps={aps:.2f}, "
            f"exponents={exp}, "
            f"n_types={self._n_types}"
        )

    def residual_log_activity_coefficients(
        self,
        T: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        log_P: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logarithms of the residual activity coefficients.

        Parameters
        ----------
        T : torch.Tensor
            Temperature in the same units as the reference temperature. Shape: (...,).
        x : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        a : torch.Tensor
            Surface areas of the components, all in the same units. Shape: (..., n).
        log_P : torch.Tensor
            Log-probabilities of segment types. Shape: (..., n, n_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the residual activity coefficients. Shape: (..., n).
        """
        # Compute temperature-dependent U/RT with proper batching
        ref_temp = cast(torch.Tensor, self.reference_temperature).to(T.device)
        beta = ref_temp / T.unsqueeze(-1).unsqueeze(-1)  # Shape: (..., 1, 1)

        # Fix: Accumulate weighted matrices properly
        U_RT = torch.zeros(
            (*beta.shape[:-2], self._n_types, self._n_types),
            dtype=beta.dtype,
            device=beta.device,
        )
        exponents = cast(torch.Tensor, self.exponents)
        for i, exponent in enumerate(exponents):
            matrix = cast(torch.Tensor, getattr(self, f"interaction_matrix_{i + 1}"))
            U_RT = U_RT + matrix * beta.pow(exponent.item())
        # Shape: (..., n_types, n_types)

        P = torch.exp(log_P)  # Shape: (..., n, n_types)
        theta = x * a / (x * a).sum(dim=-1, keepdim=True)  # Shape: (..., n)

        # Compute mixture-level log-probabilities by weighting component
        # probabilities: p_mixture = sum_i (theta_i * P_i)
        # In log space: logsumexp(log_P + log(theta))
        log_theta = theta.log().unsqueeze(-1)  # Shape: (..., n, 1)

        # Compute surface-weighted mixture probabilities
        # p_s = sum_i (theta_i * P_i) weighted by surface areas
        log_p_s = torch.logsumexp(log_P + log_theta, dim=-2)
        # Shape: (..., n_types)

        # Stack mixture-level probabilities to component-level probabilities
        log_P_all = torch.stack([log_P, log_p_s], dim=-2)
        # Shape: (..., n + 1, n_types)

        # Call CosmoSpace on mixture-level probabilities
        log_Gamma_all = CosmoSpace.apply(log_P_all, U_RT)  # type: ignore[no-untyped-call]
        # Shape: (..., n + 1, n_types)

        log_gamma_s = log_Gamma_all[..., -1]  # Shape: (..., n_types)
        log_Gamma = log_Gamma_all[..., :-1]  # Shape: (..., n, n_types)

        # Compute component-level activity coefficients
        area_per_seg = cast(torch.Tensor, self.area_per_segment)
        n = a / area_per_seg  # Shape: (..., n)

        # Compute residual activity coefficients for each component
        # P @ log_gamma_s: (..., n, n_types) @ (..., n_types) -> (..., n)
        log_gamma_s_expanded = log_gamma_s.unsqueeze(-2)  # Shape: (..., 1, n_types)
        P_log_gamma_s = (P * log_gamma_s_expanded).sum(dim=-1)  # Shape: (..., n)

        log_Gamma_expanded = log_Gamma.unsqueeze(-2)  # Shape: (..., 1, n_types)
        P_log_Gamma = (P * log_Gamma_expanded).sum(
            dim=-1, keepdim=True
        )  # Shape: (..., n, 1)

        result = n * (P_log_gamma_s.unsqueeze(-1) - P_log_Gamma).squeeze(-1)
        return cast(torch.Tensor, result)
        # Shape: (..., n)

    def combinatorial_log_activity_coefficients(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the logarithms of the combinatorial activity coefficients.

        This method implements the Staverman-Guggenheim model:

        .. math::

            \ln {\boldsymbol \gamma}_c = {\mathbf 1}
                - \frac{\mathbf v}{v}
                + \ln \frac{\mathbf v}{v}
                + \frac{Z}{2 a_0} \left[
                    a \frac{\mathbf v}{v}
                    + {\mathbf a} \odot \left(
                        \ln \frac{\mathbf a}{a} - \ln \frac{\mathbf v}{v} - {\mathbf 1}
                    \right)
                \right]

        Parameters
        ----------
        x : torch.Tensor
            Mole fractions of the mixture components. Must sum to 1. Shape: (..., n).
        a : torch.Tensor
            Surface areas of the mixture components, all in the same units.
            Shape: (..., n).
        v : torch.Tensor
            Volumes of the mixture components, all in the same units. Shape: (..., n).

        Returns
        -------
        torch.Tensor
            Logarithms of the combinatorial activity coefficients. Shape: (..., n).
        """
        am = (a * x).sum(dim=-1, keepdim=True)
        vm = (v * x).sum(dim=-1, keepdim=True)
        a_am = a / am
        v_vm = v / vm
        log_a_am = a_am.log()
        log_v_vm = v_vm.log()
        kappa = cast(torch.Tensor, self.kappa)
        ln_gamma_c = (
            1 - v_vm + log_v_vm + kappa * (am * v_vm + a * (log_a_am - log_v_vm - 1))
        )
        return cast(torch.Tensor, ln_gamma_c)

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
