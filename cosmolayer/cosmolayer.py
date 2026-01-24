"""
.. module:: cosmolayer.cosmolayer
   :synopsis: Differentiable COSMO-type activity coefficient layer.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from .cosmospace import CosmoSpace

AREA_PER_CONTACT = 79.53  # Å²
COORDINATION_NUMBER = 10


class CosmoLayer(torch.nn.Module):
    r"""Differentiable COSMO-type activity coefficient layer.

    The scaled interaction energy matrix at a given temperature is computed as:

    .. math::

        \boldsymbol{\Theta}_T = \frac{\mathbf{U}}{RT} = \sum_{n=1}^{N_m} \left(
            \frac{T_{\rm ref}}{T}
        \right)^{\alpha_n} \boldsymbol{\Theta}_n,

    where :math:`T_{\rm ref}` is the reference temperature,
    :math:`\boldsymbol{\Theta}_n = \mathbf{U}_n/(R T_{\rm ref}^{\alpha_n})` is the
    n-th scaled interaction energy matrix at the reference temperature, and
    :math:`\alpha_n` is the n-th temperature exponent.

    Parameters
    ----------
    interaction_matrices : Sequence[NDArray[np.float64]]
        The scaled interaction energy matrices at the reference temperature
        (:math:`\boldsymbol{\Theta}_1, \ldots, \boldsymbol{\Theta}_{N_m}`).
        Must be square matrices, all with the same shape.
    exponents : Sequence[int]
        Temperature exponents. Must have the same length as the number of interaction
        matrices.
    area_per_segment : float
        Area of each surface segment.
    reference_temperature : float, optional
        Reference temperature. Default is 298.15 K.
    learn_matrices : bool, optional
        Whether to register all scaled interaction energy matrices as trainable
        parameters. Default is False.

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

    Check the thermodynamic consistency (partial molar properties):

    .. math::

        \frac{G^E}{RT} = {\mathbf x} \cdot \ln {\boldsymbol \gamma}
        \quad \Rightarrow \quad
        RT \ln {\boldsymbol \gamma} = \nabla_{\mathbf x}G^E + \left(
            G^E - \mathbf x \cdot \nabla_{\mathbf x}G^E
        \right) {\mathbf 1}

    >>> gERT_c = (x * ln_gamma_c).sum()
    >>> gERT_c.backward()
    >>> x.grad
    tensor([-0.276..., -0.052...])
    >>> (x.grad + gERT_c - (x * x.grad).sum()).tolist()
    [-0.27687..., -0.052266...]
    """

    def __init__(  # noqa: PLR0913
        self,
        interaction_matrices: Sequence[NDArray[np.float64]],
        exponents: Sequence[int],
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

        self._matrices_and_exponents: list[tuple[torch.Tensor, int]] = []
        for idx, input_matrix in enumerate(interaction_matrices):
            matrix = torch.as_tensor(input_matrix)
            name = f"interaction_matrix_{idx}"
            if learn_matrices:
                param = torch.nn.Parameter(matrix)
                self.register_parameter(name, param)
                self._matrices_and_exponents.append((param, exponents[idx]))
            else:
                self.register_buffer(name, matrix)
                self._matrices_and_exponents.append((matrix, exponents[idx]))

        self._exponents = list(exponents)
        self._ref_temp = reference_temperature
        self._area_per_segment = area_per_segment
        self._kappa = COORDINATION_NUMBER / (2 * AREA_PER_CONTACT)

    def extra_repr(self) -> str:
        return (
            f"t_ref={self._ref_temp:.2f}, "
            f"aps={self._area_per_segment:.2f}, "
            f"exponents={self._exponents}, "
            f"n_types={self._n_types}"
        )

    def combinatorial_log_activity_coefficients(
        self,
        molfracs: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the logarithms of the combinatorial activity coefficients.

        This method evaluates the Staverman-Guggenheim model from the mole fractions
        :math:`\mathbf{x}`, component volumes :math:`\mathbf{v}`, and surface areas
        :math:`\mathbf{a}`:

        .. math::

            \ln {\boldsymbol \gamma}_c =
            {\mathbf 1} - \hat{\mathbf v} + \ln \hat{\mathbf v}
            - \frac{Z}{2 a_0} {\mathbf a} \odot \left(
                {\mathbf 1} - \hat{\mathbf w} + \ln \hat{\mathbf w}
            \right),

        where:

        - :math:`\hat{\mathbf v} = \mathbf v / (\mathbf x \cdot \mathbf v)` is the
          scaled volume vector
        - :math:`\hat{\mathbf a} = \mathbf a / (\mathbf x \cdot \mathbf a)` is the
          scaled area vector
        - :math:`\hat{\mathbf w} = \hat{\mathbf v} \oslash \hat{\mathbf a}` is the
          scaled volume-to-area ratio vector
        - :math:`Z = 10` is the coordination number
        - :math:`a_0 = 79.53` Å² is the reference area per segment
        - :math:`\odot`, :math:`\oslash` denote element-wise operations
        - :math:`\mathbf 1` is the vector of ones

        Parameters
        ----------
        molfracs : torch.Tensor
            Mole fractions of the mixture components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the mixture components, all in the same units.
            Shape: (..., n).
        volumes : torch.Tensor
            Volumes of the mixture components, all in the same units. Shape: (..., n).

        Returns
        -------
        torch.Tensor
            Logarithms of the combinatorial activity coefficients. Shape: (..., n).
        """
        v_hat = volumes / (molfracs * volumes).sum(dim=-1, keepdim=True)
        a_hat = areas / (molfracs * areas).sum(dim=-1, keepdim=True)
        w_hat = v_hat / a_hat
        ln_gamma_c = (
            1 - v_hat + v_hat.log() - self._kappa * areas * (1 - w_hat + w_hat.log())
        )
        return cast(torch.Tensor, ln_gamma_c)

    def mixture_log_probabilities(
        self,
        molfracs: torch.Tensor,
        areas: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log-probabilities of segment types in the mixture.

        Parameters
        ----------
        molfracs : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the components. Shape: (..., n).
        logprobs : torch.Tensor
            Log-probabilities of segment types per component. Shape: (..., n, m).

        Returns
        -------
        torch.Tensor
            Log-probabilities of segment types in the mixture. Shape: (..., m).
        """
        xa = molfracs * areas
        log_theta = xa.log() - xa.sum(dim=-1, keepdim=True).log()
        return torch.logsumexp(log_theta.unsqueeze(-1) + logprobs, dim=-2)

    def scaled_interaction_energy_matrix(self, temp: torch.Tensor) -> torch.Tensor:
        """Compute the scaled interaction energy matrix at a given temperature.

        Parameters
        ----------
        temp : torch.Tensor
            Temperature in the same units as the reference temperature. Shape: (...,).

        Returns
        -------
        torch.Tensor
            The scaled interaction energy matrix at the given temperature.
            Shape: (..., m, m).
        """
        beta = (self._ref_temp / temp).unsqueeze(-1).unsqueeze(-1)
        matrices = [
            matrix * beta**exponent for matrix, exponent in self._matrices_and_exponents
        ]
        return torch.stack(matrices).sum(dim=0)

    def log_segment_activity_coefficients(
        self,
        temp: torch.Tensor,
        molfracs: torch.Tensor,
        areas: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the logarithms of the activity coefficients of segment types.

        Parameters
        ----------
        temp : torch.Tensor
            Temperature in the same units as the reference temperature. Shape: (...,).
        molfracs : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the components. Shape: (..., n).
        logprobs : torch.Tensor
            Log-probabilities of segment types per component. Shape: (..., n, m).

        Returns
        -------
        torch.Tensor
            Logarithms of the activity coefficients of segment types in the mixture.
            Shape: (..., m).
        torch.Tensor
            Logarithms of the activity coefficients of segment types in pure compounds.
            Shape: (..., n, m).
        """
        log_p_mix = self.mixture_log_probabilities(molfracs, areas, logprobs)
        U_RT = self.scaled_interaction_energy_matrix(temp)
        gamma_mix = CosmoSpace.apply(log_p_mix, U_RT)

        U_RT_broadcast = U_RT.unsqueeze(-3)
        gamma_pure = CosmoSpace.apply(logprobs, U_RT_broadcast)

        return gamma_mix.log(), gamma_pure.log()

    def forward(
        self,
        temp: torch.Tensor,
        molfracs: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the CosmoLayer.

        Parameters
        ----------
        temp : torch.Tensor
            Temperature in the same units as the reference temperature. Shape: (...,).
        molfracs : torch.Tensor
            Mole fractions of the components. Must sum to 1. Shape: (..., n).
        areas : torch.Tensor
            Surface areas of the components, all in the same units. Shape: (..., n).
        volumes : torch.Tensor
            Volumes of the components, all in the same units. Shape: (..., n).
        logprobs : torch.Tensor
            Log-probabilities of segment types. Shape: (..., num_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the activity coefficients. Shape: (..., n).
        """
        log_gamma_c = self.combinatorial_log_activity_coefficients(
            molfracs, areas, volumes
        )
        return log_gamma_c
