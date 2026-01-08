"""
.. module:: cosmolayer.cosmolayer
   :synopsis: Differentiable COSMO-type activity coefficient layer.
"""

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

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

    def combinatorial_log_activity_coefficients(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        v: torch.Tensor,
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
          reduced volume vector
        - :math:`\hat{\mathbf a} = \mathbf a / (\mathbf x \cdot \mathbf a)` is the
          reduced area vector
        - :math:`\hat{\mathbf w} = \hat{\mathbf v} \oslash \hat{\mathbf a}` is the
          reduced volume-to-area ratio vector
        - :math:`Z = 10` is the coordination number
        - :math:`a_0 = 79.53` Å² is the reference area per segment
        - :math:`\odot`, :math:`\oslash` denote element-wise operations
        - :math:`\mathbf 1` is the vector of ones

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
        v_hat = v / (x * v).sum(dim=-1, keepdim=True)
        a_hat = a / (x * a).sum(dim=-1, keepdim=True)
        w_hat = v_hat / a_hat
        kappa = cast(torch.Tensor, self.kappa)
        ln_gamma_c = 1 - v_hat + v_hat.log() - kappa * a * (1 - w_hat + w_hat.log())
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
        log_gamma_c = self.combinatorial_log_activity_coefficients(
            mole_fractions, areas, volumes
        )
        return log_gamma_c
