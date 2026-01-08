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
        Reduced interaction energy matrices.
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
    CosmoLayer(t_ref=298.15 K, exponents=[1, 3])
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
        return f"t_ref={ref_temp:.2f} K, exponents={exp}"
