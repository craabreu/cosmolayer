"""
.. module:: cosmolayer.cosmolayer
   :synopsis: Differentiable COSMO-type activity coefficient layer.
"""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from .cosmospace import CosmoSpace

AREA_PER_CONTACT = 79.53  # Å²
COORDINATION_NUMBER = 10


class CosmoLayer(torch.nn.Module):
    r"""Differentiable COSMO-type activity coefficient layer.

    This class assumes that the interaction energy matrix :math:`\mathbf{U}` can depend
    on the temperature :math:`T` through the following relationship:

    .. math::

        \frac{\mathbf{U}}{RT} = \sum_{n=1}^N \frac{\mathbf{U}_n}{RT^{\alpha_n}},


    where each :math:`\mathbf{U}_n` is a constant interaction energy matrix, and
    :math:`\alpha_n` is a constant exponent.

    To instantiate the class, the user must provide a reference temperature
    :math:`T_{\rm ref}`, a tuple of exponents :math:`(\alpha_1, \ldots, \alpha_N)`, and
    a tuple of scaled interaction energy matrices
    :math:`(\hat{\mathbf{U}}_1, \ldots, \hat{\mathbf{U}}_N)`, defined as:

    .. math::

        \hat{\mathbf{U}}_n = \frac{\mathbf{U}_n}{RT_{\rm ref}^{\alpha_n}}

    Parameters
    ----------
    interaction_matrices : Sequence[NDArray[np.float64]]
        The scaled interaction energy matrices at the reference temperature
        (:math:`\hat{\mathbf{U}}_1, \ldots, \hat{\mathbf{U}}_N`).
        Must be square matrices, all with the same shape.
    exponents : Sequence[int]
        The temperature exponents :math:`(\alpha_1, \ldots, \alpha_N)`. Must have the
        same length as the number of interaction energy matrices.
    area_per_segment : float
        Area of each surface segment.
    reference_temperature : float, optional
        Reference temperature :math:`T_{\rm ref}`. Default is 298.15 K.
    learn_matrices : bool, optional
        Whether to register all scaled interaction energy matrices as trainable
        parameters. Default is False.

    Examples
    --------
    >>> from importlib.resources import files
    >>> from cosmolayer import CosmoLayer
    >>> from cosmolayer.cosmosac import CosmoSac2002Model
    >>> import torch
    >>> T_ref = 298.15  # K
    >>> source = files("cosmolayer.data")
    >>> components = {
    ...     "fluoromethane": (source / "CF.cosmo").read_text(),
    ...     "water": (source / "O.cosmo").read_text(),
    ... }
    >>> mixture = CosmoSac2002Model.create_mixture(components)
    >>> interaction_matrices = mixture.get_interaction_matrices(T_ref)
    >>> exponents = mixture.get_temperature_exponents()
    >>> area_per_segment = mixture.get_area_per_segment()
    >>> cosmo_layer = CosmoLayer(interaction_matrices, exponents, area_per_segment)
    >>> cosmo_layer
    CosmoLayer(
      reference_temperature=298.15
      area_per_segment=7.50
      exponents=(1,)
      num_segment_types=51
    )
    >>> T = torch.tensor(373.15)
    >>> x = torch.tensor([0.5, 0.5], requires_grad=True)
    >>> a = torch.tensor(mixture.get_areas())
    >>> v = torch.tensor(mixture.get_volumes())
    >>> P = torch.tensor(mixture.get_probabilities())
    >>> ln_gamma = cosmo_layer(T, x, a, v, P)
    >>> ln_gamma.tolist()
    [0.805809..., 0.648071...]
    >>> gE_RT = (x * ln_gamma).sum()
    >>> gE_RT.item()
    0.726940...
    >>> gE_RT.backward()
    >>> x.grad.tolist()
    [0.805809..., 0.648071...]
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

        self._matrix_names_and_exponents: list[tuple[str, int]] = []
        for idx, input_matrix in enumerate(interaction_matrices):
            matrix = torch.as_tensor(input_matrix)
            name = f"interaction_matrix_{idx}"
            if learn_matrices:
                self.register_parameter(name, torch.nn.Parameter(matrix))
            else:
                self.register_buffer(name, matrix)
            self._matrix_names_and_exponents.append((name, exponents[idx]))

        self._exponents = tuple(exponents)
        self._ref_temp = reference_temperature
        self._area_per_segment = area_per_segment
        self._kappa = COORDINATION_NUMBER / (2 * AREA_PER_CONTACT)

    def extra_repr(self) -> str:
        """Return a string representation of the CosmoLayer."""
        return (
            f"reference_temperature={self._ref_temp:.2f}\n"
            f"area_per_segment={self._area_per_segment:.2f}\n"
            f"exponents={self._exponents}\n"
            f"num_segment_types={self._n_types}"
        )

    def log_combinatorial_activity_coefficients(
        self,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the logarithms of the combinatorial activity coefficients.

        Parameters
        ----------
        fracs : torch.Tensor
            Mole fractions of the mixture components.
            Must sum to 1. Shape: (..., num_components).
        areas : torch.Tensor
            Surface areas of the mixture components, all in the same units.
            Shape: (..., num_components).
        volumes : torch.Tensor
            Volumes of the mixture components, all in the same units.
            Shape: (..., num_components).

        Returns
        -------
        torch.Tensor
            Logarithms of the combinatorial activity coefficients.
            Shape: (..., num_components).
        """
        v_hat = volumes / (fracs * volumes).sum(dim=-1, keepdim=True)
        a_hat = areas / (fracs * areas).sum(dim=-1, keepdim=True)
        w_hat = v_hat / a_hat
        ln_gamma_c: torch.Tensor = (
            1 - v_hat + v_hat.log() - self._kappa * areas * (1 - w_hat + w_hat.log())
        )
        return ln_gamma_c

    def mixture_probabilities(
        self,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the probabilities of segment types in the mixture.

        Parameters
        ----------
        fracs : torch.Tensor
            Mole fractions of the components. Must sum to 1.
            Shape: (..., num_components).
        areas : torch.Tensor
            Surface areas of the components.
            Shape: (..., num_components).
        probs : torch.Tensor
            Probabilities of segment types per component. Must sum to 1 along the
            segment type dimension.
            Shape: (..., num_components, num_segment_types).

        Returns
        -------
        torch.Tensor
            Probabilities of segment types in the mixture.
            Shape: (..., num_segment_types).
        """
        xa = fracs * areas
        theta = xa / xa.sum(dim=-1, keepdim=True)
        return (theta.unsqueeze(-1) * probs).sum(dim=-2)

    def scaled_interactions(self, temp: torch.Tensor) -> torch.Tensor:
        """Compute the scaled interactions at a given temperature.

        Parameters
        ----------
        temp : torch.Tensor
            Temperature in the same units as the reference temperature.
            Shape: (...,).

        Returns
        -------
        torch.Tensor
            The scaled interactions at the given temperature.
            Shape: (..., num_segment_types, num_segment_types).
        """
        beta = (self._ref_temp / temp).unsqueeze(-1).unsqueeze(-1)
        matrices = [
            getattr(self, name) * beta**exponent
            for name, exponent in self._matrix_names_and_exponents
        ]
        return torch.stack(matrices).sum(dim=0)

    def log_pure_segment_activity_coefficients(
        self,
        scaled_interactions: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log-activity coefficients of segment types in pure compounds.

        Parameters
        ----------
        scaled_interactions : torch.Tensor
            Scaled interaction energy matrix.
            Shape: (..., num_segment_types, num_segment_types).
        probs : torch.Tensor
            Probabilities of segment types per component. Must sum to 1 along the
            segment type dimension.
            Shape: (..., num_components, num_segment_types).

        Returns
        -------
        torch.Tensor
            Log-activity coefficients of segment types in pure compounds.
            Shape: (..., num_components, num_segment_types).
        """
        log_gamma_pure: torch.Tensor = CosmoSpace.apply(
            probs, scaled_interactions.unsqueeze(-3)
        ).log()
        return log_gamma_pure

    def log_mixture_segment_activity_coefficients(
        self,
        scaled_interactions: torch.Tensor,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log-activity coefficients of segment types in the mixture.

        Parameters
        ----------
        scaled_interactions : torch.Tensor
            Scaled interaction energy matrix.
            Shape: (..., num_segment_types, num_segment_types).
        fracs : torch.Tensor
            Mole fractions of the components. Must sum to 1.
            Shape: (..., num_components).
        areas : torch.Tensor
            Surface areas of the components.
            Shape: (..., num_components).
        probs : torch.Tensor
            Probabilities of segment types per component. Must sum to 1 along the
            segment type dimension.
            Shape: (..., num_components, num_segment_types).

        Returns
        -------
        torch.Tensor
            Log-activity coefficients of segment types in the mixture.
            Shape: (..., num_segment_types).
        """
        log_gamma_mix: torch.Tensor = CosmoSpace.apply(
            self.mixture_probabilities(fracs, areas, probs),
            scaled_interactions,
        ).log()
        return log_gamma_mix

    def log_residual_activity_coefficients(
        self,
        temperature: torch.Tensor,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logarithms of the residual activity coefficients.

        Parameters
        ----------
        temperature : torch.Tensor
            Temperature in the same units as the reference temperature.
            Shape: (...,).
        fracs : torch.Tensor
            Mole fractions of the components. Must sum to 1.
            Shape: (..., num_components).
        areas : torch.Tensor
            Surface areas of the components.
            Shape: (..., num_components).
        probs : torch.Tensor
            Probabilities of segment types per component. Must sum to 1 along the
            segment type dimension.
            Shape: (..., num_components, num_segment_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the residual activity coefficients.
            Shape: (..., num_components).
        """
        scaled_interactions = self.scaled_interactions(temperature)
        log_gamma_pure = self.log_pure_segment_activity_coefficients(
            scaled_interactions, probs
        )
        log_gamma_mix = self.log_mixture_segment_activity_coefficients(
            scaled_interactions, fracs, areas, probs
        )
        num_segments = areas / self._area_per_segment
        log_gamma_res: torch.Tensor = num_segments * (
            probs * (log_gamma_mix.unsqueeze(-2) - log_gamma_pure)
        ).sum(dim=-1)
        return log_gamma_res

    def log_activity_coefficients(
        self,
        temperature: torch.Tensor,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the logarithms of the activity coefficients.

        Parameters
        ----------
        temperature : torch.Tensor
            Temperature in the same units as the reference temperature.
            Shape: (...,).
        fracs : torch.Tensor
            Mole fractions of the components. Must sum to 1.
            Shape: (..., num_components).
        areas : torch.Tensor
            Surface areas of the components.
            Shape: (..., num_components).
        volumes : torch.Tensor
            Volumes of the components.
            Shape: (..., num_components).
        probs : torch.Tensor
            Probabilities of segment types per component. Must sum to 1 along the
            segment type dimension.
            Shape: (..., num_components, num_segment_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the activity coefficients.
            Shape: (..., num_components).
        """
        log_gamma_c = self.log_combinatorial_activity_coefficients(
            fracs, areas, volumes
        )
        log_gamma_r = self.log_residual_activity_coefficients(
            temperature, fracs, areas, probs
        )
        return log_gamma_c + log_gamma_r

    def forward(
        self,
        temp: torch.Tensor,
        fracs: torch.Tensor,
        areas: torch.Tensor,
        volumes: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the CosmoLayer.

        Parameters
        ----------
        temp : torch.Tensor
            Temperature in the same units as the reference temperature.
            Shape: (...,).
        fracs : torch.Tensor
            Mole fractions of the components. Must sum to 1.
            Shape: (..., num_components).
        areas : torch.Tensor
            Surface areas of the components.
            Shape: (..., num_components).
        volumes : torch.Tensor
            Volumes of the components.
            Shape: (..., num_components).
        probs : torch.Tensor
            Probabilities of segment types per component. Must sum to 1 along the
            segment type dimension.
            Shape: (..., num_components, num_segment_types).

        Returns
        -------
        torch.Tensor
            Logarithms of the activity coefficients.
            Shape: (..., num_components).
        """
        return self.log_activity_coefficients(temp, fracs, areas, volumes, probs)
