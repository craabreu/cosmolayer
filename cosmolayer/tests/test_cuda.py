"""Tests for CosmoLayer on different devices (CPU and CUDA).

These tests verify that CosmoLayer produces numerically identical results on all
supported devices and that autograd through CosmoSpace is correct on GPU.
"""

from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from cosmolayer import CosmoLayer
from cosmolayer.cosmosac import CosmoSac2002Model
from cosmolayer.cosmosac.constants import (
    COSMO_SAC_2002_AREA_PER_SEGMENT,
    COSMO_SAC_2002_EXPONENTS,
)
from cosmolayer.cosmospace import CosmoSpace

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_REF_TEMP = 298.15
_RTOL = 1e-4
_ATOL = 1e-6


def _load_sigma_data(smiles: str) -> tuple[NDArray[np.float64], float]:
    """Load sigma profile and area from a .sigma file."""
    sigma_file = files("cosmolayer.data") / f"{smiles}.sigma"
    data = pd.read_csv(
        str(sigma_file),
        sep=r"\s+",
        names=["sigma [e/A^2]", "p(sigma)*A [A^2]"],
        comment="#",
        skiprows=1,
    )
    area = float(data["p(sigma)*A [A^2]"].sum())
    probs = (data["p(sigma)*A [A^2]"] / area).values.astype(np.float64)
    return probs, area


@pytest.fixture(scope="module")
def _binary_mixture_data() -> dict[str, Any]:
    """Sigma-profile data for a binary CF/O mixture using the COSMO-SAC 2002 model.

    Uses pre-computed .sigma files (same source as test_cosmolayer_class_methods.py)
    to ensure well-conditioned probability distributions for the fixed-point solver.
    """
    probs_cf, area_cf = _load_sigma_data("CF")
    probs_o, area_o = _load_sigma_data("O")

    areas = np.array([area_cf, area_o], dtype=np.float64)
    probs = np.stack([probs_cf, probs_o])

    (interaction_matrix,) = CosmoSac2002Model.create_interaction_matrices(_REF_TEMP)

    return {
        "interaction_matrix": interaction_matrix,
        "areas": areas,
        "probabilities": probs,
    }


@pytest.fixture
def binary_layer(_binary_mixture_data: dict[str, Any]) -> CosmoLayer:
    """Fresh CosmoLayer instance for each test."""
    u_rt = _binary_mixture_data["interaction_matrix"]
    return CosmoLayer(
        (u_rt,), COSMO_SAC_2002_EXPONENTS, COSMO_SAC_2002_AREA_PER_SEGMENT
    )


def test_cosmo_layer_forward_on_device(
    device: torch.device,
    binary_layer: CosmoLayer,
    _binary_mixture_data: dict[str, Any],
) -> None:
    """CosmoLayer residual forward pass runs correctly on each supported device."""
    layer = binary_layer.to(device)
    T = torch.tensor(_REF_TEMP, dtype=torch.float64, device=device)
    x = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    a = torch.as_tensor(_binary_mixture_data["areas"], device=device)
    P = torch.as_tensor(_binary_mixture_data["probabilities"], device=device)

    ln_gamma_r = layer.log_residual_activity_coefficients(T, x, a, P)

    assert ln_gamma_r.shape == x.shape
    assert not ln_gamma_r.isnan().any()
    assert not ln_gamma_r.isinf().any()


@pytest.mark.cuda
@skip_if_no_cuda
def test_cosmo_layer_cpu_cuda_parity(_binary_mixture_data: dict[str, Any]) -> None:
    """CosmoLayer results on CUDA match CPU results within numerical tolerance."""

    def run(device: torch.device) -> torch.Tensor:
        u_rt = _binary_mixture_data["interaction_matrix"]
        layer = CosmoLayer(
            (u_rt,), COSMO_SAC_2002_EXPONENTS, COSMO_SAC_2002_AREA_PER_SEGMENT
        ).to(device)
        T = torch.tensor(_REF_TEMP, dtype=torch.float64, device=device)
        x = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
        a = torch.as_tensor(_binary_mixture_data["areas"], device=device)
        P = torch.as_tensor(_binary_mixture_data["probabilities"], device=device)
        return layer.log_residual_activity_coefficients(T, x, a, P)

    cpu_result = run(torch.device("cpu"))
    cuda_result = run(torch.device("cuda"))

    assert torch.allclose(cpu_result, cuda_result.cpu(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.cuda
@skip_if_no_cuda
def test_cosmospace_gradients_on_cuda() -> None:
    """CosmoSpace autograd is numerically correct on CUDA."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    n, batch_size = 4, 2

    x_raw = torch.rand(batch_size, n, dtype=torch.float64, device=device)
    x = (x_raw / x_raw.sum(dim=-1, keepdim=True)).detach().requires_grad_(True)

    u_raw = torch.rand(batch_size, n, n, dtype=torch.float64, device=device)
    u_rt = ((u_raw + u_raw.transpose(-2, -1)) / 2).detach().requires_grad_(True)

    result = torch.autograd.gradcheck(
        CosmoSpace.apply,
        (x, u_rt),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )
    assert result is True
