"""Test the backward pass of CosmoSpace using torch.autograd.gradcheck."""

import pytest
import torch

from cosmolayer.cosmospace import CosmoSpace


def create_random_problem(
    n: int,
    batch_size: int,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    normalized_x: bool = True,
    symmetric_U_RT: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random problem for testing."""
    torch.manual_seed(seed)
    x = torch.rand(batch_size, n, dtype=dtype)
    if normalized_x:
        x = x / x.sum(dim=-1, keepdim=True)
    # Use x directly as the CosmoSpace input; keep it as a leaf tensor for gradcheck.
    x = x.detach().requires_grad_(True)

    U_RT_raw = torch.rand(batch_size, n, n, dtype=dtype)
    if symmetric_U_RT:
        U_RT = (U_RT_raw + U_RT_raw.transpose(-2, -1)) / 2
    else:
        U_RT = U_RT_raw
    # Make U_RT a leaf tensor for gradcheck stability.
    U_RT = U_RT.detach().requires_grad_(True)
    return x, U_RT


def test_cosmospace_output_shapes() -> None:
    """Test that outputs are not None and have the correct shapes."""
    n = 5
    batch_size = 2
    x, U_RT = create_random_problem(n, batch_size)

    # Forward pass
    gamma: torch.Tensor = CosmoSpace.apply(x, U_RT)
    assert gamma.shape == (batch_size, n)

    # Backward pass
    loss = (gamma**2).sum()
    loss.backward()

    assert x.grad is not None
    assert U_RT.grad is not None
    assert x.grad.shape == x.shape
    assert U_RT.grad.shape == U_RT.shape


@pytest.mark.parametrize("normalized_x", [True, False])
def test_cosmospace_solution(normalized_x: bool) -> None:
    """Test that the solution satisfies the fixed-point equation."""
    n = 10
    batch_size = 3
    x, U_RT = create_random_problem(n, batch_size, normalized_x=normalized_x)

    gamma: torch.Tensor = CosmoSpace.apply(x, U_RT)

    # Verify: gamma = t / (B @ z), where z = x * gamma and t = sum(x)
    z = x * gamma
    t = x.sum(dim=-1, keepdim=True)
    B = torch.exp(-U_RT)
    Bz = (B @ z.unsqueeze(-1)).squeeze(-1)
    gamma_check = t / Bz

    rel_error_gamma = ((gamma - gamma_check) / gamma).abs().max()
    assert rel_error_gamma < 1e-6

    # Verify: s = sqrt(z^T B z) = sum(x)
    s = torch.sqrt((z * Bz).sum(dim=-1, keepdim=True))
    error_s = (s - t).abs().max()
    assert error_s < 1e-6


def test_cosmospace_gradients() -> None:
    """Test that gradients are computed correctly."""
    n = 4
    batch_size = 2
    x, U_RT = create_random_problem(n, batch_size, dtype=torch.float64)

    # Use gradcheck to verify the gradients
    result = torch.autograd.gradcheck(
        CosmoSpace.apply,
        (x, U_RT),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    )
    assert result is True
