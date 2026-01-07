"""Test the backward pass of CosmoSpace using torch.autograd.gradcheck."""

import torch

from cosmolayer.cosmospace import CosmoSpace


def create_random_problem(
    n: int,
    batch_size: int,
    seed: int = 42,
    normalized_p: bool = True,
    symmetric_B: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random problem for testing."""
    torch.manual_seed(seed)
    p = torch.rand(batch_size, n, dtype=torch.float32)
    if normalized_p:
        p = p / p.sum(dim=-1, keepdim=True)
    p.requires_grad_(True)
    B_raw = torch.rand(batch_size, n, n, dtype=torch.float32)
    if symmetric_B:
        B = (B_raw + B_raw.transpose(-2, -1)) / 2
    else:
        B = B_raw
    B.requires_grad_(True)
    return p, B


def test_cosmospace_output_shapes() -> None:
    """Test that outputs are not None and have the correct shapes."""
    n = 5
    batch_size = 2
    p, B = create_random_problem(n, batch_size)

    # Forward pass
    gamma: torch.Tensor = CosmoSpace.apply(p, B)  # type: ignore[no-untyped-call]
    assert gamma.shape == (batch_size, n)

    # Backward pass
    loss = (gamma**2).sum()
    loss.backward()  # type: ignore[no-untyped-call]

    assert p.grad is not None
    assert B.grad is not None
    assert p.grad.shape == p.shape
    assert B.grad.shape == B.shape


def test_cosmospace_solution() -> None:
    """Test that the solution satisfies the fixed-point equation."""
    n = 10
    batch_size = 3
    p, B = create_random_problem(n, batch_size)

    gamma: torch.Tensor = CosmoSpace.apply(p, B)  # type: ignore[no-untyped-call]

    # Verify: gamma = 1 / (B @ z), where z = p * gamma
    z = p * gamma
    Bz = (B @ z.unsqueeze(-1)).squeeze(-1)
    gamma_check = Bz.reciprocal()

    rel_error_gamma = ((gamma - gamma_check) / gamma).abs().max()
    assert rel_error_gamma < 1e-6

    # Verify: s = sqrt(z^T B z) = 1
    s = torch.sqrt((z * Bz).sum(dim=-1, keepdim=True))
    error_s = (s - 1.0).abs().max()
    assert error_s < 1e-6
