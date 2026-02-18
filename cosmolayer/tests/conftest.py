"""Shared pytest fixtures for the cosmolayer test suite."""

import pytest
import torch

_cuda_available = torch.cuda.is_available()
_skip_if_no_cuda = pytest.mark.skipif(not _cuda_available, reason="CUDA not available")


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[pytest.mark.cuda, _skip_if_no_cuda],
        ),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Fixture that parametrizes tests over CPU and CUDA devices.

    The ``cuda`` parameter carries both the ``cuda`` marker (for ``-m cuda`` /
    ``-m "not cuda"`` filtering) and a ``skipif`` mark so the GPU variant is
    deselected at collection time when no CUDA GPU is present.
    """
    return torch.device(request.param)
