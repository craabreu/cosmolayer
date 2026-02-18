"""
Test CosmoLayer methods against a reference implementation of the COSMO-SAC 2002 model.

Reference implementation:
    https://github.com/usnistgov/COSMOSAC/blob/master/COSMO-PurePython.ipynb
"""

import functools
import itertools
from importlib.resources import files
from typing import TypeAlias, TypedDict, cast

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from cosmolayer import CosmoLayer
from cosmolayer.cosmosac.constants import (
    COSMO_SAC_2002_AREA_PER_SEGMENT,
    COSMO_SAC_2002_EXPONENTS,
)

_NUM_POINTS = 3
_RTOL = 1e-5
_ATOL = 1e-7
_REF_TEMP = 298.15
_GAS_CONSTANT = 0.001987  # kcal/(mol·K)


_MixtureType: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class ReferenceData(TypedDict):
    """Reference data structure for test validation."""

    psigma: dict[
        int,  # mix
        dict[
            int,  # comp
            NDArray[np.float64],  # psigma
        ],
    ]
    ln_gamma_mix: dict[
        int,  # mix
        dict[
            int,  # comp
            dict[
                int,  # temp
                NDArray[np.float64],  # ln_gamma_mix
            ],
        ],
    ]
    ln_gamma_pure: dict[
        int,  # mix
        dict[
            int,  # temp
            NDArray[np.float64],  # ln_gamma_pure
        ],
    ]


_ReferenceResultsType: TypeAlias = dict[int, ReferenceData]  # n -> reference data


def assert_close(a: torch.Tensor, b: NDArray[np.float64]) -> None:
    np.testing.assert_allclose(a.numpy(), b, rtol=_RTOL, atol=_ATOL)


def get_psigma_mix(
    x: NDArray[np.float64], probs: NDArray[np.float64], areas: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Get the value of p(sigma) for the mixture
    """
    psigma_mix = sum(
        [
            fraction * prob * area
            for fraction, prob, area in zip(x, probs, areas, strict=True)
        ]
    ) / sum([fraction * area for fraction, area in zip(x, areas, strict=True)])
    return np.array(psigma_mix)


def get_Gamma(
    T: float, psigma: NDArray[np.float64], DELTAW: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Get the value of Γ (capital gamma) for the given sigma profile
    """
    Gamma = np.ones_like(psigma)
    AA = np.exp(-DELTAW / (_GAS_CONSTANT * T)) * psigma
    for _ in range(1000):
        Gammanew = 1 / np.sum(AA * Gamma, axis=1)
        difference = np.abs((Gamma - Gammanew) / Gamma)
        Gamma = (Gammanew + Gamma) / 2
        if np.max(difference) < 1e-8:
            break
        else:
            pass
    return Gamma


def get_ln_Gamma_mix(
    T: float,
    psigma_mix: NDArray[np.float64],
    DELTAW: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Get the value of ln(Γ) for the mixture
    """
    return np.log(get_Gamma(T, psigma_mix, DELTAW))


def get_ln_Gamma_pure(
    T: float,
    probs: NDArray[np.float64],
    DELTAW: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Get the value of ln(Γ) for the pure component
    """
    return np.array([np.log(get_Gamma(T, probs[i], DELTAW)) for i in range(len(probs))])


def reduced_excess_gibbs_energy(
    T: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    cosmo_layer: CosmoLayer,
) -> torch.Tensor:
    """Scalar function for gradcheck: gERT = x^T @ ln_gamma."""
    ln_gamma = cosmo_layer.log_residual_activity_coefficients(T, x, a, p)
    ge_RT: torch.Tensor = (x * ln_gamma).sum()
    return ge_RT


def get_compound_data(smiles: str) -> tuple[pd.DataFrame, float]:
    sigma_file = files("cosmolayer.data") / f"{smiles}.sigma"
    dd = pd.read_csv(
        str(sigma_file),
        sep=r"\s+",
        names=["sigma [e/A^2]", "p(sigma)*A [A^2]"],
        comment="#",
        skiprows=1,
    )
    dd["A"] = dd["p(sigma)*A [A^2]"].sum()
    dd["p(sigma)"] = dd["p(sigma)*A [A^2]"] / dd["A"]
    area = float(dd["p(sigma)*A [A^2]"].sum())
    return dd, area


def get_mixture_data(smiles: list[str]) -> _MixtureType:
    profiles, areas = zip(*[get_compound_data(smi) for smi in smiles], strict=True)
    probabilities = np.stack(
        [profile["p(sigma)"].values for profile in profiles], dtype=np.float64
    )
    return (
        np.array(areas, dtype=np.float64),
        probabilities,
    )


@pytest.fixture
def interaction_matrix() -> NDArray[np.float64]:
    c_hb = 85580.0  # kcal A^4 / mol/e^2
    sigma_hb = 0.0084
    EPS = 3.667  # (LIN AND SANDLER USE A CONSTANT FPOL WHICH YIELDS EPS=3.68)
    EO = 2.395e-4
    FPOL = (EPS - 1.0) / (EPS + 0.5)
    ALPHA = (0.3 * COSMO_SAC_2002_AREA_PER_SEGMENT ** (1.5)) / (EO)
    alpha_prime = FPOL * ALPHA
    sigma_tabulated = np.linspace(-0.025, 0.025, 51)
    sigma_m = np.tile(sigma_tabulated, (len(sigma_tabulated), 1))
    sigma_n = np.tile(np.array(sigma_tabulated, ndmin=2).T, (1, len(sigma_tabulated)))
    sigma_acc = np.tril(sigma_n) + np.triu(sigma_m, 1)
    sigma_don = np.tril(sigma_m) + np.triu(sigma_n, 1)
    delta_w: NDArray[np.float64] = (alpha_prime / 2) * (
        sigma_m + sigma_n
    ) ** 2 + c_hb * np.maximum(0, sigma_acc - sigma_hb) * np.minimum(
        0, sigma_don + sigma_hb
    )
    return delta_w


@pytest.fixture
def mixtures() -> dict[int, list[_MixtureType]]:
    return {
        2: [
            get_mixture_data(["NCCO", "O"]),
            get_mixture_data(["CF", "O"]),
            get_mixture_data(["CF", "NCCO"]),
        ],
        3: [
            get_mixture_data(["NCCO", "CF", "O"]),
            get_mixture_data(["CF", "O", "NCCO"]),
        ],
    }


@pytest.fixture
def compositions() -> dict[int, list[NDArray[np.float64]]]:
    binary_compositions = []
    ternary_compositions = []
    for i in range(_NUM_POINTS):
        j_plus_k = _NUM_POINTS - i
        binary_composition = np.array([i, j_plus_k]) / _NUM_POINTS
        binary_compositions.append(binary_composition)
        for j in range(j_plus_k):
            k = j_plus_k - j
            ternary_composition = np.array([i, j, k]) / _NUM_POINTS
            ternary_compositions.append(ternary_composition)
    return {2: binary_compositions, 3: ternary_compositions}


@pytest.fixture
def temperatures() -> list[float]:
    return [273.15 + i * 100 for i in range(4)]


@pytest.fixture
def reference_results(
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    temperatures: list[float],
    interaction_matrix: NDArray[np.float64],
) -> _ReferenceResultsType:
    """
    Reference results for the test cases.
    """
    results: _ReferenceResultsType = {}
    for n, mixtures_n in mixtures.items():
        results[n] = ReferenceData(psigma={}, ln_gamma_mix={}, ln_gamma_pure={})
        for mix, (areas, probs) in enumerate(mixtures_n):
            results[n]["psigma"][mix] = {}
            results[n]["ln_gamma_mix"][mix] = {}
            results[n]["ln_gamma_pure"][mix] = {}
            for comp, composition in enumerate(compositions[n]):
                psigma = get_psigma_mix(composition, probs, areas)
                results[n]["psigma"][mix][comp] = psigma
                results[n]["ln_gamma_mix"][mix][comp] = {}
                for temp, temperature in enumerate(temperatures):
                    results[n]["ln_gamma_mix"][mix][comp][temp] = get_ln_Gamma_mix(
                        temperature, psigma, interaction_matrix
                    )
            for temp, temperature in enumerate(temperatures):
                results[n]["ln_gamma_pure"][mix][temp] = get_ln_Gamma_pure(
                    temperature, probs, interaction_matrix
                )
    return results


@pytest.fixture
def cosmo_layer(interaction_matrix: NDArray[np.float64]) -> CosmoLayer:
    U_RT = interaction_matrix / (_GAS_CONSTANT * _REF_TEMP)
    return CosmoLayer(
        (U_RT,), COSMO_SAC_2002_EXPONENTS, COSMO_SAC_2002_AREA_PER_SEGMENT
    )


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_single_mixture_single_condition(
    n: int,
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    temperatures: list[float],
    reference_results: _ReferenceResultsType,
    cosmo_layer: CosmoLayer,
) -> None:
    ref = reference_results[n]
    for mix, (areas, probs) in enumerate(mixtures[n]):
        a = torch.as_tensor(areas)
        p = torch.as_tensor(probs)

        scaled_interaction_matrices = [
            cosmo_layer.scaled_interactions(torch.as_tensor(temperature))
            for temperature in temperatures
        ]

        for temp, scaled_interaction in enumerate(scaled_interaction_matrices):
            ln_gamma_pure_ref = ref["ln_gamma_pure"][mix][temp]
            ln_gamma_pure = cosmo_layer.log_pure_segment_activity_coefficients(
                scaled_interaction, p
            )
            assert_close(
                ln_gamma_pure.squeeze(0).squeeze(0),
                ln_gamma_pure_ref,
            )

        for comp, composition in enumerate(compositions[n]):
            x = torch.as_tensor(composition)

            p_mix_ref = ref["psigma"][mix][comp]
            p_mix = cosmo_layer.mixture_probabilities(x, a, p)
            assert_close(p_mix, p_mix_ref)

            for temp, scaled_interactions in enumerate(scaled_interaction_matrices):
                ln_gamma_mix = ref["ln_gamma_mix"][mix][comp][temp]

                ln_gamma_s = cosmo_layer.log_mixture_segment_activity_coefficients(
                    scaled_interactions, x, a, p
                )
                assert_close(ln_gamma_s, ln_gamma_mix)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_multiple_mixtures_multiple_compositions(
    n: int,
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    temperatures: list[float],
    reference_results: _ReferenceResultsType,
    cosmo_layer: CosmoLayer,
) -> None:
    data = [
        (areas, probs, temperature, composition)
        for areas, probs in mixtures[n]
        for composition in compositions[n]
        for temperature in temperatures
    ]
    a, p, T, x = (torch.as_tensor(np.array(x)) for x in zip(*data, strict=True))

    scaled_interactions = cosmo_layer.scaled_interactions(T)

    triples = list(
        itertools.product(
            range(len(mixtures[n])),
            range(len(compositions[n])),
            range(len(temperatures)),
        ),
    )
    ref = reference_results[n]

    p_mix_ref = np.array([ref["psigma"][mix][comp] for mix, comp, _ in triples])
    p_mix = cosmo_layer.mixture_probabilities(x, a, p)
    assert_close(p_mix, p_mix_ref)

    ln_gamma_mix_ref = np.array(
        [ref["ln_gamma_mix"][mix][comp][temp] for mix, comp, temp in triples]
    )
    log_gamma_mix = cosmo_layer.log_mixture_segment_activity_coefficients(
        scaled_interactions, x, a, p
    )
    assert_close(log_gamma_mix, ln_gamma_mix_ref)

    ln_gamma_pure_ref = np.array(
        [ref["ln_gamma_pure"][mix][temp] for mix, _, temp in triples]
    )
    log_gamma_pure = cosmo_layer.log_pure_segment_activity_coefficients(
        scaled_interactions, p
    )
    assert_close(log_gamma_pure, ln_gamma_pure_ref)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_broadcasting(
    n: int,
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    temperatures: list[float],
    reference_results: _ReferenceResultsType,
    cosmo_layer: CosmoLayer,
) -> None:
    num_mixtures = len(mixtures[n])
    num_compositions = len(compositions[n])
    num_temperatures = len(temperatures)

    areas_list, probs_list = zip(*mixtures[n], strict=True)

    a = torch.as_tensor(np.array(areas_list, dtype=np.float64))
    assert a.shape == (num_mixtures, n)

    p = torch.as_tensor(np.stack(probs_list))
    assert p.shape == (num_mixtures, n, 51)

    T = torch.as_tensor(np.array(temperatures, dtype=np.float64))
    assert T.shape == (num_temperatures,)

    scaled_interactions = cosmo_layer.scaled_interactions(T)
    assert scaled_interactions.shape == (num_temperatures, 51, 51)

    T = T.reshape(num_temperatures, 1, 1)
    scaled_interactions = scaled_interactions.reshape(num_temperatures, 1, 1, 51, 51)

    x = torch.as_tensor(np.array(compositions[n], dtype=np.float64))
    assert x.shape == (num_compositions, n)
    x = x.reshape(num_compositions, 1, n)

    ref = reference_results[n]

    all_mix = range(num_mixtures)
    all_comp = range(num_compositions)
    all_temp = range(num_temperatures)

    p_mix_ref = np.array(
        [[ref["psigma"][mix][comp] for mix in all_mix] for comp in all_comp]
    )
    p_mix = cosmo_layer.mixture_probabilities(x, a, p)
    assert p_mix.shape == (num_compositions, num_mixtures, 51)
    assert_close(p_mix, p_mix_ref)

    ln_gamma_mix_ref = np.array(
        [
            [
                [ref["ln_gamma_mix"][mix][comp][temp] for mix in all_mix]
                for comp in all_comp
            ]
            for temp in all_temp
        ]
    )
    log_gamma_mix = cosmo_layer.log_mixture_segment_activity_coefficients(
        scaled_interactions, x, a, p
    )
    assert log_gamma_mix.shape == (num_temperatures, num_compositions, num_mixtures, 51)
    assert_close(log_gamma_mix, ln_gamma_mix_ref)

    ln_gamma_pure_ref = np.array(
        [[[ref["ln_gamma_pure"][mix][temp] for mix in all_mix]] for temp in all_temp]
    )
    log_gamma_pure = cosmo_layer.log_pure_segment_activity_coefficients(
        scaled_interactions, p
    )
    assert log_gamma_pure.shape == (num_temperatures, 1, num_mixtures, n, 51)
    assert_close(log_gamma_pure, ln_gamma_pure_ref)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
@pytest.mark.parametrize("seed", [3445, 90745], ids=["seed0", "seed1"])
def test_composition_and_temperature_differentiation(
    seed: int,
    n: int,
    temperatures: list[float],
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    cosmo_layer: CosmoLayer,
) -> None:
    """Test that activity coefficients backpropagate correctly."""
    # Use double precision for gradcheck
    dtype = torch.float64

    rng = np.random.default_rng(seed)
    mix = rng.integers(len(mixtures[n]))
    comp = rng.integers(len(compositions[n]))
    temp = rng.integers(len(temperatures))

    areas, probs = mixtures[n][mix]
    a = torch.as_tensor(areas, dtype=dtype)
    p = torch.as_tensor(probs, dtype=dtype)

    func = functools.partial(
        reduced_excess_gibbs_energy, a=a, p=p, cosmo_layer=cosmo_layer
    )

    T = torch.as_tensor(temperatures[temp], dtype=dtype).requires_grad_(True)

    x = torch.as_tensor(compositions[n][comp], dtype=dtype).requires_grad_(True)

    # Check that the gradients are computed correctly
    assert torch.autograd.gradcheck(
        func,
        (T, x),
        atol=1e-6,
        rtol=1e-5,
        eps=1e-6,
    )


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
@pytest.mark.parametrize("seed", [3445, 90745], ids=["seed0", "seed1"])
def test_parameter_differentiation(
    seed: int,
    n: int,
    temperatures: list[float],
    mixtures: dict[int, list[_MixtureType]],
    interaction_matrix: NDArray[np.float64],
) -> None:
    """Test that gradients w.r.t. interaction matrix parameters are correct."""
    dtype = torch.float64

    U_RT = interaction_matrix / (_GAS_CONSTANT * _REF_TEMP)
    cosmo_layer = CosmoLayer(
        (U_RT,),
        COSMO_SAC_2002_EXPONENTS,
        COSMO_SAC_2002_AREA_PER_SEGMENT,
        learn_matrices=True,
    )

    rng = np.random.default_rng(seed)
    mix = rng.integers(len(mixtures[n]))
    temp = rng.integers(len(temperatures))

    mixture = mixtures[n][mix]
    areas, probs = mixture
    temperature = temperatures[temp]

    composition = rng.integers(1, 10, size=n)
    composition = composition / composition.sum()

    a = torch.as_tensor(areas, dtype=dtype)
    p = torch.as_tensor(probs, dtype=dtype)
    T = torch.as_tensor(temperature, dtype=dtype)

    U_RT_param = next(cosmo_layer.parameters())
    assert U_RT_param.requires_grad

    x = torch.as_tensor(composition, dtype=dtype)
    cosmo_layer.zero_grad()
    gERT = reduced_excess_gibbs_energy(T, x, a, p, cosmo_layer)
    gERT.backward()

    assert U_RT_param.grad is not None
    analytical_grad = U_RT_param.grad.clone()

    assert torch.isfinite(U_RT_param.grad).all()
    assert (U_RT_param.grad.abs() > 0).any()

    m = U_RT_param.shape[0]
    test_indices = [(i * m) // 5 for i in range(1, 5)]
    eps = 1e-6

    for i, j in itertools.combinations(test_indices, 2):
        original_value = U_RT_param.data[i, j].item()
        U_RT_param.data[i, j] = original_value + eps
        gERT_plus = reduced_excess_gibbs_energy(T, x, a, p, cosmo_layer)
        U_RT_param.data[i, j] = original_value - eps
        gERT_minus = reduced_excess_gibbs_energy(T, x, a, p, cosmo_layer)
        grad = (gERT_plus.item() - gERT_minus.item()) / (2 * eps)
        U_RT_param.data[i, j] = original_value
        abs_error = (analytical_grad[i, j] - grad).abs()
        assert abs_error < 0.01 * abs(grad) or abs_error < 1e-8


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_residual_thermodynamic_consistency(
    n: int,
    temperatures: list[float],
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    cosmo_layer: CosmoLayer,
) -> None:
    """Test that thermodynamic consistency is preserved for residual contribution."""
    # Use double precision for gradcheck
    dtype = torch.float64

    def reduced_excess_gibbs_energy_resid(
        x: torch.Tensor, T: torch.Tensor, a: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        """Scalar function: gERT_r = x^T @ ln_gamma_r."""
        ln_gamma_r = cosmo_layer.log_residual_activity_coefficients(T, x, a, p)
        return (x * ln_gamma_r).sum()

    for areas, probs in mixtures[n]:
        a = torch.as_tensor(areas, dtype=dtype)
        p = torch.as_tensor(probs, dtype=dtype)

        for temperature in temperatures:
            T = torch.as_tensor(temperature, dtype=dtype)

            for composition in compositions[n]:
                x = torch.as_tensor(composition, dtype=dtype).requires_grad_(True)

                gERT = reduced_excess_gibbs_energy_resid(x, T, a, p)
                gERT.backward()
                with torch.no_grad():
                    log_gamma_r = cosmo_layer.log_residual_activity_coefficients(
                        T, x, a, p
                    )
                    x_grad = cast(torch.Tensor, x.grad)
                    np.testing.assert_allclose(
                        log_gamma_r,
                        x_grad + gERT - (x * x_grad).sum(),
                        rtol=_RTOL,
                        atol=_ATOL,
                    )


def test_reduced_energy_matrix(
    interaction_matrix: NDArray[np.float64],
    temperatures: list[float],
    cosmo_layer: CosmoLayer,
) -> None:
    for temperature in temperatures:
        T = torch.as_tensor(temperature)
        U_RT = cosmo_layer.scaled_interactions(T)
        assert U_RT.shape == (51, 51)
        ref = interaction_matrix / (_GAS_CONSTANT * temperature)
        assert_close(U_RT, ref)


def test_reduced_energy_matrix_broadcasting(
    interaction_matrix: NDArray[np.float64],
    temperatures: list[float],
    cosmo_layer: CosmoLayer,
) -> None:
    T_tensor = torch.as_tensor(temperatures)
    U_RT = cosmo_layer.scaled_interactions(T_tensor)
    assert U_RT.shape == (len(temperatures), 51, 51)
    T_array = np.asarray(temperatures)[:, None, None]
    assert_close(U_RT, interaction_matrix / (_GAS_CONSTANT * T_array))
