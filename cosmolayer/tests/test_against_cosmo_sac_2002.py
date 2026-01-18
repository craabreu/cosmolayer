"""
Test CosmoLayer against a reference implementation of the COSMO-SAC 2002 model.

Reference implementation:
    https://github.com/usnistgov/COSMOSAC/blob/master/COSMO-PurePython.ipynb
"""

import re
from importlib.resources import files
from typing import TypeAlias, cast

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from cosmolayer import CosmoLayer

_NUM_POINTS = 3
_RTOL = 1e-6

_AEFFPRIME = 7.5
_Q0 = 79.53  # [A^2]
_R0 = 66.69  # [A^3]
_Z_COORDINATION = 10
_R = 8.3144598 / 4184  # 0.001987 # but really: 8.3144598/4184
_REF_TEMP = 298.15


_MixtureType: TypeAlias = tuple[
    int,
    list[pd.DataFrame],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]

_ConditionType: TypeAlias = tuple[float, NDArray[np.float64]]


def get_psigma_mix(
    x: NDArray[np.float64], profs: list[pd.DataFrame]
) -> NDArray[np.float64]:
    """
    Get the value of p(sigma) for the mixture
    """
    As = [prof["p(sigma)*A [A^2]"].sum() for prof in profs]
    psigma_mix = sum(
        [
            fraction * prof["p(sigma)*A [A^2]"].values
            for fraction, prof in zip(x, profs, strict=True)
        ]
    ) / sum([fraction * area for fraction, area in zip(x, As, strict=True)])
    return np.array(psigma_mix)


def get_Gamma(
    T: float, psigma: NDArray[np.float64], DELTAW: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Get the value of Γ (capital gamma) for the given sigma profile
    """
    Gamma = np.ones_like(psigma)
    AA = np.exp(-DELTAW / (_R * T)) * psigma
    for _ in range(1000):
        Gammanew = 1 / np.sum(AA * Gamma, axis=1)
        difference = np.abs((Gamma - Gammanew) / Gamma)
        Gamma = (Gammanew + Gamma) / 2
        if np.max(difference) < 1e-8:
            break
        else:
            pass
    return Gamma


def get_lngamma_resid(
    T: float,
    psigma_mix: NDArray[np.float64],
    prof: pd.DataFrame,
    DELTAW: NDArray[np.float64],
    lnGamma_mix: NDArray[np.float64] | None = None,
) -> float:
    """
    The residual contribution to ln(γ_i)
    """
    # For the mixture
    if lnGamma_mix is None:
        lnGamma_mix = np.log(get_Gamma(T, np.array(psigma_mix), DELTAW))
    # For this component
    psigma = np.array(prof["p(sigma)"])
    A_i = float(prof["A"].iloc[0])
    lnGammai = np.log(get_Gamma(T, psigma, DELTAW))
    lngammai = A_i / _AEFFPRIME * np.sum(psigma * (lnGamma_mix - lnGammai))
    return float(lngammai)


def get_lngamma_comb(
    x: list[float],
    i: int,
    A_COSMO_A2: NDArray[np.float64],
    V_COSMO_A3: NDArray[np.float64],
) -> float:
    """
    The combinatorial part of ln(γ_i)
    """
    q = A_COSMO_A2 / _Q0
    r = V_COSMO_A3 / _R0
    theta_i = q[i] / np.dot(x, q)
    phi_i = r[i] / np.dot(x, r)
    L = _Z_COORDINATION / 2 * (r - q) - (r - 1)
    return float(
        np.log(phi_i)
        + _Z_COORDINATION / 2 * q[i] * np.log(theta_i / phi_i)
        + L[i]
        - phi_i * np.dot(x, L)
    )


def get_lngamma(  # noqa: PLR0913
    T: float,
    x: list[float],
    i: int,
    psigma_mix: NDArray[np.float64],
    profs: list[pd.DataFrame],
    A_COSMO_A2: NDArray[np.float64],
    V_COSMO_A3: NDArray[np.float64],
    DELTAW: NDArray[np.float64],
    lnGamma_mix: NDArray[np.float64] | None = None,
) -> float:
    """
    Sum of the contributions to ln(γ_i)
    """
    return get_lngamma_resid(
        T, psigma_mix, profs[i], DELTAW, lnGamma_mix=lnGamma_mix
    ) + get_lngamma_comb(x, i, A_COSMO_A2, V_COSMO_A3)


def get_compound_data(smiles: str) -> tuple[pd.DataFrame, float, float]:
    sigma_file = files("cosmolayer.data") / f"{smiles}.sigma"
    with open(str(sigma_file)) as f:
        metadata = f.readline()
        volume_match = re.search(r'"volume \[A\^3\]": ([\d.]+)', metadata)
        if volume_match is None:
            raise ValueError(f"Could not find volume in metadata for {smiles}")
        volume = float(volume_match.group(1))
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
    return dd, area, volume


def get_mixture_data(smiles: list[str]) -> _MixtureType:
    profiles, areas, volumes = zip(
        *[get_compound_data(smi) for smi in smiles], strict=True
    )
    probabilities = np.stack(
        [profile["p(sigma)"].values for profile in profiles], dtype=np.float64
    )
    logP = np.full_like(probabilities, -np.inf)
    mask = probabilities > 0.0
    logP[mask] = np.log(probabilities[mask])
    return (
        len(smiles),
        list(profiles),
        np.array(areas, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
        logP,
    )


@pytest.fixture
def interaction_matrix() -> NDArray[np.float64]:
    c_hb = 85580.0  # kcal A^4 / mol/e^2
    sigma_hb = 0.0084
    EPS = 3.667  # (LIN AND SANDLER USE A CONSTANT FPOL WHICH YIELDS EPS=3.68)
    EO = 2.395e-4
    FPOL = (EPS - 1.0) / (EPS + 0.5)
    ALPHA = (0.3 * _AEFFPRIME ** (1.5)) / (EO)
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
def mixtures() -> list[_MixtureType]:
    return [
        get_mixture_data(["NCCO", "O"]),
        get_mixture_data(["CF", "O"]),
        get_mixture_data(["CF", "NCCO"]),
        get_mixture_data(["NCCO", "CF", "O"]),
        get_mixture_data(["CF", "O", "NCCO"]),
    ]


@pytest.fixture
def conditions() -> dict[int, list[_ConditionType]]:
    binary_conditions = []
    ternary_conditions = []
    temperature = 273.15
    for i in range(_NUM_POINTS):
        j_plus_k = _NUM_POINTS - i
        binary_composition = np.array([i, j_plus_k]) / _NUM_POINTS
        binary_conditions.append((temperature, binary_composition))
        for j in range(j_plus_k):
            temperature += 20.0
            k = j_plus_k - j
            ternary_composition = np.array([i, j, k]) / _NUM_POINTS
            ternary_conditions.append((temperature, ternary_composition))
    return {2: binary_conditions, 3: ternary_conditions}


@pytest.fixture
def cosmo_layer(interaction_matrix: NDArray[np.float64]) -> CosmoLayer:
    U_RT = interaction_matrix / (_R * _REF_TEMP)
    return CosmoLayer((U_RT,), (1,), _AEFFPRIME)


def test_single_mixture_single_composition(
    mixtures: list[_MixtureType],
    conditions: dict[int, list[_ConditionType]],
    cosmo_layer: CosmoLayer,
) -> None:
    for n, profiles, areas, volumes, log_probabilities in mixtures:
        for _, composition in conditions[n]:
            ln_gamma_c_ref = np.array(
                [
                    get_lngamma_comb(composition.tolist(), i, areas, volumes)
                    for i in range(n)
                ]
            )
            x = torch.as_tensor(composition)
            a = torch.as_tensor(areas)
            v = torch.as_tensor(volumes)
            logP = torch.as_tensor(log_probabilities)

            ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
            np.testing.assert_allclose(ln_gamma_c.numpy(), ln_gamma_c_ref, rtol=_RTOL)

            p_mix_ref = get_psigma_mix(composition, profiles)
            x = torch.as_tensor(composition)
            log_p_mix = cosmo_layer.mixture_log_probabilities(x, a, logP)
            np.testing.assert_allclose(
                torch.softmax(log_p_mix, dim=-1), p_mix_ref, rtol=_RTOL
            )


def test_single_mixture_multiple_compositions(
    mixtures: list[_MixtureType],
    conditions: dict[int, list[_ConditionType]],
    cosmo_layer: CosmoLayer,
) -> None:
    for n, profiles, areas, volumes, log_probabilities in mixtures:
        _, composition_list = zip(*conditions[n], strict=True)
        composition_batch = np.array(composition_list)
        x = torch.as_tensor(composition_batch)
        a = torch.as_tensor(areas)
        v = torch.as_tensor(volumes)
        logP = torch.as_tensor(log_probabilities)

        num_compositions = len(composition_batch)
        assert x.shape == (num_compositions, n)

        ln_gamma_c_ref = np.array(
            [
                [get_lngamma_comb(comp.tolist(), i, areas, volumes) for i in range(n)]
                for comp in composition_batch
            ]
        )

        ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
        np.testing.assert_allclose(ln_gamma_c.numpy(), ln_gamma_c_ref, rtol=_RTOL)

        p_mix_ref = np.array(
            [get_psigma_mix(comp, profiles) for comp in composition_batch]
        )

        log_p_mix = cosmo_layer.mixture_log_probabilities(x, a, logP)
        p_mix = torch.softmax(log_p_mix, dim=-1)
        np.testing.assert_allclose(p_mix, p_mix_ref, rtol=_RTOL)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_multiple_mixtures_single_composition(
    n: int,
    mixtures: list[_MixtureType],
    conditions: dict[int, list[_ConditionType]],
    cosmo_layer: CosmoLayer,
) -> None:
    mixtures = list(filter(lambda mixture: mixture[0] == n, mixtures))
    num_mixtures = len(mixtures)

    _, composition_list = zip(*conditions[n], strict=True)

    _, profiles_list, areas_list, volumes_list, log_probabilities_list = zip(
        *mixtures, strict=True
    )

    a = torch.as_tensor(np.array(areas_list, dtype=np.float64))
    assert a.shape == (num_mixtures, n)

    v = torch.as_tensor(np.array(volumes_list))
    assert v.shape == (num_mixtures, n)

    logP = torch.as_tensor(np.stack(log_probabilities_list))
    assert logP.shape == (num_mixtures, n, 51)

    for composition in composition_list:
        x = torch.as_tensor(composition)
        assert x.shape == (n,)

        composition_arr: NDArray[np.float64] = composition
        ln_gamma_c_ref = np.array(
            [
                [
                    get_lngamma_comb(composition_arr.tolist(), i, areas, volumes)
                    for i in range(n)
                ]
                for areas, volumes in zip(areas_list, volumes_list, strict=True)
            ]
        )

        ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
        assert ln_gamma_c.shape == (num_mixtures, n)

        np.testing.assert_allclose(ln_gamma_c.numpy(), ln_gamma_c_ref, rtol=_RTOL)

        p_mix_ref = np.stack(
            [get_psigma_mix(composition_arr, profiles) for profiles in profiles_list]
        )

        log_p_mix = cosmo_layer.mixture_log_probabilities(x, a, logP)
        p_mix = torch.softmax(log_p_mix, dim=-1)
        assert p_mix.shape == (num_mixtures, 51)
        np.testing.assert_allclose(p_mix, p_mix_ref, rtol=_RTOL)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_multiple_mixtures_multiple_compositions(
    n: int,
    mixtures: list[_MixtureType],
    conditions: dict[int, list[_ConditionType]],
    cosmo_layer: CosmoLayer,
) -> None:
    mixtures = list(filter(lambda mixture: mixture[0] == n, mixtures))
    num_mixtures = len(mixtures)

    _, composition_list = zip(*conditions[n], strict=True)
    num_compositions = len(composition_list)

    areas_list = []
    volumes_list = []
    log_probabilities_list = []
    compositions_array_list = []
    ln_gamma_c_ref_list = []
    p_mix_ref_list = []
    for _, profiles, areas, volumes, log_probabilities in mixtures:
        for composition in composition_list:
            composition_arr: NDArray[np.float64] = composition
            areas_list.append(areas)
            volumes_list.append(volumes)
            log_probabilities_list.append(log_probabilities)
            compositions_array_list.append(composition)
            ln_gamma_c_ref_list.append(
                [
                    get_lngamma_comb(composition_arr.tolist(), i, areas, volumes)
                    for i in range(n)
                ]
            )
            p_mix_ref_list.append(get_psigma_mix(composition_arr, profiles))
    shape = (num_compositions * num_mixtures, n)

    a = torch.as_tensor(np.array(areas_list, dtype=np.float64))
    assert a.shape == shape

    v = torch.as_tensor(np.array(volumes_list, dtype=np.float64))
    assert v.shape == shape

    logP = torch.as_tensor(np.stack(log_probabilities_list))
    assert logP.shape == (*shape, 51)

    x = torch.as_tensor(np.array(compositions_array_list, dtype=np.float64))
    assert x.shape == shape

    ln_gamma_c_ref = np.array(ln_gamma_c_ref_list)
    assert ln_gamma_c_ref.shape == shape

    p_mix_ref = np.stack(p_mix_ref_list)
    assert p_mix_ref.shape == (num_compositions * num_mixtures, 51)

    ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
    assert ln_gamma_c.shape == (num_compositions * num_mixtures, n)
    np.testing.assert_allclose(ln_gamma_c.numpy(), ln_gamma_c_ref, rtol=_RTOL)

    log_p_mix = cosmo_layer.mixture_log_probabilities(x, a, logP)
    p_mix = torch.softmax(log_p_mix, dim=-1)
    np.testing.assert_allclose(p_mix, p_mix_ref, rtol=_RTOL)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_broadcasting(
    n: int,
    mixtures: list[_MixtureType],
    conditions: dict[int, list[_ConditionType]],
    cosmo_layer: CosmoLayer,
) -> None:
    mixtures = list(filter(lambda mixture: mixture[0] == n, mixtures))
    num_mixtures = len(mixtures)

    _, composition_list = zip(*conditions[n], strict=True)
    num_compositions = len(composition_list)

    _, profiles_list, areas_list, volumes_list, log_probabilities_list = zip(
        *mixtures, strict=True
    )

    a = torch.as_tensor(np.array(areas_list, dtype=np.float64)).reshape(1, -1, n)
    assert a.shape == (1, num_mixtures, n)

    v = torch.as_tensor(np.array(volumes_list)).reshape(1, -1, n)
    assert v.shape == (1, num_mixtures, n)

    logP = torch.as_tensor(np.stack(log_probabilities_list)).reshape(1, -1, n, 51)
    assert logP.shape == (1, num_mixtures, n, 51)

    x = torch.as_tensor(np.array(composition_list, dtype=np.float64)).reshape(-1, 1, n)
    assert x.shape == (num_compositions, 1, n)

    ln_gamma_c_ref = np.array(
        [
            [
                [
                    get_lngamma_comb(composition.tolist(), i, areas, volumes)
                    for i in range(n)
                ]
                for areas, volumes in zip(areas_list, volumes_list, strict=True)
            ]
            for composition in composition_list
        ]
    )
    assert ln_gamma_c_ref.shape == (num_compositions, num_mixtures, n)

    ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
    assert ln_gamma_c.shape == (num_compositions, num_mixtures, n)
    np.testing.assert_allclose(ln_gamma_c.numpy(), ln_gamma_c_ref, rtol=_RTOL)

    p_mix_ref = np.array(
        [
            [get_psigma_mix(composition, profiles) for profiles in profiles_list]
            for composition in composition_list
        ]
    )
    assert p_mix_ref.shape == (num_compositions, num_mixtures, 51)

    log_p_mix = cosmo_layer.mixture_log_probabilities(x, a, logP)
    p_mix = torch.softmax(log_p_mix, dim=-1)
    np.testing.assert_allclose(p_mix, p_mix_ref, rtol=_RTOL)


def test_combinatorial_differentiation(
    mixtures: list[_MixtureType],
    conditions: dict[int, list[_ConditionType]],
    cosmo_layer: CosmoLayer,
) -> None:
    """Test that combinatorial activity coefficients backpropagate correctly."""
    # Use double precision for gradcheck
    dtype = torch.float64

    def reduced_excess_gibbs_energy(
        x: torch.Tensor, a: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Scalar function for gradcheck: gERT_c = x^T @ ln_gamma_c."""
        ln_gamma_c = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
        return (x * ln_gamma_c).sum()

    for mixture in mixtures:
        n, _, areas, volumes, _ = mixture
        a = torch.as_tensor(areas, dtype=dtype).requires_grad_(True)
        v = torch.as_tensor(volumes, dtype=dtype).requires_grad_(True)
        for _, composition in conditions[n]:
            x = torch.as_tensor(composition, dtype=dtype).requires_grad_(True)

            # Check that the gradients are computed correctly
            assert torch.autograd.gradcheck(
                reduced_excess_gibbs_energy,
                (x, a, v),
                atol=1e-6,
                rtol=1e-5,
                eps=1e-6,
            )

            # Check the thermodynamic consistency
            gERT = reduced_excess_gibbs_energy(x, a, v)
            gERT.backward()
            with torch.no_grad():
                log_gamma = cosmo_layer.combinatorial_log_activity_coefficients(x, a, v)
                x_grad = cast(torch.Tensor, x.grad)
                np.testing.assert_allclose(
                    log_gamma, x_grad + gERT - (x * x_grad).sum(), rtol=_RTOL
                )


def test_reduced_energy_matrix(
    interaction_matrix: NDArray[np.float64], cosmo_layer: CosmoLayer
) -> None:
    T = torch.as_tensor(523.15)
    U_RT = cosmo_layer.scaled_interaction_energy_matrix(T)
    assert U_RT.shape == (51, 51)
    np.testing.assert_allclose(U_RT, interaction_matrix / (_R * T), rtol=_RTOL)


def test_reduced_energy_matrix_broadcasting(
    interaction_matrix: NDArray[np.float64], cosmo_layer: CosmoLayer
) -> None:
    T = torch.as_tensor([273.15 + i * 100 for i in range(4)])
    U_RT = cosmo_layer.scaled_interaction_energy_matrix(T)
    assert U_RT.shape == (4, 51, 51)
    np.testing.assert_allclose(
        U_RT, interaction_matrix / (_R * T[:, None, None]), rtol=_RTOL
    )
