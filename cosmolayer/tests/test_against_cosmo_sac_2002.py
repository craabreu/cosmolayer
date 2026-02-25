"""
Test CosmoLayer against a reference implementation of the COSMO-SAC 2002 model.

Reference implementation:
    https://github.com/usnistgov/COSMOSAC
"""

import itertools
import re
from importlib.resources import files
from typing import TypeAlias, TypedDict, cast

import cCOSMO
import numpy as np
import pandas as pd
import pytest
import torch
from numpy.typing import NDArray

from cosmolayer import CosmoLayer
from cosmolayer.cosmosac import CosmoSac2002Model

_NUM_POINTS = 3
_RTOL = 1e-6
_ATOL = 1e-7
_REF_TEMP = 298.15


_MixtureType: TypeAlias = tuple[
    list[str], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]


class ReferenceData(TypedDict):
    """Reference data structure for test validation."""

    ln_gamma_c: dict[
        int,  # mix
        dict[
            int,  # comp
            NDArray[np.float64],  # ln_gamma_c
        ],
    ]

    ln_gamma_r: dict[
        int,  # mix
        dict[
            int,  # comp
            dict[
                int,  # temp
                NDArray[np.float64],  # ln_gamma_r
            ],
        ],
    ]


_ReferenceResultsType: TypeAlias = dict[int, ReferenceData]  # n -> reference data


def assert_close(a: torch.Tensor, b: NDArray[np.float64]) -> None:
    np.testing.assert_allclose(a.numpy(), b, rtol=_RTOL, atol=_ATOL)


def reduced_excess_gibbs_energy(
    T: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    cosmo_layer: CosmoLayer,
) -> torch.Tensor:
    """Scalar function for gradcheck: gERT = x^T @ ln_gamma."""
    ln_gamma = cosmo_layer(T, x, a, v, p)
    ge_RT: torch.Tensor = (x * ln_gamma).sum()
    return ge_RT


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
    return (
        smiles,
        np.array(areas, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
        probabilities,
    )


def get_cosmo_model(compounds: list[str]) -> cCOSMO.COSMO1:
    path = files("cosmolayer") / "data" / "cosmo-sac-2002"
    db = cCOSMO.VirginiaTechProfileDatabase(
        str(path / "Sigma_Profile_Database_Index_v2.txt"), str(path)
    )
    for compound in compounds:
        db.add_profile(db.normalize_identifier(compound))
    return cCOSMO.COSMO1(compounds, db)


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
) -> _ReferenceResultsType:
    """
    Reference results for the test cases.
    """
    results: _ReferenceResultsType = {}
    for n, mixtures_n in mixtures.items():
        results[n] = ReferenceData(ln_gamma_c={}, ln_gamma_r={})
        for mix, (smiles, _, _, _) in enumerate(mixtures_n):
            cosmo = get_cosmo_model(smiles)
            cosmo.get_mutable_COSMO_constants().fast_Gamma = True
            results[n]["ln_gamma_c"][mix] = {}
            results[n]["ln_gamma_r"][mix] = {}
            for comp, composition in enumerate(compositions[n]):
                results[n]["ln_gamma_c"][mix][comp] = cosmo.get_lngamma_comb(
                    0.0, composition.T
                )
                results[n]["ln_gamma_r"][mix][comp] = {}
                for temp, temperature in enumerate(temperatures):
                    results[n]["ln_gamma_r"][mix][comp][temp] = cosmo.get_lngamma_resid(
                        temperature, composition.T
                    )
    return results


@pytest.fixture
def cosmo_layer() -> CosmoLayer:
    return CosmoLayer(
        [CosmoSac2002Model.create_interaction_matrices(_REF_TEMP)[0]],
        CosmoSac2002Model.temperature_exponents,
        CosmoSac2002Model.area_per_segment,
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
    for mix, (_, areas, volumes, probs) in enumerate(mixtures[n]):
        a = torch.as_tensor(areas)
        v = torch.as_tensor(volumes)
        p = torch.as_tensor(probs)

        for comp, composition in enumerate(compositions[n]):
            x = torch.as_tensor(composition)

            ln_gamma_c_ref = ref["ln_gamma_c"][mix][comp]
            ln_gamma_c = cosmo_layer.log_combinatorial_activity_coefficients(x, a, v)
            assert_close(ln_gamma_c, ln_gamma_c_ref)

            for temp, temperature in enumerate(temperatures):
                T = torch.as_tensor(temperature)

                ln_gamma_r_ref = ref["ln_gamma_r"][mix][comp][temp]
                ln_gamma_r = cosmo_layer.log_residual_activity_coefficients(T, x, a, p)
                assert_close(ln_gamma_r, ln_gamma_r_ref)

                ln_gamma_ref = ln_gamma_c_ref + ln_gamma_r_ref
                ln_gamma = cosmo_layer.log_activity_coefficients(T, x, a, v, p)
                assert_close(ln_gamma, ln_gamma_ref)


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
        (areas, volumes, probs, temperature, composition)
        for _, areas, volumes, probs in mixtures[n]
        for composition in compositions[n]
        for temperature in temperatures
    ]
    a, v, p, T, x = (torch.as_tensor(np.array(x)) for x in zip(*data, strict=True))

    triples = list(
        itertools.product(
            range(len(mixtures[n])),
            range(len(compositions[n])),
            range(len(temperatures)),
        ),
    )
    ref = reference_results[n]

    ln_gamma_c_ref = np.array(
        [ref["ln_gamma_c"][mix][comp] for mix, comp, _ in triples]
    )
    ln_gamma_c = cosmo_layer.log_combinatorial_activity_coefficients(x, a, v)
    assert_close(ln_gamma_c, ln_gamma_c_ref)

    ln_gamma_r_ref = np.array(
        [ref["ln_gamma_r"][mix][comp][temp] for mix, comp, temp in triples]
    )
    ln_gamma_r = cosmo_layer.log_residual_activity_coefficients(T, x, a, p)
    assert_close(ln_gamma_r, ln_gamma_r_ref)

    ln_gamma_ref = ln_gamma_c_ref + ln_gamma_r_ref
    ln_gamma = cosmo_layer.log_activity_coefficients(T, x, a, v, p)
    assert_close(ln_gamma, ln_gamma_ref)


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

    _, areas_list, volumes_list, probs_list = zip(*mixtures[n], strict=True)

    a = torch.as_tensor(np.array(areas_list, dtype=np.float64))
    assert a.shape == (num_mixtures, n)

    v = torch.as_tensor(np.array(volumes_list))
    assert v.shape == (num_mixtures, n)

    p = torch.as_tensor(np.stack(probs_list))
    assert p.shape == (num_mixtures, n, 51)

    T = torch.as_tensor(np.array(temperatures, dtype=np.float64))
    assert T.shape == (num_temperatures,)

    T = T.reshape(num_temperatures, 1, 1)

    x = torch.as_tensor(np.array(compositions[n], dtype=np.float64))
    assert x.shape == (num_compositions, n)
    x = x.reshape(num_compositions, 1, n)

    ref = reference_results[n]

    all_mix = range(num_mixtures)
    all_comp = range(num_compositions)
    all_temp = range(num_temperatures)

    ln_gamma_c_ref = np.array(
        [[ref["ln_gamma_c"][mix][comp] for mix in all_mix] for comp in all_comp]
    )
    ln_gamma_c = cosmo_layer.log_combinatorial_activity_coefficients(x, a, v)
    assert ln_gamma_c.shape == (num_compositions, num_mixtures, n)
    assert_close(ln_gamma_c, ln_gamma_c_ref)

    ln_gamma_r_ref = np.array(
        [
            [
                [ref["ln_gamma_r"][mix][comp][temp] for mix in all_mix]
                for comp in all_comp
            ]
            for temp in all_temp
        ]
    )
    ln_gamma_r = cosmo_layer.log_residual_activity_coefficients(T, x, a, p)
    assert ln_gamma_r.shape == (num_temperatures, num_compositions, num_mixtures, n)
    assert_close(ln_gamma_r, ln_gamma_r_ref)

    ln_gamma_ref = ln_gamma_c_ref[None, ...] + ln_gamma_r_ref
    ln_gamma = cosmo_layer.log_activity_coefficients(T, x, a, v, p)
    assert ln_gamma.shape == (num_temperatures, num_compositions, num_mixtures, n)
    assert_close(ln_gamma, ln_gamma_ref)


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_combinatorial_differentiation(
    n: int,
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    cosmo_layer: CosmoLayer,
) -> None:
    """Test that combinatorial activity coefficients backpropagate correctly."""
    # Use double precision for gradcheck
    dtype = torch.float64

    def reduced_excess_gibbs_energy(
        x: torch.Tensor, a: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Scalar function for gradcheck: gERT_c = x^T @ ln_gamma_c."""
        ln_gamma_c = cosmo_layer.log_combinatorial_activity_coefficients(x, a, v)
        return (x * ln_gamma_c).sum()

    for mixture in mixtures[n]:
        _, areas, volumes, _ = mixture
        a = torch.as_tensor(areas, dtype=dtype).requires_grad_(True)
        v = torch.as_tensor(volumes, dtype=dtype).requires_grad_(True)
        for composition in compositions[n]:
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
                log_gamma = cosmo_layer.log_combinatorial_activity_coefficients(x, a, v)
                x_grad = cast(torch.Tensor, x.grad)
                np.testing.assert_allclose(
                    log_gamma,
                    x_grad + gERT - (x * x_grad).sum(),
                    rtol=_RTOL,
                    atol=_ATOL,
                )


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

    _, areas, volumes, probs = mixtures[n][mix]
    a = torch.as_tensor(areas, dtype=dtype)
    v = torch.as_tensor(volumes, dtype=dtype)
    p = torch.as_tensor(probs, dtype=dtype)

    def func(T: torch.Tensor, pm_sqrt_x: torch.Tensor) -> torch.Tensor:
        x = pm_sqrt_x**2
        return reduced_excess_gibbs_energy(T, x, a, v, p, cosmo_layer)

    T = torch.as_tensor(temperatures[temp], dtype=dtype).requires_grad_(True)

    x = torch.as_tensor(compositions[n][comp], dtype=dtype).requires_grad_(True)
    pm_sqrt_x = torch.sqrt(x)

    # Check that the gradients are computed correctly
    assert torch.autograd.gradcheck(
        func,
        (T, pm_sqrt_x),
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
) -> None:
    """Test that gradients w.r.t. interaction matrix parameters are correct."""
    dtype = torch.float64

    cosmo_layer = CosmoLayer(
        [CosmoSac2002Model.create_interaction_matrices(_REF_TEMP)[0]],
        CosmoSac2002Model.temperature_exponents,
        CosmoSac2002Model.area_per_segment,
        learn_matrices=True,
    )

    rng = np.random.default_rng(seed)
    mix = rng.integers(len(mixtures[n]))
    temp = rng.integers(len(temperatures))

    mixture = mixtures[n][mix]
    _, areas, volumes, probs = mixture
    temperature = temperatures[temp]

    composition = rng.integers(1, 10, size=n)
    composition = composition / composition.sum()

    a = torch.as_tensor(areas, dtype=dtype)
    v = torch.as_tensor(volumes, dtype=dtype)
    p = torch.as_tensor(probs, dtype=dtype)
    T = torch.as_tensor(temperature, dtype=dtype)

    U_RT_param = next(cosmo_layer.parameters())
    assert U_RT_param.requires_grad

    x = torch.as_tensor(composition, dtype=dtype)
    cosmo_layer.zero_grad()
    gERT = reduced_excess_gibbs_energy(T, x, a, v, p, cosmo_layer)
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
        gERT_plus = reduced_excess_gibbs_energy(T, x, a, v, p, cosmo_layer)
        U_RT_param.data[i, j] = original_value - eps
        gERT_minus = reduced_excess_gibbs_energy(T, x, a, v, p, cosmo_layer)
        grad = (gERT_plus.item() - gERT_minus.item()) / (2 * eps)
        U_RT_param.data[i, j] = original_value
        abs_error = (analytical_grad[i, j] - grad).abs()
        assert abs_error < 0.01 * abs(grad) or abs_error < 1e-8


@pytest.mark.parametrize("n", [2, 3], ids=["binary", "ternary"])
def test_thermodynamic_consistency(
    n: int,
    temperatures: list[float],
    mixtures: dict[int, list[_MixtureType]],
    compositions: dict[int, list[NDArray[np.float64]]],
    cosmo_layer: CosmoLayer,
) -> None:
    """Test that thermodynamic consistency is preserved."""
    # Use double precision for gradcheck
    dtype = torch.float64

    for _, areas, volumes, probs in mixtures[n]:
        a = torch.as_tensor(areas, dtype=dtype)
        v = torch.as_tensor(volumes, dtype=dtype)
        p = torch.as_tensor(probs, dtype=dtype)

        for temperature in temperatures:
            T = torch.as_tensor(temperature, dtype=dtype)

            for composition in compositions[n]:
                x = torch.as_tensor(composition, dtype=dtype).requires_grad_(True)

                gERT = reduced_excess_gibbs_energy(T, x, a, v, p, cosmo_layer)
                gERT.backward()
                with torch.no_grad():
                    log_gamma = cosmo_layer(T, x, a, v, p)
                    x_grad = cast(torch.Tensor, x.grad)
                    np.testing.assert_allclose(
                        log_gamma,
                        x_grad + gERT - (x * x_grad).sum(),
                        rtol=_RTOL,
                        atol=_ATOL,
                    )
