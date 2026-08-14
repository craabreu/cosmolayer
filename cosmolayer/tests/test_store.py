"""Tests for cosmolayer.store: SegmentStore, SigmaProfileTable, SigmaGrid,
and the segment-averaging / binning kernels underneath them.

Builds a real store from the four ``.cosmo`` fixtures already bundled in
``cosmolayer/data/`` (rather than synthetic data), so these are true
end-to-end tests of the parse -> average -> save -> load -> bin pipeline.
"""

import json
import pathlib
from importlib.resources import files

import numpy as np
import pytest
from rdkit import Chem

from cosmolayer.store import (
    AVERAGING_SCHEMES,
    COSMO_SAC_2010,
    AveragingScheme,
    SegmentStore,
    SigmaGrid,
    StoreMetadata,
)
from cosmolayer.store.__main__ import main as store_main
from cosmolayer.store.averaging import (
    RESERVED_SCHEME_NAMES,
    average_sigmas,
    average_sigmas_by_molecule,
)
from cosmolayer.store.binning import (
    compute_per_atom_properties,
    compute_per_molecule_properties,
    row_indices_from_offsets,
)
from cosmolayer.store.segments import (
    ATOM_INDICES_FILE,
    DATA_FILE,
    METADATA_FILE,
    MOLECULES_FILE,
)

COSMO_DATA_DIR = pathlib.Path(str(files("cosmolayer.data")))
# Each .cosmo fixture's atom table includes explicit hydrogens, so the
# SMILES used to build a store from it must too (SegmentStore.from_cosmo_files
# checks the SMILES's atom count against the COSMO file's).
SMILES_TO_FILENAME = {
    Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))): f"{smi}.cosmo"
    for smi in ["O", "CF", "NCCO", "C=C(N)O"]
}


@pytest.fixture(scope="session")
def built_store_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build a real segment-data store from the bundled .cosmo fixtures,
    once per test session."""
    storage_dir = tmp_path_factory.mktemp("segment_store")
    SegmentStore.from_cosmo_files(
        COSMO_DATA_DIR, SMILES_TO_FILENAME, storage_dir, num_threads=1
    )
    return storage_dir


@pytest.fixture(scope="session")
def store(built_store_dir: pathlib.Path) -> SegmentStore:
    return SegmentStore.load(built_store_dir)


# --------------------------------------------------------------------- #
# SigmaGrid
# --------------------------------------------------------------------- #


class TestSigmaGrid:
    def test_bin_width_and_values(self) -> None:
        grid = SigmaGrid(0.025, 51)
        assert grid.bin_width == pytest.approx(0.001)
        assert len(grid) == 51
        assert grid.values[0] == pytest.approx(-0.025)
        assert grid.values[-1] == pytest.approx(0.025)

    def test_for_centered_profiles_odd_gains_a_point_and_preserves_bin_width(
        self,
    ) -> None:
        grid = SigmaGrid(0.025, 51)
        centered = grid.for_centered_profiles()
        assert centered.num_points == 52
        assert centered.bin_width == pytest.approx(grid.bin_width)
        assert not np.any(centered.values == 0.0)

    def test_for_centered_profiles_even_is_unchanged(self) -> None:
        grid = SigmaGrid(0.025, 50)
        assert grid.for_centered_profiles() == grid

    def test_from_values_round_trips(self) -> None:
        grid = SigmaGrid(0.025, 51)
        assert SigmaGrid.from_values(grid.values) == grid


# --------------------------------------------------------------------- #
# AveragingScheme
# --------------------------------------------------------------------- #


class TestAveragingScheme:
    def test_reserved_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            AveragingScheme("metadata", averaging_radius=0.5, f_decay=1.0)

    def test_default_schemes_present(self) -> None:
        assert {"cosmo-rs", "cosmo-sac-2002", "cosmo-sac-2010"} == {
            scheme.name for scheme in AVERAGING_SCHEMES
        }

    def test_reserved_names_match_the_actual_store_filenames(self) -> None:
        """``averaging`` cannot import ``segments`` (that would be a cycle),
        so it hardcodes the reserved stems. Guard against the two drifting:
        a scheme named after a store file would overwrite it on save."""
        assert RESERVED_SCHEME_NAMES == {
            path.stem
            for path in (DATA_FILE, ATOM_INDICES_FILE, MOLECULES_FILE, METADATA_FILE)
        }


# --------------------------------------------------------------------- #
# averaging kernels
# --------------------------------------------------------------------- #


class TestAveraging:
    def test_single_scheme_matches_multi_scheme_row(self) -> None:
        rng = np.random.default_rng(0)
        n = 12
        coords = rng.normal(size=(n, 3))
        charges = rng.normal(size=n) * 1e-3
        areas = rng.uniform(0.5, 2.0, size=n)

        schemes = [COSMO_SAC_2010, AveragingScheme("x", 0.6, 1.2)]
        combined = average_sigmas(coords, charges, areas, schemes)
        single = average_sigmas(coords, charges, areas, [COSMO_SAC_2010])
        np.testing.assert_allclose(combined[0], single[0])

    def test_by_molecule_matches_per_molecule_call(self) -> None:
        rng = np.random.default_rng(1)
        sizes = [3, 5, 2]
        offsets = np.cumsum([0, *sizes[:-1]]).astype(np.int64)
        n = sum(sizes)
        coords = rng.normal(size=(n, 3))
        charges = rng.normal(size=n) * 1e-3
        areas = rng.uniform(0.5, 2.0, size=n)

        schemes = [COSMO_SAC_2010]
        whole = average_sigmas_by_molecule(
            coords, charges, areas, offsets, schemes, num_threads=1
        )
        pieces = []
        for start, size in zip(offsets, sizes, strict=True):
            stop = start + size
            pieces.append(
                average_sigmas(
                    coords[start:stop], charges[start:stop], areas[start:stop], schemes
                )[0]
            )
        np.testing.assert_allclose(whole[0], np.concatenate(pieces))


# --------------------------------------------------------------------- #
# binning helpers
# --------------------------------------------------------------------- #


def test_row_indices_from_offsets() -> None:
    np.testing.assert_array_equal(
        row_indices_from_offsets(np.array([0, 2, 3]), 5), [0, 0, 1, 2, 2]
    )


# --------------------------------------------------------------------- #
# SegmentStore: build / save / load
# --------------------------------------------------------------------- #


class TestSegmentStoreRoundTrip:
    def test_build_reports_all_four_molecules(self, store: SegmentStore) -> None:
        assert store.metadata.num_molecules == 4
        assert store.metadata.num_cosmo_parse_failures == 0
        assert len(store.molecules_df) == 4

    def test_save_load_round_trip_preserves_arrays(
        self, store: SegmentStore, tmp_path: pathlib.Path
    ) -> None:
        store.save(tmp_path)
        reloaded = SegmentStore.load(tmp_path)
        np.testing.assert_array_equal(np.asarray(store.data), np.asarray(reloaded.data))
        np.testing.assert_array_equal(
            np.asarray(store.atom_indices), np.asarray(reloaded.atom_indices)
        )
        assert reloaded.data.dtype == np.float32
        assert reloaded.atom_indices.dtype == np.int64
        assert list(reloaded.molecules_df.columns) == [
            "smiles",
            "segment_offsets",
            "atom_offsets",
            "num_atoms",
            "volume",
        ]
        scheme_names = sorted(scheme.name for scheme in AVERAGING_SCHEMES)
        assert sorted(reloaded.metadata.schemes) == scheme_names
        for name in scheme_names:
            np.testing.assert_array_equal(
                np.asarray(store.averaged_sigmas[name]),
                np.asarray(reloaded.averaged_sigmas[name]),
            )

    def test_exists_false_for_incomplete_store(self, tmp_path: pathlib.Path) -> None:
        assert not SegmentStore.exists(tmp_path)

    def test_exists_false_when_scheme_array_missing(
        self, store: SegmentStore, tmp_path: pathlib.Path
    ) -> None:
        store.save(tmp_path)
        scheme_file = tmp_path / f"{AVERAGING_SCHEMES[0].name}.npy"
        scheme_file.unlink()
        assert not SegmentStore.exists(tmp_path)

    def test_load_raises_on_missing_store(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            SegmentStore.load(tmp_path)

    def test_skip_averaging_with_empty_schemes(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, SMILES_TO_FILENAME, tmp_path, schemes=(), num_threads=1
        )
        assert s.averaged_sigmas == {}
        assert s.metadata.schemes == {}

    def test_no_successful_molecules_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="No COSMO files"):
            SegmentStore.from_cosmo_files(
                COSMO_DATA_DIR, {}, tmp_path, ignore_errors=True
            )

    def test_ignore_errors_counts_failures(self, tmp_path: pathlib.Path) -> None:
        bad_mapping = dict(SMILES_TO_FILENAME)
        bad_mapping["CCCC"] = "O.cosmo"  # atom count mismatch -> parse failure
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, bad_mapping, tmp_path, ignore_errors=True, num_threads=1
        )
        assert s.metadata.num_cosmo_parse_failures == 1
        assert s.metadata.num_molecules == 4

    def test_ignore_errors_false_raises(self, tmp_path: pathlib.Path) -> None:
        bad_mapping = {"CCCC": "O.cosmo"}
        with pytest.raises(ValueError):
            SegmentStore.from_cosmo_files(COSMO_DATA_DIR, bad_mapping, tmp_path)


class TestSegmentStoreSigmas:
    def test_raw_sigma_is_charge_over_area(self, store: SegmentStore) -> None:
        np.testing.assert_allclose(
            store.sigmas(), np.asarray(store.charges) / np.asarray(store.areas)
        )

    def test_unknown_scheme_raises_key_error(self, store: SegmentStore) -> None:
        with pytest.raises(KeyError):
            store.sigmas("not-a-real-scheme")


# --------------------------------------------------------------------- #
# SigmaProfileTable: correctness, ported from the old __main__ assertions
# --------------------------------------------------------------------- #


class TestSigmaProfileTable:
    def test_atom_profiles_sum_to_one_or_are_zero(self, store: SegmentStore) -> None:
        table = store.compute_atom_sigma_profiles(num_threads=1)
        has_area = table.areas > 0
        np.testing.assert_allclose(table.profiles[has_area].sum(axis=1), 1.0, atol=1e-5)
        assert not table.profiles[~has_area].any()

    def test_atom_properties_match_direct_sums(self, store: SegmentStore) -> None:
        table = store.compute_atom_sigma_profiles(num_threads=1)
        total_num_atoms = int(store.molecules_df["num_atoms"].sum())
        direct_areas = compute_per_atom_properties(
            np.asarray(store.areas), store.atom_indices, total_num_atoms
        )
        np.testing.assert_allclose(table.areas, direct_areas, rtol=1e-5)

    def test_centered_profiles_have_near_zero_first_moment(
        self, store: SegmentStore
    ) -> None:
        table = store.compute_atom_sigma_profiles(num_threads=1, centered=True)
        has_area = table.areas > 0
        first_moments = table.profiles[has_area].astype(np.float64) @ table.sigma_values
        bin_width = table.binning_grid.bin_width
        assert np.abs(first_moments).max() < 1e-2 * bin_width

    def test_centered_grid_has_no_zero_point(self, store: SegmentStore) -> None:
        table = store.compute_atom_sigma_profiles(num_threads=1, centered=True)
        assert not np.any(table.sigma_values == 0.0)

    def test_molecule_profiles_conserve_area_and_are_nonnegative(
        self, store: SegmentStore
    ) -> None:
        atom_table = store.compute_atom_sigma_profiles(num_threads=1, centered=True)
        molecule_table = atom_table.aggregate(num_threads=1)

        assert atom_table.atom_offsets is not None
        molecule_areas = compute_per_molecule_properties(
            atom_table.areas, atom_table.atom_offsets
        )
        mass_err = np.abs(molecule_table.profiles.sum(axis=1) / molecule_areas - 1)
        assert mass_err.max() < 1e-4
        assert (molecule_table.profiles >= 0).all()

    def test_aggregate_matches_direct_molecule_binning(
        self, store: SegmentStore
    ) -> None:
        atom_table = store.compute_atom_sigma_profiles(num_threads=1, centered=False)
        aggregated = atom_table.aggregate(num_threads=1)

        direct = store.compute_molecule_sigma_profiles(num_threads=1, centered=False)

        np.testing.assert_allclose(aggregated.areas, direct.areas, rtol=1e-5)
        ground_truth = direct.areas[:, None].astype(np.float64) * direct.profiles
        np.testing.assert_allclose(aggregated.profiles, ground_truth, atol=1e-3)

    @pytest.mark.parametrize("centered", [False, True])
    def test_aggregate_lands_on_the_base_grid(
        self, store: SegmentStore, centered: bool
    ) -> None:
        """Aggregating un-translates each atom profile back to absolute
        sigma, so the molecule result is uncentered and on the base grid --
        even when the atom profiles were centered onto a wider one."""
        grid = SigmaGrid()
        atom_table = store.compute_atom_sigma_profiles(
            num_threads=1, grid=grid, centered=centered
        )
        molecule_table = atom_table.aggregate(num_threads=1)

        assert molecule_table.centered is False
        assert molecule_table.grid == grid
        assert molecule_table.binning_grid == grid
        assert molecule_table.profiles.shape[1] == len(grid)
        assert len(molecule_table.sigma_values) == len(grid)

    def test_centered_aggregate_approximates_direct_binning(
        self, store: SegmentStore
    ) -> None:
        """The centered path loses a little resolution to the half-bin
        regrid, so compare distributions by Wasserstein-1 distance (in bin
        widths) rather than bin by bin."""
        atom_table = store.compute_atom_sigma_profiles(num_threads=1, centered=True)
        reconstructed = atom_table.aggregate(num_threads=1).profiles
        direct = store.compute_molecule_sigma_profiles(num_threads=1, centered=False)
        ground_truth = direct.areas[:, None].astype(np.float64) * direct.profiles

        normalized_recon = reconstructed / reconstructed.sum(axis=1, keepdims=True)
        normalized_truth = ground_truth / ground_truth.sum(axis=1, keepdims=True)
        profile_w1 = np.abs(
            np.cumsum(normalized_truth, axis=1) - np.cumsum(normalized_recon, axis=1)
        ).sum(axis=1)
        assert profile_w1.max() < 1.0

    def test_normalize_true_sums_to_one(self, store: SegmentStore) -> None:
        atom_table = store.compute_atom_sigma_profiles(num_threads=1, centered=True)
        normalized = atom_table.aggregate(num_threads=1, normalize=True)
        np.testing.assert_allclose(normalized.profiles.sum(axis=1), 1.0, atol=1e-5)

    def test_aggregate_on_molecule_level_table_raises(
        self, store: SegmentStore
    ) -> None:
        molecule_table = store.compute_molecule_sigma_profiles(num_threads=1)
        with pytest.raises(ValueError, match="already at molecule level"):
            molecule_table.aggregate()

    def test_level_property(self, store: SegmentStore) -> None:
        atom_table = store.compute_atom_sigma_profiles(num_threads=1)
        molecule_table = store.compute_molecule_sigma_profiles(num_threads=1)
        assert atom_table.level == "atom"
        assert molecule_table.level == "molecule"


# --------------------------------------------------------------------- #
# SegmentStore._reorder_molecule
# --------------------------------------------------------------------- #


class TestReorderMolecule:
    def test_all_zero_map_returned_unchanged(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        reordered = SegmentStore._reorder_molecule(mol)
        assert reordered is mol

    def test_zero_based_permutation_reorders(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        for atom, map_num in zip(mol.GetAtoms(), [2, 0, 1], strict=True):
            atom.SetAtomMapNum(map_num)
        reordered = SegmentStore._reorder_molecule(mol)
        symbols = [a.GetSymbol() for a in reordered.GetAtoms()]
        assert symbols == ["C", "O", "C"]

    def test_one_based_permutation_reorders(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        for atom, map_num in zip(mol.GetAtoms(), [3, 1, 2], strict=True):
            atom.SetAtomMapNum(map_num)
        reordered = SegmentStore._reorder_molecule(mol)
        symbols = [a.GetSymbol() for a in reordered.GetAtoms()]
        assert symbols == ["C", "O", "C"]

    def test_bad_map_numbers_raise(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        for atom, map_num in zip(mol.GetAtoms(), [1, 1, 2], strict=True):
            atom.SetAtomMapNum(map_num)
        with pytest.raises(ValueError, match="Bad atom map numbers"):
            SegmentStore._reorder_molecule(mol)


# --------------------------------------------------------------------- #
# StoreMetadata
# --------------------------------------------------------------------- #


def test_store_metadata_round_trips_through_dict() -> None:
    metadata = StoreMetadata(
        num_molecules=3,
        num_cosmo_parse_failures=1,
        schemes={COSMO_SAC_2010.name: COSMO_SAC_2010},
    )
    restored = StoreMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
    assert restored.num_molecules == metadata.num_molecules
    assert restored.num_cosmo_parse_failures == metadata.num_cosmo_parse_failures
    assert restored.schemes[COSMO_SAC_2010.name] == COSMO_SAC_2010


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def test_main_runs_end_to_end(
    built_store_dir: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = store_main(
        ["--storage-dir", str(built_store_dir), "--num-threads", "1"]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Molecule profile mass conservation" in out
