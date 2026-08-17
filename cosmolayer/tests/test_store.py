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

import cosmolayer.store.clustering as clustering_module
from cosmolayer.cosmosac import Component
from cosmolayer.store import (
    AVERAGING_SCHEMES,
    COSMO_SAC_2010,
    AveragingScheme,
    ClusteringSpecs,
    SegmentStore,
    SigmaGrid,
    StoreMetadata,
)
from cosmolayer.store.__main__ import main as store_main
from cosmolayer.store.averaging import average_sigmas, average_sigmas_by_molecule
from cosmolayer.store.binning import (
    compute_per_atom_properties,
    compute_per_molecule_properties,
    row_indices_from_offsets,
)
from cosmolayer.store.clustering import FingerprintGenerator, butina_cluster

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
    # Explicitly typed: TempPathFactory.mktemp's declared return type has
    # varied across pytest releases (Path vs an untyped Any), and this
    # repo pins no pytest version, so relying on the inferred type is not
    # stable across CI runs.
    storage_dir: pathlib.Path = tmp_path_factory.mktemp("segment_store")
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

    @pytest.mark.parametrize(("num_points", "has_zero_node"), [(51, True), (52, False)])
    def test_odd_and_even_point_counts_are_both_usable(
        self, num_points: int, has_zero_node: bool
    ) -> None:
        """Whether a node lands exactly on sigma = 0 is the caller's
        choice, not something the package overrides."""
        grid = SigmaGrid(0.025, num_points)
        assert len(grid) == num_points
        assert np.any(grid.values == 0.0) == has_zero_node

    def test_from_values_round_trips(self) -> None:
        grid = SigmaGrid(0.025, 51)
        assert SigmaGrid.from_values(grid.values) == grid


# --------------------------------------------------------------------- #
# AveragingScheme
# --------------------------------------------------------------------- #


class TestAveragingScheme:
    def test_construction_never_validates_the_name(self) -> None:
        """AveragingScheme is a pure value object -- it has no notion of
        storage, so any name is constructible; collision with a store's
        own files is SegmentStore's concern (see TestReservedSchemeNames)."""
        for name in ("data", "atom_indices", "molecules", "metadata", "anything"):
            assert AveragingScheme(name, averaging_radius=0.5, f_decay=1.0).name == name

    def test_default_schemes_present(self) -> None:
        assert {"cosmo-rs", "cosmo-sac-2002", "cosmo-sac-2010"} == {
            scheme.name for scheme in AVERAGING_SCHEMES
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
            "cluster_id",
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


class TestReservedSchemeNames:
    """A scheme's averaged sigmas are written to "<name>.npy" (see
    SegmentStore.save), so only a name matching one of the store's own
    *.npy* files can actually collide -- molecules.parquet and
    metadata.json have different suffixes and can never collide with any
    "<name>.npy", so schemes named after them must be allowed."""

    @pytest.mark.parametrize("name", ["data", "atom_indices"])
    def test_npy_colliding_name_rejected(self, store: SegmentStore, name: str) -> None:
        with pytest.raises(ValueError, match="reserved"):
            store.compute_averaged_sigmas(
                schemes=[AveragingScheme(name, 0.5, 1.0)], num_threads=1
            )

    @pytest.mark.parametrize("name", ["molecules", "metadata"])
    def test_non_npy_colliding_name_allowed(
        self, store: SegmentStore, name: str
    ) -> None:
        result = store.compute_averaged_sigmas(
            schemes=[AveragingScheme(name, 0.5, 1.0)], num_threads=1
        )
        assert name in result


# --------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------- #


class TestFingerprintGenerator:
    def test_generate_returns_dense_bit_array(self) -> None:
        generator = FingerprintGenerator(ClusteringSpecs(fp_size=512))
        fp = generator.generate(Chem.MolFromSmiles("CCO"))
        assert fp.shape == (512,)
        assert fp.dtype == np.int8
        assert set(np.unique(fp)) <= {0, 1}

    def test_identical_molecules_get_identical_fingerprints(self) -> None:
        generator = FingerprintGenerator(ClusteringSpecs())
        fp1 = generator.generate(Chem.MolFromSmiles("c1ccccc1O"))
        fp2 = generator.generate(Chem.MolFromSmiles("Oc1ccccc1"))
        np.testing.assert_array_equal(fp1, fp2)


class TestButinaCluster:
    def test_empty_input(self) -> None:
        result = butina_cluster(np.empty((0, 16), dtype=np.int8), cutoff=0.5)
        assert result.shape == (0,)

    def test_single_molecule_is_its_own_cluster(self) -> None:
        fp = np.array([[1, 0, 1, 0]], dtype=np.int8)
        np.testing.assert_array_equal(butina_cluster(fp, cutoff=0.5), [0])

    def test_identical_fingerprints_share_a_cluster(self) -> None:
        fp = np.tile(np.array([1, 0, 1, 0, 1], dtype=np.int8), (3, 1))
        result = butina_cluster(fp, cutoff=0.1)
        assert len(set(result.tolist())) == 1

    def test_maximally_different_fingerprints_split(self) -> None:
        fp = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int8)
        result = butina_cluster(fp, cutoff=0.1)
        assert result[0] != result[1]

    def test_tight_cutoff_separates_similar_but_distinct_molecules(self) -> None:
        generator = FingerprintGenerator(ClusteringSpecs())
        smiles = ["CCCCCCCC", "CCCCCCCCCCCCCCCC", "c1ccccc1"]
        fps = np.stack([generator.generate(Chem.MolFromSmiles(s)) for s in smiles])
        result = butina_cluster(fps, cutoff=0.05)
        # The two long-chain alkanes are near-identical; benzene is not.
        assert result[0] == result[1]
        assert result[2] != result[0]

    def test_delegates_to_vendored_chalcedon(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: butina_cluster must call the vendored
        chalcedon implementation, not a re-inlined copy of it."""
        calls = []
        real = clustering_module._chalcedon_butina_cluster

        def spy(*args: object, **kwargs: object) -> np.ndarray:
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(clustering_module, "_chalcedon_butina_cluster", spy)
        fp = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
        butina_cluster(fp, cutoff=0.1)
        assert len(calls) == 1


class TestSegmentStoreClustering:
    def test_cluster_id_column_present_and_typed(self, store: SegmentStore) -> None:
        assert "cluster_id" in store.molecules_df.columns
        cluster_ids = store.molecules_df["cluster_id"]
        assert cluster_ids.dtype == np.int64
        assert len(cluster_ids) == len(store.molecules_df)
        assert cluster_ids.min() >= 0

    def test_custom_clustering_specs_accepted(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            SMILES_TO_FILENAME,
            tmp_path,
            clustering_specs=ClusteringSpecs(cutoff=0.0, fp_size=256),
            schemes=(),
            num_threads=1,
        )
        # cutoff=0.0 means only exact fingerprint matches share a cluster;
        # these four molecules are all distinct, so each is its own cluster.
        assert sorted(s.molecules_df["cluster_id"].tolist()) == [0, 1, 2, 3]


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
        bin_width = table.grid.bin_width
        assert np.abs(first_moments).max() < 1e-2 * bin_width

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
    @pytest.mark.parametrize("grid", [SigmaGrid(0.025, 51), SigmaGrid(0.03, 52)])
    def test_binning_never_changes_the_requested_grid(
        self, store: SegmentStore, grid: SigmaGrid, centered: bool
    ) -> None:
        """The grid the caller asks for is the grid they get -- at atom
        level, at molecule level, centered or not, odd or even point
        count. This is the guard against a grid being silently
        substituted, which previously produced 52 columns for a 51-point
        request."""
        atom_table = store.compute_atom_sigma_profiles(
            num_threads=1, grid=grid, centered=centered
        )
        assert atom_table.grid == grid
        assert atom_table.centered is centered
        assert atom_table.profiles.shape[1] == len(grid)
        assert len(atom_table.sigma_values) == len(grid)

        molecule_table = atom_table.aggregate(num_threads=1)
        assert molecule_table.grid == grid
        assert molecule_table.profiles.shape[1] == len(grid)
        # Aggregation un-translates each atom back to absolute sigma, so
        # the molecule result is never centered whatever the input was.
        assert molecule_table.centered is False

        direct = store.compute_molecule_sigma_profiles(
            num_threads=1, grid=grid, centered=centered
        )
        assert direct.grid == grid
        assert direct.profiles.shape[1] == len(grid)

    def test_aggregate_onto_a_wider_grid_of_equal_bin_width(
        self, store: SegmentStore
    ) -> None:
        """A different output grid is allowed as long as it shares the
        atom grid's bin width -- column alignment is by point-count
        difference alone."""
        grid = SigmaGrid(0.025, 51)
        # Extending by N bins per side means +2N points and
        # +N*bin_width of extent, which keeps bin_width identical.
        extra_bins_per_side = 5
        wider = SigmaGrid(
            grid.max_abs_sigma + extra_bins_per_side * grid.bin_width,
            grid.num_points + 2 * extra_bins_per_side,
        )
        assert wider.bin_width == pytest.approx(grid.bin_width)

        atom_table = store.compute_atom_sigma_profiles(
            num_threads=1, grid=grid, centered=True
        )
        molecules = atom_table.aggregate(num_threads=1, grid=wider)

        assert molecules.grid == wider
        assert molecules.profiles.shape[1] == len(wider)
        # Widening the *output* grid only changes how much is clipped at
        # the aggregate step; it cannot recover mass already clipped when
        # the atom profiles were binned. So the total is unchanged, and
        # the invariant that matters is that it still equals the summed
        # atom area.
        np.testing.assert_allclose(
            molecules.profiles.sum(axis=1), molecules.areas, rtol=1e-4
        )
        np.testing.assert_allclose(
            molecules.profiles.sum(),
            atom_table.aggregate(num_threads=1).profiles.sum(),
            rtol=1e-5,
        )

    def test_aggregate_onto_mismatched_bin_width_raises(
        self, store: SegmentStore
    ) -> None:
        atom_table = store.compute_atom_sigma_profiles(
            num_threads=1, grid=SigmaGrid(0.025, 51)
        )
        with pytest.raises(ValueError, match="bin width"):
            atom_table.aggregate(num_threads=1, grid=SigmaGrid(0.025, 101))

    def test_centered_aggregate_approximates_direct_binning(
        self, store: SegmentStore
    ) -> None:
        """Centering then un-centering quantizes twice (segments -> atom
        bins, atom bins -> molecule bins), so compare distributions by
        Wasserstein-1 distance in bin widths rather than bin by bin.

        Measured at 0.12 bin widths on these fixtures. Binning onto a grid
        one half-bin wider at each end -- ``SigmaGrid(0.0255, 52)``, which
        earlier versions silently substituted here -- clips less tail mass
        and gives 0.10; the difference is the cost of honouring the
        requested ``max_abs_sigma`` exactly, and is the caller's to trade.
        """
        atom_table = store.compute_atom_sigma_profiles(num_threads=1, centered=True)
        reconstructed = atom_table.aggregate(num_threads=1).profiles
        direct = store.compute_molecule_sigma_profiles(num_threads=1, centered=False)
        ground_truth = direct.areas[:, None].astype(np.float64) * direct.profiles

        normalized_recon = reconstructed / reconstructed.sum(axis=1, keepdims=True)
        normalized_truth = ground_truth / ground_truth.sum(axis=1, keepdims=True)
        profile_w1 = np.abs(
            np.cumsum(normalized_truth, axis=1) - np.cumsum(normalized_recon, axis=1)
        ).sum(axis=1)
        assert profile_w1.max() < 0.2

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
# Cross-validation against cosmosac.Component
# --------------------------------------------------------------------- #


class TestCrossValidationAgainstComponent:
    """``cosmolayer.store`` and ``cosmolayer.cosmosac.Component`` implement
    the same COSMO-SAC segment-averaging and sigma-profile-binning physics
    independently -- different distance formulas (exact pairwise vs. a
    clipped Gram-matrix expansion), different code paths (vectorized vs. a
    scalar per-segment loop), no shared code. These tests compare their
    outputs directly, so a bug that both implementations happen to share
    is still invisible, but a bug specific to either one is not.

    ``Component.sigma_profile`` (with ``merge_profiles=True``) is an
    area-weighted profile in Å² that sums to ``Component.area``, split by
    hydrogen-bonding class (NHB/OH/OT) and recombined as
    ``profile_nhb + (1 - p) * (profile_oh + profile_ot) + p * (profile_oh
    + profile_ot)`` -- the ``p`` (hb_probability) terms cancel exactly
    regardless of its value, so the merged profile equals the plain
    averaged-sigma histogram with no HB splitting involved. That makes it
    directly comparable to ``store``'s molecule-level profile once the
    latter is scaled from an area *fraction* back to Å² by its own area.
    """

    @pytest.mark.parametrize("smi", ["O", "CF", "NCCO", "C=C(N)O"])
    def test_molecule_profile_matches_component(
        self, smi: str, tmp_path: pathlib.Path
    ) -> None:
        cosmo_text = (COSMO_DATA_DIR / f"{smi}.cosmo").read_text()
        grid = SigmaGrid(0.025, 51)

        # Component's defaults are COSMO-SAC-2010's averaging_radius/f_decay,
        # the same constants store.COSMO_SAC_2010 is built from.
        component = Component(cosmo_text, merge_profiles=True)
        assert np.allclose(component.sigma_grid, grid.values)

        mapped_smi = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi)))
        store = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, {mapped_smi: f"{smi}.cosmo"}, tmp_path, num_threads=1
        )
        table = store.compute_molecule_sigma_profiles(
            scheme="cosmo-sac-2010", grid=grid, centered=False, num_threads=1
        )

        np.testing.assert_allclose(table.areas[0], component.area, rtol=1e-4)
        store_area_profile = table.areas[0] * table.profiles[0]
        np.testing.assert_allclose(
            store_area_profile, component.sigma_profile, atol=2e-4
        )


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
