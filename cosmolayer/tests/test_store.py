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
import pandas as pd
import pytest
from numpy.typing import NDArray
from rdkit import Chem

from cosmolayer.cosmosac import Component
from cosmolayer.parser import parse_cosmo_file
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
from cosmolayer.store.clustering import (
    FingerprintGenerator,
    butina_cluster,
    cluster_medoid_distances,
)
from cosmolayer.store.coarse_graining import compute_atom_remap
from cosmolayer.store.segments import _SMILES_PARSER_PARAMS
from cosmolayer.store.splitting import greedy_cluster_split
from cosmolayer.store.subsampling import apportion_counts, restrict_to_molecules


def _atom_mapped(smi: str) -> str:
    """Return `smi` with sequential 1-based atom-map numbers, hydrogens
    included -- the convention SegmentStore requires input smiles under
    (see GH issue #43), and stores them under too."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    return Chem.MolToSmiles(mol)


COSMO_DATA_DIR = pathlib.Path(str(files("cosmolayer.data")))
FILENAMES = [f"{smi}.cosmo" for smi in ["O", "CF", "NCCO", "C=C(N)O"]]


def _write_bad_cosmo_file(directory: pathlib.Path) -> str:
    """Write a copy of O.cosmo with its oxygen's element corrupted to an
    unrecognized symbol, so RDKit can't build a molecule from it. Used to
    exercise from_cosmo_files's parse-failure path."""
    text = (COSMO_DATA_DIR / "O.cosmo").read_text()
    old_line = (
        "O1       0.000000000    0.000000000   -0.405655705 XXXX 1"
        "      xx      O   0.000"
    )
    assert text.count(old_line) == 1
    corrupted = text.replace(
        old_line, old_line.replace("      O   0.000", "      Xx   0.000")
    )
    (directory / "bad.cosmo").write_text(corrupted)
    return "bad.cosmo"


@pytest.fixture(scope="session")
def built_store_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build a real segment-data store from the bundled .cosmo fixtures,
    once per test session."""
    # Explicitly typed: TempPathFactory.mktemp's declared return type has
    # varied across pytest releases (Path vs an untyped Any), and this
    # repo pins no pytest version, so relying on the inferred type is not
    # stable across CI runs.
    storage_dir: pathlib.Path = tmp_path_factory.mktemp("segment_store")
    SegmentStore.from_cosmo_files(COSMO_DATA_DIR, FILENAMES, storage_dir, num_threads=1)
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

    def test_atoms_df_matches_source_cosmo_file(self, store: SegmentStore) -> None:
        assert list(store.atoms_df.columns) == ["id", "element", "x", "y", "z"]
        assert len(store.atoms_df) == store.molecules_df["num_atoms"].sum()

        # O (water) is the only one of the four fixtures with 3 atoms.
        water_row = store.molecules_df.loc[store.molecules_df["num_atoms"] == 3].iloc[0]
        offset = int(water_row["atom_offsets"])
        water_atoms = store.atoms_df.iloc[offset : offset + 3]

        cosmo_text = (COSMO_DATA_DIR / "O.cosmo").read_text()
        _, expected_atom_df, _, _ = parse_cosmo_file(cosmo_text)
        assert list(water_atoms["element"]) == list(expected_atom_df["element"])
        assert list(water_atoms["id"]) == list(expected_atom_df["id"])
        np.testing.assert_allclose(water_atoms["x"], expected_atom_df["x"], atol=1e-6)
        np.testing.assert_allclose(water_atoms["y"], expected_atom_df["y"], atol=1e-6)
        np.testing.assert_allclose(water_atoms["z"], expected_atom_df["z"], atol=1e-6)

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
            "cluster_distance",
        ]
        pd.testing.assert_series_equal(
            reloaded.molecules_df["cluster_distance"],
            store.molecules_df["cluster_distance"],
            check_names=True,
        )
        assert list(reloaded.atoms_df.columns) == ["id", "element", "x", "y", "z"]
        pd.testing.assert_frame_equal(
            reloaded.atoms_df.reset_index(drop=True),
            store.atoms_df.reset_index(drop=True),
        )
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
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        assert s.averaged_sigmas == {}
        assert s.metadata.schemes == {}

    def test_no_successful_molecules_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="No COSMO files"):
            SegmentStore.from_cosmo_files(
                COSMO_DATA_DIR, [], tmp_path, ignore_errors=True
            )

    def test_ignore_errors_counts_failures(self, tmp_path: pathlib.Path) -> None:
        cosmo_dir = tmp_path / "cosmo_files"
        cosmo_dir.mkdir()
        for filename in FILENAMES:
            (cosmo_dir / filename).write_text((COSMO_DATA_DIR / filename).read_text())
        bad_filename = _write_bad_cosmo_file(cosmo_dir)
        s = SegmentStore.from_cosmo_files(
            cosmo_dir,
            [*FILENAMES, bad_filename],
            tmp_path / "store",
            ignore_errors=True,
            num_threads=1,
        )
        assert s.metadata.num_cosmo_parse_failures == 1
        assert s.metadata.num_molecules == 4

    def test_ignore_errors_false_raises(self, tmp_path: pathlib.Path) -> None:
        bad_filename = _write_bad_cosmo_file(tmp_path)
        with pytest.raises(ValueError):
            SegmentStore.from_cosmo_files(tmp_path, [bad_filename], tmp_path / "store")


class TestRepeatedFilenames:
    """A filename may appear more than once in the list (e.g. multiple
    conformers of the same molecule); each occurrence gets its own
    molecules_df row."""

    def test_repeated_filename_yields_one_row_per_occurrence(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, ["O.cosmo", "O.cosmo", "CF.cosmo"], tmp_path, num_threads=1
        )
        assert s.metadata.num_molecules == 3
        assert len(s.molecules_df) == 3
        water_smi = s.molecules_df["smiles"].iloc[0]
        assert list(s.molecules_df["smiles"]).count(water_smi) == 2

    def test_conformer_rows_have_distinct_offsets(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, ["O.cosmo", "O.cosmo"], tmp_path, num_threads=1
        )
        assert list(s.molecules_df["segment_offsets"]) == sorted(
            set(s.molecules_df["segment_offsets"])
        )
        assert list(s.molecules_df["atom_offsets"]) == sorted(
            set(s.molecules_df["atom_offsets"])
        )


class TestStoredSmilesIsAlwaysAtomMapped:
    """molecules_df["smiles"] must carry atom-map numbers reflecting local
    (COSMO) atom index -- otherwise atom order isn't recoverable from a
    *stored* string after Chem.MolToSmiles's default re-canonicalization.
    See GH issue #43."""

    def test_atom_map_numbers_present(self, store: SegmentStore) -> None:
        for smi, num_atoms in zip(
            store.molecules_df["smiles"], store.molecules_df["num_atoms"], strict=True
        ):
            mol = Chem.MolFromSmiles(smi, _SMILES_PARSER_PARAMS)
            map_nums = {atom.GetAtomMapNum() for atom in mol.GetAtoms()}
            assert map_nums == set(range(1, num_atoms + 1))


class TestChargeEstimation:
    """The molecular charge fed to RDKit's bond-order perception is
    estimated as round(-sum(segment charges)) -- COSMO segment charges
    screen the solute's own charge, so they roughly cancel it. All four
    bundled fixtures are neutral molecules, so that estimate must land on
    zero for each."""

    @pytest.mark.parametrize("filename", FILENAMES)
    def test_neutral_fixtures_get_zero_estimated_charge(
        self, filename: str, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, [filename], tmp_path, schemes=(), num_threads=1
        )
        mol = Chem.MolFromSmiles(
            s.molecules_df["smiles"].iloc[0], _SMILES_PARSER_PARAMS
        )
        assert Chem.GetFormalCharge(mol) == 0


class TestSmilesDerivedFromGeometry:
    """molecules_df["smiles"] is now derived straight from each .cosmo
    file's 3D geometry (via get_rdkit_molecule), not supplied by the
    caller -- pin the exact output for two small, unambiguous fixtures."""

    def test_water_and_fluoromethane_smiles(self, store: SegmentStore) -> None:
        assert store.molecules_df["smiles"].iloc[0] == "[O:1]([H:2])[H:3]"
        assert store.molecules_df["smiles"].iloc[1] == "[C:1]([F:2])([H:3])([H:4])[H:5]"


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


class TestClusteringSpecs:
    def test_default_cutoff_matches_chalcedon(self) -> None:
        assert ClusteringSpecs().cutoff == 0.65


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
        chalcedon implementation, not a re-inlined copy of it. Patches by
        dotted-path string (rather than a static attribute reference) so
        this doesn't reach into clustering.py's private import binding in
        a way mypy's strict re-export check would flag."""
        call_sizes: list[int] = []

        def spy(fingerprints: NDArray[np.int8], cutoff: float) -> NDArray[np.intp]:
            call_sizes.append(fingerprints.shape[0])
            return np.zeros(fingerprints.shape[0], dtype=np.intp)

        monkeypatch.setattr(
            "cosmolayer.store.clustering._chalcedon_butina_cluster", spy
        )
        fp = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
        butina_cluster(fp, cutoff=0.1)
        assert call_sizes == [2]


def _tanimoto_distance(a: NDArray[np.int8], b: NDArray[np.int8]) -> float:
    """Tanimoto distance including self-similarity (empty-vs-empty → 1)."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    dot = float(a64 @ b64)
    union = float(a64 @ a64 + b64 @ b64 - dot)
    if union == 0.0:
        return 1.0
    return 1.0 - dot / union


class TestClusterMedoidDistances:
    def test_empty_input(self) -> None:
        fps = np.empty((0, 16), dtype=np.int8)
        ids = np.empty(0, dtype=np.int64)
        result = cluster_medoid_distances(fps, ids)
        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_singleton_is_zero(self) -> None:
        fps = np.array([[1, 0, 1, 0]], dtype=np.int8)
        ids = np.array([0], dtype=np.int64)
        result = cluster_medoid_distances(fps, ids)
        np.testing.assert_array_equal(result, [0.0])
        assert result.dtype == np.float64

    def test_identical_fingerprints_are_all_zero(self) -> None:
        fps = np.tile(np.array([1, 0, 1, 0, 1], dtype=np.int8), (3, 1))
        ids = np.zeros(3, dtype=np.int64)
        result = cluster_medoid_distances(fps, ids)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_unique_medoid_matches_tanimoto_to_that_member(self) -> None:
        # A=[1,1,1,0,0], B=[1,1,1,1,0], C=[0,0,0,1,1]
        # s(A,B)=0.75, s(A,C)=0, s(B,C)=0.2 → B is the unique medoid.
        fps = np.array(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 1]],
            dtype=np.int8,
        )
        ids = np.zeros(3, dtype=np.int64)
        result = cluster_medoid_distances(fps, ids)
        assert result[1] == 0.0
        np.testing.assert_allclose(
            result,
            [
                _tanimoto_distance(fps[0], fps[1]),
                0.0,
                _tanimoto_distance(fps[2], fps[1]),
            ],
        )

    def test_clusters_are_independent(self) -> None:
        fps = np.array(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 0]],
            dtype=np.int8,
        )
        ids = np.array([0, 0, 0, 1], dtype=np.int64)
        result = cluster_medoid_distances(fps, ids)
        np.testing.assert_allclose(
            result[:3],
            [
                _tanimoto_distance(fps[0], fps[1]),
                0.0,
                _tanimoto_distance(fps[2], fps[1]),
            ],
        )
        assert result[3] == 0.0

    def test_tie_breaks_to_lowest_index(self) -> None:
        # Equilateral: every pairwise Tanimoto similarity is 1/3.
        fps = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.int8)
        ids = np.zeros(3, dtype=np.int64)
        result = cluster_medoid_distances(fps, ids)
        assert result[0] == 0.0
        expected = _tanimoto_distance(fps[1], fps[0])
        np.testing.assert_allclose(result[1:], [expected, expected])

    def test_length_mismatch_raises(self) -> None:
        fps = np.array([[1, 0], [0, 1]], dtype=np.int8)
        ids = np.array([0], dtype=np.int64)
        with pytest.raises(ValueError, match="same length"):
            cluster_medoid_distances(fps, ids)


class TestSegmentStoreClustering:
    def test_cluster_id_column_present_and_typed(self, store: SegmentStore) -> None:
        assert "cluster_id" in store.molecules_df.columns
        cluster_ids = store.molecules_df["cluster_id"]
        assert cluster_ids.dtype == np.int64
        assert len(cluster_ids) == len(store.molecules_df)
        assert cluster_ids.min() >= 0

    def test_cluster_distance_column_present_and_typed(
        self, store: SegmentStore
    ) -> None:
        assert "cluster_distance" in store.molecules_df.columns
        distances = store.molecules_df["cluster_distance"]
        assert distances.dtype == np.float64
        assert len(distances) == len(store.molecules_df)
        assert (distances >= 0.0).all()
        # Each cluster has a medoid at exactly 0.
        for cluster_id, group in store.molecules_df.groupby("cluster_id"):
            assert (group["cluster_distance"] == 0.0).any(), cluster_id

    def test_custom_clustering_specs_accepted(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            clustering_specs=ClusteringSpecs(cutoff=0.0, fp_size=256),
            schemes=(),
            num_threads=1,
        )
        # cutoff=0.0 means only exact fingerprint matches share a cluster;
        # these four molecules are all distinct, so each is its own cluster.
        assert sorted(s.molecules_df["cluster_id"].tolist()) == [0, 1, 2, 3]


# --------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------- #


class TestGreedyClusterSplit:
    def test_empty_input(self) -> None:
        result = greedy_cluster_split(
            np.empty(0, dtype=np.int64), {"train": 0.8, "test": 0.2}
        )
        assert result.shape == (0,)

    def test_matches_chalcedon_doctest_example(self) -> None:
        # Mirrors chalcedon.greedy_cluster_split's own doctest: whole
        # clusters go to whichever split is furthest below target.
        cluster_ids = np.array([0, 0, 0, 1, 1, 2, 3], dtype=np.int64)
        result = greedy_cluster_split(cluster_ids, {"train": 0.6, "test": 0.4})
        expected = ["train", "train", "train", "test", "test", "train", "test"]
        assert result.tolist() == expected

    def test_clusters_never_split_across_splits(self) -> None:
        cluster_ids = np.array([0, 0, 0, 1, 1, 2, 3], dtype=np.int64)
        result = greedy_cluster_split(cluster_ids, {"train": 0.6, "test": 0.4})
        for cluster_id in np.unique(cluster_ids):
            assert len(set(result[cluster_ids == cluster_id].tolist())) == 1

    def test_invalid_fractions_raise(self) -> None:
        cluster_ids = np.array([0, 1], dtype=np.int64)
        with pytest.raises(ValueError, match="sum to 1.0"):
            greedy_cluster_split(cluster_ids, {"train": 0.5, "test": 0.6})

    def test_delegates_to_vendored_chalcedon(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: greedy_cluster_split must call the vendored
        chalcedon implementation, not a re-inlined copy of it."""
        calls: list[NDArray[np.int64]] = []

        def spy(
            cluster_ids: NDArray[np.int64], fractions: dict[str, float]
        ) -> dict[str, NDArray[np.intp]]:
            calls.append(cluster_ids)
            return {
                "train": np.array([0], dtype=np.intp),
                "test": np.array([1], dtype=np.intp),
            }

        monkeypatch.setattr(
            "cosmolayer.store.splitting._chalcedon_greedy_cluster_split", spy
        )
        cluster_ids = np.array([0, 1], dtype=np.int64)
        result = greedy_cluster_split(cluster_ids, {"train": 0.5, "test": 0.5})
        assert len(calls) == 1
        assert result.tolist() == ["train", "test"]


class TestSegmentStoreSplitting:
    def test_no_split_column_by_default(self, store: SegmentStore) -> None:
        assert "split" not in store.molecules_df.columns

    def test_assign_splits_adds_column(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        labels = s.assign_splits({"train": 0.75, "test": 0.25})
        assert "split" in s.molecules_df.columns
        np.testing.assert_array_equal(s.molecules_df["split"].values, labels)
        assert set(labels.tolist()) <= {"train", "test"}

    def test_assign_splits_overwrites_previous_column(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        s.assign_splits({"train": 0.75, "test": 0.25})
        second = s.assign_splits({"a": 0.5, "b": 0.5})
        assert set(s.molecules_df["split"].tolist()) <= {"a", "b"}
        np.testing.assert_array_equal(s.molecules_df["split"].values, second)

    def test_invalid_fractions_raise(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        with pytest.raises(ValueError):
            s.assign_splits({"train": 0.5})

    def test_split_fractions_at_build_time(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        assert "split" in s.molecules_df.columns
        assert set(s.molecules_df["split"].tolist()) <= {"train", "test"}

    def test_split_omitted_without_split_fractions(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        assert "split" not in s.molecules_df.columns

    def test_split_column_survives_save_load_round_trip(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        reloaded = SegmentStore.load(tmp_path)
        np.testing.assert_array_equal(
            s.molecules_df["split"].values, reloaded.molecules_df["split"].values
        )


# --------------------------------------------------------------------- #
# subsampling
# --------------------------------------------------------------------- #


class TestApportionCounts:
    def test_sums_to_total(self) -> None:
        result = apportion_counts(np.array([10, 20, 30]), 12)
        assert result.sum() == 12

    def test_proportional_to_sizes(self) -> None:
        result = apportion_counts(np.array([10, 90]), 10)
        np.testing.assert_array_equal(result, [1, 9])

    def test_never_exceeds_bucket_size(self) -> None:
        result = apportion_counts(np.array([1, 100]), 50)
        assert result[0] <= 1

    def test_zero_total_gives_all_zeros(self) -> None:
        result = apportion_counts(np.array([10, 20]), 0)
        np.testing.assert_array_equal(result, [0, 0])


class TestRestrictToMolecules:
    def test_atom_index_space_is_compacted_and_contiguous(
        self, store: SegmentStore
    ) -> None:
        selected = np.array([0, 2], dtype=np.int64)
        restricted = restrict_to_molecules(store, selected)
        total_num_atoms = int(restricted.molecules_df["num_atoms"].sum())
        atom_indices = np.asarray(restricted.atom_indices, dtype=np.int64)
        assert max(atom_indices.tolist()) + 1 == total_num_atoms

    def test_molecule_count_and_metadata(self, store: SegmentStore) -> None:
        selected = np.array([1, 3], dtype=np.int64)
        restricted = restrict_to_molecules(store, selected)
        assert len(restricted.molecules_df) == 2
        assert restricted.metadata.num_molecules == 2
        np.testing.assert_array_equal(
            restricted.molecules_df["smiles"].values,
            store.molecules_df["smiles"].values[selected],
        )

    def test_segment_count_matches_selected_molecules(
        self, store: SegmentStore
    ) -> None:
        selected = np.array([0, 1], dtype=np.int64)
        restricted = restrict_to_molecules(store, selected)
        offsets = store.molecules_df["segment_offsets"].values.astype("int64")
        expected_num_segments = offsets[2] - offsets[0]
        assert len(restricted.data) == expected_num_segments

    def test_averaged_sigmas_sliced_consistently_with_data(
        self, store: SegmentStore
    ) -> None:
        selected = np.array([0, 2], dtype=np.int64)
        restricted = restrict_to_molecules(store, selected)
        for name, arr in restricted.averaged_sigmas.items():
            assert len(arr) == len(restricted.data)
            assert name in store.averaged_sigmas

    def test_cluster_distance_preserved_for_kept_rows(
        self, store: SegmentStore
    ) -> None:
        selected = np.array([1, 3], dtype=np.int64)
        restricted = restrict_to_molecules(store, selected)
        np.testing.assert_array_equal(
            restricted.molecules_df["cluster_distance"].values,
            store.molecules_df["cluster_distance"].values[selected],
        )


class TestSegmentStoreSubsample:
    def test_requires_existing_split_column(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        with pytest.raises(ValueError, match="assign_splits"):
            s.subsample(2)

    def test_result_has_requested_molecule_count(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        subsampled = s.subsample(2)
        assert len(subsampled.molecules_df) == 2

    def test_kept_molecules_retain_their_original_split(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        subsampled = s.subsample(3)
        original_by_smiles = dict(
            zip(s.molecules_df["smiles"], s.molecules_df["split"], strict=True)
        )
        subsampled_smiles = subsampled.molecules_df["smiles"]
        subsampled_splits = subsampled.molecules_df["split"]
        for smi, split in zip(subsampled_smiles, subsampled_splits, strict=True):
            assert split == original_by_smiles[smi]

    def test_deterministic_without_seed(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        first = s.subsample(2)
        second = s.subsample(2)
        np.testing.assert_array_equal(
            first.molecules_df["smiles"].values, second.molecules_df["smiles"].values
        )

    def test_invalid_num_molecules_raises(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        with pytest.raises(ValueError):
            s.subsample(0)
        with pytest.raises(ValueError):
            s.subsample(100)

    def test_atoms_df_restricted_to_kept_molecules(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR,
            FILENAMES,
            tmp_path,
            split_fractions={"train": 0.75, "test": 0.25},
            schemes=(),
            num_threads=1,
        )
        subsampled = s.subsample(2)
        assert len(subsampled.atoms_df) == subsampled.molecules_df["num_atoms"].sum()

        original_by_smiles = {
            smi: (int(offset), int(n))
            for smi, offset, n in zip(
                s.molecules_df["smiles"],
                s.molecules_df["atom_offsets"],
                s.molecules_df["num_atoms"],
                strict=True,
            )
        }
        expected = pd.concat(
            [
                s.atoms_df.iloc[
                    original_by_smiles[smi][0] : original_by_smiles[smi][0]
                    + original_by_smiles[smi][1]
                ]
                for smi in subsampled.molecules_df["smiles"]
            ]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(subsampled.atoms_df, expected)


# --------------------------------------------------------------------- #
# coarse-graining
# --------------------------------------------------------------------- #


class TestComputeAtomRemap:
    def test_merges_hydrogens_into_their_heavy_neighbor(self) -> None:
        new_local_index, survivors, new_smiles = compute_atom_remap(_atom_mapped("O"))
        reduced = Chem.MolFromSmiles(new_smiles, _SMILES_PARSER_PARAMS)
        assert reduced.GetNumAtoms() == 1
        assert len(set(new_local_index.values())) == 1
        assert survivors == {0}

    def test_unmapped_input_raises(self) -> None:
        with pytest.raises(ValueError, match="atom-map"):
            compute_atom_remap("CCO")

    def test_isotope_tagged_hydrogen_is_not_merged(self) -> None:
        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                atom.SetIsotope(2)
                break
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        deuterated = Chem.MolToSmiles(mol)

        new_local_index, survivors, new_smiles = compute_atom_remap(deuterated)
        reduced = Chem.MolFromSmiles(new_smiles, _SMILES_PARSER_PARAMS)
        # Oxygen + one surviving (isotope-tagged) hydrogen; the other two
        # ordinary hydrogens are merged into the oxygen.
        assert reduced.GetNumAtoms() == 2
        assert len(set(new_local_index.values())) == 2
        assert len(survivors) == 2


class TestSegmentStoreCoarseGrain:
    def test_raises_on_unmapped_store(self, tmp_path: pathlib.Path) -> None:
        # Since GH issue #43's fix, from_cosmo_files always stamps atom
        # maps -- so simulate a store that predates that fix (or was
        # otherwise built with unmapped smiles) by stripping them back out.
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )

        def _strip_map_numbers(smi: str) -> str:
            mol = Chem.MolFromSmiles(smi, _SMILES_PARSER_PARAMS)
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return Chem.MolToSmiles(mol)

        s.molecules_df["smiles"] = [
            _strip_map_numbers(smi) for smi in s.molecules_df["smiles"]
        ]
        with pytest.raises(ValueError, match="atom-map"):
            s.coarse_grain()

    def test_reduces_atom_count_and_compacts_index_space(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        coarse = s.coarse_grain()
        coarse_num_atoms = coarse.molecules_df["num_atoms"].sum()
        assert coarse_num_atoms < s.molecules_df["num_atoms"].sum()
        atom_indices = np.asarray(coarse.atom_indices, dtype=np.int64)
        assert max(atom_indices.tolist()) + 1 == coarse_num_atoms

    def test_segments_and_molecule_count_unchanged(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        coarse = s.coarse_grain()
        assert len(coarse.molecules_df) == len(s.molecules_df)
        np.testing.assert_array_equal(
            coarse.molecules_df["segment_offsets"].values,
            s.molecules_df["segment_offsets"].values,
        )

    def test_data_and_metadata_columns_untouched(self, tmp_path: pathlib.Path) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        coarse = s.coarse_grain()
        np.testing.assert_array_equal(np.asarray(coarse.data), np.asarray(s.data))
        np.testing.assert_array_equal(
            coarse.molecules_df["volume"].values, s.molecules_df["volume"].values
        )
        np.testing.assert_array_equal(
            coarse.molecules_df["cluster_id"].values,
            s.molecules_df["cluster_id"].values,
        )

    def test_atoms_df_drops_merged_hydrogens_and_keeps_survivors(
        self, tmp_path: pathlib.Path
    ) -> None:
        s = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, FILENAMES, tmp_path, schemes=(), num_threads=1
        )
        coarse = s.coarse_grain()
        coarse_num_atoms = int(coarse.molecules_df["num_atoms"].sum())
        assert len(coarse.atoms_df) == coarse_num_atoms
        assert len(coarse.atoms_df) < len(s.atoms_df)
        # Every surviving row's id/element/coordinates are untouched --
        # merged hydrogens are dropped, not averaged or repositioned. None
        # of these fixtures have isotope-tagged hydrogens, so every H is
        # merged away.
        assert set(coarse.atoms_df["id"]).issubset(set(s.atoms_df["id"]))
        assert "H" not in set(coarse.atoms_df["element"])


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

        store = SegmentStore.from_cosmo_files(
            COSMO_DATA_DIR, [f"{smi}.cosmo"], tmp_path, num_threads=1
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


class TestCliDiscoversCosmoFiles:
    """--cosmo-files-dir is scanned for every file matching --extension
    (default 'cosmo'); --smiles-to-filenames no longer exists."""

    def test_default_extension_discovers_cosmo_files(
        self, tmp_path: pathlib.Path
    ) -> None:
        cosmo_dir = tmp_path / "cosmo_files"
        cosmo_dir.mkdir()
        for filename in ["O.cosmo", "CF.cosmo"]:
            (cosmo_dir / filename).write_text((COSMO_DATA_DIR / filename).read_text())
        storage_dir = tmp_path / "store"

        exit_code = store_main(
            [
                "--storage-dir",
                str(storage_dir),
                "--cosmo-files-dir",
                str(cosmo_dir),
                "--num-threads",
                "1",
            ]
        )

        assert exit_code == 0
        assert len(SegmentStore.load(storage_dir).molecules_df) == 2

    def test_json_extension_discovers_chaos_files(self, tmp_path: pathlib.Path) -> None:
        json_dir = tmp_path / "json_files"
        json_dir.mkdir()
        chaos_text = (COSMO_DATA_DIR / "chaos_sample.json").read_text()
        (json_dir / "sample.json").write_text(chaos_text)
        storage_dir = tmp_path / "store"

        exit_code = store_main(
            [
                "--storage-dir",
                str(storage_dir),
                "--cosmo-files-dir",
                str(json_dir),
                "--extension",
                "json",
                "--num-threads",
                "1",
            ]
        )

        assert exit_code == 0
        assert len(SegmentStore.load(storage_dir).molecules_df) == 1

    def test_no_matching_files_errors(self, tmp_path: pathlib.Path) -> None:
        cosmo_dir = tmp_path / "empty"
        cosmo_dir.mkdir()
        with pytest.raises(SystemExit):
            store_main(
                [
                    "--storage-dir",
                    str(tmp_path / "store"),
                    "--cosmo-files-dir",
                    str(cosmo_dir),
                ]
            )
