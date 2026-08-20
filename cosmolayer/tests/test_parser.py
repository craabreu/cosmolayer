"""Unit tests for ``cosmolayer.parser.get_rdkit_molecule``."""

from importlib.resources import files

import pandas as pd
import pytest
from rdkit import Chem

from cosmolayer.parser import get_rdkit_molecule, parse_cosmo_file


def _atoms_df(name: str) -> pd.DataFrame:
    path = files("cosmolayer.data") / name
    contents = path.read_text(encoding="utf-8")
    _, atoms_df, _, _ = parse_cosmo_file(contents)
    return atoms_df


@pytest.fixture
def water_atoms() -> pd.DataFrame:
    return _atoms_df("O.cosmo")


@pytest.fixture
def fluoromethane_atoms() -> pd.DataFrame:
    return _atoms_df("CF.cosmo")


@pytest.fixture
def ammonium_atoms() -> pd.DataFrame:
    # Tetrahedral NH4+, net charge +1.
    return pd.DataFrame(
        {
            "element": ["N", "H", "H", "H", "H"],
            "x": [0.0, 0.0, 0.9428, -0.4714, -0.4714],
            "y": [0.0, 0.0, -0.3333, -0.3333, 0.6667],
            "z": [0.0, 1.0, -0.3333, -0.3333, -0.3333],
        }
    )


class TestGetRdkitMolecule:
    def test_returns_atom_mapped_smiles_for_water(
        self, water_atoms: pd.DataFrame
    ) -> None:
        mol = get_rdkit_molecule(water_atoms)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == "[O:1]([H:2])[H:3]"

    def test_returns_atom_mapped_smiles_for_fluoromethane(
        self, fluoromethane_atoms: pd.DataFrame
    ) -> None:
        mol = get_rdkit_molecule(fluoromethane_atoms)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == "[C:1]([F:2])([H:3])([H:4])[H:5]"

    def test_result_is_a_sanitized_mol(self, fluoromethane_atoms: pd.DataFrame) -> None:
        mol = get_rdkit_molecule(fluoromethane_atoms)
        assert mol is not None
        # Raises if the mol isn't sanitized/valid.
        Chem.SanitizeMol(mol)

    def test_atom_map_numbers_are_one_indexed_and_unique(
        self, fluoromethane_atoms: pd.DataFrame
    ) -> None:
        mol = get_rdkit_molecule(fluoromethane_atoms)
        assert mol is not None
        map_nums = sorted(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        assert map_nums == list(range(1, mol.GetNumAtoms() + 1))

    def test_atom_map_numbers_match_input_row_order(
        self, fluoromethane_atoms: pd.DataFrame
    ) -> None:
        mol = get_rdkit_molecule(fluoromethane_atoms)
        assert mol is not None
        elements_by_map_num = {
            atom.GetAtomMapNum(): atom.GetSymbol() for atom in mol.GetAtoms()
        }
        expected = dict(enumerate(fluoromethane_atoms["element"], start=1))
        assert elements_by_map_num == expected

    def test_single_atom_molecule(self) -> None:
        df = pd.DataFrame({"element": ["Ar"], "x": [0.0], "y": [0.0], "z": [0.0]})
        mol = get_rdkit_molecule(df)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == "[Ar:1]"

    def test_returns_none_for_unrecognized_element(self) -> None:
        df = pd.DataFrame({"element": ["Xx"], "x": [0.0], "y": [0.0], "z": [0.0]})
        assert get_rdkit_molecule(df) is None

    def test_returns_none_when_bond_perception_fails(self) -> None:
        # Two coincident atoms cannot be assigned a valid bond order/charge.
        df = pd.DataFrame(
            {"element": ["C", "C"], "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
        )
        assert get_rdkit_molecule(df) is None

    def test_returns_none_for_empty_dataframe(self) -> None:
        df = pd.DataFrame({"element": [], "x": [], "y": [], "z": []})
        assert get_rdkit_molecule(df) is None

    def test_raises_for_missing_required_columns(self) -> None:
        df = pd.DataFrame({"element": ["C"], "x": [0.0]})
        with pytest.raises(KeyError):
            get_rdkit_molecule(df)

    def test_default_charge_zero_fails_for_charged_species(
        self, ammonium_atoms: pd.DataFrame
    ) -> None:
        assert get_rdkit_molecule(ammonium_atoms, print_errors=False) is None

    def test_explicit_charge_succeeds_for_charged_species(
        self, ammonium_atoms: pd.DataFrame
    ) -> None:
        mol = get_rdkit_molecule(ammonium_atoms, charge=1)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == "[N+:1]([H:2])([H:3])([H:4])[H:5]"

    def test_prints_error_message_by_default(
        self, ammonium_atoms: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_rdkit_molecule(ammonium_atoms)
        assert "Error parsing molecule" in capsys.readouterr().out

    def test_print_errors_false_suppresses_message(
        self, ammonium_atoms: pd.DataFrame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        get_rdkit_molecule(ammonium_atoms, print_errors=False)
        assert capsys.readouterr().out == ""

    def test_print_errors_false_still_returns_none_on_failure(self) -> None:
        df = pd.DataFrame({"element": ["Xx"], "x": [0.0], "y": [0.0], "z": [0.0]})
        assert get_rdkit_molecule(df, print_errors=False) is None
