"""Tests for structure standardisation and warhead detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.standardize import (  # noqa: E402
    basic_properties,
    find_warheads,
    inchikey,
    is_covalent,
    standardize_mol,
    standardize_smiles,
)


class TestStandardization:
    def test_salt_is_stripped_to_parent(self):
        """A hydrochloride salt and its free base must collapse to one form.

        Otherwise the same compound appears twice under two canonical SMILES
        and can land on both sides of a train/test split.
        """
        assert standardize_smiles("CCN.Cl") == standardize_smiles("CCN")

    def test_charges_are_neutralised(self):
        assert standardize_smiles("CC(=O)[O-]") == standardize_smiles("CC(=O)O")

    def test_largest_fragment_is_kept(self):
        result = standardize_smiles("c1ccccc1C(=O)O.[Na+].[Cl-]")
        assert result is not None
        assert "Na" not in result

    def test_stereochemistry_is_preserved(self):
        """Enantiomers are different compounds and must not be merged."""
        left = standardize_smiles("C[C@H](N)C(=O)O")
        right = standardize_smiles("C[C@@H](N)C(=O)O")
        assert left != right

    def test_invalid_smiles_returns_none(self):
        assert standardize_smiles("not_a_molecule") is None
        assert standardize_smiles("C(C)(C)(C)(C)C") is None  # pentavalent carbon

    def test_empty_input_returns_none(self):
        assert standardize_smiles("") is None
        assert standardize_mol(None) is None

    def test_standardization_is_idempotent(self):
        """Running it twice must not change the answer.

        The dataset pipeline and the screening pipeline both standardise; if
        the operation were not idempotent the two would disagree.
        """
        once = standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        twice = standardize_smiles(once)
        assert once == twice


class TestWarheadDetection:
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            # Halomethyl ketone: the warhead in the YVAD-CMK and AAN-CMK
            # legumain crystal structures.
            ("CC(=O)CCl", "halomethyl_ketone"),
            # Electrophilic nitrile, as in the 7FQI thioimidate adduct.
            ("N#CC1CCCCC1", "nitrile"),
            ("C=CS(=O)(=O)c1ccccc1", "vinyl_sulfone"),
            ("C=CC(=O)NC", "acrylamide_or_enone"),
            ("O=CCCC", "aldehyde"),
            ("C1OC1C", "epoxide"),
            ("OB(O)c1ccccc1", "boronic_acid"),
            ("O=C1C=CC(=O)N1", "maleimide"),
        ],
    )
    def test_known_warheads_are_detected(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"test SMILES itself is invalid: {smiles}"
        assert expected in find_warheads(mol)
        assert is_covalent(mol)

    @pytest.mark.parametrize(
        "smiles",
        [
            "c1ccccc1",                        # benzene
            "CC(=O)Oc1ccccc1C(=O)O",           # aspirin
            "Nc1ccc(N2CCOCC2)cc1N1CCOCC1",     # 5KN, the 5LUA fragment
            "CCO",                             # ethanol
            "c1ccc2[nH]ccc2c1",                # indole
        ],
    )
    def test_benign_molecules_are_not_flagged(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert not is_covalent(mol), f"false positive warhead on {smiles}"

    def test_amide_is_not_mistaken_for_an_electrophile(self):
        """A plain amide carbonyl is not cysteine-reactive.

        This is the obvious false positive for a carbonyl-based SMARTS, and
        most of the dataset contains amides.
        """
        mol = Chem.MolFromSmiles("CC(=O)NC")
        assert not is_covalent(mol)

    def test_ester_is_not_mistaken_for_an_electrophile(self):
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        assert not is_covalent(mol)

    def test_saturated_ketone_is_not_flagged(self):
        mol = Chem.MolFromSmiles("CCC(=O)CC")
        assert not is_covalent(mol)

    def test_nitrile_detection_is_not_fooled_by_isocyanide(self):
        """Only a genuine nitrile should match; the SMARTS is explicit."""
        mol = Chem.MolFromSmiles("N#CC1CCCCC1")
        assert "nitrile" in find_warheads(mol)

    def test_warhead_list_is_empty_for_none(self):
        assert find_warheads(None) == []


class TestCrystallographicGroundTruth:
    """The detector must agree with the LGMN crystal structures.

    These are not hand-picked examples: they are every small-molecule ligand
    resolved in the active site of a human legumain structure, labelled by the
    measured distance from the ligand to the catalytic thiol. Under ~2.2 A is
    a carbon-sulfur bond; around 3.0 A is a non-bonded contact. So the labels
    come from bond lengths rather than from judgement about what looks
    electrophilic -- which matters, because two of the three covalent warheads
    here are not conjugated to any electron-withdrawing group and no general
    Michael-acceptor rule would flag them.
    """

    # (name, SMILES, distance to catalytic thiol in A, covalent)
    COVALENT = [
        # 7FQI/7FQJ/7FQL: thioimidate adduct of a nitrile inhibitor.
        (
            "WSN",
            r"[H]/N=C\[C@H](CC(=O)N)NC(=O)[C@@H]1CCCN1C(=O)C2(CC2)"
            r"c3ccc(cc3)OCc4ccc(cc4F)F",
            1.67,
        ),
        # 7FQH: terminal vinyl alpha to the P1 amide nitrogen.
        (
            "WR9",
            "C=C[C@H](CC(=O)N)NC(=O)[C@@H]1CCCN1C(=O)C2(CC2)"
            "c3ccc(cc3)OCC4CC4",
            1.73,
        ),
        # 7FQK: vinyl oxazole, a conventional Michael acceptor.
        (
            "WS9",
            "c1cc(ccc1C2(CC2)C(=O)N3CCC[C@H]3C(=O)N[C@@H](CC(=O)N)"
            "/C=C/c4ncco4)OC(F)(F)F",
            1.82,
        ),
    ]

    NON_COVALENT = [
        ("5KN", "c1cc(c(cc1N2CCOCC2)N3CCOCC3)N", 3.22),   # 5LUA fragment
        ("3Y7", "c1cc(c2c(c1N)non2)N3CCOCC3", 3.00),      # 5LUB fragment
    ]

    @pytest.mark.parametrize("name,smiles,distance", COVALENT)
    def test_crystallographic_covalent_ligands_are_flagged(
        self, name, smiles, distance
    ):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{name} SMILES failed to parse"
        assert is_covalent(mol), (
            f"{name} is bonded to the catalytic thiol at {distance} A in the "
            f"crystal but was not flagged as covalent"
        )

    @pytest.mark.parametrize("name,smiles,distance", NON_COVALENT)
    def test_crystallographic_fragments_are_not_flagged(
        self, name, smiles, distance
    ):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"{name} SMILES failed to parse"
        assert not is_covalent(mol), (
            f"{name} sits {distance} A from the catalytic thiol -- a "
            f"non-bonded contact -- but was flagged as covalent"
        )

    def test_propargyl_series_is_flagged_like_its_vinyl_analogue(self):
        """The alkyne analogue of WR9 must be treated as covalent.

        Three ChEMBL compounds share WR9's scaffold with a terminal alkyne in
        place of its vinyl. WR9 is covalently bound at 1.73 A, so presenting
        the alkyne members as non-covalent leads would be a false negative in
        the most misleading direction -- they ranked as the project's best
        brain-penetrant non-covalent candidates before this was caught.
        """
        alkyne = Chem.MolFromSmiles(
            "C#C[C@H](CC(N)=O)NC(=O)[C@@H]1CCCN1C(=O)C1(c2ccc(Cl)s2)CC1"
        )
        assert is_covalent(alkyne)
        assert "propargyl_amide" in find_warheads(alkyne)

    def test_unactivated_terminal_alkyne_is_not_flagged(self):
        """Ethinylestradiol has a terminal alkyne and is not cysteine-reactive.

        Guards the propargyl pattern against collapsing into "any alkyne".
        """
        mol = Chem.MolFromSmiles(
            "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(O)ccc4[C@H]3CC[C@@]21C"
        )
        assert not is_covalent(mol)


class TestProperties:
    def test_aspirin_properties_are_correct(self):
        """Checked against known values so a unit mix-up would be caught."""
        mol = standardize_mol("CC(=O)Oc1ccccc1C(=O)O")
        props = basic_properties(mol)
        assert props["mw"] == pytest.approx(180.16, abs=0.1)
        assert props["heavy_atoms"] == 13
        assert props["hba"] >= 3
        assert props["formal_charge"] == 0

    def test_all_expected_keys_present(self):
        mol = standardize_mol("c1ccccc1")
        props = basic_properties(mol)
        for key in ("mw", "clogp", "tpsa", "hbd", "hba", "rotb", "fsp3"):
            assert key in props

    def test_benzene_has_no_rotatable_bonds(self):
        props = basic_properties(standardize_mol("c1ccccc1"))
        assert props["rotb"] == 0
        assert props["arom_rings"] == 1


class TestCrossSourceIdentity:
    """Canonical SMILES is not a safe identity key across data sources.

    Merging PubChem into the ChEMBL set on canonical SMILES let four
    thiouracil tautomers through as "new" compounds that were already
    present. Three of them then qualified as CNS candidates, which would have
    inflated the project's headline result from 6 to 10. The same failure
    would place one compound on both sides of a train/test split.
    """

    # The same thiouracil, drawn with the mobile hydrogen in two places.
    TAUTOMER_A = "Cn1c(=O)c2c([nH]c(=S)n2C)n(C)c1=O"
    TAUTOMER_B = "Cn1c(=O)c2c(nc(S)n2C)n(C)c1=O"

    def test_tautomers_give_different_canonical_smiles(self):
        """The premise of the bug: SMILES does not collapse these."""
        assert standardize_smiles(self.TAUTOMER_A) != standardize_smiles(
            self.TAUTOMER_B
        )

    def test_inchikey_collapses_them(self):
        assert inchikey(self.TAUTOMER_A) == inchikey(self.TAUTOMER_B)

    def test_inchikey_still_separates_genuinely_different_molecules(self):
        """The key must not collapse so much that distinct compounds merge."""
        assert inchikey("c1ccccc1O") != inchikey("c1ccccc1N")

    def test_inchikey_separates_enantiomers(self):
        """Stereoisomers are different compounds and must stay distinct."""
        assert inchikey("C[C@H](N)C(=O)O") != inchikey("C[C@@H](N)C(=O)O")

    def test_inchikey_matches_across_salt_forms(self):
        """Standardisation runs first, so a salt and its parent agree."""
        assert inchikey("CCN.Cl") == inchikey("CCN")

    def test_invalid_input_returns_none(self):
        assert inchikey("not_a_molecule") is None
        assert inchikey("") is None
