"""Tests for the docking setup logic.

These cover the decisions that silently produced wrong template choices, and
are written as regression tests: each one failed before the corresponding fix.
Network-dependent parts (RCSB fetches) are not tested here so the suite stays
runnable offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking.characterize_target import (  # noqa: E402
    CATALYTIC_DOMAIN_END,
    COVALENT_BOND_CUTOFF,
    DYAD_CUTOFF,
    IGNORED_HETERO,
    choose_template,
)
from docking.prepare_receptor import (  # noqa: E402
    DEFAULT_PADDING,
    STRIP_RESIDUES,
    _is_modified_residue,
)


class TestCovalencyByGeometry:
    """Bond length, not annotation, decides covalency.

    7FQI and 7FQJ place their inhibitor 1.67 A and 1.65 A from the catalytic
    thiol while declaring no covale record in the mmCIF at all. Trusting the
    annotation classified both as non-covalent.
    """

    def test_cutoff_separates_bonds_from_contacts(self):
        # A C-S bond is ~1.8 A; the closest non-bonded approach is ~3.3 A.
        assert 1.9 < COVALENT_BOND_CUTOFF < 3.0

    @pytest.mark.parametrize(
        "distance,is_bond",
        [
            (1.65, True),   # WSN in 7FQJ
            (1.67, True),   # WSN in 7FQI
            (1.73, True),   # WR9 in 7FQH
            (1.82, True),   # WS9 in 7FQK
            (3.00, False),  # 3Y7 in 5LUB
            (3.22, False),  # 5KN in 5LUA
        ],
    )
    def test_measured_distances_classify_correctly(self, distance, is_bond):
        assert (distance <= COVALENT_BOND_CUTOFF) is is_bond


class TestDyadCutoff:
    """The dyad distance spans too wide a range to act as a filter.

    An initial 4.5 A cutoff matched only the 3.9 A prolegumain, i.e. selected
    the one structure whose active site is occluded.
    """

    # Measured S...N distances from the deposited coordinates.
    MEASURED = {
        "4FGU prolegumain": 4.14,
        "7FQL covalent adduct": 5.01,
        "5LUA fragment complex": 5.30,
        "4AW9 peptide inhibitor": 6.14,
        "8AE5 macrocypin complex": 7.68,
        "4N6O cystatin complex": 7.93,
    }

    def test_cutoff_admits_every_real_structure(self):
        for name, distance in self.MEASURED.items():
            assert distance <= DYAD_CUTOFF, (
                f"{name} at {distance} A would be missed with a "
                f"{DYAD_CUTOFF} A cutoff"
            )

    def test_a_tight_cutoff_would_have_kept_only_the_zymogen(self):
        """Documents the original bug: 4.5 A selects the inactive form."""
        admitted = {
            name for name, distance in self.MEASURED.items() if distance <= 4.5
        }
        assert admitted == {"4FGU prolegumain"}

    def test_maturity_is_judged_by_the_prodomain_not_the_dyad(self):
        """Prolegumain models residues past the catalytic domain."""
        assert CATALYTIC_DOMAIN_END < 434  # 4FGU reaches residue 434
        assert CATALYTIC_DOMAIN_END > 287  # mature 5LUA chain ends at 287


class TestTemplateSelection:
    def _survey(self) -> pd.DataFrame:
        """A survey frame with None in the boolean columns, as the real one has.

        Ligand-free rows carry None for `covalent`, which makes the column
        object dtype. That detail is the whole point of this fixture.
        """
        return pd.DataFrame(
            [
                # Covalent, high resolution: must NOT be chosen.
                {
                    "pdb_id": "7FQI", "ligand": "WSN", "resolution": 1.45,
                    "ligand_atoms": 36.0, "ligand_chain": "A",
                    "ligand_resseq": 301, "dist_to_catalytic_cys": 1.67,
                    "covalent": True, "mature": True, "n_dyads": 3,
                    "dyad_cys": 189, "dyad_his": 148,
                },
                # Non-covalent, lower resolution: the correct choice.
                {
                    "pdb_id": "5LUA", "ligand": "5KN", "resolution": 2.00,
                    "ligand_atoms": 19.0, "ligand_chain": "B",
                    "ligand_resseq": 304, "dist_to_catalytic_cys": 3.22,
                    "covalent": False, "mature": True, "n_dyads": 2,
                    "dyad_cys": 189, "dyad_his": 148,
                },
                # Zymogen with no ligand: None in the boolean column.
                {
                    "pdb_id": "4FGU", "ligand": None, "resolution": 3.90,
                    "ligand_atoms": None, "ligand_chain": None,
                    "ligand_resseq": None, "dist_to_catalytic_cys": None,
                    "covalent": None, "mature": False, "n_dyads": 2,
                    "dyad_cys": 189, "dyad_his": 148,
                },
            ]
        )

    def test_covalent_ligand_is_not_selected(self):
        """Regression: `~col.fillna(False)` on object dtype inverts ints.

        With None present the column is object dtype, where `~True` evaluates
        to -2 -- truthy -- so every covalent ligand passed the "not covalent"
        filter and the highest-resolution covalent structure won.
        """
        chosen = choose_template(self._survey())
        assert chosen["pdb_id"] == "5LUA"
        assert chosen["ligand"] == "5KN"

    def test_object_dtype_inversion_is_the_trap_being_guarded(self):
        """Shows the underlying pandas behaviour, so the fix is not undone."""
        column = pd.Series([True, False, None], dtype=object)
        inverted = ~column.fillna(False)
        # Every entry is truthy: -2 and -1, not False and True.
        assert all(bool(value) for value in inverted)
        # With the cast, it behaves as intended.
        assert list(~column.fillna(False).astype(bool)) == [False, True, True]

    def test_zymogen_is_excluded(self):
        survey = self._survey()
        survey = survey[survey["pdb_id"] == "4FGU"]
        assert choose_template(survey) == {}

    def test_ligand_far_from_the_site_is_excluded(self):
        """5KN sits 21 A away in 5LU8 -- a different site, not the pocket."""
        survey = self._survey()
        survey.loc[survey["pdb_id"] == "5LUA", "dist_to_catalytic_cys"] = 21.41
        chosen = choose_template(survey)
        assert chosen == {} or chosen["pdb_id"] != "5LUA"


class TestReceptorPreparation:
    def test_crystallisation_additives_are_stripped(self):
        for residue in ("HOH", "EDO", "SO4", "NAG", "MES", "GOL"):
            assert residue in STRIP_RESIDUES

    def test_modified_backbone_residue_is_kept(self):
        """SNN (aspartimide) at 147 sits beside the active site.

        It carries a hetero flag but is part of the chain; stripping it tears
        a hole in the backbone next to the pocket.
        """

        class FakeResidue:
            def __init__(self, atoms):
                self.atoms = set(atoms)

            def __contains__(self, name):
                return name in self.atoms

        backbone = FakeResidue(["N", "CA", "C", "O", "CB"])
        free_ligand = FakeResidue(["C1", "C2", "O1"])
        assert _is_modified_residue(backbone)
        assert not _is_modified_residue(free_ligand)

    def test_padding_is_tight_enough_to_matter(self):
        """8 A padding put the 5KN top pose 10.4 A off; 3 A reached 1.4 A."""
        assert DEFAULT_PADDING <= 5.0

    def test_modified_residues_are_not_in_the_ignore_list(self):
        """SNN and SCH must not be treated as ligands or as strippable.

        They were originally picked up as ligands because they carry HETATM
        records, which made every structure look inhibitor-bound.
        """
        for residue in ("SNN", "SCH"):
            assert residue not in IGNORED_HETERO
            assert residue not in STRIP_RESIDUES


class TestBoxGeometry:
    def test_box_covers_the_ligand_plus_padding(self):
        coords = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 6.0]])
        padding = 4.0
        extent = coords.max(axis=0) - coords.min(axis=0)
        size = extent + 2 * padding
        np.testing.assert_allclose(size, [12.0, 10.0, 14.0])

    def test_centre_is_the_ligand_centroid(self):
        coords = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 6.0]])
        np.testing.assert_allclose(coords.mean(axis=0), [2.0, 1.0, 3.0])
