"""Tests for the ADMET and CNS-penetration filters.

The reference points are real drugs with known behaviour, because a
desirability function that is subtly inverted still produces plausible-looking
numbers for everything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from admet.filters import (  # noqa: E402
    _ramp,
    _tpsa_desirability,
    bbb_rules,
    cns_mpo,
    lipinski,
    profile,
    structural_alerts,
    veber,
)

# Diazepam: a CNS drug that must score well. Sucrose: highly polar, must not.
DIAZEPAM = "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"
SUCROSE = (
    "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
    "[C@@H](O)[C@@H]1O"
)
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class TestRamp:
    def test_decreasing_ramp(self):
        """good < bad means lower values are better."""
        assert _ramp(2.0, 3.0, 5.0) == pytest.approx(1.0)
        assert _ramp(5.0, 3.0, 5.0) == pytest.approx(0.0)
        assert _ramp(4.0, 3.0, 5.0) == pytest.approx(0.5)

    def test_increasing_ramp(self):
        """good > bad means higher values are better."""
        assert _ramp(5.0, 4.0, 2.0) == pytest.approx(1.0)
        assert _ramp(2.0, 4.0, 2.0) == pytest.approx(0.0)
        assert _ramp(3.0, 4.0, 2.0) == pytest.approx(0.5)

    def test_output_is_always_within_unit_interval(self):
        for value in (-100.0, 0.0, 3.7, 1000.0):
            assert 0.0 <= _ramp(value, 3.0, 5.0) <= 1.0


class TestTpsaDesirability:
    def test_optimal_window_scores_one(self):
        assert _tpsa_desirability(60.0) == pytest.approx(1.0)
        assert _tpsa_desirability(40.0) == pytest.approx(1.0)
        assert _tpsa_desirability(90.0) == pytest.approx(1.0)

    def test_too_polar_scores_zero(self):
        """Above ~120 A^2 passive brain penetration is not expected."""
        assert _tpsa_desirability(130.0) == pytest.approx(0.0)
        assert _tpsa_desirability(200.0) == pytest.approx(0.0)

    def test_too_greasy_scores_zero(self):
        assert _tpsa_desirability(10.0) == pytest.approx(0.0)

    def test_is_doubly_bounded_not_monotonic(self):
        """TPSA is the one property where both extremes are bad."""
        assert _tpsa_desirability(60.0) > _tpsa_desirability(10.0)
        assert _tpsa_desirability(60.0) > _tpsa_desirability(150.0)


class TestCnsMpo:
    def test_score_is_within_zero_to_six(self):
        for smiles in (DIAZEPAM, SUCROSE, ASPIRIN):
            score = cns_mpo(Chem.MolFromSmiles(smiles))["cns_mpo"]
            assert 0.0 <= score <= 6.0

    def test_cns_drug_outscores_a_sugar(self):
        """The single most important ordering this function must get right."""
        diazepam = cns_mpo(Chem.MolFromSmiles(DIAZEPAM))["cns_mpo"]
        sucrose = cns_mpo(Chem.MolFromSmiles(SUCROSE))["cns_mpo"]
        assert diazepam > sucrose

    def test_known_cns_drug_passes(self):
        assert cns_mpo(Chem.MolFromSmiles(DIAZEPAM))["cns_mpo_pass"]

    def test_sugar_reaches_four_on_mpo_alone(self):
        """Documents a real limitation of CNS MPO rather than asserting away.

        Sucrose scores exactly 4.0 and so "passes" the MPO threshold: the
        clogP, clogD and pKa desirability functions all reward low
        lipophilicity monotonically, so an extremely polar sugar collects
        three free points. MPO ranks drug-like candidates against each other;
        it was never meant to reject molecules nothing would mistake for a
        drug. This is why the combined `cns_candidate` verdict exists.
        """
        score = cns_mpo(Chem.MolFromSmiles(SUCROSE))
        assert score["cns_mpo"] == pytest.approx(4.0, abs=0.01)
        assert score["cns_mpo_pass"]

    def test_combined_verdict_rejects_the_sugar(self):
        """The criterion actually used for triage must reject sucrose."""
        assert not profile(SUCROSE)["cns_candidate"]
        assert profile(DIAZEPAM)["cns_candidate"]

    def test_all_six_components_are_present(self):
        result = cns_mpo(Chem.MolFromSmiles(DIAZEPAM))
        for key in (
            "mpo_clogp", "mpo_clogd", "mpo_mw",
            "mpo_tpsa", "mpo_hbd", "mpo_pka",
        ):
            assert key in result


class TestLipinskiAndVeber:
    def test_aspirin_has_no_violations(self):
        assert lipinski(Chem.MolFromSmiles(ASPIRIN))["lipinski_violations"] == 0

    def test_sucrose_violates_on_polarity(self):
        result = lipinski(Chem.MolFromSmiles(SUCROSE))
        assert result["lipinski_violations"] >= 1
        assert result["hbd"] > 5

    def test_veber_rejects_a_very_polar_molecule(self):
        assert not veber(Chem.MolFromSmiles(SUCROSE))["veber_pass"]

    def test_veber_accepts_a_small_drug(self):
        assert veber(Chem.MolFromSmiles(ASPIRIN))["veber_pass"]


class TestBbbRules:
    def test_cns_drug_passes_most_rules(self):
        rules = bbb_rules(Chem.MolFromSmiles(DIAZEPAM))
        assert sum(bool(v) for v in rules.values()) >= 5

    def test_sugar_fails_most_rules(self):
        rules = bbb_rules(Chem.MolFromSmiles(SUCROSE))
        assert sum(bool(v) for v in rules.values()) <= 3

    def test_charged_species_is_flagged(self):
        """A permanent cation will not cross passively."""
        rules = bbb_rules(Chem.MolFromSmiles("C[N+](C)(C)C"))
        assert not rules["bbb_neutral"]


class TestStructuralAlerts:
    def test_benign_molecule_has_no_pains(self):
        assert structural_alerts(Chem.MolFromSmiles(ASPIRIN))["pains_alerts"] == 0

    def test_catechol_triggers_an_alert(self):
        """Catechols are a classic PAINS/Brenk motif (redox cycling)."""
        result = structural_alerts(Chem.MolFromSmiles("Oc1ccccc1O"))
        assert result["pains_alerts"] + result["brenk_alerts"] > 0


class TestProfile:
    def test_profile_returns_none_for_invalid_smiles(self):
        assert profile("not_a_molecule") is None

    def test_profile_contains_every_section(self):
        result = profile(DIAZEPAM)
        assert result is not None
        for key in (
            "mw", "clogp", "lipinski_violations", "veber_pass",
            "cns_mpo", "cns_mpo_pass", "pains_alerts", "qed",
            "bbb_rules_passed",
        ):
            assert key in result

    def test_bbb_rule_tally_is_consistent(self):
        result = profile(DIAZEPAM)
        assert 0 <= result["bbb_rules_passed"] <= result["bbb_rules_total"]
