"""Tests for the candidate dossier.

The dossier is the document someone acts on, so its job is to make the
weaknesses of each compound as visible as its strengths. These check that it
does not quietly omit them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_candidates import ACTIVE_SITE_CONTACT, liabilities  # noqa: E402


def _row(**overrides) -> pd.Series:
    base = {
        "covalent": False,
        "warheads": "",
        "brenk_alerts": 0,
        "brenk_names": "",
        "pains_alerts": 0,
        "cns_mpo": 5.5,
        "cns_mpo_pass": True,
        "bbb_rules_passed": 6,
        "bbb_rules_total": 6,
        "lipinski_violations": 0,
        "dist_to_thiol": 3.0,
        "n_measurements": 3,
    }
    base.update(overrides)
    return pd.Series(base)


class TestLiabilities:
    def test_clean_compound_has_none(self):
        assert liabilities(_row()) == []

    def test_covalent_warhead_is_reported(self):
        issues = liabilities(_row(covalent=True, warheads="nitrile"))
        assert any("covalent" in issue for issue in issues)
        assert any("nitrile" in issue for issue in issues)

    def test_structural_alert_is_named_not_just_counted(self):
        """"One alert" is useless; the reader needs to know which."""
        issues = liabilities(_row(brenk_alerts=1, brenk_names="aniline"))
        assert any("aniline" in issue for issue in issues)

    def test_pose_outside_the_active_site_is_flagged(self):
        """A good score from the wrong pocket is not evidence of inhibition.

        Four of the six real candidates dock 7-9 A from the catalytic thiol,
        so this is the most consequential caveat in the dossier.
        """
        issues = liabilities(_row(dist_to_thiol=8.5))
        assert any("catalytic thiol" in issue for issue in issues)

    def test_pose_inside_the_active_site_is_not_flagged(self):
        issues = liabilities(_row(dist_to_thiol=ACTIVE_SITE_CONTACT - 0.1))
        assert not any("catalytic thiol" in issue for issue in issues)

    def test_single_measurement_is_flagged(self):
        """Potency from one paper is weaker evidence than potency from three."""
        issues = liabilities(_row(n_measurements=1))
        assert any("single published measurement" in issue for issue in issues)

    def test_failing_cns_score_is_flagged(self):
        issues = liabilities(_row(cns_mpo=3.1, cns_mpo_pass=False))
        assert any("CNS MPO" in issue for issue in issues)

    def test_partial_bbb_compliance_is_flagged(self):
        issues = liabilities(_row(bbb_rules_passed=3))
        assert any("blood-brain barrier" in issue for issue in issues)

    def test_multiple_problems_are_all_reported(self):
        """Reporting only the worst problem would understate the risk."""
        issues = liabilities(
            _row(
                covalent=True,
                warheads="nitrile",
                brenk_alerts=1,
                brenk_names="aniline",
                dist_to_thiol=9.0,
                n_measurements=1,
            )
        )
        assert len(issues) >= 4


class TestDossierOutput:
    DOSSIER = Path(__file__).resolve().parents[1] / "docs" / "CANDIDATES.md"

    @pytest.mark.skipif(
        not DOSSIER.exists(), reason="dossier not generated yet"
    )
    def test_states_that_nothing_was_tested(self):
        text = self.DOSSIER.read_text()
        assert "has been tested by this project" in text

    @pytest.mark.skipif(
        not DOSSIER.exists(), reason="dossier not generated yet"
    )
    def test_every_candidate_lists_its_liabilities(self):
        text = self.DOSSIER.read_text()
        headings = text.count("\n## ")
        assert text.count("**Liabilities**") >= headings - 2
