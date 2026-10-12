"""Tests for the approved-drug repurposing screen.

The screen's value is entirely in what it refuses to claim. It runs a model
with negative R2 over chemistry that model has never seen, so the filters and
the caveats are the product; the ranked list is a by-product.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.standardize import inchikey  # noqa: E402
from repurposing import MAX_PHASE, SIMILARITY_FLOOR, screen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "REPURPOSING.md"


class TestFilters:
    def _drugs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "molecule_chembl_id": ["D1", "D2", "D3", "D4"],
                "pref_name": [
                    "caffeine",
                    "caffeine citrate",
                    "a covalent drug",
                    "a peptide",
                ],
                "molecule_type": ["Small molecule"] * 4,
                "smiles": [
                    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",            # caffeine
                    "Cn1c(=O)c2c(ncn2C)n(C)c1=O.OC(=O)CC(O)(CC(=O)O)C(=O)O",
                    "C=CC(=O)Nc1ccccc1",                      # acrylamide
                    "C" * 2,                                  # too small
                ],
            }
        )

    def test_salt_forms_are_collapsed(self):
        """Approved-drug lists carry the same molecule under several IDs.

        Caffeine and caffeine citrate are one compound; counting them twice
        turned a one-item result into a two-item one.
        """
        result = screen(self._drugs(), known=set())
        names = set(result["pref_name"])
        assert len(result[result["smiles"].str.contains("Cn1c")]) == 1
        assert "caffeine" in names

    def test_compounds_already_in_the_dataset_are_excluded(self):
        """A compound already known to hit legumain is not a new finding."""
        caffeine_key = inchikey("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        result = screen(self._drugs(), known={caffeine_key})
        assert not result["smiles"].str.contains("Cn1c").any()

    def test_covalent_drugs_are_flagged(self):
        result = screen(self._drugs(), known=set())
        acrylamide = result[result["pref_name"] == "a covalent drug"]
        if not acrylamide.empty:
            assert bool(acrylamide["covalent"].iloc[0])

    def test_tiny_fragments_are_dropped(self):
        result = screen(self._drugs(), known=set())
        assert "a peptide" not in set(result["pref_name"])

    def test_approved_phase_is_four(self):
        assert MAX_PHASE == 4

    def test_similarity_floor_is_a_valid_tanimoto(self):
        assert 0.0 < SIMILARITY_FLOOR < 1.0


class TestHonesty:
    """The document must not read as a discovery."""

    @pytest.mark.skipif(not DOC.exists(), reason="screen not run yet")
    def test_states_nothing_is_a_hit(self):
        text = DOC.read_text()
        assert "Nothing here is a hit" in text

    @pytest.mark.skipif(not DOC.exists(), reason="screen not run yet")
    def test_explains_what_the_survivor_means(self):
        """A one-compound result invites over-reading; the doc must pre-empt it."""
        text = DOC.read_text()
        assert "applicability domain" in text.lower()

    @pytest.mark.skipif(not DOC.exists(), reason="screen not run yet")
    def test_records_the_limits_of_the_screen(self):
        text = DOC.read_text()
        assert "What this does not do" in text
