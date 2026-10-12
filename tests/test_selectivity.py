"""Tests for the selectivity analysis against related cysteine proteases."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.selectivity import (  # noqa: E402
    COUNTER_TARGETS,
    SELECTIVITY_CONCERN,
    build_selectivity,
)


def _lgmn() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_chembl_id": ["CMPD_A", "CMPD_B"],
            "pic50": [8.0, 7.0],
            "covalent": [True, False],
            "warheads": ["nitrile", ""],
            "smiles": ["N#CCC(=O)N", "c1ccccc1N"],
        }
    )


def _offtarget(relation: str = "=", value: float = 10000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule_chembl_id": ["CMPD_A"],
            "canonical_smiles": ["N#CCC(=O)N"],
            "standard_type": ["IC50"],
            "standard_relation": [relation],
            "standard_value": [value],
            "standard_units": ["nM"],
            "assay_chembl_id": ["CHEMBL_X"],
            "target_chembl_id": ["CHEMBL2334"],
            "data_validity_comment": [None],
            "off_target": ["Caspase-3"],
            "clan": ["CD"],
        }
    )


class TestCounterTargetPanel:
    def test_includes_caspases_and_cathepsins(self):
        names = {name for _, name, _ in COUNTER_TARGETS}
        assert "Caspase-3" in names   # executioner protease of apoptosis
        assert "Cathepsin B" in names

    def test_clans_are_labelled(self):
        """Clan CD shares legumain's fold; CA shares only the cysteine.

        The distinction is the point of the panel, so every entry must carry
        it and both must be represented.
        """
        clans = {clan for _, _, clan in COUNTER_TARGETS}
        assert clans == {"CD", "CA"}

    def test_accessions_look_like_uniprot(self):
        for accession, _, _ in COUNTER_TARGETS:
            assert 6 <= len(accession) <= 10
            assert accession[0].isalpha()

    def test_no_duplicate_targets(self):
        accessions = [a for a, _, _ in COUNTER_TARGETS]
        assert len(accessions) == len(set(accessions))


class TestSelectivityCalculation:
    def test_sign_convention_positive_means_prefers_legumain(self):
        """Getting this backwards would label selective compounds as liabilities."""
        result = build_selectivity(_lgmn(), _offtarget())
        row = result.iloc[0]
        # pIC50 8.0 on legumain vs 5.0 (10000 nM) off-target = 3 logs selective
        assert row["selectivity"] == pytest.approx(3.0)
        assert row["fold_selectivity"] == pytest.approx(1000.0)
        assert not row["concern"]

    def test_poor_selectivity_is_flagged(self):
        # Legumain pIC50 8.0 is 10 nM; an off-target at 50 nM is 5-fold,
        # inside the concern threshold.
        result = build_selectivity(_lgmn(), _offtarget(value=50.0))
        assert result.iloc[0]["selectivity"] < SELECTIVITY_CONCERN
        assert bool(result.iloc[0]["concern"])

    def test_exactly_tenfold_is_not_flagged(self):
        """Pins the boundary. 10 nM vs 100 nM is exactly 1.0 log.

        The threshold is strict (`< 1.0`), so exactly ten-fold passes. Worth
        fixing in a test because it is the kind of edge that gets silently
        flipped when someone 'tidies' a comparison operator.
        """
        result = build_selectivity(_lgmn(), _offtarget(value=100.0))
        assert result.iloc[0]["selectivity"] == pytest.approx(1.0)
        assert not bool(result.iloc[0]["concern"])

    def test_greater_than_records_are_kept_as_evidence_for_selectivity(self):
        """'>' means no inhibition up to the top concentration tested.

        That is evidence *for* selectivity, so unlike the potency pipeline --
        where a censored value would bias the label toward the detection
        limit -- these records are kept and flagged rather than discarded.
        """
        result = build_selectivity(_lgmn(), _offtarget(relation=">"))
        assert len(result) == 1
        assert bool(result.iloc[0]["any_censored"])

    def test_exact_records_are_not_marked_censored(self):
        result = build_selectivity(_lgmn(), _offtarget(relation="="))
        assert not bool(result.iloc[0]["any_censored"])

    def test_less_than_records_are_excluded(self):
        """'<' on an off-target means it was MORE potent than the limit.

        That is an upper bound in the unhelpful direction and cannot support
        a selectivity ratio.
        """
        assert build_selectivity(_lgmn(), _offtarget(relation="<")).empty

    def test_non_nanomolar_units_are_excluded(self):
        off = _offtarget()
        off["standard_units"] = ["uM"]
        assert build_selectivity(_lgmn(), off).empty

    def test_flagged_records_are_excluded(self):
        off = _offtarget()
        off["data_validity_comment"] = ["Outside typical range"]
        assert build_selectivity(_lgmn(), off).empty

    def test_compounds_without_counter_screens_are_absent_not_assumed_safe(self):
        """CMPD_B has no off-target data and must simply not appear.

        Silently treating it as selective would be the dangerous failure.
        """
        result = build_selectivity(_lgmn(), _offtarget())
        assert "CMPD_B" not in set(result["molecule_chembl_id"])

    def test_empty_offtarget_data_returns_empty(self):
        assert build_selectivity(_lgmn(), pd.DataFrame()).empty
