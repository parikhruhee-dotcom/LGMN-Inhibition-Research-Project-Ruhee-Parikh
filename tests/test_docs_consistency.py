"""Check that the hand-written README agrees with the generated results.

Everything in `docs/` is generated from the result files, so it cannot drift.
The README is written by hand and quotes the same numbers, which makes it the
one place where a stale figure can survive a rerun of the pipeline. A README
claiming 12 candidate compounds when the pipeline now finds 6 is the kind of
error that gets read and believed.

These tests skip when the generated files are absent, so a fresh checkout and
CI (where `data/` and `results/` are gitignored) still pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not generated yet")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def readme() -> str:
    if not README.exists():
        pytest.skip("README.md missing")
    return README.read_text()


def _claims(readme: str, pattern: str) -> list[str]:
    return re.findall(pattern, readme)


class TestDatasetClaims:
    def test_compound_count(self, readme):
        dataset = _load(PROCESSED / "lgmn_regression.csv")
        assert f"**{len(dataset)}**" in readme, (
            f"README should state {len(dataset)} compounds"
        )

    def test_scaffold_count(self, readme):
        dataset = _load(PROCESSED / "lgmn_regression.csv")
        assert f"{dataset['scaffold'].nunique()} scaffolds" in readme

    def test_covalent_count_and_percentage(self, readme):
        dataset = _load(PROCESSED / "lgmn_regression.csv")
        covalent = int(dataset["covalent"].sum())
        percent = round(100 * covalent / len(dataset))
        assert f"**{covalent} ({percent}%)**" in readme


class TestStructureClaims:
    def test_structure_counts(self, readme):
        survey = _load(PROCESSED / "target_survey.csv")
        total = survey["pdb_id"].nunique()
        mature = survey[survey["mature"].astype(bool)]["pdb_id"].nunique()
        assert f"{total} ({mature} mature" in readme

    def test_catalytic_dyad_residues(self, readme):
        """The dyad assignment is the project's most load-bearing fact."""
        survey = _load(PROCESSED / "target_survey.csv")
        dyads = survey.dropna(subset=["dyad_cys", "dyad_his"])
        if dyads.empty:
            pytest.skip("no dyads recorded")
        cys = int(dyads["dyad_cys"].mode().iloc[0])
        his = int(dyads["dyad_his"].mode().iloc[0])
        assert f"Cys{cys} / His{his}" in readme


class TestModelClaims:
    def test_reported_mae_matches_best_scaffold_model(self, readme):
        summary = _load(RESULTS / "models" / "summary.csv")
        scaffold = summary[summary["split"] == "scaffold"]
        learned = scaffold[scaffold["model"] != "mean_baseline"]
        if learned.empty:
            pytest.skip("no scaffold-split results")
        best = learned.sort_values("mae_mean").iloc[0]
        assert f"**{best['mae_mean']:.3f}**" in readme

    def test_reported_baseline_matches(self, readme):
        summary = _load(RESULTS / "models" / "summary.csv")
        scaffold = summary[summary["split"] == "scaffold"]
        baseline = scaffold[scaffold["model"] == "mean_baseline"]["mae_mean"]
        if baseline.empty:
            pytest.skip("no baseline recorded")
        assert f"{baseline.min():.3f}" in readme

    def test_negative_r2_is_still_disclosed(self, readme):
        """The unflattering number must not quietly disappear on an update."""
        summary = _load(RESULTS / "models" / "summary.csv")
        scaffold = summary[summary["split"] == "scaffold"]
        learned = scaffold[scaffold["model"] != "mean_baseline"]
        if learned.empty:
            pytest.skip("no scaffold-split results")
        best = learned.sort_values("mae_mean").iloc[0]
        if best["r2_mean"] < 0:
            assert "−0.19" in readme or f"{best['r2_mean']:.2f}" in readme, (
                "README must disclose the negative R2"
            )


class TestShortlistClaims:
    def test_cns_noncovalent_count(self, readme):
        shortlist = _load(RESULTS / "shortlist.csv")
        count = int(
            (
                shortlist["cns_candidate"].fillna(False).astype(bool)
                & ~shortlist["covalent"].fillna(False).astype(bool)
            ).sum()
        )
        assert f"**{count}**" in readme or f"Only {count} of" in readme

    def test_claim_of_one_series_is_not_dropped(self, readme):
        """The qualification matters as much as the count.

        Reporting 6 candidates without saying they are one chemical series
        overstates the result, so the README must keep both.
        """
        shortlist = RESULTS / "shortlist.csv"
        if not shortlist.exists():
            pytest.skip("shortlist not generated")
        assert "single chemical series" in readme or "one series" in readme


class TestPermeabilityClaims:
    def test_paired_compound_count(self, readme):
        paired = _load(PROCESSED / "permeability_gap.csv")
        assert f"n={len(paired)}" in readme

    def test_median_gap_and_fold(self, readme):
        paired = _load(PROCESSED / "permeability_gap.csv")
        median = paired["cell_gap"].median()
        assert f"{median:.2f} log units" in readme
        assert f"{10 ** median:.0f}-fold" in readme


class TestSelectivityClaims:
    def test_counter_screened_count(self, readme):
        selectivity = _load(PROCESSED / "selectivity.csv")
        dataset = _load(PROCESSED / "lgmn_regression.csv")
        tested = selectivity["molecule_chembl_id"].nunique()
        total = dataset["molecule_chembl_id"].nunique()
        assert f"**{tested} of {total}**" in readme

    def test_untested_majority_is_stated(self, readme):
        """Good-looking selectivity on 4 compounds must not stand alone."""
        if not (PROCESSED / "selectivity.csv").exists():
            pytest.skip("selectivity not generated")
        assert "untested" in readme.lower()


class TestHonestyMarkers:
    def test_prediction_disclaimer_present(self, readme):
        lowered = readme.lower()
        assert "computational prediction" in lowered or "predictions" in lowered
        assert "not been tested" in lowered or "no compound has been tested" in lowered

    def test_documentation_links_resolve(self, readme):
        for link in re.findall(r"\]\((docs/[^)]+)\)", readme):
            assert (ROOT / link).exists(), f"README links to missing {link}"
