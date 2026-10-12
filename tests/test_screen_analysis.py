"""Tests for the docking score versus potency analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking.analyse_screen import (  # noqa: E402
    ACTIVE_QUANTILE,
    correlation,
    enrichment,
    partial_spearman,
    representativeness,
)


def _frame(n: int = 40, seed: int = 0, working: bool = True) -> pd.DataFrame:
    """Scores where a lower (more negative) affinity means higher potency."""
    rng = np.random.default_rng(seed)
    pic50 = rng.uniform(5, 10, n)
    affinity = (-pic50 if working else pic50) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"affinity": affinity, "pic50": pic50})


class TestCorrelationDirection:
    def test_working_score_gives_negative_correlation(self):
        """A tighter predicted binding (more negative) must mean higher pIC50.

        The sign is the whole verdict, and it is easy to misread because a
        'better' docking score is a smaller number.
        """
        result = correlation(_frame(working=True), "test")
        assert result["spearman_rho"] < 0
        assert result["direction_correct"]

    def test_inverted_score_is_flagged_as_wrong_direction(self):
        result = correlation(_frame(working=False), "test")
        assert result["spearman_rho"] > 0
        assert not result["direction_correct"]

    def test_too_few_compounds_reports_a_note_not_a_number(self):
        """An unstable estimate must not be dressed up as a result."""
        result = correlation(_frame(n=5), "tiny")
        assert "spearman_rho" not in result
        assert "note" in result

    def test_both_measures_reported(self):
        result = correlation(_frame(), "test")
        for key in ("spearman_rho", "spearman_p", "pearson_r", "pearson_p"):
            assert key in result


class TestSizeControl:
    def test_pure_size_artefact_does_not_survive_the_control(self):
        """The confound this exists to catch.

        Vina-family scores grow more negative with ligand size and potency
        also rises with size here, so a score that only tracks size would
        produce a convincing raw correlation with potency. After controlling
        for size, nothing should remain.
        """
        rng = np.random.default_rng(1)
        size = rng.uniform(20, 45, 60)
        affinity = -0.15 * size + rng.normal(0, 0.2, 60)
        pic50 = 0.1 * size + rng.normal(0, 0.2, 60)
        rho, p = partial_spearman(affinity, pic50, size)
        assert abs(rho) < 0.35
        assert p > 0.01

    def test_genuine_relationship_survives_the_control(self):
        rng = np.random.default_rng(2)
        size = rng.uniform(20, 45, 60)
        pic50 = rng.uniform(5, 10, 60)
        # Affinity tracks potency directly, with only mild size dependence.
        affinity = -pic50 - 0.02 * size + rng.normal(0, 0.2, 60)
        rho, p = partial_spearman(affinity, pic50, size)
        assert rho < -0.5
        assert p < 0.01


class TestRepresentativeness:
    def test_potency_biased_subset_is_flagged(self):
        """A partial run of a potency-sorted list covers only the potent end."""
        dataset = pd.DataFrame({"pic50": np.linspace(5, 10, 180)})
        scored = pd.DataFrame({"pic50": np.linspace(9, 10, 30)})
        result = representativeness(scored, dataset)
        assert not result["representative"]
        assert result["median_gap"] > 0.5

    def test_representative_subset_passes(self):
        dataset = pd.DataFrame({"pic50": np.linspace(5, 10, 180)})
        scored = pd.DataFrame({"pic50": np.linspace(5, 10, 40)})
        assert representativeness(scored, dataset)["representative"]

    def test_empty_subset_returns_empty(self):
        dataset = pd.DataFrame({"pic50": [7.0, 8.0]})
        assert representativeness(pd.DataFrame({"pic50": []}), dataset) == {}


class TestEnrichment:
    def test_perfect_ranking_enriches(self):
        frame = pd.DataFrame(
            {"pic50": np.linspace(5, 10, 100), "affinity": np.linspace(-5, -10, 100)}
        )
        threshold = float(frame["pic50"].quantile(ACTIVE_QUANTILE))
        result = enrichment(frame, threshold)
        assert result["enrichment_factor"] > 2.0

    def test_random_ranking_gives_about_one(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {"pic50": rng.uniform(5, 10, 200), "affinity": rng.uniform(-10, -5, 200)}
        )
        threshold = float(frame["pic50"].quantile(ACTIVE_QUANTILE))
        result = enrichment(frame, threshold)
        assert 0.3 < result["enrichment_factor"] < 2.2

    def test_threshold_is_data_driven_not_fixed(self):
        """A fixed pIC50 7 cut is meaningless on a set of published inhibitors.

        88% of this dataset already clears it, which leaves no headroom for
        enrichment to mean anything.
        """
        assert 0.5 < ACTIVE_QUANTILE < 1.0

    def test_active_threshold_is_recorded_in_the_output(self):
        frame = _frame(n=60)
        threshold = float(frame["pic50"].quantile(ACTIVE_QUANTILE))
        result = enrichment(frame, threshold)
        assert result["active_threshold_pic50"] == pytest.approx(threshold)

    def test_too_few_compounds_returns_a_note(self):
        assert "note" in enrichment(_frame(n=4), 8.0)
