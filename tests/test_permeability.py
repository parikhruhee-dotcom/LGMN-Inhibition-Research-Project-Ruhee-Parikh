"""Tests for the enzymatic-to-cellular permeability analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.permeability import (  # noqa: E402
    CANDIDATE_PROPERTIES,
    LARGE_GAP,
    correlate,
    translation,
)


def _paired(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """A synthetic paired table where mw drives the gap and clogp does not."""
    rng = np.random.default_rng(seed)
    mw = rng.uniform(300, 600, n)
    gap = (mw - 300) / 150 + rng.normal(0, 0.15, n)
    enzyme = rng.uniform(7, 9.5, n)
    return pd.DataFrame(
        {
            "mw": mw,
            "clogp": rng.uniform(1, 4, n),
            "tpsa": rng.uniform(80, 160, n),
            "hbd": np.full(n, 2.0),          # constant on purpose
            "hba": rng.uniform(4, 10, n),
            "rotb": rng.uniform(3, 12, n),
            "fsp3": rng.uniform(0.2, 0.6, n),
            "pic50_enzyme": enzyme,
            "pic50_cell": enzyme - gap,
            "cell_gap": gap,
        }
    )


class TestGapConvention:
    def test_positive_gap_means_weaker_in_cells(self):
        """Sign convention: the gap must be enzyme minus cell.

        Getting this backwards would invert every conclusion in the analysis
        while leaving plausible-looking magnitudes.
        """
        enzyme, cell = 9.0, 7.0  # 100x weaker in cells
        gap = enzyme - cell
        assert gap > 0
        assert 10 ** gap == pytest.approx(100.0)

    def test_large_gap_threshold_is_tenfold(self):
        assert pytest.approx(10.0) == 10 ** LARGE_GAP


class TestCorrelate:
    def test_finds_the_driving_property(self):
        table = correlate(_paired())
        assert table.iloc[0]["property"] == "mw"
        assert table.iloc[0]["spearman_rho"] > 0.8
        assert bool(table.iloc[0]["significant"])

    def test_constant_property_is_skipped(self):
        """A zero-variance descriptor yields NaN from spearmanr.

        These compounds come largely from one series, so constant descriptors
        are expected; they must be dropped rather than reported as NaN rows.
        """
        table = correlate(_paired())
        assert "hbd" not in set(table["property"])

    def test_holm_adjustment_is_monotonic_and_not_smaller_than_raw(self):
        table = correlate(_paired())
        assert (table["p_holm"].to_numpy() >= table["p_value"].to_numpy() - 1e-12).all()
        assert (np.diff(table["p_holm"].to_numpy()) >= -1e-12).all()

    def test_holm_is_applied_not_just_reported(self):
        """A property significant raw but not after correction must be marked.

        Without correction, testing seven properties on 35 compounds makes a
        spurious hit likely.
        """
        table = correlate(_paired())
        borderline = table[
            (table["p_value"] < 0.05) & (table["p_holm"] >= 0.05)
        ]
        for row in borderline.itertuples(index=False):
            assert not row.significant

    def test_too_few_rows_returns_empty(self):
        assert correlate(_paired(n=5)).empty

    def test_all_candidate_properties_are_real_columns(self):
        """Guards against a typo silently dropping a property from testing."""
        frame = _paired()
        for prop in CANDIDATE_PROPERTIES:
            assert prop in frame.columns


class TestTranslation:
    def test_perfect_translation_gives_unit_slope(self):
        frame = _paired()
        frame["pic50_cell"] = frame["pic50_enzyme"]
        assert translation(frame)["slope"] == pytest.approx(1.0, abs=1e-6)

    def test_no_translation_gives_near_zero_slope(self):
        frame = _paired()
        frame["pic50_cell"] = 7.0
        assert translation(frame)["slope"] == pytest.approx(0.0, abs=1e-6)

    def test_slope_is_between_zero_and_one_for_attenuated_data(self):
        result = translation(_paired())
        assert 0.0 < result["slope"] < 1.0

    def test_reports_both_correlation_measures(self):
        """Spearman and Pearson disagreed on the real data (0.29 vs 0.39).

        Reporting only the more favourable one would misrepresent how weak
        the relationship is, so both must be present.
        """
        result = translation(_paired())
        for key in ("spearman_rho", "spearman_p", "pearson_r", "pearson_p"):
            assert key in result

    def test_too_few_rows_returns_empty(self):
        assert translation(_paired(n=5)) == {}
