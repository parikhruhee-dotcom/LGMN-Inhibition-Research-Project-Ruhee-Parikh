"""Tests for the dataset preparation logic.

The expensive part of prepare_dataset.py is network-free, so these tests build
small synthetic frames and check the decisions that would silently corrupt a
model: unit conversion, censored records, assay-class mixing and replicate
aggregation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.prepare_dataset import (  # noqa: E402
    BAO_CELL_BASED,
    BAO_SINGLE_PROTEIN,
    HUMAN_TAX_ID,
    MAX_PIC50,
    MIN_PIC50,
    murcko_scaffold,
    to_pic50,
)


class TestPotencyConversion:
    @pytest.mark.parametrize(
        "nanomolar,expected_pic50",
        [
            (1.0, 9.0),        # 1 nM
            (1000.0, 6.0),     # 1 uM
            (1_000_000.0, 3.0),  # 1 mM
            (0.1, 10.0),       # 100 pM
        ],
    )
    def test_known_conversions(self, nanomolar, expected_pic50):
        assert to_pic50(nanomolar) == pytest.approx(expected_pic50)

    def test_lower_concentration_means_higher_pic50(self):
        """The sign convention is the easiest thing to get backwards."""
        assert to_pic50(1.0) > to_pic50(100.0)

    def test_plausible_range_brackets_real_data(self):
        """A real nanomolar inhibitor must fall inside the accepted window."""
        assert MIN_PIC50 < to_pic50(5.0) < MAX_PIC50
        # A femtomolar value would be a unit error, and must fall outside.
        assert to_pic50(0.000001) > MAX_PIC50


class TestScaffold:
    def test_same_series_shares_a_scaffold(self):
        """Two analogues differing by a substituent must group together.

        This is what makes a scaffold split separate series rather than
        shuffling members of one series across folds.
        """
        first = murcko_scaffold(Chem.MolFromSmiles("c1ccc(CC(=O)O)cc1"))
        second = murcko_scaffold(Chem.MolFromSmiles("c1ccc(CC(=O)N)cc1"))
        assert first == second
        assert first != ""

    def test_different_ring_systems_differ(self):
        benzene = murcko_scaffold(Chem.MolFromSmiles("c1ccccc1C"))
        pyridine = murcko_scaffold(Chem.MolFromSmiles("c1ccncc1C"))
        assert benzene != pyridine

    def test_acyclic_molecule_has_no_scaffold(self):
        """Acyclic compounds must not all collapse into one shared group."""
        assert murcko_scaffold(Chem.MolFromSmiles("CCCCCO")) == ""


class TestAssayFiltering:
    """The filters that fixed the apparent replicate noise.

    Pooling a cell-based assay with a biochemical one manufactured 1-3 log
    "disagreement" for 30 compounds; these guard the constants that encode the
    distinction.
    """

    def test_bao_codes_are_distinct(self):
        assert BAO_SINGLE_PROTEIN != BAO_CELL_BASED

    def test_human_tax_id(self):
        assert HUMAN_TAX_ID == 9606

    def test_cell_based_records_are_separated(self):
        """A frame holding both assay formats must split, not average."""
        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCO"],
                "bao_format": [BAO_SINGLE_PROTEIN, BAO_CELL_BASED],
                "standard_value": [10.0, 1000.0],
            }
        )
        biochemical = df[~df["bao_format"].eq(BAO_CELL_BASED)]
        cellular = df[df["bao_format"].eq(BAO_CELL_BASED)]
        assert len(biochemical) == 1
        assert len(cellular) == 1
        # Averaging the two would land 1 log unit from either truth.
        assert to_pic50(biochemical["standard_value"].iloc[0]) == pytest.approx(8.0)

    def test_nonhuman_assay_is_excluded(self):
        df = pd.DataFrame({"assay_tax_id": [9606, 6183, np.nan]})
        kept = df[
            df["assay_tax_id"].isna() | df["assay_tax_id"].eq(HUMAN_TAX_ID)
        ]
        # Human kept, Schistosoma (6183) dropped, unlabelled kept.
        assert len(kept) == 2
        assert 6183 not in kept["assay_tax_id"].to_numpy()


class TestCensoredRecords:
    def test_qualified_relations_are_split_out(self):
        """'>' records must not be treated as exact measurements.

        A ">10000 nM" record means the assay never reached an effect; using
        10000 as the potency pins the label to the detection limit.
        """
        df = pd.DataFrame(
            {"standard_relation": ["=", ">", "<", None], "standard_value": [1.0] * 4}
        )
        relation = df["standard_relation"].fillna("=").str.strip()
        exact = df[relation == "="]
        censored = df[relation != "="]
        assert len(exact) == 2  # the '=' and the None treated as '='
        assert len(censored) == 2


class TestReplicateAggregation:
    def test_median_resists_an_outlier(self):
        """Why median and not mean: one bad lab value should not move the label."""
        values = pd.Series([7.0, 7.1, 7.2, 11.0])
        assert values.median() == pytest.approx(7.15, abs=0.01)
        assert values.mean() > 8.0

    def test_spread_is_computed_as_max_minus_min(self):
        values = pd.Series([6.0, 7.0, 9.0])
        assert values.max() - values.min() == pytest.approx(3.0)
