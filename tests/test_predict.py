"""Tests for the prediction entry point.

A bare number from a model with negative R2 invites more confidence than it
deserves, so predict.py attaches the caveats to each prediction: nearest-
neighbour similarity, an applicability-domain flag, the model's measured
error, and whether the compound carries a covalent warhead. These check that
those stay attached.

They skip when no trained model is present, since results/models is
gitignored and CI starts from a clean checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results" / "models" / "best_model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="no trained model; run src/models/train.py"
)

# A legumain-series nitrile, and two drugs unrelated to this chemistry.
IN_DOMAIN = "N#C[C@H](CC(N)=O)NC(=O)C1CCCN1C(=O)C1(c2ccccc2)CC1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
DIAZEPAM = "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"


def _predict(smiles):
    from models.predict import predict

    return predict(smiles)


class TestApplicabilityDomain:
    def test_series_compound_is_in_domain(self):
        row = _predict([IN_DOMAIN]).iloc[0]
        assert row["max_train_similarity"] > 0.4
        assert bool(row["in_applicability_domain"])

    def test_unrelated_drug_is_flagged_as_extrapolation(self):
        """Aspirin is nothing like this chemistry and must be marked so."""
        row = _predict([ASPIRIN]).iloc[0]
        assert row["max_train_similarity"] < 0.4
        assert not bool(row["in_applicability_domain"])

    def test_similarity_is_bounded(self):
        for value in _predict([IN_DOMAIN, ASPIRIN, DIAZEPAM])[
            "max_train_similarity"
        ]:
            assert 0.0 <= value <= 1.0


class TestCaveatsTravelWithThePrediction:
    def test_model_error_is_attached_to_every_row(self):
        frame = _predict([IN_DOMAIN, ASPIRIN])
        assert (frame["model_scaffold_mae"] > 0).all()
        assert (frame["mean_baseline_mae"] > 0).all()

    def test_covalent_warhead_is_reported(self):
        frame = _predict([IN_DOMAIN, ASPIRIN]).set_index("input_smiles")
        assert bool(frame.loc[IN_DOMAIN, "covalent_warhead"])
        assert not bool(frame.loc[ASPIRIN, "covalent_warhead"])

    def test_predictions_are_in_a_plausible_range(self):
        """A pIC50 outside 2-12 would mean something is badly wrong."""
        for value in _predict([IN_DOMAIN, ASPIRIN, DIAZEPAM])["predicted_pic50"]:
            assert 2.0 < value < 12.0


class TestInputHandling:
    def test_unparseable_structures_are_dropped_not_predicted(self):
        frame = _predict([ASPIRIN, "not_a_molecule"])
        assert len(frame) == 1

    def test_all_invalid_input_returns_empty(self):
        assert _predict(["not_a_molecule", ""]).empty

    def test_salt_and_parent_give_the_same_prediction(self):
        """Standardisation runs before featurisation."""
        frame = _predict(["CCN.Cl", "CCN"])
        assert frame["predicted_pic50"].nunique() == 1
