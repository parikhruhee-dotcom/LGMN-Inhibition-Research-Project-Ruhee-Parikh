"""Tests for featurisation and the applicability-domain measure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.featurize import (  # noqa: E402
    MORGAN_BITS,
    descriptor_array,
    featurize,
    morgan_array,
    tanimoto_to_training_set,
)


class TestMorgan:
    def test_shape_and_binary_values(self):
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        array = morgan_array(mol)
        assert array.shape == (MORGAN_BITS,)
        assert set(np.unique(array)).issubset({0.0, 1.0})

    def test_identical_molecules_give_identical_fingerprints(self):
        first = morgan_array(Chem.MolFromSmiles("c1ccccc1O"))
        second = morgan_array(Chem.MolFromSmiles("Oc1ccccc1"))
        np.testing.assert_array_equal(first, second)

    def test_different_molecules_differ(self):
        benzene = morgan_array(Chem.MolFromSmiles("c1ccccc1"))
        phenol = morgan_array(Chem.MolFromSmiles("c1ccccc1O"))
        assert not np.array_equal(benzene, phenol)


class TestDescriptors:
    def test_no_nans_for_an_ordinary_molecule(self):
        array = descriptor_array(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
        assert not np.isnan(array).any()

    def test_length_is_stable_across_molecules(self):
        first = descriptor_array(Chem.MolFromSmiles("CCO"))
        second = descriptor_array(Chem.MolFromSmiles("c1ccc2ccccc2c1"))
        assert first.shape == second.shape


class TestFeaturize:
    def test_combined_block_is_the_sum_of_its_parts(self):
        smiles = ["CCO", "c1ccccc1"]
        morgan, morgan_names = featurize(smiles, "morgan")
        descriptors, descriptor_names = featurize(smiles, "descriptors")
        both, both_names = featurize(smiles, "both")
        assert both.shape[1] == morgan.shape[1] + descriptors.shape[1]
        assert both_names == morgan_names + descriptor_names

    def test_row_count_matches_input(self):
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        matrix, _ = featurize(smiles, "both")
        assert matrix.shape[0] == len(smiles)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            featurize(["CCO"], "nonsense")

    def test_unparseable_smiles_raises_rather_than_silently_dropping(self):
        """A dropped row would silently misalign features from labels."""
        with pytest.raises(ValueError, match="unparseable"):
            featurize(["CCO", "not_a_molecule"], "morgan")


class TestApplicabilityDomain:
    def test_identical_compound_scores_one(self):
        similarity = tanimoto_to_training_set(["c1ccccc1O"], ["Oc1ccccc1"])
        assert similarity[0] == pytest.approx(1.0)

    def test_dissimilar_compound_scores_low(self):
        """A sugar against an aromatic training set is extrapolation."""
        similarity = tanimoto_to_training_set(
            ["OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"],
            ["c1ccc2ccccc2c1", "c1ccccc1C(=O)O"],
        )
        assert similarity[0] < 0.3

    def test_takes_the_maximum_not_the_mean(self):
        """One close neighbour is what makes a prediction supportable."""
        similarity = tanimoto_to_training_set(
            ["c1ccccc1O"],
            ["Oc1ccccc1", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"],
        )
        assert similarity[0] == pytest.approx(1.0)

    def test_empty_training_set_raises(self):
        with pytest.raises(ValueError):
            tanimoto_to_training_set(["CCO"], [])

    def test_result_length_matches_query_length(self):
        queries = ["CCO", "c1ccccc1", "CC(=O)O"]
        similarity = tanimoto_to_training_set(queries, ["CCO"])
        assert len(similarity) == len(queries)
