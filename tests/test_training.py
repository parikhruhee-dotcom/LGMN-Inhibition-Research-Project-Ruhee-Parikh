"""Tests for the evaluation machinery behind every reported model number.

The scaffold split is what makes this project's model results honest. If a
scaffold leaked across a fold boundary, the reported MAE would measure recall
of a compound series rather than prediction of a new one, and every number in
RESULTS.md would be quietly inflated. It had no tests until now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.train import (  # noqa: E402
    N_FOLDS,
    RANDOM_STATE,
    SIMILARITY_FLOOR,
    TanimotoNearestNeighbour,
    build_estimators,
)


def _scaffold_groups(scaffolds: list[str]) -> list[str]:
    """The grouping train.py builds, including the acyclic special case."""
    return [
        scaffold if scaffold else f"__acyclic_{i}"
        for i, scaffold in enumerate(scaffolds)
    ]


class TestScaffoldSplit:
    def test_no_scaffold_appears_on_both_sides_of_a_fold(self):
        """The property the whole evaluation rests on."""
        scaffolds = [f"S{i % 12}" for i in range(60)]
        groups = _scaffold_groups(scaffolds)
        X = np.arange(60).reshape(-1, 1)
        y = np.random.default_rng(0).normal(size=60)

        splitter = GroupKFold(n_splits=min(N_FOLDS, len(set(groups))))
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            train_scaffolds = {groups[i] for i in train_idx}
            test_scaffolds = {groups[i] for i in test_idx}
            assert not (train_scaffolds & test_scaffolds), (
                "a scaffold appeared in both training and test"
            )

    def test_random_split_does_leak_scaffolds(self):
        """Contrast: this is exactly what the scaffold split exists to avoid.

        Documents why the random-split numbers in the report are the ones that
        look good and mean least.
        """
        scaffolds = [f"S{i % 12}" for i in range(60)]
        groups = _scaffold_groups(scaffolds)
        X = np.arange(60).reshape(-1, 1)
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        leaked = False
        for train_idx, test_idx in splitter.split(X):
            if {groups[i] for i in train_idx} & {groups[i] for i in test_idx}:
                leaked = True
        assert leaked

    def test_acyclic_compounds_do_not_collapse_into_one_group(self):
        """Empty scaffolds must not form a single giant group.

        42 of this dataset's scaffolds appear once, and acyclic compounds have
        no Murcko scaffold at all. Grouping them together would put unrelated
        molecules in one fold and distort the split.
        """
        groups = _scaffold_groups(["", "", "", "S1", "S1"])
        assert len(set(groups[:3])) == 3
        assert groups[3] == groups[4]

    def test_every_compound_is_tested_exactly_once(self):
        scaffolds = [f"S{i % 12}" for i in range(60)]
        groups = _scaffold_groups(scaffolds)
        X = np.arange(60).reshape(-1, 1)
        splitter = GroupKFold(n_splits=min(N_FOLDS, len(set(groups))))
        seen: list[int] = []
        for _, test_idx in splitter.split(X, np.zeros(60), groups=groups):
            seen.extend(test_idx.tolist())
        assert sorted(seen) == list(range(60))

    def test_fold_count_degrades_gracefully_with_few_groups(self):
        """Fewer scaffolds than folds must not raise."""
        groups = _scaffold_groups(["S1", "S1", "S2", "S2", "S3", "S3"])
        n_splits = min(N_FOLDS, len(set(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(np.arange(6).reshape(-1, 1),
                                    np.zeros(6), groups=groups))
        assert len(folds) == 3


class TestBaseline:
    def test_nearest_neighbour_recovers_an_identical_compound(self):
        """1-NN must return the known potency of the same structure."""
        model = TanimotoNearestNeighbour().fit(
            ["c1ccccc1O", "CCCCCCCC"], np.array([9.0, 5.0])
        )
        assert model.predict(["Oc1ccccc1"])[0] == pytest.approx(9.0)

    def test_unparseable_query_falls_back_to_the_mean(self):
        model = TanimotoNearestNeighbour().fit(
            ["c1ccccc1O", "CCCCCCCC"], np.array([9.0, 5.0])
        )
        assert model.predict(["not_a_molecule"])[0] == pytest.approx(7.0)

    def test_unparseable_training_structures_are_dropped(self):
        model = TanimotoNearestNeighbour().fit(
            ["c1ccccc1O", "not_a_molecule"], np.array([9.0, 1.0])
        )
        assert len(model.y) == 1
        assert model.predict(["Oc1ccccc1"])[0] == pytest.approx(9.0)

    def test_prediction_length_matches_input(self):
        model = TanimotoNearestNeighbour().fit(["CCO"], np.array([7.0]))
        assert len(model.predict(["CCO", "c1ccccc1", "CC(=O)O"])) == 3


class TestEstimators:
    def test_every_estimator_imputes_before_fitting(self):
        """Descriptors can be NaN for odd valences; a NaN must not reach a model."""
        for name, (pipeline, _grid) in build_estimators().items():
            assert "impute" in dict(pipeline.steps), f"{name} has no imputer"

    def test_grids_are_small_enough_for_the_sample_size(self):
        """A large grid searched on ~29-compound inner folds overfits the search."""
        for name, (_pipeline, grid) in build_estimators().items():
            combinations = 1
            for values in grid.values():
                combinations *= len(values)
            assert combinations <= 12, f"{name} grid has {combinations} combinations"

    def test_similarity_floor_is_a_valid_tanimoto(self):
        assert 0.0 < SIMILARITY_FLOOR < 1.0
