"""Train and honestly evaluate pIC50 regression models for LGMN.

The evaluation design matters more than the model choice at n=180.

**Scaffold splitting, not random.** Grouping by Bemis-Murcko scaffold and
keeping whole scaffolds on one side of the split is the difference between
measuring "can this predict a new compound in a series it has seen" and "can
this predict a new series". A random split on a dataset where 55 scaffolds
cover 180 compounds mostly tests memorisation of a series, and inflates every
metric. Both are reported so the gap is visible.

**Nested cross-validation.** Hyperparameters are tuned in an inner loop on
training folds only. Tuning on the same folds used for reporting leaks the
test set into model selection and is the most common way a small-data QSAR
paper reports a score it cannot reproduce.

**A baseline that must be beaten.** Mean prediction and a 1-nearest-neighbour
Tanimoto model are evaluated alongside. If a gradient-boosted model cannot beat
predicting the training mean, the honest conclusion is that the features carry
no transferable signal -- a result worth reporting, not hiding.

Writes:
    results/models/metrics.csv          every model x split x fold score
    results/models/summary.csv          aggregated, with standard deviations
    results/models/predictions.csv      out-of-fold predictions
    results/models/<name>.joblib        final model refit on all data
    results/figures/model_performance.png
    results/figures/applicability_domain.png

Usage:
    python src/models/train.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.featurize import (  # noqa: E402
    featurize,
    morgan_fingerprint,
    tanimoto_to_training_set,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "lgmn_regression.csv"
MODEL_DIR = ROOT / "results" / "models"
FIGURE_DIR = ROOT / "results" / "figures"

N_FOLDS = 5
RANDOM_STATE = 0

# Reported alongside every prediction. 0.4 Tanimoto on ECFP4 is a conventional
# line between "same chemistry as training" and "extrapolation".
SIMILARITY_FLOOR = 0.4


class TanimotoNearestNeighbour:
    """1-NN regressor on Tanimoto similarity over Morgan fingerprints.

    Included as a baseline because for a small, series-heavy dataset "find the
    most similar known compound and copy its potency" is a strong strategy, and
    a learned model that cannot beat it has not earned its complexity.
    """

    def __init__(self) -> None:
        self.fps: list = []
        self.y: np.ndarray = np.array([])

    def fit(self, smiles: list[str], y: np.ndarray) -> "TanimotoNearestNeighbour":
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        keep = [i for i, mol in enumerate(mols) if mol is not None]
        self.fps = [morgan_fingerprint(mols[i]) for i in keep]
        self.y = np.asarray(y)[keep]
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        out = np.full(len(smiles), float(self.y.mean()))
        for index, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(
                morgan_fingerprint(mol), self.fps
            )
            if sims:
                out[index] = self.y[int(np.argmax(sims))]
        return out


def build_estimators() -> dict[str, tuple]:
    """Return {name: (pipeline, param_grid)} for the sklearn models.

    Grids are deliberately small. With ~144 training compounds per fold, a
    large grid searched on inner folds of ~29 compounds overfits the search
    itself; a handful of well-separated values is more honest.
    """
    return {
        "ridge": (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", RidgeCV(alphas=np.logspace(-2, 4, 25))),
                ]
            ),
            {},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=500,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [None, 6],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt", 0.3],
            },
        ),
        "xgboost": (
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=400,
                            learning_rate=0.05,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            tree_method="hist",
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [2, 4],
                "model__subsample": [0.7, 1.0],
                "model__min_child_weight": [1, 5],
            },
        ),
    }


def evaluate(
    df: pd.DataFrame, feature_kind: str, split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate every model under one featurisation and split scheme."""
    smiles = df["smiles"].tolist()
    y = df["pic50"].to_numpy()
    X, _names = featurize(smiles, feature_kind)

    if split == "scaffold":
        # Scaffold-less compounds (acyclic) would otherwise form one giant
        # group; give each its own so they distribute across folds.
        groups = [
            scaffold if scaffold else f"__acyclic_{i}"
            for i, scaffold in enumerate(df["scaffold"].fillna(""))
        ]
        n_groups = len(set(groups))
        splitter = GroupKFold(n_splits=min(N_FOLDS, n_groups))
        folds = list(splitter.split(X, y, groups=groups))
    else:
        splitter = KFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(splitter.split(X, y))

    estimators = build_estimators()
    rows = []
    predictions = []

    for fold_index, (train_idx, test_idx) in enumerate(folds):
        y_train, y_test = y[train_idx], y[test_idx]
        train_smiles = [smiles[i] for i in train_idx]
        test_smiles = [smiles[i] for i in test_idx]

        # Similarity of each held-out compound to its own training fold, which
        # is what makes the scaffold-split difficulty concrete.
        similarity = tanimoto_to_training_set(test_smiles, train_smiles)

        fold_preds: dict[str, np.ndarray] = {}

        mean_model = DummyRegressor(strategy="mean").fit(X[train_idx], y_train)
        fold_preds["mean_baseline"] = mean_model.predict(X[test_idx])

        knn = TanimotoNearestNeighbour().fit(train_smiles, y_train)
        fold_preds["tanimoto_1nn"] = knn.predict(test_smiles)

        for name, (pipeline, grid) in estimators.items():
            if grid:
                inner = KFold(
                    n_splits=3, shuffle=True, random_state=RANDOM_STATE
                )
                search = GridSearchCV(
                    pipeline,
                    grid,
                    cv=inner,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X[train_idx], y_train)
                fitted = search.best_estimator_
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = pipeline.fit(X[train_idx], y_train)
            fold_preds[name] = fitted.predict(X[test_idx])

        for name, y_pred in fold_preds.items():
            rows.append(
                {
                    "model": name,
                    "features": feature_kind,
                    "split": split,
                    "fold": fold_index,
                    "n_test": len(test_idx),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
                    # R2 is undefined for a single-compound fold and
                    # meaningless for two; guard rather than emit noise.
                    "r2": (
                        r2_score(y_test, y_pred) if len(y_test) > 2 else np.nan
                    ),
                    "spearman": (
                        pd.Series(y_test).corr(pd.Series(y_pred), method="spearman")
                        if len(y_test) > 2
                        else np.nan
                    ),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "features": feature_kind,
                        "split": split,
                        "fold": fold_index,
                        "smiles": test_smiles,
                        "molecule_chembl_id": df.iloc[test_idx][
                            "molecule_chembl_id"
                        ].to_numpy(),
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "max_train_similarity": similarity,
                    }
                )
            )

    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def make_figures(summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- model performance, random vs scaffold split ---
    best = summary[summary["features"] == "both"] if (
        summary["features"] == "both"
    ).any() else summary
    pivot = best.pivot_table(
        index="model", columns="split", values="mae_mean"
    ).sort_values("scaffold", ascending=False)
    errors = best.pivot_table(index="model", columns="split", values="mae_std")
    errors = errors.reindex(pivot.index)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(pivot))
    width = 0.38
    for offset, split, colour in (
        (-width / 2, "random", "#8ecae6"),
        (width / 2, "scaffold", "#fb8500"),
    ):
        if split not in pivot.columns:
            continue
        axes[0].barh(
            positions + offset,
            pivot[split],
            height=width,
            xerr=errors[split] if split in errors else None,
            label=f"{split} split",
            color=colour,
            error_kw={"ecolor": "#555", "capsize": 3, "lw": 1},
        )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel("Mean absolute error (log units)  -  lower is better")
    axes[0].set_title("Prediction error by model and split")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    # --- error vs similarity to training set ---
    scaffold_preds = predictions[
        (predictions["split"] == "scaffold")
        & (predictions["features"] == best["features"].iloc[0])
    ]
    if not scaffold_preds.empty:
        strongest = (
            best[best["split"] == "scaffold"].sort_values("mae_mean")["model"].iloc[0]
        )
        subset = scaffold_preds[scaffold_preds["model"] == strongest]
        error = (subset["y_pred"] - subset["y_true"]).abs()
        axes[1].scatter(
            subset["max_train_similarity"],
            error,
            alpha=0.6,
            color="#fb8500",
            edgecolor="none",
        )
        axes[1].axvline(
            SIMILARITY_FLOOR,
            ls="--",
            color="#555",
            lw=1,
            label=f"similarity {SIMILARITY_FLOOR}",
        )
        axes[1].set_xlabel("Max Tanimoto similarity to training fold")
        axes[1].set_ylabel("Absolute error (log units)")
        axes[1].set_title(f"{strongest}: error vs chemical novelty")
        axes[1].legend(frameon=False)
        axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_performance.png", dpi=160)
    plt.close(fig)


def main() -> int:
    if not DATA.exists():
        print(
            f"{DATA} not found. Run src/data/prepare_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(DATA)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training on {len(df)} compounds, {df['scaffold'].nunique()} scaffolds")
    print(f"pIC50 spread: {df['pic50'].std():.2f} log units (std)\n")

    all_metrics = []
    all_predictions = []
    for feature_kind in ("descriptors", "morgan", "both"):
        for split in ("random", "scaffold"):
            metrics, predictions = evaluate(df, feature_kind, split)
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            best = metrics.groupby("model")["mae"].mean().sort_values()
            print(f"{feature_kind:12} / {split:9} split  MAE by model:")
            for name, score in best.items():
                print(f"      {name:16} {score:.3f}")
            print()

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    summary = (
        metrics.groupby(["model", "features", "split"])
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            spearman_mean=("spearman", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["split", "mae_mean"])
    )

    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    summary.to_csv(MODEL_DIR / "summary.csv", index=False)
    predictions.to_csv(MODEL_DIR / "predictions.csv", index=False)

    print("=" * 72)
    print("Scaffold-split results (the honest ones), sorted by error:")
    scaffold = summary[summary["split"] == "scaffold"]
    print(
        scaffold[
            ["model", "features", "mae_mean", "mae_std", "r2_mean", "spearman_mean"]
        ].to_string(index=False)
    )

    baseline = scaffold[scaffold["model"] == "mean_baseline"]["mae_mean"].min()
    learned = scaffold[~scaffold["model"].isin(["mean_baseline"])]
    best_row = learned.sort_values("mae_mean").iloc[0]
    print()
    print(f"Best learned model: {best_row['model']} on {best_row['features']}")
    print(f"  MAE {best_row['mae_mean']:.3f} vs mean-baseline MAE {baseline:.3f}")
    if best_row["mae_mean"] >= baseline:
        print(
            "  This does NOT beat predicting the training mean. On a scaffold "
            "split\n  the features carry no signal that transfers to unseen "
            "chemistry."
        )
    else:
        improvement = 100 * (baseline - best_row["mae_mean"]) / baseline
        print(f"  {improvement:.0f}% better than the mean baseline.")

    # Refit the best configuration on everything, for use in screening.
    X, _ = featurize(df["smiles"].tolist(), best_row["features"])
    y = df["pic50"].to_numpy()
    estimators = build_estimators()
    if best_row["model"] in estimators:
        pipeline, grid = estimators[best_row["model"]]
        if grid:
            search = GridSearchCV(
                pipeline,
                grid,
                cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X, y)
            final = search.best_estimator_
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final = pipeline.fit(X, y)
        joblib.dump(
            {
                "model": final,
                "features": best_row["features"],
                "training_smiles": df["smiles"].tolist(),
                "scaffold_split_mae": float(best_row["mae_mean"]),
                "mean_baseline_mae": float(baseline),
            },
            MODEL_DIR / "best_model.joblib",
        )
        print(f"\nSaved {MODEL_DIR / 'best_model.joblib'}")

    make_figures(summary, predictions)
    print(f"Saved {FIGURE_DIR / 'model_performance.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
