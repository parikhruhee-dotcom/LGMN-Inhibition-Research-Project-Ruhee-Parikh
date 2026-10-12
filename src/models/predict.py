"""Predict LGMN pIC50 for new compounds, with the caveats attached.

Every prediction is returned with the maximum Tanimoto similarity to the
training set and the model's own measured scaffold-split error, because a
bare number from a model with negative R2 invites more confidence than it
deserves. Compounds below the similarity floor are marked as outside the
applicability domain rather than quietly scored.

Usage:
    python src/models/predict.py --smiles "CC(=O)Nc1ccccc1"
    python src/models/predict.py --input compounds.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.standardize import find_warheads, standardize_smiles  # noqa: E402
from models.featurize import featurize, tanimoto_to_training_set  # noqa: E402
from models.train import SIMILARITY_FLOOR  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results" / "models" / "best_model.joblib"


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found; run src/models/train.py first"
        )
    return joblib.load(MODEL_PATH)


def predict(smiles: list[str]) -> pd.DataFrame:
    """Predict pIC50 with applicability-domain and mechanism annotations."""
    bundle = load_model()

    standardised, originals = [], []
    for raw in smiles:
        clean = standardize_smiles(raw)
        if clean is not None:
            standardised.append(clean)
            originals.append(raw)

    if not standardised:
        return pd.DataFrame()

    X, _ = featurize(standardised, bundle["features"])
    predictions = bundle["model"].predict(X)
    similarity = tanimoto_to_training_set(standardised, bundle["training_smiles"])

    from rdkit import Chem

    warheads = [
        "|".join(find_warheads(Chem.MolFromSmiles(s))) for s in standardised
    ]

    frame = pd.DataFrame(
        {
            "input_smiles": originals,
            "standardised_smiles": standardised,
            "predicted_pic50": predictions,
            "max_train_similarity": similarity,
            "in_applicability_domain": similarity >= SIMILARITY_FLOOR,
            "covalent_warhead": [bool(w) for w in warheads],
            "warheads": warheads,
            # The model's measured error on unseen scaffolds, carried with
            # every prediction so it is not separated from the number.
            "model_scaffold_mae": bundle["scaffold_split_mae"],
            "mean_baseline_mae": bundle["mean_baseline_mae"],
        }
    )
    return frame.sort_values("predicted_pic50", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", nargs="*", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--smiles-column", default="smiles", help="column name in --input"
    )
    args = parser.parse_args()

    if args.smiles:
        queries = args.smiles
    elif args.input:
        if not args.input.exists():
            print(f"{args.input} not found", file=sys.stderr)
            return 1
        table = pd.read_csv(args.input)
        if args.smiles_column not in table.columns:
            print(
                f"column {args.smiles_column!r} not in {args.input}; "
                f"found {list(table.columns)}",
                file=sys.stderr,
            )
            return 1
        queries = table[args.smiles_column].dropna().astype(str).tolist()
    else:
        parser.error("pass --smiles or --input")

    results = predict(queries)
    if results.empty:
        print("No input structure could be parsed.", file=sys.stderr)
        return 1

    bundle_mae = results["model_scaffold_mae"].iloc[0]
    baseline = results["mean_baseline_mae"].iloc[0]
    print(
        f"Model scaffold-split MAE {bundle_mae:.3f} log units "
        f"(mean baseline {baseline:.3f}).\n"
        "R2 on unseen scaffolds is negative, so treat these as a ranking, "
        "not as\npredicted potencies.\n"
    )

    columns = [
        "standardised_smiles", "predicted_pic50", "max_train_similarity",
        "in_applicability_domain", "covalent_warhead",
    ]
    print(results[columns].to_string(index=False))

    outside = int((~results["in_applicability_domain"]).sum())
    if outside:
        print(
            f"\n{outside} of {len(results)} compounds fall below Tanimoto "
            f"{SIMILARITY_FLOOR} to anything\nin the training set. Those "
            "predictions are extrapolation."
        )

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
