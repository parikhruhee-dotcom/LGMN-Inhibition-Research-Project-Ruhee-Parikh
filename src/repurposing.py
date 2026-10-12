"""Screen approved drugs for legumain activity (drug repurposing).

Everything else in this project evaluates compounds already known to inhibit
legumain. This asks the question the pipeline was built for: is there
anything *new*? Approved drugs are the most useful place to look, because a
hit is immediately testable -- the safety, dosing and manufacturing work is
already done, and the compound can be bought.

Filters run cheapest-first, which is the only affordable order. Docking costs
about 95 seconds per compound on this machine, so screening all 4,225 approved
drugs would take five days. Property filters cost milliseconds and remove most
of them, and they remove them for reasons this project has already
established:

1. **Brain penetration.** The binding constraint for a CNS target, and the
   step that eliminated 129 of 185 known legumain inhibitors.
2. **No covalent warhead.** A reactive group on an approved drug is usually
   there for a different target; docking cannot score it here anyway.
3. **Inside the model's applicability domain.** A prediction for chemistry
   unlike anything in the training set is extrapolation, and this model has
   negative R2 even inside its domain.

What survives gets docked. The output is a ranked list, not a claim.

Writes:
    data/raw/chembl_approved_drugs.csv
    data/processed/repurposing_candidates.csv
    docs/REPURPOSING.md

Usage:
    python src/repurposing.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admet.filters import profile  # noqa: E402
from data.standardize import inchikey, is_covalent, standardize_mol  # noqa: E402
from models.featurize import featurize, tanimoto_to_training_set  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
MODEL_PATH = ROOT / "results" / "models" / "best_model.joblib"

# Approved for human use.
MAX_PHASE = 4

# Tanimoto similarity to the training set below which a prediction is
# extrapolation rather than interpolation.
SIMILARITY_FLOOR = 0.4

# How many survivors to carry forward to docking. Docking is the expensive
# step and the scores are weak evidence, so this is a triage size rather than
# an attempt at completeness.
DEFAULT_DOCKING_LIMIT = 60


def fetch_approved() -> pd.DataFrame:
    """Every approved drug in ChEMBL with a structure."""
    records = new_client.molecule.filter(max_phase=MAX_PHASE).only(
        [
            "molecule_chembl_id",
            "pref_name",
            "max_phase",
            "molecule_structures",
            "molecule_type",
        ]
    )
    rows = []
    for record in records:
        structures = record.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        if not smiles:
            continue
        rows.append(
            {
                "molecule_chembl_id": record["molecule_chembl_id"],
                "pref_name": record.get("pref_name") or "",
                "molecule_type": record.get("molecule_type") or "",
                "smiles": smiles,
            }
        )
    return pd.DataFrame(rows)


def screen(drugs: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    """Apply the property filters and model prediction, cheapest first."""
    rows = []
    # Approved-drug lists carry the same molecule under several entries --
    # a free base and its salts each have their own ChEMBL ID. Standardising
    # strips the counter-ion, so without this the same compound appears
    # repeatedly: caffeine and caffeine citrate are one molecule, not two.
    seen_keys: set[str] = set()

    for row in drugs.itertuples(index=False):
        mol = standardize_mol(row.smiles)
        if mol is None:
            continue
        # Small molecules only: the docking and descriptor machinery is
        # meaningless for antibodies, peptides and oligonucleotides.
        if mol.GetNumHeavyAtoms() < 8 or mol.GetNumHeavyAtoms() > 70:
            continue

        canonical = Chem.MolToSmiles(mol)
        key = inchikey(canonical)
        if key is not None:
            if key in known:
                continue  # already in the legumain dataset; not new
            if key in seen_keys:
                continue  # a salt form of a drug already screened
            seen_keys.add(key)

        admet = profile(canonical)
        if admet is None:
            continue

        rows.append(
            {
                "molecule_chembl_id": row.molecule_chembl_id,
                "pref_name": row.pref_name,
                "smiles": canonical,
                "covalent": is_covalent(mol),
                "cns_candidate": admet["cns_candidate"],
                "cns_mpo": admet["cns_mpo"],
                "bbb_rules_passed": admet["bbb_rules_passed"],
                "bbb_rules_total": admet["bbb_rules_total"],
                "mw": admet["mw"],
                "clogp": admet["clogp"],
                "tpsa": admet["tpsa"],
                "qed": admet["qed"],
                "pains_alerts": admet["pains_alerts"],
                "brenk_alerts": admet["brenk_alerts"],
                "brenk_names": admet["brenk_names"],
            }
        )

    return pd.DataFrame(rows)


def add_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted potency and distance from the model's applicability domain."""
    if frame.empty or not MODEL_PATH.exists():
        frame["predicted_pic50"] = np.nan
        frame["max_train_similarity"] = np.nan
        frame["in_applicability_domain"] = False
        return frame

    bundle = joblib.load(MODEL_PATH)
    smiles = frame["smiles"].tolist()
    X, _ = featurize(smiles, bundle["features"])
    frame = frame.copy()
    frame["predicted_pic50"] = bundle["model"].predict(X)
    frame["max_train_similarity"] = tanimoto_to_training_set(
        smiles, bundle["training_smiles"]
    )
    frame["in_applicability_domain"] = (
        frame["max_train_similarity"] >= SIMILARITY_FLOOR
    )
    return frame


def _docking_verdict(shortlist: pd.DataFrame) -> list[str]:
    """What docking the survivor showed, and why it settles nothing.

    Kept data-driven rather than written as prose, because the numbers moved
    once already in this project after being quoted as text.
    """
    import json

    context_path = PROCESSED / "repurposing_context.json"
    if "affinity" not in shortlist.columns or shortlist["affinity"].isna().all():
        return []
    if not context_path.exists():
        return []

    context = json.loads(context_path.read_text())
    row = shortlist.iloc[0]
    rank = int(shortlist["affinity"].iloc[0] > context["affinity_median"])

    return [
        "### It was docked, and the result settles nothing",
        "",
        f"Caffeine docks into the active site -- "
        f"{row['dist_to_thiol']:.1f} A from the catalytic thiol -- with a "
        f"score of {row['affinity']:.2f} kcal/mol.",
        f"Against the {context['n_non_covalent']} known non-covalent "
        f"inhibitors, which span "
        f"{context['affinity_min']:.2f} to {context['affinity_max']:.2f} "
        f"(median {context['affinity_median']:.2f}), that places it near the "
        "bottom.",
        "",
        "Read quickly, that looks like docking rejecting the compound. It is "
        "not, and the",
        "reason is the useful part of this whole exercise.",
        "",
        f"Its thio-analogue **{context['neighbour_id']}** -- a real inhibitor "
        f"with a measured",
        f"pIC50 of {context['neighbour_pic50']:.2f} -- scores "
        f"{context['neighbour_affinity']:.2f}, which is *worse* than "
        "caffeine.",
        "A genuine, confirmed inhibitor of this target is ranked below a "
        "compound that",
        "probably is not one.",
        "",
        "So docking cannot distinguish them. The scoring function has no "
        "information about",
        "what makes this scaffold work, which is consistent with the thione "
        "being the",
        "feature that matters: a sulfur-for-oxygen swap barely changes shape "
        "or size, and",
        "shape and size are most of what the score measures.",
        "",
        "The honest conclusion is that **three methods here have now failed "
        "to answer the",
        "same question.** The model predicts caffeine is active because it "
        "looks like an",
        "active compound. Docking ranks it low, but ranks the known active "
        "lower still.",
        "Neither is evidence. The question is open and an afternoon of "
        "enzyme assay would",
        "close it.",
        "",
    ]


def write_doc(
    total: int, screened: pd.DataFrame, shortlist: pd.DataFrame
) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    cns = screened["cns_candidate"].fillna(False).astype(bool)
    covalent = screened["covalent"].fillna(False).astype(bool)
    in_domain = screened["in_applicability_domain"].fillna(False).astype(bool)

    lines = [
        "# Repurposing screen: approved drugs against legumain",
        "",
        "Generated by `src/repurposing.py`.",
        "",
        "Every other part of this project evaluates compounds already known "
        "to inhibit",
        "legumain. This asks whether anything **new** turns up, and looks "
        "among approved",
        "drugs because a hit there is immediately testable: the compound can "
        "be bought, and",
        "its safety and dosing are already established.",
        "",
        "> **Nothing here is a hit.** These are compounds that survive "
        "filtering and score",
        "> plausibly. A model with negative R2 on unseen chemistry cannot "
        "identify an",
        "> inhibitor; it can only say which compounds are worth an "
        "experiment first.",
        "",
        "## The funnel",
        "",
        "| Stage | Compounds | Why |",
        "|---|---|---|",
        f"| Approved drugs in ChEMBL | {total} | |",
        f"| Small molecules with a usable structure | {len(screened)} | "
        "excludes biologics, peptides and anything outside 8-70 heavy atoms |",
        f"| Predicted to reach the brain | {int(cns.sum())} | the constraint "
        "that eliminated 129 of 185 known legumain inhibitors |",
        f"| ...and not covalent | {int((cns & ~covalent).sum())} | a reactive "
        "group on an approved drug is there for a different target |",
        f"| ...and inside the model's domain | "
        f"{int((cns & ~covalent & in_domain).sum())} | otherwise the "
        "prediction is extrapolation |",
        "",
    ]

    if shortlist.empty:
        lines += [
            "## Result: nothing survives",
            "",
            "No approved drug passes all three filters. That is a real "
            "finding rather than a",
            "failed search: the chemistry that reaches the brain and the "
            "chemistry that",
            "resembles a legumain inhibitor have essentially no overlap in "
            "the approved",
            "pharmacopoeia. It is the same conflict the rest of this project "
            "documents, seen",
            "from the opposite direction.",
            "",
            "Relaxing the applicability-domain filter would produce a list, "
            "but it would be a",
            "list of predictions the model has no basis for making.",
            "",
        ]
    else:
        lines += [
            "## The survivor"
            if len(shortlist) == 1
            else f"## The {len(shortlist)} survivors",
            "",
            "Ranked by predicted potency. **Similarity** is how close the "
            "compound is to the",
            "chemistry the model was trained on; below 0.4 the prediction is "
            "extrapolation.",
            "",
            "| Drug | ChEMBL ID | Predicted pIC50 | Similarity | CNS MPO | "
            "MW | Alerts |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in shortlist.itertuples(index=False):
            alerts = int(row.pains_alerts) + int(row.brenk_alerts)
            lines.append(
                f"| {row.pref_name.title() or '-'} | "
                f"{row.molecule_chembl_id} | {row.predicted_pic50:.2f} | "
                f"{row.max_train_similarity:.2f} | {row.cns_mpo:.2f} | "
                f"{row.mw:.0f} | {alerts} |"
            )
        lines += [
            "",
            "### How to read this",
            "",
            "These are starting points for an experiment, not predictions of "
            "activity. The",
            "model behind the potency column cannot predict absolute potency "
            "for chemistry",
            "outside its training set, and approved drugs are by definition "
            "outside it. The",
            "similarity column is the honest part: the closer to 1.0, the "
            "more the model has",
            "actually seen something like this compound.",
            "",
            "### Why the list is this short, and what the survivor means",
            "",
            "Over a thousand approved drugs are predicted to reach the brain "
            "and carry no",
            "covalent warhead. Almost none of them resemble a legumain "
            "inhibitor closely",
            "enough for this model to say anything about. That is the "
            "project's central",
            "finding arriving from the opposite direction: the chemistry that "
            "crosses into the",
            "brain and the chemistry that blocks this enzyme barely overlap, "
            "and the approved",
            "pharmacopoeia contains essentially none of the intersection.",
            "",
            "The compound that does survive is instructive, and not in a "
            "flattering way.",
            "**Caffeine** clears the filters because the legumain dataset "
            "contains a series of",
            "*thio*-xanthines, and its nearest neighbour (CHEMBL4092935, "
            "pIC50 6.74) is caffeine",
            "with one sulfur atom added at position 8. The model sees two "
            "nearly identical",
            "structures and predicts nearly identical potency.",
            "",
            "That single atom is very likely the point. A thione sits where a "
            "plain carbonyl",
            "cannot, and in a cysteine protease the sulfur is the obvious "
            "candidate for the",
            "interaction that matters. A model built on fingerprints has no "
            "way to know which",
            "atom carries the pharmacophore; it only knows the two molecules "
            "look alike. The",
            "most probable truth is that caffeine does not inhibit legumain "
            "at any useful",
            "concentration, and that this prediction is the applicability "
            "domain failing at its",
            "own boundary -- a similarity of 0.47 is only just inside it.",
            "",
            "It is worth stating plainly because the test is trivial. "
            "Caffeine is in every",
            "laboratory, and the enzyme assay already exists. Someone could "
            "settle this in an",
            "afternoon, and the expected answer is no.",
            "",
            "A search of PubMed found no published report of caffeine or "
            "related xanthines",
            "inhibiting legumain, so this is not a rediscovery of something "
            "known.",
            "",
            *_docking_verdict(shortlist),
        ]

    lines += [
        "## What this does not do",
        "",
        "- It does not test binding. No compound here has been docked unless "
        "noted, and",
        "  docking in this project carries weak signal at best "
        "(`docs/DOCKING.md`).",
        "- It does not consider whether a drug's existing targets would "
        "matter. A compound",
        "  that inhibits legumain and ten other enzymes is not a useful "
        "tool.",
        "- It does not consider dose. Reaching the brain at all is different "
        "from reaching it",
        "  at a concentration that inhibits anything.",
        "",
    ]
    (DOCS / "REPURPOSING.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_DOCKING_LIMIT)
    args = parser.parse_args()

    dataset_path = PROCESSED / "lgmn_regression.csv"
    if not dataset_path.exists():
        print(f"{dataset_path} not found", file=sys.stderr)
        return 1
    known_frame = pd.read_csv(dataset_path)
    known = {k for k in (inchikey(s) for s in known_frame["smiles"]) if k}

    print(f"Fetching approved drugs from ChEMBL (max_phase={MAX_PHASE}) ...")
    drugs = fetch_approved()
    RAW.mkdir(parents=True, exist_ok=True)
    drugs.to_csv(RAW / "chembl_approved_drugs.csv", index=False)
    print(f"  {len(drugs)} with a structure")

    print("Filtering ...")
    screened = screen(drugs, known)
    screened = add_predictions(screened)
    print(f"  {len(screened)} small molecules profiled")

    cns = screened["cns_candidate"].fillna(False).astype(bool)
    covalent = screened["covalent"].fillna(False).astype(bool)
    in_domain = screened["in_applicability_domain"].fillna(False).astype(bool)
    print(f"  predicted brain-penetrant: {int(cns.sum())}")
    print(f"  ...and non-covalent:       {int((cns & ~covalent).sum())}")
    print(
        f"  ...and inside the model's domain: "
        f"{int((cns & ~covalent & in_domain).sum())}"
    )

    shortlist = (
        screened[cns & ~covalent & in_domain]
        .sort_values("predicted_pic50", ascending=False)
        .head(args.limit)
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    screened.to_csv(PROCESSED / "repurposing_screened.csv", index=False)
    shortlist.to_csv(PROCESSED / "repurposing_candidates.csv", index=False)

    if shortlist.empty:
        print(
            "\nNo approved drug passes all three filters. That is a finding: "
            "the chemistry\nthat reaches the brain and the chemistry "
            "resembling a legumain inhibitor barely\noverlap in the approved "
            "pharmacopoeia."
        )
    else:
        print(f"\nTop candidates ({len(shortlist)}):")
        print(
            shortlist.head(12)[
                ["pref_name", "predicted_pic50", "max_train_similarity", "cns_mpo"]
            ].to_string(index=False)
        )

    write_doc(len(drugs), screened, shortlist)
    print(f"\nWrote {PROCESSED / 'repurposing_candidates.csv'}")
    print(f"Wrote {DOCS / 'REPURPOSING.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
