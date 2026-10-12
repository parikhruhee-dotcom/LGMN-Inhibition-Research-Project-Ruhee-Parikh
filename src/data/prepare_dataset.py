"""Turn the raw ChEMBL pull into a modelling dataset.

Decisions made here, and why:

- **Only human biochemical assays enter the modelling set.** Every LGMN record
  in ChEMBL carries assay_type 'B', so that field cannot separate assay
  classes. Two kinds of record have to be excluded using assay metadata:
  a HEK293-A *cell-based* assay (bao_format BAO_0000219) whose IC50s run 1-2
  log units weaker than the matching enzymatic measurements, because cellular
  potency folds in permeability and serum binding; and two *Schistosoma
  mansoni* assays (tax 6183) annotated against the human target. Pooling
  either with recombinant human enzyme data manufactures 1-3 log "replicate
  disagreement" that is really a difference in what was measured. Cell-based
  human records are written to their own file rather than discarded.
- **Only '=' relations are kept for regression.** A record of ">10000 nM" means
  the assay stopped before reaching an effect; treating 10000 as the potency
  biases the model toward the detection limit. Censored records are retained in
  a separate file because they are still valid evidence of *inactivity*.
- **Potency is aggregated per compound by median, not mean.** IC50 values
  spanning orders of magnitude between labs make the mean follow the outlier.
- **Compounds whose replicate measurements disagree by more than
  MAX_REPLICATE_SPREAD log units are dropped.** If the same structure is
  reported at 10 nM and 10 uM, neither number can be used as a label.
- **Covalent warheads are flagged, not removed.** They are most of the potent
  chemistry for a cysteine protease, so dropping them would discard the signal;
  but their docking scores are meaningless, so downstream code needs to know.

Writes:
    data/processed/lgmn_regression.csv   one row per compound, with pIC50
    data/processed/lgmn_cellular.csv     human cell-based records, kept apart
    data/processed/lgmn_censored.csv     qualified (>/<) records
    data/processed/lgmn_rejected.csv     every dropped record, with the reason

Usage:
    python src/data/prepare_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.standardize import (  # noqa: E402
    basic_properties,
    find_warheads,
    is_covalent,
    standardize_mol,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "chembl_lgmn_activities.csv"
RAW_ASSAYS = ROOT / "data" / "raw" / "chembl_lgmn_assays.csv"
PROCESSED = ROOT / "data" / "processed"

# BioAssay Ontology format codes. Only a purified-protein assay measures
# inhibition of the enzyme itself.
BAO_SINGLE_PROTEIN = "BAO_0000357"
BAO_CELL_BASED = "BAO_0000219"

HUMAN_TAX_ID = 9606

# ChEMBL's confidence that the assay really maps to the stated target. 8 means
# a homologous protein, 9 a direct single-protein assignment; below 8 the
# target assignment is too loose to train on.
MIN_CONFIDENCE_SCORE = 8

# Endpoints that measure a concentration producing a defined effect. Percent
# inhibition at a single dose and unitless "Activity" cannot be converted to a
# comparable potency.
USABLE_TYPES = {"IC50", "Ki", "Kd", "EC50"}

# Replicate disagreement above this (in log units) makes a label unusable.
MAX_REPLICATE_SPREAD = 1.0

# Potency bounds. Values outside are almost always unit errors in the
# deposition rather than real measurements: pIC50 of 15 would be femtomolar.
MIN_PIC50 = 2.0
MAX_PIC50 = 12.0


def murcko_scaffold(mol: Chem.Mol) -> str:
    """Bemis-Murcko scaffold SMILES, used for grouping related chemistry.

    Returns an empty string for molecules with no ring system, which would
    otherwise all collapse to one meaningless shared scaffold.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return ""
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def to_pic50(value_nm: float) -> float:
    """Convert a nanomolar concentration to pIC50 (= -log10 molar)."""
    return 9.0 - np.log10(value_nm)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW)
    assays = pd.read_csv(RAW_ASSAYS)
    rejected: list[pd.DataFrame] = []

    def reject(frame: pd.DataFrame, reason: str) -> None:
        if not frame.empty:
            dropped = frame.copy()
            dropped["reject_reason"] = reason
            rejected.append(dropped)

    assay_columns = [
        "assay_chembl_id",
        "assay_organism",
        "assay_tax_id",
        "assay_cell_type",
        "bao_format",
        "bao_label",
        "confidence_score",
    ]
    df = raw.merge(
        assays[[c for c in assay_columns if c in assays.columns]],
        on="assay_chembl_id",
        how="left",
        suffixes=("", "_assay"),
    )
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    # Wrong species. Two Schistosoma mansoni assays are annotated against the
    # human target in ChEMBL, and their potencies say nothing about human LGMN.
    # Assays with no organism recorded are old depositions against the human
    # target and are kept, since excluding an unlabelled field would discard
    # legitimate data on a technicality.
    nonhuman = df[
        df["assay_tax_id"].notna() & df["assay_tax_id"].ne(HUMAN_TAX_ID)
    ]
    reject(nonhuman, "assay_organism_not_human")
    df = df[df["assay_tax_id"].isna() | df["assay_tax_id"].eq(HUMAN_TAX_ID)]

    # Separate cell-based records before any aggregation.
    cellular = df[df["bao_format"].eq(BAO_CELL_BASED)].copy()
    df = df[~df["bao_format"].eq(BAO_CELL_BASED)]

    low_confidence = df[
        df["confidence_score"].notna()
        & df["confidence_score"].lt(MIN_CONFIDENCE_SCORE)
    ]
    reject(low_confidence, "assay_target_confidence_too_low")
    df = df[
        df["confidence_score"].isna()
        | df["confidence_score"].ge(MIN_CONFIDENCE_SCORE)
    ]

    # ChEMBL's own curators flag suspect records; respect that.
    flagged = df[df["data_validity_comment"].notna()]
    reject(flagged, "chembl_data_validity_comment")
    df = df[df["data_validity_comment"].isna()]

    wrong_type = df[~df["standard_type"].isin(USABLE_TYPES)]
    reject(wrong_type, "endpoint_not_a_concentration")
    df = df[df["standard_type"].isin(USABLE_TYPES)]

    wrong_units = df[df["standard_units"] != "nM"]
    reject(wrong_units, "units_not_nM")
    df = df[df["standard_units"] == "nM"]

    missing = df[df["standard_value"].isna() | df["canonical_smiles"].isna()]
    reject(missing, "missing_value_or_structure")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])

    nonpositive = df[df["standard_value"] <= 0]
    reject(nonpositive, "nonpositive_concentration")
    df = df[df["standard_value"] > 0]

    # Split censored from exact before aggregating.
    relation = df["standard_relation"].fillna("=").str.strip()
    censored = df[relation != "="].copy()
    df = df[relation == "="].copy()

    # Standardise structures. Done after filtering so the expensive step runs
    # on the smallest possible set.
    records = []
    unparseable = []
    for row in df.itertuples(index=False):
        mol = standardize_mol(row.canonical_smiles)
        if mol is None:
            unparseable.append(row._asdict())
            continue
        warheads = find_warheads(mol)
        records.append(
            {
                "molecule_chembl_id": row.molecule_chembl_id,
                "smiles": Chem.MolToSmiles(mol),
                "scaffold": murcko_scaffold(mol),
                "standard_type": row.standard_type,
                "value_nm": row.standard_value,
                "pic50": to_pic50(row.standard_value),
                "assay_chembl_id": row.assay_chembl_id,
                "document_chembl_id": row.document_chembl_id,
                "bao_label": row.bao_label,
                "covalent": is_covalent(mol),
                "warheads": "|".join(warheads),
                **basic_properties(mol),
            }
        )
    if unparseable:
        reject(pd.DataFrame(unparseable), "structure_unparseable")

    activities = pd.DataFrame(records)
    if activities.empty:
        raise RuntimeError("no usable activity records survived filtering")

    out_of_range = activities[
        (activities["pic50"] < MIN_PIC50) | (activities["pic50"] > MAX_PIC50)
    ]
    reject(out_of_range, "pic50_out_of_plausible_range")
    activities = activities[
        activities["pic50"].between(MIN_PIC50, MAX_PIC50)
    ]

    # Aggregate replicates per compound.
    grouped = activities.groupby("smiles", sort=False)
    aggregated = grouped.agg(
        molecule_chembl_id=("molecule_chembl_id", "first"),
        scaffold=("scaffold", "first"),
        pic50=("pic50", "median"),
        pic50_min=("pic50", "min"),
        pic50_max=("pic50", "max"),
        n_measurements=("pic50", "size"),
        n_assays=("assay_chembl_id", "nunique"),
        n_documents=("document_chembl_id", "nunique"),
        endpoint_types=("standard_type", lambda s: "|".join(sorted(set(s)))),
        covalent=("covalent", "first"),
        warheads=("warheads", "first"),
        mw=("mw", "first"),
        clogp=("clogp", "first"),
        tpsa=("tpsa", "first"),
        hbd=("hbd", "first"),
        hba=("hba", "first"),
        rotb=("rotb", "first"),
        arom_rings=("arom_rings", "first"),
        heavy_atoms=("heavy_atoms", "first"),
        fsp3=("fsp3", "first"),
        formal_charge=("formal_charge", "first"),
    ).reset_index()

    aggregated["pic50_spread"] = aggregated["pic50_max"] - aggregated["pic50_min"]

    inconsistent = aggregated[aggregated["pic50_spread"] > MAX_REPLICATE_SPREAD]
    reject(
        inconsistent.rename(columns={"smiles": "canonical_smiles"}),
        f"replicate_spread_over_{MAX_REPLICATE_SPREAD}_log",
    )
    final = aggregated[
        aggregated["pic50_spread"] <= MAX_REPLICATE_SPREAD
    ].copy()

    final = final.sort_values("pic50", ascending=False).reset_index(drop=True)

    rejected_df = (
        pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame()
    )
    return final, cellular, censored, rejected_df


def main() -> int:
    if not RAW.exists():
        print(
            f"{RAW} not found. Run src/data/fetch_chembl.py first.",
            file=sys.stderr,
        )
        return 1

    PROCESSED.mkdir(parents=True, exist_ok=True)
    final, cellular, censored, rejected = prepare()

    final.to_csv(PROCESSED / "lgmn_regression.csv", index=False)
    cellular.to_csv(PROCESSED / "lgmn_cellular.csv", index=False)
    censored.to_csv(PROCESSED / "lgmn_censored.csv", index=False)
    rejected.to_csv(PROCESSED / "lgmn_rejected.csv", index=False)

    raw_n = len(pd.read_csv(RAW))
    print(f"Raw records:                 {raw_n}")
    print(f"Rejected:                    {len(rejected)}")
    if not rejected.empty:
        for reason, count in rejected["reject_reason"].value_counts().items():
            print(f"    {reason:42} {count}")
    print(f"Cell-based (kept separate):  {len(cellular)}")
    print(f"Censored (>/< relations):    {len(censored)}")
    print(f"\nModelling set:               {len(final)} unique compounds")
    print(
        f"  covalent warhead:          {int(final['covalent'].sum())} "
        f"({100 * final['covalent'].mean():.0f}%)"
    )
    print(
        f"  pIC50 range:               {final['pic50'].min():.2f} "
        f"to {final['pic50'].max():.2f} "
        f"(median {final['pic50'].median():.2f})"
    )
    print(f"  unique Murcko scaffolds:   {final['scaffold'].nunique()}")
    print(
        f"  compounds with >1 measurement: "
        f"{int((final['n_measurements'] > 1).sum())}"
    )

    singleton_scaffolds = (
        final.groupby("scaffold").size().eq(1).sum()
    )
    print(f"  scaffolds seen only once:  {singleton_scaffolds}")

    if not final.empty:
        top = final.head(5)[["molecule_chembl_id", "pic50", "covalent", "warheads"]]
        print("\nMost potent compounds:")
        print(top.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
