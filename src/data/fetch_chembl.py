"""Fetch LGMN (legumain) bioactivity data from ChEMBL.

Resolves the ChEMBL target from the UniProt accession rather than a hard-coded
target ID, then pulls every activity record with a concentration-type endpoint.

Assay-level metadata is fetched alongside the activities because the activity
records alone cannot distinguish assays that must not be pooled. Every LGMN
record carries assay_type 'B', including a HEK293-A cell-based assay whose
IC50s run 1-2 log units weaker than the matching enzymatic ones, and including
two Schistosoma mansoni assays annotated against the human target. Separating
those requires bao_format and assay_tax_id, which live on the assay.

Writes:
    data/raw/chembl_lgmn_targets.csv    one row per matching ChEMBL target
    data/raw/chembl_lgmn_activities.csv raw activity records, unfiltered
    data/raw/chembl_lgmn_assays.csv     assay metadata for every assay seen

Nothing here cleans or filters the data — that happens in prepare_dataset.py, so
the raw pull stays reproducible and auditable.

Usage:
    python src/data/fetch_chembl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client

UNIPROT_ACCESSION = "Q99538"  # human legumain / asparaginyl endopeptidase
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Endpoints worth keeping. Everything else (percent inhibition, unspecified
# "Activity", etc.) is kept in the raw file but will be dropped downstream.
CONCENTRATION_TYPES = ["IC50", "Ki", "Kd", "EC50"]


def find_targets(accession: str) -> pd.DataFrame:
    """Return ChEMBL targets mapped to a UniProt accession."""
    hits = new_client.target.filter(
        target_components__accession=accession
    ).only(
        ["target_chembl_id", "pref_name", "target_type", "organism"]
    )
    return pd.DataFrame(list(hits))


def fetch_activities(target_ids: list[str]) -> pd.DataFrame:
    """Return all activity records for the given ChEMBL target IDs."""
    frames = []
    for target_id in target_ids:
        records = new_client.activity.filter(
            target_chembl_id=target_id
        ).only(
            [
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_type",
                "standard_relation",
                "standard_value",
                "standard_units",
                "pchembl_value",
                "assay_chembl_id",
                "assay_description",
                "assay_type",
                "target_chembl_id",
                "document_chembl_id",
                "data_validity_comment",
                "potential_duplicate",
            ]
        )
        frame = pd.DataFrame(list(records))
        print(f"  {target_id}: {len(frame)} activity records")
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_assays(assay_ids: list[str]) -> pd.DataFrame:
    """Return metadata for the given assays, in batches to keep URLs short."""
    fields = [
        "assay_chembl_id",
        "assay_organism",
        "assay_tax_id",
        "assay_cell_type",
        "assay_tissue",
        "assay_type",
        "bao_format",
        "bao_label",
        "confidence_score",
        "description",
        "target_chembl_id",
    ]
    frames = []
    batch_size = 20
    for start in range(0, len(assay_ids), batch_size):
        batch = assay_ids[start : start + batch_size]
        records = new_client.assay.filter(
            assay_chembl_id__in=batch
        ).only(fields)
        frames.append(pd.DataFrame(list(records)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Resolving ChEMBL targets for UniProt {UNIPROT_ACCESSION} ...")
    targets = find_targets(UNIPROT_ACCESSION)
    if targets.empty:
        print("No ChEMBL target found for that accession.", file=sys.stderr)
        return 1

    print(targets.to_string(index=False))
    targets.to_csv(RAW_DIR / "chembl_lgmn_targets.csv", index=False)

    print("\nFetching activities ...")
    activities = fetch_activities(targets["target_chembl_id"].tolist())

    if activities.empty:
        print(
            "\nNo activity records at all. If this holds up, a supervised "
            "classifier is off the table and the project should pivot to "
            "docking plus similarity search against known legumain "
            "inhibitors from the literature.",
            file=sys.stderr,
        )
        return 1

    out = RAW_DIR / "chembl_lgmn_activities.csv"
    activities.to_csv(out, index=False)

    print(f"\nWrote {len(activities)} records to {out}")
    print(f"Unique compounds: {activities['molecule_chembl_id'].nunique()}")
    print("\nRecords by endpoint type:")
    print(activities["standard_type"].value_counts().head(15).to_string())

    assay_ids = sorted(activities["assay_chembl_id"].dropna().unique())
    print(f"\nFetching metadata for {len(assay_ids)} assays ...")
    assays = fetch_assays(assay_ids)
    assays.to_csv(RAW_DIR / "chembl_lgmn_assays.csv", index=False)
    if not assays.empty:
        print(
            assays[
                ["assay_chembl_id", "assay_organism", "bao_label", "confidence_score"]
            ].to_string(index=False)
        )

    usable = activities[activities["standard_type"].isin(CONCENTRATION_TYPES)]
    print(
        f"\nWith a usable concentration endpoint "
        f"({'/'.join(CONCENTRATION_TYPES)}): "
        f"{len(usable)} records, "
        f"{usable['molecule_chembl_id'].nunique()} unique compounds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
