"""Add legumain bioactivity from PubChem to the ChEMBL-derived dataset.

180 compounds over 55 scaffolds -- four of which cover half the set -- is the
ceiling on what any model in this project can learn, so it is worth checking
whether ChEMBL is actually the whole of the public record. It is not: across
PubChem's legumain assays there are structures absent from the ChEMBL pull.

Two things to know about this source:

- **Much of it is ChEMBL re-deposited.** PubChem ingests ChEMBL, so assay
  names and values recur. Deduplication is by standardised canonical SMILES,
  and where a compound appears in both the ChEMBL record is kept, because it
  carries the assay metadata (bao_format, organism, confidence) that PubChem's
  concise view does not.
- **Cell-based assays must still be separated**, and here that is harder.
  There is no bao_format field, so the assay *name* is matched against
  cell-line and cell-assay wording. That is a weaker test than the one used
  for ChEMBL, and anything matched is excluded from the modelling set rather
  than risk repeating the mistake that pooled enzymatic and cellular potency.

Writes:
    data/raw/pubchem_lgmn_activities.csv   everything retrieved
    data/processed/lgmn_combined.csv       ChEMBL + PubChem-only compounds

Usage:
    python src/data/fetch_pubchem.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import (  # noqa: E402
    MAX_PIC50,
    MAX_REPLICATE_SPREAD,
    MIN_PIC50,
    murcko_scaffold,
    to_pic50,
)
from data.standardize import (  # noqa: E402
    basic_properties,
    find_warheads,
    is_covalent,
    standardize_mol,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
LGMN_GENE_ID = 5641
UNIPROT = "Q99538"

USABLE_ACTIVITY = {"IC50", "Ki", "Kd", "EC50"}

# Wording that marks an assay as cell-based. Cruder than ChEMBL's bao_format,
# so it is deliberately broad: a biochemical assay wrongly excluded costs one
# compound, while a cellular assay wrongly included corrupts the labels.
CELLULAR_MARKERS = (
    "cell", "cells", "cellular", "hek293", "mcf7", "mcf-7", "hela",
    "raw264", "thp-1", "in vivo", "lysate", "intracellular",
)

# PubChem renamed its SMILES property fields; a request for CanonicalSMILES
# comes back under ConnectivitySMILES. Reading one fixed key silently yields
# no structures at all, which looks like "PubChem has nothing to add".
SMILES_KEYS = ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES")


def _get(url: str, timeout: int = 120, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def assay_ids() -> list[int]:
    payload = json.loads(
        _get(f"{PUBCHEM}/assay/target/geneid/{LGMN_GENE_ID}/aids/JSON")
    )
    return payload.get("IdentifierList", {}).get("AID", [])


def fetch_activities(aids: list[int]) -> pd.DataFrame:
    """Concise activity rows for each assay."""
    frames = []
    for aid in aids:
        try:
            text = _get(f"{PUBCHEM}/assay/aid/{aid}/concise/CSV").decode()
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        rows = list(csv.DictReader(io.StringIO(text)))
        if rows:
            frames.append(pd.DataFrame(rows))
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_structures(cids: list[int]) -> dict[int, str]:
    """Isomeric SMILES per CID, tolerating PubChem's renamed property keys."""
    structures: dict[int, str] = {}
    for start in range(0, len(cids), 100):
        batch = cids[start : start + 100]
        url = (
            f"{PUBCHEM}/compound/cid/"
            + ",".join(str(c) for c in batch)
            + "/property/IsomericSMILES/JSON"
        )
        try:
            payload = json.loads(_get(url))
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        for record in payload.get("PropertyTable", {}).get("Properties", []):
            smiles = next(
                (record[key] for key in SMILES_KEYS if record.get(key)), None
            )
            if smiles:
                structures[record["CID"]] = smiles
        time.sleep(0.25)
    return structures


def is_cellular(assay_name: str) -> bool:
    lowered = str(assay_name).lower()
    return any(marker in lowered for marker in CELLULAR_MARKERS)


def build(activities: pd.DataFrame) -> pd.DataFrame:
    """Clean PubChem rows into the same shape as the ChEMBL dataset."""
    if activities.empty:
        return pd.DataFrame()

    df = activities.copy()
    df["value_um"] = pd.to_numeric(
        df.get("Activity Value [uM]"), errors="coerce"
    )
    df["CID"] = pd.to_numeric(df.get("CID"), errors="coerce")

    df = df[
        df["Activity Name"].isin(USABLE_ACTIVITY)
        & df["Target Accession"].eq(UNIPROT)
        & df["value_um"].gt(0)
        & df["CID"].notna()
    ].copy()
    if df.empty:
        return pd.DataFrame()

    df["cellular"] = df["Assay Name"].map(is_cellular)
    df = df[~df["cellular"]]
    if df.empty:
        return pd.DataFrame()

    structures = fetch_structures(sorted(df["CID"].astype(int).unique()))

    records = []
    for row in df.itertuples(index=False):
        raw_smiles = structures.get(int(row.CID))
        if not raw_smiles:
            continue
        mol = standardize_mol(raw_smiles)
        if mol is None:
            continue
        records.append(
            {
                "cid": int(row.CID),
                "smiles": Chem.MolToSmiles(mol),
                "pic50": to_pic50(row.value_um * 1000.0),  # uM -> nM
                "activity_name": row._asdict().get("Activity Name"),
                "assay_name": row._asdict().get("Assay Name"),
                "pubmed_id": row._asdict().get("PubMed ID"),
            }
        )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    frame = frame[frame["pic50"].between(MIN_PIC50, MAX_PIC50)]

    aggregated = (
        frame.groupby("smiles")
        .agg(
            cid=("cid", "first"),
            pic50=("pic50", "median"),
            pic50_min=("pic50", "min"),
            pic50_max=("pic50", "max"),
            n_measurements=("pic50", "size"),
            pubmed_ids=("pubmed_id", lambda s: "|".join(sorted({str(x) for x in s if x}))),
        )
        .reset_index()
    )
    aggregated["pic50_spread"] = (
        aggregated["pic50_max"] - aggregated["pic50_min"]
    )
    aggregated = aggregated[
        aggregated["pic50_spread"] <= MAX_REPLICATE_SPREAD
    ].copy()

    properties, scaffolds, covalent, warheads = [], [], [], []
    for smi in aggregated["smiles"]:
        mol = standardize_mol(smi)
        properties.append(basic_properties(mol))
        scaffolds.append(murcko_scaffold(mol))
        covalent.append(is_covalent(mol))
        warheads.append("|".join(find_warheads(mol)))

    aggregated = pd.concat(
        [aggregated, pd.DataFrame(properties, index=aggregated.index)], axis=1
    )
    aggregated["scaffold"] = scaffolds
    aggregated["covalent"] = covalent
    aggregated["warheads"] = warheads
    aggregated["molecule_chembl_id"] = [
        f"PUBCHEM_CID_{int(c)}" for c in aggregated["cid"]
    ]
    aggregated["source"] = "pubchem"
    return aggregated


def main() -> int:
    chembl_path = PROCESSED / "lgmn_regression.csv"
    if not chembl_path.exists():
        print(f"{chembl_path} not found; run prepare_dataset.py", file=sys.stderr)
        return 1

    chembl = pd.read_csv(chembl_path)
    chembl["source"] = "chembl"

    print(f"Finding PubChem assays for LGMN (GeneID {LGMN_GENE_ID}) ...")
    aids = assay_ids()
    print(f"  {len(aids)} assays")

    activities = fetch_activities(aids)
    RAW.mkdir(parents=True, exist_ok=True)
    activities.to_csv(RAW / "pubchem_lgmn_activities.csv", index=False)
    print(f"  {len(activities)} activity rows retrieved")

    pubchem = build(activities)
    if pubchem.empty:
        print("No usable PubChem potency data after filtering.")
        return 0

    overlap = pubchem["smiles"].isin(set(chembl["smiles"]))
    novel = pubchem[~overlap]

    print(f"\nPubChem compounds with usable potency: {len(pubchem)}")
    print(f"  already in the ChEMBL set: {int(overlap.sum())}")
    print(f"  new to this project:       {len(novel)}")

    shared = [c for c in chembl.columns if c in novel.columns]
    combined = pd.concat(
        [chembl, novel[shared]], ignore_index=True
    ).sort_values("pic50", ascending=False)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PROCESSED / "lgmn_combined.csv", index=False)

    print(f"\nCombined dataset: {len(combined)} compounds "
          f"(was {len(chembl)}, +{len(novel)})")
    print(
        f"  unique scaffolds: {combined['scaffold'].nunique()} "
        f"(was {chembl['scaffold'].nunique()})"
    )
    print(
        f"  covalent: {int(combined['covalent'].sum())} "
        f"({100 * combined['covalent'].mean():.0f}%)"
    )
    print(
        f"  pIC50 range: {combined['pic50'].min():.2f}-"
        f"{combined['pic50'].max():.2f} "
        f"(median {combined['pic50'].median():.2f})"
    )
    print(f"\nWrote {PROCESSED / 'lgmn_combined.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
