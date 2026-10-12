"""Rebuild the docking checkpoint from pose files already on disk.

Pose SDFs are written as each compound finishes and carry smina's score in
their `minimizedAffinity` property, so a lost or truncated checkpoint can be
reconstructed without re-docking anything. At roughly 95 seconds per compound
that is the difference between seconds and hours.

Usage:
    python src/docking/recover_checkpoint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docking.dock import (  # noqa: E402
    ACTIVE_SITE_CONTACT,
    PROCESSED,
    RESULTS,
    catalytic_thiol_coord,
    pose_distance_to_thiol,
)


def recover() -> pd.DataFrame:
    pose_dir = RESULTS / "poses"
    if not pose_dir.exists():
        return pd.DataFrame()

    dataset = pd.read_csv(PROCESSED / "lgmn_regression.csv")
    smiles_by_id = dict(
        zip(dataset["molecule_chembl_id"], dataset["smiles"])
    )
    thiol = catalytic_thiol_coord()

    rows = []
    for pose_path in sorted(pose_dir.glob("*.sdf")):
        identifier = pose_path.stem
        if pose_path.stat().st_size == 0:
            continue

        affinities = []
        for mol in Chem.SDMolSupplier(str(pose_path), removeHs=False):
            if mol is not None and mol.HasProp("minimizedAffinity"):
                affinities.append(float(mol.GetProp("minimizedAffinity")))
        if not affinities:
            continue

        distance = pose_distance_to_thiol(pose_path, thiol)
        rows.append(
            {
                "molecule_chembl_id": identifier,
                "smiles": smiles_by_id.get(identifier, ""),
                "affinity": affinities[0],
                "n_poses": len(affinities),
                "dist_to_thiol": distance,
                "in_active_site": (
                    distance is not None and distance <= ACTIVE_SITE_CONTACT
                ),
                "status": "ok",
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    recovered = recover()
    if recovered.empty:
        print("No usable pose files found.", file=sys.stderr)
        return 1

    checkpoint = RESULTS / "docking_scores.partial.csv"
    existing = (
        pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
    )
    before = len(existing)

    if not existing.empty:
        # Prefer whatever the checkpoint already holds; only fill gaps.
        missing = recovered[
            ~recovered["molecule_chembl_id"].isin(
                existing["molecule_chembl_id"]
            )
        ]
        merged = pd.concat([existing, missing], ignore_index=True)
    else:
        merged = recovered

    merged.to_csv(checkpoint, index=False)
    print(f"Checkpoint rows: {before} -> {len(merged)}")
    print(f"Recovered from {len(recovered)} pose files")
    scored = merged[merged["affinity"].notna()]
    if not scored.empty:
        print(
            f"Affinity range: {scored['affinity'].min():.2f} to "
            f"{scored['affinity'].max():.2f} kcal/mol"
        )
        print(f"Wrote {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
