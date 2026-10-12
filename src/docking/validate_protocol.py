"""Choose the docking template by measuring redocking, not by assuming it.

Every candidate LGMN structure with a reasonable active-site ligand is
prepared, its own ligand is redocked, and the pose is compared to the crystal
coordinates. The template that reproduces its known binding mode is the one
used for screening.

Two RMSDs are reported for each, and the distinction decides what the result
means:

- **top-pose RMSD** is the prospective number. In a real screen the top-scored
  pose is what you get, with no crystal structure to check it against.
- **best-of-N RMSD** separates two different failures. If the best pose is
  good but the top pose is not, sampling works and the *scoring function*
  cannot rank the right answer first. If no pose is good, the search never
  found the binding mode -- a box, receptor or ligand-preparation problem.

That distinction matters here: 5KN, a 263 Da fragment binding at about
-5 kcal/mol, is found at 1.4 A but ranked behind a 3.3 A decoy, which is the
known weakness of empirical scoring functions on weak fragment binders rather
than a broken setup.

Writes:
    results/docking/protocol_validation.csv
    data/processed/selected_template.json

Usage:
    python src/docking/validate_protocol.py [--exhaustiveness N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docking.prepare_receptor import prepare_template  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "docking"
CONDA_BIN = ROOT / ".conda-env" / "bin"
SMINA = CONDA_BIN / "smina"
OBRMS = CONDA_BIN / "obrms"

REDOCK_RMSD_THRESHOLD = 2.0

# smina's own default is 8. 16 roughly doubles the search without the cost of
# the 64 used while tuning: on two cores a 36-heavy-atom flexible ligand at 64
# takes tens of minutes, and the point here is to rank templates against each
# other, not to extract the last tenth of an angstrom.
DEFAULT_EXHAUSTIVENESS = 16
NUM_MODES = 20

# Ligands below this size are fragments: too small to define a pocket and
# notoriously hard to redock. Still evaluated, but not preferred as templates.
DRUG_SIZED_ATOMS = 25


def candidate_templates() -> pd.DataFrame:
    """Candidate templates from the structure survey, best resolution first."""
    survey = pd.read_csv(PROCESSED / "target_survey.csv")
    candidates = survey[
        survey["mature"].astype(bool)
        & survey["n_dyads"].gt(0)
        & survey["ligand"].notna()
        & survey["dist_to_catalytic_cys"].notna()
        & survey["dist_to_catalytic_cys"].le(8.0)
        & survey["ligand_atoms"].ge(10)
    ].copy()
    candidates = candidates.sort_values(["resolution", "ligand_atoms"])
    # The same ligand appears in several entries (WSN in 7FQI, 7FQJ and 7FQL).
    # Redocking it three times measures the same chemistry at worse
    # resolution, so keep the best-resolution structure per ligand.
    return candidates.drop_duplicates(subset=["ligand"], keep="first")


def redock(template: dict, exhaustiveness: int) -> dict:
    """Redock a template's own ligand and measure RMSD to the crystal pose."""
    directory = Path(template["directory"])
    reference = directory / "reference_ligand.sdf"
    docked = directory / "redocked.sdf"

    command = [
        str(SMINA),
        "--receptor", str(directory / "receptor.pdb"),
        "--ligand", str(reference),
        "--autobox_ligand", str(reference),
        "--autobox_add", str(template["padding"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(NUM_MODES),
        "--seed", "0",
        "--cpu", "2",
        "--out", str(docked),
        "--quiet",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "status": f"smina_failed: {result.stderr.strip()[:120]}",
            "top_rmsd": np.nan,
            "best_rmsd": np.nan,
            "top_affinity": np.nan,
            "n_poses": 0,
        }

    # -f compares every docked pose against the single reference structure.
    # Without it obrms pairs files record-by-record and silently reports only
    # the first pose, which makes a successful best-of-N look like a failure.
    rmsd_result = subprocess.run(
        [str(OBRMS), "-f", str(reference), str(docked)],
        capture_output=True,
        text=True,
    )
    rmsds = []
    for line in rmsd_result.stdout.splitlines():
        parts = line.strip().rsplit(" ", 1)
        if len(parts) == 2:
            try:
                rmsds.append(float(parts[1]))
            except ValueError:
                continue

    affinities = []
    from rdkit import Chem

    if docked.exists():
        for mol in Chem.SDMolSupplier(str(docked), removeHs=False):
            if mol is not None and mol.HasProp("minimizedAffinity"):
                affinities.append(float(mol.GetProp("minimizedAffinity")))

    return {
        "status": "ok" if rmsds else "no_rmsd",
        "top_rmsd": rmsds[0] if rmsds else np.nan,
        "best_rmsd": min(rmsds) if rmsds else np.nan,
        "best_rank": int(np.argmin(rmsds)) + 1 if rmsds else None,
        "top_affinity": affinities[0] if affinities else np.nan,
        "n_poses": len(rmsds),
    }


def thiol_distance(template: dict) -> float | None:
    """Distance from the crystal ligand to the catalytic thiol, as a check."""
    directory = Path(template["directory"])
    structure = PDBParser(QUIET=True).get_structure(
        "receptor", str(directory / "receptor.pdb")
    )
    thiol = None
    for chain in next(iter(structure)):
        for residue in chain:
            if residue.id[1] == template["catalytic_cys"] and "SG" in residue:
                thiol = np.array(residue["SG"].coord)
    if thiol is None:
        return None

    from rdkit import Chem

    reference = directory / "reference_ligand.sdf"
    for mol in Chem.SDMolSupplier(str(reference), removeHs=False):
        if mol is None:
            continue
        conf = mol.GetConformer()
        coords = np.array(
            [
                list(conf.GetAtomPosition(i))
                for i in range(mol.GetNumAtoms())
                if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
            ]
        )
        return float(np.linalg.norm(coords - thiol, axis=1).min())
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    candidates = candidate_templates()
    if candidates.empty:
        print("No candidate templates in the survey.", file=sys.stderr)
        return 1

    print(f"Validating {len(candidates)} candidate templates ")
    print(f"(exhaustiveness {args.exhaustiveness}, {NUM_MODES} poses each)\n")

    rows = []
    for row in candidates.itertuples(index=False):
        try:
            template = prepare_template(
                row.pdb_id,
                row.ligand,
                row.ligand_chain,
                int(row.ligand_resseq),
                int(row.dyad_cys),
                int(row.dyad_his),
            )
        except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"  {row.pdb_id}/{row.ligand}: prep failed ({exc})")
            continue

        outcome = redock(template, args.exhaustiveness)
        record = {
            "pdb_id": row.pdb_id,
            "ligand": row.ligand,
            "resolution": row.resolution,
            "ligand_atoms": int(row.ligand_atoms),
            "covalent_in_crystal": bool(row.covalent),
            "crystal_dist_to_thiol": thiol_distance(template),
            "drug_sized": int(row.ligand_atoms) >= DRUG_SIZED_ATOMS,
            "receptor_atoms": template["receptor_atoms"],
            **outcome,
        }
        record["passed"] = bool(
            record["top_rmsd"] is not None
            and not np.isnan(record["top_rmsd"])
            and record["top_rmsd"] < REDOCK_RMSD_THRESHOLD
        )
        record["sampling_found_pose"] = bool(
            record["best_rmsd"] is not None
            and not np.isnan(record["best_rmsd"])
            and record["best_rmsd"] < REDOCK_RMSD_THRESHOLD
        )
        rows.append(record)

        verdict = "PASS" if record["passed"] else (
            "scoring-rank failure" if record["sampling_found_pose"] else "FAIL"
        )
        print(
            f"  {row.pdb_id}/{row.ligand:4} {row.resolution:4} A "
            f"{int(row.ligand_atoms):3} atoms  "
            f"top={record['top_rmsd']:.2f} best={record['best_rmsd']:.2f} "
            f"(rank {record['best_rank']})  {verdict}"
        )

    validation = pd.DataFrame(rows)
    validation.to_csv(RESULTS / "protocol_validation.csv", index=False)
    print(f"\nWrote {RESULTS / 'protocol_validation.csv'}")

    # Prefer a template that passes prospectively; among those prefer a
    # drug-sized ligand (a pocket boxed by a fragment is too small for the
    # compounds being screened) and then the best resolution.
    passing = validation[validation["passed"]].copy()
    if passing.empty:
        print(
            "\nNo template reproduced its crystal pose as the top-ranked "
            "result.\nDocking can still be used, but only as a coarse filter, "
            "and that\nlimitation has to be stated in the write-up.",
            file=sys.stderr,
        )
        ranked = validation.sort_values(
            ["sampling_found_pose", "drug_sized", "top_rmsd"],
            ascending=[False, False, True],
        )
    else:
        ranked = passing.sort_values(
            ["drug_sized", "resolution", "top_rmsd"],
            ascending=[False, True, True],
        )

    best = ranked.iloc[0]
    selected = {
        "pdb_id": best["pdb_id"],
        "ligand": best["ligand"],
        "resolution": float(best["resolution"]),
        "ligand_atoms": int(best["ligand_atoms"]),
        "covalent_in_crystal": bool(best["covalent_in_crystal"]),
        "top_rmsd": float(best["top_rmsd"]),
        "best_rmsd": float(best["best_rmsd"]),
        "validated": bool(best["passed"]),
        "directory": str(
            PROCESSED / "templates" / f"{best['pdb_id']}_{best['ligand']}"
        ),
    }
    (PROCESSED / "selected_template.json").write_text(
        json.dumps(selected, indent=2)
    )

    print(
        f"\nSelected {selected['pdb_id']}/{selected['ligand']} "
        f"({selected['resolution']} A, {selected['ligand_atoms']} atoms)"
    )
    print(
        f"  top-pose RMSD {selected['top_rmsd']:.2f} A -> "
        f"{'validated' if selected['validated'] else 'NOT validated'}"
    )
    if selected["covalent_in_crystal"]:
        print(
            "  Note: this ligand is covalently bound in the crystal. Redocking "
            "it\n  non-covalently tests the pocket shape and the protocol, but "
            "the real\n  complex forms a bond that the scoring function does "
            "not model."
        )
    print(f"Wrote {PROCESSED / 'selected_template.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
