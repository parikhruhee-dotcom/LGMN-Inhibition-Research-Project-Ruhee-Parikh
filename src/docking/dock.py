"""Dock compounds into LGMN with smina, after validating the protocol.

The validation step is not optional decoration. A docking setup with a
misplaced box, a mangled receptor or bad ligand protonation still produces
confident-looking scores for everything, and nothing in the output reveals it.
Redocking the crystallographic ligand and measuring RMSD to its known pose is
the one available check that the protocol reproduces reality. Convention takes
RMSD < 2.0 A as success; this script refuses to run a screen if redocking
fails, unless explicitly overridden.

Each docked compound also gets the distance from its best pose to the
catalytic thiol. A compound can score well while sitting in a surface groove
at the edge of the box, and that number separates a real active-site pose from
a score that happens to be low.

Writes:
    results/docking/redock_validation.json
    results/docking/docking_scores.csv
    results/docking/poses/<id>.sdf

Usage:
    python src/docking/dock.py --validate-only
    python src/docking/dock.py [--limit N] [--exhaustiveness N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "docking"
CONDA_BIN = ROOT / ".conda-env" / "bin"
SMINA = CONDA_BIN / "smina"
OBRMS = CONDA_BIN / "obrms"

# Conventional threshold for "the protocol reproduced the crystal pose".
REDOCK_RMSD_THRESHOLD = 2.0

# Default search effort. smina's own default is 8; this machine has 2 cores,
# and raising it multiplies wall time across a few hundred compounds.
DEFAULT_EXHAUSTIVENESS = 8

# A pose whose nearest atom is further than this from the catalytic thiol is
# not engaging the catalytic machinery, whatever it scored.
ACTIVE_SITE_CONTACT = 6.0


def load_box() -> dict:
    path = PROCESSED / "docking_box.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run src/docking/prepare_receptor.py first"
        )
    return json.loads(path.read_text())


def catalytic_thiol_coord() -> np.ndarray:
    """Coordinates of the catalytic cysteine SG in the prepared receptor."""
    box = load_box()
    structure = PDBParser(QUIET=True).get_structure(
        "receptor", str(PROCESSED / "receptor.pdb")
    )
    for chain in next(iter(structure)):
        for residue in chain:
            if residue.id[1] == box["catalytic_cys"] and "SG" in residue:
                return np.array(residue["SG"].coord)
    raise RuntimeError("catalytic cysteine SG not found in prepared receptor")


def embed_3d(smiles: str, seed: int = 0) -> Chem.Mol | None:
    """Generate one reasonable 3D conformer.

    ETKDG with a fixed seed keeps the run reproducible. MMFF relaxation is
    attempted and silently skipped where the force field has no parameters for
    an atom type, which is common for boron and some hypervalent sulfur; an
    unrelaxed ETKDG conformer still docks acceptably.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Retry with random coordinates, which rescues strained macrocycles.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except (ValueError, RuntimeError):
        pass
    return mol


def run_smina(
    ligand_path: Path,
    out_path: Path,
    box: dict,
    exhaustiveness: int,
    seed: int = 0,
) -> list[float]:
    """Dock one ligand file and return the affinity of each returned pose."""
    command = [
        str(SMINA),
        "--receptor", str(PROCESSED / "receptor.pdb"),
        "--ligand", str(ligand_path),
        "--out", str(out_path),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--seed", str(seed),
        "--cpu", "2",
        "--quiet",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"smina failed on {ligand_path.name}: {result.stderr.strip()[:300]}"
        )

    affinities = []
    if out_path.exists():
        for mol in Chem.SDMolSupplier(str(out_path), removeHs=False):
            if mol is None:
                continue
            if mol.HasProp("minimizedAffinity"):
                affinities.append(float(mol.GetProp("minimizedAffinity")))
    return affinities


def pose_distance_to_thiol(sdf_path: Path, thiol: np.ndarray) -> float | None:
    """Nearest-atom distance from the top pose to the catalytic thiol."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
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
        if coords.size == 0:
            return None
        return float(np.linalg.norm(coords - thiol, axis=1).min())
    return None


def validate_redocking(exhaustiveness: int) -> dict:
    """Redock the crystallographic ligand and measure RMSD to its real pose."""
    box = load_box()
    reference = PROCESSED / "reference_ligand.sdf"
    if not reference.exists():
        raise FileNotFoundError(f"{reference} not found")

    RESULTS.mkdir(parents=True, exist_ok=True)
    redocked = RESULTS / "redocked_reference.sdf"

    print(
        f"Redocking {box['reference_ligand']} into {box['pdb_id']} "
        f"(exhaustiveness {exhaustiveness}) ..."
    )
    affinities = run_smina(reference, redocked, box, exhaustiveness)

    # obrms compares the docked poses against the crystal coordinates. It
    # reports one RMSD per pose in the docked file.
    result = subprocess.run(
        [str(OBRMS), str(reference), str(redocked)],
        capture_output=True,
        text=True,
    )
    rmsds = []
    for line in result.stdout.splitlines():
        parts = line.strip().rsplit(" ", 1)
        if len(parts) == 2:
            try:
                rmsds.append(float(parts[1]))
            except ValueError:
                continue

    thiol = catalytic_thiol_coord()
    summary = {
        "pdb_id": box["pdb_id"],
        "ligand": box["reference_ligand"],
        "exhaustiveness": exhaustiveness,
        "top_pose_affinity": affinities[0] if affinities else None,
        "top_pose_rmsd": rmsds[0] if rmsds else None,
        "best_pose_rmsd": min(rmsds) if rmsds else None,
        "n_poses": len(affinities),
        "threshold": REDOCK_RMSD_THRESHOLD,
        "top_pose_dist_to_thiol": pose_distance_to_thiol(redocked, thiol),
    }
    summary["passed"] = (
        summary["top_pose_rmsd"] is not None
        and summary["top_pose_rmsd"] < REDOCK_RMSD_THRESHOLD
    )

    (RESULTS / "redock_validation.json").write_text(json.dumps(summary, indent=2))

    print(f"  top pose affinity : {summary['top_pose_affinity']} kcal/mol")
    print(f"  top pose RMSD     : {summary['top_pose_rmsd']} A")
    print(f"  best pose RMSD    : {summary['best_pose_rmsd']} A  (of {len(rmsds)})")
    print(f"  dist to Cys SG    : {summary['top_pose_dist_to_thiol']} A")
    print(
        f"  verdict           : "
        f"{'PASS' if summary['passed'] else 'FAIL'} "
        f"(threshold {REDOCK_RMSD_THRESHOLD} A)"
    )
    return summary


def screen(
    compounds: pd.DataFrame,
    exhaustiveness: int,
    limit: int | None = None,
) -> pd.DataFrame:
    """Dock a table of compounds, returning scores and pose geometry."""
    box = load_box()
    thiol = catalytic_thiol_coord()
    pose_dir = RESULTS / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        compounds = compounds.head(limit)

    rows = []
    total = len(compounds)
    for counter, row in enumerate(compounds.itertuples(index=False), start=1):
        identifier = getattr(row, "molecule_chembl_id", None) or f"cmpd_{counter}"
        smiles = row.smiles

        mol = embed_3d(smiles)
        if mol is None:
            rows.append(
                {
                    "molecule_chembl_id": identifier,
                    "smiles": smiles,
                    "affinity": np.nan,
                    "dist_to_thiol": np.nan,
                    "status": "embedding_failed",
                }
            )
            continue

        pose_path = pose_dir / f"{identifier}.sdf"
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as handle:
            ligand_path = Path(handle.name)
        try:
            writer = Chem.SDWriter(str(ligand_path))
            writer.write(mol)
            writer.close()
            affinities = run_smina(
                ligand_path, pose_path, box, exhaustiveness
            )
            distance = pose_distance_to_thiol(pose_path, thiol)
            rows.append(
                {
                    "molecule_chembl_id": identifier,
                    "smiles": smiles,
                    "affinity": affinities[0] if affinities else np.nan,
                    "n_poses": len(affinities),
                    "dist_to_thiol": distance,
                    "in_active_site": (
                        distance is not None and distance <= ACTIVE_SITE_CONTACT
                    ),
                    "status": "ok" if affinities else "no_pose",
                }
            )
        except RuntimeError as exc:
            rows.append(
                {
                    "molecule_chembl_id": identifier,
                    "smiles": smiles,
                    "affinity": np.nan,
                    "dist_to_thiol": np.nan,
                    "status": f"smina_error: {exc}"[:120],
                }
            )
        finally:
            ligand_path.unlink(missing_ok=True)

        if counter % 10 == 0 or counter == total:
            done = pd.DataFrame(rows)
            ok = int((done["status"] == "ok").sum())
            print(f"  {counter}/{total} docked ({ok} scored)", flush=True)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="screen even if redocking validation fails",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED / "lgmn_regression.csv",
        help="CSV with smiles and molecule_chembl_id columns",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    validation = validate_redocking(args.exhaustiveness)
    if args.validate_only:
        return 0 if validation["passed"] else 2

    if not validation["passed"] and not args.force:
        print(
            "\nRedocking did not reproduce the crystal pose, so docking scores "
            "from\nthis setup cannot be trusted. Fix the protocol, or pass "
            "--force to\nscreen anyway and record that caveat.",
            file=sys.stderr,
        )
        return 2

    if not args.input.exists():
        print(f"{args.input} not found", file=sys.stderr)
        return 1

    compounds = pd.read_csv(args.input)
    print(f"\nScreening {len(compounds)} compounds ...")
    scores = screen(compounds, args.exhaustiveness, args.limit)

    out = RESULTS / "docking_scores.csv"
    scores.to_csv(out, index=False)
    print(f"\nWrote {out}")

    scored = scores[scores["affinity"].notna()]
    if not scored.empty:
        print(f"Scored {len(scored)}/{len(scores)} compounds")
        print(
            f"Affinity range: {scored['affinity'].min():.2f} to "
            f"{scored['affinity'].max():.2f} kcal/mol"
        )
        in_site = int(scored["in_active_site"].sum())
        print(
            f"Poses contacting the catalytic thiol "
            f"(<= {ACTIVE_SITE_CONTACT} A): {in_site}/{len(scored)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
