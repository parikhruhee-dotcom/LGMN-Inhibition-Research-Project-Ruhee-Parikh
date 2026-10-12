"""Dock compounds into LGMN with smina, using the validated template.

Redocking lives in `validate_protocol.py`, which measures every candidate
template and records the result; this script reads that verdict rather than
repeating the work, so the two cannot disagree. If the protocol did not
reproduce its crystal pose as the top-ranked result, the screen refuses to run
without `--force`.

On this target it does not, and the screen is run anyway with `--force` for a
specific reason: docking every compound whose potency is already known gives a
direct measurement of whether docking score relates to potency here at all.
The Spearman correlation it prints is the answer, reported in `docs/DOCKING.md`
instead of being assumed in either direction.

Each docked compound also gets the distance from its best pose to the
catalytic thiol. A compound can score well while sitting in a surface groove
at the edge of the box, and that number separates a real active-site pose from
a score that merely happens to be low.

Writes:
    results/docking/docking_scores.csv
    results/docking/poses/<id>.sdf

Usage:
    python src/docking/dock.py [--force] [--limit N] [--exhaustiveness N]
"""

from __future__ import annotations

import argparse
import contextlib
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


def selected_template() -> dict:
    """The template chosen by validate_protocol.py."""
    path = PROCESSED / "selected_template.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run src/docking/validate_protocol.py first"
        )
    return json.loads(path.read_text())


def template_dir() -> Path:
    return Path(selected_template()["directory"])


def load_box() -> dict:
    path = template_dir() / "box.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text())


def receptor_path() -> Path:
    return template_dir() / "receptor.pdb"


def catalytic_thiol_coord() -> np.ndarray:
    """Coordinates of the catalytic cysteine SG in the prepared receptor."""
    box = load_box()
    structure = PDBParser(QUIET=True).get_structure(
        "receptor", str(receptor_path())
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
    # MMFF has no parameters for some atom types (boron, certain hypervalent
    # sulfur); an unrelaxed ETKDG conformer still docks acceptably.
    with contextlib.suppress(ValueError, RuntimeError):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
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
        "--receptor", str(receptor_path()),
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


def protocol_status() -> dict:
    """Report what validate_protocol.py found, without redoing the work.

    Redocking lives in validate_protocol.py, which measures every candidate
    template. Repeating it here would invite the two to disagree.
    """
    template = selected_template()
    validation_path = RESULTS / "protocol_validation.csv"
    if validation_path.exists():
        table = pd.read_csv(validation_path)
        row = table[
            (table["pdb_id"] == template["pdb_id"])
            & (table["ligand"] == template["ligand"])
        ]
        if not row.empty:
            template["sampling_found_pose"] = bool(
                row.iloc[0]["sampling_found_pose"]
            )
    return template


def screen(
    compounds: pd.DataFrame,
    exhaustiveness: int,
    limit: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Dock a table of compounds, returning scores and pose geometry.

    Results are checkpointed to disk as they are produced and completed
    compounds are skipped on a restart. A full screen takes about 95 seconds
    per compound on two cores -- roughly five hours for this dataset -- and
    writing only at the end would throw all of it away if the container
    stopped partway.
    """
    box = load_box()
    thiol = catalytic_thiol_coord()
    pose_dir = RESULTS / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULTS / "docking_scores.partial.csv"

    done: dict[str, dict] = {}
    if resume and checkpoint.exists():
        previous = pd.read_csv(checkpoint)
        done = {
            str(row["molecule_chembl_id"]): dict(row)
            for _, row in previous.iterrows()
            if row.get("status") == "ok"
        }
        if done:
            print(f"  resuming: {len(done)} compounds already scored")

    if limit:
        compounds = compounds.head(limit)

    rows = []
    total = len(compounds)
    for counter, row in enumerate(compounds.itertuples(index=False), start=1):
        identifier = getattr(row, "molecule_chembl_id", None) or f"cmpd_{counter}"
        smiles = row.smiles

        if identifier in done:
            rows.append(done[identifier])
            continue

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

        # Checkpoint every compound: the per-compound cost is a few
        # milliseconds against 95 seconds of docking, so there is no reason to
        # risk losing work to save it.
        progress = pd.DataFrame(rows)
        progress.to_csv(checkpoint, index=False)

        if counter % 10 == 0 or counter == total:
            ok = int((progress["status"] == "ok").sum())
            print(f"  {counter}/{total} docked ({ok} scored)", flush=True)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="screen even though the protocol did not validate prospectively",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED / "lgmn_regression.csv",
        help="CSV with smiles and molecule_chembl_id columns",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    template = protocol_status()

    print(
        f"Template {template['pdb_id']}/{template['ligand']} "
        f"({template['resolution']} A)"
    )
    print(
        f"  redocking top-pose RMSD {template['top_rmsd']:.2f} A, "
        f"best-of-20 {template['best_rmsd']:.2f} A"
    )

    if not template.get("validated"):
        message = (
            "\nThe protocol did NOT reproduce the crystal pose as its "
            "top-ranked\nresult. Sampling finds the right pose but the "
            "scoring function ranks a\ndecoy above it, which means these "
            "scores cannot be used to rank\ncompounds against each other."
        )
        if not args.force:
            print(
                message
                + "\n\nPass --force to screen anyway and treat the output as "
                "a pocket-fit\nfilter only.",
                file=sys.stderr,
            )
            return 2
        print(
            message
            + "\n\nProceeding with --force. The screen is run to measure "
            "directly whether\ndocking score carries any relationship to "
            "measured potency on this\ntarget; the answer is reported in "
            "docs/DOCKING.md rather than assumed."
        )

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

        # The direct test: does docking score track measured potency at all?
        merged = scored.merge(
            compounds[["molecule_chembl_id", "pic50"]],
            on="molecule_chembl_id",
            how="inner",
        )
        if len(merged) > 5:
            rho = merged["affinity"].corr(merged["pic50"], method="spearman")
            print(
                f"\nSpearman(docking affinity, measured pIC50) = {rho:.3f} "
                f"over {len(merged)} compounds"
            )
            print(
                "  A more negative affinity should mean higher potency, so a "
                "working\n  score would give a NEGATIVE correlation here."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
