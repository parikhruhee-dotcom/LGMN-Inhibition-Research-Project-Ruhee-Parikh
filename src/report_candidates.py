"""Build a per-compound dossier for the shortlisted candidates.

Everything else in this project reports aggregates. This answers the question
someone actually asks about a shortlist: for each compound, what is it, what
is known about it, what is wrong with it, and what should be done next.

Three things are assembled per compound that the summary tables cannot show:

- **A picture of the molecule**, because a chemist judges a structure by
  looking at it.
- **Which residues its docked pose touches**, computed from the pose file and
  the receptor. A good score from a pocket that is not the active site is not
  a prediction of inhibition, and four of the six candidates dock 7-9 A away
  from the catalytic thiol.
- **An explicit list of liabilities**, since every one of these compounds has
  at least one and a shortlist that hides them is worse than no shortlist.

Writes:
    results/figures/candidates.png
    docs/CANDIDATES.md

Usage:
    python src/report_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admet.filters import MIN_BBB_RULES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURE_DIR = RESULTS / "figures"
DOCS = ROOT / "docs"

# Any heavy atom of the pose within this distance of a residue counts as a
# contact. 4.5 A is the usual van der Waals contact range.
CONTACT_DISTANCE = 4.5

# A pose whose nearest atom is further than this from the catalytic thiol is
# not engaging the catalytic machinery, whatever it scored.
ACTIVE_SITE_CONTACT = 6.0


def load_receptor():
    template_path = PROCESSED / "selected_template.json"
    if not template_path.exists():
        return None, None
    import json

    template = json.loads(template_path.read_text())
    receptor_path = Path(template["directory"]) / "receptor.pdb"
    if not receptor_path.exists():
        return None, None
    structure = PDBParser(QUIET=True).get_structure("receptor", str(receptor_path))
    return structure, template


def pose_contacts(pose_path: Path, structure) -> list[str]:
    """Residues of the receptor within contact distance of the top pose."""
    if structure is None or not pose_path.exists():
        return []

    coords = None
    for mol in Chem.SDMolSupplier(str(pose_path), removeHs=False):
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
        break
    if coords is None or coords.size == 0:
        return []

    contacts = []
    for chain in next(iter(structure)):
        for residue in chain:
            atoms = np.array([atom.coord for atom in residue])
            if atoms.size == 0:
                continue
            if np.linalg.norm(coords[:, None, :] - atoms[None, :, :], axis=2).min() <= (
                CONTACT_DISTANCE
            ):
                contacts.append(f"{residue.get_resname().title()}{residue.id[1]}")
    return contacts


def liabilities(row: pd.Series) -> list[str]:
    """Everything known to be wrong with a compound, stated plainly."""
    issues = []
    if row.get("covalent"):
        issues.append(
            f"covalent warhead ({row.get('warheads') or 'unclassified'}) -- "
            "irreversible binding raises selectivity and toxicity questions"
        )
    if int(row.get("brenk_alerts", 0) or 0) > 0:
        names = str(row.get("brenk_names") or "").replace("|", ", ")
        issues.append(f"structural alert: {names}")
    if int(row.get("pains_alerts", 0) or 0) > 0:
        issues.append("PAINS alert -- may interfere with assays non-specifically")
    if not row.get("cns_mpo_pass"):
        issues.append(f"CNS MPO {row.get('cns_mpo', float('nan')):.1f}, below 4.0")
    passed = row.get("bbb_rules_passed")
    total = row.get("bbb_rules_total")
    if pd.notna(passed) and passed < MIN_BBB_RULES:
        issues.append(f"meets only {int(passed)}/{int(total)} blood-brain barrier rules")
    if int(row.get("lipinski_violations", 0) or 0) > 1:
        issues.append(
            f"{int(row['lipinski_violations'])} Lipinski violations"
        )
    distance = row.get("dist_to_thiol")
    if pd.notna(distance) and distance > ACTIVE_SITE_CONTACT:
        issues.append(
            f"docked pose sits {distance:.1f} A from the catalytic thiol, so "
            "its score does not describe active-site binding"
        )
    if int(row.get("n_measurements", 1) or 1) < 2:
        issues.append("potency rests on a single published measurement")
    return issues


def draw_candidates(frame: pd.DataFrame) -> None:
    mols, legends = [], []
    for row in frame.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(
            f"{row.molecule_chembl_id}\npIC50 {row.pic50:.2f}  "
            f"MPO {row.cns_mpo:.1f}"
        )
    if not mols:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(380, 300),
        legends=legends,
        returnPNG=False,
    )
    image.save(str(FIGURE_DIR / "candidates.png"))


def write_doc(frame: pd.DataFrame, structure, template) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    pose_dir = RESULTS / "docking" / "poses"

    lines = [
        "# Candidate dossier",
        "",
        "Generated by `src/report_candidates.py`.",
        "",
        "The compounds that are both **non-covalent** and **predicted to "
        "reach the brain**.",
        "Every other table in this project reports aggregates; this one says "
        "what each",
        "individual compound is and what is wrong with it.",
        "",
        "> **None of these has been tested by this project.** Potency values "
        "are published",
        "> measurements; everything else is predicted from structure.",
        "",
        "![Candidate structures](../results/figures/candidates.png)",
        "",
    ]

    if template:
        lines += [
            f"Docking poses come from **{template['pdb_id']}**. A pose is "
            "only evidence about",
            "inhibition if it sits in the active site, so the distance from "
            "each pose to the",
            f"catalytic thiol (Cys{template.get('catalytic_cys', 189)}) is "
            "reported with it.",
            "",
        ]

    for index, row in enumerate(frame.itertuples(index=False), start=1):
        contacts = pose_contacts(
            pose_dir / f"{row.molecule_chembl_id}.sdf", structure
        )
        issues = liabilities(pd.Series(row._asdict()))
        distance = getattr(row, "dist_to_thiol", float("nan"))

        lines += [
            f"## {index}. {row.molecule_chembl_id}",
            "",
            f"`{row.smiles}`",
            "",
            "| | |",
            "|---|---|",
            f"| Measured potency | pIC50 **{row.pic50:.2f}** "
            f"({10 ** (9 - row.pic50):.0f} nM) |",
            f"| Molecular weight | {row.mw:.0f} |",
            f"| cLogP / TPSA | {row.clogp:.2f} / {row.tpsa:.0f} A2 |",
            f"| CNS MPO | {row.cns_mpo:.2f} of 6 |",
            f"| BBB rules met | {int(row.bbb_rules_passed)}/"
            f"{int(row.bbb_rules_total)} |",
            f"| Docking score | "
            f"{getattr(row, 'affinity', float('nan')):.2f} kcal/mol |",
            "| Pose to catalytic thiol | "
            + (
                f"{distance:.1f} A"
                + (
                    " — **in the active site**"
                    if distance <= ACTIVE_SITE_CONTACT
                    else " — **outside the active site**"
                )
                if pd.notna(distance)
                else "not docked"
            )
            + " |",
            "",
        ]

        if contacts:
            shown = ", ".join(contacts[:14])
            more = f" (+{len(contacts) - 14} more)" if len(contacts) > 14 else ""
            lines += [f"**Pose contacts:** {shown}{more}", ""]

            # Touching the catalytic dyad is the difference between a pose
            # that is merely inside the pocket and one that engages the
            # machinery the compound is supposed to block.
            cys = template.get("catalytic_cys", 189) if template else 189
            his = template.get("catalytic_his", 148) if template else 148
            touched = [
                name
                for name in contacts
                if name.endswith(str(cys)) or name.endswith(str(his))
            ]
            if touched:
                lines += [
                    f"The pose contacts the catalytic dyad itself "
                    f"({', '.join(touched)}), which is what a real inhibitor "
                    "has to do.",
                    "",
                ]

        if issues:
            lines += ["**Liabilities**", ""]
            lines += [f"- {issue}" for issue in issues]
            lines += [""]

    lines += [
        "## What to do with these",
        "",
        "1. **Test them.** Six compounds against recombinant legumain is a "
        "small experiment",
        "   and it converts every prediction above into evidence.",
        "2. **Run an Ames test on the series first.** All six share an "
        "ortho-phenylenediamine,",
        "   a recognised genotoxicity alert, and for a drug taken over years "
        "that question",
        "   comes before potency.",
        "3. **Treat the docking scores carefully.** Most of these poses sit "
        "outside the",
        "   active site, which means the score is describing something other "
        "than",
        "   inhibition for those compounds.",
        "",
        "The one compound whose pose does sit in the active site, "
        "CHEMBL5185780, is also",
        "the fragment resolved in the 5LUA crystal structure, so its binding "
        "mode is known",
        "rather than predicted. That makes it the sensible starting point "
        "for any",
        "structure-based optimisation, despite being far from the most "
        "potent.",
        "",
    ]
    (DOCS / "CANDIDATES.md").write_text("\n".join(lines))


def main() -> int:
    shortlist_path = RESULTS / "shortlist.csv"
    if not shortlist_path.exists():
        print(f"{shortlist_path} not found; run src/admet/screen.py", file=sys.stderr)
        return 1

    shortlist = pd.read_csv(shortlist_path)
    candidates = shortlist[
        shortlist["cns_candidate"].fillna(False).astype(bool)
        & ~shortlist["covalent"].fillna(False).astype(bool)
    ].sort_values("composite_score", ascending=False)

    if candidates.empty:
        print("No compound is both non-covalent and CNS-suitable.")
        return 0

    structure, template = load_receptor()
    draw_candidates(candidates)
    write_doc(candidates, structure, template)

    print(f"Candidates: {len(candidates)}")
    in_site = int(
        (candidates["dist_to_thiol"] <= ACTIVE_SITE_CONTACT).sum()
    )
    print(f"  poses inside the active site: {in_site}/{len(candidates)}")
    print(f"  potency range: pIC50 {candidates['pic50'].min():.2f}"
          f"-{candidates['pic50'].max():.2f}")
    print(f"\nWrote {FIGURE_DIR / 'candidates.png'}")
    print(f"Wrote {DOCS / 'CANDIDATES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
