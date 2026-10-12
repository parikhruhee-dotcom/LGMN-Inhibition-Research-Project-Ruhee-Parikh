"""Prepare an LGMN docking receptor and binding-site box from a PDB entry.

Usable as a library (prepare_template) or a CLI. Each template gets its own
directory under data/processed/templates/<PDBID>_<LIGAND>/ holding the cleaned
receptor, the reference ligand, and the box, so several templates can be
prepared and compared rather than one being assumed correct.

Decisions:

- **The reference ligand is extracted, not deleted.** Redocking it is the only
  way to test whether the protocol reproduces a known binding mode.
- **Crystallographic waters are removed.** Keeping them blocks the pocket for
  every docked compound, and there is no evidence which are conserved rather
  than incidental to this crystal.
- **The N-glycan (NAG) is removed.** A surface post-translational modification,
  not part of the binding site.
- **Modified backbone residues are kept.** Legumain carries an aspartimide
  (SNN) at residue 147, beside the active site, and sometimes an S-methylated
  catalytic cysteine. They are hetero records but part of the chain; dropping
  them tears a hole next to the pocket.
- **Box padding defaults to 4 A, not 8.** Measured on this target: redocking
  5KN into 5LUA gave a best-pose RMSD of 3.25 A at 5 A padding and 1.44 A at
  3 A padding, with the 8 A box putting the top pose 10.4 A away. A box much
  larger than the pocket lets the search wander onto the surface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, MMCIFParser, Select

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TEMPLATE_DIR = PROCESSED / "templates"
PDB_DIR = ROOT / "data" / "raw" / "pdb"
CONDA_BIN = ROOT / ".conda-env" / "bin"

DEFAULT_PADDING = 4.0

STRIP_RESIDUES = {
    "HOH", "DOD", "NAG", "BMA", "MAN", "FUC", "GAL", "NDG",
    "EDO", "GOL", "PEG", "PGE", "1PE", "P6G", "PG4", "DMS", "MPD",
    "SO4", "PO4", "ACT", "FMT", "CIT", "FLC", "MES", "EPE", "TRS",
    "CL", "NA", "K", "CA", "ZN", "MN", "HG", "CD", "NI", "IOD", "BME",
}


def _is_modified_residue(residue) -> bool:
    """True for a hetero residue that is covalently part of the chain."""
    return all(atom in residue for atom in ("N", "CA", "C"))


class ReceptorSelect(Select):
    """Keep one protein chain; drop solvent, ions, glycans and ligands."""

    def __init__(self, chain_id: str) -> None:
        self.chain_id = chain_id

    def accept_chain(self, chain):  # noqa: N802 - Biopython API
        return chain.id == self.chain_id

    def accept_residue(self, residue):  # noqa: N802 - Biopython API
        if residue.get_resname().strip() in STRIP_RESIDUES:
            return False
        return residue.id[0] == " " or _is_modified_residue(residue)

    def accept_atom(self, atom):  # noqa: N802 - Biopython API
        return not atom.is_disordered() or atom.get_altloc() in (" ", "A")


def _write_ligand(residue, out_dir: Path, name: str, chain: str, resseq: int) -> Path:
    """Write a ligand residue as SDF via Open Babel.

    Open Babel assigns bond orders and adds hydrogens, which a hand-written
    SDF would get wrong; the PDB intermediate carries only coordinates and
    elements, which is all the crystal provides.
    """
    ligand_pdb = out_dir / "reference_ligand.pdb"
    with ligand_pdb.open("w") as handle:
        for index, atom in enumerate(residue, start=1):
            x, y, z = atom.coord
            element = atom.element.strip() or atom.get_id()[0]
            handle.write(
                f"HETATM{index:5d} {atom.get_id():<4}{name:>3} "
                f"{chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{element:>2}\n"
            )
        handle.write("END\n")

    ligand_sdf = out_dir / "reference_ligand.sdf"
    subprocess.run(
        [str(CONDA_BIN / "obabel"), str(ligand_pdb), "-O", str(ligand_sdf), "-h"],
        check=True,
        capture_output=True,
    )
    return ligand_sdf


def find_ligand(structure, name: str, chain_id: str | None, resseq: int | None):
    """Locate a ligand residue, preferring an exact chain/resseq match."""
    model = next(iter(structure))
    fallback = None
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            if residue.get_resname().strip() != name:
                continue
            if resseq is None or residue.id[1] == resseq:
                return chain.id, residue
            fallback = (chain.id, residue)
    if fallback is not None:
        return fallback
    raise RuntimeError(f"ligand {name} not found (chain={chain_id}, resseq={resseq})")


def prepare_template(
    pdb_id: str,
    ligand_name: str,
    ligand_chain: str | None = None,
    ligand_resseq: int | None = None,
    catalytic_cys: int = 189,
    catalytic_his: int = 148,
    padding: float = DEFAULT_PADDING,
    box_from: Path | None = None,
) -> dict:
    """Write receptor, reference ligand and box for one template.

    `box_from` optionally sizes the box from a different ligand's coordinates
    (already in this structure's frame), so that a pocket can be boxed by a
    drug-sized ligand even when the template's own ligand is a small fragment.
    """
    cif_path = PDB_DIR / f"{pdb_id.upper()}.cif"
    if not cif_path.exists():
        raise FileNotFoundError(f"{cif_path} not found")

    # The full parser (not Fast) is required: writing a PDB needs the atom
    # metadata the fast parser omits.
    structure = MMCIFParser(QUIET=True).get_structure(pdb_id, str(cif_path))
    chain_id, residue = find_ligand(
        structure, ligand_name, ligand_chain, ligand_resseq
    )

    out_dir = TEMPLATE_DIR / f"{pdb_id.upper()}_{ligand_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ligand_coords = np.array([atom.coord for atom in residue])
    _write_ligand(
        residue, out_dir, ligand_name, chain_id, residue.id[1]
    )

    box_coords = ligand_coords
    box_source = f"{ligand_name} (own ligand)"
    if box_from is not None and box_from.exists():
        external = np.loadtxt(box_from)
        if external.size:
            box_coords = external.reshape(-1, 3)
            box_source = str(box_from.name)

    centre = box_coords.mean(axis=0)
    extent = box_coords.max(axis=0) - box_coords.min(axis=0)
    size = extent + 2 * padding

    io = PDBIO()
    io.set_structure(structure)
    receptor_path = out_dir / "receptor.pdb"
    io.save(str(receptor_path), select=ReceptorSelect(chain_id))

    atom_count = sum(
        1
        for line in receptor_path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )

    box = {
        "pdb_id": pdb_id.upper(),
        "receptor_chain": chain_id,
        "reference_ligand": ligand_name,
        "reference_resseq": residue.id[1],
        "reference_atoms": len(ligand_coords),
        "box_source": box_source,
        "center_x": float(centre[0]),
        "center_y": float(centre[1]),
        "center_z": float(centre[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "padding": padding,
        "catalytic_cys": catalytic_cys,
        "catalytic_his": catalytic_his,
        "receptor_atoms": atom_count,
        "directory": str(out_dir),
    }
    (out_dir / "box.json").write_text(json.dumps(box, indent=2))
    return box


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-id", default=None)
    parser.add_argument("--ligand", default=None)
    parser.add_argument("--chain", default=None)
    parser.add_argument("--resseq", type=int, default=None)
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    args = parser.parse_args()

    if args.pdb_id and args.ligand:
        pdb_id, ligand = args.pdb_id, args.ligand
        chain, resseq = args.chain, args.resseq
        cys, his = 189, 148
    else:
        template_path = PROCESSED / "docking_template.json"
        if not template_path.exists():
            print(
                f"{template_path} not found; run "
                "src/docking/characterize_target.py first, or pass "
                "--pdb-id and --ligand.",
                file=sys.stderr,
            )
            return 1
        template = json.loads(template_path.read_text())
        pdb_id = template["pdb_id"]
        ligand = template["ligand"]
        chain = template["ligand_chain"]
        resseq = template["ligand_resseq"]
        cys = template["catalytic_cys"]
        his = template["catalytic_his"]

    box = prepare_template(
        pdb_id, ligand, chain, resseq, cys, his, args.padding
    )

    print(f"Template {box['pdb_id']} / {box['reference_ligand']}")
    print(f"  receptor chain {box['receptor_chain']}, {box['receptor_atoms']} atoms")
    print(f"  reference ligand: {box['reference_atoms']} atoms")
    print(
        f"  box centre ({box['center_x']:.2f}, {box['center_y']:.2f}, "
        f"{box['center_z']:.2f})"
    )
    print(
        f"  box size   ({box['size_x']:.1f}, {box['size_y']:.1f}, "
        f"{box['size_z']:.1f}) A  padding {box['padding']} A"
    )
    print(f"  written to {box['directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
