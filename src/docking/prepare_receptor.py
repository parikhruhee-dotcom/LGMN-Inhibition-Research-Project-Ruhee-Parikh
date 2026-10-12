"""Prepare the LGMN docking receptor and define the binding site.

Takes the template chosen by characterize_target.py, strips everything that
should not be in a docking receptor, and writes the box definition from the
reference ligand's own coordinates.

Decisions:

- **The reference ligand is extracted, not deleted.** Redocking it is the only
  way to test whether the protocol reproduces a known binding mode, so it is
  written out separately as the validation ligand.
- **Crystallographic waters are removed.** Keeping them would block the pocket
  for every docked compound, and there is no evidence which are structurally
  conserved versus incidental to this crystal.
- **The N-glycan (NAG) is removed.** It is a post-translational modification on
  the protein surface, not part of the binding site.
- **The box is centred on the reference ligand** rather than on the catalytic
  cysteine, because the ligand marks the volume a real inhibitor occupies,
  while the thiol sits at one edge of it.

Writes:
    data/processed/receptor.pdb         cleaned receptor
    data/processed/reference_ligand.sdf reference ligand for redocking
    data/processed/docking_box.json     centre and size

Usage:
    python src/docking/prepare_receptor.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, Select

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PDB_DIR = ROOT / "data" / "raw" / "pdb"
CONDA_BIN = ROOT / ".conda-env" / "bin"

# Padding around the reference ligand, per axis. 8 A each side gives a larger
# compound room to find a pose without opening the box so wide that docking
# wanders onto the protein surface.
BOX_PADDING = 8.0

# Everything that is crystallisation furniture rather than protein or ligand.
STRIP_RESIDUES = {
    "HOH", "DOD", "NAG", "BMA", "MAN", "FUC", "GAL", "NDG",
    "EDO", "GOL", "PEG", "PGE", "1PE", "P6G", "DMS", "MPD",
    "SO4", "PO4", "ACT", "FMT", "CIT", "FLC", "MES", "EPE", "TRS",
    "CL", "NA", "K", "CA", "ZN", "MN", "HG", "CD", "NI", "IOD", "BME",
}


class ReceptorSelect(Select):
    """Keep one protein chain; drop solvent, ions, glycans and ligands."""

    def __init__(self, chain_id: str) -> None:
        self.chain_id = chain_id

    def accept_chain(self, chain):  # noqa: N802 - Biopython API
        return chain.id == self.chain_id

    def accept_residue(self, residue):  # noqa: N802 - Biopython API
        name = residue.get_resname().strip()
        if name in STRIP_RESIDUES:
            return False
        # Keep standard residues and modified amino acids (which carry a
        # hetero flag but are part of the chain), drop free ligands.
        return residue.id[0] == " " or _is_modified_residue(residue)

    def accept_atom(self, atom):  # noqa: N802 - Biopython API
        # Alternate conformers would duplicate atoms in the receptor.
        return not atom.is_disordered() or atom.get_altloc() in (" ", "A")


def _is_modified_residue(residue) -> bool:
    """True for a hetero residue that is covalently part of the chain.

    Legumain contains an aspartimide (SNN) at position 147 and sometimes
    S-methylcysteine (SCH) at the catalytic position. Both are backbone
    residues; removing them would tear a hole in the chain next to the active
    site. They are recognised by carrying backbone atoms.
    """
    return all(atom in residue for atom in ("N", "CA", "C"))


def extract_reference_ligand(structure, template: dict) -> np.ndarray:
    """Write the reference ligand to SDF and return its coordinates."""
    model = next(iter(structure))
    target_name = template["ligand"]
    target_chain = template["ligand_chain"]
    target_resseq = template["ligand_resseq"]

    found = None
    for chain in model:
        if chain.id != target_chain:
            continue
        for residue in chain:
            if (
                residue.get_resname().strip() == target_name
                and residue.id[1] == target_resseq
            ):
                found = residue
                break
    if found is None:
        raise RuntimeError(
            f"reference ligand {target_name} {target_resseq} not found "
            f"in chain {target_chain}"
        )

    # Write as PDB then convert with Open Babel, which assigns bond orders and
    # protonation that a hand-written SDF would get wrong.
    ligand_pdb = PROCESSED / "reference_ligand.pdb"
    with ligand_pdb.open("w") as handle:
        for index, atom in enumerate(found, start=1):
            x, y, z = atom.coord
            element = atom.element.strip() or atom.get_id()[0]
            handle.write(
                f"HETATM{index:5d} {atom.get_id():<4}{target_name:>3} "
                f"{target_chain}{target_resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{element:>2}\n"
            )
        handle.write("END\n")

    ligand_sdf = PROCESSED / "reference_ligand.sdf"
    subprocess.run(
        [
            str(CONDA_BIN / "obabel"),
            str(ligand_pdb),
            "-O",
            str(ligand_sdf),
            "-h",
        ],
        check=True,
        capture_output=True,
    )

    return np.array([atom.coord for atom in found])


def main() -> int:
    template_path = PROCESSED / "docking_template.json"
    if not template_path.exists():
        print(
            f"{template_path} not found. Run "
            "src/docking/characterize_target.py first.",
            file=sys.stderr,
        )
        return 1

    template = json.loads(template_path.read_text())
    pdb_id = template["pdb_id"]
    cif_path = PDB_DIR / f"{pdb_id}.cif"
    if not cif_path.exists():
        print(f"{cif_path} not found.", file=sys.stderr)
        return 1

    print(f"Template {pdb_id} at {template['resolution']} A")
    print(f"Reference ligand {template['ligand']} in chain {template['ligand_chain']}")

    # MMCIFParser (not Fast) is needed here because writing a PDB requires the
    # full atom metadata that the fast parser omits.
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(cif_path))

    ligand_coords = extract_reference_ligand(structure, template)
    centre = ligand_coords.mean(axis=0)
    extent = ligand_coords.max(axis=0) - ligand_coords.min(axis=0)
    size = extent + 2 * BOX_PADDING

    receptor_chain = template["ligand_chain"]
    io = PDBIO()
    io.set_structure(structure)
    receptor_path = PROCESSED / "receptor.pdb"
    io.save(str(receptor_path), select=ReceptorSelect(receptor_chain))

    atom_count = sum(
        1 for line in receptor_path.read_text().splitlines()
        if line.startswith("ATOM") or line.startswith("HETATM")
    )
    print(f"\nReceptor: chain {receptor_chain}, {atom_count} atoms -> {receptor_path}")
    print(f"Reference ligand: {len(ligand_coords)} atoms")
    print(
        f"Box centre: "
        f"({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f})"
    )
    print(f"Box size:   ({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}) A")

    box = {
        "pdb_id": pdb_id,
        "receptor_chain": receptor_chain,
        "reference_ligand": template["ligand"],
        "center_x": float(centre[0]),
        "center_y": float(centre[1]),
        "center_z": float(centre[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
        "padding": BOX_PADDING,
        "catalytic_cys": template["catalytic_cys"],
        "catalytic_his": template["catalytic_his"],
    }
    (PROCESSED / "docking_box.json").write_text(json.dumps(box, indent=2))
    print(f"Wrote {PROCESSED / 'docking_box.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
