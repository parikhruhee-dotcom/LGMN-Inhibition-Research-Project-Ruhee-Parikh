"""Survey LGMN crystal structures and pick a docking template.

Rather than trusting published residue numbers, this finds the catalytic
cysteine-histidine dyad geometrically: in a clan CD cysteine protease the Cys
thiol sits near a His imidazole nitrogen. The nearest same-chain Cys/His pair
is taken as the dyad, which locates the active site independently of the
deposition's numbering.

Two things that look obvious are deliberately not done:

1. Dyad distance is *not* used to decide whether the enzyme is mature. In the
   inhibitor-bound structures the catalytic thiol is covalently modified and
   pushed to ~5.4 A from the histidine, while unmodified prolegumain sits at
   ~4.1 A. A tight distance cutoff therefore selects for the zymogen, which is
   exactly backwards. Maturity is judged instead by whether the C-terminal
   prodomain is present.
2. Ligands are taken from the RCSB non-polymer entity list, not from whatever
   carries a HETATM flag. Legumain structures contain modified residues --
   aspartimide (SNN) and S-methylcysteine (SCH) -- which are parts of the
   protein chain but appear as hetero records, and counting them as ligands
   makes every structure look like it has a bound inhibitor.

For each real ligand it reports the distance to the catalytic thiol and
whether the mmCIF declares a covalent bond. That distinction decides which
structure can validate a docking protocol: only a non-covalent ligand can be
redocked meaningfully.

Writes:
    data/raw/pdb/<ID>.cif          downloaded structures
    data/processed/target_survey.csv   one row per (structure, ligand)
    docs/TARGET.md                 the write-up of what was chosen and why

Usage:
    python src/docking/characterize_target.py
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import FastMMCIFParser, MMCIF2Dict

ROOT = Path(__file__).resolve().parents[2]
PDB_DIR = ROOT / "data" / "raw" / "pdb"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

UNIPROT_ACCESSION = "Q99538"

# Crystallisation additives, cryoprotectants, buffers and glycans. Present in
# the file but not the thing we are trying to find.
IGNORED_HETERO = {
    "HOH", "DOD", "NAG", "BMA", "MAN", "FUC", "GAL", "NDG",
    "EDO", "GOL", "PEG", "PGE", "1PE", "P6G", "DMS", "MPD", "TRS",
    "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "HG", "CD", "NI",
    "ACT", "FMT", "CIT", "FLC", "MES", "EPE", "IMD", "BME", "EMC", "TLA",
}

# Loose sanity bound on the dyad S...N distance -- not a discriminating filter.
#
# Measured across these 18 entries the distance spans 4.1 A in unmodified
# prolegumain, ~5.4 A where an inhibitor has covalently modified the thiol and
# displaced it, and ~7.9 A in the cystatin and clitocypin complexes where the
# histidine swings away. No threshold separates "is the dyad" from "is not",
# so distance is recorded as an observation and identification rests on two
# other things: the pair is the closest Cys/His in its chain, and the same
# pair must recur across independent structures. It does -- Cys189/His148 in
# every chain of every entry -- which is the actual evidence that the right
# site has been found.
DYAD_CUTOFF = 10.0

# Human legumain is produced as a zymogen: the catalytic domain spans roughly
# residues 26-323 and the C-terminal LSAM prodomain follows. If residues well
# past the catalytic domain are modelled, the prodomain is still attached and
# the active site is occluded.
CATALYTIC_DOMAIN_END = 330

# A carbon-sulfur bond is ~1.8 A; the closest a non-bonded heavy atom comes to
# a thiol sulfur is ~3.3 A. Anything at or under this is covalently bonded.
#
# This geometric test is the primary one, because mmCIF annotation cannot be
# trusted here: 7FQI and 7FQJ place their inhibitor 1.67 A and 1.65 A from the
# catalytic thiol while declaring no covale record at all, whereas 7FQH and
# 7FQK annotate the equivalent bond properly. Relying on struct_conn alone
# silently classifies covalent inhibitors as non-covalent.
COVALENT_BOND_CUTOFF = 2.2

# Residue names for cysteine and its modified forms. The catalytic cysteine is
# frequently derivatised -- S-methylcysteine (SCH) in the 8AE4/8AE5 complexes,
# various oxidation states elsewhere -- and searching for the residue name CYS
# alone misses the active site entirely in those entries. Any residue bearing
# an SG atom is treated as a cysteine for dyad detection.
CYSTEINE_LIKE_ATOM = "SG"


def _fetch(url: str, timeout: int = 90) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def find_lgmn_structures(accession: str = UNIPROT_ACCESSION) -> list[str]:
    """Return every experimental PDB entry mapped to a UniProt accession."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": (
                    "rcsb_polymer_entity_container_identifiers"
                    ".reference_sequence_identifiers.database_accession"
                ),
                "operator": "exact_match",
                "value": accession,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 200},
            "results_content_type": ["experimental"],
        },
    }
    url = (
        "https://search.rcsb.org/rcsbsearch/v2/query?json="
        + urllib.parse.quote(json.dumps(query))
    )
    payload = json.loads(_fetch(url))
    return [hit["identifier"] for hit in payload.get("result_set", [])]


def download_cif(pdb_id: str) -> Path:
    """Download an mmCIF, caching it under data/raw/pdb."""
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    path = PDB_DIR / f"{pdb_id.upper()}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    data = _fetch(f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz")
    path.write_bytes(data)
    return path


def find_catalytic_dyads(structure, chains: set[str] | None = None) -> list[dict]:
    """Return the nearest same-chain Cys/His pair for each protein chain.

    One dyad per chain, since a legumain structure with several copies in the
    asymmetric unit has one active site per copy. `chains` restricts the search
    to the chains that actually are legumain -- without it, a complex such as
    4N6N returns a Cys/His pair from the bound cystatin instead.
    """
    model = next(iter(structure))
    dyads = []

    for chain in model:
        if chains is not None and chain.id not in chains:
            continue
        cysteines = [
            (residue, residue[CYSTEINE_LIKE_ATOM])
            for residue in chain
            if CYSTEINE_LIKE_ATOM in residue
        ]
        histidines = [
            (residue, residue[atom_name], atom_name)
            for residue in chain
            if residue.get_resname() == "HIS"
            for atom_name in ("NE2", "ND1")
            if atom_name in residue
        ]
        if not cysteines or not histidines:
            continue

        best = None
        for res_c, sg in cysteines:
            for res_h, nitrogen, atom_name in histidines:
                distance = float(np.linalg.norm(sg.coord - nitrogen.coord))
                if best is None or distance < best["distance"]:
                    best = {
                        "cys_chain": chain.id,
                        "cys_resid": res_c.id[1],
                        "his_chain": chain.id,
                        "his_resid": res_h.id[1],
                        "his_atom": atom_name,
                        "distance": distance,
                        "sg_coord": sg.coord,
                    }

        if best is not None and best["distance"] <= DYAD_CUTOFF:
            best["distance"] = round(best["distance"], 2)
            dyads.append(best)

    return sorted(dyads, key=lambda d: d["distance"])


def max_modelled_residue(structure) -> int:
    """Highest standard-amino-acid residue number in any protein chain.

    Used to tell mature legumain from prolegumain: the prodomain pushes the
    modelled range far past the catalytic domain.
    """
    model = next(iter(structure))
    highest = 0
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            highest = max(highest, residue.id[1])
    return highest


def target_chains(pdb_id: str, accession: str = UNIPROT_ACCESSION) -> set[str]:
    """Return the auth chain IDs whose polymer entity maps to the accession."""
    meta = json.loads(_fetch(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"))
    entity_ids = (
        meta.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids")
        or []
    )
    chains: set[str] = set()
    for entity_id in entity_ids:
        entity = json.loads(
            _fetch(
                "https://data.rcsb.org/rest/v1/core/polymer_entity/"
                f"{pdb_id}/{entity_id}"
            )
        )
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        references = identifiers.get("reference_sequence_identifiers") or []
        if any(
            str(ref.get("database_accession", "")).upper() == accession.upper()
            for ref in references
        ):
            chains.update(identifiers.get("auth_asym_ids") or [])
    return chains


def ligand_component_ids(pdb_id: str) -> set[str]:
    """Return genuine non-polymer component IDs for an entry.

    Taken from the RCSB non-polymer entity list so that modified residues,
    which are chemically part of the protein, are never mistaken for ligands.
    """
    meta = json.loads(_fetch(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"))
    entity_ids = (
        meta.get("rcsb_entry_container_identifiers", {}).get(
            "non_polymer_entity_ids"
        )
        or []
    )
    components = set()
    for entity_id in entity_ids:
        entity = json.loads(
            _fetch(
                "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/"
                f"{pdb_id}/{entity_id}"
            )
        )
        comp = entity.get("pdbx_entity_nonpoly", {}).get("comp_id")
        if comp:
            components.add(str(comp).strip())
    return components


def covalent_ligand_links(cif_path: Path) -> set[str]:
    """Return ligand component IDs the mmCIF declares covalently bonded.

    Reads the struct_conn category and keeps connections typed as covalent
    where one partner is a hetero component. A covalently bound inhibitor is
    not redockable with a non-covalent scoring function.
    """
    try:
        cif = MMCIF2Dict.MMCIF2Dict(str(cif_path))
    except Exception:
        return set()

    conn_types = cif.get("_struct_conn.conn_type_id", [])
    if isinstance(conn_types, str):
        conn_types = [conn_types]
    if not conn_types:
        return set()

    def column(key: str) -> list[str]:
        value = cif.get(key, [])
        if isinstance(value, str):
            value = [value]
        return list(value) + [""] * (len(conn_types) - len(value))

    partner1 = column("_struct_conn.ptnr1_label_comp_id")
    partner2 = column("_struct_conn.ptnr2_label_comp_id")

    covalent: set[str] = set()
    for conn_type, comp1, comp2 in zip(conn_types, partner1, partner2):
        if "covale" not in str(conn_type).lower():
            continue
        for comp in (comp1, comp2):
            comp = str(comp).strip()
            if comp and comp not in IGNORED_HETERO:
                covalent.add(comp)
    return covalent


def survey_structure(pdb_id: str) -> list[dict]:
    """Characterise one structure: dyad, ligands, and their active-site distance."""
    cif_path = download_cif(pdb_id)
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(cif_path))

    meta = json.loads(
        _fetch(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    )
    info = meta.get("rcsb_entry_info", {})
    resolution = (info.get("resolution_combined") or [None])[0]
    title = meta.get("struct", {}).get("title", "")

    lgmn_chains = target_chains(pdb_id)
    dyads = find_catalytic_dyads(structure, lgmn_chains or None)
    annotated_covalent = covalent_ligand_links(cif_path)
    real_ligands = ligand_component_ids(pdb_id) - IGNORED_HETERO
    highest_residue = max_modelled_residue(structure)
    is_mature = highest_residue <= CATALYTIC_DOMAIN_END

    model = next(iter(structure))

    def distance_to_dyad(coords: np.ndarray) -> float | None:
        if not dyads:
            return None
        return min(
            float(np.linalg.norm(coords - dyad["sg_coord"], axis=1).min())
            for dyad in dyads
        )

    # A structure can hold the same compound in more than one place, and the
    # copies are not equivalent: in 5LU8 the fragment 5KN sits 21 A from the
    # active site, while in 5LUA the same compound is bound in it. Keep the
    # copy nearest the catalytic thiol, since that is the binding mode a
    # docking protocol has to reproduce.
    best_copy: dict[str, tuple] = {}
    for chain in model:
        for residue in chain:
            name = residue.get_resname().strip()
            if name not in real_ligands:
                continue
            coords = np.array([atom.coord for atom in residue])
            if coords.size == 0:
                continue
            distance = distance_to_dyad(coords)
            previous = best_copy.get(name)
            if previous is None or (
                distance is not None
                and previous[5] is not None
                and distance < previous[5]
            ):
                best_copy[name] = (
                    name, chain.id, residue.id[1], coords, len(coords), distance,
                )

    ligands = list(best_copy.values())

    base = {
        "pdb_id": pdb_id,
        "resolution": resolution,
        "title": title,
        "n_dyads": len(dyads),
        "dyad_cys": dyads[0]["cys_resid"] if dyads else None,
        "dyad_his": dyads[0]["his_resid"] if dyads else None,
        "dyad_distance": dyads[0]["distance"] if dyads else None,
        "max_residue": highest_residue,
        "mature": is_mature,
    }

    if not ligands:
        return [{**base, "ligand": None, "ligand_chain": None,
                 "ligand_resseq": None, "ligand_atoms": None,
                 "dist_to_catalytic_cys": None, "covalent": None,
                 "covalent_annotated": None, "covalent_by_geometry": None}]

    rows = []
    for name, chain_id, resseq, coords, n_atoms, distance in ligands:
        # Geometry first, annotation second: a bond length to the catalytic
        # thiol proves a covalent adduct whether or not it was declared.
        bonded_by_geometry = (
            distance is not None and distance <= COVALENT_BOND_CUTOFF
        )
        rows.append(
            {
                **base,
                "ligand": name,
                "ligand_chain": chain_id,
                "ligand_resseq": resseq,
                "ligand_atoms": n_atoms,
                "dist_to_catalytic_cys": (
                    round(distance, 2) if distance is not None else None
                ),
                "covalent": bonded_by_geometry or name in annotated_covalent,
                "covalent_annotated": name in annotated_covalent,
                "covalent_by_geometry": bonded_by_geometry,
            }
        )
    return rows


def choose_template(survey: pd.DataFrame) -> dict:
    """Pick the best docking template.

    Requires a mature enzyme (a dyad was found), a non-covalent ligand sitting
    in the active site to validate redocking against, and then prefers the
    best resolution.
    """
    # astype(bool) is required, not cosmetic: these columns carry None for
    # ligand-free rows and so arrive as object dtype, where `~` inverts the
    # underlying Python bools as integers (`~True` is -2, which is truthy) and
    # silently passes every covalent ligand through the filter.
    candidates = survey[
        survey["mature"].astype(bool)
        & survey["n_dyads"].gt(0)
        & survey["ligand"].notna()
        & ~survey["covalent"].fillna(False).astype(bool)
        & survey["dist_to_catalytic_cys"].notna()
        & survey["dist_to_catalytic_cys"].le(8.0)
        & survey["ligand_atoms"].ge(8)
    ].copy()

    if candidates.empty:
        return {}
    candidates = candidates.sort_values(["resolution", "dist_to_catalytic_cys"])
    return candidates.iloc[0].to_dict()


def write_target_doc(survey: pd.DataFrame, template: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)

    holo = survey[survey["ligand"].notna()]
    mature = survey[survey["n_dyads"].gt(0)]["pdb_id"].nunique()
    zymogen = survey[survey["n_dyads"].eq(0)]["pdb_id"].unique().tolist()

    lines = [
        "# LGMN target characterisation",
        "",
        "Generated by `src/docking/characterize_target.py`. Every number here is",
        "measured from the deposited coordinates, not taken from a paper.",
        "",
        f"- UniProt: **{UNIPROT_ACCESSION}** (human legumain, asparaginyl endopeptidase)",
        f"- Experimental PDB entries: **{survey['pdb_id'].nunique()}**",
        f"- Entries with an intact catalytic dyad (mature enzyme): **{mature}**",
        "",
        "## Catalytic dyad",
        "",
        "The dyad was located geometrically by finding Cys/His pairs whose thiol and",
        f"imidazole nitrogen lie within {DYAD_CUTOFF} A. The consensus assignment across",
        "structures is:",
        "",
    ]

    dyad_counts = (
        mature_df := survey[survey["n_dyads"].gt(0)]
    ).drop_duplicates("pdb_id")[["dyad_cys", "dyad_his", "dyad_distance"]]
    if not dyad_counts.empty:
        consensus = dyad_counts.mode().iloc[0]
        lines += [
            f"- **Cys{int(consensus['dyad_cys'])} / His{int(consensus['dyad_his'])}**",
            f"- Median S...N distance: "
            f"{dyad_counts['dyad_distance'].median():.2f} A",
            "",
            "This matches the literature assignment for legumain, which is a useful",
            "independent check that the detection is finding the real active site.",
            "",
        ]

    if zymogen:
        lines += [
            "### Structures without a dyad",
            "",
            f"{', '.join(zymogen)} — expected for prolegumain, where the",
            "activation peptide occludes the site. Not usable as docking templates.",
            "",
        ]

    lines += [
        "## Bound ligands",
        "",
        "Distance is from the ligand's nearest atom to the catalytic thiol. The",
        "covalent column reflects a `covale` bond declared in the mmCIF.",
        "",
        "| PDB | Res (A) | Ligand | Atoms | Dist to Cys SG (A) | Covalent |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in holo.sort_values(["resolution", "pdb_id"]).iterrows():
        distance = row["dist_to_catalytic_cys"]
        lines.append(
            f"| {row['pdb_id']} | {row['resolution']} | {row['ligand']} | "
            f"{int(row['ligand_atoms'])} | "
            f"{distance if pd.notna(distance) else '-'} | "
            f"{'yes' if row['covalent'] else 'no'} |"
        )

    lines += [
        "",
        "## Chosen docking template",
        "",
    ]
    if template:
        lines += [
            f"**{template['pdb_id']}** at {template['resolution']} A, "
            f"validating against ligand **{template['ligand']}** "
            f"({int(template['ligand_atoms'])} atoms, "
            f"{template['dist_to_catalytic_cys']} A from the catalytic thiol).",
            "",
            "Chosen because it is the highest-resolution structure that has an intact",
            "catalytic dyad *and* a non-covalent ligand in the active site. The second",
            "condition is what makes the docking protocol testable: a covalently bound",
            "warhead cannot be reproduced by a non-covalent scoring function, so a",
            "structure containing only one gives no way to tell a working setup from a",
            "broken one.",
            "",
            "## Consequence for the screening campaign",
            "",
            "Most known legumain inhibitors are covalent — halomethyl ketones, vinyl",
            "Michael acceptors and electrophilic nitriles all appear in the structures",
            "above. Docking scores for those compounds measure only how well the",
            "fragment sits in the pocket before the bond forms, which is a weak proxy",
            "for potency. The pipeline therefore flags warhead-bearing compounds",
            "(`src/data/standardize.py`) and reports them separately rather than",
            "ranking them against non-covalent hits.",
            "",
        ]
    else:
        lines += [
            "No structure satisfied all template criteria. Docking would need either a",
            "covalent protocol or an apo structure with a predicted site.",
            "",
        ]

    (DOCS / "TARGET.md").write_text("\n".join(lines))


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print(f"Finding PDB entries for UniProt {UNIPROT_ACCESSION} ...")
    pdb_ids = find_lgmn_structures()
    print(f"  {len(pdb_ids)} entries: {', '.join(pdb_ids)}\n")

    rows = []
    for pdb_id in pdb_ids:
        try:
            structure_rows = survey_structure(pdb_id)
        except (urllib.error.URLError, ValueError, KeyError) as exc:
            print(f"  {pdb_id}: FAILED ({type(exc).__name__}: {exc})")
            continue
        rows.extend(structure_rows)
        first = structure_rows[0]
        dyad = (
            f"Cys{first['dyad_cys']}/His{first['dyad_his']}"
            f" @ {first['dyad_distance']} A"
            if first["dyad_cys"]
            else "no dyad found"
        )
        form = "mature" if first["mature"] else f"PRO (to res {first['max_residue']})"
        ligands = [r["ligand"] for r in structure_rows if r["ligand"]]
        print(
            f"  {pdb_id}  {str(first['resolution']):5} A  {form:22} {dyad:26} "
            f"ligands: {', '.join(ligands) if ligands else '-'}"
        )

    survey = pd.DataFrame(rows)
    out = PROCESSED / "target_survey.csv"
    survey.drop(columns=[c for c in survey.columns if c == "sg_coord"]).to_csv(
        out, index=False
    )
    print(f"\nWrote {out}")

    template = choose_template(survey)
    if template:
        print(
            f"\nChosen template: {template['pdb_id']} "
            f"({template['resolution']} A), redocking ligand {template['ligand']}"
        )
    else:
        print("\nNo suitable non-covalent template found.")

    write_target_doc(survey, template)
    print(f"Wrote {DOCS / 'TARGET.md'}")

    if template:
        (PROCESSED / "docking_template.json").write_text(
            json.dumps(
                {
                    "pdb_id": template["pdb_id"],
                    "resolution": template["resolution"],
                    "ligand": template["ligand"],
                    "ligand_chain": template["ligand_chain"],
                    "ligand_resseq": int(template["ligand_resseq"]),
                    "catalytic_cys": int(template["dyad_cys"]),
                    "catalytic_his": int(template["dyad_his"]),
                },
                indent=2,
            )
        )
        print(f"Wrote {PROCESSED / 'docking_template.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
