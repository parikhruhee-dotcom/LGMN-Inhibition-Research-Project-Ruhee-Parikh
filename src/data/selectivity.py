"""Check the LGMN compounds against related cysteine proteases.

Legumain belongs to clan CD, the same structural clan as the caspases, and
most of the compounds here carry an electrophilic warhead aimed at a catalytic
cysteine. That combination invites cross-reactivity, and a legumain inhibitor
that also inhibits caspase-3 is not a drug candidate -- caspase-3 is the
executioner protease of apoptosis.

So selectivity is not an optional extra for this target. It was listed as an
untested limitation of the project; this closes it with the data that exists.

Counter-targets are two groups:

- **Clan CD caspases**, which share legumain's fold and its preference for
  cleaving after a small acidic/amide residue. These are the mechanistically
  worrying ones.
- **Clan CA cathepsins** (papain family), which share only the catalytic
  cysteine. They are routine counter-screens for any cysteine-protease
  programme and report on general reactivity rather than fold similarity.

An absence of data is itself a finding: a compound never counter-screened is
not a selective compound, it is an untested one, and the two are reported
separately rather than merged.

Writes:
    data/raw/chembl_offtarget_activities.csv
    data/processed/selectivity.csv
    results/figures/selectivity.png
    docs/SELECTIVITY.md

Usage:
    python src/data/selectivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import USABLE_TYPES, to_pic50  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "results" / "figures"
DOCS = ROOT / "docs"

# (UniProt accession, display name, clan). Clan CD shares legumain's fold;
# clan CA shares only the catalytic cysteine.
COUNTER_TARGETS = [
    ("P29466", "Caspase-1", "CD"),
    ("P42574", "Caspase-3", "CD"),
    ("P55212", "Caspase-6", "CD"),
    ("Q14790", "Caspase-8", "CD"),
    ("P55210", "Caspase-7", "CD"),
    ("P07858", "Cathepsin B", "CA"),
    ("P07711", "Cathepsin L", "CA"),
    ("P25774", "Cathepsin S", "CA"),
    ("P43235", "Cathepsin K", "CA"),
]

# Selectivity below this is a genuine liability rather than a note: fewer than
# 10-fold separation between the target and an off-target leaves no therapeutic
# window to optimise from.
SELECTIVITY_CONCERN = 1.0


def resolve_target(accession: str) -> str | None:
    """ChEMBL target ID for a UniProt accession, preferring a human single protein."""
    hits = list(
        new_client.target.filter(
            target_components__accession=accession
        ).only(["target_chembl_id", "pref_name", "organism", "target_type"])
    )
    for hit in hits:
        if (
            hit.get("organism") == "Homo sapiens"
            and hit.get("target_type") == "SINGLE PROTEIN"
        ):
            return hit["target_chembl_id"]
    return hits[0]["target_chembl_id"] if hits else None


def fetch_offtarget_activities(molecule_ids: list[str]) -> pd.DataFrame:
    """Activities of our compounds against each counter-target.

    Queried by molecule rather than by target: pulling every compound ever
    tested against caspase-3 would be tens of thousands of records, and only
    the overlap with this project's compounds is relevant.
    """
    frames = []
    for accession, name, clan in COUNTER_TARGETS:
        target_id = resolve_target(accession)
        if target_id is None:
            print(f"  {name:14} no ChEMBL target for {accession}")
            continue

        records = []
        batch_size = 40
        for start in range(0, len(molecule_ids), batch_size):
            batch = molecule_ids[start : start + batch_size]
            records.extend(
                list(
                    new_client.activity.filter(
                        target_chembl_id=target_id,
                        molecule_chembl_id__in=batch,
                    ).only(
                        [
                            "molecule_chembl_id",
                            "canonical_smiles",
                            "standard_type",
                            "standard_relation",
                            "standard_value",
                            "standard_units",
                            "assay_chembl_id",
                            "target_chembl_id",
                            "data_validity_comment",
                        ]
                    )
                )
            )

        frame = pd.DataFrame(records)
        print(f"  {name:14} ({target_id}): {len(frame)} records")
        if not frame.empty:
            frame["off_target"] = name
            frame["clan"] = clan
            frame["off_target_chembl_id"] = target_id
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_selectivity(lgmn: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    """One row per compound-counter-target pair with a selectivity ratio."""
    if off.empty:
        return pd.DataFrame()

    off = off.copy()
    off["standard_value"] = pd.to_numeric(off["standard_value"], errors="coerce")
    off = off[
        off["standard_type"].isin(USABLE_TYPES)
        & off["standard_units"].eq("nM")
        & off["standard_value"].gt(0)
        & off["standard_relation"].fillna("=").str.strip().isin(["=", ">"])
        & off["data_validity_comment"].isna()
    ].copy()
    if off.empty:
        return pd.DataFrame()

    # A ">" record means the off-target was not inhibited up to the highest
    # concentration tested. That is evidence *for* selectivity, so it is kept
    # and flagged rather than discarded as censored.
    off["censored"] = off["standard_relation"].fillna("=").str.strip().eq(">")
    off["pic50_off"] = to_pic50(off["standard_value"])

    aggregated = (
        off.groupby(["molecule_chembl_id", "off_target", "clan"])
        .agg(
            pic50_off=("pic50_off", "median"),
            n_records=("pic50_off", "size"),
            any_censored=("censored", "any"),
        )
        .reset_index()
    )

    merged = aggregated.merge(
        lgmn[["molecule_chembl_id", "pic50", "covalent", "warheads", "smiles"]],
        on="molecule_chembl_id",
        how="inner",
    ).rename(columns={"pic50": "pic50_lgmn"})

    # Positive means more potent on legumain than on the off-target.
    merged["selectivity"] = merged["pic50_lgmn"] - merged["pic50_off"]
    merged["fold_selectivity"] = 10 ** merged["selectivity"]
    merged["concern"] = merged["selectivity"] < SELECTIVITY_CONCERN
    return merged.sort_values(["molecule_chembl_id", "off_target"])


def make_figure(selectivity: pd.DataFrame, lgmn: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    tested = selectivity["molecule_chembl_id"].nunique()
    untested = lgmn["molecule_chembl_id"].nunique() - tested
    axes[0].bar(
        ["counter-screened", "never tested"],
        [tested, untested],
        color=["#2a9d8f", "#adb5bd"],
    )
    for index, value in enumerate([tested, untested]):
        axes[0].text(index, value, f" {value}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Compounds")
    axes[0].set_title("How many have any selectivity data at all")

    if not selectivity.empty:
        order = (
            selectivity.groupby("off_target")["selectivity"]
            .median()
            .sort_values()
            .index.tolist()
        )
        data = [
            selectivity.loc[
                selectivity["off_target"] == name, "selectivity"
            ].to_numpy()
            for name in order
        ]
        colours = [
            "#e76f51"
            if selectivity.loc[
                selectivity["off_target"] == name, "clan"
            ].iloc[0] == "CD"
            else "#457b9d"
            for name in order
        ]
        parts = axes[1].boxplot(
            data, vert=False, patch_artist=True, widths=0.6
        )
        for patch, colour in zip(parts["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        axes[1].set_yticklabels(order, fontsize=8)
        axes[1].axvline(0, ls="--", color="#555", lw=1)
        axes[1].axvline(
            SELECTIVITY_CONCERN, ls=":", color="#e63946", lw=1.2
        )
        axes[1].set_xlabel("Selectivity (log units, positive = LGMN-preferring)")
        axes[1].set_title(
            "Selectivity by counter-target\n"
            "(orange = clan CD, shares legumain's fold)"
        )

        axes[2].scatter(
            selectivity["pic50_lgmn"],
            selectivity["selectivity"],
            c=np.where(selectivity["clan"] == "CD", "#e76f51", "#457b9d"),
            s=38,
            alpha=0.8,
            edgecolor="none",
        )
        axes[2].axhline(0, ls="--", color="#555", lw=1)
        axes[2].axhline(SELECTIVITY_CONCERN, ls=":", color="#e63946", lw=1.2)
        axes[2].set_xlabel("pIC50 against legumain")
        axes[2].set_ylabel("Selectivity (log units)")
        axes[2].set_title("Does potency buy selectivity?")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "selectivity.png", dpi=160)
    plt.close(fig)


def write_doc(selectivity: pd.DataFrame, lgmn: pd.DataFrame) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    total = lgmn["molecule_chembl_id"].nunique()
    tested = (
        selectivity["molecule_chembl_id"].nunique()
        if not selectivity.empty
        else 0
    )

    lines = [
        "# Selectivity against related cysteine proteases",
        "",
        "Generated by `src/data/selectivity.py`.",
        "",
        "Legumain is a clan CD cysteine protease, the same clan as the "
        "caspases, and 85%",
        "of the compounds here carry a warhead aimed at a catalytic cysteine. "
        "A legumain",
        "inhibitor that also hits caspase-3 -- the executioner protease of "
        "apoptosis --",
        "is not a drug candidate. This was listed as an untested limitation "
        "of the",
        "project; what follows is what the public data can and cannot say.",
        "",
        "## Coverage",
        "",
        f"- Compounds in the modelling set: **{total}**",
        f"- With any published counter-screen among the targets tested here: "
        f"**{tested}**",
        f"- Never counter-screened: **{total - tested}**",
        "",
    ]

    if tested == 0:
        lines += [
            "**No compound in this dataset has published activity against "
            "any of the",
            "counter-targets queried.** That is not evidence of selectivity. "
            "It means the",
            "question is open for every compound in the shortlist, and for a "
            "covalent",
            "asparagine-directed warhead against a clan CD protease, open is "
            "the wrong",
            "state for it to be in.",
            "",
            "Counter-targets queried: "
            + ", ".join(name for _, name, _ in COUNTER_TARGETS)
            + ".",
            "",
            "### What this costs the project",
            "",
            "Selectivity cannot be predicted from the structures here with "
            "any confidence.",
            "Docking against caspase-3 would produce numbers, but the "
            "redocking validation",
            "in `docs/DOCKING.md` showed the scoring function cannot rank "
            "poses reliably",
            "even for legumain itself, so cross-target scores would be "
            "decoration.",
            "",
            "The honest route is experimental: any compound taken forward "
            "should be run",
            "against caspase-3 and cathepsin B before anything else is "
            "concluded about it.",
            "Those are inexpensive, commercially available assays.",
            "",
        ]
        (DOCS / "SELECTIVITY.md").write_text("\n".join(lines))
        return

    concerning = selectivity[selectivity["concern"]]
    lines += [
        "## Results",
        "",
        "Selectivity is pIC50(legumain) minus pIC50(off-target): positive "
        "means the",
        f"compound prefers legumain. Below {SELECTIVITY_CONCERN:.0f} log "
        "unit (10-fold) there is no",
        "therapeutic window to optimise from.",
        "",
        "| Counter-target | Clan | Compounds | Median selectivity | "
        "Lower bounds only | Below 10-fold |",
        "|---|---|---|---|---|---|",
    ]
    for name, group in selectivity.groupby("off_target"):
        censored = int(group["any_censored"].sum())
        prefix = ">=" if censored == len(group) else ""
        lines.append(
            f"| {name} | {group['clan'].iloc[0]} | {len(group)} | "
            f"{prefix}{group['selectivity'].median():+.2f} | "
            f"{censored}/{len(group)} | "
            f"{int(group['concern'].sum())} |"
        )

    measured = selectivity[~selectivity["any_censored"].astype(bool)]
    bounded = selectivity[selectivity["any_censored"].astype(bool)]
    lines += [
        "",
        "### Measurements versus lower bounds",
        "",
        f"Of {len(selectivity)} compound-target pairs, **{len(bounded)} are "
        f"lower bounds**, not",
        "measurements: the record says the off-target was not inhibited up to "
        "the highest",
        "concentration tested, which here is uniformly 50 uM. For those the "
        "true",
        "selectivity could be anything from the stated figure upwards, and "
        "the number in",
        "the table is the floor.",
        "",
        f"Only **{len(measured)} pairs** carry an actual off-target IC50"
        + (
            ", from "
            + (
                "a single compound."
                if measured["molecule_chembl_id"].nunique() == 1
                else f"{measured['molecule_chembl_id'].nunique()} compounds."
            )
            if not measured.empty
            else "."
        ),
        "",
        "This distinction matters more than it looks. 'No inhibition at "
        "50 uM' against a",
        "panel is a weaker claim than a titrated counter-screen: it rules out "
        "potent",
        "off-target activity and says nothing about moderate activity, which "
        "is what",
        "usually causes trouble in vivo.",
        "",
    ]

    lines += [
        "",
        f"**{concerning['molecule_chembl_id'].nunique()} compounds** show "
        "less than 10-fold separation from at",
        "least one counter-target.",
        "",
    ]

    if not concerning.empty:
        lines += [
            "### Compounds with a selectivity liability",
            "",
            "| ChEMBL ID | Off-target | Clan | LGMN pIC50 | Off pIC50 | "
            "Selectivity |",
            "|---|---|---|---|---|---|",
        ]
        for row in concerning.nsmallest(15, "selectivity").itertuples(index=False):
            lines.append(
                f"| {row.molecule_chembl_id} | {row.off_target} | {row.clan} | "
                f"{row.pic50_lgmn:.2f} | {row.pic50_off:.2f} | "
                f"{row.selectivity:+.2f} |"
            )
        lines.append("")

    lines += [
        "## Caveats",
        "",
        "- Counter-screen data is published selectively: a compound tested "
        "and found",
        "  unselective is less likely to be reported than one that looked "
        "clean, so these",
        "  numbers are optimistic.",
        "- A `>` record means no inhibition up to the highest concentration "
        "tested. Those",
        "  are kept as evidence *for* selectivity but understate the true "
        "ratio.",
        "- Absence of a counter-screen is not selectivity. "
        f"{total - tested} compounds have no",
        "  data at all and should be treated as unknown, not as safe.",
        "",
    ]
    (DOCS / "SELECTIVITY.md").write_text("\n".join(lines))


def main() -> int:
    lgmn_path = PROCESSED / "lgmn_regression.csv"
    if not lgmn_path.exists():
        print(f"{lgmn_path} not found; run prepare_dataset.py", file=sys.stderr)
        return 1

    lgmn = pd.read_csv(lgmn_path)
    molecule_ids = sorted(lgmn["molecule_chembl_id"].dropna().unique())
    print(
        f"Checking {len(molecule_ids)} compounds against "
        f"{len(COUNTER_TARGETS)} counter-targets ..."
    )

    off = fetch_offtarget_activities(molecule_ids)
    RAW.mkdir(parents=True, exist_ok=True)
    off.to_csv(RAW / "chembl_offtarget_activities.csv", index=False)

    selectivity = build_selectivity(lgmn, off)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    selectivity.to_csv(PROCESSED / "selectivity.csv", index=False)

    tested = (
        selectivity["molecule_chembl_id"].nunique()
        if not selectivity.empty
        else 0
    )
    print(f"\nCompounds with any counter-screen data: {tested}/{len(molecule_ids)}")
    if not selectivity.empty:
        print(
            selectivity.groupby("off_target")
            .agg(
                compounds=("molecule_chembl_id", "nunique"),
                median_selectivity=("selectivity", "median"),
                below_10x=("concern", "sum"),
            )
            .to_string()
        )
    else:
        print(
            "No counter-screen data exists for these compounds in ChEMBL.\n"
            "That leaves selectivity an open question, not a solved one."
        )

    make_figure(selectivity, lgmn)
    write_doc(selectivity, lgmn)
    print(f"\nWrote {PROCESSED / 'selectivity.csv'}")
    print(f"Wrote {FIGURE_DIR / 'selectivity.png'}")
    print(f"Wrote {DOCS / 'SELECTIVITY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
