"""Apply ADMET and CNS filters to a compound set and produce a ranked shortlist.

Ranking combines what the project actually knows about each compound, and
keeps the sources separate rather than fusing them into one opaque score:

- measured or predicted potency (pIC50)
- whether the compound is predicted to reach the brain (CNS criteria)
- whether its docking pose engages the catalytic site, where available
- whether it carries a covalent warhead, reported rather than used to rank

Compounds are not silently dropped. Every exclusion is recorded with its
reason, because for a cysteine protease the usual filters remove exactly the
chemistry that works: an electrophilic warhead is both a PAINS-style alert and
the mechanism of most known legumain inhibitors.

Writes:
    results/shortlist.csv       every compound with every filter outcome
    results/figures/admet_profile.png
    docs/SHORTLIST.md           the ranked table with reasoning

Usage:
    python src/admet/screen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admet.filters import MIN_BBB_RULES, profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURE_DIR = RESULTS / "figures"
DOCS = ROOT / "docs"

SHORTLIST_SIZE = 25

# Affinity better than this is treated as a meaningful docking result. Vina-
# family scores cluster between -4 and -11 kcal/mol; -7 is a conventional
# cut for "worth looking at" without pretending the number is a free energy.
DOCKING_AFFINITY_CUT = -7.0


def load_inputs() -> pd.DataFrame:
    """Join the activity data with docking results where both exist."""
    activity_path = PROCESSED / "lgmn_regression.csv"
    if not activity_path.exists():
        raise FileNotFoundError(
            f"{activity_path} not found; run src/data/prepare_dataset.py"
        )
    compounds = pd.read_csv(activity_path)

    docking_path = RESULTS / "docking" / "docking_scores.csv"
    if docking_path.exists():
        docking = pd.read_csv(docking_path)
        keep = [
            c
            for c in (
                "molecule_chembl_id", "affinity", "dist_to_thiol",
                "in_active_site",
            )
            if c in docking.columns
        ]
        compounds = compounds.merge(
            docking[keep], on="molecule_chembl_id", how="left"
        )
    else:
        compounds["affinity"] = np.nan
        compounds["dist_to_thiol"] = np.nan
        compounds["in_active_site"] = np.nan

    return compounds


def annotate(compounds: pd.DataFrame) -> pd.DataFrame:
    """Add the full ADMET profile for every compound."""
    profiles = []
    for smiles in compounds["smiles"]:
        result = profile(smiles)
        profiles.append(result if result is not None else {"smiles": smiles})
    admet = pd.DataFrame(profiles)

    # The activity table already carries mw/clogp/tpsa etc. from
    # standardize.basic_properties; drop the duplicates from the ADMET frame
    # so the join does not produce _x/_y column pairs.
    overlap = [
        c for c in admet.columns
        if c in compounds.columns and c != "smiles"
    ]
    admet = admet.drop(columns=overlap)
    return compounds.merge(admet, on="smiles", how="left")


def rank(compounds: pd.DataFrame) -> pd.DataFrame:
    """Score and sort, keeping the reason for every compound's position."""
    df = compounds.copy()

    # Normalise potency to 0-1 across the observed range so it combines with
    # the other terms without one dominating by unit scale alone.
    potency = df["pic50"]
    span = potency.max() - potency.min()
    df["potency_norm"] = (
        (potency - potency.min()) / span if span > 0 else 0.5
    )

    # Docking contributes only where a pose was obtained in the active site.
    affinity = df.get("affinity")
    if affinity is not None and affinity.notna().any():
        valid = affinity.notna() & df["in_active_site"].fillna(False).astype(bool)
        worst, best = affinity[valid].max(), affinity[valid].min()
        docking_norm = pd.Series(np.nan, index=df.index, dtype=float)
        if pd.notna(worst) and pd.notna(best) and worst != best:
            docking_norm[valid] = (worst - affinity[valid]) / (worst - best)
        df["docking_norm"] = docking_norm
    else:
        df["docking_norm"] = np.nan

    # CNS suitability on a 0-1 scale from the MPO score.
    df["cns_norm"] = (df["cns_mpo"] / 6.0).clip(0, 1)

    # Weights: potency and brain penetration matter most for a CNS target;
    # docking is a weak signal here and is weighted accordingly. Where a
    # docking score is missing the remaining weights are renormalised rather
    # than treating the absence as a zero.
    weights = {"potency_norm": 0.45, "cns_norm": 0.40, "docking_norm": 0.15}
    total = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for column, weight in weights.items():
        values = df[column]
        present = values.notna()
        total[present] += values[present] * weight
        weight_sum[present] += weight
    df["composite_score"] = np.where(weight_sum > 0, total / weight_sum, np.nan)

    df["cns_blocker"] = np.where(
        df["cns_candidate"].fillna(False).astype(bool),
        "",
        np.where(
            ~df["cns_mpo_pass"].fillna(False).astype(bool),
            f"CNS MPO {df['cns_mpo'].round(1).astype(str)} < 4",
            f"only {df['bbb_rules_passed']}/{df['bbb_rules_total']} "
            f"BBB rules (need {MIN_BBB_RULES})",
        ),
    )

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def make_figure(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- potency vs CNS suitability, the central trade-off ---
    covalent = df["covalent"].fillna(False).astype(bool)
    for mask, label, colour, marker in (
        (~covalent, "non-covalent", "#2a9d8f", "o"),
        (covalent, "covalent warhead", "#e76f51", "^"),
    ):
        subset = df[mask]
        axes[0].scatter(
            subset["cns_mpo"],
            subset["pic50"],
            label=label,
            color=colour,
            marker=marker,
            alpha=0.75,
            edgecolor="none",
            s=38,
        )
    axes[0].axvline(4.0, ls="--", color="#555", lw=1)
    axes[0].annotate(
        "CNS MPO >= 4",
        xy=(4.05, df["pic50"].min()),
        fontsize=8,
        color="#555",
    )
    axes[0].set_xlabel("CNS MPO score (0-6)")
    axes[0].set_ylabel("pIC50")
    axes[0].set_title("Potency vs predicted brain penetration")
    axes[0].legend(frameon=False, fontsize=8)

    # --- how many compounds survive each filter ---
    stages = [
        ("All compounds", len(df)),
        ("Lipinski <= 1 viol.", int((df["lipinski_violations"] <= 1).sum())),
        ("Veber pass", int(df["veber_pass"].fillna(False).sum())),
        ("CNS MPO >= 4", int(df["cns_mpo_pass"].fillna(False).sum())),
        ("CNS candidate", int(df["cns_candidate"].fillna(False).sum())),
        ("+ no PAINS", int(
            (df["cns_candidate"].fillna(False).astype(bool)
             & df["pains_alerts"].fillna(0).eq(0)).sum()
        )),
    ]
    labels = [name for name, _ in stages]
    counts = [count for _, count in stages]
    axes[1].barh(range(len(stages)), counts, color="#457b9d")
    axes[1].set_yticks(range(len(stages)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Compounds remaining")
    axes[1].set_title("Attrition through the filter cascade")
    for index, count in enumerate(counts):
        axes[1].text(count, index, f" {count}", va="center", fontsize=8)

    # --- property distributions against CNS guidance ---
    axes[2].scatter(
        df["clogp"],
        df["tpsa"],
        c=df["pic50"],
        cmap="viridis",
        s=38,
        edgecolor="none",
    )
    axes[2].axhspan(40, 90, color="#2a9d8f", alpha=0.12)
    axes[2].axvspan(1, 4, color="#2a9d8f", alpha=0.12)
    in_box = int(
        (df["clogp"].between(1, 4) & df["tpsa"].between(40, 90)).sum()
    )
    axes[2].set_xlabel("cLogP")
    axes[2].set_ylabel("TPSA (A$^2$)")
    axes[2].set_title("Property space (shaded = CNS optimum)")
    # The emptiness of the shaded region is the project's central result, so
    # it is stated as a number rather than left to the eye.
    axes[2].annotate(
        f"{in_box} of {len(df)} compounds\nin the CNS optimum",
        xy=(0.03, 0.06),
        xycoords="axes fraction",
        fontsize=8,
        color="#1d3557",
    )
    colourbar = fig.colorbar(axes[2].collections[0], ax=axes[2])
    colourbar.set_label("pIC50", fontsize=8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "admet_profile.png", dpi=160)
    plt.close(fig)


def write_doc(df: pd.DataFrame) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    top = df.head(SHORTLIST_SIZE)

    covalent_n = int(df["covalent"].fillna(False).sum())
    cns_n = int(df["cns_candidate"].fillna(False).sum())
    both = int(
        (
            df["cns_candidate"].fillna(False).astype(bool)
            & ~df["covalent"].fillna(False).astype(bool)
        ).sum()
    )

    lines = [
        "# LGMN inhibitor shortlist",
        "",
        "Generated by `src/admet/screen.py`. **These are computational "
        "predictions,**",
        "**not validated hits.** No compound here has been tested against "
        "legumain by",
        "this project; the potency values are measurements taken from ChEMBL, "
        "and the",
        "ADMET columns are predictions from 2D structure.",
        "",
        "## What the numbers say overall",
        "",
        f"- Compounds profiled: **{len(df)}**",
        f"- Carrying a covalent warhead: **{covalent_n}** "
        f"({100 * covalent_n / max(len(df), 1):.0f}%)",
        f"- Meeting the combined CNS criteria: **{cns_n}**",
        f"- Both CNS-suitable *and* non-covalent: **{both}**",
        "",
        "That last number is the one that matters for an Alzheimer's "
        "programme, and",
        "it is the tightest constraint in the project. Legumain inhibitor "
        "chemistry is",
        "overwhelmingly covalent and peptide-like, which is close to the "
        "opposite of",
        "the small, lipophilic, low-polarity profile that crosses the "
        "blood-brain",
        "barrier.",
        "",
        "## Ranking",
        "",
        "The composite score weights measured potency (0.45), predicted CNS "
        "exposure",
        "(0.40) and docking (0.15). Docking is weighted low deliberately -- "
        "see",
        "`docs/DOCKING.md` for the redocking validation that justifies "
        "distrusting it.",
        "Where a compound has no docking result the remaining weights are "
        "renormalised",
        "rather than scoring the absence as zero.",
        "",
        f"## Top {len(top)} compounds",
        "",
        "| # | ChEMBL ID | pIC50 | CNS MPO | BBB rules | Covalent | "
        "Dock (kcal/mol) | Score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(top.itertuples(index=False), start=1):
        affinity = getattr(row, "affinity", np.nan)
        lines.append(
            f"| {index} | {row.molecule_chembl_id} | {row.pic50:.2f} | "
            f"{row.cns_mpo:.2f} | {row.bbb_rules_passed}/{row.bbb_rules_total} | "
            f"{'yes' if row.covalent else 'no'} | "
            f"{f'{affinity:.2f}' if pd.notna(affinity) else '-'} | "
            f"{row.composite_score:.3f} |"
        )

    best_cns = df[
        df["cns_candidate"].fillna(False).astype(bool)
        & ~df["covalent"].fillna(False).astype(bool)
    ].head(5)

    lines += [
        "",
        "## Non-covalent compounds meeting the CNS criteria",
        "",
        "These are the most relevant starting points for a brain-penetrant "
        "programme,",
        "regardless of where they place on the composite score.",
        "",
    ]
    if best_cns.empty:
        lines.append(
            "**None.** No compound in the dataset is both non-covalent and "
            "CNS-suitable. "
            "On this evidence a brain-penetrant legumain inhibitor would have "
            "to come "
            "from new chemistry rather than from optimising what is already "
            "published."
        )
    else:
        lines += [
            "| ChEMBL ID | pIC50 | CNS MPO | cLogP | TPSA | MW | PAINS |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in best_cns.itertuples(index=False):
            lines.append(
                f"| {row.molecule_chembl_id} | {row.pic50:.2f} | "
                f"{row.cns_mpo:.2f} | {row.clogp:.2f} | {row.tpsa:.0f} | "
                f"{row.mw:.0f} | {int(row.pains_alerts)} |"
            )

    lines += [
        "",
        "## Caveats that change how this should be read",
        "",
        "- A PAINS or Brenk alert is reported, never used to drop a compound. "
        "For a",
        "  cysteine protease the reactive group flagged as assay interference "
        "is often",
        "  the inhibitory mechanism.",
        "- Covalent compounds' docking scores describe the pre-covalent "
        "complex only.",
        "- CNS MPO ranks drug-like molecules against each other and is not a "
        "test of",
        "  whether something is drug-like at all: sucrose scores 4.0. The "
        "combined",
        "  `cns_candidate` verdict exists for that reason.",
        "- The estimated pKa feeding MPO is a substructure approximation, not "
        "a trained",
        "  model, and is the weakest input to the CNS score.",
        "",
    ]
    (DOCS / "SHORTLIST.md").write_text("\n".join(lines))


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    compounds = load_inputs()
    print(f"Profiling {len(compounds)} compounds ...")

    annotated = annotate(compounds)
    ranked = rank(annotated)
    ranked.to_csv(RESULTS / "shortlist.csv", index=False)

    cns_n = int(ranked["cns_candidate"].fillna(False).sum())
    covalent_n = int(ranked["covalent"].fillna(False).sum())
    both = int(
        (
            ranked["cns_candidate"].fillna(False).astype(bool)
            & ~ranked["covalent"].fillna(False).astype(bool)
        ).sum()
    )

    print(f"\nCNS MPO >= 4:              {int(ranked['cns_mpo_pass'].sum())}")
    print(f"Combined CNS candidate:    {cns_n}")
    print(f"Covalent warhead:          {covalent_n}")
    print(f"Non-covalent AND CNS-ok:   {both}")
    print(f"PAINS alerts:              {int(ranked['pains_alerts'].gt(0).sum())}")

    print("\nTop 10 by composite score:")
    columns = [
        "molecule_chembl_id", "pic50", "cns_mpo", "bbb_rules_passed",
        "covalent", "composite_score",
    ]
    print(ranked.head(10)[columns].to_string(index=False))

    make_figure(ranked)
    write_doc(ranked)
    print(f"\nWrote {RESULTS / 'shortlist.csv'}")
    print(f"Wrote {FIGURE_DIR / 'admet_profile.png'}")
    print(f"Wrote {DOCS / 'SHORTLIST.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
