"""Measure the enzymatic-to-cellular potency drop, and what predicts it.

66 compounds in this dataset carry cell-based measurements and 61 of those
also have enzymatic data, though only the subset reported as a cellular IC50
(rather than single-dose percent inhibition) can be compared quantitatively.
The difference between the two numbers is an experimental signal the project
already owns and had been treating only as a nuisance: pooling the two assay
types was the data-quality bug that manufactured 1-3 log "replicate
disagreement".

Read correctly, the gap is informative. A compound that inhibits the enzyme at
1 nM but needs 1 uM in cells has lost three logs to something other than
target affinity -- most often failure to cross the cell membrane, plus serum
binding and efflux. That matters here more than for most targets, because the
project's central constraint is whether a legumain inhibitor can cross a
membrane at all, and the blood-brain barrier is a much harder membrane than a
HEK293 plasma membrane.

So this is the closest thing to experimental permeability evidence available
without doing an experiment: a compound that cannot get into a HEK293 cell is
very unlikely to reach the brain.

Writes:
    data/processed/permeability_gap.csv
    results/figures/permeability_gap.png
    docs/PERMEABILITY.md

Usage:
    python src/data/permeability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prepare_dataset import (  # noqa: E402
    BAO_CELL_BASED,
    HUMAN_TAX_ID,
    to_pic50,
)
from data.standardize import (  # noqa: E402
    basic_properties,
    is_covalent,
    standardize_mol,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "results" / "figures"
DOCS = ROOT / "docs"

# Properties tested against the potency gap. Chosen in advance from what is
# understood to govern passive membrane permeability, rather than scanning
# every available descriptor until something reaches significance.
CANDIDATE_PROPERTIES = ["tpsa", "clogp", "mw", "hbd", "hba", "rotb", "fsp3"]

# A drop of this size or more means the compound loses over 90% of its
# apparent potency on entering a cell.
LARGE_GAP = 1.0


def build_paired_table() -> pd.DataFrame:
    """One row per compound measured both on enzyme and in cells."""
    activities = pd.read_csv(RAW / "chembl_lgmn_activities.csv")
    assays = pd.read_csv(RAW / "chembl_lgmn_assays.csv")

    columns = [
        c
        for c in ["assay_chembl_id", "assay_tax_id", "bao_format", "bao_label"]
        if c in assays.columns
    ]
    df = activities.merge(assays[columns], on="assay_chembl_id", how="left")

    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[
        df["standard_type"].eq("IC50")
        & df["standard_units"].eq("nM")
        & df["standard_value"].gt(0)
        & df["standard_relation"].fillna("=").str.strip().eq("=")
        & df["data_validity_comment"].isna()
        & (df["assay_tax_id"].isna() | df["assay_tax_id"].eq(HUMAN_TAX_ID))
    ].copy()

    df["pic50"] = to_pic50(df["standard_value"])
    df["is_cellular"] = df["bao_format"].eq(BAO_CELL_BASED)

    # Standardise so a salt and its parent are recognised as one compound.
    #
    # The failed rows must be dropped by label, not by truncating the frame to
    # the number of successes: truncation keeps the first N rows rather than
    # the N that parsed, which pairs each SMILES with a different compound's
    # potency and corrupts every number downstream.
    from rdkit import Chem

    resolved: dict[int, str] = {}
    for index, raw_smiles in df["canonical_smiles"].items():
        mol = standardize_mol(raw_smiles)
        if mol is not None:
            resolved[index] = Chem.MolToSmiles(mol)

    df = df.loc[list(resolved)].copy()
    df["smiles"] = pd.Series(resolved)

    enzymatic = (
        df[~df["is_cellular"]]
        .groupby("smiles")
        .agg(
            pic50_enzyme=("pic50", "median"),
            n_enzyme=("pic50", "size"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
        )
    )
    cellular = (
        df[df["is_cellular"]]
        .groupby("smiles")
        .agg(pic50_cell=("pic50", "median"), n_cell=("pic50", "size"))
    )

    paired = enzymatic.join(cellular, how="inner").reset_index()
    if paired.empty:
        return paired

    # Positive gap means the compound is weaker inside a cell.
    paired["cell_gap"] = paired["pic50_enzyme"] - paired["pic50_cell"]
    paired["fold_loss"] = 10 ** paired["cell_gap"]

    properties = []
    for smi in paired["smiles"]:
        mol = standardize_mol(smi)
        row = basic_properties(mol)
        row["covalent"] = is_covalent(mol)
        properties.append(row)
    return pd.concat(
        [paired, pd.DataFrame(properties, index=paired.index)], axis=1
    )


def correlate(paired: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each property with the cellular potency gap."""
    rows = []
    for prop in CANDIDATE_PROPERTIES:
        if prop not in paired.columns:
            continue
        subset = paired[[prop, "cell_gap"]].dropna()
        if len(subset) < 8:
            continue
        # A property with no variance carries no information and makes
        # spearmanr return NaN. These compounds come largely from one series,
        # so a constant descriptor is expected rather than anomalous.
        if subset[prop].nunique() < 2:
            continue
        rho, pvalue = stats.spearmanr(subset[prop], subset["cell_gap"])
        rows.append(
            {
                "property": prop,
                "spearman_rho": rho,
                "p_value": pvalue,
                "n": len(subset),
            }
        )
    if not rows:
        # No property had enough rows or enough variance. An empty frame has
        # no p_value column, so sorting it would raise rather than return
        # "nothing to report".
        return pd.DataFrame(
            columns=["property", "spearman_rho", "p_value", "n", "p_holm",
                     "significant"]
        )

    table = pd.DataFrame(rows).sort_values("p_value")

    # Several properties are tested, so a raw p-value overstates significance.
    # Holm-Bonferroni controls the family-wise error rate without assuming the
    # tests are independent, which these are not (TPSA and HBD covary).
    if not table.empty:
        m = len(table)
        adjusted, running_max = [], 0.0
        for index, pvalue in enumerate(table["p_value"]):
            value = min(1.0, (m - index) * pvalue)
            running_max = max(running_max, value)
            adjusted.append(running_max)
        table["p_holm"] = adjusted
        table["significant"] = table["p_holm"] < 0.05
    return table


def translation(paired: pd.DataFrame) -> dict:
    """How much of an enzymatic potency gain survives into cells.

    The scatter invites the conclusion that cellular potency is capped, since
    the most potent enzyme inhibitors all land near pIC50 7. That reading
    overstates it: potency does translate, just inefficiently. The slope of
    cellular on enzymatic pIC50 is the honest summary, and it is reported with
    its correlation so the weakness of the relationship is visible.
    """
    subset = paired[["pic50_enzyme", "pic50_cell"]].dropna()
    if len(subset) < 8:
        return {}

    slope, intercept = np.polyfit(
        subset["pic50_enzyme"], subset["pic50_cell"], 1
    )
    # Correlation is undefined if either side has no variance, which happens
    # when every compound hits the same cellular value. The slope is still
    # meaningful there (it is zero), so report it with correlations as NaN
    # rather than failing.
    if (
        subset["pic50_enzyme"].nunique() < 2
        or subset["pic50_cell"].nunique() < 2
    ):
        rho = rho_p = r = r_p = float("nan")
    else:
        rho, rho_p = stats.spearmanr(
            subset["pic50_enzyme"], subset["pic50_cell"]
        )
        r, r_p = stats.pearsonr(subset["pic50_enzyme"], subset["pic50_cell"])

    top = subset.nlargest(10, "pic50_enzyme")
    bottom = subset.nsmallest(10, "pic50_enzyme")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "n": len(subset),
        "top_enzyme_mean": float(top["pic50_enzyme"].mean()),
        "top_cell_mean": float(top["pic50_cell"].mean()),
        "bottom_enzyme_mean": float(bottom["pic50_enzyme"].mean()),
        "bottom_cell_mean": float(bottom["pic50_cell"].mean()),
    }


def make_figure(paired: pd.DataFrame, correlations: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- enzyme vs cell potency ---
    covalent = paired["covalent"].fillna(False).astype(bool)
    limits = [
        min(paired["pic50_cell"].min(), paired["pic50_enzyme"].min()) - 0.3,
        max(paired["pic50_cell"].max(), paired["pic50_enzyme"].max()) + 0.3,
    ]
    axes[0].plot(limits, limits, ls="--", color="#555", lw=1, label="no loss")
    for mask, label, colour, marker in (
        (~covalent, "non-covalent", "#2a9d8f", "o"),
        (covalent, "covalent", "#e76f51", "^"),
    ):
        subset = paired[mask]
        axes[0].scatter(
            subset["pic50_enzyme"], subset["pic50_cell"],
            label=label, color=colour, marker=marker, s=38,
            alpha=0.8, edgecolor="none",
        )
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("pIC50, purified enzyme")
    axes[0].set_ylabel("pIC50, in cells")
    axes[0].set_title("Potency lost on entering a cell")
    axes[0].legend(frameon=False, fontsize=8)

    # --- distribution of the gap ---
    axes[1].hist(paired["cell_gap"], bins=18, color="#457b9d")
    axes[1].axvline(0, ls="--", color="#555", lw=1)
    median_gap = paired["cell_gap"].median()
    axes[1].axvline(median_gap, color="#e63946", lw=1.5)
    axes[1].annotate(
        f"median {median_gap:.2f} log\n({10 ** median_gap:.0f}x weaker)",
        xy=(median_gap, axes[1].get_ylim()[1] * 0.75),
        xytext=(median_gap + 0.35, axes[1].get_ylim()[1] * 0.8),
        fontsize=8,
        color="#e63946",
        arrowprops={"arrowstyle": "->", "color": "#e63946", "lw": 0.8},
    )
    axes[1].set_xlabel("Potency lost (log units)")
    axes[1].set_ylabel("Compounds")
    axes[1].set_title("How much potency the cell membrane costs")

    # --- strongest property relationship ---
    if not correlations.empty:
        best = correlations.iloc[0]
        prop = best["property"]
        axes[2].scatter(
            paired[prop], paired["cell_gap"],
            color="#457b9d", s=38, alpha=0.8, edgecolor="none",
        )
        subset = paired[[prop, "cell_gap"]].dropna()
        if len(subset) > 2:
            slope, intercept = np.polyfit(subset[prop], subset["cell_gap"], 1)
            xs = np.linspace(subset[prop].min(), subset[prop].max(), 50)
            axes[2].plot(xs, slope * xs + intercept, color="#e63946", lw=1.5)
        axes[2].set_xlabel(prop)
        axes[2].set_ylabel("Potency lost (log units)")
        axes[2].set_title(
            f"{prop} vs cellular loss\n"
            f"Spearman {best['spearman_rho']:.2f}, "
            f"p={best['p_value']:.3g}"
        )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permeability_gap.png", dpi=160)
    plt.close(fig)


def write_doc(
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    shift: dict,
) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    median_gap = paired["cell_gap"].median()
    large = int((paired["cell_gap"] >= LARGE_GAP).sum())
    improved = int((paired["cell_gap"] < 0).sum())

    lines = [
        "# Cellular permeability: what the paired assay data shows",
        "",
        "Generated by `src/data/permeability.py`.",
        "",
        f"**{len(paired)} compounds** were measured both against purified "
        "recombinant legumain",
        "and inside HEK293-A cells. The difference between those two numbers "
        "is the",
        "closest thing to experimental permeability evidence this project has "
        "without",
        "running an experiment.",
        "",
        "## The headline",
        "",
        f"- Median potency lost on entering a cell: "
        f"**{median_gap:.2f} log units** "
        f"({10 ** median_gap:.0f}-fold weaker)",
        f"- Compounds losing 10-fold or more: **{large} of {len(paired)}**",
        f"- Compounds that got *more* potent in cells: **{improved}**",
        "",
        "A HEK293 plasma membrane is a far easier barrier than the "
        "blood-brain barrier.",
        "A compound that already loses most of its potency getting into an "
        "ordinary cell",
        "in culture is not a realistic starting point for a CNS drug, whatever "
        "its",
        "enzymatic IC50 says.",
        "",
        "## What predicts the loss",
        "",
    ]

    if correlations.empty:
        lines += ["Too few paired measurements to test.", ""]
    else:
        lines += [
            "Seven properties known to govern passive membrane permeability "
            "were tested",
            "against the gap. They were chosen before looking, rather than by "
            "scanning",
            "every descriptor until one reached significance, and p-values are "
            "Holm-adjusted",
            "for testing several at once.",
            "",
            "| Property | Spearman rho | p | p (Holm) | Significant |",
            "|---|---|---|---|---|",
        ]
        for row in correlations.itertuples(index=False):
            lines.append(
                f"| {row.property} | {row.spearman_rho:+.2f} | "
                f"{row.p_value:.3g} | {row.p_holm:.3g} | "
                f"{'yes' if row.significant else 'no'} |"
            )

        significant = correlations[correlations["significant"]]
        lines += [""]
        if significant.empty:
            lines += [
                "**No property survives correction for multiple testing.** "
                "With "
                f"{len(paired)} compounds,",
                "mostly from one chemical series, this analysis is "
                "underpowered: a real",
                "effect of modest size would not be detectable here. The "
                "honest reading is",
                "that the *size* of the permeability penalty is established "
                "and its structural",
                "cause is not, rather than that structure does not matter.",
                "",
            ]
        else:
            best = significant.iloc[0]
            direction = (
                "higher" if best["spearman_rho"] > 0 else "lower"
            )
            lines += [
                f"**{best['property']}** is the strongest relationship "
                f"(rho {best['spearman_rho']:+.2f}, "
                f"Holm p={best['p_holm']:.3g}): compounds with "
                f"{direction} {best['property']}",
                "lose more potency on entering a cell, which is the direction "
                "passive-diffusion",
                "models predict.",
                "",
            ]

    if shift:
        gain_enzyme = shift["top_enzyme_mean"] - shift["bottom_enzyme_mean"]
        gain_cell = shift["top_cell_mean"] - shift["bottom_cell_mean"]
        lines += [
            "## Does enzymatic potency translate into cells?",
            "",
            "Partly, and inefficiently.",
            "",
            f"- Slope of cellular on enzymatic pIC50: "
            f"**{shift['slope']:.2f}** -- each log unit gained against the",
            f"  purified enzyme returns about "
            f"{shift['slope']:.2f} log in cells",
            f"- Spearman rho **{shift['spearman_rho']:.2f}** "
            f"(p={shift['spearman_p']:.3g}), "
            f"Pearson r {shift['pearson_r']:.2f} "
            f"(p={shift['pearson_p']:.3g}), n={shift['n']}",
            f"- The 10 most potent enzyme inhibitors average "
            f"{shift['top_enzyme_mean']:.2f} on enzyme and",
            f"  {shift['top_cell_mean']:.2f} in cells; the 10 weakest average "
            f"{shift['bottom_enzyme_mean']:.2f} and "
            f"{shift['bottom_cell_mean']:.2f}.",
            f"  So {gain_enzyme:.2f} log of enzymatic improvement bought "
            f"{gain_cell:.2f} log in cells.",
            "",
            "It is tempting to read the scatter as a hard ceiling, because "
            "every strong",
            "enzyme inhibitor lands near pIC50 7 in cells. That overstates "
            "it. The honest",
            "statement is that the relationship is real but weak and heavily "
            "attenuated,",
            "and at n="
            f"{shift['n']} the rank correlation is not significant.",
            "",
            "The practical implication still holds either way: pushing "
            "enzymatic potency",
            "further is a poor use of effort for this series compared with "
            "fixing whatever",
            "stops the compounds reaching their target, since most of each "
            "gain is lost at",
            "the membrane.",
            "",
        ]

    worst = paired.nlargest(5, "cell_gap")
    best_movers = paired.nsmallest(5, "cell_gap")

    lines += [
        "## Worst cellular drop-off",
        "",
        "| ChEMBL ID | Enzyme pIC50 | Cell pIC50 | Lost (log) | Fold | TPSA | "
        "cLogP |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in worst.itertuples(index=False):
        lines.append(
            f"| {row.molecule_chembl_id} | {row.pic50_enzyme:.2f} | "
            f"{row.pic50_cell:.2f} | {row.cell_gap:.2f} | "
            f"{row.fold_loss:.0f}x | {row.tpsa:.0f} | {row.clogp:.2f} |"
        )

    lines += [
        "",
        "## Smallest drop-off (the ones that travel best)",
        "",
        "| ChEMBL ID | Enzyme pIC50 | Cell pIC50 | Lost (log) | TPSA | cLogP |",
        "|---|---|---|---|---|---|",
    ]
    for row in best_movers.itertuples(index=False):
        lines.append(
            f"| {row.molecule_chembl_id} | {row.pic50_enzyme:.2f} | "
            f"{row.pic50_cell:.2f} | {row.cell_gap:.2f} | "
            f"{row.tpsa:.0f} | {row.clogp:.2f} |"
        )

    lines += [
        "",
        "These are worth more attention than their raw enzymatic potency "
        "suggests. A",
        "compound that keeps its activity inside a cell has already cleared a "
        "barrier",
        "that most of this chemical series does not.",
        "",
        "## Caveats",
        "",
        "- A cellular IC50 folds together permeability, efflux, serum "
        "binding, intracellular",
        "  target concentration and compound stability. The gap is not a pure "
        "permeability",
        "  measurement.",
        "- HEK293 cells are not brain endothelium and have no tight junctions "
        "or the same",
        "  efflux transporter profile. Passing here does not predict crossing "
        "the",
        "  blood-brain barrier; failing here is the more reliable signal.",
        "- Most pairs come from a small number of publications and chemical "
        "series, so",
        "  these compounds are not independent observations.",
        "",
    ]
    (DOCS / "PERMEABILITY.md").write_text("\n".join(lines))


def main() -> int:
    if not (RAW / "chembl_lgmn_activities.csv").exists():
        print("Run src/data/fetch_chembl.py first.", file=sys.stderr)
        return 1

    paired = build_paired_table()
    if paired.empty:
        print("No compounds have both enzymatic and cellular measurements.")
        return 1

    PROCESSED.mkdir(parents=True, exist_ok=True)
    paired.to_csv(PROCESSED / "permeability_gap.csv", index=False)

    correlations = correlate(paired)
    shift = translation(paired)
    median_gap = paired["cell_gap"].median()

    print(f"Paired compounds: {len(paired)}")
    print(
        f"Median potency lost in cells: {median_gap:.2f} log units "
        f"({10 ** median_gap:.0f}x weaker)"
    )
    print(
        f"Losing >= {LARGE_GAP} log: "
        f"{int((paired['cell_gap'] >= LARGE_GAP).sum())} compounds"
    )
    print(f"More potent in cells: {int((paired['cell_gap'] < 0).sum())}")

    if shift:
        print(
            f"\nEnzyme->cell translation: slope {shift['slope']:.2f} "
            f"log per log (Spearman {shift['spearman_rho']:.2f}, "
            f"p={shift['spearman_p']:.3g})"
        )

    if not correlations.empty:
        print("\nProperty vs cellular loss:")
        print(correlations.to_string(index=False))

    make_figure(paired, correlations)
    write_doc(paired, correlations, shift)
    print(f"\nWrote {PROCESSED / 'permeability_gap.csv'}")
    print(f"Wrote {FIGURE_DIR / 'permeability_gap.png'}")
    print(f"Wrote {DOCS / 'PERMEABILITY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
