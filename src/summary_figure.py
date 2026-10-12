"""One figure that carries the whole project.

Every other figure answers a question within a stage. This one is for
presenting the project to someone who will look at a single image: what the
chemistry is, where the compounds are lost, whether the methods work, and
what is left at the end.

Four panels, each a finding rather than a description:

1. **The funnel.** 180 compounds down to 6, with the step that removes most
   of them labelled.
2. **The conflict.** Potency against predicted brain penetration, showing
   that the two requirements pull in opposite directions.
3. **The membrane penalty.** Paired enzyme and cell potency, the one piece of
   experimental evidence in the project.
4. **Whether docking worked.** Score against measured potency, split by
   mechanism.

Writes:
    results/figures/project_summary.png

Usage:
    python src/summary_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURE_DIR = RESULTS / "figures"

TEAL = "#2a9d8f"
ORANGE = "#e76f51"
BLUE = "#457b9d"
GREY = "#adb5bd"
RED = "#e63946"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shortlist_path = RESULTS / "shortlist.csv"
    if not shortlist_path.exists():
        print(f"{shortlist_path} not found; run src/admet/screen.py", file=sys.stderr)
        return 1

    shortlist = pd.read_csv(shortlist_path)
    covalent = shortlist["covalent"].fillna(False).astype(bool)
    cns = shortlist["cns_candidate"].fillna(False).astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 1. the funnel -----------------------------------------------------
    stages = [
        ("Published\ninhibitors", len(shortlist)),
        ("Drug-like\n(Veber)", int(shortlist["veber_pass"].fillna(False).sum())),
        ("Predicted to\nreach the brain", int(cns.sum())),
        ("...and not\ncovalent", int((cns & ~covalent).sum())),
    ]
    labels = [name for name, _ in stages]
    counts = [count for _, count in stages]
    bars = axes[0, 0].bar(
        range(len(stages)), counts, color=[GREY, GREY, BLUE, TEAL]
    )
    for bar, count in zip(bars, counts):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    axes[0, 0].set_xticks(range(len(stages)))
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].set_ylabel("Compounds")
    axes[0, 0].set_title(
        "Almost nothing survives both requirements", fontsize=12, fontweight="bold"
    )
    # Name the step that does the damage.
    drop = counts[1] - counts[2]
    axes[0, 0].annotate(
        f"brain penetration\nremoves {drop}",
        # Point at the bar's left edge and label from above-left, so the
        # arrow does not run through the count printed on top of the bar.
        xy=(1.78, counts[2] + max(counts) * 0.03),
        xytext=(0.95, max(counts) * 0.52),
        fontsize=9,
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.2},
    )

    # --- 2. the conflict ---------------------------------------------------
    for mask, label, colour, marker in (
        (~covalent, "non-covalent", TEAL, "o"),
        (covalent, "covalent warhead", ORANGE, "^"),
    ):
        subset = shortlist[mask]
        axes[0, 1].scatter(
            subset["cns_mpo"],
            subset["pic50"],
            label=f"{label} (n={int(mask.sum())})",
            color=colour,
            marker=marker,
            s=34,
            alpha=0.75,
            edgecolor="none",
        )
    axes[0, 1].axvline(4.0, ls="--", color="#555", lw=1)
    axes[0, 1].annotate(
        "brain-penetrant\nside →",
        xy=(4.1, shortlist["pic50"].min() + 0.2),
        fontsize=8,
        color="#555",
    )
    axes[0, 1].set_xlabel("Predicted brain penetration (CNS MPO, 0-6)")
    axes[0, 1].set_ylabel("Measured potency (pIC50)")
    axes[0, 1].set_title(
        "The two requirements pull against each other",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].legend(frameon=False, fontsize=8, loc="lower left")

    # --- 3. the membrane penalty ------------------------------------------
    paired_path = PROCESSED / "permeability_gap.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        limits = [
            min(paired["pic50_cell"].min(), paired["pic50_enzyme"].min()) - 0.3,
            max(paired["pic50_cell"].max(), paired["pic50_enzyme"].max()) + 0.3,
        ]
        axes[1, 0].plot(limits, limits, ls="--", color="#555", lw=1)
        axes[1, 0].scatter(
            paired["pic50_enzyme"],
            paired["pic50_cell"],
            color=BLUE,
            s=34,
            alpha=0.8,
            edgecolor="none",
        )
        axes[1, 0].set_xlim(limits)
        axes[1, 0].set_ylim(limits)
        median_gap = paired["cell_gap"].median()
        axes[1, 0].annotate(
            f"every point below the line\nmedian {10 ** median_gap:.0f}x weaker in cells",
            xy=(0.04, 0.84),
            xycoords="axes fraction",
            fontsize=9,
            color=RED,
        )
        axes[1, 0].set_xlabel("Potency on purified enzyme (pIC50)")
        axes[1, 0].set_ylabel("Potency inside cells (pIC50)")
        axes[1, 0].set_title(
            "They lose most of their potency at a membrane",
            fontsize=12,
            fontweight="bold",
        )

    # --- 4. did docking work ----------------------------------------------
    analysis_path = RESULTS / "docking" / "screen_analysis.json"
    scores_path = RESULTS / "docking" / "docking_scores.csv"
    dataset_path = PROCESSED / "lgmn_regression.csv"
    if analysis_path.exists() and scores_path.exists():
        analysis = json.loads(analysis_path.read_text())
        merged = pd.read_csv(scores_path).merge(
            pd.read_csv(dataset_path)[["molecule_chembl_id", "pic50", "covalent"]],
            on="molecule_chembl_id",
        )
        merged = merged[merged["affinity"].notna()]
        is_cov = merged["covalent"].fillna(False).astype(bool)
        for mask, label, colour, marker in (
            (~is_cov, "non-covalent", TEAL, "o"),
            (is_cov, "covalent", ORANGE, "^"),
        ):
            subset = merged[mask]
            axes[1, 1].scatter(
                subset["affinity"],
                subset["pic50"],
                label=label,
                color=colour,
                marker=marker,
                s=30,
                alpha=0.7,
                edgecolor="none",
            )
        subsets = analysis.get("size_controlled_subsets", {})
        non_cov = subsets.get("non_covalent", {})
        cov = subsets.get("covalent", {})
        axes[1, 1].set_xlabel("Docking score (kcal/mol, more negative = better fit)")
        axes[1, 1].set_ylabel("Measured potency (pIC50)")
        if non_cov and cov:
            axes[1, 1].set_title(
                "Docking predicts potency, weakly\n"
                f"after removing size effects: rho {non_cov['partial_rho']:.2f} "
                f"non-covalent, {cov['partial_rho']:.2f} covalent",
                fontsize=12,
                fontweight="bold",
            )
        axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Computational search for brain-penetrant legumain inhibitors",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "project_summary.png", dpi=170)
    plt.close(fig)

    print(f"Wrote {FIGURE_DIR / 'project_summary.png'}")
    print(
        "Funnel: "
        + " -> ".join(f"{c}" for c in counts)
        + f"   ({counts[0]} published inhibitors down to {counts[-1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
