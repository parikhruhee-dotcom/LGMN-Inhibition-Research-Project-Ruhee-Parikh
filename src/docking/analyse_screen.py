"""Does the docking score relate to measured potency on this target?

The redocking validation showed smina cannot rank poses of a single ligand
correctly for legumain. That predicts it cannot rank ligands either, but
predicting is not measuring. Every compound here has a published pIC50, so
the question can be settled directly rather than argued.

Three things are computed, and the distinctions matter:

- **Correlation over all scored compounds.** The headline number.
- **Restricted to poses that contact the catalytic thiol.** A compound scoring
  well from a surface groove is not making a prediction about inhibition, so
  including it tests the wrong thing.
- **Split by mechanism.** 85% of these compounds inhibit by forming a covalent
  bond, which the scoring function has no term for. Their potency is governed
  by a reaction rate the score cannot see, so the non-covalent subset is the
  only fair test of the scoring function, even though it is small.

Enrichment is reported alongside correlation because virtual screening is used
to pick a top slice, not to predict potencies: a method can rank poorly
overall and still concentrate actives at the top, which is what actually
matters in practice.

Writes:
    results/docking/screen_analysis.json
    results/figures/docking_vs_potency.png

Usage:
    python src/docking/analyse_screen.py [--partial]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "docking"
FIGURE_DIR = ROOT / "results" / "figures"

# "Active" is defined as the top quartile of measured potency, computed from
# the dataset rather than fixed.
#
# A fixed 100 nM cut is useless here: this is a set of published inhibitors,
# not a screening library with decoys, and 88% of it already clears pIC50 7,
# so enrichment against that line is meaningless. The honest question for a
# set of known binders is harder -- can docking pick the most potent out of a
# field of actives -- and a quartile cut asks exactly that with a 25% base
# rate.
ACTIVE_QUANTILE = 0.75

# Fraction of the ranked list treated as the "selected" slice when computing
# enrichment, matching how a virtual screen is actually used.
TOP_FRACTION = 0.1

# Below this many compounds a correlation is too unstable to report.
MIN_FOR_STATS = 10


def correlation(frame: pd.DataFrame, label: str) -> dict:
    """Spearman and Pearson of affinity against potency for one subset.

    A working scoring function gives a NEGATIVE correlation: more negative
    affinity (tighter predicted binding) should mean higher pIC50.
    """
    subset = frame[["affinity", "pic50"]].dropna()
    result: dict = {"subset": label, "n": len(subset)}
    if len(subset) < MIN_FOR_STATS:
        result["note"] = "too few compounds for a stable estimate"
        return result

    rho, rho_p = stats.spearmanr(subset["affinity"], subset["pic50"])
    r, r_p = stats.pearsonr(subset["affinity"], subset["pic50"])
    result.update(
        {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "pearson_r": float(r),
            "pearson_p": float(r_p),
            # The sign convention is easy to misread, so state the verdict.
            "direction_correct": bool(rho < 0),
            "significant": bool(rho_p < 0.05),
        }
    )
    return result


def enrichment(
    frame: pd.DataFrame,
    active_threshold: float,
    top_fraction: float = TOP_FRACTION,
) -> dict:
    """Enrichment factor for the best-scoring slice of the ranked list.

    EF = (share of actives in the top slice) / (share of actives overall).
    EF of 1.0 is random; a useful virtual screen gives several-fold.
    """
    subset = frame[["affinity", "pic50"]].dropna().copy()
    if len(subset) < MIN_FOR_STATS:
        return {"n": len(subset), "note": "too few compounds"}

    subset["active"] = subset["pic50"] >= active_threshold
    overall_rate = subset["active"].mean()
    n_top = max(1, int(round(len(subset) * top_fraction)))

    # More negative affinity ranks first.
    top = subset.nsmallest(n_top, "affinity")
    top_rate = top["active"].mean()

    return {
        "n": len(subset),
        "n_top": n_top,
        "top_fraction": top_fraction,
        "active_threshold_pic50": float(active_threshold),
        "overall_active_rate": float(overall_rate),
        "top_active_rate": float(top_rate),
        "enrichment_factor": (
            float(top_rate / overall_rate) if overall_rate > 0 else None
        ),
    }


def partial_spearman(x, y, control) -> tuple[float, float]:
    """Spearman of x against y with the effect of `control` removed.

    Vina-family scoring functions grow more negative with ligand size, and in
    this dataset potency also rises with size, so a raw correlation between
    score and potency can be entirely an artefact of molecular weight. This
    ranks all three, regresses the control out of each, and correlates the
    residuals.
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(control)

    def residual(a, b):
        slope, intercept = np.polyfit(b, a, 1)
        return a - (slope * b + intercept)

    r, p = stats.pearsonr(residual(rx, rz), residual(ry, rz))
    return float(r), float(p)


def representativeness(scored: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    """Is the scored subset a fair sample of the dataset's potency range?

    A partial screen of a potency-sorted list covers only the potent end, and
    a correlation over a restricted range is not comparable to one over the
    full range. This makes that visible instead of leaving the reader to
    assume the subset is representative.
    """
    if scored.empty:
        return {}
    gap = abs(scored["pic50"].median() - dataset["pic50"].median())
    return {
        "subset_median_pic50": float(scored["pic50"].median()),
        "dataset_median_pic50": float(dataset["pic50"].median()),
        "subset_pic50_range": [
            float(scored["pic50"].min()),
            float(scored["pic50"].max()),
        ],
        "dataset_pic50_range": [
            float(dataset["pic50"].min()),
            float(dataset["pic50"].max()),
        ],
        "median_gap": float(gap),
        # Half a log unit of drift in the median means the sample is skewed
        # enough that the correlation should not be read as final.
        "representative": bool(gap <= 0.5),
    }


def analyse(scores: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    """Join docking scores to measured potency and evaluate."""
    merged = scores.merge(
        dataset[["molecule_chembl_id", "pic50", "covalent", "heavy_atoms"]],
        on="molecule_chembl_id",
        how="inner",
    )
    merged = merged[merged["affinity"].notna()]

    in_site = merged[merged["in_active_site"].fillna(False).astype(bool)]
    non_covalent = merged[~merged["covalent"].fillna(False).astype(bool)]
    covalent = merged[merged["covalent"].fillna(False).astype(bool)]

    active_threshold = float(dataset["pic50"].quantile(ACTIVE_QUANTILE))

    def size_control(frame: pd.DataFrame) -> dict:
        usable = frame[["affinity", "pic50", "heavy_atoms"]].dropna()
        if len(usable) < MIN_FOR_STATS or usable["heavy_atoms"].nunique() <= 2:
            return {}
        rho, p = partial_spearman(
            usable["affinity"], usable["pic50"], usable["heavy_atoms"]
        )
        return {
            "n": len(usable),
            "partial_rho": rho,
            "partial_p": p,
            "survives_size_control": bool(p < 0.05 and rho < 0),
        }

    # The whole-set control mixes two populations with different physics. A
    # scoring function has no term for covalent bond formation, so for 85% of
    # these compounds potency is governed by a reaction rate the score cannot
    # see. The non-covalent subset is where docking is even expected to work,
    # and it deserves its own control rather than being averaged away.
    size_controlled_subsets = {
        "non_covalent": size_control(non_covalent),
        "covalent": size_control(covalent),
    }

    size_controlled = {}
    usable = merged[["affinity", "pic50", "heavy_atoms"]].dropna()
    if len(usable) >= MIN_FOR_STATS and usable["heavy_atoms"].nunique() > 2:
        rho, p = partial_spearman(
            usable["affinity"], usable["pic50"], usable["heavy_atoms"]
        )
        raw_size, raw_size_p = stats.spearmanr(
            usable["affinity"], usable["heavy_atoms"]
        )
        size_potency, size_potency_p = stats.spearmanr(
            usable["heavy_atoms"], usable["pic50"]
        )
        size_controlled = {
            "n": len(usable),
            "partial_rho": rho,
            "partial_p": p,
            "affinity_vs_size_rho": float(raw_size),
            "affinity_vs_size_p": float(raw_size_p),
            "size_vs_potency_rho": float(size_potency),
            "size_vs_potency_p": float(size_potency_p),
            "survives_size_control": bool(p < 0.05 and rho < 0),
        }

    return {
        "n_scored": len(merged),
        "active_threshold_pic50": active_threshold,
        "representativeness": representativeness(merged, dataset),
        "size_controlled": size_controlled,
        "size_controlled_subsets": size_controlled_subsets,
        "n_in_active_site": len(in_site),
        "n_non_covalent": len(non_covalent),
        "affinity_min": float(merged["affinity"].min()),
        "affinity_max": float(merged["affinity"].max()),
        "correlations": [
            correlation(merged, "all scored compounds"),
            correlation(in_site, "pose contacts catalytic thiol"),
            correlation(non_covalent, "non-covalent only"),
            correlation(covalent, "covalent only"),
        ],
        "enrichment": enrichment(merged, active_threshold),
        "enrichment_non_covalent": enrichment(non_covalent, active_threshold),
    }


def make_figure(scores: pd.DataFrame, dataset: pd.DataFrame, summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = scores.merge(
        dataset[["molecule_chembl_id", "pic50", "covalent"]],
        on="molecule_chembl_id",
        how="inner",
    )
    merged = merged[merged["affinity"].notna()]
    if merged.empty:
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    covalent = merged["covalent"].fillna(False).astype(bool)
    for mask, label, colour, marker in (
        (~covalent, "non-covalent", "#2a9d8f", "o"),
        (covalent, "covalent warhead", "#e76f51", "^"),
    ):
        subset = merged[mask]
        if subset.empty:
            continue
        axes[0].scatter(
            subset["affinity"], subset["pic50"],
            label=f"{label} (n={len(subset)})",
            color=colour, marker=marker, s=38, alpha=0.75, edgecolor="none",
        )

    overall = summary["correlations"][0]
    if "spearman_rho" in overall:
        subset = merged[["affinity", "pic50"]].dropna()
        slope, intercept = np.polyfit(subset["affinity"], subset["pic50"], 1)
        xs = np.linspace(subset["affinity"].min(), subset["affinity"].max(), 50)
        axes[0].plot(xs, slope * xs + intercept, color="#555", lw=1.5, ls="--")
        axes[0].set_title(
            f"Docking score vs measured potency\n"
            f"Spearman {overall['spearman_rho']:+.2f} "
            f"(p={overall['spearman_p']:.2g}, n={overall['n']}); "
            "a working score would be negative"
        )
    else:
        axes[0].set_title("Docking score vs measured potency")

    axes[0].set_xlabel("smina affinity (kcal/mol), more negative = better score")
    axes[0].set_ylabel("Measured pIC50")
    axes[0].legend(frameon=False, fontsize=8)

    # --- what you would actually get from a top slice ---
    enrich = summary["enrichment"]
    if "enrichment_factor" in enrich and enrich["enrichment_factor"] is not None:
        labels = ["Whole set", f"Top {int(100 * enrich['top_fraction'])}% by score"]
        values = [
            100 * enrich["overall_active_rate"],
            100 * enrich["top_active_rate"],
        ]
        axes[1].bar(labels, values, color=["#adb5bd", "#457b9d"])
        for index, value in enumerate(values):
            axes[1].text(
                index, value, f" {value:.0f}%", ha="center", va="bottom",
                fontsize=9,
            )
        axes[1].axhline(
            100 * enrich["overall_active_rate"], ls="--", color="#555", lw=1
        )
        axes[1].set_ylabel(
            f"% with pIC50 >= {enrich['active_threshold_pic50']:.0f}"
        )
        axes[1].set_title(
            f"Enrichment factor {enrich['enrichment_factor']:.2f}x\n"
            "(1.0 = no better than picking at random)"
        )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "docking_vs_potency.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="analyse the in-progress checkpoint instead of the final file",
    )
    args = parser.parse_args()

    final = RESULTS / "docking_scores.csv"
    partial = RESULTS / "docking_scores.partial.csv"
    path = partial if (args.partial or not final.exists()) else final
    if not path.exists():
        print(f"{path} not found; run src/docking/dock.py first", file=sys.stderr)
        return 1

    scores = pd.read_csv(path)
    dataset = pd.read_csv(PROCESSED / "lgmn_regression.csv")
    summary = analyse(scores, dataset)
    summary["source"] = path.name
    summary["complete"] = path == final

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "screen_analysis.json").write_text(json.dumps(summary, indent=2))

    print(f"Source: {path.name} ({summary['n_scored']} scored compounds)")
    if not summary["complete"]:
        print("NOTE: analysing a partial run; numbers will move.\n")

    for entry in summary["correlations"]:
        if "spearman_rho" not in entry:
            print(f"  {entry['subset']:34} n={entry['n']:3}  {entry.get('note', '')}")
            continue
        verdict = (
            "correct direction"
            if entry["direction_correct"]
            else "WRONG DIRECTION"
        )
        print(
            f"  {entry['subset']:34} n={entry['n']:3}  "
            f"Spearman {entry['spearman_rho']:+.3f} "
            f"(p={entry['spearman_p']:.3g})  {verdict}"
            f"{'' if entry['significant'] else ', not significant'}"
        )

    size = summary.get("size_controlled") or {}
    if size:
        print(
            f"\nControlling for ligand size: partial Spearman "
            f"{size['partial_rho']:+.3f} (p={size['partial_p']:.3g})"
        )
        print(
            f"  affinity vs size {size['affinity_vs_size_rho']:+.2f}, "
            f"size vs potency {size['size_vs_potency_rho']:+.2f} -- "
            + (
                "relationship survives the control"
                if size["survives_size_control"]
                else "relationship does NOT survive; it tracks size"
            )
        )

    for label, entry in (summary.get("size_controlled_subsets") or {}).items():
        if entry:
            print(
                f"    {label:14} n={entry['n']:3} partial rho "
                f"{entry['partial_rho']:+.3f} (p={entry['partial_p']:.3g})"
                + ("  survives" if entry["survives_size_control"] else "  does not survive")
            )

    rep = summary.get("representativeness") or {}
    if rep and not rep.get("representative", True):
        print(
            f"\nWARNING: the scored subset is not representative. "
            f"Median pIC50 {rep['subset_median_pic50']:.2f} against "
            f"{rep['dataset_median_pic50']:.2f} for the full set.\n"
            "  A correlation over a restricted potency range is not "
            "comparable to one\n  over the full range; treat these numbers "
            "as provisional."
        )

    enrich = summary["enrichment"]
    if enrich.get("enrichment_factor") is not None:
        print(
            f"\nEnrichment in the top {int(100 * enrich['top_fraction'])}%: "
            f"{enrich['enrichment_factor']:.2f}x "
            f"({100 * enrich['top_active_rate']:.0f}% actives vs "
            f"{100 * enrich['overall_active_rate']:.0f}% overall, "
            f"1.0x = random)"
        )

    make_figure(scores, dataset, summary)
    print(f"\nWrote {RESULTS / 'screen_analysis.json'}")
    print(f"Wrote {FIGURE_DIR / 'docking_vs_potency.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
