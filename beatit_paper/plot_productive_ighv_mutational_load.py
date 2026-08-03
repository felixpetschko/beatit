#!/usr/bin/env python3
"""Summarize and plot productive IgH V-region mutation load."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parent
REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3"]
GROUPS = {
    "EµMyc": ["A1", "A3", "A5", "A7", "A9", "A11"],
    "EµMyc/Tet2KO": ["C1", "C5", "C7", "C9", "C11"],
}
COLORS = {"EµMyc": "#4C78A8", "EµMyc/Tet2KO": "#E45756"}
MIN_BASES, MIN_READS = 150, 10


def load_sample(sample, genotype, min_bases=MIN_BASES, min_reads=MIN_READS):
    """Calculate per-clone and per-sample IGHV metrics."""
    path = ROOT / "mutational_load" / sample / f"{sample}_productive_IGHV_regions.tsv"
    ref_path = ROOT / "IGH_productive" / f"{sample}_productive_IGH.tsv"
    clones, ref = pd.read_csv(path, sep="\t"), pd.read_csv(ref_path, sep="\t")

    # Confirm that contig assembly preserved the productive CDR3 repertoire.
    a = clones.set_index("nSeqCDR3")["readCount"].sort_index()
    b = ref.set_index("nSeqCDR3")["readCount"].sort_index()
    if not a.index.equals(b.index) or not np.allclose(a, b):
        raise ValueError(f"Productive CDR3 repertoire changed for {sample}")
    if clones["aaSeqCDR3"].astype(str).str.contains("*", regex=False).any():
        raise ValueError(f"Nonproductive CDR3 found for {sample}")

    length_cols = [f"nLength{r}" for r in REGIONS]
    mutation_cols = [f"nMutationsCount{r}Substitutions" for r in REGIONS]
    lengths = clones[length_cols].apply(pd.to_numeric, errors="coerce")
    mutations = clones[mutation_cols].apply(pd.to_numeric, errors="coerce")
    if not lengths.notna().set_axis(REGIONS, axis=1).equals(
        mutations.notna().set_axis(REGIONS, axis=1)
    ):
        raise ValueError(f"Inconsistent region coverage for {sample}")

    clones.insert(0, "sample", sample)
    clones.insert(1, "genotype", genotype)
    clones["covered_ighv_bases"] = lengths.fillna(0).sum(axis=1)
    clones["ighv_substitutions"] = mutations.fillna(0).sum(axis=1)
    clones["covered_regions"] = lengths.notna().set_axis(REGIONS, axis=1).apply(
        lambda row: "+".join(row.index[row]), axis=1
    )
    clones["eligible"] = (clones.covered_ighv_bases >= min_bases) & (
        clones.readCount >= min_reads
    )
    clones["ighv_mutation_load_percent"] = np.where(
        clones.eligible,
        100 * clones.ighv_substitutions / clones.covered_ighv_bases,
        np.nan,
    )

    eligible = clones[clones.eligible]
    if eligible.empty:
        raise ValueError(f"No eligible IGHV clonotypes for {sample}")
    rate, weight = eligible.ighv_mutation_load_percent, eligible.readCount
    summary = {
        "sample": sample,
        "genotype": genotype,
        "total_productive_igh_clonotypes": len(clones),
        "eligible_ighv_clonotypes": len(eligible),
        "eligible_fraction_percent": 100 * len(eligible) / len(clones),
        "eligible_productive_igh_reads": int(weight.sum()),
        "median_covered_ighv_bases": eligible.covered_ighv_bases.median(),
        "clone_weighted_mean_mutation_load_percent": rate.mean(),
        "clone_weighted_median_mutation_load_percent": rate.median(),
        "covered_base_pooled_mutation_load_percent": (
            100 * eligible.ighv_substitutions.sum() / eligible.covered_ighv_bases.sum()
        ),
        "read_weighted_mean_mutation_load_percent": np.average(rate, weights=weight),
        "clone_weighted_mean_germline_identity_percent": 100 - rate.mean(),
    }
    return clones, summary


def significance_bar(ax, p, maximum, lower, upper):
    """Draw the exact Mann–Whitney comparison above two groups."""
    span = upper - lower
    y, top = maximum + 0.07 * span, maximum + 0.11 * span
    ax.plot([0, 0, 1, 1], [y, top, top, y], color="black", lw=1.1)
    ax.text(
        0.5,
        top + 0.015 * span,
        f"Exact Mann–Whitney $p$ = {p:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_ylim(lower, upper)


def make_plot(summary, p):
    """Plot mutation load and the complementary germline identity."""
    panels = [
        ("clone_weighted_mean_mutation_load_percent", "IGHV mutation load"),
        ("clone_weighted_mean_germline_identity_percent", "IGHV germline identity"),
    ]
    order = list(GROUPS)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.6))

    for i, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        sns.boxplot(
            data=summary,
            x="genotype",
            y=metric,
            hue="genotype",
            order=order,
            palette=COLORS,
            legend=False,
            width=0.52,
            saturation=0.82,
            showfliers=False,
            linewidth=1.2,
            medianprops={"color": "black", "linewidth": 1.8},
            ax=ax,
        )
        np.random.seed(20260803 + i)
        sns.stripplot(
            data=summary,
            x="genotype",
            y=metric,
            order=order,
            jitter=0.16,
            size=7,
            color="black",
            edgecolor="white",
            linewidth=0.7,
            ax=ax,
        )
        values = summary[metric]
        lower, upper = (0, max(values.max() * 1.38, 0.5)) if i == 0 else (
            max(0, values.min() - 0.13),
            100.08,
        )
        significance_bar(ax, p, values.max(), lower, upper)
        ax.set_title(title, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(f"Clone-weighted {title} (%)")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [f"{group}\n($n$ = {len(GROUPS[group])})" for group in order]
        )
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Productive IgH V-region maturation state\nin premalignant IgM⁺ B cells",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        f"FR1–FR3 and CDR1/CDR2; CDR3 excluded\n"
        f"≥{MIN_BASES} covered nt and ≥{MIN_READS} reads per clonotype",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.22, top=0.78, left=0.10, right=0.98, wspace=0.34)
    return fig


def main():
    clone_tables, records = [], []
    for genotype, samples in GROUPS.items():
        for sample in samples:
            clones, summary = load_sample(sample, genotype)
            clone_tables.append(clones)
            records.append(summary)

    clones, summary = pd.concat(clone_tables), pd.DataFrame(records)
    metric = "clone_weighted_mean_mutation_load_percent"
    a = summary.loc[summary.genotype == "EµMyc", metric]
    c = summary.loc[summary.genotype == "EµMyc/Tet2KO", metric]
    test = mannwhitneyu(a, c, alternative="two-sided", method="exact")

    summary.to_csv(
        ROOT / "productive_ighv_mutational_load_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    clones.to_csv(
        ROOT / "mutational_load/productive_ighv_mutational_load_per_clone.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
        float_format="%.6f",
    )
    figure = make_plot(summary, test.pvalue)
    output = ROOT / "figures/productive_ighv_mutational_load_premalignant_igm_positive.png"
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(summary.to_string(index=False))
    print(f"\nExact Mann–Whitney U = {test.statistic:.1f}, p = {test.pvalue:.6f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
