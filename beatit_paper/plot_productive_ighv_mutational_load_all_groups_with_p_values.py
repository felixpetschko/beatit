#!/usr/bin/env python3
"""Plot all-group productive IGHV metrics with exact genotype comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from plot_productive_ighv_mutational_load import COLORS
from plot_productive_ighv_mutational_load_all_groups import CLASS_ORDER


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "productive_ighv_mutational_load_all_groups_summary.tsv"
OUTPUT = ROOT / "figures/productive_ighv_mutational_load_all_groups_with_p_values.png"
GENOTYPES = ["EµMyc", "EµMyc/Tet2KO"]
PANELS = [
    ("clone_weighted_mean_mutation_load_percent", "IGHV mutation load"),
    ("clone_weighted_mean_germline_identity_percent", "IGHV germline identity"),
]


def exact_tests(data, metric):
    """Return two-sided exact Mann–Whitney results by sample class."""
    results = {}
    for sample_class in CLASS_ORDER:
        subset = data[data["sample_class"] == sample_class]
        first = subset.loc[subset["genotype"] == GENOTYPES[0], metric].dropna()
        second = subset.loc[subset["genotype"] == GENOTYPES[1], metric].dropna()
        results[sample_class] = mannwhitneyu(
            first, second, alternative="two-sided", method="exact"
        )
    return results


def add_p_values(ax, data, metric, tests):
    """Add one compact genotype-comparison bracket per sample class."""
    values = data[metric].dropna()
    span = values.max() - values.min()
    bracket_height = 0.025 * span
    offset = 0.045 * span
    text_offset = 0.012 * span
    highest = values.max()

    for x, sample_class in enumerate(CLASS_ORDER):
        local = data.loc[data["sample_class"] == sample_class, metric].dropna()
        bottom = local.max() + offset
        top = bottom + bracket_height
        ax.plot(
            [x - 0.20, x - 0.20, x + 0.20, x + 0.20],
            [bottom, top, top, bottom],
            color="black",
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            x,
            top + text_offset,
            f"$p$ = {tests[sample_class].pvalue:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        highest = max(highest, top + text_offset + 0.06 * span)

    lower, _ = ax.get_ylim()
    ax.set_ylim(lower, highest)


def make_plot(data):
    """Create the two-panel plot with exact Mann–Whitney p values."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0))

    for panel_no, (ax, (metric, title)) in enumerate(zip(axes, PANELS)):
        sns.boxplot(
            data=data,
            x="sample_class",
            y=metric,
            hue="genotype",
            order=CLASS_ORDER,
            hue_order=GENOTYPES,
            palette=COLORS,
            width=0.7,
            dodge=True,
            showfliers=False,
            linewidth=1.0,
            medianprops={"color": "black", "linewidth": 1.5},
            ax=ax,
        )
        np.random.seed(20260803 + panel_no)
        sns.stripplot(
            data=data,
            x="sample_class",
            y=metric,
            hue="genotype",
            order=CLASS_ORDER,
            hue_order=GENOTYPES,
            palette=COLORS,
            dodge=True,
            jitter=0.14,
            size=5.5,
            edgecolor="white",
            linewidth=0.6,
            legend=False,
            ax=ax,
        )
        add_p_values(ax, data, metric, exact_tests(data, metric))
        ax.set_title(title, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(f"Clone-weighted {title} (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        title="Genotype",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Productive IgH V-region maturation state across all sample classes",
        y=0.98,
    )
    fig.text(
        0.5,
        0.915,
        "Two-sided exact Mann–Whitney U tests; unadjusted p values",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.18, top=0.76, left=0.07, right=0.98, wspace=0.28)
    return fig


def main():
    data = pd.read_csv(INPUT, sep="\t")
    figure = make_plot(data)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    tests = exact_tests(data, PANELS[0][0])
    for sample_class in CLASS_ORDER:
        result = tests[sample_class]
        print(f"{sample_class}: U = {result.statistic:.1f}, p = {result.pvalue:.6f}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
