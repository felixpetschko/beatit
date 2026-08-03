#!/usr/bin/env python3
"""Plot productive IgH top-1 and cumulative top-5 clonal dominance."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "IGH_productive"
OUTPUT_DIR = SCRIPT_DIR / "figures"

GROUPS = {
    "EµMyc": ["A1", "A3", "A5", "A7", "A9", "A11"],
    "EµMyc/Tet2KO": ["C1", "C5", "C7", "C9", "C11"],
}

COLORS = {
    "EµMyc": "#4C78A8",
    "EµMyc/Tet2KO": "#E45756",
}

METRICS = {
    "top1_frequency_percent": "Most dominant clone",
    "top5_frequency_percent": "Cumulative top 5 clones",
}


def load_dominance_data() -> pd.DataFrame:
    """Calculate productive IgH dominance metrics from MiXCR read counts."""
    records = []

    for genotype, samples in GROUPS.items():
        for sample in samples:
            input_path = DATA_DIR / f"{sample}_productive_IGH.tsv"
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing productive IgH export: {input_path}")

            clones = pd.read_csv(input_path, sep="\t")
            required_columns = {"readCount", "nSeqCDR3", "aaSeqCDR3"}
            missing_columns = required_columns.difference(clones.columns)
            if missing_columns:
                raise ValueError(
                    f"{input_path.name} is missing columns: {sorted(missing_columns)}"
                )
            if clones.empty:
                raise ValueError(f"No productive IgH clonotypes in {input_path.name}")
            if clones["aaSeqCDR3"].astype(str).str.contains("*", regex=False).any():
                raise ValueError(f"Stop codon found in productive export {input_path.name}")
            if (clones["nSeqCDR3"].astype(str).str.len() % 3 != 0).any():
                raise ValueError(f"Out-of-frame CDR3 found in {input_path.name}")

            read_counts = clones["readCount"].astype(float).sort_values(ascending=False)
            total_reads = float(read_counts.sum())
            if total_reads <= 0:
                raise ValueError(f"No positive IgH read counts in {input_path.name}")

            records.append(
                {
                    "sample": sample,
                    "genotype": genotype,
                    "productive_igh_clonotypes": int(len(clones)),
                    "productive_igh_reads": int(total_reads),
                    "top1_frequency_percent": 100.0 * read_counts.iloc[0] / total_reads,
                    "top5_frequency_percent": 100.0 * read_counts.iloc[:5].sum() / total_reads,
                }
            )

    return pd.DataFrame.from_records(records)


def exact_mann_whitney(summary: pd.DataFrame, metric: str):
    """Compare the two genotypes with an exact two-sided Mann–Whitney test."""
    emyc = summary.loc[summary["genotype"] == "EµMyc", metric]
    tet2ko = summary.loc[summary["genotype"] == "EµMyc/Tet2KO", metric]
    return mannwhitneyu(emyc, tet2ko, alternative="two-sided", method="exact")


def add_significance_bar(ax: plt.Axes, p_value: float, observed_max: float) -> None:
    """Draw a comparison bracket above the observed samples."""
    bracket_bottom = min(observed_max + 5.0, 96.0)
    bracket_top = bracket_bottom + 3.0
    ax.plot(
        [0, 0, 1, 1],
        [bracket_bottom, bracket_top, bracket_top, bracket_bottom],
        color="black",
        linewidth=1.1,
        clip_on=False,
    )
    ax.text(
        0.5,
        bracket_top + 1.0,
        f"Exact Mann–Whitney $p$ = {p_value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def plot_dominance(summary: pd.DataFrame, tests: dict) -> plt.Figure:
    """Create the two-panel Seaborn dominance figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    order = list(GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.3), sharey=True)

    for panel_index, (ax, (metric, title)) in enumerate(zip(axes, METRICS.items())):
        sns.boxplot(
            data=summary,
            x="genotype",
            y=metric,
            hue="genotype",
            order=order,
            hue_order=order,
            palette=COLORS,
            legend=False,
            width=0.52,
            saturation=0.82,
            showfliers=False,
            linewidth=1.2,
            whiskerprops={"color": "#555555", "linewidth": 1.2},
            capprops={"color": "#555555", "linewidth": 1.2},
            medianprops={"color": "black", "linewidth": 1.8},
            ax=ax,
        )

        np.random.seed(20260803 + panel_index)
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
            zorder=3,
        )

        add_significance_bar(ax, tests[metric].pvalue, float(summary[metric].max()))
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0, 112)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"EµMyc\n($n$ = {len(GROUPS['EµMyc'])})",
                f"EµMyc/Tet2KO\n($n$ = {len(GROUPS['EµMyc/Tet2KO'])})",
            ]
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Productive IgH reads (%)" if panel_index == 0 else "")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Productive IgH clonal dominance in premalignant IgM⁺ B cells",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = load_dominance_data()
    tests = {metric: exact_mann_whitney(summary, metric) for metric in METRICS}

    summary_path = SCRIPT_DIR / "productive_igh_clonal_dominance_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.6f")

    figure = plot_dominance(summary, tests)
    output_path = (
        OUTPUT_DIR / "productive_igh_clonal_dominance_premalignant_igm_positive.png"
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(summary.to_string(index=False))
    for metric, test in tests.items():
        print(
            f"{METRICS[metric]}: exact Mann–Whitney U = {test.statistic:.1f}, "
            f"p = {test.pvalue:.6f}"
        )
    print(f"Wrote {summary_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
