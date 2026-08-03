#!/usr/bin/env python3
"""Plot productive IgH CDR3 clonotype richness for premalignant IgM+ samples."""

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

def load_richness_data() -> pd.DataFrame:
    """Load productive IgH exports and return one summary row per sample."""
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

            records.append(
                {
                    "sample": sample,
                    "genotype": genotype,
                    "productive_igh_clonotypes": int(len(clones)),
                    "productive_igh_reads": int(clones["readCount"].sum()),
                }
            )

    return pd.DataFrame.from_records(records)


def add_significance_bar(ax: plt.Axes, p_value: float, y_max: float) -> None:
    """Add a compact comparison bracket above the two groups."""
    bracket_bottom = y_max * 1.04
    bracket_top = y_max * 1.09
    ax.plot(
        [0, 0, 1, 1],
        [bracket_bottom, bracket_top, bracket_top, bracket_bottom],
        color="black",
        linewidth=1.1,
        clip_on=False,
    )
    ax.text(
        0.5,
        y_max * 1.105,
        f"Exact Mann–Whitney $p$ = {p_value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


def plot_richness(summary: pd.DataFrame, p_value: float) -> plt.Figure:
    """Create a Seaborn richness boxplot with jittered individual samples."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(5.0, 5.4))

    order = list(GROUPS)
    sns.boxplot(
        data=summary,
        x="genotype",
        y="productive_igh_clonotypes",
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

    np.random.seed(20260803)
    sns.stripplot(
        data=summary,
        x="genotype",
        y="productive_igh_clonotypes",
        order=order,
        jitter=0.16,
        size=7,
        color="black",
        edgecolor="white",
        linewidth=0.7,
        ax=ax,
        zorder=3,
    )

    observed_max = float(summary["productive_igh_clonotypes"].max())
    add_significance_bar(ax, p_value, observed_max)

    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0, observed_max * 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"EµMyc\n($n$ = {len(GROUPS['EµMyc'])})",
            f"EµMyc/Tet2KO\n($n$ = {len(GROUPS['EµMyc/Tet2KO'])})",
        ]
    )
    ax.set_ylabel("Productive IgH CDR3 clonotypes")
    ax.set_xlabel("")
    ax.set_title(
        "Productive IgH clonotype richness in premalignant IgM⁺ B cells",
        pad=14,
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = load_richness_data()

    emyc = summary.loc[
        summary["genotype"] == "EµMyc", "productive_igh_clonotypes"
    ]
    tet2ko = summary.loc[
        summary["genotype"] == "EµMyc/Tet2KO", "productive_igh_clonotypes"
    ]
    test = mannwhitneyu(emyc, tet2ko, alternative="two-sided", method="exact")

    summary_path = SCRIPT_DIR / "productive_igh_richness_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    figure = plot_richness(summary, test.pvalue)
    output_stem = OUTPUT_DIR / "productive_igh_richness_premalignant_igm_positive"
    figure.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(summary.to_string(index=False))
    print(f"\nExact Mann–Whitney U = {test.statistic:.1f}, p = {test.pvalue:.6f}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {output_stem}.png")


if __name__ == "__main__":
    main()
