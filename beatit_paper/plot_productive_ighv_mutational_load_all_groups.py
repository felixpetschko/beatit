#!/usr/bin/env python3
"""Plot productive IGHV mutation metrics across all sample classes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_productive_ighv_mutational_load import load_sample, COLORS


ROOT = Path(__file__).resolve().parent
META = ROOT / "supplementary_table_igh_summary.tsv"
OUT_TABLE = ROOT / "productive_ighv_mutational_load_all_groups_summary.tsv"
OUT_FIGURE = ROOT / "figures/productive_ighv_mutational_load_all_groups.png"
CLASS_ORDER = [
    "Premalignant IgM⁺",
    "Premalignant IgM⁻",
    "Malignant IgM⁺",
    "Malignant IgM⁻",
    "Malignant mixed",
]


def sample_class(stage):
    disease_stage, cell_fraction = stage.split(maxsplit=1)
    disease_stage = {
        "tumor": "Malignant",
        "malignant": "Malignant",
        "premalignant": "Premalignant",
    }[disease_stage]
    return f"{disease_stage} {cell_fraction}"


def summarize_all():
    metadata = pd.read_csv(META, sep="\t")
    records = []
    for _, row in metadata.iterrows():
        sample, genotype = row["Sample ID"], row["Genotype"]
        try:
            # No minimum read or coverage-length cutoff. A positive covered
            # length is mathematically required to calculate a mutation rate.
            _, record = load_sample(sample, genotype, min_bases=1, min_reads=0)
        except ValueError as error:
            if "No eligible IGHV clonotypes" not in str(error):
                raise
            record = {
                "sample": sample,
                "genotype": genotype,
                "eligible_ighv_clonotypes": 0,
                "clone_weighted_mean_mutation_load_percent": np.nan,
                "clone_weighted_mean_germline_identity_percent": np.nan,
            }
        record["sample_class"] = sample_class(row["Cell fraction / stage"])
        records.append(record)
    return pd.DataFrame(records)


def make_plot(data):
    panels = [
        ("clone_weighted_mean_mutation_load_percent", "IGHV mutation load"),
        ("clone_weighted_mean_germline_identity_percent", "IGHV germline identity"),
    ]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))

    for panel_no, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        sns.boxplot(
            data=data, x="sample_class", y=metric, hue="genotype",
            order=CLASS_ORDER, palette=COLORS, width=0.7, dodge=True,
            showfliers=False, linewidth=1.0,
            medianprops={"color": "black", "linewidth": 1.5}, ax=ax,
        )
        np.random.seed(20260803 + panel_no)
        sns.stripplot(
            data=data, x="sample_class", y=metric, hue="genotype",
            order=CLASS_ORDER, palette=COLORS, dodge=True, jitter=0.14,
            size=5.5, edgecolor="white", linewidth=0.6, legend=False, ax=ax,
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(f"Clone-weighted {title} (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        if panel_no == 0:
            ax.legend(title="Genotype", frameon=False, loc="upper right")
        else:
            ax.get_legend().remove()

    counts = (
        data.dropna(subset=["clone_weighted_mean_mutation_load_percent"])
        .groupby(["sample_class", "genotype"], observed=True).size()
    )
    count_text = "; ".join(
        f"{group}: {counts.get((group, 'EµMyc'), 0)}/{counts.get((group, 'EµMyc/Tet2KO'), 0)}"
        for group in CLASS_ORDER
    )
    fig.suptitle("Productive IgH V-region maturation state across all sample classes", y=0.98)
    fig.text(
        0.5, 0.025,
        "FR1–FR3 and CDR1/CDR2; CDR3 excluded; no minimum read or coverage-length cutoff\n"
        "Clonotypes with 0 covered IGHV nt excluded; "
        "Plotted samples (EµMyc/EµMyc-Tet2KO): " + count_text,
        ha="center", fontsize=8, color="#444444",
    )
    fig.subplots_adjust(bottom=0.25, top=0.85, left=0.07, right=0.98, wspace=0.28)
    return fig


def main():
    summary = summarize_all()
    summary.to_csv(OUT_TABLE, sep="\t", index=False, float_format="%.6f")
    figure = make_plot(summary)
    figure.savefig(OUT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(summary.groupby(["sample_class", "genotype"], observed=True)[
        "clone_weighted_mean_mutation_load_percent"
    ].agg(["count", "median"]).to_string())
    print(f"Wrote {OUT_TABLE}\nWrote {OUT_FIGURE}")


if __name__ == "__main__":
    main()
