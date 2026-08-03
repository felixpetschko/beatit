#!/usr/bin/env python3
"""Render the compact all-sample IgH supplementary table as a readable PNG."""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "supplementary_table_igh_summary.tsv"
OUTPUT_DIR = SCRIPT_DIR / "figures"
OUTPUT_PATH = OUTPUT_DIR / "supplementary_table_igh_summary.png"

DISPLAY_HEADERS = [
    "Sample\nID",
    "Genotype",
    "Cell fraction /\nstage",
    "Total IgH\nreads",
    "Productive IgH\nclonotypes",
    "Dominant clone\nfrequency (%)",
    "Top 5 clone\nfrequency (%)",
    "Comment",
]

# Relative widths optimized for the long stage and comment fields.
COLUMN_WIDTHS = [0.055, 0.105, 0.135, 0.085, 0.105, 0.115, 0.105, 0.295]


def display_rows(summary: pd.DataFrame) -> list[list[str]]:
    """Format values and wrap long comments for table rendering."""
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                str(row["Sample ID"]),
                str(row["Genotype"]),
                str(row["Cell fraction / stage"]),
                f'{int(row["Total IgH reads"]):,}',
                f'{int(row["Productive IgH clonotypes"]):,}',
                f'{float(row["Dominant clone frequency (%)"]):.2f}',
                f'{float(row["Top 5 clone frequency (%)"]):.2f}',
                textwrap.fill(str(row["Comment"]), width=47),
            ]
        )
    return rows


def row_color(comment: str, row_index: int) -> str:
    """Highlight confidence flags while retaining alternating row shading."""
    if "low-confidence" in comment:
        return "#FCE8E6"
    if "exploratory only" in comment or "interpret cautiously" in comment:
        return "#FFF4D6"
    return "#FFFFFF" if row_index % 2 == 0 else "#F4F6F8"


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing summary table: {INPUT_PATH}")

    summary = pd.read_csv(INPUT_PATH, sep="\t")
    if len(summary) != 42 or summary["Sample ID"].nunique() != 42:
        raise ValueError("Expected exactly 42 unique samples")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(26, 31))
    ax.axis("off")
    ax.set_title(
        "Supplementary Table: Productive IgH repertoire summary",
        fontsize=28,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.5,
        0.982,
        (
            "Clone frequencies are calculated among productive IgH reads. "
            "Bulk RNA-seq without UMIs; exploratory samples should not be overinterpreted."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        color="#333333",
    )

    table = ax.table(
        cellText=display_rows(summary),
        colLabels=DISPLAY_HEADERS,
        colWidths=COLUMN_WIDTHS,
        cellLoc="center",
        colLoc="center",
        bbox=[0.005, 0.01, 0.99, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.0)

    for column_index in range(len(DISPLAY_HEADERS)):
        header = table[(0, column_index)]
        header.set_facecolor("#26374A")
        header.set_text_props(color="white", weight="bold", fontsize=12.0)
        header.set_edgecolor("white")
        header.set_linewidth(0.8)
        header.set_height(0.032)

    for row_index, (_, row) in enumerate(summary.iterrows(), start=1):
        background = row_color(str(row["Comment"]), row_index)
        for column_index in range(len(DISPLAY_HEADERS)):
            cell = table[(row_index, column_index)]
            cell.set_facecolor(background)
            cell.set_edgecolor("#B8C0C8")
            cell.set_linewidth(0.45)
            cell.set_height(0.0215)
            if column_index in (0, 1):
                cell.set_text_props(weight="bold")
            if column_index in (2, 7):
                cell.get_text().set_ha("left")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
