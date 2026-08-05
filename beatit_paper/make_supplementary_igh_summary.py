#!/usr/bin/env python3
"""Create the compact all-sample productive IgH supplementary table."""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTIVE_DIR = SCRIPT_DIR / "IGH_productive"
ORIGINAL_MIXCR_DIR = Path("/scratch/c9881013/felix/mixcr")
OUTPUT_PATH = SCRIPT_DIR / "supplementary_table_igh_summary.tsv"
NOTES_PATH = SCRIPT_DIR / "supplementary_table_igh_summary_notes.txt"

# This pragmatic reporting threshold is separated by a clear gap in this data:
# eight of nine IgM-negative tumors have <=266 total IgH reads, whereas F6 has
# 5,934. It is a flag for interpretation, not a universal biological cutoff.
LOW_IGH_READ_THRESHOLD = 1_000

SAMPLE_METADATA = {
    **{
        f"A{i}": ("EµMyc", f"premalignant IgM{'+' if i % 2 else '-'}")
        for i in range(1, 13)
    },
    **{
        f"C{i}": ("EµMyc/Tet2KO", f"premalignant IgM{'+' if i % 2 else '-'}")
        for i in (1, 2, 5, 6, 7, 8, 9, 10, 11, 12)
    },
    "D1": ("EµMyc", "tumor IgM-"),
    "D2": ("EµMyc", "tumor IgM-"),
    "D3": ("EµMyc", "tumor IgM-"),
    "D4": ("EµMyc", "tumor IgM+"),
    "D5": ("EµMyc", "tumor mixed"),
    "D6": ("EµMyc", "tumor IgM-"),
    "D7": ("EµMyc", "tumor IgM-"),
    "D8": ("EµMyc", "tumor IgM+"),
    "D9": ("EµMyc", "tumor IgM+"),
    "F1": ("EµMyc/Tet2KO", "tumor IgM+"),
    "F2": ("EµMyc/Tet2KO", "tumor IgM-"),
    "F3": ("EµMyc/Tet2KO", "tumor mixed"),
    "F4": ("EµMyc/Tet2KO", "tumor IgM-"),
    "F5": ("EµMyc/Tet2KO", "tumor IgM-"),
    "F6": ("EµMyc/Tet2KO", "tumor IgM-"),
    "F7": ("EµMyc/Tet2KO", "tumor IgM+"),
    "F8": ("EµMyc/Tet2KO", "tumor IgM+"),
    "F9": ("EµMyc/Tet2KO", "tumor IgM+"),
    "F10": ("EµMyc/Tet2KO", "tumor mixed"),
    "F11": ("EµMyc/Tet2KO", "tumor mixed"),
}


def natural_sample_key(sample: str) -> tuple[int, int]:
    """Sort samples by experimental series and numeric suffix."""
    series_order = {"A": 0, "C": 1, "D": 2, "F": 3}
    return series_order[sample[0]], int(sample[1:])


def confidence_for(total_igh_reads: int) -> str:
    """Assign one of two confidence labels from total IgH read support."""
    if total_igh_reads < LOW_IGH_READ_THRESHOLD:
        return "low IgH reads (<1000)"
    return "robust IgH signal"


def load_sample(sample: str, genotype: str, stage: str) -> dict:
    """Load all-IgH and productive-IgH clone exports for one sample."""
    productive_path = PRODUCTIVE_DIR / f"{sample}_productive_IGH.tsv"
    all_igh_path = ORIGINAL_MIXCR_DIR / sample / f"{sample}_clones_IGH.tsv"
    if not productive_path.is_file():
        raise FileNotFoundError(f"Missing productive IgH export: {productive_path}")
    if not all_igh_path.is_file():
        raise FileNotFoundError(f"Missing all-IgH export: {all_igh_path}")

    productive = pd.read_csv(productive_path, sep="\t")
    all_igh = pd.read_csv(all_igh_path, sep="\t")
    required = {"cloneId", "readCount", "nSeqCDR3", "aaSeqCDR3"}
    for path, clones in ((productive_path, productive), (all_igh_path, all_igh)):
        missing = required.difference(clones.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if clones.empty:
            raise ValueError(f"No IgH clonotypes in {path.name}")

    if not set(productive["cloneId"]).issubset(set(all_igh["cloneId"])):
        raise ValueError(f"Productive clone IDs are not a subset for {sample}")
    if productive["nSeqCDR3"].duplicated().any():
        raise ValueError(f"Duplicate nucleotide CDR3 clonotype in {productive_path.name}")
    if productive["aaSeqCDR3"].astype(str).str.contains("*", regex=False).any():
        raise ValueError(f"Stop codon in productive export {productive_path.name}")
    if (productive["nSeqCDR3"].astype(str).str.len() % 3 != 0).any():
        raise ValueError(f"Out-of-frame CDR3 in productive export {productive_path.name}")

    all_read_counts = pd.to_numeric(all_igh["readCount"], errors="raise")
    productive_read_counts = pd.to_numeric(
        productive["readCount"], errors="raise"
    ).sort_values(ascending=False)
    total_igh_reads = int(all_read_counts.sum())
    total_productive_reads = float(productive_read_counts.sum())
    if total_productive_reads <= 0:
        raise ValueError(f"No productive IgH reads in {productive_path.name}")

    disease_stage, cell_fraction = stage.split(maxsplit=1)
    disease_stage = "malignant" if disease_stage == "tumor" else disease_stage
    cell_fraction = cell_fraction.replace("IgM+", "IgM⁺").replace("IgM-", "IgM⁻")

    return {
        "Sample ID": sample,
        "Genotype": genotype,
        "Stage": disease_stage,
        "Cell fraction": cell_fraction,
        "Total IgH reads": total_igh_reads,
        "Productive IgH clonotypes": int(len(productive)),
        "Dominant clone frequency (%)": (
            100.0 * productive_read_counts.iloc[0] / total_productive_reads
        ),
        "Top 5 clone frequency (%)": (
            100.0 * productive_read_counts.iloc[:5].sum() / total_productive_reads
        ),
        "Confidence": confidence_for(total_igh_reads),
    }


def main() -> None:
    discovered_samples = {
        path.name.removesuffix("_productive_IGH.tsv")
        for path in PRODUCTIVE_DIR.glob("*_productive_IGH.tsv")
    }
    expected_samples = set(SAMPLE_METADATA)
    if discovered_samples != expected_samples:
        raise ValueError(
            "Productive sample set differs from metadata: "
            f"missing={sorted(expected_samples - discovered_samples)}, "
            f"unexpected={sorted(discovered_samples - expected_samples)}"
        )

    records = []
    for sample in sorted(SAMPLE_METADATA, key=natural_sample_key):
        genotype, stage = SAMPLE_METADATA[sample]
        records.append(load_sample(sample, genotype, stage))

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(OUTPUT_PATH, sep="\t", index=False, float_format="%.2f")

    notes = f"""Supplementary IgH summary: calculation notes

- Total IgH reads is the sum of MiXCR readCount across all IgH clonotypes,
  including productive and nonproductive clonotypes.
- Productive IgH clonotypes is the number of nucleotide CDR3 clonotypes in the
  MiXCR export generated with --chains IGH --export-productive-clones-only.
- Dominant clone and top-5 frequencies are recalculated within productive IgH
  clonotypes only: productive clone readCount / all productive IgH readCount.
- MiXCR readCount measures bulk RNA-seq read/transcript support. There were no
  UMIs, so these frequencies are not direct B-cell frequencies.
- Confidence has two levels based on total IgH read support: "robust IgH
  signal" for >= {LOW_IGH_READ_THRESHOLD:,} reads and "low IgH reads (<1000)"
  below this threshold. This is a pragmatic reporting cutoff, not a universal
  biological threshold.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Wrote {NOTES_PATH}")


if __name__ == "__main__":
    main()
