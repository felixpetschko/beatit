#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="/scratch/c9881013/felix/mixcr"
OUTPUT_ROOT="${SCRIPT_DIR}/mutational_load"
MIXCR_ENV="/scratch/c9881013/.conda_envs/mixcr_env"
SAMPLES=(A1 A3 A5 A7 A9 A11 C1 C5 C7 C9 C11)

export PATH="${MIXCR_ENV}/bin:${PATH}"
mkdir -p -- "${OUTPUT_ROOT}"

for sample in "${SAMPLES[@]}"; do
    source_vdjca="${SOURCE_ROOT}/${sample}/${sample}.vdjca"
    sample_dir="${OUTPUT_ROOT}/${sample}"
    if [[ ! -f "${source_vdjca}" ]]; then
        echo "Missing source alignment: ${source_vdjca}" >&2
        exit 1
    fi
    mkdir -p -- "${sample_dir}"

    echo "[${sample}] assembling clones with retained alignments"
    mixcr assemble \
        --write-alignments \
        --report "${sample_dir}/${sample}_assemble_clna_report.txt" \
        --force-overwrite \
        "${source_vdjca}" \
        "${sample_dir}/${sample}.clna"

    echo "[${sample}] assembling longest clone contigs"
    mixcr assembleContigs \
        --assemble-longest-contigs \
        --threads 8 \
        --report "${sample_dir}/${sample}_assemble_contigs_report.txt" \
        --force-overwrite \
        "${sample_dir}/${sample}.clna" \
        "${sample_dir}/${sample}_contigs.clns"

    echo "[${sample}] exporting productive IgH regional mutation counts"
    mixcr exportClones \
        --chains IGH \
        --export-productive-clones-only \
        --force-overwrite \
        -cloneId -readCount -readFraction -vHit \
        -nFeature CDR3 -aaFeature CDR3 \
        -nLength FR1 -nMutationsCount FR1 substitutions \
        -nLength CDR1 -nMutationsCount CDR1 substitutions \
        -nLength FR2 -nMutationsCount FR2 substitutions \
        -nLength CDR2 -nMutationsCount CDR2 substitutions \
        -nLength FR3 -nMutationsCount FR3 substitutions \
        "${sample_dir}/${sample}_contigs.clns" \
        "${sample_dir}/${sample}_productive_IGHV_regions.tsv"
done

echo "Finished productive IgH V-region contig assembly for ${#SAMPLES[@]} samples."
