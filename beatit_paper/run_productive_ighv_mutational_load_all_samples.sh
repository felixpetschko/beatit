#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="/scratch/c9881013/felix/mixcr"
OUTPUT_ROOT="${SCRIPT_DIR}/mutational_load"
MIXCR_ENV="/scratch/c9881013/.conda_envs/mixcr_env"
SUMMARY="${SCRIPT_DIR}/supplementary_table_igh_summary.tsv"

export PATH="${MIXCR_ENV}/bin:${PATH}"
mkdir -p -- "${OUTPUT_ROOT}"

tail -n +2 "${SUMMARY}" | cut -f1 | while read -r sample; do
    source_vdjca="${SOURCE_ROOT}/${sample}/${sample}.vdjca"
    sample_dir="${OUTPUT_ROOT}/${sample}"
    output_tsv="${sample_dir}/${sample}_productive_IGHV_regions.tsv"

    if [[ -s "${output_tsv}" ]]; then
        echo "[${sample}] regional export already present"
        continue
    fi
    if [[ ! -f "${source_vdjca}" ]]; then
        echo "Missing source alignment: ${source_vdjca}" >&2
        exit 1
    fi
    mkdir -p -- "${sample_dir}"

    mixcr assemble --write-alignments --force-overwrite \
        --report "${sample_dir}/${sample}_assemble_clna_report.txt" \
        "${source_vdjca}" "${sample_dir}/${sample}.clna"
    mixcr assembleContigs --assemble-longest-contigs --threads 8 --force-overwrite \
        --report "${sample_dir}/${sample}_assemble_contigs_report.txt" \
        "${sample_dir}/${sample}.clna" "${sample_dir}/${sample}_contigs.clns"
    mixcr exportClones --chains IGH --export-productive-clones-only --force-overwrite \
        -cloneId -readCount -readFraction -vHit \
        -nFeature CDR3 -aaFeature CDR3 \
        -nLength FR1 -nMutationsCount FR1 substitutions \
        -nLength CDR1 -nMutationsCount CDR1 substitutions \
        -nLength FR2 -nMutationsCount FR2 substitutions \
        -nLength CDR2 -nMutationsCount CDR2 substitutions \
        -nLength FR3 -nMutationsCount FR3 substitutions \
        "${sample_dir}/${sample}_contigs.clns" "${output_tsv}"
done

