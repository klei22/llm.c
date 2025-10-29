#!/usr/bin/env bash
# Train TinyStories twice (skip-top1 and cross-entropy) while running
# HellaSwag evaluations and inference sampling. After both runs finish the
# script collates their evaluation curves into a tab-separated file for easy
# comparison.
set -euo pipefail

make train_gpt2cu USE_CUDNN=1

TRAIN_BIN="dev/data/tinystories/TinyStories_train.bin"
VAL_BIN="dev/data/tinystories/TinyStories_val.bin"
HELLASWAG_BIN="dev/data/hellaswag/hellaswag_val.bin"

if [[ ! -f "${HELLASWAG_BIN}" ]]; then
    echo "Missing HellaSwag evaluation data at ${HELLASWAG_BIN}" >&2
    echo "Run \"python dev/data/hellaswag.py\" to prepare it before launching training." >&2
    exit 1
fi

VAL_INTERVAL=200
SAMPLE_INTERVAL=${VAL_INTERVAL}
GEN_TOKENS=256
GLOBAL_BATCH=262144
MICROBATCH=32
SEQ_LEN=1024
LR=3e-4

RESULTS_FILE="hellaswag_comparison.tsv"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

extract_eval_curve() {
    local log_file="$1"
    local out_file="$2"

    if [[ ! -f "${log_file}" ]]; then
        echo "Expected log file ${log_file} to exist" >&2
        exit 1
    fi

    awk -F'[ :]+' '/eval:/{printf "%s\t%s\n", $2, $4}' "${log_file}" | sort -n -k1,1 > "${out_file}"
}

run_training() {
    local loss_flag="$1"
    local output_dir="$2"
    shift 2

    echo "=== Training TinyStories with ${loss_flag} ==="
    ./train_gpt2cu \
        -i "${TRAIN_BIN}" \
        -j "${VAL_BIN}" \
        -o "${output_dir}" \
        -e gpt2:d12 \
        "$@" \
        -b "${MICROBATCH}" \
        -t "${SEQ_LEN}" \
        -d "${GLOBAL_BATCH}" \
        -l "${LR}" \
        -v "${VAL_INTERVAL}" \
        -s "${SAMPLE_INTERVAL}" \
        -g "${GEN_TOKENS}" \
        -h 1
}

run_training "skip-top1 loss" "log_tinystories_skip_top1" -L skip_top1
run_training "cross-entropy loss" "log_tinystories_cross_entropy" -L cross_entropy

extract_eval_curve "log_tinystories_skip_top1/main.log" "${TMP_DIR}/skip.tsv"
extract_eval_curve "log_tinystories_cross_entropy/main.log" "${TMP_DIR}/cross.tsv"

{
    echo -e "step\tskip_top1\tcross_entropy"
    join -a 1 -a 2 -e "NA" -o '0 1.2 2.2' -t $'\t' "${TMP_DIR}/skip.tsv" "${TMP_DIR}/cross.tsv"
} > "${RESULTS_FILE}"

echo "HellaSwag comparison written to ${RESULTS_FILE}"
