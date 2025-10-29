#!/usr/bin/env bash
# Train TinyStories twice: once with skip-top1 loss and once with standard cross-entropy.
# Each run enables HellaSwag evaluation and periodic inference samples so their
# validation curves and generations can be compared side-by-side.
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

run_training() {
    local loss_name="$1"
    local output_dir="$2"
    shift 2

    echo "=== Training TinyStories with ${loss_name} loss ==="
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

run_training "skip-top1" "log_tinystories_skip_top1" -L skip_top1
run_training "cross-entropy" "log_tinystories_cross_entropy" -L cross_entropy
