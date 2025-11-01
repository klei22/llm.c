#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/run_embed_rmsnorm_comparison.sh [options]

Runs training with and without the optional embedding RMSNorm and collects
validation loss and HellaSwag summaries for later comparison.

Options:
  -o, --output-dir DIR   Directory to store comparison logs (default: log_embed_rmsnorm_comparison)
      --dataset NAME     Dataset to train on: tinyshakespeare, tinystories, fineweb (default: tinyshakespeare)
      --max-steps N      Number of training steps to run (passes -x to train_gpt2cu, default: 50)
      --hellaswag        Force-enable HellaSwag evaluation (default behaviour)
      --no-hellaswag     Disable HellaSwag evaluation
  -h, --help             Show this help message and exit

Environment variables:
  COMMON_ARGS  Additional arguments appended to each train_gpt2cu invocation after
               the script-managed defaults (can override them).
  EXTRA_ARGS   Further arguments appended to each train_gpt2cu invocation (last wins).
USAGE
}

err() {
    echo "Error: $*" >&2
}

require_file() {
    local path="$1"
    local hint="$2"
    if [[ ! -f "$path" ]]; then
        err "$hint not found at $path"
        exit 1
    fi
}

resolve_fineweb_patterns() {
    local candidates=(
        "dev/data/fineweb10B/fineweb_train_*.bin"
        "dev/data/fineweb100B/fineweb_train_*.bin"
        "dev/data/edu_fineweb10B/edu_fineweb_train_*.bin"
        "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin"
    )
    for pattern in "${candidates[@]}"; do
        if compgen -G "$pattern" > /dev/null; then
            local val_pattern="${pattern/_train_/_val_}"
            if ! compgen -G "$val_pattern" > /dev/null; then
                echo "Warning: found FineWeb training shards ($pattern) but no validation shards ($val_pattern)" >&2
                continue
            fi
            TRAIN_PATTERN="$pattern"
            VAL_PATTERN="$val_pattern"
            DATASET_DESC="fineweb ($(basename "$(dirname "$pattern")"))"
            return
        fi
    done
    err "Could not find FineWeb dataset shards. Generate them via: python dev/data/fineweb.py"
    exit 1
}

resolve_dataset_patterns() {
    case "$DATASET" in
        tinyshakespeare)
            TRAIN_PATTERN="dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
            VAL_PATTERN="dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
            require_file "$TRAIN_PATTERN" "Tiny Shakespeare training data"
            require_file "$VAL_PATTERN" "Tiny Shakespeare validation data"
            DATASET_DESC="tinyshakespeare"
            ;;
        tinystories)
            TRAIN_PATTERN="dev/data/tinystories/TinyStories_train.bin"
            VAL_PATTERN="dev/data/tinystories/TinyStories_val.bin"
            require_file "$TRAIN_PATTERN" "TinyStories training data"
            require_file "$VAL_PATTERN" "TinyStories validation data"
            DATASET_DESC="tinystories"
            ;;
        fineweb)
            resolve_fineweb_patterns
            ;;
        *)
            err "Unknown dataset: $DATASET"
            exit 1
            ;;
    esac
}

OUT_ROOT="log_embed_rmsnorm_comparison"
DATASET="tinyshakespeare"
MAX_STEPS=50
HELLASWAG=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            [[ $# -ge 2 ]] || { err "Missing value for $1"; exit 1; }
            OUT_ROOT="$2"
            shift 2
            ;;
        --dataset)
            [[ $# -ge 2 ]] || { err "Missing value for $1"; exit 1; }
            DATASET="${2,,}"
            shift 2
            ;;
        --max-steps)
            [[ $# -ge 2 ]] || { err "Missing value for $1"; exit 1; }
            MAX_STEPS="$2"
            shift 2
            ;;
        --hellaswag)
            HELLASWAG=1
            shift
            ;;
        --no-hellaswag)
            HELLASWAG=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$MAX_STEPS" =~ ^-?[0-9]+$ ]]; then
    err "--max-steps must be an integer"
    exit 1
fi

resolve_dataset_patterns

HELLASWAG_PATH="dev/data/hellaswag/hellaswag_val.bin"
if (( HELLASWAG )); then
    require_file "$HELLASWAG_PATH" "HellaSwag validation data"
fi

mkdir -p "$OUT_ROOT"

echo "Output directory : $OUT_ROOT"
echo "Dataset          : $DATASET_DESC"
echo "Max steps        : $MAX_STEPS"
if (( HELLASWAG )); then
    hellaswag_status="enabled"
else
    hellaswag_status="disabled"
fi

echo "HellaSwag eval   : ${hellaswag_status}"

declare -a COMMON_ARGS_ARR
COMMON_ARGS_ARR=(
    -i "$TRAIN_PATTERN"
    -j "$VAL_PATTERN"
    -b 4
    -t 1024
    -v 10
    -m 10
    -s 0
    -x "$MAX_STEPS"
    -e d12
    -h "$HELLASWAG"
)

if [[ -n "${COMMON_ARGS:-}" ]]; then
    read -r -a USER_COMMON_ARGS <<< "${COMMON_ARGS}"
    COMMON_ARGS_ARR+=("${USER_COMMON_ARGS[@]}")
fi

declare -a EXTRA_ARGS_ARR=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_USER_ARGS <<< "${EXTRA_ARGS}"
    EXTRA_ARGS_ARR=("${EXTRA_USER_ARGS[@]}")
fi

echo "Building train_gpt2cu binary..."
make train_gpt2cu USE_CUDNN=1

run_mode() {
    local mode="$1"
    local rmsnorm_flag="$2"
    local run_dir="${OUT_ROOT}/${mode}"
    local log_file="${run_dir}/train.log"

    mkdir -p "$run_dir"

    echo "Running ${mode} training (embed RMSNorm flag: ${rmsnorm_flag})"
    local cmd=(./train_gpt2cu "${COMMON_ARGS_ARR[@]}" -o "$run_dir" -nr "$rmsnorm_flag")
    cmd+=("${EXTRA_ARGS_ARR[@]}")

    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}" | tee "$log_file"

    grep "val loss" "$log_file" > "${run_dir}/val_loss.log" || true
    grep "HellaSwag" "$log_file" > "${run_dir}/hellaswag.log" || true
}

run_mode "baseline" 0
run_mode "embed_rmsnorm" 1
