#!/usr/bin/env bash
# Train GPT-2 from scratch on TinyStories using the skip-top1 loss.
set -euo pipefail

make train_gpt2cu USE_CUDNN=1

TRAIN_BIN="dev/data/tinystories/TinyStories_train.bin"
VAL_BIN="dev/data/tinystories/TinyStories_val.bin"
OUT_DIR="log_tinystories_skip_top1"

./train_gpt2cu \
    -i "$TRAIN_BIN" \
    -j "$VAL_BIN" \
    -o "$OUT_DIR" \
    -e gpt2:d12 \
    -L skip_top1 \
    -b 32 \
    -t 1024 \
    -d 262144 \
    -l 3e-4 \
    -v 200 \
    -s 2000 \
    -g 256
