#!/usr/bin/env bash
# Train GPT-2 from scratch on TinyStories using the skip-top1 loss.
set -euo pipefail

make train_gpt2cu USE_CUDNN=1

TRAIN_BIN="dev/data/tinystories/TinyStories_train.bin"
VAL_BIN="dev/data/tinystories/TinyStories_val.bin"
OUT_DIR="log_tinystories_skip_top1"

<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
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
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
VAL_INTERVAL=200
SAMPLE_INTERVAL=${VAL_INTERVAL}
GEN_TOKENS=256
GLOBAL_BATCH=262144
MICROBATCH=32
SEQ_LEN=1024
LR=3e-4

./train_gpt2cu \
    -i "${TRAIN_BIN}" \
    -j "${VAL_BIN}" \
    -o "${OUT_DIR}" \
    -e gpt2:d12 \
    -L skip_top1 \
    -b "${MICROBATCH}" \
    -t "${SEQ_LEN}" \
    -d "${GLOBAL_BATCH}" \
    -l "${LR}" \
    -v "${VAL_INTERVAL}" \
    -s "${SAMPLE_INTERVAL}" \
    -g "${GEN_TOKENS}"
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
