#!/bin/bash

# Options:
#   -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)
#   -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)
#   -e <string> input .bin filename or descriptor, see code comments as docs. (default = gpt2_124M_bf16.bin)
#   -o <string> output log dir (default = NULL, no logging)
#   -lg <int>   log gpu info every x steps (default = -1; disabled)
#   -n <int>    write optimization checkpoints every how many steps? (default 0, don't)
#   -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)
#   -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.
#   -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)
#   -b <int>    (per-GPU, micro) batch size B (default = 4)
#   -t <int>    sequence length T (default = 1024)
#   -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation
#   -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)
#   -k <string> learning rate scheduler (default = cosine)
#   -l <float>  learning rate (default = 3e-4f)
#   -u <int>    learning rate warmup iterations (default = 0, no warmup)
#   -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))
#   -c <float>  weight decay (default = 0.0f)
#   -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)
#   -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)
#   -v <int>    val_loss_every, how often we evaluate val loss (default = 20)
#   -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)
#   -s <int>    sample_every, how often we inference the model (default = 20)
#   -g <int>    genT, how many steps of inference we do (default = 64)
#   -h <int>    hellaswag eval run? (default = 0)
#   -a <int>    overfit a single batch? 0/1. useful for debugging
#   -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)
#   -w <int>    keep f32 copy of weights for the optimizer? (default: 1)
#   -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)
#   -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)
#   -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln
#   -pn <int>    num_processes (default = 1)
#   -pr <int>    process_rank (default = 0)
#   -pg <int>    gpus_per_node (default = 8)
#   -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)
#   -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)
#   -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)


  # -sg 1.0  \
  # -sl 1.0 \
  # -pm tcp \
  # -pm fs \

  # -i "dev/data/edu_fineweb100B/edu_fineweb_train_000001.bin" \
  # -j dev/data/edu_fineweb100B/edu_fineweb_val_000000.bin \

./train_gpt2cu \
  -i dev/data/tinystories/TinyStories_train.bin \
  -j dev/data/tinystories/TinyStories_val.bin \
  -sg 0.3  \
  -sl 0.3 \
  -e gpt2:d6 \
  -b 4 \
  -x 100000 \
  -c 0.1 \
  -t 256 \
  -v 200\
  -m 200\
  -s 200\
  -l "1.5e-3" \
  -q 10 \
  -ge 2 \
  -u 500 \
  -g 64
  # -r 2 \

