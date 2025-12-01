#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Usage:
#   ./run_train.sh [--debug] [--ref] [additional args...]
#
# Modes:
#   ./run_train.sh                  Normal training with your implementation (my_train.py)
#   ./run_train.sh --ref            Normal training with reference implementation (train.py)
#   ./run_train.sh --debug          Debug mode with debugpy-run (attach debugger)
#   ./run_train.sh --debug --ref    Debug mode with reference implementation
#
# Environment variables:
#   NGPU=<n>                        Number of GPUs (default: 8)
#   LOG_RANK=<rank>                 Rank(s) to log (default: 0)
#   CONFIG_FILE=<path>              Config file path (default: ./torchtitan/models/llama3/train_configs/debug_model.toml)
#   COMM_MODE="fake_backend"        Dry run for config validation without GPU
#   COMM_MODE="local_tensor"        Local tensor debugging mode
#
# Examples:
#   ./run_train.sh                              # Normal training
#   ./run_train.sh --debug                      # With debugpy
#   NGPU=4 ./run_train.sh                       # Use 4 GPUs
#   COMM_MODE="fake_backend" ./run_train.sh    # Validate config without GPU
#

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
# COMM_MODE="fake_backend" ./run_train.sh  # for config validation without GPU
# COMM_MODE="local_tensor" ./run_train.sh  # for local tensor debugging mode
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
# COMM_MODE options: "fake_backend" (dry run), "local_tensor" (debug mode), or empty for normal training
COMM_MODE=${COMM_MODE:-""}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# Check for --debug flag
DEBUG_MODE=false
USE_REF=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--debug" ]; then
        DEBUG_MODE=true
    elif [ "$arg" = "--ref" ]; then
        USE_REF=true
    else
        ARGS+=("$arg")
    fi
done

# Select training file based on --ref flag
if [ "$USE_REF" = true ]; then
    TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
    echo "Using reference implementation (train.py)"
    LOG_FILE=${LOG_FILE:-"log_train.txt"}
else
    TRAIN_FILE=${TRAIN_FILE:-"torchtitan.my_train"}
    echo "Using your implementation (my_train.py)"
    LOG_FILE=${LOG_FILE:-"log_my_train.txt"}
fi

if [ "$DEBUG_MODE" = true ]; then
    # Debug mode with debugpy
    echo "Running in debug mode with debugpy"
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    debugpy-run -m torch.distributed.run -- --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "${ARGS[@]}" 2>&1 | tee ${LOG_FILE}
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "${ARGS[@]}" 2>&1 | tee ${LOG_FILE}
fi
