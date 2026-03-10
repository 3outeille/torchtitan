#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
# COMM_MODE="fake_backend" ./run_train.sh  # for config validation without GPU
# COMM_MODE="local_tensor" ./run_train.sh  # for local tensor debugging mode
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

# Option to switch between debug and train
MODE=${MODE:-"debug"}  # Set MODE=debug or MODE=train

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
# COMM_MODE options: "fake_backend" (dry run), "local_tensor" (debug mode), or empty for normal training
COMM_MODE=${COMM_MODE:-""}

if [ "$MODE" = "debug" ]; then
    PYTHON_CMD="debugpy-run -m torch.distributed.run --"
else
    PYTHON_CMD="torchrun"
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" --job.config_file "${CONFIG_FILE}" "$@" --comm.mode=${COMM_MODE} --training.steps=1
else
    # Normal training with torchrun
    # PYTORCH_ALLOC_CONF="expandable_segments:True" \
    # TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    # torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    # --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    # -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"

    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    debugpy-run -m torch.distributed.run -- --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
fi
