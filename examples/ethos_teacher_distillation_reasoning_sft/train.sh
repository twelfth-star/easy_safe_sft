#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-easy_safe_sft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

cd /workspace/easy_safe_sft/examples/ethos_teacher_distillation_reasoning_sft

ROOT_DIR="/workspace/easy_safe_sft"
LF_ROOT="/app"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/train.py" \
  --llamafactory-root "${LF_ROOT}" \
  --config-path "${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft/train.yaml"
