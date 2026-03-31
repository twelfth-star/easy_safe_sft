#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-easy_safe_sft}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft"

LF_ROOT="${LF_ROOT:-/app}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/train.py" \
  --llamafactory-root "${LF_ROOT}" \
  --config-path "${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft/train.yaml"
