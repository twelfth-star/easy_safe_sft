#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
VLLM_ENV="${VLLM_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${VLLM_ENV}" ]]; then
  source "${VLLM_ENV}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/eval_checkpoints.py" \
  --config-path "${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft/eval.yaml" \
  --prompt-config "${PROJECT_DIR}/prompts/ethos/sft_reasoning.yaml" \
  --eval-concurrency "${EVAL_CONCURRENCY:-32}" \
  --vllm-tp-size "${VLLM_TP_SIZE:-2}" \
  --vllm-gpu-util "${VLLM_GPU_UTIL:-0.9}" \
  --vllm-maxlen "${VLLM_MAXLEN:-32768}" \
  --has-reasoning \
  --include-base-model
