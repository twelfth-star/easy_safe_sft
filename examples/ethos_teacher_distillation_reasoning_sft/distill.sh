#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Use sft_reasoning_nothink.yaml for non-thinking mode, sft_reasoning.yaml for thinking mode
PROMPT_CONFIG="${PROJECT_DIR}/prompts/ethos/sft_reasoning_nothink.yaml"
TEACHER_CONFIG="${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft/teacher.yaml"
RULES_DIR="${PROJECT_DIR}/rules/ethos"
DATASET_ROOT="${PROJECT_DIR}/datasets"

TRACE_OUTPUT_DIR="${ROOT_DIR}/data/distilled/ethos_teacher_distillation_reasoning_sft"
LF_DATA_DIR="${ROOT_DIR}/data/lf_ready/ethos_teacher_distillation_reasoning_sft"

# Step 1: Teacher distillation
"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/distill.py" \
  --prompt-config "${PROMPT_CONFIG}" \
  --teacher-config "${TEACHER_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name ethos \
  --dataset-version v1 \
  --split-policy policy_v1 \
  --rules-dir "${RULES_DIR}" \
  --output-dir "${TRACE_OUTPUT_DIR}"

# Step 2: Build LlamaFactory dataset from accepted traces
"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/build_dataset.py" \
  --prompt-config "${PROMPT_CONFIG}" \
  --train-input "${TRACE_OUTPUT_DIR}/accepted_samples.jsonl" \
  --train-source trace \
  --output-dir "${LF_DATA_DIR}" \
  --student-template qwen3_nothink \
  --distill-output-style reasoning_label
