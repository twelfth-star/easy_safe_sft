#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROMPT_CONFIG="${PROJECT_DIR}/prompts/ethos/sft_reasoning.yaml"
TRAIN_RAW="${ROOT_DIR}/data/raw/ethos/ethos_train.jsonl"
VALID_RAW="${ROOT_DIR}/data/raw/ethos/ethos_val.jsonl"
TEST_RAW="${ROOT_DIR}/data/raw/ethos/ethos_test.jsonl"
LF_DATA_DIR="${ROOT_DIR}/data/lf_ready/ethos_sft"

"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/build_dataset.py" \
  --prompt-config "${PROMPT_CONFIG}" \
  --train-input "${TRAIN_RAW}" \
  --train-source raw \
  --valid-input "${VALID_RAW}" \
  --test-input "${TEST_RAW}" \
  --output-dir "${LF_DATA_DIR}" \
  --student-template qwen3_nothink \
  --plain-output-style final_only
