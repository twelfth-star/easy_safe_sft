#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASK_CONFIG="${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft/task.yaml"
TEACHER_CONFIG="${ROOT_DIR}/examples/ethos_teacher_distillation_reasoning_sft/teacher.yaml"

TRAIN_RAW="${ROOT_DIR}/data/raw/ethos/ethos_train.jsonl"
VALID_RAW="${ROOT_DIR}/data/raw/ethos/ethos_val.jsonl"
TEST_RAW="${ROOT_DIR}/data/raw/ethos/ethos_test.jsonl"

TRACE_OUTPUT="${ROOT_DIR}/data/distilled/ethos_teacher_distillation_reasoning_sft/train_trace.jsonl"
LF_DATA_DIR="${ROOT_DIR}/data/lf_ready/ethos_teacher_distillation_reasoning_sft"

"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/distill.py" \
  --task-config "${TASK_CONFIG}" \
  --teacher-config "${TEACHER_CONFIG}" \
  --input "${TRAIN_RAW}" \
  --output "${TRACE_OUTPUT}"

"${PYTHON_BIN}" "${ROOT_DIR}/easy_safe_sft/build_dataset.py" \
  --task-config "${TASK_CONFIG}" \
  --train-input "${TRACE_OUTPUT}" \
  --train-source trace \
  --valid-input "${VALID_RAW}" \
  --test-input "${TEST_RAW}" \
  --output-dir "${LF_DATA_DIR}" \
  --student-template qwen3_nothink \
  --distill-output-style reasoning_text
