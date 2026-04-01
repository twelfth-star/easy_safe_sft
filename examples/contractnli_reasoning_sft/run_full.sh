#!/usr/bin/env bash
# =============================================================
#  ContractNLI reasoning distillation SFT — full run
#  Aligned with Tinker's run_reasoning_contractnli.sh config
#
#  Teacher: Claude Haiku 4.5 (k=4, workers=10, 2000 examples)
#  Student: Qwen3-4B (LoRA rank=32, 1 epoch, batch=32)
#  Dataset: contractNLI v1, policy_sft2000_rlfull_eval300_600
#
#  Steps:
#    1. Teacher distillation (API calls, ~20 min for 2000 examples)
#    2. Build LlamaFactory dataset
#    3. Training (2xH100 DDP)
#    4. Evaluation (2xH100 TP, vLLM)
# =============================================================
set -euo pipefail

# Activate conda for Steps 1-2
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate research 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

# ── Configuration ─────────────────────────────
PROMPT_CONFIG="${PROJECT_DIR}/prompts/contractNLI/sft_reasoning_nothink.yaml"
TEACHER_CONFIG="${ROOT_DIR}/examples/contractnli_reasoning_sft/teacher_full.yaml"
TRAIN_CONFIG="${ROOT_DIR}/examples/contractnli_reasoning_sft/train_full.yaml"
RULES_DIR="${PROJECT_DIR}/rules/contractNLI"
DATASET_ROOT="${PROJECT_DIR}/datasets"
DATASET_NAME="contractNLI"
DATASET_VERSION="v1"
SPLIT_POLICY="policy_sft2000_rlfull_eval300_600"
PHASE_NAME="phase2"

TRACE_DIR="${ROOT_DIR}/data/distilled/contractnli_reasoning_sft_full"
LF_DATA_DIR="${ROOT_DIR}/data/lf_ready/contractnli_reasoning_sft_full"
OUTPUT_DIR="${ROOT_DIR}/outputs/contractnli_reasoning_sft_full"

# Apptainer config
SIF="${PROJECT_DIR}/llamafactory_latest.sif"
SFT_PKG="${PROJECT_DIR}/sft_site_packages"
VLLM_PKG="${PROJECT_DIR}/vllm_site_packages"
HF_CACHE="/ocean/projects/cis250290p/tzhou6/hf_cache"
TMP_DIR="/ocean/projects/cis250290p/tzhou6/tmp"

# API key (default: ANTHROPIC_API_KEY from .env)
API_KEY_ENV="${API_KEY_ENV:-}"

echo "=========================================="
echo " ContractNLI Reasoning SFT — Full Run"
echo " Teacher: claude-haiku-4-5"
echo " Student: Qwen3-4B (LoRA r=32)"
echo " Dataset: ${DATASET_NAME} ${SPLIT_POLICY}"
echo "=========================================="
echo ""

# ── Step 1: Teacher distillation ──────────────
echo ">>> [1/4] Teacher distillation ..."
echo ""

cd "${ROOT_DIR}"

API_KEY_ARG=""
if [[ -n "${API_KEY_ENV}" ]]; then
  API_KEY_ARG="--api-key-env ${API_KEY_ENV}"
fi

python easy_safe_sft/distill.py \
  --prompt-config "${PROMPT_CONFIG}" \
  --teacher-config "${TEACHER_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-version "${DATASET_VERSION}" \
  --split-policy "${SPLIT_POLICY}" \
  --phase-name "${PHASE_NAME}" \
  --rules-dir "${RULES_DIR}" \
  --output-dir "${TRACE_DIR}" \
  ${API_KEY_ARG}

ACCEPTED=$(wc -l < "${TRACE_DIR}/accepted_samples.jsonl")
echo ""
echo ">>> Teacher distillation done: ${ACCEPTED} accepted samples"
cat "${TRACE_DIR}/teacher_summary.json"
echo ""

# ── Step 2: Build LlamaFactory dataset ────────
echo ">>> [2/4] Building LlamaFactory dataset ..."
echo ""

# Prepare val/test data from canonical dataset
python -c "
from easy_safe_sft.data_loading import resolve_canonical_dataset_spec, load_canonical_splits
from easy_safe_sft.utils import write_jsonl
from pathlib import Path

spec = resolve_canonical_dataset_spec(
    dataset_root=Path('${DATASET_ROOT}'),
    dataset_name='${DATASET_NAME}',
    dataset_version='${DATASET_VERSION}',
    split_policy='${SPLIT_POLICY}',
)
splits = load_canonical_splits(spec, phase_name='${PHASE_NAME}')

for split_name in ('val', 'test'):
    rows = []
    for _, r in splits[split_name].iterrows():
        rows.append({
            'id': r['example_id'],
            'example_id': r['example_id'],
            'split': split_name,
            'text': r['text'],
            'answer': str(int(r['label'])),
            'label': int(r['label']),
            'fields': {'text': r['text']},
        })
    write_jsonl(f'${TRACE_DIR}/{split_name}_raw.jsonl', rows)
    print(f'{split_name}: {len(rows)} rows')
"

python easy_safe_sft/build_dataset.py \
  --prompt-config "${PROMPT_CONFIG}" \
  --train-input "${TRACE_DIR}/accepted_samples.jsonl" \
  --train-source trace \
  --valid-input "${TRACE_DIR}/val_raw.jsonl" \
  --test-input "${TRACE_DIR}/test_raw.jsonl" \
  --output-dir "${LF_DATA_DIR}" \
  --student-template qwen3_nothink \
  --distill-output-style reasoning_label

echo ">>> Dataset build done"
echo ""

# ── Step 3: Training ──────────────────────────
echo ">>> [3/4] Training (Qwen3-4B, LoRA r=32, 2xH100 DDP) ..."
echo ""

apptainer exec --nv \
  --bind "${ROOT_DIR}:/workspace/easy_safe_sft" \
  --bind "${HF_CACHE}:/root/.cache/huggingface" \
  --env PYTHONPATH="${SFT_PKG}:/workspace/easy_safe_sft" \
  --env HF_HOME="${HF_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env WANDB_DISABLED=true \
  --env TMPDIR="${TMP_DIR}" \
  --env TRITON_CACHE_DIR="/ocean/projects/cis250290p/tzhou6/triton" \
  --env TORCH_EXTENSIONS_DIR="/ocean/projects/cis250290p/tzhou6/torch_extensions" \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env NPROC_PER_NODE=2 \
  "${SIF}" \
  bash -c "
    cd /workspace/easy_safe_sft
    python easy_safe_sft/train.py \
      --llamafactory-root /app \
      --config-path /workspace/easy_safe_sft/examples/contractnli_reasoning_sft/train_full.yaml
  "

echo ""
echo ">>> Training done"
echo ""

# ── Step 4: Evaluation ────────────────────────
echo ">>> [4/4] Evaluation (vLLM, 2xH100 TP) ..."
echo ""

apptainer exec --nv \
  --bind "${ROOT_DIR}:/workspace/easy_safe_sft" \
  --bind "${HF_CACHE}:/root/.cache/huggingface" \
  --bind "${PROJECT_DIR}/prompts:/workspace/prompts" \
  --env PYTHONPATH="${VLLM_PKG}:${SFT_PKG}:/workspace/easy_safe_sft" \
  --env HF_HOME="${HF_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env TMPDIR="${TMP_DIR}" \
  --env TRITON_CACHE_DIR="/ocean/projects/cis250290p/tzhou6/triton" \
  --env TORCH_EXTENSIONS_DIR="/ocean/projects/cis250290p/tzhou6/torch_extensions" \
  --env VLLM_USE_V1=0 \
  "${SIF}" \
  python /workspace/easy_safe_sft/easy_safe_sft/eval_checkpoints.py \
    --config-path /workspace/easy_safe_sft/examples/contractnli_reasoning_sft/train_full.yaml \
    --prompt-config /workspace/prompts/contractNLI/sft_reasoning_nothink.yaml \
    --eval-concurrency 32 \
    --vllm-tp-size 2 \
    --vllm-gpu-util 0.9 \
    --vllm-maxlen 4096 \
    --has-reasoning \
    --include-base-model

echo ""
echo "=========================================="
echo " Full Run Complete"
echo "=========================================="

# Print all eval results
for f in "${ROOT_DIR}/temp/eval/contractnli_reasoning_sft_full/"*/valid/summary.json; do
  target=$(echo "$f" | grep -oP 'contractnli_reasoning_sft_full/\K[^/]+')
  echo "  [${target}]"
  cat "$f"
  echo ""
done
