#!/usr/bin/env bash
# =============================================================
#  Smoke test: ContractNLI reasoning distillation SFT pipeline
#  Teacher: Claude Sonnet 4.6 (only 5 examples to minimize cost)
#  Student: Qwen3-4B (2 training steps)
#
#  Tests the full pipeline:
#    1. Teacher distillation (API call + rejection sampling)
#    2. Build LlamaFactory dataset from traces
#    3. Training (2 steps, 2xH100 DDP)
#    4. Evaluation (vLLM, 2xH100 TP)
# =============================================================
set -euo pipefail

# Activate conda research env for Steps 1-2 (litellm, pandas, dotenv)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate research 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

# Paths
PROMPT_CONFIG="${PROJECT_DIR}/prompts/contractNLI/sft_reasoning_nothink.yaml"
TEACHER_CONFIG="${ROOT_DIR}/examples/contractnli_reasoning_sft/teacher.yaml"
TRAIN_CONFIG="${ROOT_DIR}/examples/contractnli_reasoning_sft/smoke_train.yaml"
RULES_DIR="${PROJECT_DIR}/rules/contractNLI"
DATASET_ROOT="${PROJECT_DIR}/datasets"

TRACE_DIR="${ROOT_DIR}/data/distilled/contractnli_reasoning_sft_smoke"
LF_DATA_DIR="${ROOT_DIR}/data/lf_ready/contractnli_reasoning_sft"
SMOKE_OUTPUT="${ROOT_DIR}/outputs/contractnli_reasoning_sft_smoke"

# Apptainer config
SIF="${PROJECT_DIR}/llamafactory_latest.sif"
SFT_PKG="${PROJECT_DIR}/sft_site_packages"
VLLM_PKG="${PROJECT_DIR}/vllm_site_packages"
HF_CACHE="${PROJECT_DIR}/../hf_cache"

echo "=========================================="
echo " Smoke Test: ContractNLI Reasoning SFT"
echo "=========================================="
echo ""

# ── Clean previous smoke outputs ──────────────
rm -rf "${TRACE_DIR}" "${SMOKE_OUTPUT}" "${LF_DATA_DIR}"
rm -rf "${ROOT_DIR}/temp/eval/contractnli_reasoning_sft_smoke"

# ── Step 1: Teacher distillation (5 examples only) ──
echo ">>> [1/4] Teacher distillation (Claude Sonnet 4.6, 5 examples) ..."
echo ""

# Create a tiny subset: take first 5 train examples
cd "${ROOT_DIR}"
python -c "
from easy_safe_sft.data_loading import resolve_canonical_dataset_spec, load_canonical_splits
from pathlib import Path
import json

spec = resolve_canonical_dataset_spec(
    dataset_root=Path('${DATASET_ROOT}'),
    dataset_name='contractNLI',
    dataset_version='v1',
    split_policy='policy_v1',
)
splits = load_canonical_splits(spec, phase_name='phase2')
# Take only 5 train examples for smoke test
tiny = splits['train'].head(5)
tiny['label'] = tiny['label'].astype(int)
records = tiny.to_dict(orient='records')
print(f'Selected {len(records)} examples for smoke test')
# Write to temp file
import json
from pathlib import Path
out = Path('${TRACE_DIR}')
out.mkdir(parents=True, exist_ok=True)
with open(out / 'smoke_examples.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('Wrote smoke_examples.jsonl')
"

# Run teacher distillation using distill.py but with the tiny subset
# We bypass the canonical dataset loading and pass examples directly
python -c "
import os, sys
sys.path.insert(0, '${ROOT_DIR}')
from dotenv import load_dotenv
load_dotenv('${PROJECT_DIR}/.env')
# Use ANTHROPIC_API_KEY for Claude
from easy_safe_sft.prompt_config import load_prompt_config, read_rulebook
from easy_safe_sft.teacher import distill_dataset
from easy_safe_sft.utils import load_yaml, read_jsonl, setup_file_logging

setup_file_logging('${TRACE_DIR}')
teacher_config = load_yaml('${TEACHER_CONFIG}')
prompt_cfg = load_prompt_config('${PROMPT_CONFIG}')
examples = read_jsonl('${TRACE_DIR}/smoke_examples.jsonl')
rulebook = read_rulebook('${RULES_DIR}', teacher_config.get('rules_glob', '*.txt'))

distill_dataset(
    prompt_cfg=prompt_cfg,
    teacher_config=teacher_config,
    examples=examples,
    output_dir='${TRACE_DIR}',
    rulebook=rulebook,
)
"

# Check results
if [ -f "${TRACE_DIR}/accepted_samples.jsonl" ]; then
  ACCEPTED=$(wc -l < "${TRACE_DIR}/accepted_samples.jsonl")
  echo ""
  echo ">>> Teacher distillation PASSED: ${ACCEPTED} accepted samples"
  cat "${TRACE_DIR}/teacher_summary.json" 2>/dev/null
  echo ""
else
  echo ">>> Teacher distillation FAILED"
  exit 1
fi

# ── Step 2: Build LlamaFactory dataset ────────
echo ">>> [2/4] Building LlamaFactory dataset ..."
echo ""

# We need valid/test data too — create tiny subsets
python -c "
from easy_safe_sft.data_loading import resolve_canonical_dataset_spec, load_canonical_splits
from easy_safe_sft.utils import write_jsonl
from pathlib import Path
import json

spec = resolve_canonical_dataset_spec(
    dataset_root=Path('${DATASET_ROOT}'),
    dataset_name='contractNLI',
    dataset_version='v1',
    split_policy='policy_v1',
)
splits = load_canonical_splits(spec, phase_name='phase2')

# Write tiny valid/test as raw JSONL (5 examples each)
for split_name in ('val', 'test'):
    tiny = splits[split_name].head(5)
    rows = []
    for _, r in tiny.iterrows():
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
    print(f'Wrote {split_name}_raw.jsonl: {len(rows)} rows')
"

python "${ROOT_DIR}/easy_safe_sft/build_dataset.py" \
  --prompt-config "${PROMPT_CONFIG}" \
  --train-input "${TRACE_DIR}/accepted_samples.jsonl" \
  --train-source trace \
  --valid-input "${TRACE_DIR}/val_raw.jsonl" \
  --test-input "${TRACE_DIR}/test_raw.jsonl" \
  --output-dir "${LF_DATA_DIR}" \
  --student-template qwen3_nothink \
  --distill-output-style reasoning_label

echo ">>> Dataset build PASSED"
echo ""

# ── Step 3: Training (Apptainer, 2 steps, 2xH100) ──
echo ">>> [3/4] Training (Qwen3-4B, 2 steps, 2xH100 DDP) ..."
echo ""

apptainer exec --nv \
  --bind "${ROOT_DIR}:/workspace/easy_safe_sft" \
  --bind "${HF_CACHE}:/root/.cache/huggingface" \
  --env PYTHONPATH="${SFT_PKG}:/workspace/easy_safe_sft" \
  --env HF_HOME="${HF_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env WANDB_DISABLED=true \
  --env TMPDIR="${PROJECT_DIR}/../tmp" \
  --env TRITON_CACHE_DIR="${PROJECT_DIR}/../triton" \
  --env TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/../torch_extensions" \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env NPROC_PER_NODE=2 \
  "${SIF}" \
  bash -c "
    cd /workspace/easy_safe_sft
    python easy_safe_sft/train.py \
      --llamafactory-root /app \
      --config-path /workspace/easy_safe_sft/examples/contractnli_reasoning_sft/smoke_train.yaml
  "

if [ -d "${SMOKE_OUTPUT}/model/checkpoint-2" ]; then
  echo ""
  echo ">>> Training PASSED: checkpoint-2 generated"
else
  echo ">>> Training FAILED"
  exit 1
fi
echo ""

# ── Step 4: Evaluation (Apptainer, vLLM, 2xH100 TP) ──
echo ">>> [4/4] Evaluation (vLLM, 2xH100 TP) ..."
echo ""

apptainer exec --nv \
  --bind "${ROOT_DIR}:/workspace/easy_safe_sft" \
  --bind "${HF_CACHE}:/root/.cache/huggingface" \
  --bind "${PROJECT_DIR}/prompts:/workspace/prompts" \
  --env PYTHONPATH="${VLLM_PKG}:${SFT_PKG}:/workspace/easy_safe_sft" \
  --env HF_HOME="${HF_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env TMPDIR="${PROJECT_DIR}/../tmp" \
  --env TRITON_CACHE_DIR="${PROJECT_DIR}/../triton" \
  --env TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/../torch_extensions" \
  --env VLLM_USE_V1=0 \
  "${SIF}" \
  python /workspace/easy_safe_sft/easy_safe_sft/eval_checkpoints.py \
    --config-path /workspace/easy_safe_sft/examples/contractnli_reasoning_sft/smoke_train.yaml \
    --prompt-config /workspace/prompts/contractNLI/sft_reasoning_nothink.yaml \
    --eval-concurrency 32 \
    --vllm-tp-size 2 \
    --vllm-gpu-util 0.9 \
    --vllm-maxlen 4096 \
    --has-reasoning \
    --no-include-base-model

echo ""

# Check eval results
if ls "${ROOT_DIR}/temp/eval/contractnli_reasoning_sft_smoke/"*/valid/summary.json 1>/dev/null 2>&1; then
  echo ">>> Evaluation PASSED. Results:"
  for f in "${ROOT_DIR}/temp/eval/contractnli_reasoning_sft_smoke/"*/valid/summary.json; do
    target=$(echo "$f" | grep -oP 'contractnli_reasoning_sft_smoke/\K[^/]+')
    echo "  [${target}]"
    cat "$f"
    echo ""
  done
else
  echo ">>> Evaluation FAILED"
  exit 1
fi

echo "=========================================="
echo " Smoke Test PASSED!"
echo "=========================================="
