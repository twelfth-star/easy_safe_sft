#!/bin/bash
#SBATCH -A cis250290p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --mem=120GB
#SBATCH --ntasks-per-node=10
#SBATCH -t 03:00:00
#SBATCH -o slurm-contractnli-full-%j.out
#SBATCH -J contractnli-sft

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

cd /ocean/projects/cis250290p/tzhou6/fin-llm-finetune/easy_safe_sft
export API_KEY_ENV=ANTHROPIC_API_KEY2
bash examples/contractnli_reasoning_sft/run_full.sh

echo "End: $(date)"
