# easy_safe_sft

`easy_safe_sft` is a thin wrapper around LlamaFactory and vLLM for safety-task SFT.

It supports two training styles:

- direct SFT: the model is trained to answer directly, for example `Yes` or `No`
- reasoning distillation SFT: a teacher first produces reasoning plus a final answer, only correct traces are kept, and the student is trained on those traces

## Two environments

You need two separate environments.

### 1. LlamaFactory Docker for training

Training should run inside the LlamaFactory Docker environment.

```bash
cd /path/to/LlamaFactory/docker/docker-cuda

docker build -f ./docker/docker-cuda/Dockerfile -t llamafactory:local .

docker rm -f llamafactory 2>/dev/null || true

docker run -dit --ipc=host --gpus=all \
  -p 7860:7860 \
  -p 8000:8000 \
  -v /path/to/hf_cache:/root/.cache/huggingface \
  -v /path/to/easy_safe_sft:/workspace/easy_safe_sft \
  --name llamafactory \
  llamafactory:local

docker exec -it llamafactory bash
```

Inside the container:

```bash
cd /app
pip install -e .
pip install -r requirements/metrics.txt
pip install wandb
pip install loguru
```

When training is done:

```bash
docker stop llamafactory
```

### 2. A separate vLLM environment for evaluation

Evaluation should run outside Docker in a different environment:

```bash
pip install uv
uv venv /path/to/vllm --python 3.11
source /path/to/vllm/bin/activate
uv self update
uv pip install vllm --torch-backend=cu128  # change cu128 to your CUDA version
uv pip install transformers PyYAML loguru tqdm accelerate
```

This environment is only for inference and scoring.

## Why training and evaluation are split

Training is done with LlamaFactory. Evaluation is done later with vLLM.

This repo uses generative evaluation, not simple classification logits. That is much slower. vLLM is the practical way to make checkpoint evaluation fast.

Doing checkpoint evaluation in the middle of LlamaFactory training would be awkward: you would need to stop training, switch into a different inference stack, and then resume. LlamaFactory also does not naturally support this exact workflow.

So the intended workflow is:

1. train once with LlamaFactory
2. save checkpoints during training
3. after training finishes, evaluate the base model, every saved checkpoint, and the final model with vLLM

## Examples

### Direct SFT

`examples/ethos_sft/` is the simple path.

- `task.yaml` defines a direct prompt
- `build.sh` builds the LlamaFactory dataset from `data/raw/ethos/*.jsonl`
- the student is trained to output only `Yes` or `No`
- evaluation uses the same direct prompt

Build the dataset:

```bash
cd /path/to/easy_safe_sft/examples/ethos_sft
bash build.sh
```

Training in Docker:

```bash
cd /workspace/easy_safe_sft/examples/ethos_sft
bash train.sh
```

Evaluation on the host:

```bash
cd /path/to/easy_safe_sft/examples/ethos_sft
bash eval.sh
```

### Teacher distillation with reasoning

`examples/ethos_teacher_distillation_reasoning_sft/` is the reasoning path.

- `task.yaml` defines `reasoning_instruction`
- the teacher must produce reasoning and a final answer in a fixed format
- only teacher outputs with the correct final answer are kept
- `distill.sh` both runs teacher distillation and builds the student dataset
- the student is trained on those traces
- evaluation uses the same reasoning-style prompt

Typical order:

```bash
cd /path/to/easy_safe_sft/examples/ethos_teacher_distillation_reasoning_sft
bash distill.sh
```

Then train in Docker:

```bash
cd /workspace/easy_safe_sft/examples/ethos_teacher_distillation_reasoning_sft
bash train.sh
```

Then evaluate on the host:

```bash
cd /path/to/easy_safe_sft/examples/ethos_teacher_distillation_reasoning_sft
bash eval.sh
```

## Evaluation behavior

Evaluation includes:

- the base model
- every `checkpoint-*`
- the final model

Each model is evaluated separately. Each dataset gets its own progress bar.

You can override evaluation settings from the shell:

```bash
EVAL_CONCURRENCY=16 VLLM_TP_SIZE=2 VLLM_GPU_UTIL=0.85 bash eval.sh
```

## Data format

Raw data is JSONL.

Binary classification example:

```json
{
  "id": "train-1",
  "split": "train",
  "fields": {
    "text": "some sentence"
  },
  "answer": 1,
  "meta": {}
}
```

Built datasets use OpenAI-style `messages` and explicitly include:

```json
{"role": "system", "content": "detailed thinking off"}
```

That keeps training and evaluation prompts aligned.

## Main entry points

All Python code lives under `easy_safe_sft/`:

- `easy_safe_sft/build_dataset.py`
- `easy_safe_sft/distill.py`
- `easy_safe_sft/train.py`
- `easy_safe_sft/eval_checkpoints.py`
- `easy_safe_sft/score_predictions.py`

Evaluate all checkpoints:

```bash
python easy_safe_sft/eval_checkpoints.py \
  --config-path path/to/eval.yaml \
  --task-config path/to/task.yaml \
  --eval-concurrency 32 \
  --vllm-tp-size 2 \
  --include-base-model
```

Score one prediction file:

```bash
python easy_safe_sft/score_predictions.py \
  --task-config path/to/task.yaml \
  --student-template qwen3_nothink \
  --predictions path/to/generated_predictions.jsonl \
  --meta path/to/valid_meta.jsonl \
  --summary-output path/to/eval.json
```
