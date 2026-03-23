#!/usr/bin/env python3

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from easy_safe_sft.score_predictions import evaluate_predictions
from easy_safe_sft.utils import load_json, load_yaml
from easy_safe_sft.vllm_predict import VllmPredictor


def _setup_logger() -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("EASY_SAFE_SFT_LOG_LEVEL", "INFO"))


def _compute_total_steps(train_args: dict[str, Any]) -> int:
    if int(train_args.get("max_steps", -1)) > 0:
        return int(train_args["max_steps"])

    if "num_train_epochs" not in train_args:
        raise ValueError("YAML 中必须设置 max_steps > 0 或 num_train_epochs")

    dataset_dir = Path(train_args["dataset_dir"])
    dataset_names = [name.strip() for name in str(train_args.get("dataset", "train")).split(",")]
    dataset_info = load_json(dataset_dir / "dataset_info.json")

    total_examples = 0
    for name in dataset_names:
        file_name = dataset_info[name].get("file_name", f"{name}.jsonl")
        with open(dataset_dir / file_name, "r", encoding="utf-8") as f:
            total_examples += sum(1 for line in f if line.strip())

    num_gpus = int(os.getenv("NPROC_PER_NODE", "1"))
    per_device_train_batch_size = int(train_args.get("per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(train_args.get("gradient_accumulation_steps", 1))
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    steps_per_epoch = math.ceil(total_examples / effective_batch)
    return math.floor(float(train_args["num_train_epochs"]) * steps_per_epoch)


def _resolve_total_steps(train_args: dict[str, Any]) -> int:
    trainer_state_path = Path(train_args["output_dir"]) / "trainer_state.json"
    if trainer_state_path.exists():
        trainer_state = load_json(trainer_state_path)
        global_step = trainer_state.get("global_step")
        if isinstance(global_step, int):
            logger.info("从 trainer_state.json 读取总步数: {}", global_step)
            return global_step
    return _compute_total_steps(train_args)


def _report_to_list(train_args: dict[str, Any]) -> list[str]:
    report_to = train_args.get("report_to", "none")
    if isinstance(report_to, list):
        return [str(item) for item in report_to]
    return [str(report_to)]


def _log_to_tensorboard(log_dir: Path, metrics: dict[str, Any], step: int, dataset_name: str) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard 未安装，跳过 tensorboard 指标记录")
        return

    writer = SummaryWriter(log_dir=str(log_dir))
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"eval_{dataset_name}/{key}", value, step)
    writer.close()


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _has_saved_model(path: Path) -> bool:
    marker_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    return any((path / name).exists() for name in marker_files)


def _collect_eval_targets(output_dir: Path, total_steps: int) -> list[tuple[str, str | None, int]]:
    targets: list[tuple[str, str | None, int]] = [("base_model", None, 0)]

    for child in sorted(output_dir.iterdir(), key=lambda path: (_checkpoint_step(path) is None, _checkpoint_step(path) or -1)):
        if child.is_dir():
            step = _checkpoint_step(child)
            if step is not None:
                targets.append((child.name, str(child), step))

    if _has_saved_model(output_dir):
        targets.append(("final", str(output_dir), total_steps))

    return targets


def _eval_output_dir(train_output_dir: str, target_name: str, dataset_name: str) -> Path:
    experiment_name = Path(train_output_dir).parent.name
    return PROJECT_ROOT / "temp" / "eval" / experiment_name / target_name / dataset_name


def main() -> None:
    parser = argparse.ArgumentParser(description="用纯 vLLM 对训练输出目录里的 checkpoint 和 final 模型统一评测")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--prediction-style", default="auto")
    parser.add_argument("--has-reasoning", action="store_true")
    parser.add_argument("--eval-concurrency", type=int)
    parser.add_argument("--vllm-tp-size", type=int)
    parser.add_argument("--vllm-gpu-util", type=float)
    parser.add_argument("--vllm-maxlen", type=int)
    parser.add_argument("--include-base-model", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _setup_logger()
    train_args = load_yaml(args.config_path)
    task_config = load_yaml(args.task_config)

    eval_datasets = [name.strip() for name in str(train_args.get("eval_dataset", "")).split(",") if name.strip()]
    if not eval_datasets:
        raise ValueError("YAML 里必须设置 eval_dataset")

    total_steps = _resolve_total_steps(train_args)
    eval_concurrency = int(args.eval_concurrency or train_args.get("easy_safe_sft_eval_concurrency", 32))
    vllm_tp_size = args.vllm_tp_size
    if vllm_tp_size is None:
        config_tp_size = int(train_args.get("easy_safe_sft_vllm_tp_size", -1))
        vllm_tp_size = None if config_tp_size == -1 else config_tp_size
    vllm_gpu_util = float(args.vllm_gpu_util or train_args.get("easy_safe_sft_vllm_gpu_util", train_args.get("vllm_gpu_util", 0.7)))
    vllm_maxlen = int(args.vllm_maxlen or train_args.get("easy_safe_sft_vllm_maxlen", train_args.get("vllm_maxlen", 4096)))

    logger.info(
        "evaluate_checkpoints 开始: total_steps={}, eval_concurrency={}, vllm_tp_size={}, vllm_gpu_util={}, vllm_maxlen={}, include_base_model={}, eval_datasets={}",
        total_steps,
        eval_concurrency,
        vllm_tp_size if vllm_tp_size is not None else "all_visible",
        vllm_gpu_util,
        vllm_maxlen,
        args.include_base_model,
        eval_datasets,
    )

    eval_targets = _collect_eval_targets(Path(train_args["output_dir"]), total_steps)
    if not args.include_base_model:
        eval_targets = [target for target in eval_targets if target[0] != "base_model"]
    if not eval_targets:
        raise ValueError(f"训练结束后未在 {train_args['output_dir']} 找到可评测目标")

    logger.info("准备统一评测这些目标: {}", [name for name, _, _ in eval_targets])

    use_tensorboard = "tensorboard" in _report_to_list(train_args)
    dataset_dir = Path(train_args["dataset_dir"])
    dataset_info = load_json(dataset_dir / "dataset_info.json")

    for target_name, adapter_path, step in eval_targets:
        logger.info("=== 开始测试模型: {} (step={}, adapter_path={}) ===", target_name, step, adapter_path)
        predictor = VllmPredictor(
            base_config=train_args,
            adapter_path=adapter_path,
            concurrency=eval_concurrency,
            tp_size=vllm_tp_size,
            gpu_util=vllm_gpu_util,
            maxlen=vllm_maxlen,
        )

        try:
            for dataset_name in eval_datasets:
                logger.info("=== 预测+评测: target={}, step={}, dataset={} ===", target_name, step, dataset_name)
                result_dir = _eval_output_dir(train_args["output_dir"], target_name, dataset_name)
                result_dir.mkdir(parents=True, exist_ok=True)

                dataset_file = dataset_info[dataset_name].get("file_name", f"{dataset_name}.jsonl")
                prediction_path = result_dir / "generated_predictions.jsonl"
                summary_path = result_dir / "summary.json"
                details_path = result_dir / "details.jsonl"
                meta_path = dataset_dir / f"{dataset_name}_meta.jsonl"

                predictor.predict_dataset(
                    dataset_path=str(dataset_dir / dataset_file),
                    output_path=str(prediction_path),
                    progress_desc=f"{target_name}:{dataset_name}",
                )

                evaluate_predictions(
                    task_config=task_config,
                    student_template=str(train_args["template"]),
                    prediction_path=str(prediction_path),
                    meta_path=str(meta_path),
                    summary_output_path=str(summary_path),
                    details_output_path=str(details_path),
                    prediction_style=args.prediction_style,
                    has_reasoning=args.has_reasoning,
                )

                summary = load_json(summary_path)
                logger.info("评测结果: target={}, step={}, dataset={}, {}", target_name, step, dataset_name, summary)

                if use_tensorboard:
                    _log_to_tensorboard(result_dir, summary, step, dataset_name)
        finally:
            predictor.close()

    logger.info("evaluate_checkpoints 完成")


if __name__ == "__main__":
    main()
