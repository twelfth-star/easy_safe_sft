#!/usr/bin/env python3

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from easy_safe_sft.utils import load_json, load_yaml, prepend_pythonpath, resolve_yaml_paths, run_command, write_yaml


LOCAL_CONFIG_KEYS = {
    "easy_safe_sft_eval_concurrency",
    "easy_safe_sft_vllm_tp_size",
    "easy_safe_sft_vllm_gpu_util",
    "easy_safe_sft_vllm_maxlen",
}


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
    total_steps = math.floor(float(train_args["num_train_epochs"]) * steps_per_epoch)

    logger.info(
        "计算总步数: per_device_train_batch_size={}, gradient_accumulation_steps={}, num_gpus={}, examples={}, effective_batch={}, steps_per_epoch={}, epochs={} -> total_steps={}",
        per_device_train_batch_size,
        gradient_accumulation_steps,
        num_gpus,
        total_examples,
        effective_batch,
        steps_per_epoch,
        train_args["num_train_epochs"],
        total_steps,
    )
    return total_steps


def _make_train_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = {key: value for key, value in base_config.items() if key not in LOCAL_CONFIG_KEYS}
    config["do_train"] = True
    config["do_eval"] = False
    config["do_predict"] = False
    config["predict_with_generate"] = False
    config.pop("eval_strategy", None)
    config.pop("eval_steps", None)
    config.pop("eval_dataset", None)
    config.pop("eval_on_each_dataset", None)
    config.pop("per_device_eval_batch_size", None)
    config.pop("do_sample", None)
    config.pop("max_new_tokens", None)
    return config


def _run_llamafactory_train(llamafactory_root: str, config_path: str) -> None:
    src_dir = Path(llamafactory_root) / "src"
    env = prepend_pythonpath(os.environ, str(src_dir))
    run_command([sys.executable, "-m", "llamafactory.cli", "train", str(config_path)], cwd=llamafactory_root, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="单次完整训练，按 save_steps 保存 checkpoint")
    parser.add_argument("--llamafactory-root", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    _setup_logger()
    train_args = resolve_yaml_paths(load_yaml(args.config_path), args.config_path)

    total_steps = _compute_total_steps(train_args)
    logger.info("train 开始: total_steps={}", total_steps)

    tmp_dir = Path(tempfile.mkdtemp(prefix="train_"))
    logger.info("临时 YAML 目录: {}", tmp_dir)

    train_yaml_path = tmp_dir / "train.yaml"
    write_yaml(train_yaml_path, _make_train_config(train_args))

    logger.info("=== 开始单次完整训练 ===")
    _run_llamafactory_train(args.llamafactory_root, str(train_yaml_path))
    logger.info("train 完成: total_steps={}", total_steps)


if __name__ == "__main__":
    main()
