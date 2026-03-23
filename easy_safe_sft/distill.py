#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 teacher 模型生成 CoT，并只保留回答正确的样本")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from easy_safe_sft.teacher import distill_dataset
    from easy_safe_sft.utils import load_yaml

    distill_dataset(
        task_config=load_yaml(args.task_config),
        teacher_config=load_yaml(args.teacher_config),
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
