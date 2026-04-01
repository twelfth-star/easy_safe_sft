#!/usr/bin/env python3
"""Convert raw data or teacher traces to LlamaFactory training format."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw/trace data to LlamaFactory format")
    parser.add_argument("--prompt-config", required=True, help="Unified prompt+task YAML")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--train-source", choices=["raw", "trace"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-template", required=True)
    parser.add_argument("--plain-output-style", default="final_only")
    parser.add_argument("--distill-output-style", default="auto")
    parser.add_argument("--system-prompt", default="detailed thinking off")
    parser.add_argument("--valid-input")
    parser.add_argument("--test-input")
    args = parser.parse_args()

    from easy_safe_sft.dataset_builder import build_dataset
    from easy_safe_sft.prompt_config import load_prompt_config

    prompt_cfg = load_prompt_config(args.prompt_config)

    build_dataset(
        prompt_cfg=prompt_cfg,
        train_input_path=args.train_input,
        train_source=args.train_source,
        output_dir=args.output_dir,
        student_template=args.student_template,
        plain_output_style=args.plain_output_style,
        distill_output_style=args.distill_output_style,
        system_prompt=args.system_prompt,
        valid_input_path=args.valid_input,
        test_input_path=args.test_input,
    )


if __name__ == "__main__":
    main()
