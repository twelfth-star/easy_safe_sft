#!/usr/bin/env python3
"""Teacher distillation CLI.

Loads a canonical dataset + prompt config + rules, calls teacher API with
rejection sampling, and writes accepted/rejected traces to output-dir.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher distillation with rejection sampling")
    parser.add_argument("--teacher-config", required=True, help="Teacher YAML config (model, k, workers, etc.)")
    parser.add_argument("--prompt-config", required=True, help="Unified prompt+task YAML")
    parser.add_argument("--dataset-root", required=True, help="Root directory of canonical datasets")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--split-policy", default="policy_v1")
    parser.add_argument("--phase-name", default="phase2")
    parser.add_argument("--rules-dir", default=None, help="Rules directory (optional)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key-env", default=None,
                        help="Environment variable name for the API key (e.g. OPENAI_API_KEY2, ANTHROPIC_API_KEY2)")
    args = parser.parse_args()

    import os
    from easy_safe_sft.data_loading import load_canonical_splits, resolve_canonical_dataset_spec
    from easy_safe_sft.prompt_config import load_prompt_config, read_rulebook
    from easy_safe_sft.teacher import distill_dataset
    from easy_safe_sft.utils import load_yaml, setup_file_logging

    # Load .env file for API keys
    from dotenv import load_dotenv
    env_path = Path(args.dataset_root).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    elif Path(".env").exists():
        load_dotenv()

    # Allow selecting an alternate API key variable (e.g. OPENAI_API_KEY2)
    if args.api_key_env:
        key_value = os.environ.get(args.api_key_env)
        if not key_value:
            raise ValueError(f"API key env var '{args.api_key_env}' not found in environment or .env")
        # Map to the standard env var that litellm expects
        if "OPENAI" in args.api_key_env.upper():
            os.environ["OPENAI_API_KEY"] = key_value
        elif "ANTHROPIC" in args.api_key_env.upper():
            os.environ["ANTHROPIC_API_KEY"] = key_value
        elif "GOOGLE" in args.api_key_env.upper():
            os.environ["GOOGLE_API_KEY"] = key_value

    setup_file_logging(args.output_dir)

    teacher_config = load_yaml(args.teacher_config)
    prompt_cfg = load_prompt_config(args.prompt_config)

    # Load canonical dataset
    spec = resolve_canonical_dataset_spec(
        dataset_root=Path(args.dataset_root),
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        split_policy=args.split_policy,
    )
    splits = load_canonical_splits(spec, phase_name=args.phase_name)
    train_df = splits["train"]

    train_df = train_df.copy()
    train_df["label"] = train_df["label"].astype(int)  # ensure Python int, not numpy.int64
    examples = train_df[["example_id", "text", "label"]].to_dict(orient="records")

    rulebook = ""
    if args.rules_dir:
        rulebook = read_rulebook(args.rules_dir, teacher_config.get("rules_glob", "*.txt"))

    distill_dataset(
        prompt_cfg=prompt_cfg,
        teacher_config=teacher_config,
        examples=examples,
        output_dir=args.output_dir,
        rulebook=rulebook,
    )


if __name__ == "__main__":
    main()
