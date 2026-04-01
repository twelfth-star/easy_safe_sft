"""Teacher distillation with rejection sampling, retry, multithreading, and checkpointing.

Ported from tinker-to-gpu/tinker/SFT_reasoning_teacher.py.
Uses litellm for multi-provider API access (OpenAI, Anthropic, etc.).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from litellm import completion
from loguru import logger
from tqdm.auto import tqdm

from easy_safe_sft.prompt_config import PromptConfig, build_teacher_messages
from easy_safe_sft.tasks import normalize_answer, parse_teacher_output
from easy_safe_sft.utils import read_jsonl, write_json, write_jsonl


def _call_with_retry(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    timeout: int,
) -> tuple[str, str | None]:
    """Call litellm.completion with retry and exponential backoff.

    Returns (raw_content, error_or_None).
    """
    last_error: str | None = None
    for attempt in range(max(1, max_retries)):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = response["choices"][0]["message"]["content"]
            return str(content or ""), None
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < max_retries:
                time.sleep(min(4.0, 1.0 + attempt))
    return "", last_error


def _sample_one_example(
    prompt_cfg: PromptConfig,
    teacher_config: dict[str, Any],
    text: str,
    gold_label: int,
    example_id: str,
    *,
    rulebook: str,
    use_rulebook: bool,
) -> dict[str, Any]:
    """Rejection sampling for one example: try up to k times."""
    k = int(teacher_config.get("k", 3))
    model = str(teacher_config["model"])
    max_tokens = int(teacher_config.get("max_tokens", 512))
    retry_temperature = float(teacher_config.get("retry_temperature", 0.7))
    base_max_retries = int(teacher_config.get("max_retries", 3))
    timeout = int(teacher_config.get("request_timeout", 90))
    output_format = str(teacher_config.get("output_format", "reasoning_label"))
    min_reasoning_chars = int(teacher_config.get("min_reasoning_chars", 20))
    answer_prefix = prompt_cfg.answer_prefix

    messages = build_teacher_messages(prompt_cfg, text, rulebook=rulebook, use_rulebook=use_rulebook)

    samples: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    first_try_label: int | None = None

    for ki in range(k):
        temperature = 0.0 if ki == 0 else retry_temperature
        max_retries = 1 if ki == 0 else base_max_retries

        raw_output, api_error = _call_with_retry(
            model, messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
        )

        reasoning, raw_answer, parse_error = parse_teacher_output(raw_output, output_format, answer_prefix)

        sample = {
            "example_id": example_id,
            "k_index": ki,
            "temperature": temperature,
            "raw_output": raw_output,
            "parse_ok": parse_error is None,
            "reasoning": reasoning,
            "raw_answer": raw_answer,
            "api_error": api_error,
            "parse_error": parse_error,
            "reject_reason": None,
        }

        if api_error:
            sample["reject_reason"] = f"api_error:{api_error[:100]}"
        elif parse_error:
            sample["reject_reason"] = parse_error
        else:
            try:
                normalized = normalize_answer(prompt_cfg.as_task_config, raw_answer)
                # Map normalized answer back to 0/1 for comparison
                inv_map = {v: int(k_) for k_, v in prompt_cfg.label_map.items()}
                pred_label = inv_map.get(normalized)
            except (ValueError, KeyError):
                sample["reject_reason"] = "normalize_failed"
                pred_label = None

            if pred_label is not None:
                sample["pred_label"] = pred_label
                if ki == 0:
                    first_try_label = pred_label

                if pred_label != gold_label:
                    sample["reject_reason"] = "label_mismatch"
                elif reasoning and len(reasoning) < min_reasoning_chars:
                    sample["reject_reason"] = "reasoning_too_short"
                else:
                    # Accepted!
                    accepted = {
                        "example_id": example_id,
                        "text": text,
                        "label": gold_label,
                        "reasoning": reasoning or "",
                        "pred_label": pred_label,
                    }

        samples.append(sample)
        if accepted is not None:
            break

    return {
        "example_id": example_id,
        "samples": samples,
        "accepted": accepted,
        "first_try_label": first_try_label,
        "final_label": accepted["pred_label"] if accepted else None,
    }


def _compute_teacher_summary(
    results: list[dict[str, Any]],
    total: int,
) -> dict[str, Any]:
    """Compute summary metrics for teacher distillation."""
    accepted_count = sum(1 for r in results if r["accepted"] is not None)
    first_try_correct = sum(
        1 for r in results
        if r["first_try_label"] is not None
        and r["accepted"] is not None
        and r["first_try_label"] == r["accepted"]["label"]
    )
    first_try_total = sum(1 for r in results if r["first_try_label"] is not None)
    return {
        "total_examples": total,
        "accepted": accepted_count,
        "rejected": total - accepted_count,
        "acceptance_rate": accepted_count / total if total else 0.0,
        "first_try_accuracy": first_try_correct / first_try_total if first_try_total else 0.0,
    }


def distill_dataset(
    prompt_cfg: PromptConfig,
    teacher_config: dict[str, Any],
    examples: list[dict[str, Any]],
    output_dir: str,
    rulebook: str = "",
) -> None:
    """Run teacher distillation with rejection sampling.

    Args:
        prompt_cfg: Unified prompt+task config.
        teacher_config: Teacher model config (model, k, workers, etc.).
        examples: List of dicts with keys: example_id, text, label (int 0/1).
        output_dir: Directory for output files.
        rulebook: Concatenated rulebook text (empty string if no rules).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted_path = out / "accepted_samples.jsonl"
    rejected_path = out / "rejected_samples.jsonl"
    summary_path = out / "teacher_summary.json"

    # Skip if already complete
    if accepted_path.exists() and summary_path.exists():
        logger.info("Teacher output already exists at {}, skipping.", out)
        return

    # Load checkpoint (already processed IDs — both accepted and rejected)
    checkpoint_path = out / "checkpoint_accepted.jsonl"
    checkpoint_rej_path = out / "checkpoint_rejected.jsonl"
    processed_ids: set[str] = set()
    checkpoint_accepted: list[dict[str, Any]] = []
    checkpoint_rejected: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint_accepted = read_jsonl(str(checkpoint_path))
        processed_ids.update(r["example_id"] for r in checkpoint_accepted)
    if checkpoint_rej_path.exists():
        checkpoint_rejected = read_jsonl(str(checkpoint_rej_path))
        processed_ids.update(r["example_id"] for r in checkpoint_rejected)
    if processed_ids:
        logger.info("Resuming from checkpoint: {} examples already processed.", len(processed_ids))

    remaining = [e for e in examples if e["example_id"] not in processed_ids]
    if not remaining:
        logger.info("All examples already processed.")
        # Write final outputs from checkpoint
        write_jsonl(str(accepted_path), checkpoint_accepted)
        write_jsonl(str(rejected_path), checkpoint_rejected)
        write_json(str(summary_path), {"total_examples": len(examples), "accepted": len(checkpoint_accepted), "resumed": True})
        return

    model = str(teacher_config["model"])
    workers = int(teacher_config.get("workers", 4))
    checkpoint_interval = int(teacher_config.get("checkpoint_interval", 50))
    use_rulebook = bool(teacher_config.get("use_rulebook", True))

    logger.info(
        "Teacher distillation: model={}, examples={} (remaining={}), k={}, workers={}",
        model, len(examples), len(remaining), teacher_config.get("k", 3), workers,
    )

    all_accepted = list(checkpoint_accepted)
    all_rejected = list(checkpoint_rejected)
    all_results: list[dict[str, Any]] = []
    completed = len(processed_ids)

    def process_one(example: dict[str, Any]) -> dict[str, Any]:
        return _sample_one_example(
            prompt_cfg, teacher_config,
            text=str(example["text"]),
            gold_label=int(example["label"]),
            example_id=str(example["example_id"]),
            rulebook=rulebook,
            use_rulebook=use_rulebook,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, e): e for e in remaining}
        pbar = tqdm(total=len(remaining), desc="Teacher sampling", unit="example")

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            completed += 1

            if result["accepted"]:
                all_accepted.append(result["accepted"])
            for sample in result["samples"]:
                if sample.get("reject_reason"):
                    all_rejected.append(sample)

            pbar.update(1)

            if completed % checkpoint_interval == 0:
                write_jsonl(str(checkpoint_path), all_accepted)
                write_jsonl(str(checkpoint_rej_path), all_rejected)

        pbar.close()

    write_jsonl(str(accepted_path), all_accepted)
    write_jsonl(str(rejected_path), all_rejected)

    summary = _compute_teacher_summary(all_results, len(examples))
    summary["model"] = model
    write_json(str(summary_path), summary)

    checkpoint_path.unlink(missing_ok=True)
    checkpoint_rej_path.unlink(missing_ok=True)

    logger.info("Teacher distillation complete: accepted {}/{}", len(all_accepted), len(examples))
