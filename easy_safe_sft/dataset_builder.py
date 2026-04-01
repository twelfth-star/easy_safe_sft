"""Convert raw data or teacher traces to LlamaFactory training format."""

from pathlib import Path
from typing import Any

from loguru import logger

from easy_safe_sft.prompt_config import PromptConfig, build_sft_user_prompt, build_inference_user_prompt
from easy_safe_sft.tasks import normalize_answer, pick_output_style, render_student_output
from easy_safe_sft.utils import read_jsonl, write_json, write_jsonl

DEFAULT_SYSTEM_PROMPT = "detailed thinking off"


def _normalize_system_prompt(system_prompt: str | None) -> str:
    if system_prompt is None:
        return DEFAULT_SYSTEM_PROMPT
    return system_prompt


def _build_messages(system_prompt: str | None, user_content: str, output_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output_text})
    return messages


def _build_train_rows_from_raw(
    prompt_cfg: PromptConfig,
    raw_rows: list[dict[str, Any]],
    student_template: str,
    plain_output_style: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build training rows for direct SFT (no reasoning)."""
    task_config = prompt_cfg.as_task_config
    output_style = pick_output_style(plain_output_style, student_template, has_reasoning=False)
    system_prompt = _normalize_system_prompt(system_prompt)
    rows: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        text = str(raw_row.get("text") or raw_row.get("fields", {}).get("text", "")).strip()
        final_answer = normalize_answer(task_config, raw_row.get("answer") or raw_row.get("label"))
        output_text = render_student_output(final_answer, reasoning=None, style=output_style)
        user_content = build_sft_user_prompt(prompt_cfg, text)
        row_id = raw_row.get("id") or raw_row.get("example_id", "")
        rows.append({"id": row_id, "messages": _build_messages(system_prompt, user_content, output_text)})

    return rows


def _build_train_rows_from_trace(
    prompt_cfg: PromptConfig,
    trace_rows: list[dict[str, Any]],
    student_template: str,
    distill_output_style: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build training rows from teacher distillation traces."""
    task_config = prompt_cfg.as_task_config
    output_style = pick_output_style(distill_output_style, student_template, has_reasoning=True)
    system_prompt = _normalize_system_prompt(system_prompt)
    rows: list[dict[str, Any]] = []

    for trace_row in trace_rows:
        text = str(trace_row.get("text", "")).strip()
        # Support both old schema ("normalized_final_answer") and new schema ("label" as int)
        raw_label = trace_row.get("normalized_final_answer") or trace_row.get("label")
        final_answer = normalize_answer(task_config, raw_label)
        reasoning = trace_row.get("reasoning", "")
        output_text = render_student_output(final_answer, reasoning=reasoning, style=output_style)
        user_content = build_inference_user_prompt(prompt_cfg, text, use_rulebook=False)
        row_id = trace_row.get("id") or trace_row.get("example_id", "")
        rows.append({"id": row_id, "messages": _build_messages(system_prompt, user_content, output_text)})

    return rows


def _build_eval_rows_and_meta(
    prompt_cfg: PromptConfig,
    raw_rows: list[dict[str, Any]],
    system_prompt: str | None,
    prompt_style: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_config = prompt_cfg.as_task_config
    system_prompt = _normalize_system_prompt(system_prompt)
    lf_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for raw_row in raw_rows:
        text = str(raw_row.get("text") or raw_row.get("fields", {}).get("text", "")).strip()
        raw_label = raw_row.get("answer") or raw_row.get("label")
        normalized_gold_answer = normalize_answer(task_config, raw_label)
        output_text = render_student_output(final_answer=normalized_gold_answer, reasoning=None, style="final_only")
        if prompt_style == "reasoning":
            user_content = build_inference_user_prompt(prompt_cfg, text, use_rulebook=False)
        else:
            user_content = build_sft_user_prompt(prompt_cfg, text)
        row_id = raw_row.get("id") or raw_row.get("example_id", "")
        lf_rows.append({"id": row_id, "messages": _build_messages(system_prompt, user_content, output_text)})
        meta_rows.append(
            {
                "id": row_id,
                "split": raw_row.get("split", ""),
                "gold_answer": str(raw_label),
                "normalized_gold_answer": normalized_gold_answer,
                "fields": raw_row.get("fields", {"text": text}),
                "meta": raw_row.get("meta", {}),
            }
        )

    return lf_rows, meta_rows


def _write_dataset_info(dataset_dir: str, has_valid: bool, has_test: bool) -> None:
    split_template = {
        "formatting": "openai",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    dataset_info: dict[str, Any] = {"train": {"file_name": "train.jsonl", **split_template}}
    if has_valid:
        dataset_info["valid"] = {"file_name": "valid.jsonl", **split_template}
    if has_test:
        dataset_info["test"] = {"file_name": "test.jsonl", **split_template}
    write_json(Path(dataset_dir) / "dataset_info.json", dataset_info)


def build_dataset(
    prompt_cfg: PromptConfig,
    train_input_path: str,
    train_source: str,
    output_dir: str,
    student_template: str,
    plain_output_style: str = "final_only",
    distill_output_style: str = "auto",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    valid_input_path: str | None = None,
    test_input_path: str | None = None,
) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("Building LlamaFactory dataset: {}", output_dir_path)

    if train_source == "raw":
        prompt_style = "direct"
        train_rows = _build_train_rows_from_raw(
            prompt_cfg=prompt_cfg,
            raw_rows=read_jsonl(train_input_path),
            student_template=student_template,
            plain_output_style=plain_output_style,
            system_prompt=system_prompt,
        )
    elif train_source == "trace":
        prompt_style = "reasoning"
        train_rows = _build_train_rows_from_trace(
            prompt_cfg=prompt_cfg,
            trace_rows=read_jsonl(train_input_path),
            student_template=student_template,
            distill_output_style=distill_output_style,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unsupported train_source: {train_source}")

    write_jsonl(output_dir_path / "train.jsonl", train_rows)
    logger.info("Train set written: {} rows", len(train_rows))

    has_valid = valid_input_path is not None
    has_test = test_input_path is not None

    if valid_input_path is not None:
        valid_rows, valid_meta_rows = _build_eval_rows_and_meta(
            prompt_cfg=prompt_cfg,
            raw_rows=read_jsonl(valid_input_path),
            system_prompt=system_prompt,
            prompt_style=prompt_style,
        )
        write_jsonl(output_dir_path / "valid.jsonl", valid_rows)
        write_jsonl(output_dir_path / "valid_meta.jsonl", valid_meta_rows)
        logger.info("Valid set written: {} rows", len(valid_rows))

    if test_input_path is not None:
        test_rows, test_meta_rows = _build_eval_rows_and_meta(
            prompt_cfg=prompt_cfg,
            raw_rows=read_jsonl(test_input_path),
            system_prompt=system_prompt,
            prompt_style=prompt_style,
        )
        write_jsonl(output_dir_path / "test.jsonl", test_rows)
        write_jsonl(output_dir_path / "test_meta.jsonl", test_meta_rows)
        logger.info("Test set written: {} rows", len(test_rows))

    _write_dataset_info(str(output_dir_path), has_valid=has_valid, has_test=has_test)
    logger.info("dataset_info.json written")
