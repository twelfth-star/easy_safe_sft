from pathlib import Path

from loguru import logger

from easy_safe_sft.tasks import (
    build_reasoning_user_prompt,
    build_sft_user_prompt,
    normalize_answer,
    pick_output_style,
    render_student_output,
)
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
    task_config: dict[str, object],
    raw_rows: list[dict[str, object]],
    student_template: str,
    plain_output_style: str,
    system_prompt: str | None,
) -> list[dict[str, object]]:
    output_style = pick_output_style(plain_output_style, student_template, has_reasoning=False)
    system_prompt = _normalize_system_prompt(system_prompt)
    rows: list[dict[str, object]] = []

    for raw_row in raw_rows:
        final_answer = normalize_answer(task_config, raw_row["answer"])
        output_text = render_student_output(final_answer, reasoning=None, style=output_style)
        user_content = build_sft_user_prompt(task_config, raw_row)
        rows.append({"id": raw_row["id"], "messages": _build_messages(system_prompt, user_content, output_text)})

    return rows


def _build_train_rows_from_trace(
    task_config: dict[str, object],
    trace_rows: list[dict[str, object]],
    student_template: str,
    distill_output_style: str,
    system_prompt: str | None,
) -> list[dict[str, object]]:
    output_style = pick_output_style(distill_output_style, student_template, has_reasoning=True)
    system_prompt = _normalize_system_prompt(system_prompt)
    rows: list[dict[str, object]] = []

    for trace_row in trace_rows:
        output_text = render_student_output(
            final_answer=trace_row["normalized_final_answer"],
            reasoning=trace_row["reasoning"],
            style=output_style,
        )
        user_content = build_reasoning_user_prompt(task_config, trace_row)
        rows.append({"id": trace_row["id"], "messages": _build_messages(system_prompt, user_content, output_text)})

    return rows


def _build_eval_rows_and_meta(
    task_config: dict[str, object],
    raw_rows: list[dict[str, object]],
    system_prompt: str | None,
    prompt_style: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    system_prompt = _normalize_system_prompt(system_prompt)
    lf_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []

    for raw_row in raw_rows:
        normalized_gold_answer = normalize_answer(task_config, raw_row["answer"])
        # 这里故意统一把 valid/test 的 label 保持成 final_only。
        # 原因很简单：原始 valid/test 通常没有 teacher CoT。
        # 如果硬造一份 reasoning label，会把评测搞假。
        # 真正的任务级评测放在训练外部，通过 generated_predictions.jsonl 再解析 FINAL ANSWER。
        output_text = render_student_output(final_answer=normalized_gold_answer, reasoning=None, style="final_only")
        if prompt_style == "reasoning":
            user_content = build_reasoning_user_prompt(task_config, raw_row)
        else:
            user_content = build_sft_user_prompt(task_config, raw_row)
        lf_rows.append({"id": raw_row["id"], "messages": _build_messages(system_prompt, user_content, output_text)})
        meta_rows.append(
            {
                "id": raw_row["id"],
                "split": raw_row["split"],
                "gold_answer": str(raw_row["answer"]),
                "normalized_gold_answer": normalized_gold_answer,
                "fields": raw_row["fields"],
                "meta": raw_row.get("meta", {}),
            }
        )

    return lf_rows, meta_rows


def _write_dataset_info(dataset_dir: str, has_valid: bool, has_test: bool) -> None:
    dataset_info = {
        "train": {
            "file_name": "train.jsonl",
            "formatting": "openai",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }

    if has_valid:
        dataset_info["valid"] = {
            "file_name": "valid.jsonl",
            "formatting": "openai",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    if has_test:
        dataset_info["test"] = {
            "file_name": "test.jsonl",
            "formatting": "openai",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    write_json(Path(dataset_dir) / "dataset_info.json", dataset_info)


def build_dataset(
    task_config: dict[str, object],
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
    logger.info("开始构建 LlamaFactory 数据目录: {}", output_dir_path)

    if train_source == "raw":
        prompt_style = "direct"
        train_rows = _build_train_rows_from_raw(
            task_config=task_config,
            raw_rows=read_jsonl(train_input_path),
            student_template=student_template,
            plain_output_style=plain_output_style,
            system_prompt=system_prompt,
        )
    elif train_source == "trace":
        prompt_style = "reasoning"
        train_rows = _build_train_rows_from_trace(
            task_config=task_config,
            trace_rows=read_jsonl(train_input_path),
            student_template=student_template,
            distill_output_style=distill_output_style,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unsupported train_source: {train_source}")

    write_jsonl(output_dir_path / "train.jsonl", train_rows)
    logger.info("训练集写入完成: {} 条", len(train_rows))

    has_valid = valid_input_path is not None
    has_test = test_input_path is not None

    if valid_input_path is not None:
        valid_rows, valid_meta_rows = _build_eval_rows_and_meta(
            task_config=task_config,
            raw_rows=read_jsonl(valid_input_path),
            system_prompt=system_prompt,
            prompt_style=prompt_style,
        )
        write_jsonl(output_dir_path / "valid.jsonl", valid_rows)
        write_jsonl(output_dir_path / "valid_meta.jsonl", valid_meta_rows)
        logger.info("验证集写入完成: {} 条", len(valid_rows))

    if test_input_path is not None:
        test_rows, test_meta_rows = _build_eval_rows_and_meta(
            task_config=task_config,
            raw_rows=read_jsonl(test_input_path),
            system_prompt=system_prompt,
            prompt_style=prompt_style,
        )
        write_jsonl(output_dir_path / "test.jsonl", test_rows)
        write_jsonl(output_dir_path / "test_meta.jsonl", test_meta_rows)
        logger.info("测试集写入完成: {} 条", len(test_rows))

    _write_dataset_info(str(output_dir_path), has_valid=has_valid, has_test=has_test)
    logger.info("dataset_info.json 写入完成")
