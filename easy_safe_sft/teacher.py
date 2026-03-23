from typing import Any

from litellm import completion
from loguru import logger

from easy_safe_sft.tasks import build_reasoning_user_prompt, extract_final_answer, extract_reasoning, normalize_answer
from easy_safe_sft.utils import read_jsonl, write_jsonl


def _build_messages(teacher_config: dict[str, Any], user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if teacher_config.get("system_prompt"):
        messages.append({"role": "system", "content": teacher_config["system_prompt"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def distill_dataset(task_config: dict[str, Any], teacher_config: dict[str, Any], input_path: str, output_path: str) -> None:
    rows = read_jsonl(input_path)
    distilled_rows: list[dict[str, Any]] = []
    answer_prefix = task_config.get("answer_prefix", "FINAL ANSWER:")
    logger.info("开始 teacher distillation: model={}, input={}", teacher_config["model"], input_path)

    total = 0
    kept = 0

    for row in rows:
        total += 1
        user_prompt = build_reasoning_user_prompt(task_config, row)
        response = completion(
            model=teacher_config["model"],
            messages=_build_messages(teacher_config, user_prompt),
            temperature=teacher_config.get("temperature", 0.0),
            max_tokens=teacher_config.get("max_tokens", 512),
            **teacher_config.get("completion_kwargs", {}),
        )
        raw_completion = response["choices"][0]["message"]["content"]
        normalized_gold_answer = normalize_answer(task_config, row["answer"])

        try:
            reasoning = extract_reasoning(raw_completion, answer_prefix)
            final_answer = extract_final_answer(raw_completion, "reasoning_text", answer_prefix)
            normalized_final_answer = normalize_answer(task_config, final_answer)
        except Exception as e:
            logger.warning("Teacher 输出解析失败，跳过: id={}, error={}", row["id"], e)
            continue

        if normalized_final_answer == normalized_gold_answer:
            kept += 1
            distilled_rows.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "task_name": task_config["name"],
                    "fields": row["fields"],
                    "gold_answer": str(row["answer"]),
                    "normalized_gold_answer": normalized_gold_answer,
                    "teacher_model": teacher_config["model"],
                    "user_prompt": user_prompt,
                    "raw_completion": raw_completion,
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                    "normalized_final_answer": normalized_final_answer,
                    "is_correct": True,
                    "meta": row.get("meta", {}),
                }
            )

    write_jsonl(output_path, distilled_rows)
    logger.info("Teacher distillation 完成: 保留 {}/{} 条正确样本", kept, total)
