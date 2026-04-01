import json
import re
from typing import Any

_POSITIVE_ALIASES = frozenset({"true", "positive", "yes", "1"})
_NEGATIVE_ALIASES = frozenset({"false", "negative", "no", "0"})


def render_input_text(task_config: dict[str, Any], example: dict[str, Any]) -> str:
    # 这里故意只支持两个最常见的任务族：
    # 1. 二分类
    # 2. 多选题
    # 后面如果要扩展，就继续在这里加分支。
    task_type = task_config["task_type"]
    fields = example["fields"]

    if task_type == "binary_classification":
        return str(fields[task_config["input_field"]]).strip()

    if task_type == "multiple_choice":
        question = str(fields[task_config["question_field"]]).strip()
        choices = fields[task_config["choices_field"]]
        choice_labels = task_config["choice_labels"]
        assert len(choices) == len(choice_labels)
        choice_lines = [f"{label}. {text}" for label, text in zip(choice_labels, choices)]
        return question + "\n\nChoices:\n" + "\n".join(choice_lines)

    raise ValueError(f"Unsupported task_type: {task_type}")


def _prompt_context(task_config: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    fields = dict(example["fields"])
    context: dict[str, Any] = dict(fields)
    context["input_text"] = render_input_text(task_config, example)
    context["rendered_input"] = context["input_text"]

    if task_config["task_type"] == "multiple_choice":
        choices = fields[task_config["choices_field"]]
        choice_labels = task_config["choice_labels"]
        context["choices_text"] = "\n".join(
            f"{label}. {text}" for label, text in zip(choice_labels, choices)
        )

    return context


def render_prompt_template(template: str, task_config: dict[str, Any], example: dict[str, Any]) -> str:
    try:
        return template.strip().format_map(_prompt_context(task_config, example))
    except KeyError as e:
        missing_key = e.args[0]
        raise ValueError(f"Prompt template references missing field: {missing_key}") from e


def build_sft_user_prompt(task_config: dict[str, Any], example: dict[str, Any]) -> str:
    return render_prompt_template(task_config["sft_instruction"], task_config, example)


def build_reasoning_user_prompt(task_config: dict[str, Any], example: dict[str, Any]) -> str:
    return render_prompt_template(task_config["reasoning_instruction"], task_config, example)


def normalize_answer(task_config: dict[str, Any], answer: Any) -> str:
    """
    这里的规范化主要是为了二分类和多选题的答案对齐。
        - 二分类：根据 label_map 规范化用户输出，允许大小写不敏
            感的匹配。例如，用户输出 "Yes"、"yes"、"YES" 都会被规范化成 "Yes"。
        - 多选题：允许用户输出完整文本或者选项标签，且大小写不敏感。例如，如果选
            项是 "A. Apple"，用户输出 "A"、"a"、"A. Apple" 都会被规范化成 "A"。
    对于其他类型的任务，目前不做规范化，直接返回字符串形式。
    """
    task_type = task_config["task_type"]

    if task_type == "binary_classification":
        label_map = task_config["label_map"]
        text = str(answer).strip()
        # Direct key match: "0" → label_map["0"], "1" → label_map["1"]
        if text in label_map:
            return label_map[text]
        # Case-insensitive match against canonical values: "yes" → "Yes"
        for canonical in label_map.values():
            if text.casefold() == canonical.casefold():
                return canonical
        lowered = text.casefold()
        if lowered in _POSITIVE_ALIASES and "1" in label_map:
            return label_map["1"]
        if lowered in _NEGATIVE_ALIASES and "0" in label_map:
            return label_map["0"]

        raise ValueError(f"Cannot normalize binary answer: {answer}")

    if task_type == "multiple_choice":
        choice_labels = task_config["choice_labels"]
        text = str(answer).strip().upper()
        if text in choice_labels:
            return text
        if text and text[0] in choice_labels:
            return text[0]
        raise ValueError(f"Cannot normalize multiple choice answer: {answer}")

    raise ValueError(f"Unsupported task_type: {task_type}")


def extract_reasoning(text: str, answer_prefix: str) -> str:
    # 这里故意严格要求 teacher 输出格式正确。
    # 如果 teacher 没按约定输出，就直接暴露问题。
    before_answer = text.rsplit(answer_prefix, 1)[0]
    return before_answer.split("REASONING:", 1)[1].strip()


def extract_final_answer(text: str, style: str, answer_prefix: str) -> str:
    if style == "final_only":
        return text.strip()

    if style == "reasoning_text":
        return text.rsplit(answer_prefix, 1)[1].strip()

    if style == "reasoning_qwen":
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text.strip()

    if style == "reasoning_label":
        if "LABEL:" in text:
            return text.rsplit("LABEL:", 1)[1].strip().split("\n")[0].strip()
        return text.strip()

    raise ValueError(f"Unsupported prediction style: {style}")


def pick_output_style(style: str, student_template: str, has_reasoning: bool) -> str:
    # 这里让 style 的选择保持很直接：
    # - 用户显式指定，就用用户的。
    # - auto 时，只看 student template 和有没有 reasoning。
    if style != "auto":
        return style

    if not has_reasoning:
        return "final_only"

    if student_template in {"qwen3", "qwen3_5"}:
        return "reasoning_qwen"

    # qwen3_nothink, qwen3_5_nothink, and any other non-thinking template
    return "reasoning_label"


def render_student_output(final_answer: str, reasoning: str | None, style: str) -> str:
    if style == "final_only":
        return final_answer

    if style == "reasoning_text":
        assert reasoning is not None
        return f"REASONING:\n{reasoning.strip()}\n\nFINAL ANSWER: {final_answer}"

    if style == "reasoning_qwen":
        assert reasoning is not None
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{final_answer}"

    if style == "reasoning_label":
        assert reasoning is not None
        return f"REASONING:\n{reasoning.strip()}\n\nLABEL: {final_answer}"

    raise ValueError(f"Unsupported output style: {style}")


def is_correct(task_config: dict[str, Any], predicted_answer: str, gold_answer: Any) -> bool:
    normalized_pred = normalize_answer(task_config, predicted_answer)
    normalized_gold = normalize_answer(task_config, gold_answer)
    return normalized_pred == normalized_gold


# ---------------------------------------------------------------------------
# Teacher output parsers (selected by output_format, no fallback)
# ---------------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    """Remove markdown code fence wrapping (```json ... ```)."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```\w*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def parse_reasoning_text(raw: str, answer_prefix: str) -> tuple[str | None, str | None]:
    """Parse REASONING:...\\nFINAL ANSWER:... format."""
    if "REASONING:" not in raw or answer_prefix not in raw:
        return None, None
    try:
        before_answer = raw.rsplit(answer_prefix, 1)[0]
        reasoning = before_answer.split("REASONING:", 1)[1].strip()
        answer = raw.rsplit(answer_prefix, 1)[1].strip()
        return reasoning, answer
    except (IndexError, ValueError):
        return None, None


def parse_think_label_xml(raw: str) -> tuple[str | None, str | None]:
    """Parse <think>...</think>\\n<label>...</label> format."""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    label_match = re.search(r"<label>\s*(.*?)\s*</label>", raw, re.DOTALL)
    if think_match is None or label_match is None:
        return None, None
    return think_match.group(1).strip(), label_match.group(1).strip()


def parse_reasoning_label(raw: str) -> tuple[str | None, str | None]:
    """Parse REASONING:\\n...\\nLABEL: ... format. Label takes only the first line after LABEL:."""
    if "REASONING:" not in raw or "LABEL:" not in raw:
        return None, None
    try:
        reasoning = raw.split("REASONING:", 1)[1].rsplit("LABEL:", 1)[0].strip()
        label = raw.rsplit("LABEL:", 1)[1].strip().split("\n")[0].strip()
        return reasoning, label
    except (IndexError, ValueError):
        return None, None


def parse_json_output(raw: str) -> tuple[str | None, str | None]:
    """Parse JSON with reasoning/label keys."""
    try:
        obj = json.loads(strip_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    reasoning = obj.get("reasoning") or obj.get("thinking") or obj.get("rationale")
    label = obj.get("label")
    if label is None:
        label = obj.get("prediction")
    if label is None:
        label = obj.get("answer")
    if label is None:
        return None, None
    return str(reasoning or "").strip() or None, str(label).strip()


def parse_teacher_output(
    raw: str,
    output_format: str,
    answer_prefix: str = "FINAL ANSWER:",
) -> tuple[str | None, str | None, str | None]:
    """Parse teacher output using the specified format.

    Returns (reasoning, answer, error). On success error is None.
    """
    if output_format == "reasoning_text":
        reasoning, answer = parse_reasoning_text(raw, answer_prefix)
    elif output_format == "think_label_xml":
        reasoning, answer = parse_think_label_xml(raw)
    elif output_format == "reasoning_label":
        reasoning, answer = parse_reasoning_label(raw)
    elif output_format == "json":
        reasoning, answer = parse_json_output(raw)
    else:
        return None, None, f"unknown_output_format:{output_format}"

    if answer is None:
        return None, None, "parse_failed"
    return reasoning, answer, None
