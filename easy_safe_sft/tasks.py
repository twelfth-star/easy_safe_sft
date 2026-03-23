from typing import Any


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
        if text in label_map:
            return label_map[text]

        for canonical in label_map.values():
            if text.casefold() == canonical.casefold():
                return canonical

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

    return "reasoning_text"


def render_student_output(final_answer: str, reasoning: str | None, style: str) -> str:
    if style == "final_only":
        return final_answer

    if style == "reasoning_text":
        assert reasoning is not None
        return f"REASONING:\n{reasoning.strip()}\n\nFINAL ANSWER: {final_answer}"

    if style == "reasoning_qwen":
        assert reasoning is not None
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{final_answer}"

    raise ValueError(f"Unsupported output style: {style}")


def is_correct(task_config: dict[str, Any], predicted_answer: str, gold_answer: Any) -> bool:
    normalized_pred = normalize_answer(task_config, predicted_answer)
    normalized_gold = normalize_answer(task_config, gold_answer)
    return normalized_pred == normalized_gold
