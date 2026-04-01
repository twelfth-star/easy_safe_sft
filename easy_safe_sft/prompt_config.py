"""Unified prompt and task configuration.

Replaces the old task.yaml with a single YAML that contains both task
definition (name, type, label_map) and prompt templates (teacher, inference,
SFT label-only).  Also provides rulebook loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from easy_safe_sft.utils import load_yaml


@dataclass
class PromptConfig:
    # Task definition (previously in task.yaml)
    task_name: str
    task_type: str
    label_map: dict[str, str]
    positive_label: str
    answer_prefix: str = "FINAL ANSWER:"

    # Prompt templates
    task_instruction: str = ""
    teacher_system_prompt: str = "You are a careful binary classifier. Follow the user's instructions exactly."
    reasoning_user_prompt_template: str = ""
    reasoning_user_prompt_template_no_rulebook: str = ""
    label_only_user_prompt_template: str = ""
    inference_user_prompt_template: str = ""
    inference_user_prompt_template_no_rulebook: str = ""

    @property
    def as_task_config(self) -> dict[str, Any]:
        """Return a dict compatible with legacy task_config-based functions."""
        return {
            "name": self.task_name,
            "task_type": self.task_type,
            "label_map": self.label_map,
            "positive_label": self.positive_label,
            "answer_prefix": self.answer_prefix,
        }


def load_prompt_config(prompt_file: str | Path) -> PromptConfig:
    """Load a unified prompt+task config from YAML."""
    raw = load_yaml(prompt_file)

    return PromptConfig(
        task_name=str(raw["task_name"]),
        task_type=str(raw["task_type"]),
        label_map=dict(raw["label_map"]),
        positive_label=str(raw["positive_label"]),
        answer_prefix=str(raw.get("answer_prefix", "FINAL ANSWER:")),
        task_instruction=str(raw.get("task_instruction", "")),
        teacher_system_prompt=str(raw.get("teacher_system_prompt", PromptConfig.teacher_system_prompt)),
        reasoning_user_prompt_template=str(raw.get("reasoning_user_prompt_template", "")),
        reasoning_user_prompt_template_no_rulebook=str(raw.get("reasoning_user_prompt_template_no_rulebook", "")),
        label_only_user_prompt_template=str(raw.get("label_only_user_prompt_template", "")),
        inference_user_prompt_template=str(raw.get("inference_user_prompt_template", "")),
        inference_user_prompt_template_no_rulebook=str(raw.get("inference_user_prompt_template_no_rulebook", "")),
    )


def read_rulebook(rules_dir: str | Path, pattern: str = "*.txt") -> str:
    """Read and concatenate all rule files matching a glob pattern.

    Returns concatenated rulebook text with [RULE FILE: name] headers.
    Returns empty string if rules_dir does not exist or has no matching files.
    """
    rules_path = Path(rules_dir)
    if not rules_path.exists():
        return ""

    chunks: list[str] = []
    for rule_file in sorted(rules_path.glob(pattern)):
        if not rule_file.is_file():
            continue
        content = rule_file.read_text(encoding="utf-8").strip()
        if content:
            chunks.append(f"[RULE FILE: {rule_file.name}]\n{content}")

    return "\n\n".join(chunks)


def build_teacher_messages(
    prompt_cfg: PromptConfig,
    text: str,
    *,
    rulebook: str = "",
    use_rulebook: bool = True,
) -> list[dict[str, str]]:
    """Build message list for teacher API call."""
    if use_rulebook and rulebook:
        template = prompt_cfg.reasoning_user_prompt_template
    else:
        template = prompt_cfg.reasoning_user_prompt_template_no_rulebook

    user_content = template.format(
        task_instruction=prompt_cfg.task_instruction,
        rulebook=rulebook,
        text=text,
    )

    return [
        {"role": "system", "content": prompt_cfg.teacher_system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_sft_user_prompt(prompt_cfg: PromptConfig, text: str, *, rulebook: str = "") -> str:
    """Build user prompt for direct SFT (label-only, no reasoning)."""
    return prompt_cfg.label_only_user_prompt_template.format(
        task_instruction=prompt_cfg.task_instruction,
        rulebook=rulebook,
        text=text,
    )


def build_inference_user_prompt(
    prompt_cfg: PromptConfig,
    text: str,
    *,
    rulebook: str = "",
    use_rulebook: bool = True,
) -> str:
    """Build user prompt for student inference."""
    if use_rulebook and rulebook:
        template = prompt_cfg.inference_user_prompt_template
    else:
        template = prompt_cfg.inference_user_prompt_template_no_rulebook

    return template.format(
        task_instruction=prompt_cfg.task_instruction,
        rulebook=rulebook,
        text=text,
    )
