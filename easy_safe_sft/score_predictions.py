#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from easy_safe_sft.tasks import extract_final_answer, normalize_answer, pick_output_style
from easy_safe_sft.utils import read_jsonl, write_json, write_jsonl


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def macro_f1_from_confusion(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute macro F1 from confusion matrix (average of positive and negative F1)."""
    f1_pos = _safe_div(2.0 * tp, 2 * tp + fp + fn)
    f1_neg = _safe_div(2.0 * tn, 2 * tn + fp + fn)
    return (f1_pos + f1_neg) / 2.0


def _binary_metrics(task_config: dict[str, Any], predictions: list[str], golds: list[str]) -> dict[str, float]:
    positive_label = task_config["positive_label"]
    tp = fp = fn = tn = 0

    for pred, gold in zip(predictions, golds):
        if pred == positive_label and gold == positive_label:
            tp += 1
        elif pred == positive_label and gold != positive_label:
            fp += 1
        elif pred != positive_label and gold == positive_label:
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    recall_neg = _safe_div(tn, tn + fp)
    balanced_accuracy = (recall + recall_neg) / 2.0

    # Matthews Correlation Coefficient
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = _safe_div(tp * tn - fp * fn, denom_sq ** 0.5) if denom_sq > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1_from_confusion(tp, tn, fp, fn),
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_predictions(
    task_config: dict[str, object],
    student_template: str,
    prediction_path: str,
    meta_path: str,
    summary_output_path: str,
    details_output_path: str | None = None,
    prediction_style: str = "auto",
    has_reasoning: bool = False,
) -> None:
    logger.info("开始离线评测: predictions={}, meta={}", prediction_path, meta_path)
    raw_prediction_rows = read_jsonl(prediction_path)
    meta_rows = read_jsonl(meta_path)
    assert len(raw_prediction_rows) == len(meta_rows)

    style = pick_output_style(prediction_style, student_template, has_reasoning=has_reasoning)
    answer_prefix = task_config.get("answer_prefix", "FINAL ANSWER:")

    total = 0
    correct = 0
    invalid_count = 0
    normalized_predictions: list[str] = []
    normalized_golds: list[str] = []
    detail_rows: list[dict[str, object]] = []

    for prediction_row, meta_row in zip(raw_prediction_rows, meta_rows):
        total += 1
        raw_prediction = prediction_row["predict"]
        normalized_gold = meta_row["normalized_gold_answer"]
        parsed_answer = ""
        try:
            parsed_answer = extract_final_answer(raw_prediction, style, answer_prefix)
            normalized_prediction = normalize_answer(task_config, parsed_answer)
        except Exception as error:
            logger.warning("预测解析失败，计为错误: id={}, error={}", meta_row["id"], error)
            normalized_prediction = ""
            invalid_count += 1
        row_correct = normalized_prediction == normalized_gold
        if row_correct:
            correct += 1

        normalized_predictions.append(normalized_prediction)
        normalized_golds.append(normalized_gold)
        detail_rows.append(
            {
                "id": meta_row["id"],
                "split": meta_row["split"],
                "predict_text": raw_prediction,
                "parsed_answer": parsed_answer,
                "normalized_prediction": normalized_prediction,
                "normalized_gold_answer": normalized_gold,
                "is_correct": row_correct,
            }
        )

    summary = {
        "task_name": task_config["name"],
        "prediction_style": style,
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / total if total else 0.0,
    }
    if task_config["task_type"] == "binary_classification":
        summary.update(_binary_metrics(task_config, normalized_predictions, normalized_golds))

    write_json(summary_output_path, summary)
    if details_output_path is not None:
        write_jsonl(details_output_path, detail_rows)
    logger.info("离线评测完成: {}", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score model predictions against gold answers")
    parser.add_argument("--prompt-config", required=True, help="Unified prompt+task YAML")
    parser.add_argument("--student-template", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--details-output")
    parser.add_argument("--prediction-style", default="auto")
    parser.add_argument("--has-reasoning", action="store_true")
    args = parser.parse_args()

    from easy_safe_sft.prompt_config import load_prompt_config

    evaluate_predictions(
        task_config=load_prompt_config(args.prompt_config).as_task_config,
        student_template=args.student_template,
        prediction_path=args.predictions,
        meta_path=args.meta,
        summary_output_path=args.summary_output,
        details_output_path=args.details_output,
        prediction_style=args.prediction_style,
        has_reasoning=args.has_reasoning,
    )


if __name__ == "__main__":
    main()
