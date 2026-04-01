"""Canonical dataset loading.

Ported from tinker-to-gpu/vml/dataset_contract.py.
Supports the canonical dataset format: datasets/{name}/{version}/ with
processed.csv, metadata.json, and splits/{policy}.csv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CanonicalDatasetSpec:
    dataset_name: str
    dataset_version: str
    split_policy: str
    dataset_root: Path
    dataset_dir: Path
    processed_path: Path
    metadata_path: Path
    split_path: Path
    id_column: str
    text_column: str
    label_column: str
    csv_sep: str | None


def parse_binary_label(value: Any, *, threshold: float = 0.5) -> int | None:
    """Convert various label formats to binary 0/1."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in {0, 1} else None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value in {0.0, 1.0}:
            return int(value)
        return 1 if value >= threshold else 0

    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"0", "no", "false", "negative"}:
        return 0
    if lowered in {"1", "yes", "true", "positive"}:
        return 1

    try:
        numeric = float(raw)
    except ValueError:
        return None
    if numeric in {0.0, 1.0}:
        return int(numeric)
    return 1 if numeric >= threshold else 0


def _role_matches(actual_role: Any, expected_role: str) -> bool:
    role = str(actual_role or "").strip().lower()
    expected = str(expected_role).strip().lower()
    if not role or role == "unused":
        return False
    return role == expected or role.startswith(f"{expected}_")


def resolve_canonical_dataset_spec(
    *,
    dataset_root: Path,
    dataset_name: str,
    dataset_version: str,
    split_policy: str,
) -> CanonicalDatasetSpec:
    """Resolve paths and read metadata for a canonical dataset."""
    dataset_name = str(dataset_name).strip()
    dataset_version = str(dataset_version).strip()
    split_policy_name = str(split_policy or "policy_v1.csv").strip()
    if not split_policy_name.endswith(".csv"):
        split_policy_name = f"{split_policy_name}.csv"

    dataset_dir = dataset_root / dataset_name / dataset_version
    processed_path = dataset_dir / "processed.csv"
    metadata_path = dataset_dir / "metadata.json"
    split_path = dataset_dir / "splits" / split_policy_name

    for path, label in [
        (dataset_dir, "directory"),
        (processed_path, "processed data"),
        (metadata_path, "metadata"),
        (split_path, "split assignment"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical {label}: {path}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return CanonicalDatasetSpec(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        split_policy=split_policy_name,
        dataset_root=dataset_root,
        dataset_dir=dataset_dir,
        processed_path=processed_path,
        metadata_path=metadata_path,
        split_path=split_path,
        id_column=str(metadata.get("id_column", "example_id")).strip() or "example_id",
        text_column=str(metadata.get("text_column") or "text").strip(),
        label_column=str(metadata.get("label_column") or "label").strip(),
        csv_sep=str(metadata["csv_sep"]) if "csv_sep" in metadata else None,
    )


def load_canonical_splits(
    spec: CanonicalDatasetSpec,
    *,
    phase_name: str,
    label_threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Load train/val/test splits from a canonical dataset.

    Returns dict with keys "train", "val", "test", each a DataFrame
    with columns: example_id, text, label (int 0/1).
    """
    role_column = f"{phase_name}_role"

    split_frame = pd.read_csv(spec.split_path, encoding="utf-8", encoding_errors="replace")
    if "example_id" not in split_frame.columns:
        raise KeyError(f"Split table missing 'example_id': {spec.split_path}")
    if role_column not in split_frame.columns:
        raise KeyError(f"Split table missing '{role_column}': {spec.split_path}")

    sep = spec.csv_sep or ","
    try:
        processed = pd.read_csv(spec.processed_path, sep=sep, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    except Exception:
        processed = pd.read_csv(spec.processed_path, sep=";", encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")

    missing = {c for c in (spec.id_column, spec.text_column, spec.label_column) if c not in processed.columns}
    if missing:
        raise KeyError(f"Processed data {spec.processed_path} missing columns: {sorted(missing)}")

    processed = processed[[spec.id_column, spec.text_column, spec.label_column]].copy()
    processed = processed.dropna(subset=[spec.id_column, spec.text_column, spec.label_column])
    processed["example_id"] = processed[spec.id_column].astype(str).str.strip()
    processed["text"] = processed[spec.text_column].astype(str).str.strip()
    processed["label"] = processed[spec.label_column].apply(lambda v: parse_binary_label(v, threshold=label_threshold))
    processed = processed.dropna(subset=["example_id", "text", "label"])
    processed = processed[processed["example_id"] != ""]
    processed["label"] = processed["label"].astype(int)

    assignment = split_frame[["example_id", role_column]].copy()
    assignment["example_id"] = assignment["example_id"].astype(str).str.strip()
    assignment[role_column] = assignment[role_column].fillna("").astype(str).str.strip()

    merged = processed.merge(assignment, on="example_id", how="inner", validate="one_to_one")

    frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        mask = merged[role_column].map(lambda role, s=split_name: _role_matches(role, s))
        frames[split_name] = merged.loc[mask, ["example_id", "text", "label"]].reset_index(drop=True)

    return frames
