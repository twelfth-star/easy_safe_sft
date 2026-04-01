import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
import yaml


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    assert isinstance(obj, dict)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_PATH_KEYS = {"dataset_dir", "output_dir"}


def resolve_yaml_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Resolve relative paths in *config* against the directory of *config_path*."""
    config_dir = Path(config_path).resolve().parent
    for key in _PATH_KEYS:
        if key in config:
            p = Path(config[key])
            if not p.is_absolute():
                config[key] = str((config_dir / p).resolve())
    return config


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def setup_file_logging(log_dir: str | Path) -> None:
    """Add a loguru file sink to log_dir/run.log (in addition to stderr)."""
    log_path = Path(log_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level=os.getenv("EASY_SAFE_SFT_LOG_LEVEL", "INFO"))
    logger.info("File logging enabled: {}", log_path)


def run_command(cmd: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    logger.info("运行命令: {}", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def prepend_pythonpath(env: dict[str, str], path: str) -> dict[str, str]:
    env = dict(env)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{path}:{old_pythonpath}" if old_pythonpath else path
    return env
