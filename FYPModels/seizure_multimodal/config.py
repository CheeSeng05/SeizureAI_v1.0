from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


TRAINING_DISABLED_MESSAGE = """Training is disabled.

The current configuration uses pretrained VSViG and EEG models
for inference only.

Enable training explicitly only when an appropriate synchronized
vEEG dataset is available."""


class TrainingDisabledError(RuntimeError):
    pass


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return float(value) if any(char in value for char in ".eE") else int(value)
    except ValueError:
        return value


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested mappings used by this project's configs.

    PyYAML is preferred when installed. This fallback intentionally rejects
    lists and advanced YAML rather than ambiguously interpreting them.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in raw_line[:indent] or indent % 2:
            raise ValueError(f"Unsupported YAML indentation on line {line_number}")
        stripped = content.strip()
        if ":" not in stripped or stripped.startswith("-"):
            raise ValueError(f"Unsupported YAML syntax on line {line_number}")
        key, raw_value = stripped.split(":", 1)
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        value = _scalar(raw_value)
        parent[key.strip()] = value
        if isinstance(value, dict):
            stack.append((indent, value))
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError:
        return _minimal_yaml(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def assert_training_enabled(config: Mapping[str, Any]) -> None:
    training = config.get("training", {})
    enabled = isinstance(training, Mapping) and training.get("enabled") is True
    if not enabled:
        raise TrainingDisabledError(TRAINING_DISABLED_MESSAGE)


def validate_training_policy(config: Mapping[str, Any], *, paired_data_verified: bool) -> None:
    assert_training_enabled(config)
    fusion = config.get("fusion", {})
    if not isinstance(fusion, Mapping) or fusion.get("trainable") is not True:
        raise ValueError("Training is enabled but fusion.trainable is not true")
    if not paired_data_verified:
        raise ValueError(
            "Training requires verified synchronized video and EEG from the same "
            "patient, recording, and temporal interval."
        )
