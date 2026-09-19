from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration {path} must contain a mapping")
    return value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def normalize_sparniche_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a validated copy of a single-view SparNiche configuration."""
    result = copy.deepcopy(config)
    model = result.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model configuration must be a mapping")
    evaluation = result.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation configuration must be a mapping")
    return result


def apply_overrides(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        keys = str(dotted_key).split(".")
        if not all(keys):
            raise ValueError(f"invalid override key: {dotted_key!r}")
        target = result
        for key in keys[:-1]:
            existing = target.setdefault(key, {})
            if not isinstance(existing, dict):
                raise ValueError(f"cannot descend through non-mapping override key {key!r}")
            target = existing
        target[keys[-1]] = copy.deepcopy(value)
    return result


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"required environment variable {name} is not set")
        return os.environ[name]

    return ENVIRONMENT_PATTERN.sub(replace, value)


def save_yaml(value: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=False)
