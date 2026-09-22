"""Configuration loading: router JSON config and .env overrides."""

from __future__ import annotations

import json
import os
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dotenv_values(lines: list[str]) -> dict[str, str]:
    values = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_dotenv(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as handle:
            values = _dotenv_values(handle.readlines())
    except FileNotFoundError:
        return
    for key, value in values.items():
        os.environ.setdefault(key, value)
