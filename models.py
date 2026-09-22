"""Domain value types shared by classification, routing, and the HTTP boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Route:
    tier: str | None
    provider: str
    model: str
    confidence: float | None
    source: str
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str | None
    wire_api: str
    headers: Mapping[str, str]
    model: str | None
    drop_reasoning_content: bool = False

    @property
    def responses_url(self) -> str:
        return self.base_url.rstrip("/") + "/responses"


def _strip_reasoning_content(input_items: Any) -> Any:
    """Strict OpenAI-schema providers reject reasoning items that carry a content array."""
    if not isinstance(input_items, list):
        return input_items
    cleaned = []
    for item in input_items:
        if isinstance(item, Mapping) and item.get("type") == "reasoning":
            item = {key: value for key, value in item.items() if key != "content"}
        cleaned.append(item)
    return cleaned
