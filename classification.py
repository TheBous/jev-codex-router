"""Request classification: TypeSafe/Jev choice with a deterministic heuristic fallback."""

from __future__ import annotations

import logging
from typing import Any, Mapping

LOG = logging.getLogger("codex_router")

_EFFORT_MEANING = {
    "none": "thinking disabled, fastest and cheapest",
    "low": "minimal reasoning, very fast",
    "medium": "balanced reasoning",
    "high": "deep reasoning, slower and more expensive",
    "xhigh": "maximum reasoning effort for the hardest problems",
}


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _request_text(payload: Mapping[str, Any]) -> str:
    values = [payload.get("instructions"), payload.get("input")]
    return " ".join(_flatten_text(value) for value in values if value is not None)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _routing_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    tools = payload.get("tools") or []
    tool_names = [tool.get("name") for tool in tools if isinstance(tool, Mapping) and tool.get("name")]
    return {
        "instructions": payload.get("instructions"),
        "input": payload.get("input"),
        "model_requested": payload.get("model"),
        "tool_names": tool_names,
        "tool_count": len(tools),
        "reasoning": payload.get("reasoning"),
    }


class HeuristicClassifier:
    """Deterministic safety net for missing, slow, or uncertain Jev calls."""

    def classify(self, payload: Mapping[str, Any]) -> tuple[str, float | None]:
        text = _request_text(payload).lower()
        tools = payload.get("tools") or []
        reasoning = payload.get("reasoning") or {}
        effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None

        if effort in {"high", "xhigh", "max", "ultra"}:
            return "REASONING", None
        if _has_any(text, ("prove", "derive", "deep analysis", "trade-off", "tradeoff")):
            return "REASONING", None
        if tools or len(text) > 6000 or _has_any(
            text,
            (
                "apply_patch",
                "multi-step",
                "multi step",
                "refactor",
                "debug",
                "architecture",
                "repository",
                "codebase",
            ),
        ):
            return "COMPLEX", None
        if len(text) <= 500:
            return "SIMPLE", None
        return "MEDIUM", None


class TypeSafeClassifier:
    """Asks Jev two parallel questions: which model, and which reasoning effort."""

    def __init__(self, models: list[Mapping[str, Any]], timeout: float = 1.5) -> None:
        from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

        self._choice = Choice
        self._client_type = TypeSafeClient
        self._retry = RetryPolicy(max_retries=0, timeout=timeout)
        self._timeout = timeout
        self._model_efforts = {model["name"]: list(model["efforts"]) for model in models}
        self._model_options = {
            model["name"]: {"what": model["description"]} if model.get("description") else None
            for model in models
        }

    def _effort_criteria(self) -> dict[str, dict[str, str]]:
        efforts: list[str] = []
        for model_efforts in self._model_efforts.values():
            for effort in model_efforts:
                if effort not in efforts:
                    efforts.append(effort)
        return {effort: {"what": _EFFORT_MEANING.get(effort, f"reasoning effort '{effort}'")} for effort in efforts}

    def _log_answer(self, question: str, answer: Any) -> None:
        choice = getattr(answer, "choice", None)
        confidence = getattr(answer, "confidence", None)
        probabilities = getattr(answer, "probabilities", None) or {}
        top = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
        LOG.info(
            "TypeSafe %s answer choice=%s confidence=%s top=%s",
            question,
            choice,
            f"{float(confidence):.2f}" if confidence is not None else "n/a",
            " ".join(f"{name}={probability:.2f}" for name, probability in top),
        )

    def classify(self, payload: Mapping[str, Any]) -> tuple[str, str, float | None]:
        questions = {
            "model": self._choice(
                instructions={
                    "question": "Which model should handle this request?",
                    "focus": (
                        "Each option describes the model's strengths and cost. "
                        "Prefer the cheapest model capable of the request: simple, "
                        "direct, low-risk requests need a fast, cheap model; tool "
                        "use, multi-step work, repository changes, or difficult "
                        "analysis need the strongest model."
                    ),
                },
                criteria=self._model_options,
            ),
            "effort": self._choice(
                instructions={
                    "question": "Which reasoning effort should the response use?",
                    "focus": (
                        "Choose the effort the request needs, regardless of model: "
                        "none or low for short, direct, low-risk requests; medium "
                        "for ordinary work needing some judgment; high for tool use, "
                        "multi-step work, repository changes, or difficult analysis; "
                        "xhigh only for the hardest problems. If the selected model "
                        "does not support it, the closest supported effort is used."
                    ),
                },
                criteria=self._effort_criteria(),
            ),
        }
        with self._client_type(model="jev-latest", timeout=self._timeout) as client:
            response = client.system_one(
                state=_routing_state(payload),
                questions=questions,
                retry=self._retry,
            )
        model_answer = response.answers["model"]
        self._log_answer("model", model_answer)
        model = getattr(model_answer, "choice", None)
        model_confidence = getattr(model_answer, "confidence", None)
        if model not in self._model_efforts:
            raise ValueError(f"TypeSafe returned an invalid model: {model!r}")

        effort_answer = response.answers["effort"]
        self._log_answer("effort", effort_answer)
        effort = getattr(effort_answer, "choice", None)
        effort_confidence = getattr(effort_answer, "confidence", None)
        if effort not in self._effort_criteria():
            raise ValueError(f"TypeSafe returned an invalid effort: {effort!r}")

        confidences = [float(value) for value in (model_confidence, effort_confidence) if value is not None]
        confidence = min(confidences) if confidences else None
        return model, effort, confidence
