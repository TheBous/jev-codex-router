#!/usr/bin/env python3
"""Minimal OpenAI Responses API proxy with TypeSafe-assisted routing."""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


LOG = logging.getLogger("codex_router")


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

    @property
    def responses_url(self) -> str:
        return self.base_url.rstrip("/") + "/responses"


class SessionAffinity:
    def __init__(self, maximum: int = 10000) -> None:
        self.maximum = maximum
        self._routes: OrderedDict[str, Route] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str | None) -> Route | None:
        if not key:
            return None
        with self._lock:
            route = self._routes.get(key)
            if route:
                self._routes.move_to_end(key)
            return route

    def put(self, key: str | None, route: Route) -> None:
        if not key:
            return
        with self._lock:
            self._routes[key] = route
            self._routes.move_to_end(key)
            while len(self._routes) > self.maximum:
                self._routes.popitem(last=False)


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


_EFFORT_MEANING = {
    "none": "thinking disabled, fastest and cheapest",
    "low": "minimal reasoning, very fast",
    "medium": "balanced reasoning",
    "high": "deep reasoning, slower and more expensive",
    "xhigh": "maximum reasoning effort for the hardest problems",
}


class TypeSafeClassifier:
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


class Router:
    def __init__(self, config: Mapping[str, Any], classifier: Any | None = None) -> None:
        self.providers = {
            name: Provider(
                name=name,
                base_url=provider["base_url"],
                api_key_env=provider.get("api_key_env"),
                wire_api=provider.get("wire_api", "responses"),
                headers=provider.get("headers", {}),
                model=provider.get("model"),
            )
            for name, provider in config["providers"].items()
        }
        self.tiers = config["tiers"]
        self.models = {model["name"]: model for model in config.get("models", [])}
        self.session_affinity = bool(config.get("session_affinity", True))
        self.affinity = SessionAffinity(int(config.get("session_max", 10000)))
        self.heuristic = HeuristicClassifier()
        self.classifier = classifier
        if self.classifier is None and self.models and os.getenv("TYPESAFE_API_KEY"):
            try:
                self.classifier = TypeSafeClassifier(
                    list(self.models.values()),
                    timeout=float(config.get("typesafe_timeout_seconds", 1.5)),
                )
            except Exception:
                LOG.exception("TypeSafe classifier unavailable; using heuristics")

    def choose(self, payload: Mapping[str, Any], session_key: str | None = None) -> Route:
        if self.session_affinity:
            pinned = self.affinity.get(session_key)
            if pinned:
                return pinned

        provider_name, model, effort, confidence, source, tier = self._classify(payload)
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"route references unknown provider {provider_name!r}")
        if provider.wire_api != "responses":
            raise ValueError(
                f"provider {provider_name!r} uses {provider.wire_api!r}; "
                "this MVP only forwards Responses API providers"
            )
        route = Route(tier, provider_name, model, confidence, source, effort)
        if self.session_affinity:
            self.affinity.put(session_key, route)
        return route

    def _classify(self, payload: Mapping[str, Any]) -> tuple[str, str, str | None, float | None, str, str | None]:
        """Returns (provider_name, model, effort, confidence, source, tier)."""
        if self.classifier:
            try:
                model, effort, confidence = self.classifier.classify(payload)
                entry = self.models.get(model)
                if entry:
                    if effort not in entry["efforts"]:
                        LOG.info(
                            "TypeSafe effort %r unsupported for %s; using default",
                            effort,
                            model,
                        )
                        effort = entry["default_effort"]
                    return entry.get("provider"), model, effort, confidence, "typesafe", None
                LOG.info("TypeSafe model %r not in catalog; using heuristic fallback", model)
            except TimeoutError as error:
                LOG.warning("TypeSafe timeout (%s); using heuristic fallback", error)
            except Exception:
                LOG.exception("TypeSafe routing failed; using heuristic fallback")
        tier, _ = self.heuristic.classify(payload)
        target = self.tiers.get(tier) or self.tiers.get("MEDIUM")
        if not target:
            raise ValueError("router config has no MEDIUM tier")
        return target["provider"], target["model"], target.get("reasoning_effort"), None, "heuristic", tier


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


class Handler(BaseHTTPRequestHandler):
    router: Router
    router_api_key: str | None = None
    max_body_bytes = 2_000_000

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._json(200, {"ok": True})
            return
        if path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "codex-router", "object": "model"}]})
            return
        self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": {"message": "unauthorized", "type": "authentication_error"}})
            return
        if self.path != "/v1/responses":
            self._json(404, {"error": {"message": "only /v1/responses is supported", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_body_bytes:
                raise ValueError("invalid or oversized request body")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            session_key = self.headers.get("X-Codex-Session-Id") or self.headers.get("X-Session-Id")
            route = self.router.choose(payload, session_key)
            LOG.info(
                "route selected tier=%s provider=%s model=%s effort=%s source=%s",
                route.tier or "-",
                route.provider,
                route.model,
                route.reasoning_effort or "default",
                route.source,
            )
            self._forward(payload, route)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("client disconnected during response forwarding")
        except ValueError as error:
            self._json(400, {"error": {"message": str(error), "type": "invalid_request_error"}})
        except Exception as error:
            LOG.exception("request failed")
            self._json(502, {"error": {"message": str(error), "type": "router_error"}})

    def _forward(self, payload: dict[str, Any], route: Route) -> None:
        provider = self.router.providers[route.provider]
        body = dict(payload)
        body["model"] = route.model
        if route.reasoning_effort:
            reasoning = dict(body.get("reasoning") or {})
            reasoning["effort"] = route.reasoning_effort
            body["reasoning"] = reasoning
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if body.get("stream") else "application/json"}
        headers.update(provider.headers)
        if provider.api_key_env:
            api_key = os.getenv(provider.api_key_env)
            if not api_key:
                raise ValueError(f"missing provider credential for {provider.name}")
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(provider.responses_url, json.dumps(body).encode(), headers=headers, method="POST")
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as error:
            self._send_upstream_error(provider, error)
            return
        except URLError as error:
            raise RuntimeError(f"upstream unavailable: {error.reason}") from error
        with response:
            self.send_response(response.status)
            for key, value in response.headers.items():
                if key.lower() not in {"connection", "content-length", "transfer-encoding"}:
                    self.send_header(key, value)
            self.end_headers()
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

    def _send_upstream_error(self, provider: Provider, error: HTTPError) -> None:
        body = error.read()
        try:
            parsed = json.loads(body)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        details = parsed.get("error", {}) if isinstance(parsed, Mapping) else {}
        if not isinstance(details, Mapping):
            details = {}
        LOG.warning(
            "upstream rejected provider=%s status=%s type=%s code=%s param=%s",
            provider.name,
            error.code,
            details.get("type", "unknown"),
            details.get("code", "unknown"),
            details.get("param", "unknown"),
        )
        self.send_response(error.code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not self.router_api_key:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.router_api_key}"

    def _json(self, status: int, body: Mapping[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    load_dotenv(os.getenv("ROUTER_DOTENV", ".env"))
    config_path = os.getenv("ROUTER_CONFIG", "router.json")
    config = load_config(config_path)
    Handler.router = Router(config)
    require_auth = os.getenv("ROUTER_REQUIRE_AUTH", "0").lower() in {"1", "true", "yes"}
    Handler.router_api_key = os.getenv("CODEX_ROUTER_API_KEY") if require_auth else None
    host = os.getenv("ROUTER_HOST", "127.0.0.1")
    port = int(os.getenv("ROUTER_PORT", "4000"))
    server = ThreadingHTTPServer((host, port), Handler)
    LOG.info("Codex router listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
