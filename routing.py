"""Routing decisions: session pinning, tier fallback, and upstream request construction."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from json import dumps
from threading import Lock
from typing import Any, Mapping
from urllib.request import Request

from classification import HeuristicClassifier, TypeSafeClassifier
from models import Provider, Route, _strip_reasoning_content

LOG = logging.getLogger("codex_router")


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
                drop_reasoning_content=bool(provider.get("drop_reasoning_content", False)),
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


def build_upstream_request(payload: Mapping[str, Any], route: Route, provider: Provider) -> Request:
    """Compose the upstream HTTP request for the selected route and provider."""
    body = dict(payload)
    body["model"] = route.model
    if route.reasoning_effort:
        reasoning = dict(body.get("reasoning") or {})
        reasoning["effort"] = route.reasoning_effort
        body["reasoning"] = reasoning
    if provider.drop_reasoning_content:
        body["input"] = _strip_reasoning_content(body.get("input"))
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if body.get("stream") else "application/json",
    }
    headers.update(provider.headers)
    if provider.api_key_env:
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise ValueError(f"missing provider credential for {provider.name}")
        headers["Authorization"] = f"Bearer {api_key}"
    return Request(provider.responses_url, dumps(body).encode(), headers=headers, method="POST")
