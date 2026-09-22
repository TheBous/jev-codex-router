"""HTTP boundary: thin handlers that parse, invoke the router, and forward upstream."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from models import Provider
from routing import Router, build_upstream_request

LOG = logging.getLogger("codex_router")


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
            payload = self._parse_body()
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

    def _parse_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > self.max_body_bytes:
            raise ValueError("invalid or oversized request body")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _forward(self, payload: dict[str, Any], route: Any) -> None:
        provider = self.router.providers[route.provider]
        request = build_upstream_request(payload, route, provider)
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
        details = _error_details(body)
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


def _error_details(body: bytes) -> Mapping[str, Any]:
    try:
        parsed = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return {}
    details = parsed.get("error", {}) if isinstance(parsed, Mapping) else {}
    return details if isinstance(details, Mapping) else {}
