#!/usr/bin/env python3
"""Codex dynamic model router: composition root and entry point.

Domain modules:
- models: Route/Provider value types and provider-shape helpers
- classification: TypeSafe (Jev) and heuristic classifiers
- routing: Router, session affinity, upstream request construction
- config: router.json and .env loading
- server: thin HTTP boundary forwarding to the selected provider
"""

from __future__ import annotations

import logging
import os
from http.server import ThreadingHTTPServer

from classification import HeuristicClassifier, TypeSafeClassifier
from config import _dotenv_values, load_config, load_dotenv
from models import Provider, Route, _strip_reasoning_content
from routing import Router, SessionAffinity, build_upstream_request
from server import Handler

LOG = logging.getLogger("codex_router")

__all__ = [
    "Handler",
    "HeuristicClassifier",
    "Provider",
    "Route",
    "Router",
    "SessionAffinity",
    "TypeSafeClassifier",
    "_dotenv_values",
    "_strip_reasoning_content",
    "build_upstream_request",
    "load_config",
    "load_dotenv",
    "main",
]


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
