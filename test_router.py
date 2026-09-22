import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from router import Handler, Route, Router, TypeSafeClassifier, _dotenv_values


class FakeTypeSafe:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def classify(self, payload):
        self.calls += 1
        return self.result


CONFIG = {
    "providers": {
        "fast": {"base_url": "http://fast.test/v1", "wire_api": "responses"},
        "strong": {"base_url": "http://strong.test/v1", "wire_api": "responses"},
        "mimo": {"base_url": "https://api.xiaomimimo.com/v1", "wire_api": "responses"},
    },
    "models": [
        {"name": "fast-model", "provider": "fast", "efforts": ["low", "medium"], "default_effort": "low"},
        {"name": "coding-model", "provider": "strong", "efforts": ["medium", "high"], "default_effort": "high"},
        {"name": "mimo-v2.6-flash", "provider": "mimo", "efforts": ["none", "low", "high"], "default_effort": "low"},
    ],
    "tiers": {
        "SIMPLE": {"provider": "fast", "model": "fast-model", "reasoning_effort": "low"},
        "MEDIUM": {"provider": "fast", "model": "medium-model", "reasoning_effort": "medium"},
        "COMPLEX": {"provider": "strong", "model": "coding-model", "reasoning_effort": "high"},
        "REASONING": {"provider": "mimo", "model": "mimo-v2.6-flash", "reasoning_effort": "high"},
    },
    "session_affinity": True,
    "session_max": 8,
    "typesafe_confidence_min": 0.5,
}


def test_router() -> None:
    assert urlsplit("/v1/models?client_version=0.155.0").path == "/v1/models"
    assert _dotenv_values([
        "# comment",
        "export CODEX_ROUTER_API_KEY='local-secret'",
        "MIMO_API_KEY=sk-test",
    ]) == {
        "CODEX_ROUTER_API_KEY": "local-secret",
        "MIMO_API_KEY": "sk-test",
    }

    options = TypeSafeClassifier(CONFIG["models"])._options
    assert options == {
        "fast-model@low": None,
        "fast-model@medium": None,
        "coding-model@medium": None,
        "coding-model@high": None,
        "mimo-v2.6-flash@none": None,
        "mimo-v2.6-flash@low": None,
        "mimo-v2.6-flash@high": None,
    }

    classifier = FakeTypeSafe(("coding-model", "high", 0.91))
    router = Router(CONFIG, classifier=classifier)
    request_with_tool = {"input": "edit the repository", "tools": [{"name": "apply_patch"}]}

    first = router.choose(request_with_tool, "session-1")
    second = router.choose({"input": "a short unrelated question"}, "session-1")
    assert first == Route(None, "strong", "coding-model", 0.91, "typesafe", "high")
    assert second == first
    assert classifier.calls == 1

    unsupported = Router(CONFIG, classifier=FakeTypeSafe(("mimo-v2.6-flash", "xhigh", 0.99)))
    snapped = unsupported.choose({"input": "a quick question"})
    assert snapped.provider == "mimo"
    assert snapped.model == "mimo-v2.6-flash"
    assert snapped.reasoning_effort == "low"
    assert snapped.source == "typesafe"
    assert snapped.tier is None

    invalid = Router(CONFIG, classifier=FakeTypeSafe(("gpt-5.6-terra", "high", 0.99)))
    fallback = invalid.choose({"input": "debug a repository and refactor the API"})
    assert fallback.tier == "COMPLEX"
    assert fallback.provider == "strong"
    assert fallback.model == "coding-model"
    assert fallback.source == "heuristic"

    uncertain = Router(CONFIG, classifier=FakeTypeSafe(("fast-model", "low", 0.1)))
    low_confidence = uncertain.choose({"input": "debug a repository and refactor the API"})
    assert low_confidence.tier == "COMPLEX"
    assert low_confidence.source == "heuristic"

    reasoning = Router(CONFIG, classifier=FakeTypeSafe(("mimo-v2.6-flash", "high", 0.99))).choose(
        {"input": "prove this algorithm"}
    )
    assert reasoning.provider == "mimo"
    assert reasoning.model == "mimo-v2.6-flash"
    assert reasoning.reasoning_effort == "high"


def test_missing_provider_credential_is_not_reflected() -> None:
    config = {
        "providers": {
            "mimo": {
                "base_url": "https://api.xiaomimimo.com/v1",
                "api_key_env": "MISSING_PROVIDER_KEY",
                "wire_api": "responses",
            }
        },
        "tiers": {"REASONING": {"provider": "mimo", "model": "mimo-model"}},
        "models": [{"name": "mimo-model", "provider": "mimo", "efforts": ["high"], "default_effort": "high"}],
        "session_affinity": False,
    }
    Handler.router = Router(config, classifier=FakeTypeSafe(("mimo-model", "high", 0.99)))
    Handler.router_api_key = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/responses",
            data=json.dumps({"input": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status = None
        try:
            urlopen(request, timeout=2)
        except HTTPError as error:
            status = error.code
            body = error.read().decode()
        else:
            raise AssertionError("request unexpectedly succeeded")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert status == 400
    message = json.loads(body)["error"]["message"]
    assert message == "missing provider credential for mimo"
    assert "MISSING_PROVIDER_KEY" not in body


if __name__ == "__main__":
    test_router()
    test_missing_provider_credential_is_not_reflected()
    print("ok")
