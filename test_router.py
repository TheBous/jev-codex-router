import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from router import Handler, Route, Router, _dotenv_values


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

    classifier = FakeTypeSafe(("COMPLEX", 0.91))
    router = Router(CONFIG, classifier=classifier)
    request_with_tool = {"input": "edit the repository", "tools": [{"name": "apply_patch"}]}

    first = router.choose(request_with_tool, "session-1")
    second = router.choose({"input": "a short unrelated question"}, "session-1")
    assert first == Route("COMPLEX", "strong", "coding-model", 0.91, "typesafe", "high")
    assert second == first
    assert classifier.calls == 1

    uncertain = Router(CONFIG, classifier=FakeTypeSafe(("SIMPLE", 0.1)))
    fallback = uncertain.choose({"input": "debug a repository and refactor the API"})
    assert fallback.tier == "COMPLEX"
    assert fallback.source == "heuristic"

    reasoning = Router(CONFIG, classifier=FakeTypeSafe(("REASONING", 0.99))).choose({"input": "prove this algorithm"})
    assert reasoning.provider == "mimo"
    assert reasoning.model == "mimo-v2.6-flash"


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
        "session_affinity": False,
    }
    Handler.router = Router(config, classifier=FakeTypeSafe(("REASONING", 0.99)))
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
