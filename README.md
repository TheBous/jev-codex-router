# Codex dynamic model router

Small local proxy for Codex:

```text
Codex -> codex-router -> TypeSafe/Jev classification -> selected Responses provider
```

Jev picks the model and its reasoning effort with a single TypeSafe `Choice` over the `model@effort` pairs generated from the `models` catalog in `router.json` — each model lists only the efforts it supports. The router pins the selected route for a session and falls back to deterministic heuristics over `tiers` when TypeSafe is unavailable, below the configured confidence threshold, or returning pairs outside the catalog.

The DeepSeek and OpenAI catalog entries are placeholders until their Responses endpoints are verified; an unusable upstream surfaces as an `upstream rejected provider=...` log line and does not crash the router.

## Run

```bash
cp router.example.json router.json
# Fill router.json and .env with your local values.
cp .env.example .env
uv run codex-router
```

The proxy listens on `http://127.0.0.1:4000` and accepts `POST /v1/responses`. It only forwards providers configured with `wire_api: "responses"`; GLM is included as a placeholder because its chat-only compatibility needs a separate adapter before it can safely serve Codex tool turns.

MiMo is configured for the `REASONING` tier as `mimo-v2.6-flash`; DeepSeek is configured for `COMPLEX` as `deepseek-flash`. For Token Plan, replace that provider's `base_url` with `https://token-plan-cn.xiaomimimo.com/v1` and use a `tp-...` or `ttp-...` key through `MIMO_TOKEN_PLAN_API_KEY`.

GLM is registered as `glm-5.3-flash`, but the current Z.AI API documentation exposes it through Chat Completions. The proxy currently forwards Responses API only, so GLM stays disabled for Codex until a Chat→Responses streaming/tool adapter is added.

Configure Codex once:

```toml
model = "codex-router"
model_provider = "router"

[model_providers.router]
name = "Dynamic model router"
base_url = "http://127.0.0.1:4000/v1"
wire_api = "responses"
env_key = "CODEX_ROUTER_API_KEY"
```

## Check

```bash
uv run python test_router.py
```

TypeSafe receives the request text, instructions, tool names, and reasoning settings for classification. Do not enable it for data that cannot be sent to the TypeSafe service.

The router loads `.env` automatically. Existing shell environment variables take precedence over `.env`; the real `.env` file is ignored by Git.

Local authentication is disabled by default. Set `ROUTER_REQUIRE_AUTH=1` only when Codex also receives `CODEX_ROUTER_API_KEY` in its own environment.
# jev-codex-router
