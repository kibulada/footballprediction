"""Tests for the LLM intent router (agents/football/llm_router.py).

The router must be a pure parser: it never invents numbers, never raises,
and falls back to None whenever the LLM is unavailable or unconfigured.
"""
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import llm_router as router  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch):
    """No LLM env vars by default in tests; each test sets what it needs."""
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    yield


def _mock_transport(payload: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"].startswith("Bearer ")
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_not_configured_returns_none_immediately():
    assert router.is_configured() is False
    import asyncio
    assert asyncio.run(router.route_intent("besok ada apa")) is None


def test_configured_with_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    assert router.is_configured() is True
    assert router.base_url() == "https://router.test/v1"


def test_configured_with_openai_alias(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert router.is_configured() is True


def test_build_messages_contains_examples_and_user_text():
    msgs = router.build_messages("besok ada match ucl?")
    system = msgs[0]["content"]
    assert "analisa" in system
    assert "EXAMPLES" in system
    assert msgs[-1]["content"] == "besok ada match ucl?"


def test_extract_json_plain():
    assert router._extract_json('{"command":"top","params":{"date":"besok"}}') == {
        "command": "top", "params": {"date": "besok"}
    }


def test_extract_json_markdown_fence():
    text = '```json\n{"command":"analyse","params":{"league":"ucl"}}\n```'
    obj = router._extract_json(text)
    assert obj == {"command": "analyse", "params": {"league": "ucl"}}


def test_extract_json_surrounded_by_prose():
    text = 'Sure! Here is the intent:\n{"command":"stats","params":{}}\nHope this helps.'
    obj = router._extract_json(text)
    assert obj == {"command": "stats", "params": {}}


def test_extract_json_invalid():
    assert router._extract_json("no json here") is None
    assert router._extract_json("") is None


def test_validate_intent_whitelists_commands():
    assert router.validate_intent({"command": "TOP", "params": {}})["command"] == "top"
    assert router.validate_intent({"command": "rm -rf /", "params": {}}) is None
    assert router.validate_intent({"command": "top"}) == {"command": "top", "params": {}}
    assert router.validate_intent({"command": "top", "params": "notadict"}) == {"command": "top", "params": {}}
    assert router.validate_intent(["not", "a", "dict"]) is None
    assert router.validate_intent(None) is None


def _transport_with_body(body: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    return httpx.MockTransport(handler)


def test_parse_response_json_plain():
    obj = router._parse_response_json('{"id":"x","choices":[]}')
    assert obj == {"id": "x", "choices": []}


def test_parse_response_json_leading_whitespace_and_sse_tail():
    # 9router body: whitespace prefix + JSON + trailing SSE "data: [DONE]".
    body = (
        "\n\n     \n\n"
        '{"id":"gen-1","choices":[{"message":{"content":"x"}}]}'
        "\ndata: [DONE]\n"
    )
    obj = router._parse_response_json(body)
    assert obj["choices"][0]["message"]["content"] == "x"


def test_parse_response_json_garbage_before_json():
    body = 'garbage here\n{"choices":[]} trailing'
    obj = router._parse_response_json(body)
    assert obj == {"choices": []}


def test_parse_response_json_invalid():
    assert router._parse_response_json("") is None
    assert router._parse_response_json("no json at all") is None


def _completion_body(intent_obj: dict, prefix: str = "", suffix: str = "") -> str:
    """Build a realistic chat.completion HTTP body with whitespace/SSE quirks."""
    inner = json.dumps(intent_obj, ensure_ascii=False)
    payload = {"choices": [{"index": 0, "message": {"role": "assistant", "content": inner}}]}
    return prefix + json.dumps(payload, ensure_ascii=False) + suffix


def test_route_intent_parses_valid_response(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    body = _completion_body({"command": "analyse", "params": {"league": "ucl", "home": "lyon", "away": "sparta prague"}})
    import asyncio
    intent = asyncio.run(
        router.route_intent("analisa ucl lyon vs sparta prague", client=httpx.AsyncClient(transport=_transport_with_body(body)))
    )
    assert intent == {
        "command": "analyse",
        "params": {"league": "ucl", "home": "lyon", "away": "sparta prague"},
    }


def test_route_intent_handles_9router_body_shape(monkeypatch):
    # Real 9router response: whitespace prefix + completion JSON + SSE tail.
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    body = _completion_body(
        {"command": "top", "params": {"date": "besok", "leagues": ["ucl"], "top_n": 5}},
        prefix="\n\n   \n\n",
        suffix="\ndata: [DONE]\n\n",
    )
    import asyncio
    intent = asyncio.run(
        router.route_intent("besok ada match apa?", client=httpx.AsyncClient(transport=_transport_with_body(body)))
    )
    assert intent == {"command": "top", "params": {"date": "besok", "leagues": ["ucl"], "top_n": 5}}


def _sse_stream(chunks: list[str]) -> str:
    """Build an SSE response: role chunk + content chunks + terminal chunk."""
    lines = [
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    ]
    for c in chunks:
        lines.append('data: {"choices":[{"index":0,"delta":{"content":' + json.dumps(c) + '},"finish_reason":null}]}')
    lines.append('data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{}}')
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n"


def test_parse_sse_content_reassembles_chunks():
    body = _sse_stream(['{"command":"top",', '"params":{"date":"besok"}}'])
    content = router._parse_sse_content(body)
    assert content == '{"command":"top","params":{"date":"besok"}}'


def test_parse_sse_content_skips_non_sse_body():
    assert router._parse_sse_content('{"choices":[]}') is None
    assert router._parse_sse_content("") is None


def test_route_intent_handles_sse_stream(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    body = _sse_stream(['{"command":"analyse","params":', '{"league":"ucl","home":"lyon","away":"sparta prague"}}'])
    import asyncio
    intent = asyncio.run(
        router.route_intent("analisa ucl lyon vs sparta", client=httpx.AsyncClient(transport=_transport_with_body(body)))
    )
    assert intent == {"command": "analyse", "params": {"league": "ucl", "home": "lyon", "away": "sparta prague"}}


def test_route_intent_retries_on_truncated_stream(monkeypatch):
    # finish_reason=length truncates the intent JSON; the router must retry
    # and accept a second (valid) response.
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    truncated = (
        'data: {"choices":[{"index":0,"delta":{"content":"{\"command\":\"top\",\"params\":{"},"finish_reason":null}]}' "\n\n"
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}' "\n\n"
        "data: [DONE]\n"
    )
    valid = _sse_stream(['{"command":"top","params":{"date":"besok","leagues":["ucl"]}}'])
    responses = iter([truncated, valid])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=next(responses))

    import asyncio
    intent = asyncio.run(
        router.route_intent("besok ada match apa?", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    )
    assert intent == {"command": "top", "params": {"date": "besok", "leagues": ["ucl"]}}


def test_route_intent_returns_none_on_http_error(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    import asyncio
    intent = asyncio.run(
        router.route_intent("halo", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    )
    assert intent is None


def test_route_intent_returns_none_on_garbage(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    transport = _mock_transport({"choices": [{"message": {"content": "I am not JSON"}}]})
    import asyncio
    assert asyncio.run(router.route_intent("xyz", client=httpx.AsyncClient(transport=transport))) is None


def test_route_intent_rejects_unknown_command(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    transport = _mock_transport({"choices": [{"message": {"content": '{"command":"delete_all","params":{}}'}}]})
    import asyncio
    assert asyncio.run(router.route_intent("x", client=httpx.AsyncClient(transport=transport))) is None


def test_route_intent_never_raises_on_network_failure(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    import asyncio
    intent = asyncio.run(
        router.route_intent("x", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    )
    assert intent is None


def test_placeholder_values_count_as_unconfigured(monkeypatch):
    # The .env template ships with YOUR-... placeholders; these must never
    # cause a request to a bogus endpoint.
    monkeypatch.setenv("LLM_BASE_URL", "https://YOUR-9ROUTER-ENDPOINT/v1")
    monkeypatch.setenv("LLM_API_KEY", "YOUR-9ROUTER-API-KEY")
    assert router.is_configured() is False


def test_config_timeout_and_max_text_override(monkeypatch):
    # request_timeout/max_user_text read the llm_router config section;
    # defaults apply when the section is empty.
    import time
    import agents.football.llm_router as mod
    # Fresh _ts so the 60s TTL does not silently discard the injected values.
    monkeypatch.setattr(mod, "_CONFIG_CACHE", {"_ts": time.monotonic(), "timeout_seconds": 7, "max_user_text": 500})
    assert mod.request_timeout() == 7.0
    assert mod.max_user_text() == 500
    monkeypatch.setattr(mod, "_CONFIG_CACHE", {"_ts": time.monotonic()})
    assert mod.request_timeout() == mod.REQUEST_TIMEOUT
    assert mod.max_user_text() == mod.MAX_USER_TEXT


def test_build_messages_truncates_to_max_user_text(monkeypatch):
    import time
    monkeypatch.setattr(router, "_CONFIG_CACHE", {"_ts": time.monotonic(), "max_user_text": 50})
    msgs = router.build_messages("x" * 500)
    assert len(msgs[-1]["content"]) == 50


def test_model_name_falls_back_to_config_then_default(monkeypatch):
    import time
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    # No model in config -> DEFAULT_MODEL
    monkeypatch.setattr(router, "_CONFIG_CACHE", {"_ts": time.monotonic()})
    assert router.model_name() == router.DEFAULT_MODEL
    # Model from config llm_router section
    monkeypatch.setattr(router, "_CONFIG_CACHE", {"_ts": time.monotonic(), "model": "claude-haiku"})
    assert router.model_name() == "claude-haiku"
    # Env wins over config
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    assert router.model_name() == "gpt-4.1-mini"
