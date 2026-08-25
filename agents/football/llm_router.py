"""LLM intent router — free-form Discord text -> structured bot command.

The router is a PURE parser. It never computes probabilities, odds, or
predictions; it only maps the user's words onto one of the existing bot
commands. All execution, validation and formatting stays in the existing
handlers (``bot._handle_*``), so the LLM can never fabricate numbers.

Configuration (.env, OpenAI-compatible — works with 9router, OpenAI,
DeepSeek, Groq, Ollama/LM Studio, ...):

    LLM_BASE_URL=https://.../v1     (or OPENAI_BASE_URL)
    LLM_API_KEY=sk-...              (or OPENAI_API_KEY)
    LLM_MODEL=gpt-4o-mini           (or OPENAI_MODEL)

If no base URL / key is configured, :func:`route_intent` returns ``None``
immediately and the bot keeps its current rule-based behaviour — the LLM
router is strictly additive.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger("hermes-football-llm-router")

DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 10.0  # routing needs speed; the runner has its own 85s deadline
MAX_USER_TEXT = 800

_CONFIG_TTL = 60.0
_CONFIG_CACHE: dict[str, Any] = {"_ts": 0.0}


def _router_config() -> dict[str, Any]:
    """Lazy-read config/football.json llm_router section (60s cache)."""
    import time
    from pathlib import Path

    now = time.monotonic()
    if now - (_CONFIG_CACHE.get("_ts") or 0.0) > _CONFIG_TTL:
        cfg: dict[str, Any] = {}
        try:
            path = Path(__file__).resolve().parent.parent.parent / "config" / "football.json"
            cfg = (json.loads(path.read_text(encoding="utf-8")) or {}).get("llm_router") or {}
        except Exception:
            pass
        _CONFIG_CACHE.clear()
        _CONFIG_CACHE["_ts"] = now
        _CONFIG_CACHE.update(cfg)
    return dict(_CONFIG_CACHE)


def request_timeout() -> float:
    return float(_router_config().get("timeout_seconds", REQUEST_TIMEOUT))


def max_user_text() -> int:
    return int(_router_config().get("max_user_text", MAX_USER_TEXT))

# Commands the router may produce; anything else is rejected.
KNOWN_COMMANDS = frozenset(
    {"top", "compare", "analyse", "stats", "settle", "odds",
     "best", "bestgoalmatch", "livescore", "flashscore", "help", "none"}
)

SYSTEM_PROMPT = """You are the intent router for "Hermes Football", a Discord football-prediction bot.
The user writes in casual Indonesian or English. Your ONLY job: convert their request into exactly
ONE bot command and reply with ONLY a JSON object — no markdown fences, no extra text.

Commands:

1. analyse — deep pre-match analysis (form, H2H, odds, Poisson model, decision engine).
   params: league, home, away (ALL required).
   Trigger: "analisa/analisis/prediksi/perkiraan <tim A> vs <tim B>", "berapa odds X vs Y",
   "gimana prediksi match UCL...", any request for a detailed match prediction.

2. top — ranked list of matches with odds + value.
   params: date ("today" | "besok" | "tomorrow" | "YYYY-MM-DD", optional),
           leagues (list of league keywords, optional), top_n (int, optional, default 5).
   Trigger: "match hari ini", "besok ada apa", "top match", "jadwal besok",
   "match yang ada odds-nya hari ini".

3. compare — head-to-head comparison of two teams.
   params: home, away (required), league (optional, default EPL).
   Trigger: "bandingkan X vs Y", "h2h X dan Y", "siapa lebih kuat X atau Y".

4. stats — prediction-log statistics (hit rate, logloss, ROI, closing-line value).
   params: none (edge_threshold optional float 0-1).
   Trigger: "statistik bot", "akurasi model", "record prediksi", "stats".

5. settle — record a finished match result into the log.
   params: home, away, result ("2-1" format), OR auto: true.
   Trigger: "hasil X vs Y 2-1", "catat skor", "settle", "settle auto".

6. odds — record an odds snapshot at a labelled pre-match time (for closing-line value).
   params: timing ("T-24h"|"T-6h"|"T-1h"|"T-15m"), home, away, odds ("h,d,a" e.g. "1.62,4.30,4.60").
   Trigger: "catat odds T-6h X vs Y 1.62,4.30,4.60".

7. best — pick the SINGLE best prediction among today's (+ early tomorrow) matches
   of ONE league, ranked by the prediction engine.
   params: league (required, canonical keyword e.g. "epl", "ucl", "serie a").
   Trigger: "!best epl", "match epl mana yang paling bagus buat betting hari ini",
   "pick terbaik dari match ucl", "prediksi paling valid di liga portugal hari ini".

8. bestgoalmatch — today's most goal-friendly (banjir gol) match across leagues.
   params: league (optional; omit to scan all registered leagues).
   Trigger: "!bestgoalmatch", "match mana yang bakal banyak gol hari ini",
   "cari match banjir gol", "over gol paling gede hari ini".

9. livescore — find a specific match via LiveScore (today, then tomorrow) and run the
   full analysis. params: league, home, away (ALL required).
   Trigger: ONLY when the user EXPLICITLY names LiveScore/livescore, e.g.
   "cari barcelona vs real madrid di livescore", "!livescore laliga barcelona vs real madrid".

10. flashscore — find a specific match via Flashscore (today, then tomorrow) and run the
   full analysis. params: league, home, away (ALL required).
   Trigger: ONLY when the user EXPLICITLY names Flashscore/flashscore, e.g.
   "pakai flashscore, lyon vs sparta prague", "!flashscore laliga barcelona vs real madrid".
   For any other specific-match request without an explicit source, use "analyse" instead.

11. help — list available commands.
   params: note (optional; a clarifying question like "Liga mana? ucl / epl / serie a").
   Trigger: "help", "ada command apa saja", "bantu".

10. none — the message is NOT a football request (greeting, thanks, chit-chat).
   params: {}.

League keywords — output the CANONICAL keyword:
  ucl, uel, uecl, epl (premier league, inggris), laliga (la liga, spanyol),
  serie a (italia), bundesliga (jerman), ligue 1 (prancis), liga portugal (primeira),
  eredivisie (belanda), super lig (turki), mls, liga 1 (indonesia), saudi,
  scottish, belgian, championship, ligue 2, serie b, segunda, a-league, k-league, j1.

RULES:
- Reply with ONLY valid JSON: {"command": "<name>", "params": {...}}.
- NEVER invent odds, scores, xG, or probabilities. Only pass through numbers the user typed.
- If a required param for analyse/compare is missing, use command "help" with a note asking for it.
- Team names: keep them close to what the user wrote; strip country suffixes like "(Isr)".
- Dates: "besok"/"tomorrow" -> "besok"; "hari ini"/"today" -> "today".
- If the request is ambiguous between analyse and top, prefer "analyse" when a specific
  match (two teams) is named, "top" when it is about a list/schedule."""

EXAMPLE_CALLS = [
    ("analisa ucl lyon vs sparta prague",
     '{"command":"analyse","params":{"league":"ucl","home":"lyon","away":"sparta prague"}}'),
    ("besok ada match apa aja di ucl dan epl?",
     '{"command":"top","params":{"date":"besok","leagues":["ucl","epl"],"top_n":5}}'),
    ("bandingkan arsenal vs chelsea",
     '{"command":"compare","params":{"home":"arsenal","away":"chelsea","league":"epl"}}'),
    ("berapa akurasi model bot ini",
     '{"command":"stats","params":{}}'),
    ("hasil bodo vs union 2-1 catat",
     '{"command":"settle","params":{"home":"bodo","away":"union","result":"2-1"}}'),
    ("simpan odds T-6h bodo vs union 1.62,4.30,4.60",
     '{"command":"odds","params":{"timing":"T-6h","home":"bodo","away":"union","odds":"1.62,4.30,4.60"}}'),
    ("match epl mana paling bagus buat betting hari ini",
     '{"command":"best","params":{"league":"epl"}}'),
    ("match mana yang bakal banjir gol hari ini",
     '{"command":"bestgoalmatch","params":{}}'),
    ("makasih bro",
     '{"command":"none","params":{}}'),
]


def _env_first(*names: str) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v and "YOUR-" not in v.upper():
            return v
    return ""


def is_configured() -> bool:
    """True iff an OpenAI-compatible endpoint + key are configured.

    Placeholder values (e.g. the ``YOUR-...`` template in .env) count as
    unconfigured so the bot never fires requests at a bogus endpoint.
    """
    return bool(_env_first("LLM_BASE_URL", "OPENAI_BASE_URL") and _env_first("LLM_API_KEY", "OPENAI_API_KEY"))


def model_name() -> str:
    """Model: env LLM_MODEL/OPENAI_MODEL, then config llm_router.model, then default."""
    return (
        _env_first("LLM_MODEL", "OPENAI_MODEL")
        or str(_router_config().get("model") or "")
        or DEFAULT_MODEL
    )


def base_url() -> str:
    return _env_first("LLM_BASE_URL", "OPENAI_BASE_URL").rstrip("/")


def api_key() -> str:
    return _env_first("LLM_API_KEY", "OPENAI_API_KEY")


def build_messages(user_text: str) -> list[dict[str, str]]:
    """System prompt + few-shot examples + user text."""
    examples = "\n".join(
        f"User: {u}\nBot: {j}" for u, j in EXAMPLE_CALLS
    )
    system = SYSTEM_PROMPT + "\n\nEXAMPLES:\n" + examples
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (user_text or "").strip()[:max_user_text()]},
    ]


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced {...} object out of the model reply."""
    if not text:
        return None
    text = text.strip()
    # Drop markdown fences if the model wrapped the JSON anyway.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _parse_response_json(text: str) -> dict[str, Any] | None:
    """Parse the HTTP response body into the outer completion JSON.

    9router responses can carry leading whitespace and a trailing
    ``data: [DONE]`` SSE leftover ("Extra data" for plain json.loads), so we
    (1) drop anything after the SSE sentinel, (2) try a plain parse, then
    (3) fall back to scanning for the first valid ``{...}`` object.
    """
    if not text:
        return None
    text = text.split("data: [DONE]")[0].strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    start = 0
    while True:
        start = text.find("{", start)
        if start < 0:
            return None
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        start += 1


def _parse_sse_content(text: str) -> str | None:
    """Reassemble the assistant message from an SSE ``data:`` stream.

    Some 9router model bundles (e.g. the "hermes" combo) answer in streaming
    mode even when ``stream`` is not requested: the HTTP body is a sequence
    of ``data: {json}`` lines whose ``choices[0].delta.content`` pieces form
    the reply. Returns the concatenated content, or None when the body is
    not an SSE stream (caller then parses it as a plain JSON completion).
    """
    if "data: " not in text:
        return None
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload in ("[DONE]", "[done]"):
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        try:
            delta = obj["choices"][0].get("delta") or {}
        except (KeyError, IndexError, TypeError):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts) if parts else None


def _intent_from_body(body_text: str) -> dict[str, Any] | None:
    """Extract a validated intent from an HTTP response body.

    Accepts both shapes 9router can return: an SSE ``data:`` stream (combo
    bundles) and a plain chat.completion JSON (possibly with leading
    whitespace / trailing ``data: [DONE]``).
    """
    content = _parse_sse_content(body_text)
    if content is None:
        data = _parse_response_json(body_text)
        if not data:
            return None
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
    return validate_intent(_extract_json(content))


def validate_intent(obj: Any) -> dict[str, Any] | None:
    """Whitelist the parsed intent: known command + dict params."""
    if not isinstance(obj, dict):
        return None
    command = obj.get("command")
    if not isinstance(command, str) or command.strip().lower() not in KNOWN_COMMANDS:
        return None
    params = obj.get("params")
    if not isinstance(params, dict):
        params = {}
    return {"command": command.strip().lower(), "params": params}


async def route_intent(
    user_text: str,
    timeout: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Map free-form user text to a validated intent dict (or None).

    Never raises: any network/parse error is logged and returns None so the
    bot falls back to its rule-based behaviour.
    """
    if not is_configured():
        logger.debug("LLM router not configured; skipping")
        return None
    effective_timeout = timeout if timeout is not None else request_timeout()
    url = base_url() + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    # Reasoning models inside the combo bundle consume tokens before the
    # answer; a tight cap (300) ends in finish_reason="length" with a
    # truncated JSON. 1000 leaves ample room for the ~80-token intent JSON.
    body = {
        "model": model_name(),
        "messages": build_messages(user_text),
        "temperature": 0.0,
        "max_tokens": 1000,
    }
    own_client = client is None
    c = client or httpx.AsyncClient(timeout=effective_timeout)
    try:
        # The 9router "combo" bundle routes each call to a random member LLM,
        # so the response shape varies per call (SSE stream vs plain JSON vs
        # malformed). One retry materially improves hit rate; never raise.
        for attempt in range(2):
            resp = await c.post(url, headers=headers, json=body)
            resp.raise_for_status()
            intent = _intent_from_body(resp.text)
            if intent:
                return intent
            if attempt == 0:
                logger.warning("LLM router: unusable reply on attempt 1; retrying")
        return None
    except Exception as exc:  # noqa: BLE001 - router must never break the bot
        logger.warning("LLM router error (%s: %s); rule-based fallback", type(exc).__name__, exc)
        return None
    finally:
        if own_client:
            await c.aclose()
