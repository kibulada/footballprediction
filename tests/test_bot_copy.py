"""Tests for the Discord copy-button helpers in bot.py (pure functions).

These must not require a live Discord connection: only the in-memory store,
chunking, plain-text rendering, the ephemeral-chunk sender and disk
persistence are exercised. The persisted state file is redirected to a temp
path so tests never touch the bot's real copy_buttons.json.
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402

_COPY_TEST_STATE = Path(tempfile.gettempdir()) / "hermes_football_copy_buttons_test.json"
bot._COPY_STATE_PATH = _COPY_TEST_STATE
bot._COPY_TEXTS.clear()
if _COPY_TEST_STATE.exists():
    _COPY_TEST_STATE.unlink()


def _fake_interaction():
    sent = {"response": [], "followup": []}

    class Resp:
        async def send_message(self, *a, **kw):
            sent["response"].append((a, kw))

    class Fup:
        async def send(self, *a, **kw):
            sent["followup"].append((a, kw))

    it = type("It", (), {})()
    it.response = Resp()
    it.followup = Fup()
    it._sent = sent
    return it


def _run_send(chunks, **kw):
    """Drive _send_ephemeral_chunks with a fake interaction; sleeps recorded."""
    sleeps = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    async def runner():
        it = _fake_interaction()
        orig = bot.asyncio.sleep
        bot.asyncio.sleep = fake_sleep
        try:
            await bot._send_ephemeral_chunks(it, chunks, **kw)
        finally:
            bot.asyncio.sleep = orig
        return it, sleeps

    return asyncio.run(runner())


def test_send_ephemeral_chunks_response_plus_spaced_followups():
    it, sleeps = _run_send(["first", "second", "third"])
    args, kw = it._sent["response"][0]
    assert args == ("first",) and kw == {"ephemeral": True}
    assert [k.get("content") for _, k in it._sent["followup"]] == ["second", "third"]
    assert all(k.get("ephemeral") is True for _, k in it._sent["followup"])
    # one gap before each followup, never shorter than the burst-safe value
    assert sleeps == [bot._COPY_CHUNK_GAP] * 2
    assert bot._COPY_CHUNK_GAP >= 1.0  # 5 msgs / 5 s burst cap needs > 1.0


def test_send_ephemeral_chunks_header_replaces_response():
    it, _ = _run_send(["a", "b"], header="✅ Data siap disalin — halaman 1/3:")
    args, kw = it._sent["response"][0]
    assert args[0].startswith("✅ Data siap disalin")
    # all chunks go as followups when a header takes the response slot
    assert [k.get("content") for _, k in it._sent["followup"]] == ["a", "b"]


def test_send_ephemeral_chunks_wrap_code():
    it, _ = _run_send(["line1"], header="H", wrap_code=True)
    assert it._sent["followup"][0][1]["content"] == "```\nline1\n```"


def test_split_body_chunks_single():
    assert bot._split_body_chunks("abc") == ["abc"]


def test_split_body_chunks_on_line_boundaries():
    # 6000 chars of "line-XXXX\n" -> every chunk boundary lands on a newline,
    # so joining chunks with "\n" reconstructs the original exactly.
    body = "\n".join(f"line-{i:04d}" for i in range(600)) + "\n"
    chunks = bot._split_body_chunks(body, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "\n".join(chunks) == body


def test_copy_plain_messages_single_keeps_markdown():
    rendered = {
        "title": "**Match** Analysis",
        "body": "**Home** vs Away\n\n`odds`: 1.62 / 4.30 / 4.60",
        "footer": "Hermes Football",
    }
    chunks = bot._copy_plain_messages(rendered)
    assert len(chunks) == 1
    assert chunks[0].startswith("**Match** Analysis")  # title first
    assert "**Home** vs Away" in chunks[0]  # markdown untouched
    assert "`odds`: 1.62 / 4.30 / 4.60" in chunks[0]
    assert chunks[0].endswith("Hermes Football")  # footer pinned at the end


def test_copy_plain_messages_splits_long_body_no_cap():
    # ~9.5k chars: below _MAX_COPY_BODY (no body truncation) but > 4 messages
    # worth of chunks, proving the copy button has no message cap.
    body = "\n".join(f"line-{i:04d} abcdefghij" for i in range(500)) + "\n"
    assert len(body) < bot._MAX_COPY_BODY
    chunks = bot._copy_plain_messages({"title": "T", "body": body, "footer": "F"})
    assert len(chunks) > bot._MAX_PLAIN_MESSAGES
    # every chunk within the 2000-char plain-message limit
    assert all(len(c) <= 2000 for c in chunks)
    # title on the first chunk, footer pinned on the last
    assert chunks[0].startswith("T")
    assert chunks[-1].endswith("F")
    # the FULL body survives (no 4-message cap): first and last lines present
    joined = "\n".join(chunks)
    assert "line-0000" in joined
    assert "line-0499" in joined


def test_copy_view_stores_rendered_dict():
    rendered = {"title": "T", "body": "**bold** line"}
    view = bot._copy_view(rendered)
    custom_id = view.children[0].custom_id
    assert custom_id.startswith("football_copy_")
    entry = bot._COPY_TEXTS.get(custom_id)
    assert entry is not None
    assert entry[1] == rendered  # full rendered dict, not plain text


def test_purge_removes_expired_and_caps_size():
    bot._COPY_TEXTS.clear()
    # Past under wall-clock semantics (the store compares against time.time()
    # so the TTL survives bot restarts).
    bot._COPY_TEXTS["expired"] = (time.time() - 2 * bot.COPY_TTL_SECONDS, {"title": "x"})
    bot._copy_view({"title": "T"})
    assert "expired" not in bot._COPY_TEXTS
    assert len(bot._COPY_TEXTS) <= bot._MAX_COPY_ENTRIES


def test_copy_state_roundtrip_survives_restart():
    """Entries persist to disk; a simulated restart (fresh process memory)
    restores them so old buttons keep working."""
    bot._COPY_TEXTS.clear()
    if bot._COPY_STATE_PATH.exists():
        bot._COPY_STATE_PATH.unlink()
    view = bot._copy_view({"title": "T", "body": "**B** line", "footer": "F"})
    cid = view.children[0].custom_id
    assert bot._COPY_STATE_PATH.exists()  # written on creation
    # restart: wipe memory, reload from disk
    bot._COPY_TEXTS.clear()
    bot._load_copy_texts()
    entry = bot._COPY_TEXTS.get(cid)
    assert entry is not None
    assert entry[1] == {"title": "T", "body": "**B** line", "footer": "F"}


def test_copy_state_expired_dropped_on_load():
    """Expired entries are not resurrected by a reload (TTL survives boot)."""
    bot._COPY_TEXTS.clear()
    if bot._COPY_STATE_PATH.exists():
        bot._COPY_STATE_PATH.unlink()
    view = bot._copy_view({"title": "T", "body": "B"})
    cid = view.children[0].custom_id
    # age the stored entry past TTL, persist, then reload -> dropped
    bot._COPY_TEXTS[cid] = (time.time() - 2 * bot.COPY_TTL_SECONDS, {"title": "T", "body": "B"})
    bot._save_copy_texts()
    bot._COPY_TEXTS.clear()
    bot._load_copy_texts()
    assert cid not in bot._COPY_TEXTS


def test_copy_state_consumed_button_not_resurrected():
    """A served button is removed from disk, so a later restart cannot serve
    the report a second time."""
    bot._COPY_TEXTS.clear()
    if bot._COPY_STATE_PATH.exists():
        bot._COPY_STATE_PATH.unlink()
    view = bot._copy_view({"title": "T", "body": "B"})
    cid = view.children[0].custom_id
    bot._COPY_TEXTS.pop(cid, None)  # what the click handler does
    bot._save_copy_texts()
    bot._COPY_TEXTS.clear()
    bot._load_copy_texts()
    assert cid not in bot._COPY_TEXTS


def test_copy_state_corrupt_file_starts_empty():
    """A corrupt state file must never crash the bot on startup."""
    bot._COPY_STATE_PATH.write_text("{not json!!", encoding="utf-8")
    bot._COPY_TEXTS.clear()
    bot._load_copy_texts()
    assert bot._COPY_TEXTS == {}
    bot._COPY_STATE_PATH.unlink(missing_ok=True)


def test_copy_payload_prefers_render_full_for_analyse():
    # The analyse command's main reply is a compact summary; the Copy button
    # must serve the FULL report (render_full) instead of the summary.
    result = {
        "render": {"title": "🔬 Analisa Match", "body": "⚽ A vs B — EPL\n🚫 NO BET", "footer": " "},
        "render_full": {"title": "🔬 Analisa Match", "body": "**A vs B**\nFINAL DECISION\nlineups...", "footer": " "},
        "raw": {},
    }
    payload = bot._result_payload(result)
    assert bot._copy_payload(result, payload) == result["render_full"]


def test_copy_payload_falls_back_to_main_render():
    # Non-analyse commands have no render_full -> Copy serves the main render.
    result = {"render": {"title": "T", "body": "B", "footer": "F"}, "raw": {}}
    payload = bot._result_payload(result)
    assert bot._copy_payload(result, payload) == result["render"]


def test_copy_payload_ignores_empty_render_full():
    result = {"render": {"title": "T", "body": "B"}, "render_full": {"title": ""}, "raw": {}}
    payload = bot._result_payload(result)
    assert bot._copy_payload(result, payload) == result["render"]
