"""Tests for the LLM intent -> handler mapping in bot.py.

The router may never produce runner arguments that the existing rule-based
handlers would reject; these tests pin the shared mapping down.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


def test_top_today():
    assert bot._intent_to_action("top", {"date": "today"}) == ("_handle_top", ["today"])


def test_top_besok():
    assert bot._intent_to_action("top", {"date": "besok"}) == ("_handle_top", ["besok"])


def test_top_iso_date():
    assert bot._intent_to_action("top", {"date": "2026-08-13"}) == ("_handle_top", ["2026-08-13"])


def test_top_leagues_and_topn():
    action = bot._intent_to_action("top", {"date": "besok", "leagues": ["ucl", "epl"], "top_n": 3})
    assert action == ("_handle_top", ["besok", "--leagues", "ucl,epl", "--top-n", "3"])


def test_top_no_args_defaults():
    assert bot._intent_to_action("top", {}) == ("_handle_top", [])


def test_compare_with_league():
    assert bot._intent_to_action("compare", {"home": "arsenal", "away": "chelsea", "league": "epl"}) == (
        "_handle_compare", ["arsenal", "chelsea", "epl"]
    )


def test_compare_without_league():
    assert bot._intent_to_action("compare", {"home": "arsenal", "away": "chelsea"}) == (
        "_handle_compare", ["arsenal", "chelsea"]
    )


def test_compare_missing_team():
    assert bot._intent_to_action("compare", {"home": "arsenal"}) is None


def test_analyse_reuses_handler_format():
    action = bot._intent_to_action("analyse", {"league": "ucl", "home": "lyon", "away": "sparta prague"})
    assert action == ("_handle_analyse", ["ucl lyon vs sparta prague"])


def test_analyse_missing_field():
    assert bot._intent_to_action("analyse", {"league": "ucl", "home": "lyon"}) is None


def test_analyse_league_optional_for_autodetect():
    # No league keyword -> the raw pair flows to the auto-detect path.
    action = bot._intent_to_action("analyse", {"home": "man utd", "away": "chelsea"})
    assert action == ("_handle_analyse", ["man utd vs chelsea"])


def test_stats_no_params():
    assert bot._intent_to_action("stats", {}) == ("_handle_stats", [])


def test_settle_manual():
    assert bot._intent_to_action("settle", {"home": "bodo", "away": "union", "result": "2-1"}) == (
        "_handle_settle", ["bodo", "vs", "union", "2-1"]
    )


def test_settle_auto():
    assert bot._intent_to_action("settle", {"auto": True}) == ("_handle_settle", ["auto"])


def test_settle_missing_result():
    assert bot._intent_to_action("settle", {"home": "bodo", "away": "union"}) is None


def test_odds_snapshot():
    action = bot._intent_to_action(
        "odds", {"timing": "T-6h", "home": "bodo", "away": "union", "odds": "1.62,4.30,4.60"}
    )
    assert action == ("_handle_odds_snapshot", ["T-6h", "bodo", "vs", "union", "1.62,4.30,4.60"])


def test_odds_missing_timing():
    assert bot._intent_to_action("odds", {"home": "a", "away": "b", "odds": "1,2,3"}) is None


def test_unknown_command():
    assert bot._intent_to_action("nonsense", {}) is None


def test_removed_command_has_no_mapping():
    # uclqualify was removed end-to-end: no handler, no intent mapping.
    assert "uclqualify" not in bot._HANDLERS
    assert bot._intent_to_action("uclqualify", {}) is None


def test_handlers_table_covers_all_known_commands():
    for cmd in ("top", "compare", "analyse", "stats", "settle", "odds",
                "best", "bestgoalmatch"):
        assert cmd in bot._HANDLERS
        assert callable(getattr(bot, bot._HANDLERS[cmd], None)), f"missing handler for {cmd}"


def _fake_msg():
    class Channel:
        @staticmethod
        async def send(*a, **k):
            pass

        @staticmethod
        def typing():
            class T:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

            return T()

    class FakeMsg:
        channel = Channel()

    return FakeMsg()


def test_llm_router_dispatch_happy_path(monkeypatch):
    # Valid LLM intent -> the SAME handler the rule-based path uses, with
    # the right args. This is the feature's core happy path.
    import asyncio
    import agents.football.llm_router as router

    calls: list[tuple[str, list]] = []

    async def fake_route(text):
        assert text == "bandingkan arsenal vs chelsea"
        return {"command": "compare", "params": {"home": "arsenal", "away": "chelsea", "league": "epl"}}

    async def spy_compare(message, args):
        calls.append(("compare", args))

    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setattr(router, "route_intent", fake_route)
    monkeypatch.setattr(bot, "_handle_compare", spy_compare)

    handled = asyncio.run(bot._handle_llm(_fake_msg(), "bandingkan arsenal vs chelsea"))
    assert handled is True
    assert calls == [("compare", ["arsenal", "chelsea", "epl"])]


def test_llm_router_dispatch_analyse_reuses_parse(monkeypatch):
    # analyse intent flows back through _handle_analyse's string parsing, so
    # league resolution + validation stay in one place.
    import asyncio
    import agents.football.llm_router as router

    calls: list[tuple[str, str]] = []

    async def fake_route(text):
        return {"command": "analyse", "params": {"league": "ucl", "home": "lyon", "away": "sparta prague"}}

    async def spy_analyse(message, rest):
        calls.append(("analyse", rest))

    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setattr(router, "route_intent", fake_route)
    monkeypatch.setattr(bot, "_handle_analyse", spy_analyse)

    handled = asyncio.run(bot._handle_llm(_fake_msg(), "analisa ucl lyon vs sparta"))
    assert handled is True
    assert calls == [("analyse", "ucl lyon vs sparta prague")]


def test_help_text_lists_all_commands():
    """The help message must cover every command the bot can dispatch."""
    for cmd in (
        "!football today", "!best <liga>", "!bestgoalmatch", "analisa match",
        "!football compare", "!football settle", "!football stats",
        "!football odds",
    ):
        assert cmd in bot.HELP_TEXT, f"help text missing {cmd}"
    assert "!football top" not in bot.HELP_TEXT  # renamed; alias stays internal
    assert "uclqualify" not in bot.HELP_TEXT
    # fit in one plain Discord message (2000 chars)
    assert len(bot.HELP_TEXT) < 2000
    assert "ngetik bebas" in bot.HELP_TEXT


def test_llm_router_help_intent_sends_help(monkeypatch):
    import asyncio
    import agents.football.llm_router as router

    sent: list[str] = []

    async def fake_route(text):
        return {"command": "help", "params": {"note": "Liga mana?"}}

    class Channel:
        @staticmethod
        async def send(*a, **k):
            sent.append(a[0])

        @staticmethod
        def typing():
            class T:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

            return T()

    class FakeMsg:
        channel = Channel()

    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setattr(router, "route_intent", fake_route)

    handled = asyncio.run(bot._handle_llm(FakeMsg(), "ada command apa?"))
    assert handled is True
    assert sent and "Liga mana?" in sent[0] and "!football today" in sent[0]


def test_llm_router_flag_off_does_not_call_route(tmp_path, monkeypatch):
    # Feature flag off -> router disabled even when env is configured.
    import asyncio
    import agents.football.llm_router as router

    async def never_called(text):
        raise AssertionError("route_intent must not be called when flag is off")

    monkeypatch.setenv("LLM_BASE_URL", "https://router.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setattr(router, "route_intent", never_called)

    # Mock the config read to return flag-off; plain string avoids recursion
    # into the patched read_text. Stale ts forces the TTL cache to reload.
    flag_off_cfg = '{"feature_flags": {"enable_llm_router": false}}'
    real_read = Path.read_text
    monkeypatch.setattr(
        Path, "read_text",
        lambda self, **k: flag_off_cfg if "football.json" in str(self) else real_read(self, **k),
    )
    monkeypatch.setattr(bot, "_LLM_FLAG_CACHE", {"ts": -1000.0, "value": True})

    class FakeMsg:
        class Channel:
            @staticmethod
            async def send(*a, **k):
                pass

            @staticmethod
            def typing():
                class T:
                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *a):
                        pass

                return T()

        channel = Channel()

    handled = asyncio.run(bot._handle_llm(FakeMsg(), "besok ada apa?"))
    assert handled is False


def test_llm_router_not_configured_never_calls_route(monkeypatch):
    # Placeholder .env values -> is_configured() False -> no LLM call at all.
    import asyncio
    import agents.football.llm_router as router

    async def never_called(text):
        raise AssertionError("route_intent must not be called when unconfigured")

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(router, "route_intent", never_called)

    class FakeMsg:
        class Channel:
            @staticmethod
            async def send(*a, **k):
                pass

            @staticmethod
            def typing():
                class T:
                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *a):
                        pass

                return T()

        channel = Channel()

    handled = asyncio.run(bot._handle_llm(FakeMsg(), "besok ada apa?"))
    assert handled is False
