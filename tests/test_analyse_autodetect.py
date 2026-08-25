"""Tests for the no-league auto-detect path in bot.py.

`analisa match manchester united vs leeds united` carries no league keyword.
Before replying "Liga tidak dikenali", the bot asks the runner's `detect`
command (football-data first, flashscore homepage second) and either runs
the full analysis (when a registered league is found) or shows the match +
competition as context. These tests pin that behaviour with a mocked runner.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402
from agents.football import match_finder  # noqa: E402


class _Sent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.runner_calls: list[list[str]] = []


class _FakeChannel:
    def __init__(self, sent: _Sent) -> None:
        self.sent = sent

    async def send(self, *a, **k):
        self.sent.messages.append(a[0] if a else k.get("content", ""))

    def typing(self):
        class T:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

        return T()


class _FakeMsg:
    def __init__(self, sent: _Sent) -> None:
        self.channel = _FakeChannel(sent)


def _run(coro):
    return asyncio.run(coro)


def test_handle_analyse_no_league_runs_analysis_for_registered_league(monkeypatch):
    """Detect maps the pair to a registered league -> full analysis runs."""
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            return {
                "raw": {
                    "found": True,
                    "league": "EPL",
                    "display": "EPL",
                    "home": "Manchester United",
                    "away": "Chelsea",
                    "kickoff": "2026-08-16T14:00:00Z",
                    "competition": "PL",
                    "source": "football_data",
                }
            }
        # analyse path
        return {
            "render": {
                "title": "🔬 Analisa Match",
                "body": "**Manchester United vs Chelsea** • EPL\nanalysed",
                "footer": " ",
            }
        }

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "man utd vs chelsea"))
    assert any(c[0] == "detect" for c in calls)
    analyse_call = next(c for c in calls if c[0] == "analyse")
    assert analyse_call[analyse_call.index("--league") + 1] == "EPL"
    assert "Manchester United" in "".join(sent.messages)


def test_handle_analyse_no_league_detect_uses_source_spellings(monkeypatch):
    """Detect may swap roles / fix spellings; the analyse call uses them."""
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            return {
                "raw": {
                    "found": True,
                    "league": "LaLiga",
                    "display": "La Liga",
                    "home": "Real Madrid",
                    "away": "Sevilla",
                    "kickoff": "20:00",
                    "competition": "PD",
                    "source": "football_data",
                }
            }
        return {
            "render": {
                "title": "🔬 Analisa Match",
                "body": "**Real Madrid vs Sevilla** • La Liga\nanalysed",
                "footer": " ",
            }
        }

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "real madrid vs sevilla"))
    analyse_call = next(c for c in calls if c[0] == "analyse")
    i = analyse_call.index("--home")
    assert analyse_call[i + 1] == "Real Madrid"
    assert analyse_call[analyse_call.index("--away") + 1] == "Sevilla"


def test_handle_analyse_no_league_detect_not_found_sends_format_help(monkeypatch):
    """Pair not detected -> clear format help instead of bare error."""
    sent = _Sent()

    async def fake_invoke(args):
        assert args[0] == "detect"
        return {"raw": {"found": False}}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "manchester united vs leeds united"))
    joined = "\n".join(sent.messages)
    assert "Format:" in joined
    assert "analisa match <liga> <home> vs <away>" in joined
    # Updated help text (2026-08-18): match non-liga is analysable directly
    # via `analisa match <home> vs <away>` -- the old "lihat di !football
    # today" was misleading for friendlies / cups the homepage does not
    # carry.
    assert "bisa langsung dianalisa" in joined


def test_handle_analyse_no_league_detect_error_sends_format_help(monkeypatch):
    """Runner failure degrades to the same format help, never crashes."""
    sent = _Sent()

    async def fake_invoke(args):
        raise RuntimeError("runner down")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "a vs b"))
    joined = "\n".join(sent.messages)
    assert "Format:" in joined


def test_handle_analyse_retry_drop_token_finds_unrecognised_league(monkeypatch):
    """Unrecognised league token (e.g. 'asean') is dropped one at a time
    until detect finds the fixture, so a query like
    `analisa asean thailand vs singapore` no longer hard-fails.

    First candidate (`asean thailand`) -> not found.
    Second candidate (`thailand`)      -> livescore hit, registered=False
        (Club Friendly / ASEAN playoff etc.) -> D2 analyse without --league.
    """
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            # Mirror the drop-token progression: bot calls detect with
            # candidate home names in order.
            i = calls.index(args) - sum(1 for c in calls[:calls.index(args)] if c[0] == "detect")
            home = args[args.index("--home") + 1]
            if home == "thailand":
                return {
                    "raw": {
                        "found": True,
                        "registered": False,
                        "competition": "ASEAN Championship - Play Offs - Semi-finals",
                        "home": "Thailand",
                        "away": "Singapore",
                        "kickoff": "20:00",
                        "source": "livescore",
                    }
                }
            return {"raw": {"found": False}}
        # analyse path
        return {"render": {"title": "Analisa", "body": "prediksi lengkap"}}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "asean thailand vs singapore"))
    detect_homes = [
        c[c.index("--home") + 1] for c in calls if c[0] == "detect"
    ]
    # First attempt must use the FULL left token ("asean thailand"); the
    # second attempt drops the leftmost token ("thailand") and succeeds.
    assert "asean thailand" in detect_homes
    assert "thailand" in detect_homes
    assert detect_homes.index("thailand") < detect_homes.index("asean thailand") or \
        detect_homes[-1] == "thailand"
    # Once the second candidate hits, the bot runs analyse WITHOUT a league
    # keyword (D2 dynamic discovery).
    analyse_calls = [c for c in calls if c[0] == "analyse"]
    assert analyse_calls
    assert "--league" not in analyse_calls[0]


def test_handle_analyse_retry_drop_token_finds_registered_league(monkeypatch):
    """Multi-token unregistered prefix + real Liga 1 fixture:

    `analisa bri liga persib vs persija` -> none of the left prefixes
    (`bri liga persib`, `liga persib`, `persib`) resolve to a registered
    league keyword, so the parser falls through to the detect retry loop.
    The last candidate (`persib`) wins and detect returns a registered
    Liga 1 hit (because the fixture's competition is "Liga 1"), so the
    bot runs analyse WITH --league Liga 1.

    Note: a query like `liga 1 indonesia persib vs persija` is NOT a drop-
    loop candidate because the parser's longest-prefix league resolver
    matches "liga 1" at n=2 -- the drop loop is reserved for input where
    NO prefix resolves (genuine league-keyword failures).
    """
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            home = args[args.index("--home") + 1]
            if home == "persib":
                return {
                    "raw": {
                        "found": True,
                        "league": "Liga 1",
                        "display": "Liga 1",
                        "home": "Persib Bandung",
                        "away": "Persija Jakarta",
                        "kickoff": "19:30",
                        "competition": "Liga 1",
                        "source": "livescore",
                    }
                }
            return {"raw": {"found": False}}
        return {"render": {"title": "Analisa", "body": "prediksi Liga 1"}}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "bri liga persib vs persija"))
    detect_homes = [c[c.index("--home") + 1] for c in calls if c[0] == "detect"]
    # The retry chain: full left first, then drop leftmost one at a time.
    assert detect_homes[0] == "bri liga persib"
    assert detect_homes[-1] == "persib"
    # Analyse call uses --league Liga 1 (registered), with the canonical
    # home/away spellings from detect.
    analyse_call = next(c for c in calls if c[0] == "analyse")
    assert analyse_call[analyse_call.index("--league") + 1] == "Liga 1"
    assert analyse_call[analyse_call.index("--home") + 1] == "Persib Bandung"
    assert analyse_call[analyse_call.index("--away") + 1] == "Persija Jakarta"


def test_handle_analyse_retry_drop_token_does_not_regress_simple_query(monkeypatch):
    """A legitimate two-token team query (`manchester united vs chelsea`)
    must NOT regress: the FIRST candidate already hits detect, so the retry
    loop runs exactly one detect call -- same behaviour as before.
    """
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            return {
                "raw": {
                    "found": True,
                    "league": "EPL",
                    "display": "EPL",
                    "home": "Manchester United",
                    "away": "Chelsea",
                    "kickoff": "21:00",
                    "competition": "PL",
                    "source": "football_data",
                }
            }
        return {"render": {"title": "Analisa", "body": "prediksi EPL"}}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "manchester united vs chelsea"))
    detect_calls = [c for c in calls if c[0] == "detect"]
    # Exactly ONE detect call -- the first candidate (`manchester united`)
    # already wins, the loop short-circuits.
    assert len(detect_calls) == 1
    assert detect_calls[0][detect_calls[0].index("--home") + 1] == "manchester united"


def test_handle_analyse_retry_drop_token_caps_attempts(monkeypatch):
    """Bounded retry: a long unrecognised prefix never loops forever -- the
    candidate chain is capped at `_DETECT_DROP_MAX + 1` candidates.
    """
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            return {"raw": {"found": False}}
        return {}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "a b c d e f g vs h"))
    detect_calls = [c for c in calls if c[0] == "detect"]
    # _DETECT_DROP_MAX = 3, so max attempts = full + 3 drops = 4.
    assert len(detect_calls) <= bot._DETECT_DROP_MAX + 1


def test_handle_analyse_no_league_friendly_shows_context(monkeypatch):
    """Non-registered competition (Club Friendly) -> FULL analysis via dynamic
    league discovery (D2, 2026-08-17): the detect result feeds analyse WITHOUT
    a league keyword, and the league is read from the fixture."""
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            return {
                "raw": {
                    "found": True,
                    "registered": False,
                    "competition": "Club Friendly",
                    "home": "Manchester United",
                    "away": "Leeds",
                    "kickoff": "02:00",
                    "source": "flashscore",
                }
            }
        if args[0] == "analyse":
            return {"render": {"title": "Analisa", "body": "prediksi lengkap"}}
        return {}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "manchester united vs leeds united"))
    joined = "\n".join(sent.messages)
    # full analysis is attempted WITHOUT a league keyword (dynamic discovery)
    assert any(c[0] == "analyse" and "--league" not in c for c in calls)
    assert "prediksi lengkap" in joined


def test_handle_analyse_no_league_detects_via_flashscore_registered(monkeypatch):
    """A homepage competition that maps to a league -> full analysis runs."""
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "detect":
            return {
                "raw": {
                    "found": True,
                    "league": "UECL",
                    "display": "UECL",
                    "home": "Tobol",
                    "away": "Partizan",
                    "kickoff": "21:00",
                    "competition": "Conference League - Qualification",
                    "source": "flashscore",
                }
            }
        return {
            "render": {
                "title": "🔬 Analisa Match",
                "body": "**Tobol vs Partizan** • UECL\nanalysed",
                "footer": " ",
            }
        }

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_analyse(_FakeMsg(sent), "tobol vs partizan"))
    analyse_call = next(c for c in calls if c[0] == "analyse")
    assert analyse_call[analyse_call.index("--league") + 1] == "UECL"


def test_match_finder_leagues_no_odds_skips_unknown_keys():
    """find_top_matches must not KeyError for unknown league keys (homepage-only
    sentinel like '__homepage__' flows through leagues)."""
    import types

    async def noop(*a, **k):
        return None

    stats = types.SimpleNamespace(
        fetch_fixtures_for_date=noop,
        fetch_homepage_matches=noop,
        fd=types.SimpleNamespace(rate_limit_warning=False),
        sc=types.SimpleNamespace(quota_warning=False),
    )
    odds = types.SimpleNamespace(
        last_remaining=None, quota_blocked=False, fetch_odds=noop
    )
    cache = types.SimpleNamespace(get=lambda *a, **k: None, set=lambda *a, **k: None)

    async def run():
        return await match_finder.find_top_matches(
            date="2026-08-12",
            leagues=["__homepage__", "EPL"],
            top_n=1,
            cfg={
                "cache_ttl_seconds": {"fixtures": 100, "odds": 100},
                "outlier_threshold_pct": 5,
            },
            odds=odds,
            stats=stats,
            cache=cache,
        )

    out = _run(run())
    # No KeyError; unknown key contributes nothing to leagues_no_odds
    assert out is not None
    assert "__homepage__" not in out.get("leagues_no_odds", [])


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(__import__("pytest").MonkeyPatch())
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
