"""Tests for the flashscore homepage extra-matches feature.

The homepage lists competitions football-data does not cover (Conference
League qualification, friendlies, minor cups). Covered here:
- multi_source.fetch_homepage_matches: normalization + flashscore-disabled skip
- match_finder.find_top_matches: extra_matches only for today (WIB), grouped
- format_top: renders the "Kompetisi lain" section without breaking the top
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.format import format_top  # noqa: E402
from agents.football.multi_source import MultiSourceStatsFetcher  # noqa: E402


class _FakeFlashscore:
    def __init__(self, rows):
        self.available = True
        self.rows = rows

    async def fetch_homepage_matches(self):
        return self.rows


def _homepage_rows():
    """Raw flashscore homepage rows (the shape fc.fetch_homepage_matches
    returns: flat home_*/away_* fields), as produced by scrape_homepage_matches."""
    return [
        {"home_name": "Ludogorets", "home_id": "a1",
         "away_name": "Petrocub", "away_id": "b1",
         "competition": "Conference League - Qualification", "date_text": "21:00",
         "match_url": "https://x/1", "status": "scheduled"},
        {"home_name": "Legia", "home_id": "a2",
         "away_name": "Brondby", "away_id": "b2",
         "competition": "Conference League - Qualification", "date_text": "22:00",
         "match_url": "https://x/2", "status": "scheduled"},
        {"home_name": "Inter", "home_id": "a3",
         "away_name": "Luton", "away_id": "b3",
         "competition": "Club Friendly", "date_text": "18:00",
         "match_url": "https://x/3", "status": "scheduled"},
        {"home_name": "X", "home_id": "a4",
         "away_name": "", "away_id": None,
         "competition": "Broken", "date_text": "", "match_url": None},
    ]


def test_fetch_homepage_matches_normalizes():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = _FakeFlashscore(_homepage_rows())
        out = await fetcher.fetch_homepage_matches()
        assert out is not None
        # Broken row (empty away name) is kept by multi_source; match_finder filters it.
        assert len(out) == 4
        assert out[0]["competition"] == "Conference League - Qualification"
        assert out[0]["source"] == "flashscore"
        assert out[0]["home"]["name"] == "Ludogorets"
    asyncio.run(runner())


def test_fetch_homepage_matches_skips_when_disabled():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fake = _FakeFlashscore(_homepage_rows())
        fake.available = False
        fetcher.fc = fake
        assert await fetcher.fetch_homepage_matches() is None
    asyncio.run(runner())


def test_fetch_homepage_matches_none_when_no_rows():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = _FakeFlashscore([])
        assert await fetcher.fetch_homepage_matches() is None
    asyncio.run(runner())


def _min_cfg() -> dict:
    return {
        "cache_ttl_seconds": {"fixtures": 300, "odds": 300},
        "outlier_threshold_pct": 10.0,
        "models": {},
    }


def test_find_top_matches_extra_only_for_today(tmp_path):
    """extra_matches are collected only when target_date == today (WIB)."""

    async def runner():
        from agents.football import match_finder
        from agents.football.cache import Cache

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = _FakeFlashscore(_homepage_rows())
        stats.fd.fetch_matches_by_competition = AsyncMock(return_value=[])

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False
        odds.rate_limit_warning = False

        from agents.football.timeutil import wib_today_iso

        with patch.object(match_finder, "_load_leagues", return_value={
            "EPL": {"display": "EPL", "odds_api_key": "soccer_epl", "football_data_code": "PL"},
        }):
            payload = await match_finder.find_top_matches(
                date=wib_today_iso(), leagues=["EPL"], top_n=5,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
            )
        extras = payload.get("extra_matches") or []
        # Broken row (empty away) must be filtered out; 3 valid remain.
        assert len(extras) == 3
        comps = {e["competition"] for e in extras}
        assert "Conference League - Qualification" in comps
        assert "Club Friendly" in comps
    asyncio.run(runner())


def test_find_top_matches_homepage_sentinel_only(tmp_path):
    """Regression: `leagues=['__homepage__']` (the bot's auto-detect path)
    must NOT raise NameError and must surface extra_matches.

    The homepage block used to key its cache on a league-loop variable
    (`target_date`), which is never bound when the ONLY league is the
    homepage sentinel -- the loop body never runs. The NameError was swallowed
    by the try/except, so auto-detect silently returned zero matches even
    when the fixture was on the flashscore homepage (fixed 2026-08).
    """

    async def runner():
        from agents.football import match_finder
        from agents.football.cache import Cache

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = _FakeFlashscore(_homepage_rows())

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False

        from agents.football.timeutil import wib_today_iso

        # IMPORTANT: `__homepage__` must NOT be a key in leagues_cfg. In
        # production the sentinel is absent from leagues.json, so the league
        # loop body (and its `target_date` binding) NEVER runs. Registering it
        # here would make the loop execute, bind `target_date`, and the test
        # would pass even with the old buggy code.
        with patch.object(match_finder, "_load_leagues", return_value={}):
            payload = await match_finder.find_top_matches(
                date=wib_today_iso(), leagues=["__homepage__"], top_n=1,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
            )
        extras = payload.get("extra_matches") or []
        # No NameError: the 3 valid scheduled rows flow through (broken row
        # with empty away is filtered, as in the EPL-path test).
        assert len(extras) == 3
        comps = {e["competition"] for e in extras}
        assert "Conference League - Qualification" in comps
        assert "Club Friendly" in comps
    asyncio.run(runner())


def test_find_top_matches_skips_extra_for_other_date(tmp_path):
    """A non-today date must NOT render the homepage (protects the deadline)."""

    async def runner():
        from agents.football import match_finder
        from agents.football.cache import Cache

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = _FakeFlashscore(_homepage_rows())
        stats.fd.fetch_matches_by_competition = AsyncMock(return_value=[])

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False

        with patch.object(match_finder, "_load_leagues", return_value={
            "EPL": {"display": "EPL", "odds_api_key": "soccer_epl", "football_data_code": "PL"},
        }):
            payload = await match_finder.find_top_matches(
                date="2020-01-01", leagues=["EPL"], top_n=5,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
            )
        assert (payload.get("extra_matches") or []) == []
    asyncio.run(runner())


def test_format_top_renders_extra_section():
    """No primary matches but Flashscore competitions exist -> the new grouped
    VALUE MATCH layout (header + KOMPETISI LAIN blocks), NOT the old compact
    counts line."""
    payload = {
        "date": "2026-08-12",
        "matches": [],
        "extra_matches": [
            {"home": "Ludogorets", "away": "Petrocub",
             "competition": "Conference League - Qualification", "kickoff": "21:00"},
            {"home": "Legia", "away": "Brondby",
             "competition": "Conference League - Qualification", "kickoff": "22:00"},
            {"home": "Inter", "away": "Luton",
             "competition": "Club Friendly", "kickoff": "18:00"},
        ],
        "quota": {},
        "leagues_no_odds": [],
    }
    out = format_top(payload)
    body = out["body"]
    assert "VALUE MATCH — 12 AGU 2026" in body
    assert "Tidak ada match ditemukan pada periode & liga tersebut." in body
    assert "KOMPETISI LAIN" in body
    # Analyzable competitions are rendered tagged with the league key; the
    # non-registered one (Club Friendly) appears under the info-only section.
    assert "🏆 **Conference League - Qualification · 2 match** (UECL)" in body
    assert "BELUM TERDAFTAR (info saja)" in body
    assert "🏆 **Club Friendly · 1 match**" in body
    # every match is listed (no preview cap, no "+N lainnya")
    assert "• Ludogorets vs Petrocub" in body
    assert "• Legia vs Brondby" in body
    assert "• Inter vs Luton" in body  # friendly listed info-only
    assert "+1 lainnya" not in body  # Conference League has exactly 2 -> no +N
    # footer totals computed from actual data (all competitions shown)
    footer = out.get("footer") or ""
    assert "3 MATCH" in footer
    assert "2 KOMPETISI" in footer
    assert "2 bisa dianalisa" in footer and "1 info saja" in footer
    # paginated pages available for the bot
    assert len(out.get("pages") or []) == 1
    assert len(body) < 1900  # fits one plain-text message without truncation


def _homepage_rows_with_status():
    """Rows carrying the flashscore status classifier (scheduled/live/finished)."""
    return [
        {"home_name": "Ludogorets", "home_id": "a1",
         "away_name": "Petrocub", "away_id": "b1",
         "competition": "Conference League - Qualification", "date_text": "21:00",
         "match_url": "https://x/1", "status": "scheduled"},
        {"home_name": "Legia", "home_id": "a2",
         "away_name": "Brondby", "away_id": "b2",
         "competition": "Conference League - Qualification", "date_text": "22:00",
         "match_url": "https://x/2", "status": "scheduled"},
        {"home_name": "Inter", "home_id": "a3",
         "away_name": "Luton", "away_id": "b3",
         "competition": "Club Friendly", "date_text": "FT",
         "match_url": "https://x/3", "status": "finished"},
        {"home_name": "Milan", "home_id": "a5",
         "away_name": "Roma", "away_id": "b5",
         "competition": "Club Friendly", "date_text": "62'",
         "match_url": "https://x/5", "status": "live"},
    ]


def test_find_top_matches_filters_finished_and_live(tmp_path):
    """Finished/live homepage rows must NOT surface as 'belum bertanding'."""

    async def runner():
        from agents.football import match_finder
        from agents.football.cache import Cache

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = _FakeFlashscore(_homepage_rows_with_status())
        stats.fd.fetch_matches_by_competition = AsyncMock(return_value=[])

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False

        from agents.football.timeutil import wib_today_iso

        with patch.object(match_finder, "_load_leagues", return_value={
            "EPL": {"display": "EPL", "odds_api_key": "soccer_epl", "football_data_code": "PL"},
        }):
            payload = await match_finder.find_top_matches(
                date=wib_today_iso(), leagues=["EPL"], top_n=5,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
            )
        extras = payload.get("extra_matches") or []
        names = {(e["home"], e["away"]) for e in extras}
        assert ("Ludogorets", "Petrocub") in names
        assert ("Legia", "Brondby") in names
        assert ("Inter", "Luton") not in names   # finished -> dropped
        assert ("Milan", "Roma") not in names    # live -> dropped
        comps = {e["competition"] for e in extras}
        assert "Club Friendly" not in comps
    asyncio.run(runner())


def test_find_top_matches_filters_finished_fixtures(tmp_path):
    """football-data fixtures that are FINISHED/IN_PLAY must not become candidates."""

    async def runner():
        from agents.football import match_finder
        from agents.football.cache import Cache

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = _FakeFlashscore([])
        stats.fetch_fixtures_for_date = AsyncMock(return_value=[
            {"id": 1, "home": {"id": 11, "name": "A FC"}, "away": {"id": 12, "name": "B FC"},
             "date": "2026-08-12T10:00:00Z", "status": "SCHEDULED", "source": "football_data"},
            {"id": 2, "home": {"id": 21, "name": "C FC"}, "away": {"id": 22, "name": "D FC"},
             "date": "2026-08-12T08:00:00Z", "status": "FINISHED", "source": "football_data"},
            {"id": 3, "home": {"id": 31, "name": "E FC"}, "away": {"id": 32, "name": "F FC"},
             "date": "2026-08-12T09:00:00Z", "status": "IN_PLAY", "source": "football_data"},
        ])
        stats.fetch_team_form = AsyncMock(return_value=None)

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False

        with patch.object(match_finder, "_load_leagues", return_value={
            "EPL": {"display": "EPL", "odds_api_key": "soccer_epl", "football_data_code": "PL"},
        }):
            payload = await match_finder.find_top_matches(
                date="2026-08-12", leagues=["EPL"], top_n=5,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
            )
        matches = payload.get("matches") or []
        assert len(matches) == 1
        assert matches[0]["home"] == "A FC"
        # canonical league key travels with the candidate so the bot can build
        # `analisa match <key> <home> vs <away>` for the ⚡ analyse button.
        assert matches[0]["league_key"] == "EPL"
    asyncio.run(runner())


def test_is_upcoming_status_filter():
    """_is_upcoming whitelists not-started statuses, drops finished/live rows."""
    from agents.football.match_finder import _is_upcoming

    assert _is_upcoming("SCHEDULED")
    assert _is_upcoming("TIMED")
    assert _is_upcoming("notstarted")
    assert _is_upcoming("scheduled")
    assert not _is_upcoming("FINISHED")
    assert not _is_upcoming("IN_PLAY")
    assert not _is_upcoming("PAUSED")
    assert not _is_upcoming("POSTPONED")
    assert not _is_upcoming("CANCELLED")
    assert not _is_upcoming("live")
    assert not _is_upcoming("finished")
    # Sofascore spelling variants must also be dropped.
    assert not _is_upcoming("inprogress")
    assert not _is_upcoming("canceled")
    assert not _is_upcoming("timeToBeDefined")
    assert not _is_upcoming("waiting")
    # Missing/unknown status falls back to the kickoff time.
    assert _is_upcoming(None, "2020-01-01T00:00:00Z") is False
    assert _is_upcoming(None, "2999-01-01T00:00:00Z") is True
    assert _is_upcoming(None, None) is True
    assert _is_upcoming("") is True


def test_row_status_classifier():
    """flashscore row classifier: class modifiers + text fallbacks."""
    from agents.football.flashscore import _row_status

    assert _row_status("event__time--finished", "event__match", "2:1") == "finished"
    assert _row_status("event__time--live", "event__match", "62'") == "live"
    assert _row_status("event__time--scheduled", "event__match", "21:00") == "scheduled"
    # Class modifier absent -> text fallbacks
    assert _row_status("", "", "FT") == "finished"
    assert _row_status("", "", "2-1") == "finished"
    assert _row_status("", "", "62'") == "live"
    assert _row_status("", "", "120'+3") == "live"     # extra time minute
    assert _row_status("", "", "HT") == "live"          # half-time: started
    assert _row_status("", "", "21:00") == "scheduled"  # clock, not a score
    assert _row_status("", "", "2:00") == "scheduled"   # padded minute = clock
    assert _row_status("", "", "3:3") == "finished"
    assert _row_status("", "", "CANC") == "finished"
    # No time element => the match already kicked off (flashscore swaps the
    # time cell for the score/minute). Not live => finished.
    assert _row_status("", "event__match", None, "Kauno Zalgiris (Ltu) vs Din. Zagreb (Cro) | 2:1") == "finished"
    assert _row_status("", "event__match", None, "Team A vs Team B | FT") == "finished"
    assert _row_status("", "event__match", None, None) == "finished"
    # Live minute inside the row text => live.
    assert _row_status("", "event__match", None, "62' | Team A vs Team B | 1:0") == "live"
    # Time element present (class truthy) without modifiers => scheduled.
    assert _row_status("event__time", "event__match", "21:00") == "scheduled"


def test_format_top_no_extra_without_payload():
    payload = {"date": "2026-08-12", "matches": [], "quota": {}, "leagues_no_odds": []}
    out = format_top(payload)
    assert "Tidak ada match ditemukan" in out["body"]
    assert not (out.get("pages") or [])


def test_homepage_js_no_single_backslash_n_hazard():
    """Regression guard: the JS sent to the browser must not contain a REAL
    newline character (chr 10) inside a single-quoted JS string.

    A Python triple-quoted string compiles backslash escapes, so writing a
    bare `'\n'` (single backslash) inside these JS blocks injects a real
    newline into the JS source, truncating a `//` comment or breaking a
    single-quoted string -> "Invalid or unexpected token" (JavascriptException)
    that only appears at runtime (reproduced live 2026-08). The `includes`
    line must stay DOUBLED (`'\\n'` in Python source) so the browser receives
    the valid JS escape `'\n'`.
    """
    from agents.football.flashscore import FlashscoreBrowserClient

    const = None
    for c in FlashscoreBrowserClient.scrape_homepage_matches.__code__.co_consts:
        if isinstance(c, str) and "SELECTOR" in c:
            const = c
            break
    assert const is not None, "homepage JS constant not found"

    # The includes guard compiles to JS `'\n'` (single backslash + n = the
    # valid JS newline escape), which means the Python source keeps the
    # DOUBLED `'\\n'` form. If a future edit writes a single backslash
    # `'\n'` in the Python source, Python compiles it into a REAL newline
    # character (chr 10) inside the single-quoted JS string -> the exact
    # "Invalid or unexpected token" bug. Both forms are asserted so either
    # regression direction is caught here.
    good = "includes('" + chr(92) + "n')"  # JS `'\n'` escape (source had doubled \\n)
    bad = "includes('" + chr(10) + "')"  # real newline (source had single \n)
    assert good in const, "homepage JS must compile to includes('\\n') (doubled source backslash)"
    assert bad not in const, "homepage JS must not contain a real newline inside the string (single source backslash)"
    # Every newline in the compiled JS must be a statement/comment line
    # separator, never inside a single-quoted string: assert no real chr(10)
    # sits between an opening quote and a closing quote on the same token.
    for i, ch in enumerate(const):
        if ch == chr(10):
            prefix = const[max(0, i - 3):i]
            suffix = const[i + 1:i + 2]
            assert not (prefix.endswith("'") or prefix.endswith('"')), \
                f"real newline inside JS string at offset {i}"
