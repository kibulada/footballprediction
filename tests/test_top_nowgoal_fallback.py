"""Tests for the nowgoal schedule fallback in find_top_matches.

football-data free tier is 10 req/min while `!football today` costs ONE
request per league, so past the 10th league every call 429s and the day
comes back empty. When football-data has no fixtures, the top pipeline now
falls back to the nowgoal schedule (ONE request covers ALL leagues, via the
on-demand Tor proxy). Covered here:

- _nowgoal_league_matches: league-name matching (display + aliases)
- _nowgoal_fixtures_for_league: shape normalization to the football-data
  fixture shape the rest of the pipeline consumes
- find_top_matches: nowgoal fallback active only when football-data is empty
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

from agents.football.cache import Cache  # noqa: E402
from agents.football.multi_source import MultiSourceStatsFetcher  # noqa: E402


def _min_cfg() -> dict:
    return {
        "cache_ttl_seconds": {"fixtures": 300, "odds": 300},
        "outlier_threshold_pct": 10.0,
        "models": {},
    }


def _nowgoal_rows():
    """Raw nowgoal schedule rows (shape nowgoal._parse_match_row produces)."""
    return [
        {
            "match_id": "9001",
            "home": "Arsenal FC",
            "away": "Chelsea FC",
            "home_id": "101",
            "away_id": "102",
            "kickoff": "2026-08-15T13:00:00Z",
            "status": "1",
            "league_id": "36",
            "league_name": "Premier League",
            "source": "nowgoal",
        },
        {
            "match_id": "9002",
            "home": "Los Angeles FC",
            "away": "Inter Miami CF",
            "home_id": "201",
            "away_id": "202",
            "kickoff": "2026-08-15T23:30:00Z",
            "status": "1",
            "league_id": "44",
            "league_name": "Major League Soccer",
            "source": "nowgoal",
        },
        {
            "match_id": "9003",
            "home": "Real Madrid",
            "away": "Barcelona",
            "home_id": "301",
            "away_id": "302",
            "kickoff": "2026-08-15T19:00:00Z",
            "status": "1",
            "league_id": "5",
            "league_name": "La Liga",
            "source": "nowgoal",
        },
        {
            "match_id": "9004",
            "home": "Persija Jakarta",
            "away": "Persib Bandung",
            "home_id": "401",
            "away_id": "402",
            "kickoff": "2026-08-15T09:00:00Z",
            "status": "1",
            "league_id": "77",
            "league_name": "Indonesia Liga 1",
            "source": "nowgoal",
        },
    ]


def _epL_meta():
    return {"display": "EPL", "aliases": ["epl", "premier league", "english"]}


def test_nowgoal_league_matches_display_and_alias():
    from agents.football.match_finder import _nowgoal_league_matches

    # Display name match.
    assert _nowgoal_league_matches("Premier League", _epL_meta())
    # Alias match.
    assert _nowgoal_league_matches("English Premier League", _epL_meta())
    assert _nowgoal_league_matches("EPL", _epL_meta())
    # Wrong league -> no match.
    assert not _nowgoal_league_matches("La Liga", _epL_meta())
    # Empty name -> no match.
    assert not _nowgoal_league_matches("", _epL_meta())


def test_nowgoal_league_matches_mls_short_alias():
    from agents.football.match_finder import _nowgoal_league_matches

    mls = {"display": "MLS", "aliases": ["mls", "major league soccer", "amerika"]}
    # nowgoal's longer name contains the short alias token.
    assert _nowgoal_league_matches("Major League Soccer", mls)
    assert _nowgoal_league_matches("MLS", mls)


def test_nowgoal_league_matches_liga1():
    from agents.football.match_finder import _nowgoal_league_matches

    liga1 = {"display": "Liga 1", "aliases": ["liga 1", "indonesia"]}
    assert _nowgoal_league_matches("Indonesia Liga 1", liga1)
    assert not _nowgoal_league_matches("Ligue 1", liga1)  # France != Indonesia


def test_nowgoal_fixtures_for_league_normalizes_shape():
    from agents.football.match_finder import _nowgoal_fixtures_for_league

    class _FakeNowgoal:
        async def fetch_schedule(self, date):
            return _nowgoal_rows()

    async def runner():
        rows = await _nowgoal_fixtures_for_league(
            _FakeNowgoal(), "2026-08-15", "EPL", _epL_meta()
        )
        assert len(rows) == 1
        r = rows[0]
        # Same shape football-data produces: {id, home:{id,name}, away:{id,name},
        # date, status, source} -- the rest of the top pipeline is unchanged.
        assert r["id"] == "9001"
        assert r["home"] == {"id": "101", "name": "Arsenal FC"}
        assert r["away"] == {"id": "102", "name": "Chelsea FC"}
        assert r["date"] == "2026-08-15T13:00:00Z"
        assert r["status"] == "1"
        assert r["source"] == "nowgoal"

    asyncio.run(runner())


def test_find_top_matches_no_nowgoal_when_football_data_ok(tmp_path):
    """football-data answers -> the nowgoal fallback is NOT invoked (zero
    overhead on the happy path)."""

    async def runner():
        from agents.football import match_finder
        from agents.football.timeutil import wib_today_iso

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = AsyncMock()
        stats.fetch_fixtures_for_date = AsyncMock(return_value=[
            {"id": 1, "home": {"id": 11, "name": "Arsenal FC"}, "away": {"id": 12, "name": "Chelsea FC"},
             "date": "2026-08-15T13:00:00Z", "status": "SCHEDULED", "source": "football_data"},
        ])
        stats.fetch_team_form = AsyncMock(return_value=None)

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False
        odds.rate_limit_warning = False

        called = {"n": 0}

        class _FakeNowgoal:
            async def fetch_schedule(self, date):
                called["n"] += 1
                return _nowgoal_rows()

        with patch.object(match_finder, "_load_leagues", return_value={
            "EPL": {"display": "EPL", "odds_api_key": "soccer_epl",
                    "football_data_code": "PL", "aliases": ["epl", "premier league"]},
        }):
            payload = await match_finder.find_top_matches(
                date=wib_today_iso(), leagues=["EPL"], top_n=5,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
                nowgoal=_FakeNowgoal(),
            )
        matches = payload.get("matches") or []
        assert len(matches) == 1
        assert matches[0]["source"] == "football_data"
        assert called["n"] == 0  # nowgoal never touched
        assert payload["quota"]["nowgoal_fixtures_used"] is False

    asyncio.run(runner())


def test_find_top_matches_no_nowgoal_param_stays_empty(tmp_path):
    """No nowgoal client (feature off) -> old behaviour: empty day, no crash."""

    async def runner():
        from agents.football import match_finder
        from agents.football.timeutil import wib_today_iso

        stats = MultiSourceStatsFetcher("fd", "")
        stats.fc = AsyncMock()
        stats.fetch_fixtures_for_date = AsyncMock(return_value=[])

        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[])
        odds.last_remaining = 500
        odds.quota_blocked = False
        odds.rate_limit_warning = False

        with patch.object(match_finder, "_load_leagues", return_value={
            "EPL": {"display": "EPL", "odds_api_key": "soccer_epl", "football_data_code": "PL"},
        }):
            payload = await match_finder.find_top_matches(
                date=wib_today_iso(), leagues=["EPL"], top_n=5,
                cfg=_min_cfg(), odds=odds, stats=stats,
                cache=Cache(str(tmp_path)),
                nowgoal=None,
            )
        assert (payload.get("matches") or []) == []
        assert payload["quota"]["nowgoal_fixtures_used"] is False

    asyncio.run(runner())


if __name__ == "__main__":
    import inspect
    import tempfile

    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(td)
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
