"""Provider chain smoke tests.

Verifies the football_data -> thesportsdb chain (with SoccerData for EU
leagues) responds as documented. Flashscore is the primary provider and is
exercised by the live-flow tests; sofascore fallbacks were removed from the
live path (2026-08).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.multi_source import MultiSourceStatsFetcher


def _meta(provider: str = "football_data", fallback: str = "thesportsdb") -> dict:
    return {
        "provider": provider,
        "fallback_provider": fallback,
        "country": "Spain",
        "football_data_code": "PD",
        "display": "La Liga",
        "_league_key": "LaLiga",
    }


def test_search_team_falls_back_to_football_data():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value={
            "id": 77, "name": "Sevilla", "shortName": "Sevilla",
            "tla": "SEV", "area": {"name": "Spain"},
        })
        fetcher.ts.search_team = AsyncMock(
            side_effect=AssertionError("should not reach thesportsdb")
        )
        result = await fetcher.search_team("Sevilla", _meta())
        assert result is not None
        assert result["provider"] == "football_data"
        assert result["id"] == 77
    asyncio.run(runner())


def test_search_team_falls_back_to_thesportsdb_when_fd_misses():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value=None)
        fetcher.ts.search_team = AsyncMock(return_value={
            "idTeam": "1234", "strTeam": "Debrecen", "strTeamShort": "DEB",
            "strCountry": "Hungary",
        })
        result = await fetcher.search_team("Debrecen", _meta())
        assert result is not None
        assert result["provider"] == "thesportsdb"
        assert result["id"] == "1234"
    asyncio.run(runner())


def test_search_team_none_when_all_providers_miss():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value=None)
        fetcher.ts.search_team = AsyncMock(return_value=None)
        result = await fetcher.search_team("Unknown FC", _meta())
        assert result is None
    asyncio.run(runner())


def test_fetch_team_form_uses_soccerdata_for_eu_league():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        # Flashscore OFF: this test exercises the FBref branch only (flashscore
        # by-name would now resolve "Arsenal" via the livesport search API and
        # win as primary -- not what this test asserts).
        fetcher.fc = None
        fetcher.sd.read_team_form = AsyncMock(return_value={
            "sequence": "W-W-D", "source": "soccerdata_fbref",
        })
        fetcher.fd.fetch_last_matches = AsyncMock(
            side_effect=AssertionError("should not fallback")
        )

        meta = _meta()
        meta["_league_key"] = "EPL"
        meta["_team_names"] = {"42": "Arsenal"}
        result = await fetcher.fetch_team_form(42, meta, limit=3)
        assert result is not None
        assert result["source"] == "soccerdata_fbref"
        assert result["sequence"] == "W-W-D"
    asyncio.run(runner())


def test_fetch_team_form_falls_back_to_football_data():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.sd.read_team_form = AsyncMock(return_value=None)
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=[])
        fetcher._thesportsdb_form = AsyncMock(return_value=None)

        meta = _meta()
        meta["_league_key"] = "Liga 1"
        result = await fetcher.fetch_team_form(42, meta, limit=3)
        assert result is None
    asyncio.run(runner())


def test_fetch_team_form_falls_back_to_thesportsdb_last():
    """F1: an int (football-data) id reaches thesportsdb ONLY via name
    resolution -- the resolved thesportsdb id is used, never the fd int."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None  # flashscore OFF: exercise the F1 name-resolution
        fetcher.sd.read_team_form = AsyncMock(return_value=None)
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=[])
        fetcher.search_team = AsyncMock(return_value={
            "id": "77", "name": "Sevilla", "provider": "thesportsdb",
        })
        fetcher._thesportsdb_form = AsyncMock(return_value={
            "sequence": "W-D-W", "gf_avg": 1.6, "ga_avg": 0.6,
            "sample_size": 3, "source": "thesportsdb",
        })
        meta = _meta()
        meta["_league_key"] = "Liga 1"
        meta["_team_names"] = {"42": "Sevilla"}
        result = await fetcher.fetch_team_form(42, meta, limit=3)
        assert result is not None
        assert result["source"] == "thesportsdb"
        assert result["sequence"] == "W-D-W"
        # F1: thesportsdb must receive the resolved provider id, never the
        # football-data int 42
        fetcher._thesportsdb_form.assert_awaited_once_with("77", 3)
    asyncio.run(runner())


def test_fetch_h2h_uses_football_data_when_available():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.fetch_h2h = AsyncMock(return_value=[
            {"winner": {"id": 42}}, {"winner": {"id": 99}}, {"winner": None},
        ])
        result = await fetcher.fetch_h2h(42, 99, _meta())
        assert result is not None
        assert result["source"] == "football_data"
        assert result["wins"] == 1
        assert result["draws"] == 1
        assert result["losses"] == 1
    asyncio.run(runner())


def test_fetch_h2h_uses_soccerdata_when_available():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.fetch_h2h = AsyncMock(return_value=[])
        fetcher.sd.supports_league = lambda lk: True
        fetcher.sd.read_h2h = AsyncMock(return_value={
            "wins": 1, "draws": 0, "losses": 1,
            "sample_size": 2, "source": "soccerdata_fbref_h2h",
        })
        meta = _meta()
        meta["_team_names"] = {"42": "Santa Clara", "99": "Nacional"}
        result = await fetcher.fetch_h2h(42, 99, meta)
        assert result is not None
        assert result["source"] == "soccerdata_fbref_h2h"
        assert result["wins"] == 1
    asyncio.run(runner())


def test_fetch_fixtures_for_date_uses_football_data():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.fetch_matches_by_competition = AsyncMock(return_value=[
            {"id": 9001, "homeTeam": {"id": 1, "name": "A"},
             "awayTeam": {"id": 2, "name": "B"},
             "utcDate": "2026-08-12T10:00:00Z", "status": "TIMED"},   # 17:00 WIB same day
            {"id": 9002, "homeTeam": {"id": 3, "name": "C"},
             "awayTeam": {"id": 4, "name": "D"},
             "utcDate": "2026-08-12T19:00:00Z", "status": "TIMED"},   # 02:00 WIB NEXT day
        ])
        meta = _meta()
        meta["football_data_code"] = "CL"
        meta["_league_key"] = "UCL"
        # target_date is the WIB calendar day; UTC-evening matches belong to
        # the following WIB day and must be excluded (timezone fix).
        out = await fetcher.fetch_fixtures_for_date(meta, "2026-08-12")
        assert len(out) == 1
        assert out[0]["home"]["name"] == "A"
        assert out[0]["away"]["name"] == "B"
        assert out[0]["source"] == "football_data"
        # A match at UTC 2026-08-11T19:00 (02:00 WIB Aug 12) IS on the target
        # WIB day even though its UTC date is the previous day.
        fetcher.fd.fetch_matches_by_competition = AsyncMock(return_value=[
            {"id": 9003, "homeTeam": {"id": 5, "name": "E"},
             "awayTeam": {"id": 6, "name": "F"},
             "utcDate": "2026-08-11T19:00:00Z", "status": "TIMED"},
        ])
        out2 = await fetcher.fetch_fixtures_for_date(meta, "2026-08-12")
        assert len(out2) == 1
        assert out2[0]["home"]["name"] == "E"
    asyncio.run(runner())


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
