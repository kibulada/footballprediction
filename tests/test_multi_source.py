"""Tests for multi_source.py aggregated provider router."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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


def test_init_stores_clients():
    fetcher = MultiSourceStatsFetcher("fd", "")
    assert fetcher.fd is not None
    assert fetcher.ts is not None
    assert fetcher.fc is not None  # flashscore (primary)
    assert fetcher.sd is not None


def test_search_team_football_data_succeeds():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value={
            "id": 42, "name": "Arsenal", "shortName": "Arsenal", "tla": "ARS",
            "area": {"name": "England"}
        })
        result = await fetcher.search_team("Arsenal", _meta())
        assert result is not None
        assert result["id"] == 42
        assert result["provider"] == "football_data"
        assert result["country"] == "England"
    asyncio.run(runner())


def test_search_team_uses_alias():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value={"id": 50, "name": "Manchester City FC", "shortName": "Man City", "tla": "MCI", "area": {"name": "England"}})
        meta = _meta()
        meta["_league_key"] = "EPL"
        result = await fetcher.search_team("MCN", meta)
        assert result is not None
        assert result["id"] == 50
        assert result["_aliased"] is True
    asyncio.run(runner())


def test_search_team_all_fail():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value=None)
        fetcher.ts.search_team = AsyncMock(return_value=None)
        result = await fetcher.search_team("NoMatch", _meta())
        assert result is None
    asyncio.run(runner())


def test_fetch_team_form_football_data():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        matches = [
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 2, "away": 1}}},
            {"homeTeam": {"id": 99}, "awayTeam": {"id": 42},
             "score": {"fullTime": {"home": 0, "away": 1}}},
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 1, "away": 1}}},
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 3, "away": 0}}},
            {"homeTeam": {"id": 99}, "awayTeam": {"id": 42},
             "score": {"fullTime": {"home": 2, "away": 0}}},
        ]
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=matches)
        result = await fetcher.fetch_team_form(42, _meta(), limit=5)
        assert result is not None
        assert result["sequence"] == "W-W-D-W-L"
        assert result["source"] == "football_data"
    asyncio.run(runner())


def test_fetch_team_form_recent_goals_football_data():
    """recent_goals exposes raw per-match (gf, ga) scorelines oldest->newest
    so live predictions can use the same time-decay features as validation."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        matches = [
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 2, "away": 1}}},
            {"homeTeam": {"id": 99}, "awayTeam": {"id": 42},
             "score": {"fullTime": {"home": 0, "away": 1}}},
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 1, "away": 1}}},
            {"homeTeam": {"id": 99}, "awayTeam": {"id": 42},
             "score": {"fullTime": {"home": 2, "away": 0}}},
        ]
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=matches)
        result = await fetcher.fetch_team_form(42, _meta(), limit=5)
        assert result is not None
        # team 42 perspective: (gf, ga) per match, oldest -> newest
        assert result["recent_goals"] == [(2, 1), (1, 0), (1, 1), (0, 2)]
    asyncio.run(runner())


def test_fetch_team_form_extended_with_goals():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        matches = [
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 2, "away": 1}}},
            {"homeTeam": {"id": 99}, "awayTeam": {"id": 42},
             "score": {"fullTime": {"home": 0, "away": 1}}},
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 1, "away": 1}}},
        ]
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=matches)
        result = await fetcher.fetch_team_form(42, _meta(), limit=5)
        assert result is not None
        assert result["sequence"] == "W-W-D"
        assert result["gf_avg"] == 4 / 3
        assert result["home"]["w"] == 1
        assert result["away"]["w"] == 1
    asyncio.run(runner())


def test_fetch_h2h_football_data():
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fd.fetch_h2h = AsyncMock(return_value=[
            {"winner": {"id": 42}},
            {"winner": None},
            {"winner": {"id": 99}},
        ])
        result = await fetcher.fetch_h2h(42, 99, _meta())
        assert result is not None
        assert result["wins"] == 1
        assert result["draws"] == 1
        assert result["losses"] == 1
        assert result["source"] == "football_data"
    asyncio.run(runner())


def test_fetch_team_xg_history_flashscore_fallback():
    """xG fallback: when the NowGoal tier is absent (no client / no
    match_list), the rolling window is rebuilt from flashscore team-results
    + per-match stats -- any league flashscore covers."""
    async def runner():
        from agents.football import flashscore as fs_mod

        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_league_key"] = "UEL"
        fetcher.fc = SimpleNamespace(
            available=True,
            fetch_team_results=AsyncMock(return_value=[
                {"date": "14.08.2026", "home": "Real Betis", "away": "Sevilla",
                 "hg": "2", "ag": "1", "match_url": "https://www.flashscore.com/match/a/"},
                {"date": "09.08.2026", "home": "Sevilla", "away": "Cadiz",
                 "hg": "1", "ag": "1", "match_url": "https://www.flashscore.com/match/b/"},
                {"date": "02.08.2026", "home": "Sevilla", "away": "Granada",
                 "hg": "3", "ag": "0", "match_url": "https://www.flashscore.com/match/c/"},
            ]),
            fetch_match_statistics=AsyncMock(side_effect=[
                {"xg_home": 1.8, "xg_away": 1.1},  # a: Betis home
                {"xg_home": 1.2, "xg_away": 0.9},  # b: Sevilla home
                {"xg_home": 2.1, "xg_away": 0.4},  # c: Sevilla home
            ]),
        )
        with patch.object(fs_mod, "_suggest_team", return_value=("sevilla", "99")):
            result = await fetcher.fetch_team_xg_history("Sevilla", meta)
        assert result is not None
        assert result["source"] == "flashscore_xg"
        assert result["sample_size"] == 3
        # Sevilla perspective: a: away 1.1, b: home 1.2, c: home 2.1
        assert abs(result["xg_for_avg"] - (1.1 + 1.2 + 2.1) / 3) < 1e-6
        assert abs(result["xg_against_avg"] - (1.8 + 0.9 + 0.4) / 3) < 1e-6
    asyncio.run(runner())


def test_fetch_team_xg_history_flashscore_excludes_predicted_fixture():
    """xG fallback: a just-finished same-day match (the predicted fixture)
    is excluded by date so it can never leak its own stats."""
    async def runner():
        from agents.football import flashscore as fs_mod

        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_league_key"] = "UEL"
        fetcher.fc = SimpleNamespace(
            available=True,
            fetch_team_results=AsyncMock(return_value=[
                {"date": "17.08.2026", "home": "Sevilla", "away": "Betis",
                 "hg": "1", "ag": "2", "match_url": "https://www.flashscore.com/match/x/"},
                {"date": "09.08.2026", "home": "Sevilla", "away": "Cadiz",
                 "hg": "1", "ag": "1", "match_url": "https://www.flashscore.com/match/y/"},
            ]),
            fetch_match_statistics=AsyncMock(side_effect=[
                {"xg_home": 9.9, "xg_away": 9.9},  # excluded by date
                {"xg_home": 1.2, "xg_away": 0.9},
            ]),
        )
        with patch.object(fs_mod, "_suggest_team", return_value=("sevilla", "99")):
            result = await fetcher.fetch_team_xg_history(
                "Sevilla", meta, exclude=("Sevilla", "Betis", "2026-08-17")
            )
        assert result is not None
        assert result["sample_size"] == 1
        assert abs(result["xg_for_avg"] - 1.2) < 1e-6
        assert abs(result["xg_against_avg"] - 0.9) < 1e-6
    asyncio.run(runner())


def test_fetch_team_xg_history_nowgoal_caches_per_match_id():
    """NowGoal tier: fetch_match_xg results are cached 24h per match_id, so
    a second analysis of the same team (or the opponent reusing the same
    match) does not re-fetch the detail page."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        store: dict = {}
        fetcher.cache = SimpleNamespace(
            get=lambda k, ttl_seconds=None: store.get(k),
            set=lambda k, v: store.__setitem__(k, v),
        )
        ng = SimpleNamespace(
            fetch_match_xg=AsyncMock(return_value={"xg_home": 1.5, "xg_away": 0.7}),
        )
        meta = _meta()
        ml = [
            {"match_id": "501", "date": "2026-08-10", "home": "Sevilla", "away": "Cadiz"},
        ]
        # first call fetches; second call hits the cache (mock call count stays 1)
        r1 = await fetcher.fetch_team_xg_history("Sevilla", meta, nowgoal_client=ng, match_list=ml)
        r2 = await fetcher.fetch_team_xg_history("Sevilla", meta, nowgoal_client=ng, match_list=ml)
        assert r1 is not None and r2 is not None
        assert ng.fetch_match_xg.await_count == 1
        assert "ng_xg_501" in store
        assert abs(r1["xg_for_avg"] - 1.5) < 1e-6
        assert abs(r2["xg_for_avg"] - 1.5) < 1e-6
    asyncio.run(runner())


def test_fetch_team_xg_history_nowgoal_no_match_list_falls_to_flashscore():
    """NowGoal tier needs the analysis match_list; when it is absent (None),
    the wrapper degrades to the flashscore fallback instead of returning
    None -- same behaviour as before the NowGoal tier existed."""
    async def runner():
        from agents.football import flashscore as fs_mod

        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_league_key"] = "UEL"
        ng = SimpleNamespace(fetch_match_xg=AsyncMock(return_value=None))
        fetcher.fc = SimpleNamespace(
            available=True,
            fetch_team_results=AsyncMock(return_value=[
                {"date": "09.08.2026", "home": "Sevilla", "away": "Cadiz",
                 "hg": "1", "ag": "1", "match_url": "https://www.flashscore.com/match/y/"},
            ]),
            fetch_match_statistics=AsyncMock(return_value={"xg_home": 1.2, "xg_away": 0.9}),
        )
        with patch.object(fs_mod, "_suggest_team", return_value=("sevilla", "99")):
            result = await fetcher.fetch_team_xg_history(
                "Sevilla", meta, nowgoal_client=ng, match_list=None
            )
        assert result is not None
        assert result["source"] == "flashscore_xg"
        assert result["xg_source"] == "flashscore_xg"
        assert abs(result["xg_for_avg"] - 1.2) < 1e-6
    asyncio.run(runner())


def test_fetch_h2h_by_name_flashscore_when_fd_misses():
    """H2H wiring fix: on the football-data path (no ``_flashscore_match``)
    the pair is resolved by name and the flashscore H2H tab is rendered, so
    H2H no longer dies whenever flashscore did not resolve the pair first."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_league_key"] = "UEL"
        meta["_team_names"] = {"42": "Real Betis", "99": "Sevilla"}
        fetcher.fd.fetch_h2h = AsyncMock(return_value=None)
        fetcher.fc = SimpleNamespace(
            available=True,
            resolve_match=AsyncMock(return_value={
                "home": {"id": "x1", "name": "Real Betis"},
                "away": {"id": "x2", "name": "Sevilla"},
                "match_url": "https://www.flashscore.com/match/abc/",
            }),
            fetch_match_h2h=AsyncMock(return_value={
                "wins": 3, "draws": 1, "losses": 2, "count": 6,
            }),
        )
        result = await fetcher.fetch_h2h(42, 99, meta)
        assert result is not None
        assert result["source"] == "flashscore_h2h"
        assert result["wins"] == 3
        fetcher.fc.resolve_match.assert_awaited_once_with("UEL", "Real Betis", "Sevilla")
        fetcher.fc.fetch_match_h2h.assert_awaited_once_with(
            "https://www.flashscore.com/match/abc/", "Real Betis", "Sevilla"
        )
    asyncio.run(runner())


def test_fetch_h2h_skips_flashscore_render_when_fd_succeeds():
    """H2H wiring fix: a cheap football-data H2H wins immediately; the
    expensive flashscore by-name browser render is never attempted."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_team_names"] = {"42": "Real Betis", "99": "Sevilla"}
        fetcher.fd.fetch_h2h = AsyncMock(return_value=[
            {"winner": {"id": 42}},
            {"winner": {"id": 42}},
        ])
        fetcher.fc = SimpleNamespace(
            available=True,
            resolve_match=AsyncMock(side_effect=AssertionError("should not render")),
            fetch_match_h2h=AsyncMock(side_effect=AssertionError("should not render")),
        )
        result = await fetcher.fetch_h2h(42, 99, meta)
        assert result is not None
        assert result["source"] == "football_data"
        assert result["wins"] == 2
    asyncio.run(runner())


def test_fetch_team_form_name_fallback_football_data_for_string_id():
    """oddspapi STRING ids resolve by NAME -> football-data form."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        # Flashscore OFF: this test exercises the NAME-fallback chain only
        # (flashscore by-name would now resolve "Heart of Midlothian FC" via
        # the livesport search API and hit the real browser -- not what this
        # test asserts).
        fetcher.fc = None
        meta = _meta()
        meta["_league_key"] = "UEL"
        meta["_team_names"] = {"93811": "Heart of Midlothian FC"}
        fetcher.search_team = AsyncMock(return_value={
            "id": 42, "name": "Heart of Midlothian FC", "provider": "football_data",
        })
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=[
            {"homeTeam": {"id": 42}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 2, "away": 1}}},
        ])
        # F1: the thin 1-match window no longer short-circuits the chain, so
        # thesportsdb (and livescore) must be mocked to keep this test
        # deterministic instead of hitting the network.
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        result = await fetcher.fetch_team_form("93811", meta, limit=5)
        assert result is not None
        assert result["sequence"] == "W"
        assert result["source"] == "football_data"
        fetcher.search_team.assert_awaited_once_with("Heart of Midlothian FC", meta)
    asyncio.run(runner())


def test_fetch_team_form_name_fallback_thesportsdb():
    """string id with no football-data hit falls back to thesportsdb form."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None  # flashscore OFF: exercise the name-fallback chain
        meta = _meta()
        meta["_league_key"] = "UEL"
        meta["_team_names"] = {"77881": "SL Benfica"}
        fetcher.search_team = AsyncMock(return_value={
            "id": "77", "name": "SL Benfica", "provider": "thesportsdb",
        })
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=[
            {"idHomeTeam": "77", "idAwayTeam": "88",
             "intHomeScore": "2", "intAwayScore": "1"},
        ])
        result = await fetcher.fetch_team_form("77881", meta, limit=5)
        assert result is not None
        assert result["sequence"] == "W"
        assert result["source"] == "thesportsdb"
    asyncio.run(runner())


def test_fetch_team_form_name_fallback_none_when_no_team_resolves():
    """string id with no resolvable team degrades to None (no crash)."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None  # flashscore OFF: exercise the name-fallback chain
        meta = _meta()
        meta["_league_key"] = "UEL"
        meta["_team_names"] = {"93811": "Unknown FC"}
        fetcher.search_team = AsyncMock(return_value=None)
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        result = await fetcher.fetch_team_form("93811", meta, limit=5)
        assert result is None
    asyncio.run(runner())


def test_fetch_team_form_int_id_resolves_thesportsdb_by_name():
    """F1: a football-data INT id is never passed to eventslast.php as if it
    were a thesportsdb id -- the team is resolved by name first, and the
    thesportsdb id is used only when the resolution is genuinely thesportsdb.
    Regression: _thesportsdb_form(str(team_id)) previously sent the fd int to
    thesportsdb, returning events for an unrelated club with a colliding id."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None  # flashscore OFF: exercise the F1 name-resolution
        meta = _meta()
        meta["_league_key"] = "UEL"  # not FBref-supported: keep offline
        meta["_team_names"] = {"42": "Real Betis"}
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=None)
        fetcher.search_team = AsyncMock(return_value={
            "id": "77", "name": "Real Betis", "provider": "thesportsdb",
        })
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=[
            {"idHomeTeam": "77", "idAwayTeam": "88",
             "intHomeScore": "3", "intAwayScore": "0"},
        ])
        result = await fetcher.fetch_team_form(42, meta, limit=5)
        assert result is not None
        assert result["source"] == "thesportsdb"
        assert result["sequence"] == "W"
        # thesportsdb must be called with the RESOLVED thesportsdb id (77),
        # never the football-data int (42)
        fetcher.ts.fetch_last_matches.assert_awaited_once_with("77", 5)
    asyncio.run(runner())


def test_fetch_team_form_int_id_without_name_skips_thesportsdb():
    """F1: without a resolvable team name there is no way to obtain a genuine
    thesportsdb id, so the final thesportsdb branch must SKIP instead of
    passing the foreign (football-data) id to eventslast.php. The mocked
    events below simulate exactly the collision the fix prevents: a thesportsdb
    team whose idTeam equals the football-data int."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None
        meta = _meta()
        meta["_league_key"] = "UEL"
        # no _team_names -> team_name is None
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=None)
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=[
            {"idHomeTeam": "42", "idAwayTeam": "88",
             "intHomeScore": "2", "intAwayScore": "0"},
        ])
        result = await fetcher.fetch_team_form(42, meta, limit=5)
        assert result is None
        fetcher.ts.fetch_last_matches.assert_not_awaited()
    asyncio.run(runner())


def test_fetch_team_form_flashscore_by_name():
    """Regular query path (no _flashscore_match) resolves the team slug via
    the pure-HTTP suggest endpoint and uses the full flashscore 5-match form
    instead of silently falling to football-data's thin 1-match window."""
    async def runner():
        from agents.football import flashscore as fs_mod

        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_team_names"] = {"670": "ADO Den Haag"}
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=None)
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        fetcher.sd.supports_league = Mock(return_value=False)
        with patch.object(fs_mod, "_suggest_team", return_value=("ado-den-haag", "12345")):
            fetcher.fc.fetch_team_form = AsyncMock(return_value={
                "sequence": "W-L-W-L-L", "gf_avg": 1.2, "ga_avg": 1.0,
                "sample_size": 5, "recent_goals": [(2, 0), (0, 1), (1, 0), (0, 2), (1, 1)],
            })
            result = await fetcher.fetch_team_form(670, meta, limit=5)
        assert result is not None
        assert result["source"] == "flashscore"
        assert result["sequence"] == "W-L-W-L-L"
        assert result["sample_size"] == 5
        fetcher.fc.fetch_team_form.assert_awaited_once_with("ado-den-haag", "12345", limit=5)
    asyncio.run(runner())


def test_fetch_team_form_flashscore_by_name_miss_falls_through():
    """When the by-name suggest misses, the chain continues (football-data)
    instead of crashing."""
    async def runner():
        from agents.football import flashscore as fs_mod

        fetcher = MultiSourceStatsFetcher("fd", "")
        meta = _meta()
        meta["_team_names"] = {"670": "ADO Den Haag"}
        fetcher.sd.supports_league = Mock(return_value=False)
        with patch.object(fs_mod, "_suggest_team", return_value=None):
            fetcher.fc.fetch_team_form = AsyncMock(return_value={"sequence": "W"})
            fetcher.fd.fetch_last_matches = AsyncMock(return_value=[
                {"homeTeam": {"id": 670}, "awayTeam": {"id": 99},
                 "score": {"fullTime": {"home": 2, "away": 1}}},
            ])
            result = await fetcher.fetch_team_form(670, meta, limit=5)
        assert result is not None
        assert result["source"] == "football_data"
        assert result["sequence"] == "W"
        fetcher.fc.fetch_team_form.assert_not_awaited()
    asyncio.run(runner())


def _ls_event(home: str, away: str, hg: str, ag: str) -> dict:
    return {
        "T1": [{"Nm": home, "ID": "1"}],
        "T2": [{"Nm": away, "ID": "2"}],
        "Tr1": hg, "Tr2": ag, "Eps": "FT",
    }


def _ls_payload(events: list[dict]) -> dict:
    return {"Stages": [{"CompN": "Eredivisie", "Cnm": "Netherlands", "Events": events}]}


def test_fetch_team_form_livescore_last_resort():
    """When flashscore / FBref / football-data / thesportsdb all fail to fill
    the form window, the team's finished matches are rebuilt from the
    LiveScore date feed (OLDEST -> NEWEST recent_goals, like every provider)."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None  # flashscore unavailable -> no by-name render
        fetcher.sd.supports_league = Mock(return_value=False)
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=None)
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        # G5: the feed's competition must match the league being analysed
        # (the old fixture used LaLiga metadata with an Eredivisie feed).
        meta = _meta(provider="football_data", fallback="thesportsdb")
        meta["_league_key"] = "Eredivisie"
        meta["display"] = "Eredivisie"
        meta["football_data_code"] = "DED"
        meta["_team_names"] = {"670": "ADO Den Haag"}

        today = datetime.now(timezone.utc)

        def d8(days_back: int) -> str:
            return (today - timedelta(days=days_back)).strftime("%Y%m%d")

        feeds = {
            (d8(0), 0): _ls_payload([_ls_event("ADO Den Haag", "FC Utrecht", "2", "0")]),
            (d8(1), 0): _ls_payload([_ls_event("PEC Zwolle", "ADO Den Haag", "1", "1")]),
            (d8(2), 0): _ls_payload([_ls_event("FC Volendam", "ADO Den Haag", "3", "0")]),
        }
        empty = _ls_payload([])

        async def fake_fetch(date8: str, page: int = 0):
            return feeds.get((date8, page), empty)

        fetcher.livescore = SimpleNamespace(
            available=True, fetch_soccer_date=AsyncMock(side_effect=fake_fetch)
        )
        result = await fetcher.fetch_team_form(670, meta, limit=5)
        assert result is not None
        assert result["source"] == "livescore"
        # newest-first scan reversed -> OLDEST -> NEWEST
        assert result["sequence"] == "L-D-W"
        assert result["recent_goals"] == [(0, 3), (1, 1), (2, 0)]
        assert result["sample_size"] == 3
        # ADO home W (2-0), away D (1-1), away L (0-3)
        assert result["home"] == {"w": 1, "d": 0, "l": 0}
        assert result["away"] == {"w": 0, "d": 1, "l": 1}
        fetcher.livescore.fetch_soccer_date.assert_awaited()
    asyncio.run(runner())


def test_fetch_team_form_thin_football_data_filled_by_livescore():
    """F1: a THIN form window (football-data 1 match) is no longer returned
    as final -- the chain continues and the fuller LiveScore window wins, so
    the statistical component is not silently zeroed (ADO-Den-Haag class)."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None
        fetcher.sd.supports_league = Mock(return_value=False)
        # football-data returns a THIN 1-match window (has sequence -> "exists")
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=[
            {"homeTeam": {"id": 670}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 1, "away": 2}}},
        ])
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        # G5: league metadata must match the Eredivisie feed below.
        meta = _meta()
        meta["_league_key"] = "Eredivisie"
        meta["display"] = "Eredivisie"
        meta["football_data_code"] = "DED"
        meta["_team_names"] = {"670": "ADO Den Haag"}

        today = datetime.now(timezone.utc)

        def d8(days_back: int) -> str:
            return (today - timedelta(days=days_back)).strftime("%Y%m%d")

        # LiveScore has the team's full last-5 window available.
        feeds = {}
        for i, res in enumerate([("2", "0"), ("1", "1"), ("3", "0"), ("0", "1"), ("2", "1")]):
            hg, ag = res
            feeds[(d8(i), 0)] = _ls_payload([_ls_event("ADO Den Haag", f"Opp {i}", hg, ag)])
        empty = _ls_payload([])

        async def fake_fetch(date8: str, page: int = 0):
            return feeds.get((date8, page), empty)

        fetcher.livescore = SimpleNamespace(
            available=True, fetch_soccer_date=AsyncMock(side_effect=fake_fetch)
        )
        result = await fetcher.fetch_team_form(670, meta, limit=5)
        assert result is not None
        # The fuller LiveScore window wins over football-data's 1-match thin form.
        assert result["source"] == "livescore"
        assert result["sample_size"] == 5
        # day4 W, day3 L, day2 W, day1 D, day0 W -- oldest -> newest
        assert result["sequence"] == "W-L-W-D-W"
    asyncio.run(runner())


def test_fetch_team_form_thin_form_returned_when_livescore_empty():
    """F1 fallback: when LiveScore has no data either, the thin provider form
    is returned (honest degradation), never None-with-data-discarded."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None
        fetcher.sd.supports_league = Mock(return_value=False)
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=[
            {"homeTeam": {"id": 670}, "awayTeam": {"id": 99},
             "score": {"fullTime": {"home": 1, "away": 2}}},
        ])
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        meta = _meta()
        meta["_team_names"] = {"670": "ADO Den Haag"}
        fetcher.livescore = SimpleNamespace(
            available=True, fetch_soccer_date=AsyncMock(return_value=_ls_payload([]))
        )
        result = await fetcher.fetch_team_form(670, meta, limit=5)
        assert result is not None
        assert result["source"] == "football_data"
        assert result["sample_size"] == 1
    asyncio.run(runner())


def test_fetch_team_form_no_livescore_client_returns_none():
    """Without a configured livescore client the chain ends at thesportsdb
    exactly as before -- no new behavior when livescore is disabled."""
    async def runner():
        fetcher = MultiSourceStatsFetcher("fd", "")
        fetcher.fc = None
        fetcher.sd.supports_league = Mock(return_value=False)
        fetcher.fd.fetch_last_matches = AsyncMock(return_value=None)
        fetcher.ts.fetch_last_matches = AsyncMock(return_value=None)
        meta = _meta()
        meta["_team_names"] = {"670": "ADO Den Haag"}
        result = await fetcher.fetch_team_form(670, meta, limit=5)
        assert result is None
        assert fetcher.livescore is None
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
