"""Tests for the analyse finished-match path and the parallel odds/team run.

Finished match (kickoff < now): the analysis shows the real result (score +
post-match stats) and SKIPS prediction engine / decision engine / prediction
log / market tiers. The parallel gather must start both the odds lookup and
the flashscore team resolve together (the team resolve is never starved by a
slow odds fetch), and must not break the oddspapi wiring.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agents.football.analyse as analyse  # noqa: E402


def _make_stats(fixture=None):
    ms = AsyncMock()
    ms.search_teams_pair = AsyncMock(
        return_value=(
            {"id": 1, "name": "Arsenal", "provider": "flashscore"},
            {"id": 2, "name": "Chelsea", "provider": "flashscore"},
        )
    )
    ms.fetch_upcoming_fixture = AsyncMock(return_value=fixture if fixture is not None else {})
    ms.fetch_team_form = AsyncMock(
        return_value={"sequence": "W-W-D", "gf_avg": 1.5, "ga_avg": 0.8}
    )
    ms.fetch_h2h = AsyncMock(return_value={"wins": 1, "draws": 0, "losses": 2})
    ms.fetch_team_xg_history = AsyncMock(return_value=None)
    ms.fetch_flashscore_stats_for_match = AsyncMock(
        return_value={"xg_home": 1.2, "xg_away": 2.1, "possession_home": 41.0,
                      "possession_away": 59.0, "source": "flashscore"}
    )
    ms.fetch_flashscore_lineups_for_match = AsyncMock(return_value=None)
    ms.fetch_flashscore_event_context = AsyncMock(return_value=None)
    ms.fetch_league_standings = AsyncMock(return_value=None)
    ms.fetch_flashscore_match_info = AsyncMock(return_value=None)
    ms.fd = type("FD", (), {"rate_limit_warning": False})()
    return ms


def _run(ms, oddspapi=None, **kwargs):
    return asyncio.run(analyse.find_specific_match(
        league_query="EPL", home_query="Arsenal", away_query="Chelsea",
        cfg={
            "cache_ttl_seconds": {"odds": 900},
            "outlier_threshold_pct": 5,
            "prediction_log": {"enabled": False},
        },
        odds=AsyncMock(),
        stats=ms,
        cache=type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})(),
        oddspapi=oddspapi,
        **kwargs,
    ))


def test_finished_match_shows_result_and_skips_engine():
    fixture = {
        "date": "2020-01-01T20:00:00Z",  # long past -> finished
        "status": "notstarted",
        "source": "flashscore",
        "flashscore_url": "https://www.flashscore.com/match/football/a-AAAA1111/b-BBBB2222/?mid=123",
        "score": {"home": "1", "away": "2"},
    }
    ms = _make_stats(fixture)
    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result["match_finished"] is True
    assert result["match_result"] == {"home": "1", "away": "2"}
    # real post-match stats fetched for the result display
    ms.fetch_flashscore_stats_for_match.assert_awaited()
    assert (result.get("event_stats") or {}).get("xg_home") == 1.2
    # skipped: prediction engine, decision engine, market tiers, log
    assert result.get("prediction") is None
    assert result.get("decision") is None
    assert result.get("confidence") is None
    assert result.get("picks") == {"top_picks": [], "best_pick": None, "model_probs": {}}
    # pre-match-only context is not fetched for a finished match
    ms.fetch_flashscore_lineups_for_match.assert_not_awaited()
    ms.fetch_flashscore_event_context.assert_not_awaited()
    ms.fetch_flashscore_match_info.assert_not_awaited()
    assert result.get("quota", {}).get("oddspapi_used") is False


def test_upcoming_match_still_predicts():
    fixture = {
        "date": "2030-01-01T20:00:00Z",  # far future -> not finished
        "status": "notstarted",
        "source": "flashscore",
        "flashscore_url": "https://www.flashscore.com/match/football/a-AAAA1111/b-BBBB2222/?mid=123",
    }
    ms = _make_stats(fixture)
    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result["match_finished"] is False
    assert result["prediction"] is not None
    assert result["decision"] is not None
    assert result.get("match_result") is None


def test_league_mismatch_flags_and_pins_standings_key():
    """Fix 2026-08-17: when the resolved fixture's REAL competition differs
    from the league the user typed ("laliga" query -> fixture actually in
    LaLiga2), the result carries league_mismatch and the standings fetch is
    pinned to the CORRECT key -- the table rendered belongs to the match,
    not to the query."""
    fixture = {
        "date": "2030-01-01T20:00:00Z",
        "status": "notstarted",
        "source": "flashscore",
        "flashscore_url": "https://www.flashscore.com/match/football/a-AAAA1111/b-BBBB2222/?mid=123",
    }
    ms = _make_stats(fixture)

    async def search_with_fs_match(home, away, meta):
        # the flashscore resolve lands via a competition-aware scrape and
        # reports the REAL section title (LaLiga2), not the user's "la liga"
        meta["_flashscore_match"] = {"competition": "LaLiga2"}
        return (
            {"id": 1, "name": "Las Palmas", "provider": "flashscore"},
            {"id": 2, "name": "Albacete", "provider": "flashscore"},
        )

    ms.search_teams_pair = search_with_fs_match
    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("LaLiga", {"display": "La Liga", "odds_api_key": "soccer_spain"})), \
         patch.object(analyse, "competition_league_key",
                      side_effect=lambda c: "Segunda" if str(c).lower() == "laliga2" else None), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result["league_mismatch"] == {
        "requested": "LaLiga",
        "actual": "Segunda",
        "competition": "LaLiga2",
    }
    # standings fetch used the CORRECT key, never the query league
    ms.fetch_league_standings.assert_awaited_with("Segunda")


def test_league_no_mismatch_when_competition_matches():
    """Fix 2026-08-17: when the fixture's competition maps to the SAME league
    the user typed, no league_mismatch flag and standings use the query key."""
    fixture = {
        "date": "2030-01-01T20:00:00Z",
        "status": "notstarted",
        "source": "flashscore",
        "flashscore_url": "https://www.flashscore.com/match/football/a-AAAA1111/b-BBBB2222/?mid=123",
    }
    ms = _make_stats(fixture)

    async def search_with_fs_match(home, away, meta):
        meta["_flashscore_match"] = {"competition": "LaLiga"}
        return (
            {"id": 1, "name": "Barcelona", "provider": "flashscore"},
            {"id": 2, "name": "Real Madrid", "provider": "flashscore"},
        )

    ms.search_teams_pair = search_with_fs_match
    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("LaLiga", {"display": "La Liga", "odds_api_key": "soccer_spain"})), \
         patch.object(analyse, "competition_league_key",
                      side_effect=lambda c: "LaLiga" if str(c).lower() == "laliga" else None), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result.get("league_mismatch") is None
    ms.fetch_league_standings.assert_awaited_with("LaLiga")


def test_parallel_gather_starts_odds_and_team_resolve_together():
    """The flashscore team resolve must start while the odds fetch is still
    running (previously the serial odds-first order could starve it)."""
    events: list[str] = []

    async def slow_odds(*a, **k):
        events.append("odds:start")
        await asyncio.sleep(0.05)
        events.append("odds:done")
        return None, None

    async def tracking_search(*a, **k):
        events.append("teams:start")
        return (
            {"id": 1, "name": "Arsenal", "provider": "flashscore"},
            {"id": 2, "name": "Chelsea", "provider": "flashscore"},
        )

    ms = _make_stats({})
    ms.search_teams_pair = tracking_search
    with patch.object(analyse, "find_match_odds_payload", slow_odds), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result is not None
    assert "odds:start" in events and "teams:start" in events
    # teams resolved before the odds fetch finished -> genuinely parallel
    assert events.index("teams:start") < events.index("odds:done")


def test_parallel_exception_in_team_resolve_does_not_crash():
    async def broken_search(*a, **k):
        raise RuntimeError("boom")

    ms = _make_stats({})
    ms.search_teams_pair = broken_search
    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result is not None
    assert "tim tidak ditemukan" in (result.get("error") or "")


def test_parallel_exception_in_odds_does_not_crash():
    async def broken_odds(*a, **k):
        raise RuntimeError("odds down")

    ms = _make_stats({})
    with patch.object(analyse, "find_match_odds_payload", broken_odds), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = _run(ms, oddspapi=None)

    assert result is not None
    assert result.get("home") == "Arsenal"
    assert result.get("odds", {}).get("has_odds") is False


def test_oddspapi_fallback_fills_missing_side_only():
    """P1.3: when the provider chain resolves only ONE side, the oddspapi
    fallback fills the missing side and keeps the resolved one intact (no
    full overwrite), and the merged provenance appears in sources."""
    ms = _make_stats({})
    # flashscore + provider chain resolve only HOME; away stays None
    ms.search_teams_pair = AsyncMock(
        return_value=(
            {"id": 1, "name": "Arsenal", "provider": "flashscore"},
            None,
        )
    )
    oddspapi = AsyncMock()
    oddspapi.find_fixture = AsyncMock(
        return_value={
            "hasOdds": True,
            "participant1Id": 9001,
            "participant1Name": "Arsenal",
            "participant2Id": 9002,
            "participant2Name": "Chelsea",
        }
    )
    oddspapi.fetch_odds = AsyncMock(return_value={"bookmakers": []})
    cache = type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})()

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = analyse.find_specific_match(
            league_query="EPL", home_query="Arsenal", away_query="Chelsea",
            cfg={"cache_ttl_seconds": {"odds": 900},
                 "outlier_threshold_pct": 5,
                 "prediction_log": {"enabled": False}},
            odds=AsyncMock(), stats=ms, cache=cache, oddspapi=oddspapi,
        )
        result = asyncio.run(result)

    assert result is not None
    # resolved home identity survives (flashscore id + provider) -- NOT
    # overwritten by the oddspapi participant
    assert result["home"] == "Arsenal"
    # away filled from oddspapi
    assert result["away"] == "Chelsea"
    assert "oddspapi_fallback_merged" in (result.get("sources") or [])
    assert "oddspapi_fallback_full" not in (result.get("sources") or [])


def test_oddspapi_fallback_full_when_nothing_resolved():
    """P1.3: when NOTHING was resolved by flashscore + the provider chain,
    the full oddspapi overwrite is allowed but must be auditable via
    identity_source oddspapi_fallback_full in sources."""
    ms = _make_stats({})
    ms.search_teams_pair = AsyncMock(return_value=(None, None))
    oddspapi = AsyncMock()
    oddspapi.find_fixture = AsyncMock(
        return_value={
            "hasOdds": True,
            "participant1Id": 9001,
            "participant1Name": "Arsenal",
            "participant2Id": 9002,
            "participant2Name": "Chelsea",
        }
    )
    oddspapi.fetch_odds = AsyncMock(return_value={"bookmakers": []})
    cache = type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})()

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored",
                      return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        result = analyse.find_specific_match(
            league_query="EPL", home_query="Arsenal", away_query="Chelsea",
            cfg={"cache_ttl_seconds": {"odds": 900},
                 "outlier_threshold_pct": 5,
                 "prediction_log": {"enabled": False}},
            odds=AsyncMock(), stats=ms, cache=cache, oddspapi=oddspapi,
        )
        result = asyncio.run(result)

    assert result is not None
    assert result["home"] == "Arsenal"
    assert result["away"] == "Chelsea"
    assert "oddspapi_fallback_full" in (result.get("sources") or [])


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
