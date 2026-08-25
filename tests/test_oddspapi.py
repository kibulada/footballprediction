"""Tests for the OddsPapi secondary odds client.

Covers: tolerant fixture matching, /odds normalization into The Odds API
payload shape (h2h with real team names, totals with handicap, btts), the
inactive-player price fallback, suspended-bookmaker skip, and the
find_specific_match wiring (no key -> no fallback call).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.oddspapi import (  # noqa: E402
    OddspapiClient,
    _norm_team,
    _same_team,
)


def _odds_response(bookmaker_odds: dict) -> dict:
    return {
        "fixtureId": "id1000122271220452",
        "hasOdds": True,
        "bookmakerOdds": bookmaker_odds,
    }


def _bm(markets: dict) -> dict:
    return {"suspended": False, "markets": markets}


def _market(market_id: str, outcomes: dict) -> dict:
    return {
        "bookmakerMarketId": "1",
        "marketActive": True,
        "outcomes": {
            oid: {"players": {"0": {"active": True, "price": price}}}
            for oid, price in outcomes.items()
        },
    }


def test_find_fixture_matches_tolerant_names():
    fx = {
        "fixtureId": "id1",
        "hasOdds": True,
        "participant1Name": "FK Bodo/Glimt",
        "participant2Name": "Union Saint-Gilloise",
        "startTime": "2026-08-12T18:00:00Z",
    }

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=[fx])) as mock_get:
            found = await client.find_fixture("Bodo/Glimt", "Royale Union Saint-Gilloise", "2026-08-12T18:00:00Z")
        assert found is not None
        assert found["fixtureId"] == "id1"
        mock_get.assert_awaited_once()

    asyncio.run(runner())


def test_find_fixture_skips_no_odds():
    fx = {"fixtureId": "id1", "hasOdds": False, "participant1Name": "A", "participant2Name": "B"}

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=[fx])):
            assert await client.find_fixture("A", "B") is None

    asyncio.run(runner())


def test_find_fixture_none_on_network_failure():
    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=None)):
            assert await client.find_fixture("A", "B") is None

    asyncio.run(runner())


def test_fetch_odds_normalizes_h2h_totals_btts():
    h2h = _market("101", {"101": 2.1, "102": 3.4, "103": 3.6})
    totals = _market("1010", {"1010": 1.8, "1011": 1.9})
    btts = _market("104", {"104": 1.7, "105": 2.1})

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(
            client, "_get",
            AsyncMock(return_value=_odds_response({"BetX": _bm({"101": h2h, "1010": totals, "104": btts})})),
        ):
            payload = await client.fetch_odds({
                "fixtureId": "id1",
                "participant1Name": "Lyon",
                "participant2Name": "Sparta Prague",
                "startTime": "2026-08-12T18:00:00Z",
            })
        assert payload is not None
        assert payload["home_team"] == "Lyon"
        assert payload["away_team"] == "Sparta Prague"
        assert payload["bookmakers"][0]["title"] == "BetX"
        markets = {m["key"]: m for m in payload["bookmakers"][0]["markets"]}
        # h2h outcomes use REAL team names so extract_h2h_entries matches them
        names = [o["name"] for o in markets["h2h"]["outcomes"]]
        assert names == ["Lyon", "Draw", "Sparta Prague"]
        prices = [o["price"] for o in markets["h2h"]["outcomes"]]
        assert prices == [2.1, 3.4, 3.6]
        # totals carry the handicap as point
        tot = markets["totals"]
        assert [o["name"] for o in tot["outcomes"]] == ["Over", "Under"]
        assert tot["outcomes"][0]["point"] == 2.5
        # btts
        assert [o["name"] for o in markets["btts"]["outcomes"]] == ["Yes", "No"]

    asyncio.run(runner())


def test_fetch_odds_normalizes_asian_handicap_fulltime():
    # 1066 = fulltime Asian Handicap line -0.75; 10604 = FIRST-HALF AH
    # (period p1) which must NOT be surfaced as a match-level AH market.
    ah_ft = _market("1066", {"1066": 1.95, "1067": 2.02})
    ah_p1 = _market("10604", {"10604": 2.4, "10605": 1.6})

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(
            client, "_get",
            AsyncMock(return_value=_odds_response({"BetX": _bm({"1066": ah_ft, "10604": ah_p1})})),
        ):
            payload = await client.fetch_odds({
                "fixtureId": "id1",
                "participant1Name": "Lyon",
                "participant2Name": "Sparta Prague",
                "startTime": "2026-08-12T18:00:00Z",
            })
        assert payload is not None
        ah_markets = [m for m in payload["bookmakers"][0]["markets"] if m["key"] == "asian_handicap"]
        assert len(ah_markets) == 1
        outcomes = ah_markets[0]["outcomes"]
        # outcomeId order: id+0 -> Home, id+1 -> Away; point home-relative
        assert [o["name"] for o in outcomes] == ["Home", "Away"]
        assert [o["price"] for o in outcomes] == [1.95, 2.02]
        assert all(o["point"] == -0.75 for o in outcomes)
        # end-to-end: the pipeline's AH extractor consumes the row as-is
        from agents.football.signal_engine import extract_asian_handicap

        rows = extract_asian_handicap(payload)
        assert len(rows) == 1
        assert rows[0]["line"] == -0.75
        assert rows[0]["home"] == 1.95 and rows[0]["away"] == 2.02

    asyncio.run(runner())


def test_fetch_odds_uses_inactive_player_price_fallback():
    market = {
        "bookmakerMarketId": "1",
        "marketActive": True,
        "outcomes": {
            "101": {"players": {"0": {"active": False, "price": 2.2}}},
            "102": {"players": {"0": {"active": False, "price": 3.1}}},
            "103": {"players": {"0": {"active": False, "price": 3.4}}},
        },
    }

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(
            client, "_get", AsyncMock(return_value=_odds_response({"BetX": _bm({"101": market})}))
        ):
            payload = await client.fetch_odds({
                "fixtureId": "id1", "participant1Name": "A", "participant2Name": "B",
                "startTime": "2026-08-12T18:00:00Z",
            })
        assert payload is not None
        h2h = [m for m in payload["bookmakers"][0]["markets"] if m["key"] == "h2h"][0]
        assert [o["price"] for o in h2h["outcomes"]] == [2.2, 3.1, 3.4]

    asyncio.run(runner())


def test_fetch_odds_skips_suspended_bookmaker():
    bm = _bm({"101": _market("101", {"101": 2.0, "102": 3.2, "103": 3.5})})
    bm["suspended"] = True

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(
            client, "_get", AsyncMock(return_value=_odds_response({"BetX": bm}))
        ):
            payload = await client.fetch_odds({
                "fixtureId": "id1", "participant1Name": "A", "participant2Name": "B",
                "startTime": "2026-08-12T18:00:00Z",
            })
        assert payload is None

    asyncio.run(runner())


def test_fetch_odds_skips_unknown_market_ids():
    bm = _bm({"9999": _market("9999", {"1": 1.5, "2": 2.5})})

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(
            client, "_get", AsyncMock(return_value=_odds_response({"BetX": bm}))
        ):
            payload = await client.fetch_odds({
                "fixtureId": "id1", "participant1Name": "A", "participant2Name": "B",
                "startTime": "2026-08-12T18:00:00Z",
            })
        assert payload is None

    asyncio.run(runner())


def test_fetch_odds_none_on_http_error():
    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=None)):
            assert await client.fetch_odds({"fixtureId": "id1", "participant1Name": "A", "participant2Name": "B"}) is None

    asyncio.run(runner())


def test_match_odds_chains_find_and_fetch():
    fx = {
        "fixtureId": "id1", "hasOdds": True,
        "participant1Name": "Lyon", "participant2Name": "Sparta Prague",
        "startTime": "2026-08-12T18:00:00Z",
    }

    async def runner():
        client = OddspapiClient("k", throttle_seconds=0.0)
        with patch.object(
            client, "_get",
            AsyncMock(side_effect=[
                [fx],
                _odds_response({"BetX": _bm({"101": _market("101", {"101": 2.0, "102": 3.2, "103": 3.5})})}),
            ]),
        ):
            payload = await client.match_odds("Lyon", "Sparta Prague", "2026-08-12T18:00:00Z")
        assert payload is not None
        assert payload["home_team"] == "Lyon"

    asyncio.run(runner())


# ---- team name normalization ---------------------------------------------

def test_norm_team_strips_accents_and_punct():
    assert _norm_team("Bod\u00f8/Glimt") == "bodo glimt"
    assert _norm_team("FK Kauno \u017dalgiris") == "fk kauno zalgiris"


def test_same_team_tolerant():
    assert _same_team("Bod\u00f8/Glimt", "Bodo/Glimt")
    assert _same_team("FK Bod\u00f8/Glimt", "Bod\u00f8/Glimt")
    assert _same_team("Royale Union Saint-Gilloise", "Union Saint-Gilloise")
    assert not _same_team("Barcelona", "Manchester City")


def test_same_team_single_letter_suffix_no_match():
    # Regression: the "b" of a "B team" suffix used to be accepted as a
    # substring of any longer name ("b" in "genclerbirligi"), so a query for
    # Gen\u00e7lerbirli\u011fi vs Fenerbah\u00e7e wrongly resolved to a friendly
    # "Cadiz B vs Real Betis B" and the match was reported as already finished.
    assert not _same_team("Cadiz B", "Gen\u00e7lerbirli\u011fi")
    assert not _same_team("Real Betis B", "Fenerbah\u00e7e")
    assert not _same_team("Cadiz B", "Fenerbah\u00e7e")
    assert not _same_team("Real Betis B", "Gen\u00e7lerbirli\u011fi")
    assert not _same_team("Cadiz B", "Real Betis")


def test_same_team_country_suffix_uecl_qualification():
    # flashscore homepage names ("Tobol (Kaz)") vs OddsPapi fixture names
    # ("Tobol Kostanay"): all 5 UECL qualification matches must resolve.
    assert _same_team("Tobol (Kaz)", "Tobol Kostanay")
    assert _same_team("Partizan (Srb)", "FK Partizan Belgrade")
    assert _same_team("Flora (Est)", "Tallinna FC Flora")
    assert _same_team("Inter Escaldes (And)", "Inter Club de Escaldes")
    assert _same_team("Ilves (Fin)", "Tampereen Ilves")
    assert _same_team("Rijeka (Cro)", "HNK Rijeka")
    assert _same_team("Qarabag (Aze)", "Qarabag FK")
    assert _same_team("Dyn. Kyiv (Ukr)", "FC Dynamo Kyiv")
    assert _same_team("RFS (Lat)", "FC RFS")
    assert _same_team("Jablonec (Cze)", "FK Jablonec")
    assert not _same_team("Tobol (Kaz)", "Astana")


# ---- find_specific_match wiring -----------------------------------------

def _run_analyse(analyse, ms, oddspapi, nowgoal=None):
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
        nowgoal=nowgoal,
    ))


def _make_stats():
    import agents.football.analyse as analyse

    async def fake_search(*a, **k):
        return {"id": 1, "name": "Arsenal", "provider": "flashscore"}, {"id": 2, "name": "Chelsea", "provider": "flashscore"}

    with patch.object(analyse, "MultiSourceStatsFetcher") as mss:
        ms = mss.return_value
        ms.search_teams_pair = fake_search
        ms.fetch_upcoming_fixture = AsyncMock(return_value={})
        ms.fetch_team_form = AsyncMock(return_value={})
        ms.fetch_h2h = AsyncMock(return_value={})
        ms.fetch_team_xg_history = AsyncMock(return_value=None)
        ms.fetch_flashscore_stats_for_match = AsyncMock(return_value=None)
        ms.fetch_flashscore_lineups_for_match = AsyncMock(return_value=None)
        ms.fd = type("FD", (), {"rate_limit_warning": False})()
        return ms


def test_analyse_without_oddspapi_no_fallback_call():
    """find_specific_match must not call oddspapi when not configured."""
    import agents.football.analyse as analyse

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, oddspapi=None)
    assert result is not None
    assert result.get("quota", {}).get("oddspapi_used") is False
    assert "oddspapi_odds" not in (result.get("sources") or [])


def test_analyse_oddspapi_fallback_fires_when_primary_empty():
    """The fallback must be tried when The Odds API has no payload."""
    import agents.football.analyse as analyse

    fake_odds_payload = {
        "home_team": "Arsenal", "away_team": "Chelsea", "commence_time": "2026-08-12T18:00:00Z",
        "bookmakers": [{"title": "BetX", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.0}, {"name": "Draw", "price": 3.2},
            {"name": "Chelsea", "price": 3.5},
        ]}]}],
    }
    fake_oddspapi = AsyncMock()
    fake_oddspapi.find_fixture = AsyncMock(return_value={
        "fixtureId": "fx1", "hasOdds": True,
        "participant1Id": 111, "participant1Name": "Arsenal",
        "participant2Id": 222, "participant2Name": "Chelsea",
        "startTime": "2026-08-12T18:00:00Z",
    })
    fake_oddspapi.fetch_odds = AsyncMock(return_value=fake_odds_payload)

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        resolve_calls: list[int] = []
        original_resolve = ms.search_teams_pair

        async def counting_resolve(*a, **k):
            resolve_calls.append(1)
            return await original_resolve(*a, **k)

        ms.search_teams_pair = counting_resolve
        result = _run_analyse(analyse, ms, oddspapi=fake_oddspapi)

    fake_oddspapi.fetch_odds.assert_awaited()
    assert result.get("quota", {}).get("oddspapi_used") is True
    assert "oddspapi_odds" in (result.get("sources") or [])
    assert (result.get("odds") or {}).get("has_odds") is True
    assert (result.get("odds") or {}).get("bookmakers_count") == 1
    # flashscore tetap jalur utama: meski odds dari oddspapi, resolve pair
    # via search_teams_pair tetap dipanggil (bukan di-skip)
    assert resolve_calls, "search_teams_pair harus tetap dipanggil saat odds via oddspapi"


def test_analyse_oddspapi_teams_only_fallback_when_providers_fail():
    """Nama tim oddspapi dipakai HANYA kalau search_teams_pair gagal total."""
    import agents.football.analyse as analyse

    fake_odds_payload = {
        "home_team": "Arsenal", "away_team": "Chelsea", "commence_time": "2026-08-12T18:00:00Z",
        "bookmakers": [{"title": "BetX", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.0}, {"name": "Draw", "price": 3.2},
            {"name": "Chelsea", "price": 3.5},
        ]}]}],
    }
    fake_oddspapi = AsyncMock()
    fake_oddspapi.find_fixture = AsyncMock(return_value={
        "fixtureId": "fx1", "hasOdds": True,
        "participant1Id": 111, "participant1Name": "Arsenal",
        "participant2Id": 222, "participant2Name": "Chelsea",
        "startTime": "2026-08-12T18:00:00Z",
    })
    fake_oddspapi.fetch_odds = AsyncMock(return_value=fake_odds_payload)

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        ms.search_teams_pair = AsyncMock(return_value=(None, None))
        result = _run_analyse(analyse, ms, oddspapi=fake_oddspapi)

    assert result is not None
    assert result.get("home") == "Arsenal"
    assert result.get("away") == "Chelsea"
    assert result.get("quota", {}).get("oddspapi_used") is True
    assert (result.get("odds") or {}).get("has_odds") is True


def test_analyse_oddspapi_primary_wins_over_the_odds_api():
    """Priority (2026-08): oddspapi is PRIMARY. When oddspapi has odds, The
    Odds API (find_match_odds_payload) must NOT be called at all."""
    import agents.football.analyse as analyse

    odsp_payload = {
        "home_team": "Arsenal", "away_team": "Chelsea", "commence_time": "2026-08-12T18:00:00Z",
        "bookmakers": [{"title": "BetX", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 2.0}, {"name": "Draw", "price": 3.2},
            {"name": "Chelsea", "price": 3.5},
        ]}]}],
    }
    fake_oddspapi = AsyncMock()
    fake_oddspapi.find_fixture = AsyncMock(return_value={
        "fixtureId": "fx1", "hasOdds": True,
        "participant1Id": 111, "participant1Name": "Arsenal",
        "participant2Id": 222, "participant2Name": "Chelsea",
        "startTime": "2026-08-12T18:00:00Z",
    })
    fake_oddspapi.fetch_odds = AsyncMock(return_value=odsp_payload)

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(odsp_payload, "soccer_epl"))) as find_odds, \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, oddspapi=fake_oddspapi)

    fake_oddspapi.fetch_odds.assert_awaited()
    # The Odds API is the LAST resort -- oddspapi already supplied odds
    find_odds.assert_not_awaited()
    assert result.get("quota", {}).get("oddspapi_used") is True
    assert "oddspapi_odds" in (result.get("sources") or [])
    assert (result.get("odds") or {}).get("bookmakers_count") == 1


def test_analyse_the_odds_api_last_when_oddspapi_and_nowgoal_empty():
    """Both oddspapi and nowgoal empty -> The Odds API runs as LAST resort."""
    import agents.football.analyse as analyse

    odds_api_payload = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "bookmakers": [{"title": "Bet365", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 1.9}, {"name": "Draw", "price": 3.3},
            {"name": "Chelsea", "price": 4.0},
        ]}]}],
    }
    fake_oddspapi = AsyncMock()
    fake_oddspapi.find_fixture = AsyncMock(return_value=None)  # no fixture
    fake_nowgoal = AsyncMock()
    fake_nowgoal.match_odds = AsyncMock(return_value=None)  # no odds

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(odds_api_payload, "soccer_epl"))) as find_odds, \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, oddspapi=fake_oddspapi, nowgoal=fake_nowgoal)

    fake_oddspapi.find_fixture.assert_awaited()
    fake_nowgoal.match_odds.assert_awaited()
    find_odds.assert_awaited()
    assert result.get("quota", {}).get("oddspapi_used") is False
    assert result.get("quota", {}).get("nowgoal_used") is False
    assert (result.get("odds") or {}).get("bookmakers_count") == 1


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
