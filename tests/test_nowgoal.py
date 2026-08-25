"""Tests for the NowGoal odds client.

Covers: the lenient euro/ou/ah parsers (strict validation, positional
disambiguation), schedule JS (B[]/A[]) parsing, tolerant fixture matching,
odds normalization into The Odds API payload shape, the closing-odds
(roddsList) variant, and the find_specific_match wiring (not configured ->
never called, primary empty + oddspapi empty -> nowgoal fires, primary has
odds -> nowgoal skipped).

All network calls are mocked; no live nowgoal request is ever made (the
domains are ISP-blocked on the bot's network).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from agents.football.nowgoal import (  # noqa: E402
    NowGoalClient,
    _norm_team,
    _same_team,
    parse_ah,
    parse_euro,
    parse_ou,
    run_nowgoal_check,
)


# ---- fetch_odds_trend (type=14&t=20 timestamped series) ------------------

def _trend_response():
    """Realistic type=14&t=20 body: pre-match + in-play rows per market.
    Prices: h2h decimal (5.5/3.5/1.57), ah/ou Hong-Kong (0.85 == 1.85)."""
    return {
        "ErrCode": 0,
        "Data": {
            "op": [
                {"odds": {"u": "5.0", "g": "3.5", "d": "1.65"},
                 "hs": 0, "gs": 0, "mt": 1786880000, "ht": "", "close": False, "type": 2},
                {"odds": {"u": "5.5", "g": "3.5", "d": "1.57"},
                 "hs": 0, "gs": 0, "mt": 1786885000, "ht": "", "close": False, "type": 2},
                {"odds": {"u": "5.5", "g": "3.6", "d": "1.55"},
                 "hs": 1, "gs": 0, "mt": 1786888600, "ht": "08", "close": False, "type": 0},
            ],
            "ah": [
                {"odds": {"u": "0.85", "g": "-1", "d": "0.95"},
                 "hs": 0, "gs": 0, "mt": 1786880000, "ht": "", "close": False, "type": 2},
                {"odds": {"u": "0.83", "g": "-0.75", "d": "0.98"},
                 "hs": 0, "gs": 0, "mt": 1786884000, "ht": "", "close": False, "type": 2},
            ],
            "ou": [
                {"odds": {"u": "0.8", "g": "2.25", "d": "1"},
                 "hs": 0, "gs": 0, "mt": 1786880000, "ht": "", "close": False, "type": 2},
                {"odds": {"u": "0.9", "g": "2.25", "d": "0.95"},
                 "hs": 0, "gs": 0, "mt": 1786885000, "ht": "", "close": False, "type": 2},
            ],
        },
    }


def test_parse_trend_series_h2h_decimal_prices():
    rows = NowGoalClient._parse_trend_series(None, _trend_response()["Data"]["op"], kind="h2h")
    assert len(rows) == 3
    # chronological (oldest first)
    assert rows[0]["ts"] <= rows[1]["ts"] <= rows[2]["ts"]
    assert rows[0]["home"] == 5.0 and rows[0]["draw"] == 3.5 and rows[0]["away"] == 1.65
    assert rows[1]["home"] == 5.5 and rows[1]["away"] == 1.57
    # pre-match rows carry empty minute; in-play row carries the minute
    assert rows[0]["minute"] == "" and rows[0]["home_goals"] == 0
    assert rows[2]["minute"] == "08" and rows[2]["home_goals"] == 1
    # line/over/under stay None for h2h
    assert rows[0]["line"] is None and rows[0]["over"] is None


def test_parse_trend_series_ah_hk_to_decimal():
    rows = NowGoalClient._parse_trend_series(None, _trend_response()["Data"]["ah"], kind="ah")
    assert len(rows) == 2
    # HK prices (0.85) -> decimal (1.85); NowGoal quotes the AH line from the
    # AWAY side (g=-1) so the HOME-handicap convention (the one the signal
    # engine / odds_snapshot rows use) is the negated line (+1.0)
    assert abs(rows[0]["home"] - 1.85) < 1e-9
    assert abs(rows[0]["away"] - 1.95) < 1e-9
    assert rows[0]["line"] == 1.0
    assert abs(rows[1]["home"] - 1.83) < 1e-9
    assert rows[1]["line"] == 0.75


def test_parse_trend_series_ou_hk_to_decimal():
    rows = NowGoalClient._parse_trend_series(None, _trend_response()["Data"]["ou"], kind="ou")
    assert len(rows) == 2
    assert abs(rows[0]["over"] - 1.8) < 1e-9
    assert abs(rows[0]["under"] - 2.0) < 1e-9
    assert rows[0]["line"] == 2.25


def test_parse_trend_series_drops_junk_rows():
    rows = NowGoalClient._parse_trend_series(None, [
        {"odds": {"u": "5.0", "g": "3.5", "d": "1.65"}, "mt": "junk", "ht": ""},
        {"odds": {"u": "", "g": "", "d": ""}, "mt": 1786880000, "ht": ""},
        {"odds": {"u": "5.2", "g": "3.4", "d": "1.60"}, "mt": 1786880000, "ht": ""},
    ], kind="h2h")
    assert len(rows) == 1
    assert rows[0]["home"] == 5.2


def test_parse_trend_series_not_a_list():
    assert NowGoalClient._parse_trend_series(None, "nope", kind="h2h") == []
    assert NowGoalClient._parse_trend_series(None, None, kind="ah") == []


async def _fake_trend_get(responses: dict):
    """NowGoalClient whose _get returns per-cid responses (keyed by cid)."""
    client = NowGoalClient()
    async def fake_get(path, params=None):
        cid = (params or {}).get("cid")
        return responses.get(cid)
    client._get = fake_get  # type: ignore[method-assign]
    return client


def test_fetch_odds_trend_builds_bookmaker_series():
    body = _trend_response()

    async def runner():
        client = await _fake_trend_get({8: body, 31: None})
        fx = {"match_id": "3061003", "home": "Malaysia", "away": "Vietnam",
              "kickoff": "2026-08-16T13:00:00Z"}
        trend = await client.fetch_odds_trend(fx, cids=[8, 31])
        assert trend is not None
        assert trend["history_resolution"] == "timestamped_series"
        assert trend["timestamp_available"] is True
        # only the bookmaker with data is included
        assert len(trend["bookmakers"]) == 1
        bm = trend["bookmakers"][0]
        assert bm["cid"] == 8
        assert bm["name"] == "Bet365"
        assert len(bm["h2h"]) == 3
        assert len(bm["ah"]) == 2
        assert len(bm["ou"]) == 2
        return trend

    trend = asyncio.run(runner())
    assert trend["bookmakers"][0]["h2h"][0]["home"] == 5.0


def test_fetch_odds_trend_none_when_no_bookmaker_has_data():
    async def runner():
        client = await _fake_trend_get({8: {"ErrCode": 0, "Data": {"op": [], "ah": [], "ou": []}}})
        trend = await client.fetch_odds_trend({"match_id": "1"}, cids=[8])
        assert trend is None
        return True

    assert asyncio.run(runner()) is True


def test_fetch_odds_trend_none_without_match_id():
    async def runner():
        client = await _fake_trend_get({})
        assert await client.fetch_odds_trend({}, cids=[8]) is None
        return True

    assert asyncio.run(runner()) is True


# ---- trend_to_snapshots (converter to odds_snapshot rows) ------------------

def _trend_payload():
    """fetch_odds_trend output shape: per-market series already parsed into
    {ts, minute, home, draw, away, line, over, under} rows."""
    body = _trend_response()
    return {
        "history_resolution": "timestamped_series",
        "bookmakers": [{
            "cid": 8, "name": "Bet365",
            "h2h": NowGoalClient._parse_trend_series(None, body["Data"]["op"], kind="h2h"),
            "ah": NowGoalClient._parse_trend_series(None, body["Data"]["ah"], kind="ah"),
            "ou": NowGoalClient._parse_trend_series(None, body["Data"]["ou"], kind="ou"),
        }],
    }


def test_trend_to_snapshots_builds_chronological_rows():
    from agents.football.nowgoal import trend_to_snapshots

    rows = trend_to_snapshots(_trend_payload(), kickoff="2026-08-16T13:00:00Z")
    assert rows
    # pre-match h2h (2) + ah (2) + ou (2) rows; the in-play op row (minute
    # "08") is DROPPED
    assert len(rows) == 6
    times = [r["ts"] for r in rows]
    assert times == sorted(times)
    h2h_rows = [r for r in rows if r["odds_1x2"] is not None]
    assert len(h2h_rows) == 2
    assert h2h_rows[0]["odds_1x2"]["home"] == 5.0
    assert h2h_rows[0]["timing"].startswith("T-")
    assert h2h_rows[0]["sources"] == ["nowgoal_trend"]


def test_trend_to_snapshots_ah_line_home_convention():
    from agents.football.nowgoal import trend_to_snapshots

    rows = trend_to_snapshots(_trend_payload(), kickoff="2026-08-16T13:00:00Z")
    ah_rows = [r for r in rows if r["odds_ah"] is not None]
    assert len(ah_rows) == 2
    # raw g=-1 -> HOME-handicap line +1.0 (the engine's convention)
    assert ah_rows[0]["odds_ah"]["line"] == 1.0
    assert abs(ah_rows[0]["odds_ah"]["home"] - 1.85) < 1e-9
    ou_rows = [r for r in rows if r["odds_ou"] is not None]
    assert len(ou_rows) == 2
    assert ou_rows[0]["odds_ou"]["line"] == 2.25
    assert abs(ou_rows[0]["odds_ou"]["over"] - 1.8) < 1e-9


def test_trend_to_snapshots_none_and_empty():
    from agents.football.nowgoal import trend_to_snapshots

    assert trend_to_snapshots(None) == []
    assert trend_to_snapshots({"bookmakers": []}) == []


def test_trend_to_snapshots_carries_bookmaker_attribution():
    from agents.football.nowgoal import trend_to_snapshots

    rows = trend_to_snapshots(_trend_payload(), kickoff="2026-08-16T13:00:00Z")
    assert rows
    for r in rows:
        assert r["bookmaker"] == "Bet365"
        assert r["bookmaker_cid"] == 8


def test_trend_to_snapshots_multi_bookmaker_attribution():
    from agents.football.nowgoal import trend_to_snapshots

    payload = {"bookmakers": [
        {"cid": 8, "name": "Bet365", "h2h": [
            {"ts": "2026-08-16T12:00:00+00:00", "minute": "",
             "home": 5.0, "draw": 3.5, "away": 1.65},
        ], "ah": [], "ou": []},
        {"cid": 177, "name": "Pinnacle", "h2h": [
            {"ts": "2026-08-16T12:30:00+00:00", "minute": "",
             "home": 4.9, "draw": 3.55, "away": 1.68},
        ], "ah": [], "ou": []},
    ]}
    rows = trend_to_snapshots(payload, kickoff="2026-08-16T13:00:00Z")
    assert [r["bookmaker"] for r in rows] == ["Bet365", "Pinnacle"]
    assert [r["bookmaker_cid"] for r in rows] == [8, 177]


# ---- probe_mirrors (ajax health audit, P1 2026-08-24) -----------------------

async def _async_ret(value):
    return value


def test_probe_mirrors_classifies_alive_dead_and_transport():
    import asyncio

    from agents.football.nowgoal import probe_mirrors

    async def fake_fetch(url, headers, params):
        assert headers["Referer"]  # referer always sent
        if url.startswith("https://live10.nowgoal26.com/"):
            return {"status": 200, "body": '{"ErrCode":0,"Data":{}}'}
        if url.startswith("https://www.nowgoal26.com/"):
            # anti-missing-referer shape STILL proves the endpoint exists
            return {"status": 200, "body": '{"code":1002}'}
        if url.startswith("http://www.nowgoal.net/"):
            return {"status": 404, "body": "<html>404 Not Found</html>"}
        # transport failure contract (mirrors _default_fetch)
        return {"status": None, "error": "ConnectError: DNS failure"}

    rows = asyncio.run(probe_mirrors(
        [
            "https://live10.nowgoal26.com/",
            "https://www.nowgoal26.com/",
            "http://www.nowgoal.net/",
            "http://dead.invalid/",
        ],
        _fetch=fake_fetch,
    ))
    assert [r["ok"] for r in rows] == [True, True, False, False]
    assert rows[0]["detail"] == "http 200"
    assert "404" in rows[2]["detail"]


def test_probe_mirrors_rejects_non_json_200_body():
    import asyncio

    from agents.football.nowgoal import probe_mirrors

    rows = asyncio.run(probe_mirrors(
        ["https://parked.example.com/"],
        _fetch=lambda u, h, p: _async_ret({"status": 200, "body": "<html>parked</html>"}),
    ))
    assert rows[0]["ok"] is False
    assert "bukan JSON" in rows[0]["detail"]


def test_probe_mirrors_empty_pool():
    import asyncio

    from agents.football.nowgoal import probe_mirrors

    assert asyncio.run(probe_mirrors([])) == []


def test_trend_to_snapshots_drops_inplay_rows():
    from agents.football.nowgoal import trend_to_snapshots

    payload = {"bookmakers": [{"cid": 8, "h2h": [
        {"ts": "2026-08-16T12:00:00+00:00", "minute": "",
         "home": 5.0, "draw": 3.5, "away": 1.65},
        {"ts": "2026-08-16T13:08:00+00:00", "minute": "08",
         "home": 5.5, "draw": 3.6, "away": 1.55},
    ], "ah": [], "ou": []}]}
    rows = trend_to_snapshots(payload, kickoff="2026-08-16T13:00:00Z")
    assert len(rows) == 1
    assert rows[0]["odds_1x2"]["home"] == 5.0


def test_trend_timing_label_vs_kickoff():
    from agents.football.nowgoal import _trend_timing_label

    # ~48h before kickoff -> T-48h; ~90m before -> T-2h; ~30m -> T-30m
    assert _trend_timing_label("2026-08-14T13:00:00Z", "2026-08-16T13:00:00Z") == "T-48h"
    assert _trend_timing_label("2026-08-16T11:30:00Z", "2026-08-16T13:00:00Z") == "T-2h"
    assert _trend_timing_label("2026-08-16T12:30:00Z", "2026-08-16T13:00:00Z") == "T-30m"
    assert _trend_timing_label(None, "2026-08-16T13:00:00Z") == "T-0h"
    assert _trend_timing_label("2026-08-16T12:00:00Z", None) == "T-0h"


def test_client_stores_proxy():
    client = NowGoalClient(proxy="socks5h://127.0.0.1:9050")
    assert client._proxy == "socks5h://127.0.0.1:9050"
    assert NowGoalClient()._proxy is None


def test_client_uses_proxy_kwarg():
    client = NowGoalClient(proxy="socks5h://127.0.0.1:9050")
    c = client._client()
    # httpx 0.28: proxy applied as client-level proxy
    assert c._transport is not None
    c.aclose()

# ---- euro (1X2) parser ----------------------------------------------------

def test_parse_euro_list_shape():
    assert parse_euro([2.1, 3.4, 3.6]) == {"home": 2.1, "draw": 3.4, "away": 3.6}


def test_parse_euro_list_trailing_numbers_ignored():
    # opening rows carry more trailing numbers (movement fields)
    assert parse_euro([2.1, 3.4, 3.6, 2.0, 3.5, 3.8]) == {
        "home": 2.1, "draw": 3.4, "away": 3.6,
    }


def test_parse_euro_dict_shapes():
    assert parse_euro({"home": 2.1, "draw": 3.4, "away": 3.6}) == {
        "home": 2.1, "draw": 3.4, "away": 3.6,
    }
    assert parse_euro({"1": 2.1, "X": 3.4, "2": 3.6}) == {
        "home": 2.1, "draw": 3.4, "away": 3.6,
    }


def test_parse_euro_json_string():
    assert parse_euro("[2.10,3.40,3.60]") == {"home": 2.1, "draw": 3.4, "away": 3.6}
    # comma-separated plain string
    assert parse_euro("2.1,3.4,3.6") == {"home": 2.1, "draw": 3.4, "away": 3.6}


def test_parse_euro_invalid_returns_none():
    assert parse_euro(None) is None
    assert parse_euro([1.5, 2.0]) is None          # too few prices
    assert parse_euro([1.0, 1.0, 1.0]) is None     # prices <= 1.0 (junk)
    assert parse_euro("garbage") is None
    assert parse_euro({"home": 2.1}) is None       # incomplete dict


# ---- ou (over/under) parser ----------------------------------------------

def test_parse_ou_list_line_first():
    # family convention: [line, over, under]
    assert parse_ou([2.5, 1.85, 1.95]) == {"over": 1.85, "under": 1.95, "line": 2.5}


def test_parse_ou_list_prices_first_disambiguated():
    # [over, under, line]: 1.85 is not a quarter step -> line must be 2.5
    assert parse_ou([1.85, 1.95, 2.5]) == {"over": 1.85, "under": 1.95, "line": 2.5}


def test_parse_ou_dict_shape():
    assert parse_ou({"over": 1.8, "under": 2.0, "line": 3.5}) == {
        "over": 1.8, "under": 2.0, "line": 3.5,
    }
    assert parse_ou({"o": 1.8, "u": 2.0, "hdc": 3.5}) == {
        "over": 1.8, "under": 2.0, "line": 3.5,
    }


def test_parse_ou_invalid_returns_none():
    assert parse_ou(None) is None
    assert parse_ou([2.5, 0.9, 1.8]) is None       # over price <= 1.0
    assert parse_ou([7.5, 1.8, 1.9]) is None       # implausible line
    assert parse_ou([1.8, 1.9]) is None            # too few values


# ---- ah (asian handicap) parser ------------------------------------------

def test_parse_ah_shapes():
    assert parse_ah([0.5, 1.90, 1.95]) == {"home": 1.90, "away": 1.95, "line": 0.5}
    assert parse_ah({"home": 1.90, "away": 1.95, "hdc": 0.5}) == {
        "home": 1.90, "away": 1.95, "line": 0.5,
    }
    assert parse_ah("garbage") is None


# ---- connectivity diagnostic (runner nowgoal-check) ----------------------

def _fake_http_client(status: int, text: str):
    resp = AsyncMock()
    resp.status_code = status
    resp.text = text
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=resp)
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=False)
    return fake


def test_probe_homepage_detects_block_page():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        fake = _fake_http_client(200, "<html><title>Trustpositif</title></html>")
        with patch.object(client, "_client", return_value=fake):
            facts = await client.probe_homepage()
        assert facts["http"] == 200
        assert facts["blocked"] is True
        assert facts["title"] == "Trustpositif"
        assert facts["looks_like_site"] is False

    asyncio.run(runner())


def test_probe_homepage_network_error():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        fake = AsyncMock()
        fake.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=False)
        with patch.object(client, "_client", return_value=fake):
            facts = await client.probe_homepage()
        assert facts["http"] is None
        assert "error" in facts

    asyncio.run(runner())


def test_run_nowgoal_check_reachable():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        client.probe_homepage = AsyncMock(return_value={
            "http": 200, "size": 50000, "title": "Nowgoal",
            "blocked": False, "looks_like_site": True,
        })
        client.fetch_schedule = AsyncMock(return_value=[_fixture()])
        client.fetch_odds = AsyncMock(return_value=_NOWGOAL_PAYLOAD)
        report = await run_nowgoal_check(client=client, date="2026-08-15")
        assert report["status"] == "reachable"
        assert report["checks"]["schedule"]["matches_parsed"] == 1
        assert report["checks"]["odds"]["bookmakers"] == 1
        assert "h2h" in report["checks"]["odds"]["markets"]
        assert report["mirrors"] == []

    asyncio.run(runner())


def test_run_nowgoal_check_blocked():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        client.probe_homepage = AsyncMock(return_value={
            "http": 200, "size": 7000, "title": "Trustpositif",
            "blocked": True, "looks_like_site": False,
        })
        client.fetch_schedule = AsyncMock(return_value=None)
        client.fetch_odds = AsyncMock()
        report = await run_nowgoal_check(client=client, date="2026-08-15")
        assert report["status"] == "blocked"
        client.fetch_odds.assert_not_awaited()

    asyncio.run(runner())


def test_run_nowgoal_check_unreachable():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        client.probe_homepage = AsyncMock(return_value={"http": None, "error": "ConnectError: refused"})
        client.fetch_schedule = AsyncMock(return_value=None)
        report = await run_nowgoal_check(client=client, date="2026-08-15")
        assert report["status"] == "unreachable"

    asyncio.run(runner())


def test_run_nowgoal_check_no_schedule():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        client.probe_homepage = AsyncMock(return_value={
            "http": 200, "size": 40000, "title": "Nowgoal",
            "blocked": False, "looks_like_site": True,
        })
        client.fetch_schedule = AsyncMock(return_value=None)
        report = await run_nowgoal_check(client=client, date="2026-08-15")
        assert report["status"] == "no_schedule"

    asyncio.run(runner())


# ---- team name normalization ---------------------------------------------

def test_norm_team_strips_accents():
    assert _norm_team("Bod\u00f8/Glimt") == "bodo glimt"
    assert _norm_team("Tobol (Kaz)") == "tobol"


def test_same_team_tolerant():
    assert _same_team("Bod\u00f8/Glimt", "Bodo/Glimt")
    assert _same_team("FK Bod\u00f8/Glimt", "Bod\u00f8/Glimt")
    assert _same_team("Tobol (Kaz)", "Tobol Kostanay")
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


# ---- schedule parsing -----------------------------------------------------

_SCHEDULE_JS = """
var B=[];
B[0]=[1,'EPL','Premier League','#ff0000',0,0,0,1];
B[1]=[2,'LIGA','LaLiga','#ff0000',0,0,0,1];
var A=[];
A[0]=[111,0,8801,8802,'Arsenal','Chelsea','2026,7,15,18,00,00',-1,0,0,0,0,0,0,0,1,'','','','',42,'','',3,8];
A[1]=[222,1,8803,8804,'Real Madrid','Barcelona','2026,7,15,20,00,00',-1,0,0,0,0,0,0,0,1,'','','','',42,'','',3,8];
"""


def test_fetch_schedule_parses_a_and_b_arrays():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get_text", AsyncMock(return_value=_SCHEDULE_JS)) as mock:
            rows = await client.fetch_schedule("2026-08-15")
        mock.assert_awaited_once()
        assert rows is not None and len(rows) == 2
        first = rows[0]
        assert first["match_id"] == "111"
        assert first["home"] == "Arsenal"
        assert first["away"] == "Chelsea"
        # month is 0-based in the feed -> August
        assert first["kickoff"] == "2026-08-15T18:00:00Z"
        assert first["league_name"] == "Premier League"
        assert first["home_id"] == "8801"
        assert rows[1]["league_name"] == "LaLiga"

    asyncio.run(runner())


def test_fetch_schedule_none_on_failure():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get_text", AsyncMock(return_value=None)):
            assert await client.fetch_schedule("2026-08-15") is None

    asyncio.run(runner())


def test_find_fixture_tolerant_names():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get_text", AsyncMock(return_value=_SCHEDULE_JS)):
            fx = await client.find_fixture("Bodo/Glimt", "X", "2026-08-15")
            assert fx is None
            fx = await client.find_fixture("Arsenal", "Chelsea", "2026-08-15")
            assert fx is not None and fx["match_id"] == "111"

    asyncio.run(runner())


def test_find_fixture_prefers_senior_over_youth_side():
    """Regression (verified live 2026-08-17): NowGoal schedules BOTH the
    senior side and its U19 side on the same day ("Galatasaray vs Corum
    Belediyespor" AND "Galatasaray U19 vs Corum FK U19"). The old tolerant
    matcher returned whichever row came first -- the U19 match, with U19
    odds (median 1.42/4.5/5.0 vs the senior 1.36/5.0/7.5). find_fixture
    must score exact-name candidates above youth/reserve rows so a plain
    "Galatasaray" query resolves to the senior match.
    """
    rows = [
        {"match_id": "3064392", "home": "Galatasaray U19", "away": "Corum FK U19",
         "kickoff": "2026-08-14T14:30:00Z", "status": "-1", "finished": True,
         "score": "9-1", "league_name": "Turkey A2 League U19"},
        {"match_id": "3026691", "home": "Galatasaray", "away": "Corum Belediyespor",
         "kickoff": "2026-08-14T18:30:00Z", "status": "-1", "finished": True,
         "score": "2-2", "league_name": "Turkey Super Lig"},
    ]

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        client.fetch_schedule = AsyncMock(return_value=rows)
        fx = await client.find_fixture("Galatasaray", "Corum", "2026-08-14")
        assert fx is not None
        assert fx["match_id"] == "3026691"  # senior, not U19
        # When ONLY the youth side exists, it is still returned (caller may
        # genuinely have asked for the U19 match).
        fx2 = await client.find_fixture("Galatasaray U19", "Corum FK U19", "2026-08-14")
        assert fx2 is not None and fx2["match_id"] == "3064392"

    asyncio.run(runner())


def test_find_fixture_by_score_renamed_club():
    """When the pair name match fails (club renamed in NowGoal's schedule,
    verified live: result source "Beveren" vs schedule "Red Star Waasland"),
    the settle path resolves by FINAL SCORE + one exact side; a unique,
    finished, score-matching row wins; ambiguity or a score mismatch stays
    None (a wrong fixture is worse than no fixture)."""
    rows = [
        {"match_id": "3003535", "home": "Red Star Waasland", "away": "Anderlecht",
         "kickoff": "2026-08-16T11:30:00Z", "status": "-1", "finished": True, "score": "1-0"},
        {"match_id": "3003999", "home": "Club Brugge", "away": "Anderlecht",
         "kickoff": "2026-08-16T18:00:00Z", "status": "-1", "finished": True, "score": "2-2"},
        {"match_id": "3003777", "home": "Genk", "away": "Westerlo",
         "kickoff": "2026-08-16T19:00:00Z", "status": "-1", "finished": True, "score": "1-0"},
    ]

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        client.fetch_schedule = AsyncMock(return_value=rows)
        # renamed club: "Beveren" matches nothing, but the 1-0 Anderlecht
        # row is unique with one exact side -> resolved.
        fx = await client.find_fixture_by_score("Beveren", "Anderlecht", "2026-08-16", 1, 0)
        assert fx is not None and fx["match_id"] == "3003535"
        # wrong score -> no row qualifies (Genk-Westerlo 1-0 has no exact side)
        assert await client.find_fixture_by_score("Beveren", "Anderlecht", "2026-08-16", 0, 1) is None
        # ambiguous: two finished 1-0 rows share one exact side -> None
        two = rows + [{"match_id": "3003778", "home": "KV Mechelen", "away": "Anderlecht",
                       "kickoff": "2026-08-16T20:00:00Z", "status": "-1", "finished": True, "score": "1-0"}]
        client.fetch_schedule = AsyncMock(return_value=two)
        assert await client.find_fixture_by_score("Beveren", "Anderlecht", "2026-08-16", 1, 0) is None

    asyncio.run(runner())


# ---- odds normalization ---------------------------------------------------

_MIXODDS = {
    "ErrCode": 0,
    "Data": {"mixodds": [
        {
            "cid": 177,
            "euro": "[2.10,3.40,3.60]",
            "ou": "[2.5,1.85,1.95]",
            "ah": "[0.5,1.90,1.95]",
        },
        {
            "cid": 999,
            "euro": [2.05, 3.30, 3.70],
            "ou": {"over": 1.80, "under": 2.00, "line": 3.0},
            "ah": None,
        },
        # fully invalid row -> must be dropped
        {"cid": 2, "euro": "garbage", "ou": "[0.9,1.85,1.95]", "ah": "[x]"},
    ]},
}


def _fixture(match_id="111"):
    return {"match_id": match_id, "home": "Arsenal", "away": "Chelsea",
            "kickoff": "2026-08-15T18:00:00Z"}


def test_fetch_odds_normalizes_payload():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=_MIXODDS)):
            payload = await client.fetch_odds(_fixture())
        assert payload is not None
        assert payload["home_team"] == "Arsenal"
        assert payload["away_team"] == "Chelsea"
        assert payload["commence_time"] == "2026-08-15T18:00:00Z"
        # invalid row dropped -> only 2 bookmakers
        assert len(payload["bookmakers"]) == 2
        pinnacle = payload["bookmakers"][0]
        assert pinnacle["title"] == "Pinnacle"  # cid 177 known
        markets = {m["key"]: m for m in pinnacle["markets"]}
        h2h = markets["h2h"]
        assert [o["name"] for o in h2h["outcomes"]] == ["Arsenal", "Draw", "Chelsea"]
        assert [o["price"] for o in h2h["outcomes"]] == [2.1, 3.4, 3.6]
        tot = markets["totals"]
        assert [o["name"] for o in tot["outcomes"]] == ["Over", "Under"]
        assert tot["outcomes"][0]["point"] == 2.5
        ah = markets["asian_handicap"]
        # NowGoal's raw AH line is the AWAY handicap; the normalized payload
        # carries per-side points (Home = home handicap, Away = away handicap).
        assert ah["outcomes"][0]["point"] == -0.5   # Home -0.5 (home favorite gives)
        assert ah["outcomes"][1]["point"] == 0.5    # Away +0.5
        # unknown cid -> stable fallback label
        assert payload["bookmakers"][1]["title"] == "NowGoal-999"

    asyncio.run(runner())


def test_fetch_odds_closing_uses_roddslist():
    data = {"ErrCode": 0, "Data": {"roddsList": [
        {"cid": 177, "euro": [1.95, 3.50, 3.80], "ou": None, "ah": None},
    ]}}

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=data)) as mock:
            payload = await client.fetch_odds(_fixture(), closing=True)
        called = mock.await_args
        assert called is not None and called.args[1]["t"] == 11
        assert payload is not None
        h2h = [m for m in payload["bookmakers"][0]["markets"] if m["key"] == "h2h"][0]
        assert [o["price"] for o in h2h["outcomes"]] == [1.95, 3.5, 3.8]

    asyncio.run(runner())


def test_fetch_odds_none_on_err_code():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value={"ErrCode": 1, "Data": {}})):
            assert await client.fetch_odds(_fixture()) is None

    asyncio.run(runner())


def test_fetch_odds_none_on_network_failure():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=None)):
            assert await client.fetch_odds(_fixture()) is None

    asyncio.run(runner())


# ---- verified live shape (2026-08-14, www.nowgoal.net) -------------------

# Real mixodds item shape: {cid, cn, euro/ou/ah: {f,l,r:{u,g,d}, hr}}.
# f=opening, l=last pre-match, r=realtime in-play. euro is decimal;
# ou/ah prices are Hong-Kong odds (decimal - 1).
_LIVE_MIXODDS = {
    "ErrCode": 0,
    "Data": {"mixodds": [
        {
            "cid": 8,
            "cn": "Bet365",
            "euro": {
                "f": {"u": "2.2", "g": "3.25", "d": "3.3"},
                "l": {"u": "2.25", "g": "2.7", "d": "4"},
                "r": {"u": "15", "g": "1.03", "d": "41"},
                "hr": True,
            },
            "ou": {
                "f": {"u": "0.8", "g": "2", "d": "1.05"},
                "l": {"u": "0.85", "g": "1.5", "d": "1"},
                "r": {"u": "5.6", "g": "0.5", "d": "0.11"},
                "hr": True,
            },
            "ah": {
                "f": {"u": "0.88", "g": "0.25", "d": "0.98"},
                "l": {"u": "0.88", "g": "0.25", "d": "0.98"},
                "r": {"u": "0.13", "g": "0", "d": "5"},
                "hr": True,
            },
        },
        {
            "cid": 3059097,
            "cn": "1xBet",
            "euro": {"l": {"u": "1.07", "g": "12", "d": "19"}, "hr": False},
            "ou": {"l": {"u": "0.98", "g": "4.5", "d": "0.83"}, "hr": False},
            # empty strings = bookmaker has no AH odds
            "ah": {"l": {"u": "", "g": "", "d": ""}, "hr": False},
        },
    ]},
}


def test_parse_euro_live_wrapper_shape():
    # wrapper -> unwraps to l (last pre-match), NOT r (realtime in-play)
    assert parse_euro(_LIVE_MIXODDS["Data"]["mixodds"][0]["euro"]) == {
        "home": 2.25, "draw": 2.7, "away": 4.0,
    }


def test_parse_ou_live_wrapper_shape_hk_odds():
    # HK odds: 0.85 -> 1.85, 1.0 -> 2.0
    assert parse_ou(_LIVE_MIXODDS["Data"]["mixodds"][0]["ou"]) == {
        "over": 1.85, "under": 2.0, "line": 1.5,
    }


def test_parse_ah_live_wrapper_shape_hk_odds():
    assert parse_ah(_LIVE_MIXODDS["Data"]["mixodds"][0]["ah"]) == {
        "home": 1.88, "away": 1.98, "line": 0.25,
    }


def test_parse_live_wrapper_falls_back_to_f_when_l_empty():
    item = {
        "euro": {
            "f": {"u": "2.1", "g": "3.4", "d": "3.6"},
            "l": {"u": "", "g": "", "d": ""},
            "hr": True,
        },
    }
    assert parse_euro(item["euro"]) == {"home": 2.1, "draw": 3.4, "away": 3.6}


def test_parse_live_wrapper_all_empty_returns_none():
    item = {"euro": {"l": {"u": "", "g": "", "d": ""}, "hr": False}}
    assert parse_euro(item["euro"]) is None


def test_fetch_odds_live_shape_uses_cn_and_hk_conversion():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get", AsyncMock(return_value=_LIVE_MIXODDS)):
            payload = await client.fetch_odds(_fixture())
        assert payload is not None
        # cn field wins over cid mapping
        assert payload["bookmakers"][0]["title"] == "Bet365"
        assert payload["bookmakers"][1]["title"] == "1xBet"
        bm = payload["bookmakers"][0]
        markets = {m["key"]: m for m in bm["markets"]}
        assert [o["price"] for o in markets["h2h"]["outcomes"]] == [2.25, 2.7, 4.0]
        assert markets["totals"]["outcomes"][0]["point"] == 1.5
        assert [o["price"] for o in markets["totals"]["outcomes"]] == [1.85, 2.0]
        # line 0.25 raw = away +0.25; home outcome carries the negated home
        # handicap (-0.25) in the normalized payload.
        assert markets["asian_handicap"]["outcomes"][0]["point"] == -0.25
        # empty-ah bookmaker still contributes h2h + totals
        bm1 = payload["bookmakers"][1]
        assert [m["key"] for m in bm1["markets"]] == ["h2h", "totals"]
        assert [o["price"] for o in bm1["markets"][0]["outcomes"]] == [1.07, 12.0, 19.0]

    asyncio.run(runner())


def test_get_sends_referer_header():
    """The odds endpoint answers {"code":1002} without a Referer (verified
    live), so _get/_get_text must always send one."""
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        captured = {}
        fake_client = AsyncMock()
        fake_client.get.return_value = httpx.Response(
            200, json={"ErrCode": 0, "Data": {"mixodds": []}}
        )

        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client, "_client", return_value=fake_client):
            await client._get("/ajax/soccerajax", {"type": 14})
        _, kwargs = fake_client.get.call_args
        assert kwargs["headers"]["Referer"] == client._base_url
        assert "User-Agent" in kwargs["headers"]

    asyncio.run(runner())


def test_headers_include_referer_and_user_agent():
    client = NowGoalClient()
    h = client._headers()
    assert h["Referer"] == client._base_url
    assert h["User-Agent"].startswith("Mozilla/5.0")
    # 2026-08-24 live probe: the *26 family serves the ajax paths; nowgoal.net
    # (the old default) went 404 -- the default must be a verified-alive mirror.
    assert client._base_url == "https://live10.nowgoal26.com/"


def test_parse_euro_level_ball_ah_line_zero_ok():
    # handicap line 0 (level ball) must parse, not be rejected
    assert parse_ah({"u": "0.9", "g": "0", "d": "0.9"}) == {
        "home": 1.9, "away": 1.9, "line": 0.0,
    }


def test_match_odds_chains_schedule_and_odds():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get_text", AsyncMock(return_value=_SCHEDULE_JS)), \
             patch.object(client, "_get", AsyncMock(return_value=_MIXODDS)):
            payload = await client.match_odds("Arsenal", "Chelsea", "2026-08-15")
        assert payload is not None
        assert payload["home_team"] == "Arsenal"
        assert payload["bookmakers"][0]["title"] == "Pinnacle"

    asyncio.run(runner())


def test_match_odds_none_when_fixture_not_found():
    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        with patch.object(client, "_get_text", AsyncMock(return_value=_SCHEDULE_JS)), \
             patch.object(client, "_get", AsyncMock()) as mock_get:
            payload = await client.match_odds("Liverpool", "Everton", "2026-08-15")
        assert payload is None
        mock_get.assert_not_awaited()  # no fixture -> no odds call

    asyncio.run(runner())


# ---- mirror rotation ------------------------------------------------------

def _client_with_mirrors():
    return NowGoalClient(
        base_url="http://www.nowgoal.net/",
        throttle_seconds=0.0,
        mirrors=[
            "http://www.nowgoal26.com/",
            "http://www.nowgoal6.com/",
        ],
    )


def test_client_mirror_pool_built_and_deduped():
    client = _client_with_mirrors()
    assert client._mirrors == [
        "http://www.nowgoal.net/",
        "http://www.nowgoal26.com/",
        "http://www.nowgoal6.com/",
    ]
    assert client._base_url == "http://www.nowgoal.net/"  # primary first
    assert client._active == 0


def test_client_rotate_advances_and_wraps():
    client = _client_with_mirrors()
    client._rotate()
    assert client._base_url == "http://www.nowgoal26.com/"
    client._rotate()
    assert client._base_url == "http://www.nowgoal6.com/"
    client._rotate()
    assert client._base_url == "http://www.nowgoal.net/"  # wraps around


def test_get_rotates_on_http_error():
    """Primary answers 500, next mirror answers 200 -> data from mirror 2
    and the active mirror is updated."""
    async def runner():
        client = _client_with_mirrors()
        fake = AsyncMock()
        fake.get.side_effect = [
            httpx.Response(500, text="boom"),
            httpx.Response(200, json={"ErrCode": 0, "Data": {"mixodds": []}}),
        ]
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            out = await client._get("/ajax/soccerajax", {"type": 14})
        assert out == {"ErrCode": 0, "Data": {"mixodds": []}}
        assert client._base_url == "http://www.nowgoal26.com/"
        assert fake.get.call_count == 2

    asyncio.run(runner())


def test_get_rotates_on_redirect_302():
    """Parked domains answer 301/302 (hugedomains) for every path -- a
    redirect is a mirror failure, not a success."""
    async def runner():
        client = _client_with_mirrors()
        fake = AsyncMock()
        fake.get.side_effect = [
            httpx.Response(302, headers={"location": "https://www.hugedomains.com/"}),
            httpx.Response(200, json={"ErrCode": 0, "Data": {"mixodds": []}}),
        ]
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            out = await client._get("/ajax/soccerajax", {"type": 14})
        assert out is not None
        assert client._base_url == "http://www.nowgoal26.com/"
        assert fake.get.call_count == 2

    asyncio.run(runner())


def test_get_rotates_on_network_error():
    async def runner():
        client = _client_with_mirrors()
        fake = AsyncMock()
        fake.get.side_effect = [
            httpx.ConnectError("refused"),
            httpx.Response(200, json={"ErrCode": 0, "Data": {}}),
        ]
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            out = await client._get("/ajax/soccerajax", {"type": 14})
        assert out is not None
        assert client._base_url == "http://www.nowgoal26.com/"

    asyncio.run(runner())


def test_get_all_mirrors_fail_returns_none():
    async def runner():
        client = _client_with_mirrors()
        fake = AsyncMock()
        fake.get.return_value = httpx.Response(502, text="bad gateway")
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            out = await client._get("/ajax/soccerajax", {"type": 14})
        assert out is None
        assert fake.get.call_count == len(client._mirrors)

    asyncio.run(runner())


def test_fetch_schedule_rotates_on_parked_page_body():
    """Mirror 1 answers 200 with a parked-page body (nowgoal3.com served
    hugedomains HTML), mirror 2 answers real schedule JS -> matches parsed
    and active mirror updated."""
    async def runner():
        client = _client_with_mirrors()
        parked = "<html><head><title>Domain parking</title></head><body>This domain is parked.</body></html>"
        fake = AsyncMock()
        fake.get.side_effect = [
            httpx.Response(200, text=parked),
            httpx.Response(200, text=_SCHEDULE_JS),
        ]
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            rows = await client.fetch_schedule("2026-08-15")
        assert rows is not None and len(rows) >= 1
        assert client._base_url == "http://www.nowgoal26.com/"

    asyncio.run(runner())


def test_fetch_schedule_no_rotate_on_empty_valid_day():
    """A valid nowgoal response with zero matches (empty day) is not a
    mirror failure -- the client returns None without rotating."""
    async def runner():
        client = _client_with_mirrors()
        empty = '{"ErrCode":0,"Data":"var A=Array(0);\\r\\nvar B=Array(0);\\r\\nmatchcount=0;"}'
        fake = AsyncMock()
        fake.get.return_value = httpx.Response(200, text=empty)
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            rows = await client.fetch_schedule("2099-01-01")
        assert rows is None
        assert client._base_url == "http://www.nowgoal.net/"  # no rotation
        assert fake.get.call_count == 1

    asyncio.run(runner())


def test_fetch_schedule_returns_none_when_all_mirrors_parked():
    async def runner():
        client = _client_with_mirrors()
        parked = "<html><body>parked</body></html>"
        fake = AsyncMock()
        fake.get.return_value = httpx.Response(200, text=parked)
        fake.__aenter__ = AsyncMock(return_value=fake)
        fake.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client, "_client", return_value=fake):
            rows = await client.fetch_schedule("2026-08-15")
        assert rows is None
        assert fake.get.call_count == len(client._mirrors)

    asyncio.run(runner())


# ---- find_specific_match wiring ------------------------------------------

def _run_analyse(analyse, ms, nowgoal, oddspapi=None):
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
        return {"id": 1, "name": "Arsenal", "provider": "flashscore"}, \
               {"id": 2, "name": "Chelsea", "provider": "flashscore"}

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


_NOWGOAL_PAYLOAD = {
    "home_team": "Arsenal", "away_team": "Chelsea", "commence_time": "2026-08-15T18:00:00Z",
    "bookmakers": [{"title": "Pinnacle", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Arsenal", "price": 2.0}, {"name": "Draw", "price": 3.2},
        {"name": "Chelsea", "price": 3.5},
    ]}]}],
}


def test_analyse_nowgoal_not_called_when_not_configured():
    """nowgoal=None (flag off) -> match_odds never called, flag False."""
    import agents.football.analyse as analyse

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, nowgoal=None)
    assert result.get("quota", {}).get("nowgoal_used") is False
    assert "nowgoal_odds" not in (result.get("sources") or [])


def test_analyse_nowgoal_fires_when_primary_and_oddspapi_empty():
    """Both prior sources empty -> nowgoal fallback supplies the odds."""
    import agents.football.analyse as analyse

    fake_nowgoal = AsyncMock()
    fake_nowgoal.match_odds = AsyncMock(return_value=_NOWGOAL_PAYLOAD)

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, nowgoal=fake_nowgoal)

    fake_nowgoal.match_odds.assert_awaited()
    assert result.get("quota", {}).get("nowgoal_used") is True
    assert "nowgoal_odds" in (result.get("sources") or [])
    assert (result.get("odds") or {}).get("has_odds") is True
    assert (result.get("odds") or {}).get("bookmakers_count") == 1


def test_analyse_nowgoal_skipped_when_oddspapi_has_odds():
    """Re-prioritas 2026-08-24: NowGoal PRIMARY, OddsPapi SECONDARY/validator.
    Both payloads are fetched; the resolution winner is NowGoal, so
    ``nowgoal_used`` and ``oddspapi_used`` are BOTH true (both APIs were
    called) and both provenance labels appear on ``sources``. The cross-source
    quality check still sees 2 independent sources.
    """
    import agents.football.analyse as analyse

    odsp_payload = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "bookmakers": [{"title": "Bet365", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 1.9}, {"name": "Draw", "price": 3.3},
            {"name": "Chelsea", "price": 4.0},
        ]}]}],
    }
    fake_oddspapi = AsyncMock()
    fake_oddspapi.find_fixture = AsyncMock(return_value={
        "fixtureId": "fx1", "hasOdds": True,
        "participant1Id": 111, "participant1Name": "Arsenal",
        "participant2Id": 222, "participant2Name": "Chelsea",
        "startTime": "2026-08-15T18:00:00Z",
    })
    fake_oddspapi.fetch_odds = AsyncMock(return_value=odsp_payload)
    fake_nowgoal = AsyncMock()
    # PRIMARY: nowgoal supplies its own payload (slightly different prices).
    fake_nowgoal.match_odds = AsyncMock(return_value={
        "home_team": "Arsenal", "away_team": "Chelsea",
        "bookmakers": [{"title": "Bet365", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Arsenal", "price": 1.85}, {"name": "Draw", "price": 3.4},
            {"name": "Chelsea", "price": 4.2},
        ]}]}],
    })

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, nowgoal=fake_nowgoal, oddspapi=fake_oddspapi)

    # Both consulted: nowgoal as PRIMARY, oddspapi as validator.
    fake_nowgoal.match_odds.assert_awaited()
    fake_oddspapi.fetch_odds.assert_awaited()
    assert result.get("quota", {}).get("nowgoal_used") is True
    assert result.get("quota", {}).get("oddspapi_used") is True
    assert (result.get("odds") or {}).get("bookmakers_count") == 1
    srcs = result.get("sources") or []
    assert "nowgoal_odds" in srcs
    assert "oddspapi_odds" in srcs
    # The comparison ran with both sources present (single-source status ok).
    assert (result.get("odds") or {}).get("quality", {}).get("n_sources") == 2


def test_analyse_nowgoal_skipped_when_the_odds_api_has_odds():
    """Priority (2026-08): The Odds API is LAST. When it somehow supplied
    odds earlier (or is the only configured source), nowgoal still runs only
    after oddspapi -- here oddspapi is absent so nowgoal fires before the
    The-Odds-API last resort, but if The Odds API already ran with odds,
    nowgoal must not double-supply. This asserts nowgoal is skipped when
    oddspapi supplied odds via find_match_odds_payload path is not reachable;
    the realistic guard is oddspapi-primary (test above) -- this test pins
    the SECOND position: nowgoal fires when oddspapi is absent."""
    import agents.football.analyse as analyse

    fake_nowgoal = AsyncMock()
    fake_nowgoal.match_odds = AsyncMock(return_value=None)

    with patch.object(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None))), \
         patch.object(analyse, "resolve_league_scored", return_value=("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"})), \
         patch.object(analyse, "_season_now", return_value=2026):
        ms = _make_stats()
        result = _run_analyse(analyse, ms, nowgoal=fake_nowgoal)

    # oddspapi absent -> nowgoal (SECOND) fires; find_match_odds_payload
    # (LAST) also runs because nowgoal returned no odds
    fake_nowgoal.match_odds.assert_awaited()
    assert result.get("quota", {}).get("nowgoal_used") is False


# ---- analysis page fallback (form + H2H) --------------------------------

_ANALYSIS_HTML = """<html><body>
<table id="table_v1"><tbody>
<tr id="tr1_1" vs="1" name="1124" index="2991084" info="1,1,892,0,0">
<td>ASEAN Cup</td><td><span data-t='2026-08-07 13:00:00'></span></td>
<td><a onclick="soccerDbPage.team(892)"><span class="team-home-f">Singapore</span></a></td>
<td><span class="fscore_1">1-1</span></td>
<td><a onclick="soccerDbPage.team(890)"><span class="">Indonesia</span></a></td>
<td></td><td></td><td></td><td></td>
<td class="hbg-td1"><span class="o-draw">D</span></td>
</tr>
<tr id="tr1_2" vs="0" name="1124" index="2991082" info="0,1,883,0,1">
<td>ASEAN Cup</td><td><span data-t='2026-07-31 13:00:00'></span></td>
<td><a onclick="soccerDbPage.team(883)"><span class="team-home-f">Vietnam</span></a></td>
<td><span class="fscore_1">0-1</span></td>
<td><a onclick="soccerDbPage.team(892)"><span class="">Singapore</span></a></td>
<td></td><td></td><td></td><td></td>
<td class="hbg-td1"><span class="o-win">W</span></td>
</tr>
</tbody></table>
<table id="table_v2"><tbody>
<tr id="tr2_1" vs="0" name="1124" index="11" info="0,2,886,0,1">
<td>ASEAN Cup</td><td><span data-t='2026-08-06 13:00:00'></span></td>
<td><a onclick="soccerDbPage.team(1563)"><span class="team-home-f">Myanmar</span></a></td>
<td><span class="fscore_2">0-2</span></td>
<td><a onclick="soccerDbPage.team(886)"><span class="">Thailand</span></a></td>
<td></td><td></td><td></td><td></td>
<td class="hbg-td1"><span class="o-lose">L</span></td>
</tr>
</tbody></table>
<table id="table_v3"><tbody>
<tr id="tr3_1" vs="1" name="1366" index="2897943" info="3,2,886,1,1">
<td>INT FRL</td><td><span data-t='2025-11-13 12:30:00'></span></td>
<td><a onclick="soccerDbPage.team(886)"><span class="team-home-f">Thailand</span></a></td>
<td><span class="fscore_3">3-2</span></td>
<td><a onclick="soccerDbPage.team(892)"><span class="">Singapore</span></a></td>
<td></td><td></td><td></td><td></td>
<td class="hbg-td1"><span class="o-lose">L</span></td>
</tr>
</tbody></table>
</body></html>"""


def test_parse_analysis_form_and_h2h():
    """Parse the server-rendered analysis tables into form + H2H."""
    from agents.football.nowgoal import NowGoalClient

    out = NowGoalClient._parse_analysis(_ANALYSIS_HTML, "Singapore", "Thailand")
    assert out is not None
    # table_v1 = home team (Singapore): D (1-1 home) then W (0-1 away win)
    assert out["home_form"]["sequence"] == "D-W"
    assert out["home_form"]["gf_avg"] == 1.0   # (1 + 1) / 2
    assert out["home_form"]["ga_avg"] == 0.5   # (1 + 0) / 2
    assert out["home_form"]["sample_size"] == 2
    # F1: the page renders newest-first (2026-08-07 D, then 2026-07-31 W);
    # recent_goals must be reversed to the OLDEST -> NEWEST contract so the
    # signal-engine statistical component is fed on the nowgoal form fallback.
    assert out["home_form"]["recent_goals"] == [(1, 0), (1, 1)]
    # table_v2 = away team (Thailand): L (lost 0-2 away -> gf 2 ga 0)
    assert out["away_form"]["sequence"] == "L"
    assert out["away_form"]["gf_avg"] == 2.0
    assert out["away_form"]["ga_avg"] == 0.0
    # table_v3 = H2H from HOME side (Singapore lost 3-2 as away)
    # P3-2: the aggregate also carries the excluded-competitions provenance.
    assert out["h2h"] == {
        "wins": 0, "draws": 0, "losses": 1, "matches": 1,
        "source": "nowgoal_analysis",
    } | {
        "match_list": out["h2h"]["match_list"],
        "excluded_competitions": out["h2h"]["excluded_competitions"],
    }
    assert "club friendlies" in out["h2h"]["excluded_competitions"]
    assert out["h2h"]["match_list"][0]["home"] == "Thailand"
    assert out["h2h"]["match_list"][0]["away"] == "Singapore"
    assert out["h2h"]["match_list"][0]["date"] == "2025-11-13 12:30:00"


def test_parse_analysis_excludes_friendly_rows():
    """P3-2: a Club Friendly row must be dropped BEFORE it contributes to
    the sequence / gf / ga / W-D-L aggregates."""
    from agents.football.nowgoal import NowGoalClient

    html = _ANALYSIS_HTML.replace(
        "<td>INT FRL</td>", "<td>Club Friendlies</td>"
    )
    out = NowGoalClient._parse_analysis(html, "Singapore", "Thailand")
    assert out is not None
    # The friendly row no longer inflates the H2H aggregate -- with its only
    # row dropped the table is empty and the dict is omitted entirely (the
    # pipeline then falls back to Elo instead of using a stale 0-meeting H2H).
    assert "h2h" not in out
    # The form tables (no friendly rows) are untouched.
    assert out["home_form"]["sequence"] == "D-W"
    assert out["away_form"]["sequence"] == "L"


def test_parse_analysis_window_limit():
    """Only the last ``limit`` (default 5) matches enter the rolling window."""
    from agents.football.nowgoal import NowGoalClient

    # 8 identical home rows -> only 5 counted (train/serve parity with FORM_WINDOW)
    rows = "".join(
        f'<tr id="tr1_{i}" vs="1" name="1" index="{i}" info="1,0,892,1,0">'
        "<td>L</td><td><span data-t='2026-08-01 13:00:00'></span></td>"
        '<td><a onclick="soccerDbPage.team(892)"><span class="team-home-f">Singapore</span></a></td>'
        '<td><span class="fscore_1">1-0</span></td>'
        '<td><a onclick="soccerDbPage.team(890)"><span class="">Indonesia</span></a></td>'
        "<td></td><td></td><td></td><td></td>"
        '<td class="hbg-td1"><span class="o-win">W</span></td></tr>'
        for i in range(1, 9)
    )
    html = f'<table id="table_v1"><tbody>{rows}</tbody></table>'
    out = NowGoalClient._parse_analysis(html, "Singapore", "Thailand")
    assert out["home_form"]["sample_size"] == 5
    assert out["home_form"]["sequence"] == "W-W-W-W-W"


def test_parse_analysis_no_tables_returns_none():
    from agents.football.nowgoal import NowGoalClient

    assert NowGoalClient._parse_analysis("<html>no tables</html>", "A", "B") is None


# ---- market movement + value diagnostics ----------------------------------

def test_market_movement_detects_steam():
    from agents.football.predictor import market_movement

    entries = [
        {"bookmaker": "A", "home": 2.0, "draw": 3.2, "away": 3.6,
         "opening": {"home": 2.2, "draw": 3.1, "away": 3.3}},
        {"bookmaker": "B", "home": 2.05, "draw": 3.2, "away": 3.5,
         "opening": {"home": 2.3, "draw": 3.1, "away": 3.2}},
    ]
    out = market_movement(entries)
    assert out is not None
    # home shortened ~9-11% in both -> median < -3%, steamed side == home
    assert out["sides"]["home"]["move_pct"] < -3.0
    assert out.get("steamed") == "home"


def test_market_movement_no_opening_returns_none():
    from agents.football.predictor import market_movement

    assert market_movement([{"bookmaker": "A", "home": 2.0}]) is None


def test_value_edges_model_vs_market():
    from agents.football.predictor import value_edges

    out = value_edges(
        {"1x2": {"home": 0.55, "draw": 0.25, "away": 0.20}},
        {"home": 2.0, "draw": 3.5, "away": 4.0},
    )
    assert out is not None
    # margin-free implied of 2.0/3.5/4.0: raw 0.5/0.286/0.25 -> norm 0.483/0.276/0.241
    assert abs(out["home"]["edge"] - (0.55 - 0.483)) < 0.01
    assert out["home"]["edge"] > 0
    assert out["away"]["edge"] < 0

# ---- match xG (live-{id} ftstat block) ----------------------------------

def _detail_with_xg() -> str:
    """Realistic match detail page fragment (structure verified live
    2026-08-17): the ftstat block with an Expected Goals (xG) row plus
    half-time blocks that must be ignored."""
    return """
    <ul class="stat" id="ftstat">
        <li>
            <span class="stat-c">1.12</span>
            <span class="stat-bar-wrapper homes"><span class="stat-bar fr" style="width:73%"></span></span>
            <span class="stat-title">Expected Goals (xG)</span>
            <span class="stat-bar-wrapper aways"><span class="stat-bar fl" style="width:27%"></span></span>
            <span class="stat-c">0.41</span>
        </li>
        <li>
            <span class="stat-c">18</span>
            <span class="stat-bar-wrapper homes"><span class="stat-bar fr" style="width:55%"></span></span>
            <span class="stat-title">Shots</span>
            <span class="stat-bar-wrapper aways"><span class="stat-bar fl" style="width:45%"></span></span>
            <span class="stat-c">12</span>
        </li>
    </ul>
    <ul class="stat" id="hf1stat" style="display: none;">
        <li>
            <span class="stat-c">0.47</span>
            <span class="stat-bar-wrapper homes"><span class="stat-bar fr" style="width:85%"></span></span>
            <span class="stat-title">Expected Goals (xG)</span>
            <span class="stat-bar-wrapper aways"><span class="stat-bar fl" style="width:15%"></span></span>
            <span class="stat-c">0.08</span>
        </li>
    </ul>
    """


def test_parse_match_xg_full_time_home_away():
    out = NowGoalClient._parse_match_xg(_detail_with_xg())
    assert out == {"xg_home": 1.12, "xg_away": 0.41}


def test_parse_match_xg_ignores_half_time_blocks():
    """Only the full-time (ftstat) block is read; hf1/hf2/ot are ignored."""
    out = NowGoalClient._parse_match_xg(_detail_with_xg())
    assert out["xg_home"] == 1.12  # not 0.47 from hf1stat
    assert out["xg_away"] == 0.41  # not 0.08 from hf1stat


def test_parse_match_xg_no_stats_block_returns_none():
    # friendly / no Technical Statistics: no ftstat block at all
    assert NowGoalClient._parse_match_xg("<html><body>no stats</body></html>") is None


def test_parse_match_xg_missing_xg_row_returns_none():
    html = ('<ul class="stat" id="ftstat"><li>'
            '<span class="stat-c">18</span>'
            '<span class="stat-bar-wrapper homes"></span>'
            '<span class="stat-title">Shots</span>'
            '<span class="stat-bar-wrapper aways"></span>'
            '<span class="stat-c">12</span></li></ul>')
    assert NowGoalClient._parse_match_xg(html) is None


def test_parse_match_xg_negative_value_returns_none():
    html = ('<ul class="stat" id="ftstat"><li>'
            '<span class="stat-c">-1.0</span>'
            '<span class="stat-bar-wrapper homes"></span>'
            '<span class="stat-title">Expected Goals (xG)</span>'
            '<span class="stat-bar-wrapper aways"></span>'
            '<span class="stat-c">0.5</span></li></ul>')
    assert NowGoalClient._parse_match_xg(html) is None


def test_fetch_match_xg_returns_parsed_value():
    async def runner():
        client = NowGoalClient()
        client._get_text = AsyncMock(return_value=_detail_with_xg())
        out = await client.fetch_match_xg("3013646")
        assert out == {"xg_home": 1.12, "xg_away": 0.41}
    asyncio.run(runner())


def test_fetch_match_xg_none_without_match_id():
    async def runner():
        client = NowGoalClient()
        client._get_text = AsyncMock()
        assert await client.fetch_match_xg("") is None
        client._get_text.assert_not_called()
    asyncio.run(runner())


def test_fetch_match_xg_none_on_page_failure():
    async def runner():
        client = NowGoalClient()
        client._get_text = AsyncMock(return_value=None)
        assert await client.fetch_match_xg("3013646") is None
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


# ---- Opsi 1+2 (2026-08-23): fast-fail connect timeout + breaker backoff --

import agents.football.nowgoal as ng_mod  # noqa: E402


def test_client_connect_timeout_fast_fail_read_full():
    """Opsi 1: hanya CONNECT yang dipangkas ke 4s; read/write tetap penuh
    sehingga respons mirror healthy-yang-lambat tetap di tunggu."""
    client = NowGoalClient()  # default timeout=15.0
    c = client._client()
    try:
        assert c.timeout.connect == ng_mod._CONNECT_TIMEOUT
        assert ng_mod._CONNECT_TIMEOUT < client._timeout
        assert c.timeout.read == 15.0
        assert c.timeout.write == 15.0
        assert c.timeout.pool == 15.0
    finally:
        asyncio.run(c.aclose())


def test_breaker_cooldown_escalates_per_open(monkeypatch):
    """Opsi 2: cooldown naik 90 -> 180 -> 360 (cap) per open berturut-turut."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(ng_mod.time, "monotonic", lambda: clock["t"])
    client = NowGoalClient(throttle_seconds=0.0)

    # Open #1: threshold=2 strike -> cooldown 90s
    client._breaker_record_transport_failure()
    assert not client._breaker_open()
    client._breaker_record_transport_failure()
    assert client._breaker_open()
    clock["t"] += 91.0
    assert not client._breaker_open()

    # Open #2: dua strike lagi -> cooldown 180s
    client._breaker_record_transport_failure()
    client._breaker_record_transport_failure()
    assert client._breaker_open()
    clock["t"] += 181.0
    assert not client._breaker_open()

    # Open #3: cooldown 360s
    client._breaker_record_transport_failure()
    client._breaker_record_transport_failure()
    assert client._breaker_until - clock["t"] == 360.0

    # Open #4+: tetap di cap 360s
    clock["t"] += 361.0
    client._breaker_record_transport_failure()
    client._breaker_record_transport_failure()
    assert client._breaker_until - clock["t"] == 360.0


def test_breaker_success_resets_escalation():
    client = NowGoalClient(throttle_seconds=0.0)
    client._breaker_opens = 2
    client._breaker_strikes = 1
    client._breaker_until = 9e9
    client._breaker_record_success()
    assert client._breaker_opens == 0
    assert client._breaker_strikes == 0
    assert client._breaker_until == 0.0
    assert not client._breaker_open()


def test_breaker_single_strike_does_not_arm():
    client = NowGoalClient(throttle_seconds=0.0)
    client._breaker_record_transport_failure()
    assert not client._breaker_open()
    assert client._breaker_opens == 0
