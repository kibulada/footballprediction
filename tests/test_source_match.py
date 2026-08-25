"""Tests for the `!livescore` / `!flashscore` match-source commands.

Both commands share ONE flow: find + validate the requested match on the
named source (today, then tomorrow), collect the source's match data, then
hand the validated identity to the EXISTING analyse pipeline
(``find_specific_match``) which runs the NowGoal odds lookup + prediction
engine + decision engine + existing output format. No new prediction logic
exists anywhere in the new commands.

These tests pin:
  - the bot parsing/routing for `!livescore` / `!flashscore`,
  - the today -> tomorrow search order and league+teams validation,
  - the "match not found" short-circuit (pipeline never invoked),
  - the pipeline handoff with the source-validated names + source context,
  - best-effort source data collection (never fabricated).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402
import agents.football.source_match as source_match  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _cfg():
    return {
        "analyse": {"budget_seconds": 300.0},
        "data_sources": {"livescore": {"enabled": True, "max_pages": 1}},
        "cache_ttl_seconds": {"odds": 900},
        "outlier_threshold_pct": 5,
        "prediction_log": {"enabled": False},
    }


def _cache():
    return type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})()


# --------------------------------------------------------------------------
# bot parsing + routing
# --------------------------------------------------------------------------


def test_parse_match_query_laliga():
    assert bot._parse_match_query("laliga barcelona vs real madrid") == (
        "LaLiga", "barcelona", "real madrid",
    )


def test_parse_match_query_multi_word_league():
    assert bot._parse_match_query("liga portugal Santa Clara vs Nacional") == (
        "Primeira Liga", "Santa Clara", "Nacional",
    )


def test_parse_match_query_requires_league_and_separator():
    assert bot._parse_match_query("barcelona vs real madrid") is None  # no league
    assert bot._parse_match_query("laliga vs real madrid") is None  # empty home
    assert bot._parse_match_query("laliga barcelona real madrid") is None  # no ' vs '
    assert bot._parse_match_query("") is None


def test_handlers_table_covers_source_commands():
    for cmd in ("livescore", "flashscore"):
        assert cmd in bot._HANDLERS
        assert callable(getattr(bot, bot._HANDLERS[cmd], None))


def test_intent_action_livescore():
    action = bot._intent_to_action(
        "livescore", {"league": "laliga", "home": "barcelona", "away": "real madrid"}
    )
    assert action == ("_handle_source_match", ["livescore", "laliga barcelona vs real madrid"])


def test_intent_action_flashscore_missing_fields():
    assert bot._intent_to_action("flashscore", {"home": "a", "away": "b"}) is None


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
    def __init__(self, sent: _Sent, content: str = "") -> None:
        self.sent = sent
        self.content = content
        self.channel = _FakeChannel(sent)


def test_handle_source_match_spawns_livescore_runner(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        return {
            "render": {
                "title": "🔬 MATCH SIGNAL",
                "body": "**Barcelona vs Real Madrid**\nanalysed",
                "footer": " ",
            }
        }

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_source_match(_FakeMsg(sent), "livescore", "laliga barcelona vs real madrid"))
    assert calls == [
        ["livescore", "--league", "LaLiga", "--home", "barcelona", "--away", "real madrid"]
    ]
    assert "Barcelona vs Real Madrid" in "\n".join(sent.messages)


def test_handle_source_match_spawns_flashscore_runner(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        return {"render": {"title": "t", "body": "ok", "footer": " "}}

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_source_match(_FakeMsg(sent), "flashscore", "ucl bodo vs union sg"))
    assert calls == [
        ["flashscore", "--league", "UCL", "--home", "bodo", "--away", "union sg"]
    ]


def test_handle_source_match_bad_format_sends_help(monkeypatch):
    sent = _Sent()

    async def fake_invoke(args):
        raise AssertionError("runner must not be invoked for a malformed query")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_source_match(_FakeMsg(sent), "livescore", "barcelona vs real madrid"))
    joined = "\n".join(sent.messages)
    assert "Format: `!livescore <liga> <home> vs <away>`" in joined


def test_handle_routes_livescore_prefix(monkeypatch):
    captured: list[tuple] = []

    async def spy(message, source, rest):
        captured.append((source, rest))

    monkeypatch.setattr(bot, "_is_authorized", lambda message: True)
    monkeypatch.setattr(bot, "_handle_source_match", spy)
    _run(bot._handle(_FakeMsg(_Sent(), "!livescore laliga barcelona vs real madrid")))
    assert captured == [("livescore", "laliga barcelona vs real madrid")]


def test_handle_routes_flashscore_prefix(monkeypatch):
    captured: list[tuple] = []

    async def spy(message, source, rest):
        captured.append((source, rest))

    monkeypatch.setattr(bot, "_is_authorized", lambda message: True)
    monkeypatch.setattr(bot, "_handle_source_match", spy)
    _run(bot._handle(_FakeMsg(_Sent(), "!flashscore laliga barcelona vs real madrid")))
    assert captured == [("flashscore", "laliga barcelona vs real madrid")]


# --------------------------------------------------------------------------
# LiveScore search (fake date feeds, deterministic WIB dates)
# --------------------------------------------------------------------------


def _ls_payload(stage_comp: str, home: str, away: str, kickoff: str, eid: int = 12345):
    """One-stage LiveScore /date/soccer payload with a single event."""
    esd = kickoff.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    return {
        "Stages": [
            {
                "CompN": stage_comp,
                "Cnm": "Spain",
                "Ccd": "spain",
                "Scd": "laliga",
                "Events": [
                    {
                        "Eid": eid,
                        "T1": [{"Nm": home, "ID": 1001}],
                        "T2": [{"Nm": away, "ID": 1002}],
                        "Esd": esd,
                        "Eps": "NS",
                        "Tr1": None,
                        "Tr2": None,
                    }
                ],
            }
        ]
    }


class _FakeLiveClient:
    """fetch_soccer_date returns the fixture only for one UTC feed date."""

    def __init__(self, feeds: dict[str, dict]):
        self.feeds = feeds

    async def fetch_soccer_date(self, date8: str, page: int):
        return self.feeds.get(date8) if page == 0 else None


def _patch_live_client(monkeypatch, feeds: dict[str, dict]):
    """Make _search_livescore use a fake client instead of the network."""
    monkeypatch.setattr(source_match, "LiveScoreClient", lambda base_url=None: _FakeLiveClient(feeds))


def test_fetch_finished_livescore_results(monkeypatch):
    """Settle source: LiveScore's daily feed yields finished results with
    scores, skipping live/scheduled events and scoreless rows."""
    esd = "20260815183000"
    payload = {
        "Stages": [
            {
                "CompN": "Premier League", "Cnm": "England",
                "Ccd": "england", "Scd": "premierleague",
                "Events": [
                    {"Eid": 1, "T1": [{"Nm": "Arsenal", "ID": 1}], "T2": [{"Nm": "Chelsea", "ID": 2}],
                     "Esd": esd, "Eps": "FT", "Tr1": 2, "Tr2": 1},
                    {"Eid": 2, "T1": [{"Nm": "Liverpool", "ID": 3}], "T2": [{"Nm": "Everton", "ID": 4}],
                     "Esd": esd, "Eps": "FT", "Tr1": 1, "Tr2": 1},
                    # live match -> must be skipped (no final score yet)
                    {"Eid": 3, "T1": [{"Nm": "Man City", "ID": 5}], "T2": [{"Nm": "Man Utd", "ID": 6}],
                     "Esd": esd, "Eps": "1H", "Tr1": 1, "Tr2": 0},
                    # scheduled -> skipped
                    {"Eid": 4, "T1": [{"Nm": "Spurs", "ID": 7}], "T2": [{"Nm": "West Ham", "ID": 8}],
                     "Esd": esd, "Eps": "NS", "Tr1": None, "Tr2": None},
                ],
            }
        ]
    }
    _patch_live_client(monkeypatch, {"20260815": payload})
    res = _run(source_match.fetch_finished_livescore_results(
        _cfg(), _cache(), "2026-08-15"))
    assert len(res) == 2
    first = res[0]
    assert first["home"] == "Arsenal" and first["away"] == "Chelsea"
    assert first["home_goals"] == 2 and first["away_goals"] == 1
    assert first["competition"] == "Premier League"
    assert all(r["home_goals"] is not None and r["away_goals"] is not None for r in res)


def test_fetch_finished_livescore_results_disabled(monkeypatch):
    cfg = _cfg()
    cfg["data_sources"]["livescore"]["enabled"] = False
    res = _run(source_match.fetch_finished_livescore_results(cfg, _cache(), "2026-08-15"))
    assert res == []


def test_search_livescore_found_today(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # A fixture ON today WIB: kickoff 2026-08-14 21:00 UTC == 2026-08-15 04:00 WIB.
    feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-14T21:00:00Z")
    _patch_live_client(monkeypatch, {"20260814": feed})  # today's first UTC date
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["home"] == "Barcelona" and found["away"] == "Getafe"
    assert found["date"] == "2026-08-15"
    assert found["kickoff"] == "2026-08-14T21:00:00Z"
    assert found["source"] == "livescore"


def test_search_livescore_found_tomorrow(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 2026-08-15 20:00 UTC == 2026-08-16 03:00 WIB -> tomorrow WIB.
    feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-15T20:00:00Z")
    _patch_live_client(monkeypatch, {"20260816": feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["date"] == "2026-08-16"


def test_search_livescore_not_found_outside_window(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 2026-08-20 20:00 UTC == 2026-08-21 WIB -> outside today/tomorrow.
    feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-20T20:00:00Z")
    _patch_live_client(monkeypatch, {"20260821": feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is None


def test_search_livescore_wrong_competition_rejected(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # Same two teams, but in a cup (Copa del Rey) -> NOT the requested league.
    feed = _ls_payload("Copa del Rey", "Barcelona", "Getafe", "2026-08-15T20:00:00Z")
    _patch_live_client(monkeypatch, {"20260816": feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is None


def test_search_livescore_prefers_exact_names(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # Both fixtures in LaLiga on the same window: an exact "Barcelona" fixture
    # and a containment hit ("Barcelona B"). The exact one must win.
    exact = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-15T20:00:00Z", eid=1)
    loose = _ls_payload("LaLiga", "Barcelona B", "Getafe", "2026-08-15T20:30:00Z", eid=2)
    payload = {
        "Stages": [
            {
                "CompN": "LaLiga",
                "Cnm": "Spain",
                "Ccd": "spain",
                "Scd": "laliga",
                "Events": [
                    {
                        "Eid": 1,
                        "T1": [{"Nm": "Barcelona", "ID": 1001}],
                        "T2": [{"Nm": "Getafe", "ID": 1002}],
                        "Esd": "20260815200000",
                        "Eps": "NS",
                        "Tr1": None,
                        "Tr2": None,
                    },
                    {
                        "Eid": 2,
                        "T1": [{"Nm": "Barcelona B", "ID": 1003}],
                        "T2": [{"Nm": "Getafe", "ID": 1004}],
                        "Esd": "20260815203000",
                        "Eps": "NS",
                        "Tr1": None,
                        "Tr2": None,
                    },
                ],
            }
        ]
    }
    _patch_live_client(monkeypatch, {"20260816": payload})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["source_id"] == "1"  # exact "Barcelona" fixture wins


def test_search_livescore_today_preferred_over_tomorrow(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    today_feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-14T21:00:00Z", eid=1)
    tomorrow_feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-15T20:00:00Z", eid=2)
    _patch_live_client(monkeypatch, {"20260814": today_feed, "20260816": tomorrow_feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["date"] == "2026-08-15"  # today wins the tie


def test_collect_livescore_data_best_effort():
    class Client:
        async def fetch_lineups(self, eid):
            return {"Lu": [{"Tnb": 1, "Fo": "4-3-3", "Ps": [{"Fn": "Leo", "Ln": "Messi"}], "IS": []}]}

        async def fetch_h2h(self, eid):
            return {"H2H": [{"T1": [{"Nm": "Barcelona", "ID": 1001}], "T2": [{"Nm": "Getafe", "ID": 1002}],
                             "Tr1": 2, "Tr2": 0, "Esd": "20260501170000", "Eps": "FT"}]}

        async def fetch_form(self, eid):
            return {"T1": [{"ID": 1001, "EL": []}], "T2": [{"ID": 1002, "EL": []}]}

        async def fetch_statistics(self, eid):
            raise RuntimeError("stats endpoint down")  # must degrade, not crash

    found = {"source_id": "12345", "home": "Barcelona", "away": "Getafe", "home_id": "1001"}
    out = _run(source_match._collect_livescore_data(Client(), found))
    assert out["lineups"]["home"]["players"][0]["name"] == "Leo Messi"
    assert out["h2h"]["wins"] == 1 and out["h2h"]["losses"] == 0
    assert "statistics" not in out  # failed endpoint omitted, never fabricated


# --------------------------------------------------------------------------
# Flashscore search + collection
# --------------------------------------------------------------------------


def test_search_flashscore_uses_resolve_match():
    fc = AsyncMock()
    fc.available = True
    fc.resolve_match = AsyncMock(
        return_value={
            "home": {"slug": "barcelona", "id": "AAAA1111", "name": "Barcelona"},
            "away": {"slug": "real-madrid", "id": "BBBB2222", "name": "Real Madrid"},
            "match_url": "https://www.flashscore.com/match/football/x/",
            "date_text": "21:00",
            "source": "flashscore",
        }
    )
    stats = type("S", (), {"fc": fc})()
    found = _run(source_match._search_flashscore(stats, "LaLiga", "barcelona", "real madrid"))
    assert found is not None
    assert found["home"] == "Barcelona" and found["away"] == "Real Madrid"
    assert found["source"] == "flashscore"
    assert found["date"] is not None  # bare kickoff time -> today
    fc.resolve_match.assert_awaited_once_with("LaLiga", "barcelona", "real madrid")


def test_search_flashscore_none_when_no_match():
    fc = AsyncMock()
    fc.available = True
    fc.resolve_match = AsyncMock(return_value=None)
    stats = type("S", (), {"fc": fc})()
    assert _run(source_match._search_flashscore(stats, "LaLiga", "a", "b")) is None


def test_search_flashscore_disabled_client_returns_none():
    stats = type("S", (), {"fc": None})()
    assert _run(source_match._search_flashscore(stats, "LaLiga", "a", "b")) is None


def test_collect_flashscore_data_best_effort():
    fc = AsyncMock()
    fc.fetch_match_statistics = AsyncMock(return_value={"xg_home": 1.2})
    fc.fetch_match_lineups = AsyncMock(return_value=None)  # not announced
    fc.fetch_match_h2h = AsyncMock(return_value={"wins": 2, "draws": 1, "losses": 0})
    fc.fetch_match_info = AsyncMock(side_effect=RuntimeError("page down"))
    fc.fetch_team_form = AsyncMock(return_value={"sequence": "W-W-D", "gf_avg": 1.5})
    stats = type("S", (), {"fc": fc})()
    found = {
        "home": "Barcelona", "away": "Getafe",
        "home_slug": "barcelona", "home_id": "1",
        "away_slug": "getafe", "away_id": "2",
        "match_url": "https://www.flashscore.com/match/football/x/",
    }
    out = _run(source_match._collect_flashscore_data(stats, found))
    assert out["statistics"] == {"xg_home": 1.2}
    assert out["h2h"]["wins"] == 2
    assert "lineups" not in out and "match_info" not in out
    assert out["form"]["home"]["sequence"] == "W-W-D"
    assert out["form"]["away"]["sequence"] == "W-W-D"  # same fake for both sides


# --------------------------------------------------------------------------
# find_source_match: source search -> pipeline handoff / not-found short-circuit
# --------------------------------------------------------------------------


def test_find_source_match_livescore_found_hands_to_pipeline(monkeypatch):
    found = {
        "source": "livescore", "home": "Barcelona", "away": "Getafe",
        "kickoff": "2026-08-14T21:00:00Z", "date": "2026-08-15",
        "competition": "LaLiga", "status": "scheduled", "source_id": "1",
    }
    monkeypatch.setattr(source_match, "_search_livescore", AsyncMock(return_value=found))
    monkeypatch.setattr(source_match, "_collect_livescore_data", AsyncMock(return_value={"h2h": {}}))
    pipeline = AsyncMock(return_value={"home": "Barcelona", "away": "Getafe", "odds": {"has_odds": True}})
    monkeypatch.setattr(source_match, "find_specific_match", pipeline)

    result = _run(source_match.find_source_match(
        source="livescore", league_query="laliga", home_query="barcelona",
        away_query="getafe", cfg=_cfg(), odds=AsyncMock(), stats=AsyncMock(),
        cache=_cache(),
    ))
    assert result["odds"]["has_odds"] is True
    assert pipeline.await_count == 1
    kwargs = pipeline.await_args.kwargs
    assert kwargs["home_query"] == "Barcelona"  # source-validated name
    assert kwargs["away_query"] == "Getafe"
    assert kwargs["source_match"]["source"] == "livescore"
    assert kwargs["source_match"]["date"] == "2026-08-15"
    assert kwargs["source_match"]["h2h"] == {}


def test_find_source_match_flashscore_found_hands_to_pipeline(monkeypatch):
    found = {
        "source": "flashscore", "home": "Barcelona", "away": "Real Madrid",
        "match_url": "https://www.flashscore.com/match/football/x/",
        "date": "2026-08-15",
    }
    monkeypatch.setattr(source_match, "_search_flashscore", AsyncMock(return_value=found))
    monkeypatch.setattr(source_match, "_collect_flashscore_data", AsyncMock(return_value={}))
    pipeline = AsyncMock(return_value={"home": "Barcelona", "away": "Real Madrid"})
    monkeypatch.setattr(source_match, "find_specific_match", pipeline)

    result = _run(source_match.find_source_match(
        source="flashscore", league_query="laliga", home_query="barcelona",
        away_query="real madrid", cfg=_cfg(), odds=AsyncMock(), stats=AsyncMock(),
        cache=_cache(),
    ))
    assert result["home"] == "Barcelona"
    assert pipeline.await_args.kwargs["source_match"]["source"] == "flashscore"
    # the pipeline ran (NoGoal odds + prediction + output all live there)
    assert pipeline.await_count == 1


def test_find_source_match_not_found_skips_pipeline(monkeypatch):
    monkeypatch.setattr(source_match, "_search_livescore", AsyncMock(return_value=None))
    pipeline = AsyncMock()
    monkeypatch.setattr(source_match, "find_specific_match", pipeline)

    result = _run(source_match.find_source_match(
        source="livescore", league_query="laliga", home_query="barcelona",
        away_query="real madrid", cfg=_cfg(), odds=AsyncMock(), stats=AsyncMock(),
        cache=_cache(),
    ))
    assert pipeline.await_count == 0  # prediction engine never runs
    assert "tidak ditemukan" in (result.get("error") or "")
    assert "LiveScore" in result["error"]
    assert result["league"] == "La Liga"


def test_find_source_match_flashscore_not_found_skips_pipeline(monkeypatch):
    monkeypatch.setattr(source_match, "_search_flashscore", AsyncMock(return_value=None))
    pipeline = AsyncMock()
    monkeypatch.setattr(source_match, "find_specific_match", pipeline)

    result = _run(source_match.find_source_match(
        source="flashscore", league_query="laliga", home_query="barcelona",
        away_query="real madrid", cfg=_cfg(), odds=AsyncMock(), stats=AsyncMock(),
        cache=_cache(),
    ))
    assert pipeline.await_count == 0
    assert "Flashscore" in (result.get("error") or "")


def test_find_source_match_unknown_league():
    result = _run(source_match.find_source_match(
        source="livescore", league_query="xyz", home_query="a", away_query="b",
        cfg=_cfg(), odds=AsyncMock(), stats=AsyncMock(), cache=_cache(),
    ))
    assert "tidak dikenal" in (result.get("error") or "")


# --------------------------------------------------------------------------
# find_specific_match integration: source kickoff fallback + nowgoal date
# --------------------------------------------------------------------------


def _make_stats():
    ms = AsyncMock()
    ms.search_teams_pair = AsyncMock(
        return_value=(
            {"id": 1, "name": "Arsenal", "provider": "flashscore"},
            {"id": 2, "name": "Chelsea", "provider": "flashscore"},
        )
    )
    ms.fetch_upcoming_fixture = AsyncMock(return_value={})  # no kickoff from fixture
    ms.fetch_team_form = AsyncMock(return_value={"sequence": "W-W-D", "gf_avg": 1.5, "ga_avg": 0.8})
    ms.fetch_h2h = AsyncMock(return_value={"wins": 1, "draws": 0, "losses": 2})
    ms.fetch_team_xg_history = AsyncMock(return_value=None)
    ms.fetch_flashscore_stats_for_match = AsyncMock(return_value=None)
    ms.fetch_flashscore_lineups_for_match = AsyncMock(return_value=None)
    ms.fetch_flashscore_event_context = AsyncMock(return_value=None)
    ms.fetch_league_standings = AsyncMock(return_value=None)
    ms.fetch_flashscore_match_info = AsyncMock(return_value=None)
    ms.fd = type("FD", (), {"rate_limit_warning": False})()
    return ms


def test_find_specific_match_uses_source_kickoff_and_nowgoal_date(monkeypatch):
    """The source-validated kickoff feeds the pipeline: it narrows the NowGoal
    date scan, supplies the kickoff when nothing else has it, and rides along
    in the result for provenance. No odds -> has_odds stays False (never
    fabricated)."""
    import agents.football.analyse as analyse

    nowgoal = AsyncMock()
    nowgoal.match_odds = AsyncMock(return_value=None)

    source_match_data = {
        "source": "livescore",
        "home": "Arsenal", "away": "Chelsea",
        "kickoff": "2030-01-01T10:00:00Z",  # future -> not finished
        "date": "2030-01-01",
        "competition": "EPL",
        "source_id": "99",
    }
    monkeypatch.setattr(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(analyse, "resolve_league_scored",
                        lambda q: ("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"}))
    monkeypatch.setattr(analyse, "_season_now", lambda: 2026)
    result = _run(analyse.find_specific_match(
        league_query="EPL", home_query="Arsenal", away_query="Chelsea",
        cfg=_cfg(), odds=AsyncMock(), stats=_make_stats(), cache=_cache(),
        oddspapi=None, nowgoal=nowgoal, source_match=source_match_data,
    ))

    # kickoff came from the source (fixture + odds payload both empty)
    assert result["kickoff"] == "2030-01-01T10:00:00Z"
    assert result["match_finished"] is False
    # nowgoal scanned exactly the source-validated date (WIB)
    nowgoal.match_odds.assert_awaited_once_with("Arsenal", "Chelsea", "2030-01-01")
    # no odds found -> reported honestly, nothing fabricated
    assert result["odds"]["has_odds"] is False
    # provenance rides along
    assert result["match_source"] == "livescore"
    assert result["source_match"]["source_id"] == "99"


def test_find_specific_match_plain_path_preserves_nowgoal_scan(monkeypatch):
    """Without source_match (plain `analisa match`), the NowGoal scan keeps its
    existing default (no date -> today+tomorrow) and the result carries no
    source provenance."""
    import agents.football.analyse as analyse

    nowgoal = AsyncMock()
    nowgoal.match_odds = AsyncMock(return_value=None)

    monkeypatch.setattr(analyse, "find_match_odds_payload", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(analyse, "resolve_league_scored",
                        lambda q: ("EPL", {"display": "EPL", "odds_api_key": "soccer_epl"}))
    monkeypatch.setattr(analyse, "_season_now", lambda: 2026)
    result = _run(analyse.find_specific_match(
        league_query="EPL", home_query="Arsenal", away_query="Chelsea",
        cfg=_cfg(), odds=AsyncMock(), stats=_make_stats(), cache=_cache(),
        oddspapi=None, nowgoal=nowgoal,
    ))

    nowgoal.match_odds.assert_awaited_once_with("Arsenal", "Chelsea", None)
    assert result.get("match_source") is None
    assert result.get("source_match") is None


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        mp = __import__("pytest").MonkeyPatch()
        try:
            try:
                fn(mp)
            except TypeError:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        finally:
            mp.undo()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
