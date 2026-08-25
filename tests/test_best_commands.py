"""Tests for `!best <liga>` and `!bestgoalmatch` commands.

Covers:
- find_best_matches: fixture screening (upcoming only), per-match engine run,
  ranking by decision quality, winner payload (analyse-compatible shape).
- _goal_profile: expected total goals + over probabilities from form data.
- find_best_goal_matches: league-average ranking + top goal-friendly pick.
- format_best / format_best_goal: rendering (shortlist, winner, footer).
- bot dispatch: _parse_command + _handle wiring for best/bestgoalmatch.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402
from agents.football import format as fmt  # noqa: E402
from agents.football.best_match import (  # noqa: E402
    _goal_profile,
    find_best_goal_matches,
    find_best_matches,
)


def _cfg() -> dict:
    return {
        "leagues": ["EPL", "UCL"],
        "cache_ttl_seconds": {"fixtures": 300, "odds": 300},
        "outlier_threshold_pct": 10.0,
        # Gate `!best` (conf >= MEDIUM + non-veto) OFF untuk tes mekanika
        # ranking; perilaku gate sendiri diuji di test_find_best_matches_gate_*.
        "models": {"decision": {"best_gate_enabled": False}},
    }


def _form(gf: float, ga: float, seq: str = "W-D-L-W-W") -> dict:
    return {
        "sequence": seq,
        "gf_avg": gf,
        "ga_avg": ga,
        "home": {"w": 3, "d": 1, "l": 1},
        "away": {"w": 2, "d": 1, "l": 2},
        "recent_goals": [(2, 0), (1, 1), (0, 2), (3, 1), (2, 1)],
    }


def _odds_payload(home: str, away: str) -> list[dict]:
    return [{
        "home_team": home,
        "away_team": away,
        "commence_time": "2026-08-12T19:00:00Z",
        "bookmakers": [{
            "title": "Bookie",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home, "price": 1.60},
                    {"name": "Draw", "price": 4.20},
                    {"name": away, "price": 5.50},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 2.5, "price": 1.80},
                    {"name": "Under", "point": 2.5, "price": 1.95},
                    {"name": "Over", "point": 3.5, "price": 2.60},
                    {"name": "Under", "point": 3.5, "price": 1.50},
                ]},
                {"key": "btts", "outcomes": [
                    {"name": "Yes", "price": 1.70},
                    {"name": "No", "price": 2.10},
                ]},
            ],
        }],
    }]


class _FakeStats:
    def __init__(self, fixtures: list[dict], forms: dict):
        self.fixtures = fixtures
        self.forms = forms
        self.fetch_fixtures_for_date = AsyncMock(return_value=fixtures)

        async def _form(team_id, meta, limit=10):
            return self.forms.get(team_id)

        self.fetch_team_form = _form

        class _FD:
            rate_limit_warning = False

        self.fd = _FD()


class _FakeOdds:
    def __init__(self, payload: list[dict]):
        self.payload = payload
        self.fetch_odds = AsyncMock(return_value=payload)
        self.last_remaining = 500
        self.quota_blocked = False

    class _quota:
        quota_warning = False


def _fixture(mid: int, home: str, away: str, status: str = "SCHEDULED", date: str = "2026-08-12T19:00:00Z") -> dict:
    return {
        "id": mid,
        "home": {"id": mid * 10 + 1, "name": home},
        "away": {"id": mid * 10 + 2, "name": away},
        "date": date,
        "status": status,
        "source": "football_data",
    }


def test_find_best_matches_ranks_and_builds_winner(tmp_path):
    fixtures = [
        _fixture(1, "Arsenal", "Chelsea"),
        _fixture(2, "Liverpool", "Man City"),
        _fixture(3, "Old Team", "Done Team", status="FINISHED"),  # filtered out
    ]
    forms = {
        11: _form(2.0, 0.8), 12: _form(1.5, 1.1),
        21: _form(2.2, 1.0), 22: _form(1.8, 0.9),
    }
    odds_payload = _odds_payload("Arsenal", "Chelsea") + _odds_payload("Liverpool", "Man City")
    stats = _FakeStats(fixtures, forms)
    odds = _FakeOdds(odds_payload)

    payload = asyncio.run(find_best_matches(
        league_query="epl", cfg=_cfg(), odds=odds, stats=stats, cache=None, date="2026-08-12",
    ))

    assert payload["league"] == "EPL"
    assert not payload.get("error")
    # finished fixture excluded
    cands = payload["candidates"]
    assert len(cands) == 2
    assert all(c["home"] != "Old Team" for c in cands)
    # candidates carry decision info
    for c in cands:
        assert c["decision_type"] in ("STRONG", "GOOD", "LEAN", "NO BET", "NO CLEAR DECISION")
    # ranking: urutan mengikuti _rank_key -- prioritas tipe keputusan dulu,
    # lalu decision_score dalam tipe yang sama (F1 anchor plan v3 membuat dua
    # fixture ini berbeda tipe: NO BET vs NO CLEAR DECISION).
    import agents.football.best_match as bm

    keys = [
        (bm._DECISION_PRIORITY.get(c["decision_type"], 0), c["decision_score"])
        for c in cands
    ]
    assert keys == sorted(keys, key=lambda kv: (kv[0], kv[1]), reverse=True)
    # winner payload is analyse-compatible
    winner = payload["winner"]
    assert winner["home"] == cands[0]["home"]
    assert winner["league"] == "EPL"
    assert winner["odds"]["has_odds"] is True
    assert winner["decision"]["decision_type"] == cands[0]["decision_type"]


def test_find_best_matches_unknown_league():
    stats = _FakeStats([], {})
    odds = _FakeOdds([])
    payload = asyncio.run(find_best_matches(
        league_query="zzzzqqqq", cfg=_cfg(), odds=odds, stats=stats, cache=None,
    ))
    assert "error" in payload
    assert "tidak dikenal" in payload["error"]


def test_find_best_matches_no_fixtures(tmp_path):
    stats = _FakeStats([], {})
    odds = _FakeOdds([])
    payload = asyncio.run(find_best_matches(
        league_query="epl", cfg=_cfg(), odds=odds, stats=stats, cache=None, date="2026-08-12",
    ))
    assert "error" in payload
    assert "Tidak ada match" in payload["error"]


def test_goal_profile_computes_expected_goals():
    p = _goal_profile(_form(2.2, 1.0), _form(1.8, 0.9), {})
    assert p is not None
    # (2.2+0.9)/2 + (1.8+1.0)/2 = 1.55 + 1.40 = 2.95
    assert abs(p["expected_total"] - 2.95) < 0.01
    assert 0 < p["over_2_5"] < 1
    assert p["over_3_5"] < p["over_2_5"]
    assert p["odds_over_2_5"] is None  # no market totals passed


def test_goal_profile_includes_market_odds():
    totals = {"Over 2.5": {"odds": 1.80}, "Over 3.5": {"odds": 2.60}}
    p = _goal_profile(_form(2.2, 1.0), _form(1.8, 0.9), totals)
    assert p["odds_over_2_5"] == 1.80
    assert p["odds_over_3_5"] == 2.60


def test_goal_profile_missing_data_returns_none():
    assert _goal_profile(None, None, {}) is None
    assert _goal_profile(_form(2.0, 0.8), {"sequence": "W"}, {}) is None


def test_goal_profile_market_fallback_pair_25_only():
    """The Odds API usually exposes only the 2.5 line (no 3.5). The market
    fallback must work from the Over/Under 2.5 pair alone, e.g. matchday 1 of
    a new season when no form history exists yet."""
    totals = {"Over 2.5": {"odds": 1.80}, "Under 2.5": {"odds": 1.95}}
    p = _goal_profile(None, None, totals)
    assert p is not None
    assert p["source"] == "market"
    # implied P(Over 2.5) ~ 0.52 -> expected total clearly above 2.5
    assert p["expected_total"] > 2.4
    assert p["odds_over_2_5"] == 1.80
    assert p["odds_over_3_5"] is None  # line not offered by the market
    assert 0 < p["over_2_5"] < 1
    assert 0 < p["over_3_5"] < p["over_2_5"] < 1


def test_find_best_goal_matches_ranks_by_expected_goals(tmp_path):
    fixtures = [
        _fixture(1, "Goalfest A", "Goalfest B"),
        _fixture(2, "Defense C", "Defense D"),
    ]
    forms = {
        11: _form(2.5, 1.2), 12: _form(2.3, 1.0),   # high scoring pair
        21: _form(1.0, 0.8), 22: _form(1.1, 0.7),   # low scoring pair
    }
    odds_payload = _odds_payload("Goalfest A", "Goalfest B") + _odds_payload("Defense C", "Defense D")
    stats = _FakeStats(fixtures, forms)
    odds = _FakeOdds(odds_payload)

    payload = asyncio.run(find_best_goal_matches(
        cfg=_cfg(), odds=odds, stats=stats, cache=None, date="2026-08-12",
    ))
    assert not payload.get("error")
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["home"] == "Goalfest A"  # highest expected total first
    winner = payload["winner"]
    assert winner["home"] == "Goalfest A"
    assert winner["goal"]["expected_total"] > payload["candidates"][1]["goal"]["expected_total"]
    assert "EPL" in dict(payload["league_avg"])


def test_find_best_matches_nowgoal_fallback_when_football_data_empty(tmp_path):
    """football-data empty -> nowgoal schedule supplies the fixtures for `!best`."""

    class _FakeNowgoal:
        async def fetch_schedule(self, date):
            return [{
                "match_id": "9001", "home": "Arsenal FC", "away": "Chelsea FC",
                "home_id": "101", "away_id": "102",
                # kickoff in the future so _is_upcoming keeps the row
                "kickoff": "2099-01-01T19:00:00Z", "status": "1",
                "league_id": "36", "league_name": "Premier League", "source": "nowgoal",
            }]

    stats = _FakeStats([], {})
    odds = _FakeOdds([])

    payload = asyncio.run(find_best_matches(
        league_query="epl", cfg=_cfg(), odds=odds, stats=stats, cache=None,
        date="2099-01-01", nowgoal=_FakeNowgoal(),
    ))
    assert not payload.get("error")
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["home"] == "Arsenal FC"
    assert payload["winner"]["home"] == "Arsenal FC"


def test_find_best_matches_nowgoal_not_called_when_fd_ok(tmp_path):
    """football-data answers -> nowgoal fallback never invoked (zero overhead)."""
    fixtures = [_fixture(1, "Arsenal", "Chelsea")]
    stats = _FakeStats(fixtures, {})
    odds = _FakeOdds([])
    called = {"n": 0}

    class _FakeNowgoal:
        async def fetch_schedule(self, date):
            called["n"] += 1
            return []

    payload = asyncio.run(find_best_matches(
        league_query="epl", cfg=_cfg(), odds=odds, stats=stats, cache=None,
        date="2026-08-12", nowgoal=_FakeNowgoal(),
    ))
    assert called["n"] == 0
    assert not payload.get("error")


def test_find_best_goal_matches_nowgoal_fallback_when_football_data_empty(tmp_path):
    """football-data empty -> nowgoal schedule supplies fixtures for `!bestgoalmatch`."""

    class _FakeNowgoal:
        async def fetch_schedule(self, date):
            return [{
                "match_id": "9001", "home": "Goalfest A", "away": "Goalfest B",
                "home_id": "101", "away_id": "102",
                # kickoff in the future so _is_upcoming keeps the row
                "kickoff": "2099-01-01T19:00:00Z", "status": "1",
                "league_id": "36", "league_name": "Premier League", "source": "nowgoal",
            }]

    stats = _FakeStats([], {
        "101": _form(2.5, 1.2), "102": _form(2.3, 1.0),  # high scoring pair
    })
    odds = _FakeOdds([])

    payload = asyncio.run(find_best_goal_matches(
        cfg=_cfg(), odds=odds, stats=stats, cache=None,
        league_query="epl", date="2099-01-01", nowgoal=_FakeNowgoal(),
    ))
    assert not payload.get("error")
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["home"] == "Goalfest A"


def test_format_best_renders_shortlist_and_winner():
    payload = {
        "league": "EPL",
        "date": "2026-08-12",
        "candidates": [
            {"home": "Arsenal", "away": "Chelsea", "kickoff": "2026-08-12T19:00:00Z",
             "signal": 80, "decision_type": "GOOD", "decision_score": 0.61},
            {"home": "Liverpool", "away": "Man City", "kickoff": "2026-08-12T21:00:00Z",
             "signal": 60, "decision_type": "LEAN", "decision_score": 0.42},
        ],
        "winner": {
            "league": "EPL", "league_key": "EPL", "generated_at": "2026-08-12T10:00:00Z",
            "home": "Arsenal", "away": "Chelsea", "kickoff": "2026-08-12T19:00:00Z",
            "prediction": None, "stats": {"home_form": "W", "away_form": "W"},
            "odds": {"has_odds": True, "consensus": {"home": 1.6, "draw": 4.2, "away": 5.5},
                     "bookmakers_count": 1, "totals": {}},
            "signal": 80, "picks": {}, "decision": {
                "decision_type": "GOOD",
                "final_decision": {"market": "1X2", "selection": "Home Win",
                                   "model_prob": 0.6, "market_odds": 1.6, "edge_pp": 3.0, "ev": -0.04},
                "most_likely": None, "explanation": "x", "reasons": [], "edge_warnings": [],
                "score_breakdown": {"top": {"score": 0.61}},
            }, "sources": [], "quota": {}, "similar_signal": None,
        },
        "quota": {"odds_api_remaining": 500},
    }
    out = fmt.format_best(payload)
    body = out["body"]
    assert "BEST MATCH" in out["title"]
    assert "2 match" in body
    assert "1. **Arsenal vs Chelsea**" in body
    assert "⭐" in body  # winner marked
    assert "PILIHAN TERBAIK" in body
    # winner card is the COMPACT summary (same style as `analisa`) — the
    # single best pick per the OUTPUT POLICY; this fake winner has no
    # model_probs, so no market is evaluable and the reply says so honestly
    assert "📊 Arsenal vs Chelsea — EPL" in body
    assert "Tidak ada market dengan data cukup" in body
    assert "Not a guarantee of outcome. Betting decisions are the user's own risk." in body
    assert "FINAL DECISION" not in body  # full section lives in render_full
    assert "odds quota: 500/500" in out["footer"]


def test_format_best_full_winner_for_copy_button():
    """compact_winner=False embeds the FULL winner analysis — the runner's
    render_full, served by the 📋 Copy button."""
    payload = {
        "league": "EPL",
        "date": "2026-08-12",
        "candidates": [
            {"home": "Arsenal", "away": "Chelsea", "kickoff": "2026-08-12T19:00:00Z",
             "signal": 80, "decision_type": "GOOD", "decision_score": 0.61},
        ],
        "winner": {
            "league": "EPL", "league_key": "EPL", "generated_at": "2026-08-12T10:00:00Z",
            "home": "Arsenal", "away": "Chelsea", "kickoff": "2026-08-12T19:00:00Z",
            "prediction": None, "stats": {"home_form": "W", "away_form": "W"},
            "odds": {"has_odds": True, "consensus": {"home": 1.6, "draw": 4.2, "away": 5.5},
                     "bookmakers_count": 1, "totals": {}},
            "signal": 80, "picks": {}, "decision": {
                "decision_type": "GOOD",
                "final_decision": {"market": "1X2", "selection": "Home Win",
                                   "model_prob": 0.6, "market_odds": 1.6, "edge_pp": 3.0, "ev": -0.04},
                "most_likely": None, "explanation": "x", "reasons": [], "edge_warnings": [],
                "score_breakdown": {"top": {"score": 0.61}},
            }, "sources": [], "quota": {}, "similar_signal": None,
        },
        "quota": {"odds_api_remaining": 500},
    }
    out = fmt.format_best(payload, compact_winner=False)
    assert "PILIHAN TERBAIK" in out["body"]
    assert "FINAL DECISION" in out["body"]  # full analysis embedded
    assert "GOOD" in out["body"]


def test_format_best_error():
    out = fmt.format_best({"error": "liga 'xyz' tidak dikenal"})
    assert "Error" in out["body"]
    assert "tidak dikenal" in out["body"]


def test_format_best_goal_renders_pick():
    payload = {
        "date": "2026-08-12",
        "league_avg": [("EPL", 3.05), ("UCL", 2.40)],
        "candidates": [
            {"home": "Goalfest A", "away": "Goalfest B", "kickoff": "2026-08-12T19:00:00Z",
             "league": "EPL", "has_odds": True,
             "goal": {"expected_total": 3.10, "over_2_5": 0.72, "over_3_5": 0.50,
                      "over_4_5": 0.25, "odds_over_2_5": 1.80, "odds_over_3_5": 2.60,
                      "odds_over_4_5": None}},
        ],
        "winner": {
            "league": "EPL", "home": "Goalfest A", "away": "Goalfest B",
            "kickoff": "2026-08-12T19:00:00Z", "has_odds": True, "bookmakers_count": 1,
            "goal": {"expected_total": 3.10, "over_2_5": 0.72, "over_3_5": 0.50,
                     "over_4_5": 0.25, "odds_over_2_5": 1.80, "odds_over_3_5": 2.60,
                     "odds_over_4_5": None},
        },
        "quota": {},
    }
    out = fmt.format_best_goal(payload)
    body = out["body"]
    assert "BEST GOAL MATCH" in out["title"]
    assert "EPL: 3.05 gol/match" in body
    assert "Goalfest A vs Goalfest B" in body
    assert "expected total: **3.10 gol**" in body
    assert "Over 2.5: 72%" in body
    assert "Over 3.5: 50%" in body
    assert "1.80" in body  # market odds shown
    assert "rekomendasi: fokus **Over 3.5**" in body  # over_3_5 prob 0.50 >= 0.5


def test_format_best_goal_error():
    out = fmt.format_best_goal({"error": "Tidak ada match hari ini"})
    assert "Error" in out["body"]


def test_parse_command_best():
    # `!football best <liga>` form goes through the standard command parser.
    assert bot._parse_command("!football best epl") == ["best", "epl"]
    assert bot._parse_command("!football best liga portugal") == ["best", "liga", "portugal"]
    assert bot._parse_command("!football bestgoalmatch") == ["bestgoalmatch"]
    assert bot._parse_command("!football bestgoalmatch ucl") == ["bestgoalmatch", "ucl"]
    # Bare `!best <liga>` is NOT a `!football` command; it is routed by
    # _handle's prefix branch, not _parse_command.
    assert bot._parse_command("!best epl") is None


def test_best_handlers_require_league_arg():
    class _Ch:
        def __init__(self):
            self.sent = []

        async def send(self, *a, **k):
            self.sent.append((a, k))

        def typing(self):
            class T:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

            return T()

    class _Msg:
        channel = _Ch()

        class author:
            id = "u"

    old = bot._invoke_runner
    bot._invoke_runner = None  # must NOT be reached for the no-arg path

    async def _boom(args):
        raise AssertionError("runner must not be called for empty args")

    bot._invoke_runner = _boom
    try:
        msg = _Msg()
        asyncio.run(bot._handle_best(msg, []))
        assert msg.channel.sent and "Format: `!best <liga>`" in msg.channel.sent[0][0][0]
    finally:
        bot._invoke_runner = old


def test_best_handlers_invoke_runner_with_league():
    calls = []

    async def fake_runner(args):
        calls.append(args)
        return {"render": {"title": "🏆 BEST MATCH — EPL", "body": "ok", "footer": " "}}

    class _Ch:
        def __init__(self):
            self.sent = []

        async def send(self, *a, **k):
            self.sent.append((a, k))

        def typing(self):
            class T:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

            return T()

    class _Msg:
        channel = _Ch()

        class author:
            id = "u"

    old = bot._invoke_runner
    bot._invoke_runner = fake_runner
    try:
        msg = _Msg()
        asyncio.run(bot._handle_best(msg, ["epl"]))
        assert calls == [["best", "--league", "EPL"]]
        # _post_result sends via keyword `content=...` -> record ((), kwargs)
        assert msg.channel.sent and msg.channel.sent[0][1]["content"].startswith("🏆")

        msg2 = _Msg()
        asyncio.run(bot._handle_best_goal(msg2, ["ucl"]))
        assert calls[-1] == ["bestgoalmatch", "--league", "UCL"]
    finally:
        bot._invoke_runner = old


def test_best_handlers_reject_unknown_league():
    class _Ch:
        def __init__(self):
            self.sent = []

        async def send(self, *a, **k):
            self.sent.append((a, k))

        def typing(self):
            class T:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

            return T()

    class _Msg:
        channel = _Ch()

        class author:
            id = "u"

    msg = _Msg()
    asyncio.run(bot._handle_best(msg, ["xyzabc"]))
    assert msg.channel.sent and "Liga tidak dikenali" in msg.channel.sent[0][0][0]


def test_bestgoalmatch_in_handler_dispatch_table():
    """`!bestgoalmatch` is a known handler for the LLM router mapping."""
    assert bot._HANDLERS["best"] == "_handle_best"
    assert bot._HANDLERS["bestgoalmatch"] == "_handle_best_goal"


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


# ---- Gerbang `!best` (keputusan 2026-08-23): conf >= MEDIUM + non-veto ----

from agents.football.best_match import _passes_best_gate  # noqa: E402


def _decision(tier: str, pick_status: str = "VALID") -> dict:
    return {
        "pick_specific_confidence": {"label": tier},
        "score_breakdown": {"top": {"pick_status": pick_status}},
    }


def test_best_gate_requires_medium_or_high():
    assert _passes_best_gate(_decision("HIGH")) is True
    assert _passes_best_gate(_decision("MEDIUM")) is True
    assert _passes_best_gate(_decision("LOW")) is False
    # label hilang (MARKET PRIOR / thin-data path) -> gugur
    assert _passes_best_gate({}) is False
    assert _passes_best_gate(None) is False


def test_best_gate_requires_valid_pick_status():
    assert _passes_best_gate(_decision("MEDIUM", "VALID")) is True
    for blocked in ("INSUFFICIENT_DATA", "INSUFFICIENT_SAMPLE",
                    "AUDIT_REQUIRED", "REVIEW_REQUIRED", "NO VALUE"):
        assert _passes_best_gate(_decision("HIGH", blocked)) is False, blocked


def test_find_best_matches_gate_filters_and_reports_empty(tmp_path):
    """Gate ON + semua kandidat LOW/invalid -> error payload eksplisit."""
    fixtures = [_fixture(1, "Arsenal", "Chelsea")]
    forms = {11: _form(2.0, 0.8), 12: _form(1.5, 1.1)}
    stats = _FakeStats(fixtures, forms)
    odds = _FakeOdds(_odds_payload("Arsenal", "Chelsea"))
    cfg = {
        "leagues": ["EPL"],
        "cache_ttl_seconds": {"fixtures": 300, "odds": 300},
        "outlier_threshold_pct": 10.0,
        # Gate ON (default). Engine dipaksa selalu gagal gerbang via
        # monkeypatch di bawah -- deterministik tanpa bergantung skor asli.
        "models": {"decision": {}},
    }

    import agents.football.best_match as bm

    orig = bm._passes_best_gate
    bm._passes_best_gate = lambda d: False
    try:
        payload = asyncio.run(find_best_matches(
            league_query="epl", cfg=cfg, odds=odds, stats=stats, cache=None,
            date="2026-08-12",
        ))
    finally:
        bm._passes_best_gate = orig

    assert payload.get("error")
    assert "lolos gerbang" in payload["error"]
    assert "1 match dianalisa" in payload["error"]
    assert payload["winner"] is None and payload["candidates"] == []


def test_find_best_matches_gate_keeps_qualified_candidates(tmp_path):
    """Gate ON + satu kandidat lolos -> shortlist/winner hanya dari yang lolos."""
    fixtures = [
        _fixture(1, "Arsenal", "Chelsea"),
        _fixture(2, "Liverpool", "Man City"),
    ]
    forms = {
        11: _form(2.0, 0.8), 12: _form(1.5, 1.1),
        21: _form(2.2, 1.0), 22: _form(1.8, 0.9),
    }
    stats = _FakeStats(fixtures, forms)
    odds = _FakeOdds(
        _odds_payload("Arsenal", "Chelsea") + _odds_payload("Liverpool", "Man City")
    )
    cfg = {
        "leagues": ["EPL"],
        "cache_ttl_seconds": {"fixtures": 300, "odds": 300},
        "outlier_threshold_pct": 10.0,
        "models": {"decision": {}},
    }

    import agents.football.best_match as bm
    from agents.football.best_match import find_best_matches as _fbm

    calls = {"n": 0}

    def fake_gate(d):
        calls["n"] += 1
        # hanya kandidat pertama (Arsenal-Chelsea) yang lolos
        return calls["n"] == 1

    orig = bm._passes_best_gate
    bm._passes_best_gate = fake_gate
    try:
        payload = asyncio.run(_fbm(
            league_query="epl", cfg=cfg, odds=odds, stats=stats, cache=None,
            date="2026-08-12",
        ))
    finally:
        bm._passes_best_gate = orig

    cands = payload["candidates"]
    assert len(cands) == 1
    assert cands[0]["home"] == "Arsenal"
    assert cands[0]["confidence_tier"]
    # F1 anchor (plan v3): status top-pick mengikuti bucket kalibrasi nyata;
    # yang diuji di sini mekanika gerbang, bukan nilai bucket spesifik.
    assert cands[0]["pick_status"] in (
        "VALID", "INSUFFICIENT_DATA", "INSUFFICIENT_SAMPLE",
        "AUDIT_REQUIRED", "REVIEW_REQUIRED", "NO VALUE",
    )
    assert payload["winner"]["home"] == "Arsenal"
