"""Phase 0 tests: CLV vs closing as the only KPI.

Covers 0.1 (fetch_closing_odds + settle_auto wiring), 0.2 (benchmark age
stamping), 0.3 (clv_gate min_bets + Wilson CI) and 0.4 (CLV segment report).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.clv_gate import gate_segment, wilson_interval
from agents.football.edge_benchmark import edge_benchmark_status
from agents.football.prediction_log import (
    _settled_records,
    append_snapshot,
    clv_segment_report,
    list_unsettled,
    segment_clv_stats,
    settle,
)
from agents.football.settler import settle_auto

MID = "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z"


def _snap(path, *, prob=None, odds=None, market="1X2", decision_type="GOOD"):
    append_snapshot(
        path,
        match_id=MID,
        league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob=prob if prob is not None else {"home": 0.55, "draw": 0.25, "away": 0.20},
        odds=odds if odds is not None else {"home": 1.8, "draw": 3.6, "away": 4.4},
        edge=None, confidence=None, signal=None, calibration=None,
        model_version=None, input_hash=None,
        best_pick={"selection": "Home Win", "market": market},
        sources=[], decision_type=decision_type,
    )


# ---- 0.1: fetch_closing_odds + settle wiring ----------------------------

def test_fetch_closing_odds_median_reduction(monkeypatch):
    """fetch_closing_odds reduces the mixodds ``l``-leg (t=1) bookmakers to
    a 1X2 median -- the real closing line, NOT the t=11 roddsList feed."""
    from agents.football.nowgoal import NowGoalClient

    payload = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "bookmakers": [
            {"title": "A", "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 1.70},
                    {"name": "Draw", "price": 3.80},
                    {"name": "Chelsea", "price": 5.00},
                ],
            }]},
            {"title": "B", "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Arsenal", "price": 1.90},
                    {"name": "Draw", "price": 4.20},
                    {"name": "Chelsea", "price": 5.40},
                ],
            }]},
            {"title": "C", "markets": [{"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.9, "point": 2.5},
            ]}]},
        ],
    }

    async def _fake_fetch_odds(fixture, closing=False):
        assert closing is False, "closing line = l-leg of t=1 mixodds, not t=11"
        return payload

    client = NowGoalClient.__new__(NowGoalClient)
    client.fetch_odds = _fake_fetch_odds  # type: ignore[method-assign]

    import asyncio

    closing = asyncio.run(client.fetch_closing_odds(
        {"match_id": "123", "home": "Arsenal", "away": "Chelsea"}
    ))
    # median of [1.70, 1.90] = 1.80; draw median [3.80, 4.20] = 4.00
    assert closing == {"home": 1.80, "draw": 4.00, "away": 5.20}
    # totals-only bookmaker contributes nothing to 1X2
    assert len(closing) == 3


def test_fetch_closing_odds_uses_l_leg_not_t11_roddslist():
    """Regression (2026-08-16, verified live): for settled matches the t=11
    ``roddsList`` feed serves result-embedded FINAL prices (winner ~1.01,
    losers 50-500) -- using them as closing would fabricate CLV. The closing
    line is the ``l`` (last pre-match) leg of the t=1 mixodds feed, which
    persists after settle."""
    from agents.football.nowgoal import NowGoalClient

    mixodds = [
        {"cid": 177, "euro": {
            "f": {"u": "2.0", "g": "3.6", "d": "3.8"},
            "l": {"u": "2.2", "g": "3.4", "d": "3.6"},
            "r": {"u": "1.01", "g": "51", "d": "101"},
        }, "ou": None, "ah": None},
        {"cid": 178, "euro": {
            "f": {"u": "1.9", "g": "3.7", "d": "4.0"},
            "l": {"u": "2.0", "g": "3.5", "d": "3.8"},
            "r": {"u": "1.0", "g": "61", "d": "151"},
        }, "ou": None, "ah": None},
    ]
    data = {"ErrCode": 0, "Data": {"mixodds": mixodds}}

    import asyncio

    async def runner():
        client = NowGoalClient(throttle_seconds=0.0)
        calls: list[int] = []

        async def fake_get(path, params):
            calls.append(params.get("t"))
            return data

        client._get = fake_get  # type: ignore[method-assign]
        closing = await client.fetch_closing_odds(
            {"match_id": "123", "home": "Arsenal", "away": "Chelsea"}
        )
        assert calls == [1]  # t=1 mixodds, never t=11 roddsList
        # l-leg medians: home [2.2, 2.0]=2.1, draw [3.4, 3.5]=3.45,
        # away [3.6, 3.8]=3.7 -- the r-leg (1.01/51/101) is never used.
        assert closing == {"home": 2.1, "draw": 3.45, "away": 3.7}

    asyncio.run(runner())


def test_settle_auto_attaches_closing_odds(tmp_path):
    """settle_auto with a closing_fetcher stores closing_odds on the row."""
    path = tmp_path / "p.jsonl"
    _snap(path)
    results = [{"home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 1}]
    out = settle_auto(
        path, date="2026-08-15", results=results,
        closing_fetcher=lambda r: (
            {"home": 1.9, "draw": 3.6, "away": 4.4}
            if r.get("home") == "Arsenal" else None
        ),
    )
    assert out["status"] == "auto"
    assert out["closing_attached"] == 1
    assert out["settled"][0]["closing_odds"] == {"home": 1.9, "draw": 3.6, "away": 4.4}
    assert list_unsettled(path) == []
    # stats now see a positive price CLV (close home 1.9 > prediction 1.8)
    stats = segment_clv_stats(path)
    key = "EPL|1X2|GOOD"
    assert stats[key]["n"] == 1
    assert stats[key]["price_clv_pct"] > 0
    assert stats[key]["n_positive_clv"] == 1


def test_settle_auto_closing_fetcher_failure_does_not_break_settle(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path)
    results = [{"home": "Arsenal", "away": "Chelsea", "home_goals": 1, "away_goals": 1}]

    def _boom(r):
        raise RuntimeError("network down")

    out = settle_auto(path, date="2026-08-15", results=results, closing_fetcher=_boom)
    assert out["status"] == "auto"
    assert out["closing_attached"] == 0
    assert len(out["settled"]) == 1  # settlement still written, CLV simply absent
    assert out["settled"][0]["closing_odds"] is None


# ---- 0.2: benchmark age stamping ----------------------------------------

def test_edge_benchmark_status_fresh_and_stale():
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(hours=2)).isoformat(timespec="seconds")
    stale = (now - timedelta(hours=48)).isoformat(timespec="seconds")
    assert edge_benchmark_status(fresh)["stale"] is False
    s = edge_benchmark_status(stale)
    assert s["stale"] is True
    assert s["age_hours"] == 48.0
    # explicit threshold override
    assert edge_benchmark_status(fresh, max_age_hours=1.0)["stale"] is True
    # no ts -> not stale, age unknown
    assert edge_benchmark_status(None)["age_hours"] is None


# ---- 0.3: Wilson CI in the CLV gate -------------------------------------

def test_wilson_interval_formula():
    # p_hat=1.0, n=30 -> centre (30 + 1.96^2/2)/(30 + 1.96^2), half-w narrow
    centre, half = wilson_interval(30, 30)
    assert 0.90 <= centre <= 1.0
    assert 0.0 < half <= 0.15
    assert wilson_interval(3, 30)[1] > wilson_interval(30, 30)[1]
    assert wilson_interval(0, 0) is None


def test_gate_requires_wilson_ci_halfwidth():
    # n=30 meets min_bets; CLV + ROI positive; but 25/30 positive CLV has a
    # half-width ~0.14 > 0.05 -> gate stays closed on insufficient evidence.
    stats = {"EPL|1X2|GOOD": {"n": 30, "price_clv_pct": 3.0, "roi": 0.05, "n_positive_clv": 25}}
    g = gate_segment(stats, league="EPL", market="1X2", tier="GOOD",
                     min_bets=30, max_ci_halfwidth=0.05)
    assert g["allowed"] is False
    assert "Wilson" in g["reason"]
    # 30/30 positive CLV -> half-width ~0.06 > 0.05 at n=30 still (z=1.96);
    # a larger n with the same rate tightens the CI.
    stats2 = {"EPL|1X2|GOOD": {"n": 200, "price_clv_pct": 3.0, "roi": 0.05, "n_positive_clv": 200}}
    g2 = gate_segment(stats2, league="EPL", market="1X2", tier="GOOD",
                      min_bets=30, max_ci_halfwidth=0.05)
    assert g2["allowed"] is True
    assert g2["ci"]["half_width"] <= 0.05


def test_gate_min_bets_now_30():
    # 30 settled bets with perfect positive CLV pass the lowered min_bets.
    stats = {"EPL|1X2|GOOD": {"n": 30, "price_clv_pct": 1.0, "roi": 0.02, "n_positive_clv": 30}}
    g = gate_segment(stats, league="EPL", market="1X2", tier="GOOD", min_bets=30)
    assert g["allowed"] is True
    # without max_ci_halfwidth the CI requirement is optional (backward compat)
    stats2 = {"EPL|1X2|GOOD": {"n": 30, "price_clv_pct": 1.0, "roi": 0.02, "n_positive_clv": 16}}
    assert gate_segment(stats2, league="EPL", market="1X2", tier="GOOD", min_bets=30)["allowed"] is True


# ---- 0.4: CLV segment report --------------------------------------------

def test_clv_segment_report_writes_file(tmp_path):
    log = tmp_path / "pred.jsonl"
    _snap(log)
    settle_auto(
        log, date="2026-08-15",
        results=[{"home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 1}],
        closing_fetcher=lambda r: {"home": 1.9, "draw": 3.6, "away": 4.4},
    )
    out = tmp_path / "out"
    rep = clv_segment_report(log, out_dir=out, date="2026-08-15")
    assert rep["coverage"]["with_closing_odds"] == 1
    assert rep["coverage"]["closing_coverage_pct"] == 100.0
    assert rep["coverage"]["passed"] is True
    assert rep["n_segments"] == 1  # ALL bucket only (no timed odds snapshots)
    seg = rep["segments"][0]
    assert seg["league"] == "EPL" and seg["market"] == "1X2" and seg["timing"] == "ALL"
    assert seg["n"] == 1
    assert seg["price_clv_pct"] > 0
    assert seg["ci"]["n_positive"] == 1
    fpath = Path(rep["file"])
    assert fpath.exists()
    payload = json.loads(fpath.read_text(encoding="utf-8"))
    assert payload["coverage"]["closing_coverage_pct"] == 100.0


def test_clv_segment_report_empty_log(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    rep = clv_segment_report(log, out_dir=tmp_path / "out", date="2026-08-15")
    assert rep["coverage"]["settled"] == 0
    assert rep["coverage"]["passed"] is False  # no data -> DoD not met
    assert rep["n_segments"] == 0


# ---- CLV closing-reference fallback (2026-08-16) -------------------------
#
# When a settlement carries no closing_odds (NowGoal closing fetch
# empty/disabled),
# the LAST PRE-KICKOFF odds snapshot is used as the closing reference so
# price CLV and the edge-bucket-vs-closing audit still have a real price.
# In-play captures (ts >= kickoff) never count as closing.

FUTURE_KICKOFF = "2099-01-01T14:00:00Z"


def _snap_future(path, *, kickoff=FUTURE_KICKOFF, odds=None):
    append_snapshot(
        path,
        match_id=MID,
        league="EPL", home="Arsenal", away="Chelsea",
        kickoff=kickoff,
        prob={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds=odds if odds is not None else {"home": 1.8, "draw": 3.6, "away": 4.4},
        edge=None, confidence=None, signal=None, calibration=None,
        model_version=None, input_hash=None,
        best_pick={"selection": "Home Win", "market": "1X2"},
        sources=[], decision_type="GOOD",
    )


def _osnap_raw(path, *, timing, ts, odds):
    """Append an odds_snapshot row directly (ts fully controlled)."""
    row = {
        "event": "odds_snapshot",
        "match_id": MID,
        "ts": ts,
        "timing": timing,
        "odds_1x2": odds,
        "odds_ah": None,
        "odds_ou": None,
        "bookmakers_count": 3,
        "sources": ["nowgoal"],
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _records(path):
    rows = [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return _settled_records(rows, 0.02)


def test_clv_falls_back_to_last_prekickoff_snapshot(tmp_path):
    """No closing_odds on the settle -> closing = last PRE-KICKOFF snapshot
    (T-0h), never an in-play capture taken after kickoff."""
    path = tmp_path / "p.jsonl"
    _snap_future(path)
    _osnap_raw(path, timing="T-24h", ts="2099-01-01T09:00:00+00:00",
               odds={"home": 2.0, "draw": 3.4, "away": 4.0})
    _osnap_raw(path, timing="T-0h", ts="2099-01-01T13:00:00+00:00",
               odds={"home": 2.2, "draw": 3.3, "away": 3.9})
    # In-play capture after kickoff: must be ignored as a closing reference.
    _osnap_raw(path, timing="T-0h", ts="2099-01-01T14:30:00+00:00",
               odds={"home": 1.2, "draw": 5.0, "away": 9.0})
    settle(path, match_id=MID, home_goals=2, away_goals=1)

    rec = _records(path)[0]
    assert rec["closing_source"] == "last_snapshot"
    assert rec["close_odds"] == 2.2  # T-0h pre-kickoff, not the in-play 1.2
    # price CLV = close/prediction - 1 = 2.2/1.8 - 1 (prediction odds 1.8)
    assert rec["price_clv"] == pytest.approx(2.2 / 1.8 - 1.0)
    # The T-24h bucket compares against the same fallback close.
    assert rec["price_clv_by_timing"]["T-24h"] == pytest.approx(2.2 / 2.0 - 1.0)
    assert rec["price_clv_by_timing"]["T-0h"] == pytest.approx(0.0)

    # The CLV segment report counts it as coverage, labelled by source.
    rep = clv_segment_report(path, out_dir=tmp_path / "out", date="2099-01-01")
    assert rep["coverage"]["with_closing_odds"] == 1
    assert rep["coverage"]["closing_coverage_pct"] == 100.0
    assert rep["coverage"]["closing_by_source"] == {"last_snapshot": 1}


def test_clv_uses_settle_closing_when_present(tmp_path):
    """A real closing_odds on the settlement wins over any snapshot fallback."""
    path = tmp_path / "p.jsonl"
    _snap_future(path)
    _osnap_raw(path, timing="T-0h", ts="2099-01-01T13:00:00+00:00",
               odds={"home": 2.2, "draw": 3.3, "away": 3.9})
    settle(path, match_id=MID, home_goals=1, away_goals=1,
           closing_odds={"home": 2.5, "draw": 3.2, "away": 3.0})

    rec = _records(path)[0]
    assert rec["closing_source"] == "settle"
    assert rec["close_odds"] == 2.5
    assert rec["price_clv"] == pytest.approx(2.5 / 1.8 - 1.0)


def test_clv_rejects_implausible_settle_closing(tmp_path):
    """A settlement closing price that no pre-match market could offer (e.g.
    a t=11 result-embedded final price leaking in as closing: 83.0 vs the
    last pre-kickoff 2.2) is rejected as a data-quality artifact; the CLV
    reference falls back to the last pre-kickoff snapshot instead of
    reporting a fictitious +3500% CLV."""
    path = tmp_path / "p.jsonl"
    _snap_future(path)
    _osnap_raw(path, timing="T-0h", ts="2099-01-01T13:00:00+00:00",
               odds={"home": 2.2, "draw": 3.3, "away": 3.9})
    settle(path, match_id=MID, home_goals=0, away_goals=1,
           closing_odds={"home": 83.0, "draw": 7.625, "away": 1.06})

    rec = _records(path)[0]
    assert rec["closing_source"] == "last_snapshot"
    assert rec["close_odds"] == 2.2
    assert rec["price_clv"] == pytest.approx(2.2 / 1.8 - 1.0)

    # The CLV segment report still counts coverage from the fallback.
    rep = clv_segment_report(path, out_dir=tmp_path / "out", date="2099-01-01")
    assert rep["coverage"]["with_closing_odds"] == 1
    assert rep["coverage"]["closing_by_source"] == {"last_snapshot": 1}


def test_clv_ignores_implausible_snapshot_price_in_fallback(tmp_path):
    """A snapshot whose price exceeds any plausible pre-match market (101.0)
    is skipped when selecting the closing-reference snapshot."""
    path = tmp_path / "p.jsonl"
    _snap_future(path)
    _osnap_raw(path, timing="T-24h", ts="2099-01-01T09:00:00+00:00",
               odds={"home": 2.0, "draw": 3.4, "away": 4.0})
    _osnap_raw(path, timing="T-0h", ts="2099-01-01T13:00:00+00:00",
               odds={"home": 101.0, "draw": 3.3, "away": 3.9})
    settle(path, match_id=MID, home_goals=2, away_goals=1)

    rec = _records(path)[0]
    assert rec["closing_source"] == "last_snapshot"
    assert rec["close_odds"] == 2.0  # T-24h, not the implausible 101.0
    assert rec["price_clv"] == pytest.approx(2.0 / 1.8 - 1.0)


def test_clv_no_fallback_when_only_inplay_snapshots(tmp_path):
    """Only in-play captures exist -> no closing reference -> CLV stays absent."""
    path = tmp_path / "p.jsonl"
    _snap_future(path)
    _osnap_raw(path, timing="T-0h", ts="2099-01-01T14:30:00+00:00",
               odds={"home": 1.2, "draw": 5.0, "away": 9.0})
    settle(path, match_id=MID, home_goals=0, away_goals=0)

    rec = _records(path)[0]
    assert rec["closing_source"] is None
    assert rec["price_clv"] is None
    assert rec["close_odds"] is None

    rep = clv_segment_report(path, out_dir=tmp_path / "out", date="2099-01-01")
    assert rep["coverage"]["with_closing_odds"] == 0
    assert rep["coverage"]["closing_by_source"] == {}


def test_clv_fallback_unparseable_kickoff_is_lenient(tmp_path):
    """Legacy snapshot without a usable kickoff: the latest snapshot is used
    as the closing reference (best effort) instead of blocking CLV entirely."""
    path = tmp_path / "p.jsonl"
    _snap_future(path, kickoff=None)
    _osnap_raw(path, timing="T-24h", ts="2099-01-01T09:00:00+00:00",
               odds={"home": 2.0, "draw": 3.4, "away": 4.0})
    settle(path, match_id=MID, home_goals=2, away_goals=1)

    rec = _records(path)[0]
    assert rec["closing_source"] == "last_snapshot"
    assert rec["close_odds"] == 2.0
