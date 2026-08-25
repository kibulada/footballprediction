"""Tests for the AH/O-U odds_snapshot extension (poll persistence + signal
engine consumption of the multi-snapshot movement series)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import (  # noqa: E402
    append_odds_snapshot,
    list_odds_snapshots,
)
from agents.football.signal_engine import (  # noqa: E402
    history_movement,
    ou_consensus,
    run_signal_engine,
)
from agents.football.models import poisson_matrix, probs_from_matrix  # noqa: E402


def _model_probs(lh=1.5, la=1.5):
    m = poisson_matrix(lh, la, rho=0.0)
    p1x2, o15, o25, o35, btts = probs_from_matrix(m)
    return {
        "1x2": p1x2, "over_1.5": o15, "over_2.5": o25, "over_3.5": o35,
        "btts_yes": btts, "lambda_home": lh, "lambda_away": la,
    }


def _tmp_log(match_id="M1"):
    p = Path(tempfile.mkdtemp()) / "log.jsonl"
    p.write_text(json.dumps({"event": "snapshot", "match_id": match_id}) + "\n", encoding="utf-8")
    return p


# ---- persistence ----------------------------------------------------------

def test_append_odds_snapshot_stores_ah_and_ou():
    p = _tmp_log()
    ok = append_odds_snapshot(
        p, match_id="M1", timing="T-1h", odds={"home": 1.7, "draw": 3.7, "away": 4.6},
        odds_ah={"line": -0.75, "home": 1.95, "away": 1.85},
        odds_ou={"line": 2.5, "over": 1.9, "under": 1.9},
        bookmakers_count=12, sources=["nowgoal"],
    )
    assert ok is True
    rows = list_odds_snapshots(p, "M1")
    assert len(rows) == 1
    assert rows[0]["odds_1x2"] == {"home": 1.7, "draw": 3.7, "away": 4.6}
    assert rows[0]["odds_ah"]["line"] == -0.75
    assert rows[0]["odds_ah"]["home"] == 1.95
    assert rows[0]["odds_ah"]["away"] == 1.85
    assert rows[0]["odds_ou"]["line"] == 2.5
    assert rows[0]["odds_ou"]["over"] == 1.9
    assert rows[0]["odds_ou"]["under"] == 1.9
    assert rows[0]["ts"]  # timestamp stored


def test_append_odds_snapshot_markets_optional():
    p = _tmp_log()
    ok = append_odds_snapshot(p, match_id="M1", timing="T-6h", odds={"home": 5.6, "draw": 3.5, "away": 1.54})
    assert ok is True
    row = list_odds_snapshots(p, "M1")[0]
    assert row["odds_ah"] is None
    assert row["odds_ou"] is None


# ---- history_movement -----------------------------------------------------

def test_history_movement_price_and_consistency():
    # Over shortening steadily: 2.02 -> 1.87
    hm = history_movement([2.02, 1.98, 1.96, 1.93, 1.90, 1.87], [2.5] * 6)
    assert hm["status"] == "available"
    assert hm["direction"] == "toward"
    assert hm["price_move_pct"] < 0
    assert hm["consistency"] == 1.0  # every move same direction
    assert hm["reversal"] is False
    assert hm["late_direction"] == -1.0
    assert hm["line_move"] == 0.0


def test_history_movement_line_move():
    # AH line moved -0.75 -> -1.0 (line movement separate from price)
    hm = history_movement([1.95, 1.90], [-0.75, -1.0])
    assert hm["line_move"] == -0.25
    assert hm["opening_line"] == -0.75
    assert hm["latest_line"] == -1.0


def test_history_movement_reversal():
    # overall shortening but last move reversed away
    hm = history_movement([2.00, 1.90, 1.95], [2.5] * 3)
    assert hm["direction"] == "toward"      # 2.00 -> 1.95
    assert hm["late_direction"] == 1.0      # last move lengthened
    assert hm["reversal"] is True


def test_history_movement_insufficient_points():
    hm = history_movement([1.90], [2.5])
    assert hm["status"] == "UNAVAILABLE"
    assert hm["n"] == 1


# ---- ou_consensus ---------------------------------------------------------

def test_ou_consensus_prefers_2_5():
    totals = {
        "Over 2.25": {"odds": 1.85, "point": 2.25},
        "Under 2.25": {"odds": 1.95, "point": 2.25},
        "Over 2.5": {"odds": 1.95, "point": 2.5},
        "Under 2.5": {"odds": 1.85, "point": 2.5},
    }
    ou = ou_consensus(totals)
    assert ou["line"] == 2.5
    assert ou["over"] == 1.95
    assert ou["under"] == 1.85


def test_ou_consensus_nearest_when_no_2_5():
    totals = {
        "Over 2.25": {"odds": 1.85, "point": 2.25},
        "Under 2.25": {"odds": 1.95, "point": 2.25},
    }
    ou = ou_consensus(totals)
    assert ou["line"] == 2.25


# ---- signal engine consumes snapshots -------------------------------------

def test_signal_engine_consumes_ah_history():
    ah_rows = [{"line": -0.75, "home": 1.95, "away": 1.85,
                "home_open": 2.0, "away_open": 1.80, "line_open": -0.75,
                "bookmaker": "Bet365"}]
    # away price drifts out 1.85 -> 1.92 over three snapshots at line -0.75
    snaps = [
        {"event": "odds_snapshot", "ts": "2026-08-15T07:00:00+00:00",
         "odds_ah": {"line": -0.75, "home": 1.95, "away": 1.85}},
        {"event": "odds_snapshot", "ts": "2026-08-15T08:00:00+00:00",
         "odds_ah": {"line": -0.75, "home": 1.90, "away": 1.88}},
        {"event": "odds_snapshot", "ts": "2026-08-15T09:00:00+00:00",
         "odds_ah": {"line": -0.75, "home": 1.85, "away": 1.92}},
    ]
    res = run_signal_engine(
        model_probs=_model_probs(), stats={}, market_totals={}, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.6, cfg=None,
        history_snapshots=snaps,
    )
    ah_sig = next(r for r in res["ranking"] if r["selection"] == "Away +0.75")
    mv = ah_sig["movement"]
    # consumed from the snapshot series (not the odds-payload opening/current)
    assert mv["status"] == "available"
    assert mv["n"] == 3
    assert mv["direction"] == "away"          # 1.85 -> 1.92 lengthened
    assert mv["consistency"] == 1.0
    assert mv["line_move"] == 0.0             # line unchanged across snaps
    assert mv["late_direction"] == 1.0
    assert res["data_quality"]["ah_ou_snapshots"] == 3


def test_signal_engine_consumes_ou_history_line_move():
    ah_rows = []
    snaps = [
        {"event": "odds_snapshot", "ts": "2026-08-15T07:00:00+00:00",
         "odds_ou": {"line": 2.25, "over": 1.90, "under": 1.90}},
        {"event": "odds_snapshot", "ts": "2026-08-15T08:00:00+00:00",
         "odds_ou": {"line": 2.5, "over": 1.85, "under": 1.95}},
    ]
    res = run_signal_engine(
        model_probs=_model_probs(), stats={}, market_totals={}, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.6, cfg=None,
        history_snapshots=snaps,
    )
    # Over 2.5 signal: only the 2.5 snapshot matches -> price history thin
    # but the full line series 2.25 -> 2.5 is captured as line movement.
    over = next(r for r in res["ranking"] if r["selection"] == "Over 2.5")
    mv = over["movement"]
    assert mv["line_move"] == 0.25  # 2.25 -> 2.5 line move
    assert mv["opening_line"] == 2.25
    assert mv["latest_line"] == 2.5


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
