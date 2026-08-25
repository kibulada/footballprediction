"""Phase 3 tests: CLV hard gate."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.clv_gate import gate_segment
from agents.football.prediction_log import append_snapshot, segment_clv_stats, settle

MID = "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z"


def _snapshot(path, *, prob=None, odds=None, market="1X2", decision_type="GOOD"):
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


def test_gate_blocks_when_segment_missing():
    g = gate_segment({}, league="EPL", market="1X2", tier="GOOD", min_bets=200)
    assert g["allowed"] is False
    assert "belum punya" in g["reason"]


def test_gate_blocks_on_sample_size():
    stats = {"EPL|1X2|GOOD": {"n": 10, "price_clv_pct": 3.0, "roi": 0.1}}
    g = gate_segment(stats, league="EPL", market="1X2", tier="GOOD", min_bets=200)
    assert g["allowed"] is False
    assert "10 < 200" in g["reason"]


def test_gate_blocks_on_negative_clv():
    stats = {"EPL|1X2|GOOD": {"n": 250, "price_clv_pct": -1.5, "roi": 0.05}}
    g = gate_segment(stats, league="EPL", market="1X2", tier="GOOD", min_bets=200)
    assert g["allowed"] is False
    assert "variance" in g["reason"]


def test_gate_allows_positive_clv():
    stats = {"EPL|1X2|GOOD": {"n": 250, "price_clv_pct": 1.5, "roi": 0.05}}
    g = gate_segment(stats, league="EPL", market="1X2", tier="GOOD", min_bets=200)
    assert g["allowed"] is True


def test_segment_clv_stats_aggregates(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    # Close at higher home price than prediction -> positive price CLV for home.
    settle(path, match_id=MID, home_goals=2, away_goals=1,
           closing_odds={"home": 1.9, "draw": 3.6, "away": 4.4})
    stats = segment_clv_stats(path)
    key = "EPL|1X2|GOOD"
    assert key in stats
    assert stats[key]["n"] == 1
    assert stats[key]["price_clv_pct"] > 0
    assert stats[key]["market"] == "1X2"
