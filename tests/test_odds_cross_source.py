"""P3 — Cross-provider odds quality validation (acceptance tests).

Odds resolution is first-wins; P3 does NOT change that. It requires that
when more than one source is consulted, key lines (1X2 implied, O/U 2.5,
AH) are compared across sources and a meaningful disagreement is flagged as
``odds_quality: cross_source_disagreement``, which then docks data quality
and caps confidence at MEDIUM -- visible, never silently ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.analyse import _margin_free, cross_source_odds_check  # noqa: E402
from agents.football.signal_engine import rank_and_pick  # noqa: E402


def _payload(home: float, draw: float, away: float, over: float, under: float,
             ah_line: float, ah_home: float, ah_away: float) -> dict:
    """A normalized payload (same shape nowgoal/oddspapi/theoddsapi emit)."""
    def _mk(name: str, price: float) -> dict:
        return {"name": name, "price": price}
    return {
        "commence_time": "2026-08-15T18:30:00Z",
        "bookmakers": [
            {
                "title": "BookieX",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        _mk("Arsenal", home), _mk("Draw", draw), _mk("Chelsea", away),
                    ]},
                    {"key": "totals", "outcomes": [
                        {**_mk("Over", over), "point": 2.5},
                        {**_mk("Under", under), "point": 2.5},
                    ]},
                    {"key": "asian_handicap", "outcomes": [
                        {**_mk("Home", ah_home), "point": ah_line},
                        {**_mk("Away", ah_away), "point": ah_line},
                    ]},
                ],
            }
        ],
    }


def test_margin_free_removes_vig():
    probs = _margin_free({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert probs is not None
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    # margin-free: 0.5 / 1.0357 = 0.4828 (fair price ~2.07, not raw 2.0)
    assert 0.47 < probs["home"] < 0.49


def test_agreeing_sources_ok():
    a = _payload(2.0, 3.5, 4.0, 1.85, 1.95, -1.0, 1.90, 1.90)
    b = _payload(2.05, 3.45, 3.95, 1.87, 1.93, -1.0, 1.92, 1.88)
    out = cross_source_odds_check(
        {"oddspapi": a, "nowgoal": b},
        home_name="Arsenal", away_name="Chelsea",
        home_query="Arsenal", away_query="Chelsea",
    )
    assert out["status"] == "ok"
    assert out["n_sources"] == 2
    assert out["max_pp_diff"] is not None and out["max_pp_diff"] < 8.0


def test_disagreeing_sources_flags():
    # Same match, two sources with meaningfully different prices on every line
    # (implied gap well past the 8pp tolerance).
    a = _payload(1.50, 4.20, 6.00, 1.55, 2.40, -1.5, 1.80, 2.00)
    b = _payload(2.60, 3.20, 2.80, 2.30, 1.60, +0.5, 2.10, 1.75)
    out = cross_source_odds_check(
        {"oddspapi": a, "nowgoal": b},
        home_name="Arsenal", away_name="Chelsea",
        home_query="Arsenal", away_query="Chelsea",
        tolerance_pp=8.0,
    )
    assert out["status"] == "cross_source_disagreement"
    assert out["max_pp_diff"] > 8.0
    assert "1x2" in out["compared"]
    assert "over_2.5" in out["compared"]


def test_disagreement_docks_quality_and_caps_confidence():
    """The flag must flow into data_quality scoring: eff_completeness is
    docked (0.5x) and rank_and_pick caps the pick's confidence at MEDIUM with
    an explicit reason."""
    from agents.football.signal_engine import Signal

    def _mk_signal(dq: float) -> Signal:
        sig = Signal(
            market="1X2", selection="Home Win",
            model_prob=0.62, market_odds=1.8, implied_prob=0.55, edge_pp=7.0,
        )
        sig.components = {
            "model": 0.62, "statistical": 0.6, "market": 0.7,
            "movement": 0.5, "late_movement": 0.5, "data_quality": dq,
        }
        sig.score = 0.70
        return sig

    # Without disagreement: completeness 0.8 passes the 0.3 gate.
    ok_res = rank_and_pick(
        [_mk_signal(0.8)], best_pick_margin=0.06, no_bet_score=0.45,
        min_confluence=2, conflict_pp=8.0, min_data_quality=0.3,
        completeness=0.8, confidence_thresholds={},
    )
    assert ok_res["decision"] == "BEST PICK"
    assert ok_res["best_pick"].confidence in ("VERY HIGH", "HIGH")

    # With disagreement: completeness docked to 0.4 (still >= 0.3 gate so the
    # pick survives) but confidence capped MEDIUM and a reason is added.
    dis_res = rank_and_pick(
        [_mk_signal(0.4)], best_pick_margin=0.06, no_bet_score=0.45,
        min_confluence=2, conflict_pp=8.0, min_data_quality=0.3,
        completeness=0.4, confidence_thresholds={},
        odds_disagreement=True,
    )
    assert dis_res["decision"] == "BEST PICK"
    assert dis_res["best_pick"].confidence == "MEDIUM"
    assert any("disagreement" in r for r in dis_res["reasons"])


def test_disagreement_can_turn_pick_into_no_bet():
    """A thin-completeness match + disagreement: the docked completeness (0.5
    of 0.4 = 0.2) falls below min_data_quality 0.3 -> NO BET."""
    from agents.football.signal_engine import Signal

    sig = Signal(
        market="1X2", selection="Home Win",
        model_prob=0.62, market_odds=1.8, implied_prob=0.55, edge_pp=7.0,
    )
    sig.components = {
        "model": 0.62, "statistical": 0.6, "market": 0.7,
        "movement": 0.5, "late_movement": 0.5, "data_quality": 0.4,
    }
    sig.score = 0.70
    res = rank_and_pick(
        [sig], best_pick_margin=0.06, no_bet_score=0.45,
        min_confluence=2, conflict_pp=8.0, min_data_quality=0.3,
        completeness=0.2, confidence_thresholds={},
        odds_disagreement=True,
    )
    assert res["decision"] == "NO BET"
    assert res["best_pick"] is None
