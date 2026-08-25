"""Phase 6 tests: fractional-Kelly staking + extreme-edge auto-decline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.staking import compute_stake, kelly_full_stake


def test_kelly_full_stake_math():
    # p=0.55, odds=2.0 -> b=1 -> f* = 0.55 - 0.45 = 0.10
    assert abs(kelly_full_stake(0.55, 2.0) - 0.10) < 1e-9


def test_kelly_full_stake_no_value_is_zero():
    assert kelly_full_stake(0.40, 2.0) == 0.0


def test_extreme_edge_auto_decline():
    s = compute_stake(model_prob=0.9, decimal_odds=1.5, edge_pp=25.0,
                      decision_type="STRONG")
    assert s["declined"] is True
    assert "data error" in s["reason"]


def test_watch_tier_no_stake():
    s = compute_stake(model_prob=0.55, decimal_odds=2.0, edge_pp=5.0,
                      decision_type="WATCH")
    assert s["declined"] is True


def test_stake_capped_at_max_fraction():
    # f* = 0.10, 1/4 Kelly * STRONG(1.0) = 0.025 -> capped at 0.02.
    s = compute_stake(model_prob=0.55, decimal_odds=2.0, edge_pp=5.0,
                      decision_type="STRONG")
    assert s["declined"] is False
    assert s["stake_fraction"] == 0.02
    assert s["stake_amount"] == 2.0  # 2% of 100-unit bankroll


def test_tier_multiplier_scales_stake():
    s = compute_stake(model_prob=0.55, decimal_odds=2.0, edge_pp=5.0,
                      decision_type="LEAN")
    # f* = 0.10, 1/4 Kelly * LEAN(0.5) = 0.0125 (< cap 0.02)
    assert s["declined"] is False
    assert abs(s["stake_fraction"] - 0.0125) < 1e-9
