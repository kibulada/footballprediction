"""Phase 1 tests: lineups/injuries into lambda.

1.2 lambda' = lambda x L (confirmed full weight, predicted half weight).
1.3 leakage guard: lineup fetched at/after kickoff is rejected as an input.
1.4 rest-days/congestion multiplier (flag-gated).
All model behavior is gated by PoissonModel.lineup_weight/rest_days_weight,
both 0 by default -> existing predictions stay byte-identical.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.context import MatchContext
from agents.football.models import (
    PoissonModel,
    lineup_lambda_multiplier,
    lineup_usable,
    rest_days_multiplier,
)
from agents.football.prediction_log import append_snapshot


def _ctx(**kw) -> MatchContext:
    base = dict(
        league="EPL", home="Arsenal", away="Chelsea",
        kickoff_utc="2026-08-15T14:00:00Z",
        home_gf_avg=1.8, home_ga_avg=1.0, away_gf_avg=1.2, away_ga_avg=1.3,
        form_samples=5,
    )
    base.update(kw)
    return MatchContext(**base)


def _model(**kw) -> PoissonModel:
    params = dict(
        base_home_goals=1.45, base_away_goals=1.25, dc_rho=-0.1,
        shrinkage_samples=5, time_decay_xi=0.9, xg_weight=0.65, min_samples=2,
    )
    params.update(kw)
    return PoissonModel(**params)


# ---- 1.2: lineup lambda multiplier --------------------------------------

def test_lineup_multiplier_no_missing_is_one():
    assert lineup_lambda_multiplier(["A", "B"], None, "confirmed") == 1.0
    assert lineup_lambda_multiplier(None, None, "confirmed") == 1.0


def test_lineup_multiplier_confirmed_vs_predicted_half_effect():
    starters = ["A", "B", "C", "D"]
    missing = ["A", "B"]
    confirmed = lineup_lambda_multiplier(starters, missing, "confirmed")
    predicted = lineup_lambda_multiplier(starters, missing, "predicted")
    assert confirmed < 1.0
    # predicted = half the effect of confirmed (spec 1.2: weight x0.5)
    assert abs((1.0 - predicted) - 0.5 * (1.0 - confirmed)) < 1e-9
    # 2 missing starters at 5% each = 10% cut, capped at 20%
    assert abs(confirmed - 0.90) < 1e-9


def test_lineup_multiplier_capped():
    starters = [f"P{i}" for i in range(11)]
    missing = starters[:6]  # 6 x 5% = 30% -> capped at 20%
    assert lineup_lambda_multiplier(starters, missing, "confirmed") == 0.80


def test_poisson_lineup_weight_zero_is_noop():
    base = _model()  # lineup_weight=0
    ctx = _ctx(lineup_home=["A", "B"], missing_home=["A"], lineup_status="confirmed")
    p0 = base.predict(ctx)
    ctx_no = _ctx()
    p1 = base.predict(ctx_no)
    assert p0["lambda_home"] == p1["lambda_home"]  # byte-identical when off
    assert p0["lineup_correction_applied"] is False


def test_poisson_lineup_weight_applies_correction():
    m = _model(lineup_weight=1.0)
    ctx = _ctx(lineup_home=["A", "B", "C", "D"], missing_home=["A", "B"],
               lineup_status="confirmed", lineup_ts="2026-08-15T10:00:00Z",
               lineup_source="flashscore_lineups")
    p = m.predict(ctx)
    assert p["lineup_correction_applied"] is True
    # lambda_home reduced (2 missing starters, -10%); away unchanged
    base = _model(lineup_weight=1.0).predict(_ctx())
    assert p["lambda_home"] < base["lambda_home"]
    assert abs(p["lambda_away"] - base["lambda_away"]) < 1e-9


# ---- 1.3: leakage guard -------------------------------------------------

def test_lineup_usable_rejects_at_or_after_kickoff():
    kickoff = "2026-08-15T14:00:00Z"
    assert lineup_usable("2026-08-15T10:00:00Z", kickoff) is True   # before
    assert lineup_usable("2026-08-15T14:00:00Z", kickoff) is False  # at kickoff
    assert lineup_usable("2026-08-15T15:00:00Z", kickoff) is False  # after
    assert lineup_usable(None, kickoff) is True                     # unknown -> ok
    assert lineup_usable("2026-08-15T10:00:00Z", None) is True


def test_poisson_rejects_lineup_fetched_at_kickoff():
    m = _model(lineup_weight=1.0)
    ctx_leak = _ctx(lineup_home=["A", "B", "C", "D"], missing_home=["A", "B"],
                    lineup_status="confirmed", lineup_ts="2026-08-15T14:00:00Z")
    p = m.predict(ctx_leak)
    assert p["lineup_correction_applied"] is False
    base = _model(lineup_weight=1.0).predict(_ctx())
    assert abs(p["lambda_home"] - base["lambda_home"]) < 1e-9


def test_snapshot_stores_lineup_provenance(tmp_path):
    path = tmp_path / "p.jsonl"
    append_snapshot(
        path,
        match_id="EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z",
        league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.5, "draw": 0.25, "away": 0.25}, odds=None, edge=None,
        confidence=None, signal=None, calibration=None, model_version=None,
        input_hash=None, best_pick=None, sources=[],
        lineup_source="flashscore_lineups", lineup_ts="2026-08-15T10:00:00Z",
    )
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    row = __import__("json").loads(rows[0])
    assert row["lineup_source"] == "flashscore_lineups"
    assert row["lineup_ts"] == "2026-08-15T10:00:00Z"


# ---- 1.4: rest-days / congestion ----------------------------------------

def test_rest_days_multiplier():
    assert rest_days_multiplier(None) == 1.0          # unknown -> no change
    assert rest_days_multiplier(5.0) == 1.0           # >= 4 days -> no penalty
    assert rest_days_multiplier(4.0) == 1.0
    assert rest_days_multiplier(3.0) == 0.95          # 1 day short -> -5%
    assert rest_days_multiplier(2.0) == 0.90          # 2 days short -> -10%
    assert rest_days_multiplier(0.0) == 0.90          # floor


def test_poisson_rest_days_zero_is_noop_and_applies_when_on():
    ctx = _ctx(home_days_rest=2.0, away_days_rest=7.0)
    base = _model().predict(_ctx())
    off = _model().predict(ctx)
    assert off["lambda_home"] == base["lambda_home"]
    on = _model(rest_days_weight=1.0).predict(ctx)
    assert on["rest_correction_applied"] is True
    assert on["lambda_home"] < base["lambda_home"]   # congestion penalty
    assert abs(on["lambda_away"] - base["lambda_away"]) < 1e-9
