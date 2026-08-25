"""Phase 2/3/4 tests: selection filter, actionable gate, market blend,
multi-league validation harness, paper-trade logging."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.decision import (
    Candidate,
    actionable_gate,
    blend_model_with_market,
    market_blend_alpha,
    selection_filter,
)
from agents.football.prediction_log import append_snapshot
from agents.football.runner import timing_label
from agents.football.validate import validate_multileague

CFG = {
    "models": {
        "decision": {
            "selection": {
                "markets": ["Total", "BTTS", "Asian Handicap", "1X2"],
                "leagues_primary": ["Eredivisie", "EFL Championship"],
                "one_x2_leagues": ["Eredivisie", "EFL Championship"],
            }
        }
    }
}


def _cand(market: str, selection: str, p: float = 0.6) -> Candidate:
    return Candidate(
        market=market, selection=selection, model_prob=p,
        market_odds=2.0, implied_prob=0.5, edge_pp=(p - 0.5) * 100.0,
        ev=p * 2.0 - 1.0, independent=True,
    )


# ---- Phase 2.1: selection filter ----------------------------------------

def test_selection_filter_drops_big5_one_x2():
    cands = [_cand("1X2", "Home Win"), _cand("Total", "Over 2.5")]
    eligible, reasons = selection_filter(cands, CFG, "EPL")
    assert [c.market for c in eligible] == ["Total"]
    assert any("sanity-check" in r for r in reasons)


def test_selection_filter_allows_primary_league_one_x2():
    cands = [_cand("1X2", "Home Win"), _cand("BTTS", "Yes")]
    eligible, _ = selection_filter(cands, CFG, "Eredivisie")
    assert [c.market for c in eligible] == ["1X2", "BTTS"]


def test_selection_filter_market_whitelist():
    cands = [_cand("1X2", "Home Win"), _cand("Asian Handicap", "Home -0.5")]
    eligible, reasons = selection_filter(cands, CFG, "Eredivisie")
    assert [c.market for c in eligible] == ["1X2", "Asian Handicap"]
    # no config -> no filtering (default behavior unchanged)
    eligible2, _ = selection_filter(cands, None, "EPL")
    assert len(eligible2) == 2


# ---- Phase 2.3: actionable gate -----------------------------------------

def test_actionable_gate_requires_all_conditions():
    ok, reasons = actionable_gate(
        league_calibrated=True, edge_pp=5.0, min_edge_pp=3.0,
        benchmark_stale=False, clv_gate={"allowed": True, "reason": None},
    )
    assert ok is True and not reasons
    # uncalibrated league -> fail (Phase 1.5)
    ok2, r2 = actionable_gate(
        league_calibrated=False, edge_pp=5.0, min_edge_pp=3.0,
        benchmark_stale=False, clv_gate={"allowed": True, "reason": None},
    )
    assert ok2 is False and any("kalibrasi" in r for r in r2)
    # stale benchmark -> fail (Phase 0.2)
    ok3, r3 = actionable_gate(
        league_calibrated=True, edge_pp=5.0, min_edge_pp=3.0,
        benchmark_stale=True, clv_gate={"allowed": True, "reason": None},
    )
    assert ok3 is False and any("stale" in r for r in r3)
    # edge below threshold -> fail
    ok4, _ = actionable_gate(
        league_calibrated=True, edge_pp=1.0, min_edge_pp=3.0,
        benchmark_stale=False, clv_gate={"allowed": True, "reason": None},
    )
    assert ok4 is False
    # segment CLV blocked -> fail
    ok5, _ = actionable_gate(
        league_calibrated=True, edge_pp=5.0, min_edge_pp=3.0,
        benchmark_stale=False, clv_gate={"allowed": False, "reason": "n<30"},
    )
    assert ok5 is False


# ---- Phase 3.1: probability blend ---------------------------------------

def test_market_blend_alpha_boundary():
    assert market_blend_alpha(0.0) == 1.0   # pure market -> NO BET
    assert market_blend_alpha(1.0) == 0.0   # pure calibrated model
    assert market_blend_alpha(0.5) == 0.5


def test_blend_model_with_market_pure_market_when_uncalibrated():
    model = {"1x2": {"home": 0.70, "draw": 0.20, "away": 0.10}}
    market = {"home": 0.50, "draw": 0.30, "away": 0.20}
    out, meta = blend_model_with_market(model, market, 0.0)
    assert meta["pure_market"] is True
    assert out["1x2"] == market  # alpha=1 -> pure market, edge=0


def test_blend_model_with_market_mix():
    model = {"1x2": {"home": 0.60, "draw": 0.25, "away": 0.15}}
    market = {"home": 0.50, "draw": 0.30, "away": 0.20}
    out, meta = blend_model_with_market(model, market, 0.5)
    assert not meta["pure_market"]
    # p_final = 0.5*market + 0.5*model, renormalized
    blended = out["1x2"]
    assert abs(blended["home"] - 0.55) < 1e-6
    assert abs(sum(blended.values()) - 1.0) < 1e-6


def test_blend_no_market_probs_returns_model_unchanged():
    model = {"1x2": {"home": 0.6, "draw": 0.25, "away": 0.15}}
    out, meta = blend_model_with_market(model, None, 0.5)
    assert out["1x2"] == model["1x2"]


# ---- Phase 3.2: lambda pinning preserved (lineup correction is a
# multiplier, never a replacement source) ---------------------------------

def test_lineup_correction_does_not_change_lambda_source():
    from agents.football.context import MatchContext
    from agents.football.models import PoissonModel

    ctx = MatchContext(
        league="EPL", home="A", away="B", kickoff_utc="2026-08-15T14:00:00Z",
        home_gf_avg=1.8, home_ga_avg=1.0, away_gf_avg=1.2, away_ga_avg=1.3,
        form_samples=5,
        lineup_home=["X", "Y"], missing_home=["X"], lineup_status="confirmed",
        lineup_ts="2026-08-15T10:00:00Z",
    )
    m = PoissonModel(lineup_weight=1.0)
    p = m.predict(ctx)
    assert p["lambda_source"] in ("features", "features+xg")
    assert p["lineup_correction_applied"] is True


# ---- Phase 2.2: timing labels -------------------------------------------

def test_timing_label_t24h():
    assert timing_label(24.0) == "T-24h"
    assert timing_label(0.5) == "T-30m"
    assert timing_label(-1.0) == "T-0h"


# ---- Phase 4.1: multi-league validation harness -------------------------

def _fx(league, home, away, hg, ag, odds):
    return {
        "league": league, "home": home, "away": away,
        "home_goals": hg, "away_goals": ag, "date": "2023-01-01",
        "season": "2022-2023", "home_odds": odds[0], "draw_odds": odds[1],
        "away_odds": odds[2], "odds_source": "test",
    }


def test_validate_multileague_writes_report(tmp_path):
    fixtures = {
        "EPL": [
            _fx("EPL", "A", "B", 2, 1, (1.8, 3.5, 4.0)),
            _fx("EPL", "C", "D", 0, 0, (2.2, 3.2, 3.1)),
            _fx("EPL", "E", "F", 1, 2, (2.0, 3.4, 3.6)),
        ]
    }
    out = tmp_path / "out"
    rep = validate_multileague(
        fixtures, out_dir=out, date="2026-08-15",
        requested_leagues=["Eredivisie", "EPL"],
    )
    assert rep["available_leagues"] == ["EPL"]
    assert rep["missing_leagues"] == ["Eredivisie"]
    assert rep["n_segments"] > 0
    missing_rows = [s for s in rep["segments"] if s.get("data_missing")]
    assert any(s["league"] == "Eredivisie" for s in missing_rows)
    epl_rows = [s for s in rep["segments"] if s["league"] == "EPL" and s["model"] == "ensemble"]
    assert epl_rows and epl_rows[0]["n"] == 3
    assert epl_rows[0]["log_loss"] is not None
    fpath = Path(rep["file"])
    assert fpath.exists()
    payload = json.loads(fpath.read_text(encoding="utf-8"))
    assert payload["missing_leagues"] == ["Eredivisie"]


# ---- Phase 4.2: paper-trade flag on snapshots ---------------------------

def test_snapshot_paper_trade_flag(tmp_path):
    path = tmp_path / "p.jsonl"
    append_snapshot(
        path,
        match_id="EPL||A||B||2026-08-15",
        league="EPL", home="A", away="B", kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.5, "draw": 0.25, "away": 0.25}, odds=None, edge=None,
        confidence=None, signal=None, calibration=None, model_version=None,
        input_hash=None, best_pick=None, sources=[], paper_trade=True,
    )
    row = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["paper_trade"] is True
