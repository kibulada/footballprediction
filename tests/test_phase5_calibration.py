"""Phase 5 tests: per-league calibration + completeness merge."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.calibration import league_calibrator, league_slug
from agents.football.context import MatchContext
from agents.football.calibration import SignalScorer


def _ctx(**kw) -> MatchContext:
    return MatchContext(league="EPL", home="A", away="B", **kw)


def test_league_slug():
    assert league_slug("EPL") == "epl"
    assert league_slug("La Liga") == "la-liga"


def test_epl_uses_global_fit(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps({"a": 0.0, "b": 1.0, "samples": 500, "ece": 0.01})
    )
    cfg = {"models": {"calibration": {"file": "calibration.json", "league_min_samples": 400}}}
    assert league_calibrator("EPL", cfg, root=tmp_path) is not None


def test_foreign_league_without_fit_returns_none(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps({"a": 0.0, "b": 1.0, "samples": 500, "ece": 0.01})
    )
    cfg = {"models": {"calibration": {"file": "calibration.json", "league_min_samples": 400}}}
    assert league_calibrator("LaLiga", cfg, root=tmp_path) is None


def test_below_min_samples_returns_none(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps({"a": 0.0, "b": 1.0, "samples": 100, "ece": 0.05})
    )
    cfg = {"models": {"calibration": {"file": "calibration.json", "league_min_samples": 400}}}
    assert league_calibrator("EPL", cfg, root=tmp_path) is None


def test_completeness_single_feed_is_one_component():
    # odds + recent form/attack-defense (same feed) = 0.25 + 0.40 = 0.65,
    # NOT the old 0.25 + 0.20 + 0.20 double count.
    ctx = _ctx(
        consensus_odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        home_gf_avg=1.5, home_ga_avg=1.0,
        away_gf_avg=1.2, away_ga_avg=1.1,
        home_form="W-D-L", away_form="L-W-D",
    )
    comps = SignalScorer().components(
        ctx=ctx, ensemble_models=["elo"], model_vs_market=None, model_vs_model=None,
        calibration_quality=0.0, market_edge={}, p1x2=None,
    )
    assert comps["data_completeness"] == 0.65


def test_completeness_full_data_sums_to_one():
    ctx = _ctx(
        consensus_odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        home_gf_avg=1.5, home_ga_avg=1.0,
        away_gf_avg=1.2, away_ga_avg=1.1,
        home_form="W-D-L", away_form="L-W-D",
        home_xg_for=1.6, home_xg_against=1.1,
        away_xg_for=1.3, away_xg_against=1.2,
        h2h={"wins": 2, "draws": 1, "losses": 3},
    )
    comps = SignalScorer().components(
        ctx=ctx, ensemble_models=["elo"], model_vs_market=None, model_vs_model=None,
        calibration_quality=0.0, market_edge={}, p1x2=None,
    )
    assert comps["data_completeness"] == 1.0
