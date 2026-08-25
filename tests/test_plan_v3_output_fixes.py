"""Plan v3 (2026-08-24) — paket inti A-D+E.

F1 Elo-anchor λ, F2 kalibrasi total ke market, F14 kandidat 1X2 di signal
engine, F11 renderer NO BET saat semua kandidat diveto.
Referensi: reports/bestpick_evaluasi_elche-barca_2026-08-24.md
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.context import MatchContext
from agents.football.elo import EloModel
from agents.football.models import (
    Ensemble,
    PoissonModel,
    apply_elo_anchor,
    calibrate_total_to_market,
    poisson_matrix,
    probs_from_matrix,
    run_prediction_engine,
)
from agents.football.signal_engine import (
    extract_asian_handicap,
    run_signal_engine,
    settle_signal,
)


# ---------------------------------------------------------------- F1 anchor

def test_apply_elo_anchor_below_min_gap_is_noop():
    lh2, la2, t = apply_elo_anchor(1.5, 1.3, 1600.0, 1650.0)
    assert (lh2, la2) == (1.5, 1.3)
    assert t == 0.0


def test_apply_elo_anchor_big_gap_flips_direction_keeps_total():
    # Elche v Barcelona class: feature λ terbalik vs Elo gap 715.
    lh2, la2, t = apply_elo_anchor(1.542, 1.359, 1582.7, 2298.0,
                                   min_gap=150, full_gap=400)
    assert t == 1.0
    assert la2 > lh2, "anchor harus membalik arah mengikuti Elo favorit away"
    assert math.isclose(lh2 + la2, 1.542 + 1.359, abs_tol=1e-9)


def test_apply_elo_anchor_partial_blend_mid_gap():
    # gap 275 -> t=0.5; rating home lebih rendah -> share home < 0.5,
    # sehingga λ_home harus TURUN dan λ_away NAIK dari nilai fitur.
    lh2, la2, t = apply_elo_anchor(1.0, 1.5, 1500.0, 1775.0,
                                   min_gap=150, full_gap=400)
    assert math.isclose(t, 0.5, abs_tol=1e-12)
    assert lh2 < 1.0 and la2 > 1.5
    assert math.isclose(lh2 + la2, 2.5, abs_tol=1e-9)


# ------------------------------------------------------- F2 total calibration

def test_calibrate_total_pulls_toward_fair_market():
    mt = {"Over 2.5": {"odds": 1.50}, "Under 2.5": {"odds": 2.65}}
    # Model over 77% >> fair ~63.8% -> λ harus mengecil.
    lh, la = 2.178, 1.883
    m = poisson_matrix(lh, la, rho=0.0)
    _, _, over25, _, _ = probs_from_matrix(m)
    assert over25 > 0.70
    lh2, la2, applied = calibrate_total_to_market(lh, la, mt, weight=0.5)
    assert applied
    _, _, over25b, _, _ = probs_from_matrix(poisson_matrix(lh2, la2, rho=0.0))
    assert abs(over25b - 0.638) < abs(over25 - 0.638)


def test_calibrate_total_noop_small_gap_or_missing_pair():
    mt = {"Over 2.5": {"odds": 2.0}, "Under 2.5": {"odds": 1.85}}
    lh, la = 1.30, 1.30
    lh2, la2, applied = calibrate_total_to_market(lh, la, mt)
    assert not applied and (lh2, la2) == (lh, la)
    _, _, applied2 = calibrate_total_to_market(1.5, 1.4, {"Over 2.5": {"odds": 1.5}})
    assert not applied2


# ------------------------------------------- integrasi engine (anchor+calib)

def _ctx(**kw) -> MatchContext:
    defaults = dict(
        league="EPL", home="Arsenal", away="Chelsea",
        home_recent_goals=[(2, 0), (2, 1), (3, 0)],
        away_recent_goals=[(0, 2), (1, 2), (0, 3)],
        form_samples=5,
        consensus_odds={"home": 1.85, "draw": 3.6, "away": 4.2},
        market_totals={
            "Over 2.5": {"odds": 1.60}, "Under 2.5": {"odds": 2.40},
        },
    )
    defaults.update(kw)
    return MatchContext(**defaults)


def _seeded_elo() -> EloModel:
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1900.0, "Chelsea": 1450.0})
    return elo


def test_engine_applies_elo_anchor_and_reports_audit():
    """Elche-class di level engine: fitur bilang home kuat, Elo bilang away
    jauh lebih kuat -> anchored run harus menekan λ_home dibanding non-anchor,
    dan audit field terisi."""
    from agents.football.calibration import Calibrator, SignalScorer

    elo = EloModel()
    elo.ratings.update({"Arsenal": 1450.0, "Chelsea": 1900.0})  # away unggul
    elo._rebuild_indexes()  # ratings di-inject manual -> rebuild lookup index
    common = dict(ctx=_ctx(), elo=elo, ensemble=Ensemble(),
                  calibrator=Calibrator(), scorer=SignalScorer())
    off = run_prediction_engine(
        poisson=PoissonModel(elo_anchor={"enabled": False}), **common)
    on = run_prediction_engine(
        poisson=PoissonModel(elo_anchor={"enabled": True}), **common)
    assert off is not None and on is not None
    assert off.model_probs["elo_anchor_t"] == 0.0
    assert on.model_probs["elo_anchor_t"] > 0
    assert (on.model_probs["lambda_home"]
            < off.model_probs["lambda_home"]), \
        "anchor harus menekan λ home yang salah arah"


def test_engine_market_total_calibration_reported():
    from agents.football.calibration import Calibrator, SignalScorer

    # Form ekstrem banyak gol -> feature λ_total tinggi; market fair over 62%.
    res = run_prediction_engine(
        ctx=_ctx(), elo=EloModel(), poisson=PoissonModel(),
        ensemble=Ensemble(), calibrator=Calibrator(), scorer=SignalScorer(),
    )
    assert res is not None
    assert isinstance(res.model_probs["market_total_calibrated"], bool)


# --------------------------------------------------------- F14 kandidat 1X2

_ELCHE_MP = {
    "1x2": {"home": 0.126, "draw": 0.143, "away": 0.731},
    "over_1.5": 0.78, "over_2.5": 0.63, "over_3.5": 0.35, "btts_yes": 0.45,
    "lambda_home": 0.90, "lambda_away": 2.40,
}
_ELCHE_TOTALS = {
    "Over 2.5": {"odds": 1.50, "point": 2.5},
    "Under 2.5": {"odds": 2.65, "point": 2.5},
    "BTTS Yes": {"odds": 1.88},
    "BTTS No": {"odds": 1.99},
}
_ELCHE_AH_PAYLOAD = {
    "bookmakers": [{
        "title": "Consensus",
        "markets": [{
            "key": "asian_handicap",
            "outcomes": [
                {"name": "Home", "price": 1.97, "point": 1.5},
                {"name": "Away", "price": 1.83, "point": 1.5},
            ],
        }],
    }],
}


def _run_elche_like(cfg=None):
    # cfg meniru production: allow_negative_edge_pp -3.0 (config/football.json)
    base = {"allow_negative_edge_pp": -3.0}
    base.update(cfg or {})
    return run_signal_engine(
        model_probs=dict(_ELCHE_MP),
        stats={},
        market_totals=_ELCHE_TOTALS,
        ah_rows=extract_asian_handicap(_ELCHE_AH_PAYLOAD),
        movement_snapshot=None, context=None, completeness=0.8,
        cfg=base,
        league_name="La Liga",
        odds_1x2={"home": 9.19, "draw": 5.89, "away": 1.322},
    )


def test_1x2_candidates_created_with_margin_free_implied():
    res = _run_elche_like()
    rows = {r["selection"]: r for r in res["ranking"] if r["market"] == "1X2"}
    assert set(rows) >= {"Home Win", "Draw", "Away Win"}
    aw = rows["Away Win"]
    assert aw["implied_prob"] is not None
    s = sum(r["implied_prob"] for r in rows.values()
            if r["implied_prob"] is not None)
    assert math.isclose(s, 1.0, abs_tol=1e-3)  # implied di-round 4dp per sisi
    assert not aw["vetoed"]


def test_elche_card_no_longer_picks_btts_yes():
    """Kasus user: BEST PICK harus berpindah dari BTTS Yes ke kandidat arah."""
    res = _run_elche_like()
    assert res["decision"] == "BEST PICK"
    bp = res["best_pick"]
    assert bp is not None and bp["selection"] != "BTTS Yes"
    assert bp["selection"] in ("Away Win", "Over 2.5") or (
        bp["market"] == "Asian Handicap" and bp.get("side") == "away"
    )


def test_1x2_deviant_deviation_is_vetoed_by_g2():
    mp = dict(_ELCHE_MP)
    mp["1x2"] = {"home": 0.10, "draw": 0.15, "away": 0.40}  # market away ~73%
    res = run_signal_engine(
        model_probs=mp, stats={}, market_totals=_ELCHE_TOTALS,
        ah_rows=[], movement_snapshot=None, context=None, completeness=0.8,
        cfg={}, league_name="La Liga",
        odds_1x2={"home": 9.19, "draw": 5.89, "away": 1.322},
    )
    aw = next(r for r in res["ranking"]
              if r["market"] == "1X2" and r["selection"] == "Away Win")
    assert aw["vetoed"] and any("deviasi" in r for r in aw["veto_reasons"])


def test_1x2_disabled_via_config():
    res = _run_elche_like(cfg={"enable_1x2_signals": False})
    assert all(r["market"] != "1X2" for r in res["ranking"])


def test_settle_signal_1x2():
    win = {"market": "1X2", "selection": "Away Win"}
    assert settle_signal(win, 0, 5)["result"] == "win"
    assert settle_signal(win, 2, 1)["result"] == "loss"
    draw = {"market": "1X2", "selection": "Draw"}
    assert settle_signal(draw, 1, 1)["result"] == "win"


# ------------------------------------------------------------- F11 renderer

def test_display_best_pick_all_vetoed_now_plain_no_bet():
    se = {
        "decision": "NO BET", "display_label": "BEST PICK", "best_pick": None,
        "reasons": ["Over 2.5: lambda_total 4.11 di luar band"],
        "ranking": [{
            "market": "Total", "selection": "Over 2.5", "score": 0.78,
            "confidence": "NO SIGNAL", "model_prob": 0.82, "market_odds": 2.06,
            "implied_prob": 0.47, "edge_pp": 34.9, "movement": {},
            "components": {}, "line": None, "side": None, "line_key": None,
            "internal_notes": [], "vetoed": True,
            "veto_reasons": ["lambda_total 4.11 di luar band [1.6, 3.6]"],
        }],
    }
    from agents.football.format import _display_best_pick

    assert _display_best_pick(se) == (None, None)
