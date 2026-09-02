"""Post-mortem 2026-09-02 (26 Aug-1 Sep live, 83 picks: BEST PICK tier 42-15,
SUGGESTION 29-5).

Each test encodes ONE failure class / model fix as a general rule, never a
per-match threshold:

  K7  no conviction & no value -- BEST PICK tier needs model_prob >= 0.60 OR
                                  (>= 0.50 with edge >= 0); HIGH capped
                                  below 0.60 (signal_engine.pick_tier_for)
  K8  stale hold               -- apply_pick_stability releases a held pick
                                  when another bet beats it by
                                  best_pick_margin or it decayed itself
  K6  internal disagreement    -- Elo-led blend vs Poisson lambdas on
                                  direction -> MEDIUM cap + failure class,
                                  never a veto (LASK won with it)
  G11 total_favor_gate         -- Total/BTTS model <50% AND edge <0 vetoed
  P4  Elo canonical lookup     -- "Lille"/"Genk" resolve via teams.json
                                  canonical name (resolve_first)
  M1  market anchor            -- p = alpha*model + (1-alpha)*market, alpha
                                  by Elo evidence; raw kept for audit
  M2  ensemble evidence        -- both sides on the prior -> Elo dropped
                                  from the blend when a feature Poisson ran
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.context import MatchContext, build_match_context  # noqa: E402
from agents.football.elo import EloModel  # noqa: E402
from agents.football.models import (  # noqa: E402
    Ensemble,
    PoissonModel,
    apply_market_anchor,
    market_anchor_alpha,
    run_prediction_engine,
)
from agents.football.pick_gates import (  # noqa: E402
    lambda_direction_conflict,
    total_favor_gate,
)
from agents.football.prediction_log import classify_failure  # noqa: E402
from agents.football.signal_engine import (  # noqa: E402
    apply_pick_stability,
    pick_tier_for,
    run_signal_engine,
)


# --------------------------------------------------------------------------
# K7 -- tier rule B + confidence cap
# --------------------------------------------------------------------------

def _bp(score=0.62, conf="MEDIUM", prob=0.55, edge=2.0):
    return {"score": score, "confidence": conf, "model_prob": prob, "edge_pp": edge}


def test_tier_rule_b_conviction_or_value():
    # prob >= 0.60 -> BEST PICK even with a negative edge (Dortmund -3.9pp won)
    assert pick_tier_for(_bp(prob=0.65, edge=-3.0)) == ("BEST PICK", None)
    # 0.50-0.60 with edge >= 0 -> value pick (Real Madrid BTTS 0.56/+4.8 won)
    assert pick_tier_for(_bp(prob=0.55, edge=2.0)) == ("BEST PICK", None)
    # 0.50-0.60 with edge < 0 -> LEAN with the reason (Alaves BTTS 0.55/-2.7 lost)
    tier, reason = pick_tier_for(_bp(prob=0.55, edge=-2.0))
    assert tier == "LEAN" and "tidak ada value" in reason and "55%" in reason
    # below 0.50 -> LEAN regardless of edge (Wrexham 0.43/+0.03 lost)
    tier, reason = pick_tier_for(_bp(prob=0.46, edge=0.03))
    assert tier == "LEAN" and "46%" in reason and "< 50%" in reason


def test_tier_rule_keeps_k5_rules_and_reads_config():
    assert pick_tier_for(_bp(score=0.50, prob=0.7, edge=1.0))[0] == "LEAN"
    assert pick_tier_for(_bp(conf="LOW", prob=0.7, edge=1.0))[0] == "LEAN"
    assert pick_tier_for(None) == (None, None)
    # config knobs override the module constants
    cfg = {"best_pick_min_prob": 0.70, "best_pick_value_min_prob": 0.60}
    assert pick_tier_for(_bp(prob=0.65, edge=-1.0), cfg)[0] == "LEAN"
    assert pick_tier_for(_bp(prob=0.65, edge=1.0), cfg)[0] == "BEST PICK"
    # an old row without model_prob is tiered on score/confidence only
    assert pick_tier_for({"score": 0.6, "confidence": "MEDIUM"})[0] == "BEST PICK"


def _engine(model_probs, **kw):
    base = dict(
        model_probs=model_probs,
        stats={"home_recent_goals": [1, 2, 0, 1, 2], "away_recent_goals": [0, 1, 1, 0, 2]},
        market_totals={"Over 2.5": {"odds": 1.9}, "Under 2.5": {"odds": 1.9},
                       "BTTS Yes": {"odds": 1.85}, "BTTS No": {"odds": 1.95}},
        ah_rows=[],
        odds_1x2={"home": 1.72, "draw": 3.8, "away": 4.15},
        completeness=1.0,
        cfg={"pick_gates": {"elo_integrity": True, "agreement": False, "lambda_total_sanity": False,
                            "source_consistency": False},
             "min_edge_pp": -50.0, "allow_negative_edge_pp": -50.0, "no_bet_score": 0.0,
             "min_confluence": 0, "min_data_quality": 0.0},
    )
    base.update(kw)
    return run_signal_engine(**base)


def _mp(**over):
    mp = {"1x2": {"home": 0.50, "draw": 0.24, "away": 0.26}, "over_2.5": 0.55, "btts_yes": 0.55,
          "lambda_home": 1.75, "lambda_away": 1.18, "lambda_source": "features+xg",
          "elo_seeded": True, "elo_home_seeded": True, "elo_away_seeded": True,
          "elo_home": 1700.0, "elo_away": 1650.0}
    mp.update(over)
    return mp


def test_engine_never_prints_high_below_conviction():
    """Coventry 2026-08-29: HIGH on a 55% proposition. Every ranked candidate
    with model_prob < 0.60 must be MEDIUM at most."""
    res = _engine(_mp(**{"1x2": {"home": 0.56, "draw": 0.24, "away": 0.20}}))
    for e in res["ranking"]:
        if e["model_prob"] < 0.60:
            assert e["confidence"] not in ("HIGH", "VERY HIGH"), e


def test_engine_tier_reason_travels_with_the_pick():
    """Lorient shape: the picked 1X2 favourite sits at 47% with a negative
    edge -> LEAN, and the engine's tier/reason equal ``pick_tier_for`` on
    the pick it selected."""
    res = _engine(_mp(**{"1x2": {"home": 0.47, "draw": 0.28, "away": 0.25},
                         "over_2.5": 0.50, "btts_yes": 0.50}),
                  odds_1x2={"home": 1.98, "draw": 3.4, "away": 4.075},
                  market_totals={"Over 2.5": {"odds": 1.85}, "Under 2.5": {"odds": 1.95},
                                 "BTTS Yes": {"odds": 1.8}, "BTTS No": {"odds": 2.0}})
    assert res["best_pick"] is not None
    assert (res["pick_tier"], res["tier_reason"]) == pick_tier_for(res["best_pick"], {})
    home = next(e for e in res["ranking"] if e["selection"] == "Home Win")
    assert pick_tier_for(home)[0] == "LEAN" and "keyakinan model 47%" in pick_tier_for(home)[1]


# --------------------------------------------------------------------------
# K8 -- stale hold release
# --------------------------------------------------------------------------

def _entry(market, selection, score, conf="MEDIUM", prob=0.55, odds=1.9, edge=1.0):
    return {"market": market, "selection": selection, "score": score, "confidence": conf,
            "model_prob": prob, "market_odds": odds, "edge_pp": edge, "line": None, "side": None,
            "line_key": "", "components": {"model": prob, "market": 0.8}, "movement": {},
            "vetoed": False, "veto_reasons": []}


def _prev(selection, score, market="1X2"):
    return {"decision": "BEST PICK", "market": market, "selection": selection, "score": score,
            "ts": "2026-08-28T17:00:00+00:00", "line_key": None}


def test_stability_releases_hold_when_other_bet_is_materially_stronger():
    """Wrexham v Birmingham 2026-08-28: held Home Win 0.524 (logged 0.630)
    while BTTS Yes 0.670 HIGH stood -- and won."""
    se = {"ranking": [_entry("BTTS", "BTTS Yes", 0.67, "HIGH", 0.53),
                      _entry("1X2", "Home Win", 0.524, "MEDIUM", 0.43)],
          "best_pick": None, "decision": "BEST PICK"}
    out = apply_pick_stability(
        se, previous_pick=_prev("Home Win", 0.63), current_model={}, opening_snapshot=None,
        market_totals={}, now_ts="2026-08-28T18:25:00+00:00", cfg={"stability": {"enabled": True}},
        score_threshold=0.05,
    )
    st = out["stability"]
    assert st["status"] == "changed"
    assert st["new_selection"] == "BTTS Yes"
    assert "lebih kuat" in st["reason"] or "melemah" in st["reason"]


def test_stability_still_holds_small_gap_same_bet():
    """Lille v PSG 2026-08-28: BTTS Yes 0.613 vs held Over 2.5 0.592 (gap
    0.021 < 0.06, no decay) -> hold, exactly as before."""
    se = {"ranking": [_entry("BTTS", "BTTS Yes", 0.613, "MEDIUM", 0.63),
                      _entry("Total", "Over 2.5", 0.592, "MEDIUM", 0.61)],
          "best_pick": None, "decision": "BEST PICK"}
    out = apply_pick_stability(
        se, previous_pick=_prev("Over 2.5", 0.59, market="Total"), current_model={},
        opening_snapshot=None, market_totals={}, now_ts="2026-08-28T18:25:00+00:00",
        cfg={"stability": {"enabled": True}}, score_threshold=0.05,
    )
    assert out["stability"]["status"] == "held"
    assert out["best_pick"]["selection"] == "Over 2.5"


def test_stability_releases_hold_when_held_pick_decayed():
    """Same bet on top, but the held pick fell by more than the threshold."""
    se = {"ranking": [_entry("1X2", "Home Win", 0.55, "MEDIUM", 0.5)],
          "best_pick": None, "decision": "BEST PICK"}
    out = apply_pick_stability(
        se, previous_pick=_prev("Home Win", 0.64), current_model={}, opening_snapshot=None,
        market_totals={}, now_ts="2026-08-28T18:25:00+00:00", cfg={"stability": {"enabled": True}},
        score_threshold=0.05,
    )
    # either the pre-existing delta rule or the new decay rule may label it --
    # what matters is that a decayed hold is RELEASED with a reason.
    assert out["stability"]["status"] == "changed"
    assert "melemah" in out["stability"]["reason"] or "skor berubah" in out["stability"]["reason"]


# --------------------------------------------------------------------------
# K6 -- Elo vs Poisson direction conflict (cap, not veto)
# --------------------------------------------------------------------------

def test_lambda_direction_conflict_cases():
    coventry = {"1x2": {"home": 0.55, "draw": 0.22, "away": 0.23}, "lambda_home": 1.409, "lambda_away": 1.43}
    assert lambda_direction_conflict(coventry, "1X2", "Home Win")[0]
    lorient = {"1x2": {"home": 0.47, "draw": 0.28, "away": 0.26}, "lambda_home": 1.475, "lambda_away": 0.638}
    assert not lambda_direction_conflict(lorient, "1X2", "Home Win")[0]
    # an AH underdog line never needs its side to be the stronger one
    assert not lambda_direction_conflict(coventry, "Asian Handicap", "Away +1.5", "away")[0]
    # Draw / Totals carry no direction
    assert not lambda_direction_conflict(coventry, "1X2", "Draw")[0]
    assert not lambda_direction_conflict(coventry, "Total", "Over 2.5")[0]
    assert lambda_direction_conflict({}, "1X2", "Home Win") == (False, None)


def test_engine_k6_caps_medium_without_veto():
    res = _engine(_mp(lambda_home=1.409, lambda_away=1.43,
                      **{"1x2": {"home": 0.55, "draw": 0.22, "away": 0.23}}))
    home = next(e for e in res["ranking"] if e["selection"] == "Home Win")
    assert not home["vetoed"], home["veto_reasons"]
    assert home["confidence"] in ("MEDIUM", "LOW", "NO SIGNAL")
    # switching the cap off restores the plain path (no K6 note anywhere)
    res_off = _engine(_mp(lambda_home=1.409, lambda_away=1.43,
                          **{"1x2": {"home": 0.55, "draw": 0.22, "away": 0.23}}),
                      cfg={"pick_gates": {"elo_integrity": True, "agreement": False,
                                          "lambda_total_sanity": False, "source_consistency": False,
                                          "lambda_direction_cap": False},
                           "min_edge_pp": -50.0, "allow_negative_edge_pp": -50.0, "no_bet_score": 0.0,
                           "min_confluence": 0, "min_data_quality": 0.0})
    assert res_off["best_pick"] is not None


# --------------------------------------------------------------------------
# G11 -- total_favor_gate
# --------------------------------------------------------------------------

def test_total_favor_gate_lincoln_atalanta_class():
    ok, rs = total_favor_gate("Total", "Over 2.5", 0.46, -2.4)   # Lincoln 1 Sep, FT 0-0
    assert not ok and "46%" in rs[0]
    ok, rs = total_favor_gate("Total", "Over 2.5", 0.49, -1.6)   # Atalanta 31 Aug, FT 1-0
    assert not ok
    # contrarian Over WITH edge still passes (model 46% vs market 30%)
    assert total_favor_gate("Total", "Over 2.5", 0.46, 16.0) == (True, [])
    # model favours the side -> passes even with a negative edge
    assert total_favor_gate("BTTS", "BTTS Yes", 0.55, -2.7) == (True, [])
    # 1X2 / AH untouched, missing input never invents a veto
    assert total_favor_gate("1X2", "Home Win", 0.46, -1.8) == (True, [])
    assert total_favor_gate("Asian Handicap", "Away +0.25", 0.46, -3.0) == (True, [])
    assert total_favor_gate("Total", "Over 2.5", None, -1.0) == (True, [])


# --------------------------------------------------------------------------
# classify_failure -- K6 / K7 / K8
# --------------------------------------------------------------------------

def _snap(**over):
    s = {"model_probs": {"1x2": {"home": 0.55, "draw": 0.22, "away": 0.23}, "lambda_home": 1.5,
                         "lambda_away": 0.9, "elo_home_seeded": True, "elo_away_seeded": True,
                         "elo_home": 1800.0, "elo_away": 1700.0},
         "context_data": None, "features": {}, "league": "EPL"}
    s.update(over)
    return s


def test_classify_failure_new_classes():
    k8 = {"market": "1X2", "selection": "Home Win", "score": 0.524, "confidence": "MEDIUM",
          "tier": "BEST PICK", "model_prob": 0.43, "edge_pp": 0.03,
          "stability": {"status": "held", "suppressed_top": {"selection": "BTTS Yes", "score": 0.67}}}
    assert classify_failure(_snap(), k8, "loss") == "K8"
    k6 = {"market": "1X2", "selection": "Home Win", "score": 0.651, "confidence": "HIGH",
          "tier": "BEST PICK", "model_prob": 0.548, "edge_pp": 1.13}
    assert classify_failure(_snap(model_probs={**_snap()["model_probs"], "lambda_home": 1.409, "lambda_away": 1.43}), k6, "loss") == "K6"
    k7 = {"market": "1X2", "selection": "Home Win", "score": 0.591, "confidence": "MEDIUM",
          "tier": "BEST PICK", "model_prob": 0.466, "edge_pp": -1.78}
    assert classify_failure(_snap(), k7, "loss") == "K7"
    # old row without model_prob: falls back to the ranking entry
    k7_old = {"market": "1X2", "selection": "Home Win", "score": 0.591, "confidence": "MEDIUM", "tier": "BEST PICK"}
    snap_old = _snap(signal_engine_ranking=[{"market": "1X2", "selection": "Home Win", "model_prob": 0.466, "edge_pp": -1.78}])
    assert classify_failure(snap_old, k7_old, "loss") == "K7"
    k0 = {"market": "1X2", "selection": "Home Win", "score": 0.689, "confidence": "HIGH",
          "tier": "BEST PICK", "model_prob": 0.647, "edge_pp": -1.38}
    assert classify_failure(_snap(), k0, "loss") == "K0"
    assert classify_failure(_snap(), k0, "win") is None
    # league band: a legit 1238 reserve-XI rating is K2 under the senior band
    # but NOT under the Eerste Divisie band
    jong = {"market": "1X2", "selection": "Home Win", "score": 0.7, "confidence": "HIGH", "tier": "BEST PICK",
            "model_prob": 0.7, "edge_pp": 1.0}
    s_j = _snap(model_probs={**_snap()["model_probs"], "elo_home": 1353.0, "elo_away": 1238.0})
    assert classify_failure(s_j, jong, "loss") == "K2"
    assert classify_failure(s_j, jong, "loss", elo_band=(1150.0, 2450.0)) == "K0"


# --------------------------------------------------------------------------
# P4 -- canonical Elo lookup
# --------------------------------------------------------------------------

def _elo(tmp: Path) -> EloModel:
    elo = EloModel(path=tmp / "elo.json")
    elo.ratings = {"Lille OSC": 2027.0, "Lillestrøm SK": 1500.0, "KRC Genk": 1879.0,
                   "Jong Genk": 1300.0, "Paris Saint-Germain": 2315.0}
    elo.games = {k: 30 for k in elo.ratings}
    elo._rebuild_indexes()
    return elo


def test_resolve_first_uses_canonical_before_display_name(tmp_path):
    elo = _elo(tmp_path)
    # the K2 single-token guard still refuses the bare display name ...
    assert elo.resolve("Lille") is None and elo.resolve("Genk") is None
    # ... but the canonical teams.json name resolves exactly
    assert elo.resolve_first(("Lille OSC", "Lille")) == "Lille OSC"
    assert elo.resolve((None, "KRC Genk", "Genk")) == "KRC Genk"
    assert elo.rating(("KRC Genk", "Genk")) == 1879.0
    assert elo.rating_first(("nope", "Lille")) == 1500.0
    assert elo.known(("Lille OSC", "Lille"), ("Paris Saint-Germain",))
    lh, la = elo.expected_lambdas(("Lille OSC", "Lille"), "Paris Saint-Germain")
    assert lh < la  # PSG 2315 > Lille 2027


def test_match_context_carries_elo_keys():
    ctx = build_match_context(league="Ligue 1", home="Lille", away="Paris Saint-Germain",
                              home_elo_key="Lille OSC", away_elo_key="Paris Saint-Germain FC")
    assert ctx.home_elo_names == ("Lille OSC", "Lille")
    assert ctx.away_elo_names == ("Paris Saint-Germain FC", "Paris Saint-Germain")
    plain = MatchContext(league="x", home="A", away="B")
    assert plain.home_elo_names == ("A",) and plain.away_elo_names == ("B",)


# --------------------------------------------------------------------------
# M1 / M2 -- market anchor + evidence-weighted ensemble
# --------------------------------------------------------------------------

def test_market_anchor_alpha_by_evidence():
    cfg = {"alpha_both_seeded": 0.5, "alpha_one_prior": 0.25, "alpha_none": 0.0}
    assert market_anchor_alpha(cfg, True, True) == 0.5
    assert market_anchor_alpha(cfg, True, False) == 0.25
    assert market_anchor_alpha(cfg, False, False) == 0.0
    assert market_anchor_alpha({}, True, True) == 0.5  # defaults
    assert market_anchor_alpha({"alpha_both_seeded": "x"}, True, True) == 1.0  # bad value -> model only


def test_apply_market_anchor_mixes_and_keeps_raw_when_no_market():
    p = {"home": 0.80, "draw": 0.12, "away": 0.08}
    mk = {"home": 0.60, "draw": 0.22, "away": 0.18}
    totals = {"over_1.5": 0.8, "over_2.5": 0.40, "over_3.5": 0.2, "btts_yes": 0.40}
    mt = {"Over 2.5": {"odds": 1.8}, "Under 2.5": {"odds": 2.0}, "BTTS Yes": {"odds": 1.7}, "BTTS No": {"odds": 2.1}}
    p2, t2, applied = apply_market_anchor(p, totals, mk, mt, alpha=0.5)
    assert applied
    assert abs(p2["home"] - 0.70) < 1e-9 and abs(sum(p2.values()) - 1.0) < 1e-9
    fair_over = (1 / 1.8) / (1 / 1.8 + 1 / 2.0)
    assert abs(t2["over_2.5"] - (0.5 * 0.40 + 0.5 * fair_over)) < 1e-9
    assert t2["over_1.5"] == 0.8 and t2["over_3.5"] == 0.2  # untouched (no pair)
    assert apply_market_anchor(p, totals, None, mt, alpha=0.5) == (p, totals, False)
    assert apply_market_anchor(p, totals, mk, mt, alpha=1.0) == (p, totals, False)


def _ctx(**over):
    base = dict(
        league="Ligue 1", home="Lille", away="Paris Saint-Germain", kickoff="2026-08-28T18:45:00Z",
        stats={"home_form": "W-L-L-W-W", "away_form": "W-D-D-D-D",
               "home_gf_avg": 1.5, "home_ga_avg": 1.2, "away_gf_avg": 2.1, "away_ga_avg": 0.8,
               "home_recent_goals": [(1, 0), (0, 2), (1, 3), (2, 1), (2, 0)],
               "away_recent_goals": [(3, 0), (1, 1), (2, 2), (1, 1), (0, 0)]},
        odds={"has_odds": True, "consensus": {"home": 4.5, "draw": 3.9, "away": 1.75},
              "totals": {"Over 2.5": {"odds": 1.63}, "Under 2.5": {"odds": 2.3},
                         "BTTS Yes": {"odds": 1.6}, "BTTS No": {"odds": 2.3}}},
        home_elo_key="Lille OSC", away_elo_key="Paris Saint-Germain",
    )
    base.update(over)
    return build_match_context(**base)


def test_run_prediction_engine_anchors_toward_market(tmp_path):
    elo = _elo(tmp_path)
    poisson = PoissonModel()
    anchored = Ensemble(elo_weight=0.6, poisson_weight=0.4,
                        market_anchor={"enabled": True, "alpha_both_seeded": 0.5})
    plain = Ensemble(elo_weight=0.6, poisson_weight=0.4)
    r_a = run_prediction_engine(_ctx(), elo=elo, poisson=poisson, ensemble=anchored).model_probs
    r_p = run_prediction_engine(_ctx(), elo=elo, poisson=poisson, ensemble=plain).model_probs
    # P4: both sides resolved through the canonical names
    assert r_a["elo_home_seeded"] and r_a["elo_away_seeded"] and r_a["elo_home"] == 2027.0
    assert r_a["market_anchor_alpha"] == 0.5 and r_a["market_anchor_applied"]
    assert r_p["market_anchor_alpha"] is None and not r_p["market_anchor_applied"]
    # raw == the un-anchored engine's numbers; anchored sits between raw and market
    assert r_a["raw"]["1x2"] == r_p["1x2"]
    mk = {k: 1 / v for k, v in {"home": 4.5, "draw": 3.9, "away": 1.75}.items()}
    tot = sum(mk.values())
    mk = {k: v / tot for k, v in mk.items()}
    for k in ("home", "draw", "away"):
        lo, hi = sorted((r_p["1x2"][k], mk[k]))
        assert lo - 1e-6 <= r_a["1x2"][k] <= hi + 1e-6
    assert abs(sum(r_a["1x2"].values()) - 1.0) < 1e-3
    assert set(r_a["components_1x2"]) >= {"elo", "poisson", "market"}


def test_ensemble_drops_elo_when_both_sides_on_prior(tmp_path):
    elo = _elo(tmp_path)
    poisson = PoissonModel()
    ens = Ensemble(elo_weight=0.6, poisson_weight=0.4)
    known = ens.predict(_ctx(), elo, poisson)
    unknown = ens.predict(_ctx(home="Nobody FC", away="Nowhere United",
                               home_elo_key=None, away_elo_key=None), elo, poisson)
    assert set(known["models"]) == {"elo", "poisson"}
    assert unknown["models"] == ["poisson"]
    assert abs(unknown["weights"]["poisson"] - 1.0) < 1e-9
    # Elo stays the only component when no feature Poisson exists
    bare = ens.predict(MatchContext(league="x", home="Nobody FC", away="Nowhere United"), elo, poisson)
    assert bare is None or bare["models"] == ["elo"]
