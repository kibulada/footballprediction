"""Tests for the transparent Decision Engine (master-prompt S18/S22-S31)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import decision, predictor
from agents.football.calibration import SignalScorer, _decisiveness


def _full_candidates(**overrides):
    """Independent model vs market with an honest +4pp edge on Under 3.5."""
    model_probs = {
        "1x2": {"home": 0.55, "draw": 0.25, "away": 0.20},
        "over_2.5": 0.55,
        "over_3.5": 0.33,
        "btts_yes": 0.52,
    }
    consensus = {"home": 1.80, "draw": 3.90, "away": 4.60}
    market_totals = {
        "Over 2.5": {"odds": 1.95},
        "Under 2.5": {"odds": 1.90},
        "Over 3.5": {"odds": 2.80},
        "Under 3.5": {"odds": 1.65},
        "BTTS Yes": {"odds": 1.80},
        "BTTS No": {"odds": 2.05},
    }
    cands = decision.build_candidates(
        model_probs=model_probs, consensus_odds=consensus,
        market_totals=market_totals, independent=True,
    )
    assert cands, "candidates must be built"
    return cands


# --------------------------------------------------------------------------
# Normalization / EV math
# --------------------------------------------------------------------------

def test_margin_free_implied_removes_overround():
    odds = {"home": 2.10, "draw": 3.40, "away": 3.60}
    imp = decision.margin_free_implied(odds)
    assert imp is not None
    assert math.isclose(sum(imp.values()), 1.0, abs_tol=1e-9)
    # removing the overround renormalizes: fair home prob = raw / overround,
    # and the overround is > 1, so fair prob is BELOW raw 1/odds
    assert math.isclose(imp["home"], 0.4543, abs_tol=1e-3)
    assert imp["home"] < 1.0 / 2.10


def test_margin_free_implied_none_when_bad():
    assert decision.margin_free_implied({}) is None
    assert decision.margin_free_implied({"home": 0, "draw": 0, "away": 0}) is None


def test_fair_pair_implied_sums_to_one():
    o, u = decision.fair_pair_implied(1.95, 1.90)
    assert math.isclose(o + u, 1.0, abs_tol=1e-9)
    assert decision.fair_pair_implied(0, 1.90) is None


def test_ev_formula():
    cands = _full_candidates()
    for c in cands:
        assert math.isclose(c.ev, c.model_prob * c.market_odds - 1.0, abs_tol=1e-9)


def test_excess_probability_bounds():
    assert decision.excess_probability(1 / 3, 3) == 0.0
    assert decision.excess_probability(1.0, 3) == 1.0
    assert decision.excess_probability(0.5, 2) == 0.0
    assert decision.excess_probability(1.0, 2) == 1.0


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------

def test_build_candidates_covers_markets():
    cands = _full_candidates()
    markets = {c.market for c in cands}
    assert "1X2" in markets and "Total" in markets and "BTTS" in markets
    # 1X2 implied sums to 1 margin-free
    h = next(c for c in cands if c.selection == "Home Win")
    d = next(c for c in cands if c.selection == "Draw")
    a = next(c for c in cands if c.selection == "Away Win")
    assert math.isclose(h.implied_prob + d.implied_prob + a.implied_prob, 1.0, abs_tol=1e-9)


def test_build_candidates_no_odds_yields_empty():
    cands = decision.build_candidates(
        model_probs={"1x2": {"home": 0.5, "draw": 0.27, "away": 0.23}},
        consensus_odds={"home": 0, "draw": 0, "away": 0},
        market_totals={}, independent=True,
    )
    assert cands == []


# --------------------------------------------------------------------------
# Extreme edge protection (S18)
# --------------------------------------------------------------------------

def test_edge_level_flags():
    assert decision.edge_level(5.0, 10.0, 20.0) == "none"
    assert decision.edge_level(12.0, 10.0, 20.0) == "warning"
    assert decision.edge_level(25.0, 10.0, 20.0) == "extreme"
    assert decision.edge_level(-25.0, 10.0, 20.0) == "extreme"


def test_extreme_edge_caps_value_and_type():
    cands = _full_candidates()
    # force one candidate to look extreme
    for c in cands:
        if c.selection == "Under 3.5":
            c.edge_pp = 25.0
            c.ev = 0.45
    d = decision.decide(
        cands, model_agreement=0.85, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
        edge_warning_pp=10.0, edge_extreme_pp=20.0,
    )
    assert d["edge_warnings"], "extreme edge must produce an audit warning"
    fd = d["final_decision"]
    # value credit capped and type capped at LEAN for an extreme edge (S18)
    if fd is not None:
        assert fd.edge_level == "extreme"
        assert fd.components["market_value"] <= decision.EXTREME_VALUE_CAP
        assert d["decision_type"] in ("LEAN", "NO CLEAR DECISION", "NO BET")


# --------------------------------------------------------------------------
# NO CLEAR DECISION / NO BET (S27)
# --------------------------------------------------------------------------

def test_no_clear_decision_on_bad_agreement():
    cands = _full_candidates()
    d = decision.decide(
        cands, model_agreement=0.20, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    assert d["decision_type"] == "NO CLEAR DECISION"
    assert d["final_decision"] is None
    assert any("agreement" in r for r in d["reasons"])


def test_no_clear_decision_on_low_completeness():
    cands = _full_candidates()
    d = decision.decide(
        cands, model_agreement=0.8, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.3, bookmakers_count=10,
    )
    assert d["decision_type"] == "NO CLEAR DECISION"
    assert any("completeness" in r for r in d["reasons"])


def test_no_clear_decision_on_empty_candidates():
    d = decision.decide(
        [], model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    assert d["decision_type"] == "NO CLEAR DECISION"
    assert d["most_likely"] is None


def test_no_bet_when_no_positive_ev():
    """S19/S27: when the model agrees with the market, no EV exists -> NO BET."""
    odds = {"home": 1.50, "draw": 4.00, "away": 6.00}
    imp = decision.margin_free_implied(odds)  # model == market -> all EV <= 0
    cands = decision.build_candidates(
        model_probs={"1x2": imp}, consensus_odds=odds, market_totals={},
        independent=True,
    )
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    assert d["most_likely"].selection == "Home Win"
    assert d["decision_type"] == "NO BET"
    assert d["final_decision"] is None


def test_non_independent_candidate_no_value():
    """S25: odds-derived (market mirror) candidates carry NO market value."""
    cands = _full_candidates()
    for c in cands:
        c.independent = False
        c.ev = 0.3
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    assert d["decision_type"] == "NO BET"
    assert d["final_decision"] is None


# --------------------------------------------------------------------------
# Most likely vs best decision (S22/S30/S39)
# --------------------------------------------------------------------------

def test_most_likely_vs_best_decision_separated():
    """Home is most likely; a totals selection with better risk-adjusted
    combination must be the final decision, with an explanation."""
    cands = _full_candidates()
    d = decision.decide(
        cands, model_agreement=0.85, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    assert d["most_likely"].selection == "Home Win"
    assert d["final_decision"] is not None
    fd = d["final_decision"]
    assert "Home Win" in d["explanation"]
    # the engine must be able to say both sentences (S39)
    assert "most likely" in d["explanation"].lower() or "Most likely" in d["explanation"]


def test_decide_returns_breakdown():
    cands = _full_candidates()
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    top = d["score_breakdown"]["top"]
    assert set(top["components"].keys()) == {
        "probability_quality", "calibration_reliability", "model_agreement",
        "market_value", "data_quality", "odds_quality", "historical_reliability",
    }
    # weights documented by default (S24 initial config)
    total = sum(decision.DEFAULT_WEIGHTS.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_decision_to_dict_json_safe():
    """The runner emits the payload via json.dumps: Candidate objects must be
    converted to plain dicts (regression: non-serializable dataclasses)."""
    import json as _json

    cands = _full_candidates()
    d = decision.decide(
        cands, model_agreement=0.85, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    payload = decision.decision_to_dict(d)
    dumped = _json.dumps(payload, ensure_ascii=False)
    assert dumped
    assert payload["decision_type"] in decision.DECISION_TYPES
    fd = payload["final_decision"]
    assert fd is None or isinstance(fd, dict)
    ml = payload["most_likely"]
    assert ml is None or isinstance(ml, dict)


def test_decision_score_weights_documented():
    w = decision.DEFAULT_WEIGHTS
    assert w["probability_quality"] == 0.30
    assert w["calibration_reliability"] == 0.20
    assert w["model_agreement"] == 0.15
    assert w["market_value"] == 0.15
    assert w["data_quality"] == 0.10
    assert w["odds_quality"] == 0.05
    assert w["historical_reliability"] == 0.05


# --------------------------------------------------------------------------
# best_prob_only (S24/S37 validated option)
# --------------------------------------------------------------------------

def test_best_prob_only_credits_value_only_to_market_favourites():
    """Walk-forward (EPL 2022-26) showed that crediting market value to
    long-shot sides with large edges bets on noise. In best_prob_only mode
    value is credited ONLY to the favourite of each market (most likely 1X2
    + favoured side of each pair), never to long-shots."""
    # custom odds: favorite Home has a real edge (>= min_edge), and a
    # long-shot Away carries a huge EV that must NOT be credited
    odds = {"home": 1.55, "draw": 4.00, "away": 6.50}
    cands = decision.build_candidates(
        # fair home implied ~0.615 at these odds -> model 0.65 = real +3.5pp
        model_probs={"1x2": {"home": 0.65, "draw": 0.22, "away": 0.13}},
        consensus_odds=odds, market_totals={}, independent=True,
    )
    for c in cands:
        if c.selection == "Away Win":
            c.edge_pp = 15.0
            c.ev = 0.55
    d = decision.decide(
        cands, model_agreement=0.85, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
        best_prob_only=True,
    )
    ml = d["most_likely"]
    assert ml.selection == "Home Win"
    away = next(c for c in cands if c.selection == "Away Win")
    assert away.components["market_value"] == 0.0, "long-shot edge must not be credited"
    assert ml.components["market_value"] > 0
    # the long-shot must not win just on edge
    assert d["final_decision"] is ml and ml.selection == "Home Win"


def test_best_prob_only_allows_totals_favourite_value():
    """S30: a favoured totals side with a real edge can still be the final
    decision even though the 1X2 favourite is most likely (live Lyon case:
    Under 3.5 @ 70.8% with +15pp edge while Home Win is 46.8%)."""
    cands = _full_candidates()
    # make the totals favourite clearly favoured and profitable
    for c in cands:
        if c.selection == "Under 3.5":
            c.model_prob = 0.71
            c.edge_pp = 15.0
            c.ev = 0.24
        if c.selection == "Over 3.5":
            c.model_prob = 0.29
    d = decision.decide(
        cands, model_agreement=0.85, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
        best_prob_only=True,
    )
    assert d["most_likely"].selection == "Home Win"
    fd = d["final_decision"]
    assert fd is not None and fd.selection == "Under 3.5"
    assert fd.components["market_value"] > 0
    assert "Under 3.5" in d["explanation"]


def test_best_prob_only_allows_no_bet_when_favorite_has_no_value():
    """S39: when the most likely outcome lacks value, the engine says NO BET
    instead of forcing a long-shot pick."""
    odds = {"home": 1.50, "draw": 4.00, "away": 6.00}
    imp = decision.margin_free_implied(odds)
    cands = decision.build_candidates(
        model_probs={"1x2": imp}, consensus_odds=odds, market_totals={},
        independent=True,
    )
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
        best_prob_only=True,
    )
    assert d["decision_type"] == "NO BET"
    assert d["final_decision"] is None
    assert d["most_likely"].selection == "Home Win"


# --------------------------------------------------------------------------
# derive_picks: switch removed (S23)
# --------------------------------------------------------------------------

def test_derive_picks_ranks_by_probability_regardless_of_signal():
    """S23: the signal>=70 -> edge / else -> probability switch is REMOVED.
    Ranking must be identical for high and low signal (transparent)."""
    consensus = {"home": 1.70, "draw": 4.00, "away": 5.20}
    market_totals = {
        "Over 2.5": {"odds": 1.90}, "Under 2.5": {"odds": 1.95},
    }
    high = predictor.derive_picks(consensus, market_totals, signal=90)
    low = predictor.derive_picks(consensus, market_totals, signal=10)
    assert [p["selection"] for p in high["top_picks"]] == \
           [p["selection"] for p in low["top_picks"]]
    # ranked by model_prob descending (most likely first)
    probs = [p["model_prob"] for p in high["top_picks"]]
    assert probs == sorted(probs, reverse=True)


def test_derive_picks_reference_only_no_edge_ev():
    # Model A rule: odds-derived derive_picks is reference-only -- edge/EV are
    # never computed here (the independent engine is the only edge/EV source).
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    result = predictor.derive_picks(consensus, {}, signal=50)
    assert result["top_picks"]
    for p in result["top_picks"]:
        assert "ev" in p
        assert "edge" in p
        assert p["ev"] == 0.0
        assert p["edge"] == 0.0


# --------------------------------------------------------------------------
# Confidence + decisiveness (S26)
# --------------------------------------------------------------------------

def test_decisiveness_math():
    assert _decisiveness({"home": 0.50, "draw": 0.27, "away": 0.23}) > \
           _decisiveness({"home": 0.38, "draw": 0.33, "away": 0.29})


def _rich_ctx():
    from agents.football.context import MatchContext

    # Correction-spec weights: odds 0.25 + form 0.20 + attack/defense 0.20 +
    # xG 0.20 = 0.85 -> MEDIUM/HIGH (cap 1.0). H2H absent stays neutral here.
    return MatchContext(
        league="EPL", home="A", away="B",
        home_form="W", away_form="L",
        home_gf_avg=1.8, home_ga_avg=1.1, away_gf_avg=1.4, away_ga_avg=1.2,
        home_xg_for=1.6, home_xg_against=1.1,
        away_xg_for=1.2, away_xg_against=1.4,
        consensus_odds={"home": 1.5, "draw": 4.5, "away": 6.0},
    )


def test_signal_scorer_confidence_includes_decisiveness():
    scorer = SignalScorer()
    ctx = _rich_ctx()
    base = scorer.components(
        ctx=ctx, ensemble_models=["elo", "poisson"],
        model_vs_market=0.9, model_vs_model=0.9,
        calibration_quality=0.9, market_edge={"home": 5.0, "draw": -2.0, "away": -3.0},
    )
    comps = scorer.components(
        ctx=ctx, ensemble_models=["elo", "poisson"],
        model_vs_market=0.9, model_vs_model=0.9,
        calibration_quality=0.9, market_edge={"home": 5.0, "draw": -2.0, "away": -3.0},
        p1x2={"home": 0.80, "draw": 0.12, "away": 0.08},
    )
    assert "decisiveness" in comps
    assert comps["decisiveness"] > 0.5  # very decisive 1X2
    # S26: a decisive probability lifts confidence above the same match
    # without the probability input
    assert comps["confidence"] > base["confidence"]
    assert comps["confidence"] > 0.7


def test_signal_scorer_low_decisiveness_lower_confidence():
    scorer = SignalScorer()
    from agents.football.context import MatchContext

    ctx = MatchContext(
        league="EPL", home="A", away="B",
        home_form="W", away_form="L",
        consensus_odds={"home": 2.6, "draw": 3.2, "away": 2.9},
    )
    comps = scorer.components(
        ctx=ctx, ensemble_models=["elo", "poisson"],
        model_vs_market=0.9, model_vs_model=0.9,
        calibration_quality=0.9, market_edge={"home": 1.0, "draw": -0.5, "away": -0.5},
        p1x2={"home": 0.37, "draw": 0.33, "away": 0.30},
    )
    assert comps["decisiveness"] < 0.3


# --------------------------------------------------------------------------
# P1: form-depth floor (shallow form window bans STRONG)
# --------------------------------------------------------------------------

def _strong_fixture_candidates():
    """Under 3.5 with model 0.72 / implied 0.53 (edge 19pp, not extreme) ->
    the decision reaches STRONG without caps."""
    cands = _full_candidates()
    for c in cands:
        if c.selection == "Under 3.5":
            c.model_prob = 0.72
            c.implied_prob = 0.53
            c.edge_pp = 19.0
            c.ev = 0.72 * 1.65 - 1.0
            c.market_odds = 1.65
    return cands


def test_form_depth_shallow_bans_strong():
    """P1: a form window < 3 matches/tim must never produce STRONG — the
    type is capped at GOOD with an explicit reason (thin form is noise, not
    signal)."""
    cands = _strong_fixture_candidates()
    kw = dict(
        model_agreement=0.95, calibration_quality=0.95,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    # sanity: the same fixture legitimately reaches STRONG without the cap
    d_base = decision.decide(cands, **kw)
    assert d_base["decision_type"] == "STRONG"
    d = decision.decide(cands, **kw, form_depth_shallow=True)
    assert d["decision_type"] == "GOOD"
    assert any("form window terlalu dangkal" in r for r in d["reasons"])
    assert d["form_depth_cap_applied"] is True
    # confidence is capped at MEDIUM at most (without bucket_n the raw tier
    # is LOW, which the ceiling leaves untouched — never HIGH).
    psc = d["pick_specific_confidence"]
    assert psc["label"] in ("LOW", "MEDIUM")
    assert any("form" in c and "MEDIUM" in c for c in psc["caps"])


def test_form_depth_shallow_leaves_good_untouched():
    """The floor only bans STRONG; GOOD/LEAN decisions pass through."""
    cands = _strong_fixture_candidates()
    kw = dict(
        model_agreement=0.95, calibration_quality=0.95,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    # force the type below STRONG by lowering completeness contribution
    d_base = decision.decide(
        cands, model_agreement=0.6, calibration_quality=0.8,
        calibration_samples=5000, completeness=0.7, bookmakers_count=10,
    )
    assert d_base["decision_type"] in ("GOOD", "LEAN")
    d = decision.decide(
        cands, model_agreement=0.6, calibration_quality=0.8,
        calibration_samples=5000, completeness=0.7, bookmakers_count=10,
        form_depth_shallow=True,
    )
    assert d["decision_type"] == d_base["decision_type"]


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
