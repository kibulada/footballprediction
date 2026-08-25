"""Tests for the per-market tier engine (market_tiers.py).

Tiers are per MARKET (1X2, O/U 2.5, O/U 3.5, BTTS) — a single match can mix
PICK / LEAN / WATCH across its markets. The engine is PURE: tier comes only
from computed confidence/disagreement/edge/completeness values, never from
user requests.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.market_tiers import (  # noqa: E402
    TIER_LEAN,
    TIER_PICK,
    TIER_WATCH,
    _basis,
    data_completeness_level,
    market_confidence,
    market_tier,
    market_view,
    render_market_tiers,
    render_single_pick,
    select_best_pick,
)

# Home odds 2.25 so the fixture's 1X2 PICK (model home 0.47) has positive
# EV (0.47 * 2.25 - 1 = +5.75% > MIN_EV 3%). A negative-EV PICK must never
# exist (invariant 8.1) -- see the ev_nonpositive demotion tests below.
CONSENSUS = {"home": 2.25, "draw": 3.40, "away": 4.20}
TOTALS = {
    "Over 2.5": {"odds": 1.85},
    "Under 2.5": {"odds": 1.95},
    "Over 3.5": {"odds": 2.40},
    "Under 3.5": {"odds": 1.60},
    "BTTS Yes": {"odds": 1.75},
    "BTTS No": {"odds": 2.05},
}
MODEL_A = {
    "1x2": {"home": 0.50, "draw": 0.27, "away": 0.23},
    "over_2.5": 0.52,
    "over_3.5": 0.30,
    "btts_yes": 0.53,
}


def _model_probs(**overrides):
    mp = {
        "1x2": {"home": 0.47, "draw": 0.27, "away": 0.26},
        "over_2.5": 0.55,
        "over_3.5": 0.34,
        "btts_yes": 0.58,
    }
    mp.update(overrides)
    return mp


def _view(label, model_probs=None, consensus=CONSENSUS, totals=TOTALS,
          model_a=MODEL_A, disagreement_1x2_pp=3.0):
    return market_view(
        label=label,
        model_probs=model_probs if model_probs is not None else _model_probs(),
        consensus=consensus,
        totals=totals,
        model_a=model_a,
        disagreement_1x2_pp=disagreement_1x2_pp,
    )


# ---- market_view ---------------------------------------------------------


def test_1x2_view_edges_and_outcome():
    v = _view("1X2")
    assert v["evaluated"] is True
    # most likely side is the outcome
    assert v["outcome"] == "Home Win"
    assert v["disagreement_pp"] == 3.0
    # no selection |edge| >= 10pp -> no extremes, no larges, no contradiction
    assert v["extremes"] == []
    assert v["larges"] == []
    assert v["contradiction"] is False
    # outcome EV at the offered odds is carried on the view
    assert v["outcome_ev"] == 0.47 * 2.25 - 1.0


def test_view_1x2_without_odds_is_not_evaluated():
    v = _view("1X2", consensus={"home": 0, "draw": 0, "away": 0})
    assert v["evaluated"] is False


def test_view_pair_without_model_prob_is_not_evaluated():
    v = _view("BTTS", model_probs=_model_probs(btts_yes=None))
    assert v["evaluated"] is False


def test_view_pair_without_odds_pair_is_not_evaluated():
    v = _view("Over/Under 3.5", totals={"Over 3.5": {"odds": 2.4}})
    assert v["evaluated"] is False


def test_view_contradiction_when_both_sides_extreme():
    # 0.74 model vs ~0.513 margin-free implied -> Over +22.7 / Under -22.7pp
    v = _view("Over/Under 2.5", model_probs=_model_probs(**{"over_2.5": 0.74}))
    assert v["contradiction"] is True
    assert len(v["extremes"]) == 2


def test_view_single_extreme_in_1x2_is_not_contradiction():
    # one side extreme, the other two small -> not a contradiction
    v = _view("1X2", model_probs=_model_probs(
        **{"1x2": {"home": 0.72, "draw": 0.16, "away": 0.12}}))
    assert len(v["extremes"]) == 1
    assert v["contradiction"] is False


# ---- tier assignment -----------------------------------------------------


def test_tier1_pick_high_conf_small_disagreement():
    v = _view("1X2")
    assert market_tier(v, "HIGH", "High") == TIER_PICK


def test_tier2_lean_medium_confidence():
    v = _view("1X2")
    assert market_tier(v, "MEDIUM", "High") == TIER_LEAN


def test_tier2_lean_large_edge_no_contradiction():
    # Home +10.5pp (in [10, SANITY_PP=12)) -> large-but-plausible edge, no
    # contradiction -> LEAN. Edges >= 12pp are a data-mismatch signal (P2)
    # and demote to WATCH instead (see test_sanity_risk_demotes_...).
    v = _view("1X2", model_probs=_model_probs(
        **{"1x2": {"home": 0.56, "draw": 0.27, "away": 0.26}}))
    assert v["contradiction"] is False
    assert v["sanity_risk"] is False
    assert market_tier(v, "HIGH", "High") == TIER_LEAN


def test_tier3_watch_contradictory_extremes():
    v = _view("Over/Under 2.5", model_probs=_model_probs(**{"over_2.5": 0.74}))
    assert market_tier(v, "HIGH", "High") == TIER_WATCH


def test_tier3_watch_disagreement_over_20pp():
    v = _view("1X2", disagreement_1x2_pp=25.0)
    assert market_tier(v, "HIGH", "High") == TIER_WATCH


def test_tier3_watch_low_confidence_and_low_completeness():
    v = _view("1X2")
    assert market_tier(v, "LOW", "High") == TIER_WATCH
    assert market_tier(v, "HIGH", "Low") == TIER_WATCH


def test_tier1_requires_no_contradiction_and_medium_or_high_completeness():
    v = _view("1X2", disagreement_1x2_pp=5.0)
    assert market_tier(v, "HIGH", "Medium") == TIER_PICK
    assert market_tier(v, "HIGH", "Low") == TIER_WATCH
    assert market_tier(v, "MEDIUM", "High") == TIER_LEAN


def test_extreme_edge_caps_market_confidence_to_low():
    v = _view("1X2", model_probs=_model_probs(
        **{"1x2": {"home": 0.72, "draw": 0.16, "away": 0.12}}))
    assert market_confidence("HIGH", v) == "LOW"


def test_contradiction_caps_market_confidence_to_low():
    v = _view("Over/Under 2.5", model_probs=_model_probs(**{"over_2.5": 0.74}))
    assert market_confidence("HIGH", v) == "LOW"


def test_large_edge_keeps_global_confidence():
    v = _view("1X2", model_probs=_model_probs(
        **{"1x2": {"home": 0.62, "draw": 0.22, "away": 0.16}}))
    assert market_confidence("HIGH", v) == "HIGH"


# ---- completeness --------------------------------------------------------


def test_data_completeness_level_rubric():
    payload = {
        "stats": {
            "home_form": "W-W-D",
            "away_form": "L-D-W",
            "home_gf_avg": 1.8,
            "away_gf_avg": 1.3,
            "h2h": {"wins": 2, "draws": 1, "losses": 1},
        },
        "odds": {"has_odds": True},
    }
    assert data_completeness_level(payload) == ("High", ["xG"])
    assert data_completeness_level({}) == ("Low", ["odds", "form", "GF/GA", "xG", "H2H"])


# ---- render --------------------------------------------------------------


def _render_payload(**overrides):
    payload = {
        "league": "EPL",
        "home": "A",
        "away": "B",
        "kickoff": None,
        "stats": {
            "home_form": "W",
            "away_form": "W",
            "home_gf_avg": 1.5,
            "away_gf_avg": 1.2,
            "h2h": {"wins": 1, "draws": 0, "losses": 0},
        },
        "odds": {"has_odds": True, "consensus": CONSENSUS, "totals": TOTALS},
        "prediction": {"model_probs": _model_probs(), "data_completeness": 0.8},
        "decision": {
            "model_a": MODEL_A,
            "model_disagreement": {"flag": False, "delta_pp": 3.0},
        },
        "confidence": {"tier": "HIGH"},
    }
    payload.update(overrides)
    return payload


def test_render_shows_all_four_markets_with_own_tiers():
    lines = render_market_tiers(_render_payload())
    text = "\n".join(lines)
    assert text.count("── ") == 4  # one header per market
    for hdr in ("── 1X2 ──", "── Over/Under 2.5 ──", "── Over/Under 3.5 ──", "── BTTS ──"):
        assert hdr in text
    # every evaluated market carries tier + confidence/basis + stake
    assert "🟢 PICK: Home Win @ 2.25" in text
    assert "Stake: Normal 1 unit" in text
    assert "Basis:" in text
    assert "Data completeness:" in text
    # the renderer owns no header and no disclaimer (callers add those once)
    assert "Not a guarantee of outcome" not in text


def test_render_not_evaluated_when_no_odds_at_all():
    lines = render_market_tiers(_render_payload(
        odds={"has_odds": False, "consensus": {}, "totals": {}}))
    text = "\n".join(lines)
    assert text.count("❌ Not evaluated — insufficient data for this market") == 4


def test_render_contradiction_stated_in_basis():
    mp = _model_probs(**{"over_2.5": 0.74})
    lines = render_market_tiers(_render_payload(prediction={"model_probs": mp}))
    text = "\n".join(lines)
    assert "Over 2.5 extreme edge +22.7pp AND Under 2.5 extreme edge -22.7pp" in text
    assert "contradictory" in text


# ---- EV gate (invariant 8.1, tier layer) ---------------------------------


def test_pick_demoted_to_watch_when_ev_nonpositive():
    # All PICK criteria hold (HIGH confidence, dis < 8pp, no contradiction)
    # but the outcome's EV at the offered odds is negative -> WATCH: a stake
    # must never ride on a negative-EV selection (regression: the bot once
    # rendered "🟢 PICK: Home Win @ 2.05 ... Stake: Normal 1 unit" while the
    # engine's Pick Evaluation said "NO VALUE: EV -3.1%").
    v = _view("1X2", consensus={"home": 1.95, "draw": 3.40, "away": 4.20})
    assert v["outcome_ev"] < 0.0
    assert market_tier(v, "HIGH", "High") == TIER_WATCH
    assert v["ev_nonpositive"] is True


def test_lean_demoted_to_watch_when_ev_nonpositive():
    # LEAN criteria hold (disagreement in the 8-20pp band) but EV is negative
    # -> WATCH as well (micro stakes are still stakes).
    v = _view("1X2", consensus={"home": 1.95, "draw": 3.40, "away": 4.20},
              disagreement_1x2_pp=12.0)
    assert v["disagreement_pp"] == 12.0  # LEAN band
    assert v["outcome_ev"] < 0.0
    assert market_tier(v, "HIGH", "High") == TIER_WATCH
    assert v["ev_nonpositive"] is True


def test_pair_pick_demoted_when_edge_positive_but_ev_small():
    # Over edge is positive (outcome = Over) but EV at 1.85 is negative
    # (0.53 * 1.85 - 1 = -2.0% <= MIN_EV) -> WATCH.
    v = _view("Over/Under 2.5", model_probs=_model_probs(**{"over_2.5": 0.53}))
    assert v["outcome"] == "Over 2.5"
    assert v["outcome_ev"] <= 0.03
    assert market_tier(v, "HIGH", "High") == TIER_WATCH
    assert v["ev_nonpositive"] is True


def test_positive_ev_pick_unchanged():
    v = _view("1X2")
    assert v["outcome_ev"] > 0.03
    assert market_tier(v, "HIGH", "High") == TIER_PICK
    assert not v.get("ev_nonpositive")


def test_render_demoted_market_shows_skip_stake_and_ev_reason():
    payload = _render_payload()
    payload["odds"]["consensus"] = {"home": 1.95, "draw": 3.40, "away": 4.20}
    text = "\n".join(render_market_tiers(payload))
    assert "⚪ WATCH: Home Win @ 1.95" in text
    assert "Stake: SKIP" in text
    assert "EV -8.4% <= +3%" in text


# ---------------------------------------------------------------------------
# P2: model-vs-market sanity check (gap >= SANITY_PP is a data-mismatch
# signal, never value -> WATCH, never LEAN/PICK)
# ---------------------------------------------------------------------------


def test_sanity_risk_flags_large_model_market_gap():
    """Qarabağ-style mismatch: model says ~49% but the market prices 7.5
    (implied ~13%) — a ~33pp gap is flagged as a sanity risk, not value."""
    v = _view(
        "1X2",
        consensus={"home": 7.5, "draw": 5.0, "away": 1.9},
        model_probs=_model_probs(**{"1x2": {"home": 0.49, "draw": 0.20, "away": 0.31}}),
    )
    assert v["sanity_risk"] is True
    assert market_tier(v, "HIGH", "High") == TIER_WATCH


def test_sanity_risk_demotes_large_but_not_extreme_edge():
    """A 14.7pp edge (previously LEAN band 10-20pp) is now WATCH: gaps
    between 12-20pp are data-mismatch signals, never value (P2)."""
    v = _view(
        "1X2",
        consensus={"home": 2.70, "draw": 3.40, "away": 2.55},
        model_probs=_model_probs(**{"1x2": {"home": 0.491, "draw": 0.26, "away": 0.249}}),
    )
    assert v["sanity_risk"] is True
    assert market_tier(v, "HIGH", "High") == TIER_WATCH


def test_sanity_below_threshold_stays_leanable():
    """A large-but-plausible edge (11pp < SANITY_PP) is still LEAN-able; the
    sanity bar is 12pp, not 10pp."""
    v = _view(
        "1X2",
        consensus={"home": 1.90, "draw": 3.60, "away": 4.20},
        model_probs=_model_probs(**{"1x2": {"home": 0.62, "draw": 0.22, "away": 0.16}}),
        disagreement_1x2_pp=3.0,
    )
    assert v["sanity_risk"] is False
    # EV positive: 0.62 * 1.90 - 1 = +17.8% (large edge -> LEAN tier)
    assert v["outcome_ev"] > 0.03
    assert market_tier(v, "HIGH", "High") == TIER_LEAN


def test_sanity_basis_names_verification_not_value():
    """The Basis line says explicitly this is a fixture/odds verification
    signal, not value."""
    v = _view(
        "1X2",
        consensus={"home": 2.70, "draw": 3.40, "away": 2.55},
        model_probs=_model_probs(**{"1x2": {"home": 0.491, "draw": 0.26, "away": 0.249}}),
    )
    text = _basis(v, "HIGH", "High")
    assert "verifikasi fixture/odds" in text
    assert "bukan value" in text


def test_render_sanity_market_shows_watch_and_verification():
    payload = _render_payload()
    payload["odds"]["consensus"] = {"home": 2.70, "draw": 3.40, "away": 2.55}
    payload["prediction"]["model_probs"]["1x2"] = {"home": 0.491, "draw": 0.26, "away": 0.249}
    text = "\n".join(render_market_tiers(payload))
    assert "⚪ WATCH: Home Win @ 2.70" in text
    assert "verifikasi fixture/odds" in text
    assert "Stake: SKIP" in text


# ---- selection layer: no directional pick against MARKET PRIOR / noise ----

def _thin_payload():
    """Singapore-vs-Thailand style thin-data payload: every market falls to
    WATCH with a small prior-only edge, and the engine already concluded
    MARKET PRIOR (edge = 0 by construction, betting advice NO BET)."""
    payload = _render_payload()
    # small, prior-only edges: no market reaches >= 10pp
    payload["prediction"]["model_probs"] = _model_probs(
        **{"1x2": {"home": 0.20, "draw": 0.25, "away": 0.55},
           "over_2.5": 0.42, "over_3.5": 0.21, "btts_yes": 0.50})
    payload["decision"]["decision_type"] = "MARKET PRIOR"
    return payload


def test_select_best_pick_none_for_market_prior():
    payload = _thin_payload()
    assert select_best_pick(payload) is None


def test_render_single_pick_no_bet_for_market_prior():
    payload = _thin_payload()
    text = "\n".join(render_single_pick(payload))
    assert "NO BET" in text
    assert "BTTS Yes" not in text
    assert "directional lean" not in text


def test_select_best_pick_none_for_small_edge_watch_all():
    # Engine did NOT go MARKET PRIOR, but every market is WATCH with an edge
    # below WATCH_LEAN_MIN_PP (10pp) -> no directional claim.
    payload = _render_payload()
    payload["prediction"]["model_probs"] = _model_probs(
        **{"1x2": {"home": 0.49, "draw": 0.27, "away": 0.24},
           "over_2.5": 0.53, "over_3.5": 0.31, "btts_yes": 0.53})
    payload["confidence"] = {"tier": "LOW"}
    # all edges < 10pp and confidence LOW -> WATCH; selection must be None
    assert select_best_pick(payload) is None


def test_watch_directional_lean_still_surfaces_on_real_edge():
    # A genuine directional edge (>= 10pp) on a WATCH market is still surfaced
    # (not silenced), with Stake: SKIP.
    payload = _render_payload()
    payload["prediction"]["model_probs"] = _model_probs(
        **{"1x2": {"home": 0.62, "draw": 0.22, "away": 0.16}})
    payload["confidence"] = {"tier": "LOW"}
    pick = select_best_pick(payload)
    assert pick is not None
    assert pick["tier"] == "WATCH"
    assert pick["outcome"] == "Home Win"


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
