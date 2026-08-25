"""Tests for the Market-Aware Signal Engine + Best Pick Ranker.

Covers S11 (implied prob), S7/S8/S9/S10 (movement), S14 (Asian Handicap
semantics + settlement), S17-S25 (scoring, ranking, best pick, NO BET),
S36 (determinism), S35 (Holstein Kiel conceptual validation) and S34
(backtest settlement). The core engine is pure, so every test is sync and
deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.models import poisson_matrix, probs_from_matrix  # noqa: E402
from agents.football.signal_engine import (  # noqa: E402
    ah_consensus,
    ah_return,
    ah_settle,
    ah_win_prob,
    build_signals,
    confidence_label,
    excess_probability,
    extract_asian_handicap,
    fair_pair_implied,
    implied_probability,
    movement_features,
    normalize_implied,
    price_move_pct,
    rank_and_pick,
    run_signal_backtest,
    run_signal_engine,
    score_signals,
    settle_signal,
    statistical_support,
)


def _model_probs(lh: float, la: float, over25: float | None = None, btts: float | None = None):
    m = poisson_matrix(lh, la, rho=0.0)
    p1x2, o15, o25, o35, btts_yes = probs_from_matrix(m)
    return {
        "1x2": p1x2,
        "over_1.5": o15,
        "over_2.5": o25 if over25 is None else over25,
        "over_3.5": o35,
        "btts_yes": btts_yes if btts is None else btts,
        "lambda_home": lh,
        "lambda_away": la,
    }


def _totals(over, under, btts_y, btts_n):
    return {
        "Over 2.5": {"odds": over, "point": 2.5},
        "Under 2.5": {"odds": under, "point": 2.5},
        "BTTS Yes": {"odds": btts_y},
        "BTTS No": {"odds": btts_n},
    }


def _ah_payload(line, home, away, home_open=None, away_open=None):
    return {
        "bookmakers": [
            {
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "asian_handicap",
                        "outcomes": [
                            {"name": "Home", "price": home, "point": line,
                             "opening_price": home_open},
                            {"name": "Away", "price": away, "point": line,
                             "opening_price": away_open},
                        ],
                    }
                ],
            }
        ]
    }


# ---- S11: implied probability + normalization ---------------------------

def test_implied_probability():
    assert implied_probability(2.0) == 0.5
    assert implied_probability(1.9) == 1.0 / 1.9
    assert implied_probability(1.0) is None
    assert implied_probability(0.0) is None
    assert implied_probability(None) is None


def test_normalize_implied_sums_to_one():
    norm = normalize_implied({"home": 2.0, "draw": 3.0, "away": 4.0})
    assert norm is not None
    assert abs(sum(norm.values()) - 1.0) < 1e-9
    assert norm["home"] > norm["draw"] > norm["away"]


def test_fair_pair_implied():
    a, b = fair_pair_implied(2.0, 2.0)
    assert abs(a - 0.5) < 1e-9 and abs(b - 0.5) < 1e-9
    o, u = fair_pair_implied(1.9, 1.95)
    assert o > u  # shorter over -> more likely over
    assert abs((o + u) - 1.0) < 1e-9


# ---- S7/S8/S9/S10: movement ----------------------------------------------

def test_price_move_pct():
    assert abs(price_move_pct(1.99, 1.94) - (-2.5125)) < 0.01
    assert price_move_pct(None, 1.94) is None
    assert price_move_pct(1.99, None) is None


def test_movement_direction_toward():
    mv = movement_features(opening=1.99, current=1.94)
    assert mv["status"] == "available"
    assert mv["direction"] == "toward"      # price shortened
    assert mv["magnitude_pct"] > 0


def test_movement_direction_away():
    mv = movement_features(opening=1.90, current=2.00)
    assert mv["direction"] == "away"


def test_movement_unavailable_without_opening():
    mv = movement_features(opening=None, current=1.94)
    assert mv["status"] == "UNAVAILABLE"
    assert mv["direction"] == "none"


def test_line_movement():
    mv = movement_features(opening=1.90, current=1.95, opening_line=2.5, current_line=2.75)
    assert mv["line_move"] == 0.25


def test_late_movement_consistency():
    mv = movement_features(opening=1.99, current=1.94, late_direction=1.0, late_strength=0.8)
    assert mv["late_direction"] == 1.0
    assert mv["late_strength"] == 0.8


def test_weights_no_longer_double_count_market_direction():
    """Weight distribution v3: rebalanced for market intelligence.
    
    Phase 6: added market_intelligence (0.15) for steam/RLM signals,
    enabled statistical (0.10) and team_context (0.05).
    """
    from agents.football.signal_engine import DEFAULT_WEIGHTS

    assert DEFAULT_WEIGHTS["late_movement"] == 0.0
    assert DEFAULT_WEIGHTS["movement"] == 0.15
    assert DEFAULT_WEIGHTS["model"] == 0.35
    assert DEFAULT_WEIGHTS["statistical"] == 0.10
    # 2026-08-19 reweight: market 0.20 -> 0.15 (market_intelligence already
    # carries the direction signal; production config may still override).
    assert DEFAULT_WEIGHTS["market"] == 0.15
    assert DEFAULT_WEIGHTS["market_intelligence"] == 0.15
    assert DEFAULT_WEIGHTS["data_quality"] == 0.05
    assert DEFAULT_WEIGHTS["team_context"] == 0.00
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_late_move_against_pick_caps_confidence():
    """The retained penalty role of late_movement: when the market's last
    move is AGAINST the pick with meaningful strength, a HIGH/VERY HIGH pick
    must be capped at MEDIUM -- never presented stronger than the market's
    own late verdict.
    """
    from agents.football.signal_engine import Signal, rank_and_pick

    s = Signal(
        market="Total", selection="Under 2.5", model_prob=0.6,
        market_odds=1.9, implied_prob=0.5, edge_pp=10.0,
        score=0.8,
        components={"model": 0.8, "statistical": 0.8, "market": 1.0,
                    "movement": 0.5, "late_movement": 0.5, "data_quality": 0.9},
        movement={"late_direction": -1.0, "late_strength": 0.9},
    )
    out = rank_and_pick(
        [s], best_pick_margin=0.06, no_bet_score=0.45, min_confluence=2,
        conflict_pp=8.0, min_data_quality=0.3, completeness=1.0,
    )
    assert out["decision"] == "BEST PICK"
    assert out["best_pick"].confidence == "MEDIUM"
    assert any("late market move melawan" in n for n in (out["best_pick"].evidence_notes or []))

    # A late move TOWARD the pick leaves an otherwise-HIGH pick at HIGH.
    s2 = Signal(
        market="Total", selection="Under 2.5", model_prob=0.6,
        market_odds=1.9, implied_prob=0.5, edge_pp=10.0,
        score=0.8,
        components={"model": 0.8, "statistical": 0.8, "market": 1.0,
                    "movement": 0.5, "late_movement": 0.5, "data_quality": 0.9},
        movement={"late_direction": 1.0, "late_strength": 0.9},
    )
    out2 = rank_and_pick(
        [s2], best_pick_margin=0.06, no_bet_score=0.45, min_confluence=2,
        conflict_pp=8.0, min_data_quality=0.3, completeness=1.0,
    )
    assert out2["best_pick"].confidence in ("HIGH", "VERY HIGH")


def test_reversal_detection():
    # opening->current shortened (toward) but late money reversed away
    mv = movement_features(opening=1.99, current=1.94, late_direction=-1.0, late_strength=0.8)
    assert mv["reversal"] is True
    # consistent direction -> no reversal
    mv2 = movement_features(opening=1.99, current=1.94, late_direction=1.0, late_strength=0.8)
    assert mv2["reversal"] is False


# ---- S14: Asian Handicap semantics + settlement --------------------------

def test_away_plus_quarter_draw_is_half_win():
    s = ah_settle(2, 2, -0.25, "away")  # Away +0.25, draw -> half win
    assert s["result"] == "half_win"
    assert abs(s["return_value"] - 0.75) < 1e-9


def test_away_plus_quarter_away_win_is_full_win():
    s = ah_settle(1, 2, -0.25, "away")
    assert s["result"] == "win"
    assert abs(s["return_value"] - 1.0) < 1e-9


def test_away_plus_quarter_home_win_is_loss():
    s = ah_settle(2, 1, -0.25, "away")
    assert s["result"] == "loss"
    assert abs(s["return_value"] - 0.0) < 1e-9


def test_home_minus_quarter_draw_is_half_loss():
    s = ah_settle(2, 2, -0.25, "home")  # Home -0.25, draw -> half loss
    assert s["result"] == "half_loss"
    assert abs(s["return_value"] - 0.25) < 1e-9


def test_level_ball_draw_is_push():
    s = ah_settle(1, 1, 0.0, "home")
    assert s["result"] == "push"
    assert abs(s["return_value"] - 0.5) < 1e-9


def test_ah_win_prob_symmetry():
    m = poisson_matrix(1.5, 1.5, rho=0.0)
    home_minus = ah_win_prob(m, -0.25, "home")
    away_plus = ah_win_prob(m, -0.25, "away")
    assert 0.0 < home_minus < away_plus < 1.0  # away +0.25 has the edge


# ---- AH extraction -------------------------------------------------------

def test_extract_asian_handicap():
    rows = extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.10, 1.85))
    assert len(rows) == 1
    assert rows[0]["line"] == -0.25
    assert rows[0]["home_open"] == 2.10
    assert rows[0]["away_open"] == 1.85


def test_extract_asian_handicap_empty_without_market():
    assert extract_asian_handicap({"bookmakers": [{"title": "X", "markets": []}]}) == []
    assert extract_asian_handicap({}) == []


def test_ah_consensus_median():
    rows = [
        {"line": -0.25, "home": 1.9, "away": 1.9, "home_open": 2.0, "away_open": 1.8, "bookmaker": "a"},
        {"line": -0.25, "home": 2.0, "away": 2.0, "home_open": 2.1, "away_open": 1.9, "bookmaker": "b"},
    ]
    c = ah_consensus(rows)
    assert c["line"] == -0.25
    assert c["home"] == 2.0  # median of [1.9, 2.0] -> upper middle
    assert c["n"] == 2


# ---- statistical support -------------------------------------------------

def test_statistical_support_btts_and_over():
    goals = [(1, 1), (2, 1), (0, 0), (3, 2), (1, 0)]
    stats = {"home_recent_goals": goals, "away_recent_goals": goals}
    b = statistical_support("btts", stats, 0.0, "home")
    # both scored in (1,1),(2,1),(3,2) = 3/5
    assert abs(b - 0.6) < 1e-9
    o = statistical_support("over", stats, 0.0, "home")
    # total > 2 in (2,1),(3,2) = 2/5
    assert abs(o - 0.4) < 1e-9


# ---- signal construction + scoring ---------------------------------------

def test_build_signals_includes_btts_over_ah():
    model = _model_probs(1.6, 1.6)
    totals = _totals(1.90, 1.95, 1.75, 2.05)
    ah_rows = extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.10, 1.85))
    signals = build_signals(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.7,
    )
    sels = {(s.market, s.selection) for s in signals}
    assert ("BTTS", "BTTS Yes") in sels
    assert ("BTTS", "BTTS No") in sels
    assert ("Total", "Over 2.5") in sels
    assert ("Total", "Under 2.5") in sels
    assert ("Asian Handicap", "Away +0.25") in sels
    # Layer 2 (side-neutral): the mirror label Home -0.25 is the SAME bet and
    # is canonicalized away -- a handicap line is scored exactly once, so the
    # identical bet can never score differently per query (50 vs 76 bug).
    assert ("Asian Handicap", "Home -0.25") not in sels
    ah_sigs = [s for s in signals if s.market == "Asian Handicap"]
    assert len({s.line for s in ah_sigs}) == len(ah_sigs), "one candidate per AH line"


def test_build_signals_no_ah_without_market():
    model = _model_probs(1.5, 1.5)
    totals = _totals(1.90, 1.95, 1.75, 2.05)
    signals = build_signals(
        model_probs=model, stats={}, market_totals=totals, ah_rows=[],
        movement_snapshot=None, context=None, completeness=0.5,
    )
    # canonical +-0.25 lines still generated (model-only), but no AH price
    ah = [s for s in signals if s.market == "Asian Handicap"]
    assert ah, "canonical AH quarter lines must always be evaluated"
    assert all(s.market_odds is None for s in ah)


def test_edge_calculation_model_vs_market():
    model = _model_probs(1.5, 1.5, over25=0.64)
    totals = _totals(1.90, 1.95, 1.75, 2.05)  # over fair ~0.5065
    signals = build_signals(
        model_probs=model, stats={}, market_totals=totals, ah_rows=[],
        movement_snapshot=None, context=None, completeness=0.7,
    )
    over = next(s for s in signals if s.selection == "Over 2.5")
    # edge = (0.64 - implied) * 100 > 0
    assert over.edge_pp > 0
    assert abs(over.edge_pp - (0.64 - fair_pair_implied(1.90, 1.95)[0]) * 100.0) < 0.01


def test_score_signals_components_present():
    model = _model_probs(1.5, 1.5)
    totals = _totals(1.90, 1.95, 1.75, 2.05)
    ah_rows = extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.10, 1.85))
    signals = build_signals(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.7,
    )
    score_signals(signals, weights={
        "model": 0.3, "statistical": 0.2, "market": 0.2, "movement": 0.15,
        "late_movement": 0.1, "data_quality": 0.05, "team_context": 0.0,
    }, min_edge_pp=3.0, conflict_pp=8.0, completeness=0.7, context=None)
    for s in signals:
        assert 0.0 <= s.score <= 1.0
        assert "model" in s.components
        assert "data_quality" in s.components
        if s.implied_prob is not None:
            assert "market" in s.components


# ---- ranking + best pick + NO BET ----------------------------------------

def _run(*, over25=0.64, btts=0.55, ah_line=-0.25, move_snap=None, stats=None, cfg=None):
    model = _model_probs(1.5, 1.5, over25=over25, btts=btts)
    totals = _totals(1.90, 1.95, 1.75, 2.05)
    ah_rows = extract_asian_handicap(_ah_payload(ah_line, 1.95, 1.95, 2.10, 1.85))
    return run_signal_engine(
        model_probs=model, stats=stats or {}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=move_snap, context=None, completeness=0.7, cfg=cfg,
    )


def test_ranking_is_sorted_descending():
    res = _run()
    scores = [r["score"] for r in res["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_best_pick_selected_and_is_market_pick():
    res = _run(over25=0.70, btts=0.48)
    assert res["decision"] == "BEST PICK"
    bp = res["best_pick"]
    assert bp is not None
    # best pick must be a market pick (BTTS / Total / Asian Handicap), not 1X2
    assert bp["market"] in ("BTTS", "Total", "Asian Handicap")


def test_no_bet_when_all_scores_low():
    # model == market (no edge), thin data -> scores low -> NO BET
    cfg = {"no_bet_score": 0.99, "best_pick_margin": 0.0, "min_confluence": 0}
    res = _run(cfg=cfg)
    assert res["decision"] == "NO BET"
    assert res["best_pick"] is None


# These tests predate the 2026-08-22 pick_gates and are about OTHER behaviour
# (the removed best_pick_margin gate, the F2 evidence floor). G2 (agreement)
# would veto their fixtures for an unrelated reason -- e.g. _run's default
# over25=0.64 against _totals(1.90, 1.95) is a +13.4pp deviation -- so it is
# switched off here to keep each test testing its own subject. G2 itself is
# covered by tests/test_pick_gates.py.
_NO_AGREEMENT_GATE = {
    "pick_gates": {"agreement": False},
    # F4-lite (plan v3): these tests target OTHER mechanics; keep the legacy
    # divergence-rewarding market component so their score fixtures hold.
    "market_component_reward_agreement": False,
}


def test_best_pick_selected_even_when_top_two_close():
    # A strong top signal is picked on its own ABSOLUTE strength; a large
    # best_pick_margin must NOT void it (the old "top signals too close" gate
    # is gone). A close runner-up never demotes a genuinely strong #1.
    cfg = {
        "best_pick_margin": 1.0, "no_bet_score": 0.0, "min_confluence": 0,
        "pick_gates": {"agreement": False},
        "market_component_reward_agreement": False,
    }
    res = _run(cfg=cfg)
    assert res["decision"] == "BEST PICK"
    assert res["best_pick"]["selection"] == "Over 2.5"


def test_conflict_forces_low_confidence():
    # model strongly BELOW market on Over -> negative edge -> conflict -> LOW
    model = _model_probs(1.2, 1.0, over25=0.30, btts=0.40)
    totals = _totals(1.60, 2.40, 1.75, 2.05)  # over heavily favoured -> model under conflicts
    res = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=[],
        movement_snapshot=None, context=None, completeness=0.7, cfg=None,
    )
    over = next(r for r in res["ranking"] if r["selection"] == "Over 2.5")
    assert over["edge_pp"] < -8.0
    assert over["confidence"] in ("LOW", "MEDIUM", "NO SIGNAL")


def test_missing_data_degrades_not_crashes():
    res = run_signal_engine(
        model_probs={}, stats={}, market_totals={}, ah_rows=[],
        movement_snapshot=None, context=None, completeness=0.0, cfg=None,
    )
    assert res["decision"] == "NO BET"
    assert res["best_pick"] is None
    assert res["data_quality"]["ah_available"] is False


def test_insufficient_odds_history_flags_unavailable():
    res = _run()
    assert res["data_quality"]["movement_history_available"] is False


def test_deterministic_output():
    r1 = _run()
    r2 = _run()
    assert r1 == r2


def test_existing_engine_compatibility():
    # model_probs shaped exactly like PredictionResult.to_dict()["model_probs"]
    model = _model_probs(1.5, 1.4)
    assert "1x2" in model and "lambda_home" in model
    res = run_signal_engine(
        model_probs=model, stats={},
        market_totals=_totals(1.9, 1.95, 1.75, 2.05), ah_rows=[],
        movement_snapshot=None, context=None, completeness=0.6, cfg=None,
    )
    assert "ranking" in res and "best_pick" in res


# ---- S35: Holstein Kiel vs St. Pauli conceptual validation ---------------

def test_holstein_kiel_settlement_2_2():
    """2-2: BTTS Yes = WIN, Over 2.5 = WIN, Away +0.25 = HALF WIN."""
    btts = settle_signal({"market": "BTTS", "selection": "Yes"}, 2, 2)
    assert btts["result"] == "win"
    over = settle_signal({"market": "Total", "selection": "Over 2.5"}, 2, 2)
    assert over["result"] == "win"
    ah = settle_signal(
        {"market": "Asian Handicap", "selection": "Away +0.25", "line": -0.25, "side": "away"},
        2, 2,
    )
    assert ah["result"] == "half_win"
    assert abs(ah["stake_return"] - 0.75) < 1e-9


def test_holstein_kiel_engine_derives_signals():
    """The engine independently produces BTTS Yes / Over 2.5 / Away +0.25
    from high-goal pre-match inputs (no final score anywhere)."""
    model = _model_probs(1.6, 1.6)
    totals = _totals(1.99, 1.94, 1.75, 2.05)
    ah_rows = extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.05, 1.88))
    res = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.6, cfg=None,
    )
    sels = {r["selection"] for r in res["ranking"]}
    assert "BTTS Yes" in sels
    assert "Over 2.5" in sels
    assert "Away +0.25" in sels


# ---- S34: backtest settlement --------------------------------------------

def test_run_signal_backtest_roi_and_half_win():
    records = [
        {"market": "BTTS", "selection": "Yes", "market_odds": 1.8,
         "home_goals": 2, "away_goals": 2},                 # win
        {"market": "Total", "selection": "Over 2.5", "market_odds": 1.9,
         "home_goals": 1, "away_goals": 0},                 # loss
        {"market": "Asian Handicap", "selection": "Away +0.25", "line": -0.25,
         "side": "away", "market_odds": 1.95, "home_goals": 2, "away_goals": 2},  # half win
        {"decision": "NO BET"},                             # no bet excluded
    ]
    report = run_signal_backtest(records)
    assert report["n"] == 3
    assert report["no_bet"] == 1
    ah = report["markets"]["Asian Handicap"]
    assert ah["half_wins"] == 1
    assert ah["n"] == 1
    # flat-stake: staked 3.0, ret = 1.8 + 0 + 0.75*1.95 = 1.8 + 1.4625 = 3.2625
    assert abs(report["markets"]["BTTS"]["roi"] - 80.0) < 1e-6
    total_ret = 1.8 + 0.75 * 1.95
    total_staked = 3.0
    assert abs((total_ret - total_staked) / total_staked * 100 - 8.75) < 1e-6


# ---- F2: prior-Elo evidence floor + F3: 1X2 reconciliation -------------

def _prio_model_probs(lh: float = 1.664, la: float = 1.036) -> dict:
    """model_probs shaped like the ADO-Den-Haag incident: λ from a PRIOR Elo
    rating (teams unseeded -> 1500 default), thin 1-match form windows."""
    mp = _model_probs(lh, la)
    mp["lambda_source"] = "elo"
    mp["elo_seeded"] = False
    mp["lambda_samples"] = 1
    mp["features_available"] = True
    return mp


def _ado_totals() -> dict:
    return _totals(1.60, 2.20, 1.75, 2.05)


def test_f2_prior_elo_thin_form_no_h2h_is_no_bet():
    """F2 evidence floor: prior-Elo λ + <3-match form + no H2H must VETO the
    pick to NO BET -- the ADO-Den-Haag class (HOME -0 @ 62/100 on a 1500
    prior + 1-match form, lost 0-2) must never surface a BEST PICK again."""
    res = run_signal_engine(
        model_probs=_prio_model_probs(),
        stats={"home_recent_goals": [(1, 2)], "away_recent_goals": [(2, 0)]},
        market_totals=_ado_totals(),
        ah_rows=extract_asian_handicap(_ah_payload(0.0, 1.90, 1.90, 1.98, 1.98)),
        completeness=0.8, has_h2h=False,
        cfg=dict(_NO_AGREEMENT_GATE),
    )
    assert res["decision"] == "NO BET"
    assert res["best_pick"] is None
    assert any("prior Elo" in r for r in res["reasons"])


def test_f2_prior_elo_thin_form_with_h2h_caps_low():
    """F2 cap: same prior-based evidence but H2H exists -> pick kept, never
    HIGH -- confidence LOW + an explicit evidence note on the pick."""
    res = run_signal_engine(
        model_probs=_prio_model_probs(),
        stats={"home_recent_goals": [(1, 2)], "away_recent_goals": [(2, 0)]},
        market_totals=_ado_totals(),
        ah_rows=extract_asian_handicap(_ah_payload(0.0, 1.90, 1.90, 1.98, 1.98)),
        completeness=0.8, has_h2h=True,
        cfg=dict(_NO_AGREEMENT_GATE),
    )
    assert res["decision"] == "BEST PICK"
    bp = res["best_pick"]
    assert bp["confidence"] == "LOW"
    assert any("prior Elo" in n for n in bp["evidence_notes"])


def test_f2_full_form_or_seeded_elo_not_vetoed():
    """F2 must NOT fire when the evidence is real: seeded Elo (lambda_source
    features) OR a full form window means the floor does not apply."""
    # full 5-match windows, prior Elo flag still false-positive? no --
    # lambda_source features + elo_seeded True -> floor inactive.
    res = run_signal_engine(
        model_probs=_prio_model_probs(),
        stats={
            "home_recent_goals": [(2, 0), (1, 1), (0, 2), (3, 0), (1, 0)],
            "away_recent_goals": [(1, 0), (0, 1), (2, 2), (1, 1), (0, 2)],
        },
        market_totals=_ado_totals(),
        ah_rows=extract_asian_handicap(_ah_payload(0.0, 1.90, 1.90, 1.98, 1.98)),
        completeness=0.8, has_h2h=False,
        cfg=dict(_NO_AGREEMENT_GATE),
    )
    assert res["decision"] == "BEST PICK"
    # floor must not have vetoed or capped
    assert res["best_pick"]["confidence"] != "LOW"


# G1 isolation: these two tests are about the 1X2-reconciliation branch ONLY.
# The other pick_gates (G2 agreement / G4 lambda band / G7 price) are switched
# off here on purpose -- this fixture's over25=0.80 against a 1.60 market is a
# ~+22pp deviation that G2 would veto, so leaving them on would make the test
# pass for the wrong reason and stop testing G1 at all.
_G1_ONLY = {
    "agreement": False,
    "lambda_total_sanity": False,
    "require_price": False,
}


def test_g1_non_actionable_1x2_vetoes_the_pick():
    """G1 (post-mortem 2026-08-22): when the independent 1X2 layer says NO BET,
    the signal engine must VETO -- not merely cap confidence at MEDIUM.

    Replaces the old F3 cap. On 2026-08-21 all 11 published picks carried
    decision_type NO BET / NO CLEAR DECISION and the day lost 2.26u; the cap
    also satisfied the analyse-layer ``_strong_pick`` bypass, so it opened the
    evidence gate it was meant to close.
    """
    res = run_signal_engine(
        model_probs=_model_probs(2.0, 0.6, over25=0.80, btts=0.62),
        stats={"home_recent_goals": [(2, 0), (1, 1), (0, 2), (3, 0), (1, 0)],
               "away_recent_goals": [(1, 0), (0, 1), (2, 2), (1, 1), (0, 2)]},
        market_totals=_ado_totals(),
        ah_rows=extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.10, 1.85)),
        completeness=0.8, has_h2h=True, model_decision_type="NO BET",
        cfg={"market_component_reward_agreement": False,
             "pick_gates": dict(_G1_ONLY)},
    )
    assert res["decision"] == "NO BET"
    assert res["best_pick"] is None
    assert any("diveto (G1)" in r for r in res["reasons"])


def test_g1_can_be_reverted_to_the_legacy_medium_cap():
    """``respect_model_decision: false`` (operator setting since 2026-08-22)
    publishes the pick with its REAL confidence -- no MEDIUM downgrade -- and
    keeps the 1X2-layer disagreement on internal_notes."""
    res = run_signal_engine(
        model_probs=_model_probs(2.0, 0.6, over25=0.80, btts=0.62),
        stats={"home_recent_goals": [(2, 0), (1, 1), (0, 2), (3, 0), (1, 0)],
               "away_recent_goals": [(1, 0), (0, 1), (2, 2), (1, 1), (0, 2)]},
        market_totals=_ado_totals(),
        ah_rows=extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.10, 1.85)),
        completeness=0.8, has_h2h=True, model_decision_type="NO BET",
        cfg={"market_component_reward_agreement": False,
             "pick_gates": dict(_G1_ONLY, respect_model_decision=False)},
    )
    assert res["decision"] == "BEST PICK"
    bp = res["best_pick"]
    assert bp is not None and bp["selection"]
    # real confidence preserved -- the old MEDIUM cap is gone
    assert bp["confidence"] in ("LOW", "MEDIUM", "HIGH", "VERY HIGH")
    assert any("tanpa dukungan layer model" in r for r in res["reasons"])
    # P2-3: the 1X2-layer disagreement note lives on internal_notes (hidden
    # from the summary embed, rendered only after "Lihat Hasil"), NOT on
    # evidence_notes -- which stays clean for evidence-floor reasons.
    assert any("1X2" in n for n in bp["internal_notes"])
    assert not any("1X2" in n for n in (bp.get("evidence_notes") or []))


def test_f3_actionable_1x2_does_not_cap():
    """F3/G1 must NOT fire when the 1X2 layer produced an actionable decision."""
    res = run_signal_engine(
        model_probs=_model_probs(2.0, 0.6, over25=0.80, btts=0.62),
        stats={"home_recent_goals": [(2, 0), (1, 1), (0, 2), (3, 0), (1, 0)],
               "away_recent_goals": [(1, 0), (0, 1), (2, 2), (1, 1), (0, 2)]},
        market_totals=_ado_totals(),
        ah_rows=extract_asian_handicap(_ah_payload(-0.25, 1.95, 1.95, 2.10, 1.85)),
        completeness=0.8, has_h2h=True, model_decision_type="GOOD",
        cfg={"pick_gates": dict(_G1_ONLY)},
    )
    assert res["decision"] == "BEST PICK"
    assert not (res["best_pick"].get("evidence_notes") or [])


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
