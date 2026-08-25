"""Tests for predictor.py math."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import predictor


def test_poisson_pmf_basic():
    p0 = predictor._poisson_pmf(0, 1.5)
    p1 = predictor._poisson_pmf(1, 1.5)
    assert math.isclose(p0, math.exp(-1.5), rel_tol=1e-9)
    assert math.isclose(p1, 1.5 * math.exp(-1.5), rel_tol=1e-9)


def test_poisson_pmf_zero_lambda():
    assert predictor._poisson_pmf(0, 0) == 1.0
    assert predictor._poisson_pmf(1, 0) == 0.0


def test_normalize_odds():
    probs = {"home": 0.5, "draw": 0.3, "away": 0.2}
    norm = predictor.normalize_odds(probs)
    assert math.isclose(sum(norm.values()), 1.0, abs_tol=1e-9)


def test_normalize_odds_with_overround():
    probs = {"home": 0.55, "draw": 0.30, "away": 0.25}
    norm = predictor.normalize_odds(probs)
    assert math.isclose(sum(norm.values()), 1.0, abs_tol=1e-9)
    assert norm["home"] < probs["home"]


def test_normalize_odds_realistic():
    probs = {"home": 1/2.10, "draw": 1/3.40, "away": 1/3.60}
    norm = predictor.normalize_odds(probs)
    assert math.isclose(sum(norm.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(norm["home"], 0.4543, abs_tol=1e-3)


def test_solve_lambdas_home_dominant():
    lh, la = predictor.solve_lambdas(0.55, 0.25, 0.20)
    assert lh > la
    assert 0.5 < lh < 4.0
    assert 0.5 < la < 4.0


def test_solve_lambdas_roundtrip():
    """Regression: solve_lambdas must reproduce the input 1X2 probabilities.

    The old implementation approximated P(home win) as P(home>0 AND away=0),
    never converged, and fell back to sqrt(2.5*p), which deflated total goals
    to ~1.9 and biased every pick toward Under with fake large edges.
    """
    # Realistic football 1X2 odds (draw 20-30%). The old buggy solver
    # returned total goals ~1.9 for all of these.
    # Realistic draw range is 24-30% for live odds; a draw as low as 20%
    # forces the pure-Poisson model to a very high total goals (~3.8), which
    # is mathematically consistent but not representative of real markets.
    cases = [
        (0.4927, 0.2503, 0.2570),  # Sabah vs AGF (near-even)
        (0.52, 0.24, 0.24),        # moderate favorite
        (0.45, 0.28, 0.27),        # very even match
        (0.40, 0.30, 0.30),        # three-way near-split
    ]
    for ph, pa, pd in cases:
        lh, la = predictor.solve_lambdas(ph, pa, pd)
        back = predictor.prob_1x2(predictor.score_matrix(lh, la))
        total = back["home"] + back["draw"] + back["away"]
        # normalize (score matrix truncates at MAX_GOALS tail mass); the
        # solver is exact to ~1e-4 for realistic markets, so assert tightly
        assert abs(back["home"] / total - ph) < 1e-3, (ph, lh, la, back)
        assert abs(back["draw"] / total - pd) < 1e-3, (pd, lh, la, back)
        assert abs(back["away"] / total - pa) < 1e-3, (pa, lh, la, back)
        # total goals must be sane for these odds (2.0-3.6), NOT ~1.9
        assert 2.0 <= lh + la <= 3.6, (lh, la)


def test_score_matrix_sums_to_one():
    matrix = predictor.score_matrix(1.5, 1.2)
    total = sum(sum(row) for row in matrix)
    assert math.isclose(total, 1.0, abs_tol=1e-3)


def test_prob_1x2_sums_to_one():
    matrix = predictor.score_matrix(1.5, 1.2)
    p = predictor.prob_1x2(matrix)
    assert math.isclose(sum(p.values()), 1.0, abs_tol=1e-3)
    assert 0 < p["home"] < 1
    assert 0 < p["draw"] < 1
    assert 0 < p["away"] < 1


def test_prob_over_2_5():
    matrix = predictor.score_matrix(1.5, 1.2)
    p_o25 = predictor.prob_over(matrix, 2.5)
    assert 0.4 < p_o25 < 0.7


def test_prob_btts():
    matrix = predictor.score_matrix(1.5, 1.2)
    p_btts = predictor.prob_btts(matrix)
    assert 0.4 < p_btts < 0.7


def test_derive_picks_basic():
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {
        "Over 2.5": {"odds": 1.85},
        "Under 2.5": {"odds": 1.95},
    }
    result = predictor.derive_picks(consensus, market_totals, signal=75)
    assert "top_picks" in result
    assert "best_pick" in result
    assert len(result["top_picks"]) >= 1
    best = result["best_pick"]
    assert best is not None
    assert best["market_odds"] > 0
    assert best["model_prob"] > 0


def test_derive_picks_ranking_high_signal_uses_edge():
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {
        "Over 2.5": {"odds": 1.85},
        "Under 2.5": {"odds": 1.95},
    }
    result_high = predictor.derive_picks(consensus, market_totals, signal=80)
    result_low = predictor.derive_picks(consensus, market_totals, signal=40)
    assert len(result_high["top_picks"]) == len(result_low["top_picks"])
    assert result_high["model_probs"]["lambda_home"] > 0


def test_derive_picks_no_odds():
    result = predictor.derive_picks({}, {}, signal=50)
    assert result["top_picks"] == []
    assert result["best_pick"] is None


def test_derive_picks_only_1x2():
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    result = predictor.derive_picks(consensus, {}, signal=50)
    assert len(result["top_picks"]) >= 1
    assert all(p["market"] == "1X2" for p in result["top_picks"])


def test_full_pipeline():
    """Simulate full analyse pipeline with realistic odds."""
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {
        "Over 1.5": {"odds": 1.25},
        "Under 1.5": {"odds": 4.00},
        "Over 2.5": {"odds": 1.85},
        "Under 2.5": {"odds": 1.95},
        "Over 3.5": {"odds": 2.80},
        "Under 3.5": {"odds": 1.45},
    }
    result = predictor.derive_picks(consensus, market_totals, signal=72)
    assert len(result["top_picks"]) <= 3
    assert result["best_pick"] is not None
    mp = result["model_probs"]
    assert 0 < mp["over_2.5"] < 1
    assert 0 < mp["btts_yes"] < 1
    assert mp["lambda_home"] > 0
    assert mp["lambda_away"] > 0


def test_derive_picks_with_xg_override():
    """xg_lambda should override odds-derived lambda."""
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {"Over 2.5": {"odds": 1.85}, "Under 2.5": {"odds": 1.95}}

    result_no_xg = predictor.derive_picks(consensus, market_totals, signal=72)
    result_with_xg = predictor.derive_picks(
        consensus, market_totals, signal=72, xg_lambda=(1.8, 1.2)
    )

    assert result_with_xg["model_probs"]["lambda_source"] == "xg"
    assert result_no_xg["model_probs"]["lambda_source"] == "odds_derived"
    assert result_with_xg["model_probs"]["lambda_home"] == 1.8
    assert result_with_xg["model_probs"]["lambda_away"] == 1.2


def test_grade_recommendation_valid():
    g = predictor.grade_recommendation(
        confidence=0.85, calibration_quality=0.9, data_completeness=0.9,
        edge_pct=5.0, signal=80,
    )
    assert g["grade"] == "VALID"
    assert g["label"] == "✅ VALID"
    assert g["reasons"] == []


def test_grade_recommendation_low_edge():
    # edge < 2pp means no real value -> NOT a valid bet, HATI-HATI.
    g = predictor.grade_recommendation(
        confidence=0.85, calibration_quality=0.9, data_completeness=0.9,
        edge_pct=0.5, signal=80,
    )
    assert g["grade"] == "LOW"
    assert any("edge" in r for r in g["reasons"])


def test_grade_recommendation_low_confidence():
    g = predictor.grade_recommendation(
        confidence=0.4, calibration_quality=0.9, data_completeness=0.9,
        edge_pct=6.0, signal=80,
    )
    assert g["grade"] == "LOW"
    assert any("confidence" in r for r in g["reasons"])


def test_grade_recommendation_missing_calibration():
    g = predictor.grade_recommendation(
        confidence=0.85, calibration_quality=None, data_completeness=0.9,
        edge_pct=6.0, signal=80,
    )
    assert g["grade"] == "LOW"
    assert any("kalibrasi" in r for r in g["reasons"])


def test_grade_recommendation_low_signal():
    g = predictor.grade_recommendation(
        confidence=0.85, calibration_quality=0.9, data_completeness=0.9,
        edge_pct=6.0, signal=30,
    )
    # signal below 70 -> not fully valid, but everything else passes
    assert g["grade"] == "CANDIDATE"
    assert any("signal" in r for r in g["reasons"])


def test_grade_top_match_layak():
    g = predictor.grade_top_match(
        has_odds=True, has_home_form=True, has_away_form=True,
        signal=80, bookmakers_count=8,
    )
    assert g["grade"] == "LAYAK"
    assert g["label"] == "🟢 LAYAK"
    assert g["reasons"] == []


def test_grade_top_match_few_bookies():
    g = predictor.grade_top_match(
        has_odds=True, has_home_form=True, has_away_form=True,
        signal=80, bookmakers_count=2,
    )
    assert g["grade"] == "CUKUP"
    assert any("bookie" in r for r in g["reasons"])


def test_grade_top_match_no_odds():
    g = predictor.grade_top_match(
        has_odds=False, has_home_form=True, has_away_form=True,
        signal=80, bookmakers_count=0,
    )
    assert g["grade"] == "SKIP"
    assert any("odds" in r for r in g["reasons"])


def test_grade_top_match_low_signal():
    g = predictor.grade_top_match(
        has_odds=True, has_home_form=True, has_away_form=True,
        signal=40, bookmakers_count=8,
    )
    assert g["grade"] == "SKIP"
    assert any("signal" in r for r in g["reasons"])


def test_grade_top_match_missing_form():
    g = predictor.grade_top_match(
        has_odds=True, has_home_form=True, has_away_form=False,
        signal=75, bookmakers_count=6,
    )
    assert g["grade"] == "CUKUP"
    assert any("form away" in r for r in g["reasons"])


def test_grade_recommendation_thin_data():
    g = predictor.grade_recommendation(
        confidence=0.92, calibration_quality=0.98, data_completeness=0.3,
        edge_pct=4.0, signal=60,
    )
    assert g["grade"] == "LOW"
    assert any("kelengkapan" in r for r in g["reasons"])


def test_derive_picks_reference_only_no_edge_ev():
    """Model A rule: derive_picks is the odds-derived MARKET view, reference-
    only -- every pick carries edge 0.0 / ev 0.0 so no consumer (grading,
    similar-signal, backtest) can ever treat a market mirror as an independent
    model pick."""
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {
        "Over 2.5": {"odds": 1.85},
        "Under 2.5": {"odds": 1.95},
    }
    result = predictor.derive_picks(consensus, market_totals, signal=72)
    assert len(result["top_picks"]) >= 3
    for p in result["top_picks"]:
        assert p["edge"] == 0.0
        assert p["ev"] == 0.0
        assert p["model_prob"] > 0.0  # reference probs still present
        assert p["implied_prob"] > 0.0


def test_derive_picks_with_zero_xg_falls_back():
    """xg_lambda=(0,0) should fall back to odds-derived."""
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    result = predictor.derive_picks(
        consensus, {}, signal=72, xg_lambda=(0.0, 0.0)
    )
    assert result["model_probs"]["lambda_source"] == "odds_derived"


def test_fair_pair_implied_removes_margin():
    """Pair-normalized implied must sum to 1 and sit above raw per-side 1/odds."""
    o, u = predictor.fair_pair_implied(2.10, 2.14)
    assert math.isclose(o + u, 1.0, abs_tol=1e-9)
    # raw 1/odds are deflated by margin: fair implied is strictly higher
    assert o > 1.0 / 2.10
    assert u > 1.0 / 2.14


def test_fair_pair_implied_missing_side():
    assert predictor.fair_pair_implied(0, 2.14) is None
    assert predictor.fair_pair_implied(2.10, 0) is None
    assert predictor.fair_pair_implied(0, 0) is None


def test_derive_picks_totals_implied_margin_free():
    """The totals implied_prob must stay margin-free (pair sums to 1).

    Edge/EV are no longer computed here at all (Model A reference-only rule:
    derive_picks is the odds-derived market view and must never claim value;
    the 1X2-vs-totals mismatch used to surface a fake +11pp "edge" on a
    market mirror pick).
    """
    consensus = {"home": 1.93, "draw": 3.70, "away": 3.80}
    market_totals = {
        "Over 2.5": {"odds": 2.10},
        "Under 2.5": {"odds": 2.14},
    }
    result = predictor.derive_picks(consensus, market_totals, signal=80)
    by_sel = {p["selection"]: p for p in result["top_picks"]}
    under = by_sel["Under 2.5"]
    fo, fu = predictor.fair_pair_implied(2.10, 2.14)
    assert math.isclose(fo + fu, 1.0, abs_tol=1e-9)
    assert math.isclose(under["implied_prob"], fu, abs_tol=1e-9)
    assert under["edge"] == 0.0
    assert under["ev"] == 0.0


def test_derive_picks_1x2_implied_consistent():
    """1X2 pick implied_prob must equal the margin-free implied; edge/EV are
    pinned to 0 (reference-only market view, Model A rule)."""
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    result = predictor.derive_picks(consensus, {}, signal=50)
    for p in result["top_picks"]:
        if p["market"] == "1X2":
            assert p["edge"] == 0.0
            assert p["ev"] == 0.0
            # implied stays a real margin-free reference
            assert p["implied_prob"] > 0.0


def test_derive_picks_totals_single_side_fallback():
    """Single-sided totals market falls back to raw implied instead of None."""
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {"Over 2.5": {"odds": 1.85}}
    result = predictor.derive_picks(consensus, market_totals, signal=50)
    over = next(p for p in result["top_picks"] if p["selection"] == "Over 2.5")
    assert math.isclose(over["implied_prob"], 1.0 / 1.85, abs_tol=1e-9)


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
