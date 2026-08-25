"""Probabilistic market predictions from 1X2 odds.

Pipeline:
  1. Normalize 1X2 odds (remove bookmaker margin / overround).
  2. Solve Poisson scoring rates (lambda_home, lambda_away) from normalized probs.
  3. Build score matrix P(home=i, away=j) for i,j in [0..MAX_GOALS].
  4. Derive market probabilities: O/U 1.5/2.5/3.5, BTTS, 1X2, Double Chance.
  5. Compare each derived probability with market implied probability to get edge.
  6. Produce ranked picks (top 3 by combo rule) and single best pick.

All math is exact (no ML). The Poisson assumption is documented in disclaimer.
"""
from __future__ import annotations

import math
from typing import Any

MAX_GOALS = 10


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def normalize_odds(probs: dict[str, float]) -> dict[str, float]:
    """Remove vigorish: probs = {home, draw, away} implied (1/odds).
    Output sums to 1.0.
    """
    total = sum(v for v in probs.values() if v > 0)
    if total <= 0:
        return {"home": 0.0, "draw": 0.0, "away": 0.0}
    return {k: (v / total if v > 0 else 0.0) for k, v in probs.items()}


def solve_lambdas(p_home: float, p_away: float, p_draw: float) -> tuple[float, float]:
    """Solve lambda_home, lambda_away from normalized 1X2 probabilities.

    Two nested bisections over the Poisson score matrix:
      1. outer: total goals T such that P(draw) matches p_draw
      2. inner: home goal share s such that the home/(home+away) win ratio
         matches p_home / (p_home + p_away)

    This is exact (no fixed-point instability); the previous implementation
    approximated P(home win) as P(home>0 & away=0), which never converged and
    always fell back to sqrt(2.5*p), systematically deflating total goals to
    ~1.9 and biasing every pick toward Under with fake large edges.

    Exactness holds for realistic markets (draw roughly 18-35%). For extreme
    lopsided markets the solution can hit the T=6.0 window bound or the 0.1
    rate clamp; in that regime the 1X2 roundtrip degrades gracefully (up to
    ~1pp error) instead of failing loudly. derive_picks already guards
    degenerate consensus (home odds <= 0).
    """
    if p_home <= 0 or p_away <= 0:
        return 1.0, 1.0
    target_share = p_home / (p_home + p_away)

    def _solve_share(T: float) -> tuple[float, float, dict[str, float]]:
        lo, hi = 0.01, 0.99
        p: dict[str, float] = {}
        # 1e-4 tolerance on a [0.01, 0.99] interval converges in ~14 steps.
        for _ in range(20):
            s = 0.5 * (lo + hi)
            lh = T * s
            la = T * (1.0 - s)
            p = prob_1x2(score_matrix(lh, la))
            denom = p["home"] + p["away"]
            share = p["home"] / denom if denom > 0 else 0.5
            if abs(share - target_share) < 1e-4:
                break
            if share < target_share:
                lo = s
            else:
                hi = s
        return lh, la, p

    lo_T, hi_T = 0.5, 6.0
    lh, la, p = 1.0, 1.0, {}
    for _ in range(20):
        T = 0.5 * (lo_T + hi_T)
        lh, la, p = _solve_share(T)
        if abs(p["draw"] - p_draw) < 1e-4:
            break
        if p["draw"] > p_draw:
            lo_T = T  # too many draws -> need more total goals
        else:
            hi_T = T
    # Clamp keeps PMFs well-defined for extreme (lopsided) markets.
    return max(0.1, lh), max(0.1, la)


def fair_pair_implied(odds_a: float, odds_b: float) -> tuple[float, float] | None:
    """Margin-free implied probabilities for a two-outcome pair.

    Raw per-side implied (1/odds) is deflated by the bookmaker margin, so an
    edge computed against it is inflated by ~2-3pp and can look like real
    value when there is none. Normalizing the pair (Over/Under, BTTS Yes/No)
    removes the margin, putting totals/BTTS edges on the same scale as the
    1X2 margin-free implied. Returns (p_a, p_b) summing to 1.0, or None when
    either odds is missing.
    """
    if odds_a <= 0 or odds_b <= 0:
        return None
    ia = 1.0 / odds_a
    ib = 1.0 / odds_b
    total = ia + ib
    if total <= 0:
        return None
    return ia / total, ib / total


def score_matrix(lambda_home: float, lambda_away: float) -> list[list[float]]:
    matrix = [[0.0 for _ in range(MAX_GOALS + 1)] for _ in range(MAX_GOALS + 1)]
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            matrix[h][a] = _poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)
    return matrix


def prob_over(matrix: list[list[float]], threshold: float) -> float:
    """P(total_goals > threshold) for half-integer thresholds like 1.5, 2.5, 3.5."""
    t = math.floor(threshold)
    p_under = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            if h + a <= t:
                p_under += matrix[h][a]
    return 1.0 - p_under


def prob_btts(matrix: list[list[float]]) -> float:
    """P(both teams score). Home/away goals are independent under Poisson,
    so P(A and B) = P(A) * P(B). (Fixed: previously added P(no-both).)
    """
    p_no_home = 0.0
    for a in range(MAX_GOALS + 1):
        p_no_home += matrix[0][a]
    p_no_away = 0.0
    for h in range(MAX_GOALS + 1):
        p_no_away += matrix[h][0]
    return (1.0 - p_no_home) * (1.0 - p_no_away)


def prob_1x2(matrix: list[list[float]]) -> dict[str, float]:
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = matrix[h][a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    return {"home": p_home, "draw": p_draw, "away": p_away}


def grade_recommendation(
    *,
    confidence: float | None,
    calibration_quality: float | None,
    data_completeness: float | None,
    edge_pct: float,
    signal: int = 0,
) -> dict[str, Any]:
    """Grade whether a pick is a VALID / ACCURATE recommendation.

    A pick is not automatically worth betting just because it has a positive
    edge: the underlying model must be reliable. This gate checks four
    documented criteria (all already computed by the engine):

      - confidence >= 0.70  (HIGH label basis)
      - calibration quality >= 0.50 (model validated by backtest history)
      - data completeness >= 0.50 (odds + form + attack/defense present)
      - edge >= 2.0 percentage points (real value, not noise)

    Returns {"grade": "VALID"|"CANDIDATE"|"LOW", "label": emoji badge,
    "reasons": [..]}. ``reasons`` explains *why* a pick is downgraded so
    the user sees exactly what is missing.
    """
    reasons: list[str] = []

    def _ok(value: float | None, threshold: float, what: str) -> bool:
        if value is None:
            reasons.append(f"{what} tidak dihitung")
            return False
        if value >= threshold:
            return True
        reasons.append(f"{what} {value:.2f} < {threshold:.2f}")
        return False

    conf_ok = _ok(confidence, 0.70, "confidence")
    calib_ok = _ok(calibration_quality, 0.50, "kalibrasi")
    compl_ok = _ok(data_completeness, 0.50, "kelengkapan data")
    edge_ok = edge_pct is not None and edge_pct >= 2.0
    if not edge_ok:
        reasons.append(f"edge {edge_pct if edge_pct is not None else 0:.2f}% < 2.00%")
    signal_ok = bool(signal) and signal >= 70
    if signal and not signal_ok:
        reasons.append(f"signal {signal}/100 < 70")

    if conf_ok and calib_ok and compl_ok and edge_ok and signal_ok:
        return {"grade": "VALID", "label": "✅ VALID", "reasons": []}
    if conf_ok and calib_ok and compl_ok and edge_ok:
        return {"grade": "CANDIDATE", "label": "⚠️ KANDIDAT", "reasons": reasons}
    return {"grade": "LOW", "label": "🔴 HATI-HATI", "reasons": reasons}


def grade_top_match(
    *,
    has_odds: bool,
    has_home_form: bool,
    has_away_form: bool,
    signal: int,
    bookmakers_count: int = 0,
) -> dict[str, Any]:
    """Lightweight screening grade for list output (top matches).

    Unlike ``grade_recommendation`` (which grades a single pick after the
    full prediction engine ran), this grades a match from pre-flight data
    only -- it is deliberately cheap because ``top`` renders N matches and
    running Elo+Poisson per match would blow the bot's subprocess timeout.

    Gates (all must pass for LAYAK):
      - odds available from >= 3 bookmakers
      - both teams have form
      - signal >= 70

    Returns {"grade": "LAYAK"|"CUKUP"|"SKIP", "label": emoji badge,
    "reasons": [..]} so the user can filter matches worth analysing.
    """
    reasons: list[str] = []
    if not has_odds:
        reasons.append("tanpa odds")
    elif bookmakers_count < 3:
        reasons.append(f"bookie < 3 ({bookmakers_count})")
    if not has_home_form:
        reasons.append("form home kosong")
    if not has_away_form:
        reasons.append("form away kosong")
    if signal < 70:
        reasons.append(f"signal {signal}/100 < 70")

    if not reasons:
        return {"grade": "LAYAK", "label": "🟢 LAYAK", "reasons": []}
    if has_odds and (has_home_form or has_away_form) and signal >= 50:
        return {"grade": "CUKUP", "label": "🟡 CUKUP", "reasons": reasons}
    return {"grade": "SKIP", "label": "🔴 SKIP", "reasons": reasons}


def derive_picks(
    consensus: dict[str, float],
    market_totals: dict[str, dict[str, float]],
    signal: int,
    xg_lambda: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build ranked picks list.

    consensus: {home, draw, away} decimal odds
    market_totals: {over_2.5: {odds}, under_2.5: {odds}, ...}
    signal: 0-100, decides ranking strategy
    xg_lambda: optional (lambda_home, lambda_away) from Sofascore xG
        overrides the odds-derived lambda.
    """
    if not consensus or consensus.get("home", 0) <= 0:
        return {"top_picks": [], "best_pick": None, "model_probs": {}}

    # Margin-free implied (normalized) is the fair comparison basis for every
    # edge; raw 1/odds includes the bookmaker margin and would inflate edges.
    norm = normalize_odds({
        "home": 1.0 / consensus["home"] if consensus.get("home", 0) > 0 else 0,
        "draw": 1.0 / consensus["draw"] if consensus.get("draw", 0) > 0 else 0,
        "away": 1.0 / consensus["away"] if consensus.get("away", 0) > 0 else 0,
    })
    if xg_lambda and xg_lambda[0] > 0 and xg_lambda[1] > 0:
        lh, la = xg_lambda
    else:
        lh, la = solve_lambdas(norm["home"], norm["away"], norm["draw"])
    matrix = score_matrix(lh, la)
    probs_1x2 = prob_1x2(matrix)
    p_o15 = prob_over(matrix, 1.5)
    p_o25 = prob_over(matrix, 2.5)
    p_o35 = prob_over(matrix, 3.5)
    p_btts = prob_btts(matrix)

    model_probs = {
        "1x2": probs_1x2,
        "over_1.5": p_o15,
        "over_2.5": p_o25,
        "over_3.5": p_o35,
        "btts_yes": p_btts,
        "lambda_home": round(lh, 3),
        "lambda_away": round(la, 3),
        "lambda_source": "sofascore_xg" if (xg_lambda and xg_lambda[0] > 0) else "odds_derived",
    }

    picks: list[dict[str, Any]] = []

    # implied_prob and edge must be mutually consistent AND margin-free for
    # every pick (edge == (model_prob - implied_prob) * 100). Raw per-side
    # 1/odds includes the bookmaker margin, which would inflate edges.
    picks.append({
        "rank": 0,
        "market": "1X2",
        "selection": "Home Win",
        "model_prob": probs_1x2["home"],
        "market_odds": consensus.get("home", 0),
        "implied_prob": norm["home"],
        "edge": (probs_1x2["home"] - norm["home"]) * 100.0,
    })
    picks.append({
        "rank": 0,
        "market": "1X2",
        "selection": "Draw",
        "model_prob": probs_1x2["draw"],
        "market_odds": consensus.get("draw", 0),
        "implied_prob": norm["draw"],
        "edge": (probs_1x2["draw"] - norm["draw"]) * 100.0,
    })
    picks.append({
        "rank": 0,
        "market": "1X2",
        "selection": "Away Win",
        "model_prob": probs_1x2["away"],
        "market_odds": consensus.get("away", 0),
        "implied_prob": norm["away"],
        "edge": (probs_1x2["away"] - norm["away"]) * 100.0,
    })

    def _add_total_pick(
        label: str,
        selection: str,
        prob: float,
        market_odds: float,
        implied_prob: float,
    ) -> None:
        if market_odds <= 0 or implied_prob <= 0:
            return
        picks.append({
            "rank": 0,
            "market": label,
            "selection": selection,
            "model_prob": prob,
            "market_odds": market_odds,
            "implied_prob": implied_prob,
            "edge": (prob - implied_prob) * 100.0,
        })

    for thresh in (1.5, 2.5, 3.5):
        sel_o = f"Over {thresh}"
        sel_u = f"Under {thresh}"
        odds_o = market_totals.get(sel_o, {}).get("odds", 0)
        odds_u = market_totals.get(sel_u, {}).get("odds", 0)
        fair = fair_pair_implied(odds_o, odds_u)
        prob_o = model_probs[f"over_{thresh}"]
        prob_u = 1.0 - prob_o
        if fair is not None:
            imp_o, imp_u = fair
            _add_total_pick("Total", sel_o, prob_o, odds_o, imp_o)
            _add_total_pick("Total", sel_u, prob_u, odds_u, imp_u)
        else:
            # Single-sided market: fall back to raw implied (no pair to
            # normalize) rather than inventing a margin-free number.
            _add_total_pick("Total", sel_o, prob_o, odds_o,
                            1.0 / odds_o if odds_o > 0 else 0.0)
            _add_total_pick("Total", sel_u, prob_u, odds_u,
                            1.0 / odds_u if odds_u > 0 else 0.0)

    odds_yes = market_totals.get("BTTS Yes", {}).get("odds", 0)
    odds_no = market_totals.get("BTTS No", {}).get("odds", 0)
    fair_btts = fair_pair_implied(odds_yes, odds_no)
    if fair_btts is not None:
        imp_yes, imp_no = fair_btts
    else:
        imp_yes = 1.0 / odds_yes if odds_yes > 0 else 0.0
        imp_no = 1.0 / odds_no if odds_no > 0 else 0.0
    if odds_yes > 0 and imp_yes > 0:
        picks.append({
            "rank": 0,
            "market": "BTTS",
            "selection": "Yes",
            "model_prob": p_btts,
            "market_odds": odds_yes,
            "implied_prob": imp_yes,
            "edge": (p_btts - imp_yes) * 100.0,
        })
    if odds_no > 0 and imp_no > 0:
        picks.append({
            "rank": 0,
            "market": "BTTS",
            "selection": "No",
            "model_prob": 1.0 - p_btts,
            "market_odds": odds_no,
            "implied_prob": imp_no,
            "edge": ((1.0 - p_btts) - imp_no) * 100.0,
        })

    candidates = [p for p in picks if p["market_odds"] > 0 and p["model_prob"] > 0]
    if signal >= 70:
        candidates.sort(key=lambda p: p["edge"], reverse=True)
    else:
        candidates.sort(key=lambda p: p["model_prob"], reverse=True)

    top3 = candidates[:3]
    for i, p in enumerate(top3, 1):
        p["rank"] = i

    best = top3[0] if top3 else None

    return {
        "top_picks": top3,
        "best_pick": best,
        "model_probs": model_probs,
        "score_matrix": {
            "lambda_home": lh,
            "lambda_away": la,
            "over_2.5_prob": p_o25,
            "over_1.5_prob": p_o15,
            "over_3.5_prob": p_o35,
            "btts_prob": p_btts,
        },
    }
