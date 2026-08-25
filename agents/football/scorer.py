"""Deterministic scoring. No ML. Pure math over odds and stats."""
from __future__ import annotations

import statistics
from typing import Any


def implied_prob(odds_decimal: float) -> float:
    if odds_decimal <= 1.0:
        return 0.0
    return 1.0 / odds_decimal


def consensus_odds(
    bookmaker_odds: list[dict[str, float]],
    primary_bookmaker: str | None = None,
) -> dict[str, float]:
    """each item: {"bookmaker": name, "home": x, "draw": y, "away": z}.

    Default: median across bookmakers (robust to outliers). With
    ``primary_bookmaker`` set (sharp-book convention, 2026-08-23), that
    bookmaker's COMPLETE 1X2 quote is used verbatim when present -- a single
    sharp price is a cleaner benchmark than a median diluted by soft books.
    Falls back to the median whenever the primary is absent or its quote is
    incomplete (any side <= 1.0), so the shape of the result never changes.
    """
    if not bookmaker_odds:
        return {"home": 0.0, "draw": 0.0, "away": 0.0}
    if primary_bookmaker:
        pl = str(primary_bookmaker).strip().lower()
        for b in bookmaker_odds:
            if str(b.get("bookmaker") or "").strip().lower() != pl:
                continue
            h = b.get("home") or 0
            d = b.get("draw") or 0
            a = b.get("away") or 0
            if h > 1.0 and d > 1.0 and a > 1.0:
                return {"home": float(h), "draw": float(d), "away": float(a)}
    homes = [b["home"] for b in bookmaker_odds if b.get("home", 0) > 0]
    draws = [b["draw"] for b in bookmaker_odds if b.get("draw", 0) > 0]
    aways = [b["away"] for b in bookmaker_odds if b.get("away", 0) > 0]
    return {
        "home": statistics.median(homes) if homes else 0.0,
        "draw": statistics.median(draws) if draws else 0.0,
        "away": statistics.median(aways) if aways else 0.0,
    }


def best_odds(bookmaker_odds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return best (highest) odds per side with bookmaker name."""
    best: dict[str, dict[str, Any]] = {"home": {}, "draw": {}, "away": {}}
    for entry in bookmaker_odds:
        for side in ("home", "draw", "away"):
            odd = entry.get(side)
            if not odd or odd <= 1.0:
                continue
            current = best[side].get("odds", 0.0)
            if odd > current:
                best[side] = {"odds": odd, "bookmaker": entry.get("bookmaker", "?")}
    return best


def find_outlier(
    bookmaker_odds: list[dict[str, Any]],
    consensus: dict[str, float],
    threshold_pct: float,
    min_bm: int = 3,
) -> dict[str, Any] | None:
    """Largest edge vs consensus.

    Returns {side, value_pct, bookmaker, odds, outlier_liquidity,
    bookmaker_count} or None. ``outlier_liquidity`` is "thin" when fewer
    than ``min_bm`` bookmakers actually reported odds -- a lone 33.0 quote
    among 12 bookmakers is more likely a data error than real value, so
    consumers (``score_signal``) must not let it drive the value signal.
    The outlier itself is still returned for the audit trail.
    """
    if not bookmaker_odds or threshold_pct <= 0:
        return None
    n_with_odds = sum(
        1 for e in bookmaker_odds
        if any(e.get(side) for side in ("home", "draw", "away"))
    )
    best_edge: dict[str, Any] | None = None
    for entry in bookmaker_odds:
        for side in ("home", "draw", "away"):
            odd = entry.get(side)
            cons = consensus.get(side, 0.0)
            if not odd or cons <= 0:
                continue
            edge = (odd - cons) / cons * 100.0
            if edge >= threshold_pct:
                if best_edge is None or edge > best_edge["value_pct"]:
                    best_edge = {
                        "side": side,
                        "value_pct": round(edge, 2),
                        "bookmaker": entry.get("bookmaker", "?"),
                        "odds": odd,
                        "outlier_liquidity": "thin" if n_with_odds < min_bm else "ok",
                        "bookmaker_count": n_with_odds,
                    }
    return best_edge


def score_signal(
    bookmaker_odds: list[dict[str, Any]],
    consensus: dict[str, float],
    outlier: dict[str, Any] | None,
    home_form: str | None,
    away_form: str | None,
    has_odds: bool,
) -> int:
    """0-100 score breakdown:
        value_signal      40
        form_edge         30
        info_clarity      20
        liquidity         10
    """
    score = 0

    if has_odds and outlier:
        # P3-3: a THIN outlier (fewer than ``min_bm`` bookmakers reported
        # odds) is not value -- a lone divergent quote is treated as
        # has-odds-only so it cannot inflate the score.
        if (outlier.get("outlier_liquidity") or "ok") != "thin":
            edge = min(outlier["value_pct"], 20.0)
            score += int((edge / 20.0) * 40)
        else:
            score += 10
    elif has_odds:
        score += 10

    hw = _form_wins(home_form)
    aw = _form_wins(away_form)
    if hw is not None and aw is not None:
        diff = abs(hw - aw)
        score += int(min(diff, 3) / 3 * 30)
    elif home_form or away_form:
        score += 8

    if home_form and away_form:
        score += 20
    elif home_form or away_form:
        score += 10

    liquidity = min(len(bookmaker_odds), 12)
    score += int((liquidity / 12) * 10)

    return max(0, min(100, score))


def _form_wins(form: str | None) -> int | None:
    if not form:
        return None
    return sum(1 for c in form if c == "W")
