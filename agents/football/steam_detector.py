"""Steam Move Detector + Reverse Line Move (RLM) Detection.

Analyses timestamped odds-series from NowGoal's ``fetch_odds_trend`` to detect:

  1. **Steam Move** — sudden large odds shift in < 15 minutes, typically from
     sharp syndicate money landing on one side.
  2. **Reverse Line Move (RLM)** — odds moving OPPOSITE to the majority of
     public bets.  When 70%+ public is on Home but Home odds are DRIFTING
     (shortening away), sharp money is on Away.
  3. **Multi-bookmaker Agreement** — same direction move across 3+ bookmakers
     within a short window confirms the move is genuine (not a single-book
     outlier).

These signals feed into the signal engine as a new evidence group
``market_intelligence`` alongside the existing model/statistical/movement
groups.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEAM_MOVE_THRESHOLD_PCT = 3.0   # > 3% move in < 15 min = steam
STEAM_WINDOW_MINUTES = 15        # max time window for steam detection
RLM_PUBLIC_THRESHOLD = 65.0      # public bet % above this = heavy public
RLM_ODDS_DRIFT_THRESHOLD = 1.5   # odds moving > 1.5% against public = RLM
MULTI_BOOK_AGREEMENT_MIN = 5     # min bookmakers agreeing on direction (raised 3->5 to require sharp consensus: 5/10=50% medium, 8/10=80% high; 3 was noise)
MULTI_BOOK_MIN_DRIFT_PCT = 1.5   # per-book drift must exceed this to count as agreement (was 0% -> 0.5% noise counted)
VELOCITY_HIGH_THRESHOLD = 2.0    # > 2% per hour = fast move


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        cleaned = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _minutes_between(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds()) / 60.0


def _pct_change(old: float, new: float) -> float:
    if old <= 0:
        return 0.0
    return (old - new) / old * 100.0


def _implied_prob(odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


# ---------------------------------------------------------------------------
# Steam Move Detection
# ---------------------------------------------------------------------------

def detect_steam_moves(
    bookmaker_series: list[dict[str, Any]],
    *,
    threshold_pct: float = STEAM_MOVE_THRESHOLD_PCT,
    window_minutes: float = STEAM_WINDOW_MINUTES,
) -> list[dict[str, Any]]:
    """Detect steam moves from timestamped odds series.

    Parameters
    ----------
    bookmaker_series : list of ``{"ts", "home", "draw", "away", ...}`` rows
        from NowGoal ``_parse_trend_series`` (pre-match only, minute == "").
    threshold_pct : minimum % change to qualify as steam.
    window_minutes : maximum time window for the move.

    Returns
    -------
    list of steam move dicts, each:
        {"side": "home"|"draw"|"away",
         "magnitude_pct": float,
         "window_minutes": float,
         "from_odds": float,
         "to_odds": float,
         "from_ts": str,
         "to_ts": str,
         "velocity_per_hour": float}
    """
    if not bookmaker_series or len(bookmaker_series) < 2:
        return []

    # Sort chronologically
    rows = sorted(bookmaker_series, key=lambda r: r.get("ts") or "")
    steam_moves: list[dict[str, Any]] = []

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r_i, r_j = rows[i], rows[j]
            t_i, t_j = _parse_ts(r_i.get("ts")), _parse_ts(r_j.get("ts"))
            if not t_i or not t_j:
                continue

            diff_min = _minutes_between(t_i, t_j)
            if diff_min > window_minutes or diff_min < 0.5:
                continue

            for side in ("home", "draw", "away"):
                odds_i = r_i.get(side)
                odds_j = r_j.get(side)
                if not odds_i or not odds_j or odds_i <= 1.0 or odds_j <= 1.0:
                    continue

                change = _pct_change(odds_i, odds_j)
                if abs(change) >= threshold_pct:
                    velocity = change / (diff_min / 60.0) if diff_min > 0 else 0.0
                    steam_moves.append({
                        "side": side,
                        "magnitude_pct": round(change, 2),
                        "window_minutes": round(diff_min, 1),
                        "from_odds": round(odds_i, 4),
                        "to_odds": round(odds_j, 4),
                        "from_ts": r_i.get("ts"),
                        "to_ts": r_j.get("ts"),
                        "velocity_per_hour": round(velocity, 2),
                    })

    # Deduplicate: keep largest move per side
    best: dict[str, dict] = {}
    for mv in steam_moves:
        s = mv["side"]
        if s not in best or abs(mv["magnitude_pct"]) > abs(best[s]["magnitude_pct"]):
            best[s] = mv
    return list(best.values())


# ---------------------------------------------------------------------------
# Reverse Line Move (RLM) Detection
# ---------------------------------------------------------------------------

def detect_rlm(
    odds_series: list[dict[str, Any]],
    public_bet_pct: dict[str, float] | None = None,
    *,
    public_threshold: float = RLM_PUBLIC_THRESHOLD,
    drift_threshold: float = RLM_ODDS_DRIFT_THRESHOLD,
) -> dict[str, Any] | None:
    """Detect Reverse Line Move: odds moving against heavy public betting.

    Parameters
    ----------
    odds_series : chronological odds rows (pre-match).
    public_bet_pct : {"home": 72.0, "draw": 15.0, "away": 13.0} if available.
    public_threshold : min public % to consider "heavy public".
    drift_threshold : min odds drift % against public to trigger RLM.

    Returns
    -------
    None if no RLM detected, else:
        {"side": "away",           # the side sharp money is on
         "public_side": "home",    # where public is betting
         "public_pct": 72.0,
         "odds_drift_pct": -2.3,   # odds moving against public
         "confidence": "HIGH",     # HIGH/MEDIUM/LOW based on magnitude
         "reason": "Public 72% on Home but Home odds drifting +2.3%"}
    """
    if not odds_series or len(odds_series) < 2:
        return None

    rows = sorted(odds_series, key=lambda r: r.get("ts") or "")
    first_row, last_row = rows[0], rows[-1]

    # Get opening and latest odds
    sides_data: dict[str, tuple[float | None, float | None]] = {}
    for side in ("home", "draw", "away"):
        o_open = first_row.get(side)
        o_close = last_row.get(side)
        if o_open and o_close and o_open > 1.0 and o_close > 1.0:
            sides_data[side] = (o_open, o_close)

    if not sides_data:
        return None

    # If no public bet data, infer from odds movement direction
    if not public_bet_pct:
        # Look for sides where odds are SHORTENING (money coming in)
        shortening: dict[str, float] = {}
        for side, (o_open, o_close) in sides_data.items():
            drift = _pct_change(o_open, o_close)
            if drift > 0:  # odds shortened = money on this side
                shortening[side] = drift

        if not shortening:
            return None

        # The "public" side is the one with most shortening
        public_side = max(shortening, key=shortening.get)
        # RLM = another side is also shortening strongly
        for side, drift in shortening.items():
            if side != public_side and drift >= drift_threshold:
                return {
                    "side": side,
                    "public_side": public_side,
                    "public_pct": None,
                    "odds_drift_pct": round(drift, 2),
                    "confidence": "MEDIUM",
                    "reason": f"Multi-side shortening: {side} drift {drift:.1f}% alongside {public_side}",
                }
        return None

    # With public bet data: check for divergence
    heavy_public = {
        side: pct for side, pct in public_bet_pct.items()
        if pct >= public_threshold
    }
    if not heavy_public:
        return None

    for pub_side, pub_pct in heavy_public.items():
        # Check if pub_side odds are DRIFTING (getting worse for public)
        if pub_side in sides_data:
            o_open, o_close = sides_data[pub_side]
            drift = _pct_change(o_open, o_close)  # positive = shortened
            if drift < -drift_threshold:  # negative = drifted = bad for public
                # Find the side that's shortening (sharp side)
                sharp_side = None
                for side, (oo, oc) in sides_data.items():
                    if side != pub_side:
                        s_drift = _pct_change(oo, oc)
                        if s_drift > 0:
                            sharp_side = side
                            break

                if sharp_side:
                    confidence = "HIGH" if pub_pct >= 75 else "MEDIUM"
                    return {
                        "side": sharp_side,
                        "public_side": pub_side,
                        "public_pct": pub_pct,
                        "odds_drift_pct": round(drift, 2),
                        "confidence": confidence,
                        "reason": f"Public {pub_pct:.0f}% on {pub_side} but {pub_side} odds drifting {abs(drift):.1f}% → sharp on {sharp_side}",
                    }

    return None


# ---------------------------------------------------------------------------
# Multi-Bookmaker Agreement
# ---------------------------------------------------------------------------

def check_multi_book_agreement(
    bookmaker_trends: list[dict[str, Any]],
    *,
    min_agreement: int = MULTI_BOOK_AGREEMENT_MIN,
) -> dict[str, Any]:
    """Check if multiple bookmakers agree on odds direction.

    Parameters
    ----------
    bookmaker_trends : list of bookmaker trend dicts from NowGoal,
        each with ``{"cid", "name", "h2h": [...], "ah": [...], "ou": [...]}``.

    Returns
    -------
    {"agreement_count": int,
     "total_bookmakers": int,
     "agreement_pct": float,
     "dominant_side": "home"|"draw"|"away"|None,
     "dominant_direction": "shortening"|"drifting"|None}
    """
    if not bookmaker_trends:
        return {
            "agreement_count": 0,
            "total_bookmakers": 0,
            "agreement_pct": 0.0,
            "dominant_side": None,
            "dominant_direction": None,
        }

    # For each bookmaker, determine which side is shortening
    side_votes: dict[str, int] = {"home": 0, "draw": 0, "away": 0}
    total = 0

    for bm in bookmaker_trends:
        h2h = bm.get("h2h") or []
        if len(h2h) < 2:
            continue

        rows = sorted(h2h, key=lambda r: r.get("ts") or "")
        first_row, last_row = rows[0], rows[-1]

        # Find which side shortened the most, but only count if drift exceeds threshold
        best_side = None
        best_drift = 0.0
        for side in ("home", "draw", "away"):
            o_open = first_row.get(side)
            o_close = last_row.get(side)
            if o_open and o_close and o_open > 1.0 and o_close > 1.0:
                drift = _pct_change(o_open, o_close)
                if drift > best_drift:
                    best_drift = drift
                    best_side = side

        if best_side and best_drift >= MULTI_BOOK_MIN_DRIFT_PCT:
            side_votes[best_side] += 1
            total += 1

    if total == 0:
        return {
            "agreement_count": 0,
            "total_bookmakers": 0,
            "agreement_pct": 0.0,
            "dominant_side": None,
            "dominant_direction": None,
        }

    dominant_side = max(side_votes, key=side_votes.get)
    agreement_count = side_votes[dominant_side]
    agreement_pct = (agreement_count / total * 100) if total > 0 else 0.0

    return {
        "agreement_count": agreement_count,
        "total_bookmakers": total,
        "agreement_pct": round(agreement_pct, 1),
        "dominant_side": dominant_side,
        "dominant_direction": "shortening" if agreement_count >= min_agreement else None,
    }


# ---------------------------------------------------------------------------
# Combined Market Intelligence
# ---------------------------------------------------------------------------

def analyze_market_intelligence(
    bookmaker_trends: list[dict[str, Any]],
    public_bet_pct: dict[str, float] | None = None,
    model_side: str | None = None,
) -> dict[str, Any]:
    """Full market intelligence analysis combining steam, RLM, and agreement.

    Returns a dict suitable for the signal engine's ``market_intelligence``
    evidence group.
    """
    # Flatten all h2h series for steam/RLM detection
    all_h2h: list[dict[str, Any]] = []
    for bm in bookmaker_trends:
        all_h2h.extend(bm.get("h2h") or [])

    # Detect steam moves (use first bookmaker's series as primary)
    primary_h2h = bookmaker_trends[0].get("h2h") if bookmaker_trends else []
    steam_moves = detect_steam_moves(primary_h2h)

    # Detect RLM
    rlm = detect_rlm(all_h2h, public_bet_pct)

    # Check multi-book agreement
    agreement = check_multi_book_agreement(bookmaker_trends)

    # Determine intelligence side
    intel_side = None
    intel_confidence = 0.0
    intel_reasons: list[str] = []

    # Steam move is strongest signal
    if steam_moves:
        strongest = max(steam_moves, key=lambda m: abs(m["magnitude_pct"]))
        intel_side = strongest["side"]
        intel_confidence = min(1.0, abs(strongest["magnitude_pct"]) / 10.0)
        intel_reasons.append(f"Steam on {strongest['side']}: {strongest['magnitude_pct']:+.1f}% in {strongest['window_minutes']}min")

    # RLM reinforces
    if rlm:
        if intel_side is None:
            intel_side = rlm["side"]
            intel_confidence = 0.7 if rlm["confidence"] == "HIGH" else 0.5
        elif rlm["side"] == intel_side:
            intel_confidence = min(1.0, intel_confidence + 0.2)
        intel_reasons.append(rlm["reason"])

    # Multi-book agreement reinforces
    if agreement["dominant_direction"] == "shortening" and agreement["agreement_count"] >= MULTI_BOOK_AGREEMENT_MIN:
        if intel_side is None:
            intel_side = agreement["dominant_side"]
            intel_confidence = 0.4
        elif agreement["dominant_side"] == intel_side:
            intel_confidence = min(1.0, intel_confidence + 0.15)
        intel_reasons.append(f"{agreement['agreement_count']}/{agreement['total_bookmakers']} books agree on {agreement['dominant_side']}")

    # Agreement with model
    model_agreement = 0.5
    if intel_side and model_side:
        model_agreement = 1.0 if intel_side == model_side else 0.0

    return {
        "usable": bool(steam_moves or rlm or agreement["dominant_direction"]),
        "side": intel_side,
        "confidence": round(intel_confidence, 3),
        "model_agreement": model_agreement,
        "steam_moves": steam_moves,
        "rlm": rlm,
        "agreement": agreement,
        "reasons": intel_reasons,
    }
