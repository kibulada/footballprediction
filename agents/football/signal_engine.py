"""Market-aware Signal Engine + Best Pick Ranker (additive layer).

Sits ON TOP of the existing prediction engine (Elo+Poisson+calibration) and
the existing decision engine. It does NOT recompute or replace any of them;
it consumes their output (``model_probs``, market odds, movement) and the
Asian Handicap market that NowGoal already exposes, then:

  1. builds candidate signals -- BTTS Yes/No, Over/Under 2.5, and Asian
     Handicap quarter lines (Home/Away +-0.25, plus the exact line the
     source quotes);
  2. scores each signal over DISTINCT evidence groups (existing model,
     statistical support, market price, odds movement, late movement,
     data quality, optional team context);
  3. ranks deterministically and selects the single Best Pick on the
     ABSOLUTE strength of the top signal (a close runner-up never voids it);
     returns NO BET only when the top signal itself is not strong enough.

Determinism (S36): every function is pure. Same input -> same scores ->
same ranking -> same Best Pick. No LLM, no randomness, no I/O in the core.

Asian Handicap semantics (S14): a quarter line (e.g. Away +0.25) is settled
as half stake on the two adjacent half-lines, so draw = half win for the
+0.25 side. ``ah_settle`` and ``ah_win_prob`` implement this exactly and are
used both for scoring and for backtest settlement.

No data is invented (S3/S30): a missing market/opening/history degrades the
signal (data-quality/confidence down, group excluded) and is reported as
UNAVAILABLE -- never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import MAX_GOALS, poisson_matrix
from .steam_detector import analyze_market_intelligence
from .clv_tracker import log_clv_entry, clv_gate
from .pick_gates import (
    DEFAULT_ELO_COLLISION_EPS,
    DEFAULT_ELO_MAX,
    DEFAULT_ELO_MIN,
    DEFAULT_MAX_DEV_PP,
    DIRECTIONAL_MARKETS,
    agreement_gate,
    band_source,
    elo_evidence_scope,
    elo_integrity_gate,
    is_directional_selection,
    is_low_scoring_selection,
    lambda_1x2_gate,
    lambda_total_gate,
    market_implied_total,
    price_gate,
    resolve_lambda_total_band,
    source_consistency_gate,
)
from .tie_state import tie_state_note

# --------------------------------------------------------------------------
# Configurable defaults (mirrored in config/football.json -> models.signal_engine)
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "model": 0.35,          # Group A -- existing prediction engine support
    "statistical": 0.10,    # Group B -- empirical form/H2H frequencies
    "market": 0.15,         # Group C -- model vs market price (edge)
    "movement": 0.15,       # Group D -- opening -> current price/line move
    "late_movement": 0.00,  # Group D -- penalty only (not scored)
    "data_quality": 0.05,   # Group F -- input completeness
    "team_context": 0.00,   # Group E -- lineups/injuries (disabled)
    "market_intelligence": 0.15,  # Group G -- steam/RLM/multi-book agreement
    "ah_bonus": 0.05,       # Group H -- market variety bonus for AH picks
}

MIN_EDGE_PP = 3.0           # minimum meaningful model-market edge (S19)
CONFLICT_PP = 8.0           # model < market by this -> conflict penalty (S18)
BEST_PICK_MARGIN = 0.06     # top1 - top2 score must clear this (S22)
NO_BET_SCORE = 0.45         # best score below this -> NO BET (S24)
MIN_CONFLUENCE = 2          # minimum agreeing evidence groups (S20)
MIN_DATA_QUALITY = 0.30     # completeness floor for a bettable signal (S24)

# F2 (evidence floor): a form window shorter than this many finished matches
# per team carries no empirical signal -- the statistical component is noise
# and the model's λ may be nothing but a prior Elo rating. Matches the
# ``model_gates`` MIN_FORM_DEPTH used by the 1X2 decision layer.
MIN_EVIDENCE_FORM_MATCHES = 3

# F3 (1X2 reconciliation): when the independent 1X2 decision layer returned
# one of these non-actionable decisions, the market-aware pick is capped at
# MEDIUM and the disagreement is surfaced on the card -- the signal engine
# must never present a pick stronger than the model itself.
NON_ACTIONABLE_DECISIONS = frozenset({"NO BET", "NO CLEAR DECISION", "MARKET PRIOR"})

# P1-2: source-confidence gate. When 3+ critical fields (match, form, h2h)
# all carry ``confidence == "LOW"`` the data the engine would lean on is
# untrustworthy -- every field's primary and secondary sources disagree, so
# any aggregate score is built on noise. The engine still runs (the audit
# trail wants the full output), but the resulting ``best_pick`` is vetoed
# and the reason surfaced in the card so the user sees an honest NO BET.
EVIDENCE_GATE_FIELDS = ("match", "form", "h2h")


def evidence_gate(
    confidence_map: dict[str, str] | None,
    *,
    max_low: int = 2,
) -> tuple[bool, str | None]:
    """Return (passed, reason).

    ``confidence_map`` is the per-field confidence map ``{field_name:
    "HIGH"|"MEDIUM"|"LOW"}`` -- the ``unified_dict["confidence"]`` block
    from ``MultiSourceAggregator.to_dict``. ``passed`` is True when fewer
    than ``max_low + 1`` critical fields are LOW (default: tolerate at most
    2 LOW, veto at 3). ``reason`` is set to a short human-readable string
    when the gate vetoes.

    Accepts None / empty dict (no data -> pass, never invent a veto).
    """
    if not isinstance(confidence_map, dict) or not confidence_map:
        return True, None
    low_fields = [
        f for f in EVIDENCE_GATE_FIELDS
        if confidence_map.get(f) == "LOW"
    ]
    if len(low_fields) > max_low:
        return False, (
            f"source confidence too low ({len(low_fields)}/"
            f"{len(EVIDENCE_GATE_FIELDS)} critical fields LOW: "
            f"{', '.join(low_fields)})"
        )
    return True, None

# Confidence category thresholds (S23). Score is 0..1.
VERY_HIGH_SCORE = 0.78
HIGH_SCORE = 0.65
MEDIUM_SCORE = 0.52
LOW_SCORE = 0.40

# Movement thresholds (S7/S9/S10).
MOVEMENT_PCT_THRESHOLD = 2.0   # min price move to count as directional
REVERSAL_PCT = 1.5             # min reverse swing to flag a reversal

# Phase 5.4: maximum score for uncalibrated leagues (prevents misleading high
# confidence when the model has no validated per-league calibration fit).
UNCALIBRATED_SCORE_MAX = 0.50  # 50/100 cap

# Calibration completeness weights (mirror calibration.py _completeness).
# Used to adjust completeness when scoring components are disabled --
# a disabled field should not penalize the data-quality score.
_CALIBRATION_WEIGHTS: dict[str, float] = {
    "model": 0.45,          # odds (0.25) + xG (0.20)
    "statistical": 0.55,    # form (0.40) + H2H (0.15)
    "market": 0.00,         # covered by model (odds already in model)
    "market_intelligence": 0.00,  # derived from odds, not a data source
    "movement": 0.00,       # derived from odds snapshots
    "data_quality": 0.00,   # meta-component
    "team_context": 0.00,   # lineups/injuries (separate source)
}


def _adjust_completeness_for_weights(
    completeness: float,
    weights: dict[str, float],
) -> float:
    """Recalculate completeness based on which scoring components are active.

    Option A: when a scoring component is disabled (weight == 0), its data
    sources should not penalize the completeness calculation.  For example,
    ``team_context`` disabled means lineups/injuries are irrelevant, so
    missing lineup data should NOT count as incomplete.
    """
    total_active_calib = sum(
        _CALIBRATION_WEIGHTS.get(k, 0.0)
        for k, w in weights.items()
        if w > 0 and k not in ("late_movement", "data_quality")
    )
    if total_active_calib <= 0:
        return completeness  # nothing scored -> keep raw value
    # Scale completeness: only count the portion of calibration data that
    # contributes to *active* scoring components.
    #   active_weight / total_calibration_weight tells us what fraction of
    #   calibration data actually matters for this scoring configuration.
    #   When all components are active, ratio == 1.0 (no change).
    ratio = total_active_calib / sum(_CALIBRATION_WEIGHTS.values())
    return min(1.0, completeness / ratio) if ratio > 0 else completeness

# Layer 3 -- repeated-query stability guard (defaults; ``models.signal_engine.stability``
# in config overrides). ``score_threshold_fallback`` is the p-th percentile of the
# logged repeated-query score-delta distribution once enough post-Layer-1/2 data
# exists; until then this fallback is used (calibration: prediction_log.stability_calibration).
STABILITY_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_age_seconds": 21600,          # prior pick older than this -> re-evaluate
    "score_threshold_fallback": 0.10,  # |top - prior| score delta (fallback until calibrated) -- raised from 0.05 to reduce flip-flopping
    "score_threshold_percentile": 0.95,
    "score_threshold_min_samples": 20,
    "market_move_threshold_pct": 3.0,  # pick-side price lengthened >= this vs opening -> adverse
    "line_move_threshold": 0.25,       # handicap/goal-line move >= this -> significant
    "no_bet_hold": 0.45,               # held pick must stay above the NO-BET floor
}

# Layer 4 -- confidence rule (documented prior, not a hidden inconsistency):
# movement is ONE of seven evidence components (weight 0.15). An opposing
# market move does not by itself cap confidence -- model (0.35) + statistical
# (0.25) + market edge (0.20) can still yield HIGH. Mirrored in
# config/football.json -> models.signal_engine.movement_confidence_note.

# Double-count fix (2026-08-17): the LATE move no longer contributes to the
# weighted score (weight 0.00) -- it reads the same price series as movement,
# so scoring both inflated market-following signals by 0.25 total. Its
# retained role is a PENALTY: when the market's last move is AGAINST the
# pick with meaningful strength, the pick cannot stand above MEDIUM (the
# market is moving away from the selection into the close).
LATE_AGAINST_MIN_STRENGTH = 0.5   # |late_strength| needed to cap
LATE_AGAINST_CAP = "MEDIUM"       # cap applied to a HIGH/VERY HIGH pick

# Canonical AH quarter lines the engine always evaluates (S13).
AH_LINES = (-0.25, 0.25)


# --------------------------------------------------------------------------
# Implied probability (S11)
# --------------------------------------------------------------------------

def implied_probability(odds: float) -> float | None:
    """Raw implied probability 1/odds, or None when odds is invalid."""
    if not odds or odds <= 1.0:
        return None
    return 1.0 / odds


def normalize_implied(probs: dict[str, float]) -> dict[str, float] | None:
    """Margin-free (overround-removed) implied probabilities."""
    raw = {k: (1.0 / v if v and v > 1.0 else 0.0) for k, v in probs.items()}
    total = sum(raw.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in raw.items()}


def fair_pair_implied(odds_a: float, odds_b: float) -> tuple[float, float] | None:
    """Margin-free implied for a two-outcome pair (Over/Under, BTTS, AH)."""
    ia, ib = implied_probability(odds_a), implied_probability(odds_b)
    if ia is None or ib is None:
        return None
    total = ia + ib
    if total <= 0:
        return None
    return ia / total, ib / total


def excess_probability(p: float, n_outcomes: int) -> float:
    """Distance above the uniform baseline, scaled 0..1 (0 = no support)."""
    if n_outcomes <= 1:
        return 0.0
    base = 1.0 / n_outcomes
    return max(0.0, min(1.0, (p - base) / (1.0 - base)))


# --------------------------------------------------------------------------
# Asian Handicap: line semantics + settlement (S14)
# --------------------------------------------------------------------------

def _quarter_components(line: float) -> list[tuple[float, float]]:
    """Split a handicap line into (half_line, weight) stake components.

    A quarter line (e.g. -0.25) = half stake on each adjacent half-line
    (-0.5 and 0.0). Integer/half lines are a single component.
    """
    r = round(line * 4)
    if r % 2 == 0:  # integer or half line
        return [(line, 1.0)]
    return [(line - 0.25, 0.5), (line + 0.25, 0.5)]


def ah_return(home_goals: int, away_goals: int, line: float, side: str = "home") -> float:
    """Expected return (0..1) of an Asian Handicap bet.

    ``line`` is the HOME handicap (e.g. -0.25 = home gives a quarter ball).
    ``side`` = "home"|"away". A quarter line is split into its two adjacent
    half-lines (half stake each). 1.0 = full win, 0.75 = half win, 0.5 =
    push, 0.25 = half loss, 0.0 = full loss.
    """
    sign = 1.0 if side == "home" else -1.0
    total = 0.0
    for hl, w in _quarter_components(line):
        m = sign * (home_goals - away_goals + hl)
        if m > 0:
            total += w
        elif m == 0:
            total += 0.5 * w
    return total


def _side_line(line: float, side: str) -> float:
    """Label line for a bet on ``side``: away side quotes the negated line."""
    return line if side == "home" else -line


def _canonical_ah_side(
    p_home_1x2: float,
    p_home: float,
    home_odds: float | None,
    away_odds: float | None,
) -> str:
    """Deterministic canonical side of an AH line (Layer 2).

    Uses the model's 1X2 direction (p_home_1x2) to determine which team
    is stronger, then maps that to the AH side:
    - If model favors Home (p_home_1x2 > 0.50) → canonical side = "home"
    - If model favors Away (p_home_1x2 < 0.50) → canonical side = "away"
    - If model is neutral (p_home_1x2 == 0.50) → default "home"
      (deterministic, no tiebreakers that can flip)

    CRITICAL: The AH line probability (p_home) is NEVER used for side
    determination because p_home for positive handicaps (e.g. +1.75) is
    always high regardless of which team is stronger.

    This fixes the bug where AH lines like -1.75 (Away gives 1.75 goals)
    incorrectly canonicalize to "home" because p_home (Home covers +1.75)
    is always high for positive home handicaps.
    """
    # Primary: use model 1X2 direction (which team is stronger)
    if p_home_1x2 > 0.50:  # model favors Home
        return "home"
    if p_home_1x2 < 0.50:  # model favors Away
        return "away"
    # Model is exactly neutral (0.50): deterministic default = home
    # No AH probability or odds tiebreakers (they can flip unpredictably)
    return "home"


def ah_settle(home_goals: int, away_goals: int, line: float, side: str) -> dict[str, Any]:
    """Settle an AH bet: {result, return_value, label}.

    ``line`` is the HOME handicap. ``side`` = "home"|"away".
    result in {win, half_win, push, half_loss, loss}.
    """
    r = ah_return(home_goals, away_goals, line, side)
    if r >= 1.0 - 1e-9:
        result, label = "win", "WIN"
    elif abs(r - 0.75) < 1e-9:
        result, label = "half_win", "HALF WIN"
    elif abs(r - 0.5) < 1e-9:
        result, label = "push", "PUSH"
    elif abs(r - 0.25) < 1e-9:
        result, label = "half_loss", "HALF LOSS"
    else:
        result, label = "loss", "LOSS"
    return {"result": result, "return_value": r, "label": label}


def ah_win_prob(matrix: list[list[float]], line: float, side: str) -> float:
    """Win-equivalent probability of an AH side from a Poisson score matrix.

    Sums P(score) * ah_return over all scorelines; includes pushes and half
    results fractionally, so it is directly comparable to a margin-free
    market implied probability.
    """
    total = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            total += matrix[h][a] * ah_return(h, a, line, side)
    return total


# --------------------------------------------------------------------------
# Asian Handicap extraction from the NowGoal-normalized odds payload
# --------------------------------------------------------------------------

def extract_asian_handicap(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """AH rows ``{line, home, away, home_open, away_open, bookmaker}``.

    Reads the ``asian_handicap`` market NowGoal emits (key with outcomes
    Home/Away + ``point`` line + ``opening_price``). A row must carry a
    plausible line and both sides' prices to be kept. Empty when the source
    exposes no AH market (never invented).
    """
    rows: list[dict[str, Any]] = []
    for bm in (payload or {}).get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "asian_handicap":
                continue
            home = away = line = home_open = away_open = line_open = None
            for outcome in market.get("outcomes", []):
                name = (outcome.get("name") or "").lower()
                price = outcome.get("price")
                opening = outcome.get("opening_price")
                point = outcome.get("point")
                opening_point = outcome.get("opening_point")
                if name == "home":
                    home = price
                    home_open = opening
                    line = point
                    line_open = opening_point
                elif name == "away":
                    away = price
                    away_open = opening
            if line is None or home is None or away is None:
                continue
            if home <= 1.0 or away <= 1.0:
                continue
            rows.append({
                "line": float(line),
                "home": float(home),
                "away": float(away),
                "home_open": float(home_open) if home_open else None,
                "away_open": float(away_open) if away_open else None,
                "line_open": float(line_open) if line_open is not None else None,
                "bookmaker": bm.get("title", "?"),
            })
    return rows


def ah_consensus(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Median line + median home/away prices (+ opening medians) across rows.

    Phase 6.1 FIX: group rows by line FIRST, then pick the line with the most
    bookmakers. Medians for home/away odds are computed ONLY from rows sharing
    that line — mixing odds from different lines (e.g. Home -1.25 @ 1.83 and
    Home +0.25 @ 1.93) produced invalid consensus data.
    """
    if not rows:
        return None

    def _med(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return s[len(s) // 2]

    # Group by line — each group contains bookmakers quoting the same handicap.
    by_line: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        line = r.get("line")
        if line is None:
            continue
        by_line.setdefault(line, []).append(r)

    if not by_line:
        return None

    # Pick the line with the most bookmakers (most consensus).
    best_line = max(by_line, key=lambda l: len(by_line[l]))
    same_line = by_line[best_line]

    # Medians from same-line rows only — never mixed across lines.
    home_prices = [r["home"] for r in same_line if r.get("home") is not None]
    away_prices = [r["away"] for r in same_line if r.get("away") is not None]
    home_open = [r["home_open"] for r in same_line if r.get("home_open")]
    away_open = [r["away_open"] for r in same_line if r.get("away_open")]
    line_open = [r["line_open"] for r in same_line if r.get("line_open") is not None]

    return {
        "line": best_line,
        "home": _med(home_prices),
        "away": _med(away_prices),
        "home_open": _med(home_open) if home_open else None,
        "away_open": _med(away_open) if away_open else None,
        "line_open": _med(line_open) if line_open else None,
        "n": len(same_line),
        "n_total": len(rows),
        "lines_found": sorted(by_line.keys()),
    }


def ou_consensus(market_totals: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    """One Over/Under pair ``{line, over, under}`` from ``extract_market_totals``.

    Prefers the 2.5 line (the signal engine's target); otherwise the quoted
    line closest to 2.5. Never fabricates -- returns None when no complete
    Over/Under pair exists.
    """
    pairs: dict[float, dict[str, Any]] = {}
    for label, data in (market_totals or {}).items():
        if not label.startswith("Over ") or not data.get("odds"):
            continue
        line = data.get("point")
        if line is None:
            continue
        over = data.get("odds")
        under = (market_totals.get(f"Under {line}") or {}).get("odds")
        if not over or not under or over <= 1.0 or under <= 1.0:
            continue
        pairs[float(line)] = {"line": float(line), "over": float(over), "under": float(under)}
    if not pairs:
        return None
    if 2.5 in pairs:
        return pairs[2.5]
    return min(pairs.values(), key=lambda p: abs(p["line"] - 2.5))


# --------------------------------------------------------------------------
# Movement features (S7/S8/S9/S10)
# --------------------------------------------------------------------------

def price_move_pct(opening: float | None, current: float | None) -> float | None:
    """(current - opening)/opening * 100. Positive = price lengthened."""
    if not opening or not current or opening <= 1.0 or current <= 1.0:
        return None
    return round((current - opening) / opening * 100.0, 2)


def movement_features(
    *,
    opening: float | None,
    current: float | None,
    opening_line: float | None = None,
    current_line: float | None = None,
    late_direction: float | None = None,
    late_strength: float | None = None,
) -> dict[str, Any]:
    """Direction/magnitude/line/reversal summary for one selection.

    ``opening``/``current`` are the SELECTION's own price. Shortening
    (current < opening) is money coming in on the selection -> direction
    "toward". ``late_direction`` (+1 toward / -1 away / 0 none) and
    ``late_strength`` (0..1) come from the multi-snapshot movement layer when
    available.
    """
    move_pct = price_move_pct(opening, current)
    if move_pct is None:
        return {"status": "UNAVAILABLE", "direction": "none", "magnitude_pct": 0.0,
                "price_move_pct": None, "line_move": None, "reversal": False,
                "late_direction": late_direction, "late_strength": late_strength}

    # Reversal (S9): opening -> current price moving one way counts once; a
    # late_direction opposite the opening->current move is a reversal.
    direction = "toward" if move_pct < 0 else ("away" if move_pct > 0 else "none")
    reversal = bool(
        late_direction is not None
        and direction in ("toward", "away")
        and late_direction != 0
        and (
            (direction == "toward" and late_direction < 0)
            or (direction == "away" and late_direction > 0)
        )
    )
    line_move = None
    if opening_line is not None and current_line is not None:
        line_move = round(current_line - opening_line, 4)
    return {
        "status": "available",
        "direction": direction,
        "magnitude_pct": abs(move_pct),
        "price_move_pct": move_pct,
        "line_move": line_move,
        "reversal": reversal,
        "late_direction": late_direction,
        "late_strength": late_strength,
    }


def history_movement(
    prices: list[float],
    lines: list[float] | None = None,
    *,
    reversal_pct: float = REVERSAL_PCT,
    move_threshold_pct: float = MOVEMENT_PCT_THRESHOLD,
) -> dict[str, Any]:
    """Movement over a chronological price (and line) series.

    Consumes the accumulated ``odds_snapshot`` history for ONE market side
    and returns the same shape as ``movement_features`` plus richer fields:

      - price movement: opening -> latest, direction, magnitude
      - line movement: first line -> last line (separate from price, S10)
      - consistency: fraction of consecutive moves in the dominant direction
      - reversal: a late move that opposes the overall direction (S9)
      - late movement: direction + strength of the most recent move (S8)

    ``prices`` / ``lines`` are chronological (oldest first). Degrades to
    ``UNAVAILABLE`` below two price points (never fabricated).
    """
    prices = [p for p in (prices or []) if p is not None and p > 1.0]
    lines = [l for l in (lines or []) if l is not None]
    if len(prices) < 2 and len(lines) < 2:
        return {
            "status": "UNAVAILABLE", "n": len(prices), "direction": "none",
            "magnitude_pct": 0.0, "price_move_pct": None, "line_move": None,
            "reversal": False, "consistency": None, "late_direction": 0.0,
            "late_strength": None, "opening_price": None, "latest_price": None,
            "opening_line": None, "latest_line": None,
        }
    line_move = round(lines[-1] - lines[0], 4) if len(lines) >= 2 else None
    opening_price = prices[0] if prices else None
    latest_price = prices[-1] if prices else None

    if len(prices) < 2:
        # Line movement exists but price history is too thin to call a price
        # direction; report line movement only (honest degradation).
        return {
            "status": "available",
            "n": len(prices),
            "direction": "none",
            "magnitude_pct": 0.0,
            "price_move_pct": None,
            "line_move": line_move,
            "reversal": False,
            "consistency": None,
            "late_direction": 0.0,
            "late_strength": None,
            "opening_price": opening_price,
            "latest_price": latest_price,
            "opening_line": lines[0] if lines else None,
            "latest_line": lines[-1] if lines else None,
        }

    first, last = prices[0], prices[-1]
    move_pct = round((last - first) / first * 100.0, 2)
    direction = "toward" if move_pct < 0 else ("away" if move_pct > 0 else "none")
    deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    dominant = move_pct > 0
    consistent = sum(1 for d in deltas if abs(d) > 1e-9 and (d > 0) == dominant)
    consistency = round(consistent / len(deltas), 3) if deltas else None

    last_delta = deltas[-1] if deltas else 0.0
    late_direction = 1.0 if last_delta > 1e-9 else (-1.0 if last_delta < -1e-9 else 0.0)
    late_strength = (
        min(1.0, abs(last_delta) / first * 100.0 / move_threshold_pct)
        if late_direction else 0.0
    )
    reversal = bool(
        direction in ("toward", "away") and late_direction != 0
        and ((direction == "toward" and late_direction > 0)
             or (direction == "away" and late_direction < 0))
    )
    return {
        "status": "available",
        "n": len(prices),
        "direction": direction,
        "magnitude_pct": abs(move_pct),
        "price_move_pct": move_pct,
        "line_move": line_move,
        "reversal": reversal,
        "consistency": consistency,
        "late_direction": late_direction,
        "late_strength": late_strength,
        "opening_price": first,
        "latest_price": last,
        "opening_line": lines[0] if lines else None,
        "latest_line": lines[-1] if lines else None,
    }


def _ah_history_points(
    snapshots: list[dict[str, Any]],
    side: str,
    line: float,
) -> tuple[list[float], list[float]]:
    """(prices, lines) for one AH side from the odds_snapshot series.

    ``prices`` are the side's price at snapshots whose line equals ``line``
    (comparable -> price movement); ``lines`` is the FULL quoted-line series
    (comparable -> line movement). Both chronological.
    """
    prices: list[float] = []
    lines: list[float] = []
    key = "home" if side == "home" else "away"
    for s in snapshots:
        ah = s.get("odds_ah")
        if not isinstance(ah, dict):
            continue
        l = ah.get("line")
        if l is not None:
            lines.append(float(l))
        p = ah.get(key)
        if p is not None and l is not None and abs(float(l) - line) < 1e-9:
            prices.append(float(p))
    return prices, lines


def _ou_history_points(
    snapshots: list[dict[str, Any]],
    sel: str,
    line: float,
) -> tuple[list[float], list[float]]:
    """(prices, lines) for one Over/Under side from the odds_snapshot series."""
    prices: list[float] = []
    lines: list[float] = []
    for s in snapshots:
        ou = s.get("odds_ou")
        if not isinstance(ou, dict):
            continue
        l = ou.get("line")
        if l is not None:
            lines.append(float(l))
        p = ou.get(sel)
        if p is not None and l is not None and abs(float(l) - line) < 1e-9:
            prices.append(float(p))
    return prices, lines


# --------------------------------------------------------------------------
# Statistical support (Group B) -- empirical form/H2H frequencies
# --------------------------------------------------------------------------

def _recent_goals(v: Any) -> list[tuple[int, int]]:
    if not isinstance(v, (list, tuple)) or not v:
        return []
    out: list[tuple[int, int]] = []
    for row in v:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            try:
                out.append((int(row[0]), int(row[1])))
            except (ValueError, TypeError):
                continue
    return out


def _freq(goals: list[tuple[int, int]], fn) -> float | None:
    if not goals:
        return None
    hits = sum(1 for gf, ga in goals if fn(gf, ga))
    return hits / len(goals)


def btts_frequency(goals: list[tuple[int, int]]) -> float | None:
    return _freq(goals, lambda gf, ga: gf > 0 and ga > 0)


def over25_frequency(goals: list[tuple[int, int]]) -> float | None:
    return _freq(goals, lambda gf, ga: gf + ga > 2)


def ah_cover_frequency(goals: list[tuple[int, int]], line: float, side: str) -> float | None:
    """Mean ah_return of a side's own recent scorelines at ``line``."""
    if not goals:
        return None
    vals = [ah_return(gf, ga, line, side) for gf, ga in goals]
    return sum(vals) / len(vals)


def statistical_support(kind: str, stats: dict[str, Any], line: float, side: str) -> float | None:
    """Empirical frequency support for a signal, else None (insufficient data).

    Uses raw scorelines (form) -- an empirical view DISTINCT from the model's
    parametric Poisson/Elo estimate (S16: correlated, so weight stays low and
    configurable). BTTS/Over-2.5 average the two teams' own match frequencies;
    AH uses the relevant side's own recent scorelines.
    """
    hg = _recent_goals(stats.get("home_recent_goals"))
    ag = _recent_goals(stats.get("away_recent_goals"))
    if kind == "btts":
        hf, af = btts_frequency(hg), btts_frequency(ag)
        vals = [v for v in (hf, af) if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)
    if kind == "over":
        hf, af = over25_frequency(hg), over25_frequency(ag)
        vals = [v for v in (hf, af) if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)
    if kind == "ah":
        goals = hg if side == "home" else ag
        return ah_cover_frequency(goals, line, side)
    return None


# --------------------------------------------------------------------------
# Signal candidate + scoring
# --------------------------------------------------------------------------

@dataclass
class Signal:
    market: str                  # "BTTS" | "Total" | "Asian Handicap"
    selection: str               # "Yes"/"No" | "Over 2.5"/"Under 2.5" | "Home +0.25"...
    model_prob: float            # model win-equivalent probability (0..1)
    market_odds: float | None    # offered decimal odds (None if no price)
    implied_prob: float | None   # margin-free market implied (None if no price)
    line: float | None = None    # AH home-handicap line
    side: str | None = None      # AH side ("home"/"away")
    line_key: str = ""           # Layer 2: canonical line identity (e.g. "ah:+1.25")
    edge_pp: float = 0.0
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    movement: dict[str, Any] = field(default_factory=dict)
    confidence: str = "NO SIGNAL"
    reasons: list[str] = field(default_factory=list)
    internal_notes: list[str] = field(default_factory=list)  # P2-3: hidden from summary embed
    # 2026-08-22 pick_gates: a veto is a SEPARATE fact from the score.
    # Zeroing ``score`` to express "vetoed" (the original loser-guard did this)
    # destroys the number the card renders, so a fully-vetoed match showed
    # "Score: 0/100" on every candidate with the generic "no signal reaches the
    # actionable threshold" reason -- the real cause invisible. The score now
    # always reports what the evidence scored; ``vetoed`` decides eligibility.
    vetoed: bool = False
    veto_reasons: list[str] = field(default_factory=list)
    # K1 (2026-08-28): a gate may CAP the label without vetoing (one Elo side
    # on the prior -> directional picks never advertise HIGH). Applied in
    # ``rank_and_pick`` after the coverage floor.
    confidence_cap: str | None = None


def _build_matrix(model_probs: dict[str, Any]) -> list[list[float]] | None:
    lh = model_probs.get("lambda_home")
    la = model_probs.get("lambda_away")
    if not isinstance(lh, (int, float)) or not isinstance(la, (int, float)):
        return None
    if lh <= 0 or la <= 0:
        return None
    return poisson_matrix(float(lh), float(la), rho=0.0)


def _market_component(
    edge_pp: float,
    min_edge: float,
    conflict_pp: float,
    agreement_band_pp: float | None = None,
) -> float:
    """Group C: model vs market confirmation.

    Dua mode:
    - ``agreement_band_pp`` (Plan v3 F4-lite, default via config
      ``market_component_reward_agreement``): MENGHARGAI KESESUAIAN --
      komponen maksimum saat deviasi 0pp dan meluruh linear ke 0 di tepi band
      G2. Implementasi rekomendasi #2 postmortem 2026-08-22 ("balik tanda
      komponen market") yang sebelumnya belum dieksekusi; divergensi adalah
      alarm error, bukan value.
    - Legacy (band None): smooth scale berpusat 0.5 -- +10pp edge full credit,
      -10pp nol. Dipertahankan untuk A/B dan rollback konfigurasi.
    """
    if agreement_band_pp is not None and agreement_band_pp > 0:
        return round(max(0.0, 1.0 - abs(edge_pp) / float(agreement_band_pp)), 3)
    scale = 2.0 * min_edge + 4.0
    return round(max(0.0, min(1.0, 0.5 + edge_pp / scale)), 3)


def _total_disagreement_veto(
    signals: list["Signal"],
    *,
    max_total_dev_pp: float,
) -> list[str]:
    """Loser-guard (2026-08-22): veto TOTAL candidates whose model probability
    deviates from the margin-free market implied by more than
    ``max_total_dev_pp`` percentage points.

    Scope discipline: this gate was tuned ONLY on the 2026-08-21 losing set
    (SV Ried Over 2.5: +28.5pp deviation, lost 1-0). Every WINNING pick in
    that sample stays far below the threshold (Marseille -0.5pp, Arsenal
    -2.5pp), so winning-path behavior is untouched. Returns veto reasons.
    """
    reasons: list[str] = []
    for s in signals:
        if s.market != "Total" or s.implied_prob is None:
            continue
        dev_pp = abs((s.model_prob or 0.0) - s.implied_prob) * 100.0
        if dev_pp > max_total_dev_pp:
            reasons.append(
                f"{s.selection}: deviasi model-pasar {dev_pp:.1f}pp > "
                f"{max_total_dev_pp:.0f}pp (pola over-bias yang kalah 2026-08-21)"
            )
            s.score = 0.0
    return reasons


def _movement_component(mv: dict[str, Any], threshold: float) -> float:
    """Group D: opening->current price move toward the selection."""
    if mv.get("status") != "available":
        return 0.5
    if mv["direction"] == "toward":
        return min(1.0, 0.5 + mv["magnitude_pct"] / (2.0 * threshold))
    if mv["direction"] == "away":
        return max(0.0, 0.5 - mv["magnitude_pct"] / (2.0 * threshold))
    return 0.5


# P1-3: evidence floor thresholds. When key evidence groups (statistical,
# movement) are unavailable the score is built on a thinner basis and the
# headline number must NOT look like a strong conviction pick. Caps are
# applied to the 0..1 score -- see ``_apply_evidence_floor``.
EVIDENCE_FLOOR_DEFAULTS: dict[str, float] = {
    "score_cap_both_unavailable": 0.52,   # neither stat nor movement -> MEDIUM floor
    "score_cap_one_unavailable":  0.65,   # one missing -> MEDIUM upper
}

# P3-4: coverage floor. Confidence is a statement about the EVIDENCE; when
# field coverage is thin (lineups / injuries / standings / recent matches
# unavailable) the label must not overstate it. ``_coverage_floor`` caps
# HIGH/VERY HIGH to MEDIUM below ``downgrade_below`` and everything to LOW
# below ``low_below``.
COVERAGE_FLOOR_DEFAULTS: dict[str, float] = {
    "downgrade_below": 0.40,
    "low_below": 0.25,
}


def _coverage_floor(
    completeness: float,
    confidence: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """P3-4: downgrade a confidence label when coverage is thin."""
    cfg = cfg or {}
    downgrade = float(cfg.get("downgrade_below", COVERAGE_FLOOR_DEFAULTS["downgrade_below"]))
    low = float(cfg.get("low_below", COVERAGE_FLOOR_DEFAULTS["low_below"]))
    if completeness < low:
        return "LOW"
    if completeness < downgrade and confidence in ("VERY HIGH", "HIGH"):
        return "MEDIUM"
    return confidence


_CONF_ORDER = ("NO SIGNAL", "LOW", "MEDIUM", "HIGH", "VERY HIGH")


def _cap_confidence(label: str, cap: str | None) -> str:
    """Never let ``label`` exceed ``cap`` (order NO SIGNAL < LOW < MEDIUM < HIGH < VERY HIGH)."""
    if not cap or label not in _CONF_ORDER or cap not in _CONF_ORDER:
        return label
    return label if _CONF_ORDER.index(label) <= _CONF_ORDER.index(cap) else cap


def _movement_available(mv: dict[str, Any] | None) -> bool:
    """True when the movement block carries a real opening->latest signal."""
    if not isinstance(mv, dict):
        return False
    return mv.get("status") == "available"


def _apply_evidence_floor(
    score: float,
    components: dict[str, float],
    cfg: dict[str, Any] | None = None,
) -> float:
    """Cap the headline score when key evidence is unavailable.

    ``statistical`` absent (no empirical form frequencies) AND movement
    UNAVAILABLE (no opening price series) -> cap to MEDIUM floor; one
    missing -> cap to MEDIUM upper. Caps are opt-in via config keys
    ``score_cap_both_unavailable`` / ``score_cap_one_unavailable``; the
    defaults live in ``EVIDENCE_FLOOR_DEFAULTS``. Never lowers the score
    (a legitimately high score with all evidence stays high).

    Phase 6.1 FIX: when market component > 0 (market price exists and
    supports the pick), do NOT apply the floor — market evidence counts
    as valid evidence even when statistical/movement are unavailable.
    """
    cfg = cfg or {}
    # If market component exists (price available), evidence floor does not
    # apply — market data IS evidence.
    if components.get("market", 0.0) > 0.0:
        return score
    has_stat = "statistical" in components
    has_mv = _movement_available(components.get("_movement_block"))
    missing = (not has_stat) + (not has_mv)
    if missing == 2:
        cap = float(cfg.get("score_cap_both_unavailable", EVIDENCE_FLOOR_DEFAULTS["score_cap_both_unavailable"]))
    elif missing == 1:
        cap = float(cfg.get("score_cap_one_unavailable", EVIDENCE_FLOOR_DEFAULTS["score_cap_one_unavailable"]))
    else:
        return score
    return min(score, cap)


def _late_component(mv: dict[str, Any]) -> float:
    """Group D (late movement / consistency) from the multi-snapshot layer."""
    d = mv.get("late_direction")
    s = mv.get("late_strength")
    if d is None or s is None:
        return 0.5
    if d > 0:
        return 0.5 + 0.5 * s
    if d < 0:
        return 0.5 - 0.5 * s
    return 0.5


def _context_component(context: dict[str, Any], side: str | None) -> float:
    """Group E: team context (injuries/suspensions). Neutral 0.5 default."""
    if not context:
        return 0.5
    # Count missing players on the side(s) the signal depends on; a missing
    # key player weakens a directional signal on that side.
    sides = ("home", "away") if side is None else (side,)
    penalty = 0.0
    for s in sides:
        entry = context.get(s) or {}
        miss = entry.get("missing") or []
        penalty += min(0.3, 0.1 * len(miss))
    return max(0.0, 0.5 - penalty)


def _data_quality_component(completeness: float, n_groups: int) -> float:
    return max(0.0, min(1.0, completeness))


def _confluence(components: dict[str, float], min_agree: float = 0.55) -> int:
    return sum(1 for v in components.values() if v >= min_agree)


def confidence_label(
    score: float,
    confluence: int,
    edge_pp: float,
    min_confluence: int,
    conflict_pp: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """S23 confidence category from score + agreement + conflict."""
    t = thresholds or {}
    very_high = float(t.get("very_high_score", VERY_HIGH_SCORE))
    high = float(t.get("high_score", HIGH_SCORE))
    medium = float(t.get("medium_score", MEDIUM_SCORE))
    low = float(t.get("low_score", LOW_SCORE))
    if confluence < min_confluence or edge_pp <= -conflict_pp:
        return "LOW"
    if score >= very_high:
        return "VERY HIGH"
    if score >= high:
        return "HIGH"
    if score >= medium:
        return "MEDIUM"
    if score >= low:
        return "LOW"
    return "NO SIGNAL"


# --------------------------------------------------------------------------
# Signal construction
# --------------------------------------------------------------------------

def build_signals(
    *,
    model_probs: dict[str, Any],
    stats: dict[str, Any],
    market_totals: dict[str, dict[str, float]],
    ah_rows: list[dict[str, Any]],
    movement_snapshot: dict[str, Any] | None,
    context: dict[str, Any] | None,
    completeness: float,
    history_snapshots: list[dict[str, Any]] | None = None,
    opening_snapshot: dict[str, Any] | None = None,
    odds_1x2: dict[str, Any] | None = None,
) -> list[Signal]:
    """All candidate signals with model/market/movement features populated.
    """
    matrix = _build_matrix(model_probs)
    signals: list[Signal] = []

    def _late(side_hint: str | None) -> tuple[float | None, float | None]:
        if not movement_snapshot or not movement_snapshot.get("usable"):
            return None, None
        steam = movement_snapshot.get("steam_side")
        if not steam:
            return 0.0, 0.0
        strength = min(1.0, abs(movement_snapshot.get("steam_pct") or 0.0) / MOVEMENT_PCT_THRESHOLD)
        if side_hint and steam == side_hint:
            return 1.0, strength
        if side_hint and steam != side_hint:
            return -1.0, strength
        return 0.0, 0.0

    # ---- BTTS -----------------------------------------------------------
    p_yes = model_probs.get("btts_yes")
    if isinstance(p_yes, (int, float)) and 0 < p_yes < 1:
        o_yes = market_totals.get("BTTS Yes", {}).get("odds", 0.0)
        o_no = market_totals.get("BTTS No", {}).get("odds", 0.0)
        fair = fair_pair_implied(o_yes, o_no)
        opening_yes = market_totals.get("BTTS Yes", {}).get("opening")
        opening_no = market_totals.get("BTTS No", {}).get("opening")
        for sel, p, odds, imp, op in (
            ("BTTS Yes", p_yes, o_yes, fair[0] if fair else None, opening_yes),
            ("BTTS No", 1.0 - p_yes, o_no, fair[1] if fair else None, opening_no),
        ):
            edge = (p - imp) * 100.0 if imp is not None else 0.0
            late_d, late_s = _late(None)
            signals.append(Signal(
                market="BTTS", selection=sel, model_prob=p,
                market_odds=odds or None, implied_prob=imp,
                edge_pp=round(edge, 2),
                movement=movement_features(opening=op, current=odds or None,
                                           late_direction=late_d, late_strength=late_s),
            ))

    # ---- Over/Under 2.5 ------------------------------------------------
    p_over = model_probs.get("over_2.5")
    if isinstance(p_over, (int, float)) and 0 < p_over < 1:
        o_over = market_totals.get("Over 2.5", {}).get("odds", 0.0)
        o_under = market_totals.get("Under 2.5", {}).get("odds", 0.0)
        fair = fair_pair_implied(o_over, o_under)
        # Layer 1: the ONLY opening reference is the immutable canonical
        # opening_snapshot (pinned on first ingestion by the odds poll); the
        # per-source ``opening_price`` fields are never re-derived here. When
        # no canonical snapshot exists yet, opening is unknown (movement
        # degrades to UNAVAILABLE) -- honest, never fabricated.
        _os_ou = (opening_snapshot or {}).get("odds_ou") or {}
        _os_ou_matches = (
            _os_ou.get("line") is not None
            and abs(float(_os_ou["line"]) - 2.5) < 1e-9
        )
        open_over = _os_ou.get("over") if _os_ou_matches else None
        open_under = _os_ou.get("under") if _os_ou_matches else None
        # Line movement (opening goal line -> current goal line) is separate
        # from price movement (S10). NowGoal exposes the opening line via
        # ``opening_point``; The Odds API / OddsPapi do not -> stays None.
        cur_line = market_totals.get("Over 2.5", {}).get("point")
        open_line = market_totals.get("Over 2.5", {}).get("opening_point")
        for sel, p, odds, imp, op in (
            ("Over 2.5", p_over, o_over, fair[0] if fair else None, open_over),
            ("Under 2.5", 1.0 - p_over, o_under, fair[1] if fair else None, open_under),
        ):
            edge = (p - imp) * 100.0 if imp is not None else 0.0
            late_d, late_s = _late(None)
            signals.append(Signal(
                market="Total", selection=sel, model_prob=p,
                market_odds=odds or None, implied_prob=imp,
                edge_pp=round(edge, 2),
                movement=movement_features(
                    opening=op, current=odds or None,
                    opening_line=open_line, current_line=cur_line,
                    late_direction=late_d, late_strength=late_s,
                ),
            ))

    # ---- Over 3.5 (Fix C: tambah kandidat saat lambda tinggi + odds tersedia) ---
    _lh = float(model_probs.get("lambda_home", 0) or 0)
    _la = float(model_probs.get("lambda_away", 0) or 0)
    _lt = _lh + _la
    over_35_data = market_totals.get("Over 3.5") or {}
    over_35_odds = over_35_data.get("odds")
    if _lt > 2.8 and over_35_odds and float(over_35_odds) > 1.0:
        # Hitung P(Over 3.5) dari Poisson matrix
        from .models import poisson_matrix, probs_from_matrix
        _matrix = poisson_matrix(_lh, _la, rho=0.0)
        _p1x2, _o15, _o25, _o35, _btts = probs_from_matrix(_matrix)
        p_over_35 = _o35  # P(Over 3.5)
        if isinstance(p_over_35, (int, float)) and 0 < p_over_35 < 1:
            over_35_imp = 1.0 / float(over_35_odds)
            over_35_edge = (p_over_35 - over_35_imp) * 100.0
            over_35_open = over_35_data.get("opening")
            late_d35, late_s35 = _late(None)
            signals.append(Signal(
                market="Total", selection="Over 3.5",
                model_prob=p_over_35,
                market_odds=float(over_35_odds),
                implied_prob=over_35_imp,
                edge_pp=round(over_35_edge, 2),
                movement=movement_features(
                    opening=float(over_35_open) if over_35_open else None,
                    current=float(over_35_odds),
                    opening_line=3.5, current_line=3.5,
                    late_direction=late_d35, late_strength=late_s35,
                ),
            ))

    # ---- Asian Handicap ------------------------------------------------
    ah = ah_consensus(ah_rows)
    ah_lines: list[float] = []
    if ah is not None:
        ah_lines.append(ah["line"])
    for l in AH_LINES:
        if all(abs(l - x) > 1e-9 for x in ah_lines):
            ah_lines.append(l)

    _os_ah = (opening_snapshot or {}).get("odds_ah") or {}

    def _os_ah_matches(l: float) -> bool:
        return _os_ah.get("line") is not None and abs(float(_os_ah["line"]) - l) < 1e-9

    for line in ah_lines:
        # Market price only when this line is the actual consensus line; a
        # canonical line with no quote has no price (honest degradation).
        has_price = ah is not None and abs(ah["line"] - line) < 1e-9
        home_odds = ah["home"] if has_price else None
        away_odds = ah["away"] if has_price else None
        fair = fair_pair_implied(home_odds or 0.0, away_odds or 0.0) if has_price else None
        # SIDE-NEUTRAL (Layer 2): a handicap line is ONE bet, scored exactly
        # once. Home +1.25 and Away -1.25 are mirror labels of the same line;
        # scoring both sides made the identical bet score differently per
        # query (50/100 as Away -1.25 vs 76/100 as Home +1.25 on the same
        # fixture). The canonical side is the one the model favors
        # (prob >= 0.5); on an exact pick'em the side with the lower implied
        # vig (shorter odds) wins, still tied -> HOME (deterministic, never
        # random or request-order-dependent -- see _canonical_ah_side).
        #
        # Asymmetry audit: the statistical component below uses the CANONICAL
        # side's own recent scorelines (home vs away differ). Because the
        # canonical side is itself deterministic (model-driven), the line's
        # score is stable across queries and the mirror side is never scored
        # separately -- no silent averaging of asymmetric inputs.
        if matrix is None:
            p_home = None
        else:
            p_home = ah_win_prob(matrix, line, "home")
        if p_home is None or p_home <= 0:
            continue
        # Model 1X2 direction for canonical side determination
        _p1x2_for_side = model_probs.get("1x2") or {}
        _p_home_1x2 = float(_p1x2_for_side.get("home", 0.33))
        side = _canonical_ah_side(_p_home_1x2, p_home, home_odds, away_odds)
        p = p_home if side == "home" else 1.0 - p_home
        odds = home_odds if side == "home" else away_odds
        imp = (fair[0] if fair else None) if side == "home" else (fair[1] if fair else None)
        edge = (p - imp) * 100.0 if imp is not None else 0.0
        label = f"{side.title()} {_side_line(line, side):+g}"
        late_d, late_s = _late(side)
        signals.append(Signal(
            market="Asian Handicap", selection=label, model_prob=p,
            market_odds=odds, implied_prob=imp,
            line=line, side=side, line_key=f"ah:{line:+.2f}",
            edge_pp=round(edge, 2),
            movement=movement_features(
                # Layer 1: opening from the immutable canonical snapshot only.
                opening=(_os_ah.get("home") if side == "home" else _os_ah.get("away"))
                        if _os_ah_matches(line) else None,
                current=odds,
                opening_line=(float(_os_ah["line"]) if _os_ah_matches(line) else None),
                current_line=line,
                late_direction=late_d, late_strength=late_s,
            ),
        ))

    # ---- 1X2 (Plan v3 F14, 2026-08-24) ----------------------------------
    # Match-winner candidates. Sebelumnya signal engine TIDAK punya kandidat
    # 1X2 sama sekali sehingga "Away Win"/"Home Win" mustahil menjadi BEST
    # PICK -- kartu Elche v Barcelona hanya bisa memilih BTTS/Totals/AH walau
    # arah pertandingannya jelas. Probabilitas memakai ensemble 1X2; implied
    # probability margin-free atas TIGA outcome. Keluarga ini hanya dibuat
    # saat ada harga 1X2 (``odds_1x2``) -- tanpa harga, G7 akan memveto semua
    # dan barisnya hanya menjadi noise ranking.
    if odds_1x2:
        _p1x2m = model_probs.get("1x2") or {}
        _ox = {
            k: float(v) for k, v in odds_1x2.items()
            if isinstance(v, (int, float)) and v > 1.0
        }
        _imp3 = (
            normalize_implied({k: _ox[k] for k in ("home", "draw", "away")})
            if len(_ox) >= 3 else None
        )
        for _side, _label in (("home", "Home Win"), ("draw", "Draw"), ("away", "Away Win")):
            _p = _p1x2m.get(_side)
            if not isinstance(_p, (int, float)) or not 0 < float(_p) < 1:
                continue
            _o = _ox.get(_side)
            _imp = (_imp3 or {}).get(_side)
            _edge = (float(_p) - _imp) * 100.0 if _imp is not None else 0.0
            late_d12, late_s12 = _late(None)
            signals.append(Signal(
                market="1X2", selection=_label, model_prob=float(_p),
                market_odds=_o, implied_prob=_imp,
                edge_pp=round(_edge, 2),
                movement=movement_features(opening=None, current=_o,
                                           late_direction=late_d12, late_strength=late_s12),
            ))

    # ---- statistical support per signal --------------------------------
    for s in signals:
        if s.market == "1X2":
            continue  # no empirical form-frequency support defined yet
        if s.market == "BTTS":
            freq = statistical_support("btts", stats, 0.0, "home")
            stat = excess_probability(freq, 2) if freq is not None else None
        elif s.market == "Total":
            freq = statistical_support("over", stats, 0.0, "home")
            if freq is not None:
                freq = freq if s.selection.startswith("Over") else 1.0 - freq
            stat = excess_probability(freq, 2) if freq is not None else None
        else:
            freq = statistical_support("ah", stats, s.line or 0.0, s.side or "home")
            stat = excess_probability(freq, 2) if freq is not None else None
        if stat is not None:
            s.components["statistical"] = round(stat, 3)

    # ---- historical AH/O-U movement (odds_snapshot series) --------------
    # When the background odds-poll has accumulated snapshots, replace the
    # single opening->latest movement with the richer multi-snapshot series:
    # price movement, line movement, consistency, reversal, late movement.
    if history_snapshots:
        for s in signals:
            if s.market == "Asian Handicap":
                prices, lines = _ah_history_points(history_snapshots, s.side or "home", s.line or 0.0)
            elif s.market == "Total":
                sel = "over" if s.selection.startswith("Over") else "under"
                prices, lines = _ou_history_points(history_snapshots, sel, 2.5)
            else:
                continue
            hm = history_movement(prices, lines)
            if hm["status"] == "available":
                s.movement = hm
    return signals


def score_signals(
    signals: list[Signal],
    *,
    weights: dict[str, float],
    min_edge_pp: float,
    conflict_pp: float,
    completeness: float,
    context: dict[str, Any] | None,
    evidence_floor_cfg: dict[str, Any] | None = None,
    league_calibrated: bool = True,
    market_agreement_band_pp: float | None = None,
) -> None:
    """Fill per-signal component scores + weighted total (S17).

    ``evidence_floor_cfg`` (P1-3) optionally configures score caps when key
    evidence groups (statistical / movement) are unavailable; default caps
    come from ``EVIDENCE_FLOOR_DEFAULTS``.

    Phase 5.4: when ``league_calibrated`` is False, scores are capped at
    UNCALIBRATED_SCORE_MAX (0.50 = 50/100) to prevent misleading high
    confidence for unvalidated leagues.
    """
    for s in signals:
        comps: dict[str, float] = {}
        # Group A: existing model support = model win-equivalent probability.
        comps["model"] = round(max(0.0, min(1.0, s.model_prob)), 3)
        # Group C: market confirmation (edge). Absent price -> partial credit
        # based on model confidence (Fix: AH without odds was scoring 0.0,
        # causing high-confidence AH picks to lose to mediocre O/U picks).
        if s.implied_prob is not None:
            comps["market"] = round(_market_component(
                s.edge_pp, min_edge_pp, conflict_pp,
                agreement_band_pp=market_agreement_band_pp,
            ), 3)
        else:
            # No market price: give minimal credit based on model probability.
            # Conservative: AH without odds should NOT beat picks WITH odds.
            # Floor 0.20 max ensures Over/BTTS with real odds can compete.
            mp = max(0.0, min(1.0, s.model_prob or 0.0))
            if mp >= 0.70:
                comps["market"] = 0.20  # strong model, but no market validation
            elif mp >= 0.55:
                comps["market"] = 0.12  # moderate, mostly penalized
            else:
                comps["market"] = 0.05  # weak, heavily penalized
        # Group D: opening->current movement (neutral 0.5 if unavailable).
        comps["movement"] = round(_movement_component(s.movement, MOVEMENT_PCT_THRESHOLD), 3)
        # Group D late movement (neutral 0.5 if unavailable).
        comps["late_movement"] = round(_late_component(s.movement), 3)
        # Group F: data quality.
        comps["data_quality"] = round(_data_quality_component(completeness, 0), 3)
        # Group E: team context (optional).
        if weights.get("team_context", 0.0) > 0:
            comps["team_context"] = round(_context_component(context, s.side), 3)
        # Group B: statistical — empirical form/H2H frequencies.
        if "statistical" in s.components:
            comps["statistical"] = round(s.components["statistical"], 3)
        # Group G: market intelligence (steam/RLM/multi-book agreement).
        _mi = s.components.get("_market_intelligence")
        if _mi and _mi.get("usable"):
            comps["market_intelligence"] = round(_mi.get("confidence", 0.0), 3)
        # P1-3 internal key: keep the raw movement block accessible to
        # _apply_evidence_floor without polluting the displayed components.
        comps["_movement_block"] = s.movement or {}

        # Internal underscore-prefixed keys (e.g. ``_movement_block``, a dict
        # carrying the raw movement block for ``_apply_evidence_floor``) are
        # metadata, NOT score components -- they must never enter the weighted
        # sum (a dict value would raise on ``weight * value``).
        active = sum(weights.get(k, 0.0) for k in comps if not k.startswith("_"))
        total = sum(weights.get(k, 0.0) * comps[k] for k in comps if not k.startswith("_"))
        score = (total / active) if active > 0 else 0.0
        score = _apply_evidence_floor(score, comps, evidence_floor_cfg)
        # Phase 5.4: cap score for uncalibrated leagues to prevent
        # misleading high confidence for leagues without validated fit.
        # Safety preference: nudge picks toward safer markets when the
        # model has no validated per-league calibration.
        #
        # Hierarchy:  AH (draw cover)  ≈  Under 2.5 (low-scoring default)
        #             >  BTTS  >  Over 2.5 (high variance, no cover)
        #
        # Under 2.5 gets a flat floor because most matches end under
        # (~55%). Over 2.5 penalty is edge-dependent: weak edge → heavy
        # penalty, strong edge → no penalty (let the data speak).
        if not league_calibrated:
            score = min(score, UNCALIBRATED_SCORE_MAX)
            sel = (s.selection or "").lower()
            # Extract lambda_total for conditional Under floor
            _lh = context.get("lambda_home", 0) if context else 0
            _la = context.get("lambda_away", 0) if context else 0
            _lt = float(_lh or 0) + float(_la or 0)
            if s.market == "Asian Handicap":
                score = min(score * 1.05, UNCALIBRATED_SCORE_MAX)  # +5% boost
            elif s.market == "Total":
                if sel.startswith("under"):
                    # Fix A: Under floor HANYA aktif saat lambda rendah.
                    # Kalau model ekspektasi banyak gol (lambda > 2.5),
                    # Under bukan pick yang tepat meskipun punya edge.
                    if _lt < 2.5:
                        score = max(score, 0.48)  # floor 48/100 (safe default)
                    # else: lambda tinggi → jangan force Under, biarkan Over bersaing
                else:
                    # Over: edge-dependent penalty
                    # < 5pp edge → -10% (weak, prefer safer)
                    # 5-10pp    → -5%  (moderate)
                    # > 10pp    → 0%   (strong edge, let it compete)
                    if s.edge_pp < 5.0:
                        score *= 0.90
                    elif s.edge_pp < 10.0:
                        score *= 0.95
                    # else: no penalty for strong edge
            elif s.market == "BTTS":
                score *= 0.95  # -5% (binary, no draw cover)
        # Strip the internal key before exposing components to callers.
        comps.pop("_movement_block", None)
        s.components = comps
        s.score = round(score, 3)



# K3 (post-mortem 2026-08-28): second-leg context as SOFT penalties.
# Decided tie (aggregate margin >= 2): the leading side rotates / sits back,
# so picks that need the LEADER to win 90' are discounted (27 Aug: favourite
# failed to win 90' in 4 of 8 decided ties). Balanced tie (margin <= 1):
# cagey 90 minutes, so Over / BTTS Yes are discounted (Jablonec 1-0, Inter
# Escaldes 0-0, Rapid 1-1, Maccabi 1-1, Hapoel 0-1). Multipliers are mild on
# purpose -- measured via ``failure_class`` before any of this becomes a veto.
TIE_STATE_PENALTIES: dict[str, float] = {
    "decided_leader_directional": 0.85,
    "balanced_high_scoring": 0.92,
}


def _apply_tie_state_adjustments(
    signals: list[Signal],
    tie_state: dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
) -> None:
    if not signals or not tie_state:
        return
    pen = dict(TIE_STATE_PENALTIES)
    pen.update({k: float(v) for k, v in (cfg or {}).items() if v is not None})
    note = tie_state_note(tie_state)
    state = tie_state.get("state")
    leader = tie_state.get("leader")
    for s in signals:
        factor = 1.0
        if state == "decided" and leader and is_directional_selection(s.market, s.selection):
            side = s.side or ("home" if str(s.selection).startswith("Home") else "away")
            if side == leader:
                factor = pen["decided_leader_directional"]
        elif state == "balanced" and (
            (s.market == "Total" and str(s.selection).startswith("Over"))
            or (s.market == "BTTS" and str(s.selection).endswith("Yes"))
        ):
            factor = pen["balanced_high_scoring"]
        if factor != 1.0:
            s.score = round(s.score * factor, 3)
            s.internal_notes.append(f"{note} (score x{factor:.2f})")
        elif note and note not in s.internal_notes:
            s.internal_notes.append(note)


def _apply_post_scoring_adjustments(
    signals: list[Signal],
    model_probs: dict[str, Any],
    league_calibrated: bool = True,
) -> None:
    """Post-scoring adjustments based on match-level properties.

    Fix 3 (Direction): penalize AH picks where AH side contradicts model 1X2.
    Fix 4 (Decisive): penalize AH quarter-lines when match is decisive.
    Fix 5 (High-scoring): boost Over when lambda total is high.
    Fix 9 (Contrarian): penalize contrarian picks in uncalibrated leagues.
    """
    if not signals:
        return

    # Extract model 1X2 probabilities
    prob_1x2 = model_probs.get("1x2", {})
    p_home = float(prob_1x2.get("home", 0.33))
    p_away = float(prob_1x2.get("away", 0.33))
    p_draw = float(prob_1x2.get("draw", 0.33))
    model_direction = "home" if p_home > p_away else ("away" if p_away > p_home else "draw")

    # Extract lambda total for high-scoring detection
    lambda_home = float(model_probs.get("lambda_home", 0) or 0)
    lambda_away = float(model_probs.get("lambda_away", 0) or 0)
    lambda_total = lambda_home + lambda_away

    # Detect decisive match: one team clearly favored
    max_prob = max(p_home, p_away)
    is_decisive = max_prob > 0.60  # one team > 60% favored

    # Fix 1: Unseeded team detection — penalize AH when teams lack seeded Elo.
    # model_probs['elo_seeded'] is True only when BOTH teams exist in the
    # seeded ratings; otherwise lambda is based on a 1500 prior and is unreliable.
    _either_unseeded = model_probs.get("elo_seeded") is False

    # Market-agnostic adjustments: apply to ALL market types equally.
    # The bot should pick the RIGHT market for each match, not default to AH.
    for s in signals:
        # Fix 2b: ANY pick without market odds — edge not market-validated.
        # Reduce edge by 30% when no market price exists.
        no_odds = s.market_odds is None or (isinstance(s.market_odds, (int, float)) and s.market_odds <= 1.0)
        if no_odds:
            s.edge_pp = round(s.edge_pp * 0.70, 2)

        # Fix 1: Unseeded team penalty for AH.
        # When either team has default Elo (1500), AH predictions are unreliable.
        # Penalize AH picks more heavily to prevent wrong方向 picks.
        if _either_unseeded and s.market == "Asian Handicap":
            s.score = round(s.score * 0.85, 3)  # -15% penalty for AH with unseeded teams

        # Fix 3: Direction alignment for directional picks.
        # AH: penalize if AH side contradicts model 1X2.
        # 1X2: no adjustment needed (model IS the direction).
        if s.market == "Asian Handicap" and s.side and model_direction != "draw":
            if s.side != model_direction and max_prob > 0.60:
                # Stronger penalty when model is very confident (>75%)
                if max_prob > 0.75:
                    s.score = round(s.score * 0.85, 3)  # -15% strong penalty
                else:
                    s.score = round(s.score * 0.92, 3)  # -8% mild penalty

        # Fix 4: Decisive match → penalize half-win markets.
        # In decisive matches, AH quarter-lines and Under give weaker returns.
        # Prefer Over or 1X2 for decisive outcomes.
        if is_decisive:
            if s.market == "Asian Handicap" and s.line is not None:
                is_quarter = abs(s.line % 0.5) > 1e-9 and abs(s.line % 0.5 - 0.5) > 1e-9
                is_half = abs(s.line % 0.5) < 1e-9 and abs(s.line) % 1.0 > 1e-9
                if is_quarter or is_half:
                    s.score = round(s.score * 0.92, 3)  # -8% for quarter/half lines
            elif s.market == "Total" and s.selection and s.selection.startswith("Under"):
                s.score = round(s.score * 0.95, 3)  # -5% Under in decisive match

        # Fix 5: High-scoring detection → boost Over, penalize Under.
        # When lambda total > 2.5, model expects many goals.
        if lambda_total > 2.5 and s.market == "Total" and s.selection:
            if s.selection.startswith("Over"):
                if lambda_total > 3.0:
                    s.score = round(s.score * 1.08, 3)  # +8% boost for high-scoring
                else:
                    s.score = round(s.score * 1.04, 3)  # +4% boost for moderate
            elif s.selection.startswith("Under"):
                s.score = round(s.score * 0.94, 3)  # -6% Under when many goals expected

        # Fix 6: Low-scoring detection → boost Under, penalize Over.
        # When lambda total < 2.0, model expects few goals.
        if lambda_total < 2.0 and s.market == "Total" and s.selection:
            if s.selection.startswith("Under"):
                s.score = round(s.score * 1.05, 3)  # +5% boost for low-scoring
            elif s.selection.startswith("Over"):
                s.score = round(s.score * 0.92, 3)  # -8% penalty for Over in low-scoring

        # Fix 7: BTTS in high-scoring → boost BTTS Yes.
        if lambda_total > 2.5 and s.market == "BTTS" and s.selection:
            if "Yes" in s.selection:
                s.score = round(s.score * 1.03, 3)  # +3% boost

        # Fix 8: BTTS in low-scoring → boost BTTS No.
        if lambda_total < 2.0 and s.market == "BTTS" and s.selection:
            if "No" in s.selection:
                s.score = round(s.score * 1.03, 3)  # +3% boost

        # Fix 9: Weak model confidence penalty.
        # When model_prob < 50% but edge is positive, the pick is based on
        # market disagreement (market overpriced the opposite side), NOT on
        # model confidence. Penalize to prevent low-confidence contrarian
        # picks from dominating.
        # Scale: the lower the model_prob, the heavier the penalty.
        if s.model_prob < 0.50 and s.edge_pp > 0:
            gap = 0.50 - s.model_prob  # how far below 50%
            penalty = 0.80 + gap * 0.4  # e.g. mp=0.42 → penalty=0.832 (-17%)
            if not league_calibrated:
                penalty *= 0.92  # extra -8% for uncalibrated
            s.score = round(s.score * penalty, 3)

        # Fix 10: Market variety bonus for AH picks.
        # AH (Asian Handicap) typically has tighter lines and better edge
        # than Over/Under. Give a small bonus to encourage market diversity
        # and prevent Over/Under dominance. Bonus is small (+3%) and only
        # applies when edge is positive -- AH still must meet edge threshold.
        if s.market == "Asian Handicap" and s.edge_pp > 0:
            s.score = round(s.score * 1.03, 3)  # +3% AH variety bonus


def rank_and_pick(
    signals: list[Signal],
    *,
    best_pick_margin: float,
    no_bet_score: float,
    min_confluence: int,
    conflict_pp: float,
    min_data_quality: float,
    completeness: float,
    min_edge_pp: float = 0.0,
    confidence_thresholds: dict[str, float] | None = None,
    odds_disagreement: bool = False,
    evidence_floor: dict[str, Any] | None = None,
    model_decision_type: str | None = None,
) -> dict[str, Any]:
    """Rank signals deterministically and select the Best Pick (or NO BET).

    Selection is based on the ABSOLUTE quality of the top signal, not on the
    score gap to the runner-up: a strong top signal is picked even when the
    second signal is close (the ``best_pick_margin`` gate is gone). NO BET is
    returned only when the top signal itself is genuinely weak (score below
    the actionable floor, too little evidence confluence, a model-market
    conflict, or insufficient data quality).

    ``odds_disagreement`` (P3): when independent odds sources disagree on
    key lines beyond tolerance, the confidence of the pick is capped at
    MEDIUM and an explicit reason is added -- disagreement is visible, never
    a silent confidence inflation. The caller already docks ``completeness``
    before ranking; this caps the label and adds the reason.
    """
    ranked = sorted(
        signals,
        key=lambda s: (s.score, s.model_prob, s.edge_pp),
        reverse=True,
    )
    # P3-4: coverage floor -- a label is only as strong as the evidence
    # behind it. Thin field coverage (lineups/injuries/standings missing)
    # downgrades HIGH/VERY HIGH to MEDIUM and, below ``low_below``, to LOW.
    _cov_cfg = (confidence_thresholds or {}).get("coverage_floor") or {}
    for s in ranked:
        s.confidence = confidence_label(
            s.score, _confluence(s.components), s.edge_pp,
            min_confluence, conflict_pp, confidence_thresholds,
        )
        s.confidence = _coverage_floor(completeness, s.confidence, _cov_cfg)
        s.confidence = _cap_confidence(s.confidence, getattr(s, "confidence_cap", None))
        # 2026-08-22: a gated candidate keeps its SCORE (so the card can show
        # what the evidence was worth) but must never advertise a confidence --
        # "Score: 62/100 / Confidence: HIGH" next to a rejected pick is exactly
        # the misleading presentation the pick_gates exist to remove.
        if s.vetoed:
            s.confidence = "NO SIGNAL"
    # P3: disagreement caps the top pick's confidence at MEDIUM.
    if odds_disagreement:
        for s in ranked:
            if s.confidence in ("VERY HIGH", "HIGH"):
                s.confidence = "MEDIUM"
    # Double-count fix: the late move is a PENALTY, not a score component.
    # A market that moved AGAINST the selection with meaningful strength
    # into the close caps confidence at MEDIUM -- the pick must never be
    # presented stronger than the market's own late verdict.
    for s in ranked:
        _mv = s.movement or {}
        _ld = _mv.get("late_direction")
        _ls = _mv.get("late_strength")
        if _ld is not None and _ld < 0 and (_ls or 0.0) >= LATE_AGAINST_MIN_STRENGTH:
            if s.confidence in ("VERY HIGH", "HIGH"):
                s.confidence = LATE_AGAINST_CAP
                s.evidence_notes = list(getattr(s, "evidence_notes", None) or []) + [
                    "late market move melawan pick — confidence dibatasi MEDIUM"
                ]

    reasons: list[str] = []
    # 2026-08-22: eligibility is the ``vetoed`` flag, NOT a zeroed score. A
    # vetoed candidate keeps the score it earned so the card can show both the
    # number and why it was rejected; it simply cannot be selected.
    eligible = [s for s in ranked if not s.vetoed]
    best = eligible[0] if eligible else None
    decision = "NO BET"
    pick: Signal | None = None

    if best is None:
        if ranked:
            # Every candidate was gated. Report the ACTUAL gate reasons -- the
            # old code fell through to "best score 0.00 < 0.45", which told the
            # user nothing about why.
            _seen: set[str] = set()
            for s in ranked:
                for r in s.veto_reasons:
                    if r not in _seen:
                        _seen.add(r)
                        reasons.append(f"{s.selection}: {r}")
            if not reasons:
                reasons.append("semua kandidat diveto oleh pick_gates")
        else:
            reasons.append("no signal candidates")
    else:
        margin = best.score - (eligible[1].score if len(eligible) > 1 else 0.0)
        confl = _confluence(best.components)
        reasons.append(f"top score {best.score:.2f}, margin {margin:.2f}")
        if odds_disagreement:
            reasons.append("cross-source odds disagreement (confidence capped MEDIUM)")
        if best.score < no_bet_score:
            reasons.append(f"best score {best.score:.2f} < {no_bet_score:.2f}")
        elif confl < min_confluence:
            reasons.append(f"confluence {confl} < {min_confluence} (evidence too thin)")
        elif best.edge_pp <= -conflict_pp:
            reasons.append(
                f"{best.selection} model-market conflict (edge {best.edge_pp:+.1f}pp)"
            )
        elif best.edge_pp < float(
            (confidence_thresholds or {}).get("allow_negative_edge_pp", 0.0) or 0.0
        ):
            # F4: negative edge below the allowed floor = model disagrees with
            # market price more than ``allow_negative_edge_pp`` permits. The
            # floor (config models.signal_engine.allow_negative_edge_pp,
            # default 0.0 = strict) keeps prior behavior; a small negative
            # floor (-3.0) preserves the pre-2026-08-22 behavior of emitting
            # picks whose model sits marginally BELOW the price (verified
            # winners Marseille -0.5pp / Arsenal -2.5pp) while hard conflicts
            # remain blocked by the -conflict_pp branch above.
            reasons.append(
                f"edge negatif ({best.edge_pp:+.1f}pp) — model tidak lebih baik dari harga market"
            )
        elif best.edge_pp < (
            min_edge_pp
            if float((confidence_thresholds or {}).get("allow_negative_edge_pp", 0.0) or 0.0) >= 0.0
            else float((confidence_thresholds or {}).get("allow_negative_edge_pp", 0.0) or 0.0)
        ):
            # F5: hard edge threshold -- top signal edge is too small.
            # When ``allow_negative_edge_pp`` is negative it ALSO relaxes this
            # floor to that value (one knob, consistent semantics: the same
            # small-negative-edge picks F4 spares are eligible as BEST PICK),
            # restoring the pre-2026-08-22 emission of near-consensus picks;
            # positive (default) keeps the strict min_edge behavior. Try the
            # next signal that meets the edge threshold otherwise.
            _fallback = None
            for _s in eligible[1:]:
                _floor_eff = (
                    min_edge_pp
                    if float((confidence_thresholds or {}).get("allow_negative_edge_pp", 0.0) or 0.0) >= 0.0
                    else float((confidence_thresholds or {}).get("allow_negative_edge_pp", 0.0) or 0.0)
                )
                if _s.edge_pp >= _floor_eff and _s.score >= no_bet_score:
                    _fallback = _s
                    break
            if _fallback:
                # Found a weaker-scored signal with sufficient edge.
                reasons.append(
                    f"top signal edge terlalu kecil ({best.edge_pp:+.1f}pp < {min_edge_pp:.0f}pp), "
                    f"fallback ke {_fallback.selection} (edge {_fallback.edge_pp:+.1f}pp)"
                )
                best = _fallback
                decision = "BEST PICK"
                pick = best
            else:
                reasons.append(
                    f"edge terlalu kecil ({best.edge_pp:+.1f}pp < {min_edge_pp:.0f}pp) — tidak cukup kuat untuk BEST PICK"
                )
        elif completeness < min_data_quality:
            reasons.append(f"data quality {completeness:.2f} < {min_data_quality:.2f}")
        elif evidence_floor and evidence_floor.get("veto"):
            # F2 veto: prior-Elo λ + thin/no form + no H2H is NOT a bettable
            # signal. NO BET with the explicit reason (never a silent pick).
            reasons.append(f"{evidence_floor.get('note', 'evidence tipis')} — {best.selection}")
        else:
            decision = "BEST PICK"
            pick = best
            if evidence_floor and not evidence_floor.get("veto"):
                # F2 cap: same thin prior-based evidence, but H2H exists --
                # never present HIGH on a prior alone.
                best.confidence = "LOW"
                best.evidence_notes = [evidence_floor.get("note", "evidence tipis")]
                reasons.append(evidence_floor["note"])
            if model_decision_type in NON_ACTIONABLE_DECISIONS:
                # G1 (post-mortem 2026-08-22) -- was F3 "cap at MEDIUM".
                #
                # The independent 1X2 decision layer found no bet. Publishing a
                # BEST PICK anyway is a DISCIPLINE failure, not a modelling
                # one: on 2026-08-21 all 11 published picks carried
                # decision_type NO BET / NO CLEAR DECISION, and the day
                # returned -2.26u over 9 unique fixtures (-25.1% ROI).
                #
                # Capping confidence at MEDIUM was strictly worse than doing
                # nothing, for two reasons:
                #   1. every rejected pick landed on exactly MEDIUM, which
                #      reads as "bettable" on the card;
                #   2. MEDIUM satisfies the analyse-layer bypass
                #      ``_strong_pick = score >= 0.50 and confidence in
                #      {VERY HIGH, HIGH, MEDIUM}`` (analyse.py), which SKIPS
                #      the evidence gate -- so the cap opened the very gate it
                #      was meant to close.
                #
                # Now it is a veto. Set
                # models.signal_engine.pick_gates.respect_model_decision=false
                # to restore the old cap-only behaviour.
                _respect = bool(
                    ((confidence_thresholds or {}).get("pick_gates") or {})
                    .get("respect_model_decision", True)
                )
                # P2-3: the model-vs-model note lives on ``internal_notes``,
                # not ``evidence_notes`` -- the summary embed filters
                # ``internal_notes`` out so the user only sees it after
                # clicking "Lihat Hasil".
                _internal = list(getattr(best, "internal_notes", None) or [])
                _internal.append(
                    f"model 1X2: {model_decision_type} — pick tidak didukung layer model"
                )
                best.internal_notes = _internal
                if _respect:
                    decision = "NO BET"
                    pick = None
                    reasons.append(
                        f"model 1X2 {model_decision_type} — layer model tidak menemukan "
                        "bet, BEST PICK diveto (G1)"
                    )
                else:
                    # 2026-08-22 (operator decision): publish the pick with its
                    # REAL confidence -- the MEDIUM cap made every published
                    # pick read as "bettable" while hiding the model's actual
                    # conviction. The disagreement stays on internal_notes.
                    reasons.append(
                        f"model 1X2 {model_decision_type} — pick dipublikasikan "
                        "tanpa dukungan layer model (respect_model_decision=false)"
                    )

    return {
        "decision": decision,
        "best_pick": pick,
        "ranking": [
            {
                "market": s.market,
                "selection": s.selection,
                "score": s.score,
                "confidence": s.confidence,
                "model_prob": round(s.model_prob, 4),
                "market_odds": s.market_odds,
                "implied_prob": round(s.implied_prob, 4) if s.implied_prob is not None else None,
                "edge_pp": s.edge_pp,
                "movement": s.movement,
                "components": s.components,
                "line": s.line,
                "side": s.side,
                "line_key": s.line_key,
                "internal_notes": list(s.internal_notes or []),
                # 2026-08-22: surfaced so the card can render "score X, ditolak
                # karena <reason>" instead of a bare "Score: 0/100".
                "vetoed": bool(s.vetoed),
                "veto_reasons": list(s.veto_reasons or []),
            }
            for s in ranked
        ],
        "reasons": reasons,
    }


# --------------------------------------------------------------------------
# Layer 1: canonical opening reference (movement scoring + display MARKET
# block read the SAME immutable ``opening_snapshot``; nobody re-derives
# "opening" from per-source ``opening_price`` fields)
# --------------------------------------------------------------------------

def build_market_block(
    *,
    market_totals: dict[str, dict[str, float]],
    ah: dict[str, Any] | None,
    opening_snapshot: dict[str, Any] | None,
    history_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Opening -> latest per market for the display MARKET block.

    ``opening`` comes from the Layer-1 canonical snapshot when one exists
    (``canonical`` True); ``latest`` is the last polled snapshot price at the
    relevant line, else the current payload price. Consumers of this block
    never compute their own opening.

    P1-1 fallback: when no canonical snapshot exists yet (first-time
    ingestion, no odds-poll history), fall back to the per-source
    ``opening_price`` fields on the current payload. The fallback is
    flagged ``non_canonical: True`` so the card can render the
    "opening from current snapshot (not pinned)" warning -- the user
    sees a real number instead of "n/a", but knows it's not yet pinned
    to the immutable first-seen record.
    """
    mb: dict[str, Any] = {"ou": {"canonical": False}, "ah": {"canonical": False}}
    os = opening_snapshot or {}
    os_ou = os.get("odds_ou") or {}
    os_ah = os.get("odds_ah") or {}

    def _valid_price(v: Any) -> bool:
        try:
            return v is not None and float(v) > 1.0
        except (TypeError, ValueError):
            return False

    ou: dict[str, Any] = {"canonical": False}
    over = market_totals.get("Over 2.5") or {}
    under = market_totals.get("Under 2.5") or {}
    if os_ou.get("line") is not None and abs(float(os_ou["line"]) - 2.5) < 1e-9:
        ou.update({
            "canonical": True,
            "opening_over": os_ou.get("over"),
            "opening_under": os_ou.get("under"),
        })
    elif _valid_price(over.get("opening")) or _valid_price(under.get("opening")):
        # P1-1: non-canonical fallback when current payload carries per-
        # source opening but no canonical snapshot has been pinned yet.
        ou["non_canonical"] = True
        if _valid_price(over.get("opening")):
            ou["opening_over"] = over.get("opening")
        if _valid_price(under.get("opening")):
            ou["opening_under"] = under.get("opening")
    over_prices, _ = _ou_history_points(history_snapshots or [], "over", 2.5)
    under_prices, _ = _ou_history_points(history_snapshots or [], "under", 2.5)
    latest_over = over_prices[-1] if over_prices else over.get("odds")
    latest_under = under_prices[-1] if under_prices else under.get("odds")
    if latest_over:
        ou["latest_over"] = latest_over
    if latest_under:
        ou["latest_under"] = latest_under
    mb["ou"] = ou

    if ah and ah.get("line") is not None:
        cur_line = float(ah["line"])
        ahb: dict[str, Any] = {"canonical": False, "latest_line": cur_line}
        if os_ah.get("line") is not None:
            ahb.update({
                "canonical": True,
                "opening_line": float(os_ah["line"]),
            })
            # Price movement is only comparable when the line has NOT moved;
            # comparing prices across different lines is apples-to-oranges.
            # When the line moved, only line movement is reported (no price
            # direction claim, so narrative can never contradict display).
            if abs(float(os_ah["line"]) - cur_line) < 1e-9:
                ahb["home_open"] = os_ah.get("home")
                ahb["away_open"] = os_ah.get("away")
        elif _valid_price(ah.get("home_open")) or _valid_price(ah.get("away_open")):
            # P1-1: non-canonical fallback for AH from consensus rows.
            ahb["non_canonical"] = True
            ahb["home_open"] = ah.get("home_open")
            ahb["away_open"] = ah.get("away_open")
            if ah.get("line_open") is not None:
                ahb["opening_line"] = float(ah["line_open"])
        home_prices, _ = _ah_history_points(history_snapshots or [], "home", cur_line)
        away_prices, _ = _ah_history_points(history_snapshots or [], "away", cur_line)
        ahb["home_latest"] = home_prices[-1] if home_prices else ah.get("home")
        ahb["away_latest"] = away_prices[-1] if away_prices else ah.get("away")
        mb["ah"] = ahb
    return mb


# --------------------------------------------------------------------------
# Layer 3: repeated-query stability guard (pure)
# --------------------------------------------------------------------------

def _iso_age_seconds(now_iso: str | None, then_iso: str | None) -> float | None:
    """Seconds between two ISO-8601 UTC timestamps, or None when unparsable."""
    if not now_iso or not then_iso:
        return None
    try:
        def _parse(s: str) -> datetime:
            cleaned = s[:-1] + "+00:00" if s.endswith("Z") else s
            dt = datetime.fromisoformat(cleaned)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return (_parse(now_iso) - _parse(then_iso)).total_seconds()
    except (ValueError, TypeError):
        return None


def _market_move_direction(
    se_result: dict[str, Any],
    prev: dict[str, Any],
    opening_snapshot: dict[str, Any] | None,
    market_totals: dict[str, dict[str, float]],
) -> str:
    """Direction of the HELD pick's market move vs the canonical opening.

    Returns "supporting" when the price SHORTENED on the held pick's side
    (money into the pick -> reinforces it), else "none". Adverse moves are
    the caller's job via ``_adverse_market_move`` (thresholded + reasoned);
    this only detects reinforcement so a supporting re-price of the same
    pick is never labeled as a change.
    """
    os = opening_snapshot or {}
    market = prev.get("market")
    if market == "Total":
        os_ou = os.get("odds_ou") or {}
        if os_ou.get("line") is None or abs(float(os_ou["line"]) - 2.5) >= 1e-9:
            return "none"
        key = "under" if str(prev.get("selection", "")).startswith("Under") else "over"
        opening = os_ou.get(key)
        latest = (
            market_totals.get("Under 2.5" if key == "under" else "Over 2.5") or {}
        ).get("odds")
        if not opening or not latest or float(opening) <= 1.0 or float(latest) <= 1.0:
            return "none"
        return "supporting" if float(latest) < float(opening) else "none"
    if market == "Asian Handicap":
        os_ah = os.get("odds_ah") or {}
        if os_ah.get("line") is None:
            return "none"
        key = "home" if prev.get("side") == "home" else "away"
        opening = os_ah.get(key)
        latest = (se_result.get("ah_consensus") or {}).get(key)
        if not opening or not latest or float(opening) <= 1.0 or float(latest) <= 1.0:
            return "none"
        return "supporting" if float(latest) < float(opening) else "none"
    return "none"


def _adverse_market_move(
    se_result: dict[str, Any],
    prev: dict[str, Any],
    opening_snapshot: dict[str, Any] | None,
    market_totals: dict[str, dict[str, float]],
    st: dict[str, Any],
) -> str | None:
    """Adverse move of the HELD pick's market vs the canonical opening.

    Only a move AGAINST the held pick counts -- a move that supports the
    pick reinforces it and can never trigger a flip. Returns a human reason
    when the pick's price lengthened >= ``market_move_threshold_pct`` from
    the pinned opening, or the handicap/goal line moved
    >= ``line_move_threshold``.
    """
    pct_th = float(st.get("market_move_threshold_pct", 3.0))
    line_th = float(st.get("line_move_threshold", 0.25))
    os = opening_snapshot or {}
    market = prev.get("market")
    if market == "Total":
        os_ou = os.get("odds_ou") or {}
        if os_ou.get("line") is None or abs(float(os_ou["line"]) - 2.5) >= 1e-9:
            return None
        sel = str(prev.get("selection", ""))
        key = "under" if sel.startswith("Under") else "over"
        opening = os_ou.get(key)
        latest = (market_totals.get("Under 2.5" if key == "under" else "Over 2.5") or {}).get("odds")
        if not opening or not latest or float(opening) <= 1.0 or float(latest) <= 1.0:
            return None
        pct = (float(latest) - float(opening)) / float(opening) * 100.0
        if pct >= pct_th:
            return (
                f"market bergerak MELAWAN pick: harga {sel} memanjang "
                f"{pct:+.1f}% dari opening (≥{pct_th:.1f}%)"
            )
        return None
    if market == "Asian Handicap":
        os_ah = os.get("odds_ah") or {}
        if os_ah.get("line") is None:
            return None
        open_line = float(os_ah["line"])
        # Compare against the CURRENT consensus line, never the pick's line:
        # the pick can sit on a canonical quarter line (AH_LINES) with no
        # quote of its own, so comparing it to the opening line would report
        # a false adverse move even when the market never moved (e.g. pick
        # Away +0.25 vs an opening +1.25 line that is still +1.25).
        cur_line = (se_result.get("ah_consensus") or {}).get("line")
        if cur_line is None:
            return None
        if abs(open_line - float(cur_line)) >= line_th:
            return (
                f"garis handicap bergerak {open_line:+.2f} → {float(cur_line):+.2f} "
                f"(≥{line_th:.2f})"
            )
        side = prev.get("side")
        key = "home" if side == "home" else "away"
        opening = os_ah.get(key)
        latest = (se_result.get("ah_consensus") or {}).get(key)
        if not opening or not latest or float(opening) <= 1.0 or float(latest) <= 1.0:
            return None
        pct = (float(latest) - float(opening)) / float(opening) * 100.0
        if pct >= pct_th:
            return (
                f"market bergerak MELAWAN pick: harga {prev.get('selection')} "
                f"memanjang {pct:+.1f}% dari opening (≥{pct_th:.1f}%)"
            )
        return None
    return None


def apply_pick_stability(
    se_result: dict[str, Any],
    *,
    previous_pick: dict[str, Any] | None,
    current_model: dict[str, Any],
    opening_snapshot: dict[str, Any] | None,
    market_totals: dict[str, dict[str, float]],
    now_ts: str | None = None,
    cfg: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    """Layer 3: suppress a best-pick flip on repeated pre-match queries when
    nothing genuinely changed (pure; the caller passes the IMMUTABLE prior
    pick logged by prediction_log and the Layer-1 opening snapshot).

    The pick is HELD when, compared to the prior logged pick:
      1. the new top candidate's score moved by less than ``score_threshold``
         (calibrated from logged repeated-query noise when enough data
         exists, else the configured fallback), AND
      2. the market has not moved AGAINST the held pick beyond the configured
         price/line thresholds from the canonical opening, AND
      3. the prior pick is not stale (older than ``max_age_seconds``).
    Time is a cap, never the sole gate: a fast market move still triggers a
    change even within minutes (spec: no wall-clock-only gating).

    When held, the newly computed top candidate is recorded under
    ``stability.suppressed_top`` so the log keeps a full audit trail of what
    the model would have said at each query. When changed, the response MUST
    state the prior pick, the new pick and the specific reason (never a
    silent swap).
    """
    st = dict(STABILITY_DEFAULTS)
    st.update((cfg or {}).get("stability") or {})
    if not st["enabled"]:
        return se_result
    prev = previous_pick or {}
    prev_sel = prev.get("selection")
    if not prev_sel or prev.get("decision") != "BEST PICK":
        return se_result
    ranking = se_result.get("ranking") or []
    if not ranking:
        return se_result
    # 2026-08-22: compare against the SELECTED pick -- the top ELIGIBLE
    # candidate -- not ranking[0]. Since pick_gates stopped zeroing the score of
    # a vetoed candidate (so the card can show what it was worth), ranking[0]
    # may be a candidate that can never be picked. Comparing the logged pick's
    # score against that higher, vetoed score reported a phantom "changed" on
    # byte-identical input.
    _eligible = [r for r in ranking if not r.get("vetoed")]
    top = (_eligible or ranking)[0]
    prev_score = float(prev.get("score") or 0.0)

    def _changed(reason: str) -> dict[str, Any]:
        # When the new top is the SAME Layer-2 canonical line as the prior
        # pick (only the label flipped sides), say so explicitly -- it is a
        # relabel, not a different bet.
        relabel = ""
        if (
            prev.get("line_key")
            and top.get("line_key") == prev.get("line_key")
            and top["selection"] != prev_sel
        ):
            relabel = (
                f" — taruhan yang sama ({prev.get('line_key')}), "
                f"label berubah karena model bergeser"
            )
        se_result["stability"] = {
            "status": "changed",
            "previous_selection": prev_sel,
            "previous_score": round(prev_score, 3),
            "new_selection": top["selection"],
            "new_score": top["score"],
            "reason": reason + relabel,
        }
        return se_result

    # 3. staleness cap (time is a cap, never the sole gate).
    age = _iso_age_seconds(now_ts, prev.get("ts"))
    if age is not None and age > float(st["max_age_seconds"]):
        return _changed(
            f"analisis sebelumnya {int(age // 60)} menit lalu — evaluasi ulang"
        )

    # 2. adverse market move vs the canonical opening (supporting moves never
    # flip). Checked BEFORE the score delta: when the market moved against the
    # held pick, that is the most actionable reason and must be labeled as
    # such -- an adverse move is never presented as a bare score change, even
    # when the re-priced score also crossed the threshold.
    adverse = _adverse_market_move(
        se_result, prev, opening_snapshot, market_totals, st
    )
    if adverse:
        return _changed(adverse)

    # 1. score delta vs the prior logged pick's score. A supporting market
    # move (price shortened on the held pick's side) that merely re-prices
    # the SAME pick is NOT a genuine change -- fall through to HOLD so a
    # reinforcing move never triggers a noisy flip label.
    threshold = (
        score_threshold
        if score_threshold is not None
        else float(st["score_threshold_fallback"])
    )
    delta = abs(top["score"] - prev_score)
    same_bet = (
        top.get("market") == prev.get("market")
        and (
            top.get("selection") == prev_sel
            or (prev.get("line_key") and top.get("line_key") == prev.get("line_key"))
        )
    )
    supporting = (
        _market_move_direction(se_result, prev, opening_snapshot, market_totals)
        == "supporting"
    )
    if delta >= threshold and not (supporting and same_bet):
        return _changed(
            f"skor berubah {prev_score:.2f} → {top['selection']} "
            f"{top['score']:.2f} (Δ{delta:.2f} ≥ ambang {threshold:.2f})"
        )

    # HOLD: find the prior pick in the new ranking (by market+selection, or
    # by Layer-2 canonical line_key when the label merely flipped sides).
    entry = next(
        (r for r in ranking
         if r.get("market") == prev.get("market")
         and (r.get("selection") == prev_sel
              or (prev.get("line_key") and r.get("line_key") == prev.get("line_key")))),
        None,
    )
    if entry is None:
        return _changed(
            f"market untuk {prev_sel} tidak tersedia lagi pada query ini"
        )
    if entry["score"] < float(st["no_bet_hold"]) or entry.get("confidence") not in (
        "VERY HIGH", "HIGH", "MEDIUM"
    ):
        return _changed(
            f"pick sebelumnya melemah ({entry['selection']} skor "
            f"{entry['score']:.2f}, confidence {entry.get('confidence')})"
        )
    held = {
        "market": entry["market"],
        "selection": entry["selection"],
        "score": entry["score"],
        "confidence": entry["confidence"],
        "model_prob": entry["model_prob"],
        "market_odds": entry["market_odds"],
        "edge_pp": entry["edge_pp"],
        "line": entry.get("line"),
        "side": entry.get("side"),
        "components": entry.get("components") or {},
        "confluence": _confluence(entry.get("components") or {}),
        "movement": entry.get("movement") or {},
    }
    se_result["best_pick"] = held
    se_result["decision"] = "BEST PICK"
    se_result["stability"] = {
        "status": "held",
        "previous_selection": prev_sel,
        "previous_score": round(prev_score, 3),
        "held_selection": entry["selection"],
        "held_score": entry["score"],
        # Audit trail: what the model would have picked this query.
        "suppressed_top": {
            "selection": top["selection"],
            "score": top["score"],
            "confidence": top["confidence"],
        },
        "reason": (
            "model belum berubah signifikan dan pergerakan market masih dalam "
            "batas noise — pick dipertahankan dari query sebelumnya"
        ),
    }
    return se_result


# --------------------------------------------------------------------------
# Layer 4: narrative/confidence binding audit (pure)
# --------------------------------------------------------------------------

def movement_narrative_flags(se_result: dict[str, Any]) -> list[str]:
    """Cross-check the narrative's movement claim against the DISPLAYED
    pick-side movement (both derive from the same Layer-1 canonical opening,
    so they agree by construction; this is a safety net for future drift).

    The formatter must suppress the "confirms/opposes" bullet when any flag
    is set -- a narrative that contradicts the numbers shown in the same
    response is never emitted.
    """
    flags: list[str] = []
    bp = se_result.get("best_pick") or {}
    mv = bp.get("movement") or {}
    if mv.get("status") != "available" or not mv.get("direction"):
        return flags
    direction = mv["direction"]
    mb = se_result.get("market_block") or {}
    pick_dir: str | None = None
    market = bp.get("market")
    if market == "Total":
        key = "under" if str(bp.get("selection", "")).startswith("Under") else "over"
        ou = mb.get("ou") or {}
        opening, latest = ou.get(f"opening_{key}"), ou.get(f"latest_{key}")
        if ou.get("canonical") and opening and latest:
            pick_dir = (
                "toward" if float(latest) < float(opening)
                else ("away" if float(latest) > float(opening) else "none")
            )
    elif market == "Asian Handicap":
        side = bp.get("side") or (
            "home" if str(bp.get("selection", "")).startswith("Home") else "away"
        )
        ah = mb.get("ah") or {}
        opening, latest = ah.get(f"{side}_open"), ah.get(f"{side}_latest")
        if ah.get("canonical") and opening and latest:
            pick_dir = (
                "toward" if float(latest) < float(opening)
                else ("away" if float(latest) > float(opening) else "none")
            )
    if pick_dir in ("toward", "away") and pick_dir != direction:
        flags.append(
            f"narrative movement '{direction}' contradicts displayed pick-side "
            f"movement '{pick_dir}'"
        )
    return flags


# --------------------------------------------------------------------------
# Top-level entrypoint (pure)
# --------------------------------------------------------------------------

def run_signal_engine(
    *,
    model_probs: dict[str, Any],
    stats: dict[str, Any],
    market_totals: dict[str, dict[str, float]],
    ah_rows: list[dict[str, Any]],
    movement_snapshot: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    completeness: float = 0.0,
    cfg: dict[str, Any] | None = None,
    prediction_timestamp: str | None = None,
    history_snapshots: list[dict[str, Any]] | None = None,
    opening_snapshot: dict[str, Any] | None = None,
    previous_pick: dict[str, Any] | None = None,
    score_threshold: float | None = None,
    now_ts: str | None = None,
    odds_quality: dict[str, Any] | None = None,
    has_h2h: bool = True,
    model_decision_type: str | None = None,
    league_calibrated: bool = True,
    market_intel: dict[str, Any] | None = None,
    league_name: str | None = None,
    x2_market_dev_pp: float | None = None,
    odds_1x2: dict[str, Any] | None = None,
    team_form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, score, rank signals and return the Best Pick (JSON-safe).

    ``team_form`` (K2, 2026-08-28) = ``{"home": {"sequence", "ga_avg"},
    "away": {...}}`` -- our own form data for the source-consistency gate.
    ``context["tie_state"]`` (K3) carries the two-legged tie context.

    ``cfg`` = ``models.signal_engine`` (weights + thresholds). Deterministic:
    no I/O, no randomness. Returns a dict ready for the Discord formatter and
    for the prediction-log snapshot.

    ``history_snapshots`` (optional) are the accumulated ``odds_snapshot``
    rows from the background poll; when present, AH/Over-Under signals use the
    richer multi-snapshot movement (line + price + consistency + reversal +
    late) instead of a single opening->latest pair.
    """
    cfg = cfg or {}
    model_version = cfg.get("model_version", "v2")
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(cfg.get("weights") or {})
    min_edge_pp = float(cfg.get("min_edge_pp", MIN_EDGE_PP))
    conflict_pp = float(cfg.get("conflict_pp", CONFLICT_PP))
    best_pick_margin = float(cfg.get("best_pick_margin", BEST_PICK_MARGIN))
    no_bet_score = float(cfg.get("no_bet_score", NO_BET_SCORE))
    min_confluence = int(cfg.get("min_confluence", MIN_CONFLUENCE))
    min_data_quality = float(cfg.get("min_data_quality", MIN_DATA_QUALITY))
    # Option A: adjust completeness to exclude disabled components.
    # When team_context or statistical is off (weight=0), missing data
    # for those fields should NOT count as incomplete.
    adj_completeness = _adjust_completeness_for_weights(completeness, weights)

    signals = build_signals(
        model_probs=model_probs,
        stats=stats,
        market_totals=market_totals,
        ah_rows=ah_rows,
        movement_snapshot=movement_snapshot,
        context=context,
        completeness=completeness,
        history_snapshots=history_snapshots,
        opening_snapshot=opening_snapshot,
        # Plan v3 F14: 1X2 candidates are config-gated (default ON).
        odds_1x2=(
            odds_1x2
            if bool(cfg.get("enable_1x2_signals", True)) else None
        ),
    )
    # Inject market_intelligence into all signals for scoring.
    if market_intel and market_intel.get("usable"):
        for s in signals:
            s.components["_market_intelligence"] = market_intel
    score_signals(
        signals,
        weights=weights,
        min_edge_pp=min_edge_pp,
        conflict_pp=conflict_pp,
        completeness=adj_completeness,
        context=context,
        evidence_floor_cfg=(cfg.get("evidence_floor") if model_version == "v2" else None) if isinstance(cfg, dict) else None,
        league_calibrated=league_calibrated,
        # Plan v3 F4-lite: reward AGREEMENT with the market price instead of
        # divergence; band = G2 max_dev_pp so the component decays to 0
        # exactly where the hard gate fires.
        market_agreement_band_pp=(
            float((_pg_cfg_early := cfg.get("pick_gates") or {}).get(
                "max_dev_pp", DEFAULT_MAX_DEV_PP))
            if bool(cfg.get("market_component_reward_agreement", True))
            else None
        ),
    )
    # Post-scoring adjustments: match-level properties (direction, decisive,
    # high-scoring) applied AFTER per-signal scoring.
    _apply_post_scoring_adjustments(signals, model_probs, league_calibrated=league_calibrated)
    _tie_state = (context or {}).get("tie_state") if isinstance(context, dict) else None
    _apply_tie_state_adjustments(
        signals, _tie_state, (cfg.get("tie_state_penalties") if isinstance(cfg, dict) else None),
    )
    # ------------------------------------------------------------------
    # BEST PICK hard gates -- post-mortem 2026-08-22.
    # Evidence: reports/bestpick_postmortem_2026-08-22.md
    #
    # These SUPERSEDE the first loser-guard iteration, whose thresholds
    # (max_total_dev_pp=20 / max_total_dev_pp_unseeded=8, Total-only,
    # upper-tail-only) were tuned on the 2026-08-21 losing set alone and are
    # overfit. Out of sample on 2026-08-20 a 20pp Total veto kills OFI Crete
    # Over 2.5 (+26.5pp dev, WON +1.15u) and Gent Under 2.5 (+21.1pp dev, WON
    # +0.74u) to save SV Ried (-1.00u) -> net -0.89u.
    #
    # The supported rule is a SYMMETRIC agreement REQUIREMENT on EVERY market,
    # not an upper-tail veto on Totals. Totals book, 2026-08-20/21 combined:
    #
    #   |model - market| <= 8pp   n=4  4W-0L  +2.09u  (+52% ROI)
    #   +10..13pp                 n=4  1W-3L  -2.30u  (-57% ROI)
    #   > 20pp                    n=3  2W-1L  +0.89u
    #
    # Applying <=8pp turns that book from 7W-4L/+0.68u on 11 units into
    # 4W-0L/+2.09u on 4 units. It forgoes 3 winners and removes 4 losers --
    # net +1.41u -- and it matches the walk-forward audit on 1,520 EPL matches
    # vs real Pinnacle closing odds (reports/signal_audit_2026_08_12.md), where
    # the 0-2pp divergence bucket is the ONLY one in which the model beats the
    # closing line ("the model is worst exactly where it diverges most").
    _pg_cfg = cfg.get("pick_gates") or {}
    _dg_cfg = cfg.get("disagreement_gate") or {}
    _gate_reasons: list[str] = []

    def _veto(sig: "Signal", reason: str) -> None:
        """Mark a candidate ineligible WITHOUT touching its score.

        The score stays as scored so the card can still show what the evidence
        was worth next to the reason it was rejected.
        """
        if not sig.vetoed:
            _gate_reasons.append(f"{sig.selection}: {reason}")
        sig.vetoed = True
        sig.veto_reasons.append(reason)

    # G4 (card-level): a broken lambda_total invalidates EVERY candidate --
    # the Totals probability and the AH matrix are both built from it.
    # Verified: SV Ried lambda_total 4.09 on L-L-D-W-L form, FT 1-0.
    # 2026-08-22: the band is league-aware -- high-scoring leagues
    # (Eredivisie ~3.2 goals/game) legitimately reach totals the global
    # [1.6, 3.6] band rejects (Fortuna Sittard v AZ: lambda 3.96, market
    # Over 2.5 @ 1.44); overrides live in
    # models.signal_engine.pick_gates.lambda_total_band_by_league.
    # 2026-08-23 (P4): the reason carries its own evidence -- band source,
    # ceiling, model total and the market-implied total -- so a reader can
    # tell a +28pp divergence (SV Ried) from a +7.8pp one (Club Brugge v
    # Cercle) instead of seeing the same static text on every card.
    if bool(_pg_cfg.get("lambda_total_sanity", True)):
        _g4_lo, _g4_hi = resolve_lambda_total_band(_pg_cfg, league_name)
        _ok_g4, _rs_g4 = lambda_total_gate(
            (model_probs or {}).get("lambda_home"),
            (model_probs or {}).get("lambda_away"),
            lo=_g4_lo,
            hi=_g4_hi,
        )
        if not _ok_g4:
            _lam_h = (model_probs or {}).get("lambda_home")
            _lam_a = (model_probs or {}).get("lambda_away")
            _model_total = float(_lam_h) + float(_lam_a)
            _mkt_total = market_implied_total(market_totals)
            _gap_txt: str | None = None
            if _mkt_total is not None:
                _gap = _model_total - _mkt_total
                _gap_txt = f"{_gap:+.2f}"
            _ctx_parts = [
                f"band_source={band_source(_pg_cfg, league_name)}",
                f"ceiling={_g4_hi:.1f}",
                f"model_total_lambda={_model_total:.2f}",
                (
                    f"market_implied_total={_mkt_total:.2f}"
                    if _mkt_total is not None
                    else "market_implied_total=n/a"
                ),
            ]
            if _gap_txt is not None:
                _ctx_parts.append(f"model_market_gap={_gap_txt}")
            _reason_g4 = f"{_rs_g4[0]} [{', '.join(_ctx_parts)}]"
            for s in signals:
                s.vetoed = True
                s.veto_reasons.append(_reason_g4)
            _gate_reasons.append(_reason_g4)

    # K2 (card-level, 2026-08-28): our own team data must not contradict the
    # market's view of a heavy favourite. Verified FC Copenhagen v Inter Turku
    # 2026-08-27: market 80% home, our form D-L-L-L-L / 5.2 conceded per game
    # / lambda underdog -> the entity behind "Copenhagen" was wrong; BTTS No
    # emitted on that lambda lost 4-1. Nothing on such a card is trustworthy.
    _entity_mismatch: dict[str, Any] | None = None
    if bool(_pg_cfg.get("source_consistency", True)):
        _ok_k2, _rs_k2, _k2_detail = source_consistency_gate(
            odds_1x2, team_form, model_probs,
            fav_implied_min=float(_pg_cfg.get("consistency_fav_implied", 0.70)),
            max_wins=int(_pg_cfg.get("consistency_max_wins", 1)),
            min_form_len=int(_pg_cfg.get("consistency_min_form", 4)),
            min_ga_avg=float(_pg_cfg.get("consistency_min_ga", 3.0)),
        )
        if not _ok_k2:
            _entity_mismatch = _k2_detail
            for s in signals:
                s.vetoed = True
                s.veto_reasons.append(_rs_k2[0])
            _gate_reasons.append(_rs_k2[0])

    # G5 (2026-08-28, finally WIRED -- the gate existed since 2026-08-22 but
    # had no call site). Two parts:
    #   range/collision (card-level): a rating outside [elo_min, elo_max] or
    #     identical on both sides is a lookup failure, not a strength
    #     estimate (Rapid Wien 1291 v "Hearts" 1031 = Kelty Hearts, 2026-08-26
    #     HIGH Over 2.5, FT 1-1);
    #   evidence scope: BOTH sides on the 1500 prior -> directional picks
    #     (Home/Away Win, AH) are vetoed and the rest capped LOW; ONE side on
    #     the prior -> directional picks capped MEDIUM. Draw / Totals / BTTS
    #     do not depend on which side is stronger and stay eligible.
    _elo_scope: str | None = None
    _elo_note: str | None = None
    if bool(_pg_cfg.get("elo_integrity", False)):
        _ok_g5, _rs_g5 = elo_integrity_gate(
            model_probs,
            lo=float(_pg_cfg.get("elo_min", DEFAULT_ELO_MIN)),
            hi=float(_pg_cfg.get("elo_max", DEFAULT_ELO_MAX)),
            require_seeded=False,
            collision_eps=float(_pg_cfg.get("elo_collision_eps", DEFAULT_ELO_COLLISION_EPS)),
        )
        if not _ok_g5:
            _reason_g5 = "; ".join(_rs_g5)
            for s in signals:
                s.vetoed = True
                s.veto_reasons.append(_reason_g5)
            _gate_reasons.append(_reason_g5)
        _elo_scope, _elo_note = elo_evidence_scope(model_probs)
        if _elo_scope == "all":
            for s in signals:
                if is_directional_selection(s.market, s.selection):
                    _veto(s, _elo_note or "kedua tim tanpa Elo — pick directional diveto")
                else:
                    s.confidence_cap = "LOW"
                    if _elo_note and _elo_note not in s.internal_notes:
                        s.internal_notes.append(_elo_note)
        elif _elo_scope == "one":
            for s in signals:
                if is_directional_selection(s.market, s.selection):
                    s.confidence_cap = "MEDIUM"
                    if _elo_note and _elo_note not in s.internal_notes:
                        s.internal_notes.append(_elo_note)

    # G2 (per candidate, ALL markets): agreement with the margin-free price.
    if bool(_pg_cfg.get("agreement", True)):
        _max_dev = float(_pg_cfg.get("max_dev_pp", DEFAULT_MAX_DEV_PP))
        for s in signals:
            _ok_g2, _rs_g2 = agreement_gate(
                s.model_prob, s.implied_prob, max_dev_pp=_max_dev,
            )
            if not _ok_g2:
                _veto(s, _rs_g2[0])

    # G7 (per candidate): a pick without a tradeable price is not a pick.
    # Verified: Braga v Austria Wien Women 2026-08-20 shipped market_odds null.
    if bool(_pg_cfg.get("require_price", True)):
        for s in signals:
            _ok_g7, _rs_g7 = price_gate(s.market_odds)
            if not _ok_g7:
                _veto(s, _rs_g7[0])

    # G3 (market-scoped, DEFAULT OFF): a lambda-direction contradiction kills
    # every DIRECTIONAL candidate, not merely the opposing side. This is
    # STRICTER than the (c) side rule below and is NOT yet supported by
    # evidence: on Erzurumspor 2026-08-21 the (c) rule leaves AH Away +0.25
    # standing, and that would have WON (FT 0-4). Measure on the full log
    # before enabling. Kept wired so enabling is a config flip, not a patch.
    _g3_directional = bool(_pg_cfg.get("lambda_1x2_consistency", False))
    # G3-low (2026-08-28, default ON): the same contradiction ALSO corrupts
    # picks that need FEW goals. When the ensemble says one side is the
    # favourite but the lambda matrix has that side scoring LESS, the
    # favourite's goals are understated -> P(Under) / P(BTTS No) inflated.
    # Verified Copenhagen v Inter Turku 2026-08-27 (1X2 home 64%, lambda
    # 0.81 v 1.53 -> BTTS No @1.76, FT 4-1). Over / BTTS Yes are untouched:
    # Arsenal v Coventry and LASK v Celtic won Over 2.5 with the same
    # contradiction present.
    _g3_low = bool(_pg_cfg.get("lambda_1x2_low_scoring", True))
    if _g3_directional or _g3_low:
        _ok_g3, _rs_g3 = lambda_1x2_gate(
            model_probs,
            favourite_prob=float(_pg_cfg.get("lambda_1x2_favourite_prob", 0.60)),
        )
        if not _ok_g3:
            for s in signals:
                if _g3_directional and s.market in DIRECTIONAL_MARKETS:
                    _veto(s, _rs_g3[0])
                elif _g3_low and is_low_scoring_selection(s.market, s.selection):
                    _veto(s, _rs_g3[0] + " — pick minim gol ikut korup (gol favorit understated)")

    # Loser-guard remnants that G2 does NOT subsume, kept because each is
    # independently evidence-backed:
    #   (b) the model's 1X2 contradicting the market vetoes script-dependent
    #       markets (Real Betis class) while AH cover candidates stay eligible;
    #   (c) an AH candidate whose SIDE opposes a strong model-1X2 favourite
    #       (Erzurumspor / Al Riyadh class).
    # (a) total-only 20pp veto and (d) unseeded 8pp veto are REMOVED -- G2
    # covers both, symmetrically and across all markets.
    if bool(_dg_cfg.get("enabled", False)):
        if x2_market_dev_pp is not None and abs(x2_market_dev_pp) > float(
            _dg_cfg.get("max_x2_dev_pp", 25.0)
        ):
            _r = (f"model 1X2 menyimpang {x2_market_dev_pp:+.1f}pp dari pasar — "
                  "kandidat Total/BTTS diveto (pola Betis 2026-08-21)")
            for s in signals:
                if s.market in ("Total", "BTTS"):
                    _veto(s, _r)
        # (c) AH side-contradiction: an AH candidate whose side OPPOSES a
        # STRONG model-1X2 favorite inherits its probability from the raw
        # lambda matrix -- which, for unseeded/thin-form teams, can point the
        # OPPOSITE way from the ensemble 1X2 on the same card (verified
        # 2026-08-21: Erzurumspor lam_h>lam_a yet 1X2 away 61% -> AH Home+1
        # "prob" 0.766, lost 0-4; Al Riyadh 1X2 away 82% -> AH Home+1.75,
        # lost 0-4). When the 1X2 top side carries >= threshold probability,
        # opposing-side AH candidates are vetoed. Aligning-side AH (hedge)
        # candidates are never touched.
        _p1x2v = (model_probs or {}).get("1x2") or {}
        _thr_side = float(_dg_cfg.get("ah_side_contradiction_prob", 0.60))
        if _thr_side > 0 and _p1x2v:
            _top_side = max(("home", "away"), key=lambda k: float(_p1x2v.get(k, 0.0)))
            _top_p = float(_p1x2v.get(_top_side, 0.0))
            if _top_p >= _thr_side:
                for s in signals:
                    if s.market == "Asian Handicap" and s.side and s.side != _top_side:
                        _veto(s, (
                            f"melawan favorit 1X2 model ({_top_side} {_top_p:.0%}) — "
                            "λ dasar prob AH kontradiktif dgn 1X2 "
                            "(pola Erzurum/Al-Nassr 2026-08-21)"
                        ))
    # P3: cross-source odds disagreement docks the data-quality evidence the
    # signal is built on (the odds consensus itself is unreliable when two
    # independent sources disagree on key lines). The dock is applied BEFORE
    # scoring/ranking so it flows into the weighted score, the confidence
    # label and the min_data_quality gate -- not just a cosmetic flag.
    disagreement = bool(
        odds_quality and odds_quality.get("status") == "cross_source_disagreement"
    )
    eff_completeness = (adj_completeness * 0.5) if disagreement else adj_completeness
    # F2 (evidence floor): a model whose λ comes ONLY from a prior Elo rating
    # (teams never seeded -> 1500 default) plus a form window too thin for the
    # statistical component to mean anything (< MIN_EVIDENCE_FORM_MATCHES) is
    # NOT enough evidence to advertise a BEST PICK. When H2H is also absent
    # the pick is VETOED to NO BET (the ADO-Den-Haag-class incident: HOME -0
    # at 62/100 built on a 1500 prior + 1-match form, lost 0-2); when H2H
    # exists the confidence is capped LOW and the reason is surfaced on the
    # card. ``model_probs`` already carries elo_seeded / lambda_source from
    # the prediction engine -- this is the first consumer that actually uses
    # them instead of treating the prior as measured strength.
    # F2 (evidence floor) — only active for model_version "v2"
    evidence_floor: dict[str, Any] | None = None
    if model_version == "v2":
        _mp = model_probs or {}
        if _mp.get("lambda_source") == "elo" and _mp.get("elo_seeded") is False:
            _hg = _recent_goals((stats or {}).get("home_recent_goals"))
            _ag = _recent_goals((stats or {}).get("away_recent_goals"))
            _thin = len(_hg) < MIN_EVIDENCE_FORM_MATCHES or len(_ag) < MIN_EVIDENCE_FORM_MATCHES
            if _thin:
                evidence_floor = {
                    "veto": not has_h2h,
                    "note": (
                        "model berbasis prior Elo (tim belum terseed) dengan form tipis "
                        f"(< {MIN_EVIDENCE_FORM_MATCHES} match) dan tanpa dukungan H2H"
                        if not has_h2h
                        else "model berbasis prior Elo (tim belum terseed) dengan form tipis "
                        f"(< {MIN_EVIDENCE_FORM_MATCHES} match) — confidence dibatasi"
                    ),
                }
    if _elo_scope == "all" and evidence_floor is None:
        # K1: both sides on the prior. Directional candidates are already
        # vetoed above; whatever survives (Draw / Totals / BTTS) must never
        # advertise more than LOW -- reuse the evidence-floor cap path.
        evidence_floor = {"veto": False, "note": _elo_note or "kedua tim tanpa Elo — confidence dibatasi"}
    result = rank_and_pick(
        signals,
        best_pick_margin=best_pick_margin,
        no_bet_score=no_bet_score,
        min_confluence=min_confluence,
        conflict_pp=conflict_pp,
        min_data_quality=min_data_quality,
        completeness=eff_completeness,
        min_edge_pp=min_edge_pp,
        confidence_thresholds=cfg,
        odds_disagreement=disagreement,
        evidence_floor=evidence_floor,
        model_decision_type=model_decision_type,
    )
    result["weights"] = weights
    if _gate_reasons:
        result["disagreement_gate"] = _gate_reasons
    result["prediction_timestamp"] = prediction_timestamp
    # P0-4: Lambda vs 1X2 contradiction detection
    lambda_warning = None
    _lh = model_probs.get("lambda_home", 0)
    _la = model_probs.get("lambda_away", 0)
    _p1x2 = model_probs.get("1x2") or {}
    if _lh and _la and _p1x2:
        lambda_home_better = _lh > _la
        model_home_better = _p1x2.get("home", 0) > _p1x2.get("away", 0)
        if lambda_home_better != model_home_better:
            _dir_lambda = "Home" if lambda_home_better else "Away"
            _dir_model = "Home" if model_home_better else "Away"
            _diff = abs(_p1x2.get("home", 0) - _p1x2.get("away", 0)) * 100
            lambda_warning = (
                f"Lambda vs 1X2 contradiction: Lambda favors {_dir_lambda} "
                f"({_lh:.2f} vs {_la:.2f}), but 1X2 favors {_dir_model} "
                f"({_diff:.1f}pp gap)"
            )
    result["data_quality"] = {
        "completeness": round(completeness, 3),
        "odds_quality": (odds_quality or {}).get("status") or "ok",
        "odds_max_pp_diff": (odds_quality or {}).get("max_pp_diff"),
        "odds_sources": (odds_quality or {}).get("sources"),
        "ah_available": bool(ah_rows),
        "movement_history_available": bool(movement_snapshot and movement_snapshot.get("usable")),
        "ah_ou_snapshots": len(history_snapshots or []),
        "lambda_warning": lambda_warning,
    }
    # Raw market consensus for the presentation layer (MARKET block: opening
    # line/odds -> latest line/odds). ``market_totals`` already carries the
    # Over/Under + BTTS opening prices in the analyse payload; the AH opening
    # is only available here, so expose it JSON-safe.
    result["ah_consensus"] = ah_consensus(ah_rows)
    # K5 (2026-08-28): tier the label instead of vetoing. A pick below
    # ``medium_score`` or labelled LOW is a LEAN, not a BEST PICK -- 25-27 Aug
    # LOW picks went 8-4 while MEDIUM+ went 12-2; printing both as
    # "BEST PICK" hid that difference from the reader and from the stats.
    _medium_score = float(cfg.get("medium_score", MEDIUM_SCORE))
    if result["best_pick"] is not None:
        _bp_obj = result["best_pick"]
        result["pick_tier"] = (
            "LEAN"
            if (float(_bp_obj.score) < _medium_score or _bp_obj.confidence == "LOW")
            else "BEST PICK"
        )
    else:
        result["pick_tier"] = None
    result["tier_threshold"] = _medium_score
    # K1/K2/K3 audit trail (persisted with the snapshot via the analyser).
    result["elo_scope"] = _elo_scope
    result["entity_mismatch"] = _entity_mismatch
    result["tie_state"] = _tie_state
    if result["best_pick"] is not None:
        bp = result["best_pick"]
        result["best_pick"] = {
            "market": bp.market,
            "selection": bp.selection,
            "score": bp.score,
            "confidence": bp.confidence,
            "model_prob": round(bp.model_prob, 4),
            "market_odds": bp.market_odds,
            # Phase 5.3: the margin-free implied probability is part of the
            # honest "Why" -- the card compares model_prob vs implied instead
            # of a boilerplate narrative.
            "implied_prob": round(bp.implied_prob, 4) if bp.implied_prob else None,
            "edge_pp": bp.edge_pp,
            "line": bp.line,
            "side": bp.side,
            "components": bp.components,
            "confluence": _confluence(bp.components),
            "movement": bp.movement,
            # F2/F3: explicit reasons why the pick was capped/downgraded
            # (prior-Elo evidence floor, 1X2-layer disagreement) -- rendered
            # on the card as ⚠️ notes, never a silent confidence.
            "evidence_notes": list(getattr(bp, "evidence_notes", None) or []),
            # P2-3: F3 model-vs-model disagreements (e.g. "model 1X2 NO BET —
            # pick tidak didukung layer model") live here, NOT in
            # evidence_notes -- the summary embed filters internal_notes out;
            # only the expanded view renders them after "Lihat Hasil".
            "internal_notes": list(getattr(bp, "internal_notes", None) or []),
        }
        # CLV tracking: log entry when a pick is made.
        if league_name and bp.market_odds:
            try:
                log_clv_entry(
                    match_id=model_probs.get("match_id", "unknown"),
                    league=league_name,
                    market=bp.market,
                    selection=bp.selection,
                    entry_odds=bp.market_odds,
                    model_prob=bp.model_prob,
                    confidence=bp.confidence,
                    best_pick=bp.selection,
                )
            except Exception:
                pass  # CLV logging is best-effort, never blocks the signal
    # Layer 1: the ONE opening reference for the display MARKET block (the
    # same canonical snapshot the scoring movement reads).
    result["market_block"] = build_market_block(
        market_totals=market_totals,
        ah=result.get("ah_consensus"),
        opening_snapshot=opening_snapshot,
        history_snapshots=history_snapshots,
    )
    # Market intelligence results (steam/RLM/agreement)
    if market_intel and market_intel.get("usable"):
        result["market_intelligence"] = {
            "side": market_intel.get("side"),
            "confidence": market_intel.get("confidence", 0.0),
            "model_agreement": market_intel.get("model_agreement", 0.5),
            "steam_moves": market_intel.get("steam_moves", []),
            "rlm": market_intel.get("rlm"),
            "agreement": market_intel.get("agreement", {}),
            "reasons": market_intel.get("reasons", []),
        }
    # Layer 3: repeated-query stability guard (pure; the caller supplies the
    # IMMUTABLE prior pick from prediction_log and the calibrated threshold).
    if previous_pick is not None:
        result = apply_pick_stability(
            result,
            previous_pick=previous_pick,
            current_model=model_probs,
            opening_snapshot=opening_snapshot,
            market_totals=market_totals,
            now_ts=now_ts or prediction_timestamp,
            cfg=cfg,
            score_threshold=score_threshold,
        )
    return result


# --------------------------------------------------------------------------
# Lightweight backtest (S34): settle BTTS / Over 2.5 / AH signals
# --------------------------------------------------------------------------

def settle_signal(signal: dict[str, Any], home_goals: int, away_goals: int) -> dict[str, Any]:
    """Settle one signal's selection against a final score.

    BTTS: yes -> win if both score. Total Over/Under 2.5: compare to 2.5.
    Asian Handicap: ``ah_settle`` (quarter-line semantics). Returns
    {result, return_value, stake_return} where stake_return is the multiple
    of the stake returned (0 loss, 1 full win, 0.5 push, 0.75 half win ...).
    """
    market = signal.get("market")
    sel = signal.get("selection", "")
    if market == "1X2":
        outcome = "home" if home_goals > away_goals else (
            "draw" if home_goals == away_goals else "away"
        )
        side_map = {"Home Win": "home", "Draw": "draw", "Away Win": "away"}
        hit = side_map.get(sel) == outcome
        return {"result": "win" if hit else "loss", "return_value": 1.0 if hit else 0.0,
                "stake_return": 1.0 if hit else 0.0}
    if market == "BTTS":
        both = home_goals > 0 and away_goals > 0
        hit = both if sel in ("Yes", "BTTS Yes") else (not both)
        return {"result": "win" if hit else "loss", "return_value": 1.0 if hit else 0.0,
                "stake_return": 1.0 if hit else 0.0}
    if market == "Total":
        total = home_goals + away_goals
        over = total > 2.5
        hit = over if sel.startswith("Over") else (not over)
        return {"result": "win" if hit else "loss", "return_value": 1.0 if hit else 0.0,
                "stake_return": 1.0 if hit else 0.0}
    if market == "Asian Handicap":
        line = signal.get("line")
        side = signal.get("side") or ("home" if sel.startswith("Home") else "away")
        if line is None:
            return {"result": "n/a", "return_value": 0.0, "stake_return": 0.0}
        s = ah_settle(home_goals, away_goals, line, side)
        return {"result": s["result"], "return_value": s["return_value"],
                "stake_return": s["return_value"]}
    return {"result": "n/a", "return_value": 0.0, "stake_return": 0.0}


def run_signal_backtest(
    records: list[dict[str, Any]],
    *,
    min_edge_pp: float = MIN_EDGE_PP,
) -> dict[str, Any]:
    """Backtest a list of prediction records, each with a Best Pick + result.

    ``records``: [{market, selection, line, side, market_odds, home_goals,
    away_goals}]. No data leakage: the caller must supply only pre-match
    fields; settlement uses the final score. Reports accuracy by signal
    family, half-win accounting, and flat-stake ROI where odds exist.
    """
    fam = {}
    n_no_bet = 0
    for r in records:
        if r.get("decision") == "NO BET":
            n_no_bet += 1
            continue
        settle = settle_signal(r, int(r.get("home_goals") or 0), int(r.get("away_goals") or 0))
        market = r.get("market") or "?"
        bucket = fam.setdefault(market, {"n": 0, "wins": 0, "half_wins": 0, "pushes": 0,
                                         "half_losses": 0, "losses": 0, "ret": 0.0,
                                         "roi": 0.0, "staked": 0.0})
        bucket["n"] += 1
        res = settle["result"]
        if res == "win":
            bucket["wins"] += 1
        elif res == "half_win":
            bucket["half_wins"] += 1
        elif res == "push":
            bucket["pushes"] += 1
        elif res == "half_loss":
            bucket["half_losses"] += 1
        else:
            bucket["losses"] += 1
        odds = r.get("market_odds")
        if odds and odds > 1.0:
            bucket["staked"] += 1.0
            bucket["ret"] += settle["stake_return"] * float(odds)
    for b in fam.values():
        if b["staked"] > 0:
            b["roi"] = round((b["ret"] - b["staked"]) / b["staked"] * 100.0, 2)
        denom = b["n"]
        b["win_rate"] = round(
            (b["wins"] + 0.5 * b["half_wins"] + 0.5 * b["pushes"]) / denom, 4
        ) if denom else None
    total_n = sum(b["n"] for b in fam.values())
    return {"markets": fam, "n": total_n, "no_bet": n_no_bet}


# --------------------------------------------------------------------------
# CLI: demo + backtest (honest, no data fabrication)
# --------------------------------------------------------------------------

def _demo() -> dict[str, Any]:
    """Synthetic high-goal scenario for output verification (no network)."""
    from .models import probs_from_matrix
    lh = la = 1.6
    m = poisson_matrix(lh, la, rho=0.0)
    p1x2, o15, o25, o35, btts = probs_from_matrix(m)
    model = {
        "1x2": p1x2, "over_1.5": o15, "over_2.5": o25, "over_3.5": o35,
        "btts_yes": btts, "lambda_home": lh, "lambda_away": la,
    }
    totals = {
        "Over 2.5": {"odds": 1.99, "point": 2.5, "opening": 2.08},
        "Under 2.5": {"odds": 1.94, "point": 2.5, "opening": 1.86},
        "BTTS Yes": {"odds": 1.75, "opening": 1.80},
        "BTTS No": {"odds": 2.05, "opening": 2.00},
    }
    ah_rows = extract_asian_handicap(_AH_DEMO_PAYLOAD)
    return run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.7, cfg=None,
    )


_AH_DEMO_PAYLOAD = {
    "bookmakers": [
        {
            "title": "Pinnacle",
            "markets": [
                {
                    "key": "asian_handicap",
                    "outcomes": [
                        {"name": "Home", "price": 1.95, "point": -0.25, "opening_price": 2.05},
                        {"name": "Away", "price": 1.95, "point": -0.25, "opening_price": 1.86},
                    ],
                }
            ],
        }
    ]
}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="signal_engine")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="run the engine on a synthetic scenario")
    backtest = sub.add_parser("backtest", help="settle prediction records")
    backtest.add_argument("--records", required=True,
                          help="JSON file of [{market, selection, line, side, market_odds, home_goals, away_goals}]")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        print(json.dumps(_demo(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "backtest":
        with open(args.records, encoding="utf-8") as fh:
            records = json.load(fh)
        print(json.dumps(run_signal_backtest(records), ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
