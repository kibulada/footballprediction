"""Transparent decision engine (master-prompt Phase 1: S18, S22-S31).

Correction-spec integration (prediction-engine correction spec):
  - Section 1: Model A (odds-implied) never enters this engine; candidates are
    ALWAYS Model B (independent Elo+Poisson+calibration). The disagreement
    check runs upstream (run_decision_engine) and the flag is passed in.
  - Section 2: when ``bucket_n`` is provided (production path), every
    candidate is hard-gated (EV > min_ev, |edge| < extreme, n_bucket >= 30,
    completeness >= 0.6); failures are excluded from the final decision and
    listed under ``blocked`` with the failed condition named.
  - Section 3: ``pick_specific_confidence`` (multiplicative) is reported
    separately from ``model_calibration_score`` (global).
  - Section 4: an extreme edge on ANY market propagates a 0.5 confidence
    penalty to every λ-shared market and forbids GOOD anywhere.

Replaces the old ``signal >= 70 -> rank by edge / else -> rank by
probability`` switch in ``derive_picks`` with a documented, configurable
Decision Score. The engine is PURE: it only combines signals that are
already computed by the prediction engine (calibrated independent model
probabilities, margin-free market implied, agreement, calibration quality,
completeness, odds liquidity, similar-signal history).

Anti double-counting (S25):
  - Value is only credited to candidates whose model probability is
    INDEPENDENT of the odds being priced (the Elo+Poisson ensemble). The
    odds-derived Poisson picks (lambda solved FROM the market, see
    ``predictor.solve_lambdas``) mirror the market and therefore have no
    independent value -- their ``independent=False`` and their market_value
    component is forced to 0 so market information is never counted twice.
  - ``market_edge`` used by the model-agreement term is the same market, but
    agreement measures DISAGREEMENT (information), not value -- distinct
    role, documented.

Confidence (S26): probability magnitude and separation enter the decision
score via ``probability_quality``; the SignalScorer's confidence label also
gains a ``decisiveness`` component (configurable weight) in calibration.py.

Decision types (S27-28): STRONG / GOOD / LEAN / WATCH / NO CLEAR DECISION /
NO BET / MARKET PRIOR. NO CLEAR DECISION, NO BET and MARKET PRIOR are valid
outputs, not errors. MARKET PRIOR is the thin-data honesty fallback: when the
independent engine has no usable signal (completeness below the bettable
floor), predictions are built from the margin-free market itself -- explicitly
labelled, edge = 0 by construction, betting advice NO BET.

Extreme edge protection (S18): |edge| >= warning flags the candidate;
|edge| >= extreme caps its value credit AND caps the decision type at LEAN
(an audit is required before such a pick is treated as an opportunity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .model_gates import (
    MIN_BUCKET_N,
    MIN_COMPLETENESS,
    MIN_EV,
    TIER_VALUE,
    bucket_ci_halfwidth,
    min_tier,
    pick_confidence,
    pick_status,
)

# --------------------------------------------------------------------------
# Configurable defaults (mirrored in config/football.json -> models.decision)
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "probability_quality": 0.30,
    "calibration_reliability": 0.20,
    "model_agreement": 0.15,
    "market_value": 0.15,
    "data_quality": 0.10,
    "odds_quality": 0.05,
    "historical_reliability": 0.05,
}

EDGE_WARNING_PP = 10.0    # flag: large edge -> verify inputs
EDGE_EXTREME_PP = 20.0    # cap value + cap type at LEAN, require audit
MIN_BOOKMAKERS = 3
MIN_CALIBRATION_SAMPLES = 200

STRONG_SCORE = 0.70
GOOD_SCORE = 0.55
LEAN_SCORE = 0.40
NO_CLEAR_MAX_SCORE = 0.35

MIN_SEPARATION = 0.10     # for a STRONG label
MIN_AGREEMENT = 0.50      # below this -> NO CLEAR DECISION
MIN_COMPLETENESS = 0.50   # below this -> NO CLEAR DECISION
MIN_CALIBRATION_QUALITY = 0.50  # below this -> cap at LEAN

# Minimum margin-free edge (percentage points) for a candidate to carry ANY
# market value. Mirrors the validated OLD decision rule (edge >= 2% on the
# best pick) -- the initial 0.0 floor let the engine bet on noise and lost
# to the OLD rule in walk-forward (S24/S37 tuning loop). Walk-forward
# (EPL 2022-26) showed 3.0pp (with best_prob_only) is the first config that
# matches the OLD rule's ROI; config/football.json ships the validated
# values. This constant is the fallback when no config is present.
MIN_EDGE_PP = 3.0

MAX_EV_FOR_FULL_VALUE = 0.20     # EV >= +20% -> market_value = 1.0
EXTREME_VALUE_CAP = 0.30         # extreme-edge candidates: value credit cap

DECISION_TYPES = (
    "STRONG", "GOOD", "LEAN", "WATCH", "NO CLEAR DECISION", "NO BET",
    "MARKET PRIOR",
)

# Thin-data detection floor: when the independent model's data_completeness
# is below this, its probabilities carry no usable signal (no form/history/
# Elo seed) and the engine falls back to an honest MARKET PRIOR prediction.
# Deliberately aligned with the decision engine's own bettable-completeness
# floor (min_completeness, default 0.6): anything below the bettable line
# gets the honest market-mirror prediction instead of a bare NO CLEAR
# DECISION, so there is no "cliff" where a match at 0.36 gets nothing while
# one at 0.34 gets rich MARKET PRIOR output.
MARKET_PRIOR_MIN_COMPLETENESS = 0.6


def market_prior_decision(
    consensus: dict[str, float],
    market_totals: dict[str, dict[str, float]],
    *,
    bookmakers_count: int = 0,
    min_bookmakers: int = MIN_BOOKMAKERS,
) -> dict[str, Any]:
    """Honest MARKET PRIOR prediction for thin-data matches.

    When the independent model has no usable signal (thin data: no
    form/history/Elo seed), the best estimator available is the market
    itself. This builds the prediction (most-likely 1X2, Over/Under, BTTS)
    FROM the market's margin-free probabilities -- so by construction
    ``edge = 0`` and no value is claimed: the betting advice is NO BET
    (EV after the bookmaker margin is negative). The label is explicit so
    the output can never be mistaken for an independent-model edge.

    Returns a JSON-safe dict shaped like ``decide()`` output so the existing
    renderers/prediction-log handle it unchanged (see ``decision_to_dict``).
    """
    norm = margin_free_implied(consensus) if consensus else None
    preds: dict[str, Any] = {}
    most_likely: dict[str, Any] | None = None
    if norm and norm.get("home") and norm.get("away"):
        sides = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}
        # Normalize over the sides the market actually priced (a missing draw
        # column would otherwise leave the three values summing below 1.0).
        priced = {k: v for k, v in norm.items() if v and v > 0.0}
        total = sum(priced.values())
        if total > 0:
            priced = {k: v / total for k, v in priced.items()}
            preds["1x2"] = {k: round(priced.get(k, 0.0), 4) for k in ("home", "draw", "away")}
            top = max(sides, key=lambda k: priced.get(k, 0.0))
            most_likely = {
                "market": "1X2",
                "selection": sides[top],
                "model_prob": round(priced.get(top, 0.0), 4),
                "market_odds": float(consensus.get(top, 0.0) or 0.0),
                "implied_prob": round(priced.get(top, 0.0), 4),
                "edge_pp": 0.0,
                "ev": 0.0,
                "independent": False,
                "score": 0.0,
                "components": {},
                "edge_level": "none",
            }
    for thresh in (2.5, 3.5):
        o = market_totals.get(f"Over {thresh}", {}).get("odds", 0.0)
        u = market_totals.get(f"Under {thresh}", {}).get("odds", 0.0)
        fair = fair_pair_implied(o, u)
        if fair:
            preds[f"over_{thresh}"] = round(fair[0], 4)
            preds[f"under_{thresh}"] = round(fair[1], 4)
    y = market_totals.get("BTTS Yes", {}).get("odds", 0.0)
    n = market_totals.get("BTTS No", {}).get("odds", 0.0)
    fair = fair_pair_implied(y, n)
    if fair:
        preds["btts_yes"] = round(fair[0], 4)
        preds["btts_no"] = round(fair[1], 4)

    return {
        "decision_type": "MARKET PRIOR",
        "final_decision": None,
        "most_likely": most_likely,
        "market_predictions": preds,
        "market_prior": True,
        "betting_advice": "NO BET",
        "explanation": (
            "Data model independen tidak cukup (tanpa form/history/Elo seed memadai). "
            "Prediksi 1X2/O-U/BTTS mengikuti probabilitas market (margin-free) — ini "
            "BUKAN klaim edge: karena prediksi = market, edge = 0 dan EV setelah margin "
            "bandar negatif. Saran taruhan: NO BET."
        ),
        "reasons": [
            "thin-data: model independen tanpa sinyal — prediksi mengikuti market (margin-free)",
            f"bookmaker: {bookmakers_count} (butuh >= {min_bookmakers} untuk value)",
        ],
        "edge_warnings": [],
        "score_breakdown": {},
        "evaluated": [],
        "blocked": [],
        "pick_specific_confidence": None,
        "model_calibration_score": None,
        "hard_cap_medium_applied": False,
        "form_depth_cap_applied": False,
        "enable_watch": False,
    }

# TODO-10: ensemble spread (max-min across component home probabilities)
# above this makes the estimate too unstable to BET on (TODO-16 WATCH gate).
WATCH_UNCERTAINTY_MAX = 0.25


def actionable_gate(
    *,
    league_calibrated: bool,
    edge_pp: float | None,
    min_edge_pp: float,
    benchmark_stale: bool,
    clv_gate: dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Phase 2.3: the ONLY gate that decides whether a pick is actionable.

    A pick is "actionable" only if ALL of:
      (a) its segment (league x market x timing) has realized CLV > 0
          (checked via the CLV gate result; when the CLV gate is disabled,
          the evidence check is deferred to the caller's config),
      (b) edge vs the FRESH benchmark meets the threshold (Phase 0.2: a
          stale benchmark invalidates the edge),
      (c) the league passes the per-league calibration minimum (Phase 1.5).

    This replaces the old STRONG/GOOD/LEAN confidence tiers as the driver
    of actionable picks -- the tiers stay as descriptive labels but never
    decide actionability. Flag-gated (``models.decision.actionable_gate
    .enabled``, default off) so existing behavior is unchanged until the
    Phase 2 DoD (4 weeks paper-trading) passes.
    """
    reasons: list[str] = []
    if not league_calibrated:
        reasons.append(
            "liga gagal kalibrasi minimum (Phase 1.5) — tidak actionable, label TOP SIGNAL"
        )
    if benchmark_stale:
        reasons.append("benchmark edge stale (Phase 0.2) — edge invalid")
    if edge_pp is None or edge_pp < min_edge_pp:
        reasons.append(f"edge {edge_pp if edge_pp is not None else 'n/a'} < {min_edge_pp}pp (benchmark fresh)")
    if clv_gate is not None and not clv_gate.get("allowed"):
        reasons.append(
            f"segmen CLV belum terbukti: {(clv_gate.get('reason') or 'gate blocked').lower()}"
        )
    # A missing clv_gate result (disabled) defers the CLV-evidence check to
    # the caller -- never invent evidence.
    return (not reasons), reasons


def market_blend_alpha(calibration_quality: float) -> float:
    """Phase 3.1: blend weight for p_final = a*p_market + (1-a)*p_model.

    Simplest mapping satisfying the spec's boundary condition
    ``quality = 0 -> a = 1.0`` (pure market price, edge = 0, NO BET):

        a = clamp01(1.0 - calibration_quality)

    quality=0 (uncalibrated) -> a=1.0 -> pure market (no independent edge);
    quality=1 -> a=0 -> pure calibrated model. Linear in between.
    Documented per ground rule #4; flagged for confirmation.
    """
    q = max(0.0, min(1.0, float(calibration_quality)))
    return 1.0 - q


def blend_model_with_market(
    model_probs: dict[str, Any],
    market_probs: dict[str, float] | None,
    calibration_quality: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase 3.1: p_final = a*p_market + (1-a)*p_model_calibrated (1X2).

    Only the 1X2 vector is blended (the market's freshest margin-free
    implied from the T-1h price). Returns (blended_probs, blend_meta). When
    a >= 1.0 the blend is pure market -> edge = 0 by construction and the
    caller must NOT bet; ``blend_meta["pure_market"]`` carries that flag.
    """
    a = market_blend_alpha(calibration_quality)
    meta = {
        "alpha": round(a, 4),
        "quality": round(max(0.0, min(1.0, float(calibration_quality))), 4),
        "pure_market": a >= 1.0,
    }
    p1 = model_probs.get("1x2") or {}
    out = dict(model_probs)
    if not p1 or not market_probs or a <= 0.0:
        return out, meta
    blended = {
        k: a * market_probs.get(k, 0.0) + (1.0 - a) * p1.get(k, 0.0)
        for k in ("home", "draw", "away")
    }
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    out["1x2"] = blended
    return out, meta


def selection_filter(
    candidates: list[Candidate],
    cfg: dict[str, Any] | None,
    league: str | None,
) -> tuple[list[Candidate], list[str]]:
    """Phase 2.1: config-driven market/league eligibility filter.

    A pick is a pick target only when its market is in ``selection.markets``
    AND the league passes the selection rules. Big-5 1X2 is calibration /
    sanity-check ONLY -- never an actionable pick target (the 1X2
    probability is still computed and logged for calibration; the candidate
    is simply not eligible to drive a recommendation). All rules are read
    from config (``models.decision.selection``), never hardcoded:

      selection.markets        e.g. ["Total", "BTTS", "Asian Handicap", "1X2"]
      selection.leagues_primary e.g. ["Eredivisie", "EFL Championship", ...]
      selection.one_x2_leagues  leagues where 1X2 IS an eligible target
                               (default: ``leagues_primary``; big-5 absent)

    Returns (eligible, reasons). An empty eligible list means no actionable
    pick is possible for this match -- callers fall back to NO BET.
    """
    sel = ((cfg or {}).get("models") or {}).get("decision") or {}
    sel = sel.get("selection") or {}
    markets = sel.get("markets")
    primary = {str(x) for x in (sel.get("leagues_primary") or [])}
    one_x2 = {str(x) for x in (sel.get("one_x2_leagues") or [])}
    if not one_x2:
        one_x2 = primary
    # No selection config at all -> no filtering (existing behavior).
    if markets is None and not one_x2:
        return list(candidates), []

    eligible: list[Candidate] = []
    reasons: list[str] = []
    for c in candidates:
        if markets is not None and c.market not in markets:
            reasons.append(f"market {c.market} tidak eligible (selection.markets)")
            continue
        if c.market == "1X2" and league and league not in one_x2:
            reasons.append(
                f"1X2 di {league} = sanity-check/calibration only — bukan target actionable"
            )
            continue
        eligible.append(c)
    return eligible, reasons


def margin_free_implied(odds: dict[str, float]) -> dict[str, float] | None:
    """Margin-free 1X2 implied probabilities (remove overround)."""
    raw = {k: (1.0 / v if v and v > 1.0 else 0.0) for k, v in odds.items()}
    total = sum(raw.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in raw.items()}


def fair_pair_implied(odds_a: float, odds_b: float) -> tuple[float, float] | None:
    """Margin-free implied for a two-outcome pair (Over/Under, BTTS)."""
    if odds_a <= 0 or odds_b <= 0:
        return None
    ia, ib = 1.0 / odds_a, 1.0 / odds_b
    total = ia + ib
    if total <= 0:
        return None
    return ia / total, ib / total


def excess_probability(p: float, n_outcomes: int) -> float:
    """How far p is above the uniform baseline, scaled to 0..1.

    1X2: p=1/3 -> 0.0, p=1.0 -> 1.0. Pairs: p=0.5 -> 0.0, p=1.0 -> 1.0.
    Rewards magnitude without favouring favourites (uniform is the zero).
    """
    if n_outcomes <= 1:
        return 0.0
    base = 1.0 / n_outcomes
    return max(0.0, min(1.0, (p - base) / (1.0 - base)))


@dataclass(eq=False)
class Candidate:
    """One bettable selection with its independent model probability.

    ``eq=False`` keeps identity-based hashing so Candidate objects can be
    used as set members (``value_allowed``) despite mutable fields.
    """

    market: str
    selection: str
    model_prob: float
    market_odds: float
    implied_prob: float          # margin-free
    edge_pp: float               # (model_prob - implied) * 100
    ev: float                    # model_prob * market_odds - 1
    independent: bool = True     # False for odds-derived (market mirror)

    # per-candidate scoring output (filled by score_candidate)
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    edge_level: str = "none"     # none | warning | extreme
    # Correction-spec gates (filled by decide when gating is active)
    pick_status: str = ""
    blocked_reasons: list[str] = field(default_factory=list)
    n_bucket: int | None = None


def build_candidates(
    *,
    model_probs: dict[str, Any],
    consensus_odds: dict[str, float],
    market_totals: dict[str, dict[str, float]],
    independent: bool = True,
) -> list[Candidate]:
    """Candidates across 1X2 / totals / BTTS from the INDEPENDENT model.

    ``model_probs`` is the prediction engine's output
    (``PredictionResult.model_probs``): calibrated ensemble 1X2 and
    feature-Poisson totals/BTTS -- all independent of the market.
    """
    cands: list[Candidate] = []
    p1x2 = model_probs.get("1x2") or {}
    norm = margin_free_implied(consensus_odds) if consensus_odds else None
    if p1x2 and norm:
        for side, label in (("home", "Home Win"), ("draw", "Draw"), ("away", "Away Win")):
            p = p1x2.get(side, 0.0)
            odds = consensus_odds.get(side, 0.0)
            imp = norm.get(side, 0.0)
            if p <= 0 or odds <= 0 or imp <= 0:
                continue
            cands.append(
                Candidate(
                    market="1X2", selection=label, model_prob=p,
                    market_odds=odds, implied_prob=imp,
                    edge_pp=(p - imp) * 100.0, ev=p * odds - 1.0,
                    independent=independent,
                )
            )
    for thresh in (2.5, 3.5):
        p_over = model_probs.get(f"over_{thresh}")
        if p_over is None or p_over <= 0:
            continue
        odds_o = market_totals.get(f"Over {thresh}", {}).get("odds", 0.0)
        odds_u = market_totals.get(f"Under {thresh}", {}).get("odds", 0.0)
        fair = fair_pair_implied(odds_o, odds_u)
        if fair is None:
            continue
        imp_o, imp_u = fair
        for sel, p, odds, imp in (
            (f"Over {thresh}", p_over, odds_o, imp_o),
            (f"Under {thresh}", 1.0 - p_over, odds_u, imp_u),
        ):
            if p <= 0 or odds <= 0 or imp <= 0:
                continue
            cands.append(
                Candidate(
                    market="Total", selection=sel, model_prob=p,
                    market_odds=odds, implied_prob=imp,
                    edge_pp=(p - imp) * 100.0, ev=p * odds - 1.0,
                    independent=independent,
                )
            )
    p_btts = model_probs.get("btts_yes")
    if p_btts is not None and p_btts > 0:
        odds_yes = market_totals.get("BTTS Yes", {}).get("odds", 0.0)
        odds_no = market_totals.get("BTTS No", {}).get("odds", 0.0)
        fair = fair_pair_implied(odds_yes, odds_no)
        if fair is not None:
            imp_y, imp_n = fair
            for sel, p, odds, imp in (
                ("Yes", p_btts, odds_yes, imp_y),
                ("No", 1.0 - p_btts, odds_no, imp_n),
            ):
                if p <= 0 or odds <= 0 or imp <= 0:
                    continue
                cands.append(
                    Candidate(
                        market="BTTS", selection=sel, model_prob=p,
                        market_odds=odds, implied_prob=imp,
                        edge_pp=(p - imp) * 100.0, ev=p * odds - 1.0,
                        independent=independent,
                    )
                )
    return cands


def _separation_for(c: Candidate, pool: list[Candidate]) -> float:
    """Separation of c vs its market-mates: p - max(other probs in market)."""
    mates = [x.model_prob for x in pool if x.market == c.market and x is not c]
    if not mates:
        return 0.0
    return max(0.0, min(1.0, c.model_prob - max(mates)))


def edge_level(edge_pp: float, warning: float, extreme: float) -> str:
    if abs(edge_pp) >= extreme:
        return "extreme"
    if abs(edge_pp) >= warning:
        return "warning"
    return "none"


def score_candidate(
    c: Candidate,
    pool: list[Candidate],
    *,
    calibration_quality: float,
    calibration_samples: int,
    model_agreement: float,
    completeness: float,
    bookmakers_count: int,
    historical_reliability: float,
    weights: dict[str, float],
    edge_warning_pp: float,
    edge_extreme_pp: float,
    min_bookmakers: int,
    min_edge_pp: float = MIN_EDGE_PP,
    value_allowed: bool = True,
    ml_agreement: float | None = None,
    movement: float | None = None,
) -> dict[str, float]:
    """Per-candidate component scores (all 0..1) + weighted total.

    ``ml_agreement`` (optional, ML model vs Elo+Poisson ensemble) is added as
    an extra component only when provided; its weight comes from config
    (``models.decision.weights.ml_agreement``), default 0.0 -- so existing
    behavior is byte-identical until a weight is configured.
    """
    n_outcomes = 3 if c.market == "1X2" else 2
    sep = _separation_for(c, pool)
    prob_quality = 0.5 * excess_probability(c.model_prob, n_outcomes) + 0.5 * sep
    prob_quality = max(0.0, min(1.0, prob_quality))

    calib = calibration_quality if calibration_samples >= MIN_CALIBRATION_SAMPLES else 0.0
    calib = max(0.0, min(1.0, calib))

    agree = max(0.0, min(1.0, model_agreement))

    lvl = edge_level(c.edge_pp, edge_warning_pp, edge_extreme_pp)
    # Value requires a REAL, validated-size edge (S24/S37 tuning: a 0.0 floor
    # bet on noise and lost to the OLD rule in walk-forward).
    value = 0.0
    if value_allowed and c.independent and c.ev > 0 and c.edge_pp >= min_edge_pp:
        value = max(0.0, min(1.0, c.ev / MAX_EV_FOR_FULL_VALUE))
        if lvl == "extreme":
            value = min(value, EXTREME_VALUE_CAP)
    value = max(0.0, min(1.0, value))

    data_q = max(0.0, min(1.0, completeness))
    odds_q = min(1.0, max(0, bookmakers_count) / 12.0) if bookmakers_count >= min_bookmakers else 0.0
    hist = max(0.0, min(1.0, historical_reliability))

    comps = {
        "probability_quality": round(prob_quality, 3),
        "calibration_reliability": round(calib, 3),
        "model_agreement": round(agree, 3),
        "market_value": round(value, 3),
        "data_quality": round(data_q, 3),
        "odds_quality": round(odds_q, 3),
        "historical_reliability": round(hist, 3),
    }
    if ml_agreement is not None:
        comps["ml_agreement"] = round(max(0.0, min(1.0, ml_agreement)), 3)
    if movement is not None:
        comps["movement"] = round(max(0.0, min(1.0, movement)), 3)
    total = sum(weights.get(k, 0.0) * comps[k] for k in comps)
    # When the ML/movement components are active, renormalize by the active
    # weights so the score stays in [0,1] and the extra component only
    # redistributes weight instead of inflating it. When both are absent the
    # active-weight sum is 1.0, so existing behavior is unchanged.
    if ml_agreement is not None or movement is not None:
        active = sum(weights.get(k, 0.0) for k in comps)
        if active > 0:
            total = total / active
    return {**comps, "score": round(total, 3), "edge_level": lvl}


def _type_from_score(
    score: float,
    *,
    separation: float,
    edge_lvl: str,
    calibration_quality: float,
    strong_score: float,
    good_score: float,
    lean_score: float,
) -> str:
    if score < lean_score:
        return "NO CLEAR DECISION"
    if score >= strong_score and separation >= MIN_SEPARATION and edge_lvl != "extreme" and calibration_quality >= MIN_CALIBRATION_QUALITY:
        return "STRONG"
    if score >= good_score and edge_lvl != "extreme":
        return "GOOD"
    return "LEAN"


def decide(
    candidates: list[Candidate],
    *,
    model_agreement: float,
    calibration_quality: float,
    calibration_samples: int,
    completeness: float,
    bookmakers_count: int,
    historical_reliability: float = 0.5,
    weights: dict[str, float] | None = None,
    edge_warning_pp: float = EDGE_WARNING_PP,
    edge_extreme_pp: float = EDGE_EXTREME_PP,
    min_bookmakers: int = MIN_BOOKMAKERS,
    strong_score: float = STRONG_SCORE,
    good_score: float = GOOD_SCORE,
    lean_score: float = LEAN_SCORE,
    no_clear_max_score: float = NO_CLEAR_MAX_SCORE,
    min_edge_pp: float = MIN_EDGE_PP,
    best_prob_only: bool = False,
    # Correction-spec gates (Section 2/3). Passing ``bucket_n`` activates the
    # hard gates; omit it to keep the legacy decision behavior (tests/CLI).
    bucket_n: Callable[[float], int] | None = None,
    min_bucket_n: int = MIN_BUCKET_N,
    min_bucket_ci_halfwidth: float | None = None,
    min_ev: float = MIN_EV,
    min_completeness: float = MIN_COMPLETENESS,
    disagreement: bool = False,
    hard_cap_medium: bool = False,
    form_depth_shallow: bool = False,
    model_calibration_score: float | None = None,
    # TODO-09/10/16: reliability gates + WATCH tier + variance-aware EV.
    # Both are OPT-IN (default False) so existing behavior is byte-identical
    # until config enables them -- no silent behavior change.
    enable_watch: bool = False,
    uncertainty: float = 0.0,
    ml_agreement: float | None = None,
    movement: float | None = None,
) -> dict[str, Any]:
    """Score every candidate and produce the final decision.

    ``best_prob_only`` (validated option): credit market value ONLY to the
    favourite of each market (the most likely 1X2 side, and the favoured
    side of each Over/Under and BTTS pair). Walk-forward (EPL 2022-26)
    showed that crediting value to long-shot sides with large edges bet on
    noise and lost to the validated OLD rule; restricting value to market
    favourites keeps the engine conservative while retaining the transparent
    score, decision types and guards. A 1X2 favourite that lacks value still
    yields NO BET (never a forced long-shot pick); a favoured totals side
    with a real edge can still win (S30: most likely outcome != best
    decision).

    Returns::
      {decision_type, final_decision (Candidate|None), most_likely
       (Candidate|None), score_breakdown {top: ...}, explanation (str),
       edge_warnings [str], reasons [str]}
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    reasons: list[str] = []

    if not candidates:
        return {
            "decision_type": "NO CLEAR DECISION",
            "final_decision": None,
            "most_likely": None,
            "explanation": "Tidak ada kandidat ber-odds dengan probabilitas model.",
            "reasons": ["no candidates"],
            "edge_warnings": [],
            "score_breakdown": {},
            "evaluated": [],
            "blocked": [],
            "pick_specific_confidence": None,
            "model_calibration_score": round(calibration_quality, 3),
            "hard_cap_medium_applied": False,
            "form_depth_cap_applied": False,
        }

    # Most likely outcome = the 1X2 side with the highest calibrated model
    # probability (S22). Totals/BTTS probabilities are 2-outcome and not
    # comparable across markets, so the headline "most likely" stays the 1X2
    # view (falls back to the global max when no 1X2 candidate exists).
    one_x2 = [c for c in candidates if c.market == "1X2"]
    if one_x2:
        most_likely = max(one_x2, key=lambda c: c.model_prob)
    else:
        most_likely = max(candidates, key=lambda c: c.model_prob)

    # Favourite of each market (1X2 favourite + favoured side of each pair).
    # In best_prob_only mode only these candidates may carry market value;
    # long-shots can still win on probability quality but never on edge.
    value_allowed: set[Candidate] = set()
    if best_prob_only:
        value_allowed.add(most_likely)
        for m in {c.market for c in candidates}:
            if m == "1X2":
                continue
            mates = [c for c in candidates if c.market == m]
            if mates:
                value_allowed.add(max(mates, key=lambda c: c.model_prob))

    for c in candidates:
        comps = score_candidate(
            c, candidates,
            calibration_quality=calibration_quality,
            calibration_samples=calibration_samples,
            model_agreement=model_agreement,
            completeness=completeness,
            bookmakers_count=bookmakers_count,
            historical_reliability=historical_reliability,
            weights=weights,
            edge_warning_pp=edge_warning_pp,
            edge_extreme_pp=edge_extreme_pp,
            min_bookmakers=min_bookmakers,
            min_edge_pp=min_edge_pp,
            value_allowed=(not best_prob_only or c in value_allowed),
            ml_agreement=ml_agreement,
            movement=movement,
        )
        c.score = comps.pop("score")
        c.components = comps
        c.edge_level = comps.pop("edge_level")

    best = max(candidates, key=lambda c: c.score)
    edge_warnings: list[str] = []
    for c in candidates:
        if c.edge_level == "extreme":
            edge_warnings.append(
                f"⚠️ EXTREME EDGE ({c.selection} {c.edge_pp:+.1f}pp ≥ {edge_extreme_pp:.0f}pp): "
                "audit fixture/odds-freshness/model inputs; tidak diperlakukan sebagai value."
            )
        elif c.edge_level == "warning":
            edge_warnings.append(
                f"⚠️ Edge besar ({c.selection} {c.edge_pp:+.1f}pp ≥ {edge_warning_pp:.0f}pp): "
                "verifikasi odds freshness & fixture identity."
            )

    # ---- Correction-spec Section 2: per-candidate hard gates -------------
    # Every candidate is evaluated against EV / |edge| / n_bucket /
    # completeness. Only status == VALID may reach the Final Decision.
    gating = bucket_n is not None
    any_extreme = any(c.edge_level == "extreme" for c in candidates)
    evaluated: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for c in candidates:
        n = bucket_n(c.model_prob) if gating else None
        if gating:
            status, g_reasons = pick_status(
                ev=c.ev,
                edge_pp=c.edge_pp,
                n_bucket=n or 0,
                completeness=completeness,
                min_ev=min_ev,
                extreme_pp=edge_extreme_pp,
                min_bucket_n=min_bucket_n,
                min_completeness=min_completeness,
                min_edge_pp=min_edge_pp,
                disagreement=disagreement,
                ci_halfwidth=(
                    bucket_ci_halfwidth(c.model_prob)
                    if (gating and min_bucket_ci_halfwidth is not None)
                    else None
                ),
                max_ci_halfwidth=min_bucket_ci_halfwidth,
            )
        else:
            status, g_reasons = "VALID", []
        c.pick_status = status
        c.blocked_reasons = list(g_reasons)
        c.n_bucket = n
        row = {
            "market": c.market,
            "selection": c.selection,
            "model_prob": round(c.model_prob, 4),
            "edge_pp": round(c.edge_pp, 2),
            "ev": round(c.ev, 4),
            "n_bucket": n,
            "status": status,
            "reasons": list(g_reasons),
        }
        evaluated.append(row)
        if status != "VALID":
            blocked.append(row)

    d_type = _type_from_score(
        best.score,
        separation=_separation_for(best, candidates),
        edge_lvl=best.edge_level,
        calibration_quality=calibration_quality,
        strong_score=strong_score,
        good_score=good_score,
        lean_score=lean_score,
    )

    # TODO-10: variance-aware EV. When the ensemble spread is provided, the
    # best candidate's EV is reported as a band [ev_low, ev_high] instead of
    # a point estimate -- a 4% edge at 5% spread is not the same as one at
    # 30% spread. Reported only, and used by the WATCH gate below.
    ev_band: dict[str, float] | None = None
    if uncertainty and uncertainty > 0:
        plo = max(0.0, best.model_prob - uncertainty)
        phi = min(1.0, best.model_prob + uncertainty)
        ev_band = {
            "ev_low": round(plo * best.market_odds - 1.0, 4),
            "ev_high": round(phi * best.market_odds - 1.0, 4),
            "ev": round(best.ev, 4),
            "uncertainty": round(uncertainty, 4),
        }

    # Hard NO CLEAR DECISION guards (S27): never force a pick on broken
    # inputs (high disagreement, thin data, no separation from noise).
    hard_guard = (
        model_agreement < MIN_AGREEMENT
        or completeness < MIN_COMPLETENESS
        or best.score < no_clear_max_score
    )
    if model_agreement < MIN_AGREEMENT:
        reasons.append(f"model agreement {model_agreement:.2f} < {MIN_AGREEMENT:.2f}")
    if completeness < MIN_COMPLETENESS:
        reasons.append(f"data completeness {completeness:.2f} < {MIN_COMPLETENESS:.2f}")
    if best.score < no_clear_max_score:
        reasons.append(f"decision score terbaik {best.score:.2f} < {no_clear_max_score:.2f}")

    # Section 4: an extreme edge on ANY market propagates to every other
    # λ-shared market — none may be labeled GOOD.
    if gating and any_extreme and d_type in ("STRONG", "GOOD"):
        d_type = "LEAN"
        reasons.append(
            "λ-shared EXTREME EDGE — tidak ada market turunan (totals/BTTS) boleh GOOD"
        )

    if hard_guard:
        # TODO-16 WATCH: when a candidate carries real positive value but a
        # reliability gate fails, the honest output is WATCH (positive edge,
        # insufficient reliability to BET) -- never a forced pick and never
        # a discarded opportunity.
        if (
            enable_watch
            and d_type in ("STRONG", "GOOD", "LEAN")
            and best.components.get("market_value", 0.0) > 0
        ):
            d_type = "WATCH"
            reasons.append(
                "reliability gate gagal tapi ada value positif -> WATCH (bukan bet)"
            )
        else:
            d_type = "NO CLEAR DECISION"
    else:
        # Unvalidated calibration caps the type at LEAN (never STRONG/GOOD
        # without historical validation; S14).
        calib_unvalidated = (
            calibration_samples < MIN_CALIBRATION_SAMPLES
            or calibration_quality < MIN_CALIBRATION_QUALITY
        )
        if calib_unvalidated and d_type in ("STRONG", "GOOD"):
            d_type = "LEAN"
            reasons.append(
                f"kalibrasi belum tervalidasi (sampel {calibration_samples})" 
                "— confidence diturunkan ke LEAN"
            )
        # P1 (form-depth floor): a 1-2 match form window is noise, not
        # signal — STRONG is never justified on such thin input.
        if form_depth_shallow and d_type == "STRONG":
            d_type = "GOOD"
            reasons.append(
                "form window terlalu dangkal (< 3 match/tim) — keputusan max GOOD"
            )
        # TODO-16: post-scoring reliability gates that downgrade a would-be
        # bet to WATCH when the estimate is too unstable (high ensemble
        # spread) or the market too thin to bet into.
        if enable_watch and d_type in ("STRONG", "GOOD", "LEAN"):
            # WATCH requires real positive value: a zero-EV candidate with a
            # high spread is NO BET (no value to watch), not WATCH.
            if (
                uncertainty
                and uncertainty > WATCH_UNCERTAINTY_MAX
                and best.components.get("market_value", 0.0) > 0
            ):
                d_type = "WATCH"
                reasons.append(
                    f"ensemble spread {uncertainty:.2f} > "
                    f"{WATCH_UNCERTAINTY_MAX:.2f} — probabilitas terlalu tidak pasti untuk BET"
                )
            elif best.components.get("market_value", 0.0) > 0 and best.components.get("odds_quality", 0.0) <= 0:
                d_type = "WATCH"
                reasons.append(
                    "likuiditas/bookmaker terlalu tipis untuk BET — value positif tapi tak bisa dieksekusi"
                )

        # NO BET: a decision exists but no candidate carries positive value
        # after risk adjustment (S27/S19: low odds != safe; most likely != best).
        if d_type in ("STRONG", "GOOD", "LEAN"):
            if best.components.get("market_value", 0.0) <= 0:
                d_type = "NO BET"
                if not best.independent:
                    reasons.append("model tidak independen dari odds (cermin pasar) — tanpa value")
                elif best.edge_pp < min_edge_pp:
                    # Same candidate object for prob and price so the
                    # explanation never mixes two candidates (review fix).
                    reasons.append(
                        f"{best.selection} ({best.model_prob:.1%}) tidak terkompensasi harga "
                        f"(edge {best.edge_pp:+.1f}pp < {min_edge_pp:.0f}pp / EV {best.ev:+.0%})"
                    )
                elif best_prob_only and best is not most_likely:
                    # Long-shot with a big edge: walk-forward proved these are
                    # noise (S17/S19: big edge != value). Honest explanation.
                    reasons.append(
                        f"{best.selection} ({best.model_prob:.1%}, edge {best.edge_pp:+.1f}pp) "
                        "bukan favorit market — edge besar pada long-shot tidak dikreditkan "
                        "(validasi walk-forward: noise)"
                    )
                else:
                    reasons.append(
                        f"{best.selection} ({best.model_prob:.1%}) tanpa value "
                        f"(EV {best.ev:+.0%})"
                    )

    # Section 2: a candidate failing ANY gate is excluded from the Final
    # Decision and reported only under ``blocked`` with the failed condition.
    if gating and best.pick_status != "VALID":
        if best.pick_status in ("INSUFFICIENT_DATA", "INSUFFICIENT_SAMPLE"):
            d_type = "NO CLEAR DECISION"
        else:  # AUDIT_REQUIRED / REVIEW_REQUIRED / NO VALUE
            d_type = "NO BET"
        if best.blocked_reasons:
            reasons.append(f"{best.selection}: {best.pick_status} — {best.blocked_reasons[0]}")

    # Section 3: multiplicative pick-specific confidence (reported alongside
    # the global model_calibration_score). Computed whenever gating is active
    # or a hard cap (Section 5) must be shown.
    conf_report: dict[str, Any] | None = None
    hard_cap_applied = False
    if gating or hard_cap_medium or form_depth_shallow:
        conf_report = pick_confidence(
            calibration_score=(
                model_calibration_score if model_calibration_score is not None else calibration_quality
            ),
            n_bucket=(best.n_bucket if gating else 0) or 0,
            completeness=completeness,
            extreme_edge=any_extreme,
            disagreement=disagreement,
            min_bucket_n=min_bucket_n,
            min_completeness=min_completeness,
        )
        if hard_cap_medium:
            # Addendum v1.1 Section 4: a cap is a CEILING. min_tier() can only
            # lower the tier (or leave it unchanged); the value follows the cap
            # so it never contradicts the label. The cap is always recorded in
            # caps_applied, even when it changes nothing.
            conf_report = dict(conf_report)
            conf_report["label"] = min_tier(conf_report["label"], "MEDIUM")
            conf_report["value"] = min(conf_report["value"], TIER_VALUE["MEDIUM"])
            conf_report["caps"] = conf_report.get("caps", []) + [
                "H2H & xG keduanya absen (tidak tersubstitusi Elo) -> max MEDIUM"
            ]
            hard_cap_applied = True
        if form_depth_shallow:
            # P1: a shallow form window (< MIN_FORM_DEPTH matches per team)
            # is noise, not signal — same ceiling mechanics as hard_cap_medium.
            conf_report = dict(conf_report)
            conf_report["label"] = min_tier(conf_report["label"], "MEDIUM")
            conf_report["value"] = min(conf_report["value"], TIER_VALUE["MEDIUM"])
            conf_report["caps"] = conf_report.get("caps", []) + [
                "form < 3 match per tim -> max MEDIUM"
            ]
            hard_cap_applied = True

    explanation = _explain(best, most_likely, d_type, reasons)
    return {
        "decision_type": d_type,
        "final_decision": best if d_type in ("STRONG", "GOOD", "LEAN", "WATCH") else None,
        "most_likely": most_likely,
        "score_breakdown": {
            "top": {
                "selection": best.selection,
                "market": best.market,
                "score": best.score,
                "components": best.components,
                "model_prob": round(best.model_prob, 4),
                "edge_pp": round(best.edge_pp, 2),
                "ev": round(best.ev, 4),
                "ev_band": ev_band,
                "pick_status": best.pick_status or "VALID",
                "n_bucket": best.n_bucket,
            }
        },
        "ev_band": ev_band,
        "explanation": explanation,
        "reasons": reasons,
        "edge_warnings": edge_warnings,
        "evaluated": evaluated,
        "blocked": blocked,
        "pick_specific_confidence": conf_report,
        "model_calibration_score": round(
            model_calibration_score if model_calibration_score is not None else calibration_quality, 3
        ),
        "hard_cap_medium_applied": hard_cap_applied,
        "form_depth_cap_applied": bool(form_depth_shallow),
        "enable_watch": enable_watch,
    }


def decision_to_dict(d: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe conversion of a ``decide()`` result (Candidate -> dict).

    The runner emits the analyse payload through json.dumps, so the decision
    must never carry dataclass objects.
    """
    def _c(c: Candidate | dict[str, Any] | None) -> dict[str, Any] | None:
        if c is None:
            return None
        if isinstance(c, dict):
            # MARKET PRIOR carries a plain-dict most_likely (no Candidate
            # object behind it) -- pass it through unchanged.
            return dict(c)
        return {
            "market": c.market,
            "selection": c.selection,
            "model_prob": round(c.model_prob, 4),
            "market_odds": round(c.market_odds, 4),
            "implied_prob": round(c.implied_prob, 4),
            "edge_pp": round(c.edge_pp, 2),
            "ev": round(c.ev, 4),
            "independent": c.independent,
            "score": c.score,
            "components": c.components,
            "edge_level": c.edge_level,
        }

    return {
        "decision_type": d["decision_type"],
        "final_decision": _c(d.get("final_decision")),
        "most_likely": _c(d.get("most_likely")),
        "explanation": d.get("explanation", ""),
        "reasons": d.get("reasons", []),
        "edge_warnings": d.get("edge_warnings", []),
        "score_breakdown": d.get("score_breakdown", {}),
        "ev_band": d.get("ev_band"),
        "evaluated": d.get("evaluated", []),
        "blocked": d.get("blocked", []),
        "pick_specific_confidence": d.get("pick_specific_confidence"),
        "model_calibration_score": d.get("model_calibration_score"),
        "hard_cap_medium_applied": d.get("hard_cap_medium_applied", False),
        "form_depth_cap_applied": d.get("form_depth_cap_applied", False),
        "enable_watch": d.get("enable_watch", False),
        "market_prior": d.get("market_prior", False),
        "market_predictions": d.get("market_predictions"),
        "betting_advice": d.get("betting_advice"),
    }


def _explain(
    best: Candidate,
    most_likely: Candidate,
    d_type: str,
    reasons: list[str],
) -> str:
    if d_type == "WATCH":
        base = (
            f"{best.selection} ({best.model_prob:.1%}): value positif "
            f"(edge {best.edge_pp:+.1f}pp, EV {best.ev:+.0%}) tapi reliabilitas "
            f"belum cukup — WATCH (dipantau, bukan bet)."
        )
    elif d_type in ("NO CLEAR DECISION", "NO BET"):
        base = (
            f"Most likely: {most_likely.selection} ({most_likely.model_prob:.1%}) "
            f"— tapi tidak ada keputusan yang andal."
        )
    elif best.selection == most_likely.selection:
        base = (
            f"{best.selection} adalah hasil paling mungkin ({best.model_prob:.1%}) "
            f"sekaligus keputusan terbaik (skor {best.score:.2f}, edge {best.edge_pp:+.1f}pp)."
        )
    else:
        base = (
            f"Most likely: {most_likely.selection} ({most_likely.model_prob:.1%}) — "
            f"FINAL DECISION: {best.selection} (skor {best.score:.2f}). "
            f"{most_likely.selection} punya probabilitas tertinggi, tapi "
            f"{best.selection} punya kombinasi risiko-harga terbaik "
            f"(edge {best.edge_pp:+.1f}pp, EV {best.ev:+.0%})."
        )
    if reasons:
        base += " — " + "; ".join(reasons)
    return base
