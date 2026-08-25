"""Correction-spec gates: model separation, disagreement, pick validity, confidence.

Implements the governing directive for the prediction pipeline:

  Section 1  — Model A (odds-implied) is reference-only and may NEVER compute
               edge/EV or appear in Top Pick / Final Decision. Model B
               (independent Elo+Poisson+calibration) is the only source of
               edge/EV/picks. A disagreement check |A.home - B.home| > 15pp
               flags MODEL_DISAGREEMENT (REVIEW_REQUIRED, confidence <= MEDIUM).
  Section 2  — A selection is VALID only when ALL gates hold simultaneously:
               EV > +min_ev, |edge| < extreme_pp, n_bucket >= min_bucket_n,
               data_completeness >= min_completeness. Failures produce an
               explicit status instead of a pick.
  Section 3  — Confidence is MULTIPLICATIVE: calibration_score * sample_factor
               * completeness_factor * extreme_edge_penalty *
               disagreement_penalty. model_calibration_score (global) and
               pick_specific_confidence (local) are reported separately.
  Section 8  — Invariants enforced programmatically (assert + min() caps):
               1. EV <= 0 never under Top Pick / Final Decision.
               2. HIGH impossible when n_bucket < min_bucket_n.
               3. HIGH impossible when completeness < min_completeness.
               4. MODEL_DISAGREEMENT -> confidence cannot exceed MEDIUM.
               6. Model A never described with edge/value/EV wording (the
                  probabilities dict simply has no such keys).

All functions are PURE (no I/O, no global state) so they are trivially
testable; the only I/O (loading the empirical bucket table) lives in
``bucket_n`` / ``load_bucket_table``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .predictor import (
    normalize_odds,
    prob_1x2,
    prob_btts,
    prob_over,
    score_matrix,
    solve_lambdas,
)

# Empirical per-probability buckets (PHASE 5-6 audit edges). [0.00,0.50) is
# the underdog sweep; the fine grid 0.50-0.80 covers the playable range.
PHASE6_EDGES: tuple[float, ...] = (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0)

# Confidence tier ordering + numeric ceilings (addendum v1.1). A cap is a
# CEILING: final_tier = min(computed_tier, cap_tier) — a cap can only ever
# LOWER the displayed tier, never raise it. TIER_VALUE is the maximum value a
# tier may carry after its cap is applied (used so the value never contradicts
# the label, e.g. min(0.9, 0.69) -> MEDIUM).
TIER_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
TIER_VALUE: dict[str, float] = {"LOW": 0.49, "MEDIUM": 0.69, "HIGH": 1.0}


def tier_for_value(value: float) -> str:
    """Map a 0..1 score to its confidence tier."""
    return "HIGH" if value >= 0.7 else "MEDIUM" if value >= 0.5 else "LOW"


def min_tier(a: str, b: str) -> str:
    """Tier-wise minimum (LOW < MEDIUM < HIGH) — the addendum's cap operator."""
    return a if TIER_ORDER.get(a, 0) <= TIER_ORDER.get(b, 0) else b


# Addendum v1.1 Section 5: the ONLY fields the user-facing confidence block may
# carry. Anything else must be rejected at serialization time — no signal, no
# decisiveness, no legacy 0-1 confidence, no second tier string.
CONFIDENCE_ALLOWLIST = frozenset({
    "model_calibration_score",
    "pick_specific_confidence",
    "tier",
    "tier_before_caps",
    "caps_applied",
    "n_bucket",
    "completeness_factor",
    "clv_gate",
})


def build_confidence_block(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    """Addendum v1.1 Section 5: build the strict-allowlist confidence block.

    The block is constructed ONLY from the decision engine's
    ``pick_specific_confidence`` report, so fields outside
    ``CONFIDENCE_ALLOWLIST`` are impossible by construction; any drift raises
    ValueError (structural enforcement, not convention). Returns None when
    there is no pick-specific confidence to report (no gating / no decision),
    in which case the user-facing output must show NO confidence line at all.
    """
    if not decision:
        return None
    psc = decision.get("pick_specific_confidence")
    if not psc:
        return None
    block = {
        "model_calibration_score": round(
            float(decision.get("model_calibration_score") or 0.0), 3
        ),
        "pick_specific_confidence": round(float(psc.get("value") or 0.0), 3),
        "tier": str(psc.get("label") or "LOW"),
        "tier_before_caps": str(psc.get("tier_before_caps") or psc.get("label") or "LOW"),
        "caps_applied": [str(c) for c in (psc.get("caps") or [])],
        "n_bucket": int(psc.get("n_bucket") or 0),
        "completeness_factor": round(float(psc.get("completeness_factor") or 0.0), 3),
        "clv_gate": decision.get("clv_gate"),
    }
    extra = set(block) - CONFIDENCE_ALLOWLIST
    if extra:
        raise ValueError(f"confidence block has fields outside allowlist: {sorted(extra)}")
    return block

# Spec thresholds (Section 1/2/3). Config overrides come from
# config/football.json -> models.decision and are passed in by the caller.
DISAGREEMENT_THRESHOLD_PP = 15.0
MIN_EV = 0.03            # EV must exceed +3% (explicit, > 0)
MIN_EDGE_PP = 3.0        # |model - market| edge floor (pp) for a bettable pick
EXTREME_EDGE_PP = 20.0   # |edge| >= 20pp -> AUDIT_REQUIRED, never value
MIN_BUCKET_N = 30        # hard floor for a pick's calibration bucket sample
MIN_COMPLETENESS = 0.6   # below this no Final Decision / pick
MIN_FORM_DEPTH = 3       # form window shorter than this (per team) is noise,
                         # not signal: confidence max MEDIUM, STRONG banned


def form_depth_shallow(
    home_form: dict[str, Any] | None,
    away_form: dict[str, Any] | None,
    min_depth: int = MIN_FORM_DEPTH,
) -> bool:
    """True when either team's form window is shallower than ``min_depth``.

    A 1-2 match form window carries no reliable signal (P1: overconfidence on
    thin data), so such matches must never reach HIGH confidence or a STRONG
    decision. Form depth is read from the ``sequence`` string/list (e.g.
    "W-D-L" -> 3) with ``recent_goals`` as fallback; a team with no form at
    all counts as depth 0 (shallow).
    """
    for form in (home_form, away_form):
        depth = 0
        seq = (form or {}).get("sequence")
        if isinstance(seq, str):
            depth = len([p for p in seq.split("-") if p])
        elif isinstance(seq, (list, tuple)):
            depth = len(seq)
        else:
            recent = (form or {}).get("recent_goals")
            depth = len(recent) if isinstance(recent, (list, tuple)) else 0
        if depth < min_depth:
            return True
    return False


_BUCKET_TABLE: list[dict[str, Any]] | None = None
_BUCKET_TABLE_PATH: Path | None = None


def set_bucket_table_path(path: str | Path | None) -> None:
    """Override the bucket-table source (tests inject a synthetic file)."""
    global _BUCKET_TABLE, _BUCKET_TABLE_PATH
    _BUCKET_TABLE = None
    _BUCKET_TABLE_PATH = Path(path) if path else None


def load_bucket_table() -> list[dict[str, Any]]:
    """Empirical bucket table [{lo, hi, n, predicted, actual}] loaded once.

    Prefers cache/football/calibration_buckets.json (generated from the
    PHASE 5-6 pooled walk-forward audit); falls back to the raw audit report.
    Returns [] (=> every bucket_n is 0 -> INSUFFICIENT_SAMPLE) when missing.
    """
    global _BUCKET_TABLE
    if _BUCKET_TABLE is not None:
        return _BUCKET_TABLE
    candidates: list[Path] = []
    if _BUCKET_TABLE_PATH is not None:
        candidates.append(_BUCKET_TABLE_PATH)
    root = Path(__file__).resolve().parent.parent.parent
    candidates.append(root / "cache" / "football" / "calibration_buckets.json")
    candidates.append(root / "reports" / "phase5_6_calibration_audit.json")
    table: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        raw = payload.get("buckets") or payload.get("pooled_buckets", {}).get("buckets") or []
        for row in raw:
            if "lo" in row and "hi" in row:
                table.append({
                    "lo": float(row["lo"]), "hi": float(row["hi"]),
                    "n": int(row.get("n", 0)),
                    "predicted": row.get("predicted"),
                    "actual": row.get("actual"),
                })
            elif "bucket" in row and "n" in row:
                # audit report rows carry a "[0.50,0.55)" label -> parse edges
                label = str(row["bucket"])
                try:
                    lo, hi = _parse_bucket_label(label)
                except ValueError:
                    continue
                table.append({
                    "lo": lo, "hi": hi, "n": int(row.get("n", 0)),
                    "predicted": row.get("predicted"),
                    "actual": row.get("actual"),
                })
        if table:
            break
    _BUCKET_TABLE = table
    return table


def _parse_bucket_label(label: str) -> tuple[float, float]:
    import re

    m = re.match(r"\[([\d.]+),([\d.]+)(?:\)|\])", label)
    if not m:
        raise ValueError(f"bad bucket label: {label!r}")
    return float(m.group(1)), float(m.group(2))


def bucket_n(prob: float) -> int:
    """Sample size n of the empirical bucket containing ``prob`` (0 = none)."""
    p = max(0.0, min(1.0, float(prob)))
    for row in load_bucket_table():
        lo, hi = row["lo"], row["hi"]
        if (lo <= p < hi) or (hi == 1.0 and p == 1.0):
            return int(row["n"])
    return 0


def bucket_ci_halfwidth(prob: float) -> float | None:
    """95% binomial CI half-width (in probability units) of the bucket's
    realized rate for the bucket containing ``prob``.

    Phase 4: a bucket can only certify a tier if this half-width is small
    enough that a real edge is distinguishable from sampling noise. Returns
    None when the bucket is missing or degenerate (rate at 0/1).
    """
    p = max(0.0, min(1.0, float(prob)))
    for row in load_bucket_table():
        lo, hi = row["lo"], row["hi"]
        if not ((lo <= p < hi) or (hi == 1.0 and p == 1.0)):
            continue
        n = int(row.get("n", 0))
        actual = row.get("actual")
        if n < 1 or actual is None:
            return None
        rate = max(0.0, min(1.0, float(actual)))
        if rate in (0.0, 1.0):
            return None
        return 1.96 * math.sqrt(rate * (1.0 - rate) / n)
    return None


def model_a_probs(
    consensus: dict[str, float],
    market_totals: dict[str, dict[str, float]],
    xg_lambda: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Model A — odds-implied probabilities ONLY (no edge/EV keys by design).

    λ is solved FROM the market odds, so every probability mirrors the market
    it is compared to. Section 1: reference only — used for cross-market
    consistency checks and the disagreement check, never for picks. Returns
    None when the consensus is degenerate.
    """
    if not consensus or consensus.get("home", 0) <= 0:
        return None
    norm = normalize_odds({
        "home": 1.0 / consensus["home"] if consensus.get("home", 0) > 0 else 0,
        "draw": 1.0 / consensus["draw"] if consensus.get("draw", 0) > 0 else 0,
        "away": 1.0 / consensus["away"] if consensus.get("away", 0) > 0 else 0,
    })
    if xg_lambda and xg_lambda[0] > 0 and xg_lambda[1] > 0:
        lh, la = xg_lambda
        lam_src = "xg"  # pre-match xG (flashscore stats / understat history)
    else:
        lh, la = solve_lambdas(norm["home"], norm["away"], norm["draw"])
        lam_src = "odds_derived"
    matrix = score_matrix(lh, la)
    return {
        "1x2": prob_1x2(matrix),
        "over_1.5": prob_over(matrix, 1.5),
        "over_2.5": prob_over(matrix, 2.5),
        "over_3.5": prob_over(matrix, 3.5),
        "btts_yes": prob_btts(matrix),
        "lambda_home": round(lh, 3),
        "lambda_away": round(la, 3),
        "lambda_source": lam_src,
        # no "edge" / "ev" / "value" keys — Section 1/8#6
    }


def model_disagreement(
    model_a_1x2: dict[str, float] | None,
    model_b_1x2: dict[str, float] | None,
    threshold_pp: float = DISAGREEMENT_THRESHOLD_PP,
) -> dict[str, Any]:
    """Section 1 mandatory check: delta = |A.home - B.home| in percentage pts.

    Returns {delta_pp, flag, threshold_pp}. flag=True when delta > threshold.
    """
    if not model_a_1x2 or not model_b_1x2:
        return {"delta_pp": None, "flag": False, "threshold_pp": threshold_pp}
    delta_pp = abs(float(model_a_1x2.get("home", 0)) - float(model_b_1x2.get("home", 0))) * 100.0
    return {
        "delta_pp": round(delta_pp, 1),
        "flag": delta_pp > threshold_pp,
        "threshold_pp": threshold_pp,
    }


def pick_status(
    *,
    ev: float,
    edge_pp: float,
    n_bucket: int,
    completeness: float,
    min_ev: float = MIN_EV,
    extreme_pp: float = EXTREME_EDGE_PP,
    min_bucket_n: int = MIN_BUCKET_N,
    min_completeness: float = MIN_COMPLETENESS,
    min_edge_pp: float = MIN_EDGE_PP,
    disagreement: bool = False,
    ci_halfwidth: float | None = None,
    max_ci_halfwidth: float | None = None,
) -> tuple[str, list[str]]:
    """Section 2 hard gates -> (status, reasons). Status is one of:

      VALID               all gates pass -> may appear in Top Pick/Final Decision
      INSUFFICIENT_DATA   data_completeness below the floor -> no pick
      INSUFFICIENT_SAMPLE calibration bucket n below the floor -> no pick
      AUDIT_REQUIRED      |edge| >= extreme -> treated as error signal first
      REVIEW_REQUIRED     MODEL_DISAGREEMENT -> caution, no pick
      NO VALUE            EV <= min_ev -> not a value proposition

    Gate order: MODEL_DISAGREEMENT is checked FIRST (Section 1: "not optional
    and cannot be bypassed even if Model B's calibration confidence is
    otherwise high"), then the Section 2 data gates (completeness > sample >
    edge), then EV.
    """
    if disagreement:
        return "REVIEW_REQUIRED", [
            "Model A dan Model B divergen — at least one model likely miscalibrated, treat with caution"
        ]
    if completeness < min_completeness:
        return "INSUFFICIENT_DATA", [
            f"data completeness {completeness:.2f} < {min_completeness:.2f}"
        ]
    if n_bucket < min_bucket_n:
        return "INSUFFICIENT_SAMPLE", [
            f"n_bucket {n_bucket} < {min_bucket_n} (kalibrasi bucket terlalu kecil)"
        ]
    if (
        ci_halfwidth is not None
        and max_ci_halfwidth is not None
        and ci_halfwidth > max_ci_halfwidth
    ):
        return "INSUFFICIENT_SAMPLE", [
            f"bucket CI ±{ci_halfwidth:.1%} > ±{max_ci_halfwidth:.0%} "
            "(sample terlalu noisy untuk certify tier)"
        ]
    if abs(edge_pp) >= extreme_pp:
        return "AUDIT_REQUIRED", [
            f"|edge| {abs(edge_pp):.1f}pp >= {extreme_pp:.0f}pp — audit fixture/odds/model inputs"
        ]
    if ev <= min_ev:
        return "NO VALUE", [f"EV {ev:+.1%} <= +{min_ev:.0%}"]
    # Edge floor (2026-08-17, Galatasaray-Corum review): a pick whose model
    # probability merely mirrors the market (edge ~0) is NOT value even when
    # EV is marginally positive -- the "favorite at 1.30 with edge 0%" trap
    # (model 71.5% = implied 71.5%, EV -7% after margin). Require a real,
    # positive model-vs-market deviation; the candidate must be independently
    # better than the price, not just "most likely to win".
    if edge_pp < min_edge_pp:
        return "NO VALUE", [
            f"edge {edge_pp:+.1f}pp < {min_edge_pp:.0f}pp — model tidak lebih baik dari harga (bukan value)"
        ]
    return "VALID", []


def pick_confidence(
    *,
    calibration_score: float,
    n_bucket: int,
    completeness: float,
    extreme_edge: bool = False,
    disagreement: bool = False,
    min_bucket_n: int = MIN_BUCKET_N,
    min_completeness: float = MIN_COMPLETENESS,
) -> dict[str, Any]:
    """Section 3 multiplicative pick-specific confidence.

    confidence = calibration_score * min(1, n/min_bucket_n) * completeness *
                 (0.5 if extreme edge) * (0.4 if disagreement)

    Addendum v1.1 Section 4: caps (MODEL_DISAGREEMENT, low n_bucket, low
    completeness) are CEILINGS, never floors. The computed tier comes from the
    full multiplicative formula first (``tier_before_caps``), then
    ``final_tier = min(computed_tier, cap_tier)``. A cap can only lower the
    displayed tier or leave it unchanged — it must never raise it.

    Returns {value, label, tier_before_caps, caps, n_bucket, sample_factor,
    completeness_factor, extreme_edge_penalty, disagreement_penalty,
    model_calibration_score}.
    """
    caps: list[str] = []
    calib = max(0.0, min(1.0, calibration_score))
    sample_factor = min(1.0, max(0.0, (n_bucket or 0) / float(min_bucket_n)))
    completeness_factor = max(0.0, min(1.0, completeness))
    extreme_penalty = 0.5 if extreme_edge else 1.0
    disagree_penalty = 0.4 if disagreement else 1.0

    # 1) TRUE tier from the full multiplicative formula (no caps yet).
    raw_value = (
        calib * sample_factor * completeness_factor * extreme_penalty * disagree_penalty
    )
    tier_before_caps = tier_for_value(raw_value)

    # 2) Applicable caps as an upper bound only (Section 8 invariants 2-4).
    cap_tier = "HIGH"
    if n_bucket < min_bucket_n:
        cap_tier = min_tier(cap_tier, "LOW")
        caps.append(f"n_bucket {n_bucket} < {min_bucket_n} -> max LOW")
    if completeness < min_completeness:
        cap_tier = min_tier(cap_tier, "LOW")
        caps.append(f"completeness {completeness:.2f} < {min_completeness:.2f} -> max LOW")
    if disagreement:
        cap_tier = min_tier(cap_tier, "MEDIUM")
        caps.append("MODEL_DISAGREEMENT (max MEDIUM)")
    if extreme_edge:
        caps.append("EXTREME EDGE (λ-shared) -> penalti 0.5")

    # 3) final_tier = min(computed_tier, cap_tier); value follows the cap so
    # it can never contradict the label.
    label = min_tier(tier_before_caps, cap_tier)
    value = min(raw_value, TIER_VALUE[cap_tier])

    # Section 8 non-negotiable invariants — enforce, don't just document.
    assert not (label == "HIGH" and n_bucket < min_bucket_n), "invariant 2 violated"
    assert not (label == "HIGH" and completeness < min_completeness), "invariant 3 violated"
    assert not (label == "HIGH" and disagreement), "invariant 4 violated"

    return {
        "value": round(max(0.0, value), 3),
        "label": label,
        "tier_before_caps": tier_before_caps,
        "caps": caps,
        "n_bucket": int(n_bucket or 0),
        "model_calibration_score": round(calib, 3),
        "sample_factor": round(sample_factor, 3),
        "completeness_factor": round(completeness_factor, 3),
        "extreme_edge_penalty": extreme_penalty,
        "disagreement_penalty": disagree_penalty,
    }
