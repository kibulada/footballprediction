"""CLV hard gate (Phase 3 / Phase 0.3).

A segment (league x market x decision tier) may only emit an actionable
decision when it has >= ``min_bets`` settled bets AND average realized price
CLV > 0 over those bets. Positive ROI with negative CLV is variance, not
skill — such segments are demoted to NO BET / MARKET PRIOR with the reason
logged.

This is the "use the metric we already collect" gate: CLV was tracked but
ignored; now it decides whether a tier is allowed to recommend anything.

Phase 0.3: ``min_bets`` is lowered (200 -> 30, config) so the gate can open
on real evidence instead of staying permanently blocked, and a confidence-
interval requirement is added: the Wilson score interval (95%, z=1.96) on
the CLV-positivity rate must have half-width <= ``max_ci_halfwidth`` (0.05)
for the positive-CLV claim to count as statistically meaningful.

Exact Wilson interval used (standard score interval for a binomial
proportion p_hat = k/n with k successes):

    centre   = (k + z^2/2) / (n + z^2)
    half_w   = z * sqrt( p_hat*(1-p_hat)/n + z^2/(4*n^2) ) / (1 + z^2/n)

with z = 1.96 (95%). ``half_w <= 0.05`` means the evidence that CLV > 0 is
not a fluke of a tiny sample.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .prediction_log import segment_clv_stats

DEFAULT_WILSON_Z = 1.96


def wilson_interval(
    positive: int,
    n: int,
    z: float = DEFAULT_WILSON_Z,
) -> tuple[float, float] | None:
    """Wilson score interval (centre, half_width) for a binomial proportion.

    ``positive`` successes out of ``n`` trials. Returns None when n <= 0.
    """
    if n <= 0:
        return None
    p_hat = positive / n
    denom = 1.0 + z * z / n
    centre = (positive + z * z / 2.0) / (n + z * z)
    half_w = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
        / denom
    )
    return centre, half_w


def gate_segment(
    stats: dict[str, dict[str, Any]],
    *,
    league: str,
    market: str,
    tier: str,
    min_bets: int,
    require_roi_positive: bool = True,
    max_ci_halfwidth: float | None = None,
    wilson_z: float = DEFAULT_WILSON_Z,
) -> dict[str, Any]:
    """Return {allowed, n, clv_pct, roi, ci, reason} for one segment.

    ``allowed`` True only when the segment has >= min_bets settled bets AND a
    strictly positive realized price CLV AND (by default) a strictly positive
    ROI. Positive ROI with negative CLV is variance, not skill.

    ``max_ci_halfwidth`` (Phase 0.3): when provided, the Wilson score
    interval (95%) on the CLV-positivity rate must also have half-width
    <= ``max_ci_halfwidth`` -- without it a positive average CLV on a tiny
    sample is not evidence. ``ci`` = {centre, half_width, n_positive, z}.
    """
    key = f"{league}|{market}|{tier}"
    s = stats.get(key)
    if s is None:
        return {
            "allowed": False,
            "n": 0,
            "clv_pct": None,
            "roi": None,
            "ci": None,
            "reason": f"segmen {league} • {market} • {tier} belum punya settled bets",
        }
    if s["n"] < min_bets:
        return {
            "allowed": False,
            "n": s["n"],
            "clv_pct": s["price_clv_pct"],
            "roi": s["roi"],
            "ci": None,
            "reason": f"n={s['n']} < {min_bets} settled bets",
        }
    clv = s["price_clv_pct"]
    if clv is None or clv <= 0:
        return {
            "allowed": False,
            "n": s["n"],
            "clv_pct": clv,
            "roi": s["roi"],
            "ci": None,
            "reason": (
                f"CLV {clv if clv is not None else 'n/a'}% <= 0 "
                "(ROI positif + CLV negatif = variance, bukan skill)"
            ),
        }
    roi = s["roi"]
    if require_roi_positive and (roi is None or roi <= 0):
        return {
            "allowed": False,
            "n": s["n"],
            "clv_pct": clv,
            "roi": roi,
            "ci": None,
            "reason": (
                f"ROI {roi if roi is not None else 'n/a'} <= 0 "
                "(paper-trade: segmen belum lulus ROI+CLV)"
            ),
        }
    ci = None
    n_pos = int(s.get("n_positive_clv") or 0)
    if max_ci_halfwidth is not None:
        ci = wilson_interval(n_pos, s["n"], z=wilson_z)
        ci_out = {
            "centre": round(ci[0], 4) if ci else None,
            "half_width": round(ci[1], 4) if ci else None,
            "n_positive": n_pos,
            "z": wilson_z,
        }
        if ci is None or ci[1] > max_ci_halfwidth:
            return {
                "allowed": False,
                "n": s["n"],
                "clv_pct": clv,
                "roi": roi,
                "ci": ci_out,
                "reason": (
                    f"CLV evidence tidak meyakinkan: Wilson CI half-width "
                    f"{ci_out['half_width'] if ci_out['half_width'] is not None else 'n/a'} "
                    f"> {max_ci_halfwidth} (butuh lebih banyak settled bets)"
                ),
            }
    return {
        "allowed": True,
        "n": s["n"],
        "clv_pct": clv,
        "roi": roi,
        "ci": ci_out if max_ci_halfwidth is not None else None,
        "reason": None,
    }


def load_segment_stats(
    log_path: str | Path,
    *,
    edge_threshold: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """Load (and memoize) realized per-segment CLV stats from the log."""
    return segment_clv_stats(log_path, edge_threshold=edge_threshold)
