"""Staking layer (Phase 6).

Fractional-Kelly position sizing on the margin-free edge, with a hard
bankroll cap and a tier multiplier from the decision type. The bot is
advisory (read-only, no auto-bet): this computes the RECOMMENDED stake, never
places one.

Kelly:  f* = (p*b - (1-p))/b,  b = decimal_odds - 1,  p = model_prob.
Stake fraction = min(kelly_fraction * tier_multiplier * f*, max_stake_fraction).

Extreme edge (|edge_pp| >= edge_extreme_pp) is auto-declined: such an edge is
far more likely a data error than a genuine opportunity and must never reach
a staking calculation. Defense-in-depth — the decision engine already blocks
extreme-edge candidates (AUDIT_REQUIRED) in gated mode.
"""
from __future__ import annotations

from typing import Any

DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MAX_STAKE_FRACTION = 0.02
DEFAULT_BANKROLL = 100.0
DEFAULT_TIER_MULTIPLIER = {"STRONG": 1.0, "GOOD": 0.75, "LEAN": 0.5, "WATCH": 0.0}


def _staking_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return ((cfg or {}).get("models") or {}).get("staking") or {}


def kelly_full_stake(model_prob: float, decimal_odds: float) -> float:
    """Full-Kelly fraction of bankroll (<=0 means no bet)."""
    b = decimal_odds - 1.0
    if b <= 0 or model_prob <= 0 or model_prob >= 1:
        return 0.0
    return max(0.0, (model_prob * b - (1.0 - model_prob)) / b)


def compute_stake(
    *,
    model_prob: float,
    decimal_odds: float,
    edge_pp: float,
    decision_type: str,
    cfg: dict[str, Any] | None = None,
    edge_extreme_pp: float = 20.0,
) -> dict[str, Any]:
    """Return the recommended stake for one candidate, or a decline.

    Never raises. A decline carries ``declined=True`` + ``reason``; otherwise
    ``stake_fraction`` (of bankroll) and ``stake_amount`` (in bankroll units)
    are the recommended size.
    """
    sc = _staking_cfg(cfg)
    bankroll = float(sc.get("bankroll", DEFAULT_BANKROLL))
    kelly_frac = float(sc.get("kelly_fraction", DEFAULT_KELLY_FRACTION))
    max_frac = float(sc.get("max_stake_fraction", DEFAULT_MAX_STAKE_FRACTION))
    tier_mult = float(
        (sc.get("tier_multiplier") or DEFAULT_TIER_MULTIPLIER).get(decision_type, 0.0)
    )

    if abs(edge_pp) >= edge_extreme_pp:
        return {
            "declined": True,
            "reason": f"auto-decline: |edge| {edge_pp:+.1f}pp >= {edge_extreme_pp:.0f}pp "
            "(kemungkinan data error, bukan peluang)",
            "stake_fraction": 0.0,
            "stake_amount": 0.0,
        }

    f_full = kelly_full_stake(model_prob, decimal_odds)
    stake_fraction = min(kelly_frac * tier_mult * f_full, max_frac)
    if f_full <= 0 or stake_fraction <= 0 or tier_mult <= 0:
        return {
            "declined": True,
            "reason": (
                "auto-decline: Kelly f* <= 0 (edge belum dikompensasi harga) "
                f"atau tier '{decision_type}' tidak distake"
            ),
            "stake_fraction": 0.0,
            "stake_amount": 0.0,
        }
    return {
        "declined": False,
        "reason": None,
        "kelly_full": round(f_full, 4),
        "kelly_fraction": round(kelly_frac, 3),
        "tier_multiplier": round(tier_mult, 3),
        "stake_fraction": round(stake_fraction, 4),
        "stake_amount": round(stake_fraction * bankroll, 2),
        "bankroll": round(bankroll, 2),
    }
