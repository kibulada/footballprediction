"""Movement signal from the hourly odds-snapshot series (Plan B).

Reads the chronological ``odds_snapshot`` rows captured by the background
poll loop and turns the price curve into a decision input:

  - drift     : first-price -> last-price move per side (positive = shortened)
  - steam_side: the side money is on (largest sustained shortening), or None
  - agreement : 1.0 when steam_side == model_side, 0.0 when opposite, 0.5 when
                there is no move or no model side

Betting rule (enforced in run_decision_engine when
``models.movement.require_movement_agreement``): a side may only be bet when
the moving money agrees with the model. Model says home, line drifting to
away -> NO BET.

``movement_accuracy`` reports, over settled matches, whether the steam side
actually won more often than its implied probability — the proof that the
signal carries information.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prediction_log import _read_lines

SIDES = ("home", "draw", "away")


def _timing_hours(timing: str | None) -> float | None:
    """Map a snapshot timing label to hours before kickoff, else None."""
    if not timing:
        return None
    t = str(timing).strip().upper()
    for label, hours in (
        ("T-0H", 0.0),
        ("T-15M", 0.25),
        ("T-1H", 1.0),
        ("T-6H", 6.0),
        ("T-24H", 24.0),
    ):
        if t == label:
            return hours
    m = re.match(r"^T-(\d+(?:\.\d+)?)([HM])$", t)
    if m:
        v = float(m.group(1))
        return v if m.group(2) == "H" else v / 60.0
    return None


def _ts_before_kickoff(ts: str | None, kickoff: str | None) -> bool:
    """True when a snapshot timestamp is strictly before kickoff.

    Stale-guard: rows captured at/after kickoff are in-play prices (the
    odds-poll labels these ``T-0h`` and can capture the live ``r`` leg inside
    the live window) and must never be the drift's first/last point. An
    unparseable ts/kickoff cannot disprove pre-match -> keep (same "cannot
    disprove" rule as G5).
    """
    if not ts or not kickoff:
        return True
    try:
        kd = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        if kd.tzinfo is None:
            kd = kd.replace(tzinfo=timezone.utc)
        pd = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if pd.tzinfo is None:
            pd = pd.replace(tzinfo=timezone.utc)
        return pd < kd
    except (ValueError, TypeError):
        return True


def movement_signal(
    snapshots: list[dict[str, Any]],
    *,
    model_side: str | None = None,
    min_snapshots: int = 3,
    steam_threshold_pct: float = 2.0,
    time_decay_tau: float | None = None,
    kickoff: str | None = None,
) -> dict[str, Any]:
    """Compute the movement signal for one match's odds snapshots.

    ``snapshots`` are ``odds_snapshot`` rows (chronological by ``ts``). At
    least two valid price points are needed; ``usable`` is False until
    ``min_snapshots`` rows exist.

    ``time_decay_tau`` (hours) weights each consecutive price move by
    ``exp(-hours_before_kickoff / tau)`` so late moves (T-1h / T-15m / T-0h)
    dominate the drift -- the "sharp money lands near kickoff" signal. None
    (default) keeps the plain first->last drift.

    ``kickoff`` (ISO) gates the series: any row captured at/after kickoff is
    an in-play price and is EXCLUDED, so a live ``r``-leg capture can never
    masquerade as the drift's last pre-match point.
    """
    snaps = sorted(snapshots or [], key=lambda r: r.get("ts") or "")
    if kickoff:
        snaps = [r for r in snaps if _ts_before_kickoff(r.get("ts"), kickoff)]
    first: dict[str, float] | None = None
    last: dict[str, float] | None = None
    for s in snaps:
        o = s.get("odds_1x2") or {}
        if all(o.get(k) and o[k] > 1.0 for k in SIDES):
            if first is None:
                first = {k: float(o[k]) for k in SIDES}
            last = {k: float(o[k]) for k in SIDES}
    if first is None or last is None or first == last:
        return {
            "usable": False, "n": len(snaps), "drift_pct": {},
            "steam_side": None, "steam_pct": None, "agreement": 0.5,
            "reason": "kurang dua titik harga valid",
        }
    drift = {k: (first[k] - last[k]) / first[k] * 100.0 for k in SIDES}
    drift = {k: round(v, 2) for k, v in drift.items()}
    if time_decay_tau is not None and time_decay_tau > 0:
        points: list[tuple[float, dict[str, float]]] = []
        for s in snaps:
            o = s.get("odds_1x2") or {}
            if all(o.get(k) and o[k] > 1.0 for k in SIDES):
                h = _timing_hours(s.get("timing"))
                if h is not None:
                    points.append((h, {k: float(o[k]) for k in SIDES}))
        if len(points) >= 2:
            acc = {k: 0.0 for k in SIDES}
            wsum = 0.0
            for (h0, p0), (h1, p1) in zip(points, points[1:]):
                w = math.exp(-h1 / time_decay_tau)
                for k in SIDES:
                    if p0[k] > 0:
                        acc[k] += ((p0[k] - p1[k]) / p0[k] * 100.0) * w
                wsum += w
            if wsum > 0:
                drift = {k: round(acc[k] / wsum, 2) for k in SIDES}
    if len(snaps) < min_snapshots:
        return {
            "usable": False, "n": len(snaps), "drift_pct": drift,
            "steam_side": None, "steam_pct": None, "agreement": 0.5,
            "reason": f"snapshot {len(snaps)} < {min_snapshots}",
        }
    steam_side = max(SIDES, key=lambda k: drift[k])
    steam_pct = drift[steam_side]
    if steam_pct < steam_threshold_pct:
        steam_side = None
        steam_pct = None
    if steam_side is None:
        agreement = 0.5
    elif model_side is not None and steam_side == model_side:
        agreement = 1.0
    elif model_side is not None:
        agreement = 0.0
    else:
        agreement = 0.5
    return {
        "usable": True, "n": len(snaps), "drift_pct": drift,
        "steam_side": steam_side, "steam_pct": steam_pct,
        "agreement": agreement, "reason": None,
    }


def movement_accuracy(
    path: str | Path,
    *,
    min_snapshots: int = 3,
    steam_threshold_pct: float = 2.0,
) -> dict[str, Any]:
    """Realized hit rate of the steam side over settled matches.

    For every settled snapshot with enough odds snapshots, the steam side is
    compared against the realised outcome. If the steam side wins more often
    than its first-observed implied probability, the movement signal carries
    information.
    """
    rows = _read_lines(Path(path))
    settlements = {r["match_id"]: r for r in rows if r.get("event") == "settle"}
    by_match: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("event") == "odds_snapshot":
            by_match.setdefault(r["match_id"], []).append(r)

    n = hits = 0
    implied_sum = 0.0
    for r in rows:
        if r.get("event") != "snapshot":
            continue
        st = settlements.get(r.get("match_id"))
        if st is None:
            continue
        snaps = by_match.get(r.get("match_id"), [])
        sig = movement_signal(
            snaps, min_snapshots=min_snapshots,
            steam_threshold_pct=steam_threshold_pct,
            kickoff=r.get("kickoff"),
        )
        if not sig["usable"] or sig["steam_side"] is None:
            continue
        side = sig["steam_side"]
        outcome = st.get("outcome", "")
        if outcome not in SIDES or side not in SIDES:
            continue
        first = next(
            (s.get("odds_1x2") or {}).get(side)
            for s in sorted(snaps, key=lambda x: x.get("ts") or "")
            if (s.get("odds_1x2") or {}).get(side)
        ) or 0.0
        n += 1
        if side == outcome:
            hits += 1
        if first > 1.0:
            implied_sum += 1.0 / first
    return {
        "n": n,
        "steam_hit_rate": round(hits / n, 4) if n else None,
        "steam_implied": round(implied_sum / n, 4) if n else None,
        "min_snapshots": min_snapshots,
        "steam_threshold_pct": steam_threshold_pct,
    }
