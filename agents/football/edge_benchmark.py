"""Edge-benchmark labeling (Phase 2).

The bot has no sharp closing-line source (Pinnacle / Betfair exchange). Its
"edge" is therefore computed against a SOFT pre-match consensus (OddsPapi /
NowGoal / The Odds API free tier). A soft-consensus edge must never be
presented as "beating the market" — it is a comparison against a lagging,
low-information price. The label is threaded into the decision output, the
prediction-log snapshot, and the Discord render so historical edges recorded
under one benchmark are never silently mixed with a sharper one later.

Phase 0.2 (benchmark age stamping): every edge carries the timestamp of the
odds observation it was computed from (``benchmark_ts``). An edge computed
from a benchmark older than ``max_age_hours`` (default 24h) is INVALID --
consumers must treat it as ``edge: n/a`` and must not let it drive a
recommendation. ``edge_benchmark_status`` is the single explicit check.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SOFT_CONSENSUS = "soft_consensus"
SELF_CLOSE = "self_close"

DEFAULT_MAX_BENCHMARK_AGE_HOURS = 24.0

_DEFAULT_SOFT_LABEL = (
    "soft pre-match consensus (OddsPapi/NowGoal/TheOddsAPI) — bukan closing line"
)
_DEFAULT_SELF_CLOSE_LABEL = "own historical close (self-referential, bukan market)"


def edge_benchmark(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return {key, label, beats_market_claim} for the configured benchmark.

    ``beats_market_claim`` is always False today: neither a soft consensus nor
    the bot's own close is a claim of beating an efficient market. It exists
    so a future sharp source (Pinnacle/exchange) can set it True without
    rewriting consumers.
    """
    dec = ((cfg or {}).get("models") or {}).get("decision") or {}
    key = dec.get("edge_benchmark", SOFT_CONSENSUS)
    if key == SELF_CLOSE:
        return {
            "key": SELF_CLOSE,
            "label": dec.get("edge_benchmark_label", _DEFAULT_SELF_CLOSE_LABEL),
            "beats_market_claim": False,
        }
    return {
        "key": SOFT_CONSENSUS,
        "label": dec.get("edge_benchmark_label", _DEFAULT_SOFT_LABEL),
        "beats_market_claim": False,
    }


def benchmark_age_hours(benchmark_ts: str | None, now: str | None = None) -> float | None:
    """Age of the odds observation in hours, or None when unparseable."""
    if not benchmark_ts:
        return None
    try:
        cleaned = benchmark_ts[:-1] + "+00:00" if benchmark_ts.endswith("Z") else benchmark_ts
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now_dt = datetime.now(timezone.utc) if now is None else _parse_iso(now)
        if now_dt is None:
            return None
        return max(0.0, (now_dt - dt.astimezone(timezone.utc)).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return None


def _parse_iso(ts: str) -> datetime | None:
    try:
        cleaned = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def edge_benchmark_status(
    benchmark_ts: str | None,
    max_age_hours: float = DEFAULT_MAX_BENCHMARK_AGE_HOURS,
    now: str | None = None,
) -> dict[str, Any]:
    """Explicit staleness check for an edge's benchmark (Phase 0.2).

    Returns ``{age_hours, stale, reason}``. ``stale`` is True when the odds
    observation is older than ``max_age_hours`` (or unparseable with a ts
    present) -- the edge computed from it must be treated as invalid.
    No timestamp at all is NOT stale (legacy rows); it is reported as
    ``age_hours=None`` so callers decide.
    """
    age = benchmark_age_hours(benchmark_ts, now=now)
    if age is None:
        return {"age_hours": None, "stale": False, "reason": None}
    if age > max_age_hours:
        return {
            "age_hours": round(age, 2),
            "stale": True,
            "reason": f"benchmark {age:.1f}h > {max_age_hours:.0f}h — edge invalid",
        }
    return {"age_hours": round(age, 2), "stale": False, "reason": None}
