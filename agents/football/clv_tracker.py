"""Closing Line Value (CLV) Tracker.

CLV is the single most important metric for validating a betting model.
It measures whether the model's entry price is better than the closing odds
(the final odds at kickoff, which is the most efficient price).

  CLV = (closing_odds / entry_odds - 1) * 100

  Positive CLV → model beat the market (entry price better than close)
  Negative CLV → model lost to market (entry price worse than close)

A model with positive CLV over a large sample is statistically profitable,
regardless of short-term variance.

This module:
  1. Records entry price (when signal is generated)
  2. Records closing price (at kickoff or latest available)
  3. Calculates CLV per pick
  4. Aggregates CLV by league, market, confidence
  5. Provides gates: if CLV trend is negative → downgrade confidence
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLV_LOG_FILE = "cache/football/clv_log.jsonl"
CLV_MIN_SAMPLES_FOR_GATE = 20   # need 20+ settled picks to gate
CLV_NEGATIVE_THRESHOLD = -0.02  # -2% average CLV → downgrade confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_odds(value: Any) -> float | None:
    try:
        v = float(value)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLV Calculation
# ---------------------------------------------------------------------------

def calculate_clv(entry_odds: float, closing_odds: float) -> float:
    """Calculate Closing Line Value.

    Returns CLV as a decimal (e.g., 0.05 = +5% CLV).
    """
    if entry_odds <= 1.0 or closing_odds <= 1.0:
        return 0.0
    return (closing_odds / entry_odds) - 1.0


def implied_from_odds(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def closing_edge(entry_odds: float, closing_odds: float) -> float:
    """Calculate edge vs closing line in percentage points."""
    entry_imp = implied_from_odds(entry_odds)
    close_imp = implied_from_odds(closing_odds)
    return (entry_imp - close_imp) * 100.0


# ---------------------------------------------------------------------------
# CLV Logging
# ---------------------------------------------------------------------------

def log_clv_entry(
    match_id: str,
    league: str,
    market: str,
    selection: str,
    entry_odds: float,
    model_prob: float,
    confidence: str,
    best_pick: str | None = None,
    *,
    root: str | Path = ".",
) -> None:
    """Record a CLV entry when a pick is made.

    This should be called when the signal engine produces a pick, BEFORE
    the match is settled.
    """
    entry = {
        "event": "clv_entry",
        "ts": _now_iso(),
        "match_id": match_id,
        "league": league,
        "market": market,
        "selection": selection,
        "entry_odds": round(entry_odds, 4),
        "model_prob": round(model_prob, 4),
        "confidence": confidence,
        "best_pick": best_pick,
        "closing_odds": None,
        "clv": None,
        "settled": False,
    }
    log_path = Path(root) / CLV_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_clv_closing(
    match_id: str,
    closing_odds: float,
    *,
    root: str | Path = ".",
) -> None:
    """Update a CLV entry with closing odds (called at/near kickoff)."""
    log_path = Path(root) / CLV_LOG_FILE
    if not log_path.exists():
        return

    lines = log_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue

        if entry.get("match_id") == match_id and not entry.get("settled"):
            entry["closing_odds"] = round(closing_odds, 4)
            entry["clv"] = round(calculate_clv(entry["entry_odds"], closing_odds), 4)
            entry["settled"] = True
            updated = True
        new_lines.append(json.dumps(entry, ensure_ascii=False))

    if updated:
        log_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def log_clv_settle(
    match_id: str,
    outcome: str,
    *,
    root: str | Path = ".",
) -> None:
    """Mark a CLV entry as settled with the actual outcome."""
    log_path = Path(root) / CLV_LOG_FILE
    if not log_path.exists():
        return

    lines = log_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue

        if entry.get("match_id") == match_id:
            entry["outcome"] = outcome
            entry["settled"] = True
            # Recalculate CLV if closing odds available
            if entry.get("closing_odds") and entry.get("entry_odds"):
                entry["clv"] = round(
                    calculate_clv(entry["entry_odds"], entry["closing_odds"]), 4
                )
        new_lines.append(json.dumps(entry, ensure_ascii=False))

    log_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLV Aggregation & Analysis
# ---------------------------------------------------------------------------

def load_clv_entries(
    *,
    root: str | Path = ".",
    settled_only: bool = False,
) -> list[dict[str, Any]]:
    """Load all CLV entries from log."""
    log_path = Path(root) / CLV_LOG_FILE
    if not log_path.exists():
        return []

    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if settled_only and not entry.get("settled"):
                continue
            entries.append(entry)
        except json.JSONDecodeError:
            continue
    return entries


def aggregate_clv(
    entries: list[dict[str, Any]],
    *,
    group_by: str = "league",
) -> dict[str, Any]:
    """Aggregate CLV data by league, market, or confidence.

    Returns summary stats.
    """
    if not entries:
        return {"total": 0, "avg_clv": 0.0, "positive_rate": 0.0, "groups": {}}

    # Filter to entries with CLV
    valid = [e for e in entries if e.get("clv") is not None]
    if not valid:
        return {"total": 0, "avg_clv": 0.0, "positive_rate": 0.0, "groups": {}}

    # Overall stats
    clvs = [e["clv"] for e in valid]
    avg_clv = sum(clvs) / len(clvs)
    positive_count = sum(1 for c in clvs if c > 0)
    positive_rate = positive_count / len(clvs) if clvs else 0.0

    # Group by
    groups: dict[str, list[float]] = {}
    for e in valid:
        key = e.get(group_by, "unknown")
        groups.setdefault(key, []).append(e["clv"])

    group_stats = {}
    for key, gclvs in groups.items():
        group_stats[key] = {
            "n": len(gclvs),
            "avg_clv": round(sum(gclvs) / len(gclvs), 4) if gclvs else 0.0,
            "positive_rate": round(sum(1 for c in gclvs if c > 0) / len(gclvs), 4) if gclvs else 0.0,
            "min_clv": round(min(gclvs), 4) if gclvs else 0.0,
            "max_clv": round(max(gclvs), 4) if gclvs else 0.0,
        }

    return {
        "total": len(valid),
        "avg_clv": round(avg_clv, 4),
        "positive_rate": round(positive_rate, 4),
        "positive_count": positive_count,
        "negative_count": len(valid) - positive_count,
        "groups": group_stats,
    }


# ---------------------------------------------------------------------------
# CLV Gate (Confidence Downgrade)
# ---------------------------------------------------------------------------

def clv_gate(
    league: str,
    confidence: str,
    *,
    root: str | Path = ".",
    min_samples: int = CLV_MIN_SAMPLES_FOR_GATE,
    negative_threshold: float = CLV_NEGATIVE_THRESHOLD,
) -> tuple[str, dict[str, Any]]:
    """Check if CLV history warrants a confidence downgrade.

    Returns (adjusted_confidence, clv_info).
    """
    entries = load_clv_entries(root=root, settled_only=True)
    league_entries = [e for e in entries if e.get("league") == league and e.get("clv") is not None]

    info = {
        "league": league,
        "samples": len(league_entries),
        "avg_clv": None,
        "positive_rate": None,
        "downgraded": False,
        "reason": None,
    }

    if len(league_entries) < min_samples:
        return confidence, info

    clvs = [e["clv"] for e in league_entries]
    avg_clv = sum(clvs) / len(clvs)
    positive_rate = sum(1 for c in clvs if c > 0) / len(clvs)

    info["avg_clv"] = round(avg_clv, 4)
    info["positive_rate"] = round(positive_rate, 4)

    # Downgrade if CLV is consistently negative
    if avg_clv < negative_threshold:
        downgrade_map = {
            "VERY HIGH": "HIGH",
            "HIGH": "MEDIUM",
            "MEDIUM": "LOW",
            "LOW": "LOW",
        }
        new_conf = downgrade_map.get(confidence, confidence)
        if new_conf != confidence:
            info["downgraded"] = True
            info["reason"] = f"CLV {avg_clv:+.2%} negative over {len(league_entries)} picks → {confidence} → {new_conf}"
            return new_conf, info

    return confidence, info


# ---------------------------------------------------------------------------
# CLV Display
# ---------------------------------------------------------------------------

def clv_summary_display(entries: list[dict[str, Any]], *, max_entries: int = 10) -> list[str]:
    """Generate human-readable CLV summary for display."""
    lines = []
    valid = [e for e in entries if e.get("clv") is not None]
    if not valid:
        return ["No CLV data available yet."]

    agg = aggregate_clv(valid)
    lines.append(f"📊 CLV Summary: {agg['total']} picks analyzed")
    lines.append(f"   Average CLV: {agg['avg_clv']:+.2%}")
    lines.append(f"   Win rate (positive CLV): {agg['positive_rate']:.1%}")
    lines.append("")

    # Top leagues by CLV
    league_stats = agg.get("groups", {}).get("league", {})
    if league_stats:
        lines.append("By League:")
        sorted_leagues = sorted(league_stats.items(), key=lambda x: x[1].get("avg_clv", 0), reverse=True)
        for league, stats in sorted_leagues[:5]:
            icon = "🟢" if stats["avg_clv"] > 0 else "🔴"
            lines.append(f"  {icon} {league}: CLV {stats['avg_clv']:+.2%} ({stats['n']} picks, {stats['positive_rate']:.0%} positive)")

    return lines
