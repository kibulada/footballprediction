"""Market Intelligence Background Poll.

Extends the existing odds-poll to:
  1. Fetch timestamped odds trend from NowGoal (type=14&t=20) for steam/RLM
  2. Capture closing odds at/near kickoff for CLV tracking
  3. Run steam move detection and log results
  4. Generate alerts when steam moves are detected

This module is called by the runner after the regular odds-poll completes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clv_tracker import log_clv_closing, log_clv_settle, load_clv_entries, aggregate_clv
from .steam_detector import analyze_market_intelligence, detect_steam_moves

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEAM_ALERT_THRESHOLD = 3.0   # % move to trigger alert
CLOSING_CAPTURE_WINDOW_HOURS = 0.5  # capture closing within 30min of kickoff
CLV_LOG_FILE = "cache/football/clv_log.jsonl"
STEAM_LOG_FILE = "cache/football/steam_alerts.jsonl"


# ---------------------------------------------------------------------------
# Trend Data Fetching
# ---------------------------------------------------------------------------

async def fetch_trend_for_match(
    nowgoal_client: Any,
    fixture: dict[str, Any],
) -> dict[str, Any] | None:
    """Fetch timestamped odds trend for a single match from NowGoal.

    Returns the trend payload with bookmaker series, or None on failure.
    """
    if not nowgoal_client:
        return None
    try:
        trend = await nowgoal_client.fetch_odds_trend(fixture)
        return trend
    except Exception as exc:
        logger.warning("trend fetch failed for %s vs %s: %s",
                       fixture.get("home"), fixture.get("away"), exc)
        return None


# ---------------------------------------------------------------------------
# Steam Alert Logging
# ---------------------------------------------------------------------------

def log_steam_alert(
    match_id: str,
    home: str,
    away: str,
    league: str,
    steam_moves: list[dict[str, Any]],
    *,
    root: str | Path = ".",
) -> None:
    """Log a steam move alert for monitoring."""
    entry = {
        "event": "steam_alert",
        "ts": datetime.now(timezone.utc).isoformat(),
        "match_id": match_id,
        "home": home,
        "away": away,
        "league": league,
        "steam_moves": steam_moves,
    }
    log_path = Path(root) / STEAM_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_steam_alerts(
    *,
    root: str | Path = ".",
    hours_back: float = 24.0,
) -> list[dict[str, Any]]:
    """Load recent steam alerts."""
    log_path = Path(root) / STEAM_LOG_FILE
    if not log_path.exists():
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - (hours_back * 3600)
    alerts = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            ts = entry.get("ts")
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.timestamp() >= cutoff:
                    alerts.append(entry)
        except (json.JSONDecodeError, ValueError):
            continue
    return alerts


# ---------------------------------------------------------------------------
# Closing Odds Capture for CLV
# ---------------------------------------------------------------------------

async def capture_closing_odds(
    nowgoal_client: Any,
    fixture: dict[str, Any],
    match_id: str,
    *,
    root: str | Path = ".",
) -> dict[str, Any] | None:
    """Capture closing odds for CLV tracking.

    Called when match is within CLOSING_CAPTURE_WINDOW_HOURS of kickoff.
    """
    if not nowgoal_client:
        return None
    try:
        closing = await nowgoal_client.fetch_closing_odds(fixture)
        if closing and closing.get("home"):
            log_clv_closing(match_id, closing["home"], root=root)
            return closing
    except Exception as exc:
        logger.warning("closing odds capture failed for %s: %s", match_id, exc)
    return None


# ---------------------------------------------------------------------------
# CLV Dashboard
# ---------------------------------------------------------------------------

def clv_dashboard(
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Generate CLV dashboard data for display."""
    entries = load_clv_entries(root=root, settled_only=True)
    if not entries:
        return {
            "total": 0,
            "message": "No settled CLV data yet. Data will accumulate as matches complete.",
            "summary": [],
            "by_league": {},
            "by_market": {},
        }

    agg = aggregate_clv(entries, group_by="league")
    market_agg = aggregate_clv(entries, group_by="market")

    # Build summary lines
    summary = []
    summary.append(f"📊 **CLV Dashboard** — {agg['total']} picks analyzed")
    summary.append(f"   Average CLV: {agg['avg_clv']:+.2%}")
    summary.append(f"   Positive CLV rate: {agg['positive_rate']:.1%}")
    summary.append(f"   Positive: {agg.get('positive_count', 0)} | Negative: {agg.get('negative_count', 0)}")
    summary.append("")

    # Top leagues
    league_stats = agg.get("groups", {})
    if league_stats:
        summary.append("**By League:**")
        sorted_leagues = sorted(league_stats.items(), key=lambda x: x[1].get("avg_clv", 0), reverse=True)
        for league, stats in sorted_leagues[:8]:
            icon = "🟢" if stats["avg_clv"] > 0 else "🔴"
            summary.append(f"  {icon} {league}: CLV {stats['avg_clv']:+.2%} ({stats['n']} picks, {stats['positive_rate']:.0%} positive)")
        summary.append("")

    # By market
    market_stats = market_agg.get("groups", {})
    if market_stats:
        summary.append("**By Market:**")
        for market, stats in sorted(market_stats.items(), key=lambda x: x[1].get("avg_clv", 0), reverse=True):
            icon = "🟢" if stats["avg_clv"] > 0 else "🔴"
            summary.append(f"  {icon} {market}: CLV {stats['avg_clv']:+.2%} ({stats['n']} picks)")

    return {
        "total": agg["total"],
        "avg_clv": agg["avg_clv"],
        "positive_rate": agg["positive_rate"],
        "summary": summary,
        "by_league": league_stats,
        "by_market": market_stats,
    }


# ---------------------------------------------------------------------------
# Steam Dashboard
# ---------------------------------------------------------------------------

def steam_dashboard(
    *,
    root: str | Path = ".",
    hours_back: float = 24.0,
) -> dict[str, Any]:
    """Generate steam move dashboard data."""
    alerts = load_steam_alerts(root=root, hours_back=hours_back)

    lines = []
    lines.append(f"🚨 **Steam Move Alerts** — Last {hours_back:.0f}h")
    lines.append(f"   Total alerts: {len(alerts)}")
    lines.append("")

    if alerts:
        for alert in alerts[-10:]:  # Show last 10
            ts = alert.get("ts", "?")[:19]
            home = alert.get("home", "?")
            away = alert.get("away", "?")
            moves = alert.get("steam_moves", [])
            for mv in moves:
                side = mv.get("side", "?")
                mag = mv.get("magnitude_pct", 0)
                lines.append(f"  🔴 {ts} {home} vs {away}: {side} {mag:+.1f}%")
    else:
        lines.append("   No steam moves detected in this period.")

    return {
        "total": len(alerts),
        "alerts": alerts,
        "summary": lines,
    }


# ---------------------------------------------------------------------------
# Main Poll Integration
# ---------------------------------------------------------------------------

async def run_market_intel_poll(
    nowgoal_client: Any,
    unsettled_matches: list[dict[str, Any]],
    *,
    root: str | Path = ".",
    capture_closing: bool = True,
    detect_steam: bool = True,
) -> dict[str, Any]:
    """Run market intelligence poll for all unsettled matches.

    This extends the regular odds-poll with:
    1. Trend data fetching for steam/RLM detection
    2. Closing odds capture for CLV tracking
    3. Steam alert generation
    """
    results = {
        "trend_fetched": 0,
        "closing_captured": 0,
        "steam_alerts": 0,
    }

    now = datetime.now(timezone.utc)

    for match in unsettled_matches:
        match_id = match.get("match_id", "")
        home = match.get("home", "")
        away = match.get("away", "")
        league = match.get("league", "")
        kickoff_str = match.get("kickoff", "")

        if not match_id or not home or not away:
            continue

        # Parse kickoff
        kickoff = None
        if kickoff_str:
            try:
                kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        fixture = {"match_id": match_id, "home": home, "away": away, "kickoff": kickoff_str}

        # 1. Fetch trend data for steam detection
        if detect_steam and nowgoal_client:
            trend = await fetch_trend_for_match(nowgoal_client, fixture)
            if trend and trend.get("bookmakers"):
                results["trend_fetched"] += 1

                # Run steam detection
                for bm in trend["bookmakers"]:
                    h2h = bm.get("h2h") or []
                    steam_moves = detect_steam_moves(h2h)
                    if steam_moves:
                        log_steam_alert(match_id, home, away, league, steam_moves, root=root)
                        results["steam_alerts"] += 1
                        logger.info("STEAM ALERT: %s vs %s — %s", home, away,
                                    [(m["side"], m["magnitude_pct"]) for m in steam_moves])

        # 2. Capture closing odds for CLV (within window of kickoff)
        if capture_closing and kickoff and nowgoal_client:
            hours_to_ko = (kickoff - now).total_seconds() / 3600.0
            if -0.5 <= hours_to_ko <= CLOSING_CAPTURE_WINDOW_HOURS:
                closing = await capture_closing_odds(nowgoal_client, fixture, match_id, root=root)
                if closing:
                    results["closing_captured"] += 1

    return results
