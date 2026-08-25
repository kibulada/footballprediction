"""Re-runnable P4 signal-weight backtest (movement / late_movement A/B).

Reads SETTLED matches from ``prediction_log`` (each snapshot carries the full
scored ``signal_engine_ranking`` -- market, selection, edge_pp, movement,
components -- persisted by ``append_snapshot``). For every settled match the
stored ranking entries are reconstructed as ``Signal`` objects and re-scored
through the EXACT production ``score_signals`` / ``rank_and_pick`` code with a
candidate weight set; the resulting best pick is settled against the final
score via ``settle_signal`` (quarter-line AH semantics included). No mirrored
scoring logic, no re-fetching, no synthetic data.

Honesty guards:
- Only matches whose snapshot has a stored ``signal_engine_ranking`` are used.
- Below ``--min-samples`` settled matches the run reports "insufficient data"
  instead of producing a report that would overfit noise.
- Two non-overlapping periods (chronological split by kickoff) are reported
  separately so a weight set that only wins in one period is visible as such.

Usage:
    python -m agents.football.backtest_signal --log cache/football/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent

from .signal_engine import (  # noqa: E402
    Signal,
    rank_and_pick,
    score_signals,
    settle_signal,
)

MIN_SAMPLES_DEFAULT = 500  # settled matches with stored ranking required for a report


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _read_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_settled_ranked_matches(log_path: str | Path) -> list[dict[str, Any]]:
    """Join settle rows with their prediction snapshot (stored ranking).

    Returns one record per settled match that has a snapshot carrying a
    ``signal_engine_ranking``: {match_id, kickoff, home, away, home_goals,
    away_goals, completeness, ranking, pick}. Matches without a stored
    ranking (pre-P4 snapshots) are skipped -- there is nothing to re-weight.
    """
    rows = _read_lines(Path(log_path))
    snaps: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("event") == "snapshot" and r.get("signal_engine_ranking"):
            mid = r.get("match_id")
            if mid and mid not in snaps:
                snaps[mid] = r
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("event") != "settle":
            continue
        snap = snaps.get(r.get("match_id"))
        if snap is None:
            continue
        completeness = (snap.get("features") or {}).get("completeness") or 0.0
        out.append({
            "match_id": r["match_id"],
            "kickoff": snap.get("kickoff"),
            "home": snap.get("home"),
            "away": snap.get("away"),
            "home_goals": int(r.get("home_goals") or 0),
            "away_goals": int(r.get("away_goals") or 0),
            "completeness": float(completeness),
            "ranking": snap.get("signal_engine_ranking") or [],
            "pick": snap.get("signal_engine_pick"),
        })
    return out


# --------------------------------------------------------------------------
# Re-weighting (uses the exact production scoring code)
# --------------------------------------------------------------------------

def _reconstruct_signals(ranking: list[dict[str, Any]]) -> list[Signal]:
    """Rebuild Signal objects from a stored ranking (all raw fields kept)."""
    signals: list[Signal] = []
    for e in ranking:
        signals.append(Signal(
            market=e.get("market") or "",
            selection=e.get("selection") or "",
            model_prob=float(e.get("model_prob") or 0.0),
            market_odds=e.get("market_odds"),
            implied_prob=e.get("implied_prob"),
            line=e.get("line"),
            side=e.get("side"),
            line_key=e.get("line_key") or "",
            edge_pp=float(e.get("edge_pp") or 0.0),
            movement=dict(e.get("movement") or {}),
            components=dict(e.get("components") or {}),
        ))
    return signals


def score_ranking(
    ranking: list[dict[str, Any]],
    *,
    weights: dict[str, float],
    cfg: dict[str, Any],
    completeness: float,
) -> dict[str, Any]:
    """Re-score a stored ranking under ``weights`` and return the decision.

    Mirrors ``run_signal_engine``'s scoring pipeline: ``score_signals`` fills
    the per-signal components and weighted totals from the RECONSTRUCTED raw
    fields (model_prob, edge_pp, movement, completeness), then ``rank_and_pick``
    applies the same NO BET gates and confidence labels. ``odds_disagreement``
    is False here (the disagreement dock was already applied at prediction
    time; re-weighting must not double-dock).
    """
    signals = _reconstruct_signals(ranking)
    min_edge_pp = float(cfg.get("min_edge_pp", 3.0))
    conflict_pp = float(cfg.get("conflict_pp", 8.0))
    score_signals(
        signals,
        weights=weights,
        min_edge_pp=min_edge_pp,
        conflict_pp=conflict_pp,
        completeness=completeness,
        context=None,
    )
    result = rank_and_pick(
        signals,
        best_pick_margin=float(cfg.get("best_pick_margin", 0.06)),
        no_bet_score=float(cfg.get("no_bet_score", 0.45)),
        min_confluence=int(cfg.get("min_confluence", 2)),
        conflict_pp=conflict_pp,
        min_data_quality=float(cfg.get("min_data_quality", 0.30)),
        completeness=completeness,
        confidence_thresholds=cfg,
        odds_disagreement=False,
    )
    # Match the production ``run_signal_engine`` contract: best_pick is a
    # JSON-safe dict (rank_and_pick itself returns a Signal object).
    bp = result.get("best_pick")
    if bp is not None:
        result["best_pick"] = {
            "market": bp.market,
            "selection": bp.selection,
            "score": bp.score,
            "confidence": bp.confidence,
            "model_prob": round(bp.model_prob, 4),
            "market_odds": bp.market_odds,
            "edge_pp": bp.edge_pp,
            "line": bp.line,
            "side": bp.side,
            "components": bp.components,
            "movement": bp.movement,
        }
    return result


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _market_odds_for(pick: dict[str, Any]) -> float | None:
    odds = pick.get("market_odds")
    if odds is None:
        # Reconstructed picks keep market_odds; fall back to the stored pick.
        return None
    try:
        o = float(odds)
        return o if o > 1.0 else None
    except (TypeError, ValueError):
        return None


def evaluate_weight_set(
    records: list[dict[str, Any]],
    *,
    weights: dict[str, float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Settle every record's re-weighted best pick; aggregate by market."""
    fam: dict[str, dict[str, float | int]] = {}
    n_no_bet = 0
    n_bets = 0
    n_wins = 0
    n_push = 0
    ret = 0.0
    staked = 0.0
    for rec in records:
        res = score_ranking(
            rec["ranking"], weights=weights, cfg=cfg,
            completeness=rec["completeness"],
        )
        if res.get("decision") != "BEST PICK":
            n_no_bet += 1
            continue
        pick = res["best_pick"]
        settle = settle_signal(
            pick, rec["home_goals"], rec["away_goals"]
        )
        market = pick.get("market") or "?"
        b = fam.setdefault(market, {"n": 0, "wins": 0, "ret": 0.0, "staked": 0.0})
        b["n"] = int(b["n"]) + 1
        if settle["result"] == "win":
            b["wins"] = int(b["wins"]) + 1
            n_wins += 1
        elif settle["result"] == "push":
            n_push += 1
        odds = _market_odds_for(pick)
        if odds and odds > 1.0:
            b["staked"] = float(b["staked"]) + 1.0
            b["ret"] = float(b["ret"]) + float(settle["stake_return"]) * odds
            staked += 1.0
            ret += float(settle["stake_return"]) * odds
            n_bets += 1
    markets = {}
    for m, b in fam.items():
        b["win_rate"] = round(int(b["wins"]) / int(b["n"]), 4) if b["n"] else None
        b["roi_pct"] = (
            round((float(b["ret"]) - float(b["staked"])) / float(b["staked"]) * 100.0, 2)
            if float(b["staked"]) > 0 else None
        )
        b["n"] = int(b["n"])
        b["wins"] = int(b["wins"])
        b["ret"] = round(float(b["ret"]), 4)
        b["staked"] = round(float(b["staked"]), 4)
        markets[m] = b
    return {
        "n_settled": len(records),
        "n_bets": n_bets,
        "n_no_bet": n_no_bet,
        # Hit rate counts a push as half a win (same convention as the
        # engine's ``run_signal_backtest`` win_rate), so it is comparable.
        "hit_rate": (
            round((n_wins + 0.5 * n_push) / n_bets, 4) if n_bets > 0 else None
        ),
        "roi_pct": round((ret - staked) / staked * 100.0, 2) if staked > 0 else None,
        "markets": markets,
    }


def split_periods(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chronological split into two non-overlapping periods by kickoff."""
    def _key(r: dict[str, Any]) -> str:
        return r.get("kickoff") or r.get("match_id") or ""
    ordered = sorted(records, key=_key)
    half = len(ordered) // 2
    if half == 0:
        return ordered, []
    return ordered[:half], ordered[half:]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

WEIGHT_SETS: list[tuple[str, dict[str, float]]] = [
    # "current" reflects the double-count fix (2026-08-17): movement 0.15,
    # late_movement 0.00 (penalty-only, weight moved to model+statistical).
    ("current 0.15/0.00", {"movement": 0.15, "late_movement": 0.00}),
    ("pre-fix 0.15/0.10", {"movement": 0.15, "late_movement": 0.10}),
    ("reduced 0.10/0.05", {"movement": 0.10, "late_movement": 0.05}),
    ("increased 0.20/0.15", {"movement": 0.20, "late_movement": 0.15}),
    ("movement off 0.00/0.00", {"movement": 0.00, "late_movement": 0.00}),
]


def _load_cfg() -> dict[str, Any]:
    try:
        cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
        return (cfg.get("models") or {}).get("signal_engine") or {}
    except (OSError, json.JSONDecodeError):
        return {}


def format_report(
    results: dict[str, dict[str, Any]],
    *,
    n_total: int,
    periods: tuple[str, str],
    cfg_weights: dict[str, float],
) -> str:
    lines = [
        "# P4 (re-runnable) — Signal-weight backtest",
        "",
        f"Settled matches with stored ranking: **{n_total}** "
        f"(min for a meaningful report: {MIN_SAMPLES_DEFAULT})",
        "",
        f"Periods: **{periods[0]}** | **{periods[1]}**",
        "",
        "| Weight set | Period | Bets | ROI % | Hit % | NO BET |",
        "|---|---|---|---|---|---|",
    ]
    for name, _ in WEIGHT_SETS:
        for pi, key in enumerate(("period_a", "period_b")):
            r = (results.get(name) or {}).get(key) or {}
            if r.get("n_settled", 0) == 0:
                continue
            lines.append(
                f"| {name} | {periods[pi]} | {r.get('n_bets', 0)} | "
                f"{r.get('roi_pct') if r.get('roi_pct') is not None else '—'} | "
                f"{r.get('hit_rate') if r.get('hit_rate') is not None else '—'} | "
                f"{r.get('n_no_bet', 0)} |"
            )
    lines += [
        "",
        f"Production weights in config: `{json.dumps(cfg_weights)}`",
        "Only change production weights if a weight set beats current in BOTH",
        "periods -- otherwise retain current weights with this report as evidence.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-backtest-signal")
    parser.add_argument(
        "--log", default=str(ROOT / "cache" / "football" / "predictions.jsonl"),
        help="prediction_log JSONL path (default: cache/football/predictions.jsonl)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=MIN_SAMPLES_DEFAULT,
        help=f"settled matches with ranking required for a report (default {MIN_SAMPLES_DEFAULT})",
    )
    parser.add_argument(
        "--report", default=None,
        help="write the markdown report to this path (default: stdout only)",
    )
    args = parser.parse_args(argv)

    records = load_settled_ranked_matches(args.log)
    if len(records) < args.min_samples:
        print(
            f"INSUFFICIENT DATA: {len(records)} settled matches with stored ranking "
            f"< {args.min_samples} required. Snapshots before P4 carry no "
            f"signal_engine_ranking; they accumulate as new matches settle. "
            f"Re-run once the floor is met."
        )
        return 0

    pa, pb = split_periods(records)
    cfg = _load_cfg()
    base = dict(cfg.get("weights") or {})
    results: dict[str, dict[str, Any]] = {}
    for name, delta in WEIGHT_SETS:
        weights = dict(base)
        weights.update(delta)
        results[name] = {
            "period_a": evaluate_weight_set(pa, weights=weights, cfg=cfg),
            "period_b": evaluate_weight_set(pb, weights=weights, cfg=cfg),
        }
    report = format_report(
        results, n_total=len(records), periods=("period A", "period B"),
        cfg_weights=base,
    )
    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"\nreport written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
