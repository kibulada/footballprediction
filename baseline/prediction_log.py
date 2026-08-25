"""Append-only JSONL prediction log (audit PHASE 7).

Every pre-match prediction is appended as one immutable JSON line before
kickoff: match identity, feature-input hash, model probabilities, consensus
odds, margin-free edge, confidence, signal, calibration state, model version
and sources.

After the match, ``settle`` appends a separate settlement line keyed by the
same match_id (result + optional closing odds). ``stats`` joins snapshots and
settlements and reports realised hit rate, log-loss, CLV and flat-stake ROI.

Honesty rules:
  - Append-only by construction; snapshots are never edited in place.
  - Metrics are reported only for *settled* snapshots.
  - ROI is flat-stake on the best 1X2 pick with margin-free edge >= threshold
    (mirrors validate.py); CLV is (model_prob * closing_odds - 1) per settled
    snapshot that carries closing odds -- both clearly labelled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .timeutil import utc_now_iso

KEYS = ("home", "draw", "away")


def make_match_id(league: str, home: str, away: str, kickoff: str | None) -> str:
    return f"{league}||{home}||{away}||{kickoff or ''}"


def list_unsettled(path: str | Path) -> list[dict[str, Any]]:
    """Snapshots that do not yet have a matching settlement line."""
    rows = _read_lines(Path(path))
    settled_ids = {r["match_id"] for r in rows if r.get("event") == "settle"}
    return [
        r for r in rows
        if r.get("event") == "snapshot" and r["match_id"] not in settled_ids
    ]


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_snapshot(
    path: str | Path,
    *,
    match_id: str,
    league: str,
    home: str,
    away: str,
    kickoff: str | None,
    prob: dict[str, float] | None,
    odds: dict[str, float] | None,
    edge: dict[str, float] | None,
    confidence: float | None,
    signal: int | None,
    calibration: dict[str, Any] | None,
    model_version: str | None,
    input_hash: str | None,
    best_pick: dict[str, Any] | None,
    sources: list[str] | None,
) -> None:
    """Append one immutable pre-match prediction snapshot."""
    row = {
        "event": "snapshot",
        "match_id": match_id,
        "ts": utc_now_iso(),
        "league": league,
        "home": home,
        "away": away,
        "kickoff": kickoff,
        "prob_1x2": {k: round(float(v), 4) for k, v in (prob or {}).items()},
        "odds_1x2": (
            {k: round(float(v), 4) for k, v in (odds or {}).items()} if odds else None
        ),
        "edge_pct": (
            {k: round(float(v), 2) for k, v in (edge or {}).items()} if edge else None
        ),
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "signal": int(signal) if signal is not None else None,
        "calibration": calibration or None,
        "model_version": model_version,
        "input_hash": input_hash,
        "best_pick": best_pick,
        "sources": sources or [],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def settle(
    path: str | Path,
    *,
    match_id: str,
    home_goals: int,
    away_goals: int,
    closing_odds: dict[str, float] | None = None,
) -> bool:
    """Append a settlement for a previously logged snapshot.

    Returns False (and does NOT append) when no snapshot with that match_id
    exists -- a settlement without a prediction is meaningless.
    """
    rows = _read_lines(Path(path))
    if not any(r.get("event") == "snapshot" and r.get("match_id") == match_id for r in rows):
        return False
    outcome = "home" if home_goals > away_goals else ("draw" if home_goals == away_goals else "away")
    row = {
        "event": "settle",
        "match_id": match_id,
        "ts": utc_now_iso(),
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "outcome": outcome,
        "closing_odds": (
            {k: round(float(v), 4) for k, v in (closing_odds or {}).items()}
            if closing_odds
            else None
        ),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def compute_stats(path: str | Path, edge_threshold: float = 0.02) -> dict[str, Any]:
    """Aggregate realised metrics over settled snapshots."""
    rows = _read_lines(Path(path))
    settlements = {r["match_id"]: r for r in rows if r.get("event") == "settle"}
    snapshots = [r for r in rows if r.get("event") == "snapshot"]

    n_snapshots = len(snapshots)
    settled: list[dict[str, Any]] = []
    for s in snapshots:
        st = settlements.get(s["match_id"])
        if st is None:
            continue
        prob = s.get("prob_1x2") or {}
        odds = s.get("odds_1x2") or {}
        outcome = st.get("outcome", "")
        p_out = prob.get(outcome, 0.0)
        logloss = -math.log(max(1e-9, p_out)) if p_out > 0 else None
        pick = max(prob, key=prob.get) if prob else None
        hit = bool(pick and pick == outcome)
        predicted = bool(prob)

        roi = None
        if pick and odds.get(pick, 0) > 1.0:
            raw = {k: (1.0 / odds[k] if odds.get(k, 0) > 1.0 else 0.0) for k in KEYS}
            total = sum(raw.values())
            if total > 0:
                norm = {k: v / total for k, v in raw.items()}
                edge = prob.get(pick, 0.0) - norm.get(pick, 0.0)
                if edge >= edge_threshold:
                    roi = (odds[pick] - 1.0) if hit else -1.0

        clv = None
        close = st.get("closing_odds") or {}
        if pick and close.get(pick, 0) > 1.0:
            clv = prob.get(pick, 0.0) * close[pick] - 1.0

        settled.append(
            {
                "match_id": s["match_id"],
                "pick": pick,
                "outcome": outcome,
                "hit": hit,
                "predicted": predicted,
                "logloss": logloss,
                "roi": roi,
                "clv": clv,
            }
        )

    n = len(settled)
    predicted = sum(1 for x in settled if x.get("predicted"))
    if not n:
        return {
            "file": str(path),
            "n_snapshots": n_snapshots,
            "n_settled": 0,
            "n_predicted": 0,
            "hit_rate": None,
            "avg_logloss": None,
            "roi": None,
            "n_bets": 0,
            "clv_pct": None,
            "n_clv": 0,
        }
    loglosses = [x["logloss"] for x in settled if x["logloss"] is not None]
    rois = [x["roi"] for x in settled if x["roi"] is not None]
    clvs = [x["clv"] for x in settled if x["clv"] is not None]
    return {
        "file": str(path),
        "n_snapshots": n_snapshots,
        "n_settled": n,
        # hit_rate only over snapshots that actually carried a 1X2 prediction
        # (empty prob_1x2 = no model output, must not count as a miss).
        "n_predicted": predicted,
        "hit_rate": (
            round(sum(1 for x in settled if x["hit"]) / predicted, 4) if predicted else None
        ),
        "avg_logloss": round(sum(loglosses) / len(loglosses), 4) if loglosses else None,
        "roi": round(sum(rois) / len(rois), 4) if rois else None,
        "n_bets": len(rois),
        "clv_pct": round(sum(clvs) / len(clvs) * 100.0, 2) if clvs else None,
        "n_clv": len(clvs),
    }


def format_stats(stats: dict[str, Any], edge_threshold: float = 0.02) -> str:
    def _fmt(v: Any, suffix: str = "") -> str:
        return "n/a" if v is None else f"{v}{suffix}"

    return "\n".join(
        [
            "PREDICTION LOG STATS",
            f"  file       : {stats['file']}",
            f"  snapshots  : {stats['n_snapshots']}",
            f"  settled    : {stats['n_settled']}",
            f"  hit rate   : {_fmt(stats['hit_rate'], '%')}  (best 1X2 pick; "
            f"{stats['n_predicted']} predicted)",
            f"  log-loss   : {_fmt(stats['avg_logloss'])}  (avg over settled)",
            f"  ROI        : {_fmt(stats['roi'], '%')}  ({stats['n_bets']} bets, "
            f"flat-stake, edge>={edge_threshold:.0%})",
            f"  CLV        : {_fmt(stats['clv_pct'], '%')}  ({stats['n_clv']} "
            "settled w/ closing odds)",
        ]
    )


DEFAULT_LOG_PATH = "cache/football/predictions.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-prediction-log")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # --file is accepted both before the subcommand (global) and on it.
    parser.add_argument("--file", default=DEFAULT_LOG_PATH, help="JSONL log path")

    st = sub.add_parser("stats", help="aggregate realised metrics")
    st.add_argument("--edge-threshold", type=float, default=0.02)
    st.add_argument("--file", default=None, help="JSONL log path (override)")

    se = sub.add_parser("settle", help="append result for a logged snapshot")
    se.add_argument("--match-id", required=True)
    se.add_argument("--home-goals", type=int, required=True)
    se.add_argument("--away-goals", type=int, required=True)
    se.add_argument("--closing-odds", default=None,
                    help="closing 1X2 odds as home,draw,away (e.g. 1.62,4.30,4.60)")
    se.add_argument("--file", default=None, help="JSONL log path (override)")

    args = parser.parse_args(argv)
    log_path = args.file or DEFAULT_LOG_PATH
    if args.cmd == "stats":
        print(
            format_stats(
                compute_stats(log_path, edge_threshold=args.edge_threshold),
                edge_threshold=args.edge_threshold,
            )
        )
        return 0
    if args.cmd == "settle":
        closing = None
        if args.closing_odds:
            parts = [x.strip() for x in args.closing_odds.split(",")]
            if len(parts) == 3:
                closing = dict(zip(KEYS, (float(p) for p in parts)))
            else:
                print("--closing-odds harus 3 angka: home,draw,away", file=__import__("sys").stderr)
                return 2
        ok = settle(
            log_path, match_id=args.match_id,
            home_goals=args.home_goals, away_goals=args.away_goals,
            closing_odds=closing,
        )
        if not ok:
            print(f"tidak ada snapshot untuk match_id '{args.match_id}' "
                  f"(file: {log_path})", file=__import__("sys").stderr)
            return 1
        print(f"settled {args.match_id}: {args.home_goals}-{args.away_goals}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
