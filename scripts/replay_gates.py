"""Replay the 2026-08-28 post-mortem rules over stored prediction logs (read-only).

Usage::

    python scripts/replay_gates.py [--log PATH ...] [--since YYYY-MM-DD] [--json]

For every settled BEST PICK (newest snapshot per canonical match) the script
re-applies the general rules K1/K2/K3/K5 to the STORED fields (no engine
re-run, no network) and reports, per period and per class:

    LOSS caught / WIN affected / net units before -> after

It also reconstructs the SUGGESTION TO PICK from the stored ``market_totals``
+ ``odds_1x2`` + nearest ``odds_snapshot.odds_ah`` under BOTH the legacy rule
(always pick the max-implied market) and the K4 rule (``market_lean``), and
grades each against the settle row.

This is the generalisation check the plan demands: a class that only "fires"
on the 25-27 Aug losing set but costs units on 11-24 Aug must be weakened to
a penalty/label, not shipped as a veto. Nothing is written unless ``--json``
is given (prints a JSON blob to stdout).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.market_lean import (  # noqa: E402
    compute_suggestion,
    lean_candidates,
    suggestion_for_settlement,
)
from agents.football.pick_gates import (  # noqa: E402
    DEFAULT_ELO_MAX,
    DEFAULT_ELO_MIN,
    is_directional_selection,
    is_low_scoring_selection,
    lambda_1x2_gate,
)
from agents.football.prediction_log import _match_dedupe_key, _read_lines  # noqa: E402
from agents.football.signal_engine import settle_signal  # noqa: E402

DEFAULT_LOGS = [
    ROOT / "cache" / "football" / "predictions.jsonl",
    ROOT / "baseline" / "predictions_vps.jsonl",
]
MEDIUM_SCORE = 0.52


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _period(s: dict[str, Any]) -> str:
    ko = str(s.get("kickoff") or s.get("ts") or "")[:10]
    return "25-27 Aug (in-sample)" if ko >= "2026-08-25" else "11-24 Aug (out-of-sample)"


def _both_prior(mp: dict[str, Any], f: dict[str, Any]) -> bool:
    hs, as_ = mp.get("elo_home_seeded"), mp.get("elo_away_seeded")
    if hs is not None and as_ is not None:
        return (not hs) and (not as_)
    eh, ea = _f(f.get("elo_home")), _f(f.get("elo_away"))
    return mp.get("elo_seeded") is False and eh == 1500.0 and ea == 1500.0


def classes_for_pick(s: dict[str, Any], pick: dict[str, Any]) -> list[str]:
    """Which post-mortem rules would have acted on this stored pick."""
    mp = s.get("model_probs") or {}
    f = s.get("features") or {}
    market, sel = pick.get("market"), pick.get("selection")
    out: list[str] = []
    eh, ea = _f(f.get("elo_home")), _f(f.get("elo_away"))
    if any(v is not None and (v < DEFAULT_ELO_MIN or v > DEFAULT_ELO_MAX) for v in (eh, ea)):
        out.append("K2-elo-band")
    if _both_prior(mp, f) and is_directional_selection(market, sel):
        out.append("K1-both-prior-directional")
    ok_g3, _ = lambda_1x2_gate(mp)
    if not ok_g3 and is_low_scoring_selection(market, sel):
        out.append("K2-g3-low-scoring")
    score = _f(pick.get("score")) or 0.0
    if score < MEDIUM_SCORE or pick.get("confidence") == "LOW":
        out.append("K5-lean")
    ts = ((s.get("context_data") or {}).get("tie_state")) or {}
    if ts:
        side = pick.get("side") or ("home" if str(sel).startswith("Home") else "away")
        if ts.get("state") == "decided" and is_directional_selection(market, sel) and side == ts.get("leader"):
            out.append("K3-decided-leader")
        if ts.get("state") == "balanced" and (
            (market == "Total" and str(sel).startswith("Over")) or (market == "BTTS" and str(sel).endswith("Yes"))
        ):
            out.append("K3-balanced-over")
    return out


def _nearest_ah(odds_snaps: list[dict[str, Any]], match_id: str, ts: str) -> dict[str, Any] | None:
    """AH consensus from the odds_snapshot closest to (preferably before) ``ts``."""
    before = [r for r in odds_snaps if r.get("match_id") == match_id and r.get("odds_ah") and (r.get("ts") or "") <= ts]
    after = [r for r in odds_snaps if r.get("match_id") == match_id and r.get("odds_ah") and (r.get("ts") or "") > ts]
    best = max(before, key=lambda r: r.get("ts") or "") if before else (min(after, key=lambda r: r.get("ts") or "") if after else None)
    return (best or {}).get("odds_ah")


def legacy_suggestion(totals, consensus, ah) -> dict[str, Any] | None:
    cands = lean_candidates(totals, consensus, ah)
    if not cands:
        return None
    scored = []
    for c in cands:
        imp = float(c.get("implied") or 0.0)
        vig = max(0.0, min(0.9, float(c.get("vig") or 0.0)))
        scored.append((imp * (1.0 - vig), imp, -float(c.get("odds") or 999.0), c))
    scored.sort(key=lambda x: x[:3], reverse=True)
    return scored[0][3]


def replay(paths: list[Path], since: str | None) -> dict[str, Any]:
    report: dict[str, Any] = {"best_pick": {}, "suggestion": {}}
    for path in paths:
        if not path.exists():
            continue
        rows = _read_lines(path)
        settles = {r["match_id"]: r for r in rows if r.get("event") == "settle"}
        odds_snaps = [r for r in rows if r.get("event") == "odds_snapshot"]
        newest: dict[tuple, dict[str, Any]] = {}
        for s in rows:
            if s.get("event") != "snapshot" or s["match_id"] not in settles:
                continue
            if since and str(s.get("kickoff") or "")[:10] < since:
                continue
            key = _match_dedupe_key(s)
            cur = newest.get(key)
            if cur is None or (s.get("ts") or "") > (cur.get("ts") or ""):
                newest[key] = s
        for s in newest.values():
            st = settles[s["match_id"]]
            hg, ag = int(st.get("home_goals") or 0), int(st.get("away_goals") or 0)
            per = _period(s)
            pick = s.get("signal_engine_pick") or s.get("best_pick")
            if pick:
                res = settle_signal(pick, hg, ag)["result"]
                odds = _f(pick.get("market_odds")) or 0.0
                unit = (odds - 1.0) if res == "win" else (-1.0 if res == "loss" else 0.0)
                bucket = report["best_pick"].setdefault(per, defaultdict(lambda: {"loss_caught": 0, "win_affected": 0, "units_removed": 0.0}))
                summary = report["best_pick"].setdefault("_summary", {}).setdefault(per, {"n": 0, "w": 0, "l": 0, "units": 0.0})
                summary["n"] += 1
                summary["w"] += res == "win"
                summary["l"] += res == "loss"
                summary["units"] += unit if odds > 1.0 else 0.0
                for c in classes_for_pick(s, pick):
                    b = bucket[c]
                    if res == "loss":
                        b["loss_caught"] += 1
                    elif res == "win":
                        b["win_affected"] += 1
                    b["units_removed"] += unit if odds > 1.0 else 0.0
            totals = s.get("market_totals") or {}
            consensus = s.get("odds_1x2") or {}
            ah = _nearest_ah(odds_snaps, s["match_id"], s.get("ts") or "")
            leg = legacy_suggestion(totals, consensus, ah)
            new = compute_suggestion(
                totals=totals, consensus=consensus, ah=ah,
                model_probs=s.get("model_probs"), features=s.get("features"),
                tie_state=((s.get("context_data") or {}).get("tie_state")),
                ranking=s.get("signal_engine_ranking"),
            )["pick"]
            sb = report["suggestion"].setdefault(per, {"legacy": {"n": 0, "w": 0, "l": 0, "units": 0.0},
                                                      "k4": {"n": 0, "w": 0, "l": 0, "units": 0.0}})
            for label, p in (("legacy", leg), ("k4", new)):
                if not p:
                    continue
                sig = suggestion_for_settlement(p)
                r = settle_signal(sig, hg, ag)["result"]
                if r == "n/a":
                    continue
                o = _f(p.get("odds")) or 0.0
                sb[label]["n"] += 1
                sb[label]["w"] += r in ("win", "half_win")
                sb[label]["l"] += r in ("loss", "half_loss")
                sb[label]["units"] += (o - 1.0) if r == "win" else (-1.0 if r == "loss" else 0.0)
    # plain dicts
    for per, b in list(report["best_pick"].items()):
        if per != "_summary":
            report["best_pick"][per] = dict(b)
    return report


def _print(report: dict[str, Any]) -> None:
    print("=== BEST PICK: rule impact on stored picks (LOSS caught / WIN affected / units removed) ===")
    for per, summ in (report["best_pick"].get("_summary") or {}).items():
        print(f"\n[{per}] n={summ['n']} W={summ['w']} L={summ['l']} net={summ['units']:+.2f}u (as published)")
        for cls, b in sorted((report["best_pick"].get(per) or {}).items()):
            print(f"  {cls:28} loss_caught={b['loss_caught']:2d} win_affected={b['win_affected']:2d} "
                  f"units_removed={b['units_removed']:+.2f}")
    print("\n=== SUGGESTION: legacy (always-on max implied) vs K4 (market_lean) ===")
    for per, b in report["suggestion"].items():
        for label in ("legacy", "k4"):
            x = b[label]
            wr = (x["w"] / x["n"] * 100.0) if x["n"] else 0.0
            print(f"[{per}] {label:6} n={x['n']:3d} W={x['w']:3d} L={x['l']:3d} win-rate={wr:5.1f}% net={x['units']:+.2f}u")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", type=Path, help="prediction log (repeatable)")
    ap.add_argument("--since", help="only kickoffs on/after YYYY-MM-DD")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()
    paths = args.log or DEFAULT_LOGS
    report = replay(paths, args.since)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
