"""Rule evaluation harness (READ-ONLY, no engine re-run, no network).

For every settled pick in the prediction logs, extract the stored decision
fields (market, selection, model_prob, edge_pp, score, confidence, odds,
result) and score CANDIDATE tier rules against realised units.

Purpose: prove a proposed gate change earns units BEFORE it is merged, and
show exactly which wins it sacrifices and which losses it catches.

Usage::

    python scripts/eval_rules.py [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--json]

Nothing is ever written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import _match_dedupe_key, _pick_tier, _read_lines  # noqa: E402
from agents.football.signal_engine import settle_signal  # noqa: E402
from scripts.replay_gates import DEFAULT_LOGS, MEDIUM_SCORE, _f, _merge, _model_prob  # noqa: E402


def _pick_of(s: dict[str, Any]) -> dict[str, Any] | None:
    se = s.get("signal_engine") or {}
    pick = se.get("pick") or s.get("signal_engine_pick")
    return pick if isinstance(pick, dict) else None


def collect(paths: list[Path], since: str | None, until: str | None) -> list[dict[str, Any]]:
    """One row per settled pick with everything a tier rule could look at."""
    settles, _odds, newest = _merge(paths)
    rows: list[dict[str, Any]] = []
    for s in newest.values():
        ko = str(s.get("kickoff") or "")[:10]
        if since and ko < since:
            continue
        if until and ko > until:
            continue
        pick = _pick_of(s)
        if not pick:
            continue
        st = settles.get(s["match_id"]) or {}
        hg, ag = st.get("home_goals"), st.get("away_goals")
        if hg is None or ag is None:
            continue
        res = settle_signal(pick, hg, ag)["result"]
        if res not in ("win", "loss", "half_win", "half_loss", "push"):
            continue
        prob, edge = _model_prob(s, pick)
        odds = _f(pick.get("market_odds")) or _f(pick.get("odds")) or 0.0
        if odds <= 1.0:
            # no usable price -> cannot score units honestly, skip the row
            continue
        if res == "win":
            units = odds - 1.0
        elif res == "loss":
            units = -1.0
        elif res == "half_win":
            units = (odds - 1.0) / 2.0
        elif res == "half_loss":
            units = -0.5
        else:
            units = 0.0
        rows.append({
            "name": f"{s.get('home')} v {s.get('away')}",
            "league": s.get("league"),
            "kickoff": ko,
            "market": pick.get("market"),
            "selection": pick.get("selection"),
            "model_prob": prob,
            "edge_pp": edge,
            "score": _f(pick.get("score")),
            "confidence": pick.get("confidence"),
            "odds": odds,
            "result": res,
            "units": units,
            "tier": _pick_tier(pick, MEDIUM_SCORE),
        })
    return rows


# --- candidate rules: return True to KEEP the pick as BEST PICK -------------

def _p(r: dict[str, Any]) -> float | None:
    return r.get("model_prob")


def _e(r: dict[str, Any]) -> float | None:
    return r.get("edge_pp")


def rule_current(r: dict[str, Any]) -> bool:
    """SHIPPED (2026-09-04 revisi) == rule_b_neg_edge_goals.

    Rule B plus: Total/BTTS with edge < 0 is never a BEST PICK.
    Kept as its own entry so a future change can be diffed against it.
    """
    return rule_b_neg_edge_goals(r)


def rule_old_goals60(r: dict[str, Any]) -> bool:
    """REVERTED f2ab8be: goals markets needed prob>=0.60 absolutely."""
    p, e = _p(r), _e(r)
    if p is None:
        return True
    if r.get("market") in ("Total", "BTTS"):
        return p >= 0.60
    return p >= 0.60 or (p >= 0.50 and e is not None and e >= 0.0)


def rule_b_only(r: dict[str, Any]) -> bool:
    """Pre-f2ab8be rule B: prob>=0.60 OR (prob>=0.50 AND edge>=0)."""
    p, e = _p(r), _e(r)
    if p is None:
        return True
    return p >= 0.60 or (p >= 0.50 and e is not None and e >= 0.0)


def rule_b_plus_neg_edge(r: dict[str, Any]) -> bool:
    """Rule B, plus: any pick with edge < 0 is never a BEST PICK."""
    p, e = _p(r), _e(r)
    if p is None:
        return True
    if e is not None and e < 0.0:
        return False
    return p >= 0.60 or (p >= 0.50 and e is not None and e >= 0.0)


def rule_b_neg_edge_goals(r: dict[str, Any]) -> bool:
    """Rule B, plus: negative edge blocked ONLY on Total/BTTS."""
    p, e = _p(r), _e(r)
    if p is None:
        return True
    if r.get("market") in ("Total", "BTTS") and e is not None and e < 0.0:
        return False
    return p >= 0.60 or (p >= 0.50 and e is not None and e >= 0.0)


def rule_b_neg_edge_1x2(r: dict[str, Any]) -> bool:
    """Rule B, plus: negative edge blocked ONLY on 1X2."""
    p, e = _p(r), _e(r)
    if p is None:
        return True
    if r.get("market") == "1X2" and e is not None and e < 0.0:
        return False
    return p >= 0.60 or (p >= 0.50 and e is not None and e >= 0.0)


def rule_edge_floor(r: dict[str, Any]) -> bool:
    """Pure value rule: keep only when edge >= 0, regardless of probability."""
    e = _e(r)
    if e is None:
        return True
    return e >= 0.0


def rule_b_negedge_goals_conf(r: dict[str, Any]) -> bool:
    """Winner candidate + drop LOW confidence (K5 already does this live)."""
    if r.get("confidence") == "LOW":
        return False
    return rule_b_neg_edge_goals(r)


def rule_b_negedge_goals_strict_edge(r: dict[str, Any]) -> bool:
    """Rule B + goals markets need edge >= +1.0pp (not merely >= 0)."""
    p, e = _p(r), _e(r)
    if p is None:
        return True
    if r.get("market") in ("Total", "BTTS"):
        if e is None or e < 1.0:
            return False
    return p >= 0.60 or (p >= 0.50 and e is not None and e >= 0.0)


RULES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "SHIPPED(ruleB+negedge-goals)": rule_current,
    "REVERTED(f2ab8be goals>=60)": rule_old_goals60,
    "ruleB(pre-fix)": rule_b_only,
    "ruleB+negedge-all": rule_b_plus_neg_edge,
    "ruleB+negedge-goals": rule_b_neg_edge_goals,
    "ruleB+negedge-1x2": rule_b_neg_edge_1x2,
    "ruleB+negedge-goals+noLOW": rule_b_negedge_goals_conf,
    "ruleB+goals-edge>=1pp": rule_b_negedge_goals_strict_edge,
    "edge>=0 only": rule_edge_floor,
}


def score_rule(rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    kept = [r for r in rows if fn(r)]
    dropped = [r for r in rows if not fn(r)]
    return {
        "n_kept": len(kept),
        "w": sum(1 for r in kept if r["result"] in ("win", "half_win")),
        "l": sum(1 for r in kept if r["result"] in ("loss", "half_loss")),
        "units": round(sum(r["units"] for r in kept), 2),
        "dropped_n": len(dropped),
        "dropped_wins": sum(1 for r in dropped if r["result"] in ("win", "half_win")),
        "dropped_losses": sum(1 for r in dropped if r["result"] in ("loss", "half_loss")),
        "units_forgone": round(sum(r["units"] for r in dropped), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", default=None)
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--by-market", action="store_true", help="also break the best rule down per market")
    a = ap.parse_args()
    paths = [Path(p) for p in a.log] if a.log else list(DEFAULT_LOGS)
    rows = collect(paths, a.since, a.until)

    out: dict[str, Any] = {
        "period": f"{a.since or 'start'}..{a.until or 'end'}",
        "n_settled_picks": len(rows),
        "baseline_units": round(sum(r["units"] for r in rows), 2),
        "rules": {name: score_rule(rows, fn) for name, fn in RULES.items()},
    }
    if a.by_market:
        per: dict[str, Any] = {}
        for m in sorted({str(r["market"]) for r in rows}):
            sub = [r for r in rows if str(r["market"]) == m]
            per[m] = {
                "n": len(sub),
                "units_all": round(sum(r["units"] for r in sub), 2),
                "rules": {name: score_rule(sub, fn) for name, fn in RULES.items()},
            }
        out["by_market"] = per

    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"=== RULE EVAL [{out['period']}] n={out['n_settled_picks']} picks, "
          f"units if all published = {out['baseline_units']:+.2f}u ===\n")
    print(f"{'rule':<28} {'kept':>10} {'units':>8}   {'dropped (W/L)':>14} {'u.forgone':>10}")
    for name, st in out["rules"].items():
        print(f"{name:<28} {st['w']:>4}-{st['l']:<4} {st['units']:>+8.2f}   "
              f"{st['dropped_wins']:>6}/{st['dropped_losses']:<7} {st['units_forgone']:>+10.2f}")
    if a.by_market:
        for m, blk in out["by_market"].items():
            print(f"\n--- {m} (n={blk['n']}, all={blk['units_all']:+.2f}u) ---")
            for name, st in blk["rules"].items():
                print(f"  {name:<28} {st['w']:>3}-{st['l']:<3} {st['units']:>+8.2f}u  "
                      f"(drop {st['dropped_wins']}W/{st['dropped_losses']}L)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
