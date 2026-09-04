"""Measure the evidence-floor change (2026-09-04) on stored ranking rows.

READ-ONLY. For every settled pick we recompute what the headline score
WOULD be under (a) the old bypass `if market>0: return score` and (b) the
new always-on floor, using the stored `components` block of the matching
ranking entry. Then we score both against realised units through the live
MEDIUM_SCORE tier cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.football.signal_engine import EVIDENCE_FLOOR_DEFAULTS, _movement_available  # noqa: E402
from scripts.eval_rules import collect  # noqa: E402
from scripts.replay_gates import DEFAULT_LOGS, MEDIUM_SCORE, _f, _merge  # noqa: E402


def _floor(score: float, comps: dict[str, Any], mv: dict[str, Any] | None, *, bypass: bool) -> float:
    """Old behaviour when bypass=True (market>0 short-circuits), else new."""
    if bypass and float(comps.get("market") or 0.0) > 0.0:
        return score
    has_stat = "statistical" in comps
    has_mv = _movement_available(mv)
    missing = (not has_stat) + (not has_mv)
    if missing == 2:
        cap = float(EVIDENCE_FLOOR_DEFAULTS["score_cap_both_unavailable"])
    elif missing == 1:
        cap = float(EVIDENCE_FLOOR_DEFAULTS["score_cap_one_unavailable"])
    else:
        return score
    return min(score, cap)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    settles, _odds, newest = _merge(list(DEFAULT_LOGS))
    # index stored ranking rows by (match, market, selection)
    rank: dict[tuple, dict[str, Any]] = {}
    for s in newest.values():
        for e in s.get("signal_engine_ranking") or []:
            rank[(s.get("home"), s.get("away"), e.get("market"), e.get("selection"))] = e

    rows = collect(list(DEFAULT_LOGS), a.since, a.until)
    book = {"old": {"n": 0, "w": 0, "l": 0, "u": 0.0}, "new": {"n": 0, "w": 0, "l": 0, "u": 0.0}}
    changed: list[str] = []
    for r in rows:
        h, aw = r["name"].split(" v ", 1)
        e = rank.get((h, aw, r["market"], r["selection"]))
        if not e:
            continue
        comps = e.get("components") or {}
        raw = _f(e.get("score"))
        if raw is None:
            continue
        mv = e.get("movement")
        s_old = _floor(raw, comps, mv, bypass=True)
        s_new = _floor(raw, comps, mv, bypass=False)
        for lab, sc in (("old", s_old), ("new", s_new)):
            if sc >= MEDIUM_SCORE:  # survives the tier cut -> published as BEST
                b = book[lab]
                b["n"] += 1
                b["w"] += r["result"] in ("win", "half_win")
                b["l"] += r["result"] in ("loss", "half_loss")
                b["u"] += r["units"]
        if (s_old >= MEDIUM_SCORE) != (s_new >= MEDIUM_SCORE):
            changed.append(
                f"{r['name']} {r['market']} {r['selection']} score {raw:.2f}"
                f" old {s_old:.2f} -> new {s_new:.2f} | {r['result']} {r['units']:+.2f}u"
            )

    out = {
        "period": f"{a.since or 'start'}..{a.until or 'end'}",
        "old_bypass": {**book["old"], "u": round(book["old"]["u"], 2)},
        "new_always_on": {**book["new"], "u": round(book["new"]["u"], 2)},
        "delta_units": round(book["new"]["u"] - book["old"]["u"], 2),
        "changed": changed,
    }
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(f"=== EVIDENCE FLOOR [{out['period']}] ===")
    for lab in ("old_bypass", "new_always_on"):
        b = out[lab]
        print(f"  {lab:<15} published={b['n']:<4} {b['w']}-{b['l']}  units={b['u']:+.2f}")
    print(f"  delta (new - old) = {out['delta_units']:+.2f}u over {len(changed)} changed picks")
    for c in changed:
        print("   *", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
