"""Calibration report: how accurate are the model's probabilities? (read-only)

Usage::

    python scripts/calibration_report.py [--log PATH ...] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
                                         [--fit-weights] [--out reports/calibration_YYYY-MM-DD.md]

For every SETTLED match (not only the picks) it scores, per market:

  * Brier / log-loss of the RAW model probability (``model_probs.raw`` when the
    market anchor ran, else ``model_probs``), the ANCHORED probability, the
    devigged MARKET, and a sweep of ``alpha*model + (1-alpha)*market`` blends;
  * a reliability table per model bucket (n, actual rate, mean market);
  * 1X2 favourite hit-rate by model-vs-market gap (does "model above the
    market" mean value, or overconfidence?);
  * ``--fit-weights``: a coarse grid over Elo / Poisson / market 1X2 weights
    using the persisted ``model_probs.components_1x2`` (M2); rows without
    components are skipped.

All ``--log`` files are merged (settles unioned, newest snapshot per match).
Writes nothing unless ``--out`` is given (Markdown).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import _match_dedupe_key, _read_lines  # noqa: E402

DEFAULT_LOGS = [
    ROOT / "cache" / "football" / "predictions.jsonl",
    ROOT / "baseline" / "predictions_vps.jsonl",
]
KEYS = ("home", "draw", "away")
ALPHAS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.0)


def _odds(v: Any) -> float | None:
    if isinstance(v, dict):
        v = v.get("odds")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 1.0 else None


def _devig(d: dict[str, Any] | None) -> dict[str, float] | None:
    if not d:
        return None
    inv = {}
    for k in KEYS:
        o = _odds(d.get(k))
        if o is None:
            return None
        inv[k] = 1.0 / o
    t = sum(inv.values())
    return {k: v / t for k, v in inv.items()}


def _fair_pair(mt: dict[str, Any], a: str, b: str) -> float | None:
    oa, ob = _odds((mt or {}).get(a)), _odds((mt or {}).get(b))
    if oa is None or ob is None:
        return None
    ia, ib = 1.0 / oa, 1.0 / ob
    return ia / (ia + ib)


def _merge(paths: list[Path]) -> tuple[dict[str, dict], dict[tuple, dict]]:
    settles: dict[str, dict[str, Any]] = {}
    newest: dict[tuple, dict[str, Any]] = {}
    for p in paths:
        if not p.exists():
            continue
        rows = _read_lines(p)
        for r in rows:
            if r.get("event") == "settle":
                settles.setdefault(r["match_id"], r)
        for s in rows:
            if s.get("event") != "snapshot":
                continue
            k = _match_dedupe_key(s)
            cur = newest.get(k)
            if cur is None or (s.get("ts") or "") > (cur.get("ts") or ""):
                newest[k] = s
    return settles, {k: s for k, s in newest.items() if s["match_id"] in settles}


def _brier3(p: dict[str, float], out: str) -> float:
    return sum((float(p.get(k, 0.0)) - (1.0 if k == out else 0.0)) ** 2 for k in KEYS)


def _ll3(p: dict[str, float], out: str) -> float:
    return -math.log(max(1e-6, float(p.get(out, 0.0))))


def _blend3(p: dict[str, float], mk: dict[str, float], a: float) -> dict[str, float]:
    return {k: a * float(p[k]) + (1 - a) * float(mk[k]) for k in KEYS}


def collect(paths: list[Path], since: str | None, until: str | None) -> dict[str, Any]:
    settles, newest = _merge(paths)
    rows_1x2, rows_ou, rows_btts, comps = [], [], [], []
    for s in newest.values():
        ko = str(s.get("kickoff") or "")[:10]
        if (since and ko < since) or (until and ko > until):
            continue
        st = settles[s["match_id"]]
        hg, ag = int(st.get("home_goals") or 0), int(st.get("away_goals") or 0)
        out = "home" if hg > ag else ("away" if ag > hg else "draw")
        mp = s.get("model_probs") or {}
        raw = mp.get("raw") or {}
        p_final = mp.get("1x2") or s.get("prob_1x2")
        p_raw = raw.get("1x2") or p_final
        mk = _devig(s.get("odds_1x2"))
        if p_final and mk and all(k in p_final for k in KEYS):
            rows_1x2.append({"raw": p_raw, "final": p_final, "mk": mk, "out": out,
                             "anchored": bool(mp.get("market_anchor_applied")),
                             "seeded": (mp.get("elo_home_seeded"), mp.get("elo_away_seeded"))})
            c = mp.get("components_1x2") or {}
            if c.get("elo") and c.get("poisson") and c.get("market"):
                comps.append({"elo": c["elo"], "poisson": c["poisson"], "market": c["market"], "out": out})
        mt = s.get("market_totals") or {}
        fo = _fair_pair(mt, "Over 2.5", "Under 2.5")
        if mp.get("over_2.5") is not None and fo is not None:
            rows_ou.append({"raw": float(raw.get("over_2.5", mp["over_2.5"])), "final": float(mp["over_2.5"]),
                            "mk": fo, "y": 1 if hg + ag > 2.5 else 0})
        fb = _fair_pair(mt, "BTTS Yes", "BTTS No")
        if mp.get("btts_yes") is not None and fb is not None:
            rows_btts.append({"raw": float(raw.get("btts_yes", mp["btts_yes"])), "final": float(mp["btts_yes"]),
                              "mk": fb, "y": 1 if (hg > 0 and ag > 0) else 0})
    return {"1x2": rows_1x2, "ou": rows_ou, "btts": rows_btts, "components": comps,
            "n_matches": len(newest)}


def report_lines(data: dict[str, Any], *, fit_weights: bool = False) -> list[str]:
    L: list[str] = []
    r1 = data["1x2"]
    L.append(f"# Calibration report (settled matches: {len(r1)} with 1X2 model+market)")
    L.append("")
    L.append("## 1X2 (Brier / log-loss, lower is better; favourite hit-rate)")
    L.append("")
    L.append("| probability | n | brier | logloss | fav-hit |")
    L.append("|---|---|---|---|---|")

    def _row(label, fn, subset=None):
        rs = [r for r in r1 if (subset is None or subset(r))]
        if not rs:
            return
        b = sum(_brier3(fn(r), r["out"]) for r in rs) / len(rs)
        ll = sum(_ll3(fn(r), r["out"]) for r in rs) / len(rs)
        hit = sum(max(fn(r), key=fn(r).get) == r["out"] for r in rs) / len(rs)
        L.append(f"| {label} | {len(rs)} | {b:.4f} | {ll:.4f} | {hit * 100:.0f}% |")

    _row("model raw (pre-anchor)", lambda r: r["raw"])
    _row("model final (as logged)", lambda r: r["final"])
    _row("market devig", lambda r: r["mk"])
    for a in ALPHAS[1:-1]:
        _row(f"{a:.2f} raw + {1 - a:.2f} market", lambda r, a=a: _blend3(r["raw"], r["mk"], a))
    both = lambda r: r["seeded"] == (True, True)  # noqa: E731
    one = lambda r: r["seeded"] != (True, True)  # noqa: E731
    _row("both seeded: model raw", lambda r: r["raw"], both)
    _row("both seeded: market", lambda r: r["mk"], both)
    _row("one/none seeded: model raw", lambda r: r["raw"], one)
    _row("one/none seeded: market", lambda r: r["mk"], one)
    L.append("")
    # favourite by gap
    gap = collections.Counter()
    for r in r1:
        fav = max(r["raw"], key=r["raw"].get)
        d = (float(r["raw"][fav]) - float(r["mk"][fav])) * 100
        key = "model >= market+3pp" if d >= 3 else ("model <= market-3pp" if d <= -3 else "within 3pp")
        gap[(key, fav == r["out"])] += 1
    L.append("## 1X2 favourite hit-rate by model-vs-market gap (raw model)")
    L.append("")
    L.append("| gap | n | hit |")
    L.append("|---|---|---|")
    for key in ("model >= market+3pp", "within 3pp", "model <= market-3pp"):
        n = gap[(key, True)] + gap[(key, False)]
        if n:
            L.append(f"| {key} | {n} | {gap[(key, True)] / n * 100:.0f}% |")
    L.append("")
    # favourite reliability
    buckets = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in r1:
        fav = max(r["raw"], key=r["raw"].get)
        k = min(9, int(float(r["raw"][fav]) * 10)) / 10
        buckets[k][0] += 1
        buckets[k][1] += (fav == r["out"])
        buckets[k][2] += float(r["mk"][fav])
    L.append("## 1X2 favourite reliability (raw model bucket)")
    L.append("")
    L.append("| bucket | n | hit | mean market |")
    L.append("|---|---|---|---|")
    for k in sorted(buckets):
        c = buckets[k]
        L.append(f"| {k:.1f}-{k + 0.1:.1f} | {c[0]} | {c[1] / c[0] * 100:.0f}% | {c[2] / c[0] * 100:.0f}% |")
    L.append("")
    for name, rows in (("Over 2.5", data["ou"]), ("BTTS Yes", data["btts"])):
        if not rows:
            continue
        L.append(f"## {name} (n={len(rows)})")
        L.append("")
        L.append("| probability | brier | logloss |")
        L.append("|---|---|---|")

        def _two(label, fn):
            b = sum((fn(r) - r["y"]) ** 2 for r in rows) / len(rows)
            ll = sum(-math.log(max(1e-6, fn(r) if r["y"] else 1 - fn(r))) for r in rows) / len(rows)
            L.append(f"| {label} | {b:.4f} | {ll:.4f} |")

        _two("model raw", lambda r: r["raw"])
        _two("model final", lambda r: r["final"])
        _two("market", lambda r: r["mk"])
        for a in ALPHAS[1:-1]:
            _two(f"{a:.2f} raw + {1 - a:.2f} market", lambda r, a=a: a * r["raw"] + (1 - a) * r["mk"])
        L.append("")
        rb = collections.defaultdict(lambda: [0, 0, 0.0])
        for r in rows:
            k = min(9, int(r["raw"] * 10)) / 10
            rb[k][0] += 1
            rb[k][1] += r["y"]
            rb[k][2] += r["mk"]
        L.append(f"reliability {name} (raw model bucket): n / actual / mean market")
        L.append("")
        L.append("| bucket | n | actual | market |")
        L.append("|---|---|---|---|")
        for k in sorted(rb):
            c = rb[k]
            L.append(f"| {k:.1f}-{k + 0.1:.1f} | {c[0]} | {c[1] / c[0] * 100:.0f}% | {c[2] / c[0] * 100:.0f}% |")
        L.append("")
    if fit_weights:
        comps = data["components"]
        L.append(f"## Weight fit (grid, components_1x2 rows: {len(comps)})")
        L.append("")
        if len(comps) < 30:
            L.append(f"too few rows with persisted components ({len(comps)} < 30) -- keep current weights")
        else:
            best = []
            steps = [i / 10 for i in range(11)]
            for we in steps:
                for wp in steps:
                    wm = round(1.0 - we - wp, 10)
                    if wm < -1e-9:
                        continue
                    b = 0.0
                    for c in comps:
                        p = {k: we * float(c["elo"][k]) + wp * float(c["poisson"][k]) + wm * float(c["market"][k]) for k in KEYS}
                        b += _brier3(p, c["out"])
                    best.append((b / len(comps), we, wp, wm))
            best.sort()
            L.append("| elo | poisson | market | brier |")
            L.append("|---|---|---|---|")
            for b, we, wp, wm in best[:8]:
                L.append(f"| {we:.1f} | {wp:.1f} | {wm:.1f} | {b:.4f} |")
            L.append("")
            L.append("Fit is in-sample; treat as direction, not as a setting, until a walk-forward window confirms it.")
        L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", type=Path, help="prediction log (repeatable; merged)")
    ap.add_argument("--since", help="only kickoffs on/after YYYY-MM-DD")
    ap.add_argument("--until", help="only kickoffs on/before YYYY-MM-DD")
    ap.add_argument("--fit-weights", action="store_true", help="grid-fit Elo/Poisson/market 1X2 weights")
    ap.add_argument("--out", type=Path, help="write the Markdown report here")
    ap.add_argument("--json", action="store_true", help="print raw JSON rows instead of Markdown")
    args = ap.parse_args()
    data = collect(args.log or DEFAULT_LOGS, args.since, args.until)
    if args.json:
        print(json.dumps({k: v for k, v in data.items() if k != "components"}, ensure_ascii=False, indent=1))
        return 0
    lines = report_lines(data, fit_weights=args.fit_weights)
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"written {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
