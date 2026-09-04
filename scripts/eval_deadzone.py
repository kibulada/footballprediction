"""Evaluasi permanen gate G12 (zona maut Elo gap 100-200).

READ-ONLY, tanpa network, tidak menulis apa pun. Jalankan ini SEBELUM
mengubah/menghapus ``pick_gates.elo_gap_dead_zone`` -- ambang 100/200
ditemukan dari data 26-Agu..04-Sep (n=60) dan WAJIB diukur ulang saat
sampel bertambah.

Usage::

    python scripts/eval_deadzone.py [--since YYYY-MM-DD] [--until ...]
                                    [--lo 100] [--hi 200] [--json]

Yang diuji:
  1. hit rate zona vs luar zona + bootstrap CI95 (metrik utama -- unit
     terlalu berisik karena variance harga)
  2. gain unit + bootstrap CI95 (dilaporkan apa adanya, boleh gagal)
  3. permutasi placebo (acak label gap)
  4. walk-forward (ambang dari paruh awal, diuji di paruh akhir)
  5. leave-one-day-out
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_rules import collect, rule_current  # noqa: E402
from scripts.replay_gates import DEFAULT_LOGS, _f, _merge  # noqa: E402

N_BOOT = 20000
N_PERM = 5000


def load_rows(since: str | None, until: str | None) -> list[dict[str, Any]]:
    """Pick tersettle (yang lolos rule terkirim) + konteks Elo/tanggal."""
    settles, _odds, newest = _merge(list(DEFAULT_LOGS))
    ctx: dict[str, dict[str, Any]] = {}
    for s in newest.values():
        f = s.get("features") or {}
        eh, ea = _f(f.get("elo_home")), _f(f.get("elo_away"))
        ctx[f"{s.get('home')} v {s.get('away')}"] = {
            "gap": abs(eh - ea) if (eh is not None and ea is not None) else None,
            "ko": str(s.get("kickoff") or "")[:10],
            "league": s.get("league"),
        }
    rows = [r for r in collect(list(DEFAULT_LOGS), since, until) if rule_current(r)]
    for r in rows:
        c = ctx.get(r["name"]) or {}
        r["gap"], r["ko"], r["league"] = c.get("gap"), c.get("ko"), c.get("league")
    return [r for r in rows if r["gap"] is not None and r["result"] in ("win", "loss")]


def _boot_ci(fn, n: int = N_BOOT) -> tuple[float, float, float]:
    vals = sorted(fn() for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)], mean(vals)


def evaluate(rows: list[dict[str, Any]], lo: float, hi: float, seed: int = 42) -> dict[str, Any]:
    random.seed(seed)

    def dead(r: dict[str, Any]) -> bool:
        return lo <= r["gap"] < hi

    inside = [r for r in rows if dead(r)]
    outside = [r for r in rows if not dead(r)]
    out: dict[str, Any] = {
        "n_total": len(rows), "lo": lo, "hi": hi,
        "n_inside": len(inside), "n_outside": len(outside),
    }
    if not inside or not outside:
        out["error"] = "salah satu sisi kosong"
        return out

    hi_in = sum(1 for r in inside if r["result"] == "win") / len(inside)
    hi_out = sum(1 for r in outside if r["result"] == "win") / len(outside)
    out["hit_inside"], out["hit_outside"] = hi_in, hi_out

    # 1. bootstrap hit rate
    def _hit_diff() -> float:
        a = mean(1 if random.choice(inside)["result"] == "win" else 0 for _ in inside)
        b = mean(1 if random.choice(outside)["result"] == "win" else 0 for _ in outside)
        return b - a

    l, h, m = _boot_ci(_hit_diff)
    out["hit_diff"] = {"mean": m, "ci_lo": l, "ci_hi": h, "layak": l > 0}

    # 2. bootstrap unit gain
    base = sum(r["units"] for r in rows)
    out["gain_units"] = sum(r["units"] for r in outside) - base

    def _u_diff() -> float:
        s = [random.choice(rows) for _ in rows]
        return sum(r["units"] for r in s if not dead(r)) - sum(r["units"] for r in s)

    l, h, m = _boot_ci(_u_diff)
    out["unit_gain"] = {"mean": m, "ci_lo": l, "ci_hi": h, "layak": l > 0}

    # 3. placebo permutation
    gaps = [r["gap"] for r in rows]
    hits = 0
    for _ in range(N_PERM):
        shuf = gaps[:]
        random.shuffle(shuf)
        fake = [dict(r, gap=g) for r, g in zip(rows, shuf)]
        fb = sum(r["units"] for r in fake)
        fk = sum(r["units"] for r in fake if not (lo <= r["gap"] < hi))
        if (fk - fb) >= out["gain_units"]:
            hits += 1
    out["placebo_p"] = hits / N_PERM

    # 4. walk-forward
    dates = sorted({r["ko"] for r in rows})
    if len(dates) >= 4:
        mid = dates[len(dates) // 2]
        train = [r for r in rows if r["ko"] <= mid]
        test = [r for r in rows if r["ko"] > mid]
        worst, worst_u = None, float("inf")
        for start in range(0, 400, 50):
            sub = [r for r in train if start <= r["gap"] < start + 100]
            if len(sub) >= 4:
                u = sum(r["units"] for r in sub)
                if u < worst_u:
                    worst_u, worst = u, start
        if worst is not None and test:
            td = [r for r in test if worst <= r["gap"] < worst + 100]
            out["walk_forward"] = {
                "split": mid, "train_worst_bucket": [worst, worst + 100],
                "train_units": worst_u, "test_dropped": len(td),
                "test_gain": -sum(r["units"] for r in td),
            }

    # 5. leave-one-day-out
    neg = []
    for d in sorted({r["ko"] for r in rows}):
        sub = [r for r in rows if r["ko"] != d]
        g = sum(r["units"] for r in sub if not dead(r)) - sum(r["units"] for r in sub)
        if g <= 0:
            neg.append(d)
    out["lodo_negative_days"] = neg

    out["dropped"] = [
        f"{r['name']} {r['market']} @{r['odds']:.2f} gap {r['gap']:.0f} "
        f"{r['result']} {r['units']:+.2f}u [{r['ko']}]" for r in inside
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--lo", type=float, default=100.0)
    ap.add_argument("--hi", type=float, default=200.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    rows = load_rows(a.since, a.until)
    if not rows:
        print("tidak ada pick tersettle dengan data Elo pada rentang ini")
        return 1
    res = evaluate(rows, a.lo, a.hi)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    print(f"=== G12 ZONA MAUT ELO [{a.lo:.0f}-{a.hi:.0f}] n={res['n_total']} ===\n")
    print(f"  di dalam zona : n={res['n_inside']:<3} hit {res['hit_inside']*100:5.1f}%")
    print(f"  di luar zona  : n={res['n_outside']:<3} hit {res['hit_outside']*100:5.1f}%")
    hd = res["hit_diff"]
    print(f"\n  1. HIT RATE (utama): luar lebih baik {hd['mean']*100:+.1f}pp "
          f"CI95[{hd['ci_lo']*100:+.1f},{hd['ci_hi']*100:+.1f}]pp -> "
          f"{'LAYAK' if hd['layak'] else 'TIDAK LAYAK'}")
    ug = res["unit_gain"]
    print(f"  2. UNIT      : gain {res['gain_units']:+.2f}u "
          f"CI95[{ug['ci_lo']:+.2f},{ug['ci_hi']:+.2f}]u -> "
          f"{'layak' if ug['layak'] else 'gagal (wajar: variance harga)'}")
    print(f"  3. PLACEBO   : p={res['placebo_p']:.4f} -> "
          f"{'BUKAN kebetulan' if res['placebo_p'] < 0.05 else 'bisa kebetulan'}")
    wf = res.get("walk_forward")
    if wf:
        print(f"  4. WALK-FWD  : bucket {wf['train_worst_bucket']} dari train<={wf['split']} "
              f"-> test gain {wf['test_gain']:+.2f}u ({wf['test_dropped']} dibuang) -> "
              f"{'KONFIRMASI' if wf['test_gain'] > 0 else 'GAGAL'}")
    neg = res["lodo_negative_days"]
    print(f"  5. LODO      : {len(neg)} tanggal membalik negatif -> "
          f"{'ROBUST' if not neg else 'RAPUH: ' + ', '.join(neg)}")
    print(f"\n  pick yang diveto ({res['n_inside']}):")
    for d in res["dropped"]:
        print(f"    - {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
