"""CLI multi-bookmaker odds: line shopping + edge dari devig Pinnacle.

READ-ONLY, tanpa API key. Sumber: football-data.co.uk.

Usage::

    python scripts/odds_shop.py fixtures [--league EPL]
    python scripts/odds_shop.py match --home Gent --away "Oud-Heverlee Leuven"
    python scripts/odds_shop.py backtest [--seasons 2526,2425,2324]

``backtest`` membuktikan nilai line shopping: strategi & pilihan IDENTIK,
hanya beda tempat beli.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.football import multi_odds as mo  # noqa: E402

DEFAULT_LEAGUES = [
    "EPL", "EFL Championship", "LaLiga", "Serie A", "Bundesliga", "Ligue 1",
    "Eredivisie", "Belgian Pro League", "Primeira Liga", "Super Lig",
    "Scottish Premiership",
]


def cmd_fixtures(a) -> int:
    fx = mo.get_fixtures(a.league)
    if not fx:
        print("tidak ada fixture (sumber tidak terjangkau atau liga kosong)")
        return 1
    if a.json:
        print(json.dumps(fx, indent=2, ensure_ascii=False))
        return 0
    print(f"=== {len(fx)} FIXTURE MENDATANG ===\n")
    for f in fx:
        print(mo.summary(f))
        pos = [e for e in (f.get("edges") or []) if e["edge_pct"] > 0]
        if pos:
            best = max(pos, key=lambda e: e["edge_pct"])
            print(f"  >>> value terbaik: {best['outcome']} @ {best['best_odds']} "
                  f"({best['best_book']}) edge {best['edge_pct']:+.2f}%")
        print()
    return 0


def cmd_match(a) -> int:
    fx = mo.find_fixture(a.home, a.away, a.league)
    if not fx:
        print(f"fixture tidak ketemu: {a.home} v {a.away}")
        return 1
    if a.json:
        print(json.dumps(fx, indent=2, ensure_ascii=False))
        return 0
    print(mo.summary(fx))
    print("\n  harga per bandar (1X2):")
    for name, trio in sorted(fx["books"].items()):
        vals = "  ".join(f"{v:.2f}" if v else "  -  " for v in trio)
        print(f"    {name:<32} {vals}")
    t = fx.get("totals") or {}
    if t.get("over_2.5_best"):
        print(f"\n  O/U 2.5 terbaik: Over {t['over_2.5_best']} / Under {t['under_2.5_best']}"
              f"   (rata-rata {t.get('over_2.5_avg')} / {t.get('under_2.5_avg')})")
    return 0


def cmd_backtest(a) -> int:
    seasons = [s.strip() for s in a.seasons.split(",") if s.strip()]
    tot = {"avg": 0.0, "pin": 0.0, "best": 0.0}
    n = hit = 0
    gap = 0.0
    per_season: dict[str, tuple[int, dict[str, float]]] = {}
    for season in seasons:
        s_tot = {"avg": 0.0, "pin": 0.0, "best": 0.0}
        s_n = 0
        for lg in DEFAULT_LEAGUES:
            for m in mo.get_historical(lg, season):
                bc = m["books_close"]
                avg = (bc or {}).get("Rata-rata pasar")
                if not avg or any(x is None for x in avg):
                    continue
                if m["ftr"] not in ("H", "D", "A"):
                    continue
                out = {"H": 0, "D": 1, "A": 2}[m["ftr"]]
                k = min(range(3), key=lambda i: avg[i])   # pilihan identik
                bp = mo.best_price(bc, k)
                if bp is None:
                    continue
                pin = (bc.get("Pinnacle") or [None, None, None])[k] or avg[k]
                won = k == out
                n += 1; s_n += 1; hit += won
                gap += (bp[0] - avg[k]) / avg[k] * 100
                for key, price in (("avg", avg[k]), ("pin", pin), ("best", bp[0])):
                    d = (price - 1.0) if won else -1.0
                    tot[key] += d
                    s_tot[key] += d
        per_season[season] = (s_n, s_tot)
    if not n:
        print("tidak ada data historis terambil")
        return 1
    print(f"=== BACKTEST LINE SHOPPING — {n} taruhan ===")
    print(f"hit rate (identik di ketiganya): {hit/n*100:.1f}%")
    print(f"keunggulan harga terbaik vs rata-rata: {gap/n:+.2f}% per taruhan\n")
    for key, label in (("avg", "harga rata-rata (SEKARANG)"),
                       ("pin", "harga Pinnacle"),
                       ("best", "harga bandar TERBAIK")):
        print(f"  {label:<28} {tot[key]:>+11.2f}u  ROI {tot[key]/n*100:>+6.2f}%")
    print(f"\n  PERBAIKAN: {tot['best']-tot['avg']:+.2f}u "
          f"({(tot['best']-tot['avg'])/n*100:+.2f}% per taruhan)")
    print("\n=== PER MUSIM (out-of-sample) ===")
    for s, (sn, st) in sorted(per_season.items()):
        if sn < 50:
            continue
        print(f"  {s}: n={sn:<5} {st['avg']:+8.2f}u -> {st['best']:+8.2f}u  "
              f"selisih {st['best']-st['avg']:+7.2f}u ({(st['best']-st['avg'])/sn*100:+.2f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixtures"); p.add_argument("--league"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_fixtures)
    p = sub.add_parser("match"); p.add_argument("--home", required=True)
    p.add_argument("--away", required=True); p.add_argument("--league")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_match)
    p = sub.add_parser("backtest"); p.add_argument("--seasons", default="2526,2425,2324")
    p.set_defaults(fn=cmd_backtest)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
