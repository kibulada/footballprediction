"""Replay the post-mortem rules over stored prediction logs (read-only).

Usage::

    python scripts/replay_gates.py [--log PATH ...] [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--json]

All ``--log`` files are MERGED first (settle rows unioned, newest snapshot
per canonical match across files), so a VPS sync and the local log can be
passed together without double counting.

For every settled BEST PICK the script re-applies the general rules to the
STORED fields (no engine re-run, no network) and reports, per class:

    LOSS caught / WIN affected / net units before -> after

Classes (2026-08-28): K1 both-Elo-prior directional, K2 Elo band, K2 G3
low-scoring, K3 tie state, K5 LEAN tier. Classes (2026-09-02): K6 Elo vs
Poisson direction conflict, K7 no conviction & no value (the tier rule B in
``signal_engine.pick_tier_for``), K8 stale hold, G11 Total/BTTS model
underdog + market underdog.

It also prints the BEST PICK-tier book "as published" vs "under rule B"
(kept W-L, demoted W-L with names), and reconstructs the SUGGESTION TO PICK
under the legacy always-on rule and the K4 ``market_lean`` rule.

Nothing is written unless ``--json`` is given (prints a JSON blob to stdout).
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
    lambda_direction_conflict,
    resolve_elo_band,
)
from agents.football.prediction_log import _match_dedupe_key, _pick_tier, _read_lines  # noqa: E402
from agents.football.signal_engine import pick_tier_for, settle_signal  # noqa: E402

DEFAULT_LOGS = [
    ROOT / "cache" / "football" / "predictions.jsonl",
    ROOT / "baseline" / "predictions_vps.jsonl",
]
MEDIUM_SCORE = 0.52
BEST_PICK_MARGIN = 0.06


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_cfg() -> dict[str, Any]:
    try:
        return json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- replay must run without config too
        return {}


def _both_prior(mp: dict[str, Any], f: dict[str, Any]) -> bool:
    hs, as_ = mp.get("elo_home_seeded"), mp.get("elo_away_seeded")
    if hs is not None and as_ is not None:
        return (not hs) and (not as_)
    eh, ea = _f(f.get("elo_home")), _f(f.get("elo_away"))
    return mp.get("elo_seeded") is False and eh == 1500.0 and ea == 1500.0


def _model_prob(s: dict[str, Any], pick: dict[str, Any]) -> tuple[float | None, float | None]:
    """(model_prob, edge_pp) of the stored pick; older rows via the ranking entry."""
    prob, edge = _f(pick.get("model_prob")), _f(pick.get("edge_pp"))
    if prob is None:
        for e in s.get("signal_engine_ranking") or []:
            if e.get("market") == pick.get("market") and e.get("selection") == pick.get("selection"):
                prob = _f(e.get("model_prob"))
                if edge is None:
                    edge = _f(e.get("edge_pp"))
                break
    return prob, edge


def classes_for_pick(
    s: dict[str, Any],
    pick: dict[str, Any],
    *,
    elo_band: tuple[float, float] = (DEFAULT_ELO_MIN, DEFAULT_ELO_MAX),
    tier_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Which post-mortem rules would have acted on this stored pick."""
    mp = s.get("model_probs") or {}
    f = s.get("features") or {}
    market, sel = pick.get("market"), pick.get("selection")
    out: list[str] = []
    eh, ea = _f(f.get("elo_home")), _f(f.get("elo_away"))
    lo, hi = elo_band
    if any(v is not None and (v < lo or v > hi) for v in (eh, ea)):
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
    # 2026-09-02 classes
    st = pick.get("stability") or {}
    sup = st.get("suppressed_top") or {}
    if (
        st.get("status") == "held" and sup.get("selection") and sup.get("selection") != sel
        and (_f(sup.get("score")) or 0.0) - score >= BEST_PICK_MARGIN
    ):
        out.append("K8-stale-hold")
    mp_k6 = dict(mp)
    if "lambda_home" not in mp_k6 and pick.get("lambda_home") is not None:
        mp_k6["lambda_home"], mp_k6["lambda_away"] = pick.get("lambda_home"), pick.get("lambda_away")
    if "1x2" not in mp_k6 and s.get("prob_1x2"):
        mp_k6["1x2"] = s.get("prob_1x2")
    if lambda_direction_conflict(mp_k6, market, sel, pick.get("side"))[0]:
        out.append("K6-lambda-conflict")
    prob, edge = _model_prob(s, pick)
    if prob is not None:
        probe = {"score": 1.0, "confidence": "MEDIUM", "model_prob": prob, "edge_pp": edge, "market": market, "selection": sel}
        if pick_tier_for(probe, tier_cfg)[0] == "LEAN":
            out.append("K7-conviction")
        if market in ("Total", "BTTS") and prob < 0.5 and edge is not None and edge < 0:
            out.append("G11-total-favor")
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


def _merge(paths: list[Path]) -> tuple[dict[str, dict], list[dict], dict[tuple, dict]]:
    """Union of settle rows + newest snapshot per canonical match across logs."""
    settles: dict[str, dict[str, Any]] = {}
    odds_snaps: list[dict[str, Any]] = []
    newest: dict[tuple, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        rows = _read_lines(path)
        for r in rows:
            ev = r.get("event")
            if ev == "settle":
                settles.setdefault(r["match_id"], r)
            elif ev == "odds_snapshot":
                odds_snaps.append(r)
        for s in rows:
            if s.get("event") != "snapshot":
                continue
            key = _match_dedupe_key(s)
            cur = newest.get(key)
            if cur is None or (s.get("ts") or "") > (cur.get("ts") or ""):
                newest[key] = s
    newest = {k: s for k, s in newest.items() if s["match_id"] in settles}
    return settles, odds_snaps, newest


def replay(paths: list[Path], since: str | None, until: str | None) -> dict[str, Any]:
    cfg = _load_cfg()
    se_cfg = ((cfg.get("models") or {}).get("signal_engine") or {})
    pg_cfg = se_cfg.get("pick_gates") or {}
    label = f"{since or 'start'}..{until or 'end'}"
    settles, odds_snaps, newest = _merge(paths)
    report: dict[str, Any] = {"period": label, "best_pick": {}, "tier_rule_b": {}, "suggestion": {}}
    bucket = defaultdict(lambda: {"loss_caught": 0, "win_affected": 0, "units_removed": 0.0})
    summary = {"n": 0, "w": 0, "l": 0, "units": 0.0}
    tier_book = {"as_published": {"w": 0, "l": 0}, "kept": {"w": 0, "l": 0}, "demoted": {"w": 0, "l": 0},
                 "demoted_wins": [], "kept_losses": []}
    sb = {"legacy": {"n": 0, "w": 0, "l": 0, "units": 0.0}, "k4": {"n": 0, "w": 0, "l": 0, "units": 0.0}}
    for s in newest.values():
        ko = str(s.get("kickoff") or "")[:10]
        if since and ko < since:
            continue
        if until and ko > until:
            continue
        st = settles[s["match_id"]]
        hg, ag = int(st.get("home_goals") or 0), int(st.get("away_goals") or 0)
        pick = s.get("signal_engine_pick") or s.get("best_pick")
        if pick:
            res = settle_signal(pick, hg, ag)["result"]
            odds = _f(pick.get("market_odds")) or 0.0
            unit = (odds - 1.0) if res == "win" else (-1.0 if res == "loss" else 0.0)
            summary["n"] += 1
            summary["w"] += res == "win"
            summary["l"] += res == "loss"
            summary["units"] += unit if odds > 1.0 else 0.0
            band = resolve_elo_band(pg_cfg, s.get("league"))
            for c in classes_for_pick(s, pick, elo_band=band, tier_cfg=se_cfg):
                b = bucket[c]
                if res == "loss":
                    b["loss_caught"] += 1
                elif res == "win":
                    b["win_affected"] += 1
                b["units_removed"] += unit if odds > 1.0 else 0.0
            # BEST PICK-tier book, as published vs rule B
            if _pick_tier(pick, MEDIUM_SCORE) == "BEST PICK" and res in ("win", "loss"):
                tier_book["as_published"]["w" if res == "win" else "l"] += 1
                prob, edge = _model_prob(s, pick)
                probe = {"score": pick.get("score"), "confidence": pick.get("confidence"),
                         "model_prob": prob, "edge_pp": edge,
                         "market": pick.get("market"), "selection": pick.get("selection")}
                keep = pick_tier_for(probe, se_cfg)[0] == "BEST PICK"
                name = f"{s.get('home')} v {s.get('away')} {pick.get('market')} {pick.get('selection')}"
                if keep:
                    tier_book["kept"]["w" if res == "win" else "l"] += 1
                    if res == "loss":
                        tier_book["kept_losses"].append(f"{name} (prob {prob}, edge {edge})")
                else:
                    tier_book["demoted"]["w" if res == "win" else "l"] += 1
                    if res == "win":
                        tier_book["demoted_wins"].append(f"{name} (prob {prob}, edge {edge}, @{odds})")
        totals = s.get("market_totals") or {}
        consensus = s.get("odds_1x2") or {}
        ah = _nearest_ah(odds_snaps, s["match_id"], s.get("ts") or "")
        leg = legacy_suggestion(totals, consensus, ah)
        stored = (s.get("suggestion") or {}).get("pick")
        new = stored if stored is not None else compute_suggestion(
            totals=totals, consensus=consensus, ah=ah,
            model_probs=s.get("model_probs"), features=s.get("features"),
            tie_state=((s.get("context_data") or {}).get("tie_state")),
            ranking=s.get("signal_engine_ranking"),
        )["pick"]
        for lab, p in (("legacy", leg), ("k4", new)):
            if not p:
                continue
            sig = suggestion_for_settlement(p)
            r = settle_signal(sig, hg, ag)["result"] if sig else "n/a"
            if r == "n/a":
                continue
            o = _f(p.get("odds")) or 0.0
            sb[lab]["n"] += 1
            sb[lab]["w"] += r in ("win", "half_win")
            sb[lab]["l"] += r in ("loss", "half_loss")
            sb[lab]["units"] += (o - 1.0) if r == "win" else (-1.0 if r == "loss" else 0.0)
    report["best_pick"] = {"summary": summary, "classes": dict(bucket)}
    report["tier_rule_b"] = tier_book
    report["suggestion"] = sb
    return report


def _print(report: dict[str, Any]) -> None:
    per = report["period"]
    summ = report["best_pick"]["summary"]
    print("=== BEST PICK: rule impact on stored picks (LOSS caught / WIN affected / units removed) ===")
    print(f"\n[{per}] n={summ['n']} W={summ['w']} L={summ['l']} net={summ['units']:+.2f}u (as published, all tiers)")
    for cls, b in sorted(report["best_pick"]["classes"].items()):
        print(f"  {cls:28} loss_caught={b['loss_caught']:2d} win_affected={b['win_affected']:2d} "
              f"units_removed={b['units_removed']:+.2f}")
    tb = report["tier_rule_b"]
    ap, kp, dm = tb["as_published"], tb["kept"], tb["demoted"]

    def _wr(x):
        n = x["w"] + x["l"]
        return f"{x['w']}-{x['l']} [{(x['w'] / n * 100) if n else 0:.0f}%]"

    print("\n=== BEST PICK tier: as published vs rule B (prob >= 0.60 OR prob >= 0.50 & edge >= 0) ===")
    print(f"[{per}] as published {_wr(ap)} | rule B keeps {_wr(kp)} | demoted to LEAN {_wr(dm)}")
    for x in tb["demoted_wins"]:
        print(f"    demoted win : {x}")
    for x in tb["kept_losses"]:
        print(f"    kept loss   : {x}")
    print("\n=== SUGGESTION: legacy (always-on max implied) vs K4 (market_lean / stored) ===")
    for label in ("legacy", "k4"):
        x = report["suggestion"][label]
        wr = (x["w"] / x["n"] * 100.0) if x["n"] else 0.0
        print(f"[{per}] {label:6} n={x['n']:3d} W={x['w']:3d} L={x['l']:3d} win-rate={wr:5.1f}% net={x['units']:+.2f}u")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", type=Path, help="prediction log (repeatable; merged)")
    ap.add_argument("--since", help="only kickoffs on/after YYYY-MM-DD")
    ap.add_argument("--until", help="only kickoffs on/before YYYY-MM-DD")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()
    paths = args.log or DEFAULT_LOGS
    report = replay(paths, args.since, args.until)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
