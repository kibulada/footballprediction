#!/usr/bin/env python3
"""Compare OLD vs NEW picks — re-scored with Fix 2-5 post-scoring adjustments."""
import json, sys, os, math
from pathlib import Path
sys.path.insert(0, os.getcwd())

from agents.football.prediction_log import _read_lines, _match_dedupe_key
from agents.football.signal_engine import ah_return

PATH = Path("cache/football/predictions.jsonl")


def full_evaluate(pick, hg, ag):
    if not pick:
        return "?", 0.0
    market = pick.get("market", "")
    selection = pick.get("selection", "")
    line = pick.get("line")
    side = pick.get("side")
    outcome = "home" if hg > ag else ("draw" if hg == ag else "away")
    total = hg + ag
    diff = hg - ag

    if market == "1X2":
        sel_outcome = "home" if "Home" in selection else ("draw" if "Draw" in selection else "away")
        return ("win", 1.0) if sel_outcome == outcome else ("loss", 0.0)
    if market == "Total":
        import re
        is_over = "Over" in selection
        if line is not None:
            ln = float(line)
        else:
            m = re.search(r'(\d+\.?\d*)', selection)
            ln = float(m.group(1)) if m else 2.5
        hit = total > ln if is_over else total < ln
        return ("win", 1.0) if hit else ("loss", 0.0)
    if market == "BTTS":
        both = hg > 0 and ag > 0
        hit = ("Yes" in selection) == both
        return ("win", 1.0) if hit else ("loss", 0.0)
    if market == "Asian Handicap":
        if line is not None:
            handicap = float(line)
        else:
            import re
            m = re.search(r'([+-]?\d+\.?\d*)', selection)
            handicap = float(m.group(1)) if m else 0.0
        if side is None:
            side = "home" if "Home" in selection else "away"
        ret = ah_return(hg, ag, handicap, side)
        if ret >= 1.0 - 1e-9: return "win", 1.0
        elif abs(ret - 0.75) < 1e-9: return "half_win", 0.75
        elif abs(ret - 0.5) < 1e-9: return "push", 0.5
        elif abs(ret - 0.25) < 1e-9: return "half_loss", 0.25
        elif ret <= 1e-9: return "loss", 0.0
        return f"?{ret:.2f}", ret
    return "?", 0.0


def result_icon(label):
    return {"win": "✅", "half_win": "🟢½W", "push": "🟡P",
            "half_loss": "🔴½L", "loss": "❌"}.get(str(label), "⚪")


def rescore_ranking_with_fixes(ranking, prob_1x2, lambda_home, lambda_away):
    """Re-score ranking entries with Fix 2-5 post-scoring adjustments."""
    if not ranking:
        return None

    p_home = float(prob_1x2.get("home", 0.33))
    p_away = float(prob_1x2.get("away", 0.33))
    model_direction = "home" if p_home > p_away else ("away" if p_away > p_home else "draw")
    max_prob = max(p_home, p_away)
    is_decisive = max_prob > 0.60
    lambda_total = (lambda_home or 0) + (lambda_away or 0)

    scored = []
    for entry in ranking:
        market = entry.get("market", "")
        selection = entry.get("selection", "")
        odds = entry.get("market_odds")
        line = entry.get("line")
        side = entry.get("side")
        edge_pp = entry.get("edge_pp", 0) or 0
        model_prob = entry.get("model_prob", 0) or 0

        # Fix 2b: ANY pick without odds → edge not market-validated
        no_odds = not odds or float(odds) <= 1.0
        if no_odds:
            edge_pp = edge_pp * 0.70

        # Score: edge (35%) + ev (30%) + model_prob (35%)
        ev = (model_prob * float(odds) - 1.0) if odds and float(odds) > 1.0 and model_prob > 0 else -1.0
        score = edge_pp * 0.35 + (ev * 100 * 0.30) + (model_prob * 100 * 0.35)

        # Fix 2: AH without odds → conservative floor
        if market == "Asian Handicap" and no_odds:
            if model_prob >= 0.70: score = max(score, 40)
            elif model_prob >= 0.55: score = max(score, 30)
            else: score = max(score, 20)

        # Fix 3: Direction — AH contradicting model
        if market == "Asian Handicap" and side and model_direction != "draw":
            if side != model_direction and max_prob > 0.60:
                score *= 0.92

        # Fix 4: Decisive match — penalize half-win markets
        if is_decisive:
            if market == "Asian Handicap" and line is not None:
                is_qh = abs(line % 0.5) > 1e-9 or (abs(line % 0.5) < 1e-9 and abs(line) % 1.0 > 1e-9)
                if is_qh: score *= 0.92
            elif market == "Total" and selection and selection.startswith("Under"):
                score *= 0.95

        # Fix 5: High-scoring → boost Over, penalize Under
        if lambda_total > 2.5 and market == "Total" and selection:
            if selection.startswith("Over"):
                score *= 1.08 if lambda_total > 3.0 else 1.04
            elif selection.startswith("Under"):
                score *= 0.94

        # Fix 6: Low-scoring → boost Under, penalize Over
        if lambda_total < 2.0 and market == "Total" and selection:
            if selection.startswith("Under"):
                score *= 1.05
            elif selection.startswith("Over"):
                score *= 0.92

        # Fix 7: BTTS in high-scoring → boost Yes
        if lambda_total > 2.5 and market == "BTTS" and selection and "Yes" in selection:
            score *= 1.03

        # Fix 8: BTTS in low-scoring → boost No
        if lambda_total < 2.0 and market == "BTTS" and selection and "No" in selection:
            score *= 1.03

        scored.append({
            "market": market, "selection": selection, "line": line,
            "side": side, "market_odds": odds, "model_prob": model_prob,
            "edge_pp": edge_pp, "score": score,
        })

    if not scored:
        return None
    scored.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
    return scored[0]


def main():
    rows = _read_lines(PATH)
    settlements = {r["match_id"]: r for r in rows if r.get("event") == "settle"}
    snapshots = [r for r in rows if r.get("event") == "snapshot"]
    newest: dict = {}
    for s in snapshots:
        if settlements.get(s.get("match_id")) is None:
            continue
        key = _match_dedupe_key(s)
        cur = newest.get(key)
        if cur is None or (s.get("ts") or "") > (cur.get("ts") or ""):
            newest[key] = s

    results = []
    for key, snap in newest.items():
        st = settlements[snap["match_id"]]
        hg = int(st.get("home_goals", 0))
        ag = int(st.get("away_goals", 0))
        outcome = "home" if hg > ag else ("draw" if hg == ag else "away")
        league = snap.get("league", "?")
        home = snap.get("home", "?")
        away = snap.get("away", "?")
        kickoff = snap.get("kickoff") or ""

        se_pick = snap.get("signal_engine_pick") or snap.get("best_pick") or {}
        ranking = snap.get("signal_engine_ranking") or []
        prob = snap.get("prob_1x2") or {}
        features = snap.get("features") or {}
        lambda_h = features.get("lambda_home") or prob.get("lambda_home", 0)
        lambda_a = features.get("lambda_away") or prob.get("lambda_away", 0)

        new_pick = rescore_ranking_with_fixes(ranking, prob, lambda_h, lambda_a)

        old_label, old_ret = full_evaluate(se_pick, hg, ag)
        new_label, new_ret = full_evaluate(new_pick, hg, ag) if new_pick else ("?", 0.0)

        old_odds = se_pick.get("market_odds") if se_pick and se_pick.get("market") else None
        new_odds = new_pick.get("market_odds") if new_pick and new_pick.get("market") else None

        results.append({
            "league": league, "home": home, "away": away,
            "hg": hg, "ag": ag, "outcome": outcome, "kickoff": kickoff,
            "old_pick": se_pick, "old_label": old_label, "old_ret": old_ret, "old_odds": old_odds,
            "new_pick": new_pick, "new_label": new_label, "new_ret": new_ret, "new_odds": new_odds,
        })

    results.sort(key=lambda x: x["kickoff"] or "", reverse=True)

    current_date = None
    row_num = 0
    totals = {"old": {"w":0,"hw":0,"p":0,"hl":0,"l":0,"na":0,"roi":0.0,"n_roi":0},
              "new": {"w":0,"hw":0,"p":0,"hl":0,"l":0,"na":0,"roi":0.0,"n_roi":0}}

    def print_header():
        nonlocal row_num
        print(f"\n  {'─'*175}")
        print(f"  📅 {current_date}")
        print(f"  {'─'*175}")
        print(f"  {'#':>2} {'League':<12} {'Match':<38} {'Sc':>5} │ {'OLD PICK':<32} {'Result':>8} │ {'NEW PICK':<32} {'Result':>8} │ {'CHG':>5}")
        print(f"  {'─'*175}")
        row_num = 0

    for m in results:
        kick_date = m["kickoff"][:10] if m["kickoff"] else "Unknown"
        if kick_date != current_date:
            current_date = kick_date
            print_header()

        row_num += 1
        match_str = f"{m['home']} vs {m['away']}"

        op = m["old_pick"]
        if op and op.get("market"):
            odds_s = f" @{m['old_odds']:.2f}" if m['old_odds'] and float(m['old_odds']) > 1.0 else ""
            old_s = f"{op['market']} {op['selection']}{odds_s}"
        else:
            old_s = "NO DATA"

        np_ = m["new_pick"]
        if np_ and np_.get("market"):
            odds_s = f" @{m['new_odds']:.2f}" if m['new_odds'] and float(m['new_odds']) > 1.0 else ""
            new_s = f"{np_['market']} {np_['selection']}{odds_s}"
        else:
            new_s = "NO DATA"

        old_icon = result_icon(m["old_label"])
        new_icon = result_icon(m["new_label"])
        old_r = f"{old_icon}{m['old_label']}"
        new_r = f"{new_icon}{m['new_label']}"

        is_old_win = m["old_label"] in ("win", "half_win")
        is_old_loss = m["old_label"] in ("loss", "half_loss")
        is_new_win = m["new_label"] in ("win", "half_win")
        is_new_loss = m["new_label"] in ("loss", "half_loss")

        if is_old_loss and is_new_win: change = "⬆FIX"
        elif is_old_win and is_new_loss: change = "⬇REG"
        elif m["old_label"] != m["new_label"]: change = "~"
        else: change = "="

        print(f"  {row_num:>2} {m['league']:<12} {match_str:<38} {m['hg']}-{m['ag']:>2} │ {old_s:<32} {old_r:>8} │ {new_s:<32} {new_r:>8} │ {change:>5}")

        for side_key, label, ret_val, odds in [("old", m["old_label"], m["old_ret"], m["old_odds"]),
                                                 ("new", m["new_label"], m["new_ret"], m["new_odds"])]:
            t = totals[side_key]
            if label == "win": t["w"] += 1
            elif label == "half_win": t["hw"] += 1
            elif label == "push": t["p"] += 1
            elif label == "half_loss": t["hl"] += 1
            elif label == "loss": t["l"] += 1
            else: t["na"] += 1
            if odds and float(odds) > 1.0:
                t["n_roi"] += 1
                t["roi"] += (ret_val * float(odds) - 1.0) if ret_val > 0 else -1.0

    print(f"\n{'='*175}")
    print(f"  📊 GRAND TOTAL: OLD MODEL vs NEW MODEL (with Fix 2-5)")
    print(f"{'='*175}")

    for label, sk in [("OLD MODEL", "old"), ("NEW MODEL", "new")]:
        t = totals[sk]
        decided = t["w"] + t["hw"] + t["p"] + t["hl"] + t["l"]
        win_equiv = t["w"] + 0.5*t["hw"] + 0.5*t["p"] + 0.25*t["hl"]
        wr = win_equiv / decided * 100 if decided > 0 else 0
        roi = t["roi"] / t["n_roi"] * 100 if t["n_roi"] > 0 else 0
        print(f"\n  {label}:")
        print(f"    Wins: {t['w']} | Half Wins: {t['hw']} | Pushes: {t['p']} | Half Losses: {t['hl']} | Losses: {t['l']} | N/A: {t['na']}")
        print(f"    Decided: {decided} | Win Equiv: {win_equiv:.1f} | Win Rate: {wr:.1f}% | ROI: {roi:.1f}% ({t['n_roi']} bets)")

    fixes = sum(1 for m in results if m["old_label"] in ("loss","half_loss") and m["new_label"] in ("win","half_win"))
    regs = sum(1 for m in results if m["old_label"] in ("win","half_win") and m["new_label"] in ("loss","half_loss"))
    print(f"\n  CHANGES: ⬆ Fixes: {fixes} | ⬇ Regressions: {regs} | Net: {fixes-regs:+d}")
    print(f"{'='*175}")


if __name__ == "__main__":
    main()
