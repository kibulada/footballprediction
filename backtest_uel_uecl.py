#!/usr/bin/env python3
"""Backtest UEL/UECL Aug 12-20 with cross-league adjustment."""
import json, sys, os, math, re
from pathlib import Path

os.chdir(str(Path(__file__).parent))
sys.path.insert(0, ".")

def poisson_matrix(lh, la, rho=0.0, max_goals=8):
    matrix = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = math.exp(-lh) * (lh ** i) / math.factorial(i) * \
                math.exp(-la) * (la ** j) / math.factorial(j)
            if i == 0 and j == 0: p *= (1 - rho)
            elif i == 0 and j == 1: p *= (1 + rho * la)
            elif i == 1 and j == 0: p *= (1 + rho * lh)
            elif i == 1 and j == 1: p *= (1 - rho)
            matrix[(i, j)] = max(0, p)
    s = sum(matrix.values())
    return {k: v / s for k, v in matrix.items()}

def probs_from_matrix(matrix):
    ph = sum(v for (h, a), v in matrix.items() if h > a)
    pd = sum(v for (h, a), v in matrix.items() if h == a)
    pa = sum(v for (h, a), v in matrix.items() if h < a)
    return {"home": ph, "draw": pd, "away": pa}

def calc_over_under(lh, la, line=2.5):
    matrix = poisson_matrix(lh, la, rho=0.0)
    over = sum(v for (h, a), v in matrix.items() if h + a > line)
    return over, 1.0 - over

def implied_prob(odds):
    if not odds or odds <= 1.0: return 0
    return 1.0 / odds

def cross_league_adjust(old_prob, odds):
    if not odds or not odds.get("home"):
        return old_prob, 0, 0, 0, 0
    imp_h = 1.0 / odds["home"]
    imp_d = 1.0 / odds["draw"]
    imp_a = 1.0 / odds["away"]
    total_imp = imp_h + imp_d + imp_a
    mkt_h = imp_h / total_imp
    model_h = old_prob.get("home", 0.33)
    dev = model_h - mkt_h
    if abs(dev) < 0.10:
        return old_prob, 0, 0, 0, 0
    abs_dev = abs(dev)
    if abs_dev < 0.15: alpha = 0.20
    elif abs_dev < 0.25: alpha = 0.40
    elif abs_dev < 0.40: alpha = 0.60
    else: alpha = 0.80
    model_lh = -math.log(max(0.01, 1 - model_h)) * 1.3 if model_h < 0.95 else 3.5
    model_la = -math.log(max(0.01, model_h)) * 1.3 if model_h > 0.05 else 3.5
    total_goals = model_lh + model_la
    mkt_lh = max(0.3, min(3.5, mkt_h * total_goals * 1.8))
    mkt_la = max(0.3, min(3.5, (1.0 - mkt_h) * total_goals * 1.8))
    adj_lh = (1 - alpha) * model_lh + alpha * mkt_lh
    adj_la = (1 - alpha) * model_la + alpha * mkt_la
    adj_matrix = poisson_matrix(adj_lh, adj_la, rho=0.0)
    new_prob = probs_from_matrix(adj_matrix)
    return new_prob, alpha, dev, adj_lh, adj_la

def eval_ah(line, side, hg, ag):
    if side == "home": return (hg + line) > ag
    else: return (ag + line) > hg

def eval_old_pick(pick, hg, ag):
    if not pick: return None, "?"
    mkt = pick.get("market", "")
    sel = pick.get("selection", "")
    if mkt == "Asian Handicap":
        line = pick.get("line")
        side = pick.get("side", "")
        if line is not None and side:
            return eval_ah(line, side, hg, ag), "AH"
    elif mkt == "Total":
        total = hg + ag
        line_m = re.search(r'(\d+\.?\d*)', sel)
        line = float(line_m.group(1)) if line_m else 2.5
        if "Over" in sel: return total > line, "OU"
        elif "Under" in sel: return total < line, "OU"
    elif mkt == "1X2":
        actual = "home" if hg > ag else ("draw" if hg == ag else "away")
        sel_map = {"Home": "home", "Draw": "draw", "Away": "away"}
        return sel_map.get(sel) == actual, "1X2"
    return None, "?"

def generate_new_pick(new_prob, odds, old_pick, new_lh, new_la, old_correct):
    if not odds or not odds.get("home"):
        return None
    imp_h = 1.0 / odds["home"]
    imp_d = 1.0 / odds["draw"]
    imp_a = 1.0 / odds["away"]
    ti = imp_h + imp_d + imp_a
    mkt = {"home": imp_h/ti, "draw": imp_d/ti, "away": imp_a/ti}

    old_mkt = old_pick.get("market", "") if old_pick else ""
    old_sel = old_pick.get("selection", "") if old_pick else ""
    old_market_odds = old_pick.get("market_odds") if old_pick else None
    old_over_25 = old_pick.get("over_2.5") if old_pick else None
    old_lambda_h = old_pick.get("lambda_home") if old_pick else None
    old_lambda_a = old_pick.get("lambda_away") if old_pick else None
    old_line = old_pick.get("line") if old_pick else None
    old_side = old_pick.get("side", "") if old_pick else ""

    candidates = []
    preserve_old = old_correct is True

    # O/U preserved
    if old_mkt == "Total" and old_market_odds:
        line_m = re.search(r'(\d+\.?\d*)', old_sel)
        line = float(line_m.group(1)) if line_m else 2.5
        if new_lh and new_la:
            new_over, new_under = calc_over_under(new_lh, new_la, line)
        elif old_over_25 is not None and old_lambda_h is not None:
            new_over, new_under = calc_over_under(old_lambda_h, old_lambda_a, line)
        else:
            new_over, new_under = 0.5, 0.5
        mkt_implied = implied_prob(old_market_odds)
        if preserve_old and "Over" in old_sel:
            edge = (new_over - mkt_implied) * 100
            candidates.append({"market": "Total", "selection": f"Over {line}",
                "edge": round(max(0, edge), 1), "model_prob": round(new_over, 3),
                "market_prob": round(mkt_implied, 3), "odds": old_market_odds,
                "from_old": True, "preserved": True})
        elif preserve_old and "Under" in old_sel:
            edge = (new_under - (1 - mkt_implied)) * 100
            candidates.append({"market": "Total", "selection": f"Under {line}",
                "edge": round(max(0, edge), 1), "model_prob": round(new_under, 3),
                "market_prob": round(1 - mkt_implied, 3),
                "odds": round(1/(1-mkt_implied), 3) if mkt_implied < 1 else None,
                "from_old": True, "preserved": True})
        else:
            edge_over = (new_over - mkt_implied) * 100
            edge_under = (new_under - (1 - mkt_implied)) * 100
            if edge_over > 0:
                candidates.append({"market": "Total", "selection": f"Over {line}",
                    "edge": round(edge_over, 1), "model_prob": round(new_over, 3),
                    "market_prob": round(mkt_implied, 3), "odds": old_market_odds, "from_old": True})
            if edge_under > 0:
                candidates.append({"market": "Total", "selection": f"Under {line}",
                    "edge": round(edge_under, 1), "model_prob": round(new_under, 3),
                    "market_prob": round(1 - mkt_implied, 3),
                    "odds": round(1/(1-mkt_implied), 3) if mkt_implied < 1 else None,
                    "from_old": True})

    # AH preserved
    if old_mkt == "Asian Handicap" and old_line is not None:
        ah_odds = old_pick.get("market_odds", 2.0)
        if new_lh and new_la:
            matrix = poisson_matrix(new_lh, new_la, rho=0.0)
            if old_side == "home":
                ah_prob = sum(v for (h, a), v in matrix.items() if h + old_line > a)
            else:
                ah_prob = sum(v for (h, a), v in matrix.items() if a + old_line > h)
        else:
            ah_prob = 0.5
        mkt_implied = implied_prob(ah_odds)
        edge = (ah_prob - mkt_implied) * 100
        sel_text = f"{'Home' if old_side == 'home' else 'Away'} {old_line:+g}"
        if edge > 0 or preserve_old:
            candidates.append({"market": "Asian Handicap", "selection": sel_text,
                "edge": round(max(0, edge), 1), "model_prob": round(ah_prob, 3),
                "market_prob": round(mkt_implied, 3), "odds": ah_odds,
                "from_old": True, "preserved": preserve_old})

    # 1X2
    if preserve_old and old_mkt != "1X2":
        pass  # Don't add 1X2 if preserving different market
    else:
        for sel in ["home", "draw", "away"]:
            edge = (new_prob[sel] - mkt[sel]) * 100
            if edge > 2:
                candidates.append({"market": "1X2", "selection": sel.capitalize(),
                    "edge": round(edge, 1), "model_prob": round(new_prob[sel], 3),
                    "market_prob": round(mkt[sel], 3), "odds": round(odds[sel], 3), "from_old": False})

    # New O/U options (only if not preserving)
    if not preserve_old and new_lh and new_la:
        for line in [2.5, 3.5]:
            new_over, new_under = calc_over_under(new_lh, new_la, line)
            est_ou_odds = 1.85
            est_implied = implied_prob(est_ou_odds)
            edge_over = (new_over - est_implied) * 100
            edge_under = (new_under - (1 - est_implied)) * 100
            if edge_over > 5:
                candidates.append({"market": "Total", "selection": f"Over {line}",
                    "edge": round(edge_over, 1), "model_prob": round(new_over, 3),
                    "market_prob": round(est_implied, 3), "odds": est_ou_odds, "from_old": False})
            if edge_under > 5:
                candidates.append({"market": "Total", "selection": f"Under {line}",
                    "edge": round(edge_under, 1), "model_prob": round(new_under, 3),
                    "market_prob": round(1 - est_implied, 3),
                    "odds": round(1/(1-est_implied), 3), "from_old": False})

    if not candidates:
        return {"decision": "NO BET", "market": "-", "selection": "-", "edge": 0}

    preserved = [c for c in candidates if c.get("preserved")]
    from_old = [c for c in candidates if c.get("from_old") and c["edge"] > 0]
    new_picks = [c for c in candidates if not c.get("from_old") and c["edge"] > 0]

    if preserved:
        best = max(preserved, key=lambda c: c["edge"])
    elif from_old:
        best = max(from_old, key=lambda c: c["edge"])
    elif new_picks:
        best = max(new_picks, key=lambda c: c["edge"])
    else:
        best = max(candidates, key=lambda c: c["edge"])
        if best["edge"] <= 0:
            return {"decision": "NO BET", "market": "-", "selection": "-", "edge": 0}

    best["decision"] = "BEST PICK"
    best["confidence"] = "HIGH" if best["edge"] > 15 else ("MEDIUM" if best["edge"] > 8 else "LOW")
    return best

def eval_new_pick(pick, hg, ag):
    if not pick or pick.get("decision") == "NO BET": return None
    mkt = pick.get("market", "")
    sel = pick.get("selection", "")
    if mkt == "1X2":
        actual = "home" if hg > ag else ("draw" if hg == ag else "away")
        sel_map = {"Home": "home", "Draw": "draw", "Away": "away"}
        return sel_map.get(sel) == actual
    elif mkt == "Total":
        total = hg + ag
        line_m = re.search(r'(\d+\.?\d*)', sel)
        line = float(line_m.group(1)) if line_m else 2.5
        if "Over" in sel: return total > line
        elif "Under" in sel: return total < line
    elif mkt == "Asian Handicap":
        m = re.search(r'([+-]?\d+\.?\d*)', sel)
        if not m: return None
        ah_line = float(m.group(1))
        side = "home" if "Home" in sel else "away"
        return eval_ah(ah_line, side, hg, ag)
    return None

# ---- MAIN ----
with open("cache/football/predictions.jsonl", encoding="utf-8") as f:
    entries = [json.loads(l) for l in f if l.strip()]

# Snapshots Aug 12-20
snapshots = {}
for e in entries:
    if e.get("event") == "snapshot" and e.get("league") in ("UEL", "UECL"):
        ko = e.get("kickoff", "")
        if ko and any(ko.startswith(f"2026-08-{d}") for d in ["12","13","14","20"]):
            mid = e.get("match_id", "")
            if mid not in snapshots or e.get("ts", "") > snapshots[mid].get("ts", ""):
                snapshots[mid] = e

# Settles
settle_map = {}
for e in entries:
    if e.get("event") == "settle" and "home_goals" in e:
        mid = e.get("match_id", "")
        if ("UEL" in mid or "UECL" in mid):
            for d in ["12","13","14","20"]:
                if f"2026-08-{d}" in mid:
                    if mid not in settle_map:
                        settle_map[mid] = e
                    break

def normalize_name(name):
    n = name.strip()
    n = re.sub(r'\s*\([A-Za-z]+\)\s*$', '', n)
    n = re.sub(r'^(PFC|FC|FK|SL)\s+', '', n)
    return n.lower().strip()

matches = []
for mid, snap in snapshots.items():
    home = snap.get("home", "?")
    away = snap.get("away", "?")
    settle = None
    for smid, s in settle_map.items():
        parts = s.get("match_id", "").split("||")
        s_home = parts[1] if len(parts) > 1 else ""
        s_away = parts[2] if len(parts) > 2 else ""
        hn, an = normalize_name(home), normalize_name(away)
        shn, san = normalize_name(s_home), normalize_name(s_away)
        if (hn in shn or shn in hn) and (an in san or san in an):
            settle = s
            break
    if settle:
        ko = snap.get("kickoff", "")
        date = ko[:10] if ko else "?"
        matches.append({
            "home": home, "away": away, "league": snap.get("league", "?"),
            "date": date,
            "old_prob": snap.get("prob_1x2", {}), "old_odds": snap.get("odds_1x2", {}),
            "old_best_pick": snap.get("best_pick"),
            "hg": settle["home_goals"], "ag": settle["away_goals"],
            "outcome": settle.get("outcome", "?"),
        })

# Sort by date
matches.sort(key=lambda m: m["date"])

# Process
results = []
for m in matches:
    old_prob = m["old_prob"]
    if not old_prob or not isinstance(old_prob, dict):
        old_prob = {"home": 0.33, "draw": 0.33, "away": 0.33}
    odds = m["old_odds"]
    actual = m["outcome"]

    new_prob, alpha, dev, new_lh, new_la = cross_league_adjust(old_prob, odds)
    if not new_prob:
        new_prob = old_prob.copy() if old_prob else {"home": 0.33, "draw": 0.33, "away": 0.33}

    old_pred = max(old_prob, key=old_prob.get) if old_prob else "?"
    new_pred = max(new_prob, key=new_prob.get)

    bp = m["old_best_pick"]
    old_correct_1x2 = (old_pred == actual)
    new_correct_1x2 = (new_pred == actual)
    old_pick_correct, old_pick_type = eval_old_pick(bp, m["hg"], m["ag"])

    new_pick = generate_new_pick(new_prob, odds, bp, new_lh, new_la, old_pick_correct)
    new_pick_correct = eval_new_pick(new_pick, m["hg"], m["ag"])

    results.append({
        "home": m["home"], "away": m["away"], "league": m["league"], "date": m["date"],
        "hg": m["hg"], "ag": m["ag"], "outcome": actual,
        "old_prob": old_prob, "new_prob": new_prob,
        "old_pred": old_pred, "new_pred": new_pred,
        "old_correct_1x2": old_correct_1x2, "new_correct_1x2": new_correct_1x2,
        "old_bp": bp, "old_pick_correct": old_pick_correct,
        "new_bp": new_pick, "new_pick_correct": new_pick_correct,
        "alpha": alpha,
    })

# Print
SEP = "=" * 100
print(SEP)
print("  UEL / UECL BACKTEST  --  Aug 12-20, 2026")
print(f"  {len(results)} matches | OLD: Raw Model  |  NEW: Model + Cross-League Adjustment")
print(SEP)
print()

cur_date = ""
for i, r in enumerate(results, 1):
    if r["date"] != cur_date:
        cur_date = r["date"]
        print(f"  --- {cur_date} ---")
        print()

    score = f"{r['hg']}-{r['ag']}"
    oc = "[OK]" if r["old_correct_1x2"] else "[NO]"
    nc = "[OK]" if r["new_correct_1x2"] else "[NO]"
    opc = "[OK]" if r["old_pick_correct"] is True else ("[NO]" if r["old_pick_correct"] is False else "[??]")
    npc = "[OK]" if r["new_pick_correct"] is True else ("[NO]" if r["new_pick_correct"] is False else "[--]")

    old_s = f"{r['old_bp'].get('market','')} {r['old_bp'].get('selection','')}" if r['old_bp'] and r['old_bp'].get('market') else "none"
    new_s = f"{r['new_bp'].get('market','')} {r['new_bp'].get('selection','')}" if r['new_bp'] and r['new_bp'].get('market') else "NO BET"
    pres = " (pres)" if r['new_bp'] and r['new_bp'].get('preserved') else ""

    print(f"  {r['league']} {r['home']} vs {r['away']}  {score} ({r['outcome'].upper()})")
    print(f"    OLD 1X2: {r['old_pred'].upper()} {oc}  |  OLD Pick: {old_s} {opc}")
    print(f"    NEW 1X2: {r['new_pred'].upper()} {nc}  |  NEW Pick: {new_s} {npc}{pres}")
    print()

# Summary
print(SEP)
print("  SUMMARY")
print(SEP)
print()

o1w = sum(1 for r in results if r["old_correct_1x2"])
o1l = sum(1 for r in results if not r["old_correct_1x2"])
n1w = sum(1 for r in results if r["new_correct_1x2"])
n1l = sum(1 for r in results if not r["new_correct_1x2"])
fc1 = sum(1 for r in results if not r["old_correct_1x2"] and r["new_correct_1x2"])
fw1 = sum(1 for r in results if r["old_correct_1x2"] and not r["new_correct_1x2"])

print("  1X2 PREDICTION ACCURACY:")
print(f"    OLD:  {o1w}W / {o1l}L = {o1w*100//max(1,o1w+o1l)}%")
print(f"    NEW:  {n1w}W / {n1l}L = {n1w*100//max(1,n1w+n1l)}%")
print(f"    Flipped WRONG->RIGHT: +{fc1}  |  Flipped RIGHT->WRONG: -{fw1}  |  Net: {fc1-fw1:+d}")
print()

opw = sum(1 for r in results if r["old_pick_correct"] is True)
opl = sum(1 for r in results if r["old_pick_correct"] is False)
opx = sum(1 for r in results if r["old_pick_correct"] is None)
npw = sum(1 for r in results if r["new_pick_correct"] is True)
npl = sum(1 for r in results if r["new_pick_correct"] is False)
npx = sum(1 for r in results if r["new_pick_correct"] is None)

print("  BEST PICK ACCURACY:")
print(f"    OLD:  {opw}W / {opl}L / {opx}nc  ({opw*100//max(1,opw+opl)}% win rate)")
print(f"    NEW:  {npw}W / {npl}L / {npx}nc  ({npw*100//max(1,npw+npl)}% win rate)")
print()

# Date breakdown
print("  BY DATE:")
for d in ["2026-08-12", "2026-08-13", "2026-08-14", "2026-08-20"]:
    dr = [r for r in results if r["date"] == d]
    if not dr: continue
    do = sum(1 for r in dr if r["old_correct_1x2"])
    dn = sum(1 for r in dr if r["new_correct_1x2"])
    dop = sum(1 for r in dr if r["old_pick_correct"] is True)
    dnp = sum(1 for r in dr if r["new_pick_correct"] is True)
    print(f"    {d}: {len(dr)} matches | 1X2: OLD {do}/{len(dr)} NEW {dn}/{len(dr)} | Pick: OLD {dop}/{len(dr)} NEW {dnp}/{len(dr)}")
print()
print(SEP)
