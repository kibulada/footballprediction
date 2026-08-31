"""MARKET LEAN candidates + SUGGESTION TO PICK selection (K4, 2026-08-28).

Pure functions shared by the render layer (``format._market_lean_block``)
and the analyser (which persists the chosen suggestion on the snapshot so
it can be graded later -- before this module the suggestion was recomputed
at render time and never stored).

Post-mortem 25-27 Aug 2026 (54 graded suggestions, 40-14): the old rule
ALWAYS returned one candidate -- the market with the highest devigged
implied probability -- even when no market was dominant. In those cards the
"suggestion" was merely the least-unlikely option (Under 2.5 @1.60 = 58%
went 4-5; 1X2 favourites priced 1.48-1.54 went 0-3). Lessons encoded here,
as general rules rather than per-match thresholds:

K4a dominance  -- a suggestion must clear ``dominance_floor`` on
                  ``implied * (1 - vig)`` (default 0.60 = the engine's
                  "clear favourite" bar, ``lambda_1x2_favourite_prob``);
                  otherwise the honest answer is "—".
K4b agreement  -- the model must not contradict the suggested selection
                  by more than ``conflict_pp`` (the engine's existing
                  model-vs-market conflict threshold). Suggesting Under 2.5
                  while our lambda says 2.8 goals is a coin flip dressed up.
K1  evidence   -- when the model has NO directional evidence (both Elo on
                  the prior AND no xG-based lambda) a 1X2/AH suggestion is
                  parroting the market at 55-65%; only Totals/BTTS remain.
K3  context    -- a decided two-legged tie (aggregate margin >= 2) makes the
                  90-minute favourite unreliable (rotation / sitting back);
                  1X2/AH are not suggested. Thin form (< ``min_form_len``
                  matches) blocks 1X2/AH/Under.

Every block is reported with its reason so the card can print
``SUGGESTION TO PICK: — (alasan)`` instead of silently forcing a pick.
"""
from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    # "dominant" = clear favourite bar. 1X2 is 3-way so 40% after vig is
    # already strong (Home 2.20 -> 42% fair); 60% is rare. Totals/BTTS are
    # 2-way so 60% remains. Per-market floor applied in select_suggestion.
    "dominance_floor": 0.60,
    "dominance_floor_1x2": 0.55,
    "conflict_pp": 5.0,        # tighter than signal_engine 8pp - SUG must agree with model
    "min_form_len": 3,
    "require_direction_evidence": True,
    "decided_tie_no_directional": True,
    "under_requires_model_agreement": True,
}


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------

def lean_candidates(
    totals: dict[str, Any] | None,
    consensus: dict[str, Any] | None,
    ah: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """One candidate per market (Totals / AH / 1X2 / BTTS): the side the
    market leans to, with devigged ``implied`` and the market ``vig``.

    Candidate keys: ``label`` (display), ``raw_label`` (settlement
    selection), ``market``, ``odds``, ``implied``, ``vig``, ``line`` (AH
    home line), ``side`` (1X2/AH side).
    """
    totals = totals or {}
    consensus = consensus or {}
    ah = ah or {}
    cands: list[dict[str, Any]] = []

    over = (totals.get("Over 2.5") or {}).get("odds")
    under = (totals.get("Under 2.5") or {}).get("odds")
    o, u = _f(over), _f(under)
    if o and u and o > 1.0 and u > 1.0:
        ia, ib = 1.0 / o, 1.0 / u
        tot = ia + ib
        vig = tot - 1.0
        if u < o:
            cands.append({"label": "Under 2.5", "raw_label": "Under 2.5", "market": "Total",
                          "odds": u, "implied": ib / tot, "vig": vig, "line": 2.5, "side": None})
        else:
            cands.append({"label": "Over 2.5", "raw_label": "Over 2.5", "market": "Total",
                          "odds": o, "implied": ia / tot, "vig": vig, "line": 2.5, "side": None})

    line = _f(ah.get("line")) if ah.get("line") is not None else None
    h, a = _f(ah.get("home")), _f(ah.get("away"))
    if line is not None and h and a and h > 1.0 and a > 1.0:
        ia, ib = 1.0 / h, 1.0 / a
        tot = ia + ib
        vig = tot - 1.0
        if a < h:
            raw = f"Away {-line:+.2f}" if abs(line) > 1e-9 else "Away +0.00"
            cands.append({"label": f"AH: {raw}", "raw_label": raw, "market": "Asian Handicap",
                          "odds": a, "implied": ib / tot, "vig": vig, "line": line, "side": "away"})
        else:
            raw = f"Home {line:+.2f}"
            cands.append({"label": f"AH: {raw}", "raw_label": raw, "market": "Asian Handicap",
                          "odds": h, "implied": ia / tot, "vig": vig, "line": line, "side": "home"})

    ch, cd, ca = _f(consensus.get("home")), _f(consensus.get("draw")), _f(consensus.get("away"))
    if ch and cd and ca and ch > 1.0 and cd > 1.0 and ca > 1.0:
        inv = {"home": 1.0 / ch, "draw": 1.0 / cd, "away": 1.0 / ca}
        tot = sum(inv.values())
        vig = tot - 1.0
        side = min(inv, key=lambda k: -inv[k])
        label = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}[side]
        cands.append({"label": f"1X2: {label}", "raw_label": label, "market": "1X2",
                      "odds": {"home": ch, "draw": cd, "away": ca}[side],
                      "implied": inv[side] / tot, "vig": vig, "line": None, "side": side})

    y, n = _f((totals.get("BTTS Yes") or {}).get("odds")), _f((totals.get("BTTS No") or {}).get("odds"))
    if y and n and y > 1.0 and n > 1.0:
        ia, ib = 1.0 / y, 1.0 / n
        tot = ia + ib
        vig = tot - 1.0
        if n < y:
            cands.append({"label": "BTTS No", "raw_label": "BTTS No", "market": "BTTS",
                          "odds": n, "implied": ib / tot, "vig": vig, "line": None, "side": None})
        else:
            cands.append({"label": "BTTS Yes", "raw_label": "BTTS Yes", "market": "BTTS",
                          "odds": y, "implied": ia / tot, "vig": vig, "line": None, "side": None})
    return cands


# --------------------------------------------------------------------------
# model view of a candidate
# --------------------------------------------------------------------------

def model_prob_for(
    cand: dict[str, Any],
    model_probs: dict[str, Any] | None,
    ranking: list[dict[str, Any]] | None = None,
) -> float | None:
    """The model's probability for the candidate's selection, or None."""
    mp = model_probs or {}
    market = cand.get("market")
    raw = str(cand.get("raw_label") or "")
    if market == "Total":
        p_over = _f(mp.get("over_2.5"))
        if p_over is None:
            return None
        return p_over if raw.startswith("Over") else 1.0 - p_over
    if market == "BTTS":
        p_yes = _f(mp.get("btts_yes"))
        if p_yes is None:
            return None
        return p_yes if raw.endswith("Yes") else 1.0 - p_yes
    if market == "1X2":
        p = mp.get("1x2") or {}
        return _f(p.get(cand.get("side")))
    if market == "Asian Handicap":
        line, side = cand.get("line"), cand.get("side")
        for e in ranking or []:
            if e.get("market") != "Asian Handicap" or e.get("side") != side:
                continue
            el = _f(e.get("line"))
            if el is not None and line is not None and abs(el - float(line)) < 1e-6:
                return _f(e.get("model_prob"))
        return None
    return None


def _form_len(features: dict[str, Any] | None, side: str) -> int | None:
    seq = (features or {}).get(f"form_{side}")
    if seq is None:
        return None
    return sum(1 for c in str(seq).upper() if c in "WDL")


def _no_direction_evidence(model_probs: dict[str, Any] | None, features: dict[str, Any] | None) -> bool:
    """K1: both Elo on the prior AND lambda without xG -> no directional evidence."""
    mp = model_probs or {}
    hs, as_ = mp.get("elo_home_seeded"), mp.get("elo_away_seeded")
    if hs is None or as_ is None:
        f = features or {}
        eh, ea = _f(f.get("elo_home")), _f(f.get("elo_away"))
        both_prior = (
            mp.get("elo_seeded") is False
            and eh is not None and ea is not None
            and abs(eh - 1500.0) < 1e-6 and abs(ea - 1500.0) < 1e-6
        )
    else:
        both_prior = (not hs) and (not as_)
    src = str(mp.get("lambda_source") or "")
    no_xg = "xg" not in src
    return both_prior and no_xg


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def select_suggestion(
    cands: list[dict[str, Any]],
    *,
    model_probs: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    tie_state: dict[str, Any] | None = None,
    ranking: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick at most ONE suggestion; report every blocked candidate.

    Returns ``{"pick": cand | None, "blocked": [{label, reason}], "floor",
    "candidates": [...]}``. ``pick`` carries ``adjusted_score`` and
    ``model_prob``. Deterministic; no I/O.
    """
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (cfg or {}).items() if v is not None})
    floor = float(c["dominance_floor"])
    conflict = float(c["conflict_pp"]) / 100.0
    min_form = int(c["min_form_len"])
    blocked: list[dict[str, str]] = []
    eligible: list[dict[str, Any]] = []
    no_dir = bool(c["require_direction_evidence"]) and _no_direction_evidence(model_probs, features)
    decided = bool(c["decided_tie_no_directional"]) and (tie_state or {}).get("state") == "decided"

    for cand in cands:
        market = cand.get("market")
        raw = str(cand.get("raw_label") or "")
        imp = float(cand.get("implied") or 0.0)
        vig = max(0.0, min(0.9, float(cand.get("vig") or 0.0)))
        # Dominance floor is per-market: 1X2 55% (3-way), others 60% (2-way)
        _floor = float(c.get("dominance_floor_1x2", 0.55)) if market == "1X2" else floor
        base_adjusted = imp * (1.0 - vig)
        # Model agreement factor: 1 at edge 0, 0 at edge >= conflict (5pp for SUG)
        mprob_tmp = model_prob_for(cand, model_probs, ranking)
        if mprob_tmp is not None:
            _edge = abs(float(mprob_tmp) - imp)
            _factor = max(0.0, 1.0 - _edge / conflict) if conflict > 0 else 0.0
            adjusted = base_adjusted * (0.5 + 0.5 * _factor)  # 0.5 floor so dominant market not zeroed, but agreement still ranks higher
        else:
            adjusted = base_adjusted
        directional = market == "Asian Handicap" or (market == "1X2" and raw != "Draw")
        reason: str | None = None
        if adjusted < _floor:
            reason = f"pasar tidak dominan ({adjusted:.0%} < {_floor:.0%} setelah vig)"
        elif directional and no_dir:
            reason = "model tanpa evidensi arah (kedua Elo prior + tanpa xG) — 1X2/AH hanya menyalin pasar"
        elif directional and decided:
            reason = f"leg-2, agregat sudah selesai ({(tie_state or {}).get('first_leg')}) — favorit 90' tidak andal"
        else:
            sides = [cand.get("side")] if cand.get("side") else ["home", "away"]
            if directional or raw.startswith("Under"):
                thin = [s for s in sides if (fl := _form_len(features, s)) is not None and fl < min_form]
                if thin:
                    reason = f"form {'/'.join(thin)} < {min_form} laga — evidensi tipis"
        mprob = model_prob_for(cand, model_probs, ranking)
        if reason is None and mprob is not None and (mprob < imp - conflict) and (
            bool(c["under_requires_model_agreement"]) or not raw.startswith("Under")
        ):
            reason = f"model {mprob:.0%} vs pasar {imp:.0%} — model tidak setuju (>{conflict*100:.0f}pp)"
        entry = dict(cand)
        entry["adjusted_score"] = round(adjusted, 4)
        entry["model_prob"] = round(mprob, 4) if mprob is not None else None
        if reason:
            entry["blocked_reason"] = reason
            blocked.append({"label": str(cand.get("label")), "reason": reason})
        else:
            eligible.append(entry)

    eligible.sort(
        key=lambda e: (e["adjusted_score"], float(e.get("implied") or 0.0), -float(e.get("odds") or 999.0)),
        reverse=True,
    )
    pick = eligible[0] if eligible else None
    return {
        "pick": pick,
        "blocked": blocked,
        "floor": floor,
        "candidates": [dict(e) for e in eligible],
    }


def compute_suggestion(
    *,
    totals: dict[str, Any] | None,
    consensus: dict[str, Any] | None,
    ah: dict[str, Any] | None,
    model_probs: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    tie_state: dict[str, Any] | None = None,
    ranking: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Candidates + selection in one call (what the analyser persists)."""
    cands = lean_candidates(totals, consensus, ah)
    out = select_suggestion(
        cands, model_probs=model_probs, features=features, tie_state=tie_state,
        ranking=ranking, cfg=cfg,
    )
    out["n_candidates"] = len(cands)
    return out


def suggestion_for_settlement(pick: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map a stored suggestion pick to ``signal_engine.settle_signal`` input."""
    if not pick:
        return None
    market = pick.get("market")
    raw = str(pick.get("raw_label") or pick.get("selection") or "")
    out: dict[str, Any] = {"market": market, "selection": raw}
    if market == "Asian Handicap":
        out["line"] = pick.get("line")
        out["side"] = pick.get("side")
    return out
