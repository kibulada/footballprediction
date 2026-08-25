"""Per-market confidence tiers (OUTPUT POLICY — Single Best Pick, selection layer only).

Tiers are computed PER MARKET (1X2, Over/Under 2.5, Over/Under 3.5, BTTS),
never once per match: each market has its own disagreement, edge and
contradiction state over the same underlying model data, so the tier can
differ market by market inside one match. The bot MUST evaluate a tier for
EVERY market with available data — no market is skipped, simplified or
altered in the computation step. Only the final display is filtered down to
ONE already-computed result via the selection layer (PICK > LEAN > WATCH).

All functions are PURE (no I/O, no state, no user input), so a tier can
never be manually raised: it is derived only from the computed
confidence / disagreement / edge values.

Tier rules (applied per market, WATCH checked first, then LEAN, then PICK):
  🟢 TIER 1 PICK  — all must hold: confidence HIGH; model A vs B disagreement
                     < 8pp; no contradictory edges; data completeness
                     Medium/High; the selected outcome's EV > MIN_EV (3%, the
                     engine's NO-VALUE bar).
  🔶 TIER 2 LEAN  — at least one: confidence MEDIUM; disagreement 8-20pp; a
                     large (10-12pp, below the SANITY_PP bar) edge with no
                     direct contradiction in this market; the selected
                     outcome's EV > MIN_EV.
  ⚪ TIER 3 WATCH — at least one: confidence LOW; disagreement > 20pp; two+
                     extreme edges (|edge| >= 20pp) pointing in opposite
                     directions within this market; data completeness Low;
                     outcome EV <= MIN_EV (a stake can never be attached to a
                     negative-EV selection -- invariant 8.1, tier layer);
                     model-vs-market gap >= SANITY_PP (12pp, P2) — a gap that
                     large is a data-mismatch signal (wrong fixture / stale
                     odds), never real value, so [12, 20)pp is WATCH, not LEAN.
"""
from __future__ import annotations

from typing import Any

from .decision import fair_pair_implied, margin_free_implied
from .model_gates import MIN_EV, min_tier

TIER_PICK = "PICK"
TIER_LEAN = "LEAN"
TIER_WATCH = "WATCH"

TIER_ICONS = {TIER_PICK: "🟢", TIER_LEAN: "🔶", TIER_WATCH: "⚪"}
TIER_STAKE = {
    TIER_PICK: "Normal 1 unit",
    TIER_LEAN: "Micro 0.25-0.5 unit",
    TIER_WATCH: "SKIP",
}
CONF_TIERS = ("HIGH", "MEDIUM", "LOW")

EXTREME_PP = 20.0   # |edge| >= 20pp -> extreme edge
LARGE_PP = 10.0     # |edge| in [10, 20)pp -> large (not extreme) edge
# P2 sanity check: a model-vs-market gap >= 12pp is almost always a data
# mismatch (wrong fixture / stale odds), never real value. The tier layer's
# AUDIT bar is lowered from the engine's 20pp to this — [12, 20)pp edges are
# WATCH + "verifikasi fixture/odds", never LEAN. Band [10, 12)pp stays a
# large edge (LEAN-able).
SANITY_PP = 12.0
DISAGREE_PICK_MAX = 8.0    # disagreement < 8pp required for PICK
DISAGREE_LEAN_MAX = 20.0   # disagreement in [8, 20]pp -> LEAN; > 20pp -> WATCH
WATCH_LEAN_MIN_PP = 10.0   # WATCH surface a directional lean only when |edge| >= this

# Fixed market order for the output; each market is shown even when it has
# no data (explicit "not evaluated" line, never silently dropped).
MARKET_LABELS = ("1X2", "Over/Under 2.5", "Over/Under 3.5", "BTTS")


def _pair_keys(label: str) -> tuple[str, str, str] | None:
    """(over_key, under_key, model_prob_key) for a pair market, else None."""
    if label == "Over/Under 2.5":
        return "Over 2.5", "Under 2.5", "over_2.5"
    if label == "Over/Under 3.5":
        return "Over 3.5", "Under 3.5", "over_3.5"
    if label == "BTTS":
        return "BTTS Yes", "BTTS No", "btts_yes"
    return None


def data_completeness_level(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Data completeness as (level, missing labels) over the standard inputs.

    Level: High (>= 4/5 present), Medium (>= 2/5), Low otherwise — the same
    rubric as the rest of the bot's data-quality output.
    """
    stats = payload.get("stats") or {}
    odds = payload.get("odds") or {}
    h2h = stats.get("h2h") or {}
    no_form = (stats.get("home_form") in (None, "n/a")) and (
        stats.get("away_form") in (None, "n/a")
    )
    items = [
        ("odds", bool(odds.get("has_odds"))),
        ("form", not no_form),
        ("GF/GA", bool(stats.get("home_gf_avg") or stats.get("away_gf_avg"))),
        (
            "xG",
            any(
                stats.get(k) is not None
                for k in ("home_xg_for", "away_xg_for", "home_xg_against", "away_xg_against")
            ),
        ),
        ("H2H", any((h2h or {}).get(k) for k in ("wins", "draws", "losses"))),
    ]
    ok = sum(1 for _, present in items if present)
    level = "High" if ok >= 4 else "Medium" if ok >= 2 else "Low"
    missing = [label for label, present in items if not present]
    return level, missing


def market_view(
    *,
    label: str,
    model_probs: dict[str, Any],
    consensus: dict[str, float],
    totals: dict[str, dict[str, float]],
    model_a: dict[str, Any] | None,
    disagreement_1x2_pp: float | None,
) -> dict[str, Any]:
    """Per-market signal view; ``evaluated=False`` when input data is missing.

    Edge (percentage points) = Model B (independent engine) probability minus
    the margin-free market implied probability, per selection. Contradiction
    = two+ extreme edges (|edge| >= 20pp) in opposite directions inside the
    same market (e.g. Over 2.5 AND Under 2.5 both extreme) — a model
    inconsistency, never a convergent signal.
    """
    view: dict[str, Any] = {"label": label, "evaluated": False, "reason": "insufficient data"}

    if label == "1X2":
        norm = margin_free_implied(consensus)
        p1x2 = model_probs.get("1x2") or {}
        if not norm or not p1x2 or not consensus.get("home", 0):
            return view
        sides = []
        for side, name in (("home", "Home Win"), ("draw", "Draw"), ("away", "Away Win")):
            if consensus.get(side, 0) <= 0 or norm.get(side, 0) <= 0:
                continue
            p = p1x2.get(side, 0.0) or 0.0
            sides.append({
                "selection": name,
                "edge_pp": (p - norm[side]) * 100.0,
                "model_prob": p,
                "odds": consensus[side],
                # EV at the offered odds (same formula as the engine's
                # decision/evaluated entries: p * odds - 1.0).
                "ev": p * consensus[side] - 1.0,
            })
        if not sides:
            return view
        outcome = max(sides, key=lambda s: s["model_prob"])["selection"]
        disagreement = disagreement_1x2_pp
        if disagreement is None and model_a and model_a.get("1x2"):
            disagreement = (
                abs(model_a["1x2"].get("home", 0) - p1x2.get("home", 0)) * 100.0
            )
    else:
        pair = _pair_keys(label)
        if pair is None:
            return view
        over_key, under_key, prob_key = pair
        p_over = model_probs.get(prob_key)
        if p_over is None:
            return view
        o = totals.get(over_key, {}).get("odds", 0.0)
        u = totals.get(under_key, {}).get("odds", 0.0)
        fair = fair_pair_implied(o, u)
        if fair is None:
            return view
        edge_over = (p_over - fair[0]) * 100.0
        sides = [
            {"selection": over_key, "edge_pp": edge_over, "model_prob": p_over, "odds": o,
             "ev": p_over * o - 1.0},
            {"selection": under_key, "edge_pp": -edge_over, "model_prob": 1.0 - p_over, "odds": u,
             "ev": (1.0 - p_over) * u - 1.0},
        ]
        outcome = over_key if edge_over >= 0 else under_key
        disagreement = None
        if model_a and model_a.get(prob_key) is not None:
            disagreement = abs(model_a[prob_key] - p_over) * 100.0

    extremes = [(s["selection"], s["edge_pp"]) for s in sides if abs(s["edge_pp"]) >= EXTREME_PP]
    larges = [
        (s["selection"], s["edge_pp"])
        for s in sides if LARGE_PP <= abs(s["edge_pp"]) < EXTREME_PP
    ]
    max_edge = max((abs(s["edge_pp"]) for s in sides), default=0.0)
    return {
        "label": label,
        "evaluated": True,
        "sides": sides,
        "outcome": outcome,
        "odds": next((s["odds"] for s in sides if s["selection"] == outcome), None),
        # EV of the selected outcome at the offered odds; used to gate tiers
        # (never stake a negative-EV selection, invariant 8.1).
        "outcome_ev": next((s["ev"] for s in sides if s["selection"] == outcome), None),
        "disagreement_pp": disagreement,
        "extremes": extremes,
        "larges": larges,
        "contradiction": len(extremes) >= 2,
        "max_edge_pp": max_edge,
        # P2 sanity: |model - market implied| >= SANITY_PP anywhere in this
        # market -> data-mismatch risk, demote to WATCH (never value).
        "sanity_risk": max_edge >= SANITY_PP,
    }


def market_confidence(global_tier: str, view: dict[str, Any]) -> str:
    """Per-market confidence: global tier, capped by this market's own state.

    An extreme edge or contradictory extremes inside the market make the
    market-level estimate unreliable -> LOW (mirrors the engine's
    "extreme edge is never treated as value" rule).
    """
    tier = global_tier if global_tier in CONF_TIERS else "LOW"
    if view.get("contradiction") or view.get("extremes"):
        return "LOW"
    return tier


def market_tier(view: dict[str, Any], global_tier: str, completeness: str) -> str:
    """The market's tier per the OUTPUT POLICY rules (WATCH -> LEAN -> PICK)."""
    conf = market_confidence(global_tier, view)
    dis = view.get("disagreement_pp")
    watch = (
        conf == "LOW"
        or (dis is not None and dis > DISAGREE_LEAN_MAX)
        or bool(view.get("contradiction"))
        or bool(view.get("sanity_risk"))
        or completeness == "Low"
    )
    lean = (
        conf == "MEDIUM"
        or (dis is not None and DISAGREE_PICK_MAX <= dis <= DISAGREE_LEAN_MAX)
        or (bool(view.get("larges")) and not view.get("contradiction"))
    )
    pick = (
        conf == "HIGH"
        and (dis is None or dis < DISAGREE_PICK_MAX)
        and not view.get("contradiction")
        and completeness in ("Medium", "High")
    )
    # Invariant 8.1 (tier layer): PICK/LEAN carry a stake, so the selected
    # outcome must clear the engine's NO-VALUE bar (EV > MIN_EV). A
    # negative/small-EV selection is demoted to WATCH even when
    # confidence/disagreement/edge otherwise look strong -- the tier must
    # never contradict the engine's "NO VALUE" verdict for the same market.
    ev = view.get("outcome_ev")
    if (pick or lean) and ev is not None and ev <= MIN_EV:
        view["ev_nonpositive"] = True
        return TIER_WATCH
    if watch:
        return TIER_WATCH
    if lean:
        return TIER_LEAN
    if pick:
        return TIER_PICK
    return TIER_WATCH  # fallback: never overstate


def _basis(view: dict[str, Any], conf: str, completeness: str) -> str:
    """The Basis field: disagreement + edges, contradictions NEVER hidden."""
    parts: list[str] = []
    if view.get("ev_nonpositive"):
        ev = view.get("outcome_ev")
        parts.append(
            f"EV {ev:+.1%} <= +{MIN_EV:.0%} — tidak ada nilai taruhan (senada gate engine)"
        )
    dis = view.get("disagreement_pp")
    if dis is not None:
        label = " on Home Win" if view["label"] == "1X2" else ""
        parts.append(f"Model A vs B disagreement {dis:.1f}pp{label}")
    if view.get("contradiction"):
        clauses = " AND ".join(
            f"{sel} extreme edge {e:+.1f}pp"
            for sel, e in sorted(view["extremes"], key=lambda x: -abs(x[1]))
        )
        parts.append(f"{clauses} simultaneously — contradictory model signal, not convergent")
    elif view.get("extremes"):
        for sel, e in sorted(view["extremes"], key=lambda x: -abs(x[1])):
            parts.append(f"{sel} extreme edge {e:+.1f}pp")
    elif view.get("sanity_risk"):
        # P2: gap >= SANITY_PP (but below extreme) is a data-mismatch signal
        # (wrong fixture / stale odds), not value — say so explicitly.
        parts.append(
            f"model-vs-market gap {view.get('max_edge_pp', 0.0):.1f}pp ≥ "
            f"{SANITY_PP:.0f}pp — verifikasi fixture/odds identity (bukan value)"
        )
    elif view.get("larges"):
        for sel, e in sorted(view["larges"], key=lambda x: -abs(x[1])):
            parts.append(f"{sel} large edge {e:+.1f}pp")
    else:
        out = view.get("outcome", "?")
        edge = next(
            (s["edge_pp"] for s in view.get("sides", []) if s["selection"] == out),
            0.0,
        )
        if abs(edge) < WATCH_LEAN_MIN_PP:
            parts.append(f"edge {out} {edge:+.1f}pp — below {WATCH_LEAN_MIN_PP:.0f}pp, no executable edge")
        else:
            parts.append(f"edge {out} {edge:+.1f}pp")
    if conf == "LOW" and not (view.get("extremes") or view.get("contradiction")):
        parts.append("confidence LOW")
    if completeness == "Low":
        parts.append("data completeness Low")
    return ", ".join(parts)


def _market_views(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str, list[str]]:
    """(computed views, completeness level, missing labels) for every market.

    OUTPUT POLICY step 1+2 (compute + tier EVERY market) happens here, fully
    and unchanged; both the full breakdown and the single-pick selection read
    the SAME already-computed views, so the selection can never re-derive or
    alter a market's tier.
    """
    odds = payload.get("odds") or {}
    prediction = payload.get("prediction") or {}
    model_probs = prediction.get("model_probs") or {}
    if not model_probs:
        model_probs = (payload.get("picks") or {}).get("model_probs") or {}
    consensus = odds.get("consensus") or {}
    totals = odds.get("totals") or {}
    decision = payload.get("decision") or {}
    model_a = decision.get("model_a")
    disagreement_1x2 = (decision.get("model_disagreement") or {}).get("delta_pp")
    conf_block = payload.get("confidence") or {}
    global_tier = str(conf_block.get("tier") or "LOW")
    level, missing = data_completeness_level(payload)

    views: list[dict[str, Any]] = []
    for label in MARKET_LABELS:
        view = market_view(
            label=label,
            model_probs=model_probs,
            consensus=consensus,
            totals=totals,
            model_a=model_a,
            disagreement_1x2_pp=disagreement_1x2,
        )
        if view.get("evaluated"):
            view["tier"] = market_tier(view, global_tier, level)
            view["confidence"] = market_confidence(global_tier, view)
        views.append(view)
    return views, level, missing


def render_market_tiers(payload: dict[str, Any]) -> list[str]:
    """The per-market tier block for an analyse payload (market sections only).

    The caller owns the match header line and the closing disclaimer, so the
    disclaimer appears exactly once per match output. A market without enough
    input to compute is shown explicitly as "not evaluated" — never dropped.
    """
    views, level, missing = _market_views(payload)

    lines: list[str] = []
    for view in views:
        label = view["label"]
        lines.append(f"── {label} ──")
        if not view.get("evaluated"):
            lines.append("❌ Not evaluated — insufficient data for this market")
        else:
            tier = view["tier"]
            conf = view["confidence"]
            icon = TIER_ICONS[tier]
            out = view["outcome"]
            line = f"{icon} {tier}: {out}"
            if view.get("odds"):
                line += f" @ {view['odds']:.2f}"
            lines.append(line)
            lines.append(f"Confidence: {conf} | Basis: {_basis(view, conf, level)}")
            lines.append(f"Stake: {TIER_STAKE[tier]}")
        lines.append("")

    icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(level, "⚪")
    comp = f"Data completeness: {icon} {level}"
    if missing:
        comp += f" — missing {', '.join(missing)}"
    lines.append(comp)
    return lines


# ---- OUTPUT POLICY -- Single Best Pick (selection layer only) --------------
# Every market is STILL computed in full (see _market_views). This layer only
# picks which already-computed result to surface: a) a Tier 1 (PICK) market,
# b) else a Tier 2 (LEAN) market, c) else the most directionally consistent
# Tier 3 (WATCH) result. Tier assignment is never changed here.


def _favored_edge(view: dict[str, Any]) -> float:
    """Edge (pp) of the view's selected outcome."""
    out = view.get("outcome")
    for s in view.get("sides") or []:
        if s["selection"] == out:
            return float(s.get("edge_pp") or 0.0)
    return 0.0


def select_best_pick(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The single best result per the OUTPUT POLICY priority.

    All markets are computed first (unchanged); this only selects ONE of the
    already-computed tiers:
      a) a Tier 1 (PICK) market -- lowest Model A/B disagreement among them;
      b) else a Tier 2 (LEAN) market -- most favorable edge, then lowest
         disagreement;
      c) else the most directionally consistent Tier 3 (WATCH) result.
    Returns the selected view (with tier/confidence/basis attached) or None
    when NO market has enough data to evaluate.
    """
    views, level, missing = _market_views(payload)
    evaluated = [v for v in views if v.get("evaluated")]
    if not evaluated:
        return None

    # The engine already concluded there is no independent signal
    # (MARKET PRIOR: prediction mirrors the market, edge = 0 by construction;
    # NO CLEAR DECISION: no independent model ran). The tier layer must not
    # surface a directional WATCH edge against that verdict -- naming a
    # direction here (e.g. "BTTS Yes, edge +8pp" from a prior-only, unseeded
    # model) contradicts the FINAL DECISION and misleads. No pick.
    d_type = (payload.get("decision") or {}).get("decision_type")
    if d_type in {"MARKET PRIOR", "NO CLEAR DECISION"}:
        return None

    def _dis(view: dict[str, Any]) -> float:
        d = view.get("disagreement_pp")
        return -1.0 if d is None else d  # no disagreement data ranks as best

    picks = [v for v in evaluated if v["tier"] == TIER_PICK]
    if picks:
        chosen = min(picks, key=_dis).copy()
        dis = chosen.get("disagreement_pp")
        dis_s = "no Model A/B disagreement data" if dis is None else f"{dis:.1f}pp"
        if len(picks) > 1:
            chosen["basis"] = (
                f"lowest disagreement among {len(picks)} Tier-1 markets ({dis_s})"
            )
        else:
            chosen["basis"] = f"only market at Tier 1 (PICK); disagreement {dis_s}"
        return chosen

    leans = [v for v in evaluated if v["tier"] == TIER_LEAN]
    if leans:
        chosen = max(leans, key=lambda v: (_favored_edge(v), -_dis(v))).copy()
        edge = _favored_edge(chosen)
        dis = chosen.get("disagreement_pp")
        dis_s = f"; disagreement {dis:.1f}pp" if dis is not None else ""
        chosen["basis"] = (
            f"{chosen['label']} reached Tier 2 (LEAN) with edge {edge:+.1f}pp "
            f"on {chosen['outcome']}{dis_s} — best LEAN signal"
        )
        return chosen

    watches = [v for v in evaluated if v["tier"] == TIER_WATCH]
    if watches:
        consistent = [v for v in watches if not v.get("contradiction")]
        pool = consistent or watches
        chosen = max(pool, key=lambda v: (abs(_favored_edge(v)), v["label"])).copy()
        # A WATCH with a sub-threshold edge is noise, not a signal: don't name
        # a direction. Only surface a directional lean when the model actually
        # diverges from the market by at least WATCH_LEAN_MIN_PP.
        if abs(_favored_edge(chosen)) < WATCH_LEAN_MIN_PP:
            return None
        if level == "Low":
            comp_note = (
                "all markets at Tier 3 due to Low data completeness "
                f"(missing {', '.join(missing)})"
            )
        else:
            comp_note = "all markets fell to Tier 3 (WATCH)"
        edge = _favored_edge(chosen)
        chosen["basis"] = (
            f"{comp_note}; {chosen['label']} selected as the most "
            f"directionally consistent signal (edge {edge:+.1f}pp)"
        )
        return chosen
    return None


def render_single_pick(payload: dict[str, Any]) -> list[str]:
    """Render the ONE selected result (OUTPUT POLICY — Single Best Pick).

    The caller owns the match header line and the closing disclaimer. The
    selection reads the already-computed tiers; it never recomputes or
    alters a market. A payload with no evaluable market states that honestly.
    """
    pick = select_best_pick(payload)
    if pick is None:
        d_type = (payload.get("decision") or {}).get("decision_type")
        views, _, _ = _market_views(payload)
        if not any(v.get("evaluated") for v in views):
            return ["❌ Tidak ada market dengan data cukup — tidak ada tier yang bisa disajikan."]
        if d_type in {"MARKET PRIOR", "NO CLEAR DECISION"}:
            return ["🚫 NO BET — prediksi mengikuti market (tanpa sinyal independen) → SKIP."]
        return ["🚫 NO BET — tidak ada market dengan edge independen yang valid; semua market Tier 3 (WATCH) → SKIP."]
    tier = pick["tier"]
    icon = TIER_ICONS[tier]
    line = f"{icon} {tier}: {pick['outcome']}"
    if pick.get("odds"):
        line += f" @ {pick['odds']:.2f}"
    return [
        line,
        f"Confidence: {pick['confidence']} | Basis: {pick['basis']}",
        f"Stake: {TIER_STAKE[tier]}",
    ]
