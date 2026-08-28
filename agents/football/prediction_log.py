"""Append-only JSONL prediction log (audit PHASE 7).

Every pre-match prediction is appended as one immutable JSON line before
kickoff: match identity, feature-input hash, model probabilities, consensus
odds, margin-free edge, confidence, signal, calibration state, model version
and sources.

After the match, ``settle`` appends a separate settlement line keyed by the
same match_id (result + optional closing odds). ``stats`` joins snapshots and
settlements and reports realised hit rate, log-loss, CLV and flat-stake ROI.

Honesty rules:
  - Append-only by construction; snapshots are never edited in place.
  - Metrics are reported only for *settled* snapshots.
  - ROI is flat-stake on the best 1X2 pick with margin-free edge >= threshold
    (mirrors validate.py); CLV is (model_prob * closing_odds - 1) per settled
    snapshot that carries closing odds -- both clearly labelled.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from .timeutil import utc_now_iso

KEYS = ("home", "draw", "away")

# PHASE 32-33 (prediction timing snapshots): a prediction snapshot is the
# immutable pre-match record created once; ODDS snapshots are additional
# price observations captured at standard times before kickoff so CLV
# (closing vs prediction price) can be evaluated historically. Labels are
# descriptive only -- the evaluator must never mix snapshots from different
# timings (each row carries its own ``timing`` + ``ts``).
ODDS_TIMING_LABELS = ("T-24h", "T-6h", "T-1h", "T-15m", "T-0h")


# Fix 1 (match identity normalization): the match_id is the ONLY key the
# Layer-3 stability guard, odds snapshots and settlements use to find prior
# records for a fixture. Team names therefore MUST be canonicalized here --
# the single choke point every call site (analyse, odds poll, settle) flows
# through -- or the same real-world fixture yields different match_ids per
# query when providers resolve names differently (the "Rio Ave FC" vs
# "Rio Ave" bug). Canonical name source: teams.json (team_alias.py), the
# authoritative per-league team-name table; every provider variant maps to
# that canonical form via resolve_team_alias. Teams not in the table fall
# back to a deterministic normalization that strips common club suffixes
# (fc/cf/sc/ac/fk), so "Porto FC" and "FC Porto" still collide.
_CLUB_SUFFIXES = ("fc", "cf", "sc", "ac", "fk")


def _canonical_team_name(name: str, league: str | None = None) -> str:
    """Deterministic canonical team name for match_id construction.

    1. resolve_team_alias(name, league) when the league is known and the
       team exists in teams.json -> the canonical name VERBATIM (keeps case
       and suffixes, so "Rio Ave FC" and "Rio Ave" both become the
       authoritative "Rio Ave FC" -- matching the historical log records).
    2. Teams not in the alias table fall back to a deterministic
       lowercase+suffix-strip ("Rio Ave FC" -> "rio ave", "Porto FC" ->
       "porto") so equivalent variants still collide.
    Always deterministic; never random or request-order-dependent.
    """
    from .team_alias import resolve_team_alias

    raw = str(name or "").strip()
    if not raw:
        return raw
    if league:
        resolved = resolve_team_alias(raw, league)
        if resolved:
            return resolved
    # Not in the alias table: strip club suffixes (case-preserving) so
    # equivalent variants collide, e.g. "Rio Ave FC" -> "Rio Ave",
    # "Porto FC" -> "Porto" (and "FC Porto" -> "Porto" via prefix).
    s = " ".join(str(name or "").split())
    parts = s.split()
    if len(parts) >= 2 and parts[0].lower() in _CLUB_SUFFIXES:
        parts = parts[1:]
    if len(parts) >= 2 and parts[-1].lower() in _CLUB_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts) if parts else s


def _kickoff_date_component(kickoff: str | None) -> str:
    """Canonical date-only component for a match_id.

    P1.2: the PRECISE kickoff time is NOT part of the match identity -- the
    same real-world fixture can resolve its kickoff from different sources
    (flashscore fixture vs odds commence_time) that differ by minutes, and a
    full-timestamp component would split one match into several match_ids,
    silently defeating the Layer-3 stability guard (it would never find the
    prior pick). Only the DATE is used so a later-corrected kickoff still
    resolves to the same canonical match record.

    Trade-off, documented: two distinct fixtures of the SAME two teams in
    the SAME league on the SAME calendar date would share a match_id. This is
    extremely rare in practice (double-headers of the same pairing in one
    competition on one day) and is the accepted cost of a stable identity;
    the kickoff itself is still stored on every snapshot row for the display
    and the finished-match check.
    """
    s = str(kickoff or "").strip()
    # Accept ISO "YYYY-MM-DD..." forms and plain "YYYY-MM-DD". Anything
    # unparseable (provider garbage) keeps the raw string so it still
    # contributes to identity.
    if len(s) >= 10 and s[:4].isdigit() and s[4] == "-" and s[7] == "-":
        return s[:10]
    return s


def make_match_id(league: str, home: str, away: str, kickoff: str | None) -> str:
    return (
        f"{league}||{_canonical_team_name(home, league)}||"
        f"{_canonical_team_name(away, league)}||{_kickoff_date_component(kickoff)}"
    )


def canonical_match_id(
    entities: dict[str, Any] | None,
    league: str,
    home: str,
    away: str,
    kickoff: str | None,
) -> str:
    """ID-level match identity (Fix 2026-08-22, anti duplicate-ejaan).

    Prefers the G2 canonical team ids so provider spelling variants of the
    SAME club ("Al-Faisaly" vs "Al Faisaly", "NEOM SC" vs "Neom") produce
    ONE match_id instead of parallel snapshot streams with separate
    stability pins. Falls back to the legacy name-based ``make_match_id``
    when canonical ids are unavailable (legacy paths / empty names), so
    every caller keeps working unchanged.
    """
    h = ((entities or {}).get("home") or {}).get("canonical_id")
    a = ((entities or {}).get("away") or {}).get("canonical_id")
    if not h or not a:
        return make_match_id(league, home, away, kickoff)
    return f"{league}||cid:{h}||cid:{a}||{_kickoff_date_component(kickoff)}"


def _identity_normalize(s: str) -> str:
    """Accent-insensitive form of a match_id component for JOIN comparison.

    The SAME real club is spelled differently by the providers that feed
    the match resolution: thesportsdb/nowgoal keep diacritics ("Göztepe")
    while flashscore drops them ("Goztepe"). ``make_match_id`` preserves
    the provider's spelling on purpose (historical log records stay
    addressable), so every identity JOIN -- opening_snapshot, prior-pick
    stability, settle, odds history, dedupe -- normalizes diacritics here:
    one real match is never split across two match_ids (verified live
    2026-08-17: Samsunspor vs Göztepe produced two parallel snapshots with
    identical data except the accented team name, starving the card's
    movement + statistical components). Case is preserved; only combining
    marks are dropped.
    """
    import unicodedata

    s = unicodedata.normalize("NFD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c))


def _match_id_hits(record_id: str | None, match_id: str) -> bool:
    """True when a stored record belongs to the canonical match.

    P1.2 backward compatibility: records written before the date-only
    canonicalization carry the full kickoff timestamp in the 4th component
    (``...||2026-08-15T18:30:00Z``). A date-only ``match_id`` (``...||2026-08-15``)
    must still find them, so an exact match OR a legacy full-timestamp prefix
    (``match_id + "T"``) both count as the same match. Comparison is
    accent-insensitive (``_identity_normalize``) so provider spelling
    variants ("Göztepe" vs "Goztepe") never split one fixture.
    """
    if not record_id:
        return False
    r = _identity_normalize(record_id)
    m = _identity_normalize(match_id)
    if r == m:
        return True
    return bool(m) and r.startswith(m + "T")


def _match_dedupe_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Canonical identity of a snapshot/settle row for dedupe.

    One real-world match must count ONCE in settlement and evaluation, even
    when multiple snapshots exist for it (repeated queries, or pre-Fix-1
    match_id variants like "Rio Ave FC" vs "Rio Ave" -- same fixture, two
    ids). The key is (league, canonical home, canonical away, kickoff date),
    the same canonicalization ``make_match_id`` uses, so every variant of a
    fixture collides to one key. ``row`` is a snapshot (home/away/league/
    kickoff fields) or a settle row (league/home/away parsed from match_id).
    """
    # match_id is the canonical identity (Fix 1): league||home||away||date.
    # Snapshot rows always carry it; settle rows carry only it. Parse from
    # match_id -- never from the display home/away fields, which can differ
    # from the canonical form on pre-Fix-1 rows.
    mid = str(row.get("match_id") or "")
    parts = mid.split("||")
    if len(parts) < 4:
        return (mid, "", "", "")
    league = parts[0]
    home, away = parts[1], parts[2]
    kick = parts[3]
    return (
        _identity_normalize(league),
        _identity_normalize(_canonical_team_name(home, league or None)),
        _identity_normalize(_canonical_team_name(away, league or None)),
        _identity_normalize(_kickoff_date_component(kick)),
    )


def list_unsettled(path: str | Path) -> list[dict[str, Any]]:
    """Snapshots that do not yet have a matching settlement line.

    Deduped per canonical match (``_match_dedupe_key``): when several
    snapshots exist for the same fixture (repeated queries, pre-Fix-1
    match_id variants), only the NEWEST snapshot is returned so settle
    auto settles each match exactly once -- no double settlement rows.
    """
    rows = _read_lines(Path(path))
    settled_ids = [r.get("match_id") for r in rows if r.get("event") == "settle"]
    unsettled = [
        r for r in rows
        if r.get("event") == "snapshot"
        and not any(_match_id_hits(sid, r.get("match_id") or "") for sid in settled_ids)
    ]
    newest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in unsettled:
        key = _match_dedupe_key(r)
        cur = newest.get(key)
        if cur is None or (r.get("ts") or "") > (cur.get("ts") or ""):
            newest[key] = r
    return list(newest.values())


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_snapshot(
    path: str | Path,
    *,
    skip: bool = False,
    match_id: str,
    league: str,
    home: str,
    away: str,
    kickoff: str | None,
    prob: dict[str, float] | None,
    odds: dict[str, float] | None,
    edge: dict[str, float] | None,
    confidence: float | None,
    signal: int | None,
    calibration: dict[str, Any] | None,
    model_version: str | None,
    input_hash: str | None,
    best_pick: dict[str, Any] | None,
    sources: list[str] | None,
    features: dict[str, Any] | None = None,
    decision_type: str | None = None,
    final_decision: dict[str, Any] | None = None,
    ml_prob: dict[str, Any] | None = None,
    edge_benchmark: dict[str, Any] | None = None,
    movement: dict[str, Any] | None = None,
    flashscore_url: str | None = None,
    signal_engine_pick: dict[str, Any] | None = None,
    signal_engine_ranking: list[dict[str, Any]] | None = None,
    context_data: dict[str, Any] | None = None,
    lineup_source: str | None = None,
    lineup_ts: str | None = None,
    paper_trade: bool | None = None,
    entities: dict[str, Any] | None = None,
    market_totals: dict[str, dict[str, float]] | None = None,
    model_probs: dict[str, Any] | None = None,
    suggestion: dict[str, Any] | None = None,
) -> None:
    """Append one immutable pre-match prediction snapshot.

    ``suggestion`` (K4, 2026-08-28) is the market-only SUGGESTION TO PICK
    exactly as rendered (``{pick, blocked, floor, n_candidates}``) so it can
    be settled alongside the BEST PICK.

    ``skip`` (Anti-flap P1, 2026-08-23) short-circuits the write: MARKET
    PRIOR reference rows are not logged as decisions (see analyse.py).
    ``features`` (optional) captures the pre-match input state for later
    similar-signal analysis: Elo values, Poisson lambdas, attack/defense,
    form sequences, completeness. Stored as-is; never edited in place.
    ``decision_type`` (optional, TODO-15) is the engine's decision label
    (STRONG/GOOD/LEAN/WATCH/NO BET/NO CLEAR DECISION) so CLV and ROI can be
    tracked per decision type in production.
    ``model_probs`` (optional, Fix 2026-08-22) persists the FULL model
    probability block (1X2 + over_*/under_*/btts_yes + lambda_* +
    uncertainty) so post-hoc evaluation never has to replay the Poisson
    matrix from stored lambdas again.
    """
    if skip:
        return None
    row = {
        "event": "snapshot",
        "match_id": match_id,
        "ts": utc_now_iso(),
        "league": league,
        "home": home,
        "away": away,
        "kickoff": kickoff,
        "prob_1x2": {k: round(float(v), 4) for k, v in (prob or {}).items()},
        # Fix 2026-08-22: full model probability block (see docstring) --
        # direct O/U/BTTS audit without lambda replay.
        "model_probs": model_probs or None,
        "suggestion": suggestion or None,
        "odds_1x2": (
            {k: round(float(v), 4) for k, v in (odds or {}).items()} if odds else None
        ),
        "edge_pct": (
            {k: round(float(v), 2) for k, v in (edge or {}).items()} if edge else None
        ),
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "signal": int(signal) if signal is not None else None,
        "calibration": calibration or None,
        "model_version": model_version,
        "input_hash": input_hash,
        "best_pick": best_pick,
        "decision_type": decision_type or None,
        # Observability fix (2026-08-17): persist the decision engine's ACTUAL
        # final pick (market, selection, model_prob, market_odds, edge_pp, ev,
        # n_bucket, status) -- not just the tier label. Without it, settled
        # matches could never be scored against what the engine really bet.
        "final_decision": final_decision or None,
        "sources": sources or [],
        "features": features or None,
        "ml_prob": (
            {k: round(float(v), 4) for k, v in ml_prob.items() if k != "model"}
            if ml_prob else None
        ),
        "ml_model": (ml_prob or {}).get("model") if ml_prob else None,
        # Phase 2: which benchmark the logged edge was measured against. Prevents
        # a later sharp source from being silently mixed with soft-consensus edges.
        "edge_benchmark": edge_benchmark or None,
        # Plan B: movement signal (drift/steam/agreement) at prediction time,
        # so the movement-accuracy report can be recomputed without re-fetching.
        "movement": movement or None,
        # Plan B lineup trigger: the flashscore match URL lets the odds-poll loop
        # re-check lineup status near kickoff without re-resolving the fixture.
        "flashscore_url": flashscore_url or None,
        # Layer 3: the signal-engine best pick emitted for THIS query (selection,
        # score, confidence, model signature, ts) -- the immutable prior the
        # repeated-query stability guard compares against.
        "signal_engine_pick": signal_engine_pick or None,
        # P5: pre-match context (predicted/confirmed lineups, missing players /
        # injuries, coaches) logged as STRUCTURED data on every snapshot so a
        # historical record accumulates. Context-only today -- the no-OOS-
        # evidence rule keeps it out of the model until a backtest (same
        # discipline as P4) validates it. Stored as-is; never edited.
        "context_data": context_data or None,
        # P4 (re-runnable): the FULL scored signal ranking (per-signal market,
        # selection, score, confidence, edge_pp, movement, components) so a
        # later backtest can RE-WEIGHT the stored components with candidate
        # weight sets and settle them against the final score -- no re-fetching,
        # no mirrored scoring logic. Only the best pick is stored in
        # ``signal_engine_pick``; this keeps the rest of the evidence.
        "signal_engine_ranking": signal_engine_ranking or None,
        # Phase 1.3 (leakage guard): provenance of the lineup evidence used by
        # the flag-gated lambda correction. A lineup fetched at/after kickoff
        # must be rejected as a model input -- the timestamp makes that
        # auditable per snapshot.
        "lineup_source": lineup_source or None,
        "lineup_ts": lineup_ts or None,
        # Phase 4.2: whether this signal was PAPER-TRADED (logged with no real
        # stake attached). This bot is advisory (no auto-bet), so every signal
        # is paper-traded until the Phase 4 DoD (CLV>0, ROI CI positive, n>=100,
        # Kelly g>0) passes for its segment.
        "paper_trade": bool(paper_trade),
        # G2 (2026-08-17): canonical identity of each side at prediction time
        # -- {home: {canonical_id, provider, provider_id, name}, away: {...}}.
        # The match_id stays name-derived (backward compatible), but the
        # entities give an ID-level audit trail and let the settle verifier
        # reject a result whose canonical id contradicts the snapshot.
        "entities": entities or None,
        "market_totals": market_totals or None,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _settled_newest(
    path: str | Path,
) -> dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, Any]]]:
    """Newest settled snapshot per canonical match -> (snapshot, settlement).

    A match queried repeatedly (or carrying pre-Fix-1 match_id variants)
    must contribute its probabilities ONCE -- its newest snapshot -- or any
    re-fit overweights that fixture. Same dedupe rule as ``_settled_records``
    so stats, hit-rate/ROI and calibration all count the same real-world
    match the same way.
    """
    rows = _read_lines(Path(path))
    settlements = {r["match_id"]: r for r in rows if r.get("event") == "settle"}
    newest: dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for s in (r for r in rows if r.get("event") == "snapshot"):
        if settlements.get(s.get("match_id")) is None:
            continue
        key = _match_dedupe_key(s)
        cur = newest.get(key)
        if cur is None or (s.get("ts") or "") > ((cur[0].get("ts") if cur else "") or ""):
            newest[key] = (s, settlements[s["match_id"]])
    return newest


def _side_pairs(snapshot: dict[str, Any], settlement: dict[str, Any]) -> list[tuple[float, int]]:
    """One (p_side, outcome) pair per 1X2 side of a settled snapshot.

    The outcome comes from the settlement line, so every pair is strictly
    prediction-then-outcome (no leakage by construction) -- the same design
    validate.py uses for its 4,560-sample calibration fit.
    """
    prob = snapshot.get("prob_1x2") or {}
    outcome = settlement.get("outcome", "")
    pairs: list[tuple[float, int]] = []
    for side in KEYS:
        p = prob.get(side)
        if p is None or not (0.0 < float(p) < 1.0):
            continue
        pairs.append((float(p), 1 if side == outcome else 0))
    return pairs


def calibration_pairs(path: str | Path) -> list[tuple[float, int]]:
    """(p_side, outcome) pairs from SETTLED snapshots for calibration re-fit.

    Global aggregation over every league. See ``calibration_pairs_by_league``
    for the per-league split (dynamic ``dyn:`` leagues included).
    """
    pairs: list[tuple[float, int]] = []
    for s, st in _settled_newest(path).values():
        pairs.extend(_side_pairs(s, st))
    return pairs


def calibration_pairs_by_league(path: str | Path) -> dict[str, list[tuple[float, int]]]:
    """(p_side, outcome) pairs grouped per league key, from settled snapshots.

    The league key is the first component of the canonical match_id
    (``{league}||{home}||{away}||{date}``) -- the SAME key
    ``league_calibrator`` derives its ``calibration_<slug>.json`` path from,
    so a dynamic league (``dyn:coppa-italia``) accumulates its own fit as
    its snapshots settle. Leagues without a settled sample simply do not
    appear. D2 (2026-08-17): unregistered leagues previously could never
    leave the ``uncalibrated_league`` cap -- this split is what lets them.
    """
    grouped: dict[str, list[tuple[float, int]]] = {}
    for s, st in _settled_newest(path).values():
        league = _match_dedupe_key(s)[0] or "unknown"
        grouped.setdefault(league, []).extend(_side_pairs(s, st))
    return grouped


def settle(
    path: str | Path,
    *,
    match_id: str,
    home_goals: int,
    away_goals: int,
    closing_odds: dict[str, float] | None = None,
) -> bool:
    """Append a settlement for a previously logged snapshot.

    Returns False (and does NOT append) when no snapshot with that match_id
    exists -- a settlement without a prediction is meaningless.
    """
    rows = _read_lines(Path(path))
    snap = next(
        (r for r in rows if r.get("event") == "snapshot" and _match_id_hits(r.get("match_id"), match_id)),
        None,
    )
    if snap is None:
        return False
    # P1.2: write the settle row with the SNAPSHOT's own match_id (its stored
    # id-space, possibly a legacy full-timestamp form) so list_unsettled and
    # later lookups never split a match across two id forms.
    snap_match_id = snap.get("match_id") or match_id
    outcome = "home" if home_goals > away_goals else ("draw" if home_goals == away_goals else "away")
    row = {
        "event": "settle",
        "match_id": snap_match_id,
        "ts": utc_now_iso(),
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "outcome": outcome,
        "closing_odds": (
            {k: round(float(v), 4) for k, v in (closing_odds or {}).items()}
            if closing_odds
            else None
        ),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def _pick_tier(pick: dict[str, Any], medium_score: float = 0.52) -> str:
    """K5: ``tier`` when stored; otherwise derived from score/confidence."""
    t = pick.get("tier")
    if t in ("BEST PICK", "LEAN"):
        return t
    try:
        low = float(pick.get("score") or 0.0) < medium_score
    except (TypeError, ValueError):
        low = False
    return "LEAN" if (low or pick.get("confidence") == "LOW") else "BEST PICK"


def classify_failure(
    snapshot: dict[str, Any],
    pick: dict[str, Any],
    result: str,
    *,
    kind: str = "best_pick",
    medium_score: float = 0.52,
) -> str | None:
    """Failure class of a LOST pick from the stored snapshot fields.

    Post-mortem 2026-08-28 classes (see the plan): K1 no evidence (both Elo
    on the prior, directional pick), K2 wrong entity (Elo out of band /
    source mismatch), K3 context ignored (second-leg tie), K4 forced
    suggestion (below the dominance floor), K5 weak pick published as BEST
    PICK (LEAN tier). ``K0`` = none of the above (market variance). Only
    losses are classified; returns None otherwise.
    """
    if result not in ("loss", "half_loss"):
        return None
    from .pick_gates import is_directional_selection  # lazy: avoid cycle

    mp = snapshot.get("model_probs") or {}
    f = snapshot.get("features") or {}
    ctx = snapshot.get("context_data") or {}
    audit = ctx.get("gate_audit") or {}
    market, sel = pick.get("market"), pick.get("selection") or pick.get("raw_label")
    directional = is_directional_selection(market, sel)

    def _elo(side: str) -> float | None:
        v = mp.get(f"elo_{side}", f.get(f"elo_{side}"))
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    eh, ea = _elo("home"), _elo("away")
    if audit.get("entity_mismatch") or any(
        v is not None and (v < 1300.0 or v > 2450.0) for v in (eh, ea)
    ):
        return "K2"
    hs, as_ = mp.get("elo_home_seeded"), mp.get("elo_away_seeded")
    if hs is None or as_ is None:
        both_prior = mp.get("elo_seeded") is False and eh == 1500.0 and ea == 1500.0
    else:
        both_prior = (not hs) and (not as_)
    if both_prior and directional:
        return "K1"
    ts = ctx.get("tie_state") or {}
    if ts:
        side = pick.get("side") or ("home" if str(sel).startswith("Home") else "away")
        if ts.get("state") == "decided" and directional and side == ts.get("leader"):
            return "K3"
        if ts.get("state") == "balanced" and (
            (market == "Total" and str(sel).startswith("Over"))
            or (market == "BTTS" and str(sel).endswith("Yes"))
        ):
            return "K3"
    if kind == "suggestion":
        try:
            adj = float(pick.get("adjusted_score"))
        except (TypeError, ValueError):
            adj = None
        if adj is not None and adj < medium_score:
            return "K4"
        return "K0"
    if _pick_tier(pick, medium_score) == "LEAN":
        return "K5"
    return "K0"


def best_pick_evaluation(path: str | Path, *, medium_score: float = 0.52) -> dict[str, Any]:
    """Settle every stored signal-engine BEST PICK (and SUGGESTION) against its result.

    Each snapshot stores ``signal_engine_pick`` (market, selection, line,
    side, score, confidence, tier) -- the pick the bot actually displayed --
    and, since 2026-08-28, ``suggestion`` (the market-only SUGGESTION TO
    PICK). This joins them with their settle rows (one per canonical match),
    settles each via the production ``settle_signal`` (quarter-line AH
    semantics included) and aggregates hit-rate / ROI per market, per tier
    (BEST PICK vs LEAN) and for the suggestion. Every LOSS is tagged with a
    ``failure_class`` (K1..K5 / K0) so the next fix targets the class that
    is actually losing. Odds come from the stored pick's ``market_odds`` when
    present, else from the matching ``signal_engine_ranking`` entry; without
    odds the pick still counts toward hit-rate but not ROI.
    """
    from .signal_engine import settle_signal  # lazy: avoid import cycle
    from .market_lean import suggestion_for_settlement

    rows = _read_lines(Path(path))
    settlements = {r["match_id"]: r for r in rows if r.get("event") == "settle"}

    # One snapshot per canonical match (newest with a stored pick OR suggestion).
    newest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for s in (r for r in rows if r.get("event") == "snapshot"
              and (r.get("signal_engine_pick") or (r.get("suggestion") or {}).get("pick"))):
        if settlements.get(s["match_id"]) is None:
            continue
        key = _match_dedupe_key(s)
        cur = newest.get(key)
        if cur is None or (s.get("ts") or "") > (cur.get("ts") or ""):
            newest[key] = s

    def _pick_odds(s: dict[str, Any]) -> float | None:
        pick = s.get("signal_engine_pick") or {}
        o = pick.get("market_odds")
        if o is not None:
            try:
                o = float(o)
                return o if o > 1.0 else None
            except (TypeError, ValueError):
                pass
        # Fallback: the matching entry in the stored ranking carries odds.
        sel = pick.get("selection")
        for e in (s.get("signal_engine_ranking") or []):
            if e.get("selection") == sel and e.get("market") == pick.get("market"):
                try:
                    o = float(e.get("market_odds") or 0.0)
                    return o if o > 1.0 else None
                except (TypeError, ValueError):
                    return None
        return None

    def _bucket(store: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        return store.setdefault(key, {
            "n": 0, "wins": 0, "pushes": 0, "losses": 0, "ret": 0.0, "staked": 0.0,
        })

    def _tally(b: dict[str, Any], res: str, stake_return: float, odds: float | None) -> float | None:
        b["n"] += 1
        if res in ("win", "half_win"):
            b["wins"] += 1
        elif res == "push":
            b["pushes"] += 1
        else:
            b["losses"] += 1
        if odds and odds > 1.0:
            b["staked"] += 1.0
            b["ret"] += stake_return * odds
            return round(stake_return * odds - 1.0, 4)
        return None

    def _finish(store: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for m, b in store.items():
            denom = b["n"]
            out[m] = {
                "n": denom,
                "win_rate": round((b["wins"] + 0.5 * b["pushes"]) / denom, 4) if denom else None,
                "wins": b["wins"],
                "pushes": b["pushes"],
                "losses": b["losses"],
                "roi_pct": (
                    round((b["ret"] - b["staked"]) / b["staked"] * 100.0, 2)
                    if b["staked"] > 0 else None
                ),
                "n_roi": int(b["staked"]),
            }
        return out

    markets: dict[str, dict[str, Any]] = {}
    tiers: dict[str, dict[str, Any]] = {}
    sug_markets: dict[str, dict[str, Any]] = {}
    failure_classes: dict[str, int] = {}
    sug_failure_classes: dict[str, int] = {}
    picks: list[dict[str, Any]] = []
    sug_picks: list[dict[str, Any]] = []
    n_bp = 0
    for s in newest.values():
        st = settlements[s["match_id"]]
        hg, ag = int(st.get("home_goals") or 0), int(st.get("away_goals") or 0)
        pick = s.get("signal_engine_pick") or {}
        if pick:
            n_bp += 1
            settled = settle_signal(pick, hg, ag)
            res = settled["result"]
            market = pick.get("market") or "?"
            tier = _pick_tier(pick, medium_score)
            odds = _pick_odds(s)
            roi = _tally(_bucket(markets, market), res, float(settled["stake_return"]), odds)
            _tally(_bucket(tiers, tier), res, float(settled["stake_return"]), odds)
            fclass = classify_failure(s, pick, res, kind="best_pick", medium_score=medium_score)
            if fclass:
                failure_classes[fclass] = failure_classes.get(fclass, 0) + 1
            pick_row = {
                "match": s.get("match_id"),
                "league": s.get("league"),
                "market": market,
                "selection": pick.get("selection"),
                "confidence": pick.get("confidence"),
                "score": pick.get("score"),
                "tier": tier,
                "result": res,
                "odds": odds,
                "failure_class": fclass,
            }
            if roi is not None:
                pick_row["roi"] = roi
            picks.append(pick_row)
        sug = ((s.get("suggestion") or {}).get("pick")) or None
        if sug:
            sig = suggestion_for_settlement(sug)
            settled = settle_signal(sig, hg, ag) if sig else {"result": "n/a", "stake_return": 0.0}
            res = settled["result"]
            if res != "n/a":
                try:
                    s_odds = float(sug.get("odds") or 0.0)
                except (TypeError, ValueError):
                    s_odds = 0.0
                roi = _tally(_bucket(sug_markets, sug.get("market") or "?"), res,
                             float(settled["stake_return"]), s_odds if s_odds > 1.0 else None)
                fclass = classify_failure(s, sug, res, kind="suggestion", medium_score=medium_score)
                if fclass:
                    sug_failure_classes[fclass] = sug_failure_classes.get(fclass, 0) + 1
                row = {
                    "match": s.get("match_id"),
                    "league": s.get("league"),
                    "market": sug.get("market"),
                    "selection": sug.get("raw_label") or sug.get("label"),
                    "adjusted_score": sug.get("adjusted_score"),
                    "result": res,
                    "odds": s_odds if s_odds > 1.0 else None,
                    "failure_class": fclass,
                }
                if roi is not None:
                    row["roi"] = roi
                sug_picks.append(row)

    return {
        "n": n_bp,
        "markets": _finish(markets),
        "tiers": _finish(tiers),
        "picks": picks,
        "failure_classes": failure_classes,
        "suggestion": {
            "n": len(sug_picks),
            "markets": _finish(sug_markets),
            "picks": sug_picks,
            "failure_classes": sug_failure_classes,
        },
    }

def dedupe_settles(path: str | Path) -> dict[str, Any]:
    """Rewrite the log keeping ONE settle row per canonical match.

    Pre-Fix-1 snapshots carried match_id variants for the same fixture
    ("Rio Ave FC" vs "Rio Ave") and repeated queries produced many
    snapshots for one match -- each got its own settle row, so the log
    accrued duplicate settlements (24 rows for ~10 real matches). This
    removes every settle row except the newest per canonical match key
    (``_match_dedupe_key``); snapshot and odds_snapshot rows are untouched.
    Rewrites the file atomically (temp + rename); returns
    {removed, kept, file}.
    """
    p = Path(path)
    rows = _read_lines(p)
    if not rows:
        return {"removed": 0, "kept": 0, "file": str(p)}
    newest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("event") != "settle":
            continue
        key = _match_dedupe_key(r)
        cur = newest.get(key)
        if cur is None or (r.get("ts") or "") > (cur.get("ts") or ""):
            newest[key] = r
    kept_ids = {id(r): r for r in newest.values()}
    out: list[dict[str, Any]] = []
    removed = 0
    for r in rows:
        if r.get("event") == "settle" and id(r) not in kept_ids:
            removed += 1
            continue
        out.append(r)
    if removed:
        tmp = p.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(p)
    return {"removed": removed, "kept": len(newest), "file": str(p)}


def _round_market(value: Any) -> dict[str, Any] | None:
    """Normalize a market dict ({line, home, away} / {line, over, under})
    to rounded floats, or None when empty/non-numeric."""
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in value.items():
        if v is None or v == "":
            continue
        try:
            out[k] = round(float(v), 4)
        except (TypeError, ValueError):
            continue
    return out or None


def append_odds_snapshot(
    path: str | Path,
    *,
    match_id: str,
    timing: str,
    odds: dict[str, float] | None,
    bookmakers_count: int | None = None,
    sources: list[str] | None = None,
    odds_ah: dict[str, Any] | None = None,
    odds_ou: dict[str, Any] | None = None,
) -> bool:
    """Append one immutable odds observation for a match (PHASE 32-33).

    ``timing`` is one of ODDS_TIMING_LABELS (or a custom label) describing
    when relative to kickoff the price was captured (T-24h / T-6h / T-1h /
    T-15m / T-0h). Each observation is a separate append-only line keyed by
    the same match_id as the prediction snapshot, so the historical evaluator
    can reconstruct the price curve and compute CLV (closing vs prediction
    price) without ever mixing snapshots.

    ``odds_ah`` / ``odds_ou`` carry the Asian-Handicap and Over/Under
    consensus at the same instant: ``{line, home, away}`` and
    ``{line, over, under}`` respectively -- line AND both prices, so line
    movement is preserved separately from price movement.

    Returns False when no prediction snapshot exists for that match_id -- an
    odds observation without a prediction is meaningless.
    """
    rows = _read_lines(Path(path))
    if not any(r.get("event") == "snapshot" and _match_id_hits(r.get("match_id"), match_id) for r in rows):
        return False
    row = {
        "event": "odds_snapshot",
        "match_id": match_id,
        "ts": utc_now_iso(),
        "timing": str(timing),
        "odds_1x2": (
            {k: round(float(v), 4) for k, v in (odds or {}).items()} if odds else None
        ),
        "odds_ah": _round_market(odds_ah),
        "odds_ou": _round_market(odds_ou),
        "bookmakers_count": int(bookmakers_count) if bookmakers_count else None,
        "sources": sources or [],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Layer 1: pin the IMMUTABLE opening record per market on first
    # ingestion (idempotent -- later observations never overwrite it). The
    # pinned source is the one that recorded the first observation for that
    # market; ``opening_snapshot_lag_seconds`` is (first recorded snapshot
    # time) - (true market-open time, when provider metadata exposes it).
    # The providers we normalize expose no line-open timestamp, so it stays
    # None until such metadata exists (loggable/queryable for audit).
    for market, key in (("ou", "odds_ou"), ("ah", "odds_ah")):
        if not row.get(key):
            continue
        if opening_snapshot(path, match_id, market) is not None:
            continue
        os_row = {
            "event": "opening_snapshot",
            "match_id": match_id,
            "ts": row["ts"],
            key: row[key],
            "sources": row["sources"],
            "opening_snapshot_lag_seconds": None,
        }
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(os_row, ensure_ascii=False) + "\n")
    return True


def last_prediction_snapshot(path: str | Path, match_id: str) -> dict[str, Any] | None:
    """The most recent immutable prediction snapshot for a match, or None.

    Used by the Layer-3 stability guard as the prior pick record (its
    ``signal_engine_pick`` field holds the previous best pick + ts).
    """
    rows = [
        r for r in _read_lines(Path(path))
        if r.get("event") == "snapshot" and _match_id_hits(r.get("match_id"), match_id)
    ]
    if not rows:
        return None
    return max(rows, key=lambda r: r.get("ts") or "")


# Fase 2 anti-flap (2026-08-23): identity lock. A resolver flip must never
# silently fork the prediction log into contradictory parallel streams. Real
# cases that motivated this guard:
#   - "Forest" lawannya berubah Leeds -> Man Utd antar query (same league,
#     same date, home side equal, away side DIFFERENT -- at most one of the
#     two pairs can be a real fixture);
#   - Troyes lawan "PSG" yang diduga salah resolve dari Paris FC.
# Before a snapshot is appended, the canonical team pair is compared against
# stored history; a contradiction HOLDS the write (caller skips append and
# logs why) instead of recording a prediction for a match that was probably
# resolved to the wrong club.
IDENTITY_LOCK_MAX_AGE_DAYS = 7


def _identity_side_name(name: Any, league: str | None) -> str:
    return _identity_normalize(_canonical_team_name(str(name or ""), league))


def _identity_cid(entities_side: dict[str, Any] | None) -> str:
    return _identity_normalize(str(((entities_side or {}).get("canonical_id")) or ""))


def _sides_disagree(
    stored_name: str,
    stored_cid: str,
    cur_name: str,
    cur_cid: str,
) -> bool | None:
    """Tri-state side comparison: True = different club, False = same club,
    None = not comparable (both sides unknown / no overlap of evidence).

    Names are compared through the SAME canonicalization ``make_match_id``
    uses, so provider spelling variants never count as disagreement. A
    canonical-id mismatch is decisive when both ids exist (id-level evidence
    outranks name spelling).
    """
    if stored_cid and cur_cid:
        return stored_cid != cur_cid
    if not stored_name or not cur_name:
        return None
    return stored_name != cur_name


def identity_lock_check(
    path: str | Path,
    *,
    match_id: str,
    home: str,
    away: str,
    entities: dict[str, Any] | None = None,
    now_ts: str | None = None,
) -> dict[str, Any] | None:
    """Detect a resolver flip BEFORE a new snapshot is appended.

    Two contradiction shapes are checked against stored snapshots:

    1. same-id contradiction -- an existing row under THIS match_id whose
       canonical team pair differs from the current pair (the id stayed but
       one club was re-resolved; the Troyes/"PSG" <- Paris FC shape).
    2. opponent flip -- any recent snapshot in the SAME league on the SAME
       kickoff DATE sharing exactly ONE side with the current pair while the
       other side is a DIFFERENT club (the Forest Leeds->Man Utd shape). The
       differing match_id proves it is a separate stream, which is precisely
       the damage this guard prevents.

    One team plays at most once per league per calendar day, so shape 2 has
    essentially no false positives while catching every re-resolution flip.
    Only snapshots within ``IDENTITY_LOCK_MAX_AGE_DAYS`` are considered.

    Performance note (2026-08-23): the scan is structured CHEAP-FIRST -- age
    filter and raw match_id string comparisons run on every row, while the
    expensive per-side team-name canonicalization (teams.json fuzzy resolve,
    ~3ms/row measured) only ever runs on the handful of rows that pass the
    league+date pre-filter or hit the same-id branch. On a real production
    log (~2.6k rows / 2.9MB) this keeps the whole check around ~0.3s instead
    of multiple seconds, with IDENTICAL verdicts to a full-canonicalization
    scan.

    Returns ``{"locked": True, "kind", "reason", "conflict_match_id"}`` when
    the write should be held, else None. Never raises on malformed rows.
    """
    rows = [r for r in _read_lines(Path(path)) if r.get("event") == "snapshot"]
    if not rows:
        return None

    parts = str(match_id or "").split("||")
    if len(parts) < 4:
        return None
    cur_league = parts[0]
    cur_league_norm = _identity_normalize(cur_league)
    cur_date = _kickoff_date_component(parts[3])
    ent_home = (entities or {}).get("home") or {}
    ent_away = (entities or {}).get("away") or {}
    cur_home_name = _identity_side_name(ent_home.get("name") or home, cur_league)
    cur_away_name = _identity_side_name(ent_away.get("name") or away, cur_league)
    cur_home_cid = _identity_cid(ent_home)
    cur_away_cid = _identity_cid(ent_away)

    now_dt = _as_utc_dt(now_ts) if now_ts else _as_utc_dt(utc_now_iso())
    from datetime import timedelta

    max_age = timedelta(days=IDENTITY_LOCK_MAX_AGE_DAYS)

    def _row_sides(r: dict[str, Any]) -> tuple[str, str, str, str] | None:
        """(league, date, home_name, away_name) resolvable for comparison."""
        rp = str(r.get("match_id") or "").split("||")
        if len(rp) < 4:
            return None
        rent = r.get("entities") or {}
        rh = rent.get("home") or {}
        ra = rent.get("away") or {}
        league = rp[0]
        # cid-form components carry no name; fall back to the persisted
        # entities (G2+) -- rows without either stay out of the flip scan
        # rather than risking a false positive.
        rh_src = rp[1] if not rp[1].startswith("cid:") else (rh.get("name") or "")
        ra_src = rp[2] if not rp[2].startswith("cid:") else (ra.get("name") or "")
        if not rh_src or not ra_src:
            return None
        return (
            _identity_normalize(league),
            _kickoff_date_component(rp[3]),
            _identity_side_name(rh.get("name") or rh_src, league),
            _identity_side_name(ra.get("name") or ra_src, league),
        )

    def _row_cid(r: dict[str, Any], side: str) -> str:
        return _identity_cid(((r.get("entities") or {}).get(side)) or {})

    flip_conflicts: list[dict[str, Any]] = []
    for r in rows:
        # Age guard FIRST (cheap datetime parse): old rows never implicate
        # the resolver's current state regardless of shape.
        if now_dt is not None:
            ts_dt = _as_utc_dt(r.get("ts"))
            if ts_dt is not None and (now_dt - ts_dt) > max_age:
                continue

        rid = str(r.get("match_id") or "")

        # Shape 1: same match_id, contradicting pair (entity ids decisive
        # when both exist; names decide otherwise). Only this match's own
        # history reaches here -- a handful of rows.
        if _match_id_hits(rid, match_id):
            h_dis = _sides_disagree(
                _identity_side_name((r.get("home") or ""), cur_league),
                _row_cid(r, "home"),
                cur_home_name,
                cur_home_cid,
            )
            a_dis = _sides_disagree(
                _identity_side_name((r.get("away") or ""), cur_league),
                _row_cid(r, "away"),
                cur_away_name,
                cur_away_cid,
            )
            if (h_dis is True) != (a_dis is True):
                return {
                    "locked": True,
                    "kind": "same_id",
                    "reason": (
                        f"match_id sama tapi pasangan tim berubah: "
                        f"{r.get('home')} vs {r.get('away')} -> {home} vs {away}"
                    ),
                    "conflict_match_id": r.get("match_id"),
                }
            continue  # same id + consistent pair: no flip evidence here

        # Shape 2 CHEAP PRE-FILTER (plain string ops, no teams.json): only a
        # row in the SAME league on the SAME kickoff date can be an opponent
        # flip; everything else is skipped without paying canonicalization.
        rp = rid.split("||")
        if len(rp) < 4:
            continue
        if _identity_normalize(rp[0]) != cur_league_norm:
            continue
        if _kickoff_date_component(rp[3]) != cur_date:
            continue

        sides = _row_sides(r)
        if sides is None:
            continue
        s_league, s_date, s_home, s_away = sides
        h_same = s_home == cur_home_name
        a_same = s_away == cur_away_name
        if h_same == a_same:
            continue  # both equal (=same fixture) or both differ (unrelated)
        other_stored, other_cur = (
            (s_away, cur_away_name) if h_same else (s_home, cur_home_name)
        )
        if not other_stored or other_stored == other_cur:
            continue
        fixed_side = "home" if h_same else "away"
        flip_conflicts.append({
            "_ts": r.get("ts") or "",
            "locked": True,
            "kind": "opponent_flip",
            "reason": (
                f"sisi {fixed_side} sama di {s_league} tanggal {s_date}, "
                f"tapi lawan berubah: {other_stored} -> {other_cur} "
                "(diduga resolver flip)"
            ),
            "conflict_match_id": r.get("match_id"),
        })

    if flip_conflicts:
        # Deterministic report: the NEWEST conflict wins (file order is
        # append-order, so sort by ts instead of relying on scan order).
        flip_conflicts.sort(key=lambda c: c.pop("_ts"), reverse=True)
        return flip_conflicts[0]
    return None


def opening_snapshot(
    path: str | Path,
    match_id: str,
    market: str | None = None,
) -> dict[str, Any] | None:
    """The IMMUTABLE Layer-1 opening record for a match (one per market).

    Pinned on first ingestion by ``append_odds_snapshot``: the earliest
    observed prices for the Over/Under (``market="ou"``) or Asian Handicap
    (``market="ah"``) market. Every downstream consumer -- movement scoring,
    display MARKET block, Layer-3 market-move check -- reads opening from
    here and never re-derives it from per-source ``opening_price`` fields.
    Returns None when no odds observation for that market exists yet.
    """
    rows = [
        r for r in _read_lines(Path(path))
        if r.get("event") == "opening_snapshot" and _match_id_hits(r.get("match_id"), match_id)
    ]
    if not rows:
        return None
    if market is None:
        return rows[0]
    key = "odds_ou" if market == "ou" else "odds_ah"
    for r in rows:
        if r.get(key):
            return r
    # No record pinned for this market yet -> None (a non-None fallback here
    # would make the creation loop in append_odds_snapshot believe the
    # opening already exists and never pin it).
    return None


def stability_calibration(
    path: str | Path,
    *,
    percentile: float = 0.95,
    min_samples: int = 20,
    fallback: float = 0.05,
) -> dict[str, Any]:
    """Calibrate the Layer-3 score-delta threshold from logged data.

    Collects |score_delta| between CONSECUTIVE logged signal-engine picks for
    the same match (repeated queries where nothing should have changed) and
    returns the ``percentile``-th percentile as the threshold. Below
    ``min_samples`` the ``fallback`` is returned (``calibrated`` False) --
    the threshold becomes data-derived only once post-Layer-1/2 history
    accumulates, per spec (never a guessed constant once data exists).
    """
    rows = _read_lines(Path(path))
    by_match: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("event") != "snapshot":
            continue
        by_match.setdefault(str(r.get("match_id")), []).append(r)
    deltas: list[float] = []
    for match_rows in by_match.values():
        ordered = sorted(match_rows, key=lambda r: r.get("ts") or "")
        prev_score: float | None = None
        for r in ordered:
            p = (r.get("signal_engine_pick") or {}).get("score")
            if p is None:
                continue
            if prev_score is not None:
                deltas.append(abs(float(p) - prev_score))
            prev_score = float(p)
    if len(deltas) < min_samples:
        return {
            "threshold": float(fallback), "n": len(deltas),
            "percentile": percentile, "calibrated": False,
        }
    s = sorted(deltas)
    idx = min(len(s) - 1, int(percentile * len(s)))
    return {
        "threshold": float(s[idx]), "n": len(deltas),
        "percentile": percentile, "calibrated": True,
    }


def list_odds_snapshots(path: str | Path, match_id: str | None = None) -> list[dict[str, Any]]:
    """All odds-snapshot rows, optionally filtered to one match, ordered by
    their write timestamp (immutable -- the order of lines is the order of
    capture)."""
    rows = _read_lines(Path(path))
    snaps = [
        r for r in rows
        if r.get("event") == "odds_snapshot"
        and (match_id is None or _match_id_hits(r.get("match_id"), match_id))
    ]
    return sorted(snaps, key=lambda r: r.get("ts") or "")


def odds_snapshots_by_match(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """match_id -> chronological odds observations (all timings)."""
    by_match: dict[str, list[dict[str, Any]]] = {}
    for row in list_odds_snapshots(path):
        by_match.setdefault(row["match_id"], []).append(row)
    return by_match


def _bucket_label(value: float | None, edges: list[float]) -> str:
    """Bucket a numeric value into 'edge' style ranges: 0-5%, 5-10% etc."""
    if value is None:
        return "n/a"
    prev = 0.0
    for e in edges:
        if value < e:
            return f"{prev:.0f}-{e:.0f}%"
        prev = e
    return f"{prev:.0f}%+"


def _max_drawdown(net_series: list[float]) -> float | None:
    """Max peak-to-trough drawdown of the cumulative net-stake curve."""
    if not net_series:
        return None
    peak = 0.0
    cum = 0.0
    worst = 0.0
    for net in net_series:
        cum += net
        peak = max(peak, cum)
        if peak > 0:
            worst = min(worst, (cum - peak) / peak)
    return worst


def _sharpe(returns: list[float]) -> float | None:
    """Annualized-ish Sharpe from per-bet returns (flat stake = stake units).
    Uses sqrt(bets) as the time aggregation so it stays comparable across
    different sample sizes; None when < 2 bets or zero variance."""
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var) * math.sqrt(len(returns))


def _confidence_label(confidence: float | None) -> str:
    if confidence is None:
        return "n/a"
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.50:
        return "MEDIUM"
    return "LOW"


# A real PRE-MATCH 1X2 price this bot can actually back never exceeds this
# bound: ~51.0 corresponds to a 2% implied probability, beyond any closing
# line the covered leagues (big-5 + minors + qualifiers) actually offer.
# Anything above is treated as a data-quality artifact (a NowGoal t=11
# in-play/realtime leg leaking into "closing", a stale sentinel, or a
# misparsed longshot) and is EXCLUDED from CLV/closing-reference math so
# one bad settle cannot shift the aggregate edge/CLV gates.
MAX_PLAUSIBLE_ODDS = 51.0


def _is_plausible_price(price: Any) -> bool:
    """True for a closing/backing price the bot can meaningfully evaluate.

    Decimal odds must be a finite number in (1.0, MAX_PLAUSIBLE_ODDS];
    prices at or below 1.0 (no-value sentinels) and absurd longshots are
    rejected the same way on the settle path and the last-snapshot fallback.
    """
    try:
        f = float(price)
    except (TypeError, ValueError):
        return False
    return 1.0 < f <= MAX_PLAUSIBLE_ODDS


def _as_utc_dt(value: Any) -> Any:
    """Parse an ISO-8601 timestamp to an aware UTC datetime, else None."""
    from datetime import datetime, timezone

    if not value:
        return None
    try:
        cleaned = str(value)[:-1] + "+00:00" if str(value).endswith("Z") else str(value)
        dt = datetime.fromisoformat(cleaned)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _last_pre_kickoff_close(
    s: dict[str, Any],
    odds_snaps: list[dict[str, Any]],
    pick: str,
) -> dict[str, float] | None:
    """Closing-reference fallback: the LAST PRE-KICKOFF odds snapshot's 1X2
    prices for ``pick`` (the T-0h ``l`` leg the odds poll captured closest to
    kickoff). Used only when the settlement carries no closing_odds (NowGoal
    t=11 empty/disabled), so price CLV is still computed against the closest
    available pre-match price. In-play captures (ts >= kickoff) never count.
    Returns a {home, draw, away}-shaped dict restricted to valid (>1.0)
    prices, or None when no pre-kickoff snapshot carries a valid price."""
    candidates = [
        r for r in odds_snaps
        if _is_plausible_price((r.get("odds_1x2") or {}).get(pick))
    ]
    if not candidates:
        return None
    kickoff_dt = _as_utc_dt(s.get("kickoff"))
    if kickoff_dt is not None:
        pre = []
        for r in candidates:
            ts_dt = _as_utc_dt(r.get("ts"))
            if ts_dt is not None and ts_dt <= kickoff_dt:
                pre.append(r)
        if not pre:
            return None
        candidates = pre
    last = max(candidates, key=lambda r: r.get("ts") or "")
    odds_1x2 = last.get("odds_1x2") or {}
    out = {
        k: float(v) for k, v in odds_1x2.items()
        if k in KEYS and _is_plausible_price(v)
    }
    return out or None


def _settled_records(
    rows: list[dict[str, Any]],
    edge_threshold: float,
) -> list[dict[str, Any]]:
    """Join snapshots + settlements into one record per settled snapshot.

    Single source of truth for ROI / edge / CLV / log-loss so ``compute_stats``
    and ``similar_signal_stats`` can never drift apart. Edge is margin-free and
    expressed in percentage points; ROI gates on edge >= threshold (fraction).

    CLV split (PHASE 33 -- forecast quality vs price quality are separate
    measurements):
      - ``clv``           : model-based  = P(pick) * closing_odds - 1
      - ``price_clv``     : price-based  = closing_odds / prediction_odds - 1
                            for the pick (positive = price drifted in the
                            pick's favour between prediction and close)
      - ``price_clv_by_timing``: the SAME price CLV computed against the odds
                            captured at each T-24h/T-6h/T-1h snapshot, when
                            present -- lets the evaluator see whether earlier
                            or later capture won the line.
    """
    settlements = {r["match_id"]: r for r in rows if r.get("event") == "settle"}
    odds_snaps: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("event") == "odds_snapshot":
            odds_snaps.setdefault(r["match_id"], []).append(r)

    # Dedupe per canonical match: when several snapshots exist for the same
    # fixture (repeated queries, pre-Fix-1 match_id variants), the NEWEST
    # snapshot represents the match so hit-rate/ROI/CLV never count one
    # real-world match more than once.
    newest_snapshot: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for s in (r for r in rows if r.get("event") == "snapshot"):
        if settlements.get(s["match_id"]) is None:
            continue
        key = _match_dedupe_key(s)
        cur = newest_snapshot.get(key)
        if cur is None or (s.get("ts") or "") > (cur.get("ts") or ""):
            newest_snapshot[key] = s

    settled: list[dict[str, Any]] = []
    for s in newest_snapshot.values():
        st = settlements.get(s["match_id"])
        if st is None:
            continue
        prob = s.get("prob_1x2") or {}
        odds = s.get("odds_1x2") or {}
        outcome = st.get("outcome", "")
        p_out = prob.get(outcome, 0.0)
        logloss = -math.log(max(1e-9, p_out)) if p_out > 0 else None
        pick = max(prob, key=prob.get) if prob else None
        hit = bool(pick and pick == outcome)
        predicted = bool(prob)

        roi = None
        edge = None
        if pick and odds.get(pick, 0) > 1.0:
            raw = {k: (1.0 / odds[k] if odds.get(k, 0) > 1.0 else 0.0) for k in KEYS}
            total = sum(raw.values())
            if total > 0:
                norm = {k: v / total for k, v in raw.items()}
                edge = (prob.get(pick, 0.0) - norm.get(pick, 0.0)) * 100.0
                if edge >= edge_threshold * 100.0:
                    roi = (odds[pick] - 1.0) if hit else -1.0
        if edge is None:
            edge_pct = s.get("edge_pct") or {}
            edge = edge_pct.get(pick or "") if pick else None

        close = st.get("closing_odds") or {}
        # CLV closing reference (2026-08-16): the settlement's ``closing_odds``
        # is the true closing line when present; otherwise fall back to the
        # LAST PRE-KICKOFF odds snapshot (the ``l`` leg the poll captured
        # closest to kickoff) so price CLV / edge-bucket-vs-closing still have
        # a real reference even when the NowGoal t=11 closing fetch is empty
        # or disabled. In-play snapshots never count as closing. Both paths
        # reject implausible prices (<= 1.0 or absurd longshots) so one bad
        # settle can never shift the aggregate CLV/edge gates.
        closing_source: str | None = "settle"
        if not (pick and _is_plausible_price(close.get(pick))):
            close = (
                _last_pre_kickoff_close(s, odds_snaps.get(s["match_id"], []), pick) or {}
            )
            closing_source = "last_snapshot" if close else None
        clv = None
        price_clv = None
        price_clv_by_timing: dict[str, float] = {}
        if pick:
            pred_odds = odds.get(pick, 0.0)
            if pick and _is_plausible_price(close.get(pick)):
                clv = prob.get(pick, 0.0) * close[pick] - 1.0
                if pred_odds > 1.0:
                    # Price CLV: how the line moved for the pick from the
                    # prediction price to close (positive = we got the line
                    # on our side; independent of whether the pick won).
                    price_clv = close[pick] / pred_odds - 1.0
            # One CLV value per timing LABEL (last capture wins for a label;
            # snapshots are never averaged/mixed across labels). The join
            # uses all odds_snapshot rows regardless of file order -- the
            # timing label is authoritative, not line position. In-play
            # captures (ts >= kickoff) are EXCLUDED: the poll labels them
            # "T-0h" like the last pre-match capture, so without this they
            # would overwrite the pre-match bucket with an in-play price.
            kickoff_dt = _as_utc_dt(s.get("kickoff"))
            for osnap in sorted(odds_snaps.get(s["match_id"], []), key=lambda r: r.get("ts") or ""):
                snap_ts_dt = _as_utc_dt(osnap.get("ts"))
                if kickoff_dt is not None and (snap_ts_dt is None or snap_ts_dt > kickoff_dt):
                    continue
                snap_odds = (osnap.get("odds_1x2") or {}).get(pick, 0.0)
                if _is_plausible_price(snap_odds) and _is_plausible_price(close.get(pick)):
                    price_clv_by_timing[osnap.get("timing", "?")] = (
                        close[pick] / snap_odds - 1.0
                    )

        settled.append(
            {
                "match_id": s["match_id"],
                # Chronological order for drawdown/sharpe: snapshot ts first,
                # kickoff as fallback, then match_id as last resort.
                "ts": s.get("ts") or s.get("kickoff") or s["match_id"],
                "pick": pick,
                "outcome": outcome,
                "hit": hit,
                "predicted": predicted,
                "logloss": logloss,
                "roi": roi,
                "clv": clv,
                "price_clv": price_clv,
                # Phase 5.4: the closing price of the pick itself, so the
                # edge-bucket audit can recompute ROI against the CLOSING
                # line (closing_odds - 1 on a win, -1 on a loss).
                "close_odds": (
                    float(close[pick]) if (pick and _is_plausible_price(close.get(pick))) else None
                ),
                "price_clv_by_timing": price_clv_by_timing,
                # Which reference produced the CLV numbers: "settle" (true
                # closing_odds on the settlement) or "last_snapshot" (the
                # fallback to the last pre-kickoff odds capture). None when
                # no closing reference exists at all.
                "closing_source": closing_source,
                "n_odds_snapshots": len(odds_snaps.get(s["match_id"], [])),
                "confidence": s.get("confidence"),
                "signal": s.get("signal"),
                "edge": edge,
                "decision_type": s.get("decision_type"),
                # Phase 3 CLV gate: segment identity (league x market x tier)
                # so realized CLV/ROI can be aggregated per segment.
                "league": s.get("league"),
                "market": (s.get("best_pick") or {}).get("market"),
            }
        )
    return settled


def segment_clv_stats(
    path: str | Path,
    *,
    edge_threshold: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """Realized price CLV + ROI per (league x market x decision tier) segment.

    Phase 3 CLV gate input. Keys are ``"{league}|{market}|{tier}"``. Only
    settled snapshots that actually carried a 1X2 prediction contribute;
    ``price_clv_pct`` is the average closing/prediction − 1 (positive = the
    line moved in the pick's favour between prediction and close).
    """
    rows = _read_lines(Path(path))
    settled = _settled_records(rows, edge_threshold)
    agg: dict[str, dict[str, Any]] = {}
    for x in settled:
        if not x.get("predicted"):
            continue
        key = f"{x.get('league') or '?'}|{x.get('market') or '?'}|{x.get('decision_type') or '?'}"
        b = agg.setdefault(
            key,
            {
                "league": x.get("league"),
                "market": x.get("market"),
                "tier": x.get("decision_type"),
                "n": 0,
                "price_clvs": [],
                "rois": [],
            },
        )
        b["n"] += 1
        if x.get("price_clv") is not None:
            b["price_clvs"].append(x["price_clv"])
        if x.get("roi") is not None:
            b["rois"].append(x["roi"])
    out: dict[str, dict[str, Any]] = {}
    for key, b in agg.items():
        clvs = b["price_clvs"]
        rois = b["rois"]
        out[key] = {
            "league": b["league"],
            "market": b["market"],
            "tier": b["tier"],
            "n": b["n"],
            "price_clv_pct": round(sum(clvs) / len(clvs) * 100.0, 2) if clvs else None,
            "roi": round(sum(rois) / len(rois), 4) if rois else None,
            "n_clv": len(clvs),
            # Phase 0.3: count of settled bets whose realized price CLV is
            # strictly positive -- input for the Wilson score interval on the
            # CLV-positivity rate (gate opens only with statistically
            # meaningful evidence, half-width <= max_ci_halfwidth).
            "n_positive_clv": sum(1 for v in clvs if v > 0),
        }
    return out


def clv_segment_report(
    path: str | Path,
    out_dir: str | Path = "reports",
    date: str | None = None,
    edge_threshold: float = 0.02,
) -> dict[str, Any]:
    """CLV segment report (Phase 0.4): per league x market x timing bucket.

    Aggregates the SAME price-CLV-by-timing data ``compute_stats`` reports
    (closing/prediction − 1 per odds snapshot) into a per-segment table with
    n, mean price CLV, ROI and a Wilson confidence interval on the
    CLV-positivity rate. Writes ``reports/clv_segments_<date>.json`` and
    returns the payload. Also reports closing-odds coverage (share of
    settled matches with a non-null closing_odds) -- the Phase 0 Definition
    of Done threshold (>= 80%).

    Segment = (league, market, timing bucket). ``timing="ALL"`` rows are the
    overall price CLV per (league, market) from the prediction price itself.
    """
    rows = _read_lines(Path(path))
    settled = _settled_records(rows, edge_threshold)
    n_settled = sum(1 for x in settled if x.get("predicted"))
    n_with_closing = sum(
        1 for x in settled if x.get("predicted") and x.get("price_clv") is not None
    )
    coverage_pct = round(n_with_closing / n_settled * 100.0, 1) if n_settled else 0.0
    # Honest source breakdown: "settle" = real closing_odds on the settlement;
    # "last_snapshot" = the fallback to the last pre-kickoff odds capture.
    closing_by_source: dict[str, int] = {}
    for x in settled:
        if x.get("predicted") and x.get("price_clv") is not None:
            src = x.get("closing_source") or "unknown"
            closing_by_source[src] = closing_by_source.get(src, 0) + 1

    from .clv_gate import wilson_interval

    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for x in settled:
        if not x.get("predicted"):
            continue
        league = x.get("league") or "?"
        market = x.get("market") or "?"
        roi = x.get("roi")
        # Per-timing buckets from the odds snapshots.
        for timing, v in (x.get("price_clv_by_timing") or {}).items():
            b = agg.setdefault(
                (league, market, timing),
                {"league": league, "market": market, "timing": timing, "clvs": [], "rois": []},
            )
            b["clvs"].append(v)
            if roi is not None:
                b["rois"].append(roi)
        # Overall (ALL) bucket from the prediction price itself.
        if x.get("price_clv") is not None:
            b = agg.setdefault(
                (league, market, "ALL"),
                {"league": league, "market": market, "timing": "ALL", "clvs": [], "rois": []},
            )
            b["clvs"].append(x["price_clv"])
            if roi is not None:
                b["rois"].append(roi)

    segments: list[dict[str, Any]] = []
    for key in sorted(agg):
        b = agg[key]
        clvs = b["clvs"]
        n = len(clvs)
        n_pos = sum(1 for v in clvs if v > 0)
        ci = wilson_interval(n_pos, n)
        mean_clv = round(sum(clvs) / n * 100.0, 2) if n else None
        rois = b["rois"]
        segments.append(
            {
                "league": b["league"],
                "market": b["market"],
                "timing": b["timing"],
                "n": n,
                "price_clv_pct": mean_clv,
                "roi": round(sum(rois) / len(rois), 4) if rois else None,
                "ci": (
                    {
                        "centre": round(ci[0], 4),
                        "half_width": round(ci[1], 4),
                        "n_positive": n_pos,
                    }
                    if ci else None
                ),
            }
        )

    payload = {
        "generated_at": utc_now_iso(),
        "log": str(path),
        "date": date,
        "coverage": {
            "settled": n_settled,
            "with_closing_odds": n_with_closing,
            "closing_coverage_pct": coverage_pct,
            "threshold_pct": 80.0,
            "passed": n_settled > 0 and coverage_pct >= 80.0,
            "closing_by_source": closing_by_source,
        },
        "n_segments": len(segments),
        "segments": segments,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"clv_segments_{date or utc_now_iso()[:10]}.json"
    fpath = out / fname
    fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["file"] = str(fpath)
    return payload


EDGE_BUCKET_EDGES = [5.0, 10.0, 20.0]


def edge_bucket_closing_stats(
    path: str | Path,
    edge_threshold: float = 0.02,
    min_n: int = 10,
) -> dict[str, dict[str, Any]]:
    """Phase 5.4: realized ROI per edge bucket against CLOSING prices.

    The old bucket audit compared picks against OPENING prices -- but the
    only benchmark that means anything is the closing line. For every
    settled snapshot that carries closing odds, ROI is recomputed with the
    closing price (closing_odds - 1 on a win, -1 on a loss), grouped by the
    same edge buckets the live engine uses (0-5 / 5-10 / 10-20 / 20+pp).
    Returns {bucket: {n, roi_vs_closing, n_with_closing}}.
    """
    rows = _read_lines(Path(path))
    settled = _settled_records(rows, edge_threshold)
    agg: dict[str, dict[str, Any]] = {}
    for x in settled:
        if not x.get("predicted"):
            continue
        bucket = _bucket_label(x.get("edge"), EDGE_BUCKET_EDGES)
        b = agg.setdefault(bucket, {"n": 0, "rois_closing": [], "n_with_closing": 0})
        b["n"] += 1
        close = _closing_odds_for(x)
        if close is None:
            continue
        b["n_with_closing"] += 1
        hit = x.get("hit")
        b["rois_closing"].append((close - 1.0) if hit else -1.0)
    out: dict[str, dict[str, Any]] = {}
    for bucket, b in sorted(agg.items()):
        rois = b["rois_closing"]
        out[bucket] = {
            "n": b["n"],
            "n_with_closing": b["n_with_closing"],
            "roi_vs_closing": round(sum(rois) / len(rois), 4) if rois else None,
            "net_negative": bool(rois) and sum(rois) / len(rois) < 0.0,
        }
    return out


def _closing_odds_for(x: dict[str, Any]) -> float | None:
    """Closing price for the settled record's pick (or None)."""
    return x.get("close_odds")


def edge_bucket_gate(
    stats: dict[str, dict[str, Any]],
    edge_pp: float | None,
    min_n: int = 10,
) -> dict[str, Any]:
    """Phase 5.4 HARD filter: edge must not drive a recommendation in a
    net-negative bucket (measured against closing prices).

    Returns {allowed, bucket, n, n_with_closing, roi_vs_closing, reason}.
    A bucket with fewer than ``min_n`` closing-priced bets is not evidence
    (passes, flagged) -- the gate only blocks on measured net-negative ROI.
    """
    if edge_pp is None:
        return {"allowed": True, "bucket": None, "reason": None}
    bucket = _bucket_label(edge_pp, EDGE_BUCKET_EDGES)
    b = stats.get(bucket)
    if b is None or not b.get("n_with_closing"):
        return {
            "allowed": True, "bucket": bucket, "n": 0, "n_with_closing": 0,
            "roi_vs_closing": None,
            "reason": None,
        }
    roi = b.get("roi_vs_closing")
    if b["n_with_closing"] < min_n:
        return {
            "allowed": True, "bucket": bucket, "n": b["n"],
            "n_with_closing": b["n_with_closing"], "roi_vs_closing": roi,
            "reason": f"bucket {bucket} n={b['n_with_closing']} < {min_n} (belum evidence)",
        }
    if roi is not None and roi < 0.0:
        return {
            "allowed": False, "bucket": bucket, "n": b["n"],
            "n_with_closing": b["n_with_closing"], "roi_vs_closing": roi,
            "reason": (
                f"edge bucket {bucket} net-negative vs CLOSING "
                f"(ROI {roi:+.1%}, n={b['n_with_closing']}) — edge dilarang drive rekomendasi"
            ),
        }
    return {
        "allowed": True, "bucket": bucket, "n": b["n"],
        "n_with_closing": b["n_with_closing"], "roi_vs_closing": roi,
        "reason": None,
    }


def edge_bucket_audit(
    path: str | Path,
    out_dir: str | Path = "reports",
    date: str | None = None,
    edge_threshold: float = 0.02,
) -> dict[str, Any]:
    """Phase 5.4 automated audit: edge-bucket vs CLOSING-price ROI report.

    Writes ``reports/edge_buckets_<date>.json`` (bucket -> n, roi vs closing,
    net-negative flag) and returns the payload. Run on a schedule so a
    bucket turning net-negative blocks recommendations before it costs.
    """
    stats = edge_bucket_closing_stats(path, edge_threshold=edge_threshold)
    payload = {
        "generated_at": utc_now_iso(),
        "log": str(path),
        "date": date,
        "benchmark": "closing",
        "buckets": stats,
        "net_negative_buckets": sorted(
            k for k, b in stats.items() if b.get("net_negative")
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"edge_buckets_{date or utc_now_iso()[:10]}.json"
    fpath = out / fname
    fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["file"] = str(fpath)
    return payload


def compute_stats(path: str | Path, edge_threshold: float = 0.02) -> dict[str, Any]:
    """Aggregate realised metrics over settled snapshots.

    Prediction metrics: hit rate, avg log-loss. Betting metrics (flat-stake,
    edge >= threshold, real odds only): ROI, Max Drawdown, Sharpe, plus a
    breakdown per confidence / edge bucket and CLV over closing-odds settles.
    """
    rows = _read_lines(Path(path))
    settled = _settled_records(rows, edge_threshold)
    n_snapshots = sum(1 for r in rows if r.get("event") == "snapshot")
    # PHASE 32-33: odds observations captured at T-24h/T-6h/T-1h/... for
    # historical CLV evaluation. Reported separately from model CLV.
    odds_snap_rows = [r for r in rows if r.get("event") == "odds_snapshot"]
    n_odds_snapshots = len(odds_snap_rows)
    by_timing: dict[str, int] = {}
    for r in odds_snap_rows:
        by_timing[r.get("timing", "?")] = by_timing.get(r.get("timing", "?"), 0) + 1

    n = len(settled)
    predicted = sum(1 for x in settled if x.get("predicted"))
    if not n:
        return {
            "file": str(path),
            "n_snapshots": n_snapshots,
            "n_settled": 0,
            "n_predicted": 0,
            "hit_rate": None,
            "avg_logloss": None,
            "roi": None,
            "n_bets": 0,
            "clv_pct": None,
            "n_clv": 0,
            "price_clv_pct": None,
            "n_price_clv": 0,
            "clv_by_timing": {},
            "n_odds_snapshots": n_odds_snapshots,
            "odds_snapshots_by_timing": by_timing,
            "max_drawdown": None,
            "sharpe": None,
            "by_confidence": {},
            "by_edge": {},
            "by_decision": {},
        }
    loglosses = [x["logloss"] for x in settled if x["logloss"] is not None]
    rois = [x["roi"] for x in settled if x["roi"] is not None]
    clvs = [x["clv"] for x in settled if x["clv"] is not None]
    price_clvs = [x["price_clv"] for x in settled if x["price_clv"] is not None]
    # Aggregate price CLV per timing label (across matches that captured it).
    clv_by_timing: dict[str, list[float]] = {}
    for x in settled:
        for timing, v in (x.get("price_clv_by_timing") or {}).items():
            clv_by_timing.setdefault(timing, []).append(v)

    # Betting risk metrics over the flat-stake net series, ordered by
    # snapshot time (kickoff) so drawdown/sharpe reflect the real sequence.
    settled_sorted = sorted(settled, key=lambda x: x["ts"])
    net_series = [x["roi"] for x in settled_sorted if x["roi"] is not None]

    def _bucket_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        n_b = sum(1 for x in items if x["roi"] is not None)
        hits = sum(1 for x in items if x.get("hit"))
        n_pred = sum(1 for x in items if x.get("predicted"))
        rois_b = [x["roi"] for x in items if x["roi"] is not None]
        clvs_b = [x["clv"] for x in items if x["clv"] is not None]
        pclvs_b = [x["price_clv"] for x in items if x["price_clv"] is not None]
        return {
            "n": len(items),
            "n_bets": n_b,
            "hit_rate": round(hits / n_pred, 4) if n_pred else None,
            "roi": round(sum(rois_b) / len(rois_b), 4) if rois_b else None,
            "clv_pct": round(sum(clvs_b) / len(clvs_b) * 100.0, 2) if clvs_b else None,
            "n_clv": len(clvs_b),
            "price_clv_pct": round(sum(pclvs_b) / len(pclvs_b) * 100.0, 2) if pclvs_b else None,
            "n_price_clv": len(pclvs_b),
        }

    by_conf: dict[str, list[dict[str, Any]]] = {}
    for x in settled:
        by_conf.setdefault(_confidence_label(x.get("confidence")), []).append(x)
    by_edge: dict[str, list[dict[str, Any]]] = {}
    for x in settled:
        by_edge.setdefault(_bucket_label(x.get("edge"), [5.0, 10.0, 20.0]), []).append(x)
    # TODO-15: CLV/ROI tracked per DECISION TYPE (STRONG/GOOD/LEAN/WATCH/NO
    # BET/NO CLEAR DECISION) so the production loop can verify whether each
    # tier actually produces value -- the honest health check of the engine.
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for x in settled:
        dt = x.get("decision_type") or "unknown"
        by_decision.setdefault(dt, []).append(x)

    _md = _max_drawdown(net_series) if net_series else None
    _sh = _sharpe(net_series) if len(net_series) >= 2 else None

    return {
        "file": str(path),
        "n_snapshots": n_snapshots,
        "n_settled": n,
        # hit_rate only over snapshots that actually carried a 1X2 prediction
        # (empty prob_1x2 = no model output, must not count as a miss).
        "n_predicted": predicted,
        "hit_rate": (
            round(sum(1 for x in settled if x["hit"]) / predicted, 4) if predicted else None
        ),
        "avg_logloss": round(sum(loglosses) / len(loglosses), 4) if loglosses else None,
        "roi": round(sum(rois) / len(rois), 4) if rois else None,
        "n_bets": len(rois),
        "clv_pct": round(sum(clvs) / len(clvs) * 100.0, 2) if clvs else None,
        "n_clv": len(clvs),
        "price_clv_pct": round(sum(price_clvs) / len(price_clvs) * 100.0, 2) if price_clvs else None,
        "n_price_clv": len(price_clvs),
        "clv_by_timing": {
            t: round(sum(v) / len(v) * 100.0, 2) for t, v in sorted(clv_by_timing.items())
        } if clv_by_timing else {},
        "n_odds_snapshots": n_odds_snapshots,
        "odds_snapshots_by_timing": by_timing,
        "max_drawdown": round(_md, 4) if _md is not None else None,
        "sharpe": round(_sh, 3) if _sh is not None else None,
        "by_confidence": {k: _bucket_summary(v) for k, v in sorted(by_conf.items())},
        "by_edge": {k: _bucket_summary(v) for k, v in sorted(by_edge.items())},
        "by_decision": {k: _bucket_summary(v) for k, v in sorted(by_decision.items())},
    }


def format_stats(stats: dict[str, Any], edge_threshold: float = 0.02) -> str:
    def _fmt(v: Any, suffix: str = "") -> str:
        return "n/a" if v is None else f"{v}{suffix}"

    lines = [
        "PREDICTION LOG STATS",
        f"  file       : {stats['file']}",
        f"  snapshots  : {stats['n_snapshots']}",
        f"  settled    : {stats['n_settled']}",
        f"  hit rate   : {_fmt(stats['hit_rate'], '%')}  (best 1X2 pick; "
        f"{stats['n_predicted']} predicted)",
        f"  log-loss   : {_fmt(stats['avg_logloss'])}  (avg over settled)",
        f"  ROI        : {_fmt(stats['roi'], '%')}  ({stats['n_bets']} bets, "
        f"flat-stake, edge>={edge_threshold:.0%})",
        f"  Max Drawdown: {_fmt(stats.get('max_drawdown'), '%')}  "
        f"(peak-to-trough net curve)",
        f"  Sharpe     : {_fmt(stats.get('sharpe'))}  (per-bet, sqrt(n) scaling)",
        f"  CLV        : {_fmt(stats['clv_pct'], '%')}  ({stats['n_clv']} "
        "settled w/ closing odds)",
        f"  Price CLV  : {_fmt(stats.get('price_clv_pct'), '%')}  "
        f"({stats.get('n_price_clv', 0)} w/ prediction+closing; "
        "closing/prediction - 1, terpisah dari model CLV)",
    ]
    by_timing = stats.get("odds_snapshots_by_timing") or {}
    n_osnap = stats.get("n_odds_snapshots", 0)
    if n_osnap:
        lines.append(
            f"  odds snapshots: {n_osnap} "
            f"(timing: {', '.join(f'{k}={v}' for k, v in sorted(by_timing.items()))})"
        )
        clv_t = stats.get("clv_by_timing") or {}
        if clv_t:
            lines.append(
                f"  price CLV by timing: "
                f"{', '.join(f'{k} {v:+.2f}%' for k, v in sorted(clv_t.items()))}"
            )
    by_conf = stats.get("by_confidence") or {}
    if by_conf:
        lines.append("  per confidence:")
        for label, b in by_conf.items():
            lines.append(
                f"    {label:<8}: n={b['n']} bets={b['n_bets']} "
                f"hit={_fmt(b['hit_rate'], '%')} roi={_fmt(b['roi'], '%')} "
                f"clv={_fmt(b['clv_pct'], '%')} pclv={_fmt(b.get('price_clv_pct'), '%')}"
            )
    by_edge = stats.get("by_edge") or {}
    if by_edge:
        lines.append("  per edge bucket:")
        for label, b in by_edge.items():
            lines.append(
                f"    {label:<8}: n={b['n']} bets={b['n_bets']} "
                f"hit={_fmt(b['hit_rate'], '%')} roi={_fmt(b['roi'], '%')} "
                f"clv={_fmt(b['clv_pct'], '%')} pclv={_fmt(b.get('price_clv_pct'), '%')}"
            )
    by_decision = stats.get("by_decision") or {}
    if by_decision:
        lines.append("  per decision type (TODO-15):")
        for label, b in by_decision.items():
            lines.append(
                f"    {label:<16}: n={b['n']} bets={b['n_bets']} "
                f"hit={_fmt(b['hit_rate'], '%')} roi={_fmt(b['roi'], '%')} "
                f"clv={_fmt(b['clv_pct'], '%')} pclv={_fmt(b.get('price_clv_pct'), '%')}"
            )
    return "\n".join(lines)


def similar_signal_stats(
    path: str | Path,
    *,
    confidence: float | None = None,
    edge_pct: float | None = None,
    min_bucket_n: int = 5,
    edge_threshold: float = 0.02,
) -> dict[str, Any]:
    """Historical performance of SIMILAR signals (PHASE 4/8).

    Clusters settled snapshots into the same buckets the live prediction uses
    (confidence HIGH/MED/LOW, edge 0-5/5-10/10-20/20%+) and reports, per
    bucket, the realised hit rate, flat-stake ROI and CLV. A live pick with
    edge 8% and HIGH confidence can then be checked against what that exact
    bucket has actually done historically instead of trusting the edge alone.
    Returns the matching bucket summary (or None when the bucket has too few
    settled samples) plus the full bucket table.
    """
    rows = _read_lines(Path(path))
    settled = _settled_records(rows, edge_threshold)
    n_settled = len(settled)

    buckets: dict[str, dict[str, Any]] = {}

    def _buckets_for(conf: float | None, edge: float | None) -> tuple[str, str]:
        return _confidence_label(conf), _bucket_label(edge, [5.0, 10.0, 20.0])

    for x in settled:
        conf_label, edge_label = _buckets_for(x.get("confidence"), x.get("edge"))
        key = f"{conf_label}|{edge_label}"
        b = buckets.setdefault(
            key,
            {"n": 0, "hits": 0, "rois": [], "clvs": [], "label": f"{conf_label} • {edge_label}"},
        )
        b["n"] += 1
        b["hits"] += 1 if x.get("hit") else 0
        if x.get("roi") is not None:
            b["rois"].append(x["roi"])
        if x.get("clv") is not None:
            b["clvs"].append(x["clv"])

    table = {}
    for key, b in buckets.items():
        table[key] = {
            "label": b["label"],
            "n": b["n"],
            "hit_rate": round(b["hits"] / b["n"], 4),
            "roi": round(sum(b["rois"]) / len(b["rois"]), 4) if b["rois"] else None,
            "clv_pct": round(sum(b["clvs"]) / len(b["clvs"]) * 100.0, 2) if b["clvs"] else None,
            "n_bets": len(b["rois"]),
            "n_clv": len(b["clvs"]),
        }

    matching = None
    if confidence is not None or edge_pct is not None:
        conf_label = _confidence_label(confidence)
        edge_label = _bucket_label(edge_pct, [5.0, 10.0, 20.0])
        key = f"{conf_label}|{edge_label}"
        bucket = table.get(key)
        if bucket and bucket["n"] >= min_bucket_n:
            matching = {
                **bucket,
                "key": key,
                "confidence": conf_label,
                "edge": edge_label,
                "sufficient_sample": True,
            }
        else:
            matching = {
                "key": key,
                "confidence": conf_label,
                "edge": edge_label,
                "n": bucket["n"] if bucket else 0,
                "sufficient_sample": False,
                "min_bucket_n": min_bucket_n,
            }

    return {
        "n_settled": n_settled,
        "n_buckets": len(table),
        "table": table,
        "matching": matching,
    }


DEFAULT_LOG_PATH = "cache/football/predictions.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-prediction-log")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # --file is accepted both before the subcommand (global) and on it.
    parser.add_argument("--file", default=DEFAULT_LOG_PATH, help="JSONL log path")

    st = sub.add_parser("stats", help="aggregate realised metrics")
    st.add_argument("--edge-threshold", type=float, default=0.02)
    st.add_argument("--file", default=None, help="JSONL log path (override)")

    se = sub.add_parser("settle", help="append result for a logged snapshot")
    se.add_argument("--match-id", required=True)
    se.add_argument("--home-goals", type=int, required=True)
    se.add_argument("--away-goals", type=int, required=True)
    se.add_argument("--closing-odds", default=None,
                    help="closing 1X2 odds as home,draw,away (e.g. 1.62,4.30,4.60)")
    se.add_argument("--file", default=None, help="JSONL log path (override)")

    osn = sub.add_parser("odds-snapshot", help="append a timed odds observation (PHASE 32-33)")
    osn.add_argument("--match-id", required=True)
    osn.add_argument("--timing", required=True, choices=ODDS_TIMING_LABELS,
                     help="when relative to kickoff the price was captured")
    osn.add_argument("--odds", required=True,
                     help="1X2 odds as home,draw,away (e.g. 1.62,4.30,4.60)")
    osn.add_argument("--bookmakers", type=int, default=None)
    osn.add_argument("--sources", default=None, help="comma-separated source labels")
    osn.add_argument("--file", default=None, help="JSONL log path (override)")

    args = parser.parse_args(argv)
    log_path = args.file or DEFAULT_LOG_PATH
    if args.cmd == "stats":
        print(
            format_stats(
                compute_stats(log_path, edge_threshold=args.edge_threshold),
                edge_threshold=args.edge_threshold,
            )
        )
        return 0
    if args.cmd == "odds-snapshot":
        parts = [x.strip() for x in args.odds.split(",")]
        if len(parts) != 3:
            print("--odds harus 3 angka: home,draw,away", file=__import__("sys").stderr)
            return 2
        try:
            odds = dict(zip(KEYS, (float(p) for p in parts)))
        except ValueError:
            print("--odds harus numerik: home,draw,away", file=__import__("sys").stderr)
            return 2
        sources = [x.strip() for x in args.sources.split(",") if x.strip()] if args.sources else None
        ok = append_odds_snapshot(
            log_path, match_id=args.match_id, timing=args.timing, odds=odds,
            bookmakers_count=args.bookmakers, sources=sources,
        )
        if not ok:
            print(f"tidak ada snapshot prediksi untuk match_id '{args.match_id}' "
                  f"(file: {log_path})", file=__import__("sys").stderr)
            return 1
        print(f"odds snapshot {args.timing} tersimpan untuk {args.match_id}")
        return 0
    if args.cmd == "settle":
        closing = None
        if args.closing_odds:
            parts = [x.strip() for x in args.closing_odds.split(",")]
            if len(parts) == 3:
                closing = dict(zip(KEYS, (float(p) for p in parts)))
            else:
                print("--closing-odds harus 3 angka: home,draw,away", file=__import__("sys").stderr)
                return 2
        ok = settle(
            log_path, match_id=args.match_id,
            home_goals=args.home_goals, away_goals=args.away_goals,
            closing_odds=closing,
        )
        if not ok:
            print(f"tidak ada snapshot untuk match_id '{args.match_id}' "
                  f"(file: {log_path})", file=__import__("sys").stderr)
            return 1
        print(f"settled {args.match_id}: {args.home_goals}-{args.away_goals}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
