"""Identity firewall (plan 2026-08-24): refuse to analyse the WRONG match.

Historical wrong-match shapes this module kills BEFORE the analysis burns
its 300s budget (every one verified live in production):

- F3 (2026-08-17): user asked "Atlético Madrid vs Málaga" -- the alias
  hijack resolved it to Real Madrid CF and the pipeline analysed a PHANTOM
  fixture (Real Madrid vs Malaga).
- P0 (2026-08-24): bare "Barcelona" resolved to RCD Espanyol de Barcelona;
  "Club Atletico de Madrid" resolved to Real Madrid CF.
- Opponent flip (2026-08-23): "Forest" analysed vs Leeds on one run and vs
  Man Utd on another (same league, same date -- at most one is the real
  fixture). The post-hoc identity_lock_check HOLDS THE WRITE, but only
  after the full analysis already ran.

The firewall adds three PRE-FLIGHT gates (see analyse.find_specific_match):

G-A ``check_pair_identity`` on the DETECTED fixture (source_match context):
    query pair vs the provider-rendered fixture pair, before any browser
    render or provider chain runs.
G-B ``check_pair_identity`` on the RESOLVED teams (search_teams_pair /
    oddspapi): query pair vs the team identities every downstream fetch
    will key on.
G-C ``preflight_history_lock`` right after the canonical match_id exists:
    reuses the tested ``identity_lock_check`` so a resolver flip against
    recent history ABORTS the analysis instead of silently holding the
    write ~250s later.

Policy: identity is FAIL-CLOSED where there is POSITIVE evidence of a
mismatch (both sides confidently canonicalize to DIFFERENT clubs), and
FAIL-OPEN everywhere evidence is absent (unknown spellings, dynamic
leagues, unreadable logs). The gates never guess: an unresolvable name
can never trigger a refusal.

Canonicalization for comparison is STRICT by design:
1. league-scoped ``resolve_team_alias`` (ambiguity-guarded);
2. exact accent/punctuation-insensitive equality against every registered
   canonical club name (no fuzzy boundary pass -- a fuzzy hit must never
   be allowed to REFUSE an analysis).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .team_alias import _abbr_key, load_teams, resolve_team_alias

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # G-A/G-B: refuse when query side and provider side are provably two
    # different clubs.
    "refuse_divergence": True,
    # G-C: abort the analysis when the canonical pair flips against recent
    # snapshot history (same logic the end-of-run write hold uses).
    "refuse_history_lock": True,
}


def firewall_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """The ``identity_firewall`` config block with safe defaults."""
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("identity_firewall") or {})
    return out


# ---------------------------------------------------------------------------
# Canonical side resolution (strict)
# ---------------------------------------------------------------------------

_CANONICAL_INDEX_CACHE: dict[str, str] | None = None


def _canonical_index() -> dict[str, str]:
    """abbr_key(canonical club name) -> canonical name, ALL leagues.

    Built once per process from teams.json. Used ONLY for exact-equality
    lookups -- never substring/fuzzy matching -- so an unknown spelling
    stays unknown instead of hijacking a refusal.
    """
    global _CANONICAL_INDEX_CACHE
    if _CANONICAL_INDEX_CACHE is not None:
        return _CANONICAL_INDEX_CACHE
    idx: dict[str, str] = {}
    try:
        teams = load_teams()
    except Exception:  # noqa: BLE001 -- missing/corrupt table = fail-open
        teams = {}
    for aliases in teams.values():
        for canonical in set(aliases.values()):
            k = _abbr_key(canonical)
            if k and k not in idx:
                idx[k] = canonical
    _CANONICAL_INDEX_CACHE = idx
    return idx


def canonical_side(name: str | None, league_key: str | None = None) -> str | None:
    """Confident canonical club name for ``name``, else None.

    Two strict passes only: the ambiguity-guarded league-scoped alias
    resolver, then exact normalized equality with a registered canonical
    name (any league -- UCL/UEL fixtures legitimately host clubs from every
    domestic league). Returns None when the name cannot be pinned: None
    means "undecidable", never a guess.
    """
    raw = str(name or "").strip()
    if not raw:
        return None
    if league_key:
        try:
            resolved = resolve_team_alias(raw, league_key)
        except Exception:  # noqa: BLE001
            resolved = None
        if resolved:
            return resolved
    return _canonical_index().get(_abbr_key(raw))


def _norm_club(name: str | None, league_key: str | None = None) -> str | None:
    canon = canonical_side(name, league_key)
    return _abbr_key(canon) if canon else None


# ---------------------------------------------------------------------------
# G-A / G-B: query pair vs provider-rendered pair
# ---------------------------------------------------------------------------

def check_pair_identity(
    home_query: str,
    away_query: str,
    *,
    league_key: str | None = None,
    detected_home: str | None = None,
    detected_away: str | None = None,
    resolved_home: str | None = None,
    resolved_away: str | None = None,
) -> dict[str, Any]:
    """Verify the ANALYSED pair is the REQUESTED pair, per side.

    Compares the user's query pair against each independently rendered
    provider pair (the detected fixture and/or the resolved team entities).
    A side counts as DIVERGENT only when BOTH spellings confidently
    canonicalize to different clubs -- that is positive evidence the
    pipeline is about to analyse team C instead of team B. A confirmed
    SIDE SWAP (query A-B rendered as B-A) is a warning, not a refusal:
    some sources render neutral/pendent fixtures reversed, and refusing
    them would block real matches (the internal pipeline stays consistent
    because every fetch keys off the same resolved ids).

    Returns ``{"status": "ok"|"refuse"|"warn", "reasons": [...], ...}``.
    Never raises; malformed input degrades to ``ok`` (fail-open).
    """
    out: dict[str, Any] = {
        "status": "ok",
        "reasons": [],
        "query": {"home": home_query, "away": away_query},
        "checks": [],
    }
    try:
        qh = _norm_club(home_query, league_key)
        qa = _norm_club(away_query, league_key)
        pairs: list[tuple[str, str | None, str | None]] = []
        if detected_home or detected_away:
            pairs.append(("detected_fixture", detected_home, detected_away))
        if resolved_home or resolved_away:
            pairs.append(("resolved_teams", resolved_home, resolved_away))
        if not pairs:
            return out
        for label, ph, pa in pairs:
            ph_n, pa_n = _norm_club(ph, league_key), _norm_club(pa, league_key)
            check: dict[str, Any] = {"stage": label}
            if ph_n:
                check["home"] = ph_n
            if pa_n:
                check["away"] = pa_n
            out["checks"].append(check)

            # Both values are canonical abbr-keys, so plain inequality IS a
            # different-club verdict (never a spelling artifact).
            def _divergent(q: str | None, p: str | None) -> bool:
                return bool(q and p and q != p)

            home_div = _divergent(qh, ph_n)
            away_div = _divergent(qa, pa_n)
            if home_div and away_div and qh == pa_n and qa == ph_n:
                out["status"] = "warn"
                out["reasons"].append(
                    f"sisi tertukar pada {label}: query {home_query} vs {away_query} "
                    f"dirender {ph} vs {pa}"
                )
                continue
            if home_div:
                out["status"] = "refuse"
                out["reasons"].append(
                    f"HOME berbeda klub pada {label}: query '{home_query}' "
                    f"(kanonik {qh}) tetapi sumber memberi '{ph}' ({ph_n})"
                )
            if away_div:
                out["status"] = "refuse"
                out["reasons"].append(
                    f"AWAY berbeda klub pada {label}: query '{away_query}' "
                    f"(kanonik {qa}) tetapi sumber memberi '{pa}' ({pa_n})"
                )
        return out
    except Exception as exc:  # noqa: BLE001 -- gate must never break analyse
        logger.warning("identity_gate.check_pair_identity failed (fail-open): %s", exc)
        out.update(status="ok", reasons=[f"gate error: {type(exc).__name__}"])
        return out


def refusal_payload(
    stage: str,
    reasons: list[str],
    *,
    display: str | None,
    home_query: str,
    away_query: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Error payload shaped like every other analyse error (format.py renders
    ``payload['error']`` verbatim) plus a machine-readable guard block."""
    msg = (
        f"IDENTITY GUARD [{stage}]: analisa DIHENTIKAN — pasangan tim yang "
        "akan dianalisa berbeda dari yang diminta. Alasan: "
        + "; ".join(reasons[:3])
    )
    payload: dict[str, Any] = {
        "error": msg,
        "league": display,
        "home_query": home_query,
        "away_query": away_query,
        "identity_guard": {
            "status": "refused",
            "stage": stage,
            "reasons": reasons,
        },
    }
    if detail:
        payload["identity_guard"]["detail"] = detail
    return payload


# ---------------------------------------------------------------------------
# G-C: pre-flight history lock
# ---------------------------------------------------------------------------

DEFAULT_PREDICTIONS_FILE = "cache/football/predictions.jsonl"


def preflight_history_lock(
    log_path: str | Path,
    *,
    match_id: str,
    home: str,
    away: str,
    entities: dict[str, Any] | None = None,
    now_ts: str | None = None,
) -> dict[str, Any] | None:
    """Abort-level identity lock scan BEFORE the analysis runs.

    Same detection logic (and therefore the same accepted false-positive
    policy) as the end-of-run ``identity_lock_check`` write hold -- reused,
    not reimplemented. Fail-open: any failure returns None so the analysis
    proceeds exactly as before this module existed.
    """
    try:
        from .prediction_log import identity_lock_check

        lock = identity_lock_check(
            log_path,
            match_id=match_id,
            home=home,
            away=away,
            entities=entities,
            now_ts=now_ts,
        )
        if lock and lock.get("locked"):
            logger.warning(
                "identity_gate preflight: %s — %s (conflict=%s)",
                lock.get("kind"), lock.get("reason"), lock.get("conflict_match_id"),
            )
        return lock
    except Exception as exc:  # noqa: BLE001
        logger.warning("identity_gate preflight failed (fail-open): %s", exc)
        return None
