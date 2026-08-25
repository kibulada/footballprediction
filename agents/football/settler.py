"""Settle prediction-log snapshots with real results (closing the loop).

- ``settle_manual``: match ONE unsettled snapshot by tolerant team names
  (+ optional league/date) and append the result.
- ``settle_auto``: settle every unsettled snapshot whose kickoff falls on a
  given date, using finished results provided by the caller (pure function;
  the network fetch lives in the CLI layer).

Both return a plain report dict consumed by the CLI / Discord formatter.
Honesty: a snapshot is never edited; a settlement line is appended only when
a matching snapshot exists and (for manual) the result parses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .analyse import _teams_match
from .prediction_log import list_unsettled, settle

Matcher = Callable[[str, str], bool]


def _parse_result(result: str) -> tuple[int, int] | None:
    parts = result.replace(":", "-").split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _matches(s: dict[str, Any], matcher: Matcher, home: str, away: str) -> bool:
    return matcher(home, s.get("home") or "") and matcher(away, s.get("away") or "")


def _canonical_result_check(s: dict[str, Any], r: dict[str, Any]) -> bool | None:
    """Canonical-id verification of one finished result vs a snapshot (G2).

    When the snapshot carries G2 ``entities`` (canonical_id per side) and the
    result is resolvable, returns True when the result's canonical team ids
    match the snapshot's (either order), False when they CONFLICT (names
    look alike, clubs differ -- never settle the wrong club), and None when
    verification is impossible (no entities / unresolvable names) so the
    caller falls back to the existing name matching.
    """
    ent = s.get("entities") or {}
    h = ent.get("home") or {}
    a = ent.get("away") or {}
    home_cid = h.get("canonical_id")
    away_cid = a.get("canonical_id")
    if not home_cid or not away_cid:
        return None
    try:
        from .entity_registry import canonical_team_id
        from .league_resolver import competition_league_key

        lk = None
        comp = r.get("competition")
        if comp:
            try:
                lk = competition_league_key(str(comp))
            except Exception:  # noqa: BLE001 -- mapping must never break settle
                lk = None
        lk = lk or ent.get("league_key")
        r_home = canonical_team_id(lk, r.get("home") or "")
        r_away = canonical_team_id(lk, r.get("away") or "")
        if not r_home or not r_away:
            return None
        ordered = r_home == home_cid and r_away == away_cid
        reversed_ = r_home == away_cid and r_away == home_cid
        if ordered != reversed_:
            return True
        if ordered:
            return True  # defensive: ordered == reversed_ == True is a degenerate duplicate
        return False
    except Exception:  # noqa: BLE001 -- verification failure degrades to name matching
        return None


def _snap_meta(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "league": s.get("league"),
        "home": s.get("home"),
        "away": s.get("away"),
        "kickoff": s.get("kickoff"),
        "match_id": s.get("match_id"),
    }


def settle_manual(
    path: str | Path,
    *,
    home: str,
    away: str,
    result: str,
    matcher: Matcher = _teams_match,
    league: str | None = None,
    date: str | None = None,
    closing_odds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Settle one snapshot by tolerant team match; report the outcome."""
    score = _parse_result(result)
    if score is None:
        return {"status": "bad_result", "result": result}

    snaps = list_unsettled(path)
    cands = [
        s for s in snaps
        if (not league or (s.get("league") or "") == league)
        and (not date or (s.get("kickoff") or "")[:10] == date)
        and _matches(s, matcher, home, away)
    ]
    if not cands:
        return {
            "status": "not_found",
            "home": home,
            "away": away,
            "unsettled_count": len(snaps),
            "recent": [_snap_meta(s) for s in snaps[:8]],
        }
    if len(cands) > 1:
        return {
            "status": "ambiguous",
            "home": home,
            "away": away,
            "candidates": [_snap_meta(s) for s in cands],
        }

    s = cands[0]
    ok = settle(
        path, match_id=s["match_id"],
        home_goals=score[0], away_goals=score[1],
        closing_odds=closing_odds,
    )
    if not ok:
        return {"status": "error", "message": "settle write failed"}
    return {
        "status": "settled",
        "match_id": s["match_id"],
        "home": s.get("home"),
        "away": s.get("away"),
        "kickoff": s.get("kickoff"),
        "league": s.get("league"),
        "result": f"{score[0]}-{score[1]}",
        "home_goals": score[0],
        "away_goals": score[1],
        "closing_odds": closing_odds,
    }


ClosingFetcher = Callable[[dict[str, Any]], dict[str, float] | None]


def settle_auto(
    path: str | Path,
    *,
    date: str,
    results: list[dict[str, Any]],
    matcher: Matcher = _teams_match,
    closing_fetcher: ClosingFetcher | None = None,
) -> dict[str, Any]:
    """Settle snapshots kicked off on ``date`` using finished ``results``.

    ``results`` items: {home, away, home_goals, away_goals}. Matches are
    tolerant (prefixes/honorifics). Snapshots without a finished result are
    reported, not invented.

    ``closing_fetcher`` (optional, PHASE 0.1): a callable that takes one
    finished result dict and returns closing 1X2 decimal odds
    ``{home, draw, away}`` (or None). When provided, it is invoked per
    settled match BEFORE the settlement row is appended, and the returned
    prices are stored as the settlement's ``closing_odds`` -- the root cause
    of empty CLV today was that auto-settle never attached a closing price.
    The function stays PURE: the network fetch lives in the CLI layer, which
    passes a fetcher (e.g. a NowGoal ``l``-leg lookup).
    """
    snaps = list_unsettled(path)
    settled: list[dict[str, Any]] = []
    not_found: list[dict[str, Any]] = []
    for s in snaps:
        if (s.get("kickoff") or "")[:10] != date:
            continue
        # G2: among name-matching results, prefer one whose canonical ids
        # VERIFY against the snapshot's entities; skip any that CONFLICT
        # (never settle the wrong club); fall back to plain name matching
        # when the snapshot carries no entities (backward compatible).
        name_hits = [
            r for r in results
            if _matches(s, matcher, r.get("home", ""), r.get("away", ""))
        ]
        # verified (True) first; then name-matching results that do NOT
        # conflict (None = cannot verify -> backward-compatible name match);
        # a canonical conflict (False) is NEVER settled.
        _checks = [(_canonical_result_check(s, r), r) for r in name_hits]
        verified = [r for ok, r in _checks if ok is True]
        allowed = [r for ok, r in _checks if ok is not False]
        found = (verified or allowed or [None])[0]
        if found is None:
            not_found.append(_snap_meta(s))
            continue
        hg = int(found.get("home_goals") or 0)
        ag = int(found.get("away_goals") or 0)
        closing = None
        if closing_fetcher is not None:
            try:
                closing = closing_fetcher(found)
            except Exception:  # noqa: BLE001 -- CLV fetch must never break settle
                closing = None
        settle(
            path, match_id=s["match_id"], home_goals=hg, away_goals=ag,
            closing_odds=closing,
        )
        settled.append(
            {
                "league": s.get("league"),
                "home": s.get("home"),
                "away": s.get("away"),
                "result": f"{hg}-{ag}",
                "home_goals": hg,
                "away_goals": ag,
                "closing_odds": closing,
            }
        )
    return {
        "status": "auto",
        "date": date,
        "settled": settled,
        "not_found": not_found,
        "results_fetched": len(results),
        "unsettled_total": len(snaps),
        "closing_attached": sum(1 for x in settled if x.get("closing_odds")),
    }
