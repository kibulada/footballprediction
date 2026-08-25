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
        "closing_odds": closing_odds,
    }


def settle_auto(
    path: str | Path,
    *,
    date: str,
    results: list[dict[str, Any]],
    matcher: Matcher = _teams_match,
) -> dict[str, Any]:
    """Settle snapshots kicked off on ``date`` using finished ``results``.

    ``results`` items: {home, away, home_goals, away_goals}. Matches are
    tolerant (prefixes/honorifics). Snapshots without a finished result are
    reported, not invented.
    """
    snaps = list_unsettled(path)
    settled: list[dict[str, Any]] = []
    not_found: list[dict[str, Any]] = []
    for s in snaps:
        if (s.get("kickoff") or "")[:10] != date:
            continue
        found = next(
            (r for r in results if _matches(s, matcher, r.get("home", ""), r.get("away", ""))),
            None,
        )
        if found is None:
            not_found.append(_snap_meta(s))
            continue
        hg = int(found.get("home_goals") or 0)
        ag = int(found.get("away_goals") or 0)
        settle(path, match_id=s["match_id"], home_goals=hg, away_goals=ag)
        settled.append(
            {
                "league": s.get("league"),
                "home": s.get("home"),
                "away": s.get("away"),
                "result": f"{hg}-{ag}",
            }
        )
    return {
        "status": "auto",
        "date": date,
        "settled": settled,
        "not_found": not_found,
        "results_fetched": len(results),
        "unsettled_total": len(snaps),
    }
