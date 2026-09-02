"""Two-legged tie context (K3, post-mortem 2026-08-28).

Every UEL/UECL card of 2026-08-27 was a play-off SECOND LEG, yet no snapshot
carried any tie context (``context_data`` was null on 100% of rows). The
90-minute result of a second leg depends on the aggregate state:

* tie already decided (first-leg margin >= 2): the leading side rotates /
  sits back -- on 27 Aug the market favourite failed to win 90' in 4 of 8
  such ties (Kauno 1-0 Besiktas after 0-3, Thun 2-2 Lech after 0-7, Austria
  Wien 0-0 Braga after 0-2, Larne 0-2 Lincoln after 2-0);
* tie balanced (margin <= 1): cagey, low-scoring 90' (Jablonec 1-0, Inter
  Escaldes 0-0, Rapid 1-1, Maccabi 1-1, Hapoel 0-1).

This module derives that state from the H2H list the analyser already
fetches (livescore ``meetings`` / nowgoal ``match_list``): a finished
meeting between the SAME two clubs with REVERSED venue within
``MAX_DAYS_BETWEEN_LEGS`` of kickoff is the first leg. Pure functions, no
I/O; used as a SOFT penalty in the signal engine and as a SUGGESTION rule.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

MAX_DAYS_BETWEEN_LEGS = 10
DECIDED_MARGIN = 2


def _squash(name: str | None) -> str:
    """Lowercase alnum only, diacritics stripped ("Beşiktaş" -> "besiktas")."""
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _same_team(a: str | None, b: str | None) -> bool:
    """Token-level identity (2026-09-02: shared ``team_identity``); the old
    4-char substring rule turned any same-named reserve/youth meeting into a
    phantom first leg."""
    from .team_identity import names_match

    return names_match(a, b)


def _same_competition(a: str | None, b: str | None) -> bool | None:
    """True/False when both known (tolerant containment on squashed names),
    None when either is unknown."""
    x, y = _squash(a), _squash(b)
    if not x or not y:
        return None
    return x == y or x in y or y in x


def _to_dt(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        s = str(raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if len(s) == 10:
            s += "T00:00:00+00:00"
        dt = datetime.fromisoformat(s.replace(" ", "T"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _rows(h2h: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(h2h, dict):
        return []
    for key in ("meetings", "match_list"):
        rows = h2h.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def tie_state_from_h2h(
    h2h: dict[str, Any] | None,
    *,
    home: str,
    away: str,
    kickoff: str | None,
    max_days: int = MAX_DAYS_BETWEEN_LEGS,
    competition: str | None = None,
) -> dict[str, Any] | None:
    """Detect a second leg from the H2H rows; None when no first leg is found.

    ``competition`` (2026-09-02): when the analysed match's competition and
    the meeting's competition are both known and differ, the meeting is not
    a first leg (a rescheduled league fixture next to a cup tie).

    Returns ``{leg: 2, first_leg, first_leg_home_goals, first_leg_away_goals,
    agg_margin_home, state, leader, days_between, competition, source}``
    where goals are expressed for the CURRENT home / away sides and
    ``agg_margin_home`` = current-home goals minus current-away goals in the
    first leg (positive = current home side leads the tie).
    """
    ko = _to_dt(kickoff)
    if ko is None:
        return None
    best: dict[str, Any] | None = None
    for m in _rows(h2h):
        status = str(m.get("status") or "finished").lower()
        if status not in ("finished", "ft", "full-time", "fulltime", ""):
            continue
        # First leg = reversed venue: today's away side hosted.
        if not (_same_team(m.get("home"), away) and _same_team(m.get("away"), home)):
            continue
        if _same_competition(competition, m.get("competition") or m.get("league")) is False:
            continue
        dt = _to_dt(m.get("kickoff") if "kickoff" in m else m.get("date"))
        if dt is None:
            continue
        days = (ko - dt).total_seconds() / 86400.0
        if days <= 0 or days > max_days:
            continue
        hs = _to_int(m.get("home_score"))
        as_ = _to_int(m.get("away_score"))
        if hs is None or as_ is None:
            score = str(m.get("score") or m.get("result") or "")
            mm = re.match(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$", score)
            if not mm:
                continue
            hs, as_ = int(mm.group(1)), int(mm.group(2))
        cand = {
            "leg": 2,
            "first_leg": f"{m.get('home')} {hs}-{as_} {m.get('away')}",
            # goals for TODAY's sides: today's home side was away in leg 1.
            "first_leg_home_goals": as_,
            "first_leg_away_goals": hs,
            "agg_margin_home": as_ - hs,
            "days_between": round(days, 1),
            "competition": m.get("competition") or m.get("league"),
            "source": "h2h",
        }
        if best is None or cand["days_between"] < best["days_between"]:
            best = cand
    if best is None:
        return None
    margin = best["agg_margin_home"]
    if abs(margin) >= DECIDED_MARGIN:
        best["state"] = "decided"
        best["leader"] = "home" if margin > 0 else "away"
    else:
        best["state"] = "balanced"
        best["leader"] = "home" if margin > 0 else ("away" if margin < 0 else None)
    return best


def tie_state_note(ts: dict[str, Any] | None) -> str | None:
    """One-line Indonesian note for the card's internal notes."""
    if not ts:
        return None
    margin = int(ts.get("agg_margin_home") or 0)
    if ts.get("state") == "decided":
        who = "tuan rumah" if ts.get("leader") == "home" else "tim tamu"
        return (
            f"leg-2, agregat sudah selesai ({ts.get('first_leg')}; {who} unggul "
            f"{abs(margin)} gol) — pemimpin cenderung rotasi/bertahan, favorit 90' tidak andal"
        )
    return (
        f"leg-2, agregat seimbang ({ts.get('first_leg')}) — 90 menit cenderung tertutup, "
        "Over/BTTS Yes lebih berisiko"
    )
