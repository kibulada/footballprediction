"""Flashscore GraphQL gateway client — pure HTTP, no browser (2026-08).

The Flashscore SPA loads match context (missing players, lineups, coaches)
from an internal GraphQL gateway (``2.ds.lsapp.eu/pq_graphql``). Verified
from this network (2026-08): the endpoints answer plain GET requests keyed
by ``?_hash=<query id>`` with only Origin/Referer headers — no session,
cookies or X-Fsign signature required (unlike the ``global.flashscore.ninja``
feed API). Odds hashes return empty data for geo-ID, but the lineup and
missing-player hashes return FULL data for every match probed.

Query ids come from the app bundle ``detail.de4c547.js``:

  dmpe2  DETAIL_MISSING_PLAYERS_ENRICHED_QUERY_2
         -> eventParticipants[].lineup.{missingPlayers, unsureMissingPlayers}
  dlie2  DETAIL_LINEUPS_ENRICHED_QUERY_2
         -> eventParticipants[].lineup.{players, formation, coaches}

Every fetch degrades to None (network error / bad payload / no event id);
callers treat the result as optional pre-match context, never a model input.
The client is pure extraction: team-side matching is done by the caller
against the resolved home/away names (flashscore's participant ORDER is not
guaranteed home-first).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://2.ds.lsapp.eu/pq_graphql"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

# Query ids (hash -> name) from the app bundle; only the ones we consume.
HASH_MISSING_PLAYERS = "dmpe2"
HASH_LINEUPS_ENRICHED = "dlie2"


def _header() -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.flashscore.com",
        "Referer": "https://www.flashscore.com/",
    }


# ---- tolerant team-name matching (mirrors oddspapi._same_team) ------------

import re  # noqa: E402
import unicodedata  # noqa: E402

_STROKE_LETTERS = str.maketrans(
    {
        "\u00f8": "o", "\u0142": "l", "\u0111": "d", "\u0127": "h",
        "\u0131": "i", "\u014b": "n", "\u00df": "ss",
    }
)
_TEAM_PREFIXES = {
    "fk", "fc", "nk", "cd", "sc", "pfc", "ifk", "ss", "rc", "ca",
    "ec", "cr", "se", "ac", "cf", "us", "sd", "de", "sv", "sk",
}


def _norm_team(name: str) -> str:
    s = (name or "").lower().translate(_STROKE_LETTERS)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def same_team(a: str, b: str) -> bool:
    """Tolerant equality: prefixes stripped, then containment for long names."""
    na, nb = _norm_team(a), _norm_team(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    for pref in _TEAM_PREFIXES:
        if na.startswith(pref + " ") and na[len(pref) + 1:] == nb:
            return True
        if nb.startswith(pref + " ") and nb[len(pref) + 1:] == na:
            return True
    if len(na) >= 6 and (na in nb or nb in na):
        return True
    return False


# ---- pure parsers ---------------------------------------------------------


def _parse_missing_list(items: list[Any] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        player = item.get("player") or {}
        name = player.get("name") or player.get("listName")
        reason = item.get("reason")
        if not name:
            continue
        out.append({"name": str(name), "reason": str(reason) if reason else None})
    return out


def _parse_players(items: list[Any] | None) -> list[dict[str, str]]:
    """Starting-XI players: {name, shirt} or {} when not announced yet."""
    out: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("listName")
        if not name:
            continue
        row: dict[str, str] = {"name": str(name)}
        shirt = item.get("shirtNumber") or item.get("shirt")
        if shirt is not None:
            row["shirt"] = str(shirt)
        out.append(row)
    return out


def _parse_coaches(lineup: dict[str, Any] | None) -> list[str]:
    coaches = (lineup or {}).get("coaches") or {}
    out: list[str] = []
    for player in coaches.get("players") or []:
        name = player.get("name") or player.get("listName")
        if name:
            out.append(str(name))
    return out


def _side_from_participants(
    participants: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map side (HOME/AWAY) -> {name, missing, unsure, players, formation, coaches}.

    ``dlie2`` carries the participant names; ``dmpe2`` carries the missing
    player lists. The caller passes the dlie2 participants first so names are
    known, then merges missing/unsure by participant id.
    """
    sides: dict[str, dict[str, Any]] = {}
    for p in participants or []:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type") or {}
        side = str(ptype.get("side") or "").upper()
        if side not in ("HOME", "AWAY"):
            continue
        lineup = p.get("lineup") or {}
        entry = sides.setdefault(
            side,
            {
                "name": p.get("name"),
                "missing": [],
                "unsure": [],
                "players": [],
                "formation": None,
                "coaches": [],
            },
        )
        if p.get("name"):
            entry["name"] = p["name"]
        entry["missing"] = _parse_missing_list(lineup.get("missingPlayers"))
        entry["unsure"] = _parse_missing_list(lineup.get("unsureMissingPlayers"))
        entry["players"] = _parse_players(lineup.get("players"))
        entry["formation"] = lineup.get("formation")
        entry["coaches"] = _parse_coaches(lineup)
    return sides


def normalize_event_context(
    dlie2_data: dict[str, Any] | None,
    dmpe2_data: dict[str, Any] | None,
    home_name: str | None = None,
    away_name: str | None = None,
) -> dict[str, Any] | None:
    """Merge the two GraphQL payloads into one home/away context dict.

    Returns ``{home: {name, missing, unsure, players, formation, coaches},
    away: {...}}`` with sides resolved against the resolved home/away names
    (flashscore participant order is not guaranteed home-first). Returns None
    when nothing usable is present.
    """
    dlie2_parts = ((dlie2_data or {}).get("data") or {}).get("findEventById") or {}
    dmpe2_parts = ((dmpe2_data or {}).get("data") or {}).get("findEventById") or {}

    sides = _side_from_participants(dlie2_parts.get("eventParticipants") or [])
    if dmpe2_parts.get("eventParticipants"):
        merged = _side_from_participants(dmpe2_parts.get("eventParticipants") or [])
        for side, entry in merged.items():
            if side in sides:
                # dlie2 has names/players; dmpe2 has the fresher missing lists.
                sides[side]["missing"] = entry["missing"]
                sides[side]["unsure"] = entry["unsure"]
            else:
                sides[side] = entry

    if not sides:
        return None

    # Resolve which flashscore side is our home/away.
    def _pick(target: str | None) -> dict[str, Any] | None:
        if not target:
            return None
        for side in ("HOME", "AWAY"):
            entry = sides.get(side) or {}
            if same_team(entry.get("name") or "", target):
                return entry
        # unknown side: keep the flashscore order (HOME first) as a guess
        return None

    home = _pick(home_name) if home_name else None
    away = _pick(away_name) if away_name else None
    if home is None and away is None:
        # no names to match against: keep raw sides so the caller still gets
        # the data (order ambiguity is documented, never silently swapped)
        order = [sides.get("HOME"), sides.get("AWAY")]
        home, away = (order[0], order[1]) if len(order) >= 2 else (order[0], None)
    elif home is None and away is not None:
        home = sides.get("HOME") if sides.get("HOME") is not away else None
        if home is None:
            for side in ("HOME", "AWAY"):
                if sides.get(side) is not away:
                    home = sides.get(side)
                    break
    elif away is None and home is not None:
        for side in ("HOME", "AWAY"):
            if sides.get(side) is not home:
                away = sides.get(side)
                break

    out: dict[str, Any] = {}
    for side_key, entry in (("home", home), ("away", away)):
        if not entry:
            continue
        out[side_key] = {
            "name": entry.get("name"),
            "missing": entry.get("missing") or [],
            "unsure": entry.get("unsure") or [],
            "players": entry.get("players") or [],
            "formation": entry.get("formation"),
            "coaches": entry.get("coaches") or [],
        }
    if not out:
        return None
    out["source"] = "flashscore_graphql"
    return out


# ---- HTTP client ----------------------------------------------------------


class FlashscoreGraphqlClient:
    """Thin async GET client for the flashscore GraphQL gateway."""

    def __init__(
        self,
        throttle_seconds: float = 0.5,
        timeout: float = 12.0,
    ) -> None:
        self._throttle = throttle_seconds
        self._timeout = timeout

    async def _get(self, hash_id: str, params: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{BASE_URL}?{hash_id}={params['_hash']}" if False else BASE_URL
        q = {**params, "_hash": hash_id}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=q, headers=_header())
        except httpx.HTTPError as exc:
            logger.warning("flashscore graphql network error (%s): %s", hash_id, exc)
            return None
        if resp.status_code != 200:
            logger.warning("flashscore graphql http %s on %s", resp.status_code, hash_id)
            return None
        await asyncio.sleep(self._throttle)
        try:
            return resp.json()
        except ValueError:
            return None

    async def fetch_event_context(
        self,
        event_id: str,
        home_name: str | None = None,
        away_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Missing players + lineups + coaches for one match, side-resolved."""
        if not event_id:
            return None
        dlie2 = await self._get(
            HASH_LINEUPS_ENRICHED, {"eventId": event_id, "projectId": 2}
        )
        dmpe2 = await self._get(
            HASH_MISSING_PLAYERS, {"eventId": event_id, "projectId": 2}
        )
        return normalize_event_context(dlie2, dmpe2, home_name, away_name)
