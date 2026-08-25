"""TheSportsDB.com fallback client.

Free tier: unlimited basic. Used as last-resort fallback when neither
football-data.org nor API-Football can resolve a query. Provides:
  - /searchteams.php?t=NAME       -> team search
  - /eventslast.php?id=TEAM_ID    -> last 5 matches
  - /eventsnext.php?id=TEAM_ID    -> next 5 matches
  - /lookuph2h.php?id1=X&id2=Y    -> H2H (paid feature, may be limited)

Key: "3" is the current working public key. The legacy free key "1"
now returns 400 ("Invalid Premium API key"), so "3" is the default.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.thesportsdb.com/api/v1/json"
USER_AGENT = "hermes-football/1.0 (local-advisory)"
DEFAULT_KEY = "3"


def _league_norm(s: str) -> str:
    """Normalize a league label for the F2 wrong-club guard."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


class TheSportsDbError(Exception):
    pass


class TheSportsDbClient:
    def __init__(self, api_key: str = DEFAULT_KEY, throttle_seconds: float = 1.1) -> None:
        self._key = api_key or DEFAULT_KEY
        self._throttle = throttle_seconds

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{BASE_URL}/{self._key}/{path}"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params or {}, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as exc:
            logger.warning("thesportsdb network error: %s", exc)
            return None

        if resp.status_code >= 400:
            logger.warning("thesportsdb http %s", resp.status_code)
            return None

        try:
            await asyncio.sleep(self._throttle)
            return resp.json()
        except json.JSONDecodeError:
            return None

    # Club-type prefixes stripped before matching so "PFC Levski Sofia" and
    # "FC Ararat-Armenia" resolve to thesportsdb's "Levski Sofia" /
    # "Ararat-Armenia" instead of failing.
    _CLUB_PREFIXES = ("pfc ", "afc ", "fc ", "fk ", "nk ", "sc ", "ac ",
                      "as ", "ss ", "sk ", "cd ", "ifk ", "us ", "ud ")

    @staticmethod
    def _normalize(name: str) -> str:
        s = (name or "").lower().strip()
        for pref in TheSportsDbClient._CLUB_PREFIXES:
            if s.startswith(pref):
                s = s[len(pref):]
                break
        return s.replace("-", " ").replace("_", " ")

    @staticmethod
    def _league_matches(team: dict[str, Any], league_hint: str | None) -> bool:
        """F2 wrong-club guard: is ``team``'s league consistent with the hint?

        An empty hint or empty league fields cannot disprove -> keep (same
        "cannot disprove" rule as the G5 league-window filter). Exact name
        matches in ``search_team`` bypass this guard entirely -- it only
        applies to the no-name-match ``teams[0]`` guess, so a short /
        ambiguous query ("Lens", "Real") can never resolve to an unrelated
        club in a different competition.
        """
        if not league_hint:
            return True
        hint = _league_norm(league_hint)
        if not hint:
            return True
        for field in ("strLeague", "strLeague2"):
            league = _league_norm(team.get(field) or "")
            if not league:
                continue
            if hint in league or league in hint:
                return True
            hint_tokens = {t for t in hint.split(" ") if len(t) >= 3}
            league_tokens = {t for t in league.split(" ") if len(t) >= 3}
            if hint_tokens and league_tokens and (hint_tokens & league_tokens):
                return True
        return False

    async def search_team(self, name: str, league_hint: str | None = None) -> dict[str, Any] | None:
        q = self._normalize(name)
        # Candidate queries: the raw name, then the prefix-stripped / dash-
        # normalized form ("PFC Levski Sofia" -> "Levski Sofia", "FC
        # Ararat-Armenia" -> "Ararat Armenia") so club-type prefixes never
        # cause a failed lookup.
        queries = [name]
        if q != (name or "").lower().strip():
            queries.append(q)
        first_result: dict[str, Any] | None = None
        for query in queries:
            data = await self._get("searchteams.php", {"t": query})
            if not data:
                continue
            teams = data.get("teams")
            if not teams or not isinstance(teams, list):
                continue
            if first_result is None:
                first_result = teams[0]
            for team in teams:
                tname = self._normalize(team.get("strTeam") or "")
                if q and (q == tname or q in tname or tname in q):
                    return team
        # F2 (2026-08-17): a teams[0] guess with NO name match can be the
        # WRONG club (short/ambiguous names land on teams[0] of an unrelated
        # club). When a league context is known, keep the guess only if its
        # league is consistent; otherwise prefer an honest miss (None) over a
        # wrong-team form/H2H. Name matches above always win.
        if first_result is not None and not self._league_matches(first_result, league_hint):
            logger.info(
                "thesportsdb teams[0] fallback '%s' rejected: league mismatch (hint=%r)",
                first_result.get("strTeam") or "?",
                league_hint,
            )
            return None
        return first_result

    async def fetch_last_matches(self, team_id: str, limit: int = 5) -> list[dict[str, Any]] | None:
        data = await self._get("eventslast.php", {"id": team_id})
        if not data:
            return None
        results = data.get("results")
        if not results or not isinstance(results, list):
            return None
        return results[:limit]

    async def fetch_next_matches(self, team_id: str, limit: int = 5) -> list[dict[str, Any]] | None:
        data = await self._get("eventsnext.php", {"id": team_id})
        if not data:
            return None
        events = data.get("events")
        if not events or not isinstance(events, list):
            return None
        return events[:limit]
