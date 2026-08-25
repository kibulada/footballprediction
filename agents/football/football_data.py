"""football-data.org client.

Free tier: 10 req/minute (throttled at 6s/req). Used as primary source
for top 11 EU competitions. Provides:
  - /v4/competitions/{code}/teams   -> team list per league
  - /v4/teams/{id}                  -> team detail
  - /v4/teams/{id}/matches?limit=10 -> last N matches per team
  - /v4/matches?team1=X&team2=Y     -> H2H
  - /v4/competitions/{code}/standings

Auth header: X-Auth-Token
"""
from __future__ import annotations

import asyncio
import json
import logging
import unicodedata
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
USER_AGENT = "hermes-football/1.0 (local-advisory)"


class FootballDataError(Exception):
    pass


def _payload_has_content(payload: dict[str, Any]) -> bool:
    """True when a football-data payload carries a non-empty data list.

    Off-season responses return {"matches": [], "resultCount": 0, ...} --
    truthy as a dict but useless to callers; we must not pace those.
    """
    for value in payload.values():
        if isinstance(value, list) and value:
            return True
    return False


class FootballDataClient:
    def __init__(self, api_key: str = "", throttle_seconds: float = 6.0) -> None:
        self._key = api_key or ""
        self._throttle = throttle_seconds
        self.rate_limit_warning: bool = False
        # Per-competition team-list cache: the analyse flow resolves BOTH
        # sides through the same competition teams endpoint; without a cache
        # the second side pays a full throttled call (6s sleep) for identical
        # data, which is pure dead time inside the runner's 85s deadline.
        self._teams_cache: dict[str, tuple[float, list[dict[str, Any]] | None]] = {}
        self._teams_cache_ttl = 3600.0

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{BASE_URL}{path}"
        headers = {"X-Auth-Token": self._key, "User-Agent": USER_AGENT}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params or {}, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("football-data network error: %s", exc)
            return None

        if resp.status_code == 429:
            self.rate_limit_warning = True
            return None
        if resp.status_code == 401:
            raise FootballDataError("401 Unauthorized: token football-data invalid")
        if resp.status_code >= 400:
            logger.warning("football-data http %s", resp.status_code)
            return None

        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return None

        # Pace only responses that actually carry data. An empty payload
        # (e.g. "matches": [] during the off-season, before a new season has
        # any FINISHED games) pays the 6s sleep for nothing -- and the form
        # path calls these endpoints per team, burning ~24s of the runner's
        # 85s deadline on empty results. A short 2s sleep is kept as a safety
        # buffer so several consecutive empty-result calls cannot trip the
        # 10 req/min quota on a faster flow.
        if _payload_has_content(payload):
            await asyncio.sleep(self._throttle)
        else:
            await asyncio.sleep(2.0)
        return payload

    async def fetch_teams(self, competition_code: str) -> list[dict[str, Any]] | None:
        import time as _time

        cached = self._teams_cache.get(competition_code)
        if cached and (_time.monotonic() - cached[0]) < self._teams_cache_ttl:
            return cached[1]
        data = await self._get(f"/competitions/{competition_code}/teams")
        if not data:
            return None
        teams = data.get("teams")
        if isinstance(teams, list):
            self._teams_cache[competition_code] = (_time.monotonic(), teams)
        return teams if isinstance(teams, list) else None

    async def fetch_team(self, team_id: int) -> dict[str, Any] | None:
        return await self._get(f"/teams/{team_id}")

    async def fetch_last_matches(self, team_id: int, limit: int = 10) -> list[dict[str, Any]] | None:
        data = await self._get(f"/teams/{team_id}/matches", {"limit": limit, "status": "FINISHED"})
        if not data:
            return None
        matches = data.get("matches")
        return matches if isinstance(matches, list) else None

    async def fetch_upcoming_matches(self, team_id: int, days_ahead: int = 30) -> list[dict[str, Any]] | None:
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).date()
        end = today + timedelta(days=days_ahead)
        data = await self._get(
            f"/teams/{team_id}/matches",
            {"limit": 20, "status": "SCHEDULED", "dateFrom": today.isoformat(),
             "dateTo": end.isoformat()},
        )
        if not data:
            return None
        matches = data.get("matches")
        return matches if isinstance(matches, list) else None

    async def fetch_h2h(self, team1_id: int, team2_id: int, limit: int = 5) -> list[dict[str, Any]] | None:
        data = await self._get(
            "/matches",
            {"team1": team1_id, "team2": team2_id, "limit": limit, "status": "FINISHED"},
        )
        if not data:
            return None
        matches = data.get("matches")
        return matches if isinstance(matches, list) else None

    async def fetch_finished_matches_by_date(
        self, date_from: str, date_to: str,
    ) -> list[dict[str, Any]] | None:
        """Finished results across all competitions for a date range.

        Returns [{home, away, home_goals, away_goals}, ...] -- used by the
        `settle auto` command to close prediction-log snapshots.
        """
        data = await self._get(
            "/matches",
            {"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
        )
        if not data:
            return None
        out: list[dict[str, Any]] = []
        for m in data.get("matches") or []:
            ht = (m.get("homeTeam") or {}).get("name")
            at = (m.get("awayTeam") or {}).get("name")
            sc = (m.get("score") or {}).get("fullTime") or {}
            hg, ag = sc.get("home"), sc.get("away")
            if ht and at and hg is not None and ag is not None:
                out.append(
                    {"home": ht, "away": at,
                     "home_goals": int(hg), "away_goals": int(ag)}
                )
        return out

    async def fetch_scheduled_matches_by_date(
        self, date_from: str, date_to: str,
    ) -> list[dict[str, Any]] | None:
        """Upcoming fixtures across ALL competitions for a date range.

        One /v4/matches call (the same endpoint `settle auto` uses with
        FINISHED) returns every scheduled fixture football-data covers --
        used by the league auto-detect to find which registered league a
        free-typed match belongs to. Returns normalized rows with
        home/away names, the competition code and the UTC kickoff.
        """
        data = await self._get(
            "/matches",
            {"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED,TIMED"},
        )
        if not data:
            return None
        out: list[dict[str, Any]] = []
        for m in data.get("matches") or []:
            ht = (m.get("homeTeam") or {}).get("name")
            at = (m.get("awayTeam") or {}).get("name")
            code = (m.get("competition") or {}).get("code")
            if ht and at and code:
                out.append(
                    {
                        "home": ht,
                        "away": at,
                        "competition": code,
                        "kickoff": m.get("utcDate"),
                    }
                )
        return out

    async def fetch_matches_by_competition(
        self, competition_code: str, date_from: str, date_to: str
    ) -> list[dict[str, Any]] | None:
        data = await self._get(
            f"/competitions/{competition_code}/matches",
            {"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED,TIMED"},
        )
        if not data:
            return None
        matches = data.get("matches")
        return matches if isinstance(matches, list) else None

    # Club-type prefixes (and other noise tokens) must never be used for
    # substring matching: "FK" matches "Qarabağ Ağdam FK", "NK" matches
    # "frankfurt" (the 'nk' inside), "FC" matches almost everything. Only
    # tokens of length >= 3 that are NOT such prefixes qualify.
    _CLUB_PREFIX_TOKENS = {
        "fk", "fc", "nk", "cd", "sc", "pfc", "ifk", "ss", "rc", "ca",
        "ec", "cr", "se", "ac", "cf", "us", "sd", "de", "sv", "sk",
        "ud", "as", "afc", "bsc", "tsg", "sb", "kc",
    }

    @staticmethod
    def _norm_team_str(s: str) -> str:
        """Lowercase + strip accents/punctuation -> comparable token string.

        football-data names carry accents ("Atlético Madrid") while user
        queries usually do not ("Atletico Madrid"); bare substring compare
        then fails (or worse, matches the wrong club). Normalizing both sides
        keeps exact/whole-word matching accent-agnostic (B6, 2026-08-17)."""
        s = unicodedata.normalize("NFD", (s or "").lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        import re as _re

        return " ".join(_re.sub(r"[^a-z0-9 ]", " ", s).split())

    async def search_team_in_competition(
        self, name: str, competition_code: str
    ) -> dict[str, Any] | None:
        teams = await self.fetch_teams(competition_code)
        if not teams:
            return None
        q = self._norm_team_str(name)
        if not q:
            return None
        q_tokens = [
            p for p in q.split()
            if len(p) >= 3 and p not in self._CLUB_PREFIX_TOKENS
        ]

        # 1) EXACT equality on the full normalized name / short name / code.
        #    (The old first pass was a bare substring of the full query --
        #    "atletico madrid" could match "Atlético Madrid B" first and
        #    resolve the WRONG team id, feeding another club's form/h2h into
        #    the model. B6, verified 2026-08-17.)
        for team in teams:
            tname = self._norm_team_str(team.get("name") or "")
            tshort = self._norm_team_str(team.get("shortName") or "")
            tcode = (team.get("tla") or "").lower()
            if q == tname or q == tshort or q == tcode:
                return team

        if not q_tokens:
            return None

        # 2) every non-trivial query token as a WHOLE WORD in the team name;
        #    when several names qualify ("Atlético Madrid" AND "Atlético
        #    Madrid B") the SHORTEST name wins -- the reserve side only
        #    survives if it is the only candidate.
        best: tuple[int, dict[str, Any]] | None = None
        for team in teams:
            tname = self._norm_team_str(team.get("name") or "")
            words = set(tname.split())
            if q_tokens and all(tok in words for tok in q_tokens):
                if best is None or len(tname) < best[0]:
                    best = (len(tname), team)
        if best is not None:
            return best[1]

        # 3) LAST resort: every token as a substring of the team name.
        best = None
        for team in teams:
            tname = self._norm_team_str(team.get("name") or "")
            if all(tok in tname for tok in q_tokens):
                if best is None or len(tname) < best[0]:
                    best = (len(tname), team)
        if best is not None:
            return best[1]
        return None
