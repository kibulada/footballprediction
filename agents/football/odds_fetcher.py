"""The Odds API client.

Free tier: 500 requests/month. We read x-requests-remaining header and
surface it in the output. Calls coerce to dicts; failures return None and
the caller skips the match.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
USER_AGENT = "hermes-football/1.0 (local-advisory)"


class OddsApiError(Exception):
    pass


class OddsFetcher:
    def __init__(self, api_key: str, throttle_seconds: float = 1.1) -> None:
        if not api_key:
            raise OddsApiError("THE_ODDS_API_KEY kosong")
        self._key = api_key
        self._throttle = throttle_seconds
        self.last_remaining: int | None = None
        self.quota_blocked: bool = False

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        if self.quota_blocked:
            return None
        url = f"{BASE_URL}{path}"
        params = {**params, "apiKey": self._key}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as exc:
            logger.warning("odds fetch network error: %s", exc)
            return None

        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            try:
                self.last_remaining = int(remaining)
            except ValueError:
                pass
            if self.last_remaining is not None and self.last_remaining < 5:
                self.quota_blocked = True

        if resp.status_code == 401:
            raise OddsApiError("401 Unauthorized: token The Odds API invalid")
        if resp.status_code == 429:
            logger.warning("odds fetch 429")
            return None
        if resp.status_code >= 400:
            logger.warning("odds fetch http %s", resp.status_code)
            return None

        try:
            await asyncio.sleep(self._throttle)
            return resp.json()
        except (ValueError, json_module_error()):
            return None

    async def fetch_odds(
        self,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h,spreads,totals,btts",
    ) -> list[dict[str, Any]] | None:
        # Some sport keys (e.g. *_qualification) reject the full market set
        # with HTTP 422; retry with progressively fewer markets so odds still
        # come back (4xx responses do not consume quota).
        steps = ("h2h,spreads,totals,btts", "h2h,spreads,totals", "h2h,spreads", "h2h")
        if markets in steps:
            attempts = steps[steps.index(markets):]
        else:
            attempts = (markets,)
        for m in attempts:
            data = await self._get(
                f"/sports/{sport_key}/odds",
                {
                    "regions": regions,
                    "markets": m,
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
            )
            if isinstance(data, list):
                return data
        return None


def json_module_error() -> Exception:
    import json
    return json.JSONDecodeError("x", "", 0)
