"""Tests for odds_fetcher.py market-step fallback.

Some sport keys (qualification leagues, and even the main UCL key) reject
the full market set with HTTP 422; fetch_odds must retry with progressively
fewer markets instead of giving up (4xx responses consume no quota).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.odds_fetcher import OddsFetcher  # noqa: E402

FULL = "h2h,spreads,totals,btts"
TRIO = "h2h,spreads,totals"
DUO = "h2h,spreads"
H2H = "h2h"


def _fetcher(side_effect) -> OddsFetcher:
    f = OddsFetcher("fake-key", throttle_seconds=0)
    f._get = AsyncMock(side_effect=side_effect)
    return f


def test_falls_back_until_h2h_succeeds():
    async def runner():
        f = _fetcher([None, None, None, [{"home_team": "A", "away_team": "B"}]])
        out = await f.fetch_odds("soccer_uefa_champs_league_qualification")
        assert out == [{"home_team": "A", "away_team": "B"}]
        markets = [c.args[1]["markets"] for c in f._get.await_args_list]
        assert markets == [FULL, TRIO, DUO, H2H]
    asyncio.run(runner())


def test_stops_at_first_success():
    async def runner():
        f = _fetcher([None, [{"home_team": "X"}]])
        out = await f.fetch_odds("some_key")
        assert out == [{"home_team": "X"}]
        markets = [c.args[1]["markets"] for c in f._get.await_args_list]
        assert markets == [FULL, TRIO]
    asyncio.run(runner())


def test_h2h_only_is_single_attempt():
    async def runner():
        f = _fetcher([[{"home_team": "A"}]])
        out = await f.fetch_odds("some_key", markets="h2h")
        assert out is not None
        markets = [c.args[1]["markets"] for c in f._get.await_args_list]
        assert markets == [H2H]
    asyncio.run(runner())


def test_all_fail_returns_none():
    async def runner():
        f = _fetcher([None, None, None, None])
        out = await f.fetch_odds("some_key")
        assert out is None
        assert f._get.await_count == 4
    asyncio.run(runner())


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
