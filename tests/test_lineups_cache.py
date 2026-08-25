"""Tests for the per-match flashscore lineups cache.

The cache lives in MultiSourceStatsFetcher.fetch_flashscore_lineups_for_match:
a repeat query for the same match_url must hit the disk cache (no second
browser call), while a different match_url must fetch again. The cache key is
a hash of the match URL, so a non-fetched (None) result is NOT cached.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.cache import Cache  # noqa: E402
from agents.football.multi_source import MultiSourceStatsFetcher  # noqa: E402


class _FakeFlashscoreClient:
    """Minimal stand-in for FlashscoreClient: counts fetch_match_lineups."""

    def __init__(self, payload: dict | None) -> None:
        self.available = True
        self.payload = payload
        self.calls = 0

    async def fetch_match_lineups(self, match_url: str) -> dict | None:
        self.calls += 1
        return self.payload


def _make_fetcher(cache_dir, payload: dict | None) -> MultiSourceStatsFetcher:
    stats = MultiSourceStatsFetcher(
        cache=Cache(str(cache_dir)),
        flashscore_enabled=True,
    )
    stats.fc = _FakeFlashscoreClient(payload)
    return stats


def test_second_call_hits_cache(tmp_path):
    """The second query for the same match_url is served from disk cache."""
    stats = _make_fetcher(tmp_path, {"status": "predicted", "home_count": 11})
    url = "https://www.flashscore.com/match/football/lyon-x/1/sparta-prague-y/2/?mid=42"

    async def runner():
        first = await stats.fetch_flashscore_lineups_for_match(url)
        second = await stats.fetch_flashscore_lineups_for_match(url)
        return first, second

    first, second = asyncio.run(runner())
    assert first is not None
    assert second == first
    assert stats.fc.calls == 1  # second call served from disk cache


def test_different_match_urls_fetch_twice(tmp_path):
    """Distinct match URLs are distinct cache keys -> two fetches."""
    stats = _make_fetcher(tmp_path, {"status": "predicted", "home_count": 11})
    url_a = "https://www.flashscore.com/match/football/a-team-x/1/b-team-y/2/?mid=1"
    url_b = "https://www.flashscore.com/match/football/c-team-x/3/d-team-y/4/?mid=2"

    async def runner():
        await stats.fetch_flashscore_lineups_for_match(url_a)
        await stats.fetch_flashscore_lineups_for_match(url_b)

    asyncio.run(runner())
    assert stats.fc.calls == 2


def test_none_result_not_cached(tmp_path):
    """A 'not announced yet' (None) result must NOT poison the cache."""
    stats = _make_fetcher(tmp_path, None)
    url = "https://www.flashscore.com/match/football/a-team-x/1/b-team-y/2/?mid=3"

    async def runner():
        first = await stats.fetch_flashscore_lineups_for_match(url)
        second = await stats.fetch_flashscore_lineups_for_match(url)
        return first, second

    first, second = asyncio.run(runner())
    assert first is None
    assert second is None
    # Lineups may appear later: a second query must retry, not reuse None.
    assert stats.fc.calls == 2


def test_disabled_flashscore_skips_cache(tmp_path):
    """When flashscore is unavailable, no fetch and no cache interaction."""
    stats = _make_fetcher(tmp_path, {"status": "predicted", "home_count": 11})
    stats.fc.available = False
    url = "https://www.flashscore.com/match/football/a-team-x/1/b-team-y/2/?mid=4"

    async def runner():
        return await stats.fetch_flashscore_lineups_for_match(url)

    result = asyncio.run(runner())
    assert result is None
    assert stats.fc.calls == 0


def test_disk_cache_serves_new_process(tmp_path):
    """Disk persistence: a SECOND fetcher (fresh in-memory cache, as in a new
    Discord command subprocess) must be served from disk without rendering
    the browser again."""
    url = "https://www.flashscore.com/match/football/lyon-x/1/sparta-prague-y/2/?mid=99"

    # First 'process': fetches, writes the disk cache.
    stats_a = _make_fetcher(tmp_path, {"status": "predicted", "home_count": 11})

    async def runner_a():
        return await stats_a.fetch_flashscore_lineups_for_match(url)

    first = asyncio.run(runner_a())
    assert first is not None
    assert stats_a.fc.calls == 1

    # Second 'process': brand-new Cache (cold memory) + brand-new client.
    stats_b = _make_fetcher(tmp_path, {"status": "predicted", "home_count": 11})
    assert stats_b.cache is not None
    stats_b.cache._mem.clear()  # simulate a fresh subprocess (no mem cache)

    async def runner_b():
        return await stats_b.fetch_flashscore_lineups_for_match(url)

    second = asyncio.run(runner_b())
    assert second is not None
    assert second == first
    assert stats_b.fc.calls == 0  # served entirely from the disk cache
