"""Tests for proxy support across the data stack.

The SofascoreClient proxy tests were dropped with the API client when the
sofascore fallback paths were removed from the live path (2026-08).

2026-08: SOCKS-proxy regression tests for the httpx-based odds providers
(OddsFetcher, OddspapiClient, NowGoalClient). Latent bug: with a SOCKS
proxy configured -- the bot exports HTTPS_PROXY=socks5h://127.0.0.1:9050
when it auto-detects Tor -- every httpx client raised ImportError at
creation because the 'socksio' package was missing. With socksio in
requirements.txt, creation succeeds and an unreachable proxy degrades to
None (the providers' documented failure contract) instead of crashing the
runner. These tests FAIL when socksio is missing, so the regression cannot
come back silently.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.soccerdata_wrapper import SoccerDataWrapper


def test_soccerdata_wrapper_stores_proxy():
    sw = SoccerDataWrapper(proxy="socks5://127.0.0.1:9050")
    assert sw._proxy == "socks5://127.0.0.1:9050"


def test_soccerdata_wrapper_proxy_default_none():
    sw = SoccerDataWrapper()
    assert sw._proxy is None


# ---- SOCKS (Tor) regression tests: socksio must be present ---------------

def test_socks_proxy_client_construction_ok():
    """Regression: creating an httpx client with a socks5h proxy must NOT
    raise ImportError (the 'socksio missing' bug). No request is made, so
    this is safe even when Tor is running on the real port."""
    import httpx

    client = httpx.AsyncClient(proxy="socks5h://127.0.0.1:9050")
    asyncio.run(client.aclose())


def test_odds_fetcher_degrades_to_none_with_socks_env():
    """OddsFetcher picks the proxy up from HTTPS_PROXY (httpx trust_env). A
    request through an UNREACHABLE socks proxy must return None (ConnectError
    caught), never ImportError. Port 1 is always closed, so the test is
    deterministic even when a real Tor happens to be running."""
    from agents.football.odds_fetcher import OddsFetcher

    async def runner():
        fetcher = OddsFetcher("fake-key", throttle_seconds=0.0)
        result = await fetcher._get("/sports/soccer_epl/odds", {"markets": "h2h"})
        assert result is None
        assert fetcher.quota_blocked is False

    with patch.dict(os.environ, {"HTTPS_PROXY": "socks5h://127.0.0.1:1"}):
        asyncio.run(runner())


def test_oddspapi_degrades_to_none_with_socks_env():
    """OddspapiClient: same trust_env HTTPS_PROXY path -> graceful None."""
    from agents.football.oddspapi import OddspapiClient

    async def runner():
        client = OddspapiClient("fake-key", throttle_seconds=0.0)
        assert await client._get("/fixtures", {"sportId": 10}) is None

    with patch.dict(os.environ, {"HTTPS_PROXY": "socks5h://127.0.0.1:1"}):
        asyncio.run(runner())


def test_nowgoal_degrades_to_none_with_explicit_socks_proxy():
    """NowGoalClient takes an explicit proxy= (the runner passes the
    auto-detected Tor SOCKS URL). Both the JSON and the text (schedule)
    paths must degrade to None through an unreachable socks proxy."""
    from agents.football.nowgoal import NowGoalClient

    async def runner():
        client = NowGoalClient(proxy="socks5h://127.0.0.1:1", throttle_seconds=0.0)
        assert await client._get("/ajax/soccerajax", {"type": 14, "id": 1}) is None
        assert await client._get_text("/ajax/SoccerAjax", {"type": 6}) is None

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