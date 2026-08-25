"""Tests for football_data.py (pure unit, mock HTTP)."""
from __future__ import annotations

import asyncio
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

from agents.football.football_data import FootballDataClient, FootballDataError


def _patch_get_response(status: int, json_data: dict | None):
    """Patch httpx.AsyncClient.get to return a controlled response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json = MagicMock(return_value=json_data)

    async def fake_get(*args, **kwargs):
        return mock_resp

    cm = MagicMock()
    cm.get = fake_get

    async def fake_aenter(*args, **kwargs):
        return cm

    async def fake_aexit(*args, **kwargs):
        return False

    client = MagicMock()
    client.__aenter__ = fake_aenter
    client.__aexit__ = fake_aexit
    return client


def _patch_httpx(fake_client):
    patcher = patch("agents.football.football_data.httpx.AsyncClient")
    mock_cls = patcher.start()
    mock_cls.return_value = fake_client
    return patcher


def test_football_data_init_no_key():
    """Empty key now allowed (soccerdata doesn't require it)."""
    client = FootballDataClient("")
    assert client._key == ""


def test_football_data_init_with_key():
    client = FootballDataClient("test-key", throttle_seconds=0)
    assert client._key == "test-key"
    assert client.rate_limit_warning is False


def test_football_data_fetch_teams():
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        fake = {"teams": [{"id": 1, "name": "Test FC"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.fetch_teams("PL")
        finally:
            patcher.stop()
        assert result == [{"id": 1, "name": "Test FC"}]
    asyncio.run(runner())


def test_football_data_fetch_teams_429():
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        patcher = _patch_httpx(_patch_get_response(429, {}))
        try:
            result = await client.fetch_teams("PL")
        finally:
            patcher.stop()
        assert result is None
        assert client.rate_limit_warning is True
    asyncio.run(runner())


def test_football_data_fetch_teams_401():
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        patcher = _patch_httpx(_patch_get_response(401, {}))
        try:
            await client.fetch_teams("PL")
        except FootballDataError:
            patcher.stop()
            return
        finally:
            pass
        patcher.stop()
        raise AssertionError("expected 401 FootballDataError")
    asyncio.run(runner())


def test_search_team_in_competition_match():
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        fake = {
            "teams": [
                {"id": 50, "name": "Manchester City FC", "shortName": "Man City", "tla": "MCI"},
                {"id": 42, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
            ]
        }
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team_in_competition("Arsenal", "PL")
        finally:
            patcher.stop()
        assert result is not None
        assert result["id"] == 42
    asyncio.run(runner())


def test_search_team_in_competition_ignores_club_prefix_token():
    """'NK Celje' must NOT match 'Eintracht Frankfurt' via the 'nk' inside."""
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        fake = {
            "teams": [
                {"id": 19, "name": "Eintracht Frankfurt", "shortName": "Frankfurt", "tla": "SGE"},
                {"id": 65, "name": "NK Celje", "shortName": "Celje", "tla": "CEL"},
            ]
        }
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team_in_competition("NK Celje", "CL")
        finally:
            patcher.stop()
        assert result is not None
        assert result["id"] == 65
    asyncio.run(runner())


def test_search_team_in_competition_fk_token_does_not_force_match():
    """'FK Kauno Žalgiris' must not match 'Qarabağ Ağdam FK' (both share 'fk')."""
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        fake = {
            "teams": [
                {"id": 611, "name": "Qarabağ Ağdam FK", "shortName": "Qarabağ", "tla": "QAR"},
                {"id": 900, "name": "FK Kauno Žalgiris", "shortName": "Kauno Žalgiris", "tla": "KAZ"},
            ]
        }
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team_in_competition("FK Kauno Žalgiris", "CL")
        finally:
            patcher.stop()
        assert result is not None
        assert result["id"] == 900
    asyncio.run(runner())


def test_search_team_in_competition_no_match():
    async def runner():
        client = FootballDataClient("k", throttle_seconds=0)
        fake = {"teams": [{"id": 1, "name": "Chelsea"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team_in_competition("Santos", "PL")
        finally:
            patcher.stop()
        assert result is None
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
