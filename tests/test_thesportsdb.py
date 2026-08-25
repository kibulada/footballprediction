"""Tests for thesportsdb.py (pure unit, mock HTTP)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.thesportsdb import TheSportsDbClient


def _patch_get_response(status: int, json_data: dict | None):
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
    patcher = patch("agents.football.thesportsdb.httpx.AsyncClient")
    mock_cls = patcher.start()
    mock_cls.return_value = fake_client
    return patcher


def test_thesportsdb_default_key():
    # key "1" (legacy free) now returns 400 Invalid Premium API key;
    # "3" is the working default.
    client = TheSportsDbClient()
    assert client._key == "3"


def test_thesportsdb_custom_key():
    client = TheSportsDbClient("3")
    assert client._key == "3"


def test_thesportsdb_search_team():
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        fake = {"teams": [{"idTeam": "133604", "strTeam": "Arsenal", "strCountry": "England"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team("Arsenal")
        finally:
            patcher.stop()
        assert result is not None
        assert result["idTeam"] == "133604"
    asyncio.run(runner())


def test_thesportsdb_search_team_prefix_stripped():
    """'PFC Levski Sofia' resolves to thesportsdb's 'Levski Sofia'."""
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        # First call (raw 'PFC Levski Sofia') -> no teams; second call
        # (stripped 'Levski Sofia') -> match.
        responses = [
            MagicMock(status_code=200, json=MagicMock(return_value={"teams": None})),
            MagicMock(status_code=200, json=MagicMock(return_value={"teams": [{"idTeam": "134085", "strTeam": "Levski Sofia"}]})),
        ]

        async def fake_get(*args, **kwargs):
            return responses.pop(0)

        cm = MagicMock()
        cm.get = fake_get

        async def fake_aenter(*args, **kwargs):
            return cm

        async def fake_aexit(*args, **kwargs):
            return False

        fake_client = MagicMock()
        fake_client.__aenter__ = fake_aenter
        fake_client.__aexit__ = fake_aexit

        patcher = _patch_httpx(fake_client)
        try:
            result = await client.search_team("PFC Levski Sofia")
        finally:
            patcher.stop()
        assert result is not None
        assert result["idTeam"] == "134085"
    asyncio.run(runner())


def test_thesportsdb_search_team_normalize_dash():
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        fake = {"teams": [{"idTeam": "137892", "strTeam": "Ararat-Armenia"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team("Ararat Armenia")
        finally:
            patcher.stop()
        assert result is not None
        assert result["idTeam"] == "137892"
    asyncio.run(runner())


def test_thesportsdb_search_team_no_match():
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        patcher = _patch_httpx(_patch_get_response(200, {"teams": None}))
        try:
            result = await client.search_team("Nonexistent FC")
        finally:
            patcher.stop()
        assert result is None
    asyncio.run(runner())


def test_thesportsdb_search_team_fallback_rejected_when_league_mismatch():
    """F2: a teams[0] guess with NO name match is rejected when its league
    contradicts the requested league (wrong-club guard)."""
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        fake = {"teams": [{"idTeam": "135111", "strTeam": "Unrelated Club",
                            "strLeague": "Spanish Segunda Division"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team("Dep. A Coruna", "La Liga")
        finally:
            patcher.stop()
        assert result is None
    asyncio.run(runner())


def test_thesportsdb_search_team_fallback_kept_when_league_matches():
    """F2: the teams[0] guess survives when its league is consistent with the
    requested league."""
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        fake = {"teams": [{"idTeam": "135111", "strTeam": "Deportivo Coruna",
                            "strLeague": "Spanish La Liga"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team("Dep. A Coruna", "La Liga")
        finally:
            patcher.stop()
        assert result is not None
        assert result["idTeam"] == "135111"
    asyncio.run(runner())


def test_thesportsdb_search_team_exact_name_wins_over_league_guard():
    """F2: an exact name match always wins, regardless of league fields."""
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        fake = {"teams": [{"idTeam": "133604", "strTeam": "Barcelona",
                            "strLeague": "French Ligue 1"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team("Barcelona", "La Liga")
        finally:
            patcher.stop()
        assert result is not None
        assert result["idTeam"] == "133604"
    asyncio.run(runner())


def test_thesportsdb_search_team_fallback_without_hint_keeps_old_behavior():
    """F2: without a league hint the teams[0] fallback keeps its old behavior
    (backward compatible for callers with no league context)."""
    async def runner():
        client = TheSportsDbClient(throttle_seconds=0)
        fake = {"teams": [{"idTeam": "1", "strTeam": "Whatever FC",
                            "strLeague": "Some League"}]}
        patcher = _patch_httpx(_patch_get_response(200, fake))
        try:
            result = await client.search_team("Xyz")
        finally:
            patcher.stop()
        assert result is not None
        assert result["idTeam"] == "1"
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
