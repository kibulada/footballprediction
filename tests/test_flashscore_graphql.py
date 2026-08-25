"""Tests for the Flashscore GraphQL context client (missing players, coaches).

Covers the pure normalization (dlie2 + dmpe2 payloads -> home/away dict with
side resolution against the resolved team names), tolerant team matching, and
the client's fetch wiring. No network / no browser: payloads are synthetic
samples of the real 2026-08 responses.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.flashscore_graphql import (  # noqa: E402
    FlashscoreGraphqlClient,
    normalize_event_context,
    same_team,
)


def _dlie2() -> dict:
    """dlie2 payload sample: names + formation + coaches per side."""
    return {
        "data": {"findEventById": {"eventParticipants": [
            {
                "id": "Oj6zfFs0", "type": {"side": "HOME"}, "name": "Palmeiras",
                "lineup": {
                    "formation": "4-2-3-1",
                    "players": [{"name": "Weverton", "shirtNumber": "21"}],
                    "coaches": {"name": "Coaches", "players": [{"name": "Abel Ferreira"}]},
                },
            },
            {
                "id": "ldpAnFkt", "type": {"side": "AWAY"}, "name": "Cerro Porteno",
                "lineup": {
                    "formation": None, "players": [],
                    "coaches": {"name": "Coaches", "players": [{"name": "M. Jimenez"}]},
                },
            },
        ]}}
    }


def _dmpe2() -> dict:
    """dmpe2 payload sample: missing + unsure players per side."""
    return {
        "data": {"findEventById": {"eventParticipants": [
            {
                "id": "Oj6zfFs0", "type": {"side": "HOME"},
                "lineup": {
                    "missingPlayers": [
                        {"reason": "Surgery", "player": {"name": "Bruno Fuchs"}},
                        {"reason": "Knee Injury", "player": {"name": "Jefte"}},
                    ],
                    "unsureMissingPlayers": [],
                },
            },
            {
                "id": "ldpAnFkt", "type": {"side": "AWAY"},
                "lineup": {
                    "missingPlayers": [
                        {"reason": "Surgery", "player": {"name": "Aguayo G."}},
                    ],
                    "unsureMissingPlayers": [
                        {"reason": "Muscle Injury", "player": {"name": "Doubtful X"}},
                    ],
                },
            },
        ]}}
    }


def test_normalize_resolves_sides_by_name():
    out = normalize_event_context(_dlie2(), _dmpe2(), "Palmeiras", "Cerro Porteno")
    assert out is not None
    assert out["home"]["name"] == "Palmeiras"
    assert out["away"]["name"] == "Cerro Porteno"
    assert out["home"]["formation"] == "4-2-3-1"
    assert out["home"]["coaches"] == ["Abel Ferreira"]
    assert out["away"]["coaches"] == ["M. Jimenez"]
    assert out["home"]["players"] == [{"name": "Weverton", "shirt": "21"}]
    assert out["source"] == "flashscore_graphql"


def test_normalize_missing_players_with_reasons():
    out = normalize_event_context(_dlie2(), _dmpe2(), "Palmeiras", "Cerro Porteno")
    home_missing = out["home"]["missing"]
    assert {"name": "Bruno Fuchs", "reason": "Surgery"} in home_missing
    assert {"name": "Jefte", "reason": "Knee Injury"} in home_missing
    assert out["away"]["missing"] == [{"name": "Aguayo G.", "reason": "Surgery"}]
    assert out["away"]["unsure"] == [{"name": "Doubtful X", "reason": "Muscle Injury"}]
    assert out["home"]["unsure"] == []


def test_sides_resolved_by_name_not_by_flashscore_order():
    """The AWAY side data must map to 'home' when the caller's home team is
    flashscore's away participant (participant order is not home-first)."""
    out = normalize_event_context(_dlie2(), _dmpe2(), "Cerro Porteno", "Palmeiras")
    assert out["home"]["name"] == "Cerro Porteno"
    assert out["home"]["missing"] == [{"name": "Aguayo G.", "reason": "Surgery"}]
    assert out["away"]["name"] == "Palmeiras"
    assert out["away"]["missing"][0]["name"] == "Bruno Fuchs"


def test_no_payload_returns_none():
    assert normalize_event_context(None, None, "A", "B") is None
    assert normalize_event_context({}, {}, "A", "B") is None
    assert normalize_event_context({"data": {"findEventById": None}}, {}, "A", "B") is None


def test_normalize_without_names_keeps_order():
    out = normalize_event_context(_dlie2(), _dmpe2())
    assert out is not None
    # no names to match: HOME maps to home, AWAY to away (documented guess)
    assert out["home"]["name"] == "Palmeiras"
    assert out["away"]["name"] == "Cerro Porteno"


def test_same_team_tolerant_matching():
    assert same_team("Palmeiras", "SE Palmeiras")
    assert same_team("FK Bodo/Glimt", "Bodø/Glimt")
    assert same_team("Barcelona", "FC Barcelona")
    assert not same_team("Palmeiras", "Cerro Porteno")
    assert not same_team("", "Cerro Porteno")


async def _run_client_fetch():
    client = FlashscoreGraphqlClient(throttle_seconds=0.0)
    with patch.object(
        client, "_get", AsyncMock(side_effect=[_dlie2(), _dmpe2()])
    ) as mock_get:
        out = await client.fetch_event_context(
            "abc123", "Palmeiras", "Cerro Porteno"
        )
    assert mock_get.await_count == 2
    return out


def test_client_fetch_event_context():
    out = asyncio.run(_run_client_fetch())
    assert out is not None
    assert out["home"]["name"] == "Palmeiras"
    assert len(out["home"]["missing"]) == 2


def test_client_fetch_empty_event_id():
    client = FlashscoreGraphqlClient(throttle_seconds=0.0)
    out = asyncio.run(client.fetch_event_context(""))
    assert out is None


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
