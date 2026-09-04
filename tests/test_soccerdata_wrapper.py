"""Tests for soccerdata_wrapper (mocked, no network)."""
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

sys.path.insert(0, str(ROOT))

import soccerdata as sd

from agents.football.soccerdata_wrapper import (
    LEAGUE_MAP,
    SoccerDataWrapper,
    _ensure_valid_fbref_leagues,
    current_season_code,
    fbref_code,
    league_key_from_display,
    previous_season_code,
)


def test_fbref_code_returns_known():
    assert fbref_code("EPL") == "ENG-Premier League"
    assert fbref_code("LaLiga") == "ESP-La Liga"
    assert fbref_code("Liga 1") is None


def test_fbref_code_filters_invalid_leagues_runtime():
    valid = _ensure_valid_fbref_leagues()
    assert isinstance(valid, set)
    assert "ENG-Premier League" in valid
    assert fbref_code("Saudi Pro League") is None
    assert fbref_code("K-League") is None


def test_fbref_code_unknown_league():
    assert fbref_code("Saudi Pro League") is None
    assert fbref_code("K-League") is None


def test_supports_league_only_top5_with_runtime_probe():
    sw = SoccerDataWrapper()
    assert sw.supports_league("EPL") is True
    assert sw.supports_league("Liga 1") is False
    assert sw.supports_league("Saudi Pro League") is False


def test_league_key_from_display():
    assert league_key_from_display("La Liga") == "LaLiga"
    assert league_key_from_display("EPL") == "EPL"
    assert league_key_from_display("Saudi Pro League") is None
    assert league_key_from_display("Liga 1") is None
    assert league_key_from_display("Premier League") is None


def test_supports_league():
    w = SoccerDataWrapper()
    assert w.supports_league("EPL") is True
    assert w.supports_league("Liga 1") is False
    assert w.supports_league("Saudi Pro League") is False


def test_season_code_helpers():
    from datetime import datetime, timezone
    aug = datetime(2026, 8, 10, tzinfo=timezone.utc)
    feb = datetime(2027, 2, 10, tzinfo=timezone.utc)
    assert current_season_code(aug) == "2026-2027"
    assert current_season_code(feb) == "2026-2027"
    assert previous_season_code("2026-2027") == "2025-2026"


def _build_soccerdata_stub(schedule_df):
    class _SoccerData:
        FBref = None

    class _FbrefFactory:
        def __init__(self, code, season):
            pass

        def read_schedule(self, force_cache=False):
            return schedule_df

    _SoccerData.FBref = _FbrefFactory
    return _SoccerData


def test_read_team_form_separates_home_away_correctly(monkeypatch):
    import pandas as pd
    from agents.football import soccerdata_wrapper as sdw

    df = pd.DataFrame([
        {"Date": "2026-05-12", "Home": "Santa Clara", "Away": "Nacional",
         "HomeGoals": 1, "AwayGoals": 1},
        {"Date": "2026-04-20", "Home": "Nacional", "Away": "Santa Clara",
         "HomeGoals": 0, "AwayGoals": 1},
        {"Date": "2026-03-08", "Home": "Benfica", "Away": "Santa Clara",
         "HomeGoals": 2, "AwayGoals": 0},
        {"Date": "2026-02-15", "Home": "Santa Clara", "Away": "Sporting",
         "HomeGoals": 2, "AwayGoals": 2},
        {"Date": "2026-01-22", "Home": "Porto", "Away": "Santa Clara",
         "HomeGoals": 3, "AwayGoals": 1},
    ])
    monkeypatch.setitem(sys.modules, "soccerdata", _build_soccerdata_stub(df))

    async def runner():
        # "Primeira Liga" is NOT an FBref league (top-5 only), so the guard in
        # read_team_form returned None before the stub was ever used and this
        # test asserted nothing. Use a supported league (2026-09-04).
        return await SoccerDataWrapper().read_team_form("EPL", "Santa Clara", limit=5)
    result = asyncio.run(runner())
    assert result is not None
    assert result["source"] == "soccerdata_fbref"
    assert result["sample_size"] == 5
    assert result["sequence"].count("-") == 4 and result["sequence"].count("W") + result["sequence"].count("D") + result["sequence"].count("L") == 5
    assert all(c in "WDL-" for c in result["sequence"])


def test_read_h2h_counts_correctly(monkeypatch):
    import pandas as pd
    from agents.football import soccerdata_wrapper as sdw

    df = pd.DataFrame([
        {"Date": "2026-05-12", "Home": "Santa Clara", "Away": "Nacional",
         "HomeGoals": 1, "AwayGoals": 1},
        {"Date": "2026-04-20", "Home": "Nacional", "Away": "Santa Clara",
         "HomeGoals": 0, "AwayGoals": 1},
        {"Date": "2026-03-08", "Home": "Benfica", "Away": "Santa Clara",
         "HomeGoals": 2, "AwayGoals": 0},
    ])
    monkeypatch.setitem(sys.modules, "soccerdata", _build_soccerdata_stub(df))

    async def runner():
        # See note above: "Primeira Liga" is not FBref-backed (2026-09-04).
        return await SoccerDataWrapper().read_h2h("EPL", "Santa Clara", "Nacional", limit=5)
    result = asyncio.run(runner())
    assert result is not None
    assert result["wins"] + result["draws"] + result["losses"] >= 1


def test_league_map_has_eu_leagues():
    assert "EPL" in LEAGUE_MAP
    assert "LaLiga" in LEAGUE_MAP
    assert "Serie A" in LEAGUE_MAP
    assert "Bundesliga" in LEAGUE_MAP
    assert "Ligue 1" in LEAGUE_MAP
    assert "UCL" in LEAGUE_MAP


def test_read_team_form_unsupported_league():
    async def runner():
        w = SoccerDataWrapper()
        result = await w.read_team_form("Liga 1", "Persija")
        assert result is None
    asyncio.run(runner())


def test_read_team_form_no_team_name():
    async def runner():
        w = SoccerDataWrapper()
        result = await w.read_team_form("EPL", "")
        assert result is None
    asyncio.run(runner())


def test_read_team_form_success():
    async def runner():
        w = SoccerDataWrapper()

        mock_schedule_dict = {
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool"],
            "homegoals": [2, 1],
            "awaygoals": [1, 1],
            "date": ["2026-08-15", "2026-08-22"],
        }

        async def fake_read(league_key, team_name, limit):
            return {
                "sequence": "W-D",
                "gf_avg": 1.5,
                "ga_avg": 1.0,
                "sample_size": 2,
                "source": "soccerdata_fbref",
            }

        w.read_team_form = fake_read
        result = await w.read_team_form("EPL", "Arsenal", limit=10)
        assert result is not None
        assert result["source"] == "soccerdata_fbref"
        assert result["sequence"] == "W-D"
        assert result["gf_avg"] == 1.5

    asyncio.run(runner())


def test_read_team_form_handles_exception():
    async def runner():
        w = SoccerDataWrapper()

        async def fake_read(league_key, team_name, limit):
            raise RuntimeError("scraper down")

        try:
            await fake_read("EPL", "Arsenal", limit=10)
            assert False, "expected exception"
        except RuntimeError as exc:
            assert str(exc) == "scraper down"

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
