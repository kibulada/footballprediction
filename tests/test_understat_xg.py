"""Tests for understat_xg.py (offline xG dataset helpers).

Covers: dates_to_rows parsing, team_xg_history_from_rows rolling last-5
aggregation (home/away perspective, tolerant names, empty -> None), the
anti-leakage exclude of the predicted fixture, and the multi_source
fetch_team_xg_history wrapper under Plan B (2026-08-17): understat is no
longer a live xG source -- NowGoal primary, flashscore fallback.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.multi_source import MultiSourceStatsFetcher
from agents.football.understat_xg import dates_to_rows, team_xg_history_from_rows


def _payload():
    # 6 finished matches: Arsenal home vs X, X away vs Arsenal, ... so the
    # rolling window and the home/away perspective both get exercised.
    def m(i, h, a, xh, xa, dt, result=True):
        return {
            "id": str(i),
            "isResult": result,
            "h": {"title": h},
            "a": {"title": a},
            "goals": {"h": "1", "a": "1"},
            "xG": {"h": str(xh), "a": str(xa)},
            "datetime": f"{dt} 15:00:00",
        }

    return {
        "dates": [
            m(1, "Arsenal", "Burnley", 1.8, 0.4, "2026-01-10"),
            m(2, "Everton", "Arsenal", 1.1, 1.6, "2026-01-21"),
            m(3, "Arsenal", "Chelsea", 2.2, 0.9, "2026-02-01"),
            m(4, "Wolves", "Arsenal", 0.7, 2.4, "2026-02-14"),
            m(5, "Arsenal", "Liverpool", 1.3, 1.5, "2026-02-28"),
            m(6, "Fulham", "Arsenal", 0.9, 1.2, "2026-03-14"),
            # upcoming match (not finished) -> must be ignored
            m(7, "Arsenal", "Spurs", 0.0, 0.0, "2026-08-12", result=False),
        ]
    }


def _rows():
    return dates_to_rows(_payload(), "EPL", 2026)


def test_dates_to_rows_parses_goals_and_xg():
    rows = _rows()
    assert len(rows) == 6  # upcoming match excluded (isResult False)
    first = rows[0]
    assert first["home"] == "Arsenal"
    assert first["away"] == "Burnley"
    assert first["home_xg"] == 1.8
    assert first["away_xg"] == 0.4
    assert first["home_goals"] == 1
    assert first["season"] == "2026-2027"


def test_team_xg_history_last5_and_perspective():
    hist = team_xg_history_from_rows(_rows(), "Arsenal", limit=5)
    assert hist is not None
    assert hist["sample_size"] == 5
    # Last 5 finished matches (rows 2..6); Arsenal xG: home 1.6, away 2.4,
    # home 1.3, away 1.2 -> from Arsenal's perspective (is_home -> home_xg):
    # away games use away_xg. matches 2..6: (away 1.6), (home 2.2),
    # (away 2.4), (home 1.3), (away 1.2)
    xg_for = (1.6 + 2.2 + 2.4 + 1.3 + 1.2) / 5
    xg_against = (1.1 + 0.9 + 0.7 + 1.5 + 0.9) / 5
    assert abs(hist["xg_for_avg"] - round(xg_for, 4)) < 1e-6
    assert abs(hist["xg_against_avg"] - round(xg_against, 4)) < 1e-6
    assert hist["source"] == "understat_history"


def test_team_xg_history_tolerant_name():
    hist = team_xg_history_from_rows(_rows(), "Arsenal FC", limit=5)
    assert hist is not None and hist["sample_size"] == 5
    hist2 = team_xg_history_from_rows(_rows(), "Arsenal", limit=5)
    assert hist["xg_for_avg"] == hist2["xg_for_avg"]


def test_team_xg_history_limit_window():
    hist = team_xg_history_from_rows(_rows(), "Arsenal", limit=3)
    assert hist["sample_size"] == 3
    # last 3: (home 2.2), (away 2.4), (home 1.3) -> wait rows 4..6
    # row4 away 2.4, row5 home 1.3, row6 away 1.2
    xg_for = (2.4 + 1.3 + 1.2) / 3
    assert abs(hist["xg_for_avg"] - round(xg_for, 4)) < 1e-6


def test_team_xg_history_exclude_predicted_fixture():
    # If the predicted fixture (Arsenal vs Spurs on 2026-08-12) were somehow
    # finished, excluding it by (home, away, date) must keep it out of the
    # rolling window (anti-leakage).
    rows = _rows() + [
        {
            "date": "2026-08-12",
            "home": "Arsenal",
            "away": "Spurs",
            "home_xg": 9.9,
            "away_xg": 9.9,
            "league": "EPL",
            "season": "2026-2027",
        }
    ]
    hist = team_xg_history_from_rows(
        rows, "Arsenal", limit=5,
        exclude=("Arsenal", "Spurs", "2026-08-12"),
    )
    assert hist["sample_size"] == 5
    assert 9.9 not in (hist["xg_for_avg"], hist["xg_against_avg"])


def test_team_xg_history_canonical_query_side_join():
    """F1 (2026-08-17): the live path queries with the flashscore spelling
    ("Manchester United") while understat rows carry the canonical
    football-data.co.uk name ("Manchester Utd" after TEAM_RAW_MAP +
    TEAM_NAME_MAP). The query side must be canonicalized through the SAME
    pipeline before tolerant matching, or the join silently returns None
    for exactly the teams the backtest used it for."""
    rows = [
        {"date": "2026-04-10", "home": "Manchester Utd", "away": "Wolves",
         "home_xg": 1.8, "away_xg": 0.9, "league": "EPL", "season": "2025-2026"},
        {"date": "2026-04-24", "home": "Chelsea", "away": "Manchester Utd",
         "home_xg": 1.2, "away_xg": 1.5, "league": "EPL", "season": "2025-2026"},
    ]
    hist = team_xg_history_from_rows(rows, "Manchester United", limit=5)
    assert hist is not None and hist["sample_size"] == 2
    assert abs(hist["xg_for_avg"] - (1.8 + 1.5) / 2) < 1e-6
    assert abs(hist["xg_against_avg"] - (0.9 + 1.2) / 2) < 1e-6
    # The canonical spelling (backtest path) still resolves identically.
    hist2 = team_xg_history_from_rows(rows, "Manchester Utd", limit=5)
    assert hist2 is not None and hist2["sample_size"] == 2
    assert hist["xg_for_avg"] == hist2["xg_for_avg"]


def test_team_xg_history_canonical_exclude():
    """F1: the anti-leakage exclude must canonicalize BOTH sides too, or a
    just-finished predicted fixture (queried with the flashscore spelling)
    misses the canonical row and leaks its own xG into the history."""
    rows = [
        {"date": "2026-04-10", "home": "Manchester Utd", "away": "Wolves",
         "home_xg": 1.8, "away_xg": 0.9, "league": "EPL", "season": "2025-2026"},
        {"date": "2026-05-09", "home": "Manchester Utd", "away": "Bournemouth",
         "home_xg": 9.9, "away_xg": 9.9, "league": "EPL", "season": "2025-2026"},
    ]
    hist = team_xg_history_from_rows(
        rows, "Manchester United", limit=5,
        exclude=("Manchester United", "Bournemouth", "2026-05-09"),
    )
    assert hist is not None and hist["sample_size"] == 1
    assert 9.9 not in (hist["xg_for_avg"], hist["xg_against_avg"])


def test_team_xg_history_unknown_team_returns_none():
    assert team_xg_history_from_rows(_rows(), "Real Madrid", limit=5) is None
    assert team_xg_history_from_rows([], "Arsenal", limit=5) is None


def test_multi_source_no_xg_source_returns_none():
    """Plan B (2026-08-17): understat is REMOVED from the live xG chain, so
    with neither a nowgoal client nor flashscore enabled the wrapper returns
    None (no fabricated xG) -- understat rows are never consulted."""
    fetcher = MultiSourceStatsFetcher("fd", "", flashscore_enabled=False)
    meta = {"_league_key": "EPL", "display": "EPL", "football_data_code": "E0"}
    assert asyncio.run(fetcher.fetch_team_xg_history("Arsenal", meta)) is None


def test_multi_source_nowgoal_tier_rolls_from_match_list():
    """NowGoal primary tier: match_list -> canonical match_id -> live-{id}
    FT xG -> rolling window of max 3, provenance returned."""
    fetcher = MultiSourceStatsFetcher("fd", "", flashscore_enabled=False)
    ng = SimpleNamespace(
        fetch_match_xg=AsyncMock(side_effect=[
            {"xg_home": 1.2, "xg_away": 0.8},   # Arsenal home vs X
            {"xg_home": 0.9, "xg_away": 1.5},   # Y vs Arsenal (away)
            {"xg_home": 2.1, "xg_away": 0.4},   # Arsenal home vs Z
        ]),
    )
    match_list = [
        {"match_id": "101", "date": "2026-08-10 20:00:00", "home": "Arsenal", "away": "Wolves"},
        {"match_id": "102", "date": "2026-08-03 20:00:00", "home": "Brighton", "away": "Arsenal"},
        {"match_id": "103", "date": "2026-07-27 20:00:00", "home": "Arsenal", "away": "Bournemouth"},
    ]
    meta = {"_league_key": "EPL", "display": "EPL"}
    hist = asyncio.run(fetcher.fetch_team_xg_history(
        "Arsenal", meta, nowgoal_client=ng, match_list=match_list
    ))
    assert hist is not None
    assert hist["source"] == "nowgoal_xg"
    assert hist["xg_source"] == "nowgoal_xg"
    assert hist["sample_size"] == 3
    assert hist["match_ids"] == ["101", "102", "103"]
    # Arsenal perspective: home 1.2, away 1.5, home 2.1
    assert abs(hist["xg_for_avg"] - (1.2 + 1.5 + 2.1) / 3) < 1e-6
    assert abs(hist["xg_against_avg"] - (0.8 + 0.9 + 0.4) / 3) < 1e-6


def test_multi_source_nowgoal_tier_anti_leak_excludes_fixture_date():
    """NowGoal tier anti-leak: a finished match on the predicted fixture
    date is excluded so it can never leak its own stats."""
    fetcher = MultiSourceStatsFetcher("fd", "", flashscore_enabled=False)
    _xg_by_id = {
        "201": {"xg_home": 9.9, "xg_away": 9.9},  # 2026-08-17 -> excluded
        "202": {"xg_home": 1.2, "xg_away": 0.9},  # valid
    }
    ng = SimpleNamespace(
        fetch_match_xg=AsyncMock(side_effect=lambda mid: _xg_by_id.get(str(mid))),
    )
    match_list = [
        {"match_id": "201", "date": "2026-08-17 20:00:00", "home": "Arsenal", "away": "Chelsea"},
        {"match_id": "202", "date": "2026-08-09 20:00:00", "home": "Arsenal", "away": "Spurs"},
    ]
    meta = {"_league_key": "EPL", "display": "EPL"}
    hist = asyncio.run(fetcher.fetch_team_xg_history(
        "Arsenal", meta,
        exclude=("Arsenal", "Chelsea", "2026-08-17"),
        nowgoal_client=ng, match_list=match_list,
    ))
    assert hist is not None
    assert hist["sample_size"] == 1
    assert abs(hist["xg_for_avg"] - 1.2) < 1e-6
    assert hist["match_ids"] == ["202"]


def test_multi_source_nowgoal_tier_skips_friendly_no_xg():
    """NowGoal tier: a friendly / no-xG page returns None and is skipped;
    when no match yields xG the wrapper returns None (no fabrication)."""
    fetcher = MultiSourceStatsFetcher("fd", "", flashscore_enabled=False)
    ng = SimpleNamespace(fetch_match_xg=AsyncMock(return_value=None))
    match_list = [{"match_id": "301", "date": "2026-08-10", "home": "Arsenal", "away": "Wolves"}]
    meta = {"_league_key": "EPL", "display": "EPL"}
    assert asyncio.run(fetcher.fetch_team_xg_history(
        "Arsenal", meta, nowgoal_client=ng, match_list=match_list
    )) is None
