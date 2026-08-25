"""Tests for the top-match league/nowgoal helpers in match_finder.py.

Covers the strict nowgoal league-name matcher (regression for the 2026-08-14
flood: the loose substring test matched "premier league" inside "National
Premier Leagues NSW" / "Russian Premier League", filling the EPL fixture
cache with hundreds of unrelated matches and blowing the runner deadline)
and the accent folding in _norm_league_token.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.match_finder import (  # noqa: E402
    _norm_league_token,
    _nowgoal_fixtures_for_league,
    _nowgoal_league_matches,
)


def _leagues() -> dict:
    return json.loads((ROOT / "agents/football/leagues.json").read_text(encoding="utf-8"))


LEAGUES = _leagues()
EPL = LEAGUES["EPL"]
UCL = LEAGUES["UCL"]
UEL = LEAGUES["UEL"]
UECL = LEAGUES["UECL"]
LALIGA = LEAGUES["LaLiga"]
SERIE_A = LEAGUES["Serie A"]
BUNDESLIGA = LEAGUES["Bundesliga"]
EFL = LEAGUES["EFL Championship"]
SUPER_LIG = LEAGUES["Super Lig"]


# ---- _norm_league_token --------------------------------------------------

def test_norm_league_token_accent_fold():
    assert _norm_league_token("Süper Lig") == "super lig"
    assert _norm_league_token("Segunda División") == "segunda division"
    assert _norm_league_token("Premier League") == "premier league"
    assert _norm_league_token("") == ""


# ---- _nowgoal_league_matches: positives ----------------------------------

def test_epl_matches_premier_league_names():
    assert _nowgoal_league_matches("Premier League", EPL) is True
    assert _nowgoal_league_matches("English Premier League", EPL) is True


def test_epl_rejects_regional_variants():
    # regression: these names contain "premier league" as a substring but are
    # NOT the English Premier League
    assert _nowgoal_league_matches("National Premier Leagues NSW", EPL) is False
    assert _nowgoal_league_matches("National Premier Leagues Queensland", EPL) is False
    assert _nowgoal_league_matches("Russian Premier League", EPL) is False
    assert _nowgoal_league_matches("NPL Queensland Women", EPL) is False
    assert _nowgoal_league_matches("Australia Cup", EPL) is False


def test_ucl_matches_uefa_names():
    assert _nowgoal_league_matches("UEFA Champions League", UCL) is True
    assert _nowgoal_league_matches("Champions League", UCL) is True
    assert _nowgoal_league_matches("Champions League - Qualification", UCL) is True


def test_ucl_rejects_confederations_and_token_prefix():
    assert _nowgoal_league_matches("AFC Champions League 2 - Qualification", UCL) is False
    assert _nowgoal_league_matches("OFC Champions League", UCL) is False
    # token subset: "champions" must not match "Championship"
    assert _nowgoal_league_matches("Championship", UCL) is False


def test_uel_and_uecl_names():
    assert _nowgoal_league_matches("UEFA Europa League", UEL) is True
    assert _nowgoal_league_matches("Europa League", UEL) is True
    # Conference League is UECL, not UEL
    assert _nowgoal_league_matches("UEFA Europa Conference League", UEL) is False
    assert _nowgoal_league_matches("UEFA Europa Conference League", UECL) is True
    assert _nowgoal_league_matches("Conference League - Qualification", UECL) is True


def test_other_leagues():
    assert _nowgoal_league_matches("La Liga", LALIGA) is True
    assert _nowgoal_league_matches("LaLiga", LALIGA) is True
    assert _nowgoal_league_matches("Spanish La Liga", LALIGA) is True
    assert _nowgoal_league_matches("Serie A", SERIE_A) is True
    assert _nowgoal_league_matches("Italian Serie A", SERIE_A) is True
    assert _nowgoal_league_matches("Bundesliga", BUNDESLIGA) is True
    assert _nowgoal_league_matches("German Bundesliga", BUNDESLIGA) is True
    assert _nowgoal_league_matches("Championship", EFL) is True
    assert _nowgoal_league_matches("English Championship", EFL) is True
    assert _nowgoal_league_matches("Süper Lig", SUPER_LIG) is True
    assert _nowgoal_league_matches("Super Lig", SUPER_LIG) is True


def test_empty_and_garbage_league_name():
    assert _nowgoal_league_matches("", EPL) is False
    assert _nowgoal_league_matches(None, EPL) is False
    assert _nowgoal_league_matches("   ", EPL) is False


# ---- _nowgoal_fixtures_for_league ---------------------------------------

class _FakeNowgoal:
    def __init__(self, rows):
        self.rows = rows
        self.fetch_schedule = AsyncMock(return_value=rows)


def test_fixtures_fallback_filters_flood():
    """Only rows whose league name really is the target league survive."""
    rows = [
        {"match_id": "1", "home": "Arsenal", "away": "Chelsea",
         "home_id": "1", "away_id": "2", "kickoff": "2099-01-01T19:00:00Z",
         "status": "0", "league_id": "36", "league_name": "Premier League",
         "source": "nowgoal"},
        {"match_id": "2", "home": "Sydney FC (Youth)", "away": "St George City",
         "home_id": "3", "away_id": "4", "kickoff": "2099-01-01T19:00:00Z",
         "status": "0", "league_id": "77", "league_name": "National Premier Leagues NSW",
         "source": "nowgoal"},
        {"match_id": "3", "home": "Gazovik Orenburg", "away": "Lokomotiv Moscow",
         "home_id": "5", "away_id": "6", "kickoff": "2099-01-01T19:00:00Z",
         "status": "0", "league_id": "88", "league_name": "Russian Premier League",
         "source": "nowgoal"},
        {"match_id": "4", "home": "Peninsula Power", "away": "Brisbane City",
         "home_id": "7", "away_id": "8", "kickoff": "2099-01-01T19:00:00Z",
         "status": "0", "league_id": "99", "league_name": "NPL Queensland",
         "source": "nowgoal"},
    ]
    out = asyncio.run(_nowgoal_fixtures_for_league(
        _FakeNowgoal(rows), "2099-01-01", "EPL", EPL
    ))
    # rows are normalized to the football-data fixture shape (id, home/away)
    assert [m["id"] for m in out] == ["1"]
    assert out[0]["home"] == {"id": "1", "name": "Arsenal"}
    assert out[0]["source"] == "nowgoal"


def test_fixtures_fallback_empty_when_no_match():
    rows = [
        {"match_id": "9", "home": "A", "away": "B",
         "home_id": "1", "away_id": "2", "kickoff": "2099-01-01T19:00:00Z",
         "status": "0", "league_id": "99", "league_name": "Australia Cup",
         "source": "nowgoal"},
    ]
    out = asyncio.run(_nowgoal_fixtures_for_league(
        _FakeNowgoal(rows), "2099-01-01", "EPL", EPL
    ))
    assert out == []
