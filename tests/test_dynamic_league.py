"""D1/D2/D4 dynamic league discovery tests (2026-08-17).

- D1: dynamic_league_key / dynamic_league_meta are deterministic and never
  collide with registered keys.
- D4: _key_from_meta resolves registered displays to their registered key
  and unknown displays to a dyn: key (no more hardcoded 16-league list).
- D2: find_specific_match with an UNKNOWN league keyword detects the league
  from the fixture (flashscore homepage / livescore) and runs the pipeline
  with a dynamic key instead of failing with "liga tidak dikenal";
  league_key/league_meta short-circuit detection when the caller resolved
  the league already.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agents.football.analyse as analyse  # noqa: E402


# --------------------------------------------------------------------------
# D1 -- dynamic league key / meta
# --------------------------------------------------------------------------

def test_dynamic_league_key_deterministic_and_no_collide():
    from agents.football.league_resolver import dynamic_league_key, load_leagues

    k = dynamic_league_key("Copa del Rey")
    assert k == "dyn:copa-del-rey"
    assert k == dynamic_league_key("Copa del Rey")  # deterministic
    assert k not in load_leagues()  # never collides with registered keys
    assert dynamic_league_key("") == "dyn:unknown"


def test_dynamic_league_meta_shape():
    from agents.football.league_resolver import dynamic_league_meta

    m = dynamic_league_meta("Copa del Rey", country="Spain")
    assert m["dynamic"] is True
    assert m["odds_api_key"] is None  # The Odds API branch is skipped
    assert m["display"] == "Copa del Rey"
    assert m["country"] == "Spain"


# --------------------------------------------------------------------------
# D4 -- _key_from_meta
# --------------------------------------------------------------------------

def test_key_from_meta_registered_and_dynamic():
    from agents.football.multi_source import _key_from_meta

    # registered displays resolve to their registered key (existing behaviour
    # for the old hardcoded 16, plus the display form e.g. "La Liga")
    assert _key_from_meta({"display": "EPL"}) == "EPL"
    assert _key_from_meta({"display": "La Liga"}) == "LaLiga"
    assert _key_from_meta({"_league_key": "Segunda", "display": "X"}) == "Segunda"
    # unknown display -> deterministic dyn: key (not None, not "unknown")
    assert _key_from_meta({"display": "Copa del Rey"}) == "dyn:copa-del-rey"
    assert _key_from_meta({}) is None


# --------------------------------------------------------------------------
# D2 -- fixture-first detection in find_specific_match
# --------------------------------------------------------------------------

def _make_stats(fixture=None, pair=None):
    """``pair`` overrides the resolved team pair (2026-09-04).

    The default Barcelona/Real Madrid pair is fine for tests that query those
    clubs, but a test querying e.g. Pisa vs Empoli now trips the identity
    firewall (G-B post_resolve: "query 'Pisa' tetapi sumber memberi
    'Barcelona'") and dies before reaching the code under test. Pass ``pair``
    so the mocked resolver agrees with the query.
    """
    ms = AsyncMock()
    if pair is None:
        pair = (
            {"id": 1, "name": "Barcelona", "provider": "flashscore"},
            {"id": 2, "name": "Real Madrid", "provider": "flashscore"},
        )
    ms.search_teams_pair = AsyncMock(return_value=pair)
    ms.fetch_upcoming_fixture = AsyncMock(return_value=fixture if fixture is not None else {})
    ms.fetch_team_form = AsyncMock(
        return_value={"sequence": "W-W-D", "gf_avg": 1.5, "ga_avg": 0.8}
    )
    ms.fetch_h2h = AsyncMock(return_value={"wins": 1, "draws": 0, "losses": 2})
    ms.fetch_team_xg_history = AsyncMock(return_value=None)
    ms.fetch_flashscore_stats_for_match = AsyncMock(
        return_value={"xg_home": 1.2, "xg_away": 2.1, "possession_home": 41.0,
                      "possession_away": 59.0, "source": "flashscore"}
    )
    ms.fetch_flashscore_lineups_for_match = AsyncMock(return_value=None)
    ms.fetch_flashscore_event_context = AsyncMock(return_value=None)
    ms.fetch_league_standings = AsyncMock(return_value=None)
    ms.fetch_flashscore_match_info = AsyncMock(return_value=None)
    ms.fd = type("FD", (), {"rate_limit_warning": False})()
    # the runner always wires stats.cache = Cache(cache_dir) -- without it the
    # livescore feed scan would read an auto-created AsyncMock attribute
    ms.cache = _cache()
    return ms


def _cache():
    return type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})()


def _cfg():
    return {
        "cache_ttl_seconds": {"odds": 900},
        "outlier_threshold_pct": 5,
        "prediction_log": {"enabled": False},
    }


def test_find_specific_match_detects_dynamic_league_from_fixture():
    """Unknown league keyword -> league read from the fixture -> dyn: key."""
    ms = _make_stats()
    # flashscore homepage resolve returns a fixture with a competition title
    ms.fc = AsyncMock()
    ms.fc.available = True
    ms.fc.resolve_match = AsyncMock(
        return_value={
            "home": {"id": 1, "name": "Barcelona"},
            "away": {"id": 2, "name": "Real Madrid"},
            "competition": "Copa del Rey",
            "kickoff": "2026-08-20T19:00:00Z",
            "source": "flashscore",
        }
    )
    result = asyncio.run(analyse.find_specific_match(
        league_query="copa del rey",  # NOT a registered league keyword
        home_query="Barcelona",
        away_query="Real Madrid",
        cfg=_cfg(),
        odds=AsyncMock(),
        stats=ms,
        cache=_cache(),
    ))
    assert result is not None
    assert result.get("league_key") == "dyn:copa-del-rey"
    assert result.get("dynamic_league") is True
    assert result.get("league") == "Copa del Rey"


def test_find_specific_match_no_league_keyword_detects():
    """No league keyword at all -> detection from fixture, full pipeline."""
    ms = _make_stats()
    ms.fc = AsyncMock()
    ms.fc.available = True
    ms.fc.resolve_match = AsyncMock(
        return_value={
            "home": {"id": 1, "name": "Barcelona"},
            "away": {"id": 2, "name": "Real Madrid"},
            "competition": "Copa del Rey",
            "kickoff": "2026-08-20T19:00:00Z",
            "source": "flashscore",
        }
    )
    result = asyncio.run(analyse.find_specific_match(
        league_query=None,
        home_query="Barcelona",
        away_query="Real Madrid",
        cfg=_cfg(),
        odds=AsyncMock(),
        stats=ms,
        cache=_cache(),
    ))
    assert result is not None
    assert result.get("league_key") == "dyn:copa-del-rey"
    assert result.get("dynamic_league") is True


def test_find_specific_match_league_key_short_circuit():
    """Caller-resolved league_key/league_meta skip detection entirely."""
    ms = _make_stats()
    ms.fc = AsyncMock()  # would raise if detection were attempted
    ms.fc.available = True
    ms.fc.resolve_match = AsyncMock(side_effect=AssertionError("detection should be skipped"))
    from agents.football.league_resolver import dynamic_league_key, dynamic_league_meta

    result = asyncio.run(analyse.find_specific_match(
        league_query="dyn:copa-del-rey",
        home_query="Barcelona",
        away_query="Real Madrid",
        cfg=_cfg(),
        odds=AsyncMock(),
        stats=ms,
        cache=_cache(),
        league_key=dynamic_league_key("Copa del Rey"),
        league_meta=dynamic_league_meta("Copa del Rey"),
    ))
    assert result is not None
    assert result.get("league_key") == "dyn:copa-del-rey"


def test_find_specific_match_unknown_and_no_fixture_errors():
    """Unknown league AND no fixture -> honest error (unchanged behaviour)."""
    ms = _make_stats()
    ms.fc = AsyncMock()
    ms.fc.available = True
    ms.fc.resolve_match = AsyncMock(return_value=None)  # fixture not found
    result = asyncio.run(analyse.find_specific_match(
        league_query="xyz-nope",
        home_query="A",
        away_query="B",
        cfg=_cfg(),
        odds=AsyncMock(),
        stats=ms,
        cache=_cache(),
    ))
    assert "tidak dikenal" in (result.get("error") or "")


def test_find_specific_match_flashscore_without_competition_falls_to_livescore():
    """Regression (2026-08-17): flashscore's team-fixtures fallback returns
    the fixture WITHOUT a competition tag; the league must then be read from
    the LiveScore feed instead of failing with "tidak dikenal".

    Verified live with Pisa vs Empoli (Coppa Italia): resolve_match(None)
    reached the team-fixtures path (no competition in the dict), and
    detection died at ``if not competition: return None`` before ever
    consulting livescore.
    """
    ms = _make_stats(pair=(
        {"id": 1, "name": "Pisa", "provider": "flashscore"},
        {"id": 2, "name": "Empoli", "provider": "flashscore"},
    ))
    ms.fc = AsyncMock()
    ms.fc.available = True
    ms.fc.resolve_match = AsyncMock(
        return_value={
            "home": {"id": 1, "name": "Pisa"},
            "away": {"id": 2, "name": "Empoli"},
            "match_url": "https://example.invalid/m",
            "score": None,
            "source": "flashscore",
            # NOTE: no "competition" key -- team-fixtures rows lack it
        }
    )
    # LiveScore date feed carries the competition title. Kickoff is built
    # from today WIB so the scan window (today -> tomorrow) always matches.
    from agents.football.timeutil import wib_today_iso

    today = wib_today_iso().replace("-", "")
    ms.livescore = AsyncMock()
    ms.livescore.available = True
    feed_events = {
        "Stages": [
            {
                "CompN": "Coppa Italia", "Cnm": "Italy", "Ccd": "italy",
                "Scd": "coppaitalia",
                "Events": [
                    {
                        "Eid": "1806950",
                        "T1": [{"ID": "4806", "Nm": "Pisa"}],
                        "T2": [{"ID": "4642", "Nm": "Empoli"}],
                        "Eps": "NS",
                        "Esd": int(today + "160000"),
                    }
                ],
            }
        ]
    }
    ms.livescore.fetch_soccer_date = AsyncMock(return_value=feed_events)

    result = asyncio.run(analyse.find_specific_match(
        league_query=None,
        home_query="Pisa",
        away_query="Empoli",
        cfg=_cfg(),
        odds=AsyncMock(),
        stats=ms,
        cache=_cache(),
    ))
    assert result is not None
    assert result.get("league_key") == "dyn:coppa-italia"
    assert result.get("dynamic_league") is True
    assert result.get("league") == "Coppa Italia"
