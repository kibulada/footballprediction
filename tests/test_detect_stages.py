"""Tests for the new detect stages added 2026-08-18 to ``detect_league_match``.

The detect engine now scans, in order, football-data (registered only) ->
livescore date feed (any competition, cached 15 min) -> flashscore homepage
(today only, curated) -> thesportsdb (next-fixtures) -> flashscore
team-fixtures (browser fallback, budget-guarded). The two new stages close
the gap that produced `Liga tidak dikenali` for:

  - ASEAN / AFF / friendly playoffs that never reach football-data
  - club friendlies several days out that the homepage omits
  - ASEAN Championship whose title doesn't match any registered alias

These tests stub every provider so each stage can be exercised
independently without touching the network.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import detect_match  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _stats(*, fd_rows=None, fc=None, ls_client=None, ts_client=None):
    """Build a minimal `stats` namespace with the four providers detect
    consults: football-data (.fd), flashscore (.fc), livescore
    (.livescore + .cache), thesportsdb (via stats.ts on the same object).
    """

    async def fetch_scheduled_matches_by_date(start, end):
        return fd_rows

    fd = types.SimpleNamespace(fetch_scheduled_matches_by_date=fetch_scheduled_matches_by_date)

    cache = types.SimpleNamespace(get=lambda *a, **k: None, set=lambda *a, **k: None)
    stats = types.SimpleNamespace(
        fd=fd,
        fc=fc,
        livescore=ls_client,
        cache=cache,
        ts=ts_client,
        fetch_homepage_matches=lambda: None,
    )
    return stats


def test_stage_livescore_feed_finds_unregistered_competition():
    """Livescore feed returns a row tagged with a non-registered
    competition title -> detect returns `found: True, registered: False`
    (D2 path -- the bot runs analyse without a league keyword).
    """
    # Tomorrow 20:00 WIB -> Esd = the corresponding UTC YYYYMMDDHHMMSS so
    # `_search_livescore_any` recognises the row as today/tomorrow WIB.
    from datetime import datetime, timedelta, timezone
    WIB = timezone(timedelta(hours=7))
    esd = (datetime.now(WIB) + timedelta(days=1)).replace(
        hour=20, minute=0, second=0, microsecond=0
    ).astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    payload = {
        "Stages": [
            {
                "CompN": "ASEAN Championship - Play Offs - Semi-finals",
                "Cnm": "World",
                "Events": [
                    {
                        "T1": [{"Nm": "Thailand"}],
                        "T2": [{"Nm": "Singapore"}],
                        "Esd": esd,
                        "Eps": "20:00",
                    }
                ],
            }
        ]
    }

    class _LS:
        available = True

        def __init__(self):
            self.calls = 0

        async def fetch_soccer_date(self, *a, **k):
            self.calls += 1
            return payload

    ls = _LS()
    stats = _stats(ls_client=ls)
    out = _run(detect_match.detect_league_match(
        home="Thailand", away="Singapore", stats=stats, cache=stats.cache,
    ))
    assert out["found"] is True, out
    assert out.get("registered") is False
    assert out["competition"].startswith("ASEAN Championship")
    assert out["source"] == "livescore"


def test_stage_livescore_feed_finds_registered_league():
    """Livescore feed row with competition that maps to a registered league
    (via ``competition_league_key``) -> detect returns the league key, so
    the bot can run analyse with --league Liga 1 (full quality).
    """
    from datetime import datetime, timedelta, timezone
    WIB = timezone(timedelta(hours=7))
    esd = (datetime.now(WIB) + timedelta(days=1)).replace(
        hour=19, minute=30, second=0, microsecond=0
    ).astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    payload = {
        "Stages": [
            {
                "CompN": "Liga 1",
                "Cnm": "Indonesia",
                "Events": [
                    {
                        "T1": [{"Nm": "Persib Bandung"}],
                        "T2": [{"Nm": "Persija Jakarta"}],
                        "Esd": esd,
                        "Eps": "19:30",
                    }
                ],
            }
        ]
    }

    class _LS:
        available = True

        async def fetch_soccer_date(self, *a, **k):
            return payload

    stats = _stats(ls_client=_LS())
    out = _run(detect_match.detect_league_match(
        home="Persib Bandung", away="Persija Jakarta", stats=stats, cache=stats.cache,
    ))
    assert out["found"] is True, out
    assert out.get("league") == "Liga 1"
    assert out["source"] == "livescore"


def test_stage_flashscore_team_fixtures_finds_unregistered():
    """When football-data / livescore / homepage / thesportsdb all miss,
    the flashscore team-fixtures fallback should kick in and return a
    fixture with its real competition tag ("Club Friendly" for a friendly,
    a cup title for a knockout, etc.).
    """

    class _FC:
        available = True

        def __init__(self):
            self.calls = 0

        async def resolve_match(self, league_key, home, away):
            self.calls += 1
            return {
                "home": {"slug": "manchester-united", "id": "x1", "name": "Manchester United"},
                "away": {"slug": "leeds", "id": "y1", "name": "Leeds"},
                "match_url": "https://flashscore/match/x",
                "date_text": "Tomorrow 02:00",
                "competition": "Club Friendly",
            }

    fc = _FC()
    stats = _stats(fc=fc)
    out = _run(detect_match.detect_league_match(
        home="Manchester United", away="Leeds", stats=stats, cache=stats.cache,
    ))
    assert out["found"] is True
    assert out.get("registered") is False
    assert out["competition"] == "Club Friendly"
    assert out["source"] == "flashscore"
    assert fc.calls == 1


def test_stage_flashscore_team_fixtures_finds_registered():
    """Team-fixtures fallback for a Liga 1 fixture: competition="Liga 1"
    is registered, so detect returns the league key (full-quality
    analyse via --league).
    """

    class _FC:
        available = True

        async def resolve_match(self, league_key, home, away):
            return {
                "home": {"slug": "persib", "id": "p1", "name": "Persib Bandung"},
                "away": {"slug": "bali-united", "id": "b1", "name": "Bali United"},
                "match_url": "https://flashscore/match/y",
                "date_text": "Tomorrow 19:30",
                "competition": "Liga 1",
            }

    fc = _FC()
    stats = _stats(fc=fc)
    out = _run(detect_match.detect_league_match(
        home="Persib Bandung", away="Bali United", stats=stats, cache=stats.cache,
    ))
    assert out["found"] is True
    assert out.get("league") == "Liga 1"
    assert out["source"] == "flashscore"


def test_stage_flashscore_team_fixtures_swallows_failure():
    """When the flashscore client is None or unavailable, the detect
    engine must NOT raise -- it falls through to `found: False`.
    """
    stats = _stats(fc=None)
    out = _run(detect_match.detect_league_match(
        home="A", away="B", stats=stats, cache=stats.cache,
    ))
    assert out == {"found": False}


def test_fixture_to_result_registered_returns_league_key():
    """Pure unit test for the new `_fixture_to_result` helper."""
    found = {
        "home": "Thailand",
        "away": "Singapore",
        "competition": "Liga 1",  # not actually true but tests the mapping
        "kickoff": "20:00",
    }
    out = detect_match._fixture_to_result(found, "livescore")
    assert out["found"] is True
    assert out["league"] == "Liga 1"
    assert out["source"] == "livescore"


def test_fixture_to_result_unregistered_returns_info_only():
    found = {
        "home": "Thailand",
        "away": "Singapore",
        "competition": "ASEAN Championship - Semi-finals",
        "kickoff": "20:00",
    }
    out = detect_match._fixture_to_result(found, "livescore")
    assert out["found"] is True
    assert out.get("registered") is False
    assert out["competition"].startswith("ASEAN")
    assert "league" not in out


def test_fixture_to_result_empty_competition_is_not_found():
    """Defensive: livescore/flashscore rows with no competition title must
    NOT be treated as `found: True` (would leak an unlabelled fixture
    into D2 with no league at all)."""
    out = detect_match._fixture_to_result(
        {"home": "A", "away": "B", "competition": "", "kickoff": None}, "livescore"
    )
    assert out == {"found": False}


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
