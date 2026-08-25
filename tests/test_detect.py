"""Unit tests for the runner-side league auto-detect (detect_match.py).

The `detect` command resolves which registered league hosts a free-typed
`home vs away`: football-data scheduled scan first (one /v4/matches call),
flashscore homepage fallback second. Tests mock the data providers and pin
the name-matching + code->league mapping + both fallback branches.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import detect_match as dm  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _stats(*, scheduled=None, homepage=None, ts=None):
    async def _fd(*a, **k):
        return scheduled

    async def _hp():
        return homepage

    return types.SimpleNamespace(
        fd=types.SimpleNamespace(fetch_scheduled_matches_by_date=_fd),
        fetch_homepage_matches=_hp,
        ts=ts or types.SimpleNamespace(
            search_team=lambda *a, **k: asyncio.sleep(0) or None,
            fetch_next_matches=lambda *a, **k: asyncio.sleep(0) or None,
        ),
    )


def _cache():
    return types.SimpleNamespace(get=lambda *a, **k: None, set=lambda *a, **k: None)


def test_code_to_league_key_mapping():
    m = dm._code_to_league_key()
    assert m["PL"] == "EPL"
    assert m["CL"] == "UCL"
    assert m["EL"] == "UEL"
    assert m["EC"] == "UECL"
    assert "PD" in m  # LaLiga


def test_find_pair_prefers_exact_over_loose():
    rows = [
        {"home": "Real Sociedad", "away": "Barcelona", "competition": "PD"},
        {"home": "Real Madrid", "away": "Sevilla", "competition": "PD"},
    ]
    found = dm._find_pair_in_rows(rows, "real madrid", "sevilla")
    assert found is not None
    assert found["home"] == "Real Madrid"


def test_find_pair_swapped_sides():
    rows = [{"home": "Chelsea", "away": "Manchester United", "competition": "PL"}]
    found = dm._find_pair_in_rows(rows, "manchester united", "chelsea")
    assert found is not None
    assert found["home"] == "Chelsea"


def test_find_pair_typo_tolerance():
    """2026-08-17 (Gnistan vs Ilven -> Ilves): a 1-char user typo must not
    hard-fail detect -- the typo tier returns the unique row and the bot
    then runs the FULL analysis with the corrected name. Close-but-different
    clubs must still be rejected (unambiguous guard)."""
    rows = [
        {"home": "Gnistan", "away": "Ilves", "competition": "Veikkausliiga"},
        {"home": "Real Betis", "away": "Getafe", "competition": "PD"},
    ]
    # typo on the away side only -> still found, corrected name wins
    found = dm._find_pair_in_rows(rows, "gnistan", "ilven")
    assert found is not None
    assert found["away"] == "Ilves"
    # a genuinely different club must not typo-match
    assert dm._find_pair_in_rows(rows, "gnistan", "betis") is None
    # too-short names are never typo-matched
    assert dm._typo_match("abc", "abd") is False


def test_detect_football_data_path_with_alias():
    """'man utd' resolves via teams.json alias -> EPL match found."""
    stats = _stats(scheduled=[
        {"home": "Manchester United FC", "away": "Chelsea FC",
         "competition": "PL", "kickoff": "2026-08-16T14:00:00Z"},
        {"home": "Other", "away": "Game", "competition": "SA",
         "kickoff": "2026-08-16T16:00:00Z"},
    ])
    out = _run(dm.detect_league_match(home="man utd", away="chelsea", stats=stats, cache=_cache()))
    assert out["found"] is True
    assert out["league"] == "EPL"
    assert out["source"] == "football_data"
    assert out["home"] == "Manchester United FC"


def test_detect_football_data_not_matched_falls_back_to_homepage():
    stats = _stats(
        scheduled=[],
        homepage=[
            {"home": {"name": "Tobol"}, "away": {"name": "Partizan"},
             "competition": "Conference League - Qualification",
             "date_text": "21:00", "status": "scheduled"},
        ],
    )
    out = _run(dm.detect_league_match(home="tobol", away="partizan", stats=stats, cache=_cache()))
    assert out["found"] is True
    assert out["league"] == "UECL"
    assert out["source"] == "flashscore"
    assert out["competition"] == "Conference League - Qualification"


def test_detect_homepage_unregistered_competition():
    stats = _stats(
        scheduled=[],
        homepage=[
            {"home": {"name": "Manchester United"}, "away": {"name": "Leeds"},
             "competition": "Club Friendly", "date_text": "02:00",
             "status": "scheduled"},
        ],
    )
    out = _run(dm.detect_league_match(home="man utd", away="leeds", stats=stats, cache=_cache()))
    assert out["found"] is True
    assert out.get("registered") is False
    assert out["competition"] == "Club Friendly"
    assert "league" not in out


def test_detect_thesportsdb_league_for_far_match():
    """A registered-league match the global feed / homepage miss -> detected
    via thesportsdb next-fixtures (works weeks ahead of kickoff)."""
    async def _search(name):
        return {"idTeam": "42"}

    async def _next_matches(tid, limit=10):
        return [
            {"strHomeTeam": "Manchester United", "strAwayTeam": "Leeds United",
             "strLeague": "English Premier League",
             "dateEvent": "2026-08-23", "strTime": "15:00"},
        ]

    stats = _stats(
        scheduled=[],
        homepage=[],
        ts=types.SimpleNamespace(
            search_team=_search,
            fetch_next_matches=_next_matches,
        ),
    )
    out = _run(dm.detect_league_match(home="man utd", away="leeds", stats=stats, cache=_cache()))
    assert out["found"] is True
    assert out["league"] == "EPL"
    assert out["source"] == "thesportsdb"
    assert "2026-08-23" in (out["kickoff"] or "")


def test_detect_thesportsdb_la_liga_2_resolves_segunda_not_laliga():
    """Fix 2026-08-17: thesportsdb competition resolution uses the prefix /
    longest-first ``competition_league_key``, so "LaLiga2" maps to Segunda,
    never to LaLiga (the loose substring path mis-resolved it)."""
    async def _search(name):
        return {"idTeam": "42"}

    async def _next_matches(tid, limit=10):
        return [
            {"strHomeTeam": "Las Palmas", "strAwayTeam": "Albacete",
             "strLeague": "LaLiga2",
             "dateEvent": "2026-08-23", "strTime": "19:30"},
        ]

    stats = _stats(
        scheduled=[],
        homepage=[],
        ts=types.SimpleNamespace(
            search_team=_search,
            fetch_next_matches=_next_matches,
        ),
    )
    out = _run(dm.detect_league_match(home="las palmas", away="albacete", stats=stats, cache=_cache()))
    assert out["found"] is True
    assert out["league"] == "Segunda"
    assert out["source"] == "thesportsdb"


def test_detect_thesportsdb_trophee_never_resolves_ucl():
    """Fix 2026-08-17: "Trophée des Champions" (a cup) must NOT resolve onto
    UCL via a substring match -- the prefix resolver returns None and the
    detect falls through to the homepage path."""
    async def _search(name):
        return {"idTeam": "42"}

    async def _next_matches(tid, limit=10):
        return [
            {"strHomeTeam": "PSG", "strAwayTeam": "Monaco",
             "strLeague": "Trophée des Champions",
             "dateEvent": "2026-08-23", "strTime": "19:30"},
        ]

    stats = _stats(
        scheduled=[],
        homepage=[],
        ts=types.SimpleNamespace(
            search_team=_search,
            fetch_next_matches=_next_matches,
        ),
    )
    out = _run(dm.detect_league_match(home="psg", away="monaco", stats=stats, cache=_cache()))
    # Not a registered league -> not an analyzable registered match.
    assert out == {"found": False}


def test_detect_nothing_matched():
    stats = _stats(scheduled=[], homepage=[])
    out = _run(dm.detect_league_match(home="foo", away="bar", stats=stats, cache=_cache()))
    assert out == {"found": False}


def test_detect_skips_finished_homepage_rows():
    stats = _stats(
        scheduled=[],
        homepage=[
            {"home": {"name": "A"}, "away": {"name": "B"},
             "competition": "Premier League", "date_text": "12:00",
             "status": "finished"},
        ],
    )
    out = _run(dm.detect_league_match(home="a", away="b", stats=stats, cache=_cache()))
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
