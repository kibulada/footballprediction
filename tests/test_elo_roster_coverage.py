"""P4 (2026-09-02): every canonical club in teams.json must resolve in the
live Elo store -- alias whack-a-mole replaced by one automated check.

Background: Lille (store "Lille OSC" 2027), Genk ("KRC Genk" 1879), Nice and
Brighton were all modelled on the 1500 prior during 26-31 Aug because the
lookup used the live display name. 81 of 147 settled matches that week had
at least one side on the prior; the model's 1X2 Brier on those was 0.534 vs
the market's 0.457. The engine now resolves the canonical teams.json name
first (``MatchContext.home_elo_names``); this test keeps that path honest:
a club that is in the roster but not in the store is a data gap that must be
visible, not a silent 1500.

Leagues the store does not cover at all (elofootball.com is Europe-only;
those leagues run on the market-Elo fallback) are listed, not asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.elo import EloModel  # noqa: E402
from agents.football.team_alias import load_teams  # noqa: E402

STORE = ROOT / "cache" / "football" / "elo.json"
MIN_ROSTER = 10

# Zero-coverage leagues (store is Europe-only): reported, never asserted.
UNCOVERED_LEAGUES = {"Saudi Pro League", "MLS", "Liga 1", "K-League", "J1 League"}

# Known gaps in the store itself (club absent from elofootball.com's export),
# each one a data item, not an alias bug. Remove an entry once the store has
# the club so the test starts guarding it.
KNOWN_MISSING = {
    "Primeira Liga": {"Boavista FC"},
    "UECL": {"KF Tirana", "Indiane"},
}


def _rosters() -> dict[str, list[str]]:
    teams = load_teams()
    return {
        league: sorted(set(aliases.values()))
        for league, aliases in teams.items()
        if len(set(aliases.values())) >= MIN_ROSTER
    }


@pytest.mark.skipif(not STORE.exists(), reason="live Elo store not present")
def test_every_covered_league_roster_resolves_in_elo_store():
    elo = EloModel(path=STORE)
    rosters = _rosters()
    assert rosters, "teams.json has no league with >= 10 clubs"
    failures: dict[str, list[str]] = {}
    for league, clubs in rosters.items():
        if league in UNCOVERED_LEAGUES:
            continue
        allowed = KNOWN_MISSING.get(league, set())
        missing = [c for c in clubs if c not in allowed and elo.resolve(c) is None]
        if missing:
            failures[league] = missing
    assert not failures, (
        "canonical clubs without an Elo rating (add the store entry or an alias in elo._EXTRA_ALIASES): "
        + "; ".join(f"{lg}: {', '.join(m)}" for lg, m in sorted(failures.items()))
    )


@pytest.mark.skipif(not STORE.exists(), reason="live Elo store not present")
def test_known_missing_list_is_still_accurate():
    """A club listed as missing that now resolves should be removed from the
    allowlist so the guard covers it again."""
    elo = EloModel(path=STORE)
    stale = [
        f"{lg}: {c}" for lg, clubs in KNOWN_MISSING.items() for c in clubs if elo.resolve(c) is not None
    ]
    assert not stale, f"remove from KNOWN_MISSING (now resolves): {stale}"


@pytest.mark.skipif(not STORE.exists(), reason="live Elo store not present")
def test_canonical_first_lookup_beats_display_name():
    """The incident pairs: display name alone fails the K2 guard, the
    canonical name resolves, and the tuple lookup takes the canonical."""
    elo = EloModel(path=STORE)
    for display, canonical in (("Lille", "Lille OSC"), ("Genk", "KRC Genk")):
        assert elo.resolve(canonical) is not None, canonical
        assert elo.resolve((canonical, display)) == elo.resolve(canonical)
        assert elo.rating((canonical, display)) != 1500.0
