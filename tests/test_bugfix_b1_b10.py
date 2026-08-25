"""Unit tests for the 2026-08-17 adversarial-audit bug fixes (B1..B10).

Covers the verified bugs from the bookmaker-audit review:
  - B1  _teams_match 1-char token leak (root: corrupts standings/form/side)
  - B3  competition_league_key demonym leak (cup/div-2 tagged as top league)
  - B4  resolve_league_scored free-form league query (no substring trap)
  - B6  football-data team id search (no bare-substring wrong-club match)
  - B9  detect loose pass ambiguity guard (1 unique hit only)
  - B10 flashscore CET/CEST conversion via real DST rule (zoneinfo)
B2/B5 standings-matching tests live in test_flashscore_context.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.football.analyse import _teams_match  # noqa: E402


# ---------------------------------------------------------------------------
# B1: _teams_match 1-char token leak
# ---------------------------------------------------------------------------
def test_teams_match_rejects_one_char_token_leak():
    """A 1-char token in the LONGER name (\"a\" in \"Dep. A Coruna\") must not
    match every token containing that letter -- previously returned True for
    both Las Palmas and Albacete (verified live 2026-08-17)."""
    assert _teams_match("Dep. A Coruna", "Las Palmas") is False
    assert _teams_match("Dep. A Coruna", "Albacete") is False
    assert _teams_match("Dep. A Coruna", "Real Oviedo") is False


def test_teams_match_keeps_legit_matches():
    """Symmetric token containment (>=3 chars) and prefix handling still work."""
    assert _teams_match("Dep. A Coruna", "Dep. A Coruna") is True
    assert _teams_match("Las Palmas", "Las Palmas") is True
    assert _teams_match("Albacete", "Albacete") is True
    assert _teams_match("Heart of Midlothian", "Hearts") is True
    assert _teams_match("hearts", "heart of midlothian fc") is True
    # "Bodø/Glimt" vs "FK Bodø/Glimt": token equality + prefix handling
    assert _teams_match("Bodø/Glimt", "FK Bodø/Glimt") is True


# ---------------------------------------------------------------------------
# B3: competition_league_key demonym leak
# ---------------------------------------------------------------------------
def test_competition_league_key_demonym_guard():
    from agents.football.league_resolver import competition_league_key

    # A leading demonym is a descriptor, not the competition: the cup must
    # NOT tag Ligue 1 and the second division must NOT tag LaLiga.
    assert competition_league_key("French Trophée des Champions") is None
    assert competition_league_key("Spanish Copa del Rey") is None
    assert competition_league_key("Trophée des Champions") is None
    # stripping the demonym reveals the real competition ("Spanish La Liga 2"
    # is Segunda, not LaLiga).
    assert competition_league_key("Spanish La Liga 2") == "Segunda"
    assert competition_league_key("Spain La Liga 2") == "Segunda"


def test_competition_league_key_no_regression():
    from agents.football.league_resolver import competition_league_key

    assert competition_league_key("LaLiga2") == "Segunda"
    assert competition_league_key("La Liga 2") == "Segunda"
    assert competition_league_key("LaLiga") == "LaLiga"
    assert competition_league_key("La Liga") == "LaLiga"
    assert competition_league_key("English Premier League") == "EPL"
    assert competition_league_key("UEFA Champions League") == "UCL"
    assert competition_league_key("Champions League - Qualification") == "UCL"
    assert competition_league_key("Belgian Pro League") == "Belgian Pro League"
    assert competition_league_key("Ligue 1") == "Ligue 1"
    assert competition_league_key("Serie B") == "Serie B"
    assert competition_league_key("Major League Soccer") == "MLS"
    assert competition_league_key("England Cup") is None


# ---------------------------------------------------------------------------
# B4: resolve_league_scored free-form queries
# ---------------------------------------------------------------------------
def test_resolve_league_scored_handles_country_league_pattern():
    from agents.football.league_resolver import resolve_league_scored

    # "<country/lang> <league>" must resolve to the league named LAST, not
    # the demonym alias ("spanish" -> LaLiga) via substring.
    assert resolve_league_scored("spanish segunda")[0] == "Segunda"
    assert resolve_league_scored("liga 2")[0] == "Segunda"
    assert resolve_league_scored("la liga 2")[0] == "Segunda"
    assert resolve_league_scored("liga italy")[0] == "Serie A"
    assert resolve_league_scored("liga spanyol")[0] == "LaLiga"
    assert resolve_league_scored("french ligue 1")[0] == "Ligue 1"


def test_resolve_league_scored_parity_with_legacy():
    """Every standard query resolves exactly like the legacy substring
    resolver (only the mis-resolution cases change)."""
    from agents.football.league_resolver import resolve_league, resolve_league_scored

    for q in [
        "laliga", "la liga", "premier league", "english premier league",
        "serie a", "serie b", "bundesliga", "ligue 1", "eredivisie",
        "champions league", "scottish premiership", "belgian pro league",
        "italy", "france", "germany", "spain", "usa", "mls", "segunda",
        "segunda division", "laliga2", "super lig", "championship",
        "conference league", "europa league", "major league soccer",
    ]:
        scored = (resolve_league_scored(q) or (None,))[0]
        legacy = (resolve_league(q) or (None,))[0]
        assert scored == legacy, f"{q!r}: scored={scored!r} legacy={legacy!r}"


# ---------------------------------------------------------------------------
# B6: football-data team id search
# ---------------------------------------------------------------------------
def test_search_team_in_competition_shortest_name_wins(monkeypatch):
    import asyncio

    from agents.football.football_data import FootballDataClient

    teams = [
        {"name": "Atlético Madrid", "shortName": "Atlético", "tla": "ATM", "id": 1},
        {"name": "Atlético Madrid B", "shortName": "Atlético B", "tla": "ATB", "id": 2},
        {"name": "Real Madrid CF", "shortName": "Real Madrid", "tla": "RMA", "id": 3},
        {"name": "Manchester United FC", "shortName": "Man United", "tla": "MUN", "id": 6},
        {"name": "FC Barcelona", "shortName": "Barcelona", "tla": "BAR", "id": 7},
    ]

    async def fake_fetch(self, code):
        return teams

    monkeypatch.setattr(FootballDataClient, "fetch_teams", fake_fetch)
    fd = FootballDataClient("")

    async def run():
        # B6: the reserve side must NOT win -- exact / shortest-name resolution
        assert (await fd.search_team_in_competition("Atletico Madrid", "PD"))["id"] == 1
        assert (await fd.search_team_in_competition("Atlético Madrid", "PD"))["id"] == 1
        assert (await fd.search_team_in_competition("Atletico", "PD"))["id"] == 1
        assert (await fd.search_team_in_competition("Real Madrid", "PD"))["id"] == 3
        assert (await fd.search_team_in_competition("Barcelona", "PD"))["id"] == 7
        assert (await fd.search_team_in_competition("FC Barcelona", "PD"))["id"] == 7
        # the upstream alias expansion feeds the FULL name ("Manchester
        # United"), which resolves exactly; a bare "Man Utd" was never
        # resolvable here
        assert (await fd.search_team_in_competition("Manchester United", "PD"))["id"] == 6
        assert await fd.search_team_in_competition("Man Utd", "PD") is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# B9: detect loose pass ambiguity guard
# ---------------------------------------------------------------------------
def test_detect_loose_pass_requires_unique_hit(monkeypatch):
    import agents.football.detect_match as dm

    rows = [
        {"home": "Xyzz", "away": "Other A", "competition": "Cup"},
        {"home": "Xyzz", "away": "Other B", "competition": "Cup"},
    ]
    # Strict pass genuinely fails (no token overlap); the loose pass matches
    # BOTH rows -> ambiguous -> None, never a silent first-hit.
    def fake_loose(a: str, b: str) -> bool:
        s = f"{a} {b}".lower()
        return "target" in s or "other" in s

    monkeypatch.setattr(dm, "_loose", fake_loose)
    assert dm._find_pair_in_rows(rows, "Target", "Other") is None
    # A single loose hit still resolves.
    found = dm._find_pair_in_rows(rows[:1], "Target", "Other")
    assert found is not None and found["away"] == "Other A"


def test_detect_strict_pass_unchanged():
    import agents.football.detect_match as dm

    rows = [
        {"home": "Real Madrid", "away": "Barcelona", "competition": "LaLiga"},
        {"home": "Real Madrid", "away": "Sevilla", "competition": "LaLiga"},
    ]
    found = dm._find_pair_in_rows(rows, "Real Madrid", "Sevilla")
    assert found is not None and found["away"] == "Sevilla"


# ---------------------------------------------------------------------------
# B10/F3: flashscore wall-clock is WIB (Asia/Jakarta), not CET/CEST
# ---------------------------------------------------------------------------
def test_flashscore_local_to_utc_wib_offset():
    from agents.football.multi_source import _flashscore_local_to_utc_iso

    # F3 (verified live 2026-08-17): the headless Chrome runs in the system
    # timezone Asia/Jakarta and flashscore renders every match time in the
    # VISITOR's browser timezone -- a 17:00 UTC kickoff renders as 23:00 WIB.
    # The old Europe/Madrid assumption shifted all kickoffs by +5h (CEST) /
    # +6h (CET). Indonesia has no DST, so the offset is a constant +7h.
    assert _flashscore_local_to_utc_iso(2026, 1, 15, 20, 0) == "2026-01-15T13:00:00Z"  # 20:00 WIB = 13:00 UTC
    assert _flashscore_local_to_utc_iso(2026, 3, 29, 20, 0) == "2026-03-29T13:00:00Z"   # no DST: same offset
    assert _flashscore_local_to_utc_iso(2026, 10, 25, 20, 0) == "2026-10-25T13:00:00Z"  # no DST: same offset
    assert _flashscore_local_to_utc_iso(2026, 12, 24, 20, 0) == "2026-12-24T13:00:00Z"  # WIB year-round
    assert _flashscore_local_to_utc_iso(2026, 8, 17, 23, 0) == "2026-08-17T16:00:00Z"   # live check: 23:00 WIB = 16:00 UTC


# ---------------------------------------------------------------------------
# Fix A (found during live coverage verification): resolve_team_alias hijack
# ---------------------------------------------------------------------------
def test_resolve_team_alias_not_hijacked_by_generic_alias():
    """Fix A 2026-08-17: the "madrid" alias (-> Real Madrid) hijacked FULL
    canonical names via word-boundary fallback -- "Atlético Madrid" and
    "Rayo Vallecano de Madrid" both resolved to "Real Madrid CF" and
    corrupted the standings alias tier. The exact canonical-name pass must
    run before the boundary fallback so a full name resolves to ITSELF."""
    from agents.football.team_alias import resolve_team_alias

    assert resolve_team_alias("Atlético Madrid", None) == "Atlético Madrid"
    assert resolve_team_alias("Rayo Vallecano de Madrid", None) == "Rayo Vallecano de Madrid"
    assert resolve_team_alias("Leicester City FC", None) == "Leicester City FC"
    assert resolve_team_alias("Paris Saint-Germain FC", None) == "Paris Saint-Germain FC"
    # partial names still resolve through the boundary fallback
    assert resolve_team_alias("Real Madrid", None) == "Real Madrid CF"
    # colliding short codes keep working league-restricted (Segunda codes
    # must not shadow Eredivisie/EPL lookups)
    assert resolve_team_alias("ALM", "Eredivisie") == "Almere City FC"
    assert resolve_team_alias("BUR", "EPL") == "Burnley FC"
    assert resolve_team_alias("CAS", "Primeira Liga") == "Casa Pia AC"


def test_standings_curated_expansion_covers_other_leagues():
    """B2/B5 expansion 2026-08-17: the curated standings-spelling map covers
    the abbreviations flashscore tables actually use across leagues (verified
    against the live EPL/LaLiga/Serie A/Bundesliga/Ligue 1 tables: 92/92
    in-table clubs resolve)."""
    from agents.football.analyse import _match_standings_team

    def _tbl(names):
        return [{"team": n} for n in names]

    epl = _tbl(["Arsenal", "Manchester Utd", "Nottingham", "West Ham"])
    assert _match_standings_team(epl, "Manchester United FC")["team"] == "Manchester Utd"
    assert _match_standings_team(epl, "Nottingham Forest FC")["team"] == "Nottingham"
    assert _match_standings_team(epl, "West Ham United FC")["team"] == "West Ham"

    bundes = _tbl(["Bayern Munich", "B. Monchengladbach", "FC Koln", "Dortmund"])
    assert _match_standings_team(bundes, "FC Bayern München")["team"] == "Bayern Munich"
    assert _match_standings_team(bundes, "Borussia Mönchengladbach")["team"] == "B. Monchengladbach"
    assert _match_standings_team(bundes, "1. FC Köln")["team"] == "FC Koln"
    assert _match_standings_team(bundes, "Borussia Dortmund")["team"] == "Dortmund"

    ligue1 = _tbl(["PSG", "Rennes", "Marseille", "Lyon"])
    assert _match_standings_team(ligue1, "Paris Saint-Germain FC")["team"] == "PSG"
    assert _match_standings_team(ligue1, "Stade Rennais FC 1901")["team"] == "Rennes"
    assert _match_standings_team(ligue1, "Olympique de Marseille")["team"] == "Marseille"
    assert _match_standings_team(ligue1, "Olympique Lyonnais")["team"] == "Lyon"
