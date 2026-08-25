"""Unit tests for the flashscore lineups parser (parse_lineups_page).

The parser is pure (dict in -> dict out) so it is tested without a browser;
the DOM scrape itself is exercised by the live EPL/UCL flow tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.flashscore import (  # noqa: E402
    FlashscoreBrowserClient,
    _find_pair_in_rows,
    _parse_h2h_section,
    _squash_variants,
    parse_lineups_page,
)


def test_parse_h2h_section_layout_rows():
    """F5 (2026-08-17): the current flashscore layout renders each direct
    meeting as a column of lines [date, comp, home, away, hg, ag] -- the old
    single-line parser matched nothing. Home-side perspective: the first
    team listed wins when its score is higher."""
    section = """\
HEAD-TO-HEAD MATCHES
17.08.26
ACN
Cameroon W
Malawi W
3
0
02.08.26
ACN
Cameroon W
Malawi W
1
1
PINNED LEAGUES
Premier League
"""
    out = _parse_h2h_section(section, "Cameroon W", "Malawi W")
    assert out is not None
    assert out["wins"] == 1 and out["draws"] == 1 and out["losses"] == 0
    assert out["count"] == 2
    assert out["source"] == "flashscore_h2h"


def test_parse_h2h_section_reversed_roles():
    """The pair may appear in either order in the section; roles are
    assigned by matching the requested home/away names, and the win side
    follows the first listed team's score."""
    section = """\
HEAD-TO-HEAD MATCHES
10.08.26
EPL
Arsenal
Chelsea
0
2
"""
    # Queried home=Arsenal, away=Chelsea; Arsenal lost 0-2 -> loss.
    out = _parse_h2h_section(section, "Arsenal", "Chelsea")
    assert out is not None
    assert out["wins"] == 0 and out["draws"] == 0 and out["losses"] == 1
    # Same section, reversed query -> Chelsea is home, Chelsea won 2-0.
    out2 = _parse_h2h_section(section, "Chelsea", "Arsenal")
    assert out2 is not None
    assert out2["wins"] == 1 and out2["draws"] == 0 and out2["losses"] == 0


def test_parse_h2h_section_no_meetings_returns_none():
    assert _parse_h2h_section("HEAD-TO-HEAD MATCHES\nPINNED LEAGUES", "A", "B") is None
    assert _parse_h2h_section("", "A", "B") is None


def test_parse_h2h_section_ignores_other_team_pair():
    """Lines for a DIFFERENT pair (other h2h rows / pinned leagues) must not
    be counted -- the run must match the queried home/away names exactly."""
    section = """\
HEAD-TO-HEAD MATCHES
17.08.26
ACN
Zambia W
Ghana W
2
1
01.08.26
EPL
Arsenal
Chelsea
1
0
PINNED LEAGUES
"""
    out = _parse_h2h_section(section, "Arsenal", "Chelsea")
    assert out is not None
    assert out["count"] == 1 and out["wins"] == 1 and out["losses"] == 0


def test_find_pair_preserves_real_competition_section():
    """Regression (verified live 2026-08-16): the ``laliga`` command resolved
    Las Palmas-Albacete via the homepage fallback, where the row's real
    competition section is "LaLiga2" -- but ``_find_pair_in_rows`` dropped
    the field, so the canonical identity claimed "La Liga" (the requested
    league) and the cross-source validator flagged a false "laliga" vs
    "laliga2" discrepancy. The resolved match must carry the row's real
    competition section (or None on a plain league page).
    """
    row = {
        "home_name": "Las Palmas",
        "away_name": "Albacete",
        "home_slug": "las-palmas", "home_id": "A1",
        "away_slug": "albacete", "away_id": "B2",
        "match_url": "https://www.flashscore.com/match/football/albacete-B2/las-palmas-A1/",
        "date_text": "02:30",
        "competition": "LaLiga2",
    }
    hv, av = _squash_variants("Las Palmas"), _squash_variants("Albacete")
    found = _find_pair_in_rows([row], hv, av)
    assert found is not None
    assert found["competition"] == "LaLiga2"

    # Swapped home/away order in the row must still carry the section.
    found_swapped = _find_pair_in_rows([row], _squash_variants("Albacete"), _squash_variants("Las Palmas"))
    assert found_swapped is not None
    assert found_swapped["competition"] == "LaLiga2"

    # Plain league-page rows have no section tag -> None (caller falls back
    # to the requested league label).
    row_no_comp = {k: v for k, v in row.items() if k != "competition"}
    assert _find_pair_in_rows([row_no_comp], hv, av)["competition"] is None


def test_find_pair_carries_competition_from_team_fixture_row():
    """Regression (2026-08-17): the team-fixtures fallback resolved matches
    several days ahead (e.g. Oviedo-Leganes on 22/08) but its rows carried
    no competition, so ``league_mismatch`` could never fire for them. Rows
    now carry the ``headerLeague__title`` section they belong to, and
    ``_find_pair_in_rows`` must pass it through unchanged so the caller can
    cross-check it against the requested league.
    """
    row = {
        "home_name": "Real Oviedo",
        "away_name": "Leganes",
        "home_slug": "real-oviedo", "home_id": "SzYzw34K",
        "away_slug": "leganes", "away_id": "Mi0rXQg7",
        "match_url": "https://www.flashscore.com/match/football/leganes-Mi0rXQg7/real-oviedo-SzYzw34K/",
        "date_text": "22/08 20:30",
        "competition": "LaLiga2",
    }
    hv, av = _squash_variants("Real Oviedo"), _squash_variants("Leganes")
    found = _find_pair_in_rows([row], hv, av)
    assert found is not None
    assert found["competition"] == "LaLiga2"

    # Swapped sides must still carry the section.
    found_swapped = _find_pair_in_rows([row], _squash_variants("Leganes"), _squash_variants("Real Oviedo"))
    assert found_swapped is not None
    assert found_swapped["competition"] == "LaLiga2"

    # A team-fixture row before any section header has no tag -> None.
    row_no_tag = {k: v for k, v in row.items() if k != "competition"}
    assert _find_pair_in_rows([row_no_tag], hv, av)["competition"] is None


def test_team_fixtures_scrape_tags_rows_with_competition_section():
    """The team-fixtures scrape JS must walk section headers and rows in DOM
    order, assigning each row the last ``headerLeague__title`` seen before it
    (verified live 2026-08: the page groups a team's fixtures by competition).
    The newline guard must stay DOUBLED in the Python source (same rule as
    the homepage scrape) so the compiled JS receives a valid ``'\\n'``.
    """
    src = FlashscoreBrowserClient.scrape_team_fixtures.__code__.co_consts
    const = next((c for c in src if isinstance(c, str) and "lastCompetition" in c), None)
    assert const is not None, "team-fixtures JS constant not found"
    assert 'headerLeague__title' in const
    assert "event__match" in const
    good = "includes('" + chr(92) + "n')"  # JS `'\n'` escape (source had doubled \\n)
    bad = "includes('" + chr(10) + "')"  # real newline (source had single \n)
    assert good in const, "team-fixtures JS must compile to includes('\\n') (doubled source backslash)"
    assert bad not in const, "team-fixtures JS must not contain a real newline inside the string (single source backslash)"


def test_lineups_scrape_selector_does_not_require_extended_modifier():
    """Regression (verified live 2026-08-16): Flashscore rendered the HOME
    XI container as a bare ``lf__formation`` (no ``lf__formation--extended``)
    for Espanyol-Levante, so the old home selector matched ZERO players while
    the away side worked. The scrape JS must select the home XI structurally
    (first formation block, away excluded) and never depend on the modifier."""
    import inspect

    src = inspect.getsource(FlashscoreBrowserClient.scrape_match_lineups)
    home_sel = '.lf__formation:not(.lf__formationAway) [data-testid="wcl-lineupsParticipantName"]'
    assert home_sel in src
    # The old brittle selector must be gone.
    assert ".lf__formation.lf__formation--extended [" not in src
    # Away keeps its dedicated container selector.
    assert '.lf__formation.lf__formationAway [data-testid="wcl-lineupsParticipantName"]' in src


def test_parse_predicted_lineups():
    """Predicted lineups: body mentions 'predicted lineup', both XIs parsed
    with jersey numbers, formations extracted home-then-away."""
    data = {
        "homePlayers": [
            "22 | Mikelionis", "45 | Moutachy", "3 | Tolordava",
            "77 | Lekiatas", "33 | Konatar",
        ],
        "awayPlayers": [
            "1 | Livakovic", "2 | Moharrami", "7 | Stojkovic", "9 | Beljo",
        ],
        "headers": ["4 - 1 - 4 - 1", "FORMATION", "4 - 3 - 3"],
        "body": "PREDICTED LINEUPS for this match ... FORMATION",
    }
    out = parse_lineups_page(data)
    assert out is not None
    assert out["status"] == "predicted"
    assert out["home_count"] == 5
    assert out["away_count"] == 4
    assert out["home"][0] == {"number": "22", "name": "Mikelionis"}
    assert out["home"][1] == {"number": "45", "name": "Moutachy"}
    assert out["away"][-1] == {"number": "9", "name": "Beljo"}
    assert out["formations"] == ["4-1-4-1", "4-3-3"]


def test_parse_confirmed_lineups():
    """Confirmed starting lineups (post-announcement) are NOT flagged predicted."""
    data = {
        "homePlayers": ["1 | Onana", "4 | Maguire", "10 | Rashford"],
        "awayPlayers": ["1 | Alisson", "66 | Alexander-Arnold", "11 | Salah"],
        "headers": ["4 - 2 - 3 - 1", "FORMATION", "4 - 3 - 3"],
        "body": "STARTING LINEUPS ... FORMATION",
    }
    out = parse_lineups_page(data)
    assert out is not None
    assert out["status"] == "confirmed"
    assert out["formations"] == ["4-2-3-1", "4-3-3"]
    assert out["home_count"] == 3


def test_parse_no_players_returns_none():
    """Lineups tab present but no players rendered (not announced yet)."""
    data = {
        "homePlayers": [],
        "awayPlayers": [],
        "headers": ["FORMATION"],
        "body": "LINEUPS",
    }
    assert parse_lineups_page(data) is None


def test_parse_player_without_number():
    """Some variants render the name without a leading jersey number."""
    data = {
        "homePlayers": ["Mikelionis", "22 | Moutachy"],
        "awayPlayers": ["Livakovic"],
        "headers": ["FORMATION"],
        "body": "lineups",
    }
    out = parse_lineups_page(data)
    assert out is not None
    assert out["home"][0] == {"number": None, "name": "Mikelionis"}
    assert out["home"][1] == {"number": "22", "name": "Moutachy"}
    # No explicit marker -> default PREDICTED (never claim confirmed).
    assert out["status"] == "predicted"


def test_parse_live_newline_format():
    """Live DOM renders jersey and name on separate lines ('1\nGreif')."""
    data = {
        "homePlayers": ["1\nGreif", "98\nMaitland-Niles", "21\nKluivert"],
        "awayPlayers": ["44\nSurovcik"],
        "headers": ["4 - 2 - 3 - 1", "FORMATION", "4 - 2 - 3 - 1"],
        "body": "PREDICTED LINEUPS ...",
    }
    out = parse_lineups_page(data)
    assert out is not None
    assert out["status"] == "predicted"
    assert out["home"][0] == {"number": "1", "name": "Greif"}
    assert out["home"][1] == {"number": "98", "name": "Maitland-Niles"}
    assert out["formations"] == ["4-2-3-1", "4-2-3-1"]
    assert out["home_count"] == 3


def test_parse_formation_dense_format():
    """Formation values may render as '4-3-3' without spaces."""
    data = {
        "homePlayers": ["1 | A"],
        "awayPlayers": ["1 | B"],
        "headers": ["4-3-3", "FORMATION", "3-5-2"],
        "body": "starting lineups",
    }
    out = parse_lineups_page(data)
    assert out["formations"] == ["4-3-3", "3-5-2"]
    assert out["status"] == "confirmed"


def test_parse_marker_case_insensitive():
    """Marker matching is case-insensitive on the truncated body."""
    data = {
        "homePlayers": ["1 | A"],
        "awayPlayers": ["1 | B"],
        "headers": ["FORMATION"],
        "body": "STARTING LINEUPS ... FORMATION 4-3-3",
    }
    out = parse_lineups_page(data)
    assert out["status"] == "confirmed"
