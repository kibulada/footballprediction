"""Tests for cross-provider player identity matching + injury-list merging.

Real-world fixture: Espanyol-Levante (2026-08-16) where flashscore reports
"surname + initial" ("Garcia K.") while nowgoal reports "given + surname"
("Kike Garcia") or full names with quoted nicknames ("Enrique Garcia
Martinez, Kike") -- the same humans must merge, different players must not.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.player_identity import (  # noqa: E402
    merge_missing_lists,
    players_match,
)

# ---- unit: players_match --------------------------------------------------

def test_surname_initial_vs_given_surname():
    # flashscore "Garcia K." == nowgoal "Kike Garcia" (same human)
    assert players_match("Garcia K.", "Kike Garcia")
    assert players_match("Kike Garcia", "Garcia K.")
    # Puado J. == Javi Puado
    assert players_match("Puado J.", "Javi Puado")
    # Brugue R. == Roger Brugue
    assert players_match("Brugue R.", "Roger Brugue")


def test_full_name_with_nickname_vs_short_form():
    # nowgoal full name w/ quoted nickname vs flashscore short form
    assert players_match("Enrique Garcia Martinez, Kike", "Garcia K.")
    # and vs its own nickname form
    assert players_match("Enrique Garcia Martinez, Kike", "Kike Garcia")


def test_different_players_do_not_match():
    # different surnames -> never match
    assert not players_match("Primo A.", "Roger Brugue")
    assert not players_match("Garcia K.", "Puado J.")
    # same surname, different initial -> no match
    assert not players_match("Garcia K.", "Garcia J.")
    # same surname, unrelated given names -> no match
    assert not players_match("Kike Garcia", "Sergio Garcia")
    # surname overlap but nothing else -> no match
    assert not players_match("Luis Garcia", "Garcia Fernandez")


def test_accents_particles_and_exact_single_names():
    # accents normalize away
    assert players_match("García K.", "Kike Garcia")
    # particles stay with the surname
    assert players_match("Virgil van Dijk", "Van Dijk V.")
    # single-token surname-only names match on exact equality only
    assert players_match("Mikelionis", "Mikelionis")
    assert not players_match("Mikelionis", "Livakovic")


# ---- integration: merge_missing_lists ------------------------------------

# Real Espanyol-Levante data (2026-08-16).
_FLASH = {
    "home": {"missing": [
        {"name": "Garcia K.", "reason": "Hamstring Injury"},
        {"name": "Puado J.", "reason": "Knee Injury"},
    ], "unsure": []},
    "away": {"missing": [
        {"name": "Brugue R.", "reason": "Red Card"},
        {"name": "Primo A.", "reason": "Shoulder Injury"},
    ], "unsure": []},
}

_NOWGOAL = {
    "home": [
        {"player_id": "89849", "position": "CF", "number": "19",
         "name": "Enrique Garcia Martinez, Kike"},
        {"player_id": "167210", "position": "LW", "number": "7", "name": "Javi Puado"},
        {"player_id": "305989", "position": "DC", "number": None, "name": "Kike Garcia"},
    ],
    "away": [
        {"player_id": "161107", "position": "LW", "number": "7", "name": "Roger Brugue"},
    ],
}


def test_merge_real_espanyol_levante():
    out = merge_missing_lists(_FLASH, _NOWGOAL)
    assert out is not None
    assert sorted(out.keys()) == ["away", "home"]

    home = out["home"]
    assert len(home) == 2  # one Garcia, one Puado -- no "Kike" duplication
    by_name = {e["name"]: e for e in home}
    garcia = by_name.get("Enrique Garcia Martinez, Kike") or by_name.get("Kike Garcia")
    assert garcia is not None
    assert garcia["reason"] == "Hamstring Injury"          # from flashscore
    assert garcia["player_id"] == "89849"                   # from nowgoal (CF row)
    assert garcia["position"] == "CF"
    assert sorted(garcia["sources"]) == ["flashscore", "nowgoal"]
    puado = by_name.get("Javi Puado") or by_name.get("Puado J.")
    assert puado is not None
    assert puado["reason"] == "Knee Injury"
    assert sorted(puado["sources"]) == ["flashscore", "nowgoal"]

    away = out["away"]
    assert len(away) == 2  # Brugue (both sources) + Primo (flashscore only)
    brugue = next(e for e in away if e["player_id"] == "161107")
    assert sorted(brugue["sources"]) == ["flashscore", "nowgoal"]
    assert any(e["name"] == "Primo A." and e["sources"] == ["flashscore"]
               for e in away)


def test_merge_flashscore_only():
    out = merge_missing_lists(_FLASH, None)
    assert out is not None
    assert len(out["home"]) == 2
    assert all(e["sources"] == ["flashscore"] for e in out["home"])
    assert len(out["away"]) == 2


def test_merge_nowgoal_only():
    out = merge_missing_lists(None, _NOWGOAL)
    assert out is not None
    # the two "Kike" rows (CF #19 and DC #null) are the same human -> 1 entry
    assert len(out["home"]) == 2
    kike = next(e for e in out["home"] if "kike" in e["name"].lower())
    assert kike["sources"] == ["nowgoal"]


def test_merge_none_when_no_data():
    assert merge_missing_lists(None, None) is None
    assert merge_missing_lists({}, {}) is None
