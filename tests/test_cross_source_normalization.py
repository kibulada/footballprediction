"""Regression tests for the 2026-08-22 cross-source normalization fixes.

Everton vs Crystal Palace (EPL, 2026-08-22) e2e audit found every field
flagged agreement=false despite identical facts: competition spelling
("epl" vs "Premier League"), form sequence orientation (newest-first vs
oldest-first), H2H window depth (last 5 vs last 10 meetings) and standings
shape (full table vs two-team snapshot). These tests pin the semantic
comparators, the evidence-gate completeness propagation and the O/U goal-line
disclosure in the MARKET block.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.datasources import (  # noqa: E402
    CONF_HIGH,
    available,
    canonical_competition,
    canonical_match_identity,
    field_values_agree,
    merge_field,
)
from agents.football.format import format_market_signal  # noqa: E402
from agents.football.signal_engine import _demo  # noqa: E402


# ---- Fix 1a: competition canonicalization ----------------------------------

def test_canonical_competition_converges_registered_leagues():
    assert canonical_competition("EPL") == "epl"
    assert canonical_competition("Premier League") == "epl"
    assert canonical_competition("England Premier League") == "epl"
    # different registered leagues stay distinct
    assert canonical_competition("La Liga") != "epl"


def test_canonical_competition_unresolvable_keeps_own_token():
    tok = canonical_competition("Club Friendlies 2026")
    assert tok and "clubfriendlies" in tok.replace(" ", "")
    assert canonical_competition(None) is None


def test_canonical_match_identity_epl_vs_premier_league_agree():
    a = canonical_match_identity(
        home="Everton", away="Crystal Palace",
        kickoff="2026-08-22T14:00:00Z", competition="EPL",
    )
    b = canonical_match_identity(
        home="Everton", away="Crystal Palace",
        kickoff="2026-08-22T14:00:00Z", competition="Premier League",
    )
    assert field_values_agree("match", a, b)


def test_match_agree_rejects_different_fixture_or_kickoff():
    a = canonical_match_identity(home="Everton", away="Crystal Palace",
                                 kickoff="2026-08-22T14:00:00Z", competition="EPL")
    other = canonical_match_identity(home="Arsenal", away="Chelsea",
                                     kickoff="2026-08-22T14:00:00Z", competition="epl")
    assert not field_values_agree("match", a, other)
    late = canonical_match_identity(home="everton", away="crystal palace",
                                    kickoff="2026-08-23T14:00:00Z", competition="premierleague")
    assert not field_values_agree("match", a, late)


# ---- Fix 1b: form orientation ----------------------------------------------

# Real observed shapes: flashscore sequence newest-first, livescore
# oldest-first; identical averages + oldest->newest scorelines.
_FS_FORM = {
    "home": {"sequence": "D-W-L-W-L", "gf_avg": 1.4, "ga_avg": 1.4,
             "sample_size": 5, "recent_goals": [[0, 1], [2, 1], [1, 3], [3, 1], [1, 1]],
             "source": "flashscore"},
    "away": {"sequence": "L-W-W-L", "gf_avg": 1.25, "ga_avg": 2.0,
             "sample_size": 4, "recent_goals": [[0, 3], [3, 1], [2, 1], [0, 3]],
             "source": "flashscore"},
}
_LS_FORM = {
    "home": {"sequence": "L-W-L-W-D", "gf_avg": 1.4, "ga_avg": 1.4,
             "sample_size": 5, "recent_goals": [[0, 1], [2, 1], [1, 3], [3, 1], [1, 1]]},
    "away": {"sequence": "W-L-W-L", "gf_avg": 1.25, "ga_avg": 2.0,
             "sample_size": 4, "recent_goals": [[0, 3], [3, 1], [2, 1], [0, 3]]},
}


def test_form_agrees_across_sequence_orientation():
    assert field_values_agree("form", _FS_FORM, _LS_FORM)


def test_form_disagrees_on_real_fact_change():
    import copy
    changed = copy.deepcopy(_LS_FORM)
    changed["away"]["ga_avg"] = 2.6  # genuinely worse away defense
    assert not field_values_agree("form", _FS_FORM, changed)
    changed2 = copy.deepcopy(_LS_FORM)
    changed2["home"]["recent_goals"] = [[0, 2], [2, 1], [1, 3], [3, 1], [1, 1]]
    assert not field_values_agree("form", _FS_FORM, changed2)


def test_form_uncomparable_shape_falls_back_to_strict():
    legacy_a = {"home": ["W", "D"], "away": ["L"]}
    legacy_b = {"home": ["W", "D"], "away": ["L"]}
    legacy_c = {"home": ["W"], "away": ["L"]}
    assert field_values_agree("form", legacy_a, legacy_b) is True
    assert field_values_agree("form", legacy_a, legacy_c) is False


# ---- Fix 1c: h2h window clamp ----------------------------------------------

_FS_H2H = {"wins": 3, "draws": 2, "losses": 0, "count": 5,
           "h2h_window": "3y", "h2h_total_meetings": 5, "source": "flashscore_h2h"}

# Livescore's own (deeper) history: 10 meetings, 6W-4D from Everton's view.
_MEETINGS = [
    {"home": "Crystal Palace", "away": "Everton", "home_score": 2, "away_score": 2,
     "kickoff": "2026-05-10T13:00:00Z", "status": "finished"},   # D
    {"home": "Everton", "away": "Crystal Palace", "home_score": 2, "away_score": 1,
     "kickoff": "2025-10-05T13:00:00Z", "status": "finished"},   # W
    {"home": "Crystal Palace", "away": "Everton", "home_score": 1, "away_score": 2,
     "kickoff": "2025-02-15T17:30:00Z", "status": "finished"},   # W
    {"home": "Everton", "away": "Crystal Palace", "home_score": 2, "away_score": 1,
     "kickoff": "2024-09-28T14:00:00Z", "status": "finished"},   # W
    {"home": "Everton", "away": "Crystal Palace", "home_score": 1, "away_score": 1,
     "kickoff": "2024-02-19T20:00:00Z", "status": "finished"},   # D
    {"home": "Everton", "away": "Crystal Palace", "home_score": 1, "away_score": 0,
     "kickoff": "2024-01-17T19:45:00Z", "status": "finished"},   # W
    {"home": "Crystal Palace", "away": "Everton", "home_score": 0, "away_score": 0,
     "kickoff": "2024-01-04T20:00:00Z", "status": "finished"},   # D
    {"home": "Crystal Palace", "away": "Everton", "home_score": 2, "away_score": 3,
     "kickoff": "2023-11-11T15:00:00Z", "status": "finished"},   # W
    {"home": "Crystal Palace", "away": "Everton", "home_score": 0, "away_score": 0,
     "kickoff": "2023-04-22T14:00:00Z", "status": "finished"},   # D
    {"home": "Everton", "away": "Crystal Palace", "home_score": 3, "away_score": 0,
     "kickoff": "2022-10-22T14:00:00Z", "status": "finished"},   # W (outside clamp)
]
_LS_H2H = {"wins": 6, "draws": 4, "losses": 0, "meetings": _MEETINGS}

_REF_EVERTON = {"home": "Everton", "away": "Crystal Palace"}


def test_h2h_agrees_after_clamping_to_primary_window():
    # flashscore's last-5 tally must equal livescore's newest-5 meetings
    # recomputed from Everton's perspective (the current match's home team).
    assert field_values_agree("h2h", _FS_H2H, _LS_H2H, _REF_EVERTON)


def test_h2h_still_flags_genuine_conflict():
    wrong = {"wins": 9, "draws": 1, "losses": 0, "meetings": _MEETINGS}
    assert not field_values_agree("h2h", _FS_H2H, wrong, _REF_EVERTON)


def test_h2h_direct_counts_equal_is_agreement():
    same = {"wins": 3, "draws": 2, "losses": 0, "source": "livescore"}
    assert field_values_agree("h2h", _FS_H2H, same, None)


# ---- Fix 1d: lineup name-set comparison ------------------------------------

# Same eleven, different order/shirt numbers/full-vs-surname + coach entry;
# plus the observed genuine divergence (Mingueza vs Sosa).
_FS_LU = {
    "status": "predicted", "formations": ["4-1-4-1", "3-4-3"],
    "home": [{"number": "1", "name": "Pickford"}, {"number": "15", "name": "O'Brien"},
             {"number": "32", "name": "Branthwaite"}, {"number": "6", "name": "Tarkowski"},
             {"number": "16", "name": "Mykolenko"}, {"number": "7", "name": "Hackney"}],
    "away": [{"number": "44", "name": "Benitez"}, {"number": "30", "name": "Mingueza"}],
}
_LS_LU_SAME = {
    "home": {"formation": [4, 2, 3, 1], "players": [
        {"name": "Jordan Pickford", "shirt": 1, "position": "Goalkeeper"},
        {"name": "Jake O'Brien", "shirt": 15, "position": "Defender"},
        {"name": "James Tarkowski", "shirt": 6, "position": "Defender"},
        {"name": "Jarrad Branthwaite", "shirt": 32, "position": "Defender"},
        {"name": "Vitaliy Mykolenko", "shirt": 16, "position": "Defender"},
        {"name": "Hayden Hackney", "shirt": 30, "position": "Midfielder"},
        {"name": "David Moyes", "shirt": None, "position": "COACH"},
    ]},
}


def test_lineup_agrees_same_eleven_different_presentation():
    assert field_values_agree("lineup", {"home": _FS_LU["home"]}, _LS_LU_SAME)


def test_lineup_disagrees_on_different_player():
    ls_away = {"away": {"players": [{"name": "Walter Benitez", "shirt": 44},
                                    {"name": "Borna Sosa", "shirt": 24}]}}
    assert not field_values_agree("lineup", _FS_LU, ls_away)


# ---- Fix 1e: standings row comparison --------------------------------------

_FS_TABLE = {"tables": {"overall": [
    {"pos": 1, "team": "Arsenal", "mp": 1, "w": 1, "d": 0, "l": 0, "gf": 3, "ga": 0, "gd": 3, "pts": 3},
    {"pos": 5, "team": "Everton", "mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0},
    {"pos": 7, "team": "Crystal Palace", "mp": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0},
]}}
_LS_TABLE = {
    "away": {"pos": 12, "team": "Crystal Palace", "played": 0, "wins": 0,
             "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
    "home": {"pos": 8, "team": "Everton", "played": 0, "wins": 0,
             "draws": 0, "losses": 0, "gf": 0, "ga": 0, "gd": 0, "points": 0},
}


def test_standings_agree_despite_shape_and_tie_order():
    # pos 5/7 vs pos 8/12 among level-points teams is tie-break noise; the
    # played/wins/gf/ga/gd state is identical -> agreement.
    assert field_values_agree("standings", _FS_TABLE, _LS_TABLE, _REF_EVERTON)


def test_standings_disagrees_on_real_stat_change():
    import copy
    changed = copy.deepcopy(_LS_TABLE)
    changed["home"]["points"] = 3
    assert not field_values_agree("standings", _FS_TABLE, changed, _REF_EVERTON)


# ---- merge_field end-to-end with ref context -------------------------------

def test_merge_field_uses_semantic_comparator_and_confidence():
    # Timestamps must be FRESH relative to now: field_confidence downgrades
    # HIGH -> MEDIUM when the samples are stale, so hardcoded dates turned
    # this into a time bomb (passed on its authoring day, failed forever
    # after the h2h freshness window).
    from datetime import datetime, timedelta, timezone

    _fresh = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    fv = merge_field(
        "h2h",
        {"flashscore": available(_FS_H2H, _fresh),
         "livescore": available(_LS_H2H, _fresh)},
        priority={"flashscore": 100, "livescore": 80},
        ref=_REF_EVERTON,
    )
    assert fv.agreement is True
    assert fv.discrepancy is False
    assert fv.confidence == CONF_HIGH
    assert fv.value == _FS_H2H


def test_merge_field_without_ref_still_clamps_via_count():
    fv = merge_field(
        "h2h",
        {"flashscore": available(_FS_H2H),
         "livescore": available(_LS_H2H)},
        priority={"flashscore": 100, "livescore": 80},
        ref=None,
    )
    # no ref home name -> recount impossible -> strict comparison keeps the
    # honest disagreement (never silently over-claims agreement).
    assert fv.agreement is False


# ---- Fix 2: evidence-gate veto reaches the quality numbers -----------------

def test_source_confidence_gate_caps_completeness_and_records_trail():
    from agents.football.analyse import _apply_source_confidence_gate

    prediction = {"data_completeness": 1.0}
    _apply_source_confidence_gate(
        prediction, passed=False, reason="3/3 critical fields LOW",
    )
    assert prediction["data_completeness"] == 0.5
    gate = prediction["source_confidence_gate"]
    assert gate["passed"] is False
    assert gate["reason"] == "3/3 critical fields LOW"
    assert gate["completeness_before"] == 1.0
    assert gate["completeness_capped"] == 0.5


def test_source_confidence_gate_noop_when_passed_or_missing():
    from agents.football.analyse import _apply_source_confidence_gate

    prediction = {"data_completeness": 0.9}
    _apply_source_confidence_gate(prediction, passed=True, reason=None)
    assert prediction["data_completeness"] == 0.9
    assert "source_confidence_gate" not in prediction
    _apply_source_confidence_gate(None, passed=False, reason="x")  # never raises
    low = {"data_completeness": 0.3}
    _apply_source_confidence_gate(low, passed=False, reason="x")
    assert low["data_completeness"] == 0.3  # cap never raises completeness


# ---- Fix 3 surfaces ---------------------------------------------------------

def test_market_block_discloses_goal_line_change():
    se = _demo()
    se["market_block"] = {"ou": {"canonical": False}, "ah": {}}
    se["ah_consensus"] = None
    payload = {
        "league": "EPL", "home": "Everton", "away": "Crystal Palace",
        "kickoff": "2026-08-22T14:00:00Z",
        "signal_engine": se,
        "odds": {"totals": {
            "Over 2.5": {"odds": 2.02, "opening": 2.04, "point": 2.5, "opening_point": 2.75},
            "Under 2.5": {"odds": 1.92, "opening": 1.90, "point": 2.5, "opening_point": 2.75},
        }},
    }
    body = format_market_signal(payload)["body"]
    assert "Opening: 2.04 (garis 2.75)" in body
    assert "Latest: 2.02 (garis 2.50)" in body


def test_market_block_plain_when_line_unchanged():
    se = _demo()
    se["market_block"] = {"ou": {"canonical": False}, "ah": {}}
    se["ah_consensus"] = None
    payload = {
        "league": "EPL", "home": "Everton", "away": "Crystal Palace",
        "kickoff": "2026-08-22T14:00:00Z",
        "signal_engine": se,
        "odds": {"totals": {
            "Over 2.5": {"odds": 1.87, "opening": 1.95, "point": 2.5},
            "Under 2.5": {"odds": 1.95, "opening": 1.86, "point": 2.5},
        }},
    }
    body = format_market_signal(payload)["body"]
    assert "Opening: 1.95" in body
    assert "garis" not in body


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
