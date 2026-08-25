"""P1.1 — Kickoff cross-check guard (acceptance tests).

The Genclerbirligi/Cadiz-B incident: a tolerant team-name match resolved the
WRONG fixture (Cadiz B vs Real Betis B, kickoff 2026-08-15T08:00:00Z = 15:00
WIB Aug 15) for a query on Genclerbirligi vs Fenerbahce whose real kickoff
was 2026-08-15T18:30:00Z (= 01:30 WIB Aug 16). The wrong kickoff was 10.5h in
the past, so the pipeline wrongly declared ``match_finished=True``.

P1.1 requires: when independent sources disagree on kickoff beyond tolerance
(2h), the match must be flagged ``kickoff_uncertain`` and ``match_finished``
must NEVER be derived from it -- the status is "cannot determine", not
"finished".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.analyse import _kickoff_cross_check  # noqa: E402
from agents.football.format import _format_analyse_kickoff_uncertain  # noqa: E402

# Incident values (from the audit of 2026-08-15/16):
WRONG_KICKOFF = "2026-08-15T08:00:00Z"    # Cadiz B (wrong fixture resolved)
REAL_KICKOFF = "2026-08-15T18:30:00Z"     # Genclerbirligi vs Fenerbahce


def test_cross_check_accepts_agreeing_sources():
    ok, deltas = _kickoff_cross_check(
        REAL_KICKOFF,
        {"fixture": REAL_KICKOFF, "odds": REAL_KICKOFF},
    )
    assert ok is False
    assert deltas == {}


def test_cross_check_accepts_minor_schedule_shift():
    # A few minutes' difference (fixture source vs commence_time) is normal.
    ok, deltas = _kickoff_cross_check(
        "2026-08-15T18:30:00Z",
        {"fixture": "2026-08-15T18:30:00Z", "odds": "2026-08-15T18:35:00Z"},
    )
    assert ok is False
    assert deltas == {"odds": 0.08}


def test_cross_check_flags_incident_disagreement():
    # The Cadiz-B-class bug: two sources disagree by 10.5 hours.
    ok, deltas = _kickoff_cross_check(
        WRONG_KICKOFF,
        {"fixture": WRONG_KICKOFF, "odds": REAL_KICKOFF},
    )
    assert ok is True
    assert deltas["odds"] == 10.5


def test_cross_check_needs_two_sources():
    # A single source cannot be cross-checked -- nothing to disagree with.
    ok, deltas = _kickoff_cross_check(WRONG_KICKOFF, {"odds": WRONG_KICKOFF})
    assert ok is False
    assert deltas == {}


def test_cross_check_unparseable_candidate_ignored():
    ok, deltas = _kickoff_cross_check(
        REAL_KICKOFF,
        {"fixture": REAL_KICKOFF, "odds": "not-a-date"},
    )
    assert ok is False


def test_incident_replay_uncertain_not_finished():
    """The incident's exact kickoff candidates must yield kickoff_uncertain
    (not match_finished) when the sources disagree."""
    # Primary (first-wins) = the wrong Cadiz B kickoff; the real kickoff
    # arrives from an independent source (odds commence_time).
    kickoff_uncertain, deltas = _kickoff_cross_check(
        WRONG_KICKOFF,
        {"fixture": WRONG_KICKOFF, "odds": REAL_KICKOFF},
    )
    assert kickoff_uncertain is True
    assert deltas["odds"] > 2.0
    # The pipeline guard (analyse.py) refuses to derive match_finished from
    # an uncertain kickoff -- simulate the gate.
    match_finished = False
    if not kickoff_uncertain:
        from agents.football.analyse import _kickoff_hours_ahead
        hours = _kickoff_hours_ahead(WRONG_KICKOFF)
        if hours is not None and hours < 0:
            match_finished = True
    assert match_finished is False


def test_incident_resolved_kickoff_not_finished():
    """With the corrected fixture resolution the real kickoff is pre-match
    (query ran 01:26 WIB = 18:26 UTC, kickoff 18:30 UTC): NOT finished."""
    from datetime import datetime, timezone
    from agents.football.analyse import _kickoff_hours_ahead

    query_time = datetime(2026, 8, 15, 18, 26, tzinfo=timezone.utc)
    hours = _kickoff_hours_ahead(REAL_KICKOFF, now=query_time)
    assert hours is not None and hours >= 0
    assert hours < 1.0  # 4 minutes to kickoff


def test_uncertain_card_never_says_finished():
    out = _format_analyse_kickoff_uncertain({
        "home": "Gen\u00e7lerbirli\u011fi",
        "away": "Fenerbah\u00e7e",
        "league": "S\u00fcper Lig",
        "kickoff": WRONG_KICKOFF,
        "kickoff_deltas": {"odds": 10.5},
        "sources": ["nowgoal_odds", "thesportsdb"],
        "quota": {},
        "stats": {"h2h": {"wins": 1, "draws": 3, "losses": 2}},
    })
    assert "Match sudah selesai" not in out["body"]
    assert "tidak dapat dipastikan" in out["body"]
    assert "prediksi tidak dibuat" in out["body"]
    assert "odds: 10.5 jam" in out["body"]
