"""P1.2 — Canonical kickoff role in match_id (acceptance tests).

The same real-world fixture can resolve its kickoff from different sources
(flashscore fixture vs odds commence_time) that differ by minutes. Before
P1.2 the full timestamp was part of ``match_id``, so two queries for one
match produced two different match_ids and the Layer-3 stability guard could
never find the prior pick.

P1.2 requires: only the match DATE is part of the canonical match_id; two
queries with kickoffs a few minutes apart must yield the SAME match_id and
Layer 3 must retrieve the prior pick. Legacy records (full-timestamp
match_ids) must still resolve via the date-only canonical id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import (  # noqa: E402
    append_snapshot,
    last_prediction_snapshot,
    make_match_id,
    settle,
)


def _snapshot(path, *, match_id, kickoff, pick=None, prob=None):
    append_snapshot(
        path,
        match_id=match_id,
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        kickoff=kickoff,
        prob=prob if prob is not None else {"home": 0.5, "draw": 0.25, "away": 0.25},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge={"home": 2.1, "draw": -1.0, "away": -3.5},
        confidence=0.6,
        signal=60,
        calibration={"quality": 0.9, "ece": 0.013, "samples": 4560},
        model_version="0.1.0-elo-poisson",
        input_hash="abc123",
        best_pick={"selection": "Home Win", "market": "1X2"},
        sources=["football_data"],
        signal_engine_pick=pick,
    )


def test_same_date_different_minutes_same_match_id():
    # Same match, kickoff resolved from two sources minutes apart.
    mid_a = make_match_id("Super Lig", "Gen\u00e7lerbirli\u011fi", "Fenerbah\u00e7e",
                          "2026-08-15T18:30:00Z")
    mid_b = make_match_id("Super Lig", "Gen\u00e7lerbirli\u011fi", "Fenerbah\u00e7e",
                          "2026-08-15T18:35:00Z")
    assert mid_a == mid_b
    assert mid_a.endswith("||2026-08-15")


def test_different_dates_different_match_id():
    a = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-15T18:30:00Z")
    b = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-16T18:30:00Z")
    assert a != b


def test_layer3_guard_finds_prior_pick_across_minor_kickoff_delta(tmp_path):
    """Two queries for one match, kickoff differing by minutes: the second
    query's canonical match_id must find the first query's snapshot."""
    path = tmp_path / "pred.jsonl"
    mid_q1 = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-15T18:30:00Z")
    _snapshot(
        path, match_id=mid_q1, kickoff="2026-08-15T18:30:00Z",
        pick={
            "decision": "BEST PICK", "market": "Under 2.5",
            "selection": "Under 2.5", "score": 0.76, "confidence": "HIGH",
            "ts": "2026-08-15T18:20:00Z",
        },
    )
    # Second query: kickoff from a different source, 5 minutes later.
    mid_q2 = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-15T18:35:00Z")
    assert mid_q2 == mid_q1
    prev = last_prediction_snapshot(path, mid_q2)
    assert prev is not None
    assert (prev.get("signal_engine_pick") or {}).get("selection") == "Under 2.5"


def test_legacy_full_timestamp_record_still_resolves(tmp_path):
    """Backward compatibility: records written before P1.2 carry the full
    kickoff timestamp in the match_id; the date-only canonical id must find
    them (exact or legacy-prefix match).

    Note: 'Arsenal' and 'Chelsea' are now resolved via alias table to
    'Arsenal FC' and 'Chelsea FC', so the legacy match_id also uses the
    canonical names.
    """
    path = tmp_path / "pred.jsonl"
    legacy_mid = "EPL||Arsenal FC||Chelsea FC||2026-08-15T18:30:00Z"
    _snapshot(
        path, match_id=legacy_mid, kickoff="2026-08-15T18:30:00Z",
        pick={"decision": "BEST PICK", "selection": "Over 2.5",
              "score": 0.71, "ts": "2026-08-15T18:20:00Z"},
    )
    canonical = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-15T18:35:00Z")
    assert canonical == "EPL||Arsenal FC||Chelsea FC||2026-08-15"
    prev = last_prediction_snapshot(path, canonical)
    assert prev is not None
    assert (prev.get("signal_engine_pick") or {}).get("selection") == "Over 2.5"


def test_settle_accepts_canonical_id_for_legacy_snapshot(tmp_path):
    path = tmp_path / "pred.jsonl"
    legacy_mid = "EPL||Arsenal FC||Chelsea FC||2026-08-15T18:30:00Z"
    _snapshot(path, match_id=legacy_mid, kickoff="2026-08-15T18:30:00Z")
    canonical = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-15T18:35:00Z")
    assert settle(path, match_id=canonical, home_goals=1, away_goals=0) is True
    from agents.football.prediction_log import list_unsettled
    assert list_unsettled(path) == []
