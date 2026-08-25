"""Tests for settler.py (manual + auto settle of prediction-log snapshots)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import append_snapshot, list_unsettled
from agents.football.settler import settle_auto, settle_manual

MID = "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z"


def _snap(path, *, match_id=MID, home="Arsenal", away="Chelsea",
          kickoff="2026-08-15T14:00:00Z", league="EPL", prob=None):
    append_snapshot(
        path, match_id=match_id, league=league, home=home, away=away, kickoff=kickoff,
        prob=prob if prob is not None else {"home": 0.55, "draw": 0.25, "away": 0.20},
        odds=None, edge=None, confidence=None, signal=None, calibration=None,
        model_version=None, input_hash=None, best_pick=None, sources=None,
    )


def test_settle_manual_settles(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path)
    out = settle_manual(path, home="Arsenal", away="Chelsea", result="2-1")
    assert out["status"] == "settled"
    assert out["result"] == "2-1"
    assert list_unsettled(path) == []


def test_settle_manual_tolerant_names(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path, home="FK Arsenal", away="Chelsea")  # snapshot has FK prefix
    out = settle_manual(path, home="Arsenal", away="Chelsea", result="1-0")
    assert out["status"] == "settled"


def test_settle_manual_not_found(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path)
    out = settle_manual(path, home="Liverpool", away="Chelsea", result="1-0")
    assert out["status"] == "not_found"
    assert list_unsettled(path)  # nothing settled


def test_settle_manual_ambiguous_then_date_disambiguates(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path, kickoff="2026-08-15T14:00:00Z",
          match_id="EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z")
    _snap(path, kickoff="2026-08-16T14:00:00Z",
          match_id="EPL||Arsenal||Chelsea||2026-08-16T14:00:00Z")
    out = settle_manual(path, home="Arsenal", away="Chelsea", result="2-1")
    assert out["status"] == "ambiguous"
    assert len(out["candidates"]) == 2
    out2 = settle_manual(path, home="Arsenal", away="Chelsea", result="2-1", date="2026-08-15")
    assert out2["status"] == "settled"


def test_settle_manual_bad_result(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path)
    out = settle_manual(path, home="Arsenal", away="Chelsea", result="abc")
    assert out["status"] == "bad_result"
    assert list_unsettled(path)  # nothing settled


def test_settle_auto_by_date(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path, kickoff="2026-08-15T14:00:00Z",
          match_id="EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z")
    _snap(path, kickoff="2026-08-16T14:00:00Z",
          match_id="EPL||A||B||2026-08-16T14:00:00Z", home="A", away="B")
    results = [
        {"home": "Arsenal", "away": "Chelsea", "home_goals": 3, "away_goals": 1},
        {"home": "Man City", "away": "United", "home_goals": 2, "away_goals": 0},
    ]
    out = settle_auto(path, date="2026-08-15", results=results)
    assert out["status"] == "auto"
    assert len(out["settled"]) == 1
    assert out["settled"][0]["result"] == "3-1"
    assert out["not_found"] == []
    # the other-date snapshot is untouched
    assert len(list_unsettled(path)) == 1


def test_settle_auto_tolerant_and_not_found(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap(path, home="Bodo/Glimt", away="Union Saint-Gilloise",
          kickoff="2026-08-15T14:00:00Z", league="UCL",
          match_id="UCL||Bodo/Glimt||Union Saint-Gilloise||2026-08-15T14:00:00Z")
    results = [
        {"home": "FK Bodo/Glimt", "away": "Union Saint-Gilloise",
         "home_goals": 2, "away_goals": 1},
    ]
    out = settle_auto(path, date="2026-08-15", results=results)
    assert len(out["settled"]) == 1
    assert out["settled"][0]["result"] == "2-1"

    # unmatched snapshot on the same date -> reported, never invented
    path2 = tmp_path / "p2.jsonl"
    _snap(path2, home="X", away="Y", kickoff="2026-08-15T14:00:00Z",
          match_id="L||X||Y||2026-08-15T14:00:00Z")
    out2 = settle_auto(path2, date="2026-08-15", results=results)
    assert out2["settled"] == []
    assert len(out2["not_found"]) == 1
