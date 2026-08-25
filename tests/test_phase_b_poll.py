"""Plan B (tapered poll) tests: cadence, timing labels, lineup URL storage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import append_snapshot
from agents.football.runner import cadence_for, timing_label

SCHEDULE = [
    {"until_hours": 1, "interval_minutes": 5},
    {"until_hours": 24, "interval_minutes": 30},
]


def test_cadence_final_hour_is_5min():
    assert cadence_for(0.5, SCHEDULE) == 5
    assert cadence_for(1.0, SCHEDULE) == 5


def test_cadence_outside_final_hour_is_30min():
    assert cadence_for(2.0, SCHEDULE) == 30
    assert cadence_for(23.0, SCHEDULE) == 30


def test_cadence_beyond_last_tier_is_none():
    assert cadence_for(25.0, SCHEDULE) is None


def test_timing_label_hourly():
    assert timing_label(5.0) == "T-5h"
    assert timing_label(1.2) == "T-1h"


def test_timing_label_minutes_in_final_hour():
    assert timing_label(0.5) == "T-30m"
    assert timing_label(0.0833) == "T-5m"


def test_snapshot_stores_flashscore_url(tmp_path):
    path = tmp_path / "pred.jsonl"
    append_snapshot(
        path, match_id="X", league="EPL", home="A", away="B",
        kickoff="2026-08-15T20:00:00Z",
        prob={"home": 0.5, "draw": 0.3, "away": 0.2},
        odds=None, edge=None, confidence=None, signal=None, calibration=None,
        model_version=None, input_hash=None, best_pick=None, sources=[],
        flashscore_url="https://www.flashscore.com/match/abc#/match-summary",
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "flashscore" in row["flashscore_url"]
