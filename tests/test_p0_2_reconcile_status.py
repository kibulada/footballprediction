"""P0-2: reconcile_status unit tests."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.source_match import reconcile_status


def _tok(h: int, m: int = 0) -> datetime:
    return datetime(2026, 8, 17, h, m, tzinfo=timezone.utc)


def test_livescore_live_in_window_overrides_flashscore_finished() -> None:
    kickoff = _tok(15)
    now = _tok(15, 30)
    assert reconcile_status(kickoff, now, flashscore_status="finished", livescore_status="live") == "live"


def test_pre_match_window_returns_scheduled() -> None:
    kickoff = _tok(15)
    now = _tok(10)
    assert reconcile_status(kickoff, now, flashscore_status="scheduled", livescore_status="scheduled") == "scheduled"


def test_post_match_no_live_source_returns_finished() -> None:
    kickoff = _tok(15)
    now = kickoff + timedelta(hours=5)
    assert reconcile_status(kickoff, now, flashscore_status="finished", livescore_status="finished") == "finished"


def test_in_window_no_source_returns_live() -> None:
    kickoff = _tok(15)
    now = _tok(15, 30)
    assert reconcile_status(kickoff, now, flashscore_status="scheduled", livescore_status="scheduled") == "live"


def test_invalid_kickoff_returns_unknown() -> None:
    assert reconcile_status(None, _tok(15)) == "unknown"
    assert reconcile_status(_tok(15), None) == "unknown"


def test_live_source_outside_window_falls_through() -> None:
    kickoff = _tok(15)
    now = _tok(10)
    assert reconcile_status(kickoff, now, flashscore_status="scheduled", livescore_status="live") == "scheduled"


def test_naive_datetime_treated_as_utc() -> None:
    naive_kickoff = datetime(2026, 8, 17, 15)
    naive_now = datetime(2026, 8, 17, 15, 30)
    assert reconcile_status(naive_kickoff, naive_now, livescore_status="live") == "live"


if __name__ == "__main__":
    test_livescore_live_in_window_overrides_flashscore_finished()
    test_pre_match_window_returns_scheduled()
    test_post_match_no_live_source_returns_finished()
    test_in_window_no_source_returns_live()
    test_invalid_kickoff_returns_unknown()
    test_live_source_outside_window_falls_through()
    test_naive_datetime_treated_as_utc()
    print("P0-2 reconcile_status: 7/7 tests passed")
