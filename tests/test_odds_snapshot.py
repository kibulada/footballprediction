"""Tests for the PHASE 32-33 odds-snapshot flow: runner fuzzy lookup +
Discord renderer. No live network: fixtures are synthetic temp files."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import append_snapshot, make_match_id  # noqa: E402
from agents.football.runner import _run  # noqa: E402
from agents.football.format import format_odds_snapshot  # noqa: E402


class _Args:
    """Minimal argparse.Namespace stand-in for runner modes we test."""

    def __init__(self, mode: str, **kw):
        self.mode = mode
        for k, v in kw.items():
            setattr(self, k, v)


def _point_log_at(log_path: Path, monkeypatch) -> None:
    """Redirect the runner's prediction_log config to a temp file."""
    import agents.football.runner as runner

    cfg = runner.load_config()
    cfg["prediction_log"] = {"enabled": True, "file": str(log_path)}
    monkeypatch.setattr(runner, "load_config", lambda: cfg)


def _write_snapshot(path: Path) -> str:
    mid = make_match_id("EPL", "Arsenal", "Chelsea", "2026-08-15T14:00:00Z")
    append_snapshot(
        path,
        match_id=mid,
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge={"home": 2.1},
        confidence=0.6,
        signal=60,
        calibration={"quality": 0.9},
        model_version="0.1.0",
        input_hash="h1",
        best_pick=None,
        sources=["football_data"],
    )
    return mid


def test_runner_odds_snapshot_by_match_id(tmp_path, monkeypatch):
    log = tmp_path / "pred.jsonl"
    mid = _write_snapshot(log)
    _point_log_at(log, monkeypatch)

    args = _Args("odds-snapshot", match_id=mid, timing="T-6h",
                 odds="1.75,3.70,4.50", bookmakers=None, sources=None,
                 home=None, away=None, league=None)
    result = asyncio.run(_run(args))
    assert result["status"] == "odds_snapshot"
    assert result["timing"] == "T-6h"
    assert result["odds"]["home"] == 1.75
    # row persisted
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[-1]["event"] == "odds_snapshot"
    assert rows[-1]["timing"] == "T-6h"


def test_runner_odds_snapshot_fuzzy_lookup(tmp_path, monkeypatch):
    log = tmp_path / "pred.jsonl"
    _write_snapshot(log)
    _point_log_at(log, monkeypatch)

    args = _Args("odds-snapshot", match_id=None, timing="T-1h",
                 odds="1.70,3.70,4.60", bookmakers=12, sources="the-odds-api",
                 home="Arsenal", away="Chelsea", league=None)
    result = asyncio.run(_run(args))
    assert result["status"] == "odds_snapshot"
    # Note: 'Arsenal' and 'Chelsea' are resolved via alias table to
    # 'Arsenal FC' and 'Chelsea FC', so the match_id includes the
    # canonical names.
    assert "Arsenal FC||Chelsea FC" in result["match_id"]
    assert result["bookmakers_count"] == 12


def test_runner_odds_snapshot_no_match(tmp_path, monkeypatch):
    log = tmp_path / "pred.jsonl"
    _write_snapshot(log)
    _point_log_at(log, monkeypatch)

    args = _Args("odds-snapshot", match_id=None, timing="T-1h",
                 odds="1.70,3.70,4.60", bookmakers=None, sources=None,
                 home="TeamX", away="TeamY", league=None)
    result = asyncio.run(_run(args))
    assert "error" in result


def test_runner_odds_snapshot_bad_odds(tmp_path, monkeypatch):
    log = tmp_path / "pred.jsonl"
    mid = _write_snapshot(log)
    _point_log_at(log, monkeypatch)

    args = _Args("odds-snapshot", match_id=mid, timing="T-6h",
                 odds="1.75,3.70", bookmakers=None, sources=None,
                 home=None, away=None, league=None)
    result = asyncio.run(_run(args))
    assert "error" in result


def test_format_odds_snapshot():
    payload = {
        "status": "odds_snapshot",
        "match_id": "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z",
        "timing": "T-6h",
        "odds": {"home": 1.75, "draw": 3.7, "away": 4.5},
        "bookmakers_count": 14,
        "sources": ["the-odds-api"],
    }
    rendered = format_odds_snapshot(payload)
    assert rendered["title"] == "⏱️ Odds Snapshot"
    assert "T-6h" in rendered["body"]
    assert "1.75" in rendered["body"]
    assert "bookie: 14" in rendered["footer"]


def test_format_odds_snapshot_error():
    rendered = format_odds_snapshot({"error": "tidak ada snapshot"})
    assert "tidak ada snapshot" in rendered["body"]
