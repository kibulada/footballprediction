"""Tests for context.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.context import build_match_context, input_hash


def test_input_hash_stable():
    a = input_hash({"x": 1, "y": [1, 2]})
    b = input_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert len(a) == 16


def test_build_context_from_stats():
    ctx = build_match_context(
        league="UCL",
        home="Bodo/Glimt",
        away="Union Saint-Gilloise",
        kickoff="2026-08-13T19:00:00Z",
        stats={
            "home_form": "W-W-D-W-W",
            "away_form": "W-W-L-D-W",
            "home_gf_avg": 1.8,
            "home_ga_avg": 0.9,
            "away_gf_avg": 1.4,
            "away_ga_avg": 1.2,
            "h2h": {"wins": 1, "draws": 0, "losses": 0},
        },
        odds={"has_odds": True, "consensus": {"home": 2.1, "draw": 3.4, "away": 3.6}},
        sources=["football_data"],
    )
    assert ctx.home == "Bodo/Glimt"
    assert ctx.has_attack_defense
    assert ctx.has_odds
    assert ctx.form_samples == 5
    assert ctx.h2h == {"wins": 1, "draws": 0, "losses": 0}
    assert ctx.input_hash


def test_build_context_missing_data_stays_missing():
    ctx = build_match_context(
        league="EPL",
        home="A",
        away="B",
        stats={"home_form": "n/a", "away_form": None, "home_gf_avg": None},
        odds={"has_odds": False, "consensus": {"home": 0, "draw": 0, "away": 0}},
    )
    assert ctx.home_form is None
    assert ctx.away_form is None
    assert ctx.has_attack_defense is False
    assert ctx.has_odds is False
    assert ctx.consensus_odds is None


def test_build_context_no_n_marks_forms():
    ctx = build_match_context(
        league="EPL", home="A", away="B",
        stats={"home_form": "n/a", "away_form": "W-W"},
    )
    assert ctx.home_form is None
    assert ctx.away_form == "W-W"


def test_recent_goals_passthrough_and_attack_defense():
    ctx = build_match_context(
        league="EPL", home="A", away="B",
        stats={
            "home_recent_goals": [[2, 0], [1, 1]],
            "away_recent_goals": [(0, 1), (0, 2)],
        },
    )
    assert ctx.home_recent_goals == [(2, 0), (1, 1)]
    assert ctx.away_recent_goals == [(0, 1), (0, 2)]
    # raw scorelines alone are enough for attack/defense features
    assert ctx.has_attack_defense is True
    assert ctx.home_gf_avg is None


def test_recent_goals_malformed_becomes_none():
    ctx = build_match_context(
        league="EPL", home="A", away="B",
        stats={
            "home_recent_goals": [[2], [1, 1, 3], "x", [2, 0]],
            "away_recent_goals": [],
        },
    )
    assert ctx.home_recent_goals == [(2, 0)]
    assert ctx.away_recent_goals is None


def test_snapshot_is_json_serializable():
    ctx = build_match_context(
        league="EPL", home="A", away="B",
        stats={"home_form": "W-W", "away_form": "L-L"},
    )
    snap = ctx.snapshot()
    assert snap["league"] == "EPL"
    assert isinstance(snap["sources"], list)


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
