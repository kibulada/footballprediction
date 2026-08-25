"""Tests for the nowgoal P0/P1 wiring: format rendering of the context
bundle, the poll-loop realtime ``r`` capture path, and the signal-engine
team-context slot.

All pure / mocked -- no live network calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import format as fmt  # noqa: E402
from agents.football.runner import timing_label  # noqa: E402
from agents.football.signal_engine import _context_component  # noqa: E402


def test_timing_label_negative_is_t0h():
    # in-play captures label T-0h: the movement series' final, heaviest point
    assert timing_label(-0.25) == "T-0h"
    assert timing_label(-2.0) == "T-0h"
    # pre-match labels unchanged
    assert timing_label(24.0) == "T-24h"
    assert timing_label(0.25) == "T-15m"


_NOWGOAL_CONTEXT = {
    "standings": {
        "home": {"team": "FC Utrecht", "rank": 13, "league": "HOL D1",
                 "ft": {"total": {"pts": 0}}},
        "away": {"team": "AZ Alkmaar", "rank": 3, "league": "HOL D1",
                 "ft": {"total": {"pts": 3}}},
    },
    "team_stats": {
        "Goal": {"home_recent10": 1.2, "away_recent10": 2.3},
        "Corners": {"home_recent10": 4.5, "away_recent10": 6.0},
        "Possession": {"home_recent10": 46.1, "away_recent10": 57.7},
    },
    "htft": {
        "rows": {
            "HT-W/FT-W": {"home": {"home": 8, "away": 3},
                          "away": {"home": 7, "away": 1}},
            "HT-L/FT-L": {"home": {"home": 2, "away": 7},
                          "away": {"home": 2, "away": 5}},
        }
    },
    "injuries": {
        "home": [{"position": "CM", "name": "Victor Jensen"}],
        "away": [{"position": "CB", "name": "Suspended Player"}],
    },
    "lineups": {
        "home_team": "FC Utrecht", "away_team": "AZ Alkmaar",
        "home_formation": "3-4-2-1", "away_formation": "4-2-3-1",
        "lineups": {
            "home": {"starters": [{"name": "Barkas V."}], "bench": []},
            "away": {"starters": [{"name": "Meerdink M."}], "bench": []},
        },
    },
}


def _analyse_payload(with_nowgoal: bool = True):
    return {
        "league": "EPL", "home": "Arsenal FC", "away": "Chelsea FC",
        "kickoff": "2026-08-20T15:00:00Z",
        "stats": {
            "home_form": "W-W-W-D-W", "away_form": "W-W-L-D-W",
            "home_gf_avg": 1.8, "home_ga_avg": 0.9,
            "away_gf_avg": 1.4, "away_ga_avg": 1.2,
            "h2h": {"wins": 2, "draws": 0, "losses": 1},
        },
        "odds": {
            "consensus": {"home": 2.10, "draw": 3.40, "away": 3.60},
            "best": {}, "outlier": None, "bookmakers_count": 8,
            "has_odds": True, "totals": {},
        },
        "picks": {}, "signal": 60, "sources": ["nowgoal"],
        "quota": {"odds_api_remaining": 500},
        "decision": {"decision_type": "GOOD"},
        "nowgoal_context": _NOWGOAL_CONTEXT if with_nowgoal else None,
    }


def test_format_renders_nowgoal_context_block():
    rendered = fmt.format_analyse(_analyse_payload())
    body = rendered["body"]
    assert "nowgoal standings" in body
    assert "pos 13 (0pt) • away pos 3 (3pt)" in body
    assert "nowgoal team stats" in body
    assert "Corners 4.5/6" in body
    assert "Possession 46.1%/57.7%" in body
    assert "nowgoal HT/FT" in body
    assert "nowgoal injuries" in body
    assert "Victor Jensen" in body
    # flashscore lineups absent -> nowgoal lineup line shown
    assert "nowgoal lineups" in body
    assert "3-4-2-1" in body


def test_format_nowgoal_block_absent_when_no_context():
    rendered = fmt.format_analyse(_analyse_payload(with_nowgoal=False))
    assert "nowgoal standings" not in rendered["body"]
    assert "nowgoal lineups" not in rendered["body"]


def test_format_nowgoal_block_hidden_when_flashscore_lineups_exist():
    payload = _analyse_payload()
    payload["lineups"] = {"home_count": 11, "status": "predicted"}
    body = fmt.format_analyse(payload)["body"]
    # flashscore lineups win -> no duplicate nowgoal lineup line
    assert "nowgoal lineups" not in body


def test_context_component_reads_nowgoal_injury_fallback():
    ctx = {
        "home": {"missing": ["Victor Jensen"]},
        "away": {"missing": []},
        "team_stats": {},
    }
    # one missing player -> small penalty below the neutral 0.5
    score = _context_component(ctx, "home")
    assert 0.0 <= score < 0.5
    # side without missing players stays neutral
    assert _context_component(ctx, "away") == 0.5
    assert _context_component(None, None) == 0.5


def test_movement_signal_accepts_t0h_live_point():
    """The T-0h (in-play) snapshot extends the movement series past T-15m."""
    from agents.football.movement import movement_signal

    snaps = [
        {"ts": "2026-08-15T10:00:00Z", "timing": "T-6h",
         "odds_1x2": {"home": 2.4, "draw": 3.2, "away": 3.0}},
        {"ts": "2026-08-15T15:44:00Z", "timing": "T-15m",
         "odds_1x2": {"home": 2.2, "draw": 3.3, "away": 3.1}},
        {"ts": "2026-08-15T16:10:00Z", "timing": "T-0h",
         "odds_1x2": {"home": 1.9, "draw": 3.5, "away": 3.4}},
    ]
    sig = movement_signal(snaps, min_snapshots=3, steam_threshold_pct=2.0)
    assert sig["usable"] is True
    assert sig["n"] == 3
    # live point pushed home further -> steam home
    assert sig["steam_side"] == "home"
    assert sig["drift_pct"]["home"] > 5.0


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
