"""Plan B tests: movement signal + accuracy."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.movement import movement_accuracy, movement_signal
from agents.football.prediction_log import append_odds_snapshot, append_snapshot, settle


def _snaps():
    return [
        {"ts": "2026-08-15T00:00:00", "timing": "T-3h",
         "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "2026-08-15T01:00:00", "timing": "T-2h",
         "odds_1x2": {"home": 2.1, "draw": 3.4, "away": 3.6}},
        {"ts": "2026-08-15T02:00:00", "timing": "T-1h",
         "odds_1x2": {"home": 2.2, "draw": 3.3, "away": 3.2}},
    ]


def test_steam_detects_shortening_side():
    sig = movement_signal(_snaps(), model_side="away", min_snapshots=3)
    assert sig["usable"] is True
    assert sig["steam_side"] == "away"
    assert sig["drift_pct"]["away"] > 0
    assert sig["steam_pct"] >= 2.0


def test_agreement_matches_model_side():
    assert movement_signal(_snaps(), model_side="away", min_snapshots=3)["agreement"] == 1.0
    assert movement_signal(_snaps(), model_side="home", min_snapshots=3)["agreement"] == 0.0


def test_below_min_snapshots_unusable():
    sig = movement_signal(_snaps()[:2], model_side="away", min_snapshots=3)
    assert sig["usable"] is False


def test_no_clear_move_gives_neutral():
    flat = [
        {"ts": "t1", "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "t2", "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "t3", "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
    ]
    sig = movement_signal(flat, model_side="away", min_snapshots=3)
    assert sig["steam_side"] is None
    assert sig["agreement"] == 0.5


def test_time_decay_emphasises_late_moves():
    # A big move far from kickoff (T-24h -> T-6h) followed by a flat late
    # curve: plain first->last drift sees the full move, the decayed drift
    # (tau=1h) nearly cancels it because the informative window is empty.
    snaps = [
        {"ts": "t0", "timing": "T-24h",
         "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "t1", "timing": "T-6h",
         "odds_1x2": {"home": 1.5, "draw": 3.5, "away": 5.0}},
        {"ts": "t2", "timing": "T-0h",
         "odds_1x2": {"home": 1.5, "draw": 3.5, "away": 5.0}},
    ]
    plain = movement_signal(snaps, min_snapshots=3)
    decayed = movement_signal(snaps, min_snapshots=3, time_decay_tau=1.0)
    assert plain["drift_pct"]["home"] > 0
    assert abs(decayed["drift_pct"]["home"]) < abs(plain["drift_pct"]["home"])


def test_kickoff_stale_guard_excludes_inplay_rows():
    """Stale-guard: a row captured at/after kickoff (in-play T-0h capture) is
    excluded, so the drift's last point stays pre-match. Pre-match curve:
    away shortens 4.0 -> 3.8 (steam away). The in-play row shorts home to
    1.7 -- without the guard it flips the steam side to home."""
    kickoff = "2026-08-15T20:00:00Z"
    snaps = [
        {"ts": "2026-08-15T14:00:00", "timing": "T-6h",
         "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "2026-08-15T17:00:00", "timing": "T-3h",
         "odds_1x2": {"home": 2.05, "draw": 3.4, "away": 3.9}},
        {"ts": "2026-08-15T19:00:00", "timing": "T-1h",
         "odds_1x2": {"home": 2.1, "draw": 3.4, "away": 3.8}},
        # in-play capture: ts AFTER kickoff, labeled T-0h by the poll
        {"ts": "2026-08-15T20:30:00", "timing": "T-0h",
         "odds_1x2": {"home": 1.7, "draw": 3.1, "away": 5.2}},
    ]
    guarded = movement_signal(snaps, min_snapshots=3, kickoff=kickoff)
    unguarded = movement_signal(snaps, min_snapshots=3)
    # guard: pre-match drift only -> away is the steam side
    assert guarded["usable"] is True
    assert guarded["steam_side"] == "away"
    # without the guard the in-play row fabricates a home steam side
    assert unguarded["steam_side"] == "home"


def test_kickoff_stale_guard_all_inplay_rows_unusable():
    """Stale-guard: when EVERY row is at/after kickoff there are no valid
    pre-match points left -> unusable (never scores a live curve as movement)."""
    kickoff = "2026-08-15T20:00:00Z"
    snaps = [
        {"ts": "2026-08-15T20:30:00", "timing": "T-0h",
         "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "2026-08-15T21:00:00", "timing": "T-0h",
         "odds_1x2": {"home": 2.2, "draw": 3.4, "away": 3.5}},
    ]
    sig = movement_signal(snaps, min_snapshots=2, kickoff=kickoff)
    assert sig["usable"] is False
    assert sig["reason"] == "kurang dua titik harga valid"


def test_kickoff_stale_guard_unparseable_ts_kept():
    """Stale-guard: unparseable ts/kickoff cannot disprove pre-match -> rows
    are kept (same 'cannot disprove' rule as the G5 league-window filter)."""
    snaps = [
        {"ts": "t0", "timing": "T-6h",
         "odds_1x2": {"home": 2.0, "draw": 3.5, "away": 4.0}},
        {"ts": "t1", "timing": "T-1h",
         "odds_1x2": {"home": 2.1, "draw": 3.4, "away": 3.6}},
        {"ts": "t2", "timing": "T-0h",
         "odds_1x2": {"home": 2.2, "draw": 3.3, "away": 3.2}},
    ]
    sig = movement_signal(snaps, min_snapshots=3, kickoff="2026-08-15T20:00:00Z")
    assert sig["usable"] is True
    assert sig["steam_side"] == "away"


def test_movement_accuracy_counts_steam_hits(tmp_path):
    path = tmp_path / "pred.jsonl"
    # Future kickoff: the odds-snapshot rows below are captured "now" and the
    # stale-guard must keep them (they are genuinely pre-match); a kickoff in
    # the past would correctly filter every row out.
    kickoff = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    mid = f"EPL||A||B||{kickoff}"
    append_snapshot(
        path, match_id=mid, league="EPL", home="A", away="B",
        kickoff=kickoff,
        prob={"home": 0.4, "draw": 0.3, "away": 0.3},
        odds={"home": 2.0, "draw": 3.5, "away": 4.0},
        edge=None, confidence=None, signal=None, calibration=None,
        model_version=None, input_hash=None,
        best_pick={"selection": "Home Win", "market": "1X2"}, sources=[],
    )
    for ts, away in (("T-3h", 4.0), ("T-2h", 3.6), ("T-1h", 3.2)):
        append_odds_snapshot(
            path, match_id=mid, timing=ts,
            odds={"home": 2.0, "draw": 3.5, "away": away},
        )
    settle(path, match_id=mid, home_goals=0, away_goals=1)  # away wins
    acc = movement_accuracy(path, min_snapshots=3)
    assert acc["n"] == 1
    assert acc["steam_hit_rate"] == 1.0  # steam away == outcome away


def test_decide_movement_component():
    from agents.football.decision import build_candidates, decide

    model_probs = {"1x2": {"home": 0.50, "draw": 0.26, "away": 0.24}}
    consensus = {"home": 1.80, "draw": 3.80, "away": 4.50}
    cands = build_candidates(
        model_probs=model_probs, consensus_odds=consensus,
        market_totals={}, independent=True,
    )
    d1 = decide(
        cands, model_agreement=0.8, calibration_quality=0.9,
        calibration_samples=500, completeness=0.8, bookmakers_count=10,
        movement=1.0,
    )
    assert d1["score_breakdown"]["top"]["components"]["movement"] == 1.0
    d0 = decide(
        cands, model_agreement=0.8, calibration_quality=0.9,
        calibration_samples=500, completeness=0.8, bookmakers_count=10,
        movement=0.0,
    )
    assert d0["score_breakdown"]["top"]["components"]["movement"] == 0.0


def test_run_decision_engine_passes_movement_through():
    from agents.football.analyse import run_decision_engine

    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    totals = {}
    cfg = {"models": {"decision": {"min_bookmakers": 3}}, "cache_ttl_seconds": {}}
    mv = {"usable": True, "agreement": 0.0, "steam_side": "away"}
    d = run_decision_engine(None, consensus, totals, True, 4, cfg, movement=mv)
    assert d["movement"] == mv
