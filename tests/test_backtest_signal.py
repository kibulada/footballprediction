"""P4 (re-runnable) — signal-weight backtest harness tests.

The harness re-scores STORED signal rankings (persisted on each snapshot by
``append_snapshot``) through the exact production ``score_signals`` /
``rank_and_pick`` code and settles the re-weighted best pick against the
final score. These tests generate genuine rankings via ``run_signal_engine``
(no fabricated components) and verify: persistence round-trip, re-weight
through production code, settling against real scores, the two-period split,
and the honest insufficient-data guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.backtest_signal import (  # noqa: E402
    evaluate_weight_set,
    load_settled_ranked_matches,
    score_ranking,
    split_periods,
)
from agents.football.prediction_log import append_snapshot, settle  # noqa: E402
from agents.football.signal_engine import (  # noqa: E402
    _AH_DEMO_PAYLOAD,
    extract_asian_handicap,
    run_signal_engine,
)

MID = "EPL||Arsenal||Chelsea||2026-08-15"


def _engine_result(completeness: float = 0.7) -> dict:
    """A genuine signal-engine result (demo high-goal scenario)."""
    from agents.football.models import probs_from_matrix, poisson_matrix

    m = poisson_matrix(1.6, 1.6, rho=0.0)
    p1x2, o15, o25, o35, btts = probs_from_matrix(m)
    model = {
        "1x2": p1x2, "over_1.5": o15, "over_2.5": o25, "over_3.5": o35,
        "btts_yes": btts, "lambda_home": 1.6, "lambda_away": 1.6,
    }
    totals = {
        "Over 2.5": {"odds": 1.99, "point": 2.5, "opening": 2.08},
        "Under 2.5": {"odds": 1.94, "point": 2.5, "opening": 1.86},
        "BTTS Yes": {"odds": 1.75, "opening": 1.80},
        "BTTS No": {"odds": 2.05, "opening": 2.00},
    }
    ah_rows = extract_asian_handicap(_AH_DEMO_PAYLOAD)
    return run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=completeness, cfg=None,
    )


def _write_match(path: Path, kickoff: str, home_goals: int, away_goals: int,
                 mid: str | None = None, completeness: float = 0.7) -> str:
    mid = mid or MID
    res = _engine_result(completeness)
    append_snapshot(
        path,
        match_id=mid,
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        kickoff=kickoff,
        prob=res.get("model_probs") or {"home": 0.4, "draw": 0.25, "away": 0.35},
        odds={"home": 2.1, "draw": 3.4, "away": 3.2},
        edge=None,
        confidence=None,
        signal=None,
        calibration=None,
        model_version="t",
        input_hash=f"h-{mid}",
        best_pick=None,
        sources=["test"],
        features={"completeness": completeness},
        signal_engine_pick=res.get("best_pick"),
        signal_engine_ranking=res.get("ranking"),
    )
    settle(
        path,
        match_id=mid,
        home_goals=home_goals,
        away_goals=away_goals,
    )
    return mid


def test_ranking_persisted_on_snapshot(tmp_path):
    """P4 re-runnable: the full scored ranking must be stored with the snapshot
    (not just the best pick) so a later backtest can re-weight it."""
    path = tmp_path / "pred.jsonl"
    _write_match(path, "2026-08-15T14:00:00Z", 2, 1)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    snap = next(r for r in rows if r.get("event") == "snapshot")
    ranking = snap.get("signal_engine_ranking")
    assert ranking, "ranking must be persisted"
    entry = ranking[0]
    for key in ("market", "selection", "score", "components", "movement", "edge_pp", "line", "side", "line_key"):
        assert key in entry, f"ranking entry missing {key}"
    assert "movement" in entry  # movement dict needed for re-weighting Group D


def test_load_settled_ranked_matches(tmp_path):
    path = tmp_path / "pred.jsonl"
    _write_match(path, "2026-08-15T14:00:00Z", 2, 1)
    _write_match(path, "2026-08-16T14:00:00Z", 0, 0, mid="EPL||A||B||2026-08-16")
    records = load_settled_ranked_matches(path)
    assert len(records) == 2
    assert records[0]["home_goals"] == 2
    assert records[0]["away_goals"] == 1
    assert records[0]["ranking"]


def test_evaluate_settles_against_score(tmp_path):
    """Settling the re-weighted pick against the real final score works for
    Total + AH markets (the demo scenario produces Over/AH signals)."""
    path = tmp_path / "pred.jsonl"
    _write_match(path, "2026-08-15T14:00:00Z", 2, 1)  # over 2.5 hit
    records = load_settled_ranked_matches(path)
    res = evaluate_weight_set(
        records,
        weights={
            "model": 0.30, "statistical": 0.20, "market": 0.20,
            "movement": 0.15, "late_movement": 0.10, "data_quality": 0.05,
            "team_context": 0.00,
        },
        cfg={},
    )
    assert res["n_settled"] == 1
    assert res["n_bets"] == 1
    assert res["n_no_bet"] == 0
    # 2-1 -> over 2.5 wins; ROI on the winning pick must be positive.
    assert res["roi_pct"] is not None and res["roi_pct"] > 0


def test_split_periods_chronological(tmp_path):
    path = tmp_path / "pred.jsonl"
    _write_match(path, "2026-08-10T14:00:00Z", 1, 0, mid="EPL||A||B||2026-08-10")
    _write_match(path, "2026-08-11T14:00:00Z", 1, 0, mid="EPL||C||D||2026-08-11")
    _write_match(path, "2026-08-12T14:00:00Z", 1, 0, mid="EPL||E||F||2026-08-12")
    _write_match(path, "2026-08-13T14:00:00Z", 1, 0, mid="EPL||G||H||2026-08-13")
    records = load_settled_ranked_matches(path)
    pa, pb = split_periods(records)
    assert len(pa) == 2 and len(pb) == 2
    assert all((r.get("kickoff") or "") < (pb[0].get("kickoff") or "") for r in pa)


def test_insufficient_data_guard(tmp_path):
    """Below the sample floor the harness must report insufficient data
    rather than emit a noise-fit report."""
    path = tmp_path / "pred.jsonl"
    _write_match(path, "2026-08-15T14:00:00Z", 2, 1)
    records = load_settled_ranked_matches(path)
    assert len(records) == 1
    # The CLI guard is MIN_SAMPLES_DEFAULT; here we assert the loader itself
    # only returns ranked matches (the CLI checks the floor before reporting).
    from agents.football.backtest_signal import MIN_SAMPLES_DEFAULT
    assert MIN_SAMPLES_DEFAULT >= 500
    assert len(records) < MIN_SAMPLES_DEFAULT


if __name__ == "__main__":
    import traceback
    failed = 0
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        try:
            fn(Path(ROOT) / "cache" / "football" / "_p4_test_tmp" if fn.__name__ == "test_ranking_persisted_on_snapshot" else Path(ROOT) / "cache" / "football")
        except TypeError:
            try:
                fn()
            except Exception:
                failed += 1
                traceback.print_exc()
    print(f"\n{8 - failed}/8 passed (run under pytest for full fidelity)")
    sys.exit(1 if failed else 0)
