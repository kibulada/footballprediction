"""Tests for prediction_log.py (append-only JSONL prediction log)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.prediction_log import (
    append_odds_snapshot,
    append_snapshot,
    compute_stats,
    list_odds_snapshots,
    list_unsettled,
    make_match_id,
    odds_snapshots_by_match,
    settle,
    similar_signal_stats,
)

MID = "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z"


def _snapshot(path: Path, *, prob=None, odds=None, match_id: str = MID,
              confidence: float | None = 0.6, signal: int | None = 60,
              edge=None, kickoff: str = "2026-08-15T14:00:00Z",
              entities: dict | None = None,
              home: str = "Arsenal", away: str = "Chelsea",
              league: str = "EPL") -> None:
    append_snapshot(
        path,
        match_id=match_id,
        league=league,
        home=home,
        away=away,
        kickoff=kickoff,
        prob=prob if prob is not None else {"home": 0.55, "draw": 0.25, "away": 0.20},
        odds=odds if odds is not None else {"home": 1.8, "draw": 3.6, "away": 4.4},
        edge=edge if edge is not None else {"home": 2.1, "draw": -1.0, "away": -3.5},
        confidence=confidence,
        signal=signal,
        calibration={"quality": 0.9, "ece": 0.013, "samples": 4560},
        model_version="0.1.0-elo-poisson",
        input_hash="abc123",
        best_pick={"selection": "Home Win", "market": "1X2"},
        sources=["football_data"],
        entities=entities,
    )


def test_make_match_id_stable():
    # F4 (2026-08-18): names that are a word inside EXACTLY ONE canonical
    # club in the league resolve to that canonical ("Arsenal" ->
    # "Arsenal FC", "Chelsea" -> "Chelsea FC"), so the odds poll and the
    # analyse run can never split one match across two match_ids (the
    # "US Lecce" vs "Lecce" split-identity bug).
    assert make_match_id("EPL", "Arsenal", "Chelsea", None) == "EPL||Arsenal FC||Chelsea FC||"
    assert make_match_id("EPL", "Arsenal", "Chelsea", "X") == "EPL||Arsenal FC||Chelsea FC||X"


def test_snapshot_persists_final_decision(tmp_path):
    """Observability fix (2026-08-17): the decision engine's ACTUAL pick
    (market, selection, model_prob, market_odds, edge_pp, ev, n_bucket,
    pick_status) must be logged -- the tier label alone cannot tell a
    settled-match scorer which selection the engine really chose.
    """
    path = tmp_path / "pred.jsonl"
    fd = {
        "market": "Total", "selection": "Under 2.5", "model_prob": 0.62,
        "market_odds": 1.85, "implied_prob": 0.54, "edge_pp": 8.0,
        "ev": 0.05, "n_bucket": 665, "pick_status": "VALID",
    }
    append_snapshot(
        path, match_id=MID, league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge={"home": 2.0, "draw": -1.0, "away": -3.0},
        confidence=0.7, signal=70,
        calibration={"quality": 0.9, "ece": 0.013, "samples": 4560},
        model_version="0.1.0", input_hash="abc",
        best_pick={"selection": "Under 2.5"}, sources=["nowgoal"],
        decision_type="GOOD",
        final_decision=fd,
    )
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    snap = lines[0]
    assert snap["decision_type"] == "GOOD"
    assert snap["final_decision"] == fd
    # And a NO BET row records final_decision = None.
    path2 = tmp_path / "pred2.jsonl"
    append_snapshot(
        path2, match_id=MID, league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge={"home": -2.0, "draw": -1.0, "away": -3.0},
        confidence=0.4, signal=30,
        calibration={"quality": 0.9, "ece": 0.013, "samples": 4560},
        model_version="0.1.0", input_hash="def",
        best_pick=None, sources=["nowgoal"],
        decision_type="NO BET",
        final_decision=None,
    )
    snap2 = [json.loads(l) for l in path2.read_text(encoding="utf-8").splitlines() if l.strip()][0]
    assert snap2["final_decision"] is None


def test_snapshot_and_settle_roundtrip(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    assert settle(path, match_id=MID, home_goals=2, away_goals=1) is True
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [l["event"] for l in lines] == ["snapshot", "settle"]
    st = compute_stats(path)
    assert st["n_snapshots"] == 1
    assert st["n_settled"] == 1
    assert st["n_predicted"] == 1
    assert st["hit_rate"] == 1.0  # best pick was home, outcome home
    assert st["avg_logloss"] is not None
    # flat-stake ROI at consensus odds: home won at 1.8 -> +0.8
    assert st["roi"] == 0.8
    assert st["n_bets"] == 1


def test_settle_requires_snapshot(tmp_path):
    path = tmp_path / "pred.jsonl"
    assert settle(path, match_id=MID, home_goals=1, away_goals=1) is False
    assert not path.exists()


def test_settle_updates_outcome_and_hit(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    settle(path, match_id=MID, home_goals=0, away_goals=2)
    st = compute_stats(path)
    assert st["n_settled"] == 1
    assert st["hit_rate"] == 0.0  # pick home, outcome away


def test_stats_without_settlements(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    st = compute_stats(path)
    assert st["n_snapshots"] == 1
    assert st["n_settled"] == 0
    assert st["hit_rate"] is None
    assert st["roi"] is None


def test_list_unsettled_tracks_settlements(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    _snapshot(path, match_id="EPL||A||B||T")
    assert len(list_unsettled(path)) == 2
    settle(path, match_id=MID, home_goals=1, away_goals=0)
    unsettled = list_unsettled(path)
    assert len(unsettled) == 1
    assert unsettled[0]["match_id"] == "EPL||A||B||T"


def test_append_only_immutability(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    before = path.read_text(encoding="utf-8")
    _snapshot(path, match_id="EPL||A||B||T")  # second snapshot, same file
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before)  # first line untouched, only appended


def test_clv_uses_closing_odds(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path, odds={"home": 1.8, "draw": 3.6, "away": 4.4})
    settle(path, match_id=MID, home_goals=2, away_goals=1,
           closing_odds={"home": 1.72, "draw": 3.7, "away": 4.6})
    st = compute_stats(path)
    # CLV = model_prob_home * closing_home - 1 = 0.55 * 1.72 - 1 = -0.054
    assert st["n_clv"] == 1
    assert abs(st["clv_pct"] - (-5.4)) < 0.01


def test_empty_prob_snapshot_not_counted_in_hit_rate(tmp_path):
    """Snapshots without a 1X2 prediction (no model output) must not be
    counted as misses in hit_rate -- missing data stays missing."""
    path = tmp_path / "pred.jsonl"
    _snapshot(path, prob={})
    settle(path, match_id=MID, home_goals=2, away_goals=1)
    st = compute_stats(path)
    assert st["n_settled"] == 1
    assert st["n_predicted"] == 0
    assert st["hit_rate"] is None
    assert st["avg_logloss"] is None


def test_cli_stats_and_settle_with_subcommand_file(tmp_path, capsys):
    from agents.football.prediction_log import main

    path = tmp_path / "p.jsonl"
    _snapshot(path)
    rc = main(["stats", "--file", str(path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "snapshots" in out and "settled" in out

    rc2 = main(["settle", "--match-id", MID, "--home-goals", "2",
                "--away-goals", "1", "--file", str(path)])
    assert rc2 == 0
    assert capsys.readouterr().out.startswith("settled")


def test_roi_skipped_below_edge_threshold(tmp_path):
    path = tmp_path / "pred.jsonl"
    # edge home = 0.55 - (1/1.8)/(sum) ; make edge tiny by tight odds
    _snapshot(path, prob={"home": 0.52, "draw": 0.26, "away": 0.22},
              odds={"home": 2.0, "draw": 3.6, "away": 4.2})
    settle(path, match_id=MID, home_goals=2, away_goals=1)
    st = compute_stats(path, edge_threshold=0.10)  # threshold too high
    assert st["n_bets"] == 0
    assert st["roi"] is None


def test_settle_deduped_per_canonical_match(tmp_path):
    """Settle + evaluation count each real-world match ONCE, even when many
    snapshots exist for it (repeated queries / pre-Fix-1 match_id variants
    like 'Rio Ave FC' vs 'Rio Ave')."""
    path = tmp_path / "pred.jsonl"
    # Three snapshots for the SAME fixture: two match_id variants + one repeat.
    _snapshot(path, match_id="EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z")
    _snapshot(path, match_id="EPL||Rio Ave||FC Porto||2026-08-15T19:30:00Z")
    _snapshot(path, match_id="EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z")
    # Each gets its own settle (pre-dedupe behaviour) ...
    for mid in ("EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z",
                "EPL||Rio Ave||FC Porto||2026-08-15T19:30:00Z",
                "EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z"):
        settle(path, match_id=mid, home_goals=0, away_goals=2)
    from agents.football.prediction_log import dedupe_settles, _match_dedupe_key
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len([r for r in rows if r.get("event") == "settle"]) == 3
    report = dedupe_settles(path)
    assert report["removed"] == 2 and report["kept"] == 1
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    settles = [r for r in rows if r.get("event") == "settle"]
    assert len(settles) == 1
    assert len({_match_dedupe_key(r) for r in settles}) == 1
    # Snapshot rows are untouched by cleanup.
    snaps = [r for r in rows if r.get("event") == "snapshot"]
    assert len(snaps) == 3


def test_stats_count_each_match_once(tmp_path):
    """compute_stats must not double-count a match with repeated snapshots."""
    path = tmp_path / "pred.jsonl"
    _snapshot(path, match_id="EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z")
    _snapshot(path, match_id="EPL||Rio Ave||FC Porto||2026-08-15T19:30:00Z")
    for mid in ("EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z",
                "EPL||Rio Ave||FC Porto||2026-08-15T19:30:00Z"):
        settle(path, match_id=mid, home_goals=0, away_goals=2)
    st = compute_stats(path)
    # n_settled comes from _settled_records, deduped per canonical match:
    # two snapshots for the same fixture -> ONE settled record, not two.
    assert st["n_settled"] == 1


def test_best_pick_evaluation(tmp_path):
    """BEST PICK vs settled result: stored signal_engine_pick is settled via
    the production settle_signal and aggregated per market with ROI."""
    from agents.football.prediction_log import best_pick_evaluation

    path = tmp_path / "pred.jsonl"
    # Under 2.5 pick, final 2-1 (over) -> loss. odds 1.94.
    append_snapshot(
        path, match_id=MID, league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.4, "draw": 0.25, "away": 0.35},
        odds={"home": 2.1, "draw": 3.4, "away": 3.2},
        edge=None, confidence=None, signal=None, calibration=None,
        model_version="t", input_hash="h", best_pick=None, sources=["test"],
        features={"completeness": 0.7},
        signal_engine_pick={
            "decision": "BEST PICK", "market": "Total",
            "selection": "Under 2.5", "score": 0.63,
            "confidence": "MEDIUM", "line": None, "side": None,
            "market_odds": 1.94,
        },
        signal_engine_ranking=[],
    )
    settle(path, match_id=MID, home_goals=2, away_goals=1)
    ev = best_pick_evaluation(path)
    assert ev["n"] == 1
    assert ev["markets"]["Total"]["n"] == 1
    assert ev["markets"]["Total"]["wins"] == 0
    assert ev["markets"]["Total"]["losses"] == 1
    assert ev["markets"]["Total"]["roi_pct"] == -100.0  # staked 1, lost all
    assert ev["picks"][0]["result"] == "loss"
    assert ev["picks"][0]["odds"] == 1.94
    # No stored pick -> counted as n/a, not fabricated.
    path2 = tmp_path / "pred2.jsonl"
    append_snapshot(
        path2, match_id=MID, league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.5, "draw": 0.25, "away": 0.25},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge=None, confidence=None, signal=None, calibration=None,
        model_version="t", input_hash="h2", best_pick=None, sources=["test"],
        features={"completeness": 0.7},
    )
    settle(path2, match_id=MID, home_goals=2, away_goals=1)
    assert best_pick_evaluation(path2)["n"] == 0


def test_best_pick_evaluation_ah_quarter_line(tmp_path):
    """AH picks settle with quarter-line semantics (half win on the line)."""
    from agents.football.prediction_log import best_pick_evaluation

    path = tmp_path / "pred.jsonl"
    # Home -0.25 @ 1.95, final 1-1 -> quarter line splits into (-0.5, 0.0);
    # the -0.5 leg loses, the 0.0 leg pushes -> half loss, return 0.25.
    append_snapshot(
        path, match_id=MID, league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.4, "draw": 0.25, "away": 0.35},
        odds={"home": 2.1, "draw": 3.4, "away": 3.2},
        edge=None, confidence=None, signal=None, calibration=None,
        model_version="t", input_hash="h", best_pick=None, sources=["test"],
        features={"completeness": 0.7},
        signal_engine_pick={
            "decision": "BEST PICK", "market": "Asian Handicap",
            "selection": "Home -0.25", "score": 0.76,
            "confidence": "HIGH", "line": -0.25, "side": "home",
            "market_odds": 1.95,
        },
        signal_engine_ranking=[],
    )
    settle(path, match_id=MID, home_goals=1, away_goals=1)
    ev = best_pick_evaluation(path)
    assert ev["picks"][0]["result"] == "half_loss"
    # Production ah_return: 0.25 for this line/score -> ROI = 0.25*1.95-1.
    assert abs(ev["picks"][0]["roi"] - (0.25 * 1.95 - 1.0)) < 0.01


def test_calibration_pairs_deduped_per_match(tmp_path):
    """Calibration re-fit must not overweight a match queried repeatedly:
    the same fixture with two snapshot variants contributes ONE set of
    (p, outcome) pairs, from its newest snapshot."""
    from agents.football.prediction_log import calibration_pairs

    path = tmp_path / "pred.jsonl"
    _snapshot(path, match_id="EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z")
    _snapshot(path, match_id="EPL||Rio Ave||FC Porto||2026-08-15T19:30:00Z")
    for mid in ("EPL||Rio Ave FC||FC Porto||2026-08-15T19:30:00Z",
                "EPL||Rio Ave||FC Porto||2026-08-15T19:30:00Z"):
        settle(path, match_id=mid, home_goals=0, away_goals=2)
    pairs = calibration_pairs(path)
    # One match -> 3 pairs (home/draw/away), not 6.
    assert len(pairs) == 3


def test_snapshot_stores_features(tmp_path):
    """PHASE 1: pre-match features (Elo, lambda, form, completeness) are
    persisted so similar-signal analysis can explain a prediction."""
    path = tmp_path / "pred.jsonl"
    append_snapshot(
        path,
        match_id=MID,
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
        sources=["sofascore"],
        features={
            "elo_home": 1700.0,
            "elo_away": 1600.0,
            "lambda_home": 1.8,
            "lambda_away": 1.1,
            "attack_home": 1.6,
            "defense_home": 1.1,
            "form_home": "WWDLW",
            "completeness": 0.7,
        },
    )
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    f = rows[0]["features"]
    assert f["elo_home"] == 1700.0
    assert f["lambda_home"] == 1.8
    assert f["form_home"] == "WWDLW"
    assert f["completeness"] == 0.7


def test_snapshot_stores_context_data(tmp_path):
    """P5: lineups / missing players / coaches are logged as STRUCTURED
    context on every snapshot so a historical record accumulates. Context-
    only by design: the no-OOS-evidence rule keeps them out of the model
    until a backtest validates them."""
    path = tmp_path / "pred.jsonl"
    append_snapshot(
        path,
        match_id=MID,
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge={"home": 2.1},
        confidence=0.6,
        signal=60,
        calibration=None,
        model_version="0.1.0",
        input_hash="h2",
        best_pick=None,
        sources=["flashscore"],
        context_data={
            "lineups": {
                "status": "predicted",
                "formations": ["4-3-3", "4-2-3-1"],
                "home_count": 11,
                "away_count": 11,
                "home": [{"number": "1", "name": "A GK"}],
                "away": [],
                "source": "flashscore_lineups",
            },
            "missing_players": {
                "home": {"missing": ["Saka"], "unsure": ["Rice"]},
                "away": {"missing": [], "unsure": []},
            },
            "coaches": {"home": ["Arteta"], "away": ["Maresca"]},
        },
    )
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cd = rows[0]["context_data"]
    assert cd["lineups"]["status"] == "predicted"
    assert cd["lineups"]["formations"] == ["4-3-3", "4-2-3-1"]
    assert cd["lineups"]["home"][0]["name"] == "A GK"
    assert cd["missing_players"]["home"]["missing"] == ["Saka"]
    assert cd["missing_players"]["home"]["unsure"] == ["Rice"]
    assert cd["coaches"]["home"] == ["Arteta"]
    # Absent context -> explicit None, not a missing key (auditable).
    path2 = tmp_path / "pred2.jsonl"
    append_snapshot(
        path2,
        match_id=MID,
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob={"home": 0.55, "draw": 0.25, "away": 0.20},
        odds={"home": 1.8, "draw": 3.6, "away": 4.4},
        edge=None,
        confidence=None,
        signal=None,
        calibration=None,
        model_version=None,
        input_hash=None,
        best_pick=None,
        sources=[],
    )
    rows2 = [json.loads(l) for l in path2.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows2[0]["context_data"] is None


def test_max_drawdown_and_sharpe(tmp_path):
    """PHASE 7: betting risk metrics appear once bets exist."""
    path = tmp_path / "pred.jsonl"
    # Two settled bets: one win (+0.8) then one loss (-1.0) -> net -0.2,
    # peak +0.8, drawdown (0.0-(-0.2))/0.8 = -0.25 at the trough.
    _snapshot(path, match_id=MID)
    settle(path, match_id=MID, home_goals=2, away_goals=1)
    _snapshot(path, match_id="EPL||A||B||T")
    settle(path, match_id="EPL||A||B||T", home_goals=0, away_goals=2)
    st = compute_stats(path)
    assert st["n_bets"] == 2
    assert st["max_drawdown"] is not None
    assert st["max_drawdown"] < 0
    assert st["sharpe"] is not None
    # buckets include the confidence label of the snapshot (0.6 -> MEDIUM)
    assert "MEDIUM" in st["by_confidence"]
    assert st["by_confidence"]["MEDIUM"]["n"] == 2


def test_similar_signal_bucket_lookup(tmp_path):
    """PHASE 4/8: similar-signal stats cluster settled snapshots by
    confidence/edge and expose the matching bucket."""
    path = tmp_path / "pred.jsonl"
    for i in range(6):
        mid = f"EPL||T{i}||U||K"
        _snapshot(path, match_id=mid, confidence=0.8, signal=80,
                  prob={"home": 0.6, "draw": 0.25, "away": 0.15},
                  odds={"home": 2.2, "draw": 3.8, "away": 5.0})
        settle(path, match_id=mid, home_goals=2, away_goals=1)
    # odds 2.2 with model 0.6 -> margin-free implied ~0.495 -> edge ~10.5%
    res = similar_signal_stats(path, confidence=0.8, edge_pct=12.0, min_bucket_n=5)
    assert res["n_buckets"] >= 1
    m = res["matching"]
    assert m is not None
    assert m["confidence"] == "HIGH"
    assert m["sufficient_sample"] is True
    assert m["n"] >= 5
    # every bucket carries hit rate / ROI / CLV so the bot can show history
    for b in res["table"].values():
        assert "hit_rate" in b and "roi" in b and "clv_pct" in b


def test_similar_signal_insufficient_sample(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path, match_id=MID, confidence=0.8, signal=80)
    settle(path, match_id=MID, home_goals=1, away_goals=1)
    # default odds 1.8 with model 0.55 -> edge ~2.6% -> bucket 0-5%
    res = similar_signal_stats(path, confidence=0.8, edge_pct=3.0, min_bucket_n=5)
    m = res["matching"]
    assert m is not None
    assert m["sufficient_sample"] is False
    assert m["n"] == 1


# ---- PHASE 32-33: timed odds snapshots & price CLV ----------------------


def test_odds_snapshot_requires_prediction_snapshot(tmp_path):
    path = tmp_path / "pred.jsonl"
    assert append_odds_snapshot(
        path, match_id=MID, timing="T-6h", odds={"home": 1.75, "draw": 3.7, "away": 4.5}
    ) is False
    assert not path.exists()


def test_odds_snapshot_roundtrip_and_immutable(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    ok = append_odds_snapshot(
        path, match_id=MID, timing="T-6h",
        odds={"home": 1.75, "draw": 3.7, "away": 4.5},
        bookmakers_count=14, sources=["the-odds-api"],
    )
    assert ok is True
    snaps = list_odds_snapshots(path, match_id=MID)
    assert len(snaps) == 1
    s = snaps[0]
    assert s["event"] == "odds_snapshot"
    assert s["timing"] == "T-6h"
    assert s["odds_1x2"]["home"] == 1.75
    assert s["bookmakers_count"] == 14
    assert s["sources"] == ["the-odds-api"]
    # append-only: adding another snapshot never edits the first line
    append_odds_snapshot(path, match_id=MID, timing="T-1h", odds={"home": 1.7, "draw": 3.7, "away": 4.6})
    before = path.read_text(encoding="utf-8")
    append_odds_snapshot(path, match_id=MID, timing="T-0h", odds={"home": 1.68, "draw": 3.7, "away": 4.7})
    after = path.read_text(encoding="utf-8")
    assert after.startswith(before)
    assert [x["timing"] for x in list_odds_snapshots(path, match_id=MID)] == ["T-6h", "T-1h", "T-0h"]


def test_odds_snapshots_by_match_groups(tmp_path):
    path = tmp_path / "pred.jsonl"
    _snapshot(path)
    _snapshot(path, match_id="EPL||A||B||T")
    append_odds_snapshot(path, match_id=MID, timing="T-24h", odds={"home": 1.9, "draw": 3.6, "away": 4.2})
    append_odds_snapshot(path, match_id=MID, timing="T-6h", odds={"home": 1.8, "draw": 3.6, "away": 4.4})
    append_odds_snapshot(path, match_id="EPL||A||B||T", timing="T-1h", odds={"home": 2.0, "draw": 3.5, "away": 4.0})
    by = odds_snapshots_by_match(path)
    assert set(by.keys()) == {MID, "EPL||A||B||T"}
    assert [x["timing"] for x in by[MID]] == ["T-24h", "T-6h"]


def test_price_clv_separated_from_model_clv(tmp_path):
    """PHASE 33: price CLV (closing/prediction - 1) is reported SEPARATELY
    from model CLV (P * closing - 1). Forecast quality != price quality."""
    path = tmp_path / "pred.jsonl"
    _snapshot(path, prob={"home": 0.55, "draw": 0.25, "away": 0.20},
              odds={"home": 2.0, "draw": 3.6, "away": 4.4})
    settle(path, match_id=MID, home_goals=2, away_goals=1,
           closing_odds={"home": 1.8, "draw": 3.7, "away": 4.6})
    st = compute_stats(path)
    assert st["n_clv"] == 1
    assert st["n_price_clv"] == 1
    # model CLV = 0.55 * 1.8 - 1 = -0.01 -> -1.0%
    assert abs(st["clv_pct"] - (-1.0)) < 0.01
    # price CLV = 1.8 / 2.0 - 1 = -0.10 -> -10% (line drifted against pick)
    assert abs(st["price_clv_pct"] - (-10.0)) < 0.01


def test_price_clv_by_timing(tmp_path):
    """CLV vs each captured timing: T-24h odds 1.9 -> close 1.8 = -5.3%;
    T-6h odds 1.8 -> close 1.8 = 0%. Kickoff is in the future so the
    appended snapshots (ts = now) are PRE-kickoff captures."""
    path = tmp_path / "pred.jsonl"
    _snapshot(path, prob={"home": 0.55, "draw": 0.25, "away": 0.20},
              odds={"home": 2.0, "draw": 3.6, "away": 4.4},
              kickoff="2099-01-01T14:00:00Z")
    append_odds_snapshot(path, match_id=MID, timing="T-24h", odds={"home": 1.9, "draw": 3.6, "away": 4.2})
    append_odds_snapshot(path, match_id=MID, timing="T-6h", odds={"home": 1.8, "draw": 3.6, "away": 4.4})
    settle(path, match_id=MID, home_goals=2, away_goals=1,
           closing_odds={"home": 1.8, "draw": 3.7, "away": 4.6})
    st = compute_stats(path)
    assert st["n_odds_snapshots"] == 2
    assert st["odds_snapshots_by_timing"] == {"T-24h": 1, "T-6h": 1}
    clv_t = st["clv_by_timing"]
    assert abs(clv_t["T-24h"] - (1.8 / 1.9 - 1.0) * 100) < 0.05
    assert abs(clv_t["T-6h"] - 0.0) < 0.05


def test_cli_odds_snapshot(tmp_path, capsys):
    from agents.football.prediction_log import main

    path = tmp_path / "p.jsonl"
    _snapshot(path)
    rc = main(["odds-snapshot", "--match-id", MID, "--timing", "T-1h",
               "--odds", "1.75,3.70,4.50", "--file", str(path)])
    assert rc == 0
    assert capsys.readouterr().out.startswith("odds snapshot T-1h tersimpan")
    rc2 = main(["odds-snapshot", "--match-id", MID, "--timing", "T-0h",
                "--odds", "1.70,3.70,4.60", "--file", str(path)])
    assert rc2 == 0
    st = compute_stats(path)
    assert st["n_odds_snapshots"] == 2


def test_cli_odds_snapshot_rejects_bad_odds(tmp_path, capsys):
    from agents.football.prediction_log import main

    path = tmp_path / "p.jsonl"
    _snapshot(path)
    rc = main(["odds-snapshot", "--match-id", MID, "--timing", "T-1h",
               "--odds", "1.75,3.70", "--file", str(path)])
    assert rc == 2
    assert "--odds harus 3 angka" in capsys.readouterr().err


# ---- Fase 2 anti-flap: identity lock (blueprint 2026-08-23) --------------

from datetime import datetime, timedelta, timezone

from agents.football.prediction_log import identity_lock_check


def _mid(league: str, home: str, away: str, kickoff: str) -> str:
    return make_match_id(league, home, away, kickoff)


def test_identity_lock_empty_log_is_open(tmp_path):
    path = tmp_path / "p.jsonl"
    assert identity_lock_check(
        path, match_id=_mid("EPL", "Arsenal", "Chelsea", "2026-08-23T14:00:00Z"),
        home="Arsenal", away="Chelsea",
    ) is None


def test_identity_lock_allows_repeated_query(tmp_path):
    """Query ulang match yang sama (pasangan identik) TIDAK dikunci."""
    path = tmp_path / "p.jsonl"
    mid = _mid("EPL", "Arsenal", "Chelsea", "2026-08-23T14:00:00Z")
    _snapshot(path, match_id=mid)
    assert identity_lock_check(
        path, match_id=mid, home="Arsenal", away="Chelsea",
    ) is None


def test_identity_lock_blocks_opponent_flip(tmp_path):
    """Kasus nyata Forest: lawan berubah Leeds -> Man Utd pada league+tanggal
    yang sama -> snapshot baru DITAHAN (resolver flip)."""
    path = tmp_path / "p.jsonl"
    first = _mid("EPL", "Forest", "Leeds", "2026-08-23T14:00:00Z")
    _snapshot(path, match_id=first, home="Forest", away="Leeds")
    flipped = _mid("EPL", "Forest", "Man Utd", "2026-08-23T14:00:00Z")
    verdict = identity_lock_check(
        path, match_id=flipped, home="Forest", away="Man Utd",
    )
    assert verdict and verdict["locked"] is True
    assert verdict["kind"] == "opponent_flip"
    assert verdict["conflict_match_id"] == first


def test_identity_lock_blocks_flip_from_away_side(tmp_path):
    """Sisi tetap = away (Troyes): lawan 'PSG' berubah -> Paris FC juga terkunci."""
    path = tmp_path / "p.jsonl"
    first = _mid("L1", "PSG", "Troyes", "2026-08-23T20:00:00Z")
    _snapshot(path, match_id=first, home="PSG", away="Troyes", league="L1")
    second = _mid("L1", "Paris FC", "Troyes", "2026-08-23T20:00:00Z")
    verdict = identity_lock_check(
        path, match_id=second, home="Paris FC", away="Troyes",
    )
    assert verdict and verdict["locked"] is True
    assert verdict["kind"] == "opponent_flip"


def test_identity_lock_ignores_unrelated_same_day_match(tmp_path):
    """Dua match BEDA di liga+tanggal sama (kedua sisi beda) tidak dikunci."""
    path = tmp_path / "p.jsonl"
    _snapshot(path, match_id=_mid("EPL", "Arsenal", "Chelsea", "2026-08-23T14:00:00Z"))
    assert identity_lock_check(
        path,
        match_id=_mid("EPL", "Liverpool", "Man City", "2026-08-23T16:30:00Z"),
        home="Liverpool", away="Man City",
    ) is None


def test_identity_lock_ignores_old_snapshots(tmp_path):
    """Snapshot > IDENTITY_LOCK_MAX_AGE_DAYS tidak mengunci (musim berganti)."""
    path = tmp_path / "p.jsonl"
    _snapshot(
        path, match_id=_mid("EPL", "Forest", "Leeds", "2026-01-10T14:00:00Z"),
        home="Forest", away="Leeds",
    )
    now_ts = (
        datetime.now(timezone.utc) + timedelta(days=IDENTITY_LOCK_MAX_AGE_DAYS + 2)
    ).isoformat()
    assert identity_lock_check(
        path,
        match_id=_mid("EPL", "Forest", "Man Utd", "2026-08-23T14:00:00Z"),
        home="Forest", away="Man Utd", now_ts=now_ts,
    ) is None


from agents.football.prediction_log import IDENTITY_LOCK_MAX_AGE_DAYS  # noqa: E402


def test_identity_lock_same_id_entity_contradiction(tmp_path):
    """Kasus Troyes/'PSG': match_id sama tapi canonical id satu sisi berubah
    (salah resolve Paris FC sbg PSG) -> ditahan via bukti id-level."""
    path = tmp_path / "p.jsonl"
    mid = _mid("FR1", "PSG", "Troyes", "2026-08-23T20:00:00Z")
    _snapshot(
        path, match_id=mid, home="PSG", away="Troyes", league="FR1",
        entities={
            "home": {"canonical_id": "paris-saint-germain", "name": "PSG"},
            "away": {"canonical_id": "troyes", "name": "Troyes"},
        },
    )
    verdict = identity_lock_check(
        path, match_id=mid, home="PSG", away="Troyes",
        entities={
            "home": {"canonical_id": "paris-fc", "name": "Paris FC"},
            "away": {"canonical_id": "troyes", "name": "Troyes"},
        },
    )
    assert verdict and verdict["locked"] is True
    assert verdict["kind"] == "same_id"


def test_identity_lock_same_id_consistent_entities_pass(tmp_path):
    path = tmp_path / "p.jsonl"
    mid = _mid("EPL", "Arsenal", "Chelsea", "2026-08-23T14:00:00Z")
    ents = {
        "home": {"canonical_id": "arsenal-fc", "name": "Arsenal"},
        "away": {"canonical_id": "chelsea-fc", "name": "Chelsea"},
    }
    _snapshot(path, match_id=mid, entities=ents)
    assert identity_lock_check(
        path, match_id=mid, home="Arsenal", away="Chelsea", entities=ents,
    ) is None


def test_identity_lock_spelling_variant_does_not_lock(tmp_path):
    """Varian ejaan provider ('Göztepe' vs 'Goztepe') tidak dianggap flip."""
    p = tmp_path / "tr.jsonl"
    mid1 = make_match_id("trs", "Goztepe", "Rizespor", "2026-08-23T17:00:00Z")
    _snapshot(p, match_id=mid1, home="Goztepe", away="Rizespor", league="trs")
    # nama tampilan beda aksen tapi match_id sama -> shape-1 pakai nama:
    # _identity_normalize melucuti aksen sehingga tidak dikunci.
    assert identity_lock_check(
        p, match_id=mid1, home="Göztepe", away="Rizespor",
    ) is None


def test_snapshot_accepts_entities_kwarg(tmp_path):
    """append_snapshot meneruskan field entities (dipakai identity lock)."""
    path = tmp_path / "p.jsonl"
    ents = {"home": {"canonical_id": "x"}, "away": {"canonical_id": "y"}}
    _snapshot(path, entities=ents)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["entities"] == ents


def test_identity_lock_flip_with_legacy_timestamp_id(tmp_path):
    """Baris pre-P1.2 membawa timestamp kickoff penuh di komponen ke-4 id;
    pre-filter murah (string ops) tetap menangkapnya via perbandingan
    date-only -- bukti restructure Opsi 3 tidak mengubah verdict."""
    path = tmp_path / "p.jsonl"
    mid = make_match_id("EPL", "Forest", "Leeds", "2026-08-23T14:00:00Z")
    parts = mid.split("||")
    legacy_id = "||".join(parts[:3] + ["2026-08-23T14:00:00Z"])
    _snapshot(path, match_id=legacy_id, home="Forest", away="Leeds")
    flipped = make_match_id("EPL", "Forest", "Man Utd", "2026-08-23T14:00:00Z")
    verdict = identity_lock_check(
        path, match_id=flipped, home="Forest", away="Man Utd",
    )
    assert verdict and verdict["locked"] is True
    assert verdict["kind"] == "opponent_flip"
    assert verdict["conflict_match_id"] == legacy_id


def test_identity_lock_cheap_prefilter_skips_other_dates(tmp_path):
    """Lawan beda tapi tanggal beda (jadwal wajar akhir pekan) tidak dikunci --
    pre-filter league+date membuangnya sebelum kanonisasi mahal."""
    path = tmp_path / "p.jsonl"
    _snapshot(
        path,
        match_id=make_match_id("EPL", "Forest", "Leeds", "2026-08-24T14:00:00Z"),
        home="Forest", away="Leeds",
    )
    assert identity_lock_check(
        path,
        match_id=make_match_id("EPL", "Forest", "Man Utd", "2026-08-23T14:00:00Z"),
        home="Forest", away="Man Utd",
    ) is None
