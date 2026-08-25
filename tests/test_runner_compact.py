"""Runner emit contract for the compact analyse summary.

The analyse command must emit ``render`` (the 5-7 line compact summary the
bot posts) AND ``render_full`` (the full report the 📋 Copy button serves).
Other modes carry no ``render_full``. Everything is mocked -- no network.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agents.football.runner as runner  # noqa: E402


def _fake_payload():
    return {
        "league": "EPL",
        "home": "Arsenal",
        "away": "Chelsea",
        "kickoff": None,
        "prediction": {
            "model_probs": {"1x2": {"home": 0.5, "draw": 0.28, "away": 0.22}},
            "data_completeness": 0.7,
        },
        "stats": {},
        "odds": {"consensus": {}, "has_odds": False, "totals": {}},
        "decision": {
            "decision_type": "NO BET",
            "final_decision": None,
            "explanation": "Tidak ada kandidat ber-odds dengan probabilitas model.",
            "edge_warnings": [],
            "model_disagreement": {"flag": False, "delta_pp": None},
            "evaluated": [],
        },
        "confidence": {
            "tier": "LOW", "tier_before_caps": "LOW", "caps_applied": [],
            "n_bucket": 0, "completeness_factor": 0.7, "pick_specific_confidence": 0.0,
        },
        "quota": {},
    }


def test_analyse_emit_carries_render_full(monkeypatch):
    import asyncio
    import os

    emitted: dict = {}

    async def fake_find_specific_match(**kwargs):
        return _fake_payload()

    def fake_emit(obj):
        emitted.update(obj)

    monkeypatch.setattr(runner, "find_specific_match", fake_find_specific_match)
    monkeypatch.setattr(runner, "_emit", fake_emit)
    monkeypatch.setattr(runner, "_arm_deadline", lambda seconds: None)
    monkeypatch.setattr(os, "_exit", lambda code: None)

    rc = runner.main(["analyse", "--league", "EPL", "--home", "Arsenal", "--away", "Chelsea"])
    assert rc is None  # os._exit patched away; main has no return after it
    assert "render" in emitted and "render_full" in emitted
    # main reply is the clean MARKET SIGNAL card (no data -> honest NO BET).
    # The Detail button serves the slightly richer (still debug-free) detail.
    assert emitted["render"]["title"] == "🔬 MATCH SIGNAL"
    assert emitted["render"]["body"].startswith("Arsenal vs Chelsea")
    assert "No actionable signal." in emitted["render"]["body"]
    assert "FINAL DECISION" not in emitted["render"]["body"]
    assert "🏆 BEST PICK" in emitted["render_full"]["body"]
    assert "FINAL DECISION" not in emitted["render_full"]["body"]


def _fake_best_payload():
    return {
        "league": "EPL",
        "league_key": "EPL",
        "date": "2026-08-12",
        "candidates": [
            {"home": "Arsenal", "away": "Chelsea", "kickoff": "2026-08-12T19:00:00Z",
             "signal": 80, "decision_type": "GOOD", "decision_score": 0.61,
             "has_odds": True, "bookmakers_count": 1},
        ],
        "winner": {
            "league": "EPL", "home": "Arsenal", "away": "Chelsea",
            "kickoff": "2026-08-12T19:00:00Z",
            "prediction": None,
            "stats": {"home_form": "W", "away_form": "W"},
            "odds": {"has_odds": True, "consensus": {"home": 1.6, "draw": 4.2, "away": 5.5},
                     "bookmakers_count": 1, "totals": {}},
            "picks": {},
            "decision": {
                "decision_type": "GOOD",
                "final_decision": {"market": "1X2", "selection": "Home Win",
                                   "model_prob": 0.6, "market_odds": 1.6,
                                   "edge_pp": 3.0, "ev": -0.04},
                "most_likely": None, "explanation": "x", "reasons": [],
                "edge_warnings": [], "score_breakdown": {"top": {"score": 0.61}},
            },
            "sources": [], "quota": {}, "similar_signal": None,
        },
        "quota": {"odds_api_remaining": 500},
    }


def test_best_emit_carries_render_full(monkeypatch):
    import os

    emitted: dict = {}

    async def fake_find_best_matches(**kwargs):
        return _fake_best_payload()

    def fake_emit(obj):
        emitted.update(obj)

    monkeypatch.setattr("agents.football.best_match.find_best_matches", fake_find_best_matches)
    monkeypatch.setattr(runner, "_emit", fake_emit)
    monkeypatch.setattr(runner, "_arm_deadline", lambda seconds: None)
    monkeypatch.setattr(os, "_exit", lambda code: None)

    rc = runner.main(["best", "--league", "EPL"])
    assert rc is None
    assert "render" in emitted and "render_full" in emitted
    # main reply: ranked shortlist + COMPACT winner card (single best pick)
    assert "PILIHAN TERBAIK" in emitted["render"]["body"]
    assert "📊 Arsenal vs Chelsea — EPL" in emitted["render"]["body"]
    assert "Tidak ada market dengan data cukup" in emitted["render"]["body"]
    assert "FINAL DECISION" not in emitted["render"]["body"]
    # Copy button serves the full best report (full winner analysis)
    assert "FINAL DECISION" in emitted["render_full"]["body"]


def test_analyse_error_emit_render_matches_full(monkeypatch):
    import asyncio
    import os

    emitted: dict = {}

    async def fake_find_specific_match(**kwargs):
        return {"error": "tim tidak ditemukan"}

    def fake_emit(obj):
        emitted.update(obj)

    monkeypatch.setattr(runner, "find_specific_match", fake_find_specific_match)
    monkeypatch.setattr(runner, "_emit", fake_emit)
    monkeypatch.setattr(runner, "_arm_deadline", lambda seconds: None)
    monkeypatch.setattr(os, "_exit", lambda code: None)

    rc = runner.main(["analyse", "--league", "EPL", "--home", "X", "--away", "Y"])
    assert rc is None
    assert "tim tidak ditemukan" in emitted["render"]["body"]
    assert emitted["render_full"]["body"] == emitted["render"]["body"]


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(__import__("pytest").MonkeyPatch())
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
