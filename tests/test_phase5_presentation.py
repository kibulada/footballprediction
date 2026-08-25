"""Phase 5 tests: honest presentation.

5.1 BEST PICK -> TOP SIGNAL when uncalibrated; edge: n/a on stale benchmark.
5.2 card metadata (benchmark age, bookmakers, movement snapshots, direction).
5.3 real "Why" components (model prob vs implied, line, lineup status).
5.4 edge-bucket-vs-ROI audit against CLOSING prices as a HARD filter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.format import (
    _benchmark_meta,
    _best_pick_block,
    _edge_display,
    _pick_label,
    _signal_why,
)
from agents.football.prediction_log import (
    append_snapshot,
    edge_bucket_audit,
    edge_bucket_closing_stats,
    edge_bucket_gate,
    settle,
)

MID = "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z"


def _se(**kw) -> dict:
    base = {
        "decision": "BEST PICK",
        "display_label": "BEST PICK",
        "edge_invalid": False,
        "edge_benchmark": {"key": "soft_consensus", "ts": None, "age_hours": 1.0, "stale": False},
        "best_pick": {
            "market": "Total", "selection": "Over 2.5", "score": 0.8,
            "confidence": "HIGH", "model_prob": 0.62, "implied_prob": 0.52,
            "market_odds": 1.9, "edge_pp": 10.0, "line": 2.5, "side": None,
            "components": {"model": 0.8, "market": 0.7, "movement": 0.6},
            "confluence": 3, "movement": {"status": "UNAVAILABLE"},
            "evidence_notes": [],
        },
        "data_quality": {"bookmakers_count": 4, "ah_ou_snapshots": 5},
    }
    base.update(kw)
    return base


# ---- 5.1: label switching + edge: n/a -----------------------------------

def test_pick_label_defaults_to_best_pick():
    assert _pick_label(_se()) == "BEST PICK"
    # Phase 5.4: uncalibrated leagues show explicit warning
    assert _pick_label(_se(display_label="TOP SIGNAL")) == "TOP SIGNAL ⚠️ LIGA TIDAK TERKALIBRASI"
    assert _pick_label({}) == "BEST PICK"


def test_best_pick_block_uses_top_signal_label():
    lines = _best_pick_block(_se(display_label="TOP SIGNAL"))
    # Phase 5.4: uncalibrated leagues show explicit warning in label
    assert any(l.startswith("🔥 TOP SIGNAL ⚠️ LIGA TIDAK TERKALIBRASI: OVER 2.5") for l in lines)


def test_edge_display_n_a_when_invalid():
    se = _se(edge_invalid=True)
    assert _edge_display(se, se["best_pick"]) == "edge: n/a"
    assert _edge_display(_se(), _se()["best_pick"]) == "edge +10.0pp"
    assert _edge_display(_se(), {"edge_pp": None}) == "edge: n/a"


def test_best_pick_block_shows_edge_n_a_on_stale_benchmark():
    se = _se(edge_invalid=True)
    lines = _best_pick_block(se)
    assert any("edge: n/a" in l for l in lines)
    assert any("stale" in l.lower() or "n/a" in l for l in lines)


def test_nobet_block_uses_label():
    lines = _best_pick_block(_se(display_label="TOP SIGNAL", decision="NO BET"))
    # Phase 5.4: uncalibrated league label includes warning
    assert any("NO BET (TOP SIGNAL ⚠️" in l for l in lines)


# ---- 5.2: card metadata -------------------------------------------------

def test_benchmark_meta_lines():
    se = _se()
    meta = _benchmark_meta(se)
    joined = "\n".join(meta)
    assert "Benchmark age: 1.0h" in joined
    assert "Bookmakers: 4" in joined
    assert "Movement snapshots: 5" in joined


def test_benchmark_meta_movement_direction_and_magnitude():
    bp = {
        "movement": {"status": "available", "direction": "toward", "magnitude_pct": 3.5},
    }
    se = _se(best_pick={**_se()["best_pick"], **bp})
    meta = _benchmark_meta(se)
    assert any("→ 3.5%" in m and "toward" in m for m in meta)


def test_best_pick_block_includes_metadata():
    lines = _best_pick_block(_se())
    joined = "\n".join(lines)
    assert "Benchmark age" in joined and "Bookmakers" in joined


# ---- 5.3: real Why components -------------------------------------------

def test_signal_why_uses_model_vs_implied_and_line():
    bp = _se()["best_pick"]
    why = _signal_why(bp, [])
    joined = " ".join(why)
    assert "62% vs implied 52%" in joined
    assert "Line: 2.5 goals" in joined


def test_signal_why_lineup_status():
    bp = dict(_se()["best_pick"])
    why = _signal_why(dict(bp, lineup_status="confirmed"), [])
    assert any("Lineup: confirmed" in w for w in why)
    why2 = _signal_why(dict(bp, lineup_status="predicted"), [])
    assert any("half weight" in w for w in why2)


def test_signal_why_ah_line():
    bp = dict(_se()["best_pick"])
    bp.update({"market": "Asian Handicap", "selection": "Home -0.5", "line": -0.5, "side": "home"})
    why = _signal_why(bp, [])
    assert any("Line: Home -0.50" in w for w in why)


# ---- 5.4: edge-bucket vs CLOSING hard filter -----------------------------

def _snap_close(path, *, prob, odds, closing, edge_bucket_proxy=None):
    append_snapshot(
        path,
        match_id=MID,
        league="EPL", home="Arsenal", away="Chelsea",
        kickoff="2026-08-15T14:00:00Z",
        prob=prob,
        odds=odds,
        edge={"home": 7.0},  # edge bucket 5-10%
        confidence=0.8, signal=70, calibration=None,
        model_version=None, input_hash=None,
        best_pick={"selection": "Home Win", "market": "1X2"},
        sources=[], decision_type="GOOD",
    )
    settle(path, match_id=MID, home_goals=0, away_goals=2, closing_odds=closing)


def test_edge_bucket_stats_uses_closing_prices(tmp_path):
    path = tmp_path / "p.jsonl"
    # pick home @ 2.2 (edge ~6pp -> bucket 5-10%); close home @ 2.0;
    # home loses -> roi_vs_closing = -1.0 (recomputed with the CLOSING price)
    _snap_close(path, prob={"home": 0.50, "draw": 0.28, "away": 0.22},
                odds={"home": 2.2, "draw": 3.5, "away": 3.4},
                closing={"home": 2.0, "draw": 3.5, "away": 3.4})
    stats = edge_bucket_closing_stats(path)
    bucket = stats["5-10%"]
    assert bucket["n"] == 1
    assert bucket["n_with_closing"] == 1
    assert bucket["roi_vs_closing"] == -1.0
    assert bucket["net_negative"] is True


def test_edge_bucket_gate_blocks_net_negative_bucket():
    stats = {"5-10%": {"n": 20, "n_with_closing": 20, "roi_vs_closing": -0.05, "net_negative": True}}
    g = edge_bucket_gate(stats, edge_pp=7.0, min_n=10)
    assert g["allowed"] is False
    assert "net-negative" in g["reason"]
    # positive bucket passes
    stats2 = {"5-10%": {"n": 20, "n_with_closing": 20, "roi_vs_closing": 0.05, "net_negative": False}}
    assert edge_bucket_gate(stats2, edge_pp=7.0, min_n=10)["allowed"] is True
    # thin sample -> not evidence -> passes but flagged
    stats3 = {"5-10%": {"n": 5, "n_with_closing": 5, "roi_vs_closing": -0.3, "net_negative": True}}
    g3 = edge_bucket_gate(stats3, edge_pp=7.0, min_n=10)
    assert g3["allowed"] is True
    assert "belum evidence" in g3["reason"]
    # no edge -> no bucket -> passes
    assert edge_bucket_gate(stats, edge_pp=None)["allowed"] is True


def test_edge_bucket_audit_writes_report(tmp_path):
    path = tmp_path / "p.jsonl"
    _snap_close(path, prob={"home": 0.50, "draw": 0.28, "away": 0.22},
                odds={"home": 2.2, "draw": 3.5, "away": 3.4},
                closing={"home": 2.0, "draw": 3.5, "away": 3.4})
    out = tmp_path / "out"
    rep = edge_bucket_audit(path, out_dir=out, date="2026-08-15")
    assert rep["benchmark"] == "closing"
    assert "5-10%" in rep["buckets"]
    assert rep["net_negative_buckets"] == ["5-10%"]
    fpath = Path(rep["file"])
    assert fpath.exists()
    payload = json.loads(fpath.read_text(encoding="utf-8"))
    assert payload["net_negative_buckets"] == ["5-10%"]


# ---- Keputusan 2026-08-23: BEST PICK renderer non-vetoed / HIGH RISK -----

def _all_vetoed_se() -> dict:
    """Semua kandidat diveto pick_gates: best_pick=None, ranking utuh."""
    return {
        "decision": "NO BET",
        "display_label": "BEST PICK",
        "edge_invalid": False,
        "best_pick": None,
        "reasons": ["Over 2.5: broken lambda — veto"],
        "ranking": [
            {
                "market": "Total", "selection": "Over 2.5", "score": 0.71,
                "confidence": "NO SIGNAL", "model_prob": 0.6,
                "market_odds": 1.9, "implied_prob": 0.52, "edge_pp": 8.0,
                "movement": {}, "components": {}, "line": 2.5, "side": None,
                "line_key": None, "internal_notes": [],
                "vetoed": True, "veto_reasons": ["broken lambda"],
            },
            {
                "market": "BTTS", "selection": "BTTS Yes", "score": 0.55,
                "confidence": "NO SIGNAL", "model_prob": 0.58,
                "market_odds": 1.8, "implied_prob": 0.55, "edge_pp": 3.0,
                "movement": {}, "components": {}, "line": None, "side": None,
                "line_key": None, "internal_notes": [],
                "vetoed": True, "veto_reasons": ["n_bucket < minimum"],
            },
        ],
        "data_quality": {"bookmakers_count": 4},
    }


def test_display_best_pick_normal_best_pick_has_no_risk():
    from agents.football.format import _display_best_pick

    se = _se()
    pick, risk = _display_best_pick(se)
    assert pick is se["best_pick"]
    assert risk is None


def test_display_best_pick_all_vetoed_returns_nothing_f11():
    """F11 (plan v3 2026-08-24): SEMUA kandidat diveto -> (None, None).
    Kebijakan lama menampilkan rank #1 berlabel HIGH RISK; di lapangan itu
    terbaca sebagai BEST PICK biasa (kasus Goztepe v Genclerbirligi)."""
    from agents.football.format import _display_best_pick

    se = _all_vetoed_se()
    assert _display_best_pick(se) == (None, None)


def test_display_best_pick_nobet_with_eligible_candidates_stays_hidden():
    """NO BET karena gagal floor (kandidat tidak diveto) TETAP tidak menampilkan
    pick -- disiplin post-mortem 2026-08-22."""
    from agents.football.format import _display_best_pick

    se = {
        "decision": "NO BEST PICK", "best_pick": None,
        "ranking": [
            {"selection": "Over 2.5", "score": 0.30, "vetoed": False,
             "veto_reasons": []},
        ],
    }
    assert _display_best_pick(se) == (None, None)


def test_best_pick_block_all_vetoed_renders_plain_no_bet():
    lines = _best_pick_block(_all_vetoed_se())
    joined = "\n".join(lines)
    assert any(l.startswith("⚪ NO BET") for l in lines)
    # blok Reason tetap tampil (isi memakai headline gate/generik)
    assert any(l.strip() == "Reason:" for l in lines)
    assert not any(l.startswith("🔥") for l in lines)


def test_summary_best_pick_value_all_vetoed_is_no_bet():
    from agents.football.discord_signal_card_accordion import (
        _summary_best_pick_value,
    )

    out = _summary_best_pick_value(_all_vetoed_se())
    assert out == "⚪ NO BET"


def test_summary_best_pick_value_normal_path_unchanged():
    from agents.football.discord_signal_card_accordion import (
        _summary_best_pick_value,
    )

    out = _summary_best_pick_value(_se())
    assert not out.startswith("⚠️")
    assert "OVER 2.5 @ 1.90" in out
    assert "Confidence: HIGH" in out
