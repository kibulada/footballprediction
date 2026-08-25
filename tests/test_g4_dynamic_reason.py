"""Tests for the dynamic G4 rejection reason (P4, 2026-08-23).

The reason must carry its own evidence -- band_source / ceiling /
model_total_lambda / market_implied_total / model_market_gap -- and must NOT
reference a hardcoded incident ("pola SV Ried") on every card.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.football.pick_gates import (  # noqa: E402
    band_source,
    lambda_total_gate,
    market_implied_total,
)
from agents.football.signal_engine import run_signal_engine  # noqa: E402


# ---- band_source -----------------------------------------------------------

def test_band_source_global_without_override():
    cfg = {"lambda_total_band_by_league": {"eredivisie": {"max": 4.0}}}
    assert band_source(cfg, "Belgian Pro League") == "global"
    assert band_source(None, "EPL") == "global"
    assert band_source({}) == "global"


def test_band_source_league_override():
    cfg = {"lambda_total_band_by_league": {"belgian pro league": {"max": 4.0}}}
    assert band_source(cfg, "Belgian Pro League") == "league_override"
    assert band_source(cfg, "  belgian pro league  ") == "league_override"


# ---- market_implied_total --------------------------------------------------

def test_market_implied_total_from_full_ladder():
    ladder = {
        "Over 2.5": {"odds": 1.369}, "Under 2.5": {"odds": 3.17},
        "Over 3.5": {"odds": 1.99}, "Under 3.5": {"odds": 1.877},
        "Over 4.5": {"odds": 3.15}, "Under 4.5": {"odds": 1.373},
    }
    mit = market_implied_total(ladder)
    assert mit is not None and 3.5 < mit < 3.75


def test_market_implied_total_insufficient_data_returns_none():
    assert market_implied_total(None) is None
    assert market_implied_total({}) is None
    # whole-number lines only -> no push-free pair
    ladder = {"Over 3.0": {"odds": 1.55}, "Under 3.0": {"odds": 2.52}}
    assert market_implied_total(ladder) is None
    # garbage odds never crash, just degrade to None
    ladder = {"Over 2.5": {"odds": 0}, "Under 2.5": {"odds": 3.0}}
    assert market_implied_total(ladder) is None


# ---- gate text -------------------------------------------------------------

def test_lambda_gate_reason_has_no_hardcoded_incident():
    ok, rs = lambda_total_gate(2.56, 1.528)  # the SV Ried shape itself
    assert not ok
    assert "SV Ried" not in rs[0]
    assert "di luar band" in rs[0]


# ---- end-to-end: veto reason carries the dynamic context -------------------

def _run(ladder: dict | None):
    model_probs = {
        "1x2": {"home": 0.75, "draw": 0.13, "away": 0.12},
        "over_1.5": 0.92, "over_2.5": 0.78, "over_3.5": 0.59,
        "btts_yes": 0.74,
        "lambda_home": 2.441, "lambda_away": 1.663,
        "elo_seeded": True,
    }
    market_totals = ladder or {}
    return run_signal_engine(
        model_probs=model_probs,
        stats={},
        market_totals=market_totals,
        ah_rows=[],
        completeness=1.0,
        cfg={"pick_gates": {"lambda_total_sanity": True}},
    )


def test_veto_reason_includes_dynamic_context():
    ladder = {
        "Over 2.5": {"odds": 1.369}, "Under 2.5": {"odds": 3.17},
        "Over 3.5": {"odds": 1.99}, "Under 3.5": {"odds": 1.877},
    }
    res = _run(ladder)
    top = res["ranking"][0]
    assert top["vetoed"]
    reason = top["veto_reasons"][0]
    assert "band_source=global" in reason
    assert "ceiling=3.6" in reason
    assert "model_total_lambda=4.10" in reason
    assert "market_implied_total=" in reason
    assert "n/a" not in reason
    assert "model_market_gap=+" in reason


def test_veto_reason_degrades_honestly_without_market():
    res = _run(None)
    reason = res["ranking"][0]["veto_reasons"][0]
    assert "band_source=global" in reason
    assert "ceiling=3.6" in reason
    assert "model_total_lambda=4.10" in reason
    assert "market_implied_total=n/a" in reason
    assert "model_market_gap" not in reason
