"""G12: zona maut Elo gap (pick_gates.elo_gap_dead_zone), 2026-09-04.

Selisih Elo 100-200 = "favorit tanggung": harga sudah dipatok seperti
favorit tapi realisasi hit 33% (4W/8L) vs 91.7% di luar zona. Bukti penuh
di ``scripts/eval_deadzone.py`` dan docstring gate di signal_engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.signal_engine import run_signal_engine  # noqa: E402


def _base_kwargs(**over):
    kw = dict(
        model_probs={
            "1x2": {"home": 0.55, "draw": 0.25, "away": 0.20},
            "over_2.5": 0.62,
            "btts_yes": 0.58,
            "lambda_home": 1.6,
            "lambda_away": 1.1,
            "elo_seeded": True,
        },
        stats={},
        market_totals={"2.5": {"over": 1.85, "under": 1.95}},
        ah_rows=[],
        odds_1x2={"home": 1.90, "draw": 3.40, "away": 4.20},
        completeness=0.8,
    )
    kw.update(over)
    return kw


def _vetoed_reasons(res):
    out = []
    for e in res.get("ranking") or []:
        out.extend(e.get("veto_reasons") or [])
    return " | ".join(out)


def test_dead_zone_vetoes_all_candidates():
    """Gap 150 (di dalam 100-200) -> semua kandidat kena veto."""
    res = run_signal_engine(**_base_kwargs(elo_home=1600.0, elo_away=1450.0))
    assert res.get("pick") is None, "gap 150 harus menghasilkan NO BET"
    assert "zona maut" in _vetoed_reasons(res).lower()


def test_below_dead_zone_not_vetoed_by_g12():
    """Gap 40 (di bawah zona) -> G12 tidak ikut campur."""
    res = run_signal_engine(**_base_kwargs(elo_home=1540.0, elo_away=1500.0))
    assert "zona maut" not in _vetoed_reasons(res).lower()


def test_above_dead_zone_not_vetoed_by_g12():
    """Gap 350 (mismatch besar, hit historis tertinggi) -> lolos G12."""
    res = run_signal_engine(**_base_kwargs(elo_home=1850.0, elo_away=1500.0))
    assert "zona maut" not in _vetoed_reasons(res).lower()


def test_boundaries_are_half_open():
    """Batas bawah inklusif, batas atas eksklusif: [100, 200)."""
    at_lo = run_signal_engine(**_base_kwargs(elo_home=1600.0, elo_away=1500.0))
    assert "zona maut" in _vetoed_reasons(at_lo).lower(), "gap 100 harus kena"
    at_hi = run_signal_engine(**_base_kwargs(elo_home=1700.0, elo_away=1500.0))
    assert "zona maut" not in _vetoed_reasons(at_hi).lower(), "gap 200 harus lolos"


def test_missing_elo_does_not_veto():
    """Elo tidak tersedia -> gate fail-open, jangan blokir apa pun."""
    res = run_signal_engine(**_base_kwargs(elo_home=None, elo_away=None))
    assert "zona maut" not in _vetoed_reasons(res).lower()


def test_gate_can_be_disabled_via_config():
    res = run_signal_engine(**_base_kwargs(
        elo_home=1600.0, elo_away=1450.0,
        cfg={"pick_gates": {"elo_gap_dead_zone": False}},
    ))
    assert "zona maut" not in _vetoed_reasons(res).lower()


def test_custom_bounds_from_config():
    """Ambang bisa digeser lewat config (untuk re-kalibrasi nanti)."""
    res = run_signal_engine(**_base_kwargs(
        elo_home=1900.0, elo_away=1500.0,   # gap 400
        cfg={"pick_gates": {
            "elo_gap_dead_zone_min": 300.0,
            "elo_gap_dead_zone_max": 500.0,
        }},
    ))
    assert "zona maut" in _vetoed_reasons(res).lower()
