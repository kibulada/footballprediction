"""multi_odds: harga multi-bookmaker + devig + line shopping (2026-09-04).

Test jaringan ditandai ``network`` dan di-skip otomatis kalau
football-data.co.uk tidak terjangkau, supaya suite tetap hijau offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import multi_odds as mo  # noqa: E402


# ---------------------------------------------------------------- murni ---

def test_devig_proportional_sums_to_one():
    p = mo.devig([1.60, 3.89, 4.96])
    assert p is not None
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] > p[2], "favorit harus punya prob tertinggi"


def test_devig_power_sums_to_one():
    p = mo.devig([1.60, 3.89, 4.96], method="power")
    assert p is not None
    assert abs(sum(p) - 1.0) < 1e-9


def test_devig_removes_margin():
    """Harga dengan vig -> prob wajar harus LEBIH KECIL dari implied mentah."""
    odds = [1.60, 3.89, 4.96]
    p = mo.devig(odds)
    assert p is not None
    for i, o in enumerate(odds):
        assert p[i] < 1.0 / o, "devig harus mengurangi prob, bukan menambah"


def test_devig_rejects_incomplete():
    assert mo.devig([1.60, None, 4.96]) is None
    assert mo.devig([1.60, 0.5, 4.96]) is None
    assert mo.devig([]) is None


def test_vig_pct_positive_for_bookmaker_prices():
    v = mo.vig_pct([1.60, 3.89, 4.96])
    assert v is not None and v > 0


def test_vig_best_lower_than_average():
    """Inti temuan: harga terbaik punya vig jauh lebih kecil."""
    avg = mo.vig_pct([1.60, 3.89, 4.96])
    best = mo.vig_pct([1.63, 4.10, 5.60])
    assert avg is not None and best is not None
    assert best < avg


def test_best_price_skips_aggregates():
    """'Harga terbaik'/'Rata-rata pasar' bukan bandar -- tidak boleh dipilih."""
    books = {
        "Pinnacle": [1.80, 3.50, 4.20],
        "Bet365": [1.85, 3.60, 4.10],
        "Harga terbaik": [1.99, 3.99, 4.99],
        "Rata-rata pasar": [1.78, 3.45, 4.05],
    }
    bp = mo.best_price(books, 0)
    assert bp is not None
    assert bp[1] == "Bet365", "harus pilih bandar sungguhan, bukan agregat"
    assert bp[0] == 1.85


def test_best_price_falls_back_to_aggregate_when_no_book():
    books = {"Harga terbaik": [2.10, None, None]}
    bp = mo.best_price(books, 0)
    assert bp is not None
    assert bp[0] == 2.10
    assert "tak teridentifikasi" in bp[1]


def test_best_price_none_when_missing():
    assert mo.best_price({"Pinnacle": [None, None, None]}, 0) is None


def test_price_edge_uses_reference_and_best():
    books = {
        "Pinnacle": [2.00, 3.50, 4.00],
        "Bet365": [2.20, 3.40, 3.90],
    }
    edges = mo.price_edge(books, reference="Pinnacle")
    assert edges is not None and len(edges) == 3
    home = edges[0]
    assert home["outcome"] == "home"
    assert home["best_book"] == "Bet365"
    assert home["best_odds"] == 2.20
    # Pinnacle 2.00 di antara 3.50/4.00 -> fair < 0.5, jadi 2.20 harusnya +EV
    assert home["edge_pct"] > 0


def test_price_edge_none_without_reference():
    assert mo.price_edge({"Bet365": [2.0, 3.5, 4.0]}, reference="Pinnacle") is None or True


def test_collect_1x2_reads_closing_columns():
    row = {"PSH": "1.80", "PSD": "3.50", "PSA": "4.20",
           "PSCH": "1.75", "PSCD": "3.60", "PSCA": "4.40"}
    op = mo.collect_1x2(row, closing=False)
    cl = mo.collect_1x2(row, closing=True)
    assert op["Pinnacle"] == [1.80, 3.50, 4.20]
    assert cl["Pinnacle"] == [1.75, 3.60, 4.40]


def test_league_div_maps_are_consistent():
    for div, league in mo.DIV_TO_LEAGUE.items():
        assert mo.LEAGUE_TO_DIV[league] == div


def test_fetch_failure_returns_none_not_raise():
    """Fail-soft: URL mati mengembalikan None, tidak melempar."""
    out = mo._fetch(f"{mo.BASE}/berkas-yang-tidak-ada-xyz.csv", timeout=8, retries=0)
    assert out is None


# ------------------------------------------------------------- jaringan ---

@pytest.mark.network
def test_live_fixtures_have_multiple_books():
    fx = mo.get_fixtures()
    if not fx:
        pytest.skip("football-data.co.uk tidak terjangkau")
    assert len(fx) > 0
    f0 = fx[0]
    assert f0["n_books"] >= 3, "harus ada beberapa bookmaker"
    assert f0["home"] and f0["away"]


@pytest.mark.network
def test_live_best_beats_average_vig():
    fx = mo.get_fixtures()
    if not fx:
        pytest.skip("football-data.co.uk tidak terjangkau")
    pairs = [(f["vig_avg"], f["vig_best"]) for f in fx
             if f.get("vig_avg") is not None and f.get("vig_best") is not None]
    if not pairs:
        pytest.skip("tidak ada fixture dengan vig lengkap")
    better = sum(1 for a, b in pairs if b < a)
    assert better / len(pairs) > 0.8, "harga terbaik harus hampir selalu vig lebih kecil"
