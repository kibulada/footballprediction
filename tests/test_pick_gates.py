"""Tests for agents/football/pick_gates.py — BEST PICK hard gates.

Every case is anchored to a REAL logged incident from
reports/bestpick_postmortem_2026-08-22.md so the gates stay tied to evidence
rather than to taste.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.football.pick_gates import (  # noqa: E402
    agreement_gate,
    elo_integrity_gate,
    entity_integrity_gate,
    lambda_1x2_gate,
    lambda_total_gate,
    price_gate,
    resolve_elo_band,
)


# --------------------------------------------------------------------------
# G2 — agreement
# --------------------------------------------------------------------------

def test_agreement_passes_for_marseille_and_arsenal_winners():
    """Both 2026-08-21 Over 2.5 WINNERS sat within 3pp of the market."""
    ok, rs = agreement_gate(0.603, 0.608)   # Marseille, dev -0.5pp -> WON
    assert ok and rs == []
    ok, rs = agreement_gate(0.569, 0.594)   # Arsenal, dev -2.5pp -> WON
    assert ok and rs == []


def test_agreement_vetoes_ried_and_betis_losers():
    """SV Ried (+28.5pp) and Real Betis (+10.7pp) both LOST."""
    ok, rs = agreement_gate(0.775, 0.490)
    assert not ok and "28.5pp" in rs[0]
    ok, rs = agreement_gate(0.622, 0.515)
    assert not ok


def test_agreement_boundary_is_inclusive():
    ok, _ = agreement_gate(0.58, 0.50, max_dev_pp=8.0)   # exactly 8.0pp
    assert ok
    ok, _ = agreement_gate(0.581, 0.50, max_dev_pp=8.0)  # 8.1pp
    assert not ok


def test_agreement_missing_input_never_invents_veto():
    assert agreement_gate(None, 0.5)[0] is True
    assert agreement_gate(0.5, None)[0] is True


# --------------------------------------------------------------------------
# G4 — lambda_total sanity
# --------------------------------------------------------------------------

def test_lambda_total_vetoes_sv_ried():
    """lam_h 2.56 + lam_a 1.528 = 4.09 on L-L-D-W-L form; FT was 1-0."""
    ok, rs = lambda_total_gate(2.56, 1.528)
    assert not ok and "4.09" in rs[0]


def test_lambda_total_allows_real_cards():
    assert lambda_total_gate(1.846, 1.273)[0] is True   # Marseille  3.12 -> WON
    assert lambda_total_gate(1.441, 1.524)[0] is True   # Arsenal    2.97 -> WON
    assert lambda_total_gate(1.017, 0.680)[0] is True   # Gent       1.70 -> WON


def test_lambda_total_vetoes_absurdly_low():
    ok, _ = lambda_total_gate(0.4, 0.5)
    assert not ok


def test_lambda_total_missing_input_passes():
    assert lambda_total_gate(None, 1.0)[0] is True


# --------------------------------------------------------------------------
# G3 — lambda vs 1X2 consistency
# --------------------------------------------------------------------------

def test_lambda_1x2_vetoes_erzurumspor_contradiction():
    """1X2 away 60.9% but lam_h 1.372 > lam_a 1.208; AH Home +1 lost 0-4."""
    ok, rs = lambda_1x2_gate({
        "1x2": {"home": 0.202, "draw": 0.189, "away": 0.609},
        "lambda_home": 1.372, "lambda_away": 1.208,
    })
    assert not ok and "kontradiksi" in rs[0]


def test_lambda_1x2_does_not_fire_for_al_riyadh_real_values():
    """Al Riyadh v Al Nassr with its REAL logged lambdas (1.358 / 1.638) is NOT
    a lambda-direction contradiction: 1X2 favours away 81.7% and lambda also
    favours away. G3 correctly stays silent.

    That card is covered by two OTHER gates instead, and this test exists to
    stop anyone 'fixing' G3 to cover it with fabricated numbers:
      - the published Total Under 3.5 dies on G2 (model P(u3.5) ~0.648 from
        lambda_total 2.996 vs a margin-free market ~0.52 -> deviation >8pp)
      - the ranking[0] AH Home +1.75 dies on the pre-existing
        ``ah_side_contradiction_prob`` gate (favourite away 81.7%, side home)
    """
    ok, rs = lambda_1x2_gate({
        "1x2": {"home": 0.0523, "draw": 0.1303, "away": 0.8174},
        "lambda_home": 1.358, "lambda_away": 1.638,
    })
    assert ok and rs == []


def test_lambda_1x2_detects_arsenal_contradiction():
    """Arsenal v Coventry genuinely contradicted: 1X2 home 66.0% but
    lam_h 1.441 < lam_a 1.524. The DETECTOR must fire..."""
    ok, rs = lambda_1x2_gate({
        "1x2": {"home": 0.660, "draw": 0.214, "away": 0.126},
        "lambda_home": 1.441, "lambda_away": 1.524,
    })
    assert not ok and "kontradiksi" in rs[0]


def test_lambda_1x2_passes_when_directions_agree():
    ok, rs = lambda_1x2_gate({
        "1x2": {"home": 0.660, "draw": 0.214, "away": 0.126},
        "lambda_home": 1.900, "lambda_away": 1.100,
    })
    assert ok and rs == []


def test_lambda_1x2_inactive_without_a_clear_favourite():
    """No side above the threshold -> the gate must not fire."""
    ok, _ = lambda_1x2_gate({
        "1x2": {"home": 0.44, "draw": 0.26, "away": 0.30},
        "lambda_home": 1.10, "lambda_away": 1.90,
    })
    assert ok


def test_lambda_1x2_missing_lambdas_passes():
    assert lambda_1x2_gate({"1x2": {"home": 0.8, "away": 0.1}})[0] is True


# --------------------------------------------------------------------------
# G5 — Elo integrity
# --------------------------------------------------------------------------

def test_elo_vetoes_out_of_range_sociedad():
    """Sociedad 2361 was a lookup COLLISION, not a legit rating. 2026-08-28:
    the ceiling moved to 2450 because Barcelona 2298 / Real Madrid 2243 /
    Arsenal 2361 are real ratings in the live store (all BEST PICK winners);
    the incident band is kept reproducible via an explicit ``hi``."""
    ok, rs = elo_integrity_gate(
        {"elo_seeded": True, "elo_home": 2036.0, "elo_away": 2361.0}, hi=2100.0,
    )
    assert not ok and any("2361" in r for r in rs)
    # Default band now accepts the verified real ratings...
    assert elo_integrity_gate({"elo_seeded": True, "elo_home": 2298.0, "elo_away": 1958.0})[0]
    # ...but still rejects the impossible (Kelty Hearts 1031 on a UECL card).
    ok, rs = elo_integrity_gate({"elo_seeded": True, "elo_home": 1291.0, "elo_away": 1031.0})
    assert not ok and any("1031" in r for r in rs)


def test_elo_vetoes_identical_rating_collision():
    ok, rs = elo_integrity_gate({"elo_seeded": True, "elo_home": 1500.0, "elo_away": 1500.0})
    assert not ok and any("identik" in r for r in rs)


def test_elo_vetoes_unseeded():
    """Al-Faisaly v Neom: both ratings default -> lambda is a form streak."""
    ok, rs = elo_integrity_gate({"elo_seeded": False})
    assert not ok and any("seed" in r for r in rs)


def test_elo_passes_credible_pair():
    ok, rs = elo_integrity_gate({"elo_seeded": True, "elo_home": 1607.0, "elo_away": 1601.0})
    assert ok and rs == []


def test_elo_without_ratings_only_checks_seeded_flag():
    assert elo_integrity_gate({"elo_seeded": True})[0] is True


def test_elo_band_resolver_defaults_and_overrides():
    """No cfg / unknown league -> the global senior band, byte-identical
    behaviour for every league except the configured override."""
    assert resolve_elo_band(None) == (1300.0, 2450.0)
    assert resolve_elo_band({"elo_min": 1300.0, "elo_max": 2450.0}, "EPL") == (1300.0, 2450.0)
    cfg = {"elo_min": 1300.0, "elo_max": 2450.0,
           "elo_band_by_league": {"eerste divisie": {"min": 1150.0}}}
    assert resolve_elo_band(cfg, "Eerste Divisie") == (1150.0, 2450.0)
    assert resolve_elo_band(cfg, "eerste divisie") == (1150.0, 2450.0)
    assert resolve_elo_band(cfg, "Eredivisie") == (1300.0, 2450.0)
    # bare number means "max" (same convention as lambda_total_band_by_league)
    cfg2 = {"elo_band_by_league": {"eerste divisie": 2600}}
    assert resolve_elo_band(cfg2, "Eerste Divisie") == (1300.0, 2600.0)


def test_elo_gate_passes_jong_pair_under_league_band():
    """Jong PSV 1353 v Jong Ajax 1238 (2026-09-01): real reserve-XI ratings
    from the live elofootball.com store, card-wide vetoed by the senior
    floor. Under the per-league band they pass -- and the collision check
    stays alive (it does not depend on the band)."""
    mp = {"elo_seeded": True, "elo_home": 1353.0, "elo_away": 1238.0}
    ok, rs = elo_integrity_gate(mp, lo=1300.0, hi=2450.0)
    assert not ok and any("1238" in r for r in rs)
    ok, rs = elo_integrity_gate(mp, lo=1150.0, hi=2450.0)
    assert ok and rs == []
    ok, rs = elo_integrity_gate(
        {"elo_seeded": True, "elo_home": 1238.0, "elo_away": 1238.0}, lo=1150.0, hi=2450.0,
    )
    assert not ok and any("identik" in r for r in rs)


# --------------------------------------------------------------------------
# G6 — entity integrity
# --------------------------------------------------------------------------

def test_entity_vetoes_women_vs_men():
    ok, rs = entity_integrity_gate("Braga", "Austria Wien Women")
    assert not ok and "fabrikasi" in rs[0]
    ok, _ = entity_integrity_gate("KÍ Women", "Lech Poznań")
    assert not ok


def test_entity_vetoes_reserve_vs_first_team():
    ok, _ = entity_integrity_gate("Hearts of Oak", "Rapid Wien II")
    assert not ok


def test_entity_allows_symmetric_womens_fixture_with_note():
    ok, rs = entity_integrity_gate("Barcelona Women", "Chelsea Women")
    assert ok
    assert any("kedua tim bertanda" in r for r in rs)


def test_entity_flags_basketball_suffix_without_vetoing():
    """Atalanta v 'Hapoel Tel Aviv BC' was a REAL UECL fixture that WON."""
    ok, rs = entity_integrity_gate("Atalanta", "Hapoel Tel Aviv BC")
    assert ok
    assert any("artefak penamaan" in r for r in rs)


def test_entity_passes_clean_names():
    ok, rs = entity_integrity_gate("Olympique de Marseille", "RC Strasbourg Alsace")
    assert ok and rs == []
    ok, rs = entity_integrity_gate("Arsenal", "Coventry")
    assert ok and rs == []


def test_entity_does_not_flag_nordic_bk_suffix():
    """'BK' is a legitimate football suffix (Boldklub) -- must not trip 'BC'."""
    ok, rs = entity_integrity_gate("IFK Göteborg", "Brommapojkarna BK")
    assert ok and rs == []


def test_entity_roman_numeral_is_case_sensitive():
    """'II' marks a reserve XI; the Finnish first-team club 'Ii' must not.

    Guards the case-sensitivity split in _MARKER_PATTERNS -- a case-insensitive
    numeral pattern would veto a legitimate fixture.
    """
    ok, _ = entity_integrity_gate("Rapid Wien II", "Sturm Graz")
    assert not ok, "uppercase II is a reserve marker"
    ok, rs = entity_integrity_gate("Ii", "KuPS")
    assert ok and rs == [], "the Finnish club 'Ii' is a first team"


# --------------------------------------------------------------------------
# G7 — price
# --------------------------------------------------------------------------

def test_price_vetoes_null_odds_braga():
    ok, rs = price_gate(None)
    assert not ok and "tidak ada harga" in rs[0]


def test_price_vetoes_thin_bookmaker_count():
    ok, rs = price_gate(1.90, bookmakers_count=2, min_bookmakers=3)
    assert not ok and "bookmaker" in rs[0]


def test_price_passes_real_quote():
    ok, rs = price_gate(1.53, bookmakers_count=6)
    assert ok and rs == []


def test_price_vetoes_impossible_odds():
    assert price_gate(1.0)[0] is False
    assert price_gate(0.5)[0] is False
