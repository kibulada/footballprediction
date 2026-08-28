"""Post-mortem 2026-08-28 (25-27 Aug live, BEST PICK 20-6 / SUGGESTION 40-14).

Each test encodes ONE failure class as a general rule, not a per-match
threshold:

  K1  no evidence   -- both Elo on the 1500 prior -> directional veto
  K2  wrong entity  -- elo.resolve strictness, source-consistency gate,
                       Elo range/collision wired
  K3  tie context   -- second leg detected from H2H, soft penalties
  K4  suggestion    -- may be "—"; dominance floor; model agreement
  K5  tiering       -- LEAN vs BEST PICK label + evaluation split
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.elo import EloModel  # noqa: E402
from agents.football.market_lean import (  # noqa: E402
    compute_suggestion,
    lean_candidates,
    select_suggestion,
    suggestion_for_settlement,
)
from agents.football.pick_gates import (  # noqa: E402
    elo_evidence_scope,
    is_directional_selection,
    is_low_scoring_selection,
    source_consistency_gate,
)
from agents.football.signal_engine import run_signal_engine, settle_signal  # noqa: E402
from agents.football.tie_state import tie_state_from_h2h, tie_state_note  # noqa: E402


# --------------------------------------------------------------------------
# K2 -- elo.resolve
# --------------------------------------------------------------------------

def _elo(tmp: Path) -> EloModel:
    elo = EloModel(path=tmp / "elo.json")
    elo.ratings = {
        "Kelty Hearts": 1031.0,
        "Heart of Midlothian": 1772.0,
        "SL Benfica": 2108.0,
        "SL Benfica B": 1558.0,
        "Olympique Lyonnais": 2027.0,
        "Celtic FC": 1900.0,
        "Union Berlin": 1485.0,
        "Union Saint-Gilloise": 1480.0,
        "Newcastle": 1560.0,
    }
    elo.games = {k: 30 for k in elo.ratings}
    elo._rebuild_indexes()
    return elo


def test_single_token_query_never_partial_matches_a_longer_key(tmp_path):
    """'Hearts' must not become Kelty Hearts (Rapid v Hearts 2026-08-26)."""
    elo = _elo(tmp_path)
    elo.ratings.pop("Heart of Midlothian")
    elo._rebuild_indexes()
    assert elo.resolve("Hearts") is None
    assert elo.rating("Hearts") == 1500.0


def test_alias_maps_short_live_name_to_seed_key(tmp_path):
    elo = _elo(tmp_path)
    assert elo.resolve("Hearts") == "Heart of Midlothian"
    assert elo.resolve("Lyon") == "Olympique Lyonnais"
    assert elo.resolve("Celtic") == "Celtic FC"


def test_reserve_key_never_wins_a_first_team_query(tmp_path):
    """'Benfica' ties SL Benfica / SL Benfica B on tokens -> first team."""
    elo = _elo(tmp_path)
    assert elo.resolve("Benfica") == "SL Benfica"
    assert elo.resolve("SL Benfica B") == "SL Benfica B"


def test_ambiguous_single_token_still_none(tmp_path):
    elo = _elo(tmp_path)
    assert elo.resolve("Union") is None
    assert elo.resolve("Newcastle United") == "Newcastle"


# --------------------------------------------------------------------------
# K1 -- evidence scope
# --------------------------------------------------------------------------

def test_elo_scope_both_prior_is_card_wide():
    scope, note = elo_evidence_scope({"elo_home_seeded": False, "elo_away_seeded": False})
    assert scope == "all" and "kedua tim" in note


def test_elo_scope_one_side_only_caps():
    scope, _ = elo_evidence_scope({"elo_home_seeded": True, "elo_away_seeded": False})
    assert scope == "one"
    assert elo_evidence_scope({"elo_home_seeded": True, "elo_away_seeded": True}) == (None, None)


def test_elo_scope_legacy_flag_is_conservative():
    assert elo_evidence_scope({"elo_seeded": False})[0] == "one"
    assert elo_evidence_scope({"elo_seeded": True}) == (None, None)


def test_directional_and_low_scoring_helpers():
    assert is_directional_selection("1X2", "Home Win")
    assert is_directional_selection("Asian Handicap", "Away +0.25")
    assert not is_directional_selection("1X2", "Draw")
    assert not is_directional_selection("Total", "Under 2.5")
    assert is_low_scoring_selection("Total", "Under 2.5")
    assert is_low_scoring_selection("BTTS", "BTTS No")
    assert not is_low_scoring_selection("BTTS", "BTTS Yes")


# --------------------------------------------------------------------------
# K2 -- source consistency
# --------------------------------------------------------------------------

def test_consistency_gate_fires_on_copenhagen_class():
    ok, rs, detail = source_consistency_gate(
        {"home": 1.255, "draw": 5.6, "away": 9.6},
        {"home": {"sequence": "D-L-L-L-L", "ga_avg": 5.2}, "away": {"sequence": "D-W-W-D-D", "ga_avg": 0.0}},
        {"lambda_home": 0.809, "lambda_away": 1.526},
    )
    assert not ok and detail["favourite"] == "home" and "Copenhagen" in rs[0]


def test_consistency_gate_needs_both_signals():
    # bottom-tier form but lambda agrees with the market -> no veto
    ok, _, _ = source_consistency_gate(
        {"home": 1.25, "draw": 5.6, "away": 9.6},
        {"home": {"sequence": "D-L-L-L-L", "ga_avg": 5.2}},
        {"lambda_home": 2.0, "lambda_away": 0.8},
    )
    assert ok
    # lambda underdog but form is fine -> no veto (a real upset spot)
    ok, _, _ = source_consistency_gate(
        {"home": 1.25, "draw": 5.6, "away": 9.6},
        {"home": {"sequence": "W-W-D-W-L", "ga_avg": 1.0}},
        {"lambda_home": 0.8, "lambda_away": 1.5},
    )
    assert ok


def test_consistency_gate_ignores_non_heavy_favourites_and_missing_input():
    ok, _, _ = source_consistency_gate(
        {"home": 1.9, "draw": 3.4, "away": 4.0},
        {"home": {"sequence": "L-L-L-L-L", "ga_avg": 5.0}},
        {"lambda_home": 0.5, "lambda_away": 1.5},
    )
    assert ok
    assert source_consistency_gate(None, None, None)[0]


# --------------------------------------------------------------------------
# K3 -- tie state
# --------------------------------------------------------------------------

def _h2h(home_prev: str, away_prev: str, hs: int, as_: int, when: str) -> dict:
    return {"meetings": [{
        "home": home_prev, "away": away_prev, "home_score": hs, "away_score": as_,
        "kickoff": when, "status": "finished",
    }]}


def test_tie_state_decided_from_reversed_first_leg():
    ts = tie_state_from_h2h(
        _h2h("Besiktas", "Kauno Zalgiris", 3, 0, "2026-08-20T17:00:00Z"),
        home="FK Kauno Zalgiris", away="Beşiktaş", kickoff="2026-08-27T17:00:00Z",
    )
    assert ts and ts["state"] == "decided" and ts["leader"] == "away"
    assert ts["agg_margin_home"] == -3 and ts["first_leg_home_goals"] == 0
    assert "rotasi" in tie_state_note(ts)


def test_tie_state_balanced_and_none_cases():
    ts = tie_state_from_h2h(
        _h2h("Rangers", "Jablonec", 1, 0, "2026-08-20T19:00:00Z"),
        home="Jablonec", away="Rangers FC", kickoff="2026-08-27T16:00:00Z",
    )
    assert ts and ts["state"] == "balanced" and ts["leader"] == "away"
    # same venue (not a second leg) / too old / unfinished -> None
    assert tie_state_from_h2h(
        _h2h("Jablonec", "Rangers", 1, 0, "2026-08-20T19:00:00Z"),
        home="Jablonec", away="Rangers", kickoff="2026-08-27T16:00:00Z",
    ) is None
    assert tie_state_from_h2h(
        _h2h("Rangers", "Jablonec", 1, 0, "2026-07-20T19:00:00Z"),
        home="Jablonec", away="Rangers", kickoff="2026-08-27T16:00:00Z",
    ) is None
    assert tie_state_from_h2h(None, home="A", away="B", kickoff="2026-08-27T16:00:00Z") is None


# --------------------------------------------------------------------------
# K4 -- suggestion
# --------------------------------------------------------------------------

TOTALS_UNDER = {"Over 2.5": {"odds": 2.25}, "Under 2.5": {"odds": 1.60},
                "BTTS Yes": {"odds": 2.11}, "BTTS No": {"odds": 1.70}}
CONS_EVEN = {"home": 1.9, "draw": 3.5, "away": 4.2}


def test_suggestion_is_none_when_no_market_is_dominant():
    """Celta v Osasuna 2026-08-27: Under @1.60 was 'suggested' at 58%."""
    out = compute_suggestion(totals=TOTALS_UNDER, consensus=CONS_EVEN, ah=None,
                             model_probs={"over_2.5": 0.4, "btts_yes": 0.45, "1x2": {"home": 0.5, "draw": 0.25, "away": 0.25}})
    assert out["pick"] is None
    assert out["blocked"] and all("tidak dominan" in b["reason"] for b in out["blocked"])


def test_suggestion_picks_dominant_market_when_model_agrees():
    out = compute_suggestion(
        totals={"Over 2.5": {"odds": 1.33}, "Under 2.5": {"odds": 3.2}},
        consensus={"home": 1.5, "draw": 4.2, "away": 6.0}, ah=None,
        model_probs={"over_2.5": 0.69, "1x2": {"home": 0.6, "draw": 0.2, "away": 0.2}},
    )
    assert out["pick"]["raw_label"] == "Over 2.5"


def test_suggestion_blocks_model_contradiction():
    """Partizan v Getafe: market Under @1.81 while lambda_total 2.76 (Over)."""
    out = compute_suggestion(
        totals={"Over 2.5": {"odds": 1.40}, "Under 2.5": {"odds": 2.9}}, consensus=None, ah=None,
        model_probs={"over_2.5": 0.40},
    )
    assert out["pick"] is None and "model" in out["blocked"][0]["reason"]
    # same market, model agrees -> suggested
    out = compute_suggestion(
        totals={"Over 2.5": {"odds": 1.40}, "Under 2.5": {"odds": 2.9}}, consensus=None, ah=None,
        model_probs={"over_2.5": 0.66},
    )
    assert out["pick"] and out["pick"]["raw_label"] == "Over 2.5"


def test_suggestion_no_directional_without_evidence():
    """Gangwon v Gwangju (K-League): both Elo prior, no xG -> no 1X2 suggestion."""
    out = compute_suggestion(
        totals=None, consensus={"home": 1.25, "draw": 5.5, "away": 9.0}, ah=None,
        model_probs={"elo_home_seeded": False, "elo_away_seeded": False, "lambda_source": "features",
                     "1x2": {"home": 0.75, "draw": 0.15, "away": 0.10}},
    )
    assert out["pick"] is None and "evidensi arah" in out["blocked"][0]["reason"]
    # with xG the same card is allowed
    out = compute_suggestion(
        totals=None, consensus={"home": 1.25, "draw": 5.5, "away": 9.0}, ah=None,
        model_probs={"elo_home_seeded": False, "elo_away_seeded": False, "lambda_source": "features+xg",
                     "1x2": {"home": 0.75, "draw": 0.15, "away": 0.10}},
    )
    assert out["pick"] and out["pick"]["raw_label"] == "Home Win"


def test_suggestion_no_directional_in_decided_tie_and_thin_form():
    cons = {"home": 7.7, "draw": 4.8, "away": 1.35}
    mp = {"elo_home_seeded": True, "elo_away_seeded": True, "1x2": {"home": 0.05, "draw": 0.15, "away": 0.80}}
    out = compute_suggestion(totals=None, consensus=cons, ah=None, model_probs=mp,
                             tie_state={"state": "decided", "leader": "away", "first_leg": "Besiktas 3-0 Kauno"})
    assert out["pick"] is None and "agregat" in out["blocked"][0]["reason"]
    out = compute_suggestion(totals=None, consensus={"home": 1.22, "draw": 6.1, "away": 11.6}, ah=None,
                             model_probs={"elo_home_seeded": True, "elo_away_seeded": True,
                                          "1x2": {"home": 0.8, "draw": 0.12, "away": 0.08}},
                             features={"form_home": "D-D", "form_away": "W-D-D-L-L"})
    assert out["pick"] is None and "form" in out["blocked"][0]["reason"]


def test_suggestion_settlement_mapping():
    cands = lean_candidates(None, None, {"line": -0.5, "home": 1.9, "away": 1.95})
    assert cands[0]["market"] == "Asian Handicap" and cands[0]["side"] == "home"
    sig = suggestion_for_settlement(cands[0])
    assert settle_signal(sig, 2, 0)["result"] == "win"
    assert settle_signal(suggestion_for_settlement({"market": "Total", "raw_label": "Under 2.5"}), 1, 0)["result"] == "win"
    out = select_suggestion([], model_probs={})
    assert out["pick"] is None and out["blocked"] == []


# --------------------------------------------------------------------------
# Engine wiring: K1 veto + K5 tier + K3 penalty + K2 veto
# --------------------------------------------------------------------------

def _engine(model_probs, **kw):
    base = dict(
        model_probs=model_probs,
        stats={"home_recent_goals": [1, 2, 0, 1, 2], "away_recent_goals": [0, 1, 1, 0, 2]},
        market_totals={"Over 2.5": {"odds": 1.9}, "Under 2.5": {"odds": 1.9},
                       "BTTS Yes": {"odds": 1.85}, "BTTS No": {"odds": 1.95}},
        ah_rows=[],
        odds_1x2={"home": 1.72, "draw": 3.8, "away": 4.15},
        completeness=1.0,
        cfg={"pick_gates": {"elo_integrity": True, "agreement": False, "lambda_total_sanity": False,
                            "source_consistency": True},
             "min_edge_pp": -50.0, "allow_negative_edge_pp": -50.0, "no_bet_score": 0.0,
             "min_confluence": 0, "min_data_quality": 0.0},
    )
    base.update(kw)
    return run_signal_engine(**base)


def _mp(**over):
    mp = {"1x2": {"home": 0.50, "draw": 0.24, "away": 0.26}, "over_2.5": 0.55, "btts_yes": 0.55,
          "lambda_home": 1.75, "lambda_away": 1.18, "lambda_source": "features+xg",
          "elo_seeded": True, "elo_home_seeded": True, "elo_away_seeded": True,
          "elo_home": 1700.0, "elo_away": 1650.0}
    mp.update(over)
    return mp


def test_engine_vetoes_directional_when_both_elo_prior():
    """Al Shabab v Al Riyadh 2026-08-25: Home Win @1.72 on two 1500 priors."""
    res = _engine(_mp(elo_seeded=False, elo_home_seeded=False, elo_away_seeded=False,
                      elo_home=1500.0, elo_away=1500.0))
    for e in res["ranking"]:
        if is_directional_selection(e["market"], e["selection"]):
            assert e["vetoed"], e
        else:
            assert not e["vetoed"] or "kedua tim" not in " ".join(e["veto_reasons"]), e
    assert res["elo_scope"] == "all"
    if res["best_pick"]:
        assert res["best_pick"]["confidence"] == "LOW"


def test_engine_vetoes_card_on_elo_out_of_band():
    """Rapid Wien 1291 v 'Hearts' 1031 (Kelty Hearts) 2026-08-26."""
    res = _engine(_mp(elo_home=1291.0, elo_away=1031.0))
    assert res["decision"] == "NO BET"
    assert any("1031" in r for r in res.get("disagreement_gate") or [])


def test_engine_caps_directional_medium_when_one_side_prior():
    res = _engine(_mp(elo_seeded=False, elo_away_seeded=False, elo_away=1500.0))
    assert res["elo_scope"] == "one"
    for e in res["ranking"]:
        if is_directional_selection(e["market"], e["selection"]):
            assert e["confidence"] in ("MEDIUM", "LOW", "NO SIGNAL"), e


def test_engine_source_consistency_vetoes_copenhagen_card():
    res = _engine(
        _mp(lambda_home=0.809, lambda_away=1.526, **{"1x2": {"home": 0.64, "draw": 0.18, "away": 0.18}}),
        odds_1x2={"home": 1.255, "draw": 5.6, "away": 9.6},
        team_form={"home": {"sequence": "D-L-L-L-L", "ga_avg": 5.2}, "away": {"sequence": "D-W-W-D-D", "ga_avg": 0.0}},
    )
    assert res["decision"] == "NO BET" and res["entity_mismatch"]["favourite"] == "home"


def test_engine_tier_is_lean_for_low_confidence_pick():
    res = _engine(_mp(), cfg={"pick_gates": {"elo_integrity": True, "agreement": False, "lambda_total_sanity": False},
                              "min_edge_pp": -50.0, "allow_negative_edge_pp": -50.0, "no_bet_score": 0.0,
                              "min_confluence": 0, "min_data_quality": 0.0, "medium_score": 0.99})
    assert res["best_pick"] is not None and res["pick_tier"] == "LEAN"


def test_engine_tie_state_penalises_leader_directional():
    ts = {"state": "decided", "leader": "home", "first_leg": "B 0-3 A", "agg_margin_home": 3}
    plain = _engine(_mp())
    with_ts = _engine(_mp(), context={"tie_state": ts})
    def _score(res, sel):
        return next(e["score"] for e in res["ranking"] if e["selection"] == sel)
    assert _score(with_ts, "Home Win") < _score(plain, "Home Win")
    assert abs(_score(with_ts, "Over 2.5") - _score(plain, "Over 2.5")) < 1e-9
    assert with_ts["tie_state"] == ts
    bal = _engine(_mp(), context={"tie_state": {"state": "balanced", "leader": None, "first_leg": "B 1-1 A", "agg_margin_home": 0}})
    assert _score(bal, "Over 2.5") < _score(plain, "Over 2.5")
    assert abs(_score(bal, "Home Win") - _score(plain, "Home Win")) < 1e-9


# --------------------------------------------------------------------------
# K5 / loop -- evaluation split + failure classes + suggestion persisted
# --------------------------------------------------------------------------

def test_evaluation_reports_tiers_suggestion_and_failure_classes():
    from agents.football.prediction_log import append_snapshot, best_pick_evaluation, settle

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "pred.jsonl"
        common = dict(
            league="UECL", kickoff="2026-08-27T17:00:00.000Z", prob={"home": 0.64, "draw": 0.18, "away": 0.18},
            odds={"home": 1.255, "draw": 5.6, "away": 9.6}, edge=None, confidence=0.5, signal=60,
            calibration=None, model_version="t", input_hash="x", sources=[],
        )
        append_snapshot(
            p, match_id="UECL||Copenhagen||Inter Turku||2026-08-27", home="Copenhagen", away="Inter Turku",
            best_pick={"market": "BTTS", "selection": "BTTS No", "score": 0.454, "confidence": "LOW", "market_odds": 1.757},
            signal_engine_pick={"decision": "BEST PICK", "tier": "LEAN", "market": "BTTS", "selection": "BTTS No",
                                "score": 0.454, "confidence": "LOW", "market_odds": 1.757},
            model_probs={"elo_home_seeded": False, "elo_away_seeded": True, "elo_home": 1500.0, "elo_away": 1682.0},
            suggestion={"pick": {"market": "1X2", "raw_label": "Home Win", "label": "1X2: Home Win",
                                 "odds": 1.255, "implied": 0.78, "adjusted_score": 0.74, "side": "home"},
                        "blocked": [], "floor": 0.52, "n_candidates": 2},
            **common,
        )
        append_snapshot(
            p, match_id="UECL||Kauno||Besiktas||2026-08-27", home="Kauno", away="Besiktas",
            best_pick={"market": "1X2", "selection": "Away Win", "score": 0.7, "confidence": "HIGH", "market_odds": 1.35},
            signal_engine_pick={"decision": "BEST PICK", "tier": "BEST PICK", "market": "1X2", "selection": "Away Win",
                                "score": 0.7, "confidence": "HIGH", "market_odds": 1.35, "side": "away"},
            model_probs={"elo_home_seeded": True, "elo_away_seeded": True, "elo_home": 1401.0, "elo_away": 1899.0},
            context_data={"tie_state": {"state": "decided", "leader": "away", "first_leg": "Besiktas 3-0 Kauno"}},
            suggestion={"pick": None, "blocked": [{"label": "1X2: Away Win", "reason": "agregat"}], "floor": 0.52, "n_candidates": 1},
            **common,
        )
        settle(p, match_id="UECL||Copenhagen||Inter Turku||2026-08-27", home_goals=4, away_goals=1)
        settle(p, match_id="UECL||Kauno||Besiktas||2026-08-27", home_goals=1, away_goals=0)
        ev = best_pick_evaluation(p)
        assert ev["n"] == 2
        assert ev["tiers"]["LEAN"]["losses"] == 1 and ev["tiers"]["BEST PICK"]["losses"] == 1
        classes = {r["selection"]: r["failure_class"] for r in ev["picks"]}
        assert classes["BTTS No"] == "K5"
        assert classes["Away Win"] == "K3"
        assert ev["suggestion"]["n"] == 1
        assert ev["suggestion"]["markets"]["1X2"]["wins"] == 1


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as t:
                    fn(Path(t))
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
