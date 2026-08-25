"""Tests for the correction-spec gates (model separation, disagreement, pick
gates, multiplicative confidence, invariants)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import decision, model_gates
from agents.football.model_gates import (
    bucket_n,
    form_depth_shallow,
    model_a_probs,
    model_disagreement,
    pick_confidence,
    pick_status,
    set_bucket_table_path,
)

# synthetic bucket table: [0.50,0.55) and [0.65,1.0] are big enough; the
# 0.55-0.65 band is deliberately too small (< 30) to exercise the sample gate.
_TABLE = {
    "buckets": [
        {"lo": 0.0, "hi": 0.5, "n": 100, "predicted": 0.28, "actual": 0.28},
        {"lo": 0.5, "hi": 0.55, "n": 40, "predicted": 0.52, "actual": 0.53},
        {"lo": 0.55, "hi": 0.6, "n": 15, "predicted": 0.57, "actual": 0.58},
        {"lo": 0.6, "hi": 0.65, "n": 15, "predicted": 0.62, "actual": 0.60},
        {"lo": 0.65, "hi": 1.0, "n": 500, "predicted": 0.70, "actual": 0.71},
    ]
}


def _install_table(tmp_path: Path) -> None:
    p = tmp_path / "buckets.json"
    p.write_text(json.dumps(_TABLE), encoding="utf-8")
    set_bucket_table_path(p)


def test_model_a_probs_no_edge_keys():
    """Section 1/8#6: Model A dict must never carry edge/value/EV keys."""
    probs = model_a_probs(
        {"home": 2.10, "draw": 3.40, "away": 3.60},
        {"Over 2.5": {"odds": 1.90}, "Under 2.5": {"odds": 1.95}},
    )
    assert probs is not None
    assert set(probs.keys()) == {
        "1x2", "over_1.5", "over_2.5", "over_3.5", "btts_yes",
        "lambda_home", "lambda_away", "lambda_source",
    }
    assert abs(sum(probs["1x2"].values()) - 1.0) < 1e-5


def test_model_a_probs_degenerate_none():
    assert model_a_probs({"home": 0, "draw": 0, "away": 0}, {}) is None
    assert model_a_probs({}, {}) is None


def test_model_disagreement_flag():
    a = {"home": 0.60, "draw": 0.24, "away": 0.16}
    b_near = {"home": 0.66, "draw": 0.21, "away": 0.13}  # delta 6pp -> OK
    b_far = {"home": 0.44, "draw": 0.30, "away": 0.26}   # delta 16pp -> flag
    assert model_disagreement(a, b_near, threshold_pp=15.0)["flag"] is False
    assert model_disagreement(a, b_near, threshold_pp=15.0)["delta_pp"] == 6.0
    d = model_disagreement(a, b_far, threshold_pp=15.0)
    assert d["flag"] is True
    assert d["delta_pp"] == 16.0
    assert model_disagreement(None, b_far)["flag"] is False


def test_bucket_n_from_table(tmp_path):
    _install_table(tmp_path)
    assert bucket_n(0.52) == 40
    assert bucket_n(0.57) == 15
    assert bucket_n(0.62) == 15
    assert bucket_n(0.80) == 500
    assert bucket_n(0.30) == 100
    set_bucket_table_path(None)


# --------------------------------------------------------------------------
# pick_status (Section 2 hard gates)
# --------------------------------------------------------------------------

def test_pick_status_all_gates_pass():
    status, reasons = pick_status(
        ev=0.06, edge_pp=5.0, n_bucket=40, completeness=0.8,
    )
    assert status == "VALID"
    assert reasons == []


def test_pick_status_failures_ordered():
    # completeness floor is the most fundamental failure
    s, r = pick_status(ev=0.06, edge_pp=5.0, n_bucket=40, completeness=0.5)
    assert s == "INSUFFICIENT_DATA"
    # sample floor
    s, r = pick_status(ev=0.06, edge_pp=5.0, n_bucket=20, completeness=0.8)
    assert s == "INSUFFICIENT_SAMPLE"
    # extreme edge -> audit, regardless of EV sign
    s, r = pick_status(ev=0.5, edge_pp=25.0, n_bucket=40, completeness=0.8)
    assert s == "AUDIT_REQUIRED"
    s, r = pick_status(ev=-0.5, edge_pp=-25.0, n_bucket=40, completeness=0.8)
    assert s == "AUDIT_REQUIRED"
    # disagreement -> review, even with a positive EV
    s, r = pick_status(ev=0.06, edge_pp=5.0, n_bucket=40, completeness=0.8, disagreement=True)
    assert s == "REVIEW_REQUIRED"
    # EV <= 3% -> NO VALUE
    s, r = pick_status(ev=0.01, edge_pp=5.0, n_bucket=40, completeness=0.8)
    assert s == "NO VALUE"
    # Edge floor (2026-08-17 Galatasaray-Corum review): a "favourite at 1.30"
    # whose model probability merely mirrors the market (edge ~0pp) is NOT
    # value even with a marginally positive EV -- model 71.5% = implied 71.5%
    # is a market mirror, not an independent edge.
    s, r = pick_status(ev=0.04, edge_pp=0.5, n_bucket=40, completeness=0.8)
    assert s == "NO VALUE"
    assert any("edge" in x and "pp" in x for x in r)
    # A real edge still passes when EV clears the floor.
    s, r = pick_status(ev=0.06, edge_pp=6.0, n_bucket=40, completeness=0.8)
    assert s == "VALID"


# --------------------------------------------------------------------------
# pick_confidence (Section 3 multiplicative + Section 8 invariants)
# --------------------------------------------------------------------------

def test_pick_confidence_multiplicative():
    c = pick_confidence(calibration_score=0.9, n_bucket=60, completeness=1.0)
    # 0.9 * min(1, 60/30=2->1) * 1.0 = 0.9
    assert c["value"] == 0.9
    assert c["label"] == "HIGH"
    assert c["sample_factor"] == 1.0


def test_pick_confidence_sample_factor_scales():
    c = pick_confidence(calibration_score=0.9, n_bucket=15, completeness=1.0)
    # 0.9 * (15/30=0.5) = 0.45, plus LOW cap 0.49
    assert c["value"] <= 0.49
    assert c["label"] == "LOW"
    assert "n_bucket 15 < 30" in c["caps"][0]


def test_pick_confidence_invariant_high_needs_bucket(tmp_path):
    _install_table(tmp_path)
    # even perfect calibration cannot be HIGH with n_bucket 20 (< 30)
    c = pick_confidence(calibration_score=1.0, n_bucket=20, completeness=1.0)
    assert c["label"] != "HIGH"
    assert c["value"] < 0.7


def test_pick_confidence_invariant_high_needs_completeness():
    c = pick_confidence(calibration_score=1.0, n_bucket=100, completeness=0.5)
    assert c["label"] != "HIGH"
    assert c["value"] < 0.7


def test_pick_confidence_disagreement_max_medium():
    c = pick_confidence(
        calibration_score=1.0, n_bucket=100, completeness=1.0, disagreement=True,
    )
    assert c["label"] in ("LOW", "MEDIUM")
    assert c["value"] <= 0.69
    assert c["disagreement_penalty"] == 0.4


def test_pick_confidence_extreme_penalty():
    c = pick_confidence(
        calibration_score=0.9, n_bucket=100, completeness=1.0, extreme_edge=True,
    )
    assert c["extreme_edge_penalty"] == 0.5
    assert c["value"] == 0.45


def test_cap_is_ceiling_not_floor_regression():
    """Addendum v1.1 Section 4: a cap must never RAISE the displayed tier.

    GIVEN: raw pick_specific_confidence = 0.26 (maps to LOW)
    AND:   MODEL_DISAGREEMENT flag = True (cap = MEDIUM)
    THEN:  displayed tier MUST be LOW (min(LOW, MEDIUM) = LOW), not MEDIUM.
    """
    c = pick_confidence(
        calibration_score=1.0, n_bucket=30, completeness=0.65,
        disagreement=True,
    )
    # 1.0 * min(1, 30/30) * 0.65 * 1.0 * 0.4 = 0.26 -> LOW before caps
    assert round(c["value"], 2) == 0.26
    assert c["tier_before_caps"] == "LOW"
    # the MEDIUM cap is a ceiling: it must NOT raise LOW to MEDIUM
    assert c["label"] == "LOW"
    assert c["value"] == 0.26
    assert "MODEL_DISAGREEMENT (max MEDIUM)" in c["caps"]


def test_completeness_cap_lowers_medium_to_low():
    """A cap can only LOWER the tier: raw MEDIUM (0.55) + completeness cap
    LOW -> displayed LOW, with the value following the cap (0.49)."""
    c = pick_confidence(
        calibration_score=0.917, n_bucket=100, completeness=0.6,
        min_completeness=0.7,
    )
    # raw = 0.917 * 1.0 * 0.6 = 0.5502 -> MEDIUM before caps
    assert c["tier_before_caps"] == "MEDIUM"
    # completeness 0.6 < 0.7 -> cap LOW; min(MEDIUM, LOW) = LOW
    assert c["label"] == "LOW"
    assert c["value"] == 0.49


def test_build_confidence_block_allowlist():
    """Addendum v1.1 Section 5: the user-facing confidence block contains
    EXACTLY the allowlisted fields — no signal / decisiveness / legacy
    0-1 confidence key can sneak in."""
    from agents.football.model_gates import (
        CONFIDENCE_ALLOWLIST,
        build_confidence_block,
    )

    decision = {
        "model_calibration_score": 0.98,
        "pick_specific_confidence": {
            "value": 0.26,
            "label": "LOW",
            "tier_before_caps": "LOW",
            "caps": ["MODEL_DISAGREEMENT (max MEDIUM)"],
            "n_bucket": 0,
            "completeness_factor": 0.65,
        },
    }
    block = build_confidence_block(decision)
    assert set(block) == set(CONFIDENCE_ALLOWLIST)
    assert block["pick_specific_confidence"] == 0.26
    assert block["tier"] == "LOW"
    assert block["tier_before_caps"] == "LOW"
    assert block["caps_applied"] == ["MODEL_DISAGREEMENT (max MEDIUM)"]
    assert block["n_bucket"] == 0
    assert block["completeness_factor"] == 0.65
    # no decision / no pick-specific confidence -> no block (nothing to show)
    assert build_confidence_block({}) is None
    assert build_confidence_block({"pick_specific_confidence": None}) is None


# --------------------------------------------------------------------------
# decide() gating integration (production path passes bucket_n)
# --------------------------------------------------------------------------

def _cands(**overrides):
    model_probs = {
        "1x2": {"home": 0.60, "draw": 0.24, "away": 0.16},
        "over_2.5": 0.58,
        "over_3.5": 0.30,
        "btts_yes": 0.50,
    }
    consensus = {"home": 1.75, "draw": 3.80, "away": 4.90}  # home fair ~0.545
    market_totals = {
        "Over 2.5": {"odds": 1.90}, "Under 2.5": {"odds": 1.95},
    }
    cands = decision.build_candidates(
        model_probs=model_probs, consensus_odds=consensus,
        market_totals=market_totals, independent=True,
    )
    for k, v in overrides.items():
        for c in cands:
            if getattr(c, "selection", None) == k:
                setattr(c, k, v)
    return cands


def test_decide_gating_excludes_no_value(tmp_path):
    _install_table(tmp_path)
    # all candidates mirror the market -> EV ~0 -> NO VALUE -> NO BET
    cands = decision.build_candidates(
        model_probs={"1x2": {"home": 0.545, "draw": 0.26, "away": 0.195}},
        consensus_odds={"home": 1.75, "draw": 3.80, "away": 4.90},
        market_totals={}, independent=True,
    )
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.8, bookmakers_count=10,
        bucket_n=bucket_n,
    )
    assert d["decision_type"] == "NO BET"
    assert d["final_decision"] is None
    assert d["evaluated"]
    assert all(e["status"] != "VALID" for e in d["evaluated"])
    assert d["blocked"], "no-value candidates must be listed as blocked"
    assert d["blocked"][0]["status"] == "NO VALUE"


def test_decide_gating_insufficient_sample_blocks(tmp_path):
    _install_table(tmp_path)
    # _cands() puts the positive-EV totals candidate (model 0.58) in the
    # [0.55,0.60) bucket with n=15 (< 30) and Home Win in [0.60,0.65) n=15;
    # every candidate fails a gate -> NO CLEAR DECISION, no pick issued.
    cands = _cands()
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.8, bookmakers_count=10,
        bucket_n=bucket_n,
    )
    blocked_statuses = {e["status"] for e in d["blocked"]}
    assert "INSUFFICIENT_SAMPLE" in blocked_statuses
    assert d["final_decision"] is None
    assert d["decision_type"] == "NO CLEAR DECISION"


def test_decide_gating_extreme_edge_audit_and_propagation(tmp_path):
    _install_table(tmp_path)
    cands = _cands()
    for c in cands:
        if c.selection == "Away Win":
            c.edge_pp = 30.0
            c.ev = 0.5
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.8, bookmakers_count=10,
        bucket_n=bucket_n,
    )
    assert d["decision_type"] in ("NO BET", "NO CLEAR DECISION")
    assert d["final_decision"] is None
    assert any(e["status"] == "AUDIT_REQUIRED" for e in d["evaluated"])
    # Section 4: extreme edge propagates a confidence penalty to ALL markets
    psc = d["pick_specific_confidence"]
    assert psc is not None and psc["extreme_edge_penalty"] == 0.5


def test_decide_gating_disagreement_review(tmp_path):
    _install_table(tmp_path)
    cands = _cands()
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.8, bookmakers_count=10,
        bucket_n=bucket_n, disagreement=True,
    )
    # MODEL_DISAGREEMENT -> REVIEW_REQUIRED on the top candidate, confidence
    # capped at MEDIUM (never HIGH)
    assert d["final_decision"] is None
    assert d["decision_type"] in ("NO BET", "NO CLEAR DECISION")
    psc = d["pick_specific_confidence"]
    assert psc is not None and psc["label"] != "HIGH"
    assert psc["disagreement_penalty"] == 0.4
    assert psc["value"] <= 0.69


def test_decide_gating_valid_pick_allowed(tmp_path):
    _install_table(tmp_path)
    cands = _cands()
    # Under 2.5: model 0.70 lands in the [0.65,1.0] bucket (n=500 >= 30),
    # EV 0.365 > 3%, |edge| 19pp < 20pp -> VALID and eligible as final pick.
    for c in cands:
        if c.selection == "Under 2.5":
            c.model_prob = 0.70
            c.implied_prob = 0.51
            c.edge_pp = 19.0
            c.ev = 0.365
            c.market_odds = 1.95
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.85, bookmakers_count=10,
        bucket_n=bucket_n,
    )
    valid = [e for e in d["evaluated"] if e["status"] == "VALID"]
    assert valid, "a gated VALID candidate must exist"
    assert all(e["selection"] == "Under 2.5" for e in valid)
    # the VALID candidate is the only one allowed as final decision
    assert d["final_decision"] is not None
    assert d["final_decision"].pick_status == "VALID"
    # model_calibration_score vs pick_specific_confidence reported separately
    assert d["model_calibration_score"] == 0.9
    psc = d["pick_specific_confidence"]
    assert psc["value"] > 0
    # 0.9 * min(1, 500/30) * 0.85 = 0.765 -> HIGH is CORRECT here (both
    # n_bucket >= 30 and completeness >= 0.6 hold, invariants satisfied)
    assert psc["label"] == "HIGH"


def test_decide_legacy_without_bucket_n_unaffected():
    """Without bucket_n the legacy decision behavior is preserved."""
    cands = _cands()
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    assert "evaluated" in d
    assert d["pick_specific_confidence"] is None  # gating off -> no report


def test_decision_to_dict_serializes_gates():
    cands = _cands()
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.9, bookmakers_count=10,
    )
    payload = decision.decision_to_dict(d)
    assert "evaluated" in payload and "blocked" in payload
    assert "pick_specific_confidence" in payload
    import json as _json
    _json.dumps(payload, ensure_ascii=False)

# --------------------------------------------------------------------------
# P1: form-depth floor (form window < MIN_FORM_DEPTH matches/tim)
# --------------------------------------------------------------------------

def test_form_depth_shallow_counts_sequence_string():
    """A "W-D-L" sequence is 3 matches -> not shallow."""
    deep = {"sequence": "W-D-L"}
    assert form_depth_shallow(deep, deep) is False
    assert form_depth_shallow({"sequence": "W-D-L"}, {"sequence": "D-D-L"}) is False


def test_form_depth_shallow_flags_thin_windows():
    """1-2 match windows are noise, not signal -> shallow (P1)."""
    assert form_depth_shallow({"sequence": "W"}, {"sequence": "W-D-L"}) is True
    assert form_depth_shallow({"sequence": "W-D"}, {"sequence": "W-D-L"}) is True
    # either team shallow is enough
    assert form_depth_shallow({"sequence": "W-D-L"}, {"sequence": "L"}) is True


def test_form_depth_shallow_missing_form_is_shallow():
    assert form_depth_shallow(None, {"sequence": "W-D-L"}) is True
    assert form_depth_shallow({}, {"sequence": "W-D-L"}) is True
    assert form_depth_shallow(None, None) is True


def test_form_depth_shallow_recent_goals_fallback():
    """recent_goals (list of scorelines) is the fallback depth source."""
    deep = {"recent_goals": [(1, 0), (2, 1), (0, 0)]}
    assert form_depth_shallow(deep, {"sequence": "W-D-L"}) is False
    thin = {"recent_goals": [(1, 0)]}
    assert form_depth_shallow(thin, {"sequence": "W-D-L"}) is True


def test_form_depth_shallow_list_sequence():
    assert form_depth_shallow({"sequence": ["W", "D", "L"]}, {"sequence": "W-D-L"}) is False
    assert form_depth_shallow({"sequence": ["W"]}, {"sequence": "W-D-L"}) is True


def test_decide_form_depth_caps_confidence_and_flags(tmp_path):
    """P1: shallow form caps confidence at MEDIUM and records the cap."""
    _install_table(tmp_path)
    cands = _cands()
    for c in cands:
        if c.selection == "Under 2.5":
            c.model_prob = 0.70
            c.implied_prob = 0.51
            c.edge_pp = 19.0
            c.ev = 0.365
            c.market_odds = 1.95
    kw = dict(
        model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.85, bookmakers_count=10,
        bucket_n=bucket_n,
    )
    d = decision.decide(cands, **kw, form_depth_shallow=True)
    assert d["form_depth_cap_applied"] is True
    psc = d["pick_specific_confidence"]
    assert psc["label"] == "MEDIUM"  # never HIGH on shallow form
    assert any("form" in c and "MEDIUM" in c for c in psc["caps"])
    payload = decision.decision_to_dict(d)
    assert payload["form_depth_cap_applied"] is True


def test_decide_form_depth_cap_absent_without_flag(tmp_path):
    """Without the flag the confidence report is unchanged (cap is a ceiling,
    never applied silently)."""
    _install_table(tmp_path)
    cands = _cands()
    for c in cands:
        if c.selection == "Under 2.5":
            c.model_prob = 0.70
            c.implied_prob = 0.51
            c.edge_pp = 19.0
            c.ev = 0.365
            c.market_odds = 1.95
    d = decision.decide(
        cands, model_agreement=0.9, calibration_quality=0.9,
        calibration_samples=5000, completeness=0.85, bookmakers_count=10,
        bucket_n=bucket_n,
    )
    assert d["form_depth_cap_applied"] is False
    assert d["pick_specific_confidence"]["label"] == "HIGH"  # unchanged


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
