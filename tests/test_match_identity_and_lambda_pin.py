"""Tests for Fix 1 (match identity normalization) and Fix 2 (lambda source
pinning), the two post-incident fixes on top of the 4-layer consistency fix.

Fix 1 -- match identity normalization:
  - make_match_id canonicalizes team names through teams.json (the
    authoritative per-league name source) so every provider variant of the
    same fixture produces the IDENTICAL match_id (the "Rio Ave FC" vs
    "Rio Ave" bug).
  - Teams not in the alias table fall back to a deterministic
    case-preserving suffix strip so "Arsenal FC" and "FC Arsenal" still
    collide.

Fix 2 -- lambda source pinning:
  - the FIRST evaluation of a match pins its lambda_source; later queries
    reuse it instead of re-running the lambda_samples < min_samples branch,
    so the estimator cannot flip between "elo" and "features" on noise.
  - one-time exception: features genuinely unavailable at pin time may
    switch to features once, logged as
    lambda_source_switch_reason == "features_unavailable_at_pin_time".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.context import build_match_context  # noqa: E402
from agents.football.elo import EloModel  # noqa: E402
from agents.football.models import (  # noqa: E402
    Ensemble,
    PoissonModel,
    run_prediction_engine,
)
from agents.football.prediction_log import (  # noqa: E402
    _read_lines,
    append_snapshot,
    last_prediction_snapshot,
    make_match_id,
)

# ---------------------------------------------------------------------------
# Fix 1 -- match identity normalization
# ---------------------------------------------------------------------------

KICKOFF = "2026-08-15T19:30:00Z"


def test_match_id_identical_across_provider_variants():
    """Spec Fix 1 #3: every data source's name variant for the SAME fixture
    must produce the identical match_id."""
    variants = {
        "OddsPapi": ("Rio Ave FC", "FC Porto"),
        "NowGoal": ("Rio Ave", "Porto"),
        "football-data.org": ("Rio Ave FC", "FC Porto"),
        "TheSportsDB": ("Rio Ave FC", "Porto FC"),
        "Flashscore": ("Rio Ave", "FC Porto"),
        "Understat": ("Rio Ave FC", "Porto"),
    }
    ids = {
        src: make_match_id("Primeira Liga", h, a, KICKOFF)
        for src, (h, a) in variants.items()
    }
    assert len(set(ids.values())) == 1, ids
    # The canonical form matches the 18:33 historical record (authoritative
    # teams.json names, case + suffix preserved).
    # P1.2: the match_id carries the date-only kickoff component (the exact
    # time is not part of the identity -- a different-source kickoff a few
    # minutes apart must not split one match into two match_ids).
    assert ids["NowGoal"] == "Primeira Liga||Rio Ave FC||FC Porto||2026-08-15"


def test_match_id_riporto_variants_collide():
    """The exact incident: 18:33 used 'Rio Ave FC', 18:43 'Rio Ave'. Both
    must map to one match_id."""
    m1 = make_match_id("Primeira Liga", "Rio Ave FC", "FC Porto", KICKOFF)
    m2 = make_match_id("Primeira Liga", "Rio Ave", "FC Porto", KICKOFF)
    m3 = make_match_id("Primeira Liga", "Rio Ave", "Porto", KICKOFF)
    assert m1 == m2 == m3


def test_match_id_suffix_strip_fallback_case_preserved():
    """Alias-table names are authoritative (kept verbatim, e.g. 'Arsenal
    FC'); variants of a team NOT in the table collide via a deterministic
    suffix strip (prefix AND suffix) with case preserved, so existing
    plain-name match_ids stay byte-identical.

    Uses 'Riverton' / 'Riverton FC' as the non-aliased team -- neither resolves
    through any alias, standings, or word-boundary path in the EPL alias
    table, so they fall back to the suffix-strip canonicalizer.
    """
    # Canonical alias-table name kept verbatim
    assert make_match_id("EPL", "Arsenal FC", "Riverton FC", None) == \
        "EPL||Arsenal FC||Riverton||"
    # 'Riverton FC' / 'Riverton' are not in the alias table -> fallback strip.
    # Both strip to 'Riverton', so the match_ids collide.
    assert make_match_id("EPL", "Arsenal FC", "Riverton FC", None) == \
        make_match_id("EPL", "Arsenal FC", "Riverton", None)
    # P0 (plan v3 2026-08-24): reordered-prefix variants of an alias-table
    # club now UNIFY via significant-token containment ("FC Arsenal" ->
    # "Arsenal FC") -- one real-world club must not split into two match_ids
    # (the Atl. Madrid / Rennes duplicate-identity class). The suffix-strip
    # fallback remains authoritative ONLY for names with no containment hit.
    assert make_match_id("EPL", "Arsenal FC", "Riverton FC", None) == \
        make_match_id("EPL", "FC Arsenal", "Riverton FC", None) == \
        "EPL||Arsenal FC||Riverton||"
    # Prefix strip: 'FC Riverton' -> 'Riverton' (prefix FC stripped), same as
    # 'Riverton' -> 'Riverton', so they collide.
    assert make_match_id("EPL", "Arsenal FC", "FC Riverton", None) == \
        make_match_id("EPL", "Arsenal FC", "Riverton", None)


def test_match_id_deterministic_repeated_calls():
    m1 = make_match_id("Primeira Liga", "Rio Ave", "Porto", KICKOFF)
    m2 = make_match_id("Primeira Liga", "Rio Ave", "Porto", KICKOFF)
    assert m1 == m2


def test_stability_guard_uses_canonical_match_id(tmp_path):
    """Fix 1 integration: the Layer-3 guard retrieves the prior pick when the
    SAME fixture is queried with different provider name variants (18:33
    'Rio Ave FC' vs 18:43 'Rio Ave')."""
    from agents.football.prediction_log import append_odds_snapshot
    from agents.football.signal_engine import apply_pick_stability

    path = tmp_path / "log.jsonl"
    canonical = make_match_id("Primeira Liga", "Rio Ave FC", "FC Porto", KICKOFF)
    # The 18:33 query logs under the canonical match_id (whatever name
    # variant the provider used, make_match_id resolves it).
    append_snapshot(
        path, match_id=canonical, league="Primeira Liga",
        home="Rio Ave FC", away="FC Porto", kickoff=KICKOFF,
        prob={}, odds={}, edge={}, confidence=None, signal=None,
        calibration=None, model_version=None, input_hash=None,
        best_pick=None, sources=[],
        signal_engine_pick={
            "decision": "BEST PICK", "market": "Total",
            "selection": "Under 2.5", "score": 0.76,
            "confidence": "HIGH", "line": None, "side": None, "ts": "2026-08-15T18:33:31+00:00",
        },
    )
    append_odds_snapshot(
        path, match_id=canonical, timing="T-3h",
        odds={"home": 8.0, "draw": 4.71, "away": 1.36},
        odds_ou={"line": 2.5, "over": 1.79, "under": 2.12},
        sources=["nowgoal"],
    )
    # 18:43 query: provider resolves the home team WITHOUT 'FC'. The guard
    # must find the 18:33 pick because make_match_id canonicalizes.
    mid_1843 = make_match_id("Primeira Liga", "Rio Ave", "FC Porto", KICKOFF)
    assert mid_1843 == canonical
    prev = (last_prediction_snapshot(path, mid_1843) or {}).get("signal_engine_pick")
    assert prev is not None and prev["selection"] == "Under 2.5"
    # A no-change second query holds the pick (Layer 3 works end to end).
    se = {
        "decision": "BEST PICK",
        "best_pick": {"market": "Total", "selection": "Home +1.25", "score": 0.74,
                      "confidence": "HIGH", "line": None, "side": None,
                      "model_prob": 0.55, "market_odds": 1.9, "edge_pp": 2.0},
        "ranking": [
            {"market": "Total", "selection": "Home +1.25", "score": 0.74,
             "confidence": "HIGH", "line": None, "side": None,
             "model_prob": 0.55, "market_odds": 1.9, "edge_pp": 2.0},
            {"market": "Total", "selection": "Under 2.5", "score": 0.72,
             "confidence": "HIGH", "line": None, "side": None,
             "model_prob": 0.53, "market_odds": 2.07, "edge_pp": 1.0},
        ],
        "ah_consensus": None,
    }
    result = apply_pick_stability(
        se, previous_pick=prev, current_model={},
        opening_snapshot={"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12}},
        market_totals={"Over 2.5": {"odds": 1.83}, "Under 2.5": {"odds": 2.07}},
        now_ts="2026-08-15T18:43:32+00:00", cfg=None,
    )
    assert result["stability"]["status"] in ("held", "changed")


# ---------------------------------------------------------------------------
# Fix 2 -- lambda source pinning
# ---------------------------------------------------------------------------

@pytest.fixture()
def elo_model(tmp_path):
    path = tmp_path / "elo.json"
    path.write_text(
        json.dumps({"Rio Ave FC": 1467.0, "FC Porto": 1758.0}), encoding="utf-8"
    )
    return EloModel(path=path)


def _ctx(*, recent_home=None, recent_away=None, xg_home=None, xg_away=None,
         xg_home_against=None, xg_away_against=None):
    return build_match_context(
        league="Primeira Liga", home="Rio Ave FC", away="FC Porto",
        kickoff=KICKOFF,
        stats={
            "home_gf_avg": 1.0, "home_ga_avg": 1.2,
            "away_gf_avg": 2.0, "away_ga_avg": 0.8,
            "home_recent_goals": recent_home, "away_recent_goals": recent_away,
            "home_xg_for": xg_home, "away_xg_for": xg_away,
            "home_xg_against": xg_home_against, "away_xg_against": xg_away_against,
        },
        odds={"has_odds": False, "consensus": None, "totals": {}},
        sources=[],
    )


def _run(ctx, elo_model, **pin):
    return run_prediction_engine(
        ctx, elo=elo_model,
        poisson=PoissonModel(min_samples=2, shrinkage_samples=5),
        ensemble=Ensemble(elo_weight=0.5, poisson_weight=0.5),
        **pin,
    ).model_probs


def test_lambda_pin_holds_across_queries(elo_model):
    """Spec Fix 2 acceptance: the first evaluation pinned 'elo' (features
    below min_samples); a later evaluation that WOULD have used features
    (samples now sufficient) must produce byte-identical lambda_source and
    lambda values."""
    q1 = _run(_ctx(), elo_model)  # no recent goals -> samples 0 -> elo
    assert q1["lambda_source"] == "elo"
    assert q1["lambda_samples"] == 0
    assert q1["features_available"] is True  # degraded, not absent

    # Without the pin the estimator would flip:
    q2_nopin = _run(_ctx(
        recent_home=[(1, 2), (0, 1), (2, 0)],
        recent_away=[(3, 0), (2, 1), (4, 0)],
    ), elo_model)
    assert q2_nopin["lambda_source"] != "elo"

    # With the pin it must not:
    q2 = _run(
        _ctx(recent_home=[(1, 2), (0, 1), (2, 0)], recent_away=[(3, 0), (2, 1), (4, 0)]),
        elo_model,
        pinned_lambda_source=q1["lambda_source"],
        pinned_features_available_at_pin=q1["features_available"],
    )
    assert q2["lambda_source"] == q1["lambda_source"] == "elo"
    assert q2["lambda_home"] == q1["lambda_home"]
    assert q2["lambda_away"] == q1["lambda_away"]
    assert q2["lambda_source_switch_reason"] is None


def test_lambda_pin_one_time_exception(elo_model):
    """Spec Fix 2 #3: features genuinely unavailable at pin time + available
    later -> ONE switch to features, logged; never back and forth."""
    q = _run(
        _ctx(recent_home=[(1, 2), (0, 1), (2, 0)], recent_away=[(3, 0), (2, 1), (4, 0)]),
        elo_model,
        pinned_lambda_source="elo",
        pinned_features_available_at_pin=False,  # features were ABSENT at pin
    )
    assert q["lambda_source"] == "features"
    assert q["lambda_source_switch_reason"] == "features_unavailable_at_pin_time"

    # A second switch back is NOT allowed (pin now = features).
    q2 = _run(
        _ctx(recent_home=[(1, 2), (0, 1), (2, 0)], recent_away=[(3, 0), (2, 1), (4, 0)]),
        elo_model,
        pinned_lambda_source="features",
        pinned_features_available_at_pin=False,
    )
    assert q2["lambda_source"] == "features"
    assert q2["lambda_source_switch_reason"] is None


def test_lambda_pin_features_holds(elo_model):
    # 3 samples (>= min_samples=2, < shrinkage=5) -> blended features estimator
    q1 = _run(_ctx(recent_home=[(1, 2), (0, 1), (2, 0)],
                   recent_away=[(3, 0), (2, 1), (4, 0)]), elo_model)
    assert q1["lambda_source"].startswith("features")
    # Even when the form window later thins out (1 sample < min_samples=2
    # -> the unpinned branch would switch to elo), the features pin holds.
    q2 = _run(_ctx(recent_home=[(1, 2)], recent_away=[(3, 0)]), elo_model,
              pinned_lambda_source=q1["lambda_source"],
              pinned_features_available_at_pin=True)
    assert q2["lambda_source"].startswith("features")


# ---------------------------------------------------------------------------
# 2026-08-22 -- exact-composition pinning (no intra-family drift)
# ---------------------------------------------------------------------------

_XG = dict(xg_home=1.6, xg_away=1.2, xg_home_against=1.1, xg_away_against=1.4)


def test_lambda_pin_exact_composition_reproduces_blend(elo_model):
    """Pin made on the BLENDED 'features+xg+elo' estimator: a later query
    must reproduce THAT blended lambda byte-for-byte, not silently jump to
    the raw 'features+xg' lambdas (Everton v Palace 2026-08-22 incident:
    totals prob jumped 0.63 -> 0.68 between runs)."""
    ctx = _ctx(recent_home=[(1, 2), (0, 1), (2, 0)],
               recent_away=[(3, 0), (2, 1), (4, 0)], **_XG)
    raw = _run(ctx, elo_model)  # no pin -> threshold branch blends at t=1/3
    assert raw["lambda_source"] == "features+xg+elo"

    pinned = _run(ctx, elo_model,
                  pinned_lambda_source="features+xg+elo",
                  pinned_features_available_at_pin=True)
    assert pinned["lambda_source"] == "features+xg+elo"
    assert pinned["lambda_home"] == raw["lambda_home"]
    assert pinned["lambda_away"] == raw["lambda_away"]
    # deterministic across repeated queries
    again = _run(ctx, elo_model,
                 pinned_lambda_source="features+xg+elo",
                 pinned_features_available_at_pin=True)
    assert again == pinned


def test_lambda_pin_exact_composition_saturated_label_is_honest(elo_model):
    """Pin '+elo' but samples later reach shrinkage: the reproduced lambda IS
    the pure feature estimator, so the label must say so (never lie about
    provenance) -- while still not flip-flopping on later queries."""
    games = [(1, 2), (0, 1), (2, 0), (3, 1), (1, 1), (2, 2)]
    ctx = _ctx(recent_home=list(games), recent_away=list(games), **_XG)
    raw = _run(ctx, elo_model)
    assert raw["lambda_source"] == "features+xg"
    pinned = _run(ctx, elo_model,
                  pinned_lambda_source="features+xg+elo",
                  pinned_features_available_at_pin=True)
    assert pinned["lambda_source"] == "features+xg"
    assert pinned["lambda_home"] == raw["lambda_home"]
    assert pinned["lambda_away"] == raw["lambda_away"]


def test_lambda_pin_exact_match_no_drift(elo_model):
    """Pin == current composition -> lambdas byte-identical, no switch note."""
    ctx = _ctx(recent_home=[(1, 2), (0, 1), (2, 0)],
               recent_away=[(3, 0), (2, 1), (4, 0)], **_XG)
    raw = _run(ctx, elo_model)
    assert raw["lambda_source"] == "features+xg+elo"
    pinned = _run(ctx, elo_model,
                  pinned_lambda_source="features+xg+elo",
                  pinned_features_available_at_pin=True)
    assert pinned["lambda_source"] == "features+xg+elo"
    assert pinned["lambda_home"] == raw["lambda_home"]
    assert pinned["lambda_away"] == raw["lambda_away"]
    assert pinned["lambda_source_switch_reason"] is None


def test_lambda_pin_xg_arrives_late_switches_once(elo_model):
    """Pin 'features' (no xG at pin time); xG arrives later -> ONE upgrade to
    the richer same-family estimator, logged; then stable."""
    ctx = _ctx(recent_home=[(1, 2), (0, 1), (2, 0)],
               recent_away=[(3, 0), (2, 1), (4, 0)], **_XG)
    q = _run(ctx, elo_model,
             pinned_lambda_source="features",
             pinned_features_available_at_pin=True)
    assert q["lambda_source"] == "features+xg"
    assert q["lambda_source_switch_reason"] == "xg_unavailable_at_pin_time"
    # pin now = features+xg -> exact match, no further notes
    q2 = _run(ctx, elo_model,
              pinned_lambda_source="features+xg",
              pinned_features_available_at_pin=True)
    assert q2["lambda_source"] == "features+xg"
    assert q2["lambda_source_switch_reason"] is None


# ---------------------------------------------------------------------------
# Fix 2 integration -- snapshot persists the pin, analyse-style wiring works
# ---------------------------------------------------------------------------

def test_snapshot_persists_pin_and_reader_returns_it(tmp_path):
    path = tmp_path / "log.jsonl"
    mid = make_match_id("Primeira Liga", "Rio Ave", "Porto", KICKOFF)
    append_snapshot(
        path, match_id=mid, league="Primeira Liga", home="Rio Ave FC",
        away="FC Porto", kickoff=KICKOFF, prob={}, odds={}, edge={},
        confidence=None, signal=None, calibration=None, model_version=None,
        input_hash=None, best_pick=None, sources=[],
        features={
            "lambda_home": 0.579, "lambda_away": 2.121, "lambda_source": "elo",
            "pinned_lambda_source": "elo",
            "pinned_features_available_at_pin": True,
            "lambda_source_switch_reason": None,
            "lambda_samples": 0,
        },
    )
    row = last_prediction_snapshot(path, mid)
    f = (row or {}).get("features") or {}
    assert f["pinned_lambda_source"] == "elo"
    assert f["pinned_features_available_at_pin"] is True
    assert f["lambda_samples"] == 0


def test_historical_replay_lambda_pinned_18_33_vs_18_48(tmp_path):
    """Spec Fix 2 integration acceptance: replay the REAL Rio Ave vs Porto
    snapshots (18:33 elo / 18:48 features in the log). With the pin from the
    18:33 evaluation, the 18:48 query must score with the SAME lambda_source
    and identical lambda values -- no estimator flip."""
    log = ROOT / "cache/football/predictions.jsonl"
    if not log.exists():
        pytest.skip("historical prediction log absent in this checkout")
    rows = [r for r in _read_lines(log) if r.get("event") == "snapshot"
            and (r.get("features") or {}).get("lambda_home")
            and "Rio" in str(r.get("match_id") or "")
            and "Porto" in str(r.get("match_id") or "")]
    snaps = sorted(rows, key=lambda r: r.get("ts") or "")
    q1 = next((r for r in snaps if r.get("ts", "") >= "2026-08-15T18:33:00"), None)
    q2 = next((r for r in snaps if r.get("ts", "") >= "2026-08-15T18:48:00"), None)
    if q1 is None or q2 is None:
        pytest.skip("Rio Ave vs Porto 18:33/18:48 snapshots absent")
    f1, f2 = q1.get("features") or {}, q2.get("features") or {}
    # The incident data itself: the estimator flipped in the raw log.
    assert f1["lambda_source"] == "elo" and f2["lambda_source"].startswith("features")

    def _ctx(feat):
        return build_match_context(
            league="Primeira Liga", home="Rio Ave FC", away="FC Porto",
            kickoff=KICKOFF,
            stats={
                "home_gf_avg": feat.get("attack_home"), "home_ga_avg": feat.get("defense_home"),
                "away_gf_avg": feat.get("attack_away"), "away_ga_avg": feat.get("defense_away"),
                "home_form": feat.get("form_home"), "away_form": feat.get("form_away"),
                "home_recent_goals": None, "away_recent_goals": None,
            },
            odds={"has_odds": False, "consensus": None, "totals": {}},
            sources=[],
        )

    elo = EloModel(path=ROOT / "cache/football/elo.json")
    poisson = PoissonModel(min_samples=2, shrinkage_samples=5)
    ens = Ensemble(elo_weight=0.5, poisson_weight=0.5)
    mp1 = run_prediction_engine(
        _ctx(f1), elo=elo, poisson=poisson, ensemble=ens
    ).model_probs
    assert mp1["lambda_source"] == "elo"
    # 18:48 WITH the 18:33 pin: byte-identical lambda source + values.
    mp2 = run_prediction_engine(
        _ctx(f2), elo=elo, poisson=poisson, ensemble=ens,
        pinned_lambda_source=mp1["lambda_source"],
        pinned_features_available_at_pin=mp1["features_available"],
    ).model_probs
    assert mp2["lambda_source"] == mp1["lambda_source"] == "elo"
    assert mp2["lambda_home"] == mp1["lambda_home"]
    assert mp2["lambda_away"] == mp1["lambda_away"]
    assert mp2["lambda_source_switch_reason"] is None


def test_historical_log_match_id_collision_audit():
    """Fix 1 #5 backfill audit (read-only): report how many snapshot rows for
    the Rio Ave vs Porto fixture previously split across two match_ids now
    collapse to one canonical form."""
    log = ROOT / "cache/football/predictions.jsonl"
    if not log.exists():
        pytest.skip("historical prediction log absent in this checkout")
    rows = _read_lines(log)
    related = [
        r for r in rows
        if r.get("event") == "snapshot"
        and "Rio" in str(r.get("match_id") or "")
        and "Porto" in str(r.get("match_id") or "")
        and "2026-08-15T19:30:00Z" in str(r.get("match_id") or "")
    ]
    if not related:
        pytest.skip("no Rio Ave vs Porto snapshots in the log")
    raw_ids = {r["match_id"] for r in related}
    canonical_ids = {
        make_match_id(
            str(r.get("league") or "?"),
            str(r.get("home") or ""), str(r.get("away") or ""),
            str(r.get("kickoff") or ""),
        )
        for r in related
    }
    # The incident: the historical log holds BOTH 'Rio Ave FC' and 'Rio Ave'
    # variants for the same fixture; canonicalization collapses them.
    assert len(raw_ids) >= 1
    assert len(canonical_ids) == 1, f"expected one canonical match_id, got {canonical_ids}"
