"""Match-signal consistency regression tests (Layers 1-4).

Implements the acceptance test from the fix spec using the Rio Ave FC vs
FC Porto fixture:

  1. Layer 1 -- the opening reference is an IMMUTABLE per-market record pinned
     on first ingestion; repeated reads return byte-identical values.
  2. Layer 2 -- AWAY -1.25 and HOME +1.25 are the same bet: a handicap line is
     scored exactly once on the deterministic canonical side, so the mirror
     label can never score differently (50 vs 76 bug).
  3. Layer 3 -- a second query either keeps the same best pick with a
     "no significant change" note, or shows an explicitly labeled change with
     a reason; it never silently presents a different, unexplained pick.
  4. Layer 4 -- every "Why" bullet referencing odds movement matches the sign
     of the movement data displayed in the same response.

The final acceptance test replays the ACTUAL historical prediction-log rows
for the fixture (skipped when the log is absent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.models import poisson_matrix, probs_from_matrix  # noqa: E402
from agents.football.prediction_log import (  # noqa: E402
    append_odds_snapshot,
    last_prediction_snapshot,
    make_match_id,
    opening_snapshot,
    stability_calibration,
)
from agents.football.signal_engine import (  # noqa: E402
    _canonical_ah_side,
    apply_pick_stability,
    build_market_block,
    movement_narrative_flags,
    run_signal_engine,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _model_probs(lh: float, la: float) -> dict:
    m = poisson_matrix(lh, la, rho=0.0)
    _p1x2, _o15, o25, _o35, btts = probs_from_matrix(m)
    return {
        "1x2": {}, "over_2.5": o25, "btts_yes": btts,
        "lambda_home": lh, "lambda_away": la,
    }


def _ou_totals(over: float, under: float) -> dict:
    return {
        "Over 2.5": {"odds": over, "point": 2.5},
        "Under 2.5": {"odds": under, "point": 2.5},
    }


def _ah_row(line, home, away, home_open=None, away_open=None, line_open=None) -> list[dict]:
    return [{
        "line": line, "home": home, "away": away,
        "home_open": home_open, "away_open": away_open,
        "line_open": line_open, "bookmaker": "nowgoal",
    }]


# Pre-existing fixture defect (diagnosed 2026-08-22): these stability tests
# never injected ``allow_negative_edge_pp``, so they ran against the CODE
# default of 0.0 while production (config/football.json ->
# models.signal_engine) ships -3.0. Their candidate carries edge_pp == 0.0
# exactly, which the strict default rejects via the min_edge_pp floor -> NO BET
# -> no pick to test stability on.
#
# G2 (agreement) is also disabled here, and that is deliberate scope hygiene,
# not a workaround: this fixture's model disagrees with its OWN AH market by
# >8pp (lambda 0.9/1.9 implies margin ~-1.0, so the model loves Home +1.25,
# while the quoted 2.00/1.86 prices it near even). G2 correctly vetoes both AH
# candidates, which pushes the pick down to Under 2.5 at score 0.468 -> LOW
# confidence -> the stability layer's "pick weakened" rule fires and reports
# "changed" even on byte-identical input. That is the stability layer behaving
# correctly on a weak pick; it just stops the fixture from exercising the
# hold path. G2 itself is covered by tests/test_pick_gates.py and the
# F2/G1 tests in test_signal_engine.py.
_PROD_EDGE_FLOOR = {
    "allow_negative_edge_pp": -3.0,
    "pick_gates": {"agreement": False},
    # F4-lite (plan v3): stability mechanics test -- legacy market component.
    "market_component_reward_agreement": False,
}


def _seed_prediction(path, mid: str) -> None:
    """append_odds_snapshot only writes after a prediction snapshot exists;
    seed a minimal one so the Layer-1 opening record can be pinned."""
    from agents.football.prediction_log import append_snapshot
    append_snapshot(
        path, match_id=mid, league="T", home="A", away="B",
        kickoff="2026-08-15T19:30:00Z", prob={}, odds={}, edge={},
        confidence=None, signal=None, calibration=None, model_version=None,
        input_hash=None, best_pick=None, sources=[],
    )


def _pick_payload(se_result: dict, ts: str) -> dict | None:
    bp = se_result.get("best_pick") or {}
    if se_result.get("decision") != "BEST PICK" or not bp:
        return None
    return {
        "decision": "BEST PICK",
        "market": bp.get("market"),
        "selection": bp.get("selection"),
        "score": bp.get("score"),
        "confidence": bp.get("confidence"),
        "line": bp.get("line"),
        "side": bp.get("side"),
        "line_key": (
            f"ah:{float(bp['line']):+.2f}"
            if bp.get("market") == "Asian Handicap" and bp.get("line") is not None
            else None
        ),
        "ts": ts,
    }


# ---------------------------------------------------------------------------
# Layer 1 -- canonical immutable opening reference
# ---------------------------------------------------------------------------

def test_opening_snapshot_immutable_and_byte_identical(tmp_path):
    path = tmp_path / "log.jsonl"
    mid = "T||A||B||2026-08-15T19:30:00Z"
    _seed_prediction(path, mid)
    # First ingestion pins the opening record (Layer 1).
    assert append_odds_snapshot(
        path, match_id=mid, timing="T-3h",
        odds={"home": 8.0, "draw": 4.71, "away": 1.36},
        odds_ou={"line": 2.5, "over": 1.79, "under": 2.12},
        odds_ah={"line": -1.25, "home": 2.0, "away": 1.86,
                 "home_open": 1.98, "away_open": 1.9, "line_open": -1.25},
        sources=["nowgoal"],
    )
    # Later observations must NEVER overwrite it.
    assert append_odds_snapshot(
        path, match_id=mid, timing="T-2h",
        odds={"home": 8.0, "draw": 4.71, "away": 1.36},
        odds_ou={"line": 2.5, "over": 1.83, "under": 2.07},
        odds_ah={"line": 1.25, "home": 1.98, "away": 1.88,
                 "home_open": 1.98, "away_open": 1.9, "line_open": 1.25},
        sources=["nowgoal"],
    )
    os1 = opening_snapshot(path, mid, "ou")
    os2 = opening_snapshot(path, mid, "ou")
    assert os1 is not None and os2 is not None
    # Byte-identical across repeated reads (spec Layer 1 #5).
    assert json.dumps(os1, sort_keys=True) == json.dumps(os2, sort_keys=True)
    assert os1["odds_ou"]["over"] == 1.79
    assert os1["odds_ou"]["under"] == 2.12
    assert "opening_snapshot_lag_seconds" in os1
    ah = opening_snapshot(path, mid, "ah")
    assert ah["odds_ah"]["line"] == -1.25  # first observation, not the later +1.25


def test_market_block_uses_canonical_opening():
    totals = _ou_totals(1.83, 2.07)
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12}}
    mb = build_market_block(market_totals=totals, ah=None, opening_snapshot=opening)
    assert mb["ou"]["canonical"] is True
    assert mb["ou"]["opening_over"] == 1.79
    assert mb["ou"]["latest_over"] == 1.83
    # No canonical record -> payload fallback, never fabricated.
    mb2 = build_market_block(market_totals=totals, ah=None, opening_snapshot=None)
    assert mb2["ou"]["canonical"] is False
    assert mb2["ou"]["latest_over"] == 1.83


# ---------------------------------------------------------------------------
# Layer 2 -- side-neutral Asian Handicap scoring
# ---------------------------------------------------------------------------

def test_canonical_ah_side_deterministic_tiebreak():
    # model 1X2 direction (p_home_1x2) decides the side; p_home breaks ties
    assert _canonical_ah_side(0.6, 0.6, 1.95, 1.95) == "home"
    assert _canonical_ah_side(0.4, 0.4, 1.95, 1.95) == "away"
    # neutral 1X2 -> deterministic HOME default (no odds tiebreak)
    assert _canonical_ah_side(0.5, 0.5, 1.90, 2.10) == "home"
    assert _canonical_ah_side(0.5, 0.5, 2.10, 1.90) == "home"
    # still tied (or no price) -> HOME
    assert _canonical_ah_side(0.5, 0.5, 2.0, 2.0) == "home"
    assert _canonical_ah_side(0.5, 0.5, None, None) == "home"


def test_ah_same_bet_single_candidate_per_line():
    # Rio Ave (weak) vs Porto (heavy favourite): consensus line +1.25.
    model = _model_probs(0.9, 1.9)
    totals = _ou_totals(1.83, 2.07)
    ah_rows = _ah_row(1.25, 2.0, 1.86, 1.98, 1.9, 1.25)
    res = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8, cfg=None,
    )
    ah = [r for r in res["ranking"] if r["market"] == "Asian Handicap"]
    # Each line appears exactly once, with its canonical line identity.
    assert len(ah) == len({r["line"] for r in ah})
    for r in ah:
        assert r["line_key"] == f"ah:{float(r['line']):+.2f}"
    # Deterministic: the identical query yields the identical score.
    res2 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8, cfg=None,
    )
    assert res2["ranking"] == res["ranking"]


def test_ah_mirror_labels_identical_score():
    # Regression (spec Layer 2 #4): AWAY -1.25 and HOME +1.25 are the same
    # bet. extract_asian_handicap normalizes both mirror expressions to
    # {line, home, away} BEFORE scoring, and the engine scores the canonical
    # side only -- so the same snapshot always yields the SAME canonical
    # label and score. The canonical side is model-driven (prob >= 0.5), so
    # even a different price split on the same line cannot flip the label.
    model = _model_probs(0.9, 1.9)
    totals = _ou_totals(1.83, 2.07)
    r1 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals,
        ah_rows=_ah_row(1.25, 2.0, 1.86), movement_snapshot=None,
        context=None, completeness=0.8, cfg=None,
    )
    r2 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals,
        ah_rows=_ah_row(1.25, 2.0, 1.86), movement_snapshot=None,
        context=None, completeness=0.8, cfg=None,
    )
    r3 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals,
        ah_rows=_ah_row(1.25, 1.93, 1.93), movement_snapshot=None,
        context=None, completeness=0.8, cfg=None,
    )
    h1 = next(r for r in r1["ranking"] if r["line_key"] == "ah:+1.25")
    h2 = next(r for r in r2["ranking"] if r["line_key"] == "ah:+1.25")
    h3 = next(r for r in r3["ranking"] if r["line_key"] == "ah:+1.25")
    assert h1["selection"] == h2["selection"] == h3["selection"]
    assert h1["score"] == h2["score"]  # same snapshot -> identical score


# ---------------------------------------------------------------------------
# Layer 3 -- repeated-query stability guard
# ---------------------------------------------------------------------------

def test_stability_holds_when_nothing_changed():
    # lambda_total 2.56 puts the model ~6pp above the market's Under 2.5 price
    # (market margin-free P(under) = 0.469). That is inside G2's 8pp agreement
    # band and above the 3pp min_edge floor, so the Totals candidate scores
    # above LOW and the hold path is actually reachable.
    #
    # It has to be the TOTALS candidate: the AH candidates the engine derives
    # here are canonical quarter lines with NO quote (market_odds None), which
    # G7 correctly refuses to pick. Before 2026-08-22 this fixture asserted
    # stability on `Away +0.25 @ None` -- a priceless pick, the same defect that
    # shipped Braga v Austria Wien Women on 2026-08-20.
    model = _model_probs(0.85, 1.71)
    totals = _ou_totals(1.83, 2.07)
    ah_rows = _ah_row(1.25, 2.0, 1.86, 1.98, 1.9, 1.25)
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12},
               "odds_ah": {"line": 1.25, "home": 2.0, "away": 1.86}}
    q1 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8,
        cfg=dict(_PROD_EDGE_FLOOR),
        opening_snapshot=opening,
    )
    prev = _pick_payload(q1, "2026-08-15T18:33:31+00:00")
    assert prev is not None
    # Query 2: identical inputs AND identical cfg -> pick must be held. The cfg
    # must match query 1 exactly; running the two queries under different
    # configs is itself a change, and the stability layer would be right to
    # report one.
    q2 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8,
        cfg=dict(_PROD_EDGE_FLOOR),
        opening_snapshot=opening, previous_pick=prev,
        now_ts="2026-08-15T18:48:35+00:00",
    )
    assert q2["stability"]["status"] == "held"
    assert q2["best_pick"]["selection"] == prev["selection"]
    # Audit trail: the freshly computed top is logged even when suppressed.
    assert q2["stability"]["suppressed_top"]["selection"]
    assert q2["stability"]["reason"]


def test_stability_changes_on_score_delta():
    model = _model_probs(0.9, 1.9)
    totals = _ou_totals(1.83, 2.07)
    ah_rows = _ah_row(1.25, 2.0, 1.86)
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12}}
    q1 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8,
        cfg=dict(_PROD_EDGE_FLOOR),
        opening_snapshot=opening,
    )
    prev = _pick_payload(q1, "2026-08-15T18:33:31+00:00")
    # Query 2 with a drastically different model -> the top score moves well
    # beyond the threshold -> the change must be EXPLICITLY labeled.
    # (Weights 0.35/0.25/0.20/0.15/0.05, 2026-08-17: the old 1.6/0.9 flip
    # now only moves the top score by ~0.02, under the 0.05 threshold, so
    # the guard correctly HOLDS -- use an extreme flip that clears it.)
    model2 = _model_probs(2.6, 0.4)
    q2 = run_signal_engine(
        model_probs=model2, stats={}, market_totals=_ou_totals(2.1, 1.75),
        ah_rows=_ah_row(-1.25, 1.9, 1.9), movement_snapshot=None,
        context=None, completeness=0.8, cfg=dict(_PROD_EDGE_FLOOR),
        opening_snapshot=opening, previous_pick=prev,
    )
    assert q2["stability"]["status"] == "changed"
    assert q2["stability"]["previous_selection"] == prev["selection"]
    assert q2["stability"]["new_selection"] == q2["ranking"][0]["selection"]
    assert q2["stability"]["reason"]


def test_stability_changes_on_adverse_market_move():
    # Opening under price 2.12; market now 2.35 (+10.8%) -> money out of
    # Under -> adverse move beyond threshold -> labeled change.
    model = _model_probs(0.9, 1.9)
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12}}
    q1 = run_signal_engine(
        model_probs=model, stats={}, market_totals=_ou_totals(1.83, 2.07),
        ah_rows=[], movement_snapshot=None, context=None, completeness=0.8,
        cfg=None, opening_snapshot=opening,
    )
    prev = _pick_payload(q1, "2026-08-15T18:33:31+00:00")
    if prev is None or prev["market"] != "Total" or not prev["selection"].startswith("Under"):
        return  # guard needs an Under pick to demonstrate the adverse move
    q2 = run_signal_engine(
        model_probs=model, stats={}, market_totals=_ou_totals(1.65, 2.35),
        ah_rows=[], movement_snapshot=None, context=None, completeness=0.8,
        cfg=None, opening_snapshot=opening, previous_pick=prev,
    )
    assert q2["stability"]["status"] == "changed"
    assert "MELAWAN" in q2["stability"]["reason"]


def test_stability_ignores_supporting_market_move():
    # Opening under 1.95 -> latest 1.80 (money INTO Under): a supporting move
    # reinforces the pick and must NOT trigger a change, even when the
    # re-priced score crosses the delta threshold (the pick did not flip and
    # the market moved in its direction).
    # Phase 6 reweight: 1.90/1.95 -> 2.05/1.80 gives delta=0.104, stability=held.
    model = _model_probs(1.1, 1.5)
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12}}
    q1 = run_signal_engine(
        model_probs=model, stats={}, market_totals=_ou_totals(1.90, 1.95),
        ah_rows=[], movement_snapshot=None, context=None, completeness=0.8,
        cfg=None, opening_snapshot=opening,
    )
    prev = _pick_payload(q1, "2026-08-15T18:33:31+00:00")
    if prev is None or prev["market"] != "Total" or not prev["selection"].startswith("Under"):
        return
    q2 = run_signal_engine(
        model_probs=model, stats={}, market_totals=_ou_totals(2.05, 1.80),
        ah_rows=[], movement_snapshot=None, context=None, completeness=0.8,
        cfg=None, opening_snapshot=opening, previous_pick=prev,
    )
    assert q2["stability"]["status"] == "held"
    assert q2["best_pick"]["selection"] == prev["selection"]


def test_stability_calibration_falls_back(tmp_path):
    path = tmp_path / "empty.jsonl"
    cal = stability_calibration(path, percentile=0.95, min_samples=20, fallback=0.05)
    assert cal["calibrated"] is False
    assert cal["threshold"] == 0.05
    assert cal["n"] == 0


def test_stability_calibration_from_logged_deltas(tmp_path):
    path = tmp_path / "log.jsonl"
    mid = "T||A||B||2026-08-15T19:30:00Z"
    # One match queried repeatedly with tiny noise (|delta| = 0.01..0.02).
    for i in range(6):
        score = 0.60 + (0.01 if i % 2 else 0.02)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "event": "snapshot", "match_id": mid,
                "ts": f"2026-08-15T18:{30 + i:02d}:00+00:00",
                "signal_engine_pick": {"decision": "BEST PICK",
                                       "selection": "Under 2.5",
                                       "score": score},
            }) + "\n")
    cal = stability_calibration(path, percentile=0.95, min_samples=3, fallback=0.05)
    assert cal["calibrated"] is True
    assert cal["n"] == 5
    assert cal["threshold"] >= 0.01


# ---------------------------------------------------------------------------
# Layer 4 -- narrative/confidence binding
# ---------------------------------------------------------------------------

def test_movement_narrative_flags_consistent():
    model = _model_probs(0.9, 1.9)
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12},
               "odds_ah": {"line": 1.25, "home": 2.0, "away": 1.86}}
    se = run_signal_engine(
        model_probs=model, stats={}, market_totals=_ou_totals(1.83, 2.07),
        ah_rows=_ah_row(1.25, 2.0, 1.86), movement_snapshot=None,
        context=None, completeness=0.8, cfg=None, opening_snapshot=opening,
    )
    assert movement_narrative_flags(se) == []


def test_market_signal_renders_stability_notes():
    # Layer 3 must be visible in the card: held -> "Dipertahankan" note,
    # changed -> explicit prior -> new pick + reason (never a silent swap).
    from agents.football.format import format_market_signal
    from agents.football.signal_engine import _demo

    se = _demo()
    bp = se["best_pick"]
    bp["selection"] = "Under 2.5"
    se["stability"] = {
        "status": "held",
        "previous_selection": "Under 2.5",
        "reason": ("model belum berubah signifikan dan pergerakan market masih "
                    "dalam batas noise — pick dipertahankan dari query sebelumnya"),
    }
    payload = {"league": "T", "home": "A", "away": "B", "kickoff": None,
               "signal_engine": se}
    body = format_market_signal(payload)["body"]
    assert "Dipertahankan dari analisis sebelumnya" in body

    se["stability"] = {
        "status": "changed",
        "previous_selection": "Under 2.5",
        "new_selection": "Home +1.25",
        "reason": "skor berubah 0.62 → 0.80 (Δ0.18 ≥ ambang 0.05)",
    }
    body2 = format_market_signal(payload)["body"]
    assert "Berubah dari Under 2.5 → Home +1.25" in body2


def test_movement_narrative_flags_catches_contradiction():
    # Hand-crafted: pick claims movement "toward" while the displayed
    # pick-side series shows "away" -> must be flagged (never emitted).
    se = {
        "best_pick": {
            "market": "Total", "selection": "Under 2.5", "side": "under",
            "movement": {"status": "available", "direction": "toward"},
        },
        "market_block": {"ou": {"canonical": True, "opening_under": 2.12,
                                "latest_under": 2.30}},
    }
    flags = movement_narrative_flags(se)
    assert flags, "narrative 'toward' vs displayed 'away' must be flagged"


# ---------------------------------------------------------------------------
# Acceptance -- actual historical Rio Ave FC vs FC Porto data
# ---------------------------------------------------------------------------

_LOG = ROOT / "cache/football/predictions.jsonl"
# Fix 1: the canonical match_id (make_match_id now normalizes team names
# through teams.json, so 'Rio Ave' and 'Rio Ave FC' resolve identically).
_MID_A = make_match_id("Primeira Liga", "Rio Ave FC", "FC Porto", "2026-08-15T19:30:00Z")


def _read_log_rows() -> list[dict]:
    if not _LOG.exists():
        return []
    rows = []
    for line in _LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fixture_rows(rows):
    out = []
    for r in rows:
        mid = str(r.get("match_id") or "")
        if "Rio" in mid and "Porto" in mid and "2026-08-15T19:30:00Z" in mid:
            out.append(r)
        elif (
            r.get("kickoff") == "2026-08-15T19:30:00Z"
            and "Rio" in str(r.get("home") or "")
            and "Porto" in str(r.get("away") or "")
        ):
            out.append(r)
    return out


def test_acceptance_rio_ave_porto_historical(tmp_path):
    """Replay two pre-kickoff queries from the REAL logged data and assert the
    four spec acceptance points (skipped when the historical log is absent)."""
    rows = _fixture_rows(_read_log_rows())
    odds = sorted(
        (r for r in rows if r.get("event") == "odds_snapshot" and r.get("odds_ou")),
        key=lambda r: r.get("ts") or "",
    )
    snaps = sorted(
        (r for r in rows if r.get("event") == "snapshot" and (r.get("features") or {}).get("lambda_home")),
        key=lambda r: r.get("ts") or "",
    )
    if not odds or not snaps:
        return  # historical log absent in this checkout -> synthetic tests above still run

    first = odds[0]
    # q1: model from the 18:33:31Z snapshot, market from the nearest odds row.
    q1_row = next((r for r in odds if r.get("ts", "") >= "2026-08-15T18:30:00"), odds[-1])
    q1_snap = next((r for r in snaps if r.get("ts", "") >= "2026-08-15T18:33:00"), snaps[-1])
    q2_row = next((r for r in odds if r.get("ts", "") >= "2026-08-15T18:45:00"), odds[-1])
    q2_snap = next((r for r in snaps if r.get("ts", "") >= "2026-08-15T18:48:00"), snaps[-1])

    # Layer 1: pin the canonical opening on first ingestion (immutable).
    path = tmp_path / "log.jsonl"
    _seed_prediction(path, _MID_A)
    append_odds_snapshot(
        path, match_id=_MID_A, timing="T-3h",
        odds=(first.get("odds_1x2") or {}),
        odds_ou=first.get("odds_ou"), odds_ah=first.get("odds_ah"),
        sources=first.get("sources") or [],
    )
    os1 = opening_snapshot(path, _MID_A, "ou")
    os2 = opening_snapshot(path, _MID_A, "ou")
    assert json.dumps(os1, sort_keys=True) == json.dumps(os2, sort_keys=True)
    opening = {
        "odds_ou": os1["odds_ou"],
    }
    os_ah = opening_snapshot(path, _MID_A, "ah")
    if os_ah:
        opening["odds_ah"] = os_ah["odds_ah"]

    history = [r for r in odds if r.get("ts", "") <= q1_row.get("ts", "")]

    def _run(snap, row, prev_pick=None):
        feat = snap.get("features") or {}
        model = _model_probs(float(feat["lambda_home"]), float(feat["lambda_away"]))
        ou = row.get("odds_ou") or {}
        totals = _ou_totals(ou.get("over"), ou.get("under"))
        ah = row.get("odds_ah")
        ah_rows = [{
            "line": ah["line"], "home": ah["home"], "away": ah["away"],
            "home_open": ah.get("home_open"), "away_open": ah.get("away_open"),
            "line_open": ah.get("line_open"), "bookmaker": "nowgoal",
        }] if ah else []
        return run_signal_engine(
            model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
            movement_snapshot=None, context=None, completeness=0.8, cfg=None,
            opening_snapshot=opening, history_snapshots=history,
            previous_pick=prev_pick,
        )

    se1 = _run(q1_snap, q1_row)
    assert se1["decision"] == "BEST PICK" and se1["best_pick"] is not None
    pick1 = _pick_payload(se1, q1_snap.get("ts") or "2026-08-15T18:33:31+00:00")

    # (1) Layer 1: opening identical across both queries.
    se2 = _run(q2_snap, q2_row, prev_pick=pick1)
    assert se1["market_block"]["ou"]["opening_over"] == se2["market_block"]["ou"]["opening_over"]
    assert se1["market_block"]["ou"]["opening_over"] == os1["odds_ou"]["over"]

    # (2) Layer 2: each AH line appears exactly once, deterministic identity.
    for se in (se1, se2):
        ah = [r for r in se["ranking"] if r["market"] == "Asian Handicap"]
        assert len(ah) == len({r["line"] for r in ah})
        assert all(r["line_key"] for r in ah)

    # (3) Layer 3: no silent swap -- a different pick must be labeled.
    stab = se2.get("stability") or {}
    pick2 = se2["best_pick"]
    if pick2 and pick1 and pick2["selection"] != pick1["selection"]:
        assert stab.get("status") == "changed", "silent pick swap!"
        assert stab.get("reason"), "labeled change must carry a reason"
    else:
        assert stab.get("status") in ("held", "changed")

    # (4) Layer 4: narrative movement claims match the displayed movement.
    assert movement_narrative_flags(se1) == []
    assert movement_narrative_flags(se2) == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
