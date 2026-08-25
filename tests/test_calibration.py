"""Tests for calibration.py."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.calibration import (
    Calibrator,
    SignalScorer,
    completeness_level,
    expected_calibration_error,
)
from agents.football.context import MatchContext


def test_expected_calibration_error_perfect():
    probs: list[float] = []
    outcomes: list[int] = []
    for i in range(1, 10):
        p = i / 10.0
        probs += [p] * 10
        outcomes += [1] * i + [0] * (10 - i)
    assert expected_calibration_error(probs, outcomes) < 1e-6


def test_expected_calibration_error_miscalibrated():
    probs = [0.9, 0.9, 0.9, 0.9]
    outcomes = [0, 0, 0, 0]
    assert expected_calibration_error(probs, outcomes) > 0.8


def test_calibrator_not_fitted_returns_input():
    cal = Calibrator(min_samples=200)
    assert cal.samples == 0
    assert cal.apply(0.6) == 0.6
    q = cal.quality()
    assert q["quality"] == 0.0
    assert q["samples"] == 0


def test_calibrator_fit_recovers_logit_linear_curve():
    """IRLS logistic regression must recover an exactly logit-linear curve."""
    from agents.football.calibration import _logit, _sigmoid

    cal = Calibrator(min_samples=1)
    probs = [0.05 + 0.05 * i for i in range(19)]  # 0.05 .. 0.95
    ys = [_sigmoid(0.3 + 1.7 * _logit(p)) for p in probs]
    cal.fit(probs, ys)
    assert cal.samples == len(probs)
    assert abs(cal.a - 0.3) < 0.05
    assert abs(cal.b - 1.7) < 0.1
    applied = [cal.apply(p) for p in probs]
    for a, y in zip(applied, ys):
        assert abs(a - y) < 0.02
    assert all(b >= a for a, b in zip(applied, applied[1:]))
    assert cal.quality()["samples"] == len(probs)


def test_calibrator_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.json"
        cal = Calibrator(path=path, min_samples=1)
        probs = [0.1 + 0.05 * i for i in range(18)]
        outcomes = [1 if p > 0.5 else 0 for p in probs]
        cal.fit(probs, outcomes)
        cal2 = Calibrator(path=path, min_samples=1)
        assert cal2.samples == cal.samples
        assert math.isclose(cal2.a, cal.a, abs_tol=1e-9)


def test_calibrator_min_samples_gate():
    cal = Calibrator(min_samples=1000)
    probs = [0.1 + 0.05 * i for i in range(18)]
    outcomes = [1 if p > 0.5 else 0 for p in probs]
    cal.fit(probs, outcomes)
    assert cal.apply(0.5) == 0.5  # gated: no-op until enough samples


def test_signal_scorer_components_separated():
    ctx = MatchContext(
        league="EPL", home="A", away="B",
        home_form="W-W-W", away_form="L-L-L",
        home_gf_avg=2.0, home_ga_avg=1.0, away_gf_avg=1.0, away_ga_avg=2.0,
        home_xg_for=1.8, home_xg_against=1.0, away_xg_for=1.0, away_xg_against=1.8,
        h2h={"wins": 2, "draws": 0, "losses": 1},
        consensus_odds={"home": 2.0, "draw": 3.4, "away": 3.8},
    )
    scorer = SignalScorer()
    comps = scorer.components(
        ctx=ctx,
        ensemble_models=["elo", "poisson"],
        model_vs_market=0.9,
        model_vs_model=0.8,
        calibration_quality=0.6,
        market_edge={"home": 2.0, "draw": -1.0, "away": -1.0},
    )
    assert comps["data_completeness"] >= 0.9
    assert comps["model_agreement"] > 0.8
    assert comps["market_edge_pct"] == 2.0
    assert 0 <= comps["signal_strength"] <= 100
    assert 0 <= comps["confidence"] <= 1


def test_signal_scorer_missing_data_low_completeness():
    ctx = MatchContext(league="EPL", home="A", away="B")
    scorer = SignalScorer()
    comps = scorer.components(
        ctx=ctx,
        ensemble_models=["elo"],
        model_vs_market=None,
        model_vs_model=None,
        calibration_quality=0.0,
        market_edge={},
    )
    assert comps["data_completeness"] == 0.0
    # PHASE 3: completeness 0 -> LOW level, confidence capped at 0.49
    assert comps["data_completeness_level"] == "LOW"
    assert comps["confidence"] <= 0.49


# --------------------------------------------------------------------------
# D2 (2026-08-17): per-league calibration refresh -- dynamic leagues can now
# accumulate their own fit from the LIVE log instead of being permanently
# stuck on the uncalibrated_league cap (per-league files were previously
# only seeded from football-data.co.uk history by seed-league).
# --------------------------------------------------------------------------

def _write_log(tmp: str, rows: list[dict]) -> str:
    """Write one JSONL row per dict; return the log path."""
    path = Path(tmp) / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_refresh_from_pairs_skips_too_small():
    cal = Calibrator(min_samples=10)
    rep = cal.refresh_from_pairs([(0.4, 0), (0.6, 1)], min_samples=10)
    assert rep["status"] == "skipped"
    assert cal.samples == 0


def test_refresh_from_pairs_refits_and_keeps_guard():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.json"
        # First fit: honest curve.
        cal = Calibrator(path=path, min_samples=10)
        probs = [0.1 + 0.05 * i for i in range(18)]
        ys = [1 if p > 0.5 else 0 for p in probs]
        pairs = list(zip(probs, ys))
        rep = cal.refresh_from_pairs(pairs, min_samples=10)
        assert rep["status"] == "refit"
        first_a, first_b = cal.a, cal.b
        first_ece = cal.ece
        # Regression guard: a worse refit must be rejected (status
        # "kept") -- conflicting outcomes at the same high probability
        # cannot be fit as well as the honest curve, so the refit ECE is
        # measurably worse and the old params are restored.
        bad_pairs = [(0.9, 1), (0.9, 0)] * 6
        rep2 = cal.refresh_from_pairs(bad_pairs, min_samples=10)
        assert rep2["status"] == "kept"
        assert cal.a == first_a and cal.b == first_b
        assert cal.ece == first_ece
        assert (Path(tmp) / "cal.json.bak").exists()


def test_refresh_from_log_delegates_to_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i in range(12):
            p = 0.4 + 0.04 * i
            rows.append({
                "event": "snapshot",
                "match_id": f"LaLiga||Team{i}||Team{i+1}||2026-08-{10+i%5:02d}",
                "ts": f"2026-08-0{i+1}T10:00:00Z",
                "prob_1x2": {"home": p, "draw": 0.3, "away": 0.3},
            })
            rows.append({
                "event": "settle",
                "match_id": f"LaLiga||Team{i}||Team{i+1}||2026-08-{10+i%5:02d}",
                "outcome": "home" if p > 0.5 else "draw",
            })
        log = _write_log(tmp, rows)
        cal = Calibrator(min_samples=10)
        rep = cal.refresh_from_log(log, min_samples=10)
        assert rep["status"] == "refit"
        assert rep["pairs"] >= 12


def test_refresh_leagues_from_log_groups_and_fits():
    """D2: per-league fits (incl. a dynamic ``dyn:`` key) are written to
    calibration_<slug>.json with the same discipline as the global refresh;
    leagues below the floor do not appear."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        # 12 settled snapshots in a dynamic league (Coppa Italia), each on
        # its OWN canonical match (distinct date) so dedupe keeps all 12.
        for i in range(12):
            p = 0.4 + 0.04 * i
            mid = f"dyn:coppa-italia||Pisa||Empoli||2026-08-{10+i:02d}"
            rows.append({
                "event": "snapshot", "match_id": mid,
                "ts": f"2026-08-0{i+1}T10:00:00Z",
                "prob_1x2": {"home": p, "draw": 0.3, "away": 0.3},
            })
            rows.append({"event": "settle", "match_id": mid,
                         "outcome": "home" if p > 0.5 else "draw"})
        # 3 settled snapshots in another league -> below the 10-pair floor.
        for i in range(3):
            mid = f"MLS||A{i}||B{i}||2026-08-1{i}"
            rows.append({
                "event": "snapshot", "match_id": mid,
                "ts": f"2026-08-2{i}T10:00:00Z",
                "prob_1x2": {"home": 0.5, "draw": 0.3, "away": 0.2},
            })
            rows.append({"event": "settle", "match_id": mid, "outcome": "home"})
        log = _write_log(tmp, rows)
        from agents.football.calibration import refresh_leagues_from_log

        reports = refresh_leagues_from_log(log, cal_dir=tmp, min_samples=10)
        # Only the dyn league cleared the floor.
        assert set(reports) == {"dyn:coppa-italia"}
        assert reports["dyn:coppa-italia"]["status"] == "refit"
        # The per-league file exists at the SAME path league_calibrator reads.
        assert (Path(tmp) / "calibration_dyn-coppa-italia.json").exists()


def test_league_calibrator_reads_dyn_fit():
    """D2: once a dynamic league has a calibration_<slug>.json file with
    enough samples, league_calibrator returns a real calibrator -- the dyn
    league leaves the uncalibrated_league cap."""
    from agents.football.calibration import league_calibrator

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "cache/football"
        base.mkdir(parents=True)
        cal = Calibrator(path=base / "calibration_dyn-coppa-italia.json", min_samples=20)
        probs = [0.1 + 0.02 * i for i in range(40)]  # 40 samples >= min 20
        cal.fit(probs, [1 if p > 0.5 else 0 for p in probs])
        cfg = {
            "models": {
                "calibration": {
                    "file": "cache/football/calibration.json",
                    "league_min_samples": 20,
                }
            }
        }
        found = league_calibrator("dyn:coppa-italia", cfg, root=tmp)
        assert found is not None
        assert found.samples >= 20
        assert found.apply(0.5) != 0.5  # real fit, not identity


def test_completeness_level_mapping():
    """PHASE 3 cap rules: 90+ -> HIGH, 70-89 -> MED/HIGH, 50-69 -> max MEDIUM,
    below 50 -> LOW only."""
    assert completeness_level(0.95) == ("HIGH", 1.0)
    assert completeness_level(0.90) == ("HIGH", 1.0)
    assert completeness_level(0.75) == ("MEDIUM/HIGH", 1.0)
    assert completeness_level(0.60) == ("MEDIUM", 0.69)
    assert completeness_level(0.50) == ("MEDIUM", 0.69)
    assert completeness_level(0.40) == ("LOW", 0.49)


def test_confidence_capped_by_completeness():
    """Even when all other components score highly, low completeness must cap
    confidence so the HIGH label cannot be reached with sparse data."""
    # Only odds present -> completeness 0.25 (spec weight) -> LOW cap 0.49
    ctx = MatchContext(
        league="EPL", home="A", away="B",
        consensus_odds={"home": 2.0, "draw": 3.4, "away": 3.8},
    )
    scorer = SignalScorer()
    comps = scorer.components(
        ctx=ctx,
        ensemble_models=["elo", "poisson"],
        model_vs_market=1.0,
        model_vs_model=1.0,
        calibration_quality=1.0,
        market_edge={"home": 8.0},
    )
    assert comps["data_completeness_level"] == "LOW"
    assert comps["confidence"] <= 0.49  # capped despite perfect agreement

    # Form + odds + attack/defense -> completeness 0.65 (spec weights) ->
    # maximum MEDIUM (cap 0.69): the 0.60 Section-5 floor is load-bearing.
    ctx2 = MatchContext(
        league="EPL", home="A", away="B",
        home_form="W-W-W", away_form="L-L-L",
        home_gf_avg=2.0, home_ga_avg=1.0, away_gf_avg=1.0, away_ga_avg=2.0,
        consensus_odds={"home": 2.0, "draw": 3.4, "away": 3.8},
    )
    comps2 = scorer.components(
        ctx=ctx2,
        ensemble_models=["elo", "poisson"],
        model_vs_market=1.0,
        model_vs_model=1.0,
        calibration_quality=1.0,
        market_edge={"home": 8.0},
    )
    assert comps2["data_completeness_level"] == "MEDIUM"
    assert comps2["confidence"] > 0.49
    assert comps2["confidence"] <= 0.69  # capped at MEDIUM ceiling


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
