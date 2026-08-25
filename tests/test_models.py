"""Tests for models.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.context import MatchContext
from agents.football.elo import EloModel
from agents.football.models import (
    Ensemble,
    PoissonModel,
    poisson_matrix,
    probs_from_matrix,
    run_prediction_engine,
)


def _ctx(**kw) -> MatchContext:
    defaults = dict(
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        home_form="W-W-W-W-W",
        away_form="L-L-L-L-L",
        home_gf_avg=2.2,
        home_ga_avg=0.7,
        away_gf_avg=1.1,
        away_ga_avg=1.6,
        form_samples=5,
    )
    defaults.update(kw)
    return MatchContext(**defaults)


def test_poisson_matrix_sums_to_one():
    m = poisson_matrix(1.5, 1.2, rho=-0.1)
    assert math.isclose(sum(sum(r) for r in m), 1.0, abs_tol=1e-6)


def test_poisson_matrix_plain_rho_zero():
    m = poisson_matrix(1.5, 1.2, rho=0.0)
    assert math.isclose(sum(sum(r) for r in m), 1.0, abs_tol=1e-6)


def test_probs_from_matrix_1x2_sums_to_one():
    m = poisson_matrix(1.5, 1.2, rho=-0.1)
    p1x2, o15, o25, o35, btts = probs_from_matrix(m)
    assert math.isclose(sum(p1x2.values()), 1.0, abs_tol=1e-6)
    assert 0 < p1x2["home"] < 1
    assert 0 < p1x2["draw"] < 1
    assert 0 < p1x2["away"] < 1
    assert 0 < o25 < 1
    assert 0 < btts < 1


def test_poisson_model_strong_home():
    model = PoissonModel()
    out = model.predict(_ctx())
    assert out is not None
    assert out["lambda_home"] > out["lambda_away"]
    assert out["1x2"]["home"] > out["1x2"]["away"]
    assert math.isclose(sum(out["1x2"].values()), 1.0, abs_tol=1e-6)


def test_poisson_model_missing_features_returns_none():
    model = PoissonModel()
    ctx = MatchContext(league="EPL", home="A", away="B")
    assert model.predict(ctx) is None


def test_poisson_model_xg_blend():
    model = PoissonModel()
    ctx = _ctx(home_xg_for=2.6, home_xg_against=0.8, away_xg_for=0.9, away_xg_against=2.0)
    out = model.predict(ctx)
    assert out is not None
    assert out["lambda_source"] == "features+xg"


def test_ensemble_blend_sums_to_one():
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1700.0, "Chelsea": 1300.0})
    poisson = PoissonModel()
    ens = Ensemble(elo_weight=0.5, poisson_weight=0.5)
    out = ens.predict(_ctx(), elo, poisson)
    assert out is not None
    assert set(out["models"]) == {"elo", "poisson"}
    assert math.isclose(sum(out["1x2"].values()), 1.0, abs_tol=1e-6)


def test_ensemble_renormalizes_weights():
    elo = EloModel()
    poisson = PoissonModel()
    ens = Ensemble(elo_weight=0.6, poisson_weight=0.4)
    out = ens.predict(_ctx(), elo, poisson)
    assert math.isclose(sum(out["weights"].values()), 1.0, abs_tol=1e-6)


def test_engine_produces_prediction_result():
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(
        consensus_odds={"home": 1.85, "draw": 3.6, "away": 4.2},
        h2h={"wins": 3, "draws": 1, "losses": 1},
    )
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1680.0, "Chelsea": 1420.0})
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    assert math.isclose(sum(mp["1x2"].values()), 1.0, abs_tol=1e-3)
    assert result.confidence >= 0.0
    assert 0 <= result.signal_strength <= 100
    assert "home" in result.market_edge
    assert result.input_hash
    assert result.model_version


def test_engine_without_odds_keeps_edge_empty():
    from agents.football.calibration import Calibrator, SignalScorer

    result = run_prediction_engine(
        _ctx(),
        elo=EloModel(),
        poisson=PoissonModel(),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    assert result.market_edge == {}


def _ctx_recent(**kw) -> MatchContext:
    """Context with raw recent scorelines instead of precomputed averages."""
    defaults = dict(
        league="EPL",
        home="Arsenal",
        away="Chelsea",
        home_recent_goals=[(2, 0), (2, 0), (3, 0)],  # old -> new, strong attack
        away_recent_goals=[(1, 1), (1, 0), (0, 1)],
    )
    defaults.update(kw)
    return MatchContext(**defaults)


def test_poisson_decay_equal_weight_when_xi_one():
    """xi=1.0 must reproduce the plain equal-weight averages exactly."""
    recent = _ctx_recent()
    averaged = _ctx(
        home_gf_avg=7 / 3, home_ga_avg=0.0,
        away_gf_avg=2 / 3, away_ga_avg=2 / 3,
        form_samples=3,
    )
    m = PoissonModel(time_decay_xi=1.0)
    out_recent = m.predict(recent)
    out_avg = m.predict(averaged)
    assert out_recent is not None and out_avg is not None
    assert math.isclose(out_recent["lambda_home"], out_avg["lambda_home"], abs_tol=1e-9)
    assert math.isclose(out_recent["lambda_away"], out_avg["lambda_away"], abs_tol=1e-9)


def test_poisson_decay_prefers_recent_form():
    """With recent matches stronger than the average, xi<1 must lift the
    decayed average (Dixon-Coles time decay)."""
    m_recent = PoissonModel(time_decay_xi=0.5)  # strong decay
    m_flat = PoissonModel(time_decay_xi=1.0)  # no decay
    lh_decay = m_recent.predict(_ctx_recent())["lambda_home"]
    lh_flat = m_flat.predict(_ctx_recent())["lambda_home"]
    assert lh_decay > lh_flat


def test_poisson_uses_recent_when_gf_avg_missing():
    ctx = _ctx_recent()
    assert ctx.has_attack_defense  # via raw scorelines only
    out = PoissonModel().predict(ctx)
    assert out is not None


def test_poisson_xg_weight_zero_equals_no_xg():
    base = _ctx()  # no xG
    ctx = _ctx(home_xg_for=2.6, home_xg_against=0.8, away_xg_for=0.9, away_xg_against=2.0)
    m = PoissonModel(xg_weight=0.0)
    out_base = m.predict(base)
    out_xg0 = m.predict(ctx)
    assert out_base is not None and out_xg0 is not None
    assert math.isclose(out_base["lambda_home"], out_xg0["lambda_home"], abs_tol=1e-9)
    assert math.isclose(out_base["lambda_away"], out_xg0["lambda_away"], abs_tol=1e-9)


def test_poisson_xg_weight_one_is_pure_xg():
    ctx = _ctx(
        home_recent_goals=None, away_recent_goals=None,
        home_gf_avg=1.0, home_ga_avg=1.0,
        away_gf_avg=1.0, away_ga_avg=1.0,
        form_samples=5,
        home_xg_for=2.6, home_xg_against=0.8,
        away_xg_for=0.9, away_xg_against=2.0,
    )
    m = PoissonModel(xg_weight=1.0)
    out = m.predict(ctx)
    assert out is not None
    # xh = (home_xg_for + away_xg_against)/2 = 2.3 ; xa = (away_xg_for + home_xg_against)/2 = 0.85
    assert math.isclose(out["lambda_home"], 2.3, abs_tol=1e-9)
    assert math.isclose(out["lambda_away"], 0.85, abs_tol=1e-9)


def test_poisson_lambda_samples_exposed():
    """predict() must expose the effective form-window depth (min over both
    sides) so the engine can blend toward Elo when the window is thin."""
    out = PoissonModel().predict(_ctx())  # form_samples=5
    assert out is not None
    assert out["lambda_samples"] == 5
    thin = _ctx(form_samples=1, home_gf_avg=4.0, home_ga_avg=0.0,
                away_gf_avg=2.0, away_ga_avg=2.0)
    out_thin = PoissonModel().predict(thin)
    assert out_thin is not None
    assert out_thin["lambda_samples"] == 1


def test_engine_thin_sample_uses_elo_lambda():
    """Option 1: below min_samples the feature λ is replaced by the Elo λ
    (a 1-match window is noise, not signal -- the Excelsior/PSV case)."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(
        form_samples=1,
        home_gf_avg=4.0, home_ga_avg=0.0,   # Excelsior: 1x 4-0 win
        away_gf_avg=2.0, away_ga_avg=2.0,   # PSV: 1x 2-2 draw
    )
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1519.5, "Chelsea": 1782.3})  # away clearly stronger
    elo._rebuild_indexes()  # manual ratings need the lookup index rebuilt
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(
            min_samples=2,
            # test menarget lapisan SELEKSI lambda -> anchor/kalibrasi v3 dimatikan
            elo_anchor={"enabled": False},
            market_total_calibration={"enabled": False},
        ),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    assert mp["lambda_source"] == "elo"
    lh_e, la_e = elo.expected_lambdas("Arsenal", "Chelsea")
    assert math.isclose(mp["lambda_home"], lh_e, abs_tol=1e-3)
    assert math.isclose(mp["lambda_away"], la_e, abs_tol=1e-3)
    # The inverted feature λ (home>away) must be gone: PSV now favored.
    assert mp["lambda_away"] > mp["lambda_home"]


def test_engine_thin_sample_blends_toward_elo():
    """Option 2: between min_samples and shrinkage_samples the λ is a linear
    blend of the feature and Elo λ (ramping to full feature trust)."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(
        form_samples=3,   # min_samples=2, shrinkage=5 -> t=(3-2)/(5-2)=1/3
        home_gf_avg=4.0, home_ga_avg=0.0,
        away_gf_avg=2.0, away_ga_avg=2.0,
    )
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1519.5, "Chelsea": 1782.3})
    elo._rebuild_indexes()
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(
            min_samples=2,
            # test menarget lapisan SELEKSI lambda -> anchor/kalibrasi v3 dimatikan
            elo_anchor={"enabled": False},
            market_total_calibration={"enabled": False},
        ),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    assert mp["lambda_source"] == "features+elo"
    lh_e, la_e = elo.expected_lambdas("Arsenal", "Chelsea")
    pm = PoissonModel(min_samples=2).predict(ctx)
    lh_p, la_p = pm["lambda_home"], pm["lambda_away"]
    t = 1 / 3
    assert math.isclose(mp["lambda_home"], t * lh_p + (1 - t) * lh_e, abs_tol=1e-3)
    assert math.isclose(mp["lambda_away"], t * la_p + (1 - t) * la_e, abs_tol=1e-3)


def test_engine_full_sample_keeps_feature_lambda():
    """At/above shrinkage_samples the feature λ stands as-is (unchanged
    behaviour for normal windows)."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx()  # form_samples=5 == shrinkage
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1680.0, "Chelsea": 1420.0})
    elo._rebuild_indexes()
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(
            min_samples=2,
            # test menarget lapisan SELEKSI lambda -> anchor/kalibrasi v3 dimatikan
            elo_anchor={"enabled": False},
            market_total_calibration={"enabled": False},
        ),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    pm = PoissonModel(min_samples=2).predict(ctx)
    assert mp["lambda_source"] == "features"
    assert math.isclose(mp["lambda_home"], pm["lambda_home"], abs_tol=1e-3)
    assert math.isclose(mp["lambda_away"], pm["lambda_away"], abs_tol=1e-3)


def test_engine_full_sample_keeps_xg_provenance_label():
    """2026-08-17 fix: the λ selection layer must NOT flatten the
    ``features+xg`` label back to ``features`` -- the λ is already
    xG-blended and the label must stay honest (provenance audit)."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(
        home_xg_for=2.6, home_xg_against=0.8,
        away_xg_for=0.9, away_xg_against=2.0,
    )
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1680.0, "Chelsea": 1420.0})
    elo._rebuild_indexes()
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(
            min_samples=2,
            # test menarget lapisan SELEKSI lambda -> anchor/kalibrasi v3 dimatikan
            elo_anchor={"enabled": False},
            market_total_calibration={"enabled": False},
        ),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    assert mp["lambda_source"] == "features+xg"
    # The λ must equal the xG-blended feature λ, not the raw feature λ.
    pm = PoissonModel(min_samples=2).predict(ctx)
    assert math.isclose(mp["lambda_home"], pm["lambda_home"], abs_tol=1e-3)


def test_engine_thin_sample_with_xg_blend_label():
    """Blend band + xG: the blend label carries the xG provenance too
    (features+xg+elo, not a bare features+elo that hides the blend)."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(
        form_samples=3,
        home_xg_for=2.6, home_xg_against=0.8,
        away_xg_for=0.9, away_xg_against=2.0,
    )
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1519.5, "Chelsea": 1782.3})
    elo._rebuild_indexes()
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(
            min_samples=2,
            # test menarget lapisan SELEKSI lambda -> anchor/kalibrasi v3 dimatikan
            elo_anchor={"enabled": False},
            market_total_calibration={"enabled": False},
        ),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    assert mp["lambda_source"] == "features+xg+elo"


def test_engine_min_samples_zero_disables_gate():
    """min_samples=0 must keep the pre-fix behaviour (pure feature λ even
    with a 1-match window)."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(
        form_samples=1,
        home_gf_avg=4.0, home_ga_avg=0.0,
        away_gf_avg=2.0, away_ga_avg=2.0,
    )
    elo = EloModel()
    elo.ratings.update({"Arsenal": 1519.5, "Chelsea": 1782.3})
    elo._rebuild_indexes()
    result = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=PoissonModel(
            min_samples=0,
            elo_anchor={"enabled": False},
            market_total_calibration={"enabled": False},
        ),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    assert result is not None
    mp = result.model_probs
    assert mp["lambda_source"] == "features"
    pm = PoissonModel(min_samples=0).predict(ctx)
    assert math.isclose(mp["lambda_home"], pm["lambda_home"], abs_tol=1e-3)


def test_engine_edge_uses_normalized_implied():
    """With a model close to the market, edges should be small, not ~0 by
    construction, but computed on a margin-free basis."""
    from agents.football.calibration import Calibrator, SignalScorer

    ctx = _ctx(consensus_odds={"home": 1.85, "draw": 3.6, "away": 4.2})
    result = run_prediction_engine(
        ctx,
        elo=EloModel(),
        poisson=PoissonModel(),
        ensemble=Ensemble(),
        calibrator=Calibrator(),
        scorer=SignalScorer(),
    )
    total_margin = (1 / 1.85 + 1 / 3.6 + 1 / 4.2) - 1.0
    assert total_margin > 0.01
    # model probs are independent of the odds -> edges not forced to zero
    assert any(abs(v) > 0.0 for v in result.market_edge.values())


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
