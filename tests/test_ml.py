"""ML layer tests: feature port, walk-forward training, live inference, and
the decision-engine ml_agreement component (Fase ML integration)."""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from agents.football import ml_features as mf
from agents.football.ml_train import (
    _chunk_boundaries,
    build_targets,
    walk_forward_eval,
)
from agents.football import decision
from agents.football.decision import Candidate, decide, score_candidate

ROOT = Path(__file__).resolve().parent.parent


def _synthetic_fixtures(teams=("A", "B", "C"), n_seasons=2, per_season=20):
    """Deterministic league history: A always beats B/C at home, draws away."""
    fx = []
    start = date(2022, 8, 6)
    for season in range(n_seasons):
        season_label = f"{2022 + season}-{2023 + season}"
        for i in range(per_season):
            d = start + timedelta(days=i + season * 300)
            home = teams[i % 3]
            away = teams[(i + 1) % 3]
            if home == teams[0]:
                hg, ag = 2, 0
            elif away == teams[0]:
                hg, ag = 0, 1
            else:
                hg, ag = 1, 1
            fx.append({
                "date": d.isoformat(), "season": season_label,
                "home": home, "away": away, "home_goals": hg, "away_goals": ag,
                "league": "TST",
            })
    return fx


class TestFeaturePort:
    def test_upcoming_row_survives(self):
        hist = _synthetic_fixtures(per_season=24)
        upcoming = {
            "date": "2024-01-15", "season": "2023-2024",
            "home": "A", "away": "B", "home_goals": None, "away_goals": None,
            "league": "TST",
        }
        df = mf.build_feature_frame(hist + [upcoming], window=3, gd_margin=2)
        assert len(df) == len(hist) + 1
        row = df.iloc[-1]
        assert row["Home"] == "A" and row["Away"] == "B"
        # A dominates at home: HW should be >= 1 and features finite.
        assert float(row["HW"]) >= 1.0

    def test_thin_team_nan(self):
        fx = _synthetic_fixtures(per_season=12)
        df = mf.build_feature_frame(fx, window=5, gd_margin=2)
        # First 5 home matches of each team are NaN by design (honest floor).
        first_rows = df[df["Date"] <= "2022-10-01"]
        assert first_rows["HW"].isna().any()

    def test_no_leakage_shift(self):
        fx = _synthetic_fixtures(per_season=12)
        df = mf.build_feature_frame(fx, window=3, gd_margin=2)
        # A home match's HW must never include that match itself: the row's
        # own result cannot be counted (shift(1)). Verify by checking that
        # HGF on A's 4th home game equals the sum of the 3 prior home scores.
        a_home = df[(df["Home"] == "A") & (df["Season"] == "2022-2023")]
        assert len(a_home) >= 4
        # 4th home row: HGF = sum of the 3 previous A-home HG values.
        prior = a_home.iloc[:3]["HG"].sum()
        assert float(a_home.iloc[3]["HGF"]) == prior

    def test_cards_and_fouls_rolling_columns(self):
        """Fouls/yellow-card rolling features compute only when the raw
        per-match columns exist, with the same leakage-safe window."""
        fx = _synthetic_fixtures(per_season=12)
        for i, m in enumerate(fx):
            m["HF"] = 10 + (i % 5)
            m["AF"] = 12 - (i % 4)
            m["HY"] = 1 + (i % 3)
            m["AY"] = 2 + (i % 2)
        df = mf.build_feature_frame(fx, window=3, gd_margin=2)
        for col in ("HFCF", "AFCF", "HYCF", "AYCF"):
            assert col in df.columns
            assert col in mf.available_columns(df)
        # snake_case aliases also feed the same columns
        fx2 = _synthetic_fixtures(per_season=6)
        for m in fx2:
            m["home_fouls"] = 9
            m["away_fouls"] = 8
            m["home_yellow_cards"] = 1
            m["away_yellow_cards"] = 2
        df2 = mf.build_feature_frame(fx2, window=3, gd_margin=2)
        for col in ("HFCF", "AFCF", "HYCF", "AYCF"):
            assert col in df2.columns
        # absent raw columns -> columns simply not emitted
        bare = mf.build_feature_frame(_synthetic_fixtures(per_season=6), window=3, gd_margin=2)
        assert "HFCF" not in mf.available_columns(bare)

    def test_targets(self):
        df = mf.build_feature_frame(_synthetic_fixtures(), window=3, gd_margin=2)
        y = build_targets(df, "result").to_numpy()
        assert set(y) <= {0, 1, 2}
        yo = build_targets(df, "over-under").to_numpy()
        assert set(yo) <= {0, 1}


class TestTrain:
    def test_chunk_boundaries(self):
        assert _chunk_boundaries(100, 4) == [0, 25, 50, 75, 100]
        assert _chunk_boundaries(10, 1) == [0, 10]

    def test_walk_forward_runs(self):
        df = mf.build_feature_frame(_synthetic_fixtures(), window=3, gd_margin=2)
        features = mf.available_columns(df)
        df = df.dropna(subset=features).reset_index(drop=True)
        assert len(df) >= 15
        out = walk_forward_eval(df, "lr", "result", folds=3, sampler_name="none")
        assert out["aggregate"].get("logloss") is not None
        assert out["aggregate"]["n"] > 0


@pytest.fixture(scope="module")
def predictor():
    models_dir = ROOT / "cache" / "football" / "models"
    if not (models_dir / "result" / "pipeline.pkl").exists():
        pytest.skip("ML artifacts not trained; run `runner train-model` first")
    from agents.football.ml_predict import MlPredictor

    return MlPredictor(models_dir, window=5, gd_margin=2)


class TestPredict:
    def test_thin_data_none(self, predictor):
        # First matchday of a season: no team has 5 home matches -> honest None.
        assert predictor.predict_1x2("EPL", "Arsenal", "Chelsea", "2024-08-17") is None

    def test_midseason_probs(self, predictor):
        r = predictor.predict_1x2("EPL", "Arsenal", "Everton", "2026-03-14")
        assert r is not None
        total = r["home"] + r["draw"] + r["away"]
        assert math.isclose(total, 1.0, abs_tol=1e-2)
        assert all(0.0 <= r[k] <= 1.0 for k in ("home", "draw", "away"))


class TestDecisionMlAgreement:
    def _candidate(self, model_prob=0.6, edge_pp=4.0, ev=0.05):
        return Candidate(
            market="1X2", selection="home", model_prob=model_prob,
            market_odds=2.5, implied_prob=0.4,
            edge_pp=edge_pp, ev=ev, independent=True,
        )

    def test_component_present_and_normalized(self):
        c = self._candidate()
        weights = {**decision.DEFAULT_WEIGHTS, "ml_agreement": 0.10}
        comps = score_candidate(
            c, [c], calibration_quality=0.8, calibration_samples=500,
            model_agreement=0.9, completeness=0.8, bookmakers_count=8,
            historical_reliability=0.6, weights=weights,
            edge_warning_pp=10.0, edge_extreme_pp=20.0, min_bookmakers=3,
            ml_agreement=0.9,
        )
        assert "ml_agreement" in comps
        assert 0.0 <= comps["score"] <= 1.0

    def test_absent_keeps_legacy_components(self):
        c = self._candidate()
        comps = score_candidate(
            c, [c], calibration_quality=0.8, calibration_samples=500,
            model_agreement=0.9, completeness=0.8, bookmakers_count=8,
            historical_reliability=0.6, weights=dict(decision.DEFAULT_WEIGHTS),
            edge_warning_pp=10.0, edge_extreme_pp=20.0, min_bookmakers=3,
        )
        assert "ml_agreement" not in comps

    def test_decide_accepts_ml_agreement(self):
        c = self._candidate()
        d = decide(
            [c], model_agreement=0.9, calibration_quality=0.8,
            calibration_samples=500, completeness=0.8, bookmakers_count=8,
            historical_reliability=0.6, ml_agreement=0.9,
        )
        assert d["decision_type"] in ("STRONG", "GOOD", "LEAN", "NO CLEAR DECISION")
