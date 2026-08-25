"""One-off ML feature diagnostics (Fase D).

Runs on the cached fixture history (the same data ml_train.py uses) and
reports which features carry signal before a retrain:

  - boruta:      simplified Boruta (port of ProphitBet src/analysis/boruta_.py)
                 -- RandomForest importance vs a permuted "shadow" copy;
                 a feature that beats its shadow in most iterations carries
                 real signal.
  - correlation: |correlation| of every feature with the 1X2 target, plus
                 the most collinear feature pairs (drop one of each).
  - variance:    per-feature variance (near-zero = no information).
  - coefficients: standardized LogisticRegression coefficients per class.

All diagnostics are offline and read-only -- nothing is trained or saved.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ml_train import _load_elo_config, build_targets, load_frames


def _frame_for(league: str | None, window: int, gd_margin: int):
    leagues = [league] if league else None
    df, used = load_frames(leagues, window, gd_margin)
    from . import ml_features as mf

    features = [c for c in mf.available_columns(df) if c not in mf.XG_FEATURE_COLUMNS]
    df = df.dropna(subset=features).reset_index(drop=True)
    return df, features, used


def _variance_report(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    var = df[features].var()
    return {
        "per_feature": {
            c: {"variance": round(float(var[c]), 4), "std": round(float(np.sqrt(var[c])), 4)}
            for c in features
        },
        "low_variance_floor": {
            c: round(float(var[c]), 4)
            for c in features if float(var[c]) < 1e-3
        },
    }


def _correlation_report(
    df: pd.DataFrame, features: list[str], target: str
) -> dict[str, Any]:
    y = build_targets(df, target).astype(float)
    with_target = df[features].corrwith(y).abs().sort_values(ascending=False)
    mat = df[features].corr()
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            v = mat.loc[a, b]
            if not np.isnan(v) and abs(v) >= 0.6:
                pairs.append({"a": a, "b": b, "corr": round(float(v), 3)})
    pairs.sort(key=lambda p: abs(p["corr"]), reverse=True)
    return {
        "target_correlation": {
            c: round(float(v), 4) for c, v in with_target.items()
        },
        "collinear_pairs_ge_0.6": pairs[:20],
    }


def _coefficient_report(
    df: pd.DataFrame, features: list[str], target: str
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(df[features].to_numpy(dtype=np.float32))
    y = build_targets(df, target).to_numpy(dtype=np.int32)
    clf = LogisticRegression(max_iter=3000, random_state=0).fit(x, y)
    names = ["home", "draw", "away"][: len(clf.classes_)]
    return {
        "classes": names,
        "intercepts": [round(float(v), 4) for v in clf.intercept_],
        "coefficients": {
            c: {
                name: round(float(coef), 4)
                for name, coef in zip(names, clf.coef_[:, i])
            }
            for i, c in enumerate(features)
        },
    }


def _boruta_report(
    df: pd.DataFrame, features: list[str], target: str, *, n_iter: int = 25
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    x = df[features].to_numpy(dtype=np.float32)
    y = build_targets(df, target).to_numpy(dtype=np.int32)
    rng = np.random.default_rng(0)
    importances: dict[str, list[float]] = {c: [] for c in features}
    hits: dict[str, int] = {c: 0 for c in features}
    for _ in range(n_iter):
        shadow = np.column_stack(
            [rng.permutation(x[:, i]) for i in range(x.shape[1])]
        )
        x_aug = np.hstack([x, shadow])
        rf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=0, class_weight="balanced"
        )
        rf.fit(x_aug, y)
        thr = float(np.max(rf.feature_importances_[len(features):]))
        for i, c in enumerate(features):
            imp = float(rf.feature_importances_[i])
            importances[c].append(imp)
            if imp > thr:
                hits[c] += 1
    report: dict[str, Any] = {}
    for c in features:
        mean_imp = float(np.mean(importances[c]))
        report[c] = {
            "mean_importance": round(mean_imp, 5),
            "hits_above_shadow": hits[c],
            "hit_rate": round(hits[c] / n_iter, 3),
            "verdict": "keep" if hits[c] >= max(2, n_iter // 2) else "drop",
        }
    return {"n_iterations": n_iter, "features": report}


def run_analysis(
    *,
    league: str | None,
    metric: str,
    window: int = 5,
    gd_margin: int = 2,
) -> dict[str, Any]:
    df, features, used = _frame_for(league, window, gd_margin)
    if df.empty or len(df) < 100:
        return {
            "error": f"too few rows ({len(df)}) after NaN drop for {league or 'all'}",
            "leagues": used,
        }
    target = "result"
    out: dict[str, Any] = {
        "league": league or "all",
        "leagues": used,
        "n_rows": int(len(df)),
        "features": features,
    }
    if metric in ("all", "variance"):
        out["variance"] = _variance_report(df, features)
    if metric in ("all", "correlation"):
        out["correlation"] = _correlation_report(df, features, target)
    if metric in ("all", "coefficients"):
        out["coefficients"] = _coefficient_report(df, features, target)
    if metric in ("all", "boruta"):
        out["boruta"] = _boruta_report(df, features, target)
    return out
