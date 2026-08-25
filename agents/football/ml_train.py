"""Offline ML training + walk-forward evaluation for 1X2 and Over/Under 2.5.

Feature engineering is the ProphitBet port (ml_features.py). The training
protocol is the bot's own honesty standard: chronological walk-forward folds
(train on the past, eval on the future) -- mirroring backtest.py so the ML
probabilities are validated the same way the Elo+Poisson ensemble is.

Evaluation metrics go beyond ProphitBet's accuracy/F1/Precision/Recall by
adding logloss and Brier (the bot's existing backtest compares those).

Artifacts are stored under cache/football/models/<target>/:
  pipeline.pkl   sklearn Pipeline(scaler -> CalibratedClassifierCV(model))
  features.json  feature schema + training context (must match at predict)
  metrics.json   per-fold + aggregate walk-forward metrics, baseline
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import ml_features as mf
from .backtest import load_fixtures_from_json

ROOT = Path(__file__).resolve().parent.parent.parent

TARGET_RESULT = "result"
TARGET_OVER_UNDER = "over-under"
SUPPORTED_TARGETS = (TARGET_RESULT, TARGET_OVER_UNDER)

MODEL_LR = "lr"
MODEL_RF = "rf"
MODEL_XGB = "xgb"
SUPPORTED_MODELS = (MODEL_LR, MODEL_RF, MODEL_XGB, "auto")

RESULT_LABELS = (0, 1, 2)  # H / D / A


def _outcome_index(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def build_targets(df: pd.DataFrame, target: str) -> pd.Series:
    if target == TARGET_RESULT:
        return df["Result"].map({"H": 0, "D": 1, "A": 2}).astype(int)
    if target == TARGET_OVER_UNDER:
        return (df["HG"] + df["AG"]).ge(2.5).astype(int)
    raise ValueError(f"unsupported target: {target}")


def discover_fixture_files() -> list[Path]:
    """All fixture JSONs under cache/football (+ backtest subdir).

    ``*_xg.json`` caches are excluded from the default model: only the EPL
    cache carries pre-match xG, so a pooled multi-league model cannot rely on
    it and the train/serve feature schema must stay league-agnostic.
    """
    roots = [ROOT / "cache" / "football", ROOT / "cache" / "football" / "backtest"]
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.json")):
            if p.name in (
                "elo.json", "calibration.json", "calibration.json.bak",
                "predictions.jsonl", "multileague_fixtures.json",
            ):
                continue
            if p.name.endswith("_xg.json"):
                continue
            if "_fixtures_" in p.name or "multileague" in p.name:
                out.append(p)
    return out


def load_frames(
    leagues: list[str] | None,
    window: int,
    gd_margin: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Load + feature-engineer every cached league into one frame.

    Returns (feature_frame, leagues_used). Rows whose league key is not in
    ``leagues`` (when provided) are skipped before feature engineering.
    """
    frames: list[pd.DataFrame] = []
    used: list[str] = []
    for path in discover_fixture_files():
        try:
            fixtures = load_fixtures_from_json(path)
        except Exception:  # noqa: BLE001 -- one bad cache must not kill training
            continue
        if not fixtures:
            continue
        if leagues:
            fixtures = [
                f for f in fixtures
                if (f.get("league") or "") in leagues
            ]
            if not fixtures:
                continue
        frame = mf.build_feature_frame(fixtures, window=window, gd_margin=gd_margin)
        if frame.empty:
            continue
        frame["league"] = [
            f.get("league", "") for f in fixtures[: len(frame)]
        ]
        used.extend(sorted({f.get("league", "") for f in fixtures}))
        frames.append(frame)
    if not frames:
        raise RuntimeError("no fixture caches found; run `runner cache-odds` first")
    df = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    # Same match can exist in multiple caches (EPL base vs augmented xG file);
    # keep the first (richer) row per (Date, Home, Away, league).
    df = df.drop_duplicates(subset=["Date", "Home", "Away", "league"], keep="first")
    df = mf.add_elo_features(df, _load_elo_config())
    return df, sorted(set(used))


def _load_elo_config() -> dict[str, Any]:
    """Production Elo params (config/football.json -> models.elo), minus file."""
    try:
        cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
        return {
            k: v for k, v in cfg.get("models", {}).get("elo", {}).items()
            if k != "file"
        }
    except (OSError, json.JSONDecodeError):
        return {}


def build_estimator(name: str, params: dict[str, Any] | None = None):
    """Base (uncalibrated) estimator for ``name``, with optional hyperparams."""
    p = params or {}
    if name == MODEL_LR:
        return LogisticRegression(
            C=float(p.get("C", 1.0)), max_iter=3000,
            class_weight="balanced", random_state=0,
        )
    if name == MODEL_RF:
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            min_samples_leaf=int(p.get("min_samples_leaf", 2)),
            max_features=p.get("max_features", "sqrt"),
            class_weight="balanced_subsample", n_jobs=-1, random_state=0,
        )
    if name == MODEL_XGB:
        return XGBClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=int(p.get("max_depth", 4)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            random_state=0, tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
        )
    raise ValueError(f"unknown model: {name}")


def build_model(name: str, params: dict[str, Any] | None = None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            estimator=build_estimator(name, params), method="isotonic", cv=3,
        )),
    ])


def build_models() -> dict[str, Pipeline]:
    return {name: build_model(name) for name in (MODEL_LR, MODEL_RF, MODEL_XGB)}


def _chunk_boundaries(n_rows: int, folds: int) -> list[int]:
    """Chronological split points: fold i evals rows [splits[i], splits[i+1])."""
    if folds <= 1:
        return [0, n_rows]
    chunk = n_rows // folds
    return [min(i * chunk, n_rows) for i in range(folds)] + [n_rows]


def _eval_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y_pred = probs.argmax(axis=1)
    n = len(y_true)
    log_loss = 0.0
    for i, p in enumerate(probs):
        p_out = float(min(max(p[int(y_true[i])], 1e-9), 1.0 - 1e-9))
        log_loss += -math.log(p_out)
    brier = sum(
        np.sum((probs[i] - (np.arange(probs.shape[1]) == y_true[i])) ** 2)
        for i in range(n)
    )
    return {
        "n": int(n),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)), 4),
        "logloss": round(log_loss / n, 4),
        "brier": round(brier / n, 4),
        "hit_rate": round(float((y_pred == y_true).mean()), 4),
    }


def walk_forward_eval(
    df: pd.DataFrame,
    model_name: str,
    target: str,
    *,
    folds: int = 5,
    sampler_name: str = "smote",
    seed: int = 0,
) -> dict[str, Any]:
    """Chronological walk-forward: train on the past, eval on the next chunk."""
    features = mf.available_columns(df)
    x = df[features].to_numpy(dtype=np.float32)
    y = build_targets(df, target).to_numpy(dtype=np.int32)
    n_classes = int(len(np.unique(y)))
    splits = _chunk_boundaries(len(df), folds)
    per_fold: list[dict[str, Any]] = []
    for i in range(1, len(splits) - 1):
        ev_start, ev_end = splits[i], splits[i + 1]
        xtr, ytr = x[:ev_start], y[:ev_start]
        xev, yev = x[ev_start:ev_end], y[ev_start:ev_end]
        if len(np.unique(ytr)) < 2 or len(yev) == 0:
            continue
        pipe = build_model(model_name)
        try:
            xtr_f, ytr_f = _apply_sampler(xtr, ytr, sampler_name, seed)
            pipe.fit(xtr_f, ytr_f)
            probs = pipe.predict_proba(xev)
            if probs.shape[1] != n_classes:
                continue
            m = _eval_metrics(yev, probs)
            m["fold"] = i
            m["train_rows"] = int(len(xtr_f))
            m["eval_rows"] = int(len(yev))
            m["start_date"] = str(df["Date"].iloc[ev_start])[:10]
            m["end_date"] = str(df["Date"].iloc[ev_end - 1])[:10]
            per_fold.append(m)
        except Exception as exc:  # noqa: BLE001 -- a broken fold must not kill training
            per_fold.append({"fold": i, "error": f"{type(exc).__name__}: {exc}"})
    if not per_fold:
        raise RuntimeError("walk-forward produced no folds (data too small?)")
    good = [m for m in per_fold if "error" not in m]
    agg: dict[str, Any] = {}
    if good:
        agg = {
            "n": sum(m["n"] for m in good),
            "accuracy": round(float(np.mean([m["accuracy"] for m in good])), 4),
            "f1_macro": round(float(np.mean([m["f1_macro"] for m in good])), 4),
            "logloss": round(float(np.mean([m["logloss"] for m in good])), 4),
            "brier": round(float(np.mean([m["brier"] for m in good])), 4),
            "hit_rate": round(float(np.mean([m["hit_rate"] for m in good])), 4),
        }
    return {"model": model_name, "target": target, "aggregate": agg, "folds": per_fold}


def _apply_sampler(
    x: np.ndarray, y: np.ndarray, sampler_name: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if sampler_name == "none":
        return x, y
    counts = {v: int((y == v).sum()) for v in np.unique(y)}
    if min(counts.values()) < 2:
        return x, y
    if sampler_name == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=seed, k_neighbors=min(4, min(counts.values()) - 1))
    elif sampler_name == "nearmiss":
        from imblearn.under_sampling import NearMiss

        sampler = NearMiss(version=3, n_jobs=-1)
    else:
        raise ValueError(f"unknown sampler: {sampler_name}")
    try:
        return sampler.fit_resample(x, y)
    except Exception:  # noqa: BLE001 -- fall back to raw when resampling fails
        return x, y


def tune_model(
    df: pd.DataFrame,
    base_name: str,
    target: str,
    *,
    trials: int,
    folds: int,
    sampler_name: str,
) -> dict[str, Any]:
    """Optuna tuning: minimize mean eval logloss over the walk-forward folds."""
    import optuna

    features = mf.available_columns(df)
    x = df[features].to_numpy(dtype=np.float32)
    y = build_targets(df, target).to_numpy(dtype=np.int32)
    splits = _chunk_boundaries(len(df), folds)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        if base_name == MODEL_XGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif base_name == MODEL_RF:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            }
        elif base_name == MODEL_LR:
            params = {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
        else:
            raise ValueError(f"unsupported tuning base: {base_name}")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=build_estimator(base_name, params), method="isotonic", cv=3,
            )),
        ])
        losses: list[float] = []
        for i in range(1, len(splits) - 1):
            ev_start, ev_end = splits[i], splits[i + 1]
            if ev_end <= ev_start or ev_start == 0:
                continue
            try:
                xtr, ytr = _apply_sampler(x[:ev_start], y[:ev_start], sampler_name, 0)
                pipe.fit(xtr, ytr)
                probs = pipe.predict_proba(x[ev_start:ev_end])
                for j, p in enumerate(probs):
                    p_out = float(min(max(p[int(y[ev_start + j])], 1e-9), 1.0 - 1e-9))
                    losses.append(-math.log(p_out))
            except Exception:  # noqa: BLE001 -- trial fold failure = bad trial
                return -1e9
        return -float(np.mean(losses)) if losses else -1e9

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_neg_logloss": study.best_value}


def _baseline_metrics(df: pd.DataFrame, target: str) -> dict[str, Any]:
    y = build_targets(df, target).to_numpy()
    n = len(y)
    p = np.array([(y == c).mean() for c in np.unique(y)])
    majority = p.max()
    logloss = -np.mean(np.log(p[y]))
    return {
        "n": int(n),
        "class_rates": {int(c): round(float((y == c).mean()), 4) for c in np.unique(y)},
        "majority_accuracy": round(float(majority), 4),
        "base_rate_logloss": round(float(logloss), 4),
    }


def train_model(
    *,
    leagues: list[str] | None,
    model: str,
    target: str,
    folds: int,
    sampler_name: str,
    tune_trials: int,
    models_dir: Path,
    window: int,
    gd_margin: int,
) -> dict[str, Any]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target: {target}")
    df, used = load_frames(leagues, window, gd_margin)
    before = len(df)
    feature_cols = [
        c for c in mf.available_columns(df) if c not in mf.XG_FEATURE_COLUMNS
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(df)
    if len(df) < 200:
        raise RuntimeError(
            f"only {len(df)} rows after NaN drop ({dropped} dropped); "
            "need >= 200 -- widen the window (smaller window) or add seasons"
        )

    candidates = [m for m in SUPPORTED_MODELS if m != "auto"] if model == "auto" else [model]
    results = {}
    for name in candidates:
        results[name] = walk_forward_eval(
            df, name, target, folds=folds, sampler_name=sampler_name
        )

    best_name = min(
        (n for n in results if results[n]["aggregate"].get("logloss") is not None),
        key=lambda n: results[n]["aggregate"]["logloss"],
    )
    tuned = None
    if tune_trials > 0:
        tuned = tune_model(
            df, best_name, target, trials=tune_trials, folds=folds, sampler_name=sampler_name
        )

    # Final production fit on ALL rows (labeled honestly; metrics.json keeps
    # the walk-forward numbers, never an in-sample claim). Tuned params, when
    # present, are applied here -- the tuning result must reach the artifact.
    pipe = build_model(best_name, (tuned or {}).get("best_params"))
    x_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = build_targets(df, target).to_numpy(dtype=np.int32)
    x_all, y_all = _apply_sampler(x_all, y_all, sampler_name, seed=0)
    pipe.fit(x_all, y_all)

    out_dir = models_dir / target
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pipeline.pkl", "wb") as f:
        import pickle

        pickle.dump(pipe, f)
    features_meta = {
        "columns": feature_cols,
        "window": window,
        "gd_margin": gd_margin,
        "target": target,
        "model": best_name,
        "sampler": sampler_name,
        "leagues": used,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train": int(len(x_all)),
        "rows_total": int(before),
        "rows_dropped_nan": int(dropped),
        "tune_trials": int(tune_trials),
        "tuned_params": (tuned or {}).get("best_params"),
    }
    (out_dir / "features.json").write_text(json.dumps(features_meta, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({
            "walk_forward": {n: results[n] for n in results},
            "baseline": _baseline_metrics(df, target),
            "best_model": best_name,
            "tuning": tuned,
        }, indent=2), encoding="utf-8"
    )
    return {
        "target": target,
        "best_model": best_name,
        "walk_forward": {n: results[n]["aggregate"] for n in results},
        "baseline": _baseline_metrics(df, target),
        "tuning": tuned,
        "n_rows": int(before),
        "n_rows_dropped_nan": int(dropped),
        "artifacts_dir": str(out_dir),
    }
