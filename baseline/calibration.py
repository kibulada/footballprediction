"""Calibration + SignalScorer v2.

Calibrator: fits ``p' = sigmoid(a + b * logit(p))`` on backtest residuals via
ordinary least squares (pure Python, no deps). Persists params locally.

SignalScorer v2: evaluates FIVE named components separately instead of one
opaque integer:
  - data_completeness  (0..1)  what pre-match data exists (odds/form/xG/H2H)
  - model_agreement    (0..1)  model-vs-model and model-vs-market agreement
  - market_edge        (ppts)  best margin-free edge across 1X2 selections
  - calibration_quality(0..1)  from the Calibrator (0 = not yet validated)
  - signal_strength    (0..100) documented weighted combination of the above
  - confidence         (0..1)  documented weighted combination (label basis)

The components stay accessible individually on PredictionResult.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .context import MatchContext


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def expected_calibration_error(probs: list[float], outcomes: list[int], bins: int = 10) -> float:
    """ECE over the given (probability, binary outcome) pairs."""
    if not probs or len(probs) != len(outcomes):
        return float("nan")
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    count = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idxs = [i for i, p in enumerate(probs) if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if not idxs:
            continue
        avg_p = sum(probs[i] for i in idxs) / len(idxs)
        freq = sum(outcomes[i] for i in idxs) / len(idxs)
        total += len(idxs) * abs(avg_p - freq)
        count += len(idxs)
    return (total / count) if count else float("nan")


class Calibrator:
    """Log-odds linear calibration fitted on historical (prob, outcome) pairs."""

    def __init__(self, path: str | Path | None = None, min_samples: int = 200) -> None:
        self.path = Path(path) if path else None
        self.min_samples = min_samples
        self.a = 0.0
        self.b = 1.0
        self.samples = 0
        self.ece = None
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.a = float(payload.get("a", 0.0))
            self.b = float(payload.get("b", 1.0))
            self.samples = int(payload.get("samples", 0))
            self.ece = payload.get("ece")
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            self.a, self.b, self.samples, self.ece = 0.0, 1.0, 0, None

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"a": self.a, "b": self.b, "samples": self.samples, "ece": self.ece},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ---- fit / apply ---------------------------------------------------
    def fit(self, probs: list[float], outcomes: list[Any]) -> None:
        """Fit P(outcome=1) = sigmoid(a + b * logit(p)) by IRLS logistic
        regression (Newton-Raphson, pure Python). Requires both 0 and 1
        outcomes present; fractional targets are accepted for tests."""
        if len(probs) < 10 or len(probs) != len(outcomes):
            return
        xs = [_logit(p) for p in probs]
        ys = [float(o) for o in outcomes]
        if len(set(round(y) for y in ys)) < 2 and not any(0.0 < y < 1.0 for y in ys):
            return  # degenerate target (all same outcome)
        a, b = 0.0, 1.0
        for _ in range(50):
            mu = [_sigmoid(a + b * x) for x in xs]
            g_a = sum(y - m for y, m in zip(ys, mu))
            g_b = sum((y - m) * x for y, m, x in zip(ys, mu, xs))
            w = [m * (1.0 - m) for m in mu]
            h_aa = sum(w)
            h_ab = sum(wi * x for wi, x in zip(w, xs))
            h_bb = sum(wi * x * x for wi, x in zip(w, xs))
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            da = (g_a * h_bb - g_b * h_ab) / det
            db = (g_b * h_aa - g_a * h_ab) / det
            a2, b2 = a + da, b + db
            if not (abs(a2) < 1e6 and abs(b2) < 1e6):
                break  # perfect separation: coefficients diverge; keep last stable
            a, b = a2, b2
            if abs(da) < 1e-9 and abs(db) < 1e-9:
                break
        self.a, self.b = a, b
        self.samples = len(probs)
        self.ece = expected_calibration_error(
            [self.apply(p) for p in probs], [1 if y > 0.5 else 0 for y in ys]
        )
        self._save()

    def apply(self, p: float) -> float:
        if self.samples < self.min_samples:
            return p
        return _sigmoid(self.a + self.b * _logit(p))

    def quality(self) -> dict[str, Any]:
        """{quality 0..1, ece, samples}; quality 0 until validated by backtest."""
        if self.samples < self.min_samples:
            return {"quality": 0.0, "ece": self.ece, "samples": self.samples}
        ece = self.ece if self.ece is not None else float("nan")
        if math.isnan(ece):
            return {"quality": 0.0, "ece": None, "samples": self.samples}
        quality = max(0.0, min(1.0, 1.0 - ece * 2.0))
        return {"quality": quality, "ece": round(ece, 4), "samples": self.samples}


class SignalScorer:
    """Component-based signal scorer (see module docstring for the breakdown)."""

    # Component weights for the OVERALL signal strength (documented).
    W_SIGNAL = {
        "completeness": 0.40,
        "agreement": 0.30,
        "calibration": 0.30,
    }
    # Confidence weights (label basis).
    W_CONF = {
        "completeness": 0.20,
        "agreement": 0.30,
        "calibration": 0.50,
    }

    def components(
        self,
        *,
        ctx: MatchContext,
        ensemble_models: list[str],
        model_vs_market: float | None,
        model_vs_model: float | None,
        calibration_quality: float,
        market_edge: dict[str, float],
    ) -> dict[str, Any]:
        completeness = self._completeness(ctx)
        agreement = self._agreement(model_vs_market, model_vs_model)
        edge_pct = max((abs(v) for v in market_edge.values()), default=0.0)
        signal = 100.0 * (
            self.W_SIGNAL["completeness"] * completeness
            + self.W_SIGNAL["agreement"] * agreement
            + self.W_SIGNAL["calibration"] * min(1.0, calibration_quality)
        )
        confidence = (
            self.W_CONF["completeness"] * completeness
            + self.W_CONF["agreement"] * agreement
            + self.W_CONF["calibration"] * min(1.0, calibration_quality)
        )
        return {
            "data_completeness": round(completeness, 3),
            "model_agreement": round(agreement, 3),
            "market_edge_pct": round(edge_pct, 2),
            "calibration_quality": round(min(1.0, calibration_quality), 3),
            "signal_strength": round(signal),
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
        }

    @staticmethod
    def _completeness(ctx: MatchContext) -> float:
        parts = []
        parts.append(0.30 if ctx.has_odds else 0.0)
        parts.append(0.20 if (ctx.home_form and ctx.away_form) else 0.0)
        parts.append(0.20 if ctx.has_attack_defense else 0.0)
        parts.append(0.20 if ctx.has_xg else 0.0)
        parts.append(0.10 if (ctx.h2h and any(ctx.h2h.values())) else 0.0)
        return sum(parts)

    @staticmethod
    def _agreement(
        model_vs_market: float | None,
        model_vs_model: float | None,
    ) -> float:
        vals = [v for v in (model_vs_market, model_vs_model) if v is not None]
        if not vals:
            return 0.0
        return max(0.0, min(1.0, sum(vals) / len(vals)))
