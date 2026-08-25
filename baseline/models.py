"""Prediction models: Poisson (Dixon-Coles), Ensemble, PredictionResult.

The Poisson model estimates goal rates from pre-match FEATURES (form
attack/defense + xG history) -- NOT from market odds. That keeps it an
independent estimator whose edge vs the market is meaningful. The Elo model
is rating-based. The ensemble blends their 1X2 probability vectors.

All returned probabilities are renormalized so 1X2 sums to exactly 1.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .context import MatchContext, utc_now_iso
from .elo import EloModel

MODEL_VERSION = "0.1.0-elo-poisson"

MAX_GOALS = 10


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dc_tau(lh: float, la: float, h: int, a: int, rho: float) -> float:
    """Dixon-Coles low-score adjustment (default rho=0 -> plain Poisson)."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def poisson_matrix(lh: float, la: float, rho: float = 0.0) -> list[list[float]]:
    """Score matrix P(home=h, away=a) for h,a in [0..MAX_GOALS].

    Renormalized so the matrix sums to 1.0 -> every derived probability is a
    proper probability.
    """
    matrix = [[0.0 for _ in range(MAX_GOALS + 1)] for _ in range(MAX_GOALS + 1)]
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            matrix[h][a] = (
                poisson_pmf(h, lh) * poisson_pmf(a, la) * dc_tau(lh, la, h, a, rho)
            )
    total = sum(sum(row) for row in matrix)
    if total > 0:
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                matrix[h][a] /= total
    return matrix


def probs_from_matrix(
    matrix: list[list[float]],
) -> tuple[dict[str, float], float, float, float, float]:
    """Return (1x2 dict, p_over_1.5, p_over_2.5, p_over_3.5, p_btts_yes).

    1X2 always sums to 1.0 by construction.
    """
    p_home = p_draw = p_away = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = matrix[h][a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

    p_under15 = sum(
        matrix[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h + a <= 1
    )
    p_under25 = sum(
        matrix[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h + a <= 2
    )
    p_under35 = sum(
        matrix[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h + a <= 3
    )
    p_no_both = matrix[0][0] + sum(
        matrix[0][a] for a in range(1, MAX_GOALS + 1)
    ) + sum(matrix[h][0] for h in range(1, MAX_GOALS + 1))
    return (
        {"home": p_home, "draw": p_draw, "away": p_away},
        1.0 - p_under15,
        1.0 - p_under25,
        1.0 - p_under35,
        1.0 - p_no_both,
    )


class PoissonModel:
    """Feature-based Poisson with Dixon-Coles rho.

    Lambda home/away come from each team's pre-match attack/defense. When raw
    recent scorelines are available (``ctx.home_recent_goals`` etc.) the
    averages are TIME-DECAY weighted (Dixon-Coles xi): recent matches count
    more than old ones. Otherwise precomputed gf/ga averages are used as-is.
    xG history, when present, is blended with weight ``xg_weight`` (xG is more
    predictive of future results than raw goals, so the default favors it).
    No market odds involved. Returns None when the context lacks the minimum
    features (missing data stays missing).
    """

    def __init__(
        self,
        base_home_goals: float = 1.45,
        base_away_goals: float = 1.25,
        dc_rho: float = -0.1,
        shrinkage_samples: int = 5,
        time_decay_xi: float = 0.9,
        xg_weight: float = 0.65,
        rest_days_k: float = 0.0,
    ) -> None:
        self.base_home = base_home_goals
        self.base_away = base_away_goals
        self.rho = dc_rho
        self.shrinkage = max(1, shrinkage_samples)
        self.xi = max(0.0, min(1.0, time_decay_xi))
        self.xg_weight = max(0.0, min(1.0, xg_weight))
        # EXPERIMENTAL rest-day adjustment (audit PHASE 2): 0 = disabled. When
        # enabled and both teams' rest days are known pre-match, lambdas are
        # scaled by 1 + k*(rest_days - 7) (clamped) -- fewer rest days ->
        # slightly weaker. Off by default until walk-forward evidence.
        self.rest_days_k = max(0.0, min(0.1, rest_days_k))

    def predict(self, ctx: MatchContext) -> dict[str, Any] | None:
        if not ctx.has_attack_defense:
            return None

        hgf, hga, hs = self._team_stats(ctx.home_gf_avg, ctx.home_ga_avg, ctx.form_samples, ctx.home_recent_goals)
        agf, aga, as_ = self._team_stats(ctx.away_gf_avg, ctx.away_ga_avg, ctx.form_samples, ctx.away_recent_goals)
        if hgf is None or hga is None or agf is None or aga is None:
            return None

        w = min(1.0, min(hs, as_) / self.shrinkage) if (hs and as_) else 0.6

        ha, hd = self._strength(hgf, hga)
        aa, ad = self._strength(agf, aga)
        lh = self.base_home * math.sqrt(ha * ad)
        la = self.base_away * math.sqrt(aa * hd)
        lh = lh * w + self.base_home * (1.0 - w)
        la = la * w + self.base_away * (1.0 - w)

        if ctx.has_xg:
            xh = (ctx.home_xg_for + ctx.away_xg_against) / 2.0
            xa = (ctx.away_xg_for + ctx.home_xg_against) / 2.0
            if xh > 0 and xa > 0:
                lh = lh * (1.0 - self.xg_weight) + xh * self.xg_weight
                la = la * (1.0 - self.xg_weight) + xa * self.xg_weight

        if (
            self.rest_days_k > 0.0
            and ctx.home_rest_days is not None
            and ctx.away_rest_days is not None
        ):
            def _rest_adj(rest: int) -> float:
                f = 1.0 + self.rest_days_k * (rest - 7.0)
                return max(0.92, min(1.08, f))

            lh *= _rest_adj(ctx.home_rest_days)
            la *= _rest_adj(ctx.away_rest_days)

        lh = max(0.2, min(4.5, lh))
        la = max(0.2, min(4.5, la))
        p1x2, o15, o25, o35, btts = probs_from_matrix(poisson_matrix(lh, la, self.rho))
        return {
            "1x2": p1x2,
            "over_1.5": o15,
            "over_2.5": o25,
            "over_3.5": o35,
            "btts_yes": btts,
            "lambda_home": lh,
            "lambda_away": la,
            "lambda_source": "features+xg" if ctx.has_xg else "features",
        }

    def _team_stats(
        self,
        gf_avg: float | None,
        ga_avg: float | None,
        form_samples: int,
        recent: list[tuple[int, int]] | None,
    ) -> tuple[float | None, float | None, int]:
        """Prefer time-decayed averages from raw scorelines when available;
        fall back to the precomputed (equal-weight) averages."""
        if recent:
            n = len(recent)
            weights = [self.xi ** (n - 1 - i) for i in range(n)]
            sw = sum(weights)
            gf = sum(g[0] * w_i for g, w_i in zip(recent, weights)) / sw
            ga = sum(g[1] * w_i for g, w_i in zip(recent, weights)) / sw
            return gf, ga, n
        if gf_avg is not None and ga_avg is not None:
            return gf_avg, ga_avg, form_samples
        return None, None, 0

    def _strength(self, gf: float, ga: float) -> tuple[float, float]:
        atk = (gf / self.base_home) if self.base_home > 0 else 1.0
        deff = (ga / self.base_away) if self.base_away > 0 else 1.0
        return atk, deff


class Ensemble:
    """Weighted blend of available 1X2 probability vectors.

    Only models that produced a prediction participate; weights are
    renormalized over the active set. The blended vector is renormalized so
    it sums to exactly 1.0.
    """

    def __init__(self, elo_weight: float = 0.5, poisson_weight: float = 0.5) -> None:
        self.weights = {"elo": elo_weight, "poisson": poisson_weight}

    def predict(
        self,
        ctx: MatchContext,
        elo: EloModel,
        poisson: PoissonModel,
    ) -> dict[str, Any] | None:
        parts: list[tuple[float, dict[str, float], str]] = []
        lh, la = elo.expected_lambdas(ctx.home, ctx.away)
        p1x2_elo, _, _, _, _ = probs_from_matrix(poisson_matrix(lh, la, rho=0.0))
        parts.append((self.weights["elo"], p1x2_elo, "elo"))

        pm = poisson.predict(ctx)
        if pm is not None:
            parts.append((self.weights["poisson"], pm["1x2"], "poisson"))

        total_w = sum(w for w, _, _ in parts)
        blended = {
            k: sum(w * p[k] for w, p, _ in parts) / total_w
            for k in ("home", "draw", "away")
        }
        s = sum(blended.values())
        blended = {k: v / s for k, v in blended.items()}
        return {
            "1x2": blended,
            "models": [name for _, _, name in parts],
            "weights": {name: w / total_w for w, _, name in parts},
        }


@dataclass
class PredictionResult:
    """Separate the concepts: probability / confidence / signal / edge."""

    model_probs: dict[str, Any]          # {1x2, over_*, btts_yes, lambda_*}
    confidence: float                    # 0..1 label threshold
    signal_strength: int                 # 0..100 overall quality
    market_edge: dict[str, float]        # per-selection edge % vs margin-free market
    calibration: dict[str, Any]          # {quality, ece, samples}
    agreement: dict[str, Any]            # {model_vs_model, model_vs_market, models}
    data_completeness: float             # 0..1
    model_version: str
    as_of: str
    input_hash: str
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_probs": self.model_probs,
            "confidence": round(self.confidence, 3),
            "signal_strength": self.signal_strength,
            "market_edge": self.market_edge,
            "calibration": self.calibration,
            "agreement": self.agreement,
            "data_completeness": round(self.data_completeness, 3),
            "model_version": self.model_version,
            "as_of": self.as_of,
            "input_hash": self.input_hash,
            "sources": self.sources,
        }


def _normalize_implied(odds: dict[str, float]) -> dict[str, float] | None:
    raw = {k: (1.0 / v if v and v > 1.0 else 0.0) for k, v in odds.items()}
    total = sum(raw.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in raw.items()}


def run_prediction_engine(
    ctx: MatchContext,
    *,
    elo: EloModel,
    poisson: PoissonModel,
    ensemble: Ensemble,
    calibrator: Any = None,
    scorer: Any = None,
) -> PredictionResult | None:
    """Orchestrate models + calibration + scoring into a PredictionResult.

    Returns None when no model can produce a prediction (missing data).
    """
    ens = ensemble.predict(ctx, elo, poisson)
    if ens is None:
        return None
    p1x2 = ens["1x2"]

    # Totals/BTTS: prefer the feature-based Poisson lambdas; fall back to Elo.
    pm = poisson.predict(ctx)
    if pm is not None:
        totals = {k: pm[k] for k in ("over_1.5", "over_2.5", "over_3.5", "btts_yes")}
        lh, la, lam_src = pm["lambda_home"], pm["lambda_away"], pm["lambda_source"]
    else:
        lh, la = elo.expected_lambdas(ctx.home, ctx.away)
        _, o15, o25, o35, btts = probs_from_matrix(poisson_matrix(lh, la, rho=0.0))
        totals = {"over_1.5": o15, "over_2.5": o25, "over_3.5": o35, "btts_yes": btts}
        lam_src = "elo"

    # Market comparison uses margin-free implied probabilities so model and
    # market are on the same scale (edge is honest).
    norm_implied = _normalize_implied(ctx.consensus_odds) if ctx.has_odds else None
    market_edge: dict[str, float] = {}
    model_vs_market: float | None = None
    if norm_implied:
        market_edge = {
            side: round((p1x2[side] - norm_implied[side]) * 100.0, 2)
            for side in ("home", "draw", "away")
        }
        model_vs_market = 1.0 - min(
            1.0, sum(abs(p1x2[k] - norm_implied[k]) for k in ("home", "draw", "away"))
        )

    model_vs_model: float | None = None
    if len(ens["models"]) >= 2:
        p_elo, _, _, _, _ = probs_from_matrix(
            poisson_matrix(*elo.expected_lambdas(ctx.home, ctx.away), rho=0.0)
        )
        model_vs_model = max(
            0.0,
            min(1.0, 1.0 - sum(abs(p1x2[k] - p_elo[k]) for k in ("home", "draw", "away"))),
        )

    # Calibration (log-odds linear) applied to the blended 1X2 vector.
    calib_info = {"quality": 0.0, "ece": None, "samples": 0}
    if calibrator is not None:
        applied = {k: calibrator.apply(p1x2[k]) for k in ("home", "draw", "away")}
        s = sum(applied.values())
        if s > 0:
            p1x2 = {k: v / s for k, v in applied.items()}
        calib_info = calibrator.quality()

    components = scorer.components(
        ctx=ctx,
        ensemble_models=ens["models"],
        model_vs_market=model_vs_market,
        model_vs_model=model_vs_model,
        calibration_quality=calib_info["quality"],
        market_edge=market_edge,
    ) if scorer is not None else {
        "data_completeness": 0.0, "agreement": 0.0,
        "signal_strength": 0, "confidence": 0.0,
    }

    model_probs = {
        "1x2": {k: round(v, 4) for k, v in p1x2.items()},
        "over_1.5": round(totals["over_1.5"], 4),
        "over_2.5": round(totals["over_2.5"], 4),
        "over_3.5": round(totals["over_3.5"], 4),
        "btts_yes": round(totals["btts_yes"], 4),
        "lambda_home": round(lh, 3),
        "lambda_away": round(la, 3),
        "lambda_source": lam_src,
        "models": ens["models"],
        "model_weights": ens["weights"],
        # True once both teams exist in the seeded ratings; otherwise the Elo
        # contribution is only the home-advantage prior (honesty labeling).
        "elo_seeded": elo.known(ctx.home, ctx.away),
    }

    return PredictionResult(
        model_probs=model_probs,
        confidence=round(components["confidence"], 3),
        signal_strength=int(components["signal_strength"]),
        market_edge=market_edge,
        calibration=calib_info,
        agreement={
            "model_vs_model": round(model_vs_model, 3) if model_vs_model is not None else None,
            "model_vs_market": round(model_vs_market, 3) if model_vs_market is not None else None,
            "models": ens["models"],
        },
        data_completeness=round(components["data_completeness"], 3),
        model_version=MODEL_VERSION,
        as_of=utc_now_iso(),
        input_hash=ctx.input_hash,
        sources=ctx.sources,
    )
