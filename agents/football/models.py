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


# ---------------------------------------------------------------------------
# Phase 1: lineups/injuries + rest days as lambda CORRECTION FACTORS.
#
# All three helpers are pure and flag-gated (default weight 0 in
# PoissonModel), so existing predictions stay byte-identical until the
# feature flag is enabled AND the Phase 1 backtest DoD passes. The lineup
# multiplier is documented as a placeholder mapping (ground rule #4): the
# available sources (flashscore lineups + missing players) carry player
# NAMES, not ratings, so "attacking output share / GK save share" cannot be
# computed from real data today. Chosen proxy: each confirmed missing
# STARTER costs ``missing_starter_cost`` (5%) of the side's lambda, capped at
# ``max_cut`` (20%); a PREDICTED (unconfirmed) lineup halves the effect
# (spec 1.2: weight x0.5); no lineup data -> no change.
# ---------------------------------------------------------------------------


def lineup_usable(lineup_ts: str | None, kickoff: str | None) -> bool:
    """Phase 1.3 leakage guard: a lineup fetched at or after kickoff is
    rejected as a model input. No timestamp -> usable (legacy rows)."""
    from datetime import datetime, timezone

    if not lineup_ts or not kickoff:
        return True

    def _parse(ts: str) -> datetime | None:
        try:
            cleaned = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
            dt = datetime.fromisoformat(cleaned)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

    lu = _parse(lineup_ts)
    ko = _parse(kickoff)
    if lu is None or ko is None:
        return True
    return lu.astimezone(timezone.utc) < ko.astimezone(timezone.utc)


def lineup_lambda_multiplier(
    starters: list[str] | None,
    missing: list[str] | None,
    status: str | None,
    *,
    missing_starter_cost: float = 0.05,
    max_cut: float = 0.20,
) -> float:
    """L_side for lambda' = lambda x L (Phase 1.2).

    1.0 = no change. Confirmed lineups get full weight; predicted/
    unconfirmed lineups apply the effect at half weight (spec: x0.5). The
    cut is derived from the count of missing players that intersect the
    starter list (attacking-output-share proxy -- player ratings are not
    available in the data sources).
    """
    missing = [m for m in (missing or []) if m]
    if not missing:
        return 1.0
    if starters:
        n_missing_starters = sum(1 for m in missing if m in starters)
    else:
        # No lineup list: cannot verify who is a starter -- half weight.
        n_missing_starters = len(missing) * 0.5
    effect = min(max_cut, missing_starter_cost * n_missing_starters)
    if status != "confirmed":
        effect *= 0.5  # predicted/unconfirmed lineups: half weight
    return max(0.0, 1.0 - effect)


def rest_days_multiplier(
    days_rest: float | None,
    *,
    threshold_days: float = 4.0,
    penalty: float = 0.05,
) -> float:
    """Phase 1.4: congestion/travel penalty for teams on < ``threshold_days``
    rest. 1.0 when days_rest is unknown (no data -> no change). A team on
    2 days' rest gets 1.0 - 0.05 = 0.95 of its lambda; the penalty never
    exceeds 0.10 (2+ days below threshold). Long-travel inputs are NOT
    available in the current data sources; the hook takes an explicit
    ``days_rest`` so the caller can feed travel-adjusted rest when a source
    provides it."""
    if days_rest is None:
        return 1.0
    if days_rest >= threshold_days:
        return 1.0
    short = max(0.0, threshold_days - days_rest)
    return max(0.90, 1.0 - penalty * short)


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


def apply_elo_anchor(
    lh: float,
    la: float,
    rating_home: float,
    rating_away: float,
    *,
    min_gap: float = 150.0,
    full_gap: float = 400.0,
) -> tuple[float, float, float]:
    """F1 (plan v3): pull feature λ toward the seeded-Elo λ by rating gap.

    Pure. ``t`` ramps 0→1 between ``min_gap`` and ``full_gap`` Elo points;
    the returned λ is ``(1-t)*feature + t*elo_directional_share`` where the
    Elo target keeps the SAME total as the input λ and only re-splits it by
    strength (re-splitting the total avoids dragging Totals/BTTS around --
    that is F2's job). Returns ``(lh, la, t)``; ``t == 0`` means untouched.

    Verified failure this fixes: Elche v Barcelona 2026-08-23 -- features
    gave λh 1.542 > λa 1.359 against a 715-point Elo gap, producing
    P(AH Away -1.5) = 15% vs market 51%; BTTS Yes then "won" the card.
    """
    gap = abs(float(rating_home) - float(rating_away))
    if full_gap <= min_gap:
        return lh, la, 0.0
    t = max(0.0, min(1.0, (gap - float(min_gap)) / (float(full_gap) - float(min_gap))))
    if t <= 0.0:
        return lh, la, 0.0
    total = float(lh) + float(la)
    if total <= 0.0:
        return lh, la, 0.0
    diff = float(rating_home) - float(rating_away)
    share = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    elo_lh = total * share
    elo_la = total * (1.0 - share)
    return (1.0 - t) * lh + t * elo_lh, (1.0 - t) * la + t * elo_la, t


def calibrate_total_to_market(
    lh: float,
    la: float,
    market_totals: dict[str, Any] | None,
    *,
    weight: float = 0.5,
) -> tuple[float, float, bool]:
    """F2 (plan v3): pull the FINAL selected λ toward the market-implied total.

    Pure. Uses the margin-free (devigged) Over/Under 2.5 pair -- NOT the raw
    1/odds the Ensemble's internal copy uses -- and closes HALF the gap per
    application (``weight``), clamped to [0.7, 1.8] like the original.
    Returns ``(lh, la, applied)``; ``applied`` False when no usable pair or
    the gap is below the 5pp noise floor. Keeps the home/away RATIO fixed:
    it corrects the TOTAL, never the direction (that is F1's job).
    """
    mt = market_totals or {}
    try:
        o = float((mt.get("Over 2.5") or {}).get("odds") or 0)
        u = float((mt.get("Under 2.5") or {}).get("odds") or 0)
    except (TypeError, ValueError):
        return lh, la, False
    if o <= 1.0 or u <= 1.0:
        return lh, la, False
    io, iu = 1.0 / o, 1.0 / u
    fair_over = io / (io + iu)
    matrix = poisson_matrix(float(lh), float(la), rho=0.0)
    _, _, model_over25, _, _ = probs_from_matrix(matrix)
    gap = fair_over - float(model_over25)
    if abs(gap) < 0.05:
        return lh, la, False
    scale = max(0.7, min(1.8, 1.0 + gap * float(weight)))
    new_lh = float(lh) * scale
    new_la = float(la) * scale
    return (
        max(0.3, min(3.5, new_lh)),
        max(0.3, min(3.5, new_la)),
        True,
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
        min_samples: int = 2,
        lineup_weight: float = 0.0,
        rest_days_weight: float = 0.0,
        elo_anchor: dict[str, Any] | None = None,
        market_total_calibration: dict[str, Any] | None = None,
    ) -> None:
        self.base_home = base_home_goals
        self.base_away = base_away_goals
        self.rho = dc_rho
        self.shrinkage = max(1, shrinkage_samples)
        # Form-window floor for the feature lambdas: fewer finished matches
        # than this (per team) is noise, not signal, so the engine replaces
        # the feature λ with the Elo λ entirely (option 1). 0 disables the
        # gate (pure feature Poisson, pre-fix behaviour).
        self.min_samples = max(0, int(min_samples))
        self.xi = max(0.0, min(1.0, time_decay_xi))
        self.xg_weight = max(0.0, min(1.0, xg_weight))
        # Phase 1: lineup/injury + rest-day lambda CORRECTION factors. Both
        # default to 0 (off) -- the Phase 1 DoD (backtest: lineup subset
        # improves log-loss without adding ROI bias) must pass before these
        # are enabled in production. Lineup-λ is a correction applied to the
        # PINNED lambda, never a replacement lambda source (Phase 3.2).
        self.lineup_weight = max(0.0, min(1.0, lineup_weight))
        self.rest_days_weight = max(0.0, min(1.0, rest_days_weight))
        # Plan v3 (2026-08-24, laporan bestpick_evaluasi_elche-barca):
        # F1 -- anchor the selected λ toward the seeded-Elo λ when the rating
        # gap is large (feature-only λ on a thin early-season window produced
        # Elche λ >= Barcelona λ against a 715-point Elo gap). F2 -- pull the
        # FINAL selected λ toward the margin-free market-implied total so the
        # Totals/BTTS path stops diverging structurally from the 1X2 path
        # (the Ensemble's internal _calibrate_total_goals never reached this
        # λ). Both are pure post-selection adjustments applied identically on
        # every query -- pin labels and reproducibility are untouched; audit
        # fields carry what happened (elo_anchor_t / market_total_calibrated).
        self.elo_anchor = dict(elo_anchor or {})
        self.market_total_calibration = dict(market_total_calibration or {})

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

        # Phase 1.2 (lineup/injury correction, flag-gated). Applied to the
        # pinned lambda as a MULTIPLIER -- never a replacement lambda source.
        # Phase 1.3 leakage guard: a lineup fetched at/after kickoff is
        # rejected as a model input.
        lineup_applied = False
        if self.lineup_weight > 0 and lineup_usable(ctx.lineup_ts, ctx.kickoff_utc):
            lh *= 1.0 - self.lineup_weight * (
                1.0 - lineup_lambda_multiplier(ctx.lineup_home, ctx.missing_home, ctx.lineup_status)
            )
            la *= 1.0 - self.lineup_weight * (
                1.0 - lineup_lambda_multiplier(ctx.lineup_away, ctx.missing_away, ctx.lineup_status)
            )
            lineup_applied = bool(ctx.lineup_home or ctx.lineup_away or ctx.missing_home or ctx.missing_away)
        # Phase 1.4 (rest days / congestion, flag-gated). Unknown rest -> 1.0.
        rest_applied = False
        if self.rest_days_weight > 0:
            rh = rest_days_multiplier(ctx.home_days_rest)
            ra = rest_days_multiplier(ctx.away_days_rest)
            lh *= 1.0 - self.rest_days_weight * (1.0 - rh)
            la *= 1.0 - self.rest_days_weight * (1.0 - ra)
            rest_applied = bool(ctx.home_days_rest is not None or ctx.away_days_rest is not None)

        lh = max(0.3, min(3.5, lh))
        la = max(0.3, min(3.5, la))
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
            # Phase 1: whether lineup/rest corrections were actually applied
            # (auditable; both off by default).
            "lineup_correction_applied": lineup_applied,
            "rest_correction_applied": rest_applied,
            # Effective per-team form-window depth (min over both sides);
            # the engine uses it to blend toward Elo when the window is thin.
            "lambda_samples": int(min(hs, as_)) if (hs and as_) else 0,
        }

    def _team_stats(
        self,
        gf_avg: float | None,
        ga_avg: float | None,
        form_samples: int,
        recent: list[tuple] | None,
    ) -> tuple[float | None, float | None, int]:
        """Prefer time-decayed averages from raw scorelines when available;
        fall back to the precomputed (equal-weight) averages."""
        if recent:
            n = len(recent)

            def _weight(i: int, g: tuple) -> float:
                return self.xi ** (n - 1 - i)

            weights = [_weight(i, g) for i, g in enumerate(recent)]
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
        self._league_quality: dict[str, float] | None = None

    def _load_league_quality(self) -> dict[str, float]:
        """Load league quality coefficients (lazy, cached)."""
        if self._league_quality is not None:
            return self._league_quality
        try:
            import json
            from pathlib import Path
            # Search in cache dir relative to project root
            for base in [Path(__file__).resolve().parent.parent.parent / "cache" / "football",
                         Path("cache/football")]:
                path = base / "league_quality.json"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        self._league_quality = json.load(f)
                    return self._league_quality
        except Exception:
            pass
        self._league_quality = {}
        return self._league_quality

    def _league_quality_coeff(self, league: str) -> float:
        """Get quality coefficient for a league (0.0-1.0)."""
        lq = self._load_league_quality()
        key = (league or "").lower().strip()
        # Direct match
        if key in lq:
            return float(lq[key])
        # Try slug (replace spaces/special chars with -)
        slug = "".join(c if c.isalnum() else "-" for c in key).strip("-")
        if slug in lq:
            return float(lq[slug])
        return float(lq.get("default", 0.4))

    def _cross_league_adjustment(self, ctx: MatchContext, elo_lh: float, elo_la: float,
                                  poi_lh: float, poi_la: float) -> tuple[float, float, float, float]:
        """Balanced model-market blend when model deviates from market.
        Uses market odds as proxy for league quality and team strength.
        """
        odds = ctx.consensus_odds or {}
        if not odds.get('home') or not odds.get('away'):
            return elo_lh, elo_la, poi_lh, poi_la

        imp_h = 1.0 / odds['home']
        imp_d = 1.0 / odds['draw']
        imp_a = 1.0 / odds['away']
        total_imp = imp_h + imp_d + imp_a
        mkt_p_h = imp_h / total_imp

        avg_lh = (elo_lh + poi_lh) / 2.0
        avg_la = (elo_la + poi_la) / 2.0
        p_home_model_h, _, _, _, _ = probs_from_matrix(poisson_matrix(avg_lh, avg_la, rho=0.0))
        model_p_h = p_home_model_h.get('home', 0.33)

        dev_h = model_p_h - mkt_p_h

        if abs(dev_h) < 0.10:
            return elo_lh, elo_la, poi_lh, poi_la

        abs_dev = abs(dev_h)
        if abs_dev < 0.15:
            alpha = 0.20
        elif abs_dev < 0.25:
            alpha = 0.40
        elif abs_dev < 0.40:
            alpha = 0.60
        else:
            alpha = 0.80

        import math as _m
        model_lh = -_m.log(max(0.01, 1 - model_p_h)) * 1.3 if model_p_h < 0.95 else 3.5
        model_la = -_m.log(max(0.01, model_p_h)) * 1.3 if model_p_h > 0.05 else 3.5
        total_goals = model_lh + model_la
        mkt_lh = max(0.3, min(3.5, mkt_p_h * total_goals * 1.8))
        mkt_la = max(0.3, min(3.5, (1.0 - mkt_p_h) * total_goals * 1.8))

        adj_elo_h = max(0.3, min(3.5, (1.0 - alpha) * model_lh + alpha * mkt_lh))
        adj_elo_a = max(0.3, min(3.5, (1.0 - alpha) * model_la + alpha * mkt_la))
        adj_poi_h = max(0.3, min(3.5, (1.0 - alpha) * model_lh + alpha * mkt_lh))
        adj_poi_a = max(0.3, min(3.5, (1.0 - alpha) * model_la + alpha * mkt_la))

        return adj_elo_h, adj_elo_a, adj_poi_h, adj_poi_a

    def _calibrate_total_goals(self, lh: float, la: float, ctx: MatchContext) -> tuple[float, float]:
        """Calibrate lambdas using market O/U 2.5 odds.

        If model lambda implies different Over 2.5 probability than market,
        adjust lambdas proportionally to match market expectations.
        This prevents Under 3.5 over-prediction when market expects more goals.
        """
        totals = ctx.market_totals or {}
        ou25 = totals.get('Over 2.5') or totals.get('over_2.5') or {}
        ou_odds = ou25.get('odds') or ou25.get('price') or 0

        if not ou_odds or ou_odds <= 1.0:
            return lh, la  # no O/U data available

        # Market-implied P(Over 2.5)
        market_p_over = 1.0 / ou_odds

        # Model P(Over 2.5) from current lambdas
        model_p_over = 0.0
        for i in range(9):
            for j in range(9):
                if i + j > 2.5:
                    import math
                    model_p_over += math.exp(-lh) * (lh**i) / math.factorial(i) * \
                                    math.exp(-la) * (la**j) / math.factorial(j)

        # If model is too conservative (Under too high), increase lambda
        # If model is too aggressive (Over too high), decrease lambda
        gap = market_p_over - model_p_over

        if abs(gap) < 0.05:
            return lh, la  # gap too small, no adjustment needed

        # Adjust lambdas proportionally
        # Positive gap = market expects more goals = increase lambda
        # Negative gap = market expects fewer goals = decrease lambda
        total_lambdas = lh + la
        if total_lambdas <= 0:
            return lh, la

        # Scale factor: how much to adjust
        # Use 50% of the gap — aggressive enough to close the difference
        scale = 1.0 + gap * 0.5
        scale = max(0.7, min(1.8, scale))  # clamp to 70%-180%

        new_total = total_lambdas * scale
        # Maintain same home/away ratio
        new_lh = new_total * (lh / total_lambdas)
        new_la = new_total * (la / total_lambdas)

        return max(0.3, min(3.5, new_lh)), max(0.3, min(3.5, new_la))

    def predict(
        self,
        ctx: MatchContext,
        elo: EloModel,
        poisson: PoissonModel,
    ) -> dict[str, Any] | None:
        parts: list[tuple[float, dict[str, float], str]] = []
        elo_lh, elo_la = elo.expected_lambdas(ctx.home, ctx.away)

        # Get Poisson lambdas if available
        pm = poisson.predict(ctx)
        poi_lh = elo_lh  # default to Elo if Poisson not available
        poi_la = elo_la
        if pm is not None:
            poi_lh = pm.get("lambda_home", elo_lh)
            poi_la = pm.get("lambda_away", elo_la)

        # Apply cross-league quality adjustment (uses market odds as proxy)
        adj_elo_h, adj_elo_a, adj_poi_h, adj_poi_a = self._cross_league_adjustment(
            ctx, elo_lh, elo_la, poi_lh, poi_la
        )

        # Calibrate total goals using O/U market odds
        adj_elo_h, adj_elo_a = self._calibrate_total_goals(adj_elo_h, adj_elo_a, ctx)
        adj_poi_h, adj_poi_a = self._calibrate_total_goals(adj_poi_h, adj_poi_a, ctx)

        # Track if adjustment was actually applied
        odds = ctx.consensus_odds or {}
        league_quality_applied = False
        if odds.get('home'):
            imp_h = 1.0 / odds['home']
            imp_d = 1.0 / odds['draw']
            imp_a = 1.0 / odds['away']
            total_imp = imp_h + imp_d + imp_a
            mkt_p_h = imp_h / total_imp
            p_home_model_h, _, _, _, _ = probs_from_matrix(poisson_matrix(elo_lh, elo_la, rho=0.0))
            model_p_h = p_home_model_h.get('home', 0.33)
            dev_h = model_p_h - mkt_p_h
            league_quality_applied = abs(dev_h) >= 0.10

        # Calculate probabilities from adjusted lambdas
        p1x2_elo, _, _, _, _ = probs_from_matrix(poisson_matrix(adj_elo_h, adj_elo_a, rho=0.0))

        # Zero-weight components are skipped (not just weighted by 0): with
        # elo_weight=0 and no feature-Poisson data the naive ``parts`` would
        # hold only zero weights and total_w would be 0 -> ZeroDivisionError.
        if self.weights["elo"] > 0:
            parts.append((self.weights["elo"], p1x2_elo, "elo"))

        if pm is not None and self.weights["poisson"] > 0:
            # Use adjusted Poisson lambdas
            p1x2_poi, _, _, _, _ = probs_from_matrix(poisson_matrix(adj_poi_h, adj_poi_a, poisson.rho))
            parts.append((self.weights["poisson"], p1x2_poi, "poisson"))

        if not parts:
            # No active component carries a positive weight: an empty blend
            # is not a prediction. Report None (missing data) so every caller
            # degrades honestly instead of crashing.
            return None

        total_w = sum(w for w, _, _ in parts)
        blended = {
            k: sum(w * p[k] for w, p, _ in parts) / total_w
            for k in ("home", "draw", "away")
        }
        s = sum(blended.values())
        blended = {k: v / s for k, v in blended.items()}
        # TODO-13: ensemble spread = max-min across component home
        # probabilities. Large spread = the components disagree = the blended
        # estimate is less reliable; surfaced as ``uncertainty`` downstream
        # (decision WATCH gate, TODO-10/16). Zero when a single model ran.
        home_probs = [p["home"] for _, p, _ in parts]
        spread = (max(home_probs) - min(home_probs)) if len(home_probs) > 1 else 0.0

        # Calculate average adjusted lambdas for reporting
        avg_lh = (adj_elo_h + adj_poi_h) / 2.0 if pm is not None else adj_elo_h
        avg_la = (adj_elo_a + adj_poi_a) / 2.0 if pm is not None else adj_elo_a

        # O/U and BTTS probabilities from lambdas (probs_from_matrix returns them directly)
        _p1x2, _p_ou15, _p_ou25, _p_ou35, _p_btts = probs_from_matrix(
            poisson_matrix(avg_lh, avg_la, rho=0.0)
        )
        over_25 = _p_ou25
        under_25 = 1.0 - over_25
        over_35 = _p_ou35
        under_35 = 1.0 - over_35

        # Draw detection: if both teams low-scoring and probs are close
        total_exp_goals = avg_lh + avg_la
        draw_boosted = (
            blended["draw"] > 0.22
            and abs(blended["home"] - blended["away"]) < 0.15
            and total_exp_goals < 2.5
        )

        return {
            "1x2": blended,
            "models": [name for _, _, name in parts],
            "weights": {name: w / total_w for w, _, name in parts},
            "spread": round(spread, 4),
            "lambda_home": round(avg_lh, 4),
            "lambda_away": round(avg_la, 4),
            "over_2.5": round(over_25, 4),
            "under_2.5": round(under_25, 4),
            "over_3.5": round(over_35, 4),
            "under_3.5": round(under_35, 4),
            "draw_boosted": draw_boosted,
            "league_quality_applied": league_quality_applied,
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
    decisiveness: float = 0.0            # S26: 1X2 magnitude + separation
    model_version: str = MODEL_VERSION
    as_of: str = field(default_factory=utc_now_iso)
    input_hash: str = ""
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
            "decisiveness": round(self.decisiveness, 3),
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


# Option 3 (lambda blend, EXPERIMENTAL -- backtest-only, NOT SHIPPED):
# "threshold" is the CURRENT production selection (Elo fallback below
# min_samples, linear blend toward Elo up to shrinkage, features as-is
# above); "blend" ALWAYS mixes the feature λ toward Elo by sample count,
# so a thin form window degrades the feature estimate smoothly instead of
# switching estimators at a hard cutoff (the Rio Ave 0.58/2.12 -> 1.04/1.15
# flip). Both modes return the same shape, so the backtest can A/B them on
# the SAME walk-forward state with byte-identical bookkeeping.
#
# DECISION (2026-08, reviewed): the A/B on 1,520 real EPL matches
# (cache/football/epl_fixtures_2022_2026.json) showed NO meaningful
# difference -- hit rate within 0.2pp, logloss within 0.0002, ROI ~-14%
# for BOTH modes. Blend was REJECTED for production: it changes the
# totals/BTTS calibration surface for every league without measured gain,
# and the estimator-flip problem it targets is already solved by the
# lambda pin (Fix 2). Keep production on "threshold". The "blend" mode
# remains here ONLY as a research tool (backtest --lambda-mode blend); it
# is never called from the analyse/prediction path.
LAMBDA_BLEND_TARGET = 5  # form samples at which blend reaches full feature weight


def lambda_from_mode(
    *,
    lh_e: float,
    la_e: float,
    lh_f: float,
    la_f: float,
    samples: int,
    min_samples: int,
    shrinkage: int,
    rho: float,
    mode: str = "pinned",
    feature_label: str = "features",
) -> dict[str, Any]:
    """Pure λ selection: returns {lambda_home, lambda_away, lambda_source,
    over_1.5, over_2.5, over_3.5, btts_yes}.

    ``feature_label`` carries the estimator's OWN label ("features" or
    "features+xg" from PoissonModel.predict) so the chosen source keeps its
    xG provenance instead of being flattened to plain "features"
    (2026-08-17: the label lied while the λ was already xG-blended).

    This is the NO-PIN fallback (a caller-provided ``pinned_lambda_source``
    is resolved BEFORE this function by run_prediction_engine). Modes:

    ``mode="pinned"`` (P1.4, the PRODUCTION DEFAULT): the first evaluation
    of a match chooses its estimator here and the caller pins it; every
    later query reuses the pin (see run_prediction_engine). This fallback
    itself behaves exactly like "threshold" -- the pinning happens one
    layer up, at the caller, not inside this selection.

    ``mode="threshold"``: the pre-Fix-2 production branch (Elo replacement /
    blend / features-as-is), kept as a comparison mode for backtests.

    ``mode="blend"`` (Option 3, backtest-only): always blends the feature λ
    toward Elo with weight min(1, samples / target) -- never a hard
    estimator switch, never pure-Elo except at zero samples. The blended λ
    keeps the "features+elo" label so callers can see it ran the blend path.
    """
    if mode == "blend":
        target = max(1, LAMBDA_BLEND_TARGET)
        w = min(1.0, max(0.0, samples) / target)
        lh = w * lh_f + (1.0 - w) * lh_e
        la = w * la_f + (1.0 - w) * la_e
        lam_src = "features+elo"
    else:  # "pinned" (production default) and "threshold" share this branch
        lh, la, lam_src = lh_f, la_f, feature_label
        if min_samples > 0 and samples < min_samples:
            lh, la, lam_src = lh_e, la_e, "elo"
        elif min_samples > 0 and samples < shrinkage:
            t = (samples - min_samples) / max(1.0, float(shrinkage - min_samples))
            t = max(0.0, min(1.0, t))
            lh = t * lh_f + (1.0 - t) * lh_e
            la = t * la_f + (1.0 - t) * la_e
            lam_src = f"{feature_label}+elo"
    _, o15, o25, o35, btts = probs_from_matrix(poisson_matrix(lh, la, rho))
    return {
        "lambda_home": lh,
        "lambda_away": la,
        "lambda_source": lam_src,
        "over_1.5": o15,
        "over_2.5": o25,
        "over_3.5": o35,
        "btts_yes": btts,
    }


def run_prediction_engine(
    ctx: MatchContext,
    *,
    elo: EloModel,
    poisson: PoissonModel,
    ensemble: Ensemble,
    calibrator: Any = None,
    scorer: Any = None,
    pinned_lambda_source: str | None = None,
    pinned_features_available_at_pin: bool | None = None,
    lambda_mode: str = "pinned",
) -> PredictionResult | None:
    """Orchestrate models + calibration + scoring into a PredictionResult.

    Returns None when no model can produce a prediction (missing data).
    """
    ens = ensemble.predict(ctx, elo, poisson)
    if ens is None:
        return None
    p1x2 = ens["1x2"]

    # Totals/BTTS: prefer the feature-based Poisson lambdas; fall back to Elo.
    # Thin form windows (fewer finished matches than the model can trust)
    # must not let a single freak result dominate the lambdas, so the feature
    # λ is blended toward the Elo λ: below ``min_samples`` the feature λ is
    # replaced by Elo entirely (option 1); between ``min_samples`` and
    # ``shrinkage_samples`` the weight ramps linearly (option 2); at/above
    # ``shrinkage_samples`` the feature λ stands as-is (unchanged).
    pm = poisson.predict(ctx)
    lh_e, la_e = elo.expected_lambdas(ctx.home, ctx.away)
    # Fix 2 (lambda source pinning): once a match has been evaluated, its
    # FIRST lambda_source is pinned for every later pre-match query of the
    # same fixture. The pin overrides the ``lambda_samples < min_samples``
    # branch so repeated queries cannot flip between the "elo" and
    # "features" estimators on threshold noise (the 0.58/2.12 -> 1.04/1.15
    # incident). TRADE-OFF, intentional: pinning trades potential later
    # accuracy (features-based λ arriving after the pin) for consistency
    # across repeated queries of a signal tool -- do not "fix" this by
    # silently re-introducing dynamic switching.
    #
    # One-time exception (allowed, then locked): if the pin is "elo" ONLY
    # because features were genuinely UNAVAILABLE at pin time (no feature
    # Poisson output at all -- API error / no data returned), and features
    # become available later, the estimator may switch to features exactly
    # ONCE, logged as ``lambda_source_switch_reason:
    # features_unavailable_at_pin_time``. Threshold wobble (features existed
    # in degraded form, just below min_samples) NEVER triggers a switch.
    switch_reason: str | None = None
    lambda_samples: int | None = None
    elo_anchor_t = 0.0
    market_total_calibrated = False
    if pm is not None:
        lh, la = pm["lambda_home"], pm["lambda_away"]
        samples = int(pm.get("lambda_samples") or ctx.form_samples or 0)
        lambda_samples = samples
        lam_src = pm["lambda_source"]
        features_available_now = True
        pinned_features_family = bool(
            pinned_lambda_source and str(pinned_lambda_source).startswith("features")
        )
        if pinned_features_family:
            # Pin (2026-08-22 exact-composition fix): the pinned label must be
            # REPRODUCED, not merely family-matched. The family check used to
            # accept any "features*" composition, so a pin made on the blended
            # "features+xg+elo" estimator silently flipped to raw
            # "features+xg" lambdas on the next query (Everton v Crystal
            # Palace: totals prob jumped 0.63 -> 0.68 between runs, P0-4
            # contradiction fired). Rules:
            #   exact match              -> pm lambdas as-is;
            #   pinned "<base>+elo"      -> re-apply the SAME shrinkage ramp
            #                               toward Elo so the blend is
            #                               reproduced deterministically from
            #                               current samples; when the ramp
            #                               saturates (t >= 1) the lambda IS
            #                               the base estimator, so label it
            #                               honestly as the base;
            #   pinned "features", now
            #   "features+xg" available  -> richer same-family data arrived
            #                               late: switch ONCE, logged
            #                               "xg_unavailable_at_pin_time"
            #                               (mirror of the elo-pin exception);
            #   anything else            -> degrade to the current base.
            _base_now = str(pm["lambda_source"])
            if pinned_lambda_source == _base_now:
                lam_src = _base_now
            elif pinned_lambda_source == f"{_base_now}+elo":
                _min, _shr = poisson.min_samples, poisson.shrinkage
                if _min > 0 and samples < _min:
                    t = 0.0
                elif _shr > _min:
                    t = max(0.0, min(1.0, (samples - _min) / float(_shr - _min)))
                else:
                    t = 1.0
                lh = t * lh + (1.0 - t) * lh_e
                la = t * la + (1.0 - t) * la_e
                lam_src = _base_now if t >= 1.0 else str(pinned_lambda_source)
            elif pinned_lambda_source == "features" and _base_now == "features+xg":
                lam_src = _base_now
                switch_reason = "xg_unavailable_at_pin_time"
            else:
                lam_src = _base_now
        elif pinned_lambda_source == "elo" and not (pinned_features_available_at_pin is False):
            # Pin: keep the Elo estimator (pin was made while features
            # existed in degraded form -> no switch allowed).
            lh, la, lam_src = lh_e, la_e, "elo"
        elif pinned_lambda_source == "elo" and pinned_features_available_at_pin is False:
            # One-time exception: features were genuinely unavailable at pin
            # time and now exist -> switch to features exactly once.
            lam_src = pm["lambda_source"]
            switch_reason = "features_unavailable_at_pin_time"
        else:
            # No pin: the configured lambda selection mode. "threshold" is
            # the production behavior; "blend" (Option 3, backtest-only) is
            # the experimental always-blend that the A/B measures.
            _sel = lambda_from_mode(
                lh_e=lh_e, la_e=la_e, lh_f=lh, la_f=la,
                samples=samples,
                min_samples=poisson.min_samples,
                shrinkage=poisson.shrinkage,
                rho=poisson.rho,
                mode=lambda_mode,
                # Keep the estimator's own xG provenance ("features+xg") so
                # the label never lies about whether the λ was xG-blended.
                feature_label=pm.get("lambda_source") or "features",
            )
            lh, la, lam_src = _sel["lambda_home"], _sel["lambda_away"], _sel["lambda_source"]
        # ---- Plan v3 F1: Elo anchor on the FINAL selected λ (2026-08-24) --
        # Feature-only λ with a thin early-season window ignored team strength
        # entirely (Elche v Barcelona: λh > λa vs a 715-point Elo gap). The
        # anchor re-splits the SAME total by seeded-Elo share, ramping in
        # between min_gap/full_gap rating points. Deterministic per query;
        # pin labels untouched (audit lives in model_probs.elo_anchor_t).
        _anchor_cfg = getattr(poisson, "elo_anchor", None) or {}
        if _anchor_cfg.get("enabled", True) and elo.known(ctx.home, ctx.away):
            lh, la, elo_anchor_t = apply_elo_anchor(
                lh, la,
                float(elo.rating(ctx.home)), float(elo.rating(ctx.away)),
                min_gap=float(_anchor_cfg.get("min_gap", 150.0)),
                full_gap=float(_anchor_cfg.get("full_gap", 400.0)),
            )
        # ---- Plan v3 F2: market-total calibration on the FINAL λ ----------
        # The Ensemble's internal _calibrate_total_goals never reached the λ
        # that feeds Totals/BTTS (structural split, see F2 in
        # reports/bestpick_evaluasi_elche-barca_2026-08-24.md). Close HALF the
        # model-vs-fair-market gap here, ratio-preserving.
        _mtc_cfg = getattr(poisson, "market_total_calibration", None) or {}
        if _mtc_cfg.get("enabled", True):
            lh, la, market_total_calibrated = calibrate_total_to_market(
                lh, la, ctx.market_totals,
                weight=float(_mtc_cfg.get("weight", 0.5)),
            )
        lh = max(0.3, min(3.5, lh))
        la = max(0.3, min(3.5, la))
        # Recompute totals from the FINAL λ so over/btts stay consistent
        # with the λ that is reported (a blended λ must not carry the raw
        # feature totals).
        _, o15, o25, o35, btts = probs_from_matrix(poisson_matrix(lh, la, poisson.rho))
        totals = {"over_1.5": o15, "over_2.5": o25, "over_3.5": o35, "btts_yes": btts}
    else:
        # Features genuinely unavailable this query (no feature Poisson
        # output): Elo is the only estimator, pin or not.
        features_available_now = False
        lh, la = lh_e, la_e
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
        p1x2=p1x2,
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
        # Fix 2 audit trail: how much feature data existed when the λ was
        # computed, whether features were available at all, and -- when the
        # one-time pin exception fired -- why the estimator switched.
        "lambda_samples": lambda_samples,
        "features_available": features_available_now,
        "pinned_lambda_source": pinned_lambda_source,
        "lambda_source_switch_reason": switch_reason,
        "lambda_mode": lambda_mode,
        # Plan v3 audit trail: how much the Elo anchor blended (0 = off/no gap)
        # and whether the market-total calibration moved the final λ.
        "elo_anchor_t": round(elo_anchor_t, 4),
        "market_total_calibrated": market_total_calibrated,
        "models": ens["models"],
        "model_weights": ens["weights"],
        # TODO-13: model disagreement as an explicit uncertainty signal.
        "uncertainty": round(ens.get("spread", 0.0), 4),
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
        decisiveness=components.get("decisiveness", 0.0),
        model_version=MODEL_VERSION,
        as_of=utc_now_iso(),
        input_hash=ctx.input_hash,
        sources=ctx.sources,
    )
