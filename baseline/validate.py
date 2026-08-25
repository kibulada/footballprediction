"""Multi-season chronological validation of the prediction engine (EPL).

Strict walk-forward: all seasons are replayed in kickoff order in ONE pass.
Elo ratings and rolling form features carry across season boundaries and are
never updated with a match before that match was predicted. There is NO
pooling-then-random-split.

Models evaluated (no new models added -- existing engine only):
  - baseline : expanding empirical base rates (prior before any data)
  - elo      : rating-based (K-adaptive, home advantage)
  - poisson  : feature-based Poisson with rho=0 (plain)
  - dc       : feature-based Poisson with rho=-0.1 (Dixon-Coles) -- the
               production setting, independently evaluated as its own row
  - ensemble : production blend Elo + Dixon-Coles

Metrics per model per season: Log Loss, Brier, ECE, Hit Rate. Approximate
95% confidence intervals (normal approximation on per-match values) are
reported for Log Loss, Brier and Hit Rate; ECE is a point estimate.

No ROI: historical odds are unavailable (The Odds API free tier does not
expose them), so any ROI claim would be fabricated.

Usage::

    python -m agents.football.validate --leagues EPL \\
        --seasons 2022-2023,2023-2024,2024-2025,2025-2026 [--proxy ...]
    python -m agents.football.validate --fixtures fixtures.json  # offline
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from .backtest import BASE_RATE_PRIOR, load_fbref_fixtures, load_fixtures_from_json
from .calibration import Calibrator, expected_calibration_error
from .context import MatchContext
from .elo import EloModel
from .models import Ensemble, PoissonModel, poisson_matrix, probs_from_matrix
from .odds_history import LEAGUE_CODES, load_history_fixtures

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "football.json"

MODELS = ("baseline", "elo", "poisson", "dc", "ensemble")
# The market is not a model we control: it is the benchmark. It only gets
# evaluated when the dataset carries real historical odds.
MARKET = "market"


def _load_model_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("models", {})
        elo_cfg = {k: v for k, v in cfg.get("elo", {}).items() if k != "file"}
        return elo_cfg, cfg.get("poisson", {}), cfg.get("ensemble", {})
    except (OSError, json.JSONDecodeError):
        return {}, {}, {}


def _season_of(fx: dict[str, Any]) -> str:
    return fx.get("season") or "unknown"


def _ctx_for(
    fixture: dict[str, Any],
    forms: dict[str, deque],
    last_date: dict[str, str],
) -> MatchContext:
    home, away = fixture["home"], fixture["away"]

    def _stats(team: str) -> tuple[float | None, float | None, int]:
        dq = forms.get(team)
        if not dq:
            return None, None, 0
        gfs = [g[0] for g in dq]
        gas = [g[1] for g in dq]
        return sum(gfs) / len(gfs), sum(gas) / len(gas), len(dq)

    def _rest(team: str) -> int | None:
        """Days since the team's previous match (pre-match only)."""
        prev = last_date.get(team)
        if not prev or fixture["date"] <= prev:
            return None
        try:
            from datetime import date

            return (date.fromisoformat(fixture["date"]) - date.fromisoformat(prev)).days
        except ValueError:
            return None

    hgf, hga, hs = _stats(home)
    agf, aga, as_ = _stats(away)
    odds = (
        {
            "home": fixture["home_odds"],
            "draw": fixture["draw_odds"],
            "away": fixture["away_odds"],
        }
        if all(fixture.get(k) for k in ("home_odds", "draw_odds", "away_odds"))
        else None
    )
    return MatchContext(
        league=fixture.get("league", ""),
        home=home,
        away=away,
        kickoff_utc=fixture["date"],
        home_gf_avg=hgf,
        home_ga_avg=hga,
        away_gf_avg=agf,
        away_ga_avg=aga,
        # Raw scorelines enable time-decay weighting (Dixon-Coles xi) in the
        # Poisson model instead of equal-weight rolling averages.
        home_recent_goals=[tuple(g) for g in forms.get(home, ())] or None,
        away_recent_goals=[tuple(g) for g in forms.get(away, ())] or None,
        # Same shrinkage rule as backtest.py: min when both sides have form,
        # otherwise the larger sample.
        form_samples=min(hs, as_) if (hs and as_) else max(hs, as_),
        home_rest_days=_rest(home),
        away_rest_days=_rest(away),
        consensus_odds=odds,
    )


def _init_bucket() -> dict[str, Any]:
    return {"ll": [], "brier": [], "hits": 0, "n": 0, "cal_pairs": [],
            "bets": 0, "net": 0.0}


def _record(bucket: dict[str, Any], probs: dict[str, float], outcome: int) -> None:
    keys = ("home", "draw", "away")
    p_out = max(1e-9, probs[keys[outcome]])
    bucket["ll"].append(-math.log(p_out))
    bucket["brier"].append(
        sum((probs[k] - (1.0 if i == outcome else 0.0)) ** 2 for i, k in enumerate(keys))
    )
    if keys.index(max(probs, key=probs.get)) == outcome:
        bucket["hits"] += 1
    bucket["n"] += 1
    # Proper multiclass calibration pairs: (P(k), indicator k) pooled across
    # the three outcomes (3 pairs per match) -> ECE is a well-defined
    # binary-event calibration measure.
    for i, k in enumerate(keys):
        bucket["cal_pairs"].append((probs[k], 1.0 if outcome == i else 0.0))


def _market_probs(odds: dict[str, float] | None) -> dict[str, float] | None:
    """Margin-free implied probabilities from decimal odds. This is the fair
    benchmark: model probabilities are compared on the same scale."""
    if not odds:
        return None
    raw = {k: (1.0 / v if v and v > 1.0 else 0.0) for k, v in odds.items()}
    total = sum(raw.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in raw.items()}


def _record_roi(
    bucket: dict[str, Any],
    probs: dict[str, float],
    outcome: int,
    odds: dict[str, float] | None,
    edge_threshold: float,
) -> None:
    """Flat-stake ROI on the best 1X2 pick with margin-free edge >= threshold.
    Real historical odds only -- never fabricated."""
    if not odds:
        return
    keys = ("home", "draw", "away")
    raw = [1.0 / odds[k] if odds.get(k) and odds[k] > 1.0 else 0.0 for k in keys]
    total_raw = sum(raw)
    if total_raw <= 0:
        return
    norm = [r / total_raw for r in raw]
    best = max(probs, key=probs.get)
    idx = keys.index(best)
    edge = probs[best] - norm[idx]
    if edge >= edge_threshold:
        bucket["bets"] += 1
        bucket["net"] += odds[keys[idx]] - 1.0 if outcome == idx else -1.0


def run_multi_season_validation(
    fixtures: list[dict[str, Any]],
    *,
    elo_cfg: dict[str, Any] | None = None,
    poisson_cfg: dict[str, Any] | None = None,
    ensemble_cfg: dict[str, Any] | None = None,
    edge_threshold: float = 0.02,
    calibration_out: str | Path | None = None,
    seed_elo_path: str | Path | None = None,
) -> dict[str, Any]:
    """One chronological pass over all seasons; per-season + aggregate metrics.

    When fixtures carry historical odds (home_odds/draw_odds/away_odds), a
    MARKET baseline row (margin-free implied) is evaluated alongside the
    models and flat-stake ROI is computed for every model. Without odds, the
    market row stays empty and ROI stays "not reported" (honesty rule).

    When ``seed_elo_path`` is given, the final Elo ratings (the state after
    replaying every match) are persisted there so the live bot shares the
    exact ratings this walk-forward pass produced.
    """
    elo_cfg = dict(elo_cfg or {})
    poisson_cfg = dict(poisson_cfg or {})
    ensemble_cfg = dict(ensemble_cfg or {})

    elo = EloModel(**elo_cfg)
    poisson = PoissonModel(
        base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
        base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
        dc_rho=0.0,  # plain Poisson row
        shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
        time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
        xg_weight=poisson_cfg.get("xg_weight", 0.65),
        rest_days_k=poisson_cfg.get("rest_days_k", 0.0),
    )
    dc = PoissonModel(
        base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
        base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
        dc_rho=poisson_cfg.get("dc_rho", -0.1),  # Dixon-Coles (production)
        shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
        time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
        xg_weight=poisson_cfg.get("xg_weight", 0.65),
        rest_days_k=poisson_cfg.get("rest_days_k", 0.0),
    )
    ensemble = Ensemble(
        elo_weight=ensemble_cfg.get("elo_weight", 0.5),
        poisson_weight=ensemble_cfg.get("poisson_weight", 0.5),
    )

    forms: dict[str, deque] = {}
    last_date: dict[str, str] = {}
    base = {"home": 0, "draw": 0, "away": 0}
    base_n = 0
    seasons: "OrderedDict[str, dict[str, dict[str, Any]]]" = OrderedDict()
    aggregate: dict[str, dict[str, Any]] = {m: _init_bucket() for m in MODELS}
    aggregate[MARKET] = _init_bucket()

    for fixture in sorted(fixtures, key=lambda f: f["date"]):
        season = _season_of(fixture)
        if season not in seasons:
            seasons[season] = {m: _init_bucket() for m in MODELS}
            seasons[season][MARKET] = _init_bucket()
        hg, ag = fixture["home_goals"], fixture["away_goals"]
        outcome = 0 if hg > ag else (1 if hg == ag else 2)
        ctx = _ctx_for(fixture, forms, last_date)

        p_base = (
            {k: base[k] / base_n for k in ("home", "draw", "away")}
            if base_n
            else dict(BASE_RATE_PRIOR)
        )
        lh_e, la_e = elo.expected_lambdas(ctx.home, ctx.away)
        p_elo, _, _, _, _ = probs_from_matrix(poisson_matrix(lh_e, la_e, rho=0.0))
        pm_plain = poisson.predict(ctx)
        pm_dc = dc.predict(ctx)
        ens = ensemble.predict(ctx, elo, dc)
        p_market = _market_probs(ctx.consensus_odds)

        for name, probs in (
            ("baseline", p_base),
            ("elo", p_elo),
            ("poisson", pm_plain["1x2"] if pm_plain else None),
            ("dc", pm_dc["1x2"] if pm_dc else None),
            ("ensemble", ens["1x2"] if ens else None),
            (MARKET, p_market),
        ):
            if probs is None:
                continue
            _record(seasons[season][name], probs, outcome)
            _record(aggregate[name], probs, outcome)
            if name != MARKET:
                # ROI is for models only: betting the market on itself is a
                # zero-edge game by construction.
                _record_roi(
                    seasons[season][name], probs, outcome,
                    ctx.consensus_odds, edge_threshold,
                )
                _record_roi(
                    aggregate[name], probs, outcome,
                    ctx.consensus_odds, edge_threshold,
                )

        # Update state with the RESULT (strictly after prediction).
        elo.update(ctx.home, ctx.away, hg, ag, persist=False)
        forms.setdefault(ctx.home, deque(maxlen=5)).append((hg, ag))
        forms.setdefault(ctx.away, deque(maxlen=5)).append((ag, hg))
        last_date[ctx.home] = fixture["date"]
        last_date[ctx.away] = fixture["date"]
        base[("home", "draw", "away")[outcome]] += 1
        base_n += 1

    def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
        n = bucket["n"]
        if not n:
            return {"n": 0, "log_loss": None, "log_loss_ci": None,
                    "brier": None, "brier_ci": None, "ece": None,
                    "hit_rate": None, "hit_rate_ci": None,
                    "roi": None, "bets": 0}
        mean_ll, ci_ll = _mean_ci(bucket["ll"])
        mean_br, ci_br = _mean_ci(bucket["brier"])
        p_hit, ci_hit = _prop_ci(bucket["hits"], n)
        cal_probs = [p for p, _ in bucket["cal_pairs"]]
        cal_outs = [y for _, y in bucket["cal_pairs"]]
        ece = expected_calibration_error(cal_probs, cal_outs)
        return {
            "n": n,
            "log_loss": round(mean_ll, 4),
            "log_loss_ci": round(ci_ll, 4) if ci_ll is not None else None,
            "brier": round(mean_br, 4),
            "brier_ci": round(ci_br, 4) if ci_br is not None else None,
            "ece": round(ece, 4) if not math.isnan(ece) else None,
            "hit_rate": round(p_hit, 4),
            "hit_rate_ci": round(ci_hit, 4) if ci_hit is not None else None,
            "roi": round(bucket["net"] / bucket["bets"], 4) if bucket["bets"] else None,
            "bets": bucket["bets"],
        }

    season_metrics = {s: {m: _finish(b) for m, b in models.items()} for s, models in seasons.items()}
    aggregate_metrics = {m: _finish(b) for m, b in aggregate.items()}

    # Seed Elo: persist the final rating state (after ALL matches replayed),
    # exactly matching what the live bot loads. Nothing is saved when the
    # flag is absent.
    seeded_elo: dict[str, Any] = {}
    if seed_elo_path:
        elo.path = Path(seed_elo_path)
        elo._save()
        seeded_elo = elo.snapshot()

    # Calibration: raw ensemble ECE (honest) + in-sample fitted ECE (labeled).
    # When calibration_out is set, the fitted log-odds params are ALSO
    # persisted there so the live prediction path (Calibrator in analyse.py)
    # applies the same correction -- fit on the aggregate ensemble pairs.
    ens_bucket = aggregate["ensemble"]
    ens_probs = [p for p, _ in ens_bucket["cal_pairs"]]
    ens_outs = [y for _, y in ens_bucket["cal_pairs"]]
    calibrator = Calibrator(path=calibration_out, min_samples=1)
    if ens_bucket["n"] >= 20 and 0 < sum(ens_outs) < len(ens_outs):
        calibrator.fit(ens_probs, ens_outs)
    elif calibration_out:
        print(
            "note: --calibration-out given but ensemble fit skipped "
            f"(n={ens_bucket['n']}, degenerate outcomes?); nothing written"
        )
    calibration = {
        "ensemble_raw_ece": aggregate_metrics["ensemble"]["ece"],
        "ensemble_calibrated_ece": (
            round(expected_calibration_error([calibrator.apply(p) for p in ens_probs], ens_outs), 4)
            if calibrator.samples
            else None
        ),
        "calibrated_in_sample": bool(calibrator.samples),
        "samples": calibrator.samples,
        "a": calibrator.a,
        "b": calibrator.b,
        "file": str(calibration_out) if calibration_out else None,
    }

    # Consistency vs baseline AND vs market (log-loss) per model, per season --
    # objective support for any accuracy claim (requirement: consistent across
    # seasons, and meaningful only when the model beats the market).
    consistency: dict[str, dict[str, Any]] = {}
    for name in MODELS:
        wins = 0
        worse: list[str] = []
        mkt_wins = 0
        mkt_worse: list[str] = []
        for s, models in season_metrics.items():
            bl = models["baseline"]["log_loss"]
            mm = models[name]["log_loss"]
            if bl is not None and mm is not None:
                if mm < bl:
                    wins += 1
                else:
                    worse.append(s)
            mk = models.get(MARKET, {}).get("log_loss")
            if mk is not None and mm is not None:
                if mm < mk:
                    mkt_wins += 1
                else:
                    mkt_worse.append(s)
        consistency[name] = {
            "beats_baseline_seasons": wins,
            "of_seasons": len(season_metrics),
            "worse_seasons": worse,
            "beats_market_seasons": mkt_wins,
            "market_seasons": len(
                [s for s, m in season_metrics.items() if m.get(MARKET, {}).get("n")]
            ),
            "market_worse_seasons": mkt_worse,
        }

    missing_data: dict[str, dict[str, int]] = {}
    for s, models in season_metrics.items():
        missing_data[s] = {
            "baseline_n": models["baseline"]["n"],
            "poisson_skipped": models["baseline"]["n"] - models["poisson"]["n"],
            "dc_skipped": models["baseline"]["n"] - models["dc"]["n"],
        }

    return {
        "seasons": season_metrics,
        "aggregate": aggregate_metrics,
        "calibration": calibration,
        "consistency": consistency,
        "missing_data": missing_data,
        "n_matches_total": aggregate["baseline"]["n"],
        "models": list(MODELS),
        "roi_available": aggregate.get(MARKET, {}).get("n", 0) > 0,
        "seeded_elo": seeded_elo,
        "edge_threshold": edge_threshold,
        "note": (
            "One chronological walk-forward pass; Elo/form state carries across "
            "seasons; CIs are approximate 95% normal-approximation intervals on "
            "per-match values; ECE is computed on pooled per-outcome calibration "
            "pairs (3 per match); calibrated ECE is fitted in-sample (not "
            "out-of-sample); market row = margin-free implied probabilities from "
            "historical odds (only when the dataset provides them); ROI is "
            "flat-stake on the best 1X2 pick with margin-free edge >= threshold; "
            "matches are ordered by date only, so same-day matches may be "
            "processed in an arbitrary order (FBref provides no kickoff time) "
            "-- a known day-granularity caveat; beats-market compares model "
            "log-loss (all matches) against market log-loss (odds-carrying "
            "matches only; the subsets coincide under full odds coverage)."
        ),
    }


def _mean_ci(values: list[float]) -> tuple[float, float | None]:
    n = len(values)
    if n == 0:
        return float("nan"), None
    m = sum(values) / n
    if n < 2:
        return m, None
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return m, 1.96 * se


def _prop_ci(hits: int, n: int) -> tuple[float, float | None]:
    if n == 0:
        return float("nan"), None
    p = hits / n
    if n < 2 or p in (0.0, 1.0):
        return p, None
    se = math.sqrt(p * (1.0 - p) / n)
    return p, 1.96 * se


def _fmt_row(name: str, m: dict[str, Any]) -> str:
    if not m["n"]:
        return f"{name:<10}{0:>7}      n/a         n/a        n/a       n/a      n/a"
    ll = f"{m['log_loss']:.4f}"
    if m["log_loss_ci"]:
        ll += f"±{m['log_loss_ci']:.3f}"
    br = f"{m['brier']:.4f}"
    if m["brier_ci"]:
        br += f"±{m['brier_ci']:.3f}"
    hit = f"{m['hit_rate'] * 100:.1f}%"
    if m["hit_rate_ci"]:
        hit += f"±{m['hit_rate_ci'] * 100:.1f}"
    ece = "n/a" if m["ece"] is None else f"{m['ece']:.4f}"
    roi = "n/a" if m["roi"] is None else f"{m['roi'] * 100:.1f}%"
    return f"{name:<10}{m['n']:>7}{ll:>15}{br:>15}{ece:>12}{hit:>14}{roi:>9}"


def _table_header() -> str:
    return f"{'model':<10}{'n':>7}{'logloss':>15}{'brier':>15}{'ece':>12}{'hit%':>14}{'roi':>9}"


def _table_names(result: dict[str, Any]) -> list[str]:
    return list(result.get("models", [])) + [MARKET]


def format_validation_report(result: dict[str, Any]) -> str:
    header = _table_header()
    lines = [
        "=" * 84,
        "MULTI-SEASON VALIDATION (strict chronological walk-forward)",
        "=" * 84,
        result["note"],
        "",
    ]
    for season, models in result["seasons"].items():
        lines.append(f"--- Season {season} ---")
        lines.append(header)
        lines.append("-" * 84)
        for name in _table_names(result):
            lines.append(_fmt_row(name, models.get(name, {"n": 0})))
        lines.append("")
    lines.append("--- AGGREGATE (all seasons, chronological pool) ---")
    lines.append(header)
    lines.append("-" * 84)
    for name in _table_names(result):
        lines.append(_fmt_row(name, result["aggregate"][name]))
    lines.append("")
    cal = result["calibration"]
    lines.append("--- Calibration (ensemble) ---")
    lines.append(f"  raw ECE             : {cal['ensemble_raw_ece']}")
    lines.append(f"  calibrated ECE      : {cal['ensemble_calibrated_ece']}")
    lines.append(f"  fit is in-sample    : {cal['calibrated_in_sample']} "
                 f"(samples={cal['samples']})")
    if cal.get("a") is not None:
        lines.append(f"  fitted params       : a={cal['a']:.4f}, b={cal['b']:.4f}")
    if cal.get("file"):
        lines.append(f"  persisted to        : {cal['file']}")
    lines.append("")
    lines.append("--- Consistency (log-loss) ---")
    for name, c in result.get("consistency", {}).items():
        worse = ", ".join(c["worse_seasons"]) or "none"
        mkt_worse = ", ".join(c.get("market_worse_seasons", [])) or "none"
        mkt_of = c.get("market_seasons", 0)
        lines.append(
            f"  {name:<10}: beats baseline in {c['beats_baseline_seasons']}/{c['of_seasons']} seasons "
            f"(worse: {worse})"
        )
        if mkt_of:
            lines.append(
                f"  {'':10}  beats MARKET in {c['beats_market_seasons']}/{mkt_of} seasons "
                f"(worse: {mkt_worse})"
            )
    lines.append("")
    lines.append("--- Missing data ---")
    lines.append("  Poisson/DC skip matches lacking pre-match form features")
    lines.append("  (early-season / newly promoted teams).")
    for s, m in result.get("missing_data", {}).items():
        lines.append(
            f"  {s}: baseline={m['baseline_n']}, poisson_skipped={m['poisson_skipped']}, "
            f"dc_skipped={m['dc_skipped']}"
        )
    lines.append("")
    lines.append("CIs: approximate 95% normal-approximation on per-match values.")
    if result.get("roi_available"):
        lines.append(
            f"ROI: flat-stake 1X2 bets, best pick with margin-free edge >= "
            f"{result.get('edge_threshold', 0.02):.0%} (real historical odds)."
        )
    else:
        lines.append("ROI: not reported (historical odds unavailable).")
    lines.append("NOTE: lower log-loss/Brier/ECE is better; higher hit% is better.")
    lines.append("NOTE: ECE uses pooled per-outcome calibration pairs (3 per match).")
    return "\n".join(lines)


def run_cross_league_validation(
    fixtures: list[dict[str, Any]],
    *,
    elo_cfg: dict[str, Any] | None = None,
    poisson_cfg: dict[str, Any] | None = None,
    ensemble_cfg: dict[str, Any] | None = None,
    edge_threshold: float = 0.02,
) -> dict[str, Any]:
    """One independent chronological replay PER LEAGUE (own Elo, base rates,
    form). Returns per-league results plus a cross-league consistency summary."""
    by_league: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for fx in fixtures:
        by_league.setdefault(fx.get("league") or "unknown", []).append(fx)

    per_league: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for league, lfx in by_league.items():
        if not lfx:
            continue
        per_league[league] = run_multi_season_validation(
            lfx, elo_cfg=elo_cfg, poisson_cfg=poisson_cfg,
            ensemble_cfg=ensemble_cfg, edge_threshold=edge_threshold,
        )

    # Cross-league consistency: sum beats-baseline AND beats-market counts
    # across leagues.
    totals: dict[str, dict[str, Any]] = {}
    for name in MODELS:
        wins = 0
        of = 0
        won_leagues: list[str] = []
        lost_leagues: list[str] = []
        mkt_wins = 0
        mkt_of = 0
        mkt_won_leagues: list[str] = []
        mkt_lost_leagues: list[str] = []
        for league, res in per_league.items():
            c = res.get("consistency", {}).get(name, {})
            wins += c.get("beats_baseline_seasons", 0)
            of += c.get("of_seasons", 0)
            if c.get("worse_seasons"):
                lost_leagues.append(league)
            elif c.get("beats_baseline_seasons", 0) == c.get("of_seasons", 0):
                won_leagues.append(league)
            mkt_wins += c.get("beats_market_seasons", 0)
            mkt_of += c.get("market_seasons", 0)
            if c.get("market_worse_seasons"):
                mkt_lost_leagues.append(league)
            elif c.get("beats_market_seasons", 0) == c.get("market_seasons", 0) and c.get("market_seasons", 0):
                mkt_won_leagues.append(league)
        totals[name] = {
            "wins": wins,
            "of": of,
            "leagues_won": won_leagues,
            "leagues_lost": lost_leagues,
            "market_wins": mkt_wins,
            "market_of": mkt_of,
            "market_leagues_won": mkt_won_leagues,
            "market_leagues_lost": mkt_lost_leagues,
        }

    # Best/worst league by ensemble margin over baseline (aggregate log-loss).
    margins: dict[str, float] = {}
    for league, res in per_league.items():
        agg = res["aggregate"]
        bl = agg["baseline"]["log_loss"]
        en = agg["ensemble"]["log_loss"]
        margins[league] = round((bl - en) if (bl is not None and en is not None) else float("nan"), 4)
    best = max(margins, key=margins.get) if margins else None
    worst = min(margins, key=margins.get) if margins else None

    return {
        "per_league": per_league,
        "cross_summary": {
            "consistency": totals,
            "ensemble_ll_margin_by_league": margins,
            "best_league_by_margin": best,
            "worst_league_by_margin": worst,
        },
        "n_matches_total": sum(res["n_matches_total"] for res in per_league.values()),
        "models": list(MODELS),
        "roi_available": any(res.get("roi_available") for res in per_league.values()),
        "edge_threshold": edge_threshold,
        "note": (
            "Each league is replayed independently (own Elo, base rates, form) "
            "in strict chronological order across its seasons; teams never meet "
            "across leagues in this data, so per-league Elo is equivalent to a "
            "global Elo restricted to league matches. Market rows appear only "
            "for leagues whose data carries historical odds."
        ),
    }


def format_cross_league_report(result: dict[str, Any]) -> str:
    leagues = list(result["per_league"].keys())
    lines = [
        "=" * 84,
        "CROSS-LEAGUE VALIDATION",
        "=" * 84,
        result.get("note", ""),
        f"dataset : {result.get('dataset', '')}",
        f"leagues : {', '.join(leagues)}",
        f"matches : {result['n_matches_total']} across {len(leagues)} leagues",
        "",
    ]
    header = _table_header()
    for league in leagues:
        res = result["per_league"][league]
        lines.append(f"--- {league} (aggregate over its seasons) ---")
        lines.append(header)
        lines.append("-" * 84)
        for name in _table_names(result):
            lines.append(_fmt_row(name, res["aggregate"].get(name, {"n": 0})))
        lines.append("")

    lines.append("--- Consistency: model beats baseline on log-loss ---")
    cs = result["cross_summary"]["consistency"]
    lines.append(
        f"{'model':<10}" + "".join(f"{lg[:8]:>9}" for lg in leagues) + f"{'total':>8}"
    )
    lines.append("-" * 84)
    for name, t in cs.items():
        per = []
        for league in leagues:
            res = result["per_league"][league]
            c = res.get("consistency", {}).get(name, {})
            per.append(f"{c.get('beats_baseline_seasons', 0)}/{c.get('of_seasons', 0)}")
        lines.append(
            f"{name:<10}" + "".join(f"{p:>9}" for p in per) + f"{t['wins']}/{t['of']:>6}"
        )
    lines.append("")
    lines.append("--- Beats market (log-loss, seasons with historical odds) ---")
    for name, t in cs.items():
        if not t.get("market_of"):
            continue
        won = ", ".join(t.get("market_leagues_won", [])) or "none"
        lost = ", ".join(t.get("market_leagues_lost", [])) or "none"
        lines.append(
            f"  {name:<10}: {t['market_wins']}/{t['market_of']} season-league cells "
            f"(won: {won}; lost: {lost})"
        )
    lines.append("")
    lines.append("--- Ensemble margin over baseline (aggregate log-loss) ---")
    margins = result["cross_summary"]["ensemble_ll_margin_by_league"]
    for lg, m in margins.items():
        lines.append(f"  {lg:<14}: {m:+.4f}  (lower is better for log-loss)")
    lines.append(
        f"  best league for ensemble: {result['cross_summary']['best_league_by_margin']}, "
        f"worst: {result['cross_summary']['worst_league_by_margin']}"
    )
    lines.append("")
    lines.append("CIs: approximate 95% normal-approximation on per-match values.")
    if result.get("roi_available"):
        lines.append(
            f"ROI: flat-stake 1X2 bets, best pick with margin-free edge >= "
            f"{result.get('edge_threshold', 0.02):.0%} (real historical odds)."
        )
    else:
        lines.append("ROI: not reported (historical odds unavailable).")
    lines.append("NOTE: ECE uses pooled per-outcome calibration pairs (3 per match).")
    lines.append("NOTE: matches ordered by date only; same-day order is arbitrary.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-validate")
    parser.add_argument("--fixtures", default=None, help="local fixtures JSON (offline)")
    parser.add_argument("--leagues", default="EPL", help="comma-separated league keys")
    parser.add_argument("--seasons", default="2022-2023,2023-2024,2024-2025,2025-2026",
                        help="comma-separated FBref season codes")
    parser.add_argument(
        "--odds-source", default=None, choices=[None, "football-data"],
        help="load historical results + odds from football-data.co.uk (free "
             "CSV) instead of FBref; enables the market baseline and ROI.",
    )
    parser.add_argument("--edge-threshold", type=float, default=0.02,
                        help="minimum margin-free edge for ROI bets (default 0.02)")
    parser.add_argument("--proxy", default=None, help="SOCKS/HTTPS proxy for FBref")
    parser.add_argument("--out", default=None, help="write JSON report to this path")
    parser.add_argument(
        "--calibration-out", default=None,
        help="fit ensemble calibration on this run's aggregate pairs and save "
             "the params to this path (single-league mode only; typically "
             "cache/football/calibration.json)",
    )
    parser.add_argument(
        "--seed-elo",
        nargs="?",
        const=str(ROOT / "cache" / "football" / "elo.json"),
        default=None,
        help="write the walk-forward Elo ratings to this path after replay "
             "(bare --seed-elo uses cache/football/elo.json). Works in both "
             "single- and cross-league mode; in cross-league mode the ratings "
             "are the per-league replays merged into one file.",
    )
    parser.add_argument(
        "--elo-weight", type=float, default=None,
        help="override ensemble elo weight (experiment: walk-forward before/after)",
    )
    parser.add_argument(
        "--poisson-weight", type=float, default=None,
        help="override ensemble poisson weight (experiment)",
    )
    parser.add_argument(
        "--rest-days-k", type=float, default=None,
        help="experimental rest-day adjustment strength in the Poisson model "
             "(0 = off; verify via walk-forward before/after)",
    )
    args = parser.parse_args(argv)

    elo_cfg, poisson_cfg, ensemble_cfg = _load_model_config()
    if args.elo_weight is not None:
        ensemble_cfg["elo_weight"] = args.elo_weight
    if args.poisson_weight is not None:
        ensemble_cfg["poisson_weight"] = args.poisson_weight
    if args.rest_days_k is not None:
        poisson_cfg["rest_days_k"] = args.rest_days_k
    leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]

    if args.fixtures:
        fixtures = load_fixtures_from_json(args.fixtures)
        dataset = f"local file {args.fixtures}"
    elif args.odds_source == "football-data":
        bad = [lg for lg in leagues if lg not in LEAGUE_CODES]
        if bad:
            parser.error(f"football-data.co.uk has no data for: {', '.join(bad)}")
        fixtures = load_history_fixtures(leagues, seasons, proxy=args.proxy)
        dataset = (f"football-data.co.uk {args.leagues} {args.seasons} "
                   f"(real historical odds)")
    else:
        fixtures = load_fbref_fixtures(leagues, seasons, proxy=args.proxy)
        dataset = f"FBref {args.leagues} {args.seasons} (real data)"

    if args.seed_elo:
        seed_path = args.seed_elo
        if args.calibration_out:
            print("note: --calibration-out ignored in --seed-elo mode "
                  "(calibration is fitted by the normal validation run)")
        result = run_multi_season_validation(
            fixtures, elo_cfg=elo_cfg, poisson_cfg=poisson_cfg,
            ensemble_cfg=ensemble_cfg, edge_threshold=args.edge_threshold,
            seed_elo_path=seed_path,
        )
        ratings = result.get("seeded_elo", {}).get("ratings", {})
        print(
            f"Elo seeded to {seed_path}: {len(ratings)} teams, "
            f"{result['n_matches_total']} matches replayed"
        )
        return 0

    league_set = {fx.get("league") or "unknown" for fx in fixtures}
    if len(league_set) > 1:
        if args.calibration_out:
            print("note: --calibration-out ignored in cross-league mode "
                  "(fit per league via the single-league path)")
        result = run_cross_league_validation(
            fixtures, elo_cfg=elo_cfg, poisson_cfg=poisson_cfg,
            ensemble_cfg=ensemble_cfg, edge_threshold=args.edge_threshold,
        )
        result["dataset"] = dataset
        report = format_cross_league_report(result)
    else:
        result = run_multi_season_validation(
            fixtures, elo_cfg=elo_cfg, poisson_cfg=poisson_cfg,
            ensemble_cfg=ensemble_cfg, edge_threshold=args.edge_threshold,
            calibration_out=args.calibration_out,
        )
        result["dataset"] = dataset
        report = format_validation_report(result)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report written to {out}")

    cal = result.get("calibration", {})
    if args.calibration_out and cal.get("samples"):
        print(
            f"Calibration written to {args.calibration_out}: "
            f"a={cal['a']:.4f}, b={cal['b']:.4f}, samples={cal['samples']}, "
            f"calibrated_ece={cal.get('ensemble_calibrated_ece')}"
        )

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
