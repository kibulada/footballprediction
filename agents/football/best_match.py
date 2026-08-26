"""`!best <liga>` and `!bestgoalmatch` — pick the single best bet / goal match.

Both commands reuse the SAME validated building blocks as `analisa` and `top`:
  - fixtures:  stats.fetch_fixtures_for_date (football-data primary, cached)
  - odds:      odds.fetch_odds + extract_h2h_entries / extract_market_totals
  - form:      stats.fetch_team_form (cached per team)
  - engine:    build_engine_stack + build_match_context + run_prediction_engine
               + run_decision_engine (the exact stack `analisa` uses)

`best <liga>`: screen today + tomorrow-early (dini hari) fixtures of ONE
league, run the engine per match, rank by decision quality, return the
shortlist + the single best pick (winner payload is analyse-compatible so
`format_analyse` renders the full report).

`bestgoalmatch`: screen today's fixtures across leagues, score each match by
goal-friendliness (expected total goals + Poisson over-probabilities), rank
by expected goals, return the most goal-friendly match pick.

Anti-deadline: the engine per match is pure computation (<50ms). The only
network work is the same cached fixtures/odds/form `top` already performs.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

from .analyse import (
    _teams_match,
    build_engine_stack,
    extract_h2h_entries,
    extract_market_totals,
    run_decision_engine,
)
from .model_gates import TIER_ORDER, build_confidence_block, form_depth_shallow
from .context import build_match_context
from .league_resolver import resolve_league_scored
from .match_finder import _is_upcoming, _load_leagues
from .models import MAX_GOALS, run_prediction_engine
from .scorer import consensus_odds
from .timeutil import utc_now_iso, wib_today_iso

WIB = timezone(timedelta(hours=7))

_DECISION_PRIORITY = {
    "STRONG": 6,
    "GOOD": 5,
    "LEAN": 4,
    "NO BET": 3,
    "NO CLEAR DECISION": 1,
}


def _passes_best_gate(decision: dict[str, Any] | None) -> bool:
    """Gerbang `!best` (keputusan user 2026-08-23): hanya pick dengan
    confidence >= MEDIUM dan NON-veto (pick_status VALID di score_breakdown
    top) yang boleh jadi kandidat shortlist/winner.

    - ``pick_specific_confidence.label`` selalu terisi di production path
      (``run_decision_engine`` selalu mengaktifkan gating via ``bucket_n``);
      label hilang (MARKET PRIOR / thin-data path) diperlakukan LOW -> gugur.
    - ``pick_status`` != VALID berarti kandidat tertahan gerbang Section 2
      (INSUFFICIENT_DATA/SAMPLE, AUDIT_REQUIRED, REVIEW_REQUIRED, NO VALUE)
      -- ekuivalen "veto" di decision layer.
    """
    d = decision or {}
    tier = str(((d.get("pick_specific_confidence") or {}).get("label")) or "LOW")
    if TIER_ORDER.get(tier, -1) < TIER_ORDER["MEDIUM"]:
        return False
    top = ((d.get("score_breakdown") or {}).get("top") or {})
    return str(top.get("pick_status") or "") == "VALID"


def _season_now() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _league_meta(league_key: str) -> dict[str, Any] | None:
    return _load_leagues().get(league_key)


def _run_engine(
    *,
    home: str,
    away: str,
    kickoff: str | None,
    home_form: dict[str, Any] | None,
    away_form: dict[str, Any] | None,
    consensus: dict[str, float],
    totals: dict[str, dict[str, float]],
    has_odds: bool,
    bookmakers_count: int,
    display: str,
    league_key: str,
    cfg: dict[str, Any],
    stack: tuple[Any, Any, Any, Any, Any],
    ml_probs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run the validated prediction + decision engine for one match.

    Returns (prediction_dict, decision_dict). xG/H2H are intentionally not
    fetched here (they cost network calls); form attack/defense alone still
    drives Elo+Poisson — data quality honestly reports what is missing.
    """
    stats_in = {
        "home_form": (home_form or {}).get("sequence"),
        "away_form": (away_form or {}).get("sequence"),
        "home_gf_avg": (home_form or {}).get("gf_avg"),
        "home_ga_avg": (home_form or {}).get("ga_avg"),
        "away_gf_avg": (away_form or {}).get("gf_avg"),
        "away_ga_avg": (away_form or {}).get("ga_avg"),
        "home_recent_goals": (home_form or {}).get("recent_goals"),
        "away_recent_goals": (away_form or {}).get("recent_goals"),
    }
    ctx = build_match_context(
        league=display,
        home=home,
        away=away,
        kickoff=kickoff,
        stats=stats_in,
        odds={"has_odds": has_odds, "consensus": consensus, "totals": totals},
    )
    # Correction-spec Section 5: H2H & xG both absent -> confidence max MEDIUM.
    hard_cap_medium = bool(
        not ctx.has_xg and not (ctx.h2h and any(ctx.h2h.values()))
    )
    # P1 (form-depth floor): a form window shorter than 3 matches per team is
    # noise, not signal -> confidence max MEDIUM, STRONG banned.
    shallow_form = form_depth_shallow(home_form, away_form)
    elo, poisson, ensemble, calibrator, scorer = stack
    # P1.4: the `!best` path must apply the SAME lambda pin as `analisa` --
    # Fix 2 (lambda source pinning) is the production default here too, so a
    # match first evaluated by `!best` and later by `analisa` cannot flip
    # between estimators. The pin is read from the same append-only log via
    # the canonical match_id; the lambda_mode config key is backtest-only.
    _pin_src: str | None = None
    _pin_features_at_pin: bool | None = None
    try:
        _pl_cfg0 = cfg.get("prediction_log") or {}
        if _pl_cfg0.get("enabled") and _pl_cfg0.get("file"):
            from .prediction_log import last_prediction_snapshot, make_match_id
            _prev_row = last_prediction_snapshot(
                ROOT / _pl_cfg0["file"],
                make_match_id(league_key, home, away, kickoff),
            )
            _prev_feat = (_prev_row or {}).get("features") or {}
            _pin_src = _prev_feat.get("pinned_lambda_source")
            if _pin_src:
                _pin_features_at_pin = _prev_feat.get("pinned_features_available_at_pin")
    except Exception as exc:
        logger.warning("best: lambda pin lookup failed (prediction unaffected): %s", exc)
    _lam_mode = str((cfg.get("models") or {}).get("poisson", {}).get("lambda_mode", "pinned"))
    prediction = run_prediction_engine(
        ctx,
        elo=elo,
        poisson=poisson,
        ensemble=ensemble,
        calibrator=calibrator,
        scorer=scorer,
        pinned_lambda_source=_pin_src,
        pinned_features_available_at_pin=_pin_features_at_pin,
        lambda_mode=_lam_mode,
    )
    prediction_dict = prediction.to_dict() if prediction is not None else None
    from .calibration import league_calibrator as _league_calibrator
    # Per-league calibration is keyed by the CANONICAL league key (same as
    # analyse.py) -- ``display`` ("La Liga") would produce a different slug
    # than the registered key ("LaLiga") and miss the fit file entirely.
    _league_calibrated = _league_calibrator(league_key, cfg, ROOT) is not None
    decision = run_decision_engine(
        prediction_dict, consensus, totals, has_odds, bookmakers_count, cfg,
        hard_cap_medium=hard_cap_medium,
        form_depth_shallow=shallow_form,
        ml_probs=ml_probs,
        league=display,
        league_calibrated=_league_calibrated,
    )
    return prediction_dict, decision


def _winner_payload(
    *,
    league_key: str,
    display: str,
    home: str,
    away: str,
    kickoff: str | None,
    home_form: dict[str, Any] | None,
    away_form: dict[str, Any] | None,
    consensus: dict[str, float],
    totals: dict[str, dict[str, float]],
    has_odds: bool,
    bookmakers_count: int,
    signal: int,
    prediction: dict[str, Any] | None,
    decision: dict[str, Any],
    sources: list[str],
    quota: dict[str, Any],
) -> dict[str, Any]:
    """Build an analyse-compatible payload for the winner: `format_compact`
    renders the compact card on the main reply, and `format_analyse` renders
    the full report (model 1X2, totals, FINAL DECISION, data quality) that
    the runner ships as ``render_full`` for the 📋 Copy button."""
    return {
        "league": display,
        "league_key": league_key,
        "generated_at": utc_now_iso(),
        "prediction": prediction,
        "home": home,
        "away": away,
        "kickoff": kickoff,
        "venue": None,
        "match_found": True,
        "fixture_source": "best_match",
        "stats": {
            "home_form": (home_form or {}).get("sequence", "n/a"),
            "away_form": (away_form or {}).get("sequence", "n/a"),
            "home_gf_avg": (home_form or {}).get("gf_avg", 0),
            "home_ga_avg": (home_form or {}).get("ga_avg", 0),
            "away_gf_avg": (away_form or {}).get("gf_avg", 0),
            "away_ga_avg": (away_form or {}).get("ga_avg", 0),
            "home_split": (home_form or {}).get("home", {}),
            "away_split": (away_form or {}).get("away", {}),
            "h2h": {"wins": 0, "draws": 0, "losses": 0},
        },
        "odds": {
            "consensus": consensus,
            "best": {},
            "outlier": None,
            "bookmakers_count": bookmakers_count,
            "has_odds": has_odds,
            "totals": totals,
        },
        "signal": signal,
        "picks": {"top_picks": [], "best_pick": None, "model_probs": {}},
        "sources": sources,
        "similar_signal": None,
        "decision": decision,
        "confidence": build_confidence_block(decision),
        "quota": quota,
    }


async def find_best_matches(
    *,
    league_query: str,
    cfg: dict[str, Any],
    odds: Any,
    stats: Any,
    cache: Any,
    date: str | None = None,
    nowgoal: Any = None,
) -> dict[str, Any]:
    """`!best <liga>`: today (+ tomorrow early) fixtures of one league, ranked
    by the decision engine; the top candidate is the single best pick."""
    resolved = resolve_league_scored(league_query)
    if not resolved:
        return {"error": f"liga '{league_query}' tidak dikenal"}
    league_key, meta = resolved
    display = meta["display"]
    odds_key = meta.get("odds_api_key")
    meta_with_season = {**meta, "season": _season_now(), "_league_key": league_key}

    if date:
        dates = [date]
    else:
        today = datetime.now(WIB).date().isoformat()
        tomorrow = (datetime.now(WIB) + timedelta(days=1)).date().isoformat()
        dates = [today, tomorrow]

    # Collect upcoming fixtures across the requested dates (dedupe). Cached per
    # league+date exactly like find_top_matches: football-data throttles at 6s
    # per data-bearing call, so repeat runs must not re-hit the API.
    #
    # NowGoal schedule fallback (same as top): when football-data has nothing
    # (10 req/min free quota 429s past the 10th call), take the nowgoal
    # schedule once and filter per league so `!best` still finds its match.
    from .match_finder import _nowgoal_fixtures_for_league

    fixtures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    ttl = cfg.get("cache_ttl_seconds") or {}
    for d in dates:
        # P2: provider-aware fixture caches (football-data primary, nowgoal
        # fallback stored separately) so a nowgoal-filled day cannot later be
        # served as football-data within the TTL window.
        fd_cache_key = f"fixtures_football_data_{league_key}_{d}"
        ng_cache_key = f"fixtures_nowgoal_{league_key}_{d}"
        day_fixtures = cache.get(fd_cache_key, ttl.get("fixtures", 21600)) if cache else None
        if day_fixtures is None:
            try:
                day_fixtures = await stats.fetch_fixtures_for_date(meta_with_season, d) or []
                if day_fixtures:
                    if cache:
                        cache.set(fd_cache_key, day_fixtures)
                elif nowgoal is not None:
                    day_fixtures = cache.get(ng_cache_key, ttl.get("fixtures", 21600)) if cache else None
                    if day_fixtures is None:
                        day_fixtures = await _nowgoal_fixtures_for_league(
                            nowgoal, d, league_key, meta
                        )
                        if cache and day_fixtures:
                            cache.set(ng_cache_key, day_fixtures)
            except Exception as exc:  # noqa: BLE001 - never break the flow
                logger.warning("best: fixtures fetch failed for %s %s: %s", league_key, d, exc)
                day_fixtures = []
        for fix in day_fixtures:
            if not _is_upcoming(fix.get("status"), fix.get("date")):
                continue
            home = (fix.get("home") or {}).get("name")
            away = (fix.get("away") or {}).get("name")
            if not home or not away:
                continue
            key = (home, away, str(fix.get("date") or d))
            if key in seen:
                continue
            seen.add(key)
            fixtures.append(fix)

    if not fixtures:
        return {
            "error": f"Tidak ada match belum-bertanding untuk {display} "
                     f"({', '.join(dates)}).",
            "league": display,
            "league_key": league_key,
        }

    # Odds payload once, cached per league+date like find_top_matches.
    odds_payload: list[dict[str, Any]] = []
    if odds_key:
        odds_cache_key = f"odds_{odds_key}_{dates[0]}"
        odds_payload = cache.get(odds_cache_key, ttl.get("odds", 900)) if cache else None
        if odds_payload is None:
            try:
                odds_payload = await odds.fetch_odds(odds_key) or []
                if cache:
                    cache.set(odds_cache_key, odds_payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("best: odds fetch failed for %s: %s", league_key, exc)
                odds_payload = []

    stack = build_engine_stack(cfg)
    # ML (trained-model) probabilities, computed once for the whole batch:
    # per (league, date) feature frames are shared, so a full matchday is a
    # few seconds instead of per-match rebuilds. Absent models -> empty map
    # and the engine runs exactly as before (ML is an additive signal).
    ml_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        from .ml_predict import MlPredictor

        ml_cfg = cfg.get("models", {}).get("ml", {})
        ml_predictor = MlPredictor(
            ROOT / ml_cfg.get("models_dir", "cache/football/models"),
            window=int(ml_cfg.get("window", 5)),
            gd_margin=int(ml_cfg.get("gd_margin", 2)),
        )
        ml_entries = [
            (league_key, fix["home"]["name"], fix["away"]["name"], str(fix.get("date") or "")[:10])
            for fix in fixtures
        ]
        ml_map = ml_predictor.predict_matches(ml_entries)
    except Exception as exc:  # noqa: BLE001 -- ML must never break the pick flow
        logger.warning("best: ml predictor unavailable: %s", exc)

    candidates: list[dict[str, Any]] = []
    # Hard time budget like find_best_goal_matches: football-data throttles ~6s
    # per data-bearing call, so a cold-cache run over many fixtures must never
    # blow the runner deadline. Stop ranking once ~55s have elapsed and return
    # whatever was already ranked.
    import time as _time

    t0 = _time.monotonic()
    for fix in fixtures:
        if _time.monotonic() - t0 > 55.0:
            logger.warning("best: time budget hit, using %d ranked matches", len(candidates))
            break
        home = fix["home"]["name"]
        away = fix["away"]["name"]
        kickoff = fix.get("date") or ""
        home_id = fix["home"]["id"]
        away_id = fix["away"]["id"]

        match_odds: list[dict[str, Any]] = []
        totals: dict[str, dict[str, float]] = {}
        _primary_bm = (
            str(
                ((cfg.get("models", {}) or {}).get("decision", {}) or {}).get("primary_bookmaker")
                or ""
            ).strip()
            or None
        )
        for m in odds_payload:
            if _teams_match(m.get("home_team", ""), home) and _teams_match(
                m.get("away_team", ""), away
            ):
                match_odds = extract_h2h_entries(
                    m, m.get("home_team", ""), m.get("away_team", "")
                )
                totals = extract_market_totals(m, prefer_bookmaker=_primary_bm)
                break
        has_odds = bool(match_odds)
        consensus = (
            consensus_odds(match_odds, primary_bookmaker=_primary_bm)
            if match_odds else {"home": 0, "draw": 0, "away": 0}
        )
        bookmakers_count = len(match_odds)

        try:
            home_form, away_form = await asyncio.gather(
                stats.fetch_team_form(home_id, meta_with_season),
                stats.fetch_team_form(away_id, meta_with_season),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("best: form fetch failed %s vs %s: %s", home, away, exc)
            home_form = away_form = None

        prediction, decision = _run_engine(
            home=home, away=away, kickoff=kickoff,
            home_form=home_form, away_form=away_form,
            consensus=consensus, totals=totals,
            has_odds=has_odds, bookmakers_count=bookmakers_count,
            display=display, league_key=league_key, cfg=cfg, stack=stack,
            ml_probs=(
                (ml_map.get((league_key, home, away, kickoff[:10])) or {}).get("1x2")
                or None
            ),
        )
        d_type = (decision or {}).get("decision_type", "NO CLEAR DECISION")
        d_score = ((decision or {}).get("score_breakdown") or {}).get("top", {}).get("score", 0.0)

        candidates.append({
            "league_key": league_key,
            "display": display,
            "home": home,
            "away": away,
            "kickoff": kickoff,
            "has_odds": has_odds,
            "consensus": consensus,
            "bookmakers_count": bookmakers_count,
            "signal": int((prediction or {}).get("signal_strength") or 0),
            "decision_type": d_type,
            "decision_score": float(d_score),
            "prediction": prediction,
            "decision": decision,
            "home_form": home_form,
            "away_form": away_form,
            "totals": totals,
        })

    # Gerbang `!best` (keputusan 2026-08-23): hanya pick conf >= MEDIUM dan
    # non-veto yang boleh muncul di shortlist/winner. Flag-gated via
    # models.decision.best_gate_enabled (default ON = keputusan produk);
    # tests mekanika-ranking mematikan gate lewat cfg eksplisit.
    _best_gate_cfg = ((cfg.get("models") or {}).get("decision") or {})
    _gate_enabled = bool(_best_gate_cfg.get("best_gate_enabled", True))
    n_analyzed = len(candidates)
    if _gate_enabled and n_analyzed:
        candidates = [c for c in candidates if _passes_best_gate(c.get("decision"))]
        if not candidates:
            logger.info(
                "best: gate menaring semua kandidat (%d dianalisa) untuk %s",
                n_analyzed, league_key,
            )
            return {
                "error": (
                    f"Tidak ada pick yang lolos gerbang untuk {display}: "
                    f"{n_analyzed} match dianalisa, tidak ada yang memenuhi "
                    "syarat confidence ≥ MEDIUM + tanpa veto."
                ),
                "league": display,
                "league_key": league_key,
                "date": dates[0],
                "generated_at": utc_now_iso(),
                "candidates": [],
                "winner": None,
            }

    def _rank_key(c: dict[str, Any]) -> tuple:
        return (
            _DECISION_PRIORITY.get(c["decision_type"], 0),
            c["decision_score"],
            c["signal"],
        )

    candidates.sort(key=_rank_key, reverse=True)

    shortlist = [
        {
            "home": c["home"],
            "away": c["away"],
            "kickoff": c["kickoff"],
            "signal": c["signal"],
            "decision_type": c["decision_type"],
            "decision_score": c["decision_score"],
            "has_odds": c["has_odds"],
            "bookmakers_count": c["bookmakers_count"],
            # Transparansi gerbang: tier + status gerbang kandidat.
            "confidence_tier": str(
                ((c.get("decision") or {}).get("pick_specific_confidence") or {}).get("label")
                or "LOW"
            ),
            "pick_status": str(
                (((c.get("decision") or {}).get("score_breakdown") or {}).get("top") or {}).get(
                    "pick_status"
                )
                or "VALID"
            ),
        }
        for c in candidates
    ]

    winner = candidates[0] if candidates else None
    sources = ["best_match", "football_data", "theoddsapi" if odds_payload else "no-odds"]
    quota = {
        "odds_api_remaining": getattr(odds, "last_remaining", None),
        "odds_blocked": getattr(odds, "quota_blocked", False),
        "football_data_warning": getattr(stats.fd, "rate_limit_warning", False),
    }
    winner_payload = None
    if winner:
        winner_payload = _winner_payload(
            league_key=league_key, display=display,
            home=winner["home"], away=winner["away"], kickoff=winner["kickoff"],
            home_form=winner["home_form"], away_form=winner["away_form"],
            consensus=winner["consensus"], totals=winner["totals"],
            has_odds=winner["has_odds"], bookmakers_count=winner["bookmakers_count"],
            signal=winner["signal"], prediction=winner["prediction"],
            decision=winner["decision"], sources=sources, quota=quota,
        )

    return {
        "league": display,
        "league_key": league_key,
        "date": dates[0],
        "generated_at": utc_now_iso(),
        "candidates": shortlist,
        "winner": winner_payload,
        "quota": quota,
    }


def _expected_from_over25(p_over25: float) -> float:
    """Invert P(Over 2.5) into an expected total goals figure.

    Binary search over the symmetric-Poisson total T (lh=la=T/2) such that
    P(goals > 2.5) matches the market-implied probability. Lets a market-only
    profile produce an expected_total on the SAME scale as the form-based
    profile, so the ranking stays consistent when form data is missing.
    """
    from .models import poisson_matrix, probs_from_matrix

    p = max(0.05, min(0.95, p_over25))
    lo, hi = 1.0, 5.5
    for _ in range(24):
        t = 0.5 * (lo + hi)
        # probs_from_matrix -> (1x2, over1.5, over2.5, over3.5, btts)
        _, _, o25, _, _ = probs_from_matrix(poisson_matrix(t / 2.0, t / 2.0, rho=0.0))
        if o25 > p:
            hi = t
        else:
            lo = t
    return round(0.5 * (lo + hi), 2)


def _goal_profile(
    home_form: dict[str, Any] | None,
    away_form: dict[str, Any] | None,
    totals: dict[str, dict[str, float]],
    ml_over: float | None = None,
) -> dict[str, Any] | None:
    """Goal-friendliness of a match.

    Primary signal: team attack/defense + recent scores (expected_total =
    (home_gf + away_ga)/2 + (away_gf + home_ga)/2, Poisson over-probs from
    those lambdas). When form is missing, the trained ML Over/Under model
    (``ml_over`` = P(Over 2.5), calibrated) takes over; when that is also
    missing, fall back to the MARKET totals odds. ``source`` tells the user
    which signal drove the profile.
    """
    from .models import poisson_matrix, probs_from_matrix

    def _market(label: str) -> float | None:
        o = totals.get(label, {}).get("odds", 0)
        return float(o) if o and o > 0 else None

    odds_over_25 = _market("Over 2.5")
    odds_over_35 = _market("Over 3.5")
    odds_over_45 = _market("Over 4.5")

    hgf = (home_form or {}).get("gf_avg")
    hga = (home_form or {}).get("ga_avg")
    agf = (away_form or {}).get("gf_avg")
    aga = (away_form or {}).get("ga_avg")
    has_form = all(isinstance(v, (int, float)) for v in (hgf, hga, agf, aga))

    if has_form:
        lh = max(0.2, (float(hgf) + float(aga)) / 2.0)
        la = max(0.2, (float(agf) + float(hga)) / 2.0)
        expected_total = lh + la
        # probs_from_matrix -> (1x2, over1.5, over2.5, over3.5, btts)
        _, _, o25, o35, _ = probs_from_matrix(poisson_matrix(lh, la, rho=0.0))
        m = poisson_matrix(lh, la, rho=0.0)
        p_over45 = 1.0 - sum(
            m[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h + a <= 4
        )
        source = "form"
    elif ml_over is not None and 0.0 < ml_over < 1.0:
        # Trained-model fallback (only reached when form is missing): the
        # calibrated ML Over/Under 2.5 probability is inverted into an
        # expected total on the same scale as the form profile.
        expected_total = _expected_from_over25(ml_over)
        _, _, o25, o35, _ = probs_from_matrix(
            poisson_matrix(expected_total / 2.0, expected_total / 2.0, rho=0.0)
        )
        m = poisson_matrix(expected_total / 2.0, expected_total / 2.0, rho=0.0)
        p_over45 = 1.0 - sum(
            m[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h + a <= 4
        )
        source = "ml"
    elif odds_over_25:
        # Market-only fallback: fair implied from the Over/Under 2.5 pair.
        # (The Odds API usually exposes only the 2.5 line — Over 3.5 above is
        # optional and used just for display; it must not gate the fallback.)
        u25 = totals.get("Under 2.5", {}).get("odds", 0)
        if not (u25 and u25 > 0):
            return None
        ia, ib = 1.0 / odds_over_25, 1.0 / u25
        p25 = ia / (ia + ib) if (ia + ib) > 0 else 0.5
        expected_total = _expected_from_over25(p25)
        # probs_from_matrix -> (1x2, over1.5, over2.5, over3.5, btts)
        _, _, o25, o35, _ = probs_from_matrix(
            poisson_matrix(expected_total / 2.0, expected_total / 2.0, rho=0.0)
        )
        m = poisson_matrix(expected_total / 2.0, expected_total / 2.0, rho=0.0)
        p_over45 = 1.0 - sum(
            m[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h + a <= 4
        )
        source = "market"
    else:
        return None

    return {
        "expected_total": round(expected_total, 2),
        "over_2_5": round(o25, 4),
        "over_3_5": round(o35, 4),
        "over_4_5": round(p_over45, 4),
        "odds_over_2_5": odds_over_25,
        "odds_over_3_5": odds_over_35,
        "odds_over_4_5": odds_over_45,
        "source": source,
    }


async def find_best_goal_matches(
    *,
    cfg: dict[str, Any],
    odds: Any,
    stats: Any,
    cache: Any,
    league_query: str | None = None,
    date: str | None = None,
    nowgoal: Any = None,
    oddspapi: Any = None,
) -> dict[str, Any]:
    """`!bestgoalmatch`: today's fixtures across leagues ranked by expected
    total goals; the top match is the most goal-friendly pick (banjir gol)."""
    # Window like `!best` (today + tomorrow early WIB) — user expectation
    # "hari ini dan dini hari" (00-06 WIB besok) harus tercakup. Dulu hanya
    # 1 tanggal WIB sehingga UEL/UECL yang kickoff di UTC 2026-08-27 sore
    # (WIB 27 malam / 28 dini hari) terlewat. Samakan dengan find_best_matches.
    if date:
        dates = [date]
        today = date
    else:
        today = wib_today_iso()
        tomorrow = (datetime.now(WIB) + timedelta(days=1)).date().isoformat()
        dates = [today, tomorrow]
    leagues_cfg = _load_leagues()

    if league_query:
        resolved = resolve_league_scored(league_query)
        if not resolved:
            return {"error": f"liga '{league_query}' tidak dikenal"}
        league_keys = [resolved[0]]
    else:
        league_keys = list(cfg.get("leagues") or leagues_cfg.keys())

    # League-level goal profile: average expected total per league today.
    # Phase 1 collects fixtures/forms/odds; phase 2 (after the loops) computes
    # the ML batch and the goal profiles so the ML O/U model is one call.
    pending: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    ttl = cfg.get("cache_ttl_seconds") or {}

    # Hard time budget: scanning every league's fixtures + per-team form can
    # exceed the 85s runner deadline on a cold cache (football-data throttles
    # 6s per data-bearing call). Stop scanning once ~55s have elapsed and
    # return whatever goal-friendly matches were already found — never time
    # out with nothing.
    import time as _time
    t0 = _time.monotonic()

    for league_key in league_keys:
        if _time.monotonic() - t0 > 55.0:
            logger.warning("bestgoalmatch: time budget hit, using %d matches", len(pending))
            break
        meta = leagues_cfg.get(league_key)
        if not meta:
            continue
        display = meta["display"]
        meta_with_season = {**meta, "season": _season_now(), "_league_key": league_key}
        # P2: provider-aware fixture caches (football-data primary, nowgoal
        # fallback stored separately) — same 2-date window as find_best_matches
        # sehingga dini hari ter-cover.
        fixtures: list[dict[str, Any]] = []
        seen_dates_for_league: set[str] = set()
        for d in dates:
            if _time.monotonic() - t0 > 55.0:
                break
            fd_cache_key = f"fixtures_football_data_{league_key}_{d}"
            ng_cache_key = f"fixtures_nowgoal_{league_key}_{d}"
            day_fixtures = cache.get(fd_cache_key, ttl.get("fixtures", 21600)) if cache else None
            if day_fixtures is None:
                try:
                    day_fixtures = await stats.fetch_fixtures_for_date(meta_with_season, d) or []
                    if day_fixtures:
                        if cache:
                            cache.set(fd_cache_key, day_fixtures)
                    else:
                        if nowgoal is not None:
                            from .match_finder import _nowgoal_fixtures_for_league

                            day_fixtures = cache.get(ng_cache_key, ttl.get("fixtures", 21600)) if cache else None
                            if day_fixtures is None:
                                day_fixtures = await _nowgoal_fixtures_for_league(
                                    nowgoal, d, league_key, meta
                                )
                                if cache and day_fixtures:
                                    cache.set(ng_cache_key, day_fixtures)
                            day_fixtures = day_fixtures or []
                        else:
                            day_fixtures = []
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bestgoalmatch: fixtures failed %s %s: %s", league_key, d, exc)
                    day_fixtures = []
                    continue
            # dedupe per (league, date) already via cache; merge across dates
            if day_fixtures:
                # avoid double-adding same date twice if loop re-enters
                if d not in seen_dates_for_league:
                    fixtures.extend(day_fixtures)
                    seen_dates_for_league.add(d)
        if not fixtures:
            continue

        # Odds for THIS league (cached per league+date) — needed so the goal
        # profile can fall back to the market Over/Under lines when form data
        # is missing. Cukup sekali per league (payload TheOddsAPI mencakup semua
        # commence_time mendatang); cache key pakai start date agar reuse.
        odds_key = meta.get("odds_api_key")
        odds_payload: list[dict[str, Any]] = []
        if odds_key:
            odds_cache_key = f"odds_{odds_key}_{dates[0]}"
            odds_payload = cache.get(odds_cache_key, ttl.get("odds", 900)) if cache else None
            if odds_payload is None:
                try:
                    odds_payload = await odds.fetch_odds(odds_key) or []
                    if cache:
                        cache.set(odds_cache_key, odds_payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bestgoalmatch: odds failed %s: %s", league_key, exc)
                    odds_payload = []

        for fix in fixtures:
            if _time.monotonic() - t0 > 55.0:
                break
            if not _is_upcoming(fix.get("status"), fix.get("date")):
                continue
            home = (fix.get("home") or {}).get("name")
            away = (fix.get("away") or {}).get("name")
            if not home or not away:
                continue
            kickoff = fix.get("date") or ""
            key = (home, away, str(kickoff)[:10])
            if key in seen:
                continue
            seen.add(key)

            try:
                home_form, away_form = await asyncio.gather(
                    stats.fetch_team_form(fix["home"]["id"], meta_with_season),
                    stats.fetch_team_form(fix["away"]["id"], meta_with_season),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("bestgoalmatch: form failed %s vs %s: %s", home, away, exc)
                home_form = away_form = None

            totals: dict[str, dict[str, float]] = {}
            bk_count = 0
            for m in odds_payload:
                if _teams_match(m.get("home_team", ""), home) and _teams_match(
                    m.get("away_team", ""), away
                ):
                    totals = extract_market_totals(m)
                    bk_count = len(
                        extract_h2h_entries(m, m.get("home_team", ""), m.get("away_team", ""))
                    )
                    break

            # Fase A: NowGoal market fallback untuk cup/ liga odds_api_key null
            # atau TheOddsAPI kosong. Tanpa ini Japan Emperor's Cup dkk
            # selalu error "no form/odds" karena form tim cup tipis.
            if not totals.get("Over 2.5") and nowgoal is not None:
                try:
                    if _time.monotonic() - t0 <= 55.0:
                        ng_date = kickoff[:10] if kickoff else None
                        ng_payload = await nowgoal.match_odds(home, away, ng_date)
                        if ng_payload:
                            ng_totals = extract_market_totals(ng_payload)
                            if ng_totals.get("Over 2.5"):
                                totals = ng_totals
                                bk_count = len(
                                    extract_h2h_entries(
                                        ng_payload,
                                        ng_payload.get("home_team") or "",
                                        ng_payload.get("away_team") or "",
                                    )
                                )
                                logger.info("bestgoalmatch: nowgoal odds fallback %s vs %s", home, away)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bestgoalmatch: nowgoal odds fallback failed %s vs %s: %s", home, away, exc)

            # Fase A2: OddsPapi fallback kalau NowGoal juga kosong
            if not totals.get("Over 2.5") and oddspapi is not None:
                try:
                    if _time.monotonic() - t0 <= 55.0:
                        op_payload = await oddspapi.match_odds(home, away, kickoff)
                        if op_payload:
                            op_totals = extract_market_totals(op_payload)
                            if op_totals.get("Over 2.5"):
                                totals = op_totals
                                bk_count = len(
                                    extract_h2h_entries(
                                        op_payload,
                                        op_payload.get("home_team") or "",
                                        op_payload.get("away_team") or "",
                                    )
                                )
                                logger.info("bestgoalmatch: oddspapi odds fallback %s vs %s", home, away)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("bestgoalmatch: oddspapi odds fallback failed %s vs %s: %s", home, away, exc)

            pending.append({
                "league_key": league_key,
                "display": display,
                "home": home,
                "away": away,
                "kickoff": fix.get("date") or "",
                "totals": totals,
                "bookmakers_count": bk_count,
                "home_form": home_form,
                "away_form": away_form,
            })

    # ML Over/Under probabilities, computed once for the whole batch.
    ml_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        from .ml_predict import MlPredictor

        ml_cfg = cfg.get("models", {}).get("ml", {})
        ml_predictor = MlPredictor(
            ROOT / ml_cfg.get("models_dir", "cache/football/models"),
            window=int(ml_cfg.get("window", 5)),
            gd_margin=int(ml_cfg.get("gd_margin", 2)),
        )
        ml_entries = [
            (p["league_key"], p["home"], p["away"], p["kickoff"][:10]) for p in pending
        ]
        ml_map = ml_predictor.predict_matches(ml_entries)
    except Exception as exc:  # noqa: BLE001 -- ML must never break the goal flow
        logger.warning("bestgoalmatch: ml predictor unavailable: %s", exc)

    league_avg: dict[str, float] = {}
    matches: list[dict[str, Any]] = []
    league_counts: dict[str, tuple[float, int]] = {}
    for p in pending:
        ml_over = None
        item = ml_map.get((p["league_key"], p["home"], p["away"], p["kickoff"][:10]))
        if item and item.get("over"):
            ml_over = item["over"].get("over")
        profile = _goal_profile(p["home_form"], p["away_form"], p["totals"], ml_over=ml_over)
        if profile is None:
            continue
        exp, n = league_counts.get(p["league_key"], (0.0, 0))
        league_counts[p["league_key"]] = (exp + profile["expected_total"], n + 1)
        matches.append({
            "league_key": p["league_key"],
            "display": p["display"],
            "home": p["home"],
            "away": p["away"],
            "kickoff": p["kickoff"],
            "has_odds": bool(profile.get("odds_over_2_5")),
            "bookmakers_count": p["bookmakers_count"],
            "goal": profile,
        })
    for key, (exp, n) in league_counts.items():
        if n:
            display = next((p["display"] for p in pending if p["league_key"] == key), key)
            league_avg[display] = round(exp / n, 2)

    if not matches:
        date_label = f"{dates[0]} → {dates[-1]}" if len(dates) > 1 else today
        return {"error": f"Tidak ada match hari ini ({date_label}) dengan data form atau odds untuk goal profile."}

    matches.sort(key=lambda m: m["goal"]["expected_total"], reverse=True)
    top = matches[:10]
    winner = matches[0]

    # Ranked league profiles, highest average expected total first.
    league_ranked = sorted(league_avg.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "date": dates[0] if len(dates) == 1 else f"{dates[0]} → {dates[-1]}",
        "date_range": f"{dates[0]} → {dates[-1]}" if len(dates) > 1 else today,
        "generated_at": utc_now_iso(),
        "league_avg": league_ranked,
        "candidates": [
            {
                "home": m["home"],
                "away": m["away"],
                "kickoff": m["kickoff"],
                "league": m["display"],
                "goal": m["goal"],
            }
            for m in top
        ],
        # winner carries `league` (display name) for the formatter: matches[]
        # use `display`, while format_best_goal reads `league`.
        "winner": {**winner, "league": winner["display"]},
        "quota": {
            "odds_api_remaining": getattr(odds, "last_remaining", None),
            "odds_blocked": getattr(odds, "quota_blocked", False),
            "football_data_warning": getattr(stats.fd, "rate_limit_warning", False),
        },
    }
