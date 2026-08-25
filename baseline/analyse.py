"""Analyse a specific match using multi-source stats fetcher.

Provider chain resolves per league: sofascore (primary) -> football-data.org ->
thesportsdb (fallback). Each field (form, H2H) is fetched independently so a
failure in one provider doesn't break the whole analysis.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from .cache import Cache
from .league_resolver import resolve_league
from .multi_source import MultiSourceStatsFetcher
from .odds_fetcher import OddsFetcher
from .predictor import derive_picks
from .scorer import best_odds, consensus_odds, find_outlier, score_signal

ROOT = Path(__file__).resolve().parent.parent.parent


def _season_now() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


_TEAM_NAME_PREFIXES = {
    "fk", "fc", "nk", "cd", "sc", "pfc", "ifk", "ss", "rc", "ca",
    "ec", "cr", "se", "ac", "cf", "us", "sd", "de", "sv", "sk",
}


# Letters whose Unicode NFD does not decompose to an ASCII base (strokes and
# dots are not combining marks, so they would be dropped entirely): map them
# explicitly so cross-provider names like "Bodø/Glimt" vs "Bodo/Glimt"
# normalize to the same token.
_STROKE_LETTERS = str.maketrans(
    {
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ħ": "h",
        "ı": "i",
        "ŋ": "n",
        "ß": "ss",
    }
)


def _norm_team_name(name: str) -> str:
    """Lowercase, strip accents and punctuation -> comparable token string."""
    import re
    import unicodedata

    s = (name or "").lower().translate(_STROKE_LETTERS)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _teams_match(a: str, b: str) -> bool:
    """Tolerant team-name equality across providers.

    Providers disagree on prefixes ("FK Bodø/Glimt" vs "Bodø/Glimt") and
    honorifics ("Royale Union Saint-Gilloise" vs "Union Saint-Gilloise"),
    so we normalize, drop common prefixes, and fall back to containment.
    """
    na, nb = _norm_team_name(a), _norm_team_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    for pref in _TEAM_NAME_PREFIXES:
        if na.startswith(pref + " ") and na[len(pref) + 1:] == nb:
            return True
        if nb.startswith(pref + " ") and nb[len(pref) + 1:] == na:
            return True
    if len(na) >= 6 and (na in nb or nb in na):
        return True
    return False


def extract_h2h_entries(
    payload: dict[str, Any],
    home_name: str,
    away_name: str,
    home_query: str | None = None,
    away_query: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-bookmaker 1X2 entries from an odds payload.

    Outcome names come from the odds provider and can differ from the
    resolved team names ("Bodø/Glimt" vs "FK Bodø/Glimt", "Union
    Saint-Gilloise" vs "Royale Union Saint-Gilloise"), so the home/away
    sides are matched tolerantly via _teams_match instead of exact equality.
    The raw user query is used as a secondary fallback (e.g. the odds
    provider lists "Sabah FK" while our resolution returns "Sabah Baku").
    """
    entries: list[dict[str, Any]] = []
    home_candidates = [n for n in (home_name, home_query) if n]
    away_candidates = [n for n in (away_name, away_query) if n]
    for bm in payload.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            entry = {"bookmaker": bm.get("title", "?")}
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if any(_teams_match(name, h) for h in home_candidates):
                    entry["home"] = price
                elif any(_teams_match(name, a) for a in away_candidates):
                    entry["away"] = price
                elif (name or "").lower() == "draw":
                    entry["draw"] = price
            if "home" in entry and "away" in entry:
                entries.append(entry)
    return entries


async def find_match_odds_payload(
    odds_keys: list[str],
    home_name: str,
    away_name: str,
    odds: "OddsFetcher",
    cache: "Cache",
    cache_ttl_seconds: dict[str, int],
    home_query: str | None = None,
    away_query: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Find odds for a match across candidate sport keys.

    The primary league key (e.g. soccer_uefa_champs_league) often has no
    fixtures while the qualification variant does
    (soccer_uefa_champs_league_qualification); each candidate key is tried
    in order and cached separately. Returns (match_payload, key_that_hit)
    or (None, None). The raw user queries are accepted as fallback names
    (resolved names may differ from the odds provider's spelling).
    """
    ttl = cache_ttl_seconds.get("odds", 900)
    home_candidates = [n for n in (home_name, home_query) if n]
    away_candidates = [n for n in (away_name, away_query) if n]
    for key in odds_keys:
        if not key:
            continue
        odds_cache_key = f"odds_{key}_recent"
        payload = cache.get(odds_cache_key, ttl) or []
        if not payload:
            payload = await odds.fetch_odds(key) or []
            cache.set(odds_cache_key, payload)
        for match in payload:
            if any(
                _teams_match(match.get("home_team"), h) for h in home_candidates
            ) and any(
                _teams_match(match.get("away_team"), a) for a in away_candidates
            ):
                return match, key
    return None, None


async def find_specific_match(
    *,
    league_query: str,
    home_query: str,
    away_query: str,
    cfg: dict[str, Any],
    odds: OddsFetcher,
    stats: MultiSourceStatsFetcher,
    cache: Cache,
) -> dict[str, Any]:
    resolved = resolve_league(league_query)
    if not resolved:
        return {"error": f"liga '{league_query}' tidak dikenal", "teams": []}
    league_key, meta = resolved
    display = meta["display"]
    odds_key = meta.get("odds_api_key")
    season = _season_now()
    meta_with_season = {**meta, "season": season, "_league_key": league_key}

    home_team, away_team = await stats.search_teams_pair(
        home_query, away_query, meta_with_season
    )

    if home_team and away_team:
        meta_with_season["_team_names"] = {
            str(home_team["id"]): home_team["name"],
            str(away_team["id"]): away_team["name"],
        }

    soft_id_map: dict[str, int] = {}
    if home_team and (home_team.get("provider") == "sofascore") and isinstance(home_team.get("id"), int):
        soft_id_map[str(home_team["id"])] = home_team["id"]
    if away_team and (away_team.get("provider") == "sofascore") and isinstance(away_team.get("id"), int):
        soft_id_map[str(away_team["id"])] = away_team["id"]

    if not soft_id_map and (home_team or away_team):
        sofascore_tid = meta_with_season.get("sofascore_tournament_id")
        if sofascore_tid:
            for side, team in (("home", home_team), ("away", away_team)):
                if not team:
                    continue
                found = await stats._sofascore_team_in_tournament(sofascore_tid, team.get("name") or "")
                if found and isinstance(found.get("id"), int):
                    soft_id_map[str(team["id"])] = found["id"]
    if soft_id_map:
        meta_with_season["_sofascore_team_ids"] = soft_id_map

    home_id = (home_team or {}).get("id")
    away_id = (away_team or {}).get("id")
    if not home_id or not away_id:
        missing = []
        if not home_id:
            missing.append(f"home '{home_query}'")
        if not away_id:
            missing.append(f"away '{away_query}'")
        quota_notes = []
        if stats.fd.rate_limit_warning:
            quota_notes.append("football-data rate limit")
        if stats.sc.quota_warning:
            quota_notes.append("Sofascore 403/429")
        quota_msg = ""
        if quota_notes:
            quota_msg = f" (provider issue: {', '.join(quota_notes)})"
        return {
            "error": f"tim tidak ditemukan: {', '.join(missing)}{quota_msg}",
            "league": display,
            "home_query": home_query,
            "away_query": away_query,
            "home_candidates": home_team,
            "away_candidates": away_team,
        }

    home_name = home_team["name"]
    away_name = away_team["name"]

    fixture = await stats.fetch_upcoming_fixture(home_id, away_id, meta_with_season)
    home_form = await stats.fetch_team_form(home_id, meta_with_season)
    away_form = await stats.fetch_team_form(away_id, meta_with_season)
    h2h = await stats.fetch_h2h(home_id, away_id, meta_with_season)

    # No data leakage: live/final event stats are only usable pre-match. If the
    # queried fixture has kicked off, its own stats would leak into the model.
    fixture_sofascore_id = fixture.get("sofascore_id") if fixture else None
    fixture_is_prematch = bool(
        fixture
        and fixture.get("source") == "sofascore"
        and fixture.get("status") == "notstarted"
    )
    sofascore_stats = None
    if fixture_sofascore_id and fixture_is_prematch:
        sofascore_stats = await stats.fetch_event_stats_extended(fixture_sofascore_id)

    home_history = None
    away_history = None
    home_soft = meta_with_season.get("_sofascore_team_ids", {}).get(str(home_id))
    away_soft = meta_with_season.get("_sofascore_team_ids", {}).get(str(away_id))
    history_ttl = cfg["cache_ttl_seconds"].get("sofascore_history", 1800)
    if home_soft:
        cache_key = f"sofascore_history_{home_soft}_5"
        cached = cache.get(cache_key, history_ttl)
        if cached is None:
            home_history = await stats.fetch_team_history_stats(
                home_soft, limit=5, exclude_event_id=fixture_sofascore_id
            )
            cache.set(cache_key, home_history)
        else:
            home_history = cached
    if away_soft:
        cache_key = f"sofascore_history_{away_soft}_5"
        cached = cache.get(cache_key, history_ttl)
        if cached is None:
            away_history = await stats.fetch_team_history_stats(
                away_soft, limit=5, exclude_event_id=fixture_sofascore_id
            )
            cache.set(cache_key, away_history)
        else:
            away_history = cached

    match_odds_payload: dict[str, Any] | None = None
    if odds_key:
        # Primary key first, then qualification fallbacks (odds_alt_keys in
        # league meta) -- e.g. UCL play-off matches live under the
        # *_qualification sport key, not the main league key.
        odds_keys = [odds_key] + list(meta.get("odds_alt_keys") or [])
        match_odds_payload, _ = await find_match_odds_payload(
            odds_keys, home_name, away_name, odds, cache,
            cfg["cache_ttl_seconds"],
            home_query=home_query, away_query=away_query,
        )

    kickoff = (fixture or {}).get("date")
    # The odds payload knows the exact kickoff even when the fixture endpoint
    # has no entry (e.g. qualification rounds live on a different
    # competition), so fall back to commence_time when fixture is missing.
    if not kickoff and match_odds_payload and match_odds_payload.get("commence_time"):
        kickoff = match_odds_payload["commence_time"]

    bookmaker_odds_h2h: list[dict[str, Any]] = []
    market_totals: dict[str, dict[str, float]] = {}
    if match_odds_payload:
        bookmaker_odds_h2h = extract_h2h_entries(
            match_odds_payload, home_name, away_name,
            home_query=home_query, away_query=away_query,
        )
        for bm in match_odds_payload.get("bookmakers", []):
            for market in bm.get("markets", []):
                mkey = market.get("key")
                if mkey == "totals":
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        point = outcome.get("point", 0)
                        price = outcome.get("price", 0)
                        label = f"{name} {point}"
                        existing = market_totals.get(label, {})
                        if existing.get("odds", 0) < price:
                            market_totals[label] = {
                                "odds": price,
                                "point": point,
                                "bookmaker": bm.get("title", "?"),
                            }
                elif mkey == "btts":
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if not name or not price:
                            continue
                        label = f"BTTS {name}"  # "BTTS Yes" / "BTTS No"
                        existing = market_totals.get(label, {})
                        if existing.get("odds", 0) < price:
                            market_totals[label] = {
                                "odds": price,
                                "point": None,
                                "bookmaker": bm.get("title", "?"),
                            }

    has_odds = bool(bookmaker_odds_h2h)
    consensus = consensus_odds(bookmaker_odds_h2h) if bookmaker_odds_h2h else {"home": 0, "draw": 0, "away": 0}
    outlier = find_outlier(bookmaker_odds_h2h, consensus, cfg["outlier_threshold_pct"]) if bookmaker_odds_h2h else None
    best = best_odds(bookmaker_odds_h2h) if bookmaker_odds_h2h else {}
    signal = score_signal(
        bookmaker_odds_h2h,
        consensus,
        outlier,
        home_form.get("sequence") if home_form else None,
        away_form.get("sequence") if away_form else None,
        has_odds,
    )

    xg_lambda = None
    if sofascore_stats:
        xg_h = sofascore_stats.get("xg_home")
        xg_a = sofascore_stats.get("xg_away")
        if isinstance(xg_h, (int, float)) and isinstance(xg_a, (int, float)):
            xg_lambda = (float(xg_h), float(xg_a))

    if xg_lambda is None and home_history and away_history:
        h_for = home_history.get("xg_for_avg")
        a_for = away_history.get("xg_for_avg")
        h_against = home_history.get("xg_against_avg")
        a_against = away_history.get("xg_against_avg")
        if (
            isinstance(h_for, (int, float))
            and isinstance(a_for, (int, float))
            and isinstance(h_against, (int, float))
            and isinstance(a_against, (int, float))
        ):
            home_xg = (float(h_for) + float(a_against)) / 2.0
            away_xg = (float(a_for) + float(h_against)) / 2.0
            xg_lambda = (home_xg, away_xg)

    picks_payload = {"top_picks": [], "best_pick": None, "model_probs": {}}
    if has_odds and consensus.get("home", 0) > 0:
        picks_payload = derive_picks(consensus, market_totals, signal, xg_lambda=xg_lambda)

    sources = sorted({
        (home_team.get("provider") if home_team else None),
        (away_team.get("provider") if away_team else None),
        (home_form.get("source") if home_form else None),
        (away_form.get("source") if away_form else None),
        (h2h.get("source") if h2h else None),
        ((fixture or {}).get("source")),
        ("sofascore_xg" if (sofascore_stats and "xg_home" in sofascore_stats) else None),
        ("sofascore_history" if (home_history or away_history) else None),
    } - {None})

    # ---- Prediction engine (Elo + feature Poisson + ensemble + calibration) ----
    prediction = None
    if (home_form or away_form or home_history or away_history):
        from .calibration import Calibrator, SignalScorer
        from .context import build_match_context
        from .elo import EloModel
        from .models import Ensemble, PoissonModel, run_prediction_engine

        ctx = build_match_context(
            league=display,
            home=home_name,
            away=away_name,
            kickoff=kickoff,
            stats={
                "home_form": (home_form or {}).get("sequence"),
                "away_form": (away_form or {}).get("sequence"),
                "home_gf_avg": (home_form or {}).get("gf_avg"),
                "home_ga_avg": (home_form or {}).get("ga_avg"),
                "away_gf_avg": (away_form or {}).get("gf_avg"),
                "away_ga_avg": (away_form or {}).get("ga_avg"),
                # Raw scorelines (oldest->newest) enable time-decay weighting
                # so LIVE predictions use the same features as backtest/validate.
                "home_recent_goals": (home_form or {}).get("recent_goals"),
                "away_recent_goals": (away_form or {}).get("recent_goals"),
                "home_xg_for": (home_history or {}).get("xg_for_avg"),
                "home_xg_against": (home_history or {}).get("xg_against_avg"),
                "away_xg_for": (away_history or {}).get("xg_for_avg"),
                "away_xg_against": (away_history or {}).get("xg_against_avg"),
                "h2h": (h2h or {}),
            },
            odds={
                "has_odds": has_odds,
                "consensus": consensus,
                "totals": market_totals,
            },
            sources=sorted(sources),
        )
        elo_cfg = cfg.get("models", {}).get("elo", {})
        poisson_cfg = cfg.get("models", {}).get("poisson", {})
        ens_cfg = cfg.get("models", {}).get("ensemble", {})
        cal_cfg = cfg.get("models", {}).get("calibration", {})
        elo = EloModel(
            k=elo_cfg.get("k", 32.0),
            home_advantage=elo_cfg.get("home_advantage", 65.0),
            initial_rating=elo_cfg.get("initial_rating", 1500.0),
            base_total_goals=elo_cfg.get("base_total_goals", 2.7),
            path=ROOT / elo_cfg.get("file", "cache/football/elo.json"),
        )
        poisson = PoissonModel(
            base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
            base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
            dc_rho=poisson_cfg.get("dc_rho", -0.1),
            shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
            time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
            xg_weight=poisson_cfg.get("xg_weight", 0.65),
        )
        ensemble = Ensemble(
            elo_weight=ens_cfg.get("elo_weight", 0.5),
            poisson_weight=ens_cfg.get("poisson_weight", 0.5),
        )
        calibrator = Calibrator(
            path=ROOT / cal_cfg.get("file", "cache/football/calibration.json"),
            min_samples=cal_cfg.get("min_samples", 200),
        )
        scorer = SignalScorer()
        prediction = run_prediction_engine(
            ctx,
            elo=elo,
            poisson=poisson,
            ensemble=ensemble,
            calibrator=calibrator,
            scorer=scorer,
        )
        if prediction is not None:
            prediction = prediction.to_dict()

    # ---- Recommendation grading (VALID / CANDIDATE / HATI-HATI) ---------
    # A pick is only advertised as a valid bet when the model behind it is
    # reliable: confidence HIGH, calibration validated, data complete, real
    # edge, strong signal. Everything else is downgraded with reasons. When
    # the prediction engine did not run (no form/history at all), picks are
    # still graded but can never reach VALID/CANDIDATE (all gates report
    # "tidak dihitung").
    if picks_payload.get("top_picks"):
        from .predictor import grade_recommendation

        conf = prediction.get("confidence") if prediction else None
        calib = (prediction.get("calibration") or {}).get("quality") if prediction else None
        compl = prediction.get("data_completeness") if prediction else None
        sig = prediction.get("signal_strength") if prediction else None
        for p in picks_payload["top_picks"]:
            p["grade"] = grade_recommendation(
                confidence=conf,
                calibration_quality=calib,
                data_completeness=compl,
                edge_pct=p.get("edge"),
                signal=sig or 0,
            )

    # ---- PHASE 7: immutable prediction snapshot (append-only JSONL) ----
    # Logging must NEVER break the prediction flow.
    try:
        pl_cfg = cfg.get("prediction_log") or {}
        if pl_cfg.get("enabled") and pl_cfg.get("file"):
            from .prediction_log import append_snapshot, make_match_id

            pred = prediction or {}
            sig = pred.get("signal_strength")
            append_snapshot(
                ROOT / pl_cfg["file"],
                match_id=make_match_id(league_key, home_name, away_name, kickoff),
                league=display,
                home=home_name,
                away=away_name,
                kickoff=kickoff,
                prob=(pred.get("model_probs") or {}).get("1x2"),
                odds=consensus if has_odds else None,
                edge=pred.get("market_edge"),
                confidence=pred.get("confidence"),
                signal=int(sig) if sig is not None else (int(signal) if isinstance(signal, int) else None),
                calibration=pred.get("calibration"),
                model_version=pred.get("model_version"),
                input_hash=pred.get("input_hash"),
                best_pick=(picks_payload or {}).get("best_pick"),
                sources=sources,
            )
    except Exception as exc:
        logger.warning("prediction log write failed (prediction unaffected): %s", exc)

    from .timeutil import utc_now_iso

    return {
        "league": display,
        "league_key": league_key,
        "generated_at": utc_now_iso(),
        "prediction": prediction,
        "home": home_name,
        "away": away_name,
        "kickoff": kickoff,
        "venue": (fixture or {}).get("venue"),
        "match_found": fixture is not None,
        "fixture_source": (fixture or {}).get("source"),
        "stats": {
            "home_form": (home_form or {}).get("sequence", "n/a"),
            "away_form": (away_form or {}).get("sequence", "n/a"),
            "home_gf_avg": (home_form or {}).get("gf_avg", 0),
            "home_ga_avg": (home_form or {}).get("ga_avg", 0),
            "away_gf_avg": (away_form or {}).get("gf_avg", 0),
            "away_ga_avg": (away_form or {}).get("ga_avg", 0),
            "home_split": (home_form or {}).get("home", {}),
            "away_split": (away_form or {}).get("away", {}),
            "h2h": {
                "wins": (h2h or {}).get("wins", 0),
                "draws": (h2h or {}).get("draws", 0),
                "losses": (h2h or {}).get("losses", 0),
            },
            "home_xg_for": (home_history or {}).get("xg_for_avg"),
            "away_xg_for": (away_history or {}).get("xg_for_avg"),
            "home_xg_against": (home_history or {}).get("xg_against_avg"),
            "away_xg_against": (away_history or {}).get("xg_against_avg"),
            "home_corners_for": (home_history or {}).get("corners_for_avg"),
            "away_corners_for": (away_history or {}).get("corners_for_avg"),
            "home_corners_against": (home_history or {}).get("corners_against_avg"),
            "away_corners_against": (away_history or {}).get("corners_against_avg"),
            "home_yellow_for": (home_history or {}).get("yellow_for_avg"),
            "away_yellow_for": (away_history or {}).get("yellow_for_avg"),
        },
        "odds": {
            "consensus": consensus,
            "best": best,
            "outlier": outlier,
            "bookmakers_count": len(bookmaker_odds_h2h),
            "has_odds": has_odds,
            "totals": market_totals,
        },
        "signal": signal,
        "picks": picks_payload,
        "sources": sources,
        "sofascore_event_stats": sofascore_stats,
        "quota": {
            "odds_api_remaining": odds.last_remaining,
            "odds_blocked": odds.quota_blocked,
            "football_data_warning": stats.fd.rate_limit_warning,
            "sofascore_warning": stats.sc.quota_warning,
        },
    }
