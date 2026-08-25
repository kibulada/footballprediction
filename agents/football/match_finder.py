"""Top H+X match finder using multi-source stats fetcher.

Fixtures are sourced from football-data.org via MultiSourceStatsFetcher.
Falls back gracefully when league has no scheduled matches.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from .analyse import _teams_match, extract_h2h_entries
from .cache import Cache
from .league_resolver import competition_league_key
from .multi_source import MultiSourceStatsFetcher
from .odds_fetcher import OddsFetcher
from .predictor import grade_top_match
from .scorer import best_odds, consensus_odds, find_outlier, score_signal

LEAGUES_PATH = Path(__file__).parent / "leagues.json"


def _load_leagues() -> dict[str, dict[str, Any]]:
    return json.loads(LEAGUES_PATH.read_text(encoding="utf-8"))


def _resolve_date(date_str: str | None) -> str:
    WIB = timezone(timedelta(hours=7))
    today = datetime.now(WIB).date()
    return today.isoformat() if not date_str else date_str


# Status strings emitted by football-data.org ('SCHEDULED', 'TIMED',
# 'FINISHED', ...) and the flashscore homepage row classifier
# ('scheduled' | 'live' | 'finished').
_UPCOMING_STATUSES = {"scheduled", "timed", "notstarted", "not_started"}
_DONE_OR_INVALID_STATUSES = {
    "finished", "ft", "fulltime", "awarded", "cancelled", "canceled",
    "postponed", "abandoned", "suspended", "paused", "interrupted",
    "delayed", "inplay", "in_play", "inprogress", "live", "ht", "aet",
    "et", "pen", "pens", "timetobedefined", "waiting",
}


def _kickoff_utc(iso: str | None) -> datetime | None:
    """ISO-8601 kickoff -> aware UTC datetime (None when unparseable)."""
    if not iso:
        return None
    try:
        cleaned = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _is_upcoming(status: Any, kickoff: str | None = None) -> bool:
    """True when a fixture/homepage row has not kicked off yet.

    The top command must only surface matches that are still playable:
    finished and in-play rows are dropped. Status strings that we do not
    recognize (or a missing status on an old cached row) fall back to the
    kickoff time -- a match that started more than 2h45m ago is treated as
    done; anything unclassifiable is kept rather than hidden.
    """
    s = str(status or "").strip().lower()
    if s in _UPCOMING_STATUSES:
        return True
    if s in _DONE_OR_INVALID_STATUSES:
        return False
    kt = _kickoff_utc(kickoff)
    if kt is not None:
        return (kt + timedelta(hours=2, minutes=45)) > datetime.now(timezone.utc)
    return True


def _norm_league_token(name: str) -> str:
    """Lowercase + accent-fold + tokenize a league name (diacritics stripped
    so "Süper Lig" == "super lig")."""
    s = unicodedata.normalize("NFD", (name or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# Tokens that only qualify a nowgoal league name as the SAME competition:
# UEFA wrappers, the qualification stage, and country adjectives that
# prefix the competition's own name ("English Premier League" == EPL,
# "Italian Serie A" == Serie A). Regional and level markers -- national,
# nsw, queensland, russian, afc, ofc, concacaf, women, youth, etc. -- are
# deliberately NOT here: the old loose substring matcher let "premier
# league" match inside "National Premier Leagues NSW" and "Russian Premier
# League", flooding the EPL fixture cache with hundreds of unrelated matches
# (observed live 2026-08-14) and blowing the runner deadline.
_NOWGOAL_NEUTRAL_TOKENS = frozenset({
    "uefa", "europa", "qualification", "qualifiers",
    "english", "spanish", "italian", "italy", "german", "french",
    "portuguese", "dutch", "belgian", "scottish", "saudi", "turkish",
    "indonesia", "indonesian", "american", "usa", "australian", "korean",
    "japanese",
})
_NOWGOAL_MAX_EXTRA_TOKENS = 2


def _nowgoal_league_matches(league_name: str, meta: dict[str, Any]) -> bool:
    """True when a nowgoal schedule row's league name refers to this league.

    Strict token matching: every significant token of the league's display
    name or one of its aliases must appear in the row's league name (token
    subset -- "champions" never matches "Championship"), and the row may
    only carry a tiny set of neutral extra tokens (UEFA wrappers /
    qualification). So "Premier League" matches EPL, but "National Premier
    Leagues NSW", "Russian Premier League" and "AFC Champions League 2 -
    Qualification" do not."""
    hay = set(_norm_league_token(league_name).split())
    if not hay:
        return False
    for t in [meta.get("display") or ""] + list(meta.get("aliases") or []):
        needle = set(_norm_league_token(t).split())
        if not needle:
            continue
        if needle <= hay:
            extra = hay - needle
            if len(extra) <= _NOWGOAL_MAX_EXTRA_TOKENS and extra <= _NOWGOAL_NEUTRAL_TOKENS:
                return True
    return False


async def _nowgoal_fixtures_for_league(
    nowgoal: Any,
    target_date: str,
    league_name: str,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fixtures for ONE league from the nowgoal schedule (football-data 429 fallback).

    The nowgoal schedule endpoint returns ALL leagues in a single request, so
    this is cheap even when many leagues need it (the client caches the raw
    schedule per date internally). Rows are filtered by league name and
    normalized to the same shape football-data produces so the rest of the
    top pipeline (odds matching, form, ranking) is unchanged.
    """
    try:
        rows = await nowgoal.fetch_schedule(target_date)
    except Exception as exc:
        logger.warning("nowgoal schedule fallback failed (%s): %s", league_name, exc)
        return []
    if not rows:
        return []
    out = []
    for r in rows:
        if not _nowgoal_league_matches((r or {}).get("league_name") or "", meta):
            continue
        out.append({
            "id": r.get("match_id"),
            "home": {"id": r.get("home_id"), "name": r.get("home")},
            "away": {"id": r.get("away_id"), "name": r.get("away")},
            "date": r.get("kickoff"),
            "status": r.get("status"),
            "source": "nowgoal",
        })
    return out


async def find_top_matches(
    *,
    date: str | None,
    leagues: list[str],
    top_n: int,
    cfg: dict[str, Any],
    odds: OddsFetcher,
    stats: MultiSourceStatsFetcher,
    cache: Cache,
    days: int = 1,
    nowgoal: Any = None,
) -> dict[str, Any]:
    """Top-N matches over a WIB calendar-day window (default: today only).

    ``days`` widens the window (e.g. 2 = today + tomorrow WIB) so matches
    that kick off 00:00-06:59 WIB -- which belong to the NEXT WIB calendar
    date -- are not silently dropped by a single-day window. The Odds API
    payload is fetched once per league (it carries all upcoming commence
    times) and filtered to the window dates; football-data fixtures are
    fetched per WIB date (cached per league+date as before).
    """
    leagues_cfg = _load_leagues()
    start_date = _resolve_date(date)
    days = max(1, int(days or 1))
    ttl = cfg["cache_ttl_seconds"]

    # WIB calendar dates covered by the window: [start, start + days - 1].
    try:
        start_d = datetime.strptime(start_date, "%Y-%m-%d").date()
        target_dates: list[str] = [
            (start_d + timedelta(days=i)).isoformat() for i in range(days)
        ]
    except ValueError:
        target_dates = [start_date]

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    from .timeutil import wib_date_from_iso

    # Hard wall-clock budget for the whole scan (mirrors find_best_matches):
    # the per-fixture form chain (flashscore browser render -> football-data
    # -> thesportsdb) is the expensive part, so a busy day or a fallback
    # flood must never blow the runner deadline. Once ~55s have elapsed we
    # stop scanning new leagues/fixtures and return the ranked prefix.
    _SCAN_BUDGET = 55.0
    _scan_t0 = time.monotonic()

    for league_name in leagues:
        if time.monotonic() - _scan_t0 > _SCAN_BUDGET:
            logger.warning("top: scan time budget hit after %d candidates", len(candidates))
            break
        meta = leagues_cfg.get(league_name)
        if not meta:
            continue
        display = meta["display"]
        odds_key = meta.get("odds_api_key")
        meta_with_season = {**meta, "season": datetime.now(timezone.utc).year, "_league_key": league_name}

        # Fixtures across the whole window (cached per league + WIB date).
        # football-data free tier is 10 req/min and this loop costs ONE request
        # per league, so past the 10th league every call 429s and the day comes
        # back empty. When football-data has nothing, fall back to the nowgoal
        # schedule for that date (one shared request covers ALL leagues) -- the
        # day then still surfaces matches instead of a misleading "no matches".
        fixtures: list[dict[str, Any]] = []
        nowgoal_hit = False
        for target_date in target_dates:
            # P2: provider-aware cache keys -- the football-data and nowgoal
            # fallback results are stored separately so a day once filled by
            # the nowgoal schedule (when football-data was rate-limited/empty)
            # cannot be served to a later query that would prefer football-data
            # within the TTL window.
            fd_cache_key = f"fixtures_football_data_{league_name}_{target_date}"
            ng_cache_key = f"fixtures_nowgoal_{league_name}_{target_date}"
            day_fixtures = cache.get(fd_cache_key, ttl["fixtures"])
            if day_fixtures is None:
                day_fixtures = await stats.fetch_fixtures_for_date(meta_with_season, target_date)
                if day_fixtures:
                    cache.set(fd_cache_key, day_fixtures)
                elif nowgoal is not None:
                    # football-data empty -> check the nowgoal cache, else fetch
                    day_fixtures = cache.get(ng_cache_key, ttl["fixtures"])
                    if day_fixtures is None:
                        day_fixtures = await _nowgoal_fixtures_for_league(
                            nowgoal, target_date, league_name, meta
                        )
                        if day_fixtures:
                            cache.set(ng_cache_key, day_fixtures)
                    if day_fixtures:
                        nowgoal_hit = True
            fixtures.extend(day_fixtures or [])
        if not fixtures:
            continue
        if nowgoal_hit:
            logger.info("fixtures %s via nowgoal schedule (football-data kosong)", league_name)

        # Map fixture ids to names so the form provider chain can run
        # (flashscore/football-data/thesportsdb lookups by id).
        team_names: dict[str, str] = {}
        for fix in fixtures:
            team_names[str(fix["home"]["id"])] = fix["home"]["name"]
            team_names[str(fix["away"]["id"])] = fix["away"]["name"]
        meta_with_season["_team_names"] = team_names

        odds_matches: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        # Qualification rounds live under a shared *_qualification sport key
        # (The Odds API lumps ALL UEFA qualifiers -- UCL, UEL, UECL -- into
        # soccer_uefa_champs_league_qualification), so try the league's alt
        # keys too when the main key has nothing.
        for okey in [odds_key] + list(meta.get("odds_alt_keys") or []):
            if not okey:
                continue
            # The Odds API payload lists every upcoming commence time for the
            # sport key, so ONE fetch per league covers the whole window
            # (cache key uses the start date; a single-day run reuses it).
            odds_cache_key = f"odds_{okey}_{start_date}"
            odds_payload = cache.get(odds_cache_key, ttl["odds"])
            if odds_payload is None:
                odds_payload = await odds.fetch_odds(okey) or []
                cache.set(odds_cache_key, odds_payload)

            for match in odds_payload:
                commence = match.get("commence_time", "")
                if wib_date_from_iso(commence) not in target_dates:
                    continue
                entries = extract_h2h_entries(
                    match, match.get("home_team", ""), match.get("away_team", "")
                )
                if entries:
                    odds_matches.append((match, entries))

        # Defensive per-league cap: the ranked prefix needs only a handful of
        # candidates, and form fetching is the expensive part of the scan.
        for fix in fixtures[:30]:
            if time.monotonic() - _scan_t0 > _SCAN_BUDGET:
                logger.warning("top: scan time budget hit after %d candidates", len(candidates))
                break
            # Only surface matches that have not kicked off yet: football-data
            # re-lists finished/played fixtures for the day inside the date
            # window, and they must not appear as "value matches".
            if not _is_upcoming(fix.get("status"), fix.get("date")):
                continue
            home = fix["home"]["name"]
            away = fix["away"]["name"]
            kickoff = fix.get("date", "")
            # Dedupe across the multi-day window (a fixture is fetched once
            # per WIB date; overlapping rows must not double-count).
            dedupe_key = (league_name, home, away, str(kickoff))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            home_id = fix["home"]["id"]
            away_id = fix["away"]["id"]

            match_odds: list[dict[str, Any]] = []
            for odds_match, entries in odds_matches:
                if _teams_match(odds_match.get("home_team", ""), home) and _teams_match(
                    odds_match.get("away_team", ""), away
                ):
                    match_odds = entries
                    break
            has_odds = bool(match_odds)
            consensus = consensus_odds(match_odds) if match_odds else {"home": 0, "draw": 0, "away": 0}
            outlier = find_outlier(match_odds, consensus, cfg["outlier_threshold_pct"]) if match_odds else None
            best = best_odds(match_odds) if match_odds else {}

            home_form = await stats.fetch_team_form(home_id, meta_with_season)
            away_form = await stats.fetch_team_form(away_id, meta_with_season)

            signal = score_signal(
                match_odds,
                consensus,
                outlier,
                home_form.get("sequence") if home_form else None,
                away_form.get("sequence") if away_form else None,
                has_odds,
            )

            # Lightweight screening grade (LAYAK/CUKUP/SKIP) from pre-flight
            # data only -- cheap enough for N matches (no per-match engine).
            grade = grade_top_match(
                has_odds=has_odds,
                has_home_form=bool(home_form and home_form.get("sequence")),
                has_away_form=bool(away_form and away_form.get("sequence")),
                signal=signal,
                bookmakers_count=len(match_odds),
            )

            candidates.append(
                {
                    "home": home,
                    "away": away,
                    "league": display,
                    # Canonical league key (e.g. 'EPL') so the Discord bot can
                    # run `analisa match <key> <home> vs <away>` straight from
                    # the ⚡ analyse button without re-resolving anything.
                    "league_key": league_name,
                    "kickoff": kickoff,
                    "odds": {
                        "consensus": consensus,
                        "best": best,
                        "outlier": outlier,
                    },
                    "stats": {
                        "home_form": (home_form or {}).get("sequence", "n/a"),
                        "away_form": (away_form or {}).get("sequence", "n/a"),
                    },
                    "signal": signal,
                    "has_odds": has_odds,
                    "bookmakers_count": len(match_odds),
                    "grade": grade,
                    "source": fix.get("source"),
                }
            )

    candidates.sort(key=lambda m: m["signal"], reverse=True)

    # ---- Flashscore homepage: competitions football-data does not cover ----
    # Rendered ONCE per command (not per league): the homepage lists today's
    # matches from every competition -- Conference League qualification,
    # friendlies, AFC qualifiers, minor cups. These have no odds/form model
    # (no registered league key), so they are surfaced as CONTEXT, not ranked
    # candidates. Cached per date to avoid a 19s render on every query.
    extra_matches: list[dict[str, Any]] = []
    try:
        from .timeutil import wib_today_iso

        # The flashscore homepage lists TODAY's matches only; show it when
        # today is part of the requested window (single-day runs always are).
        if wib_today_iso() in target_dates:
            # _v2: rows now carry a status field (scheduled/live/finished);
            # the version bump invalidates cached pre-status payloads so the
            # "only not-yet-played" filter applies immediately.
            #
            # NOTE: the key MUST use wib_today_iso(), not a league-loop
            # variable: the homepage sentinel league ('__homepage__', used by
            # the bot's auto-detect path) never enters the league loop, so any
            # per-league variable would raise NameError and silently empty
            # extra_matches (regression fixed 2026-08). The homepage always
            # reflects TODAY's fixtures, so today is also the correct key.
            hp_key = f"flashscore_homepage_{wib_today_iso()}_v3"
            # Short TTL (default 20 min): a match's status flips to live then
            # finished as the day progresses; the generic 6h fixtures TTL made
            # finished matches keep showing as 'belum bertanding' all day.
            hp = cache.get(hp_key, ttl.get("homepage", 1200))
            if hp is None:
                hp = await stats.fetch_homepage_matches() or []
                if hp:
                    cache.set(hp_key, hp)
            for m in hp or []:
                home_name = (m.get("home") or {}).get("name")
                away_name = (m.get("away") or {}).get("name")
                if not (home_name and away_name):
                    continue
                # Drop finished/live homepage rows: the top output must only
                # list matches that have not kicked off yet.
                if not _is_upcoming(m.get("status")):
                    continue
                # Warm the per-pair flashscore resolve cache for analyzable
                # homepage fixtures (same key shape search_teams_pair reads).
                # The follow-up `analyse` subprocess then hits a cache hit (0s)
                # instead of re-rendering the league page + homepage (~43s),
                # which previously ate the whole 72s analysis budget. One
                # homepage scrape resolves every pair: no extra browser renders.
                comp_key = competition_league_key(m.get("competition") or "")
                if comp_key:
                    _safe = lambda s: " ".join((s or "").lower().split())
                    rkey = (
                        "flashscore_resolve_"
                        f"{comp_key}_{wib_today_iso()}_"
                        f"{_safe(home_name)}_{_safe(away_name)}"
                    )
                    if cache.get(rkey, ttl_seconds=24 * 3600) is None:
                        home_ref = {
                            "id": (m.get("home") or {}).get("id"),
                            "name": home_name,
                            "slug": m.get("home_slug"),
                        }
                        away_ref = {
                            "id": (m.get("away") or {}).get("id"),
                            "name": away_name,
                            "slug": m.get("away_slug"),
                        }
                        def _team(side: dict[str, Any]) -> dict[str, Any]:
                            return {
                                "id": side.get("id"),
                                "name": side.get("name"),
                                "short_name": None,
                                "country": None,
                                "provider": "flashscore",
                                "_role": "primary_flashscore",
                            }
                        cache.set(
                            rkey,
                            {
                                "home": _team(home_ref),
                                "away": _team(away_ref),
                                "_match": {
                                    "home": home_ref,
                                    "away": away_ref,
                                    "match_url": m.get("match_url"),
                                    "date_text": m.get("date_text"),
                                    "source": "flashscore",
                                },
                            },
                        )
                extra_matches.append(
                    {
                        "home": home_name,
                        "away": away_name,
                        "competition": m.get("competition") or "Other",
                        "kickoff": m.get("date_text"),
                        "source": m.get("source"),
                    }
                )
    except Exception as exc:
        logger.warning("flashscore homepage extra matches failed (top unaffected): %s", exc)

    from .timeutil import utc_now_iso

    return {
        "date": start_date,
        "days": days,
        "date_range": (
            f"{target_dates[0]}" if len(target_dates) == 1
            else f"{target_dates[0]} → {target_dates[-1]}"
        ),
        "generated_at": utc_now_iso(),
        "matches": candidates[:top_n],
        "extra_matches": extra_matches,
        "quota": {
            "odds_api_remaining": odds.last_remaining,
            "odds_blocked": odds.quota_blocked,
            "football_data_warning": stats.fd.rate_limit_warning,
            "nowgoal_fixtures_used": any(
                (m.get("source") == "nowgoal") for m in (candidates or [])
            ) or any(
                (m.get("source") == "nowgoal") for m in (extra_matches or [])
            ),
        },
        "leagues_no_odds": [
            leagues_cfg[name]["display"]
            for name in leagues
            if name in leagues_cfg
            and leagues_cfg.get(name, {}).get("odds_api_key") is None
        ],
    }
