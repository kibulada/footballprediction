"""Match-source commands: ``!livescore`` / ``!flashscore``.

Both commands share ONE pipeline; the only difference is the initial
match-data source:

    User input
    -> LiveScore / Flashscore match search (today, then tomorrow)
    -> match validation (league + teams + date)
    -> collect the source's match data (form/H2H/stats/lineups, best-effort)
    -> EXISTING analyse pipeline (find_specific_match): NowGoal odds lookup
       -> prediction engine -> decision engine -> existing output format

No prediction logic lives here. The source is used ONLY to find + validate
the requested match and to carry extra match context; every number the user
sees still comes from the existing validated pipeline. A match that cannot
be found today/tomorrow returns a clear "match not found" response and the
prediction engine is never invoked.

Match identification honours the existing conventions:
  - team names are matched tolerantly (``analyse._teams_match``) with the
    team-alias canonical spelling as a first-class variant;
  - the competition is validated against the requested league
    (``competition_league_key`` + tolerant containment);
  - among multiple similar fixtures the strongest combination wins:
    exact competition > exact team names > today before tomorrow.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

from .analyse import (  # noqa: E402
    _norm_team_name,
    _teams_match,
    find_specific_match,
    resolve_or_detect_league,
)
from .league_resolver import competition_league_key, resolve_league_scored  # noqa: E402
from .livescore import (  # noqa: E402
    LiveScoreClient,
    parse_form,
    parse_h2h,
    parse_lineups,
    parse_soccer_payload,
    parse_statistics,
)
from .timeutil import (  # noqa: E402
    utc_range_for_wib_date,
    wib_date_from_iso,
    wib_today_iso,
)

SOURCE_LABELS = {"livescore": "LiveScore", "flashscore": "Flashscore"}

DATE_FEED_TTL_SECONDS = 900  # matches livescore.DATE_FEED_TTL_SECONDS (15 min)


def _name_variants(name: str) -> list[str]:
    """Candidate spellings for one side: team-alias canonical name first."""
    from .team_alias import resolve_team_alias

    aliased = resolve_team_alias(name, None)
    if aliased and aliased != name:
        return [aliased, name]
    return [name]


def _pair_matches(home: str, away: str, home_cand: str, away_cand: str) -> bool:
    """Home/away pair matches the requested pair, in either order."""
    return (_teams_match(home, home_cand) and _teams_match(away, away_cand)) or (
        _teams_match(home, away_cand) and _teams_match(away, home_cand)
    )


def _competition_matches(comp: str | None, league_key: str, display: str) -> bool:
    """True when a source competition title belongs to the requested league.

    Registered titles map through ``competition_league_key`` (alias index,
    prefix-tolerant). Unregistered spellings fall back to tolerant normalized
    containment against the league display name so e.g. a feed stage named
    "Spain - LaLiga" still validates for La Liga.
    """
    if not comp:
        return False
    try:
        key = competition_league_key(comp)
        if key and key == league_key:
            return True
    except Exception:  # noqa: BLE001 -- competition mapping must never raise
        pass
    nc = _norm_team_name(comp)
    nd = _norm_team_name(display)
    if not nc or not nd:
        return False
    return nc == nd or nc in nd or nd in nc


def _score_candidate(
    fx: dict[str, Any],
    league_key: str,
    display: str,
    home_variants: list[str],
    away_variants: list[str],
) -> int | None:
    """Score one source fixture, or None when it is not the requested match.

    Both teams must match (tolerantly, either order) AND the competition must
    belong to the requested league -- a cup/friendly between the same two
    teams is never selected. Exact normalized team names score higher than
    containment hits so the strongest combination wins.
    """
    home = fx.get("home")
    away = fx.get("away")
    if not home or not away:
        return None
    if not any(
        _pair_matches(home, away, hc, ac)
        for hc in home_variants
        for ac in away_variants
    ):
        return None
    if not _competition_matches(fx.get("competition"), league_key, display):
        return None
    score = 0
    for hc in home_variants:
        if _norm_team_name(home) == _norm_team_name(hc):
            score += 2
            break
    for ac in away_variants:
        if _norm_team_name(away) == _norm_team_name(ac):
            score += 2
            break
    return score


# --------------------------------------------------------------------------
# LiveScore search (verified lsmedia1.com public API, no key)
# --------------------------------------------------------------------------


async def _cached_date_feed(client: LiveScoreClient, cache: Any, date8: str, page: int) -> dict[str, Any] | None:
    """Date feed with the same cache key shape as ``LiveScoreDataSource``."""
    key = f"livescore_date_{date8}_{page}"
    if cache is not None:
        hit = cache.get(key, ttl_seconds=DATE_FEED_TTL_SECONDS)
        if hit is not None:
            return hit
    payload = await client.fetch_soccer_date(date8, page)
    if payload is not None and cache is not None:
        cache.set(key, payload)
    return payload


def _wib_dates_today_tomorrow() -> list[str]:
    today = wib_today_iso()
    tomorrow = (
        datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    return [today, tomorrow]


def _utc_feed_dates_today_then_tomorrow() -> list[str]:
    """UTC YYYYMMDD feed dates covering today+tomorrow WIB, today first.

    One WIB calendar day spans two UTC dates (WIB = UTC+7), so today's WIB
    window needs both UTC dates; tomorrow's window is appended after it.
    """
    out: list[str] = []
    for wib_date in _wib_dates_today_tomorrow():
        for utc in utc_range_for_wib_date(wib_date):
            d8 = utc.replace("-", "")
            if d8 not in out:
                out.append(d8)
    return out


async def fetch_finished_livescore_results(
    cfg: dict[str, Any],
    cache: Any,
    date: str,
    *,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Finished soccer results for one UTC date from the LiveScore feed.

    Returns [{home, away, home_goals, away_goals, competition, country},
    ...] for every event on ``date`` (YYYY-MM-DD) whose status is finished
    and whose score is present. Used by ``settle auto`` as a no-key source
    of real results -- football-data.org returned ZERO finished matches for
    some dates (data lag), while LiveScore carries every day's results.
    Pages beyond the first are scanned when ``max_pages`` is given; the
    config default (``data_sources.livescore.max_pages``) applies otherwise.
    Empty result when livescore is disabled or unreachable -- the caller
    falls back to another source, never fabricates a score.
    """
    ds_cfg = (cfg.get("data_sources") or {}).get("livescore") or {}
    if not ds_cfg.get("enabled", True):
        return []
    client = LiveScoreClient(base_url=ds_cfg.get("base_url") or None)
    pages = max(1, int(max_pages if max_pages is not None else ds_cfg.get("max_pages", 3)))
    date8 = date.replace("-", "")
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for page in range(pages):
        payload = await _cached_date_feed(client, cache, date8, page)
        for fx in parse_soccer_payload(payload):
            if fx.get("status") != "finished":
                continue
            sc = fx.get("score") or {}
            hg, ag = sc.get("home"), sc.get("away")
            if hg is None or ag is None:
                continue
            key = (fx.get("home") or "", fx.get("away") or "")
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "home": fx.get("home"),
                "away": fx.get("away"),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "competition": fx.get("competition"),
                "country": fx.get("country"),
                # G2: livescore's own team ids + event id, carried through so
                # settle_auto can verify a result against the snapshot's
                # canonical entities instead of relying on name alone.
                "home_id": fx.get("home_id"),
                "away_id": fx.get("away_id"),
                "source_id": fx.get("source_id"),
            })
    return out


async def _search_livescore(
    cfg: dict[str, Any],
    cache: Any,
    league_key: str,
    display: str,
    home: str,
    away: str,
) -> dict[str, Any] | None:
    """Find the requested match in the LiveScore date feeds (today -> tomorrow).

    Scans every UTC feed date covering today WIB first, then tomorrow WIB.
    Every candidate must match the league AND both teams; among candidates
    the strongest (exact competition + exact names) wins, ties broken by
    earlier date. Returns the resolved fixture identity + source match id,
    or None when the match is not scheduled today/tomorrow.
    """
    ds_cfg = (cfg.get("data_sources") or {}).get("livescore") or {}
    if not ds_cfg.get("enabled", True):
        return None
    client = LiveScoreClient(base_url=ds_cfg.get("base_url") or None)
    max_pages = max(1, int(ds_cfg.get("max_pages", 3)))
    home_variants = _name_variants(home)
    away_variants = _name_variants(away)
    today, tomorrow = _wib_dates_today_tomorrow()

    best: tuple[tuple[int, int], dict[str, Any]] | None = None
    # Extended window: search up to 7 days ahead for upcoming fixtures.
    # Today+tomorrow use the standard UTC feed dates; beyond that, scan
    # each day individually (LiveScore API supports any YYYYMMDD date).
    from datetime import timedelta as _td
    _feed_dates: list[str] = list(_utc_feed_dates_today_then_tomorrow())
    _now_utc = datetime.now(timezone.utc)
    for _day_offset in range(2, 7):
        _future = (_now_utc + _td(days=_day_offset)).strftime("%Y%m%d")
        if _future not in _feed_dates:
            _feed_dates.append(_future)
    for date_index, d8 in enumerate(_feed_dates):
        for page in range(max_pages):
            payload = await _cached_date_feed(client, cache, d8, page)
            for fx in parse_soccer_payload(payload):
                kick_wib = wib_date_from_iso(fx.get("kickoff"))
                # For extended dates (>tomorrow), accept any match on that day.
                _today_tmrw = _wib_dates_today_tomorrow()
                if kick_wib not in _today_tmrw and date_index < len(_utc_feed_dates_today_then_tomorrow()):
                    continue
                score = _score_candidate(
                    fx, league_key, display, home_variants, away_variants
                )
                if score is None:
                    continue
                # Strength first; ties go to the EARLIER searched day (today
                # before tomorrow), so negate the date index.
                key = (score, -date_index)
                if best is None or key > best[0]:
                    best = (key, fx)
    if best is None:
        return None
    _, fx = best
    return {
        "source": "livescore",
        "home": fx["home"],
        "away": fx["away"],
        "home_id": fx.get("home_id"),
        "away_id": fx.get("away_id"),
        "kickoff": fx.get("kickoff"),  # ISO UTC
        "date": wib_date_from_iso(fx.get("kickoff")),
        "competition": fx.get("competition"),
        "status": fx.get("status"),
        "score": fx.get("score"),
        "source_id": fx.get("source_id"),
    }


async def _search_livescore_any(
    stats: Any,
    home: str,
    away: str,
) -> dict[str, Any] | None:
    """Find ``home vs away`` in the LiveScore date feeds WITHOUT a league key.

    D2 (dynamic league discovery): used when the league keyword is unknown /
    absent -- the fixture is found by team pair only, and its competition
    title becomes the dynamic league. Scans today -> tomorrow WIB feed dates
    (same window as ``_search_livescore``), both pages, and returns the
    STRONGEST pair match (exact normalized names beat containment). Returns
    the fixture identity (home/away/competition/kickoff/source_id) or None.
    """
    client = getattr(stats, "livescore", None)
    cache = getattr(stats, "cache", None)
    # Guard: the client must be a REAL LiveScore client (has the fetch
    # method); mocks / absent clients degrade to undetected, never crash.
    if client is None or not callable(getattr(client, "fetch_soccer_date", None)):
        return None
    if not getattr(client, "available", False):
        return None
    try:
        from .livescore import parse_soccer_payload

        home_variants = _name_variants(home)
        away_variants = _name_variants(away)
        today, tomorrow = _wib_dates_today_tomorrow()
        best: tuple[int, dict[str, Any]] | None = None
        # Extended window: scan up to 7 days ahead for upcoming fixtures.
        _feed_dates: list[str] = list(_utc_feed_dates_today_then_tomorrow())
        _now_utc = datetime.now(timezone.utc)
        for _day_offset in range(2, 7):
            _future = (_now_utc + timedelta(days=_day_offset)).strftime("%Y%m%d")
            if _future not in _feed_dates:
                _feed_dates.append(_future)
        _today_tmrw = _wib_dates_today_tomorrow()
        for date_index, d8 in enumerate(_feed_dates):
            for page in (0, 1):
                payload = await _cached_date_feed(client, cache, d8, page)
                for fx in parse_soccer_payload(payload):
                    kick_wib = wib_date_from_iso(fx.get("kickoff"))
                    # For extended dates (>tomorrow), accept any match on that day.
                    if kick_wib not in _today_tmrw and date_index < len(_utc_feed_dates_today_then_tomorrow()):
                        continue
                    if not any(
                        _pair_matches(fx.get("home") or "", fx.get("away") or "", hc, ac)
                        for hc in home_variants
                        for ac in away_variants
                    ):
                        continue
                    score = 0
                    if _norm_team_name(fx.get("home") or "") == _norm_team_name(home_variants[0]):
                        score += 2
                    if _norm_team_name(fx.get("away") or "") == _norm_team_name(away_variants[0]):
                        score += 2
                    key = (score, -date_index)
                    if best is None or key > best[0]:
                        best = (key, fx)
        if best is None:
            return None
        _, fx = best
        return {
            "source": "livescore",
            "home": fx["home"],
            "away": fx["away"],
            "home_id": fx.get("home_id"),
            "away_id": fx.get("away_id"),
            "kickoff": fx.get("kickoff"),
            "competition": fx.get("competition"),
            "country": fx.get("country"),
            "status": fx.get("status"),
            "score": fx.get("score"),
            "source_id": fx.get("source_id"),
        }
    except Exception as exc:  # noqa: BLE001 -- detection must never raise
        logger.warning("livescore any-league search failed: %s", type(exc).__name__)
        return None


async def _collect_livescore_data(client: LiveScoreClient, found: dict[str, Any]) -> dict[str, Any]:
    """Best-effort field collection from the resolved LiveScore event id.

    Lineups / H2H / form / statistics come from the verified per-event
    endpoints parsed by the pure livescore parsers. Any failure degrades to
    omitting the field -- never fabricated.
    """
    eid = found.get("source_id")
    out: dict[str, Any] = {}
    if not eid:
        return out
    tasks = (
        ("lineups", client.fetch_lineups(eid), parse_lineups),
        ("h2h", client.fetch_h2h(eid), lambda p: parse_h2h(p, found)),
        ("form", client.fetch_form(eid), parse_form),
        ("statistics", client.fetch_statistics(eid), parse_statistics),
    )
    for key, coro, parse in tasks:
        try:
            raw = await coro
            parsed = parse(raw)
            if parsed:
                out[key] = parsed
        except Exception as exc:  # noqa: BLE001 -- context is best-effort
            logger.warning("livescore %s collect failed (best-effort): %s", key, type(exc).__name__)
    return out


# --------------------------------------------------------------------------
# Flashscore search (browser client, existing resolve_match logic)
# --------------------------------------------------------------------------


def _flashscore_date(date_text: str | None) -> str | None:
    """Best-effort WIB date from a flashscore row's date text.

    Handles "Today 21:00" / "Tomorrow 15:30" / "20.02.2026 16:00"; a bare
    kickoff time (league-page rows are today-scoped) resolves to today.
    Returns None when the text carries no usable date.
    """
    t = (date_text or "").strip()
    if not t:
        return None
    low = t.lower()
    today, tomorrow = _wib_dates_today_tomorrow()
    if "today" in low:
        return today
    if "tomorrow" in low:
        return tomorrow
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", t)
    if m:
        try:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        except ValueError:
            return None
    if re.fullmatch(r"\d{1,2}:\d{2}", t):
        return today  # league-page rows carry only the kickoff time (today)
    return None


async def _search_flashscore(
    stats: Any,
    league_key: str,
    home: str,
    away: str,
) -> dict[str, Any] | None:
    """Find the requested match via the existing flashscore resolve logic.

    ``FlashscoreClient.resolve_match`` already searches today first (league
    page, then homepage) and falls back to the team-fixtures pages for
    upcoming fixtures (tomorrow and beyond) -- the today -> tomorrow order
    the spec requires, using the existing tested resolution.
    """
    fc = getattr(stats, "fc", None)
    if fc is None or not getattr(fc, "available", True):
        return None
    try:
        resolved = await fc.resolve_match(league_key, home, away)
    except Exception as exc:  # noqa: BLE001 -- search must fail soft
        logger.warning("flashscore resolve failed (match search): %s", exc)
        return None
    if not resolved:
        return None
    home_t = resolved.get("home") or {}
    away_t = resolved.get("away") or {}
    h_name = home_t.get("name") or home
    a_name = away_t.get("name") or away
    return {
        "source": "flashscore",
        "home": h_name,
        "away": a_name,
        "home_slug": home_t.get("slug"),
        "home_id": home_t.get("id"),
        "away_slug": away_t.get("slug"),
        "away_id": away_t.get("id"),
        "match_url": resolved.get("match_url"),
        "date_text": resolved.get("date_text"),
        "date": _flashscore_date(resolved.get("date_text")),
        "status": "scheduled",
        "score": resolved.get("score"),
    }


async def _collect_flashscore_data(stats: Any, found: dict[str, Any]) -> dict[str, Any]:
    """Best-effort match context from the resolved flashscore match page."""
    fc = getattr(stats, "fc", None)
    out: dict[str, Any] = {}
    if fc is None:
        return out
    match_url = found.get("match_url")
    if match_url:
        for key, coro in (
            ("statistics", fc.fetch_match_statistics(match_url)),
            ("lineups", fc.fetch_match_lineups(match_url)),
            ("h2h", fc.fetch_match_h2h(match_url, found["home"], found["away"])),
            ("match_info", fc.fetch_match_info(match_url)),
        ):
            try:
                value = await coro
                if value:
                    out[key] = value
            except Exception as exc:  # noqa: BLE001 -- context is best-effort
                logger.warning("flashscore %s collect failed (best-effort): %s", key, type(exc).__name__)
    if found.get("home_slug") and found.get("home_id"):
        try:
            form = await fc.fetch_team_form(found["home_slug"], found["home_id"])
            if form:
                out.setdefault("form", {})["home"] = form
        except Exception as exc:  # noqa: BLE001
            logger.warning("flashscore home form collect failed (best-effort): %s", type(exc).__name__)
    if found.get("away_slug") and found.get("away_id"):
        try:
            form = await fc.fetch_team_form(found["away_slug"], found["away_id"])
            if form:
                out.setdefault("form", {})["away"] = form
        except Exception as exc:  # noqa: BLE001
            logger.warning("flashscore away form collect failed (best-effort): %s", type(exc).__name__)
    return out


# --------------------------------------------------------------------------
# Entry point (shared by both commands)
# --------------------------------------------------------------------------


async def find_source_match(
    *,
    source: str,
    league_query: str,
    home_query: str,
    away_query: str,
    cfg: dict[str, Any],
    odds: Any,
    stats: Any,
    cache: Any,
    oddspapi: Any = None,
    nowgoal: Any = None,
) -> dict[str, Any]:
    """Run the full match-source command flow for one source.

    Returns the SAME dict shape as ``find_specific_match`` (the existing
    analyse pipeline) so the existing formatters render it unchanged, with
    ``source_match`` + ``match_source`` attached for provenance. When the
    match cannot be found today/tomorrow the pipeline is NOT invoked and a
    clear "match not found" error is returned instead.
    """
    if source not in SOURCE_LABELS:
        return {"error": f"source tidak dikenal: {source}"}
    label = SOURCE_LABELS[source]
    # D2 (dynamic league discovery): unknown / absent league keyword is
    # resolved FROM THE FIXTURE via the flashscore homepage / livescore feed
    # (resolve_or_detect_league). ``detected`` carries the fixture identity;
    # ``league_key`` may be a ``dyn:`` key for unregistered competitions.
    _resolved = await resolve_or_detect_league(
        league_query=league_query,
        home_query=home_query,
        away_query=away_query,
        stats=stats,
    )
    if _resolved is None:
        return {
            "error": f"liga '{league_query}' tidak dikenal",
            "home_query": home_query,
            "away_query": away_query,
        }
    league_key, meta, detected = _resolved
    display = meta["display"]

    # Shared analysis budget so provider fallbacks inside the pipeline (and
    # the source search itself) skip expensive steps near the runner deadline.
    from .multi_source import set_analysis_budget

    set_analysis_budget(float((cfg.get("analyse") or {}).get("budget_seconds", 72.0)))

    # ---- 1) Match search: today first, then tomorrow ----------------------
    if detected:
        # The league was read FROM the fixture -- reuse that identity.
        found = detected
    elif source == "livescore":
        found = await _search_livescore(cfg, cache, league_key, display, home_query, away_query)
    else:
        found = await _search_flashscore(stats, league_key, home_query, away_query)

    # ---- 6) Match not found: clear response, pipeline never runs ----------
    if not found:
        return {
            "error": (
                f"Match '{home_query} vs {away_query}' tidak ditemukan di {label} "
                f"({display}, hari ini / besok). Pastikan nama tim dan liga benar."
            ),
            "league": display,
            "home_query": home_query,
            "away_query": away_query,
        }

    # ---- 5) Collect the source's match data (best-effort) -----------------
    if source == "livescore":
        ds_cfg = (cfg.get("data_sources") or {}).get("livescore") or {}
        client = LiveScoreClient(base_url=ds_cfg.get("base_url") or None)
        source_data = await _collect_livescore_data(client, found)
    else:
        source_data = await _collect_flashscore_data(stats, found)

    source_match: dict[str, Any] = {**found, **source_data}
    source_match["source"] = source
    source_match["league_key"] = league_key

    # ---- 7/8/10) EXISTING pipeline: NowGoal odds -> prediction -> output ---
    # The validated identity (source home/away names + kickoff) is handed to
    # the existing analyse pipeline; it runs the NowGoal odds lookup (existing
    # oddspapi -> nowgoal -> The Odds API priority), the prediction engine,
    # the decision engine and the existing output formatting. No duplicated
    # prediction logic.
    return await find_specific_match(
        league_query=league_key,
        home_query=found["home"],
        away_query=found["away"],
        cfg=cfg,
        odds=odds,
        stats=stats,
        cache=cache,
        oddspapi=oddspapi,
        nowgoal=nowgoal,
        source_match=source_match,
        league_key=league_key,
        league_meta=meta,
    )


# ---------------------------------------------------------------------------
# Match-status reconciliation (P0-2)
# ---------------------------------------------------------------------------
# Goal: ``analyse.match_finished`` was derived from kickoff < now alone. That
# is correct for finished matches but it FLIPS a "live" match to "finished"
# the moment live coverage disagrees within the live window (e.g. flashscore
# pushes a stale terminal status while livescore still reports "live"). The
# analysis then bails out without producing a signal, even though the match
# is in play and the snapshot is still usable.
#
# This helper takes the union of (a) live source-status strings and
# (b) the kickoff-time window, and returns the unified status:
#   "scheduled" | "live" | "finished" | "unknown"
#
# Rules (kept conservative -- when sources disagree we prefer the most
# "in-doubt" verdict so the pipeline never declares a live match finished):
#   1. live_source says "live" AND now in [kickoff - 15m, kickoff + 4h]
#      -> "live" (regardless of flashscore stale-finished flag)
#   2. live_source says "finished" AND now > kickoff + 4h
#      -> "finished"
#   3. now < kickoff - 15m
#      -> "scheduled"
#   4. now >= kickoff + 4h AND no source says live
#      -> "finished"
#   5. otherwise (e.g. kickoff is in the past but within the live window
#      and no source says live)
#      -> "live" (the match almost certainly is in play)
#   6. nothing usable
#      -> "unknown"

_LIVE_WINDOW_BEFORE_MIN = 15
_LIVE_WINDOW_AFTER_H = 4


def reconcile_status(
    kickoff: datetime | None,
    now: datetime | None,
    flashscore_status: str | None = None,
    livescore_status: str | None = None,
) -> str:
    """Return unified status: "scheduled" | "live" | "finished" | "unknown"."""
    if kickoff is None or now is None:
        return "unknown"
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    pre = kickoff - timedelta(minutes=_LIVE_WINDOW_BEFORE_MIN)
    post = kickoff + timedelta(hours=_LIVE_WINDOW_AFTER_H)
    ls = (livescore_status or "").lower().strip()
    fs = (flashscore_status or "").lower().strip()
    ls_live = ls in ("live", "in_play", "1h", "2h", "ht", "et", "pen")
    ls_finished = ls in ("finished", "ft", "aet", "after_extra_time")
    fs_live = fs in ("live", "in_play")
    fs_finished = fs in ("finished", "ft")
    if ls_live and pre <= now <= post:
        return "live"
    if pre > now:
        return "scheduled"
    if now >= post and not ls_live and not fs_live:
        return "finished"
    if ls_finished and now >= post:
        return "finished"
    if fs_finished and now >= post and not ls_live:
        return "finished"
    if pre <= now <= post:
        return "live"
    return "unknown"
