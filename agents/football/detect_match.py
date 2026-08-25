"""League auto-detect for `analisa match <home> vs <away>` (no league keyword).

Three-stage scan, cheapest/broadest-precision first:
  1. football-data /v4/matches (one call) -- scheduled fixtures across the
     competitions its global feed covers; the competition code maps back to
     a registered league key.
  2. thesportsdb next-fixtures -- covers registered-league matches weeks
     away (e.g. an EPL tie) that the global feed / homepage do not carry.
  3. flashscore homepage (today, cached like `!football today`) -- friendlies,
     cups, qualifiers. A registered competition returns the league key; an
     unregistered one is surfaced as info-only (the bot shows the fixture
     without analysis).

Returns a plain dict -- the runner just forwards it.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

from .analyse import _norm_team_name, _teams_match  # noqa: E402
from .league_resolver import competition_league_key, load_leagues  # noqa: E402
from .match_finder import _is_upcoming  # noqa: E402
from .team_alias import resolve_team_alias  # noqa: E402

DETECT_WINDOW_DAYS = 7


def _fixture_to_result(found: dict[str, Any], source: str) -> dict[str, Any]:
    """Normalize a Livescore/Flashscore/team-fixtures hit into the detect
    contract: registered competition -> league key, otherwise -> info-only
    payload (`registered: False`) so the bot enters the existing D2 path
    (`bot.py:1822-1836`) and runs the full analysis under a ``dyn:`` key.

    Reused by stage 1 (livescore feed), stage 3 (homepage), and stage 4
    (flashscore team-fixtures) so every source feeds the same output shape.
    """
    comp = str(found.get("competition") or "").strip()
    if not comp:
        return {"found": False}
    key = competition_league_key(comp)
    if key:
        leagues = load_leagues()
        meta = leagues.get(key) or {}
        return {
            "found": True,
            "league": key,
            "display": meta.get("display", key),
            "home": found.get("home") or "",
            "away": found.get("away") or "",
            "kickoff": found.get("kickoff"),
            "competition": comp,
            "source": source,
        }
    return {
        "found": True,
        "registered": False,
        "competition": comp,
        "home": found.get("home") or "",
        "away": found.get("away") or "",
        "kickoff": found.get("kickoff"),
        "source": source,
    }


def _code_to_league_key(leagues: dict[str, dict] | None = None) -> dict[str, str]:
    """football-data competition code -> registered league key (reverse map)."""
    out: dict[str, str] = {}
    for key, meta in (leagues if leagues is not None else load_leagues()).items():
        code = meta.get("football_data_code")
        if code:
            out[str(code)] = key
    return out


def _name_variants(name: str) -> list[str]:
    """Candidate spellings for one side: team-alias canonical name first."""
    aliased = resolve_team_alias(name, None)
    if aliased and aliased != name:
        return [aliased, name]
    return [name]


def _loose(a: str, b: str) -> bool:
    """Tolerant containment for short names ('Leeds' vs 'Leeds United')."""
    if _teams_match(a, b):
        return True
    na, nb = _norm_team_name(a), _norm_team_name(b)
    if not na or not nb or len(na) < 4 or len(nb) < 4:
        return False
    return na in nb or nb in na


def _lev_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1] (1 = identical)."""
    na, nb = _norm_team_name(a), _norm_team_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    la, lb = len(na), len(nb)
    if min(la, lb) < 4:
        return 0.0  # too short to judge a typo reliably
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if na[i - 1] == nb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def _typo_match(a: str, b: str) -> bool:
    """True when a and b are the same side under a small edit (1 typo / 1
    missing char) -- e.g. 'Ilven' vs 'Ilves' (verified 2026-08-17). The
    stricter tiers (_teams_match / _loose) already ran; this tier only needs
    to detect the near-equality itself. Strict guard: >= 4 chars and
    similarity >= 0.8, so a close-but-different club name ('Real Betis' vs
    'Real Madrid') can never pass."""
    sim = _lev_sim(a, b)
    return sim >= 0.8


def _find_pair_in_rows(
    rows: list[dict[str, Any]], home: str, away: str
) -> dict[str, Any] | None:
    """Locate the fixture in rows; strict _teams_match pass first, then loose
    containment, so a short token ('real') cannot latch onto the first
    'Real ...' row before an exact pair later in the list."""
    h_cands = _name_variants(home)
    a_cands = _name_variants(away)

    def _pair(h: str, a: str, hc: str, ac: str) -> bool:
        return (_teams_match(h, hc) and _teams_match(a, ac)) or (
            _teams_match(h, ac) and _teams_match(a, hc)
        )

    for r in rows:
        h, a = r.get("home"), r.get("away")
        if not h or not a:
            continue
        for hc in h_cands:
            for ac in a_cands:
                if _pair(h, a, hc, ac):
                    return r
    # Loose pass: containment is the LEAST strict matcher, so the first hit
    # is not trustworthy -- a short query ("real") can latch onto the wrong
    # "Real ..." row after the strict pass missed the real pair. Like the
    # standings matcher's containment tier, the loose pass returns only when
    # EXACTLY ONE row matches; an ambiguous multiple-hit is None (B9,
    # verified 2026-08-17: previously the first loose hit won silently).
    loose_hits: list[dict[str, Any]] = []
    for r in rows:
        h, a = r.get("home"), r.get("away")
        if not h or not a:
            continue
        for hc in h_cands:
            for ac in a_cands:
                if (_loose(h, hc) and _loose(a, ac)) or (
                    _loose(h, ac) and _loose(a, hc)
                ):
                    loose_hits.append(r)
                    break
            else:
                continue
            break
    if len(loose_hits) == 1:
        return loose_hits[0]
    # Typo pass (2026-08-17): a user typo ('Ilven' vs 'Ilves') must not
    # hard-fail the whole detect. One side may be an exact match while the
    # other is a typo ('Gnistan' exact + 'Ilven'->'Ilves'), so each side is
    # checked independently: match = strict/loose OR typo. Same ambiguity
    # guard as the loose tier: return only when EXACTLY ONE row matches; a
    # multi-hit (e.g. 'Athletic' near 'Ath Bilbao' AND 'Athletico
    # Paranaense') stays None.
    def _side_ok(row_name: str, cand: str) -> bool:
        return _teams_match(row_name, cand) or _loose(row_name, cand) or _typo_match(row_name, cand)

    typo_hits: list[dict[str, Any]] = []
    for r in rows:
        h, a = r.get("home"), r.get("away")
        if not h or not a:
            continue
        for hc in h_cands:
            for ac in a_cands:
                if (_side_ok(h, hc) and _side_ok(a, ac)) or (
                    _side_ok(h, ac) and _side_ok(a, hc)
                ):
                    typo_hits.append(r)
                    break
            else:
                continue
            break
    if len(typo_hits) == 1:
        return typo_hits[0]
    return None


def _row_flat(r: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a homepage row (home/away may be dicts or bare names)."""

    def _nm(x: Any) -> Any:
        return x.get("name") if isinstance(x, dict) else x

    h, a = _nm(r.get("home")), _nm(r.get("away"))
    if not h or not a:
        return None
    return {
        "home": h,
        "away": a,
        "competition": r.get("competition") or "Other",
        "kickoff": r.get("kickoff") or r.get("date_text"),
        "status": r.get("status"),
    }


async def _thesportsdb_detect(stats: Any, home: str, away: str) -> dict[str, Any] | None:
    """thesportsdb: search both teams, then their next fixtures, and look for
    the shared fixture. No date window -- works for a match weeks away that
    football-data's global feed / today's homepage do not carry. Returns the
    registered-league hit or None.

    Competition -> league key via ``competition_league_key`` (prefix,
    longest-first) instead of ``resolve_league``'s loose substring match,
    which mis-resolved titles like "Trophée des Champions" (a cup) onto UCL
    or "La Liga 2" onto La Liga (2026-08-17).
    """

    try:
        for hq, aq in ((home, away), (away, home)):
            team = await stats.ts.search_team(_name_variants(hq)[0])
            if not team or not team.get("idTeam"):
                continue
            events = await stats.ts.fetch_next_matches(str(team["idTeam"]), limit=10) or []
            rows = []
            for ev in events:
                eh, ea = ev.get("strHomeTeam"), ev.get("strAwayTeam")
                if not eh or not ea:
                    continue
                kick = " ".join(
                    str(ev.get(k) or "") for k in ("dateEvent", "strTime") if ev.get(k)
                ).strip()
                rows.append({
                    "home": eh,
                    "away": ea,
                    "kickoff": kick or None,
                    "competition": ev.get("strLeague") or "",
                })
            found = _find_pair_in_rows(rows, home, away)
            if found:
                key = competition_league_key(str(found["competition"]))
                if key:
                    meta = load_leagues()[key]
                    return {
                        "found": True,
                        "league": key,
                        "display": meta["display"],
                        "home": found["home"],
                        "away": found["away"],
                        "kickoff": found["kickoff"],
                        "competition": found["competition"],
                        "source": "thesportsdb",
                    }
    except Exception as exc:  # noqa: BLE001 -- detection must never raise
        logger.warning("detect thesportsdb scan failed: %s", exc)
    return None


async def _homepage_rows(stats: Any, cache: Any) -> list[dict[str, Any]]:
    """Today's flashscore homepage rows, using the same cache key + TTL as
    `!football today` so a warm top run serves the detect for free."""
    from .timeutil import wib_today_iso

    # Cache key version matches `match_finder.py:414` (`_v3`); the homepage
    # scraper now returns rows with `status` so this detect reads the same
    # warm cache as `!football today` / `top`.
    hp_key = f"flashscore_homepage_{wib_today_iso()}_v3"
    if cache is not None:
        cached = cache.get(hp_key, ttl_seconds=1200)
        if cached is not None:
            return cached
    raw = await stats.fetch_homepage_matches() or []
    if raw and cache is not None:
        cache.set(hp_key, raw)
    return raw


async def detect_league_match(
    *,
    home: str,
    away: str,
    stats: Any,
    cache: Any = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Find which registered league (if any) hosts `home vs away`.

    Returns:
      found + league/display/home/away/kickoff/source     -> full analysis
      found + registered=False/competition/...            -> info-only (bot)
      found=False                                         -> nothing matched
    """
    today = datetime.now(timezone.utc).date()
    if date:
        try:
            start = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            start = today
    else:
        start = today
    end = start + timedelta(days=DETECT_WINDOW_DAYS - 1)

    # 1) football-data: one call, all competitions, registered leagues.
    try:
        rows = await stats.fd.fetch_scheduled_matches_by_date(
            start.isoformat(), end.isoformat()
        )
        if rows:
            found = _find_pair_in_rows(rows, home, away)
            if found:
                key = _code_to_league_key().get(found["competition"])
                if key:
                    return {
                        "found": True,
                        "league": key,
                        "display": load_leagues()[key]["display"],
                        "home": found["home"],
                        "away": found["away"],
                        "kickoff": found["kickoff"],
                        "competition": found["competition"],
                        "source": "football_data",
                    }
    except Exception as exc:  # noqa: BLE001 -- detection must never raise
        logger.warning("detect football-data scan failed: %s", exc)

    # 2) livescore date feed (cheap HTTP, no browser, cached 15 min):
    #    carries every competition in one feed page per UTC date and tags
    #    each row with `competition` + `country`, so a friendly/cup/ASEAN
    #    playoff row resolves with its real competition title. Goes BEFORE
    #    thesportsdb because the feed is one request + the cache stays warm
    #    from settle-auto / today scans; thesportsdb is two requests.
    try:
        from .source_match import _search_livescore_any

        ls_hit = await _search_livescore_any(stats, home, away)
        if ls_hit:
            out = _fixture_to_result(ls_hit, "livescore")
            if out.get("found"):
                return out
    except Exception as exc:  # noqa: BLE001 -- detection must never raise
        logger.warning("detect livescore feed scan failed: %s", exc)

    # 3) thesportsdb: next-fixtures scan -- covers registered-league matches
    #    (e.g. an EPL tie two weeks out) that neither the football-data global
    #    feed nor today's homepage carries.
    ts_hit = await _thesportsdb_detect(stats, home, away)
    if ts_hit:
        return ts_hit

    # 4) flashscore homepage: today's friendlies/cups/qualifiers + everything
    #    football-data does not cover.
    try:
        raw = await _homepage_rows(stats, cache)
        rows = [f for f in (r and _row_flat(r) for r in raw) if f]
        rows = [r for r in rows if _is_upcoming(r.get("status"), r.get("kickoff"))]
        found = _find_pair_in_rows(rows, home, away)
        if found:
            out = _fixture_to_result(found, "flashscore")
            if out.get("found"):
                return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("detect flashscore homepage scan failed: %s", exc)

    # 5) flashscore team-fixtures fallback (browser, budget-guarded): the
    #    suggest API + team-fixtures pages drop the pair into any competition
    #    the team has scheduled (verified already used by `analyse.resolve_
    #    or_detect_league` -- `analyse.py:1388`). ``resolve_match(None, ..)``
    #    walks the homepage/team-fixtures chain so the pair is found even
    #    when the competition is missing from today's homepage (e.g. a
    #    friendly several days out, or a cup the homepage simply does not
    #    display). The team-fixtures page DOES carry the competition tag
    #    (`flashscore.py:749-848`), so even unregistered competitions come
    #    back with a real competition title -> flows into the same D2 path.
    try:
        fc = getattr(stats, "fc", None)
        if fc is not None and getattr(fc, "available", True):
            resolved = await fc.resolve_match(None, home, away)
            if isinstance(resolved, dict):
                home_team = resolved.get("home") or {}
                away_team = resolved.get("away") or {}
                h_name = home_team.get("name") if isinstance(home_team, dict) else None
                a_name = away_team.get("name") if isinstance(away_team, dict) else None
                comp = (resolved.get("competition") or "").strip()
                if h_name and a_name and comp:
                    out = _fixture_to_result(
                        {
                            "home": h_name,
                            "away": a_name,
                            "competition": comp,
                            "kickoff": resolved.get("date_text"),
                        },
                        "flashscore",
                    )
                    if out.get("found"):
                        return out
    except Exception as exc:  # noqa: BLE001 -- detection must never raise
        logger.warning("detect flashscore team-fixtures scan failed: %s", exc)

    return {"found": False}
