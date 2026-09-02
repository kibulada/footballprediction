"""Multi-source stats router.

Provider chain resolves per league:
  flashscore (primary) -> livescore -> FBref -> football-data.org -> thesportsdb
  (livescore moved to SECOND after flashscore: it is cheap, keyless and
  always returns the full last-5 window, so flashscore outages no longer
  strand the pipeline on football-data's thin 1-match form).
  soccerdata (FBref/Understat/WhoScored) when supported.

Flashscore is the primary live provider (2026-08): api.sofascore.* is
Cloudflare-blocked on this network while flashscore.com renders fine in the
UC browser and carries form, H2H, xG and fixtures. The remaining sofascore
fallback paths were REMOVED from the live path (2026-08): on this network
they only hung against the blocked API and burned the runner's deadline.

Each public method returns a normalized dict so caller doesn't care
which provider answered.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from .cache import Cache  # noqa: F401  used as type hint
from .entity_registry import canonical_team_id, registry as _entity_registry
from .flashscore import FlashscoreClient
from .football_data import FootballDataClient, FootballDataError
from .soccerdata_wrapper import SoccerDataWrapper
from .team_alias import resolve_team_alias
from .thesportsdb import TheSportsDbClient

logger = logging.getLogger(__name__)


def _safe_canonical(
    provider: str,
    provider_id: Any,
    league_key: str | None,
    name: str,
) -> str | None:
    """Register (provider, provider_id) -> canonical id, never raising.

    G1: the registry is additive and must never break the resolve path --
    any failure (disk, schema) degrades to the deterministic id computed
    in-memory, which is all the downstream verifier needs.
    """
    try:
        return _entity_registry().register(provider, provider_id, league_key, name)
    except Exception:  # noqa: BLE001 -- registry never breaks resolve
        try:
            return canonical_team_id(league_key, name)
        except Exception:  # noqa: BLE001
            return None


def _flashscore_local_to_utc_iso(
    yyyy: int,
    mm: int,
    dd: int,
    hh: int,
    mi: int,
) -> str:
    """Flashscore wall-clock (WIB) -> UTC ISO-8601 (Z).

    Flashscore renders every match time in the VISITOR'S browser timezone,
    and this bot's headless Chrome runs with the system timezone
    Asia/Jakarta (verified live 2026-08-17: ``Intl.DateTimeFormat().
    resolvedOptions().timeZone`` == "Asia/Jakarta", and a 17:00 UTC kickoff
    renders as 23:00 -- the +7h WIB offset). The old CET/CEST assumption
    (Europe/Madrid) shifted every flashscore kickoff by +5h (CEST) or +6h
    (CET); B10's month-DST refinement fixed the wrong clock entirely.
    Indonesia has no DST, so WIB is a constant +7h offset year-round.
    Returns "YYYY-MM-DDTHH:MM:00Z".
    """
    from zoneinfo import ZoneInfo

    dt = datetime(yyyy, mm, dd, hh, mi, tzinfo=ZoneInfo("Asia/Jakarta"))
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---- shared analysis time budget -------------------------------------------
#
# The Discord bot kills the runner at SUBPROCESS_TIMEOUT (380s) and the
# runner hard-exits at HERMES_RUNNER_DEADLINE (default 340s). The analyse
# pipeline chains
# several provider fallbacks (flashscore browser -> football-data ->
# thesportsdb -> soccerdata) and every blocked/slow provider used to eat its
# own full HTTP timeout before falling through, so the SUM of the chain
# could exceed the deadline and the user got a dead error. This module-level
# clock is the ONE shared budget every fetch consults: once it is nearly
# spent, optional/expensive steps (browser renders, slow fallbacks) are
# skipped and the caller returns a best-effort result instead of blowing the
# deadline. Set once per command via ``set_analysis_budget``; a fetch that
# runs without a configured budget behaves exactly as before.

ANALYSIS_BUDGET_DEFAULT = 72.0  # conservative; config overrides to 300.0 (< 340s runner deadline)

global _analysis_started
_analysis_started: float | None = None
_analysis_budget: float = ANALYSIS_BUDGET_DEFAULT


def set_analysis_budget(seconds: float) -> None:
    """(Re)start the analysis clock with a hard budget in seconds."""
    global _analysis_started, _analysis_budget
    _analysis_started = time.monotonic()
    _analysis_budget = max(1.0, float(seconds))


def reset_analysis_budget() -> None:
    """Disarm the analysis clock (no budget -> nothing is ever skipped).

    Used by tests so a clock armed by one test cannot bleed into the next;
    production arms the clock per command (fresh runner process per request).
    """
    global _analysis_started
    _analysis_started = None


def analysis_elapsed() -> float:
    """Seconds since the analysis clock started (0.0 when not armed)."""
    if _analysis_started is None:
        return 0.0
    return time.monotonic() - _analysis_started


def analysis_remaining() -> float | None:
    """Seconds of budget left, or None when no clock is armed (never bail)."""
    if _analysis_started is None:
        return None
    return max(0.0, _analysis_budget - analysis_elapsed())


def analysis_budget_exhausted(margin_seconds: float = 0.0) -> bool:
    """True when the shared budget is armed AND nearly spent.

    ``margin_seconds`` reserves a safety gap before the hard deadline (e.g.
    asking "do I still have 10s left?"). Returns False (keep going) when no
    clock is armed, so callers outside the runner keep old behaviour.
    """
    if _analysis_started is None:
        return False
    return analysis_elapsed() >= _analysis_budget - margin_seconds


def _timeout_aware(coro, seconds: float):
    """Run a coroutine with a hard timeout, returning None on timeout.

    Every wrapped call is a best-effort data fetch whose contract is "return
    data or None", so a slow/blocked provider degrading to None (instead of
    raising asyncio.TimeoutError that would have to be caught at every call
    site) keeps the provider chain moving and never crashes the analysis.
    """

    async def _run():
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            logger.warning("provider call timed out after %.0fs; degrading to None", seconds)
            return None

    return _run()


# Per-provider caps so ONE blocked provider cannot hang the whole chain.
# Each provider also has its own internal HTTP timeout; these are the outer
# bound that includes queuing/throttle sleeps, so the chain's worst case is
# bounded by (per-call cap x chain length), not by (network timeout x chain
# length).
_BUDGET_MARGIN = 8.0  # stay clear of the runner's hard deadline (default 340s)
_CALL_CAP = 12.0      # single provider call (incl. throttle sleep)


def _key_from_meta(league_meta: dict[str, Any]) -> str | None:
    """League key from a meta dict, dynamic-league aware (D4, 2026-08-17).

    1. explicit ``_league_key`` (registered path, existing);
    2. display matching a REGISTERED key (loaded from leagues.json, not the
       old hardcoded 16-item list -- same results for those 16, so no cache
       invalidation);
    3. dynamic fallback: an unregistered display resolves to a deterministic
       ``dyn:`` key so dynamic leagues flow through the form/h2h cache keys
       and the G1 entity registry instead of collapsing into "unknown".
    """
    explicit = league_meta.get("_league_key")
    if explicit:
        return explicit
    display = (league_meta.get("display") or "").strip().lower()
    if not display:
        return None
    try:
        from .league_resolver import load_leagues

        def _sq(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        target = _sq(display)
        for key, meta in load_leagues().items():
            # key itself ("laliga") or its display ("la liga" -> "laliga")
            if _sq(key) == target or _sq(meta.get("display") or "") == target:
                return key
    except Exception:  # noqa: BLE001 -- key detection must never raise
        pass
    try:
        from .league_resolver import dynamic_league_key

        return dynamic_league_key(display)
    except Exception:  # noqa: BLE001
        return None


def _form_depth(form: dict[str, Any] | None) -> int:
    """Number of finished matches in a form dict (any provider shape).

    ``sequence`` ("W-D-L") is the source of truth; ``recent_goals`` is the
    fallback. Used by the F1 thin-form logic: a form window shallower than
    the requested ``limit`` is treated as a fallback, not the final answer,
    so the chain keeps trying richer providers (LiveScore last).
    """
    seq = (form or {}).get("sequence")
    if isinstance(seq, str):
        return len([p for p in seq.split("-") if p])
    if isinstance(seq, (list, tuple)):
        return len(seq)
    rg = (form or {}).get("recent_goals")
    return len(rg) if isinstance(rg, (list, tuple)) else 0


class MultiSourceStatsFetcher:
    def __init__(
        self,
        football_data_key: str = "",
        thesportsdb_key: str = "",
        football_data_throttle: float = 6.0,
        thesportsdb_throttle: float = 1.1,
        flashscore_throttle: float = 1.5,
        soccerdata_dir: str = "cache/soccerdata",
        cache: "Cache | None" = None,
        proxy: str | None = None,
        flashscore_enabled: bool = True,
        understat_cache_dir: str = "cache/football",
        livescore_client: Any = None,
        flashscore_lanes: int = 1,
    ) -> None:
        self.fd = FootballDataClient(football_data_key, football_data_throttle)
        self.ts = TheSportsDbClient(thesportsdb_key, thesportsdb_throttle)
        self.fc = FlashscoreClient(throttle_seconds=flashscore_throttle) if flashscore_enabled else None
        # Lane B (optional): an INDEPENDENT browser session so two render
        # chains can genuinely overlap. The single _browser_lock on ``fc``
        # serializes every render, so asyncio.gather across flashscore-bound
        # work gains nothing without a second session. The driver itself is
        # LAZY (spawns on first render), so an unused lane costs nothing.
        # Used only by the xG history away-chain; every other phase stays on
        # the primary lane exactly as before.
        self.fc_secondary = (
            FlashscoreClient(throttle_seconds=flashscore_throttle)
            if flashscore_enabled and flashscore_lanes >= 2
            else None
        )
        self.sd = SoccerDataWrapper(soccerdata_dir, proxy=proxy)
        # Optional LiveScore client (verified no-key lsmedia1.com API) used
        # only as a LAST-RESORT form provider: when flashscore / FBref /
        # football-data / thesportsdb all fail to fill the form window, the
        # team's finished matches are rebuilt from the LiveScore date feed
        # (same source the settle path uses). None when livescore is
        # disabled in config -- the chain then ends at thesportsdb as before.
        self.livescore = livescore_client
        self.football_data_key = football_data_key
        self.cache = cache
        # Understat was REMOVED from the live xG chain (Plan B, 2026-08-17):
        # it only covered big-5, needed a browser session, and never auto-
        # rolled into the new season (disk-only guard). NowGoal (HTTP, all
        # leagues) is primary, flashscore is the fallback. The
        # ``understat_cache_dir`` constructor param is kept for backward
        # compatibility but is no longer used by any live path.
        # Lazy pure-HTTP flashscore GraphQL client (missing players, coaches).
        # No browser session involved; created on first use so disabled /
        # untested deployments pay nothing.
        self._fsql = None
        self._fsql_lock = asyncio.Lock()

    def _flashscore_team_ref(self, team_id: int | str, league_meta: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve {slug,id,name} for a team resolved via the flashscore
        match link (stored in league_meta['_flashscore_match'])."""
        if self.fc is None:
            return None
        match = (league_meta or {}).get("_flashscore_match") if isinstance(league_meta, dict) else None
        if not match:
            return None
        for side in ("home", "away"):
            t = match.get(side) or {}
            if t.get("id") is not None and str(t["id"]) == str(team_id):
                return t
        return None

    async def _football_data_team(self, name: str, league_meta: dict[str, Any]) -> dict[str, Any] | None:
        code = league_meta.get("football_data_code")
        if not code:
            return None
        team = await _timeout_aware(
            self.fd.search_team_in_competition(name, code), _CALL_CAP
        )
        if not team:
            return None
        return {
            "id": team.get("id"),
            "name": team.get("name"),
            "short_name": team.get("shortName"),
            "country": team.get("area", {}).get("name") if isinstance(team.get("area"), dict) else None,
            "provider": "football_data",
        }

    async def _thesportsdb_team(self, name: str, league_meta: dict[str, Any]) -> dict[str, Any] | None:
        # F2: pass the league context so the teams[0] fallback (no name
        # match) is rejected when its league contradicts the requested one
        # (wrong-club guard, e.g. "Lens" -> an unrelated first result).
        hint = league_meta.get("display") or league_meta.get("_league_key")
        team = await _timeout_aware(self.ts.search_team(name, hint), _CALL_CAP)
        if not team:
            return None
        return {
            "id": team.get("idTeam"),
            "name": team.get("strTeam"),
            "short_name": team.get("strTeamShort"),
            "country": team.get("strCountry"),
            "provider": "thesportsdb",
        }

    async def search_team(self, name: str, league_meta: dict[str, Any]) -> dict[str, Any] | None:
        league_key = league_meta.get("_league_key") or _key_from_meta(league_meta)
        aliased = resolve_team_alias(name, league_key)
        search_name = aliased or name

        for provider in ("football_data", "thesportsdb"):
            # Each fallback pays a full HTTP timeout when the provider is
            # slow/blocked; the budget guard keeps the chain from stacking
            # those timeouts past the deadline.
            if analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
                logger.warning("team search skipped: analysis budget nearly spent (%.0fs elapsed)", analysis_elapsed())
                return None
            try:
                result = (
                    await self._football_data_team(search_name, league_meta)
                    if provider == "football_data"
                    else await self._thesportsdb_team(search_name, league_meta)
                )
                if result and not self._fallback_identity_ok(provider, result, search_name, league_key):
                    result = None
                if result:
                    result["_role"] = "fallback"
                    result["_aliased"] = aliased is not None
                    # G1: record (provider, provider_id) -> canonical id so
                    # the same club resolved later from another source maps
                    # back to the same identity (wrong-club guard).
                    try:
                        result["canonical_id"] = _entity_registry().register(
                            result.get("provider"), result.get("id"), league_key, result.get("name") or name
                        )
                    except Exception:  # noqa: BLE001 -- registry never breaks resolve
                        logger.warning("entity registry register failed: %s", name)
                    return result
            except FootballDataError as exc:
                logger.warning("football-data auth error: %s", exc)
                return None
            except asyncio.TimeoutError:
                logger.warning("team search timeout on %s: %s", provider, search_name)
                continue
        logger.warning("no provider returned team for %s", name)
        return None

    @staticmethod
    def _fallback_identity_ok(
        provider: str, result: dict[str, Any], query: str, league_key: str | None,
    ) -> bool:
        """2026-09-02 (wrong-team audit): a by-NAME provider result must be
        the club that was asked for.

        Three checks, all fail-closed: (1) when both the query and the
        provider's name canonicalise through teams.json they must be the
        SAME club; (2) the entity registry is READ: a (provider, id) already
        mapped to a different canonical id is a wrong-club hit; (3) with no
        alias evidence at all the names themselves must match token-wise.
        """
        from .entity_registry import canonical_team_id
        from .team_alias import resolve_team_alias
        from .team_identity import names_match

        name = str(result.get("name") or "")
        if not name:
            return False
        q_alias = resolve_team_alias(query, league_key)
        r_alias = resolve_team_alias(name, league_key)
        if q_alias and r_alias:
            if q_alias != r_alias:
                logger.warning("%s resolved %r to %r (%s) -- different club, rejected", provider, query, name, r_alias)
                return False
            return True
        q_cid = canonical_team_id(league_key, query) if q_alias else None
        prior = None
        try:
            prior = _entity_registry().resolve(result.get("provider") or provider, result.get("id"))
        except Exception:  # noqa: BLE001 -- registry never breaks resolve
            prior = None
        if prior and q_cid and prior != q_cid:
            logger.warning("%s id %s is registered as %s, query %r is %s -- rejected", provider, result.get("id"), prior, query, q_cid)
            return False
        if names_match(query, name) or (q_alias and names_match(q_alias, name)) or (r_alias and names_match(query, r_alias)):
            return True
        logger.warning("%s resolved %r to %r -- name does not identify the same club, rejected", provider, query, name)
        return False

    async def search_teams_pair(
        self, home: str, away: str, league_meta: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve both teams, flashscore first, then per-team providers.

        Primary path: flashscore resolves BOTH teams in a single league-page
        render (team slugs + ids + match url). When flashscore is disabled,
        unavailable, or the pair is not found (e.g. qualification rounds),
        fall back to the per-team provider chain (football-data,
        thesportsdb).
        """
        league_key = league_meta.get("_league_key") or _key_from_meta(league_meta)
        if self.fc is not None and self.fc.available and league_key:
            if not analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
                # The league-page render costs 20-40s (cold Chrome launch +
                # SPA render) and the SAME pair is re-resolved for every
                # analyse/batch match, so cache the resolved pair per day.
                cache_key = None
                if self.cache is not None:
                    from .timeutil import wib_today_iso

                    _safe = lambda s: " ".join((s or "").lower().split())
                    cache_key = (
                        "flashscore_resolve_"
                        f"{league_key}_{wib_today_iso()}_"
                        f"{_safe(home)}_{_safe(away)}"
                    )
                    cached = self.cache.get(cache_key, ttl_seconds=24 * 3600)
                    if cached:
                        h, a = (cached or {}).get("home"), (cached or {}).get("away")
                        if h and a:
                            league_meta["_flashscore_match"] = cached.get("_match") or {}
                            return h, a
                resolved = await self.fc.resolve_match(league_key, home, away)
                if resolved:
                    h = resolved.get("home") or {}
                    a = resolved.get("away") or {}
                    if h.get("id") and a.get("id"):
                        # stash for later form/h2h/stats lookups
                        league_meta["_flashscore_match"] = resolved
                        out_home = {
                            "id": h["id"],
                            "name": h.get("name") or home,
                            "short_name": None,
                            "country": None,
                            "provider": "flashscore",
                            "_role": "primary_flashscore",
                            # G1: canonical identity recorded for the flashscore id.
                            "canonical_id": _safe_canonical(
                                "flashscore", h["id"], league_key, h.get("name") or home
                            ),
                        }
                        out_away = {
                            "id": a["id"],
                            "name": a.get("name") or away,
                            "short_name": None,
                            "country": None,
                            "provider": "flashscore",
                            "_role": "primary_flashscore",
                            # G1: canonical identity recorded for the flashscore id.
                            "canonical_id": _safe_canonical(
                                "flashscore", a["id"], league_key, a.get("name") or away
                            ),
                        }
                        if self.cache is not None and cache_key:
                            self.cache.set(
                                cache_key,
                                {"home": out_home, "away": out_away, "_match": resolved},
                            )
                        return out_home, out_away

        # LiveScore fallback: when Flashscore can't find the match (e.g.
        # qualification rounds), scan LiveScore's date feed which carries
        # today's full schedule including qualifiers.  The LiveScore event
        # carries home_id + away_id directly so we can skip the per-team
        # provider chain entirely.
        if self.livescore is not None and getattr(self.livescore, "available", False):
            if not analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
                try:
                    from .source_match import _search_livescore_any

                    _ls_match = await _search_livescore_any(self, home, away)
                    if _ls_match and _ls_match.get("home_id") and _ls_match.get("away_id"):
                        league_meta["_livescore_match"] = _ls_match
                        out_home = {
                            "id": _ls_match["home_id"],
                            "name": _ls_match.get("home") or home,
                            "short_name": None,
                            "country": _ls_match.get("country"),
                            "provider": "livescore",
                            "_role": "primary_livescore",
                            "canonical_id": _safe_canonical(
                                "livescore", _ls_match["home_id"], league_key,
                                _ls_match.get("home") or home,
                            ),
                        }
                        out_away = {
                            "id": _ls_match["away_id"],
                            "name": _ls_match.get("away") or away,
                            "short_name": None,
                            "country": _ls_match.get("country"),
                            "provider": "livescore",
                            "_role": "primary_livescore",
                            "canonical_id": _safe_canonical(
                                "livescore", _ls_match["away_id"], league_key,
                                _ls_match.get("away") or away,
                            ),
                        }
                        logger.info(
                            "team pair resolved via LiveScore fallback: %s vs %s",
                            out_home["name"], out_away["name"],
                        )
                        return out_home, out_away
                except Exception as exc:  # noqa: BLE001
                    logger.warning("livescore team pair fallback failed: %s", exc)

        home_team = await self.search_team(home, league_meta)
        away_team = await self.search_team(away, league_meta)

        # A team may still be unresolved (e.g. qualifier fixture the primary
        # providers do not cover); the caller handles None teams with a clear
        # "tim tidak ditemukan" error.
        return home_team, away_team

    async def _football_data_form(self, team_id: int, limit: int = 10) -> dict[str, Any] | None:
        matches = await _timeout_aware(
            self.fd.fetch_last_matches(team_id, limit), _CALL_CAP
        )
        if not matches:
            return None
        results: list[str] = []
        gf_list: list[int] = []
        ga_list: list[int] = []
        home_w = home_d = home_l = 0
        away_w = away_d = away_l = 0
        for m in matches:
            home_id = (m.get("homeTeam") or {}).get("id")
            away_id = (m.get("awayTeam") or {}).get("id")
            score = m.get("score", {}).get("fullTime", {})
            home_goals = score.get("home")
            away_goals = score.get("away")
            if home_goals is None or away_goals is None:
                continue
            is_home = (home_id == team_id)
            gf = home_goals if is_home else away_goals
            ga = away_goals if is_home else home_goals
            gf_list.append(int(gf))
            ga_list.append(int(ga))
            if home_goals > away_goals:
                if is_home:
                    results.append("W"); home_w += 1
                else:
                    results.append("L"); away_l += 1
            elif home_goals < away_goals:
                if is_home:
                    results.append("L"); home_l += 1
                else:
                    results.append("W"); away_w += 1
            else:
                results.append("D")
                if is_home:
                    home_d += 1
                else:
                    away_d += 1
        return {
            "sequence": "-".join(results) if results else None,
            "gf_avg": (sum(gf_list) / len(gf_list)) if gf_list else 0.0,
            "ga_avg": (sum(ga_list) / len(ga_list)) if ga_list else 0.0,
            "home": {"w": home_w, "d": home_d, "l": home_l},
            "away": {"w": away_w, "d": away_d, "l": away_l},
            "sample_size": len(results),
            # Raw (gf, ga) scorelines per match, OLDEST -> NEWEST, so the
            # Poisson model can apply Dixon-Coles time-decay weighting.
            # football-data.org returns matches ascending by date.
            "recent_goals": list(zip(gf_list, ga_list)) or None,
        }

    async def _thesportsdb_form(self, team_id: str, limit: int = 5) -> dict[str, Any] | None:
        matches = await _timeout_aware(
            self.ts.fetch_last_matches(team_id, limit), _CALL_CAP
        )
        if not matches:
            return None
        results: list[str] = []
        gf_list: list[int] = []
        ga_list: list[int] = []
        for m in matches:
            home_id = (m.get("idHomeTeam") or "")
            away_id = (m.get("idAwayTeam") or "")
            score = (m.get("intHomeScore"), m.get("intAwayScore"))
            if None in score or "" in score:
                continue
            try:
                hs, aws = int(score[0]), int(score[1])
            except (ValueError, TypeError):
                continue
            is_home = (home_id == str(team_id))
            gf = hs if is_home else aws
            ga = aws if is_home else hs
            gf_list.append(gf)
            ga_list.append(ga)
            if hs == aws:
                results.append("D")
            elif home_id == str(team_id):
                results.append("W" if hs > aws else "L")
            else:
                results.append("W" if aws > hs else "L")
        return {
            "sequence": "-".join(results) if results else None,
            "gf_avg": (sum(gf_list) / len(gf_list)) if gf_list else 0.0,
            "ga_avg": (sum(ga_list) / len(ga_list)) if ga_list else 0.0,
            "sample_size": len(results),
            # eventslast returns newest-first -> reverse to oldest->newest.
            "recent_goals": list(reversed(list(zip(gf_list, ga_list)))) or None,
        }

    # TODO-03 (train/serve parity): the validated walk-forward backtest builds
    # form from the last-5 league matches only (deque(maxlen=5)). The live
    # path must use the SAME window or the Poisson time-decay sees a
    # different distribution in production than in validation.
    FORM_WINDOW = 5

    async def fetch_team_form(
        self,
        team_id: int | str,
        league_meta: dict[str, Any],
        limit: int = FORM_WINDOW,
    ) -> dict[str, Any] | None:
        # P2: the provider-aware cache lives INSIDE _fetch_team_form_uncached
        # (keyed per provider), so a cross-provider id collision can never
        # serve provider A's data to a provider B lookup. No provider-blind
        # cache key exists here anymore.
        # Shared-budget guard: form is a core feature, so only bail when the
        # clock is armed and virtually exhausted (no time left for the chain).
        if analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
            logger.warning("team_form skipped: analysis budget nearly spent (%.0fs elapsed)", analysis_elapsed())
            return None
        return await self._fetch_team_form_uncached(team_id, league_meta, limit)

    def _form_cache_key(self, provider: str, league_key: str, team_id: int | str, limit: int) -> str | None:
        """P2: provider-aware form cache key.

        The provider is part of the key so the same numeric/string id under
        two providers (e.g. a flashscore string id vs a football-data int id
        that happen to collide) can never serve cross-provider data within
        the TTL window.
        """
        if self.cache is None:
            return None
        return f"team_form_{provider}_{league_key}_{team_id}_{limit}"

    async def _cached_form(self, key: str | None) -> dict[str, Any] | None:
        if key is None:
            return None
        cached = self.cache.get(key, ttl_seconds=3600)
        return cached if cached is not None else None

    async def _store_form(self, key: str | None, form: dict[str, Any] | None) -> dict[str, Any] | None:
        if key is not None and form is not None:
            self.cache.set(key, form)
        return form

    async def _fetch_team_form_uncached(
        self,
        team_id: int | str,
        league_meta: dict[str, Any],
        limit: int = FORM_WINDOW,
    ) -> dict[str, Any] | None:
        league_key = league_meta.get("_league_key") or _key_from_meta(league_meta)
        fs = self._flashscore_team_ref(team_id, league_meta)
        if fs:
            key = self._form_cache_key("flashscore", league_key or "unknown", team_id, limit)
            hit = await self._cached_form(key)
            if hit is not None:
                return hit
            form = await self.fc.fetch_team_form(fs["slug"], fs["id"], limit=limit)
            return await self._store_form(key, form)

        # LiveScore BY EVENT + TEAM ID (2026-09-02, wrong-team post-mortem):
        # when the pair was resolved on LiveScore the match carries the event
        # id and both team ids. The per-event ``/form-e`` payload is keyed by
        # team id, so this can never return another club -- it runs BEFORE
        # every by-name path below. Cache key = provider + event + team id.
        _ls_match = (league_meta or {}).get("_livescore_match") if isinstance(league_meta, dict) else None
        if (
            isinstance(_ls_match, dict)
            and _ls_match.get("source_id")
            and str(team_id) in (str(_ls_match.get("home_id")), str(_ls_match.get("away_id")))
            and getattr(self, "livescore", None) is not None
            and getattr(self.livescore, "available", False)
            and not analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN)
        ):
            try:
                from .livescore import team_form_by_id

                _eid = str(_ls_match["source_id"])
                key = self._form_cache_key("livescore_event", _eid, team_id, limit)
                hit = await self._cached_form(key)
                if hit is not None:
                    return hit
                _raw = await self.livescore.fetch_form(_eid)
                form = team_form_by_id(_raw, team_id, limit=limit)
                if form and form.get("sequence"):
                    return await self._store_form(key, form)
            except Exception as exc:  # noqa: BLE001 -- best-effort, next source
                logger.warning("livescore event form failed (best-effort): %s", exc)

        # Flashscore BY NAME: on the regular query path (no !livescore /
        # !flashscore prefix) the teams are resolved via football-data, so
        # ``_flashscore_match`` is never populated and the flashscore form
        # (full 5-match window) is silently skipped even though it is the
        # richest source. Resolve the team slug via the pure-HTTP livesport
        # search API (no browser render) and fetch its team-results page form.
        # Only when flashscore is enabled AND the budget allows (the team
        # results page is a browser render, ~15-25s).
        team_name = (league_meta.get("_team_names") or {}).get(str(team_id))
        if (
            team_name
            and self.fc is not None
            and getattr(self.fc, "available", True)
            and not analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN)
        ):
            try:
                from .flashscore import _suggest_team

                slug_id = await asyncio.to_thread(_suggest_team, team_name)
                if slug_id:
                    slug, fsid = slug_id
                    # 2026-09-02: the suggest API resolved a NAME; verify the
                    # returned club is the requested one before trusting its
                    # results page (the "name lottery" behind Copenhagen /
                    # Lincoln Red Imps), and cache under the FLASHSCORE id --
                    # never under a foreign provider's id.
                    from .team_identity import names_match

                    if not names_match(team_name, slug.replace("-", " ")):
                        logger.warning(
                            "flashscore by-name form: suggest returned %r for %r -- rejected",
                            slug, team_name,
                        )
                        slug_id = None
                if slug_id:
                    slug, fsid = slug_id
                    key = self._form_cache_key("flashscore", league_key or "unknown", fsid, limit)
                    hit = await self._cached_form(key)
                    if hit is not None:
                        return hit
                    form = await self.fc.fetch_team_form(slug, fsid, limit=limit)
                    if form and form.get("sequence"):
                        form["source"] = "flashscore"
                        return await self._store_form(key, form)
            except Exception as exc:  # noqa: BLE001 -- best-effort, next source
                logger.warning("flashscore by-name form failed (best-effort): %s", exc)

        # LiveScore SECOND: right after flashscore, before FBref /
        # football-data / thesportsdb. It rebuilds the team's last-N finished
        # matches from the verified no-key date feed (cheap, no browser, no
        # key) -- so when flashscore is off-line, LiveScore fills the FULL
        # 5-match window instead of the thin 1-match fallbacks below. A thin
        # LiveScore result is kept as ``thin_form`` while the chain continues;
        # the fullest form wins at the end.
        thin_form: dict[str, Any] | None = None
        # getattr: tests construct the fetcher via __new__ (no __init__), so
        # ``livescore`` may be unset -- treat as not configured.
        if getattr(self, "livescore", None) is not None and getattr(self.livescore, "available", False):
            try:
                if analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
                    logger.warning("livescore form skipped: analysis budget nearly spent (%.0fs elapsed)", analysis_elapsed())
                else:
                    team_name = (league_meta.get("_team_names") or {}).get(str(team_id))
                    # 2026-09-02: this is a BY-NAME lookup -- key it by the
                    # name it looked up, never by a foreign provider's id.
                    key = self._form_cache_key(
                        "livescore_byname", league_key or "unknown",
                        re.sub(r"[^a-z0-9]+", "-", str(team_name or team_id).lower()).strip("-"), limit,
                    )
                    hit = await self._cached_form(key)
                    if hit is not None:
                        if thin_form is None or _form_depth(hit) >= _form_depth(thin_form):
                            return hit
                    else:
                        # G5: pass league_key so cup / other-division matches
                        # between the same teams do not pollute the form window.
                        form = await self._livescore_form(
                            team_name, limit, league_key=league_key,
                            league_country=(league_meta or {}).get("country"),
                        )
                        if form and form.get("sequence"):
                            form["source"] = "livescore"
                            stored = await self._store_form(key, form)
                            if thin_form is None or _form_depth(stored) >= _form_depth(thin_form):
                                return stored
            except Exception as exc:  # noqa: BLE001 -- best-effort, next source
                logger.warning("livescore form fallback failed (best-effort): %s", exc)

        if (
            self.sd.supports_league(league_key or "")
            and isinstance(team_id, int)
            and team_id > 0
        ):
            team_name = league_meta.get("_team_names", {}).get(str(team_id))
            if team_name:
                key = self._form_cache_key("soccerdata_fbref", league_key or "unknown", team_id, limit)
                hit = await self._cached_form(key)
                if hit is not None:
                    return hit
                form = await self.sd.read_team_form(league_key, team_name, limit)
                if form and form.get("sequence"):
                    form["source"] = "soccerdata_fbref"
                    return await self._store_form(key, form)

        # F1 (thin-form fill): a form window shallower than the requested
        # ``limit`` is NOT treated as the final answer -- it is kept as
        # ``thin_form`` while the chain keeps trying richer providers, and the
        # FULLEST form wins at the end (LiveScore already ran above).

        # oddspapi/fallback-resolved teams carry provider-internal STRING ids
        # that match no provider's lookup (no flashscore ref, not an int, not a
        # thesportsdb id). Resolve the team BY NAME (alias-aware, budget-guarded)
        # and read its form from the provider that answers.
        team_name = (league_meta.get("_team_names") or {}).get(str(team_id))
        if team_name and not (isinstance(team_id, int) and team_id > 0):
            resolved = await self.search_team(team_name, league_meta)
            if resolved:
                form = None
                key = None
                if resolved["provider"] == "football_data":
                    key = self._form_cache_key("football_data", league_key or "unknown", team_id, limit)
                    hit = await self._cached_form(key)
                    if hit is not None:
                        if _form_depth(hit) >= limit:
                            return hit
                        thin_form = hit
                    else:
                        form = await self._football_data_form(int(resolved["id"]), limit)
                elif resolved["provider"] == "thesportsdb":
                    key = self._form_cache_key("thesportsdb", league_key or "unknown", team_id, limit)
                    hit = await self._cached_form(key)
                    if hit is not None:
                        if _form_depth(hit) >= limit:
                            return hit
                        thin_form = hit
                    else:
                        form = await self._thesportsdb_form(str(resolved["id"]), limit)
                if form and form.get("sequence"):
                    form["source"] = resolved["provider"]
                    stored = await self._store_form(key, form)
                    if _form_depth(stored) >= limit:
                        return stored
                    if thin_form is None or _form_depth(stored) > _form_depth(thin_form):
                        thin_form = stored

        try:
            if isinstance(team_id, int):
                key = self._form_cache_key("football_data", league_key or "unknown", team_id, limit)
                hit = await self._cached_form(key)
                if hit is not None:
                    if _form_depth(hit) >= limit:
                        return hit
                    if thin_form is None or _form_depth(hit) > _form_depth(thin_form):
                        thin_form = hit
                else:
                    form = await self._football_data_form(team_id, limit)
                    if form and form.get("sequence"):
                        form["source"] = "football_data"
                        stored = await self._store_form(key, form)
                        if _form_depth(stored) >= limit:
                            return stored
                        if thin_form is None or _form_depth(stored) > _form_depth(thin_form):
                            thin_form = stored
        except FootballDataError as exc:
            logger.warning("football-data auth error in form: %s", exc)

        key = self._form_cache_key("thesportsdb", league_key or "unknown", team_id, limit)
        hit = await self._cached_form(key)
        if hit is not None:
            if _form_depth(hit) >= limit:
                return hit
            if thin_form is None or _form_depth(hit) > _form_depth(thin_form):
                thin_form = hit
        else:
            # F1 (2026-08-17): thesportsdb answers only with ITS OWN provider
            # ids (idTeam). A football-data INT id passed to eventslast.php
            # either returns events for an unrelated club whose idTeam
            # collides with the fd int (wrong-team form, home/away labels
            # from a DIFFERENT team) or returns None. Resolve the team by
            # name first and call thesportsdb only when the resolution is
            # genuinely thesportsdb. String ids were already resolved in the
            # name-fallback block above (thesportsdb id used if it won), so
            # nothing more to try here for them.
            ts_id: str | None = None
            if isinstance(team_id, int) and team_id > 0 and team_name:
                try:
                    resolved_final = await self.search_team(team_name, league_meta)
                except Exception:  # noqa: BLE001 -- best-effort, next source
                    resolved_final = None
                if resolved_final and resolved_final.get("provider") == "thesportsdb":
                    ts_id = str(resolved_final["id"])
            if ts_id is not None:
                form = await self._thesportsdb_form(ts_id, limit)
                if form:
                    form["source"] = "thesportsdb"
                    stored = await self._store_form(key, form)
                    if _form_depth(stored) >= limit:
                        return stored
                    if thin_form is None or _form_depth(stored) > _form_depth(thin_form):
                        thin_form = stored

        return thin_form

    async def _livescore_form(
        self,
        team_name: str | None,
        limit: int = FORM_WINDOW,
        lookback_days: int = 45,
        league_key: str | None = None,
        league_country: str | None = None,
    ) -> dict[str, Any] | None:
        """Last-N form for ``team_name`` rebuilt from the LiveScore date feed.

        Scans the no-key ``/date/soccer`` feed backwards from today (page 0-1
        per date, TTL-cached like the rest of the pipeline), collects the
        team's finished matches by tolerant name match, and returns the same
        flashscore-compatible form shape (OLDEST -> NEWEST ``recent_goals``
        so the Poisson model's Dixon-Coles time-decay sees the same ordering
        as every other provider). Stops early once ``limit`` matches are
        found; returns None when the team cannot be found (never fabricates).

        G5 (2026-08-17): when ``league_key`` is known, a finished match whose
        competition RESOLVES to a DIFFERENT registered league is skipped --
        the same two teams meeting in a cup or another division must not
        pollute the league form window (name matching alone cannot tell them
        apart). Unregistered competitions are kept (cannot disprove).
        """
        if not team_name or self.livescore is None or not getattr(self.livescore, "available", False):
            return None
        from .livescore import DATE_FEED_TTL_SECONDS, parse_soccer_payload
        from .team_identity import country_matches, match_side

        seq: list[str] = []
        gf_list: list[int] = []
        ga_list: list[int] = []
        home_w = home_d = home_l = 0
        away_w = away_d = away_l = 0
        # 2026-09-02: feed pages overlap (the same event can sit on page 0 and
        # page 1 of a date) -- one finished match must count once.
        seen_events: set[str] = set()
        matched_clubs: list[str] = []
        today = datetime.now(timezone.utc)
        for back in range(max(1, int(lookback_days))):
            if analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
                break
            date8 = (today - timedelta(days=back)).strftime("%Y%m%d")
            for page in (0, 1):
                key = f"livescore_date_{date8}_{page}"
                payload = self.cache.get(key, ttl_seconds=DATE_FEED_TTL_SECONDS) if self.cache is not None else None
                if payload is None:
                    payload = await self.livescore.fetch_soccer_date(date8, page)
                    if payload is not None and self.cache is not None:
                        self.cache.set(key, payload)
                for fx in parse_soccer_payload(payload):
                    if fx.get("status") != "finished":
                        continue
                    sc = fx.get("score") or {}
                    hg, ag = sc.get("home"), sc.get("away")
                    if hg is None or ag is None:
                        continue
                    # P3-2 parity (2026-08-22): friendlies never contribute to
                    # form -- the G5 check below only rejects leagues that
                    # RESOLVE to a different key, and "Club Friendlies 2026"
                    # resolves to nothing (Fortuna-AZ lambda audit).
                    try:
                        from .nowgoal import is_friendly_competition

                        if is_friendly_competition(str(fx.get("competition") or "")):
                            continue
                    except Exception:  # noqa: BLE001 -- guard never breaks form
                        pass
                    # G5: reject matches of a DIFFERENT resolved league.
                    comp_key = None
                    if league_key and fx.get("competition"):
                        try:
                            from .league_resolver import competition_league_key

                            comp_key = competition_league_key(str(fx.get("competition")))
                            if comp_key and comp_key != league_key:
                                continue
                        except Exception:  # noqa: BLE001 -- guard never breaks form
                            comp_key = None
                    # 2026-09-02 (wrong-team post-mortem): a by-NAME row must
                    # also be geographically plausible -- an English club's
                    # form never comes from USL League Two / NPL Victoria --
                    # and, when the competition is not the analysed league
                    # (unregistered cup, other tier), only a STRICT identity
                    # match counts (no extra token: "Lyon" != "Lyon la Duchere").
                    _same_country = country_matches(league_country, fx.get("country"))
                    if _same_country is False:
                        continue
                    _strict = not (comp_key and comp_key == league_key)
                    side = match_side(team_name, fx.get("home") or "", fx.get("away") or "", strict=_strict)
                    if side is None:
                        continue
                    _ev_key = str(fx.get("source_id") or "") or f"{fx.get('kickoff')}|{fx.get('home')}|{fx.get('away')}"
                    if _ev_key in seen_events:
                        continue
                    seen_events.add(_ev_key)
                    is_home = side == "home"
                    matched_clubs.append(str(fx.get("home") if is_home else fx.get("away")))
                    gf, ga = (int(hg), int(ag)) if is_home else (int(ag), int(hg))
                    gf_list.append(gf)
                    ga_list.append(ga)
                    if gf > ga:
                        seq.append("W")
                        if is_home:
                            home_w += 1
                        else:
                            away_w += 1
                    elif gf == ga:
                        seq.append("D")
                        if is_home:
                            home_d += 1
                        else:
                            away_d += 1
                    else:
                        seq.append("L")
                        if is_home:
                            home_l += 1
                        else:
                            away_l += 1
                    if len(seq) >= limit:
                        break
                if len(seq) >= limit:
                    break
            if len(seq) >= limit:
                break
        if not seq:
            return None
        # 2026-09-02: rows from TWO different clubs ("Inter Milan" and
        # "FC Inter Turku" for the query "Inter") mean the name is ambiguous
        # here -- refuse rather than blend two clubs into one window.
        from .team_identity import distinct_clubs

        if distinct_clubs(matched_clubs) > 1:
            logger.warning("livescore by-name form for %r matched several clubs %s -- refused", team_name, sorted(set(matched_clubs)))
            return None
        # Feed scanned newest-first -> reverse to OLDEST -> NEWEST.
        seq = list(reversed(seq))
        gf_list = list(reversed(gf_list))
        ga_list = list(reversed(ga_list))
        window = min(limit, len(seq))
        return {
            "sequence": "-".join(seq) if seq else None,
            "gf_avg": round(sum(gf_list) / window, 2) if window else 0.0,
            "ga_avg": round(sum(ga_list) / window, 2) if window else 0.0,
            "home": {"w": home_w, "d": home_d, "l": home_l},
            "away": {"w": away_w, "d": away_d, "l": away_l},
            "sample_size": window,
            # OLDEST -> NEWEST scorelines, matching every other provider.
            "recent_goals": list(zip(gf_list, ga_list)) or None,
        }

    async def _football_data_h2h(self, team1_id: int, team2_id: int) -> dict[str, int] | None:
        matches = await _timeout_aware(
            self.fd.fetch_h2h(team1_id, team2_id), _CALL_CAP
        )
        if not matches:
            return None
        wins = draws = losses = 0
        for m in matches:
            winner = m.get("winner") if isinstance(m.get("winner"), dict) else None
            winner_id = winner.get("id") if winner else None
            if winner_id is None:
                draws += 1
            elif winner_id == team1_id:
                wins += 1
            else:
                losses += 1
        return {"wins": wins, "draws": draws, "losses": losses}

    async def fetch_h2h(
        self,
        team1_id: int | str,
        team2_id: int | str,
        league_meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        # Shared-budget guard: H2H is context (the model substitutes it with
        # Elo when missing), so skip the whole chain once time is short.
        if analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
            logger.warning("h2h skipped: analysis budget nearly spent (%.0fs elapsed)", analysis_elapsed())
            return None
        # The flashscore H2H render costs ~15-25s; the same pair is re-fetched
        # for every analyse/batch match, so cache the result per pair (24h --
        # H2H history barely changes intraday). P2: the provider is part of
        # the key so an id collision across providers can never serve
        # cross-provider data.
        league_key = league_meta.get("_league_key") or _key_from_meta(league_meta)
        team_names = league_meta.get("_team_names", {}) if isinstance(league_meta, dict) else {}
        team_a_name = team_names.get(str(team1_id))
        team_b_name = team_names.get(str(team2_id))

        def _h2h_key(provider: str) -> str | None:
            if self.cache is None:
                return None
            return f"h2h_{provider}_{team1_id}_{team2_id}"

        def _h2h_hit(key: str | None):
            if key is None:
                return None
            return self.cache.get(key, ttl_seconds=24 * 3600)

        def _emit(result: dict[str, Any] | None, key: str | None) -> dict[str, Any] | None:
            if result is not None and key is not None:
                self.cache.set(key, result)
            return result

        # Flashscore H2H from the match page is the PRIMARY provider: it is
        # the freshest source (renders the actual matchup's H2H tab) and the
        # rest of the pipeline already renders this page for stats. Fall back
        # to football-data / FBref only when flashscore is unavailable.
        fs_match = (league_meta or {}).get("_flashscore_match") if isinstance(league_meta, dict) else None
        if fs_match and fs_match.get("match_url"):
            h_name = (fs_match.get("home") or {}).get("name") or team_a_name
            a_name = (fs_match.get("away") or {}).get("name") or team_b_name
            key = _h2h_key("flashscore_h2h")
            hit = _h2h_hit(key)
            if hit:
                return hit
            result = await self.fc.fetch_match_h2h(fs_match["match_url"], h_name, a_name)
            if result:
                result["source"] = "flashscore_h2h"
                return _emit(result, key)

        if isinstance(team1_id, int) and isinstance(team2_id, int):
            try:
                key = _h2h_key("football_data")
                hit = _h2h_hit(key)
                if hit:
                    return hit
                result = await self._football_data_h2h(team1_id, team2_id)
                if result and (result.get("wins") or result.get("draws") or result.get("losses")):
                    result["source"] = "football_data"
                    return _emit(result, key)
            except FootballDataError as exc:
                logger.warning("football-data auth error in h2h: %s", exc)

        if league_key and team_a_name and team_b_name and self.sd.supports_league(league_key):
            key = _h2h_key("soccerdata_fbref_h2h")
            hit = _h2h_hit(key)
            if hit:
                return hit
            result = await self.sd.read_h2h(league_key, team_a_name, team_b_name, limit=5)
            if result and (result.get("wins") or result.get("draws") or result.get("losses")):
                result["source"] = "soccerdata_fbref_h2h"
                return _emit(result, key)

        # H2H wiring fix (2026-08-17): on the regular query path teams resolve
        # via football-data, so ``_flashscore_match`` is never set and the
        # flashscore H2H tab -- the freshest source -- was silently skipped
        # even though the pair IS on flashscore (H2H fell to fbref / None).
        # Resolve the match URL BY NAME (same browser call the pair resolver
        # uses) and render the H2H tab. LAST in the chain because it is the
        # only step that needs a fresh browser render (~15-30s): the cheap
        # football-data / FBref paths get their chance first. Budget-guarded;
        # when time is short the chain ends at fbref as before.
        if (
            self.fc is not None
            and getattr(self.fc, "available", True)
            and team_a_name
            and team_b_name
            and not analysis_budget_exhausted(margin_seconds=28.0)
        ):
            try:
                key = _h2h_key("flashscore_h2h")
                hit = _h2h_hit(key)
                if hit:
                    return hit
                resolved = await self.fc.resolve_match(league_key, team_a_name, team_b_name)
                if resolved and resolved.get("match_url"):
                    result = await self.fc.fetch_match_h2h(
                        resolved["match_url"], team_a_name, team_b_name
                    )
                    if result:
                        result["source"] = "flashscore_h2h"
                        return _emit(result, key)
            except Exception as exc:  # noqa: BLE001 -- best-effort, next source
                logger.warning("flashscore by-name h2h failed (best-effort): %s", exc)

        # LiveScore H2H fallback: when flashscore/football-data/fbref all
        # failed, try LiveScore's /H2H endpoint. This is the same structure
        # as flashscore H2H (wins/draws/losses from home perspective) so the
        # model consumes it identically. Budget-guarded.
        if (
            self.livescore is not None
            and getattr(self.livescore, "available", False)
            and not analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN)
        ):
            try:
                key = _h2h_key("livescore_h2h")
                hit = _h2h_hit(key)
                if hit:
                    return hit
                # Resolve the fixture via LiveScore date feed
                from .livescore import parse_soccer_payload
                # Lazy import (circular-import safe): _norm_team_name lives
                # in analyse.py -- same pattern as datasources._norm().
                from .analyse import _norm_team_name
                home_variants_l = [team_a_name or ""]
                away_variants_l = [team_b_name or ""]
                _today = datetime.now(timezone.utc)
                for _back in range(2):  # today + yesterday
                    _d8 = (_today - timedelta(days=_back)).strftime("%Y%m%d")
                    for _pg in (0, 1):
                        _payload = await self.livescore.fetch_soccer_date(_d8, _pg)
                        for _fx in parse_soccer_payload(_payload):
                            if not (_fx.get("home") and _fx.get("away")):
                                continue
                            _h_norm = _norm_team_name(_fx["home"])
                            _a_norm = _norm_team_name(_fx["away"])
                            if (_h_norm == _norm_team_name(home_variants_l[0]) and
                                    _a_norm == _norm_team_name(away_variants_l[0])):
                                _eid = _fx.get("source_id")
                                if _eid:
                                    _raw = await self.livescore.fetch_h2h(str(_eid))
                                    if _raw:
                                        from .livescore import parse_h2h
                                        _parsed = parse_h2h(_raw, _fx)
                                        if _parsed and (_parsed.get("wins") or _parsed.get("draws") or _parsed.get("losses")):
                                            _parsed["source"] = "livescore_h2h"
                                            return _emit(_parsed, key)
            except Exception as exc:  # noqa: BLE001 -- best-effort
                logger.warning("livescore h2h fallback failed (best-effort): %s", exc)

        return None

    async def fetch_upcoming_fixture(
        self,
        team1_id: int | str,
        team2_id: int | str,
        league_meta: dict[str, Any],
        max_days_ahead: int = 30,
    ) -> dict[str, Any] | None:
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone.utc).date()
        end = today + timedelta(days=max_days_ahead)
        if analysis_budget_exhausted(margin_seconds=_BUDGET_MARGIN):
            logger.warning("fixture skipped: analysis budget nearly spent (%.0fs elapsed)", analysis_elapsed())
            return None

        # Flashscore primary: the match url + kickoff come from the resolved
        # match (the league page already gave us the fixture).
        fs_match = (league_meta or {}).get("_flashscore_match") if isinstance(league_meta, dict) else None
        if fs_match and fs_match.get("match_url"):
            date_iso = None
            date_text = fs_match.get("date_text")
            if date_text:
                # flashscore formats: "20.02.2022 16:00" or "Today 21:00" /
                # "Tomorrow 15:30". Parse the full form; relative forms are
                # resolved by the caller via the odds payload kickoff.
                m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})", date_text)
                if m:
                    dd, mm, yyyy, hh, mi = m.groups()
                    try:
                        date_iso = _flashscore_local_to_utc_iso(
                            int(yyyy), int(mm), int(dd), int(hh), int(mi)
                        )
                    except (ValueError, OverflowError):
                        date_iso = None
            return {
                "date": date_iso,
                "status": "notstarted",
                "venue": "home",
                "source": "flashscore",
                "flashscore_url": fs_match["match_url"],
                # Carried for the finished-match result display (the league /
                # homepage row exposes the final score once the match is done).
                "score": fs_match.get("score") or None,
            }

        code = league_meta.get("football_data_code")
        if code and isinstance(team1_id, int):
            fixtures = await self.fd.fetch_matches_by_competition(code, today.isoformat(), end.isoformat())
            if fixtures:
                for m in fixtures:
                    home = (m.get("homeTeam") or {}).get("id")
                    away = (m.get("awayTeam") or {}).get("id")
                    if home == team1_id and away == team2_id:
                        return {
                            "date": m.get("utcDate"),
                            "status": m.get("status"),
                            "venue": "home",
                            "source": "football_data",
                        }
                    if home == team2_id and away == team1_id:
                        return {
                            "date": m.get("utcDate"),
                            "status": m.get("status"),
                            "venue": "away",
                            "source": "football_data",
                        }

        return None

    async def fetch_homepage_matches(
        self,
    ) -> list[dict[str, Any]] | None:
        """Today's matches from the flashscore homepage (all competitions).

        Complements football-data, which only covers registered league codes:
        the homepage carries Conference League qualification, friendlies, AFC
        qualifiers, minor cups, etc. Returns normalized rows tagged with the
        competition section they belong to.
        """
        if self.fc is None or not self.fc.available:
            return None
        rows = await self.fc.fetch_homepage_matches()
        if not rows:
            return None
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "home": {"id": r.get("home_id"), "name": r.get("home_name")},
                "away": {"id": r.get("away_id"), "name": r.get("away_name")},
                # Slugs ride along so top can pre-resolve analyzable pairs into
                # the flashscore_resolve_* cache without another browser render
                # (fetch_team_form needs the slug; match_url needs no re-scrape).
                "home_slug": r.get("home_slug"),
                "away_slug": r.get("away_slug"),
                "competition": r.get("competition") or "Other",
                "date_text": r.get("date_text"),
                "match_url": r.get("match_url"),
                "status": r.get("status") or "scheduled",
                "source": "flashscore",
            })
        return out

    async def fetch_flashscore_stats_for_match(
        self,
        match_url: str,
    ) -> dict[str, Any] | None:
        """xG/possession/shots for a match from the flashscore summary page."""
        if self.fc is None or not self.fc.available:
            return None
        return await self.fc.fetch_match_statistics(match_url)

    async def fetch_flashscore_lineups_for_match(
        self,
        match_url: str,
    ) -> dict[str, Any] | None:
        """Predicted (pre-match) or confirmed lineups for a match, best-effort.

        Context info only -- NOT a model feature (flashscore predicted
        lineups cannot be validated historically, so per the project's
        no-OOS-evidence rule they must not feed the engine).

        Per-match disk cache (30 min TTL): lineups shift as kickoff nears
        (predicted -> confirmed, injury updates), so a short TTL avoids
        serving stale XIs while still saving a browser render for
        back-to-back queries of the same fixture.
        """
        if self.fc is None or not self.fc.available:
            return None
        cache_key = None
        if self.cache is not None:
            url_hash = hashlib.sha256(match_url.encode("utf-8")).hexdigest()[:16]
            cache_key = f"flashscore_lineups_{url_hash}"
            cached = self.cache.get(cache_key, ttl_seconds=1800)
            if cached is not None:
                return cached
        lineups = await self.fc.fetch_match_lineups(match_url)
        if lineups is not None and self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, lineups)
        return lineups

    def _flashscore_graphql(self):
        """Lazy pure-HTTP GraphQL client (missing players / coaches)."""
        if self._fsql is None:
            from .flashscore_graphql import FlashscoreGraphqlClient

            self._fsql = FlashscoreGraphqlClient()
        return self._fsql

    async def fetch_flashscore_event_context(
        self,
        match_url: str,
        home_name: str | None = None,
        away_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Missing players + lineups + coaches via the GraphQL gateway.

        Pure HTTP (no browser render) keyed by the ?mid= event id in the
        match url. CONTEXT ONLY -- never a model feature (same no-OOS-rule
        as predicted lineups). Returns
        {home: {name, missing, unsure, players, formation, coaches}, away:
        {...}} side-resolved against home_name/away_name, or None. The
        disk cache TTL is 30 min so injury updates near kickoff still
        propagate.
        """
        m = re.search(r"[?&]mid=([A-Za-z0-9]+)", match_url or "")
        if not m:
            return None
        event_id = m.group(1)
        cache_key = f"flashscore_ctx_{event_id}"
        if self.cache is not None:
            cached = self.cache.get(cache_key, ttl_seconds=1800)
            if cached is not None:
                return cached
        try:
            async with self._fsql_lock:
                ctx = await _timeout_aware(
                    self._flashscore_graphql().fetch_event_context(
                        event_id, home_name, away_name
                    ),
                    _CALL_CAP,
                )
        except Exception as exc:
            logger.warning("flashscore graphql context failed: %s", exc)
            return None
        if ctx is not None and self.cache is not None:
            self.cache.set(cache_key, ctx)
        return ctx

    async def fetch_league_standings(
        self,
        league_key: str,
    ) -> dict[str, Any] | None:
        """Overall league table from the flashscore standings page (context).

        Browser render, so callers gate it on the shared analysis budget; a
        league without a registered flashscore path returns None. Cached 1h.
        """
        if self.fc is None or not self.fc.available:
            return None
        cache_key = f"flashscore_standings_{league_key}"
        if self.cache is not None:
            cached = self.cache.get(cache_key, ttl_seconds=3600)
            if cached is not None:
                return cached
        standings = await self.fc.fetch_league_standings(league_key)
        if standings is not None and self.cache is not None:
            self.cache.set(cache_key, standings)
        return standings

    async def fetch_flashscore_match_info(
        self,
        match_url: str,
    ) -> dict[str, Any] | None:
        """Referee/venue/capacity/neutral flag for a match (context).

        Browser render, budget-gated by the caller. Cached 24h (venue and
        referee are stable for a fixture).
        """
        if self.fc is None or not self.fc.available:
            return None
        url_hash = hashlib.sha256((match_url or "").encode("utf-8")).hexdigest()[:16]
        cache_key = f"flashscore_match_info_{url_hash}"
        if self.cache is not None:
            cached = self.cache.get(cache_key, ttl_seconds=24 * 3600)
            if cached is not None:
                return cached
        info = await self.fc.fetch_match_info(match_url)
        if info is not None and self.cache is not None:
            self.cache.set(cache_key, info)
        return info

    async def fetch_fixtures_for_date(
        self,
        league_meta: dict[str, Any],
        target_date: str,
    ) -> list[dict[str, Any]]:
        """Fixtures on a WIB calendar day. football-data queries UTC date
        ranges, so we fetch the UTC range covering the whole WIB day and then
        filter by the WIB date (matches kicking off 00:00-06:59 WIB fall on
        the previous UTC day and were previously missed).
        """
        from .timeutil import utc_range_for_wib_date, wib_date_from_iso

        out = []
        code = league_meta.get("football_data_code")
        if code:
            utc_from, utc_to = utc_range_for_wib_date(target_date)
            data = await self.fd.fetch_matches_by_competition(code, utc_from, utc_to)
            if data:
                for m in data:
                    if wib_date_from_iso(m.get("utcDate")) != target_date:
                        continue
                    home = (m.get("homeTeam") or {})
                    away = (m.get("awayTeam") or {})
                    out.append({
                        "id": m.get("id"),
                        "home": {"id": home.get("id"), "name": home.get("name")},
                        "away": {"id": away.get("id"), "name": away.get("name")},
                        "date": m.get("utcDate"),
                        "status": m.get("status"),
                        "source": "football_data",
                    })
                return out

        return out

    def close(self) -> None:
        """Release browser resources (flashscore client).

        The runner hard-exits via os._exit, so in production this is mostly
        hygiene for tests/embedding; the bot-level Chrome cleanup covers
        orphaned processes. Never raises.
        """
        for client in (self.fc, self.fc_secondary):
            if client is None:
                continue
            try:
                close = getattr(client, "close", None)
                if close:
                    close()
            except Exception:
                pass

    async def fetch_team_xg_history(
        self,
        team_name: str,
        league_meta: dict[str, Any],
        exclude: tuple[str, str, str] | None = None,
        nowgoal_client: Any = None,
        match_list: list[dict[str, Any]] | None = None,
        flashscore_client: Any = None,
        flashscore_ref: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Rolling PRE-MATCH xG averages for a team.

        Sources, in priority order (Plan B, 2026-08-17):
         1. NowGoal (PRIMARY, any league) -- the analysis page's ``match_list``
            carries the team's finished matches with a canonical ``match_id``
            (no fuzzy re-resolution); each ``live-{match_id}`` detail page
            reports full-time xG for both sides. Rolling window = max 3 most
            recent finished matches; the predicted fixture is excluded by
            date so it can never leak its own stats. Cached 24h per
            match_id; budget-guarded.
         2. Flashscore (fallback) -- team-results + per-match statistics
            renders (browser), same rolling-window construction.

        Returns {xg_for_avg, xg_against_avg, sample_size, source, xg_source,
        match_ids} or None. Never raises; a failure leaves xG features inert
        (model unchanged) and the caller proceeds without xG.
        """
        # 1. NowGoal primary (HTTP, any league): analysis match_list ->
        #    canonical match_id -> live-{id} FT xG.
        if (
            team_name
            and nowgoal_client is not None
            and match_list
            and not analysis_budget_exhausted(margin_seconds=45.0)
        ):
            try:
                from .nowgoal import _same_team as _ng_same_team

                xg_for: list[float] = []
                xg_against: list[float] = []
                match_ids: list[str] = []
                for row in match_list:
                    if len(xg_for) >= 3:
                        break
                    mid = row.get("match_id")
                    if not mid:
                        continue
                    # Anti-leak: skip a finished match on the predicted
                    # fixture date (a just-finished same-day match must not
                    # leak its own stats). Row dates are "YYYY-MM-DD ...".
                    if exclude and exclude[2]:
                        rd = (row.get("date") or "").strip()[:10]
                        if rd == exclude[2]:
                            continue
                    cache_key = f"ng_xg_{mid}"
                    xg = (
                        self.cache.get(cache_key, ttl_seconds=24 * 3600)
                        if self.cache is not None else None
                    )
                    if xg is None:
                        xg = await nowgoal_client.fetch_match_xg(str(mid))
                        if xg and self.cache is not None:
                            self.cache.set(cache_key, xg)
                    xg_h = (xg or {}).get("xg_home")
                    xg_a = (xg or {}).get("xg_away")
                    if not (
                        isinstance(xg_h, (int, float))
                        and isinstance(xg_a, (int, float))
                    ):
                        # friendly / no xG on this page -> next match
                        continue
                    # Attribution by the RENDERED row names -- never resolve
                    # the match again by fuzzy matching.
                    row_home = row.get("home") or ""
                    row_away = row.get("away") or ""
                    if _ng_same_team(team_name, row_home):
                        is_home = True
                    elif _ng_same_team(team_name, row_away):
                        is_home = False
                    else:
                        # Row does not involve this team -> never attribute.
                        continue
                    xg_for.append(float(xg_h if is_home else xg_a))
                    xg_against.append(float(xg_a if is_home else xg_h))
                    match_ids.append(str(mid))
                if xg_for and xg_against:
                    return {
                        "xg_for_avg": round(sum(xg_for) / len(xg_for), 4),
                        "xg_against_avg": round(sum(xg_against) / len(xg_against), 4),
                        "sample_size": len(xg_for),
                        "source": "nowgoal_xg",
                        "xg_source": "nowgoal_xg",
                        "match_ids": match_ids,
                    }
            except Exception as exc:  # noqa: BLE001 -- best-effort, next source
                logger.warning("nowgoal xG history failed (prediction unaffected): %s", exc)

        # 2. Flashscore fallback (browser): team-results + per-match
        #    statistics renders. The team results page (already the form
        #    source) carries a link per finished match (``match_url``); each
        #    finished match's statistics tab reports xG for both sides, so
        #    the rolling window can be rebuilt for ANY league flashscore
        #    covers. Budget-guarded: 1 results render + up to 3 statistics
        #    renders per team, all cached 24h.
        #
        # ``flashscore_client`` lets the caller route this chain to the
        # SECOND browser lane (fc_secondary) so home/away chains run in
        # parallel; default keeps legacy behaviour on the primary lane. The
        # client is stateless across calls (explicit slug/url args), and the
        # cache is shared, so results are identical regardless of lane --
        # a dead/unavailable secondary simply degrades to the primary.
        fc = flashscore_client or self.fc
        if (
            team_name
            and fc is not None
            and getattr(fc, "available", True)
            and not analysis_budget_exhausted(margin_seconds=60.0)
        ):
            try:
                from .flashscore import _norm_name, _squash, _suggest_team

                # Prefer the ALREADY-RESOLVED flashscore ref (from
                # resolve_match via league_meta._flashscore_match): the
                # suggest-API name lottery ("Den Haag" vs "ADO Den Haag" vs
                # "G.A. Eagles") could land on the wrong/empty team page,
                # silently killing the whole xG tier for that run.
                slug_id = None
                if (
                    isinstance(flashscore_ref, dict)
                    and flashscore_ref.get("slug")
                    and flashscore_ref.get("id")
                ):
                    slug_id = (str(flashscore_ref["slug"]), str(flashscore_ref["id"]))
                else:
                    slug_id = await asyncio.to_thread(_suggest_team, team_name)
                if slug_id:
                    slug, fsid = slug_id
                    rows_key = f"fs_team_results_{slug}_{fsid}"
                    rows = (
                        self.cache.get(rows_key, ttl_seconds=24 * 3600)
                        if self.cache is not None else None
                    )
                    if rows is None:
                        # 10 candidates: friendly/early-season rows often lack
                        # xG on flashscore, so more candidates keep the 3-match
                        # window filled during pre-season.
                        rows = await fc.fetch_team_results(slug, fsid, limit=10)
                        if rows and self.cache is not None:
                            self.cache.set(rows_key, rows)
                    if rows:
                        team_sq = _squash(_norm_name(slug.replace("-", " ")))
                        xg_for = []
                        xg_against = []
                        for r in rows:
                            if len(xg_for) >= 3:
                                break
                            url = r.get("match_url")
                            if not url:
                                continue
                            stats_key = f"fs_match_stats_{url}"
                            miss_key = f"{stats_key}_miss"
                            stats = (
                                self.cache.get(stats_key, ttl_seconds=24 * 3600)
                                if self.cache is not None else None
                            )
                            if (
                                stats is None
                                and self.cache is not None
                                and self.cache.get(miss_key, ttl_seconds=1800)
                            ):
                                # Negative cache (30 min): a recent render of
                                # this page yielded nothing -- re-rendering it
                                # every run wasted ~5-10s each time and made
                                # the xG tier flaky near budget edges.
                                continue
                            if stats is None:
                                rendered = await fc.fetch_match_statistics(url)
                                if rendered:
                                    stats = rendered
                                    if self.cache is not None:
                                        self.cache.set(stats_key, rendered)
                                elif self.cache is not None:
                                    self.cache.set(miss_key, {"miss": True})
                            xg_h = (stats or {}).get("xg_home")
                            xg_a = (stats or {}).get("xg_away")
                            if not (
                                isinstance(xg_h, (int, float))
                                and isinstance(xg_a, (int, float))
                            ):
                                continue
                            # Exclude the predicted fixture by date: the results
                            # page only shows finished matches, but a just-
                            # finished same-day match must not leak its own stats.
                            # Row dates are DD.MM. (sometimes with kickoff time).
                            if exclude and exclude[2]:
                                _dm = re.match(r"(\d{1,2})\.(\d{1,2})", (r.get("date") or "").strip())
                                if _dm and f"{_dm.group(2)}-{_dm.group(1)}" == exclude[2][5:]:
                                    continue
                            row_home_sq = _squash(_norm_name(r.get("home") or ""))
                            row_away_sq = _squash(_norm_name(r.get("away") or ""))
                            if team_sq == row_home_sq:
                                is_home = True
                            elif team_sq == row_away_sq:
                                is_home = False
                            else:
                                continue
                            xg_for.append(float(xg_h if is_home else xg_a))
                            xg_against.append(float(xg_a if is_home else xg_h))
                        if xg_for and xg_against:
                            return {
                                "xg_for_avg": sum(xg_for) / len(xg_for),
                                "xg_against_avg": sum(xg_against) / len(xg_against),
                                "sample_size": len(xg_for),
                                "source": "flashscore_xg",
                                "xg_source": "flashscore_xg",
                                "match_ids": [],
                            }
            except Exception as exc:  # noqa: BLE001 -- best-effort, next source
                logger.warning("flashscore xG history fallback failed (prediction unaffected): %s", exc)
        return None



