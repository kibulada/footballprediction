"""LiveScore secondary data source (verified real API).

VERIFIED SOURCE (2026-08): LiveScore's public API is served from the CDN host
``prod-cdn-public-api.lsmedia1.com``. The ``livescore.com`` domain is
government-blocked on the production network, but the ``lsmedia1.com`` CDN is
reachable directly. Endpoints below were extracted from the site's own
JavaScript (the ``PublicApi`` client, module 40745) and every one was verified
with a real HTTP 200 response:

    GET /v1/api/app/date/soccer/{YYYYMMDD}/{page}             fixtures
    GET /v1/api/app/lineups/soccer/{eid}?locale=en            starting XI + subs
    GET /v1/api/app/H2H/soccer/{eid}?locale=en                head-to-head
    GET /v1/api/app/form-e/soccer/{eid}?limit=10&locale=en    recent form (both teams)
    GET /v1/api/app/statistics/soccer/{eid}                   match statistics
    GET /v1/api/app/incidents/soccer/{eid}?locale=en          goal/card events
    GET /v1/api/app/leagueTable-s/soccer/{cat}/{stage}?locale=en  standings

All parsers are PURE and deterministic (unit-testable, no HTTP). Unsupported
or empty fields return ``missing()`` -- never fabricated. LiveScore team IDs
differ from Flashscore IDs, so matches are resolved by normalized names +
kickoff + competition (``same_match``), and the resolved ``Eid`` drives every
field endpoint.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .datasources import (
    FieldSample,
    FootballDataSource,
    available,
    canonical_match_identity,
    missing,
    normalize_team_name,
    same_match,
    teams_match,
)

logger = logging.getLogger(__name__)

USER_AGENT = "hermes-football/1.0 (local-advisory)"

DEFAULT_API_BASE_URL = "https://prod-cdn-public-api.lsmedia1.com"
# Failover hosts (2026-08-22): the lsmedia1 CDN is the same PublicApi the
# livescore.com site calls; the livescore.com hostnames serve the IDENTICAL
# /v1/api/app/* paths but sit on a domain several ISPs resolve-block, so they
# only answer through a proxy. Tried in order on network-level failures only
# (HTTP >= 400 does NOT rotate -- a 404/blocked page is an answer, not an
# outage). Verified live 2026-08-22: lsmedia1 200 w/ Stages payload;
# livescore.com hosts ConnectError direct.
FALLBACK_API_BASE_URLS = (
    "https://prod-cdn-public-api.livescore.com",
    "https://prod-public-api.livescore.com",
)
SPORT_CODE = "soccer"

FEED_TIMEOUT_SECONDS = 15.0
FEED_THROTTLE_SECONDS = 0.5

DATE_FEED_TTL_SECONDS = 15 * 60
LINEUPS_TTL_SECONDS = 30 * 60
STATS_TTL_SECONDS = 30 * 60
FORM_TTL_SECONDS = 6 * 3600
TABLE_TTL_SECONDS = 6 * 3600
H2H_TTL_SECONDS = 24 * 3600

MAX_FEED_PAGES = 3
KICKOFF_TOLERANCE_MINUTES = 180.0
FORM_WINDOW = 5


# --------------------------------------------------------------------------
# Pure parsers (deterministic, unit-testable, no HTTP coupling)
# --------------------------------------------------------------------------

def parse_esd(esd: Any) -> str | None:
    """``Esd`` 20260815173000 -> ISO UTC ``2026-08-15T17:30:00Z``."""
    digits = re.sub(r"\D", "", str(esd or ""))
    if len(digits) < 14:
        return None
    try:
        dt = datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.isoformat().replace("+00:00", "Z")


# P2-1: H2H window. Direct meetings older than this are stale evidence
# (squads / managers / divisions all change) -- they must not silently drive
# the H2H strength feature. ``apply_h2h_window`` filters dated meeting lists
# to the window and flags ``stale`` when nothing survives.
H2H_WINDOW_YEARS = 3


def _meeting_dt(m: dict[str, Any]) -> datetime | None:
    """Parse a meeting's date: livescore ``kickoff`` (ISO Z) or nowgoal
    ``date`` (``YYYY-MM-DD HH:MM:SS``). Naive timestamps are UTC."""
    raw = m.get("kickoff") if "kickoff" in (m or {}) else m.get("date")
    if not raw:
        return None
    try:
        s = str(raw).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def filter_h2h_recent(
    meetings: list[dict[str, Any]],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Keep meetings within ``H2H_WINDOW_YEARS`` of ``now`` (P2-1)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365 * H2H_WINDOW_YEARS)
    return [m for m in meetings if (d := _meeting_dt(m)) is not None and d >= cutoff]


def apply_h2h_window(
    h2h: dict[str, Any] | None,
    now: datetime | None = None,
    *,
    home_name: str | None = None,
) -> dict[str, Any] | None:
    """P2-1: restrict an H2H dict to the last ``H2H_WINDOW_YEARS``.

    - Dated sources (livescore ``meetings`` / nowgoal ``match_list``) are
      filtered in place; W/D/L counts are recomputed from the SURVIVORS so
      the strength numbers always match the displayed window (stale meetings
      never silently feed the feature).
    - Sources without dates (flashscore / football-data) cannot be windowed:
      counts are kept and ``h2h_in_window`` is ``None`` (not stale -- we
      simply cannot judge).
    - Metadata ``h2h_window`` / ``h2h_total_meetings`` / ``h2h_in_window``
      are attached, and ``h2h_relevance: stale`` is flagged when the window
      contains zero dated meetings.
    """
    if not isinstance(h2h, dict):
        return h2h
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365 * H2H_WINDOW_YEARS)
    _m = h2h.get("meetings")
    if isinstance(_m, list):
        total = len(_m)
    elif isinstance(_m, (int, float)):
        total = int(_m)
    else:
        total = int(h2h.get("matches") or h2h.get("count") or 0)
    h2h["h2h_window"] = f"{H2H_WINDOW_YEARS}y"
    h2h["h2h_total_meetings"] = total
    in_window: int | None = None

    meetings = h2h.get("meetings")
    if isinstance(meetings, list):
        kept = filter_h2h_recent(meetings, now)
        h2h["meetings"] = kept
        in_window = len(kept)
        h2h["h2h_in_window"] = in_window
        # Recompute W/D/L from the surviving FINISHED meetings, home-team
        # perspective (same rule parse_h2h uses; here the fixture home is
        # matched by name). Unattributable rows are simply not counted.
        if home_name:
            w = d = l = 0
            for m in kept:
                if str(m.get("status") or "").lower() not in ("finished", "ft"):
                    continue
                tr1, tr2 = _to_int(m.get("home_score")), _to_int(m.get("away_score"))
                if tr1 is None or tr2 is None:
                    continue
                hm = _squash(str(m.get("home") or ""))
                am = _squash(str(m.get("away") or ""))
                hf = _squash(str(home_name or ""))
                if hm == hf:
                    w, d, l = w + (tr1 > tr2), d + (tr1 == tr2), l + (tr1 < tr2)
                elif am == hf:
                    w, d, l = w + (tr2 > tr1), d + (tr2 == tr1), l + (tr2 < tr1)
            h2h["wins"], h2h["draws"], h2h["losses"] = w, d, l
    else:
        ml = h2h.get("match_list")
        if isinstance(ml, list):
            kept = [
                m for m in ml
                if (d := _meeting_dt(m)) is not None and d >= cutoff
            ]
            h2h["match_list"] = kept
            in_window = len(kept)
            h2h["h2h_in_window"] = in_window
            # NowGoal match_list rows carry ``result`` (W/D/L) already scored
            # from the HOME side -- recounting is exact.
            w = sum(1 for m in kept if m.get("result") == "W")
            d = sum(1 for m in kept if m.get("result") == "D")
            l = sum(1 for m in kept if m.get("result") == "L")
            h2h["wins"], h2h["draws"], h2h["losses"] = w, d, l
            h2h["matches"] = len(kept)
        else:
            h2h["h2h_in_window"] = None

    if in_window == 0:
        h2h["h2h_relevance"] = "stale"
    return h2h


def normalize_status(eps: Any) -> str:
    """LiveScore ``Eps`` -> unified status: scheduled | live | finished."""
    e = str(eps or "").upper()
    if e in ("NS", "POSTP", "CANCL", "ABAN", "INT", "SUSP", "DELAY"):
        return "scheduled"
    if e in ("FT", "AET", "PEN"):
        return "finished"
    return "live" if e else "unknown"


def _to_int(v: Any) -> int | None:
    try:
        return int(float(str(v)))
    except (ValueError, TypeError):
        return None


def _squash(name: str) -> str:
    """Minimal name normalization for H2H side attribution (accents /
    punctuation stripped, case-folded) -- mirrors datasources' comparator."""
    import unicodedata
    s = unicodedata.normalize("NFD", name.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


def _parse_event(
    ev: dict[str, Any],
    comp: str | None,
    country: str | None,
    category: str | None,
    stage_code: str | None,
) -> dict[str, Any] | None:
    t1 = ev.get("T1") or [{}]
    t2 = ev.get("T2") or [{}]
    t1 = t1[0] if isinstance(t1[0], dict) else {}
    t2 = t2[0] if isinstance(t2[0], dict) else {}
    home = (t1.get("Nm") or "").strip()
    away = (t2.get("Nm") or "").strip()
    if not home or not away:
        return None  # malformed event -> skipped, never fabricated
    tr1, tr2 = ev.get("Tr1"), ev.get("Tr2")
    return {
        "source_id": str(ev.get("Eid") or ""),
        "home": home,
        "away": away,
        "home_id": str(t1.get("ID") or "") or None,
        "away_id": str(t2.get("ID") or "") or None,
        "kickoff": parse_esd(ev.get("Esd")),
        "status": normalize_status(ev.get("Eps")),
        "status_raw": ev.get("Eps"),
        "competition": comp,
        "country": country,
        "category": category,   # league-table path segment (Ccd)
        "stage": stage_code,    # league-table path segment (Scd)
        "score": {
            "home": _to_int(tr1) if tr1 is not None else None,
            "away": _to_int(tr2) if tr2 is not None else None,
        },
    }


def parse_soccer_payload(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Deterministic parser for the verified ``/date/soccer`` feed schema."""
    fixtures: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return fixtures
    for stage in payload.get("Stages") or []:
        if not isinstance(stage, dict):
            continue
        comp = stage.get("CompN") or stage.get("Snm")
        country = stage.get("Cnm")
        category = stage.get("Ccd") or stage.get("CnmT")
        stage_code = stage.get("Scd")
        for ev in stage.get("Events") or []:
            if not isinstance(ev, dict):
                continue
            fx = _parse_event(ev, comp, country, category, stage_code)
            if fx is not None:
                fixtures.append(fx)
    return fixtures


def _player(p: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(p, dict):
        return None
    name = " ".join(x for x in (p.get("Fn"), p.get("Ln")) if x).strip()
    if not name:
        return None
    return {
        "name": name,
        "shirt": p.get("Snu"),
        "position": p.get("Pon") or p.get("Pos"),
    }


def parse_lineups(payload: dict[str, Any] | None, _fx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Verified ``/lineups`` schema -> {home, away: {formation, players, substitutes}}."""
    if not isinstance(payload, dict):
        return None
    result: dict[str, Any] = {}
    for unit in payload.get("Lu") or []:
        if not isinstance(unit, dict):
            continue
        side = "home" if unit.get("Tnb") == 1 else ("away" if unit.get("Tnb") == 2 else None)
        if not side:
            continue
        result[side] = {
            "formation": unit.get("Fo"),
            "players": [p for p in (_player(x) for x in unit.get("Ps") or []) if p],
            "substitutes": [p for p in (_player(x) for x in unit.get("IS") or []) if p],
        }
    return result or None


def parse_h2h(payload: dict[str, Any] | None, fx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Verified ``/H2H`` schema -> {wins, draws, losses, meetings} from the
    match's home-team perspective (home_id from the resolved fixture)."""
    if not isinstance(payload, dict):
        return None
    home_id = (fx or {}).get("home_id")
    meetings: list[dict[str, Any]] = []
    home_w = home_d = home_l = 0
    for ev in payload.get("H2H") or []:
        if not isinstance(ev, dict):
            continue
        t1 = (ev.get("T1") or [{}])[0] if (ev.get("T1") or [{}]) else {}
        t2 = (ev.get("T2") or [{}])[0] if (ev.get("T2") or [{}]) else {}
        t1 = t1 if isinstance(t1, dict) else {}
        t2 = t2 if isinstance(t2, dict) else {}
        tr1, tr2 = _to_int(ev.get("Tr1")), _to_int(ev.get("Tr2"))
        status = normalize_status(ev.get("Eps"))
        meeting = {
            "home": t1.get("Nm"), "away": t2.get("Nm"),
            "home_score": tr1, "away_score": tr2,
            "kickoff": parse_esd(ev.get("Esd")),
            "status": status,
        }
        meetings.append(meeting)
        # Draw-count integrity (2026-08-17): only FINISHED meetings may
        # contribute to the W/D/L tally -- a live 1-1 or a postponed match
        # with a partial score must never inflate the draw count. The meeting
        # itself is still listed (with its status) for context.
        if status != "finished":
            continue
        if tr1 is None or tr2 is None or home_id is None:
            continue
        if str(t1.get("ID")) == str(home_id):
            w, d, l = tr1 > tr2, tr1 == tr2, tr1 < tr2
        elif str(t2.get("ID")) == str(home_id):
            w, d, l = tr2 > tr1, tr2 == tr1, tr2 < tr1
        else:
            continue
        if w:
            home_w += 1
        elif d:
            home_d += 1
        else:
            home_l += 1
    return {"wins": home_w, "draws": home_d, "losses": home_l, "meetings": meetings}


def _team_form(team: dict[str, Any], events: list[Any]) -> dict[str, Any]:
    """Last-N W/D/L form from a team's verified ``EL`` event list (newest first)."""
    from .nowgoal import is_friendly_competition

    seq: list[str] = []
    goals: list[tuple[int, int]] = []
    recent: list[dict[str, Any]] = []
    tid = str(team.get("ID") or "")
    for ev in reversed(list(events or [])):  # oldest -> newest
        if not isinstance(ev, dict):
            continue
        # P3-2 parity (2026-08-22): pre-season friendlies must never reach the
        # form aggregates -- early-season windows are otherwise dominated by
        # them and inflate lambda's attack/defense inputs (Fortuna-AZ audit).
        stg = ev.get("Stg")
        comp = (stg or {}).get("Snm") if isinstance(stg, dict) else None
        if is_friendly_competition(comp):
            continue
        t1 = ev.get("T1") or [{}]
        t2 = ev.get("T2") or [{}]
        t1 = t1[0] if isinstance(t1[0], dict) else {}
        t2 = t2[0] if isinstance(t2[0], dict) else {}
        tr1, tr2 = _to_int(ev.get("Tr1")), _to_int(ev.get("Tr2"))
        if tr1 is None or tr2 is None:
            continue
        is_home = str(t1.get("ID") or "") == tid
        gf = tr1 if is_home else tr2
        ga = tr2 if is_home else tr1
        goals.append((gf, ga))
        seq.append("W" if gf > ga else ("D" if gf == ga else "L"))
        recent.append({
            "opponent": (t2 if is_home else t1).get("Nm"),
            "home_score": tr1, "away_score": tr2,
            "kickoff": parse_esd(ev.get("Esd")),
            "competition": (ev.get("Stg") or {}).get("Snm") if isinstance(ev.get("Stg"), dict) else None,
        })
    window = goals[-FORM_WINDOW:]
    return {
        "sequence": "-".join(seq[-FORM_WINDOW:]) or None,
        "gf_avg": round(sum(g for g, _ in window) / len(window), 2) if window else None,
        "ga_avg": round(sum(a for _, a in window) / len(window), 2) if window else None,
        "sample_size": len(window),
        "recent_goals": window or None,
        "recent": list(reversed(recent[-FORM_WINDOW:])),
    }


def parse_form(payload: dict[str, Any] | None, _fx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Verified ``/form-e`` schema -> {home, away} form (flashscore-compatible)."""
    if not isinstance(payload, dict):
        return None
    t1 = (payload.get("T1") or [{}])[0] if (payload.get("T1") or [{}]) else {}
    t2 = (payload.get("T2") or [{}])[0] if (payload.get("T2") or [{}]) else {}
    t1 = t1 if isinstance(t1, dict) else {}
    t2 = t2 if isinstance(t2, dict) else {}
    if not t1 or not t2:
        return None
    return {
        "home": _team_form(t1, t1.get("EL") or []),
        "away": _team_form(t2, t2.get("EL") or []),
    }


_STAT_KEYS = {
    "Shon": "shots_on_target", "Shof": "shots_off_target", "Shbl": "shots_blocked",
    "Crs": "corners", "Ycs": "yellow_cards", "Rcs": "red_cards", "Fls": "fouls",
    "Pss": "passes", "Ths": "possession", "Ofs": "offsides", "Gks": "saves",
}


def parse_statistics(payload: dict[str, Any] | None, _fx: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Verified ``/statistics`` schema -> {home, away} per-team match stats."""
    if not isinstance(payload, dict):
        return None
    result: dict[str, Any] = {}
    for s in payload.get("Stat") or []:
        if not isinstance(s, dict):
            continue
        side = "home" if s.get("Tnb") == 1 else ("away" if s.get("Tnb") == 2 else None)
        if not side:
            continue
        result[side] = {new: s.get(old) for old, new in _STAT_KEYS.items()}
    return result or None


def _table_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pos": row.get("rnk"), "team": row.get("Tnm"),
        "played": row.get("pld"), "wins": row.get("win"),
        "draws": row.get("drw"), "losses": row.get("lst"),
        "gf": row.get("gf"), "ga": row.get("ga"), "gd": row.get("gd"),
        "points": row.get("pts"),
    }


def parse_league_table(
    payload: dict[str, Any] | None,
    fx: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Verified ``/leagueTable-s`` schema -> {home, away} standing rows."""
    if not isinstance(payload, dict):
        return None
    rows: list[dict[str, Any]] = []
    for table in ((payload.get("LeagueTable") or {}).get("L") or []):
        if not isinstance(table, dict):
            continue
        for entry in table.get("Tables") or []:
            if not isinstance(entry, dict):
                continue
            rows.extend(x for x in (entry.get("team") or []) if isinstance(x, dict))
    home = (fx or {}).get("home")
    away = (fx or {}).get("away")
    out: dict[str, Any] = {}
    for row in rows:
        nm = row.get("Tnm")
        # Both names come from the SAME LiveScore feed, so exact normalized
        # equality is the safe matcher (the loose token matcher can false-
        # positive on substrings, e.g. "alaves" containing "la").
        norm = normalize_team_name(str(nm or ""))
        if home and norm and norm == normalize_team_name(str(home)):
            out["home"] = _table_row(row)
        elif away and norm and norm == normalize_team_name(str(away)):
            out["away"] = _table_row(row)
    return out or None


# --------------------------------------------------------------------------
# Client (verified lsmedia1.com public API)
# --------------------------------------------------------------------------

class LiveScoreClient:
    """Minimal httpx client for the verified LiveScore public API (no key)."""

    def __init__(
        self,
        base_url: str | None = None,
        throttle_seconds: float = FEED_THROTTLE_SECONDS,
        timeout: float = FEED_TIMEOUT_SECONDS,
    ) -> None:
        # None -> default (reachable lsmedia1.com CDN); "" -> explicitly
        # disabled (available False, no network); otherwise the given host.
        if base_url is None:
            self.base_url = DEFAULT_API_BASE_URL
        elif str(base_url).strip():
            self.base_url = str(base_url).strip().rstrip("/")
        else:
            self.base_url = ""
        # Failover rotation: primary first, then the livescore.com mirrors.
        self._hosts: list[str] = [self.base_url] if self.base_url else []
        for fb in FALLBACK_API_BASE_URLS:
            if fb not in self._hosts:
                self._hosts.append(fb)
        self._host_idx = 0
        self._throttle = throttle_seconds
        self._timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    BLOCKED_MARKERS = ("situs diblokir", "trustpositif", "website blocked", "komdigi")

    @staticmethod
    def _classify(status: int | None, text: str | None) -> str:
        if status is None:
            return "unreachable"
        if status == 403:
            return "forbidden"
        if status >= 400:
            return f"http_{status}"
        low = (text or "").lower()
        for marker in LiveScoreClient.BLOCKED_MARKERS:
            if marker in low:
                return "blocked"
        return "ok"

    async def _get_json(self, path: str) -> dict[str, Any] | None:
        """GET via the current host; on network-level failure rotate to the
        next mirror and remember the winner so later calls skip dead hosts."""
        if not self._hosts:
            return None
        last_exc: Exception | None = None
        for _ in range(len(self._hosts)):
            host = self._hosts[self._host_idx]
            url = f"{host}/{path.lstrip('/')}"
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(url, headers={"User-Agent": USER_AGENT})
            except httpx.HTTPError as exc:
                logger.warning("livescore network error (%s): %s", host, exc)
                last_exc = exc
                self._host_idx = (self._host_idx + 1) % len(self._hosts)
                if self._host_idx == 0:
                    break  # full rotation done, all hosts dead
                continue
            if resp.status_code >= 400:
                logger.warning("livescore http %s", resp.status_code)
                return None
            try:
                await asyncio.sleep(self._throttle)
                return resp.json()
            except (json.JSONDecodeError, ValueError):
                logger.warning("livescore response not JSON")
                return None
        if last_exc is not None:
            logger.warning("livescore all %d hosts unreachable", len(self._hosts))
        return None

    # -- verified endpoints (paths from the site's own PublicApi client) ----

    async def fetch_soccer_date(self, date: str, page: int = 0) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/date/soccer/{date}/{page}")

    async def fetch_lineups(self, event_id: str) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/lineups/{SPORT_CODE}/{event_id}?locale=en")

    async def fetch_h2h(self, event_id: str) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/H2H/{SPORT_CODE}/{event_id}?locale=en")

    async def fetch_form(self, event_id: str, limit: int = 10) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/form-e/{SPORT_CODE}/{event_id}?limit={limit}&locale=en")

    async def fetch_statistics(self, event_id: str) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/statistics/{SPORT_CODE}/{event_id}")

    async def fetch_incidents(self, event_id: str) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/incidents/{SPORT_CODE}/{event_id}?locale=en")

    async def fetch_league_table(self, category: str, stage: str) -> dict[str, Any] | None:
        return await self._get_json(f"/v1/api/app/leagueTable-s/{SPORT_CODE}/{category}/{stage}?locale=en")

    async def health(self) -> dict[str, Any]:
        if not self.available:
            return {"reachable": False, "reason": "no_config"}
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/api/app/date/soccer/{today}/0",
                    headers={"User-Agent": USER_AGENT},
                )
            reason = self._classify(resp.status_code, resp.text)
        except httpx.HTTPError as exc:
            reason = f"unreachable:{type(exc).__name__}"
        return {"reachable": reason == "ok", "reason": reason}


# --------------------------------------------------------------------------
# Adapter over the generic interface
# --------------------------------------------------------------------------

class LiveScoreDataSource(FootballDataSource):
    """Adapter exposing the verified LiveScore feed via ``FootballDataSource``.

    ``get_match`` resolves the fixture (Eid) by name+kickoff; the resolved Eid
    then drives lineups/H2H/form/statistics/standings. Unsupported or empty
    fields return ``missing()`` -- never fabricated.
    """

    name = "livescore"

    def __init__(
        self,
        client: LiveScoreClient | None = None,
        cache: Any | None = None,
        max_pages: int = MAX_FEED_PAGES,
    ) -> None:
        super().__init__()
        self.client = client or LiveScoreClient()
        self.cache = cache
        self.max_pages = max(1, int(max_pages))
        self._fixture_cache: dict[tuple, dict[str, Any] | None] = {}

    def _active(self) -> bool:
        return bool(self.client and self.client.available)

    # -- cache helpers -------------------------------------------------------

    async def _cached_date(self, date: str, page: int) -> dict[str, Any] | None:
        key = f"livescore_date_{date}_{page}"
        if self.cache is not None:
            hit = self.cache.get(key, ttl_seconds=DATE_FEED_TTL_SECONDS)
            if hit is not None:
                return hit
        payload = await self.client.fetch_soccer_date(date, page)
        if payload is not None and self.cache is not None:
            self.cache.set(key, payload)
        return payload

    async def _cached_endpoint(self, key: str, fetch, ttl: int) -> dict[str, Any] | None:
        if self.cache is not None:
            hit = self.cache.get(key, ttl_seconds=ttl)
            if hit is not None:
                return hit
        raw = await fetch()
        if raw is not None and self.cache is not None:
            self.cache.set(key, raw)
        return raw

    # -- fixture resolution ----------------------------------------------------

    def _candidate_dates(self, kickoff: str | None) -> list[str]:
        dates = set()
        if kickoff:
            try:
                s = kickoff.replace("Z", "+00:00") if kickoff.endswith("Z") else kickoff
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dates.add(dt.astimezone(timezone.utc).strftime("%Y%m%d"))
            except (ValueError, TypeError):
                pass
        dates.add(datetime.now(timezone.utc).strftime("%Y%m%d"))
        return sorted(dates)

    def _ref_key(self, ref: dict[str, Any]) -> tuple:
        return (
            normalize_team_name(ref.get("home")),
            normalize_team_name(ref.get("away")),
            ref.get("kickoff"),
        )

    def _orient(self, fx: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
        ordered = teams_match(ref.get("home"), fx.get("home")) and teams_match(
            ref.get("away"), fx.get("away")
        )
        if ordered:
            return fx
        score = fx.get("score") or {}
        return {
            "source_id": fx.get("source_id"),
            "home": fx.get("away"), "away": fx.get("home"),
            "home_id": fx.get("away_id"), "away_id": fx.get("home_id"),
            "kickoff": fx.get("kickoff"),
            "status": fx.get("status"), "status_raw": fx.get("status_raw"),
            "competition": fx.get("competition"), "country": fx.get("country"),
            "category": fx.get("category"), "stage": fx.get("stage"),
            "score": {"home": score.get("away"), "away": score.get("home")},
        }

    async def _find_first_fixture(self, ref: dict[str, Any]) -> dict[str, Any] | None:
        for date in self._candidate_dates(ref.get("kickoff")):
            for page in range(self.max_pages):
                payload = await self._cached_date(date, page)
                for fx in parse_soccer_payload(payload):
                    if same_match(ref, fx, kickoff_tolerance_minutes=KICKOFF_TOLERANCE_MINUTES):
                        return self._orient(fx, ref)
        return None

    async def _resolve_fixture(self, ref: dict[str, Any]) -> dict[str, Any] | None:
        key = self._ref_key(ref)
        if key not in self._fixture_cache:
            self._fixture_cache[key] = await self._find_first_fixture(ref)
        return self._fixture_cache[key]

    # -- field extraction ------------------------------------------------------

    async def get_match(self, ref: dict[str, Any]) -> FieldSample:
        if not self._active():
            return missing()
        try:
            fx = await self._resolve_fixture(ref)
            if fx is None:
                return missing()
            from .timeutil import utc_now_iso
            return available(
                canonical_match_identity(
                    home=fx["home"], away=fx["away"],
                    kickoff=fx["kickoff"], competition=fx["competition"],
                ),
                utc_now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("livescore fixture lookup failed: %s", type(exc).__name__)
            return missing()

    async def _event_field(self, ref, kind: str, fetch, parse, ttl: int) -> FieldSample:
        if not self._active():
            return missing()
        try:
            fx = await self._resolve_fixture(ref)
            eid = (fx or {}).get("source_id")
            if fx is None or not eid:
                return missing()
            raw = await self._cached_endpoint(
                f"livescore_{kind}_{eid}", lambda: fetch(eid), ttl
            )
            if not raw:
                return missing()
            parsed = parse(raw, fx)
            if not parsed:
                return missing()
            from .timeutil import utc_now_iso
            return available(parsed, utc_now_iso())
        except Exception as exc:  # noqa: BLE001
            logger.warning("livescore %s failed: %s", kind, type(exc).__name__)
            return missing()

    async def get_lineup(self, ref: dict[str, Any]) -> FieldSample:
        return await self._event_field(
            ref, "lineups", self.client.fetch_lineups, parse_lineups, LINEUPS_TTL_SECONDS
        )

    async def get_h2h(self, ref: dict[str, Any]) -> FieldSample:
        return await self._event_field(
            ref, "h2h", self.client.fetch_h2h, parse_h2h, H2H_TTL_SECONDS
        )

    async def get_form(self, ref: dict[str, Any]) -> FieldSample:
        return await self._event_field(
            ref, "form", self.client.fetch_form, parse_form, FORM_TTL_SECONDS
        )

    async def get_statistics(self, ref: dict[str, Any]) -> FieldSample:
        return await self._event_field(
            ref, "stats", self.client.fetch_statistics, parse_statistics, STATS_TTL_SECONDS
        )

    async def get_standings(self, ref: dict[str, Any]) -> FieldSample:
        if not self._active():
            return missing()
        try:
            fx = await self._resolve_fixture(ref)
            category = (fx or {}).get("category")
            stage = (fx or {}).get("stage")
            if fx is None or not category or not stage:
                return missing()
            raw = await self._cached_endpoint(
                f"livescore_table_{category}_{stage}",
                lambda: self.client.fetch_league_table(category, stage),
                TABLE_TTL_SECONDS,
            )
            if not raw:
                return missing()
            parsed = parse_league_table(raw, fx)
            if not parsed:
                return missing()
            from .timeutil import utc_now_iso
            return available(parsed, utc_now_iso())
        except Exception as exc:  # noqa: BLE001
            logger.warning("livescore standings failed: %s", type(exc).__name__)
            return missing()

    async def get_injuries(self, ref: dict[str, Any]) -> FieldSample:
        return missing()  # no verified injuries/suspensions field in the API
