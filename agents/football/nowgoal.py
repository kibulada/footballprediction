"""NowGoal (nowgoal.com family) odds client.

NowGoal's own internal AJAX API -- the same data family as win007/007sport
-- serves the site's live odds. The client exposes pre-match 1X2, Over/Under
and Asian-handicap odds normalized into the exact The-Odds-API payload shape
so ``analyse.extract_h2h_entries`` / ``extract_market_totals`` work unchanged.

Verified endpoints (reference: MaksimPerepeliuk/nowgoal-scraper and
pantherchild14/nowgoal, 2026-08):

  schedule  GET {base}/ajax/SoccerAjax?type=6&date=YYYY-MM-DD&order=time&timezone=0&flesh={ms}
            -> response body is JS array source:  B[i]=[league rows...]
                                                 A[i]=[match rows...]
            Match row fields (verified positions): 0 id, 1 league index,
            2 home id, 3 away id, 4 home name, 5 away name, 6-11 year /
            0-based month / day / hour / minute / second, 12 status,
            13 home score, 14 away score.

  odds      GET {base}/ajax/soccerajax?type=14&t=1&id={match_id}&h=0&flesh={ms}
            -> {"ErrCode":0, "Data":{"mixodds":[{cid, cn, ah, euro, ou, ...}]}}
            t=11 -> Data.roddsList (same item shape). NOTE: this is NOT a
            closing line -- verified live 2026-08-16 it serves the
            POST-SETTLEMENT final prices (winner ~1.01, losers 50-500, the
            result embedded in the price). The real closing line is the
            ``l`` (last pre-match) leg of the t=1 mixodds feed.

Verified live 2026-08-14 (via Tor, www.nowgoal.net): the mixodds item shape
is ``{cid, cn, euro: {f,l,r:{u,g,d}, hr}, ou: {...}, ah: {...}}`` where
``f`` = first/opening odds, ``l`` = last pre-match odds, ``r`` = realtime
in-play series (never used for pre-match predictions). Key mapping inside
each market is ``u`` = home/over, ``g`` = draw/line, ``d`` = away/under;
values are strings (prices "2.2", lines "0.25"), empty string when a
bookmaker has no odds. Handicap/goal lines move in quarter steps (0.25,
0.5, ...) and may be 0 (level ball). The schedule kickoff month is
0-based (2026,7,14 == 2026-08-14).

Additional endpoints verified live 2026-08-15 (via Tor, www.nowgoal.net):

  analysis  GET {base}/analysis/{id}     -> 301 -> the /match/h2h-{id} page.
            Server-renders form tables (tr1_x / tr2_x), the H2H table
            (tr3_x), standings, fixtures and injuries -- parsed by
            ``_parse_analysis`` + ``_parse_standings/_parse_fixtures/
            _parse_injuries`` (context data, never model inputs).
  detail    GET {base}/match/live-{id}   -> server-renders team statistics
            (recent 3/10 matches), HT/FT statistics (last 2 seasons),
            goal-timing distributions (last 30/50) and lineups with
            formations -- parsed by ``_parse_detail``.
  lineups   GET {base}/ajax/soccerajax?type=18&id={match_id}
            -> {"Data":{"hList":[...], "gList":[...]}} full player lists
            with position/valid/rating (``valid`` = starter).
  splits    GET {base}/ajax/soccerajax?type=22&id={match_id}
            -> AH/OU/OP historical W/D/L splits by scope (All/Main/Same).
  trend     GET {base}/ajax/soccerajax?type=14&t=20&id={match_id}&cid={company}
            -> {"Data":{"op":[...], "ah":[...], "ou":[...]}} -- the FULL
            timestamped odds-movement series behind the oddscomp "Trends"
            popup (``_oddsDetailWin.open``). Every recorded odds change per
            bookmaker per market, each row carrying ``mt`` (unix seconds),
            ``ht`` (match minute; empty = pre-match), ``hs``/``gs`` (score
            at that moment), ``close`` and ``odds:{u,g,d}``. This is the
            one-call source of pre-match movement history -- no polling
            needed -- parsed by ``fetch_odds_trend`` (verified live
            2026-08-16: Bet365/Sbobet/1xBet rows span ~2 days pre-match).
  realtime  the ``r`` leg of type=14 mixodds is exposed separately by
            ``fetch_live_odds`` (the oddscomp page's "Live" column) and as
            ``snapshot: "live"`` rows in ``fetch_odds_history``.

The parsers still accept every shape documented for this data family (list /
dict / JSON string, ``[line, over, under]`` positional order for totals) and
STRICTLY validate the result (prices > 1.0, quarter-step lines). A wrong
shape fails validation and returns None -- the caller simply proceeds without
nowgoal odds, so bad data can never reach the model.

Domain mirrors rotate (verified 2026-08-14: nowgoal3.com is parked, the
nowgoal.net / nowgoal26.com family is live). The client accepts a list of
mirrors and rotates automatically -- on transport/HTTP errors, or when a
mirror answers HTTP 200 with a parked page / ISP block page instead of real
content -- keeping the last working mirror active. Override the primary base
URL via the NOWGOAL_BASE_URL env var.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Verified live 2026-08-14 (nowgoal3.com is parked / redirects to a domain
# parking page; www.nowgoal.net and the nowgoal26.com family answer the real
# site). Mirrors rotate, so the check command probes several.
DEFAULT_BASE_URL = "https://live10.nowgoal26.com/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_MAX_BODY = 5 * 1024 * 1024  # schedule bodies are JS sources; hard cap
# Circuit breaker tuning (see NowGoalClient.__init__): ``_BREAKER_THRESHOLD``
# consecutive all-mirror transport failures arm a short-circuit whose cooldown
# ESCALATES per consecutive open (90s -> 180s -> 360s cap). While the whole
# network path is down every call already ends in None -- output identical,
# only repeated opens in one long-running process stop re-paying full
# all-mirror failure rounds. ANY successful response resets both the strike
# counter and the escalation.
_BREAKER_THRESHOLD = 2
_BREAKER_COOLDOWN_BASE = 90.0
_BREAKER_COOLDOWN_MAX = 360.0

# Fast-fail CONNECT (Opsi 1, 2026-08-23): mirror yang reachable menyelesaikan
# TCP connect dalam <2s; yang unreachable sebelumnya membakar timeout penuh
# (default 15s) PER mirror PER panggilan -- teramati live 2026-08-23: 4 mirror
# mati serentak ~= 1 menit terbuang per putaran _get. Hanya fase connect yang
# dipangkas; read/write/pool tetap timeout penuh sehingga respons healthy-yang-
# lambat tetap selesai (kelengkapan data tidak berubah -- yang dipotong hanya
# menunggu kegagalan yang sudah pasti).
_CONNECT_TIMEOUT = 4.0

# Company id -> display name. Ids verified live 2026-08-14 from real
# mixodds payloads (cid/cn pairs served by the API itself). Unknown ids fall
# back to a stable "NowGoal-<cid>" label (display only -- never a model input).
# P3-2: competitions that must never pollute the recent-form / H2H
# aggregates. A preseason friendly (e.g. Aris 6-2 Napoli) carries no
# competitive signal -- squads rotate and intensity differs -- so its rows
# are skipped BEFORE they contribute to sequence / gf / ga / W-D-L.
EXCLUDED_COMPETITIONS: frozenset[str] = frozenset({
    "club friendlies", "club friendly", "pre-season", "friendly",
    "international friendlies", "intl friendlies",
})


def _is_excluded_competition(name: str | None) -> bool:
    """True when the row's league name is a friendly/pre-season context."""
    if not name:
        return False
    norm = unicodedata.normalize("NFD", str(name).lower().strip())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return norm in EXCLUDED_COMPETITIONS


_FRIENDLY_PHRASES = ("friendl", "pre-season", "preseason")


def is_friendly_competition(name: str | None) -> bool:
    """Substring-tolerant friendly/pre-season detector (2026-08-22).

    ``_is_excluded_competition`` is exact-match and misses the live spellings
    other feeds emit ("Club Friendlies 2026", "Featured Club Friendlies",
    "Preseason"). Phrase containment catches all of them and cannot false-
    positive: no real competition name contains 'friendl' / 'pre-season' /
    'preseason'. Shared by every form producer so lambda's attack/defense
    inputs never ingest pre-season scorelines (Fortuna-AZ 2026-08-22 audit:
    friendly-polluted form drove lambda_total 3.96 vs a market at ~3.2).
    """
    if not name:
        return False
    norm = unicodedata.normalize("NFD", str(name).lower().strip())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    if _is_excluded_competition(norm):
        return True
    return any(phrase in norm for phrase in _FRIENDLY_PHRASES)


def filter_recent_matches(
    matches: list[dict[str, Any]],
    exclude: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """P3-2: drop rows whose ``competition`` is an excluded context
    (Club Friendlies / Pre-Season) from a recent-match list."""
    ex = {str(e).lower().strip() for e in (exclude or EXCLUDED_COMPETITIONS)}
    out: list[dict[str, Any]] = []
    for m in matches:
        comp = str((m or {}).get("competition") or "").lower().strip()
        if comp in ex:
            continue
        out.append(m)
    return out


KNOWN_COMPANIES: dict[int, str] = {
    8: "Bet365",
    31: "Sbobet",
    50: "1xBet",
    17: "Mansion88",
    24: "12bet",
    3: "Crown",
    42: "18Bet",
    12: "Easybet",
    1: "Macauslot",
    4: "Ladbrokes",
    14: "Vcbet",
    19: "Interwetten",
    2: "Betfair",
    177: "Pinnacle",
    474: "SBOBET",
    816: "Marathonbet",
    1047: "1xBet",
}

# Schedule JS rows: B[i]=[league rows], A[i]=[match rows].
_SCHEDULE_RE_B = re.compile(r"B\[(\d+)\]=\[(.*?)\];", re.DOTALL)
_SCHEDULE_RE_A = re.compile(r"A\[(\d+)\]=\[(.*?)\];", re.DOTALL)


# ---- lenient value coercion (never raises) -------------------------------

def _coerce_float(value: Any) -> float | None:
    """Parse a number from int/float/string; None on junk or non-positive."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if f > 0 else None
    if isinstance(value, str):
        m = re.search(r"\d+(?:\.\d+)?", value.strip())
        if m:
            f = float(m.group())
            return f if f > 0 else None
    return None


def _coerce_odds(value: Any) -> float | None:
    """Decimal odds price: must be a number > 1.0."""
    f = _coerce_float(value)
    return f if f is not None and f > 1.0 else None


def _coerce_line(value: Any) -> float | None:
    """Handicap/goal line: quarter-step number, NEGATIVE allowed.

    ``_coerce_float`` rejects <= 0, but a 0 handicap (level ball) and a
    negative Asian-Handicap line (home gives goals, e.g. "-1") are real
    markets and must keep their sign.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if -6.5 <= f <= 6.5 else None
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.strip())
        if m:
            f = float(m.group())
            return f if -6.5 <= f <= 6.5 else None
    return None


def _coerce_odds_hk(value: Any) -> float | None:
    """Price that may be Hong-Kong odds -> decimal odds, or None.

    Verified live 2026-08-14: nowgoal serves ``ou`` and ``ah`` prices in
    Hong-Kong format (payout ratio, decimal odds minus 1: 0.85 == 1.85),
    while ``euro`` is plain decimal. A real decimal price is always > 1.0, so
    any value <= 1.0 must be HK; the whole family uses HK for these two
    markets, so values > 1.0 are also treated as HK (decimal = value + 1).
    """
    f = _coerce_float(value)
    if f is None:
        return None
    dec = f + 1.0
    return dec if dec > 1.0 else None


def _is_plausible_line(line: float) -> bool:
    """Goal/handicap lines move in quarter steps: 0.25, 0.5, ..., <= 6.5.

    Doubles as the disambiguator between [line, price, price] and
    [price, price, line] positional layouts: an odds price like 1.85 is not a
    quarter step, so it can never be accepted as the line.
    """
    if not (0.0 < line <= 6.5):
        return False
    return abs(line * 4 - round(line * 4)) < 1e-9


def _is_plausible_line0(line: float) -> bool:
    """Like ``_is_plausible_line`` but allows 0.0 and NEGATIVE lines.

    Used only for the verified ``{u,g,d}`` dict shape where ``g`` is
    unambiguously the line -- no positional disambiguation needed, a 0.0
    handicap ("level ball") and a negative Asian-Handicap line are real
    markets.
    """
    if not (-6.5 <= line <= 6.5):
        return False
    return abs(line * 4 - round(line * 4)) < 1e-9


def _unwrap_market(value: Any) -> Any:
    """Verified live wrapper ``{f, l, r, hr}`` -> the inner ``{u,g,d}`` dict.

    ``f`` = opening odds, ``l`` = last pre-match odds, ``r`` = realtime
    in-play series. For pre-match predictions we want ``l`` (latest before
    kickoff), falling back to ``f`` (opening) when ``l`` is empty; ``r`` is
    never used. Non-wrapper values (lists, plain price dicts, JSON strings)
    pass through unchanged and are handled by the lenient paths below.
    """
    if not isinstance(value, dict):
        return value
    if not any(k in value for k in ("f", "l", "r")):
        return value
    for k in ("l", "f"):
        inner = value.get(k)
        if isinstance(inner, dict) and any(v not in (None, "") for v in inner.values()):
            return inner
    return value


def _opening_leg(value: Any) -> Any:
    """The ``f`` (opening) leg of the verified ``{f,l,r,hr}`` wrapper.

    Returns the opening ``{u,g,d}`` dict when present, else the wrapper
    itself (so the lenient parse paths can try their normal unwrapping).
    Used only to expose opening prices for market-movement detection; the
    prediction path keeps using ``l`` (last pre-match) via ``_unwrap_market``.
    """
    if isinstance(value, dict) and any(k in value for k in ("f", "l", "r")):
        inner = value.get("f")
        if isinstance(inner, dict) and any(v not in (None, "") for v in inner.values()):
            return inner
    return value


def _realtime_leg(value: Any) -> Any:
    """The ``r`` (realtime in-play) leg of the ``{f,l,r,hr}`` wrapper.

    Returns the realtime ``{u,g,d}`` dict when present (the "Live" column on
    the oddscomp page), else None -- a plain price dict without the wrapper
    carries no realtime snapshot. The ``hr`` flag marks the row as having
    realtime odds. Used only by ``fetch_live_odds`` / the live snapshot in
    ``fetch_odds_history`` -- pre-match predictions stay on ``l``.
    """
    if not isinstance(value, dict) or "r" not in value:
        return None
    inner = value.get("r")
    if isinstance(inner, dict) and any(v not in (None, "") for v in inner.values()):
        return inner
    return None


def _as_seq(value: Any) -> list[Any] | None:
    """Normalize a value that may be a list, dict, or JSON string -> list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = json.loads(s)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return list(parsed.values())
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if parts and all(_coerce_float(p) is not None for p in parts):
            return parts
    return None


def _pick(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """First present key (case-insensitive; numeric keys normalized to str).

    The feed mixes spellings ("X" vs "x", "Over" vs "over") across rows, so
    lookups must not be case-sensitive.
    """
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for k in keys:
        hit = lowered.get(k.lower())
        if hit is not None:
            return hit
    return None


# ---- market parsers (lenient in, strict out) -----------------------------

def parse_euro(value: Any) -> dict[str, float] | None:
    """1X2 (European) odds -> {"home","draw","away"} or None.

    Accepts the verified live shape ``{f,l,r:{u,g,d}}`` (u=home, g=draw,
    d=away; wrapper unwrapped to the last pre-match odds), plain dicts keyed
    home/draw/away (or 1/X/2, h/d/a) and positional sequences [h, d, a, ...]
    (extra trailing numbers are ignored). Every price must be > 1.0 or the
    whole market is rejected.
    """
    value = _unwrap_market(value)
    if isinstance(value, dict):
        # Verified live shape: u=home, g=draw, d=away, always all three
        # present (empty string = no odds).
        if all(k in value for k in ("u", "g", "d")):
            home = _coerce_odds(value.get("u"))
            draw = _coerce_odds(value.get("g"))
            away = _coerce_odds(value.get("d"))
            if home is not None and draw is not None and away is not None:
                return {"home": home, "draw": draw, "away": away}
            return None
        pairs = (
            ("home", ("home", "h", "1")),
            ("draw", ("draw", "d", "x")),
            ("away", ("away", "a", "2")),
        )
        out: dict[str, float] = {}
        for pos, keys in pairs:
            price = _coerce_odds(_pick(value, keys))
            if price is None:
                return None
            out[pos] = price
        return out
    seq = _as_seq(value)
    if not seq:
        return None
    prices = [p for v in seq if (p := _coerce_odds(v)) is not None]
    if len(prices) < 3:
        return None
    return {"home": prices[0], "draw": prices[1], "away": prices[2]}


def parse_ou(value: Any) -> dict[str, float] | None:
    """Over/Under odds -> {"over","under","line"} or None.

    Accepts the verified live shape ``{f,l,r:{u,g,d}}`` (u=over, g=line,
    d=under; wrapper unwrapped to the last pre-match odds) and unambiguous
    dicts (over/under/o/u + line/point/hdc). Positional arrays follow this
    data family's convention ``[line, over, under]``; the quarter-step line
    check disambiguates it from ``[over, under, line]``.
    """
    value = _unwrap_market(value)
    if isinstance(value, dict):
        # Verified live shape: u=over, g=line, d=under (HK odds; empty string
        # = no odds). Requires all three keys -- lenient dicts may use ``u``
        # for "under" and must not be misread as the live shape.
        if all(k in value for k in ("u", "g", "d")):
            over = _coerce_odds_hk(value.get("u"))
            under = _coerce_odds_hk(value.get("d"))
            line = _coerce_line(value.get("g"))
            if (
                over is not None
                and under is not None
                and line is not None
                and _is_plausible_line0(line)
            ):
                return {"over": over, "under": under, "line": line}
            return None
        line = _coerce_float(_pick(value, ("line", "point", "hdc", "total", "goals")))
        over = _coerce_odds(_pick(value, ("over", "o", "up", "big")))
        under = _coerce_odds(_pick(value, ("under", "u", "down", "small")))
        if line is None or over is None or under is None:
            return None
        return {"over": over, "under": under, "line": line}
    seq = _as_seq(value)
    if not seq or len(seq) < 3:
        return None
    nums = [n for v in seq if (n := _coerce_float(v)) is not None]
    if len(nums) < 3:
        return None
    # [line, over, under] (family convention)
    if _is_plausible_line(nums[0]) and nums[1] > 1.0 and nums[2] > 1.0:
        return {"over": nums[1], "under": nums[2], "line": nums[0]}
    # [over, under, line] fallback
    if _is_plausible_line(nums[2]) and nums[0] > 1.0 and nums[1] > 1.0:
        return {"over": nums[0], "under": nums[1], "line": nums[2]}
    return None


def _ah_side_line(line: float | None, side: str) -> float | None:
    """Convert NowGoal's raw AH line (AWAY handicap) to one side's line.

    NowGoal quotes the AH line from the AWAY side (verified live against the
    independent The Odds API market: for Excelsior vs PSV the real market is
    Home +1.25 / Away -1.25, yet NowGoal serves ``g = -1.25``). The normalized
    payload and the signal engine use the HOME-handicap convention, so the
    home side carries the negated line and the away side the raw line.
    """
    if line is None:
        return None
    return -line if side == "home" else line


def parse_ah(value: Any) -> dict[str, float] | None:
    """Asian handicap -> {"home","away","line"} or None (diagnostic only).

    Accepts the verified live shape ``{f,l,r:{u,g,d}}`` (u=home, g=line,
    d=away; wrapper unwrapped to the last pre-match odds) and unambiguous
    dicts (home/h + line/hdc/handicap). Positional arrays follow this data
    family's convention ``[line, home, away]`` with quarter-step
    disambiguation.

    NOTE: the raw ``line`` is the AWAY handicap (negative = away gives,
    positive = away receives). Convert with ``_ah_side_line`` when emitting a
    normalized payload so the Home outcome carries the HOME handicap.
    """
    value = _unwrap_market(value)
    if isinstance(value, dict):
        # Verified live shape: u=home, g=line, d=away (HK odds; empty string
        # = no odds). Requires all three keys -- lenient dicts may use ``u``
        # for "under" and must not be misread as the live shape.
        if all(k in value for k in ("u", "g", "d")):
            home = _coerce_odds_hk(value.get("u"))
            away = _coerce_odds_hk(value.get("d"))
            line = _coerce_line(value.get("g"))
            if (
                home is not None
                and away is not None
                and line is not None
                and _is_plausible_line0(line)
            ):
                return {"home": home, "away": away, "line": line}
            return None
        line = _coerce_float(_pick(value, ("line", "hdc", "handicap", "point")))
        home = _coerce_odds(_pick(value, ("home", "h", "主")))
        away = _coerce_odds(_pick(value, ("away", "a", "客")))
        if line is None or home is None or away is None:
            return None
        return {"home": home, "away": away, "line": line}
    seq = _as_seq(value)
    if not seq or len(seq) < 3:
        return None
    nums = [n for v in seq if (n := _coerce_float(v)) is not None]
    if len(nums) < 3:
        return None
    # [line, home, away] convention, quarter-step disambiguation
    if _is_plausible_line(nums[0]) and nums[1] > 1.0 and nums[2] > 1.0:
        return {"home": nums[1], "away": nums[2], "line": nums[0]}
    if _is_plausible_line(nums[2]) and nums[0] > 1.0 and nums[1] > 1.0:
        return {"home": nums[0], "away": nums[1], "line": nums[2]}
    return None


# ---- schedule (type=6) ---------------------------------------------------

def _clean_token(token: str) -> str:
    return token.strip().strip("'").strip('"')


def _parse_match_row(row: str, leagues: dict[str, dict[str, str]]) -> dict[str, Any] | None:
    """One A[i] row -> normalized match dict (verified field positions).

    Row example (from reference code)::
      2346247,29,8856,1036,'Deportes Santa Cruz','Deportes Temuco',
      '2023,5,5,00,00,00',-1,1,1,1,0,1,0,4,1,'13','4','','',42,'','',3,8

    Split naively on commas exactly like the reference implementation (the
    quoted datetime field is comma-separated internally and must split), then
    strip quotes per token. Team names never contain commas in this feed.
    """
    p = [_clean_token(t) for t in row.split(",")]
    if len(p) < 15:
        return None
    match_id = p[0]
    if not match_id.isdigit():
        return None
    try:
        year, month, day = int(p[6]), int(p[7]) + 1, int(p[8])
        hour, minute = int(p[9]), int(p[10])
        second = int(p[11]) if len(p) > 11 and p[11].isdigit() else 0
        # timezone=0 requested; schedule times are treated as UTC (assumption
        # documented -- odds themselves never depend on this value).
        kickoff = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
    except (ValueError, IndexError):
        return None
    league = leagues.get(p[1], {})
    return {
        "match_id": match_id,
        "home": p[4],
        "away": p[5],
        "home_id": p[2],
        "away_id": p[3],
        "kickoff": kickoff,
        "status": p[12],
        # Phase 2.2: the feed marks finished matches with status "-1" and
        # carries the final score at positions 13/14 -- used to trigger the
        # closing-price capture (the ``l`` leg, last pre-match) without
        # waiting for settle. t=11/roddsList is NOT used: it serves
        # result-embedded final prices (winner ~1.01), not a closing line.
        "finished": p[12] == "-1" if len(p) > 12 else False,
        "score": (
            f"{p[13]}-{p[14]}"
            if len(p) > 14 and p[13].lstrip("-").isdigit() and p[14].lstrip("-").isdigit()
            else None
        ),
        "league_id": league.get("league_id"),
        "league_name": league.get("name"),
        "source": "nowgoal",
    }


# ---- tolerant team-name matching (mirrors oddspapi._same_team) -----------

_STROKE_LETTERS = str.maketrans(
    {
        "\u00f8": "o", "\u0142": "l", "\u0111": "d", "\u0127": "h",
        "\u0131": "i", "\u014b": "n", "\u00df": "ss",
    }
)
_TEAM_PREFIXES = {
    "fk", "fc", "nk", "cd", "sc", "pfc", "ifk", "ss", "rc", "ca",
    "ec", "cr", "se", "ac", "cf", "us", "sd", "de", "sv", "sk",
}


def _is_youth_reserve_name(name: str) -> bool:
    """True when a team name carries a youth/reserve marker ("U19",
    "Reserves", "A2", "B"), so a schedule containing BOTH the senior side
    and its youth side on the same day never silently resolves to the
    youth match for a plain team query (verified 2026-08-17: Galatasaray vs
    Corum Belediyespor vs the U19 fixture). Markers must be tokens, not
    substrings: "U19" matches, "Fu19nction" does not.
    """
    s = (name or "").lower()
    tokens = re.split(r"[^a-z0-9]", s)
    for t in tokens:
        if not t:
            continue
        if re.fullmatch(r"u\d{1,2}", t):  # u19, u21, u23
            return True
        if t in ("reserve", "reserves", "youth", "junior", "juniors", "b", "a2"):
            return True
    return False


def _norm_team(name: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", name or "")
    s = s.lower().translate(_STROKE_LETTERS)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _same_team(a: str, b: str) -> bool:
    na, nb = _norm_team(a), _norm_team(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    for pref in _TEAM_PREFIXES:
        if na.startswith(pref + " ") and na[len(pref) + 1:] == nb:
            return True
        if nb.startswith(pref + " ") and nb[len(pref) + 1:] == na:
            return True
    ta, tb = na.split(), nb.split()
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not shorter:
        return False
    # Substring containment is only meaningful when BOTH tokens are at least
    # 3 chars. Without the ``len(w) >= 3`` guard, a single-letter token on the
    # longer side (e.g. "b" from a "B team" suffix) makes every name that
    # contains that letter match -- e.g. "Cadiz B" matched "Genclerbirligi"
    # ("b" in "genclerbirligi") and the wrong fixture was picked for a real
    # match. Exact-token and prefix matches above are unaffected.
    return all(
        any(
            t == w or (len(t) >= 3 and len(w) >= 3 and (t in w or w in t))
            for w in longer
        )
        for t in shorter
    )


# ---- client ---------------------------------------------------------------

class NowGoalClient:
    def __init__(
        self,
        base_url: str | None = None,
        throttle_seconds: float = 1.1,
        timeout: float = 15.0,
        proxy: str | None = None,
        mirrors: list[str] | None = None,
    ) -> None:
        # Mirror pool: the primary base URL first, then any alternates. The
        # active mirror is self._base_url; when it dies (network error, HTTP
        # error, or a parked-page body), _rotate() advances to the next one.
        primary = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/"
        pool: list[str] = [primary]
        for m in mirrors or []:
            m = str(m).strip()
            if not m:
                continue
            m = m.rstrip("/") + "/"
            if m not in pool:
                pool.append(m)
        self._mirrors = pool
        self._base_url = pool[0]
        self._active = 0
        self._throttle = throttle_seconds
        self._timeout = timeout
        # Explicit proxy (e.g. the bot's auto-detected Tor socks5). Passed to
        # httpx so http:// nowgoal requests also route through it -- the bot
        # only exports HTTPS_PROXY, which httpx applies to https URLs only.
        self._proxy = proxy
        # Schedule fetched per date window (one analyse batch resolves many
        # pairs on the same day) -- cache it so one call covers every pair.
        self._schedule_cache: dict[str, list[dict[str, Any]]] = {}
        # Circuit breaker (2026-08-23): when EVERY mirror fails at the
        # TRANSPORT level (connection refused / DNS / timeout -- the whole
        # network path is down, not a parked page), further attempts inside
        # the same run are pure latency: each _get burns len(mirrors) x
        # connect-timeout for a guaranteed None. After ``_BREAKER_THRESHOLD``
        # consecutive all-mirrors-transport failures the client short-
        # circuits to None for an escalating cooldown
        # (``_BREAKER_COOLDOWN_BASE`` doubling per consecutive open, capped at
        # ``_BREAKER_COOLDOWN_MAX``); any successful response resets both the
        # strike counter and the escalation. Output is IDENTICAL while the
        # network is genuinely down (every call already ended in None) -- only
        # the dead wait is removed. HTTP-status / non-JSON failures never arm
        # the breaker (the site is reachable).
        self._breaker_strikes = 0
        self._breaker_until = 0.0
        self._breaker_opens = 0

    def _breaker_open(self) -> bool:
        return time.monotonic() < self._breaker_until

    def _breaker_cooldown(self) -> float:
        """Escalating cooldown (Opsi 2): 90s -> 180s -> 360s cap per
        consecutive open without an intervening success."""
        return min(
            _BREAKER_COOLDOWN_BASE * (2 ** self._breaker_opens),
            _BREAKER_COOLDOWN_MAX,
        )

    def _breaker_record_transport_failure(self) -> None:
        self._breaker_strikes += 1
        if self._breaker_strikes >= _BREAKER_THRESHOLD:
            cooldown = self._breaker_cooldown()
            logger.warning(
                "nowgoal circuit breaker OPEN for %ds (consecutive opens=%d)",
                cooldown, self._breaker_opens + 1,
            )
            self._breaker_until = time.monotonic() + cooldown
            self._breaker_opens += 1
            self._breaker_strikes = 0

    def _breaker_record_success(self) -> None:
        self._breaker_strikes = 0
        self._breaker_until = 0.0
        self._breaker_opens = 0

    def _rotate(self) -> None:
        """Move to the next mirror in the pool (wrap-around)."""
        if len(self._mirrors) > 1:
            self._active = (self._active + 1) % len(self._mirrors)
            self._base_url = self._mirrors[self._active]

    def _client(self) -> httpx.AsyncClient:
        # Opsi 1: hanya fase CONNECT yang fast-fail (_CONNECT_TIMEOUT);
        # read/write/pool tetap ``_timeout`` penuh -- respons mirror yang
        # healthy tapi lambat tetap di tunggu sampai selesai.
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(
                connect=_CONNECT_TIMEOUT,
                read=self._timeout,
                write=self._timeout,
                pool=self._timeout,
            ),
        }
        if self._proxy:
            kwargs["proxy"] = self._proxy
        return httpx.AsyncClient(**kwargs)

    # ---- low-level HTTP -------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Request headers. The odds endpoint answers ``{"code":1002}``
        without a Referer (verified live), so always send one."""
        return {
            "User-Agent": USER_AGENT,
            "Referer": self._base_url,
        }

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """GET JSON, retrying across mirrors. Rotates on transport errors,
        HTTP >= 400 (parked domains answer redirects / errors for ajax
        paths), and bodies that are not JSON. Returns None if every mirror
        failed. The active mirror stays on the last one that worked."""
        if self._breaker_open():
            return None
        transport_failures = 0
        for _ in range(len(self._mirrors)):
            url = self._base_url.rstrip("/") + path
            try:
                async with self._client() as client:
                    resp = await client.get(url, params=params, headers=self._headers())
            except httpx.HTTPError as exc:
                logger.warning("nowgoal network error (%s via %s): %s", path, self._base_url, exc)
                transport_failures += 1
                self._rotate()
                continue
            if resp.status_code == 429:
                logger.warning("nowgoal 429 on %s", path)
                await asyncio.sleep(self._throttle)
                self._rotate()
                continue
            # >= 300: parked domains answer 301/302 (hugedomains) or 404 for
            # every path; the live site answers 200. Redirects are failures.
            if resp.status_code >= 300:
                logger.warning("nowgoal http %s on %s via %s", resp.status_code, path, self._base_url)
                self._rotate()
                continue
            await asyncio.sleep(self._throttle)
            try:
                data = resp.json()
            except ValueError:
                logger.warning("nowgoal non-JSON body on %s via %s", path, self._base_url)
                self._rotate()
                continue
            self._breaker_record_success()
            return data
        if transport_failures == len(self._mirrors):
            self._breaker_record_transport_failure()
        return None

    async def _get_text(self, path: str, params: dict[str, Any]) -> str | None:
        """GET raw text, retrying across mirrors (see ``_get``). Returns the
        first non-error body; callers validate content (e.g. schedule body
        shape) and rotate further when a mirror answers with a parked page."""
        if self._breaker_open():
            return None
        transport_failures = 0
        for _ in range(len(self._mirrors)):
            url = self._base_url.rstrip("/") + path
            try:
                async with self._client() as client:
                    resp = await client.get(url, params=params, headers=self._headers())
            except httpx.HTTPError as exc:
                logger.warning("nowgoal network error (%s via %s): %s", path, self._base_url, exc)
                transport_failures += 1
                self._rotate()
                continue
            if resp.status_code == 429:
                logger.warning("nowgoal 429 on %s", path)
                await asyncio.sleep(self._throttle)
                self._rotate()
                continue
            # >= 300: parked domains answer 301/302 (hugedomains) or 404 for
            # every path; the live site answers 200. Redirects are failures.
            if resp.status_code >= 300:
                logger.warning("nowgoal http %s on %s via %s", resp.status_code, path, self._base_url)
                self._rotate()
                continue
            if len(resp.content) > _MAX_BODY:
                logger.warning("nowgoal body too large on %s (%d bytes)", path, len(resp.content))
                self._rotate()
                continue
            await asyncio.sleep(self._throttle)
            self._breaker_record_success()
            return resp.text
        if transport_failures == len(self._mirrors):
            self._breaker_record_transport_failure()
        return None

    # ---- schedule --------------------------------------------------------

    async def fetch_schedule(self, date: str) -> list[dict[str, Any]] | None:
        """Matches for one WIB-style calendar date (JS source -> dicts).

        Retries across mirrors: a mirror may answer HTTP 200 with a parked
        page / redirect body instead of the schedule JS (verified live: a
        parked nowgoal3.com served hugedomains HTML for every path), so the
        body is validated and the client rotates on garbage.
        """
        cached = self._schedule_cache.get(date)
        if cached is not None:
            return cached
        # _get_text already tried every mirror on transport errors (returns
        # None only when all failed), so here we only re-try when a mirror
        # answered with a parked page / block page / garbage body.
        for _ in range(len(self._mirrors)):
            text = await self._get_text(
                "/ajax/SoccerAjax",
                {
                    "type": 6,
                    "date": date,
                    "order": "time",
                    "timezone": 0,
                    "flesh": int(time.time() * 1000),
                },
            )
            if not text:
                return None
            if _looks_like_schedule(text):
                matches = self._parse_schedule(text)
                if matches:
                    self._schedule_cache[date] = matches
                    return matches
                # Valid nowgoal response but no matches for this date (empty
                # day). Not a mirror failure -- don't rotate.
                return None
            # Parked page / block page / garbage body -> next mirror.
            self._rotate()
        return None

    @staticmethod
    def _parse_schedule(text: str) -> list[dict[str, Any]]:
        leagues: dict[str, dict[str, str]] = {}
        for m in _SCHEDULE_RE_B.finditer(text):
            parts = [_clean_token(t) for t in m.group(2).split(",")]
            if len(parts) >= 3:
                leagues[m.group(1)] = {
                    "league_id": parts[0],
                    "name": parts[2],
                }
        matches: list[dict[str, Any]] = []
        for m in _SCHEDULE_RE_A.finditer(text):
            item = _parse_match_row(m.group(2), leagues)
            if item is not None:
                matches.append(item)
        return matches

    async def probe_homepage(self) -> dict[str, Any]:
        """GET the base URL and classify the response.

        Pure diagnostic (``runner nowgoal-check``); never used by the odds
        pipeline. Detects the Indonesian ISP "Trustpositif" block page, which
        answers 200 for every nowgoal domain from blocked networks.
        """
        url = self._base_url.rstrip("/") + "/"
        try:
            async with self._client() as client:
                resp = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            return {"http": None, "error": f"{type(exc).__name__}: {exc}"}
        text = resp.text or ""
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        title = m.group(1).strip()[:80] if m else ""
        return {
            "http": resp.status_code,
            "size": len(text),
            "title": title,
            "blocked": "trustpositif" in text.lower(),
            "looks_like_site": _looks_like_site(resp.status_code, text, title),
        }

    async def _find_by_id(self, match_id: str, date: str | None = None) -> dict[str, Any] | None:
        for d in self._dates_to_scan(date):
            rows = await self.fetch_schedule(d)
            if not rows:
                continue
            for r in rows:
                if r["match_id"] == str(match_id):
                    return r
        return None

    async def find_fixture(
        self,
        home: str,
        away: str,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a fixture from the schedule via tolerant team matching.

        Scans the requested date (default: today + tomorrow UTC) so a
        slightly-off kickoff date cannot hide the fixture.

        Youth/reserve guard (verified live 2026-08-17): NowGoal schedules
        BOTH the senior team and the U19/reserve side on the same day
        ("Galatasaray vs Corum Belediyespor" AND "Galatasaray U19 vs
        Corum FK U19"). Tolerant matching happily returns whichever row
        appears first -- the U19 match, with U19 odds (median 1.42/4.5/5.0
        vs the senior 1.36/5.0/7.5). A prediction built on the youth match
        is silently wrong. Candidates are therefore scored: an EXACT
        normalized name match beats a containment/prefix hit, and a
        candidate whose name carries a youth/reserve marker is penalized
        (still returned when it is the ONLY candidate -- the caller may
        genuinely have asked for the U19 side).
        """
        nh, na = _norm_team(home), _norm_team(away)
        best: tuple[int, dict[str, Any]] | None = None  # (score, row)
        for d in self._dates_to_scan(date):
            rows = await self.fetch_schedule(d)
            if not rows:
                continue
            for r in rows:
                rh, ra = _norm_team(r.get("home") or ""), _norm_team(r.get("away") or "")
                if not (_same_team(r.get("home") or "", home) and _same_team(r.get("away") or "", away)):
                    continue
                # Score: exact both sides >> exact one side / prefix >> containment only.
                exact_both = (rh == nh and ra == na)
                exact_one = (rh == nh and _same_team(r.get("away") or "", away)) or (
                    ra == na and _same_team(r.get("home") or "", home)
                )
                score = 2 if exact_both else (1 if exact_one else 0)
                if _is_youth_reserve_name(r.get("home") or "") or _is_youth_reserve_name(r.get("away") or ""):
                    score -= 2
                if best is None or score > best[0]:
                    best = (score, r)
        return best[1] if best is not None else None

    async def find_fixture_by_score(
        self,
        home: str,
        away: str,
        date: str | None,
        home_goals: int,
        away_goals: int,
    ) -> dict[str, Any] | None:
        """Resolve a SETTLED fixture whose schedule name differs from the
        caller's (club renamed / provider spelling, verified 2026-08-16:
        the result source said "Beveren", NowGoal's schedule says
        "Red Star Waasland").

        Scans the date's schedule for a FINISHED match with the SAME FINAL
        SCORE and at least one side matching exactly (normalized); returns it
        ONLY when exactly one candidate exists -- with the final score as an
        extra discriminator a wrong fixture is effectively impossible, and
        ambiguity stays None (no fixture is safer than a wrong one).
        """
        want = f"{int(home_goals)}-{int(away_goals)}"
        nh, na = _norm_team(home), _norm_team(away)
        if not (nh and na):
            return None
        for d in self._dates_to_scan(date):
            rows = await self.fetch_schedule(d)
            if not rows:
                continue
            candidates: list[dict[str, Any]] = []
            for r in rows:
                if not (r.get("finished") and r.get("score") == want):
                    continue
                rh, ra = _norm_team(r["home"]), _norm_team(r["away"])
                # One side exact, same orientation (a swapped row would
                # label home/away prices wrongly).
                if (rh == nh or ra == na) and not (rh == na and ra == nh):
                    candidates.append(r)
            if len(candidates) == 1:
                return candidates[0]
        return None

    @staticmethod
    def _dates_to_scan(date: str | None) -> list[str]:
        if date:
            return [date]
        today = datetime.now(timezone.utc).date()
        return [today.isoformat(), (today + timedelta(days=1)).isoformat()]

    # ---- odds (type=14) --------------------------------------------------

    async def fetch_odds(
        self,
        fixture: dict[str, Any],
        closing: bool = False,
    ) -> dict[str, Any] | None:
        """Normalized The-Odds-API-shaped payload for a fixture, or None.

        ``closing=False`` (default) reads the t=1 ``mixodds`` feed and emits
        the ``l`` (last pre-match) leg -- the real closing line, which also
        persists after settle. ``closing=True`` reads the t=11
        ``roddsList`` feed: DIAGNOSTIC ONLY, never a closing line -- for
        settled matches it serves result-embedded final prices (winner
        ~1.01, losers 50-500).

        Markets emitted: h2h (from ``euro``), totals (from ``ou``) and
        asian_handicap (from ``ah``; diagnostic only -- current model
        consumers ignore it). Rows that fail strict validation are dropped;
        no bookmaker with a valid market -> None.
        """
        match_id = fixture.get("match_id") or fixture.get("id")
        if not match_id:
            return None
        data = await self._get(
            "/ajax/soccerajax",
            {
                "type": 14,
                "t": 11 if closing else 1,
                "id": match_id,
                "h": 0,
                "flesh": int(time.time() * 1000),
            },
        )
        if not isinstance(data, dict):
            return None
        if data.get("ErrCode") not in (0, None):
            logger.warning("nowgoal odds ErrCode=%r for match %s", data.get("ErrCode"), match_id)
            return None
        body = data.get("Data") or {}
        rows = body.get("roddsList" if closing else "mixodds")
        if not isinstance(rows, list):
            return None

        home_name = fixture.get("home") or ""
        away_name = fixture.get("away") or ""
        bookmakers: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            euro = parse_euro(item.get("euro"))
            ou = parse_ou(item.get("ou"))
            ah = parse_ah(item.get("ah"))
            # Opening (``f``) vs latest (``l``) prices: the raw mixodds wrapper
            # keeps both, so market movement (steam/drift) can be detected
            # per bookmaker per market. Parse the opening leg separately;
            # ``parse_*`` already unwraps to ``l`` (fallback ``f``).
            euro_open = parse_euro(_opening_leg(item.get("euro")))
            ou_open = parse_ou(_opening_leg(item.get("ou")))
            ah_open = parse_ah(_opening_leg(item.get("ah")))
            markets: list[dict[str, Any]] = []
            if euro is not None:
                markets.append({
                    "key": "h2h",
                    "outcomes": [
                        {"name": home_name or "Home", "price": euro["home"], "opening_price": (euro_open or {}).get("home")},
                        {"name": "Draw", "price": euro["draw"], "opening_price": (euro_open or {}).get("draw")},
                        {"name": away_name or "Away", "price": euro["away"], "opening_price": (euro_open or {}).get("away")},
                    ],
                })
            if ou is not None:
                # ``opening_point`` preserves the opening goal line so line
                # movement (2.25 -> 2.5) is separable from price movement.
                markets.append({
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": ou["over"], "point": ou["line"],
                         "opening_price": (ou_open or {}).get("over"),
                         "opening_point": (ou_open or {}).get("line")},
                        {"name": "Under", "price": ou["under"], "point": ou["line"],
                         "opening_price": (ou_open or {}).get("under"),
                         "opening_point": (ou_open or {}).get("line")},
                    ],
                })
            if ah is not None:
                markets.append({
                    "key": "asian_handicap",
                    "outcomes": [
                        {"name": "Home", "price": ah["home"], "point": _ah_side_line(ah["line"], "home"),
                         "opening_price": (ah_open or {}).get("home"),
                         "opening_point": _ah_side_line((ah_open or {}).get("line"), "home")},
                        {"name": "Away", "price": ah["away"], "point": _ah_side_line(ah["line"], "away"),
                         "opening_price": (ah_open or {}).get("away"),
                         "opening_point": _ah_side_line((ah_open or {}).get("line"), "away")},
                    ],
                })
            if not markets:
                continue
            cid = item.get("cid")
            title: str | None = None
            cn = item.get("cn")
            if isinstance(cn, str) and cn.strip():
                title = cn.strip()  # verified live: API serves the real name
            elif cid is not None:
                try:
                    title = KNOWN_COMPANIES.get(int(cid))
                except (TypeError, ValueError):
                    title = None
            bookmakers.append({
                "title": title or (f"NowGoal-{cid}" if cid is not None else "NowGoal"),
                "markets": markets,
            })
        if not bookmakers:
            return None
        return {
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": fixture.get("kickoff"),
            "bookmakers": bookmakers,
        }

    async def match_odds(
        self,
        home: str,
        away: str,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        """One-shot: resolve fixture from schedule, then fetch normalized odds."""
        fixture = await self.find_fixture(home, away, date)
        if not fixture:
            return None
        return await self.fetch_odds(fixture)

    async def fetch_live_odds(
        self,
        fixture: dict[str, Any],
        include_closing: bool = False,
    ) -> dict[str, Any] | None:
        """        Realtime in-play odds (the ``r`` leg -- the oddscomp "Live" column).

        Same normalized shape as ``fetch_odds`` but every market is the
        current in-play snapshot per bookmaker, present only while the match
        is live (or a bookmaker streams early prices). Returns None when no
        bookmaker carries realtime odds yet -- pre-match ``l`` is untouched
        by this method. ``is_live`` is True in the payload so callers can
        gate movement tracking on it.

        Phase 2.2: ``include_closing=True`` also captures the CLOSING price
        (the ``l`` leg, last pre-match -- it persists after settle) when the
        match is finished, so one in-play poll pass captures it without
        waiting for the settle command. The closing line is attached as
        ``closing_odds`` {home, draw, away}.
        """
        match_id = fixture.get("match_id") or fixture.get("id")
        if not match_id:
            return None
        data = await self._get(
            "/ajax/soccerajax",
            {
                "type": 14,
                "t": 1,
                "id": match_id,
                "h": 0,
                "flesh": int(time.time() * 1000),
            },
        )
        if not isinstance(data, dict) or data.get("ErrCode") not in (0, None):
            return None
        body = data.get("Data") or {}
        rows = body.get("mixodds")
        if not isinstance(rows, list):
            return None

        home_name = fixture.get("home") or ""
        away_name = fixture.get("away") or ""
        bookmakers: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            euro = parse_euro(_realtime_leg(item.get("euro")))
            ou = parse_ou(_realtime_leg(item.get("ou")))
            ah = parse_ah(_realtime_leg(item.get("ah")))
            markets: list[dict[str, Any]] = []
            if euro is not None:
                markets.append({
                    "key": "h2h",
                    "outcomes": [
                        {"name": home_name or "Home", "price": euro["home"]},
                        {"name": "Draw", "price": euro["draw"]},
                        {"name": away_name or "Away", "price": euro["away"]},
                    ],
                })
            if ou is not None:
                markets.append({
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": ou["over"], "point": ou["line"]},
                        {"name": "Under", "price": ou["under"], "point": ou["line"]},
                    ],
                })
            if ah is not None:
                markets.append({
                    "key": "asian_handicap",
                    "outcomes": [
                        {"name": "Home", "price": ah["home"], "point": _ah_side_line(ah["line"], "home")},
                        {"name": "Away", "price": ah["away"], "point": _ah_side_line(ah["line"], "away")},
                    ],
                })
            if not markets:
                continue
            cid = item.get("cid")
            cn = item.get("cn")
            title: str | None = None
            if isinstance(cn, str) and cn.strip():
                title = cn.strip()
            elif cid is not None:
                try:
                    title = KNOWN_COMPANIES.get(int(cid))
                except (TypeError, ValueError):
                    title = None
            bookmakers.append({
                "title": title or (f"NowGoal-{cid}" if cid is not None else "NowGoal"),
                "markets": markets,
            })
        if not bookmakers:
            return None
        out: dict[str, Any] = {
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": fixture.get("kickoff"),
            "is_live": True,
            "bookmakers": bookmakers,
        }
        # Phase 2.2: closing-price capture during in-play for finished
        # matches (the ``l`` leg persists post-settle; ``finished``/``score``
        # come from the schedule row). Best-effort -- a failed/no-data
        # closing fetch degrades to the live payload unchanged.
        if include_closing and (fixture.get("finished") or fixture.get("score")):
            try:
                closing = await self.fetch_closing_odds(fixture)
                if closing:
                    out["closing_odds"] = closing
            except Exception:  # noqa: BLE001
                pass
        return out

    async def fetch_closing_odds(self, fixture: dict[str, Any]) -> dict[str, float] | None:
        """Closing 1X2 prices for a match (the ``l`` leg of the mixodds feed).

        The true closing line is the ``l`` (last pre-match) leg of
        type=14&t=1 mixodds, and it PERSISTS after the match is settled --
        verified live 2026-08-16. The t=11 ``roddsList`` endpoint is NOT a
        closing line: for settled matches it serves the post-settlement
        FINAL prices (winner collapses to ~1.01, losers blow out to
        50-500 -- the result embedded in the price), which would fabricate
        CLV. So this reads the same ``l``-leg prices the pre-match odds poll
        captures, via ``fetch_odds`` (t=1), and is the settle-time companion
        callers use to attach a real closing line to a settlement.

        Returns ``{home, draw, away}`` median decimal odds across every
        bookmaker that priced the 1X2 market, or None when no bookmaker has
        valid closing prices (no data).
        """
        payload = await self.fetch_odds(fixture, closing=False)
        if not payload:
            return None
        home_name = fixture.get("home") or ""
        away_name = fixture.get("away") or ""
        prices: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
        for bm in payload.get("bookmakers") or []:
            for m in bm.get("markets") or []:
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes") or []:
                    name = str(o.get("name") or "")
                    side = None
                    if name and home_name and name == home_name:
                        side = "home"
                    elif name and away_name and name == away_name:
                        side = "away"
                    elif name.lower() == "draw":
                        side = "draw"
                    price = o.get("price")
                    if side and isinstance(price, (int, float)) and price > 1.0:
                        prices[side].append(float(price))
        closing = {
            side: round(statistics.median(vals), 4)
            for side, vals in prices.items() if vals
        }
        return closing or None

    # ---- odds movement history (opening -> latest, per bookmaker/market) ----

    @staticmethod
    def _history_row(
        market_key: str,
        selection: str,
        *,
        line: float | None,
        line_open: float | None,
        price: float | None,
        price_open: float | None,
        bookmaker: str,
    ) -> dict[str, Any] | None:
        """One normalized opening->latest movement row, or None if no data."""
        if price is None and price_open is None:
            return None
        return {
            "market": market_key,
            "selection": selection,
            "bookmaker": bookmaker,
            "opening_line": line_open,
            "opening_price": price_open,
            "latest_line": line,
            "latest_price": price,
        }

    async def fetch_odds_history(self, fixture: dict[str, Any]) -> dict[str, Any] | None:
        """Normalized opening -> latest odds movement per bookmaker per market.

        NowGoal's odds feed exposes, per bookmaker per market, the opening
        leg (``f``) and the latest pre-match leg (``l``), plus realtime (``r``)
        once in-play. There is NO intermediate timestamped series -- the full
        history NowGoal actually provides is opening -> latest (2 points),
        with both the PRICE and the LINE (Asian Handicap / Over-Under goal
        line) at each point. The closing line is the ``l`` leg itself (the
        last pre-match price), which persists once the match is settled --
        it is NOT a separate endpoint: t=11/roddsList only serves
        result-embedded final prices and must never be used as closing.

        Returns a JSON-safe structure whose ``history_resolution`` is exactly
        one of ``opening_latest`` (upcoming/live) or ``opening_closing``
        (settled, when closing odds are available). ``timestamp_available``
        is always False -- NowGoal does not timestamp these legs. Never
        fabricates intermediate snapshots.
        """
        match_id = fixture.get("match_id") or fixture.get("id")
        if not match_id:
            return None
        raw = await self._get(
            "/ajax/soccerajax",
            {
                "type": 14,
                "t": 1,
                "id": match_id,
                "h": 0,
                "flesh": int(time.time() * 1000),
            },
        )
        if not isinstance(raw, dict):
            return None
        body = raw.get("Data") or {}
        rows = body.get("mixodds")
        if not isinstance(rows, list) or not rows:
            return None

        home_name = fixture.get("home") or ""
        away_name = fixture.get("away") or ""
        markets: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            cid = item.get("cid")
            title = item.get("cn") or (
                KNOWN_COMPANIES.get(int(cid)) if cid is not None else None
            ) or f"NowGoal-{cid}"

            euro = parse_euro(item.get("euro"))
            euro_open = parse_euro(_opening_leg(item.get("euro")))
            if euro is not None:
                for side in ("home", "draw", "away"):
                    label = home_name if side == "home" else away_name if side == "away" else "Draw"
                    row = self._history_row(
                        "h2h", label,
                        line=None, line_open=None,
                        price=(euro or {}).get(side),
                        price_open=(euro_open or {}).get(side),
                        bookmaker=title,
                    )
                    if row:
                        markets.append(row)

            ou = parse_ou(item.get("ou"))
            ou_open = parse_ou(_opening_leg(item.get("ou")))
            if ou is not None:
                for sel in ("over", "under"):
                    row = self._history_row(
                        "totals", sel.title(),
                        line=(ou or {}).get("line"),
                        line_open=(ou_open or {}).get("line"),
                        price=(ou or {}).get(sel),
                        price_open=(ou_open or {}).get(sel),
                        bookmaker=title,
                    )
                    if row:
                        markets.append(row)

            ah = parse_ah(item.get("ah"))
            ah_open = parse_ah(_opening_leg(item.get("ah")))
            if ah is not None:
                for side in ("home", "away"):
                    row = self._history_row(
                        "asian_handicap", side.title(),
                        line=_ah_side_line((ah or {}).get("line"), side),
                        line_open=_ah_side_line((ah_open or {}).get("line"), side),
                        price=(ah or {}).get(side),
                        price_open=(ah_open or {}).get(side),
                        bookmaker=title,
                    )
                    if row:
                        markets.append(row)

            # Realtime in-play snapshot (the ``r`` leg -- the "Live" column
            # on the oddscomp page). Present only once the match is in play
            # or when a bookmaker streams prices early; flagged so consumers
            # never mistake it for the pre-match ``l`` leg.
            euro_live = parse_euro(_realtime_leg(item.get("euro")))
            ou_live = parse_ou(_realtime_leg(item.get("ou")))
            ah_live = parse_ah(_realtime_leg(item.get("ah")))
            for market_key, live_odds, sides in (
                ("h2h", euro_live, ("home", "draw", "away")),
                ("totals", ou_live, ("over", "under")),
                ("asian_handicap", ah_live, ("home", "away")),
            ):
                if live_odds is None:
                    continue
                for side in sides:
                    row = self._history_row(
                        market_key, side.title(),
                        line=(
                            _ah_side_line((live_odds or {}).get("line"), side)
                            if market_key == "asian_handicap"
                            else (live_odds or {}).get("line")
                        ),
                        line_open=None,
                        price=(live_odds or {}).get(side),
                        price_open=None,
                        bookmaker=title,
                    )
                    if row:
                        row["snapshot"] = "live"
                        markets.append(row)

        if not markets:
            return None
        return {
            "match_id": str(match_id),
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": fixture.get("kickoff"),
            "timestamp_available": False,
            "history_resolution": "opening_latest",
            "has_live": any((m or {}).get("snapshot") == "live" for m in markets),
            "source": "nowgoal",
            "markets": markets,
        }

    # ---- full timestamped odds-movement series (type=14&t=20) ------------

    async def fetch_odds_trend(
        self,
        fixture: dict[str, Any],
        cids: list[int] | None = None,
    ) -> dict[str, Any] | None:
        """Full timestamped odds-movement series per bookmaker per market.

        This is the data behind the oddscomp page's "Trends" popup
        (``_oddsDetailWin.open(sid, cid, ...)`` -> ``type=14&t=20&cid=``,
        verified live 2026-08-16): ONE call returns EVERY recorded odds
        change for a match -- not just the opening->latest pair that
        ``fetch_odds_history`` exposes. Each row carries a unix timestamp
        (``mt``, seconds), the match minute (``ht``; empty string = the row
        is PRE-MATCH), the live score at that moment (``hs``/``gs``), a
        ``close`` flag, and the odds/line tuple (``u``/``g``/``d``).

        Market encoding inside each row (verified live):
          - ``op`` (1X2): u = home, g = draw, d = away -- plain DECIMAL odds
          - ``ah`` (Asian Handicap): u = home price, g = handicap line,
            d = away price -- Hong-Kong odds (decimal = price + 1)
          - ``ou`` (Over/Under): u = over price, g = goal line, d = under
            price -- Hong-Kong odds

        ``cids`` defaults to the site's major bookmakers (Bet365, Sbobet,
        1xBet, ...); each is fetched independently and included only when it
        returns at least one parsable row, so a bookmaker without odds never
        pollutes the series. Returns a JSON-safe structure:

          {"match_id", "home_team", "away_team", "commence_time",
           "source": "nowgoal", "timestamp_available": True,
           "history_resolution": "timestamped_series",
           "bookmakers": [{cid, name, "h2h": [rows], "ah": [rows],
                            "ou": [rows]}]}

        where each row is {"ts": ISO8601, "minute": str|'', "home_goals":
        int, "away_goals": int, "close": bool, "home": float|None,
        "draw": float|None, "away": float|None, "line": float|None,
        "over": float|None, "under": float|None} (the relevant price
        fields filled per market). Rows are chronological (oldest first).
        Never fabricates intermediate points -- it only reports what the
        provider recorded.
        """
        match_id = fixture.get("match_id") or fixture.get("id")
        if not match_id:
            return None
        companies = cids or [8, 31, 50, 3, 17, 24]  # Bet365, Sbobet, 1xBet, Crown, Mansion88, 12bet
        bookmakers: list[dict[str, Any]] = []
        for cid in companies:
            raw = await self._get(
                "/ajax/soccerajax",
                {"type": 14, "t": 20, "id": match_id, "cid": cid, "h": 0},
            )
            if not isinstance(raw, dict) or raw.get("ErrCode") not in (0, None):
                continue
            body = raw.get("Data") or {}
            h2h = self._parse_trend_series(body.get("op"), kind="h2h")
            ah = self._parse_trend_series(body.get("ah"), kind="ah")
            ou = self._parse_trend_series(body.get("ou"), kind="ou")
            if not (h2h or ah or ou):
                continue
            bookmakers.append({
                "cid": cid,
                "name": KNOWN_COMPANIES.get(cid, f"NowGoal-{cid}"),
                "h2h": h2h,
                "ah": ah,
                "ou": ou,
            })
        if not bookmakers:
            return None
        return {
            "match_id": str(match_id),
            "home_team": fixture.get("home"),
            "away_team": fixture.get("away"),
            "commence_time": fixture.get("kickoff"),
            "source": "nowgoal",
            "timestamp_available": True,
            "history_resolution": "timestamped_series",
            "bookmakers": bookmakers,
        }

    def _parse_trend_series(
        self,
        rows: Any,
        kind: str,
    ) -> list[dict[str, Any]]:
        """Normalize one type=14&t=20 market array into chronological rows.

        ``kind`` is ``h2h`` (op), ``ah`` or ``ou``. Every row keeps the raw
        ``mt`` unix timestamp (converted to ISO), the match minute (empty =
        pre-match), the score and close flag, plus the decoded prices:
        h2h -> home/draw/away (decimal); ah -> home/away + handicap line
        (HK->decimal); ou -> over/under + goal line (HK->decimal). Rows that
        fail validation (no timestamp, no parsable price) are dropped.
        """
        if not isinstance(rows, list):
            return []
        parsed: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            mt = r.get("mt")
            try:
                ts = datetime.fromtimestamp(float(mt), tz=timezone.utc).isoformat()
            except (TypeError, ValueError, OverflowError):
                continue
            minute = str(r.get("ht") or "")
            row: dict[str, Any] = {
                "ts": ts,
                "minute": minute,
                "home_goals": int(r.get("hs") or 0),
                "away_goals": int(r.get("gs") or 0),
                "close": bool(r.get("close")),
                "home": None, "draw": None, "away": None,
                "line": None, "over": None, "under": None,
            }
            odds = r.get("odds") or {}
            if kind == "h2h":
                row["home"] = _coerce_odds(odds.get("u"))
                row["draw"] = _coerce_odds(odds.get("g"))
                row["away"] = _coerce_odds(odds.get("d"))
            elif kind == "ah":
                row["home"] = _coerce_odds_hk(odds.get("u"))
                row["away"] = _coerce_odds_hk(odds.get("d"))
                raw_line = _coerce_line(odds.get("g"))
                # NowGoal quotes the AH line from the AWAY side (verified
                # live: real market Home +1.25 is served as g=-1.25); the
                # signal engine / odds_snapshot rows use the HOME-handicap
                # convention, so flip the sign to stay comparable.
                row["line"] = -raw_line if raw_line is not None else None
            else:
                row["over"] = _coerce_odds_hk(odds.get("u"))
                row["under"] = _coerce_odds_hk(odds.get("d"))
                row["line"] = _coerce_line(odds.get("g"))
            if not any(row[k] is not None for k in ("home", "draw", "away", "over", "under")):
                continue
            parsed.append(row)
        parsed.sort(key=lambda r: r["ts"])
        return parsed

    # ---- analysis page (form + H2H fallback) ------------------------------

    async def fetch_analysis(
        self,
        home: str,
        away: str,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        """Form + H2H for a fixture from the analysis page (fallback source).

        The page /analysis/{match_id}.htm server-renders three match tables:
        table_v1 (home team's recent matches), table_v2 (away team's recent
        matches) and table_v3 (H2H between the two). Each row carries the
        result from the HOME team's perspective (span class o-win/o-draw/
        o-lose), the full-time score, and a ``vs`` flag (1 = the table's team
        played at home). Used ONLY when flashscore cannot supply form/H2H;
        the shape mirrors the flashscore/understat form dicts so callers can
        merge it transparently.

        Returns {
          "home_form": {sequence, gf_avg, ga_avg, sample_size, source, match_list},
          "away_form": {sequence, gf_avg, ga_avg, sample_size, source, match_list},
          "h2h": {wins, draws, losses, matches, source, match_list},
          "standings": {home, away},      # FT + HT blocks (context)
          "fixtures": {home, away},       # next 3 matches per team (context)
          "injuries": {home, away},       # injury/suspension lists (context)
        } or None. The existing form/h2h keys are unchanged; the new keys
        are context-only extras riding the same page fetch.
        """
        fixture = await self.find_fixture(home, away, date)
        if not fixture:
            return None
        match_id = fixture.get("match_id")
        if not match_id:
            return None
        for _ in range(len(self._mirrors)):
            # The site 301-redirects /analysis/{id}.htm -> /analysis/{id}
            # (verified live); httpx does not follow redirects by default and
            # a 3xx would count as a mirror failure, so request the canonical
            # path directly.
            text = await self._get_text(f"/analysis/{match_id}", {})
            if not text:
                return None
            if "Head to Head Statistics" in text and "Previous Scores Statistics" in text:
                out = self._parse_analysis(text, home, away)
                if out:
                    return out
            # Parked page / block page / garbage body -> next mirror.
            self._rotate()
        return None

    @staticmethod
    def _parse_analysis(
        text: str,
        home: str,
        away: str,
        limit: int = 5,
        limit_h2h: int = 10,
    ) -> dict[str, Any] | None:
        """Parse the three server-rendered match tables of the analysis page.

        Row anatomy (verified live):
          <tr id="tr1_1" vs="1" name="<league_id>" index="<match_id>"
              info="fh,fa,home_team_id,hh,ha">
            <td>league</td>
            <td><span data-t='YYYY-MM-DD HH:MM:SS'>...</span></td>
            <td><a onclick=team(home_id)>home</a></td>
            <td><span class="fscore_1">3-2</span><span class="hscore_1">(1-1)</span></td>
            <td><a onclick=team(away_id)>away</a></td>
            <td><span class="fcorner_1">11-3</span>...</td>
            ... <td class="hbg-td1"><span class="o-lose">L</span></td>
        ``info`` is fh,fa (full-time home/away goals), home_team_id, hh,ha
        (half-time). ``vs``=1 means the TABLE's team played at home; the
        W/D/L span is ALWAYS from that table's team perspective.

        Beyond the W/D/L + goal averages the model consumes, every row also
        carries its date, match id, HT score and corners, and the page's
        standings / fixture / injury sections are parsed -- all returned as
        NEW keys so the existing form/h2h contract is unchanged.
        """
        out: dict[str, Any] = {}
        # Which team each table belongs to: table_v1 = ``home``, table_v2 =
        # ``away``, table_v3 = ``home`` (H2H is always from the home side).
        table_team = {"1": home, "2": away, "3": home}
        for table_id, key in (("1", "home_form"), ("2", "away_form"), ("3", "h2h")):
            seq: list[str] = []
            gf_list: list[float] = []
            ga_list: list[float] = []
            wins = draws = losses = 0
            rows = 0
            match_rows: list[dict[str, Any]] = []
            team_ref = table_team[table_id]
            # The page renders newest-first; take the last ``limit`` matches
            # so the rolling window matches the validated FORM_WINDOW=5 used
            # by the backtest and the flashscore form path (train/serve parity).
            # H2H is NOT a rolling-window model feature, so it reads the full
            # 10-game table the page renders instead of the 5-match cap.
            row_limit = limit_h2h if table_id == "3" else limit
            matches = list(
                re.finditer(
                    rf'<tr id="tr{table_id}_\d+"([^>]*?)>(.*?)</tr>',
                    text,
                    re.S,
                )
            )
            for m in matches[:row_limit]:
                attrs, body = m.group(1), m.group(2)
                info_m = re.search(r'info="([^"]*)"', attrs)
                # class may be quoted ("o-draw") or bare (class=o-lose) -- the
                # live site uses both forms across tables.
                res_m = re.search(r'class=["\']?o-(win|draw|lose)["\']?>\s*([WDL])\s*<', body)
                if not info_m or not res_m:
                    continue
                parts = [p.strip() for p in info_m.group(1).split(",")]
                if len(parts) < 5:
                    continue
                try:
                    fh, fa = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                # P3-2: the first <td> is the row's league name -- a friendly /
                # pre-season row must be dropped BEFORE it contributes to the
                # sequence / gf / ga / W-D-L aggregates (a 6-2 preseason
                # result is not competitive evidence).
                league_cell = re.search(r"<td[^>]*>(.*?)</td>", body, re.S)
                league_name = (
                    re.sub(r"<[^>]+>", "", league_cell.group(1)).strip()
                    if league_cell else ""
                )
                if _is_excluded_competition(league_name):
                    continue
                # Resolve the row's home/away sides by the rendered team NAMES
                # (the ``vs`` flag is NOT a reliable home/away indicator across
                # tables -- verified live). The row renders the home team in
                # the first team cell and the away team in the second.
                names = re.findall(
                    r"soccerDbPage\.team\(\d+\)\"><span[^>]*>([^<]+)</span>",
                    body,
                )
                row_home = names[0].strip() if len(names) >= 1 else ""
                row_away = names[1].strip() if len(names) >= 2 else ""
                is_home = bool(
                    row_home and _same_team(team_ref, row_home)
                ) or (
                    bool(row_away) and not _same_team(team_ref, row_away)
                )
                gf = fh if is_home else fa
                ga = fa if is_home else fh
                gf_list.append(gf)
                ga_list.append(ga)
                seq.append(res_m.group(2))
                if res_m.group(1) == "win":
                    wins += 1
                elif res_m.group(1) == "draw":
                    draws += 1
                else:
                    losses += 1
                rows += 1
                # Richer per-row capture (context only -- never a model input).
                date_m = re.search(r"data-t='([^']+)'", body)
                idx_m = re.search(r'index="(\d+)"', attrs)
                league_m = re.search(r'name="(\d+)"', attrs)
                score_m = re.search(r'class="fscore_\d[^"]*"[^>]*>([\d-]+)', body)
                ht_m = re.search(r'class="hscore_\d[^"]*"[^>]*>\(([\d-]+)\)', body)
                corner_m = re.search(r'class="fcorner_\d[^"]*"[^>]*>([\d-]+)', body)
                match_rows.append({
                    "date": date_m.group(1) if date_m else None,
                    "match_id": idx_m.group(1) if idx_m else None,
                    "league_id": league_m.group(1) if league_m else None,
                    "competition": league_name or None,
                    "home": row_home or None,
                    "away": row_away or None,
                    "score": score_m.group(1) if score_m else None,
                    "ht_score": ht_m.group(1) if ht_m else None,
                    "corners": corner_m.group(1) if corner_m else None,
                    "result": res_m.group(2),
                    "team_perspective": "home" if is_home else "away",
                })
            if not seq:
                continue
            if key == "h2h":
                out["h2h"] = {
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "matches": rows,
                    "source": "nowgoal_analysis",
                    "match_list": match_rows,
                    # P3-2: provenance -- friendlies were dropped from the
                    # aggregates above; the consumer can surface this.
                    "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
                }
            else:
                out[key] = {
                    "sequence": "-".join(seq),
                    "gf_avg": round(sum(gf_list) / len(gf_list), 3),
                    "ga_avg": round(sum(ga_list) / len(ga_list), 3),
                    "sample_size": len(seq),
                    "source": "nowgoal_analysis",
                    # F1: the page renders NEWEST-FIRST; reverse to the
                    # OLDEST -> NEWEST contract every other provider uses
                    # (flashscore/thesportsdb/football-data), so the Poisson
                    # time-decay and the signal-engine statistical component
                    # consume the same ordering live as in backtest/validate.
                    "recent_goals": list(reversed(list(zip(gf_list, ga_list)))) or None,
                    "match_list": match_rows,
                    # P3-2: provenance -- friendlies were dropped from the
                    # aggregates above; the consumer can surface this.
                    "excluded_competitions": sorted(EXCLUDED_COMPETITIONS),
                }
        # Standings / fixtures / injuries are extras on the SAME page; they
        # never block the form+H2H result when a section is absent.
        standings = NowGoalClient._parse_standings(text)
        if standings:
            out["standings"] = standings
        fixtures = NowGoalClient._parse_fixtures(text)
        if fixtures:
            out["fixtures"] = fixtures
        injuries = NowGoalClient._parse_injuries(text)
        if injuries:
            out["injuries"] = injuries
        return out or None

    # ---- standings / fixtures / injuries (analysis-page sections) --------

    @staticmethod
    def _parse_standings(text: str) -> dict[str, Any] | None:
        """League standings for BOTH teams from the analysis page.

        Two ``team-table-home`` / ``team-table-guest`` tables each render a
        FT block and a HT block, with rows Total / Home / Away / Last 6 and
        columns Matches, Win, Draw, Lose, Scored, Conceded, Pts, Rank, Rate.
        The title cell carries the league + current rank::

          [HOL D1-12] FC Utrecht

        Returns {"home": {"league":..., "rank":..., "ft": {...}, "ht": {...}},
                 "away": {...}} or None.
        """
        out: dict[str, Any] = {}
        for side in ("home", "guest"):
            table_m = re.search(
                rf'<table[^>]*class=["\']team-table-{side}["\'][^>]*>(.*?)</table>',
                text,
                re.S,
            )
            if not table_m:
                continue
            title_m = re.search(
                r"\[([^\]]+)\]\s*(?:&nbsp;)*\s*(?:</a>)?([^<]*?)\s*(?:<|$)",
                table_m.group(1),
            )
            if not title_m:
                continue
            lg = title_m.group(1)
            team_name = re.sub(r"\s+", " ", title_m.group(2)).strip()
            # "[HOL D1-12]" -> league "HOL D1", current rank 12.
            rank: int | None = None
            league = lg
            rank_m = re.search(r"^(.*)-(\d+)$", lg.strip())
            if rank_m:
                league, rank = rank_m.group(1).strip(), int(rank_m.group(2))
            blocks: dict[str, Any] = {}
            # Each block is a <tr><th>FT</th>...</tr> header followed by 4
            # data rows (Total/Home/Away/Last 6). Split the table on the
            # section header rows.
            sections = re.split(r'<tr[^>]*>\s*<th[^>]*>(FT|HT)</th>', table_m.group(1))
            # sections: [pre, "FT", ft_body, "HT", ht_body]
            for i in range(1, len(sections) - 1, 2):
                label, body = sections[i], sections[i + 1]
                block: dict[str, Any] = {}
                for row_m in re.finditer(r"<tr[^>]*>(.*?)</tr>", body, re.S):
                    cells = [c.strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row_m.group(1), re.S)]
                    if len(cells) < 10:
                        continue
                    label_row = re.sub(r"<[^>]+>", "", cells[0]).strip()

                    def _cell(v: str) -> Any:
                        txt = re.sub(r"<[^>]+>", "", v).strip()
                        if txt.endswith("%"):
                            return txt
                        try:
                            return int(txt)
                        except ValueError:
                            return txt if txt else None

                    vals = [_cell(c) for c in cells[1:10]]
                    block[label_row.lower().replace(" ", "_")] = {
                        "matches": vals[0], "win": vals[1], "draw": vals[2],
                        "lose": vals[3], "scored": vals[4], "conceded": vals[5],
                        "pts": vals[6], "rank": vals[7], "rate": vals[8],
                    }
                if block:
                    blocks[label.lower()] = block
            if not blocks:
                continue
            out[side] = {
                "league": league,
                "rank": rank,
                "team": team_name,
                **blocks,
            }
        if not out:
            return None
        if "home" in out and "guest" in out:
            out["away"] = out.pop("guest")
        elif "guest" in out:
            out["home"] = out.pop("guest")
        return out or None

    @staticmethod
    def _parse_fixtures(text: str) -> dict[str, Any] | None:
        """Upcoming 3 fixtures per team ("Fixture (3 Matches)" section).

        Rows render League/Cup, Date (data-t), Type (Home/Away), VS and a
        countdown. Returns {"home": [...], "away": [...]} or None.
        """
        out: dict[str, Any] = {}
        i = text.find("Fixture (3 Matches)")
        if i < 0:
            return None
        seg = text[i:]
        for side in ("home", "guest"):
            rows: list[dict[str, Any]] = []
            for row_m in re.finditer(
                rf'<table[^>]*class="team-table-{side}[^"]*"[^>]*>(.*?)</table>',
                seg,
                re.S,
            ):
                for tr_m in re.finditer(r"<tr[^>]*>(.*?)</tr>", row_m.group(1), re.S):
                    cells = [c.strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr_m.group(1), re.S)]
                    if len(cells) < 5:
                        continue
                    date_m = re.search(r"data-t='([^']+)'", cells[1])
                    rows.append({
                        "league": re.sub(r"<[^>]+>", "", cells[0]).strip(),
                        "date": date_m.group(1) if date_m else None,
                        "type": re.sub(r"<[^>]+>", "", cells[2]).strip(),
                        "opponent": re.sub(r"<[^>]+>", "", cells[3]).strip(),
                    })
            if rows:
                out[side] = rows
        if out and "home" in out and "guest" in out:
            out["away"] = out.pop("guest")
        return out or None

    @staticmethod
    def _parse_injuries(text: str) -> dict[str, Any] | None:
        """Injury/suspension lists ("Injury and Suspension" section).

        Player rows render position (b), number (span) and name (a)::

          <div playerid="162512" class="player-row"><b>CM</b><span>7</span><a>Victor Jensen</a></div>

        Returns {"home": [{position, number, name}...], "away": [...]} or None.
        """
        i = text.find("Injury and Suspension")
        if i < 0:
            return None
        seg = text[i:]
        out: dict[str, Any] = {}
        # Slice each side's list STRICTLY between adjacent structural markers.
        # Verified failure 2026-08-23 (Club Brugge v Cercle Brugge): when the
        # home marker is absent the old guest slice ``seg[g_start:]`` ran to
        # the end of the page and swallowed BOTH squads' player rows (~44
        # names) into the away list -- the cross-provider merge then logged
        # every starter as "missing". Each side's segment therefore ends at
        # the NEXT known marker (the other injury id, or a section terminator
        # that always follows the injury area: the standings tables /
        # fixtures header parsed by _parse_standings/_parse_fixtures).
        h_start = seg.find('id="injuryH"')
        g_start = seg.find('id="injuryG"')
        if h_start < 0 and g_start < 0:
            return None

        def _terminator(after: int) -> int | None:
            candidates = []
            for needle in ('team-table-home', 'team-table-guest', "Fixture (3 Matches)"):
                pos = seg.find(needle, after)
                if pos >= 0:
                    candidates.append(pos)
            return min(candidates) if candidates else None

        segments: dict[str, str | None] = {}
        for side, start in (("home", h_start), ("guest", g_start)):
            if start < 0:
                segments[side] = None
                continue
            others = [
                p for p in (h_start, g_start)
                if p >= 0 and p > start
            ]
            term = _terminator(start + 1)
            bounds = [p for p in others if term is None or p <= term]
            ends = bounds + ([term] if term is not None else [])
            # No sibling marker and no known section terminator -> fail open
            # to the page tail rather than crash (some pages carry neither
            # standings nor fixtures after the injury block).
            end = min(ends) if ends else len(seg)
            segments[side] = seg[start:end] if end > start else seg[start:start]

        for side, block in segments.items():
            if not block:
                continue
            players: list[dict[str, Any]] = []
            for row_m in re.finditer(
                r'<div playerid="(\d+)" class="player-row">\s*<b>\s*([^<]+?)\s*</b>\s*<span[^>]*>\s*(?:&nbsp;)?([^<]*)\s*</span>\s*<a>\s*([^<]+?)\s*</a>',
                block,
            ):
                players.append({
                    "player_id": row_m.group(1),
                    "position": row_m.group(2).strip(),
                    "number": row_m.group(3).strip() or None,
                    "name": row_m.group(4).strip(),
                })
            if players:
                out[side] = players
        # guest -> away unconditionally: an away-ONLY section (home marker
        # absent) must still surface under "away", never stay "guest" where
        # every consumer reads ["away"] and silently sees nothing.
        if "guest" in out:
            out["away"] = out.pop("guest")
        return out or None

    # ---- match detail page (team stats + HT/FT + lineups) -----------------

    async def fetch_match_detail(
        self,
        home: str,
        away: str,
        date: str | None = None,
    ) -> dict[str, Any] | None:
        """Context bundle from the match detail page (/match/live-{id}).

        Server-rendered: team statistics (recent 3 + recent 10 matches:
        goals, conceded, opponent shots, corners, yellow cards, fouls,
        possession), HT/FT statistics (last 2 seasons), goal-timing
        distributions (last 30/50 matches) and lineups with formations.
        All CONTEXT only -- nothing here is a model feature (no historical
        parity for these pages). Falls back to None quietly on any failure.
        """
        fixture = await self.find_fixture(home, away, date)
        if not fixture:
            return None
        match_id = fixture.get("match_id")
        if not match_id:
            return None
        for _ in range(len(self._mirrors)):
            text = await self._get_text(f"/match/live-{match_id}", {})
            if not text:
                return None
            if "Team Statistics" in text and "HT/FT Statistics" in text:
                out = self._parse_detail(text, fixture)
                if out:
                    return out
            self._rotate()
        return None

    async def fetch_match_xg(self, match_id: str) -> dict[str, Any] | None:
        """Full-time xG for one FINISHED match (the detail page's ftstat block).

        Pure HTTP via ``_get_text`` -- NO ``find_fixture`` re-resolution (the
        match is historical, outside the schedule window). Returns
        {"xg_home": float, "xg_away": float} or None when the match page has
        no xG block (friendly / no Technical Statistics). Never raises.
        """
        if not match_id:
            return None
        text = await self._get_text(f"/match/live-{match_id}", {})
        if not text:
            return None
        return NowGoalClient._parse_match_xg(text)

    @staticmethod
    def _parse_detail(text: str, fixture: dict[str, Any]) -> dict[str, Any] | None:
        """Parse the server-rendered sections of the match detail page."""
        out: dict[str, Any] = {
            "match_id": fixture.get("match_id"),
            "home_team": fixture.get("home"),
            "away_team": fixture.get("away"),
            "kickoff": fixture.get("kickoff"),
            "source": "nowgoal_detail",
        }
        team_stats = NowGoalClient._parse_team_stats(text)
        if team_stats:
            out["team_stats"] = team_stats
        htft = NowGoalClient._parse_htft(text)
        if htft:
            out["htft"] = htft
        goal_timing = NowGoalClient._parse_goal_timing(text)
        if goal_timing:
            out["goal_timing"] = goal_timing
        lineups = NowGoalClient._parse_lineups(text)
        if lineups:
            out["lineups"] = lineups
        if len(out) == 6:
            return None
        return out

    @staticmethod
    def _parse_team_stats(text: str) -> dict[str, Any] | None:
        """Team statistics table: rows Goal/Loss/Opponent Shots/Corners/
        Yellow Cards/Fouls/Possession x (recent 3, recent 10) per team.

        Columns per row: [home_r3] [label] [away_r3] [home_r10] [label]
        [away_r10]. Possession carries a "%" suffix (stripped to float).
        """
        i = text.find("Recent 3 Matches")
        if i < 0:
            return None
        start = text.rfind("<table", 0, i)
        end = text.find("</table>", i)
        if start < 0 or end < 0:
            return None
        table = text[start:end + 8]
        out: dict[str, Any] = {}
        for row_m in re.finditer(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [c.strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_m.group(1), re.S)]
            if len(cells) != 6:
                continue
            label = re.sub(r"<[^>]+>", "", cells[1]).strip()
            if not label:
                continue
            vals: list[float] = []
            ok = True
            for c in (cells[0], cells[2], cells[3], cells[5]):
                txt = re.sub(r"<[^>]+>", "", c).strip().rstrip("%")
                try:
                    vals.append(float(txt))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            out[label] = {
                "home_recent3": vals[0],
                "away_recent3": vals[1],
                "home_recent10": vals[2],
                "away_recent10": vals[3],
            }
        return out or None

    @staticmethod
    def _parse_match_xg(text: str) -> dict[str, Any] | None:
        """Full-time xG (home, away) from the match detail page's ``ftstat`` block.

        The Technical Statistics section renders one ``<li>`` per metric; the
        xG row carries the home value in the ``stat-c`` span BEFORE the
        ``homes`` bar wrapper and the away value AFTER the ``aways`` wrapper
        (verified live 2026-08-17: Malaysia 1.00/1.07, Sao Paulo 1.12/0.41,
        Everton CD 0.23/2.01). Only the full-time block (``id="ftstat"``) is
        read; half-time (hf1stat/hf2stat) and extra-time (otstat) blocks are
        ignored.

        Returns {"xg_home": float, "xg_away": float} or None when the page
        has no xG block (friendlies render no Technical Statistics, or the
        section is absent on this mirror) or the values are invalid.
        """
        i = text.find('id="ftstat"')
        if i < 0:
            return None
        j = text.find("<ul", i + 4)
        block = text[i:j] if j > 0 else text[i:i + 20000]
        for li in re.finditer(r"<li>(.*?)</li>", block, re.S):
            body = li.group(1)
            if "Expected Goals (xG)" not in body:
                continue
            homes = re.search(
                r'<span class="stat-c">([\d.]+)</span>\s*<span class="stat-bar-wrapper homes">',
                body,
            )
            aways = re.search(
                r'<span class="stat-bar-wrapper aways">.*?<span class="stat-c">([\d.]+)</span>',
                body,
                re.S,
            )
            if homes and aways:
                try:
                    xg_home = float(homes.group(1))
                    xg_away = float(aways.group(1))
                except ValueError:
                    return None
                if xg_home < 0 or xg_away < 0:
                    return None
                return {"xg_home": xg_home, "xg_away": xg_away}
        return None

    @staticmethod
    def _parse_htft(text: str) -> dict[str, Any] | None:
        """HT/FT statistics (last 2 seasons): 9 combos x home/away per team.

        Header row names both teams with match counts, e.g.
        "FC Utrecht ( 37 Matches)"; each data row renders 4 counts
        (home-team home, home-team away, away-team home, away-team away).
        """
        i = text.find("HT/FT Statistics")
        if i < 0:
            return None
        start = text.find("<table", i)
        end = text.find("</table>", i)
        if start < 0 or end < 0:
            return None
        table = text[start:end + 8]
        team_m = re.findall(r"<th[^>]*>([^<(]+?)\s*\(\s*(\d+)\s*Matches\)", table)
        if len(team_m) < 2:
            return None
        teams = [
            {"name": re.sub(r"\s+", " ", t[0]).strip(), "matches": int(t[1])}
            for t in team_m[:2]
        ]
        rows: dict[str, Any] = {}
        for row_m in re.finditer(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row_m.group(1), re.S)]
            if len(cells) != 5:
                continue
            combo = cells[0].replace(" ", "")
            try:
                vals = [int(c) for c in cells[1:5]]
            except ValueError:
                continue
            rows[combo] = {
                "home": {"home": vals[0], "away": vals[1]},
                "away": {"home": vals[2], "away": vals[3]},
            }
        if not rows:
            return None
        return {
            "home_team": teams[0]["name"],
            "home_matches": teams[0]["matches"],
            "away_team": teams[1]["name"],
            "away_matches": teams[1]["matches"],
            "rows": rows,
        }

    @staticmethod
    def _parse_goal_timing(text: str) -> dict[str, Any] | None:
        """Goal-timing distributions ("The Rate of Scored / Conceded").

        Two blocks per team: rateOfScored1 (last 30 matches) and
        rateOfScored2 (last 50). Each has six minute buckets (1~15 .. 76~90)
        with four counts: hScoredLi1 (home scored), hScoredLi2 (home
        conceded), gScoredLi1 (away scored), gScoredLi2 (away conceded).
        """
        out: dict[str, Any] = {}
        for window, div in (("last30", "rateOfScored1"), ("last50", "rateOfScored2")):
            i = text.find(f'id="{div}"')
            if i < 0:
                continue
            j = text.find('id="rateOfScored', i + 4)
            block = text[i:j] if j > 0 else text[i:i + 8000]
            comps = list(re.finditer(r'<div class="fx-comparision ([^"]*)"[^>]*>(.*?)</div>', block, re.S))
            buckets: list[dict[str, Any]] = []
            for k in range(0, len(comps) - 1, 2):
                scored, missed = comps[k], comps[k + 1]
                lb_m = re.search(r'<span class="fx-c-3[^"]*">\s*<span>([^<]+)</span>', scored.group(2))
                if not lb_m:
                    continue
                s_counts = re.findall(r"class='fx-c2[^']*'>([^<]+)<", scored.group(2))
                m_counts = re.findall(r"class='fx-c2[^']*'>([^<]+)<", missed.group(2))
                if len(s_counts) < 2 or len(m_counts) < 2:
                    continue
                try:
                    buckets.append({
                        "minutes": lb_m.group(1).strip(),
                        "home_scored": int(s_counts[0]),
                        "away_scored": int(s_counts[1]),
                        "home_conceded": int(m_counts[0]),
                        "away_conceded": int(m_counts[1]),
                    })
                except ValueError:
                    continue
            if buckets:
                out[window] = buckets
        return out or None

    @staticmethod
    def _parse_lineups(text: str) -> dict[str, Any] | None:
        """Starting XIs + benches with formations (match detail page).

        ``#lineupBox`` renders four play containers in order: ``home five``
        (home XI), ``guest five`` (away XI), ``home`` (home bench), ``guest``
        (away bench). Each player is ``.play[techWinId]`` with number + name.
        """
        i = text.find('id="lineupBox"')
        if i < 0:
            return None
        j = text.find('id="matchBox"', i)
        if j < 0:
            return None
        seg = text[i:j]
        team_names = {"home": None, "away": None}
        formations = {"home": None, "away": None}
        name_m = re.search(
            r'class="tn-home[^"]*"[^>]*>.*?>([^<]+)</a>\s*<span>([^<]+)</span>',
            seg,
            re.S,
        )
        if name_m:
            team_names["home"] = name_m.group(1).strip()
            formations["home"] = name_m.group(2).strip()
        name_m = re.search(
            r'class="tn-away[^"]*"[^>]*>.*?<span>([^<]+)</span>\s*<a[^>]*>([^<]+)</a>',
            seg,
            re.S,
        )
        if name_m:
            formations["away"] = name_m.group(1).strip()
            team_names["away"] = name_m.group(2).strip()

        box = text[j:]
        bounds = [
            ("home", "starters", 'class="home five"', 'class="guest five"'),
            ("away", "starters", 'class="guest five"', 'class="home">'),
            ("home", "bench", 'class="home">', 'class="guest">'),
            ("away", "bench", 'class="guest">', None),
        ]
        players: dict[str, Any] = {"home": {"starters": [], "bench": []},
                                   "away": {"starters": [], "bench": []}}
        for side, slot, marker, end_marker in bounds:
            start = box.find(marker)
            if start < 0:
                continue
            end = box.find(end_marker, start + len(marker)) if end_marker else -1
            block = box[start:end] if end > start else box[start:]
            for p in re.finditer(
                r"techWinId='(\d+)'[^>]*>.*?<div class='number'>([^<]*)</div>.*?<div class='name'>([^<]+)</div>",
                block,
                re.S,
            ):
                players[side][slot].append({
                    "player_id": p.group(1),
                    "number": p.group(2).strip(),
                    "name": p.group(3).strip(),
                })
        if not players["home"]["starters"] and not players["away"]["starters"]:
            return None
        return {
            "home_team": team_names["home"],
            "away_team": team_names["away"],
            "home_formation": formations["home"],
            "away_formation": formations["away"],
            "lineups": players,
        }

    # ---- lineups / market splits (structured AJAX) -----------------------

    async def fetch_lineups(self, fixture: dict[str, Any]) -> dict[str, Any] | None:
        """Full player lists (starters + bench + per-player stats) for both
        teams via type=18. ``valid`` marks a starter; ``pName`` is the
        position; ``rating`` is the player's match rating. Context only.
        """
        match_id = fixture.get("match_id") or fixture.get("id")
        if not match_id:
            return None
        data = await self._get(
            "/ajax/soccerajax",
            {"type": 18, "id": match_id, "flesh": int(time.time() * 1000)},
        )
        if not isinstance(data, dict) or data.get("ErrCode") not in (0, None):
            return None
        body = data.get("Data") or {}
        out: dict[str, Any] = {
            "match_id": str(match_id),
            "home_team": fixture.get("home"),
            "away_team": fixture.get("away"),
            "source": "nowgoal",
        }
        for key, name in (("hList", "home"), ("gList", "away")):
            players = body.get(key)
            if not isinstance(players, list):
                continue
            out[name] = [
                {
                    "player_id": (p or {}).get("id"),
                    "name": (p or {}).get("name"),
                    "position": (p or {}).get("pName"),
                    "number": (p or {}).get("no"),
                    "starter": bool((p or {}).get("valid")),
                    "rating": (p or {}).get("rating"),
                }
                for p in players
                if isinstance(p, dict) and (p.get("name") or p.get("id"))
            ]
        if "home" not in out and "away" not in out:
            return None
        return out

    async def fetch_market_splits(self, fixture: dict[str, Any]) -> dict[str, Any] | None:
        """Historical AH / O/U / 1X2 probability splits by scope (type=22).

        ``*AllSclass`` = all matches, ``*MainSclass`` = main leagues,
        ``*SameSclass`` = this league. Each has Sum/Up/Draw/Down counts
        (AH: Up=home cover, Down=away cover; OU: Up=over, Down=under;
        OP: Up=home win, Down=away win). Context only.
        """
        match_id = fixture.get("match_id") or fixture.get("id")
        if not match_id:
            return None
        data = await self._get(
            "/ajax/soccerajax",
            {"type": 22, "id": match_id, "flesh": int(time.time() * 1000)},
        )
        if not isinstance(data, dict) or data.get("ErrCode") not in (0, None):
            return None
        body = data.get("Data") or {}
        groups: dict[str, dict[str, Any]] = {}
        for key in ("AH", "OU", "OP"):
            for scope in ("All", "Main", "Same"):
                raw = body.get(f"{key}{scope}Sclass")
                if not isinstance(raw, dict):
                    continue
                try:
                    groups[f"{key}_{scope}"] = {
                        "sum": int(raw.get("Sum") or 0),
                        "up": int(raw.get("Up") or 0),
                        "draw": int(raw.get("Draw") or 0),
                        "down": int(raw.get("Down") or 0),
                    }
                except (TypeError, ValueError):
                    continue
        if not groups:
            return None
        first = body.get("FirstOdds") or {}
        return {
            "match_id": str(match_id),
            "source": "nowgoal",
            "groups": groups,
            "first_odds": {
                k: v for k, v in first.items()
                if isinstance(v, (int, float)) and v > 0
            } or None,
        }


# ---- trend -> odds_snapshot rows ------------------------------------------

def trend_to_snapshots(
    trend: dict[str, Any] | None,
    *,
    kickoff: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a ``fetch_odds_trend`` payload into ``odds_snapshot``-shaped
    rows the movement engine / signal engine can consume directly.

    The background ``odds-poll`` builds its series from repeated snapshots;
    the trend endpoint gives the SAME kind of timestamped series in ONE
    call (every recorded odds change per bookmaker). For each bookmaker's
    market series, every PRE-MATCH point (``minute == ""``) becomes one row:

      {"event": "odds_snapshot", "ts": ISO, "timing": "T-…h/m",
       "odds_1x2": {home, draw, away} | None,
       "odds_ah": {line, home, away} | None,
       "odds_ou": {line, over, under} | None,
       "bookmakers_count": int, "sources": ["nowgoal_trend"],
       "bookmaker": str, "bookmaker_cid": int}

    ``bookmaker``/``bookmaker_cid`` attribute EACH row to its source book
    (P2 2026-08-24) -- the raw material for per-book sharp-money analysis
    (steam sync / sharp-vs-soft divergence) downstream. The movement engine
    ignores unknown fields, so poll-sourced rows (no attribution) and
    trend rows stay interchangeable. Prices are the parsed decimal odds
    (HK already converted); AH lines are in the HOME-handicap convention
    (flipped in ``_parse_trend_series``), matching the engine. In-play rows
    (non-empty ``minute``) are DROPPED -- live prices must never leak into a
    pre-match movement series. Rows are chronological across all
    bookmakers; duplicate instants across bookmakers are kept (each is a
    real observation, and the movement engine only requires chronological
    price points). Never fabricates a point that the provider did not
    record.
    """
    rows: list[dict[str, Any]] = []
    bookmakers = (trend or {}).get("bookmakers") or []
    for bm in bookmakers:
        bm_name = str(bm.get("name") or f"NowGoal-{bm.get('cid')}")
        bm_cid = bm.get("cid")
        for market, key in (("h2h", "odds_1x2"), ("ah", "odds_ah"), ("ou", "odds_ou")):
            series = bm.get(market) or []
            for r in series:
                if r.get("minute"):
                    continue  # in-play point, never part of pre-match history
                snapshot: dict[str, Any] = {
                    "event": "odds_snapshot",
                    "ts": r.get("ts"),
                    "timing": _trend_timing_label(r.get("ts"), kickoff),
                    "odds_1x2": None,
                    "odds_ah": None,
                    "odds_ou": None,
                    "bookmakers_count": len(bookmakers),
                    "sources": ["nowgoal_trend"],
                    "bookmaker": bm_name,
                    "bookmaker_cid": bm_cid,
                }
                if key == "odds_1x2":
                    o = {k: r.get(k) for k in ("home", "draw", "away")}
                    if all(v is not None and v > 1.0 for v in o.values()):
                        snapshot["odds_1x2"] = {k: round(float(v), 4) for k, v in o.items()}
                elif key == "odds_ah":
                    if r.get("home") is not None and r.get("away") is not None and r.get("line") is not None:
                        snapshot["odds_ah"] = {
                            "line": round(float(r["line"]), 4),
                            "home": round(float(r["home"]), 4),
                            "away": round(float(r["away"]), 4),
                        }
                else:
                    if r.get("over") is not None and r.get("under") is not None and r.get("line") is not None:
                        snapshot["odds_ou"] = {
                            "line": round(float(r["line"]), 4),
                            "over": round(float(r["over"]), 4),
                            "under": round(float(r["under"]), 4),
                        }
                if snapshot["odds_1x2"] is None and snapshot["odds_ah"] is None and snapshot["odds_ou"] is None:
                    continue
                rows.append(snapshot)
    rows.sort(key=lambda r: r.get("ts") or "")
    return rows


def _trend_timing_label(ts: str | None, kickoff: str | None) -> str:
    """Timing label (T-{h}h / T-{m}m) for one trend point vs kickoff.

    Mirrors runner.timing_label: >=1h to kickoff -> T-{h}h, inside the final
    hour -> T-{m}m. Unknown kickoff degrades to "T-0h" (the series is still
    chronological, so movement direction/magnitude remain computable)."""
    if not ts or not kickoff:
        return "T-0h"
    try:
        kd = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        if kd.tzinfo is None:
            kd = kd.replace(tzinfo=timezone.utc)
        pd = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if pd.tzinfo is None:
            pd = pd.replace(tzinfo=timezone.utc)
        hours = (kd - pd).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return "T-0h"
    if hours <= 0:
        return "T-0h"
    if hours >= 1.0:
        return f"T-{max(1, int(round(hours)))}h"
    return f"T-{max(1, int(round(hours * 60)))}m"


# ---- connectivity diagnostic ------------------------------------------------

def _looks_like_schedule(text: str) -> bool:
    """True iff the body is real nowgoal schedule JS (not a parked page /
    redirect body / ISP block page). Schedule responses are JS array source
    ("var A=Array(...)") or the error envelope {"ErrCode":...}.
    """
    if not text:
        return False
    low = text.lower()
    if "trustpositif" in low:
        return False
    return (
        "var a=array" in low
        or "matchcount=" in low
        or "errcode" in low
        or _SCHEDULE_RE_A.search(text) is not None
    )


def _looks_like_site(status: int, text: str, title: str) -> bool:
    """Heuristic: a real nowgoal page vs the ISP block page / error page.

    Pure diagnostic signal; the authoritative check is the schedule parse
    (see ``run_nowgoal_check``).
    """
    if status != 200 or not text or "trustpositif" in text.lower():
        return False
    low = text.lower()
    return (
        "nowgoal" in low
        or "football" in low
        or "live" in low
        or len(text) > 20000
    )


async def run_nowgoal_check(
    *,
    base_url: str | None = None,
    proxy: str | None = None,
    date: str | None = None,
    match_id: str | None = None,
    mirrors: list[str] | None = None,
    client: "NowGoalClient | None" = None,
) -> dict[str, Any]:
    """Diagnostic report: is NowGoal reachable from this network?

    ``status`` is one of:
      reachable   -- schedule endpoint parsed matches (authoritative)
      blocked     -- homepage answered with the ISP "Trustpositif" block page
      no_schedule -- site answers but no matches parsed (wrong date / moved)
      unreachable -- no connection at all (DNS/TCP refused)

    ``mirrors`` probes alternate domains with a light homepage+schedule check
    each (nowgoal domains rotate, so a blocked primary can often be replaced
    by a working mirror). ``client`` is a test hook -- when omitted one is
    built from ``base_url``/``proxy`` exactly like the runner does.
    """
    client = client or NowGoalClient(base_url=base_url, proxy=proxy)
    report: dict[str, Any] = {
        "base_url": client._base_url,
        "proxy": proxy or None,
        "status": "unknown",
        "checks": {},
    }

    try:
        hp = await client.probe_homepage()
    except Exception as exc:  # pragma: no cover -- defensive
        hp = {"http": None, "error": f"{type(exc).__name__}: {exc}"}
    report["checks"]["homepage"] = hp
    if hp.get("blocked"):
        report["status"] = "blocked"

    probe_date = date or datetime.now(timezone.utc).date().isoformat()
    sched = None
    try:
        sched = await client.fetch_schedule(probe_date)
    except Exception as exc:  # pragma: no cover -- defensive
        sched = None
    report["checks"]["schedule"] = {
        "date": probe_date,
        "matches_parsed": len(sched) if sched else 0,
        "sample": dict(sched[0]) if sched else None,
        "samples": [dict(m) for m in (sched or [])[:5]],
    }

    fixture = None
    if sched:
        if match_id:
            fixture = next((m for m in sched if m["match_id"] == str(match_id)), None)
        else:
            fixture = sched[0]
    if fixture is not None:
        try:
            payload = await client.fetch_odds(fixture)
        except Exception as exc:  # pragma: no cover -- defensive
            payload = None
        report["checks"]["odds"] = {
            "match_id": fixture["match_id"],
            "home": fixture["home"],
            "away": fixture["away"],
            "kickoff": fixture.get("kickoff"),
            "bookmakers": len((payload or {}).get("bookmakers") or []),
            "markets": sorted({
                m["key"]
                for bm in (payload or {}).get("bookmakers") or []
                for m in bm.get("markets") or []
            }),
            "normalized": payload is not None,
        }
    elif match_id:
        report["checks"]["odds"] = {
            "error": f"match_id {match_id} tidak ada di jadwal {probe_date}",
        }

    if report["status"] == "unknown":
        if (report["checks"].get("schedule") or {}).get("matches_parsed", 0) > 0:
            report["status"] = "reachable"
        elif (report["checks"].get("homepage") or {}).get("http") is None:
            report["status"] = "unreachable"
        else:
            report["status"] = "no_schedule"

    report["mirrors"] = []
    for url in mirrors or []:
        mclient = NowGoalClient(base_url=url, proxy=proxy)
        mreport: dict[str, Any] = {"base_url": mclient._base_url}
        try:
            mhp = await mclient.probe_homepage()
        except Exception:  # pragma: no cover -- defensive
            mhp = {"http": None, "error": "probe failed"}
        msched = None
        try:
            msched = await mclient.fetch_schedule(probe_date)
        except Exception:  # pragma: no cover -- defensive
            msched = None
        mreport["homepage"] = mhp
        mreport["matches_parsed"] = len(msched) if msched else 0
        report["mirrors"].append(mreport)

    return report


# ---- probe CLI -------------------------------------------------------------
#
# The bot's network (Indonesia) ISP-blocks nowgoal ("Trustpositif"), so this
# module cannot be exercised live here. From a network where nowgoal resolves
# (or with the bot's Tor proxy up), run:
#   python -m agents.football.nowgoal --home Arsenal --away Chelsea [--date 2026-08-15]
#   python -m agents.football.nowgoal --match-id 2346247
#   python -m agents.football.nowgoal --probe-mirrors
# It prints the resolved fixture, the RAW mixodds JSON (to verify the live
# euro/ou/ah shapes) and the normalized payload.


async def probe_mirrors(
    mirrors: list[str],
    *,
    proxy: str | None = None,
    timeout: float = 8.0,
    _fetch: Any = None,
) -> list[dict[str, Any]]:
    """Probe each mirror's AJAX endpoint (NOT the homepage).

    A parked/resurrecting mirror can serve a 200 homepage while the ajax
    paths the client depends on are gone (verified 2026-08-24: nowgoal.net
    homepage alive, ``/ajax/soccerajax`` -> nginx 404). The verdict is
    therefore "alive" only when the ajax path answers JSON -- ANY shape
    (``{"ErrCode":..}`` or even the anti-missing-referer ``{"code":1002}``)
    proves the endpoint exists; HTML/404/transport failure = dead.

    Returns one row per mirror: {mirror, ok, status, detail}, input order.
    ``_fetch`` injects a fake GET for tests:
    ``await fetch(url, headers, params) -> {"status": int|None, "body": str}``.
    """
    import httpx

    if not mirrors:
        return []

    async def _default_fetch(
        url: str, headers: dict[str, str], params: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(
                    connect=min(timeout, 4.0),
                    read=timeout,
                    write=timeout,
                    pool=timeout,
                ),
                "follow_redirects": True,
            }
            if proxy:
                kwargs["proxy"] = proxy
            async with httpx.AsyncClient(**kwargs) as cli:
                resp = await cli.get(url, params=params, headers=headers)
            return {"status": resp.status_code, "body": resp.text}
        except httpx.HTTPError as exc:
            return {"status": None, "error": f"{type(exc).__name__}: {exc}"}

    fetch_fn = _fetch or _default_fetch
    rows: list[dict[str, Any]] = []
    for m in mirrors:
        base = str(m).strip().rstrip("/") + "/"
        res = await fetch_fn(
            base.rstrip("/") + "/ajax/soccerajax",
            {"User-Agent": USER_AGENT, "Referer": base},
            {"type": 14, "t": 1, "id": 1, "h": 0},
        )
        status = res.get("status")
        body = res.get("body") or ""
        detail = res.get("error") or f"http {status}"
        ok = False
        if status == 200:
            stripped = body.lstrip()
            if not stripped.startswith("{"):
                # parked page / wrong path serving HTML with a 200 -- the
                # ajax contract this client depends on is absent.
                detail = "http 200 tapi body bukan JSON (bukan endpoint ajax)"
            else:
                try:
                    json.loads(body)
                    ok = True
                except json.JSONDecodeError:
                    detail = "http 200 tapi body bukan JSON valid"
        rows.append({"mirror": base, "ok": ok, "status": status, "detail": detail})
    return rows


def _probe(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.football.nowgoal")
    parser.add_argument("--home", help="home team name")
    parser.add_argument("--away", help="away team name")
    parser.add_argument("--match-id", help="nowgoal match id (from schedule)")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD schedule date")
    parser.add_argument("--proxy", default=None,
                        help="SOCKS/HTTP proxy, e.g. socks5h://127.0.0.1:9050 "
                             "(fallback: NOWGOAL_PROXY / HTTPS_PROXY / SOCKS_PROXY)")
    parser.add_argument("--out", default=None,
                        help="simpan fixture + raw mixodds ke file JSON (mis. raw_mixodds.json) "
                             "biar tinggal kirim nama filenya")
    parser.add_argument("--probe-mirrors", action="store_true",
                        help="audit kesehatan mirror ajax (config/football.json) lalu keluar")
    args = parser.parse_args(argv)

    if args.probe_mirrors:
        async def run_probe() -> int:
            import os as _os
            from pathlib import Path as _Path

            proxy = (args.proxy
                     or _os.getenv("NOWGOAL_PROXY")
                     or _os.getenv("HTTPS_PROXY")
                     or _os.getenv("SOCKS_PROXY"))
            cfg_mirrors: list[str] = []
            try:
                _cfg = json.loads(
                    (_Path(__file__).resolve().parent.parent.parent / "config" / "football.json")
                    .read_text(encoding="utf-8")
                )
                cfg_mirrors = list(((_cfg.get("nowgoal") or {}).get("mirrors")) or [])
            except (OSError, json.JSONDecodeError, ValueError):
                cfg_mirrors = []
            pool = cfg_mirrors or [DEFAULT_BASE_URL]
            rows = await probe_mirrors(pool, proxy=proxy)
            print(f"{'MIRROR':<38} {'STATUS':<8} DETAIL")
            for r in rows:
                print(
                    f"{r['mirror']:<38} "
                    f"{'ALIVE' if r['ok'] else 'DEAD':<8} {r['detail']}"
                )
            alive = [r["mirror"] for r in rows if r["ok"]]
            dead = [r["mirror"] for r in rows if not r["ok"]]
            if alive:
                print("\nsaran urutan config nowgoal.mirrors (alive duluan):")
                for a in alive:
                    print(f"  {a}")
                for d in dead:
                    print(f"  {d}   # mati saat probe -- boleh dipertahankan di ekor")
            return 0 if alive else 1

        try:
            return asyncio.run(run_probe())
        except KeyboardInterrupt:
            return 130

    async def run() -> int:
        import os
        proxy = (args.proxy
                 or os.getenv("NOWGOAL_PROXY")
                 or os.getenv("HTTPS_PROXY")
                 or os.getenv("SOCKS_PROXY")
                 or os.getenv("SOCCERDATA_PROXY"))
        client = NowGoalClient(proxy=proxy)
        if proxy:
            print(f"(proxy: {proxy})")
        fixture = None
        if args.home and args.away:
            fixture = await client.find_fixture(args.home, args.away, args.date)
        elif args.match_id:
            fixture = await client._find_by_id(args.match_id, args.date)
        if not fixture:
            print("fixture tidak ditemukan (cek nama tim / tanggal / jaringan)")
            return 1
        print("=== FIXTURE ===")
        print(json.dumps(fixture, ensure_ascii=False, indent=2))
        payload = await client.fetch_odds(fixture)
        print("\n=== NORMALIZED (The-Odds-API shape) ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2) if payload else "None")
        print("\n=== RAW mixodds (untuk verifikasi shape euro/ou/ah) ===")
        raw = await client._get(
            "/ajax/soccerajax",
            {"type": 14, "t": 1, "id": fixture["match_id"], "h": 0,
             "flesh": int(time.time() * 1000)},
        )
        print(json.dumps(raw, ensure_ascii=False, indent=2) if raw else "None")
        if args.out:
            from pathlib import Path

            out_path = Path(args.out)
            out_path.write_text(
                json.dumps(
                    {"fixture": fixture, "raw_mixodds": raw},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\n--> tersimpan ke {out_path.resolve()}")
        return 0

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(_probe(__import__("sys").argv[1:]))
