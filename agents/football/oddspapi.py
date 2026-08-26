"""OddsPapi (api.oddspapi.io/v4) client — secondary odds source.

Used only as a FALLBACK when The Odds API has no odds for a fixture. The
client resolves the fixture by tolerant team-name matching against the
daily fixtures feed, then fetches ``/odds?fixtureId=`` and normalizes the
response into the exact payload shape The Odds API produces::

    {"home_team": ..., "away_team": ..., "commence_time": ...,
     "bookmakers": [{"title": ..., "markets": [{"key": "h2h"|"totals"|"btts"|"asian_handicap",
                     "outcomes": [{"name", "price", "point"?}]}]}]}

so ``analyse.extract_h2h_entries`` and the totals/BTTS loops work unchanged.

Market metadata is hard-coded from a live probe of ``/markets?sportId=10``
(verified 2026-08; AH lines re-probed 2026-08-23): outcome IDs are sequential
from the marketId and names are canonical per market type — h2h -> 1/X/2,
btts -> Yes/No, totals -> Over/Under with a fixed handicap, and full-time
Asian Handicap -> Home/Away per line (see ``_AH_FULLTIME_MARKETS``). Fetching
the real catalog is avoided because it is ~9 MB per call. ``hasOdds`` fixtures
only. Note: once a fixture kicks off the free tier answers 403
RESTRICTED_ACCESS ("fixture is live") — pre-match only.

Rate limits on the free tier are tight (429 after a few calls), so the
client sleeps between calls and every network failure degrades to None —
the caller simply proceeds without odds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.oddspapi.io/v4"
USER_AGENT = "hermes-football/1.0 (local-advisory)"


def _parse_keys(raw: str | list[str] | None) -> list[str]:
    """Split comma/whitespace/semicolon separated keys into a deduped list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for k in raw:
            s = str(k).strip()
            if s:
                # allow comma inside list elements too
                if "," in s or ";" in s or "\n" in s:
                    out.extend(_parse_keys(s))
                else:
                    out.append(s)
        # dedupe preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq
    import re

    parts = re.split(r"[,\s;]+", str(raw).strip())
    seen = set()
    uniq = []
    for p in parts:
        s = p.strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

SOCCER_SPORT_ID = 10

# marketId -> {key, handicap} for the markets we normalise (probe-verified).
MARKET_SPECS: dict[str, dict[str, Any]] = {
    "101": {"key": "h2h", "handicap": 0.0},      # 1 / X / 2
    "104": {"key": "btts", "handicap": 0.0},     # Yes / No
    "106": {"key": "totals", "handicap": 0.5},   # Over / Under
    "1010": {"key": "totals", "handicap": 2.5},
    "1012": {"key": "totals", "handicap": 3.5},
    "1014": {"key": "totals", "handicap": 4.5},
    "10174": {"key": "totals", "handicap": 3.0},
    "10180": {"key": "totals", "handicap": 4.0},
}

# Canonical outcome names per market key, assigned by ascending outcomeId
# (OddsPapi numbers outcome IDs sequentially from the marketId).
_OUTCOME_NAMES: dict[str, list[str]] = {
    "h2h": ["1", "X", "2"],
    "btts": ["Yes", "No"],
    "totals": ["Over", "Under"],
}

# Asian Handicap FULL-TIME lines, marketId -> home-relative line
# (probe-verified from /markets?sportId=10, 2026-08-23). Each line is its own
# market ("Asian Handicap", period=fulltime, marketType=spreads) with two
# sequential outcomes: id+0 -> Home, id+1 -> Away; handicap is signed relative
# to HOME (negative = home gives). Orientation verified live against known
# favourites (Frosinone-Juventus: away @1.15 on level ball; Venezia-Lecce:
# home -0.25 @1.58). First/Second-Half AH families (ids ~10602+/~106xx p1/p2)
# are deliberately EXCLUDED -- extract_asian_handicap consumes MATCH-level
# lines only. No extra API call is needed: these ids arrive inside the same
# /odds response already fetched.
_AH_FULLTIME_MARKETS: dict[str, float] = {
    "1024": -6.0, "1026": -5.75, "1028": -5.5, "1030": -5.25,
    "1032": -5.0, "1034": -4.75, "1036": -4.5, "1038": -4.25,
    "1040": -4.0, "1042": -3.75, "1044": -3.5, "1046": -3.25,
    "1048": -3.0, "1050": -2.75, "1052": -2.5, "1054": -2.25,
    "1056": -2.0, "1058": -1.75, "1060": -1.5, "1062": -1.25,
    "1064": -1.0, "1066": -0.75, "1068": -0.5, "1070": -0.25,
    "1072": 0.0, "1074": 0.25, "1076": 0.5, "1078": 0.75,
    "1080": 1.0, "1082": 1.25, "1084": 1.5, "1086": 1.75,
    "1088": 2.0, "1090": 2.25, "1092": 2.5, "1094": 2.75,
    "1096": 3.0, "1098": 3.25, "10100": 3.5, "10102": 3.75,
    "10104": 4.0, "10106": 4.25, "10108": 4.5, "10110": 4.75,
    "10112": 5.0, "10114": 5.25, "10116": 5.5, "10118": 5.75,
    "10120": 6.0,
}


class OddspapiClient:
    def __init__(
        self,
        api_key: str | list[str] | None = None,
        throttle_seconds: float = 2.5,
        timeout: float = 20.0,
        *,
        api_keys: list[str] | None = None,
        state_path: str | os.PathLike[str] | None = None,
    ) -> None:
        # ---- pool init (rolling keys 2026-08-26) --------------------------
        # Accept: single str, comma/whitespace-separated str, list[str], or
        # both api_key + api_keys. Backwards compat: OddspapiClient("k") still
        # works as single-key pool.
        raw_keys: list[str] = []
        if api_key is not None:
            raw_keys.extend(_parse_keys(api_key))
        if api_keys is not None:
            raw_keys.extend(_parse_keys(api_keys))
        if not raw_keys:
            raise ValueError("OddspapiClient needs at least one apiKey")
        # dedupe preserve order
        seen: set[str] = set()
        self._keys: list[str] = []
        for k in raw_keys:
            if k not in seen:
                seen.add(k)
                self._keys.append(k)
        self._current: int = 0
        self._exhausted: dict[int, float] = {}  # index -> expiry epoch
        self._state_path: Path | None = Path(state_path) if state_path else None
        self._load_state()
        # legacy alias: self._key returns current key (property below)
        # but keep instance attr for direct writes via setter
        self._throttle = throttle_seconds
        self._timeout = timeout
        # The /fixtures feed is fetched per date window; one analysis batch
        # resolves many pairs on the same day, so cache the feed once per
        # window and reuse it -- one /fixtures call instead of one per pair.
        # The free tier 429s after a handful of calls, so this is what makes
        # multi-match analysis survivable.
        self._feed_cache: dict[str, list[dict[str, Any]]] = {}
        # Quota status (2026-08-22): surfaced on the Discord card so a silent
        # 429 no longer looks like "odds are just missing". Stays exhausted
        # until one request succeeds again (free-tier windows reset hourly).
        # Rolling pool: True only when ALL keys exhausted.
        self.quota_exhausted = False
        self.last_remaining: int | None = None
        if len(self._keys) > 1:
            logger.info("oddspapi pool: %d keys, active #%d", len(self._keys), self._current + 1)

    # ---- pool helpers -------------------------------------------------

    @property
    def _key(self) -> str:  # noqa: D401 -- backward compat alias
        return self._keys[self._current] if self._keys else ""

    @_key.setter
    def _key(self, value: str) -> None:
        # Allow legacy direct assignment to override current key
        if value and value not in self._keys:
            self._keys[self._current] = value
        elif value:
            # switch to that key if it exists in pool
            try:
                self._current = self._keys.index(value)
            except ValueError:
                self._keys[self._current] = value

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    @property
    def active_key_index(self) -> int:
        return self._current

    @property
    def pool_status(self) -> dict[str, Any]:
        return {
            "total": len(self._keys),
            "active_index": self._current,
            "active_preview": self._mask(self._keys[self._current]) if self._keys else None,
            "exhausted_count": sum(1 for e in self._exhausted.values() if e > time.time()),
            "quota_exhausted": self.quota_exhausted,
            "last_remaining": self.last_remaining,
        }

    @staticmethod
    def _mask(key: str) -> str:
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    def _load_state(self) -> None:
        if not self._state_path or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            idx = int(data.get("current_index", 0))
            if 0 <= idx < len(self._keys):
                self._current = idx
            # load exhausted expiries, drop expired ones
            now = time.time()
            raw_ex = data.get("exhausted") or {}
            for k, v in raw_ex.items():
                try:
                    ki = int(k)
                    exp = float(v)
                    if 0 <= ki < len(self._keys) and exp > now:
                        self._exhausted[ki] = exp
                except (TypeError, ValueError):
                    continue
            # if current is exhausted, advance to next available
            if self._current in self._exhausted and self._exhausted[self._current] > now:
                self._rotate(silent=True)
        except Exception as exc:  # noqa: BLE001 -- state is best-effort
            logger.debug("oddspapi pool state load failed: %s", exc)

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "current_index": self._current,
                "exhausted": {str(k): v for k, v in self._exhausted.items()},
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            self._state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("oddspapi pool state save failed: %s", exc)

    def _is_exhausted(self, idx: int) -> bool:
        exp = self._exhausted.get(idx)
        if exp is None:
            return False
        if exp <= time.time():
            self._exhausted.pop(idx, None)
            return False
        return True

    def _all_exhausted(self) -> bool:
        now = time.time()
        # purge expired
        for k in list(self._exhausted.keys()):
            if self._exhausted[k] <= now:
                self._exhausted.pop(k)
        return len([i for i in range(len(self._keys)) if self._is_exhausted(i)]) >= len(self._keys)

    def _mark_exhausted(self, idx: int, ttl: float = 3600.0) -> None:
        self._exhausted[idx] = time.time() + ttl
        self._save_state()

    def _rotate(self, silent: bool = False) -> int:
        """Advance to next non-exhausted key; returns new index."""
        if len(self._keys) <= 1:
            return self._current
        start = self._current
        for _ in range(len(self._keys)):
            self._current = (self._current + 1) % len(self._keys)
            if not self._is_exhausted(self._current):
                break
        else:
            # all exhausted -- stay on next slot anyway
            self._current = (start + 1) % len(self._keys)
        if not silent:
            logger.warning(
                "oddspapi rolling: [%d/%d] %s -> [%d/%d] %s",
                start + 1, len(self._keys), self._mask(self._keys[start]),
                self._current + 1, len(self._keys), self._mask(self._keys[self._current]),
            )
            self._save_state()
        return self._current

    def _available_keys(self) -> int:
        return sum(1 for i in range(len(self._keys)) if not self._is_exhausted(i))

    # ---- low-level HTTP -------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        url = f"{BASE_URL}{path}"
        max_attempts = len(self._keys) if self._keys else 1
        attempts = 0
        while attempts < max_attempts:
            cur_key = self._keys[self._current] if self._keys else ""
            cur_preview = self._mask(cur_key)
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        url,
                        params={**params, "apiKey": cur_key},
                        headers={"User-Agent": USER_AGENT},
                    )
            except httpx.HTTPError as exc:
                logger.warning("oddspapi network error (%s): %s", path, exc)
                return None
            if resp.status_code == 401:
                logger.warning(
                    "oddspapi 401 invalid key [%d/%d] %s on %s",
                    self._current + 1, len(self._keys), cur_preview, path,
                )
                # Invalid key -> ban 24 jam (bukan selamanya, biar bisa di-fix)
                self._mark_exhausted(self._current, ttl=86400.0)
                if self._all_exhausted():
                    logger.warning("oddspapi pool exhausted: semua %d key invalid/habis", len(self._keys))
                    self.quota_exhausted = True
                    await asyncio.sleep(self._throttle)
                    return None
                self._rotate()
                attempts += 1
                await asyncio.sleep(0.3)
                continue
            if resp.status_code == 429:
                logger.warning(
                    "oddspapi 429 quota habis key [%d/%d] %s on %s -> rolling",
                    self._current + 1, len(self._keys), cur_preview, path,
                )
                self._mark_exhausted(self._current, ttl=3600.0)
                if self._all_exhausted():
                    logger.warning("oddspapi pool exhausted: semua %d key habis quota (TTL 1 jam)", len(self._keys))
                    self.quota_exhausted = True
                    await asyncio.sleep(self._throttle)
                    return None
                self._rotate()
                attempts += 1
                await asyncio.sleep(0.3)
                continue
            if resp.status_code >= 400:
                logger.warning("oddspapi http %s on %s", resp.status_code, path)
                return None
            await asyncio.sleep(self._throttle)
            self.quota_exhausted = False
            for header, value in resp.headers.items():
                if "remaining" in header.lower():
                    try:
                        self.last_remaining = int(value)
                    except (TypeError, ValueError):
                        pass
                    break
            try:
                return resp.json()
            except ValueError:
                return None
        self.quota_exhausted = True
        return None

    # ---- fixture resolution ----------------------------------------------

    async def find_fixture(
        self,
        home: str,
        away: str,
        kickoff: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the OddsPapi fixture for a match via tolerant name matching.

        Scans the fixtures feed for the kickoff date (or today/tomorrow when
        kickoff is unknown) and returns the fixture dict with odds. Returns
        None when the fixture cannot be matched.
        """
        try:
            if kickoff:
                dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                day = dt.date()
            else:
                day = datetime.now(timezone.utc).date()
        except ValueError:
            day = datetime.now(timezone.utc).date()

        frm = (day - timedelta(days=1)).isoformat()
        to = (day + timedelta(days=2)).isoformat()
        cache_key = f"{frm}|{to}"
        items = self._feed_cache.get(cache_key)
        if items is None:
            data = await self._get(
                "/fixtures",
                {
                    "sportId": SOCCER_SPORT_ID,
                    # day-1..day+2: covers fixtures whose kickoff is slightly off
                    # the resolved date (the feed lists them by startTime, which
                    # can drift a day in either direction for qualification legs).
                    "from": frm,
                    "to": to,
                },
            )
            # /fixtures returns a bare list; tolerate dict wrappers too.
            data = data if isinstance(data, list) else (data or {}).get("data", data or {})
            if isinstance(data, dict):
                data = data.get("fixtures", data)
            if not isinstance(data, list):
                return None
            self._feed_cache[cache_key] = data
            items = data

        candidates: list[dict[str, Any]] = []
        for fx in items:
            if not fx.get("hasOdds"):
                continue
            h = fx.get("participant1Name") or ""
            a = fx.get("participant2Name") or ""
            if _same_team(h, home) and _same_team(a, away):
                candidates.append(fx)
        if not candidates:
            return None
        # Prefer the earliest kickoff when several legs match.
        candidates.sort(key=lambda f: f.get("startTime") or "")
        return candidates[0]

    # ---- odds ------------------------------------------------------------

    async def fetch_odds(self, fixture: dict[str, Any]) -> dict[str, Any] | None:
        """Normalized The-Odds-API-shaped payload for a fixture, or None."""
        fixture_id = fixture.get("fixtureId")
        if not fixture_id:
            return None
        data = await self._get("/odds", {"fixtureId": fixture_id})
        if not isinstance(data, dict):
            return None
        body = data.get("data", data)
        if isinstance(body, dict):
            body = body.get("odds", body)
        bookmaker_odds = (body or {}).get("bookmakerOdds")
        if not isinstance(bookmaker_odds, dict):
            return None

        home_name = fixture.get("participant1Name") or ""
        away_name = fixture.get("participant2Name") or ""

        bookmakers: list[dict[str, Any]] = []
        for title, bm in bookmaker_odds.items():
            if not isinstance(bm, dict):
                continue
            if bm.get("suspended") or not bm.get("bookmakerIsActive", True):
                continue
            markets = bm.get("markets")
            if not isinstance(markets, dict):
                continue
            out_markets: list[dict[str, Any]] = []
            for mid, mk in markets.items():
                spec = MARKET_SPECS.get(str(mid))
                ah_line = _AH_FULLTIME_MARKETS.get(str(mid))
                if spec is None and ah_line is None:
                    continue
                if not isinstance(mk, dict):
                    continue
                if not mk.get("marketActive", True):
                    continue
                outcomes_raw = mk.get("outcomes")
                if not isinstance(outcomes_raw, dict):
                    continue
                if spec is not None:
                    key = spec["key"]
                    names = _OUTCOME_NAMES[key]
                    if key == "h2h":
                        # The Odds API names h2h outcomes with the real team
                        # names ("Draw" for X) so extract_h2h_entries matches
                        # them directly against the resolved fixture names.
                        names = [home_name, "Draw", away_name]
                    point = spec["handicap"]
                else:
                    # Asian Handicap full-time: outcome id+0 -> Home,
                    # id+1 -> Away, line home-relative (see _AH_FULLTIME_MARKETS).
                    key = "asian_handicap"
                    names = ["Home", "Away"]
                    point = ah_line
                # collect (outcomeId, price) sorted by outcomeId; odds
                # outcome IDs are sequential from the marketId, so this
                # recovers the canonical Over/Under, 1/X/2, Yes/No order.
                priced: list[tuple[int, float]] = []
                for oid, oc in outcomes_raw.items():
                    if not isinstance(oc, dict):
                        continue
                    players = oc.get("players")
                    if not isinstance(players, dict):
                        continue
                    # Prefer an active player entry, but accept an inactive
                    # one too: right around kickoff (or for fast-closing
                    # markets) the API marks lines inactive while still
                    # returning their last price. Without this fallback a
                    # perfectly good price is discarded.
                    price = None
                    for pl in players.values():
                        if isinstance(pl, dict) and isinstance(pl.get("price"), (int, float)):
                            if pl.get("active"):
                                price = float(pl["price"])
                                break
                            if price is None:
                                price = float(pl["price"])
                    if price is None:
                        continue
                    try:
                        priced.append((int(oid), price))
                    except (TypeError, ValueError):
                        continue
                priced.sort(key=lambda t: t[0])
                if len(priced) < len(names):
                    continue
                outcomes: list[dict[str, Any]] = []
                for i, name in enumerate(names[: len(priced)]):
                    outcomes.append({"name": name, "price": priced[i][1], "point": point})
                out_markets.append({"key": key, "outcomes": outcomes})
            if out_markets:
                bookmakers.append({"title": title, "markets": out_markets})
        if not bookmakers:
            return None
        return {
            "home_team": home_name,
            "away_team": away_name,
            "commence_time": fixture.get("startTime") or "",
            "bookmakers": bookmakers,
        }

    async def match_odds(
        self,
        home: str,
        away: str,
        kickoff: str | None = None,
    ) -> dict[str, Any] | None:
        """One-shot: resolve fixture and fetch normalized odds."""
        fixture = await self.find_fixture(home, away, kickoff)
        if not fixture:
            return None
        return await self.fetch_odds(fixture)


# ---- tolerant team-name matching (mirrors analyse._teams_match) ----------

import re
import unicodedata

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


def _norm_team(name: str) -> str:
    # Parenthetical country suffixes ("Tobol (Kaz)") come from the flashscore
    # homepage and are not part of OddsPapi's names; drop them so the suffix
    # cannot break matching ("tobol" must match "Tobol Kostanay").
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
    # Token containment: every token of the shorter name must appear in the
    # longer one (as an exact token or a substring, min 3 chars). Short names
    # like "tobol"/"rfs" that the old len>=6 containment gate rejected now
    # match ("Tobol (Kaz)" -> "tobol" in "Tobol Kostanay"). Containment is
    # SYMMETRIC so "Hearts" matches the feed's "Heart of Midlothian FC"
    # ("heart" in "hearts"). ponytail: a short token that is a substring of
    # many feed names ("viking" -> "Vikingur") could match the wrong tie; the
    # kickoff-window narrowing in find_fixture bounds that risk. Upgrade: real
    # token-level aliases (teams.json) instead of substring heuristics.
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
