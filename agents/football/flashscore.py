"""Flashscore provider (browser UC, SPA render).

Flashscore (https://www.flashscore.com) has no official API and serves
fixtures/results as a JS SPA. Plain requests return the shell HTML with no
match data; the data is rendered by JavaScript. A real Chrome driven by
seleniumbase in UC (undetected) mode renders it fine, and Flashscore does
NOT challenge the browser (verified 2026-08: homepage/league/match/team
pages load, no Cloudflare interstitial).

Feasibility tests (2026-08) confirmed:
  - League page  /football/{region}/{league}/  -> .event__match rows with
    participant names + match links shaped
    /match/football/{home-slug}-{home-hash}/{away-slug}-{away-hash}/?mid={mid}
  - Team results /team/{slug}/{id}/results/    -> rows "date | home | away |
    hg | ag | W/D/L" (form source)
  - Team fixtures /team/{slug}/{id}/fixtures/  -> upcoming rows
  - Match summary page -> [data-testid='wcl-statistics-category'] +
    [data-testid='wcl-statistics-value'] > strong (xG, possession, shots...)
  - Match h2h page    -> class*="h2h" blocks ("LAST MATCHES: ...")

The internal JSON API (global.flashscore.ninja/x/feed + X-Fsign header) is
reachable but match payloads are hash-encoded and not worth decoding; we use
the DOM instead.

This module is PURE data extraction: no model logic. Callers (multi_source)
normalize the returned structures.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

from .team_alias import resolve_team_alias  # noqa: E402

BASE_URL = "https://www.flashscore.com"

# league key -> flashscore URL path. Fallback for unknown leagues: scrape the
# homepage match links (today's matches) like sofascore's match-link resolver.
LEAGUE_PATHS: dict[str, str] = {
    "EPL": "/football/england/premier-league",
    "LaLiga": "/football/spain/laliga",
    "Serie A": "/football/italy/serie-a",
    "Bundesliga": "/football/germany/bundesliga",
    "Ligue 1": "/football/france/ligue-1",
    "Ligue 2": "/football/france/ligue-2",
    "Primeira Liga": "/football/portugal/liga-portugal",
    "Eredivisie": "/football/netherlands/eredivisie",
    "UCL": "/football/europe/champions-league",
    "UEL": "/football/europe/europa-league",
    "UECL": "/football/europe/conference-league",
    "Scottish Premiership": "/football/scotland/premiership",
    "Super Lig": "/football/turkey/super-lig",
    "Belgian Pro League": "/football/belgium/jupiler-pro-league",
    "EFL Championship": "/football/england/championship",
    "Saudi Pro League": "/football/saudi-arabia/saudi-professional-league",
    "MLS": "/football/usa/mls",
    "Serie B": "/football/italy/serie-b",
    "Segunda": "/football/spain/laliga2",
    "A-League": "/football/australia/a-league",
    "K-League": "/football/south-korea/k-league-1",
    "J1 League": "/football/japan/j1-league",
    "Liga 1": "/football/indonesia/super-league",
}

_TRANSLIT = {
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "å": "a", "Å": "a",
    "ö": "o", "Ö": "o", "ü": "u", "Ü": "u", "ä": "a", "Ä": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "á": "a", "à": "a",
    "â": "a", "ã": "a", "í": "i", "ì": "i", "î": "i", "ó": "o",
    "ò": "o", "ô": "o", "õ": "o", "ú": "u", "ù": "u", "û": "u",
    "ç": "c", "Ç": "c", "ñ": "n", "Ñ": "n", "š": "s", "č": "c",
    "ž": "z", "ß": "ss", "ő": "o", "ű": "u", "ă": "a", "ș": "s",
    "ț": "t", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ż": "z",
    "ą": "a", "ę": "e", "ś": "s", "ć": "c", "ń": "n", "ó": "o",
    "ý": "y", "ř": "r", "ů": "u",
}


def _slugify(name: str) -> str:
    """Lowercase, transliterate, hyphenate like flashscore slugs.

    Transliteration happens BEFORE splitting so accented letters ('ø' in
    'Bodø/Glimt') are mapped to their ASCII base ('o') instead of being
    treated as separators. 'Bodø/Glimt' -> 'bodo-glimt'.
    """
    out: list[str] = []
    for ch in (name or "").lower():
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            out.append(ch)
        elif ch in _TRANSLIT:
            out.append(_TRANSLIT[ch])
        elif ch in " -/._&,":
            out.append(" ")
        # other punctuation is dropped
    parts = [p for p in "".join(out).split() if p]
    return "-".join(parts)


def _squash(s: str) -> str:
    """Remove every separator for tolerant slug matching."""
    return re.sub(r"[^a-z0-9]+", "", _slugify(s))


def _norm_name(name: str) -> str:
    """Lowercase team name, strip country tags like '(Nor)', ' (Bel)'."""
    s = (name or "").lower()
    s = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _squash_matches(sq_name: str, sq_slug: str) -> bool:
    """True when a squashed name and slug refer to the same team.

    Substring both ways tolerates abbreviations ("Ind. del Valle" vs
    "independiente-del-valle"); the longer of the two must contain the
    shorter so "real" (slug) doesn't match "Real Madrid" when the full
    slug is also present.
    """
    if not sq_name or not sq_slug:
        return False
    short, long = (sq_name, sq_slug) if len(sq_name) <= len(sq_slug) else (sq_slug, sq_name)
    return short in long


def _assign_slug_roles(
    slug_a: str, id_a: str, slug_b: str, id_b: str, home_name: str, away_name: str
) -> tuple[str, str, str, str]:
    """Return (home_slug, home_id, away_slug, away_id) by matching names to
    slugs. Flashscore match URLs do not guarantee home-first slug order, and
    the DOM participant elements are authoritative for which team is home, so
    we match each name against both slugs and swap when needed. The home name
    decides; the away name is a tiebreaker. Falls back to the URL order when
    neither name matches reliably."""
    sq_a = _squash(slug_a.replace("-", " "))
    sq_b = _squash(slug_b.replace("-", " "))
    sq_home = _squash(_norm_name(home_name))
    sq_away = _squash(_norm_name(away_name))
    a_matches_home = _squash_matches(sq_home, sq_a)
    b_matches_home = _squash_matches(sq_home, sq_b)
    if a_matches_home and not b_matches_home:
        return slug_a, id_a, slug_b, id_b
    if b_matches_home and not a_matches_home:
        return slug_b, id_b, slug_a, id_a
    if a_matches_home and b_matches_home:
        # both slugs match home (one is a substring of the other) -- prefer
        # the longer slug, which is the fuller team name
        if len(sq_b) > len(sq_a):
            return slug_b, id_b, slug_a, id_a
        return slug_a, id_a, slug_b, id_b
    # home name unreliable -- away name decides the roles
    a_matches_away = _squash_matches(sq_away, sq_a)
    b_matches_away = _squash_matches(sq_away, sq_b)
    if a_matches_away and not b_matches_away:
        return slug_b, id_b, slug_a, id_a
    if b_matches_away and not a_matches_away:
        return slug_a, id_a, slug_b, id_b
    # no reliable match: keep URL order
    return slug_a, id_a, slug_b, id_b


def _row_status(
    status_cls: Any,
    row_cls: Any = None,
    date_text: Any = None,
    txt: Any = None,
) -> str:
    """Classify a homepage/league match row as 'scheduled' | 'live' | 'finished'.

    Signals, in order of reliability:
      1. Modifier class on the time cell / row (event__time--live, --finished,
         --scheduled) -- verified live.
      2. The rendered text: a live minute ("62'"), a full-time marker ("FT"),
         a kickoff time ("21:00"), or a score ("2:1").
      3. The row innerText (``txt``): "FT"/"AET"/... markers or a live minute.
      4. ABSENCE of the time cell: flashscore replaces ``.event__time`` with
         the score/minute as soon as a match kicks off (verified live on the
         homepage: finished fixtures have no time element and no 'finished'
         class), so a row with no time element that is not live is DONE.

    This drives the "hanya match yang belum bertanding" filter for the top
    command (finished/live rows are dropped before rendering).
    """
    cls = f"{status_cls or ''} {row_cls or ''}".lower()
    if "finished" in cls:
        return "finished"
    if "live" in cls:
        return "live"
    if "scheduled" in cls:
        return "scheduled"
    body = (txt or "").lower()
    if re.search(r"\b(ft|aet|pen|pens|awd|abd|canc|postp|susp|int|delay|tbd)\b", body):
        return "finished"
    if re.search(r"\d{1,3}\s*[’']", body):
        return "live"  # live minute inside the row text
    dt = str(date_text or "").strip().lower()
    if dt in ("ft", "aet", "pen", "pens", "awd", "abd", "canc", "postp",
              "susp", "int", "delay", "tbd"):
        return "finished"
    if dt == "ht":
        return "live"  # half-time break: the match HAS started
    if re.fullmatch(r"\d{1,3}\s*[’']\s*(\+\d+)?", dt):
        return "live"  # live minute, e.g. "62'" / "120'+3"
    # A clock ("21:00", "02:00") is a kickoff time; a score ("2:1", "3-3")
    # has an unpadded minute side, so it never matches the clock pattern.
    if re.fullmatch(r"\d{1,2}:[0-5]\d", dt):
        return "scheduled"
    if re.fullmatch(r"\d{1,2}[:.-]\d{1,2}", dt):
        return "finished"
    # No time element at all -> the match has already kicked off (the time
    # cell is swapped for score/minute); since it is not live, it is done.
    if not (status_cls or ""):
        return "finished"
    return "scheduled"


def _squash_variants(kw: str) -> list[str]:
    """Squashed match variants for a query: the raw name plus any canonical
    team-alias resolution (e.g. "Hearts" -> "Heart of Midlothian", which the
    raw "hearts" token can never contain). A fixture that renders under the
    canonical name is then found for free, without an extra render.

    F3 (2026-08-17, wrong-team bug): also include the standings-table
    spellings ("Atletico Madrid" -> "Atl. Madrid", squashed "atlmadrid").
    Before this, a query whose canonical differs from the rendered
    abbreviation ("Atletico Madrid" vs the league page's "Atl. Madrid")
    never matched the correct row, the team-fixtures fallback ran, and the
    poisoned alias variant could match an unrelated team's fixture (the
    phantom Real Madrid vs Malaga). The abbreviations are squashed into the
    same key space as the row names so "Atl. Madrid" == "atlmadrid"."""
    base = _squash(_norm_name(kw))
    out = [base] if base else []
    try:
        aliased = resolve_team_alias(kw, None)
    except Exception:
        aliased = None
    if aliased:
        sq = _squash(_norm_name(aliased))
        if sq and sq not in out:
            out.append(sq)
    try:
        from .team_alias import standings_abbreviations

        for ab in standings_abbreviations(kw):
            sq = _squash(_norm_name(ab))
            if sq and sq not in out:
                out.append(sq)
    except Exception:  # noqa: BLE001 -- abbreviations are best-effort
        pass
    return out


def _find_pair_in_rows(
    rows: list[dict[str, Any]], norm_home: str | list[str], norm_away: str | list[str]
) -> dict[str, Any] | None:
    """Find the home/away pair inside scraped league/homepage match rows.

    Matching is substring-tolerant on squashed names (so 'royale union'
    matches 'Royale Union SG (Bel)') and honors the side the caller asked
    for: if the pair appears swapped in the row (caller typed the away team
    as home), the returned home/away roles are swapped accordingly. Either
    side accepts a LIST of squashed variants (raw name + team-alias
    canonical) and matches when ANY variant hits. Returns the resolved
    match dict (with ``score`` when the row carries one) or None.
    """
    home_variants = [n for n in (norm_home if isinstance(norm_home, list) else [norm_home]) if n]
    away_variants = [n for n in (norm_away if isinstance(norm_away, list) else [norm_away]) if n]
    if not (home_variants and away_variants):
        return None
    for r in rows:
        h = _squash(_norm_name(r.get("home_name") or ""))
        a = _squash(_norm_name(r.get("away_name") or ""))
        if not (h and a):
            continue
        home_in_home = any(nh in h or h in nh for nh in home_variants)
        away_in_away = any(na in a or a in na for na in away_variants)
        if home_in_home and away_in_away:
            return {
                "home": {"slug": r["home_slug"], "id": r["home_id"], "name": r["home_name"]},
                "away": {"slug": r["away_slug"], "id": r["away_id"], "name": r["away_name"]},
                "match_url": r["match_url"],
                "date_text": r.get("date_text"),
                "score": r.get("score"),
                # Competition section tag when the row came from a
                # competition-aware scrape (homepage). None for plain league
                # pages, where the caller supplies the league label instead.
                "competition": r.get("competition"),
                "source": "flashscore",
            }
        # swapped sides in the row (defensive): home variant on the away side
        # AND away variant on the home side
        home_in_away = any(nh in a or a in nh for nh in home_variants)
        away_in_home = any(na in h or h in na for na in away_variants)
        if home_in_away and away_in_home:
            return {
                "home": {"slug": r["away_slug"], "id": r["away_id"], "name": r["away_name"]},
                "away": {"slug": r["home_slug"], "id": r["home_id"], "name": r["home_name"]},
                "match_url": r["match_url"],
                "date_text": r.get("date_text"),
                "score": r.get("score"),
                "competition": r.get("competition"),
                "source": "flashscore",
            }
    return None


def _to_int(v: Any) -> int | None:
    """Strict integer coercion (never raises); None on junk."""
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _parse_h2h_section(
    section: str,
    team_a_name: str,
    team_b_name: str,
) -> dict[str, Any] | None:
    """Pure parser for the flashscore H2H section (F5, 2026-08-17).

    The current layout renders each direct meeting as a column of lines
    (verified live 2026-08-17)::

        HEAD-TO-HEAD MATCHES
        17.08.26
        ACN
        Cameroon W
        Malawi W
        3
        0

    i.e. [date, competition, home, away, home_goals, away_goals] per
    meeting. The old parser looked for a single line containing BOTH team
    names plus a score -- it matched nothing on this layout. This parser
    walks the lines and scores a meeting when it finds a run of
    [home, away, hg, ag] whose home/away names match the two teams (either
    order). Returns {wins, draws, losses, count, source} or None when no
    direct meeting is found (no h2h history -> caller falls back).
    """
    lines = [ln.strip() for ln in (section or "").splitlines() if ln.strip()]
    if not lines:
        return None
    na = _squash(_norm_name(team_a_name))
    nb = _squash(_norm_name(team_b_name))
    if not (na and nb):
        return None
    wins = draws = losses = 0
    count = 0
    # Skip the section title itself.
    start = 1 if lines[0].upper().startswith("HEAD-TO-HEAD") else 0
    i = start
    while i < len(lines) - 3:
        ha = _squash(_norm_name(lines[i]))
        hb = _squash(_norm_name(lines[i + 1]))
        hg = _to_int(lines[i + 2])
        ag = _to_int(lines[i + 3])
        if hg is not None and ag is not None:
            if ha == na and hb == nb:
                first_is_a = True
            elif ha == nb and hb == na:
                first_is_a = False
            else:
                i += 1
                continue
            count += 1
            if hg == ag:
                draws += 1
            elif (first_is_a and hg > ag) or (not first_is_a and ag > hg):
                wins += 1
            else:
                losses += 1
            i += 4  # consume the whole meeting row
            continue
        i += 1
    if count == 0:
        return None
    return {"wins": wins, "draws": draws, "losses": losses, "count": count, "source": "flashscore_h2h"}


def parse_lineups_page(data: dict[str, Any]) -> dict[str, Any] | None:
    """Pure parser for the flashscore lineups tab payload (testable).

    ``data`` is the raw scrape: {homePlayers, awayPlayers, headers, body}.
    Player entries look like "22 | Mikelionis" (jersey | name); formation
    values look like "4 - 3 - 3" and live next to a "FORMATION" label in
    the header elements (home first, then away). Returns
    {status, formations, home, away, home_count, away_count, source} or
    None when no players are present (lineups not announced yet).
    """
    def _parse_players(raw: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for text in raw or []:
            # The element innerText renders "<jersey>\n<name>" (verified live)
            # but some page variants use "<jersey> | <name>"; split on both.
            parts = [p.strip() for p in re.split(r"[|\n]", text or "") if p.strip()]
            if not parts:
                continue
            number: str | None = None
            if parts[0].isdigit():
                number = parts[0]
            name = parts[-1]
            if name:
                out.append({"number": number, "name": name})
        return out

    home = _parse_players(data.get("homePlayers"))
    away = _parse_players(data.get("awayPlayers"))
    if not home and not away:
        return None

    # Formation values are digit-dash sequences inside the header elements
    # ("4 - 3 - 3"), in DOM order: home first, then away. The "FORMATION"
    # label itself carries no digits and is filtered out naturally.
    formations: list[str] = []
    for header in data.get("headers") or []:
        for m in re.finditer(r"\d\s*-\s*\d(?:\s*-\s*\d)*", re.sub(r"\s+", " ", header or "")):
            formations.append(re.sub(r"\s*", "", m.group(0)))

    # Default is PREDICTED (the safe, non-overclaiming label). Only a page
    # that explicitly renders "STARTING LINEUPS" (confirmed XIs, ~1h before
    # kickoff) flips the status to confirmed; a truncated/absent marker must
    # never be presented as an official lineup.
    body_lower = (data.get("body") or "").lower()
    status = "confirmed" if "starting lineup" in body_lower else "predicted"
    return {
        "status": status,
        "formations": formations,
        "home": home,
        "away": away,
        "home_count": len(home),
        "away_count": len(away),
        "source": "flashscore_lineups",
    }


class FlashscoreBrowserClient:
    """Persistent UC-browser client used to render Flashscore pages."""

    # Resource types whose BYTES are never read by any scraper here: every
    # parser consumes DOM text / element attributes only (team names, scores,
    # tables, lineups). Blocking them at the network layer cuts RAM, CPU and
    # bandwidth per render -- DOM structure is untouched, so scraped output
    # is byte-identical. CSS/JS/XHR are deliberately NOT blocked: the pages
    # are JS-rendered SPAs, and XHR carries the data feeds themselves.
    _BLOCKED_URL_PATTERNS = (
        "*.jpg", "*.jpeg", "*.png", "*.gif", "*.webp", "*.avif",
        "*.svg", "*.ico", "*.bmp",
        "*.woff", "*.woff2", "*.ttf", "*.otf", "*.eot",
        "*.mp4", "*.webm", "*.mp3",
    )

    def __init__(self, throttle_seconds: float = 1.5) -> None:
        self._driver = None
        self._throttle = throttle_seconds
        self.available: bool = True
        self._resource_block_applied = False

    def _ensure_driver(self):
        if self._driver is None:
            from seleniumbase import Driver
            self._driver = Driver(uc=True, headless2=True, browser="chrome")
            self._apply_resource_blocking()
        return self._driver

    def _apply_resource_blocking(self) -> None:
        """Block image/font/media downloads for this session (best-effort).

        Uses CDP Network.setBlockedURLs, which works regardless of the
        selenium wrapper version. Failure must NEVER break rendering: the
        session simply stays unblocked (legacy behaviour).
        """
        if self._resource_block_applied:
            return
        try:
            self._driver.execute_cdp_cmd("Network.enable", {})
            self._driver.execute_cdp_cmd(
                "Network.setBlockedURLs",
                {"urls": list(self._BLOCKED_URL_PATTERNS)},
            )
            self._resource_block_applied = True
            logger.info(
                "flashscore resource blocking ON (%d patterns)",
                len(self._BLOCKED_URL_PATTERNS),
            )
        except Exception as exc:
            logger.warning(
                "flashscore resource blocking unavailable (%s); continuing unblocked",
                type(exc).__name__,
            )
            self._resource_block_applied = True  # don't retry every render

    def close(self) -> None:
        if self._driver is not None:
            driver = self._driver
            self._driver = None

            def _quit() -> None:
                try:
                    driver.quit()
                except Exception:
                    pass

            # seleniumbase UC quit can stall in a timeout-less socket connect
            # (a wedged Chrome renderer); bounding it keeps the runner's
            # finally from hanging past its deadline -- the result must be
            # emitted even if the teardown never finishes. Any orphaned Chrome
            # is swept by the bot's post-run zombie cleanup.
            t = threading.Thread(target=_quit, daemon=True)
            t.start()
            t.join(timeout=8.0)

    # -- page helpers -----------------------------------------------------
    def _open(self, url: str, settle: float = 3.0) -> bool:
        """Navigate to a page and wait for its SPA to render.

        Flashscore does NOT run a Cloudflare interstitial (verified 2026-08:
        plain ``driver.get`` loads every page), so the slow UC reconnect dance
        (used for sofascore) is unnecessary: plain get + short settle is ~3x
        faster (3.1s vs 10.8s measured). The UC driver is still used so the
        session is not fingerprinted as a bot.
        """
        try:
            driver = self._ensure_driver()
            driver.get(url)
            time.sleep(settle)
            return True
        except Exception as exc:
            logger.warning("flashscore open failed %s: %s", url, type(exc).__name__)
            self.available = False
            return False

    def _throttle_sleep(self) -> None:
        if self._throttle:
            time.sleep(self._throttle)

    # -- league/homepage page: match rows -----------------------------------
    def scrape_league_matches(self, league_key: str) -> list[dict[str, Any]] | None:
        """Rows from a league page (or the homepage when league_key is None /
        not registered): {home_name, away_name, home_slug, home_id, away_slug,
        away_id, match_url, date_text}.

        Names are read from the DOM participant elements (home/away), not from
        row innerText: on the homepage a row also contains the live minute and
        score, so positional text parsing mislabels home/away. The URL slug
        order is ALSO not guaranteed to be home-first (verified 2026-08), so
        each name is matched back to its slug and the roles are swapped when
        necessary.
        """
        path = LEAGUE_PATHS.get(league_key)
        url = f"{BASE_URL}{path}/" if path else BASE_URL + "/"
        if not self._open(url, settle=3.0):
            return None
        try:
            driver = self._ensure_driver()
            rows = driver.execute_script(
                """
                const out = [];
                document.querySelectorAll('.event__match').forEach(row => {
                  const link = row.querySelector('a[href*="/match/"]');
                  const href = link ? link.getAttribute('href') : null;
                  const homeEl = row.querySelector(
                    '[class*="event__homeParticipant"], [class*="participant--home"]');
                  const awayEl = row.querySelector(
                    '[class*="event__awayParticipant"], [class*="participant--away"]');
                  const homeName = homeEl ? homeEl.innerText.trim() : null;
                  const awayName = awayEl ? awayEl.innerText.trim() : null;
                  const txt = (row.innerText || '').replace(/\\n+/g, ' | ').trim();
                  const timeEl = row.querySelector('.event__time');
                  const dateText = timeEl ? timeEl.innerText.trim() : null;
                  const scoreHomeEl = row.querySelector('.event__score--home');
                  const scoreAwayEl = row.querySelector('.event__score--away');
                  out.push({href, homeName, awayName, txt, dateText,
                            scoreHome: scoreHomeEl ? scoreHomeEl.innerText.trim() : null,
                            scoreAway: scoreAwayEl ? scoreAwayEl.innerText.trim() : null});
                });
                return out.slice(0, 120);
                """
            )
            if not rows:
                return None
            parsed: list[dict[str, Any]] = []
            for r in rows:
                href = r.get("href") or ""
                m = re.search(
                    r"/match/football/([a-z0-9-]+)-([A-Za-z0-9]{8})/([a-z0-9-]+)-([A-Za-z0-9]{8})/",
                    href,
                )
                if not m:
                    continue
                slug_a, id_a, slug_b, id_b = m.groups()
                home_name = r.get("homeName")
                away_name = r.get("awayName")
                if not (home_name and away_name):
                    # Selector miss (page variant): degrade to positional
                    # innerText parsing instead of dropping the row.
                    parts = [p.strip() for p in (r.get("txt") or "").split("|") if p.strip()]
                    home_name = parts[1] if len(parts) > 1 else slug_a.replace("-", " ").title()
                    away_name = parts[2] if len(parts) > 2 else slug_b.replace("-", " ").title()
                # The URL slug order is not guaranteed home-first; match each
                # participant name back to its slug so roles stay correct.
                home_slug, home_id, away_slug, away_id = _assign_slug_roles(
                    slug_a, id_a, slug_b, id_b, home_name, away_name
                )
                score = None
                sh, sa = r.get("scoreHome"), r.get("scoreAway")
                if sh is not None and sa is not None and sh != "" and sa != "":
                    score = {"home": sh, "away": sa}
                parsed.append({
                    "home_name": home_name,
                    "away_name": away_name,
                    "home_slug": home_slug,
                    "home_id": home_id,
                    "away_slug": away_slug,
                    "away_id": away_id,
                    "match_url": href if href.startswith("http") else f"{BASE_URL}{href}",
                    "date_text": r.get("dateText"),
                    "score": score,
                })
            return parsed or None
        except Exception as exc:
            logger.warning(
                "flashscore league scrape failed: %s: %s",
                type(exc).__name__, str(exc)[:400],
            )
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- homepage: today's matches across ALL competitions (with section) ----
    def scrape_homepage_matches(self) -> list[dict[str, Any]] | None:
        """Today's matches from the homepage, tagged with their competition.

        Flashscore's homepage lists matches from every competition (friendlies,
        qualification rounds, minor leagues) that football-data does NOT cover.
        Competition titles are siblings of the match rows (not ancestors), so
        each row is assigned to the last competition title that appears before
        it in DOM order (verified live: 238/238 rows mapped). Returns the same
        row shape as ``scrape_league_matches`` plus ``competition``.
        """
        url = BASE_URL + "/"
        if not self._open(url, settle=3.5):
            return None
        try:
            driver = self._ensure_driver()
            rows = driver.execute_script(
                """
                const SELECTOR = '[class*="title"], [class*="league__name"], [class*="sportName"], [class*="event__round"]';
                const out = [];
                let lastTitle = null;
                const seenTitles = new Set();
                document.querySelectorAll('.event__match, ' + SELECTOR).forEach(node => {
                  if (node.classList && node.classList.contains('event__match')) {
                    const link = node.querySelector('a[href*="/match/"]');
                    const href = link ? link.getAttribute('href') : null;
                    const homeEl = node.querySelector(
                      '[class*="event__homeParticipant"], [class*="participant--home"]');
                    const awayEl = node.querySelector(
                      '[class*="event__awayParticipant"], [class*="participant--away"]');
                    const timeEl = node.querySelector('.event__time');
                    const scoreHomeEl = node.querySelector('.event__score--home');
                    const scoreAwayEl = node.querySelector('.event__score--away');
                    out.push({
                      href, competition: lastTitle,
                      rowCls: (node.className || '').toString(),
                      homeName: homeEl ? homeEl.innerText.trim() : null,
                      awayName: awayEl ? awayEl.innerText.trim() : null,
                      dateText: timeEl ? timeEl.innerText.trim() : null,
                      statusCls: timeEl ? (timeEl.className || '').toString() : '',
                      scoreHome: scoreHomeEl ? scoreHomeEl.innerText.trim() : null,
                      scoreAway: scoreAwayEl ? scoreAwayEl.innerText.trim() : null,
                      txt: (node.innerText || '').replace(/\\n+/g, ' | ').trim(),
                    });
                    return;
                  }
                  // Only the first match in document order counts as the section
                  // header (avoid repeated text from descendants re-assigning).
                  if (seenTitles.has(node)) return;
                  seenTitles.add(node);
                  const cls = (node.className || '').toString();
                  const txt = (node.innerText || '').trim();
                  // A competition title is a single line of plain text;
                  // reject anything multi-line so paragraphs cannot leak in.
                  // (Escape note: a backslash-n written here as one char would
                  // become a real newline in the compiled JS string and break
                  // the single-quoted string below -- keep it doubled.)
                  if (!txt || txt.length < 2 || txt.length > 70 || txt.includes('\\n')) return;
                  if (/sportName/i.test(cls) && /soccer/i.test(txt)) return;
                  lastTitle = txt;
                });
                return out;
                """
            )
            if not rows:
                return None
            parsed: list[dict[str, Any]] = []
            for r in rows:
                href = r.get("href") or ""
                m = re.search(
                    r"/match/football/([a-z0-9-]+)-([A-Za-z0-9]{8})/([a-z0-9-]+)-([A-Za-z0-9]{8})/",
                    href,
                )
                if not m:
                    continue
                slug_a, id_a, slug_b, id_b = m.groups()
                home_name = r.get("homeName")
                away_name = r.get("awayName")
                if not (home_name and away_name):
                    continue
                home_slug, home_id, away_slug, away_id = _assign_slug_roles(
                    slug_a, id_a, slug_b, id_b, home_name, away_name
                )
                score = None
                sh, sa = r.get("scoreHome"), r.get("scoreAway")
                if sh is not None and sa is not None and sh != "" and sa != "":
                    score = {"home": sh, "away": sa}
                parsed.append({
                    "home_name": home_name,
                    "away_name": away_name,
                    "home_slug": home_slug,
                    "home_id": home_id,
                    "away_slug": away_slug,
                    "away_id": away_id,
                    "match_url": href if href.startswith("http") else f"{BASE_URL}{href}",
                    "date_text": r.get("dateText"),
                    "competition": r.get("competition") or "Other",
                    "status": _row_status(
                        r.get("statusCls"), r.get("rowCls"), r.get("dateText"), r.get("txt")
                    ),
                    "score": score,
                })
            return parsed or None
        except Exception as exc:
            logger.warning(
                "flashscore homepage scrape failed: %s: %s",
                type(exc).__name__, str(exc)[:400],
            )
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- team results page: form rows --------------------------------------
    def scrape_team_results(self, slug: str, team_id: str, limit: int = 10) -> list[dict[str, Any]] | None:
        """Rows from /team/{slug}/{id}/results/: {date, home, away, hg, ag,
        result(W/D/L)} in page order (newest first), plus ``competition``:
        the results page groups a team's matches by competition, each section
        headed by a ``headerLeague__title``-class element (same component the
        fixtures page uses, verified live 2026-08), and every row inherits the
        last section title seen before it. ``competition`` is None when no
        header preceded the row -- callers must treat that as UNKNOWN and keep
        the row (fail-open), never as a friendly."""
        url = f"{BASE_URL}/team/{slug}/{team_id}/results/"
        if not self._open(url, settle=3.0):
            return None
        try:
            driver = self._ensure_driver()
            rows = driver.execute_script(
                f"""
                const out = [];
                let lastCompetition = null;
                document.querySelectorAll('.event__match, [class*="headerLeague"]').forEach(row => {{
                  if (!(row.classList && row.classList.contains('event__match'))) {{
                    const t = (row.innerText || '').trim();
                    if (t) lastCompetition = t;
                    return;
                  }}
                  const link = row.href || (row.querySelector('a') ? row.querySelector('a').href : '') || '';
                  const parts = (row.innerText || '').replace(/\\n+/g, ' | ').trim().split('|').map(s => s.trim());
                  if (parts.length >= 5) {{
                    out.push({{
                      date: parts[0],
                      home: parts[1],
                      away: parts[2],
                      hg: parts[3],
                      ag: parts[4],
                      result: parts[5] || null,
                      competition: lastCompetition,
                      // Per-match link so the xG fallback can open each
                      // finished match's statistics tab (2026-08-17).
                      match_url: link ? ('https://www.flashscore.com' + link.replace(/^https?:\\/\\/www\\.flashscore\\.com/, '')) : null
                    }});
                  }}
                }});
                return out.slice(0, {limit});
                """
            )
            return rows or None
        except Exception as exc:
            logger.warning("flashscore team results scrape failed: %s", type(exc).__name__)
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- team fixtures page -------------------------------------------------
    def scrape_team_fixtures(self, slug: str, team_id: str, limit: int = 12) -> list[dict[str, Any]] | None:
        """Upcoming rows from /team/{slug}/{id}/fixtures/.

        Same row shape as ``scrape_league_matches`` (home/away names + slugs
        + match_url + date_text) plus ``competition``: the fixtures page
        groups a team's matches by competition, each section headed by a
        ``headerLeague__title`` element ("LaLiga", "Super Cup", "Club
        Friendly" -- verified live 2026-08: Barcelona's page renders the
        sections in DOM order), and every row inherits the last section title
        seen before it. Carrying the section lets the caller cross-check the
        fixture's real competition against the requested league even when
        resolution lands on this last-resort path (matches several days away
        that missed the league page and the today-only homepage).
        """
        url = f"{BASE_URL}/team/{slug}/{team_id}/fixtures/"
        if not self._open(url, settle=3.0):
            return None
        try:
            driver = self._ensure_driver()
            rows = driver.execute_script(
                f"""
                const out = [];
                let lastCompetition = null;
                document.querySelectorAll('.event__match, [class*="headerLeague__title"]').forEach(node => {{
                  if (node.classList && node.classList.contains('event__match')) {{
                    const link = node.querySelector('a[href*="/match/"]');
                    const href = link ? link.getAttribute('href') : null;
                    const homeEl = node.querySelector(
                      '[class*="event__homeParticipant"], [class*="participant--home"]');
                    const awayEl = node.querySelector(
                      '[class*="event__awayParticipant"], [class*="participant--away"]');
                    const homeName = homeEl ? homeEl.innerText.trim() : null;
                    const awayName = awayEl ? awayEl.innerText.trim() : null;
                    const timeEl = node.querySelector('.event__time');
                    const dateText = timeEl ? timeEl.innerText.trim() : null;
                    const scoreHomeEl = node.querySelector('.event__score--home');
                    const scoreAwayEl = node.querySelector('.event__score--away');
                    out.push({{
                      href, homeName, awayName,
                      dateText, competition: lastCompetition,
                      scoreHome: scoreHomeEl ? scoreHomeEl.innerText.trim() : null,
                      scoreAway: scoreAwayEl ? scoreAwayEl.innerText.trim() : null,
                    }});
                    return;
                  }}
                  // Competition section header. The title element is a single
                  // line of plain text; reject anything multi-line or long so
                  // non-section nodes cannot leak in. (The includes guard must
                  // stay DOUBLED in the Python source -- see test_homepage_matches.)
                  const txt = (node.innerText || '').trim();
                  if (txt && txt.length > 1 && txt.length <= 70 && !txt.includes('\\n')) {{
                    lastCompetition = txt;
                  }}
                }});
                return out.slice(0, {limit});
                """
            )
            if not rows:
                return None
            parsed: list[dict[str, Any]] = []
            for r in rows:
                href = r.get("href") or ""
                m = re.search(
                    r"/match/football/([a-z0-9-]+)-([A-Za-z0-9]{8})/([a-z0-9-]+)-([A-Za-z0-9]{8})/",
                    href,
                )
                if not m:
                    continue
                slug_a, id_a, slug_b, id_b = m.groups()
                home_name = r.get("homeName")
                away_name = r.get("awayName")
                if not (home_name and away_name):
                    continue
                home_slug, home_id, away_slug, away_id = _assign_slug_roles(
                    slug_a, id_a, slug_b, id_b, home_name, away_name
                )
                score = None
                sh, sa = r.get("scoreHome"), r.get("scoreAway")
                if sh is not None and sa is not None and sh != "" and sa != "":
                    score = {"home": sh, "away": sa}
                parsed.append({
                    "home_name": home_name,
                    "away_name": away_name,
                    "home_slug": home_slug,
                    "home_id": home_id,
                    "away_slug": away_slug,
                    "away_id": away_id,
                    "match_url": href if href.startswith("http") else f"{BASE_URL}{href}",
                    "date_text": r.get("dateText"),
                    "competition": r.get("competition"),
                    "score": score,
                })
            return parsed or None
        except Exception as exc:
            logger.warning("flashscore team fixtures scrape failed: %s", type(exc).__name__)
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- match page: statistics (xG etc) -----------------------------------
    def scrape_match_statistics(self, match_url: str) -> dict[str, Any] | None:
        """Parse [data-testid='wcl-statistics'] blocks from a match page.

        Returns {xg_home, xg_away, possession_home, possession_away,
        shots_home, shots_away, shots_on_target_*, corners_*, fouls_*,
        yellow_*} (only keys present on the page).
        """
        # Match pages embed the statistics section directly (verified: match
        # summary shows xG/possession/shots without an extra tab).
        url = match_url
        # settle 3.5s: the statistics section is the LAST SPA section to
        # render (probe 2026-08-17: at 2.5s the blocks were empty, at 5s they
        # were present). 3.5s is the safe midpoint.
        if not self._open(url, settle=3.5):
            return None
        try:
            driver = self._ensure_driver()
            stats = driver.execute_script(
                """
                const out = {};
                document.querySelectorAll("div[data-testid='wcl-statistics']").forEach(el => {
                  const cat = el.querySelector("div[data-testid='wcl-statistics-category']");
                  if (!cat) return;
                  const name = (cat.innerText || '').trim().toLowerCase();
                  // Fix 2026-08-17: the value lives in a <span data-testid=
                  // 'wcl-scores-simple-text-01'> inside the value div -- the
                  // old "> strong" selector never matched (the DOM shipped a
                  // span, not a strong), so xG/possession/shots were silently
                  // None in every match-stats fetch. Reading the value div's
                  // innerText works for both DOM generations.
                  const vals = Array.from(el.querySelectorAll(
                    "div[data-testid='wcl-statistics-value']"
                  )).map(v => (v.innerText || '').trim());
                  if (vals.length >= 2) {
                    const toF = s => { const n = parseFloat(String(s).replace('%','')); return isNaN(n) ? null : n; };
                    if (name.includes('expected goals') || name === 'xg') {
                      out.xg_home = toF(vals[0]); out.xg_away = toF(vals[1]);
                    } else if (name.includes('ball possession')) {
                      out.possession_home = toF(vals[0]); out.possession_away = toF(vals[1]);
                    } else if (name.includes('total shots')) {
                      out.shots_home = toF(vals[0]); out.shots_away = toF(vals[1]);
                    } else if (name.includes('shots on target')) {
                      out.shots_on_target_home = toF(vals[0]); out.shots_on_target_away = toF(vals[1]);
                    } else if (name.includes('corner kicks')) {
                      out.corners_home = toF(vals[0]); out.corners_away = toF(vals[1]);
                    } else if (name.includes('fouls')) {
                      out.fouls_home = toF(vals[0]); out.fouls_away = toF(vals[1]);
                    } else if (name.includes('yellow cards')) {
                      out.yellow_home = toF(vals[0]); out.yellow_away = toF(vals[1]);
                    }
                  }
                });
                return out;
                """
            )
            if not stats or not any(k.startswith(("xg_", "possession_", "shots_", "corners_", "fouls_", "yellow_")) for k in stats):
                return None
            stats["source"] = "flashscore"
            return stats
        except Exception as exc:
            logger.warning("flashscore match stats scrape failed: %s", type(exc).__name__)
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- match page: lineups (predicted pre-match / confirmed) --------------
    def scrape_match_lineups(self, match_url: str) -> dict[str, Any] | None:
        """Starting XIs + formations from the Lineups tab of a match page.

        Pre-match the tab shows PREDICTED lineups for major competitions
        (algorithm-driven, updated with injuries/suspensions); from ~1h
        before kickoff it shows the CONFIRMED starting XIs. The parser is
        ``parse_lineups_page`` (pure). Returns None when no players render
        (lineups not announced / page without a lineups tab).
        """
        base = match_url.split("?")[0].rstrip("/")
        mid = match_url.split("?", 1)[1] if "?" in match_url else ""
        url = f"{base}/summary/lineups/" + (f"?{mid}" if mid else "")
        if not self._open(url, settle=2.5):
            return None
        try:
            driver = self._ensure_driver()
            # The lineups tab renders async (React SPA); a blind settle can
            # grab an empty DOM on a cold page (verified live 2026-08-16:
            # first load at 2.5s -> 0 players, warm load -> 11/11). Poll for
            # the first participant element instead of sleeping blindly.
            deadline = time.time() + 6.0
            while time.time() < deadline:
                if driver.execute_script(
                    "return !!document.querySelector('.lf__formation [data-testid=\"wcl-lineupsParticipantName\"]');"
                ):
                    break
                time.sleep(0.5)
            data = driver.execute_script(
                """
                function grab(sel) {
                  return Array.from(document.querySelectorAll(sel))
                    .map(e => (e.innerText || '').trim()).filter(Boolean);
                }
                const bodyTxt = (document.body.innerText || '');
                return {
                  // Home XI container is ``lf__formation`` (the away one adds
                  // ``lf__formationAway``). The old selector also required
                  // ``lf__formation--extended``, but Flashscore dropped that
                  // modifier on some page variants (verified live 2026-08-16:
                  // Espanyol-Levante rendered home with a bare ``lf__formation``
                  // and the home XI silently came back empty) -- select
                  // structurally (first formation block, away excluded) instead.
                  homePlayers: grab('.lf__formation:not(.lf__formationAway) [data-testid="wcl-lineupsParticipantName"]'),
                  awayPlayers: grab('.lf__formation.lf__formationAway [data-testid="wcl-lineupsParticipantName"]'),
                  headers: grab('.lf__formationHeader'),
                  body: bodyTxt.slice(0, 3000),
                };
                """
            )
            return parse_lineups_page(data or {})
        except Exception as exc:
            logger.warning("flashscore lineups scrape failed: %s", type(exc).__name__)
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- match page: H2H ----------------------------------------------------
    def scrape_match_h2h(self, match_url: str, team_a_name: str, team_b_name: str) -> dict[str, Any] | None:
        """Parse the h2h tab: aggregate recent meetings + last matches.

        Returns {wins, draws, losses, source} computed from the 'H2H' section
        of the h2h page (direct meetings), or None when unavailable.

        F5 (2026-08-17): the old ``[class*="h2h"]`` selector matches nothing
        on the current layout (verified live: the section renders with a
        plain text header, no h2h class anywhere), so the h2h silently
        returned None for every match. The scrape now grabs the section by
        its rendered header text ("HEAD-TO-HEAD MATCHES") and the pure
        ``_parse_h2h_section`` parser decodes its rows.
        """
        # Keep the ?mid= query param: the h2h tab renders from it.
        base = match_url.split("?")[0].rstrip("/")
        mid = match_url.split("?", 1)[1] if "?" in match_url else ""
        url = f"{base}/h2h/" + (f"?{mid}" if mid else "")
        if not self._open(url, settle=2.5):
            return None
        try:
            driver = self._ensure_driver()
            section = driver.execute_script(
                """
                const text = document.body.innerText || '';
                const start = text.toUpperCase().indexOf('HEAD-TO-HEAD MATCHES');
                if (start < 0) return '';
                // The section ends at the next pinned-leagues / footer block.
                let end = text.length;
                for (const marker of ['PINNED LEAGUES', 'SHOW MORE MATCHES', 'FOLLOW ']) {
                  const i = text.toUpperCase().indexOf(marker, start + 30);
                  if (i > start && i < end) end = i;
                }
                return text.slice(start, end).slice(0, 4000);
                """
            )
            if not section or not section.strip():
                return None
            out = _parse_h2h_section(section, team_a_name, team_b_name)
            return out
        except Exception as exc:
            logger.warning("flashscore h2h scrape failed: %s", type(exc).__name__)
            self.available = False
            return None
        finally:
            self._throttle_sleep()

    # -- league page: standings table (context-only) ------------------------
    def scrape_league_standings(self, league_key: str) -> dict[str, Any] | None:
        """Overall league table from /football/{region}/{league}/standings/.

        Returns {tables: {overall: [rows]}, source} where each row is
        {pos, team, mp, w, d, l, gf, ga, gd, pts, form} (see
        ``parse_standings_rows``). Context-only input for the analysis: league
        position/points are league-recency signal, never a model feature.
        Fails soft: a parse error here must NOT disable flashscore for the
        rest of the command (unlike core scrape methods) -- the caller only
        loses an optional context line.
        """
        path = LEAGUE_PATHS.get(league_key)
        if not path:
            return None
        url = f"{BASE_URL}{path}/standings/"
        if not self._open(url, settle=4.5):
            return None
        try:
            driver = self._ensure_driver()
            # The table renders asynchronously a beat after the settle window
            # on a reused driver (same flakiness class as match info), so retry
            # once after a short wait when no rows came back.
            for _attempt in (1, 2):
                rows = driver.execute_script(
                    """
                    const out = [];
                    document.querySelectorAll('.ui-table__body .ui-table__row').forEach(row => {
                      const rankEl = row.querySelector('.tableCellRank');
                      const nameEl = row.querySelector('.tableCellParticipant__name');
                      const cells = Array.from(row.querySelectorAll('.table__cell--value'))
                        .map(e => (e.innerText || '').trim());
                      const formEl = row.querySelector('[class*="tableCellForm"]');
                      const ptsEl = row.querySelector('.table__cell--points');
                      out.push({
                        rank: rankEl ? rankEl.innerText.trim() : null,
                        team: nameEl ? nameEl.innerText.trim() : null,
                        cells: cells,
                        form: formEl ? formEl.innerText.trim() : null,
                        pts: ptsEl ? ptsEl.innerText.trim() : null,
                      });
                    });
                    return out;
                    """
                )
                parsed = parse_standings_rows(rows)
                if parsed:
                    return {"tables": {"overall": parsed}, "source": "flashscore_standings"}
                time.sleep(1.5)
            return None
        except Exception as exc:
            logger.warning("flashscore standings scrape failed: %s", type(exc).__name__)
            return None
        finally:
            self._throttle_sleep()

    # -- match page: match information (referee / venue / capacity) ---------
    def scrape_match_info(self, match_url: str) -> dict[str, Any] | None:
        """Referee, venue, capacity + neutral-location flag from a match page.

        Returns {referee, referee_country, venue, town, capacity, neutral,
        source} or None (see ``parse_match_info``). Context-only: the neutral
        flag tells the caller a "home" team is NOT playing at home (final /
        cup tie on a neutral pitch), which the model's home-advantage
        assumption does not know about. Fails soft like standings.
        """
        if not self._open(match_url, settle=4.0):
            return None
        try:
            driver = self._ensure_driver()
            # The SPA sometimes renders the match-information block a beat later
            # than the settle window (especially after several navigations on
            # the same driver), so the JS is fully defensive (try/catch, null
            # guards) and the scrape is retried once after a short wait.
            for _attempt in (1, 2):
                data = driver.execute_script(
                    """
                    const out = {labels: [], neutral: false};
                    try {
                      const info = document.querySelector('[data-testid="wcl-summaryMatchInformation"]');
                      if (info) {
                        // Each label lives in its own .infoLabelWrapper; the VALUE is
                        // the NEXT sibling .infoValue element. innerText is uppercase
                        // (CSS text-transform) so labels read 'REFEREE:'/'VENUE:'.
                        const items = Array.from(info.children);
                        for (let i = 0; i < items.length; i++) {
                          const cls = (items[i].className || '').toString();
                          if (!cls.includes('infoLabelWrapper')) continue;
                          let label = (items[i].innerText || '').trim();
                          let value = '';
                          const nxt = items[i + 1];
                          if (nxt && ((nxt.className || '').toString()).includes('infoValue')) {
                            value = (nxt.innerText || '').trim();
                          }
                          if (label || value) out.labels.push(label + (value ? '\\n' + value : ''));
                        }
                      }
                      const body = document.body;
                      if (body && (body.innerText || '').includes('Neutral location')) out.neutral = true;
                    } catch (e) {
                      out.error = String(e && e.message || e);
                    }
                    return out;
                    """
                )
                parsed = parse_match_info(data or {})
                if parsed:
                    parsed["source"] = "flashscore_match_info"
                    return parsed
                time.sleep(1.5)
            return None
        except Exception as exc:
            logger.warning(
                "flashscore match info scrape failed: %s: %s",
                type(exc).__name__, str(exc)[:300],
            )
            return None
        finally:
            self._throttle_sleep()


# ---- standings + match info (pure parsers, context-only) ------------------


def parse_standings_rows(data: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Pure parser for standings table rows (testable).

    ``data`` is the raw scrape: [{rank, team, cells, pts, form}] where cells
    are the ``table__cell--value`` texts in order MP, W, D, L, G ("0:0"),
    GD, PTS. Returns [{pos, team, mp, w, d, l, gf, ga, gd, pts, form}] or
    None when no usable rows.
    """

    def _int(v: Any) -> int | None:
        try:
            return int(str(v).strip().replace(" ", ""))
        except (TypeError, ValueError):
            return None

    rows: list[dict[str, Any]] = []
    for r in data or []:
        team = (r.get("team") or "").strip()
        if not team:
            continue
        cells = [str(c).strip() for c in (r.get("cells") or [])]
        pos = str(r.get("rank") or "").strip().rstrip(".")
        mp = _int(cells[0]) if len(cells) > 0 else None
        w = _int(cells[1]) if len(cells) > 1 else None
        d = _int(cells[2]) if len(cells) > 2 else None
        l = _int(cells[3]) if len(cells) > 3 else None
        gf = ga = None
        if len(cells) > 4 and ":" in cells[4]:
            parts = cells[4].split(":")
            gf, ga = _int(parts[0]), _int(parts[1])
        gd = _int(cells[5]) if len(cells) > 5 else None
        pts = _int(r.get("pts")) if r.get("pts") not in (None, "") else (_int(cells[6]) if len(cells) > 6 else None)
        rows.append({
            "pos": _int(pos) if pos else None,
            "team": team,
            "mp": mp, "w": w, "d": d, "l": l,
            "gf": gf, "ga": ga, "gd": gd, "pts": pts,
            "form": ((r.get("form") or "").strip() or None) if (r.get("form") or "").strip() not in ("?", "-") else None,
        })
    return rows or None


def parse_match_info(data: dict[str, Any]) -> dict[str, Any] | None:
    """Pure parser for the match-information block (testable).

    ``data`` is the raw scrape: {labels: [...], neutral: bool}. Labels look
    like "REFEREE:\nChiffi D.\n(Ita)" / "VENUE:\nCentralnyj Stadion\n(Kostanay)"
    / "CAPACITY:\n10 500". Returns {referee, referee_country, venue, town,
    capacity, neutral} or None.
    """
    out: dict[str, Any] = {}
    for label in data.get("labels") or []:
        t = str(label).strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith("referee"):
            rest = re.sub(r"^referee\s*:\s*", "", t, flags=re.I).strip()
            parts = rest.split()
            # Multi-word names ("Chiffi D.") stay whole; only the country
            # parenthetical is split out.
            country = next((p for p in parts if p.startswith("(") and p.endswith(")")), None)
            name_parts = [p for p in parts if p is not country]
            if name_parts:
                out["referee"] = " ".join(name_parts)
            if country:
                out["referee_country"] = country.strip("()")
        elif low.startswith("venue"):
            rest = re.sub(r"^venue\s*:\s*", "", t, flags=re.I).strip()
            m = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", rest)
            out["venue"] = (m.group(1) if m else rest).strip()
            if m:
                out["town"] = m.group(2).strip()
        elif low.startswith("capacity"):
            out["capacity"] = re.sub(r"^capacity\s*:\s*", "", t, flags=re.I).strip()
    if data.get("neutral"):
        out["neutral"] = True
    return out or None


# ---- team slug resolution (fails soft: None on any failure) ----------------

# PRIMARY (2026-08): the livesport Sphinx search backend behind the
# flashscore SPA search box -- s.livesport.services/api/v2/search/. The SPA's
# old suggest host (suggest.flashscore.com, below) is DNS-blocked on this
# network (getaddrinfo fails), which silently killed flashscore by-name team
# resolution and left the pipeline on football-data's thin 1-match form.
# This endpoint answers plain GETs with the standard flashscore Origin/
# Referer headers (no session, cookies or signature). Verified live: returns
# {id (8-char flashscore team id), url (slug), name, type, sport, ...}
# entries, e.g. "SK Beveren" -> id QaqfE8WE / slug sk-beveren.
_LIVESPORT_SEARCH_URL = "https://s.livesport.services/api/v2/search/"

# Legacy fallback: the search-suggest endpoint is not an official API; the
# URL file name (language code) has varied over time, so a couple of
# candidates are tried. Kept for networks where the livesport host is
# unreachable but this one answers.
_SUGGEST_URLS = (
    "https://suggest.flashscore.com/suggest/1.php",
    "https://suggest.flashscore.com/suggest/en.php",
)


def _flatten_suggest_entries(data: Any) -> list[tuple[str, str]]:
    """Yield (name, url) pairs from the suggest JSON whatever its nesting."""
    out: list[tuple[str, str]] = []
    if isinstance(data, list):
        for item in data:
            out.extend(_flatten_suggest_entries(item))
    elif isinstance(data, dict):
        url = str(data.get("url") or data.get("href") or "")
        name = str(data.get("value") or data.get("name") or data.get("label") or "")
        if url and name:
            out.append((name, url))
        for key in ("data", "items", "entries", "results"):
            if key in data:
                out.extend(_flatten_suggest_entries(data[key]))
    return out


def _pick_suggest_team(text: str, sq: str | list[str]) -> tuple[str, str] | None:
    """Best (slug, id) from a suggest response whose name matches ``sq``.

    ``sq`` accepts the query's squashed variants (raw + team-alias canonical),
    so "Hearts" still resolves the "Heart of Midlothian" suggest entry.
    """
    import json

    sqs = [s for s in (sq if isinstance(sq, list) else [sq]) if s]
    url_re = re.compile(r"/team/([a-z0-9-]+)/([A-Za-z0-9]{8})")

    def _matches(s: str) -> bool:
        if not s:
            return False
        if not sqs:
            return True
        return any(sq == s or sq in s or s in sq for sq in sqs)

    entries: list[tuple[str, str]] = []
    try:
        entries = _flatten_suggest_entries(json.loads(text))
    except (ValueError, TypeError):
        entries = []
    best: tuple[str, str] | None = None
    for name, url in entries:
        s = _squash(_norm_name(name))
        if not _matches(s):
            continue
        m = url_re.search(url)
        if m and (best is None or len(m.group(1)) > len(best[0])):
            best = (m.group(1), m.group(2))
    if best:
        return best
    # Unknown JSON shape: fall back to the first slug pattern whose slug
    # still agrees with the query token.
    for m in url_re.finditer(text):
        slug, tid = m.groups()
        s = _squash(slug.replace("-", " "))
        if _matches(s):
            return (slug, tid)
    return None


def _pick_sphinx_team(data: Any, sq: list[str]) -> tuple[str, str] | None:
    """Best (slug, id) from the livesport search JSON for a TEAM entry.

    Only Soccer Team entries count (search also returns players, leagues,
    women/youth teams -- filtered out). Matching honors the API's own
    relevance order: an EXACT name match (e.g. query "Beveren" -> "Beveren",
    not "Bosdam Beveren") wins outright; otherwise the FIRST containment
    match in API order wins. A longest-slug tiebreak is deliberately NOT used
    -- it picked lower-league teams whose slug merely contains the query.
    ``sq`` is the squashed query variants (raw + canonical alias), same
    contract as ``_pick_suggest_team``.
    """
    items = data if isinstance(data, list) else (data or {}).get("results") or []
    exact: tuple[str, str] | None = None
    contain: tuple[str, str] | None = None
    for it in items:
        if not isinstance(it, dict):
            continue
        typ = it.get("type") or {}
        if str(typ.get("name") or "").lower() != "team":
            continue
        sport = (it.get("sport") or {}).get("name") or ""
        if sport and str(sport).lower() != "soccer":
            continue
        name = str(it.get("name") or "")
        slug = str(it.get("url") or "")
        tid = str(it.get("id") or "")
        if not name or not slug or not tid:
            continue
        s = _squash(_norm_name(name))
        if not s or (sq and not any(sq in s or s in sq for sq in sq)):
            # Alias bridge: the API often names a team by its SHORT form
            # ("Hearts") while the query is the canonical name
            # ("Heart of Midlothian FC"). Resolve the entry through the
            # team-alias table and retry the match on the canonical form.
            aliased = None
            try:
                from .team_alias import resolve_team_alias

                _a = resolve_team_alias(name, None)
                if _a:
                    aliased = _squash(_norm_name(_a))
            except Exception:  # noqa: BLE001 -- alias is best-effort
                aliased = None
            if not aliased or (sq and not any(sq in aliased or aliased in sq for sq in sq)):
                continue
            s = aliased
        if any(s == sq for sq in sq):
            if exact is None:
                exact = (slug, tid)
        elif contain is None:
            contain = (slug, tid)
    return exact or contain


def _suggest_team(query: str) -> tuple[str, str] | None:
    """Find (slug, id) for a team via the flashscore search APIs.

    Cheap pure-HTTP call (no browser render): tries the livesport Sphinx
    search endpoint first (works on this network), then the legacy suggest
    hosts. Returns None on any failure so the caller's resolve behaviour is
    unchanged.
    """
    from urllib.parse import quote

    q = quote((query or "").strip())
    variants = _squash_variants(query)
    if not q or not variants:
        return None
    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Origin": "https://www.flashscore.com",
        "Referer": "https://www.flashscore.com/",
    }
    try:
        resp = httpx.get(
            _LIVESPORT_SEARCH_URL,
            params={"q": (query or "").strip(), "sport": "soccer"},
            headers=headers,
            timeout=8.0,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            cand = _pick_sphinx_team(resp.json(), variants)
            if cand:
                return cand
    except Exception as exc:
        logger.warning("flashscore livesport search failed (%s): %s", type(exc).__name__, exc)
    for base in _SUGGEST_URLS:
        try:
            resp = httpx.get(f"{base}?term={q}&count=10", timeout=8.0, follow_redirects=True)
            if resp.status_code != 200:
                continue
            cand = _pick_suggest_team(resp.text, variants)
            if cand:
                return cand
        except Exception as exc:
            logger.warning("flashscore suggest request failed (%s): %s", base, type(exc).__name__)
            continue
    return None


class FlashscoreClient:
    """Async wrapper around the browser client (lock + bounded waits)."""

    def __init__(self, throttle_seconds: float = 1.5) -> None:
        self._browser = FlashscoreBrowserClient(throttle_seconds=throttle_seconds)
        self._browser_lock = asyncio.Lock()
        self.available: bool = True
        self.quota_warning: bool = False

    def close(self) -> None:
        self._browser.close()

    async def _run(self, fn, *args, timeout: float = 45.0, disable_on_timeout: bool = True):
        if not self._browser.available or not self.available:
            return None
        acquired = False
        try:
            await asyncio.wait_for(self._browser_lock.acquire(), timeout=10.0)
            acquired = True
        except asyncio.TimeoutError:
            logger.warning("flashscore browser busy; skipped")
            return None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn, *args), timeout=timeout
            )
        except asyncio.TimeoutError:
            if disable_on_timeout:
                logger.warning("flashscore browser call timed out (%.0fs); disabling", timeout)
                self.available = False
            else:
                logger.warning("flashscore browser call timed out (%.0fs); not disabling", timeout)
            return None
        except Exception as exc:
            logger.warning("flashscore browser call error: %s; disabling", type(exc).__name__)
            self.available = False
            return None
        finally:
            self._browser_lock.release()

    async def resolve_match(
        self, league_key: str, home_kw: str, away_kw: str
    ) -> dict[str, Any] | None:
        """Find the fixture home vs away, with a homepage + team-fixtures fallback.

        Primary source: the league page (or the homepage directly when the
        league has no registered path). If the fixture is not found there,
        fall back to the homepage -- today's matches across all leagues -- so
        a fixture that exists today but did not render on its league page can
        still be resolved (e.g. cup ties, promotion play-offs, or a league
        page that only shows a subset of the day's fixtures). Last resort:
        resolve each team's slug via the suggest endpoint and scrape their
        team-fixtures pages for the pair (~15-30s, budget-guarded).

        Names are matched through team-alias variants first ("Hearts" ->
        "Heart of Midlothian"), so a short nickname matches the canonical
        fixture name without an extra render.

        Returns {home: {slug,id,name}, away: {slug,id,name}, match_url} or
        None when the fixture cannot be found on any page.
        """
        home_variants = _squash_variants(home_kw)
        away_variants = _squash_variants(away_kw)
        if not (home_variants and away_variants):
            return None
        # 1) league page (scrape_league_matches(None) already scrapes the
        #    homepage, so unregistered leagues get their fallback here).
        #    Timeouts must NOT disable flashscore: the first render includes
        #    the cold Chrome launch (10s+) on a slow SOCKS proxy, and a valid
        #    render must never downgrade form/h2h/stats for the rest of the
        #    command. The cap only bounds the "match is not today" case;
        #    the homepage fallback below carries the tight cap and is what
        #    disables flashscore when it is genuinely dead.
        rows = await self._run(self._browser.scrape_league_matches, league_key,
                               timeout=28.0, disable_on_timeout=False)
        found = _find_pair_in_rows(rows, home_variants, away_variants) if rows else None
        if found:
            return found
        # 2) homepage fallback -- only when the league has its own page and
        #    the fixture did not show up there. A tighter cap keeps a match
        #    that is NOT on today's page (e.g. a fixture weeks away) from
        #    burning the runner's 85s deadline; if BOTH renders time out the
        #    source is genuinely slow/dead and gets disabled so form/h2h
        #    skip straight to the fast fallbacks. Uses the competition-aware
        #    homepage scrape so rows carry their real competition section
        #    (verified 2026-08-16: Las Palmas-Albacete resolves via this
        #    fallback with competition="LaLiga2", NOT the requested league's
        #    label -- carrying it fixes the false "laliga" vs "laliga2"
        #    cross-source discrepancy).
        if league_key in LEAGUE_PATHS:
            home_rows = await self._run(self._browser.scrape_homepage_matches, timeout=15.0)
            found = _find_pair_in_rows(home_rows, home_variants, away_variants) if home_rows else None
            if found:
                return found
        # 3) team-fixtures fallback (suggest API + up to two team renders).
        return await self._resolve_via_team_fixtures(
            home_kw, away_kw, home_variants, away_variants
        )

    async def _resolve_via_team_fixtures(
        self,
        home_kw: str,
        away_kw: str,
        home_variants: list[str],
        away_variants: list[str],
    ) -> dict[str, Any] | None:
        """Fallback 3: suggest API -> each team's fixtures page -> find pair.

        The suggest endpoint is unofficial and the extra renders cost ~15-30s,
        so the whole path is silent-fail and budget-guarded: when it fails,
        resolve behaves exactly as before (None -> provider chain).
        """
        try:
            from .multi_source import analysis_remaining

            rem = analysis_remaining()
            if rem is not None and rem < 40.0:
                logger.info("flashscore team-fixtures fallback skipped: budget nearly spent")
                return None
        except Exception:
            pass
        try:
            slug_home, id_home = await asyncio.to_thread(_suggest_team, home_kw)
            slug_away, id_away = await asyncio.to_thread(_suggest_team, away_kw)
        except Exception as exc:
            logger.warning("flashscore suggest lookup failed (resolve unchanged): %s", exc)
            return None
        if not (slug_home and id_home and slug_away and id_away):
            return None
        # The pair appears in either team's upcoming fixtures; scrape home
        # first (the opponent is always in its own fixture list).
        for slug, team_id in ((slug_home, id_home), (slug_away, id_away)):
            rows = await self._run(
                self._browser.scrape_team_fixtures, slug, team_id, 20,
                timeout=30.0, disable_on_timeout=False,
            )
            found = _find_pair_in_rows(rows, home_variants, away_variants) if rows else None
            if found:
                # F3 (2026-08-17, wrong-team bug): a team-fixtures page can
                # carry rows from MULTIPLE seasons (the scraper walks the
                # section headers but not the dates). A finished historical
                # "Real Madrid vs Malaga" (from a prior season) must never
                # resolve as the upcoming "Atletico Madrid vs Malaga" the
                # user asked about -- the pipeline would analyze the wrong
                # match. Prefer the first row WITHOUT a score (scheduled /
                # live only); a scored row is a finished match and is only
                # accepted when NO scheduled row exists at all.
                if not found.get("score"):
                    return found
                for r2 in rows or []:
                    f2 = _find_pair_in_rows([r2], home_variants, away_variants)
                    if f2 and not f2.get("score"):
                        return f2
                # Only finished rows matched -> keep the first (best-effort).
                return found
        return None

    async def fetch_homepage_matches(self) -> list[dict[str, Any]] | None:
        """Today's homepage rows (all competitions) with section tags."""
        return await self._run(self._browser.scrape_homepage_matches, timeout=25.0, disable_on_timeout=False)

    async def fetch_team_results(self, slug: str, team_id: str, limit: int = 10) -> list[dict[str, Any]] | None:
        """Raw team-results rows (date/home/away/hg/ag/result/match_url)."""
        return await self._run(self._browser.scrape_team_results, slug, team_id, limit)

    async def fetch_team_form(self, slug: str, team_id: str, limit: int = 5) -> dict[str, Any] | None:
        """Form from the team results page: normalized like football_data.

        P3-2 parity (2026-08-22): pre-season/friendly sections are dropped
        BEFORE the W/D-L aggregates -- early-season windows are otherwise
        dominated by friendlies and inflate lambda's attack/defense inputs
        (Fortuna Sittard v AZ audit: lambda_total 3.96 vs market ~3.2). The
        scraper over-fetches (``limit * 3`` rows) so ``limit`` COMPETITIVE
        games survive the filter; rows without a section header carry
        ``competition: None`` and are kept (fail-open, never guessed away).
        """
        from .nowgoal import is_friendly_competition

        rows = await self._run(
            self._browser.scrape_team_results, slug, team_id, max(limit * 3, 12)
        )
        if not rows:
            return None
        results: list[str] = []
        gf_list: list[int] = []
        ga_list: list[int] = []
        home_w = home_d = home_l = 0
        away_w = away_d = away_l = 0
        team_sq = _squash(_norm_name(slug.replace("-", " ")))
        for r in rows:
            if len(results) >= limit:
                break
            if is_friendly_competition(str(r.get("competition") or "")):
                continue
            try:
                hg, ag = int(r["hg"]), int(r["ag"])
            except (ValueError, TypeError):
                continue
            # The team is always one of the two sides on its own results page.
            # If the display name diverges from the slug (e.g. "Paris SG" vs
            # slug "paris-saint-germain") we must NOT silently assume the
            # away side -- that would flip gf/ga and the W/D/L direction.
            row_home_sq = _squash(_norm_name(r.get("home") or ""))
            row_away_sq = _squash(_norm_name(r.get("away") or ""))
            if team_sq == row_home_sq:
                is_home = True
            elif team_sq == row_away_sq:
                is_home = False
            else:
                continue
            gf = hg if is_home else ag
            ga = ag if is_home else hg
            gf_list.append(gf)
            ga_list.append(ga)
            if hg == ag:
                results.append("D")
                if is_home:
                    home_d += 1
                else:
                    away_d += 1
            elif (is_home and hg > ag) or (not is_home and ag > hg):
                results.append("W")
                if is_home:
                    home_w += 1
                else:
                    away_w += 1
            else:
                results.append("L")
                if is_home:
                    home_l += 1
                else:
                    away_l += 1
        if not results:
            return None
        # page order is newest-first -> reverse to oldest->newest
        recent_goals = list(reversed(list(zip(gf_list, ga_list))))
        return {
            "sequence": "-".join(results),
            "gf_avg": sum(gf_list) / len(gf_list),
            "ga_avg": sum(ga_list) / len(ga_list),
            "home": {"w": home_w, "d": home_d, "l": home_l},
            "away": {"w": away_w, "d": away_d, "l": away_l},
            "sample_size": len(results),
            "recent_goals": recent_goals,
            "source": "flashscore",
        }

    async def fetch_team_fixtures(self, slug: str, team_id: str, limit: int = 12) -> list[dict[str, Any]] | None:
        """Upcoming fixture rows (league-row shape incl. match_url)."""
        return await self._run(
            self._browser.scrape_team_fixtures, slug, team_id, limit,
            timeout=30.0, disable_on_timeout=False,
        )

    async def fetch_match_statistics(self, match_url: str) -> dict[str, Any] | None:
        return await self._run(self._browser.scrape_match_statistics, match_url)

    async def fetch_match_h2h(self, match_url: str, team_a_name: str, team_b_name: str) -> dict[str, Any] | None:
        return await self._run(self._browser.scrape_match_h2h, match_url, team_a_name, team_b_name)

    async def fetch_match_lineups(self, match_url: str) -> dict[str, Any] | None:
        # Lineups are context info only; a slow/absent tab must NEVER disable
        # flashscore for the rest of the command (same guard resolve_match
        # uses for its primary render).
        return await self._run(
            self._browser.scrape_match_lineups, match_url,
            timeout=25.0, disable_on_timeout=False,
        )

    async def fetch_league_standings(self, league_key: str) -> dict[str, Any] | None:
        """Overall league table (context-only; never disables on failure)."""
        return await self._run(
            self._browser.scrape_league_standings, league_key,
            timeout=25.0, disable_on_timeout=False,
        )

    async def fetch_match_info(self, match_url: str) -> dict[str, Any] | None:
        """Referee/venue/capacity/neutral flag (context-only; fails soft)."""
        return await self._run(
            self._browser.scrape_match_info, match_url,
            timeout=25.0, disable_on_timeout=False,
        )
