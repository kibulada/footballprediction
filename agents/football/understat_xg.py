"""Understat per-match xG downloader via a real Chrome session.

understat.com serves its league match data to a real Chrome session as
page-global JavaScript variables (``window.datesData`` = per-match rows
with team names, score and xG for both sides). Plain HTTP clients
(requests / curl_cffi even with Chrome impersonation) are actively 404'd
by the server -- verified 2026-08: the data renders only in a real
browser. We reuse the project's seleniumbase UC driver pattern (proven
for the Sofascore fallback) to open each season page and read
``window.datesData`` directly.

Rows are emitted as ``{date, home, away, home_xg, away_xg, season,
league}`` so they can be joined onto the baseline fixture caches by
(date, home, away) -- apples-to-apples, no fabricated xG (MASTER PROMPT
PHASE 9).

Usage::

    python -m agents.football.understat_xg --league EPL \
        --seasons 2022,2023,2024,2025 \
        --out cache/football/understat_xg_rows.json
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from .odds_history import _num, normalize_team

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# Bot league key -> understat league slug used in the URL path.
UNDERSTAT_LEAGUE: dict[str, str] = {
    "EPL": "EPL",
    "LaLiga": "La Liga",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
}

# understat title -> football-data.co.uk RAW team name. ``normalize_team``
# then maps the raw name to the canonical name used by the frozen baseline
# cache (e.g. 'Man United' -> 'Manchester Utd'), keeping the join consistent.
TEAM_RAW_MAP: dict[str, str] = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolverhampton",
    "Leeds": "Leeds",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Newcastle United": "Newcastle",
    "Sheffield United": "Sheffield United",
    "Sunderland": "Sunderland",
    "West Bromwich Albion": "West Brom",
    "Norwich City": "Norwich City",
}

SEASON_LABEL: dict[int, str] = {
    2022: "2022-2023", 2023: "2023-2024",
    2024: "2024-2025", 2025: "2025-2026", 2026: "2026-2027",
}


def _canonical_team(title: str) -> str:
    """understat title -> canonical baseline-cache team name."""
    raw = TEAM_RAW_MAP.get(title, title)
    return normalize_team(raw)


def _tolerant_key(name: str) -> str:
    """Lowercase alphanumeric-only key for tolerant team-name matching."""
    import re

    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _same_team(key: str, other: str) -> bool:
    """Tolerant team equality: exact, or the longer key contains the shorter
    ("Arsenal FC" vs "Arsenal", "Manchester United" vs "Man United").
    Short tokens are NOT substring-matched to avoid false positives
    ("real" matching "Real Madrid")."""
    if not key or not other:
        return False
    if key == other:
        return True
    short, long = (key, other) if len(key) <= len(other) else (other, key)
    return len(short) >= 5 and short in long


def team_xg_history_from_rows(
    rows: list[dict[str, Any]],
    team_name: str,
    limit: int = 5,
    exclude: tuple[str, str, str] | None = None,
) -> dict[str, Any] | None:
    """Rolling PRE-MATCH xG averages for a team from understat rows.

    Only finished matches count (rows already filtered by ``dates_to_rows``
    via ``isResult``). Matches are sorted chronologically and the last
    ``limit`` finished matches involving the team are used, so the feature is
    exactly what the model saw pre-match -- the same construction as the
    backtest ``_build_xg_features`` (window 5).

    ``exclude=(home, away, date)`` drops one specific fixture (the match
    being predicted) so a just-finished fixture cannot leak its own stats
    into the team history (anti-leakage, mirrors sofascore
    ``exclude_event_id``).

    Returns {xg_for_avg, xg_against_avg, sample_size} or None.
    """
    # F1 (2026-08-17): the ROW side is canonicalized via ``_canonical_team``
    # (TEAM_RAW_MAP + TEAM_NAME_MAP), so the QUERY side must go through the
    # SAME pipeline before ``_tolerant_key``. Previously the query used the
    # raw flashscore name ("Manchester United" -> "manchesterunited") while
    # the row carried the canonical name ("Manchester Utd" ->
    # "manchesterutd") -- ``_same_team`` (substring, min 5 chars) then
    # FAILED for every EPL team whose canonical name differs from the
    # flashscore spelling, so the live xG history silently came back None
    # for exactly the teams the backtest used it for.
    key = _tolerant_key(_canonical_team(team_name))
    if not key:
        return None
    ex_home, ex_away, ex_date = exclude or (None, None, None)
    # Same canonicalization for the anti-leakage exclude: a just-finished
    # predicted fixture must be matched canonically too, or the exclude
    # misses the row and the match leaks its own xG into the history.
    ex_home_key = _tolerant_key(_canonical_team(ex_home)) if ex_home else None
    ex_away_key = _tolerant_key(_canonical_team(ex_away)) if ex_away else None
    matches: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x.get("date") or ""):
        if r.get("home_xg") is None or r.get("away_xg") is None:
            continue
        h_key = _tolerant_key(r.get("home"))
        a_key = _tolerant_key(r.get("away"))
        if not (_same_team(key, h_key) or _same_team(key, a_key)):
            continue
        if ex_date and r.get("date") == ex_date:
            if (
                ex_home_key and ex_away_key
                and (_same_team(ex_home_key, h_key) or _same_team(ex_home_key, a_key))
                and (_same_team(ex_away_key, h_key) or _same_team(ex_away_key, a_key))
            ):
                continue
        matches.append(r)
    matches = matches[-limit:]
    if not matches:
        return None
    xg_for: list[float] = []
    xg_against: list[float] = []
    for m in matches:
        if _same_team(key, _tolerant_key(m.get("home"))):
            xg_for.append(float(m["home_xg"]))
            xg_against.append(float(m["away_xg"]))
        else:
            xg_for.append(float(m["away_xg"]))
            xg_against.append(float(m["home_xg"]))
    return {
        "xg_for_avg": round(sum(xg_for) / len(xg_for), 4),
        "xg_against_avg": round(sum(xg_against) / len(xg_against), 4),
        "sample_size": len(matches),
        "source": "understat_history",
    }


class UnderstatBrowserClient:
    """seleniumbase UC (undetected) Chrome used to fetch understat JSON.

    The understat server fingerprints HTTP clients; a real Chrome session is
    required. Mirrors the Sofascore fallback's driver lifecycle.
    """

    def __init__(self, wait_after_open: float = 3.0) -> None:
        self._driver = None
        self._wait = wait_after_open
        self._league_page_opened = False

    def _ensure_driver(self):
        if self._driver is None:
            from seleniumbase import Driver

            self._driver = Driver(uc=True, headless2=True, browser="chrome")
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            driver = self._driver
            self._driver = None

            def _quit() -> None:
                try:
                    driver.quit()
                except Exception:
                    pass

            # Bounded quit: seleniumbase UC teardown can stall indefinitely on
            # a wedged renderer; never let it hang the caller's cleanup.
            t = threading.Thread(target=_quit, daemon=True)
            t.start()
            t.join(timeout=8.0)

    def open_league_page(self, league_key: str, season: int) -> bool:
        """Open the understat league page for a season.

        The season-scoped URL ``/league/{slug}/{season}`` renders the match
        data into the page-global ``datesData`` (verified in a real browser).
        Plain ``driver.get`` is used (like the flashscore client): understat
        does not run a Cloudflare interstitial, so the slow UC reconnect
        dance is unnecessary -- measured ~3x faster, which matters inside the
        runner's 85s deadline.
        """
        slug = UNDERSTAT_LEAGUE.get(league_key)
        if not slug:
            return False
        try:
            driver = self._ensure_driver()
            url = f"https://understat.com/league/{slug.replace(' ', '%20')}/{season}"
            driver.get(url)
            time.sleep(self._wait)
            self._league_page_opened = True
            return True
        except Exception as exc:
            logger.warning("understat open league page failed: %s", type(exc).__name__)
            return False

    def read_dates_data(self) -> list[dict[str, Any]] | None:
        """Return ``window.datesData`` (per-match rows) from the open page.

        The site stores the full season match data (teams, score, xG,
        datetime) in this page-global after load. None if absent.
        """
        try:
            driver = self._ensure_driver()
            driver.set_script_timeout(20)
            for _ in range(3):
                out = driver.execute_script(
                    "return typeof datesData !== 'undefined' ? "
                    "JSON.stringify(datesData) : null;"
                )
                if out:
                    try:
                        data = json.loads(out)
                    except json.JSONDecodeError:
                        return None
                    if isinstance(data, list) and data:
                        return data
                time.sleep(2.0)
            return None
        except Exception as exc:
            logger.warning("understat read datesData failed: %s", type(exc).__name__)
            return None

    def fetch_season(self, league_key: str, season: int) -> dict[str, Any] | None:
        """Open the season page and read the match data into a payload.

        Returns the same shape as the legacy ajax payload (``{"dates": [...]}``)
        so the rest of the pipeline is unchanged.
        """
        if not self.open_league_page(league_key, season):
            return None
        dates = self.read_dates_data()
        if not dates:
            return None
        return {"dates": dates}


def dates_to_rows(
    payload: dict[str, Any],
    league_key: str,
    season: int,
) -> list[dict[str, Any]]:
    """Convert the ``getLeagueData`` JSON ``dates`` array to xG rows.

    Goals are included (understat ``goals``) so the same rows can serve
    as a standalone fixture source for xG evaluation.
    """
    dates = payload.get("dates") or []
    label = SEASON_LABEL.get(season, str(season))
    rows: list[dict[str, Any]] = []
    for m in dates:
        if not m.get("isResult"):
            continue
        h, a = m.get("h") or {}, m.get("a") or {}
        hg, ag = _num((m.get("xG") or {}).get("h")), _num((m.get("xG") or {}).get("a"))
        if hg is None or ag is None:
            continue
        home = _canonical_team(str(h.get("title", "")))
        away = _canonical_team(str(a.get("title", "")))
        if not home or not away:
            continue
        dt = str(m.get("datetime", ""))[:10]
        g = m.get("goals") or {}
        h_goals = _num(g.get("h"))
        a_goals = _num(g.get("a"))
        rows.append({
            "date": dt,
            "home": home,
            "away": away,
            "home_goals": int(h_goals) if h_goals is not None else None,
            "away_goals": int(a_goals) if a_goals is not None else None,
            "home_xg": hg,
            "away_xg": ag,
            "league": league_key,
            "season": label,
        })
    return rows


def download_league_xg(
    league_key: str,
    seasons: list[int],
    out_dir: Path,
) -> dict[str, Any]:
    """Download raw JSON per season + combined xG rows. Returns meta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    client = UnderstatBrowserClient()
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "league": league_key,
        "seasons": {},
        "unmatched_team_names": [],
    }
    try:
        for season in seasons:
            cached = out_dir / f"understat_{league_key}_{season}.json"
            if cached.exists():
                try:
                    payload = json.loads(cached.read_text(encoding="utf-8"))
                    ok = bool(payload.get("dates"))
                except (ValueError, OSError):
                    ok = False
                    payload = None
            else:
                payload = client.fetch_season(league_key, season)
                ok = bool(payload and payload.get("dates"))
                if ok:
                    tmp = cached.with_suffix(f".{__import__('os').getpid()}.tmp")
                    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    __import__("os").replace(tmp, cached)
            season_rows = dates_to_rows(payload or {}, league_key, season)
            rows.extend(season_rows)
            meta["seasons"][str(season)] = {
                "downloaded": ok,
                "matches": len(season_rows),
                "with_xg": len(season_rows),
            }
            print(
                f"  [understat] {league_key} {season}: "
                f"{'OK' if ok else 'FAIL'} {len(season_rows)} matches with xG"
            )
    finally:
        client.close()
    return meta


def collect_league_rows(
    league_key: str,
    seasons: list[int],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ensure per-season JSONs exist (download what is missing), then return
    the combined rows for a league. ``download_league_xg`` does this in one
    pass."""
    meta = download_league_xg(league_key, seasons, out_dir)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        cached = out_dir / f"understat_{league_key}_{season}.json"
        if not cached.exists():
            continue
        payload = json.loads(cached.read_text(encoding="utf-8"))
        rows.extend(dates_to_rows(payload, league_key, season))
    return rows, meta


def _verify_teams(rows: list[dict[str, Any]], known: set[str]) -> list[str]:
    unknown: list[str] = []
    for r in rows:
        for t in (r["home"], r["away"]):
            if t not in known and t not in unknown:
                unknown.append(t)
    return sorted(unknown)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-understat-xg")
    parser.add_argument("--league", default="EPL", help="bot league key (EPL, LaLiga, ...)")
    parser.add_argument(
        "--seasons", default="2022,2023,2024,2025",
        help="comma-separated understat season years",
    )
    parser.add_argument(
        "--out-dir", default=str(ROOT / "cache" / "football"),
        help="directory for raw per-season JSON snapshots",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "cache" / "football" / "understat_xg_rows.json"),
        help="combined rows file",
    )
    parser.add_argument(
        "--fixtures", default=str(ROOT / "cache" / "football" / "epl_fixtures_2022_2026.json"),
        help="baseline fixture cache used to verify team-name coverage",
    )
    args = parser.parse_args(argv)

    seasons = [int(x) for x in args.seasons.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    rows, meta = collect_league_rows(args.league, seasons, out_dir)

    rows_path = Path(args.out)
    if rows:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        # Verify team-name coverage against the baseline fixture cache -- only
        # meaningful for EPL (the baseline cache is the EPL football-data
        # dataset); other leagues are verified by their own join.
        if args.league == "EPL":
            from .backtest import load_fixtures_from_json

            fx = load_fixtures_from_json(args.fixtures)
            known = {f["home"] for f in fx} | {f["away"] for f in fx}
            meta["unmatched_team_names"] = _verify_teams(rows, known)
            print(f"  [understat] unmatched team names vs baseline cache: "
                  f"{meta['unmatched_team_names'] or 'none'}")
        print(f"  [understat] rows written to {rows_path} "
              f"({len(rows)} total)")
    else:
        print("  [understat] no season data downloaded; nothing written")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
