"""Historical odds + results ingestion from football-data.co.uk (free CSV).

The live APIs cannot provide multi-season HISTORICAL odds on free tiers:
The Odds API historical endpoint costs 10x credits (a 500-credit month buys
~8 requests), and API-Football free keeps odds history locked. football-data
.co.uk publishes decades of results + bookmaker odds (Bet365, Pinnacle,
William Hill, averages, maxima) as stable CSV files with no key, no login.

League codes (football-data.co.uk)::

    E0=EPL  SP1=LaLiga  I1=Serie A  D1=Bundesliga  F1=Ligue 1

Seasons are encoded ``yyxy`` (2023-2024 -> 2324). CSV URL pattern::

    https://www.football-data.co.uk/mmz4281/{yyxy}/{code}.csv

(Note: the folder prefix changed from ``mmzz`` to ``mmz4281`` -- the
``mmzz`` paths now return 404.)

This is the data source for two honesty-critical parts of validation:
  - market baseline (margin-free implied probabilities) -- the benchmark any
    model must beat to claim predictive value;
  - ROI (flat-stake, edge >= threshold) -- the only profit measurement we
    will ever report, because it uses REAL historical odds.
"""
from __future__ import annotations

import csv
import io
import os
import time
import urllib.request
from typing import Any

try:
    from curl_cffi import requests as cffi_requests

    HAS_CURL_CFFI = True
except ImportError:  # pragma: no cover -- curl_cffi is in requirements
    HAS_CURL_CFFI = False

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk division code -> our league key.
# Covers the big-5 plus the leagues most likely to feed UCL/UEL/UECL
# qualifiers and group stages, so Elo seeding reaches beyond 20 EPL teams.
LEAGUE_CODES: dict[str, str] = {
    "EPL": "E0",
    "LaLiga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "Eredivisie": "N1",
    "Primeira Liga": "P1",
    "Süper Lig": "T1",
    "Super League 1": "G1",
    "Pro League": "B1",
    "Scottish Premiership": "SC0",
}

# Best-effort mapping from football-data.co.uk team names to the canonical
# names our Elo seed / teams.json use (FBref/sofascore style). Unmapped names
# pass through unchanged; validation stays self-consistent either way.
TEAM_NAME_MAP: dict[str, str] = {
    # EPL
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Man Utd": "Manchester Utd",
    "Nott'm Forest": "Nottingham",
    "Nottm Forest": "Nottingham",
    "AFC Bournemouth": "Bournemouth",
    "Leeds": "Leeds United",
    "Wolverhampton": "Wolves",
    "Norwich": "Norwich City",
    "Stoke": "Stoke City",
    # LaLiga
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Vallecano": "Rayo Vallecano",
    "Espanol": "Espanyol",
    # Serie A
    "Hellas Verona": "Verona",
    "Sampdoria": "Sampdoria",
    # Bundesliga
    "M'gladbach": "Borussia M'gladbach",
    "Dortmund": "Borussia Dortmund",
    "FC Koln": "Koln",
    "Frankfurt": "Eintracht Frankfurt",
    "Leverkusen": "Bayer Leverkusen",
    "Freiburg": "Freiburg",
    # Ligue 1
    "PSG": "Paris Saint-Germain",
    "Paris SG": "Paris Saint-Germain",
    "Marseille": "Marseille",
    "St Etienne": "Saint-Etienne",
    "Lille": "Lille OSC",
    "Lyon": "Olympique Lyonnais",
    "Monaco": "AS Monaco FC",
    # Eredivisie
    "Ajax": "AFC Ajax",
    "PSV": "PSV Eindhoven",
    "Feyenoord": "Feyenoord Rotterdam",
    "Twente": "FC Twente",
    "Utrecht": "FC Utrecht",
    "Nijmegen": "NEC Nijmegen",
    "Heerenveen": "SC Heerenveen",
    "Go Ahead Eagles": "Go Ahead Eagles Deventer",
    "Go Ahead Eagles Deventer": "Go Ahead Eagles Deventer",
    "Sparta Rotterdam": "Sparta Rotterdam",
    "Fortuna Sittard": "Fortuna Sittard",
    "For Sittard": "Fortuna Sittard",
    "PEC Zwolle": "PEC Zwolle",
    "Zwolle": "PEC Zwolle",
    "RKC Waalwijk": "RKC Waalwijk",
    "Excelsior": "Excelsior Rotterdam",
    "Heracles": "Heracles Almelo",
    "Emmen": "FC Emmen",
    "VVV": "VVV-Venlo",
    "Groningen": "FC Groningen",
    # Primeira Liga
    "Benfica": "SL Benfica",
    "Porto": "FC Porto",
    "Sporting": "Sporting CP",
    "Sp Lisbon": "Sporting CP",
    "Braga": "SC Braga",
    "Sp Braga": "SC Braga",
    "Vitoria Guimaraes": "Vitória SC",
    "Guimaraes": "Vitória SC",
    "Rio Ave": "Rio Ave FC",
    "Gil Vicente": "Gil Vicente FC",
    "Estoril": "GD Estoril Praia",
    "Farense": "SC Farense",
    "Boavista": "Boavista FC",
    "Chaves": "GD Chaves",
    "Vizela": "FC Vizela",
    "Casa Pia": "Casa Pia AC",
    "Arouca": "FC Arouca",
    "Moreirense": "Moreirense FC",
    "Famalicao": "FC Famalicão",
    "Nacional": "CD Nacional",
    # Süper Lig
    "Fenerbahce": "Fenerbahçe",
    "Galatasaray": "Galatasaray",
    "Besiktas": "Beşiktaş",
    "Trabzonspor": "Trabzonspor",
    "Basaksehir": "İstanbul Başakşehir",
    "Buyuksehyr": "İstanbul Başakşehir",
    "Adana Demirspor": "Adana Demirspor",
    # Super League 1 (Greece)
    "Olympiakos": "Olympiacos FC",
    "PAOK": "PAOK FC",
    "AEK Athens": "AEK Athens FC",
    "AEK": "AEK Athens FC",
    "Panathinaikos": "Panathinaikos FC",
    "Aris": "Aris Thessaloniki FC",
    "Aris Thessaloniki FC": "Aris Thessaloniki FC",
    # Pro League (Belgium)
    "Union SG": "Union Saint-Gilloise",
    "Royale Union": "Union Saint-Gilloise",
    "St. Gilloise": "Union Saint-Gilloise",
    "Club Brugge": "Club Brugge",
    "Anderlecht": "RSC Anderlecht",
    "Genk": "KRC Genk",
    "Antwerp": "Royal Antwerp FC",
    "Gent": "KAA Gent",
    "Standard Liege": "Standard Liège",
    "Standard": "Standard Liège",
    # Scottish Premiership
    "Celtic": "Celtic FC",
    "Rangers": "Rangers FC",
    "Aberdeen": "Aberdeen FC",
    "Hearts": "Heart of Midlothian FC",
    "Hibs": "Hibernian FC",
    "Hibernian": "Hibernian FC",
    "St Mirren": "St Mirren FC",
    "Motherwell": "Motherwell FC",
    "Dundee United": "Dundee United FC",
    "Kilmarnock": "Kilmarnock FC",
    "Dundee": "Dundee FC",
    "Ross County": "Ross County FC",
    "St Johnstone": "St Johnstone FC",
    "Livingston": "Livingston FC",
}

# Preferred odds column families, in priority order (Pinnacle is the sharpest
# book; Avg is a stable consensus; Bet365 as last resort).
ODDS_SOURCES: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("pinnacle", ("PSH", "PSD", "PSA")),
    ("avg", ("AvgH", "AvgD", "AvgA")),
    ("bet365", ("B365H", "B365D", "B365A")),
)

TOTALS_SOURCES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("pinnacle", ("P>2.5", "P<2.5")),
    ("avg", ("Avg>2.5", "Avg<2.5")),
    ("bet365", ("B365>2.5", "B365<2.5")),
)


def season_code(season: str) -> str:
    """'2023-2024' | '2023/24' | '2324' -> '2324'."""
    s = season.strip().replace("/", "-")
    if len(s) == 4 and s.isdigit():
        return s
    parts = [p for p in s.split("-") if p]
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0][-2:]}{parts[1][-2:]}"
    raise ValueError(f"cannot parse season code from {season!r}")


def url_for(league: str, season: str) -> str:
    code = LEAGUE_CODES.get(league)
    if not code:
        raise KeyError(f"league {league!r} not in {sorted(LEAGUE_CODES)}")
    return f"{BASE_URL}/{season_code(season)}/{code}.csv"


def normalize_team(name: str) -> str:
    name = (name or "").strip()
    return TEAM_NAME_MAP.get(name, name)


def _parse_date(value: str) -> str:
    """football-data.co.uk dates are DD/MM/YYYY (older files DD/MM/YY)."""
    import datetime

    value = (value or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _odds_from_row(row: dict[str, str]) -> tuple[dict[str, float] | None, str | None]:
    """Pick the best available h2h odds family: Pinnacle > Avg > Bet365."""
    for source, (h, d, a) in ODDS_SOURCES:
        oh, od, oa = _num(row.get(h)), _num(row.get(d)), _num(row.get(a))
        if oh and od and oa and oh > 1.0 and od > 1.0 and oa > 1.0:
            return {"home": oh, "draw": od, "away": oa}, source
    return None, None


def _totals_from_row(row: dict[str, str]) -> tuple[float | None, float | None]:
    for source, (o, u) in TOTALS_SOURCES:
        po, pu = _num(row.get(o)), _num(row.get(u))
        if po and pu and po > 1.0 and pu > 1.0:
            return po, pu
    return None, None


def parse_csv(text: str, league: str = "", season: str = "") -> list[dict[str, Any]]:
    """Parse a football-data.co.uk CSV into normalized fixture dicts.

    Only rows with a result (goals + outcome) are kept; odds are optional and
    stay None when the file has none for that row.
    """
    fixtures: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        hg, ag = _num(row.get("FTHG")), _num(row.get("FTAG"))
        if hg is None or ag is None:
            continue
        home = normalize_team(row.get("HomeTeam", ""))
        away = normalize_team(row.get("AwayTeam", ""))
        if not home or not away:
            continue
        odds, odds_source = _odds_from_row(row)
        over25, under25 = _totals_from_row(row)
        fixtures.append(
            {
                "date": _parse_date(row.get("Date", "")),
                "home": home,
                "away": away,
                "home_goals": int(hg),
                "away_goals": int(ag),
                "league": league,
                "season": season,
                "home_odds": (odds or {}).get("home"),
                "draw_odds": (odds or {}).get("draw"),
                "away_odds": (odds or {}).get("away"),
                "odds_source": odds_source,
                "over25_odds": over25,
                "under25_odds": under25,
            }
        )
    return sorted(fixtures, key=lambda f: f["date"])


def _resolve_proxy(proxy: str | None) -> str | None:
    """Explicit proxy wins; ``""`` explicitly means NO proxy. Otherwise follow
    project convention and read SOCCERDATA_PROXY / SOCKS_PROXY / HTTPS_PROXY
    from the environment."""
    if proxy is not None:
        return proxy or None
    for key in ("SOCCERDATA_PROXY", "SOCKS_PROXY", "HTTPS_PROXY"):
        val = os.getenv(key)
        if val:
            return val
    return None


def _fetch_csv(url: str, proxy: str | None, timeout: int) -> str:
    """Download a CSV, honoring SOCKS/HTTPS proxies. curl_cffi (a project
    dependency) handles SOCKS5; urllib is the no-proxy fallback."""
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}
    if HAS_CURL_CFFI:
        resp = cffi_requests.get(
            url,
            proxies=proxies,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (hermes-football validation)"},
        )
        resp.raise_for_status()
        # football-data.co.uk files are Windows-1252; decode as cp1252 so
        # accented team names survive intact.
        return resp.content.decode("cp1252", errors="replace")
    # urllib fallback: only supports http(s) proxies natively.
    handlers: list[urllib.request.BaseHandler] = []
    if proxies and (proxy or "").startswith(("http://", "https://")):
        handlers.append(urllib.request.ProxyHandler(proxies))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (hermes-football validation)"}
    )
    with opener.open(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("cp1252", errors="replace")


def download_league_history(
    league: str,
    seasons: list[str],
    *,
    sleep_seconds: float = 0.3,
    timeout: int = 30,
    proxy: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Download historical results + odds for a league across seasons.

    Returns (fixtures, warnings). A failed season is skipped with a warning --
    one bad season must not kill the whole validation run.
    """
    proxy = _resolve_proxy(proxy)
    fixtures: list[dict[str, Any]] = []
    warnings: list[str] = []
    for season in seasons:
        url = url_for(league, season)
        try:
            text = _fetch_csv(url, proxy, timeout)
            rows = parse_csv(text, league=league, season=season)
            fixtures.extend(rows)
            print(f"  [odds_history] {league} {season}: {len(rows)} matches "
                  f"({url})")
        except Exception as exc:  # noqa: BLE001 -- one season failing is survivable
            warnings.append(f"{league} {season}: {exc}")
        time.sleep(sleep_seconds)
    return fixtures, warnings


def load_history_fixtures(
    leagues: list[str],
    seasons: list[str],
    *,
    sleep_seconds: float = 0.3,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience: download all requested leagues/seasons, raise if empty."""
    fixtures: list[dict[str, Any]] = []
    warnings: list[str] = []
    for league in leagues:
        fx, warn = download_league_history(
            league, seasons, sleep_seconds=sleep_seconds, proxy=proxy
        )
        fixtures.extend(fx)
        warnings.extend(warn)
    if not fixtures:
        raise RuntimeError(
            "football-data.co.uk returned no fixtures. "
            + ("Warnings: " + "; ".join(warnings) if warnings else "Check league/season codes.")
        )
    return fixtures
