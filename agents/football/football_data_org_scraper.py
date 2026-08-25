"""Scraper for football-data.org API — downloads match results + odds.

Free tier (TIER_TWO): UEL (EL), MLS, Scottish Premiership (SPL)
Paid tier (TIER_FOUR): UECL (UCL), Copa Libertadores (CLI)

Setup:
  1. Register at https://www.football-data.org/client/register
  2. Get API key from email
  3. Set env var: export FOOTBALL_DATA_ORG_KEY=your_key_here
     Or pass via --key flag

Usage:
  python -m agents.football.football_data_org_scraper --key YOUR_KEY
  python -m agents.football.football_data_org_scraper --key YOUR_KEY --leagues EL,MLS
  python -m agents.football.football_data_org_scraper --key YOUR_KEY --season 2024
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


BASE_URL = "https://api.football-data.org/v4"

# Free tier competitions (TIER_TWO)
FREE_LEAGUES = {
    "EL": {"name": "UEFA Europa League", "seasons": [2023, 2024, 2025]},
    "MLS": {"name": "MLS", "seasons": [2023, 2024, 2025]},
    "SPL": {"name": "Scottish Premiership", "seasons": [2023, 2024, 2025]},
}

# Paid tier competitions (TIER_FOUR)
PAID_LEAGUES = {
    "UCL": {"name": "UEFA Conference League", "seasons": [2023, 2024, 2025]},
    "CLI": {"name": "Copa Libertadores", "seasons": [2023, 2024, 2025]},
}

# All known competitions
ALL_LEAGUES = {**FREE_LEAGUES, **PAID_LEAGUES}


def _api_get(path: str, key: str, params: dict | None = None) -> dict:
    """Make authenticated GET request to football-data.org API."""
    url = f"{BASE_URL}{path}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{query}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "HermesFootball/1.0",
            "X-Auth-Token": key,
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def fetch_matches(key: str, league_code: str, season: int) -> list[dict]:
    """Fetch all finished matches for a league/season."""
    matches = []
    offset = 0
    limit = 50  # API max per request

    while True:
        data = _api_get(
            f"/competitions/{league_code}/matches",
            key,
            params={
                "season": str(season),
                "status": "FINISHED",
                "limit": str(limit),
                "offset": str(offset),
            },
        )

        batch = data.get("matches", [])
        if not batch:
            break

        for m in batch:
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            score = m.get("score", {})
            ft = score.get("fullTime", {})

            matches.append({
                "id": m.get("id"),
                "competition": league_code,
                "season": season,
                "utcDate": m.get("utcDate"),
                "home_team": home.get("name"),
                "home_id": home.get("id"),
                "away_team": away.get("name"),
                "away_id": away.get("id"),
                "home_score": ft.get("home"),
                "away_score": ft.get("away"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "stage": m.get("stage"),
            })

        offset += limit
        if len(batch) < limit:
            break

        # Rate limit: 10 req/min for free tier
        time.sleep(6.5)

    return matches


def fetch_odds(key: str, league_code: str, season: int) -> list[dict]:
    """Fetch odds for matches in a league/season (TIER_TWO+)."""
    odds_data = []
    offset = 0
    limit = 50

    while True:
        try:
            data = _api_get(
                f"/competitions/{league_code}/matches",
                key,
                params={
                    "season": str(season),
                    "status": "FINISHED",
                    "limit": str(limit),
                    "offset": str(offset),
                    "odds": "true",
                },
            )
        except RuntimeError as e:
            if "403" in str(e):
                print(f"  ⚠️ Odds not available for {league_code} (need higher tier)")
                break
            raise

        batch = data.get("matches", [])
        if not batch:
            break

        for m in batch:
            match_id = m.get("id")
            odds = m.get("odds", {})
            if odds:
                odds_data.append({
                    "match_id": match_id,
                    "home_team": m.get("homeTeam", {}).get("name"),
                    "away_team": m.get("awayTeam", {}).get("name"),
                    "utcDate": m.get("utcDate"),
                    "bookmakers": odds.get("bookmakers", []),
                })

        offset += limit
        if len(batch) < limit:
            break

        time.sleep(6.5)

    return odds_data


def scrape_league(key: str, league_code: str, seasons: list[int], output_dir: Path) -> dict:
    """Scrape all data for a league across seasons."""
    report = {"league": league_code, "seasons": {}, "total_matches": 0, "total_odds": 0}

    for season in seasons:
        print(f"\n  📥 {league_code} season {season}-{season+1}...")

        # Fetch matches
        matches = fetch_matches(key, league_code, season)
        print(f"    Matches: {len(matches)}")

        # Fetch odds (may fail for paid tiers)
        odds = []
        try:
            odds = fetch_odds(key, league_code, season)
            print(f"    Odds: {len(odds)}")
        except Exception as e:
            print(f"    Odds: ❌ {e}")

        # Save
        season_file = output_dir / f"football_data_{league_code}_{season}.json"
        with open(season_file, "w", encoding="utf-8") as f:
            json.dump({
                "league": league_code,
                "season": season,
                "scraped_at": datetime.utcnow().isoformat(),
                "matches": matches,
                "odds": odds,
            }, f, indent=2, ensure_ascii=False)

        report["seasons"][str(season)] = {
            "matches": len(matches),
            "odds": len(odds),
        }
        report["total_matches"] += len(matches)
        report["total_odds"] += len(odds)

        print(f"    ✅ Saved: {season_file.name}")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="football-data-org-scraper",
        description="Scrape match results + odds from football-data.org",
    )
    parser.add_argument("--key", required=True, help="football-data.org API key")
    parser.add_argument(
        "--leagues",
        default="EL,MLS",
        help="Comma-separated league codes (default: EL,MLS)",
    )
    parser.add_argument(
        "--seasons",
        default="2023,2024,2025",
        help="Comma-separated start years (default: 2023,2024,2025)",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/football/football_data_org",
        help="Output directory",
    )

    args = parser.parse_args(argv)
    key = args.key
    league_codes = [c.strip().upper() for c in args.leagues.split(",")]
    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🌐 football-data.org Scraper")
    print(f"   Leagues: {league_codes}")
    print(f"   Seasons: {seasons}")
    print(f"   Output: {output_dir}")

    all_reports = {}
    for code in league_codes:
        if code not in ALL_LEAGUES:
            print(f"\n⚠️ Unknown league code: {code}")
            print(f"   Available: {list(ALL_LEAGUES.keys())}")
            continue

        info = ALL_LEAGUES[code]
        tier = "FREE" if code in FREE_LEAGUES else "PAID"
        print(f"\n🏆 {info['name']} ({code}) [{tier}]")

        report = scrape_league(key, code, seasons, output_dir)
        all_reports[code] = report

        # Rate limit between leagues
        time.sleep(2)

    # Save summary
    summary_file = output_dir / "scrape_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_at": datetime.utcnow().isoformat(),
            "reports": all_reports,
        }, f, indent=2)

    print(f"\n✅ Summary saved: {summary_file}")
    print(f"\n📊 Total:")
    for code, r in all_reports.items():
        print(f"   {code}: {r['total_matches']} matches, {r['total_odds']} odds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
