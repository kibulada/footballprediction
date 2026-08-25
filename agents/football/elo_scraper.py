"""
ELO Rating Scraper — fetches latest club ELO ratings from elofootball.com.

Usage:
    python -m agents.football.elo_scraper          # scrape + update elo.json
    python -m agents.football.elo_scraper --dry-run # scrape only, don't write
"""

import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ELOFOOTBALL_URL = "https://elofootball.com/"
ELO_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "cache" / "football" / "elo.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _fetch_html(url: str = ELOFOOTBALL_URL, timeout: int = 30) -> str:
    """Fetch HTML from elofootball.com."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_teams(html: str) -> dict[str, dict[str, Any]]:
    """
    Parse team names and ELO ratings from elofootball.com HTML.
    Per-<tr> parsing of the main ranking + form-history tables.
    The old line-based forward search picked up opponent ratings
    from history (Elche 1896 inside Real Betis row -> Elche 2298).
    Filtering by `title =` keeps only the ranking-adjacent cells
    (country flag) and excludes pure prob tables.
    """
    teams: dict[str, dict[str, Any]] = {}
    for tr in re.split(r'<tr>', html):
        m_team = re.search(r'clubid=(\d+)[^>]*>([^<]+)', tr)
        m_rating = re.search(r'class\s*=\s*"ratings"[^>]*>(\d{3,4})', tr)
        if not m_team or not m_rating:
            continue
        if 'title =' not in tr:
            continue
        clubid = m_team.group(1)
        name = m_team.group(2).strip().split('<')[0].strip()
        rating = int(m_rating.group(1))
        if not name or not rating:
            continue
        if name not in teams or rating > teams[name].get("rating", 0):
            teams[name] = {
                "rating": rating,
                "elofootball_id": clubid,
            }
    return teams


def _normalize_name(name: str) -> str:
    """Normalize team name for matching with existing elo.json."""
    mappings = {
        "ManCity": "Manchester City",
        "ManUtd": "Manchester United",
        "Man United": "Manchester United",
        "Real": "Real Madrid",
        "Barca": "Barcelona",
        "PSG": "Paris Saint-Germain",
        "Villa": "Aston Villa",
        "Bayern": "Bayern Munich",
        "BVB": "Borussia Dortmund",
        "Tottenham": "Tottenham Hotspur",
        "Spurs": "Tottenham Hotspur",
        "Athletic": "Athletic Bilbao",
        "Atletico": "Atletico Madrid",
        "Leipzig": "RB Leipzig",
        "Leverkusen": "Bayer Leverkusen",
        "Gladbach": "Borussia M'gladbach",
        "Wolfsburg": "VfL Wolfsburg",
        "Frankfurt": "Ein Frankfurt",
        "Koln": "1. FC Koln",
        "Freiburg": "SC Freiburg",
        "Mainz": "Mainz 05",
        "Hoffenheim": "TSG Hoffenheim",
        "Bremen": "Werder Bremen",
        "Stuttgart": "VfB Stuttgart",
        "Monaco": "AS Monaco FC",
        "Lille": "Lille OSC",
        "Lyon": "Olympique Lyonnais",
        "Marseille": "Olympique de Marseille",
        "Nice": "OGC Nice",
        "Rennes": "Stade Rennais",
        "Lens": "RC Lens",
        "Nantes": "FC Nantes",
        "Brest": "Stade Brestois",
        "Utrecht": "FC Utrecht",
        "Twente": "FC Twente",
        "Feyenoord": "Feyenoord Rotterdam",
        "Heerenveen": "SC Heerenveen",
        "AZ": "AZ Alkmaar",
        "Ajax": "AFC Ajax",
        "PSV": "PSV Eindhoven",
        "Braga": "SC Braga",
        "Porto": "FC Porto",
        "Benfica": "SL Benfica",
        "Sporting": "Sporting CP",
        "Galatasaray": "Galatasaray SK",
        "Fenerbahce": "Fenerbahçe",
        "Besiktas": "Beşiktaş",
        "Trabzonspor": "Trabzonspor",
        "Inter": "Inter Milan",
        "Milan": "AC Milan",
        "Juve": "Juventus",
        "Roma": "AS Roma",
        "Lazio": "SS Lazio",
        "Napoli": "SSC Napoli",
        "Atalanta": "Atalanta BC",
        "Fiorentina": "ACF Fiorentina",
        "Torino": "Torino FC",
        "Bologna": "Bologna FC",
        "Genoa": "Genoa CFC",
        "Cagliari": "Cagliari Calcio",
        "Parma": "Parma Calcio",
        "Como": "Como 1907",
        "Leeds": "Leeds United",
        "Burnley": "Burnley FC",
        "Sheffield Utd": "Sheffield United",
        "Sunderland": "Sunderland AFC",
        "Norwich": "Norwich City",
        "Watford": "Watford FC",
        "West Brom": "West Bromwich Albion",
        "Coventry": "Coventry City",
        "Hull": "Hull City",
        "Swansea": "Swansea City",
        "Middlesbrough": "Middlesbrough FC",
        "Millwall": "Millwall FC",
        "Blackburn": "Blackburn Rovers",
        "QPR": "Queens Park Rangers",
        "Cardiff": "Cardiff City",
        "Stoke": "Stoke City",
        "Birmingham": "Birmingham City",
        "Preston": "Preston North End",
        "Derby": "Derby County",
        "Plymouth": "Plymouth Argyle",
        "Oxford": "Oxford United",
        "Charlton": "Charlton Athletic",
        "Wrexham": "Wrexham AFC",
        "Portsmouth": "Portsmouth FC",
        "Celtic": "Celtic FC",
        "Rangers": "Rangers FC",
        "Hearts": "Heart of Midlothian FC",
        "Hibernian": "Hibernian FC",
        "Aberdeen": "Aberdeen FC",
        "Dundee": "Dundee FC",
        "Dundee Utd": "Dundee United FC",
        "Kilmarnock": "Kilmarnock FC",
        "St Mirren": "St Mirren FC",
        "Motherwell": "Motherwell FC",
        "Livingston": "Livingston FC",
        "AEK Athens": "AEK Athens FC",
        "PAOK": "PAOK FC",
        "Olympiacos": "Olympiacos FC",
        "Panathinaikos": "Panathinaikos FC",
        "Aris": "Aris Thessaloniki FC",
        "Fenerbahçe": "Fenerbahçe",
        "Basaksehir": "İstanbul Başakşehir",
        "Antalyaspor": "Antalyaspor",
        "Konyaspor": "Konyaspor",
        "Gaziantep": "Gaziantep FK",
        "Kayserispor": "Kayserispor",
        "Sivasspor": "Sivasspor",
        "Alanyaspor": "Alanyaspor",
        "Kasimpasa": "Kasımpaşa",
        "Rizespor": "Çaykur Rizespor",
        "Samsunspor": "Samsunspor",
        "Goztepe": "Göztepe",
    }

    # Try exact match first
    if name in mappings:
        return mappings[name]

    # Try case-insensitive
    name_lower = name.lower()
    for k, v in mappings.items():
        if k.lower() == name_lower:
            return v

    return name


def scrape_elo() -> dict[str, dict[str, Any]]:
    """Scrape ELO ratings from elofootball.com."""
    print("Fetching elofootball.com...")
    html = _fetch_html()

    print("Parsing teams...")
    teams = _parse_teams(html)

    print(f"Found {len(teams)} teams from elofootball.com")

    # Normalize names
    result = {}
    for name, data in teams.items():
        normalized = _normalize_name(name)
        result[normalized] = {
            "rating": data["rating"],
            "elofootball_id": data.get("elofootball_id"),
        }

    return result


def update_elo_json(new_data: dict[str, dict[str, Any]], dry_run: bool = False, single_source: bool = False) -> dict[str, Any]:
    """
    Update elo.json with new ELO data.
    single_source=True = 1 sumber elofootball.com saja (hapus rating lama
    yang tidak ada di new_data) — sesuai permintaan 2026-08-24 biar tidak rancu.
    """
    # Load existing
    existing = {"ratings": {}, "games": {}}
    if ELO_JSON_PATH.exists():
        try:
            existing = json.loads(ELO_JSON_PATH.read_text(encoding="utf-8"))
            if "ratings" not in existing:
                existing = {"ratings": existing, "games": {}}
        except Exception:
            pass

    old_ratings = existing.get("ratings", {})
    old_games = existing.get("games", {})

    # Single-source: hanya elofootball.com (hapus duplikat Jong/B & football-data lama)
    if single_source:
        updated_ratings: dict[str, float] = {}
        updated_games: dict[str, int] = {}
        changes: list[str] = []
        for team, data in new_data.items():
            new_rating = data["rating"]
            old_rating = old_ratings.get(team)
            if old_rating is None:
                changes.append(f"ADDED: {team} = {new_rating}")
            elif old_rating != new_rating:
                diff = new_rating - old_rating
                changes.append(f"UPDATED: {team} {old_rating} -> {new_rating} ({'+'if diff>0 else ''}{diff})")
            else:
                changes.append(f"UNCHANGED: {team} = {new_rating}")
            updated_ratings[team] = new_rating
            # games dipertahankan kalau ada, else 0 (elofootball tidak kasih games)
            updated_games[team] = int(old_games.get(team, 0))
        # Team lama yang tidak ada di elofootball = REMOVED (single source)
        for team in old_ratings:
            if team not in new_data:
                changes.append(f"REMOVED: {team} = {old_ratings[team]} (hanya elofootball.com)")
        single_source_note = " (single-source elofootball.com)"
    else:
        # Merge: new data overwrites existing (legacy, keep old)
        updated_ratings = dict(old_ratings)
        updated_games = dict(old_games)
        changes = []
        for team, data in new_data.items():
            new_rating = data["rating"]
            old_rating = old_ratings.get(team)
            if old_rating is None:
                changes.append(f"ADDED: {team} = {new_rating}")
                updated_ratings[team] = new_rating
            elif old_rating != new_rating:
                diff = new_rating - old_rating
                changes.append(f"UPDATED: {team} {old_rating} -> {new_rating} ({'+'if diff>0 else ''}{diff})")
                updated_ratings[team] = new_rating
        for team in old_ratings:
            if team not in new_data:
                changes.append(f"KEPT: {team} = {old_ratings[team]} (not in new data)")
        single_source_note = ""

    # Write
    output = {
        "ratings": updated_ratings,
        "games": updated_games,
        "last_updated": datetime.now().isoformat(),
        "source": "elofootball.com" + single_source_note,
    }

    if not dry_run:
        # Backup dulu biar bisa rollback
        if ELO_JSON_PATH.exists():
            backup = ELO_JSON_PATH.with_suffix(".json.backup-elofootball-" + datetime.now().strftime("%Y%m%d"))
            try:
                backup.write_text(ELO_JSON_PATH.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"Backup: {backup}")
            except Exception:
                pass
        ELO_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        ELO_JSON_PATH.write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nWritten to {ELO_JSON_PATH}")
    else:
        print(f"\nDry run - would write to {ELO_JSON_PATH}")

    return {
        "total_teams": len(updated_ratings),
        "updated": len([c for c in changes if c.startswith("UPDATED")]),
        "added": len([c for c in changes if c.startswith("ADDED")]),
        "kept": len([c for c in changes if c.startswith("KEPT")]),
        "removed": len([c for c in changes if c.startswith("REMOVED")]),
        "unchanged": len([c for c in changes if c.startswith("UNCHANGED")]),
        "changes": changes,
    }


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    single_source = "--single-source" in sys.argv or "--single" in sys.argv

    teams = scrape_elo()
    if not teams:
        print("No teams scraped")
        return 1

    result = update_elo_json(teams, dry_run=dry_run, single_source=single_source)

    print(f"\nSummary:")
    print(f"   Total teams: {result['total_teams']}")
    print(f"   Updated: {result['updated']}")
    print(f"   Added: {result['added']}")
    print(f"   Kept (unchanged): {result['kept']}")

    if result["changes"]:
        print(f"\nChanges:")
        for c in result["changes"][:20]:
            print(f"   {c}")
        if len(result["changes"]) > 20:
            print(f"   ... and {len(result['changes']) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
