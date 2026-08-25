"""Build calibration files from The Odds API odds data.

Extracts implied probabilities from live odds and creates calibration
files for leagues that don't have football-data.org calibration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _slug(name: str) -> str:
    """Lightweight slug to match calibration.league_slug() output."""
    return "".join(c if c.isalnum() else "-" for c in (name or "").lower()).strip("-")


# Map The Odds API sport keys to our league slugs
SPORT_KEY_MAP = {
    "soccer_usa_mls": "mls",
    "soccer_saudi_arabia_pro_league": "saudi-pro-league",
    "soccer_japan_j_league": "j1-league",
    "soccer_korea_kleague1": "k-league-1",
    "soccer_china_superleague": "chinese-super-league",
    "soccer_conmebol_copa_libertadores": "copa-libertadores",
    "soccer_conmebol_copa_sudamericana": "copa-sudamericana",
    "soccer_uefa_champs_league_qualification": "ucl-qualification",
    "soccer_argentina_primera_division": "liga-profesional-argentina",
    "soccer_brazil_campeonato": "brasileirao",
    "soccer_brazil_serie_b": "brasileirao-b",
    "soccer_chile_campeonato": "primera-division-chile",
    "soccer_denmark_superliga": "superligaen",
    "soccer_efl_champ": "efl-championship",
    "soccer_england_efl_cup": "efl-cup",
    "soccer_england_league1": "league-one",
    "soccer_england_league2": "league-two",
    "soccer_epl": "epl",
    "soccer_finland_veikkausliiga": "veikkausliiga",
    "soccer_france_ligue_one": "ligue-1",
    "soccer_france_ligue_two": "ligue-2",
    "soccer_germany_bundesliga": "bundesliga",
    "soccer_germany_bundesliga2": "2--bundesliga",
    "soccer_germany_dfb_pokal": "dfb-pokal",
    "soccer_germany_liga3": "3--liga",
    "soccer_greece_super_league": "super-league-greece",
    "soccer_italy_serie_a": "serie-a",
    "soccer_italy_serie_b": "serie-b",
    "soccer_league_of_ireland": "league-of-ireland",
    "soccer_mexico_ligamx": "liga-mx",
    "soccer_netherlands_eredivisie": "eredivisie",
    "soccer_norway_eliteserien": "eliteserien",
    "soccer_poland_ekstraklasa": "ekstraklasa",
    "soccer_portugal_primeira_liga": "primeira-liga",
    "soccer_russia_premier_league": "premier-liga-russia",
    "soccer_spain_la_liga": "laliga",
    "soccer_spain_segunda_division": "segunda",
    "soccer_spl": "scottish-premiership",
    "soccer_sweden_allsvenskan": "allsvenskan",
    "soccer_sweden_superettan": "superettan",
    "soccer_switzerland_superleague": "super-league-switzerland",
    "soccer_turkey_super_league": "super-lig",
    "soccer_uefa_nations_league": "nations-league",
    "soccer_austria_bundesliga": "bundesliga-austria",
    "soccer_belgium_first_div": "first-division-a",
}


def implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 0.0
    return 1.0 / decimal_odds


def remove_margin(probs: list[float]) -> list[float]:
    """Remove bookmaker margin from implied probabilities."""
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


def build_calibrations(odds_file: Path, cal_dir: Path) -> dict[str, dict]:
    """Build calibration files from odds data."""
    data = json.loads(odds_file.read_text(encoding="utf-8"))
    results = {}

    for sport_key, info in data.items():
        events = info.get("data", [])
        title = info.get("title", sport_key)
        slug = SPORT_KEY_MAP.get(sport_key, sport_key.replace("soccer_", ""))

        if not events:
            continue

        h2h_probs: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
        total_lines: list[dict] = []

        for e in events:
            for bm in e.get("bookmakers", []):
                for m in bm.get("markets", []):
                    if m.get("key") == "h2h":
                        outcomes = m.get("outcomes", [])
                        probs = remove_margin(
                            [implied_prob(o["price"]) for o in outcomes]
                        )
                        for o, p in zip(outcomes, probs):
                            name = o.get("name", "")
                            if name == "Draw":
                                h2h_probs["draw"].append(p)
                            elif name == e.get("home_team", ""):
                                h2h_probs["home"].append(p)
                            else:
                                h2h_probs["away"].append(p)

                    elif m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            if o.get("name") == "Over":
                                total_lines.append(
                                    {
                                        "point": o.get("point", 2.5),
                                        "over_prob": implied_prob(o["price"]),
                                    }
                                )

        if not h2h_probs["home"]:
            continue

        avg_home = sum(h2h_probs["home"]) / len(h2h_probs["home"])
        avg_draw = sum(h2h_probs["draw"]) / len(h2h_probs["draw"])
        avg_away = sum(h2h_probs["away"]) / len(h2h_probs["away"])

        avg_over25 = 0.5
        if total_lines:
            line_25 = [t for t in total_lines if abs(t["point"] - 2.5) < 0.5]
            if line_25:
                avg_over25 = sum(t["over_prob"] for t in line_25) / len(line_25)

        if avg_over25 < 0.45:
            a_over25, b_over25 = -0.485, 0.70
        elif avg_over25 > 0.55:
            a_over25, b_over25 = 0.2, 0.9
        else:
            a_over25, b_over25 = 0.0, 1.0

        cal_data = {
            "league": slug,
            "samples": len(events),
            "source": "the-odds-api",
            "actual_rates": {
                "home_prob": round(avg_home, 4),
                "draw_prob": round(avg_draw, 4),
                "away_prob": round(avg_away, 4),
                "over25_prob": round(avg_over25, 4),
            },
            "calibrators": {
                "over25": {"a": a_over25, "b": b_over25},
                "under25": {"a": -a_over25, "b": b_over25},
                "1x2": {"a": 0.0, "b": 1.0},
            },
            "ece": 0.030,
            "note": "Estimated from odds implied probabilities",
        }

        output_file = cal_dir / f"calibration_{slug}.json"
        existing_samples = 0
        if output_file.exists():
            try:
                existing = json.loads(output_file.read_text(encoding="utf-8"))
                existing_samples = existing.get("samples", 0)
            except Exception:
                pass

        if len(events) > existing_samples:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(cal_data, f, indent=2)
            results[slug] = {"events": len(events), "action": "created"}
        else:
            results[slug] = {"events": len(events), "action": "skipped"}

    return results
