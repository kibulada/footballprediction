"""Compare two teams using multi-source stats fetcher."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .cache import Cache
from .multi_source import MultiSourceStatsFetcher
from .odds_fetcher import OddsFetcher

LEAGUES_PATH = Path(__file__).parent / "leagues.json"


def _load_leagues() -> dict[str, dict[str, Any]]:
    return json.loads(LEAGUES_PATH.read_text(encoding="utf-8"))


def _season_now() -> int:
    now = datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1


async def compare_teams(
    *,
    home_alias: str,
    away_alias: str,
    league: str,
    cfg: dict[str, Any],
    odds: OddsFetcher,
    stats: MultiSourceStatsFetcher,
    cache: Cache,
) -> dict[str, Any]:
    leagues = _load_leagues()
    meta = leagues.get(league)
    if not meta:
        return {"error": f"liga '{league}' tidak dikenal"}

    meta_with_season = {**meta, "season": _season_now()}

    home_team = await stats.search_team(home_alias, meta_with_season)
    away_team = await stats.search_team(away_alias, meta_with_season)

    home_id = (home_team or {}).get("id")
    away_id = (away_team or {}).get("id")
    if not home_id or not away_id:
        return {
            "error": "tim tidak ditemukan",
            "home_query": home_alias,
            "away_query": away_alias,
        }

    home_name = home_team["name"]
    away_name = away_team["name"]

    home_form = await stats.fetch_team_form(home_id, meta_with_season)
    away_form = await stats.fetch_team_form(away_id, meta_with_season)
    h2h = await stats.fetch_h2h(home_id, away_id, meta_with_season)

    sources = sorted({
        (home_team.get("provider") if home_team else None),
        (away_team.get("provider") if away_team else None),
        (home_form.get("source") if home_form else None),
        (away_form.get("source") if away_form else None),
        (h2h.get("source") if h2h else None),
    } - {None})

    from .timeutil import utc_now_iso

    return {
        "home": home_name,
        "away": away_name,
        "league": meta["display"],
        "generated_at": utc_now_iso(),
        "stats": {
            "home_form": (home_form or {}).get("sequence", "n/a"),
            "away_form": (away_form or {}).get("sequence", "n/a"),
            "h2h": {
                "wins": (h2h or {}).get("wins", 0),
                "draws": (h2h or {}).get("draws", 0),
                "losses": (h2h or {}).get("losses", 0),
            },
        },
        "sources": sources,
        "quota": {
            "odds_api_remaining": odds.last_remaining,
            "odds_blocked": odds.quota_blocked,
            "football_data_warning": stats.fd.rate_limit_warning,
        },
    }
