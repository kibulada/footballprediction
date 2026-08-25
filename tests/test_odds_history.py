"""Offline tests for odds_history.py (football-data.co.uk ingestion)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.odds_history import (
    LEAGUE_CODES,
    normalize_team,
    parse_csv,
    season_code,
    url_for,
)

CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA,P>2.5,P<2.5,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5
E0,13/08/2023,Arsenal,Nottingham,2,1,H,1.44,4.5,8.0,1.5,4.2,7.5,1.7,2.1,1.53,4.33,6.95,1.75,2.05
E0,19/08/2023,Man City,Newcastle,1,0,H,1.6,4.0,6.0,,,,,,,,,,
E0,26/08/2023,Brighton,West Ham,1,3,A,2.1,3.6,3.4,2.2,3.5,3.2,1.9,1.8,2.15,3.55,3.3,1.9,1.8
"""


def test_league_code_mapping():
    assert LEAGUE_CODES["EPL"] == "E0"
    assert LEAGUE_CODES["LaLiga"] == "SP1"
    assert LEAGUE_CODES["Serie A"] == "I1"
    assert LEAGUE_CODES["Bundesliga"] == "D1"
    assert LEAGUE_CODES["Ligue 1"] == "F1"


def test_league_code_mapping_ucl_feeder_leagues():
    """UCL-feeder leagues are seedable so Elo covers more than 20 EPL teams."""
    assert LEAGUE_CODES["Eredivisie"] == "N1"
    assert LEAGUE_CODES["Primeira Liga"] == "P1"
    assert LEAGUE_CODES["Süper Lig"] == "T1"
    assert LEAGUE_CODES["Super League 1"] == "G1"
    assert LEAGUE_CODES["Pro League"] == "B1"
    assert LEAGUE_CODES["Scottish Premiership"] == "SC0"


def test_season_code_conversion():
    assert season_code("2023-2024") == "2324"
    assert season_code("2025-2026") == "2526"
    assert season_code("2324") == "2324"
    assert season_code("2023/24") == "2324"


def test_url_for():
    assert url_for("EPL", "2023-2024") == (
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    )
    assert url_for("Serie A", "2425") == (
        "https://www.football-data.co.uk/mmz4281/2425/I1.csv"
    )


def test_parse_csv_team_normalization():
    fx = parse_csv(CSV, league="EPL", season="2023-2024")
    by_home = {f["home"]: f for f in fx}
    assert "Manchester City" in by_home  # Man City -> Manchester City
    assert "Brighton" in by_home  # unchanged
    arsenal = [f for f in fx if f["home"] == "Arsenal"][0]
    assert arsenal["away"] == "Nottingham"  # unchanged passthrough


def test_parse_csv_pinnacle_preferred():
    fx = parse_csv(CSV)
    arsenal = [f for f in fx if f["home"] == "Arsenal"][0]
    assert arsenal["odds_source"] == "pinnacle"
    assert arsenal["home_odds"] == 1.5
    assert arsenal["draw_odds"] == 4.2
    assert arsenal["away_odds"] == 7.5


def test_parse_csv_fallback_bet365():
    fx = parse_csv(CSV)
    city = [f for f in fx if f["home"] == "Manchester City"][0]
    assert city["odds_source"] == "bet365"
    assert city["home_odds"] == 1.6


def test_parse_csv_totals_captured():
    fx = parse_csv(CSV)
    arsenal = [f for f in fx if f["home"] == "Arsenal"][0]
    assert arsenal["over25_odds"] == 1.7
    assert arsenal["under25_odds"] == 2.1


def test_parse_csv_date_conversion():
    fx = parse_csv(CSV)
    arsenal = [f for f in fx if f["home"] == "Arsenal"][0]
    assert arsenal["date"] == "2023-08-13"


def test_parse_csv_result_and_goals():
    fx = parse_csv(CSV)
    arsenal = [f for f in fx if f["home"] == "Arsenal"][0]
    assert arsenal["home_goals"] == 2
    assert arsenal["away_goals"] == 1


def test_parse_csv_skips_row_without_result():
    fx = parse_csv(CSV + "\nE0,02/09/2023,Chelsea,Liverpool,,\n")
    assert not any(f["home"] == "Chelsea" for f in fx)


def test_parse_csv_sorted_by_date():
    fx = parse_csv(CSV)
    dates = [f["date"] for f in fx]
    assert dates == sorted(dates)


def test_normalize_team_unknown_passthrough():
    assert normalize_team("Some Unknown FC") == "Some Unknown FC"
    assert normalize_team("") == ""


def test_normalize_team_ucl_feeder_mapping():
    """Feeder-league spellings map to the canonical names used by teams.json
    so seeded Elo ratings are found by the live resolver."""
    assert normalize_team("Union SG") == "Union Saint-Gilloise"
    assert normalize_team("Royale Union") == "Union Saint-Gilloise"
    assert normalize_team("Celtic") == "Celtic FC"
    assert normalize_team("Rangers") == "Rangers FC"
    assert normalize_team("Fenerbahce") == "Fenerbahçe"
    assert normalize_team("Olympiakos") == "Olympiacos FC"
    assert normalize_team("Benfica") == "SL Benfica"
    assert normalize_team("Ajax") == "AFC Ajax"
    assert normalize_team("PSV") == "PSV Eindhoven"


# ---- live value baseline (closing fair line) -----------------------------

def _fx(home_odds, draw_odds, away_odds, home="Arsenal", away="Chelsea"):
    return {
        "date": "2025-08-10", "home": home, "away": away,
        "home_goals": 1, "away_goals": 0, "league": "EPL", "season": "2025-2026",
        "home_odds": home_odds, "draw_odds": draw_odds, "away_odds": away_odds,
        "odds_source": "pinnacle",
    }


def test_league_closing_baseline_averages_implied():
    from agents.football.odds_history import league_closing_baseline

    fx = [
        _fx(2.0, 3.5, 4.0),
        _fx(2.0, 3.5, 4.0),
        _fx(3.0, 3.2, 2.4),
    ]
    out = league_closing_baseline(fx)
    assert out is not None
    assert out["n"] == 3
    # margin = (1/2+1/3.5+1/4) - 1 = 0.5357... avg over rows
    assert out["margin"] > 0.0
    # home implied must be in a sane band (raw 1/2=0.5 normalized ~0.483)
    assert 0.4 < out["home"]["implied"] < 0.6
    assert out["home"]["n"] == 3


def test_league_closing_baseline_skips_rows_without_odds():
    from agents.football.odds_history import league_closing_baseline

    fx = [_fx(None, None, None), _fx(2.0, 3.5, 4.0)]
    out = league_closing_baseline(fx)
    assert out is not None
    assert out["n"] == 1


def test_league_closing_baseline_none_when_no_odds():
    from agents.football.odds_history import league_closing_baseline

    assert league_closing_baseline([_fx(None, None, None)]) is None


def test_live_value_signal_compares_consensus_vs_history():
    from agents.football.odds_history import league_closing_baseline, live_value_signal

    base = league_closing_baseline([_fx(2.0, 3.5, 4.0)] * 10)
    # Consensus heavily favours home (1.6) vs historical 2.0 -> home shortened
    # vs the fair line -> negative value (odds moved AWAY from history).
    out = live_value_signal({"home": 1.6, "draw": 3.6, "away": 4.4}, base)
    assert out is not None
    assert out["home"]["value"] < 0
    assert out["away"]["value"] > 0
    # consensus implying more home prob than baseline -> implied > baseline
    assert out["home"]["implied"] > out["home"]["baseline"]


def test_live_value_signal_none_without_inputs():
    from agents.football.odds_history import live_value_signal

    assert live_value_signal(None, {}) is None
    assert live_value_signal({"home": 2.0}, None) is None


def test_load_league_baseline_missing_cache_returns_none():
    from agents.football.odds_history import load_league_baseline

    assert load_league_baseline("NotARealLeagueXYZ", root="cache/football") is None


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
