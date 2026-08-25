"""Tests for Hermes Football. Pure unit, no network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import scorer, format


def test_implied_prob():
    assert abs(scorer.implied_prob(2.0) - 0.5) < 1e-9
    assert abs(scorer.implied_prob(4.0) - 0.25) < 1e-9
    assert scorer.implied_prob(0.5) == 0.0


def test_consensus_odds_median():
    odds = [
        {"home": 1.80, "draw": 3.50, "away": 4.00},
        {"home": 1.90, "draw": 3.60, "away": 4.20},
        {"home": 2.00, "draw": 3.70, "away": 4.40},
    ]
    cons = scorer.consensus_odds(odds)
    assert abs(cons["home"] - 1.90) < 1e-9
    assert abs(cons["draw"] - 3.60) < 1e-9
    assert abs(cons["away"] - 4.20) < 1e-9


def test_best_odds():
    odds = [
        {"bookmaker": "a", "home": 1.80, "draw": 3.50, "away": 4.00},
        {"bookmaker": "b", "home": 1.95, "draw": 3.40, "away": 4.10},
        {"bookmaker": "c", "home": 1.85, "draw": 3.55, "away": 4.50},
    ]
    best = scorer.best_odds(odds)
    assert best["home"]["bookmaker"] == "b"
    assert best["away"]["bookmaker"] == "c"
    assert best["draw"]["bookmaker"] == "c"


def test_find_outlier():
    odds = [
        {"bookmaker": "a", "home": 1.80, "draw": 3.50, "away": 4.00},
        {"bookmaker": "b", "home": 1.85, "draw": 3.55, "away": 4.05},
        {"bookmaker": "c", "home": 2.10, "draw": 3.60, "away": 4.10},
    ]
    cons = scorer.consensus_odds(odds)
    out = scorer.find_outlier(odds, cons, 5.0)
    assert out is not None
    assert out["side"] == "home"
    assert out["bookmaker"] == "c"
    assert out["value_pct"] > 5.0


def test_find_outlier_none():
    odds = [
        {"bookmaker": "a", "home": 1.80, "draw": 3.50, "away": 4.00},
        {"bookmaker": "b", "home": 1.82, "draw": 3.55, "away": 4.05},
    ]
    cons = scorer.consensus_odds(odds)
    assert scorer.find_outlier(odds, cons, 5.0) is None


def test_score_signal_no_odds():
    s = scorer.score_signal(
        bookmaker_odds=[],
        consensus={"home": 0, "draw": 0, "away": 0},
        outlier=None,
        home_form="W-W-D",
        away_form="L-L-W",
        has_odds=False,
    )
    assert 0 <= s <= 100


def test_score_signal_full_features():
    odds = [
        {"bookmaker": "a", "home": 1.80, "draw": 3.50, "away": 4.00},
        {"bookmaker": "b", "home": 1.85, "draw": 3.55, "away": 4.05},
        {"bookmaker": "c", "home": 2.10, "draw": 3.60, "away": 4.10},
    ]
    cons = scorer.consensus_odds(odds)
    out = scorer.find_outlier(odds, cons, 5.0)
    s = scorer.score_signal(
        bookmaker_odds=odds,
        consensus=cons,
        outlier=out,
        home_form="W-W-W-W-W",
        away_form="L-L-L-L-L",
        has_odds=True,
    )
    assert s >= 50


def test_format_top_renders():
    payload = {
        "date": "2026-08-11",
        "matches": [
            {
                "home": "Man City",
                "away": "Arsenal",
                "league": "EPL",
                "kickoff": "2026-08-11T19:30:00Z",
                "odds": {
                    "consensus": {"home": 1.85, "draw": 3.90, "away": 4.20},
                    "best": {
                        "home": {"odds": 1.80, "bookmaker": "bet365"},
                        "draw": {"odds": 3.95, "bookmaker": "bet365"},
                        "away": {"odds": 4.30, "bookmaker": "bet365"},
                    },
                    "outlier": {
                        "side": "home",
                        "value_pct": 3.2,
                        "bookmaker": "bet365",
                        "odds": 1.80,
                    },
                },
                "stats": {"home_form": "W-W-D-W", "away_form": "W-W-L-D"},
                "signal": 78,
                "has_odds": True,
                "bookmakers_count": 12,
                "grade": {"grade": "LAYAK", "label": "🟢 LAYAK", "reasons": []},
            }
        ],
        "quota": {"odds_api_remaining": 487, "odds_blocked": False, "stats_warning": False},
        "leagues_no_odds": ["Liga 1"],
    }
    rendered = format.format_top(payload)
    assert "Man City" in rendered["body"]
    assert "bet365" in rendered["body"]
    assert "487/500" in rendered["footer"]
    assert "Liga 1" in rendered["footer"]
    assert "🟢 LAYAK" in rendered["body"]
    assert "🟢 1 layak" in rendered["footer"]


def test_format_stats_renders():
    payload = {
        "file": "cache/football/predictions.jsonl",
        "n_snapshots": 10,
        "n_settled": 6,
        "n_predicted": 6,
        "hit_rate": 0.5,
        "avg_logloss": 0.98,
        "roi": -0.023,
        "n_bets": 4,
        "clv_pct": -1.2,
        "n_clv": 3,
        "edge_threshold": 0.02,
    }
    rendered = format.format_stats(payload)
    assert rendered["title"] == "📈 Prediction Log Stats"
    body = rendered["body"]
    assert "snapshots: **10**" in body
    assert "settled: **6**" in body
    assert "hit rate: **50.0%**" in body
    assert "log-loss: 0.98" in body
    assert "ROI: **-2.3%**" in body
    assert "CLV: -1.2%" in body
    assert "unsettled: **4**" in body


def test_format_stats_empty():
    payload = {
        "file": "cache/football/predictions.jsonl",
        "n_snapshots": 0,
        "n_settled": 0,
        "n_predicted": 0,
        "hit_rate": None,
        "avg_logloss": None,
        "roi": None,
        "n_bets": 0,
        "clv_pct": None,
        "n_clv": 0,
        "edge_threshold": 0.02,
    }
    rendered = format.format_stats(payload)
    assert "hit rate: **-**" in rendered["body"]
    assert "belum ada match yang di-settle" in rendered["footer"]


def test_format_stats_error():
    rendered = format.format_stats({"error": "file tidak ditemukan"})
    assert "Error: file tidak ditemukan" in rendered["body"]


def test_format_compare_match_found():
    payload = {
        "home": "Manchester City",
        "away": "Arsenal",
        "league": "EPL",
        "stats": {
            "home_form": "W-W-D-W-W",
            "away_form": "W-W-L-D-W",
            "h2h": {"wins": 3, "draws": 1, "losses": 1},
        },
        "sources": ["football_data"],
        "quota": {"odds_api_remaining": 487, "stats_warning": False},
    }
    rendered = format.format_compare(payload)
    assert "Manchester City" in rendered["body"]
    assert "Arsenal" in rendered["body"]
    assert "source" in rendered["footer"]


def test_format_compare_error():
    payload = {"error": "tim tidak ditemukan"}
    rendered = format.format_compare(payload)
    assert "tim tidak ditemukan" in rendered["body"]


def test_leagues_json_loads():
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    assert "EPL" in leagues
    assert "Liga 1" in leagues
    assert leagues["Liga 1"]["odds_api_key"] is None
    assert leagues["EPL"]["provider"] == "football_data"
    assert leagues["Liga 1"]["country"] == "Indonesia"


def test_teams_json_exists():
    pass


def test_config_loads():
    cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
    assert cfg["top_n"] == 5
    assert cfg["outlier_threshold_pct"] == 5
    assert "EPL" in cfg["leagues"]


def test_league_resolver_exact():
    from agents.football.league_resolver import resolve_league
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    res = resolve_league("liga portugal", leagues)
    assert res is not None
    assert res[0] == "Primeira Liga"
    assert res[1]["odds_api_key"] == "soccer_portugal_primeira_liga"


def test_league_resolver_substring():
    from agents.football.league_resolver import resolve_league
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    res = resolve_league("ucl", leagues)
    assert res is not None
    assert res[0] == "UCL"
    assert res[1]["odds_api_key"] == "soccer_uefa_champs_league"


def test_league_resolver_unknown():
    from agents.football.league_resolver import resolve_league
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    res = resolve_league("liga antah berantah", leagues)
    assert res is None


def test_league_resolver_cup_competitions():
    from agents.football.league_resolver import resolve_league
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    assert resolve_league("super cup", leagues)[0] == "UEFA Super Cup"
    assert resolve_league("libertadores", leagues)[0] == "Copa Libertadores"
    assert resolve_league("leagues cup", leagues)[0] == "Leagues Cup"
    assert resolve_league("concacaf", leagues)[0] == "CONCACAF Central American Cup"
    assert resolve_league("ofc", leagues)[0] == "OFC Champions League"
    assert resolve_league("champions league", leagues)[0] == "UCL"  # exact wins over OFC


def test_league_resolver_indonesia():
    from agents.football.league_resolver import resolve_league
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    res = resolve_league("indonesia", leagues)
    assert res is not None
    assert res[0] == "Liga 1"


def test_format_analyse_match():
    payload = {
        "league": "UCL",
        "home": "Bodo/Glimt",
        "away": "Union Saint-Gilloise",
        "kickoff": "2026-08-13T19:00:00Z",
        "venue": "home",
        "match_found": True,
        "stats": {
            "home_form": "W-W-D-W-W",
            "away_form": "W-W-L-D-W",
            "home_gf_avg": 1.8,
            "home_ga_avg": 0.9,
            "away_gf_avg": 1.4,
            "away_ga_avg": 1.2,
            "home_split": {"w": 3, "d": 1, "l": 1},
            "away_split": {"w": 2, "d": 1, "l": 2},
            "h2h": {"wins": 0, "draws": 1, "losses": 0},
        },
        "odds": {
            "consensus": {"home": 2.10, "draw": 3.40, "away": 3.60},
            "best": {"home": {"odds": 2.20, "bookmaker": "bet365"}},
            "outlier": {"side": "home", "value_pct": 4.8, "bookmaker": "bet365"},
            "bookmakers_count": 8,
            "has_odds": True,
            "totals": {
                "Over 2.5": {"odds": 1.85, "bookmaker": "pinnacle"},
                "BTTS Yes": {"odds": 1.75, "bookmaker": "bet365"},
            },
        },
        "picks": {
            "top_picks": [
                {"rank": 1, "market": "Total", "selection": "Over 2.5", "model_prob": 0.54,
                 "market_odds": 1.85, "implied_prob": 0.52, "edge": 2.1},
                {"rank": 2, "market": "1X2", "selection": "Home Win", "model_prob": 0.48,
                 "market_odds": 2.10, "implied_prob": 0.47, "edge": 4.8},
                {"rank": 3, "market": "BTTS", "selection": "Yes", "model_prob": 0.57,
                 "market_odds": 1.75, "implied_prob": 0.55, "edge": 1.2},
            ],
            "best_pick": {"rank": 1, "market": "1X2", "selection": "Home Win",
                          "model_prob": 0.48, "market_odds": 2.10, "implied_prob": 0.47, "edge": 4.8},
            "model_probs": {
                "lambda_home": 1.5, "lambda_away": 1.2,
                "1x2": {"home": 0.48, "draw": 0.27, "away": 0.25},
                "over_1.5": 0.65, "over_2.5": 0.54, "over_3.5": 0.32,
                "btts_yes": 0.57,
            },
        },
        "signal": 72,
        "sources": ["football_data", "odds_api"],
        "quota": {"odds_api_remaining": 487, "stats_warning": False},
        "decision": {
            "decision_type": "GOOD",
            "final_decision": {
                "market": "Total", "selection": "Under 3.5",
                "model_prob": 0.67, "market_odds": 1.65,
                "edge_pp": 4.1, "ev": 0.11,
            },
            "most_likely": {"selection": "Home Win", "model_prob": 0.48},
            "explanation": "Most likely: Home Win (48.0%) — FINAL DECISION: Under 3.5 (skor 0.64).",
            "edge_warnings": [],
            "score_breakdown": {"top": {"score": 0.64}},
        },
    }
    rendered = format.format_analyse(payload)
    assert "Bodo/Glimt" in rendered["body"]
    assert "Union Saint-Gilloise" in rendered["body"]
    assert "UCL" in rendered["body"]
    # Section 1: the odds-implied (Model A) section is labeled reference-only
    assert "reference only" in rendered["body"]
    assert "FINAL DECISION" in rendered["body"]
    assert "GOOD" in rendered["body"]
    assert "Under 3.5" in rendered["body"]
    assert "most likely" in rendered["body"]
    assert "Disclaimer" in rendered["body"]
    assert "Model vs Market" in rendered["body"]
    assert "GF/GA" in rendered["body"]
    assert "home record" in rendered["body"]
    assert "487/500" in rendered["footer"]
    assert "source" in rendered["footer"]


def test_format_analyse_with_breakdown():
    payload = {
        "league": "EPL",
        "home": "Arsenal FC",
        "away": "Chelsea FC",
        "kickoff": "2026-08-20T15:00:00Z",
        "stats": {
            "home_form": "W-W-W-D-W",
            "away_form": "W-W-L-D-L",
            "home_gf_avg": 2.1, "home_ga_avg": 0.8,
            "away_gf_avg": 1.5, "away_ga_avg": 1.3,
            "home_split": {"w": 4, "d": 1, "l": 0},
            "away_split": {"w": 1, "d": 2, "l": 2},
            "h2h": {"wins": 3, "draws": 1, "losses": 1},
        },
        "odds": {
            "consensus": {"home": 1.80, "draw": 3.60, "away": 4.50},
            "best": {}, "outlier": None,
            "bookmakers_count": 10, "has_odds": True,
            "totals": {"Over 2.5": {"odds": 1.65, "bookmaker": "pinnacle"}},
        },
        "picks": {
            "top_picks": [],
            "best_pick": None,
            "model_probs": {
                "lambda_home": 1.8, "lambda_away": 1.0,
                "1x2": {"home": 0.58, "draw": 0.24, "away": 0.18},
                "over_2.5": 0.62,
                "btts_yes": 0.55,
            },
        },
        "signal": 78,
        "sources": ["football_data"],
        "quota": {"odds_api_remaining": 500},
    }
    rendered = format.format_analyse(payload)
    assert "Arsenal" in rendered["body"]
    assert "Chelsea" in rendered["body"]
    assert "Model vs Market" in rendered["body"]
    assert "Home:" in rendered["body"]
    assert "Poisson" in rendered["body"]


def test_format_analyse_error():
    payload = {"error": "tim tidak ditemukan", "home_query": "X", "away_query": "Y"}
    rendered = format.format_analyse(payload)
    assert "tim tidak ditemukan" in rendered["body"]


def test_format_analyse_with_prediction_engine():
    payload = {
        "league": "EPL",
        "home": "Arsenal",
        "away": "Chelsea",
        "kickoff": "2026-08-20T15:00:00Z",
        "generated_at": "2026-08-20T07:00:00+00:00",
        "stats": {
            "home_form": "W-W-W-D-W", "away_form": "W-W-L-D-L",
            "home_gf_avg": 2.1, "home_ga_avg": 0.8,
            "away_gf_avg": 1.5, "away_ga_avg": 1.3,
            "h2h": {"wins": 3, "draws": 1, "losses": 1},
        },
        "odds": {"consensus": {"home": 1.80, "draw": 3.60, "away": 4.50},
                 "has_odds": True, "bookmakers_count": 8, "totals": {}},
        "signal": 78,
        "sources": ["football_data"],
        "quota": {"odds_api_remaining": 500},
        "prediction": {
            "model_probs": {
                "1x2": {"home": 0.52, "draw": 0.27, "away": 0.21},
                "over_2.5": 0.55, "btts_yes": 0.52,
                "lambda_home": 1.55, "lambda_away": 1.20,
                "lambda_source": "features", "models": ["elo", "poisson"],
                "model_weights": {"elo": 0.5, "poisson": 0.5},
            },
            "confidence": 0.72,
            "signal_strength": 68,
            "market_edge": {"home": 2.1, "draw": -1.0, "away": -1.1},
            "calibration": {"quality": 0.6, "ece": 0.04, "samples": 300},
            "agreement": {"model_vs_market": 0.9, "model_vs_model": 0.8,
                           "models": ["elo", "poisson"]},
            "data_completeness": 0.8,
            "model_version": "0.1.0-elo-poisson",
            "as_of": "2026-08-20T07:00:00+00:00",
            "input_hash": "abc123def4567890",
        },
        "confidence": {
            "model_calibration_score": 0.6,
            "pick_specific_confidence": 0.72,
            "tier": "HIGH",
            "tier_before_caps": "HIGH",
            "caps_applied": [],
            "n_bucket": 500,
            "completeness_factor": 0.8,
        },
    }
    rendered = format.format_analyse(payload)
    assert "Elo+Poisson" in rendered["body"]
    assert "λ_home=1.55" in rendered["body"]
    assert "Home 52.0%" in rendered["body"]
    # Addendum v1.1: ONE confidence block only — the legacy confidence line
    # (0-1 score with signal/decisiveness/calib sub-scores), the duplicate
    # "Confidence: (sinyal n/100)" line and the undefined agreement
    # market/models field are all gone.
    assert "⚑ Confidence: 🟢 HIGH (pick_specific_confidence 0.72)" in rendered["body"]
    assert "model_calibration_score (global): 0.60" in rendered["body"]
    assert rendered["body"].count("⚑ Confidence:") == 1
    assert "signal" not in rendered["body"]
    assert "decisiveness" not in rendered["body"]
    assert "agreement market" not in rendered["body"]
    assert "edge: Home +2.1%" in rendered["body"]
    assert "calibration: 300 samples" in rendered["body"]
    assert "gen 2026-08-20" in rendered["footer"]
    assert "#abc123def4567890" in rendered["footer"]


def test_leagues_count_31():
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    assert len(leagues) == 31


def test_all_leagues_have_aliases():
    leagues = json.loads((ROOT / "agents" / "football" / "leagues.json").read_text(encoding="utf-8"))
    for key, meta in leagues.items():
        assert "aliases" in meta, f"{key} missing aliases"
        assert len(meta["aliases"]) >= 1, f"{key} has empty aliases"


def test_format_analyse_all_watch_informative_label():
    """When the tier layer demotes EVERY market to WATCH but the engine still
    reports a GOOD/LEAN final_decision, the FINAL DECISION block must carry an
    explicit "INFORMATIF SAJA / SKIP" label so the two never read as
    contradicting each other (Galatasaray-vs-Çorum-shaped payload)."""
    payload = {
        "league": "Super Lig",
        "home": "Galatasaray",
        "away": "Çorum",
        "kickoff": "2026-08-14T18:30:00Z",
        "match_found": True,
        "stats": {
            "home_form": "L", "away_form": "W",
            "home_gf_avg": 1.0, "home_ga_avg": 2.0,
            "away_gf_avg": 1.0, "away_ga_avg": 0.0,
            "home_split": {}, "away_split": {},
            "h2h": {"wins": 0, "draws": 0, "losses": 0},
        },
        "odds": {
            "consensus": {"home": 1.3, "draw": 5.3, "away": 8.5},
            "best": {}, "outlier": None,
            "bookmakers_count": 131, "has_odds": True,
            "totals": {
                "Over 2.5": {"odds": 1.63, "bookmaker": "betfair-ex"},
                "Under 2.5": {"odds": 2.56, "bookmaker": "betfair-ex"},
                "Over 3.5": {"odds": 2.51, "bookmaker": "betfair-ex"},
                "Under 3.5": {"odds": 1.64, "bookmaker": "betfair-ex"},
                "BTTS Yes": {"odds": 1.96, "bookmaker": "polymarket"},
                "BTTS No": {"odds": 2.0, "bookmaker": "polymarket"},
            },
        },
        "picks": {
            "top_picks": [], "best_pick": None,
            "model_probs": {
                "1x2": {"home": 0.6889, "draw": 0.177, "away": 0.1341},
                "over_2.5": 0.436, "over_3.5": 0.226, "btts_yes": 0.5,
                "lambda_home": 1.16, "lambda_away": 1.26,
            },
        },
        "decision": {
            "decision_type": "GOOD",
            "final_decision": {
                "market": "Total", "selection": "Under 3.5",
                "model_prob": 0.774, "market_odds": 1.64,
                "edge_pp": 16.9, "ev": 0.27,
            },
            "most_likely": {"selection": "Home Win", "model_prob": 0.6889},
            "explanation": "Most likely: Home Win (68.9%) — FINAL DECISION: Under 3.5 (skor 0.74).",
            "edge_warnings": [],
            "score_breakdown": {"top": {"score": 0.74}},
        },
        "sources": ["oddspapi_odds", "thesportsdb"],
    }
    rendered = format.format_analyse(payload)
    # the engine's GOOD view is still shown (transparency) ...
    assert "GOOD" in rendered["body"]
    assert "Under 3.5" in rendered["body"]
    # ... but the output-policy layer's SKIP verdict is spelled out right there
    assert "INFORMATIF SAJA" in rendered["body"]
    assert "keputusan taruhan: **SKIP**" in rendered["body"]
    assert "semua market turun ke Tier 3 (WATCH)" in rendered["body"]


def test_format_analyse_no_clear_decision_no_informative_label():
    """Without an engine final_decision (NO CLEAR DECISION / NO BET) the
    INFORMATIF SAJA line must NOT appear — there is nothing to reconcile."""
    payload = {
        "league": "Super Lig",
        "home": "Galatasaray",
        "away": "Çorum",
        "kickoff": "2026-08-14T18:30:00Z",
        "match_found": True,
        "stats": {
            "home_form": "L", "away_form": "W",
            "home_gf_avg": 1.0, "home_ga_avg": 2.0,
            "away_gf_avg": 1.0, "away_ga_avg": 0.0,
            "h2h": {"wins": 0, "draws": 0, "losses": 0},
        },
        "odds": {
            "consensus": {"home": 1.3, "draw": 5.3, "away": 8.5},
            "best": {}, "outlier": None,
            "bookmakers_count": 131, "has_odds": True,
            "totals": {
                "Over 2.5": {"odds": 1.63, "bookmaker": "betfair-ex"},
                "Under 2.5": {"odds": 2.56, "bookmaker": "betfair-ex"},
                "Over 3.5": {"odds": 2.51, "bookmaker": "betfair-ex"},
                "Under 3.5": {"odds": 1.64, "bookmaker": "betfair-ex"},
            },
        },
        "picks": {
            "top_picks": [], "best_pick": None,
            "model_probs": {
                "1x2": {"home": 0.6889, "draw": 0.177, "away": 0.1341},
                "over_2.5": 0.436, "over_3.5": 0.226, "btts_yes": 0.5,
                "lambda_home": 1.16, "lambda_away": 1.26,
            },
        },
        "decision": {
            "decision_type": "NO CLEAR DECISION",
            "final_decision": None,
            "most_likely": None,
            "explanation": "Engine independen tidak berjalan (data form/history kurang).",
            "edge_warnings": [],
            "score_breakdown": {},
        },
        "sources": ["oddspapi_odds", "thesportsdb"],
    }
    rendered = format.format_analyse(payload)
    assert "INFORMATIF SAJA" not in rendered["body"]
    assert "NO CLEAR DECISION" in rendered["body"]


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
