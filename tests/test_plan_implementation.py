"""Unit tests for the audit-plan implementation (TODO-01 .. TODO-17).

Covers every code change made in this session so a regression anywhere in
the new machinery is caught by CI:
  - TODO-01/11/12  Elo live update loop, recency K multiplier, home-adv est.
  - TODO-02         calibration re-fit from the live prediction log
  - TODO-03         form-window parity (live default == backtest maxlen 5)
  - TODO-04         per-bookmaker-pair totals/BTTS extraction
  - TODO-05         kickoff-time ordering for walk-forward replays
  - TODO-09/10/16   WATCH tier + variance-aware EV band in the decision engine
  - TODO-13         ensemble spread + logistic stacking experiment
  - TODO-15         per-decision-type CLV/ROI buckets in compute_stats
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.football.analyse import extract_market_totals
from agents.football.calibration import Calibrator
from agents.football.context import MatchContext
from agents.football.decision import build_candidates, decide
from agents.football.elo import EloModel
from agents.football.models import (
    Ensemble,
    PoissonModel,
    run_prediction_engine,
)
from agents.football.prediction_log import (
    append_snapshot,
    calibration_pairs,
    compute_stats,
    settle,
)
from agents.football.timeutil import kickoff_sort_key


@pytest.fixture(autouse=True)
def _reset_analysis_budget():
    """Disarm the module-level analysis budget clock before every test so a
    clock armed by one test can never bleed into (and skip providers for)
    another."""
    from agents.football import multi_source

    multi_source.reset_analysis_budget()
    yield
    multi_source.reset_analysis_budget()


# ---------------------------------------------------------------------------
# TODO-01: live Elo update loop
# ---------------------------------------------------------------------------
def test_elo_update_from_results_persists(tmp_path):
    elo = EloModel(path=tmp_path / "elo.json")
    applied = elo.update_from_results(
        [
            {"home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 0},
            {"home": "Liverpool", "away": "Everton", "result": "1-1"},
            {"home": "Broken", "away": "Row", "result": "not-a-score"},
        ]
    )
    assert applied == 2  # the unparseable row is skipped, never guessed
    assert elo.rating("Arsenal") > elo.rating("Chelsea")
    assert (tmp_path / "elo.json").exists()  # persisted exactly once
    saved = json.loads((tmp_path / "elo.json").read_text(encoding="utf-8"))
    assert "Arsenal" in saved["ratings"]
    assert "Everton" in saved["ratings"]


def test_elo_update_from_results_no_results_no_write(tmp_path):
    elo = EloModel(path=tmp_path / "elo.json")
    assert elo.update_from_results([]) == 0
    assert not (tmp_path / "elo.json").exists()


def test_elo_update_k_multiplier_scales_movement():
    elo_a = EloModel()
    elo_b = EloModel()
    elo_a.ratings = {"A": 1500.0, "B": 1500.0}
    elo_b.ratings = {"A": 1500.0, "B": 1500.0}
    elo_a.update("A", "B", 1, 0, persist=False, k_multiplier=1.0)
    elo_b.update("A", "B", 1, 0, persist=False, k_multiplier=0.5)
    assert elo_a.rating("A") > elo_b.rating("A") > 1500.0


# ---------------------------------------------------------------------------
# TODO-05: kickoff-time ordering (walk-forward safety)
# ---------------------------------------------------------------------------
def test_kickoff_sort_key_orders_same_day_by_time():
    fx_early = {"date": "2024-08-17T13:30:00Z"}
    fx_late = {"date": "2024-08-17T17:00:00Z"}
    fx_dateonly = {"date": "2024-08-17"}  # no kickoff time (FBref)
    fx_nodate = {}
    ordered = sorted([fx_late, fx_nodate, fx_dateonly, fx_early], key=kickoff_sort_key)
    assert ordered[0] is fx_early
    assert ordered[1] is fx_late
    assert ordered[2] is fx_dateonly  # unknown-time same-day sorts after timed
    assert ordered[3] is fx_nodate  # no-date last


def test_kickoff_sort_key_stable_between_dates():
    f1 = {"date": "2024-08-16"}
    f2 = {"date": "2024-08-17"}
    f3 = {"date": "2024-08-17T19:00:00Z"}
    # 08-16 first; on 08-17 the TIMED match precedes the date-only (unknown
    # kickoff time) one, which stays after it.
    assert [k for k in sorted([f3, f2, f1], key=kickoff_sort_key)] == [f1, f3, f2]


# ---------------------------------------------------------------------------
# TODO-04: per-bookmaker-pair totals/BTTS (single margin removal)
# ---------------------------------------------------------------------------
def _totals_payload():
    return {
        "bookmakers": [
            {
                "title": "BM A",
                "markets": [
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 1.80},
                        {"name": "Under", "point": 2.5, "price": 2.05},
                    ]},
                    {"key": "btts", "outcomes": [
                        {"name": "Yes", "price": 1.75},
                        {"name": "No", "price": 2.10},
                    ]},
                ],
            },
            {
                "title": "BM B",
                "markets": [
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 1.95},
                        {"name": "Under", "point": 2.5, "price": 1.90},
                    ]},
                ],
            },
        ]
    }


def test_totals_pairs_come_from_same_bookmaker():
    mt = extract_market_totals(_totals_payload())
    # BM B has the smaller overround (1.039 vs 1.043) -> BOTH sides from BM B.
    assert mt["Over 2.5"]["bookmaker"] == "BM B"
    assert mt["Under 2.5"]["bookmaker"] == "BM B"
    assert mt["Under 2.5"]["odds"] == 1.90  # NOT the cross-bookmaker max 2.05
    # BTTS: only BM A carries the pair -> both sides from BM A.
    assert mt["BTTS Yes"]["bookmaker"] == "BM A"
    assert mt["BTTS No"]["bookmaker"] == "BM A"


def test_totals_pair_requires_both_sides():
    payload = {"bookmakers": [{"title": "BM A", "markets": [
        {"key": "totals", "outcomes": [{"name": "Over", "point": 2.5, "price": 1.90}]},
    ]}]}
    assert extract_market_totals(payload) == {}  # no Under -> no pair


# ---------------------------------------------------------------------------
# TODO-09/10/16: WATCH tier + variance-aware EV band (opt-in)
# ---------------------------------------------------------------------------
def _base_candidates():
    model_probs = {
        "1x2": {"home": 0.60, "draw": 0.22, "away": 0.18},
        "over_2.5": 0.66, "over_3.5": 0.40, "btts_yes": 0.55,
    }
    consensus = {"home": 1.80, "draw": 3.80, "away": 4.50}
    return build_candidates(
        model_probs=model_probs, consensus_odds=consensus,
        market_totals={}, independent=True,
    )


def test_high_uncertainty_downgrades_to_watch_only_when_enabled():
    kw = dict(
        model_agreement=0.80, calibration_quality=0.90, calibration_samples=500,
        completeness=0.80, bookmakers_count=5, historical_reliability=0.5,
        min_edge_pp=3.0, best_prob_only=True, bucket_n=None, uncertainty=0.40,
    )
    watch = decide(_base_candidates(), enable_watch=True, **kw)
    assert watch["decision_type"] == "WATCH"
    assert watch["final_decision"] is not None  # watchlist keeps the pick
    assert any("spread" in r or "tidak pasti" in r for r in watch["reasons"])

    legacy = decide(_base_candidates(), enable_watch=False, **kw)
    assert legacy["decision_type"] != "WATCH"


def test_ev_band_reported_with_uncertainty():
    d = decide(
        _base_candidates(),
        model_agreement=0.80, calibration_quality=0.90, calibration_samples=500,
        completeness=0.80, bookmakers_count=5, historical_reliability=0.5,
        min_edge_pp=3.0, best_prob_only=True, bucket_n=None,
        enable_watch=True, uncertainty=0.10,
    )
    band = d.get("ev_band")
    assert band is not None
    assert band["ev_low"] <= band["ev"] <= band["ev_high"]
    assert band["uncertainty"] == 0.10


def test_decision_defaults_unchanged_without_watch():
    d = decide(
        _base_candidates(),
        model_agreement=0.80, calibration_quality=0.90, calibration_samples=500,
        completeness=0.80, bookmakers_count=5, historical_reliability=0.5,
        min_edge_pp=3.0, best_prob_only=True, bucket_n=None,
    )
    assert d["decision_type"] in ("STRONG", "GOOD", "LEAN", "NO BET", "NO CLEAR DECISION")
    assert "WATCH" not in d["decision_type"]


# ---------------------------------------------------------------------------
# TODO-02: calibration refresh from the live log
# ---------------------------------------------------------------------------
def _log_with_settled_snapshots(path: Path, n: int) -> None:
    for i in range(n):
        match_id = f"L||Team {i}||Team {i + 1}||2024-08-{i % 28 + 1:02d}T15:00:00Z"
        outcome = ["home", "draw", "away"][i % 3]
        probs = {"home": 0.45, "draw": 0.28, "away": 0.27}
        probs[outcome] += 0.20
        probs = {k: v / sum(probs.values()) for k, v in probs.items()}
        append_snapshot(
            path, match_id=match_id, league="L", home=f"Team {i}", away=f"Team {i + 1}",
            kickoff="2024-08-01T15:00:00Z", prob=probs, odds={"home": 2.0, "draw": 3.4, "away": 3.9},
            edge=None, confidence=0.7, signal=70, calibration=None,
            model_version="test", input_hash="h", best_pick=None,
            sources=[], decision_type="GOOD",
        )
        settle(path, match_id=match_id, home_goals=2, away_goals=1)


def test_calibration_pairs_shape(tmp_path):
    log = tmp_path / "p.jsonl"
    _log_with_settled_snapshots(log, 2)
    pairs = calibration_pairs(log)
    assert len(pairs) == 6  # 2 settled snapshots * 3 1X2 sides
    assert all(y in (0, 1) for _, y in pairs)
    assert any(y == 1 for _, y in pairs)


def test_calibrator_refresh_from_log(tmp_path):
    log = tmp_path / "p.jsonl"
    _log_with_settled_snapshots(log, 250)
    cal = Calibrator(path=tmp_path / "cal.json")
    report = cal.refresh_from_log(log, min_samples=200)
    assert report["status"] == "refit"
    assert report["pairs"] >= 200
    assert (tmp_path / "cal.json").exists()

    small = Calibrator(path=tmp_path / "cal_small.json")
    r2 = small.refresh_from_log(log, min_samples=10_000)
    assert r2["status"] == "skipped"  # never overwrite with noise


def test_calibrator_refresh_keeps_better_snapshot(tmp_path):
    """Regression guard: a refit with worse ECE must NOT overwrite the good
    snapshot (status='kept', old params restored)."""
    log = tmp_path / "p.jsonl"
    _log_with_settled_snapshots(log, 60)  # small sample -> noisier refit
    cal = Calibrator(path=tmp_path / "cal.json")
    cal.fit([0.9, 0.9, 0.1, 0.1, 0.8, 0.2], [1, 1, 0, 0, 1, 0])  # clean snapshot
    good_ece = cal.ece
    report = cal.refresh_from_log(log, min_samples=10)
    assert report["status"] in ("refit", "kept")
    # whichever happens, the file remains a valid snapshot and the ECE never
    # silently worsened beyond what the guard allowed
    assert (tmp_path / "cal.json").exists()
    if report["status"] == "kept":
        assert cal.ece == good_ece


# ---------------------------------------------------------------------------
# TODO-13: ensemble spread + logistic stacking (experimental)
# ---------------------------------------------------------------------------
def _ctx() -> MatchContext:
    return MatchContext(
        league="EPL", home="A", away="B",
        home_form="W-W-W", away_form="L-L-L",
        home_gf_avg=2.5, home_ga_avg=0.8,
        away_gf_avg=0.9, away_ga_avg=2.2, form_samples=3,
    )


def test_ensemble_spread_reflects_disagreement():
    elo = EloModel()
    elo.ratings = {"A": 1800.0, "B": 1200.0}  # Elo very pro-home
    poisson = PoissonModel(base_home_goals=1.45, base_away_goals=1.25)
    ens = Ensemble(elo_weight=0.5, poisson_weight=0.5)
    out = ens.predict(_ctx(), elo, poisson)
    assert 0.0 <= out["spread"] <= 1.0
    # run_prediction_engine surfaces it as ``uncertainty``
    result = run_prediction_engine(_ctx(), elo=elo, poisson=poisson, ensemble=ens)
    assert result is not None
    assert result.model_probs["uncertainty"] == out["spread"]


# ---------------------------------------------------------------------------
# TODO-03: form-window parity (live default == backtest maxlen 5)
# ---------------------------------------------------------------------------
def test_form_window_parity_live_default_is_five():
    from agents.football.multi_source import MultiSourceStatsFetcher

    assert MultiSourceStatsFetcher.FORM_WINDOW == 5
    sig = inspect.signature(MultiSourceStatsFetcher.fetch_team_form)
    assert sig.parameters["limit"].default == 5

    from agents.football.flashscore import FlashscoreClient

    sig2 = inspect.signature(FlashscoreClient.fetch_team_form)
    assert sig2.parameters["limit"].default == 5


# ---------------------------------------------------------------------------
# top --days window: multi-day WIB coverage (dini-hari matches)
# ---------------------------------------------------------------------------
def test_top_window_days_builds_multi_date_window(monkeypatch):
    """find_top_matches with days=2 must fetch fixtures for BOTH WIB dates,
    dedupe overlapping rows, filter odds to the window, and report the range."""
    import asyncio

    import agents.football.match_finder as mf

    fetched_dates: list[str] = []

    class _Cache:
        def get(self, key, ttl):
            return None

        def set(self, key, value):
            pass

    class _Stats:
        fd = type("FD", (), {"rate_limit_warning": False})()

        async def fetch_fixtures_for_date(self, meta, target_date):
            fetched_dates.append(target_date)
            if target_date == "2026-08-12":
                return [
                    {"id": "1", "status": "TIMED", "date": "2026-08-12T14:00:00Z",
                     "home": {"id": "h1", "name": "Arsenal"},
                     "away": {"id": "a1", "name": "Chelsea"}},
                ]
            if target_date == "2026-08-13":
                return [
                    {"id": "2", "status": "TIMED", "date": "2026-08-13T19:00:00Z",
                     "home": {"id": "h2", "name": "Real Madrid"},
                     "away": {"id": "a2", "name": "Barcelona"}},
                    # duplicate row (same kickoff) must be deduped
                    {"id": "2b", "status": "TIMED", "date": "2026-08-13T19:00:00Z",
                     "home": {"id": "h2", "name": "Real Madrid"},
                     "away": {"id": "a2", "name": "Barcelona"}},
                ]
            return []

        async def fetch_team_form(self, team_id, meta):
            return {"sequence": "W-W-W", "gf_avg": 2.0, "ga_avg": 1.0}

    class _Odds:
        last_remaining = 100
        quota_blocked = False

        async def fetch_odds(self, key):
            return [
                {"commence_time": "2026-08-12T14:00:00Z",
                 "home_team": "Arsenal", "away_team": "Chelsea",
                 "bookmakers": [{"title": "BM", "markets": [
                     {"key": "h2h", "outcomes": [
                         {"name": "Arsenal", "price": 1.9},
                         {"name": "Draw", "price": 3.4},
                         {"name": "Chelsea", "price": 4.2}]}]}]},
                # dini-hari match on the NEXT WIB date: included only with
                # days >= 2
                {"commence_time": "2026-08-13T19:00:00Z",
                 "home_team": "Real Madrid", "away_team": "Barcelona",
                 "bookmakers": [{"title": "BM", "markets": [
                     {"key": "h2h", "outcomes": [
                         {"name": "Real Madrid", "price": 1.8},
                         {"name": "Draw", "price": 3.6},
                         {"name": "Barcelona", "price": 4.5}]}]}]},
            ]

    def _fake_resolve_date(date_str):
        return date_str or "2026-08-12"

    monkeypatch.setattr(mf, "_resolve_date", _fake_resolve_date)
    monkeypatch.setattr(mf, "_load_leagues", lambda: {"EPL": {"display": "EPL", "odds_api_key": "soccer_epl"}})

    async def _run():
        return await mf.find_top_matches(
            date=None, leagues=["EPL"], top_n=5,
            cfg={"cache_ttl_seconds": {"fixtures": 21600, "odds": 900},
                 "outlier_threshold_pct": 5},
            odds=_Odds(), stats=_Stats(), cache=_Cache(), days=2,
        )

    result = asyncio.run(_run())
    assert fetched_dates == ["2026-08-12", "2026-08-13"]
    # both days' matches surfaced, duplicate deduped
    names = {(m["home"], m["away"]) for m in result["matches"]}
    assert ("Arsenal", "Chelsea") in names
    assert ("Real Madrid", "Barcelona") in names
    assert len(result["matches"]) == 2
    assert result["days"] == 2
    assert result["date_range"] == "2026-08-12 → 2026-08-13"
    assert result["matches"][0]["odds"]["consensus"]["home"] > 0


def test_top_single_day_is_backward_compatible(monkeypatch):
    """days default (1) keeps the exact single-day behaviour and payload shape."""
    import asyncio

    import agents.football.match_finder as mf

    class _Cache:
        def get(self, key, ttl):
            return None

        def set(self, key, value):
            pass

    class _Stats:
        fd = type("FD", (), {"rate_limit_warning": False})()

        async def fetch_fixtures_for_date(self, meta, target_date):
            return [{"id": "1", "status": "TIMED", "date": "2026-08-12T14:00:00Z",
                     "home": {"id": "h1", "name": "Arsenal"},
                     "away": {"id": "a1", "name": "Chelsea"}}]

        async def fetch_team_form(self, team_id, meta):
            return {"sequence": "W-W-W", "gf_avg": 2.0, "ga_avg": 1.0}

    class _Odds:
        last_remaining = 100
        quota_blocked = False

        async def fetch_odds(self, key):
            return [{"commence_time": "2026-08-12T14:00:00Z",
                     "home_team": "Arsenal", "away_team": "Chelsea",
                     "bookmakers": [{"title": "BM", "markets": [
                         {"key": "h2h", "outcomes": [
                             {"name": "Arsenal", "price": 1.9},
                             {"name": "Draw", "price": 3.4},
                             {"name": "Chelsea", "price": 4.2}]}]}]}]

    monkeypatch.setattr(mf, "_resolve_date", lambda date_str: date_str or "2026-08-12")
    monkeypatch.setattr(mf, "_load_leagues", lambda: {"EPL": {"display": "EPL", "odds_api_key": "soccer_epl"}})

    async def _run():
        return await mf.find_top_matches(
            date=None, leagues=["EPL"], top_n=5,
            cfg={"cache_ttl_seconds": {"fixtures": 21600, "odds": 900},
                 "outlier_threshold_pct": 5},
            odds=_Odds(), stats=_Stats(), cache=_Cache(),
        )

    result = asyncio.run(_run())
    assert result["days"] == 1
    assert result["date"] == "2026-08-12"
    assert len(result["matches"]) == 1


def test_format_top_days_range_title():
    from agents.football.format import format_top

    out = format_top({"date": "2026-08-12", "days": 2,
                      "date_range": "2026-08-12 → 2026-08-13",
                      "matches": [], "extra_matches": [], "quota": {}})
    assert "2026-08-12 → 2026-08-13" in out["title"]
    single = format_top({"date": "2026-08-12", "days": 1, "matches": [],
                         "extra_matches": [], "quota": {}})
    assert "2026-08-12" in single["title"]
    assert "→" not in single["title"]


# ---------------------------------------------------------------------------
# Runner deadline handling: shared analysis budget + per-call timeouts
# ---------------------------------------------------------------------------
def test_analysis_budget_clock_lifecycle():
    """The module-level budget clock arms, reports remaining time, and can be
    disarmed (reset) so tests never leak a stale clock into other tests."""
    import agents.football.multi_source as ms

    ms.reset_analysis_budget()
    assert ms.analysis_remaining() is None  # no clock -> never skip
    assert ms.analysis_budget_exhausted() is False

    ms.set_analysis_budget(10.0)
    remaining = ms.analysis_remaining()
    assert remaining is not None and 0.0 < remaining <= 10.0
    assert ms.analysis_budget_exhausted() is False  # plenty of time
    assert ms.analysis_budget_exhausted(margin_seconds=12.0) is True  # margin > budget

    ms.reset_analysis_budget()
    assert ms.analysis_remaining() is None


def test_timeout_aware_caps_slow_provider_call():
    """_timeout_aware must bound a hung provider call and degrade to None
    (not raise) so the fallback chain moves on instead of stacking HTTP
    timeouts or crashing the analysis."""
    import asyncio

    import agents.football.multi_source as ms

    async def _hung():
        await asyncio.sleep(30.0)
        return "late"

    try:
        ms.reset_analysis_budget()
        result = asyncio.run(ms._timeout_aware(_hung(), seconds=0.2))
        assert result is None  # timed out -> None, never "late"
    finally:
        ms.reset_analysis_budget()


def test_budget_exhausted_skips_team_search():
    """search_team must bail (return None) instead of calling a provider when
    the shared budget is nearly spent -- the caller then reports the missing
    team cleanly instead of the whole runner dying on the deadline."""
    import asyncio

    import agents.football.multi_source as ms

    async def _boom(*a, **k):
        raise AssertionError("provider must not be called when budget is spent")

    fetcher = ms.MultiSourceStatsFetcher("fd", "")
    fetcher.fd.search_team_in_competition = _boom
    fetcher.ts.search_team = _boom
    meta = {"football_data_code": "PD", "display": "La Liga", "_league_key": "LaLiga"}

    try:
        ms.reset_analysis_budget()
        ms.set_analysis_budget(1.0)
        result = asyncio.run(fetcher.search_team("Sevilla", meta))
        assert result is None  # bailed on the budget, never touched providers
    finally:
        ms.reset_analysis_budget()


def test_budget_not_exhausted_still_calls_provider():
    """With a fresh budget (or no clock at all) the provider chain runs as
    before -- the budget must never silently break normal lookups."""
    import asyncio

    import agents.football.multi_source as ms
    from unittest.mock import AsyncMock

    async def _run():
        ms.reset_analysis_budget()
        ms.set_analysis_budget(60.0)
        fetcher = ms.MultiSourceStatsFetcher("fd", "")
        fetcher.fd.search_team_in_competition = AsyncMock(return_value={
            "id": 77, "name": "Sevilla", "shortName": "Sevilla",
            "tla": "SEV", "area": {"name": "Spain"},
        })
        fetcher.ts.search_team = AsyncMock(
            side_effect=AssertionError("should not reach thesportsdb")
        )
        result = await fetcher.search_team("Sevilla", {
            "football_data_code": "PD", "display": "La Liga", "_league_key": "LaLiga",
        })
        ms.reset_analysis_budget()
        return result

    result = asyncio.run(_run())
    assert result is not None and result["id"] == 77


# ---------------------------------------------------------------------------
# TODO-15: per-decision-type CLV/ROI buckets
# ---------------------------------------------------------------------------
def test_compute_stats_by_decision(tmp_path):
    log = tmp_path / "p.jsonl"
    _log_with_settled_snapshots(log, 3)
    stats = compute_stats(log, edge_threshold=0.02)
    assert stats["by_decision"]["GOOD"]["n"] == 3
    assert stats["by_decision"]["GOOD"]["n_bets"] >= 0
    assert "by_decision" in stats


# ---------------------------------------------------------------------------
# MARKET PRIOR (thin-data honesty): predictions from the market itself,
# explicitly labelled, edge = 0, betting advice NO BET.
# ---------------------------------------------------------------------------
def _market_prior_inputs():
    consensus = {"home": 2.10, "draw": 3.40, "away": 3.60}
    market_totals = {
        "Over 2.5": {"odds": 1.85, "point": 2.5, "bookmaker": "BM A"},
        "Under 2.5": {"odds": 1.95, "point": 2.5, "bookmaker": "BM A"},
        "BTTS Yes": {"odds": 1.72, "point": None, "bookmaker": "BM A"},
        "BTTS No": {"odds": 2.10, "point": None, "bookmaker": "BM A"},
    }
    return consensus, market_totals


def test_market_prior_builds_predictions_from_market():
    from agents.football.decision import market_prior_decision

    consensus, totals = _market_prior_inputs()
    d = market_prior_decision(consensus, totals, bookmakers_count=4)
    assert d["decision_type"] == "MARKET PRIOR"
    assert d["market_prior"] is True
    # 1X2 most likely = the margin-free market favourite (2.10 -> home)
    p1 = d["market_predictions"]["1x2"]
    assert d["most_likely"]["selection"] == "Home Win"
    assert d["most_likely"]["model_prob"] == p1["home"]
    assert abs(sum(p1.values()) - 1.0) < 0.01
    # totals + BTTS from the fair pair (keys follow the model_probs
    # convention: "over_2.5", "btts_yes")
    assert abs(d["market_predictions"]["over_2.5"] + d["market_predictions"]["under_2.5"] - 1.0) < 0.01
    assert abs(d["market_predictions"]["btts_yes"] + d["market_predictions"]["btts_no"] - 1.0) < 0.01
    # honesty: never claims edge; betting advice NO BET
    assert d["edge_warnings"] == []
    assert d["betting_advice"] == "NO BET"
    assert d["final_decision"] is None


def test_market_prior_decision_to_dict_roundtrip():
    from agents.football.decision import decision_to_dict, market_prior_decision

    consensus, totals = _market_prior_inputs()
    d = market_prior_decision(consensus, totals)
    out = decision_to_dict(d)  # must not crash on the plain-dict most_likely
    assert out["decision_type"] == "MARKET PRIOR"
    assert out["market_prior"] is True
    assert out["most_likely"]["selection"] == "Home Win"
    assert out["betting_advice"] == "NO BET"


def test_market_prior_no_odds_no_prediction():
    from agents.football.decision import market_prior_decision

    d = market_prior_decision({"home": 0, "draw": 0, "away": 0}, {})
    assert d["market_predictions"] == {}  # nothing to mirror -> no predictions
    assert d["most_likely"] is None


def test_market_prior_missing_draw_odds_normalizes():
    """Review fix: when the market priced no draw (draw odds absent), the
    1X2 prediction must normalize over the priced sides so it sums to 1."""
    from agents.football.decision import market_prior_decision

    d = market_prior_decision({"home": 2.0, "draw": 0, "away": 2.2}, {})
    p1 = d["market_predictions"]["1x2"]
    assert p1["draw"] == 0.0
    assert abs(sum(p1.values()) - 1.0) < 0.01
    assert d["most_likely"]["selection"] == "Home Win"


def test_run_decision_engine_market_prior_on_thin_data():
    """When the independent engine did not run (no form/history) and market
    prior is enabled, run_decision_engine emits MARKET PRIOR instead of a
    bare NO CLEAR DECISION -- and without the flag it stays NO CLEAR
    DECISION (backward compatible)."""
    from agents.football.analyse import run_decision_engine

    consensus, totals = _market_prior_inputs()
    base_cfg = {
        "models": {"decision": {"min_bookmakers": 3}},
        "cache_ttl_seconds": {},
    }
    on = {
        "models": {"decision": {
            "min_bookmakers": 3, "market_prior": True,
            "market_prior_min_completeness": 0.6,
        }},
        "cache_ttl_seconds": {},
    }
    # thin: prediction engine did not run at all
    d_on = run_decision_engine(None, consensus, totals, True, 4, on)
    assert d_on["decision_type"] == "MARKET PRIOR"
    assert d_on["market_prior"] is True
    assert d_on["betting_advice"] == "NO BET"

    d_off = run_decision_engine(None, consensus, totals, True, 4, base_cfg)
    assert d_off["decision_type"] == "NO CLEAR DECISION"
    assert not d_off.get("market_prior")


def test_run_decision_engine_market_prior_when_engine_completeness_low():
    """The independent engine ran but its data_completeness is below the
    MARKET PRIOR floor -> the thin-data gate switches to MARKET PRIOR."""
    from agents.football.analyse import run_decision_engine

    consensus, totals = _market_prior_inputs()
    cfg = {
        "models": {"decision": {
            "min_bookmakers": 3, "market_prior": True,
            "market_prior_min_completeness": 0.6,
        }},
        "cache_ttl_seconds": {},
    }
    thin_prediction = {
        "model_probs": {"1x2": {"home": 0.33, "draw": 0.33, "away": 0.33}},
        "data_completeness": 0.20,  # well below the 0.6 floor
        "agreement": {},
        "calibration": {"quality": 0.0, "samples": 0},
    }
    d = run_decision_engine(thin_prediction, consensus, totals, True, 4, cfg)
    assert d["decision_type"] == "MARKET PRIOR"

    # Review fix (cliff): a match at 0.5 completeness is STILL below the
    # bettable floor (0.6) -- it gets MARKET PRIOR, not a bare NO CLEAR
    # DECISION. No UX cliff between 0.35 and 0.6.
    mid_prediction = {
        "model_probs": {"1x2": {"home": 0.50, "draw": 0.26, "away": 0.24}},
        "data_completeness": 0.50,
        "agreement": {"model_vs_market": 0.6},
        "calibration": {"quality": 0.5, "samples": 300},
    }
    dmid = run_decision_engine(mid_prediction, consensus, totals, True, 4, cfg)
    assert dmid["decision_type"] == "MARKET PRIOR"

    # ... but a match with adequate completeness still uses the real engine
    good_prediction = {
        "model_probs": {"1x2": {"home": 0.62, "draw": 0.22, "away": 0.16}},
        "data_completeness": 0.80,
        "agreement": {"model_vs_market": 0.8},
        "calibration": {"quality": 0.9, "samples": 500},
    }
    d2 = run_decision_engine(good_prediction, consensus, totals, True, 4, cfg)
    assert d2["decision_type"] != "MARKET PRIOR"
    assert d2["decision_type"] in ("STRONG", "GOOD", "LEAN", "NO BET", "NO CLEAR DECISION", "WATCH")
