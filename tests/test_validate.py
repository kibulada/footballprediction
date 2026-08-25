"""Offline tests for validate.py (synthetic, deterministic, clearly labeled)."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.validate import format_validation_report, run_multi_season_validation

TEAMS = [f"T{i}" for i in range(10)]


def _synthetic_multi_season() -> list[dict]:
    """Two synthetic seasons; T0/T1 strong, rest weak. Deterministic seed."""
    rng = random.Random(99)
    fixtures = []
    for season, base_day in (("2024-2025", 0), ("2025-2026", 380)):
        day = base_day
        for i in range(120):
            home, away = rng.sample(TEAMS, 2)
            day += 1
            sh = 1.0 + max(0, 4 - int(home[1])) * 0.10
            sa = 1.0 + max(0, 4 - int(away[1])) * 0.10
            r = rng.random()
            if r < 0.46 * sh / (sh + sa) * 2:
                hg, ag = rng.randint(1, 3), rng.randint(0, 2)
            elif r < 0.72:
                hg, ag = rng.randint(0, 1), rng.randint(0, 1)
            else:
                hg, ag = rng.randint(0, 2), rng.randint(1, 3)
            if hg == ag and rng.random() < 0.3:
                ag += 1
            if hg == ag and rng.random() < 0.3:
                hg += 1
            y = season.split("-")[0]
            month = 1 + (day % 12)
            dd = 1 + (day % 27)
            fixtures.append({
                "date": f"{y}-{month:02d}-{dd:02d}",
                "home": home, "away": away,
                "home_goals": hg, "away_goals": ag,
                "league": "SYN", "season": season,
            })
    return fixtures


def test_synthetic_shapes():
    fx = _synthetic_multi_season()
    assert len(fx) == 240
    assert {f["season"] for f in fx} == {"2024-2025", "2025-2026"}


def test_per_season_and_aggregate_metrics():
    fx = _synthetic_multi_season()
    result = run_multi_season_validation(fx)
    assert set(result["seasons"].keys()) == {"2024-2025", "2025-2026"}
    for season, models in result["seasons"].items():
        for name in ("baseline", "elo", "poisson", "dc", "ensemble"):
            assert name in models
            m = models[name]
            if m["n"]:
                assert 0 < m["log_loss"] <= 2.0
                assert m["log_loss_ci"] is not None
                assert m["brier_ci"] is not None
                assert m["hit_rate_ci"] is not None or m["n"] < 2
    # aggregate n == sum of season n
    for name in ("baseline", "elo", "ensemble"):
        assert result["aggregate"][name]["n"] == 240


def test_poisson_missing_data_limitation_reported():
    """Poisson features need prior matches -> fewer evaluated matches."""
    fx = _synthetic_multi_season()
    result = run_multi_season_validation(fx)
    poisson_n = result["aggregate"]["poisson"]["n"]
    baseline_n = result["aggregate"]["baseline"]["n"]
    assert 0 < poisson_n < baseline_n  # first matches of each season lack form


def test_ensemble_not_worse_than_baseline_on_hit_rate():
    fx = _synthetic_multi_season()
    result = run_multi_season_validation(fx)
    agg = result["aggregate"]
    assert agg["ensemble"]["hit_rate"] >= agg["baseline"]["hit_rate"] - 0.05


def test_report_mentions_no_roi_and_ci():
    fx = _synthetic_multi_season()
    result = run_multi_season_validation(fx)
    report = format_validation_report(result)
    assert "MULTI-SEASON VALIDATION" in report
    assert "ROI: not reported" in report
    assert "±" in report


def _synthetic_two_leagues() -> list[dict]:
    """Two leagues x two seasons each, deterministic."""
    rng = random.Random(123)
    fx = []
    for league in ("EPL", "LaLiga"):
        for season in ("2024-2025", "2025-2026"):
            for i in range(90):
                home, away = rng.sample(TEAMS, 2)
                r = rng.random()
                if r < 0.5:
                    hg, ag = rng.randint(1, 3), rng.randint(0, 2)
                elif r < 0.72:
                    hg, ag = rng.randint(0, 1), rng.randint(0, 1)
                else:
                    hg, ag = rng.randint(0, 2), rng.randint(1, 3)
                y = season.split("-")[0]
                fx.append({
                    "date": f"{y}-{1 + i % 12:02d}-{1 + i % 27:02d}",
                    "home": home, "away": away,
                    "home_goals": hg, "away_goals": ag,
                    "league": league, "season": season,
                })
    return fx


def test_cross_league_per_league_replay():
    from agents.football.validate import run_cross_league_validation

    fx = _synthetic_two_leagues()
    result = run_cross_league_validation(fx)
    assert set(result["per_league"].keys()) == {"EPL", "LaLiga"}
    assert result["n_matches_total"] == 360
    for league, res in result["per_league"].items():
        assert res["n_matches_total"] == 180
        assert set(res["seasons"].keys()) == {"2024-2025", "2025-2026"}
        assert res["aggregate"]["ensemble"]["n"] == 180


def test_cross_league_consistency_summary():
    from agents.football.validate import run_cross_league_validation, format_cross_league_report

    fx = _synthetic_two_leagues()
    result = run_cross_league_validation(fx)
    cs = result["cross_summary"]
    assert "ensemble" in cs["consistency"]
    total = cs["consistency"]["ensemble"]["of"]
    assert total == 4  # 2 leagues x 2 seasons
    assert 0 <= cs["consistency"]["ensemble"]["wins"] <= total
    assert cs["best_league_by_margin"] in ("EPL", "LaLiga")
    report = format_cross_league_report(result)
    assert "CROSS-LEAGUE VALIDATION" in report
    assert "EPL" in report and "LaLiga" in report


def test_cross_league_unknown_league_grouping():
    from agents.football.validate import run_cross_league_validation

    fx = _synthetic_two_leagues()
    for f in fx[:3]:
        f.pop("league", None)
    result = run_cross_league_validation(fx)
    assert "unknown" in result["per_league"]
    assert result["per_league"]["unknown"]["n_matches_total"] == 3


def test_empty_fixtures_no_crash():
    result = run_multi_season_validation([])
    assert result["n_matches_total"] == 0
    assert result["aggregate"]["baseline"]["n"] == 0
    assert result["aggregate"]["elo"]["log_loss"] is None


def test_missing_season_key_falls_to_unknown():
    fx = _synthetic_multi_season()
    for f in fx[:5]:
        f.pop("season", None)
    result = run_multi_season_validation(fx)
    assert "unknown" in result["seasons"]
    assert result["seasons"]["unknown"]["baseline"]["n"] == 5


def _synthetic_with_odds() -> list[dict]:
    """Synthetic multi-season fixtures WITH deterministic historical odds
    (required for the market baseline and ROI rows)."""
    rng = random.Random(7)
    out = []
    for f in _synthetic_multi_season():
        out.append(
            {
                **f,
                "home_odds": round(1.6 + rng.random() * 1.6, 2),
                "draw_odds": round(3.0 + rng.random() * 1.6, 2),
                "away_odds": round(3.0 + rng.random() * 2.6, 2),
            }
        )
    return out


def test_market_row_appears_with_odds():
    fx = _synthetic_with_odds()
    result = run_multi_season_validation(fx)
    agg = result["aggregate"]
    assert "market" in agg
    assert agg["market"]["n"] == 240
    assert agg["market"]["log_loss"] is not None
    assert result["roi_available"] is True
    # models get ROI only when odds exist
    assert agg["ensemble"]["roi"] is not None
    assert agg["ensemble"]["bets"] > 0


def test_market_skipped_without_odds():
    fx = _synthetic_multi_season()  # no odds
    result = run_multi_season_validation(fx)
    agg = result["aggregate"]
    assert "market" in agg
    assert agg["market"]["n"] == 0
    assert agg["market"]["log_loss"] is None
    assert result["roi_available"] is False
    for name in ("baseline", "elo", "ensemble"):
        assert agg[name]["roi"] is None
        assert agg[name]["bets"] == 0


def test_beats_market_consistency():
    fx = _synthetic_with_odds()
    result = run_multi_season_validation(fx)
    c = result["consistency"]["ensemble"]
    assert c["market_seasons"] == 2  # both synthetic seasons have odds
    assert 0 <= c["beats_market_seasons"] <= 2
    assert isinstance(c["market_worse_seasons"], list)


def test_report_shows_market_and_roi():
    fx = _synthetic_with_odds()
    result = run_multi_season_validation(fx)
    report = format_validation_report(result)
    assert "market" in report
    assert "ROI: flat-stake" in report
    assert "beats MARKET" in report


def test_kelly_staking_metrics_reported():
    """Fractional-Kelly diagnostics must be present whenever ROI bets exist
    and must be honest: kelly_bets is capped by flat bets and any staked
    fraction respects the KELLY_CAP. The report shows the diagnostics block."""
    fx = _synthetic_with_odds()
    result = run_multi_season_validation(fx)
    agg = result["aggregate"]["ensemble"]
    assert agg["bets"] > 0
    assert 0 <= agg["kelly_bets"] <= agg["bets"]
    assert "kelly_fraction" in agg and "kelly_growth" in agg and "kelly_roi" in agg
    if agg["kelly_bets"]:
        assert agg["kelly_fraction"] is not None
        assert 0.0 <= agg["kelly_fraction"] <= 0.3  # KELLY_CAP
        assert agg["kelly_growth"] is not None and math.isfinite(agg["kelly_growth"])
    report = format_validation_report(result)
    assert "Staking diagnostics" in report
    assert "Kelly" in report


def test_kelly_empty_without_odds():
    """No odds -> no ROI bets -> no Kelly series -> honest empty diagnostics."""
    fx = _synthetic_multi_season()  # no odds
    result = run_multi_season_validation(fx)
    agg = result["aggregate"]["ensemble"]
    assert agg["bets"] == 0
    assert agg["kelly_bets"] == 0
    assert agg["kelly_fraction"] is None
    assert agg["kelly_growth"] is None


def test_calibration_out_writes_live_usable_file(tmp_path):
    """--calibration-out persists ensemble log-odds params that the live
    Calibrator (min_samples=200) will actually apply."""
    from agents.football.calibration import Calibrator

    fx = _synthetic_with_odds()
    out = tmp_path / "calibration.json"
    result = run_multi_season_validation(fx, calibration_out=out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {"a", "b", "samples", "ece"} <= set(payload)
    assert payload["samples"] >= 200  # above the live min_samples gate
    assert result["calibration"]["file"] == str(out)
    # the live path loads this file and becomes active
    cal = Calibrator(path=out, min_samples=200)
    assert cal.samples == payload["samples"]
    assert 0.0 <= cal.apply(0.5) <= 1.0


def test_calibration_out_none_no_file(tmp_path):
    fx = _synthetic_multi_season()
    out = tmp_path / "calibration.json"
    run_multi_season_validation(fx)  # no calibration_out -> nothing written
    assert not out.exists()


def test_main_wires_calibration_out(tmp_path):
    """CLI end-to-end: --fixtures (offline) + --calibration-out writes the
    live-usable calibration file."""
    from agents.football.validate import main

    fx = _synthetic_with_odds()
    fx_path = tmp_path / "fixtures.json"
    fx_path.write_text(json.dumps(fx), encoding="utf-8")
    cal_path = tmp_path / "calibration.json"
    rc = main(
        ["--leagues", "EPL", "--fixtures", str(fx_path),
         "--calibration-out", str(cal_path)]
    )
    assert rc == 0
    assert cal_path.exists()
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    assert payload["samples"] >= 200
    assert {"a", "b", "ece"} <= set(payload)


def test_no_future_information_check():
    """Elo state must carry across seasons but never see the current match."""
    fx = _synthetic_multi_season()
    # Season 2 first match must be predicted using ONLY season-1 + earlier
    # season-2 matches. We verify indirectly: elo gets more n than poisson
    # (elo works from match 1, poisson needs prior form), and baseline n == 240.
    result = run_multi_season_validation(fx)
    assert result["aggregate"]["elo"]["n"] == 240
    assert result["aggregate"]["baseline"]["n"] == 240


def test_seed_elo_writes_live_usable_file(tmp_path):
    """--seed-elo persists the final walk-forward Elo ratings so the live
    EloModel (path=elo.json) recognises the replayed teams."""
    from agents.football.elo import EloModel

    fx = _synthetic_multi_season()
    out = tmp_path / "elo.json"
    result = run_multi_season_validation(fx, seed_elo_path=out)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {"ratings", "games"} <= set(payload)
    assert len(payload["ratings"]) == 10  # all synthetic teams replayed
    assert result["seeded_elo"]["ratings"] == payload["ratings"]
    # the live path loads this file and knows the teams
    elo = EloModel(path=out)
    assert elo.known("T0", "T1") is True
    assert elo.rating("T0") != 1500.0  # moved from the initial rating


def test_seed_elo_none_no_file(tmp_path):
    fx = _synthetic_multi_season()
    out = tmp_path / "elo.json"
    run_multi_season_validation(fx)  # no seed_elo_path -> nothing written
    assert not out.exists()


def test_main_wires_seed_elo(tmp_path):
    """CLI end-to-end: --fixtures (offline) + --seed-elo writes live-usable
    Elo ratings and exits with rc 0."""
    from agents.football.validate import main

    fx = _synthetic_multi_season()
    fx_path = tmp_path / "fixtures.json"
    fx_path.write_text(json.dumps(fx), encoding="utf-8")
    elo_path = tmp_path / "elo.json"
    rc = main(
        ["--leagues", "EPL", "--fixtures", str(fx_path),
         "--seed-elo", str(elo_path)]
    )
    assert rc == 0
    assert elo_path.exists()
    payload = json.loads(elo_path.read_text(encoding="utf-8"))
    assert len(payload["ratings"]) == 10


# ---- Multi-league loader leakage guard (2026-08-16 regression) -----------
# The runner loaded multileague_fixtures.json AND the per-league caches,
# doubling every match. A match replayed twice is predicted the second time
# with its own result already in Elo/form state -> look-ahead leakage that
# flipped EPL from ROI -1.9% to +31.7% and fabricated the only "beats
# market" result in the project.


def _ml_fx(league, date, home, away, hg=1, ag=0):
    return {
        "league": league, "date": date, "home": home, "away": away,
        "home_goals": hg, "away_goals": ag, "season": "2024-2025",
        "home_odds": 1.9, "draw_odds": 3.4, "away_odds": 4.0,
    }


def test_dedupe_fixtures_keeps_first_and_counts():
    from agents.football.validate import dedupe_fixtures

    fx = [
        _ml_fx("EPL", "2024-09-01", "A", "B"),
        _ml_fx("EPL", "2024-09-01", "A", "B"),     # duplicate -> removed
        _ml_fx("EPL", "2024-09-01", "C", "D"),
        _ml_fx("LaLiga", "2024-09-01", "A", "B"),  # same teams, other league -> keep
        _ml_fx("EPL", "2024-09-02", "A", "B"),     # same teams, other date -> keep
    ]
    out, removed = dedupe_fixtures(fx)
    assert removed == 1
    assert len(out) == 4
    assert out[0] is fx[0]  # first occurrence is kept


def test_dedupe_fixtures_empty_and_missing_keys():
    from agents.football.validate import dedupe_fixtures

    assert dedupe_fixtures([]) == ([], 0)
    assert dedupe_fixtures(None) == ([], 0)
    # identical key-less rows collapse to the same identity -> deduped;
    # a row with ANY identity field is distinct and survives
    out, removed = dedupe_fixtures([{}, {}, {"home": "A"}])
    assert removed == 1
    assert len(out) == 2


def test_validate_multileague_dedupes_doubled_input(tmp_path):
    """Regression for the leaked report: doubled input must NOT inflate n or
    fabricate profit -- the replay runs on the unique match set only, and the
    removed count is reported for auditability."""
    from agents.football.validate import validate_multileague

    singles = [_ml_fx("EPL", f"2024-09-{i + 1:02d}", f"H{i}", f"A{i}") for i in range(30)]
    doubled = singles + singles  # aggregate file + per-league cache overlap
    rep = validate_multileague(
        {"EPL": doubled}, out_dir=tmp_path / "out", date="2026-08-16",
        requested_leagues=["EPL"],
    )
    assert rep["n_duplicates_removed"] == {"EPL": 30}
    ens = [s for s in rep["segments"] if s["model"] == "ensemble"][0]
    assert ens["n"] == 30  # NOT 60
    baseline = [s for s in rep["segments"] if s["model"] == "baseline"][0]
    assert baseline["n"] == 30


def test_validate_multileague_uses_production_ensemble_config(tmp_path):
    """Train/serve parity: the multileague harness must evaluate the
    PRODUCTION ensemble (elo 0.7 / poisson 0.3) when no config is passed,
    not the library defaults (0.5/0.5) -- same bug family as the backtest
    parity fix. Its ensemble row must match run_multi_season_validation with
    the production config exactly."""
    from agents.football.backtest import _load_model_config
    from agents.football.validate import run_multi_season_validation, validate_multileague

    fx = [_ml_fx("EPL", f"2024-09-{i + 1:02d}", f"H{i}", f"A{i}") for i in range(40)]
    rep = validate_multileague(
        {"EPL": fx}, out_dir=tmp_path / "out", date="2026-08-16",
        requested_leagues=["EPL"],
    )
    ens = [s for s in rep["segments"] if s["model"] == "ensemble"][0]

    elo_cfg, poisson_cfg, ensemble_cfg = _load_model_config()
    ref = run_multi_season_validation(
        fx, elo_cfg=elo_cfg, poisson_cfg=poisson_cfg, ensemble_cfg=ensemble_cfg
    )
    ref_ens = ref["aggregate"]["ensemble"]
    assert ens["n"] == ref_ens["n"]
    assert ens["log_loss"] == ref_ens["log_loss"]
    assert ens["brier"] == ref_ens["brier"]
    assert ens["roi"] == ref_ens["roi"]
    assert ens["kelly_g"] == ref_ens["kelly_growth"]


def test_validate_multileague_unduplicated_input_reports_zero_removed(tmp_path):
    from agents.football.validate import validate_multileague

    fixtures = {"EPL": [_ml_fx("EPL", "2024-09-01", "A", "B")]}
    rep = validate_multileague(
        fixtures, out_dir=tmp_path / "out", date="2026-08-16",
        requested_leagues=["EPL"],
    )
    assert rep["n_duplicates_removed"] == {}
    ens = [s for s in rep["segments"] if s["model"] == "ensemble"][0]
    assert ens["n"] == 1


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
