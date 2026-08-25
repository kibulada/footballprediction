"""Chronological backtest harness.

Replays finished matches in strict kickoff order. Every model sees ONLY
matches strictly before the current kickoff (walk-forward, no leakage):

  - baseline : empirical base rates (expanding window)
  - elo      : rating-based, updated after each replayed match
  - poisson  : feature-based (rolling last-5 GF/GA from prior matches only)
  - ensemble : weighted blend of elo + poisson

Metrics: 1X2 log-loss, Brier, expected calibration error (ECE), hit rate of
the top-probability pick, and flat-stake ROI *when historical odds are
provided*. Honesty rule: results are only as meaningful as the dataset.
Synthetic/offline data never claims to validate anything.

CLI::

    python -m agents.football.backtest --fixtures fixtures.json
    python -m agents.football.backtest --leagues EPL --seasons 2025-2026 --seed-elo
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

from .calibration import Calibrator, expected_calibration_error
from .context import MatchContext
from .elo import EloModel
from .models import Ensemble, PoissonModel, poisson_matrix, probs_from_matrix

ROOT = Path(__file__).resolve().parent.parent.parent

BASE_RATE_PRIOR = {"home": 0.46, "draw": 0.26, "away": 0.28}


def _stderr_showwarning(message, category, filename, lineno, file=None, line=None) -> None:
    """Route warnings to stderr instead of the logging tree.

    soccerdata installs a rich RichHandler on the root logger; a version
    mismatch makes it crash while rendering any warning record. Bypassing
    logging for warnings keeps the backtest CLI stable while preserving
    visibility.
    """
    sys.stderr.write(f"{category.__name__}: {message}\n")


warnings.showwarning = _stderr_showwarning

# soccerdata calls logging.captureWarnings(True), which re-routes every
# warning into the logging tree and lets a rich RichHandler render it.
# That render path crashes in this environment, so disable capture entirely.
logging.captureWarnings = lambda *a, **k: None  # type: ignore[assignment]


def _silence_root_handlers() -> None:
    """Drop non-stderr handlers (soccerdata's RichHandler) from the root
    logger and mute third-party loggers after importing soccerdata."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler is logging.lastResort:
            continue
        stream = getattr(handler, "stream", None)
        if stream is None or stream is sys.stderr:
            continue
        root.removeHandler(handler)
    for name in list(logging.root.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    root.setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_fixtures_from_json(path: str | Path) -> list[dict[str, Any]]:
    """Load fixtures from a JSON file (list of dicts).

    Required keys: date, home, away, home_goals, away_goals.
    Optional: league, home_odds, draw_odds, away_odds (historical odds).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("fixtures", [])
    if not isinstance(payload, list):
        raise ValueError("fixtures JSON must be a list or {fixtures: [...]}")
    return _normalize_fixtures(payload)


def load_fbref_fixtures(
    leagues: list[str],
    seasons: list[str],
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch full league schedules from FBref via soccerdata (network)."""
    from .soccerdata_wrapper import SoccerDataWrapper

    import asyncio

    _silence_root_handlers()
    wrapper = SoccerDataWrapper(proxy=proxy)
    rows: list[dict[str, Any]] = []
    for league in leagues:
        for season in seasons:
            got = asyncio.run(wrapper.read_league_schedule(league, [season]))
            if got:
                rows.extend(got)
    if not rows:
        raise RuntimeError(
            "FBref returned no fixtures (network blocked or league/season invalid). "
            "Use --fixtures with a local JSON file instead."
        )
    return _normalize_fixtures(rows)


def _normalize_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in fixtures:
        date = m.get("date") or m.get("kickoff") or m.get("utcDate")
        if not date:
            continue
        home = str(m.get("home") or m.get("home_team") or "").strip()
        away = str(m.get("away") or m.get("away_team") or "").strip()
        hg, ag = m.get("home_goals"), m.get("away_goals")
        if not home or not away or hg is None or ag is None:
            continue
        try:
            hg, ag = int(hg), int(ag)
        except (ValueError, TypeError):
            continue
        out.append(
            {
                "date": str(date)[:10],
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "league": m.get("league", ""),
                "season": m.get("season", ""),
                "home_odds": m.get("home_odds"),
                "draw_odds": m.get("draw_odds"),
                "away_odds": m.get("away_odds"),
            }
        )
    return sorted(out, key=lambda x: x["date"])


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
def _ctx_for(fixture: dict[str, Any], forms: dict[str, deque]) -> MatchContext:
    home, away = fixture["home"], fixture["away"]

    def _stats(team: str) -> tuple[float | None, float | None, int]:
        dq = forms.get(team)
        if not dq:
            return None, None, 0
        gfs = [g[0] for g in dq]
        gas = [g[1] for g in dq]
        return sum(gfs) / len(gfs), sum(gas) / len(gas), len(dq)

    hgf, hga, hs = _stats(home)
    agf, aga, as_ = _stats(away)
    return MatchContext(
        league=fixture.get("league", ""),
        home=home,
        away=away,
        kickoff_utc=fixture["date"],
        home_gf_avg=hgf,
        home_ga_avg=hga,
        away_gf_avg=agf,
        away_ga_avg=aga,
        # Raw scorelines enable time-decay weighting (Dixon-Coles xi) in the
        # Poisson model instead of equal-weight rolling averages.
        home_recent_goals=[tuple(g) for g in forms.get(home, ())] or None,
        away_recent_goals=[tuple(g) for g in forms.get(away, ())] or None,
        form_samples=min(hs, as_) if hs and as_ else max(hs, as_),
        consensus_odds=(
            {
                "home": fixture["home_odds"],
                "draw": fixture["draw_odds"],
                "away": fixture["away_odds"],
            }
            if all(fixture.get(k) for k in ("home_odds", "draw_odds", "away_odds"))
            else None
        ),
    )


def _outcome_index(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def run_backtest(
    fixtures: list[dict[str, Any]],
    *,
    elo_cfg: dict[str, Any] | None = None,
    poisson_cfg: dict[str, Any] | None = None,
    ensemble_cfg: dict[str, Any] | None = None,
    edge_threshold: float = 0.02,
    seed_elo_path: str | None = None,
) -> dict[str, Any]:
    """Walk-forward replay. Returns metrics + optional seeded Elo model."""
    elo_cfg = elo_cfg or {}
    poisson_cfg = poisson_cfg or {}
    ensemble_cfg = ensemble_cfg or {}

    elo = EloModel(**elo_cfg)
    poisson = PoissonModel(
        base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
        base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
        dc_rho=poisson_cfg.get("dc_rho", -0.1),
        shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
        time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
        xg_weight=poisson_cfg.get("xg_weight", 0.65),
    )
    ensemble = Ensemble(
        elo_weight=ensemble_cfg.get("elo_weight", 0.5),
        poisson_weight=ensemble_cfg.get("poisson_weight", 0.5),
    )

    forms: dict[str, deque] = {}
    base_counts = {"home": 0, "draw": 0, "away": 0}
    base_n = 0

    acc: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "elo", "poisson", "ensemble"):
        acc[name] = {"log_loss": 0.0, "brier": 0.0, "n": 0, "hit": 0,
                     "bets": 0, "net": 0.0, "cal_pairs": []}

    for fixture in fixtures:
        hg, ag = fixture["home_goals"], fixture["away_goals"]
        outcome = _outcome_index(hg, ag)
        odds = (
            (fixture["home_odds"], fixture["draw_odds"], fixture["away_odds"])
            if all(fixture.get(k) for k in ("home_odds", "draw_odds", "away_odds"))
            else None
        )
        ctx = _ctx_for(fixture, forms)

        # ---- baseline (expanding base rates, prior before any data) ----
        if base_n:
            p_base = {
                "home": base_counts["home"] / base_n,
                "draw": base_counts["draw"] / base_n,
                "away": base_counts["away"] / base_n,
            }
        else:
            p_base = dict(BASE_RATE_PRIOR)

        # ---- elo ----
        lh_e, la_e = elo.expected_lambdas(ctx.home, ctx.away)
        p_elo, _, _, _, _ = probs_from_matrix(poisson_matrix(lh_e, la_e, rho=0.0))

        # ---- poisson (features only; None when no form yet) ----
        pm = poisson.predict(ctx)
        p_poisson = pm["1x2"] if pm else None

        # ---- ensemble ----
        ens = ensemble.predict(ctx, elo, poisson)
        p_ens = ens["1x2"] if ens else None

        for name, probs in (
            ("baseline", p_base),
            ("elo", p_elo),
            ("poisson", p_poisson),
            ("ensemble", p_ens),
        ):
            if probs is None:
                continue
            a = acc[name]
            a["n"] += 1
            keys = ("home", "draw", "away")
            p_out = max(1e-9, probs[keys[outcome]])
            a["log_loss"] += -math.log(p_out)
            a["brier"] += sum(
                (probs[k] - (1.0 if i == outcome else 0.0)) ** 2
                for i, k in enumerate(keys)
            )
            if max(probs, key=probs.get) == keys[outcome]:
                a["hit"] += 1
            # Proper multiclass calibration pairs: (P(k), indicator k) pooled
            # across the three outcomes (3 pairs per match).
            for i, k in enumerate(keys):
                a["cal_pairs"].append((probs[k], 1.0 if outcome == i else 0.0))

            # flat-stake ROI on the best 1X2 pick with edge (historical odds).
            # Edge uses MARGIN-FREE implied so it is consistent with the engine.
            if odds:
                keys = ["home", "draw", "away"]
                raw = [1.0 / o if o and o > 1.0 else 0.0 for o in odds]
                total_raw = sum(raw)
                norm = [r / total_raw if total_raw > 0 else 0.0 for r in raw]
                best = max(probs, key=probs.get)
                idx = keys.index(best)
                edge = probs[best] - norm[idx]
                if edge >= edge_threshold:
                    a["bets"] += 1
                    a["net"] += odds[idx] - 1.0 if outcome == idx else -1.0

        # ---- update state with the RESULT (after prediction) ----
        elo.update(ctx.home, ctx.away, hg, ag, persist=False)
        # gf/ga from the team's own perspective:
        forms.setdefault(ctx.home, deque(maxlen=5)).append((hg, ag))
        forms.setdefault(ctx.away, deque(maxlen=5)).append((ag, hg))
        base_counts[["home", "draw", "away"][outcome]] += 1
        base_n += 1

    metrics: dict[str, dict[str, Any]] = {}
    for name, a in acc.items():
        n = a["n"]
        if not n:
            metrics[name] = {"n": 0, "log_loss": None, "brier": None,
                             "ece": None, "hit_rate": None, "roi": None}
            continue
        cal_probs = [p for p, _ in a["cal_pairs"]]
        cal_outs = [y for _, y in a["cal_pairs"]]
        ece = expected_calibration_error(cal_probs, cal_outs)
        metrics[name] = {
            "n": n,
            "log_loss": round(a["log_loss"] / n, 4),
            "brier": round(a["brier"] / n, 4),
            "ece": round(ece, 4) if not math.isnan(ece) else None,
            "hit_rate": round(a["hit"] / n, 4),
            "roi": round(a["net"] / a["bets"], 4) if a["bets"] else None,
            "bets": a["bets"],
        }

    # Calibrate the ensemble on the full replayed history (in-sample fit,
    # labelled honestly) and report pre/post ECE.
    ens_acc = acc["ensemble"]
    ens_probs = [p for p, _ in ens_acc["cal_pairs"]]
    ens_outs = [y for _, y in ens_acc["cal_pairs"]]
    calibrator = Calibrator()
    if ens_acc["n"] >= 20 and any(o == 1 for o in ens_outs) and any(o == 0 for o in ens_outs):
        calibrator.fit(ens_probs, ens_outs)

    if seed_elo_path:
        elo.path = Path(seed_elo_path)
        elo._save()

    return {
        "n_matches": len(fixtures),
        "metrics": metrics,
        "calibration": {
            "fitted": calibrator.samples,
            "ece_before": metrics["ensemble"]["ece"],
            "ece_after": (
                round(expected_calibration_error(
                    [calibrator.apply(p) for p in ens_probs], ens_outs
                ), 4)
                if calibrator.samples else None
            ),
            "a": calibrator.a,
            "b": calibrator.b,
        },
        "seeded_elo": seed_elo_path,
    }


def format_report(result: dict[str, Any], dataset_note: str) -> str:
    lines = [
        "=" * 64,
        "BACKTEST REPORT",
        "=" * 64,
        f"dataset : {dataset_note}",
        f"matches : {result['n_matches']}",
        "",
        f"{'model':<10}{'n':>7}{'logloss':>10}{'brier':>9}{'ece':>9}{'hit%':>9}{'roi':>9}",
        "-" * 64,
    ]
    for name, m in result["metrics"].items():
        if not m["n"]:
            lines.append(f"{name:<10}{0:>7}      n/a       n/a       n/a      n/a      n/a")
            continue
        ece_s = "n/a" if m["ece"] is None else f"{m['ece']:.4f}"
        roi_s = "n/a" if m["roi"] is None else f"{m['roi'] * 100:.1f}%"
        lines.append(
            f"{name:<10}{m['n']:>7}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
            f"{ece_s:>9}{m['hit_rate'] * 100:>8.1f}%{roi_s:>9}"
        )
    c = result["calibration"]
    lines += [
        "-" * 64,
        f"ensemble calibration: fitted={c['fitted']} samples, "
        f"ECE before={c['ece_before']}, after={c['ece_after']}",
        "",
        "NOTE: lower log-loss/Brier/ECE is better. 'roi' is flat-stake ROI on",
        "top-pick bets with >=2% edge and REQUIRES historical odds in the data.",
        "Synthetic/offline data does not validate any model. Only a backtest",
        "on real, chronologically-split data may be used to claim accuracy.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-backtest")
    parser.add_argument("--fixtures", default=None, help="local fixtures JSON file")
    parser.add_argument("--leagues", default=None, help="comma-separated league keys (FBref)")
    parser.add_argument("--seasons", default=None, help="comma-separated season codes, e.g. 2025-2026")
    parser.add_argument(
        "--seed-elo",
        nargs="?",
        const=str(ROOT / "cache" / "football" / "elo.json"),
        default=None,
        help="write seeded Elo ratings JSON to this path (bare --seed-elo uses cache/football/elo.json)",
    )
    parser.add_argument("--proxy", default=None, help="SOCKS/HTTPS proxy for FBref fetch")
    args = parser.parse_args(argv)

    dataset_note = ""
    if args.fixtures:
        fixtures = load_fixtures_from_json(args.fixtures)
        dataset_note = f"local file {args.fixtures}"
    elif args.leagues and args.seasons:
        leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
        seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
        fixtures = load_fbref_fixtures(leagues, seasons, proxy=args.proxy)
        dataset_note = f"FBref {args.leagues} {args.seasons} (real data)"
    else:
        parser.error("need --fixtures or --leagues+--seasons")

    result = run_backtest(fixtures, seed_elo_path=args.seed_elo)
    print(format_report(result, dataset_note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
