"""Automated anti-leakage audit (MASTER PROMPT PHASE 2).

This module does NOT change production behaviour. It inspects the existing
pipeline and REPORTS on whether the strict temporal leakage controls hold:

  1. FEATURE PROVENANCE — every field of :class:`MatchContext` is documented
     (source, pre-match availability, update frequency, missing behaviour,
     leakage risk). A test-level check verifies the registry covers every
     dataclass field, so a new feature cannot enter the context silently.
  2. PREDICT-BEFORE-UPDATE — a chronological replay identical to validate.py
     proves that at prediction time the state (form deques, Elo, base rates)
     contains ONLY matches strictly before the predicted match. The invariant
     is checked against an INDEPENDENT reference: each team's form deque must
     exactly equal the last min(prior, FORM_MAXLEN) scorelines that precede
     the match in the sorted fixture list (content-based, so the capped deque
     cannot hide an update-before-predict refactor). False-positive-free:
     identical repeated scorelines from earlier matches are part of the
     reference and cannot trigger it.
  3. DETERMINISM / INPUT_HASH STABILITY — the replay runs twice; the per-match
     ``input_hash`` sequence and the aggregate metrics must be byte-identical.
     Same snapshot -> same prediction (reproducibility requirement).
  4. PIPELINE EQUIVALENCE — the audit replay must reproduce the aggregate
     ensemble metrics of ``validate.run_multi_season_validation``, proving the
     audited context is the production context (up to the ``as_of`` timestamp,
     which is fixed to the kickoff date here so hashes are comparable).
  5. CLV-NOT-A-FEATURE — closing odds / CLV may appear ONLY in evaluation and
     logging paths (prediction_log, settler, runner, format). Any occurrence
     inside a model/context/feature module is a hard violation.
  6. SAME-DAY CAVEAT — matches ordered by date only (FBref provides no kickoff
     time) are counted and reported as a known limitation, never hidden.

Usage::

    python -m agents.football.leakage_audit --fixtures cache/football/epl_fixtures_2022_2026.json
    python -m agents.football.leakage_audit --fixtures f.json --out reports/leakage_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Any

from .backtest import BASE_RATE_PRIOR, load_fixtures_from_json
from .context import MatchContext
from .elo import EloModel
from .models import Ensemble, PoissonModel, poisson_matrix, probs_from_matrix
from .timeutil import kickoff_sort_key, utc_now_iso
from .validate import _ctx_for  # production context builder (no drift)

# Modules where closing odds / CLV are ALLOWED (evaluation, logging and
# audit/reporting paths only). Prediction/context/model/feature modules must
# never reference closing odds -- the scan flags anything else.
CLV_ALLOWED_MODULES = {
    "prediction_log.py", "settler.py", "runner.py", "format.py",
    "leakage_audit.py", "baseline_freeze.py",
    # bot.py is the Discord transport (same output layer as runner/format);
    # it mentions CLV only in docstrings/help for the !football odds command.
    "bot.py",
    # Phase 0/3/4: nowgoal.py is the odds DATA SOURCE (it owns the t=11
    # closing leg); clv_gate.py IS the gate; decision.py holds the gate
    # logic; analyse.py wires the gates into the decision flow; model_gates.py
    # passes the clv_gate config through; validate.py is the EVALUATION
    # harness (closing KPIs are its whole purpose). None of these feed
    # closing data into the prediction/context/model/feature path -- the scan
    # still flags any module that does.
    "nowgoal.py", "clv_gate.py", "decision.py", "analyse.py",
    "model_gates.py", "validate.py",
}

# ── 1. Feature provenance registry ─────────────────────────────────────────
# Every MatchContext field, its source, pre-match availability, update
# frequency, missing-data behaviour and leakage risk.
FEATURE_PROVENANCE: list[dict[str, Any]] = [
    {
        "field": "league",
        "source": "config/football.json + league_resolver (fixture metadata)",
        "availability": "pre-match",
        "update_frequency": "fixed per fixture",
        "missing_behavior": "'' when unknown",
        "leakage_risk": "none",
        "historical_backtest": "yes (fixture['league'])",
    },
    {
        "field": ["home", "away"],
        "source": "league_resolver / match_finder (fixture identity)",
        "availability": "pre-match",
        "update_frequency": "fixed per fixture",
        "missing_behavior": "n/a (identity)",
        "leakage_risk": "none; fixture-mismatch guard in match_finder (identity must be verified before any bet)",
        "historical_backtest": "yes",
    },
    {
        "field": "kickoff_utc",
        "source": "fixture (FBref / football-data / flashscore)",
        "availability": "pre-match",
        "update_frequency": "fixed per fixture",
        "missing_behavior": "None when unknown",
        "leakage_risk": "none (day-granularity caveat: same-day order arbitrary)",
        "historical_backtest": "yes (sort key)",
    },
    {
        "field": "as_of",
        "source": "utc_now_iso() at prediction timestamp",
        "availability": "prediction timestamp",
        "update_frequency": "per prediction",
        "missing_behavior": "always set",
        "leakage_risk": "none (timestamp marker; fixed in audit replay for determinism)",
        "historical_backtest": "yes (fixed to kickoff in audit)",
    },
    {
        "field": ["home_form", "away_form"],
        "source": "flashscore / football-data.org (team last-5 W/D/L sequence)",
        "availability": "pre-match",
        "update_frequency": "post-kickoff only; exclude_event_id skips the fixture being predicted",
        "missing_behavior": "None -> Poisson falls back to GF/GA averages",
        "leakage_risk": "none (history fetcher excludes the live fixture id)",
        "historical_backtest": "yes (validate.py form deques)",
    },
    {
        "field": ["home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"],
        "source": "rolling averages over settled past matches (flashscore / football-data)",
        "availability": "pre-match",
        "update_frequency": "post-kickoff only",
        "missing_behavior": "None when no prior matches",
        "leakage_risk": "none (computed strictly before kickoff)",
        "historical_backtest": "yes (rolling form deques)",
    },
    {
        "field": ["home_xg_for", "home_xg_against", "away_xg_for", "away_xg_against"],
        "source": "team-level xG averages from pre-match provider data (flashscore)",
        "availability": "pre-match only",
        "update_frequency": "post-kickoff only",
        "missing_behavior": "None when unavailable; NEVER fabricated (no fake xG)",
        "leakage_risk": "none if sourced pre-match; live/in-play xG must never be used",
        "historical_backtest": "no (no historical xG dataset; xg_weight applies only when present)",
    },
    {
        "field": "h2h",
        "source": "flashscore / thesportsdb head-to-head history (settled matches)",
        "availability": "pre-match",
        "update_frequency": "post-kickoff only",
        "missing_behavior": "None when no prior meetings",
        "leakage_risk": "none (historical meetings only)",
        "historical_backtest": "no (not replayed in validate.py; informational)",
    },
    {
        "field": "consensus_odds",
        "source": "The Odds API consensus (median across bookmakers) at prediction time",
        "availability": "prediction timestamp (odds freshness tracked)",
        "update_frequency": "per fetch; cache_ttl_seconds gates staleness",
        "missing_behavior": "None -> no market comparison, no ROI, model only",
        "leakage_risk": "stale odds risk (must be odds available AT prediction, never closing odds)",
        "historical_backtest": "yes (fixture home_odds/draw_odds/away_odds are pre-match odds)",
    },
    {
        "field": "market_totals",
        "source": "The Odds API totals (Over/Under lines) at prediction time",
        "availability": "prediction timestamp",
        "update_frequency": "per fetch",
        "missing_behavior": "None",
        "leakage_risk": "stale odds risk only",
        "historical_backtest": "no (no historical totals dataset)",
    },
    {
        "field": ["home_recent_goals", "away_recent_goals"],
        "source": "raw (gf, ga) scorelines of settled past matches, oldest->newest",
        "availability": "pre-match",
        "update_frequency": "post-kickoff only",
        "missing_behavior": "None -> equal-weight averages instead of time-decay",
        "leakage_risk": "none (audit verifies the predicted match's own scoreline is never present)",
        "historical_backtest": "yes (validate.py deques, maxlen=5)",
    },
    {
        "field": "form_samples",
        "source": "derived: min W/D/L token count across both form strings",
        "availability": "pre-match",
        "update_frequency": "per prediction",
        "missing_behavior": "0 when no form",
        "leakage_risk": "none",
        "historical_backtest": "yes",
    },
    {
        "field": "xg_samples",
        "source": "derived: count of non-null xG values",
        "availability": "pre-match",
        "update_frequency": "per prediction",
        "missing_behavior": "0",
        "leakage_risk": "none",
        "historical_backtest": "yes (0 in replay; xG absent)",
    },
    {
        "field": "sources",
        "source": "provider chain resolution (flashscore / football-data / thesportsdb)",
        "availability": "prediction timestamp",
        "update_frequency": "per prediction",
        "missing_behavior": "[]",
        "leakage_risk": "none",
        "historical_backtest": "yes (informational)",
    },
    {
        "field": "source_meta",
        "source": "multi-source aggregation layer (per-field provenance/confidence)",
        "availability": "prediction timestamp",
        "update_frequency": "per prediction",
        "missing_behavior": "None",
        "leakage_risk": "none (metadata only, never a model feature)",
        "historical_backtest": "no (not replayed; informational)",
    },
]


def provenance_registry() -> list[dict[str, Any]]:
    """Registry of documented MatchContext fields, expanded to one row per
    field (a single row may list several fields)."""
    out: list[dict[str, Any]] = []
    for row in FEATURE_PROVENANCE:
        fields = row["field"] if isinstance(row["field"], list) else [row["field"]]
        for f in fields:
            entry = dict(row)
            entry["field"] = f
            out.append(entry)
    return out


def check_provenance_coverage() -> dict[str, Any]:
    """Every MatchContext dataclass field must be documented."""
    documented = {r["field"] for r in provenance_registry()}
    actual = {f.name for f in fields(MatchContext)}
    missing = sorted(actual - documented)
    extra = sorted(documented - actual)
    return {
        "documented": sorted(documented),
        "actual": sorted(actual),
        "missing": missing,
        "extra": extra,
        "covered": not missing and not extra,
    }


def check_clv_scope(package_dir: Path | None = None) -> dict[str, Any]:
    """Closing odds / CLV must never appear in model/context/feature modules.

    Scans the agents/football package source for closing-odds references and
    reports any file outside the evaluation/logging allowlist.
    """
    package_dir = package_dir or Path(__file__).resolve().parent
    hits: list[str] = []
    for py in sorted(package_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        if not any(k in text for k in ("closing_odds", "closing odds", "clv", "CLV")):
            continue
        hits.append(py.name)
    violations = sorted(set(hits) - CLV_ALLOWED_MODULES)
    return {
        "scan_dir": str(package_dir),
        "files_with_closing_references": sorted(set(hits)),
        "allowed_modules": sorted(CLV_ALLOWED_MODULES),
        "violations": violations,
        "passed": not violations,
    }


# ── 2. Chronological replay with invariant checks ──────────────────────────

def _audit_ctx_for(
    fixture: dict[str, Any],
    forms: dict[str, deque],
    last_date: dict[str, str],
) -> MatchContext:
    """Production context via validate._ctx_for, with ``as_of`` pinned to the
    kickoff date so per-match input hashes are deterministic and comparable
    across passes. Reusing the production builder means zero drift: whatever
    the live pipeline computes is exactly what the audit checks."""
    ctx = _ctx_for(fixture, forms, last_date)
    ctx.as_of = fixture["date"]
    return ctx


FORM_MAXLEN = 5  # must mirror validate.run_multi_season_validation's deque(maxlen=5)


def _state_content_violations(
    ctx: MatchContext,
    forms: dict[str, deque],
    prior_ref: dict[str, list[tuple[int, int]]],
) -> list[str]:
    """Invariant A: each team's form state must EXACTLY equal the last
    ``min(prior, FORM_MAXLEN)`` scorelines of matches that PRECEDE the
    predicted match in the chronological replay.

    ``prior_ref`` is the independent reference: per-team prior scoreline
    sequences derived from the SORTED FIXTURE LIST (home perspective (gf, ga),
    away perspective (ga, gf) -- exactly how validate.py stores the deques),
    NOT from the mutated state. That independence is what catches the
    update-before-predict bug: a refactor that moves the state update ahead of
    the prediction leaves the current match's scoreline in the deque, which
    cannot match a reference that excludes it -- regardless of where the
    mutation happens.

    The check is CONTENT-based, not count-based, deliberately: the form deques
    are capped at FORM_MAXLEN, so a count-only check is blind for teams with
    >= 5 prior matches (the deque holds 5 entries whether or not the current
    match was appended -- the common case mid/late season). Comparing content
    catches the bug in that regime too; the only theoretical miss is a
    coincidentally periodic deque whose last 5 entries equal the reference
    after the append.
    """
    violations: list[str] = []
    for team in (ctx.home, ctx.away):
        # Compare only the (gf, ga) prefix: the --opp-adj-form experiment
        # stores an optional opponent-strength third element in the same
        # deques, which is extra state, not part of the scoreline invariant.
        actual = [tuple(g[:2]) for g in forms.get(team, ())]
        expected = prior_ref.get(team, [])
        if actual != expected:
            violations.append(
                f"{team}: form state {actual} != prior reference {expected} "
                f"(update-before-predict?)"
            )
    return violations


def _aggregate_ensemble_metrics(
    fixtures: list[dict[str, Any]],
    elo: EloModel,
    poisson: PoissonModel,
    dc: PoissonModel,
    ensemble: Ensemble,
) -> tuple[dict[str, Any], list[str]]:
    """One deterministic chronological replay + invariant checks.

    Returns (metrics, hashes) where hashes is the per-match input_hash
    sequence. Mirrors validate.run_multi_season_validation's loop exactly
    (update strictly AFTER prediction) so the invariants are checked on the
    production context shape.
    """
    forms: dict[str, deque] = {}
    last_date: dict[str, str] = {}
    base = {"home": 0, "draw": 0, "away": 0}
    base_n = 0
    ll_series: list[float] = []
    hashes: list[str] = []
    violations: list[str] = []

    # INVARIANT A reference: per-team prior scoreline sequences derived from
    # the sorted fixture list itself (independent of the mutated state). For
    # match i the expected form state is the last min(prior, FORM_MAXLEN)
    # scorelines of each team before index i, in the same (home/away
    # perspective) representation validate.py stores in its deques.
    # TODO-05: same-day matches are ordered by kickoff time when known so an
    # earlier-kickoff result can never leak into a later-kickoff match.
    sorted_fx = sorted(fixtures, key=kickoff_sort_key)
    team_seqs: dict[str, list[tuple[int, int]]] = {}
    prior_refs: list[dict[str, list[tuple[int, int]]]] = []
    for fx in sorted_fx:
        prior_refs.append(
            {t: list(seq[-FORM_MAXLEN:]) for t, seq in team_seqs.items()}
        )
        team_seqs.setdefault(fx["home"], []).append(
            (fx["home_goals"], fx["away_goals"])
        )
        team_seqs.setdefault(fx["away"], []).append(
            (fx["away_goals"], fx["home_goals"])
        )

    for idx, fixture in enumerate(sorted_fx):
        hg, ag = fixture["home_goals"], fixture["away_goals"]
        outcome = 0 if hg > ag else (1 if hg == ag else 2)
        ctx = _audit_ctx_for(fixture, forms, last_date)
        hashes.append(ctx.input_hash)

        # INVARIANT A: state must contain exactly the matches strictly before
        # the predicted match (predict-before-update).
        violations.extend(_state_content_violations(ctx, forms, prior_refs[idx]))

        p_base = (
            {k: base[k] / base_n for k in ("home", "draw", "away")}
            if base_n
            else dict(BASE_RATE_PRIOR)
        )
        lh_e, la_e = elo.expected_lambdas(ctx.home, ctx.away)
        p_elo, _, _, _, _ = probs_from_matrix(poisson_matrix(lh_e, la_e, rho=0.0))
        ens = ensemble.predict(ctx, elo, dc)
        p_ens = ens["1x2"] if ens else None
        if p_ens:
            keys = ("home", "draw", "away")
            ll_series.append(-math.log(max(1e-9, p_ens[keys[outcome]])))

        # Update state with the RESULT (strictly after prediction).
        elo.update(ctx.home, ctx.away, hg, ag, persist=False)
        forms.setdefault(ctx.home, deque(maxlen=5)).append((hg, ag))
        forms.setdefault(ctx.away, deque(maxlen=5)).append((ag, hg))
        last_date[ctx.home] = fixture["date"]
        last_date[ctx.away] = fixture["date"]
        base[("home", "draw", "away")[outcome]] += 1
        base_n += 1

    n = len(ll_series)
    metrics = {
        "n_ensemble_evaluated": n,
        "ensemble_mean_log_loss": round(sum(ll_series) / n, 4) if n else None,
        "n_matches": len(fixtures),
    }
    return metrics, hashes, violations


def audit_replay(
    fixtures: list[dict[str, Any]],
    *,
    elo_cfg: dict[str, Any] | None = None,
    poisson_cfg: dict[str, Any] | None = None,
    ensemble_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full automated leakage audit on a fixture list (offline).

    By default the PRODUCTION model config (config/football.json) is used so
    the pipeline-equivalence check compares the audit replay against the real
    production pipeline; explicit cfg overrides are only for unit tests.
    """
    if elo_cfg is None and poisson_cfg is None and ensemble_cfg is None:
        from .validate import _load_model_config

        elo_cfg, poisson_cfg, ensemble_cfg = _load_model_config()
    elo_cfg = dict(elo_cfg or {})
    poisson_cfg = dict(poisson_cfg or {})
    ensemble_cfg = dict(ensemble_cfg or {})

    def _fresh() -> tuple[EloModel, PoissonModel, PoissonModel, Ensemble]:
        elo = EloModel(**elo_cfg)
        poisson = PoissonModel(
            base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
            base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
            dc_rho=0.0,
            shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
            time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
            xg_weight=poisson_cfg.get("xg_weight", 0.65),
            min_samples=poisson_cfg.get("min_samples", 2),
        )
        dc = PoissonModel(
            base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
            base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
            dc_rho=poisson_cfg.get("dc_rho", -0.1),
            shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
            time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
            xg_weight=poisson_cfg.get("xg_weight", 0.65),
            min_samples=poisson_cfg.get("min_samples", 2),
        )
        ensemble = Ensemble(
            elo_weight=ensemble_cfg.get("elo_weight", 0.5),
            poisson_weight=ensemble_cfg.get("poisson_weight", 0.5),
        )
        return elo, poisson, dc, ensemble

    # Pass 1 and pass 2 must produce identical input hashes + metrics.
    e1, p1, d1, en1 = _fresh()
    m1, h1, v1 = _aggregate_ensemble_metrics(fixtures, e1, p1, d1, en1)
    e2, p2, d2, en2 = _fresh()
    m2, h2, v2 = _aggregate_ensemble_metrics(fixtures, e2, p2, d2, en2)

    determinism = {
        "hashes_identical": h1 == h2,
        "metrics_identical": m1 == m2,
        "n_matches": len(h1),
        "hash_diffs": sum(1 for a, b in zip(h1, h2) if a != b),
    }

    # INVARIANT B: audit replay must reproduce the production aggregate
    # ensemble metrics (validate.run_multi_season_validation).
    from .validate import run_multi_season_validation

    prod = run_multi_season_validation(fixtures, elo_cfg=elo_cfg,
                                       poisson_cfg=poisson_cfg,
                                       ensemble_cfg=ensemble_cfg)
    prod_ll = prod["aggregate"]["ensemble"]["log_loss"]
    audit_ll = m1["ensemble_mean_log_loss"]
    # Both None (empty dataset) counts as matching; otherwise must agree.
    both_empty = prod_ll is None and audit_ll is None
    equivalence = {
        "production_aggregate_ensemble_log_loss": prod_ll,
        "audit_replay_ensemble_log_loss": audit_ll,
        "match": both_empty or (
            prod_ll is not None and audit_ll is not None
            and abs(prod_ll - audit_ll) < 0.0005
        ),
        "production_n": prod["aggregate"]["ensemble"]["n"],
        "audit_n": m1["n_ensemble_evaluated"],
    }

    # INVARIANT C: same-day ordering caveat (known limitation, counted).
    dates: dict[str, int] = {}
    for f in fixtures:
        dates[f["date"]] = dates.get(f["date"], 0) + 1
    same_day_groups = {d: c for d, c in dates.items() if c > 1}

    invariants = [
        {
            "name": "predict_before_update",
            "passed": not v1,
            "detail": (
                f"{len(v1)} violation(s); form state must equal the independent "
                f"prior-match reference (update-before-predict guard)"
            ),
            "violations": v1[:10],
        },
        {
            "name": "determinism_input_hash",
            "passed": determinism["hashes_identical"] and determinism["metrics_identical"],
            "detail": (
                f"{determinism['n_matches']} matches, "
                f"{determinism['hash_diffs']} hash differences across 2 passes"
            ),
        },
        {
            "name": "pipeline_equivalence",
            "passed": equivalence["match"],
            "detail": (
                f"audit replay LL {equivalence['audit_replay_ensemble_log_loss']} vs "
                f"production validate LL {equivalence['production_aggregate_ensemble_log_loss']} "
                f"({equivalence['audit_n']} vs {equivalence['production_n']} matches)"
            ),
        },
    ]

    coverage = check_provenance_coverage()
    clv = check_clv_scope()

    n_same_day = sum(same_day_groups.values()) - len(same_day_groups)
    return {
        "generated_at": utc_now_iso(),
        "n_matches": len(fixtures),
        "provenance": coverage,
        "clv_scope": clv,
        "invariants": invariants,
        "all_invariants_passed": all(i["passed"] for i in invariants)
        and coverage["covered"]
        and clv["passed"],
        "same_day_caveat": {
            "n_dates_with_multiple_matches": len(same_day_groups),
            "n_matches_involved": n_same_day,
            "note": (
                "FBref provides no kickoff time: same-day matches are ordered "
                "arbitrarily (date-only sort). In principle a match that "
                "kicked off LATER in real time may be processed EARLIER in the "
                "replay, making its result visible to an earlier-kickoff "
                "match's context (within-day look-ahead). This is a KNOWN "
                "day-granularity limitation of the historical dataset, not a "
                "production leak: the live bot orders by kickoff timestamp. "
                "Mitigation: re-order historical matches by kickoff time when "
                "the data provides it."
            ),
        },
        "verdict": (
            "PASS" if (
                all(i["passed"] for i in invariants)
                and coverage["covered"] and clv["passed"]
            )
            else "FAIL"
        ),
    }


def format_audit_report(audit: dict[str, Any]) -> str:
    lines = [
        "=" * 84,
        "ANTI-LEAKAGE AUDIT (strict temporal leakage control)",
        "=" * 84,
        f"generated_at : {audit['generated_at']}",
        f"matches      : {audit['n_matches']}",
        "",
        "--- Feature provenance (MatchContext) ---",
        f"  documented fields : {len(audit['provenance']['documented'])}",
        f"  actual fields     : {len(audit['provenance']['actual'])}",
        f"  missing docs      : {audit['provenance']['missing'] or 'none'}",
        f"  extra docs        : {audit['provenance']['extra'] or 'none'}",
        "",
        "--- CLV scope (closing odds never a feature) ---",
        f"  files with closing refs : {audit['clv_scope']['files_with_closing_references']}",
        f"  allowed (eval/log only) : {audit['clv_scope']['allowed_modules']}",
        f"  violations              : {audit['clv_scope']['violations'] or 'none'}",
        "",
        "--- Invariants ---",
    ]
    for inv in audit["invariants"]:
        lines.append(f"  [{'PASS' if inv['passed'] else 'FAIL'}] {inv['name']}")
        lines.append(f"        {inv['detail']}")
        for v in inv.get("violations", [])[:5]:
            lines.append(f"        ! {v}")
    caveat = audit["same_day_caveat"]
    lines.extend([
        "",
        "--- Same-day caveat (known limitation) ---",
        f"  dates with multiple matches : {caveat['n_dates_with_multiple_matches']}",
        f"  matches involved            : {caveat['n_matches_involved']}",
        f"  {caveat['note']}",
        "",
        f"VERDICT: {audit['verdict']}",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-leakage-audit")
    parser.add_argument("--fixtures", required=True,
                        help="local fixtures JSON (offline chronological replay)")
    parser.add_argument("--out", default=None, help="write JSON audit report here")
    args = parser.parse_args(argv)

    fixtures = load_fixtures_from_json(args.fixtures)
    audit = audit_replay(fixtures)
    print(format_audit_report(audit))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON audit report written to {out}")
    return 0 if audit["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
