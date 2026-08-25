"""Throwaway diagnostic for test_stability_holds_when_nothing_changed.

Run:  python -m pytest tests/_diag_stability.py -q
Reads the failure message from the AssertionError (rtk truncates -s output).
Delete once the question is settled.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.football.models import poisson_matrix, probs_from_matrix  # noqa: E402
from agents.football.signal_engine import run_signal_engine  # noqa: E402
from tests.test_signal_consistency import _pick_payload  # noqa: E402

PROD = {"allow_negative_edge_pp": -3.0, "pick_gates": {"agreement": False}}


def _mp(lh, la):
    m = poisson_matrix(lh, la, rho=0.0)
    _p, _o15, o25, _o35, btts = probs_from_matrix(m)
    return {"1x2": {}, "over_2.5": o25, "btts_yes": btts,
            "lambda_home": lh, "lambda_away": la}


def test_diag():
    model = _mp(0.9, 1.9)
    totals = {"Over 2.5": {"odds": 1.83, "point": 2.5},
              "Under 2.5": {"odds": 2.07, "point": 2.5}}
    ah_rows = [{"line": 1.25, "home": 2.0, "away": 1.86, "home_open": 1.98,
                "away_open": 1.9, "line_open": 1.25, "bookmaker": "nowgoal"}]
    opening = {"odds_ou": {"line": 2.5, "over": 1.79, "under": 2.12},
               "odds_ah": {"line": 1.25, "home": 2.0, "away": 1.86}}

    q1 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8,
        cfg=dict(PROD), opening_snapshot=opening,
    )
    prev = _pick_payload(q1, "2026-08-15T18:33:31+00:00")
    q2 = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=ah_rows,
        movement_snapshot=None, context=None, completeness=0.8,
        cfg=dict(PROD), opening_snapshot=opening, previous_pick=prev,
        now_ts="2026-08-15T18:48:35+00:00",
    )
    msg = [
        f"q1.dec={q1['decision']} q1.pick={(q1.get('best_pick') or {}).get('selection')}"
        f" s={(q1.get('best_pick') or {}).get('score')}",
        f"prev.sel={prev.get('selection') if prev else None} prev.score={prev.get('score') if prev else None}"
        f" prev.lk={prev.get('line_key') if prev else None}",
        f"q2.dec={q2['decision']} q2.pick={(q2.get('best_pick') or {}).get('selection')}",
        f"q2.stability={q2.get('stability')}",
        "q2.rank=" + "; ".join(
            f"{r['selection']}/s={r['score']}/odds={r.get('market_odds')}"
            f"/v={r.get('vetoed')}/{(r.get('veto_reasons') or ['-'])[0][:38]}"
            for r in q2["ranking"][:4]
        ),
    ]
    raise AssertionError(" || ".join(msg))
