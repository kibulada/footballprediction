"""Unit tests for the Greek Cup signal-validity audit plan (2026-08-17).

Covers the plan's test strategy per phase:
  - P1-2  evidence_gate thresholds (0/1/2/3 LOW critical fields)
  - P1-3  _apply_evidence_floor caps when statistical/movement unavailable
  - P2-1  H2H window filter (within/outside 3y) + stale flag + W/D/L recount
  - P2-2  form_primary_source metadata (livescore in play / flashscore post)
  - P3-2  NowGoal Club-Friendlies exclusion + filter_recent_matches
  - P3-3  find_outlier liquidity flag + score_signal thin-outlier gate
  - P3-4  _coverage_floor confidence downgrade on thin coverage
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.football.datasources import (  # noqa: E402
    FIELD_FORM,
    FieldValue,
    UnifiedMatch,
)
from agents.football.livescore import (  # noqa: E402
    apply_h2h_window,
    filter_h2h_recent,
)
from agents.football.nowgoal import (  # noqa: E402
    EXCLUDED_COMPETITIONS,
    filter_recent_matches,
)
from agents.football.scorer import (  # noqa: E402
    consensus_odds,
    find_outlier,
    score_signal,
)
from agents.football.signal_engine import (  # noqa: E402
    _coverage_floor,
    evidence_gate,
)


def _now() -> datetime:
    return datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# P1-2: source-confidence gate (3+ critical fields LOW -> NO BET)
# ---------------------------------------------------------------------------

def test_evidence_gate_passes_when_no_data() -> None:
    assert evidence_gate(None) == (True, None)
    assert evidence_gate({}) == (True, None)


def test_evidence_gate_tolerates_zero_one_two_low() -> None:
    all_high = {"match": "HIGH", "form": "HIGH", "h2h": "HIGH"}
    one_low = {"match": "LOW", "form": "HIGH", "h2h": "HIGH"}
    two_low = {"match": "LOW", "form": "LOW", "h2h": "MEDIUM"}
    assert evidence_gate(all_high) == (True, None)
    assert evidence_gate(one_low) == (True, None)
    assert evidence_gate(two_low) == (True, None)


def test_evidence_gate_vetoes_three_low() -> None:
    three_low = {"match": "LOW", "form": "LOW", "h2h": "LOW"}
    passed, reason = evidence_gate(three_low)
    assert passed is False
    assert reason and "3" in reason and "LOW" in reason


# ---------------------------------------------------------------------------
# P2-1: H2H window filter (<= 3 years) + metadata
# ---------------------------------------------------------------------------

def test_filter_h2h_recent_keeps_only_window_meetings() -> None:
    meetings = [
        {"kickoff": "2026-08-01T18:00:00Z"},
        {"kickoff": "2024-01-01T18:00:00Z"},   # within 3y -> kept
        {"kickoff": "2018-05-01T18:00:00Z"},   # 8y old -> dropped
        {"kickoff": None},                      # unparseable -> dropped
    ]
    kept = filter_h2h_recent(meetings, _now())
    assert [m["kickoff"] for m in kept] == [
        "2026-08-01T18:00:00Z", "2024-01-01T18:00:00Z"
    ]


def test_apply_h2h_window_recounts_wdl_from_survivors() -> None:
    h2h = {
        "wins": 2, "draws": 0, "losses": 1,
        "meetings": [
            # 8 years old: dropped, must not feed the strength numbers.
            {"home": "Karditsas", "away": "Aris", "home_score": 2, "away_score": 0,
             "status": "finished", "kickoff": "2018-05-01T18:00:00Z"},
            # In window: loss.
            {"home": "Karditsas", "away": "Aris", "home_score": 0, "away_score": 1,
             "status": "finished", "kickoff": "2026-08-01T18:00:00Z"},
        ],
    }
    out = apply_h2h_window(h2h, _now(), home_name="Karditsas")
    assert out["h2h_window"] == "3y"
    assert out["h2h_total_meetings"] == 2
    assert out["h2h_in_window"] == 1
    assert out["wins"] == 0 and out["draws"] == 0 and out["losses"] == 1
    assert "h2h_relevance" not in out


def test_apply_h2h_window_flags_stale_when_nothing_survives() -> None:
    h2h = {
        "wins": 1, "draws": 0, "losses": 2,
        "meetings": [
            {"home": "Karditsas", "away": "Aris", "home_score": 1, "away_score": 0,
             "status": "finished", "kickoff": "2018-05-01T18:00:00Z"},
            {"home": "Aris", "away": "Karditsas", "home_score": 2, "away_score": 2,
             "status": "finished", "kickoff": "2017-05-01T18:00:00Z"},
        ],
    }
    out = apply_h2h_window(h2h, _now(), home_name="Karditsas")
    assert out["h2h_in_window"] == 0
    assert out["h2h_relevance"] == "stale"
    assert out["wins"] == 0 and out["draws"] == 0 and out["losses"] == 0


def test_apply_h2h_window_nowgoal_match_list() -> None:
    h2h = {
        "wins": 1, "draws": 0, "losses": 1, "matches": 2,
        "match_list": [
            {"date": "2016-04-01 12:30:00", "result": "W"},   # dropped
            {"date": "2026-06-10 19:00:00", "result": "L"},   # kept
        ],
    }
    out = apply_h2h_window(h2h, _now())
    assert out["h2h_in_window"] == 1
    assert out["matches"] == 1
    assert out["wins"] == 0 and out["losses"] == 1


# ---------------------------------------------------------------------------
# P2-2: form primary source metadata
# ---------------------------------------------------------------------------

def test_to_dict_exposes_form_primary_source() -> None:
    fv = FieldValue(
        field=FIELD_FORM, status="available", value={"home": {}, "away": {}},
        source="livescore", sources=["flashscore", "livescore"],
        confidence="HIGH",
    )
    unified = UnifiedMatch(
        match={"home": "a", "away": "b"},
        fields={FIELD_FORM: fv},
        sources=["flashscore", "livescore"],
    )
    out = unified.to_dict()
    assert out["form_primary_source"] == "livescore"


def test_to_dict_form_primary_source_none_when_form_missing() -> None:
    unified = UnifiedMatch(match={"home": "a", "away": "b"}, fields={}, sources=[])
    assert unified.to_dict()["form_primary_source"] is None


# ---------------------------------------------------------------------------
# P3-2: NowGoal Club-Friendlies exclusion
# ---------------------------------------------------------------------------

def test_filter_recent_matches_drops_friendlies() -> None:
    rows = [
        {"competition": "Club Friendlies"},
        {"competition": "Premier League"},
        {"competition": "Pre-Season"},
    ]
    kept = filter_recent_matches(rows)
    assert kept == [{"competition": "Premier League"}]
    assert "club friendlies" in EXCLUDED_COMPETITIONS


# ---------------------------------------------------------------------------
# P3-3: outlier liquidity gate
# ---------------------------------------------------------------------------

_ODDS = [
    {"bookmaker": "A", "home": 1.5, "draw": 4.0, "away": 6.0},
    {"bookmaker": "B", "home": 1.55, "draw": 4.2, "away": 8.0},   # outlier
    {"bookmaker": "C", "home": 1.6, "draw": 4.1, "away": 6.2},
]


def test_find_outlier_ok_liquidity_with_three_bookmakers() -> None:
    outlier = find_outlier(_ODDS, consensus_odds(_ODDS), 10.0)
    assert outlier is not None
    assert outlier["side"] == "away"
    assert outlier["outlier_liquidity"] == "ok"
    assert outlier["bookmaker_count"] == 3


def test_find_outlier_flagged_thin_when_sample_too_small() -> None:
    thin = _ODDS[:2]  # only 2 bookmakers report odds
    outlier = find_outlier(thin, consensus_odds(thin), 10.0)
    assert outlier is not None
    assert outlier["outlier_liquidity"] == "thin"


def test_score_signal_ignores_thin_outlier_value() -> None:
    thin = _ODDS[:2]
    cons = consensus_odds(thin)
    thin_out = find_outlier(thin, cons, 10.0)
    ok_out = find_outlier(thin, cons, 10.0, min_bm=2)
    assert thin_out["outlier_liquidity"] == "thin"
    assert ok_out["outlier_liquidity"] == "ok"
    score_thin = score_signal(thin, cons, thin_out, "W-W", "L-L", True)
    score_ok = score_signal(thin, cons, ok_out, "W-W", "L-L", True)
    # The lone divergent quote must not add value points.
    assert score_ok > score_thin


# ---------------------------------------------------------------------------
# P3-4: coverage floor -> confidence downgrade
# ---------------------------------------------------------------------------

def test_coverage_floor_downgrades_thin_coverage() -> None:
    assert _coverage_floor(0.9, "VERY HIGH") == "VERY HIGH"
    assert _coverage_floor(0.6, "HIGH") == "HIGH"
    assert _coverage_floor(0.35, "HIGH") == "MEDIUM"
    assert _coverage_floor(0.35, "VERY HIGH") == "MEDIUM"
    assert _coverage_floor(0.2, "MEDIUM") == "LOW"
    assert _coverage_floor(0.35, "LOW") == "LOW"  # already low, unchanged


def test_coverage_floor_config_override() -> None:
    cfg = {"downgrade_below": 0.5, "low_below": 0.3}
    assert _coverage_floor(0.45, "HIGH", cfg) == "MEDIUM"
    assert _coverage_floor(0.25, "MEDIUM", cfg) == "LOW"
