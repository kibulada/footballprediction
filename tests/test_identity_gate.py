"""Identity firewall (plan 2026-08-24): unit + regression tests.

Regression set = every documented wrong-match incident:
  - F3 (2026-08-17): "Atletico Madrid" hijacked onto Real Madrid CF ->
    phantom fixture analysed.
  - P0 (2026-08-24): bare "Barcelona" -> RCD Espanyol de Barcelona.
  - Opponent flip (2026-08-23): Forest vs Leeds on one run, Forest vs Man
    Utd on another (same league/date) -- pre-flight history lock must
    abort BEFORE the analysis burns its budget.

Policy asserted everywhere: refuse only on POSITIVE divergence evidence;
unknown spellings / dynamic leagues / broken logs are fail-open (status ok).
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.football.identity_gate import (
    canonical_side,
    check_pair_identity,
    firewall_cfg,
    preflight_history_lock,
    refusal_payload,
)
from agents.football.team_alias import resolve_team_alias


# ---------------------------------------------------------------------------
# canonical_side: strict resolution, never guesses
# ---------------------------------------------------------------------------

def test_canonical_side_resolves_league_canonical():
    assert canonical_side("Real Madrid CF", "LaLiga") == "Real Madrid CF"


def test_canonical_side_accent_insensitive_query():
    # F3 query spelling (ASCII) must land on the accented canonical.
    assert resolve_team_alias("Atletico Madrid", "LaLiga") == "Atlético Madrid"
    assert canonical_side("Atlético Madrid", "LaLiga") == "Atlético Madrid"


def test_canonical_side_unknown_name_returns_none_not_guess():
    assert canonical_side("Klub Fanta Bulan Jupiter", "LaLiga") is None
    assert canonical_side("", "EPL") is None
    assert canonical_side(None, "EPL") is None


# ---------------------------------------------------------------------------
# check_pair_identity: the two historical wrong-club vectors must REFUSE
# ---------------------------------------------------------------------------

def test_f3_regression_atletico_hijacked_to_real_madrid_refuses():
    res = check_pair_identity(
        "Atlético Madrid",
        "Malaga",
        league_key="LaLiga",
        detected_home="Real Madrid CF",
        detected_away=None,
    )
    assert res["status"] == "refuse"
    assert any("HOME" in r for r in res["reasons"])


def test_p0_regression_barcelona_resolved_to_espanyol_refuses():
    res = check_pair_identity(
        "Barcelona",
        "Alaves",
        league_key="LaLiga",
        resolved_home="RCD Espanyol de Barcelona",
        resolved_away=None,
    )
    assert res["status"] == "refuse"
    assert any("HOME" in r for r in res["reasons"])


def test_same_club_provider_spelling_passes():
    res = check_pair_identity(
        "Manchester United",
        "Liverpool FC",
        league_key="EPL",
        resolved_home="Manchester Utd",
        resolved_away="Liverpool FC",
    )
    assert res["status"] == "ok"


def test_confirmed_side_swap_warns_but_does_not_refuse():
    res = check_pair_identity(
        "Arsenal FC",
        "Chelsea FC",
        league_key="EPL",
        detected_home="Chelsea FC",
        detected_away="Arsenal FC",
    )
    assert res["status"] == "warn"
    assert res["reasons"]


def test_dynamic_league_and_unknown_names_fail_open():
    res = check_pair_identity(
        "Tim X", "Tim Y", league_key="dyn:piala-dunia-cup",
        detected_home="Tim X FC", detected_away="Tim Y United",
    )
    assert res["status"] == "ok"


def test_no_provider_pairs_is_ok():
    res = check_pair_identity("A", "B", league_key="EPL")
    assert res["status"] == "ok"
    assert res["checks"] == []


# ---------------------------------------------------------------------------
# refusal_payload shape (bot/format render payload["error"] verbatim)
# ---------------------------------------------------------------------------

def test_refusal_payload_shape():
    p = refusal_payload(
        "preflight_fixture", ["HOME berbeda klub"],
        display="LaLiga", home_query="A", away_query="B",
        detail={"kind": "same_id"},
    )
    assert p["error"].startswith("IDENTITY GUARD")
    assert p["identity_guard"]["stage"] == "preflight_fixture"
    assert p["identity_guard"]["detail"] == {"kind": "same_id"}
    assert p["league"] == "LaLiga"


# ---------------------------------------------------------------------------
# G-C: pre-flight history lock reuses identity_lock_check semantics
# ---------------------------------------------------------------------------

def _write_log(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return p


def test_preflight_history_lock_catches_opponent_flip(tmp_path):
    log = _write_log(tmp_path, [
        {
            "event": "snapshot",
            "match_id": "EPL||Nottingham Forest FC||Leeds United FC||2026-08-20",
            "ts": "2026-08-20T10:00:00+00:00",
            "home": "Nottingham Forest FC",
            "away": "Leeds United FC",
        }
    ])
    lock = preflight_history_lock(
        log,
        match_id=(
            "EPL||cid:t:epl:nottingham-forest-fc"
            "||cid:t:epl:manchester-united-fc||2026-08-20"
        ),
        home="Nottingham Forest FC",
        away="Manchester United FC",
        now_ts="2026-08-21T00:00:00+00:00",
    )
    assert lock and lock.get("locked") and lock.get("kind") == "opponent_flip"


def test_preflight_history_lock_passes_consistent_pair(tmp_path):
    log = _write_log(tmp_path, [
        {
            "event": "snapshot",
            "match_id": "EPL||Nottingham Forest FC||Leeds United FC||2026-08-20",
            "ts": "2026-08-20T10:00:00+00:00",
            "home": "Nottingham Forest FC",
            "away": "Leeds United FC",
        }
    ])
    lock = preflight_history_lock(
        log,
        match_id="EPL||Nottingham Forest FC||Leeds United FC||2026-08-20",
        home="Nottingham Forest FC",
        away="Leeds United FC",
        now_ts="2026-08-21T00:00:00+00:00",
    )
    assert lock is None


def test_preflight_history_lock_fail_open_on_missing_or_broken_log(tmp_path):
    missing = tmp_path / "nope.jsonl"
    assert preflight_history_lock(
        missing,
        match_id="EPL||a||b||2026-01-01", home="a", away="b",
    ) is None

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    assert preflight_history_lock(
        broken,
        match_id="EPL||a||b||2026-01-01", home="a", away="b",
    ) is None


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def test_firewall_cfg_defaults_and_overrides():
    cfg = firewall_cfg(None)
    assert cfg["enabled"] is True
    assert cfg["refuse_divergence"] is True
    assert cfg["refuse_history_lock"] is True

    cfg2 = firewall_cfg({"identity_firewall": {"enabled": False}})
    assert cfg2["enabled"] is False
    assert cfg2["refuse_divergence"] is True  # default preserved
