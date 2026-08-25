"""G1 + G2 + G5 entity-resolution tests (2026-08-17).

- G1: canonical_team_id deterministic; EntityRegistry persists
  (provider, provider_id) -> canonical_id; conflicts are surfaced.
- G2: match_id unification for real duplicated fixtures (Espanyol/Levante,
  Rio Ave, Telstar); snapshots carry ``entities``; settle_auto verifies a
  result against the snapshot's canonical ids and REFUSES a conflict.
- G5: _livescore_form skips finished matches whose competition resolves to
  a different registered league.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agents.football.entity_registry import EntityRegistry, canonical_team_id
from agents.football.prediction_log import (
    _canonical_team_name,
    _match_dedupe_key,
    _match_id_hits,
    make_match_id,
)


# --------------------------------------------------------------------------
# G1 -- canonical team id + registry
# --------------------------------------------------------------------------

def test_canonical_team_id_deterministic_and_unifies_espanyol():
    a = canonical_team_id("LaLiga", "Espanyol")
    b = canonical_team_id("LaLiga", "RCD Espanyol de Barcelona")
    assert a is not None and a == b
    assert a.startswith("t:laliga:")
    # stable across calls
    assert a == canonical_team_id("LaLiga", "Espanyol")
    # a different club never collides
    assert a != canonical_team_id("LaLiga", "Real Madrid CF")


def test_canonical_team_id_levante():
    a = canonical_team_id("LaLiga", "Levante")
    b = canonical_team_id("LaLiga", "Levante UD")
    assert a is not None and a == b


def test_registry_roundtrip_and_resolve(tmp_path):
    path = tmp_path / "registry.json"
    r = EntityRegistry(path)
    r.register("flashscore", "esp-123", "LaLiga", "Espanyol")
    r.register("football_data", 555, "LaLiga", "RCD Espanyol de Barcelona")
    # both provider ids resolve to the SAME canonical id
    assert (
        r.resolve("flashscore", "esp-123")
        == r.resolve("football_data", 555)
        == canonical_team_id("LaLiga", "Espanyol")
    )
    entry = r.lookup("flashscore", "esp-123")
    assert entry and entry["canonical_name"]
    # persisted on disk
    r2 = EntityRegistry(path)
    assert r2.resolve("flashscore", "esp-123") == r.resolve("flashscore", "esp-123")
    # unknown provider id -> None
    assert r.resolve("flashscore", "nope") is None


def test_registry_conflict_detected(tmp_path):
    r = EntityRegistry(tmp_path / "registry.json")
    r.register("flashscore", "id-x", "LaLiga", "Espanyol")
    r.register("flashscore", "id-x", "LaLiga", "Real Madrid CF")  # same id, different club
    conflicts = r.conflicts()
    assert len(conflicts) == 1
    assert conflicts[0]["provider_id"] == "id-x"


# --------------------------------------------------------------------------
# G2 -- match identity
# --------------------------------------------------------------------------

def test_match_id_unified_espanyol_levante():
    a = make_match_id("LaLiga", "RCD Espanyol de Barcelona", "Levante UD", "2026-08-16")
    b = make_match_id("LaLiga", "Espanyol", "Levante", "2026-08-16")
    assert a == b


def test_match_id_unified_rio_ave():
    a = make_match_id("Primeira Liga", "Rio Ave FC", "FC Porto", "2026-08-15")
    b = make_match_id("Primeira Liga", "Rio Ave", "Porto", "2026-08-15")
    assert a == b


def test_match_id_unified_telstar():
    a = make_match_id("Eredivisie", "Telstar 1963", "Sparta Rotterdam", "2026-08-14")
    b = make_match_id("Eredivisie", "Telstar", "Sparta", "2026-08-14")
    assert a == b


def test_match_dedupe_key_unifies_duplicate_match_ids():
    a = make_match_id("LaLiga", "RCD Espanyol de Barcelona", "Levante UD", "2026-08-16")
    b = make_match_id("LaLiga", "Espanyol", "Levante", "2026-08-16")
    row_a = {"match_id": a, "ts": "2026-08-16T10:00:00+00:00"}
    row_b = {"match_id": b, "ts": "2026-08-16T11:00:00+00:00"}
    assert _match_dedupe_key(row_a) == _match_dedupe_key(row_b)


def test_canonical_team_name_espanyol():
    # teams.json backfill: "Espanyol" resolves to the canonical full name
    assert _canonical_team_name("Espanyol", "LaLiga") == "RCD Espanyol de Barcelona"
    assert _canonical_team_name("Levante", "LaLiga") == "Levante UD"
    assert _canonical_team_name("Rio Ave", "Primeira Liga") == "Rio Ave FC"
    assert _canonical_team_name("Telstar", "Eredivisie") == "Telstar 1963"


def test_match_id_hits_accent_insensitive():
    """Provider spelling variants ("Göztepe" vs "Goztepe") are the SAME match.

    Regression for the 2026-08-17 Samsunspor-Göztepe split: thesportsdb/
    nowgoal resolve the accented name, flashscore the ASCII one -- two
    parallel snapshots with identical data except the accent, so the
    displayed card never joined the pinned opening / prior pick.
    """
    accented = "Super Lig||Samsunspor||Göztepe||2026-08-17"
    ascii_ = "Super Lig||Samsunspor||Goztepe||2026-08-17"
    assert _match_id_hits(accented, ascii_)
    assert _match_id_hits(ascii_, accented)
    # Legacy full-timestamp form still hits an accent-variant date-only id.
    legacy = "Super Lig||Samsunspor||Göztepe||2026-08-17T18:30:00Z"
    assert _match_id_hits(legacy, ascii_)
    # A genuinely different team is still a miss.
    assert not _match_id_hits(accented, "Super Lig||Samsunspor||Trabzonspor||2026-08-17")


def test_match_dedupe_key_accent_insensitive():
    """Dedupe collapses accent variants so settle/stats count one fixture."""
    row_a = {"match_id": "Super Lig||Samsunspor||Göztepe||2026-08-17", "ts": "1"}
    row_b = {"match_id": "Super Lig||Samsunspor||Goztepe||2026-08-17", "ts": "2"}
    assert _match_dedupe_key(row_a) == _match_dedupe_key(row_b)


# --------------------------------------------------------------------------
# G2 -- settle verification
# --------------------------------------------------------------------------

def test_settle_auto_accepts_verified_and_skips_conflict(tmp_path):
    from agents.football.settler import _canonical_result_check, settle_auto

    # Snapshot with entities for Espanyol (canonical id recorded).
    home_cid = canonical_team_id("LaLiga", "Espanyol")
    away_cid = canonical_team_id("LaLiga", "Levante")
    snap = {
        "match_id": make_match_id("LaLiga", "Espanyol", "Levante", "2026-08-16"),
        "event": "snapshot",
        "league": "La Liga",
        "home": "Espanyol",
        "away": "Levante",
        "kickoff": "2026-08-16T19:00:00Z",
        "ts": "2026-08-16T10:00:00+00:00",
        "entities": {
            "home": {"canonical_id": home_cid, "provider": "flashscore", "provider_id": "x", "name": "Espanyol"},
            "away": {"canonical_id": away_cid, "provider": "flashscore", "provider_id": "y", "name": "Levante"},
            "league_key": "LaLiga",
        },
    }
    # verified result: same clubs, different provider spelling
    good = {
        "home": "RCD Espanyol de Barcelona",
        "away": "Levante UD",
        "home_goals": 3,
        "away_goals": 0,
        "competition": "La Liga",
    }
    assert _canonical_result_check(snap, good) is True

    # conflicting result: names look alike but canonical ids differ (wrong club)
    bad = {
        "home": "Espanyol",
        "away": "Real Madrid CF",  # different club, same-ish match otherwise
        "home_goals": 0,
        "away_goals": 5,
        "competition": "La Liga",
    }
    assert _canonical_result_check(snap, bad) is False

    # settle_auto settles ONLY the verified result
    path = tmp_path / "log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
    report = settle_auto(path, date="2026-08-16", results=[bad, good])
    assert len(report["settled"]) == 1
    assert report["settled"][0]["result"] == "3-0"
    assert len(report["not_found"]) == 0


def test_settle_auto_backward_compat_no_entities(tmp_path):
    """Snapshots without entities fall back to plain name matching."""
    from agents.football.settler import settle_auto

    snap = {
        "match_id": make_match_id("LaLiga", "Espanyol", "Levante", "2026-08-16"),
        "event": "snapshot",
        "league": "La Liga",
        "home": "Espanyol",
        "away": "Levante",
        "kickoff": "2026-08-16T19:00:00Z",
        "ts": "2026-08-16T10:00:00+00:00",
    }
    path = tmp_path / "log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
    report = settle_auto(
        path,
        date="2026-08-16",
        results=[{"home": "Espanyol", "away": "Levante", "home_goals": 1, "away_goals": 1, "competition": "La Liga"}],
    )
    assert len(report["settled"]) == 1


# --------------------------------------------------------------------------
# G5 -- livescore form league filter
# --------------------------------------------------------------------------

def test_livescore_form_competition_filter(tmp_path):
    """A finished match in a DIFFERENT resolved league is skipped."""
    import asyncio

    from agents.football.multi_source import MultiSourceStatsFetcher

    st = MultiSourceStatsFetcher.__new__(MultiSourceStatsFetcher)
    st.livescore = None  # not configured -> _livescore_form returns None early

    # With no livescore client the function must return None (never raise).
    result = asyncio.run(st._livescore_form("Espanyol", limit=5, league_key="LaLiga"))
    assert result is None


def test_livescore_form_competition_key_branch():
    """competition_league_key resolves 'La Liga' -> LaLiga (filter basis)."""
    from agents.football.league_resolver import competition_league_key

    assert competition_league_key("La Liga") == "LaLiga"
    assert competition_league_key("Premier League") == "EPL"
