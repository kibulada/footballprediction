"""P1.3 — oddspapi_teams fill-missing (acceptance tests).

Before P1.3, when flashscore + the provider chain failed, ``home_team`` /
``away_team`` were FULLY overwritten by ``oddspapi_teams`` (name + id),
breaking downstream form/H2H lookups that depend on the previously-resolved
id. P1.3 requires: the oddspapi fallback fills only missing fields; a full
overwrite happens only when nothing was resolved at all, and is then logged
as ``identity_source: oddspapi_fallback_full`` in the sources list.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.analyse import _merge_team_fields  # noqa: E402


def test_merge_fills_only_missing_fields():
    base = {"id": "flash-123", "name": "Arsenal", "provider": "flashscore"}
    extra = {"id": "odsp-999", "name": "Arsenal FC", "country": "England",
             "provider": "oddspapi", "_role": "oddspapi"}
    merged = _merge_team_fields(base, extra)
    # existing fields kept verbatim
    assert merged["id"] == "flash-123"
    assert merged["name"] == "Arsenal"
    assert merged["provider"] == "flashscore"
    # missing fields filled from oddspapi
    assert merged["country"] == "England"


def test_merge_fills_empty_name_but_keeps_id():
    base = {"id": "flash-123", "name": "", "provider": "flashscore"}
    extra = {"id": "odsp-999", "name": "Arsenal", "provider": "oddspapi"}
    merged = _merge_team_fields(base, extra)
    assert merged["id"] == "flash-123"      # resolved id survives
    assert merged["name"] == "Arsenal"      # empty name filled
    assert merged["provider"] == "flashscore"  # provider not overwritten


def test_merge_none_base_is_full_copy():
    extra = {"id": "odsp-999", "name": "Arsenal", "provider": "oddspapi"}
    merged = _merge_team_fields(None, extra)
    assert merged == extra


def test_merge_none_extra_keeps_base():
    base = {"id": "flash-123", "name": "Arsenal", "provider": "flashscore"}
    assert _merge_team_fields(base, None) == base
