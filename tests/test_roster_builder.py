"""Roster builder (identity layer L0, plan 2026-08-24) unit tests."""
from __future__ import annotations

import json
from pathlib import Path

from agents.football.entity_registry import canonical_team_id
from agents.football.roster_builder import (
    build_rosters,
    coverage_report,
    main,
)


def _registry_entry(cid: str, name: str, league: str, provider_name: str) -> dict:
    return {
        "canonical_id": cid,
        "canonical_name": name,
        "league_key": league,
        "name": provider_name,
    }


def test_build_rosters_merges_aliases_and_provider_ids():
    teams = {
        "LIG": {
            "tm a": "Team A FC",
            "a fc": "Team A FC",
            "tm b": "Team B FC",
        }
    }
    # cid computed via the SAME function the live registry writes with --
    # never hardcode (the suffix-strip fallback would diverge).
    cid_a = canonical_team_id("LIG", "Team A FC")
    entries = {
        "flashscore": {
            "777": _registry_entry(cid_a, "Team A FC", "LIG", "Team A"),
        },
        "football_data": {
            "90210": _registry_entry(cid_a, "Team A FC", "LIG", "Team A FC"),
            "42": _registry_entry("t:other:x", "Other X", "OTHER", "X"),
        },
    }
    rosters = build_rosters(teams=teams, registry_entries=entries)

    assert set(rosters["LIG"]) == {"Team A FC", "Team B FC"}
    ta = rosters["LIG"]["Team A FC"]
    assert ta["aliases"] == ["a fc", "tm a"]  # sorted, both spellings kept
    assert ta["providers"]["flashscore"] == ["777"]
    assert ta["providers"]["football_data"] == ["90210"]
    assert ta["canonical_id"] == cid_a
    # Team B has no observed provider ids -> empty providers map, still listed.
    tb = rosters["LIG"]["Team B FC"]
    assert tb["aliases"] == ["tm b"]
    assert tb["providers"] == {}


def test_build_rosters_empty_registry_still_lists_canonicals():
    teams = {"LIG": {"x": "X FC"}}
    rosters = build_rosters(teams=teams, registry_entries={})
    assert rosters["LIG"]["X FC"]["canonical_id"].startswith("t:lig:")
    assert rosters["LIG"]["X FC"]["providers"] == {}


def test_coverage_report_counts():
    rosters = build_rosters(
        teams={"LIG": {"x": "X FC", "y": "Y FC"}},
        registry_entries={
            "flashscore": {
                "1": _registry_entry(
                    canonical_team_id("LIG", "X FC"), "X FC", "LIG", "X"
                ),
            },
        },
    )
    lines = coverage_report(rosters)
    assert len(lines) == 1
    assert "roster=  2" in lines[0]
    assert "dengan_provider_id=  1" in lines[0]


def test_main_writes_output_file(tmp_path, capsys):
    reg = tmp_path / "entity_registry.json"
    reg.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    out = tmp_path / "nested" / "rosters.json"
    rc = main(["--out", str(out), "--registry", str(reg)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data  # real teams.json leagues present
    assert "written" in capsys.readouterr().out


def test_main_report_mode_skips_write(tmp_path):
    reg = tmp_path / "missing.json"  # unreadable registry -> entries {}
    out = tmp_path / "rosters.json"
    rc = main(["--report", "--out", str(out), "--registry", str(reg)])
    assert rc == 0
    assert not out.exists()
