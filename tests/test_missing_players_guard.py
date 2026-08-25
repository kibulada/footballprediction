"""Regression tests for the missing-players home/away separation guards.

Verified failure 2026-08-23 (Club Brugge KV v Cercle Brugge KSV): the
snapshot logged 48 "missing" names for the AWAY side -- flashscore's real 4
Cercle absentees PLUS both squads' full rosters that leaked through the
nowgoal analysis-page injury slice and the cross-provider merge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.nowgoal import NowGoalClient  # noqa: E402
from agents.football.player_identity import merge_missing_lists  # noqa: E402


def _row(pid: int, pos: str, num: int, name: str) -> str:
    return (
        f'<div playerid="{pid}" class="player-row"><b>{pos}</b>'
        f"<span>{num}</span><a>{name}</a></div>"
    )


def _page(injury_html: str, *, with_standings: bool = True) -> str:
    tail = '<table class="team-table-home">[BE D1-3] Team A</table>' if with_standings else ""
    return f"Injury and Suspension{injury_html}{tail}"


# ---- _parse_injuries -------------------------------------------------------

def test_parse_injuries_home_and_away_sections():
    html = (
        'Injury and Suspension<div id="injuryH">'
        + _row(1, "CM", 8, "Home Out")
        + '</div><div id="injuryG">'
        + _row(2, "GK", 1, "Away Out")
        + "</div>"
    )
    out = NowGoalClient._parse_injuries(_page(html))
    assert out is not None
    assert [p["name"] for p in out["home"]] == ["Home Out"]
    assert [p["name"] for p in out["away"]] == ["Away Out"]


def test_parse_injuries_home_only():
    html = (
        'Injury and Suspension<div id="injuryH">'
        + _row(1, "CM", 8, "Home Out")
        + "</div>"
    )
    out = NowGoalClient._parse_injuries(_page(html))
    assert out is not None
    assert [p["name"] for p in out["home"]] == ["Home Out"]
    assert "away" not in out or not out.get("away")


def test_parse_injuries_away_only_bounded_by_next_section():
    # Club Brugge failure shape: NO injuryH marker; whatever follows injuryG
    # must be cut at the next section boundary, not run to page end.
    html = (
        'Injury and Suspension<div id="injuryG">'
        + _row(2, "GK", 1, "Away Out")
        + "</div>"
        "<p>Standings follow here</p>"
        '<table class="team-table-home">[BE D1-3] Team A rows</table>'
        + _row(99, "CF", 9, "Roster Noise After Standings")
    )
    out = NowGoalClient._parse_injuries(_page(html))
    assert out is not None
    names = [p["name"] for p in out.get("away") or []]
    assert names == ["Away Out"]
    assert "Roster Noise After Standings" not in names


def test_parse_injuries_swapped_marker_order():
    # Different section layout: guest container BEFORE home container.
    html = (
        'Injury and Suspension<div id="injuryG">'
        + _row(2, "GK", 1, "Away Out")
        + '</div><div id="injuryH">'
        + _row(1, "CM", 8, "Home Out")
        + "</div>"
    )
    out = NowGoalClient._parse_injuries(_page(html))
    assert out is not None
    assert [p["name"] for p in out["away"]] == ["Away Out"]
    assert [p["name"] for p in out["home"]] == ["Home Out"]


def test_parse_injuries_empty_section_returns_none():
    assert NowGoalClient._parse_injuries("no such section here") is None


# ---- merge_missing_lists guards -------------------------------------------

def _ng_side(names: list[str]) -> list[dict[str, object]]:
    return [
        {"player_id": str(i), "position": "CM", "number": str(i), "name": n}
        for i, n in enumerate(names)
    ]


def test_merge_home_absence_only():
    fs = {"home": {"missing": [{"name": "Ordonez J.", "reason": "Foot Injury"}], "unsure": []}}
    out = merge_missing_lists(fs, None)
    assert out is not None
    assert [e["name"] for e in out["home"]] == ["Ordonez J."]
    assert "away" not in out


def test_merge_away_absence_only():
    fs = {"away": {"missing": [{"name": "Herrmann C.", "reason": "Muscle Injury"}], "unsure": []}}
    out = merge_missing_lists(None, {})  # no nowgoal payload at all
    assert out is None  # nothing passed for nowgoal -> guard treats as absent

    out = merge_missing_lists(fs, None)
    assert out is not None
    assert [e["name"] for e in out["away"]] == ["Herrmann C."]
    assert "home" not in out


def test_merge_one_team_without_absence():
    fs = {
        "home": {"missing": [{"name": "A One", "reason": "X"}], "unsure": []},
        "away": {"missing": [], "unsure": []},
    }
    ng = {"home": _ng_side(["B Two"]), "away": []}
    out = merge_missing_lists(fs, ng)
    assert out is not None
    assert {e["name"] for e in out["home"]} >= {"A One", "B Two"}
    assert "away" not in out


def test_merge_duplicate_player_across_sides_dropped():
    # The same human reported for BOTH teams cannot be absent twice: the
    # away occurrence must be suppressed (home wins deterministically).
    fs = {
        "home": {"missing": [{"name": "Garcia K.", "reason": "Knock"}], "unsure": []},
        "away": {"missing": [], "unsure": []},
    }
    ng = {
        "home": [],
        "away": _ng_side(["Kike Garcia"]),
    }
    out = merge_missing_lists(fs, ng)
    assert out is not None
    assert [e["name"] for e in out["home"]] == ["Garcia K."]
    assert "away" not in out


def test_merge_dedupes_within_same_side():
    fs = {"home": {"missing": [{"name": "Puado J.", "reason": "Knee"}], "unsure": []}}
    ng = {"home": _ng_side(["Javi Puado"]), "away": []}
    out = merge_missing_lists(fs, ng)
    assert out is not None
    assert len(out["home"]) == 1
    assert sorted(out["home"][0]["sources"]) == ["flashscore", "nowgoal"]


def test_merge_squad_dump_guard_drops_nowgoal_noise():
    # Verified Club Brugge shape: one side's nowgoal list carries BOTH
    # squads (44 rows) while flashscore holds the real absentees. The whole
    # nowgoal contribution must be dropped, flashscore stands alone.
    dump = _ng_side(
        [f"Player Number {i:02d} Surname" for i in range(44)]
    )
    fs = {
        "home": {"missing": [{"name": "Ordonez J.", "reason": "Foot Injury"}], "unsure": []},
        "away": {"missing": [{"name": "Herrmann C.", "reason": "Muscle Injury"}], "unsure": []},
    }
    ng = {"home": [], "away": dump}
    out = merge_missing_lists(fs, ng)
    assert out is not None
    assert [e["name"] for e in out["home"]] == ["Ordonez J."]
    assert [e["name"] for e in out["away"]] == ["Herrmann C."]
    assert all(e["sources"] == ["flashscore"] for side in out.values() for e in side)


def test_merge_normal_volume_still_merges():
    # Below the dump threshold the merge behaves exactly as before.
    fs = {"home": {"missing": [{"name": "Garcia K.", "reason": "Hamstring"}], "unsure": []}}
    ng = {"home": _ng_side(["Enrique Garcia Martinez, Kike"]), "away": []}
    out = merge_missing_lists(fs, ng)
    assert out is not None
    assert len(out["home"]) == 1
    assert sorted(out["home"][0]["sources"]) == ["flashscore", "nowgoal"]
