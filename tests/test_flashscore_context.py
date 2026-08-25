"""Tests for the flashscore standings + match-info parsers (context data).

The parsers are pure (dict in -> dict out) so they are tested without a
browser, matching the project's parser-test convention. The DOM scrape
itself is exercised by live flow tests / manual probes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.flashscore import (  # noqa: E402
    FlashscoreBrowserClient,
    parse_match_info,
    parse_standings_rows,
)


def _standings_row(rank="1.", team="Arema FC", pts="0", form="?"):
    return {
        "rank": rank,
        "team": team,
        "cells": ["0", "0", "0", "0", "0:0", "0", "0"],
        "pts": pts,
        "form": form,
    }


def test_parse_standings_rows_basic():
    rows = parse_standings_rows([_standings_row()])
    assert rows is not None
    row = rows[0]
    assert row["pos"] == 1
    assert row["team"] == "Arema FC"
    assert row["mp"] == 0 and row["w"] == 0 and row["d"] == 0 and row["l"] == 0
    assert row["gf"] == 0 and row["ga"] == 0 and row["gd"] == 0
    assert row["pts"] == 0
    # '?' is the DOM's "form unavailable" marker and normalizes to None
    assert row["form"] is None


def test_parse_standings_rows_mid_season():
    row = {
        "rank": "3.",
        "team": "Persib Bandung",
        "cells": ["18", "10", "5", "3", "32:14", "18", "35"],
        "pts": "35",
        "form": "WDWLW",
    }
    rows = parse_standings_rows([row])
    assert rows is not None
    r = rows[0]
    assert r["pos"] == 3
    assert r["mp"] == 18 and r["w"] == 10 and r["d"] == 5 and r["l"] == 3
    assert r["gf"] == 32 and r["ga"] == 14 and r["gd"] == 18
    assert r["pts"] == 35
    assert r["form"] == "WDWLW"


def test_parse_standings_rows_skips_empty_team():
    rows = parse_standings_rows([{"rank": "1.", "team": "", "cells": [], "pts": None, "form": None}])
    assert rows is None


def test_parse_standings_rows_pts_fallback_cell():
    """When the dedicated pts cell is missing, fall back to the last value cell."""
    row = {
        "rank": "2.", "team": "Persija", "pts": None, "form": None,
        "cells": ["5", "3", "1", "1", "9:4", "5", "10"],
    }
    rows = parse_standings_rows([row])
    assert rows is not None
    assert rows[0]["pts"] == 10


def _match_info_labels():
    return {
        "labels": [
            "REFEREE:\nChiffi D.\n(Ita)",
            "VENUE:\nCentralnyj Stadion\n(Kostanay)",
            "CAPACITY:\n10 500",
        ],
        "neutral": True,
    }


def test_parse_match_info_full():
    out = parse_match_info(_match_info_labels())
    assert out is not None
    assert out["referee"] == "Chiffi D."
    assert out["referee_country"] == "Ita"
    assert out["venue"] == "Centralnyj Stadion"
    assert out["town"] == "Kostanay"
    assert out["capacity"] == "10 500"
    assert out["neutral"] is True


def test_parse_match_info_no_labels():
    assert parse_match_info({"labels": [], "neutral": False}) is None
    assert parse_match_info({}) is None


def test_parse_match_info_venue_without_town():
    out = parse_match_info({"labels": ["VENUE:\nRed Bull Arena"], "neutral": False})
    assert out is not None
    assert out["venue"] == "Red Bull Arena"
    assert "town" not in out


def test_scrape_standings_unregistered_league_no_browser():
    """Unregistered leagues must return None without launching a browser."""
    client = FlashscoreBrowserClient()
    assert client.scrape_league_standings("XYZ-Unknown-League") is None


def _tbl(names):
    return [{"team": n} for n in names]


def test_match_standings_team_exact_first():
    """Fix 2026-08-17: tiered standings matching -- exact normalized name is
    the first tier and wins over containment."""
    from agents.football.analyse import _match_standings_team

    tbl = _tbl(["Paris FC", "Paris Saint-Germain"])
    assert _match_standings_team(tbl, "Paris FC")["team"] == "Paris FC"
    assert _match_standings_team(tbl, "Paris Saint-Germain")["team"] == "Paris Saint-Germain"


def test_match_standings_team_containment_guard():
    """Fix 2026-08-17: containment is the LAST tier and ambiguity-guarded --
    a short target matching MORE THAN ONE row must return None (never guess
    the wrong club), while an unambiguous unique containment still matches.
    (A bare "Paris" target is NOT used here: teams.json aliases it to
    Paris Saint-Germain FC, so the alias tier resolves it -- the pipeline
    always passes the full resolved name anyway.)"""
    from agents.football.analyse import _match_standings_team

    # "Lens" is NOT usable as a probe anymore: teams.json aliases it to
    # "RC Lens" (Ligue 1 entry), so the alias tier legitimately resolves it
    # before containment -- same reason a bare "Paris" is excluded above.
    # "Royal" is unaliased, so the guarded containment tier is exercised.
    tbl = _tbl(["Royal Antwerp", "Royal Union"])
    assert _match_standings_team(tbl, "Royal") is None  # ambiguous: 2 rows
    # unique containment: only one row contains "Royal"
    tbl2 = _tbl(["Paris FC", "RC Lens", "Royal Antwerp"])
    assert _match_standings_team(tbl2, "Royal")["team"] == "Royal Antwerp"


def test_match_standings_team_alias_and_prefix():
    """Fix 2026-08-17: alias tier (psg -> Paris Saint-Germain) and club-token
    strip tier (FK Bodø/Glimt vs Bodø/Glimt) resolve before containment."""
    from agents.football.analyse import _match_standings_team

    tbl = _tbl(["Paris FC", "Paris Saint-Germain"])
    assert _match_standings_team(tbl, "psg")["team"] == "Paris Saint-Germain"
    # full-name club-token strip: FK Bodø/Glimt == Bodø/Glimt
    tbl2 = _tbl(["FK Bodø/Glimt", "Bodø/Glimt B"])
    assert _match_standings_team(tbl2, "Bodø/Glimt")["team"] == "FK Bodø/Glimt"


def test_match_standings_team_reverse_abbreviation():
    """B2/B5 fix 2026-08-17: flashscore standings tables abbreviate several
    clubs ("Paris SG", "Atl. Madrid") in ways the alias tier cannot reverse.
    A FULL canonical target ("Paris Saint-Germain FC") previously fell to the
    containment tier and latched onto "Paris FC" (verified live); the reverse
    abbreviation tier must map it to the "Paris SG" row first."""
    from agents.football.analyse import _match_standings_team

    tbl = _tbl(["Paris FC", "Paris SG"])
    assert _match_standings_team(tbl, "Paris Saint-Germain FC")["team"] == "Paris SG"
    assert _match_standings_team(tbl, "Paris Saint-Germain")["team"] == "Paris SG"
    assert _match_standings_team(tbl, "PSG")["team"] == "Paris SG"
    # unrelated club still maps to itself
    assert _match_standings_team(tbl, "Paris FC")["team"] == "Paris FC"

    # curated standings spelling ("Atl. Madrid") reached from the full name
    tbl2 = _tbl(["Atl. Madrid", "Paris FC"])
    assert _match_standings_team(tbl2, "Atletico Madrid")["team"] == "Atl. Madrid"
    assert _match_standings_team(tbl2, "Atlético Madrid")["team"] == "Atl. Madrid"
    # Athletic Club has its own abbreviation ("Ath Bilbao") and must not
    # capture the "Atl. Madrid" row
    assert _match_standings_team(tbl2, "Athletic Club") is None


def test_match_standings_team_segunda_backfill():
    """B5 fix 2026-08-17: teams.json Segunda backfill + B1 containment guard
    mean a Segunda target resolves to ITS OWN row -- "Dep. A Coruna" must
    never land on "Las Palmas" / "Albacete" via the 1-char token leak."""
    from agents.football.analyse import _match_standings_team

    tbl = _tbl(["Las Palmas", "Albacete", "Dep. A Coruna", "Real Oviedo"])
    assert _match_standings_team(tbl, "Las Palmas")["team"] == "Las Palmas"
    assert _match_standings_team(tbl, "Albacete")["team"] == "Albacete"
    assert _match_standings_team(tbl, "Dep. A Coruna")["team"] == "Dep. A Coruna"
    assert _match_standings_team(tbl, "Deportivo de La Coruna")["team"] == "Dep. A Coruna"
    assert _match_standings_team(tbl, "Real Oviedo")["team"] == "Real Oviedo"
    assert _match_standings_team(tbl, "Oviedo")["team"] == "Real Oviedo"


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
