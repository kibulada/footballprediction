"""Tests for flashscore resolve improvements (alias matching + team-fixtures
fallback rows + suggest-endpoint parsing).

Pure functions only (no browser, no network): _squash_variants,
_find_pair_in_rows, _pick_suggest_team, _suggest_team failure path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.flashscore import (  # noqa: E402
    _find_pair_in_rows,
    _pick_sphinx_team,
    _pick_suggest_team,
    _squash_variants,
    _suggest_team,
)


def _row(home_name, away_name, home_slug="heart-of-midlothian", home_id="hQHPlg3X",
         away_slug="benfica", away_id="B1NF1CA1", match_url="https://www.flashscore.com/match/football/heart-of-midlothian-hQHPlg3X/benfica-B1NF1CA1/", score=None):
    return {
        "home_name": home_name, "away_name": away_name,
        "home_slug": home_slug, "home_id": home_id,
        "away_slug": away_slug, "away_id": away_id,
        "match_url": match_url, "date_text": "Today 02:00",
        "score": score,
    }


def test_squash_variants_includes_alias():
    # "Hearts" -> canonical "Heart of Midlothian" via team_alias (UEL table)
    variants = _squash_variants("Hearts")
    assert "hearts" in variants
    assert "heartofmidlothian" in variants


def test_squash_variants_unknown_query_single():
    assert _squash_variants("Random FC XYZ") == ["randomfcxyz"]


def test_squash_variants_includes_standings_abbreviation():
    # Regression (2026-08-17 wrong-team bug): "Atletico Madrid" must yield
    # the standings spelling "Atl. Madrid" (squashed "atlmadrid") so the
    # correct league-page row matches. Without it the team-fixtures fallback
    # ran and the poisoned alias variant could match a phantom fixture.
    variants = _squash_variants("Atletico Madrid")
    assert "atleticomadrid" in variants
    assert "atlmadrid" in variants
    assert "realmadridcf" not in variants


def test_find_pair_in_rows_matches_standings_abbreviation():
    # The league page renders "Atl. Madrid" -- the variant list from the
    # user's "Atletico Madrid" query must match it directly.
    rows = [_row("Atl. Madrid", "Malaga",
                 home_slug="atl-madrid", home_id="jaarqpLQ",
                 away_slug="malaga", away_id="25tIqYiJ")]
    found = _find_pair_in_rows(rows, _squash_variants("Atletico Madrid"), _squash_variants("Malaga"))
    assert found is not None
    assert found["home"]["name"] == "Atl. Madrid"
    assert found["away"]["name"] == "Malaga"


def test_find_pair_in_rows_rejects_phantom_real_madrid():
    # The phantom fixture (Real Madrid vs Malaga, a past-season match) must
    # NOT match a query about Atlético Madrid -- the poisoned "realmadridcf"
    # variant is gone.
    rows = [_row("Real Madrid", "Malaga",
                 home_slug="real-madrid", home_id="W8mj7MDD",
                 away_slug="malaga", away_id="25tIqYiJ")]
    found = _find_pair_in_rows(rows, _squash_variants("Atletico Madrid"), _squash_variants("Malaga"))
    assert found is None


def test_find_pair_in_rows_matches_alias_variant():
    # The raw query "hearts" cannot match "Heart of Midlothian", but the
    # alias variant can -- no extra render needed.
    rows = [_row("Heart of Midlothian", "Benfica")]
    found = _find_pair_in_rows(rows, _squash_variants("Hearts"), _squash_variants("Benfica"))
    assert found is not None
    assert found["home"]["name"] == "Heart of Midlothian"
    assert found["away"]["name"] == "Benfica"
    assert found["home"]["id"] == "hQHPlg3X"


def test_find_pair_in_rows_string_variants_backward_compatible():
    # Raw squash strings still work exactly as before.
    rows = [_row("Royale Union Saint-Gilloise", "Bodo/Glimt",
                 home_slug="royale-union-saint-gilloise", home_id="XXXX1111",
                 away_slug="bodo-glimt", away_id="YYYY2222")]
    found = _find_pair_in_rows(rows, "royaleunionsaintgilloise", "bodoglimt")
    assert found is not None
    assert found["home"]["name"] == "Royale Union Saint-Gilloise"


def test_find_pair_in_rows_team_fixture_row_carries_match_url():
    # Team-fixtures rows expose match_url (like league rows), so a pair found
    # there can be opened directly for stats/h2h.
    rows = [_row("Heart of Midlothian", "Benfica")]
    found = _find_pair_in_rows(rows, ["heartofmidlothian", "hearts"], ["benfica"])
    assert found is not None
    assert found["match_url"].startswith("https://www.flashscore.com/match/")


def test_find_pair_in_rows_carries_score():
    rows = [_row("Heart of Midlothian", "Benfica", score={"home": "1", "away": "2"})]
    found = _find_pair_in_rows(rows, ["heartofmidlothian"], ["benfica"])
    assert found is not None
    assert found["score"] == {"home": "1", "away": "2"}


def test_find_pair_in_rows_swapped_sides():
    rows = [_row("Benfica", "Heart of Midlothian")]
    found = _find_pair_in_rows(rows, ["heartofmidlothian"], ["benfica"])
    assert found is not None
    assert found["home"]["name"] == "Heart of Midlothian"  # caller's home kept
    assert found["away"]["name"] == "Benfica"


def test_find_pair_in_rows_no_match():
    assert _find_pair_in_rows([_row("A", "B")], ["c"], ["d"]) is None


def test_pick_suggest_team_structured_json():
    text = (
        '{"data":[[{"type":"team","id":"hQHPlg3X",'
        '"value":"Heart of Midlothian",'
        '"url":"/team/heart-of-midlothian/hQHPlg3X/"},'
        '{"type":"team","id":"XXXX","value":"Heart of Oak","url":"/team/heart-of-oak/XXXX/"}]]}'
    )
    assert _pick_suggest_team(text, "heartofmidlothian") == ("heart-of-midlothian", "hQHPlg3X")


def test_pick_suggest_team_raw_scan_fallback():
    # Unknown JSON shape: the slug pattern + query agreement still resolves.
    text = (
        'junk {"results":[["x","/team/heart-of-midlothian/hQHPlg3X/",1]]} junk'
    )
    assert _pick_suggest_team(text, ["hearts", "heartofmidlothian"]) == ("heart-of-midlothian", "hQHPlg3X")


def test_pick_suggest_team_unrelated_result_rejected():
    text = '{"data":[[{"id":"XXXX","value":"Some Other Team","url":"/team/some-other-team/XXXX/"}]]}'
    assert _pick_suggest_team(text, "heartofmidlothian") is None


def _sphinx_entry(name, tid, url, typ="Team", sport="Soccer"):
    return {
        "id": tid, "url": url, "name": name,
        "type": {"id": 2, "name": typ}, "sport": {"id": 1, "name": sport},
    }


def test_pick_sphinx_team_exact_match_wins_over_containment():
    """The livesport search relevance order must win: query "Beveren" must
    resolve SK Beveren (exact) -- NOT the lower-league Bosdam Beveren whose
    slug merely contains the query (the old longest-slug tiebreak bug)."""
    data = [
        _sphinx_entry("Beveren", "QaqfE8WE", "sk-beveren"),
        _sphinx_entry("Beveren U21", "GErk6iRn", "sk-beveren"),
        _sphinx_entry("Bosdam Beveren", "KO3zCtCc", "bosdam-beveren"),
        _sphinx_entry("Anderlecht", "vslqAKNo", "anderlecht-milan"),  # non-soccer? no, soccer
    ]
    assert _pick_sphinx_team(data, ["beveren"]) == ("sk-beveren", "QaqfE8WE")
    # Anderlecht must NOT resolve to "Anderlecht Milan" via containment when
    # the exact "Anderlecht" entry is present.
    data2 = [_sphinx_entry("Anderlecht Milan", "ATRG4VZs", "anderlecht-milan"),
             _sphinx_entry("Anderlecht", "vslqAKNo", "anderlecht")]
    assert _pick_sphinx_team(data2, ["anderlecht"]) == ("anderlecht", "vslqAKNo")


def test_pick_sphinx_team_filters_non_team_entries():
    data = [
        _sphinx_entry("Beveren", "QaqfE8WE", "sk-beveren"),
        _sphinx_entry("Some Player", "P1", "player", typ="Player"),
        _sphinx_entry("Basketball Team", "B1", "basket", sport="Basketball"),
    ]
    assert _pick_sphinx_team(data, ["beveren"]) == ("sk-beveren", "QaqfE8WE")


def test_pick_sphinx_team_alias_bridge():
    """The API names the team "Hearts" while the query is the canonical
    "Heart of Midlothian FC" -- the alias bridge resolves it."""
    data = [_sphinx_entry("Hearts", "0rGrwwNc", "hearts"),
            _sphinx_entry("Kelty Hearts", "UHbVkIde", "kelty-hearts")]
    assert _pick_sphinx_team(data, ["heartofmidlothianfc"]) == ("hearts", "0rGrwwNc")


def test_suggest_team_uses_livesport_search_primary():
    """The livesport search endpoint (working on this network) is tried FIRST;
    a hit returns without ever touching the DNS-blocked suggest hosts."""
    class _Resp:
        status_code = 200

        def json(self):
            return [_sphinx_entry("ADO Den Haag", "lAgLCkT3", "den-haag"),
                    _sphinx_entry("Den Haag W", "WY3qQqYl", "den-haag")]

    with patch("httpx.get", return_value=_Resp()) as mocked:
        assert _suggest_team("ADO Den Haag") == ("den-haag", "lAgLCkT3")
    assert mocked.call_count == 1


def test_suggest_team_falls_back_to_legacy_suggest_on_livesport_miss():
    """When the livesport search returns no team, the legacy suggest hosts are
    still tried (networks where only those answer)."""
    class _Empty:
        status_code = 200

        def json(self):
            return []

    class _LegacyHit:
        status_code = 200
        text = '{"data":[[{"type":"team","id":"hQHPlg3X","value":"Heart of Midlothian","url":"/team/heart-of-midlothian/hQHPlg3X/"}]]}'

    with patch("httpx.get", side_effect=[_Empty(), _LegacyHit()]) as mocked:
        assert _suggest_team("Hearts") == ("heart-of-midlothian", "hQHPlg3X")
    assert mocked.call_count == 2


def test_suggest_team_fails_silently_on_network_error():
    with patch("httpx.get", side_effect=Exception("network down")):
        assert _suggest_team("Hearts") is None


def test_suggest_team_fails_silently_on_bad_status():
    class _Resp:
        status_code = 403
        text = ""

    with patch("httpx.get", return_value=_Resp()):
        assert _suggest_team("Hearts") is None


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
