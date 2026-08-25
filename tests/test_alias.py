"""Tests for team alias resolver."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.team_alias import resolve_team_alias, load_teams


def test_alias_exact_short_code():
    result = resolve_team_alias("MCN", "EPL")
    assert result == "Manchester City FC"


def test_alias_exact_full_name():
    result = resolve_team_alias("man utd", "EPL")
    assert result == "Manchester United FC"


def test_alias_exact_without_league():
    result = resolve_team_alias("MCN")
    assert result == "Manchester City FC"


def test_alias_case_insensitive():
    result = resolve_team_alias("mcn", "EPL")
    assert result == "Manchester City FC"


def test_alias_unknown():
    result = resolve_team_alias("xyz123unknown")
    assert result is None


def test_alias_nickname():
    result = resolve_team_alias("spurs", "EPL")
    assert result == "Tottenham Hotspur FC"


def test_alias_laliga():
    result = resolve_team_alias("barça", "LaLiga")
    assert result == "FC Barcelona"


def test_alias_saudi():
    result = resolve_team_alias("HILAL", "Saudi Pro League")
    assert result == "Al-Hilal Saudi FC"


def test_alias_unique_word_inside_canonical():
    """F4 (2026-08-18, split-identity bug): "Lecce" must resolve to the
    canonical "US Lecce" when it is a word inside exactly ONE Serie A club,
    so the odds poll ("US Lecce") and the analyse run ("Lecce") can never
    split one match into two match_ids (starving movement/statistical).
    """
    assert resolve_team_alias("Lecce", "Serie A") == "US Lecce"
    assert resolve_team_alias("US Lecce", "Serie A") == "US Lecce"
    # Mirrors resolve to the same canonical regardless of prefix spelling.
    assert resolve_team_alias("Milan", "Serie A") == "AC Milan"
    assert resolve_team_alias("AC Milan", "Serie A") == "AC Milan"


def test_alias_ambiguous_word_never_guessed():
    """F4 guard: a word inside MULTIPLE clubs in the league stays None -- a
    wrong guess would hijack the wrong team (the "madrid" hijack class).
    """
    assert resolve_team_alias("Real", "LaLiga") is None  # Real Madrid / Betis / Sociedad / Valladolid
    assert resolve_team_alias("Madrid", "LaLiga") == "Real Madrid CF"  # explicit alias, not F4
    # Unique accent-insensitive match resolves ("Atletico" -> "Atlético Madrid").
    assert resolve_team_alias("Atletico", "LaLiga") == "Atlético Madrid"


def test_alias_liga1():
    result = resolve_team_alias("PERSIB", "Liga 1")
    assert result == "Persib Bandung"


def test_alias_sabah_fk_ucl():
    # UCL qualifier 'SABAH FK' is the Azerbaijani club (Sabah Baku); the
    # alias must NOT point to the Malaysian 'Sabah' side.
    result = resolve_team_alias("SABAH FK", "UCL")
    assert result == "Sabah Baku"


def test_alias_agf_aarhus_ucl():
    result = resolve_team_alias("AGF", "UCL")
    assert result == "AGF Aarhus"


def test_alias_case_insensitive_ucl():
    result = resolve_team_alias("agf", "UCL")
    assert result == "AGF Aarhus"


def test_alias_union_sg_not_inter():
    """Regression: 'Union Saint-Gilloise' must NOT resolve via the 'INT'
    short code embedded inside 'saINT-Gilloise'."""
    result = resolve_team_alias("Union Saint-Gilloise", "UCL")
    assert result == "Union Saint-Gilloise"
    assert "Internazionale" not in (result or "")


def test_alias_short_code_word_boundary():
    # exact short code still resolves
    assert resolve_team_alias("INT", "UCL") == "FC Internazionale Milano"
    # a name merely containing 'int' inside a word (saINT) does not
    assert resolve_team_alias("Saint-Gilloise", "UCL") != "FC Internazionale Milano"
    # full canonical name resolves to itself
    assert resolve_team_alias("FC Internazionale Milano", "UCL") == "FC Internazionale Milano"


def test_alias_atletico_madrid_not_real_madrid():
    """Regression (2026-08-17 wrong-team bug): the user query 'Atletico
    Madrid' (ASCII) must resolve to Atlético Madrid, NOT Real Madrid CF.
    The old accent-sensitive exact-canonical pass failed on 'atletico' vs
    'atlético' and the generic 'MADRID' alias hijacked the word-boundary
    fallback -> phantom Real Madrid vs Malaga fixture on the flashscore
    team-fixtures path (nowgoal + flashscore league page both showed
    Atl. Madrid vs Malaga)."""
    assert resolve_team_alias("Atletico Madrid", None) == "Atlético Madrid"
    assert resolve_team_alias("Atlético Madrid", None) == "Atlético Madrid"
    assert resolve_team_alias("atletico madrid", None) == "Atlético Madrid"
    assert resolve_team_alias("ATM", "LaLiga") == "Atlético Madrid"
    assert resolve_team_alias("Real Madrid", None) == "Real Madrid CF"


def test_alias_deportivo_alaves_not_coruna():
    """Regression (2026-08-17): 'Deportivo Alaves' is Deportivo Alavés (a
    distinct club), NOT Deportivo de La Coruna. The generic 'DEPORTIVO'
    alias must not hijack the full club name."""
    assert resolve_team_alias("Deportivo Alaves", None) == "Deportivo Alavés"
    assert resolve_team_alias("Deportivo Alavés", None) == "Deportivo Alavés"
    assert resolve_team_alias("Alaves", None) == "Deportivo Alavés"
    assert resolve_team_alias("Deportivo", None) == "Deportivo de La Coruna"


def test_alias_racing_santander():
    assert resolve_team_alias("Racing Santander", None) == "Real Racing Club de Santander"
    assert resolve_team_alias("Racing", None) == "Real Racing Club de Santander"


def test_alias_accent_insensitive_leganes():
    assert resolve_team_alias("Leganes", None) == "Leganes"
    assert resolve_team_alias("Leganés", None) == "Leganes"


def test_teams_loads():
    teams = load_teams()
    assert "EPL" in teams
    assert "UCL" in teams
    assert "Liga 1" in teams


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
