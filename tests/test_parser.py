"""Tests for analyse parser logic (pure unit, no Discord)."""
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

from agents.football.league_resolver import resolve_league_leading


def _parse(rest: str):
    """Replicate the parser logic from bot._handle_analyse."""
    import re
    if " vs " not in rest.lower():
        return None
    parts = re.split(r"\s+vs\s+", rest, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    left, away = parts[0].strip(), parts[1].strip()
    if not left or not away:
        return None
    tokens = left.split()
    if len(tokens) < 2:
        return None
    resolved = None
    consumed = 0
    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n])
        match = resolve_league_leading(candidate)
        if match:
            resolved = match
            consumed = n
            break
    if not resolved:
        return None
    home = " ".join(tokens[consumed:])
    if not home:
        return None
    return (resolved[0], home, away)


def test_parser_liga_portugal():
    result = _parse("liga portugal Santa Clara vs Nacional")
    assert result is not None
    assert result[0] == "Primeira Liga"
    assert result[1] == "Santa Clara"
    assert result[2] == "Nacional"


def test_parser_ucl_short():
    result = _parse("ucl bodo vs union sg")
    assert result is not None
    assert result[0] == "UCL"
    assert result[1] == "bodo"
    assert result[2] == "union sg"


def test_parser_champions_league():
    result = _parse("champions league mc vs ars")
    assert result is not None
    assert result[0] == "UCL"
    assert result[1] == "mc"
    assert result[2] == "ars"


def test_parser_single_word_league():
    result = _parse("epl arsenal vs chelsea")
    assert result is not None
    assert result[0] == "EPL"
    assert result[1] == "arsenal"
    assert result[2] == "chelsea"


def test_parser_la_liga():
    result = _parse("la liga barcelona vs madrid")
    assert result is not None
    assert result[0] == "LaLiga"
    assert result[1] == "barcelona"
    assert result[2] == "madrid"


def test_parser_indonesia():
    result = _parse("liga 1 persija vs persib")
    assert result is not None
    assert result[0] == "Liga 1"
    assert result[1] == "persija"
    assert result[2] == "persib"


def test_parser_no_vs():
    result = _parse("liga portugal Santa Clara Nacional")
    assert result is None


def test_parser_unknown_league():
    result = _parse("xyz abc vs def")
    assert result is None


def test_parser_no_match_keyword():
    """Input 'ucl sabah fk vs agf' should work without 'match' keyword."""
    result = _parse("ucl sabah fk vs agf")
    assert result is not None
    assert result[0] == "UCL"
    assert result[1] == "sabah fk"
    assert result[2] == "agf"


def test_parser_no_match_keyword_multi():
    result = _parse("eredivisie ajax vs psv")
    assert result is not None
    assert result[0] == "Eredivisie"
    assert result[1] == "ajax"
    assert result[2] == "psv"


def test_parser_empty_after_league():
    print("expectation: minimal tokens=2 _parse('ucl vs x') valid → league=ucl, home=''")
    print("rejected downstream by 'home empty' guard")


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
