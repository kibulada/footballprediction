"""Tests for elo.py."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.elo import EloModel


def test_initial_rating():
    elo = EloModel()
    assert elo.rating("A") == 1500.0
    assert elo.known("A", "B") is False


def test_expected_lambdas_home_advantage():
    elo = EloModel(home_advantage=65.0)
    lh, la = elo.expected_lambdas("A", "B")
    assert lh > la  # equal teams but home advantage


def test_expected_lambdas_strong_team():
    elo = EloModel()
    elo.ratings["Strong"] = 1700.0
    elo.ratings["Weak"] = 1300.0
    lh, la = elo.expected_lambdas("Strong", "Weak")
    assert lh > la
    assert 0.2 <= lh <= 3.8
    assert 0.2 <= la <= 3.8


def test_update_moves_ratings():
    elo = EloModel(k=32.0)
    elo.update("A", "B", 2, 0, persist=False)
    assert elo.rating("A") > 1500.0
    assert elo.rating("B") < 1500.0
    assert elo.games_played("A") == 1
    assert elo.games_played("B") == 1


def test_update_draw_moves_less():
    elo = EloModel(k=32.0)
    elo.update("A", "B", 1, 1, persist=False)
    # Home was expected to win (home advantage), so a draw underperforms:
    # home loses a little, away gains. Both move less than a decisive result.
    assert elo.rating("B") > 1500.0
    assert abs(elo.rating("A") - 1500.0) < 32.0
    assert abs(elo.rating("B") - 1500.0) < 32.0


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "elo.json"
        elo = EloModel(path=path)
        elo.update("A", "B", 3, 1, persist=True)
        elo2 = EloModel(path=path)
        assert elo2.rating("A") == elo.rating("A")
        assert elo2.games_played("B") == 1


def test_corrupt_file_does_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "elo.json"
        path.write_text("{not json", encoding="utf-8")
        elo = EloModel(path=path)
        assert elo.rating("X") == 1500.0


def _seeded_model(tmp: Path) -> EloModel:
    """EloModel with a small hand-written seed set + indexes rebuilt."""
    elo = EloModel(path=tmp / "elo.json")
    elo.ratings = {
        "Arsenal": 1700.0,
        "Liverpool": 1680.0,
        "Manchester City": 1720.0,
        "Manchester Utd": 1650.0,
        "Newcastle": 1560.0,
        "Celtic FC": 1520.0,
        "Rangers FC": 1510.0,
        "Bayern Munich": 1690.0,
        "Barcelona": 1660.0,
        "Inter": 1640.0,
        "Union Saint-Gilloise": 1480.0,
        "Union Berlin": 1485.0,
        "Olympique Lyonnais": 1500.0,
    }
    elo.games = {k: 30 for k in elo.ratings}
    elo._rebuild_indexes()
    return elo


def test_resolve_live_provider_spellings(tmp_path):
    """Live API names ("Arsenal FC", "FC Internazionale Milano") must hit the
    seeded ratings instead of silently falling back to the 1500 prior."""
    elo = _seeded_model(tmp_path)
    cases = {
        "Arsenal FC": "Arsenal",
        "Liverpool FC": "Liverpool",
        "arsenal": "Arsenal",  # case-insensitive
        "FC Bayern München": "Bayern Munich",  # diacritics stripped
        "FC Barcelona": "Barcelona",
        "FC Internazionale Milano": "Inter",  # teams.json alias
        "Royale Union Saint-Gilloise": "Union Saint-Gilloise",  # partial token
        "Manchester United": "Manchester Utd",  # synonym united~utd
        "Man City": "Manchester City",
        "Celtic": "Celtic FC",
    }
    for query, expected in cases.items():
        assert elo.resolve(query) == expected, f"{query!r} -> {elo.resolve(query)!r}"


def test_resolve_unknown_returns_none(tmp_path):
    elo = _seeded_model(tmp_path)
    assert elo.resolve("Sturm Graz") is None
    assert elo.resolve("FK Bodø/Glimt") is None
    assert elo.resolve("") is None
    assert elo.known("Arsenal FC", "Sturm Graz") is False
    assert elo.known("Arsenal FC", "Liverpool FC") is True


def test_resolve_tie_uses_contained_name_not_games(tmp_path):
    """A tied score must not be decided by game counts (arbitrary). Instead,
    the candidate whose ENTIRE token set is inside the query wins -- here
    'Newcastle' is fully contained in 'Newcastle United', 'Manchester Utd'
    is not."""
    elo = _seeded_model(tmp_path)
    assert elo.resolve("Newcastle United") == "Newcastle"
    assert elo.resolve("Manchester United") == "Manchester Utd"
    assert elo.resolve("Man City") == "Manchester City"


def test_resolve_ambiguous_returns_none(tmp_path):
    """'Union' alone matches Union Berlin AND Union Saint-Gilloise with equal
    partial scores and neither is fully contained in the query -> honest
    'not known' (None) instead of guessing one of them."""
    elo = _seeded_model(tmp_path)
    assert elo.resolve("Union") is None
    assert elo.known("Union", "Arsenal FC") is False
    # Full names still resolve to their own seed keys.
    assert elo.resolve("Union Berlin") == "Union Berlin"
    assert elo.resolve("Royale Union Saint-Gilloise") == "Union Saint-Gilloise"


def test_update_lands_on_seeded_key(tmp_path):
    """update() must resolve live spellings to the seeded key so ratings stay
    consistent across backtest seed + live bot."""
    elo = _seeded_model(tmp_path)
    before = elo.rating("Arsenal")
    elo.update("Arsenal FC", "Liverpool FC", 2, 0, persist=False)
    assert elo.rating("Arsenal") > before
    assert elo.games_played("Arsenal") == 31
    # The resolved key is used, not the provider spelling.
    assert "Arsenal FC" not in elo.ratings


def test_backtest_no_leakage_basic():
    """Ratings for a match must not include that match's own result."""
    elo = EloModel()
    before = (elo.rating("A"), elo.rating("B"))
    lh, la = elo.expected_lambdas("A", "B")
    elo.update("A", "B", 5, 0, persist=False)
    # prediction used ratings before update
    assert before == (1500.0, 1500.0)
    assert elo.rating("A") > before[0]
    assert math.isclose(lh + la, 2.7, abs_tol=0.01)


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
