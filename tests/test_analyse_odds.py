"""Tests for analyse.py odds-key fallback (qualification rounds).

The UCL primary sport key (soccer_uefa_champs_league) has no fixtures while
play-off matches live under soccer_uefa_champs_league_qualification; the
lookup must try candidate keys in order and cache each separately.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.analyse import (  # noqa: E402
    _norm_team_name,
    _teams_match,
    extract_h2h_entries,
    find_match_odds_payload,
)

PRIMARY = "soccer_uefa_champs_league"
QUAL = "soccer_uefa_champs_league_qualification"


class _Cache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def get(self, key: str, ttl: int = 0) -> object | None:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


def test_primary_key_hit_no_fallback():
    async def runner():
        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[{"home_team": "A", "away_team": "B"}])
        cache = _Cache()
        payload, hit = await find_match_odds_payload(
            [PRIMARY, QUAL], "A", "B", odds, cache, {"odds": 900}
        )
        assert hit == PRIMARY
        assert payload["home_team"] == "A"
        # only the primary key was fetched
        odds.fetch_odds.assert_awaited_once_with(PRIMARY)
    asyncio.run(runner())


def test_fallback_qual_key_hit():
    async def runner():
        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(
            side_effect=[
                [{"home_team": "X", "away_team": "Y"}],                       # primary: no match
                [{"home_team": "Bodo/Glimt", "away_team": "Union Saint-Gilloise"}],  # qual: hit
            ]
        )
        cache = _Cache()
        payload, hit = await find_match_odds_payload(
            [PRIMARY, QUAL], "Bodo/Glimt", "Union Saint-Gilloise", odds, cache, {"odds": 900}
        )
        assert hit == QUAL
        assert payload["home_team"] == "Bodo/Glimt"
        assert odds.fetch_odds.await_count == 2
    asyncio.run(runner())


def test_cache_per_key_no_refetch():
    async def runner():
        odds = AsyncMock()
        # primary has no match for A-B, qual does -> both keys must be fetched
        odds.fetch_odds = AsyncMock(
            side_effect=[
                [{"home_team": "X", "away_team": "Y"}],
                [{"home_team": "A", "away_team": "B"}],
            ]
        )
        cache = _Cache()
        ttl = {"odds": 900}
        payload, hit = await find_match_odds_payload([PRIMARY, QUAL], "A", "B", odds, cache, ttl)
        assert hit == QUAL
        # both keys fetched on the first call
        assert odds.fetch_odds.await_count == 2
        # second call is served entirely from cache: no extra fetches
        payload2, hit2 = await find_match_odds_payload([PRIMARY, QUAL], "A", "B", odds, cache, ttl)
        assert hit2 == QUAL and payload2["away_team"] == "B"
        assert odds.fetch_odds.await_count == 2
    asyncio.run(runner())


def test_no_match_any_key():
    async def runner():
        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[{"home_team": "X", "away_team": "Y"}])
        cache = _Cache()
        payload, hit = await find_match_odds_payload(
            [PRIMARY, QUAL], "Z", "W", odds, cache, {"odds": 900}
        )
        assert payload is None
        assert hit is None
        assert odds.fetch_odds.await_count == 2
    asyncio.run(runner())


def test_norm_team_name_strips_accents_and_punct():
    # non-ASCII written as escapes so encoding quirks can't corrupt them
    assert _norm_team_name("Bod\u00f8/Glimt") == "bodo glimt"
    assert _norm_team_name("FK Kauno \u017dalgiris") == "fk kauno zalgiris"
    assert _norm_team_name("Hapoel Be'er Sheva") == "hapoel be er sheva"


def test_teams_match_exact_and_prefix():
    assert _teams_match("Bod\u00f8/Glimt", "Bod\u00f8/Glimt")
    assert _teams_match("FK Bod\u00f8/Glimt", "Bod\u00f8/Glimt")  # FK prefix
    assert _teams_match("Bod\u00f8/Glimt", "FK Bod\u00f8/Glimt")  # either side
    assert _teams_match("NK Celje", "Celje")
    assert _teams_match("Bodo/Glimt", "Bod\u00f8/Glimt")  # ASCII vs stroked


def test_teams_match_containment_honorific():
    # football-data resolves "Royale Union Saint-Gilloise", odds use
    # "Union Saint-Gilloise" -> containment must catch it.
    assert _teams_match("Royale Union Saint-Gilloise", "Union Saint-Gilloise")
    assert _teams_match("Union Saint-Gilloise", "Royale Union Saint-Gilloise")


def test_teams_match_no_false_positive():
    assert not _teams_match("A", "B")  # too short for containment
    assert not _teams_match("Barcelona", "Manchester City")
    assert not _teams_match("Athletic Club", "Atletico Madrid")


def test_teams_match_country_suffix_and_short_names():
    # flashscore homepage names carry a country suffix; odds providers use
    # the full name. Short names ("tobol", "rfs") must match too.
    assert _teams_match("Tobol (Kaz)", "Tobol Kostanay")
    assert _teams_match("Partizan (Srb)", "FK Partizan Belgrade")
    assert _teams_match("RFS (Lat)", "FC RFS")
    assert _teams_match("Dyn. Kyiv (Ukr)", "FC Dynamo Kyiv")
    assert _teams_match("Qarabag (Aze)", "Qarabag FK")
    assert _teams_match("Ilves (Fin)", "Tampereen Ilves")
    assert _teams_match("Rijeka (Cro)", "HNK Rijeka")
    assert _teams_match("Flora (Est)", "Tallinna FC Flora")
    assert _teams_match("Inter Escaldes (And)", "Inter Club de Escaldes")
    assert not _teams_match("Tobol (Kaz)", "Astana")


def test_empty_keys_returns_none():
    async def runner():
        odds = AsyncMock()
        cache = _Cache()
        payload, hit = await find_match_odds_payload([], "A", "B", odds, cache, {"odds": 900})
        assert payload is None and hit is None
        odds.fetch_odds.assert_not_awaited()
    asyncio.run(runner())


def _payload_with(home_out, away_out, home_price=2.1, away_price=3.6, draw_price=3.4):
    return {
        "home_team": home_out,
        "away_team": away_out,
        "bookmakers": [
            {
                "title": "Bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home_out, "price": home_price},
                            {"name": away_out, "price": away_price},
                            {"name": "Draw", "price": draw_price},
                        ],
                    }
                ],
            }
        ],
    }


def test_extract_h2h_entries_exact_names():
    o = chr(0xF8)  # o-slash, built at runtime to dodge encoding quirks
    payload = _payload_with("Bod" + o + "/Glimt", "Union Saint-Gilloise")
    entries = extract_h2h_entries(payload, "Bod" + o + "/Glimt", "Union Saint-Gilloise")
    assert len(entries) == 1
    assert entries[0]["bookmaker"] == "Bet365"
    assert entries[0]["home"] == 2.1
    assert entries[0]["away"] == 3.6
    assert entries[0]["draw"] == 3.4


def test_extract_h2h_entries_tolerant_names():
    # resolved names differ (FK prefix, Royale honorific) from outcome names
    o = chr(0xF8)
    payload = _payload_with("Bod" + o + "/Glimt", "Union Saint-Gilloise")
    entries = extract_h2h_entries(payload, "FK Bod" + o + "/Glimt", "Royale Union Saint-Gilloise")
    assert len(entries) == 1
    assert entries[0]["home"] == 2.1
    assert entries[0]["away"] == 3.6


def test_extract_h2h_entries_skips_incomplete():
    payload = {
        "bookmakers": [
            {
                "title": "X",
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "A", "price": 1.5}]}
                ],
            }
        ]
    }
    assert extract_h2h_entries(payload, "A", "B") == []


def test_find_match_odds_payload_rejects_ambiguous_cross_match():
    """A row whose home name also matches the AWAY candidates is ambiguous
    ("Inter" could be Inter Milan or part of "Inter Turku") and must not be
    accepted as the fixture -- pricing the wrong match is worse than no odds."""
    async def runner():
        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[
            {"home_team": "Inter", "away_team": "Inter Turku"},
        ])
        cache = _Cache()
        payload, hit = await find_match_odds_payload(
            [PRIMARY], "Inter", "Inter Turku", odds, cache, {"odds": 900}
        )
        assert payload is None
        assert hit is None
    asyncio.run(runner())


def test_find_match_odds_payload_prefers_exact_name():
    """Among valid rows the exact-name fixture wins over a fuzzy containment
    hit, so an "Inter Turku" row can never outrank the true "Inter"."""
    async def runner():
        odds = AsyncMock()
        odds.fetch_odds = AsyncMock(return_value=[
            {"home_team": "Inter Turku", "away_team": "Bodo/Glimt"},
            {"home_team": "Inter", "away_team": "Bodo/Glimt"},
        ])
        cache = _Cache()
        payload, hit = await find_match_odds_payload(
            [PRIMARY], "Inter", "Bodo/Glimt", odds, cache, {"odds": 900}
        )
        assert hit == PRIMARY
        assert payload["home_team"] == "Inter"
        assert payload["away_team"] == "Bodo/Glimt"
    asyncio.run(runner())


def test_extract_h2h_entries_ambiguous_side_skipped():
    """An outcome name matching BOTH sides ("Inter" matches home "Inter" and
    away "Inter Turku") is ambiguous -> assigned to neither side, and the
    incomplete entry is dropped instead of mispricing a team."""
    payload = _payload_with("Inter", "Inter Turku")
    entries = extract_h2h_entries(payload, "Inter", "Inter Turku")
    assert entries == []


def test_extract_h2h_entries_ambiguous_outcome_does_not_poison_siblings():
    """Only the ambiguous outcome is skipped; an unambiguous sibling outcome
    in the SAME market is still assigned (e.g. a provider listing the draw
    plus only one resolvable team keeps a valid entry)."""
    payload = {
        "home_team": "Inter",
        "away_team": "Inter Turku",
        "bookmakers": [
            {
                "title": "Bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Inter", "price": 1.5},
                            {"name": "Draw", "price": 3.4},
                            {"name": "Inter Turku", "price": 7.0},
                        ],
                    }
                ],
            }
        ],
    }
    # Both team outcomes are ambiguous (each matches both sides) -> dropped;
    # draw alone cannot form an entry.
    assert extract_h2h_entries(payload, "Inter", "Inter Turku") == []


def test_extract_h2h_entries_draw_and_unambiguous_sides():
    """Normal case still works: each outcome matches exactly one side and the
    draw is picked up by its exact name."""
    payload = _payload_with("Bodo/Glimt", "Union Saint-Gilloise")
    entries = extract_h2h_entries(payload, "Bodo/Glimt", "Union Saint-Gilloise")
    assert len(entries) == 1
    assert entries[0]["home"] == 2.1
    assert entries[0]["away"] == 3.6
    assert entries[0]["draw"] == 3.4


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

# ---- Sharp-book benchmark (2026-08-23): primary_bookmaker / prefer_bookmaker
from agents.football.analyse import extract_market_totals  # noqa: E402
from agents.football.scorer import consensus_odds  # noqa: E402


def _bm_row(name, h, d, a):
    return {"bookmaker": name, "home": h, "draw": d, "away": a}


def test_consensus_odds_primary_bookmaker_used_when_complete():
    rows = [
        _bm_row("Betfair", 2.10, 3.30, 3.40),
        _bm_row("Pinnacle", 2.05, 3.45, 3.60),
        _bm_row("SoftBook", 1.95, 3.20, 3.80),
    ]
    cons = consensus_odds(rows, primary_bookmaker="pinnacle")
    assert cons == {"home": 2.05, "draw": 3.45, "away": 3.60}


def test_consensus_odds_primary_falls_back_on_incomplete_quote():
    rows = [
        _bm_row("Pinnacle", 2.05, 0.0, 3.60),  # draw missing -> unusable
        _bm_row("Betfair", 2.10, 3.30, 3.40),
        _bm_row("SoftBook", 1.95, 3.20, 3.80),
    ]
    cons = consensus_odds(rows, primary_bookmaker="Pinnacle")
    assert cons["home"] == 2.05 and cons["away"] == 3.60  # median of 3/2 values
    assert cons["draw"] == 3.25  # median(3.30, 3.20)


def test_consensus_odds_primary_absent_keeps_median():
    rows = [_bm_row("A", 2.00, 3.00, 4.00), _bm_row("B", 2.20, 3.40, 3.20)]
    cons = consensus_odds(rows, primary_bookmaker="Pinnacle")
    assert cons == {"home": 2.10, "draw": 3.20, "away": 3.60}


def _totals_payload():
    return {
        "bookmakers": [
            {
                "title": "ThinMargin",
                "markets": [{
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": 1.95, "point": 2.5},
                        {"name": "Under", "price": 1.95, "point": 2.5},
                    ],
                }],
            },
            {
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.88, "point": 2.5},
                            {"name": "Under", "price": 2.02, "point": 2.5},
                        ],
                    },
                    {
                        "key": "btts",
                        "outcomes": [
                            {"name": "Yes", "price": 1.75},
                            {"name": "No", "price": 2.10},
                        ],
                    },
                ],
            },
            {
                "title": "Other",
                "markets": [{
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": 1.85, "point": 3.5},
                        {"name": "Under", "price": 2.05, "point": 3.5},
                    ],
                }],
            },
        ]
    }


def test_extract_market_totals_preferred_bookmaker_wins_its_lines():
    totals = extract_market_totals(_totals_payload(), prefer_bookmaker="Pinnacle")
    assert totals["Over 2.5"]["bookmaker"] == "Pinnacle"
    assert totals["Under 2.5"]["bookmaker"] == "Pinnacle"
    assert totals["BTTS Yes"]["bookmaker"] == "Pinnacle"
    # line Pinnacle does not quote: smallest-margin winner stands
    assert totals["Over 3.5"]["bookmaker"] == "Other"
    assert "_margin" not in totals["Over 2.5"] and "_preferred" not in totals["Over 2.5"]


def test_extract_market_totals_without_preference_keeps_margin_rule():
    totals = extract_market_totals(_totals_payload())
    # ThinMargin has the smallest margin on O/U 2.5 (1.90+1.96)
    assert totals["Over 2.5"]["bookmaker"] == "ThinMargin"
