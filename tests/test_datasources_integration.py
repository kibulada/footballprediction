"""Phase 2 integration tests: multi-source layer wired into the production flow.

Exercises the real integration points (analyse._primary_fields,
analyse._build_secondary_source, datasources.aggregate_collected,
context.build_match_context source_meta) plus team-alias identity resolution.
All offline -- fake secondary sources, no network.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football import analyse  # noqa: E402
from agents.football.context import build_match_context  # noqa: E402
from agents.football.datasources import (  # noqa: E402
    CONF_HIGH,
    STATUS_AVAILABLE,
    STATUS_EMPTY,
    STATUS_UNAVAILABLE,
    FootballDataSource,
    aggregate_collected,
    available,
    missing,
    same_match,
    teams_match,
)
from agents.football.team_alias import canonical_abbreviation  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class FakeSecondary(FootballDataSource):
    def __init__(self, name="livescore", fields=None, calls=None):
        super().__init__(name)
        self.fields = fields or {}
        self.calls = calls if calls is not None else []

    async def fetch_fields(self, ref):
        self.calls.append(self.name)
        return dict(self.fields)


class ExplodingSecondary(FootballDataSource):
    def __init__(self, name="livescore"):
        super().__init__(name)

    async def fetch_fields(self, ref):
        raise RuntimeError("livescore down")


def _cfg(livescore_enabled=False, validation=True):
    # FLAT data_sources config (what MultiSourceAggregator consumes).
    return {
        "priority": {"flashscore": 100, "livescore": 80},
        "validation": {"enabled": validation},
        "livescore": {"enabled": livescore_enabled, "base_url": ""},
    }


def _full_cfg(livescore_enabled=False):
    return {"data_sources": _cfg(livescore_enabled=livescore_enabled)}


# ---- Test A: LiveScore disabled -> flashscore only -------------------------

def test_secondary_source_disabled_is_none():
    assert analyse._build_secondary_source(_full_cfg(livescore_enabled=False)) is None


def test_secondary_source_enabled_builds_adapter():
    src = analyse._build_secondary_source(_full_cfg(livescore_enabled=True))
    assert src is not None
    assert src.name == "livescore"


# ---- Test C/D: field-level fallback from primary -> secondary --------------

def _primary(**overrides):
    fields = {
        "match": available({"home": "A", "away": "B"}),
        "form": available({"home": ["W"], "away": ["L"]}),
        "h2h": available({"wins": 1, "draws": 0, "losses": 0}),
        "lineup": missing(),
        "injuries": missing(),
        "standings": available({"tables": {"overall": []}}),
        "statistics": missing(),
    }
    fields.update(overrides)
    return fields


def test_fallback_fills_missing_lineup_and_injuries():
    secondary = FakeSecondary(fields={
        "lineup": available({"home": ["GK"], "away": ["GK"]}),
        "injuries": available({"home": ["Injured A"], "away": []}),
    })
    unified = _run(aggregate_collected(
        primary_name="flashscore", primary_fields=_primary(),
        secondary=secondary, ref={"home": "A", "away": "B"}, config=_cfg(),
    ))
    assert unified.fields["lineup"].source == "livescore"
    assert unified.fields["injuries"].source == "livescore"
    # primary fields untouched
    assert unified.fields["form"].source == "flashscore"
    assert unified.fields["h2h"].source == "flashscore"


# ---- Test E/F: validation ---------------------------------------------------

def test_agreement_true_when_sources_agree():
    secondary = FakeSecondary(fields={"form": available({"home": ["W"], "away": ["L"]})})
    unified = _run(aggregate_collected(
        primary_name="flashscore", primary_fields=_primary(),
        secondary=secondary, ref={"home": "A", "away": "B"}, config=_cfg(),
    ))
    fv = unified.fields["form"]
    assert fv.agreement is True
    assert fv.confidence == CONF_HIGH
    assert set(fv.sources) == {"flashscore", "livescore"}


def test_disagreement_preserves_primary_and_secondary():
    secondary = FakeSecondary(fields={"form": available({"home": ["W", "W"], "away": ["L", "L"]})})
    unified = _run(aggregate_collected(
        primary_name="flashscore", primary_fields=_primary(),
        secondary=secondary, ref={"home": "A", "away": "B"}, config=_cfg(),
    ))
    fv = unified.fields["form"]
    assert fv.discrepancy is True
    assert fv.source == "flashscore"
    assert fv.value == {"home": ["W"], "away": ["L"]}
    assert fv.secondary and fv.secondary[0]["value"] == {"home": ["W", "W"], "away": ["L", "L"]}


# ---- Test G: secondary failure -> continue ----------------------------------

def test_secondary_failure_does_not_crash():
    unified = _run(aggregate_collected(
        primary_name="flashscore", primary_fields=_primary(),
        secondary=ExplodingSecondary(), ref={"home": "A", "away": "B"}, config=_cfg(),
    ))
    assert unified.sources == ["flashscore"]
    assert unified.fields["form"].source == "flashscore"
    assert unified.fields["lineup"].status == STATUS_UNAVAILABLE


# ---- Test B: no unnecessary secondary request -------------------------------

def test_lazy_fallback_skips_secondary_when_complete():
    calls: list[str] = []
    secondary = FakeSecondary(fields={"lineup": available({})}, calls=calls)
    _run(aggregate_collected(
        primary_name="flashscore", primary_fields=_primary(),
        secondary=secondary, ref={"home": "A", "away": "B"},
        config=_cfg(validation=False),
        required_fields=["match", "form", "h2h", "standings"],
    ))
    assert calls == []


# ---- _primary_fields mapping (known-empty vs unavailable) -------------------

def test_primary_fields_distinguish_empty_from_unavailable():
    fields = analyse._primary_fields(
        home="Arsenal", away="Chelsea", kickoff="2026-08-16T16:00:00Z", competition="EPL",
        home_form={"sequence": "W-W-D"}, away_form={"sequence": "L-D-W"},
        h2h={"wins": 0, "draws": 0, "losses": 0},
        lineups=None,
        missing_players={"home": {"missing": []}, "away": {"missing": []}},
        standings=None,
        match_stats=None,
    )
    assert fields["match"].status == STATUS_AVAILABLE
    assert fields["match"].value["home"] == "arsenal"  # canonical identity
    assert fields["form"].status == STATUS_AVAILABLE
    assert fields["h2h"].status == STATUS_EMPTY          # explicitly no H2H meetings
    assert fields["lineup"].status == STATUS_UNAVAILABLE  # never fetched
    assert fields["injuries"].status == STATUS_EMPTY      # explicitly no injuries
    assert fields["standings"].status == STATUS_UNAVAILABLE


def test_primary_fields_injuries_available_when_missing_players():
    fields = analyse._primary_fields(
        home="A", away="B", kickoff=None, competition=None,
        home_form=None, away_form=None, h2h=None,
        lineups=None,
        missing_players={"home": {"missing": [{"name": "X"}]}, "away": {"missing": []}},
        standings=None, match_stats=None,
    )
    assert fields["injuries"].status == STATUS_AVAILABLE


# ---- provenance reaches MatchContext ---------------------------------------

def test_build_match_context_carries_source_meta():
    ctx = build_match_context(
        league="EPL", home="A", away="B",
        stats={"home_form": "W-W", "away_form": "L-L", "h2h": {"wins": 1}},
        odds={"has_odds": False},
        sources=["flashscore"],
        source_meta={"form": {"source": "flashscore", "confidence": "HIGH"}},
    )
    assert ctx.source_meta == {"form": {"source": "flashscore", "confidence": "HIGH"}}


def test_build_match_context_source_meta_default_none():
    ctx = build_match_context(league="EPL", home="A", away="B")
    assert ctx.source_meta is None


# ---- Test H/I: identity + aliases -------------------------------------------

def test_team_alias_man_utd_matches_manchester_united():
    assert canonical_abbreviation("Man Utd") == "Manchester United"
    assert canonical_abbreviation("PSG") == "Paris Saint-Germain"
    assert canonical_abbreviation("Inter") is None  # ambiguous -> never guessed
    assert teams_match("Man Utd", "Manchester United") is True


def test_same_match_via_alias():
    a = {"home": "Man Utd", "away": "Chelsea", "kickoff": "2026-08-16T16:00:00Z"}
    b = {"home": "Manchester United", "away": "Chelsea", "kickoff": "2026-08-16T16:05:00Z"}
    assert same_match(a, b) is True


def test_similar_but_different_matches_not_merged():
    a = {"home": "Man City", "away": "Arsenal"}
    b = {"home": "Manchester City", "away": "Manchester United"}
    # away sides differ -> not the same fixture
    assert same_match(a, b) is False


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
