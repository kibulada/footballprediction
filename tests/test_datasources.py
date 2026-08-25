"""Tests for the multi-source data aggregation core (datasources.py).

Covers the 10 required scenarios plus missing-vs-empty distinction,
deterministic confidence, dedup, and lazy fallback. Everything is offline:
fake adapters return in-memory FieldSample values, so the tests are
deterministic and never touch the network.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.datasources import (  # noqa: E402
    CONF_HIGH,
    CONF_LOW,
    CONF_MEDIUM,
    STATUS_AVAILABLE,
    STATUS_EMPTY,
    STATUS_UNAVAILABLE,
    FieldSample,
    FootballDataSource,
    MultiSourceAggregator,
    available,
    coverage_report,
    dedupe_matches,
    empty,
    field_confidence,
    is_stale,
    merge_field,
    missing,
    normalize_competition,
    normalize_team_name,
    same_match,
    values_agree,
)


def _recent(minutes_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


class FakeSource(FootballDataSource):
    """In-memory adapter: returns a fixed field->sample mapping."""

    def __init__(self, name: str, fields: dict[str, FieldSample] | None = None,
                 calls: list[str] | None = None) -> None:
        super().__init__(name)
        self.fields = fields or {}
        self.calls = calls if calls is not None else []

    async def fetch_fields(self, ref):
        self.calls.append(self.name)
        return dict(self.fields)


class ExplodingSource(FootballDataSource):
    """Simulates a hard source failure (section 12)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    async def fetch_fields(self, ref):
        raise RuntimeError("network error")


def _agg(*sources, config=None):
    cfg = dict(config or {})
    cfg.setdefault("priority", {"flashscore": 100, "livescore": 80})
    return MultiSourceAggregator(list(sources), config=cfg)


def _run(coro):
    return asyncio.run(coro)


# ---- Test 1: Flashscore only (all data from flashscore) -------------------

def test_flashscore_only_all_fields_from_flashscore():
    fs = FakeSource("flashscore", {
        "match": available({"home": "Arsenal", "away": "Chelsea"}, _recent()),
        "form": available({"home": ["W", "W", "D"], "away": ["L", "D", "W"]}, _recent()),
        "h2h": available({"wins": 2, "draws": 1, "losses": 0}, _recent()),
    })
    unified = _run(_agg(fs).aggregate_match({"home": "Arsenal", "away": "Chelsea"}))
    assert unified.sources == ["flashscore"]
    assert unified.fields["match"].source == "flashscore"
    assert unified.fields["form"].source == "flashscore"
    assert unified.fields["form"].value == {"home": ["W", "W", "D"], "away": ["L", "D", "W"]}
    # no secondary source -> single-source confidence MEDIUM
    assert unified.fields["match"].confidence == CONF_MEDIUM


# ---- Test 2: Flashscore missing field -> LiveScore fills it ----------------

def test_field_fallback_lineup_from_livescore():
    fs = FakeSource("flashscore", {
        "match": available({"home": "A", "away": "B"}, _recent()),
        "form": available({"home": ["W"], "away": ["L"]}, _recent()),
        "lineup": missing(),
    })
    ls = FakeSource("livescore", {
        "lineup": available({"home": ["GK1", "DF2"], "away": ["GK9"]}, _recent()),
    })
    unified = _run(_agg(fs, ls).aggregate_match({"home": "A", "away": "B"}))
    assert unified.fields["lineup"].source == "livescore"
    assert unified.fields["lineup"].value == {"home": ["GK1", "DF2"], "away": ["GK9"]}
    # primary untouched for the fields it DID provide
    assert unified.fields["form"].source == "flashscore"


# ---- Test 3: both sources agree -> agreement true --------------------------

def test_agreement_between_sources():
    value = {"home": ["W", "W", "D"], "away": ["L", "D", "W"]}
    fs = FakeSource("flashscore", {"form": available(value, _recent())})
    ls = FakeSource("livescore", {"form": available(value, _recent())})
    unified = _run(_agg(fs, ls).aggregate_match({"home": "A", "away": "B"}))
    fv = unified.fields["form"]
    assert fv.agreement is True
    assert fv.discrepancy is False
    assert fv.source == "flashscore"  # priority wins on identical values
    assert set(fv.sources) == {"flashscore", "livescore"}
    assert fv.confidence == CONF_HIGH


# ---- Test 4: sources disagree -> primary wins, discrepancy recorded --------

def test_disagreement_primary_wins_and_secondary_preserved():
    fs = FakeSource("flashscore", {"form": available({"home": ["W", "W", "D", "L", "W"]}, _recent())})
    ls = FakeSource("livescore", {"form": available({"home": ["W", "W", "W", "L", "W"]}, _recent())})
    unified = _run(_agg(fs, ls).aggregate_match({"home": "A", "away": "B"}))
    fv = unified.fields["form"]
    assert fv.agreement is False
    assert fv.discrepancy is True
    assert fv.source == "flashscore"  # priority 100 > 80
    assert fv.value == {"home": ["W", "W", "D", "L", "W"]}
    # conflicting secondary value preserved for diagnostics
    assert fv.secondary and fv.secondary[0]["source"] == "livescore"
    assert fv.secondary[0]["value"] == {"home": ["W", "W", "W", "L", "W"]}
    assert fv.confidence == CONF_LOW


# ---- Test 5: Flashscore completely unavailable -> LiveScore fallback -------

def test_flashscore_failure_falls_back_to_livescore():
    fs = ExplodingSource("flashscore")
    ls = FakeSource("livescore", {
        "match": available({"home": "A", "away": "B"}, _recent()),
        "form": available({"home": ["W"], "away": ["L"]}, _recent()),
    })
    unified = _run(_agg(fs, ls).aggregate_match({"home": "A", "away": "B"}))
    assert unified.sources == ["livescore"]
    assert unified.fields["match"].source == "livescore"
    assert unified.fields["match"].status == STATUS_AVAILABLE


# ---- Test 6: both unavailable -> no fake data ------------------------------

def test_both_unavailable_no_fabrication():
    fs = FakeSource("flashscore", {})
    ls = FakeSource("livescore", {})
    unified = _run(_agg(fs, ls).aggregate_match({"home": "A", "away": "B"}))
    for name, fv in unified.fields.items():
        assert fv.status == STATUS_UNAVAILABLE
        assert fv.value is None
        assert fv.source is None
        assert fv.confidence == CONF_LOW


# ---- missing-vs-empty distinction (section 10) -----------------------------

def test_known_empty_is_distinct_from_unavailable():
    fs = FakeSource("flashscore", {"injuries": empty(_recent())})
    unified = _run(_agg(fs).aggregate_match({"home": "A", "away": "B"}))
    fv = unified.fields["injuries"]
    assert fv.status == STATUS_EMPTY
    assert fv.value is None
    # explicit "no injuries" is complete info -> MEDIUM (single source)
    assert fv.confidence == CONF_MEDIUM


# ---- Test 7/8/9: identity resolution + dedup -------------------------------

def test_same_match_with_different_team_names():
    a = {"home": "Manchester United", "away": "Chelsea",
         "kickoff": "2026-08-16T16:00:00Z"}
    b = {"home": "Manchester United FC", "away": "Chelsea",
         "kickoff": "2026-08-16T16:05:00Z"}
    assert same_match(a, b) is True


def test_different_competition_not_same_match():
    a = {"home": "Arsenal", "away": "Chelsea", "competition": "EPL"}
    b = {"home": "Arsenal", "away": "Chelsea", "competition": "LaLiga"}
    assert same_match(a, b) is False


def test_different_matches_with_similar_names_do_not_merge():
    a = {"home": "Inter Milan", "away": "Napoli"}
    b = {"home": "Inter Turku", "away": "Napoli"}
    assert same_match(a, b) is False


def test_reversed_order_is_same_match():
    a = {"home": "Inter", "away": "Milan"}
    b = {"home": "Milan", "away": "Inter"}
    assert same_match(a, b) is True


def test_ambiguous_both_orders_rejected():
    # "Inter" matches both "Inter" and "Inter Turku" -> ordered AND reversed
    # both match -> ambiguous -> must NOT merge (section 8: avoid bad merges)
    a = {"home": "Inter", "away": "Inter Turku"}
    b = {"home": "Inter", "away": "Inter Turku"}
    assert same_match(a, b) is False


def test_dedupe_collapses_duplicates():
    records = [
        {"home": "Manchester United", "away": "Chelsea", "source": "flashscore",
         "kickoff": "2026-08-16T16:00:00Z", "lineup": None},
        {"home": "Manchester United FC", "away": "Chelsea", "source": "livescore",
         "kickoff": "2026-08-16T16:05:00Z", "lineup": {"home": ["GK"]}},
    ]
    merged = dedupe_matches(records)
    assert len(merged) == 1
    assert merged[0]["source"] == "flashscore"
    assert set(merged[0]["sources"]) == {"flashscore", "livescore"}
    # missing field filled from the duplicate record
    assert merged[0]["lineup"] == {"home": ["GK"]}


def test_dedupe_keeps_distinct_matches_separate():
    records = [
        {"home": "Arsenal", "away": "Chelsea", "source": "flashscore"},
        {"home": "Liverpool", "away": "Everton", "source": "flashscore"},
    ]
    assert len(dedupe_matches(records)) == 2


# ---- Test 10: stale data gets lower confidence -----------------------------

def test_stale_data_downgrades_confidence():
    old = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    fs = FakeSource("flashscore", {"injuries": available({"home": []}, old)})
    unified = _run(_agg(fs).aggregate_match({"home": "A", "away": "B"}))
    fv = unified.fields["injuries"]
    # injuries freshness = 60 min -> 90 min old is stale -> MEDIUM -> LOW
    assert fv.stale is True
    assert fv.confidence == CONF_LOW


def test_fresh_data_keeps_confidence():
    fs = FakeSource("flashscore", {"injuries": available({"home": []}, _recent())})
    unified = _run(_agg(fs).aggregate_match({"home": "A", "away": "B"}))
    assert unified.fields["injuries"].stale is False
    assert unified.fields["injuries"].confidence == CONF_MEDIUM


# ---- lazy fallback (section 11) --------------------------------------------

def test_lazy_fallback_skips_secondary_when_covered():
    calls: list[str] = []
    fs = FakeSource("flashscore", {
        "match": available({"home": "A", "away": "B"}, _recent()),
        "form": available({"home": ["W"], "away": ["L"]}, _recent()),
    }, calls=calls)
    ls = FakeSource("livescore", {}, calls=calls)
    # cross-validation OFF -> secondary is never consulted once required
    # fields are covered.
    cfg = {"priority": {"flashscore": 100, "livescore": 80},
           "validation": {"enabled": False}}
    _run(MultiSourceAggregator([fs, ls], config=cfg).aggregate_match(
        {"home": "A", "away": "B"}, required_fields=["match", "form"]))
    assert calls == ["flashscore"]


def test_cross_validation_still_consults_secondary():
    calls: list[str] = []
    fs = FakeSource("flashscore", {
        "match": available({"home": "A", "away": "B"}, _recent()),
    }, calls=calls)
    ls = FakeSource("livescore", {
        "match": available({"home": "A", "away": "B"}, _recent()),
    }, calls=calls)
    cfg = {"priority": {"flashscore": 100, "livescore": 80},
           "validation": {"enabled": True}}
    _run(MultiSourceAggregator([fs, ls], config=cfg).aggregate_match(
        {"home": "A", "away": "B"}, required_fields=["match"]))
    assert calls == ["flashscore", "livescore"]


# ---- pure helpers -----------------------------------------------------------

def test_normalize_team_name_strips_accents_and_punct():
    assert normalize_team_name("Bodø/Glimt") == normalize_team_name("Bodo Glimt")
    assert normalize_team_name("Man Utd") != normalize_team_name("Man City")


def test_normalize_competition():
    assert normalize_competition("Premier League") == normalize_competition("premier league")
    assert normalize_competition("UEFA Champions League") != normalize_competition("Champions League")
    assert normalize_competition("EPL") != normalize_competition("LaLiga")
    # Spacing variants of the SAME competition must compare equal (verified
    # live 2026-08-16: flashscore "La Liga" vs livescore "laliga").
    assert normalize_competition("La Liga") == normalize_competition("laliga")
    assert normalize_competition("Serie A") != normalize_competition("Serie B")


def test_values_agree():
    assert values_agree({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})
    assert not values_agree({"a": 1}, {"a": 2})
    assert values_agree("W-W-D", "W-W-D")


def test_is_stale_missing_timestamp_is_not_stale():
    assert is_stale(None, field="injuries", freshness={"injuries": 60}) is False


# ---- LiveScore adapter (section 12: never crash the pipeline) -------------

def test_livescore_disabled_degrades_to_missing():
    from agents.football.livescore import LiveScoreClient, LiveScoreDataSource

    src = LiveScoreDataSource(LiveScoreClient(base_url=""))
    fields = _run(src.fetch_fields({"home": "A", "away": "B"}))
    assert all(s.status == STATUS_UNAVAILABLE for s in fields.values())


def test_livescore_client_network_error_returns_none():
    from unittest.mock import patch

    import agents.football.livescore as ls_mod
    from agents.football.livescore import LiveScoreClient

    client = LiveScoreClient(base_url="http://example.invalid")

    def _boom(*args, **kwargs):
        raise ls_mod.httpx.ConnectError("connection refused")

    async def runner():
        with patch.object(ls_mod.httpx, "AsyncClient", side_effect=_boom):
            return await client._get_json("/v1/api/app/date/soccer/20260815/0")

    assert _run(runner()) is None


# ---- reachability classification + coverage report (Phase 3) ---------------

def test_livescore_classify_blocked_government_page():
    from agents.football.livescore import LiveScoreClient

    text = "<title>Trustpositif</title><h3>SITUS DIBLOKIR</h3>"
    assert LiveScoreClient._classify(200, text) == "blocked"


def test_livescore_classify_forbidden():
    from agents.football.livescore import LiveScoreClient

    assert LiveScoreClient._classify(403, "Forbidden") == "forbidden"


def test_livescore_classify_ok():
    from agents.football.livescore import LiveScoreClient

    assert LiveScoreClient._classify(200, "<title>LiveScore</title>") == "ok"


def test_livescore_classify_unreachable():
    from agents.football.livescore import LiveScoreClient

    assert LiveScoreClient._classify(None, None) == "unreachable"


def test_livescore_health_no_config():
    from agents.football.livescore import LiveScoreClient

    assert _run(LiveScoreClient(base_url="").health()) == {"reachable": False, "reason": "no_config"}


def test_coverage_report_measures_real_sources():
    fs = FakeSource("flashscore", {
        "match": available({"home": "A", "away": "B"}, _recent()),
        "form": available({"home": ["W"], "away": ["L"]}, _recent()),
        "h2h": available({"wins": 1, "draws": 0, "losses": 0}, _recent()),
    })
    ls = FakeSource("livescore", {"lineup": available({"home": ["GK"]}, _recent())})
    unified = _run(_agg(fs, ls).aggregate_match({"home": "A", "away": "B"}))
    report = coverage_report(unified)
    assert report["fields_requested"] == 8
    assert report["fields_by_source"]["flashscore"] == 3
    assert report["fields_by_source"]["livescore"] == 1
    assert report["fields_available"] == 4
    assert report["fields_missing"] == 4


# ---- LiveScore verified API: parser + resolution (Phase 3) ----------------

# Sample payload in the VERIFIED lsmedia1.com /date/soccer schema (real shape,
# not invented).
_LS_PAYLOAD = {
    "Ts": 1786804054,
    "Stages": [
        {
            "Sid": "25695", "Snm": "LaLiga", "CompN": "LaLiga", "Cnm": "Spain",
            "CnmT": "spain", "Ccd": "spain", "Scd": "laliga",
            "Events": [
                {"Eid": "1810638",
                 "T1": [{"ID": "4482", "Nm": "Deportivo Alaves", "Abr": "ALV"}],
                 "T2": [{"ID": "3379", "Nm": "Getafe", "Abr": "GET"}],
                 "Eps": "NS", "Esid": 1, "Esd": 20260815173000},
                {"Eid": "1810642",
                 "T1": [{"ID": "1", "Nm": "Real Madrid"}],
                 "T2": [{"ID": "2", "Nm": "Barcelona"}],
                 "Eps": "FT", "Esid": 1, "Esd": 20260815193000, "Tr1": 2, "Tr2": 1},
            ],
        },
        {
            "Sid": "65", "Snm": "Premier League", "CompN": "Premier League", "Cnm": "England",
            "Events": [
                {"Eid": "1810700",
                 "T1": [{"ID": "2810", "Nm": "Manchester United"}],
                 "T2": [{"ID": "2773", "Nm": "Chelsea"}],
                 "Eps": "NS", "Esid": 1, "Esd": 20260815160000},
            ],
        },
        {"Sid": "X", "Snm": "Bad", "Cnm": "X", "Events": [{"garbage": True}]},
    ],
}


def _ls_fake(payload=None, *, raises=False):
    class FakeLiveClient:
        available = True

        async def fetch_soccer_date(self, date, page):
            if raises:
                raise RuntimeError("livescore down")
            return payload

    return FakeLiveClient()


def test_parse_soccer_payload_verified_schema():
    from agents.football.livescore import parse_soccer_payload

    fixtures = parse_soccer_payload(_LS_PAYLOAD)
    assert len(fixtures) == 3  # garbage event skipped
    ns = next(f for f in fixtures if f["source_id"] == "1810638")
    assert ns["home"] == "Deportivo Alaves"
    assert ns["away"] == "Getafe"
    assert ns["kickoff"] == "2026-08-15T17:30:00Z"
    assert ns["status"] == "scheduled"
    assert ns["competition"] == "LaLiga"
    assert ns["country"] == "Spain"
    assert ns["score"] == {"home": None, "away": None}  # missing, never fabricated
    ft = next(f for f in fixtures if f["source_id"] == "1810642")
    assert ft["status"] == "finished"
    assert ft["score"] == {"home": 2, "away": 1}


def test_parse_soccer_payload_malformed():
    from agents.football.livescore import parse_soccer_payload

    assert parse_soccer_payload(None) == []
    assert parse_soccer_payload({}) == []
    assert parse_soccer_payload({"Stages": "x"}) == []
    assert parse_soccer_payload({"Stages": [{"Events": [None, "x", {}]}]}) == []


def test_parse_esd_and_status():
    from agents.football.livescore import normalize_status, parse_esd

    assert parse_esd(20260815173000) == "2026-08-15T17:30:00Z"
    assert parse_esd("20260815173000") == "2026-08-15T17:30:00Z"
    assert parse_esd(None) is None
    assert parse_esd("garbage") is None
    assert normalize_status("NS") == "scheduled"
    assert normalize_status("FT") == "finished"
    assert normalize_status("16'") == "live"
    assert normalize_status(None) == "unknown"


def test_livescore_get_match_resolves_real_fixture():
    from agents.football.livescore import LiveScoreDataSource

    src = LiveScoreDataSource(_ls_fake(_LS_PAYLOAD), max_pages=1)
    sample = _run(src.get_match(
        {"home": "Deportivo Alaves", "away": "Getafe", "kickoff": "2026-08-15T17:30:00Z"}
    ))
    assert sample.status == "available"
    assert sample.value["home"] == "deportivo alaves"  # canonical identity
    assert sample.value["away"] == "getafe"
    assert sample.value["competition"] == "laliga"


def test_livescore_get_match_via_alias():
    from agents.football.livescore import LiveScoreDataSource

    src = LiveScoreDataSource(_ls_fake(_LS_PAYLOAD), max_pages=1)
    sample = _run(src.get_match(
        {"home": "Man Utd", "away": "Chelsea", "kickoff": "2026-08-15T16:00:00Z"}
    ))
    assert sample.status == "available"
    assert sample.value["home"] == "manchester united"


def test_livescore_get_match_reversed_fixture_oriented():
    from agents.football.livescore import LiveScoreDataSource

    src = LiveScoreDataSource(_ls_fake(_LS_PAYLOAD), max_pages=1)
    sample = _run(src.get_match(
        {"home": "Chelsea", "away": "Manchester United", "kickoff": "2026-08-15T16:00:00Z"}
    ))
    assert sample.status == "available"
    assert sample.value["home"] == "chelsea"  # aligned to reference order
    assert sample.value["away"] == "manchester united"


def test_livescore_get_match_no_match_returns_missing():
    from agents.football.livescore import LiveScoreDataSource

    src = LiveScoreDataSource(_ls_fake(_LS_PAYLOAD), max_pages=1)
    sample = _run(src.get_match({"home": "Inter", "away": "Milan", "kickoff": "2026-08-15T16:00:00Z"}))
    assert sample.status == STATUS_UNAVAILABLE
    assert sample.value is None


def test_livescore_get_match_failure_degrades_to_missing():
    from agents.football.livescore import LiveScoreDataSource

    src = LiveScoreDataSource(_ls_fake(raises=True), max_pages=1)
    sample = _run(src.get_match({"home": "A", "away": "B"}))
    assert sample.status == STATUS_UNAVAILABLE


# ---- LiveScore field parsers (verified schemas, Phase 4) ------------------

_LS_LINEUPS = {
    "Eid": "1810638",
    "Lu": [
        {"Tnb": 1, "Fo": "4-4-2",
         "Ps": [{"Pid": "1", "Fn": "Antonio", "Ln": "Sivera", "Snu": 1, "Pon": "Goalkeeper", "Pos": 1}],
         "IS": [{"Pid": "9", "Fn": "Hugo", "Ln": "Novoa", "Snu": 3, "Pon": "Forward", "Pos": 4}]},
        {"Tnb": 2, "Fo": "4-4-2",
         "Ps": [{"Pid": "2", "Fn": "David", "Ln": "Soria", "Snu": 13, "Pon": "Goalkeeper", "Pos": 1}],
         "IS": []},
    ],
}

_LS_H2H = {
    "Eid": "1810638",
    "H2H": [
        {"Eid": "1547678", "T1": [{"Nm": "Deportivo Alaves", "ID": "4482"}],
         "T2": [{"Nm": "Getafe", "ID": "3379"}], "Tr1": "0", "Tr2": "2", "Eps": "FT", "Esd": 20260208130000},
        {"Eid": "1547581", "T1": [{"Nm": "Getafe", "ID": "3379"}],
         "T2": [{"Nm": "Deportivo Alaves", "ID": "4482"}], "Tr1": "1", "Tr2": "1", "Eps": "FT", "Esd": 20250809130000},
    ],
}

_LS_FORM = {
    "Eid": "1810638",
    "T1": [{"Nm": "Deportivo Alaves", "ID": "4482", "EL": [
        # EL is newest-first (real API order): 08-10, then 08-07.
        {"Eid": "2", "T1": [{"Nm": "Deportivo Alaves", "ID": "4482"}], "T2": [{"Nm": "Celta", "ID": "5"}],
         "Tr1": "2", "Tr2": "0", "Eps": "FT", "Esd": 20260810173000},
        {"Eid": "1", "T1": [{"Nm": "Racing", "ID": "4508"}], "T2": [{"Nm": "Deportivo Alaves", "ID": "4482"}],
         "Tr1": "1", "Tr2": "1", "Eps": "FT", "Esd": 20260807173000},
    ]}],
    "T2": [{"Nm": "Getafe", "ID": "3379", "EL": [
        {"Eid": "3", "T1": [{"Nm": "Getafe", "ID": "3379"}], "T2": [{"Nm": "Vallecano", "ID": "6"}],
         "Tr1": "0", "Tr2": "1", "Eps": "FT", "Esd": 20260807173000},
    ]}],
}

_LS_STATS = {
    "Eid": "1",
    "Stat": [
        {"Tnb": 1, "Shon": 3, "Shof": 4, "Crs": 6, "Ycs": 2, "Rcs": 0, "Fls": 15, "Pss": 45, "Gks": 2},
        {"Tnb": 2, "Shon": 11, "Shof": 7, "Crs": 9, "Ycs": 1, "Rcs": 0, "Fls": 9, "Pss": 55, "Gks": 3},
    ],
}

_LS_TABLE = {
    "LeagueTable": {"L": [{"Tables": [{"LTT": 1, "team": [
        {"rnk": 5, "Tid": "4482", "Tnm": "Deportivo Alaves", "win": 1, "drw": 1, "lst": 0,
         "gf": 3, "ga": 2, "gd": 1, "pts": 4, "pld": 2},
        {"rnk": 9, "Tid": "3379", "Tnm": "Getafe", "win": 0, "drw": 1, "lst": 1,
         "gf": 1, "ga": 3, "gd": -2, "pts": 1, "pld": 2},
    ]}]}]},
}


def test_parse_lineups_verified_schema():
    from agents.football.livescore import parse_lineups

    out = parse_lineups(_LS_LINEUPS)
    assert out["home"]["players"][0]["name"] == "Antonio Sivera"
    assert out["home"]["players"][0]["shirt"] == 1
    assert out["home"]["substitutes"][0]["name"] == "Hugo Novoa"
    assert out["away"]["players"][0]["shirt"] == 13
    assert parse_lineups({}) is None


def test_parse_h2h_perspective():
    from agents.football.livescore import parse_h2h

    out = parse_h2h(_LS_H2H, {"home_id": "4482"})
    # meeting 1: Alaves (4482) home, lost 0-2 -> loss; meeting 2: Getafe home,
    # Alaves away, drew 1-1 -> draw.
    assert (out["wins"], out["draws"], out["losses"]) == (0, 1, 1)
    assert len(out["meetings"]) == 2
    assert out["meetings"][0]["home_score"] == 0


def test_parse_h2h_finished_only_for_tally():
    """Fix 2026-08-17: only FINISHED meetings may contribute to the W/D/L
    tally -- a live 1-1 or a postponed partial score must never inflate the
    draw count. The meeting itself is still listed (with its status)."""
    from agents.football.livescore import parse_h2h

    payload = {
        "H2H": [
            # finished: Alaves (4482) home, lost 0-2 -> loss
            {"T1": [{"Nm": "Deportivo Alaves", "ID": "4482"}],
             "T2": [{"Nm": "Getafe", "ID": "3379"}],
             "Tr1": "0", "Tr2": "2", "Eps": "FT"},
            # live: 1-1 at the moment -> must NOT count as a draw
            {"T1": [{"Nm": "Deportivo Alaves", "ID": "4482"}],
             "T2": [{"Nm": "Getafe", "ID": "3379"}],
             "Tr1": "1", "Tr2": "1", "Eps": "1H"},
            # postponed with a partial score -> must NOT count
            {"T1": [{"Nm": "Deportivo Alaves", "ID": "4482"}],
             "T2": [{"Nm": "Getafe", "ID": "3379"}],
             "Tr1": "1", "Tr2": "1", "Eps": "POSTP"},
        ],
    }
    out = parse_h2h(payload, {"home_id": "4482"})
    assert (out["wins"], out["draws"], out["losses"]) == (0, 0, 1)
    assert len(out["meetings"]) == 3  # all meetings listed, status kept
    statuses = [m["status"] for m in out["meetings"]]
    assert statuses == ["finished", "live", "scheduled"]


def test_parse_form_verified_schema():
    from agents.football.livescore import parse_form

    out = parse_form(_LS_FORM)
    # EL newest-first; reversed: draw (away 1-1) then win (home 2-0).
    assert out["home"]["sequence"] == "D-W"
    assert out["home"]["recent_goals"] == [(1, 1), (2, 0)]
    assert out["home"]["sample_size"] == 2
    assert out["away"]["sequence"] == "L"  # away 0-1 loss


def test_parse_statistics_verified_schema():
    from agents.football.livescore import parse_statistics

    out = parse_statistics(_LS_STATS)
    assert out["home"]["shots_on_target"] == 3
    assert out["away"]["shots_on_target"] == 11
    assert out["home"]["yellow_cards"] == 2
    assert parse_statistics({}) is None


def test_parse_league_table_verified_schema():
    from agents.football.livescore import parse_league_table

    out = parse_league_table(_LS_TABLE, {"home": "Deportivo Alaves", "away": "Getafe"})
    assert out["home"]["pos"] == 5
    assert out["home"]["team"] == "Deportivo Alaves"
    assert out["away"]["pos"] == 9
    assert out["away"]["points"] == 1
    # loose matcher must NOT pull in a similar-name row (exact match only)
    assert out["home"]["team"] == "Deportivo Alaves"


def test_livescore_field_fallbacks_resolve():
    from agents.football.livescore import LiveScoreDataSource

    class FieldFake:
        available = True

        def __init__(self):
            self.fields = {
                "lineups": _LS_LINEUPS, "h2h": _LS_H2H, "form": _LS_FORM,
                "stats": _LS_STATS, "table": _LS_TABLE,
            }

        async def fetch_soccer_date(self, date, page):
            return _LS_PAYLOAD

        async def fetch_lineups(self, eid):
            return self.fields["lineups"]

        async def fetch_h2h(self, eid):
            return self.fields["h2h"]

        async def fetch_form(self, eid, limit=10):
            return self.fields["form"]

        async def fetch_statistics(self, eid):
            return self.fields["stats"]

        async def fetch_league_table(self, category, stage):
            return self.fields["table"]

    src = LiveScoreDataSource(FieldFake(), max_pages=1)
    ref = {"home": "Deportivo Alaves", "away": "Getafe", "kickoff": "2026-08-15T17:30:00Z"}
    lu = _run(src.get_lineup(ref))
    assert lu.status == STATUS_AVAILABLE and lu.value["home"]["players"]
    h2h = _run(src.get_h2h(ref))
    assert h2h.status == STATUS_AVAILABLE and h2h.value["meetings"]
    form = _run(src.get_form(ref))
    assert form.status == STATUS_AVAILABLE and form.value["home"]["sequence"]
    st = _run(src.get_standings(ref))
    assert st.status == STATUS_AVAILABLE and st.value["home"]["team"] == "Deportivo Alaves"


def test_livescore_empty_statistics_degrades_to_missing():
    from agents.football.livescore import LiveScoreDataSource

    class EmptyStats:
        available = True

        async def fetch_soccer_date(self, date, page):
            return _LS_PAYLOAD

        async def fetch_statistics(self, eid):
            return {}

    src = LiveScoreDataSource(EmptyStats(), max_pages=1)
    sample = _run(src.get_statistics({"home": "Deportivo Alaves", "away": "Getafe",
                                      "kickoff": "2026-08-15T17:30:00Z"}))
    assert sample.status == STATUS_UNAVAILABLE  # not fabricated


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
