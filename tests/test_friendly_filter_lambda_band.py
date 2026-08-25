"""Tests for the 2026-08-22 lambda-input + lambda-band fixes (#1/#2).

Audit context: Fortuna Sittard v AZ (Eredivisie) carried lambda_total 3.96
against a market at ~3.2 -- pre-season friendlies polluted the form inputs
(flashscore/livescore form producers had no P3-2 guard), and the football-wide
G4 band [1.6, 3.6] ignored league scoring baselines.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.nowgoal import is_friendly_competition  # noqa: E402
from agents.football.livescore import parse_form  # noqa: E402
from agents.football.pick_gates import resolve_lambda_total_band  # noqa: E402


# ---- #1a: substring-tolerant friendly detector ------------------------------

def test_is_friendly_catches_live_spellings():
    for name in (
        "Club Friendlies", "Club Friendlies 2026", "Featured Club Friendlies",
        "Club Friendly Games", "Pre-Season", "Preseason Friendly",
        "International Friendlies", "club friendlies",
    ):
        assert is_friendly_competition(name), name


def test_is_friendly_keeps_real_competitions():
    for name in ("Premier League", "Eredivisie", "LaLiga", "Serie A",
                 "UEFA Champions League", "EFL Championship", None, ""):
        assert not is_friendly_competition(name), name


# ---- #1b: livescore form excludes friendlies --------------------------------

def _ev(eid, t1, t2, tr1, tr2, comp=None):
    ev = {
        "Eid": str(eid),
        "T1": [{"Nm": t1, "ID": t1}], "T2": [{"Nm": t2, "ID": t2}],
        "Tr1": str(tr1), "Tr2": str(tr2), "Eps": "FT",
    }
    if comp is not None:
        ev["Stg"] = {"Snm": comp}
    return ev


def test_livescore_parse_form_excludes_friendlies():
    payload = {
        "T1": [{"Nm": "Everton", "ID": "Everton", "EL": [
            # newest first; the two most recent are FRIENDLIES and must be
            # dropped before they reach sequence/gf/ga/recent_goals.
            _ev(1, "Everton", "Lille", 1, 1, "Club Friendlies 2026"),
            _ev(2, "Sunderland", "Everton", 0, 3, "Featured Club Friendlies"),
            _ev(3, "Everton", "Brighton", 2, 1, "Premier League"),
            _ev(4, "Leeds", "Everton", 1, 1, "Premier League"),
        ]}],
        "T2": [{"Nm": "Brighton", "ID": "Brighton", "EL": [
            _ev(5, "Brighton", "Napoli", 0, 0, "Club Friendlies 2026"),
            _ev(6, "Brighton", "Fulham", 1, 2, "Premier League"),
        ]}],
    }
    out = parse_form(payload)
    # only competitive games count: W-D for Everton (oldest->newest), L for Brighton
    assert out["home"]["sequence"] == "D-W"
    assert out["home"]["sample_size"] == 2
    assert out["home"]["recent_goals"] == [(1, 1), (2, 1)]
    assert out["away"]["sequence"] == "L"


def test_livescore_parse_form_without_stg_fails_open():
    # legacy/other feeds that omit Stg entirely must NOT lose their rows
    payload = {
        "T1": [{"Nm": "A", "ID": "A", "EL": [
            _ev(1, "A", "B", 2, 0),
        ]}],
        "T2": [{"Nm": "B", "ID": "B", "EL": [
            _ev(2, "B", "C", 1, 1),
        ]}],
    }
    out = parse_form(payload)
    assert out["home"]["sequence"] == "W"


# ---- #1c: multi_source livescore-form fallback excludes friendlies ----------

def _ls_feed_payload(matches):
    stages = {}
    for fx in matches:
        comp = fx.pop("_comp")
        stages.setdefault(comp, []).append(fx)
    return {"Stages": [
        {"CompN": comp, "Events": evs} for comp, evs in stages.items()
    ]}


def test_multi_source_livescore_form_skips_friendlies():
    from agents.football.multi_source import MultiSourceStatsFetcher

    feed = _ls_feed_payload([
        # finished LEAGUE match (counts)
        {"_comp": "Premier League", "Eid": "1",
         "T1": [{"Nm": "Everton"}], "T2": [{"Nm": "Brighton"}],
         "Eps": "FT", "Tr1": 2, "Tr2": 1},
        # finished FRIENDLY with an inflated scoreline (must be dropped)
        {"_comp": "Club Friendlies 2026", "Eid": "2",
         "T1": [{"Nm": "Everton"}], "T2": [{"Nm": "Stoke City"}],
         "Eps": "FT", "Tr1": 6, "Tr2": 0},
    ])

    class FakeLS:
        available = True

        async def fetch_soccer_date(self, date, page):
            return feed if page == 0 else None

    fake_self = types.SimpleNamespace(livescore=FakeLS(), cache=None)
    out = asyncio.run(MultiSourceStatsFetcher._livescore_form(
        fake_self, "Everton", limit=5, lookback_days=1,
    ))
    assert out is not None
    assert out["sample_size"] == 1
    assert out["recent_goals"] == [(2, 1)]
    assert out["sequence"] == "W"


# ---- #1d: flashscore fetch_team_form filters via section headers ------------

class _FakeBrowser:
    available = True

    def __init__(self, rows):
        self._rows = rows

    def scrape_team_results(self, slug, team_id, limit):
        return self._rows[:limit]


def _flash_client(rows):
    from agents.football.flashscore import FlashscoreClient

    client = object.__new__(FlashscoreClient)
    client._browser = _FakeBrowser(rows)
    client._browser_lock = asyncio.Lock()
    client.available = True
    client._throttle_sleep = lambda: None
    return client


def test_flashscore_fetch_team_form_drops_friendlies_and_windows():
    rows = [
        # newest first; header-tracked competition per row
        {"date": "a", "home": "Fortuna Sittard", "away": "AZ Alkmaar",
         "hg": "6", "ag": "2", "result": "W", "competition": "CLUB FRIENDLIES"},
        {"date": "b", "home": "Fortuna Sittard", "away": "Excelsior",
         "hg": "3", "ag": "2", "result": "W", "competition": "EREDIVISIE"},
        {"date": "c", "home": "NEC", "away": "Fortuna Sittard",
         "hg": "3", "ag": "1", "result": "L", "competition": "EREDIVISIE"},
        {"date": "d", "home": "Fortuna Sittard", "away": "Twente",
         "hg": "2", "ag": "2", "result": "D", "competition": "EREDIVISIE"},
    ]
    client = _flash_client(rows)
    form = asyncio.run(client.fetch_team_form("fortuna-sittard", "1234", limit=2))
    # the 6-2 friendly is dropped BEFORE aggregation; window keeps the last
    # 2 COMPETITIVE games (newest first: W 3-2, then L 1-3).
    assert form["sequence"] == "W-L"
    assert form["sample_size"] == 2
    assert list(form["recent_goals"]) == [(1, 3), (3, 2)]


def test_flashscore_fetch_team_form_rows_without_header_fail_open():
    rows = [
        {"date": "a", "home": "Fortuna Sittard", "away": "Twente",
         "hg": "1", "ag": "1", "result": "D", "competition": None},
    ]
    client = _flash_client(rows)
    form = asyncio.run(client.fetch_team_form("fortuna-sittard", "1234", limit=5))
    assert form is not None and form["sample_size"] == 1


# ---- #2: league-aware G4 band ------------------------------------------------

def test_band_defaults_and_global_override():
    assert resolve_lambda_total_band(None, None) == (1.6, 3.6)
    assert resolve_lambda_total_band({}, "EPL") == (1.6, 3.6)
    cfg = {"lambda_total_min": 1.8, "lambda_total_max": 3.9}
    assert resolve_lambda_total_band(cfg, "EPL") == (1.8, 3.9)


def test_band_league_override_case_insensitive_and_partial():
    cfg = {"lambda_total_band_by_league": {"eredivisie": {"max": 4.0}}}
    assert resolve_lambda_total_band(cfg, "Eredivisie") == (1.6, 4.0)
    assert resolve_lambda_total_band(cfg, " eredivisie ") == (1.6, 4.0)
    # unknown league keeps the global band
    assert resolve_lambda_total_band(cfg, "EPL") == (1.6, 3.6)
    # bare-number override means max; min-only also supported
    cfg2 = {"lambda_total_band_by_league": {"bundesliga": 3.8}}
    assert resolve_lambda_total_band(cfg2, "Bundesliga") == (1.6, 3.8)
    cfg3 = {"lambda_total_band_by_league": {"mls": {"min": 2.0}}}
    assert resolve_lambda_total_band(cfg3, "MLS") == (2.0, 3.6)


def test_signal_engine_g4_uses_league_band():
    """lambda_total 3.96: vetoed under the global band, allowed for the
    Eredivisie override -- same card, only the league differs."""
    from agents.football.signal_engine import run_signal_engine

    model = {
        "1x2": {"home": 0.55, "draw": 0.20, "away": 0.25},
        "over_1.5": 0.90, "over_2.5": 0.755, "over_3.5": 0.52,
        "btts_yes": 0.66,
        "lambda_home": 2.20, "lambda_away": 1.76,
    }
    totals = {
        "Over 2.5": {"odds": 1.44, "point": 2.5},
        "Under 2.5": {"odds": 2.50, "point": 2.5},
        "BTTS Yes": {"odds": 1.70},
        "BTTS No": {"odds": 2.10},
    }

    def _band_vetoed(res):
        return any("di luar band" in r
                   for r in res.get("disagreement_gate", []))

    base_cfg = {"pick_gates": {}}
    res_global = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=[],
        movement_snapshot=None, context=None, completeness=1.0,
        cfg=base_cfg, league_name="Eredivisie",
    )
    assert _band_vetoed(res_global)

    ere_cfg = {"pick_gates": {"lambda_total_band_by_league":
                              {"eredivisie": {"max": 4.0}}}}
    res_nl = run_signal_engine(
        model_probs=model, stats={}, market_totals=totals, ah_rows=[],
        movement_snapshot=None, context=None, completeness=1.0,
        cfg=ere_cfg, league_name="Eredivisie",
    )
    assert not _band_vetoed(res_nl)


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
