"""WIB/kickoff boundary regression tests (2026-08-17).

The livescore date feed is keyed by UTC date, but every "match day" the bot
displays, searches and settles is a WIB (UTC+7) calendar day. One WIB day
spans TWO UTC dates (WIB 00:00 == previous UTC day 17:00), so any off-by-one
hour in the boundary flips a kickoff between today and tomorrow -- which
corrupts the T-24h/T-6h windows, the today/tomorrow search order, the
movement window and the settle date.

These tests pin the exact boundary so a timezone regression is caught
immediately:

  - utc_range_for_wib_date spans two UTC dates (incl. DST transitions),
  - _utc_feed_dates_today_then_tomorrow dedupes + orders today first,
  - wib_date_from_iso flips at 17:00 UTC (== 00:00 WIB next day),
  - parse_esd round-trips the feed's numeric Esd,
  - _search_livescore classifies 16:59Z as today and 17:00Z as tomorrow,
  - _search_livescore_any (dynamic league) honours the same boundary.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agents.football.source_match as source_match  # noqa: E402
from agents.football.livescore import parse_esd  # noqa: E402
from agents.football.timeutil import (  # noqa: E402
    utc_range_for_wib_date,
    wib_date_from_iso,
)


def _run(coro):
    return asyncio.run(coro)


def _cfg():
    return {
        "analyse": {"budget_seconds": 300.0},
        "data_sources": {"livescore": {"enabled": True, "max_pages": 1}},
        "cache_ttl_seconds": {"odds": 900},
        "outlier_threshold_pct": 5,
        "prediction_log": {"enabled": False},
    }


def _cache():
    return type("C", (), {"get": lambda *a, **k: None, "set": lambda *a, **k: None})()


def _ls_payload(stage_comp: str, home: str, away: str, kickoff: str, eid: int = 12345):
    """One-stage LiveScore /date/soccer payload with a single event."""
    esd = kickoff.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    return {
        "Stages": [
            {
                "CompN": stage_comp,
                "Cnm": "Spain",
                "Ccd": "spain",
                "Scd": "laliga",
                "Events": [
                    {
                        "Eid": eid,
                        "T1": [{"Nm": home, "ID": 1001}],
                        "T2": [{"Nm": away, "ID": 1002}],
                        "Esd": esd,
                        "Eps": "NS",
                        "Tr1": None,
                        "Tr2": None,
                    }
                ],
            }
        ]
    }


class _FakeLiveClient:
    """fetch_soccer_date returns the fixture only for one UTC feed date."""

    available = True  # _search_livescore_any guards on client.available

    def __init__(self, feeds: dict[str, dict]):
        self.feeds = feeds

    async def fetch_soccer_date(self, date8: str, page: int):
        return self.feeds.get(date8) if page == 0 else None


def _patch_live_client(monkeypatch, feeds: dict[str, dict]):
    monkeypatch.setattr(source_match, "LiveScoreClient", lambda base_url=None: _FakeLiveClient(feeds))


class _Stats:
    """Minimal stats stand-in carrying a fake livescore client + cache."""

    def __init__(self, feeds: dict[str, dict]):
        self.livescore = _FakeLiveClient(feeds)
        self.cache = _cache()


# --------------------------------------------------------------------------
# utc_range_for_wib_date -- one WIB day spans two UTC dates
# --------------------------------------------------------------------------

def test_utc_range_spans_two_utc_dates():
    # WIB 2026-08-15 00:00 == UTC 2026-08-14 17:00, so the WIB day covers
    # parts of BOTH UTC dates.
    assert utc_range_for_wib_date("2026-08-15") == ("2026-08-14", "2026-08-15")


def test_utc_range_around_dst_transitions():
    # DST transitions (last Sunday of March / October) must not move the
    # range -- WIB has no DST, the UTC+7 offset is constant.
    assert utc_range_for_wib_date("2026-03-29") == ("2026-03-28", "2026-03-29")
    assert utc_range_for_wib_date("2026-10-25") == ("2026-10-24", "2026-10-25")


# --------------------------------------------------------------------------
# _utc_feed_dates_today_then_tomorrow -- dedupe + today-first ordering
# --------------------------------------------------------------------------

def test_feed_dates_today_then_tomorrow_deduped(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # today WIB needs UTC 2026-08-14 + 2026-08-15; tomorrow needs
    # 2026-08-15 + 2026-08-16 -> deduped to three, today's dates first.
    assert source_match._utc_feed_dates_today_then_tomorrow() == [
        "20260814", "20260815", "20260816",
    ]


# --------------------------------------------------------------------------
# wib_date_from_iso -- the 17:00 UTC hard boundary
# --------------------------------------------------------------------------

def test_wib_date_flips_at_1700_utc():
    # 16:59 UTC == 23:59 WIB same day; 17:00 UTC == 00:00 WIB NEXT day.
    assert wib_date_from_iso("2026-08-15T16:59:00Z") == "2026-08-15"
    assert wib_date_from_iso("2026-08-15T17:00:00Z") == "2026-08-16"
    assert wib_date_from_iso("2026-08-15T16:00:00Z") == "2026-08-15"  # 23:00 WIB


# --------------------------------------------------------------------------
# parse_esd -- numeric feed kickoff round-trips
# --------------------------------------------------------------------------

def test_parse_esd_roundtrip():
    assert parse_esd(20260815160000) == "2026-08-15T16:00:00Z"
    assert parse_esd("20260815160000") == "2026-08-15T16:00:00Z"
    assert parse_esd(None) is None
    assert parse_esd("garbage") is None


# --------------------------------------------------------------------------
# _search_livescore -- 16:59Z is TODAY, 17:00Z is TOMORROW
# --------------------------------------------------------------------------

def test_search_livescore_kickoff_1659z_is_today(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 2026-08-15 16:59 UTC == 23:59 WIB 08-15 -> TODAY.
    feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-15T16:59:00Z")
    _patch_live_client(monkeypatch, {"20260815": feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["date"] == "2026-08-15"


def test_search_livescore_kickoff_1700z_is_tomorrow(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 2026-08-15 17:00 UTC == 00:00 WIB 08-16 -> TOMORROW, not today.
    feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-15T17:00:00Z")
    _patch_live_client(monkeypatch, {"20260815": feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["date"] == "2026-08-16"


def test_search_livescore_early_utc_still_today(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 2026-08-14 21:00 UTC == 2026-08-15 04:00 WIB -> today (the previous
    # UTC date's late evening is this WIB day's small hours).
    feed = _ls_payload("LaLiga", "Barcelona", "Getafe", "2026-08-14T21:00:00Z")
    _patch_live_client(monkeypatch, {"20260814": feed})
    found = _run(source_match._search_livescore(_cfg(), _cache(), "LaLiga", "La Liga",
                                                "barcelona", "getafe"))
    assert found is not None
    assert found["date"] == "2026-08-15"


# --------------------------------------------------------------------------
# _search_livescore_any (dynamic league) -- same WIB boundary
# --------------------------------------------------------------------------

def test_search_livescore_any_kickoff_boundary(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # Coppa Italia kickoff 16:59Z -> today WIB -> detected as today's fixture.
    feed = _ls_payload("Coppa Italia", "Pisa", "Empoli", "2026-08-15T16:59:00Z")
    stats = _Stats({"20260815": feed})
    found = _run(source_match._search_livescore_any(stats, "Pisa", "Empoli"))
    assert found is not None
    assert found["competition"] == "Coppa Italia"
    assert wib_date_from_iso(found.get("kickoff")) == "2026-08-15"


def test_search_livescore_any_kickoff_1700z_tomorrow(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 17:00Z == 00:00 WIB next day -> STILL inside the today/tomorrow scan
    # window (tomorrow), so detection succeeds and labels it tomorrow.
    feed = _ls_payload("Coppa Italia", "Pisa", "Empoli", "2026-08-15T17:00:00Z")
    stats = _Stats({"20260815": feed})
    found = _run(source_match._search_livescore_any(stats, "Pisa", "Empoli"))
    assert found is not None
    assert wib_date_from_iso(found.get("kickoff")) == "2026-08-16"


def test_search_livescore_any_outside_window_rejected(monkeypatch):
    monkeypatch.setattr(source_match, "wib_today_iso", lambda: "2026-08-15")
    # 2026-08-17 16:00 UTC == 2026-08-17 23:00 WIB -> the day AFTER tomorrow
    # -> outside the scan window -> undetected.
    feed = _ls_payload("Coppa Italia", "Pisa", "Empoli", "2026-08-17T16:00:00Z")
    stats = _Stats({"20260817": feed})
    found = _run(source_match._search_livescore_any(stats, "Pisa", "Empoli"))
    assert found is None
