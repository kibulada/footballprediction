"""Tests for the NowGoal odds-movement history extension.

The new ``NowGoalClient.fetch_odds_history`` exposes the opening -> latest
movement NowGoal actually serves (per bookmaker per market, PRICE and LINE
at each leg), and ``fetch_odds`` now carries the opening LINE (``opening_point``)
so the Signal Engine can separate line movement from price movement.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.nowgoal import NowGoalClient  # noqa: E402
from agents.football.signal_engine import extract_asian_handicap  # noqa: E402
from agents.football.analyse import extract_market_totals  # noqa: E402

# Canned mixodds shaped exactly like the verified live feed (2026-08-14):
# euro prices are decimal; ou/ah prices are Hong-Kong format (0.88 == 1.88).
_MIXODDS = {
    "ErrCode": 0,
    "Data": {
        "mixodds": [
            {
                "cid": 8,
                "cn": "Bet365",
                "euro": {
                    "f": {"u": "5.0", "g": "3.5", "d": "1.65"},
                    "l": {"u": "5.5", "g": "3.6", "d": "1.57"},
                    "r": {"u": "5.5", "g": "3.6", "d": "1.57"},
                    "hr": False,
                },
                "ou": {
                    "f": {"u": "0.9", "g": "2.25", "d": "0.9"},
                    "l": {"u": "0.88", "g": "2.5", "d": "0.93"},
                    "r": {"u": "0.88", "g": "2.5", "d": "0.93"},
                    "hr": False,
                },
                "ah": {
                    "f": {"u": "0.95", "g": "-0.75", "d": "0.85"},
                    "l": {"u": "0.78", "g": "-1", "d": "1.03"},
                    "r": {"u": "0.78", "g": "-1", "d": "1.03"},
                    "hr": False,
                },
            }
        ]
    },
}


class _FakeClient(NowGoalClient):
    def __init__(self, payload):
        super().__init__()
        self._payload = payload

    async def _get(self, path, params):
        return self._payload


def _fixture():
    return {"match_id": "3061003", "home": "Malaysia", "away": "Vietnam",
            "kickoff": "2026-08-16T13:00:00Z"}


def test_fetch_odds_history_opening_latest_structure():
    async def run():
        c = _FakeClient(_MIXODDS)
        h = await c.fetch_odds_history(_fixture())
        assert h is not None
        assert h["match_id"] == "3061003"
        assert h["timestamp_available"] is False
        assert h["history_resolution"] == "opening_latest"
        assert h["source"] == "nowgoal"
        rows = h["markets"]
        # markets present: h2h (3) + totals (2) + asian_handicap (2) = 7
        keys = {(r["market"], r["selection"]) for r in rows}
        assert ("h2h", "Malaysia") in keys
        assert ("h2h", "Draw") in keys
        assert ("h2h", "Vietnam") in keys
        assert ("totals", "Over") in keys and ("totals", "Under") in keys
        assert ("asian_handicap", "Home") in keys and ("asian_handicap", "Away") in keys
    asyncio.run(run())


def test_fetch_odds_history_preserves_line_and_price_movement():
    async def run():
        c = _FakeClient(_MIXODDS)
        h = await c.fetch_odds_history(_fixture())
        # Pre-match movement rows only -- realtime (``r``) snapshots are
        # additive and flagged ``snapshot: "live"``, never merged into the
        # opening->latest movement pair.
        rows = {
            r["selection"]: r for r in h["markets"]
            if r["market"] == "asian_handicap" and r.get("snapshot") != "live"
        }
        home = rows["Home"]
        # line movement: opening +0.75 -> latest +1.0 (a LINE move, not price).
        # Raw NowGoal line is the AWAY handicap (Vietnam -0.75 -> -1.0); the
        # Home row carries the home handicap (Malaysia +0.75 -> +1.0).
        assert home["opening_line"] == 0.75
        assert home["latest_line"] == 1.0
        # price movement preserved alongside
        assert home["opening_price"] is not None
        assert home["latest_price"] is not None
        assert home["bookmaker"] == "Bet365"
        ou = {
            r["selection"]: r for r in h["markets"]
            if r["market"] == "totals" and r.get("snapshot") != "live"
        }
        assert ou["Over"]["opening_line"] == 2.25
        assert ou["Over"]["latest_line"] == 2.5
        # the realtime leg is present and flagged
        live = [r for r in h["markets"] if r.get("snapshot") == "live"]
        assert live and all(r["snapshot"] == "live" for r in live)
        assert h["has_live"] is True
    asyncio.run(run())


def test_fetch_odds_history_none_without_data():
    async def run():
        c = _FakeClient({"ErrCode": 0, "Data": {"mixodds": []}})
        h = await c.fetch_odds_history(_fixture())
        assert h is None
    asyncio.run(run())


def test_fetch_odds_emits_opening_point_for_line_movement():
    """fetch_odds normalized payload now carries the opening LINE so the
    signal engine can separate line movement from price movement (S10)."""
    async def run():
        c = _FakeClient(_MIXODDS)
        payload = await c.fetch_odds(_fixture())
        ah_market = None
        ou_market = None
        for bm in payload["bookmakers"]:
            for m in bm["markets"]:
                if m["key"] == "asian_handicap":
                    ah_market = m
                elif m["key"] == "totals":
                    ou_market = m
        ah_home = next(o for o in ah_market["outcomes"] if o["name"] == "Home")
        # Raw NowGoal line is the AWAY handicap (Vietnam -1.0); the normalized
        # payload carries per-side points -> Home handicap +1.0 / opening +0.75.
        assert ah_home["point"] == 1.0           # latest home handicap
        assert ah_home["opening_point"] == 0.75  # opening home handicap
        ou_over = next(o for o in ou_market["outcomes"] if o["name"] == "Over")
        assert ou_over["point"] == 2.5
        assert ou_over["opening_point"] == 2.25
    asyncio.run(run())


def test_extract_asian_handicap_captures_opening_line():
    async def run():
        c = _FakeClient(_MIXODDS)
        payload = await c.fetch_odds(_fixture())
        rows = extract_asian_handicap(payload)
        assert len(rows) == 1
        # home handicap (Malaysia +1.0, opening +0.75) after the away-handicap
        # raw line is normalized per side
        assert rows[0]["line"] == 1.0
        assert rows[0]["line_open"] == 0.75
    asyncio.run(run())


def test_extract_market_totals_captures_opening_line():
    async def run():
        c = _FakeClient(_MIXODDS)
        payload = await c.fetch_odds(_fixture())
        totals = extract_market_totals(payload)
        assert totals["Over 2.5"]["point"] == 2.5
        assert totals["Over 2.5"]["opening_point"] == 2.25
        assert totals["Under 2.5"]["opening_point"] == 2.25
    asyncio.run(run())


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
