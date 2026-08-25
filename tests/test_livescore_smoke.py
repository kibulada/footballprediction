"""REAL LiveScore connectivity smoke test (separated from the unit suite).

Not part of the deterministic unit tests. Runs ONLY when the env var
``HERMES_LIVESCORE_SMOKE=1`` is set; otherwise it passes trivially (no
network). This proves the verified ``lsmedia1.com`` public API is reachable
from the production network and that real fixtures parse + resolve.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.livescore import (  # noqa: E402
    LiveScoreClient,
    LiveScoreDataSource,
    parse_soccer_payload,
)

_ENABLED = os.getenv("HERMES_LIVESCORE_SMOKE") == "1"


def _run(coro):
    return asyncio.run(coro)


def test_real_feed_reachable():
    if not _ENABLED:
        return
    health = _run(LiveScoreClient().health())
    assert health["reachable"] is True, f"livescore unreachable: {health}"


def test_real_feed_parses_fixtures():
    if not _ENABLED:
        return
    client = LiveScoreClient()
    payload = _run(client.fetch_soccer_date("20260815", 0))
    assert payload is not None
    fixtures = parse_soccer_payload(payload)
    assert len(fixtures) > 0
    assert all(f["home"] and f["away"] and f["kickoff"] for f in fixtures)


def test_real_match_resolves():
    if not _ENABLED:
        return
    src = LiveScoreDataSource(LiveScoreClient(), max_pages=3)
    sample = _run(src.get_match(
        {"home": "Deportivo Alaves", "away": "Getafe", "kickoff": "2026-08-15T17:30:00Z"}
    ))
    assert sample.status == "available", f"match not resolved: {sample}"
    assert sample.value["home"] == "deportivo alaves"


def test_real_field_endpoints_return_data():
    if not _ENABLED:
        return
    src = LiveScoreDataSource(LiveScoreClient(), max_pages=3)
    ref = {"home": "Deportivo Alaves", "away": "Getafe", "kickoff": "2026-08-15T17:30:00Z"}
    lu = _run(src.get_lineup(ref))
    assert lu.status == "available", f"lineups: {lu.status}"
    assert lu.value["home"]["players"]
    h2h = _run(src.get_h2h(ref))
    assert h2h.status == "available", f"h2h: {h2h.status}"
    form = _run(src.get_form(ref))
    assert form.status == "available" and form.value["home"]["sequence"]
    table = _run(src.get_standings(ref))
    assert table.status == "available" and table.value["home"]["team"] == "Deportivo Alaves"


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
