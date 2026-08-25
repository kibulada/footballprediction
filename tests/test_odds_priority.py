"""Re-prioritas odds 2026-08-24 ("NowGoal primary, OddsPapi validator").

Covers the pure decision surface of the flip:
- ``_merge_missing_btts``: BTTS prices exist ONLY in the OddsPapi/TheOddsAPI
  payload shape; when NowGoal is primary its totals must inherit BTTS from
  the secondary payload or G7 (require_price) vetoes every BTTS pick.
- Payload selection semantics are asserted indirectly by the full analyse
  suite (mocked nowgoal/oddspapi paths); the helper below is the piece with
  real merge logic.
"""
from __future__ import annotations

from agents.football.analyse import _merge_missing_btts


def _totals(**entries):
    return {label: {"odds": odds} for label, odds in entries.items()}


def test_merge_fills_missing_btts_from_secondary():
    primary = _totals(**{"Over 2.5": 1.85})
    secondary = _totals(**{"BTTS Yes": 1.70, "BTTS No": 2.10})
    out = _merge_missing_btts(primary, secondary)
    assert out["Over 2.5"] == {"odds": 1.85}
    assert out["BTTS Yes"] == {"odds": 1.70}
    assert out["BTTS No"] == {"odds": 2.10}


def test_merge_never_overrides_primary_prices():
    primary = _totals(**{"BTTS Yes": 1.90})
    secondary = _totals(**{"BTTS Yes": 1.70, "BTTS No": 2.10})
    out = _merge_missing_btts(primary, secondary)
    assert out["BTTS Yes"] == {"odds": 1.90}  # primary is the single writer
    assert out["BTTS No"] == {"odds": 2.10}


def test_merge_copies_entry_dict_not_reference():
    sec_entry = {"odds": 1.7, "bookmaker": "Pinnacle"}
    out = _merge_missing_btts({}, {"BTTS Yes": sec_entry})
    out["BTTS Yes"]["bookmaker"] = "MUTATED"
    assert sec_entry["bookmaker"] == "Pinnacle"  # secondary untouched


def test_merge_handles_none_and_empty_inputs():
    assert _merge_missing_btts(None, None) == {}
    assert _merge_missing_btts({}, None) == {}
    only_sec = _merge_missing_btts(None, _totals(**{"BTTS No": 2.0}))
    assert only_sec == {"BTTS No": {"odds": 2.0}}
    # non-BTTS labels never copied
    out = _merge_missing_btts({}, _totals(**{"Over 2.5": 1.8}))
    assert out == {}


def test_nowgoal_payload_shape_has_no_btts_market():
    """Guard for the PREMISE of the merge: NowGoal's normalized payload
    (euro/ou/ah) must not grow a btts key unnoticed -- if it ever does,
    the secondary-BTTS merge premise needs re-evaluation."""
    import inspect

    from agents.football import nowgoal

    src = inspect.getsource(nowgoal.NowGoalClient.fetch_odds)
    assert '"btts"' not in src and "'btts'" not in src
