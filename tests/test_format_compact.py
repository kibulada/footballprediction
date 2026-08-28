"""Tests for the compact analyse summary (format_compact).

The main reply follows the OUTPUT POLICY — Single Best Pick (selection layer
only): every market (1X2, O/U 2.5, O/U 3.5, BTTS) is STILL computed and
tiered in full, but only ONE already-computed result is surfaced
(PICK > LEAN > WATCH) with its tier, confidence, basis and stake. Tiers are
derived purely from the computed confidence/disagreement/edge values — the
selection can never manually raise or lower one. Finished matches show the
real result instead (no tiers, no prediction).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.football.format import (  # noqa: E402
    format_compact,
    format_market_signal,
    format_signal_detail,
)
from agents.football.signal_engine import _demo  # noqa: E402


def _base_payload(**overrides):
    payload = {
        "league": "EPL",
        "home": "Arsenal",
        "away": "Chelsea",
        "kickoff": "2026-08-16T18:30:00Z",
        "stats": {
            "home_form": "W-W-D",
            "away_form": "L-D-W",
            "home_gf_avg": 1.8,
            "home_ga_avg": 0.9,
            "away_gf_avg": 1.3,
            "away_ga_avg": 1.1,
            "h2h": {"wins": 2, "draws": 1, "losses": 1},
            "home_xg_for": None,
            "away_xg_for": None,
            "home_xg_against": None,
            "away_xg_against": None,
        },
        "odds": {
            "has_odds": True,
            "bookmakers_count": 14,
            # Home odds 2.25 keeps the 1X2 PICK at positive EV
            # (0.47 * 2.25 - 1 = +5.75% > MIN_EV 3%); over_2.5 0.58 keeps
            # O/U 2.5 at positive EV too (0.58 * 1.85 - 1 = +7.3%).
            "consensus": {"home": 2.25, "draw": 3.40, "away": 4.20},
            "totals": {
                "Over 2.5": {"odds": 1.85, "point": 2.5},
                "Under 2.5": {"odds": 1.95, "point": 2.5},
            },
        },
        "prediction": {
            "model_probs": {
                "1x2": {"home": 0.47, "draw": 0.27, "away": 0.26},
                "over_2.5": 0.58,
                "models": ["elo", "poisson"],
            },
            "data_completeness": 0.85,
        },
        "decision": {
            "decision_type": "GOOD",
            "final_decision": {
                "market": "Total",
                "selection": "Under 3.5",
                "market_odds": 1.82,
                "model_prob": 0.55,
                "edge_pp": 4.1,
                "ev": 0.07,
            },
            "most_likely": {"selection": "Under 3.5", "model_prob": 0.55},
            "explanation": "Some long engine explanation that should be clipped.",
            "edge_warnings": [],
            "model_disagreement": {"flag": False, "delta_pp": 3.0},
        },
        "confidence": {
            "tier": "HIGH",
            "tier_before_caps": "HIGH",
            "caps_applied": [],
            "n_bucket": 40,
            "completeness_factor": 0.85,
            "pick_specific_confidence": 0.8,
        },
    }
    payload.update(overrides)
    return payload


def _body(payload):
    return format_compact(payload)["body"]


def _lines(payload):
    return _body(payload).splitlines()


def test_match_line_first():
    lines = _lines(_base_payload())
    assert lines[0].startswith("📊 Arsenal vs Chelsea — EPL • ")
    assert lines[1] == ""


def test_exactly_one_tier_line():
    # OUTPUT POLICY: one result per match — never the full market breakdown.
    body = _body(_base_payload())
    tier_lines = [l for l in body.splitlines() if l.startswith(("🟢 ", "🔶 ", "⚪ "))]
    assert len(tier_lines) == 1


def test_single_pick_among_two_pick_markets():
    # 1X2 (disagreement 3.0pp) and O/U 2.5 (no disagreement data) are BOTH
    # Tier 1 -> the one with the lowest disagreement is surfaced.
    lines = _lines(_base_payload())
    assert lines[2] == "🟢 PICK: Over 2.5 @ 1.85"
    assert lines[3] == (
        "Confidence: HIGH | Basis: lowest disagreement among 2 Tier-1 markets "
        "(no Model A/B disagreement data)"
    )
    assert lines[4] == "Stake: Normal 1 unit"


def test_pick_beats_lean():
    # 1X2 has a large edge (LEAN) but O/U 2.5 is still Tier 1 -> the PICK is
    # surfaced; the LEAN market was computed but is not the one displayed.
    payload = _base_payload()
    payload["prediction"]["model_probs"]["1x2"] = {"home": 0.62, "draw": 0.22, "away": 0.16}
    lines = _lines(payload)
    assert lines[2] == "🟢 PICK: Over 2.5 @ 1.85"
    assert lines[4] == "Stake: Normal 1 unit"
    assert "BET" not in _body(payload)


def test_disagreement_over_20pp_shifts_1x2_to_watch():
    # 1X2 drops to WATCH (25pp disagreement); O/U 2.5 remains the only PICK.
    payload = _base_payload()
    payload["decision"]["model_disagreement"] = {"flag": True, "delta_pp": 25.0}
    lines = _lines(payload)
    assert lines[2] == "🟢 PICK: Over 2.5 @ 1.85"
    assert "only market at Tier 1 (PICK)" in lines[3]
    assert lines[4] == "Stake: Normal 1 unit"


def test_contradictory_extreme_edges_are_watch_and_never_hidden():
    # O/U 2.5 gets contradictory extreme edges -> WATCH; 1X2 stays PICK and is
    # the surfaced result. Prohibition: no BET wording.
    payload = _base_payload()
    payload["prediction"]["model_probs"]["over_2.5"] = 0.74
    lines = _lines(payload)
    assert lines[2] == "🟢 PICK: Home Win @ 2.25"
    assert lines[3] == (
        "Confidence: HIGH | Basis: only market at Tier 1 (PICK); "
        "disagreement 3.0pp"
    )
    assert lines[4] == "Stake: Normal 1 unit"
    body = _body(payload)
    assert "BET" not in body
    assert "BETTING RECOMMENDATION" not in body


def test_all_watch_below_threshold_is_no_bet():
    # No decision/confidence -> every market is WATCH with a sub-threshold
    # edge (< 10pp). The honest reply is NO BET, never a fabricated
    # "directional lean" direction for a noise edge.
    payload = _base_payload()
    payload["decision"] = None
    payload["confidence"] = None
    lines = _lines(payload)
    assert lines[2].startswith("🚫 NO BET")
    assert "BETTING RECOMMENDATION" not in _body(payload)


def test_no_data_states_honestly():
    # No evaluable market: explicit honest line, never a fabricated pick.
    payload = {
        "league": "EPL", "home": "A", "away": "B", "kickoff": None,
        "stats": {}, "odds": {}, "prediction": {}, "decision": {},
    }
    body = _body(payload)
    assert "Tidak ada market dengan data cukup" in body


def test_negative_ev_1x2_never_surfaced_as_pick():
    # Regression: the 1X2 market can qualify on confidence/disagreement while
    # its EV at 1.95 is negative (0.47 * 1.95 - 1 = -8.4% <= MIN_EV). The EV
    # gate demotes it to WATCH, so the compact reply surfaces the positive-EV
    # O/U 2.5 PICK instead of "Home Win ... Stake: Normal 1 unit".
    payload = _base_payload()
    payload["odds"]["consensus"] = {"home": 1.95, "draw": 3.40, "away": 4.20}
    lines = _lines(payload)
    assert lines[2] == "🟢 PICK: Over 2.5 @ 1.85"
    assert "Home Win" not in lines[2]
    assert "Stake: Normal 1 unit" in lines[4]


def test_disclaimer_single_line_at_end():
    lines = _lines(_base_payload())
    assert lines[-1] == "Not a guarantee of outcome. Betting decisions are the user's own risk."
    assert sum(1 for l in lines if "Not a guarantee of outcome" in l) == 1


def test_error_payload_keeps_full_error():
    rendered = format_compact({"error": "tim tidak ditemukan", "home_query": "X", "away_query": "Y"})
    assert "tim tidak ditemukan" in rendered["body"]


def test_compact_within_plain_message_limit():
    rendered = format_compact(_base_payload())
    body = rendered["title"] + "\n" + rendered["body"]
    assert len(body) < 1900


# ---- finished match (real result, no prediction / no tiers) ---------------


def _finished_payload(**overrides):
    payload = {
        "league": "UEL",
        "home": "Hearts",
        "away": "Benfica",
        "kickoff": "2026-08-13T20:00:00Z",
        "match_finished": True,
        "match_result": {"home": "1", "away": "2"},
        "event_stats": {
            "xg_home": 1.2, "xg_away": 2.1,
            "possession_home": 41.0, "possession_away": 59.0,
        },
        "stats": {
            "home_form": "W-D", "away_form": "W-W",
            "h2h": {"wins": 1, "draws": 1, "losses": 2},
        },
    }
    payload.update(overrides)
    return payload


def test_finished_match_shows_result_no_tiers():
    body = _body(_finished_payload())
    assert "**Match sudah selesai**" in body
    assert "prediksi tidak dibuat untuk match yang sudah selesai" in body
    assert "⚽ Hasil: **Hearts 1 - 2 Benfica**" in body
    assert "xG: 1.2 vs 2.1" in body
    assert "Possession: 41.0% vs 59.0%" in body
    assert "form (konteks): W-D vs W-W" in body
    # no prediction artifacts, no market tiers
    assert "── " not in body
    assert "PICK" not in body
    assert "Not a guarantee of outcome" not in body


def test_finished_match_without_result_is_honest():
    body = _body(_finished_payload(match_result=None, event_stats=None))
    assert "Hasil tidak tersedia (resolve gagal / data belum rilis)." in body
    assert "prediksi tidak dibuat" in body


# ---- MARKET SIGNAL primary output (analyse) ------------------------------

def _signal_payload(se=None, **overrides):
    payload = {
        "league": "ASEAN", "home": "Singapore", "away": "Thailand",
        "kickoff": "2026-08-16T13:00:00Z",
        "signal_engine": se if se is not None else _demo(),
        "odds": {"totals": {
            "Over 2.5": {"odds": 1.87, "opening": 1.95, "point": 2.5},
            "Under 2.5": {"odds": 1.95, "opening": 1.86, "point": 2.5},
        }},
    }
    payload.update(overrides)
    return payload


def test_market_signal_renders_sections_and_best_pick():
    body = format_market_signal(_signal_payload())["body"]
    assert format_market_signal(_signal_payload())["title"] == "🔬 MATCH SIGNAL"
    assert "Singapore vs Thailand" in body
    assert "📊 SIGNALS" in body
    # K5 (2026-08-28): the demo's top signal is LOW (51/100) -> it is shown
    # as a LEAN, never dressed up as a BEST PICK. A MEDIUM+ pick keeps the
    # "🏆 BEST PICK" header (see test_market_signal_best_pick_header_for_strong_pick).
    assert "🏆 LEAN" in body
    assert "📈 MARKET" in body
    assert "⚠️ DISCLAIMER" in body
    # the engine's strongest signal is surfaced (LEAN icon for a weak pick)
    assert "📌 " in body
    # no engine internals leak into the primary card
    for banned in ("n_bucket", "λ", "calibration", "FINAL DECISION",
                   "disagreement", "lolos gate", "confluence", "REVIEW_REQUIRED"):
        assert banned not in body


def test_market_signal_best_pick_header_for_strong_pick():
    """K5: a MEDIUM+ pick above medium_score keeps the BEST PICK header."""
    se = _demo()
    se["pick_tier"] = "BEST PICK"
    if se.get("best_pick"):
        se["best_pick"]["confidence"] = "MEDIUM"
        se["best_pick"]["score"] = 0.6
    body = format_market_signal(_signal_payload(se))["body"]
    assert "🏆 BEST PICK" in body
    assert "🔥 " in body


def test_market_signal_no_bet_when_all_weak():
    se = _demo()
    se["decision"] = "NO BET"
    se["best_pick"] = None
    se["reasons"] = ["top score 0.40, margin 0.10", "best score 0.40 < 0.45"]
    body = format_market_signal(_signal_payload(se))["body"]
    assert "⚪ NO BET" in body
    assert "No signal reaches the actionable threshold." in body


def test_market_signal_omits_unavailable_markets():
    # no AH consensus -> the AH market line is simply omitted, never a
    # "not evaluated" placeholder.
    se = _demo()
    se["ah_consensus"] = None
    body = format_market_signal(_signal_payload(se))["body"]
    assert "Asian Handicap" not in body
    assert "Not evaluated" not in body


def test_market_block_shows_movement():
    body = format_market_signal(_signal_payload())["body"]
    assert "Over 2.5" in body
    assert "Opening: 1.95" in body
    assert "Latest: 1.87" in body
    assert "Movement: ↓ 4.1%" in body


def test_signal_detail_has_no_debug_internals():
    body = format_signal_detail(_signal_payload())["body"]
    assert "🏆 LEAN" in body  # K5: demo top signal is LOW -> LEAN header
    assert "📈 MARKET" in body
    assert "📊 Data quality" in body
    for banned in ("n_bucket", "λ_home", "calibration", "FINAL DECISION",
                   "MODEL_DISAGREEMENT", "REVIEW_REQUIRED", "lolos gate"):
        assert banned not in body


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
