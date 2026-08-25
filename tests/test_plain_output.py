"""Tests for plain-text Discord output (no embeds, no truncation).

Verifies: _plain_messages chunks a long report into <=2000-char messages
(capped at _MAX_PLAIN_MESSAGES with a Copy-button note), short reports stay
single-message, and format_top's extra-matches section renders compactly so
the whole top report fits without hitting the old 4096-char embed cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402
from agents.football.format import format_top  # noqa: E402


def test_plain_messages_short_single_chunk():
    rendered = {
        "title": "⚽ Value Match — 2026-08-12",
        "body": "**1. A vs B** • UCL • 20:00 WIB",
        "footer": "odds quota: 394/500",
    }
    chunks = bot._plain_messages(rendered)
    assert len(chunks) == 1
    assert "**1. A vs B**" in chunks[0]
    assert "odds quota: 394/500" in chunks[0]


def test_plain_messages_long_capped_no_truncation():
    body = "\n\n".join(
        f"**{i}. Team{i} vs Opp{i}** • UCL • 20:00 WIB\n"
        f"   odds: 1.5 / 3.2 / 5.0 (5 bookie)\n"
        f"   edge: - • form W vs L\n   sinyal: **{i}/100**"
        for i in range(1, 600)
    )
    rendered = {"title": "⚽ Value Match", "body": body, "footer": "odds quota: 394/500"}
    chunks = bot._plain_messages(rendered)
    assert len(chunks) == bot._MAX_PLAIN_MESSAGES
    assert all(len(c) <= 2000 for c in chunks)
    # truncation marker present; footer always survives on the final message
    assert any("…(lanjutan via tombol 📋 Detail)" in c for c in chunks)
    assert chunks[-1].endswith("odds quota: 394/500")


def test_plain_messages_each_chunk_within_limit():
    body = "\n".join(f"line-{i:04d} abcdefghij" for i in range(400))  # ~8k chars
    chunks = bot._plain_messages({"title": "T", "body": body, "footer": "F"})
    assert all(len(c) <= 2000 for c in chunks)
    assert chunks[0].startswith("T\n\nline-")


def test_plain_messages_empty_rendered():
    chunks = bot._plain_messages({})
    assert chunks and all(len(c) <= 2000 for c in chunks)


def test_plain_messages_footer_survives_truncation():
    """The footer (quota/sources/grade summary) must survive the 4-message
    cap — it is pinned to the last visible message, not dropped with the
    discarded body chunks."""
    body = "\n\n".join(
        f"**{i}. Team{i} vs Opp{i}**\n   odds: 1.5 / 3.2 / 5.0 (5 bookie)\n   sinyal: **{i}/100**"
        for i in range(1, 800)
    )
    footer = "🟢 2 layak • 🟡 1 cukup • odds quota: 394/500"
    chunks = bot._plain_messages({"title": "⚽ Value Match", "body": body, "footer": footer})
    assert len(chunks) == bot._MAX_PLAIN_MESSAGES
    # truncation marker present; footer is still visible on the final message
    assert any("…(lanjutan via tombol 📋 Detail)" in c for c in chunks)
    assert chunks[-1].endswith(footer)
    assert all(len(c) <= 2000 for c in chunks)


def test_format_top_compact_extra_matches():
    payload = {
        "date": "2026-08-12",
        "matches": [
            {
                "home": "Lyon", "away": "Sparta Prague", "league": "UCL",
                "kickoff": "2026-08-12T19:00:00Z",
                "odds": {"consensus": {"home": 1.37, "draw": 5.1, "away": 7.5}, "outlier": None},
                "stats": {"home_form": "W", "away_form": "W"},
                "signal": 80, "has_odds": True, "bookmakers_count": 14,
                "grade": {"grade": "LAYAK", "label": "LAYAK"},
            }
        ],
        "extra_matches": [
            {"home": f"H{i}", "away": f"A{i}", "competition": c, "kickoff": None, "source": "flashscore"}
            for i, c in enumerate(["Conf L", "Conf L", "UCL", "UCL", "UCL", "Friendly"] * 40)
        ],
        "quota": {"odds_api_remaining": 394},
    }
    out = format_top(payload)
    extra_part = out["body"][out["body"].find("Kompetisi lain"):]
    # Only analyzable competitions are rendered: 'Conf L' and 'Friendly' are
    # hidden, only UCL's 120 matches remain (40 rounds x 3 UCL rows).
    assert "Kompetisi lain (flashscore, bisa dianalisa): 120 match" in extra_part
    assert "**UCL** (UCL) 120" in extra_part
    assert "Friendly" not in extra_part
    assert "Conf L" not in extra_part
    # example fixtures are capped: top 3 competitions x 2 matches
    assert extra_part.count("`") <= 12
    # whole report fits comfortably inside one plain message
    assert len(out["body"]) < 1900


def test_format_top_no_extra_matches():
    payload = {
        "date": "2026-08-12",
        "matches": [],
        "extra_matches": [],
        "quota": {},
    }
    out = format_top(payload)
    assert "Tidak ada match ditemukan" in out["body"]


def test_footer_pinned_on_last_message_real_top_report():
    """Real format_top render: the footer (quota + grade summary) must be
    pinned to the LAST message even when the body spans several chunks."""
    matches = []
    for i in range(1, 12):
        matches.append({
            "home": f"Team {i}", "away": f"Opponent {i}", "league": "UCL",
            "kickoff": "2026-08-12T19:00:00Z",
            "odds": {"consensus": {"home": 1.37, "draw": 5.10, "away": 7.50}, "outlier": None},
            "stats": {"home_form": "W-W-D-L", "away_form": "L-D-W-W"},
            "signal": 80 - i, "has_odds": True, "bookmakers_count": 14,
            "grade": {"grade": "LAYAK", "label": "LAYAK"},
        })
    rendered = format_top({
        "date": "2026-08-12",
        "matches": matches,
        "extra_matches": [
            {"home": f"H{i}", "away": f"A{i}", "competition": c, "kickoff": None, "source": "flashscore"}
            for i, c in enumerate(["Conf L", "UCL Qual", "Friendly", "Cup"] * 30)
        ],
        "quota": {"odds_api_remaining": 394},
        "leagues_no_odds": [],
    })
    footer = (rendered.get("footer") or "").strip()
    assert footer and footer != " "
    chunks = bot._plain_messages(rendered)
    assert len(chunks) <= bot._MAX_PLAIN_MESSAGES
    assert all(len(c) <= 2000 for c in chunks)
    # footer quota + grade summary survive on the LAST message
    assert footer in chunks[-1]
    assert "odds quota: 394/500" in chunks[-1]


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
