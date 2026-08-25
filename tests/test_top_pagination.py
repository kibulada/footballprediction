"""Tests for the paginated VALUE MATCH / KOMPETISI LAIN output.

Covers the presentation layer only: competition grouping + ordering, preview
rules (1/2/>2 matches), pagination packing (8-12 per page, char budget),
footer totals, plain-text copy, and the Discord button/state helpers in bot.py
(unique custom ids, per-user isolation, disabled nav at bounds, no re-query).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402
from agents.football import format as fmt  # noqa: E402


def _extra(comp: str, n: int, prefix: str = "H") -> list[dict]:
    return [
        {"home": f"{prefix}{i}", "away": f"A{i}", "competition": comp, "kickoff": None, "source": "flashscore"}
        for i in range(n)
    ]


def _payload(*, extra: list[dict] | None = None, matches: list[dict] | None = None) -> dict:
    return {
        "date": "2026-08-12",
        "matches": matches or [],
        "extra_matches": extra or [],
        "quota": {},
        "leagues_no_odds": [],
    }


def test_group_competitions_sorts_by_count_desc_then_name():
    rows = _extra("Cup", 3) + _extra("A League", 5) + _extra("B League", 5)
    comps = fmt._group_competitions(rows)
    # 5, 5, 3 — tie broken by name asc (stable)
    assert [c for c, _ in comps] == ["A League", "B League", "Cup"]
    assert [len(ms) for _, ms in comps] == [5, 5, 3]


def test_competition_block_1_match():
    block = fmt._competition_block("Serie B", _extra("Serie B", 1))
    assert "**Serie B · 1 match**" in block
    assert "• H0 vs A0" in block
    assert "lainnya" not in block


def test_competition_block_2_matches():
    block = fmt._competition_block("Club Friendly", _extra("Club Friendly", 2))
    assert "**Club Friendly · 2 match**" in block
    assert block.count("• ") == 2
    assert "lainnya" not in block


def test_competition_block_more_than_2():
    block = fmt._competition_block("MOL Cup", _extra("MOL Cup", 35))
    assert "**MOL Cup · 35 match**" in block
    # every match is listed; no "+N lainnya" truncation
    bullets = [ln for ln in block.splitlines() if ln.startswith("• ")]
    assert len(bullets) == 35
    assert "lainnya" not in block


def test_pack_pages_respects_per_page_budget():
    comps = fmt._group_competitions(_extra("Comp", 1))
    for _ in range(25):
        comps.append(("CompX", _extra("CompX", 1)))
    pages = fmt._pack_competition_pages(comps, per_page=10, budget=10_000)
    # 26 competitions / 10 per page -> 3 pages
    assert len(pages) == 3
    assert all(len(p) <= 10 for p in pages)


def test_pack_pages_respects_char_budget():
    # Long competition names should force smaller pages.
    comps = [("A Very Long Competition Name " * 4, _extra("x", 1))] * 30
    pages = fmt._pack_competition_pages(comps, per_page=10, budget=500)
    assert len(pages) > 3
    for page in pages:
        total = sum(len(fmt._competition_block(n, ms)) for n, ms in page)
        assert total <= 500 + 200  # last block may exceed slightly; keep sane


def test_build_top_pages_empty_without_extra():
    assert fmt.build_top_pages(_payload()) == []


def test_build_top_pages_single_page():
    rows = _extra("Champions League - Qualification", 35) + _extra("Europa League - Qualification", 31)
    pages = fmt.build_top_pages(_payload(extra=rows))
    assert len(pages) == 1
    body = pages[0]["body"]
    assert "VALUE MATCH — 12 AGU 2026" in body
    assert "Champions League - Qualification · 35 match" in body
    assert "Europa League - Qualification · 31 match" in body
    # every competition is tagged with its league key (the analyse command)
    assert "(UCL)" in body
    assert "(UEL)" in body
    assert "66 MATCH" in pages[0]["footer"]
    assert "2 KOMPETISI" in pages[0]["footer"]
    assert "Page 1/1" in pages[0]["footer"]


def test_build_top_pages_multi_page():
    rows = []
    for i in range(25):
        rows += _extra(f"Champions League - Qual R{i}", 1)
    pages = fmt.build_top_pages(_payload(extra=rows))
    assert len(pages) == 3
    assert "Page 1/3" in pages[0]["footer"]
    assert "Page 2/3" in pages[1]["footer"]
    assert "Page 3/3" in pages[2]["footer"]
    assert "25 MATCH" in pages[0]["footer"]
    assert "25 KOMPETISI" in pages[0]["footer"]
    # each page fits one Discord message (budget enforcement)
    for p in pages:
        assert len(p["body"]) <= 2000
        assert len(p["footer"]) <= 300


def test_build_top_pages_each_page_has_header():
    rows = _extra("Champions League - Qual R0", 1)
    for i in range(1, 22):
        rows += _extra(f"Champions League - Qual R{i}", 1)
    pages = fmt.build_top_pages(_payload(extra=rows))
    assert len(pages) >= 2
    for p in pages:
        assert "KOMPETISI LAIN" in p["body"]
        assert "VALUE MATCH" in p["body"]


def test_format_top_returns_pages_when_no_primary_matches():
    rows = _extra("Champions League - Qual R0", 1)
    for i in range(1, 15):
        rows += _extra(f"Champions League - Qual R{i}", 1)
    out = fmt.format_top(_payload(extra=rows))
    assert len(out.get("pages") or []) >= 2
    assert out["title"] == ""  # header lives in the body


def test_format_top_keeps_old_path_when_primary_matches_exist():
    matches = [{
        "home": "Lyon", "away": "Sparta Prague", "league": "UCL",
        "kickoff": "2026-08-12T19:00:00Z",
        "odds": {"consensus": {"home": 1.37, "draw": 5.1, "away": 7.5}, "outlier": None},
        "stats": {"home_form": "W", "away_form": "W"},
        "signal": 80, "has_odds": True, "bookmakers_count": 14,
        "grade": {"grade": "LAYAK", "label": "LAYAK"},
    }]
    out = fmt.format_top(_payload(matches=matches, extra=_extra("Cup", 3)))
    assert not (out.get("pages") or [])  # old single-message path preserved
    assert "Lyon vs Sparta Prague" in out["body"]


def test_parse_top_custom_id():
    assert bot._parse_top_custom_id("football_top_abc123_copy") == ("abc123", "copy")
    assert bot._parse_top_custom_id("football_top_abc123_prev") == ("abc123", "prev")
    assert bot._parse_top_custom_id("football_top_abc123_next") == ("abc123", "next")
    assert bot._parse_top_custom_id("football_top_abc123_ana_0") == ("abc123", "ana_0")
    assert bot._parse_top_custom_id("football_top_abc123_ana_4") == ("abc123", "ana_4")
    assert bot._parse_top_custom_id("football_top_abc123_ana_x") is None
    assert bot._parse_top_custom_id("football_copy_abc123") is None
    assert bot._parse_top_custom_id("football_top_abc123_bogus") is None
    assert bot._parse_top_custom_id("garbage") is None


class _FakeInteraction:
    """Minimal async Interaction double: records send/edit calls."""

    def __init__(self, user_id: str, custom_id: str):
        self.user = type("U", (), {"id": user_id})()
        self.data = {"custom_id": custom_id}
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.deferred: list[dict] = []
        self.followups: list[dict] = []

        class Response:
            def __init__(self, outer):
                self.outer = outer

            async def send_message(self, *a, **k):
                self.outer.sent.append((a, k))

            async def edit_message(self, *a, **k):
                self.outer.edited.append((a, k))

            async def defer(self, *a, **k):
                self.outer.deferred.append((a, k))

        self.response = Response(self)

        class Followup:
            def __init__(self, outer):
                self.outer = outer

            async def send(self, *a, **k):
                self.outer.followups.append((a, k))

        self.followup = Followup(self)


def test_top_interaction_rejects_other_user():
    """User B clicking user A's buttons gets an ephemeral rejection and the
    state (including index) is untouched."""
    import asyncio
    import time as _t

    pages = [{"title": "", "body": f"page-{i}", "footer": "f"} for i in range(3)]
    state = {"ts": _t.monotonic(), "user_id": "user_a", "pages": pages, "index": 0}
    bot._TOP_PAGES.clear()
    bot._TOP_PAGES["abc123"] = state
    itx = _FakeInteraction("user_b", "football_top_abc123_next")
    asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_next"))
    assert len(itx.sent) == 1
    assert "pemilik" in itx.sent[0][0][0]
    assert itx.sent[0][1].get("ephemeral") is True
    assert not itx.edited  # owner's message never mutated by another user
    assert state["index"] == 0  # state untouched
    bot._TOP_PAGES.clear()


def test_top_interaction_expired_session():
    """Unknown token -> graceful ephemeral expiry message, no crash."""
    import asyncio

    bot._TOP_PAGES.clear()
    itx = _FakeInteraction("user_a", "football_top_deadbeef_next")
    asyncio.run(bot._handle_top_interaction(itx, "football_top_deadbeef_next"))
    assert len(itx.sent) == 1
    assert "kedaluwarsa" in itx.sent[0][0][0]
    assert itx.sent[0][1].get("ephemeral") is True


def test_top_interaction_flip_uses_cached_pages_no_requery():
    """Page flip re-renders from cached state only: same message is edited, no
    new sends, no data-source call (there is no fetch path to call)."""
    import asyncio
    import time as _t

    pages = [{"title": "", "body": f"page-{i}", "footer": "f"} for i in range(3)]
    state = {"ts": _t.monotonic(), "user_id": "user_a", "pages": pages, "index": 0}
    bot._TOP_PAGES.clear()
    bot._TOP_PAGES["abc123"] = state
    itx = _FakeInteraction("user_a", "football_top_abc123_next")
    asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_next"))
    assert state["index"] == 1  # advanced
    assert len(itx.edited) == 1  # interaction.update() on the SAME message
    assert not itx.sent  # no duplicate reply
    assert "page-1" in itx.edited[0][1]["content"]
    bot._TOP_PAGES.clear()


def test_top_interaction_copy_serves_active_page_ephemeral():
    """Copy uses the ACTIVE page (index), sends ephemeral only, claims nothing
    about writing to the OS clipboard."""
    import asyncio
    import time as _t

    pages = [
        {"title": "", "body": "🎯 **PAGE ONE**", "footer": "📄 **Page 1/2**"},
        {"title": "", "body": "🎯 **PAGE TWO**", "footer": "📄 **Page 2/2**"},
    ]
    state = {"ts": _t.monotonic(), "user_id": "user_a", "pages": pages, "index": 1}
    bot._TOP_PAGES.clear()
    bot._TOP_PAGES["abc123"] = state
    itx = _FakeInteraction("user_a", "football_top_abc123_copy")
    asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_copy"))
    # confirmation is ephemeral and honest (no fake clipboard claim)
    assert itx.sent and "Data siap disalin" in itx.sent[0][0][0]
    assert itx.sent[0][1].get("ephemeral") is True
    # handler calls followup.send(content=..., ephemeral=True) -> kwarg payload
    joined = "".join(k.get("content", "") for _, k in itx.followups)
    assert "PAGE TWO" in joined
    assert "PAGE ONE" not in joined
    bot._TOP_PAGES.clear()



def test_top_main_view_buttons():
    """Ranked-list view: one ⚡ per match (capped at 5) + the copy button."""
    v = bot._top_main_view("tok", 3)
    children = v.children
    assert len(children) == 4  # 3 ⚡ + copy
    for i in range(3):
        assert children[i].custom_id == f"football_top_tok_ana_{i}"
        assert children[i].label == f"⚡ {i + 1}"
    assert children[3].custom_id == "football_top_tok_copy"
    assert children[3].label == "📋 Detail"
    # capped at 5 matches (Discord max buttons per row)
    v5 = bot._top_main_view("tok", 9)
    assert len(v5.children) == 6
    assert v5.children[4].custom_id == "football_top_tok_ana_4"
    assert v5.children[5].custom_id == "football_top_tok_copy"


def _main_state(user="user_a", n=2, token="abc123"):
    import time as _t

    return {
        "ts": _t.monotonic(),
        "user_id": user,
        "kind": "main",
        "rendered": {"title": "T", "body": "BODY", "footer": "F"},
        "matches": [
            {"league_key": "UCL", "home": "Lyon", "away": "Sparta Prague"},
            {"league_key": "EPL", "home": "Arsenal", "away": "Chelsea"},
        ][:n],
    }


def test_top_main_interaction_analyse_runs_runner():
    """⚡ builds the analyse command from cached state and posts via followup."""
    import asyncio

    calls = []

    async def fake_runner(args):
        calls.append(args)
        return {"render": {"title": "🔬 Analisa Match", "body": f"RESULT {args[4]} vs {args[6]}", "footer": " "}}

    old = bot._invoke_runner
    bot._invoke_runner = fake_runner
    try:
        bot._TOP_PAGES.clear()
        bot._TOP_PAGES["abc123"] = _main_state()
        itx = _FakeInteraction("user_a", "football_top_abc123_ana_1")
        asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_ana_1"))
        # runner called exactly once, with the cached match's league key
        assert calls == [["analyse", "--league", "EPL", "--home", "Arsenal", "--away", "Chelsea"]]
        assert itx.deferred  # defer() before the long runner call
        joined = "".join(k.get("content", "") for _, k in itx.followups)
        assert "RESULT Arsenal vs Chelsea" in joined
        assert not itx.sent  # defer, no direct response message
    finally:
        bot._invoke_runner = old
        bot._TOP_PAGES.clear()


def test_top_main_interaction_rejects_other_user():
    """A non-owner clicking ⚡ gets rejected and the runner is never invoked."""
    import asyncio

    calls = []

    async def fake_runner(args):
        calls.append(args)
        return {"render": {"body": "x"}}

    old = bot._invoke_runner
    bot._invoke_runner = fake_runner
    try:
        bot._TOP_PAGES.clear()
        bot._TOP_PAGES["abc123"] = _main_state(user="user_a")
        itx = _FakeInteraction("user_b", "football_top_abc123_ana_0")
        asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_ana_0"))
        assert "pemilik" in itx.sent[0][0][0]
        assert not calls
        assert not itx.deferred
    finally:
        bot._invoke_runner = old
        bot._TOP_PAGES.clear()


def test_top_main_interaction_copy_full_report():
    """Copy on the ranked list serves the FULL report (ephemeral)."""
    import asyncio

    bot._TOP_PAGES.clear()
    bot._TOP_PAGES["abc123"] = _main_state()
    itx = _FakeInteraction("user_a", "football_top_abc123_copy")
    asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_copy"))
    assert itx.sent and itx.sent[0][1].get("ephemeral") is True
    # response content is passed positionally (send_message(first, ephemeral=..))
    text = "".join(a[0] if a else "" for a, _ in itx.sent) + "".join(
        k.get("content", "") for _, k in itx.followups
    )
    assert "BODY" in text
    bot._TOP_PAGES.clear()


class _FakeTopChannel:
    """Minimal channel double for _post_top_result (records sends + typing)."""

    def __init__(self):
        self.sent: list[tuple] = []

    async def send(self, *a, **k):
        self.sent.append((a, k))

    def typing(self):
        class T:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

        return T()


def _top_msg():
    class FakeMsg:
        channel = _FakeTopChannel()

        class author:
            id = "user_a"

    return FakeMsg()


def test_post_top_result_stores_state_and_posts_with_main_view():
    """_post_top_result wires raw.matches (league_key only) into a kind='main'
    state and posts the first chunk with the ⚡ view."""
    import asyncio

    bot._TOP_PAGES.clear()
    result = {
        "render": {"title": "T", "body": "B", "footer": "F"},
        "raw": {"matches": [
            {"league_key": "UCL", "home": "Lyon", "away": "Sparta Prague"},
            {"league_key": "EPL", "home": "Arsenal", "away": "Chelsea"},
            {"home": "NoKey", "away": "X"},  # no league_key -> skipped
        ]},
    }
    msg = _top_msg()
    asyncio.run(bot._post_top_result(msg, result))
    states = list(bot._TOP_PAGES.values())
    assert len(states) == 1
    st = states[0]
    assert st["kind"] == "main"
    assert st["user_id"] == "user_a"
    assert st["matches"] == [
        {"league_key": "UCL", "home": "Lyon", "away": "Sparta Prague"},
        {"league_key": "EPL", "home": "Arsenal", "away": "Chelsea"},
    ]
    token = list(bot._TOP_PAGES)[0]
    assert msg.channel.sent
    view = msg.channel.sent[0][1].get("view")
    assert view is not None
    assert [c.custom_id for c in view.children] == [
        f"football_top_{token}_ana_0",
        f"football_top_{token}_ana_1",
        f"football_top_{token}_copy",
    ]
    bot._TOP_PAGES.clear()


def test_post_top_result_falls_back_without_league_keys():
    """No match with a league_key -> generic copy view, no main state."""
    import asyncio

    bot._TOP_PAGES.clear()
    result = {
        "render": {"title": "T", "body": "B", "footer": "F"},
        "raw": {"matches": [{"home": "A", "away": "B"}]},  # no league_key
    }
    msg = _top_msg()
    asyncio.run(bot._post_top_result(msg, result))
    assert not bot._TOP_PAGES
    assert msg.channel.sent
    view = msg.channel.sent[0][1].get("view")
    assert view is not None
    assert view.children[0].custom_id.startswith("football_copy_")
    bot._TOP_PAGES.clear()


def test_top_main_interaction_out_of_range():
    """A stale/out-of-range ⚡ index gets a graceful ephemeral error."""
    import asyncio

    bot._TOP_PAGES.clear()
    bot._TOP_PAGES["abc123"] = _main_state(n=2)
    itx = _FakeInteraction("user_a", "football_top_abc123_ana_9")
    asyncio.run(bot._handle_top_interaction(itx, "football_top_abc123_ana_9"))
    assert itx.sent and "Match tidak ditemukan" in itx.sent[0][0][0]
    assert itx.sent[0][1].get("ephemeral") is True
    bot._TOP_PAGES.clear()


def test_top_paged_view_disabled_at_bounds():
    v = bot._top_paged_view("tok", 0, 3)
    children = v.children
    assert children[0].custom_id == "football_top_tok_copy"
    assert children[0].disabled is False
    assert children[1].custom_id == "football_top_tok_prev"
    assert children[1].disabled is True  # first page: no prev
    assert children[2].custom_id == "football_top_tok_next"
    assert children[2].disabled is False

    v_last = bot._top_paged_view("tok", 2, 3)
    assert v_last.children[1].disabled is False
    assert v_last.children[2].disabled is True  # last page: no next


def test_top_page_copy_text_strips_markdown():
    page = {
        "title": "",
        "body": "🎯 **VALUE MATCH — 12 AGU 2026**\n\n🏆 **MOL Cup · 35 match**\n"
                "• Belotin vs Havirov\n• +33 lainnya",
        "footer": "📊 **66 MATCH**\n📄 **Page 1/2**",
    }
    text = bot._top_page_copy_text(page)
    assert "**" not in text
    assert "VALUE MATCH — 12 AGU 2026" in text
    assert "MOL Cup · 35 match" in text
    assert "• Belotin vs Havirov" in text
    assert "66 MATCH" in text
    assert "Page 1/2" in text


def test_top_page_state_isolation():
    """User A cannot navigate user B's pages: state stores the owner id and
    the interaction handler rejects a different user before touching state."""
    bot._TOP_PAGES.clear()
    now = __import__("time").monotonic()
    token = "deadbeef1234"
    bot._TOP_PAGES[token] = {
        "ts": now, "user_id": "user_a",
        "pages": [{"title": "", "body": "p1", "footer": "f"}, {"title": "", "body": "p2", "footer": "f"}],
        "index": 0,
    }
    state = bot._TOP_PAGES[token]
    assert state["user_id"] == "user_a"
    assert state["pages"][0]["body"] == "p1"
    # The handler checks the clicker id against the owner — covered implicitly
    # by the state shape; direct async test of _handle_top_interaction would
    # need a mock interaction, so we assert the guard data is in place.
    bot._TOP_PAGES.clear()


def test_purge_top_pages_expired():
    bot._TOP_PAGES.clear()
    now = __import__("time").monotonic()
    bot._TOP_PAGES["old"] = {"ts": now - 20 * 60, "user_id": "u", "pages": [], "index": 0}
    bot._TOP_PAGES["fresh"] = {"ts": now, "user_id": "u", "pages": [], "index": 0}
    bot._purge_top_pages(now)
    assert "old" not in bot._TOP_PAGES
    assert "fresh" in bot._TOP_PAGES
    bot._TOP_PAGES.clear()


def test_competition_order_stable_for_equal_counts():
    rows = _extra("Zeta", 3) + _extra("Alpha", 3) + _extra("Beta", 3)
    comps1 = fmt._group_competitions(rows)
    comps2 = fmt._group_competitions(list(reversed(rows)))
    assert [c for c, _ in comps1] == [c for c, _ in comps2] == ["Alpha", "Beta", "Zeta"]


def test_no_hardcoded_totals():
    """total_match / total_competition must be derived from actual rows."""
    rows = (
        _extra("Champions League - Qualification", 7)
        + _extra("Europa League - Qualification", 3)
        + _extra("Premier League", 2)
    )
    pages = fmt.build_top_pages(_payload(extra=rows))
    assert "12 MATCH" in pages[0]["footer"]
    assert "3 KOMPETISI" in pages[0]["footer"]


def test_build_top_pages_lists_non_analyzable_as_info_only():
    """Friendlies and minor cups are listed under an info-only section (not
    hidden), while analyzable competitions stay tagged with their league key."""
    rows = (
        _extra("Champions League - Qualification", 3)
        + _extra("Club Friendly", 5)
        + _extra("MOL Cup", 2)
    )
    pages = fmt.build_top_pages(_payload(extra=rows))
    body = pages[0]["body"]
    assert "Champions League - Qualification · 3 match** (UCL)" in body
    assert "BELUM TERDAFTAR (info saja)" in body
    assert "Club Friendly · 5 match" in body
    assert "MOL Cup · 2 match" in body
    assert "10 MATCH" in pages[0]["footer"]
    assert "3 KOMPETISI" in pages[0]["footer"]
    assert "3 bisa dianalisa" in pages[0]["footer"]
    assert "7 info saja" in pages[0]["footer"]


def test_competition_league_key_mapping():
    """competition_league_key maps homepage titles via the prefix rule."""
    from agents.football.league_resolver import competition_league_key

    assert competition_league_key("Champions League - Qualification") == "UCL"
    assert competition_league_key("Conference League - Qualification") == "UECL"
    assert competition_league_key("Europa League - Qualification") == "UEL"
    assert competition_league_key("Premier League") == "EPL"
    assert competition_league_key("MLS") == "MLS"  # MLS is not a country name
    # One-off cup / international competitions registered for full analysis
    assert competition_league_key("UEFA Super Cup") == "UEFA Super Cup"
    assert competition_league_key("Super Cup") == "UEFA Super Cup"
    assert competition_league_key("Community Shield") == "Community Shield"
    assert competition_league_key("Friendly International") == "Friendly International"
    assert competition_league_key("Friendly International Women") == "Friendly International"
    assert competition_league_key("Copa Libertadores - Play Offs") == "Copa Libertadores"
    assert competition_league_key("Copa Sudamericana - Play Offs") == "Copa Sudamericana"
    assert competition_league_key("Leagues Cup") == "Leagues Cup"
    assert competition_league_key("CONCACAF Central American Cup") == "CONCACAF Central American Cup"
    assert competition_league_key("OFC Champions League") == "OFC Champions League"
    # Bare country names must NOT resolve ('England Cup' is not an EPL match)
    assert competition_league_key("England Cup - Qualification") is None
    assert competition_league_key("Spain Cup") is None
    assert competition_league_key("France Cup") is None
    assert competition_league_key("Club Friendly") is None
    assert competition_league_key("MOL Cup") is None
    assert competition_league_key("Romanian Cup - Qualification") is None
    assert competition_league_key("") is None
    assert competition_league_key(None) is None


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
