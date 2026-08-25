"""Tests for `!analisa <liga> [today|besok|YYYY-MM-DD]` (bot.py batch analyse).

The command runs `top` once to collect the league's matches (main candidates
+ flashscore extra_matches mapped via competition_league_key), then streams
one full analyse subprocess per match (capped at _TOP_BATCH_MAX). A failing
match must not stop the rest.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


class _Sent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.runner_calls: list[list[str]] = []


class _FakeChannel:
    def __init__(self, sent: _Sent) -> None:
        self.sent = sent

    async def send(self, *a, **k):
        self.sent.messages.append(a[0] if a else k.get("content", ""))

    def typing(self):
        class T:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

        return T()


class _FakeMsg:
    def __init__(self, sent: _Sent) -> None:
        self.channel = _FakeChannel(sent)


def _run(coro):
    return asyncio.run(coro)


def _analyse_render(tag: str) -> dict:
    return {"render": {"title": "🔬 Analisa Match", "body": f"**{tag}** • analisa", "footer": " "}}


def _uecl_extra(n: int) -> list[dict]:
    return [
        {"home": f"H{i}", "away": f"A{i}",
         "competition": "Conference League - Qualification",
         "kickoff": f"2{i}:00", "source": "flashscore"}
        for i in range(n)
    ]


def test_batch_analyse_usage_without_args(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []
    async def fake_invoke(args):
        calls.append(args)
        return {}
    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), []))
    assert not calls
    assert "Format:" in "\n".join(sent.messages)


def test_batch_analyse_unknown_league(monkeypatch):
    sent = _Sent()
    async def fake_invoke(args):
        return {}
    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["liga antah berantah", "today"]))
    assert "tidak dikenal" in "\n".join(sent.messages)


def test_batch_analyse_streams_results(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "top":
            return {"raw": {"matches": [], "extra_matches": _uecl_extra(3)}}
        return _analyse_render(f"{args[args.index('--home') + 1]} vs {args[args.index('--away') + 1]}")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["uecl", "today"]))
    analyse_calls = [c for c in calls if c[0] == "analyse"]
    assert len(analyse_calls) == 3
    for c in analyse_calls:
        assert c[c.index("--league") + 1] == "UECL"
    joined = "\n".join(sent.messages)
    assert "Menganalisa 3 match" in joined
    assert "H0 vs A0" in joined and "H2 vs A2" in joined


def test_batch_analyse_filters_other_competitions(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "top":
            return {"raw": {"matches": [], "extra_matches":
                _uecl_extra(2) + [
                    {"home": "X", "away": "Y", "competition": "Club Friendly", "kickoff": "1:00"},
                ]}}
        return _analyse_render("ok")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["uecl"]))
    analyse_calls = [c for c in calls if c[0] == "analyse"]
    assert len(analyse_calls) == 2  # friendly excluded


def test_batch_analyse_error_continues(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []
    analyse_runs = {"n": 0}

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "top":
            return {"raw": {"matches": [], "extra_matches": _uecl_extra(2)}}
        if args[0] == "analyse":
            analyse_runs["n"] += 1
            if analyse_runs["n"] == 1:
                raise RuntimeError("runner down")
        return _analyse_render("H1 vs A1")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["uecl"]))
    analyse_calls = [c for c in calls if c[0] == "analyse"]
    assert len(analyse_calls) == 2  # both attempted despite the first failing
    joined = "\n".join(sent.messages)
    assert "analisa gagal" in joined
    assert "H1 vs A1" in joined


def test_batch_analyse_caps_at_max(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "top":
            return {"raw": {"matches": [], "extra_matches": _uecl_extra(10)}}
        return _analyse_render("ok")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["uecl"]))
    analyse_calls = [c for c in calls if c[0] == "analyse"]
    assert len(analyse_calls) == bot._TOP_BATCH_MAX
    assert "8/10" in "\n".join(sent.messages)


def test_batch_analyse_no_matches(monkeypatch):
    sent = _Sent()
    async def fake_invoke(args):
        return {"raw": {"matches": [], "extra_matches": []}}
    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["uecl"]))
    joined = "\n".join(sent.messages)
    assert "Tidak ada match" in joined


def test_batch_analyse_besok_passes_date(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "top":
            return {"raw": {"matches": [], "extra_matches": _uecl_extra(1)}}
        return _analyse_render("ok")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    _run(bot._handle_batch_analyse(_FakeMsg(sent), ["uecl", "besok"]))
    top_call = next(c for c in calls if c[0] == "top")
    assert top_call[top_call.index("--date") + 1] != "today"
    assert "--leagues" in top_call


def test_dispatch_analisa_prefix_routes(monkeypatch):
    sent = _Sent()
    calls: list[list[str]] = []

    async def fake_invoke(args):
        calls.append(args)
        if args[0] == "top":
            return {"raw": {"matches": [], "extra_matches": _uecl_extra(1)}}
        return _analyse_render("ok")

    monkeypatch.setattr(bot, "_invoke_runner", fake_invoke)
    monkeypatch.setattr(bot, "_is_authorized", lambda m: True)
    msg = _FakeMsg(sent)
    msg.content = "!analisa uecl today"
    msg.author = type("A", (), {"id": "1", "bot": False})()
    _run(bot._handle(msg))
    assert any(c[0] == "top" for c in calls)


if __name__ == "__main__":
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(__import__("pytest").MonkeyPatch())
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
