"""Tests for the auto-settle background task in bot.py.

The bot must record finished match results into the prediction log WITHOUT a
manual `!football settle`: a background loop settles the last N days on an
interval (config auto_settle.*), and `!football stats` catches up first so
statistics are always fresh.
"""
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


def _wib_today() -> str:
    WIB = timezone(timedelta(hours=7))
    return datetime.now(WIB).date().isoformat()


def test_auto_settle_dates_defaults_to_two_days(monkeypatch):
    monkeypatch.setattr(bot, "_auto_settle_cfg", lambda: {})
    dates = asyncio.run(bot._auto_settle_dates())
    WIB = timezone(timedelta(hours=7))
    assert len(dates) == 2
    assert dates[0] == _wib_today()
    assert dates[1] == (datetime.now(WIB) - timedelta(days=1)).date().isoformat()


def test_auto_settle_dates_respects_days_back(monkeypatch):
    monkeypatch.setattr(bot, "_auto_settle_cfg", lambda: {"days_back": 1})
    dates = asyncio.run(bot._auto_settle_dates())
    assert dates == [_wib_today()]


def test_run_auto_settle_once_settles_every_date(monkeypatch):
    calls = []

    async def fake_runner(args):
        calls.append(list(args))
        return {"render": {}, "raw": {"status": "auto", "settled": [{"home": "A"}]}}

    monkeypatch.setattr(bot, "_auto_settle_cfg", lambda: {})
    monkeypatch.setattr(bot, "_invoke_runner", fake_runner)

    summary = asyncio.run(bot._run_auto_settle_once())
    assert summary["settled_total"] == 2
    # one settle invocation per date, each carrying the settle-auto args
    settle_calls = [c for c in calls if c[:2] == ["settle", "auto"]]
    assert len(settle_calls) == 2
    for call in settle_calls:
        assert call[2] == "--date"
        assert call[3] in summary["dates"]
    # Phase 0.4/5.4: the auto-settle tick also schedules the daily CLV
    # segment report and the edge-bucket-vs-closing audit once per day.
    modes = [c[0] for c in calls]
    assert "clv-report" in modes
    assert "bucket-audit" in modes


def test_run_auto_settle_once_survives_runner_error(monkeypatch):
    async def fake_runner(args):
        return {"error": "football-data rate limit"}

    monkeypatch.setattr(bot, "_auto_settle_cfg", lambda: {})
    monkeypatch.setattr(bot, "_invoke_runner", fake_runner)

    summary = asyncio.run(bot._run_auto_settle_once())
    assert summary["settled_total"] == 0


def test_auto_settle_loop_exits_when_disabled(monkeypatch):
    slept = []

    async def never_sleep(seconds):
        slept.append(seconds)
        raise AssertionError("loop must not sleep when disabled")

    monkeypatch.setattr(bot, "_auto_settle_enabled", lambda: False)
    monkeypatch.setattr(asyncio, "sleep", never_sleep)

    asyncio.run(bot._auto_settle_loop())
    assert slept == []


def test_auto_settle_enabled_defaults_true(monkeypatch):
    # config section absent -> enabled (the feature is on by default)
    monkeypatch.setattr(bot, "_auto_settle_cfg", lambda: {})
    assert bot._auto_settle_enabled() is True


def test_auto_settle_enabled_flag_off(monkeypatch):
    real_read = Path.read_text

    def fake_read(self, **k):
        if "football.json" in str(self):
            return json.dumps({"auto_settle": {"enabled": False}})
        return real_read(self, **k)

    monkeypatch.setattr(Path, "read_text", fake_read)
    assert bot._auto_settle_enabled() is False


def test_auto_settle_cfg_reads_real_config():
    # the repo config ships auto_settle.enabled: true
    cfg = bot._auto_settle_cfg()
    assert cfg.get("enabled") is True
    assert cfg.get("interval_hours", 6) >= 1


def test_stats_handler_pre_settles(monkeypatch):
    """`!football stats` catches up results before computing stats."""
    calls = []

    async def fake_runner(args):
        calls.append(list(args))
        if args[0] == "settle":
            return {"raw": {"settled": [{"home": "A"}]}}
        return {"render": {"title": "📈 Prediction Log Stats", "body": "stats", "footer": " "}}

    class _Ch:
        def __init__(self):
            self.sent = []

        async def send(self, *a, **k):
            self.sent.append((a, k))

        def typing(self):
            class T:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    pass

            return T()

    class _Msg:
        channel = _Ch()

    monkeypatch.setattr(bot, "_auto_settle_cfg", lambda: {})
    monkeypatch.setattr(bot, "_invoke_runner", fake_runner)

    msg = _Msg()
    asyncio.run(bot._handle_stats(msg, []))
    # settle ran first (once per date), then the stats command
    assert calls[-1] == ["stats"]
    assert any(c[:2] == ["settle", "auto"] for c in calls)


if __name__ == "__main__":
    import inspect

    import pytest

    mp = pytest.MonkeyPatch()
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            if len(inspect.signature(fn).parameters):
                fn(mp)
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        finally:
            mp.undo()  # global patches (asyncio.sleep, Path.read_text) must not leak
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
