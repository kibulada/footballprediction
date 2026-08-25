"""Tests for bot.py local proxy auto-detection.

The bot must find and use a local SOCKS/HTTP proxy (Tor, Clash, v2ray, SS...)
fully automatically -- no manual toggle, no env vars -- so ISP-blocked
networks (nowgoal "Trustpositif" block) still get proxy routing. Covers the
SOCKS5/HTTP liveness probes, SOCKS-over-HTTP priority, config-driven ports
and TTLs, the detection cache, and the env vars the runner subprocess gets.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot  # noqa: E402


def _fake_socket(recv_data: bytes = b"\x05\x00", raises: bool = False):
    class FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, *a):
            pass

        def sendall(self, *a):
            pass

        def recv(self, n):
            if raises:
                raise OSError("refused")
            return recv_data

    return FakeSock()


# ---- liveness probes ------------------------------------------------------

def test_socks_alive_true_on_greeting():
    with patch("socket.create_connection", return_value=_fake_socket(b"\x05\x00")):
        assert bot._socks_alive("127.0.0.1", 9050) is True


def test_socks_alive_false_on_wrong_reply():
    # server picked a method we did not offer -> not a usable no-auth proxy
    with patch("socket.create_connection", return_value=_fake_socket(b"\x05\x02")):
        assert bot._socks_alive("127.0.0.1", 9050) is False


def test_socks_alive_false_on_refused():
    with patch("socket.create_connection", return_value=_fake_socket(raises=True)):
        assert bot._socks_alive("127.0.0.1", 9050) is False


def test_http_alive_true_on_http_status_line():
    # real HTTP proxies answer a bare CRLF locally with a 400 status line
    with patch("socket.create_connection", return_value=_fake_socket(b"HTTP/1.1 400 Bad Request\r\n")):
        assert bot._http_alive("127.0.0.1", 7890) is True


def test_http_alive_false_on_garbage():
    with patch("socket.create_connection", return_value=_fake_socket(b"\x05\x00")):
        assert bot._http_alive("127.0.0.1", 7890) is False


# ---- detection ------------------------------------------------------------

def _reset_detect_cache():
    bot._PROXY_DETECT_CACHE["ts"] = 0.0
    bot._PROXY_DETECT_CACHE["value"] = None
    bot._PROXY_DETECT_CONFIG_CACHE["ts"] = 0.0
    bot._PROXY_DETECT_CONFIG_CACHE["value"] = None


def test_detect_proxy_socks_preferred_over_http():
    _reset_detect_cache()
    cfg = {"enabled": True, "socks_ports": [9050, 7891], "http_ports": [7890]}
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", side_effect=[True, False]) as socks, \
         patch.object(bot, "_http_alive", return_value=False) as http:
        found = bot._detect_proxy()
    assert found == {"kind": "socks", "url": "socks5h://127.0.0.1:9050"}
    assert socks.call_count == 1  # first port answered, no further scan
    http.assert_not_called()


def test_detect_proxy_http_when_no_socks():
    _reset_detect_cache()
    cfg = {"enabled": True, "socks_ports": [9050], "http_ports": [7890, 8118]}
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=False), \
         patch.object(bot, "_http_alive", side_effect=[True, False]):
        found = bot._detect_proxy()
    assert found == {"kind": "http", "url": "http://127.0.0.1:7890"}


def test_detect_proxy_none_when_nothing_listens():
    _reset_detect_cache()
    cfg = {"enabled": True, "socks_ports": [9050], "http_ports": [7890]}
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=False), \
         patch.object(bot, "_http_alive", return_value=False):
        assert bot._detect_proxy() is None


def test_detect_proxy_respects_enabled_false():
    _reset_detect_cache()
    cfg = {"enabled": False, "socks_ports": [9050], "http_ports": [7890]}
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=True) as socks, \
         patch.object(bot, "_http_alive", return_value=True) as http:
        assert bot._detect_proxy() is None
    socks.assert_not_called()
    http.assert_not_called()


def test_detect_proxy_cache_skips_rescan_within_ttl():
    _reset_detect_cache()
    cfg = {"enabled": True, "socks_ports": [9050], "http_ports": [],
           "ttl_found_seconds": 300}
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=True) as socks:
        first = bot._detect_proxy()
        second = bot._detect_proxy()
    assert first == second == {"kind": "socks", "url": "socks5h://127.0.0.1:9050"}
    assert socks.call_count == 1  # cached -> no rescan


def test_detect_proxy_rescans_after_ttl():
    _reset_detect_cache()
    cfg = {"enabled": True, "socks_ports": [9050], "http_ports": [],
           "ttl_found_seconds": 0.0}  # expired immediately
    # monotonic clock: t=100 for the first scan, t=100.5 (0.5s later) for the
    # second -- 0.5 > ttl 0.0, so the proxy must be re-detected, not cached
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=True) as socks, \
         patch("bot.time.monotonic", side_effect=[100.0, 100.5]):
        bot._detect_proxy()
        bot._detect_proxy()
    assert socks.call_count == 2


# ---- env vars for the runner subprocess -----------------------------------

def test_proxy_env_vars_socks_sets_all():
    env = bot._proxy_env_vars({"kind": "socks", "url": "socks5h://127.0.0.1:9050"})
    assert env == {
        "SOCCERDATA_PROXY": "socks5h://127.0.0.1:9050",
        "SOCKS_PROXY": "socks5h://127.0.0.1:9050",
        "HTTPS_PROXY": "socks5h://127.0.0.1:9050",
        "HTTP_PROXY": "socks5h://127.0.0.1:9050",  # http:// nowgoal URLs route too
        "ALL_PROXY": "socks5h://127.0.0.1:9050",
    }


def test_proxy_env_vars_http_no_socks_var():
    env = bot._proxy_env_vars({"kind": "http", "url": "http://127.0.0.1:7890"})
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert "SOCKS_PROXY" not in env


# ---- Tor auto-start -------------------------------------------------------

def _tor_cfg(**overrides):
    """Raw proxy_auto_detect config (as in football.json). _tor_config()
    converts these to the internal keys (auto_start, port, ...)."""
    cfg = {
        "tor_auto_start": True,
        "tor_socks_port": 9050,
        "tor_start_timeout_seconds": 120.0,
        "tor_exe_paths": [],
        "tor_idle_shutdown_seconds": 45.0,
    }
    cfg.update(overrides)
    return cfg


def _reset_tor_state():
    bot._TOR_PROC = None
    bot._TOR_EXE_CANDIDATES = []
    _reset_detect_cache()


def test_ensure_tor_running_disabled():
    _reset_tor_state()
    with patch.object(bot, "_proxy_detect_config", return_value=_tor_cfg(tor_auto_start=False)), \
         patch.object(bot, "_socks_alive", return_value=True) as socks:
        result = bot._ensure_tor_running()
    assert result == {"status": "disabled"}
    socks.assert_not_called()


def test_ensure_tor_running_already_running():
    _reset_tor_state()
    with patch.object(bot, "_proxy_detect_config", return_value=_tor_cfg()), \
         patch.object(bot, "_socks_alive", return_value=True) as socks, \
         patch.object(bot.subprocess, "Popen") as popen:
        result = bot._ensure_tor_running()
    assert result == {"status": "running"}
    popen.assert_not_called()  # no launch needed
    socks.assert_called_once()


def test_ensure_tor_running_no_tor_exe():
    _reset_tor_state()
    with patch.object(bot, "_proxy_detect_config", return_value=_tor_cfg()), \
         patch.object(bot, "_socks_alive", return_value=False), \
         patch.object(bot, "_find_tor_exe", return_value=None), \
         patch.object(bot.subprocess, "Popen") as popen:
        result = bot._ensure_tor_running()
    assert result == {"status": "no_tor"}
    popen.assert_not_called()


def test_ensure_tor_running_launches_and_waits_for_socks():
    """tor.exe launched, then the SOCKS port comes up -> started, and the
    proxy detection cache is invalidated so the next detect rescans."""
    _reset_tor_state()
    fake_proc = type("P", (), {"poll": lambda self: None, "pid": 4242})()
    with patch.object(bot, "_proxy_detect_config", return_value=_tor_cfg(tor_start_timeout_seconds=5.0)), \
         patch.object(bot, "_socks_alive", side_effect=[False, True]) as socks, \
         patch.object(bot, "_find_tor_exe", return_value="C:/tor/tor.exe"), \
         patch.object(bot, "_write_torrc") as write_torrc, \
         patch.object(bot.subprocess, "Popen", return_value=fake_proc) as popen:
        result = bot._ensure_tor_running(wait_seconds=5.0)
    assert result == {"status": "started"}
    popen.assert_called_once()
    write_torrc.assert_called_once()
    assert bot._TOR_PROC is fake_proc
    # cache invalidated -> a subsequent detect rescans instead of reusing None
    assert bot._PROXY_DETECT_CACHE["ts"] == 0.0


def test_ensure_tor_running_timeout_returns_launching():
    _reset_tor_state()
    fake_proc = type("P", (), {"poll": lambda self: None, "pid": 4242})()
    with patch.object(bot, "_proxy_detect_config", return_value=_tor_cfg(tor_start_timeout_seconds=1.0)), \
         patch.object(bot, "_socks_alive", return_value=False), \
         patch.object(bot, "_find_tor_exe", return_value="C:/tor/tor.exe"), \
         patch.object(bot, "_write_torrc"), \
         patch.object(bot.subprocess, "Popen", return_value=fake_proc), \
         patch("bot.time.sleep"):
        result = bot._ensure_tor_running(wait_seconds=1.0)
    assert result == {"status": "launching"}


def test_ensure_tor_running_does_not_relaunch_while_proc_alive():
    """If we already launched tor and it is still running, wait for the
    SOCKS port instead of launching a second instance."""
    _reset_tor_state()
    fake_proc = type("P", (), {"poll": lambda self: None, "pid": 4242})()
    bot._TOR_PROC = fake_proc
    with patch.object(bot, "_proxy_detect_config", return_value=_tor_cfg(tor_start_timeout_seconds=5.0)), \
         patch.object(bot, "_socks_alive", side_effect=[False, True]), \
         patch.object(bot, "_find_tor_exe", return_value="C:/tor/tor.exe") as find_exe, \
         patch.object(bot.subprocess, "Popen") as popen:
        result = bot._ensure_tor_running(wait_seconds=5.0)
    assert result == {"status": "started"}
    find_exe.assert_not_called()
    popen.assert_not_called()  # no second launch


def test_write_torrc_contains_port_data_dir_and_geoip():
    _reset_tor_state()
    with patch.object(bot, "_find_tor_exe", return_value="C:/Tor Browser/Browser/TorBrowser/Tor/tor.exe"), \
         patch.object(bot, "_tor_geoip_paths", return_value=("C:/geoip", "C:/geoip6")), \
         patch.object(bot, "_TOR_DIR", bot.ROOT / "cache" / "tor"):
        torrc = bot._write_torrc(9050)
    text = torrc.read_text(encoding="utf-8")
    assert "SocksPort 9050" in text
    assert "DataDirectory" in text
    assert "GeoIPFile C:/geoip" in text
    assert "GeoIPv6File C:/geoip6" in text
    assert "ClientOnly 1" in text


def test_find_tor_exe_uses_config_path_first():
    _reset_tor_state()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "tor.exe"
        fake.write_bytes(b"x")
        cfg = _tor_cfg(tor_exe_paths=[str(fake)])
        with patch.object(bot, "_proxy_detect_config", return_value=cfg):
            assert bot._find_tor_exe() == str(fake)


# ---- Tor shutdown on bot exit --------------------------------------------

def test_shutdown_tor_noop_when_nothing_launched():
    """Never kill a Tor we did not launch (user-provided Tor is left alone)."""
    _reset_tor_state()
    with patch.object(bot.subprocess, "run") as run:
        bot._shutdown_tor()
    run.assert_not_called()


def test_shutdown_tor_terminates_own_proc():
    _reset_tor_state()
    fake_proc = type("P", (), {"pid": 4242, "poll": lambda self: None})()
    bot._TOR_PROC = fake_proc
    with patch.object(bot.subprocess, "run") as run:
        bot._shutdown_tor()
    # Windows path: taskkill /PID <pid> /T /F
    run.assert_called_once()
    args = run.call_args.args[0]
    assert "taskkill" in args and "4242" in args
    assert bot._TOR_PROC is None  # cleared after shutdown


def test_shutdown_tor_noop_when_proc_already_dead():
    _reset_tor_state()
    fake_proc = type("P", (), {"pid": 4242, "poll": lambda self: 0})()
    bot._TOR_PROC = fake_proc
    with patch.object(bot.subprocess, "run") as run:
        bot._shutdown_tor()
    run.assert_not_called()


# ---- Tor health-check -----------------------------------------------------

def test_tor_health_check_running_when_socks_up():
    """Port alive -> healthy, no relaunch, detection cache untouched."""
    _reset_tor_state()
    cfg = _tor_cfg()
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=True) as socks, \
         patch.object(bot, "_ensure_tor_running") as ensure:
        result = bot._tor_health_check()
    assert result == {"status": "running"}
    socks.assert_called_once()
    ensure.assert_not_called()


def test_tor_health_check_relaunches_when_socks_down_in_use():
    """Port dead WHILE IN USE (refcount > 0) -> relaunch + invalidate cache."""
    _reset_tor_state()
    bot._TOR_REFCOUNT = 1
    try:
        bot._PROXY_DETECT_CACHE["ts"] = 123.0
        bot._PROXY_DETECT_CACHE["value"] = {"kind": "socks", "url": "socks5h://127.0.0.1:9050"}
        cfg = _tor_cfg()
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
             patch.object(bot, "_socks_alive", return_value=False), \
             patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}) as ensure:
            result = bot._tor_health_check()
        assert result == {"status": "started"}
        ensure.assert_called_once()
        # stale proxy detection invalidated -> next detect rescans
        assert bot._PROXY_DETECT_CACHE["ts"] == 0.0
        assert bot._PROXY_DETECT_CACHE["value"] is None
    finally:
        bot._TOR_REFCOUNT = 0


def test_tor_health_check_idle_does_not_relaunch():
    """Port dead while IDLE (refcount 0) is the expected on-demand state:
    no relaunch -- the next command's acquire starts Tor again."""
    _reset_tor_state()
    cfg = _tor_cfg()
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive", return_value=False), \
         patch.object(bot, "_ensure_tor_running") as ensure:
        result = bot._tor_health_check()
    assert result == {"status": "idle"}
    ensure.assert_not_called()


def test_tor_health_check_disabled_when_auto_start_off():
    _reset_tor_state()
    cfg = _tor_cfg(tor_auto_start=False)
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_socks_alive") as socks, \
         patch.object(bot, "_ensure_tor_running") as ensure:
        result = bot._tor_health_check()
    assert result == {"status": "disabled"}
    socks.assert_not_called()
    ensure.assert_not_called()


def test_tor_health_loop_skips_when_disabled_in_config():
    """tor_health_check_enabled=false -> loop exits immediately."""
    cfg = _tor_cfg()
    cfg["health_check"] = False
    with patch.object(bot, "_tor_config", return_value=cfg), \
         patch.object(bot, "_tor_health_check") as check:
        asyncio.run(bot._tor_health_loop())
    check.assert_not_called()


# ---- on-demand lifecycle (refcount) --------------------------------------


def _reset_refcount():
    bot._TOR_REFCOUNT = 0
    if bot._TOR_IDLE_TASK is not None:
        bot._TOR_IDLE_TASK.cancel()
        bot._TOR_IDLE_TASK = None


def test_command_needs_tor_excludes_local_modes():
    assert bot._command_needs_tor(["analyse", "--home", "A", "--away", "B"])
    assert bot._command_needs_tor(["top"])
    assert bot._command_needs_tor(["compare", "--home", "A"])
    assert not bot._command_needs_tor(["settle"])
    assert not bot._command_needs_tor(["stats"])
    assert not bot._command_needs_tor(["cache-odds"])
    assert not bot._command_needs_tor(["audit"])
    assert not bot._command_needs_tor(["calib-refresh"])


def test_tor_acquire_first_starts_and_second_shares():
    _reset_refcount()
    cfg = _tor_cfg()
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}) as ensure:
        r1 = bot._tor_acquire()
        assert r1 == {"status": "started"}
        assert bot._TOR_REFCOUNT == 1
        ensure.assert_called_once()
        # second holder shares the running instance -- no relaunch
        r2 = bot._tor_acquire()
        assert r2 == {"status": "running"}
        assert bot._TOR_REFCOUNT == 2
        ensure.assert_called_once()
    bot._TOR_REFCOUNT = 0


def test_tor_acquire_disabled_when_auto_start_off():
    _reset_refcount()
    cfg = _tor_cfg(tor_auto_start=False)
    with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
         patch.object(bot, "_ensure_tor_running") as ensure:
        result = bot._tor_acquire()
    assert result == {"status": "disabled"}
    ensure.assert_not_called()
    assert bot._TOR_REFCOUNT == 0


def test_tor_release_schedules_idle_shutdown_at_zero():
    """Release at refcount 0 schedules a delayed shutdown; a re-acquire before
    the timer fires cancels it (warm reuse)."""
    _reset_refcount()
    cfg = _tor_cfg(tor_idle_shutdown_seconds=0.05)

    async def scenario():
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
             patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}), \
             patch.object(bot, "_shutdown_tor") as shutdown:
            bot._tor_acquire()
            bot._tor_release()
            assert bot._TOR_REFCOUNT == 0
            assert bot._TOR_IDLE_TASK is not None and not bot._TOR_IDLE_TASK.done()
            # quick re-acquire cancels the pending shutdown -> Tor stays up
            bot._tor_acquire()
            assert bot._TOR_REFCOUNT == 1
            await asyncio.sleep(0.12)  # longer than the 0.05s timer that was cancelled
            shutdown.assert_not_called()
            bot._tor_release()  # clean up: now the idle timer is re-armed
            await asyncio.sleep(0.12)
            shutdown.assert_called_once()
        bot._TOR_REFCOUNT = 0
        bot._TOR_IDLE_TASK = None

    asyncio.run(scenario())


def test_tor_release_runs_shutdown_after_idle():
    """No re-acquire -> the idle timer actually stops Tor."""
    _reset_refcount()
    cfg = _tor_cfg(tor_idle_shutdown_seconds=0.05)

    async def scenario():
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
             patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}), \
             patch.object(bot, "_shutdown_tor") as shutdown:
            bot._tor_acquire()
            bot._tor_release()
            await asyncio.sleep(0.15)
            shutdown.assert_called_once()
        _reset_refcount()

    asyncio.run(scenario())


def test_tor_release_keeps_tor_while_holders_remain():
    _reset_refcount()
    cfg = _tor_cfg(tor_idle_shutdown_seconds=0.05)

    async def scenario():
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
             patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}), \
             patch.object(bot, "_shutdown_tor") as shutdown:
            bot._tor_acquire()
            bot._tor_acquire()
            bot._tor_release()  # still 1 holder -> no shutdown scheduled
            assert bot._TOR_IDLE_TASK is None or bot._TOR_IDLE_TASK.done() is False
            await asyncio.sleep(0.1)
            shutdown.assert_not_called()
            bot._tor_release()  # 0 holders -> idle timer now schedules shutdown
            await asyncio.sleep(0.12)
            shutdown.assert_called_once()
        _reset_refcount()

    asyncio.run(scenario())


def test_tor_release_disabled_noop():
    _reset_refcount()
    cfg = _tor_cfg(tor_auto_start=False)

    async def scenario():
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
             patch.object(bot, "_shutdown_tor") as shutdown:
            bot._tor_release()
        shutdown.assert_not_called()

    asyncio.run(scenario())


def test_tor_release_waits_for_cold_bootstrap_before_shutdown():
    """Cold-start guard: a freshly launched Tor that is still bootstrapping
    (SOCKS down) must NOT be killed by the idle timer -- the idle task waits
    for the port to come up (bounded by the start timeout) and only then
    stops it, so a cold-started instance is never killed before it was
    usable."""
    _reset_refcount()
    cfg = _tor_cfg(tor_idle_shutdown_seconds=0.01, tor_start_timeout_seconds=5.0, tor_idle_poll_seconds=0.01)

    async def scenario():
        # SOCKS stays down for the first 12 polls (cold bootstrap), then
        # comes up: the idle task must keep waiting the whole time the
        # port is down, and only shut down after it became usable.
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
                 patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}), \
                 patch.object(bot, "_tor_process_alive", side_effect=[True] * 15 + [False]), \
                 patch.object(bot, "_socks_alive", side_effect=[False] * 12 + [True, True, False]), \
                 patch.object(bot, "_shutdown_tor") as shutdown:
                bot._tor_acquire()
                bot._tor_release()
                # while SOCKS is down (12 polls) the idle task must keep
                # waiting, so after the grace + a few polls elapsed no
                # shutdown has happened yet (Windows timer granularity makes
                # each 0.01s poll take ~0.02s, so 12 polls ~0.24s)
                await asyncio.sleep(0.12)
                shutdown.assert_not_called()
                # once the port comes up (side_effect flips to True) and no
                # holder remains, the shutdown runs
                await asyncio.sleep(0.3)
                shutdown.assert_called_once()
        _reset_refcount()

    asyncio.run(scenario())


def test_tor_release_gives_up_after_boot_timeout():
    """Cold-start guard is bounded: if the port never comes up within the
    start timeout, the idle task stops waiting and shuts Tor down anyway."""
    _reset_refcount()
    cfg = _tor_cfg(tor_idle_shutdown_seconds=0.01, tor_start_timeout_seconds=0.05, tor_idle_poll_seconds=0.01)

    async def scenario():
        with patch.object(bot, "_proxy_detect_config", return_value=cfg), \
             patch.object(bot, "_ensure_tor_running", return_value={"status": "started"}), \
             patch.object(bot, "_tor_process_alive", return_value=True), \
             patch.object(bot, "_socks_alive", return_value=False), \
             patch.object(bot, "_shutdown_tor") as shutdown:
            bot._tor_acquire()
            bot._tor_release()
            await asyncio.sleep(0.2)  # > grace + boot timeout
            shutdown.assert_called_once()
        _reset_refcount()

    asyncio.run(scenario())


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
