"""Discord bot for Hermes Football.

Spawns the CLI runner as subprocess per request, parses JSON stdout,
posts a Discord embed. Hard filter: user_id + channel name.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import discord
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

TOKEN = os.getenv("DISCORD_FOOTBALL_TOKEN")
ALLOWED_USER_ID = os.getenv("FOOTBALL_ALLOWED_USER_ID")
DEFAULT_CHANNEL = os.getenv("FOOTBALL_DEFAULT_CHANNEL", "football-picks")
# Runner subprocess kill ceiling: must stay above the runner's own hard
# deadline (HERMES_RUNNER_DEADLINE, default 340s) so the runner always exits
# first with a clean JSON error instead of being killed with a generic one.
# Discord interactions allow 15 minutes; a cold-cache run on a slow network
# can legitimately take several minutes.
SUBPROCESS_TIMEOUT = 380

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("hermes-football-bot")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def _runner_path() -> str:
    # Cross-platform: Windows .venv\Scripts\python.exe, Linux .venv/bin/python
    if sys.platform == "win32":
        return str(ROOT / ".venv" / "Scripts" / "python.exe")
    p = ROOT / ".venv" / "bin" / "python"
    if p.exists():
        return str(p)
    return sys.executable


def _cleanup_browser_zombies() -> None:
    """Kill leftover seleniumbase Chrome instances (bot-spawned only).

    The flashscore/sofascore/understat browser fallbacks launch a Chrome with
    a temp user-data-dir (``AppData\\Local\\Temp\\tmp*``) and the sofascore
    fallback adds ``--host-resolver-rules`` network isolation. The runner
    exits via ``os._exit``, and Windows does not reap child processes on
    parent death, so that Chrome survives as an orphan. A normal browsing
    Chrome never carries a temp user-data-dir or ``--host-resolver-rules``,
    so the filter is safe.
    """
    if sys.platform != "win32":
        return
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'chrome.exe' -and ("
        "$_.CommandLine -like '*--host-resolver-rules*' -or "
        "$_.CommandLine -like '*--user-data-dir=*Temp\\tmp*') } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
        "-ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            timeout=20,
        )
    except Exception:
        pass


# ---- local proxy auto-detection ------------------------------------------
#
# The bot routes upstream HTTP/HTTPS through a local proxy when one answers
# (the user's ISP blocks nowgoal/win007 etc. with the "Trustpositif" page;
# a local SOCKS/HTTP proxy such as Tor, Clash, v2ray or Shadowsocks -- set
# to auto-start with Windows -- is the escape hatch). Detection is fully
# automatic: common ports are probed, the FIRST working proxy wins, and the
# result is cached (found: 5 min, none: 1 min) so every command does not
# rescan dead ports. No config, no manual toggle, no env vars needed from
# the user; the bot simply uses whatever proxy is available.

_PROXY_DETECT_TTL_FOUND = 300.0   # proxy up: trust the detection for 5 min
_PROXY_DETECT_TTL_NONE = 60.0     # none found: rescan every minute
_PROXY_DETECT_CACHE: dict[str, Any] = {"ts": 0.0, "value": None}
_PROXY_DETECT_CONFIG_CACHE: dict[str, Any] = {"ts": 0.0, "value": None}


def _proxy_detect_config() -> dict[str, Any]:
    """Read config/football.json -> proxy_auto_detect (60s TTL cache)."""
    now = time.monotonic()
    if now - _PROXY_DETECT_CONFIG_CACHE["ts"] > 60.0:
        cfg: dict[str, Any] = {"enabled": True}
        try:
            data = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
            cfg.update(data.get("proxy_auto_detect") or {})
        except Exception:
            pass
        _PROXY_DETECT_CONFIG_CACHE["ts"] = now
        _PROXY_DETECT_CONFIG_CACHE["value"] = cfg
    return _PROXY_DETECT_CONFIG_CACHE["value"]


def _socks_alive(host: str, port: int, timeout: float = 2.0) -> bool:
    """True iff a WORKING SOCKS5 proxy answers on host:port.

    A bare TCP connect only proves that *something* listens; a proxy can be
    bound but not actually routing (half-open / firewalled port), which used
    to make the runner hang in SYN_SENT while every HTTP client retried its
    connect timeout. A real SOCKS5 greeting (no-auth method) proves routing.
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(b"\x05\x01\x00")  # SOCKS5, 1 method, no-auth
            buf = b""
            while len(buf) < 2:
                chunk = s.recv(2 - len(buf))
                if not chunk:
                    return False
                buf += chunk
            return buf == b"\x05\x00"
    except OSError:
        return False


def _http_alive(host: str, port: int, timeout: float = 2.0) -> bool:
    """True iff an HTTP proxy answers on host:port (no upstream traffic).

    Sends a bare CRLF pair: a real HTTP proxy (Clash, v2ray http, privoxy)
    answers locally with an HTTP status line (usually 400 Bad Request) without
    forwarding anything upstream.
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(b"\r\n\r\n")
            buf = b""
            while b"HTTP/" not in buf and len(buf) < 256:
                chunk = s.recv(64)
                if not chunk:
                    return False
                buf += chunk
            return b"HTTP/" in buf
    except OSError:
        return False


def _detect_proxy() -> dict[str, Any] | None:
    """Find a working local proxy (SOCKS preferred, then HTTP).

    Returns {"kind": "socks"|"http", "url": "socks5h://127.0.0.1:PORT" |
    "http://127.0.0.1:PORT"} or None. Cached: a found proxy is trusted for
    _PROXY_DETECT_TTL_FOUND, a miss rescans after _PROXY_DETECT_TTL_NONE.
    """
    now = time.monotonic()
    cached = _PROXY_DETECT_CACHE
    cfg = _proxy_detect_config()
    ttl_found = float(cfg.get("ttl_found_seconds", _PROXY_DETECT_TTL_FOUND))
    ttl_none = float(cfg.get("ttl_none_seconds", _PROXY_DETECT_TTL_NONE))
    ttl = ttl_found if cached["value"] is not None else ttl_none
    if now - cached["ts"] <= ttl:
        return cached["value"]

    found: dict[str, Any] | None = None
    if cfg.get("enabled", True):
        for port in cfg.get("socks_ports") or []:
            if _socks_alive("127.0.0.1", int(port)):
                found = {"kind": "socks", "url": f"socks5h://127.0.0.1:{port}"}
                break
        if found is None:
            for port in cfg.get("http_ports") or []:
                if _http_alive("127.0.0.1", int(port)):
                    found = {"kind": "http", "url": f"http://127.0.0.1:{port}"}
                    break
    _PROXY_DETECT_CACHE["ts"] = now
    _PROXY_DETECT_CACHE["value"] = found
    if found:
        logger.info("auto-detected %s proxy at %s", found["kind"], found["url"])
    return found


def _proxy_env_vars(proxy: dict[str, Any]) -> dict[str, str]:
    """Env vars that route upstream HTTP/HTTPS through the proxy.

    HTTP_PROXY/ALL_PROXY are set too -- the runner's httpx clients read
    trust_env, and without HTTP_PROXY an http:// URL (nowgoal uses http)
    would never go through the proxy.
    """
    url = proxy["url"]
    env = {
        "SOCCERDATA_PROXY": url,
        "HTTPS_PROXY": url,
        "HTTP_PROXY": url,
        "ALL_PROXY": url,
    }
    if proxy.get("kind") == "socks":
        env["SOCKS_PROXY"] = url
    return env


# ---- Tor auto-start ------------------------------------------------------
#
# The bot routes upstream HTTP/HTTPS through Tor when it is running (see
# _detect_proxy). If Tor is NOT running but the config asks for auto-start,
# the bot launches its own tor.exe (background, no window) with a torrc it
# generates into cache/tor/, waits for the SOCKS port to come up, and then
# the normal proxy detection finds it. Fully automatic: no manual launch,
# no scheduled task -- starting the bot is enough.

_TOR_DIR = ROOT / "cache" / "tor"
_TORRC_PATH = _TOR_DIR / "torrc"
_TOR_LOG_PATH = _TOR_DIR / "tor.log"
_TOR_PID_PATH = _TOR_DIR / "tor.pid"
_TOR_PROC: subprocess.Popen[Any] | None = None

_TOR_EXE_CANDIDATES: list[str] = []  # lazily filled by _find_tor_exe


def _tor_config() -> dict[str, Any]:
    """proxy_auto_detect.* keys relevant to Tor auto-start + health-check."""
    cfg = _proxy_detect_config()
    return {
        "auto_start": bool(cfg.get("tor_auto_start", False)),
        "port": int(cfg.get("tor_socks_port", 9050)),
        "timeout": float(cfg.get("tor_start_timeout_seconds", 120.0)),
        "exe_paths": [str(p) for p in (cfg.get("tor_exe_paths") or []) if str(p).strip()],
        "health_check": bool(cfg.get("tor_health_check_enabled", True)),
        "health_interval": float(cfg.get("tor_health_check_interval_seconds", 60.0)),
        "idle_shutdown_seconds": float(cfg.get("tor_idle_shutdown_seconds", 45.0)),
        "idle_poll_seconds": float(cfg.get("tor_idle_poll_seconds", 3.0)),
    }


def _find_tor_exe() -> str | None:
    """Locate tor.exe: explicit config paths, TOR_EXE_PATH, PATH, then the
    usual Tor Browser install spots (Desktop / Downloads / Program Files)."""
    import shutil

    global _TOR_EXE_CANDIDATES
    if not _TOR_EXE_CANDIDATES:
        cands: list[str] = list(_tor_config()["exe_paths"])
        env_exe = os.getenv("TOR_EXE_PATH")
        if env_exe:
            cands.append(env_exe)
        which = shutil.which("tor")
        if which:
            cands.append(which)
        home = Path.home()
        for folder in ("Downloads", "Desktop", "OneDrive/Desktop"):
            base = home / folder
            if not base.exists():
                continue
            for pattern in (
                "Tor Browser/Browser/TorBrowser/Tor/tor.exe",
                "TorBrowser*/Browser/TorBrowser/Tor/tor.exe",
                "tor*/tor.exe",
            ):
                cands.extend(str(p) for p in base.glob(pattern))
        for folder in (Path(os.environ.get("ProgramFiles", "C:/Program Files")),):
            cands.extend(
                str(p)
                for p in folder.glob("Tor Browser/Browser/TorBrowser/Tor/tor.exe")
            )
        # de-dup, keep order
        seen: set[str] = set()
        _TOR_EXE_CANDIDATES = [
            c for c in cands if not (c in seen or seen.add(c))
        ]
    for cand in _TOR_EXE_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def _tor_geoip_paths(exe: str) -> tuple[str | None, str | None]:
    """GeoIP files live next to the Tor Browser install
    (Browser/TorBrowser/Data/Tor/geoip[6]); tor.exe is at
    Browser/TorBrowser/Tor/tor.exe."""
    exe_path = Path(exe)
    data_tor = exe_path.parents[1] / "Data" / "Tor"
    geoip = data_tor / "geoip"
    geoip6 = data_tor / "geoip6"
    return (
        str(geoip) if geoip.is_file() else None,
        str(geoip6) if geoip6.is_file() else None,
    )


def _write_torrc(port: int) -> Path:
    """Generate cache/tor/torrc for this machine (data dir + log inside the
    project so nothing outside ROOT is touched)."""
    _TOR_DIR.mkdir(parents=True, exist_ok=True)
    exe = _find_tor_exe()
    geoip, geoip6 = _tor_geoip_paths(exe) if exe else (None, None)
    lines = [
        "# Generated by the bot (auto-start) -- edit proxy_auto_detect in",
        "# config/football.json instead of this file.",
        f"SocksPort {port}",
        f"DataDirectory {_TOR_DIR / 'data'}",
        "ClientOnly 1",
        f"Log notice file {_TOR_LOG_PATH}",
    ]
    if geoip:
        lines.append(f"GeoIPFile {geoip}")
    if geoip6:
        lines.append(f"GeoIPv6File {geoip6}")
    _TORRC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _TORRC_PATH


def _tor_process_alive() -> bool:
    """True iff the tor.exe we launched is still running."""
    proc = _TOR_PROC
    return proc is not None and proc.poll() is None


def _ensure_tor_running(wait_seconds: float | None = None) -> dict[str, Any]:
    """Launch Tor if needed and wait for its SOCKS port (blocking).

    Returns {"status": ...} where status is one of:
      running  -- SOCKS already answers (our launch or a user-provided Tor)
      started  -- we launched tor.exe and its SOCKS port came up
      launching -- we launched but the port did not come up in time
      no_tor  -- auto-start enabled but no tor.exe found anywhere
      disabled -- tor_auto_start is off in config
    Invalidate the proxy-detection cache on success so _detect_proxy rescans.
    """
    global _TOR_PROC
    tcfg = _tor_config()
    if not tcfg["auto_start"]:
        return {"status": "disabled"}
    host, port = "127.0.0.1", tcfg["port"]
    if _socks_alive(host, port):
        _PROXY_DETECT_CACHE["ts"] = 0.0
        _PROXY_DETECT_CACHE["value"] = None
        return {"status": "running"}
    if _tor_process_alive():
        # We launched it; just wait for bootstrap.
        pass
    else:
        exe = _find_tor_exe()
        if exe is None:
            logger.warning(
                "tor_auto_start enabled but no tor.exe found "
                "(set TOR_EXE_PATH or proxy_auto_detect.tor_exe_paths)"
            )
            return {"status": "no_tor"}
        _write_torrc(port)
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW
        try:
            _TOR_PROC = subprocess.Popen(
                [exe, "-f", str(_TORRC_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
            _TOR_PID_PATH.write_text(str(_TOR_PROC.pid), encoding="utf-8")
            logger.info("auto-started tor.exe (pid %s) with %s", _TOR_PROC.pid, _TORRC_PATH)
        except OSError as exc:
            logger.warning("tor auto-start failed to launch %s: %s", exe, exc)
            return {"status": "no_tor"}

    wait = wait_seconds if wait_seconds is not None else tcfg["timeout"]
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _socks_alive(host, port):
            _PROXY_DETECT_CACHE["ts"] = 0.0
            _PROXY_DETECT_CACHE["value"] = None
            return {"status": "started"}
        time.sleep(1.0)
    logger.warning("tor SOCKS port %s not up after %.0fs", port, wait)
    return {"status": "launching"}


# ---- Tor health-check ----------------------------------------------------
#
# Periodic loop: if the auto-started tor.exe dies mid-run (crash, kill,
# laptop suspend), the SOCKS port stops answering and every provider would
# silently fall back to direct -- re-blocking nowgoal. The loop re-checks
# every tor_health_check_interval_seconds and relaunches via
# _ensure_tor_running, so the escape hatch heals itself without a bot
# restart.


def _tor_health_check() -> dict[str, Any]:
    """One health probe: is the configured Tor SOCKS port up?

    Returns {"status": ...} like _ensure_tor_running. When the port is down
    and auto-start is enabled AND Tor is currently in use (refcount > 0 --
    i.e. a command holds the proxy), relaunches Tor (waits briefly -- the
    full bootstrap happens in the background). When nothing is using Tor
    (refcount 0) the port being down is the EXPECTED on-demand state, so no
    relaunch happens -- the next command's acquire starts it again. Also
    invalidates the proxy detection cache on any transition so _detect_proxy
    rescans instead of reusing a stale result.
    """
    tcfg = _tor_config()
    if not tcfg["auto_start"]:
        return {"status": "disabled"}
    host, port = "127.0.0.1", tcfg["port"]
    if _socks_alive(host, port):
        return {"status": "running"}
    _PROXY_DETECT_CACHE["ts"] = 0.0
    _PROXY_DETECT_CACHE["value"] = None
    if _TOR_REFCOUNT <= 0:
        # On-demand lifecycle: idle Tor is supposed to be down. The next
        # acquire starts it; do not fight the design.
        return {"status": "idle"}
    logger.warning("tor SOCKS port %s down while in use -- relaunching via health check", port)
    return _ensure_tor_running(wait_seconds=8.0)


async def _tor_health_loop() -> None:
    """Background task: probe Tor SOCKS periodically and relaunch on death."""
    tcfg = _tor_config()
    if not tcfg["health_check"]:
        return
    while True:
        try:
            await asyncio.sleep(tcfg["health_interval"])
            result = await asyncio.to_thread(_tor_health_check)
            status = result.get("status") if isinstance(result, dict) else str(result)
            if status == "started":
                logger.info("tor health check: relaunched (status=%s)", status)
            elif status == "running":
                pass
            elif status in ("no_tor", "disabled"):
                pass
            else:
                logger.warning("tor health check: SOCKS not up yet (status=%s)", status)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("tor health check failed")


# ---- on-demand Tor lifecycle (refcount) ---------------------------------
#
# Tor is NOT kept running for the bot's lifetime. It is started lazily the
# first time a command needs the nowgoal/proxy escape hatch, and stopped a
# short while AFTER the last user of it finishes (idle grace). The refcount
# lets concurrent commands share one Tor instance: the first acquirer starts
# it, every acquirer releases it, and only when the count drops to zero does
# the idle timer schedule the shutdown. This keeps the ISP-block escape
# hatch available exactly when needed without a permanently-on local proxy.

_TOR_REFCOUNT = 0
_TOR_IDLE_TASK: "asyncio.Task[Any] | None" = None

# Modes that only touch local state (prediction log / caches / backtest) and
# never need the proxy: acquiring Tor for them would cold-start a process
# just to shut it down again.
_TOR_LOCAL_ONLY_MODES = {"settle", "stats", "calib-refresh", "audit", "cache-odds"}


def _command_needs_tor(args: list[str]) -> bool:
    """True iff a runner invocation should hold a Tor reference."""
    mode = args[0].lower() if args else ""
    return mode not in _TOR_LOCAL_ONLY_MODES


def _tor_acquire(wait_seconds: float = 6.0) -> dict[str, Any]:
    """Take a Tor reference; start Tor when this is the first holder.

    The first acquisition (0 -> 1) launches tor.exe if it is not already
    answering on its SOCKS port and cancels any pending idle shutdown.
    Subsequent acquisitions are no-ops (the instance is shared). Returns the
    ``_ensure_tor_running`` status dict ("running" for a shared instance).
    """
    global _TOR_REFCOUNT, _TOR_IDLE_TASK
    tcfg = _tor_config()
    if not tcfg["auto_start"]:
        return {"status": "disabled"}
    _TOR_REFCOUNT += 1
    if _TOR_REFCOUNT == 1:
        if _TOR_IDLE_TASK is not None and not _TOR_IDLE_TASK.done():
            _TOR_IDLE_TASK.cancel()
            _TOR_IDLE_TASK = None
        return _ensure_tor_running(wait_seconds=wait_seconds)
    return {"status": "running"}


def _tor_release() -> None:
    """Drop a Tor reference; schedule shutdown when the count hits zero.

    The shutdown is deferred by ``tor_idle_shutdown_seconds`` so a quick
    follow-up command (e.g. a batch of analyses) reuses the warm instance
    instead of paying a 30-120s cold bootstrap each time. Only our OWN
    launched tor.exe is stopped; a user-provided Tor is left alone.

    Cold-start guard: a freshly launched tor.exe needs ~30-120s to finish
    its first bootstrap, which can exceed the idle grace. The idle task
    therefore ALSO waits for the SOCKS port to come up (bounded by
    ``tor_start_timeout_seconds``) before shutting down -- so a command
    that cold-started Tor never kills it before it was ever usable, and as
    soon as it IS usable with no holder, it is stopped again.
    """
    global _TOR_REFCOUNT, _TOR_IDLE_TASK
    tcfg = _tor_config()
    if not tcfg["auto_start"]:
        return
    _TOR_REFCOUNT = max(0, _TOR_REFCOUNT - 1)
    if _TOR_REFCOUNT > 0:
        return
    idle = float(tcfg.get("idle_shutdown_seconds", 45.0))
    host, port = "127.0.0.1", tcfg["port"]
    boot_timeout = float(tcfg.get("timeout", 120.0))
    poll = float(tcfg.get("idle_poll_seconds", 3.0))

    async def _idle_shutdown() -> None:
        try:
            await asyncio.sleep(idle)
        except asyncio.CancelledError:
            return  # a new acquisition cancelled us -- Tor stays up
        if _TOR_REFCOUNT != 0:
            return
        # Cold-start guard: wait for our own launch to finish bootstrapping
        # (SOCKS answers) before stopping it. A user-provided Tor is never
        # touched by _shutdown_tor anyway.
        deadline = time.monotonic() + boot_timeout
        while _tor_process_alive() and not _socks_alive(host, port):
            if time.monotonic() >= deadline:
                break
            try:
                await asyncio.sleep(poll)
            except asyncio.CancelledError:
                return
            if _TOR_REFCOUNT != 0:
                return
        if _TOR_REFCOUNT == 0:
            await asyncio.to_thread(_shutdown_tor)

    if _TOR_IDLE_TASK is not None and not _TOR_IDLE_TASK.done():
        _TOR_IDLE_TASK.cancel()
    _TOR_IDLE_TASK = asyncio.create_task(_idle_shutdown())


def _shutdown_tor() -> None:
    """Stop the tor.exe the bot auto-started (only its OWN instance).

    Runs on exit: registered via atexit + SIGTERM handler + main() finally.
    A Tor that was already running before the bot (never launched by us --
    ``_TOR_PROC`` is only set for our own launch) is left alone, so the bot
    never kills a proxy the user started manually.
    """
    global _TOR_PROC
    proc = _TOR_PROC
    _TOR_PROC = None
    if proc is None or proc.poll() is not None:
        return
    logger.info("shutting down auto-started tor.exe (pid %s)", proc.pid)
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    except Exception:
        logger.exception("error stopping auto-started tor")
    try:
        _TOR_PID_PATH.unlink(missing_ok=True)
    except OSError:
        pass


async def _invoke_runner(args: list[str]) -> dict[str, Any]:
    cmd = [_runner_path(), "-m", "agents.football.runner", *args]
    proc_env = os.environ.copy()
    proc_env["PYTHONUNBUFFERED"] = "1"
    proc_env["PYTHONIOENCODING"] = "utf-8"

    # On-demand Tor + auto-detect a local proxy (Tor/Clash/v2ray/SS...): if
    # one answers and the user hasn't explicitly set a proxy, route all
    # upstream HTTP/HTTPS through it so ISP-blocked networks can still reach
    # nowgoal/Sofascore/FBref. Fully automatic -- no manual toggle. Tor is
    # acquired per command (refcount): the first acquirer starts tor.exe, and
    # after the last acquirer finishes, an idle timer stops it again, so the
    # escape hatch is up exactly when a provider needs it and down otherwise.
    needs_tor = _command_needs_tor(args) and not any(
        k in proc_env for k in ("SOCCERDATA_PROXY", "SOCKS_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
    )
    if needs_tor:
        # Launch tor.exe (background, no window) and wait a short grace
        # period for its SOCKS port. Cold bootstrap can take ~30-120s, so
        # this only waits briefly per command; the full bootstrap continues
        # in the background and the next command picks the proxy up via
        # _detect_proxy. _tor_release() in the finally below schedules the
        # shutdown once the command is done.
        try:
            _tor_acquire(wait_seconds=6.0)
        except Exception:
            logger.exception("tor acquire failed")
    try:
        return await _invoke_runner_subprocess(cmd, proc_env)
    finally:
        if needs_tor:
            try:
                _tor_release()
            except Exception:
                logger.exception("tor release failed")


async def _invoke_runner_subprocess(cmd: list[str], proc_env: dict[str, str]) -> dict[str, Any]:
    """Spawn the runner subprocess, capture its JSON payload, clean up."""
    # Auto-detect a local proxy (Tor/Clash/v2ray/SS...): if one answers (the
    # caller already acquired Tor when needed), route all upstream HTTP/HTTPS
    # through it so ISP-blocked networks can still reach nowgoal/Sofascore/
    # FBref. Fully automatic -- no manual toggle, the bot uses whatever proxy
    # is running (see _detect_proxy).
    proxy = _detect_proxy()
    if proxy is not None:
        proc_env.update(_proxy_env_vars(proxy))
        logger.info("routing upstream via %s proxy %s", proxy["kind"], proxy["url"])

    marker_start = "RJSON_START "
    marker_end = " RJSON_END"

    # The runner's payload (and its deadline error) goes to stderr. Capture it
    # in a temp FILE instead of a pipe:
    #   1. A stray Chrome child holding the pipe's write end could block
    #      communicate() forever; a file cannot be held open, so the reply can
    #      never be lost to a hung orphan.
     #      2. wait_for(communicate()) cancellation pops bytes out of the asyncio
     #      StreamReader buffer into the cancelled task's local and they are
     #      lost. A file keeps every byte, so the runner's own
     #      "deadline terlampaui" JSON always survives even when we hit our
     #      380s timeout first and taskkill it.
    stderr_fd, stderr_path = tempfile.mkstemp(prefix="hermes_runner_", suffix=".log")
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=stderr_fd,
            env=proc_env,
        )
        os.close(stderr_fd)  # child has its own duplicated handle now
        stderr_fd = -1
        try:
            await asyncio.wait_for(proc.wait(), timeout=SUBPROCESS_TIMEOUT)
        except asyncio.TimeoutError:
            # Kill the whole process tree (Windows): the runner may have
            # spawned a headless Chrome (sofascore fallback). Run taskkill off
            # the event loop so the bot stays responsive while it executes.
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
        text = Path(stderr_path).read_text(encoding="utf-8", errors="replace").strip()
    finally:
        if stderr_fd >= 0:
            try:
                os.close(stderr_fd)
            except OSError:
                pass
        try:
            os.unlink(stderr_path)
        except OSError:
            pass

    # The runner may have left a seleniumbase Chrome behind (os._exit skips
    # driver cleanup); sweep bot-spawned Chrome orphans so they can't pile
    # up into zombies that hang on a dead Tor proxy.
    try:
        await asyncio.to_thread(_cleanup_browser_zombies)
    except Exception:
        pass

    if marker_start in text and marker_end in text:
        try:
            start = text.index(marker_start) + len(marker_start)
            end = text.index(marker_end, start)
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

    if proc is not None and proc.returncode != 0:
        return {"error": f"runner exit {proc.returncode}: {text[:2000]}"}

    return {"error": "runner output bukan JSON", "raw": text[:2000]}


def _is_authorized(message: discord.Message) -> bool:
    if not ALLOWED_USER_ID:
        return False
    if str(message.author.id) != ALLOWED_USER_ID:
        return False
    if message.channel.name != DEFAULT_CHANNEL:
        return False
    return True


def _parse_command(content: str) -> list[str] | None:
    content = content.strip()
    if not content.startswith("!football"):
        return None
    rest = content[len("!football"):].strip()
    if not rest:
        return None
    return shlex.split(rest, posix=False)


# ---- LLM intent router ----------------------------------------------------
#
# Free-form messages that do not match a rule-based pattern are sent to an
# OpenAI-compatible LLM endpoint (9router) which maps them onto ONE of the
# existing commands. The router is a pure parser: it only produces
# (handler, args); every number the user sees still comes from the validated
# prediction pipeline, never from the LLM.


HELP_TEXT = (
    "**⚽ Hermes Football — Daftar Perintah**\n\n"
    "`!football today [today|besok|YYYY-MM-DD] [--leagues a,b] [--top-n N]`\n"
    "   Value match hari ini + dini hari (ranked, tombol ⚡ analyse per match)\n"
    "`!best <liga>` — 1 prediction terbaik dari match hari ini + dini hari\n"
    "`!bestgoalmatch [liga]` — match paling banjir gol hari ini (expected goals)\n"
    "`analisa match <liga> <home> vs <away>` — analisa match spesifik\n"
    "   (liga boleh dikosongkan: bot auto-detect dari jadwal + Flashscore)\n"
    "`!livescore <liga> <home> vs <away>` — cari match via LiveScore (hari ini/besok) lalu analisa penuh\n"
    "`!flashscore <liga> <home> vs <away>` — cari match via Flashscore (hari ini/besok) lalu analisa penuh\n"
    "`!analisa <liga> [today|besok|YYYY-MM-DD]` — analisa semua match liga itu hari itu (streaming)\n"
    "`!football compare <HOME> <AWAY> [league]` — bandingkan 2 tim\n"
    "`!football settle <home> vs <away> <2-1>` — catat hasil match manual\n"
    "`!football settle auto [YYYY-MM-DD]` — sinkron hasil selesai otomatis\n"
    "`!football stats [edge%]` — statistik realisasi prediksi (hit rate, ROI, CLV)\n"
    "`!football odds <T-24h|T-6h|T-1h|T-15m> <home> vs <away> <h,d,a> [league]`\n"
    "   Snapshot odds 1X2 untuk evaluasi CLV\n"
    "`!football update elo` — update ELO ratings dari elofootball.com\n"
    "`!football clv` — CLV (Closing Line Value) dashboard\n"
    "`!football steam [hours]` — steam move alerts (default 24h)\n\n"
    "📝 Kamu juga bisa ngetik bebas, misal: *\"besok ada match apa di ucl?\"*"
)


# Map an LLM intent onto the exact handler + args the rule-based path uses,
# so router and manual commands share one validated implementation.
_HANDLERS: dict[str, str] = {
    "top": "_handle_top",
    "compare": "_handle_compare",
    "analyse": "_handle_analyse",
    "stats": "_handle_stats",
    "settle": "_handle_settle",
    "odds": "_handle_odds_snapshot",
    "best": "_handle_best",
    "bestgoalmatch": "_handle_best_goal",
    "livescore": "_handle_source_match",
    "flashscore": "_handle_source_match",
    "clv": "_handle_clv",
    "steam": "_handle_steam",
}


_LLM_FLAG_TTL = 60.0
_LLM_FLAG_CACHE: dict[str, Any] = {"ts": 0.0, "value": True}


def _llm_router_enabled() -> bool:
    """Respect config/football.json feature flag enable_llm_router (60s cache)."""
    now = time.monotonic()
    if now - _LLM_FLAG_CACHE["ts"] > _LLM_FLAG_TTL:
        enabled = True
        try:
            cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
            enabled = bool((cfg.get("feature_flags") or {}).get("enable_llm_router", True))
        except Exception:
            pass
        _LLM_FLAG_CACHE["ts"] = now
        _LLM_FLAG_CACHE["value"] = enabled
    return _LLM_FLAG_CACHE["value"]


async def _handle_llm(message: discord.Message, raw: str) -> bool:
    """Route a free-form message through the LLM intent router.

    Returns True if a command was dispatched (or the message was intentionally
    ignored, e.g. chit-chat), False if the LLM produced nothing usable.
    """
    from agents.football.llm_router import is_configured, route_intent

    # Cheap env check first (no file IO); feature flag only when a real
    # endpoint is configured.
    if not is_configured() or not _llm_router_enabled():
        return False  # silent: keep old rule-based-only behaviour
    async with message.channel.typing():
        intent = await route_intent(raw)
    if not intent:
        return False

    command = intent.get("command")
    params = intent.get("params") or {}

    if command == "none":
        return True  # not a football request -> ignore politely
    if command == "help":
        note = params.get("note")
        await message.channel.send((f"{note}\n\n" if note else "") + HELP_TEXT)
        return True

    handler_name = _HANDLERS.get(command)
    if not handler_name:
        return False

    action = _intent_to_action(command, params)
    if action is None:
        await message.channel.send(
            "Mohon lengkapi informasinya. " + HELP_TEXT
        )
        return True
    handler_name, args = action
    handler = globals().get(handler_name)
    if handler is None:
        return False
    if handler_name == "_handle_analyse":
        await handler(message, args[0])
    elif handler_name == "_handle_source_match":
        # args = [source, "<liga> <home> vs <away>"] (see _intent_to_action).
        await handler(message, args[0], args[1])
    else:
        await handler(message, args)
    return True


def _intent_to_action(command: str, params: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Convert an LLM intent to (handler_name, args). None = missing data."""
    if command == "top":
        args: list[str] = []
        date = params.get("date")
        if date in ("besok", "tomorrow"):
            args.append("besok")
        elif date in ("today", "hari ini", "hariini"):
            args.append("today")
        elif isinstance(date, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            args.append(date)
        leagues = params.get("leagues")
        if leagues:
            if isinstance(leagues, list):
                leagues = ",".join(str(x) for x in leagues)
            args += ["--leagues", str(leagues)]
        top_n = params.get("top_n")
        if top_n:
            args += ["--top-n", str(top_n)]
        return "_handle_top", args
    if command == "compare":
        home, away = params.get("home"), params.get("away")
        if not home or not away:
            return None
        args = [str(home), str(away)]
        if params.get("league"):
            args.append(str(params["league"]))
        return "_handle_compare", args
    if command == "analyse":
        league, home, away = params.get("league"), params.get("home"), params.get("away")
        if not home or not away:
            return None
        # League optional: without it the handler runs the runner-side
        # auto-detect (football-data -> thesportsdb -> flashscore homepage).
        if league:
            return "_handle_analyse", [f"{league} {home} vs {away}"]
        return "_handle_analyse", [f"{home} vs {away}"]
    if command == "stats":
        return "_handle_stats", []
    if command == "settle":
        if params.get("auto"):
            return "_handle_settle", ["auto"]
        home, away, result = params.get("home"), params.get("away"), params.get("result")
        if not home or not away or not result:
            return None
        return "_handle_settle", [str(home), "vs", str(away), str(result)]
    if command == "odds":
        timing = params.get("timing")
        home, away, odds = params.get("home"), params.get("away"), params.get("odds")
        if not timing or not home or not away or not odds:
            return None
        return "_handle_odds_snapshot", [str(timing), str(home), "vs", str(away), str(odds)]
    if command == "best":
        league = params.get("league")
        if not league:
            return None
        return "_handle_best", [str(league)]
    if command == "bestgoalmatch":
        args = []
        if params.get("league"):
            args.append(str(params["league"]))
        return "_handle_best_goal", args
    if command in ("livescore", "flashscore"):
        league, home, away = params.get("league"), params.get("home"), params.get("away")
        if not league or not home or not away:
            return None
        # Same string form the rule-based `!<command> <liga> <home> vs <away>`
        # path feeds _handle_source_match.
        return "_handle_source_match", [command, f"{league} {home} vs {away}"]
    return None


# ---- Copy button (📋) ----------------------------------------------------
#
# Discord buttons cannot write to a user's clipboard directly. The standard
# pattern: clicking the button makes the bot reply with an *ephemeral*
# message (visible only to the clicker) showing the FULL report as plain
# text (markdown rendered) — same style as the main output. Copy it with
# right-click -> Copy Text (desktop) or long-press -> Copy (mobile).

COPY_TTL_SECONDS = 15 * 60  # Discord component buttons expire after 15 min
_MAX_COPY_ENTRIES = 100
_COPY_TEXTS: dict[str, tuple[float, dict[str, Any]]] = {}

# Copy state is persisted to disk so buttons keep working across bot
# restarts (otherwise every pre-restart button answers "kedaluwarsa").
# Timestamps are WALL-CLOCK (time.time) because time.monotonic() resets to
# ~0 on boot, which would corrupt the TTL across restarts. Override the path
# via HERMES_COPY_STATE (tests point it at a temp file).
_COPY_STATE_PATH = Path(os.getenv("HERMES_COPY_STATE", "")) if os.getenv("HERMES_COPY_STATE") else ROOT / "cache" / "football" / "copy_buttons.json"

# Gap between ephemeral copy followups. Discord's burst cap is 5 msgs / 5 s
# per webhook; with a 1.0s gap a 6+ chunk report lands 6 messages inside one
# 5-second window and the tail can be dropped to a 429. >=1.25s keeps every
# 5-second window at <=5 messages no matter how long the report is.
_COPY_CHUNK_GAP = 1.25


async def _send_ephemeral_chunks(
    interaction: discord.Interaction,
    chunks: list[str],
    *,
    header: str | None = None,
    wrap_code: bool = False,
) -> None:
    """Send copy content as ONE ephemeral response + spaced followups.

    Every copy path (analyse 📋 Copy, top ranked-list 📋 Copy, paginated
    📋 Salin) funnels through here so the followup spacing is consistent and
    the burst can never exceed Discord's 5 msgs / 5 s cap — previously the
    paginated path sent its chunks back-to-back with no delay and could drop
    the tail of long pages.

    ``header`` replaces the first chunk in the response line (the page path
    answers with a status message and sends ALL chunks as followups);
    ``wrap_code`` wraps every chunk in a ``` code fence (page copy text).
    """
    first = header if header is not None else chunks[0]
    await interaction.response.send_message(first, ephemeral=True)
    rest = chunks if header is not None else chunks[1:]
    for chunk in rest:
        await asyncio.sleep(_COPY_CHUNK_GAP)
        await interaction.followup.send(
            content=f"```\n{chunk}\n```" if wrap_code else chunk,
            ephemeral=True,
        )


def _purge_copy_texts(now: float) -> None:
    expired = [k for k, (ts, _) in _COPY_TEXTS.items() if now - ts > COPY_TTL_SECONDS]
    changed = bool(expired)
    for k in expired:
        _COPY_TEXTS.pop(k, None)
    while len(_COPY_TEXTS) > _MAX_COPY_ENTRIES:
        oldest = min(_COPY_TEXTS, key=lambda k: _COPY_TEXTS[k][0])
        _COPY_TEXTS.pop(oldest, None)
        changed = True
    if changed:
        _save_copy_texts()


def _save_copy_texts() -> None:
    """Persist the copy store atomically (never a torn file on crash)."""
    try:
        data = {
            cid: {"ts": ts, "rendered": rendered}
            for cid, (ts, rendered) in _COPY_TEXTS.items()
        }
        _COPY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _COPY_STATE_PATH.with_name(_COPY_STATE_PATH.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, _COPY_STATE_PATH)
    except OSError as exc:
        logger.warning("copy state save failed (buttons still work in-memory): %s", exc)


def _load_copy_texts() -> None:
    """Load persisted copy state at startup so buttons survive restarts."""
    if not _COPY_STATE_PATH.exists():
        return
    try:
        data = json.loads(_COPY_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("copy state load failed (starting empty): %s", exc)
        return
    if not isinstance(data, dict):
        return
    for cid, entry in data.items():
        if not isinstance(cid, str) or not isinstance(entry, dict):
            continue
        ts = entry.get("ts")
        rendered = entry.get("rendered")
        if not isinstance(ts, (int, float)) or not isinstance(rendered, dict):
            continue
        _COPY_TEXTS[cid] = (float(ts), rendered)
    _purge_copy_texts(time.time())


# Startup: pick up persisted buttons (expired ones are dropped on load).
_load_copy_texts()


def _split_body_chunks(body: str, limit: int = 3700) -> list[str]:
    """Split report text on line boundaries so markdown stays readable.

    Discord plain messages are capped at 2000 chars; callers pass their own
    limit (1900 for replies, leaving headroom for a long title/footer). Chunk
    boundaries land on newlines so ``**bold**`` and inline-code spans are
    never cut mid-token.
    """
    if len(body) <= limit:
        return [body] if body else [""]
    chunks: list[str] = []
    while len(body) > limit:
        cut = body.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(body[:cut].rstrip())
        body = body[cut:].lstrip("\n")
    chunks.append(body)
    return chunks


# The Copy button serves the ENTIRE report (bounded only by _MAX_COPY_BODY),
# so unlike the main reply it is not capped at _MAX_PLAIN_MESSAGES.
_MAX_COPY_BODY = 12000  # ~6 chunks @1900, keeps the ephemeral burst rate-safe


def _copy_plain_messages(rendered: dict[str, Any]) -> list[str]:
    """Full report as ephemeral plain-text chunks (markdown intact).

    Same plain-text style as the main output; the only difference is there is
    no message cap — the copy button is the escape hatch for reports longer
    than what the main reply shows.
    """
    return _plain_messages(rendered, max_messages=None, max_body=_MAX_COPY_BODY)


def _copy_payload(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Full report for the 📋 Copy button, when the main reply is compact.

    The analyse command renders a 5-7 line summary as the main reply and
    carries the FULL report under ``render_full``; the Copy button serves
    that instead of the compact summary. Every other command has no
    ``render_full`` and falls back to the main render unchanged.
    """
    if isinstance(result, dict):
        full = result.get("render_full")
        if isinstance(full, dict) and (full.get("body") or full.get("title")):
            return full
    return payload


def _copy_view(rendered: dict[str, Any]) -> discord.ui.View:
    """One button that serves the full report as ephemeral embed(s)."""
    now = time.time()
    _purge_copy_texts(now)
    custom_id = f"football_copy_{secrets.token_hex(6)}"
    _COPY_TEXTS[custom_id] = (now, rendered)
    _save_copy_texts()
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label="📋 Detail",
            style=discord.ButtonStyle.secondary,
            custom_id=custom_id,
        )
    )
    return view


# ---- Paginated "top" (KOMPETISI LAIN) -------------------------------------
#
# When the primary league filter has no matches but Flashscore lists other
# competitions (200+ matches), the reply is paginated: 8-12 competitions per
# page with [📋 Salin] [◀️] [▶️] buttons. Page flips and the copy button only
# re-render the already-fetched pages — never re-run the runner or hit any
# data source again. State lives in memory, keyed by a random token, and is
# bound to the owner user so user A cannot flip user B's pages.

_TOP_PAGE_TTL = 15 * 60  # Discord buttons expire after 15 min anyway
_MAX_TOP_PAGES = 50

# `!analisa <liga>` batch cap: each match runs the full analyse engine as its
# own runner subprocess (~15-40s), so an unbounded batch would keep the user
# waiting many minutes. Results stream in, then the rest point to `analisa match`.
_TOP_BATCH_MAX = 8

# `analisa match <home> vs <away>` retry cap: when the user typed a token that
# looks like a league keyword but it failed to resolve (e.g. "asean thailand
# vs singapore" -- "asean" is not a registered league), drop the leftmost
# token and retry detect up to this many times. The full-left candidate is
# always tried first, so existing behaviour for legitimate user input is
# unchanged (no regression).
_DETECT_DROP_MAX = 3
_TOP_PAGES: dict[str, dict[str, Any]] = {}


def _purge_top_pages(now: float) -> None:
    expired = [k for k, v in _TOP_PAGES.items() if now - v["ts"] > _TOP_PAGE_TTL]
    for k in expired:
        _TOP_PAGES.pop(k, None)
    while len(_TOP_PAGES) > _MAX_TOP_PAGES:
        oldest = min(_TOP_PAGES, key=lambda k: _TOP_PAGES[k]["ts"])
        _TOP_PAGES.pop(oldest, None)


def _top_paged_view(token: str, index: int, total: int) -> discord.ui.View:
    """Buttons for one page: [📋 Salin] [◀️] [▶️], nav disabled at bounds."""
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label="📋 Detail",
            style=discord.ButtonStyle.secondary,
            custom_id=f"football_top_{token}_copy",
        )
    )
    view.add_item(
        discord.ui.Button(
            label="◀️",
            style=discord.ButtonStyle.secondary,
            custom_id=f"football_top_{token}_prev",
            disabled=index <= 0,
        )
    )
    view.add_item(
        discord.ui.Button(
            label="▶️",
            style=discord.ButtonStyle.secondary,
            custom_id=f"football_top_{token}_next",
            disabled=index >= total - 1,
        )
    )
    return view


def _top_main_view(token: str, n_matches: int) -> discord.ui.View:
    """One ⚡ analyse button per ranked match + the 📋 Copy button.

    top_n is capped at 5 (config), so one row holds every ⚡ button (⚡ N
    matches the match number in the list) and a second row the copy button.
    Clicking ⚡ runs the full analyse engine for that match without retyping
    the command. Note: with `--top-n > 5` only the first 5 matches get a
    button (Discord allows max 5 buttons per row); the list still shows all
    of them.
    """
    view = discord.ui.View(timeout=None)
    for i in range(min(n_matches, 5)):
        view.add_item(
            discord.ui.Button(
                label=f"⚡ {i + 1}",
                style=discord.ButtonStyle.primary,
                custom_id=f"football_top_{token}_ana_{i}",
            )
        )
    view.add_item(
        discord.ui.Button(
            label="📋 Detail",
            style=discord.ButtonStyle.secondary,
            custom_id=f"football_top_{token}_copy",
        )
    )
    return view


def _top_page_copy_text(page: dict[str, Any]) -> str:
    """Plain-text version of ONE page (easy to copy, markdown stripped)."""
    parts = [
        (page.get("title") or "").strip(),
        (page.get("body") or "").strip(),
        (page.get("footer") or "").strip(),
    ]
    text = "\n\n".join(p for p in parts if p and p != " ")
    # strip **bold** and `code` markers so the text pastes cleanly
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


async def _post_paginated_top(message: discord.Message, payload: dict[str, Any]) -> None:
    """Post page 1 of a paginated top report with nav + copy buttons.

    The pages were already rendered by the runner; no re-query happens when
    the user flips pages or presses Salin.
    """
    pages = payload.get("pages") or []
    if not pages:
        await _post_result(message, {"render": payload})
        return
    now = time.monotonic()
    _purge_top_pages(now)
    token = secrets.token_hex(6)
    _TOP_PAGES[token] = {
        "ts": now,
        "user_id": str(message.author.id),
        "pages": pages,
        "index": 0,
    }
    chunks = _plain_messages(pages[0])
    await message.channel.send(content=chunks[0], view=_top_paged_view(token, 0, len(pages)))
    for chunk in chunks[1:]:
        await asyncio.sleep(0.35)
        await message.channel.send(content=chunk)


def _parse_top_custom_id(custom_id: str) -> tuple[str, str] | None:
    """'football_top_<token>_copy|prev|next|ana_<i>' -> (token, action)."""
    m = re.match(r"^football_top_([0-9a-f]+)_(copy|prev|next|ana_\d+)$", custom_id)
    if not m:
        return None
    return m.group(1), m.group(2)


async def _handle_top_interaction(interaction: discord.Interaction, custom_id: str) -> None:
    parsed = _parse_top_custom_id(custom_id)
    if parsed is None:
        return
    token, action = parsed
    _purge_top_pages(time.monotonic())
    state = _TOP_PAGES.get(token)
    if state is None:
        await interaction.response.send_message(
            "Sesi halaman sudah kedaluwarsa (button aktif 15 menit) — "
            "jalankan ulang `!football today`.",
            ephemeral=True,
        )
        return
    if str(interaction.user.id) != state["user_id"]:
        await interaction.response.send_message(
            "Tombol ini hanya untuk pemilik perintah.", ephemeral=True
        )
        return

    # ---- Ranked match list (kind == "main"): ⚡ analyse buttons -----------
    if state.get("kind") == "main":
        rendered = state.get("rendered") or {}
        if action == "copy":
            await _send_ephemeral_chunks(interaction, _copy_plain_messages(rendered))
            return
        if action.startswith("ana_"):
            try:
                idx = int(action[4:])
            except ValueError:
                await interaction.response.send_message(
                    "Perintah tidak dikenali.", ephemeral=True
                )
                return
            entries: list[dict[str, Any]] = state.get("matches") or []
            if not (0 <= idx < len(entries)):
                await interaction.response.send_message(
                    "Match tidak ditemukan — jalankan ulang `!football today`.",
                    ephemeral=True,
                )
                return
            m = entries[idx]
            # Analyse can take minutes on a cold cache (runner deadline
            # HERMES_RUNNER_DEADLINE, default 340s): defer first, then post
            # via followup (Discord allows 15 minutes for followups).
            await interaction.response.defer()
            result = await _invoke_runner(
                ["analyse", "--league", m["league_key"],
                 "--home", m["home"], "--away", m["away"]]
            )
            await _post_followup_result(interaction, result)
            return
        await interaction.response.send_message("Aksi tidak dikenali.", ephemeral=True)
        return

    pages: list[dict[str, Any]] = state["pages"]
    index: int = state["index"]
    total = len(pages)

    if action == "copy":
        text = _top_page_copy_text(pages[index])
        await _send_ephemeral_chunks(
            interaction,
            _split_body_chunks(text, limit=_PLAIN_CHUNK),
            header=f"✅ Data siap disalin — halaman {index + 1}/{total}:",
            wrap_code=True,
        )
        return

    if action == "prev" and index > 0:
        index -= 1
    elif action == "next" and index < total - 1:
        index += 1
    state["index"] = index
    chunks = _plain_messages(pages[index])
    # Update the SAME message — no duplicate replies when flipping pages.
    await interaction.response.edit_message(
        content=chunks[0],
        view=_top_paged_view(token, index, total),
    )
    for chunk in chunks[1:]:
        await interaction.followup.send(content=chunk, ephemeral=True)


async def _handle(message: discord.Message) -> None:
    if not _is_authorized(message):
        return

    raw = message.content.strip()
    lower = raw.lower()
    if lower.startswith("analisa"):
        rest = raw[len("analisa"):].strip()
        if lower.startswith("analisa match"):
            rest = raw[len("analisa match"):].strip()
        await _handle_analyse(message, rest)
        return

    # `!livescore <liga> <home> vs <away>` / `!flashscore ...`: find the match
    # on the named source (today -> tomorrow) then run the SAME analyse
    # pipeline (NowGoal odds + prediction engine + existing output format).
    if lower == "!livescore" or lower.startswith("!livescore "):
        rest = raw[len("!livescore"):].strip()
        await _handle_source_match(message, "livescore", rest)
        return
    if lower == "!flashscore" or lower.startswith("!flashscore "):
        rest = raw[len("!flashscore"):].strip()
        await _handle_source_match(message, "flashscore", rest)
        return

    # Bare `!best <liga>` / `!bestgoalmatch [liga]` prefixes (user-facing
    # format) — the `!football best ...` form is handled by the dispatch below.
    if lower.startswith("!bestgoalmatch") or lower.startswith("!bestgoal"):
        rest = raw[len("!bestgoalmatch") if lower.startswith("!bestgoalmatch") else len("!bestgoal"):].strip()
        await _handle_best_goal(message, shlex.split(rest, posix=False) if rest else [])
        return
    if lower == "!best" or lower.startswith("!best "):
        rest = raw[len("!best"):].strip()
        await _handle_best(message, shlex.split(rest, posix=False) if rest else [])
        return

    # `!analisa <liga> [today|besok|YYYY-MM-DD]` — batch: analyse every match
    # of one league on that date (results stream in one by one).
    if lower == "!analisa" or lower.startswith("!analisa "):
        rest = raw[len("!analisa"):].strip()
        await _handle_batch_analyse(message, shlex.split(rest, posix=False))
        return

    args = _parse_command(raw)
    if args is None:
        # Free-form message (no "!football" prefix): try the LLM intent router.
        handled = await _handle_llm(message, raw)
        if handled:
            return
        # Router disabled / unusable -> keep old silent behaviour.
        return

    mode = args[0].lower()
    if mode in ("top", "today"):
        await _handle_top(message, args[1:])
    elif mode == "compare":
        await _handle_compare(message, args[1:])
    elif mode == "best":
        await _handle_best(message, args[1:])
    elif mode in ("bestgoal", "bestgoalmatch"):
        await _handle_best_goal(message, args[1:])
    elif mode == "settle":
        await _handle_settle(message, args[1:])
    elif mode == "stats":
        await _handle_stats(message, args[1:])
    elif mode in ("odds", "odds-snapshot"):
        await _handle_odds_snapshot(message, args[1:])
    elif mode == "update":
        await _handle_update(message, args[1:])

    else:
        await message.channel.send(HELP_TEXT)


def _result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Render dict from a runner result, including the error fallback."""
    if result.get("error"):
        return result.get("render") or {
            "title": "Hermes Football",
            "body": f"Error: {result['error']}",
            "footer": " ",
        }
    return result.get("render") or {}


def _signal_card_accordion(
    result: dict[str, Any],
) -> tuple[discord.Embed, discord.ui.View] | None:
    """Accordion (summary embed + toggle view) for an analyse MATCH SIGNAL
    card, or None when the result is not a signal card.

    Detection: the runner's primary render for analyse/livescore/flashscore
    is the ``🔬 MATCH SIGNAL`` card (``format_market_signal``) and the raw
    payload carries ``signal_engine``. Both embeds (summary + expanded) are
    built from that SAME raw payload by the accordion module — never a
    second data path — so the two views cannot drift apart. Non-signal
    renders (errors, kickoff-uncertain, finished matches, other commands)
    fall through to the plain-text path unchanged.
    """
    payload = _result_payload(result)
    if (payload.get("title") or "") != "🔬 MATCH SIGNAL":
        return None
    raw = result.get("raw")
    if not isinstance(raw, dict) or not raw.get("signal_engine"):
        return None
    from agents.football.discord_signal_card_accordion import (
        SignalCardView,
        build_summary_embed,
    )

    view = SignalCardView(raw)
    return build_summary_embed(raw), view


async def _post_followup_result(interaction: discord.Interaction, result: dict[str, Any]) -> None:
    """Post an analyse result via the interaction's followup channel.

    Called after ``interaction.response.defer()`` so the long runner call can
    finish before the report lands, attached under the top message the user
    clicked.
    """
    card = _signal_card_accordion(result)
    if card is not None:
        embed, view = card
        sent = await interaction.followup.send(embed=embed, view=view)
        view.attach_message(sent)
        return
    payload = _result_payload(result)
    chunks = _plain_messages(payload)
    await interaction.followup.send(
        content=chunks[0],
        view=_copy_view(_copy_payload(result, payload)),
    )
    for chunk in chunks[1:]:
        await asyncio.sleep(0.35)
        await interaction.followup.send(content=chunk)


async def _post_top_result(message: discord.Message, result: dict[str, Any]) -> None:
    """Post a top result; the ranked match list gets one ⚡ analyse button per
    match (state lives in _TOP_PAGES so the click never re-queries the list)."""
    payload = _result_payload(result)
    if payload.get("pages"):
        await _post_paginated_top(message, payload)
        return
    raw = result.get("raw") or {}
    entries: list[dict[str, Any]] = []
    for m in raw.get("matches") or []:
        if m.get("league_key") and m.get("home") and m.get("away"):
            entries.append({
                "league_key": m["league_key"],
                "home": m["home"],
                "away": m["away"],
            })
    if entries:
        now = time.monotonic()
        _purge_top_pages(now)
        token = secrets.token_hex(6)
        _TOP_PAGES[token] = {
            "ts": now,
            "user_id": str(message.author.id),
            "kind": "main",
            "rendered": payload,
            "matches": entries,
        }
        await _post_result(message, result, view=_top_main_view(token, len(entries)))
        return
    await _post_result(message, result)


async def _handle_top(message: discord.Message, args: list[str]) -> None:
    runner_args: list[str] = ["top"]
    leagues: list[str] = []
    top_n: int | None = None
    date: str | None = None

    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--top-n" and i + 1 < len(args):
            try:
                top_n = int(args[i + 1])
            except ValueError:
                await message.channel.send("`--top-n` harus angka")
                return
            i += 2
        elif tok == "--leagues" and i + 1 < len(args):
            leagues = [x.strip() for x in args[i + 1].split(",") if x.strip()]
            i += 2
        elif tok in ("besok", "tomorrow"):
            from datetime import datetime, timedelta, timezone
            WIB = timezone(timedelta(hours=7))
            date = (datetime.now(WIB) + timedelta(days=1)).date().isoformat()
            i += 1
        elif tok in ("today", "hari-ini", "hariini"):
            from datetime import datetime, timedelta, timezone
            WIB = timezone(timedelta(hours=7))
            date = datetime.now(WIB).date().isoformat()
            i += 1
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", tok):
            date = tok
            i += 1
        else:
            i += 1

    if date is None and "besok" not in args and "tomorrow" not in args:
        date = None
    runner_args.append("--top-n")
    runner_args.append(str(top_n if top_n is not None else 5))
    if date:
        runner_args.extend(["--date", date])
    if leagues:
        runner_args.extend(["--leagues", ",".join(leagues)])

    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_top_result(message, result)


async def _handle_batch_analyse(message: discord.Message, args: list[str]) -> None:
    """`!analisa <liga> [today|besok|YYYY-MM-DD]` — analyse EVERY match of one
    league on that date; results stream in one by one (capped at
    _TOP_BATCH_MAX because each match runs the full engine as its own runner)."""
    if not args:
        await message.channel.send(
            "Format: `!analisa <liga> [today|besok|YYYY-MM-DD]`. "
            "Contoh: `!analisa uecl today`, `!analisa ucl besok`."
        )
        return
    league_query = args[0]

    from datetime import datetime, timedelta, timezone

    WIB = timezone(timedelta(hours=7))
    date: str | None = None
    for tok in args[1:]:
        if tok in ("besok", "tomorrow"):
            date = (datetime.now(WIB) + timedelta(days=1)).date().isoformat()
        elif tok in ("today", "hari-ini", "hariini"):
            date = datetime.now(WIB).date().isoformat()
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", tok):
            date = tok
    if date is None:
        date = datetime.now(WIB).date().isoformat()

    from agents.football.league_resolver import competition_league_key, resolve_league

    resolved = resolve_league(league_query)
    if not resolved:
        await message.channel.send(
            f"Liga `{league_query}` tidak dikenal.\n"
            f"Contoh: `!analisa uecl today`, `!analisa epl besok`, "
            f"`!analisa copa libertadores 2026-08-16`."
        )
        return
    league_key, meta = resolved
    display = meta["display"]

    async with message.channel.typing():
        result = await _invoke_runner(
            ["top", "--date", date, "--leagues", league_key, "--top-n", "5"]
        )
    raw = result.get("raw") or {}

    matches: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in raw.get("matches") or []:
        if m.get("league_key") != league_key or not m.get("home") or not m.get("away"):
            continue
        pair = (m["home"], m["away"])
        if pair not in seen:
            seen.add(pair)
            matches.append(pair)
    for m in raw.get("extra_matches") or []:
        if competition_league_key(str(m.get("competition") or "")) != league_key:
            continue
        h, a = m.get("home"), m.get("away")
        if not h or not a:
            continue
        pair = (h, a)
        if pair not in seen:
            seen.add(pair)
            matches.append(pair)

    if not matches:
        await message.channel.send(
            f"Tidak ada match `{display}` pada {date} yang bisa dianalisa."
        )
        return

    total = len(matches)
    if total > _TOP_BATCH_MAX:
        await message.channel.send(
            f"⚡ Menganalisa {_TOP_BATCH_MAX}/{total} match `{display}` ({date}) — "
            f"{total - _TOP_BATCH_MAX} sisanya ketik `analisa match ...` satu-satu."
        )
    else:
        await message.channel.send(
            f"⚡ Menganalisa {total} match `{display}` ({date}) — hasil menyusul satu-satu."
        )

    for i, (home, away) in enumerate(matches[:_TOP_BATCH_MAX], 1):
        try:
            async with message.channel.typing():
                m_result = await _invoke_runner(
                    ["analyse", "--league", league_key, "--home", home, "--away", away]
                )
        except Exception:
            await message.channel.send(
                f"⚠️ {i}/{total} {home} vs {away}: analisa gagal."
            )
            continue
        await _post_result(message, m_result)


async def _handle_best(message: discord.Message, args: list[str]) -> None:
    """`!best <liga>`: dari semua match hari ini + dini hari di liga tersebut,
    pilih 1 prediction paling akurat/valid untuk betting (engine independen)."""
    if not args:
        await message.channel.send(
            "Format: `!best <liga>`. Contoh: `!best epl`, `!best ucl`, `!best liga portugal`."
        )
        return
    league_query = " ".join(args).strip()
    from agents.football.league_resolver import resolve_league_leading

    resolved = resolve_league_leading(league_query)
    if not resolved:
        await message.channel.send(
            f"Liga tidak dikenali dari `{league_query}`. "
            "Liga dikenal: ucl, epl, serie a, liga portugal, eredivisie, dll."
        )
        return
    league_key = resolved[0]
    runner_args = ["best", "--league", league_key]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_best_goal(message: discord.Message, args: list[str]) -> None:
    """`!bestgoalmatch`: match paling banjir gol hari ini (expected goals
    tertinggi dari form attack/defense + market totals kalau ada)."""
    runner_args = ["bestgoalmatch"]
    if args:
        league_query = " ".join(args).strip()
        from agents.football.league_resolver import resolve_league_leading

        resolved = resolve_league_leading(league_query)
        if not resolved:
            await message.channel.send(
                f"Liga tidak dikenali dari `{league_query}` — scan semua liga "
                "atau pakai liga terdaftar (epl, ucl, dll)."
            )
            return
        runner_args += ["--league", resolved[0]]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_compare(message: discord.Message, args: list[str]) -> None:
    if len(args) < 2:
        await message.channel.send(
            "Format: `!football compare <HOME> <AWAY> [league]`. "
            "Contoh: `!football compare MCN ARS EPL`."
        )
        return
    home = args[0]
    away = args[1]
    league = args[2] if len(args) >= 3 else "EPL"

    runner_args = ["compare", "--home", home, "--away", away, "--league", league]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


def _parse_match_query(rest: str) -> tuple[str, str, str] | None:
    """Parse '<league> <home> vs <away>' -> (league_key, home, away).

    League keyword first; every token prefix is tried so multi-word leagues
    ('liga portugal Santa Clara vs Nacional') resolve correctly. Returns None
    when the separator, a side, or the league cannot be resolved.
    """
    from agents.football.league_resolver import resolve_league_leading

    rest = rest.strip()
    if not rest or " vs " not in rest.lower():
        return None
    parts = re.split(r"\s+vs\s+", rest, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    left, away = parts[0].strip(), parts[1].strip()
    if not left or not away:
        return None
    tokens = left.split()
    if len(tokens) < 2:
        return None
    resolved = None
    consumed = 0
    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n])
        match = resolve_league_leading(candidate)
        if match:
            resolved = match
            consumed = n
            break
    if not resolved:
        return None
    home = " ".join(tokens[consumed:])
    if not home:
        return None
    return resolved[0], home, away


async def _handle_source_match(message: discord.Message, source: str, rest: str) -> None:
    """`!livescore <liga> <home> vs <away>` / `!flashscore <liga> <home> vs <away>`.

    League is REQUIRED (unlike `analisa match`, there is no auto-detect):
    the runner searches the named source for today, then tomorrow, validates
    the match (league + teams + date), collects the source's match data, then
    reuses the existing analyse pipeline (NowGoal odds -> prediction -> the
    existing output format).
    """
    parsed = _parse_match_query(rest)
    if parsed is None:
        await message.channel.send(
            f"Format: `!{source} <liga> <home> vs <away>`. "
            f"Contoh: `!{source} laliga barcelona vs real madrid`."
        )
        return
    league_key, home, away = parsed
    runner_args = [source, "--league", league_key, "--home", home, "--away", away]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_analyse(message: discord.Message, rest: str) -> None:
    rest = rest.strip()
    if not rest:
        await message.channel.send(
            "Format: `analisa match <liga> <home> vs <away>`. "
            "Contoh: `analisa match liga portugal Santa Clara vs Nacional`."
        )
        return
    if " vs " not in rest.lower():
        await message.channel.send(
            "Format butuh separator `vs`. Contoh: `analisa match liga portugal Santa Clara vs Nacional`."
        )
        return
    parts = re.split(r"\s+vs\s+", rest, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return
    left, away = parts[0].strip(), parts[1].strip()
    if not left or not away:
        await message.channel.send("Home dan away tidak boleh kosong.")
        return
    tokens = left.split()
    from agents.football.league_resolver import resolve_league_leading
    resolved = None
    consumed = 0
    if len(tokens) >= 2:
        # League keyword first (`analisa match <liga> <home> vs <away>`); try
        # every prefix so 'liga portugal Santa Clara' resolves to the league.
        for n in range(len(tokens), 0, -1):
            candidate = " ".join(tokens[:n])
            match = resolve_league_leading(candidate)
            if match:
                resolved = match
                consumed = n
                break
    if not resolved:
        # User typed the match WITHOUT a registered league keyword
        # (`analisa match manchester united vs leeds united`), or the keyword
        # they typed is not registered (e.g. "asean thailand vs singapore"
        # where "asean" is not a league). Build a candidate list of home names
        # by dropping the leftmost token one at a time -- the FULL left is
        # always tried first, so legitimate boxes are unchanged (no
        # regression). For each candidate, ask the runner's detect to find
        # the fixture across its registered leagues (football-data ->
        # livescore -> homepage -> thesportsdb -> flashscore team-fixtures).
        # The first candidate whose detect returns a hit wins. The cap
        # (_DETECT_DROP_MAX) bounds the worst case so nonsense input never
        # blows the runner deadline.
        cand_from = (
            [left] + [" ".join(tokens[i:]) for i in range(1, len(tokens))]
        )[: _DETECT_DROP_MAX + 1]
        detected = None
        last_cand = ""
        for cand in cand_from:
            if not cand:
                continue
            last_cand = cand
            try:
                async with message.channel.typing():
                    detected = await _invoke_runner(
                        ["detect", "--home", cand, "--away", away]
                    )
            except Exception:
                detected = None
            if (detected or {}).get("raw", {}).get("found"):
                break
        d = (detected or {}).get("raw") or {}
        if d.get("found") and d.get("league"):
            league_key = d["league"]
            runner_args = [
                "analyse", "--league", league_key,
                "--home", d["home"], "--away", d["away"],
            ]
            async with message.channel.typing():
                result = await _invoke_runner(runner_args)
            await _post_result(message, result)
            return
        if d.get("found"):
            # Non-registered competition (friendly, minor cup, qualifier...):
            # D2 (dynamic league discovery, 2026-08-17): run the FULL analysis
            # pipeline -- the league is read from the fixture (flashscore /
            # livescore) and becomes a deterministic dyn: key; the output is
            # honestly labelled uncalibrated_league when the competition has
            # no registered calibration fit.
            runner_args = [
                "analyse",
                "--home", d["home"], "--away", d["away"],
            ]
            async with message.channel.typing():
                result = await _invoke_runner(runner_args)
            await _post_result(message, result)
            return
        await message.channel.send(
            f"Liga tidak dikenali dari `{last_cand.split()[0] if last_cand else left.split()[0]}`.\n\n"
            f"Format: `analisa match <liga> <home> vs <away>`\n"
            f"Contoh: `analisa match epl arsenal vs chelsea`,\n"
            f"`analisa match liga portugal Santa Clara vs Nacional`.\n"
            f"Liga dikenal: ucl, epl, serie a, liga portugal, eredivisie, dll.\n"
            f"Match non-liga (friendly, cup) bisa langsung dianalisa: "
            f"`analisa match <home> vs <away>`."
        )
        return
    home = " ".join(tokens[consumed:])
    if not home:
        await message.channel.send("Home team kosong. Sebut nama tim setelah liga.")
        return
    league_key = resolved[0]
    runner_args = ["analyse", "--league", league_key, "--home", home, "--away", away]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_settle(message: discord.Message, args: list[str]) -> None:
    """Catat hasil match ke prediction log (manual atau auto per tanggal)."""
    if not args:
        await message.channel.send(
            "Format: `!football settle <home> vs <away> <2-1>` (manual) atau "
            "`!football settle auto [YYYY-MM-DD]` (ambil hasil selesai)."
        )
        return

    if args[0].lower() in ("auto", "otomatis"):
        date = None
        if len(args) >= 2 and re.match(r"^\d{4}-\d{2}-\d{2}$", args[1]):
            date = args[1]
        runner_args = ["settle", "auto"]
        if date:
            runner_args += ["--date", date]
        async with message.channel.typing():
            result = await _invoke_runner(runner_args)
        await _post_result(message, result)
        return

    m = re.match(
        r"^(.*?)\s+vs\s+(.+?)\s+(\d{1,2})-(\d{1,2})$", " ".join(args), re.IGNORECASE
    )
    if not m:
        await message.channel.send(
            "Format: `!football settle <home> vs <away> <2-1>`. "
            "Contoh: `!football settle bodo vs union 2-1`."
        )
        return
    left, away, hg, ag = m.group(1).strip(), m.group(2).strip(), m.group(3), m.group(4)

    from agents.football.league_resolver import resolve_league_leading

    league_key = None
    tokens = left.split()
    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n])
        resolved = resolve_league_leading(candidate)
        if resolved:
            league_key = resolved[0]
            left = " ".join(tokens[n:]).strip()
            break
    if not left:
        await message.channel.send("Nama home team kosong setelah liga.")
        return

    runner_args = ["settle", "--home", left, "--away", away, "--result", f"{hg}-{ag}"]
    if league_key:
        runner_args += ["--league", league_key]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_odds_snapshot(message: discord.Message, args: list[str]) -> None:
    """Catat odds 1X2 pada timing tertentu untuk evaluasi CLV historis.

    Format: `!football odds <T-24h|T-6h|T-1h|T-15m> <home> vs <away> <h,d,a> [league]`
    Contoh: `!football odds T-6h bodo vs union 1.62,4.30,4.60`
    """
    if len(args) < 3:
        await message.channel.send(
            "Format: `!football odds <T-24h|T-6h|T-1h|T-15m> <home> vs <away> <h,d,a> [league]`. "
            "Contoh: `!football odds T-6h bodo vs union 1.62,4.30,4.60`."
        )
        return
    timing = args[0].upper()
    if timing not in ("T-24H", "T-6H", "T-1H", "T-15M"):
        await message.channel.send("timing harus T-24h / T-6h / T-1h / T-15m.")
        return
    rest = " ".join(args[1:])
    m = re.match(
        r"^(.*?)\s+vs\s+(.+?)\s+([\d.]+),([\d.]+),([\d.]+)(?:\s+(.+))?$",
        rest, re.IGNORECASE,
    )
    if not m:
        await message.channel.send(
            "Format: `!football odds <timing> <home> vs <away> <h,d,a> [league]`. "
            "Contoh: `!football odds T-6h bodo vs union 1.62,4.30,4.60`."
        )
        return
    left, away, h, d, a, trailing = (
        m.group(1).strip(), m.group(2).strip(),
        m.group(3), m.group(4), m.group(5),
        (m.group(6) or "").strip(),
    )
    league_key = None
    home = left
    from agents.football.league_resolver import resolve_league_leading

    tokens = left.split()
    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n])
        resolved = resolve_league_leading(candidate)
        if resolved:
            league_key = resolved[0]
            home = " ".join(tokens[n:]).strip()
            break
    if not home or not away:
        await message.channel.send("Nama tim tidak boleh kosong.")
        return
    if not trailing and not league_key:
        league_key = None
    runner_args = [
        "odds-snapshot", "--timing", timing, "--home", home, "--away", away,
        "--odds", f"{h},{d},{a}",
    ]
    if league_key:
        runner_args += ["--league", league_key]
    async with message.channel.typing():
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_update(message: discord.Message, args: list[str]) -> None:
    """Handle !football update elo — scrape latest ELO ratings."""
    sub = args[0].lower() if args else ""
    if sub != "elo":
        await message.channel.send("Usage: `!football update elo`")
        return

    await message.channel.send("Fetching latest ELO ratings from elofootball.com...")
    async with message.channel.typing():
        try:
            from agents.football.elo_scraper import scrape_elo, update_elo_json
            import asyncio
            loop = asyncio.get_event_loop()
            teams = await loop.run_in_executor(None, scrape_elo)
            if not teams:
                await message.channel.send("Failed to scrape ELO data.")
                return
            result = await loop.run_in_executor(None, lambda: update_elo_json(teams))
            
            msg = (
                f"ELO ratings updated!\n"
                f"Total teams: {result['total_teams']}\n"
                f"Updated: {result['updated']}\n"
                f"Added: {result['added']}\n"
                f"Source: elofootball.com"
            )
            await message.channel.send(msg)
        except Exception as e:
            await message.channel.send(f"Error updating ELO: {e}")


async def _handle_stats(message: discord.Message, args: list[str]) -> None:
    """Tampilkan statistik realisasi prediction log (hit rate, logloss, ROI, CLV)."""
    runner_args: list[str] = ["stats"]
    for arg in args:
        if arg.isdigit():
            # edge threshold dalam persen, mis. `!football stats 3` = edge >= 3%
            runner_args += ["--edge-threshold", f"{int(arg) / 100.0:.4f}"]
            break
    async with message.channel.typing():
        # Settle finished matches first so the stats always reflect the
        # latest results without a manual `!football settle` (no-op when
        # auto_settle is disabled or nothing new settled). Serialized with
        # the background loop via _AUTO_SETTLE_LOCK; typing shows while it
        # runs so a slow catch-up never looks like a hung bot.
        if _auto_settle_enabled():
            try:
                async with _AUTO_SETTLE_LOCK:
                    await _run_auto_settle_once()
            except Exception:
                logger.exception("pre-stats auto-settle failed")
        result = await _invoke_runner(runner_args)
    await _post_result(message, result)


async def _handle_clv(message: discord.Message, args: list[str]) -> None:
    """Tampilkan CLV (Closing Line Value) dashboard."""
    from agents.football.market_intel_poll import clv_dashboard
    async with message.channel.typing():
        dashboard = clv_dashboard(root=str(ROOT))
    lines = dashboard.get("summary", [])
    if not lines:
        lines = ["No CLV data available yet."]
    await message.channel.send("\n".join(lines))


async def _handle_steam(message: discord.Message, args: list[str]) -> None:
    """Tampilkan steam move alerts."""
    from agents.football.market_intel_poll import steam_dashboard
    hours = 24.0
    for arg in args:
        if arg.replace(".", "").isdigit():
            hours = float(arg)
            break
    async with message.channel.typing():
        dashboard = steam_dashboard(root=str(ROOT), hours_back=hours)
    lines = dashboard.get("summary", [])
    if not lines:
        lines = ["No steam moves detected."]
    await message.channel.send("\n".join(lines))


# Discord plain messages are capped at 2000 chars. We split reports on line
# boundaries at 1900 (leaving headroom) and send up to MAX_PLAIN_MESSAGES
# messages; anything beyond that stays available via the 📋 Copy button.
_PLAIN_CHUNK = 1900
_MAX_PLAIN_MESSAGES = 4


def _plain_messages(
    rendered: dict[str, Any],
    max_messages: int | None = _MAX_PLAIN_MESSAGES,
    max_body: int | None = None,
) -> list[str]:
    """Full report as plain-text chunks: title + body + footer, <=1900 each.

    Plain Discord messages render markdown exactly like embed descriptions, so
    ``**bold**``, emoji and inline code all keep working — without the 4096-char
    embed cap that previously truncated long reports to ``…(terpotong)``.

    - ``max_messages`` caps how many messages the reply may use (None =
      unlimited — the Copy button passes None so the FULL report is served).
    - ``max_body`` optionally truncates the body before chunking (Copy path).

    The footer (quota, sources, grade summary) is short but important, so it
    is pinned to the LAST visible message instead of being appended to the
    joined text where it would land in a dropped chunk on long reports.
    """
    title = (rendered.get("title") or "").strip()
    body = (rendered.get("body") or "").strip()
    footer = (rendered.get("footer") or "").strip()
    if max_body is not None and len(body) > max_body:
        body = body[:max_body].rstrip() + "\n…(terpotong, laporan sangat panjang)"
    if not (title or body or footer):
        return ["(tidak ada konten)"]
    # Reserve message slots for the title and footer so the final count can
    # never exceed max_messages once they are added below.
    reserved = (1 if title else 0) + (1 if (footer and footer != " ") else 0)
    body_source = body if body else title
    chunks = _split_body_chunks(body_source, limit=_PLAIN_CHUNK)
    if max_messages is not None and len(chunks) > max_messages - reserved:
        chunks = chunks[:max_messages - reserved]
        chunks[-1] = chunks[-1] + "\n\n…(lanjutan via tombol 📋 Detail)"
    # Prepend the title to the first chunk (splitting keeps it intact). When
    # body is empty, body_source == title so it is already the first chunk.
    if title and body:
        if chunks[0] and len(title) + 2 + len(chunks[0]) <= _PLAIN_CHUNK:
            chunks[0] = title + "\n\n" + chunks[0]
        else:
            chunks.insert(0, title)
    # Pin the footer to the final visible message.
    if footer and footer != " ":
        if chunks[-1] and len(chunks[-1]) + 2 + len(footer) <= _PLAIN_CHUNK:
            chunks[-1] = chunks[-1] + "\n\n" + footer
        else:
            chunks.append(footer)
    return chunks


async def _post_result(
    message: discord.Message,
    result: dict[str, Any],
    view: discord.ui.View | None = None,
) -> None:
    payload = _result_payload(result)
    # Paginated top (many competitions, no primary matches): render pages from
    # the already-fetched data, never re-querying.
    if payload.get("pages"):
        await _post_paginated_top(message, payload)
        return
    # MATCH SIGNAL card: post the accordion (summary embed + toggle view)
    # instead of the plain-text card. Everything else keeps the existing
    # plain-text + 📋 Detail path untouched.
    card = _signal_card_accordion(result)
    if card is not None:
        embed, card_view = card
        sent = await message.channel.send(embed=embed, view=card_view)
        card_view.attach_message(sent)
        return
    chunks = _plain_messages(payload)
    await message.channel.send(
        content=chunks[0],
        view=view if view is not None else _copy_view(_copy_payload(result, payload)),
    )
    for chunk in chunks[1:]:
        # Small gap keeps a multi-message burst under Discord's 5 msg / 5 s
        # per-channel rate limit when several commands run back-to-back.
        await asyncio.sleep(0.35)
        await message.channel.send(content=chunk)


# ---- Auto-settle (background) --------------------------------------------
#
# When a match the bot predicted finishes, its result should reach the
# prediction log WITHOUT a manual `!football settle` — that is the data
# (hit rate / ROI / CLV / Elo updates / calibration refit) the model is
# evaluated and improved on. The bot runs a background task that settles
# the last N days on an interval. Config (config/football.json):
#   auto_settle.enabled / interval_hours / days_back / refresh_calibration
# Note: disabling mid-run stops the loop at the next tick; re-enabling
# without a restart takes effect on the next reconnect (on_ready).

_AUTO_SETTLE_TASK: asyncio.Task | None = None
_TOR_HEALTH_TASK: asyncio.Task | None = None

# Phase 0.4: the CLV segment report runs on a schedule (not just manual
# invocation). Track the last date it wrote so the 6h auto-settle loop only
# regenerates it once per calendar day (config clv_report.enabled /
# interval_hours).
_CLV_REPORT_LAST_DATE: str | None = None

# Serializes the two bot-initiated settle paths (background loop and the
# `!football stats` catch-up) so two settle subprocesses can never race on
# the same snapshot (duplicate rows in the append-only log) or hit
# football-data concurrently.
_AUTO_SETTLE_LOCK = asyncio.Lock()


def _auto_settle_cfg() -> dict[str, Any]:
    """auto_settle config section (empty dict on any read failure)."""
    try:
        cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
        return dict(cfg.get("auto_settle") or {})
    except Exception:
        return {}


def _clv_report_cfg() -> dict[str, Any]:
    """clv_report config section (Phase 0.4; empty dict on any read failure)."""
    try:
        cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
        return dict(cfg.get("clv_report") or {})
    except Exception:
        return {}


def _edge_bucket_audit_cfg() -> dict[str, Any]:
    """edge_bucket_audit config section (Phase 5.4; empty dict on any read failure)."""
    try:
        cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
        return dict(cfg.get("edge_bucket_audit") or {})
    except Exception:
        return {}


def _auto_settle_enabled() -> bool:
    return bool(_auto_settle_cfg().get("enabled", True))


async def _auto_settle_dates() -> list[str]:
    """ISO dates to settle: today and up to days_back-1 days before (WIB)."""
    from datetime import datetime, timedelta, timezone

    WIB = timezone(timedelta(hours=7))
    days_back = max(1, int(_auto_settle_cfg().get("days_back", 2)))
    today = datetime.now(WIB).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(days_back)]


async def _run_auto_settle_once() -> dict[str, Any]:
    """Settle the last N days via the runner; returns {dates, settled_total}.

    The runner's `settle auto` also advances the live Elo ratings with the
    real results, so this is the full model-feedback step.

    Phase 0.4: when ``clv_report.enabled`` and the report has not been
    written for the current WIB date, regenerate it (scheduled, not only on
    manual invocation).
    """
    global _CLV_REPORT_LAST_DATE
    dates = await _auto_settle_dates()
    total = 0
    for d in dates:
        result = await _invoke_runner(["settle", "auto", "--date", d])
        raw = result.get("raw") or {}
        n = len(raw.get("settled") or [])
        if n:
            total += n
            logger.info("auto-settle %s: %d match settled", d, n)
        elif result.get("error"):
            logger.warning("auto-settle %s failed: %s", d, result["error"])
    # Scheduled CLV segment report (Phase 0.4): once per WIB calendar day.
    try:
        cr_cfg = _clv_report_cfg()
        if cr_cfg.get("enabled", True):
            from datetime import datetime, timedelta, timezone

            wib_now = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
            if _CLV_REPORT_LAST_DATE != wib_now:
                rep = await _invoke_runner(["clv-report", "--date", wib_now])
                raw_rep = rep.get("raw") or {}
                if raw_rep.get("file"):
                    _CLV_REPORT_LAST_DATE = wib_now
                    logger.info(
                        "clv report written: %s (coverage %s%%)",
                        raw_rep["file"],
                        (raw_rep.get("coverage") or {}).get("closing_coverage_pct"),
                    )
                else:
                    logger.warning("clv report generation failed: %s", rep.get("error"))
    except Exception:
        logger.exception("clv report scheduled generation failed")
    # Scheduled edge-bucket-vs-CLOSING audit (Phase 5.4): regenerated on the
    # same cadence so a bucket turning net-negative blocks recommendations
    # via the hard filter in run_decision_engine.
    try:
        eba_cfg = _edge_bucket_audit_cfg()
        if eba_cfg.get("enabled", True):
            from datetime import datetime, timedelta, timezone

            wib_now = datetime.now(timezone(timedelta(hours=7))).date().isoformat()
            if _CLV_REPORT_LAST_DATE != wib_now:
                rep = await _invoke_runner(["bucket-audit", "--date", wib_now])
                raw_rep = rep.get("raw") or {}
                if raw_rep.get("file"):
                    _CLV_REPORT_LAST_DATE = wib_now
                    logger.info(
                        "edge-bucket audit written: %s (net-negative buckets: %s)",
                        raw_rep["file"],
                        ", ".join(raw_rep.get("net_negative_buckets") or []) or "none",
                    )
                else:
                    logger.warning("edge-bucket audit failed: %s", rep.get("error"))
    except Exception:
        logger.exception("edge-bucket audit scheduled generation failed")
    return {"dates": dates, "settled_total": total}


async def _auto_settle_loop() -> None:
    """Background loop: settle immediately on start, then every interval."""
    while True:
        if not _auto_settle_enabled():
            return
        try:
            async with _AUTO_SETTLE_LOCK:
                summary = await _run_auto_settle_once()
                if summary["settled_total"] and _auto_settle_cfg().get("refresh_calibration", True):
                    # New settled results -> refit the calibration curve. The
                    # runner guards internally: a too-small sample keeps the
                    # old fit, so this is safe to run every tick.
                    await _invoke_runner(["calib-refresh", "--min-samples", "100"])
        except Exception:
            logger.exception("auto-settle iteration failed")
        try:
            hours = float(_auto_settle_cfg().get("interval_hours", 6))
        except (TypeError, ValueError):
            hours = 6.0
        await asyncio.sleep(max(0.5, hours) * 3600.0)


_ODDS_POLL_TASK: asyncio.Task | None = None


def _odds_poll_cfg() -> dict[str, Any]:
    """auto_odds_poll config section (empty dict on any read failure)."""
    try:
        cfg = json.loads((ROOT / "config" / "football.json").read_text(encoding="utf-8"))
        return dict(cfg.get("auto_odds_poll") or {})
    except Exception:
        return {}


async def _odds_poll_loop() -> None:
    """Background loop: hourly odds capture for unsettled matches (Plan B).

    Polls the runner's ``odds-poll`` mode, which snapshots the current 1X2
    consensus for every unsettled match kicking off within the lookahead
    window. The stored price curve drives the movement signal + CLV gate.
    """
    while True:
        if not _odds_poll_cfg().get("enabled", False):
            return
        try:
            result = await _invoke_runner(["odds-poll"])
            raw = result.get("raw") or {}
            n = raw.get("n_polled", 0)
            if n:
                logger.info("odds-poll: %d snapshots appended", n)
            elif result.get("error"):
                logger.warning("odds-poll failed: %s", result["error"])
        except Exception:
            logger.exception("odds-poll iteration failed")
        try:
            minutes = float(_odds_poll_cfg().get("loop_interval_minutes", 5))
        except (TypeError, ValueError):
            minutes = 5.0
        await asyncio.sleep(max(1.0, minutes) * 60.0)


@client.event
async def on_ready() -> None:
    logger.info("hermes-football connected as %s", client.user)
    global _AUTO_SETTLE_TASK, _TOR_HEALTH_TASK, _ODDS_POLL_TASK
    if _auto_settle_enabled() and (_AUTO_SETTLE_TASK is None or _AUTO_SETTLE_TASK.done()):
        _AUTO_SETTLE_TASK = asyncio.create_task(_auto_settle_loop())
        logger.info("auto-settle background task started")
    if _odds_poll_cfg().get("enabled", False) and (_ODDS_POLL_TASK is None or _ODDS_POLL_TASK.done()):
        _ODDS_POLL_TASK = asyncio.create_task(_odds_poll_loop())
        logger.info("odds-poll background task started")
    # Periodic health check: Tor is on-demand (started per command, stopped
    # after an idle grace), so there is nothing to bootstrap at bot start.
    # The health loop only relaunches Tor when it dies WHILE IN USE -- an
    # idle Tor is supposed to be down, and the next acquire starts it.
    if _TOR_HEALTH_TASK is None or _TOR_HEALTH_TASK.done():
        _TOR_HEALTH_TASK = asyncio.create_task(_tor_health_loop())


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    try:
        await _handle(message)
    except Exception:
        logger.exception("error handling message")
        try:
            await message.channel.send("Internal error, lihat log lokal.")
        except discord.HTTPException:
            pass


@client.event
async def on_interaction(interaction: discord.Interaction) -> None:
    try:
        if interaction.type != discord.InteractionType.component:
            return
        custom_id = (interaction.data or {}).get("custom_id", "")
        if custom_id.startswith("football_top_"):
            await _handle_top_interaction(interaction, custom_id)
            return
        if not custom_id.startswith("football_copy_"):
            return
        _purge_copy_texts(time.time())
        entry = _COPY_TEXTS.pop(custom_id, None)
        if entry is None:
            await interaction.response.send_message(
                "Sesi copy sudah kedaluwarsa (button hanya aktif 15 menit) — "
                "jalankan ulang perintahnya.",
                ephemeral=True,
            )
            return
        # Consumed button is removed from disk too, so a later restart cannot
        # resurrect it and serve the report twice.
        _save_copy_texts()
        await _send_ephemeral_chunks(interaction, _copy_plain_messages(entry[1]))
    except Exception:
        logger.exception("error handling button interaction")


def _acquire_single_instance() -> bool:
    """Reserve the bot to a single process.

    Windows: OS file lock via ``msvcrt`` (auto-released when the process
    dies; no stale locks). POSIX: TCP port guard. A second `run` call fails
    and exits cleanly with code 2.
    """
    if sys.platform == "win32":
        import msvcrt

        lock_path = ROOT / ".bot.lock"
        try:
            fh = open(lock_path, "a+")  # noqa: SIM115 - held open for process lifetime
        except OSError:
            return False
        try:
            fh.write("\0")
            fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            return False
        _SINGLE_INSTANCE_GUARD.append(fh)
        return True

    # POSIX: keep the socket guard (works there).
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 47115))
    except OSError:
        sock.close()
        return False
    sock.listen(1)
    _SINGLE_INSTANCE_GUARD.append(sock)
    return True


_SINGLE_INSTANCE_GUARD: list[object] = []


def main() -> int:
    if not TOKEN:
        print("DISCORD_FOOTBALL_TOKEN kosong di .env", file=sys.stderr)
        return 1
    if not ALLOWED_USER_ID:
        print("FOOTBALL_ALLOWED_USER_ID kosong di .env", file=sys.stderr)
        return 1

    if not _acquire_single_instance():
        print("Hermes-Football bot sudah jalan (single-instance lock dipakai), exit", file=sys.stderr)
        return 2

    # Clean up the tor.exe we auto-started on every exit path (normal return,
    # KeyboardInterrupt, SIGTERM, atexit). A pre-existing Tor is left alone.
    import atexit
    import signal

    atexit.register(_shutdown_tor)
    for sig in (getattr(signal, "SIGTERM", None),):
        if sig is not None:
            try:
                signal.signal(sig, lambda *a: _shutdown_tor())
            except (ValueError, OSError):
                pass
    try:
        client.run(TOKEN, log_level=logging.INFO)
    finally:
        _shutdown_tor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
