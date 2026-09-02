"""CLI entrypoint. Outputs JSON to stdout for the Discord bot to parse."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from .analyse import find_specific_match
from .cache import Cache
from .compare import compare_teams
from .source_match import find_source_match
from .format import (
    format_analyse,
    format_best,
    format_best_goal,
    format_compare,
    format_market_signal,
    format_odds_snapshot,
    format_settle,
    format_signal_detail,
    format_stats,
    format_top,
)
from .football_data import FootballDataClient, FootballDataError
from .match_finder import find_top_matches
from .model_gates import CONFIDENCE_ALLOWLIST
from .multi_source import MultiSourceStatsFetcher
from .odds_fetcher import OddsApiError, OddsFetcher
from .settler import settle_auto, settle_manual
from .prediction_log import DEFAULT_LOG_PATH, compute_stats


def _build_nowgoal(cfg: dict[str, Any], proxy_url: str | None) -> Any | None:
    """NowGoal client (feature-gated). Shared by top/best/bestgoalmatch for
    the schedule fallback and by analyse for odds + analysis fallback."""
    if not (cfg.get("feature_flags") or {}).get("enable_nowgoal", False):
        return None
    from .nowgoal import NowGoalClient

    ng_cfg = cfg.get("nowgoal") or {}
    return NowGoalClient(
        base_url=os.getenv("NOWGOAL_BASE_URL") or None,
        throttle_seconds=float(cfg.get("rate_limit_seconds", 1.1)),
        proxy=proxy_url,
        mirrors=ng_cfg.get("mirrors"),
    )


def _collect_oddspapi_keys() -> list[str]:
    """Kumpulkan semua OddsPapi keys dari env + file (rolling pool).

    Sumber (digabung, dedup preserve order):
    1. ODDSPAPI_KEYS (jamak, koma/newline/space)
    2. ODDSPAPI_KEY  (tunggal ATAU koma-list backward-compat)
    3. ODDSPAPI_KEY_1 .. _100 / ODDSPAPI_KEYS_1 ..
    4. File via ODDSPAPI_KEYS_FILE / ODDSPAPI_KEY_FILE
    5. File default: apikeys.txt, apikeys, oddspapi_keys.txt,
       cache/football/oddspapi_keys.txt (satu key per baris)
    """
    import re

    ROOT_LOCAL = Path(__file__).resolve().parent.parent.parent
    keys: list[str] = []

    def _add_raw(raw: str | None) -> None:
        if not raw:
            return
        for p in re.split(r"[,\s;]+", raw.strip()):
            s = p.strip()
            if s and s not in keys:
                keys.append(s)

    _add_raw(os.getenv("ODDSPAPI_KEYS"))
    _add_raw(os.getenv("ODDSPAPI_KEY"))
    # Numbered env vars (ODDSPAPI_KEY_1 .. 100)
    for i in range(1, 101):
        for prefix in ("ODDSPAPI_KEY_", "ODDSPAPI_KEYS_"):
            v = os.getenv(f"{prefix}{i}")
            if v:
                v = v.strip()
                if v and v not in keys:
                    keys.append(v)

    # File candidates
    file_env = os.getenv("ODDSPAPI_KEYS_FILE") or os.getenv("ODDSPAPI_KEY_FILE") or os.getenv("ODDSPAPI_APIKEYS_FILE")
    candidates: list[Path] = []
    if file_env:
        candidates.append(Path(file_env))
        # also relative to ROOT
        if not Path(file_env).is_absolute():
            candidates.append(ROOT_LOCAL / file_env)
    # default files
    candidates.extend([
        ROOT_LOCAL / "apikeys.txt",
        ROOT_LOCAL / "apikeys",
        ROOT_LOCAL / "oddspapi_keys.txt",
        ROOT_LOCAL / "cache" / "football" / "oddspapi_keys.txt",
        Path("apikeys.txt"),
    ])
    seen_files: set[str] = set()
    for p in candidates:
        try:
            key = str(p.resolve()) if p.exists() else str(p)
            if key in seen_files:
                continue
            seen_files.add(key)
            if p.exists() and p.is_file():
                txt = p.read_text(encoding="utf-8", errors="ignore")
                for line in txt.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    for part in re.split(r"[,\s;]+", line):
                        s = part.strip()
                        if s and s not in keys:
                            keys.append(s)
        except Exception:
            continue

    # dedupe preserve order (already but double-check)
    seen: set[str] = set()
    uniq: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def _build_oddspapi(cfg: dict[str, Any]) -> Any | None:
    """OddsPapi client (feature-gated, rolling pool). Shared by the
    analyse and match-source (livescore/flashscore) modes."""
    if not (cfg.get("feature_flags") or {}).get("enable_oddspapi", True):
        return None
    keys = _collect_oddspapi_keys()
    if not keys:
        return None
    from .oddspapi import OddspapiClient

    ROOT_LOCAL = Path(__file__).resolve().parent.parent.parent
    state_path = ROOT_LOCAL / "cache" / "football" / "oddspapi_pool_state.json"
    try:
        return OddspapiClient(keys, state_path=state_path)
    except Exception as exc:  # noqa: BLE001 -- fail-open
        logger.warning("oddspapi pool init failed (%s), fallback single key", exc)
        return OddspapiClient(keys[0])


def _silence_root_handlers() -> None:
    """Drop 3rd-party stdout handlers (SoccerData RichHandler) from root logger.

    soccerdata/_config.py configures a RichHandler that streams formatted INFO
    banners to stdout; that corrupts the JSON contract we expose to the bot.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if stream is sys.stderr:
            continue
        root.removeHandler(handler)


_silence_root_handlers()

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

CONFIG_PATH = ROOT / "config" / "football.json"
DEFAULT_LEAGUES = [
    "EPL", "LaLiga", "Serie A", "Bundesliga", "Ligue 1",
    "Primeira Liga", "Eredivisie", "UCL", "UEL", "UECL",
    "Liga 1", "Saudi Pro League", "MLS",
]

logger = logging.getLogger("hermes-football")

# Stats fetchers created in this process (flashscore/understat browsers).
# main() closes them in a finally so headless Chrome is not orphaned after
# each run (zombies piled up over time and slowed the whole machine).
_STATS_REGISTRY: list = []


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def cadence_for(hours: float, schedule: list[dict[str, Any]]) -> int | None:
    """Poll cadence (minutes) for a match ``hours`` before kickoff.

    ``schedule`` is a list of {until_hours, interval_minutes}; the first tier
    whose ``until_hours`` >= ``hours`` wins. Returns None when no tier covers
    the match (e.g. it is further out than the last tier).
    """
    for tier in sorted(schedule, key=lambda t: float(t.get("until_hours", 999))):
        if hours <= float(tier.get("until_hours", 999)):
            return int(tier.get("interval_minutes", 60))
    return None


def timing_label(hours: float) -> str:
    """Snapshot timing label: T-{h}h for >= 1h, T-{m}m inside the final hour.

    In-play captures (``hours < 0``) label T-0h -- the movement series' last
    and heaviest point (time-decay weights it with exp(0/tau) = 1).
    """
    if hours < 0:
        return "T-0h"
    if hours >= 1.0:
        return f"T-{max(1, int(round(hours)))}h"
    return f"T-{max(1, int(round(hours * 60)))}m"


def _proxy_alive(proxy_url: str, timeout: float = 1.0) -> bool:
    """TCP-reachability probe of a configured proxy endpoint (2026-08-23).

    ``.env`` may pin SOCCERDATA_PROXY to a local Tor/Clash SOCKS port that is
    only up when the bot auto-started it. When it is down, every proxied
    client (nowgoal, football-data, thesportsdb) failed INSTANTLY with
    ConnectError even though the upstream sites were reachable direct --
    silently degrading odds/form for whole runs (verified live: Cambuur and
    Newcastle runs on a Tor-less evening). A refused TCP connect is a
    definitive "proxy not there" answer; a slow-to-bootstrap proxy passes
    this probe and keeps its per-request connect timeouts as backstop.
    """
    try:
        from urllib.parse import urlparse

        parsed = urlparse(proxy_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (9050 if str(parsed.scheme).startswith("socks") else 8080)
        import socket

        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config()
    _silence_root_handlers()
    proxy_url = os.getenv("SOCCERDATA_PROXY") or os.getenv("SOCKS_PROXY") or os.getenv("HTTPS_PROXY")
    if proxy_url and not _proxy_alive(proxy_url):
        # Dead local proxy must never poison the run (see _proxy_alive).
        logger.warning(
            "configured proxy %s unreachable -- fallback DIRECT untuk run ini",
            proxy_url,
        )
        proxy_url = None
    if proxy_url:
        logger.info("data layer proxy = %s", proxy_url)
    odds = OddsFetcher(
        os.getenv("THE_ODDS_API_KEY", ""),
        throttle_seconds=cfg["rate_limit_seconds"],
    )
    # LiveScore client wired in when enabled: used as the LAST-RESORT form
    # provider (flashscore/FBref/football-data/thesportsdb all failed).
    livescore_client = None
    ls_cfg = (cfg.get("data_sources") or {}).get("livescore") or {}
    if ls_cfg.get("enabled", False):
        from .livescore import LiveScoreClient

        livescore_client = LiveScoreClient(base_url=ls_cfg.get("base_url") or None)
    stats = MultiSourceStatsFetcher(
        football_data_key=os.getenv("FOOTBALL_DATA_KEY", ""),
        thesportsdb_key=os.getenv("THE_SPORTS_DB_KEY", ""),
        football_data_throttle=6.0,
        proxy=proxy_url,
        flashscore_enabled=bool(
            (cfg.get("feature_flags") or {}).get("enable_flashscore", True)
        ),
        livescore_client=livescore_client,
        # Two browser lanes: the away-team xG chain renders on its own
        # session so the two chains overlap instead of queueing on the
        # single _browser_lock (measured: history+xg phase 161s sequential).
        flashscore_lanes=2,
    )
    cache = Cache(cfg["cache_dir"])
    stats.cache = cache
    _STATS_REGISTRY.append(stats)

    if args.mode == "top":
        if args.leagues:
            leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
        else:
            leagues = cfg["leagues"]
        # NowGoal schedule fallback for the top command: football-data has a
        # 10 req/min free quota, and `!football today` touches one request PER
        # league -- past the 10th league every league 429s and the day comes
        # back empty. NowGoal's schedule endpoint returns ALL leagues in ONE
        # request (via the on-demand Tor proxy), so when football-data fails
        # we take the nowgoal schedule once and filter it per league. No key
        # required; feature-gated like the analyse path.
        nowgoal = _build_nowgoal(cfg, proxy_url)
        return await find_top_matches(
            date=args.date,
            leagues=leagues,
            top_n=args.top_n,
            cfg=cfg,
            odds=odds,
            stats=stats,
            cache=cache,
            days=args.days,
            nowgoal=nowgoal,
        )
    if args.mode == "best":
        from .best_match import find_best_matches

        return await find_best_matches(
            league_query=args.league,
            cfg=cfg,
            odds=odds,
            stats=stats,
            cache=cache,
            date=args.date,
            nowgoal=_build_nowgoal(cfg, proxy_url),
        )
    if args.mode == "bestgoalmatch":
        from .best_match import find_best_goal_matches

        return await find_best_goal_matches(
            cfg=cfg,
            odds=odds,
            stats=stats,
            cache=cache,
            league_query=args.league,
            date=args.date,
            nowgoal=_build_nowgoal(cfg, proxy_url),
            oddspapi=_build_oddspapi(cfg),
        )
    if args.mode == "compare":
        return await compare_teams(
            home_alias=args.home,
            away_alias=args.away,
            league=args.league,
            cfg=cfg,
            odds=odds,
            stats=stats,
            cache=cache,
        )
    if args.mode in ("analyse", "livescore", "flashscore"):
        # Odds source priority (see find_specific_match): oddspapi is PRIMARY,
        # nowgoal SECOND, The Odds API THIRD/last resort (its free-tier quota
        # is tiny). Each fallback is feature-gated; absent key/flag -> None
        # (no fallback). oddspapi needs an API key; nowgoal needs no key and
        # its live response shapes were verified 2026-08-14 (see nowgoal.py).
        oddspapi = _build_oddspapi(cfg)
        nowgoal = _build_nowgoal(cfg, proxy_url)
        if args.mode == "analyse":
            return await find_specific_match(
                league_query=args.league,
                home_query=args.home,
                away_query=args.away,
                cfg=cfg,
                odds=odds,
                stats=stats,
                cache=cache,
                oddspapi=oddspapi,
                nowgoal=nowgoal,
            )
        # Match-source commands: find + validate the match on LiveScore /
        # Flashscore (today -> tomorrow), then hand it to the SAME pipeline.
        return await find_source_match(
            source=args.mode,
            league_query=args.league,
            home_query=args.home,
            away_query=args.away,
            cfg=cfg,
            odds=odds,
            stats=stats,
            cache=cache,
            oddspapi=oddspapi,
            nowgoal=nowgoal,
        )
    if args.mode == "nowgoal-check":
        # Diagnostic (see agents/football/nowgoal.py): is NowGoal reachable
        # from this network, or ISP-blocked? Probes the homepage (block-page
        # detection), the schedule endpoint, and an odds sample, plus
        # alternate mirrors (nowgoal domains rotate).
        from .nowgoal import run_nowgoal_check

        mirrors = (
            [u.strip() for u in args.mirrors.split(",") if u.strip()]
            if args.mirrors
            else [
                "http://www.nowgoal.net/",
                "http://www.nowgoal26.com/",
                "http://www.nowgoal6.com/",
                "http://www.nowgoal7.com/",
            ]
        )
        return await run_nowgoal_check(
            base_url=os.getenv("NOWGOAL_BASE_URL") or None,
            proxy=proxy_url,
            date=args.date,
            match_id=args.match_id,
            mirrors=mirrors,
        )
    if args.mode == "detect":
        # `analisa match <home> vs <away>` without a league keyword: find the
        # registered league (football-data first, flashscore homepage second)
        # so the bot can run the full analysis without the user naming it.
        # Arm the shared analysis budget so the flashscore team-fixtures
        # fallback (`flashscore.py:1525-1532`) honors the budget guard when
        # the user invokes detect from the bot's retry-drop loop -- without
        # a budget armed, `_resolve_via_team_fixtures` would see `rem is None`
        # and run unconstrained.
        from .detect_match import detect_league_match
        from .multi_source import set_analysis_budget

        set_analysis_budget(
            float((cfg.get("detect") or {}).get("budget_seconds", 120.0))
        )

        return await detect_league_match(
            home=args.home,
            away=args.away,
            stats=stats,
            cache=cache,
            date=args.date,
        )
    if args.mode == "stats":
        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        stats = compute_stats(log_path, edge_threshold=args.edge_threshold)
        stats["edge_threshold"] = args.edge_threshold
        return stats
    if args.mode == "validate-multileague":
        # Phase 4.1: multi-league validation harness over the target leagues.
        # Loads whatever fixture caches exist locally and reports missing
        # target leagues honestly (data_missing rows) instead of dropping them.
        from .validate import validate_multileague

        vcfg = cfg.get("validate") or {}
        requested = (
            [x.strip() for x in args.leagues.split(",") if x.strip()]
            if args.leagues else vcfg.get("multileague_leagues")
        )
        fixtures_by_league: dict[str, list[dict[str, Any]]] = {}
        ml_file = ROOT / "cache/football/multileague_fixtures.json"
        if ml_file.exists():
            try:
                rows = json.loads(ml_file.read_text(encoding="utf-8"))
                if isinstance(rows, list):
                    for fx in rows:
                        fixtures_by_league.setdefault(fx.get("league") or "?", []).append(fx)
                elif isinstance(rows, dict):
                    fixtures_by_league = {k: v for k, v in rows.items() if isinstance(v, list)}
            except Exception as exc:  # noqa: BLE001
                logger.warning("multileague fixtures load failed: %s", exc)
        for f in sorted((ROOT / "cache/football").glob("*_fixtures_2022_2026.json")):
            if str(f) == str(ml_file):
                continue
            try:
                rows = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(rows, list) and rows:
                    league = rows[0].get("league") or f.stem
                    fixtures_by_league.setdefault(league, []).extend(rows)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fixtures load failed %s: %s", f.name, exc)
        # Leakage guard (2026-08-16): the aggregate multileague_fixtures.json
        # and the per-league caches contain the SAME matches; loading both
        # without dedupe doubled every league and leaked each match's own
        # result into its replay (EPL n 3040, ROI +31.7% -- fabricated).
        # validate_multileague dedupes before replay and reports the removed
        # counts in the artifact; warn LOUDLY here so a silent reintroduction
        # is impossible.
        rep = validate_multileague(
            fixtures_by_league,
            out_dir=args.out_dir or ROOT / "reports",
            date=args.date,
            requested_leagues=requested or [],
        )
        # NOTE: main() sets the root logging level to ERROR, so a plain
        # logger.warning is silently dropped -- this guard must use ERROR
        # (a double-loaded input is a data-integrity violation, not a note).
        if rep.get("n_duplicates_removed"):
            logger.error(
                "validate-multileague: removed duplicate fixture rows per league "
                "%s (aggregate file + per-league caches overlap). The 2026-08-16 "
                "report was invalidated by this look-ahead leak; this run is deduped.",
                rep["n_duplicates_removed"],
            )
        body = [f"🧪 Multi-league validation {rep.get('date') or 'hari ini'}", ""]
        body.append(
            f"Liga tersedia: {', '.join(rep['available_leagues']) or 'none'} | "
            f"Liga target tanpa data: {', '.join(rep['missing_leagues']) or 'none'}"
        )
        body.append("")
        for s in sorted(
            rep["segments"],
            key=lambda x: (x.get("league") or "", x.get("model") or ""),
        ):
            if s.get("data_missing"):
                body.append(f"• {s['league']}: DATA TIDAK ADA (belum di-cache)")
                continue
            ll = f"{s['log_loss']}" if s.get("log_loss") is not None else "n/a"
            mkt = f"{s['market_log_loss']}" if s.get("market_log_loss") is not None else "n/a"
            kelly = f"{s['kelly_g']:+.4f}" if s.get("kelly_g") is not None else "n/a"
            body.append(
                f"• {s['league']} | {s['model']}: n={s['n']} ll={ll} "
                f"(market {mkt}) brier={s.get('brier')} ece={s.get('ece')} "
                f"roi={s.get('roi')} kelly_g={kelly}"
            )
        return {
            "render": {"title": "🧪 Multi-league validation", "body": "\n".join(body)},
            "raw": rep,
        }
    if args.mode == "bucket-audit":
        # Phase 5.4: automated edge-bucket vs CLOSING-price ROI audit. Any
        # bucket that is net-negative blocks recommendations (hard filter in
        # run_decision_engine); this regenerates the evidence on a schedule.
        from .prediction_log import edge_bucket_audit

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        rep = edge_bucket_audit(
            log_path,
            out_dir=args.out_dir or ROOT / "reports",
            date=args.date,
        )
        body = [f"🪣 Edge-bucket vs CLOSING audit {rep.get('date') or 'hari ini'}", ""]
        for bucket, b in sorted(rep["buckets"].items()):
            roi_txt = f"{b['roi_vs_closing']:+.1%}" if b.get("roi_vs_closing") is not None else "n/a"
            flag = " ⛔ NET-NEGATIVE" if b.get("net_negative") else ""
            body.append(
                f"• {bucket}: n={b['n']} (closing {b['n_with_closing']}) ROI {roi_txt}{flag}"
            )
        if not rep["buckets"]:
            body.append("Belum ada settled bets dengan closing odds.")
        return {
            "render": {"title": "🪣 Edge-bucket audit (closing)", "body": "\n".join(body)},
            "raw": rep,
        }
    if args.mode == "clv-report":
        # Phase 0.4: per-segment (league x market x timing) CLV report against
        # REAL closing prices, written to reports/clv_segments_<date>.json.
        from .prediction_log import clv_segment_report

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        rep = clv_segment_report(
            log_path,
            out_dir=args.out_dir or ROOT / "reports",
            date=args.date,
            edge_threshold=args.edge_threshold,
        )
        body = [
            f"📊 CLV report {rep.get('date') or 'hari ini'}",
            "",
            f"Closing-odds coverage: {rep['coverage']['closing_coverage_pct']}% "
            f"({rep['coverage']['with_closing_odds']}/{rep['coverage']['settled']} settled, "
            f"threshold >= 80%: {'PASS' if rep['coverage']['passed'] else 'FAIL'})",
            f"Segmen: {rep['n_segments']}",
            "",
        ]
        for s in sorted(
            rep["segments"],
            key=lambda x: (x["league"], x["market"], x["timing"]),
        ):
            ci_txt = (
                f"±{s['ci']['half_width']:.3f}" if s.get("ci") and s["ci"]["half_width"] is not None else "n/a"
            )
            body.append(
                f"• {s['league']} | {s['market']} | {s['timing']}: n={s['n']} "
                f"priceCLV {s['price_clv_pct']}% • ROI {s['roi']} • CI±{ci_txt}"
            )
        return {
            "render": {"title": "📊 CLV per segmen", "body": "\n".join(body)},
            "raw": rep,
        }
    if args.mode == "paper-trade":
        # Phase 8: paper-trade graduation report. A segment (league x market x
        # tier) only graduates to live when it has >= min_bets settled bets AND
        # positive realized ROI AND positive price CLV. Until then it stays on
        # MARKET_PRIOR / NO BET (the CLV gate in run_decision_engine enforces
        # this on every live decision).
        from .clv_gate import gate_segment
        from .prediction_log import segment_clv_stats

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        seg = segment_clv_stats(log_path)
        clv_cfg = (cfg.get("models") or {}).get("decision", {}).get("clv_gate") or {}
        min_bets = int(clv_cfg.get("min_bets", 200))
        segments = []
        for key in sorted(seg):
            s = seg[key]
            g = gate_segment(
                seg, league=s["league"], market=s["market"], tier=s["tier"],
                min_bets=min_bets,
                require_roi_positive=bool(clv_cfg.get("require_roi_positive", True)),
                max_ci_halfwidth=(
                    float(clv_cfg["max_ci_halfwidth"])
                    if clv_cfg.get("max_ci_halfwidth") is not None else None
                ),
            )
            segments.append({
                "league": s["league"], "market": s["market"], "tier": s["tier"],
                "n": s["n"], "roi": s["roi"], "price_clv_pct": s["price_clv_pct"],
                "graduates": g["allowed"], "reason": g["reason"],
            })
        return {
            "status": "paper_trade",
            "min_bets": min_bets,
            "require_roi_positive": bool(clv_cfg.get("require_roi_positive", True)),
            "n_segments": len(segments),
            "n_graduated": sum(1 for x in segments if x["graduates"]),
            "segments": segments,
        }
    if args.mode == "odds-snapshot":
        # PHASE 32-33: capture the current 1X2 price at a labelled time before
        # kickoff (T-24h/T-6h/T-1h/...) so CLV can be evaluated historically.
        from .prediction_log import append_odds_snapshot, list_unsettled

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        parts = [x.strip() for x in args.odds.split(",")]
        if len(parts) != 3:
            return {"error": "--odds harus 3 angka: home,draw,away"}
        try:
            odds = dict(zip(("home", "draw", "away"), (float(p) for p in parts)))
        except ValueError:
            return {"error": "--odds harus numerik: home,draw,away"}
        match_id = args.match_id
        if not match_id:
            # Friendly lookup: resolve the match_id from unsettled snapshots by
            # tolerant team names (mirror settle_manual) so Discord users do
            # not have to type the long match_id.
            from .analyse import _teams_match

            snaps = list_unsettled(log_path)
            cands = [
                s for s in snaps
                if (not args.league or (s.get("league") or "") == args.league)
                and _teams_match(args.home or "", s.get("home") or "")
                and _teams_match(args.away or "", s.get("away") or "")
            ]
            if not cands:
                return {"error": "tidak ada snapshot unsettled yang cocok; "
                                "pakai --match-id eksplisit"}
            if len(cands) > 1:
                return {"error": "lebih dari satu snapshot cocok; sertakan --league"}
            match_id = cands[0]["match_id"]
        sources = [x.strip() for x in args.sources.split(",") if x.strip()] if args.sources else None
        ok = append_odds_snapshot(
            log_path, match_id=match_id, timing=args.timing, odds=odds,
            bookmakers_count=args.bookmakers, sources=sources,
        )
        if not ok:
            return {"error": f"tidak ada snapshot prediksi untuk match_id '{match_id}'"}
        return {
            "status": "odds_snapshot",
            "match_id": match_id,
            "timing": args.timing,
            "odds": odds,
            "bookmakers_count": args.bookmakers,
            "sources": sources,
            "log_file": str(log_path),
        }
    if args.mode == "odds-poll":
        # Plan B: tapered odds capture for unsettled snapshots kicking off
        # within the lookahead window. Primary poller = nowgoal (no key, Tor),
        # fallback = oddspapi. Cadence follows time-to-kickoff (schedule config)
        # and a lineup flip near kickoff triggers an immediate snapshot.
        from .analyse import extract_h2h_entries, extract_market_totals
        from .prediction_log import (
            append_odds_snapshot,
            list_odds_snapshots,
            list_unsettled,
        )
        from .scorer import consensus_odds
        from .signal_engine import ah_consensus, extract_asian_handicap, ou_consensus

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        poll_cfg = cfg.get("auto_odds_poll") or {}
        lookahead = float(poll_cfg.get("lookahead_hours", 24))
        schedule = sorted(
            poll_cfg.get("schedule") or [],
            key=lambda t: float(t.get("until_hours", 999)),
        )
        lineup_cfg = poll_cfg.get("lineup_trigger") or {}

        nowgoal = None
        if (cfg.get("feature_flags") or {}).get("enable_nowgoal", False):
            from .nowgoal import NowGoalClient

            ng_cfg = cfg.get("nowgoal") or {}
            nowgoal = NowGoalClient(
                base_url=os.getenv("NOWGOAL_BASE_URL") or None,
                throttle_seconds=float(cfg.get("rate_limit_seconds", 1.1)),
                proxy=proxy_url,
                mirrors=ng_cfg.get("mirrors"),
            )
        oddspapi = _build_oddspapi(cfg)

        def _cadence_for(hours: float) -> int | None:
            return cadence_for(hours, schedule)

        def _timing_label(hours: float) -> str:
            return timing_label(hours)

        async def _fetch_markets(home: str, away: str, live: bool = False):
            """(1x2, bookmakers, source, ah, ou) for one match from one source.

            ``live`` requests the realtime in-play leg (``r`` -- the oddscomp
            page's "Live" column) from nowgoal, extending the movement series
            past the T-0h pre-match point into the match itself. Phase 2.2:
            in-play polls also request the CLOSING price (the ``l`` leg,
            last pre-match -- it persists after settle) once the match is
            finished, so the closing line is captured without waiting for
            the settle command.
            """
            for src in (poll_cfg.get("sources") or ["nowgoal", "oddspapi"]):
                payload = None
                try:
                    if src == "nowgoal" and nowgoal is not None:
                        if live:
                            fx = await nowgoal.find_fixture(home, away)
                            if fx:
                                payload = await nowgoal.fetch_live_odds(
                                    fx, include_closing=live
                                )
                        else:
                            payload = await nowgoal.match_odds(home, away)
                    elif src == "oddspapi" and oddspapi is not None:
                        fx = await oddspapi.find_fixture(home, away)
                        if fx and fx.get("hasOdds"):
                            payload = await oddspapi.fetch_odds(fx)
                except Exception as exc:  # noqa: BLE001 -- next source / skip match
                    logger.warning("odds-poll %s failed %s vs %s: %s", src, home, away, exc)
                    payload = None
                if payload and payload.get("bookmakers"):
                    entries = extract_h2h_entries(payload, home, away, home_query=home, away_query=away)
                    cons = consensus_odds(entries)
                    if cons and cons.get("home", 0) > 0:
                        ah = ah_consensus(extract_asian_handicap(payload))
                        ou = ou_consensus(extract_market_totals(payload))
                        return cons, len(entries), src, ah, ou
            return None, 0, None, None, None

        now = datetime.now(timezone.utc)
        live_window = float(poll_cfg.get("live_window_hours", 2.0))
        polled: list[dict[str, Any]] = []
        for s in list_unsettled(log_path):
            kickoff = s.get("kickoff")
            if not kickoff:
                continue
            try:
                kd = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
                if kd.tzinfo is None:
                    kd = kd.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            hours = (kd - now).total_seconds() / 3600.0
            # In-play captures (0 .. -live_window) feed the realtime ``r`` leg
            # into the movement series; anything further out is stale.
            if hours > lookahead or hours < -live_window:
                continue
            home, away = s.get("home") or "", s.get("away") or ""
            mid = s["match_id"]
            existing = list_odds_snapshots(log_path, mid)

            # Lineup trigger: near kickoff, when lineups flip to confirmed, snap
            # odds immediately (captures the repricing the uniform poll misses).
            if (
                lineup_cfg.get("enabled", False)
                and hours <= float(lineup_cfg.get("check_window_hours", 2))
                and s.get("flashscore_url")
                and not any((x.get("timing") or "").startswith("LINEUP") for x in existing)
            ):
                try:
                    lineups = await stats.fetch_flashscore_lineups_for_match(s["flashscore_url"])
                    if lineups and lineups.get("status") == "confirmed":
                        cons, n_bk, src, ah, ou = await _fetch_markets(home, away)
                        if cons:
                            append_odds_snapshot(
                                log_path, match_id=mid, timing="LINEUP-CONFIRMED",
                                odds=cons, bookmakers_count=n_bk, sources=[src],
                                odds_ah=ah, odds_ou=ou,
                            )
                            polled.append({
                                "match_id": mid, "timing": "LINEUP-CONFIRMED",
                                "source": src, "hours": round(hours, 1),
                            })
                            continue
                except Exception as exc:  # noqa: BLE001 -- lineup check must not break the loop
                    logger.warning("odds-poll lineup check failed %s: %s", mid, exc)

            cadence = _cadence_for(hours)
            if cadence is None:
                continue
            # Phase 2.2 (T-24h stale-opening capture): a match with NO
            # snapshot yet is captured immediately at whatever hours it sits
            # (up to the lookahead) -- the first capture pins the opening
            # price so opening->closing movement is never lost. Config:
            # auto_odds_poll.capture_stale_opening (default true).
            if not existing and not bool(poll_cfg.get("capture_stale_opening", True)):
                if hours > 1.0:
                    continue
            if existing:
                last_ts = max((x.get("ts") or "") for x in existing)
                try:
                    last_dt = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    if (now - last_dt).total_seconds() < cadence * 60.0:
                        continue
                except ValueError:
                    pass
            cons, n_bk, src, ah, ou = await _fetch_markets(home, away, live=hours < 0)
            if not cons:
                continue
            label = _timing_label(hours)
            ok = append_odds_snapshot(
                log_path, match_id=mid, timing=label, odds=cons,
                bookmakers_count=n_bk, sources=[src],
                odds_ah=ah, odds_ou=ou,
            )
            if ok:
                polled.append({
                    "match_id": mid, "timing": label,
                    "source": src, "hours": round(hours, 1),
                })
        # Phase 6: Market Intelligence Poll (steam detection + CLV closing)
        mi_results = {"trend_fetched": 0, "closing_captured": 0, "steam_alerts": 0}
        if (cfg.get("feature_flags") or {}).get("enable_market_intel", True):
            try:
                from .market_intel_poll import run_market_intel_poll
                unsettled_for_mi = [
                    {"match_id": s["match_id"], "home": s.get("home"),
                     "away": s.get("away"), "league": s.get("league", ""),
                     "kickoff": s.get("kickoff")}
                    for s in list_unsettled(log_path)
                ]
                mi_results = await run_market_intel_poll(
                    nowgoal, unsettled_for_mi,
                    root=str(ROOT),
                )
            except Exception as exc:
                logger.warning("market-intel poll failed: %s", exc)
        return {
            "status": "odds_poll",
            "n_polled": len(polled),
            "polled": polled,
            "market_intel": mi_results,
        }
    if args.mode == "movement-report":
        from .movement import movement_accuracy

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        mv_cfg = (cfg.get("models") or {}).get("movement") or {}
        acc = movement_accuracy(
            log_path,
            min_snapshots=int(mv_cfg.get("min_snapshots", 3)),
            steam_threshold_pct=float(mv_cfg.get("steam_threshold_pct", 2.0)),
        )
        return {"status": "movement_report", **acc}
    if args.mode == "settle":
        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or "cache/football/predictions.jsonl")
        # P0 (2026-09-02): ``--log`` points settle / --dedupe / --best-pick at
        # another prediction log -- e.g. baseline/predictions_vps.jsonl after a
        # VPS sync -- so a synced log can be settled and evaluated locally.
        if getattr(args, "log", None):
            _lp = Path(args.log)
            log_path = _lp if _lp.is_absolute() else ROOT / _lp
        if args.dedupe:
            from .prediction_log import dedupe_settles

            report = dedupe_settles(log_path)
            return {
                "status": "dedupe",
                **report,
                "note": "settle rows duplikat dihapus (keep 1 per canonical match); snapshot tidak diubah",
            }
        if args.best_pick:
            # Evaluate stored BEST PICKS against their settled results
            # (hit-rate/ROI per market) -- the pick the bot displayed, not
            # the 1X2 model probability.
            from .prediction_log import best_pick_evaluation

            ev = best_pick_evaluation(log_path, cfg=cfg)
            _fc_names = {
                "K1": "tanpa evidensi (Elo prior)", "K2": "entitas/data salah",
                "K3": "konteks leg-2", "K4": "suggestion dipaksa", "K5": "pick lemah (LEAN)",
                "K6": "model internal tidak sepakat (Elo vs Poisson)",
                "K7": "tanpa keyakinan & tanpa value (prob < 60%)",
                "K8": "hold basi (kandidat lebih kuat ditahan)",
                "K0": "variance pasar",
            }

            def _fmt_bucket(m: str, b: dict[str, Any]) -> str:
                roi_txt = f"{b['roi_pct']}%" if b.get("roi_pct") is not None else "n/a"
                return (
                    f"**{m}**: {b['n']} pick | win-rate {b['win_rate']:.1%} "
                    f"| ROI {roi_txt} ({b['wins']}W/{b['pushes']}P/{b['losses']}L)"
                )

            body = [f"🧾 Evaluasi BEST PICK vs hasil ({ev['n']} match)", ""]
            for m, b in sorted(ev["markets"].items()):
                body.append(_fmt_bucket(m, b))
            if not ev["markets"]:
                body.append("Belum ada BEST PICK tersimpan yang settled.")
            if ev.get("tiers"):
                body.append("")
                body.append("Per tier (K5):")
                for m, b in sorted(ev["tiers"].items()):
                    body.append(_fmt_bucket(m, b))
            if ev.get("failure_classes"):
                body.append("")
                body.append("Kelas kegagalan LOSS BEST PICK:")
                for k, n in sorted(ev["failure_classes"].items()):
                    body.append(f"• {k} {_fc_names.get(k, '')}: {n}")
            sug = ev.get("suggestion") or {}
            if sug.get("n"):
                body.append("")
                body.append(f"💡 Evaluasi SUGGESTION TO PICK ({sug['n']} match)")
                for m, b in sorted(sug["markets"].items()):
                    body.append(_fmt_bucket(m, b))
                if sug.get("failure_classes"):
                    body.append("Kelas kegagalan LOSS SUGGESTION:")
                    for k, n in sorted(sug["failure_classes"].items()):
                        body.append(f"• {k} {_fc_names.get(k, '')}: {n}")
            body.append("")
            body.append("Per match:")
            for p in ev["picks"]:
                roi = f" ROI {p['roi']:+.2f}" if p.get("roi") is not None else ""
                fc = f" [{p['failure_class']}]" if p.get("failure_class") else ""
                _mp = p.get("model_prob")
                _mp_txt = f", prob {float(_mp):.0%}" if isinstance(_mp, (int, float)) else ""
                body.append(
                    f"• [{p.get('tier', 'BEST PICK')}] {p['market']} {p['selection']} "
                    f"({p['confidence']}{_mp_txt}) → {p['result']}{roi}{fc}"
                )
            for p in sug.get("picks") or []:
                roi = f" ROI {p['roi']:+.2f}" if p.get("roi") is not None else ""
                fc = f" [{p['failure_class']}]" if p.get("failure_class") else ""
                body.append(f"• [SUG] {p['market']} {p['selection']} → {p['result']}{roi}{fc}")
            return {
                "render": {"title": "🧾 Evaluasi BEST PICK", "body": "\n".join(body)},
                "raw": ev,
            }
        closing = None
        if args.closing_odds:
            parts = [x.strip() for x in args.closing_odds.split(",")]
            if len(parts) == 3:
                closing = dict(zip(("home", "draw", "away"), (float(p) for p in parts)))
            else:
                return {"error": "--closing-odds harus 3 angka: home,draw,away"}

        # TODO-01: keep the LIVE Elo state moving with settled results -- the
        # ratings must advance exactly like the walk-forward validation does,
        # or live predictions drift from the validated state machine.
        def _update_elo_from(results: list[dict[str, Any]]) -> dict[str, Any]:
            try:
                from .elo import EloModel
                from .prediction_log import _canonical_team_name

                elo_cfg = (cfg.get("models") or {}).get("elo", {})
                # 2026-09-02: same K / home advantage / prior as prediction,
                # and the SAME canonical-first key resolution -- a settle on
                # the display name "Lille" must move "Lille OSC", never fork
                # a new 1500-rated "Lille".
                elo = EloModel(
                    k=float(elo_cfg.get("k", 32.0)),
                    home_advantage=float(elo_cfg.get("home_advantage", 65.0)),
                    initial_rating=float(elo_cfg.get("initial_rating", 1500.0)),
                    base_total_goals=float(elo_cfg.get("base_total_goals", 2.7)),
                    path=ROOT / elo_cfg.get("file", "cache/football/elo.json"),
                )

                def _key(name: Any, league: Any) -> Any:
                    if not name:
                        return name
                    try:
                        canon = _canonical_team_name(str(name), str(league) if league else None) or None
                    except Exception:  # noqa: BLE001
                        canon = None
                    return elo.resolve_first((canon, str(name))) or canon or name

                mapped = [
                    {**r, "home": _key(r.get("home"), r.get("league")), "away": _key(r.get("away"), r.get("league"))}
                    for r in (results or [])
                ]
                applied = elo.update_from_results(mapped)
                return {"elo_updated": applied, "elo_file": str(elo.path)}
            except Exception as exc:  # noqa: BLE001 -- settle must never break
                logger.warning("elo update after settle failed: %s", exc)
                return {"elo_updated": 0, "elo_warning": f"{type(exc).__name__}: {exc}"}

        if args.home and args.away:
            if not args.result:
                return {"error": "settle manual butuh --result '2-1'"}
            report = settle_manual(
                log_path, home=args.home, away=args.away, result=args.result,
                league=args.league, date=args.date, closing_odds=closing,
            )
            if report.get("status") == "settled":
                report.update(_update_elo_from([report]))
            return report
        # auto: settle by date against real finished results. LiveScore is
        # the PRIMARY source (no key, full daily coverage -- football-data
        # occasionally lags a day and returns ZERO finished matches for dates
        # LiveScore already carries, verified 2026-08-15); football-data is
        # the fallback when LiveScore is unreachable/empty. --source forces
        # one provider; "auto" tries LiveScore then football-data.
        WIB = timezone(timedelta(hours=7))
        date = args.date or datetime.now(WIB).date().isoformat()
        source = (args.source or "auto").lower()
        results: list[dict[str, Any]] = []
        used_source = source
        if source in ("auto", "livescore"):
            from .source_match import fetch_finished_livescore_results

            try:
                lv = await fetch_finished_livescore_results(cfg, cache, date)
            except Exception as exc:  # noqa: BLE001 -- settle must never break
                logger.warning("settle livescore failed: %s", exc)
                lv = []
            if lv:
                results = lv
                used_source = "livescore"
        if not results and source in ("auto", "football-data"):
            fd = FootballDataClient(os.getenv("FOOTBALL_DATA_KEY", ""), throttle_seconds=6.0)
            try:
                results = await fd.fetch_finished_matches_by_date(date, date) or []
            except Exception as exc:  # noqa: BLE001 -- fall back cleanly
                logger.warning("settle football-data failed: %s", exc)
                results = []
            if results:
                used_source = "football-data"
        # PHASE 0.1: attach REAL closing 1X2 prices (NowGoal ``l`` leg, last
        # pre-match -- it persists after settle; t=11/roddsList is NOT a
        # closing line, it serves result-embedded final prices) to every
        # settlement -- the root cause of empty CLV was that auto-settle
        # never fetched a closing line. Network stays in this CLI layer;
        # settle_auto receives a sync lookup.
        # Closing-fetch accounting (2026-08-16): a failed OR EMPTY closing
        # fetch used to be silent -- closing coverage could sit at 0% with
        # no trace.
        # Every outcome is now classified (fixture not found / closing empty /
        # exception) and reported under ``closing_fetch``; the legacy
        # ``closing_fetch_errors`` count (exceptions only) is kept for
        # backward compatibility.
        _closing_map: dict[tuple[str, str], dict[str, float]] = {}
        _closing_fetch: dict[str, Any] = {
            "attempted": 0, "ok": 0, "no_fixture": 0, "no_closing": 0,
            "error": 0, "detail": [],
        }
        if results and ((cfg.get("auto_settle") or {}).get("fetch_closing_odds", True)):
            ng_client = _build_nowgoal(cfg, proxy_url)
            if ng_client is None:
                _closing_fetch["attempted"] = len(results)
                _closing_fetch["error"] = len(results)
                _closing_fetch["detail"].append(
                    "nowgoal client unavailable (closing fetch skipped)"
                )
            else:
                for r in results:
                    _closing_fetch["attempted"] += 1
                    home_r, away_r = r.get("home", ""), r.get("away", "")
                    try:
                        _fx = await ng_client.find_fixture(home_r, away_r, date)
                        if not _fx:
                            # Club renamed in NowGoal's schedule (verified
                            # 2026-08-16: "Beveren" vs "Red Star Waasland")
                            # -> resolve by final score; unique + finished
                            # + score match makes a wrong pick impossible.
                            _fx = await ng_client.find_fixture_by_score(
                                home_r, away_r, date,
                                int(r.get("home_goals") or 0),
                                int(r.get("away_goals") or 0),
                            )
                        if not _fx:
                            _closing_fetch["no_fixture"] += 1
                            _closing_fetch["detail"].append(
                                f"{home_r} vs {away_r}: fixture not found"
                            )
                            continue
                        _close = await ng_client.fetch_closing_odds(_fx)
                        if _close:
                            _closing_map[(home_r, away_r)] = _close
                            _closing_fetch["ok"] += 1
                        else:
                            _closing_fetch["no_closing"] += 1
                            _closing_fetch["detail"].append(
                                f"{home_r} vs {away_r}: closing (l-leg) empty"
                            )
                    except Exception as exc:  # noqa: BLE001 -- CLV never breaks settle
                        _closing_fetch["error"] += 1
                        _closing_fetch["detail"].append(
                            f"{home_r} vs {away_r}: {type(exc).__name__}"
                        )
                        logger.warning(
                            "closing odds fetch failed %s vs %s: %s",
                            home_r, away_r, type(exc).__name__,
                        )
        report = settle_auto(
            log_path,
            date=date,
            results=results,
            closing_fetcher=(
                (lambda r: _closing_map.get((r.get("home", ""), r.get("away", ""))))
                if _closing_map else None
            ),
        )
        report["source"] = used_source
        report["results_fetched"] = len(results)
        report["closing_fetch_errors"] = _closing_fetch["error"]
        report["closing_fetch"] = _closing_fetch
        if report.get("settled"):
            report.update(_update_elo_from(report["settled"]))
        return report
    if args.mode == "cache-odds":
        # TODO-07: build/refresh multi-league historical odds caches so
        # decision validation is no longer EPL-only. Downloads from
        # football-data.co.uk (same source the EPL cache used) and writes one
        # JSON per league under cache/football/backtest. Network required.
        from .odds_history import load_history_fixtures

        bcfg = cfg.get("backtest") or {}
        leagues = (
            [x.strip() for x in args.leagues.split(",") if x.strip()]
            if args.leagues else bcfg.get("leagues", ["EPL"])
        )
        seasons = (
            [x.strip() for x in args.seasons.split(",") if x.strip()]
            if args.seasons else bcfg.get("seasons", ["2024-2025", "2025-2026"])
        )
        out_dir = ROOT / (bcfg.get("fixtures_dir") or "cache/football/backtest")
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        warnings: list[str] = []
        for league in leagues:
            fx, warn = load_history_fixtures(
                [league], seasons,
                sleep_seconds=float(cfg.get("rate_limit_seconds", 1.1)),
                proxy=os.getenv("SOCCERDATA_PROXY") or os.getenv("HTTPS_PROXY"),
            )
            if warn:
                warnings.extend(warn)
            if not fx:
                continue
            path = out_dir / f"{league}_fixtures_{seasons[0].replace('-', '_')}_{seasons[-1].replace('-', '_')}.json"
            path.write_text(
                json.dumps(fx, ensure_ascii=False), encoding="utf-8"
            )
            written.append(f"{league}: {len(fx)} matches -> {path.name}")
        return {
            "status": "cache_odds",
            "leagues": leagues,
            "seasons": seasons,
            "out_dir": str(out_dir),
            "written": written,
            "warnings": warnings[:10],
            "note": (
                "Build the multi-league historical-odds caches first, then run "
                "`python -m agents.football.validate --fixtures <files>` "
                "(comma-separated) so the decision layer is validated on more "
                "than EPL (TODO-08)."
            ),
        }
    if args.mode == "seed-league":
        # Build per-league calibration (calibration_<slug>.json) + a merged
        # elo.json from football-data.co.uk history. One bad league/season is
        # skipped with a warning (never kills the run); the merged Elo keeps
        # the higher-games rating when a team reappears across leagues.
        from .calibration import league_slug
        from .elo import EloModel
        from .odds_history import LEAGUE_CODES, download_league_history
        from .validate import run_multi_season_validation

        elo_cfg = (cfg.get("models") or {}).get("elo", {})
        poisson_cfg = (cfg.get("models") or {}).get("poisson", {})
        ensemble_cfg = (cfg.get("models") or {}).get("ensemble", {})
        cal_cfg = (cfg.get("models") or {}).get("calibration", {})
        # run_multi_season_validation builds EloModel(**elo_cfg); the config
        # carries a "file" key EloModel doesn't accept (it takes path).
        elo_seed_cfg = {k: v for k, v in elo_cfg.items() if k != "file"}
        elo_path = ROOT / elo_cfg.get("file", "cache/football/elo.json")
        cal_dir = (ROOT / cal_cfg.get("file", "cache/football/calibration.json")).parent
        seasons = (
            [x.strip() for x in args.seasons.split(",") if x.strip()]
            if args.seasons else ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
        )
        leagues = (
            [x.strip() for x in args.leagues.split(",") if x.strip()]
            if args.leagues else [lg for lg in (cfg.get("leagues") or [])]
        )
        proxy = os.getenv("SOCCERDATA_PROXY") or os.getenv("SOCKS_PROXY") or os.getenv("HTTPS_PROXY")
        if not proxy:
            # No env proxy: fall back to the bot's auto-detect socks ports
            # (Tor/Clash/v2ray) so the seed works with the same on-demand
            # proxy the live data layer uses.
            pad = cfg.get("proxy_auto_detect") or {}
            if pad.get("enabled", True):
                import socket as _socket
                for port in pad.get("socks_ports") or []:
                    try:
                        with _socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
                            proxy = f"socks5h://127.0.0.1:{port}"
                            break
                    except OSError:
                        continue

        master = EloModel(
            k=float(elo_cfg.get("k", 32.0)),
            home_advantage=float(elo_cfg.get("home_advantage", 65.0)),
            initial_rating=float(elo_cfg.get("initial_rating", 1500.0)),
            base_total_goals=float(elo_cfg.get("base_total_goals", 2.7)),
            path=elo_path,
        )
        tmp_seed = cal_dir / "_seed_tmp.json"
        report: dict[str, Any] = {
            "leagues": [], "warnings": [], "elo_teams": len(master.ratings),
        }
        # Results+odds via football-data.co.uk ONLY. FBref is Cloudflare-gated
        # on this network (its internal retries leak threads past the timeout
        # and hang the run), so non-football-data leagues (MLS, UCL, UEL,
        # UECL, Saudi Pro League, Liga 1) are skipped with a warning -- they
        # stay on the Layer-1 uncalibrated prediction path.
        for league in leagues:
            fx: list[dict[str, Any]] = []
            warn: list[str] = []
            if league in LEAGUE_CODES:
                fx, warn = download_league_history(
                    league, seasons,
                    sleep_seconds=float(cfg.get("rate_limit_seconds", 1.1)),
                    proxy=proxy,
                )
            else:
                warn = [f"{league}: skip — tidak ada sumber historis bebas yang reachable (FBref Cloudflare-blocked / API-Football unsubscribed)"]
            report["warnings"].extend(warn)
            if not fx:
                continue
            res = run_multi_season_validation(
                fx,
                elo_cfg=elo_seed_cfg,
                poisson_cfg=poisson_cfg,
                ensemble_cfg=ensemble_cfg,
                calibration_out=cal_dir / f"calibration_{league_slug(league)}.json",
                seed_elo_path=tmp_seed,
            )
            snap = res.get("seeded_elo") or {}
            ratings = snap.get("ratings") or {}
            games = snap.get("games") or {}
            for team, r in ratings.items():
                if team not in master.games or games.get(team, 0) >= master.games[team]:
                    master.ratings[team] = float(r)
                    master.games[team] = int(games.get(team, 0))
            report["leagues"].append({
                "league": league,
                "matches": res.get("n_matches_total", 0),
                "teams": len(ratings),
                "calibration_samples": (res.get("calibration") or {}).get("samples"),
            })
        if tmp_seed.exists():
            tmp_seed.unlink()
        master._rebuild_indexes()
        master.path = elo_path
        master._save()
        report["elo_teams"] = len(master.ratings)
        report["elo_file"] = str(elo_path)
        report["seasons"] = seasons
        return {"status": "seed_league", **report}
    if args.mode == "train-model":
        # ML (ProphitBet-port features): train + walk-forward eval, save
        # artifacts under cache/football/models/<target>/. Offline only.
        from .ml_train import train_model as _train_ml

        ml_cfg = cfg.get("models", {}).get("ml", {})
        leagues = None
        if args.leagues:
            leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
            if leagues == ["all"]:
                leagues = None
        elif ml_cfg.get("leagues"):
            leagues = ml_cfg["leagues"]
        return {"status": "trained", **_train_ml(
            leagues=leagues,
            model=args.model or ml_cfg.get("model", "auto"),
            target=args.target,
            folds=args.folds,
            sampler_name=args.sampler,
            tune_trials=args.tune,
            models_dir=ROOT / ml_cfg.get("models_dir", "cache/football/models"),
            window=int(ml_cfg.get("window", 5)),
            gd_margin=int(ml_cfg.get("gd_margin", 2)),
        )}
    if args.mode == "predict-model":
        # ML live prediction for fixtures on a date. Falls back per-match to
        # the existing Elo+Poisson engine when ML features are unavailable.
        from .ml_predict import predict_fixtures

        ml_cfg = cfg.get("models", {}).get("ml", {})
        leagues = None
        if args.leagues:
            leagues = [x.strip() for x in args.leagues.split(",") if x.strip()]
            if leagues == ["all"]:
                leagues = None
        elif ml_cfg.get("leagues"):
            leagues = ml_cfg["leagues"]
        return await predict_fixtures(
            date=args.date,
            leagues=leagues,
            models_dir=ROOT / ml_cfg.get("models_dir", "cache/football/models"),
            window=int(ml_cfg.get("window", 5)),
            gd_margin=int(ml_cfg.get("gd_margin", 2)),
        )
    if args.mode == "ml-analysis":
        # One-off feature diagnostics: boruta / correlation / variance /
        # logistic coefficients on the cached fixtures.
        from .ml_analyse import run_analysis

        ml_cfg = cfg.get("models", {}).get("ml", {})
        return run_analysis(
            league=args.league,
            metric=args.metric,
            window=int(ml_cfg.get("window", 5)),
            gd_margin=int(ml_cfg.get("gd_margin", 2)),
        )
    if args.mode == "calib-refresh":
        # TODO-02: scheduled re-fit of the calibration curve from the LIVE
        # prediction log (pre-match probabilities of settled outcomes only).
        # Skipped when the settled sample is below the guard; backs up to
        # <path>.bak and keeps the old fit when the refit ECE is worse
        # (regression guard) -- never a silent downgrade.
        from .calibration import Calibrator, refresh_leagues_from_log

        pl_cfg = cfg.get("prediction_log") or {}
        log_path = ROOT / (pl_cfg.get("file") or DEFAULT_LOG_PATH)
        cal_cfg = (cfg.get("models") or {}).get("calibration", {})
        cal_path = ROOT / cal_cfg.get("file", "cache/football/calibration.json")
        cal = Calibrator(path=cal_path)
        # M3 (2026-09-02): the pooled guard follows config (min_samples 100,
        # same as league_min_samples) unless overridden on the command line.
        _min_samples = (
            int(args.min_samples) if args.min_samples is not None
            else int(cal_cfg.get("min_samples", 200))
        )
        report = cal.refresh_from_log(log_path, min_samples=_min_samples)
        report["min_samples"] = _min_samples
        report["file"] = str(cal_path)
        # D2 (2026-08-17): also re-fit EVERY per-league calibration file
        # (incl. dynamic ``dyn:`` leagues) from the same live log -- per-
        # league files were previously only seeded from football-data.co.uk
        # history, so unregistered leagues could never leave the
        # uncalibrated_league cap. Same skip/backup/regression discipline;
        # leagues without enough settled pairs simply do not appear.
        per_league = refresh_leagues_from_log(
            log_path,
            cal_dir=cal_path.parent,
            min_samples=int(
                cal_cfg.get("league_min_samples", 400)
            ),
        )
        report["leagues_refreshed"] = len(per_league)
        report["leagues"] = per_league
        return report
    if args.mode == "audit":
        # TODO-14: one-command leakage audit for CI/health. Needs fixtures;
        # prefer an explicit --fixtures JSON, else any cached backtest JSON.
        from .backtest import load_fixtures_from_json
        from .leakage_audit import audit_replay

        fixtures_path = args.fixtures
        if not fixtures_path:
            cache_dir = ROOT / (cfg.get("cache_dir") or "cache/football") / "backtest"
            if cache_dir.is_dir():
                for f in sorted(cache_dir.glob("*.json")):
                    try:
                        fx = load_fixtures_from_json(f)
                        if fx:
                            fixtures_path = str(f)
                            break
                    except Exception:  # noqa: BLE001 -- try next cache file
                        continue
        if not fixtures_path:
            return {
                "error": "tidak ada fixtures lokal untuk audit; jalankan `runner audit --fixtures <json>` "
                          "(atau bangun cache dulu via validate/backtest)",
            }
        audit = audit_replay(load_fixtures_from_json(fixtures_path))
        audit["fixtures_file"] = str(fixtures_path)
        return audit
    return {"error": f"mode '{args.mode}' tidak dikenal"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes-football")
    sub = parser.add_subparsers(dest="mode", required=True)

    top = sub.add_parser("top", help="top N match pada tanggal/liga")
    top.add_argument("--date", default=None, help="YYYY-MM-DD, default hari ini WIB")
    top.add_argument("--days", type=int, default=1,
                     help="berapa hari WIB ke depan (default 1 = hari ini; 2 = +dini hari besok)")
    top.add_argument("--leagues", default=None, help="comma-separated, default semua")
    top.add_argument("--top-n", type=int, default=5, dest="top_n")

    cmp = sub.add_parser("compare", help="banding 2 tim")
    cmp.add_argument("--home", required=True)
    cmp.add_argument("--away", required=True)
    cmp.add_argument("--league", default="EPL")

    ana = sub.add_parser("analyse", help="analisa match spesifik via keyword")
    ana.add_argument("--league", default=None, help="liga keyword (e.g. 'liga portugal', 'ucl'); opsional -- tanpa liga, liga di-detect dari fixture")
    ana.add_argument("--home", required=True)
    ana.add_argument("--away", required=True)
    for _mode, _help in (
        ("livescore", "cari match via LiveScore (hari ini/besok) lalu pipeline analisa penuh"),
        ("flashscore", "cari match via Flashscore (hari ini/besok) lalu pipeline analisa penuh"),
    ):
        src = sub.add_parser(_mode, help=_help)
        src.add_argument("--league", default=None, help="liga keyword (e.g. 'laliga', 'epl'); opsional -- tanpa liga, liga di-detect dari fixture")
        src.add_argument("--home", required=True)
        src.add_argument("--away", required=True)

    det = sub.add_parser("detect", help="cari liga dari nama tim tanpa keyword liga")
    det.add_argument("--home", required=True)
    det.add_argument("--away", required=True)
    det.add_argument("--date", default=None, help="YYYY-MM-DD, default hari ini")

    sts = sub.add_parser("stats", help="statistik realisasi prediction log (hit rate, logloss, ROI, CLV)")
    sts.add_argument("--edge-threshold", type=float, default=0.02,
                     help="edge minimum untuk dihitung sebagai bet (default 0.02)")

    pt = sub.add_parser("paper-trade",
                        help="Phase 8: laporan kelulusan segmen (ROI+CLV) untuk go-live")

    cr = sub.add_parser("clv-report",
                        help="Phase 0.4: laporan CLV per segmen (liga x market x timing) -> reports/clv_segments_<date>.json")
    cr.add_argument("--date", default=None, help="label tanggal laporan (default: hari ini)")
    cr.add_argument("--out-dir", default=None, help="direktori output (default: reports/)")
    cr.add_argument("--edge-threshold", type=float, default=0.02,
                    help="edge minimum untuk dihitung sebagai bet (default 0.02)")

    ba = sub.add_parser("bucket-audit",
                        help="Phase 5.4: audit edge-bucket vs ROI terhadap harga CLOSING -> reports/edge_buckets_<date>.json")
    ba.add_argument("--date", default=None, help="label tanggal laporan (default: hari ini)")
    ba.add_argument("--out-dir", default=None, help="direktori output (default: reports/)")

    vm = sub.add_parser("validate-multileague",
                        help="Phase 4.1: validasi walk-forward per liga target -> reports/validation_multileague_<date>.json")
    vm.add_argument("--date", default=None, help="label tanggal laporan (default: hari ini)")
    vm.add_argument("--out-dir", default=None, help="direktori output (default: reports/)")
    vm.add_argument("--leagues", default=None,
                    help="comma-separated leagues untuk laporan (default: config validate.multileague_leagues)")

    op = sub.add_parser("odds-poll",
                        help="Plan B: poll odds untuk snapshot unsettled dalam jendela lookahead (hourly)")
    mv = sub.add_parser("movement-report",
                        help="Plan B: akurasi sinyal movement (steam side) atas settled matches")

    stl = sub.add_parser("settle", help="catat hasil match ke prediction log")
    stl.add_argument("--home", default=None)
    stl.add_argument("--away", default=None)
    stl.add_argument("--result", default=None, help="skor '2-1' (manual)")
    stl.add_argument("--league", default=None, help="disambiguasi snapshot (manual)")
    stl.add_argument("--date", default=None, help="YYYY-MM-DD; filter manual / tanggal auto")
    stl.add_argument("--closing-odds", default=None, help="closing 1X2: home,draw,away")
    stl.add_argument(
        "auto", nargs="?", const="auto", default=None,
        help="'auto' = sinkron otomatis dengan hasil riil (default bila --home/--away kosong)",
    )
    stl.add_argument(
        "--source", default=None,
        choices=("auto", "football-data", "livescore"),
        help="sumber hasil untuk settle auto: livescore (primary), football-data, atau auto (livescore lalu football-data)",
    )
    stl.add_argument(
        "--dedupe", action="store_true",
        help="hapus settle rows duplikat (keep 1 per canonical match) dari prediction log",
    )
    stl.add_argument(
        "--best-pick", action="store_true",
        help="evaluasi BEST PICK tersimpan vs hasil settle (hit-rate/ROI per market)",
    )
    stl.add_argument(
        "--log", default=None,
        help="prediction log lain (mis. baseline/predictions_vps.jsonl hasil sync VPS); default cfg.prediction_log.file",
    )

    crf = sub.add_parser("calib-refresh",
                         help="TODO-02: re-fit kalibrasi dari prediction log (pre-match prob of settled outcomes)")
    crf.add_argument("--min-samples", type=int, default=None,
                     help="minimal pasangan settled untuk refit (default: models.calibration.min_samples)")

    aud = sub.add_parser("audit", help="TODO-14: leakage audit + source overview satu perintah")
    aud.add_argument("--fixtures", default=None, help="fixtures JSON lokal (default: cache/backtest terpasang)")

    co = sub.add_parser("cache-odds", help="TODO-07: bangun cache odds historis multi-liga (network)")
    co.add_argument("--leagues", default=None, help="comma-separated league keys (default: config.backtest.leagues)")
    co.add_argument("--seasons", default=None, help="comma-separated seasons, e.g. 2024-2025,2025-2026")

    sl = sub.add_parser("seed-league",
                        help="seed per-league calibration + merged elo.json dari football-data.co.uk (network)")
    sl.add_argument("--leagues", default=None, help="comma-separated league keys (default: config.leagues)")
    sl.add_argument("--seasons", default=None,
                    help="comma-separated seasons, e.g. 2022-2023,2023-2024,2024-2025,2025-2026")

    osn = sub.add_parser("odds-snapshot", help="catat odds 1X2 pada timing tertentu (PHASE 32-33)")
    osn.add_argument("--match-id", default=None,
                     help="eksplisit; default resolve via --home/--away dari snapshot unsettled")
    osn.add_argument("--home", default=None)
    osn.add_argument("--away", default=None)
    osn.add_argument("--league", default=None, help="disambiguasi lookup fuzzy")
    osn.add_argument("--timing", required=True, choices=("T-24h", "T-6h", "T-1h", "T-15m", "T-0h"),
                     help="kapan sebelum kickoff harga dicatat")
    osn.add_argument("--odds", required=True, help="1X2: home,draw,away")
    osn.add_argument("--bookmakers", type=int, default=None)
    osn.add_argument("--sources", default=None, help="comma-separated source labels")

    bst = sub.add_parser("best", help="1 prediction terbaik dari match liga hari ini/dini hari")
    bst.add_argument("--league", required=True, help="liga keyword (e.g. 'epl', 'ucl')")
    bst.add_argument("--date", default=None, help="YYYY-MM-DD, default hari ini + besok WIB")

    bgm = sub.add_parser("bestgoalmatch", help="match dengan potensi gol tinggi (banjir gol) hari ini")
    bgm.add_argument("--league", default=None, help="filter satu liga (opsional, default semua)")
    bgm.add_argument("--date", default=None, help="YYYY-MM-DD, default hari ini WIB")

    ngc = sub.add_parser(
        "nowgoal-check",
        help="diagnosa koneksi NowGoal: reachable / blocked (Trustpositif) / unreachable",
    )
    ngc.add_argument("--match-id", default=None, help="uji odds untuk match id tertentu")
    ngc.add_argument("--date", default=None, help="YYYY-MM-DD jadwal yang dicek (default hari ini UTC)")
    ngc.add_argument("--mirrors", default=None,
                     help="comma-separated base URL alternatif (default nowgoal6/nowgoal7)")

    trm = sub.add_parser("train-model",
                         help="ML: train + walk-forward eval (features port ProphitBet)")
    trm.add_argument("--model", default=None, choices=("lr", "rf", "xgb", "auto"),
                     help="model; auto = pilih terbaik via walk-forward")
    trm.add_argument("--target", default="result", choices=("result", "over-under"),
                     help="target klasifikasi")
    trm.add_argument("--folds", type=int, default=5, help="jumlah fold walk-forward kronologis")
    trm.add_argument("--sampler", default="smote", choices=("smote", "nearmiss", "none"),
                     help="penanganan kelas imbalanced (1X2 draw jarang)")
    trm.add_argument("--tune", type=int, default=0,
                     help="jumlah trial Optuna untuk tuning hyperparameter (0 = off)")
    trm.add_argument("--leagues", default=None,
                     help="comma-separated league keys; 'all' = semua cache historis")

    prm = sub.add_parser("predict-model",
                         help="ML: prediksi fixture 1X2 + O/U 2.5 dari model terlatih")
    prm.add_argument("--date", default=None, help="YYYY-MM-DD, default besok WIB")
    prm.add_argument("--leagues", default=None,
                     help="comma-separated league keys; 'all' = semua cache historis")

    man = sub.add_parser("ml-analysis",
                         help="ML: diagnostik fitur sekali-off (boruta/correlation/variance/coefficients)")
    man.add_argument("--league", default=None, help="league key (default semua cache)")
    man.add_argument("--metric", default="all",
                     choices=("all", "boruta", "correlation", "variance", "coefficients"))

    return parser.parse_args(argv)


def _suppress_library_logs() -> None:
    """Silence third-party (soccerdata/RichHandler) logging to stdout.

    SoccerData wires a RichHandler at the root logger that streams formatted
    INFO banners to stdout, which would corrupt the JSON contract we expose
    to the Discord bot. Strip every non-stderr handler from the root logger
    after import time.
    """
    import logging
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler is logging.lastResort:
            continue
        stream = getattr(handler, "stream", None)
        if stream is None or stream is sys.stderr:
            continue
        root.removeHandler(handler)


def _strip_legacy_confidence(payload: dict[str, Any]) -> None:
    """Addendum v1.1 Section 2/5: strip legacy confidence fields at the
    serialization boundary.

    ``confidence`` (legacy 0-1), ``signal_strength`` and ``decisiveness`` are
    still computed internally (grade gates / prediction log / similar-signal
    buckets), but NONE of them may appear in the serialized user-facing
    payload — enforced here structurally (recursively, so the ``best``
    winner payload is covered too), not by trusting every code path to
    remember to omit them. The legacy 0-100 ``signal`` score is no longer
    rendered by any formatter, so it is dropped as well.
    """
    def _clean_pred(pred: Any) -> None:
        if isinstance(pred, dict):
            for key in ("confidence", "signal_strength", "decisiveness"):
                pred.pop(key, None)

    _clean_pred(payload.get("prediction"))
    winner = payload.get("winner")
    if isinstance(winner, dict):
        _clean_pred(winner.get("prediction"))
        winner.pop("signal", None)
    payload.pop("signal", None)
    for key in ("matches", "candidates"):
        items = payload.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item.pop("signal", None)


def _emit(obj: dict) -> None:
    """Emit single-line JSON to stderr (the bot parses stderr, not stdout).

    Stderr is reserved exclusively for our payload. Third-party libs are
    silenced via ``_silence_root_handlers`` and ``contextlib.redirect_stdout``,
    so stderr stays clean across all paths. The lock keeps the watchdog
    thread from interleaving with the main thread's write on deadline races.

    Addendum v1.1 Section 5: the user-facing confidence block is validated
    against ``CONFIDENCE_ALLOWLIST`` before serialization — an unknown field
    fails the request (ValueError) instead of silently passing through.
    """
    raw = obj.get("raw") if isinstance(obj, dict) else None
    if isinstance(raw, dict):
        conf = raw.get("confidence")
        if isinstance(conf, dict):
            extra = set(conf) - set(CONFIDENCE_ALLOWLIST)
            if extra:
                raise ValueError(
                    "confidence block has fields outside allowlist: "
                    + ", ".join(sorted(extra))
                )
    payload = json.dumps(obj, ensure_ascii=False)
    with _EMIT_LOCK:
        sys.stderr.write("RJSON_START " + payload + " RJSON_END\n")
        sys.stderr.flush()


def _arm_deadline(seconds: float) -> None:
    """Hard-exit the runner after `seconds` with a clean JSON error.

    The event loop can be blocked by synchronous code that asyncio deadlines
    cannot interrupt (e.g. a seleniumbase driver-quit stuck in a timeout-less
    socket.connect inside a worker thread), so a watchdog thread calling
    ``os._exit`` is the only reliable kill switch. The bot kills us at 380s
    anyway (SUBPROCESS_TIMEOUT) -- this guarantees a parseable JSON error
    reaches it first, so the user always gets a reply.
    """
    import threading

    def _die() -> None:
        _emit({"error": f"runner deadline {seconds:g}s terlampaui (provider lambat/terblokir)"})
        os._exit(2)

    timer = threading.Timer(seconds, _die)
    timer.daemon = True
    timer.start()


_EMIT_LOCK = threading.Lock()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    import logging
    # HERMES_LOG_LEVEL=INFO reveals per-phase timing (phase x: Ns) when
    # diagnosing which provider pushes the runner past its deadline.
    _level = getattr(logging, os.getenv("HERMES_LOG_LEVEL", "ERROR").upper(), logging.ERROR)
    logging.basicConfig(level=_level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    _silence_root_handlers()
    # Some third-party code (seleniumbase driver teardown) opens sockets with
    # NO timeout; a half-open port then hangs forever instead of failing. A
    # process-wide default keeps every such connect bounded (libraries that
    # set their own explicit timeouts are unaffected).
    import socket as _socket
    _socket.setdefaulttimeout(15.0)
    discarded = io.StringIO()
    # Hard deadline a few seconds under the bot's subprocess timeout (380s) so
    # we always exit with a clean JSON error instead of being killed and
    # answered with a generic timeout. Tune via HERMES_RUNNER_DEADLINE.
    _arm_deadline(float(os.getenv("HERMES_RUNNER_DEADLINE", "340")))
    try:
        with contextlib.redirect_stdout(discarded):
            payload = asyncio.run(_run(args))
    except (OddsApiError, FootballDataError) as exc:
        _emit({"error": str(exc)})
        return 2
    except BaseException as exc:  # noqa: BLE001 -- the bot must ALWAYS get a
        # parseable reply. ``asyncio.CancelledError`` is a BaseException since
        # 3.8, so ``except Exception`` let it (and KeyboardInterrupt) escape
        # without emitting RJSON -- the subprocess died with a bare traceback
        # and the bot answered "output bukan JSON" or nothing at all. Emit
        # FIRST: if a watchdog / external kill lands mid-log, stderr already
        # carries the reply.
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        logger.exception("runner failure")
        return 2
    finally:
        # Release flashscore/understat browsers so headless Chrome is not
        # orphaned after every run (zombies piled up and slowed the machine).
        for stats in _STATS_REGISTRY:
            try:
                stats.close()
            except Exception:  # noqa: BLE001 -- cleanup must never break the reply
                pass

    rendered_full: dict[str, Any] | None = None
    if args.mode == "top":
        rendered = format_top(payload)
    elif args.mode == "best":
        # Winner card is the compact summary; the FULL best report (with the
        # full winner analysis) rides along under ``render_full`` for Copy.
        rendered = format_best(payload)
        rendered_full = format_best(payload, compact_winner=False)
    elif args.mode == "bestgoalmatch":
        rendered = format_best_goal(payload)
    elif args.mode in ("analyse", "livescore", "flashscore"):
        # Primary reply is the clean MARKET SIGNAL card; the 📋 Detail button
        # serves the slightly richer (still debug-free) signal detail. The
        # match-source commands reuse the identical analyse output format.
        rendered = format_market_signal(payload)
        rendered_full = format_signal_detail(payload)
    elif args.mode == "settle":
        rendered = format_settle(payload)
    elif args.mode == "stats":
        rendered = format_stats(payload)
    elif args.mode == "odds-snapshot":
        rendered = format_odds_snapshot(payload)
    elif args.mode in ("calib-refresh", "audit", "cache-odds", "detect", "nowgoal-check",
                       "train-model", "predict-model", "ml-analysis", "paper-trade",
                       "odds-poll", "movement-report", "seed-league",
                       "clv-report", "bucket-audit", "validate-multileague"):
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        rendered = format_compare(payload)

    _strip_legacy_confidence(payload)
    out: dict[str, Any] = {"render": rendered, "raw": payload}
    if rendered_full is not None:
        out["render_full"] = rendered_full
    _emit(out)
    # Skip interpreter shutdown: a hung browser-fallback thread (non-daemon,
    # parked in the thread pool) would otherwise stall process exit until the
    # watchdog fires, and the bot would reply "timeout" despite a valid
    # result. stdout is DEVNULL (bot side) / discarded, stderr is flushed by
    # _emit, so nothing is lost.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
