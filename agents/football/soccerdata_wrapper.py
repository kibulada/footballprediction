"""soccerdata library wrapper.

soccerdata (https://github.com/probberechts/soccerdata) is a Python library
that scrapes 8 sources: ClubElo, ESPN, FBref, MatchHistory, SoFIFA, Sofascore,
Understat, WhoScored.

We use:
  - FBref   -> schedule + team match stats (form, GF, GA, xG)
  - Understat -> league-level xG data

soccerdata is sync. We run all calls via asyncio.to_thread to keep the
Discord bot event loop non-blocking.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

# Disable soccerdata's auto-install of PyAutoGUI for CAPTCHA solving; that
# path silently spawns a headless Chrome in this environment which we want
# to avoid. Capture must happen before soccerdata is imported anywhere.
os.environ.setdefault("SOCCERDATA_NOCACHE", "false")
os.environ.setdefault("SOCCERDATA_NOSTORE", "false")
os.environ.setdefault("SOCCERDATA_MAXAGE", "3600")
os.environ.setdefault("SOCCERDATA_LOGLEVEL", "WARNING")

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


LEAGUE_MAP: dict[str, dict[str, str]] = {
    "EPL": {"fbref": "ENG-Premier League"},
    "LaLiga": {"fbref": "ESP-La Liga"},
    "Serie A": {"fbref": "ITA-Serie A"},
    "Bundesliga": {"fbref": "GER-Bundesliga"},
    "Ligue 1": {"fbref": "FRA-Ligue 1"},
    "Eredivisie": {"fbref": "NED-Eredivisie"},
    "Primeira Liga": {"fbref": "POR-Liga Portugal"},
    "UCL": {"fbref": "UEFA-Champions League"},
    "UEL": {"fbref": "UEFA-Europa League"},
    "UECL": {"fbref": "UEFA-Europa Conference League"},
    "MLS": {"fbref": "USA-MLS"},
    "EFL Championship": {"fbref": "ENG-Championship"},
}


def fbref_code(league_key: str) -> str | None:
    """Map bot league key to soccerdata FBref code, or None if unsupported."""
    meta = LEAGUE_MAP.get(league_key)
    return meta.get("fbref") if meta else None


def league_key_from_display(display: str) -> str | None:
    """Inverse lookup: 'La Liga' -> 'LaLiga', 'EPL' -> 'EPL'."""
    target = display.lower().strip().replace(" ", "")
    for key in LEAGUE_MAP:
        if key.lower().replace(" ", "") == target:
            return key
    return None


_VALID_FBREF_LEAGUES: set[str] | None = None


def _ensure_valid_fbref_leagues() -> set[str]:
    """Lazily probe soccerdata.FBref.available_leagues() to filter invalid codes."""
    global _VALID_FBREF_LEAGUES
    if _VALID_FBREF_LEAGUES is not None:
        return _VALID_FBREF_LEAGUES
    leagues: set[str] = set()
    try:
        import contextlib as _cl
        import io as _io
        import soccerdata as sd
        with _cl.redirect_stdout(_io.StringIO()), _cl.redirect_stderr(_io.StringIO()):
            leagues = set(sd.FBref.available_leagues())
    except Exception:
        leagues = set()
    _VALID_FBREF_LEAGUES = leagues
    return _VALID_FBREF_LEAGUES


def _disable_captcha_solver() -> None:
    """Disable soccerdata's Selenium/Chrome-based scraping path entirely.

    SoccerData's FBref/Understat scrapers rely on ``seleniumbase`` (seleniumbase
    opens a real Chrome instance). On networks with Cloudflare "I'm Under
    Attack" mode, every request triggers a captcha, which causes the library to
    auto-install PyAutoGUI and spawn a headless Chrome that hangs our runner
    subprocess for minutes. We patch the entire Selenium path so that the
    library uses its HTTP fallback (curl_cffi) instead, and any captcha
    detection immediately aborts without opening a browser.

    Patches applied to ``SeleniumRequestMixin`` (and module-level equivalents):

    * ``__init__`` -> sets ``headless=True`` and forces Selenium off.
    * ``_init_webdriver`` -> raises ``Exception("SELENIUM_DISABLED")`` so the
      library never instantiates a Chrome instance.
    * ``solve_captcha`` -> no-op (returns immediately, no PyAutoGUI install).
    * ``_is_captcha_present`` -> always ``True`` so the retry loop aborts.
    * ``_download_and_save`` -> raises ``Exception("CAPTCHA_BLOCKED")`` so our
      wrapper catches and falls back.
    * ``_request`` / ``_validate_page`` / ``read_page`` -> raise the same
      captcha-blocked exception.
    """
    try:
        from soccerdata import _common as _sc  # type: ignore[import-not-found]
    except Exception:
        return
    if _sc is None:
        return

    def _no_solve(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        return None

    def _captcha_present(self: Any, *args: Any, **kwargs: Any) -> bool:  # noqa: ARG001
        return True

    def _download_no_solve(
        self: Any,
        url: str,
        filepath: Any = None,
        var: Any = None,
    ) -> None:
        raise Exception("CAPTCHA_BLOCKED")

    def _no_webdriver(self: Any) -> None:
        raise Exception("SELENIUM_DISABLED")

    def _init_disabled(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        # Mirror what SeleniumRequestMixin.__init__ does for the curl_cffi
        # fallback: store the same attributes (rate_limit, no_cache, ...) but
        # never touch a real WebDriver. We re-use ``__init__`` of the underlying
        # concrete class via duck typing: the curl_cffi path requires only
        # ``self.rate_limit`` and ``self.session``-like attrs which are set by
        # the parent ``BaseReader.__init__`` (also subclassed here).
        self.rate_limit = float(getattr(self, "rate_limit", 0))
        self.no_cache = bool(getattr(self, "no_cache", False))
        self.no_store = bool(getattr(self, "no_store", False))
        self.max_delay = float(getattr(self, "max_delay", 0))
        self.path_to_browser = None
        self._driver = None  # never instantiated

    mixin = getattr(_sc, "SeleniumRequestMixin", None)
    if mixin is not None:
        # Disable WebDriver instantiation entirely.
        if hasattr(mixin, "_init_webdriver"):
            mixin._init_webdriver = _no_webdriver  # type: ignore[attr-defined]
        if hasattr(mixin, "solve_captcha"):
            mixin.solve_captcha = _no_solve  # type: ignore[attr-defined]
        if hasattr(mixin, "_is_captcha_present"):
            mixin._is_captcha_present = _captcha_present  # type: ignore[attr-defined]
        if hasattr(mixin, "_download_and_save"):
            mixin._download_and_save = _download_no_solve  # type: ignore[attr-defined]
        if hasattr(mixin, "_request"):
            mixin._request = _download_no_solve  # type: ignore[attr-defined]
        if hasattr(mixin, "_validate_page"):
            mixin._validate_page = _download_no_solve  # type: ignore[attr-defined]
        if hasattr(mixin, "read_page"):
            mixin.read_page = _download_no_solve  # type: ignore[attr-defined]

    if hasattr(_sc, "_solve_captcha"):
        _sc._solve_captcha = lambda *a, **k: None  # type: ignore[attr-defined]


_disable_captcha_solver()


def fbref_code(league_key: str) -> str | None:
    """Return FBref code for the given bot league key, or None if unsupported.

    Validates against the league list reported by
    `soccerdata.FBref.available_leagues()`. Codes that are not present in
    the probe result yield None so callers skip the provider without
    triggering a `ValueError` at scrape time.
    """
    meta = LEAGUE_MAP.get(league_key)
    if not meta:
        return None
    code = meta.get("fbref")
    if not code:
        return None
    valid = _ensure_valid_fbref_leagues()
    if not valid:
        return None
    if code in valid:
        return code
    if "Big 5 European Leagues Combined" in valid:
        mapped = {
            "ENG-Premier League": "ENG-Premier League",
            "ESP-La Liga": "ESP-La Liga",
            "GER-Bundesliga": "GER-Bundesliga",
            "ITA-Serie A": "ITA-Serie A",
            "FRA-Ligue 1": "FRA-Ligue 1",
        }
        if code in mapped and mapped[code] in valid:
            return code
    return None


def current_season_code(today: datetime | None = None) -> str:
    """Return FBref season code (e.g. '2026-2027') for the given date.

    Football seasons in the northern hemisphere run Aug->May. A match after
    August belongs to the upcoming season.
    """
    today = today or datetime.now(timezone.utc)
    if today.month >= 7:
        start = today.year
    else:
        start = today.year - 1
    return f"{start}-{start + 1}"


def previous_season_code(season_code: str) -> str:
    start = int(season_code.split("-")[0])
    return f"{start - 1}-{start}"


class SoccerDataWrapper:
    def __init__(
        self,
        data_dir: str = "cache/soccerdata",
        proxy: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        self._proxy = proxy
        self._fbref_dead = False  # set once FBref is unreachable/blocked
        valid = _ensure_valid_fbref_leagues()
        self.available_leagues = {
            key for key in LEAGUE_MAP
            if fbref_code(key) is not None
        } if valid else set(LEAGUE_MAP.keys())

    def supports_league(self, league_key: str) -> bool:
        return league_key in self.available_leagues

    async def read_team_form(
        self, league_key: str, team_name: str, limit: int = 10
    ) -> dict[str, Any] | None:
        """Fetch last N matches for a team from FBref. Returns form dict.

        Live-time budget: on this network FBref is Cloudflare-blocked, so a
        full 20s timeout per attempt was eating ~40s of the runner's 85s
        deadline. Attempts are capped at 8s; a network/parse failure marks
        the source dead for the rest of the process (subsequent reads return
        None instantly) and skips the previous-season retry -- re-probing a
        dead source only wastes time. Backtest paths (read_league_schedule)
        are unaffected: they use their own timeouts and never touch the flag.
        """
        if not league_key or not team_name:
            return None
        if not self.supports_league(league_key):
            return None
        if self._fbref_dead:
            return None
        code = fbref_code(league_key)
        if not code:
            return None

        async def _run_once(season: str):
            season_id = season.split("-")[0][-2:] + season.split("-")[1][-2:]

            def _sync_fetch() -> pd.DataFrame | None:
                import soccerdata as sd
                kwargs = {}
                if self._proxy:
                    kwargs["proxy"] = self._proxy
                fbref = sd.FBref(code, season_id, **kwargs)
                try:
                    schedule = fbref.read_schedule(force_cache=True)
                except Exception as exc:
                    logger.warning("soccerdata FBref schedule failed: %s", exc)
                    return None
                if schedule is None or schedule.empty:
                    return None
                df = schedule.copy()
                df.columns = [str(c).lower() for c in df.columns]
                return df

            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_sync_fetch), timeout=8.0
                )
            except asyncio.TimeoutError:
                logger.warning("soccerdata FBref schedule timeout (8s); marking FBref dead")
                return None
            except Exception as exc:
                logger.warning("soccerdata FBref schedule error: %s", exc)
                return None

        df = await _run_once(current_season_code())
        if df is None:
            # Network/parse failure (not an empty season): don't re-probe a
            # dead source -- the second attempt just costs another 8s.
            self._fbref_dead = True
            return None
        if len(df) == 0:
            df = await _run_once(previous_season_code(current_season_code()))
            if df is None:
                self._fbref_dead = True
                return None

        if df is None or len(df) == 0:
            return None

        date_col = next((c for c in df.columns if "date" in c), None)
        score_col_global = next((c for c in df.columns if c in ("score", "result")), None)
        if score_col_global is not None:
            df = df[df[score_col_global].notna() & ~df[score_col_global].astype(str).str.lower().str.match(r"^(nan|na|<na>|none|\-1)$")]
        if date_col:
            try:
                df["_dt"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
                df = df.sort_values("_dt", ascending=False)
            except Exception as exc:
                logger.debug("soccerdata date sort skipped: %s", exc)

        team_col = next((c for c in df.columns if c in ("home", "hometeam", "home_team")), None)
        if team_col is None:
            team_col = next((c for c in df.columns if "home" in c and "team" in c), None)
        if team_col is None:
            team_col = next((c for c in df.columns if "team" in c), None)
        away_col = next((c for c in df.columns if c in ("away", "awayteam", "away_team")), None)
        if away_col is None:
            away_col = next((c for c in df.columns if "away" in c and "team" in c), None)

        hg_col = None
        ag_col = None
        score_col = next((c for c in df.columns if c in ("score", "result")), None)
        if score_col is None:
            hg_col = next((c for c in df.columns if c in ("homegoals", "home_score", "home_goals")), None)
            ag_col = next((c for c in df.columns if c in ("awaygoals", "away_score", "away_goals")), None)
            if hg_col is None:
                hg_col = next((c for c in df.columns if "home" in c and ("goal" in c or "score" in c)), None)
            if ag_col is None:
                ag_col = next((c for c in df.columns if "away" in c and ("goal" in c or "score" in c)), None)

        if not (team_col and away_col):
            logger.warning("soccerdata columns unresolved (team cols), cols=%s", list(df.columns))
            return None

        # 2026-09-02 (wrong-team post-mortem): token-level identity with an
        # ambiguity guard instead of raw ``in`` ("inter" matched "Winterthur"
        # and the substring also decided the home/away side).
        from .team_identity import match_side as _match_side

        _sides = df.apply(
            lambda r: _match_side(team_name, str(r.get(team_col, "")), str(r.get(away_col, ""))),
            axis=1,
        )
        team_mask = _sides.notna()
        matches = df[team_mask].copy()
        matches["_side"] = _sides[team_mask]
        if matches is None or len(matches) == 0:
            return None
        today_utc = pd.Timestamp.now(tz="UTC")
        matches = matches.head(limit)

        # FBref schedules carry per-match expected goals (home_xg/away_xg after
        # the lowercase rename above). Extracting them here lets the live xG
        # history fall back to FBref for leagues understat does not cover
        # (non-big-5), using the SAME rolling construction the model validates
        # against. A missing xG column leaves the form feature unchanged.
        home_xg_col = next(
            (c for c in df.columns if c in ("home_xg", "xg_home", "homexg")), None
        )
        away_xg_col = next(
            (c for c in df.columns if c in ("away_xg", "xg_away", "awayxg")), None
        )

        results: list[str] = []
        gf_list: list[int] = []
        ga_list: list[int] = []
        xg_for_list: list[float] = []
        xg_against_list: list[float] = []
        home_w = home_d = home_l = 0
        away_w = away_d = away_l = 0
        for _, row in matches.iterrows():
            is_home = row.get("_side") == "home"
            try:
                if score_col is not None:
                    raw_score = str(row.get(score_col))
                    if raw_score in ("nan", "<NA>", "", "None"):
                        continue
                    parts = (
                        raw_score.replace(":", "-")
                        .replace(" ", "")
                    )
                    import re as _re
                    parts = _re.split(r"[-–?−]", parts)
                    if len(parts) != 2:
                        continue
                    hg, ag = int(parts[0]), int(parts[1])
                else:
                    hg = int(row.get(hg_col))
                    ag = int(row.get(ag_col))
            except (TypeError, ValueError):
                continue
            if date_col and "_dt" in row and not pd.isna(row["_dt"]) and row["_dt"] > today_utc:
                continue
            gf = hg if is_home else ag
            ga = ag if is_home else hg
            gf_list.append(gf)
            ga_list.append(ga)
            if home_xg_col and away_xg_col:
                try:
                    hxg = float(row.get(home_xg_col))
                    axg = float(row.get(away_xg_col))
                except (TypeError, ValueError):
                    hxg = axg = None
                if hxg is not None and axg is not None:
                    xg_for_list.append(hxg if is_home else axg)
                    xg_against_list.append(axg if is_home else hxg)
            if hg == ag:
                results.append("D")
                if is_home:
                    home_d += 1
                else:
                    away_d += 1
            elif (hg > ag) == is_home:
                results.append("W")
                if is_home:
                    home_w += 1
                else:
                    away_w += 1
            else:
                results.append("L")
                if is_home:
                    home_l += 1
                else:
                    away_l += 1
        if not results:
            return None
        out = {
            "sequence": "-".join(results),
            "gf_avg": (sum(gf_list) / len(gf_list)),
            "ga_avg": (sum(ga_list) / len(ga_list)),
            "home": {"w": home_w, "d": home_d, "l": home_l},
            "away": {"w": away_w, "d": away_d, "l": away_l},
            "sample_size": len(results),
            "source": "soccerdata_fbref",
        }
        if xg_for_list and xg_against_list:
            out["xg_for_avg"] = sum(xg_for_list) / len(xg_for_list)
            out["xg_against_avg"] = sum(xg_against_list) / len(xg_against_list)
            out["xg_sample_size"] = len(xg_for_list)
        return out

    async def read_league_xg(self, league_key: str, season: str | None = None) -> dict[str, Any] | None:
        league_to_understat = {
            "EPL": "EPL",
            "LaLiga": "La Liga",
            "Serie A": "Serie A",
            "Bundesliga": "Bundesliga",
            "Ligue 1": "Ligue 1",
        }
        if not self.supports_league(league_key):
            return None
        league_name = league_to_understat.get(league_key)
        if not league_name:
            return None
        season = season or current_season_code()

        def _sync_fetch():
            import soccerdata as sd
            kwargs = {}
            if self._proxy:
                kwargs["proxy"] = self._proxy
            try:
                season_id = int(season.split("-")[0])
                understat = sd.Understat(league=league_name, season=season_id, **kwargs)
                df = understat.read_league_table()
            except Exception as exc:
                logger.warning("Understat fetch failed: %s", exc)
                return None
            if df is None or df.empty:
                return None
            return df

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_sync_fetch), timeout=20.0
            )
        except asyncio.TimeoutError:
            logger.warning("Understat timeout (20s)")
            return None
        except Exception as exc:
            logger.warning("Understat error: %s", exc)
            return None

    async def read_h2h(
        self, league_key: str, team_a: str, team_b: str, limit: int = 5
    ) -> dict[str, Any] | None:
        """Build H2H aggregate from FBref schedule (last N finished between two teams)."""
        if not league_key or not team_a or not team_b:
            return None
        if not self.supports_league(league_key):
            return None
        if self._fbref_dead:
            return None
        code = fbref_code(league_key)
        if not code:
            return None

        async def _load_season(season: str):
            season_id = season.split("-")[0][-2:] + season.split("-")[1][-2:]

            def _sync_fetch():
                import soccerdata as sd
                kwargs = {}
                if self._proxy:
                    kwargs["proxy"] = self._proxy
                fbref = sd.FBref(code, season_id, **kwargs)
                try:
                    schedule = fbref.read_schedule(force_cache=True)
                except Exception as exc:
                    logger.warning("FBref schedule for h2h failed: %s", exc)
                    return None
                if schedule is None or schedule.empty:
                    return None
                df = schedule.copy()
                df.columns = [str(c).lower() for c in df.columns]
                return df
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_sync_fetch), timeout=8.0
                )
            except asyncio.TimeoutError:
                logger.warning("FBref h2h schedule timeout (8s); marking FBref dead")
                return None
            except Exception as exc:
                logger.warning("FBref h2h schedule error: %s", exc)
                return None

        df = await _load_season(current_season_code())
        if df is None:
            self._fbref_dead = True
            return None
        if len(df) == 0:
            df = await _load_season(previous_season_code(current_season_code()))
            if df is None:
                self._fbref_dead = True
                return None
        if df is None or len(df) == 0:
            return None

        home_col = next((c for c in df.columns if c in ("home", "hometeam", "home_team")), None)
        away_col = next((c for c in df.columns if c in ("away", "awayteam", "away_team")), None)
        score_col = next((c for c in df.columns if c in ("score", "result")), None)
        hg_col = None
        ag_col = None
        if score_col is None:
            hg_col = next((c for c in df.columns if c in ("homegoals", "home_score", "home_goals")), None)
            ag_col = next((c for c in df.columns if c in ("awaygoals", "away_score", "away_goals")), None)
        date_col = next((c for c in df.columns if "date" in c), None)
        if not all([home_col, away_col]) or (score_col is None and (hg_col is None or ag_col is None)):
            return None

        # 2026-09-02: fixture identity by tokens, orientation carried per row.
        from .team_identity import same_fixture as _same_fixture

        _orient = df.apply(
            lambda r: _same_fixture(team_a, team_b, str(r.get(home_col, "")), str(r.get(away_col, ""))),
            axis=1,
        )
        mask = _orient.notna()
        rows = df[mask].copy()
        rows["_orient"] = _orient[mask]
        if rows is None or len(rows) == 0:
            return None
        if date_col:
            try:
                rows["_dt"] = pd.to_datetime(rows[date_col], errors="coerce", utc=True)
                rows = rows.sort_values("_dt", ascending=False)
            except Exception:
                pass
        rows = rows.head(limit)

        wins = draws = losses = 0
        for _, row in rows.iterrows():
            try:
                if score_col is not None:
                    raw = str(row.get(score_col))
                    parts = (
                        raw.replace(":", "-").replace(" ", "")
                    )
                    import re as _re_h2h
                    parts = _re_h2h.split(r"[-–?−]", parts)
                    if len(parts) != 2:
                        continue
                    hg, ag = int(parts[0]), int(parts[1])
                else:
                    hg = int(row.get(hg_col))
                    ag = int(row.get(ag_col))
            except (TypeError, ValueError):
                continue
            team_a_is_home = row.get("_orient") == "ordered"
            if hg == ag:
                draws += 1
            elif (hg > ag) == team_a_is_home:
                wins += 1
            else:
                losses += 1
        if wins == 0 and draws == 0 and losses == 0:
            return None
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "source": "soccerdata_fbref_h2h",
            "sample_size": len(rows),
        }

    async def read_league_schedule(
        self,
        league_key: str,
        season_codes: list[str] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Full league schedule rows (finished matches) for backtesting.

        Returns list of {date (ISO date), home, away, home_goals, away_goals,
        league, season} or None. Only finished matches (score present, date
        in the past) are returned.
        """
        if not league_key or not self.supports_league(league_key):
            return None
        code = fbref_code(league_key)
        if not code:
            return None
        season_codes = season_codes or [current_season_code(), previous_season_code(current_season_code())]
        today_utc = pd.Timestamp.now(tz="UTC") if pd is not None else None
        rows: list[dict[str, Any]] = []

        for season in season_codes:
            season_id = season.split("-")[0][-2:] + season.split("-")[1][-2:]

            def _sync_fetch():
                import soccerdata as sd
                kwargs = {}
                if self._proxy:
                    kwargs["proxy"] = self._proxy
                fbref = sd.FBref(code, season_id, **kwargs)
                try:
                    schedule = fbref.read_schedule(force_cache=True)
                except Exception as exc:
                    logger.warning("FBref schedule (backtest) failed: %s", exc)
                    return None
                if schedule is None or schedule.empty:
                    return None
                df = schedule.copy()
                df.columns = [str(c).lower() for c in df.columns]
                return df

            try:
                df = await asyncio.wait_for(
                    asyncio.to_thread(_sync_fetch), timeout=25.0
                )
            except asyncio.TimeoutError:
                logger.warning("FBref schedule (backtest) timeout for %s %s", league_key, season)
                continue
            except Exception as exc:
                logger.warning("FBref schedule (backtest) error: %s", exc)
                continue
            if df is None or len(df) == 0:
                continue

            home_col = next((c for c in df.columns if c in ("home", "hometeam", "home_team")), None)
            away_col = next((c for c in df.columns if c in ("away", "awayteam", "away_team")), None)
            date_col = next((c for c in df.columns if "date" in c), None)
            score_col = next((c for c in df.columns if c in ("score", "result")), None)
            hg_col = ag_col = None
            if score_col is None:
                hg_col = next((c for c in df.columns if c in ("homegoals", "home_score", "home_goals")), None)
                ag_col = next((c for c in df.columns if c in ("awaygoals", "away_score", "away_goals")), None)
                if hg_col is None:
                    hg_col = next((c for c in df.columns if "home" in c and ("goal" in c or "score" in c)), None)
                if ag_col is None:
                    ag_col = next((c for c in df.columns if "away" in c and ("goal" in c or "score" in c)), None)
            if not (home_col and away_col) or (score_col is None and (hg_col is None or ag_col is None)):
                logger.warning("FBref schedule (backtest) unresolved columns for %s", league_key)
                continue

            for _, row in df.iterrows():
                try:
                    if score_col is not None:
                        raw = str(row.get(score_col))
                        if raw.lower() in ("nan", "<na>", "none", "", "-1"):
                            continue
                        parts = _split_score(raw)
                        if parts is None:
                            continue
                        hg, ag = parts
                    else:
                        hg = int(row.get(hg_col))
                        ag = int(row.get(ag_col))
                except (TypeError, ValueError):
                    continue
                home = str(row.get(home_col, "")).strip()
                away = str(row.get(away_col, "")).strip()
                if not home or not away:
                    continue
                date_str = str(row.get(date_col, "")) if date_col else ""
                if date_str and today_utc is not None:
                    try:
                        dt = pd.to_datetime(date_str, errors="coerce", utc=True)
                        if pd.isna(dt) or dt > today_utc:
                            continue
                        date_iso = dt.date().isoformat()
                    except Exception:
                        date_iso = date_str[:10]
                else:
                    date_iso = date_str[:10]
                rows.append({
                    "date": date_iso,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "league": league_key,
                    "season": season,
                })
        return rows or None


def _split_score(raw: str) -> tuple[int, int] | None:
    """Parse '1-0', '1:0', '1 - 0', '1–0' into (home, away)."""
    import re

    parts = re.split(r"[-–?−:]", raw.replace(" ", ""))
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None
