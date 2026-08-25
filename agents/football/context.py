"""MatchContext + FeatureBuilder.

MatchContext is a leakage-safe, PRE-MATCH-ONLY snapshot of everything known
about a fixture before kickoff. Anything that could only be known after
kickoff (result, live xG, post-match stats) must NOT be placed here.

FeatureBuilder assembles a MatchContext from the already-fetched stats dict
produced by analyse.py, keeping the provider layer untouched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .timeutil import utc_now_iso


def input_hash(payload: dict[str, Any]) -> str:
    """Stable 16-hex hash of a canonical JSON snapshot (reproducibility)."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class MatchContext:
    league: str
    home: str
    away: str
    kickoff_utc: str | None = None
    as_of: str = field(default_factory=utc_now_iso)
    home_form: str | None = None
    away_form: str | None = None
    home_gf_avg: float | None = None
    home_ga_avg: float | None = None
    away_gf_avg: float | None = None
    away_ga_avg: float | None = None
    home_xg_for: float | None = None
    home_xg_against: float | None = None
    away_xg_for: float | None = None
    away_xg_against: float | None = None
    h2h: dict[str, int] | None = None  # {wins, draws, losses} from home perspective
    consensus_odds: dict[str, float] | None = None  # decimal {home, draw, away}
    market_totals: dict[str, dict[str, float]] | None = None
    # Raw recent scorelines per team, (gf, ga) per finished match, ordered
    # OLDEST -> NEWEST. When present, feature models apply time-decay
    # weighting (Dixon-Coles xi) instead of treating precomputed gf/ga
    # averages as equally weighted.
    home_recent_goals: list[tuple[int, int]] | None = None
    away_recent_goals: list[tuple[int, int]] | None = None
    form_samples: int = 0
    xg_samples: int = 0
    # Phase 1 (lineups/injuries into lambda): pre-match lineup evidence used
    # ONLY by the flag-gated ``lineup_weight`` correction in PoissonModel --
    # absent by default, so existing predictions are byte-identical until the
    # feature flag is on and the backtest DoD passes. ``lineup_ts`` enables
    # the Phase 1.3 leakage guard: a lineup fetched at/after kickoff is
    # rejected as a model input.
    lineup_home: list[str] | None = None       # confirmed/predicted starter names
    lineup_away: list[str] | None = None
    lineup_status: str | None = None           # "confirmed" | "predicted"
    lineup_ts: str | None = None
    lineup_source: str | None = None
    missing_home: list[str] | None = None      # missing/unsure player names
    missing_away: list[str] | None = None
    home_days_rest: float | None = None        # days since last finished match
    away_days_rest: float | None = None
    sources: list[str] = field(default_factory=list)
    # Per-field provenance/confidence metadata (optional; not a model feature).
    # Populated by the multi-source aggregation layer. Never used by the
    # prediction engine directly -- debugging / evaluation only.
    source_meta: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def input_hash(self) -> str:
        return input_hash(self.snapshot())

    @property
    def has_attack_defense(self) -> bool:
        return (
            all(
                v is not None
                for v in (self.home_gf_avg, self.home_ga_avg, self.away_gf_avg, self.away_ga_avg)
            )
            or bool(self.home_recent_goals and self.away_recent_goals)
        )

    @property
    def has_xg(self) -> bool:
        return all(
            v is not None
            for v in (
                self.home_xg_for, self.home_xg_against,
                self.away_xg_for, self.away_xg_against,
            )
        )

    @property
    def has_odds(self) -> bool:
        return bool(
            self.consensus_odds
            and all(self.consensus_odds.get(k, 0) > 0 for k in ("home", "draw", "away"))
        )


def _recent_goals(v: Any) -> list[tuple[int, int]] | None:
    """Sanitize raw recent scorelines to [(gf, ga), ...] oldest->newest."""
    if not isinstance(v, (list, tuple)) or not v:
        return None
    out: list[tuple[int, int]] = []
    for row in v:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        try:
            out.append((int(row[0]), int(row[1])))
        except (ValueError, TypeError):
            continue
    return out or None


def build_match_context(
    *,
    league: str,
    home: str,
    away: str,
    kickoff: str | None = None,
    stats: dict[str, Any] | None = None,
    odds: dict[str, Any] | None = None,
    sources: list[str] | None = None,
    source_meta: dict[str, Any] | None = None,
) -> MatchContext:
    """FeatureBuilder: assemble a MatchContext from analyse-style dicts.

    All values flow through unchanged; missing data stays missing (None),
    never fabricated or defaulted.
    """
    stats = stats or {}
    odds = odds or {}
    h2h = stats.get("h2h") or {}
    h2h_clean = None
    if h2h and (h2h.get("wins") or h2h.get("draws") or h2h.get("losses")):
        h2h_clean = {
            "wins": int(h2h.get("wins", 0)),
            "draws": int(h2h.get("draws", 0)),
            "losses": int(h2h.get("losses", 0)),
        }

    ctx = MatchContext(
        league=league,
        home=home,
        away=away,
        kickoff_utc=kickoff,
        home_form=stats.get("home_form") if stats.get("home_form") not in (None, "n/a") else None,
        away_form=stats.get("away_form") if stats.get("away_form") not in (None, "n/a") else None,
        home_gf_avg=_num(stats.get("home_gf_avg")),
        home_ga_avg=_num(stats.get("home_ga_avg")),
        away_gf_avg=_num(stats.get("away_gf_avg")),
        away_ga_avg=_num(stats.get("away_ga_avg")),
        home_xg_for=_num(stats.get("home_xg_for")),
        home_xg_against=_num(stats.get("home_xg_against")),
        away_xg_for=_num(stats.get("away_xg_for")),
        away_xg_against=_num(stats.get("away_xg_against")),
        home_recent_goals=_recent_goals(stats.get("home_recent_goals")),
        away_recent_goals=_recent_goals(stats.get("away_recent_goals")),
        h2h=h2h_clean,
        lineup_home=_lineup_side(stats, "home"),
        lineup_away=_lineup_side(stats, "away"),
        lineup_status=_str_or_none(stats.get("lineup_status")),
        lineup_ts=_str_or_none(stats.get("lineup_ts")),
        lineup_source=_str_or_none(stats.get("lineup_source")),
        missing_home=_str_list(stats.get("missing_home")),
        missing_away=_str_list(stats.get("missing_away")),
        home_days_rest=_num(stats.get("home_days_rest")),
        away_days_rest=_num(stats.get("away_days_rest")),
        consensus_odds=dict(odds.get("consensus") or {}) if odds.get("has_odds") else None,
        market_totals=dict(odds.get("totals") or {}),
        sources=sorted({s for s in (sources or []) if s}),
        source_meta=source_meta,
    )

    seqs = [s for s in (ctx.home_form, ctx.away_form) if s]
    if seqs:
        # Count actual result tokens (W/D/L), not the string length (dashes).
        ctx.form_samples = min(sum(1 for c in s if c in "WDL") for s in seqs)
    ctx.xg_samples = sum(1 for v in (ctx.home_xg_for, ctx.away_xg_for) if v is not None)
    return ctx


def _lineup_side(stats: dict[str, Any], side: str) -> list[str] | None:
    """Starter names for one side from the stats dict (or None)."""
    lu = stats.get("lineup") or {}
    players = lu.get(side)
    if isinstance(players, list) and players:
        return [str(p) for p in players if str(p).strip()]
    return None


def _str_or_none(v: Any) -> str | None:
    return str(v) if v else None


def _str_list(v: Any) -> list[str] | None:
    if not isinstance(v, (list, tuple)):
        return None
    out = [str(x) for x in v if str(x).strip()]
    return out or None


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _int(v: Any) -> int | None:
    """Int-parse a value; missing/garbage stays None (never fabricated)."""
    if v is None:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None
