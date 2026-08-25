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
    # Days since each team's previous match (pre-match only; None when unknown).
    home_rest_days: int | None = None
    away_rest_days: int | None = None
    sources: list[str] = field(default_factory=list)

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
        home_rest_days=_int(stats.get("home_rest_days")),
        away_rest_days=_int(stats.get("away_rest_days")),
        h2h=h2h_clean,
        consensus_odds=dict(odds.get("consensus") or {}) if odds.get("has_odds") else None,
        market_totals=dict(odds.get("totals") or {}),
        sources=sorted({s for s in (sources or []) if s}),
    )

    seqs = [s for s in (ctx.home_form, ctx.away_form) if s]
    if seqs:
        # Count actual result tokens (W/D/L), not the string length (dashes).
        ctx.form_samples = min(sum(1 for c in s if c in "WDL") for s in seqs)
    ctx.xg_samples = sum(1 for v in (ctx.home_xg_for, ctx.away_xg_for) if v is not None)
    return ctx


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
