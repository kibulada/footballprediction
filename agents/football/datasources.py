"""Multi-source football data aggregation core (pure, deterministic).

Sits BETWEEN the source adapters (flashscore, livescore, future sources) and
the existing prediction engine. It does NOT scrape anything itself and does
NOT replace the existing ``MultiSourceStatsFetcher`` provider chain (which
already serves flashscore as primary). It adds the missing layer:

  1. a generic ``FootballDataSource`` adapter interface (extensible)
  2. field-level fallback across sources (never whole-match fallback)
  3. cross-source validation: agreement vs discrepancy, primary-wins
  4. deterministic per-field confidence (HIGH/MEDIUM/LOW)
  5. team/competition name normalization + match identity + dedup
  6. missing-data honesty: known-empty vs unknown/unavailable
  7. data freshness tracking (stale -> confidence downgrade)
  8. source provenance + timestamp on every merged field

The output is ONE normalized match dataset with provenance and confidence,
ready to feed the existing engine. More sources => more coverage, never more
ambiguity: conflicting values are preserved as ``secondary`` and the winner is
chosen by configured priority, never silently overwritten.

The core imports nothing heavy at module load; name normalization reuses
``analyse._norm_team_name`` / ``analyse._teams_match`` lazily so this module
stays import-light and the existing code is not duplicated.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---- field status (section 10: never fabricate missing data) -------------

STATUS_AVAILABLE = "available"      # a source returned a value
STATUS_EMPTY = "empty"              # a source EXPLICITLY reports none (e.g. no injuries)
STATUS_UNAVAILABLE = "unavailable"  # no source provided this field at all

# ---- confidence levels (section 7) ---------------------------------------

CONF_HIGH = "HIGH"
CONF_MEDIUM = "MEDIUM"
CONF_LOW = "LOW"

# ---- canonical field names (section 4, adapted to the existing schema) ----

FIELD_MATCH = "match"
FIELD_FORM = "form"
FIELD_H2H = "h2h"
FIELD_LINEUP = "lineup"
FIELD_INJURIES = "injuries"
FIELD_STANDINGS = "standings"
FIELD_STATISTICS = "statistics"
FIELD_RECENT_MATCHES = "recent_matches"

ALL_FIELDS = (
    FIELD_MATCH, FIELD_FORM, FIELD_H2H, FIELD_LINEUP, FIELD_INJURIES,
    FIELD_STANDINGS, FIELD_STATISTICS, FIELD_RECENT_MATCHES,
)

# Default freshness (minutes) per field (section 15). Dynamic fields are
# short; static history is long.
DEFAULT_FRESHNESS = {
    FIELD_MATCH: 24 * 60,
    FIELD_FORM: 6 * 60,
    FIELD_H2H: 24 * 60,
    FIELD_LINEUP: 30,
    FIELD_INJURIES: 60,
    FIELD_STANDINGS: 6 * 60,
    FIELD_STATISTICS: 60,
    FIELD_RECENT_MATCHES: 6 * 60,
}

_DEFAULT_KICKOFF_TOLERANCE_MINUTES = 180.0


# --------------------------------------------------------------------------
# Name normalization + match identity + dedup (section 8/9)
# --------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    """Reuse the existing tolerant team-name normalization (analyse)."""
    from .analyse import _norm_team_name
    return _norm_team_name(name)


def teams_match(a: str, b: str) -> bool:
    """Tolerant team-name equality: strict token matcher + unambiguous aliases."""
    from .team_identity import names_match as _teams_match
    if _teams_match(a, b):
        return True
    from .team_alias import canonical_abbreviation
    ca, cb = canonical_abbreviation(a), canonical_abbreviation(b)
    if ca and cb:
        return _teams_match(ca, cb)
    if ca:
        return _teams_match(ca, b)
    if cb:
        return _teams_match(a, cb)
    return False


def normalize_competition(name: str) -> str:
    """Lowercase, strip accents/punctuation AND SPACES -> comparable token.

    Competition names are identifiers, not prose: "La Liga" and "laliga"
    (flashscore vs livescore spellings, verified live 2026-08-16) are the
    same competition and must compare equal, so internal spaces are removed
    too ("serie a" -> "seriea", still distinct from "serie b").
    """
    s = unicodedata.normalize("NFD", (name or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


def canonical_match_identity(
    *,
    home: str,
    away: str,
    kickoff: str | None = None,
    competition: str | None = None,
) -> dict[str, Any]:
    """Canonical, comparable fixture identity for cross-source validation.

    Normalizes team names (accents/punctuation + unambiguous aliases), kickoff
    and competition so the SAME fixture reported by two sources produces an
    identical value (agreement = true), while a different fixture does not.
    Sources must store this shape for the ``match`` field -- raw names and
    per-source IDs are intentionally excluded (they would always differ).

    Competition canonicalization (2026-08-22 fix): each source spells the
    same league differently ("epl" vs livescore "Premier League"); the raw
    squashed tokens would never compare equal. When the title resolves to a
    registered league key (leagues.json alias index), the KEY token is stored
    so both sources converge ("premierleague" -> "epl"). Unresolvable
    competitions (friendlies, minor cups) keep their own squashed token --
    distinct titles stay distinct.
    """
    from .team_alias import canonical_abbreviation

    def _canon(name: Any) -> str:
        raw = str(name or "")
        ab = canonical_abbreviation(raw)
        return normalize_team_name(ab or raw)

    return {
        "home": _canon(home),
        "away": _canon(away),
        "kickoff": kickoff,
        "competition": canonical_competition(competition),
    }


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _kickoff_close(a: str | None, b: str | None, tol_minutes: float) -> bool:
    if not a or not b:
        return True  # one side missing -> cannot reject on kickoff
    da, db = _parse_iso(a), _parse_iso(b)
    if da is None or db is None:
        return True
    return abs((da - db).total_seconds()) <= tol_minutes * 60.0


def same_match(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    kickoff_tolerance_minutes: float = _DEFAULT_KICKOFF_TOLERANCE_MINUTES,
) -> bool:
    """Whether two records are the same real-world fixture (section 8).

    Uses multiple signals: normalized home/away team names (tolerant), kickoff
    proximity, and competition. Ordered AND reversed team assignments are
    tried (sources may list the fixture in either home/away order), but if
    BOTH assignments match the records are considered ambiguous and NOT the
    same match (guards against "Inter" vs "Inter Turku" style false merges).
    """
    a_home, a_away = (a.get("home") or ""), (a.get("away") or "")
    b_home, b_away = (b.get("home") or ""), (b.get("away") or "")

    ordered = teams_match(a_home, b_home) and teams_match(a_away, b_away)
    reversed_ = teams_match(a_home, b_away) and teams_match(a_away, b_home)
    if ordered == reversed_:
        return False  # both (ambiguous) or neither (different)

    a_comp = normalize_competition(a.get("competition") or "")
    b_comp = normalize_competition(b.get("competition") or "")
    if a_comp and b_comp and a_comp != b_comp:
        return False

    return _kickoff_close(a.get("kickoff"), b.get("kickoff"), kickoff_tolerance_minutes)


def dedupe_matches(
    records: list[dict[str, Any]],
    *,
    kickoff_tolerance_minutes: float = _DEFAULT_KICKOFF_TOLERANCE_MINUTES,
) -> list[dict[str, Any]]:
    """Collapse duplicate records of the same fixture into one (section 9).

    The FIRST record (highest-priority source) is kept as the canonical shape;
    later matches of the same fixture contribute their ``source`` and fill any
    field the canonical record is missing (field-level fallback, primary kept).
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        merged = False
        for existing in out:
            if same_match(existing, rec, kickoff_tolerance_minutes=kickoff_tolerance_minutes):
                src = rec.get("source")
                sources = existing.setdefault("sources", [])
                if existing.get("source") and existing["source"] not in sources:
                    sources.append(existing["source"])
                if src and src not in sources:
                    sources.append(src)
                for key, val in rec.items():
                    if key in ("source", "sources"):
                        continue
                    if key not in existing or existing[key] in (None, "", [], {}):
                        existing[key] = val
                merged = True
                break
        if not merged:
            rec = dict(rec)
            src = rec.get("source")
            rec["sources"] = [src] if src else []
            out.append(rec)
    return out


# --------------------------------------------------------------------------
# Per-field samples and merged values (sections 3/4/5/6/10/15)
# --------------------------------------------------------------------------

@dataclass
class FieldSample:
    """One field as reported by ONE source (status distinguishes empty vs
    unknown; ``value`` is the already-normalized representation).

    ``entity`` (2026-09-02, wrong-team audit) names WHICH fixture the source
    fetched the value for -- ``{"home", "away"}`` names, optionally
    ``home_id`` / ``away_id`` / ``home_cid`` / ``away_cid`` / ``provider``.
    The merge verifies it against the analysed pair before the value can win
    or count as agreement; a value about another pair is rejected.
    """
    status: str = STATUS_UNAVAILABLE
    value: Any = None
    fetched_at: str | None = None
    entity: dict[str, Any] | None = None


def available(
    value: Any,
    fetched_at: str | None = None,
    entity: dict[str, Any] | None = None,
) -> FieldSample:
    return FieldSample(STATUS_AVAILABLE, value, fetched_at, entity)


def empty(fetched_at: str | None = None) -> FieldSample:
    return FieldSample(STATUS_EMPTY, None, fetched_at)


def missing() -> FieldSample:
    return FieldSample(STATUS_UNAVAILABLE, None, None)


@dataclass
class FieldValue:
    """The merged, provenance-carrying value for one field."""
    field: str
    status: str = STATUS_UNAVAILABLE
    value: Any = None
    source: str | None = None          # winning source
    sources: list[str] = field(default_factory=list)  # all sources that reported
    agreement: bool | None = None      # None when fewer than 2 sources
    discrepancy: bool = False
    confidence: str = CONF_LOW
    timestamp: str | None = None
    stale: bool = False
    secondary: list[dict[str, Any]] = field(default_factory=list)  # preserved conflicts
    # 2026-09-02: identity of the WINNING value ("verified" / "reversed" /
    # "unknown") and every sample the merge refused because it described a
    # different pair (source, reason). Provenance now says WHICH club, not
    # only which provider.
    identity: str | None = None
    identity_rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "source": self.source,
            "sources": self.sources,
            "agreement": self.agreement,
            "discrepancy": self.discrepancy,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "stale": self.stale,
            "secondary": self.secondary,
            "identity": self.identity,
            "identity_rejected": self.identity_rejected,
        }


def _canon(v: Any) -> str:
    if v is None:
        return "<none>"
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
    return str(v)


def values_agree(a: Any, b: Any) -> bool:
    return _canon(a) == _canon(b)


# ---- competition canonicalization (2026-08-22 fix) ------------------------
#
# Sources spell the same league differently ("epl", "Premier League",
# "England Premier League"). The alias index in leagues.json already knows
# these are one competition -- route resolvable titles to the registered
# league KEY so cross-source identities converge. Memoized: load_leagues
# re-reads JSON on every call.

_COMPETITION_KEY_CACHE: dict[str, str | None] = {}


def canonical_competition(name: str | None) -> str | None:
    """Squashed token, or the registered league-key token when resolvable."""
    raw = str(name or "").strip()
    if not raw:
        return None
    tok = normalize_competition(raw)
    if not tok:
        return None
    if tok not in _COMPETITION_KEY_CACHE:
        key: str | None = None
        try:
            from .league_resolver import competition_league_key

            key = competition_league_key(raw)
        except Exception:  # noqa: BLE001 -- identity must never raise
            key = None
        _COMPETITION_KEY_CACHE[tok] = normalize_competition(key) if key else None
    return _COMPETITION_KEY_CACHE[tok] or tok


# ---- per-field semantic comparators (2026-08-22 fix) ----------------------
#
# ``values_agree`` is exact canonical-JSON equality. Real sources describe the
# SAME fact with different representations (form sequences newest-first vs
# oldest-first, H2H over different meeting windows, standings as a full table
# vs a two-team snapshot), which produced constant agreement=false and a
# permanently-firing evidence gate. Each comparator below compares the
# SOURCE-INVARIANT facts for its field; uncomparable shapes fall back to the
# strict comparison so genuine conflicts still surface as discrepancies.


def _match_agree(a: Any, b: Any, ref: dict[str, Any] | None = None) -> bool:
    """Same fixture: ordered team names + compatible competition + close KO."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return values_agree(a, b)
    if not (
        teams_match(str(a.get("home") or ""), str(b.get("home") or ""))
        and teams_match(str(a.get("away") or ""), str(b.get("away") or ""))
    ):
        return False
    ca, cb = a.get("competition"), b.get("competition")
    # Both sides already store canonical tokens; a one-sided value carries no
    # comparable signal, so only two present-but-different tokens conflict.
    if ca and cb and ca != cb:
        return False
    return _kickoff_close(a.get("kickoff"), b.get("kickoff"), _DEFAULT_KICKOFF_TOLERANCE_MINUTES)


def _form_side_facts(d: Any) -> dict[str, Any] | None:
    """Source-invariant form facts for ONE side (dict shape only)."""
    if not isinstance(d, dict):
        return None
    facts: dict[str, Any] = {}
    for k in ("gf_avg", "ga_avg"):
        v = d.get(k)
        if isinstance(v, (int, float)):
            facts[k] = round(float(v), 2)
    ss = d.get("sample_size")
    if isinstance(ss, int):
        facts["sample_size"] = ss
    rg = d.get("recent_goals")
    if isinstance(rg, list):
        goals = [
            [int(g), int(ga)]
            for g, ga in rg
            if isinstance(g, (int, float)) and isinstance(ga, (int, float))
        ]
        if goals:
            facts["recent_goals"] = goals
    return facts or None


def _form_agree(a: Any, b: Any, ref: dict[str, Any] | None = None) -> bool:
    """Compare gf/ga averages, sample size and the oldest->newest scorelines.

    The ``sequence`` string is deliberately IGNORED: flashscore emits it
    newest-first while livescore emits oldest-first (same results either way,
    verified 2026-08-22 Everton-Palace), so string equality is meaningless.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return values_agree(a, b)
    compared_any = False
    for side in ("home", "away"):
        fa = _form_side_facts(a.get(side))
        fb = _form_side_facts(b.get(side))
        if fa is None or fb is None:
            continue
        for k in set(fa) & set(fb):
            compared_any = True
            if k == "recent_goals":
                if fa[k] != fb[k]:
                    return False
            elif k == "sample_size":
                if fa[k] != fb[k]:
                    return False
            elif abs(float(fa[k]) - float(fb[k])) > 0.05:
                return False
    if not compared_any:
        return values_agree(a, b)
    return True


def _h2h_counts_from_meetings(
    meetings: list[Any],
    home_name: str,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """W/D/L of the CURRENT match's home team from meeting rows.

    Finished meetings only (mirrors parse_h2h's tally integrity rule),
    newest first, optionally clamped to the other source's meeting count.
    """
    rows = [m for m in meetings if isinstance(m, dict) and m.get("status") == "finished"]
    rows.sort(
        key=lambda m: _parse_iso(m.get("kickoff")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    if limit is not None and limit > 0:
        rows = rows[:limit]
    w = d = l = 0
    for m in rows:
        hg, ag = m.get("home_score"), m.get("away_score")
        if not isinstance(hg, int) or not isinstance(ag, int):
            continue
        mh, ma = str(m.get("home") or ""), str(m.get("away") or "")
        from .team_identity import match_side as _match_side

        _side = _match_side(home_name, mh, ma)
        if _side == "home":
            cur, opp = hg, ag
        elif _side == "away":
            cur, opp = ag, hg
        else:
            continue
        if cur > opp:
            w += 1
        elif cur == opp:
            d += 1
        else:
            l += 1
    return w, d, l


def _h2h_agree(a: Any, b: Any, ref: dict[str, Any] | None = None) -> bool:
    """Agree when counts match directly OR within the other source's window.

    Sources scan different history depths (flashscore H2H tab = last N
    meetings, livescore = its own window): 3-2-0/5 vs 6-4-0/10 described the
    same unbeaten run. When either side ships its meeting list, the list is
    recounted twice: unclamped it must reproduce THAT side's own headline
    (a summary contradicting its own history cannot certify anything), and
    clamped to the other side's ``count`` it must reproduce the other
    headline. Otherwise a direct W/D/L comparison decides.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return values_agree(a, b)
    if (a.get("wins"), a.get("draws"), a.get("losses")) == (
        b.get("wins"), b.get("draws"), b.get("losses")
    ):
        return True
    home_name = str((ref or {}).get("home") or "")
    if not home_name:
        return False

    def _headline(v: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (v.get("wins"), v.get("draws"), v.get("losses"))

    for with_meetings, other in ((a, b), (b, a)):
        meetings = with_meetings.get("meetings")
        if not isinstance(meetings, list) or not meetings:
            continue
        if _h2h_counts_from_meetings(meetings, home_name, None) != _headline(with_meetings):
            continue
        limit = other.get("count") if isinstance(other.get("count"), int) else None
        counted = _h2h_counts_from_meetings(meetings, home_name, limit)
        if counted == _headline(other):
            return True
    return False


def _lineup_names(entry: Any) -> set[str] | None:
    """Starting-XI surnames from either lineup representation.

    Ignores shirt numbers/order/formation/coaches: predicted XIs differ in
    presentation between sources while naming the same eleven.
    """
    players: Any = entry
    if isinstance(entry, dict):
        players = entry.get("players")
    if not isinstance(players, list):
        return None
    names: set[str] = set()
    for p in players:
        if not isinstance(p, dict):
            continue
        if str(p.get("position") or "").strip().lower() in ("coach", "manager"):
            continue
        nm = normalize_team_name(str(p.get("name") or ""))
        if nm:
            names.add(nm.split()[-1])
    return names or None


def _lineup_sides(v: Any) -> dict[str, Any]:
    """Extract per-side player lists from either adapter's lineup shape."""
    out: dict[str, Any] = {}
    if not isinstance(v, dict):
        return out
    for side in ("home", "away"):
        entry = v.get(side)
        if isinstance(entry, list):
            out[side] = entry
        elif isinstance(entry, dict):
            out[side] = entry.get("players")
    return {k: e for k, e in out.items() if isinstance(e, list) and e}


def _lineup_agree(a: Any, b: Any, ref: dict[str, Any] | None = None) -> bool:
    """Agree iff every commonly-reported side names the SAME eleven."""
    sa, sb = _lineup_sides(a), _lineup_sides(b)
    common = [s for s in ("home", "away") if s in sa and s in sb]
    if not common:
        return values_agree(a, b)
    for side in common:
        na, nb = _lineup_names(sa[side]), _lineup_names(sb[side])
        if na is None or nb is None:
            continue
        if na != nb:
            return False
    return True


_STANDINGS_FACT_KEYS = (
    # (primary-shape key, secondary-shape key); row.get picks whichever exists
    ("played", "mp"), ("points", "pts"), ("wins", "w"),
    ("draws", "d"), ("losses", "l"), ("gf", "gf"), ("ga", "ga"), ("gd", "gd"),
)


def _standings_row_facts(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    facts: dict[str, Any] = {}
    for p, s in _STANDINGS_FACT_KEYS:
        v = row.get(p)
        if v is None:
            v = row.get(s)
        if v is not None:
            facts[p] = v
    return facts or None


def _standings_rows(v: Any, ref: dict[str, Any] | None) -> dict[str, Any]:
    """{side: row} from either a full-table or a two-team snapshot shape.

    Position (``pos``) is intentionally excluded from comparison: among
    level-points teams each source's tie-break orders differently, so pos is
    presentation noise while played/points/gf/ga/gd are the real state.
    """
    out: dict[str, Any] = {}
    if not isinstance(v, dict):
        return out
    tables = (v.get("tables") or {}).get("overall")
    if isinstance(tables, list):
        for side in ("home", "away"):
            nm = str((ref or {}).get(side) or "")
            if not nm:
                continue
            hits = [
                r for r in tables
                if isinstance(r, dict) and teams_match(str(r.get("team") or ""), nm)
            ]
            if len(hits) == 1:
                out[side] = hits[0]
        return out
    for side in ("home", "away"):
        r = v.get(side)
        if isinstance(r, dict):
            out[side] = r
    return out


def _standings_agree(a: Any, b: Any, ref: dict[str, Any] | None = None) -> bool:
    ra, rb = _standings_rows(a, ref), _standings_rows(b, ref)
    common = [
        s for s in ("home", "away")
        if ra.get(s) is not None and rb.get(s) is not None
    ]
    if not common:
        return values_agree(a, b)
    for side in common:
        if _standings_row_facts(ra[side]) != _standings_row_facts(rb[side]):
            return False
    return True


_FIELD_COMPARATORS = {
    FIELD_MATCH: _match_agree,
    FIELD_FORM: _form_agree,
    FIELD_H2H: _h2h_agree,
    FIELD_LINEUP: _lineup_agree,
    FIELD_STANDINGS: _standings_agree,
}


def field_values_agree(
    field_name: str,
    a: Any,
    b: Any,
    ref: dict[str, Any] | None = None,
) -> bool:
    """Field-aware agreement: semantic comparator when registered, else strict."""
    comparator = _FIELD_COMPARATORS.get(field_name)
    if comparator is None:
        return values_agree(a, b)
    try:
        return bool(comparator(a, b, ref))
    except Exception:  # noqa: BLE001 -- a broken comparator degrades to strict
        return values_agree(a, b)


def is_stale(
    fetched_at: str | None,
    *,
    field: str,
    freshness: dict[str, float] | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether ``fetched_at`` exceeds the freshness threshold for ``field``."""
    if not fetched_at:
        return False  # no timestamp -> cannot prove staleness (never over-penalize)
    threshold = (freshness or {}).get(field, DEFAULT_FRESHNESS.get(field))
    if threshold is None:
        return False
    dt = _parse_iso(fetched_at)
    if dt is None:
        return False
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return (ref - dt).total_seconds() > float(threshold) * 60.0


def _downgrade(conf: str) -> str:
    return {CONF_HIGH: CONF_MEDIUM, CONF_MEDIUM: CONF_LOW, CONF_LOW: CONF_LOW}[conf]


def field_confidence(
    *,
    status: str,
    agreement: bool | None,
    discrepancy: bool,
    stale: bool,
    n_sources: int,
) -> str:
    """Deterministic confidence (section 7). Pure function, no LLM."""
    if status in (STATUS_UNAVAILABLE,):
        return CONF_LOW
    if status == STATUS_EMPTY:
        # a source explicitly reports "none" -> complete info, just empty
        base = CONF_HIGH if n_sources >= 2 else CONF_MEDIUM
    elif discrepancy:
        base = CONF_LOW
    elif agreement is True and n_sources >= 2:
        base = CONF_HIGH
    else:  # single source, or partial agreement
        base = CONF_MEDIUM
    return _downgrade(base) if stale else base


IDENTITY_VERIFIED = "verified"
IDENTITY_REVERSED = "reversed"
IDENTITY_UNKNOWN = "unknown"
IDENTITY_REJECT = "reject"

_SIDE_KEYED_FIELDS = (FIELD_FORM, FIELD_LINEUP, FIELD_INJURIES, FIELD_STATISTICS)


def _side_name(entry: Any) -> str | None:
    """Team name carried inside a per-side value (form ``team_name``, lineup
    / context ``name``), or None."""
    if not isinstance(entry, dict):
        return None
    for k in ("team_name", "name", "team"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def sample_identity(
    field_name: str,
    sample: FieldSample,
    ref: dict[str, Any] | None,
) -> tuple[str, str | None]:
    """Does this sample describe the analysed pair? -> (verdict, reason).

    Verdicts: ``verified`` (same pair, same orientation), ``reversed`` (same
    pair, sides swapped -- the merge swaps side-keyed values), ``unknown``
    (nothing in the sample names a team; accepted as before) and ``reject``
    (names another pair / another club). Identity comes from
    ``sample.entity`` when the adapter set it; otherwise from names carried
    by the value itself (match home/away, H2H meetings, per-side
    ``team_name`` / ``name``). Canonical ids win over names when both sides
    carry them.
    """
    from .team_identity import names_match, same_fixture

    ref = ref or {}
    rh, ra = ref.get("home"), ref.get("away")
    if not (rh and ra):
        return IDENTITY_UNKNOWN, None
    ent = sample.entity if isinstance(sample.entity, dict) else None
    if ent and (ent.get("home") or ent.get("away")):
        eh, ea = ent.get("home"), ent.get("away")
        rcids = (ref.get("home_cid"), ref.get("away_cid"))
        ecids = (ent.get("home_cid"), ent.get("away_cid"))
        if all(rcids) and all(ecids):
            if (ecids[0], ecids[1]) == (rcids[0], rcids[1]):
                return IDENTITY_VERIFIED, None
            if (ecids[1], ecids[0]) == (rcids[0], rcids[1]):
                return IDENTITY_REVERSED, None
            return IDENTITY_REJECT, f"entity {eh} v {ea} is another pair (canonical ids differ)"
        o = same_fixture(rh, ra, eh, ea)
        if o == "ordered":
            return IDENTITY_VERIFIED, None
        if o == "reversed":
            return IDENTITY_REVERSED, None
        return IDENTITY_REJECT, f"entity {eh} v {ea} does not name {rh} v {ra}"
    v = sample.value
    if field_name == FIELD_MATCH and isinstance(v, dict) and v.get("home") and v.get("away"):
        o = same_fixture(rh, ra, str(v.get("home")), str(v.get("away")))
        if o == "ordered":
            return IDENTITY_VERIFIED, None
        # a match record is only the same fixture in the same orientation
        return IDENTITY_REJECT, f"match {v.get('home')} v {v.get('away')} is not {rh} v {ra}"
    if field_name == FIELD_H2H and isinstance(v, dict):
        rows = v.get("meetings") if isinstance(v.get("meetings"), list) else v.get("match_list")
        checked = 0
        for m in (rows or [])[:6]:
            if not isinstance(m, dict) or not (m.get("home") and m.get("away")):
                continue
            checked += 1
            if same_fixture(rh, ra, str(m.get("home")), str(m.get("away"))) is None:
                return IDENTITY_REJECT, f"h2h meeting {m.get('home')} v {m.get('away')} is not {rh} v {ra}"
        return (IDENTITY_VERIFIED if checked else IDENTITY_UNKNOWN), None
    if field_name in _SIDE_KEYED_FIELDS and isinstance(v, dict):
        hn, an = _side_name(v.get("home")), _side_name(v.get("away"))
        if hn or an:
            h_ok = names_match(rh, hn) if hn else None
            a_ok = names_match(ra, an) if an else None
            if h_ok is not False and a_ok is not False:
                return IDENTITY_VERIFIED, None
            h_x = names_match(ra, hn) if hn else None
            a_x = names_match(rh, an) if an else None
            if h_x is not False and a_x is not False and (h_x or a_x):
                return IDENTITY_REVERSED, None
            return IDENTITY_REJECT, f"{field_name} names {hn} / {an}, not {rh} / {ra}"
    return IDENTITY_UNKNOWN, None


def _swap_sides(field_name: str, value: Any) -> Any:
    if field_name in _SIDE_KEYED_FIELDS and isinstance(value, dict) and ("home" in value or "away" in value):
        out = dict(value)
        out["home"], out["away"] = value.get("away"), value.get("home")
        return out
    return value


def merge_field(
    field_name: str,
    samples: dict[str, FieldSample],
    *,
    priority: dict[str, int] | None = None,
    freshness: dict[str, float] | None = None,
    now: datetime | None = None,
    ref: dict[str, Any] | None = None,
) -> FieldValue:
    """Merge one field across sources (sections 3/5/6).

    Priority order decides the winning value; conflicting secondary values are
    preserved (never silently overwritten). Known-empty and unavailable are
    kept distinct. ``ref`` (home/away names + kickoff) gives the field-aware
    comparators their match context; without it they degrade gracefully.
    """
    priority = priority or {}
    now = now or datetime.now(timezone.utc)
    fv = FieldValue(field=field_name)

    # 2026-09-02 (wrong-team audit): identity BEFORE value. A sample that
    # describes another pair can neither win nor "agree"; a reversed one is
    # swapped onto the analysed orientation. Rejections are kept on the
    # merged value for the audit trail (source + reason).
    identities: dict[str, str] = {}
    samples = dict(samples)
    for s in list(samples.keys()):
        smp = samples[s]
        if smp.status != STATUS_AVAILABLE:
            continue
        verdict, reason = sample_identity(field_name, smp, ref)
        if verdict == IDENTITY_REJECT:
            fv.identity_rejected.append({"source": s, "reason": reason, "fetched_at": smp.fetched_at})
            samples.pop(s)
            continue
        if verdict == IDENTITY_REVERSED:
            samples[s] = FieldSample(smp.status, _swap_sides(field_name, smp.value), smp.fetched_at, smp.entity)
        identities[s] = verdict

    ordered = sorted(samples.keys(), key=lambda s: (-priority.get(s, 0), s))
    avail = [s for s in ordered if samples[s].status == STATUS_AVAILABLE]
    empties = [s for s in ordered if samples[s].status == STATUS_EMPTY]
    reported = [s for s in ordered if samples[s].status in (STATUS_AVAILABLE, STATUS_EMPTY)]
    fv.sources = list(reported)

    if not avail:
        if empties:
            # known-empty: at least one source explicitly reported none.
            fv.status = STATUS_EMPTY
            fv.source = empties[0]
            fv.timestamp = samples[empties[0]].fetched_at
            fv.agreement = True if len(empties) >= 2 else None
        else:
            fv.status = STATUS_UNAVAILABLE
        fv.confidence = field_confidence(
            status=fv.status, agreement=fv.agreement, discrepancy=False,
            stale=is_stale(fv.timestamp, field=field_name, freshness=freshness, now=now),
            n_sources=len(reported),
        )
        return fv

    primary = avail[0]
    fv.status = STATUS_AVAILABLE
    fv.source = primary
    fv.value = samples[primary].value
    fv.timestamp = samples[primary].fetched_at
    fv.identity = identities.get(primary, IDENTITY_UNKNOWN)

    if len(avail) == 1:
        fv.agreement = None
    else:
        agreeing = True
        for s in avail[1:]:
            if not field_values_agree(field_name, samples[primary].value, samples[s].value, ref):
                agreeing = False
                fv.secondary.append({
                    "source": s,
                    "value": samples[s].value,
                    "fetched_at": samples[s].fetched_at,
                })
        fv.agreement = agreeing
        fv.discrepancy = not agreeing

    fv.stale = is_stale(fv.timestamp, field=field_name, freshness=freshness, now=now)
    fv.confidence = field_confidence(
        status=fv.status, agreement=fv.agreement, discrepancy=fv.discrepancy,
        stale=fv.stale, n_sources=len(avail),
    )
    return fv


# --------------------------------------------------------------------------
# Adapter interface (section 21) + aggregator (sections 1/2/11/12/13/17)
# --------------------------------------------------------------------------

class FootballDataSource(ABC):
    """Normalized source interface. Each method returns a ``FieldSample``.

    Implementations normalize their raw output BEFORE returning (the
    aggregator compares values across sources, so representations must match).
    A method should never raise for a data miss -- return ``missing()``.
    """
    name: str = "base"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name

    async def get_match(self, ref: dict[str, Any]) -> FieldSample:
        return missing()

    async def get_form(self, ref: dict[str, Any]) -> FieldSample:
        return missing()

    async def get_h2h(self, ref: dict[str, Any]) -> FieldSample:
        return missing()

    async def get_lineup(self, ref: dict[str, Any]) -> FieldSample:
        return missing()

    async def get_injuries(self, ref: dict[str, Any]) -> FieldSample:
        return missing()

    async def get_standings(self, ref: dict[str, Any]) -> FieldSample:
        return missing()

    async def fetch_fields(self, ref: dict[str, Any]) -> dict[str, FieldSample]:
        return {
            FIELD_MATCH: await self.get_match(ref),
            FIELD_FORM: await self.get_form(ref),
            FIELD_H2H: await self.get_h2h(ref),
            FIELD_LINEUP: await self.get_lineup(ref),
            FIELD_INJURIES: await self.get_injuries(ref),
            FIELD_STANDINGS: await self.get_standings(ref),
        }


@dataclass
class UnifiedMatch:
    """The single normalized dataset handed to the prediction engine."""
    match: Any = None
    fields: dict[str, FieldValue] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"match": self.match, "sources": self.sources}
        for f, fv in self.fields.items():
            out[f] = fv.to_dict()
        out["source_metadata"] = {
            f: {
                "source": fv.source, "timestamp": fv.timestamp, "status": fv.status,
                # 2026-09-02: which CLUB pair the winning value was verified for.
                "identity": fv.identity,
                "identity_rejected": [r.get("source") for r in fv.identity_rejected],
            }
            for f, fv in self.fields.items()
        }
        out["confidence"] = {f: fv.confidence for f, fv in self.fields.items()}
        # P2-2: explicit form primary source -- livescore when the match is
        # in play (form sequence updates near-real-time), flashscore once it
        # is finished (more stable post-match). This is the winning source
        # after the status-aware ``_form_priority_for`` rule, surfaced here
        # so the card / audit can show WHICH source drove the form sequence.
        _form = self.fields.get(FIELD_FORM)
        out["form_primary_source"] = _form.source if _form else None
        return out


class MultiSourceAggregator:
    """Fetch + merge a match from every enabled source, field by field.

    Config shape (mirrors config/football.json -> ``data_sources``)::

        {
          "enabled": ["flashscore", "livescore"],
          "priority": {"flashscore": 100, "livescore": 80},
          "fallback": {"enabled": true},
          "validation": {"enabled": true},
          "merge": {"prefer_primary_on_conflict": true},
          "freshness": {"lineup": 30, "injuries": 60, ...},
        }
    """

    def __init__(
        self,
        sources: list[FootballDataSource],
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}
        self.sources = list(sources)
        self.priority = {s.name: int(v) for s, v in (
            (s, (cfg.get("priority") or {}).get(s.name, 0)) for s in self.sources
        )}
        self.fallback_enabled = bool((cfg.get("fallback") or {}).get("enabled", True))
        self.cross_validation_enabled = bool((cfg.get("validation") or {}).get("enabled", True))
        self.prefer_primary_on_conflict = bool((cfg.get("merge") or {}).get("prefer_primary_on_conflict", True))
        self.freshness = dict(DEFAULT_FRESHNESS)
        self.freshness.update({k: float(v) for k, v in (cfg.get("freshness") or {}).items()})

    # P2-2: form-source priority hint keyed off match status.
    _FORM_PRIORITY_LIVE: dict[str, int] = {"livescore": 100, "flashscore": 50}
    _FORM_PRIORITY_FINISHED: dict[str, int] = {"flashscore": 100, "livescore": 50}

    def _form_priority_for(self, match_status: str) -> dict[str, int]:
        """Per-field priority override for the ``form`` field.

        Livescore is more accurate when the match is in play (form
        sequence updates in near-real-time); flashscore is more stable
        once a match is finished. Unknown / absent status keeps the
        aggregator's global priority unchanged so we never silently
        invent a winner.
        """
        if match_status in ("live", "scheduled", "in_play"):
            return self._FORM_PRIORITY_LIVE
        if match_status == "finished":
            return self._FORM_PRIORITY_FINISHED
        return self.priority

    async def aggregate_match(
        self,
        ref: dict[str, Any],
        *,
        required_fields: list[str] | None = None,
        now: datetime | None = None,
    ) -> UnifiedMatch:
        """Collect, fallback and merge one match. A source failure never
        crashes the pipeline (section 12); partial results are returned."""
        samples: dict[str, dict[str, FieldSample]] = {f: {} for f in ALL_FIELDS}
        fetched: list[str] = []
        ordered = sorted(self.sources, key=lambda s: -self.priority.get(s.name, 0))

        for src in ordered:
            try:
                fields = await src.fetch_fields(ref)
            except Exception as exc:  # noqa: BLE001 -- a source must never crash the merge
                logger.warning("[DATA] match=%s source=%s status=failed reason=%s",
                               ref.get("id") or ref.get("home"), src.name, type(exc).__name__)
                fields = None
            if not fields:
                continue
            fetched.append(src.name)
            n = 0
            for f, sample in fields.items():
                if sample is None or sample.status == STATUS_UNAVAILABLE:
                    continue
                samples.setdefault(f, {})[src.name] = sample
                n += 1
            logger.info("[DATA] match=%s source=%s status=success fields=%d",
                        ref.get("id") or ref.get("home"), src.name, n)

            # Lazy fallback (section 11): stop once every required field is
            # covered, unless cross-validation of a second source is requested.
            if (
                not self.cross_validation_enabled
                and required_fields
                and all(samples.get(f) for f in required_fields)
            ):
                break

        fields: dict[str, FieldValue] = {}
        # P2-2: when the match is in play (``status in {scheduled, live}``)
        # trust ``livescore`` for the form sequence (most-recent scores are
        # fresher there); post-match (``finished``) prefer ``flashscore``
        # (more stable once all goals are recorded). The status hint comes
        # from the caller via ``ref["match_status"]`` when available.
        match_status = (ref.get("match_status") or "").lower().strip()
        form_priority = self._form_priority_for(match_status)
        for f in ALL_FIELDS:
            pri = form_priority if f == "form" else self.priority
            fv = merge_field(f, samples.get(f, {}), priority=pri,
                             freshness=self.freshness, now=now, ref=ref)
            fields[f] = fv
            if fv.discrepancy:
                logger.info("[VALIDATION] match=%s field=%s agreement=false",
                            ref.get("id") or ref.get("home"), f)
            elif fv.agreement is True:
                logger.info("[VALIDATION] match=%s field=%s agreement=true",
                            ref.get("id") or ref.get("home"), f)

        match_value = fields.get(FIELD_MATCH)
        unified = UnifiedMatch(
            match=match_value.value if match_value else None,
            fields=fields,
            sources=fetched,
        )
        return unified


class StaticSource(FootballDataSource):
    """In-memory adapter for already-collected data (e.g. flashscore fields).

    Lets the aggregator wrap pre-fetched values with provenance/confidence and
    merge them against live secondary sources WITHOUT re-fetching the primary.
    """

    def __init__(self, name: str, fields: dict[str, FieldSample] | None = None) -> None:
        super().__init__(name)
        self.fields = fields or {}

    async def fetch_fields(self, ref: dict[str, Any]) -> dict[str, FieldSample]:
        return dict(self.fields)


async def aggregate_collected(
    *,
    primary_name: str,
    primary_fields: dict[str, FieldSample],
    secondary: FootballDataSource | None = None,
    ref: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    required_fields: list[str] | None = None,
) -> UnifiedMatch:
    """Aggregate already-collected primary data with an optional secondary.

    The primary fields are wrapped in a ``StaticSource`` (no re-fetch); the
    secondary source is fetched lazily per the aggregator's fallback rules.
    """
    sources: list[FootballDataSource] = [StaticSource(primary_name, primary_fields)]
    if secondary is not None:
        sources.append(secondary)
    agg = MultiSourceAggregator(sources, config=config)
    return await agg.aggregate_match(ref or {}, required_fields=required_fields)


def coverage_report(unified: UnifiedMatch) -> dict[str, Any]:
    """Measured source-coverage summary for one match (section 8/20).

    Counts, per field, which source won the merge and how many fields are
    still unavailable. Real measured values, never hard-coded -- this is what
    tells us whether a secondary source actually improves the dataset.
    """
    by_source: dict[str, int] = {}
    missing = 0
    for fv in unified.fields.values():
        if fv.status == STATUS_UNAVAILABLE:
            missing += 1
        elif fv.source:
            by_source[fv.source] = by_source.get(fv.source, 0) + 1
    total = len(unified.fields)
    return {
        "fields_requested": total,
        "fields_available": total - missing,
        "fields_by_source": by_source,
        "fields_missing": missing,
    }


# --------------------------------------------------------------------------
# Flashscore adapter (delegates to the EXISTING MultiSourceStatsFetcher)
# --------------------------------------------------------------------------

class FlashscoreDataSource(FootballDataSource):
    """Adapter over the existing ``MultiSourceStatsFetcher`` provider chain.

    Flashscore remains the primary source; this adapter only exposes the
    already-normalized fields through the generic interface so the aggregator
    can merge/validate them against secondary sources. It does NOT re-fetch or
    re-normalize anything -- it converts the fetcher's dicts into FieldSample.
    """

    name = "flashscore"

    def __init__(self, fetcher: Any) -> None:
        super().__init__()
        self.fetcher = fetcher

    @staticmethod
    def _entity(ref: dict[str, Any]) -> dict[str, Any]:
        """The pair this adapter fetched for (the resolved ids + names)."""
        return {
            "provider": "flashscore",
            "home": ref.get("home"), "away": ref.get("away"),
            "home_id": ref.get("home_id"), "away_id": ref.get("away_id"),
            "home_cid": ref.get("home_cid"), "away_cid": ref.get("away_cid"),
        }

    async def get_match(self, ref: dict[str, Any]) -> FieldSample:
        try:
            fixture = await self.fetcher.fetch_upcoming_fixture(
                ref.get("home_id"), ref.get("away_id"), ref.get("league_meta") or {}
            )
        except Exception:  # noqa: BLE001
            return missing()
        if not fixture:
            return missing()
        ent = None
        if isinstance(fixture, dict) and fixture.get("home") and fixture.get("away"):
            ent = {"provider": "flashscore", "home": fixture.get("home"), "away": fixture.get("away")}
        return available(fixture, entity=ent)

    async def get_form(self, ref: dict[str, Any]) -> FieldSample:
        try:
            home = await self.fetcher.fetch_team_form(ref.get("home_id"), ref.get("league_meta") or {})
            away = await self.fetcher.fetch_team_form(ref.get("away_id"), ref.get("league_meta") or {})
        except Exception:  # noqa: BLE001
            return missing()
        if home is None and away is None:
            return missing()
        return available({"home": home, "away": away}, entity=self._entity(ref))

    async def get_h2h(self, ref: dict[str, Any]) -> FieldSample:
        try:
            h2h = await self.fetcher.fetch_h2h(ref.get("home_id"), ref.get("away_id"), ref.get("league_meta") or {})
        except Exception:  # noqa: BLE001
            return missing()
        return available(h2h, entity=self._entity(ref)) if h2h else missing()

    async def get_lineup(self, ref: dict[str, Any]) -> FieldSample:
        url = ref.get("match_url")
        if not url:
            return missing()
        try:
            lu = await self.fetcher.fetch_flashscore_lineups_for_match(url)
        except Exception:  # noqa: BLE001
            return missing()
        return available(lu) if lu else missing()

    async def get_injuries(self, ref: dict[str, Any]) -> FieldSample:
        url = ref.get("match_url")
        if not url:
            return missing()
        try:
            ctx = await self.fetcher.fetch_flashscore_event_context(
                url, ref.get("home"), ref.get("away")
            )
        except Exception:  # noqa: BLE001
            return missing()
        if not ctx:
            return missing()
        injuries = {
            side: (ctx.get(side) or {}).get("missing") or []
            for side in ("home", "away")
        }
        return available(injuries)

    async def get_standings(self, ref: dict[str, Any]) -> FieldSample:
        league_key = (ref.get("league_meta") or {}).get("_league_key")
        if not league_key:
            return missing()
        try:
            standings = await self.fetcher.fetch_league_standings(league_key)
        except Exception:  # noqa: BLE001
            return missing()
        return available(standings) if standings else missing()
