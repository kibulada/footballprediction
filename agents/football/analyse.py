"""Analyse a specific match using multi-source stats fetcher.

Provider chain resolves per league: flashscore (primary) -> football-data.org
-> thesportsdb (fallback). Each field (form, H2H) is fetched independently so
a failure in one provider doesn't break the whole analysis.
"""
from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from .cache import Cache
from .clv_gate import gate_segment, load_segment_stats
from .edge_benchmark import edge_benchmark
from .entity_registry import canonical_team_id
from .identity_gate import firewall_cfg as _identity_firewall_cfg
from .league_resolver import competition_league_key, load_leagues, resolve_league_scored
from .model_gates import build_confidence_block
from .multi_source import MultiSourceStatsFetcher
from .odds_fetcher import OddsFetcher
from .predictor import derive_picks, market_movement, value_edges
from .scorer import best_odds, consensus_odds, find_outlier, score_signal
from .timeutil import utc_now_iso

ROOT = Path(__file__).resolve().parent.parent.parent

# The optional flashscore browser renders (pre-match stats / lineups) must not
# blow the runner's deadline on a slow network: once the shared analysis
# budget is nearly spent they are skipped (see _budget_short). Stats are a
# model feature but pre-match xG is rarely present on flashscore, and the
# lineups render is context-only, so skipping both late in a run costs little.


def _budget_short(remaining: float | None, margin: float = 10.0) -> bool:
    """True when the shared analysis budget is nearly spent.

    ``remaining`` comes from ``multi_source.analysis_remaining()`` (None when
    no clock is armed -> never skip). The margin reserves safety headroom
    before the runner's hard deadline so the reply always lands instead of
    dying with "runner deadline terlampaui".
    """
    if remaining is None:
        return False
    return remaining <= margin


def _season_now() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _parse_kickoff_utc(kickoff: str | None) -> datetime | None:
    """Parse an ISO kickoff string to an aware UTC datetime, or None."""
    if not kickoff:
        return None
    try:
        s = str(kickoff).strip()
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _kickoff_hours_ahead(kickoff: str | None, now: datetime | None = None) -> float | None:
    """Hours from now until kickoff (None when unparseable).

    Used to skip the flashscore lineups render for fixtures far away: predicted
    lineups are only published for near matches, so a far fixture would pay a
    full browser render for an empty tab.
    """
    dt = _parse_kickoff_utc(kickoff)
    if dt is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (dt - now).total_seconds() / 3600.0


# P1.1: kickoff cross-check tolerance. Real broadcast/schedule shifts are
# rare and small (minutes, occasionally an hour); timezone/date bugs (the
# Genclerbirligi/Cadiz-B class) produce multi-hour deltas. Anything beyond
# this tolerance marks the kickoff UNCERTAIN instead of trusting the
# first-wins value.
_KICKOFF_TOLERANCE_HOURS = 2.0


def _kickoff_cross_check(
    primary: str | None,
    candidates: dict[str, str],
    tolerance_hours: float = _KICKOFF_TOLERANCE_HOURS,
) -> tuple[bool, dict[str, float]]:
    """Cross-check the primary kickoff against independent sources.

    Returns ``(uncertain, deltas_hours)``. ``uncertain`` is True when any
    independent candidate disagrees with ``primary`` by more than
    ``tolerance_hours``. ``match_finished`` must NEVER be derived from an
    uncertain kickoff -- the match status is "cannot determine", not
    "finished".
    """
    if not primary or len(candidates) < 2:
        return False, {}
    p = _parse_kickoff_utc(primary)
    if p is None:
        return False, {}
    deltas: dict[str, float] = {}
    for src, val in candidates.items():
        if not val or val == primary:
            continue
        v = _parse_kickoff_utc(val)
        if v is None:
            continue
        deltas[src] = round(abs((v - p).total_seconds()) / 3600.0, 2)
    return any(d > tolerance_hours for d in deltas.values()), deltas


_TEAM_NAME_PREFIXES = {
    "fk", "fc", "nk", "cd", "sc", "pfc", "ifk", "ss", "rc", "ca",
    "ec", "cr", "se", "ac", "cf", "us", "sd", "de", "sv", "sk",
}


# Letters whose Unicode NFD does not decompose to an ASCII base (strokes and
# dots are not combining marks, so they would be dropped entirely): map them
# explicitly so cross-provider names like "Bodø/Glimt" vs "Bodo/Glimt"
# normalize to the same token.
_STROKE_LETTERS = str.maketrans(
    {
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ħ": "h",
        "ı": "i",
        "ŋ": "n",
        "ß": "ss",
    }
)


def _norm_team_name(name: str) -> str:
    """Lowercase, strip accents and punctuation -> comparable token string."""
    import re
    import unicodedata

    # Parenthetical country suffixes ("Tobol (Kaz)", "Partizan (Srb)") come
    # from the flashscore homepage and are not part of any provider's team
    # name; dropping them lets "Tobol (Kaz)" match "Tobol Kostanay".
    s = re.sub(r"\([^)]*\)", " ", name or "")
    s = s.lower().translate(_STROKE_LETTERS)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _teams_match(a: str, b: str) -> bool:
    """Tolerant team-name equality across providers.

    Providers disagree on prefixes ("FK Bodø/Glimt" vs "Bodø/Glimt") and
    honorifics ("Royale Union Saint-Gilloise" vs "Union Saint-Gilloise"),
    so we normalize, drop common prefixes, and fall back to token
    containment: every token of the shorter name must appear in the longer
    one (as an exact token or a substring of a longer token, min 3 chars).
    """
    na, nb = _norm_team_name(a), _norm_team_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    for pref in _TEAM_NAME_PREFIXES:
        if na.startswith(pref + " ") and na[len(pref) + 1:] == nb:
            return True
        if nb.startswith(pref + " ") and nb[len(pref) + 1:] == na:
            return True
    ta, tb = na.split(), nb.split()
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not shorter:
        return False
    # Symmetric containment: "hearts" must match "heart of midlothian fc"
    # ("heart" in "hearts"). The both-sides ambiguity guard (_side_role)
    # rejects a name that matches BOTH home and away, so symmetric substrings
    # cannot silently misassign sides. Containment is length-guarded on BOTH
    # sides (2026-08-17, B1 fix): a 1-char token in the longer name ("a" in
    # "Dep. A Coruna") must not match every token containing that letter
    # ("las", "palmas") -- that made "Dep. A Coruna" match "Las Palmas" /
    # "Albacete" and corrupted standings/form/side matching. Token EQUALITY
    # stays unguarded ("sg" == "sg" is fine); only containment needs >= 3.
    return all(
        any(
            t == w
            or (len(t) >= 3 and t in w)
            or (len(w) >= 3 and w in t)
            for w in longer
        )
        for t in shorter
    )


def _form_depth(form: dict[str, Any] | None) -> int:
    """Finished matches in a form dict (any provider shape)."""
    seq = (form or {}).get("sequence")
    if isinstance(seq, str):
        return len([p for p in seq.split("-") if p])
    if isinstance(seq, (list, tuple)):
        return len(seq)
    rg = (form or {}).get("recent_goals")
    return len(rg) if isinstance(rg, (list, tuple)) else 0


def _form_depth_thin(home_form: dict[str, Any] | None, away_form: dict[str, Any] | None) -> bool:
    """True when either team's form window is < 3 matches (noise, not signal).

    Reuses the same floor as ``model_gates.form_depth_shallow`` (MIN_FORM_DEPTH
    = 3): a None window counts as 0 (shallow), so this also covers the legacy
    "form is None" case. Named ``_form_depth_thin`` (NOT ``_form_depth_shallow``)
    because the analyse flow already binds that exact name to a local import
    inside the prediction block.
    """
    from .model_gates import form_depth_shallow

    return form_depth_shallow(home_form, away_form)


def _merge_team_fields(
    base: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fill only MISSING fields of ``base`` from ``extra`` (P1.3).

    The oddspapi fallback must not discard a partially-resolved identity
    (e.g. flashscore resolved the id but the provider chain failed before the
    name): each field of ``base`` is kept, and only fields that are absent /
    empty in ``base`` are taken from ``extra``. ``provider``/``_role`` follow
    the same rule, so a resolved identity keeps its original provider label
    unless that field was never set.
    """
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if out.get(k) in (None, ""):
            out[k] = v
    return out


def _side_role(
    name: str,
    home_candidates: list[str],
    away_candidates: list[str],
) -> str | None:
    """Which side of the fixture ``name`` unambiguously matches.

    Returns "home" / "away" when the name matches candidates of exactly ONE
    side, None when it matches neither or BOTH. The BOTH case is the
    wrong-match guard: with containment matching ("Inter" matches both
    "Inter" and "Inter Turku"), assigning a name that fits both sides would
    silently misprice the match (e.g. an odds row for Inter Turku priced as
    Inter Milan). An ambiguous name is never assigned to either side.
    """
    mh = any(_teams_match(name, h) for h in home_candidates)
    ma = any(_teams_match(name, a) for a in away_candidates)
    if mh == ma:
        return None  # neither (no match) or both (ambiguous) -> no side
    return "home" if mh else "away"


def _data_sources_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The data_sources config block (feature-gated multi-source layer)."""
    return (cfg or {}).get("data_sources") or {}


def _safe_entity_canonical(
    provider: str | None,
    provider_id: Any,
    league_key: str | None,
    name: str,
) -> str | None:
    """Deterministic canonical_id for an entity, never raising (G2)."""
    try:
        return canonical_team_id(league_key, name)
    except Exception:  # noqa: BLE001 -- identity must never break analysis
        return None


def _build_secondary_source(cfg: dict[str, Any], cache: Any | None = None):
    """LiveScore secondary source when enabled, else None (flashscore only).

    Uses the verified ``lsmedia1.com`` LiveScore public API (no key). The
    existing TTL cache is passed through for date-feed reuse.
    """
    ds_cfg = _data_sources_config(cfg)
    ls_cfg = ds_cfg.get("livescore") or {}
    if not ls_cfg.get("enabled", False):
        return None
    from .livescore import LiveScoreClient, LiveScoreDataSource

    return LiveScoreDataSource(
        LiveScoreClient(base_url=ls_cfg.get("base_url") or None),
        cache=cache,
        max_pages=int(ls_cfg.get("max_pages", 3)),
    )


def _strip_team_club_token(name: str) -> str:
    """Drop a leading OR trailing club-type token (FK/FC/...): both
    "FK Bodø/Glimt" and "Paris Saint-Germain FC" normalize to the bare name
    so exact comparison after stripping is club-prefix/suffix agnostic. The
    prefix set is lowercase, so comparison is case-insensitive -- the RAW
    name ("Paris Saint-Germain FC") must strip as well as the normalized
    one (2026-08-17, B2 fix)."""
    tokens = name.split()
    if len(tokens) > 1:
        if tokens[0].lower() in _TEAM_NAME_PREFIXES:
            tokens = tokens[1:]
        elif tokens[-1].lower() in _TEAM_NAME_PREFIXES:
            tokens = tokens[:-1]
    return " ".join(tokens)


def _match_standings_team(
    tbl: list[dict[str, Any]],
    target: str,
) -> dict[str, Any] | None:
    """Match one standings row to a team, in strictness tiers.

    Tier order (2026-08-17, fix for wrong-club capture):
      1. exact normalized team name;
      2. unambiguous alias / abbreviation (``resolve_team_alias`` /
         ``canonical_abbreviation``), matched exactly and after stripping a
         leading/trailing club token (alias "Paris Saint-Germain FC" lands on
         standings "Paris Saint-Germain");
      3. club-token-strip equality ("FK Bodø/Glimt" vs "Bodø/Glimt");
      4. tolerant containment -- LAST, and ONLY when exactly ONE row matches.
    The containment tier is ambiguity-guarded: when several rows contain the
    target token ("Paris FC" vs "Paris Saint-Germain" for a "Paris" target)
    the function returns None instead of picking the wrong club. Returns None
    when nothing matches or the match is ambiguous.
    """
    from .team_alias import canonical_abbreviation, resolve_team_alias, standings_abbreviations

    nt = _norm_team_name(target)
    if not nt:
        return None

    def _rows_by_norm(norm: str) -> list[dict[str, Any]]:
        return [r for r in tbl if _norm_team_name((r or {}).get("team") or "") == norm]

    # Tier 1: exact normalized.
    exact = _rows_by_norm(nt)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None  # duplicate rows -> ambiguous

    # Tier 2: unambiguous alias / abbreviation ("psg" -> "Paris
    # Saint-Germain FC"), matched exactly AND after stripping a club token
    # so an alias carrying a suffix/prefix club token still lands on the
    # standings spelling without it.
    aliased = resolve_team_alias(target, None) or canonical_abbreviation(target)
    if aliased and aliased != target:
        na = _norm_team_name(aliased)
        if len(_rows_by_norm(na)) == 1:
            return _rows_by_norm(na)[0]
        stripped = _norm_team_name(_strip_team_club_token(na))
        if stripped and stripped != na:
            hit = _rows_by_norm(stripped)
            if len(hit) == 1:
                return hit[0]

    # Tier 2b: REVERSE abbreviation lookup (full name -> standings spelling).
    # Flashscore tables abbreviate several clubs ("Paris SG", "Atl. Madrid")
    # in ways tier 2 cannot reverse: when the target IS the canonical name,
    # ``resolve_team_alias`` returns it unchanged (guarding ``aliased !=
    # target`` blocks the path) and the plain names never equal the table
    # row. Known reverse spellings are tried for both the raw name and its
    # club-token-stripped form ("Paris Saint-Germain FC" -> "Paris SG"),
    # BEFORE the containment tier can latch onto a wrong club (B2/B5,
    # verified 2026-08-17).
    for base in (target, _strip_team_club_token(target)):
        for cand in standings_abbreviations(base):
            hit = _rows_by_norm(_norm_team_name(cand))
            if len(hit) == 1:
                return hit[0]

    # Tier 3: club-token-strip equality ("FK Bodø/Glimt" vs "Bodø/Glimt").
    # Only FULL multi-token names qualify: a one-token short target ("Paris",
    # "Lens") must NOT strip-match "Paris FC" at this tier -- that is exactly
    # the ambiguous case the containment guard (tier 4) must decide, so a
    # short target always falls through to the guarded tier.
    if len(nt.split()) > 1:
        target_stripped = _norm_team_name(_strip_team_club_token(nt))
        for r in tbl:
            nr = _norm_team_name((r or {}).get("team") or "")
            if not nr:
                continue
            if target_stripped and nr == target_stripped:
                return r
            nr_stripped = _norm_team_name(_strip_team_club_token(nr))
            if nr_stripped and nr_stripped == nt:
                return r
            if (
                target_stripped
                and nr_stripped
                and nr_stripped == target_stripped
                and nr_stripped != nr
            ):
                return r

    # Tier 4: containment, ambiguity-guarded (never guess the wrong club).
    hits = [r for r in tbl if _teams_match((r or {}).get("team") or "", target)]
    if len(hits) == 1:
        return hits[0]
    return None


def _apply_source_confidence_gate(
    prediction: dict[str, Any] | None,
    *,
    passed: bool,
    reason: str | None,
    cap: float = 0.5,
) -> None:
    """P1-2 (2026-08-22): propagate an evidence-gate veto into the quality
    numbers every downstream consumer displays.

    Previously ``data_completeness`` stayed at its engine value (often 1.0)
    while the card showed a source-confidence veto -- contradictory quality
    signals in one payload. Capping completeness ONCE here makes the grade
    gates, the decision layer's ``data_quality`` component + pick-confidence
    ``completeness_factor`` and the signal engine's ``data_quality`` block all
    reflect it, and records the audit trail on the prediction.
    """
    if passed or not isinstance(prediction, dict):
        return
    before = float(prediction.get("data_completeness") or 0.0)
    prediction["data_completeness"] = round(min(before, cap), 3)
    prediction["source_confidence_gate"] = {
        "passed": False,
        "reason": reason,
        "completeness_before": round(before, 3),
        "completeness_capped": prediction["data_completeness"],
    }


def _primary_fields(
    *,
    home: str,
    away: str,
    kickoff: str | None,
    competition: str | None,
    home_form: dict[str, Any] | None,
    away_form: dict[str, Any] | None,
    h2h: dict[str, Any] | None,
    lineups: dict[str, Any] | None,
    missing_players: dict[str, Any] | None,
    standings: dict[str, Any] | None,
    match_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Wrap already-collected flashscore data as FieldSamples.

    Preserves the known-empty vs unavailable distinction (section 10): a
    provider that explicitly reported "no injuries" becomes STATUS_EMPTY,
    while a field never fetched becomes STATUS_UNAVAILABLE. The ``match``
    field uses the shared canonical identity shape so LiveScore validation
    can compare it (same fixture -> agreement, different -> discrepancy).
    """
    from .datasources import (
        FIELD_FORM, FIELD_H2H, FIELD_INJURIES, FIELD_LINEUP, FIELD_MATCH,
        FIELD_STANDINGS, FIELD_STATISTICS,
        available, canonical_match_identity, empty, missing,
    )
    from .timeutil import utc_now_iso

    now = utc_now_iso()
    fields: dict[str, Any] = {}

    if home and away:
        fields[FIELD_MATCH] = available(
            canonical_match_identity(home=home, away=away, kickoff=kickoff, competition=competition),
            now,
        )
    else:
        fields[FIELD_MATCH] = missing()

    fields[FIELD_FORM] = (
        available({"home": home_form, "away": away_form}, now)
        if (home_form or away_form) else missing()
    )

    if h2h is None:
        fields[FIELD_H2H] = missing()
    elif any(h2h.get(k) for k in ("wins", "draws", "losses")):
        fields[FIELD_H2H] = available(h2h, now)
    else:
        fields[FIELD_H2H] = empty(now)

    fields[FIELD_LINEUP] = (
        available(lineups, now) if (lineups and lineups.get("home_count")) else missing()
    )

    if missing_players is None:
        fields[FIELD_INJURIES] = missing()
    elif any((v or {}).get("missing") for v in missing_players.values()):
        fields[FIELD_INJURIES] = available(missing_players, now)
    else:
        fields[FIELD_INJURIES] = empty(now)

    fields[FIELD_STANDINGS] = (
        available(standings, now)
        if (standings and (standings.get("tables") or {}).get("overall")) else missing()
    )

    fields[FIELD_STATISTICS] = available(match_stats, now) if match_stats else missing()

    return fields


def extract_h2h_entries(
    payload: dict[str, Any],
    home_name: str,
    away_name: str,
    home_query: str | None = None,
    away_query: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-bookmaker 1X2 entries from an odds payload.

    Outcome names come from the odds provider and can differ from the
    resolved team names ("Bodø/Glimt" vs "FK Bodø/Glimt", "Union
    Saint-Gilloise" vs "Royale Union Saint-Gilloise"), so the home/away
    sides are matched tolerantly via _teams_match instead of exact equality.
    The raw user query is used as a secondary fallback (e.g. the odds
    provider lists "Sabah FK" while our resolution returns "Sabah Baku").

    Anti wrong-match guard: an outcome name that matches BOTH sides
    (e.g. provider "Inter" vs candidates home "Inter" / away "Inter
    Turku") is ambiguous and is NOT assigned to either side, so a price can
    never be silently attached to the wrong team.
    """
    entries: list[dict[str, Any]] = []
    home_candidates = [n for n in (home_name, home_query) if n]
    away_candidates = [n for n in (away_name, away_query) if n]
    for bm in payload.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            entry: dict[str, Any] = {"bookmaker": bm.get("title", "?")}
            # Opening price (when the provider exposes it, e.g. nowgoal ``f``
            # leg) rides along so market movement can be detected per side.
            # The opening side is resolved with the SAME role matching as the
            # current price, so provider-vs-resolved name differences cannot
            # mislabel a side.
            opening: dict[str, float] = {}
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                op = outcome.get("opening_price")
                if (name or "").lower() == "draw":
                    entry["draw"] = price
                    if op and op > 1.0:
                        opening["draw"] = op
                    continue
                role = _side_role(name, home_candidates, away_candidates)
                if role == "home":
                    entry["home"] = price
                    if op and op > 1.0:
                        opening["home"] = op
                elif role == "away":
                    entry["away"] = price
                    if op and op > 1.0:
                        opening["away"] = op
                # role None: no match or ambiguous (matches both sides) -> skip
            if "home" in entry and "away" in entry:
                if opening.get("home") and opening.get("away"):
                    entry["opening"] = {
                        "home": opening["home"],
                        "draw": opening.get("draw"),
                        "away": opening["away"],
                    }
                entries.append(entry)
    return entries


async def find_match_odds_payload(
    odds_keys: list[str],
    home_name: str,
    away_name: str,
    odds: "OddsFetcher",
    cache: "Cache",
    cache_ttl_seconds: dict[str, int],
    home_query: str | None = None,
    away_query: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Find odds for a match across candidate sport keys.

    The primary league key (e.g. soccer_uefa_champs_league) often has no
    fixtures while the qualification variant does
    (soccer_uefa_champs_league_qualification); each candidate key is tried
    in order and cached separately. Returns (match_payload, key_that_hit)
    or (None, None). The raw user queries are accepted as fallback names
    (resolved names may differ from the odds provider's spelling).

    Anti wrong-match guard: a fixture row is accepted only when BOTH its
    home and away names match exactly one side (see ``_side_role``) -- a row
    whose name also matches the OTHER side's candidates is ambiguous and
    rejected. Among valid rows, the most specific (exact-name) fixture wins
    so a fuzzy containment hit ("Inter" matching "Inter Turku") can never
    outrank the true exact identity.
    """
    ttl = cache_ttl_seconds.get("odds", 900)
    home_candidates = [n for n in (home_name, home_query) if n]
    away_candidates = [n for n in (away_name, away_query) if n]
    home_norms = {_norm_team_name(c) for c in home_candidates}
    away_norms = {_norm_team_name(c) for c in away_candidates}
    for key in odds_keys:
        if not key:
            continue
        odds_cache_key = f"odds_{key}_recent"
        payload = cache.get(odds_cache_key, ttl) or []
        if not payload:
            payload = await odds.fetch_odds(key) or []
            cache.set(odds_cache_key, payload)
        best_match: dict[str, Any] | None = None
        best_score = -1
        for match in payload:
            h_name = match.get("home_team") or ""
            a_name = match.get("away_team") or ""
            if _side_role(h_name, home_candidates, away_candidates) != "home":
                continue
            if _side_role(a_name, home_candidates, away_candidates) != "away":
                continue
            # Specificity: exact normalized equality beats fuzzy containment.
            score = (
                (1 if _norm_team_name(h_name) in home_norms else 0)
                + (1 if _norm_team_name(a_name) in away_norms else 0)
            )
            if score > best_score:
                best_score = score
                best_match = match
        if best_match is not None:
            return best_match, key
    return None, None


def _margin_free(odds: dict[str, float]) -> dict[str, float] | None:
    """Remove bookmaker margin from decimal 1X2 odds -> implied probs."""
    inv = {k: (1.0 / v) if v and v > 1.0 else 0.0 for k, v in odds.items()}
    s = sum(inv.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in inv.items()}


def cross_source_odds_check(
    payloads: dict[str, dict[str, Any]],
    *,
    home_name: str,
    away_name: str,
    home_query: str | None = None,
    away_query: str | None = None,
    tolerance_pp: float = 8.0,
) -> dict[str, Any]:
    """P3: compare key lines across independent odds sources.

    ``payloads`` maps a source name to its normalized payload (the same
    The-Odds-API shape every tier emits). For each market that BOTH sources
    expose we compare the margin-free implied probabilities (1X2 per side,
    Over 2.5) and the Asian-Handicap consensus line; the largest
    implied-probability gap (in percentage points) across any compared line
    is the disagreement metric.

    Returns ``{status, n_sources, max_pp_diff, compared}`` where ``status``
    is ``"ok"`` or ``"cross_source_disagreement"``. First-wins resolution is
    UNCHANGED -- this is a visibility/confidence input, never a merge.
    """
    from .signal_engine import ah_consensus, extract_asian_handicap
    from .scorer import consensus_odds

    def _lines(src: str, payload: dict[str, Any]) -> dict[str, Any]:
        entries = extract_h2h_entries(
            payload, home_name, away_name,
            home_query=home_query, away_query=away_query,
        )
        cons = consensus_odds(entries) if entries else None
        probs = _margin_free(cons) if cons else None
        totals = extract_market_totals(payload)
        over = totals.get("Over 2.5") or {}
        under = totals.get("Under 2.5") or {}
        over_prob = None
        if (over.get("odds") or 0) > 1.0 and (under.get("odds") or 0) > 1.0:
            inv = 1.0 / over["odds"] + 1.0 / under["odds"]
            over_prob = (1.0 / over["odds"]) / inv if inv > 0 else None
        ah_rows = extract_asian_handicap(payload)
        ah = ah_consensus(ah_rows) if ah_rows else None
        return {"probs": probs, "over_prob": over_prob, "ah": ah}

    lines = {src: _lines(src, p) for src, p in payloads.items() if p}
    names = [s for s, l in lines.items() if l]
    if len(names) < 2:
        return {"status": "ok", "n_sources": len(names), "max_pp_diff": None, "compared": {}}

    compared: dict[str, Any] = {}
    max_diff = 0.0
    srcs = list(names)
    a, b = srcs[0], srcs[1]
    la, lb = lines[a], lines[b]

    # 1X2: margin-free implied per side.
    if la["probs"] and lb["probs"]:
        side_diffs = {
            k: abs(la["probs"].get(k, 0.0) - lb["probs"].get(k, 0.0)) * 100.0
            for k in ("home", "draw", "away")
        }
        compared["1x2"] = {k: round(v, 2) for k, v in side_diffs.items()}
        max_diff = max(max_diff, max(side_diffs.values()))
    # Over 2.5: implied over probability.
    if la["over_prob"] is not None and lb["over_prob"] is not None:
        d = abs(la["over_prob"] - lb["over_prob"]) * 100.0
        compared["over_2.5"] = round(d, 2)
        max_diff = max(max_diff, d)
    # Asian handicap: line + implied home side.
    if la["ah"] and lb["ah"]:
        line_diff = abs(float(la["ah"].get("line") or 0.0) - float(lb["ah"].get("line") or 0.0))
        compared["ah_line"] = round(line_diff, 3)
        ha, hb = la["ah"].get("home"), lb["ah"].get("home")
        if ha and hb and ha > 1.0 and hb > 1.0:
            pa = 1.0 / ha / (1.0 / ha + 1.0 / (la["ah"].get("away") or 2.0))
            pb = 1.0 / hb / (1.0 / hb + 1.0 / (lb["ah"].get("away") or 2.0))
            d = abs(pa - pb) * 100.0
            compared["ah_price"] = round(d, 2)
            max_diff = max(max_diff, d)

    status = "cross_source_disagreement" if max_diff > tolerance_pp else "ok"
    return {
        "status": status,
        "n_sources": len(names),
        "max_pp_diff": round(max_diff, 2),
        "compared": compared,
        "sources": names,
    }


def _final_decision_payload(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact, JSON-safe summary of the decision engine's ACTUAL final pick.

    ``decision`` is already ``decision_to_dict``-shaped (Candidate -> dict).
    The tier label alone (``decision_type``) cannot tell a settled-match
    scorer which market/selection was really chosen, so the snapshot persists
    the pick itself: market, selection, model_prob, market_odds, edge_pp, ev,
    plus the calibration bucket info from score_breakdown.top.

    Returns None when the engine produced no final pick (NO BET / NO CLEAR
    DECISION / MARKET PRIOR) or when the decision dict is absent.
    """
    if not isinstance(decision, dict):
        return None
    fd = decision.get("final_decision")
    if not isinstance(fd, dict):
        return None
    top = (decision.get("score_breakdown") or {}).get("top") or {}
    out: dict[str, Any] = {
        "market": fd.get("market"),
        "selection": fd.get("selection"),
        "model_prob": fd.get("model_prob"),
        "market_odds": fd.get("market_odds"),
        "implied_prob": fd.get("implied_prob"),
        "edge_pp": fd.get("edge_pp"),
        "ev": fd.get("ev"),
        "n_bucket": top.get("n_bucket"),
        "pick_status": top.get("pick_status"),
    }
    if all(v is None for v in out.values()):
        return None
    return out


def extract_market_totals(
    payload: dict[str, Any],
    prefer_bookmaker: str | None = None,
) -> dict[str, dict[str, float]]:
    """Per-bookmaker-pair totals + BTTS markets from an odds payload (TODO-04).

    Both sides of every Over/Under (and BTTS Yes/No) pair come from the SAME
    bookmaker -- the bookmaker with the smallest margin on that line -- so
    margin removal happens exactly once. The previous best-of-both-sides
    behaviour (max price per side across different bookmakers) double-removed
    margin and inflated the model's apparent edge on totals/BTTS.

    ``prefer_bookmaker`` (2026-08-23): when that bookmaker offers a pair for
    a line it wins the line outright regardless of margin (sharp-book
    convention, mirrors ``scorer.consensus_odds(primary_bookmaker=...)``);
    lines it does not quote keep the smallest-margin winner.

    Output shape is unchanged for consumers (``decision.build_candidates``
    reads ``market_totals["Over 2.5"]["odds"]`` etc.):
    ``{"Over 2.5": {odds, point, bookmaker}, "Under 2.5": {...},
      "BTTS Yes": {odds, point: None, bookmaker}, ...}``
    """
    market_totals: dict[str, dict[str, float]] = {}
    pref = str(prefer_bookmaker or "").strip().lower() or None
    for bm in payload.get("bookmakers", []):
        bm_name = bm.get("title", "?")
        is_preferred = bool(pref) and str(bm_name).strip().lower() == pref
        over_by_point: dict[Any, float] = {}
        under_by_point: dict[Any, float] = {}
        over_open_by_point: dict[Any, float] = {}
        under_open_by_point: dict[Any, float] = {}
        open_point_by_point: dict[Any, float] = {}
        btts: dict[str, float] = {}
        btts_open: dict[str, float] = {}
        for market in bm.get("markets", []):
            mkey = market.get("key")
            if mkey == "totals":
                for outcome in market.get("outcomes", []):
                    name = (outcome.get("name") or "").lower()
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)
                    opening = outcome.get("opening_price")
                    opening_point = outcome.get("opening_point")
                    if not price or price <= 1.0:
                        continue
                    if name in ("over", "more"):
                        over_by_point[point] = price
                        if opening and opening > 1.0:
                            over_open_by_point[point] = opening
                        if opening_point is not None:
                            open_point_by_point[point] = opening_point
                    elif name in ("under", "less"):
                        under_by_point[point] = price
                        if opening and opening > 1.0:
                            under_open_by_point[point] = opening
                        if opening_point is not None:
                            open_point_by_point[point] = opening_point
            elif mkey == "btts":
                for outcome in market.get("outcomes", []):
                    name = (outcome.get("name") or "").lower()
                    price = outcome.get("price", 0)
                    opening = outcome.get("opening_price")
                    if not price or price <= 1.0:
                        continue
                    if name in ("yes", "both teams to score", "btts"):
                        btts["yes"] = price
                        if opening and opening > 1.0:
                            btts_open["yes"] = opening
                    elif name in ("no", "any other result", "none"):
                        btts["no"] = price
                        if opening and opening > 1.0:
                            btts_open["no"] = opening

        # Per point: only a bookmaker with BOTH sides can provide a fair pair;
        # pick the bookmaker with the smallest margin (most consistent line),
        # unless a preferred (sharp) bookmaker quotes the pair outright.
        for point in over_by_point:
            if point not in under_by_point:
                continue
            o, u = over_by_point[point], under_by_point[point]
            margin = (1.0 / o + 1.0 / u) if (o > 1.0 and u > 1.0) else float("inf")
            for label, price, op in (
                (f"Over {point}", o, over_open_by_point.get(point)),
                (f"Under {point}", u, under_open_by_point.get(point)),
            ):
                existing = market_totals.get(label)
                ex_pref = bool(existing and existing.get("_preferred"))
                wins = (
                    existing is None
                    or (is_preferred and not ex_pref)
                    or (is_preferred == ex_pref and margin < existing.get("_margin", float("inf")))
                )
                if wins:
                    market_totals[label] = {
                        "odds": price, "point": point,
                        "bookmaker": bm_name, "_margin": margin,
                        "_preferred": is_preferred,
                    }
                    if op:
                        market_totals[label]["opening"] = op
                    if point in open_point_by_point:
                        market_totals[label]["opening_point"] = open_point_by_point[point]
        if btts.get("yes") and btts.get("no"):
            o, u = btts["yes"], btts["no"]
            margin = (1.0 / o + 1.0 / u) if (o > 1.0 and u > 1.0) else float("inf")
            for label, price, op in (
                ("BTTS Yes", o, btts_open.get("yes")),
                ("BTTS No", u, btts_open.get("no")),
            ):
                existing = market_totals.get(label)
                ex_pref = bool(existing and existing.get("_preferred"))
                wins = (
                    existing is None
                    or (is_preferred and not ex_pref)
                    or (is_preferred == ex_pref and margin < existing.get("_margin", float("inf")))
                )
                if wins:
                    market_totals[label] = {
                        "odds": price, "point": None,
                        "bookmaker": bm_name, "_margin": margin,
                        "_preferred": is_preferred,
                    }
                    if op:
                        market_totals[label]["opening"] = op
    for entry in market_totals.values():
        entry.pop("_margin", None)
        entry.pop("_preferred", None)
    return market_totals


def _merge_missing_btts(
    primary: dict[str, dict[str, Any]] | None,
    secondary: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Fill BTTS Yes/No lines missing from ``primary`` from ``secondary``.

    Re-prioritas 2026-08-24: the BTTS market exists ONLY in the
    OddsPapi/TheOddsAPI payload shape -- NowGoal serves no such market. When
    NowGoal is the primary payload its totals carry no BTTS prices, so the
    secondary source's BTTS entries are copied in verbatim. Only MISSING
    labels are filled (the primary stays the single writer for anything it
    already priced) and non-BTTS labels are never touched. Pure function.
    """
    out = dict(primary or {})
    for label in ("BTTS Yes", "BTTS No"):
        if out.get(label):
            continue
        val = (secondary or {}).get(label)
        if val:
            out[label] = dict(val)
    return out


def build_engine_stack(cfg: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """Construct (elo, poisson, ensemble, calibrator, scorer) from config.

    Shared by find_specific_match and the !best / !bestgoalmatch commands so
    the SAME validated model stack backs every prediction path.
    """
    from .calibration import Calibrator, SignalScorer
    from .elo import EloModel
    from .models import Ensemble, PoissonModel

    elo_cfg = cfg.get("models", {}).get("elo", {})
    poisson_cfg = cfg.get("models", {}).get("poisson", {})
    ens_cfg = cfg.get("models", {}).get("ensemble", {})
    cal_cfg = cfg.get("models", {}).get("calibration", {})
    ss_cfg = cfg.get("models", {}).get("signal_scorer", {})
    elo = EloModel(
        k=elo_cfg.get("k", 32.0),
        home_advantage=elo_cfg.get("home_advantage", 65.0),
        initial_rating=elo_cfg.get("initial_rating", 1500.0),
        base_total_goals=elo_cfg.get("base_total_goals", 2.7),
        path=ROOT / elo_cfg.get("file", "cache/football/elo.json"),
    )
    poisson = PoissonModel(
        base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
        base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
        dc_rho=poisson_cfg.get("dc_rho", -0.1),
        shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
        time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
        xg_weight=poisson_cfg.get("xg_weight", 0.65),
        min_samples=poisson_cfg.get("min_samples", 2),
        # Plan v3 (2026-08-24): F1 Elo-anchor + F2 market-total calibration.
        elo_anchor=poisson_cfg.get("elo_anchor"),
        market_total_calibration=poisson_cfg.get("market_total_calibration"),
    )
    ensemble = Ensemble(
        elo_weight=ens_cfg.get("elo_weight", 0.5),
        poisson_weight=ens_cfg.get("poisson_weight", 0.5),
    )
    calibrator = Calibrator(
        path=ROOT / cal_cfg.get("file", "cache/football/calibration.json"),
        min_samples=cal_cfg.get("min_samples", 200),
    )
    scorer = SignalScorer(conf_weights=ss_cfg.get("conf_weights"))
    return elo, poisson, ensemble, calibrator, scorer


_ML_PREDICTOR_CACHE: Any = None


def _ml_probs_for(
    cfg: dict[str, Any], league: str, home: str, away: str, kickoff: str | None
) -> dict[str, Any] | None:
    """Trained-model 1X2 probabilities for one match, or None.

    None covers every failure mode (no models trained, league without cached
    history, thin season form) -- the caller's Elo+Poisson engine then runs
    unchanged. The predictor (history + models) is built once per process.
    """
    global _ML_PREDICTOR_CACHE
    if _ML_PREDICTOR_CACHE is None:
        try:
            from .ml_predict import MlPredictor

            ml_cfg = cfg.get("models", {}).get("ml", {})
            _ML_PREDICTOR_CACHE = MlPredictor(
                ROOT / ml_cfg.get("models_dir", "cache/football/models"),
                window=int(ml_cfg.get("window", 5)),
                gd_margin=int(ml_cfg.get("gd_margin", 2)),
            )
        except Exception as exc:  # noqa: BLE001 -- ML is an additive signal
            logger.warning("ml predictor unavailable: %s", exc)
            _ML_PREDICTOR_CACHE = False
    if not _ML_PREDICTOR_CACHE:
        return None
    try:
        key = (league, home, away, (kickoff or "")[:10])
        item = _ML_PREDICTOR_CACHE.predict_matches([key]).get(key)
        return (item or {}).get("1x2") or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ml predict failed (prediction unaffected): %s", exc)
        return None


def run_decision_engine(
    prediction: dict[str, Any] | None,
    consensus: dict[str, float],
    market_totals: dict[str, dict[str, float]],
    has_odds: bool,
    bookmakers_count: int,
    cfg: dict[str, Any],
    similar_signal: dict[str, Any] | None = None,
    hard_cap_medium: bool = False,
    form_depth_shallow: bool = False,
    ml_probs: dict[str, float] | None = None,
    league: str | None = None,
    league_calibrated: bool = True,
    movement: dict[str, Any] | None = None,
    benchmark_ts: str | None = None,
    benchmark_max_age_hours: float = 24.0,
) -> dict[str, Any]:
    """Phase 1 decision engine (S23-S31) over the INDEPENDENT engine output.

    Transparent Decision Score; NO CLEAR DECISION / NO BET are valid outputs.
    Shared by find_specific_match and the !best commands so the winner pick
    uses exactly the validated decision logic. Returns a JSON-safe dict
    (never a dataclass); caller wraps in try/except if it wants the flow to
    continue on failure.

    Correction spec: Model A (odds-implied) vs Model B (independent)
    disagreement is computed here and the resulting flag drives
    REVIEW_REQUIRED + confidence caps. The Section 2 hard gates (EV > 3%,
    |edge| < 20pp, n_bucket >= 30, completeness >= 0.6) are active on this
    production path via ``bucket_n``.
    """
    from .decision import (
        MARKET_PRIOR_MIN_COMPLETENESS,
        build_candidates,
        decide,
        decision_to_dict,
        market_prior_decision,
    )
    from .model_gates import bucket_n, model_a_probs, model_disagreement

    dec_cfg = cfg.get("models", {}).get("decision", {})
    weights = dec_cfg.get("weights") or None
    # Phase 2.1: selection-filter reasons (populated only when the filter is
    # enabled; initialized here so the thin-data path never hits UnboundLocal).
    _sel_reasons: list[str] = []
    # Phase 3.1: market-blend metadata (same initialization discipline).
    _blend_meta: dict[str, Any] | None = None
    # MARKET PRIOR (thin-data honesty): when the independent engine has no
    # usable signal (no form/history/Elo seed) the best available estimator
    # is the market itself. Predictions (1X2/O-U/BTTS) are built from the
    # margin-free market -- explicitly labelled, edge = 0, betting advice
    # NO BET. Config-gated (models.decision.market_prior, default off) so
    # existing behaviour is unchanged until enabled.
    market_prior_enabled = bool(dec_cfg.get("market_prior", False))
    # Keep the MARKET PRIOR floor in sync with the engine's bettable
    # completeness floor by default (explicit config still wins).
    market_prior_min_completeness = float(
        dec_cfg.get(
            "market_prior_min_completeness",
            dec_cfg.get("min_completeness", MARKET_PRIOR_MIN_COMPLETENESS),
        )
    )

    # Section 1: Model A (odds-implied, reference-only) vs Model B
    # (independent Elo+Poisson) disagreement check — mandatory, cannot be
    # bypassed even when Model B calibration is otherwise strong.
    model_a = None
    disagreement = False
    if has_odds and consensus.get("home", 0) > 0 and prediction is not None:
        model_a = model_a_probs(consensus, market_totals)
        b1x2 = (prediction.get("model_probs") or {}).get("1x2")
        if model_a and b1x2:
            disagreement = model_disagreement(
                model_a["1x2"], b1x2,
                threshold_pp=dec_cfg.get("disagreement_threshold_pp", 15.0),
            )["flag"]

    # Thin-data gate: when the independent engine ran but its input
    # completeness is below the MARKET PRIOR floor, its probabilities carry
    # no usable signal -- fall back to the honest market-mirror prediction
    # (edge = 0, betting advice NO BET) instead of pretending the model saw
    # enough data. NOTE: an UNcalibrated league is no longer forced here --
    # its model probabilities are real (feature Poisson + Elo prior), so it
    # is decided normally with calibration_quality=0 (confidence capped,
    # NO BET unless a candidate clears every gate).
    engine_completeness = (
        (prediction or {}).get("data_completeness") or 0.0
    )
    thin_data = bool(
        prediction is None
        or not prediction.get("model_probs")
        or engine_completeness < market_prior_min_completeness
    )

    if thin_data:
        if market_prior_enabled and has_odds and consensus.get("home", 0) > 0:
            decision = market_prior_decision(
                consensus,
                market_totals,
                bookmakers_count=bookmakers_count,
                min_bookmakers=dec_cfg.get("min_bookmakers", 3),
            )
        else:
            decision = {
                "decision_type": "NO CLEAR DECISION",
                "final_decision": None,
                "most_likely": None,
                "explanation": "Engine independen tidak berjalan (data form/history kurang).",
                "reasons": ["no independent model"],
                "edge_warnings": [],
                "score_breakdown": {},
            }
    elif prediction is not None and prediction.get("model_probs"):
        ag = prediction.get("agreement") or {}
        agreement = ag.get("model_vs_market")
        if agreement is None and ag.get("model_vs_model") is not None:
            agreement = ag["model_vs_model"]
        if agreement is None:
            agreement = 0.0
        calib = prediction.get("calibration") or {}
        hist = 0.5  # neutral until a sufficient similar-signal sample
        ss = similar_signal or {}
        if (ss.get("matching") or {}).get("sufficient_sample"):
            roi = (ss["matching"] or {}).get("roi")
            hist = max(0.0, min(1.0, 0.5 + (roi if roi is not None else 0.0)))
        # Phase 3.1: probability blend (flag-gated, off by default).
        # p_final = a * p_market(margin-free, freshest T-1h) + (1-a) *
        # p_model_calibrated, with a = 1 - calibration.quality (boundary:
        # quality=0 -> a=1 -> pure market, edge=0, NO BET). The blend runs
        # BEFORE candidate building so the decision edge is measured against
        # the blended probability.
        _blend_meta: dict[str, Any] | None = None
        _blend_probs = (prediction or {}).get("model_probs") or {}
        _mb_cfg = dec_cfg.get("market_blend") or {}
        if bool(_mb_cfg.get("enabled", False)) and consensus.get("home", 0) > 0:
            from .decision import blend_model_with_market
            from .decision import margin_free_implied as _mfi

            _blend_probs, _blend_meta = blend_model_with_market(
                _blend_probs,
                _mfi(consensus),
                calib.get("quality", 0.0),
            )
        cands = build_candidates(
            model_probs=_blend_probs if _blend_meta is not None else prediction["model_probs"],
            consensus_odds=consensus if has_odds else {},
            market_totals=market_totals,
            independent=True,
        )
        # Phase 2.1: config-driven market/league selection filter (flag-gated,
        # off by default -- existing behavior unchanged until enabled). Big-5
        # 1X2 becomes calibration/sanity-check only, never an actionable pick.
        # When every candidate is filtered out, ``decide([])`` returns NO
        # CLEAR DECISION naturally; the reasons are appended below.
        _sel_reasons: list[str] = []
        _sel_cfg = (cfg.get("models") or {}).get("decision", {}).get("selection") or {}
        if bool(_sel_cfg.get("enabled", False)):
            from .decision import selection_filter

            cands, _sel_reasons = selection_filter(cands, cfg, league)
        # ML (trained-model) agreement: 1 - mean |ml - ensemble| over 1X2.
        # Only counts when BOTH the ML model and the independent engine
        # produced probabilities; the weight is config-gated (default 0).
        b1x2 = (prediction.get("model_probs") or {}).get("1x2") or {}
        ml_agreement = None
        if ml_probs and b1x2 and all(k in ml_probs for k in ("home", "draw", "away")):
            mean_abs_diff = sum(
                abs(ml_probs[k] - b1x2.get(k, 0.0)) for k in ("home", "draw", "away")
            ) / 3.0
            ml_agreement = max(0.0, min(1.0, 1.0 - mean_abs_diff / (2.0 / 3.0)))
        decision = decide(
            cands,
            model_agreement=agreement,
            calibration_quality=calib.get("quality", 0.0),
            calibration_samples=calib.get("samples", 0),
            completeness=prediction.get("data_completeness", 0.0),
            bookmakers_count=bookmakers_count,
            historical_reliability=hist,
            weights=weights,
            edge_warning_pp=dec_cfg.get("edge_warning_pp", 10.0),
            edge_extreme_pp=dec_cfg.get("edge_extreme_pp", 20.0),
            min_bookmakers=dec_cfg.get("min_bookmakers", 3),
            min_edge_pp=dec_cfg.get("min_edge_pp", 2.0),
            strong_score=dec_cfg.get("strong_score", 0.70),
            good_score=dec_cfg.get("good_score", 0.55),
            lean_score=dec_cfg.get("lean_score", 0.40),
            no_clear_max_score=dec_cfg.get("no_clear_max_score", 0.35),
            best_prob_only=bool(dec_cfg.get("best_prob_only", False)),
            bucket_n=bucket_n,
            min_bucket_n=dec_cfg.get("min_bucket_n", 200),
            min_bucket_ci_halfwidth=dec_cfg.get("min_bucket_ci_halfwidth", None),
            min_ev=dec_cfg.get("min_ev", 0.03),
            min_completeness=dec_cfg.get("min_completeness", 0.6),
            disagreement=disagreement,
            hard_cap_medium=hard_cap_medium,
            form_depth_shallow=form_depth_shallow,
            model_calibration_score=calib.get("quality", 0.0),
            # TODO-09/10/16: reliability gates (WATCH tier + variance-aware
            # EV) are OPT-IN via config so existing behavior is unchanged
            # until enabled. Uncertainty = ensemble spread (model_probs).
            enable_watch=bool(dec_cfg.get("enable_watch", False)),
            uncertainty=float(
                (prediction.get("model_probs") or {}).get("uncertainty", 0.0) or 0.0
            ),
            ml_agreement=ml_agreement,
            movement=(movement or {}).get("agreement"),
        )
    result = decision_to_dict(decision)
    result["model_disagreement"] = model_disagreement(
        (model_a or {}).get("1x2"),
        (prediction or {}).get("model_probs", {}).get("1x2") if prediction else None,
        threshold_pp=dec_cfg.get("disagreement_threshold_pp", 15.0),
    )
    result["model_a"] = model_a  # reference-only probabilities (no edge/EV)
    result["edge_benchmark"] = edge_benchmark(cfg)  # Phase 2: label the edge benchmark
    # Phase 2.1: record when the selection filter removed markets/leagues.
    if _sel_reasons:
        result["selection_filtered"] = True
        result["selection_reasons"] = list(_sel_reasons)
        result["reasons"] = (result.get("reasons") or []) + list(_sel_reasons)
    # Phase 3.1: record the blend metadata; pure-market blend (a=1, edge=0)
    # is a NO BET by construction.
    if _blend_meta is not None:
        result["blend"] = _blend_meta
        if _blend_meta.get("pure_market"):
            result["decision_type"] = "NO BET"
            result["final_decision"] = None
            result["betting_advice"] = "NO BET"
            result["reasons"] = (result.get("reasons") or []) + [
                "market blend: kalibrasi 0 -> alpha=1 -> pure market, edge=0 (NO BET)"
            ]
    # Phase 0.2: stamp the odds observation timestamp on every edge's
    # benchmark and enforce the 24h freshness limit as an EXPLICIT check --
    # an edge computed from a stale benchmark is invalid and must never drive
    # a recommendation (displayed as edge: n/a, decision forced NO BET).
    from .edge_benchmark import edge_benchmark_status

    _eb = result["edge_benchmark"]
    _eb["ts"] = benchmark_ts
    _eb.update(edge_benchmark_status(benchmark_ts, max_age_hours=benchmark_max_age_hours))
    result["edge_benchmark"] = _eb
    if _eb.get("stale") and result.get("decision_type") not in {"MARKET PRIOR", "NO CLEAR DECISION", "NO BET"}:
        result["edge_benchmark_stale"] = True
        result["edge_invalid"] = True
        result["decision_type"] = "NO BET"
        result["final_decision"] = None
        result["betting_advice"] = "NO BET"
        result["reasons"] = (result.get("reasons") or []) + [
            f"edge benchmark stale ({_eb.get('age_hours')}h > {benchmark_max_age_hours:.0f}h) — edge: n/a"
        ]
    # Uncalibrated-league honesty: the model ran (feature Poisson + Elo), but
    # there is no per-league calibration fit -- label it so the prediction is
    # never mistaken for a calibrated/validated one.
    if not league_calibrated and result.get("decision_type") not in {"MARKET PRIOR", "NO CLEAR DECISION"}:
        result["uncalibrated_league"] = True
        result["edge_warnings"] = (result.get("edge_warnings") or []) + [
            "liga tanpa kalibrasi per-league — prediksi model mentah (Elo+Poisson), "
            "confidence dibatasi; saran NO BET kecuali edge lolos gate"
        ]
    # Plan B movement signal: surface it, and (when configured) require the
    # model side to agree with the steam side — model vs market drift
    # disagreement is a NO BET.
    result["movement"] = movement
    _mv_cfg = (cfg.get("models") or {}).get("movement") or {}
    if (
        bool(_mv_cfg.get("require_movement_agreement", False))
        and movement
        and movement.get("usable")
        and movement.get("agreement") == 0.0
        and result.get("decision_type") in {"STRONG", "GOOD", "LEAN", "WATCH"}
    ):
        result["decision_type"] = "NO BET"
        result["final_decision"] = None
        result["betting_advice"] = "NO BET"
        result["reasons"] = (result.get("reasons") or []) + [
            f"movement: steam ke '{movement.get('steam_side')}' berlawanan model — NO BET"
        ]
    # Phase 3 CLV hard gate: a segment (league x market x tier) may only emit
    # an actionable decision when it has >= min_bets settled bets AND realized
    # price CLV > 0. Failing segments are demoted to NO BET with the reason.
    _clv_cfg = (cfg.get("models") or {}).get("decision", {}).get("clv_gate") or {}
    result["clv_gate"] = None
    _ACTIONABLE = {"STRONG", "GOOD", "LEAN", "WATCH"}
    if (
        bool(_clv_cfg.get("enabled", False))
        and league
        and result.get("decision_type") in _ACTIONABLE
    ):
        _fd = result.get("final_decision") or {}
        _market = _fd.get("market") or (result.get("most_likely") or {}).get("market")
        if _market:
            _pl_cfg = cfg.get("prediction_log") or {}
            try:
                _stats = load_segment_stats(
                    ROOT / (_pl_cfg.get("file") or "cache/football/predictions.jsonl")
                )
                _gate = gate_segment(
                    _stats,
                    league=league,
                    market=_market,
                    tier=result["decision_type"],
                    min_bets=int(_clv_cfg.get("min_bets", 200)),
                    require_roi_positive=bool(_clv_cfg.get("require_roi_positive", True)),
                    max_ci_halfwidth=(
                        float(_clv_cfg["max_ci_halfwidth"])
                        if _clv_cfg.get("max_ci_halfwidth") is not None else None
                    ),
                )
                result["clv_gate"] = _gate
                if not _gate["allowed"]:
                    result["decision_type"] = "NO BET"
                    result["final_decision"] = None
                    result["betting_advice"] = "NO BET"
                    result["reasons"] = (result.get("reasons") or []) + [
                        f"CLV gate: {_gate['reason']}"
                    ]
            except Exception as exc:  # noqa: BLE001 -- gate must never break the flow
                logger.warning("clv gate lookup failed (decision unaffected): %s", exc)
    # Phase 5.4 HARD filter: edge must not drive a recommendation in an
    # edge bucket that is net-negative against CLOSING prices (audited on a
    # schedule). Implemented as a hard demotion to NO BET -- never a warning
    # label. Config: models.decision.edge_bucket_gate.{enabled, min_n}.
    _ebg_cfg = dec_cfg.get("edge_bucket_gate") or {}
    result["edge_bucket_gate"] = None
    _ACTIONABLE = {"STRONG", "GOOD", "LEAN", "WATCH"}
    if (
        bool(_ebg_cfg.get("enabled", False))
        and league
        and result.get("decision_type") in _ACTIONABLE
    ):
        _fd = result.get("final_decision") or {}
        _edge_pp = _fd.get("edge_pp")
        if _edge_pp is not None:
            try:
                from .prediction_log import (
                    edge_bucket_closing_stats,
                    edge_bucket_gate,
                )

                _pl_cfg = cfg.get("prediction_log") or {}
                _bstats = edge_bucket_closing_stats(
                    ROOT / (_pl_cfg.get("file") or "cache/football/predictions.jsonl")
                )
                _bg = edge_bucket_gate(
                    _bstats, _edge_pp, min_n=int(_ebg_cfg.get("min_n", 10))
                )
                result["edge_bucket_gate"] = _bg
                if not _bg["allowed"]:
                    result["decision_type"] = "NO BET"
                    result["final_decision"] = None
                    result["betting_advice"] = "NO BET"
                    result["reasons"] = (result.get("reasons") or []) + [
                        f"Edge bucket gate: {_bg['reason']}"
                    ]
            except Exception as exc:  # noqa: BLE001 -- gate must never break the flow
                logger.warning("edge bucket gate failed (decision unaffected): %s", exc)
    # Phase 2.3: the actionable gate (flag-gated, off by default) replaces
    # the old STRONG/GOOD/LEAN tiers as the driver of actionable picks: a
    # pick is actionable only when its segment CLV > 0 (clv_gate), its edge
    # is computed against a FRESH benchmark (Phase 0.2) and meets the
    # threshold, and the league passes the calibration minimum (Phase 1.5).
    _ag_cfg = dec_cfg.get("actionable_gate") or {}
    result["actionable"] = None
    if bool(_ag_cfg.get("enabled", False)) and result.get("decision_type") in _ACTIONABLE:
        from .decision import actionable_gate

        _fd = result.get("final_decision") or {}
        _ok, _ag_reasons = actionable_gate(
            league_calibrated=league_calibrated,
            edge_pp=_fd.get("edge_pp"),
            min_edge_pp=float(_ag_cfg.get("min_edge_pp", dec_cfg.get("min_edge_pp", 3.0))),
            benchmark_stale=bool(result.get("edge_benchmark_stale")),
            clv_gate=result.get("clv_gate"),
            cfg=cfg,
        )
        result["actionable"] = _ok
        if not _ok:
            result["decision_type"] = "NO BET"
            result["final_decision"] = None
            result["betting_advice"] = "NO BET"
            result["reasons"] = (result.get("reasons") or []) + [
                f"Actionable gate: {r}" for r in _ag_reasons
            ]
    # Phase 6 staking: recommended fractional-Kelly stake for the final
    # decision. Extreme edge is auto-declined (never reaches a stake number).
    # Phase 4.3: Kelly staking is HARD-DISABLED for any league that has not
    # passed the Phase 1.5 per-league calibration minimum -- an uncalibrated
    # league must never produce a stake number (spec iron rule).
    result["stake"] = None
    _stk_cfg = (cfg.get("models") or {}).get("staking") or {}
    _fd_stake = result.get("final_decision") or {}
    if (
        bool(_stk_cfg.get("enabled", False))
        and result.get("decision_type") in _ACTIONABLE
        and _fd_stake
    ):
        if not league_calibrated:
            result["stake_disabled_reason"] = (
                "liga belum lulus kalibrasi minimum per-league (Phase 1.5) — "
                "Kelly staking hard-disabled"
            )
        else:
            from .staking import compute_stake

            result["stake"] = compute_stake(
                model_prob=_fd_stake.get("model_prob", 0.0),
                decimal_odds=_fd_stake.get("market_odds", 0.0),
                edge_pp=_fd_stake.get("edge_pp", 0.0),
                decision_type=result["decision_type"],
                cfg=cfg,
                edge_extreme_pp=dec_cfg.get("edge_extreme_pp", 20.0),
            )
    if ml_probs:
        result["ml"] = {
            "probabilities": {
                k: round(float(v), 4) for k, v in ml_probs.items() if k != "model"
            },
            "model": ml_probs.get("model"),
            "agreement": round(ml_agreement, 3) if ml_agreement is not None else None,
        }
    return result


async def resolve_or_detect_league(
    *,
    league_query: str | None,
    home_query: str,
    away_query: str,
    stats: MultiSourceStatsFetcher,
) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
    """Resolve a league query, or DETECT the league from the fixture (D2).

    Dynamic league discovery (2026-08-17): when the user's league keyword is
    unknown OR absent, the league is read FROM THE FIXTURE instead of the
    whitelist -- the flashscore homepage fallback (and, when available, the
    LiveScore date feed) finds ``home vs away`` and its competition title
    becomes a deterministic ``dyn:`` league key. Returns
    ``(league_key, meta, detected_fixture)`` or None when nothing resolves.
    ``detected_fixture`` (flashscore/livescore identity, when detection ran)
    carries the competition + kickoff the pipeline would otherwise miss.
    """
    if league_query:
        resolved = resolve_league_scored(league_query)
        if resolved:
            key, meta = resolved
            return key, meta, None

    # Fixture-first: find the match, read the competition from it.
    detected: dict[str, Any] | None = None
    try:
        fc = getattr(stats, "fc", None)
        if fc is not None and getattr(fc, "available", True):
            # resolve_match(None) already falls back to the homepage for
            # unregistered leagues (flashscore.py:1435).
            _d = await fc.resolve_match(None, home_query, away_query)
            # isinstance guard: mocks / coroutine-like results are NOT a
            # resolved fixture -- treat as undetected, never crash.
            if isinstance(_d, dict):
                detected = _d
    except Exception as exc:  # noqa: BLE001 -- detection must never raise
        logger.warning("league detect via flashscore failed (dynamic league): %s", exc)
        detected = None
    if detected is None or not (detected.get("competition") or "").strip():
        # The flashscore team-fixtures fallback returns the fixture WITHOUT a
        # competition tag (only homepage rows carry one). Without a league
        # there is nothing to analyse -- fall through to the LiveScore date
        # feed, whose rows carry the competition title, before giving up.
        try:
            from .source_match import _search_livescore_any

            detected = await _search_livescore_any(stats, home_query, away_query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("league detect via livescore failed (dynamic league): %s", exc)
            detected = None
    if not detected:
        return None

    competition = (detected.get("competition") or "").strip()
    if not competition:
        # The fixture exists but carries no competition title -- honest
        # failure: without a league we cannot run the analysis.
        return None
    try:
        from .league_resolver import dynamic_league_key, dynamic_league_meta

        key = competition_league_key(competition) or dynamic_league_key(competition)
        if key in _registered_league_keys():
            meta = load_leagues().get(key) or dynamic_league_meta(competition)
        else:
            meta = dynamic_league_meta(competition, country=detected.get("country"))
    except Exception as exc:  # noqa: BLE001 -- detection must never break analyse
        logger.warning("dynamic league meta build failed: %s", exc)
        return None
    return key, meta, detected


def _registered_league_keys() -> set[str]:
    try:
        from .league_resolver import load_leagues

        return set(load_leagues())
    except Exception:  # noqa: BLE001
        return set()


async def find_specific_match(
    *,
    league_query: str | None,
    home_query: str,
    away_query: str,
    cfg: dict[str, Any],
    odds: OddsFetcher,
    stats: MultiSourceStatsFetcher,
    cache: Cache,
    oddspapi: Any = None,
    nowgoal: Any = None,
    source_match: dict[str, Any] | None = None,
    league_key: str | None = None,
    league_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyse one specific match (``analisa match``, ``!livescore``,
    ``!flashscore``).

    ``source_match`` is the optional context collected by the match-source
    commands (``agents/football/source_match.py``): the match identity
    (home/away/kickoff/competition) already validated against LiveScore or
    Flashscore plus that source's extra data. When present it (a) narrows the
    NowGoal odds lookup to the validated kickoff date, (b) supplies the
    kickoff when neither the fixture endpoint nor the odds payload carries
    one, and (c) is attached to the result for provenance. The prediction
    pipeline itself is unchanged and identical for every entry path.
    """
    # Shared analysis budget (multi_source clock): every provider fallback
    # consults the SAME clock, so a slow/blocked provider skips expensive
    # steps instead of stacking per-provider HTTP timeouts past the runner's
    # hard deadline (HERMES_RUNNER_DEADLINE, default 340s; the bot kills the
    # subprocess at 380s). Budget is configurable; config/football.json sets
    # 300.0s, a comfortable margin under the runner deadline.
    from .multi_source import set_analysis_budget, analysis_remaining

    budget_seconds = float(
        (cfg.get("analyse") or {}).get("budget_seconds", 72.0)
    )
    set_analysis_budget(budget_seconds)
    _t_start = _time.monotonic()
    # D2 (dynamic league discovery): when the league keyword is unknown or
    # absent, read the league FROM THE FIXTURE (flashscore homepage /
    # livescore date feed) and build a deterministic ``dyn:`` key instead of
    # failing with "liga tidak dikenal". ``detected`` carries the resolved
    # fixture identity (competition + kickoff) for the pipeline below.
    # ``league_key``/``league_meta`` (optional) short-circuit detection when
    # the caller already resolved the league (find_source_match passes them).
    if league_key is None or league_meta is None:
        _resolved = await resolve_or_detect_league(
            league_query=league_query,
            home_query=home_query,
            away_query=away_query,
            stats=stats,
        )
        if _resolved is None:
            return {
                "error": (
                    f"liga '{league_query}' tidak dikenal dan fixture tidak ditemukan "
                    f"({home_query} vs {away_query})"
                ),
                "teams": [],
            }
        league_key, meta, detected = _resolved
    else:
        meta, detected = league_meta, None
    if detected and source_match is None:
        # Detection carried the fixture identity -- feed it to the pipeline
        # as the source_match context (competition + kickoff) so odds
        # lookup / kickoff checks see the validated values.
        source_match = {
            "source": detected.get("source") or "flashscore",
            "home": detected.get("home") or home_query,
            "away": detected.get("away") or away_query,
            "kickoff": detected.get("kickoff"),
            "competition": detected.get("competition"),
            "league_key": league_key,
            **detected,
        }
    display = meta["display"]

    # Identity firewall G-A (plan 2026-08-24): query pair vs DETECTED fixture.
    # When the detection already rendered a DIFFERENT home/away club than the
    # user asked for (the F3 class: "Atletico Madrid" resolved onto Real
    # Madrid and a phantom fixture analysed), refuse BEFORE any browser
    # render or provider chain burns budget. Fail-open on undecidable names.
    _fw = _identity_firewall_cfg(cfg)
    if _fw.get("enabled", True) and _fw.get("refuse_divergence", True):
        try:
            from .identity_gate import check_pair_identity, refusal_payload

            _fwA = check_pair_identity(
                home_query,
                away_query,
                league_key=league_key,
                detected_home=(source_match or {}).get("home"),
                detected_away=(source_match or {}).get("away"),
            )
            if _fwA.get("status") == "refuse":
                logger.warning(
                    "identity_gate G-A refuse %s vs %s: %s",
                    home_query, away_query, _fwA.get("reasons"),
                )
                return refusal_payload(
                    "preflight_fixture",
                    _fwA.get("reasons") or [],
                    display=display,
                    home_query=home_query,
                    away_query=away_query,
                    detail={"checks": _fwA.get("checks")},
                )
        except Exception as exc:  # noqa: BLE001 -- firewall never blocks analyse
            logger.warning("identity_gate G-A failed (proceed): %s", exc)

    odds_key = meta.get("odds_api_key")
    season = _season_now()
    meta_with_season = {**meta, "season": season, "_league_key": league_key}
    _dynamic_league = bool(meta.get("dynamic"))

    _t0 = _time.monotonic()
    # Odds + flashscore team resolve run in PARALLEL (they are independent):
    # the odds payload does not depend on the resolved team ids, so serial
    # execution used to delay the flashscore pair render behind the odds
    # fetch -- and a slow odds source could starve the flashscore resolve of
    # budget. gather() finishes both together.
    #
    # Source priority (2026-08): oddspapi is PRIMARY, then nowgoal, then The
    # Odds API last -- the oddspapi branch keys off its own find_fixture, so
    # it is the one running in parallel with the flashscore team resolve.
    match_odds_payload: dict[str, Any] | None = None
    oddspapi_source = False
    nowgoal_source = False
    home_team: dict[str, Any] | None = None
    away_team: dict[str, Any] | None = None

    async def _oddspapi_resolve() -> tuple[dict[str, Any] | None, tuple[dict[str, Any], dict[str, Any]] | None]:
        """OddsPapi find_fixture + fetch_odds (PRIMARY source)."""
        if oddspapi is None or _budget_short(analysis_remaining(), margin=30.0):
            return None, None
        try:
            odsp_fixture = await oddspapi.find_fixture(home_query, away_query)
            if odsp_fixture and odsp_fixture.get("hasOdds"):
                okey = f"oddspapi_{home_query}_{away_query}".replace("(", "").replace(")", "")
                cached_payload = cache.get(okey, ttl_seconds=3600) if cache is not None else None
                odsp_payload = cached_payload or await oddspapi.fetch_odds(odsp_fixture)
                if odsp_payload:
                    teams = (
                        {
                            "id": str(odsp_fixture.get("participant1Id")),
                            "name": odsp_fixture.get("participant1Name") or home_query,
                            "provider": "oddspapi",
                            "_role": "oddspapi",
                        },
                        {
                            "id": str(odsp_fixture.get("participant2Id")),
                            "name": odsp_fixture.get("participant2Name") or away_query,
                            "provider": "oddspapi",
                            "_role": "oddspapi",
                        },
                    )
                    if cache is not None and not cached_payload:
                        cache.set(okey, odsp_payload)
                    return odsp_payload, teams
        except Exception as exc:
            logger.warning("oddspapi resolve failed (prediction unaffected): %s", exc)
        return None, None

    _tasks = []
    if oddspapi is not None:
        _tasks.append(_oddspapi_resolve())
    _tasks.append(stats.search_teams_pair(home_query, away_query, meta_with_season))
    oddspapi_teams: tuple[dict[str, Any], dict[str, Any]] | None = None
    _results = await asyncio.gather(*_tasks, return_exceptions=True)
    oddspapi_result = _results[0] if oddspapi is not None else None
    team_result = _results[-1]
    if isinstance(oddspapi_result, BaseException):
        logger.warning("oddspapi resolve failed (parallel; prediction path unchanged): %s", oddspapi_result)
    elif oddspapi_result is not None:
        match_odds_payload, oddspapi_teams = oddspapi_result
        if match_odds_payload is not None:
            oddspapi_source = True
            logger.info("odds via oddspapi (PRIMARY)")
    if isinstance(team_result, BaseException):
        logger.warning("team pair resolve failed (parallel; provider chain unchanged): %s", team_result)
    else:
        home_team, away_team = team_result
    logger.info("analyse phases: oddspapi+search_teams_pair=%.1fs", _time.monotonic() - _t0)

    # Competition cross-check (root of misperception, 2026-08-17): the league
    # the user TYPED may differ from the competition the resolved fixture
    # actually belongs to. A flashscore resolve that landed via the homepage
    # carries the real competition section (e.g. "LaLiga2" for a match the
    # user queried as "la liga"); when it maps to a DIFFERENT registered
    # league, flag league_mismatch and pin the standings fetch to the correct
    # key so the table rendered is the right one. Falls back silently to the
    # user league when the competition is unregistered or absent.
    league_mismatch: dict[str, Any] | None = None
    standings_key = league_key
    _fs_match_comp = None
    try:
        _fs_match = (meta_with_season or {}).get("_flashscore_match") or {}
        if isinstance(_fs_match, dict):
            _fs_match_comp = _fs_match.get("competition")
    except Exception:  # noqa: BLE001 -- cross-check must never block analysis
        _fs_match_comp = None
    if _fs_match_comp:
        try:
            _actual_key = competition_league_key(str(_fs_match_comp))
        except Exception:  # noqa: BLE001
            _actual_key = None
        if _actual_key and _actual_key != league_key:
            league_mismatch = {
                "requested": league_key,
                "actual": _actual_key,
                "competition": str(_fs_match_comp),
            }
            standings_key = _actual_key
            logger.info(
                "league mismatch: user '%s' -> fixture competition '%s' -> '%s'",
                league_key, _fs_match_comp, _actual_key,
            )

    # P3 + RE-PRIORITAS 2026-08-24 ("NowGoal primary, OddsPapi validator"):
    # collect EVERY consulted odds payload so key lines can be cross-checked
    # across independent sources. NowGoal is now PRIMARY -- per-bookmaker
    # f/l/r prices + AH depth, no API quota; OddsPapi is SECONDARY whose
    # unique value is the BTTS market (NowGoal serves none) plus independent
    # cross-validation; TheOddsAPI stays LAST. Every fetch stays best-effort
    # within budget: a failure or no-data degrades to None and the pipeline
    # proceeds exactly like before the flip.
    _odds_payloads: dict[str, dict[str, Any]] = {}
    _odds_oddspapi_payload: dict[str, Any] | None = None
    if match_odds_payload is not None and oddspapi_source:
        _odds_payloads["oddspapi"] = match_odds_payload
        _odds_oddspapi_payload = match_odds_payload
        # Demote: the primary is chosen AFTER the nowgoal fetch below.
        match_odds_payload = None
    if nowgoal is not None and not _budget_short(analysis_remaining(), margin=30.0):
        try:
            # The source search already validated the match date, so narrow
            # the NowGoal schedule scan to that date (existing logic scans
            # today+tomorrow when no date is given -- the analyse path keeps
            # that behaviour since it has no source_match).
            ng_date = None
            if source_match and source_match.get("kickoff"):
                from .timeutil import wib_date_from_iso

                ng_date = wib_date_from_iso(source_match["kickoff"])
            ng_payload = await nowgoal.match_odds(home_query, away_query, ng_date)
            if ng_payload and ng_payload.get("bookmakers"):
                _odds_payloads["nowgoal"] = ng_payload
                nowgoal_source = True
                logger.info("odds PRIMARY via nowgoal")
                match_odds_payload = ng_payload
        except Exception as exc:
            logger.warning("nowgoal odds fetch failed (prediction unaffected): %s", exc)
    # Primary selection: nowgoal > oddspapi > theoddsapi (branch below).
    if match_odds_payload is None:
        match_odds_payload = _odds_oddspapi_payload
        if match_odds_payload is not None:
            logger.info("odds via oddspapi (nowgoal kosong)")

    # The Odds API (THIRD / LAST resort): only reached when both oddspapi
    # and nowgoal have no odds. Same normalized payload shape; cache + key
    # fallback (qualification variants) handled by find_match_odds_payload.
    if match_odds_payload is None and odds_key:
        odds_keys = [odds_key] + list(meta.get("odds_alt_keys") or [])
        try:
            odds_result = await find_match_odds_payload(
                odds_keys, home_query, away_query, odds, cache,
                cfg["cache_ttl_seconds"],
                home_query=home_query, away_query=away_query,
            )
            if odds_result and odds_result[0]:
                match_odds_payload = odds_result[0]
                _odds_payloads["theoddsapi"] = match_odds_payload
                logger.info("odds via The Odds API (oddspapi + nowgoal kosong)")
        except Exception as exc:
            logger.warning("odds payload fetch failed (prediction path unchanged): %s", exc)

    # P1.3: never discard a partially-resolved identity. The oddspapi fallback
    # fills ONLY the missing side / missing fields (same fill-missing principle
    # as form/H2H); a full overwrite is the last resort when NOTHING was
    # resolved, and is then logged audibly via ``identity_source`` so the
    # user-facing sources list shows where the identity came from.
    identity_source: str | None = None
    if (home_team is None or away_team is None) and oddspapi_teams is not None:
        od_home, od_away = oddspapi_teams
        home_full = bool(home_team and home_team.get("id") and home_team.get("name"))
        away_full = bool(away_team and away_team.get("id") and away_team.get("name"))
        if home_team is None or not home_full:
            home_team = _merge_team_fields(home_team, od_home)
        if away_team is None or not away_full:
            away_team = _merge_team_fields(away_team, od_away)
        if not home_full and not away_full:
            identity_source = "oddspapi_fallback_full"
        else:
            identity_source = "oddspapi_fallback_merged"
        logger.warning(
            "tim fallback dari oddspapi (%s) — data stats terbatas", identity_source
        )

    if home_team and away_team:
        meta_with_season["_team_names"] = {
            str(home_team["id"]): home_team["name"],
            str(away_team["id"]): away_team["name"],
        }

    home_id = (home_team or {}).get("id")
    away_id = (away_team or {}).get("id")
    if not home_id or not away_id:
        missing = []
        if not home_id:
            missing.append(f"home '{home_query}'")
        if not away_id:
            missing.append(f"away '{away_query}'")
        quota_notes = []
        if stats.fd.rate_limit_warning:
            quota_notes.append("football-data rate limit")
        quota_msg = ""
        if quota_notes:
            quota_msg = f" (provider issue: {', '.join(quota_notes)})"
        return {
            "error": f"tim tidak ditemukan: {', '.join(missing)}{quota_msg}",
            "league": display,
            "home_query": home_query,
            "away_query": away_query,
            "home_candidates": home_team,
            "away_candidates": away_team,
        }

    home_name = home_team["name"]
    away_name = away_team["name"]

    # G2: canonical entity identity per side, persisted on the snapshot.
    # ``home_team``/``away_team`` carry the resolving provider's id + (after
    # G1) a deterministic canonical_id; when the provider chain did not
    # attach one (oddspapi fallback, legacy paths) compute it on the fly so
    # every NEW snapshot has an ID-level identity for settle verification.
    def _entity(side: dict[str, Any] | None, fallback_name: str) -> dict[str, Any]:
        pid = (side or {}).get("id")
        return {
            "canonical_id": (side or {}).get("canonical_id")
            or _safe_entity_canonical(
                (side or {}).get("provider"), pid, league_key, (side or {}).get("name") or fallback_name
            ),
            "provider": (side or {}).get("provider"),
            "provider_id": str(pid) if pid is not None else None,
            "name": (side or {}).get("name") or fallback_name,
        }

    entities: dict[str, Any] = {
        "home": _entity(home_team, home_name),
        "away": _entity(away_team, away_name),
        # G2: the league key used to compute the canonical ids above, so the
        # settle verifier recomputes the result side with the SAME league
        # (a wrong-league result then fails the canonical comparison).
        "league_key": league_key,
    }

    # Identity firewall G-B (plan 2026-08-24): query pair vs RESOLVED teams.
    # ``home_team``/``away_team`` are the identities every downstream fetch
    # (form/H2H/xG/standings) keys on -- when either one canonicalizes to a
    # DIFFERENT club than the requested side, everything fetched below would
    # describe the wrong match. Refuse here (fail-closed), warn on a
    # confirmed side swap (some sources render pendent fixtures reversed).
    _identity_checks: list[dict[str, Any]] = []
    if _fw.get("enabled", True):
        try:
            from .identity_gate import check_pair_identity

            _fwB = check_pair_identity(
                home_query,
                away_query,
                league_key=league_key,
                detected_home=(source_match or {}).get("home"),
                detected_away=(source_match or {}).get("away"),
                resolved_home=home_name,
                resolved_away=away_name,
            )
            _identity_checks = [
                {"stage": "preflight_fixture"},
                *(_fwB.get("checks") or []),
            ]
            if _fwB.get("status") == "refuse" and _fw.get("refuse_divergence", True):
                logger.warning(
                    "identity_gate G-B refuse %s vs %s: %s",
                    home_query, away_query, _fwB.get("reasons"),
                )
                return refusal_payload(
                    "post_resolve",
                    _fwB.get("reasons") or [],
                    display=display,
                    home_query=home_query,
                    away_query=away_query,
                    detail={"checks": _fwB.get("checks")},
                )
        except Exception as exc:  # noqa: BLE001 -- firewall never blocks analyse
            logger.warning("identity_gate G-B failed (proceed): %s", exc)

    # NOTE: the canonical/legacy match-id block used to sit here, but it reads
    # ``kickoff`` -- which is only resolved ~90 lines below (it needs
    # ``fixture``, fetched at fetch_upcoming_fixture). Because ``kickoff`` is
    # assigned later in this same function, Python treats it as a local for the
    # whole body, so reading it here raised
    #   UnboundLocalError: cannot access local variable 'kickoff'
    # on every analysis. The block now lives immediately after the kickoff
    # cross-check; nothing between here and its first consumer (_row_by_ids at
    # the stability read) touches those ids.

    # P3: cross-source odds validation. First-wins resolution stays; a
    # disagreement between independent sources on key lines is surfaced as
    # ``odds_quality`` and feeds the signal engine's data-quality scoring.
    # Runs here (after team names resolve) because the comparison matches
    # sides by name.
    odds_quality: dict[str, Any] | None = None
    if len(_odds_payloads) >= 2:
        try:
            _tol = float(
                ((cfg.get("models") or {}).get("signal_engine") or {})
                .get("odds_disagreement_pp", 8.0)
            )
            odds_quality = cross_source_odds_check(
                _odds_payloads,
                home_name=home_name,
                away_name=away_name,
                home_query=home_query,
                away_query=away_query,
                tolerance_pp=_tol,
            )
            if odds_quality.get("status") == "cross_source_disagreement":
                logger.warning(
                    "cross-source odds disagreement %s vs %s: %r",
                    home_query, away_query, odds_quality,
                )
        except Exception as exc:
            logger.warning("cross-source odds check failed (prediction unaffected): %s", exc)
            odds_quality = None

    fixture = await stats.fetch_upcoming_fixture(home_id, away_id, meta_with_season)
    logger.info("analyse phases: fixture=%.1fs", _time.monotonic() - _t0)
    _t0 = _time.monotonic()
    home_form = await stats.fetch_team_form(home_id, meta_with_season)
    away_form = await stats.fetch_team_form(away_id, meta_with_season)
    # NowGoal analysis-page fallback for form: only when flashscore (and the
    # provider chain) came up empty AND the nowgoal client is in play. The
    # page server-renders both teams' last-5 with the same WDL/gf/ga shape
    # the model validates against, so a flashscore outage no longer blanks
    # the form features entirely.
    # F1: the nowgoal analysis page fallback fires not only when the form
    # chain returned NOTHING, but also when it returned a THIN window (< 3
    # matches per team) -- a 1-match form is noise, and before this fix it
    # silently short-circuited the fallback (football-data's 1-match window
    # "counted" as form, starving the statistical component).
    ng_analysis = None
    # 2026-08-17: pass the fixture date so find_fixture scans the SCHEDULE
    # DATE instead of today+tomorrow -- a match whose kickoff is >1 day out
    # (e.g. queried on 08-17 for an 08-19 fixture) was never found, silently
    # disabling the nowgoal form/H2H/xG tiers for upcoming matches.
    _ng_date = ((fixture or {}).get("date") or "")[:10] or None
    if _form_depth_thin(home_form, away_form) and nowgoal is not None:
        try:
            ng_analysis = await nowgoal.fetch_analysis(home_name, away_name, date=_ng_date)
        except Exception as exc:
            logger.warning("nowgoal analysis fallback failed (prediction unaffected): %s", exc)
        if isinstance(ng_analysis, dict) and ng_analysis:
            if ng_analysis.get("home_form") and _form_depth(ng_analysis["home_form"]) > _form_depth(home_form):
                home_form = ng_analysis["home_form"]
                logger.info("home form via nowgoal_analysis (form tipis -> diperkaya)")
            if ng_analysis.get("away_form") and _form_depth(ng_analysis["away_form"]) > _form_depth(away_form):
                away_form = ng_analysis["away_form"]
                logger.info("away form via nowgoal_analysis (form tipis -> diperkaya)")
    logger.info("analyse phases: form_x2=%.1fs", _time.monotonic() - _t0)
    _t0 = _time.monotonic()

    # Odds FIRST: has_odds/consensus/signal feed the decision engine, and the
    # oddspapi fallback (Conference League qualification etc.) only helps while
    # the budget is still healthy. Everything after this is optional context
    # (H2H, renders, xG history) and budget-guarded/cheap to skip.
    logger.info("analyse phases: odds=%.1fs", _time.monotonic() - _t0)

    # P1.1: collect EVERY independent kickoff candidate before picking the
    # first-wins value, so the winner can be cross-checked against the rest.
    # A single source's bad timezone/date (the Cadiz-B class) must not
    # silently declare the match finished (or pre-match) -- the sources must
    # agree within tolerance or the status is "cannot determine".
    _kickoff_candidates: dict[str, str] = {}
    if (fixture or {}).get("date"):
        _kickoff_candidates["fixture"] = fixture["date"]
    if match_odds_payload and match_odds_payload.get("commence_time"):
        _kickoff_candidates["odds"] = match_odds_payload["commence_time"]
    if source_match and source_match.get("kickoff"):
        _kickoff_candidates["source_match"] = source_match["kickoff"]
    kickoff = (
        _kickoff_candidates.get("fixture")
        or _kickoff_candidates.get("odds")
        or _kickoff_candidates.get("source_match")
        or None
    )
    # Cross-check the winner against independent sources; a multi-hour
    # disagreement marks the kickoff UNCERTAIN. match_finished is never
    # derived from an uncertain kickoff.
    kickoff_uncertain, kickoff_deltas = _kickoff_cross_check(kickoff, _kickoff_candidates)
    if kickoff_uncertain:
        logger.warning(
            "kickoff_uncertain %s vs %s: primary=%r candidates=%r deltas=%r",
            home_query, away_query, kickoff, _kickoff_candidates, kickoff_deltas,
        )

    # Fix 2026-08-22 (identity): ONE id per real-world match regardless of
    # provider spelling ("Al-Faisaly" vs "Al Faisaly" produced two parallel
    # match_ids with separate stability pins). New snapshots are WRITTEN
    # under the canonical id; every READ falls back to the legacy name-based
    # id so pre-fix snapshots stay addressable.
    #
    # Placement is load-bearing: this MUST come after ``kickoff`` is resolved
    # above (both ids are keyed on it) and it may not move earlier, because
    # ``kickoff`` depends on ``fixture`` which is fetched further down. Its
    # first consumer is the stability read much later, so this position is safe.
    from .prediction_log import canonical_match_id as _canon_mid
    from .prediction_log import make_match_id as _legacy_mid

    _mid_canon = _canon_mid(entities, league_key, home_name, away_name, kickoff)
    _mid_legacy = _legacy_mid(league_key, home_name, away_name, kickoff)

    # Identity firewall G-C (plan 2026-08-24): pre-flight history lock. The
    # same detection the end-of-run write hold uses -- but run BEFORE the
    # expensive phases, so a resolver flip against recent history ABORTS the
    # analysis (~2s in) instead of holding one snapshot write ~250s later.
    # Fail-open: a broken log never blocks an analysis.
    if _fw.get("enabled", True) and _fw.get("refuse_history_lock", True):
        try:
            from .identity_gate import DEFAULT_PREDICTIONS_FILE, preflight_history_lock, refusal_payload

            _pl_file = (
                (cfg.get("prediction_log") or {}).get("file")
                or DEFAULT_PREDICTIONS_FILE
            )
            _fw_lock = preflight_history_lock(
                ROOT / _pl_file,
                match_id=_mid_canon,
                home=home_name,
                away=away_name,
                entities=entities,
            )
            if _fw_lock and _fw_lock.get("locked"):
                return refusal_payload(
                    "history_lock",
                    [
                        str(_fw_lock.get("reason") or "resolver flip vs riwayat log"),
                        f"conflict_match_id={_fw_lock.get('conflict_match_id')}",
                    ],
                    display=display,
                    home_query=home_query,
                    away_query=away_query,
                    detail={"kind": _fw_lock.get("kind")},
                )
        except Exception as exc:  # noqa: BLE001 -- firewall never blocks analyse
            logger.warning("identity_gate G-C failed (proceed): %s", exc)

    def _row_by_ids(fn: Any) -> Any:
        """fn(match_id) -> row|None, tried on the canonical then legacy id."""
        return fn(_mid_canon) or fn(_mid_legacy)

    # Match already finished? kickoff < now -> there is no pre-match state to
    # predict from (and its own stats would leak into the model), so the
    # analysis shows the REAL result instead of a prediction.
    # P0-2: do NOT flip to finished purely on kickoff<now when a live source
    # (livescore "live", flashscore "live") still says the match is in play;
    # ``reconcile_status`` unifies both. A live window of [kickoff-15m,
    # kickoff+4h] is the conservative tolerance.
    match_finished = False
    # P0-2: the reconciled verdict (flashscore + livescore + kickoff window),
    # None when the kickoff is uncertain. Exposed on ``data_sources.match``
    # after the aggregation below so downstream consumers see ONE status
    # instead of per-source contradictions.
    _reconciled: str | None = None
    if not kickoff_uncertain:
        from .source_match import reconcile_status as _reconcile_status
        from .timeutil import utc_now_iso
        from datetime import datetime as _dt_cls, timezone as _tz
        # P0-2: the source-status pair comes from the validated source_match
        # context (the dict the match-source commands / league detection
        # produced) -- never from ``stats`` (the MultiSourceStatsFetcher
        # client, which has no match aggregation) and never from the
        # data_sources match field (its livescore secondary sample carries no
        # status). When only one source was queried, the kickoff-window rules
        # in reconcile_status still catch a stale "finished" verdict inside
        # the live window (rule 5 -> "live").
        _src = source_match or {}
        _flash_status = _src.get("status") if _src.get("source") == "flashscore" else None
        _live_status = _src.get("status") if _src.get("source") == "livescore" else None
        _kickoff_dt: datetime | None = None
        if isinstance(kickoff, str):
            try:
                _kickoff_dt = _dt_cls.fromisoformat(kickoff.replace("Z", "+00:00"))
                if _kickoff_dt.tzinfo is None:
                    _kickoff_dt = _kickoff_dt.replace(tzinfo=_tz.utc)
            except (ValueError, TypeError):
                _kickoff_dt = None
        elif isinstance(kickoff, datetime):
            _kickoff_dt = kickoff
        _now_iso = utc_now_iso()
        try:
            _now_dt = _dt_cls.fromisoformat(_now_iso.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            _now_dt = _dt_cls.now(tz=_tz.utc)
        _reconciled = _reconcile_status(
            _kickoff_dt, _now_dt,
            flashscore_status=_flash_status, livescore_status=_live_status,
        )
        if _reconciled == "finished":
            match_finished = True
        elif _reconciled == "live":
            # explicit live verdict: never treat as finished even if kickoff
            # is in the past (catches the flashscore-stale-finished case).
            match_finished = False
        else:
            _hours_to_kickoff = _kickoff_hours_ahead(kickoff)
            if _hours_to_kickoff is not None and _hours_to_kickoff < 0:
                match_finished = True
    # P1.1: an UNCERTAIN kickoff (sources disagree beyond tolerance) means we
    # cannot determine whether the match is pre-match, live or finished -- so
    # no prediction/snapshot is produced (a possibly-live match would leak its
    # own stats into the model), but the card must NOT claim "Match sudah
    # selesai" either. The format layer renders a distinct "kickoff tidak
    # dapat dipastikan" block instead.
    # Skip prediction for live matches — pre-match odds/model are stale
    # once the ball is rolling (the LIVE card shows current score instead).
    _skip_prediction = bool(match_finished or kickoff_uncertain)
    _t0 = _time.monotonic()

    # H2H (budget-guarded context; the decision already has odds in hand).
    if not _budget_short(analysis_remaining(), margin=12.0):
        h2h = await stats.fetch_h2h(home_id, away_id, meta_with_season)
    else:
        h2h = None
    # NowGoal analysis-page H2H fallback: reuse the page already fetched for
    # the form fallback when it exists, otherwise fetch it now (only when
    # flashscore could not supply H2H). The analysis page renders W/D/L from
    # the HOME side, matching fetch_h2h's contract, so it plugs in
    # transparently.
    if not h2h:
        if ng_analysis is None and nowgoal is not None and not _budget_short(
            analysis_remaining(), margin=12.0
        ):
            try:
                ng_analysis = await nowgoal.fetch_analysis(home_name, away_name, date=_ng_date)
            except Exception as exc:
                logger.warning("nowgoal analysis H2H fallback failed (prediction unaffected): %s", exc)
        if isinstance(ng_analysis, dict) and ng_analysis.get("h2h"):
            h2h = ng_analysis["h2h"]
            logger.info("h2h via nowgoal_analysis (flashscore kosong)")
    # P2-1: H2H window (<= 3 years). Direct meetings older than that are
    # stale evidence (squads / managers / divisions all change) -- they are
    # filtered out and the dict is flagged ``h2h_relevance: stale`` when
    # nothing survives, so the card can surface it instead of silently
    # letting a 7-year-old meeting drive the H2H feature.
    if h2h is not None:
        try:
            from .livescore import apply_h2h_window
            h2h = apply_h2h_window(h2h, home_name=home_name)
        except Exception as exc:  # noqa: BLE001 -- windowing is best-effort
            logger.warning("h2h window filter failed (prediction unaffected): %s", exc)
    logger.info("analyse phases: h2h=%.1fs", _time.monotonic() - _t0)
    _t0 = _time.monotonic()

    # NowGoal rich context (team stats / HT/FT / goal timing / lineups /
    # standings / fixtures / injuries): CONTEXT ONLY -- never a model input.
    # These pages have no historical record for training rows, so they feed
    # the decision/formatting layer exactly like the flashscore standings and
    # lineups context blocks. Budget-guarded and failure-isolated.
    nowgoal_context = None
    _t0 = _time.monotonic()
    # Overlap optimisation (data-identical): the NowGoal detail page is pure
    # HTTP -- start the fetch NOW so it downloads while the flashscore
    # BROWSER renders below are running, instead of before them. The budget
    # guard is evaluated HERE, at the original semantic position, and the
    # result is merely consumed after those renders; a failed guard never
    # starts the task, which matches the legacy skip exactly.
    _detail_task = None
    if nowgoal is not None and not _budget_short(analysis_remaining(), margin=15.0):
        _detail_task = asyncio.ensure_future(
            nowgoal.fetch_match_detail(home_name, away_name)
        )
    _t0 = _time.monotonic()

    # No data leakage: live/final event stats are only usable pre-match. If the
    # queried fixture has kicked off, its own stats would leak into the model.
    fixture_flash_url = fixture.get("flashscore_url") if fixture else None
    fixture_is_prematch = bool(
        fixture
        and fixture.get("status") == "notstarted"
    )
    # Overlap optimisation (data-identical): the GraphQL event context is
    # pure HTTP -- start it now so it downloads during the browser renders
    # below. Static preconditions are checked here; the budget/hours guards
    # stay at the original consumption position further down.
    _ctx_task = None
    if fixture_is_prematch and fixture_flash_url and not match_finished:
        _ctx_task = asyncio.ensure_future(
            stats.fetch_flashscore_event_context(
                fixture_flash_url, home_name, away_name
            )
        )
    match_stats = None
    lineups = None
    _t0 = _time.monotonic()
    if match_finished:
        # Post-match: the real result stats (xG/possession/shots) for the
        # finished-match report. Pre-match context (lineups, missing players,
        # match info) is skipped -- there is nothing left to predict.
        if fixture_flash_url and not _budget_short(analysis_remaining()):
            match_stats = await stats.fetch_flashscore_stats_for_match(fixture_flash_url)
    elif (
        fixture_is_prematch
        and fixture_flash_url
        and not _budget_short(analysis_remaining())
    ):
        match_stats = await stats.fetch_flashscore_stats_for_match(fixture_flash_url)
        # Predicted/confirmed lineups as CONTEXT INFO only (never a model
        # feature: flashscore predicted lineups have no historical record
        # to validate against). A failure here must not break the flow.
        # Skipped when the match is >72h away (no predicted XIs yet) or when
        # the analysis has already used most of the runner deadline -- the
        # render costs up to 25s and the output is context-only.
        try:
            _hours = _kickoff_hours_ahead((fixture or {}).get("date"))
            if (_hours is None or _hours <= 72.0) and not _budget_short(analysis_remaining()):
                lineups = await stats.fetch_flashscore_lineups_for_match(fixture_flash_url)
        except Exception as exc:
            logger.warning("flashscore lineups fetch failed (prediction unaffected): %s", exc)
        # LiveScore lineup fallback: when Flashscore has no lineups (e.g.
        # qualification rounds not covered), fetch from LiveScore's /lineups
        # endpoint which carries the full starting XI + subs.
        if not lineups and getattr(stats, "livescore", None) is not None and getattr(stats.livescore, "available", False):
            if not _budget_short(analysis_remaining()):
                try:
                    _ls_match = (meta_with_season or {}).get("_livescore_match") or {}
                    _ls_eid = _ls_match.get("source_id")
                    if _ls_eid:
                        from .livescore import parse_lineups
                        _ls_lineup = await stats.livescore.fetch_lineups(_ls_eid)
                        if _ls_lineup:
                            parsed = parse_lineups(_ls_lineup, _ls_match)
                            if parsed:
                                lineups = parsed
                                logger.info("lineups via LiveScore fallback")
                except Exception as exc:
                    logger.warning("livescore lineups fallback failed: %s", exc)
    logger.info("analyse phases: match_stats+lineups=%.1fs", _time.monotonic() - _t0)

    # Consume the overlapped NowGoal detail fetch (started before the browser
    # renders above; guards were evaluated at the original semantic position,
    # so reaching this point with a live task == the legacy "fetch allowed"
    # case). The merge order below is byte-for-byte the legacy one.
    if _detail_task is not None:
        try:
            detail = await _detail_task
            if isinstance(detail, dict) and detail:
                nowgoal_context = {
                    k: detail[k] for k in
                    ("team_stats", "htft", "goal_timing", "lineups")
                    if detail.get(k)
                }
        except Exception as exc:
            logger.warning("nowgoal detail context failed (prediction unaffected): %s", exc)
        # Standings / fixtures / injuries ride the SAME analysis page the
        # form+H2H fallback already fetched (no extra request).
        if isinstance(ng_analysis, dict):
            for k in ("standings", "fixtures", "injuries"):
                if ng_analysis.get(k):
                    nowgoal_context = nowgoal_context or {}
                    nowgoal_context[k] = ng_analysis[k]
            # P3-2: provenance -- the analysis-page form/H2H aggregates
            # dropped Club Friendlies / pre-season rows; surface that on the
            # context block so the card can say so instead of hiding it.
            _ng_excluded = (
                (ng_analysis.get("home_form") or {})
                .get("excluded_competitions")
            ) or (
                (ng_analysis.get("h2h") or {}).get("excluded_competitions")
            )
            if _ng_excluded:
                nowgoal_context = nowgoal_context or {}
                nowgoal_context["excluded_competitions"] = _ng_excluded
        logger.info("analyse phases: nowgoal_context=%.1fs", _time.monotonic() - _t0)

    # ---- Flashscore GraphQL context: missing players + coaches -------------
    # Pure-HTTP (no browser render). Missing players (injuries/suspensions,
    # incl. doubtful) and coach names are pre-match CONTEXT ONLY -- never
    # model features, per the same no-OOS-evidence rule as predicted lineups.
    # A failure here must never break the flow.
    missing_players = None
    coaches = None
    _t0 = _time.monotonic()
    if (
        fixture_is_prematch
        and fixture_flash_url
        and not match_finished
        and not _budget_short(analysis_remaining(), margin=12.0)
    ):
        try:
            _hours = _kickoff_hours_ahead((fixture or {}).get("date"))
            if _hours is None or _hours <= 72.0:
                # Overlapped fetch (started before the browser renders);
                # identical endpoint/arguments as the legacy inline call.
                _ctx = await _ctx_task
                if _ctx:
                    missing_players = {
                        side: {
                            "missing": (entry or {}).get("missing") or [],
                            "unsure": (entry or {}).get("unsure") or [],
                        }
                        for side, entry in _ctx.items()
                        if side in ("home", "away") and isinstance(entry, dict)
                    }
                    coaches = {
                        side: (entry or {}).get("coaches") or []
                        for side, entry in _ctx.items()
                        if side in ("home", "away") and isinstance(entry, dict)
                    }
            elif _ctx_task is not None:
                # >72h to kickoff: legacy code never fetched -- discard.
                _ctx_task.cancel()
        except Exception as exc:
            logger.warning("flashscore context fetch failed (prediction unaffected): %s", exc)
    elif _ctx_task is not None:
        # Budget guard failed at its original position -> discard the fetch,
        # matching the legacy skip outcome exactly.
        _ctx_task.cancel()

    # League standings (browser render, budget-guarded, context only).
    # ``standings_key`` is the user league, corrected to the fixture's real
    # competition when the resolved match showed a league mismatch (so the
    # table rendered belongs to the match, not to the query).
    standings = None
    if not match_finished and not _budget_short(analysis_remaining(), margin=12.0):
        try:
            standings = await stats.fetch_league_standings(standings_key)
            if standings:
                tbl = (standings.get("tables") or {}).get("overall") or []
                matched = {}
                for side, target in (("home", home_name), ("away", away_name)):
                    # Tiered matcher (exact -> alias -> prefix-drop ->
                    # guarded containment) so "Paris FC" never captures the
                    # "Paris Saint-Germain" row (2026-08-17).
                    row = _match_standings_team(tbl, target)
                    if row:
                        matched[side] = row
                if matched:
                    standings["teams"] = matched
        except Exception as exc:
            logger.warning("flashscore standings fetch failed (prediction unaffected): %s", exc)

    # Match info: venue/referee/capacity + neutral-location flag (context
    # only -- a neutral venue means the "home" side has NO home advantage,
    # which the Elo home-advantage term does not know about).
    match_info = None
    if (
        fixture_is_prematch
        and fixture_flash_url
        and not match_finished
        and not _budget_short(analysis_remaining(), margin=12.0)
    ):
        try:
            match_info = await stats.fetch_flashscore_match_info(fixture_flash_url)
        except Exception as exc:
            logger.warning("flashscore match info fetch failed (prediction unaffected): %s", exc)
    logger.info("analyse phases: context+standings+info=%.1fs", _time.monotonic() - _t0)

    home_history = None
    away_history = None
    _t0 = _time.monotonic()
    # xG history (Plan B, 2026-08-17): NowGoal PRIMARY (analysis-page
    # match_list -> canonical match_id -> live-{id} FT xG), flashscore
    # fallback. Understat/FBref are no longer consulted (big-5 only +
    # blocked/0% usage respectively). Same xg_weight applies -- the source
    # change does NOT lower the xG blend. The predicted fixture is excluded
    # by (home, away, date) so it can never leak. A failure leaves xG inert
    # -- prediction unaffected.
    try:
        kickoff_date = (fixture or {}).get("date") or ""
        if match_finished:
            logger.info("xG history skipped: match sudah selesai")
        elif _budget_short(analysis_remaining(), margin=15.0):
            logger.info("xG history skipped: analysis budget nearly spent")
        else:
            # 2026-08-17 (Plan B live verification): NowGoal is the PRIMARY
            # xG tier for every league, but its match_list only exists when
            # the analysis page was fetched (previously: thin form or missing
            # H2H). On the flashscore source_match path neither fired, so the
            # tier was silently skipped and the expensive browser fallback
            # ran instead. Ensure the page once here (cheap HTTP ~3-4s,
            # budget-guarded) so the NowGoal tier runs consistently and the
            # flashscore fallback only fires when NowGoal genuinely has no xG.
            if ng_analysis is None and nowgoal is not None and not _budget_short(
                analysis_remaining(), margin=12.0
            ):
                try:
                    ng_analysis = await nowgoal.fetch_analysis(home_name, away_name, date=kickoff_date[:10] or None)
                except Exception as exc:
                    logger.warning("nowgoal analysis fetch for xG failed (prediction unaffected): %s", exc)

            async def _xg_side(side: str, team: Any, fc_lane: Any) -> Any:
                # Per-side chain, identical body to the former sequential
                # loop. ``fc_lane`` routes the flashscore fallback tier to a
                # dedicated browser session (lane B) so home/away overlap;
                # None keeps the legacy primary-lane behaviour.
                if not team:
                    return None
                ml = None
                if isinstance(ng_analysis, dict):
                    fk = "home_form" if side == "home" else "away_form"
                    ml = (ng_analysis.get(fk) or {}).get("match_list")
                # Resolved flashscore ref (slug/id from resolve_match): makes
                # the fallback tier deterministic instead of a name-lottery.
                _fs_match = (meta_with_season or {}).get("_flashscore_match") or {}
                return await stats.fetch_team_xg_history(
                    home_name if side == "home" else away_name,
                    meta_with_season,
                    exclude=(home_name, away_name, kickoff_date[:10]),
                    nowgoal_client=nowgoal,
                    match_list=ml,
                    flashscore_client=fc_lane,
                    flashscore_ref=_fs_match.get(side),
                )

            # Home and away chains are independent (distinct teams, distinct
            # cache keys, read-only ng_analysis access) and now run in
            # parallel; wall time ~= the slower chain instead of their sum.
            _home_h, _away_h = await asyncio.gather(
                _xg_side("home", home_team, None),
                _xg_side("away", away_team, getattr(stats, "fc_secondary", None)),
            )
            home_history = _home_h
            away_history = _away_h
    except Exception as exc:
        logger.warning("xG history fallback failed (prediction unaffected): %s", exc)
    logger.info("analyse phases: history+xg=%.1fs", _time.monotonic() - _t0)

    bookmaker_odds_h2h: list[dict[str, Any]] = []
    market_totals: dict[str, dict[str, float]] = {}
    # Sharp-book benchmark (2026-08-23): when configured (models.decision.
    # primary_bookmaker), that bookmaker's quote IS the market benchmark for
    # 1X2 and wins totals/BTTS lines outright instead of median/min-margin.
    _primary_bm = (
        str(((cfg.get("models", {}) or {}).get("decision", {}) or {}).get("primary_bookmaker") or "").strip()
        or None
    )
    if match_odds_payload:
        bookmaker_odds_h2h = extract_h2h_entries(
            match_odds_payload, home_name, away_name,
            home_query=home_query, away_query=away_query,
        )
        # TODO-04: single shared per-bookmaker-pair totals extractor (the old
        # inline best-of-both-sides loop double-removed margin).
        market_totals = extract_market_totals(match_odds_payload, prefer_bookmaker=_primary_bm)
        # Re-prioritas 2026-08-24: when the PRIMARY payload is NowGoal it
        # carries no BTTS market -- fill BTTS Yes/No from the secondary
        # payload so BTTS candidates keep tradeable prices (G7 require_price
        # would veto every BTTS pick otherwise). No-op when oddspapi IS the
        # primary (same object) or the secondary has nothing.
        _odds_secondary = (
            _odds_payloads.get("oddspapi")
            or _odds_payloads.get("theoddsapi")
        )
        if _odds_secondary is not None and _odds_secondary is not match_odds_payload:
            try:
                _sec_totals = extract_market_totals(
                    _odds_secondary, prefer_bookmaker=_primary_bm
                )
                market_totals = _merge_missing_btts(market_totals, _sec_totals)
            except Exception as exc:
                logger.warning(
                    "btts merge dari sumber sekunder gagal (totals primary utuh): %s", exc
                )

    has_odds = bool(bookmaker_odds_h2h)
    consensus = (
        consensus_odds(bookmaker_odds_h2h, primary_bookmaker=_primary_bm)
        if bookmaker_odds_h2h else {"home": 0, "draw": 0, "away": 0}
    )

    def _margin_free_home(cons: dict[str, Any]) -> float | None:
        """Margin-free implied home probability from 1X2 consensus odds."""
        try:
            h, d, a = (float(cons[k]) for k in ("home", "draw", "away"))
        except (KeyError, TypeError, ValueError):
            return None
        if min(h, d, a) <= 1.0:
            return None
        inv = 1.0 / h + 1.0 / d + 1.0 / a
        return (1.0 / h) / inv
    outlier = find_outlier(bookmaker_odds_h2h, consensus, cfg["outlier_threshold_pct"]) if bookmaker_odds_h2h else None
    best = best_odds(bookmaker_odds_h2h) if bookmaker_odds_h2h else {}
    signal = score_signal(
        bookmaker_odds_h2h,
        consensus,
        outlier,
        home_form.get("sequence") if home_form else None,
        away_form.get("sequence") if away_form else None,
        has_odds,
    )

    xg_lambda = None
    if match_stats:
        xg_h = match_stats.get("xg_home")
        xg_a = match_stats.get("xg_away")
        if isinstance(xg_h, (int, float)) and isinstance(xg_a, (int, float)):
            xg_lambda = (float(xg_h), float(xg_a))

    if xg_lambda is None and home_history and away_history:
        h_for = home_history.get("xg_for_avg")
        a_for = away_history.get("xg_for_avg")
        h_against = home_history.get("xg_against_avg")
        a_against = away_history.get("xg_against_avg")
        if (
            isinstance(h_for, (int, float))
            and isinstance(a_for, (int, float))
            and isinstance(h_against, (int, float))
            and isinstance(a_against, (int, float))
        ):
            home_xg = (float(h_for) + float(a_against)) / 2.0
            away_xg = (float(a_for) + float(h_against)) / 2.0
            xg_lambda = (home_xg, away_xg)

    picks_payload = {"top_picks": [], "best_pick": None, "model_probs": {}}
    if has_odds and consensus.get("home", 0) > 0 and not _skip_prediction:
        picks_payload = derive_picks(consensus, market_totals, signal, xg_lambda=xg_lambda)

    # Market movement (opening -> current) + value edges (model vs margin-free
    # implied) are pure diagnostics on data we already have; they never gate
    # the engine, but expose steam/drift and value candidates in the report.
    # Both stay None when the provider exposes no opening prices.
    movement = market_movement(bookmaker_odds_h2h) if bookmaker_odds_h2h else None
    value = None
    if picks_payload.get("model_probs") and consensus.get("home", 0) > 0:
        value = value_edges(picks_payload["model_probs"], consensus)
    # Historical fair-line comparison (odds_history): when a fixtures cache
    # exists for this league, the live consensus is benchmarked against the
    # league's average CLOSING implied probability, exposing odds that moved
    # away from the historical fair line (value drift). Best-effort; a
    # missing cache leaves it None (never fabricated).
    value_history = None
    if consensus.get("home", 0) > 0 and not _budget_short(analysis_remaining(), margin=15.0):
        try:
            from .odds_history import load_league_baseline, live_value_signal

            baseline = await asyncio.to_thread(load_league_baseline, display, "cache/football")
            if baseline:
                value_history = live_value_signal(consensus, baseline)
        except Exception as exc:
            logger.warning("odds_history value baseline failed (prediction unaffected): %s", exc)

    sources = sorted({
        (home_team.get("provider") if home_team else None),
        (away_team.get("provider") if away_team else None),
        (home_form.get("source") if home_form else None),
        (away_form.get("source") if away_form else None),
        (h2h.get("source") if h2h else None),
        ((fixture or {}).get("source")),
        ("flashscore_xg" if (match_stats and "xg_home" in match_stats and (match_stats.get("source") or "") == "flashscore") else None),
        ("nowgoal_xg" if ((home_history or {}).get("source") == "nowgoal_xg" or (away_history or {}).get("source") == "nowgoal_xg") else None),
        ("flashscore_xg" if ((home_history or {}).get("source") == "flashscore_xg" or (away_history or {}).get("source") == "flashscore_xg") else None),
        ("flashscore_history" if (home_form or away_form) and ((home_form or {}).get("source") == "flashscore" or (away_form or {}).get("source") == "flashscore") else None),
        ("flashscore_lineups" if lineups and lineups.get("home_count") else None),
        ("flashscore_missing_players" if missing_players and any((v or {}).get("missing") for v in missing_players.values()) else None),
        ("flashscore_standings" if standings and (standings.get("tables") or {}).get("overall") else None),
        ("flashscore_match_info" if match_info else None),
        ("oddspapi_odds" if oddspapi_source else None),
        ("nowgoal_odds" if nowgoal_source else None),
        # P1.3: identity provenance when oddspapi filled the team resolution
        # (full overwrite vs merged) -- auditable in the user-facing sources.
        (identity_source if identity_source else None),
    } - {None})

    # ---- Multi-source aggregation layer (additive, feature-gated) ----
    # Wraps the already-collected primary (flashscore) data with provenance +
    # confidence and, when a secondary source is enabled, fills genuinely
    # missing fields via field-level fallback. Read-only w.r.t. the prediction
    # inputs: the engine still sees the SAME values, plus optional metadata.
    # A failure here never affects the prediction (best-effort, section 17).
    unified_dict: dict[str, Any] | None = None
    coverage_dict: dict[str, Any] | None = None
    try:
        from .datasources import aggregate_collected, coverage_report

        # The canonical identity's competition must be the competition the
        # fixture ACTUALLY belongs to, not the league the user typed. When
        # flashscore resolved the pair via a competition-aware scrape (e.g. a
        # league-page miss falling back to the homepage, verified 2026-08-16:
        # "laliga" query -> Las Palmas-Albacete actually in LaLiga2), the
        # resolved match carries the real competition section; fall back to
        # the display label only when it is absent.
        _fs_resolved = (meta_with_season or {}).get("_flashscore_match") or {}
        _real_competition = (
            _fs_resolved.get("competition") if isinstance(_fs_resolved, dict) else None
        ) or display
        unified = await aggregate_collected(
            primary_name="flashscore",
            primary_fields=_primary_fields(
                home=home_name, away=away_name, kickoff=kickoff, competition=_real_competition,
                home_form=home_form, away_form=away_form,
                h2h=h2h, lineups=lineups, missing_players=missing_players,
                standings=standings, match_stats=match_stats,
            ),
            secondary=_build_secondary_source(cfg, cache=cache),
            ref={"home": home_name, "away": away_name, "kickoff": kickoff,
                 "match_status": (source_match or {}).get("status") if (source_match or {}).get("source") == "flashscore" else None},
            config=_data_sources_config(cfg),
        )
        unified_dict = unified.to_dict()
        coverage_dict = coverage_report(unified)
    except Exception as exc:  # noqa: BLE001
        logger.warning("multi-source aggregation failed (prediction unaffected): %s", exc)

    # P0-2: pin the reconciled verdict on ``data_sources.match.value`` so the
    # card / audit trail read ONE status (never flashscore "finished" next to
    # a livescore "live" on the same fixture).
    if _reconciled is not None and isinstance(unified_dict, dict):
        _mv = (unified_dict.get("match") or {}).get("value")
        if isinstance(_mv, dict):
            _mv["reconciled_status"] = _reconciled

    # P1-2: source-confidence gate. When 3+ critical fields are LOW we
    # still run the engine (audit trail) but the resulting best_pick will
    # be vetoed and surfaced as NO BET with an explicit reason.
    _gate_passed = True
    _gate_reason: str | None = None
    if isinstance(unified_dict, dict):
        # ``unified_dict["confidence"]`` is the per-field map
        # ``{match: HIGH|MEDIUM|LOW, form: ..., h2h: ..., ...}`` (see
        # ``datasources.to_dict``). Local import: the module-level import
        # below sits in a LATER try block of the same function scope, which
        # would otherwise make ``evidence_gate`` an unbound local here
        # (UnboundLocalError) -- alias it so the two never collide.
        from .signal_engine import evidence_gate as _evidence_gate
        _gate_passed, _gate_reason = _evidence_gate(unified_dict.get("confidence"))
        if not _gate_passed:
            logger.warning("evidence_gate veto: %s", _gate_reason)

    # ---- Prediction engine (Elo + feature Poisson + ensemble + calibration) ----
    # Skipped entirely for finished matches: there is no pre-match state to
    # predict from, so only the real result is shown (see format.py).
    prediction = None
    elo = None  # set when the engine runs; used for feature snapshot below
    hard_cap_medium = False  # Section 5: H2H & xG both absent -> max MEDIUM
    form_depth_shallow = False  # P1: form window < 3 matches/tim -> max MEDIUM
    if (home_form or away_form or home_history or away_history) and not _skip_prediction:
        from .calibration import Calibrator, SignalScorer, league_calibrator
        from .context import build_match_context
        from .elo import EloModel
        from .models import Ensemble, PoissonModel, run_prediction_engine

        ctx = build_match_context(
            league=display,
            home=home_name,
            away=away_name,
            kickoff=kickoff,
            stats={
                "home_form": (home_form or {}).get("sequence"),
                "away_form": (away_form or {}).get("sequence"),
                "home_gf_avg": (home_form or {}).get("gf_avg"),
                "home_ga_avg": (home_form or {}).get("ga_avg"),
                "away_gf_avg": (away_form or {}).get("gf_avg"),
                "away_ga_avg": (away_form or {}).get("ga_avg"),
                # Raw scorelines (oldest->newest) enable time-decay weighting
                # so LIVE predictions use the same features as backtest/validate.
                "home_recent_goals": (home_form or {}).get("recent_goals"),
                "away_recent_goals": (away_form or {}).get("recent_goals"),
                "home_xg_for": (home_history or {}).get("xg_for_avg"),
                "home_xg_against": (home_history or {}).get("xg_against_avg"),
                "away_xg_for": (away_history or {}).get("xg_for_avg"),
                "away_xg_against": (away_history or {}).get("xg_against_avg"),
                "h2h": (h2h or {}),
                # Phase 1: pre-match lineup evidence as flag-gated lambda
                # CORRECTION factors (PoissonModel.lineup_weight /
                # rest_days_weight, both off by default). ``lineup_ts`` is the
                # observation time and feeds the Phase 1.3 leakage guard
                # (lineup fetched at/after kickoff is rejected as an input).
                "lineup": (
                    {"home": list(lineups.get("home") or []), "away": list(lineups.get("away") or [])}
                    if lineups and lineups.get("home_count") else None
                ),
                "lineup_status": (
                    lineups.get("status")
                    if lineups and lineups.get("status") in ("confirmed", "predicted")
                    else None
                ),
                "lineup_ts": utc_now_iso() if (lineups and lineups.get("home_count")) else None,
                "lineup_source": (lineups or {}).get("source"),
                "missing_home": list((missing_players or {}).get("home", {}).get("missing") or []),
                "missing_away": list((missing_players or {}).get("away", {}).get("missing") or []),
            },
            odds={
                "has_odds": has_odds,
                "consensus": consensus,
                "totals": market_totals,
            },
            sources=sorted(sources),
            source_meta=(unified_dict or {}).get("source_metadata"),
        )
        elo_cfg = cfg.get("models", {}).get("elo", {})
        poisson_cfg = cfg.get("models", {}).get("poisson", {})
        ens_cfg = cfg.get("models", {}).get("ensemble", {})
        elo = EloModel(
            k=elo_cfg.get("k", 32.0),
            home_advantage=elo_cfg.get("home_advantage", 65.0),
            initial_rating=elo_cfg.get("initial_rating", 1500.0),
            base_total_goals=elo_cfg.get("base_total_goals", 2.7),
            path=ROOT / elo_cfg.get("file", "cache/football/elo.json"),
        )
        poisson = PoissonModel(
            base_home_goals=poisson_cfg.get("base_home_goals", 1.45),
            base_away_goals=poisson_cfg.get("base_away_goals", 1.25),
            dc_rho=poisson_cfg.get("dc_rho", -0.1),
            shrinkage_samples=poisson_cfg.get("shrinkage_samples", 5),
            time_decay_xi=poisson_cfg.get("time_decay_xi", 0.9),
            xg_weight=poisson_cfg.get("xg_weight", 0.65),
            min_samples=poisson_cfg.get("min_samples", 2),
            # Phase 1: lineup/injury + rest-day lambda corrections. Both
            # default to 0 (off) until the Phase 1 backtest DoD passes.
            lineup_weight=float(poisson_cfg.get("lineup_weight", 0.0)),
            rest_days_weight=float(poisson_cfg.get("rest_days_weight", 0.0)),
            # Plan v3 (2026-08-24): F1 Elo-anchor + F2 market-total calibration.
            elo_anchor=poisson_cfg.get("elo_anchor"),
            market_total_calibration=poisson_cfg.get("market_total_calibration"),
        )
        ensemble = Ensemble(
            elo_weight=ens_cfg.get("elo_weight", 0.5),
            poisson_weight=ens_cfg.get("poisson_weight", 0.5),
        )
        # Phase 5: per-league calibration. EPL uses the global (EPL-fitted)
        # file; any other league must have its own fit or the calibrator is a
        # no-op (quality 0) and the decision layer falls back to MARKET PRIOR.
        calibrator = league_calibrator(league_key, cfg, ROOT) or Calibrator(min_samples=10**9)
        ss_cfg = cfg.get("models", {}).get("signal_scorer", {})
        scorer = SignalScorer(conf_weights=ss_cfg.get("conf_weights"))
        # Fix 2 (lambda source pinning): read the pinned lambda_source from
        # the most recent prediction snapshot for this match (canonical
        # match_id). The FIRST evaluation of a match pins its estimator;
        # later pre-match queries reuse it so the model cannot flip between
        # "elo" and "features" on threshold noise. See
        # models.run_prediction_engine for the one-time exception rule.
        _pin_src: str | None = None
        _pin_features_at_pin: bool | None = None
        try:
            _pl_cfg0 = cfg.get("prediction_log") or {}
            if _pl_cfg0.get("enabled") and _pl_cfg0.get("file"):
                from .prediction_log import last_prediction_snapshot
                _prev_row = _row_by_ids(
                    lambda mid: last_prediction_snapshot(ROOT / _pl_cfg0["file"], mid)
                )
                _prev_feat = (_prev_row or {}).get("features") or {}
                _pin_src = _prev_feat.get("pinned_lambda_source")
                if _pin_src:
                    _pin_features_at_pin = _prev_feat.get("pinned_features_available_at_pin")
        except Exception as exc:
            logger.warning("lambda pin lookup failed (prediction unaffected): %s", exc)
        # P1.4: production default is the PINNED lambda mode (Fix 2); the
        # ``lambda_mode`` config key exists for backtest/comparison only
        # ("threshold", "blend").
        _lam_mode = str(poisson_cfg.get("lambda_mode", "pinned"))
        prediction = run_prediction_engine(
            ctx,
            elo=elo,
            poisson=poisson,
            ensemble=ensemble,
            calibrator=calibrator,
            scorer=scorer,
            pinned_lambda_source=_pin_src,
            pinned_features_available_at_pin=_pin_features_at_pin,
            lambda_mode=_lam_mode,
        )
        if prediction is not None:
            prediction = prediction.to_dict()
        # Section 5 hard cap: H2H AND xG both completely absent are the two
        # least Elo-substitutable inputs -> confidence_tier max MEDIUM.
        hard_cap_medium = bool(
            not ctx.has_xg and not (ctx.h2h and any(ctx.h2h.values()))
        )
        # P1 (form-depth floor): a form window shorter than 3 matches per team
        # is noise, not signal -> confidence max MEDIUM, STRONG banned.
        from .model_gates import form_depth_shallow as _form_depth_shallow
        form_depth_shallow = _form_depth_shallow(home_form, away_form)
        # P1-2 propagation (2026-08-22): an evidence-gate veto must reach the
        # quality numbers every downstream consumer displays.
        _apply_source_confidence_gate(prediction, passed=_gate_passed, reason=_gate_reason)

    # ---- Recommendation grading (VALID / CANDIDATE / HATI-HATI) ---------
    # A pick is only advertised as a valid bet when the model behind it is
    # reliable: confidence HIGH, calibration validated, data complete, real
    # edge, strong signal. Everything else is downgraded with reasons. When
    # the prediction engine did not run (no form/history at all), picks are
    # still graded but can never reach VALID/CANDIDATE (all gates report
    # "tidak dihitung").
    if picks_payload.get("top_picks"):
        from .predictor import grade_recommendation

        conf = prediction.get("confidence") if prediction else None
        calib = (prediction.get("calibration") or {}).get("quality") if prediction else None
        compl = prediction.get("data_completeness") if prediction else None
        sig = prediction.get("signal_strength") if prediction else None
        for p in picks_payload["top_picks"]:
            p["grade"] = grade_recommendation(
                confidence=conf,
                calibration_quality=calib,
                data_completeness=compl,
                edge_pct=p.get("edge"),
                signal=sig or 0,
            )

    # ---- PHASE 4/8: historical performance of SIMILAR signals --------------
    # Never breaks the flow; shows what the same confidence/edge bucket has
    # actually done historically (real settled snapshots only).
    similar_signal: dict[str, Any] | None = None
    try:
        pl_cfg = cfg.get("prediction_log") or {}
        if pl_cfg.get("enabled") and pl_cfg.get("file"):
            from .prediction_log import similar_signal_stats

            best = (picks_payload or {}).get("best_pick")
            if best is not None:
                # Addendum v1.1 Section 2: this n>=5 bucket is the SIMILAR-
                # SIGNAL bucket (clusters historical SETTLED snapshots),
                # not the pick's calibration bucket (n>=30). Its
                # "belum cukup sampel (n<5)" LINE is removed from the
                # user-facing output; the internal threshold stays (it only
                # gates internal historical analysis, never a confidence tier).
                #
                # Model A rule: the market mirror (best_pick) carries edge 0.0
                # by construction now -- cluster on the INDEPENDENT engine's
                # edge instead, matching exactly how settled records store it
                # (_settled_records: prob_1x2 vs margin-free implied, per the
                # model's top 1X2 side).
                _me = (prediction or {}).get("market_edge") or {}
                _p1 = ((prediction or {}).get("model_probs") or {}).get("1x2") or {}
                _side = max(_p1, key=_p1.get) if _p1 else None
                _edge_pct = _me.get(_side) if _side and isinstance(_me, dict) else None
                similar_signal = similar_signal_stats(
                    ROOT / pl_cfg["file"],
                    confidence=(prediction or {}).get("confidence"),
                    edge_pct=_edge_pct,
                    min_bucket_n=5,
                )
    except Exception as exc:
        logger.warning("similar-signal lookup failed (prediction unaffected): %s", exc)

    # ---- Phase 1 decision engine (S23-S31) ------------------------------
    # Transparent Decision Score over the INDEPENDENT engine's calibrated
    # probabilities vs the market. Never forced: NO CLEAR DECISION / NO BET
    # are valid outputs. Never breaks the flow. Single shared entrypoint
    # (run_decision_engine) so `analisa` and `!best` use identical logic
    # including the correction-spec gates (Model A/B disagreement, EV/edge/
    # n_bucket/completeness gates, multiplicative confidence).
    decision: dict[str, Any] | None = None
    signal_engine_result: dict[str, Any] | None = None
    if not _skip_prediction:
        ml_probs = _ml_probs_for(cfg, league_key, home_name, away_name, kickoff)
        from .calibration import league_calibrator as _league_calibrator
        _league_calibrated = _league_calibrator(league_key, cfg, ROOT) is not None
        # Plan B movement signal: load the accumulated hourly odds snapshots
        # for this match and derive drift/steam/agreement vs the model side.
        _mv_signal: dict[str, Any] | None = None
        _snaps: list[dict[str, Any]] = []
        try:
            pl_cfg2 = cfg.get("prediction_log") or {}
            if pl_cfg2.get("enabled") and pl_cfg2.get("file"):
                from .movement import movement_signal
                from .prediction_log import list_odds_snapshots

                # Merge snapshots under BOTH ids (canonical + legacy) so a
                # match queried pre-fix keeps its full movement history.
                _snap_path = ROOT / pl_cfg2["file"]
                _seen_ts: set[tuple] = set()
                _snaps: list[dict[str, Any]] = []
                for _mid_i in (_mid_canon, _mid_legacy):
                    for _r in list_odds_snapshots(_snap_path, _mid_i):
                        _key = (
                            _r.get("ts"), _r.get("timing"),
                            repr(_r.get("odds_1x2") or {}),
                        )
                        if _key in _seen_ts:
                            continue
                        _seen_ts.add(_key)
                        _snaps.append(_r)
                # One-shot full movement history (P2, 2026-08-24 -- PRIMARY,
                # no longer gated on an empty poll series): NowGoal's trend
                # endpoint (type=14&t=20) returns EVERY recorded odds change
                # per bookmaker with a timestamp in ONE call. The background
                # odds-poll only accumulates snapshots for matches already
                # analysed and samples every 5-30 minutes, so the trend series
                # is strictly richer -- it is merged into (not replaced by)
                # the poll series whenever the budget is healthy. Each row now
                # carries its source bookmaker for downstream sharp-money
                # analysis. Cached ~30 min per match so repeated queries of
                # the same fixture do not re-hit the mirrors. Best-effort: a
                # failure or no-data degrades to the poll series unchanged.
                _trend_rows: list[dict[str, Any]] = []
                if (
                    nowgoal is not None
                    and not _budget_short(analysis_remaining(), margin=45.0)
                ):
                    try:
                        from .nowgoal import trend_to_snapshots

                        _trend_key = (
                            "ng_trend_"
                            + (_mid_canon or f"{home_name}_{away_name}")
                        ).replace(" ", "_")
                        _trend_cached = (
                            cache.get(_trend_key, ttl_seconds=1800)
                            if cache is not None
                            else None
                        )
                        if isinstance(_trend_cached, list):
                            _trend_rows = _trend_cached
                        else:
                            _ng_fx = await nowgoal.find_fixture(home_name, away_name)
                            if _ng_fx:
                                _trend = await nowgoal.fetch_odds_trend(_ng_fx)
                                if _trend:
                                    _trend_rows = trend_to_snapshots(_trend, kickoff=kickoff)
                                    if _trend_rows and cache is not None:
                                        cache.set(_trend_key, _trend_rows)
                    except Exception as exc:
                        logger.warning(
                            "nowgoal trend fetch failed (prediction unaffected): %s", exc
                        )
                if _trend_rows:
                    _snaps = sorted(
                        _snaps + _trend_rows,
                        key=lambda r: r.get("ts") or "",
                    )
                if _snaps:
                    _b1x2 = ((prediction or {}).get("model_probs") or {}).get("1x2") or {}
                    _model_side = max(_b1x2, key=_b1x2.get) if _b1x2 else None
                    _mcfg = (cfg.get("models") or {}).get("movement") or {}
                    _tau = _mcfg.get("time_decay_tau")
                    _mv_signal = movement_signal(
                        _snaps,
                        model_side=_model_side,
                        min_snapshots=int(_mcfg.get("min_snapshots", 3)),
                        steam_threshold_pct=float(_mcfg.get("steam_threshold_pct", 2.0)),
                        time_decay_tau=float(_tau) if _tau else None,
                        # Stale-guard: in-play captures (ts >= kickoff) must
                        # never be the drift's last point.
                        kickoff=kickoff,
                    )
        except Exception as exc:
            logger.warning("movement signal failed (decision unaffected): %s", exc)
        # Phase 0.2: odds observation timestamp for the edge benchmark (aliased
        # import -- this function binds ``utc_now_iso`` locally later).
        from .timeutil import utc_now_iso as _benchmark_now
        try:
            decision = run_decision_engine(
                prediction,
                consensus,
                market_totals,
                has_odds,
                len(bookmaker_odds_h2h),
                cfg,
                similar_signal=similar_signal,
                hard_cap_medium=hard_cap_medium,
                form_depth_shallow=form_depth_shallow,
                ml_probs=ml_probs,
                league=display,
                league_calibrated=_league_calibrated,
                movement=_mv_signal,
                # Phase 0.2: odds observation timestamp (aliased import --
                # this function binds ``utc_now_iso`` locally later).
                benchmark_ts=_benchmark_now(),
                benchmark_max_age_hours=float(
                    ((cfg.get("models") or {}).get("decision") or {}).get(
                        "benchmark_max_age_hours", 24.0
                    )
                ),
            )
        except Exception as exc:
            logger.warning("decision engine failed (prediction unaffected): %s", exc)

        # ---- Market-aware Signal Engine + Best Pick Ranker (additive) ----
        # Consumes the EXISTING engine output (model_probs) + the odds the
        # payload already fetched (incl. NowGoal Asian Handicap + opening
        # prices). Pure/deterministic; never breaks the flow.
        # Layer 1 + Layer 3: the engine reads the IMMUTABLE canonical opening
        # snapshot and the prior logged best pick, so repeated queries of the
        # same match cannot silently flip the recommendation on noise.
        try:
            from .prediction_log import (
                last_prediction_snapshot,
                opening_snapshot,
                stability_calibration,
            )
            from .signal_engine import (
                apply_pick_stability,
                evidence_gate,
                extract_asian_handicap,
                run_signal_engine,
            )
            from .timeutil import utc_now_iso as _utc_now_iso

            _pl = cfg.get("prediction_log") or {}
            _pl_enabled = bool(_pl.get("enabled") and _pl.get("file"))
            _opening: dict[str, Any] = {}
            if _pl_enabled:
                _os_ou = _row_by_ids(
                    lambda mid: opening_snapshot(ROOT / _pl["file"], mid, "ou")
                )
                _os_ah = _row_by_ids(
                    lambda mid: opening_snapshot(ROOT / _pl["file"], mid, "ah")
                )
                if _os_ou:
                    _opening["odds_ou"] = _os_ou["odds_ou"]
                if _os_ah:
                    _opening["odds_ah"] = _os_ah["odds_ah"]

            _ah_rows = extract_asian_handicap(match_odds_payload or {})
            _ah_supplemented = False
            # 2026-08-22: Oddspapi (primary) carries no Asian-handicap market,
            # so when it wins the odds race every Home/Away ±x line died with
            # "tidak ada harga" (Brentford/Inter-class cards). Supplement from
            # the NowGoal comparison payload P3 already collected -- no extra
            # network call; opening prices ride along so movement keeps its
            # anchor.
            if not _ah_rows and match_odds_payload is not None and oddspapi_source:
                _ng_ah = extract_asian_handicap(_odds_payloads.get("nowgoal") or {})
                if _ng_ah:
                    _ah_rows = _ng_ah
                    _ah_supplemented = True
                    logger.info("AH via nowgoal supplement (oddspapi has no handicap market)")
            # Team context for the decision layer (Group E, weight 0 by
            # default -- opt-in): flashscore missing players when available,
            # nowgoal injury lists as fallback, plus nowgoal team
            # stats / HT/FT / goal-timing attached for downstream use. Never
            # a model feature.
            # DEEP-ish copy: the merge below OVERWRITES ``missing`` per side;
            # a shallow dict() would share the inner dicts and write the
            # merged lists through to ``missing_players`` -- corrupting the
            # PHASE 7 snapshot (verified 2026-08-23 Club Brugge v Cercle:
            # snapshot logged 48 merged names instead of flashscore's raw 5).
            team_context: dict[str, Any] = {
                side: dict(entry or {})
                for side, entry in (missing_players or {}).items()
            }
            # Cross-provider merge (2026-08-16): flashscore and nowgoal name
            # the same injured players differently ("Garcia K." vs "Kike
            # Garcia" vs "Enrique Garcia Martinez, Kike") -- merge them into
            # ONE deduplicated per-side list so the engine never double-
            # counts a player and the display never looks contradictory.
            try:
                from .player_identity import merge_missing_lists
                _merged_inj = merge_missing_lists(
                    missing_players,
                    (nowgoal_context or {}).get("injuries") or {},
                ) or {}
            except Exception:  # noqa: BLE001 -- context, never blocks
                _merged_inj = {}
            for _side in ("home", "away"):
                _entries = _merged_inj.get(_side) or []
                if not _entries:
                    continue
                team_context.setdefault(_side, {})["missing"] = [
                    e["name"] for e in _entries
                ]
            for _k in ("team_stats", "htft", "goal_timing"):
                if (nowgoal_context or {}).get(_k):
                    team_context[_k] = nowgoal_context[_k]

            signal_engine_result = run_signal_engine(
                    model_probs=(prediction or {}).get("model_probs") or {},
                    stats={
                        "home_recent_goals": (home_form or {}).get("recent_goals"),
                        "away_recent_goals": (away_form or {}).get("recent_goals"),
                    },
                    market_totals=market_totals,
                    # Plan v3 F14: 1X2 jadi kandidat BEST PICK penuh (Home/
                    # Draw/Away Win) -- sebelumnya signal engine tidak punya
                    # kandidat 1X2 sama sekali sehingga "Away Win" mustahil
                    # keluar. Odds konsensus dipakai untuk implied margin-free.
                    odds_1x2={k: consensus.get(k) for k in ("home", "draw", "away")},
                    ah_rows=_ah_rows,
                    movement_snapshot=_mv_signal,
                    context=team_context or None,
                    completeness=(prediction or {}).get("data_completeness", 0.0),
                    cfg=(cfg.get("models") or {}).get("signal_engine"),
                    # 2026-08-22: the G4 lambda band is league-aware -- the
                    # engine needs the league to resolve per-league overrides
                    # (Eredivisie-class high-scoring leagues).
                    league_name=display,
                    prediction_timestamp=_utc_now_iso(),
                    history_snapshots=_snaps,
                    opening_snapshot=_opening or None,
                    # P3: cross-source odds disagreement surfaces here.
                    odds_quality=odds_quality,
                    # F2: the signal engine may VETO a pick built on a prior
                    # Elo λ (teams unseeded) + thin form when there is no H2H
                    # context either (ADO-Den-Haag-class incident).
                    has_h2h=bool(h2h and any((h2h or {}).values())),
                    # F3: reconcile with the independent 1X2 decision layer --
                    # a NO BET / NO CLEAR DECISION there caps the market-aware
                    # pick and surfaces the disagreement on the card.
                    model_decision_type=(decision or {}).get("decision_type"),
                    # Phase 5.4: pass league calibration status to cap scores
                    # for uncalibrated leagues (prevents misleading high confidence).
                    league_calibrated=_league_calibrated,
                    # Loser-guard (2026-08-22): deviation of the model's 1X2
                    # home probability from the margin-free market implied --
                    # feeds the disagreement_gate (Total/BTTS veto on the
                    # Real Betis-class contradiction).
                    x2_market_dev_pp=(
                        (
                            (((prediction or {}).get("model_probs") or {}).get("1x2") or {}).get("home", 0.0)
                            - _mkt_home
                        ) * 100.0
                        if (_mkt_home := _margin_free_home(consensus)) is not None else None
                    ),
                )
            # Layer 3: read the prior logged pick for this match and apply
            # the stability guard (calibrated score threshold).
            _prev_pick: dict[str, Any] | None = None
            if _pl_enabled:
                _prev_row = _row_by_ids(
                    lambda mid: last_prediction_snapshot(ROOT / _pl["file"], mid)
                )
                _prev_pick = (_prev_row or {}).get("signal_engine_pick")
            if _prev_pick and signal_engine_result is not None:
                _scfg = (cfg.get("models") or {}).get("signal_engine", {}).get("stability") or {}
                _cal = stability_calibration(
                    ROOT / _pl["file"],
                    percentile=float(_scfg.get("score_threshold_percentile", 0.95)),
                    min_samples=int(_scfg.get("score_threshold_min_samples", 20)),
                    fallback=float(_scfg.get("score_threshold_fallback", 0.05)),
                )
                signal_engine_result = apply_pick_stability(
                    signal_engine_result,
                    previous_pick=_prev_pick,
                    current_model=(prediction or {}).get("model_probs") or {},
                    opening_snapshot=_opening or None,
                    market_totals=market_totals,
                    now_ts=_utc_now_iso(),
                    cfg=(cfg.get("models") or {}).get("signal_engine"),
                    score_threshold=_cal["threshold"],
                )
            # Phase 5.1 (honest presentation): "BEST PICK" becomes "TOP
            # SIGNAL" whenever the league lacks a validated per-league
            # calibration (Phase 1.5 threshold), and a stale edge benchmark
            # (Phase 0.2) must render as ``edge: n/a`` -- never a stale
            # number. Threaded into the render layers via the payload.
            if signal_engine_result is not None:
                _dec = decision or {}
                signal_engine_result["display_label"] = (
                    "TOP SIGNAL"
                    if (_dec.get("uncalibrated_league") or not _league_calibrated)
                    else "BEST PICK"
                )
                signal_engine_result["edge_invalid"] = bool(_dec.get("edge_invalid"))
                signal_engine_result["edge_benchmark"] = _dec.get("edge_benchmark")
                # P1-2: source-confidence gate veto. When the engine ran but
                # 3+ critical fields were LOW, override the verdict to NO
                # BET with the explicit reason. We KEEP the ranking so the
                # card can show why each candidate failed.
                # Exception: if signal engine produced a strong pick (score >= 0.50
                # and confidence >= MEDIUM), don't veto — the engine's own scoring
                # already accounts for data quality.
                if not _gate_passed:
                    _bp = signal_engine_result.get("best_pick") or {}
                    _bp_score = float(_bp.get("score") or 0)
                    _bp_conf = _bp.get("confidence", "LOW")
                    _strong_pick = _bp_score >= 0.50 and _bp_conf in ("VERY HIGH", "HIGH", "MEDIUM")
                    if _strong_pick:
                        # Signal engine found a good pick despite thin evidence —
                        # downgrade confidence but keep the pick.
                        signal_engine_result.setdefault("reasons", []).append(
                            f"evidence_gate: {_gate_reason} — pick dipertahankan (engine score cukup kuat)"
                        )
                        if _bp_conf in ("VERY HIGH", "HIGH"):
                            _bp["confidence"] = "MEDIUM"
                    else:
                        signal_engine_result["decision"] = "NO BET"
                        signal_engine_result["best_pick"] = None
                        signal_engine_result.setdefault("reasons", []).append(
                            f"evidence_gate: {_gate_reason}"
                        )
                # Phase 5.2: bookmaker count for the card metadata (the
                # signal engine's own data_quality lacks it).
                _dq = signal_engine_result.setdefault("data_quality", {})
                _dq["bookmakers_count"] = len(bookmaker_odds_h2h)
                if _ah_supplemented:
                    _dq["ah_source"] = "nowgoal_supplement"
                # Phase 5.3: lineup status feeds the real "Why" (confirmed /
                # predicted / none). Predicted-only lineups are half-weight
                # inputs, so the card must say which.
                _lu = lineups or {}
                if _lu.get("home_count"):
                    signal_engine_result["lineup_status"] = (
                        _lu.get("status") if _lu.get("status") in ("confirmed", "predicted")
                        else "predicted"
                    )
                elif missing_players:
                    signal_engine_result["lineup_status"] = "none"
                else:
                    signal_engine_result["lineup_status"] = "none"
            # Fix 2: the ONE-TIME lambda pin exception (features genuinely
            # unavailable at pin time, now available) surfaces in the same
            # "🔄 Berubah" pattern as Layer 3 -- never a silent estimator
            # switch. Threshold wobble never reaches here (the engine only
            # sets lambda_source_switch_reason for the allowed case).
            _sw = ((prediction or {}).get("model_probs") or {}).get("lambda_source_switch_reason")
            if _sw and signal_engine_result is not None:
                _stab = signal_engine_result.setdefault("stability", {})
                _note = (
                    "estimator λ beralih sekali (features semula tidak tersedia "
                    "saat pin dibuat, kini tersedia) — pemilihan λ dipin dari "
                    "query pertama; switch berikutnya dilarang"
                )
                if _stab.get("status") == "changed":
                    _stab["reason"] = f"{_stab.get('reason', '')} • {_note}"
                else:
                    _stab.setdefault("status", "changed")
                    _stab.setdefault("previous_selection", (_prev_pick or {}).get("selection"))
                    _stab.setdefault("new_selection", (signal_engine_result.get("best_pick") or {}).get("selection"))
                    _stab["reason"] = _note
        except Exception as exc:
            logger.warning("signal engine failed (prediction unaffected): %s", exc)

    # ---- PHASE 7: immutable prediction snapshot (append-only JSONL) ----
    # Logging must NEVER break the prediction flow. Runs AFTER the decision
    # engine so the snapshot can carry the decision_type label (TODO-15),
    # enabling per-tier realised-performance tracking in production.
    try:
        pl_cfg = cfg.get("prediction_log") or {}
        if pl_cfg.get("enabled") and pl_cfg.get("file") and not _skip_prediction:
            from .prediction_log import append_snapshot

            pred = prediction or {}
            sig = pred.get("signal_strength")
            mp = (pred.get("model_probs") or {})
            hf = home_form or {}
            af = away_form or {}
            # Pre-match input snapshot (PHASE 1): Elo values, Poisson lambdas,
            # attack/defense, form sequences, completeness -- so similar-signal
            # analysis can explain WHY a prediction was made.
            features = {
                "elo_home": elo.rating(home_name) if elo else None,
                "elo_away": elo.rating(away_name) if elo else None,
                "lambda_home": mp.get("lambda_home"),
                "lambda_away": mp.get("lambda_away"),
                "lambda_source": mp.get("lambda_source"),
                # Fix 2: persist the lambda pin for THIS match so later
                # queries reuse the first evaluation's estimator (see
                # models.run_prediction_engine). ``features_available`` at
                # pin time distinguishes the one-time "no data -> data"
                # exception from a threshold wobble.
                "pinned_lambda_source": mp.get("lambda_source"),
                "pinned_features_available_at_pin": mp.get("features_available"),
                "lambda_source_switch_reason": mp.get("lambda_source_switch_reason"),
                "lambda_samples": mp.get("lambda_samples"),
                "attack_home": hf.get("gf_avg"),
                "defense_home": hf.get("ga_avg"),
                "attack_away": af.get("gf_avg"),
                "defense_away": af.get("ga_avg"),
                "form_home": hf.get("sequence"),
                "form_away": af.get("sequence"),
                "completeness": pred.get("data_completeness"),
                "models": mp.get("models"),
                "model_weights": mp.get("model_weights"),
                "elo_seeded": mp.get("elo_seeded"),
            }
            # MARKET PRIOR honesty: a market-mirror prediction must never be
            # recorded as a model bet -- no prob_1x2, no best_pick, no edge.
            # The odds are still logged (for price-CLV tracking) and the
            # decision_type marks the row MARKET PRIOR so the stats stay
            # honest (it can never count as a flat-stake bet or a calibrated
            # model prediction).
            is_market_prior = bool(
                (decision or {}).get("decision_type") == "MARKET PRIOR"
            )
            # Layer 3: persist the signal-engine best pick (selection, score,
            # model signature, ts) as the immutable prior for the next query's
            # stability guard -- including what the model would have picked
            # when the displayed pick was held (audit trail).
            _se_res = signal_engine_result or {}
            _se_bp = _se_res.get("best_pick") or {}
            _se_pick_payload: dict[str, Any] | None = None
            if _se_res.get("decision") == "BEST PICK" and _se_bp:
                _se_pick_payload = {
                    "decision": "BEST PICK",
                    "market": _se_bp.get("market"),
                    "selection": _se_bp.get("selection"),
                    "score": _se_bp.get("score"),
                    "confidence": _se_bp.get("confidence"),
                    "line": _se_bp.get("line"),
                    "side": _se_bp.get("side"),
                    "line_key": (
                        f"ah:{float(_se_bp['line']):+.2f}"
                        if _se_bp.get("market") == "Asian Handicap"
                        and _se_bp.get("line") is not None else None
                    ),
                    # Best-pick evaluation (BEST PICK vs settled result):
                    # the offered odds for the pick, so ROI can be computed
                    # per market without re-fetching. Older snapshots lack it
                    # -- the evaluator falls back to the matching ranking
                    # entry, then to hit-rate-without-ROI.
                    "market_odds": _se_bp.get("market_odds"),
                    "edge_pp": _se_bp.get("edge_pp"),
                    "implied_prob": _se_bp.get("implied_prob"),
                    "over_2.5": mp.get("over_2.5"),
                    "lambda_home": mp.get("lambda_home"),
                    "lambda_away": mp.get("lambda_away"),
                    "ts": utc_now_iso(),
                }
                _stab = _se_res.get("stability") or {}
                if _stab:
                    _se_pick_payload["stability"] = _stab
            # P5: structured pre-match context (lineups / injuries / coaches)
            # for the historical record. Context-only: never a model feature
            # until a backtest validates it (no-OOS-evidence rule). Compact
            # shape: lineups summary + missing/unsure player names per side.
            _context_data: dict[str, Any] | None = None
            if lineups or missing_players or coaches:
                _context_data = {}
                if lineups and lineups.get("home_count"):
                    _context_data["lineups"] = {
                        "status": lineups.get("status"),
                        "formations": list(lineups.get("formations") or []),
                        "home_count": lineups.get("home_count"),
                        "away_count": lineups.get("away_count"),
                        "home": list(lineups.get("home") or []),
                        "away": list(lineups.get("away") or []),
                        "source": lineups.get("source"),
                    }
                if missing_players:
                    _context_data["missing_players"] = {
                        side: {
                            "missing": list((v or {}).get("missing") or []),
                            "unsure": list((v or {}).get("unsure") or []),
                        }
                        for side, v in missing_players.items()
                    }
                if coaches:
                    _context_data["coaches"] = {
                        side: list(v or []) for side, v in coaches.items()
                    }
            # Model A rule: the stored best_pick is the DISPLAYED signal-engine
            # pick (independent engine = Model B) when one exists -- never the
            # market mirror (reference-only, edge 0 by construction).
            # F11 (plan v3 2026-08-24): when the signal engine RAN but emitted
            # NO BET, store NULL -- the old fallback to the decision-layer
            # pick is exactly the Goztepe-class leak (an ungated Over 2.5 got
            # logged/displayed while every candidate was vetoed G4 card-level).
            _snap_best: dict[str, Any] | None = None
            if not is_market_prior:
                if _se_res.get("decision") == "BEST PICK" and _se_pick_payload:
                    _snap_best = _se_pick_payload
                elif signal_engine_result is not None:
                    _snap_best = None
            # Fase 2 anti-flap (blueprint 2026-08-23): identity lock. Sebelum
            # menulis snapshot, bandingkan pasangan tim kanonik dengan riwayat
            # log -- resolver flip (Forest Leeds->Man Utd; Troyes "PSG" dari
            # Paris FC) menahan write daripada mencatat prediksi untuk match
            # yang kemungkinan salah resolve. Kegagalan check TIDAK pernah
            # memblokir logging (fail-open, sama seperti seluruh PHASE 7).
            _identity_lock: dict[str, Any] | None = None
            if not is_market_prior:
                try:
                    from .prediction_log import identity_lock_check

                    _identity_lock = identity_lock_check(
                        ROOT / pl_cfg["file"],
                        match_id=_mid_canon,
                        home=home_name,
                        away=away_name,
                        entities=entities,
                    )
                    if _identity_lock and _identity_lock.get("locked"):
                        logger.warning(
                            "identity_lock: snapshot DITAHAN (%s) — %s (conflict=%s)",
                            _identity_lock.get("kind"),
                            _identity_lock.get("reason"),
                            _identity_lock.get("conflict_match_id"),
                        )
                except Exception as exc:
                    logger.warning("identity lock check failed (write proceeds): %s", exc)
            append_snapshot(
                # Anti-flap P1 (2026-08-23): MARKET PRIOR rows are reference-
                # only market mirrors (no model). Logging them made same-match
                # histories flap between "MARKET PRIOR" and real decisions
                # depending on whether the engine succeeded on a given rerun
                # (Dortmund-Bayern 00:31/00:58 vs 01:05). Default: do NOT log
                # them; the rendered card is unaffected. Anti-flap Fase 2:
                # identity-lock conflict juga menahan write.
                skip=(
                    (
                        is_market_prior
                        and bool(
                            ((cfg.get("models", {}) or {}).get("decision", {}) or {}).get(
                                "market_prior_skip_log", True
                            )
                        )
                    )
                    or bool(_identity_lock and _identity_lock.get("locked"))
                ),
                path=ROOT / pl_cfg["file"],
                # Fix 2026-08-22: canonical entity-id based identity (written
                # side); legacy name-based rows stay readable via _row_by_ids.
                match_id=_mid_canon,
                league=display,
                home=home_name,
                away=away_name,
                kickoff=kickoff,
                prob=(mp.get("1x2") if not is_market_prior else None),
                odds=consensus if has_odds else None,
                edge=(pred.get("market_edge") if not is_market_prior else None),
                confidence=(pred.get("confidence") if not is_market_prior else None),
                signal=(int(sig) if sig is not None else (int(signal) if isinstance(signal, int) else None)) if not is_market_prior else None,
                calibration=(pred.get("calibration") if not is_market_prior else None),
                model_version=(pred.get("model_version") if not is_market_prior else None),
                input_hash=(pred.get("input_hash") if not is_market_prior else None),
                best_pick=_snap_best,
                sources=sources,
                features=features,
                decision_type=(decision or {}).get("decision_type"),
                # Observability fix: persist the decision engine's ACTUAL pick
                # (not just the tier label) so settled matches can be scored
                # against what the engine really chose. n_bucket/pick_status
                # come from the score_breakdown.top (the same candidate).
                final_decision=_final_decision_payload(decision),
                ml_prob=ml_probs,
                edge_benchmark=(decision or {}).get("edge_benchmark"),
                movement=(decision or {}).get("movement"),
                flashscore_url=fixture_flash_url,
                signal_engine_pick=_se_pick_payload,
                # P4 re-runnable: persist the full scored ranking so a later
                # settled-matches backtest can re-weight the stored components
                # with candidate weight sets (movement/late_movement A/B) and
                # settle them against the final score -- no re-fetching needed.
                signal_engine_ranking=(
                    (signal_engine_result or {}).get("ranking") or None
                ),
                context_data=_context_data,
                # Fix 2026-08-22: persist the FULL model probability block so
                # post-hoc evaluation never replays the Poisson matrix again.
                model_probs=mp or None,
                # Phase 1.3: lineup provenance on every snapshot (auditable
                # leakage guard -- reject lineups fetched at/after kickoff).
                lineup_source=(lineups or {}).get("source"),
                lineup_ts=(
                    utc_now_iso() if (lineups and lineups.get("home_count")) else None
                ),
                # Phase 4.2: every signal is logged as paper-traded (no real
                # stake) until its segment passes the Phase 4 DoD.
                paper_trade=bool((cfg.get("prediction_log") or {}).get("paper_trade", True)),
                # G2: canonical entity identity per side for settle verification.
                entities=entities,
                # O/U market odds for total goals calibration
                market_totals=market_totals if market_totals else None,
            )
    except Exception as exc:
        logger.warning("prediction log write failed (prediction unaffected): %s", exc)

    # Cross-provider deduplicated injury list (flashscore + nowgoal) for the
    # user-facing output: one entry per real player, both sources tagged.
    # The raw per-source fields (``missing_players`` / ``nowgoal_context``)
    # stay untouched for auditability.
    injuries_merged = None
    try:
        if missing_players or (nowgoal_context or {}).get("injuries"):
            from .player_identity import merge_missing_lists

            injuries_merged = merge_missing_lists(
                missing_players,
                (nowgoal_context or {}).get("injuries") or {},
            )
    except Exception:  # noqa: BLE001 -- context field, never blocks output
        injuries_merged = None

    return {
        "league": display,
        "league_key": league_key,
        # D2 (dynamic league discovery): True when the league was read FROM
        # the fixture (unregistered competition) -- the prediction is honestly
        # labelled uncalibrated_league and there is no registered odds key.
        "dynamic_league": _dynamic_league,
        # Competition cross-check (2026-08-17): set when the fixture's real
        # competition differs from the league the user typed; the standings
        # fetch already used the correct key. Shown so a mis-query is never
        # silent ("requested laliga, actual laliga2").
        "league_mismatch": league_mismatch,
        "generated_at": utc_now_iso(),
        "prediction": prediction,
        "home": home_name,
        "away": away_name,
        "kickoff": kickoff,
        "venue": (fixture or {}).get("venue"),
        "match_found": fixture is not None,
        "fixture_source": (fixture or {}).get("source"),
        # Finished-match reporting: when kickoff is in the past the analysis
        # shows the real result (score + post-match stats) instead of a
        # prediction; market tiers / decision engine are skipped.
        "match_finished": match_finished,
        "kickoff_uncertain": kickoff_uncertain,
        "kickoff_deltas": kickoff_deltas,
        "match_result": (fixture or {}).get("score") if match_finished else None,
        "stats": {
            "home_form": (home_form or {}).get("sequence", "n/a"),
            "away_form": (away_form or {}).get("sequence", "n/a"),
            "home_gf_avg": (home_form or {}).get("gf_avg", 0),
            "home_ga_avg": (home_form or {}).get("ga_avg", 0),
            "away_gf_avg": (away_form or {}).get("gf_avg", 0),
            "away_ga_avg": (away_form or {}).get("ga_avg", 0),
            "home_split": (home_form or {}).get("home", {}),
            "away_split": (away_form or {}).get("away", {}),
            "h2h": {
                "wins": (h2h or {}).get("wins", 0),
                "draws": (h2h or {}).get("draws", 0),
                "losses": (h2h or {}).get("losses", 0),
            },
            "home_xg_for": (home_history or {}).get("xg_for_avg"),
            "away_xg_for": (away_history or {}).get("xg_for_avg"),
            "home_xg_against": (home_history or {}).get("xg_against_avg"),
            "away_xg_against": (away_history or {}).get("xg_against_avg"),
            "home_corners_for": (home_history or {}).get("corners_for_avg"),
            "away_corners_for": (away_history or {}).get("corners_for_avg"),
            "home_corners_against": (home_history or {}).get("corners_against_avg"),
            "away_corners_against": (away_history or {}).get("corners_against_avg"),
            "home_yellow_for": (home_history or {}).get("yellow_for_avg"),
            "away_yellow_for": (away_history or {}).get("yellow_for_avg"),
        },
        "odds": {
            "consensus": consensus,
            "best": best,
            "outlier": outlier,
            "bookmakers_count": len(bookmaker_odds_h2h),
            "has_odds": has_odds,
            "totals": market_totals,
            "movement": movement,
            "value": value,
            "value_history": value_history,
            "quality": odds_quality,
        },
        "signal": signal,
        "picks": picks_payload,
        "sources": sources,
        "similar_signal": similar_signal,
        "decision": decision,
        "signal_engine": signal_engine_result,
        "data_sources": unified_dict,
        "coverage": coverage_dict,
        "confidence": build_confidence_block(decision),
        "event_stats": match_stats,
        "lineups": lineups,
        "missing_players": missing_players,
        # Cross-provider deduplicated injury list (flashscore + nowgoal) --
        # one entry per player with provenance; the raw per-source fields
        # above are kept for auditability.
        "injuries_merged": injuries_merged,
        "coaches": coaches,
        "standings": standings,
        "match_info": match_info,
        "nowgoal_context": nowgoal_context,
        # Match-source provenance: which source found the match (livescore /
        # flashscore) and the collected source context (identity + fields).
        # None for the plain `analisa match` path.
        "match_source": (source_match or {}).get("source"),
        "source_match": source_match,
        # Identity firewall provenance (plan 2026-08-24): the pinned canonical
        # entity per side + which pre-flight checks ran, so an audit can show
        # WHICH clubs were analysed without re-deriving them from names.
        "identity": {
            "entities": entities,
            "checks": _identity_checks,
            "firewall": {
                k: _fw.get(k)
                for k in ("enabled", "refuse_divergence", "refuse_history_lock")
            },
        },
        "quota": {
            "odds_api_remaining": odds.last_remaining,
            "odds_blocked": odds.quota_blocked,
            "football_data_warning": stats.fd.rate_limit_warning,
            "oddspapi_used": oddspapi_source,
            # 2026-08-22: surfaced on the card -- a silent 429 must not read
            # as "odds are just missing" when the user can top up the quota.
            "oddspapi_quota_exhausted": bool(
                getattr(oddspapi, "quota_exhausted", False)
            ),
            "oddspapi_remaining": getattr(oddspapi, "last_remaining", None),
            "oddspapi_pool": getattr(oddspapi, "pool_status", None),
            "nowgoal_used": nowgoal_source,
        },
    }
