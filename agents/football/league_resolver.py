"""League mapping + alias resolution."""

from __future__ import annotations

import json
from pathlib import Path

LEAGUES_PATH = Path(__file__).parent / "leagues.json"


def load_leagues() -> dict[str, dict]:
    return json.loads(LEAGUES_PATH.read_text(encoding="utf-8"))


def iter_alias_index(leagues: dict[str, dict] | None = None) -> dict[str, str]:
    """Map every alias (lowercased) -> canonical league key."""
    leagues = leagues if leagues is not None else load_leagues()
    idx: dict[str, str] = {}
    for key, meta in leagues.items():
        idx[key.lower()] = key
        idx[meta["display"].lower()] = key
        for alias in meta.get("aliases", []):
            idx[alias.lower()] = key
    return idx


def resolve_league(query: str, leagues: dict[str, dict] | None = None) -> tuple[str, dict] | None:
    """Substring match a free-form league keyword against the alias index.

    Returns (canonical_key, meta) or None.
    """
    if not query:
        return None
    leagues = leagues if leagues is not None else load_leagues()
    idx = iter_alias_index(leagues)
    q = query.strip().lower()

    if q in idx:
        key = idx[q]
        return key, leagues[key]

    for alias, key in idx.items():
        if alias in q or q in alias:
            return key, leagues[key]

    return None


def resolve_league_scored(
    query: str,
    leagues: dict[str, dict] | None = None,
) -> tuple[str, dict] | None:
    """Free-form league query -> (key, meta) without the substring trap.

    User queries are free text ("spanish segunda", "liga 2", "liga italy")
    and ``resolve_league``'s bare-substring fallback mis-resolves them (B4,
    verified 2026-08-17: "spanish segunda" hit the "spanish" alias and
    resolved to LaLiga instead of Segunda). Resolution order:
      1. exact full-match against the alias index;
      2. the LAST query token as an exact alias (handles the
         "<country/lang> <league>" pattern: "spanish segunda" -> Segunda,
         "liga italy" -> Serie A, "liga spanyol" -> LaLiga, "french ligue 1"
         -> Ligue 1 via the "french" prefix);
      3. every full-query PREFIX as an exact alias, longest first (so
         "champions league qualifiers" -> UCL and "bundesliga 2" falls
         through to the "bundesliga" prefix instead of "ligue 2");
      4. legacy substring fallback (``resolve_league``) so every other
         query keeps its current behaviour exactly.
    """
    if not query:
        return None
    leagues = leagues if leagues is not None else load_leagues()
    idx = iter_alias_index(leagues)
    q = query.strip().lower()
    if q in idx:
        key = idx[q]
        return key, leagues[key]
    tokens = q.split()
    if not tokens:
        return None
    last = tokens[-1]
    if last in idx:
        key = idx[last]
        return key, leagues[key]
    for n in range(len(tokens), 0, -1):
        prefix = " ".join(tokens[:n])
        if prefix in idx:
            key = idx[prefix]
            return key, leagues[key]
    return resolve_league(query, leagues)


def resolve_league_leading(query: str, leagues: dict[str, dict] | None = None) -> tuple[str, dict] | None:
    """Match only if the full query equals an alias exactly.

    Strict: the entire query string (after lowercasing) must be a key in
    the alias index. The caller controls how many tokens to consume.
    """
    if not query:
        return None
    leagues = leagues if leagues is not None else load_leagues()
    idx = iter_alias_index(leagues)
    q = query.strip().lower()
    if q in idx:
        return idx[q], leagues[idx[q]]
    return None


def _country_alias_set(leagues: dict[str, dict] | None = None) -> set[str]:
    """Lowercased country/region names declared in leagues.json.

    The alias index also maps bare country names ('england' -> EPL) for user
    queries like `analisa match italy ...`. In the competition-title context
    those must NOT resolve, otherwise a homepage title like 'England Cup'
    would be tagged as EPL (its teams are not analyzable as a league match).
    """
    leagues = leagues if leagues is not None else load_leagues()
    return {
        str(meta.get("country") or "").strip().lower()
        for meta in leagues.values()
        if meta.get("country")
    }


# Language demonyms used as league aliases ('french' -> Ligue 1, 'spanish' ->
# LaLiga) for user queries. In a COMPETITION TITLE a leading demonym is a
# descriptor, not the competition itself: 'French Trophée des Champions' is a
# cup (not Ligue 1) and 'Spanish La Liga 2' is the SECOND division. The
# country-name guard below only covered 'france'/'spain' -- the demonyms
# leaked and mis-tagged cups / lower divisions as the top league (B3,
# verified 2026-08-17).
_DEMONYM_SET = {
    "french", "spanish", "english", "italian", "german", "dutch",
    "portuguese", "turkish", "scottish", "belgian", "austrian",
    "swiss", "danish", "swedish", "norwegian", "greek", "polish",
    "russian", "ukrainian", "romanian", "croatian", "serbian",
    "czech", "slovak", "hungarian", "bulgarian", "japanese",
    "korean", "australian", "mexican", "brazilian", "argentine",
    "american", "saudi", "qatari", "emirati", "indonesian",
    "malaysian", "thai", "chinese", "indian", "south african",
}


def _resolve_tokens(
    tokens: list[str],
    countries: set[str],
) -> str | None:
    """Longest-prefix resolution of a token list, skipping bare country /
    demonym prefixes (they are descriptors, not league titles)."""
    for n in range(len(tokens), 0, -1):
        prefix = " ".join(tokens[:n])
        resolved = resolve_league_leading(prefix)
        if resolved:
            low = prefix.strip().lower()
            if low in countries or low in _DEMONYM_SET:
                continue
            return resolved[0]
    return None


def competition_league_key(competition: str) -> str | None:
    """Map a homepage competition title to a registered league key.

    Tries every prefix of the title so 'Champions League - Qualification'
    resolves through the 'Champions League' prefix to 'UCL'. Bare country /
    language prefixes ('England Cup', 'French Trophée des Champions') are
    skipped -- they are descriptors, not league titles. When the first pass
    fails, a leading country/demonym token is STRIPPED and the remainder is
    re-matched, so 'Spanish La Liga 2' resolves to Segunda (not LaLiga) and
    'French Trophée des Champions' stays None (a cup). Returns None for
    competitions with no registered league (friendlies, minor cups) -- exactly
    the matches the top command must hide because they cannot be analyzed
    end-to-end.
    """
    if not competition:
        return None
    tokens = competition.split()
    countries = _country_alias_set()
    hit = _resolve_tokens(tokens, countries)
    if hit is not None:
        return hit
    first = tokens[0].strip(".,").lower()
    if first in countries or first in _DEMONYM_SET:
        return _resolve_tokens(tokens[1:], countries)
    return None


DYNAMIC_PREFIX = "dyn:"


def dynamic_league_key(competition: str) -> str:
    """Deterministic league key for an UNREGISTERED competition.

    Format ``dyn:{slug}`` (e.g. "dyn:copa-del-rey"). Properties:
      - never collides with registered keys (distinct ``dyn:`` prefix);
      - recomputable at any time -- no persistence needed;
      - stable across calls, so the G1 entity registry and the form/h2h
        cache keys derived from it stay consistent run to run.
    """
    import re as _re

    slug = _re.sub(r"[^a-z0-9]+", "-", (competition or "").lower()).strip("-")
    return f"{DYNAMIC_PREFIX}{slug or 'unknown'}"


def dynamic_league_meta(
    competition: str,
    *,
    country: str | None = None,
) -> dict:
    """Meta dict for an unregistered competition (dynamic league).

    Carries ``dynamic: True`` so downstream code can label the analysis as
    uncalibrated / no registered odds key. ``odds_api_key`` is None so the
    The-Odds-API branch is skipped and odds come from oddspapi/nowgoal by
    name (the existing fallback chain).
    """
    return {
        "display": competition or "Dynamic",
        "country": country or "",
        "aliases": [competition] if competition else [],
        "football_data_code": None,
        "odds_api_key": None,
        "provider": None,
        "fallback_provider": None,
        "dynamic": True,
    }
