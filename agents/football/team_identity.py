"""Team-name identity: ONE strict, shared matcher for every provider lookup.

Post-mortem 2026-09-02 (wrong-team data). Every provider matched team names
with its own copy of a *substring* heuristic (``analyse._teams_match``,
``nowgoal._same_team``, ``oddspapi._same_team``, ``tie_state._same_team``,
``soccerdata_wrapper`` raw ``in``). Substring containment accepted:

    "Southampton"      <- "South Carolina United FC"   ("south" in "southampton")
    "Stoke City"       <- "Basingstoke"                ("stoke" in "basingstoke")
    "Portsmouth"       <- "Port City FC"               ("port" in "portsmouth")
    "Birmingham City"  <- "Birmingham City U18"        (youth side, no marker check)
    "Parma"            <- "Parma U20"

so the LiveScore by-name form (used for every VPS analysis on 31 Aug-1 Sep)
built form / attack / defence from other clubs' results (Southampton stored
as L-D-L-L-D while the real club had gone W-L-W-L-W). The lambdas, the
Poisson probabilities and the BEST PICK were then computed on a different
team.

Rules here (pure, no I/O):

* names are compared at TOKEN level -- never substring;
* club prefixes/suffixes without identity (FC, FK, SC, AFC, ...) are noise;
* a reserve / youth / women marker must be present on BOTH sides or the
  names are different teams ("Ajax" is not "Jong Ajax");
* a club-type qualifier (City, United, Town, ...) present on BOTH sides must
  agree ("Lincoln City" is not "Lincoln United"); one side may omit it
  ("Stoke" == "Stoke City");
* the shorter name's identity tokens must all be matched (exact, plural
  stem, or a dotted abbreviation prefix like "Dyn." -> "Dynamo"); the longer
  name may carry at most ONE extra identity token (a city / honorific:
  "Tobol Kostanay", "Royale Union Saint-Gilloise"); ``strict=True`` allows
  none (used when nothing else -- id, league, country -- can vouch);
* ``match_side`` refuses a name that fits BOTH sides of a fixture.

The residual ambiguity of a single-token name ("Inter" vs "FC Inter Turku",
"Wolves" vs "Wollongong Wolves") is by design NOT solved by names: callers
must scope by provider id, league or country (see ``multi_source``).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

_STROKE_LETTERS = str.maketrans({
    "ø": "o", "ł": "l", "đ": "d", "ħ": "h", "ı": "i", "ŋ": "n", "ß": "ss", "æ": "ae", "œ": "oe",
})

# Tokens that carry no club identity on either side.
NOISE_TOKENS = frozenset({
    "fc", "fk", "sc", "afc", "ac", "as", "cf", "cd", "ca", "ec", "cr", "se", "sd", "sv", "sk",
    "nk", "hnk", "gnk", "kv", "krc", "rsc", "ss", "ssc", "us", "pfc", "ifk", "rc", "bk", "if",
    "ik", "ff", "kf", "fk", "ks", "sl", "vfb", "vfl", "tsg", "tsv", "fsv", "sg", "spvgg",
    "club", "clube", "cf", "de", "del", "da", "do", "di", "la", "le", "les", "el", "los",
    "the", "of", "and", "y", "e", "team", "calcio", "futebol", "sociedade", "esportiva",
    "asociacion", "association", "sportive", "sportif", "spor", "kulubu", "kulübü",
    # Spanish / Portuguese / Dutch / German club-type prefixes and suffixes
    "rcd", "ud", "sad", "cp", "asd", "gd", "bsc", "kaa", "vv", "sbv", "nac", "ssv",
    "cfr", "csm", "fcsb", "kfc", "kvc", "kvk", "rfc", "afk", "sfc", "cs", "cska",
})

# Club-type qualifiers: identity-bearing when BOTH sides carry one.
QUALIFIER_TOKENS = frozenset({
    "city", "united", "town", "rovers", "athletic", "wanderers", "county", "albion",
    "hotspur", "wednesday", "orient", "argyle", "villa", "forest", "palace", "rangers",
    "celtic", "thistle", "vale", "north", "south", "east", "west", "olympic", "olympique",
    "sporting", "real", "atletico", "athletic", "dinamo", "dynamo", "lokomotiv", "spartak",
    "zenit", "torpedo", "inter", "internacional", "nacional", "national", "racing", "union",
    "stade", "sc", "junior",
})

# Markers of a SECOND side of a club (reserve / youth / women). Asymmetric
# presence == different team. Matched on lowercase tokens; Roman numerals
# are matched on the raw name (uppercase) so the Finnish club "Ii" survives.
_MARKER_TOKENS = frozenset({
    "women", "womens", "ladies", "feminine", "feminin", "femenino", "femminile", "damen",
    "frauen", "kobiet", "dames", "w", "reserves", "reserve", "youth", "academy", "jugend",
    "junior", "juniors", "primavera", "amateur", "amateurs", "jong", "u15", "u16", "u17",
    "u18", "u19", "u20", "u21", "u23", "b", "2", "3",
})
# Roman numerals are matched on the RAW name in uppercase only: the Finnish
# first-team club "Ii" must not read as a reserve XI.
_ROMAN_RE = re.compile(r"\b(?:II|III|IV)\b")


def _fold(name: str | None) -> str:
    s = re.sub(r"\([^)]*\)", " ", name or "")          # "(Kaz)", "(w)" country / gender tags
    s = s.lower().translate(_STROKE_LETTERS)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\bmunchen\b", "munich", s)
    return s


def _gender_tag(name: str | None) -> bool:
    """"(W)" / "(Women)" tags in parentheses are markers, not country tags."""
    return re.search(r"\(\s*(?:w|women|ladies|fem\w*)\s*\)", (name or "").lower()) is not None


class Identity:
    """Tokenised identity of one team name."""

    __slots__ = ("raw", "core", "qualifiers", "markers", "abbrevs", "folded")

    def __init__(self, name: str | None) -> None:
        self.raw = name or ""
        folded = _fold(name)
        self.folded = " ".join(re.sub(r"[^a-z0-9. ]", " ", folded).replace(".", ". ").split())
        raw_tokens = re.findall(r"[a-z0-9]+\.?", self.folded)
        self.abbrevs: set[str] = set()
        core: list[str] = []
        markers: set[str] = set()
        for tok in raw_tokens:
            dotted = tok.endswith(".")
            t = tok.rstrip(".")
            if not t:
                continue
            if t in _MARKER_TOKENS:
                markers.add("ii" if t in ("b", "2", "3") else ("youth" if t.startswith("u") or t in ("youth", "academy", "jugend", "junior", "juniors", "primavera", "jong") else ("women" if t in ("women", "womens", "ladies", "feminine", "feminin", "femenino", "femminile", "damen", "frauen", "kobiet", "dames", "w") else "reserve")))
                continue
            if t in NOISE_TOKENS or len(t) == 1:
                continue
            if dotted and len(t) <= 4:
                self.abbrevs.add(t)
            core.append(t)
        if _ROMAN_RE.search(self.raw) and "ii" not in markers:
            markers.add("ii")
        if _gender_tag(self.raw):
            markers.add("women")
        # A bare trailing "b"/"2" only counts as a marker when other tokens exist.
        self.core = core
        self.markers = markers
        self.qualifiers = {t for t in core if t in QUALIFIER_TOKENS}


def _token_eq(a: str, b: str, abbrevs: set[str]) -> bool:
    if a == b:
        return True
    # plural / possessive stem: hearts == heart, wolves == wolf? (keep simple: trailing s)
    if len(a) > 3 and len(b) > 3 and (a.rstrip("s") == b.rstrip("s")):
        return True
    # dotted abbreviation prefix: "dyn." -> "dynamo", "dep." -> "deportivo", "atl." -> "atletico"
    if a in abbrevs and len(a) >= 2 and b.startswith(a):
        return True
    if b in abbrevs and len(b) >= 2 and a.startswith(b):
        return True
    return False


def names_match(a: str | None, b: str | None, *, strict: bool = False) -> bool:
    """True when ``a`` and ``b`` name the same first/second team (see module doc)."""
    ia, ib = Identity(a), Identity(b)
    if not ia.core or not ib.core:
        return bool(ia.folded) and ia.folded == ib.folded
    if ia.folded == ib.folded:
        return True
    if ia.markers != ib.markers:
        return False
    if ia.qualifiers and ib.qualifiers and ia.qualifiers != ib.qualifiers:
        return False
    short, long_ = (ia, ib) if len(ia.core) <= len(ib.core) else (ib, ia)
    abbrevs = ia.abbrevs | ib.abbrevs
    unmatched_long = list(long_.core)
    for t in short.core:
        hit = next((w for w in unmatched_long if _token_eq(t, w, abbrevs)), None)
        if hit is None:
            return False
        unmatched_long.remove(hit)
    # Extra tokens on the longer name: club-type qualifiers ("City",
    # "United") never add identity when the other side simply omits them;
    # any other extra token is a city / honorific -- one is tolerated
    # ("Tobol Kostanay"), none in strict mode.
    extra_identity = [t for t in unmatched_long if t not in QUALIFIER_TOKENS]
    if strict:
        return not extra_identity
    return len(extra_identity) <= 1


def match_side(
    target: str | None,
    home: str | None,
    away: str | None,
    *,
    strict: bool = False,
) -> str | None:
    """"home" / "away" when ``target`` names exactly ONE side, else None.

    None covers both "neither" and "both" -- an ambiguous name is never
    assigned to a side (the Inter / Inter Turku guard).
    """
    mh = names_match(target, home, strict=strict)
    ma = names_match(target, away, strict=strict)
    if mh == ma:
        return None
    return "home" if mh else "away"


def same_fixture(
    home: str | None,
    away: str | None,
    cand_home: str | None,
    cand_away: str | None,
    *,
    strict: bool = False,
) -> str | None:
    """"ordered" / "reversed" when the candidate pair is the requested pair.

    None when it is a different fixture OR when the assignment is ambiguous
    (a name matching both candidate sides).
    """
    ordered = names_match(home, cand_home, strict=strict) and names_match(away, cand_away, strict=strict)
    reversed_ = names_match(home, cand_away, strict=strict) and names_match(away, cand_home, strict=strict)
    if ordered and not reversed_:
        return "ordered"
    if reversed_ and not ordered:
        return "reversed"
    return None


def any_match(target: str | None, candidates: Iterable[str | None], *, strict: bool = False) -> bool:
    return any(names_match(target, c, strict=strict) for c in candidates)


def distinct_clubs(names: Iterable[str | None]) -> int:
    """Number of DIFFERENT clubs among ``names`` (names that match each other
    count once). Used to refuse a by-name lookup that gathered rows from two
    clubs ("Inter Milan" and "FC Inter Turku" for the query "Inter")."""
    clusters: list[str] = []
    for n in names:
        if not n:
            continue
        if not any(names_match(n, c) for c in clusters):
            clusters.append(n)
    return len(clusters)


_FS_URL_RE = re.compile(r"/match/(?:football/)?([a-z0-9-]+?)(?:-[A-Za-z0-9]{8})?/([a-z0-9-]+?)(?:-[A-Za-z0-9]{8})?/")


def flashscore_url_matches(url: str | None, home: str | None, away: str | None) -> str | None:
    """"ordered" / "reversed" when the two team slugs in a flashscore match
    url name the analysed pair; None when they name ANOTHER pair; "unknown"
    when the url carries no readable slugs (fail-open for the caller).

    Every context fetch (stats, lineups, match info, event context) is keyed
    by this url, so a url that belongs to another fixture would attach the
    other fixture's data to this analysis. Slug order is not home-first on
    flashscore, so either orientation is accepted.
    """
    if not url:
        return None
    m = _FS_URL_RE.search(url)
    if not m:
        return "unknown"
    s1, s2 = m.group(1).replace("-", " "), m.group(2).replace("-", " ")
    # Placeholder / truncated slugs ("a", "b") cannot be verified -- say so
    # instead of vetoing (callers treat "unknown" as fail-open).
    if len(re.sub(r"[^a-z0-9]", "", s1)) < 3 or len(re.sub(r"[^a-z0-9]", "", s2)) < 3:
        return "unknown"
    return same_fixture(home, away, s1, s2)


def has_marker(name: str | None) -> bool:
    """True when the name carries a reserve / youth / women marker."""
    return bool(Identity(name).markers)


def country_matches(a: Any, b: Any) -> bool | None:
    """Compare two country labels; None when either is unknown."""
    if not a or not b:
        return None
    return _fold(str(a)).strip() == _fold(str(b)).strip()
