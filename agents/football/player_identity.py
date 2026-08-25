"""Cross-provider player identity for injury / missing-player list merging.

Flashscore and NowGoal report the same injured players under different name
styles (verified live 2026-08-16, Espanyol-Levante):

  - flashscore  : "Garcia K."  / "Puado J."  / "Brugue R."  (surname + initial)
  - nowgoal     : "Kike Garcia" / "Javi Puado" / "Roger Brugue" (given + surname)
  - nowgoal     : "Enrique Garcia Martinez, Kike"  (full name + quoted nickname)

Merging therefore needs tolerant identity, not exact strings. The rules:

  1. Normalize every name: lowercase, strip accents/punctuation, drop
     honorific suffixes (jr/sr/ii/iii/...), keep single letters as initials.
  2. Split each name into surname tokens (the last word; the last TWO for a
     >=3-word name, which captures Spanish compound surnames "Garcia
     Martinez"), given tokens, initials and quoted nicknames (nicknames
     count as given names too).
  3. Two entries are the SAME player iff their surname sets intersect AND a
     secondary signal agrees: nickname overlap, given-name overlap, or a
     single-letter initial matching the start of the other side's given name.
     This merges "Garcia K." with "Kike Garcia" and "Enrique Garcia
     Martinez, Kike" while keeping "Primo A." distinct from "Roger Brugue".

The merge is used for CONTEXT ONLY (signal-engine Group E, weight 0 by
default) and for the user-facing output -- never a model feature.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# Honorific / generational suffixes that are never part of a surname.
_SUFFIX_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Surname particles: the last word is the surname, but a preceding particle
# ("van", "de", "da", ...) belongs to it ("Virgil van Dijk" -> van Dijk).
_PARTICLES = frozenset(
    {"de", "da", "di", "del", "van", "von", "der", "den", "la", "le",
     "el", "do", "dos", "du", "d", "l"}
)


def _norm_word(token: str) -> str:
    s = unicodedata.normalize("NFD", (token or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s)


def _tokens(name: str) -> tuple[list[str], set[str], list[str]]:
    """Normalized tokens of a player name.

    Returns (primary_tokens, quoted_nicknames, extra_tokens): the primary
    part (before any comma) holds the real name; the trailing comma parts
    ("Enrique Garcia Martinez, Kike") are given-name/nickname candidates;
    quoted nicknames ("… 'Kike'") are tracked separately.
    """
    raw = name or ""
    nick_raw = re.findall(r"\"([^\"]+)\"", raw)
    nicknames = {_norm_word(n) for n in nick_raw} - {""}
    s = re.sub(r"\"[^\"]*\"", " ", raw)
    parts = [p for p in s.split(",") if p.strip()]
    primary, *extra = parts if parts else [s]

    def _clean(text: str) -> list[str]:
        # Strip accents BEFORE splitting: an accented char ("í") is not
        # [a-zA-Z0-9], so splitting first would cut "García" into "Garc"+"a".
        s = unicodedata.normalize("NFD", (text or "").lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        out = []
        for t in re.split(r"[^a-z0-9]+", s):
            w = _norm_word(t)
            if w and w not in _SUFFIX_TOKENS:
                out.append(w)
        return out

    return _clean(primary), nicknames, [w for p in extra for w in _clean(p)]


def _player_identity(name: str) -> dict[str, set[str]]:
    """Identity facets of a player name: surnames / given / initials / nicknames."""
    tokens, nicknames, extra = _tokens(name)
    initials = {t for t in tokens if len(t) == 1}
    words = [t for t in tokens if len(t) > 1]
    if not words and not nicknames and not extra:
        return {"surnames": set(), "given": set(),
                "initials": initials, "nicknames": set()}
    if len(words) >= 2 and words[-1] in _PARTICLES:
        surnames = {words[-2], words[-1]}
    elif len(words) >= 3:
        surnames = {words[-2], words[-1]}  # compound surname ("garcia martinez")
    else:
        surnames = {words[-1]}
    given = set(words[:-1] if len(words) >= 2 else [])
    given |= nicknames            # a quoted nickname acts as a given name
    given |= set(extra)           # post-comma tokens are given/nickname too
    return {"surnames": surnames, "given": given,
            "initials": initials, "nicknames": nicknames | set(extra)}


def players_match(a: str, b: str) -> bool:
    """True when two player-name spellings refer to the same human.

    Same surname (set overlap) is the hard gate; a secondary signal must
    then agree: shared nickname / given name, or a single-letter initial
    matching the start of the other side's given name ("Garcia K." vs
    "Kike Garcia"). Two surname-only names match only on exact surname
    equality (there is nothing else to disambiguate).
    """
    ia, ib = _player_identity(a), _player_identity(b)
    if not (ia["surnames"] and ib["surnames"]):
        return False
    if not (ia["surnames"] & ib["surnames"]):
        return False
    # Secondary signals.
    if ia["nicknames"] & ib["nicknames"]:
        return True
    if ia["nicknames"] & ib["given"] or ib["nicknames"] & ia["given"]:
        return True
    if ia["given"] & ib["given"]:
        return True
    for ini in ia["initials"] | ib["initials"]:
        other_given = ib["given"] if ini in ia["initials"] else ia["given"]
        if any(g.startswith(ini) for g in other_given):
            return True
    # Both surname-only ("Mikelionis" vs "Mikelionis"): surname AND initial
    # equality ("Garcia K." vs "Garcia J." are different people).
    return (
        not ia["given"] and not ib["given"]
        and ia["initials"] == ib["initials"]
        and ia["surnames"] == ib["surnames"]
    )


def _side_entries(
    flashscore_missing: dict[str, Any] | None,
    nowgoal_injuries: dict[str, Any] | None,
    side: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    fs = (flashscore_missing or {}).get(side) or {}
    for item in fs.get("missing") or []:
        if isinstance(item, dict):
            entries.append({"name": str(item.get("name") or ""),
                            "reason": item.get("reason"),
                            "sources": ["flashscore"], "unsure": False})
        elif item:
            entries.append({"name": str(item), "sources": ["flashscore"], "unsure": False})
    for item in fs.get("unsure") or []:
        if isinstance(item, dict):
            entries.append({"name": str(item.get("name") or ""),
                            "reason": item.get("reason"),
                            "sources": ["flashscore"], "unsure": True})
        elif item:
            entries.append({"name": str(item), "sources": ["flashscore"], "unsure": True})
    for item in (nowgoal_injuries or {}).get(side) or []:
        if isinstance(item, dict):
            entries.append({"name": str(item.get("name") or ""),
                            "reason": None,
                            "position": item.get("position"),
                            "number": item.get("number"),
                            "player_id": item.get("player_id"),
                            "sources": ["nowgoal"], "unsure": False})
        elif item:
            entries.append({"name": str(item), "sources": ["nowgoal"], "unsure": False})
    return entries


# A single team's REAL absence list is single digits to low teens; a full
# squad is 22-30+ rows. Verified failure 2026-08-23 (Club Brugge v Cercle
# Brugge): the analysis page's injury container carried BOTH squads (~44
# player-rows) under one side marker, and the merge dutifully logged every
# starter of both teams as "missing". When ONE side's nowgoal list alone
# exceeds this row count, the nowgoal payload is an unsegmented dump, not a
# two-team absence table -- drop the whole nowgoal contribution (fail open
# toward flashscore, the richer verified source for this field).
_MAX_NOWGOAL_ROWS_PER_SIDE = 18


def merge_missing_lists(
    flashscore_missing: dict[str, Any] | None,
    nowgoal_injuries: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Merge flashscore + nowgoal injury lists into ONE deduplicated list.

    Returns ``{home: [entry...], away: [...]}`` (only sides with entries),
    or None when neither source reports anything. Each entry carries the
    union of details ({name, reason?, position?, number?, player_id?,
    sources, unsure}) -- the same human reported by both providers appears
    exactly once, with the more informative display name and both sources
    tagged.

    Two guards keep a broken provider payload out of the merged record:
      - squad-dump: one side's nowgoal list alone carrying more than
        ``_MAX_NOWGOAL_ROWS_PER_SIDE`` rows is an unsegmented page dump
        (both squads under one marker), so the ENTIRE nowgoal contribution
        is dropped and flashscore stands alone;
      - cross-side duplicate: the same human cannot be absent for BOTH
        teams, so an entry that matches one already merged for the other
        side is dropped (home wins the tie deterministically).
    """
    # Squad-dump guard (see _MAX_NOWGOAL_ROWS_PER_SIDE): evaluate on the RAW
    # nowgoal lists BEFORE any merging inflates counts.
    ng_rows = {
        side: len((nowgoal_injuries or {}).get(side) or [])
        for side in ("home", "away")
    }
    if any(n > _MAX_NOWGOAL_ROWS_PER_SIDE for n in ng_rows.values()):
        nowgoal_injuries = None

    out: dict[str, list[dict[str, Any]]] = {}
    seen_other_side: list[dict[str, Any]] = []
    for side in ("home", "away"):
        merged: list[dict[str, Any]] = []
        for e in _side_entries(flashscore_missing, nowgoal_injuries, side):
            if not e.get("name"):
                continue
            if any(players_match(m["name"], e["name"]) for m in seen_other_side):
                # Same human already recorded for the opposite team: a
                # player cannot be missing from two opposing lineups at
                # once -- one of the two sides mislabelled them.
                continue
            hit = next(
                (m for m in merged if players_match(m["name"], e["name"])),
                None,
            )
            if hit is None:
                merged.append(dict(e))
                continue
            for key in ("reason", "position", "number", "player_id"):
                if not hit.get(key) and e.get(key):
                    hit[key] = e[key]
            hit["sources"] = sorted(set(hit["sources"]) | set(e["sources"]))
            hit["unsure"] = bool(hit["unsure"] or e["unsure"])
            if len(e["name"]) > len(hit["name"]):
                hit["name"] = e["name"]
        if merged:
            out[side] = merged
            seen_other_side.extend(merged)
    return out or None
