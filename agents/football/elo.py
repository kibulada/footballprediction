"""Elo rating model. Pure Python, no external deps.

Ratings persist to a local JSON file so the live bot shares the exact
ratings the backtest seeded (``python -m agents.football.backtest --seed-elo``).

Leakage rule: ``update()`` must only be called AFTER a match has finished
(the backtest calls it after each replayed kickoff). ``expected_lambdas`` /
``probabilities`` use ONLY the current rating state -> pre-match by
construction.

Name resolution: seeded ratings come from football-data.co.uk CSVs ("Arsenal")
while the live provider chain returns its own spellings (football-data.org
"Arsenal FC", thesportsdb "Celtic"). ``rating`` / ``known`` / ``update``
resolve a query name to a seeded key via normalized exact match first, then a
teams.json canonical-name alias index, then partial token scoring with
spelling synonyms (united~utd, man~manchester). Ambiguous matches resolve to
None (honest "not known") rather than guessing, so live lookups hit the seeded
ratings without silently mis-claiming teams.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from .team_alias import load_teams

# Built-in extra aliases (2026-08-23): abbreviated/dotted LIVE spellings
# whose significant-token overlap with their seed key is too thin for the
# fuzzy pass ("G.A. Eagles" -> seed "Go Ahead Eagles": drop single-letter
# tokens and only "eagles" survives -> unseeded 1500 prior). Explicit
# mapping beats loosening global thresholds (false-positive risk). Every
# target MUST exist in ratings; checked at build time.
_EXTRA_ALIASES: dict[str, str] = {
    "G.A. Eagles": "Go Ahead Eagles",
    "GA Eagles": "Go Ahead Eagles",
    "Go Ahead": "Go Ahead Eagles",
    "Den Haag": "ADO Den Haag",
    # K2 (post-mortem 2026-08-28): short LIVE spellings of European-cup
    # regulars whose seed key shares no significant token ("Lyon" vs
    # "Olympique Lyonnais") or ties against a reserve key ("Benfica" vs
    # "SL Benfica" / "SL Benfica B"). Without these the ensemble ran the
    # 60%-weight Elo on a 1500 prior for AGF v Benfica, Lyon v Fenerbahce.
    "Lyon": "Olympique Lyonnais",
    "Olympique Lyon": "Olympique Lyonnais",
    "Benfica": "SL Benfica",
    "Hearts": "Heart of Midlothian",
    "Heart of Midlothian FC": "Heart of Midlothian",
    "Copenhagen": "FC Kobenhavn",
    "FC Copenhagen": "FC Kobenhavn",
    "FC København": "FC Kobenhavn",
    "Celtic": "Celtic FC",
    "Rangers": "Rangers FC",
    "Ajax": "AFC Ajax",
    "AFC Ajax": "AFC Ajax",
    "Ferencvaros": "Ferencvárosi TC",
    "Ferencvárosi TC": "Ferencvárosi TC",
    "Anderlecht": "RSC Anderlecht",
    "Thun": "FC Thun Berner Oberland",
    "Brann": "SK Brann",
    "Hibernian": "Hibernian FC",
    "Sporting CP": "Sporting CP",
    "Sporting": "Sporting CP",
    "PSV": "PSV Eindhoven",
    "Feyenoord": "Feyenoord Rotterdam",
    "Birmingham": "Birmingham City",
    "Stuttgart": "VfB Stuttgart",
    "Bayern Munich": "Bayern München",
    "FC Bayern Munich": "Bayern München",
    "FC Internazionale Milano": "Inter Milan",
    "Borussia Mönchengladbach": "Bor. Mönchengladbach",
    "Excelsior Rotterdam": "SBV Excelsior",
    # Paris/Nice 2025-26 - Ligue 1: seed keys are full "OGC Nice" / "Paris FC",
    # but live feeds use short "Nice" / "Paris FC". Single-token "Nice" would be
    # vetoed by the K2 single-token guard (ogc+nice not <= nice) -> 1500 prior.
    "Nice": "OGC Nice",
    "OGC Nice": "OGC Nice",
    # Serie A 2025-26 - seed keys carry generic suffixes (Calcio/CFC/SSc) or
    # founding years (1907/1913) or double identity (Inter+Milan, Lazio+Roma,
    # Hellas+Verona). Single-token LIVE names ("Cagliari","Inter","Genoa",...)
    # vetoed by K2 guard before 2026-08-31 fix -> 1500 prior for both sides
    # (Cagliari v Inter, Lazio v Genoa). Explicit alias restores 1-token hit.
    "Cagliari": "Cagliari Calcio",
    "Inter": "Inter Milan",
    "Internazionale": "Inter Milan",
    "Genoa": "Genoa CFC",
    "Lazio": "Lazio Roma",
    "Napoli": "SSC Napoli",
    "Verona": "Hellas Verona",
    "Hellas Verona": "Hellas Verona",
    "Fiorentina": "ACF Fiorentina",
    "Parma": "Parma Calcio 1913",
    "Como": "Como 1907",
    "Sassuolo": "Sassuolo Calcio",
    "Udinese": "Udinese Calcio",
    "Frosinone": "Frosinone Calcio",
    # Ligue 1 single-token short that slipped through K2 (Brest/Angers SCO)
    "Brest": "Stade Brestois 29",
    "Angers": "Angers SCO",
    # LaLiga - Athletic Bilbao short forms (provider "Athletic Club", user "Athletic Bilbao"/"Ath Bilbao")
    # seed "Athletic Club" 1958: query ["athletic","bilbao"] ambigu 10*Athletic* -> None without alias
    "Athletic Bilbao": "Athletic Club",
    "Ath Bilbao": "Athletic Club",
    "Bilbao": "Athletic Club",
    "Celta Vigo": "RC Celta",
    "Celta": "RC Celta",
}

# K2: tokens that mark a SECOND team of the same club (reserve / B side /
# youth / women). Kept separate from ``_CLUB_TOKEN_PREFIXES`` because they
# DO carry identity -- "SL Benfica B" is not "SL Benfica" -- but a query
# without the marker must prefer the first team when both keys tie.
_SECOND_TEAM_TOKENS = {
    "b", "ii", "iii", "reserves", "reserve", "u19", "u21", "u23",
    "women", "youth", "amateur", "amateurs",
}

# Club-prefix / suffix tokens that carry no identity ("Arsenal FC" vs "Arsenal").
_CLUB_TOKEN_PREFIXES = {
    "fc", "cf", "ac", "sc", "afc", "fk", "nk", "cd", "pfc", "ifk",
    "rc", "ca", "ec", "cr", "se", "ud", "as", "bsc", "tsg", "sb",
    "kc", "sk", "de", "sv", "ss", "1.",
    "calcio",
}


def _norm(name: str) -> str:
    """Lowercase + strip accents/diacritics ("Bodø" -> "bodo")."""
    s = unicodedata.normalize("NFD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


def _significant_tokens(name: str) -> list[str]:
    """Lowercased identity tokens, dropping club prefixes and 1-2 char noise."""
    tokens = re.findall(r"[a-z0-9]+", _norm(name))
    return [t for t in tokens if t not in _CLUB_TOKEN_PREFIXES and len(t) >= 3]


def _is_second_team(name: str) -> bool:
    """True when the RAW key names a reserve/B/youth/women side."""
    raw_tokens = re.findall(r"[a-z0-9]+", _norm(name))
    return any(t in _SECOND_TEAM_TOKENS for t in raw_tokens)


def _expand_synonyms(tokens: list[str]) -> set[str]:
    out = set(tokens)
    for t in tokens:
        alt = _TOKEN_SYNONYMS.get(t)
        if alt:
            out.add(alt)
    return out


# Spelling variants that carry the same identity ("Manchester United" vs the
# football-data.co.uk seed "Manchester Utd", "Man City" vs "Manchester City").
# Token scoring treats each pair as equivalent so live provider spellings hit
# the seeded ratings instead of falling back to the 1500 prior.
_TOKEN_SYNONYMS: dict[str, str] = {
    "united": "utd",
    "utd": "united",
    "man": "manchester",
    "wolves": "wolverhampton",
    "wolverhampton": "wolves",
    "atl": "atletico",
    "atletico": "atl",
}


def _token_score(tokens: list[str], nkey: str) -> int:
    """How many query tokens appear in a normalized key, honoring synonyms."""
    score = 0
    for t in tokens:
        if re.search(rf"\b{re.escape(t)}\b", nkey):
            score += 1
        else:
            alt = _TOKEN_SYNONYMS.get(t)
            if alt and re.search(rf"\b{re.escape(alt)}\b", nkey):
                score += 1
    return score

def _goals_from_result(r: dict[str, Any]) -> tuple[int | None, int | None]:
    """Parse (home_goals, away_goals) from a result row.

    Accepts explicit ``home_goals``/``away_goals`` keys or a ``result``
    string like "2-1" / "2:1". Returns (None, None) when unparseable.
    """
    hg, ag = r.get("home_goals"), r.get("away_goals")
    if hg is not None and ag is not None:
        try:
            return int(hg), int(ag)
        except (TypeError, ValueError):
            return None, None
    m = re.match(r"^(\d{1,2})\s*[-:]\s*(\d{1,2})$", str(r.get("result") or "").strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


class EloModel:
    def __init__(
        self,
        k: float = 32.0,
        home_advantage: float = 65.0,
        initial_rating: float = 1500.0,
        base_total_goals: float = 2.7,
        path: str | Path | None = None,
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.base_total_goals = base_total_goals
        self.path = Path(path) if path else None
        self.ratings: dict[str, float] = {}
        self.games: dict[str, int] = {}
        self._norm_index: dict[str, str] = {}
        self._alias_index: dict[str, str] = {}
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.ratings = {str(k): float(v) for k, v in payload.get("ratings", {}).items()}
            self.games = {str(k): int(v) for k, v in payload.get("games", {}).items()}
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            # Corrupt file: start empty rather than crash the bot.
            self.ratings = {}
            self.games = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """(Re)build the lookup indexes from the current ratings.

        ``_norm_index`` maps normalized spelling -> seed key. ``_alias_index``
        maps canonical provider spellings (teams.json) -> seed key so names
        like "FC Internazionale Milano" hit the "Inter" rating. Two mapping
        directions are used: when a canonical's tokens are contained in a
        seed key ("Arsenal FC" -> "Arsenal") and when a seed key's tokens are
        contained in a canonical ("Newcastle" -> "Newcastle United FC").
        """
        self._norm_index = {_norm(k): k for k in self.ratings}
        self._alias_index = {}
        try:
            teams = load_teams()
        except Exception:  # noqa: BLE001 -- teams.json issues must not crash load
            teams = {}
        # Precompute canonical spellings once (not per seed key): norm of the
        # alias, norm of the canonical, and the canonical's significant tokens.
        entries = [
            (
                _norm(alias),
                _norm(canonical),
                set(_significant_tokens(canonical)),
            )
            for league in teams.values()
            for alias, canonical in league.items()
        ]
        # Build alias_index by picking the MOST SPECIFIC seed for each canon
        # (previous setdefault kept first in file order -> Juventus FC -> YF Zurich
        # and RCD Espanyol -> FC Barcelona). Now choose minimal extra tokens.
        best: dict[str, tuple[str, int, int]] = {}  # canon_norm -> (seed_key, extra, -games)
        for seed_key in self.ratings:
            seed_norm = _norm(seed_key)
            seed_tokens = set(_significant_tokens(seed_key))
            seed_games = int(self.games.get(seed_key, 0))
            for alias_norm, canon_norm, canon_tokens in entries:
                matched = False
                extra = 10**9
                if alias_norm == seed_norm:
                    matched = True
                    extra = -1  # exact alias -> best possible
                elif canon_tokens and canon_tokens <= seed_tokens:
                    matched = True
                    extra = len(seed_tokens - canon_tokens)
                elif seed_tokens and seed_tokens <= canon_tokens:
                    matched = True
                    extra = len(canon_tokens - seed_tokens)
                if not matched:
                    continue
                # keep most specific (smallest extra), tie -> most games
                cur = best.get(canon_norm)
                if cur is None or extra < cur[1] or (extra == cur[1] and seed_games > -cur[2]):
                    best[canon_norm] = (seed_key, extra, -seed_games)
        self._alias_index = {k: v[0] for k, v in best.items()}
        # Built-in abbreviated-spelling aliases -> seed keys that exist.
        for raw, seed_key in _EXTRA_ALIASES.items():
            if seed_key in self.ratings:
                self._alias_index.setdefault(_norm(raw), seed_key)

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"ratings": self.ratings, "games": self.games},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ---- accessors -----------------------------------------------------
    def resolve(self, name: str) -> str | None:
        """Map a live provider name to the seeded rating key, or None.

        Exact (case/diacritic-insensitive) match first; then the teams.json
        canonical alias index; then partial token scoring (a query like
        "Royale Union Saint-Gilloise" matches "Union Saint-Gilloise" even
        though "royale" is absent from the seed). Ties prefer the key with
        the most games. Returns None only when nothing plausibly matches.
        """
        if not name:
            return None
        n = _norm(name)
        if n in self._norm_index:
            return self._norm_index[n]
        if n in self._alias_index:
            return self._alias_index[n]
        tokens = _significant_tokens(name)
        if not tokens:
            return None
        query_tokens = set(tokens)
        query_expanded = _expand_synonyms(tokens)
        best_score = 0
        candidates: list[str] = []
        for nkey, key in self._norm_index.items():
            score = _token_score(tokens, nkey)
            if score > best_score:
                best_score = score
                candidates = [key]
            elif score == best_score and score > 0:
                candidates.append(key)
        if not candidates:
            return None
        # K2 (post-mortem 2026-08-28): a SINGLE-token query ("Hearts",
        # "Union", "Benfica") must not partial-match a seed key that carries
        # EXTRA identity tokens -- "Hearts" hit "Kelty Hearts" (Scottish
        # tier-4, rating 1031) and drove Rapid Wien v Hearts to a HIGH Over
        # 2.5 built on the wrong club. A short live name maps to a longer
        # seed key only through the alias index (handled above); the fuzzy
        # pass may only accept keys whose identity tokens are all in the
        # query. Multi-token queries keep the partial rule ("Royale Union
        # Saint-Gilloise" -> "Union Saint-Gilloise").
        if len(tokens) == 1:
            candidates = [
                key for key in candidates
                if set(_significant_tokens(key)) <= query_expanded
            ]
            if not candidates:
                return None
        if len(candidates) == 1:
            return candidates[0]
        # Tie: prefer the candidate whose ENTIRE token set is contained in the
        # query (its seeded name is a strict part of the query name) -- strong
        # identity evidence. Otherwise the match is ambiguous: return None so
        # the bot honestly reports the team as unseeded instead of guessing.
        contained = [
            key for key in candidates
            if set(_significant_tokens(key)) <= query_tokens
        ]
        if len(contained) == 1:
            return contained[0]
        # K2: "SL Benfica" vs "SL Benfica B" tie on identical significant
        # tokens ("b" is noise). A query WITHOUT a second-team marker means
        # the first team; drop the marked keys and accept a unique survivor.
        if len(contained) > 1 and not _is_second_team(name):
            firsts = [key for key in contained if not _is_second_team(key)]
            if len(firsts) == 1:
                return firsts[0]
        return None

    def rating(self, team: str) -> float:
        key = self.resolve(team)
        return self.ratings.get(key, self.initial_rating) if key else self.initial_rating

    def games_played(self, team: str) -> int:
        key = self.resolve(team)
        return self.games.get(key, 0) if key else 0

    def known(self, home: str, away: str) -> bool:
        return self.resolve(home) is not None and self.resolve(away) is not None

    def snapshot(self) -> dict[str, Any]:
        return {"ratings": self.ratings, "games": self.games}

    # ---- prediction ----------------------------------------------------
    def expected_lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Map rating difference (with home advantage) to expected goals.

        share = logistic of the rating gap; total expected goals is split by
        that share. Clamped to a sane football range.
        """
        d = self.rating(home) + self.home_advantage - self.rating(away)
        share = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        total = self.base_total_goals
        lh = max(0.2, min(3.8, total * share))
        la = max(0.2, min(3.8, total * (1.0 - share)))
        return lh, la

    # ---- update (post-match only) --------------------------------------
    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        persist: bool = True,
        k_multiplier: float = 1.0,
    ) -> None:
        """Move ratings after a finished match. ``persist=False`` for bulk replay.

        ``k_multiplier`` (0..2, default 1.0) scales K for this single match
        (experimental recency weighting: recent matches move ratings more).
        """
        if home_goals == away_goals:
            result = 0.5
        elif home_goals > away_goals:
            result = 1.0
        else:
            result = 0.0
        # Resolve live spellings to seeded keys so updates land on the same
        # key the backtest trained (and the norm index stays consistent).
        home = self.resolve(home) or home
        away = self.resolve(away) or away
        rh, ra = self.rating(home), self.rating(away)
        eh = 1.0 / (1.0 + 10.0 ** (-(rh + self.home_advantage - ra) / 400.0))
        ea = 1.0 - eh
        # K shrinks as a team accrues games (standard Elo practice).
        km = max(0.0, min(2.0, float(k_multiplier)))
        kh = self.k / (1.0 + 0.1 * math.sqrt(self.games_played(home))) * km
        ka = self.k / (1.0 + 0.1 * math.sqrt(self.games_played(away))) * km
        self.ratings[home] = rh + kh * (result - eh)
        self.ratings[away] = ra + ka * ((1.0 - result) - ea)
        self.games[home] = self.games.get(home, 0) + 1
        self.games[away] = self.games.get(away, 0) + 1
        # Keep the lookup index in sync so freshly-updated teams are findable
        # (a team not in the seed must still be readable back right after).
        self._norm_index[_norm(home)] = home
        self._norm_index[_norm(away)] = away
        if persist:
            self._save()

    def update_from_results(self, results: list[dict[str, Any]]) -> int:
        """Apply a batch of finished results and persist ONCE (TODO-01).

        ``results`` items carry either {home, away, home_goals, away_goals}
        or {home, away, result: "2-1"}. Only entries with parseable goals are
        applied; the rest are skipped (never guessed). Returns the number of
        applied matches.

        Leakage rule: call ONLY after the results are final and BEFORE any
        prediction for a match these results belong to is built. The live
        settle path is the only production caller. Results are applied in
        kickoff order (when kickoff times are known) so the live ratings
        advance exactly like the walk-forward validation does -- Elo updates
        are order-dependent.
        """
        from .timeutil import kickoff_sort_key

        applied = 0
        for r in sorted(results or [], key=kickoff_sort_key):
            hg, ag = _goals_from_result(r)
            if hg is None or ag is None:
                continue
            home, away = (r.get("home") or ""), (r.get("away") or "")
            if not home or not away:
                continue
            self.update(home, away, hg, ag, persist=False)
            applied += 1
        if applied:
            self._save()
        return applied
