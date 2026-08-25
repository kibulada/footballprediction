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

# Club-prefix / suffix tokens that carry no identity ("Arsenal FC" vs "Arsenal").
_CLUB_TOKEN_PREFIXES = {
    "fc", "cf", "ac", "sc", "afc", "fk", "nk", "cd", "pfc", "ifk",
    "rc", "ca", "ec", "cr", "se", "ud", "as", "bsc", "tsg", "sb",
    "kc", "sk", "de", "sv", "ss", "1.",
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
        for seed_key in self.ratings:
            seed_norm = _norm(seed_key)
            seed_tokens = set(_significant_tokens(seed_key))
            for alias_norm, canon_norm, canon_tokens in entries:
                if alias_norm == seed_norm:
                    # INTER -> FC Internazionale Milano  (seed "Inter")
                    self._alias_index.setdefault(canon_norm, seed_key)
                    continue
                if canon_tokens and canon_tokens <= seed_tokens:
                    # Arsenal FC -> Arsenal, Celtic FC -> Celtic FC
                    self._alias_index.setdefault(canon_norm, seed_key)
                    continue
                if seed_tokens and seed_tokens <= canon_tokens:
                    # Newcastle -> Newcastle United FC (reverse direction):
                    # the seeded short name is a strict part of the canonical
                    # provider name, so the full live spelling maps back.
                    self._alias_index.setdefault(canon_norm, seed_key)

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
    ) -> None:
        """Move ratings after a finished match. ``persist=False`` for bulk replay."""
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
        kh = self.k / (1.0 + 0.1 * math.sqrt(self.games_played(home)))
        ka = self.k / (1.0 + 0.1 * math.sqrt(self.games_played(away)))
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
