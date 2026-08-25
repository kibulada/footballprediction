"""Canonical team identity + (provider, provider_id) -> canonical_id registry.

G1 (2026-08-17): the audit found the system resolves teams by NAME everywhere
(teams.json alias -> canonical name) and stores per-provider ids (flashscore
slug, football-data int, thesportsdb idTeam) with no mapping between them --
a wrong-club resolve by any provider silently feeds another club's data into
the model. This module adds the missing layer:

- ``canonical_team_id`` -- a DETERMINISTIC canonical id derivable from
  (league_key, name): ``t:{league}:{slug(canonical_name)}``. No storage
  needed to recompute it; the canonical name comes from teams.json via
  ``_canonical_team_name`` (prediction_log), so every provider spelling of
  the same club maps to the same id.
- ``EntityRegistry`` -- persisted ``{provider: {provider_id: entry}}`` so a
  provider id seen today resolves to the same canonical id tomorrow, and
  CONFLICTS (one provider id registered under two different canonical ids)
  are surfaced as a guard instead of silently picking one.

The registry is additive only: it never changes an existing resolution path,
and every failure degrades to "no mapping" (the caller falls back to the
existing name matching).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = ROOT / "cache" / "football" / "entity_registry.json"


def _slug(name: str) -> str:
    """Lowercase, punctuation-stripped, hyphen-joined identifier."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "unknown"


def canonical_team_id(league_key: str | None, name: str) -> str | None:
    """Deterministic canonical team id: ``t:{league}:{slug(canonical_name)}``.

    The canonical name is resolved via ``_canonical_team_name`` (teams.json
    first, deterministic suffix-strip fallback), so "Espanyol" and "RCD
    Espanyol de Barcelona" both produce the same id -- the G2 fix for
    duplicated match_ids. Returns None only when the name is empty.
    """
    from .prediction_log import _canonical_team_name  # lazy: avoid cycles

    cn = _canonical_team_name(name, league_key or None)
    if not cn:
        return None
    league = re.sub(r"[^a-z0-9]+", "-", (league_key or "unknown").lower()).strip("-") or "unknown"
    return f"t:{league}:{_slug(cn)}"


class EntityRegistry:
    """Persisted ``{provider: {provider_id: entry}}`` mapping.

    Entry shape: {canonical_id, canonical_name, league_key, name} where
    ``name`` is the provider's own spelling at registration time (audit
    trail) and ``canonical_name`` is the teams.json-resolved form.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or DEFAULT_REGISTRY_PATH)
        self._entries: dict[str, dict[str, dict[str, Any]]] = {}
        self._load()

    # ---- persistence ----------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = data.get("entries") or {}
        except (OSError, json.JSONDecodeError, ValueError):
            self._entries = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"entries": self._entries}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ---- registry API ---------------------------------------------------
    def register(
        self,
        provider: str,
        provider_id: Any,
        league_key: str | None,
        name: str,
        sport: str | None = None,
    ) -> str | None:
        """Record (provider, provider_id) -> canonical id; return the id.

        Idempotent: re-registering the same provider id with the same
        canonical id keeps the first entry (no rewrite). Registering the
        same provider id with a DIFFERENT canonical id is a conflict: the
        newest entry wins (last observation) and the old one is kept under
        ``_conflicts`` for audit -- the mapping must never guess.

        ``sport`` (optional, e.g. "football"/"basketball") is recorded when
        supplied. Backward-compat: existing entries without ``sport`` keep
        their shape; missing field on read defaults to ``"unknown"`` so
        the resolver never breaks older cache files.
        """
        cid = canonical_team_id(league_key, name)
        if not cid or provider_id is None:
            return cid
        provider = str(provider or "unknown")
        pid = str(provider_id)
        bucket = self._entries.setdefault(provider, {})
        existing = bucket.get(pid)
        if existing is None:
            entry: dict[str, Any] = {
                "canonical_id": cid,
                "canonical_name": _canonical_name(name, league_key),
                "league_key": league_key,
                "name": name,
            }
            if sport:
                entry["sport"] = sport
            bucket[pid] = entry
            self.save()
            return cid
        if existing.get("canonical_id") != cid:
            self._conflicts.append(
                {
                    "provider": provider,
                    "provider_id": pid,
                    "previous_canonical_id": existing.get("canonical_id"),
                    "new_canonical_id": cid,
                    "previous_name": existing.get("name"),
                    "new_name": name,
                }
            )
            entry = {
                "canonical_id": cid,
                "canonical_name": _canonical_name(name, league_key),
                "league_key": league_key,
                "name": name,
            }
            if sport:
                entry["sport"] = sport
            bucket[pid] = entry
            self.save()
        elif sport and "sport" not in existing:
            # backfill sport tag on existing entry; idempotent
            existing["sport"] = sport
            self.save()
        return cid

    def lookup(self, provider: str, provider_id: Any) -> dict[str, Any] | None:
        if provider_id is None:
            return None
        return (self._entries.get(str(provider or "unknown")) or {}).get(str(provider_id))

    def resolve(self, provider: str, provider_id: Any) -> str | None:
        entry = self.lookup(provider, provider_id)
        return (entry or {}).get("canonical_id")

    def sport(self, provider: str, provider_id: Any) -> str:
        """Return the registered sport for (provider, provider_id).

        Always returns a string ("unknown" when the field is absent or the
        provider id is not registered). Never raises; callers can use this
        to filter football vs basketball entries safely.
        """
        entry = self.lookup(provider, provider_id) or {}
        return str(entry.get("sport") or "unknown")

    def conflicts(self) -> list[dict[str, Any]]:
        """Provider ids observed under more than one canonical id."""
        return list(self._conflicts)

    # ---- module state ---------------------------------------------------
    _conflicts: list[dict[str, Any]] = []


def _canonical_name(name: str, league_key: str | None) -> str:
    from .prediction_log import _canonical_team_name  # lazy: avoid cycles

    return _canonical_team_name(name, league_key or None) or str(name or "")


# Module-level singleton: cheap (lazy file load), shared by callers that do
# not want to thread a registry instance through the pipeline.
_REGISTRY: EntityRegistry | None = None


def registry() -> EntityRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = EntityRegistry()
    return _REGISTRY
