"""Per-league team roster builder (plan 2026-08-24, layer L0).

Answers "harus dilist tiap tim per liga?" with a DETERMINISTIC, offline
generator instead of a hand-typed list:

- ``teams.json`` (team_alias.py) is the authoritative alias -> canonical
  table per league; its VALUES are the canonical roster.
- ``cache/football/entity_registry.json`` (entity_registry.py) accumulates
  (provider, provider_id) observations from every live resolve (G1).

This tool merges both into ``cache/football/rosters.json``::

    {league_key: {"<canonical name>": {
        "canonical_id": "t:{league}:{slug}",
        "aliases": ["...", ...],          # every registered spelling
        "providers": {"flashscore": ["id", ...], "football_data": [...]},
    }}}

and prints a per-league coverage report (roster size, how many clubs carry
provider ids). Zero network access -- safe to run anytime; refresh after a
season turnover (promosi/degradasi) or whenever entity_registry grows.

Usage:
    python -m agents.football.roster_builder            # build + report
    python -m agents.football.roster_builder --report   # report only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = ROOT / "cache" / "football" / "rosters.json"
DEFAULT_REGISTRY = ROOT / "cache" / "football" / "entity_registry.json"


def build_rosters(
    teams: dict[str, dict[str, str]] | None = None,
    registry_entries: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Merge the canonical tables with observed provider ids.

    ``teams`` defaults to the live teams.json; ``registry_entries`` to the
    persisted entity registry ``{provider: {pid: entry}}``. Pure function --
    all I/O stays with the caller (testable without fixtures on disk).
    """
    if teams is None:
        from .team_alias import load_teams

        teams = load_teams()

    # 1) canonical rosters + reverse alias lists per league.
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for league_key, aliases in (teams or {}).items():
        bucket: dict[str, dict[str, Any]] = {}
        for alias, canonical in aliases.items():
            entry = bucket.setdefault(canonical, {"aliases": [], "providers": {}})
            entry["aliases"].append(alias)
        out[league_key] = bucket

    # 2) overlay provider-id observations grouped by (league, canonical).
    by_cid: dict[tuple[str | None, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    names_by_cid: dict[tuple[str | None, str], str] = {}
    for provider, pids in (registry_entries or {}).items():
        for pid, entry in (pids or {}).items():
            cid = entry.get("canonical_id")
            if not cid:
                continue
            key = (entry.get("league_key"), str(cid))
            ids = by_cid[key][str(provider)]
            if pid not in ids:
                ids.append(pid)
            names_by_cid.setdefault(key, entry.get("canonical_name") or "")

    def _cid(league_key: str, canonical: str) -> str:
        from .entity_registry import canonical_team_id

        return canonical_team_id(league_key, canonical) or ""

    for league_key, members in out.items():
        for canonical, entry in members.items():
            cid = _cid(league_key, canonical)
            entry["canonical_id"] = cid
            observed = by_cid.get((league_key, cid)) or {}
            for provider, ids in observed.items():
                merged = entry["providers"].setdefault(provider, [])
                for pid in ids:
                    if pid not in merged:
                        merged.append(pid)
            entry["aliases"].sort()
    return out


def coverage_report(rosters: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Human-readable per-league coverage lines (roster size vs provider ids)."""
    lines: list[str] = []
    for league_key in sorted(rosters):
        members = rosters[league_key]
        total = len(members)
        with_ids = sum(1 for e in members.values() if e.get("providers"))
        providers = sorted({p for e in members.values() for p in (e.get("providers") or {})})
        prov_txt = ",".join(providers) if providers else "-"
        lines.append(
            f"{league_key:<22} roster={total:>3}  dengan_provider_id={with_ids:>3}  "
            f"[{prov_txt}]"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the coverage report without writing the output file",
    )
    args = parser.parse_args(argv)

    try:
        entries = json.loads(Path(args.registry).read_text(encoding="utf-8")).get("entries") or {}
    except (OSError, json.JSONDecodeError, ValueError):
        entries = {}

    rosters = build_rosters(registry_entries=entries)
    print(f"identity L0 roster build: {sum(len(v) for v in rosters.values())} klub kanonik")
    for line in coverage_report(rosters):
        print(line)
    if not args.report:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(rosters, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
