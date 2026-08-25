"""Competition pagination for the "Kompetisi lain" section (TODO-17).

Extracted verbatim from ``format.py`` -- pure structural decomposition, the
formatter re-imports ``build_top_pages`` so behavior and output are
byte-identical.
"""
from __future__ import annotations

from typing import Any

from .format_utils import _fmt_value_date
from .league_resolver import competition_league_key

# Competition pagination for the "Kompetisi lain" section.
TOP_COMPS_PER_PAGE = 10
TOP_PAGE_CHAR_BUDGET = 1750  # body budget so page+footer fit ONE Discord message


def _group_competitions(extra_matches: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group extra matches by competition; sort by count desc, then name asc
    (stable, deterministic)."""
    from collections import OrderedDict

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for m in extra_matches:
        grouped.setdefault(str(m.get("competition") or "Other"), []).append(m)
    return sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))


def _competition_block(name: str, ms: list[dict[str, Any]], tag: str | None = None) -> str:
    """Render one competition block: header + every match (paginated by budget).

    ``tag`` is the registered league key (e.g. 'UCL') when the competition is
    analyzable; it tells the user exactly which league to use in
    `analisa match <liga> ...`. Large blocks are split across pages by
    ``_pack_competition_pages``.
    """
    header = f"🏆 **{name} · {len(ms)} match**"
    if tag:
        header += f" ({tag})"
    lines = [header]
    for m in ms:
        lines.append(f"• {m.get('home', '?')} vs {m.get('away', '?')}")
    return "\n".join(lines)


def _pack_competition_pages(
    comps: list[tuple[str, list[dict[str, Any]]]],
    per_page: int = TOP_COMPS_PER_PAGE,
    budget: int = TOP_PAGE_CHAR_BUDGET,
) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    """Pack competitions into pages: <= per_page comps and <= budget chars so
    each page (incl. footer) fits one Discord message (no truncation)."""
    pages: list[list[tuple[str, list[dict[str, Any]]]]] = []
    cur: list[tuple[str, list[dict[str, Any]]]] = []
    cur_chars = 0
    for name, ms in comps:
        cost = len(_competition_block(name, ms)) + 2
        if cur and (len(cur) >= per_page or cur_chars + cost > budget):
            pages.append(cur)
            cur, cur_chars = [], 0
        cur.append((name, ms))
        cur_chars += cost
    if cur:
        pages.append(cur)
    return pages


def _analyzable_competitions(
    extra_matches: list[dict[str, Any]],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Group extra matches by competition, KEEPING only competitions that map
    to a registered league key (analyzable via `analisa match <liga> ...`).

    Friendlies, minor cups and other non-registered competitions are dropped
    from the rendered list so every listed match can actually be analyzed and
    receive a prediction. Returns [(competition, league_key, matches)] sorted
    by count desc, then name asc (stable).
    """
    out: list[tuple[str, str, list[dict[str, Any]]]] = []
    for name, ms in _group_competitions(extra_matches):
        key = competition_league_key(name)
        if key:
            out.append((name, key, ms))
    return out


def _non_analyzable_competitions(
    extra_matches: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group extra matches by competition, keeping ONLY competitions that do
    NOT map to a registered league key (friendlies, minor cups, small
    leagues). Listed info-only: they cannot be analyzed end-to-end, but the
    user still sees every fixture instead of a hidden count. Returns
    [(competition, matches)] sorted by count desc, then name asc (stable).
    """
    return [
        (name, ms)
        for name, ms in _group_competitions(extra_matches)
        if not competition_league_key(name)
    ]


def build_top_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Paginasi KOMPETISI LAIN untuk kasus tanpa match di filter utama.

    Every competition on the flashscore homepage is listed: registered ones
    are tagged with their league key (analyzable), the rest appear under a
    separate info-only section. Returns page dicts (title/body/footer); each
    page stays within one plain Discord message. Empty list when there is
    nothing to show.
    """
    extra = payload.get("extra_matches") or []
    comps = _analyzable_competitions(extra)
    info_comps = _non_analyzable_competitions(extra)
    if not comps and not info_comps:
        return []
    date = payload.get("date", "?")
    total_match = sum(len(ms) for _, _, ms in comps) + sum(
        len(ms) for _, ms in info_comps
    )
    n_comp = len(comps) + len(info_comps)
    tags = {name: key for name, key, _ in comps}
    packed = _pack_competition_pages(
        [(n, ms) for n, _k, ms in comps] + list(info_comps)
    )
    n_pages = len(packed)
    header = (
        f"🎯 **VALUE MATCH — {_fmt_value_date(date)}**\n\n"
        "❌ Tidak ada match ditemukan pada periode & liga tersebut.\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📋 **KOMPETISI LAIN**\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )
    pages: list[dict[str, Any]] = []
    for i, chunk in enumerate(packed, 1):
        blocks = [header]
        seen_analyzable = False
        seen_info = False
        for n, ms in chunk:
            if n in tags:
                blocks.append(_competition_block(n, ms, tag=tags[n]))
                seen_analyzable = True
            else:
                if not seen_info:
                    blocks.append(
                        "\n🔍 **BELUM TERDAFTAR (info saja)**"
                        " — belum bisa dianalisa, tanpa odds/form model"
                    )
                    seen_info = True
                blocks.append(_competition_block(n, ms))
        body = "\n\n".join(blocks)
        stats_parts = []
        if comps:
            n_an = sum(len(ms) for _, _, ms in comps)
            stats_parts.append(f"{n_an} bisa dianalisa")
        if info_comps:
            n_info = sum(len(ms) for _, ms in info_comps)
            stats_parts.append(f"{n_info} info saja")
        footer_lines = [
            "━━━━━━━━━━━━━━━━━━━━",
            f"📊 **{total_match} MATCH** • 🏟️ **{n_comp} KOMPETISI** ({', '.join(stats_parts)})",
            "💡 Sumber: Flashscore",
            "💡 Ketik `analisa match <liga> <home> vs <away>` — liga di header tiap kompetisi.",
        ]
        footer_lines.append(f"📄 **Page {i}/{n_pages}**")
        pages.append({"title": "", "body": body, "footer": "\n".join(footer_lines)})
    return pages
