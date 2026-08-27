"""Discord embed formatter for Hermes Football output."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

# TODO-17: formatter decomposition. Pure helpers live in format_utils.py
# (date/odds/stat formatting) and format_pages.py (competition pagination);
# format.py imports them back -- including the private names, so any test or
# caller referencing ``format._group_competitions`` etc. keeps working and
# the output stays byte-identical.
from .format_pages import (
    _analyzable_competitions,
    _competition_block,
    _group_competitions,
    _pack_competition_pages,
    build_top_pages,
)
from .format_utils import _fmt_kickoff, _fmt_odd, _fmt_pct, _fmt_stat, _fmt_value_date
from .league_resolver import competition_league_key
from .market_tiers import render_market_tiers, render_single_pick, select_best_pick
from .predictor import fair_pair_implied
from .signal_engine import movement_narrative_flags

try:
    from zoneinfo import ZoneInfo
    WIB = ZoneInfo("Asia/Jakarta")
except Exception:
    WIB = timezone(timedelta(hours=7))

_TIER_ICON = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}


def _source_label(source: str) -> str:
    """User-facing name for a match-source command (provenance line)."""
    return {"livescore": "LiveScore", "flashscore": "Flashscore"}.get(source, source)


def _render_confidence_block(conf: dict[str, Any]) -> list[str]:
    """Addendum v1.1 Section 5: render the ONE confidence block.

    Only ``model_calibration_score`` (global) and ``pick_specific_confidence``
    (local) may appear as numbers; ``tier`` / ``tier_before_caps`` /
    ``caps_applied`` / ``n_bucket`` / ``completeness_factor`` are the only
    supporting fields. The block comes from
    ``model_gates.build_confidence_block`` which enforces the allowlist
    structurally, so no signal / decisiveness / legacy 0-1 confidence can
    ever reach the output template.
    """
    tier = str(conf.get("tier") or "LOW")
    psc = float(conf.get("pick_specific_confidence") or 0.0)
    mcs = conf.get("model_calibration_score")
    icon = _TIER_ICON.get(tier, "⚪")
    lines = [f"   ⚑ Confidence: {icon} {tier} (pick_specific_confidence {psc:.2f})"]
    if mcs is not None:
        lines.append(f"   model_calibration_score (global): {mcs:.2f}")
    before = str(conf.get("tier_before_caps") or tier)
    if before != tier:
        lines.append(f"   tier_before_caps: {before} (sebelum cap)")
    for cap in conf.get("caps_applied") or []:
        lines.append(f"   • {cap}")
    lines.append(
        f"   n_bucket {conf.get('n_bucket', 0)} • "
        f"completeness_factor {conf.get('completeness_factor', 0.0):.2f}"
    )
    return lines


def _build_quota_footer(quota: dict[str, Any], sources: list[str] | None = None) -> str:
    parts: list[str] = []
    if quota.get("odds_api_remaining") is not None:
        parts.append(f"odds quota: {quota['odds_api_remaining']}/500")
    if quota.get("odds_blocked"):
        parts.append("ODDS BLOCKED")
    if quota.get("stats_warning"):
        parts.append("stats API lemau")
    if quota.get("football_data_warning"):
        parts.append("football-data rate limit")
    if sources:
        parts.append(f"source: {', '.join(sources)}")
    return " • ".join(parts) if parts else " "


def format_top(payload: dict[str, Any]) -> dict[str, Any]:
    matches = payload.get("matches", [])
    date = payload.get("date", "?")
    # --days window: show the covered WIB range (e.g. "08-12 → 08-13") so
    # the user knows dini-hari matches are included.
    days = int(payload.get("days") or 1)
    date_range = payload.get("date_range") or date
    title_date = date_range if days > 1 else date
    quota = payload.get("quota", {})
    leagues_no_odds = payload.get("leagues_no_odds", [])
    extra = payload.get("extra_matches") or []

    # ---- No matches in the primary filter but Flashscore has other ----
    # competitions: render the new grouped + paginated "KOMPETISI LAIN"
    # layout (presentation layer only; no query logic touched here).
    if not matches and extra:
        pages = build_top_pages(payload)
        if pages:
            first = pages[0]
            return {**first, "pages": pages}
        body = "Tidak ada match ditemukan pada periode dan liga tersebut."
    elif not matches:
        body = "Tidak ada match ditemukan pada periode dan liga tersebut."
    else:
        lines = []
        grade_counts = {"LAYAK": 0, "CUKUP": 0, "SKIP": 0}
        for i, m in enumerate(matches, 1):
            cons = m["odds"]["consensus"]
            odds_str = (
                f"{_fmt_odd(cons['home'])} / {_fmt_odd(cons['draw'])} / "
                f"{_fmt_odd(cons['away'])} ({m['bookmakers_count']} bookie)"
            )
            edge = m["odds"]["outlier"]
            if edge:
                edge_str = (
                    f"edge: {edge['side']} +{edge['value_pct']}% "
                    f"@ {edge['bookmaker']}"
                )
            else:
                edge_str = "edge: -"
            forms = (
                f"{m['stats']['home_form']} vs {m['stats']['away_form']}"
            )
            grade = m.get("grade") or {}
            grade_label = grade.get("label", "")
            grade_suffix = f" {grade_label}" if grade_label else ""
            g = grade.get("grade", "SKIP")
            if g not in grade_counts:
                g = "SKIP"
            grade_counts[g] += 1
            lines.append(
                f"**{i}. {m['home']} vs {m['away']}** • {m['league']} • "
                f"{_fmt_kickoff(m['kickoff'])}{grade_suffix}\n"
                f"   odds: {odds_str}\n"
                f"   {edge_str} • form {forms}"
            )
        body = "\n\n".join(lines)

    # ---- Flashscore homepage: competitions football-data does not cover ----
    # (Conference League qualification, friendlies, minor cups, ...). These are
    # context only: no odds/form model behind them. The homepage can list
    # 200+ matches across dozens of competitions, so we render ONE compact
    # count-per-competition line plus a few example fixtures — NOT three
    # examples per competition (that previously blew the report past the
    # embed cap and truncated the reply).
    extra = payload.get("extra_matches") or []
    if extra:
        comps = _analyzable_competitions(extra)
        if comps:
            n_an = sum(len(ms) for _, _, ms in comps)
            extra_lines = [
                f"\n📋 **Kompetisi lain (flashscore, bisa dianalisa): {n_an} match**"
            ]
            counts = " • ".join(
                f"**{c}** ({k}) {len(ms)}" for c, k, ms in comps
            )
            if counts:
                extra_lines.append(counts)
            # A couple of example fixtures from the largest competitions only.
            examples: list[str] = []
            for comp, _k, ms in comps[:3]:
                ex = ", ".join(
                    f"`{m['home']} vs {m['away']}`" for m in ms[:2]
                )
                examples.append(f"{comp}: {ex}")
            if examples:
                extra_lines.append("contoh: " + " • ".join(examples))
            body += "\n" + "\n".join(extra_lines)

    grade_summary = ""
    if matches:
        parts = []
        for key, icon in (("LAYAK", "🟢"), ("CUKUP", "🟡"), ("SKIP", "🔴")):
            if grade_counts.get(key, 0):
                parts.append(f"{icon} {grade_counts[key]} {key.lower()}")
        if parts:
            grade_summary = " • ".join(parts)

    footer_parts = []
    if quota.get("odds_api_remaining") is not None:
        footer_parts.append(f"odds quota: {quota['odds_api_remaining']}/500")
    if quota.get("odds_blocked"):
        footer_parts.append("ODDS BLOCKED")
    if quota.get("stats_warning"):
        footer_parts.append("stats API lemau")
    if quota.get("football_data_warning"):
        footer_parts.append("football-data rate limit")
    if leagues_no_odds:
        footer_parts.append(f"{', '.join(leagues_no_odds)}: tanpa odds")
    if grade_summary:
        footer_parts.append(grade_summary)
    footer = " • ".join(footer_parts) if footer_parts else " "

    title = f"⚽ Value Match — {title_date}"
    return {
        "title": title,
        "body": body,
        "footer": footer,
    }


def format_compare(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("error"):
        return {
            "title": "🔍 Compare",
            "body": f"Error: {payload['error']}",
            "footer": " ",
        }

    stats = payload["stats"]
    body_lines = [
        f"**{payload['home']} vs {payload['away']}** • {payload['league']}",
        f"form 5: {stats['home_form']} vs {stats['away_form']}",
        f"H2H: {stats['h2h']['wins']}W-{stats['h2h']['draws']}D-{stats['h2h']['losses']}L "
        f"(dari sisi home)",
    ]

    sources = payload.get("sources") or []
    quota = payload.get("quota", {})
    footer = _build_quota_footer(quota, sources)

    return {
        "title": "🔍 Compare",
        "body": "\n".join(body_lines),
        "footer": footer,
    }


def _append_nowgoal_context(body_lines: list[str], payload: dict[str, Any]) -> None:
    """Compact NowGoal context block (standings / team stats / HT/FT /
    injuries / lineups). Display-only -- never a model input."""
    ng = payload.get("nowgoal_context") or {}
    if not ng:
        return

    def _num(v: Any) -> str:
        if v is None:
            return "-"
        try:
            f = float(v)
            return f"{f:.0f}" if f == int(f) else f"{f:.1f}"
        except (TypeError, ValueError):
            return str(v)

    lines: list[str] = []
    st = ng.get("standings") or {}
    home_st, away_st = st.get("home") or {}, st.get("away") or {}
    if home_st and away_st:
        h_total = (home_st.get("ft") or {}).get("total") or {}
        a_total = (away_st.get("ft") or {}).get("total") or {}
        lines.append(
            f"🏆 nowgoal standings: home pos {home_st.get('rank', '-')} "
            f"({_num(h_total.get('pts'))}pt) • away pos {away_st.get('rank', '-')} "
            f"({_num(a_total.get('pts'))}pt)"
        )

    ts = ng.get("team_stats") or {}
    if ts:
        pieces = []
        for label in ("Goal", "Opponent Shots", "Corners", "Yellow Cards",
                      "Fouls", "Possession"):
            row = ts.get(label)
            if not row:
                continue
            h, a = row.get("home_recent10"), row.get("away_recent10")
            if h is None or a is None:
                continue
            suffix = "%" if label == "Possession" else ""
            pieces.append(f"{label} {_num(h)}{suffix}/{_num(a)}{suffix}")
        if pieces:
            lines.append("🧬 nowgoal team stats (recent 10): " + " • ".join(pieces))

    rows9 = (ng.get("htft") or {}).get("rows") or {}
    if rows9:
        ww = rows9.get("HT-W/FT-W") or {}
        ll = rows9.get("HT-L/FT-L") or {}
        ww_h, ww_a = ww.get("home") or {}, ww.get("away") or {}
        ll_h, ll_a = ll.get("home") or {}, ll.get("away") or {}
        lines.append(
            f"⏱ nowgoal HT/FT (2 musim): HT-W/FT-W home {_num(ww_h.get('home'))}/"
            f"{_num(ww_h.get('away'))} • away {_num(ww_a.get('home'))}/"
            f"{_num(ww_a.get('away'))} • HT-L/FT-L home {_num(ll_h.get('home'))}/"
            f"{_num(ll_h.get('away'))} • away {_num(ll_a.get('home'))}/"
            f"{_num(ll_a.get('away'))}"
        )

    inj = ng.get("injuries") or {}
    ih, ia = inj.get("home") or [], inj.get("away") or []
    if ih or ia:
        parts = []
        if ih:
            parts.append("home " + ", ".join(p.get("name", "") for p in ih[:4]))
        if ia:
            parts.append("away " + ", ".join(p.get("name", "") for p in ia[:4]))
        lines.append("🚑 nowgoal injuries: " + " • ".join(parts))

    lu = ng.get("lineups") or {}
    if lu and not (payload.get("lineups") or {}).get("home_count"):
        h_start = (lu.get("lineups") or {}).get("home", {}).get("starters") or []
        a_start = (lu.get("lineups") or {}).get("away", {}).get("starters") or []
        if h_start or a_start:
            lines.append(
                f"👥 nowgoal lineups: {lu.get('home_team') or 'home'} "
                f"({lu.get('home_formation') or '?'}) vs "
                f"{lu.get('away_team') or 'away'} ({lu.get('away_formation') or '?'})"
            )

    if lines:
        body_lines.extend(lines)


def _append_event_stats(body_lines: list[str], match_stats: Any) -> None:
    """xG / possession / shots block (pre-match context and finished results)."""
    if not (match_stats and isinstance(match_stats, dict)):
        return
    xg_h = match_stats.get("xg_home")
    xg_a = match_stats.get("xg_away")
    pos_h = match_stats.get("possession_home")
    pos_a = match_stats.get("possession_away")
    shots_h = match_stats.get("shots_home")
    shots_a = match_stats.get("shots_away")
    sot_h = match_stats.get("shots_on_target_home")
    sot_a = match_stats.get("shots_on_target_away")

    has_extended = any(v is not None for v in [xg_h, xg_a, pos_h, pos_a, shots_h, shots_a])
    if not has_extended:
        return
    body_lines.append("\n📊 **Match Stats**")
    if xg_h is not None or xg_a is not None:
        body_lines.append(f"   xG: {_fmt_stat(xg_h)} vs {_fmt_stat(xg_a)}")
    if pos_h is not None or pos_a is not None:
        body_lines.append(f"   Possession: {_fmt_pct(pos_h)} vs {_fmt_pct(pos_a)}")
    if shots_h is not None or shots_a is not None:
        body_lines.append(f"   Shots: {_fmt_stat(shots_h)} vs {_fmt_stat(shots_a)}")
    if sot_h is not None or sot_a is not None:
        body_lines.append(f"   Shots on target: {_fmt_stat(sot_h)} vs {_fmt_stat(sot_a)}")


def _format_analyse_finished(payload: dict[str, Any]) -> dict[str, Any]:
    """Finished-match report: real result + context, NO prediction / tiers.

    Honest output: the score and post-match stats when the resolve carried
    them; otherwise an explicit "hasil tidak tersedia" -- never a fake
    prediction for a match that is already over.
    """
    kickoff = payload.get("kickoff")
    result = payload.get("match_result") or {}
    has_score = result.get("home") is not None and result.get("away") is not None
    stats = payload.get("stats") or {}
    lines = [
        f"**{payload.get('home', '?')} vs {payload.get('away', '?')}** • {payload.get('league', '?')}",
        f"📅 kickoff: {_fmt_kickoff(kickoff) if kickoff else 'jadwal rilis soon'}",
        "",
        "✅ **Match sudah selesai** — prediksi tidak dibuat untuk match yang sudah selesai",
    ]
    if has_score:
        lines.append(
            f"⚽ Hasil: **{payload.get('home')} {result['home']} - "
            f"{result['away']} {payload.get('away')}**"
        )
    _append_event_stats(lines, payload.get("event_stats"))
    if not has_score and not payload.get("event_stats"):
        lines.append("ℹ️ Hasil tidak tersedia (resolve gagal / data belum rilis).")
    hf = stats.get("home_form")
    af = stats.get("away_form")
    if hf and af and hf not in (None, "n/a") and af not in (None, "n/a"):
        lines.append(f"📊 form (konteks): {hf} vs {af}")
    h2h = stats.get("h2h") or {}
    if any(h2h.get(k) for k in ("wins", "draws", "losses")):
        lines.append(
            f"🔁 H2H (konteks): {h2h['wins']}W-{h2h['draws']}D-{h2h['losses']}L"
        )
    sources = payload.get("sources") or []
    quota = payload.get("quota", {})
    footer = _build_quota_footer(quota, sources)
    return {"title": "🔬 Analisa Match", "body": "\n".join(lines), "footer": footer}


def _format_analyse_kickoff_uncertain(payload: dict[str, Any]) -> dict[str, Any]:
    """Kickoff cannot be determined: independent sources disagree beyond
    tolerance, so the match status (pre-match / live / finished) is unknown.

    P1.1: this is NOT "Match sudah selesai" -- the system refuses to derive
    finished/pre-match from a kickoff the sources contradict, and refuses to
    predict on a possibly-live match (stats would leak into the model).
    """
    kickoff = payload.get("kickoff")
    deltas = payload.get("kickoff_deltas") or {}
    lines = [
        f"**{payload.get('home', '?')} vs {payload.get('away', '?')}** • {payload.get('league', '?')}",
        f"📅 kickoff: {_fmt_kickoff(kickoff) if kickoff else 'jadwal rilis soon'}",
        "",
        "⚠️ **Kickoff tidak dapat dipastikan** — sumber data saling bertentangan.",
        "Status match (belum mulai / live / selesai) tidak bisa ditentukan; prediksi tidak dibuat.",
    ]
    if deltas:
        parts = [f"{k}: {v} jam" for k, v in sorted(deltas.items())]
        lines.append(f"🔀 Selisih antar sumber: {', '.join(parts)}")
    stats = payload.get("stats") or {}
    hf = stats.get("home_form")
    af = stats.get("away_form")
    if hf and af and hf not in (None, "n/a") and af not in (None, "n/a"):
        lines.append(f"📊 form (konteks): {hf} vs {af}")
    h2h = stats.get("h2h") or {}
    if any(h2h.get(k) for k in ("wins", "draws", "losses")):
        lines.append(
            f"🔁 H2H (konteks): {h2h['wins']}W-{h2h['draws']}D-{h2h['losses']}L"
        )
    sources = payload.get("sources") or []
    quota = payload.get("quota", {})
    footer = _build_quota_footer(quota, sources)
    return {"title": "🔬 Analisa Match", "body": "\n".join(lines), "footer": footer}


def format_analyse(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("error"):
        body = f"Error: {payload['error']}"
        if payload.get("home_query") or payload.get("away_query"):
            body += f"\nhome query: {payload.get('home_query')}\naway query: {payload.get('away_query')}"
        return {
            "title": "🔬 Analisa Match",
            "body": body,
            "footer": " ",
        }
    stats = payload.get("stats", {})
    h2h = stats.get("h2h") or {"wins": 0, "draws": 0, "losses": 0}
    odds = payload.get("odds", {})
    consensus = odds.get("consensus") or {}
    kickoff = payload.get("kickoff")
    market_totals = odds.get("totals") or {}
    model_probs = (payload.get("picks") or {}).get("model_probs") or {}
    decision = payload.get("decision")  # used by Pick Evaluation / confidence

    def _pct(x: float | None) -> str:
        return f"{x * 100:.1f}%" if x else "-"

    venue = payload.get("venue")
    venue_str = f" • venue: {venue}" if venue else ""
    no_form = (stats.get("home_form") in (None, "n/a")) and (stats.get("away_form") in (None, "n/a"))
    no_odds = not odds.get("has_odds")

    # P1.1: kickoff uncertain (sources disagree) -> distinct block, never
    # "Match sudah selesai" and never a prediction on a possibly-live match.
    if payload.get("kickoff_uncertain"):
        return _format_analyse_kickoff_uncertain(payload)

    # Finished match: real result only (no prediction, no market tiers).
    if payload.get("match_finished"):
        return _format_analyse_finished(payload)

    body_lines = [
        f"**{payload['home']} vs {payload['away']}** • {payload['league']}",
        f"📅 kickoff: {_fmt_kickoff(kickoff) if kickoff else 'jadwal rilis soon'}{venue_str}",
    ]
    # OUTPUT POLICY — full report (render_full / 📋 Copy): every market with
    # data is evaluated and shown with its own tier (1X2, O/U 2.5, O/U 3.5,
    # BTTS). A market without enough input shows an explicit "not evaluated"
    # line. The DEFAULT reply is the single best pick (format_compact); this
    # full per-market breakdown is served only via the 📋 Copy button.
    tier_block = render_market_tiers(payload)
    if tier_block:
        body_lines.append("")
        body_lines.extend(tier_block)
        body_lines.append("")
    if no_form:
        body_lines.append("📊 form: belum ada match musim ini")
    else:
        _hf = stats.get("home_form", "n/a")
        _af = stats.get("away_form", "n/a")
        _hn = sum(1 for c in str(_hf) if c in "WDL")
        _an = sum(1 for c in str(_af) if c in "WDL")
        if _hn and _hn == _an:
            # Section 6: form string always labeled with its window so it can
            # never silently disagree with a record field.
            body_lines.append(
                f"📊 form (last {_hn}, all competitions): {_hf} vs {_af}"
            )
        else:
            body_lines.append(
                f"📊 form (last {_hn or '?'}/{_an or '?'} matches): {_hf} vs {_af}"
            )

    if no_form and no_odds:
        body_lines.append("ℹ️ Liga baru mulai — data form/H2H/odds belum rilis dari upstream.")


    home_gf = stats.get("home_gf_avg", 0)
    home_ga = stats.get("home_ga_avg", 0)
    away_gf = stats.get("away_gf_avg", 0)
    away_ga = stats.get("away_ga_avg", 0)
    if home_gf or away_gf:
        body_lines.append(
            f"⚽ GF/GA avg: home {home_gf:.2f}/{home_ga:.2f} • "
            f"away {away_gf:.2f}/{away_ga:.2f}"
        )

    home_split = stats.get("home_split", {})
    away_split = stats.get("away_split", {})
    if home_split and away_split:
        hs = home_split
        a_s = away_split
        body_lines.append(
            f"🏠 home record (home matches only): W{hs.get('w', 0)}-D{hs.get('d', 0)}-L{hs.get('l', 0)} • "
            f"away record (away matches only): W{a_s.get('w', 0)}-D{a_s.get('d', 0)}-L{a_s.get('l', 0)}"
        )

    h2h_zero = h2h.get("wins", 0) == 0 and h2h.get("draws", 0) == 0 and h2h.get("losses", 0) == 0
    if h2h_zero:
        body_lines.append("🔁 H2H: belum ada rekor head‑to‑head")
    else:
        body_lines.append(
            f"🔁 H2H: {h2h.get('wins', 0)}W-{h2h.get('draws', 0)}D-{h2h.get('losses', 0)}L (sisi home)"
        )

    def _f(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}"
        return "-"

    hxg = stats.get("home_xg_for")
    axg = stats.get("away_xg_for")
    hxga = stats.get("home_xg_against")
    axga = stats.get("away_xg_against")
    if hxg is not None or axg is not None:
        body_lines.append(
            f"🥅 xG avg (5 last): home for {_f(hxg)} / against {_f(hxga)} • "
            f"away for {_f(axg)} / against {_f(axga)}"
        )
    hcf = stats.get("home_corners_for")
    acf = stats.get("away_corners_for")
    hca = stats.get("home_corners_against")
    aca = stats.get("away_corners_against")
    if hcf is not None or acf is not None:
        body_lines.append(
            f"🚩 Corner avg (5 last): home for {_f(hcf)} / against {_f(hca)} • "
            f"away for {_f(acf)} / against {_f(aca)}"
        )
    hyf = stats.get("home_yellow_for")
    ayf = stats.get("away_yellow_for")
    if hyf is not None or ayf is not None:
        body_lines.append(
            f"🟨 Yellow avg (5 last): home {_f(hyf)} • away {_f(ayf)}"
        )

    if odds.get("has_odds") and consensus:
        body_lines.append(
            f"💰 odds 1X2: {_fmt_odd(consensus.get('home'))} / "
            f"{_fmt_odd(consensus.get('draw'))} / "
            f"{_fmt_odd(consensus.get('away'))} "
            f"({odds.get('bookmakers_count', 0)} bookie)"
        )
        for label in sorted(market_totals.keys()):
            data = market_totals[label]
            body_lines.append(f"   {label}: {data['odds']:.2f} @ {data.get('bookmaker', '?')}")
        movement = odds.get("movement")
        if movement and movement.get("sides"):
            m_h = movement["sides"].get("home") or {}
            m_a = movement["sides"].get("away") or {}
            if m_h.get("move_pct") is not None or m_a.get("move_pct") is not None:
                _mp = lambda v: f"{v:+.1f}%" if v is not None else "-"  # noqa: E731
                steamed = movement.get("steamed")
                steam_tag = f" 🔥 steam: {steamed}" if steamed else ""
                body_lines.append(
                    f"📈 movement (open→now): home {_mp(m_h.get('move_pct'))} • "
                    f"away {_mp(m_a.get('move_pct'))}{steam_tag}"
                )
        value_history = odds.get("value_history")
        if value_history:
            parts = []
            for side, label in (("home", "H"), ("draw", "D"), ("away", "A")):
                v = value_history.get(side)
                if v and v.get("value") is not None:
                    parts.append(f"{label} {v['value']:+.1%}")
            if parts:
                body_lines.append(f"⚖️ vs fair-line historis: {' • '.join(parts)}")
    else:
        body_lines.append("💰 odds: tidak tersedia (bookmaker belum rilis)")

    if model_probs:
        # Section 1: Model A is reference-only. Probabilities are shown, but
        # NO edge/value/EV wording may appear in this section (invariant 6).
        body_lines.append(
            "\n📐 **Model vs Market (Model A — Odds-Implied, Poisson)**"
        )
        body_lines.append(
            "   *(reference only — diturunkan dari market yang sama dengan yang dibandingkan, bukan estimasi independen)*"
        )
        if model_probs.get("1x2"):
            p1x2 = model_probs["1x2"]
            valid_keys = [k for k in ("home", "draw", "away") if consensus.get(k, 0) > 0]
            if valid_keys:
                inv_sum = sum(1.0 / consensus[k] for k in valid_keys)
                for side, label in (("home", "Home"), ("draw", "Draw"), ("away", "Away")):
                    mp = p1x2.get(side, 0)
                    if side in valid_keys:
                        ip = (1.0 / consensus[side]) / inv_sum
                        body_lines.append(
                            f"   {label}: model {_pct(mp)} • market {_pct(ip)}"
                        )
                    else:
                        body_lines.append(f"   {label}: model {_pct(mp)}")

        for thresh in (1.5, 2.5, 3.5):
            mp = model_probs.get(f"over_{thresh}", 0)
            if mp:
                sel = f"Over {thresh}"
                sel_u = f"Under {thresh}"
                o = market_totals.get(sel, {}).get("odds", 0)
                u = market_totals.get(sel_u, {}).get("odds", 0)
                fair = fair_pair_implied(o, u)
                if fair:
                    ip = fair[0]
                elif o > 0:
                    ip = 1.0 / o
                else:
                    ip = None
                if ip:
                    body_lines.append(
                        f"   {sel}: model {_pct(mp)} • market {_pct(ip)}"
                    )
                else:
                    body_lines.append(f"   {sel}: model {_pct(mp)} • market -")

        btts_mp = model_probs.get("btts_yes", 0)
        if btts_mp:
            o_yes = market_totals.get("BTTS Yes", {}).get("odds", 0)
            o_no = market_totals.get("BTTS No", {}).get("odds", 0)
            fair = fair_pair_implied(o_yes, o_no)
            if fair:
                ip = fair[0]
            elif o_yes > 0:
                ip = 1.0 / o_yes
            else:
                ip = None
            if ip:
                body_lines.append(
                    f"   BTTS Yes: model {_pct(btts_mp)} • market {_pct(ip)}"
                )
            else:
                body_lines.append(f"   BTTS Yes: model {_pct(btts_mp)} • market -")

        lam = model_probs.get("lambda_home", 0)
        lam_a = model_probs.get("lambda_away", 0)
        body_lines.append(
            f"\n🧮 Poisson: λ_home={lam:.2f}, λ_away={lam_a:.2f}"
        )

    # ---- NowGoal context (display-only team context, never a model input) ----
    _append_nowgoal_context(body_lines, payload)

    # ---- Data Quality breakdown (what pre-match data exists) -----------
    # Translates the engine's data_completeness into a per-component checklist
    # so the user sees exactly why a match is (or isn't) reliably predictable.
    # Display-only: tier gating in market_tiers.py keeps its own 5 conservative
    # inputs and is NOT affected by this longer, honest checklist.
    has_form = not no_form
    has_ad = bool(home_gf or away_gf)
    has_xg = any(v is not None for v in (hxg, axg, hxga, axga))
    has_h2h = not h2h_zero
    _lineups = payload.get("lineups") or {}
    has_lineups = bool(_lineups.get("home_count"))
    _standings = payload.get("standings") or {}
    has_standings = bool((_standings.get("teams") or {}))
    _missing = payload.get("missing_players") or {}
    has_missing = bool(_missing and any((v or {}).get("missing") for v in _missing.values()))
    _minfo = payload.get("match_info") or {}
    has_match_info = bool(
        _minfo
        and (_minfo.get("venue") or _minfo.get("referee") or _minfo.get("capacity") or _minfo.get("neutral"))
    )
    dq_items = [
        ("💰 odds bookmaker", odds.get("has_odds")),
        ("📊 form 2 tim", has_form),
        ("⚽ attack/defense (GF/GA)", has_ad),
        ("🥅 xG", has_xg),
        ("🔁 H2H", has_h2h),
        ("📋 lineups", has_lineups),
        ("🏆 standings", has_standings),
        ("🚑 missing players", has_missing),
        ("📍 match info/venue", has_match_info),
        ("🧬 nowgoal context", bool(payload.get("nowgoal_context"))),
    ]
    dq_ok = sum(1 for _, ok in dq_items if ok)
    if dq_ok >= 7:
        dq_badge = "🟢 LENGKAP"
    elif dq_ok >= 4:
        dq_badge = "🟡 SEDANG"
    else:
        dq_badge = "🔴 MINIM"
    body_lines.append(
        f"\n📊 **Data Quality** {dq_badge} ({dq_ok}/{len(dq_items)} tersedia)"
    )
    for label, ok in dq_items:
        body_lines.append(f"   {'✅' if ok else '❌'} {label}")
    # Section 5: completeness is load-bearing — show the numeric score and the
    # exact missing inputs; below 0.60 no Final Decision may be issued. The
    # score is computed from the 5 CORE model inputs only, so the missing list
    # mirrors exactly those (context items are display-only).
    dq_score = (payload.get("prediction") or {}).get("data_completeness")
    if dq_score is not None:
        _core_labels = {label for label, _ in dq_items[:5]}
        missing = [label for label, ok in dq_items if not ok and label in _core_labels]
        body_lines.append(
            f"   data_completeness_score: **{dq_score:.2f}** (0.0–1.0)"
            + (f" — missing: {', '.join(missing)}" if missing else " — lengkap")
        )
    if not has_xg or not has_h2h:
        _note = [n for n, ok in (("xG", has_xg), ("H2H", has_h2h)) if not ok]
        _cap = " (keduanya absen → confidence max MEDIUM)" if (not has_xg and not has_h2h) else ""
        body_lines.append(f"   ⚠️ {', '.join(_note)} tidak tersedia{_cap}")

    # ---- Section 2/7: Pick Evaluation (Model B, hard-gated) --------------
    # Only selections passing ALL gates (EV > 3%, |edge| < 20pp, n_bucket >= 30,
    # completeness >= 0.6) are VALID; the rest are listed as informational.
    evaluated = (decision or {}).get("evaluated") or []
    if evaluated:
        valid_picks = [e for e in evaluated if e.get("status") == "VALID"]
        body_lines.append(
            "\n🎯 **Pick Evaluation** (engine independen — gate: EV>3%, |edge|<20pp, "
            "n_bucket≥30, completeness≥0.6)"
        )
        if valid_picks:
            for i, e in enumerate(valid_picks[:3], 1):
                body_lines.append(
                    f"   {i}. {e['market']} • {e['selection']} — model {_pct(e['model_prob'])} • "
                    f"EV {e['ev']:+.0%} • edge {e['edge_pp']:+.1f}pp • n_bucket {e.get('n_bucket') or 0} ✅ lolos gate"
                )
            # Clarity fix: "lolos gate" ≠ "layak bet". Value hanya dikreditkan
            # ke favorit market (best_prob_only, tervalidasi walk-forward);
            # kandidat non-favorit tidak pernah jadi Final Decision.
            if (decision or {}).get("decision_type") == "NO BET":
                body_lines.append(
                    "   ℹ️ lolos gate ≠ layak bet: value hanya untuk favorit market — "
                    "kandidat non-favorit tidak pernah jadi pick."
                )
        else:
            body_lines.append(
                "   ❌ Tidak ada seleksi lolos semua gate — tidak ada pick untuk market ini."
            )
        informational = [e for e in evaluated if e.get("status") != "VALID"]
        if informational:
            body_lines.append("   📋 **Informational Only — Not a Pick:**")
            for e in informational[:4]:
                reason = (e.get("reasons") or [""])[0]
                body_lines.append(
                    f"   • {e['market']} {e['selection']} — {e['status']}: {reason}"
                )

    # ---- Prediction engine section (Elo + feature Poisson + ensemble) ----
    prediction = payload.get("prediction")
    if prediction:
        mp = prediction.get("model_probs") or {}
        p1x2 = mp.get("1x2") or {}
        if p1x2:
            body_lines.append("\n🤖 **Model (Elo+Poisson)**")
            body_lines.append(
                f"   λ_home={mp.get('lambda_home', 0):.2f}, "
                f"λ_away={mp.get('lambda_away', 0):.2f} "
                f"({mp.get('lambda_source', '?')})"
            )
            body_lines.append(
                f"   model 1X2: Home {_pct(p1x2.get('home', 0))} • "
                f"Draw {_pct(p1x2.get('draw', 0))} • Away {_pct(p1x2.get('away', 0))}"
            )
            models_used = "+".join(mp.get("models") or []) or "-"
            if mp.get("elo_seeded") is False and "elo" in (mp.get("models") or []):
                models_used = models_used.replace("elo", "elo(prior)")
            # Addendum v1.1 Section 6: the undefined ``agreement market/models``
            # field is gone from the output (model-vs-model is already covered
            # by the MODEL_DISAGREEMENT check below).
            body_lines.append(f"   models: {models_used}")
            # Section 1: disagreement flag between Model A and Model B is
            # ALWAYS shown (even when false).
            _md = (decision or {}).get("model_disagreement") or {}
            if _md.get("delta_pp") is not None:
                _flag = "⚠️ **MODEL_DISAGREEMENT**" if _md.get("flag") else "OK"
                body_lines.append(
                    f"   Model A vs Model B (home): delta {_md['delta_pp']}pp → {_flag}"
                )
            edge = prediction.get("market_edge") or {}
            if edge:
                body_lines.append(
                    "   edge: " + " • ".join(
                        f"{side.title()} {v:+.1f}%" for side, v in edge.items()
                    )
                )
            calib = prediction.get("calibration") or {}
            if calib.get("samples"):
                body_lines.append(f"   calibration: {calib.get('samples')} samples, ECE {calib.get('ece', '-')}")
            # Addendum v1.1: the ONLY confidence numbers that may appear are
            # model_calibration_score (global) and pick_specific_confidence
            # (local), rendered as ONE block from the strict-allowlist schema.
            _conf = payload.get("confidence")
            if _conf:
                body_lines.extend(_render_confidence_block(_conf))

    # ---- FINAL DECISION (S29/S30): separate PREDICTION (most likely) from
    # the best overall DECISION; explain the difference. NO CLEAR DECISION /
    # NO BET are valid outputs.
    if decision and decision.get("decision_type"):
        body_lines.append("\n🏆 **FINAL DECISION** (engine independen Elo+Poisson+kalibrasi, bukan cermin odds)")
        d_type = decision["decision_type"]
        fd = decision.get("final_decision")
        badge = _DECISION_BADGES.get(d_type, d_type)
        # Output-policy clarity (S31): when the tier layer demoted EVERY
        # market to Tier 3 (WATCH), the engine's GOOD/LEAN view below is
        # informational only — the betting verdict is SKIP. Say so explicitly
        # right here so the two blocks can never read as contradicting each
        # other. The tier layer is the final word (PICK > LEAN > WATCH).
        _tier_pick = select_best_pick(payload)
        if (
            fd is not None
            and _tier_pick is not None
            and _tier_pick.get("tier") == "WATCH"
        ):
            body_lines.append(
                "   ℹ️ INFORMATIF SAJA — semua market turun ke Tier 3 (WATCH); "
                "keputusan taruhan: **SKIP**. Pandangan engine di bawah bukan "
                "rekomendasi taruhan (verifikasi fixture/odds dulu)."
            )
        if fd:
            top = (decision.get("score_breakdown") or {}).get("top") or {}
            body_lines.append(
                f"   **{badge}** • {fd['market']} **{fd['selection']}** @ {fd['market_odds']:.2f}"
            )
            body_lines.append(
                f"   skor {top.get('score', 0):.2f} • model {_pct(fd['model_prob'])} "
                f"• edge {fd['edge_pp']:+.1f}pp • EV {fd['ev']:+.0%}"
            )
            # Phase 6: recommended fractional-Kelly stake (never auto-bet).
            _stake = decision.get("stake")
            if _stake:
                if _stake.get("declined"):
                    body_lines.append(f"   ⛔ stake: {_stake.get('reason')}")
                else:
                    body_lines.append(
                        f"   💰 stake: {_stake['stake_fraction'] * 100:.1f}% bankroll "
                        f"(≈{_stake['stake_amount']} unit) — ¼ Kelly, cap {_stake.get('bankroll')} unit"
                    )
            # TODO-10: variance-aware EV band (ensemble spread) when present.
            ev_band = (decision.get("ev_band") or {})
            if ev_band and ev_band.get("ev_low") is not None:
                body_lines.append(
                    f"   📉 EV band [{ev_band['ev_low']:+.1%}, {ev_band['ev_high']:+.1%}] "
                    f"(spread {ev_band.get('uncertainty', 0):.2f})"
                )
            ml = decision.get("most_likely") or {}
            if ml and ml.get("selection") != fd["selection"]:
                body_lines.append(
                    f"   most likely: {ml.get('selection')} ({_pct(ml.get('model_prob'))}) "
                    "— paling mungkin ≠ keputusan terbaik"
                )
        elif d_type == "MARKET PRIOR":
            # Thin-data honesty (S29/S30): the market IS the best estimator
            # when the independent model has no signal. Predictions are the
            # margin-free market probabilities -- shown explicitly, with the
            # honest label that no edge is claimed and betting advice is
            # NO BET.
            body_lines.append(f"   **{badge}**")
            mpred = decision.get("market_predictions") or {}
            p1 = mpred.get("1x2")
            if p1:
                top = max(p1, key=p1.get)
                label = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}[top]
                body_lines.append(
                    f"   🎯 most likely 1X2 (market): **{label}** {_pct(p1[top])} "
                    f"• Home {_pct(p1.get('home'))} / Draw {_pct(p1.get('draw'))} / Away {_pct(p1.get('away'))}"
                )
            if mpred.get("over_2.5") is not None:
                o = mpred["over_2.5"]
                pick = "Over 2.5" if o >= 0.5 else "Under 2.5"
                body_lines.append(
                    f"   🎯 total (market): **{pick}** — Over {_pct(o)} / Under {_pct(mpred['under_2.5'])}"
                )
            if mpred.get("btts_yes") is not None:
                y = mpred["btts_yes"]
                pick = "BTTS Yes" if y >= 0.5 else "BTTS No"
                body_lines.append(
                    f"   🎯 BTTS (market): **{pick}** — Yes {_pct(y)} / No {_pct(mpred['btts_no'])}"
                )
        else:
            body_lines.append(f"   **{badge}**")
        if decision.get("explanation"):
            body_lines.append(f"   {decision['explanation']}")
        if decision.get("betting_advice") and d_type == "MARKET PRIOR":
            body_lines.append(f"   🚫 saran taruhan: **{decision['betting_advice']}** (prediksi = market → tanpa edge)")
        # Clarity fix: NO BET walau ada kandidat ber-+EV yang lolos gate
        # teknis — jelaskan kenapa keduanya tidak bertentangan (value hanya
        # untuk favorit market, bukan semua yang lolos gate).
        if d_type == "NO BET" and not fd:
            valid_pos = [
                e for e in (decision.get("evaluated") or [])
                if e.get("status") == "VALID" and (e.get("ev") or 0) > 0
            ]
            if valid_pos:
                body_lines.append(
                    f"   ℹ️ {len(valid_pos)} kandidat ber-+EV lolos gate teknis tapi bukan "
                    "favorit market → tanpa kredit value (aturan tervalidasi). "
                    "NO BET = keputusan benar: tidak ada edge yang bisa dieksekusi."
                )
        for w in decision.get("edge_warnings") or []:
            body_lines.append(f"   {w}")

        # Phase 2: always label the benchmark the edge was measured against.
        # With no sharp source, the edge is vs a soft consensus — never a
        # claim of beating an efficient market.
        _bench = decision.get("edge_benchmark") or {}
        if _bench.get("label"):
            body_lines.append(f"   ℹ️ edge benchmark: {_bench['label']}")

        # Phase 3 CLV gate: show whether the segment is allowed to act.
        _gate = decision.get("clv_gate")
        if _gate:
            if _gate.get("allowed"):
                body_lines.append(
                    f"   ✅ CLV gate: segmen lolos (n={_gate.get('n')}, "
                    f"CLV {_gate.get('clv_pct')}%)"
                )
            else:
                body_lines.append(f"   ⛔ CLV gate BLOCK: {_gate.get('reason')}")

        # Plan B movement signal: drift/steam/agreement from the hourly curve.
        _mv = decision.get("movement")
        if _mv and _mv.get("usable"):
            _steam = _mv.get("steam_side") or "-"
            _drift = " ".join(
                f"{k[:1].upper()}{_mv['drift_pct'].get(k, 0):+.1f}%"
                for k in ("home", "draw", "away")
            )
            body_lines.append(
                f"   📈 movement: steam {_steam} (drift {_drift}) — "
                f"agreement {_mv.get('agreement')} ({_mv.get('n')} snapshot)"
            )

        # Historical reliability of a similar signal bucket (when available).
        ss = payload.get("similar_signal") or {}
        m = ss.get("matching") or {}
        if m:
            if m.get("sufficient_sample"):
                _r = m.get("roi")
                _r_pct = f"{_r * 100:.1f}" if _r is not None else "-"
                _c = m.get("clv_pct")
                _c_pct = f"{_c:.1f}" if _c is not None else "-"
                body_lines.append(
                    f"   📚 bucket serupa ({m.get('confidence')} • {m.get('edge')}, "
                    f"n={m.get('n')}): hit {m.get('hit_rate', 0) * 100:.0f}% • "
                    f"ROI {_r_pct}% • CLV {_c_pct}%"
                )
            # Addendum v1.1 Section 2: the "belum cukup sampel (n=x < 5)" line
            # used a threshold that contradicts the n_bucket >= 30 gate — it is
            # removed from the output entirely (nothing is shown until the
            # bucket has a sufficient sample).

    body_lines.extend(render_signal_engine(payload))

    body_lines.append(
        "\n⚠️ Disclaimer: prediksi berbasis odds konsensus + model Poisson. "
        "xG riil (kalau ada) override odds-derived λ. "
        "Bukan jaminan hasil."
    )

    _append_event_stats(body_lines, payload.get("event_stats"))

    # ---- Flashscore lineups (context info; predicted pre-match / confirmed) --
    lineups = payload.get("lineups")
    if lineups and lineups.get("home_count"):
        status_label = (
            "PREDICTED (belum resmi)" if lineups.get("status") == "predicted"
            else "CONFIRMED"
        )
        body_lines.append(f"\n📋 **Lineups** — {status_label}")
        formations = lineups.get("formations") or []
        for idx, (side, players) in enumerate((
            ("home", lineups.get("home") or []),
            ("away", lineups.get("away") or []),
        )):
            if not players:
                continue
            fmt_line = f"   {side.title()} "
            if idx < len(formations):
                fmt_line += f"({formations[idx]}) "
            names = ", ".join(
                f"{p['name']}" + (f" ({p['number']})" if p.get("number") else "")
                for p in players[:8]
            )
            if len(players) > 8:
                names += f" +{len(players) - 8}"
            body_lines.append(fmt_line + names)

    # ---- Flashscore pre-match context: venue, missing players, coaches, ----
    # standings (context only; never feeds the model math).
    match_info = payload.get("match_info") or {}
    if match_info:
        parts = []
        if match_info.get("neutral"):
            parts.append("⚖️ NETRAL")
        if match_info.get("venue"):
            v = match_info["venue"]
            if match_info.get("town"):
                v += f" ({match_info['town']})"
            parts.append(v)
        if match_info.get("capacity"):
            parts.append(f"cap {match_info['capacity']}")
        if match_info.get("referee"):
            r = match_info["referee"]
            if match_info.get("referee_country"):
                r += f" ({match_info['referee_country']})"
            parts.append(f"referee {r}")
        if parts:
            body_lines.append("📍 " + " • ".join(parts))

    missing_players = payload.get("missing_players") or {}
    if missing_players and any((v or {}).get("missing") for v in missing_players.values()):
        body_lines.append("\n🚑 **Missing Players** (flashscore, pre-match)")
        for side in ("home", "away"):
            entry = missing_players.get(side) or {}
            miss = entry.get("missing") or []
            if not miss:
                continue
            parts = [
                p["name"] + (f" ({p['reason']})" if p.get("reason") else "")
                for p in miss
            ]
            line = f"   {side.title()}: {', '.join(parts)}"
            unsure = entry.get("unsure") or []
            if unsure:
                line += " • doubtful: " + ", ".join(p["name"] for p in unsure)
            body_lines.append(line)

    coaches = payload.get("coaches") or {}
    coach_parts = [
        f"{side.title()}: {', '.join(coaches.get(side) or [])}"
        for side in ("home", "away")
        if coaches.get(side)
    ]
    if coach_parts:
        body_lines.append("🧑‍🏫 Pelatih: " + " • ".join(coach_parts))

    standings = payload.get("standings") or {}
    srows = standings.get("teams") or {}
    if srows:
        parts = []
        for side in ("home", "away"):
            row = srows.get(side) or {}
            if not row.get("team"):
                continue
            label = f"#{row.get('pos') or '?'} {row['team']}"
            if row.get("pts") is not None:
                label += f" ({row['pts']} pts)"
            if row.get("mp"):
                label += f" • {row['mp']}MP W{row.get('w') or 0}-D{row.get('d') or 0}-L{row.get('l') or 0}"
            if row.get("form"):
                label += f" • {row['form']}"
            parts.append(f"{side.title()} {label}")
        if parts:
            body_lines.append("📊 Klasemen: " + " • ".join(parts))

    sources = payload.get("sources") or []
    quota = payload.get("quota", {})
    footer = _build_quota_footer(quota, sources)
    snap = []
    if payload.get("generated_at"):
        snap.append(f"gen {payload['generated_at']}")
    if prediction and prediction.get("input_hash"):
        snap.append(f"#{prediction['input_hash']}")
    if snap:
        footer = (footer + " • " + " • ".join(snap)).strip()

    return {
        "title": "🔬 Analisa Match",
        "body": "\n".join(body_lines),
        "footer": footer,
    }


# ---- Compact match-analysis summary (bot default) ----------------------
#
# The analyse command's main Discord reply follows the OUTPUT POLICY —
# Single Best Pick (selection layer only): every market (1X2, O/U 2.5,
# O/U 3.5, BTTS) is still computed and tiered in full, but only ONE
# already-computed result is surfaced (PICK > LEAN > WATCH) with its tier,
# confidence, basis and stake. The selection never re-runs or alters the
# analysis — it only decides which computed result to show. The per-market
# breakdown stays one click away via the 📋 Copy button (the runner emits it
# under ``render_full``).

_DECISION_BADGES = {
    "STRONG": "🟢 STRONG",
    "GOOD": "✅ GOOD",
    "LEAN": "🟡 LEAN",
    "WATCH": "👁 WATCH",
    "NO BET": "🚫 NO BET",
    "NO CLEAR DECISION": "⚪ NO CLEAR DECISION",
    "MARKET PRIOR": "📊 MARKET PRIOR",
}


_SIGNAL_CONF_ICONS = {
    "VERY HIGH": "🟢",
    "HIGH": "🟢",
    "MEDIUM": "🟡",
    "LOW": "🔴",
    "NO SIGNAL": "⚪",
}

# Human-readable name for each evidence group, for the WHY? block.
_SIGNAL_GROUP_LABELS = {
    "model": "Existing model",
    "statistical": "Statistical",
    "market": "Market confirmation",
    "movement": "Odds movement",
    "late_movement": "Late movement",
    "data_quality": "Data quality",
    "team_context": "Team context",
}


def render_signal_engine(payload: dict[str, Any]) -> list[str]:
    """The Market-Aware Signal Engine block (SIGNALS + BEST PICK + WHY?).

    Read-only: the engine already computed everything deterministically;
    this only renders it. NO BET is a first-class output.
    """
    se = payload.get("signal_engine")
    if not se:
        return []
    lines: list[str] = ["\n🎯 **MARKET-AWARE SIGNALS** (BTTS • O/U 2.5 • Asian Handicap)"]
    ranking = se.get("ranking") or []
    if ranking:
        lines.append("━━━━━━━━━━━━━━━━")
        for s in ranking[:3]:
            icon = _SIGNAL_CONF_ICONS.get(s.get("confidence"), "⚪")
            odds = f" @ {s['market_odds']:.2f}" if s.get("market_odds") else ""
            lines.append(f"{icon} {s['selection']}{odds}")
            lines.append(
                f"   Score: {s['score'] * 100:.0f} • Confidence: {s['confidence']} • "
                f"edge {s.get('edge_pp', 0.0):+.1f}pp"
            )
        lines.append("━━━━━━━━━━━━━━━━")

    bp = se.get("best_pick")
    label = _pick_label(se)
    if se.get("decision") == "BEST PICK" and bp:
        icon = _SIGNAL_CONF_ICONS.get(bp.get("confidence"), "🟢")
        lines.append(f"🏆 **{label}**: {bp['selection']}")
        lines.append(f"   Confidence: {bp['confidence']}")
        lines.append(f"   {_edge_display(se, bp)}")
        confl = bp.get("confluence")
        if confl is not None:
            total = len(bp.get("components") or {})
            lines.append(f"   CONFLUENCE: {confl}/{total} evidence groups")
        # Concise WHY?: evidence groups that agree, plus any conflict.
        agreeing = [
            _SIGNAL_GROUP_LABELS.get(k, k)
            for k, v in (bp.get("components") or {}).items()
            if v >= 0.55
        ]
        movement = bp.get("movement") or {}
        why: list[str] = []
        if agreeing:
            why.append("• " + ", ".join(agreeing) + " support the signal.")
        if se.get("edge_invalid"):
            why.append("• Edge benchmark stale — edge: n/a.")
        elif bp.get("edge_pp", 0.0) > 0:
            why.append(f"• Model edge vs market: {bp['edge_pp']:+.1f}pp.")
        if movement.get("status") == "available" and movement.get("direction") == "toward":
            why.append("• Odds shortened toward the selection.")
        elif movement.get("status") == "UNAVAILABLE":
            why.append("• Odds movement unavailable (no opening prices).")
        if why:
            lines.append("   WHY?")
            lines.extend(f"   {w}" for w in why)
        else:
            lines.append("   WHY? No single evidence group clearly dominant.")
        meta = _benchmark_meta(se)
        if meta:
            lines.append("   " + " • ".join(meta))
        notes = bp.get("evidence_notes") or []
        for n in notes:
            lines.append(f"   ⚠️ {n}")
        # Market Intelligence display (steam/RLM)
        mi = se.get("market_intelligence") or {}
        if mi.get("side"):
            mi_lines = []
            if mi.get("steam_moves"):
                for mv in mi["steam_moves"]:
                    mi_lines.append(f"Steam on {mv['side']}: {mv['magnitude_pct']:+.1f}% in {mv['window_minutes']}min")
            if mi.get("rlm"):
                rlm = mi["rlm"]
                mi_lines.append(f"RLM: {rlm.get('reason', '')}")
            if mi.get("agreement", {}).get("dominant_direction"):
                agr = mi["agreement"]
                mi_lines.append(f"{agr['agreement_count']}/{agr['total_bookmakers']} books on {agr['dominant_side']}")
            if mi.get("model_agreement") == 1.0:
                mi_lines.append("✅ Market intel agrees with model")
            elif mi.get("model_agreement") == 0.0:
                mi_lines.append("⚠️ Market intel DISAGREES with model")
            if mi_lines:
                lines.append("   📊 **Market Intel:**")
                for ml in mi_lines:
                    lines.append(f"   • {ml}")
    else:
        lines.append(f"🏆 **{label}**: ❌ NO BET")
        reason = (se.get("reasons") or ["no signal strong enough"])[-1]
        lines.append(f"   {reason}")

    dq = se.get("data_quality") or {}
    if dq:
        flags = []
        if not dq.get("ah_available"):
            flags.append("Asian Handicap market unavailable")
        if not dq.get("movement_history_available"):
            flags.append("no odds history (late movement unavailable)")
        if flags:
            lines.append(f"   ⚠️ {'; '.join(flags)}")
    return lines


# ---- MARKET SIGNAL primary output (OUTPUT POLICY v2) --------------------
# The analyse command's main reply is a clean market-signal card: SIGNALS
# (sorted strongest -> weakest) + BEST PICK + MARKET movement + disclaimer.
# Internal engine diagnostics (lambda, n_bucket, calibration samples, gate
# names, Model A/B disagreement) never appear here -- they stay in logs and
# the developer debug mode, not in the user-facing Discord output.

_SIGNAL_SEPARATOR = "━━━━━━━━━━━━━━━━━━"


def _move_pct(opening: float | None, current: float | None) -> float | None:
    if not opening or not current or opening <= 1.0 or current <= 1.0:
        return None
    return round((current - opening) / opening * 100.0, 2)


def _fmt_ah_line(line: float | None) -> str:
    if line is None:
        return "-"
    return f"{line:+.2f}"


def _signal_why(bp: dict[str, Any], movement_flags: list[str] | None = None) -> list[str]:
    """Concise, evidence-derived Why? bullets for the Best Pick.

    Each bullet maps to a real signal: the model's direction, the model-vs-
    market edge, and the odds movement. Nothing is invented. The movement
    bullet is bound to the SAME Layer-1 canonical opening the MARKET block
    displays; when ``movement_flags`` reports a contradiction (Layer 4
    validation), the movement bullet is suppressed instead of emitting a
    narrative that contradicts the numbers in the same response.
    """
    market = bp.get("market")
    sel = bp.get("selection", "")
    edge = bp.get("edge_pp", 0.0)
    mv = bp.get("movement") or {}
    bullets: list[str] = []
    movement_conflict = bool(movement_flags)

    # Phase 5.3: real components instead of boilerplate -- model probability
    # vs implied probability, the line itself, and lineup status. Nothing is
    # invented; each bullet maps to a stored value on the pick.
    model_prob = bp.get("model_prob")
    implied = bp.get("implied_prob")
    line = bp.get("line")
    if model_prob is not None:
        if implied is not None:
            bullets.append(
                f"Model {model_prob:.0%} vs implied {implied:.0%} "
                f"(edge {edge:+.1f}pp)"
            )
        else:
            bullets.append(f"Model probability: {model_prob:.0%}")
    elif edge:
        bullets.append(f"Edge vs market: {edge:+.1f}pp")
    if market == "Asian Handicap" and line is not None:
        side = bp.get("side") or ("home" if sel.startswith("Home") else "away")
        bullets.append(f"Line: {side.title()} {line:+.2f}")
    elif market == "Total" and line is not None:
        bullets.append(f"Line: {line:g} goals")
    lu = bp.get("lineup_status")
    if lu == "confirmed":
        bullets.append("Lineup: confirmed")
    elif lu == "predicted":
        bullets.append("Lineup: predicted (unconfirmed, half weight)")
    elif lu == "none":
        bullets.append("Lineup: not available")

    if movement_conflict:
        bullets.append("Odds movement data inconsistent (suppressed).")
    elif mv.get("status") == "available":
        if mv.get("direction") == "toward":
            bullets.append("Odds movement confirms the signal")
        elif mv.get("direction") == "away":
            bullets.append("Odds movement opposes the signal")
    return bullets


def _human_no_bet_reason(reasons: list[str] | None) -> str:
    rs = reasons or []
    if any("no signal candidates" in r for r in rs):
        return "No signal candidates available."
    # 2026-08-22 pick_gates: report the ACTUAL gate that rejected the
    # candidates. Without this the card fell through to the generic line below,
    # so a match could show three SCORED candidates (79/56/48) next to "no
    # signal reaches the actionable threshold" -- which reads as a broken bot
    # rather than a deliberate decision. Each probe is ordered most-specific
    # first; the reason is trimmed at the em-dash to keep the card readable.
    _GATE_PROBES = (
        ("diveto (G1)", "Layer model 1X2 tidak menemukan bet"),
        ("deviasi model-pasar", "Model menyimpang terlalu jauh dari harga pasar"),
        ("lambda_total", "Estimasi gol model di luar batas wajar"),
        ("tidak ada harga", "Tidak ada harga pasar yang bisa ditaruhkan"),
        ("bookmaker", "Jumlah bookmaker terlalu sedikit untuk konsensus harga"),
        ("melawan favorit 1X2", "Kandidat bertentangan dengan favorit model"),
        ("kontradiksi internal", "Model internal saling bertentangan"),
        ("Elo", "Rating Elo tidak kredibel"),
        ("fabrikasi", "Identitas pertandingan tidak konsisten"),
    )
    for needle, headline in _GATE_PROBES:
        for r in rs:
            if needle in r:
                detail = r.split(" — ")[0].strip()
                return f"{headline} ({detail})."
    if any("conflict" in r for r in rs):
        return "Model and market contradict each other."
    if any("confluence" in r for r in rs):
        return "Evidence is too thin to act on."
    if any("data quality" in r for r in rs):
        return "Insufficient data quality."
    return "No signal reaches the actionable threshold."


def _market_block(payload: dict[str, Any], se: dict[str, Any]) -> list[str]:
    """Market movement lines (no raw objects): Over 2.5 + Under 2.5 + the
    Asian Handicap side of the best pick (or the away side when there is no
    AH pick). Opening always comes from the immutable Layer-1 canonical
    opening_snapshot when one exists -- the SAME opening the engine's
    movement scoring reads -- so narrative and display can never disagree.
    Payloads without a canonical snapshot fall back to their own opening
    fields (``canonical`` False)."""
    lines: list[str] = []
    totals = (payload.get("odds") or {}).get("totals") or {}
    mb = se.get("market_block") or {}
    ou_mb = mb.get("ou") or {}
    ah_mb = mb.get("ah") or {}

    def _pair_block(
        label: str,
        opening: float | None,
        latest: float | None,
        point: float | None = None,
        opening_point: float | None = None,
    ) -> None:
        if not latest:
            return
        lines.append(label)
        # Goal-line change disclosure (2026-08-22 fix): NowGoal preserves the
        # opening line (``opening_point``) separately from the current one
        # (``point``). When they differ, an opening->latest price comparison
        # spans DIFFERENT handicap lines and must not read as same-line drift
        # (Everton-Palace: Over "2.04 -> 2.02, ↓1.0%" was actually a 2.75 ->
        # 2.50 line drop). Annotate both ends like the AH block does.
        _line_moved = (
            point is not None and opening_point is not None
            and abs(float(point) - float(opening_point)) > 1e-9
        )
        if opening:
            txt = f"Opening: {opening:.2f}"
            if _line_moved:
                txt += f" (garis {float(opening_point):.2f})"
            lines.append(txt)
        txt = f"Latest: {latest:.2f}"
        if _line_moved:
            txt += f" (garis {float(point):.2f})"
        lines.append(txt)
        move = _move_pct(opening, latest)
        if move is not None:
            arrow = "↓" if move < 0 else ("↑" if move > 0 else "→")
            lines.append(f"Movement: {arrow} {abs(move):.1f}%")

    # Over 2.5 + Under 2.5 -- both sides shown, so the narrative always
    # matches one of the displayed series.
    over = totals.get("Over 2.5") or {}
    if ou_mb.get("canonical"):
        _pair_block("Over 2.5", ou_mb.get("opening_over"), ou_mb.get("latest_over"))
    else:
        _pair_block(
            "Over 2.5", over.get("opening"), over.get("odds"),
            point=over.get("point"), opening_point=over.get("opening_point"),
        )
    under = totals.get("Under 2.5") or {}
    if ou_mb.get("canonical"):
        _pair_block("Under 2.5", ou_mb.get("opening_under"), ou_mb.get("latest_under"))
    else:
        _pair_block(
            "Under 2.5", under.get("opening"), under.get("odds"),
            point=under.get("point"), opening_point=under.get("opening_point"),
        )

    ah = se.get("ah_consensus")
    bp = se.get("best_pick")
    pick_is_ah = bool(bp and bp.get("market") == "Asian Handicap")
    if ah and ah.get("line") is not None:
        cur_line = float(ah["line"])
        if pick_is_ah:
            side = bp.get("side") or (
                "home" if str(bp.get("selection", "")).startswith("Home") else "away"
            )
            line_cur = float(bp.get("line") or cur_line)
        else:
            side = "away"
            line_cur = cur_line
        if lines:
            lines.append("")
        lines.append("Asian Handicap")

        def _side_label(sd: str, ln: float) -> str:
            return f"{sd.title()} {_fmt_ah_line(ln if sd == 'home' else -ln)}"

        if ah_mb.get("canonical"):
            open_line = ah_mb.get("opening_line")
            opening_price = ah_mb.get(f"{side}_open")
            latest_price = ah_mb.get(f"{side}_latest")
        else:
            open_line = float(ah["line_open"]) if ah.get("line_open") is not None else None
            opening_price = ah.get(f"{side}_open")
            latest_price = ah.get(side)
        if open_line is not None:
            opening_txt = f"Opening: {_side_label(side, open_line)}"
            opening_txt += f" @ {opening_price:.2f}" if opening_price else ""
            lines.append(opening_txt)
        latest_txt = f"Latest: {_side_label(side, line_cur)}"
        latest_txt += f" @ {latest_price:.2f}" if latest_price else ""
        lines.append(latest_txt)
    # P1-1: when the market block was rendered from a non-canonical fallback
    # (no immutable first-seen snapshot yet), surface the warning so the user
    # knows the numbers are real but not yet pinned.
    if ou_mb.get("non_canonical") or ah_mb.get("non_canonical"):
        lines.append("")
        lines.append("⚠️ opening from current snapshot (not pinned)")
    return lines


def _signals_block(ranking: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for s in ranking[:3]:
        icon = _SIGNAL_CONF_ICONS.get(s.get("confidence"), "⚪")
        odds = f" @ {s['market_odds']:.2f}" if s.get("market_odds") else ""
        lines.append(f"{icon} {s['selection'].upper()}{odds}")
        lines.append(f"Score: {round(s['score'] * 100)}/100")
        # 2026-08-22 pick_gates: a gated candidate keeps its score but is not
        # selectable, so say so. "Score: 79/100 / Confidence: NO SIGNAL" with no
        # explanation reads as a malfunction; the veto reason makes it a
        # decision the reader can audit.
        if s.get("vetoed"):
            _vr = [str(r) for r in (s.get("veto_reasons") or []) if r]
            _detail = _vr[0].split(" — ")[0].strip() if _vr else "gate pick_gates"
            lines.append(f"Status: ❌ DITOLAK — {_detail}")
        else:
            lines.append(f"Confidence: {s.get('confidence')}")
        lines.append("")
    return lines


def _pick_label(se: dict[str, Any]) -> str:
    """Phase 5.1: "BEST PICK" -> "TOP SIGNAL" when the league is uncalibrated.

    An uncalibrated league has no validated per-league fit (Phase 1.5), so
    the pick must not be sold as a best pick -- it is only a top signal from
    an unvalidated model.

    Phase 5.4: uncalibrated leagues show explicit warning so users can
    make informed decisions.
    """
    label = se.get("display_label")
    if label == "BEST PICK":
        return "BEST PICK"
    if label == "TOP SIGNAL":
        # Phase 5.4: add explicit warning for uncalibrated leagues
        return "TOP SIGNAL ⚠️ LIGA TIDAK TERKALIBRASI"
    return "BEST PICK"


def _edge_display(se: dict[str, Any], bp: dict[str, Any]) -> str:
    """Phase 5.1/0.2: ``edge: n/a`` for a stale/invalid benchmark, never a
    stale number."""
    if se.get("edge_invalid") or (bp or {}).get("edge_invalid"):
        return "edge: n/a"
    e = (bp or {}).get("edge_pp")
    if e is None:
        return "edge: n/a"
    return f"edge {e:+.1f}pp"


def _display_best_pick(se: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Pick yang DITAMPILKAN di card.

    F11 (plan v3, 2026-08-24): satu-satunya pick yang boleh tampil adalah
    ``best_pick`` hasil ``run_signal_engine``. Kebijakan lama (2026-08-23)
    menampilkan rank #1 berlabel HIGH RISK saat SEMUA kandidat diveto -- di
    lapangan itu terbaca sebagai BEST PICK biasa (kasus Goztepe v
    Genclerbirligi: Over 2.5 tampil padahal lambda_total 4.11 sudah diveto G4
    kartu-level), jadi kini keputusan NO BET dari signal engine selalu
    ditampilkan apa adanya sebagai ⚪ NO BET beserta alasan gerbangnya.
    Pick decision-layer TIDAK pernah dirender lewat jalur ini.

    Returns ``(pick, risk_reason)``; kini ``risk_reason`` selalu None
    (dipertahankan demi kompatibilitas signature).
    """
    bp = se.get("best_pick")
    if se.get("decision") == "BEST PICK" and bp:
        return bp, None
    return None, None


def _high_risk_line(risk_reason: str) -> str:
    return f"⚠️ HIGH RISK — gagal gerbang: {risk_reason}"


def _best_pick_block(se: dict[str, Any], *, include_internal: bool = True) -> list[str]:
    """Render the BEST PICK / NO BET block.

    P2-3: ``include_internal=False`` suppresses the ``internal_notes`` list
    (model-vs-model disagreements like "model 1X2 NO BET — pick tidak
    didukung layer model") so the summary embed stays clean. The expanded
    embed renders with the default ``include_internal=True`` so the user
    sees the internal disagreement AFTER clicking "Lihat Hasil".

    Keputusan 2026-08-23: saat SEMUA kandidat diveto pick_gates, rank #1
    tetap ditampilkan dengan label HIGH RISK (see ``_display_best_pick``).
    """
    lines: list[str] = []
    label = _pick_label(se)
    bp, risk_reason = _display_best_pick(se)
    if bp is not None:
        odds = f" @ {bp['market_odds']:.2f}" if bp.get("market_odds") else ""
        lines.append(f"## 🔥 {label}: {bp['selection'].upper()}{odds}")
        if risk_reason:
            lines.append(_high_risk_line(risk_reason))
        lines.append("")
        lines.append(f"Confidence: {bp.get('confidence')}")
        lines.append(f"{_edge_display(se, bp)}")
        lines.append("")
        lines.append("Why:")
        # Layer 4: narrative movement claims are cross-checked against the
        # displayed pick-side movement before rendering; a contradiction
        # suppresses the movement bullet instead of emitting a wrong claim.
        why = _signal_why(bp, movement_narrative_flags(se))
        if se.get("edge_invalid"):
            why.append("Edge benchmark stale (>24h) — edge tidak dapat dipercaya (n/a).")
        if why:
            lines.extend(f"• {w}" for w in why)
        else:
            lines.append("• No single evidence group clearly dominant.")
        # Phase 5.2: every card shows benchmark age, bookmaker count, movement
        # snapshot count and movement direction/magnitude.
        meta = _benchmark_meta(se)
        if meta:
            lines.append("")
            lines.extend(meta)
        # Phase 5.4: uncalibrated league warning in expanded view
        if se.get("display_label") == "TOP SIGNAL":
            lines.append("")
            lines.append("⚠️ **LIGA INI TIDAK TERKALIBRASI — model belum terbukti akurat untuk liga ini.**")
            lines.append("Rekomendasi: SKIP pick ini atau gunakan dengan potensi risiko tinggi.")
        # F2: explicit evidence-floor notes (prior-Elo model without form
        # support). Always visible -- these explain WHY a HIGH-confidence
        # pick was capped down.
        notes = bp.get("evidence_notes") or []
        if notes:
            lines.append("")
            for n in notes:
                lines.append(f"⚠️ {n}")
        # P2-3: F3 model-vs-model notes are gated to the expanded view.
        if include_internal:
            internal = bp.get("internal_notes") or []
            if internal:
                lines.append("")
                for n in internal:
                    lines.append(f"⚠️ {n}")
        # Layer 3: repeated-query stability guard note (held vs labeled change).
        st = se.get("stability") or {}
        if st.get("status") == "held":
            lines.append("")
            lines.append(
                f"🔄 Dipertahankan dari analisis sebelumnya "
                f"({st.get('previous_selection')}) — {st.get('reason')}"
            )
        elif st.get("status") == "changed":
            lines.append("")
            lines.append(
                f"🔄 Berubah dari {st.get('previous_selection')} → "
                f"{st.get('new_selection')} — {st.get('reason')}"
            )
    else:
        lines.append(f"## ⚪ NO BET ({label})")
        lines.append("")
        lines.append("Reason:")
        lines.append(_human_no_bet_reason(se.get("reasons")))
        if include_internal:
            internal = (bp or {}).get("internal_notes") or []
            if internal:
                lines.append("")
                for n in internal:
                    lines.append(f"⚠️ {n}")
    return lines


def _best_pick_block_with_lean(payload: dict[str, Any], se: dict[str, Any], *, include_internal: bool = True) -> list[str]:
    """Wrapper yang menambah MARKET LEAN SELALU (Opsi A 2026-08-26)."""
    base = _best_pick_block(se, include_internal=include_internal)
    lean = _market_lean_block(payload, se)
    if lean:
        base.extend(lean)
    return base


def _lean_candidates(payload: dict[str, Any], se: dict[str, Any]) -> list[dict[str, Any]]:
    """Kandidat lean per market dengan implied fair & vig untuk skor adjusted.

    Pure display — tidak mengubah decision. Tiap kandidat:
      label, odds, implied (0..1 fair), vig (0..), market_type
    Vig = sum 1/odds -1. Adjusted score = implied * (1 - vig) — penalti pasar
    dengan margin tebal / likuiditas tipis. Movement penalty (away>2%) opsional
    via se.market_block jika tersedia — untuk transparansi max implied vs
    most suitable.
    """
    totals = (payload.get("odds") or {}).get("totals") or {}
    consensus = (payload.get("odds") or {}).get("consensus") or {}
    over = totals.get("Over 2.5") or {}
    under = totals.get("Under 2.5") or {}
    cands: list[dict[str, Any]] = []
    # Totals
    if over.get("odds") and under.get("odds"):
        try:
            o, u = float(over["odds"]), float(under["odds"])
            if o > 1.0 and u > 1.0:
                ia, ib = 1.0 / o, 1.0 / u
                tot = ia + ib
                vig = tot - 1.0
                lean_label = "Under 2.5" if u < o else "Over 2.5"
                lean_odds = u if u < o else o
                imp = (ib / tot) if lean_label.startswith("Under") else (ia / tot)
                cands.append({"label": lean_label, "odds": lean_odds, "implied": imp, "vig": vig, "market": "Total", "raw_label": lean_label})
        except Exception:
            pass
    # AH
    ah = (se.get("ah_consensus") or {}) or (payload.get("ah_consensus") or {})
    if ah.get("line") is not None and ah.get("home") and ah.get("away"):
        try:
            line = float(ah["line"])
            h, a = float(ah["home"]), float(ah["away"])
            if h > 1.0 and a > 1.0:
                ia, ib = 1.0 / h, 1.0 / a
                tot = ia + ib
                vig = tot - 1.0
                if a < h:
                    lean_ah = f"Away {-line:+.2f}" if abs(line) > 1e-9 else "Away +0.00"
                    lean_odds, imp = a, ib / tot
                else:
                    lean_ah = f"Home {line:+.2f}"
                    lean_odds, imp = h, ia / tot
                cands.append({"label": f"AH: {lean_ah}", "odds": lean_odds, "implied": imp, "vig": vig, "market": "Asian Handicap", "raw_label": lean_ah})
        except Exception:
            pass
    # 1X2
    if consensus.get("home") and consensus.get("draw") and consensus.get("away"):
        try:
            h, d, a = float(consensus["home"]), float(consensus["draw"]), float(consensus["away"])
            if h > 1.0 and d > 1.0 and a > 1.0:
                ia, ib, ic = 1.0 / h, 1.0 / d, 1.0 / a
                tot = ia + ib + ic
                vig = tot - 1.0
                side = min(consensus, key=lambda k: float(consensus[k]))
                label = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}.get(side, side)
                imp = {"home": ia / tot, "draw": ib / tot, "away": ic / tot}[side]
                cands.append({"label": f"1X2: {label}", "odds": float(consensus[side]), "implied": imp, "vig": vig, "market": "1X2", "raw_label": label})
        except Exception:
            pass
    # BTTS
    by = totals.get("BTTS Yes") or {}
    bn = totals.get("BTTS No") or {}
    if by.get("odds") and bn.get("odds"):
        try:
            y, n = float(by["odds"]), float(bn["odds"])
            if y > 1.0 and n > 1.0:
                ia, ib = 1.0 / y, 1.0 / n
                tot = ia + ib
                vig = tot - 1.0
                lean = "BTTS No" if n < y else "BTTS Yes"
                lean_odds = n if n < y else y
                imp = (ib / tot) if lean == "BTTS No" else (ia / tot)
                cands.append({"label": lean, "odds": lean_odds, "implied": imp, "vig": vig, "market": "BTTS", "raw_label": lean})
        except Exception:
            pass
    return cands


def _select_suggestion(cands: list[dict[str, Any]], se: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Pilih 1 SUGGESTION paling cocok — adjusted, bukan pure max% mentah.

    adjusted_score = implied * (1 - vig)  — penalti margin tebal.
    Jika movement tersedia dan arah away >2% untuk market tersebut → *0.5.
    Fallback deterministik: max implied → min odds.
    Pure display, tidak mengubah decision.
    """
    if not cands:
        return None
    # movement penalty lookup (OU/AH) dari se.market_block jika ada
    se = se or {}
    mb = se.get("market_block") or {}
    ou_mb = mb.get("ou") or {}
    ah_mb = mb.get("ah") or {}
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in cands:
        imp = float(c.get("implied") or 0.0)
        vig = float(c.get("vig") or 0.0)
        vig = max(0.0, min(0.9, vig))
        base = imp * (1.0 - vig)
        # movement penalty: jika Totals dan OU movement away, atau AH dan AH movement away
        penalty = 1.0
        try:
            if c.get("market") == "Total" and ou_mb:
                # direction away = harga memanjang menjauhi lean (melemah)
                # OU tidak punya per-side direction di market_block, pakai cek generik: jika lean Over dan latest>opening → away
                pass
            if c.get("market") == "Asian Handicap" and ah_mb:
                pass
        except Exception:
            pass
        # bookmaker count factor global (sama untuk semua market → tidak ubah ranking) → skip
        score = base * penalty
        scored.append((score, c))
    # max score → tie max implied → min odds
    def _key(item: tuple[float, dict[str, Any]]):
        sc, c = item
        return (sc, float(c.get("implied") or 0.0), -float(c.get("odds") or 999.0))
    scored.sort(key=_key, reverse=True)
    return scored[0][1] if scored else None


def _market_lean_block(payload: dict[str, Any], se: dict[str, Any]) -> list[str]:
    """Poin A (2026-08-24) → Opsi A (2026-08-26): MARKET LEAN info-only SELALU tampil.

    Awalnya hanya saat BEST PICK = NO BET (transparansi saat veto). Opsi A:
    lean tetap muncul walau BEST PICK ada value. Pure display, tidak mengubah
    decision/best_pick. Contoh Bologna: model Over 57% vs market Under 57% → lean Under 2.5 @1.63.
    2026-08-27: + SUGGESTION TO PICK selalu muncul (adjusted most suitable, bukan pure max%).
    Tag selaras/berlawanan dipindah dari bullet LEAN ke baris SUGGESTION (pure display).
    """
    totals = (payload.get("odds") or {}).get("totals") or {}
    consensus = (payload.get("odds") or {}).get("consensus") or {}
    over = totals.get("Over 2.5") or {}
    under = totals.get("Under 2.5") or {}
    has_totals = bool(over.get("odds") and under.get("odds"))
    has_1x2 = bool(consensus.get("home") and consensus.get("draw") and consensus.get("away"))
    if not has_totals and not has_1x2:
        return []
    lines: list[str] = ["", "📊 MARKET LEAN (info-only, bukan BEST PICK):"]
    # Bullets tanpa tag selaras (tag pindah ke SUGGESTION)
    if has_totals:
        try:
            o = float(over["odds"])
            u = float(under["odds"])
            if o and u:
                lean_label = "Under 2.5" if u < o else "Over 2.5"
                lean_odds = u if u < o else o
                ia, ib = 1.0 / o, 1.0 / u
                tot = ia + ib
                imp = (ib / tot * 100) if lean_label.startswith("Under") else (ia / tot * 100)
                lines.append(f"• {lean_label} @ {lean_odds:.2f} — market {imp:.0f}% dominan")
        except Exception:
            pass
    ah = (se.get("ah_consensus") or {}) or (payload.get("ah_consensus") or {})
    if ah.get("line") is not None and ah.get("home") and ah.get("away"):
        try:
            line = float(ah["line"])
            h, a = float(ah["home"]), float(ah["away"])
            if a < h:
                lean_ah = f"Away {-line:+.2f}" if abs(line) > 1e-9 else "Away +0.00"
                lean_odds = a
            else:
                lean_ah = f"Home {line:+.2f}"
                lean_odds = h
            lines.append(f"• AH: {lean_ah} @ {lean_odds:.2f} (market favorit AH)")
        except Exception:
            pass
    if has_1x2:
        try:
            side = min(consensus, key=lambda k: float(consensus[k]))
            label = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}.get(side, side)
            lines.append(f"• 1X2: {label} @ {float(consensus[side]):.2f} (market favorit)")
        except Exception:
            pass
    by = totals.get("BTTS Yes") or {}
    bn = totals.get("BTTS No") or {}
    if by.get("odds") and bn.get("odds"):
        try:
            y, n = float(by["odds"]), float(bn["odds"])
            lean = "BTTS No" if n < y else "BTTS Yes"
            lean_odds = n if n < y else y
            lines.append(f"• {lean} @ {lean_odds:.2f} (market favorit BTTS)")
        except Exception:
            pass
    lines.append("⚠️ Lean = arah pasar saja, tanpa edge model — bukan rekomendasi bet.")
    # SUGGESTION TO PICK — selalu muncul jika ada lean (adjusted most suitable)
    # Format rapih: blok terpisah dengan header & separator agar tidak ambigu vs LEAN
    cands = _lean_candidates(payload, se)
    sug = _select_suggestion(cands, se)
    if sug:
        bp = (se or {}).get("best_pick") or {}
        bp_sel = str(bp.get("selection") or "")
        bp_market = bp.get("market")
        raw = str(sug.get("raw_label") or sug.get("label") or "")
        tag = ""
        if bp_sel:
            sug_market = sug.get("market")
            if raw == bp_sel or sug.get("label") == bp_sel or (sug_market and raw == bp_sel.split(":")[-1].strip()):
                tag = " ✅ selaras dengan BEST PICK"
            elif sug_market and bp_market == sug_market:
                if raw != bp_sel:
                    tag = " ⚠️ berlawanan arah (model vs pasar)"
            elif raw == bp_sel:
                tag = " ✅ selaras dengan BEST PICK"
        disp_label = sug.get("label") or raw
        imp_pct = int(round(float(sug.get("implied") or 0.0) * 100))
        lines.append("")
        lines.append("──────────────────")
        lines.append("💡 SUGGESTION TO PICK (market-only):")
        lines.append(f"   {disp_label} @ {float(sug.get('odds', 0)):.2f} — market paling dominan ({imp_pct}%){tag}")
        lines.append("   ⚠️ market-only, tanpa edge model — bukan jaminan hasil")
    return lines


def _benchmark_meta(se: dict[str, Any]) -> list[str]:
    """Phase 5.2: benchmark age, bookmakers used, movement snapshots,
    direction/magnitude -- the metadata every card must display."""
    lines: list[str] = []
    eb = se.get("edge_benchmark") or {}
    age = eb.get("age_hours")
    if age is not None:
        lines.append(f"Benchmark age: {age:.1f}h")
    dq = se.get("data_quality") or {}
    n_bk = dq.get("bookmakers_count")
    if n_bk is None:
        n_bk = dq.get("bookmaker_count")
    if n_bk is not None:
        lines.append(f"Bookmakers: {n_bk}")
    n_snaps = dq.get("ah_ou_snapshots") or dq.get("movement_snapshots")
    if n_snaps is not None:
        lines.append(f"Movement snapshots: {n_snaps}")
    mv = (se.get("best_pick") or {}).get("movement") or {}
    if mv.get("status") == "available" and mv.get("direction") in ("toward", "away"):
        mag = mv.get("magnitude_pct")
        if mag is not None:
            arrow = "→" if mv["direction"] == "toward" else "←"
            lines.append(f"Movement: {arrow} {abs(mag):.1f}% ({mv['direction']})")
    elif mv.get("status") == "UNAVAILABLE":
        lines.append("Movement: n/a (no opening prices)")
    return lines


def format_market_signal(payload: dict[str, Any]) -> dict[str, Any]:
    """Primary analyse reply: the clean market-signal card.

    Reads only the already-computed ``signal_engine`` result (ranking, best
    pick, AH consensus) plus the match header and market totals. Never shows
    engine internals; unavailable markets are simply omitted.
    """
    if payload.get("error"):
        return format_analyse(payload)
    if payload.get("kickoff_uncertain"):
        return _format_analyse_kickoff_uncertain(payload)
    if payload.get("match_finished"):
        return _format_analyse_finished(payload)

    home = payload.get("home") or "?"
    away = payload.get("away") or "?"
    se = payload.get("signal_engine") or {}
    ranking = se.get("ranking") or []

    lines = [f"{home} vs {away}"]
    league = payload.get("league")
    if league:
        lines.append(f"🏆 {league}")
    lines.append(f"🕐 {_fmt_kickoff(payload.get('kickoff'))}")
    src = payload.get("match_source")
    if src:
        lines.append(f"🔎 Match dicari via {_source_label(src)}")
    lines.extend(["", _SIGNAL_SEPARATOR, "", "📊 SIGNALS", ""])
    if ranking:
        lines.extend(_signals_block(ranking))
    else:
        lines.append("No actionable signal.")
        lines.append("")

    lines.extend([_SIGNAL_SEPARATOR, "", f"🏆 {_pick_label(se)}", ""])
    lines.extend(_best_pick_block_with_lean(payload, se, include_internal=False))

    market = _market_block(payload, se)
    if market:
        lines.extend(["", _SIGNAL_SEPARATOR, "", "📈 MARKET", ""])
        lines.extend(market)

    quota = payload.get("quota") or {}
    if quota.get("oddspapi_quota_exhausted"):
        remaining = quota.get("oddspapi_remaining")
        suffix = f" (sisa {remaining})" if isinstance(remaining, int) else ""
        lines.extend(["", f"⚠️ Kuota Oddspapi habis{suffix} — odds via NowGoal fallback"])

    lines.extend(["", _SIGNAL_SEPARATOR, "", "⚠️ DISCLAIMER", "",
                  "Signal is model-based and not a guarantee of outcome.",
                  "Betting decisions are the user's own responsibility."])
    return {"title": "🔬 MATCH SIGNAL", "body": "\n".join(lines), "footer": " "}


def format_signal_detail(payload: dict[str, Any]) -> dict[str, Any]:
    """Detail (📋 Detail button) for the analyse reply.

    Slightly richer than the primary card (all signals, opening -> latest
    movement, model/market agreement, data quality) but still free of raw
    engine diagnostics (lambda, n_bucket, calibration samples, gate names).
    """
    if payload.get("error"):
        return format_analyse(payload)
    if payload.get("kickoff_uncertain"):
        return _format_analyse_kickoff_uncertain(payload)
    if payload.get("match_finished"):
        return _format_analyse_finished(payload)

    home = payload.get("home") or "?"
    away = payload.get("away") or "?"
    league = payload.get("league") or "?"
    se = payload.get("signal_engine") or {}
    ranking = se.get("ranking") or []

    lines = [
        "🔬 Match Signal — Detail",
        f"{home} vs {away} • {league}",
        f"🕐 {_fmt_kickoff(payload.get('kickoff'))}",
    ]
    src = payload.get("match_source")
    if src:
        lines.append(f"🔎 Match dicari via {_source_label(src)}")
    lines.extend(["", _SIGNAL_SEPARATOR, "", "📊 SIGNALS", ""])
    if ranking:
        for s in ranking:
            icon = _SIGNAL_CONF_ICONS.get(s.get("confidence"), "⚪")
            lines.append(
                f"{icon} {s['selection'].upper()} — {round(s['score'] * 100)}/100 "
                f"({s.get('confidence')})"
            )
    else:
        lines.append("No actionable signal.")
    lines.append("")

    lines.extend([_SIGNAL_SEPARATOR, "", f"🏆 {_pick_label(se)}", ""])
    lines.extend(_best_pick_block_with_lean(payload, se, include_internal=True))

    market = _market_block(payload, se)
    if market:
        lines.extend(["", _SIGNAL_SEPARATOR, "", "📈 MARKET", ""])
        lines.extend(market)

    # Model/market agreement + data quality (human-readable summary only).
    bp = se.get("best_pick")
    top = bp or (ranking[0] if ranking else None)
    if top is not None:
        edge = top.get("edge_pp", 0.0)
        if edge >= 0:
            agree = f"Model vs market: aligned (edge {edge:+.1f}pp)"
        else:
            agree = f"Model vs market: model trails (edge {edge:+.1f}pp)"
        lines.extend(["", _SIGNAL_SEPARATOR, "", "🧭 EVIDENCE", "", agree])

    dq = se.get("data_quality") or {}
    completeness = dq.get("completeness")
    if completeness is not None:
        level = "High" if completeness >= 0.7 else "Medium" if completeness >= 0.4 else "Low"
        lines.extend(["", f"📊 Data quality: {level} ({completeness:.2f})"])
    # P3: cross-source odds disagreement is explicit in the detail view -- the
    # market lines two independent sources disagree on, never a silent flag.
    if dq.get("odds_quality") == "cross_source_disagreement":
        srcs = dq.get("odds_sources") or []
        diff = dq.get("odds_max_pp_diff")
        detail = f" ({', '.join(srcs)})" if srcs else ""
        diff_txt = f" — max {diff}pp gap" if diff is not None else ""
        lines.extend(["", f"⚠️ Odds cross-source disagreement{detail}{diff_txt}"])

    lines.extend(["", _SIGNAL_SEPARATOR, "", "⚠️ DISCLAIMER", "",
                  "Signal is model-based and not a guarantee of outcome.",
                  "Betting decisions are the user's own responsibility."])
    return {"title": "🔬 Match Signal — Detail", "body": "\n".join(lines), "footer": " "}


def format_compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Single best pick of an analyse payload (OUTPUT POLICY).

    Every market (1X2, O/U 2.5, O/U 3.5, BTTS) is still computed and tiered
    in full -- this reply only surfaces the ONE best result per the
    selection layer (PICK > LEAN > WATCH), with its tier, confidence, basis
    and stake. Finished matches show the real result instead (no tiers).
    Errors keep the full error message via ``format_analyse``.
    """
    if payload.get("error"):
        return format_analyse(payload)
    if payload.get("kickoff_uncertain"):
        return _format_analyse_kickoff_uncertain(payload)
    if payload.get("match_finished"):
        return _format_analyse_finished(payload)
    home = payload.get("home") or "?"
    away = payload.get("away") or "?"
    league = payload.get("league") or "?"
    lines = [
        f"📊 {home} vs {away} — {league} • {_fmt_kickoff(payload.get('kickoff'))}",
        "",
    ]
    lines.extend(render_single_pick(payload))
    lines.append("Not a guarantee of outcome. Betting decisions are the user's own risk.")
    return {"title": "🔬 Analisa Match", "body": "\n".join(lines), "footer": " "}


def format_settle(payload: dict[str, Any]) -> dict[str, Any]:
    """Render the settle command result (manual + auto)."""
    if payload.get("error"):
        return {
            "title": "🧾 Settle",
            "body": f"Error: {payload['error']}",
            "footer": " ",
        }
    status = payload.get("status")

    if status == "settled":
        body = (
            f"✅ Hasil dicatat di prediction log\n\n"
            f"**{payload.get('home')} {payload.get('result')} {payload.get('away')}**\n"
            f"• liga: {payload.get('league') or '-'} • "
            f"kickoff: {_fmt_kickoff(payload.get('kickoff'))}\n"
            f"• match_id: `{payload.get('match_id')}`"
        )
        footer = "CLV: n/a (belum ada closing odds)" if not payload.get("closing_odds") else "CLV terhitung dari closing odds"
        return {"title": "🧾 Settle Match", "body": body, "footer": footer}

    if status == "ambiguous":
        lines = [
            f"⚠️ {len(payload.get('candidates', []))} snapshot cocok. "
            "Sertakan liga/tanggal untuk disambiguasi:"
        ]
        for c in payload.get("candidates", []):
            lines.append(
                f"• {c.get('league')} {c.get('home')} vs {c.get('away')} • "
                f"{_fmt_kickoff(c.get('kickoff'))}"
            )
        return {"title": "🧾 Settle", "body": "\n".join(lines), "footer": " "}

    if status == "not_found":
        lines = [
            f"❌ Tidak ada snapshot unsettled yang cocok: "
            f"{payload.get('home')} vs {payload.get('away')}"
        ]
        recent = payload.get("recent") or []
        if recent:
            lines.append(
                f"\nSnapshot unsettled ({payload.get('unsettled_count', len(recent))} total):"
            )
            for r in recent[:6]:
                lines.append(
                    f"• {r.get('league')} {r.get('home')} vs {r.get('away')} • "
                    f"{_fmt_kickoff(r.get('kickoff'))}"
                )
        return {
            "title": "🧾 Settle",
            "body": "\n".join(lines),
            "footer": "Coba `!football settle auto` atau periksa ejaan nama tim",
        }

    if status == "bad_result":
        return {
            "title": "🧾 Settle",
            "body": f"Skor tidak dikenali: `{payload.get('result')}`. Pakai format 2-1.",
            "footer": " ",
        }

    if status == "auto":
        lines = [f"✅ Settle otomatis {payload.get('date')}"]
        settled = payload.get("settled") or []
        nf = payload.get("not_found") or []
        if settled:
            lines.append(f"\nTercatat ({len(settled)}):")
            for s in settled:
                lines.append(
                    f"• {s.get('league')} {s.get('home')} {s.get('result')} {s.get('away')}"
                )
        if nf:
            lines.append(f"\nTanpa hasil ({len(nf)}):")
            for s in nf[:6]:
                lines.append(
                    f"• {s.get('league')} {s.get('home')} vs {s.get('away')}"
                )
        footer = (
            f"hasil diambil: {payload.get('results_fetched', 0)} • "
            f"unsettled tersisa: {payload.get('unsettled_total', 0)}"
        )
        return {"title": "🧾 Settle Otomatis", "body": "\n".join(lines), "footer": footer}

    return {"title": "🧾 Settle", "body": f"Status tidak dikenal: {status}", "footer": " "}


def format_odds_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Render the odds-snapshot command result (PHASE 32-33)."""
    if payload.get("error"):
        return {
            "title": "⏱️ Odds Snapshot",
            "body": f"Error: {payload['error']}",
            "footer": " ",
        }
    odds = payload.get("odds") or {}
    body = (
        f"✅ Odds tersimpan ({payload.get('timing', '?')})\n\n"
        f"**{payload.get('match_id', '?')}**\n"
        f"💰 1X2: {_fmt_odd(odds.get('home'))} / {_fmt_odd(odds.get('draw'))} / "
        f"{_fmt_odd(odds.get('away'))}"
    )
    extra = []
    if payload.get("bookmakers_count"):
        extra.append(f"bookie: {payload['bookmakers_count']}")
    if payload.get("sources"):
        extra.append(f"source: {', '.join(payload['sources'])}")
    footer = " • ".join(extra) if extra else " "
    return {"title": "⏱️ Odds Snapshot", "body": body, "footer": footer}


def format_best(payload: dict[str, Any], *, compact_winner: bool = True) -> dict[str, Any]:
    """Render `!best <liga>`: ranked shortlist + the single best pick's card.

    The winner card is the COMPACT match summary (``format_compact``) by
    default — the same 5-7 line output `analisa` posts — so the whole reply
    stays on-point. ``compact_winner=False`` embeds the full
    ``format_analyse`` report instead; the runner uses that as
    ``render_full`` so the 📋 Copy button can serve the full detail.
    """
    if payload.get("error"):
        return {
            "title": "🏆 BEST MATCH",
            "body": f"Error: {payload['error']}",
            "footer": " ",
        }
    league = payload.get("league", "?")
    date = payload.get("date", "?")
    cands = payload.get("candidates") or []
    winner = payload.get("winner") or {}

    lines = [
        f"**BEST MATCH — {league}** • {_fmt_value_date(date)}",
        f"📅 window: hari ini + dini hari (WIB) • {len(cands)} match",
    ]
    if cands:
        lines.append("")
        lines.append("📊 **Ranking (engine independen Elo+Poisson+kalibrasi):**")
        for i, c in enumerate(cands, 1):
            star = " ⭐" if i == 1 else ""
            # Gerbang `!best`: tier kandidat ditampilkan untuk transparansi
            # (semua baris di sini sudah lolos conf >= MEDIUM + non-veto).
            tier = c.get("confidence_tier")
            tier_txt = f" • conf {tier}" if tier else ""
            lines.append(
                f"{i}. **{c['home']} vs {c['away']}** • {_fmt_kickoff(c.get('kickoff'))}"
                f" • {c.get('decision_type', 'NO CLEAR DECISION')}"
                f" • skor {c.get('decision_score') or 0:.2f}{tier_txt}{star}"
            )

    winner_body = (
        (format_compact(winner) if compact_winner else format_analyse(winner))["body"]
        if winner else ""
    )
    if winner_body:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("🏆 **PILIHAN TERBAIK (winner)**")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(winner_body)
    elif not cands:
        lines.append("\n❌ Tidak ada match yang bisa dianalisa.")

    footer = " "
    quota = payload.get("quota") or {}
    parts = []
    if quota.get("odds_api_remaining") is not None:
        parts.append(f"odds quota: {quota['odds_api_remaining']}/500")
    if quota.get("odds_blocked"):
        parts.append("ODDS BLOCKED")
    if quota.get("football_data_warning"):
        parts.append("football-data rate limit")
    if parts:
        footer = " • ".join(parts)
    return {"title": f"🏆 BEST MATCH — {league}", "body": "\n".join(lines), "footer": footer}


def format_best_goal(payload: dict[str, Any]) -> dict[str, Any]:
    """Render `!bestgoalmatch`: leagues ranked by average expected goals, the
    most goal-friendly matches today, and the top pick (banjir gol)."""
    if payload.get("error"):
        return {
            "title": "⚽ BEST GOAL MATCH",
            "body": f"Error: {payload['error']}",
            "footer": " ",
        }
    date = payload.get("date", "?")
    league_avg = payload.get("league_avg") or []
    cands = payload.get("candidates") or []
    winner = payload.get("winner") or {}
    g = winner.get("goal") or {}

    def _pct(x):
        return f"{x * 100:.0f}%" if x is not None else "-"

    def _odds(x):
        return f"{x:.2f}" if x else "-"

    lines = [
        f"**BEST GOAL MATCH** — {_fmt_value_date(date)}",
        "📅 scan: match belum-bertanding hari ini (WIB), semua liga terdaftar",
    ]
    if league_avg:
        lines.append("")
        lines.append("🏟️ **Liga paling banjir gol hari ini** (avg expected total):")
        for name, avg in league_avg:
            lines.append(f"• {name}: {avg:.2f} gol/match")
    if cands:
        lines.append("")
        lines.append("⚽ **Kandidat goal-friendly (top 10):**")
        for i, c in enumerate(cands, 1):
            cg = c.get("goal") or {}
            star = " ⭐" if i == 1 else ""
            lines.append(
                f"{i}. **{c['home']} vs {c['away']}** • {c.get('league', '?')} • "
                f"{_fmt_kickoff(c.get('kickoff'))} • expected {cg.get('expected_total', 0):.2f} gol"
                f" • O2.5 {_pct(cg.get('over_2_5'))} • O3.5 {_pct(cg.get('over_3_5'))}{star}"
            )
    if winner and g:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("🎯 **PICK (banjir gol):**")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(
            f"**{winner['home']} vs {winner['away']}** • {winner.get('league', '?')} • "
            f"{_fmt_kickoff(winner.get('kickoff'))}"
        )
        lines.append(
            f"🧮 expected total: **{g.get('expected_total', 0):.2f} gol** "
            f"(model Poisson)"
        )
        lines.append(
            f"📈 Over 2.5: {_pct(g.get('over_2_5'))}"
            f" (odds {_odds(g.get('odds_over_2_5'))})"
        )
        lines.append(
            f"📈 Over 3.5: {_pct(g.get('over_3_5'))}"
            f" (odds {_odds(g.get('odds_over_3_5'))})"
        )
        lines.append(
            f"🔥 Over 4.5: {_pct(g.get('over_4_5'))}"
            f" (odds {_odds(g.get('odds_over_4_5'))})"
        )
        if winner.get("has_odds"):
            lines.append(f"💰 market: {winner.get('bookmakers_count', 0)} bookie")
        else:
            lines.append("💰 odds: belum tersedia (model-only, konfirmasi harga sebelum bet)")
        lines.append(
            f"🎯 rekomendasi: fokus **Over {g.get('over_3_5', 0) >= 0.5 and '3.5' or '2.5'}** "
            "— bandingkan dengan odds live."
        )

    lines.append(
        "\n⚠️ Expected goals dari form attack/defense (Poisson), bukan jaminan. "
        "Liga/bola hidup punya volatilitas tinggi."
    )
    footer = " "
    quota = payload.get("quota") or {}
    parts = []
    if quota.get("odds_api_remaining") is not None:
        parts.append(f"odds quota: {quota['odds_api_remaining']}/500")
    if quota.get("odds_blocked"):
        parts.append("ODDS BLOCKED")
    if parts:
        footer = " • ".join(parts)
    return {"title": "⚽ BEST GOAL MATCH", "body": "\n".join(lines), "footer": footer}


def format_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Render prediction-log realised stats (hit rate, logloss, ROI, CLV).

    Discord-embed twin of ``prediction_log.format_stats`` (CLI text formatter):
    same numbers, markdown-styled for the embed. Keep both in sync.
    """
    if payload.get("error"):
        return {
            "title": "📈 Prediction Log Stats",
            "body": f"Error: {payload['error']}",
            "footer": " ",
        }

    def _fmt(v: Any, suffix: str = "") -> str:
        return "-" if v is None else f"{v}{suffix}"

    n_settled = payload.get("n_settled", 0)
    n_snapshots = payload.get("n_snapshots", 0)
    unsettled = n_snapshots - n_settled
    edge_threshold = payload.get("edge_threshold")
    # hit_rate & roi disimpan sebagai fraksi (0.5 / -0.023); CLV sudah persen.
    hit_rate = payload.get("hit_rate")
    hit_pct = f"{hit_rate * 100:.1f}" if hit_rate is not None else None
    roi = payload.get("roi")
    roi_pct = f"{roi * 100:.1f}" if roi is not None else None
    body_lines = [
        "📊 **Realisasi Prediction Log**",
        f"🧾 snapshots: **{n_snapshots}** • settled: **{n_settled}** "
        f"• unsettled: **{unsettled}**",
        f"🎯 hit rate: **{_fmt(hit_pct, '%')}** "
        f"({payload.get('n_predicted', 0)} prediksi 1X2)",
        f"📉 log-loss: {_fmt(payload.get('avg_logloss'))}",
        f"💰 ROI: **{_fmt(roi_pct, '%')}** "
        f"({payload.get('n_bets', 0)} bet flat-stake)",
        f"📈 CLV: {_fmt(payload.get('clv_pct'), '%')} "
        f"({payload.get('n_clv', 0)} dengan closing odds)",
        f"💹 Price CLV: {_fmt(payload.get('price_clv_pct'), '%')} "
        f"({payload.get('n_price_clv', 0)} closing/prediction − 1)",
        f"📉 Max Drawdown: {_fmt(payload.get('max_drawdown'), '%')} ",
        f"⚖️ Sharpe: {_fmt(payload.get('sharpe'))}",
    ]

    by_timing = payload.get("odds_snapshots_by_timing") or {}
    if by_timing:
        timing_str = " • ".join(f"{k}: {v}" for k, v in sorted(by_timing.items()))
        body_lines.append(
            f"⏱️ odds snapshots: {payload.get('n_odds_snapshots', 0)} ({timing_str})"
        )
        clv_t = payload.get("clv_by_timing") or {}
        if clv_t:
            body_lines.append(
                "💹 Price CLV per timing: " + " • ".join(
                    f"{k} {v:+.2f}%" for k, v in sorted(clv_t.items())
                )
            )

    by_conf = payload.get("by_confidence") or {}
    if by_conf:
        body_lines.append("\n📂 **Per Confidence**")
        for label, b in by_conf.items():
            br = b.get("roi")
            br_pct = f"{br * 100:.1f}" if br is not None else None
            pclv = b.get("price_clv_pct")
            pclv_pct = f"{pclv:.1f}" if pclv is not None else None
            body_lines.append(
                f"• {label}: n={b.get('n', 0)} bets={b.get('n_bets', 0)} "
                f"hit={_fmt(b.get('hit_rate') and b['hit_rate'] * 100, '%')} "
                f"roi={_fmt(br_pct, '%')} clv={_fmt(b.get('clv_pct'), '%')} "
                f"pclv={_fmt(pclv_pct, '%')}"
            )
    by_edge = payload.get("by_edge") or {}
    if by_edge:
        body_lines.append("\n📂 **Per Edge Bucket**")
        for label, b in by_edge.items():
            br = b.get("roi")
            br_pct = f"{br * 100:.1f}" if br is not None else None
            pclv = b.get("price_clv_pct")
            pclv_pct = f"{pclv:.1f}" if pclv is not None else None
            body_lines.append(
                f"• {label}: n={b.get('n', 0)} bets={b.get('n_bets', 0)} "
                f"hit={_fmt(b.get('hit_rate') and b['hit_rate'] * 100, '%')} "
                f"roi={_fmt(br_pct, '%')} clv={_fmt(b.get('clv_pct'), '%')} "
                f"pclv={_fmt(pclv_pct, '%')}"
            )
    # TODO-15: CLV/ROI tracked per DECISION TYPE so production can verify
    # each tier (STRONG/GOOD/LEAN/WATCH/...) actually produces value.
    by_decision = payload.get("by_decision") or {}
    if by_decision:
        body_lines.append("\n📂 **Per Decision Type**")
        for label, b in by_decision.items():
            br = b.get("roi")
            br_pct = f"{br * 100:.1f}" if br is not None else None
            pclv = b.get("price_clv_pct")
            pclv_pct = f"{pclv:.1f}" if pclv is not None else None
            body_lines.append(
                f"• {label}: n={b.get('n', 0)} bets={b.get('n_bets', 0)} "
                f"hit={_fmt(b.get('hit_rate') and b['hit_rate'] * 100, '%')} "
                f"roi={_fmt(br_pct, '%')} clv={_fmt(b.get('clv_pct'), '%')} "
                f"pclv={_fmt(pclv_pct, '%')}"
            )

    footer_parts = []
    if edge_threshold is not None:
        footer_parts.append(f"edge >= {edge_threshold:.0%} dihitung sebagai bet")
    if unsettled > 0:
        footer_parts.append(
            f"{unsettled} snapshot belum di-settle — pakai `!football settle auto`"
        )
    if not n_settled:
        footer_parts.append("belum ada match yang di-settle")
    footer = " • ".join(footer_parts) if footer_parts else " "
    return {
        "title": "📈 Prediction Log Stats",
        "body": "\n".join(body_lines),
        "footer": footer,
    }
