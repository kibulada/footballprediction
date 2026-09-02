"""Accordion MATCH SIGNAL card for Discord (presentation layer only).

Replaces the plain-text ``━━━━━━━━━━━━━━━━━━`` MATCH SIGNAL card with an
embed that shows a compact summary by default (match, kickoff, BEST PICK,
one short reason line) and a "🔽 Lihat Hasil" button that expands the SAME
message in place to the full detail (all signals, market movement, full
reasoning) — and "🔼 Sembunyikan Hasil" collapses it back.

Design rules (Execution Spec: Accordion Toggle for Discord MATCH SIGNAL
Cards):

- Both embeds are built from ONE ``match_data`` structure — the runner's
  analyse payload (``result["raw"]``) — via the same format.py helpers the
  plain-text card uses, so summary and expanded views can never drift apart.
  This module never re-runs or modifies ``run_signal_engine`` /
  ``run_prediction_engine`` / scoring; it only re-renders their output.
- The toggle callback uses ``interaction.response.edit_message`` to update
  the existing message in place — no ``channel.send`` / ``send_message``
  anywhere, so a toggle can never spawn a duplicate card.
- Expand/collapse state belongs to each message's View instance only (no
  global store), so one user's toggle can never affect another card.
- ``timeout=600`` (the reference default). On timeout the button disables
  itself (and re-renders the message disabled when a message reference was
  attached at send time).
- Signal types are NOT hardcoded: whatever markets the engine produced
  (Asian Handicap, Over/Under, BTTS, ...) render from ``signal_engine``
  directly, so AH-style and BTTS-style matches both work.
"""
from __future__ import annotations

import secrets
from typing import Any

import discord

from .format import (
    _SIGNAL_CONF_ICONS,
    _best_pick_block,
    _display_best_pick,
    _fmt_kickoff,
    _high_risk_line,
    _market_block,
    _market_lean_block,
    _pick_label,
    _signal_why,
    _signals_block,
)
from .signal_engine import movement_narrative_flags

_COLLAPSED_LABEL = "🔽 Lihat Hasil"
_EXPANDED_LABEL = "🔼 Sembunyikan Hasil"
_CARD_TIMEOUT = 600  # reference default; buttons disable themselves on expiry

_DISCLAIMER = (
    "Signal is model-based and not a guarantee of outcome.\n"
    "Betting decisions are the user's own responsibility."
)

_GREEN = 0x2ECC71


def _header_lines(match_data: dict[str, Any]) -> list[str]:
    """Match header shared by both embeds (home vs away, league, kickoff)."""
    home = match_data.get("home") or "?"
    away = match_data.get("away") or "?"
    league = match_data.get("league")
    lines = [f"{home} vs {away}"]
    if league:
        lines.append(f"🏆 {league}")
    lines.append(f"🕐 {_fmt_kickoff(match_data.get('kickoff'))}")
    return lines


def _summary_best_pick_value(se: dict[str, Any], match_data: dict[str, Any] | None = None) -> str:
    """BEST PICK / TOP SIGNAL line for the compact embed (Phase 5.1: the
    label switches to TOP SIGNAL for uncalibrated leagues).

    Phase 5.4: uncalibrated leagues show explicit skip recommendation.
    Keputusan 2026-08-23: semua-kandidat-diveto -> rank #1 tetap tampil
    dengan label HIGH RISK (see ``format._display_best_pick``).
    Opsi A (2026-08-26): MARKET LEAN selalu tampil (BEST PICK ada pun tetap),
    plus tag selaras/berlawanan vs BEST PICK.
    """
    bp, risk_reason = _display_best_pick(se)
    if bp is None:
        # 20% lebih besar: pakai bold+underline agar standout tanpa literal ##
        base = "**__⚪ NO BET__**"
        if match_data is not None:
            lean = _market_lean_block(match_data, se)
            if lean:
                base += "\n\n" + "\n".join(l for l in lean if l)
        return base
    icon = _SIGNAL_CONF_ICONS.get(bp.get("confidence"), "🟢")
    odds = f" @ {bp['market_odds']:.2f}" if bp.get("market_odds") else ""
    score = round(bp.get("score", 0) * 100)
    why = _signal_why(bp, movement_narrative_flags(se))
    if se.get("edge_invalid"):
        why.append("Edge benchmark stale — edge: n/a")
    short = why[0] if why else "No single evidence group clearly dominant."
    _warn = ""
    if se.get("display_label") == "TOP SIGNAL":
        _warn = "\n⚠️ **LIGA TIDAK TERKALIBRASI — REKOMENDASI: SKIP**"
    if se.get("pick_tier") == "LEAN":
        # K5: a LEAN is shown, but never dressed up as a BEST PICK.
        # K7 (2026-09-02): with the reason (score / confidence / conviction).
        _reason = se.get("tier_reason")
        _warn += (
            f"\n📌 **LEAN — bukan BEST PICK ({_reason})**" if _reason
            else "\n📌 **LEAN — pick lemah (score/confidence rendah), bukan BEST PICK**"
        )
    # K7: the score is a composite, not a probability -- show both.
    _prob_txt = ""
    _mp = bp.get("model_prob")
    if _mp is not None:
        try:
            _prob_txt = f" • Peluang model: {float(_mp):.0%}"
        except (TypeError, ValueError):
            _prob_txt = ""
    # 20% lebih besar: bold+underline untuk BEST PICK (≈20% visual)
    out = (
        f"**__{icon} {bp['selection'].upper()}{odds}__**\n"
        f"Score: {score}/100 • Confidence: {bp.get('confidence')}{_prob_txt}\n"
        f"{short}{_warn}"
    )
    if risk_reason:
        out = f"{_high_risk_line(risk_reason)}\n{out}"
    # Opsi A: lean selalu tampil walau BEST PICK ada (keduanya)
    if match_data is not None:
        lean = _market_lean_block(match_data, se)
        if lean:
            out += "\n\n" + "\n".join(l for l in lean if l)
    return out


def build_summary_embed(match_data: dict[str, Any]) -> discord.Embed:
    """Compact default state: match, kickoff, BEST PICK/TOP SIGNAL + reason."""
    se = match_data.get("signal_engine") or {}
    embed = discord.Embed(
        title="🔬 MATCH SIGNAL",
        description="\n".join(_header_lines(match_data)),
        color=_GREEN,
    )
    embed.add_field(name=f"🏆 {_pick_label(se)}", value=_summary_best_pick_value(se, match_data), inline=False)
    embed.set_footer(text="Tekan tombol di bawah untuk detail lengkap")
    return embed


def build_expanded_embed(match_data: dict[str, Any]) -> discord.Embed:
    """Full detail state: summary + all signals + market + full reasoning."""
    se = match_data.get("signal_engine") or {}
    ranking = se.get("ranking") or []
    embed = discord.Embed(
        title="🔬 MATCH SIGNAL — Detail",
        description="\n".join(_header_lines(match_data)),
        color=_GREEN,
    )
    if ranking:
        signals_txt = "\n".join(_signals_block(ranking)).strip()
    else:
        signals_txt = "No actionable signal."
    embed.add_field(name="📊 SIGNALS", value=signals_txt, inline=False)
    # Opsi A (2026-08-26): lean selalu tampil (BEST PICK ada pun tetap) — sama kayak format.py
    best_lines = _best_pick_block(se)
    lean = _market_lean_block(match_data, se)
    if lean:
        best_lines.extend(lean)
    embed.add_field(
        name=f"🏆 {_pick_label(se)}",
        value="\n".join(best_lines).strip(),
        inline=False,
    )
    market = _market_block(match_data, se)
    if market:
        embed.add_field(name="📈 MARKET", value="\n".join(market).strip(), inline=False)
    embed.add_field(name="⚠️ DISCLAIMER", value=_DISCLAIMER, inline=False)
    return embed


class SignalCardView(discord.ui.View):
    """Per-message accordion toggle.

    ``match_data`` is the SAME payload both embeds are built from (the
    runner's analyse ``raw``). Expand state lives on this instance only —
    never a module/global store — so two cards in one channel are fully
    independent. On timeout the button disables itself; if a message
    reference was attached at send time, the message is re-rendered with the
    disabled button so the card visibly stops being interactive.
    """

    def __init__(
        self,
        match_data: dict[str, Any],
        *,
        timeout: float | None = _CARD_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout)
        self.match_data = match_data
        self.expanded = False
        self._message: discord.Message | None = None
        self._toggle = discord.ui.Button(
            label=_COLLAPSED_LABEL,
            style=discord.ButtonStyle.secondary,
            custom_id=f"football_card_{secrets.token_hex(6)}",
        )
        self._toggle.callback = self._on_toggle
        self.add_item(self._toggle)

    def attach_message(self, message: discord.Message) -> None:
        """Bind the sent message so on_timeout can re-render it disabled."""
        self._message = message

    def _current_embed(self) -> discord.Embed:
        return (
            build_expanded_embed(self.match_data)
            if self.expanded
            else build_summary_embed(self.match_data)
        )

    async def _on_toggle(self, interaction: discord.Interaction) -> None:
        # Flip state, embed and label ATOMICALLY in one edit_message call —
        # never a state where the label says one thing and the embed shows
        # the other, and never a new message (edit in place only).
        self.expanded = not self.expanded
        self._toggle.label = _EXPANDED_LABEL if self.expanded else _COLLAPSED_LABEL
        embed = self._current_embed()
        try:
            await interaction.response.edit_message(embed=embed, view=self)
        except discord.NotFound:
            # 10062 Unknown interaction: token expired (3s window) atau double-click.
            # Fallback: edit via message langsung (tidak butuh token).
            try:
                if interaction.message:
                    await interaction.message.edit(embed=embed, view=self)
                else:
                    await interaction.followup.edit_message(
                        message_id=interaction.message.id if interaction.message else None,
                        embed=embed, view=self
                    )
            except discord.HTTPException:
                pass
        except discord.HTTPException:
            pass

    async def on_timeout(self) -> None:
        # Expired card: disable the button so clicks can't silently go
        # nowhere, and re-render the message with the disabled button when we
        # have a reference. Best-effort — a failed edit must never crash the
        # timeout path.
        self._toggle.disabled = True
        if self._message is not None:
            try:
                await self._message.edit(view=self)
            except discord.HTTPException:
                pass
