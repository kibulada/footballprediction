"""Pure formatting helpers for the Discord embed formatter (TODO-17).

Extracted verbatim from ``format.py`` so the output is byte-identical; the
formatter re-imports these names, keeping every call site unchanged. No
behavior change -- this is a structural decomposition only.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    WIB = ZoneInfo("Asia/Jakarta")
except Exception:  # noqa: BLE001 -- fallback offset keeps display stable
    WIB = timezone(timedelta(hours=7))


_MONTHS_ID = ("", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des")

# English month abbreviations for the VALUE MATCH header ("12 AUG 2026").
_MONTHS_EN = ("", "JAN", "FEB", "MAR", "APR", "MEI", "JUN",
              "JUL", "AGU", "SEP", "OKT", "NOV", "DES")


def _fmt_value_date(date: str) -> str:
    """'2026-08-12' -> '12 AGU 2026' (Indonesian-style English month)."""
    try:
        y, m, d = date.split("-")
        return f"{int(d):02d} {_MONTHS_EN[int(m)]} {y}"
    except (ValueError, IndexError):
        return date or "?"


def _fmt_odd(value: float | None) -> str:
    if not value or value <= 0:
        return "-"
    return f"{value:.2f}"


def _fmt_kickoff(iso: str) -> str:
    """Convert ISO8601 (UTC or with offset) → Asia/Jakarta HH:MM [WIB][ • DD Mon YYYY].

    Returns "-" if input missing. Falls back to a sanitized string if parsing fails.
    """
    if not iso:
        return "-"
    try:
        cleaned = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(WIB)
        today_wib = datetime.now(WIB).date()
        target_date = local.date()
        delta_days = (target_date - today_wib).days
        time_part = local.strftime("%H:%M")
        if delta_days == 0:
            return f"{time_part} WIB"
        if delta_days == 1:
            return f"{time_part} WIB • besok"
        date_part = f"{local.day:02d} {_MONTHS_ID[local.month]} {local.year}"
        return f"{time_part} WIB • {date_part}"
    except (IndexError, ValueError):
        return iso


def _fmt_stat(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value}"


def _fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value}%"
    return str(value)
