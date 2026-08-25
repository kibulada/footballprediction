"""Time helpers: UTC internally, Asia/Jakarta (WIB) for display/day boundaries.

Policy: every timestamp inside the prediction engine is ISO-8601 UTC.
WIB (UTC+7) is used ONLY for the user-facing "match day" boundary and for
Discord display.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

WIB = timezone(timedelta(hours=7))


def utc_now_iso() -> str:
    """Current time as ISO-8601 UTC string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def wib_today_iso() -> str:
    """Today's WIB calendar date as 'YYYY-MM-DD'."""
    return datetime.now(WIB).date().isoformat()


def wib_date_from_iso(iso: str | None) -> str | None:
    """Convert an ISO-8601 timestamp (UTC or offset) to the WIB calendar date.

    Returns 'YYYY-MM-DD' or None if the input can't be parsed.
    """
    if not iso:
        return None
    try:
        cleaned = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(WIB).date().isoformat()
    except (ValueError, TypeError):
        return None


def utc_range_for_wib_date(wib_date: str) -> tuple[str, str]:
    """UTC [dateFrom, dateTo] that fully covers one WIB calendar day.

    WIB 00:00 == previous UTC day 17:00, so the range spans two UTC dates.
    Callers fetch the range then filter each row by ``wib_date_from_iso``.
    """
    d = datetime.strptime(wib_date, "%Y-%m-%d").date()
    start_wib = datetime(d.year, d.month, d.day, tzinfo=WIB)
    end_wib = start_wib + timedelta(days=1)
    start_utc = start_wib.astimezone(timezone.utc).date().isoformat()
    end_utc = end_wib.astimezone(timezone.utc).date().isoformat()
    return start_utc, end_utc


def kickoff_sort_key(fx: dict) -> tuple:
    """Walk-forward-safe fixture sort key (TODO-05).

    Sorts by (date, has_kickoff_time, kickoff_utc):
      - matches with a parseable kickoff time are ordered by that time;
      - same-day matches WITHOUT a time (e.g. FBref date-only rows) sort
        AFTER timed matches on the same date, keeping their original relative
        order (stable) -- the documented residual caveat: with no kickoff
        time available they cannot be ordered, so an earlier-kickoff result
        must never leak into a later-kickoff match's features through the
        replay. Matches with NO date at all sort last.
    """
    d = str(fx.get("date") or "")
    if not d:
        return ("9999-12-31", 1, datetime.min.replace(tzinfo=timezone.utc))
    date_part = d[:10]
    cleaned = d.replace("Z", "+00:00") if d.endswith("Z") else d
    # A bare 'YYYY-MM-DD' string parses fine as midnight -- it must be
    # treated as UNKNOWN kickoff time (sorted after timed same-day matches),
    # NOT as 00:00, or date-only fixtures would leak ahead of real kickoffs.
    if len(cleaned) == 10:
        try:
            datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            return (date_part, 1, datetime.min.replace(tzinfo=timezone.utc))
        return (date_part, 1, datetime.min.replace(tzinfo=timezone.utc))
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return (date_part, 1, datetime.min.replace(tzinfo=timezone.utc))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return (date_part, 0, dt)
