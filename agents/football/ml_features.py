"""Rolling-window team statistics for ML features.

Ported from ProphitBet (https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor,
src/preprocessing/statistics.py, MIT license, author Vasileos Kochliaridis).
The StatisticsEngine is adapted to the bot's fixture-dict format and kept
ascending-by-date (the bot's chronological convention) instead of the
original descending output order.

Leakage rules (preserved from the original, audited by leakage_audit):
  - shift(1): a match never contributes to its own feature row.
  - groupby('Season'): windows are intra-season (teams reset each season).
  - rolling(min_periods=window): a team with fewer than ``window`` finished
    matches in the season yields NaN (dropped in training, flagged as
    "unavailable" in live prediction) -- never fabricated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BASIC_FEATURE_COLUMNS = [
    "HW", "AW", "HL", "AL",
    "HGF", "AGF", "HAGF",
    "HGA", "AGA", "HAGA",
    "HGD", "AGD", "HAGD",
    "HWGD", "AWGD", "HAWGD",
    "HLGD", "ALGD", "HALGD",
    "HW%", "HL%", "AW%", "AL%",
]

EXTENDED_FEATURE_COLUMNS = ["HSTF", "ASTF", "HCF", "ACF"]

# Rolling fouls/yellow-card totals (same leakage-safe _agg_prev window as the
# extended shots/corners columns). Computed only when the fixture source
# carries the raw per-match values (football-data.co.uk HF/AF/HY/AY columns,
# API-Football, or any provider exposing home/away fouls + yellow cards), so
# train and serve frames agree by construction.
CARDS_FEATURE_COLUMNS = ["HFCF", "AFCF", "HYCF", "AYCF"]

# Pre-match rolling xG columns carried by the augmented EPL cache built
# offline by the xG dataset builder. They are rolling averages over each team's
# finished matches STRICTLY before this fixture -- pre-match by construction,
# audited by leakage_audit -- so they are safe ML features.
XG_FEATURE_COLUMNS = [
    "HOME_XG_FOR", "HOME_XG_AGAINST", "AWAY_XG_FOR", "AWAY_XG_AGAINST",
]

# Columns that never enter the model (kept on the frame for joining back).
NON_TRAINABLE = ["Date", "Season", "Home", "Away", "HG", "AG", "Result"]

# Pre-match Elo ratings (replayed chronologically per league, see
# add_elo_features). Same signal family as the live Elo+Poisson ensemble.
ELO_FEATURE_COLUMNS = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF"]


def fixtures_to_frame(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize bot fixture dicts to the StatisticsEngine's input shape.

    Requires: date, home, away, home_goals, away_goals, season. Optional
    HST/AST/HC/AC columns enable the extended stats when the source carries
    them (football-data.co.uk caches currently do not).
    """
    rows = []
    for m in fixtures:
        hg = m.get("home_goals")
        ag = m.get("away_goals")
        date = m.get("date") or m.get("kickoff")
        home = m.get("home")
        away = m.get("away")
        if not date or not home or not away:
            continue
        try:
            hg = int(hg) if hg is not None else None
            ag = int(ag) if ag is not None else None
        except (ValueError, TypeError):
            continue
        # Upcoming fixtures (no goals yet) are kept as feature rows: their own
        # values are never counted (all windows shift(1)); training rows always
        # carry a result.
        if hg is None or ag is None:
            result: str | None = None
        elif hg > ag:
            result = "H"
        elif ag > hg:
            result = "A"
        else:
            result = "D"
        row: dict[str, Any] = {
            "Date": str(date)[:10],
            "Season": str(m.get("season") or ""),
            "Home": str(home).strip(),
            "Away": str(away).strip(),
            "HG": hg,
            "AG": ag,
            "Result": result,
        }
        for col in ("HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY"):
            v = m.get(col) or m.get(col.lower())
            if v is not None:
                try:
                    row[col] = float(v)
                except (ValueError, TypeError):
                    pass
        # snake_case aliases some sources use (e.g. API-Football exports)
        for col, key in (
            ("HF", "home_fouls"), ("AF", "away_fouls"),
            ("HY", "home_yellow_cards"), ("AY", "away_yellow_cards"),
        ):
            v = m.get(key)
            if v is not None and col not in row:
                try:
                    row[col] = float(v)
                except (ValueError, TypeError):
                    pass
        for col, key in (
            ("HOME_XG_FOR", "home_xg_for"),
            ("HOME_XG_AGAINST", "home_xg_against"),
            ("AWAY_XG_FOR", "away_xg_for"),
            ("AWAY_XG_AGAINST", "away_xg_against"),
        ):
            v = m.get(key)
            if v is not None:
                try:
                    row[col] = float(v)
                except (ValueError, TypeError):
                    pass
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    return df


def _agg_prev(series: pd.Series, window: int) -> pd.Series:
    """Sum of the previous ``window`` values, excluding the current row.

    Original: ``shift(1).rolling(window, min_periods=window).sum()``.
    """
    return series.shift(periods=1).rolling(window=window, min_periods=window).sum()


def _season_stats(df: pd.DataFrame, window: int, gd_margin: int) -> pd.DataFrame:
    """Compute all basic (and, when present, extended) stats for one season."""
    result = df["Result"]

    # --- win/loss counts over the team's last N home/away matches ---
    hw = df[["Home", "Result"]].copy()
    hw["V"] = result.eq("H").astype(int)
    df["HW"] = hw.groupby("Home")["V"].transform(lambda s: _agg_prev(s, window))

    aw = df[["Away", "Result"]].copy()
    aw["V"] = result.eq("A").astype(int)
    df["AW"] = aw.groupby("Away")["V"].transform(lambda s: _agg_prev(s, window))

    hl = df[["Home", "Result"]].copy()
    hl["V"] = result.eq("A").astype(int)
    df["HL"] = hl.groupby("Home")["V"].transform(lambda s: _agg_prev(s, window))

    al = df[["Away", "Result"]].copy()
    al["V"] = result.eq("H").astype(int)
    df["AL"] = al.groupby("Away")["V"].transform(lambda s: _agg_prev(s, window))

    # --- goals for/against ---
    df["HGF"] = df.groupby("Home")["HG"].transform(lambda s: _agg_prev(s, window))
    df["AGF"] = df.groupby("Away")["AG"].transform(lambda s: _agg_prev(s, window))
    df["HGA"] = df.groupby("Home")["AG"].transform(lambda s: _agg_prev(s, window))
    df["AGA"] = df.groupby("Away")["HG"].transform(lambda s: _agg_prev(s, window))
    df["HAGF"] = df["HGF"] - df["AGF"]
    df["HAGA"] = df["HGA"] - df["AGA"]
    df["HGD"] = df["HGF"] - df["HGA"]
    df["AGD"] = df["AGF"] - df["AGA"]
    df["HAGD"] = df["HGD"] - df["AGD"]

    # --- outstanding win/loss margins (|GD| >= gd_margin) ---
    # fillna(0): upcoming-fixture rows carry no goals and must not crash the
    # margin arithmetic (their own margin is never counted -- shift(1)).
    hg, ag = df["HG"].fillna(0), df["AG"].fillna(0)
    home_win_margin = (hg - ag).ge(gd_margin).astype(int)
    away_win_margin = (ag - hg).ge(gd_margin).astype(int)
    df["HWGD"] = df.groupby("Home")["HG"].transform(
        lambda s: _agg_prev(pd.Series(home_win_margin.to_numpy(), index=df.index), window)
    )
    df["AWGD"] = df.groupby("Away")["AG"].transform(
        lambda s: _agg_prev(pd.Series(away_win_margin.to_numpy(), index=df.index), window)
    )
    df["HAWGD"] = df["HWGD"] - df["AWGD"]
    df["HLGD"] = df.groupby("Home")["HG"].transform(
        lambda s: _agg_prev(pd.Series(away_win_margin.to_numpy(), index=df.index), window)
    )
    df["ALGD"] = df.groupby("Away")["AG"].transform(
        lambda s: _agg_prev(pd.Series(home_win_margin.to_numpy(), index=df.index), window)
    )
    df["HALGD"] = df["HLGD"] - df["ALGD"]

    # --- cumulative win/loss rates since season start (excluding current) ---
    df["HW%"] = _cumulative_rate(df, team_col="Home", result_col="Result", outcome="H")
    df["HL%"] = _cumulative_rate(df, team_col="Home", result_col="Result", outcome="A")
    df["AW%"] = _cumulative_rate(df, team_col="Away", result_col="Result", outcome="A")
    df["AL%"] = _cumulative_rate(df, team_col="Away", result_col="Result", outcome="H")

    # --- extended stats (only when the raw columns exist) ---
    if "HST" in df.columns:
        df["HSTF"] = df.groupby("Home")["HST"].transform(lambda s: _agg_prev(s, window))
        df["ASTF"] = df.groupby("Away")["AST"].transform(lambda s: _agg_prev(s, window))
    if "HC" in df.columns:
        df["HCF"] = df.groupby("Home")["HC"].transform(lambda s: _agg_prev(s, window))
        df["ACF"] = df.groupby("Away")["AC"].transform(lambda s: _agg_prev(s, window))
    if "HF" in df.columns:
        df["HFCF"] = df.groupby("Home")["HF"].transform(lambda s: _agg_prev(s, window))
        df["AFCF"] = df.groupby("Away")["AF"].transform(lambda s: _agg_prev(s, window))
    if "HY" in df.columns:
        df["HYCF"] = df.groupby("Home")["HY"].transform(lambda s: _agg_prev(s, window))
        df["AYCF"] = df.groupby("Away")["AY"].transform(lambda s: _agg_prev(s, window))

    return df


def _cumulative_rate(
    df: pd.DataFrame, *, team_col: str, result_col: str, outcome: str
) -> pd.Series:
    """Rate of ``outcome`` results before the current match, since season start."""
    tmp = df[[team_col, result_col]].copy()
    tmp["hit"] = tmp[result_col].eq(outcome).astype(float)
    tmp["cum_hit"] = tmp.groupby(team_col)["hit"].cumsum() - tmp["hit"]
    tmp["cum_count"] = tmp.groupby(team_col).cumcount()
    return (tmp["cum_hit"] / tmp["cum_count"] * 100.0).round(decimals=1)


def build_feature_frame(
    fixtures: list[dict[str, Any]], *, window: int = 5, gd_margin: int = 2
) -> pd.DataFrame:
    """Full pipeline: normalize fixtures -> per-season rolling stats.

    Result is ascending by date with feature columns. Rows for teams with
    < ``window`` finished matches in their season carry NaN (training drops
    them; live prediction treats them as "model unavailable").
    """
    df = fixtures_to_frame(fixtures)
    if df.empty:
        return df
    if not df["Date"].is_monotonic_increasing:
        df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    # Manual per-season loop: pandas >= 3.0 groupby.apply drops the grouping
    # column (include_groups=False default) and can drop rows; a plain concat
    # keeps every column and row untouched.
    parts = [
        _season_stats(group, window=window, gd_margin=gd_margin)
        for _, group in df.groupby("Season", sort=False)
    ]
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def available_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns actually computed on this frame (basic + any extended)."""
    cols = [c for c in BASIC_FEATURE_COLUMNS if c in df.columns]
    cols += [c for c in EXTENDED_FEATURE_COLUMNS if c in df.columns]
    cols += [c for c in CARDS_FEATURE_COLUMNS if c in df.columns]
    cols += [c for c in XG_FEATURE_COLUMNS if c in df.columns]
    cols += [c for c in ELO_FEATURE_COLUMNS if c in df.columns]
    return cols


def add_elo_features(
    df: pd.DataFrame, elo_cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Attach pre-match Elo ratings via a chronological per-league replay.

    Leakage-safe: the rating is read BEFORE ``elo.update(result)`` for each
    match, mirroring backtest.py's walk-forward replay and the live engine's
    ``elo.json`` state. Requires a ``league`` column (set by load_frames).
    """
    from .elo import EloModel

    cfg = dict(elo_cfg or {})
    cfg.pop("file", None)
    initial = float(cfg.get("initial_rating", 1500.0))
    home_ratings = np.full(len(df), initial)
    away_ratings = np.full(len(df), initial)
    for league, g in df.groupby("league", sort=False):
        elo = EloModel(**cfg)
        for i in g.index:
            row = df.loc[i]
            home_ratings[i] = elo.rating(row["Home"])
            away_ratings[i] = elo.rating(row["Away"])
            hg, ag = row.get("HG"), row.get("AG")
            if hg is None or ag is None or pd.isna(hg) or pd.isna(ag):
                # upcoming fixture row (no result yet): read ratings, no update
                continue
            elo.update(
                row["Home"], row["Away"], int(hg), int(ag), persist=False
            )
    df["HOME_ELO"] = home_ratings
    df["AWAY_ELO"] = away_ratings
    df["ELO_DIFF"] = home_ratings - away_ratings
    return df
