"""ML live inference (hybrid decision support).

Trained models (ml_train.py) produce independent 1X2 / Over-Under 2.5
probabilities for upcoming fixtures. Feature construction is the ProphitBet
port (ml_features.py): intra-season rolling windows + a per-league Elo replay
-- all strictly pre-match (audited).

Honesty contract: when a team has fewer than ``window`` finished matches in
the current season (or the league has no cached history), the feature row is
NaN and ``predict`` returns None -- the caller must fall back to the existing
Elo+Poisson engine (never fabricate a feature).
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ml_features as mf
from .backtest import load_fixtures_from_json
from .ml_train import _load_elo_config, discover_fixture_files


def season_label(date: str) -> str:
    """'2026-03-15' -> '2025-2026' (season label the caches use)."""
    y, m = int(date[:4]), int(date[5:7])
    return f"{y - 1}-{y}" if m < 7 else f"{y}-{y + 1}"


def load_model(target: str, models_dir: str | Path):
    """(pipeline, features_meta) or (None, None) when the model is missing."""
    d = Path(models_dir) / target
    if not (d / "pipeline.pkl").exists():
        return None, None
    with open(d / "pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)
    meta = json.loads((d / "features.json").read_text(encoding="utf-8"))
    return pipe, meta


class MlPredictor:
    """Loads cached history + models once, predicts per fixture."""

    def __init__(
        self,
        models_dir: str | Path,
        *,
        window: int = 5,
        gd_margin: int = 2,
        elo_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._window = window
        self._gd_margin = gd_margin
        self._elo_cfg = elo_cfg if elo_cfg is not None else _load_elo_config()
        self._models = {
            t: load_model(t, models_dir) for t in ("result", "over-under")
        }
        self._history = self._load_history()

    def _load_history(self) -> dict[str, list[dict[str, Any]]]:
        hist: dict[str, list[dict[str, Any]]] = {}
        for path in discover_fixture_files():
            try:
                fixtures = load_fixtures_from_json(path)
            except Exception:  # noqa: BLE001 -- one bad cache must not kill predict
                continue
            for f in fixtures:
                hist.setdefault(f.get("league") or "", []).append(f)
        return hist

    def available(self, target: str) -> bool:
        return self._models.get(target, (None, None))[1] is not None

    def _build_rows(
        self, entries: list[tuple[str, str, str, str]]
    ) -> dict[tuple[str, str, str, str], pd.Series]:
        """Feature row per (league, home, away, date).

        One feature frame + one Elo replay per (league, date) group -- all
        fixtures sharing a matchday are built together (a per-match rebuild
        is O(n^2) across a season).
        """
        by_group: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for league, home, away, date in entries:
            by_group.setdefault((league, str(date)), []).append((home, away))

        rows: dict[tuple[str, str, str, str], pd.Series] = {}
        for (league, date), pairs in by_group.items():
            hist = [
                f for f in self._history.get(league, [])
                if str(f["date"]) < str(date)
            ]
            synths = [
                {
                    "date": str(date), "home": h, "away": a,
                    "home_goals": None, "away_goals": None,
                    "league": league, "season": season_label(str(date)),
                }
                for h, a in pairs
            ]
            df = mf.build_feature_frame(
                hist + synths, window=self._window, gd_margin=self._gd_margin
            )
            if df.empty:
                continue
            df["league"] = league
            df = mf.add_elo_features(df, self._elo_cfg)
            for i, (home, away) in enumerate(pairs):
                rows[(league, home, away, str(date))] = df.iloc[-len(pairs) + i]
        return rows

    @staticmethod
    def _row_vector(row: pd.Series, columns: list[str]) -> np.ndarray | None:
        try:
            vals = [float(row[c]) for c in columns]
        except (KeyError, TypeError, ValueError):
            return None
        if any(np.isnan(v) for v in vals):
            return None
        return np.array([vals], dtype=np.float32)

    def predict_matches(
        self, entries: list[tuple[str, str, str, str]]
    ) -> dict[tuple[str, str, str, str], dict[str, Any]]:
        """Batch 1X2 + O/U probabilities for (league, home, away, date) entries."""
        rows = self._build_rows(entries)
        out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for key in entries:
            row = rows.get(key)
            if row is None:
                continue
            item: dict[str, Any] = {"1x2": None, "over": None}
            for target in ("result", "over-under"):
                pipe, meta = self._models.get(target, (None, None))
                if pipe is None:
                    continue
                x = self._row_vector(row, meta.get("columns") or [])
                if x is None:
                    continue
                probs = pipe.predict_proba(x)[0]
                if target == "result":
                    item["1x2"] = {
                        "home": float(probs[0]), "draw": float(probs[1]),
                        "away": float(probs[2]), "model": meta.get("model"),
                    }
                else:
                    item["over"] = {
                        "over": float(probs[1]), "under": float(probs[0]),
                        "model": meta.get("model"),
                    }
            if item["1x2"] is not None or item["over"] is not None:
                out[key] = item
        return out

    def predict_1x2(self, league: str, home: str, away: str, date: str) -> dict[str, Any] | None:
        """{home, draw, away} probabilities or None (model/features unavailable)."""
        item = self.predict_matches([(league, home, away, date)]).get(
            (league, home, away, str(date))
        )
        return item["1x2"] if item and item["1x2"] else None

    def predict_over_under(self, league: str, home: str, away: str, date: str) -> dict[str, Any] | None:
        """{over: P(>=2.5)} or None."""
        item = self.predict_matches([(league, home, away, date)]).get(
            (league, home, away, str(date))
        )
        return item["over"] if item and item["over"] else None


async def predict_fixtures(
    date: str | None = None,
    leagues: list[str] | None = None,
    *,
    models_dir: str | Path = "cache/football/models",
    window: int = 5,
    gd_margin: int = 2,
) -> dict[str, Any]:
    """ML predictions for every scheduled match on ``date`` (default tomorrow WIB).

    Requires FOOTBALL_DATA_KEY (schedule source) and trained models. Matches
    whose league has no cached history / teams with thin season form return
    ``status: "unavailable"`` instead of a fabricated probability.
    """
    from .detect_match import _code_to_league_key
    from .football_data import FootballDataClient

    wib = timezone(timedelta(hours=7))
    date = date or (datetime.now(wib).date() + timedelta(days=1)).isoformat()
    fd = FootballDataClient(os.getenv("FOOTBALL_DATA_KEY", ""), throttle_seconds=6.0)
    # Fetch a small range: football-data's free tier returns nothing for a
    # single-day dateFrom==dateTo window, so pull a few days and filter the
    # requested date here.
    date_to = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=3)).date().isoformat()
    rows = await fd.fetch_scheduled_matches_by_date(date, date_to) or []
    rows = [m for m in rows if (m.get("kickoff") or "")[:10] == date]
    code_map = _code_to_league_key()
    predictor = MlPredictor(models_dir, window=window, gd_margin=gd_margin)

    entries: list[tuple[str, str, str, str]] = []
    for m in rows:
        league = code_map.get(m.get("competition"))
        if not league:
            continue
        if leagues and league not in leagues:
            continue
        kickoff = (m.get("kickoff") or "")[:10]
        entries.append((league, m.get("home"), m.get("away"), kickoff))
    results = predictor.predict_matches(entries)

    per_match: list[dict[str, Any]] = []
    skipped: list[str] = []
    for league, home, away, kickoff in entries:
        key = (league, home, away, kickoff)
        item: dict[str, Any] = {
            "league": league, "home": home, "away": away, "kickoff": kickoff,
            "ml_1x2": None, "ml_over25": None, "status": "unavailable",
        }
        res = results.get(key)
        if res:
            if res["1x2"]:
                item["ml_1x2"] = {
                    k: round(float(v), 4) for k, v in res["1x2"].items() if k != "model"
                }
                item["ml_model"] = res["1x2"].get("model")
                item["status"] = "predicted"
            if res["over"]:
                item["ml_over25"] = {
                    k: round(float(v), 4) for k, v in res["over"].items() if k != "model"
                }
                item["ml_model"] = res["over"].get("model")
                item["status"] = "predicted"
        if item["status"] == "unavailable":
            skipped.append(f"{league}: {home} vs {away}")
        per_match.append(item)

    return {
        "date": date,
        "n_scheduled": len(per_match),
        "n_predicted": sum(1 for e in per_match if e["status"] == "predicted"),
        "n_unavailable": sum(1 for e in per_match if e["status"] == "unavailable"),
        "matches": per_match,
        "unavailable": skipped[:20],
    }
