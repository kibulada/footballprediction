"""Calibration + SignalScorer v2.

Calibrator: fits ``p' = sigmoid(a + b * logit(p))`` on backtest residuals via
ordinary least squares (pure Python, no deps). Persists params locally.

SignalScorer v2: evaluates FIVE named components separately instead of one
opaque integer:
  - data_completeness  (0..1)  what pre-match data exists (odds/form/xG/H2H)
  - model_agreement    (0..1)  model-vs-model and model-vs-market agreement
  - market_edge        (ppts)  best margin-free edge across 1X2 selections
  - calibration_quality(0..1)  from the Calibrator (0 = not yet validated)
  - signal_strength    (0..100) documented weighted combination of the above
  - confidence         (0..1)  documented weighted combination (label basis)

The components stay accessible individually on PredictionResult.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .context import MatchContext


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def expected_calibration_error(probs: list[float], outcomes: list[int], bins: int = 10) -> float:
    """ECE over the given (probability, binary outcome) pairs."""
    if not probs or len(probs) != len(outcomes):
        return float("nan")
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    count = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idxs = [i for i, p in enumerate(probs) if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if not idxs:
            continue
        avg_p = sum(probs[i] for i in idxs) / len(idxs)
        freq = sum(outcomes[i] for i in idxs) / len(idxs)
        total += len(idxs) * abs(avg_p - freq)
        count += len(idxs)
    return (total / count) if count else float("nan")


class Calibrator:
    """Log-odds linear calibration fitted on historical (prob, outcome) pairs."""

    def __init__(self, path: str | Path | None = None, min_samples: int = 200) -> None:
        self.path = Path(path) if path else None
        self.min_samples = min_samples
        self.a = 0.0
        self.b = 1.0
        self.samples = 0
        self.ece = None
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.a = float(payload.get("a", 0.0))
            self.b = float(payload.get("b", 1.0))
            self.samples = int(payload.get("samples", 0))
            self.ece = payload.get("ece")
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            self.a, self.b, self.samples, self.ece = 0.0, 1.0, 0, None

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"a": self.a, "b": self.b, "samples": self.samples, "ece": self.ece},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ---- fit / apply ---------------------------------------------------
    def fit(self, probs: list[float], outcomes: list[Any]) -> None:
        """Fit P(outcome=1) = sigmoid(a + b * logit(p)) by IRLS logistic
        regression (Newton-Raphson, pure Python). Requires both 0 and 1
        outcomes present; fractional targets are accepted for tests."""
        if len(probs) < 10 or len(probs) != len(outcomes):
            return
        xs = [_logit(p) for p in probs]
        ys = [float(o) for o in outcomes]
        if len(set(round(y) for y in ys)) < 2 and not any(0.0 < y < 1.0 for y in ys):
            return  # degenerate target (all same outcome)
        a, b = 0.0, 1.0
        for _ in range(50):
            mu = [_sigmoid(a + b * x) for x in xs]
            g_a = sum(y - m for y, m in zip(ys, mu))
            g_b = sum((y - m) * x for y, m, x in zip(ys, mu, xs))
            w = [m * (1.0 - m) for m in mu]
            h_aa = sum(w)
            h_ab = sum(wi * x for wi, x in zip(w, xs))
            h_bb = sum(wi * x * x for wi, x in zip(w, xs))
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            da = (g_a * h_bb - g_b * h_ab) / det
            db = (g_b * h_aa - g_a * h_ab) / det
            a2, b2 = a + da, b + db
            if not (abs(a2) < 1e6 and abs(b2) < 1e6):
                break  # perfect separation: coefficients diverge; keep last stable
            a, b = a2, b2
            if abs(da) < 1e-9 and abs(db) < 1e-9:
                break
        self.a, self.b = a, b
        self.samples = len(probs)
        self.ece = expected_calibration_error(
            [self.apply(p) for p in probs], [1 if y > 0.5 else 0 for y in ys]
        )
        self._save()

    def apply(self, p: float) -> float:
        if self.samples < self.min_samples:
            return p
        return _sigmoid(self.a + self.b * _logit(p))

    def refresh_from_pairs(
        self,
        pairs: list[tuple[float, Any]],
        min_samples: int = 200,
    ) -> dict[str, Any]:
        """Re-fit calibration from explicit (p, outcome) pairs.

        Guards: skips (status="skipped") when fewer than ``min_samples``
        pairs exist so a good snapshot is never overwritten with noise. On a
        refit the previous snapshot is first backed up to ``<path>.bak``
        (only when a refit is actually going to happen), and the refit is
        KEPT only when its ECE is not worse than the existing snapshot -- a
        regression returns status="kept" with the old params restored.
        """
        if len(pairs) < max(10, int(min_samples)):
            return {
                "status": "skipped",
                "pairs": len(pairs),
                "min_samples": int(min_samples),
                "reason": "settled sample too small for a meaningful re-fit",
                "backup": None,
            }
        backup = None
        if self.path is not None and self.path.exists():
            try:
                backup = str(self.path.with_suffix(self.path.suffix + ".bak"))
                Path(backup).write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                backup = None
        old = (self.a, self.b, self.samples, self.ece)
        probs = [p for p, _ in pairs]
        outcomes = [y for _, y in pairs]
        self.fit(probs, outcomes)
        if (
            old[3] is not None
            and self.ece is not None
            and self.ece > old[3] + 1e-6
        ):
            # Regression guard: the new fit is measurably worse than the
            # existing snapshot -- keep the old one (never a downgrade).
            self.a, self.b, self.samples, self.ece = old
            self._save()
            return {
                "status": "kept",
                "pairs": len(pairs),
                "reason": "refit ECE worse than existing snapshot (no regression)",
                "ece": round(self.ece, 4) if self.ece is not None else None,
                "backup": backup,
            }
        return {
            "status": "refit",
            "pairs": len(pairs),
            "a": round(self.a, 4),
            "b": round(self.b, 4),
            "ece": round(self.ece, 4) if self.ece is not None else None,
            "samples": self.samples,
            "backup": backup,
        }

    def refresh_from_log(
        self,
        log_path: str | Path,
        min_samples: int = 200,
    ) -> dict[str, Any]:
        """Re-fit calibration from the LIVE prediction log (TODO-02).

        Uses ONLY pre-match probabilities of settled outcomes
        (``prediction_log.calibration_pairs``): each settled snapshot
        contributes one (p_side, outcome) pair per 1X2 side, so the fit is
        strictly prediction-then-outcome with no leakage by construction.
        Delegates to ``refresh_from_pairs`` (shared with the per-league
        refresh, which groups the same pairs by league key).
        """
        from .prediction_log import calibration_pairs

        return self.refresh_from_pairs(calibration_pairs(log_path), min_samples=min_samples)

    def quality(self) -> dict[str, Any]:
        """{quality 0..1, ece, samples}; quality 0 until validated by backtest."""
        if self.samples < self.min_samples:
            return {"quality": 0.0, "ece": self.ece, "samples": self.samples}
        ece = self.ece if self.ece is not None else float("nan")
        if math.isnan(ece):
            return {"quality": 0.0, "ece": None, "samples": self.samples}
        quality = max(0.0, min(1.0, 1.0 - ece * 2.0))
        return {"quality": quality, "ece": round(ece, 4), "samples": self.samples}


class SignalScorer:
    """Component-based signal scorer (see module docstring for the breakdown)."""

    # Component weights for the OVERALL signal strength (documented).
    W_SIGNAL = {
        "completeness": 0.40,
        "agreement": 0.30,
        "calibration": 0.30,
    }
    # Confidence weights (label basis). S26: probability magnitude and
    # separation now enter confidence via ``decisiveness`` (0.15), taken from
    # calibration's previous share (0.50 -> 0.35). All overridable per
    # instance so config/football.json -> models.signal_scorer can tune them.
    W_CONF = {
        "completeness": 0.20,
        "agreement": 0.30,
        "calibration": 0.35,
        "decisiveness": 0.15,
    }

    def __init__(self, conf_weights: dict[str, float] | None = None) -> None:
        self.w_conf = dict(self.W_CONF)
        if conf_weights:
            self.w_conf.update({k: v for k, v in conf_weights.items() if v is not None})

    def components(
        self,
        *,
        ctx: MatchContext,
        ensemble_models: list[str],
        model_vs_market: float | None,
        model_vs_model: float | None,
        calibration_quality: float,
        market_edge: dict[str, float],
        p1x2: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        completeness = self._completeness(ctx)
        agreement = self._agreement(model_vs_market, model_vs_model)
        edge_pct = max((abs(v) for v in market_edge.values()), default=0.0)
        # S26: decisiveness = how far the most likely outcome is above both
        # the uniform baseline (magnitude) and the next-best outcome
        # (separation). Missing probabilities -> neutral 0.
        decisiveness = _decisiveness(p1x2) if p1x2 else 0.0
        signal = 100.0 * (
            self.W_SIGNAL["completeness"] * completeness
            + self.W_SIGNAL["agreement"] * agreement
            + self.W_SIGNAL["calibration"] * min(1.0, calibration_quality)
        )
        confidence = (
            self.w_conf["completeness"] * completeness
            + self.w_conf["agreement"] * agreement
            + self.w_conf["calibration"] * min(1.0, calibration_quality)
            + self.w_conf["decisiveness"] * decisiveness
        )
        level, cap = completeness_level(completeness)
        # Data completeness caps the FINAL confidence label (PHASE 3): missing
        # data must reduce the ceiling, not inflate it. 90-100% -> HIGH allowed;
        # 70-89% -> MEDIUM/HIGH; 50-69% -> max MEDIUM; below 50% -> LOW only.
        confidence = min(confidence, cap)
        return {
            "data_completeness": round(completeness, 3),
            "data_completeness_level": level,
            "model_agreement": round(agreement, 3),
            "decisiveness": round(decisiveness, 3),
            "market_edge_pct": round(edge_pct, 2),
            "calibration_quality": round(min(1.0, calibration_quality), 3),
            "signal_strength": round(signal),
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
        }

    @staticmethod
    def _completeness(ctx: MatchContext) -> float:
        # Phase 5: form sequence and gf/ga both derive from the SAME
        # recent-match feed. Scoring them as two independent 0.20 components
        # double-counted one source. Merge them into a single 0.40 component
        # so one feed can only ever contribute 0.40, not 0.20 + 0.20.
        # Weights still sum to 1.0: odds 0.25, recent-form 0.40, xG 0.20, H2H 0.15.
        parts = []
        parts.append(0.25 if ctx.has_odds else 0.0)
        parts.append(
            0.40
            if (ctx.has_attack_defense or (ctx.home_form and ctx.away_form))
            else 0.0
        )
        parts.append(0.20 if ctx.has_xg else 0.0)
        parts.append(0.15 if (ctx.h2h and any(ctx.h2h.values())) else 0.0)
        return sum(parts)

    @staticmethod
    def _agreement(
        model_vs_market: float | None,
        model_vs_model: float | None,
    ) -> float:
        vals = [v for v in (model_vs_market, model_vs_model) if v is not None]
        if not vals:
            return 0.0
        return max(0.0, min(1.0, sum(vals) / len(vals)))


def _decisiveness(p1x2: dict[str, float]) -> float:
    """0..1: how decisive is the most likely 1X2 outcome.

    magnitude: (p_max - 1/3) / (2/3)  -- how far above uniform
    separation: p_max - second_best   -- how far ahead of the field
    Both are needed (S26): 55/27/18 is more decisive than 52/27/21 even
    though the winner is the same.
    """
    vals = [p for p in (p1x2.get("home"), p1x2.get("draw"), p1x2.get("away")) if p is not None]
    if len(vals) < 3 or max(vals) <= 0:
        return 0.0
    pmax = max(vals)
    second = sorted(vals, reverse=True)[1]
    magnitude = max(0.0, min(1.0, (pmax - 1.0 / 3.0) / (2.0 / 3.0)))
    separation = max(0.0, min(1.0, pmax - second))
    return round(0.5 * magnitude + 0.5 * separation, 3)


def completeness_level(completeness: float) -> tuple[str, float]:
    """Map a 0..1 completeness score to (level, confidence_cap).

    PHASE 3 data-quality rules: missing data must lower the confidence
    ceiling, so a HIGH-confidence label is impossible without rich data.

      completeness >= 0.90 -> HIGH allowed, cap 1.0
      0.70 .. 0.89        -> MEDIUM/HIGH, cap 1.0
      0.50 .. 0.69        -> maximum MEDIUM, cap 0.69
      < 0.50              -> LOW only, cap 0.49
    """
    if completeness >= 0.90:
        return "HIGH", 1.0
    if completeness >= 0.70:
        return "MEDIUM/HIGH", 1.0
    if completeness >= 0.50:
        return "MEDIUM", 0.69
    return "LOW", 0.49


# Manual overrides for leagues whose auto-slug doesn't match the calibration filename
_LEAGUE_SLUG_OVERRIDES: dict[str, str] = {
    "uefa europa league": "uel",
    "uefa europa league qualification": "uel",
    "europa league": "uel",
    "europa league qualification": "uel",
    "uel": "uel",
    "uefa conference league": "uecl",
    "uefa conference league qualification": "uecl",
    "conference league": "uecl",
    "conference league qualification": "uecl",
    "uecl": "uecl",
    "liga 1": "liga-1-indonesia",
    "liga 1 indonesia": "liga-1-indonesia",
    "indonesian liga 1": "liga-1-indonesia",
    "psl": "south-african-premiership",
    "south african premiership": "south-african-premiership",
    "premier soccer league": "south-african-premiership",
    "primera a": "primera-a",
    "categoría primera a": "primera-a",
    "liga betplay": "liga-betplay",
    "liga betplay dimayor": "liga-betplay",
    "division profesional": "division-profesional-bolivia",
    "división profesional": "division-profesional-bolivia",
    "liga pro serie a": "ligapro-serie-a",
    "ligapro serie a": "ligapro-serie-a",
    "croatian football league": "croatian-football-league",
    "hnl": "croatian-football-league",
    "prva hnl": "croatian-football-league",
    "serbian superliga": "serbian-superliga",
    "super liga serbia": "serbian-superliga",
    "prva liga serbia": "serbian-superliga",
    "stars league": "qatar-stars-league",
    "qatar stars league": "qatar-stars-league",
    "uae pro league": "uae-pro-league",
    "uae league": "uae-pro-league",
    "persian gulf pro league": "persian-gulf-pro-league",
    "iran pro league": "persian-gulf-pro-league",
    "ligue professionnelle 1": "ligue-professionnelle-1-tunisia",
    "ligue 1 tunisia": "ligue-professionnelle-1-tunisia",
    "tunisia ligue 1": "ligue-professionnelle-1-tunisia",
    "ligue 1 algeria": "algeria-ligue-1",
    "algerian ligue 1": "algeria-ligue-1",
    "botola pro": "botola-pro",
    "botola": "botola-pro",
    "morocco botola pro": "botola-pro",
    "czech first league": "czech-first-league",
    "1. liga czech": "czech-first-league",
    "fortuna liga": "czech-first-league",
    "liga i": "liga-i",
    "superliga romania": "liga-i",
    "romanian superliga": "liga-i",
}


def league_slug(league: str) -> str:
    key = (league or "").lower().strip()
    if key in _LEAGUE_SLUG_OVERRIDES:
        return _LEAGUE_SLUG_OVERRIDES[key]
    return "".join(c if c.isalnum() else "-" for c in key).strip("-")


def league_calibrator(
    league: str,
    cfg: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> Calibrator | None:
    """Per-league calibrator, or None when the league has no valid fit.

    Phase 5: the global ``calibration.json`` was fitted on EPL only. It must
    NOT be applied to leagues with different scoring/draw distributions.
    ``league_calibrator`` returns:

      - the global calibrator for EPL (its file IS the EPL fit),
      - a ``cache/football/calibration_<slug>.json`` fit for any other league
        whose samples >= ``league_min_samples``,
      - ``None`` otherwise (caller forces MARKET PRIOR — never a foreign
        calibration).

    Returns None (not a silent identity calibrator) so the caller can
    distinguish "calibrated here" from "no calibration available".
    """
    cal_cfg = ((cfg or {}).get("models") or {}).get("calibration") or {}
    min_samples = int(cal_cfg.get("league_min_samples", 400))
    always_cal = cal_cfg.get("always_calibrated", []) or []
    base = Path(cal_cfg.get("file", "cache/football/calibration.json"))
    if root is not None:
        base = Path(root) / base
    slug = league_slug(league)
    path = base if slug == "epl" else base.parent / f"calibration_{slug}.json"
    if not path.exists():
        return None
    cal = Calibrator(path=path, min_samples=min_samples)
    # Always calibrated leagues: skip sample threshold (config override)
    if league in always_cal or league.lower().strip() in [a.lower().strip() for a in always_cal]:
        return cal
    if cal.samples >= min_samples:
        return cal
    return None


def refresh_leagues_from_log(
    log_path: str | Path,
    cal_dir: str | Path,
    min_samples: int = 200,
) -> dict[str, Any]:
    """Re-fit EVERY league's calibration from the LIVE prediction log.

    D2 (2026-08-17): per-league calibration files were previously only
    seeded from football-data.co.uk history (``seed-league``), so dynamic
    leagues (``dyn:...`` keys) could NEVER leave the ``uncalibrated_league``
    cap -- no fit existed and no path could create one. This refreshes
    ``calibration_<slug>.json`` for every league present in the live log
    (``calibration_pairs_by_league``) with enough settled pairs, using the
    same skip/backup/regression-guard discipline as the global refresh.

    Returns {league_key: refresh_report} for every league that had enough
    pairs; leagues without samples are absent (not an error).
    """
    from .prediction_log import calibration_pairs_by_league

    grouped = calibration_pairs_by_league(log_path)
    reports: dict[str, Any] = {}
    for league_key, pairs in grouped.items():
        if len(pairs) < max(10, int(min_samples)):
            continue
        slug = league_slug(league_key)
        path = Path(cal_dir) / f"calibration_{slug}.json"
        cal = Calibrator(path=path, min_samples=int(min_samples))
        reports[league_key] = cal.refresh_from_pairs(pairs, min_samples=int(min_samples))
    return reports

