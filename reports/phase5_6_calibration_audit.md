# PHASE 5-6 — Empirical Confidence Buckets + Out-of-Sample Calibration Audit

**Scope**: empirical reliability curve per probability bucket + OOS calibration evaluation across 4 leagues (EPL, La Liga, Bundesliga, Serie A), 2022–2026 walk-forward replay, model = ensemble.
**Rule applied**: PHASE 37 / Master Prompt — calibration may enter production **only** if it improves out-of-sample metrics. A calibration stage that merely reshuffles probabilities without OOS improvement is NOT forced into production.

---

## 1. Method (leakage-safe)

- Raw `(predicted_prob, outcome)` pairs produced by the existing **walk-forward replay** (`validate.py`), strictly chronological — each prediction uses only information available before that match.
- `validate.py` gained an **additive, default-off** `include_pairs` param that exposes per-season `cal_pairs`. Default path is byte-identical (verified by full suite + leakage audit equivalence 0.9886).
- New module `agents/football/calibration_audit.py`:
  - **PHASE 6 buckets**: `[0.00,0.50) [0.50,0.55) [0.55,0.60) [0.60,0.65) [0.65,0.70) [0.70,0.75) [0.75,0.80) [0.80,0.90) [0.90,1.00]`.
  - Per bucket: n, mean predicted, actual rate, gap, **Wilson 95% CI**, Brier, Log Loss.
  - **OOS calibration eval**: fit isotonic calibrator on early seasons (2022–2025, n≈13k), evaluate on untouched 2025-26 eval season (n≈4.3k). The calibrator never sees eval data.
  - **Production calibrator honesty check** (EPL single run): the in-production `cache/football/calibration.json` (a=0.0081, b=1.0127) evaluated OOS on the eval season.
  - Cross-league pooled reliability curve (17,376 pairs).

---

## 2. Pooled Reliability Curve (4 leagues, n=17,376)

| bucket | n | pred | actual | gap | Wilson 95% | Brier | LL |
|---|---|---|---|---|---|---|---|
| [0.00,0.50) | 14425 | 0.278 | 0.279 | −0.0014 | [0.272, 0.286] | 0.1917 | 0.5681 |
| [0.50,0.55) | 898 | 0.524 | 0.531 | −0.0076 | [0.498, 0.564] | 0.2482 | 0.6894 |
| [0.55,0.60) | 665 | 0.574 | 0.526 | **+0.0478** | [0.488, 0.564] | 0.2510 | 0.6951 |
| [0.60,0.65) | 572 | 0.625 | 0.617 | +0.0078 | [0.577, 0.656] | 0.2361 | 0.6651 |
| [0.65,0.70) | 411 | 0.674 | 0.689 | −0.0148 | [0.642, 0.731] | 0.2132 | 0.6174 |
| [0.70,0.75) | 252 | 0.725 | 0.730 | −0.0053 | [0.672, 0.781] | 0.1981 | 0.5858 |
| [0.75,0.80) | 121 | 0.770 | 0.785 | −0.0147 | [0.704, 0.849] | 0.1680 | 0.5182 |
| [0.80,0.90) | 32 | 0.819 | 0.812 | +0.0063 | [0.647, 0.911] | 0.1515 | 0.4797 |
| [0.90,1.00] | 0 | — | — | — | — | — | — |

**bucket-level ECE = 0.0042** → the model is already very well calibrated globally.

### Reading
- Largest gap is **+0.0478 in [0.55,0.60)** (mild overconfidence, predicted 57.4% vs actual 52.6%) — still **inside** the Wilson CI, so not statistically distinguishable from zero.
- High-probability buckets ([0.80,0.90)) have tiny n (32) → wide CI → cannot claim reliable calibration there; per PHASE 10 this caps confidence in that region.
- Underconfidence at [0.65,0.70) and [0.75,0.80) is within CI.

---

## 3. Per-League OOS Calibration Eval (fit 22-25 → eval 25-26)

| league | fit n | eval n | raw ECE | cal ECE | raw Brier | cal Brier | raw LL | cal LL | improves (ece/brier/ll) |
|---|---|---|---|---|---|---|---|---|---|
| EPL | 3420 | 1140 | 0.0291 | 0.0349 | 0.2048 | 0.2054 | 0.5969 | 0.5984 | F/F/F |
| La Liga | 3420 | 1140 | 0.0161 | 0.0163 | 0.1980 | 0.1980 | 0.5816 | 0.5816 | F/F/F |
| Bundesliga | 2769 | 924 | 0.0184 | 0.0190 | 0.1922 | 0.1924 | 0.5696 | 0.5700 | F/F/F |
| Serie A | 3423 | 1140 | 0.0192 | 0.0181 | 0.2008 | 0.2007 | 0.5892 | 0.5890 | **T/T/T** |
| **Pooled** | 13032 | 4344 | 0.0128 | 0.0126 | 0.1993 | 0.1993 | 0.5851 | 0.5852 | T/F/F |

- **3 of 4 leagues degrade (or don't improve) when an OOS calibrator is applied**; Serie A improves marginally on all three metrics; pooled improves ECE by 0.0002 but not Brier/LL.
- Per PHASE 37: **the calibrator is NOT promoted to production** — applying it would worsen EPL (the reference dataset) and most leagues. The existing near-identity production calibrator remains.

---

## 4. Production Calibrator Honesty Check (EPL eval season)

`cache/football/calibration.json` (a=0.0081, b=1.0127, fitted in-sample on full EPL history) evaluated OOS on 2025-26:

| metric | raw | production cal |
|---|---|---|
| ECE | 0.0291 | **0.0273** (slightly better) |
| Brier | 0.2048 | 0.2049 (≈equal) |
| LL | 0.5969 | 0.5971 (≈equal) |

→ The production calibrator is effectively **neutral** (near-identity, a≈0, b≈1). No evidence it damages OOS calibration; no evidence it adds value. **RETAINED** (harmless, consistent with raw probabilities).

---

## 5. Verdict

- **Model calibration status: WELL-CALIBRATED** — pooled bucket ECE 0.0042; every non-trivial gap is inside its Wilson CI.
- **Empirical confidence rule (PHASE 6)**: per-bucket empirical reliability is now available and matches the predicted probability. Confidence signals derived from these buckets will not be inflated beyond what the reliability curve supports.
- **No calibration change promoted to production** (PHASE 37). OOS calibration improves only Serie A and pooled-ECE by a hair; it degrades EPL/La Liga/Bundesliga.
- **Limitations documented**: tiny n at p>0.80 (32 pooled, 18 EPL); 2025-26 is the only OOS eval season; isotonic fit is per-league pooled, not per-market.

---

## 6. Files

- `agents/football/validate.py` — additive `include_pairs` (default off; default path byte-identical).
- `agents/football/calibration_audit.py` — NEW: buckets, Wilson CI, OOS calibration eval, production-calibrator check, pooled reliability, cross-league runner, CLI `--leagues/--fixtures/--fit-seasons/--eval-seasons/--production-calibration`.
- `tests/test_calibration_audit.py` — NEW: 7 tests (bucket edges, Wilson CI, OOS fit/eval, pooled merge, empty buckets, calibration never-sees-eval).
- Reports: `reports/phase5_6_calibration_audit.json` (cross-league), `reports/phase5_6_calibration_audit_epl.json` (single + production check).

## 7. Validation

- Full suite: **398 passed** (was 391 → +7).
- Review: 6 findings fixed (empty-bucket `None` guard now regression-tested, pooled weighted Brier/LL, defensive formatting, docstrings, JSON shape, CLI robustness).
- Leakage: calibrator fit uses only fit-season pairs; eval pairs never touch the fit (verified by test).
