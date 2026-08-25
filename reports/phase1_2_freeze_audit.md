# PHASE 1–2 REPORT — Formal Baseline Freeze + Automated Anti-Leakage Audit

**Scope:** MASTER PROMPT Phase 1 (frozen baseline) + Phase 2 (strict temporal
leakage control). No prediction/decision/production code was modified — only
audit tooling and additive metrics.

**Status:** ✅ Baseline re-frozen, byte-identical metrics to the previous
freeze (no regression). ✅ Anti-leakage audit **PASS** on EPL 2022–2026
(1,520 matches).

---

## 1. Files changed / added

| File | Type | Change |
|---|---|---|
| `agents/football/validate.py` | modified | ADDITIVE: per-bet `net_series` tracking + `max_drawdown` + `max_losing_streak` in the betting backtest. All pre-existing metrics (`log_loss`, `brier`, `ece`, `hit_rate`, `roi`, `bets`) byte-identical. |
| `agents/football/leakage_audit.py` | **new** | Automated anti-leakage audit: 25-field MatchContext provenance registry, independent-reference predict-before-update invariant (content-based, catches update-before-predict even under the 5-match rolling-window cap), 2-pass determinism / input_hash check, pipeline-equivalence check vs `validate`, CLV-scope static scan, same-day caveat reporting. CLI: `python -m agents.football.leakage_audit --fixtures …`. |
| `agents/football/baseline_freeze.py` | **new** | Formal re-freeze: model/feature version, data snapshot (sha256, counts, seasons, date range), full walk-forward metrics, calibration params, betting risk metrics, CLV=n/a (honest), regression reference. CLI: `python -m agents.football.baseline_freeze --fixtures …`. |
| `tests/test_leakage_audit.py` | **new** | 10 tests: provenance coverage, CLV scope, verdict, determinism, bug detection (incl. capped regime), false-positive guards, empty input. |
| `tests/test_baseline_freeze.py` | **new** | 8 tests: snapshot metadata, metrics shape, CLV=n/a, calibration-cache safety, regression reference, report markers. |
| `reports/baseline_freeze_v2.json` / `.md` | artifact | The frozen regression reference. |
| `reports/leakage_audit_epl.json` | artifact | Audit report for EPL 2022–2026. |

Nothing in `baseline/` (the immutable frozen baseline) was touched.

---

## 2. Baseline re-freeze — results (EPL 2022–2026, 1,520 matches, chronological walk-forward)

Data snapshot: `cache/football/epl_fixtures_2022_2026.json` (sha256 `249ad217…`,
4 × 380 matches, 2022-2023 … 2025-2026, carries historical 1X2 odds).

| model | n | LogLoss | Brier | ECE | Hit% | ROI (edge≥2%) | bets | max drawdown | losing streak |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1520 | 1.0971 | 0.6495 | 0.0196 | 44.4% | −7.5% | 805 | −95.94 | 15 |
| elo | 1520 | 0.9914 | 0.5921 | 0.0205 | 52.3% | −2.7% | 882 | −65.83 | 8 |
| poisson | 1505 | 1.0341 | 0.6213 | 0.0157 | 49.2% | −0.5% | 765 | −59.21 | 26 |
| dc | 1505 | 1.0363 | 0.6223 | 0.0174 | 48.9% | −0.1% | 740 | −55.72 | 27 |
| **ensemble** | **1520** | **0.9886** | **0.5901** | **0.0110** | **53.1%** | **−1.9%** | **623** | **−55.84** | **10** |
| market | 1520 | 0.9652 | 0.5739 | 0.0084 | 54.5% | — | — | — | — |

- ✅ **Identical to the previous freeze** (LL 0.9886, hit 53.1%, ROI −1.9%, 623 bets,
  ECE 0.0110) → the regression reference is valid; additive metrics changed nothing.
- **New risk metrics** (this phase): ensemble max drawdown **−55.84 stake units**,
  longest losing streak **10 bets**. Honest reading: the historical betting
  backtest loses money and the market benchmark is not beaten on log-loss
  (0/4 seasons) — no bookmaker-beating claim is made anywhere.
- Per-season ensemble log-loss: 0.9888 / 0.9503 / 0.9933 / 1.0220 (stable, no
  season collapses).
- Calibration fit: a=0.0082, b=1.0129, samples=4560, calibrated ECE 0.0104
  (in-sample fit, labelled as such).
- **CLV = n/a** (dataset has no closing odds) — explicitly reported, never
  fabricated. CLV remains a live-only evaluation metric via `prediction_log`.

---

## 3. Anti-leakage audit — results (EPL 2022–2026)

**VERDICT: PASS** — all invariants hold, 0 violations.

| Check | Result |
|---|---|
| Feature provenance | ✅ 25/25 `MatchContext` fields documented (source, pre-match availability, update frequency, missing behaviour, leakage risk); a missing doc now fails the audit → new features cannot enter the context silently |
| predict_before_update | ✅ 0 violations — each team's form deque exactly equals the independent prior-match reference (content-based, immune to the 5-match window cap) |
| determinism_input_hash | ✅ 1,520 matches, 0 hash differences across 2 passes — same snapshot ⇒ same prediction |
| pipeline_equivalence | ✅ audit replay LL 0.9886 == production `validate` LL 0.9886 (1,520 matches) — the audited context IS the production context |
| CLV scope | ✅ closing-odds references exist only in evaluation/logging/audit modules (`prediction_log`, `settler`, `runner`, `format`, audit tooling) — never in context/models/features |

Same-day caveat (honest, documented): 310 dates with >1 match (1,060 matches).
FBref provides no kickoff time, so same-day order is arbitrary; in principle a
later-kickoff match processed earlier could expose its result to an
earlier-kickoff match's context. This is a historical-dataset limitation —
the live bot orders by kickoff timestamp — and is disclosed, not hidden.
Mitigation (future phase): re-order historical matches by kickoff time.

---

## 4. Regression reference (for Phase 15+ candidates)

Any candidate change must beat **out-of-sample** (walk-forward, untouched
future data):
- ensemble log-loss **0.9886**, hit **53.1%**
- ensemble ROI **−1.9%** (623 bets), drawdown **−55.84**, streak **10**
- calibration a=0.0082, b=1.0129 (ECE 0.0110 raw)

If a candidate does not robustly improve these numbers across chronological
periods → **KEEP THE EXISTING MODEL**.

---

## 5. Validation

- Full test suite: **374 passed** (was 357 → +17 new tests, no regressions).
- Re-freeze reproducible: freeze v2 == baseline_freeze.md metrics exactly.
- No production/cache files touched by the freeze (calibration cache safe
  unless `--calibration-out` is explicitly passed).

## 6. Remaining limitations (unchanged, re-affirmed)

1. Same-day arbitrary ordering in historical replay (kickoff time unavailable).
2. xG never backtested (no historical xG dataset) — `xg_weight` only active
   when pre-match xG exists.
3. Calibration trained on EPL, applied to all leagues.
4. Totals 3.5 / BTTS not historically validated for the decision layer.
5. Market benchmark (LL 0.9652) still beats the ensemble (0.9886) — honest.
