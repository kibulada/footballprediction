# FRESH AUDIT — Hermes Football Prediction Model (2026-08-12)

Scope: full pipeline audit (data → features → models → decision → evaluation),
following the master audit prompt checklist. This audit builds on the earlier
phases (0–20, plan TODOs 1–18) and reports **what is already sound**, the
**genuine defects found this pass**, the **fixes implemented**, and the
**honest validation evidence** for each.

All numbers below were produced by the local harnesses in this repository
from the cached EPL 2022-2026 dataset (1,520 matches with real historical
odds). Nothing is fabricated; every claim is reproducible with the commands
listed.

---

## 1. Repository Summary

### Folder structure

| Path | Purpose |
|---|---|
| `agents/football/` | Core package: data ingestion, features, models, decision, harnesses |
| `agents/football/runner.py` | CLI entry (top / analisa / settle / audit / cache-odds / calib-refresh) |
| `agents/football/multi_source.py` | Provider chain (flashscore → football-data → thesportsdb → understat) |
| `agents/football/odds_fetcher.py` / `oddspapi.py` | The Odds API + secondary odds source |
| `agents/football/predictor.py` | Odds-derived Poisson picks (Model A, reference only) |
| `agents/football/elo.py`, `models.py` | Elo, feature-Poisson (Dixon-Coles), ensemble (Model B, independent) |
| `agents/football/calibration.py` | Log-odds calibrator + SignalScorer |
| `agents/football/decision.py`, `model_gates.py` | Decision engine (STRONG/GOOD/LEAN/NO BET/…) + hard gates |
| `agents/football/analyse.py` | Orchestrates one match analysis (live path) |
| `agents/football/backtest.py`, `validate.py` | Walk-forward harnesses |
| `agents/football/ablation.py`, `calibration_audit.py`, `market_audit.py`, `leakage_audit.py`, `decision_validation.py` | Evidence harnesses |
| `agents/football/prediction_log.py` | Immutable JSONL prediction log + settle/CLV |
| `config/football.json` | Production model & decision config |
| `cache/football/` | Seeded Elo (255 teams), calibrator, fixture caches (EPL w/ odds, 4 leagues results), understat xG |
| `reports/` | Phase reports (0–20, plan implementation, audits) |
| `tests/` | 565 tests (pytest) |

### Data pipeline (as it actually works today)

1. **Ingestion**: live — multi-source provider chain per league; historical —
   cached football-data.co.uk (EPL with odds), FBref (LaLiga/Bundesliga/Serie A
   results), understat (real xG, big-5).
2. **Features (all pre-match by construction)**: consensus odds (median),
   margin-free implied probabilities, recent form (last-5, time-decay
   weighted), attack/defense averages, rolling xG for/against, H2H, Elo
   ratings (seeded 255 teams, K-adaptive), rest days (off by default),
   home/away splits (display only).
3. **Target**: 1X2 (primary), plus Over/Under 1.5/2.5/3.5 and BTTS derived
   from the Poisson score matrix.
4. **Model (Model B, independent of the market)**: `ensemble = 0.7·Elo + 0.3·feature-Poisson(Dixon-Coles)`
   (production weights), calibrated with a log-odds linear map, scored by
   SignalScorer into confidence/signal/completeness.
5. **Decision layer**: transparent Decision Score; hard gates (EV > 3%,
   |edge| < 20pp, calibration-bucket n ≥ 30, completeness ≥ 0.6, no
   Model-A/B disagreement); outputs STRONG/GOOD/LEAN/NO BET/NO CLEAR
   DECISION/MARKET PRIOR. Read-only advisory — no staking module in the bot.
6. **Evaluation**: chronological walk-forward (single pass, no random split),
   log-loss / Brier / ECE / hit rate with normal-approx CIs, flat-stake ROI
   on real odds, max drawdown / losing streak, market baseline comparison.

---

## 2. Diagnosis (ranked by severity)

### 🔴 HIGH — Train/serve parity bug in the backtest CLI
**`agents/football/backtest.py` `main()`** ran `run_backtest(...)` with no
model config, so the CLI silently evaluated the **library defaults**
(elo 0.5 / poisson 0.5) while production and `validate.py` use
**config/football.json (elo 0.7 / poisson 0.3)**. Same data, different model:

- `python -m agents.football.backtest --fixtures epl…json` → ensemble **LL 0.9943**
- `python -m agents.football.validate  --fixtures epl…json` → ensemble **LL 0.9886**

Anyone using `backtest` (including the documented experiment path) was
describing a different ensemble than the live bot — the exact class of
train/serve drift the checklist asks about.

### 🔴 HIGH — Loader silently drops the pre-match xG feature
**`agents/football/backtest.py` `_normalize_fixtures()`** re-serialized only a
fixed key set, dropping the rolling xG columns
(`home_xg_for/home_xg_against/away_xg_for/away_xg_against`). The augmented
dataset `cache/football/epl_fixtures_2022_2026_xg.json` carries real values in
1,506/1,520 rows, but **every standard harness re-loading it (`backtest`,
`validate --fixtures`) lost the feature**, so the production xG blend
(`xg_weight=0.65`) was inert there. The phase-9 xG evidence existed only via
the in-memory ablation path; the persisted dataset could not reproduce it
(verified: after `load_fixtures_from_json`, `home_xg_for` present in 0 rows).

### 🟡 MEDIUM — No bankroll/staking evaluation anywhere
The checklist asks for staking logic review (if present). The bot is
read-only (no auto-bet — good), but the **backtest only reported flat-stake
ROI**. There was no fractional-Kelly or growth-rate diagnostic, so nothing
answered "would a sound staking policy bet at all?".

### 🟡 MEDIUM — Latent crash in the ensemble when a component weight is 0
**`agents/football/models.py` `Ensemble.predict()`** crashed with
`ZeroDivisionError` when a component had weight 0 and the other was
unavailable (e.g. `elo_weight=0` with no feature-Poisson data → `total_w=0`).
Production (0.7/0.3) never triggers it, but any weight experiment could.

### ✅ Sound — confirmed by evidence (no change needed)
- **No leakage**: fixture ordering is kickoff-sorted; models update strictly
  after prediction; `leakage_audit` PASS (0 violations, determinism OK,
  CLV-scope PASS). xG rolling features are pre-match by construction; raw
  per-match xG is correctly excluded from every model input.
- **Chronological split**: walk-forward, Elo/form carry across seasons, no
  pooling-then-random-split (verified in `validate.py`/`backtest.py`).
- **Proper metrics**: log-loss, Brier, ECE (pooled per-outcome pairs),
  hit rate with CIs — not accuracy alone.
- **Calibration**: pooled ECE ≈ 0.008–0.011; the calibrator is near-identity
  (a=0.008, b=1.013), i.e. the ensemble is already well-calibrated and the
  live correction is honest and small.
- **Market baseline**: every model is evaluated against margin-free market
  implied probabilities; models beat the naive baseline in 4/4 seasons but
  **never** beat the market — correctly documented as "diagnostic value, not
  edge".
- **Decision honesty**: NO BET / NO CLEAR DECISION / MARKET PRIOR are valid
  outputs; thin-data matches get an explicitly-labelled market-mirror
  prediction with edge=0.
- **Class imbalance**: draws are handled via base-rate priors and pooled ECE
  rather than a naive majority classifier.

---

## 3. Proposed Changes → Implemented

### Fix 1 — `backtest.py`: load the production config (parity)
**Files**: `agents/football/backtest.py` (`main()`, new `_load_model_config()`).
**Change**: the CLI now loads `config/football.json` model sections (the same
helper logic `validate.py` uses) and passes them into `run_backtest`; new
`--elo-weight` / `--poisson-weight` overrides exist for experiments.
**Rationale**: one harness, one model description; `backtest` results now mean
the same thing as production and `validate`.
**Validation**: `backtest` ensemble LL moved **0.9943 → 0.9886**, identical to
`validate` (below).

### Fix 2 — `backtest.py`: carry pre-match xG features through the loader
**Files**: `agents/football/backtest.py` (`_normalize_fixtures`,
`load_fixtures_from_json` docstring).
**Change**: the four rolling pre-match xG columns are preserved. Raw per-match
`home_xg`/`away_xg` remain **deliberately excluded** (post-match → leakage).
**Rationale**: makes the standard harnesses able to evaluate the production xG
blend from the persisted dataset; nothing in the live path changes.
**Validation**: `validate --fixtures …epl_xg.json` now reproduces the phase-9
claim: ensemble LL **0.9886 → 0.9851** (below).

### Fix 3 — `validate.py`: fractional-Kelly staking diagnostics
**Files**: `agents/football/validate.py` (`_record_roi`, `_kelly_stats`,
`_finish`, `format_validation_report`, `KELLY_CAP`).
**Change**: per flat-stake bet the full-Kelly fraction
`f* = max(0, p − (1−p)/(odds−1))` (capped at 30% of bankroll) is computed and
reported as: number of bets with f*>0, mean f*, cumulative log-growth
`g = Σ ln(1 + f*·R)`, and return per Kelly stake. Evaluation-only — no stake
is ever placed (bot stays read-only).
**Rationale**: Kelly's growth criterion is the standard sound-bankroll answer
to "should we bet at all?" — **g ≤ 0 ⇒ the honest stake is 0**. It also
surfaces the favorite-overconfidence problem: the criterion *wants* to stake
13–20% of bankroll yet still achieves negative growth.

### Fix 4 — `models.py`: zero-weight ensemble robustness
**Files**: `agents/football/models.py` (`Ensemble.predict`).
**Change**: components with weight ≤ 0 are skipped; if no component remains,
`predict` returns `None` (missing data) instead of dividing by zero.
**Rationale**: removes a latent crash reachable by any weight experiment;
default config (0.7/0.3) behaviour is byte-identical.

---

## 4. Validation Results (before / after)

### 4.1 Harness parity — `backtest` vs `validate` (EPL 1,520 matches)

| Row | before fix (`backtest`) | after fix (`backtest`) | `validate` (reference) |
|---|---|---|---|
| ensemble log-loss | 0.9943 | **0.9886** | 0.9886 |
| ensemble Brier | 0.5940 | **0.5901** | 0.5901 |
| ensemble ECE | 0.0128 | **0.0110** | 0.0110 |
| ensemble hit rate | 52.8% | **53.1%** | 53.1% |
| ensemble ROI | −2.1% | **−1.9%** | −1.9% |

`backtest` now reproduces the production/validate numbers exactly.

### 4.2 xG feature now flows through the standard harness (EPL, walk-forward)

| Row | no-xG (before) | with xG (after) |
|---|---|---|
| ensemble log-loss | 0.9886 | **0.9851** |
| ensemble Brier | 0.5901 | **0.5873** |
| ensemble ECE | 0.0110 | **0.0081** |
| ensemble hit rate | 53.1% | **53.2%** |
| ensemble ROI | −1.9% | **−1.1%** |
| poisson ROI | −0.5% | **+3.7%** (n=1,505, 1X2 flat-stake) |

This reproduces the phase-9/14 finding (improves every metric, 0 degradations)
from the persisted dataset via the **standard** harness — previously only the
in-memory ablation path could show it.

### 4.3 Staking diagnostics (new) — honest verdict

| Model (EPL, no-xG) | bets f*>0 | mean f* | log-growth g | return/Kelly stake |
|---|---|---|---|---|
| baseline | 802 | 20.4% | −63.71 | −9.8% |
| elo | 854 | 14.9% | −20.31 | −0.6% |
| poisson | 762 | 18.1% | −34.79 | −3.6% |
| dc | 731 | 17.6% | −31.99 | −3.3% |
| **ensemble** | 600 | 13.2% | **−11.48** | +2.8% |

With xG: ensemble **g = −8.08**, mean f* 11.4% (566 bets).

**Read**: the production ensemble has the *least negative* growth of any model
and even a positive return-per-Kelly-stake, but **log-growth is still
negative in every configuration** — under full Kelly the bankroll decays.
The statistically sound policy, exactly as the decision engine already says:
**stake 0 (NO BET)**. No staking module in the bot is therefore correct.

### 4.4 Tests

Full suite: **565 passed** (560 pre-existing + 5 new:
`test_load_fixtures_carries_prematch_xg_only`, `test_backtest_loads_production_config_weights`,
`test_ensemble_weights_flow_into_backtest`, `test_kelly_staking_metrics_reported`,
`test_kelly_empty_without_odds`). No regressions.

### How to re-run

```bash
python -m agents.football.backtest --fixtures cache/football/epl_fixtures_2022_2026.json
python -m agents.football.validate  --fixtures cache/football/epl_fixtures_2022_2026_xg.json --out reports/fresh_audit_validate_epl_xg.json
python -m agents.football.validate  --fixtures cache/football/epl_fixtures_2022_2026.json   --out reports/fresh_audit_validate_epl_noxg.json
python -m pytest tests/ -q
```

---

## 5. Remaining Risks & Limitations (honest)

1. **The model never beats the market.** On log-loss, every model loses to
   margin-free implied in 4/4 seasons (even with xG: ensemble 0.9851 vs
   market 0.9652). The model's defensible value is **diagnostic** (calibration
   checks, uncertainty, market sanity-checking) — **not** a demonstrated
   betting edge. Any profit claim would require new out-of-sample evidence.
2. **xG coverage is big-5 only** and early-season windows are dominated by
   tail of the previous season; non-big-5 matches still run without xG
   (feature inert, completeness honestly reduced).
3. **No lineup/injury data.** Pre-match XI/absence data has no reliable
   historical source in this stack; it remains excluded (documented).
4. **Day-granularity replay**: same-day matches without kickoff times are
   ordered arbitrarily in historical caches (654 EPL dates affected); the
   live bot orders by kickoff timestamp. Effect is small but real.
5. **Calibration is pooled cross-league** (EPL-fitted params applied
   elsewhere); per-league OOS refits were tested and not promoted because they
   did not consistently improve (phase 5-6/37).
6. **Kelly diagnostics assume model probabilities are exact.** They are not
   (negative ROI), which is *why* the criterion over-stakes; the cap at 30%
   and the negative-g result are the honest reading, not a recommendation to
   stake 13-20%.
7. **Decision ROI is unstable across seasons** (+13.9% → −15.6%); STRONG/GOOD/
   LEAN buckets are within statistical noise (Wilson intervals wide). The
   WATCH tier and watch-gates remain config-off pending multi-league decision
   validation.
8. **Historical CLV unavailable** (single closing-price snapshot per match);
   CLV tracking is live-only via the odds-snapshot mechanism.
9. **No multi-league decision validation** for non-EPL leagues: `cache-odds`
   needs network access to football-data.co.uk (unavailable in this sandbox).
10. **Market-prior floor at completeness 0.6** means some thin-data matches
    are labelled MARKET PRIOR rather than given a model probability — by
    design, and honest, but a UX/tone trade-off.

**Bottom line**: this pass fixed three real defects (harness parity, silent
xG feature loss, missing staking evaluation, plus a latent crash). The model
itself is well-calibrated and leak-free, and it **remains honestly
unprofitable versus the market** — the report's correct conclusion, not a bug.
