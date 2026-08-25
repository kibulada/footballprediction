# SIGNAL AUDIT — Divergence vs the Pinnacle Closing Line (2026-08-12)

Research tool only. Produces probability/divergence estimates and THEORETICAL
sizing figures. No bet placement, no bookmaker API execution, no real-money
management anywhere in this system. Acting on the signal is a separate human
decision, outside this system's scope.

Engine: `agents/football/signal.py` (+ `tests/test_signal.py`, 8 tests).
Run: `python -m agents.football.signal --fixtures cache/football/epl_fixtures_2022_2026.json --out reports/signal_audit_epl_noxg.json`
All numbers below are from this repository's walk-forward harness on the EPL
2022-2026 cache (1,520 matches, real Pinnacle closing odds). Nothing fabricated.

---

## 1. Signal thesis

**Category B (structural/behavioural market signal), primary — measured as
model-vs-closing-line divergence.** Rationale: the historical cache provides a
sharp closing reference (Pinnacle, football-data.co.uk), so the one question
that can be answered with evidence is *"does the independent model's deviation
from the closing line carry information?"* — not "build a better model" (that
converges to the market; already proven in the phase 0-20 audits).

- **B (structural)**: model divergence vs closing; closing-line favourite-longshot
  bias; cross-book dispersion (live snapshot only).
- **A (speed)**: the odds-snapshot instrument (T-24h→T-15m, `price_clv_by_timing`)
  is already built in `prediction_log.py` but has **zero historical data** in the
  cache → future work requiring data collection, not an implemented feature.
- **C (alternative)**: xG is the one orthogonal feature with a historical record
  (understat, 5/5 leagues improve log-loss) — evaluated as the xG variant below.
  Referee/congestion data: not available → not claimed.

## 2. Data pipeline design

| Component | Source | Terms / availability | Status |
|---|---|---|---|
| Closing reference | football-data.co.uk CSVs (Pinnacle `PSH/PSD/PSA`), cached EPL 2022-26 | Free CSV, research use; already cached | ✅ available (1,520 matches) |
| Model probs | Elo + feature-Poisson ensemble, walk-forward replay (`signal._replay`) | in-repo | ✅ |
| Cross-book dispersion | The Odds API live payloads (`odds_soccer_*_recent.json`) | licensed odds-data API; cached | ⚠️ snapshot only (23 matches, no outcomes) |
| Speed (timing) | `prediction_log.append_odds_snapshot` T-24h/6h/1h/15m | in-repo | ⚠️ instrument only, 0 settled observations |
| Historical multi-book CSVs | football-data.co.uk full columns | free | ❌ blocked from this sandbox (network); future work |

Latency: no claim. Dispersion is measured from cached snapshots of the same
moment; no timestamp-to-kickoff analysis is possible with existing data.

## 3. Model design

**Target: the divergence d = p_model − p_close (per 1X2 side), not the outcome
alone.** The independent ensemble (raw, uncalibrated — same replay as
market_audit) generates the divergence; the audit then asks whether divergence
is informative:

1. **Head-to-head vs closing**: Brier/log-loss of model vs closing line
   (negative `brier_gap` = model better). Only claim of divergence information.
2. **Conditional divergence quality**: bucket matches by max |d| (0-2, 2-5,
   5-10, >10pp); per bucket, model-vs-closing Brier/LL + would-be flat ROI at
   closing odds with Wilson CI.
3. **Closing-line bias**: bucket closing margin-free implied → implied vs
   realised frequency (favourite-longshot test on the reference itself).
4. **Totals divergence**: feature-Poisson O/U 2.5 vs closing totals implied
   (head-to-head + would-be bets).
5. **Theoretical sizing**: full-Kelly f* = max(0, p − (1−p)/(odds−1)), capped
   30%; quarter-Kelly bankroll drawdown simulation. Analytical only.

## 4. Backtest & closing-line results

### 4.1 Model vs Pinnacle closing (Brier; gap < 0 = model better)

| model | n | model Brier | close Brier | gap | model LL | close LL | median \|d\| |
|---|---|---|---|---|---|---|---|
| baseline | 1520 | 0.6495 | 0.5739 | +0.0755 | 1.0971 | 0.9652 | 0.078 |
| elo | 1520 | 0.5921 | 0.5739 | +0.0181 | 0.9914 | 0.9652 | 0.039 |
| poisson | 1505 | 0.6223 | 0.5747 | +0.0476 | 1.0363 | 0.9664 | 0.069 |
| ensemble | 1520 | 0.5901 | 0.5739 | **+0.0162** | 0.9886 | 0.9652 | 0.035 |
| ensemble +xG | 1520 | 0.5873 | 0.5739 | +0.0134 | 0.9851 | 0.9652 | 0.033 |

Every model is worse than the closing line on Brier and log-loss. The xG
variant narrows the gap but does not cross it.

### 4.2 Conditional divergence quality — the decisive test (ensemble)

| \|d\| bucket | n | model Brier | close Brier | gap | bets | ROI@closing | hit (95% Wilson) |
|---|---|---|---|---|---|---|---|
| 0-2pp | 182 | 0.5621 | 0.5638 | −0.0016 | 0 | — | — |
| 2-5pp | 438 | 0.5843 | 0.5825 | +0.0018 | 176 | −3.3% | [40,55] |
| 5-10pp | 490 | 0.5892 | 0.5704 | +0.0188 | 224 | −8.9% | [34,47] |
| >10pp | 410 | 0.6099 | 0.5735 | **+0.0364** | 223 | +6.2% | [31,43] |
| >10pp (+xG) | 351 | 0.6095 | 0.5747 | +0.0347 | 167 | +1.0% | [26,40] |

**The model is worst exactly where it diverges most.** Divergence is noise,
not information. The positive >10pp ROI (no-xG, +6.2%, n=223) is contradicted
by the model's own Brier there (0.6099 vs 0.5735) and by the >10pp+xG ROI
(+1.0%) — a long-shot-variance artifact, not signal.

### 4.3 Closing-line bias of the reference itself (favourite-longshot test)

| implied bucket | n | implied | realised | delta | 95% Wilson |
|---|---|---|---|---|---|
| 0-10% | 154 | 0.073 | 0.039 | **−0.034** | [2,8] |
| 10-20% | 758 | 0.157 | 0.164 | +0.007 | [14,19] |
| 20-80% | 3587 | — | — | −0.018…+0.011 | — |
| 80-100% | 61 | 0.840 | 0.885 | +0.046 | [78,94] |

Robustness check (per season): 0-10% deltas −2.7, −5.5, **+2.1**, −7.7pp
(n=43/59/32/20); 80-100% deltas **−8.7**, +7.7, +9.4, +16.8pp (n=16/25/15/5).
**Signs flip across seasons; per-season n is tiny → not a robust structural
bias.** The aggregate deltas are small-sample/multiple-testing artifacts
(9 buckets tested, extremes only).

### 4.4 Totals O/U 2.5 divergence

| variant | n | model Brier | close Brier | gap | bets (O/U) | ROI@closing |
|---|---|---|---|---|---|---|
| no-xG | 1520 | 0.2593 | 0.2388 | +0.0204 | 1332 (493/839) | −8.8% |
| +xG | 1520 | 0.2461 | 0.2388 | +0.0073 | 1241 (715/526) | −4.9% |

Model worse than closing; would-be ROI negative both ways. No totals signal.

### 4.5 Sample size honesty

Per-bet return sd (ensemble, 1X2): ~1.56. Matches needed before an ROI is
distinguishable from noise (two-sided 95%): **1pp edge → 61,864 bets; 2pp →
15,466; 5pp → 2,475.** The EPL dataset (1,520 matches, ~600 bets) is ~25× too
small to certify even a 2pp edge.

## 5. Theoretical edge sizing (ANALYTIC ONLY — not an execution plan)

| model | would-be bets | f*>0 | mean f* | full-Kelly log-growth g | 25%-Kelly terminal bankroll | 25%-Kelly max DD | flat ROI@close |
|---|---|---|---|---|---|---|---|
| baseline | 805 | 803 | 20.4% | −64.07 | 0.00 | 100% | −7.5% |
| elo | 882 | 854 | 14.9% | −20.31 | 0.22 | 96.5% | −2.7% |
| poisson+xG | 594 | 590 | 15.8% | −20.98 | 0.21 | 96.1% | +3.4% |
| ensemble | 623 | 600 | 13.2% | **−11.48** | 0.66 | 89.6% | −1.9% |
| ensemble+xG | 583 | 566 | 11.3% | −8.08 | 0.94 | 85.2% | −1.1% |

**g < 0 in every configuration → the Kelly criterion stakes 0.** The
quarter-Kelly simulation destroys 85-100% of bankroll at peak drawdown — the
positive flat ROI (poisson+xG +3.4%, n=594, CI spans zero) coexists with
strongly negative growth because Kelly over-bets overconfident short-priced
picks. There is no positive sizing figure to report.

## 6. Honest verdict

**No informative signal found.** The system delivers exactly what the null
hypothesis predicts: a model that is **well-calibrated but market-matching** —
slightly *worse* than the Pinnacle closing line (Brier gap +0.0134..+0.0162),
whose divergence from the market is pure noise (worst exactly where it
diverges most), and whose would-be edge at closing odds is negative with
ruinous Kelly drawdowns. The one superficially interesting finding — extreme
closing-line buckets (0-10% and 80-100% implied) deviating by −3.4/+4.6pp —
**flips sign across seasons and rests on n=5-59 per season**: small-sample
artifact, not a structural bias.

**Future work (not implemented, honestly):** (a) timestamped multi-book odds
collection from the licensed The Odds API to actually measure cross-book lag —
the live snapshot already shows the instruments exist (23 matches, 346
book-rows; French books pmu_fr/winamax_fr run ~10.6% overrounds vs betfair_ex_eu
~1.4%, median cross-book home-implied spread 2.8pp, p90 3.6pp); (b) the
odds-snapshot speed log needs settled observations before any speed claim;
(c) any edge claim requires ~15,000+ matches, i.e. multi-league/multi-season
collection beyond the single EPL cache.

A decision to act on any of these estimates in a live betting context is a
separate human decision — including whether it is legal in the user's
jurisdiction — and is explicitly outside this system's scope.
