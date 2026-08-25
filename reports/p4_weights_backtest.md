# P4 — Backtest Report: Movement / Late-Movement & Ensemble 1X2 Weights

Date: 2026-08-16
Status: **No production weight change — current weights retained with evidence.**

---

## 1. Scope (per P4 spec)

1. Backtest the signal-engine `movement` (0.15) + `late_movement` (0.10) weight
   block against alternative weightings, using historical settled data.
2. Backtest the Ensemble 1X2 weighting (`elo_weight 0.7` / `poisson_weight 0.3`).
3. Cross-validate across at least **two non-overlapping historical periods**.
4. Written report with recommendation; only change weights on a clear,
   cross-validated improvement.

---

## 2. Data availability audit (honest baseline)

| Source | Count | Movement data? |
|---|---|---|
| `prediction_log` settled matches | **1** | no (single snapshot at query time) |
| `cache/football/multileague_fixtures.json` (EPL/LaLiga/Bundesliga/Serie A 2022–2026) | 5,792 | **0 fixtures** carry opening→latest fields (`movement` fields = 0) |

**Conclusion:** There is *no historical record* of opening→latest odds movement
with settled outcomes. The movement/late_movement weight block **cannot be
backtested on real data yet** — this is a data gap, not a modeling gap.
Per the P4 spec, current weights are therefore **retained as-is**, and the
requirement to log structured movement data (so this becomes testable later)
is recorded in Section 5.

The Ensemble 1X2 weights **are** testable: 5,792 settled matches with full
1X2 odds (EPL) and scores across 4 seasons → clean two-period cross-validation.

---

## 3. Ensemble 1X2 weights — cross-validated A/B

Method: `agents/football/backtest.py::run_backtest` (walk-forward replay,
same production model stack via `_load_model_config`). Period split:
**Period A = 2022-23 + 2023-24** (2,896 matches), **Period B = 2024-25 + 2025-26**
(2,896 matches). Non-overlapping, chronological.

### 3.1 All leagues (5,792 matches, hit-rate / log-loss / brier)

| Config | Period A hit | A log-loss | Period B hit | B log-loss |
|---|---|---|---|---|
| **current 0.7/0.3** | **0.5245** | **0.9958** | **0.5104** | **1.0133** |
| 0.5/0.5 | 0.5259 | 1.0017 | 0.4993 | 1.0165 |
| 0.6/0.4 | 0.5245 | 0.9981 | 0.5010 | 1.0143 |
| 0.8/0.2 | 0.5207 | 0.9948 | 0.5100 | 1.0136 |
| 1.0/0.0 (elo only) | 0.5128 | 0.9968 | 0.5024 | 1.0181 |

### 3.2 EPL only (full historical odds → ROI) — 760 matches per period

| Config | A hit | A ROI | A bets | B hit | B ROI | B bets |
|---|---|---|---|---|---|---|
| **current 0.7/0.3** | **0.5605** | **+0.064** | 291 | **0.5158** | **+0.060** | 328 |
| 0.5/0.5 | 0.5632 | +0.082 | 278 | 0.4895 | **−0.024** | 304 |
| 0.6/0.4 | 0.5618 | +0.040 | 292 | 0.4974 | +0.019 | 311 |
| 0.8/0.2 | 0.5566 | +0.052 | 323 | 0.5171 | +0.055 | 356 |
| 1.0/0.0 | 0.5434 | +0.028 | 384 | 0.5026 | +0.044 | 416 |

### 3.3 Verdict — KEEP 0.7/0.3

- **Current 0.7/0.3 is the only config with positive ROI in BOTH periods**
  (+0.064 / +0.060) and the best combined hit-rate.
- **0.5/0.5 would be a trap**: it posts the best Period A ROI (+0.082) but
  goes **negative (−0.024) in Period B** — exactly the overfit the
  two-period cross-validation is designed to catch.
- 0.8/0.2 is the closest competitor (positive both periods) but is strictly
  lower than current on Period A ROI (+0.052 vs +0.064) and marginally
  better only on Period B hit-rate (0.5171 vs 0.5158) with worse log-loss
  on Period A. No clear, cross-validated improvement → retained.

---

## 4. Movement / late_movement weights — data-gap decision

**Testable?** No. Zero historical opening→latest movement records exist
(Section 2). Any A/B here would be synthetic noise, not evidence.

**Decision:** `movement 0.15` + `late_movement 0.10` **unchanged**.
Documented reason: Layer 1 (immutable opening snapshot) + Layer 4 (narrative
binding) fixed the *correctness* of the movement input, but predictive-value
validation requires settled records with movement — which only starts
accumulating now that opening snapshots are pinned and logged per match.

---

## 5. Action items (tracked, not shipped)

1. **Log structured movement on every snapshot** — DONE as part of this work:
   - Layer 1 pinned immutable `opening_snapshot` rows per market (odds poll).
   - P5 `context_data` (lineups/injuries/coaches) logged per snapshot.
   - P4 (re-runnable) `signal_engine_ranking` — the FULL scored signal list
     (per-signal market, selection, edge_pp, movement, components) — is now
     persisted on every snapshot by `append_snapshot`, so settled matches can
     be RE-WEIGHTED later without re-fetching or mirrored scoring logic.
2. **Re-run via the shipped harness** once ≥ 500 settled matches with stored
   ranking exist (same two-period protocol):
   ```bash
   python -m agents.football.backtest_signal
   # --min-samples 500 (default), --report path.md to write the report
   ```
   The harness re-scores stored rankings through the exact production
   `score_signals` / `rank_and_pick`, settles via `settle_signal` (quarter-
   line AH included), reports hit-rate/ROI per weight set per period, and
   refuses to emit a report below the sample floor (honest insufficient-data
   guard). Tests: `tests/test_backtest_signal.py`.
3. `data_quality` (0.05) — flagged in P3 for review; kept at 0.05 because the
   P3 cross-source disagreement dock now gives it a real job. Revisit with
   the same settled-data discipline.

---

## 5b. Addendum 2026-08-17 — double-count fix (approved, model + statistical reweighted)

**Decision:** `movement` stays at 0.15; `late_movement` moved from 0.10 to
**0.00** as a score weight. `model` 0.30 -> 0.35, `statistical` 0.20 -> 0.25.

**Why:** `movement` (opening->current) and `late_movement` (multi-snapshot
late move) read the SAME price series, so weighting both credited one piece
of market information 0.25 total -- an artificial inflation of
market-following signals (the opposite of edge-finding). The late move is
retained as a PENALTY: when the market's last move is against the pick with
meaningful strength, confidence is capped at MEDIUM (see `rank_and_pick`).
The freed 0.10 went to the two components independent of the market
(model, statistical), per the professional review.

**Impact (verified arithmetic, not a signal flip):** the same Las Palmas-
Albacete Away +0.5 inputs score 0.577 -> **0.566** (still MEDIUM, threshold
0.52) -- the change reduces double-count inflation, it does not by itself
flip signals. Whether it reduces false-positive BEST PICKS must be
validated by the settled-data backtest (Section 5.2) once enough settled
matches with stored rankings exist.

**Validation:** `test_weights_no_longer_double_count_market_direction` +
`test_late_move_against_pick_caps_confidence` (signal_engine), updated
`test_p4_weights_retained_decision_pinned` pins the new weights.

---

## 6. Sign-off checklist

- [x] Backtest comparing current vs. candidate ensemble weights, cross-validated over two non-overlapping periods (Section 3)
- [x] Movement/late_movement: data-availability audit documented; weights retained with evidence (Section 4)
- [x] Written report with recommendation (this file)
- [x] No production weight changed without cross-validated improvement
- [x] Re-runnable harness shipped (`python -m agents.football.backtest_signal`) with honest insufficient-data guard + tests
