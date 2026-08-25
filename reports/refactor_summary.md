# Refactor summary — "statistically honest loser" to validated-gate engine

Executed 2026-08-15, in the prescribed order. Each phase has a validation result
attached; no phase was marked done on code alone.

## Phase 1 — Diagnostic (report: `refactor_phase1_diagnostic.md`)
- Only EPL carries historical odds (1,520 matches); LaLiga/Serie A/Bundesliga
  caches have results but **0 odds rows** → no market benchmark, no ROI there.
- EPL 1X2: ensemble logloss 0.9886 vs market 0.9652; `beats_market_seasons =
  0/4` for every model; flat-stake ROI negative for all models.
- O/U 2.5/3.5/BTTS and decision-tier segments: **no historical validation
  exists** (data gap, must be built forward).

## Phase 2 — Edge benchmark (`agents/football/edge_benchmark.py`)
- No sharp source (Pinnacle/exchange) is configured, so every logged edge is
  labeled `soft_consensus` ("bukan closing line"). Label threaded into the
  decision dict, `predictions.jsonl` (`edge_benchmark` field), and Discord
  render. Config: `models.decision.edge_benchmark`.

## Phase 3 — CLV hard gate (`agents/football/clv_gate.py`)
- `segment_clv_stats` (prediction_log) + `gate_segment` require ≥200 settled
  bets AND price CLV > 0 (and ROI > 0) before a segment may act. Wired into
  `run_decision_engine`; failing segments demoted to NO BET. Status surfaced in
  `build_confidence_block` (`clv_gate` field) + Discord. Config:
  `models.decision.clv_gate`.

## Phase 4 — Sample-size / CI (`refactor_phase4_buckets.md`)
- `min_bucket_n` 30 → 200; added `bucket_ci_halfwidth` binomial 95% CI gate
  (≤3pp). Before/after on current bucket table: **8/9 → 1/9** buckets qualify;
  the only survivor is the underdog sweep (never value-credited). All playable
  buckets are now INSUFFICIENT_SAMPLE until they grow. Config:
  `min_bucket_n`, `min_bucket_ci_halfwidth`.

## Phase 5 — Per-league calibration + completeness (`refactor_phase5...`)
- `league_calibrator`: EPL uses the global fit; any other league must have its
  own `calibration_<slug>.json` (≥400 samples) or the decision layer forces
  MARKET PRIOR (never a foreign calibration). Config:
  `calibration.league_min_samples`.
- Completeness double-count fixed: form + attack/defense (same feed) merged into
  a single 0.40 component (was 0.20 + 0.20).

## Phase 6 — Staking (`agents/football/staking.py`)
- Fractional Kelly (¼), tier multiplier (STRONG 1.0 → WATCH 0.0), hard
  bankroll cap (2%/bet). Extreme edge (≥20pp) auto-declines. Recommended stake
  surfaced in Discord (`decision.stake`). Config: `models.staking`.

## Phase 7 — Decision-layer validation (`refactor_phase7_decision_validation.md`)
- Walk-forward on EPL (n=1,520): **Spearman(decision score, realized ROI) =
  0.0015**. Tercile ROI non-monotonic and negative everywhere. The 7-component
  score does NOT rank profitable bets. Weights marked `weights_validated: false`
  with the null result recorded; betting decisions now ride the hard gates, not
  the score.

## Phase 8 — Paper-trade harness (`refactor_phase8_paper_trade.md`)
- `runner paper-trade` reports per-segment graduation (ROI>0 AND CLV>0 AND
  n≥min). Current state: 0 segments graduated (1 settled match) → engine
  refuses all bets until data accumulates. Graduation re-evaluates continuously,
  never permanent.

## Tests
837 pass (816 baseline + 21 new across Phases 2/3/4/5/6).

## Net effect on production behaviour
Before: the bot would emit STRONG/GOOD/LEAN picks backed by soft-consensus
edges and EPL-only calibration, on 30-sample buckets, with no staking.
After: every actionable decision must pass (1) ≥200 settled bets with positive
ROI AND positive CLV per segment, (2) a ≥200-sample bucket with ≤3pp CI,
(3) a league-specific calibration fit, and (4) a ¼-Kelly stake capped at 2% of
bankroll. With the current data that means **the honest default is NO BET /
MARKET PRIOR everywhere** until the log accumulates enough settled evidence —
which is the correct state for a model whose own walk-forward shows it losing to
the market.
