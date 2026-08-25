# FOOTBALL PREDICTION SYSTEM — FULL PROJECT AUDIT + IMPLEMENTATION TODO PLAN

**Scope:** everything under `agents/football/`, `bot.py`, `config/football.json`, `cache/`, `reports/`, `baseline/`, and `tests/`.

**Method:** direct code reading of every module, cross-checked against the existing audit reports (baseline freeze, leakage audit, ablation, calibration audit, market audit, decision validation, XG cross-league).

**Status:** AUDIT + ANALYSIS + DESIGN ONLY. No files modified, no tests run. Implementation begins only after approval of this plan.

---

# EXECUTIVE SUMMARY

**System quality: this is already an unusually disciplined, statistically honest pipeline.** It has strict chronological walk-forward validation, an automated anti-leakage audit that verifies predict-before-update content, margin-free implied probability everywhere, a hard separation between the odds-derived market model (Model A / MARKET MODEL) and the independent Elo+Poisson model (Model B / independent model), NO BET / NO CLEAR DECISION as first-class outputs, real historical Pinnacle odds for ROI evaluation, a frozen byte-identical baseline, and 431 passing tests. The team has repeatedly chosen "no production improvement" over adding unvalidated features — the correct behaviour under the stated rules. The decision-layer report is honest: it concludes "statistically inconclusive" rather than claiming improvement.

**The biggest problems are not leakage or dishonest math — they are train/serve skew and state staleness:**

1. **P0 — Live Elo is frozen at the last seed.** The live bot loads `cache/football/elo.json` and never calls `elo.update()` after results. The walk-forward validation that justifies the model updates Elo continuously; production does not. As a season progresses, live ratings drift further from reality.
2. **P0 — Calibration parameters and the empirical bucket table are static snapshots** with no refresh path in production, even though they are validated as continuously-updated objects in backtesting.
3. **P0/P1 — The live feature window differs from the backtest window.** Backtest form deques are `maxlen=5` and league-only; live `fetch_team_form` returns up to 10 matches across *all competitions*. The Poisson time-decay weighting therefore sees a different distribution in production than in validation. (xG history — 5 matches, league-only — matches between the two paths.)
4. **P1 — Totals/BTTS market comparison uses a synthetic "best-of-both-sides" line.** `extract_market_totals` takes the maximum price per side *across different bookmakers*, then normalizes the pair. This double-removes margin and inflates the model's apparent edge on totals.
5. **P1 — 70% of backtest matches (1,060/1,520) are same-day with arbitrary ordering.** The leakage audit documents this; the mitigation (kickoff-time ordering) has not been implemented.
6. **P1 — The decision layer is validated on EPL only** (the only cache with historical odds), and the out-of-sample evidence says it does not produce value (ROI −1.6% to −4.5% across rule variants; season-by-season it swings +13.9% → −15.6%). This is reported honestly, but the engine is still shipped live.

The correct next move is **NOT** to add features. It is to (a) make live state update exactly like validation state, (b) make live features identical to backtest features, (c) fix the odds-normalization defects that create false edges, (d) make the backtest temporally airtight, and (e) validate the decision layer on multiple leagues before trusting it — accepting NO BET and NO PRODUCTION IMPROVEMENT as valid outcomes.

---

# WHAT IS ALREADY GOOD

Components that should be **preserved unchanged** unless specific evidence says otherwise:

- **Chronological walk-forward validation harness** (`validate.py`) — strict temporal ordering, model state updated only after each match, no random train/test splits.
- **Automated anti-leakage audit** (`leakage_audit.py`) — module-scoped CLV/closing-odds restrictions (closing odds allowed only in evaluation/logging modules, never in model/context/feature modules) plus per-match leakage checks.
- **Model A / Market Model vs Model B / Independent Model separation** — edge is never claimed from two calculations derived from the same odds. The independent engine is used for value decisions.
- **Margin-free implied probabilities everywhere** — raw implied probability, overround, and margin-free probability are computed and used consistently; vig removal is explicit and documented.
- **NO BET / NO CLEAR DECISION as first-class outputs** — 967 of 1,520 walk-forward matches produced NO CLEAR DECISION rather than a forced pick.
- **Frozen byte-identical baseline** (`baseline/` + `baseline_freeze.py`) — regression protection for any future change.
- **Real historical odds for evaluation** — Pinnacle-based historical odds in the EPL cache; ROI evaluated against real market prices, not synthetic ones.
- **Dixon-Coles + Poisson + Elo ensemble with modest, evidence-based weights** — components were added only after ablation showed improvement; `ablation.py` exists to re-test this.
- **Calibration infrastructure** — bucketed empirical calibration, ECE 0.0103 on 4,560 samples, per-league calibration audit, model-gate integration (`model_gates.py`).
- **xG feature validated cross-league** — 5/5 big-5 leagues improved log-loss with `xg_weight=0.65` (reports/phase14_xg_cross_league.json); the live xG history window matches the backtest (5, league-only, finished matches only).
- **Honest experiment reporting** — rejected experiments are documented with their outcomes (see `reports/decision_engine_fase1.md`, S24/S37): `min_edge_pp=0` → ROI −8.6% (rejected); value credit for non-favourites → −7.8%, hit 28.3% (rejected); `market_value` weight 0.25 → no improvement (rejected). This discipline is the model for future work.
- **Production time-budget discipline** — 85s runner deadline, 55s `best_match` scan budget, 50s match-stats/lineups cap, graceful degradation (prediction unaffected by lineups failure).
- **JSON-safe decision output** (`decision_to_dict`) and anti-double-count rules between model prob, edge, and EV.
- **431 passing tests** covering normalization, overround, edge/EV, agreement, data quality, no-bet, JSON-safety, `best_prob_only`, leakage-config, alias/team matching, odds parsing, provider chains, and more.

---

# CRITICAL PROBLEMS

Ranked. P0 = must fix before any further model work; P1 = high; P2 = medium; P3 = optional.

## P0 — Critical

### P0-1. Live Elo is frozen at the last seed
- **Problem:** The live bot loads `cache/football/elo.json` (a snapshot) and never calls `elo.update()` with settled results. Validation updates Elo continuously.
- **Evidence:** `cache/football/elo.json` read at startup; no results-driven update path in the live pipeline (`analyse.py`/`runner.py`). Walk-forward validation in `validate.py` updates ratings after every settled match.
- **Consequence:** Live ratings drift from the ratings the model was validated with; live probabilities are stale by a growing amount as the season progresses. This is a pure correctness defect, not a tuning choice.

### P0-2. Calibration params and empirical buckets are static snapshots
- **Problem:** The calibration curve / empirical probability buckets and any fitted calibration parameters used live are static snapshots with no refresh path, even though the validation harness treats them as continuously-updated objects.
- **Consequence:** Live probability outputs are calibrated against a stale curve; over a season the miscalibration grows.

## P1 — High

### P1-1. Live form window ≠ backtest form window (train/serve skew)
- **Problem:** Backtest form deques are `maxlen=5` and league-only. Live `fetch_team_form` returns up to 10 matches across all competitions. The Poisson time-decay weighting sees different input distributions in production vs validation.
- **Consequence:** The model's live behaviour is not the behaviour that was validated. Any conclusion drawn from backtests does not transfer to production unchanged.

### P1-2. Totals/BTTS synthetic "best-of-both-sides" line
- **Problem:** `extract_market_totals` takes the maximum price per side across *different bookmakers*, then normalizes the pair.
- **Consequence:** Margin is removed twice; the model's apparent edge on totals/BTTS is inflated. This plausibly contributes to totals dragging the decision engine in the EPL decision validation.

### P1-3. Backtest same-day ordering: 1,060/1,520 matches with arbitrary ordering
- **Problem:** A large share of backtest matches are same-day, and their evaluation order is arbitrary rather than by kickoff time. The leakage audit documents this as a residual risk; the kickoff-time ordering mitigation has not been implemented.
- **Consequence:** A same-day result (earlier kickoff) can be used as a feature for a later-kickoff match — real information leakage, even if currently small in measured impact.

### P1-4. Decision layer validated on EPL only, and shipped without positive OOS evidence
- **Problem:** The decision engine (weights 30/20/15/15/10/5/5, `min_edge_pp=3.0`, `best_prob_only=true`) was tuned and validated on EPL only (the sole cache with historical odds). Walk-forward results: NEW 553 bets, hit 40.0%, ROI −2.2% (OLD: 623 bets, 41.2%, −1.9%). Per-bucket: STRONG 28 bets, hit 57.1%, ROI +3.1%; GOOD 434 bets, 38.5%, ROI −9.3%; LEAN 91 bets, 41.8%, ROI +29.5%. Season-by-season ROI swings +13.9% → −15.6%.
- **Consequence:** The engine ships live even though the honest reading is "statistically inconclusive / no demonstrated value". Small bucket sizes (STRONG n=28) make per-tier conclusions unreliable.

## P2 — Medium

### P2-1. Odds freshness and bookmaker-quality variance
- **Problem:** Odds are aggregated across sources with varying freshness (odds near kickoff noted as fast-closing in `oddspapi.py`); exchange prices must never be silently mixed with bookmaker prices.
- **Consequence:** Stale or mixed odds distort margin-free probabilities and EV in either direction.

### P2-2. Cache staleness without explicit invalidation rules
- **Problem:** `cache.py` persists fixtures/odds/form; there is no uniform TTL or staleness policy across caches.
- **Consequence:** Stale cached data can silently feed a prediction; recovery requires manual clearing.

### P2-3. Multi-source fallback compatibility
- **Problem:** Fallback sources (`multi_source.py` chain, TheSportsDB, SofaScore, SoccerData wrapper) can produce structurally different data than the primary source.
- **Consequence:** A silent fallback can inject incompatible features without raising any signal.

### P2-4. Per-decision-tier sample sizes too small
- **Problem:** STRONG (n=28), LEAN (n=91) buckets are too small for reliable per-tier ROI claims.
- **Consequence:** False confidence in a tier that happened to win/lose this season.

### P2-5. Same-day live features guard is soft
- **Problem:** The live pipeline has kickoff-hours-ahead guards for lineups rendering, but no hard, validated pre-kickoff cut for form/xG features on the same day.
- **Consequence:** Same-day matches can consume information from matches that have not yet kicked off (mirror of P1-3 on the live side).

## P3 — Optional

### P3-1. No WATCH tier in production output
- BET / NO BET exist; a WATCH tier for positive-but-unreliable edges is not implemented (designed in the decision framework, not shipped).

### P3-2. `format.py` is monolithic (~1,100+ lines)
- Single-responsibility and testability issues; output refactor is low risk but not urgent.

### P3-3. No CLV-centric production tracking loop
- CLV is computed in validation/logs, but there is no production loop that tracks CLV per pick type and feeds the decision engine back.

### P3-4. xG coverage limited to big-5 leagues (Understat)
- Cross-league xG features are unavailable; model must fall back to non-xG features outside the big-5.

---

# MODEL IMPROVEMENTS

Each proposal: Problem → Current behaviour → Proposed change → Statistical rationale → Data required → Leakage risk → Expected benefit → Validation required → Risk of regression.

## M-1. Live Elo refresh loop
- **Problem:** Live ratings frozen at seed (P0-1).
- **Current behaviour:** `cache/football/elo.json` loaded; `elo.update()` never called live.
- **Proposed change:** After settle (or before the next day's prediction), update Elo from settled results using the exact `elo.update()` code path used by validation; persist back to cache.
- **Statistical rationale:** The model was validated under continuous Elo updating; production must execute the same state machine to transfer the validated behaviour.
- **Data required:** Settled results (already available via settler/result feeds).
- **Leakage risk:** None if update happens only after results are final and before the next prediction is built. Must not update from results of matches whose prediction is currently being built.
- **Expected benefit:** Removes growing staleness; live predictions behave like validated ones.
- **Validation required:** Unit test: two consecutive runs produce ratings updated by settled results; parity test comparing live-updated ratings vs walk-forward ratings on the same matches.
- **Risk of regression:** Low — changes predictions, so gate behind baseline-equivalence comparison and a feature flag.

## M-2. Calibration & empirical bucket refresh path
- **Problem:** Static calibration snapshots (P0-2).
- **Current behaviour:** Calibration curve / buckets loaded from snapshot.
- **Proposed change:** Re-fit calibration parameters and empirical bucket tables from accumulated settled prediction logs on a schedule (e.g. weekly), using only pre-match predictions vs outcomes; never intraday.
- **Statistical rationale:** Calibration is a property of the model+state; it must be estimated on out-of-sample log data, refreshed as the state evolves.
- **Leakage risk:** None if buckets are fit only on predictions logged *before* the outcomes they are compared to.
- **Expected benefit:** Maintains ECE at validated levels over time instead of degrading.
- **Validation required:** Holdout-period ECE before/after refresh; ensure ECE does not degrade.
- **Risk of regression:** Low-medium; monitor ECE, rollback snapshot if ECE worsens.

## M-3. Feature-window parity (form)
- **Problem:** Live form window ≠ backtest window (P1-1).
- **Current behaviour:** Live `fetch_team_form` ≤10 matches, all competitions; backtest deques `maxlen=5`, league-only.
- **Proposed change:** Make the live path use the identical window definition (5, league-only) as the backtest; ideally extract the window logic into a single shared function used by both paths.
- **Statistical rationale:** Train/serve consistency is a precondition for any claim that backtest results transfer.
- **Leakage risk:** None (window is strictly past matches).
- **Expected benefit:** Live outputs become comparable to validated outputs.
- **Validation required:** Parity test: same inputs → same form features in live vs backtest code path.
- **Risk of regression:** Low; a small prediction shift is expected and acceptable.

## M-4. Dynamic home advantage (evaluate only)
- **Problem:** Home advantage may vary by league/season (e.g., COVID-era neutral venues), currently likely a static constant.
- **Current behaviour:** Static home advantage parameter.
- **Proposed change:** Estimate home advantage per league-season in a walk-forward manner (only from matches before the prediction date).
- **Statistical rationale:** Home advantage is empirically non-stationary; walk-forward per-league estimation is legitimate if strictly time-bounded.
- **Data required:** Historical results per league (available).
- **Leakage risk:** Medium if estimated on current-season data — must be strictly before prediction date; safe if trained on prior seasons only.
- **Expected benefit:** Small; likely a few basis points of log-loss. Not a priority.
- **Validation required:** Walk-forward ablation on multi-league validation; adopt only if log-loss improves beyond CI.
- **Risk of regression:** Low-medium; parameterization risk.

## M-5. Recency weighting for Elo/form (validate what exists)
- **Problem:** The Poisson component already time-decays form; Elo's internal weighting is fixed.
- **Current behaviour:** Fixed Elo K-factor; Poisson decay weighting exists.
- **Proposed change:** Ablate recency-weighted Elo (higher weight for recent matches) before considering any other Elo change.
- **Statistical rationale:** Team strength changes over time; but the existing Poisson decay may already capture this — the ablation decides whether Elo weighting adds anything orthogonal.
- **Leakage risk:** None if weights depend only on match dates.
- **Expected benefit:** Uncertain; must be measured.
- **Validation required:** Walk-forward ablation across leagues; adopt only if log-loss/Brier improve beyond CI.
- **Risk of regression:** Low if ablated.

## M-6. Totals/BTTS line consistency
- **Problem:** Best-of-both-sides synthetic line (P1-2).
- **Current behaviour:** Max price per side across bookmakers, then normalization.
- **Proposed change:** Use per-bookmaker price pairs (same bookmaker for Over and Under), or an explicit policy (consensus line with margin removal computed per bookmaker then averaged). Never mix sides from different bookmakers.
- **Statistical rationale:** A fair market probability requires a consistent line and margin from one bookmaker; mixing sides double-removes margin.
- **Data required:** Raw per-bookmaker over/under prices (available in odds feeds).
- **Leakage risk:** None.
- **Expected benefit:** Removes a false edge source on totals/BTTS; decision engine totals evaluation becomes meaningful.
- **Validation required:** Re-run market audit + decision validation on EPL totals; edge distribution should compress to honest levels.
- **Risk of regression:** Expected — fewer/more honest totals picks; that is the point.

## M-7. Uncertainty estimation (ensemble spread)
- **Problem:** No explicit prediction uncertainty; confidence is heuristic.
- **Current behaviour:** Confidence derived from decisiveness/agreement heuristics.
- **Proposed change:** Compute ensemble disagreement (spread across Elo/Poisson/DC components, and across the 1X2 distribution) as a formal uncertainty signal; feed it to the decision layer as a gate.
- **Statistical rationale:** Model disagreement predicts error; using it to suppress bets is a principled uncertainty gate (well documented in ensemble literature).
- **Leakage risk:** None (computed from model outputs at prediction time).
- **Expected benefit:** Reduces false-positive bets on noisy matches; improves per-bucket ROI.
- **Validation required:** Decision-layer walk-forward: does the uncertainty gate improve ROI/log-loss of bet subset?
- **Risk of regression:** Low; gating only removes picks.

## M-8. Market movement as a feature (deferred, high leakage risk)
- **Problem:** Opening vs current odds could carry signal.
- **Current behaviour:** Not used.
- **Proposed change:** Evaluate only if a clean opening-odds history with timestamps is available pre-kickoff.
- **Statistical rationale:** Line movement is a market opinion; it is legitimate as a *feature of the market model*, not as an independent edge.
- **Leakage risk:** High — closing odds must never be used; only snapshots strictly before prediction time.
- **Expected benefit:** Uncertain; likely small for the independent model.
- **Validation required:** Leakage-audited walk-forward; only if `odds_history.py` snapshots have reliable timestamps.
- **Risk of regression:** Medium; do not ship without the leakage audit.

## M-9. Model stacking (deferred)
- **Problem:** Whether stacking components into a meta-model helps OOS is unknown.
- **Current behaviour:** Fixed-weight ensemble.
- **Proposed change:** Only after M-1..M-3 land and multi-league validation exists: train a stacking meta-model (logistic regression on component probabilities) strictly walk-forward with early stopping, and gate against leakage audit.
- **Statistical rationale:** Stacking can exploit component complementarity — but is prone to overfitting with small samples; needs multi-league data.
- **Leakage risk:** High if meta-model is fit with future data — must be walk-forward only.
- **Expected benefit:** Uncertain; adopt only with clear OOS improvement.
- **Validation required:** Strict walk-forward, multi-league, leakage-audited.
- **Risk of regression:** Medium; default is "reject unless proven".

---

# DATA SOURCE IMPROVEMENTS

Recommendation: PRIMARY / SECONDARY / FALLBACK per capability, and which sources must NOT feed critical model features.

| Source | Role | Use for | Reliability | Pre-match availability | Notes |
|---|---|---|---|---|---|
| **Flashscore** | PRIMARY (fixtures, lineups) | Fixture discovery, kickoff times, predicted/confirmed lineups | Good; browser/endpoint quirks | Yes | Lineups are context-only today; don't make them model features without validation |
| **OddsPapi** | PRIMARY (odds) | Raw bookmaker odds, over/under pairs, freshness | Medium; rate-limited; near-kickoff fast-closing noted | Yes | Must keep per-bookmaker pairs (M-6); never exchange-vs-bookmaker mixing (P2-1) |
| **football-data.org** | PRIMARY (results/fixtures) | Historical results, fixture schedules, seasons | High; throttled (~6s/data call) | Yes | Backbone for backtests; respect throttle |
| **Understat xG** | PRIMARY (xG, big-5 only) | xG form features | Medium; scraped; big-5 only | Yes | Keep the validated 5-match league-only window; do not extend to other leagues without re-validation |
| **SofaScore** | SECONDARY | Odds cross-check, stats, lineups fallback | Medium; browser-based | Yes | Use only as cross-check, not primary features |
| **SoccerData (wrapper)** | SECONDARY | Historical stats via FBref | Medium; scraping stability risk | Mostly | For backtest enrichment only; not live-critical |
| **TheSportsDB** | FALLBACK / context only | Team metadata, badges, context | Low; inconsistent | Yes | **Do NOT use for critical model features** |
| **Multi-source chain** | FALLBACK | Fixtures/odds when primary fails | Medium | Yes | Must signal which source produced the data (P2-3); never silently mix |

**Rules:**
1. The independent model's features must come from PRIMARY sources only.
2. Odds for market comparison: OddsPapi pairs + Pinnacle where available; exchange prices clearly labelled and never mixed with bookmaker prices.
3. Any fallback that changes data structure must be recorded in the prediction log (`prediction_log.py`) for auditability.
4. xG stays Understat-only (big-5) until a second source is validated on the same walk-forward harness.

---

# BETTING DECISION IMPROVEMENTS

## BET / WATCH / NO BET framework

- **BET** — only when ALL of: (a) independent model edge vs market exceeds the confidence floor (`min_edge_pp`), (b) model agreement and uncertainty gates pass (M-7), (c) odds are fresh and from quality bookmakers (P2-1), (d) calibration within tolerance, (e) sample support exists (enough historical evidence for the league/market type), and (f) the edge survives the leakage-audited, multi-league decision validation.
- **WATCH** (new tier, TODO-16) — positive edge that fails a reliability gate (e.g., small sample, model disagreement, stale odds, untested league/market). Output to a watchlist with a clear "insufficient evidence" label; each watched pick is logged and its CLV tracked so the tier can be promoted or killed on evidence.
- **NO BET** — everything else. Explicitly documented as a successful decision when evidence is insufficient. Most-likely-outcome ≠ bet (70% home win with un-compensating odds ⇒ NO BET). A large mathematical edge with unreliable inputs is also NO BET.

## Principles encoded in the engine
- Most likely outcome, best value, and NO BET are separate outputs; the final pick is always value-driven, never probability-driven alone.
- Edge/EV computed only from the INDEPENDENT model vs margin-free market; never from two same-market-derived numbers.
- EV should be reported with uncertainty (M-7 / TODO-10): a 4% edge with 30% spread is not the same as a 4% edge with 5% spread.
- Bookmaker quality, odds freshness, and liquidity (Pinnacle preferred) are decision inputs, not just warnings.

---

# BACKTESTING IMPROVEMENTS

1. **Kickoff-time ordering (TODO-05):** order same-day matches by kickoff time in `validate.py`; a match's features may only include matches whose kickoff precedes its own. Directly addresses P1-3 and closes the documented residual leakage.
2. **Train/serve code-path parity (TODO-03):** single shared feature-construction functions used by both live and backtest, so the two cannot drift again. Verify with a parity test.
3. **Multi-league, multi-season evaluation (TODO-07/08):** extend historical odds caches and decision validation beyond EPL — at minimum to the big-5 where historical odds exist, plus one lower tier. All claims must be reported per league and per season, with confidence intervals.
4. **Per-market-type evaluation:** 1X2, totals, BTTS, (Asian handicap if odds permit) reported separately; never aggregate markets with different margins into one ROI.
5. **Full metric suite** (already partly present; make mandatory on every validation report): log-loss, Brier, ECE, ROC-AUC, hit rate, precision, recall (where meaningful), ROI, yield, CLV, max drawdown, number of bets, per-bucket confidence intervals. Hit rate alone is never used as evidence of model quality.
6. **Hindsight-free feature selection:** any feature or parameter choice must be decided on walk-forward information only; experiments are pre-registered in reports before execution (this is already the house style — keep it).
7. **Same-day live guard (P2-5):** the live pipeline needs the mirror of rule 1 — a hard pre-kickoff cut for same-day features, validated by a leakage audit check.

---

# TEST PLAN

Tests that must pass before any implementation is considered successful:

1. **All 431 existing tests pass** — the frozen suite is the regression floor.
2. **Baseline equivalence:** after any model/state change, the frozen baseline comparison (`baseline_freeze.py`) shows equivalence or a deliberately re-frozen new baseline with a documented reason.
3. **Leakage audit PASS** (`leakage_audit.py`) after every phase, including the new same-day ordering rules and the live Elo update loop (update must occur strictly after result finality).
4. **New unit tests:**
   - Elo live-update loop: settled results update persisted ratings; no update from unsettled/current fixtures.
   - Calibration refresh: refit from pre-match logs only; ECE non-worse on holdout.
   - Form-window parity: live vs backtest path produce identical features from identical inputs.
   - Totals line: same-bookmaker pairs only; margin removal single-pass; exchange never mixed with bookmaker.
   - Same-day ordering: a later-kickoff match never sees an earlier-kickoff same-day result (and vice versa).
   - WATCH tier: classification, watchlist logging, CLV tracking, promotion/demotion logic.
   - Uncertainty gate: high-spread candidates are gated; gate is monotone in spread.
5. **Decision validation on ≥2 leagues** with per-season and per-tier confidence intervals, before any decision-engine config change ships.
6. **Production smoke test:** full `analyse` run within the 85s deadline with a cold cache, live state updated, output JSON-safe.

---

# IMPLEMENTATION TODO

Ordered checklist. Each item: affected module, objective, reason, dependencies, validation.

## Phase A — Correctness & train/serve parity (do first; highest value, lowest risk)

### [ ] TODO-01 — Live Elo update loop
- **Module:** `agents/football/elo.py`, `agents/football/analyse.py`, `agents/football/runner.py`, `agents/football/settler.py`, `cache/football/elo.json`
- **Objective:** After results settle, update Elo ratings using the identical `elo.update()` path validation uses, and persist. Before building any prediction, ratings reflect all results up to (not including) the target match.
- **Reason:** P0-1. Live predictions currently use stale ratings that diverge from the validated state machine.
- **Dependencies:** none.
- **Validation:** unit tests (update only from settled results; persistence round-trip); parity spot-check vs walk-forward ratings; baseline-equivalence report; leakage audit PASS.

### [ ] TODO-02 — Calibration & empirical bucket refresh
- **Module:** `agents/football/calibration.py`, `agents/football/model_gates.py`, `agents/football/prediction_log.py`
- **Objective:** Scheduled re-fit of calibration parameters and empirical buckets from pre-match prediction logs (only predictions logged before their outcomes); versioned snapshots with rollback.
- **Reason:** P0-2. Static snapshots decay.
- **Dependencies:** TODO-01 (state consistency).
- **Validation:** holdout ECE before/after; unit tests; baseline-equivalence.

### [ ] TODO-03 — Feature-window parity (form 5, league-only)
- **Module:** `agents/football/context.py`, `agents/football/multi_source.py` (form fetch), `agents/football/validate.py` (backtest form)
- **Objective:** Extract shared form-window logic; live and backtest both use 5-match league-only windows with identical construction.
- **Reason:** P1-1 train/serve skew.
- **Dependencies:** none.
- **Validation:** parity unit test (identical inputs → identical features); regression comparison of live predictions before/after; leakage audit PASS.

## Phase B — Odds/market integrity & backtest temporal airtightness

### [ ] TODO-04 — Totals/BTTS per-bookmaker line fix
- **Module:** `agents/football/odds_fetcher.py`, `agents/football/oddspapi.py`, `agents/football/models.py` (totals extraction)
- **Objective:** Replace best-of-both-sides line with same-bookmaker over/under pairs; explicit policy for consensus; single-pass margin removal.
- **Reason:** P1-2; false edge on totals/BTTS.
- **Dependencies:** none.
- **Validation:** market audit re-run; totals edge distribution compression; EPL decision validation re-run; unit tests for pair integrity and exchange/bookmaker separation.

### [ ] TODO-05 — Same-day kickoff-time ordering in backtest
- **Module:** `agents/football/validate.py`, `agents/football/leakage_audit.py`
- **Objective:** Order same-day matches by kickoff; features for a match include only matches with strictly earlier kickoff.
- **Reason:** P1-3; closes the documented residual leakage (1,060/1,520 same-day matches).
- **Dependencies:** none.
- **Validation:** leakage audit PASS; ordering unit tests; re-run EPL walk-forward (metrics may shift slightly — expected).

### [ ] TODO-06 — Odds freshness & exchange/bookmaker separation audit
- **Module:** `agents/football/market_audit.py`, `agents/football/odds_fetcher.py`, `agents/football/oddspapi.py`
- **Objective:** Extend the market audit to flag stale odds (age vs kickoff) and to verify exchange prices are never treated as bookmaker prices in margin removal or EV.
- **Reason:** P2-1.
- **Dependencies:** TODO-04.
- **Validation:** market audit report; unit tests for freshness guards.

### [ ] TODO-07 — Multi-league historical odds cache
- **Module:** `cache/`, `agents/football/odds_history.py`, `agents/football/backtest.py`
- **Objective:** Build historical odds caches for ≥2 more leagues (big-5 where historical odds exist + one lower tier) so decision validation is no longer EPL-only.
- **Reason:** P1-4; single-league conclusions are not generalizable.
- **Dependencies:** TODO-04, TODO-05 (so new caches are built under the fixed pipeline).
- **Validation:** cache completeness report; decision validation on new leagues.

## Phase C — Decision engine evidence

### [ ] TODO-08 — Multi-league decision-layer walk-forward validation
- **Module:** `agents/football/decision_validation.py`
- **Objective:** Run the existing decision engine walk-forward on all leagues with historical odds; report per league, per season, per tier with CIs.
- **Reason:** P1-4. Ship only what survives multi-league evidence.
- **Dependencies:** TODO-07.
- **Validation:** report with metrics (ROI, yield, CLV, hit, n, CIs); explicit GO/NO-GO recommendation.

### [ ] TODO-09 — Decision reliability gates
- **Module:** `agents/football/decision.py`, `agents/football/model_gates.py`
- **Objective:** Add gates: model agreement, ensemble-spread uncertainty, sample-size floor per league/market, odds freshness, bookmaker quality. A candidate failing any gate is downgraded to WATCH/NO BET, not force-ranked.
- **Reason:** False-confidence reduction (P2-4, P3-1 groundwork); "large edge ≠ bet".
- **Dependencies:** TODO-08 (gates tuned on multi-league evidence, not EPL alone).
- **Validation:** decision validation before/after gates; false-positive-bet reduction measured.

### [ ] TODO-10 — EV under uncertainty (variance-aware)
- **Module:** `agents/football/decision.py`, `agents/football/scorer.py`
- **Objective:** Report EV with uncertainty (ensemble spread / confidence interval) and use it in ranking; prefer narrower-CI edges.
- **Reason:** M-7; a 4% edge at 5% spread ≠ 4% at 30% spread.
- **Dependencies:** TODO-09.
- **Validation:** decision validation; CI-aware ranking vs current ranking.

## Phase D — Model experiments (adopt ONLY with OOS, multi-league, beyond-CI evidence)

### [ ] TODO-11 — Dynamic home advantage (walk-forward)
- **Module:** `agents/football/elo.py`, `agents/football/context.py`
- **Objective:** Walk-forward per-league-season home advantage estimation; ablate.
- **Reason:** M-4.
- **Dependencies:** TODO-03.
- **Validation:** multi-league walk-forward ablation; adopt only if log-loss improves beyond CI.

### [ ] TODO-12 — Recency-weighted Elo ablation
- **Module:** `agents/football/elo.py`
- **Objective:** Ablate recency-weighted Elo vs fixed-weight vs the existing Poisson decay baseline.
- **Reason:** M-5.
- **Dependencies:** TODO-03.
- **Validation:** same ablation harness; adopt only with evidence.

### [ ] TODO-13 — Stacking / ensemble-spread uncertainty experiment
- **Module:** `agents/football/models.py`, `agents/football/scorer.py`
- **Objective:** (a) Ship ensemble spread as an uncertainty feature (feeds TODO-09/10); (b) optionally test a walk-forward logistic stacking meta-model.
- **Reason:** M-7/M-9.
- **Dependencies:** TODO-08.
- **Validation:** leakage-audited, multi-league, walk-forward; reject unless clearly better.

## Phase E — Maintenance automation & WATCH tier

### [ ] TODO-14 — Validation/leakage automation in CI
- **Module:** `agents/football/validate.py`, `agents/football/leakage_audit.py`, `tests/`
- **Objective:** Automated leakage audit + baseline-equivalence + walk-forward smoke as a standard command run after every change.
- **Reason:** Make the existing discipline mechanical.
- **Dependencies:** TODO-05 (audit covers new ordering).
- **Validation:** command runs green on the untouched baseline.

### [ ] TODO-15 — CLV-centric production tracking loop
- **Module:** `agents/football/prediction_log.py`, `agents/football/settler.py`, `agents/football/format.py`
- **Objective:** Production loop that tracks CLV and ROI per pick type/league/market as results settle; feeds a periodic decision-engine health report.
- **Reason:** P3-3; production truth replaces backtest assumptions.
- **Dependencies:** TODO-01 (live state truth), TODO-08.
- **Validation:** tracking report matches manually verified CLV on a sample.

### [ ] TODO-16 — WATCH tier
- **Module:** `agents/football/decision.py`, `agents/football/format.py`, `agents/football/prediction_log.py`
- **Objective:** Ship BET / WATCH / NO BET; WATCH = positive edge failing a reliability gate, logged with CLV tracking for later promotion/demotion.
- **Reason:** P3-1; WATCH is a successful, honest output.
- **Dependencies:** TODO-09, TODO-15.
- **Validation:** classification unit tests; watchlist CLV review after a full season.

## Phase F — Cleanup

### [ ] TODO-17 — `format.py` decomposition
- **Module:** `agents/football/format.py`
- **Objective:** Split the ~1,100-line output module into focused sections (top renders, decision section, stats section) without changing output text.
- **Reason:** P3-2; testability.
- **Dependencies:** none (pure refactor).
- **Validation:** all output tests pass byte-identical; full suite green.

### [ ] TODO-18 — Documentation consolidation
- **Module:** `reports/`, `audit/`
- **Objective:** Consolidate the audit story into one living document (this plan + per-phase results), marking accepted/rejected experiments.
- **Reason:** P3-3 groundwork; reproducible decision trail.
- **Dependencies:** none.
- **Validation:** every report referenced by this plan exists and links correctly.

---

# EXPECTED IMPACT

No accuracy percentage is promised. Success is defined by measurable criteria:

- **Lower out-of-sample log-loss / Brier** than the current ensemble benchmark (EPL walk-forward log-loss 0.9886 vs market 0.9652) on multi-league validation — or an explicit, documented "no improvement found" verdict.
- **Calibration maintained:** ECE at or below the current 0.0103 level (4,560 samples) across refreshed snapshots, in every league.
- **Positive CLV** sustained across seasons (the market-beating signal), reported per league with confidence intervals.
- **Reduced false-positive bets:** fewer, higher-quality picks; per-tier ROI becomes stable across seasons instead of swinging +13.9% → −15.6%; no bucket with n<50 is used to justify a claim.
- **Leakage-free backtest:** the same-day ordering residual (1,060/1,520 matches) is closed and the audit stays PASS under automation.
- **Train/serve parity:** live and backtest consume identical feature definitions; parity tests are part of CI.
- **Stable performance in unseen periods:** every change is validated on periods/leagues not used for its design.

---

# RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Elo update loop changes live predictions vs frozen baseline | Medium | Medium | Feature flag; baseline-equivalence report before shipping; rollback = old snapshot |
| Calibration refresh degrades ECE | Low | Medium | Versioned snapshots, holdout ECE check, auto-rollback |
| Totals-line fix reduces apparent edge | Certain (by design) | Medium | Expected and documented; re-validate before concluding totals are unprofitable |
| Same-day reordering shifts historical metrics | Medium | Low | Expected; metrics recomputed under the new ordering, never compared to old numbers as "regression" |
| Multi-league odds data scarce/unreliable | Medium | High (for TODO-07/08) | Source-agnostic cache builder; only leagues with adequate sample enter validation; WATCH tier covers untested leagues |
| Tuning on EPL-only evidence (again) | Medium | High | Multi-league gate (TODO-08) is mandatory before any config change ships |
| Stacking overfits | Medium | High | Strict walk-forward, pre-registered experiment, default-reject (TODO-13) |
| Production stability (85s deadline, cache pressure) | Low | High | Time-budget discipline preserved; smoke test in test plan; no new network calls in the hot path |

---

# IMPLEMENTATION ORDER

Safest sequence — each phase ends with: full test suite green, baseline equivalence (or a deliberately re-frozen baseline), leakage audit PASS, and a written report.

1. **Phase A (TODO-01 → 02 → 03):** correctness and train/serve parity. Highest value, lowest risk. No tuning, no new features.
2. **Phase B (TODO-04 → 05 → 06 → 07):** odds/market integrity and airtight backtest timing, then multi-league odds caches built on the fixed pipeline.
3. **Phase C (TODO-08 → 09 → 10):** decision-engine evidence on multiple leagues; reliability gates and uncertainty-aware EV.
4. **Phase D (TODO-11 → 12 → 13):** model experiments with pre-registered, leakage-audited, multi-league walk-forward validation; adopt only beyond CI.
5. **Phase E (TODO-14 → 15 → 16):** automation of validation, CLV-centric production tracking, and the WATCH tier.
6. **Phase F (TODO-17 → 18):** cleanup and documentation.

**Nothing enters production config without out-of-sample, multi-season, beyond-CI evidence — and NO BET / WATCH / NO PRODUCTION IMPROVEMENT remain fully acceptable outcomes.**

*This document is the audit + TODO plan only. No code was modified; approval is required before implementation begins.*

---

# IMPLEMENTATION STATUS (2026-08-12 — APPROVED & EXECUTED)

See `reports/implementation_report.md` for the full per-TODO write-up.

| TODO | Status | Note |
|---|---|---|
| TODO-01 Live Elo update loop | ✅ DONE | live `settle` updates `elo.json` (kickoff order); pure correctness fix |
| TODO-02 Calibration refresh | ✅ DONE | `calib-refresh` CLI; skip guard + `.bak` + no-regression ECE guard |
| TODO-03 Form-window parity | ✅ DONE | live form window 10 → 5 (== backtest maxlen 5) |
| TODO-04 Totals per-bookmaker pairs | ✅ DONE | single margin removal; inline dup removed |
| TODO-05 Kickoff-time ordering | ✅ DONE | applied to all 5 replay harnesses; date-only = unknown time |
| TODO-06 Odds-quality audit | ✅ DONE | `odds_quality` section in market_audit |
| TODO-07 Multi-league odds caches | ⏳ INFRA DONE | `runner cache-odds` shipped; network unavailable in sandbox — run on a networked host |
| TODO-08 Multi-league decision validation | ⏳ INFRA DONE | `--fixtures` accepts comma-separated files; needs TODO-07 caches |
| TODO-09 Reliability gates | ✅ DONE (opt-in) | `enable_watch` config-gated, OFF in config |
| TODO-10 EV under uncertainty | ✅ DONE | `ev_band` from ensemble spread, rendered in Discord |
| TODO-11 Dynamic home advantage | ✅ DONE (experiment) | estimator + tests; not wired, adopt only with evidence |
| TODO-12 Recency-weighted Elo | ✅ DONE (experiment) | `k_multiplier_for_gap` + `update(k_multiplier=)`; off by default |
| TODO-13 Ensemble spread + stacking | ✅ DONE (experiment) | `uncertainty` in model_probs; IRLS stacker; not wired |
| TODO-14 Audit automation | ✅ DONE | `runner audit` command; CLV scope PASS |
| TODO-15 CLV per decision type | ✅ DONE | `decision_type` logged; `by_decision` in stats + renderers |
| TODO-16 WATCH tier | ✅ DONE (opt-in) | 👁 WATCH badge; requires positive value; config-gated OFF |
| TODO-17 format.py decomposition | ✅ DONE | `format_utils.py` + `format_pages.py`; byte-identical output |
| TODO-18 Documentation | ✅ DONE | this status + `reports/implementation_report.md` |

**Validation:** `547 passed` (527 pre-existing + 20 new), CLV scope **PASS**, all modules compile.

**What changed in production behaviour without a flag (by design):** the live
Elo update after settle (TODO-01) and the 5-match form window (TODO-03) — both
restore parity with the validated harness. Everything else is opt-in or
experimental.
