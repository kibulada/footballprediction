# ADVERSARIAL AUDIT — Hermes Football as a betting engine (market-maker perspective)

Audited: 2026-08-16. Scope: `agents/football/*`, `config/football.json`, `cache/football/*`, `reports/*`.
Method: direct code reading of every model/decision/evaluation module, cross-checked against the repo's
own reports, then **re-running the harnesses on this machine** to verify every disputed number.
Nothing in this report is asserted from memory; every claim is reproducible with the commands listed.

**Bottom line, stated plainly:** this is an unusually honest, disciplined, leak-free *forecasting*
pipeline — and it is **not a betting engine**. It has no demonstrated edge over the market. Every
component that would make it profitable is either unvalidated, blocked by its own gates, or — in the
one place where a positive result appeared — an artifact of a **data-leakage bug in the evaluation
harness that I reproduced on this machine**. Once that bug is fixed, the only "we beat the market"
number in the entire project disappears. The correct operating state of this bot is exactly what its
own gates already enforce: **NO BET, everywhere, until real out-of-sample evidence accumulates.**

---

## 1. The evidence, as it actually stands

All figures are from the repo's own walk-forward harnesses (EPL 2022–2026, 1,520 matches, real
Pinnacle closing odds, single chronological pass):

| Metric | Ensemble (0.7 Elo + 0.3 DC) | Margin-free market | Verdict |
|---|---|---|---|
| Log-loss (no xG) | 0.9886 | 0.9652 | **Loses to market** |
| Log-loss (with xG) | 0.9851 | 0.9652 | **Loses to market** |
| Brier (with xG) | 0.5873 | 0.5739 | Loses |
| Beats market, seasons | 0/4 | — | Loses every season |
| Flat-stake ROI (edge ≥ 2pp vs closing) | −1.1% to −1.9% | — | Negative |
| Kelly log-growth g | −8.1 to −11.5 | — | **Negative ⇒ correct stake = 0** |

Decision layer (Phase 7, EPL walk-forward, n=1,520): **Spearman(decision score, realized ROI) = 0.0015**.
The 7-component "Decision Score" does not rank profitable bets; tercile ROI is non-monotonic and
negative at every tercile. Per-season decision ROI swings +13.9% → −15.6% — pure noise.

The repo documents all of this honestly (fresh_audit, refactor_summary, final_phase_report). The
problem is not that the team hid it; the problem is that one harness later **accidentally produced the
opposite result and nobody noticed**.

---

## 2. 🔴 P0 — THE finding: the only "beats market" result in the project is a leakage artifact

**The anomaly.** `reports/validation_multileague_2026-08-16.json` shows EPL `n = 3040` (not 1520),
ensemble log-loss **0.9424** vs market 0.9652 → `ll_le_market: true`, flat-stake ROI **+31.7%**,
Kelly growth **+52.7**. This is the *only* cell in the entire repo where a model beats the market and
is "profitable". Every other harness says the opposite. Both cannot be true.

**Root cause (verified in code, then reproduced).** `agents/football/runner.py` `validate-multileague`:

```python
ml_file = ROOT / "cache/football/multileague_fixtures.json"      # already contains EPL/LaLiga/Bundesliga/SerieA
if ml_file.exists():
    ... fixtures_by_league[league].append(fx)                    # EPL += 1520
for f in sorted((ROOT / "cache/football").glob("*_fixtures_2022_2026.json")):
    ... fixtures_by_league.setdefault(league, []).extend(rows)   # EPL += 1520 again
```

`multileague_fixtures.json` (5,792 rows) already holds every league. The glob then appends the same
leagues **again** from `epl_fixtures_2022_2026.json` etc. Every league is doubled (EPL 3040, LaLiga
3040, Serie A 3042, Bundesliga 2462 — all exactly 2×). In the chronological replay, each match is
therefore predicted **twice**, and the second copy is predicted with Elo/form state that already
contains that match's **own result** → direct look-ahead leakage. The "improved" numbers are the model
predicting matches it already knows the outcome of.

**Reproduced on this machine** (same code path, production config):

```
[A] single EPL (n=1520):  ensemble ll=0.9886  roi=-0.0191  kelly_g=-11.48   ← the truth
[B] doubled EPL (n=3040): ensemble ll=0.946   roi=+0.2335  kelly_g=+35.24   ← the bug
    report claimed:        n=3040  ll=0.9424   roi=+0.3166  kelly_g=+52.74
```

**Issue → root cause → change → test → success metric:**
- **Issue:** the only positive validation result in the project is fabricated by look-ahead leakage.
- **Root cause:** `validate-multileague` loads the same fixtures twice (aggregate file + per-league files) with no dedupe.
- **Change:** dedupe by `(league, date, home, away)` in `validate_multileague()` (or skip the glob when `multileague_fixtures.json` exists). Add a global invariant in `validate_multileague`: report `n` must equal the count of unique `(league, date, home, away)` keys; refuse to emit a report otherwise.
- **Test:** unit test feeding `multileague_fixtures.json` + the per-league files and asserting EPL `n == 1520`; plus a regression test asserting the doubled input is deduplicated before replay.
- **Success metric:** re-run reproduces the single-pass numbers exactly (n=1520, ll 0.9886, ROI −1.9%, `ll_le_market: false`), and the `ll_le_market` flag never appears for EPL again.

**Also fix the ML path's near-miss:** `ml_train.load_frames` reads `multileague_fixtures.json` *and* the
per-league files but does dedupe with `drop_duplicates(subset=["Date","Home","Away","league"])` — that
one is safe *today*, but the two loaders must share one dedupe helper so they can never drift.

---

## 3. 🟠 P1 — Edges are measured against a benchmark the model demonstrably cannot beat

**Issue:** the live bot reports "edge" vs **soft pre-match consensus** (OddsPapi/NowGoal/TheOddsAPI —
config `edge_benchmark: soft_consensus`, correctly labelled "bukan closing line"). The backtest that
proves the model unprofitable uses **Pinnacle closing**. Those are different games.

**Root cause:** soft consensus (retail books, 30-min polling, mixed freshness) lags the sharp line and
carries a wider margin. An edge vs soft consensus is systematically overstated — it is the price
*you cannot actually take*; by kickoff the sharp line will have moved against you. The bot has no
executable "closing" benchmark at all in production.

**Live example from the bot's own log** (`predictions.jsonl`, Singapore vs Thailand, UCL qualifier):
model says home 48.9% vs market 5.20 → **edge +26.5pp**, confidence 0.73, signal 81 — on a match where
`elo_seeded: false` (both teams at the 1500 prior) and the form window is **one match**. This is not
value; it is the market being right and the model being wrong, wearing a "+26pp edge" costume. The
decision engine correctly output NO BET — the only thing that saved it.

**Change:** never display or log an "edge" without its benchmark identity (already partially done via
`edge_benchmark` — make it *enforced*, not descriptive). Add the closest-to-close line (last snapshot
before kickoff; or NowGoal `t=11` after settle) as the primary comparison for every pick, and report
`price_clv = close/prediction − 1` per pick as the only number that means anything.
**Test:** a logged pick's displayed edge must never be computed from a benchmark newer than the pick.
**Success metric:** per-pick CLV distribution; when the median price CLV is ≤ 0 (it will be, until the
model improves), the "edge" language on the card is downgraded to "model deviation".

---

## 4. 🟠 P1 — The core economics: "value picks" are where the model is wrong

This is the single most important thing a bookmaker needs to hear about this bot:

> **When your model's log-loss is worse than the market's, your model's deviations from the market are,
> on average, errors. The engine's "edge picks" (model_prob > market implied) are therefore
> systematically the places where the model is wrong.**

Mechanism, in the repo's own numbers: ensemble log-loss 0.9851 vs market 0.9652 (with xG). The model
is a *slightly worse* forecaster than the closing line. The decision engine only acts where the model
disagrees with the market by ≥3pp — i.e. exactly the subset where the model's relative error is
concentrated. The Phase-7 validation confirms the result: decision ROI negative in aggregate,
non-monotonic across score terciles, unstable across seasons. The "edge" that the engine hunts for is
an anti-signal, not a signal. A bookmaker's optimal strategy against this bot is trivial: **fade every
displayed value pick**. It wins in expectation.

**Change (philosophical but enforced in code):** an edge may only drive a recommendation after the
segment shows *realized* positive CLV and ROI out-of-sample — which is exactly what the CLV gate and
edge-bucket gate already implement. Keep them on. The gates are the only thing standing between this
bot and losing money, and today they correctly block everything.

---

## 5. 🟠 P1 — False confidence is real and observable

The prompt asks: *can the signal/confidence engine generate false confidence?* **Yes.** Evidence from
the live log: `confidence: 0.731, signal: 81, edge +26.45pp` on a match with unseeded Elo (1500 prior
both sides), a 1-match form window, and no calibration lineage beyond a per-league curve of dubious
provenance (see §7). The confidence weights are explicitly unvalidated in config
(`weights_validated: false`), and the only validation that exists (Spearman 0.0015) says the score
does **not** rank outcomes. The `market_tiers`/`signal_engine` stack adds a *third and fourth*
unvalidated scoring layer on top of the unvalidated Decision Score.

The NO BET output saves the user, but the displayed confidence/signal numbers are not calibrated to
anything, and a user (or a downstream automation) reading "signal 81" as "the model is sure" is being
misled by construction.

**Change:** derive confidence labels from *bucket-calibrated realized outcomes* (e.g. bins of the
logged score → realized hit rate/ROI with Wilson CIs), and refuse to print a confidence label for a
bin with < 30 settled bets. Until then, print the raw score with the explicit label "unvalidated
heuristic". **Success metric:** label calibration — mean realized hit rate of "HIGH" bins > "MEDIUM"
bins > "LOW" bins, out-of-sample, with non-overlapping CIs.

---

## 6. 🟠 P1 — The entire CLV apparatus currently runs on zero evidence

Live log state (2026-08-16): 167 prediction snapshots, 14 settles, 304 odds snapshots, **0 settles
carrying closing odds**. The CLV segment report itself says: `closing_coverage_pct: 0.0`,
`passed: false`; the edge-bucket audit has `n_with_closing: 0` in every bucket.

Consequences:
- The CLV hard gate, the edge-bucket gate, and paper-trade graduation all evaluate **empty** segments.
  Gates "pass" vacuously (no data → not blocked) or block everything. Either way, **no gate has ever
  seen a single real CLV number**, so no segment can ever graduate, and — equally important — nothing
  is measuring whether the "soft consensus" edges the bot displays are real. The single most useful
  metric a betting bot can collect is being collected but not fed.
- Root cause: the closing-odds fetch at settle (NowGoal `t=11`) is evidently failing/empty on this
  network, and there is no fallback.
- **Change:** (a) make `settle` fetch the closing line from football-data.co.uk Pinnacle columns
  (already available for settled dates) as a fallback; (b) log `closing_fetch_errors` per settle so a
  silent 0% is impossible; (c) make the CLV report a required daily artifact, not an optional one.
- **Success metric:** closing-odds coverage ≥ 80% of settles; then a real per-segment CLV table.

---

## 7. 🟡 P2 — Calibration: in-sample, near-identity, and one unexplained outlier

- The global/EPL calibrator is fit **in-sample** on the same walk-forward pairs it is then evaluated
  against (`validate.py` fits `calibrator.fit(ens_probs, ens_outs)` on the aggregate and reports the
  "calibrated ECE" on those same pairs). The repo labels this honestly ("fit is in-sample"). The curve
  is near-identity (a=0.008, b=1.013), so the practical harm is small — but the "ECE 0.0042–0.011"
  claims are in-sample and should be labelled as such on every card.
- **Outlier:** `cache/football/calibration_ucl.json` = `{a: 0.293, b: 1.449, samples: 564}`. A strong,
  non-identity curve with a sample size that **cannot be reproduced from the current live log** (only
  ~14 settled matches exist; 564 pairs would need ~188). Its provenance is unknown to the repo's own
  standards, and it is applied to UCL predictions (league_min_samples=400, so UCL's 564 "qualifies").
  A b=1.45 stretch on unseeded-Elo UCL qualifiers is exactly the kind of false precision that inflates
  "edge" numbers (Singapore +26pp). **Verify where this file came from, or delete it and force MARKET
  PRIOR/NO BET for UCL** until a reproducible fit exists.
- Per-league calibrators exist for leagues with real historical data (Bundesliga 3,672, Süper Lig
  4,110, Eredivisie 3,672, etc.), but **none of those leagues has historical odds in the cache** (the
  multileague report's `data_missing` rows for the 7 target leagues confirm it), so their calibration
  is untestable against the market. Calibrated against what you can't bet is a nice property; it is
  not evidence of edge.

---

## 8. 🟡 P2 — Staking: the math is right, the inputs are wrong

`staking.py` implements ¼-Kelly with a 2% bankroll cap and auto-decline on extreme edges — that is a
sound, conservative staking layer. But Kelly's output is only as good as `p`. The repo's own Kelly
diagnostics are the correct verdict: **negative log-growth for every model in every configuration ⇒
the statistically correct stake is 0**. The staking module is harmless today only because the decision
gates (correctly) never emit an actionable pick. Keep the module; never let it receive a pick that has
not cleared the realized-CLV evidence gates (currently impossible — good).

---

## 9. 🟡 P2 — Bookmaker exploit list (how a sharp operator beats this bot)

1. **Fade every displayed edge.** The model loses to the closing line; its deviations are error, not
   value (§4). Betting the opposite of every STRONG/GOOD pick the bot *would* emit is +EV for the house.
2. **Exploit the soft-consensus lag.** The bot's edges are measured vs retail consensus that lags the
   sharp line. Take the sharp side early; the bot's "edge" will be gone (negative CLV) by close —
   the CLV apparatus will prove this the moment closing capture works.
3. **Feed it thin data on unseeded leagues.** UCL qualifiers, Saudi Pro League, Liga 1, internationals:
   unseeded 1500 Elo + 1-match form + (for UCL) an aggressive b=1.45 calibration = wild edges
   (+26pp on Singapore). The model has no information there, and its displayed confidence is
   fabricating it. The NO BET gate is the only protection.
4. **Move the soft books.** With `require_movement_agreement: false`, the bot's movement component is
   decorative. A sharp operator can nudge soft prices toward the model's side to flatter its edge,
   then settle at the true line.
5. **Lineup/injury news.** The model weights lineups, rest days, and team context at **0.0**. Publish
   a key absence after the bot's snapshot and take the other side of the stale probability. The bot
   has no information the sharp market doesn't have first.

---

## 10. Component disposition — KEEP / FIX / REDUCE / REWRITE / REMOVE

### ✅ KEEP (sound, evidence-backed, don't touch)
- **Leak-free chronological walk-forward** (`validate.py` single-pass replay; update strictly after prediction).
- **Margin-free implied probability everywhere** (consistent overround removal; never vig-free vs raw mixing).
- **Model A (odds-derived) vs Model B (independent) separation** — the independent engine never consumes market odds as features; this is the one structural choice that makes any future edge claim meaningful.
- **NO BET / NO CLEAR DECISION / MARKET PRIOR as first-class outputs**, with the thin-data honesty floor.
- **Hard evidence gates**: CLV gate, edge-bucket-vs-closing gate, sample-size + Wilson-CI gates. Today they block everything, which is correct. These are the only components that make the bot safe to run.
- **Append-only prediction log + settle/CLV/segment accounting** (`prediction_log.py`) — the best data infrastructure in the project; the CLV story just needs data (§6).
- **`leakage_audit.py`** and the module-scoped CLV restriction list.
- **Kelly diagnostics in validation** — the honest "g ≤ 0 ⇒ stake 0" verdict.
- The repo's **reporting culture** (pre-registered experiments, rejected-experiment log, honest notes).

### 🔧 FIX (high priority)
1. **P0 — `validate-multileague` double-load leak** (§2). This is the #1 fix in the project: it fabricates the only positive result.
2. **P1 — Closing-line capture at settle** (0% coverage today, §6) with a football-data.co.uk Pinnacle fallback and per-settle error logging.
3. **P1 — Edge-benchmark enforcement** (§3): the displayed edge must always state its benchmark; add close-based CLV per pick.
4. **P2 — `calibration_ucl.json` provenance** (§7): verify or delete; add a reproducibility check to `calib-refresh` (store the log path + line range a fit came from).
5. **P2 — Same-day kickoff ordering** in the backtest replay (`kickoff_sort_key` exists; the day-granularity caveat is still documented as open in 1,060/1,520 EPL matches).
6. **P2 — Dedupe the two fixture loaders** behind one shared function (runner glob vs `ml_train.load_frames`).

### 📉 REDUCE
- **The four-layer scoring stack** — SignalScorer confidence, Decision Score, market tiers, signal-engine
  weights — is four overlapping unvalidated heuristics computing different numbers from the same
  inputs. Reduce to **one** evidence gate (realized CLV/ROI by segment) + **one** display layer.
  The config even admits this: `weights_validated: false` on both the decision weights and the signal-engine weights.

### ♻️ REWRITE
- **Confidence/decision scoring** — replace heuristic weights with outcome-derived calibration: bin the
  logged score, measure realized ROI/CLV per bin with Wilson CIs, and let only bins with positive
  realized evidence drive actionability (this is the Phase-7 null result, Spearman 0.0015, being
  honored in code rather than papered over).

### ❌ REMOVE / RETIRE
- **Any "beats market" claim** produced by the multileague harness until §2 is fixed (the only
  occurrence in the repo today is the leaked one).
- **Movement/steam claims** (`movement_accuracy`, steam side) — currently run on ~no settled data;
  retire the output until it has ≥ a few hundred settled matches and a win rate above implied.
- **ML models as "improvement" evidence** — README already says 1X2 logloss ≈ 1.06 ≈ base rate. Keep
  them as a de-correlated signal input only if they add calibration value; they are not edge evidence.
- **The `predictor.py` odds-derived Poisson "Model A"** — reference-only is fine; it must never enter value math (already enforced).

---

## 11. Prioritized plan (issue → root cause → change → test → success metric)

### Phase A — stop lying to yourself (this week)
| | |
|---|---|
| **Issue** | Multileague validation reports a profitable, market-beating model. |
| **Root cause** | Fixtures loaded twice → each match predicted with its own result in state. |
| **Change** | Dedupe `(league, date, home, away)` in `validate_multileague`; shared loader helper; refuse to emit without dedupe check. |
| **Test** | Input doubled fixtures, assert EPL `n == 1520`; re-run `validate-multileague`, assert it reproduces single-pass numbers. |
| **Success metric** | Report matches `validate` exactly: n=1520, ll 0.9886, ROI −1.9%, `ll_le_market: false` everywhere. |

### Phase B — get real prices (1–2 weeks)
| | |
|---|---|
| **Issue** | 0% closing-odds coverage; all CLV gates run on empty evidence; live edges vs soft consensus are unverifiable. |
| **Root cause** | Closing fetch at settle (NowGoal t=11) fails silently; no fallback; no per-settle error accounting. |
| **Change** | football-data.co.uk Pinnacle closing as settle fallback; per-settle `closing_fetch_errors`; make CLV report a required daily artifact. |
| **Test** | Settle a known past date; assert ≥80% of settles carry closing odds. |
| **Success metric** | `closing_coverage_pct ≥ 80`; first real per-segment CLV table. Expected reading: price CLV ≤ 0 for soft-consensus "edges" — the honest confirmation of §4. |

### Phase C — evidence-based decisioning (1–2 months)
| | |
|---|---|
| **Issue** | Confidence labels and decision scores don't rank outcomes (Spearman 0.0015); four unvalidated scoring layers. |
| **Root cause** | Heuristic weights chosen as priors, never replaced by outcome data. |
| **Change** | Collapse to one gate: realized CLV/ROI per (league × market × tier) with Wilson CI; confidence labels derived from settled score bins; pre-register segments before they may act. |
| **Test** | No segment graduates without ≥200 settled bets, positive ROI and positive price CLV with CI half-width ≤ 5pp. |
| **Success metric** | Graduated segments show positive OOS ROI/CLV; until then NO BET is the only possible output. |

### Phase D — only after C says yes (future)
Lineups/injuries as features (currently weight 0 — the single biggest information gap vs a sharp book),
recency-weighted Elo, model stacking — each behind the same walk-forward + leakage-audit + realized-CLV
discipline. The bar is not "improves log-loss"; it is "produces positive realized CLV out-of-sample".

---

## 12. What is genuinely good (say it, because it's true)

- The independent model is **not** a market tracker — Elo+Poisson features contain no odds. That is
  the correct foundation for any future edge claim.
- The team repeatedly chose "no improvement" over unvalidated changes and documented the rejects.
- The gates that block all betting today are the correct response to the evidence, not over-engineering.
- The prediction log and settlement/CLV accounting are production-grade and will pay off the moment
  closing prices flow (§6).
- The leak found in §2 is a harness bug, not a model bug — and it is exactly the class of bug a
  serious betting operation must kill before trusting any validation output.

**Final verdict:** as a *forecast* engine, this is calibrated and honest. As a *betting* engine, it
has no edge, and its only positive evidence was fabricated by a data-leak bug. The correct posture is
the one its own gates already enforce: read-only, NO BET default, edges shown as "model deviations vs
soft consensus" — not opportunities — until Phases A–C produce positive realized CLV out-of-sample.
No amount of Elo tuning, xG blending, or weight tweaking changes the conclusion; the market is simply
better than this model, and the model's job for now is to keep measuring that gap honestly.

---

## 13. Implementation log — P0 fix applied and verified (2026-08-16)

### Fix 1 — Multi-league double-load leak (`validate_multileague` dedupes)
- **`agents/football/validate.py`**: new `fixture_identity()` + `dedupe_fixtures()` (keep-first by
  `(league, date, home, away)`); `validate_multileague()` dedupes each league before replay and
  reports `n_duplicates_removed` per league in the artifact. This is the single dedupe point.
- **`agents/football/runner.py`**: `validate-multileague` now emits an **ERROR**-level log line when
  `n_duplicates_removed` is non-empty (the root logger is configured at ERROR, so a plain warning
  was silently dropped — verified before switching).
- **Why keep-first:** the aggregate file and the per-league caches carry identical essential fields;
  keeping the first occurrence is arbitrary but deterministic and documented.

### Fix 2 — Train/serve parity in the multileague harness (found while verifying Fix 1)
- `validate_multileague()` was called with **no model config**, so it evaluated the library-default
  ensemble (elo 0.5 / poisson 0.5) instead of production (0.7 / 0.3) — the same class of bug the
  fresh-audit backtest-parity fix addressed, but missed here. `validate_multileague()` now loads
  `_load_model_config()` (production config) when no config is passed; explicit overrides still win.

### Regression tests (5 new, in `tests/test_validate.py`)
1. `test_dedupe_fixtures_keeps_first_and_counts` — duplicates dropped, first kept, count returned.
2. `test_dedupe_fixtures_empty_and_missing_keys` — empty/None/key-less inputs are safe.
3. `test_validate_multileague_dedupes_doubled_input` — doubled input ⇒ `n` = unique count, not 2×;
   `n_duplicates_removed` recorded (the exact 2026-08-16 leak scenario).
4. `test_validate_multileague_unduplicated_input_reports_zero_removed` — clean input, no removal.
5. `test_validate_multileague_uses_production_ensemble_config` — harness ensemble row is
   byte-identical to `run_multi_season_validation` with the production config.

### Verification (re-run on this machine, `python -m agents.football.runner validate-multileague`)

| League | Before (buggy report) | After fix (deduped + production config) |
|---|---|---|
| EPL | n=3040, ll 0.9424, ROI **+31.7%**, g +52.7, beats market | n=**1520**, ll **0.9886**, ROI **−1.9%**, g **−11.48**, `ll_le_market: false` |
| LaLiga | n=3040 | n=1520 |
| Bundesliga | n=2462 | n=1231 |
| Serie A | n=3042 | n=1521 |

Artifact now carries `n_duplicates_removed: {EPL: 1520, LaLiga: 1520, Bundesliga: 1231, Serie A: 1521}`
and the run logs `ERROR hermes-football: validate-multileague: removed duplicate fixture rows ...`.
The corrected EPL ensemble numbers are byte-identical to the repo's canonical single-pass reference
(fresh audit: ll 0.9886, ROI −1.9%, Kelly g −11.48, 600 bets). Full suite: **1177 passed** (1172 +
5 new; `tests/test_ml.py` excluded — it fails at collection because `sklearn` is not installed in
this environment, a pre-existing environment gap unrelated to this change).
