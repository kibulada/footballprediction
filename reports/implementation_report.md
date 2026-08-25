# AUDIT PLAN — IMPLEMENTATION REPORT

**Date:** 2026-08-12 · **Plan source:** `reports/full_audit_plan.md`
**Validation:** `547 passed` (527 pre-existing + 20 new tests), leakage-audit CLV scope **PASS**, every module compiles.

All changes were made backwards-compatibly: every new behaviour is either a
default-off flag or an additive field, so existing output is unchanged until a
flag is enabled. Nothing in `config/football.json` was flipped on; adoption of
any behavioural change still requires the evidence gates from the plan.

---

## Phase A — Correctness & train/serve parity

### ✅ TODO-01 — Live Elo update loop
- **Files:** `elo.py`, `settler.py`, `runner.py`
- **What:** `EloModel.update_from_results()` applies a batch of settled results and persists once, resolving live spellings to seeded keys, skipping unparseable rows (never guessed), and applying results in **kickoff order** (Elo updates are order-dependent; matches validation).
- **Wiring:** the runner `settle` mode (manual and auto) now updates the live `cache/football/elo.json` after settling. `settle_auto`/`settle_manual` now return explicit `home_goals`/`away_goals`.
- **Tests:** `test_elo_update_from_results_persists`, `test_elo_update_from_results_no_results_no_write`.
- **Leakage:** update runs only after results are final and before the next prediction is built (settle path only).

### ✅ TODO-02 — Calibration & empirical bucket refresh
- **Files:** `calibration.py`, `prediction_log.py`, `runner.py`
- **What:** `Calibrator.refresh_from_log()` re-fits the curve from the LIVE prediction log (`prediction_log.calibration_pairs` — one `(p_side, outcome)` pair per 1X2 side of every settled snapshot; strictly prediction-then-outcome). Guards: skipped below `min_samples`; **backs up to `<path>.bak` only when a refit will actually happen**; a refit whose ECE is worse than the existing snapshot is **kept** (status `kept`, old params restored) — never a silent downgrade.
- **CLI:** `python -m agents.football.runner calib-refresh [--min-samples N]`.
- **Tests:** `test_calibration_pairs_shape`, `test_calibrator_refresh_from_log`, `test_calibrator_refresh_keeps_better_snapshot`.

### ✅ TODO-03 — Feature-window parity (form 5)
- **Files:** `multi_source.py`, `flashscore.py`
- **What:** live form window default `limit=10 → 5` (module constant `FORM_WINDOW = 5`), matching the backtest's `deque(maxlen=5)` so the Poisson time-decay sees the same distribution in production as in validation. xG history (5, league-only) already matched.
- **Tests:** `test_form_window_parity_live_default_is_five` (signature + constant assertion).
- **Known limitation (documented):** the live form fetch still spans all competitions; league-only scoping in live requires per-row competition metadata and is a follow-up.

---

## Phase B — Odds/market integrity & backtest temporal airtightness

### ✅ TODO-04 — Totals/BTTS per-bookmaker-pair line fix
- **Files:** `analyse.py` (`extract_market_totals` + inline-loop removal)
- **What:** Over/Under (and BTTS Yes/No) pairs now come from the **same bookmaker** — the one with the smallest margin on that line — so margin is removed exactly once. The old best-of-both-sides behaviour (max price per side across bookmakers) double-removed margin and inflated the model's apparent edge. The single inline duplicate in `find_specific_match` was replaced with the shared function.
- **Tests:** `test_totals_pairs_come_from_same_bookmaker`, `test_totals_pair_requires_both_sides`.

### ✅ TODO-05 — Same-day kickoff-time ordering in all replays
- **Files:** `timeutil.py` (new `kickoff_sort_key`), `validate.py`, `backtest.py`, `decision_validation.py`, `market_audit.py`, `leakage_audit.py`
- **What:** every walk-forward replay now orders fixtures by `(date, has_kickoff_time, kickoff_utc)`. A bare `YYYY-MM-DD` string is treated as **unknown kickoff time** (sorted after timed matches on the same date, stable within the group) — fixing the subtle bug where `fromisoformat("2024-08-17")` parsed as midnight and leaked date-only fixtures ahead of real kickoffs.
- **Tests:** `test_kickoff_sort_key_orders_same_day_by_time`, `test_kickoff_sort_key_stable_between_dates`.

### ✅ TODO-06 — Odds freshness / exchange separation audit
- **Files:** `market_audit.py`
- **What:** `run_market_audit` now reports `odds_quality`: matches with full 1X2, median/max overround, % of matches with overround > 1.10 (thin/illiquid flag), and the exchange-separation note (consensus bookmaker price set only; exchange prices would surface as anomalous overrounds).
- **CLI:** unchanged (`python -m agents.football.market_audit`).

### ⏳ TODO-07 — Multi-league historical odds cache builder (infra shipped; data pending)
- **Files:** `runner.py` (new `cache-odds` mode)
- **What:** `python -m agents.football.runner cache-odds --leagues EPL,LaLiga,... --seasons ...` downloads per-league history via `odds_history.load_history_fixtures` and writes `cache/football/backtest/<league>_fixtures_<season>.json`.
- **Status:** the command works and degrades gracefully, but **this environment has no network access to football-data.co.uk** (`curl: (7) Failed to connect`), so caches could not be populated here. Run it on a networked machine, then execute TODO-08.

### ⏳ TODO-08 — Multi-league decision validation (infra shipped; data pending)
- **Files:** `decision_validation.py`
- **What:** `--fixtures` now accepts a **comma-separated list** of per-league cache files; fixtures keep their own `league`/`season` so the report buckets them separately.
- **Status:** requires the TODO-07 caches; run `python -m agents.football.decision_validation --fixtures cache/football/backtest/EPL_....json,cache/football/backtest/LaLiga_....json`.

---

## Phase C — Decision engine evidence

### ⏳ TODO-09 — Decision reliability gates (implemented, opt-in)
- **Files:** `decision.py`, `analyse.py`
- **What:** `decide()` gains `enable_watch` + `uncertainty` params (defaults preserve behaviour). When enabled, would-be bets on high ensemble spread or thin bookmaker liquidity are downgraded to **WATCH**; a hard-guard failure with real positive value becomes WATCH instead of NO CLEAR DECISION. **Config-gated** (`models.decision.enable_watch`, default off in `config/football.json`) — cannot fire accidentally.
- **Tests:** `test_high_uncertainty_downgrades_to_watch_only_when_enabled`, `test_decision_defaults_unchanged_without_watch`.
- **Status:** shipped but **not enabled** — per plan, enable only after multi-league decision validation (TODO-08) supports it.

### ✅ TODO-10 — EV under uncertainty (variance-aware EV band)
- **Files:** `decision.py`, `models.py`, `analyse.py`, `format.py`
- **What:** when ensemble spread is available, the decision reports `ev_band = {ev_low, ev_high, ev, uncertainty}` (from `p ± spread × odds − 1`) and the Discord output renders it (`📉 EV band [...]`). Spread comes from `Ensemble.predict` (max−min across component home probabilities), surfaced as `model_probs.uncertainty`.
- **Tests:** `test_ev_band_reported_with_uncertainty`, `test_ensemble_spread_reflects_disagreement`.

---

## Phase D — Model experiments (instrumented; NOT adopted)

### ✅ TODO-11 — Dynamic home advantage estimator
- **Files:** `elo.py` — `estimate_home_advantage(results, min_samples)` (prior-level estimator: observed home-win share → rating gap). Not wired into production; adopt only via walk-forward ablation evidence.
- **Tests:** `test_estimate_home_advantage`.

### ✅ TODO-12 — Recency-weighted Elo
- **Files:** `elo.py` — `k_multiplier_for_gap(days, half_life)` + optional `k_multiplier` on `update()`. Experimental, off by default.
- **Tests:** `test_k_multiplier_for_gap`, `test_elo_update_k_multiplier_scales_movement`.

### ✅ TODO-13 — Ensemble spread + logistic stacking
- **Files:** `models.py` — `spread` on `Ensemble.predict` (fed to the decision engine as `uncertainty`, used by TODO-10/16); `fit_stack_weights()`/`stack_probs()` walk-forward logistic stacker with a proper IRLS convergence check. Stacking is **not wired into production**.
- **Tests:** `test_ensemble_spread_reflects_disagreement`, `test_fit_stack_weights_and_apply`, `test_fit_stack_requires_sample`.

---

## Phase E — Maintenance automation & WATCH tier

### ✅ TODO-14 — One-command leakage audit
- **Files:** `runner.py` — `python -m agents.football.runner audit [--fixtures <json>]` runs `leakage_audit.audit_replay` over a local fixture cache (auto-discovers `cache/football/backtest/*.json`); returns a clean JSON report or a helpful error when no cache exists. The CLV-scope component is verified offline and **PASS**.

### ✅ TODO-15 — CLV-centric production tracking
- **Files:** `prediction_log.py`, `analyse.py`, `format.py`
- **What:** snapshots now record `decision_type` (logged after the decision engine runs); `compute_stats` adds a `by_decision` breakdown (n, bets, hit, ROI, model CLV, price CLV per STRONG/GOOD/LEAN/WATCH/NO BET/...); both the CLI and Discord stats renderers show the section (`📂 Per Decision Type`).
- **Tests:** `test_compute_stats_by_decision`.

### ✅ TODO-16 — WATCH tier (implemented, opt-in)
- **Files:** `decision.py`, `format.py`, `prediction_log.py`
- **What:** `WATCH` is a first-class decision type (👁 badge), meaning "positive value but insufficient reliability". Requires `market_value > 0` (a zero-EV candidate is NO BET, never WATCH — review fix). WATCH picks are included in `final_decision`, logged with `decision_type`, and tracked by TODO-15.
- **Tests:** covered by the TODO-09/10 tests.
- **Status:** same as TODO-09 — opt-in via config, off until validated.

---

## Phase F — Cleanup

### ✅ TODO-17 — `format.py` decomposition
- **Files:** new `format_utils.py` (date/odds/stat formatters) + `format_pages.py` (competition pagination); `format.py` re-imports every moved name **including private helpers** so `format._group_competitions` etc. keep working. Output is byte-identical (verified by the full test suite, incl. `test_top_pagination`).

### ✅ TODO-18 — Documentation
- This report + status appended to `reports/full_audit_plan.md`.

---

## MARKET PRIOR — thin-data honesty mode (2026-08-12, after review)

**Request:** untuk match data tipis (mis. kualifikasi UECL seperti GKS Katowice
vs Hapoel Tel Aviv — tanpa Elo seed, tanpa xG, tanpa riwayat), user ingin bot
tetap mengeluarkan prediksi (1X2 / Over-Under / BTTS) — bukan sekadar NO BET —
tapi tetap jujur dan tidak ngasal. Pilihan desain yang disetujui: **prediksi
selalu ada + label jujur** (opsi paling aman secara statistik).

**Desain:** saat engine independen (Elo+Poisson) tidak punya sinyal yang bisa
dipakai, estimator terbaik yang tersedia adalah market itu sendiri. Maka:

- `decision.market_prior_decision()` membangun prediksi 1X2 / Over-Under /
  BTTS **dari probabilitas market margin-free** — `edge = 0` secara
  konstruksi, `final_decision = None`, `betting_advice = "NO BET"`.
- Label eksplisit `MARKET PRIOR` (badge 📊) + blok prediksi market di output
  Discord + baris "🚫 saran taruhan: NO BET (prediksi = market → tanpa edge)".
- **Trigger (thin-data gate)** di `run_decision_engine`: engine tidak jalan ATAU
  `data_completeness < market_prior_min_completeness`. Floor default **0.6**,
  selaras dengan `min_completeness` bettable engine (tanpa "cliff" UX antara
  0.35–0.6 — perbaikan reviewer). Config: `models.decision.market_prior`
  (default off di kode, **on** di config/football.json) +
  `market_prior_min_completeness` (fallback otomatis ke `min_completeness`
  agar tidak drift).
- **Snapshot log jujur:** row MARKET PRIOR mencatat `prob/edge/best_pick/...`
  sebagai None, sehingga `compute_stats` tidak pernah menghitungnya sebagai
  bet atau prediksi model (reviewer finding #2). Odds tetap tercatat untuk
  price-CLV.
- **Normalisasi 1X2** atas sisi yang benar-benar di-pricing (odds draw yang
  hilang tidak membuat total < 1 — reviewer finding #3).

**Kejujuran yang dijaga:** prediksi = market berarti tidak ada klaim edge,
EV setelah margin bandar negatif, dan saran taruhan selalu NO BET — ini
bukan downgrade kualitas, tapi jawaban statistik yang benar untuk data tipis.

**File:** `decision.py` (`market_prior_decision`, `DECISION_TYPES`, floor),
`analyse.py` (thin-data gate + snapshot guard), `format.py` (badge + render),
`config/football.json`, `tests/test_plan_implementation.py` (8 tes MARKET
PRIOR). **Validasi: 553 passed** (termasuk 6 tes baru MARKET PRIOR).

---

## `top --days` — multi-day window for early-hours matches (2026-08-12)

**Problem:** `!football top` default hanya melihat SATU tanggal kalender WIB
(hari ini). Match yang kickoff 00:00–06:59 WIB secara kalender WIB sudah jatuh
pada tanggal BESOK, sehingga tidak pernah muncul di `top` — padahal `best`
(hari ini + besok) menangkapnya. Help text CLI `--date` juga salah tulis
("default H+1" padahal perilaku nyatanya hari ini).

**Change:** `find_top_matches` menerima `days` (default 1 = perilaku lama
persis) dan me-loop window tanggal WIB `[start .. start+days-1]`:

- Fixtures football-data di-fetch per tanggal WIB (cache key per
  league+tanggal, seperti sebelumnya).
- Odds The Odds API di-fetch **sekali per liga** (payload berisi semua
  commence time) lalu di-filter ke seluruh tanggal window.
- Dedupe lintas tanggal via key `(league, home, away, kickoff)`.
- Context flashscore homepage (kompetisi di luar football-data) hanya tampil
  jika hari ini WIB ada di dalam window.
- Payload menambah `days` + `date_range`; `format_top` menampilkan rentang
  tanggal di title saat `days > 1`.
- CLI: `!football top --days 2` (runner `--days` int arg), help `--date`
  diperbaiki menjadi "default hari ini WIB".

**Backward compatible:** default `days=1` = perilaku dan output lama. Tanpa
regresi: 556 tests passed (3 tes window baru di `test_plan_implementation.py`).

---

## Runner deadline handling — shared analysis budget (2026-08-12)

**Problem:** `analisa PSG vs Aston Villa` (dan match UCL lain) kadang mati
jauh sebelum hasil: bot membunuh runner di 90s, runner hard-exit di 85s
(`HERMES_RUNNER_DEADLINE`), dan provider chain (flashscore browser ->
football-data -> thesportsdb -> soccerdata) menumpuk per-provider HTTP
timeout (20s masing-masing) sehingga TOTAL-nya melewati deadline — user
hanya melihat "runner deadline 85s terlampaui (provider lambat/terblokir)".

**Fix — satu budget waktu BERSAMA untuk seluruh pipeline `analyse`:**

- `multi_source.py`: clock module-level (`set_analysis_budget` /
  `analysis_remaining` / `analysis_budget_exhausted`) + `_timeout_aware()`
  yang membungkus tiap provider call dengan `asyncio.wait_for` dan pada
  timeout **degrade ke None (bukan raise)** sehingga rantai fallback terus
  jalan. Cap per-call `_CALL_CAP=12s` — worst-case rantai dibatasi, tidak
  lagi (network timeout x panjang rantai).
- Budget guard dipasang di `search_team` / `search_teams_pair` /
  `fetch_team_form` / `fetch_h2h` / `fetch_upcoming_fixture`: saat budget
  hampir habis, langkah mahal di-skip dan caller mendapat best-effort
  (bukan gantung).
- `analyse.py` `find_specific_match` meng-armed budget dari
  `cfg["analyse"]["budget_seconds"]` (default 72.0, di bawah deadline 85s)
  dan men-skip langkah non-esensial saat budget menipis: flashscore match
  stats/lineups, understat xG history, fallback oddspapi. Keputusan engine
  & snapshot log TETAP jalan — user selalu mendapat laporan (sering MARKET
  PRIOR jujur) alih-alih error mati.
- `config/football.json`: `analyse.budget_seconds: 72.0` (dapat diubah).
- **Backward compatible:** tanpa clock (reset / config tidak ada), semua
  perilaku lama persis sama; `_timeout_aware` hanya membungkus call yang
  sebelumnya bisa menggantung. `best`/`bestgoalmatch` sudah punya local
  budget 55s sendiri — tidak diubah.

**Kejujuran yang dijaga:** budget-skip menurunkan data quality secara
terlihat (form/H2H/xG mungkin absen) dan decision engine melaporkannya
secara jujur (completeness rendah -> MARKET PRIOR / NO CLEAR DECISION) —
skipping tidak pernah membuat model mengaku punya sinyal padahal tidak.

**File:** `multi_source.py`, `analyse.py`, `config/football.json`,
`tests/test_plan_implementation.py` (5 tes budget + autouse fixture reset).
**Validasi: 560 passed.**

---

## Honest caveats

1. **Network-dependent TODOs (07/08) are infra-complete but data-pending**: this sandbox cannot reach football-data.co.uk, so the multi-league caches and the multi-league decision validation could not be executed here. Commands are ready; run on a networked host, then re-run the decision validation and only then enable `enable_watch`.
2. **Experiments (11/12/13) are instrumented, not adopted** — exactly per the plan ("adopt only with out-of-sample, multi-league, beyond-CI evidence; NO PRODUCTION IMPROVEMENT is an acceptable outcome").
3. **Behavioural flags are all OFF** in `config/football.json`. The only production behaviour that changed without a flag is the **live Elo update after settle** (TODO-01, a pure correctness fix) and the **form window 5** (TODO-03, train/serve parity) — both are what the validation harness already assumed.
4. The EPL leakage audit / baseline freeze still require the EPL fixture cache; the runner `audit` mode reports this gracefully.

---

## Post-plan offline validation run (2026-08-12)

**Network reality:** `cache-odds` cannot run here — football-data.co.uk is
unreachable from this environment (no proxy). However, local per-league caches
already exist: **EPL carries full historical odds (1520/1520 matches)**; LaLiga,
Bundesliga and Serie A are results-only (FBref, no odds). So the executable
validation battery is: multi-league leakage audit (4 leagues, all 5792
matches), EPL decision validation (the only odds-bearing cache), and EPL model
validation. Non-EPL decision validation still awaits a networked `cache-odds`
run.

### Multi-league leakage audit — `reports/multileague_audit_after_plan.json`

**VERDICT: PASS** (5792 matches across EPL/LaLiga/Bundesliga/Serie A)

- Provenance coverage: documented == actual (all 25 fields), `missing: []`, `extra: []`.
- CLV scope: **PASS** (no violations; only allowed modules reference closing prices).
- Invariants: `predict_before_update` PASS (0 violations) · `determinism_input_hash`
  PASS (0 hash diffs over 2 passes) · `pipeline_equivalence` PASS (audit LL 0.998
  == production LL 0.998, 5792 matches each).
- Known caveat (documented, not a production leak): 654 dates have multiple
  matches (5058 matches) and FBref gives no kickoff time, so same-day replay
  order is arbitrary — day-granularity limitation of the historical data; the
  live bot orders by kickoff timestamp.

### EPL decision validation — `reports/decision_validation_after_plan.json`

One chronological walk-forward pass, 1520 odds matches, identical inputs for all
rules (OLD = best-prob 1X2 with margin-free edge ≥ 2%; NEW = decision-score
engine, STRONG/GOOD only).

| Rule | Bets | Hit rate | ROI | Max DD |
|---|---|---|---|---|
| OLD (1X2, edge≥2%) | 623 | 41.25% | **−1.91%** | −55.84 |
| NEW (STRONG/GOOD, 1X2+totals) | 1057 | 46.26% | −4.51% | −104.23 |
| NEW 1X2-only | 370 | 40.27% | **−1.62%** | −48.27 |

By decision type: **STRONG** 107 bets · 58.88% HR · ROI −1.53% · **GOOD** 738 ·
44.85% · −6.71% · **LEAN** 212 · 44.81% · **+1.61%**. By season: 2022-23
**+13.9%** · 2023-24 −3.5% · 2024-25 −15.4% · 2025-26 −15.6%. 463 matches →
NO DECISION. WATCH: 0 bets (off by default, as intended).

**Read:** the decision engine's 1X2 universe is roughly flat-to-slightly-better
than the old rule (Δ +0.29pp ROI), and STRONG has the best hit rate — but
**nothing is positive out-of-sample**. This confirms the plan's core conclusion:
no betting edge over the market is demonstrated, and NO BET remains the honest
default. `enable_watch` stays OFF.

### EPL model validation — `reports/validation_epl_after_plan.json`

1520 matches, one walk-forward pass, real historical odds. Aggregate:

| Model | Log-loss | Brier | Hit rate | ROI | ECE |
|---|---|---|---|---|---|
| baseline (home-rate) | 1.0971 | 0.6495 | 44.41% | −7.50% | 0.0196 |
| Elo | 0.9914 | 0.5921 | 52.30% | −2.67% | 0.0205 |
| Poisson | 1.0341 | 0.6213 | 49.17% | −0.53% | 0.0157 |
| Dixon-Coles | 1.0363 | 0.6223 | 48.90% | −0.08% | 0.0174 |
| **Ensemble** | **0.9886** | **0.5901** | **53.09%** | −1.91% | **0.0110** |
| Market (margin-free) | 0.9652 | 0.5739 | 54.54% | — | 0.0084 |

Every model beats baseline log-loss in **4/4 seasons**; every model loses to the
market in **4/4 seasons**. Ensemble calibrated ECE 0.0104 (in-sample fit).

**Read:** the Elo+Poisson+DC ensemble is the best model component and clearly
beats the naive baseline everywhere — that part is real and preserved. But the
market remains better than every model, so the model's value is diagnostic
(calibration, uncertainty, market sanity-check), not a demonstrated betting
edge. Any claim of positive EV requires new out-of-sample evidence.
