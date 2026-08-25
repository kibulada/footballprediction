====================================================================================
BASELINE FREEZE — baseline_freeze_v2
====================================================================================
generated_at   : 2026-08-11T19:46:00+00:00
model version  : ensemble-v1 (elo 0.5 + dixon-coles poisson 0.5; dc_rho=-0.1; xg_weight=0.65; rest_days_k=0.0)
feature version: match-context-v1 (form5 W/D/L, rolling GF/GA, rest days, margin-free consensus odds; xG optional)

--- Data snapshot ---
  file          : cache\football\epl_fixtures_2022_2026.json
  sha256        : 249ad217163c3168…
  matches       : 1520
  leagues       : EPL
  seasons       : 2022-2023, 2023-2024, 2024-2025, 2025-2026
  date range    : 2022-08-05 → 2026-05-24
  per season    : {"2022-2023": 380, "2023-2024": 380, "2024-2025": 380, "2025-2026": 380}

--- CLV ---
  n/a: the historical dataset carries no closing odds; CLV is only tracked live via prediction_log settlement lines. Never fabricated for the freeze.

--- Walk-forward validation ---
====================================================================================
MULTI-SEASON VALIDATION (strict chronological walk-forward)
====================================================================================
One chronological walk-forward pass; Elo/form state carries across seasons; CIs are approximate 95% normal-approximation intervals on per-match values; ECE is computed on pooled per-outcome calibration pairs (3 per match); calibrated ECE is fitted in-sample (not out-of-sample); market row = margin-free implied probabilities from historical odds (only when the dataset provides them); ROI is flat-stake on the best 1X2 pick with margin-free edge >= threshold; matches are ordered by date only, so same-day matches may be processed in an arbitrary order (FBref provides no kickoff time) -- a known day-granularity caveat; beats-market compares model log-loss (all matches) against market log-loss (odds-carrying matches only; the subsets coincide under full odds coverage).

--- Season 2022-2023 ---
model           n        logloss          brier         ece          hit%      roi
------------------------------------------------------------------------------------
baseline      380   1.1638±0.147   0.6450±0.024      0.0357     48.2%±5.0    11.5%
elo           380   0.9806±0.043   0.5841±0.031      0.0279     53.2%±5.0    18.0%
poisson       370   1.0430±0.043   0.6270±0.029      0.0166     48.6%±5.1     6.4%
dc            370   1.0484±0.042   0.6299±0.028      0.0234     48.4%±5.1     7.2%
ensemble      380   0.9888±0.039   0.5899±0.028      0.0310     55.0%±5.0    25.0%
market        380   0.9663±0.049   0.5740±0.035      0.0283     55.8%±5.0      n/a

--- Season 2023-2024 ---
model           n        logloss          brier         ece          hit%      roi
------------------------------------------------------------------------------------
baseline      380   1.0565±0.031   0.6388±0.021      0.0196     46.1%±5.0   -13.2%
elo           380   0.9525±0.052   0.5631±0.036      0.0076     55.5%±5.0   -12.7%
poisson       377   0.9942±0.046   0.5919±0.032      0.0344     53.3%±5.0    -8.6%
dc            377   0.9967±0.045   0.5932±0.031      0.0325     53.6%±5.0    -4.8%
ensemble      380   0.9503±0.046   0.5617±0.033      0.0238     57.1%±5.0   -13.3%
market        380   0.9085±0.051   0.5326±0.036      0.0274     59.0%±5.0      n/a

--- Season 2024-2025 ---
model           n        logloss          brier         ece          hit%      roi
------------------------------------------------------------------------------------
baseline      380   1.0843±0.028   0.6581±0.019      0.0345     40.8%±4.9   -26.2%
elo           380   1.0016±0.057   0.6004±0.040      0.0423     52.1%±5.0    -8.8%
poisson       379   1.0351±0.044   0.6216±0.031      0.0232     48.0%±5.0    -8.1%
dc            379   1.0364±0.043   0.6218±0.030      0.0341     47.5%±5.0    -6.9%
ensemble      380   0.9933±0.049   0.5945±0.035      0.0213     51.3%±5.0   -16.3%
market        380   0.9703±0.050   0.5788±0.035      0.0180     53.9%±5.0      n/a

--- Season 2025-2026 ---
model           n        logloss          brier         ece          hit%      roi
------------------------------------------------------------------------------------
baseline      380   1.0839±0.027   0.6560±0.018      0.0271     42.6%±5.0    -3.9%
elo           380   1.0310±0.055   0.6207±0.038      0.0513     48.4%±5.0    -5.0%
poisson       379   1.0639±0.039   0.6446±0.028      0.0283     46.7%±5.0     6.6%
dc            379   1.0637±0.037   0.6443±0.026      0.0247     46.2%±5.0     3.0%
ensemble      380   1.0220±0.046   0.6144±0.032      0.0291     48.9%±5.0    -2.9%
market        380   1.0158±0.044   0.6103±0.032      0.0283     49.5%±5.0      n/a

--- AGGREGATE (all seasons, chronological pool) ---
model           n        logloss          brier         ece          hit%      roi
------------------------------------------------------------------------------------
baseline     1520   1.0971±0.039   0.6495±0.011      0.0196     44.4%±2.5    -7.5%
elo          1520   0.9914±0.026   0.5921±0.018      0.0205     52.3%±2.5    -2.7%
poisson      1505   1.0341±0.022   0.6213±0.015      0.0157     49.2%±2.5    -0.5%
dc           1505   1.0363±0.021   0.6223±0.014      0.0174     48.9%±2.5    -0.1%
ensemble     1520   0.9886±0.023   0.5901±0.016      0.0110     53.1%±2.5    -1.9%
market       1520   0.9652±0.024   0.5739±0.017      0.0084     54.5%±2.5      n/a

--- Calibration (ensemble) ---
  raw ECE             : 0.011
  calibrated ECE      : 0.0104
  fit is in-sample    : True (samples=4560)
  fitted params       : a=0.0082, b=1.0129

--- Consistency (log-loss) ---
  baseline  : beats baseline in 0/4 seasons (worse: 2022-2023, 2023-2024, 2024-2025, 2025-2026)
              beats MARKET in 0/4 seasons (worse: 2022-2023, 2023-2024, 2024-2025, 2025-2026)
  elo       : beats baseline in 4/4 seasons (worse: none)
              beats MARKET in 0/4 seasons (worse: 2022-2023, 2023-2024, 2024-2025, 2025-2026)
  poisson   : beats baseline in 4/4 seasons (worse: none)
              beats MARKET in 0/4 seasons (worse: 2022-2023, 2023-2024, 2024-2025, 2025-2026)
  dc        : beats baseline in 4/4 seasons (worse: none)
              beats MARKET in 0/4 seasons (worse: 2022-2023, 2023-2024, 2024-2025, 2025-2026)
  ensemble  : beats baseline in 4/4 seasons (worse: none)
              beats MARKET in 0/4 seasons (worse: 2022-2023, 2023-2024, 2024-2025, 2025-2026)

--- Missing data ---
  Poisson/DC skip matches lacking pre-match form features
  (early-season / newly promoted teams).
  2022-2023: baseline=380, poisson_skipped=10, dc_skipped=10
  2023-2024: baseline=380, poisson_skipped=3, dc_skipped=3
  2024-2025: baseline=380, poisson_skipped=1, dc_skipped=1
  2025-2026: baseline=380, poisson_skipped=1, dc_skipped=1

CIs: approximate 95% normal-approximation on per-match values.
ROI: flat-stake 1X2 bets, best pick with margin-free edge >= 2% (real historical odds).
Risk (aggregate ensemble, stake units): max drawdown -55.8400, longest losing streak 10 bets.
NOTE: lower log-loss/Brier/ECE is better; higher hit% is better.
NOTE: ECE uses pooled per-outcome calibration pairs (3 per match).

--- Regression reference (aggregate ensemble) ---
  log loss          : 0.9886
  hit rate          : 0.5309
  ROI (edge>=2%)    : -0.0191
  max drawdown      : -55.84 (stake units)
  losing streak     : 10 bets
  Any candidate change must beat THESE numbers out-of-sample (walk-forward, untouched future data) to enter production.