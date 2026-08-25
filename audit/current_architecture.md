# Audit — Arsitektur Prediction Engine Saat Ini

Tanggal audit: 2026-08-11
Status: baseline di-snapshot ke `baseline/` (12 modul inti).

## 1. Alur Prediksi (live `analyse`)

```
query tim + liga
  → league_resolver (key odds API, sofascore tournament id)
  → multi_source provider chain (sofascore → football-data.org → thesportsdb)
      - search_teams_pair, upcoming fixture, form, H2H, xG history
  → odds_fetcher (The Odds API) → bookmaker 1X2 + totals + BTTS
  → scorer: consensus (median), best odds, outlier, signal 0-100
  → predictor.derive_picks(consensus, market_totals, signal, xg_lambda)
      - normalize_odds (margin-free implied)
      - solve_lambdas (bisection atas matrix Poisson; total goals realistis)
      - score matrix → probs 1X2 / O-U 1.5-3.5 / BTTS
      - edge = model_prob − implied (margin-free; totals/BTTS via fair_pair_implied)
      - top 3 picks ranked by edge (signal ≥ 70) / prob (low signal)
  → prediction engine (models.py run_prediction_engine)
      - EloModel.expected_lambdas → matrix → 1X2
      - PoissonModel.predict (attack/def + xG blend + rest days)
      - Ensemble (bobot elo/poisson dari config)
      - Calibrator.apply (log-odds linear, kalibrasi EPL)
      - SignalScorer.components → confidence, completeness, agreement, edge
  → grade_recommendation → VALID / KANDIDAT / HATI-HATI (4 gate)
  → prediction_log.append_snapshot (JSONL immutable) → settle → stats
```

## 2. Formula Inti

- **Elo**: `share = 1/(1+10^((rh+HA−ra)/400))`, K shrinks `k/(1+0.1√games)`,
  home advantage 65, base total goals 2.7.
- **Poisson**: `lambda_home = base_home · √(atk_h · def_a)` dengan shrinkage,
  time-decay xi=0.9 pada raw scorelines, blend xG 0.65, DC rho=-0.1.
- **solve_lambdas**: dua bisection nested — total goals T dari P(draw), share
  dari P(home)/(P(home)+P(away)). Eksak (roundtrip < 1e-4).
- **Margin removal**: 1X2 `normalize_odds`; pasangan Over/Under, BTTS via
  `fair_pair_implied` (kedua sisi). Semua edge: `edge = (model − implied)·100`.
- **Calibration**: `p' = sigmoid(a + b·logit(p))`, fit IRLS atas backtest
  EPL 2022-2026 (4.560 samples, ECE ≈ 0.010).
- **Confidence** (SignalScorer):
  `0.20·completeness + 0.30·agreement + 0.50·calibration_quality`.
- **Signal**: `0.40·completeness + 0.30·agreement + 0.30·calibration` (×100).
- **Completeness**: odds 0.30 + form 0.20 + attack/defense 0.20 + xG 0.20 + H2H 0.10.
- **Grade gate** (VALID): confidence ≥ 0.70, kalibrasi ≥ 0.50,
  completeness ≥ 0.50, edge ≥ 2.0pp, signal ≥ 70.

## 3. Fitur (pre-match only, anti-leakage)

- odds konsensus (1X2/totals/BTTS), best odds, outlier
- form home/away (sequence, gf/ga avg, raw scorelines → time decay)
- attack/defense (gf/ga), xG for/against (avg + history)
- H2H, rest days, home/away split, sources per field
- Elo rating (seeded 255 tim / 11 liga dari football-data.co.uk)

## 4. Validasi

- **validate.py**: walk-forward kronologis satu pass, Elo/form menyebrang
  musim, 5 model (baseline/elo/poisson/dc/ensemble) + market baseline,
  metrik: log-loss, Brier, ECE, hit rate + CI 95%, ROI flat-stake
  (edge ≥ 2%, odds historis nyata), consistency per season vs baseline/market.
- **backtest.py**: same, plus `--seed-elo` dan eksperimen weight/rest-days.
- Hasil validasi EPL 2022-2026 (1.520 match): Elo terkuat; ensemble ≈ Elo;
  Poisson/DC belum terbukti menambah; market tetap benchmark kuat.

## 5. Kelemahan yang Diketahui (sebelum transformasi ini)

1. Snapshot prediction log tidak menyimpan fitur detail (Elo, lambda, form,
   completeness) — tidak bisa analisis *why* pasca-settle.
2. Confidence bisa overstate saat completeness rendah (tidak ada cap level).
3. Tidak ada Max Drawdown / Sharpe / breakdown per bucket (confidence/edge).
4. Tidak ada historical performance dari *sinyal serupa* (similar-signal CLV).
5. Tidak ada laporan before/after yang dibakukan.
6. Kalibrasi EPL diterapkan ke liga lain (efek kecil, tapi bukan bukti di liga itu).

## 6. Komponen yang TIDAK BOLEH diubah (immutable baseline)

- `elo.py` — rating + resolusi nama (seeded 255 tim; jangan ubah K/HA/seed)
- `models.py` — Poisson/DC/Ensemble math & bobot default
- `predictor.py` — solve_lambdas / fair_pair_implied / derive_picks (margin-free)
- `calibration.py` — Calibrator fit & SignalScorer weights
- `scorer.py` — consensus/best/outlier/signal
- `validate.py` / `backtest.py` — walk-forward + ROI (margin-free, odds nyata)

Perubahan transformasi ini hanya MENAMBAH lapisan (features logging,
completeness caps, similar-signal stats, drawdown/sharpe, report) — tidak
mengubah satu pun formula model di atas.
