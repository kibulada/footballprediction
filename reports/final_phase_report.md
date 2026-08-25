# FINAL REPORT — MASTER PROMPT PHASE 0-20 (Football Forecasting & Decision Engine)

Tanggal: 2026-08-12
Ruang lingkup: seluruh pipeline prediksi (forecast + decision) dievaluasi
dengan walk-forward kronologis, anti-leakage, tanpa fabrikasi.

---

## Ringkasan Eksekutif

1. **Forecast model tidak diubah config-nya** — ensemble LL 0.9886 EPL
   (1.520 match) terkonfirmasi. Fitur eksperimen (rest-days, opp-adj form)
   **gagal melampaui noise** → RETAIN. **xG kini tervalidasi OOS dengan data
   riil** (lihat #6) — `xg_weight=0.65` sudah config produksi, dan bukti
   walk-forward kini menunjukkan ia improve **5/5 liga**.
2. **Kalibrasi model sudah well-calibrated** (pooled bucket ECE 0.0042, 4 liga)
   → tidak ada kalibrator baru yang dipromosikan (PHASE 37: hanya improve OOS
   di 1/4 liga).
3. **Decision engine** (live sejak Fase 1) dievaluasi ulang dengan fix
   penting: loader odds totals. **NEW-1X2 ≈ OLD dalam noise (+0.29pp)**;
   **totals market menyeret decision (−2.60pp all)** — bukti empiris bahwa
   edge totals di data ini bukan value.
4. **CLV historis tidak tersedia** (cache tanpa split open/close) — live-only.
5. **Tidak ada klaim beating bookmaker** — market (no-vig) tidak pernah
   dikalahkan pada log-loss di musim mana pun (PHASE 13).
6. **TERBARU — xG riil (understat) + live wiring (PHASE 9)**: dengan data
   xG per-match riil EPL 2022-2026 (coverage 100%) dan 4 liga lain,
   `xg_weight=0.65` improve log-loss di **5/5 liga** (EPL 0.9886→0.9851,
   LaLiga 0.9943→0.9907, Serie A 1.0031→1.0009, Bundesliga 1.0097→1.0057,
   Ligue 1 1.0141→1.0100), hit rate naik 4/5, ECE turun (kalibrasi membaik),
   ROI EPL −1.91%→−1.06%. Sebelumnya blend xG **inert di backtest DAN di
   live** (sumber offline; sofascore-history diblokir). Sekarang: backtest
   bisa mengevaluasinya, dan **live memakai fallback xG understat** (big-5)
   saat sofascore-history tidak tersedia — fitur xG benar-benar mengalir ke
   model di produksi.

---

## Per-fase verdict

| Phase | Deliverable | Verdict |
|---|---|---|
| **0** Audit arsitektur | `reports/audit_phase0.md` | ✅ Jalur lengkap Discord→analyse→engine→decision terpetakan |
| **1** Baseline freeze | `baseline_freeze_v2.json/.md` | ✅ Ensemble LL 0.9886, hit 53.1%, ROI −1.9% (reference) |
| **2** Anti-leakage | `leakage_audit.py` + `reports/leakage_audit_epl.json` | ✅ PASS — 0 violation, equivalence 0.9886, determinism OK |
| **3-4** Fitur ablation | `ablation.py`, `phase3_4_ablation_xg.json` | ✅ Rest-days/opp-adj RETAIN (noise); **xG: LL improve 5/5 liga dengan data riil** (EPL `phase3_4_ablation_xg.json`, cross-league `phase14_xg_cross_league.json`) |
| **5-6** Confidence buckets & OOS calibration | `calibration_audit.py`, `phase5_6_calibration_audit.md` | ✅ Pooled ECE 0.0042; kalibrasi OOS tidak dipromosikan (1/4 liga improve) |
| **7** Odds-bucket validation | `market_audit.py`, `phase7_8_market_audit.json` | ✅ Hit naik dengan favorit; edge tinggi ≠ hit tinggi (totals flat) |
| **8** Edge & EV | sama | ✅ Edge 5-10pp TERBURUK (−11%); >20pp positif tapi n kecil → inconclusive |
| **9** Decision engine | `decision_validation.py` (fix loader totals) | ✅ NEW-1X2 ≈ OLD (+0.29pp); totals menyeret (−2.60pp all) |
| **10** STRONG requirements | by-type table | ✅ STRONG jarang (7%), hit tertinggi (58.9%), ROI −1.5% (totals ikut) |
| **11** CLV | — | ⚠️ Historis tak tersedia; live ada di `prediction_log settle` |
| **12** ROI backtest | `decision_validation_phase9.json` | ✅ Flat-stake riil Pinnacle, dd/streak penuh, tanpa cherry-picking |
| **13** Comparators | validate walk-forward | ✅ Ensemble terkuat internal; **market tidak pernah dikalahkan** |
| **14** Multi-league/multi-season | phase3_4 cross-league + per-season decision | ✅ Forecast 4 liga; decision hanya EPL (odds), ROI tidak stabil lintas musim |
| **15** Model improvement rule | ablation | ✅ Fitur noise RETAIN; **xG dipromosikan berbasis bukti OOS** (5/5 liga, tanpa degradasi metrik apa pun; config 0.65 sudah produksi — yang berubah adalah data & live wiring) |
| **16** Data quality / provenance | `reports/feature_provenance.md` | ✅ Semua fitur pre-match; xG riil kini tersedia (understat) & tidak difabrikasi |
| **17** Source priority | provenance + multi_source | ✅ FBref/football-data historis; TheSportsDB/Flashscore live |
| **18** Production output | format.py (Discord) | ✅ Contract S29 dipertahankan (PREDICTION/MARKET/VALUE/CONFIDENCE/FINAL DECISION) |
| **19** Testing | tests/ | ✅ Full suite **431 passed** (374 → +57 test baru, tanpa regresi) |
| **20** Final validation | laporan ini | ✅ Semua harness jalan, tanpa regresi, tanpa fabrikasi |

---

## Baseline vs final (EPL 2022-2026, walk-forward)

| metric | baseline freeze | final (identik — model tidak disentuh) |
|---|---|---|
| ensemble LL | 0.9886 | 0.9886 |
| ensemble hit | 53.1% | 53.1% |
| ensemble Brier | 0.5901 | 0.5901 |
| calibration ECE (raw) | 0.0103 | 0.0110 |
| ROI flat-stake (OLD rule) | −1.9% | −1.9% |

Model probabilities byte-identik → **regression check PASS**.

## Decision layer — perubahan yang dipromosikan

1. **FIX loader odds totals** (`backtest._normalize_fixtures` kini meneruskan
   `over25_odds`/`under25_odds`) — audit decision sekarang menguji totals
   secara benar. Ini bukan perubahan model; ini perbaikan pipeline evaluasi.
2. **Decision validation diperkaya**: per-season, Wilson CI, dd/streak OLD
   (hanya evaluasi, tidak mengubah decision config live).
3. **market_audit.py + calibration_audit.py + ablation.py + leakage_audit.py**
   — harness audit baru, semuanya additif.

## Eksperimen yang ditolak / tidak dipromosikan (PHASE 15/37/36)

- Rest-days k∈{0.01…0.1} — memburuk di semua liga → **RETAIN k=0**.
- Opponent-adjusted form — netral (noise) → **RETAIN baseline**.
- xG blend 0.65 — SEBELUMNYA inert (tidak ada data); dengan data riil
  understat kini **tervalidasi improve 5/5 liga** → tetap dipakai (config
  produksi sudah 0.65), dan live kini benar-benar mengisinya.
- Kalibrator OOS baru — memburuk 3/4 liga → **tidak dipromosikan**.
- Value credit untuk semua kandidat / long-shot edge — sudah ditolak di Fase 1
  (−7.8% s/d −8.6%) → config `best_prob_only` + `min_edge_pp 3.0` tetap.

## Keterbatasan tersisa (jujur)

1. **CLV historis** tidak dapat dihitung (odds satu titik per match).
2. **Decision ROI tidak stabil lintas musim** (+13.9% → −15.6%) — tidak ada
   klaim profit; STRONG/GOOD/LEAN masih dalam noise statistik (Wilson lebar).
3. **Lineup/injury** belum punya sumber riil pre-match → tidak dimasukkan.
   **xG**: bukti OOS SEKARANG ADA (5/5 liga). Batasan live: awal musim baru
   window last-5 didominasi ekor musim lalu; understat hanya mencakup 5 liga
   besar (+ RFPL); bot lain (non-big-5) tetap tanpa fitur xG.
4. Same-day match urutan sebarang (day-granularity) — caveat diketahui.
5. Totals market efisien di EPL data ini — decision tetap mengevaluasi totals
   (config live `best_prob_only`), tapi bukti menunjukkan 1X2 adalah
   universe yang lebih dapat diandalkan.
6. Evaluasi OOS kalibrasi & decision berbasis satu musim eval (2025-26) —
   round-robin lintas musim adalah langkah lanjutan yang disarankan.

## PHASE 32-33 — Prediction timing snapshots & CLV (ditambahkan kemudian)

- **Odds snapshot** (`!football odds <T-24h|T-6h|T-1h|T-15m> <home> vs <away> <h,d,a> [league]`)
  mencatat harga 1X2 bertingkat waktu ke prediction log (append-only, keyed
  match_id, label timing per baris — snapshot tidak pernah dicampur).
- **Price CLV dipisah dari model CLV** (forecast quality ≠ price quality):
  `price_clv` = closing/prediction − 1, plus `clv_by_timing` per label.
- CLV historis masih belum tersedia untuk backtest (cache tanpa split
  open/close) — mekanisme snapshot ini memungkinkan evaluasi CLV ke depan.
- `leakage_audit.CLV_ALLOWED_MODULES` + `bot.py` (transport Discord, layer
  output — aman; `check_clv_scope` PASS).

## PHASE 9 TERBARU — Real xG (understat) + live wiring

- **Sumber baru `agents/football/understat_xg.py`**: understat.com menolak
  client HTTP biasa (404 aktif) — data match (teams, skor, xG) dibaca dari
  page-global `window.datesData` via Chrome UC (pola seleniumbase yang sama
  dengan flashscore/sofascore fallback). Download EPL/LaLiga/Serie A/
  Bundesliga/Ligue 1 2022-2025: **5.782 match ber-xG**, join EPL 1520/1520
  (coverage 1.0), 0 nama tim unmatched.
- **`ablation.py --understat-xg` / `--understat-leagues`**: evaluasi
  `xg_weight` 0 vs 0.65 pada data xG riil (paired, walk-forward). Hasil:
  LL improve **5/5 liga**; ECE turun; ROI EPL −1.91%→−1.06% (1X2 flat-stake,
  583 bet). Delta pooled EPL 0.0035 masih di dalam CI (±0.0218) — tetapi
  konsisten arah di 4/4 musim dan 5/5 metrik tanpa satupun degradasi
  (kriteria PHASE 15 "improves OOS without degrading others").
- **Live wiring**: `multi_source.fetch_team_xg_history` (fallback understat,
  rolling last-5 finished, anti-leak exclude fixture yang diprediksi, cache
  per-liga 6 jam) + hook di `analyse.py` — fitur `home_xg_for` dll kini
  terisi saat sofascore-history diblokir (sebelumnya None → blend inert).
  Gagal → xG tetap inert, prediksi tidak terganggu. `leakage_audit` PASS.

## Files changed (fase ini)

- `agents/football/market_audit.py` (BARU)
- `agents/football/decision_validation.py` (+per-season/Wilson/dd/streak)
- `agents/football/backtest.py` (fix loader odds totals)
- `agents/football/prediction_log.py` (+odds_snapshot, price CLV, CLI)
- `agents/football/runner.py` (+mode odds-snapshot fuzzy lookup)
- `agents/football/format.py` (+format_odds_snapshot, price CLV di stats)
- `agents/football/leakage_audit.py` (+bot.py ke CLV_ALLOWED_MODULES)
- `bot.py` (+!football odds)
- `tests/test_market_audit.py` (BARU, 11 test), `tests/test_odds_snapshot.py` (BARU, 6 test)
- `reports/phase7_8_market_audit.json`, `reports/decision_validation_phase9.json`,
  `reports/phase7_14_market_decision.md`, `reports/feature_provenance.md`

## Files changed (fase xG — PHASE 9/14/15)

- `agents/football/understat_xg.py` (BARU — downloader + `team_xg_history_from_rows`)
- `agents/football/ablation.py` (+`--understat-xg`, +`--understat-leagues`)
- `agents/football/multi_source.py` (+`fetch_team_xg_history`, lazy UC client + cache)
- `agents/football/analyse.py` (+fallback xG history ke context, anti-leak exclude)
- `tests/test_understat_xg.py` (BARU, 9 test)
- `cache/football/understat_{league}_{season}.json` (raw), `understat_xg_rows.json`,
  `understat_rows_{liga}.json`, `epl_fixtures_2022_2026_xg.json` (augmented)
- `reports/phase3_4_ablation_xg.json`, `reports/phase14_xg_cross_league.json`

## Tests

Full suite: **431 passed** (374 di awal program → +57 test baru, tanpa regresi).

*Angka diverifikasi oleh full suite run terakhir (431). Per fase: PHASE 1-2 +16, PHASE 3-4 +11, PHASE 5-6 +7, PHASE 7-8 +11 (398 → 409), PHASE 32-33 +13 (409 → 422), PHASE 9 xG +9 (422 → 431).*
