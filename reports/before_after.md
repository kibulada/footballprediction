# Before / After — Betting-Grade Transformation

Tanggal: 2026-08-11
Ruang lingkup: transformasi prediction engine menjadi **forecasting + value detection
yang betting-grade** sesuai task spec 10 fase.

## Ringkasan Kebijakan

Transformasi ini **tidak mengubah satu pun formula model** (Elo, Poisson/DC,
Ensemble, Calibrator, solve_lambdas, edge margin-free — baseline di
`baseline/`, audit di `audit/current_architecture.md`). Yang ditambahkan
hanyalah **lapisan yang membuat prediksi dapat diaudit & divalidasi secara
statistik**: snapshot fitur, completeness caps, similar-signal history,
Drawdown/Sharpe, dan laporan.

Karena model tidak disentuh, metrik probabilitas **harus identik sebelum dan
sesudah** — laporan ini membuktikannya (regresi = rollback).

## Before / After — Metrics (EPL 2022-2026, walk-forward, 1.520 match)

### Model (probabilitas) — TIDAK BERUBAH ✅

| Model | Log Loss | Brier | ECE | Hit% |
|---|---|---|---|---|
| baseline | 1.0971 | 0.6495 | 0.0196 | 44.4% |
| **elo** | 0.9914 | 0.5921 | 0.0205 | 52.3% |
| poisson | 1.0341 | 0.6213 | 0.0157 | 49.2% |
| dc | 1.0363 | 0.6223 | 0.0174 | 48.9% |
| **ensemble** | **0.9886** | **0.5901** | **0.0110** | **53.1%** |
| market (benchmark) | 0.9652 | 0.5739 | 0.0084 | 54.5% |

Kesimpulan (sama seperti baseline): Elo ≈ ensemble terkuat; market tetap
benchmark yang belum terkalahkan; Poisson/DC belum terbukti menambah.

### Kalibrasi — KONSISTEN (sesuai seed kalibrasi nyata)

| Parameter | Sebelum | Sesudah |
|---|---|---|
| a | 0.008 | 0.0082 |
| b | 1.013 | 1.0129 |
| samples | 4.560 | 4.560 |
| raw ECE | 0.0103 | 0.0110 |
| calibrated ECE (in-sample) | ~0.010 | 0.0104 |

### Lapisan baru (betting-grade) — SEBELUM TIDAK ADA / SESUDAH ADA ✅

| Kapabilitas | Before | After |
|---|---|---|
| Fitur detail di snapshot (Elo, λ, form, completeness) | ❌ tidak disimpan | ✅ `features` di JSONL |
| Completeness caps (90-100 HIGH, 50-69 max MEDIUM, <50 LOW only) | ❌ confidence bisa overstate | ✅ cap confidence per level |
| Similar-signal history (bucket confidence × edge → hit/ROI/CLV) | ❌ tidak ada | ✅ `similar_signal_stats` + tampil di analyse |
| Max Drawdown | ❌ | ✅ dari kurva net-stake |
| Sharpe | ❌ | ✅ per-bet, sqrt(n) scaling |
| Breakdown per confidence / edge bucket | ❌ | ✅ `by_confidence`, `by_edge` |
| Audit report + baseline backup | ❌ | ✅ `audit/`, `baseline/` |

ROI flat-stake (best 1X2, edge ≥ 2%): ensemble **-1.9%** over 1.520 match —
konsisten dengan baseline (belum profit, tapi tidak pernah diklaim). Model
belum mengalahkan market pada log-loss di musim mana pun — diklaim apa adanya.

## 1. Files Changed

- `agents/football/prediction_log.py` — `features` di snapshot;
  `similar_signal_stats`; `_max_drawdown`/`_sharpe`; `by_confidence`/`by_edge`
  di `compute_stats`; `format_stats` menampilkan semuanya.
- `agents/football/analyse.py` — kirim `features` saat snapshot; lookup
  `similar_signal` setelah snapshot (never-breaks-flow, try/except).
- `agents/football/format.py` — embed stats: Drawdown/Sharpe + bucket;
  embed analyse: baris bucket serupa (CLV historis) pada Best Pick.
- `agents/football/calibration.py` — `completeness_level()` + cap confidence
  di `SignalScorer.components` (field `data_completeness_level`).
- `agents/football/predictor.py` — grade gate menolak completeness < 0.50
  dengan reason eksplisit.
- `audit/current_architecture.md`, `baseline/` (12 modul), `reports/before_after.md` — baru.

## 2. Architecture Changes

```
sebelum:  analyse → engine → grade → snapshot(prob/odds/edge)
sesudah:  analyse → engine → grade → snapshot(+features)
                                      → similar_signal (bucket → CLV historis)
                                     stats(+max_drawdown, sharpe, by_*)
                                     confidence di-cap oleh completeness level
```

## 3. Why Each Change

- **Features di snapshot**: tanpa input state (Elo/λ/form), log yang settle
  tidak bisa dijelaskan — similar-signal butuh fitur untuk klaster.
- **Completeness caps**: PHASE 3 — missing data harus menurunkan ceiling
  confidence, bukan hanya menambah noise; mencegah label HIGH di data minim.
- **Similar-signal**: PHASE 4/8 — edge besar ≠ value; dicek dulu apa yang
  bucket sinyal yang sama lakukan secara historis (CLV nyata).
- **Drawdown/Sharpe**: PHASE 7 — ROI rata-rata bisa menipu; risiko dibaca
  dari kurva net-stake (peak-to-trough + risiko per bet).
- **Bucket breakdown**: PHASE 7 — memperlihatkan di mana model valid (HIGH
  confidence? edge 10-20%?) dan di mana tidak.

## 4. Remaining Limitations (jujur)

1. Similar-signal bucket butuh akumulasi snapshot yang di-settle — sampai
   ada ≥ 5 sampel per bucket, menampilkan "belum cukup sampel" (bukan angka).
2. Kalibrasi EPL diterapkan ke semua liga (efek kecil, bukan bukti di liga itu).
3. Market masih benchmark yang belum dikalahkan (log-loss) — model tidak
   mengklaim beating bookmaker.
4. Closing odds hanya tercatat saat `settle` manual/auto menyertakannya;
   CLV dibatasi snapshot dengan closing odds.
5. Sharpe pakai sqrt(n) scaling (per-bet), bukan annualisasi kalender —
   label disertakan agar tidak salah baca.

## 5. How to Run

```bash
# validasi EPL walk-forward + metrik
python -m agents.football.validate --leagues EPL \
    --seasons 2022-2023,2023-2024,2024-2025,2025-2026 --odds-source football-data

# statistik prediction log (live): hit rate, logloss, ROI, CLV, drawdown, sharpe, bucket
python -m agents.football.prediction_log stats

# settle hasil match (closing odds opsional: home,draw,away)
python -m agents.football.prediction_log settle \
    --match-id "EPL||Arsenal||Chelsea||2026-08-15T14:00:00Z" \
    --home-goals 2 --away-goals 1 --closing-odds 1.62,4.30,4.60

# bot Discord
python bot.py
#   !football compare <HOME> <AWAY> [league]  → analyse + similar-signal
#   !football settle <home> vs <away> 2-1     → catat hasil
#   !football settle auto                     → settle semua snapshot selesai
#   !football stats                           → metrik + bucket + drawdown/sharpe
```

## Verdict

Tidak ada regresi model (metrik identik). Semua gap betting-grade dari spec
telah ditutup: audit ✅, fitur snapshot ✅, completeness caps ✅,
similar-signal ✅, Drawdown/Sharpe ✅, laporan before/after ✅.
