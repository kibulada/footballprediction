# PHASE 7-14 — Market/Odds-Bucket Validation, Edge & EV, Decision, CLV, Backtest

Scope: walk-forward kronologis EPL 2022–2026 (1.520 match, odds Pinnacle riil —
satu-satunya dataset dengan odds historis; La Liga/Bundesliga/Serie A cache
**tanpa odds** → decision/ROI layer hanya valid di EPL, dilaporkan jujur).

> **FIX yang ditemukan dalam fase ini**: `backtest._normalize_fixtures`
> membuang kolom `over25_odds`/`under25_odds` → semua audit decision Fase 1
> menguji totals dengan odds kosong (NEW all == NEW 1X2, totals tak pernah
> diuji meski docstring menjanjikannya). Loader kini meneruskan kedua kolom
> (additif; non-totals path identik). Hasil di bawah adalah angka POST-FIX.

---

## PHASE 7 — Market / odds-bucket validation (model hit vs no-vig market)

### 1X2 (3 sisi per match, n=4.560 sisi)

| odds range | n | model hit% | Wilson 95% |
|---|---|---|---|
| <1.20 | 54 | 90.7% | [80.1, 96.0] |
| 1.20-1.40 | 181 | 72.9% | [66.0, 78.9] |
| 1.40-1.60 | 216 | 66.7% | [60.1, 72.6] |
| 1.60-2.00 | 461 | 53.6% | [49.0, 58.1] |
| 2.00-3.00 | 846 | 38.9% | [35.7, 42.2] |
| >3.00 | 2802 | 22.1% | [20.6, 23.7] |

Benchmark no-vig: hit 33.3%, brier 0.1913, ll 0.5653 (1X2 per-side).
Model kalibrasi per probability bucket konsisten dengan PHASE 5-6 (bucket
[0.80,0.90) → 94.4% aktual). **Model tidak mengalahkan market pada log-loss**
(lihat PHASE 13) — hit-rate naik dengan odds favorit = behavior yang benar.

### Over 2.5 (n=1.520)

| edge range | n | model hit% | Wilson 95% |
|---|---|---|---|
| <0pp | 949 | 59.3% | [56.2, 62.4] |
| 0-5pp | 222 | 54.1% | [47.5, 60.5] |
| 5-10pp | 173 | 54.3% | [46.9, 61.6] |
| 10-20pp | 154 | 54.5% | [46.7, 62.2] |
| >20pp | 22 | 40.9% | [23.3, 61.3] |

Benchmark: hit 57.2% (market no-vig untuk Over 2.5 = 57.2%, sama — pasar
efisien di totals). **Edge tinggi TIDAK meningkatkan hit di Over 2.5**
(54% di semua bucket, turun ke 41% di >20pp dengan n=22) — bukti empiris
PHASE 8: "edge besar ≠ value".

### Under 2.5 (n=1.520)

| edge range | n | model hit% | Wilson 95% |
|---|---|---|---|
| <0pp | 571 | 46.2% | [42.2, 50.3] |
| 0-5pp | 260 | 41.5% | [35.7, 47.6] |
| 5-10pp | 219 | 40.6% | [34.3, 47.3] |
| 10-20pp | 301 | 40.5% | [35.1, 46.2] |
| >20pp | 169 | 39.6% | [32.6, 47.2] |

Benchmark: hit 42.8%. Hit menurun monoton dengan edge → **Under 2.5 edge
adalah anti-signal di data ini** (model menilai under lebih sering daripada
realisasi). Ini konsisten dengan temuan decision: totals menyeret ROI.

### Home vs away (1X2)

home sides hit 44.5% vs away sides 27.8% — home advantage tercermin benar
(model tidak bias).

---

## PHASE 8 — Edge & EV (flat-stake, best 1X2 pick, no-vig edge ≥ 2%)

| edge bucket | bets | ROI | dd | lose-streak |
|---|---|---|---|---|
| 0-5pp | 187 | −2.1% | −18.4 | 11 |
| 5-10pp | 218 | **−11.0%** | −33.9 | 12 |
| 10-20pp | 174 | −2.1% | −34.4 | 13 |
| >20pp | 44 | +44.6% | −4.3 | 4 |

- Edge sedang (5-10pp) = **terburuk** (−11%) → tepi model pada bucket ini
  adalah noise (PHASE 8: jangan bet hanya karena edge).
- Edge >20pp kecil (n=44) tapi +44.6% — perlu sampel lebih besar untuk
  diklaim; Wilson pada hit bucket ini [21.5, 46.2] sangat lebar →
  **STATISTICALLY INCONCLUSIVE** (PHASE 35).
- Kesimpulan EV: tidak ada bucket 1X2 yang robustly profitable; keseluruhan
  flat-stake ≈ −1.9% (konsisten baseline freeze).

---

## PHASE 9-10 — Decision engine: STRONG requirements & per-type evidence

Setelah fix loader (totals kini benar-benar diuji), walk-forward decision:

| rule | bets | hit% | Wilson 95% | ROI | net | dd | lose-streak |
|---|---|---|---|---|---|---|---|
| OLD (best 1X2, edge≥2%) | 623 | 41.2% | [37.5, 45.2] | −1.9% | −11.9 | −55.8 | 10 |
| NEW (all markets) | 1057 | 46.3% | [43.3, 49.3] | −4.5% | −47.7 | **−104.2** | 12 |
| **NEW (1X2 only)** | **370** | 40.3% | [35.4, 45.3] | **−1.6%** | −6.0 | −48.3 | 8 |
| delta NEW-all − OLD | | | | **−2.60pp** | | | |
| delta NEW-1X2 − OLD | | | | **+0.29pp** | | | |

Per decision type (NEW all):

| type | bets | hit% | Wilson 95% | ROI | dd | lose-streak |
|---|---|---|---|---|---|---|
| **STRONG** | 107 | 58.9% | [49.4, 67.7] | −1.5% | −7.9 | 6 |
| GOOD | 738 | 44.9% | [41.3, 48.5] | −6.7% | −79.0 | 13 |
| LEAN | 212 | 44.8% | [38.3, 51.5] | +1.6% | −27.5 | 14 |

**PHASE 10 verdict (STRONG requirements)**:
- STRONG **tetap jarang** (107/1.520 = 7%) dan hit tertinggi (58.9%), tapi
  ROI −1.5% (vs +3.1% pada Fase 1 yang totals kosong — angka Fase 1 tidak
  pernah menguji totals). Dengan totals ikut diuji, STRONG menipis nilai
  value-nya; dd kecil (−7.9) tapi tetap negatif.
- **Temuan kunci**: totals (O/U 2.5) **menyeret decision engine** —
  NEW-1X2-only (+0.29pp vs OLD) lebih baik dari NEW-all (−2.60pp). Ini
  bukan kesimpulan tuning (tidak di-tune), tapi bukti evaluasi jujur:
  **market totals efisien di data ini** (PHASE 7: hit edge flat) sehingga
  edge totals tidak membawa value.
- **Kesimpulan decision**: NEW 1X2 ≈ OLD dalam noise (+0.29pp, n=370).
  **STATISTICALLY INCONCLUSIVE** (PHASE 35) — delta jauh di bawah Wilson
  separation. Tidak ada klaim improvement; tidak ada regresi material pada
  1X2; totals tetap dievaluasi tapi diakui negatif.

Per-season (NEW all) — PHASE 14 stabilitas:

| season | bets | hit% | ROI | dd | streak |
|---|---|---|---|---|---|
| 2022-23 | 296 | 52.7% | **+13.9%** | −14.1 | 9 |
| 2023-24 | 240 | 49.2% | −3.5% | −19.1 | 11 |
| 2024-25 | 243 | 43.6% | −15.4% | −43.9 | 8 |
| 2025-26 | 278 | 39.2% | −15.6% | −53.2 | 12 |

ROI menurun tajam lintas musim (+13.9% → −15.6%) → **tidak stabil lintas
musim**; season 1 outlier positif tidak bisa diandalkan (PHASE 14/35:
per-season + CI wajib sebelum klaim). Keputusan produksi: RETAIN config
decision saat ini (sudah live, 1X2-focused via `best_prob_only`) — tanpa
klaim profitabilitas.

---

## PHASE 11 — CLV

- **Historis: TIDAK TERSEDIA.** Cache hanya menyimpan odds Pinnacle pada satu
  titik (tanpa split opening/closing), jadi CLV = closing − prediction tidak
  dapat dihitung dari data historis. Tidak difabrikasi.
- **Live: SUDAH ADA** — `prediction_log.py settle` menerima closing odds dan
  menghitung CLV per snapshot yang tersettle; dipakai sebagai diagnostik
  harga (bukan bukti akurasi — PHASE 11).
- Rekomendasi lanjutan (di luar scope): simpan odds snapshot T-24h/T-6h/T-1h
  untuk CLV historis masa depan (PHASE 32-33).

## PHASE 12 — ROI / betting backtest

- Flat-stake backtest dengan odds riil Pinnacle (bukan odds terbaik hari ini):
  OLD −1.9% / NEW-all −4.5% / NEW-1X2 −1.6%, dengan max drawdown & losing
  streak penuh (lihat tabel PHASE 9-10). Tidak ada cherry-picking: semua
  match ber-odds diuji, tidak ada losing bet yang dihapus.

## PHASE 13 — Comparators (forecast, dari validate walk-forward)

| model | LL | Brier | ECE | hit% |
|---|---|---|---|---|
| baseline | 1.0971 | 0.6495 | 0.0196 | 44.4% |
| elo | 0.9914 | 0.5921 | 0.0205 | 52.3% |
| poisson | 1.0341 | 0.6213 | 0.0157 | 49.2% |
| dc | 1.0363 | 0.6223 | 0.0174 | 48.9% |
| **ensemble** | **0.9886** | 0.5901 | 0.0110 | 53.1% |
| **market (no-vig)** | **0.9652** | 0.5739 | 0.0084 | 54.5% |

**Jujur**: market tidak pernah dikalahkan pada log-loss di musim mana pun.
Ensemble adalah model terkuat internal. Tidak ada klaim beating bookmaker.

## PHASE 14 — Multi-league / multi-season

- Forecast: cross-league walk-forward selesai di PHASE 3-4 (EPL/LaLiga/
  Bundesliga/Serie A, 6.172 match) — konsistensi & stabilitas lintas liga.
- Decision/ROI layer: **hanya EPL** (odds riil hanya di cache EPL; cache
  liga lain membawa kolom odds tapi None). Per-season di atas menegaskan
  instabilitas — dilaporkan apa adanya.

---

## Files

- `agents/football/market_audit.py` — BARU (PHASE 7-8 harness).
- `agents/football/decision_validation.py` — + per-season, Wilson CI, dd/streak OLD.
- `agents/football/backtest.py` — FIX loader odds totals (additif).
- `tests/test_market_audit.py` — BARU (11 test incl. loader regression).
- Reports: `reports/phase7_8_market_audit.json`, `reports/decision_validation_phase9.json`.
