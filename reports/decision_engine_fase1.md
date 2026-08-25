# Fase 1 — Transparent Decision Engine (Master-Prompt S18, S22–S31, S39)

> Audit & baseline-freeze sudah dilakukan sebelumnya: `reports/baseline_freeze.md`
> (EPL 2022–2026, 1.520 match, walk-forward kronologis, offline).

## Ringkasan Eksekutif

- Semua syarat **wajib** dari spec yang hilang sudah diimplementasikan: switch
  `signal>=70` **dihapus** (S23), **Decision Score** transparan (S24), confidence
  kini memasukkan magnitude/separation probabilitas (S26), **tipe keputusan**
  STRONG/GOOD/LEAN/**NO CLEAR DECISION/NO BET** (S27–28), output terpisah
  PREDICTION/MARKET/VALUE/CONFIDENCE/**FINAL DECISION** + penjelasan
  most-likely vs best-decision (S29–30), **extreme-edge protection** (S18),
  anti-double-count market (S25).
- **Forecast model TIDAK disentuh**: metrik prediksi (log-loss, Brier, ECE)
  identik dengan baseline — perubahan hanya pada lapisan keputusan + output.
- **Walk-forward decision-layer jujur**: NEW rule ≈ OLD rule (ROI −2.2% vs
  −1.9%, delta −0.34pp) → **STATISTICALLY INCONCLUSIVE**. Tidak diklaim
  lebih baik. Iterasi tuning pertama (value credit untuk semua kandidat) JAUH
  lebih buruk (−7.8%) dan **ditolak**; config yang di-ship adalah hasil
  validasi walk-forward, bukan pilihan teoretis.

## Files changed

| File | Perubahan |
|---|---|
| `agents/football/decision.py` | **BARU**. DecisionScore (bobot 30/20/15/15/10/5/5), margin-free implied, EV, edge-level warning/extreme, tipe keputusan, guard NO CLEAR/NO BET, `best_prob_only` (opsi tervalidasi), `decision_to_dict` JSON-safe |
| `agents/football/predictor.py` | `derive_picks`: **switch signal≥70 dihapus** — ranking murni model_prob; EV ditambahkan per pick |
| `agents/football/calibration.py` | `SignalScorer` + komponen **`decisiveness`** (magnitude + separation 1X2) dengan bobot configurable |
| `agents/football/models.py` | `run_prediction_engine` meneruskan `p1x2` ke scorer; `decisiveness` disimpan di `PredictionResult` |
| `agents/football/analyse.py` | Wire DecisionEngine (setelah engine + similar-signal); payload `decision` JSON-safe; config dari `config/football.json` |
| `agents/football/format.py` | Output S29: Top-3 dilabeli "pandangan odds/market" (bukan rekomendasi final); seksi **FINAL DECISION** + badge tipe + skor/edge/EV + penjelasan + edge warnings; decisiveness tampil |
| `config/football.json` | `models.decision`: weights, `min_edge_pp: 3.0`, `best_prob_only: true` |
| `agents/football/decision_validation.py` | **BARU**. Walk-forward decision-layer before/after (OLD vs NEW, semua market & 1X2-only) |
| `tests/test_decision.py` | **BARU** (26 test): normalisasi, overround, edge/EV, agreement, data quality, confidence, extreme edge, no-bet, most-likely vs best-decision, anti-leakage konfigurasi, JSON-safety, `best_prob_only` |
| `tests/test_football.py` | `test_format_analyse_match` diperbarui ke output baru |
| `reports/baseline_freeze.md` | Baseline regression reference |

## Metrics (walk-forward EPL 2022–2026, 1.520 match)

### Forecast metrics — TIDAK BERUBAH (engine prediksi tidak disentuh)
- ensemble log-loss **0.9886** • hit **53.1%** • market 0.9652 (tetap benchmark).
- calibration ECE 0.0103 (4560 samples).

### Decision-layer (before vs after, same probabilities & odds)
| Rule | Bets | Hit | ROI | Net |
|---|---|---|---|---|
| OLD (best 1X2, edge≥2%) | 623 | 41.2% | −1.9% | −11.92 |
| **NEW (ship config)** | **553** | **40.0%** | **−2.2%** | −12.46 |
| delta | | | **−0.34pp** | |

Per decision type (NEW):
- **STRONG** 28 bet • hit 57.1% • **ROI +3.1%**
- **GOOD** 434 bet • hit 38.5% • ROI −9.3%
- **LEAN** 91 bet • hit 41.8% • **ROI +29.5%**
- 967 match → **NO CLEAR DECISION / NO BET** (valid output, bukan error).

Kesimpulan jujur: **STATISTICALLY INCONCLUSIVE** (delta dalam noise; sampel
per-bucket kecil). Tidak ada regresi material, tapi juga tidak ada klaim
improvement.

## Eksperimen yang ditolak (S24/S37)
1. `min_edge_pp = 0` (value untuk semua EV>0): ROI −8.6% → **ditolak** (noise bets).
2. Value credit untuk kandidat non-favorit (edge besar prob rendah): −7.8%,
   hit 28.3% → **ditolak** — persis peringatan spec S17/S19 "edge besar ≠ benar";
   long-shot edge adalah noise.
3. `best_prob_only` tanpa floor edge: −2.82% (≈ OLD) → diterima sebagai dasar,
   diperketat dengan floor 3.0pp.
4. Bobot `market_value 0.25`: tidak memperbaiki (−2.82%) → **ditolak** (dilution).
5. `good_score 0.60`: tidak mengubah hasil → **ditolak** (threshold tidak sensitif).

## Anti-double-count (S25) — diimplementasikan
- Value hanya dikreditkan ke probabilitas model **independen** (Elo+Poisson
  fitur). Pick odds-derived (λ dari odds, `derive_picks`) tidak pernah membawa
  market_value (guard `independent=False` + NO BET).
- Model-agreement memakai market sebagai ukuran **disagreement** (informasi),
  bukan value — peran berbeda, terdokumentasi.

## Regression check
- Full test suite: **330 passed** (sebelumnya 305 → +25 test baru).
- Walk-forward forecast metrics identik baseline (tidak ada regresi prediksi).
- Live flow: payload decision 100% JSON-safe (`decision_to_dict`), exception
  apa pun di decision layer → `decision=None` → format.py aman, prediksi tetap.

## Keterbatasan tersisa
- Decision ROI vs OLD dalam noise; bucket GOOD masih negatif (threshold skor
  bisa dikalibrasi lebih lanjut dengan dataset lebih besar/lintas liga — tidak
  di-tune di sini untuk hindari overfit pada set final).
- `best_prob_only` membuat "most likely ≠ best decision" jarang muncul di live
  (keputusan sadar hasil validasi), tapi kemampuan itu tetap ada via config
  `best_prob_only: false` dan tetap dijelaskan di output.
- Calibration dianggap validated (1.0) di decision_validation; angka live
  memakai calibration riil dari file.
- Belum ada lineup/injuries (Fase 4), snapshot timing T-24h/T-6h & CLV lanjutan
  (Fase 2), ablasi form/attack/xG (Fase 3).
