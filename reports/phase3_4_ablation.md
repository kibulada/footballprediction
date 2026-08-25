# PHASE 3-4 — Ablasi Fitur Walk-Forward (Rest Days, Opponent-Adjusted Form, xG)

**Verdict: NO PRODUCTION IMPROVEMENT — EXISTING MODEL RETAINED** (PHASE 15/37)

Semua kandidat fitur diuji *before/after* terhadap baseline beku
(`baseline_freeze_v2.json`, ensemble LL **0.9886**, hit 53.1%, ROI −1.9%,
623 bets). Tidak ada satu pun yang melampaui noise → tidak ada yang masuk
produksi. Ini adalah hasil yang sah menurut MASTER PROMPT: sistem boleh
(malah wajib) menjawab "tidak ada perbaikan yang cukup bukti".

---

## Metode

- **Dataset identik dengan freeze**: `cache/football/epl_fixtures_2022_2026.json`
  (EPL 2022–2026, 1.520 match, odds football-data.co.uk pinnacle/avg).
- **Replay kronologis ketat** (walk-forward satu-pass, `validate.py`), state
  (Elo/form/base-rate) hanya di-update SETELAH prediksi.
- **Harness baru** `agents/football/ablation.py` (PHASE 36: BASELINE vs
  BASELINE+FEATURE) → `reports/phase3_4_ablation.json`.
- **Reproducibility check**: run baseline pada harness menghasilkan LL 0.9886,
  hit 0.5309, ROI −0.0191, bets 623 — **identik persis dengan freeze** → alat
  ukur valid, perbandingan apples-to-apples.

## A. Rest Days (`rest_days_k` di model Poisson)

| k      | LL (aggr) | Δ vs baseline | CI ±   | hit%  | ROI    | bets | per-season LL (22-23 / 23-24 / 24-25 / 25-26) |
|--------|-----------|---------------|--------|-------|--------|------|-----------------------------------------------|
| **0.0**| **0.9886** | —             | 0.0226 | 53.1% | −1.9%  | 623  | 0.9888 / 0.9503 / 0.9933 / 1.0220 |
| 0.01   | 0.9888     | +0.0002       | 0.0226 | 53.0% | −2.3%  | 630  | 0.9890 / 0.9508 / 0.9934 / 1.0221 |
| 0.03   | 0.9891     | +0.0005       | 0.0225 | 53.0% | −1.9%  | 629  | 0.9891 / 0.9513 / 0.9936 / 1.0224 |
| 0.05   | 0.9893     | +0.0007       | 0.0225 | 53.0% | −2.0%  | 627  | 0.9891 / 0.9515 / 0.9938 / 1.0227 |
| 0.1    | 0.9895     | +0.0009       | 0.0225 | 53.0% | −1.9%  | 632  | 0.9890 / 0.9515 / 0.9942 / 1.0232 |

**Hasil**: setiap nilai k **memperburuk** log-loss, dan memburuk **di keempat
musim** (tidak ada musim yang membaik → bukan offset). Delta maksimum
(+0.0009) jauh di bawah CI ±0.0226 → **statistically inconclusive-to-worse**,
arah konsisten negatif.

**Keputusan: RETAIN `rest_days_k=0` (off)**. Infra `--rest-days-k` tetap ada
sebagai eksperimen, tidak diaktifkan di config.

## B. Opponent-Adjusted Form (`--opp-adj-form`)

Bobot setiap hasil lini masa dengan kekuatan lawan saat itu
(`10^((opp_elo−1500)/400)`, di-capture pre-match, di-clamp [0.2, 5.0]).

| variant      | LL (aggr) | hit%  | ROI    | bets | per-season LL |
|--------------|-----------|-------|--------|------|---------------|
| baseline     | 0.9886     | 53.1% | −1.9%  | 623  | 0.9888 / 0.9503 / 0.9933 / 1.0220 |
| opp_adj_form | 0.9886     | 53.0% | −2.0%  | 631  | 0.9887 / 0.9503 / 0.9932 / 1.0222 |

**Hasil**: LL identik pada 4 desimal; per-season beda ±0.0001–0.0002 **dua
arah** (3 musim membaik tipis, 1 memburuk) — tidak ada sinyal konsisten,
kalah jauh dari CI. Prediksi berubah (bets 623→631) tapi tanpa gain.

**Keputusan: RETAIN baseline (flag off)**. Implementasi additif
(`_team_stats` mendukung triple `(gf, ga, opp_strength)`) tetap tersedia
untuk evaluasi ulang saat data/hipotesis berubah.

## C. xG (`xg_weight` blend 0.65)

1. **Bukti inertness** pada dataset tanpa xG: `xg_weight` 0.0 vs 0.65
   menghasilkan metrik **identik** (agregat & keempat musim) →
   `inert_proven: True`. Blend xG produksi **tidak pernah aktif dalam
   backtest** karena tidak ada dataset xG historis — fitur ini murni live-only.
2. **Sumber xG riil tidak tersedia offline saat ini**: football-data.co.uk
   unreachable (proxy Tor 127.0.0.1:9050 mati, koneksi langsung ditolak),
   cache FBref lokal (`~/soccerdata`) tidak memuat kolom xG, tidak ada cache
   xG lain. Sesuai PHASE 9: **xG yang hilang TIDAK difabrikasi**.
3. Builder dataset xG riil (`ablation.py --download-xg`): unduh CSV
   football-data (kolom xG/xGA), join by (date, home, away) ke fixture freeze,
   hitung rolling pre-match team xG for/against (window 5, kronologis, tanpa
   leakage), tulis `cache/football/epl_fixtures_2022_2026_xg.json` — siap
   dieksekusi begitu sumber reachable (coverage ≥0.9 gate).

**Keputusan: RETAIN produksi** (blend 0.65 tidak diubah — tidak ada bukti
untuk mengubahnya, dan tidak ada bukti untuk menghapusnya). Status dicatat
jujur: *xG blend adalah fitur live-only yang belum pernah divalidasi
walk-forward* (limitation yang sudah di-flag sejak audit PHASE 0).

## Keputusan Akhir (PHASE 15 / 37)

| Fitur | Δ LL vs baseline | Luar noise? | Konsisten lintas musim? | Keputusan |
|-------|------------------|-------------|--------------------------|-----------|
| Rest days (k>0) | +0.0002…+0.0009 | Tidak (CI ±0.0226) | Konsisten buruk (4/4) | **RETAIN k=0** |
| Opp-adj form | +0.0000 | Tidak | Tidak (2 arah) | **RETAIN baseline** |
| xG (backtest) | inert (0.0000) | — | — | **RETAIN; live-only, unverifiable** |

**Tidak ada perubahan config produksi, tidak ada perubahan model produksi,
tidak ada regresi Discord.** Sistem menjawab dengan jujur: *no production
improvement — existing model retained*.

## File

**Berubah (additif, default-off):**
- `agents/football/validate.py` — `_ctx_for` membaca field xG pre-match dari
  fixture (absent → None, never fabricated); `_opp_strength`; parameter
  kw-only `opp_adj_form` di `_ctx_for` / `run_multi_season_validation` /
  `run_cross_league_validation`; flag CLI `--opp-adj-form`; key
  `opp_adj_form` di result.
- `agents/football/models.py` — `_team_stats` memberi bobot ekstra pada
  elemen ketiga tuple recent (backward-compat: 2-tuple byte-identik);
  annotation diwidening.
- `agents/football/leakage_audit.py` — invariant membandingkan prefix
  `g[:2]` (robust terhadap triple opp-adj).

**Baru:**
- `agents/football/ablation.py` — harness eksperimen + builder dataset xG.
- `tests/test_ablation.py` — 8 test offline (opp-adj smoke, _opp_strength,
  xG passthrough, rolling xG tanpa leakage, parse CSV xG, join).
- `reports/phase3_4_ablation.json` — hasil lengkap + delta table.

**Tidak berubah:** semua modul `baseline/`, `config/football.json`, pipeline
produksi, kontrak Discord.

## Validasi

- Full suite: **385 passed** (sebelumnya 374 → +11, tanpa regresi).
- Audit anti-leakage EPL: **PASS** (equivalence LL 0.9886 == produksi;
  determinism 0 diff; 0 violation predict-before-update; provenance & CLV
  scope tetap covered).
- Test baru: opp-adj (smoke + `_team_stats` 3-tuple + backward-compat),
  `_opp_strength`, passthrough xG (berpengaruh saat data ada / inert saat
  tidak), rolling xG (kronologis, tanpa leakage), parse CSV, join.

## Limitasi (jujur)

1. **xG tidak dapat divalidasi offline saat ini** — evaluasi riil tertunda
   sampai sumber reachable (`--download-xg` siap pakai). Tidak ada xG
   sintetis yang dibuat.
2. **Window xG** di `_build_xg_features` adalah "5 match terakhir yang punya
   data xG" (match tanpa xG tidak ikut window); dibatasi aman oleh gate
   coverage ≥0.9 pada join same-source.
3. **Caveat same-day** (ordering tanggal-only) tetap berlaku seperti di
   audit — limitasi dataset historis, bukan kebocoran produksi.
4. Ablasi mencakup EPL 4 musim (dataset yang ada). Perluasan liga/musim bisa
   mengubah kesimpulan; infra siap (cross-league validate).
