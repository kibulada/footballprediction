# PHASE 3-4 — Ablasi Cross-League: Rest Days & Opponent-Adjusted Form

**Verdict: konfirmasi lintas liga — NO PRODUCTION IMPROVEMENT — EXISTING MODEL RETAINED.**

Kesimpulan EPL (rest-days memburuk / opp-adj netral) **terkonfirmasi di
La Liga, Bundesliga, dan Serie A**. Tidak ada satu liga pun yang menunjukkan
improvement melampaui noise → `rest_days_k=0` dan baseline opp-adj dipertahankan
di semua liga. Ligue 1 tidak dapat dievaluasi offline (jujur, tanpa fabrikasi).

## Metode

- **Sumber**: FBref **cache lokal** (`~/soccerdata/data/FBref/`) via soccerdata
  `read_schedule(force_cache=True)` — offline, skor asli 2022–2026, tanpa
  fabrikasi. Data di-cache ke `cache/football/{liga}_fixtures_2022_2026.json`
  (write atomik tmp+`os.replace`, PID-suffix untuk race dua proses paralel).
- **Replay kronologis independen per liga** (Elo/form/base-rate sendiri),
  konfigurasi produksi yang sama dengan freeze EPL.
- **Tanpa odds** untuk liga baru → ROI n/a; perbandingan LL/Brier/ECE/hit
  tetap valid (model vs model pada fixture yang sama).
- Eksperimen per liga: `rest_days_k ∈ {0, 0.1}` (titik kerusakan maksimum pada
  kurva EPL yang monoton; `--rest-ks` memungkinkan sweep penuh) + opp-adj
  on/off. Kriteria PHASE 15: improvement hanya dihitung jika melampaui CI.
- Hasil: `reports/phase3_4_ablation_cross_league.json` (merge) +
  `reports/phase3_4_cross_{laliga,bundesliga,seriea}.json`.

## Hasil (ensemble log-loss, walk-forward 2022–2026)

| Liga | n | baseline LL | CI ± | k=0.1 LL | Δ rest | opp-adj LL | Δ opp |
|------|----|------------|------|----------|--------|-----------|-------|
| EPL (referensi) | 1520 | 0.9886 | 0.0226 | 0.9895 | **+0.0009** | 0.9886 | 0.0000 |
| La Liga | 1520 | 0.9975 | 0.0218 | 0.9978 | **+0.0003** | 0.9974 | −0.0001 |
| Bundesliga | 1231 | 1.0036 | 0.0248 | 1.0046 | **+0.0010** | 1.0035 | −0.0001 |
| Serie A | 1521 | 1.0033 | 0.0224 | 1.0031 | **−0.0002** | 1.0034 | +0.0001 |

### Rest days (k=0 vs k=0.1)
- **3/4 liga memburuk** (+0.0003 s/d +0.0010), Serie A −0.0002 — jauh di bawah
  CI ±0.0224 (statistik = nol). **Tidak ada liga yang melampaui CI** → arah
  konsisten dengan EPL. **RETAIN `rest_days_k=0` di semua liga.**

### Opponent-adjusted form (on vs off)
- Delta −0.0001…+0.0001 di semua liga, arah campur → **noise murni**, tidak
  ada sinyal konsisten. **RETAIN baseline (flag off) di semua liga.**

## Ligue 1 — tidak tersedia offline

Tidak ada halaman FBref ter-cache (`FRA-Ligue 1` tidak ada di
`~/soccerdata/data/FBref/`) dan football-data.co.uk unreachable (proxy Tor
9050 mati). Sesuai PHASE 9: **tidak ada data yang difabrikasi**. Evaluasi Ligue 1
tertunda sampai sumber reachable; harness siap via `--leagues "Ligue 1"`.

## Per-season baseline LL (konteks kesulitan tiap liga)

| Liga | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|------|---------|---------|---------|---------|
| EPL | 0.9888 | 0.9503 | 0.9933 | 1.0220 |
| La Liga | 1.0206 | 0.9832 | 0.9885 | 0.9976 |
| Bundesliga | 1.0276 | 0.9796 | 1.0336 | 0.9733 |
| Serie A | 1.0198 | 0.9992 | 0.9855 | 1.0088 |

## File

- **`agents/football/ablation.py`** (additif) — mode cross-league:
  `load_league_fixtures` (cache offline + write atomik + sanity check cache
  korup), `run_cross_league_ablation`, `_cross_league_conclude` (klaim hanya
  bila ada baseline; tidak pernah klaim "no improvement" tanpa perbandingan),
  `_cross_league_main`, CLI `--leagues/--seasons/--experiments/--rest-ks`.
- **Cache baru**: `cache/football/{laliga,bundesliga,serie_a}_fixtures_2022_2026.json`
  (data asli FBref, reproducible).
- **Laporan**: `reports/phase3_4_ablation_cross_league.json` + per-liga JSON.
- **Tests** (+5): naming cache path, conclude retain/flag/silent/partial.

## Validasi

- Full suite: **389 passed** (sebelumnya 385 → +4, tanpa regresi; test_ablation
  13/13).
- Code review: semua temuan diterapkan (race tmp shared → PID-suffix, KeyError
  print `rd["0.0"]` → `.get` guard, conclusion tanpa baseline → silent,
  sanity check cache korup).
- Audit anti-leakage EPL tetap **PASS** (tidak ada perubahan jalur produksi;
  modul ini eksperimen murni).

## Limitasi

1. **Ligue 1 tidak dievaluasi** (data offline tidak tersedia; tanpa fabrikasi).
2. **Tanpa odds** pada liga baru → ROI/market baseline tidak terukur di sana
  (valid untuk perbandingan fitur, tidak untuk klaim value).
3. Nama tim FBref vs football-data tidak dinormalisasi antar liga — tidak
  relevan di sini karena replay independen per liga.
4. Caveat same-day (ordering tanggal-only) berlaku seperti EPL.
