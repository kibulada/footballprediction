# Feature Provenance Report (MASTER PROMPT PHASE 2 / 16-17)

Dokumentasi tiap fitur yang masuk ke prediction engine: sumber, timestamp,
ketersediaan pre-match, perilaku missing, dan status leakage. Prinsip PHASE 2:
**tidak ada fitur yang boleh masuk model produksi kecuali benar-benar
tersedia pada prediction timestamp** (`feature_available_at <= prediction_timestamp`).

Status audit: **PASS** — semua fitur pre-match. Detail verifikasi otomatis di
`reports/leakage_audit_epl.json` (equivalence 0.9886, 0 violations, determinism OK).

---

## Feature table (production path: analyse → predictor → models)

| Feature | Source | Timestamp semantik | Pre-match? | Missing behavior | Leakage risk |
|---|---|---|---|---|---|
| Elo rating (home/away) | `elo.py` — walk-forward/seed, K-adaptive | ratings hanya update **setelah** match (strictly post-result) | ✅ Ya (rating lawan dipakai sebelum update) | default 1500 saat baru | **TIDAK** — update setelah prediksi; diverifikasi `_state_count_violations` |
| Recent form (GF/GA avg, rolling 5) | `validate.py`/`analyse.py` — deque hasil match sebelumnya | match sebelumnya saja | ✅ Ya | None → Poisson skip (shrinkage) | **TIDAK** — append setelah prediksi match ini |
| Home/away attack & defense (λ via Elo) | `elo.expected_lambdas` + `PoissonModel` | pre-match rating | ✅ Ya | shrinkage ke base rate | **TIDAK** |
| Home advantage | `models.py` (home λ boost) | konstanta historis | ✅ Ya | — | **TIDAK** |
| Rest days (`rest_days_k`, default 0 = OFF) | `context.py` — dari tanggal match sebelumnya | perbedaan tanggal ≤ match ini | ✅ Ya | None (tanggal rusak) | **TIDAK** — hanya tanggal sebelumnya; ablasi PHASE 3-4: **RETAIN k=0** (tidak improve) |
| Opponent-adjusted form (`--opp-adj-form`, OFF) | `validate.py` — rating lawan pre-match | pre-match opponent rating | ✅ Ya | — | **TIDAK** — capture sebelum `elo.update`; ablasi: netral (RETAIN baseline) |
| xG/xGA (`home_xg_for` dll) | **understat.com** per-match (real browser, `window.datesData`) — backtest EPL 2022-2026 coverage **100%**; live: rolling last-5 finished | rolling **sebelum** match (append setelah compute) | ✅ Ya | **tidak difabrikasi** — absen → xg_weight inert | **TIDAK** — window rolling pre-match; validated OOS: **5/5 liga improve LL** (PHASE 9/14/15) |
| Odds 1X2 consensus | `odds_fetcher`/`multi_source` (14 bookie live; Pinnacle historis) | snapshot pada prediction timestamp | ✅ Ya | odds None → market row kosong, edge tidak dihitung | **TIDAK** — odds yang dipakai adalah yang ADA saat prediksi; closing odds TIDAK dipakai untuk prediksi historis (CLV hanya evaluasi) |
| Odds totals O/U 2.5 | sama (Pinnacle historis) | sama | ✅ Ya | None → market totals kosong | **TIDAK** |
| Margin-free implied | turunan odds (decision.py) | sama dengan odds | ✅ Ya | — | **TIDAK** — fungsi murni odds |
| Calibration (a,b log-odds) | `calibration.py` — fit in-sample aggregate | fit dari sejarah ≤ now | ✅ Ya (live) | min_samples < 200 → LEAN cap | **TIDAK** — evaluasi OOS PHASE 5-6: near-identity, netral |
| Completeness / data quality | dari ketersediaan fitur match ini | pre-match | ✅ Ya | skor turun → cap confidence | **TIDAK** |
| Model agreement (elo/poisson/DC/market) | `models.py`/`decision.py` | probabilitas pre-match | ✅ Ya | — | **TIDAK** |
| Historical reliability (similar signal) | `prediction_log.py` — snapshot tersettle SEBELUM sekarang | bucket history ≤ now | ✅ Ya | <5 sampel → "belum cukup" | **TIDAK** — hanya snapshot tersettle |
| H2H | `context.py` | match sebelumnya kedua tim | ✅ Ya | None → tidak dipakai | **TIDAK** |
| Lineups/injuries | **belum tersedia** (tidak ada sumber riil pre-match) | — | ❌ Tidak diimplementasikan | dilaporkan sebagai keterbatasan, TIDAK difabrikasi | n/a |
| xG riil (backtest) | understat.com `datesData` per-match (EPL/LaLiga/Serie A/Bundesliga/Ligue 1 2022-2026) via `understat_xg.py` | pre-match rolling (window 5) | ✅ coverage 1.0 (EPL join 1520/1520) | offline (network block) → inert, tidak difabrikasi | **TIDAK** — verified OOS: LL improve di 5/5 liga |

---

## Invariants anti-leakage yang diuji otomatis (`leakage_audit.py`)

- **A — state count**: form state tim berisi PERSIS match yang sudah
  diproses (update-before-predict refactor → tertangkap).
- **B — state content**: isi deque (gf, ga, dan bila opp-adj: opp_strength)
  cocok dengan match sebelumnya yang diproses.
- **C — pipeline equivalence**: audit replay LL == validate walk-forward LL
  (0.9886 EPL) → jalur produksi dan audit identik.
- **D — determinism**: input_hash stabil; snapshot bisa reproduksi prediksi.

## Sumber data (PHASE 17 — prioritas)

1. **FBref** (via soccerdata, cache `~/soccerdata` + `cache/football/*_fixtures_*.json`)
   — fixtures + skor historis (EPL/LaLiga/Bundesliga/Serie A 2022-2026, offline OK).
2. **football-data.co.uk** — odds Pinnacle historis (EPL cache
   `odds_source=pinnacle`); jaringan saat ini unreachable tanpa proxy.
3. **understat.com** — xG per-match riil (5 liga besar, 2022-2026) via
   browser UC; API ajax aktif mem-404 client HTTP biasa, hanya browser asli
   yang dilayani (data dibaca dari page-global `window.datesData`).
   Validated OOS: xg_weight 0.65 improve log-loss di **5/5 liga**
   (EPL 0.9886→0.9851, LaLiga 0.9943→0.9907, Serie A 1.0031→1.0009,
   Bundesliga 1.0097→1.0057, Ligue 1 1.0141→1.0100); hit rate naik di 4/5,
   ECE turun (kalibrasi membaik). Sumber sama untuk live (PHASE 9 fallback).
3. **TheSportsDB / SofaScore / Flashscore (live)** — fixture/odds/stat live;
   SofaScore diganti Flashscore (block 403 berulang) — sudah diimplementasikan.
4. **The Odds API (free)** — odds konsensus live (14 bookie).

Rekonsiliasi: fixture identity dicek (tim alias, tanggal, liga) sebelum
prediksi; jika tidak bisa diverifikasi → NO BET (PHASE 17).

## PHASE 32-33 — Prediction timing snapshots & CLV (baru)

- **Prediksi** dicatat SATU KALI sebagai snapshot immutable (T-prediksi).
- **Odds snapshot** (`event: odds_snapshot`) menangkap harga 1X2 pada label
  waktu T-24h / T-6h / T-1h / T-15m / T-0h (append-only, keyed match_id,
  tiap baris membawa `timing` + `ts` sendiri — **snapshot tidak pernah
  dicampur**).
- **CLV dipisah** (forecast quality ≠ price quality):
  - `clv` (model) = P(pick) × closing_odds − 1
  - `price_clv` = closing_odds / prediction_odds − 1
  - `clv_by_timing` = price CLV vs tiap timing snapshot
- CLI/bot: `!football odds <T-24h|T-6h|T-1h|T-15m> <home> vs <away> <h,d,a> [league]`.
- CLV **hanya evaluasi** — closing/CLV tetap dilarang sebagai fitur model
  (leakage_audit `check_clv_scope` PASS).

## Keterbatasan jujur

- Lineup/injury pre-match: **tidak tersedia** → tidak dimasukkan (PHASE 11).
- xG: **sekarang punya sumber riil** (understat) — backtest validated 5/5 liga.
  Live: fallback understat dipasang untuk big-5 saat sofascore-history diblokir;
  batasan live: awal musim baru (Agustus-Desember) window last-5 didominasi
  ekor musim lalu (tidak ada sumber per-match xG yang lebih baru tersedia),
  dan understat hanya mencakup 5 liga besar (+ RFPL).
- Same-day matches diproses dalam urutan sebarang (FBref tanpa jam kickoff) —
  caveat day-granularity yang diketahui.
- CLV historis (backtest) masih tidak tersedia (cache tanpa split open/close);
  odds snapshot bertingkat sekarang memungkinkan evaluasi CLV ke depan begitu
  snapshot tersettle dengan closing odds.
