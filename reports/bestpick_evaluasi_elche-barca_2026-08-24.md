# Evaluasi BEST PICK — Kasus Elche vs Barcelona (FT 0-5)

Tanggal evaluasi: 2026-08-24. Metode: bedah snapshot riil
`cache/football/predictions.jsonl` (2 snapshot pre-match: 2026-08-23T18:06:17Z
dan 18:21:05Z) + telusur kode lengkap rantai keputusan. Status: **EVALUASI
SAJA — belum ada perubahan kode**. Rencana perbaikan menunggu approval.

---

## 1. Ringkasan eksekutif

Bot mengeluarkan `🔥 BEST PICK: BTTS YES @ 1.88 (MEDIUM, 59/100)` untuk match
yang berakhir 0-5. Intuisi operator (Away -1.5 / Over 2.5 / Away Win) adalah
**pandangan market**; model gagal melihatnya karena satu angka rusak di hulu:

> `lambda_home 1.542 > lambda_away 1.359` — model menilai Elche lebih
> produktif dari Barcelona, padahal Elo gap 715 poin (1582.7 vs 2298.0).

Dari λ terbalik ini lahir semua gejala di kartu: P(Barca menang 2+ gol) = 15%
vs market 51% (deviasi 36pp), BTTS Yes 58% vs market 51% (deviasi 7pp — lolos
gate), Over/Under 2.5 deviasi 8.4pp (ditolak gate). Kandidat yang "kebetulan
paling dekat ke pasar" (BTTS) selamat dan dipublikasi, padahal model dan
market sedang tidak menonton pertandingan yang sama.

---

## 2. Bukti angka (snapshot 18:06:17Z, kartu yang diterima user)

| Sinyal | Model | Market (fair) | Dev | Hasil gate |
|---|---|---|---|---|
| BTTS Yes @1.884 | 58.41% | 51.37% | 7.0pp | ✅ lolos G2 → BEST PICK 0.586 MEDIUM |
| Over 2.5 @1.502 | 55.43% | 63.82% | 8.4pp | ❌ G2 (>8pp), skor dipertahankan 0.373 |
| Under 2.5 @2.65 | 44.57% | 36.18% | 8.4pp | ❌ G2, skor 0.345 |
| AH Away -1.5 @1.83 | **15.03%** | 51.07% | **36.0pp** | ❌ G2, skor 0.155 |
| AH ±0.25 | — | — | — | ❌ G7 tanpa harga (konsensus line -1.5 saja) |
| 1X2 Away | 68.0% | ~75.7% | edge -6.4pp | layer keputusan: NO CLEAR DECISION |

Komponen skor BTTS Yes (rekonstruksi persis): model .584, market **1.0**,
movement .5, late .5, data_quality .8, statistical .2 →
`(0.35·0.584 + 0.25·0.2 + 0.20·1.0 + 0.15·0.5 + 0.05·0.8) × 1.03` (boost BTTS
λ_total>2.5) `= 0.586` ✓ cocok dengan log.

Snapshot 18:21:05Z (xG tiba, λ membaik arahnya 1.252 < 1.614): BTTS Yes tetap
dipertahankan stability guard (skor 0.57, edge menyusut ke 3.91pp).

---

## 3. Temuan (F1–F10, urut dampak)

### 🔴 F1 — Akar masalah: λ Totals/BTTS nol kontribusi Elo saat samples ≥ shrinkage
- `agents/football/models.py:245-252` — blend weight `w = min(1, min(hs,as_)/shrinkage)`;
  config `models.poisson.shrinkage_samples: 5`, `min_samples: 2`. Dengan
  `lambda_samples: 5` → `w = 1.0` → λ murni rasio gol 5 match terakhir
  (attack_home 1.6 / defense_away 1.2 dst.) — musim awal = noise dominan.
- Elo 2298 vs 1583 **tidak pernah masuk** jalur ini. Formula
  `λ = base × √(atk×def)` tidak punya anchor kekuatan tim.
- A/B backtest "blend mode" (models.py:633-650) menyimpulkan no-difference,
  tapi pengukurannya rata-rata liga EPL — kasus gap-Elo ekstrem tidak diukur.
  Blind spot metodologi, bukan pembuktian aman.

### 🔴 F2 — Struktur model belah: 1X2 dikalibrasi market, Totals/BTTS tidak
- `Ensemble.predict` (models.py:498-504) menjalankan `_cross_league_adjustment`
  + `_calibrate_total_goals` (tarik λ harga pasar O/U) — tapi hasilnya hanya
  dipakai internal untuk 1X2 ensemble.
- `run_prediction_engine` (models.py:744-856) menghitung over/BTTS/AH dari λ
  jalur seleksi terpisah yang TIDAK tersentuh kalibrasi tersebut.
- Efek: 1X2 away tampak waras (68%, tertarik market), sementara Totals/BTTS
  sungguh percaya Elche ≈ Barca → kontradiksi internal pada satu kartu.

### 🟠 F3 — Gate G2 per-kandidat; tidak ada cek konsistensi bentuk antar-market
- `agents/football/pick_gates.py:185-209`, wiring `signal_engine.py:2331-2339`.
- Dev Under ≡ dev Over (mirror `p_under = 1−p_over`) — satu ketidaksesuaian
  menolak dua sisi sekaligus (by design oke).
- BTTS dinilai sendirian: 7.0pp < 8pp lolos, padahal kartu yang sama AH
  Away -1.5 menyimpang 36pp. Gabungan keduanya = sinyal regime mismatch yang
  tidak tertangkap siapa pun → kandidat paling "kebetulan dekat" selamat.

### 🟠 F4 — Komponen market MENGHADIAHI divergensi
- `signal_engine.py:795-803` `_market_component`: credit penuh 1.0 pada
  edge +7..10pp. Bukti: `components.market = 1.0` pada BTTS Yes yang kalah.
- Rekomendasi postmortem 2026-08-22 #2 ("balik tanda komponen market") belum
  diimplementasi; hanya gate keras G2 yang jalan.

### 🟠 F5 — Pick dipublikasi meski layer 1X2 menolak (KEBIJAKAN OPERATOR)
- Config `models.signal_engine.pick_gates.respect_model_decision: false`
  → veto G1 OFF (`signal_engine.py:1629-1679`); catatan
  "model 1X2: NO CLEAR DECISION" hanya `internal_notes`.
- Bypass kedua: `analyse.py:2997-3015` `_strong_pick` (skor ≥0.50 & ≥MEDIUM)
  mempertahankan pick meski evidence gate (P1-2) gagal.
- **Keputusan operator 2026-08-24: biarkan seperti sekarang.** Dicatat di sini
  agar konsekuensinya eksplisit: pick tanpa dukungan layer model akan terus
  tayang sebagai 🔥 BEST PICK dan bisa lolos gerbang `!best`.

### 🟡 F6 — Detektor kontradiksi λ-vs-1X2 ada tapi tidak dirender
- `signal_engine.py:2460-2486` (P0-4 `lambda_warning`) menghitung persis kasus
  ini ("Lambda favors Home ... 1X2 favors Away") — `format.py` tidak pernah
  mereferensikannya. Operator buta terhadap alarm ini.

### 🟡 F7 — Gate G3 (`lambda_1x2_consistency`) default OFF
- Config `"lambda_1x2_consistency": false`; wiring `signal_engine.py:2349-2363`.
  Kalau ON, semua kandidat AH diveto di 18:06 dengan alasan kontradiksi
  eksplisit (favorit away ≥60% vs λ menunjuk home).

### 🟠 F8 — Bug entitas: "Barcelona" ter-resolve jadi RCD Espanyol de Barcelona
- Bukti: `match_id` snapshot `"LaLiga||cid:t:laliga:elche-cf||cid:t:laliga:
  rcd-espanyol-de-barcelona||2026-08-23"` padahal display away = "Barcelona"
  dan Elo lookup (jalur lain) menemukan FC Barcelona 2298.
- Jalur bug: `team_alias.resolve_team_alias` pass boundary
  (`team_alias.py:103-107`) — alias key `"RCD ESPANYOL DE BARCELONA"`
  (teams.json:67) mengandung kata `\bbarcelona\b`; bare "Barcelona" tidak
  punya alias eksplisit (hanya "BAR"/"BARÇA" → teams.json:48,73,212).
- Dua lapis resolusi tidak konsisten → risiko form/H2H/xG/cache salah routing,
  identity-lock & stability guard salah join riwayat. Pola sama dengan kasus
  Forest/Leeds→Man Utd dan Troyes "PSG" sebelumnya.

### 🟡 F9 — Pin λ membekukan estimator yang salah
- models.py:750-758: pin konsistensi-vs-akurasi disengaja. Di 18:21 xG datang
  (λ arah benar) tapi pin + stability guard menahan BTTS Yes. Trade-off sah,
  namun mengunci kesalahan F1 lebih lama.

### ⚪ Housekeeping
- `_total_disagreement_veto` (`signal_engine.py:806-831`) dead code — masih
  men-zero `s.score`, kontradiktif desain baru "veto ≠ zero score".
- Poisson `rho=0.0` (tanpa Dixon-Coles) — bias kecil ke arah BTTS/Under.
- Kartu SIGNALS hanya render kandidat teratas; Away -1.5 yang diveto 36pp tak
  terlihat — info paling diagnosable justru tersembunyi.

Yang bekerja BENAR (catatan adil): G2/G7 menolak sesuai desain; skor kandidat
ditolak tetap ditampilkan (37/100, 34/100 di kartu); stability guard jalan;
log JSONL lengkap sehingga kasus ini bisa diaudit mundur. Masalah utamanya
bukan gerbangnya — melainkan data hulu (F1/F2), insentif skor (F4), dan
kebijakan publikasi (F5).

---

## 4. Rencana perbaikan (PROPOSAL — belum dieksekusi)

| Fase | Isi | File target | Risiko | Verifikasi |
|---|---|---|---|---|
| **0 — Quick win** | (a) Alias `BARCELONA`→FC Barcelona + regresi test resolve; audit nama-kota-bare lain ("Valencia", "Sevilla", dll). (b) Render `lambda_warning` di card. (c) Hapus dead-code `_total_disagreement_veto` | teams.json, team_alias.py, format.py, signal_engine.py | Sangat rendah | Repro resolve sebelum/sesudah; pytest test_elo/test_signal_engine |
| **1 — Akar masalah** | Elo-anchor λ Totals/BTTS: jika kedua tim seeded → blend λ fitur ke λ Elo seiring gap `t = clip((\|Δelo\|−150)/400, 0, 1)` + jamin arah λ = arah favorit Elo. Config `models.poisson.elo_anchor.{enabled,min_gap,max_blend}` | models.py (run_prediction_engine/PoissonModel), config/football.json | Sedang (ubah prob global) | Unit test pure; replay input Elche offline (btts harus turun ~40%, AH away -1.5 naik ~50%); protokol A/B AGENTS.md (bot stop, cache disetarakan); smoke suite |
| **2 — Gate bentuk G9** | Shape-consistency: bandingkan distribusi margin model (`ah_win_prob`) vs implied AH consensus; dev pasangan >20–25pp → regime mismatch: veto kandidat non-directional / NO BET card-level. Config `pick_gates.shape_consistency` | pick_gates.py, signal_engine.py | Sedang | Ukur dulu di book JSONL historis sebelum ON; test mekanika via cfg |
| **3 — Balik tanda komponen market** | Reward kesesuaian, penalti mulus \|dev\| dalam band gate | signal_engine.py `_market_component` | Sedang | Backtest `run_signal_backtest` atas JSONL sebelum enable |
| **4 — Kebijakan publikasi** | ~~Opsi a/b~~ → **DIPUTUSKAN 2026-08-24: biarkan seperti sekarang** (respect_model_decision tetap false). Konsekuensi didokumentasikan di F5 | — | — | — |

Urutan disarankan bila nanti dieksekusi: **0 → 1 → (ukur) → 2/3**. Fase 0+1
saja sudah menyembuhkan kasus Elche: dengan λ ter-anchor Elo, BTTS Yes jatuh
~40% (tidak lolos floor skor) sementara Over 2.5 / AH Away -1.5 naik ke
deviasi <8pp dan kompetitif — keduanya MENANG malam itu.

---

## 5. VALIDASI BUKU PENUH (update 2026-08-24, permintaan operator)

Operator melaporkan 3 kasus gagal tambahan dan meminta review seluruh buku.
Sumber: `cache/football/predictions.jsonl` — 434 snapshot, 120 baris settle,
**67 pick terpublikasi unik yang sudah ada settlement-nya** (settlement
infrastruktur sudah ada: `settler.py` `settle_auto` + laporan ROI/CLV di
`prediction_log.py`; sebagian match 23 Agu belum ter-settle otomatis).

### 5.1 Rekap buku terpublikasi (flat stake 1u)

| Bucket | n | Staked | Return | ROI |
|---|---|---|---|---|
| **ALL** | **67** | **58u** | **53.41u** | **−7.9%** |
| conf MEDIUM | 60 | 52u | 45.65u | −12.2% |
| conf HIGH | 4 | 4u | 5.64u | +41.0% (n kecil) |
| dt: NO BET | 31 | 29u | 26.18u | −9.7% |
| dt: NO CLEAR DECISION | 36 | 29u | 27.22u | −6.1% |
| market BTTS | 10 | 7u | 5.24u | **−25.2%** |
| market Asian Handicap | 30 | 24u | 22.06u | −8.1% |
| market Total | 27 | 27u | 26.11u | −3.3% |
| score ≥0.70 | 16 | 16u | 13.56u | **−15.2%** |
| score 0.55–0.70 | 40 | 33u | 33.11u | **+0.3%** |
| score <0.55 | 11 | 9u | 6.74u | −25.1% |

Temuan kunci dari buku:
1. **100% pick terpublikasi berasal dari layer keputusan yang MENOLAK**
   (`NO BET` / `NO CLEAR DECISION`) — konsekuensi langsung
   `respect_model_decision=false`. Buku ini = "pick yang model tolak", ROI −7.9%.
2. **Skor TIDAK monoton terhadap profit**: bucket skor tertinggi (≥0.70) justru
   −15.2%, bucket tengah +0.3%. Mengkonfirmasi ulang postmortem ("score bukan
   prediktor") — DAN membuktikan ide "naikkan threshold skor" tidak menyelesaikan
   apa pun. Label confidence perlu kalibrasi empiris, bukan threshold baru.
3. **BTTS keluarga terburuk (−25.2%)** — konsisten dengan pola Elche/Atalanta:
   BTTS paling sering jadi "yang lolos kebetulan" saat bentuk model≠market.

### 5.2 Analisis 3 kasus operator

**Kasus 1 — Goztepe vs Genclerbirligi (FT 1 gol; card: OVER 2.5)**
Snapshot 17:39:58Z: λ_total = 2.302+1.807 = **4.109** (features+xg) → SEMUA 8
kandidat signal engine diveto G4 (band [1.6, 3.6]) — gate bekerja benar.
TAPI kartu tetap menampilkan "BEST PICK OVER 2.5" karena itu berasal dari
**decision layer** (`best_pick` format rank, grade LOW) yang **tidak pernah
melewati pick_gates sama sekali**.
→ **TEMUAN BARU F11**: pick decision-layer bocor ke display tanpa gating;
renderer menampilkannya (dengan label ⚠️ HIGH RISK) meski SE sudah NO BET
kartu-level. Akar λ meledak = keluarga F1/F2 (market implied total ≈2.6,
model bilang 4.1).

**Kasus 2 — Atalanta vs Sassuolo (FT 2-1; card: BTTS No LOW 0.509)**
λ_total 2.34 vs market imlied ±3.0+; di kartu yang sama AH Home -1.5 deviasi
25.3pp (model bilang Atalanta tidak besar margin, market bilang blowout).
BTTS No lolos G2 karena deviasinya cuma 5.4pp → pola **shape-blindness yang
sama persis dengan Elche** (F3). Confidence LOW tetap dipublikasi (F5).
→ Dicover oleh Fase 2 (G9 shape gate) + catatan khusus BTTS (5.3).

**Kasus 3 — CFR Cluj vs FCSB (FT total 1 gol; card: Over 2.5 VERY HIGH 0.815)**
Model 60.9% vs market fair ~56.4% — deviasi hanya 4.6pp, model DAN market
sepakat expect gol. Ini **bukan kesalahan gate**: P(total ≤1 | λ≈3.15) ≈ 18%
— varian normal sepak bola. Tidak ada sistem pre-match yang bisa menyaring
kasus ini secara andal tanpa ikut membuang mayoritas winner.
→ Pelajaran penting: ekspektasi "BEST PICK tidak pernah salah lagi" **tidak
dapat dijanjikan oleh sistem apa pun**; target yang valid adalah (a) buang
pola negatif-EV terukur, (b) label jujur via kalibrasi, (c) ROI buku naik
dan terukur mingguan.

### 5.3 Matriks cover plan vs 4 kasus

| Kasus | F0 entity/warn | F1 Elo-anchor λ | F2 calibr-total | F3→G9 shape | F4 market comp | F5 policy | Tambahan v2 |
|---|---|---|---|---|---|---|---|
| Elche BTTS Yes | — | ✅ akar | ✅ | ✅ | ✅ | ➖ | — |
| Goztepe leak | — | ✅ (λ 4.1→wajar) | ✅ | — | — | — | ✅ **F11 fix** |
| Atalanta BTTS No | — | ✅ | ✅ | ✅ | ➖ | ➖ | ✅ BTTS guard |
| CFR Cluj Over VH | — | ❌ variance | ❌ | ❌ | ❌ | ❌ | ✅ **kalibrasi label** |

### 5.4 TEMUAN & RENCANA TAMBAHAN (v2)

**F11 — Decision-layer pick bocor tanpa gating** (Goztepe): saat signal engine
berjalan dan SEMUA kandidat diveto, `best_pick` decision-layer (format rank)
tetap dirender sebagai BEST PICK (⚠️ HIGH RISK). pick_gates tidak pernah
menyentuhnya. Fix: renderer wajib menampilkan NO BET + alasan gate ketika SE
aktif dan semua kandidat diveto; atau jalankan pick_gates atas pick
decision-layer sebelum render. File: `format.py` `_display_best_pick`,
`analyse.py` payload assembly.

**F12 — Kalibrasi label confidence** (CFR Cluj): mapping skor→label tidak
sesuai realisasi (bucket ≥0.70 −15.2% vs 0.55–0.70 +0.3%). Fix: tabel
kalibrasi empiris dari log settle (bucket ROI/hit-rate per rentang skor ×
liga), dipublikasi sebagai label probabilitas jujur ("MEDIUM ≈ 52% hit
historis"). File: `signal_engine.py` confidence_label / modul kalibrasi baru,
data dari `prediction_log` report. Re-kalibrasi berkala otomatis.

**F13 — Guard khusus BTTS** (buku −25.2%): BTTS hanya bettable bila confluence
≥3 DAN \|dev\| ≤5pp ANDA dukungan statistik kedua tim; selain itu NO BET.
Config-gated, diukur lewat replay sebelum ON.

**Fase 6 (baru) — Settlement loop harian**: jalankan `settler.settle_auto`
terjadwal (runner mode yang sudah ada) supaya setiap pick punya hasil ≤24 jam,
+ laporan mingguan ROI per bucket (conf/market/score/dev) sebagai dasar
keputusan enable/disable tiap fase. Tanpa ini, "perbaikan valid" tidak bisa
diverifikasi.

### 5.5 Ekspektasi realistis (penting)

Permintaan "setelah perbaikan BEST PICK tidak ada lagi kesalahan dan selalu
WIN" tidak dapat dijanjikan siapa pun — termasuk pasar sendiri (CFR Cluj:
model & market sepakat, tetap kalah 18% kemungkinan tersebut). Yang dapat
dijamin dan diverifikasi angka:
1. Pola negatif terukur hilang (Elche-class λ rusak; Goztepe-class leak;
   Atalanta-class shape mismatch; buku BTTS −25%).
2. Label confidence dikalibrasi empiris (VERY HIGH benar-benar berarti hit-rate
   tinggi historis).
3. ROI buku terpublikasi naik dari baseline **−7.9%** dan dilaporkan mingguan
   dari settlement loop.

---

## 6. HASIL EKSEKUSI PAKET INTI v3 (2026-08-24 — selesai)

Diimplementasi & diverifikasi (suite penuh **1465 passed**):

| Fix | Implementasi | Lokasi |
|---|---|---|
| F1 | `apply_elo_anchor()` — blend λ final ke share Elo saat gap ≥150 (penuh di ≥400), audit `model_probs.elo_anchor_t` | models.py |
| F2 | `calibrate_total_to_market()` — tarik λ final ke fair O/U (devig, setengah gap), audit `market_total_calibrated` | models.py |
| F4-lite | `_market_component(agreement_band_pp)` — komponen market menghargai KESESUAIAN (maks di 0pp, 0 di tepi band G2 8pp); config `market_component_reward_agreement` default ON | signal_engine.py |
| F14 | Kandidat **1X2 Home/Draw/Away Win** (implied margin-free 3-outcome), config `enable_1x2_signals`; settlement 1X2 | signal_engine.py, analyse.py threading `odds_1x2` |
| F11 | `_display_best_pick` hanya menampilkan pick SE; snapshot tidak lagi menyimpan decision-layer pick saat SE sudah memutuskan | format.py, analyse.py |

Config baru (`config/football.json`): `models.poisson.elo_anchor`,
`models.poisson.market_total_calibration`, `models.signal_engine.enable_1x2_signals`,
`models.signal_engine.market_component_reward_agreement`.

### Replay 4 fixture (input murni dari log, transformasi = kode produksi baru)

| Match | Pick lama (hasil) | Pick v3 | FT | Selisih |
|---|---|---|---|---|
| Elche v Barcelona | BTTS Yes MEDIUM (**LOSS**) | **Over 2.5 @1.50 — 0.764 HIGH** ✅ | 0-5 | **WIN** |
| Atalanta v Sassuolo | BTTS No LOW (**LOSS**) | **Home Win @1.56 — 0.558 MEDIUM** ✅ | 2-1 | **WIN** |
| Goztepe v Genclerbirligi | bocoran Over 2.5 tanpa gating (**LOSS**) | **NO BET** (λ_total 4.11→3.48 via F2; tak ada kandidat lolos) ✅ | 1 gol | kerugian dihindari |
| CFR Cluj v FCSB | Over 2.5 VERY HIGH (**LOSS**, varian) | BTTS Yes @1.66 — 0.636 MEDIUM (**LOSS**, varian yang sama; label jujur turun dari VERY HIGH) | 1 gol | netral |

Catatan replay:
- Elche: anchor penuh (t=1.00, gap 715) membalik λ 1.542/1.359 → arah Barcelona,
  btts_yes 58%→25%, over 55%→64%; kartu kini menunjuk pasar yang benar.
- Atalanta: anchor penuh (gap 613) membalik λ (home favorit sesuai Elo/market),
  AH Home -1.5 tak lagi menyimpang 25pp; Home Win jadi pick.
- Goztepe: gap Elo cuma 124 → anchor off, tapi F2 meredam ledakan λ_total
  4.11→3.48 sehingga kartu bukan lagi "Over 2.5 palsu"; sisa deviasi model-market
  masih besar → NO BET jujur.
- CFR Cluj: kasus varian murni (model+market sepakat) — tidak ada gate pre-match
  yang bisa menangkapnya; kontribusi v3 adalah label yang lebih jujur.

Test: `tests/test_plan_v3_output_fixes.py` (13 tes baru) + penyesuaian fixture
lama yang menarget mekanika seleksi λ / komponen market legacy (anchor & F4-lite
dimatikan eksplisit di fixture tersebut agar tetap menguji maksud aslinya).
Suite penuh: **1465 passed, 0 failed**.

---

## Lampiran — jejak audit cepat

```
cache/football/predictions.jsonl:
  ts=2026-08-23T18:06:17Z  best_pick=BTTS Yes 0.586 MEDIUM edge 7.04pp
                           ranking[7]=AH Away -1.5 model .1503 implied .5107 VETO 36.0pp
                           decision_type=NO CLEAR DECISION  final_decision=null
  ts=2026-08-23T18:21:05Z  best_pick=BTTS Yes 0.57 MEDIUM (stability held)
config/football.json: models.signal_engine.pick_gates.respect_model_decision=false
                      lambda_1x2_consistency=false ; shrinkage_samples=5 ; min_samples=2
Kode: models.py:245-252,498-504,700-707,744-856 · signal_engine.py:795-803,
      1629-1679,2331-2363,2460-2486 · pick_gates.py:185-209 · analyse.py:2997-3015
      team_alias.py:103-107 · teams.json:48,65-67,73,212
```
