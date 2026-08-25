# Post-Mortem BEST PICK — Evaluasi Sharp Bettor

**Tanggal**: 2026-08-22
**Sampel**: pick live 19–21 Agustus 2026 (`cache/football/predictions.jsonl`, 2.161 baris)
**Metode**: grading manual dari baris log mentah (event `snapshot` × `settle`), disilangkan dengan
`pick_diagnosis.txt`, `bias_table.txt`, `livescore_eval.txt`, `replay_current.txt`.

> Catatan eksekusi: script agregat `eval_bestpick_postmortem.py` sudah ditulis tapi tool Bash
> tidak tersedia sepanjang sesi (classifier down). Angka di bawah hand-graded, bukan output script.

---

## 1. Ringkasan eksekutif

Tiga hari live, **~25 pick ternilai, ROI mendekati nol**. Itu bukan kesialan — itu tanda
sistem tanpa edge yang membayar vig.

| Hari | Pick ternilai | W–L | PnL (flat 1u) | ROI |
|---|---|---|---|---|
| 19 Agu (UCL playoff) | 4 | 1–3 | harga tidak tercatat | — |
| 20 Agu (UEL/UECL) | 12 | 8.5–3.5 | **+4.01u** | **+33.4%** |
| 21 Agu (multi-liga) | 9 | 4–5 | **−2.26u** | **−25.1%** |
| **20+21 gabungan** | **21** | **12.5–8.5** | **+1.75u** | **+8.3%** |

+8.3% pada n=21 punya standard error ROI ≈ ±22% (±44% pada 95% CI). **Tidak bisa dibedakan
dari nol.** Siapa pun yang menyimpulkan "bot sudah profit" dari sampel ini salah baca.

**Temuan inti**: mesin pemilih BEST PICK secara struktural memilih *error terbesar model
sendiri*. Komponen `market` dalam skor komposit **memberi nilai tinggi justru saat model
paling menyimpang dari pasar** — dan repo ini sendiri sudah membuktikan (n=1.520, closing
Pinnacle) bahwa deviasi model = noise, bukan informasi.

---

## 2. Ledger lengkap

### 2.1 — 20 Agustus (UEL/UECL): +4.01u / 12u

| # | Match | BEST PICK | Odds | FT | Hasil | PnL |
|---|---|---|---|---|---|---|
| 1 | Kairat Almaty v Anderlecht | AH Home +0.25 | 1.91 | 0–3 | LOSE | −1.00 |
| 2 | Inter Turku v Copenhagen | AH Home +1 | 2.00 | 0–0 | WIN | +1.00 |
| 3 | Lincoln Red Imps v Larne | AH Away +0.5 | 1.87 | 0–2 | WIN | +0.87 |
| 4 | OFI Crete v CSKA Sofia | Over 2.5 | 2.15 | 3–0 | WIN | +1.15 |
| 5 | Benfica v AGF | Over 2.5 | 1.36 | 3–1 | WIN | +0.36 |
| 6 | Sion v Ajax | AH Home +0.75 | 1.84 | 2–4 | LOSE | −1.00 |
| 7 | Atalanta v Hapoel TA | AH Away +2 | 1.97 | 0–0 | WIN | +0.97 |
| 8 | Lugano v Maccabi TA | AH Home −0.5 | 1.86 | 2–1 | WIN | +0.86 |
| 9 | Hearts of Oak v Rapid Wien II | Over 2.5 | 1.63 | 2–2 | WIN | +0.63 |
| 10 | Gent v Hibernian | Under 2.5 | 1.74 | 0–0 | WIN | +0.74 |
| 11 | Motherwell v Freiburg | AH Home +1 | 1.78 | 1–3 | LOSE | −1.00 |
| 12 | Rangers v Jablonec | AH Away +1.25 | 1.93 | 1–0 | HALF-W | +0.465 |
| — | Braga v Austria Wien Women | AH Home +0.25 | `null` | 2–0 | **tak ternilai** | — |

Braga: `market_odds: null` — pick dipublikasikan **tanpa harga**. Pick tanpa harga bukan pick.

### 2.2 — 21 Agustus: −2.26u / 9u

| # | Match | BEST PICK | Odds | FT | Hasil | PnL |
|---|---|---|---|---|---|---|
| 1 | Al Riyadh v Al Nassr | Under 3.5 | 1.86 | 0–4 | LOSE | −1.00 |
| 2 | SV Ried v Grazer AK | Over 2.5 | 1.99 | 1–0 | LOSE | −1.00 |
| 3 | Al-Faisaly v Neom | Over 2.5 | 1.86 | 0–2 | LOSE | −1.00 |
| 4 | Al Qadsiah v Al Ittihad | AH Away +1.5 | 1.94 | 0–1 | WIN | +0.94 |
| 5 | Erzurumspor v Galatasaray | AH Home +1 | 2.06 | 0–4 | LOSE | −1.00 |
| 6 | Marseille v Strasbourg | Over 2.5 | 1.53 | 4–0 | WIN | +0.53 |
| 7 | Cordoba v Girona | Over 2.5 | 1.70 | 2–1 | WIN | +0.70 |
| 8 | Real Betis v Real Sociedad | Over 2.5 | 1.91 | 1–0 | LOSE | −1.00 |
| 9 | Arsenal v Coventry | Over 2.5 | 1.57 | 3–0 | WIN | +0.57 |

**Koreksi terhadap tally bot sendiri.** `livescore_eval.txt` melaporkan 4W/7L dari 11 pick.
Itu salah dua kali:
- **Al-Faisaly v Neom di-log dua kali** (`Al-Faisaly||Neom` dan `Al Faisaly||NEOM SC`,
  keduanya 0–2) → satu kekalahan dihitung dua kali.
- **Erzurumspor v Galatasaray di-log dua kali dengan skor berbeda**: satu entri `0–4`
  (settle line 95, closing 5.75/4.20/1.53 — Galatasaray favorit berat), satu entri **phantom
  `2–3`** yang membuat `AH Home +1` ter-grade **WIN**. Skor sebenarnya 0–4 → Home +1 KALAH.
  Bot mencatat kemenangan yang tidak pernah terjadi.

---

## 3. Anatomi tiap pick yang salah

### L1 — Al Riyadh v Al Nassr, `Under 3.5` @1.86, FT 0–4
```
model 1X2: away 81.7%   |  lam_h 1.358  lam_a 1.638  (total 3.00)
market O/U2.5: over@1.40 under@2.70 -> P(over2.5) 65.9%
RANKING[0] = AH Home +1.75 (0.598)  <-- tapi PICK yang keluar = Total Under 3.5
```
Tiga kesalahan bertumpuk:
1. **Pick ≠ ranking teratas.** Ranking bilang AH Home +1.75; yang dipublikasikan Total Under 3.5.
2. **Bertaruh melawan pasar pada arah yang pasar paling yakin.** Over 2.5 dihargai 1.40 —
   pasar mengharapkan ~3.4 gol. Model bilang 3.00. Mengambil Under 3.5 @1.86 di sini adalah
   menjual gol ke pasar yang sedang membeli gol.
3. **Sinyal terkuat model diabaikan.** Model 81.7% Al Nassr menang — dan Al Nassr menang 4–0.
   Model *benar* pada 1X2, lalu bot bertaruh di market lain yang model *salah*.

### L2 — SV Ried v Grazer AK, `Over 2.5` @1.99, FT 1–0
```
lam_h 2.56  lam_a 1.528  -> total 4.09 gol ekspektasi
model P(over2.5) 77.5%  vs  market 49.0%   -> deviasi +28.5pp
seeded=True complete=0.8  |  form h: L-L-D-W-L   a: L-W-L-W-L
```
4,09 gol ekspektasi untuk dua tim dengan form L-L-D-W-L / L-W-L-W-L adalah **omong kosong
numerik**. Pasar memasang 49%; model memasang 77.5%. Deviasi +28.5pp bukan value — itu
diagnosis bahwa λ rusak. Liga juga salah label: SV Ried v Grazer AK adalah **2. Liga Austria**,
di-log sebagai `Bundesliga`.

### L3 — Al-Faisaly v Neom, `Over 2.5` @1.86, FT 0–2
```
Elo kedua tim = default (unseeded)  ->  lam hanya dari window 5-match
form W-W-W-W-W  ->  lam_tot 3.33  ->  "value" +12.6pp
```
Tanpa prior Elo, λ = ekstrapolasi hot streak. Bot menghitung streak sebagai kekuatan.

### L4 — Erzurumspor v Galatasaray, `AH Home +1` @2.06, FT 0–4
```
model 1X2: away 60.9%   TAPI   lam_h 1.372 > lam_a 1.208
elo_h 1486.96  elo_a 1785.81  complete=0.65
```
**Kontradiksi internal**: ensemble 1X2 bilang Galatasaray 61% menang, tapi matriks λ bilang
tuan rumah cetak lebih banyak gol. Probabilitas AH diwarisi dari λ → AH Home +1 dapat
"prob 0.766". Dua bagian model yang sama saling bertentangan di kartu yang sama, dan bot
memilih bagian yang salah.

### L5 — Real Betis v Real Sociedad, `Over 2.5` @1.91, FT 1–0
```
elo_h 2036.0   elo_a 2361.0        <-- Sociedad 2361?
model 1X2: home 16.5%  vs  market home 44.7%   -> deviasi -28.2pp
model P(over2.5) 62.2%  vs  market 51.5%       -> deviasi +10.7pp
```
Elo 2361 untuk Real Sociedad tidak masuk akal — **dan angka 2361 yang identik juga muncul
sebagai Elo Arsenal** di kartu Arsenal v Coventry hari yang sama. Itu **tabrakan lookup Elo**,
bukan penilaian kekuatan. Akibatnya model salah 28pp pada 1X2, dan λ yang diturunkan dari Elo
rusak itu menghasilkan "value" Over 2.5 yang palsu. Betis menang 1–0.

### L6/L7 — 19 Agustus (dari `BUG_REPORT_5MATCH_AUG19.md`)
- **Nijmegen v Bodø/Glimt** — signal engine konsisten memilih `Over 2.5` di 3 snapshot
  berturut-turut, lalu snapshot terakhir **di-override** jadi `Under 3.5`. FT 1–3 (4 gol).
  Pick yang menang diubah jadi pick yang kalah oleh stability layer.
- **Hapoel Beer Sheva v Sabah** — Elo home 1844 > away 1756, 1X2 home 55.9%, tapi
  λ_home 0.728 **<** λ_away 1.526. Under 2.5 dipilih karena λ total 2.254. FT 2–1 (3 gol).
  Kontradiksi λ-vs-1X2 yang sama seperti L4.

---

## 4. Lima cacat struktural (dengan bukti kode)

### C1 — Skor pemilih BEST PICK tidak berkorelasi dengan profit
`config/football.json:176` — catatan repo sendiri:
> *"Spearman(decision score, realized ROI) = 0.0015 pada EPL 2022-26 walk-forward (n=1520).
> The 7-component score does NOT rank profitable bets; tercile ROI is non-monotonic and
> negative at every tercile."*

Skor komposit **tidak mengurutkan taruhan yang menguntungkan** — dan BEST PICK dipilih dengan
mengurutkan tepat skor itu. Pemilihan pick efektif acak terhadap profit.

### C2 — Komponen `market` adalah selektor anti-edge
`agents/football/signal_engine.py:776`
```python
def _market_component(edge_pp, min_edge, conflict_pp):
    scale = 2.0 * min_edge + 4.0          # min_edge=3.0 -> scale=10.0
    return clamp(0.5 + edge_pp / scale)   # SATURASI di edge_pp = +5pp
```
Docstring bilang "+10pp → full credit"; realitanya **saturasi di +5pp**. Makin besar error
model, makin tinggi skornya — tanpa plafon. Diverifikasi terhadap dump komponen asli:

| Match | `market` comp | edge implisit | Hasil |
|---|---|---|---|
| Marseille | **0.452** | −0.48pp | **WIN** |
| Arsenal | **0.246** | −2.54pp | **WIN** |
| Cordoba | 1.000 | +12.4pp | WIN |
| Betis | **1.000** | +10.7pp | **LOSE** |
| Al-Faisaly | **1.000** | +12.6pp | **LOSE** |
| SV Ried | **1.000** | +28.5pp | **LOSE** |

Dua pick dengan komponen market **paling rendah** keduanya menang. Tiga dari empat pick
dengan komponen market **maksimum** kalah. Komponen ini berkorelasi *negatif* dengan menang.

### C3 — Cap-ke-MEDIUM mengubah "NO BET" jadi kelihatan bettable
`signal_engine.py:1587` — ketika layer keputusan 1X2 mengembalikan
`NO BET / NO CLEAR DECISION / MARKET PRIOR`, pick hanya **di-cap ke MEDIUM**, tidak diveto.

Akibatnya: **11/11 pick 21 Agustus keluar dengan `conf=MEDIUM`, dan 11/11 punya
`decision_type` = `NO CLEAR DECISION` atau `NO BET`.** Mesin keputusan menolak setiap
taruhan; kartu tetap mengiklankan BEST PICK MEDIUM. Ini kegagalan disiplin, bukan kegagalan model.

### C4 — Loop bypass evidence-gate
`analyse.py:2846`
```python
_strong_pick = _bp_score >= 0.50 and _bp_conf in ("VERY HIGH", "HIGH", "MEDIUM")
if _strong_pick:   # evidence gate DILEWATI
```
C3 memaksa confidence jadi MEDIUM → MEDIUM memenuhi `_strong_pick` → evidence gate dilewati.
Cap yang dimaksudkan sebagai pengaman justru membuka gerbang.

### C5 — Korupsi identitas match (data yang dimodelkan bukan pertandingannya)
Dari `match_id` mentah di log:

| Yang di-log | Masalah |
|---|---|
| `UECL\|\|Braga\|\|Austria Wien Women` | tim **wanita** vs tim pria |
| `UEL\|\|KÍ Women\|\|Lech Poznań` | tim **wanita** vs tim pria |
| `UCL\|\|Crvena Zvezda\|\|Hapoel Beer Sheva BC` | **BC = Basketball Club** |
| `UECL\|\|Atalanta\|\|Hapoel Tel Aviv BC` | **BC = Basketball Club** |
| `Belgian Pro League\|\|Leuven Bears\|\|Club NXT` | **Leuven Bears = klub basket** |
| `UECL\|\|Hearts of Oak\|\|Rapid Wien II` | klub **Ghana** vs **tim cadangan** |
| `EPL\|\|Van\|\|Syunik` | liga **Armenia** dilabeli EPL |
| `Serie A\|\|Internacional\|\|Remo` | **Brasil** dilabeli Serie A Italia |
| `UCL\|\|Singapore\|\|Thailand`, `UCL\|\|Malaysia\|\|Vietnam` | **ASEAN Championship** dilabeli UCL |
| `Bundesliga\|\|SV Ried\|\|Grazer AK` | **2. Liga Austria** dilabeli Bundesliga |
| `Primeira Liga\|\|DVO Sittard\|\|Cambuur` | tim **Belanda** di liga Portugal |
| `LaLiga\|\|Alavés Gloriosas\|\|Getafe` | **Alavés Gloriosas = tim wanita** |
| `LaLiga\|\|Las Palmas\|\|Albacete` | **Segunda** dilabeli LaLiga |

Plus duplikasi entitas yang memecah riwayat satu fixture jadi dua:
`Lyon\|\|Sparta Prague` vs `Lyon (Fra)\|\|Sparta Prague (Cze)`;
`Espanyol\|\|Levante` vs `RCD Espanyol de Barcelona\|\|Levante UD`;
`Hearts of Midlothian FC\|\|SL Benfica` vs `Hearts (Sco)\|\|Benfica (Por)`;
`UEL\|\|Fenerbahçe\|\|Lyon` vs `UCL\|\|Fenerbahçe\|\|Olympique Lyonnais` (dua-duanya di-settle).
Seluruh batch 20 Agustus **di-settle dua kali** (07:01:1x dan 07:02:5x, skor identik).

Dan `UCL||Malaysia||Vietnam` di-settle dengan `closing_odds {home: 83.0, away: 1.06}` —
harga yang tidak mungkin, artinya market mapping juga tertukar.

Ini menjelaskan L2/L3/L5: Elo dan form yang masuk ke λ bukan milik tim yang bertanding.

### C6 — CLV tidak terukur pada ~60% sampel
Dari 100 baris `settle`, sekitar **60 punya `closing_odds: null`**. CLV adalah satu-satunya
metrik yang membedakan skill dari variance, dan pada mayoritas match ia tidak terekam.
Ditambah `edge_benchmark: "soft_consensus"` — edge diukur melawan konsensus bandar lunak,
bukan Pinnacle/Betfair.

---

## 5. Bukti statistik: repo ini sudah membuktikan tesisnya sendiri

`reports/signal_audit_2026_08_12.md` — walk-forward EPL 2022-26, **1.520 match, closing
Pinnacle nyata**:

| \|deviasi model−closing\| | n | Brier gap (model−closing) | ROI @closing |
|---|---|---|---|
| 0–2pp | 182 | **−0.0016** (model *lebih baik*) | — |
| 2–5pp | 438 | +0.0018 | −3.3% |
| 5–10pp | 490 | +0.0188 | −8.9% |
| >10pp | 410 | **+0.0364** (model jauh lebih buruk) | +6.2% (artefak) |

> *"The model is worst exactly where it diverges most. Divergence is noise, not information."*

Totals O/U 2.5: model Brier 0.2593 vs closing 0.2388; would-be ROI **−8.8%** (tanpa xG),
**−4.9%** (dengan xG). **Tidak ada signal pada totals.**

Kelly: `g < 0` di **semua** konfigurasi; simulasi quarter-Kelly menghancurkan 85–100% bankroll
di puncak drawdown.

Kebutuhan sampel: edge 1pp → **61.864** taruhan; 2pp → **15.466**; 5pp → **2.475**.
Sampel live kita: **25**.

**Sampel live 21 Agustus dan backtest n=1520 menceritakan hal yang persis sama.** Ini bukan
kebetulan — ini mekanisme yang sama muncul dua kali.

---

## 6. PERINGATAN: gate baru yang sudah aktif adalah overfit

`disagreement_gate` sudah `enabled: true` di config, dengan threshold yang di-tuning
**hanya pada set kalah 21 Agustus** (komentar kodenya mengakui ini: *"tuned ONLY on the
2026-08-21 losing set"*). `replay_current.txt` mengklaim **+51.3% ROI**.

Klaim itu tidak transferable. Uji out-of-sample ke 20 Agustus — hari yang tidak dipakai tuning:

| Pick 20 Agu | Deviasi model−pasar | Gate 20pp | Hasil nyata |
|---|---|---|---|
| OFI Crete `Over 2.5` @2.15 | **+26.5pp** | **DIVETO** | **WIN +1.15** |
| Gent `Under 2.5` @1.74 | **+21.1pp** | **DIVETO** | **WIN +0.74** |
| SV Ried `Over 2.5` (21 Agu) | +28.5pp | DIVETO | LOSE −1.00 |

**Net efek gate >20pp pada sampel gabungan: −0.89u.** Ia membunuh dua pemenang untuk
menyelamatkan satu pecundang. Bucket >20pp secara keseluruhan **2W/1L**.

Bucket deviasi pada 11 pick Totals (20+21 Agu):

| Bucket deviasi | n | W–L | PnL | ROI |
|---|---|---|---|---|
| **≤ 8pp** | 4 | **4–0** | **+2.09u** | **+52%** |
| +10…13pp | 4 | 1–3 | −2.30u | −57% |
| > 20pp | 3 | 2–1 | +0.89u | +30% |

Pelajarannya **bukan** "veto outlier". Pelajarannya: **hanya bucket agreement (≤8pp) yang
konsisten menang**, dan itu cocok dengan bucket 0–2pp di backtest n=1520 yang jadi
satu-satunya tempat model mengalahkan closing line. Perbaikan yang benar adalah
**syarat positif (wajib agreement)**, bukan veto ekstrem.

---

## 7. Aturan keputusan BEST PICK yang benar

Prinsip yang harus menggantikan filosofi sekarang:

> **Deviasi model dari pasar bukan sumber edge — itu alarm error. Edge datang dari HARGA
> (price advantage vs closing line), bukan dari OPINI (model vs consensus).**

### Gate wajib (semua harus lulus, else NO BET)

| # | Gate | Ambang | Dasar |
|---|---|---|---|
| G1 | **Hormati layer keputusan** | `decision_type ∈ {STRONG, GOOD, LEAN}`. `NO BET`/`NO CLEAR DECISION`/`MARKET PRIOR` → **veto**, bukan cap MEDIUM | C3: 11/11 pick 21 Agu dipublikasi meski ditolak mesin |
| G2 | **Agreement, bukan deviasi** | \|p_model − p_market_margin_free\| **≤ 8pp**. Di luar itu → NO BET, apa pun skornya | §6: bucket ≤8pp 4–0; backtest 0–2pp satu-satunya bucket positif |
| G3 | **Konsistensi internal** | arah λ (λ_h vs λ_a) **wajib sepakat** dengan sisi favorit 1X2. Kontradiksi → veto seluruh kartu | L4, L7: kontradiksi λ-vs-1X2 |
| G4 | **Sanity λ** | λ_total di luar `[1.6, 3.6]` → veto. λ_total > 3.6 tanpa dukungan pasar = λ rusak | L2: λ 4.09 pada form L-L-D-W-L |
| G5 | **Elo terverifikasi** | `elo_seeded=True` **dan** Elo dalam `[1300, 2100]` **dan** tidak ada dua tim berbagi Elo identik di run yang sama | L5: Sociedad 2361 = Arsenal 2361 |
| G6 | **Identitas match bersih** | tolak jika nama memuat `Women`, ` BC`, ` II`, `Reserve`, `U19/U21`, atau liga↔tim tidak konsisten di `leagues.json` | C5 |
| G7 | **Harga wajib ada** | `market_odds` non-null, `bookmakers_count ≥ 3` | Braga: pick tanpa harga |
| G8 | **Pick = ranking[0]** | pick yang dipublikasi **harus** identik dengan ranking teratas; override apa pun harus dicatat + diberi alasan | L1: pick ≠ ranking[0]; Nijmegen: override stability |
| G9 | **EV pada harga terbaik** | `EV > +3%` dihitung dari **best available odds**, bukan median | Gap #1/#3 `SHARP_BETTOR_REVIEW.md` |

### Yang harus berhenti dilakukan

1. **Berhenti memakai skor komposit untuk memilih pick.** Spearman 0.0015 (C1). Skor boleh
   dipakai untuk *menyortir tampilan*, tidak untuk *memutuskan taruhan*. Keputusan =
   gate G1–G9, biner.
2. **Balik tanda komponen `market`.** Sekarang ia menghadiahi deviasi. Ubah jadi menghukum:
   nilai maksimum pada deviasi ≈ 0, turun ke 0 saat deviasi > 8pp.
3. **Berhenti mem-publish market Totals sebagai default.** 7 dari 9 pick 21 Agu adalah Totals;
   backtest n=1520 menunjukkan ROI totals −8.8%/−4.9% dan Brier lebih buruk dari closing.
   Totals adalah market terlemah model ini, tapi paling sering dipilih.
4. **Berhenti menampilkan "BEST PICK" saat mesin bilang NO BET.** Ganti label jadi
   `NO BET — <alasan>`. Nol pick lebih baik daripada pick MEDIUM yang ditolak mesinnya sendiri.
5. **Jangan percaya +51.3% dari `replay_current.txt`.** In-sample, tuned pada hari yang sama
   yang diukur (§6).

### Yang harus mulai diukur (tanpa ini, evaluasi berikutnya sama buta)

| Instrumen | Kondisi sekarang | Target |
|---|---|---|
| `closing_odds` pada settle | **null di ~60%** | 100% — CLV satu-satunya metrik skill |
| Referensi sharp line | soft consensus | Pinnacle/Betfair closing |
| Line shopping | median odds | best available + nama bandar |
| Dedup entitas | duplikat + double-settle | satu `match_id` kanonik per fixture |
| Grading | phantom skor 2–3 pada Erzurumspor | satu sumber skor, di-cross-check |

---

## 8. Verdict

Model ini **well-calibrated tapi market-matching** — sedikit lebih buruk dari closing line
(Brier gap +0.013…+0.016). Itu bukan kegagalan; itu titik konvergensi normal untuk
Elo+Poisson tanpa data proprietary.

Kesalahannya adalah **lapisan keputusan di atas model itu**, yang mengonversi
"model tidak tahu apa-apa lebih dari pasar" menjadi "BEST PICK MEDIUM" — dan yang secara
sistematis memilih match di mana model paling salah karena mengira deviasi = value.

Selama G1–G9 belum dipasang dan CLV belum terukur, **jumlah BEST PICK yang benar per hari
adalah nol sampai beberapa, bukan sembilan.** Volume adalah musuh di sini. Pada 21 Agustus,
jawaban yang benar untuk 7 dari 9 kartu adalah tidak bertaruh.
