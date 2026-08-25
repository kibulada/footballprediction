# PHASE 0 — AUDIT REPORT (Master-Prompt: Safe Improvement / No-Regression)

Tanggal audit: 2026-08-12
Status: **AUDIT ONLY — tidak ada satu pun file kode diubah.** Dokumen ini
hanya mencatat kondisi terverifikasi.

Reference lain yang masih berlaku:
- `audit/current_architecture.md` (arsitektur + formula inti)
- `reports/baseline_freeze.md` + `reports/baseline_freeze_epl.json` (baseline)
- `reports/decision_engine_fase1.md` + `reports/decision_validation_final.json` (decision layer)
- `reports/before_after.md` (transformasi betting-grade)
- `baseline/` (12 modul snapshot immutable)

---

## 1. Alur Pipeline Lengkap (terverifikasi dari kode)

```
Discord command ("analisa x vs y")
  → bot.py            (subprocess runner, deadline 90s, parse RJSON dari stderr)
  → runner.py         (mode analyse; _arm_deadline 85s; os._exit contract)
  → league_resolver   (league key → odds key + sofascore/flashscore meta)
  → multi_source.py   (provider chain: flashscore primary → football-data →
                       thesportsdb; sofascore last-resort w/ 15s cap)
      • search_teams_pair, upcoming fixture, form, H2H, xG history
  → odds_fetcher.py   (The Odds API: h2h/totals/btts, retry market-set, quota)
  → scorer.py         (consensus=median, best odds, outlier, signal 0-100)
  → predictor.py      (derive_picks: normalize_odds → solve_lambdas →
                       score matrix → probs 1X2/O-U 1.5-3.5/BTTS → top3 by
                       model_prob; EV per pick)
  → models.py         (run_prediction_engine: Elo → Poisson(features+xG) →
                       Ensemble → Calibrator.apply → SignalScorer)
  → calibration.py    (Calibrator log-odds linear; SignalScorer v2 +
                       decisiveness + completeness caps)
  → decision.py       (Decision Score 30/20/15/15/10/5/5; STRONG/GOOD/LEAN/
                       NO CLEAR DECISION/NO BET; extreme-edge protection;
                       best_prob_only)
  → prediction_log.py (snapshot append-only + similar-signal bucket)
  → format.py         (Discord output; FINAL DECISION section)
```

## 2. Model Inputs (pre-match only, verified)

| Fitur | Sumber | Pre-match? | Ada di backtest EPL? |
|---|---|---|---|
| consensus odds 1X2/totals/BTTS | The Odds API / football-data.co.uk | ✅ | ✅ (home/draw/away + O/U 2.5) |
| form sequence + gf/ga avg | flashscore / fd / thesportsdb | ✅ | ✅ (dari skor historis) |
| raw recent scorelines (time-decay) | flashscore | ✅ | ✅ (deque 5, xi=0.9) |
| xG for/against avg | sofascore history / flashscore | ✅ | ❌ (FBref/CSV tidak punya) |
| H2H | flashscore / fd | ✅ | ❌ |
| rest days | fixture dates | ✅ | ✅ (dari tanggal) |
| Elo rating (seeded 255 tim) | football-data.co.uk CSV | ✅ | ✅ |
| odds-derived λ (derive_picks) | market | ✅ (cermin pasar, `independent=False`) | n/a |

## 3. Calibration Method

- `p' = sigmoid(a + b·logit(p))`, fit IRLS (Newton-Raphson) atas pasangan
  (prob, outcome) agregat **in-sample** EPL 2022-2026.
- Verified live: `a=0.0082, b=1.0129, samples=4560, raw ECE 0.011 → calibrated
  ECE 0.0104` (dari `cache/football/calibration.json`).
- **Dokumentasi kejujuran**: calibrated ECE di backtest dihitung in-sample
  (bukan out-of-sample). Params EPL diterapkan ke semua liga live.
- `Calibrator.apply` hanya aktif bila `samples >= min_samples (200)`.
- `SignalScorer` confidence: 0.20 completeness + 0.30 agreement + 0.35
  calibration + 0.15 decisiveness; di-cap oleh `completeness_level`
  (LOW≤0.49, MEDIUM≤0.69, HIGH=1.0).

## 4. Decision Logic (terverifikasi — Fase 1 sudah live)

- **MOST LIKELY**: 1X2 side dengan calibrated model prob tertinggi (independen).
- **BEST DECISION**: Decision Score per candidate (1X2 + totals 2.5/3.5 + BTTS).
- Tipe: STRONG/GOOD/LEAN/NO CLEAR DECISION/NO BET (NO BET valid, bukan error).
- `best_prob_only=true` (tervalidasi walk-forward): value hanya untuk favorit
  per market; long-shot edge besar → NO BET (terbukti noise, eksperimen
  ditolak: −7.8% ROI).
- Extreme-edge (≥20pp) → cap value 0.30 + cap tipe LEAN; warning ≥10pp.
- Config shipped: `min_edge_pp=3.0`, weights 30/20/15/15/10/5/5.
- **Verified walk-forward**: NEW ≈ OLD (ROI −2.2% vs −1.9%, delta −0.34pp =
  statistically inconclusive). STRONG +3.1% ROI (n=28), LEAN +29.5% (n=91),
  GOOD −9.3% (n=434) — bucket kecil, tidak diklaim.

## 5. Backtest Methodology (terverifikasi, re-run hari ini)

- **Walk-forward kronologis satu pass** (kickoff order), Elo/form state
  menyebrang musim. Tidak ada random split.
- `validate.py --fixtures cache/football/epl_fixtures_2022_2026.json` —
  **BASELINE REPRODUCIBLE 100%** (angka identik baseline_freeze.md):

| Model | n | LogLoss | Brier | ECE | Hit% | ROI (flat, edge≥2%) | bets |
|---|---|---|---|---|---|---|---|
| baseline | 1520 | 1.0971 | 0.6495 | 0.0196 | 44.4% | −7.5% | 805 |
| elo | 1520 | 0.9914 | 0.5921 | 0.0205 | 52.3% | −2.7% | 882 |
| poisson | 1505 | 1.0341 | 0.6213 | 0.0157 | 49.2% | −0.5% | 765 |
| dc | 1505 | 1.0363 | 0.6223 | 0.0174 | 48.9% | −0.1% | 740 |
| **ensemble** | 1520 | **0.9886** | **0.5901** | **0.0110** | **53.1%** | **−1.9%** | 623 |
| market | 1520 | 0.9652 | 0.5739 | 0.0084 | 54.5% | — | 0 |

- Per-season consistency: semua model beat baseline 4/4 musim; **0/4 beat
  market** — market tetap benchmark terkalahkan (diklaim apa adanya).
- `decision_validation.py` — decision-layer walk-forward (OLD vs NEW) sudah
  ada dan menyimpan hasil JSON (`reports/decision_validation_final.json`).
- `backtest.py` — dukungan `--seed-elo`, eksperimen weight/rest-days.

## 6. Odds Methodology

- **Live**: The Odds API — median consensus (robust), best odds, outlier vs
  consensus; margin dihilangkan: 1X2 via `normalize_odds`, pasangan O/U & BTTS
  via `fair_pair_implied` (kedua sisi). Edge selalu margin-free.
- **Backtest**: football-data.co.uk (Pinnacle > Avg > Bet365), margin-free
  implied sebagai market baseline; ROI flat-stake hanya dengan odds historis
  nyata — tidak pernah fabrikasi.
- CLV: dihitung dari closing odds saat `settle` menyertainya; tidak dipakai
  sebagai fitur.

## 7. Anti-Leakage — jalur yang sudah ditutup (terverifikasi)

1. `MatchContext` pre-match-only by construction (result/xG live tidak masuk).
2. `elo.update()` hanya dipanggil SETELAH kickoff (backtest: setelah replay).
3. Live `analyse` hanya fetch event stats bila `fixture.status == "notstarted"`
   (guard `fixture_is_prematch`) — statistik match sendiri tidak bocor.
4. `exclude_event_id` pada history stats → match yang dianalisa tidak ikut.
5. Kalibrasi fit hanya dari training data; `Calibrator.apply` guard samples.
6. Similar-signal hanya dari snapshot yang sudah settle (outcome nyata).
7. Decision engine value hanya untuk model independen (`independent=True`);
   odds-derived picks tidak membawa market_value (anti double-count S25).
8. `input_hash` snapshot → reproducibility; snapshot append-only immutable.
9. Threshold decision (min_edge_pp, best_prob_only) dipilih via walk-forward
   dan ditulis apa adanya (delta dalam noise), bukan diklaim menang.

## 8. Sisa Risiko Leakage / Keterbatasan Data (jujur)

1. **Same-day ordering**: validate mengurutkan hanya per tanggal; kickoff time
   tidak tersedia di FBref/CSV → match same-day bisa diproses dalam urutan
   arbitrer (caveat didokumentasikan di `note` validate.py). Risiko kecil
   karena form/Elo hanya pakai match SEBELUM-nya di urutan file.
2. **xG tidak ada di backtest**: live Poisson blend xG (weight 0.65) tidak
   pernah dievaluasi out-of-sample. Fitur xG masuk production hanya karena
   pre-match, bukan karena bukti walk-forward.
3. **Kalibrasi EPL → semua liga**: params EPL diterapkan lintas liga tanpa
   bukti per-liga; efeknya kecil tapi bukan bukti.
4. **H2H & totals/BTTS tidak di-backtest**: walk-forward EPL hanya 1X2 +
   O/U 2.5 odds. Decision engine menilai totals 3.5/BTTS live tanpa
   validasi historis per-market (dictatakan di Fase 1).
5. **Data quality bervariasi per provider** (flashscore names vs slug,
   football-data.org vs thesportsdb) — dimitigasi toleransi nama
   (`_teams_match`, `_assign_slug_roles`, Elo resolve), tapi mismatch fixture
   identity tetap mungkin → decision engine sudah punya edge-warning/extreme
   guard.
6. **Sample bucket kecil** (STRONG n=28, LEAN n=91) — tidak boleh dipakai
   untuk klaim kinerja.

## 9. Regression Risks (jika menyentuh production)

- `baseline/` (12 modul) = immutable reference; metrik di atas = angka
  pembanding wajib.
- `elo.py` K/HA/seed, `models.py` math, `predictor.py` solve_lambdas,
  `calibration.py` weights, `scorer.py` consensus — jangan diubah tanpa
  walk-forward bukti.
- Discord output contract (format.py) — runner/bot hanya parsing
  `RJSON_START...RJSON_END` dari stderr; payload harus tetap JSON-safe
  (`decision_to_dict`).
- Test suite terakhir: **357 passed** (sebelum audit ini; tidak ada kode
  diubah pada audit).

## 10. Kesimpulan Audit

- Pipeline lengkap, anti-leakage kuat, baseline **terverifikasi ulang hari ini**
  (identik), decision layer sudah transparan & tervalidasi walk-forward.
- **Kandidat perbaikan yang bisa dievaluasi selanjutnya** (BELUM diputuskan,
  butuh bukti out-of-sample — sesuai PHASE 15):
  a) Perbaikan same-day ordering (pakai kickoff time bila tersedia).
  b) xG di backtest (perlu dataset historis xG → baru bisa klaim).
  c) Calibration per-league / out-of-sample ECE reporting.
  d) Ablasi rest-days/features (infra `--rest-days-k` sudah ada).
  e) Validasi totals/BTTS historis untuk decision layer.
- Tidak ada satu pun perubahan dipaksa; setiap kandidat harus mengalahkan
  baseline di atas secara out-of-sample atau **EXISTING MODEL RETAINED**.
