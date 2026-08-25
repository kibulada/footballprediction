# Hermes-Football — Project Context (untuk AI assistant)

> Baca file ini DULU sebelum eksplorasi kode. Terakhir diupdate: 2026-08-24.

## Apa ini
Discord bot prediksi match bola. Alur: `bot.py` (Discord layer) → subprocess
`python -m agents.football.runner <mode>` → pipeline analisa → snapshot JSONL +
card Discord. Venv: `.venv\Scripts\python.exe`. Bukan git repo!

## Peta file kunci
| File | Peran |
|---|---|
| `bot.py` (~99KB) | Discord commands (`!analisa`, `!best`, dll), spawn runner subprocess (timeout 380s), Tor/proxy mgmt |
| `agents/football/runner.py` | CLI entry. Watchdog deadline 340s (`HERMES_RUNNER_DEADLINE`). Log level via env `HERMES_LOG_LEVEL=INFO` (per-phase timings ke stderr) |
| `agents/football/analyse.py` (~3.3K lines) | Pipeline analisa utama. Fase berurutan; budget 300s (`config/football.json analyse.budget_seconds`) |
| `agents/football/multi_source.py` | Orkestrasi sumber data: form, H2H, standings, xG history |
| `agents/football/flashscore.py` | Scraper utama. **satu Chrome (seleniumbase UC headless2) per client**, semua render serialize via `_browser_lock`. Settle sleep 3s/render |
| `agents/football/nowgoal.py` | HTTP client 4 mirror (sering down). Punya circuit breaker |
| `agents/football/signal_engine.py` | Ranking pick: score 0-100 + komponen (model .35/statistical .10/market .15/movement .15/data_quality .05/market_intel .15) + **veto layer** (`vetoed`, `veto_reasons`) |
| `agents/football/prediction_log.py` | Snapshot JSONL `cache/football/predictions.jsonl`. `append_snapshot(skip=...)` |
| `agents/football/elo.py` | ELO resolver: exact→alias(teams.json)→token fuzzy. `_EXTRA_ALIASES` built-in |
| `cache/football/elo.json` | Seed ratings dari football-data.co.uk (`--seed-elo`). Backup lama: `elo.json.backup-fdcouk` |

## Optimasi 2026-08-23 (JANGAN di-revert)
1. **Dual-lane browser** (`multi_source.py` `flashscore_lanes=2`, `fc_secondary`): rantai xG home‖away paralel. Lane B fallback otomatis ke primary kalau mati
2. **Overlap HTTP** (`analyse.py`): task `_detail_task` (nowgoal detail) & `_ctx_task` (GraphQL injuries) dimulai SEBELUM render browser, dikonsumsi di posisi semantik asli
3. **Circuit breaker NowGoal**: 2x gagal transport semua mirror → tidur 90s
4. **Negative cache xG**: key `{stats_key}_miss` TTL 1800s untuk render kosong
5. **xG pakai resolved ref**: `flashscore_ref` dari `meta._flashscore_match` (bukan lotre `_suggest_team`)
6. **Image-blocking Chrome** (`flashscore.py _apply_resource_blocking`, CDP setBlockedURLs 17 pola): render −51%, data identik
7. **Anti-flap P1**: `append_snapshot(skip=is_market_prior...)` — MARKET PRIOR rows tidak dilog
8. **NowGoal fast-fail connect** (`nowgoal.py _CONNECT_TIMEOUT=4s`): hanya fase connect yang dipangkas, read/write/pool tetap timeout penuh — mirror mati tidak lagi membakar 15s/mirror/putaran (4 mirror down ≈ 1 menit → ≤16s)
9. **Breaker backoff eksponensial**: cooldown 90s→180s→360s (cap) per open berturut-turut; sukses apa pun reset penuh
10. **Identity lock cheap-first scan** (`identity_lock_check`): age-filter + pre-filter league/tanggal pakai string ops dulu, kanonisasi teams.json (~3ms/baris) hanya untuk baris yang lolos filter → 2.63s → 0.25s per run, verdict identik

Hasil: analisa tipikal **275s → ~148s (−46%)**. Metode verifikasi: run baseline vs
optimized kondisi cache sama, diff rekursif field-by-field snapshot JSONL.

## Struktur data penting
- Snapshot JSONL fields: event, ts, home/away, prob_1x2, best_pick, signal_engine_pick,
  signal_engine_ranking[], decision_type, features (elo/lambda/completeness), sources[]
- **3 lapisan "pick" yang BEDA**: `best_pick` (decision layer, pre-veto),
  `signal_engine_pick` (null jika semua veto), ranking top-score. JANGAN dicampur!
- Cache TTL kunci: team_form 1h, h2h/xg/stats 24h, odds_recent 900s, livescore_date 15m

## Masalah terbuka + blueprint
1. **Migrasi ELO ke elofootball.com**: DITUNDA — situs masih unreachable
   (500/timeout; retry `--dry-run` 2026-08-23 tetap gagal). Retry:
   `python -m agents.football.elo_scraper` → sukses ditandai field `"source"`
   di elo.json

## Selesai 2026-08-23 (eksekusi blueprint)
1. **Identity lock (Fase 2 anti-flap)**: `identity_lock_check()` di
   `prediction_log.py`. Dipanggil analyse.py tepat sebelum `append_snapshot`;
   konflik pasangan tim kanonik vs riwayat log MENAHAN write (+ warning log).
   Dua deteksi: same-id contradiction (bandingkan nama/entities canonical_id
   via `_match_id_hits`) dan opponent-flip (liga+tanggal sama, satu sisi sama,
   lawan beda — kasus Forest Leeds→Man Utd; Troyes "PSG"←Paris FC). Hanya
   snapshot ≤7 hari (`IDENTITY_LOCK_MAX_AGE_DAYS`) yang dipertimbangkan.
   Fail-open: error check tidak pernah memblokir logging.
2. **BEST PICK renderer** (keputusan user): helper `_display_best_pick(se)`
   di format.py → `(pick, risk_reason)`. Semua kandidat diveto pick_gates →
   rank #1 tetap ditampilkan + label `⚠️ HIGH RISK — gagal gerbang:
   {veto_reasons[0]}` (di `_best_pick_block` & accordion
   `_summary_best_pick_value`). NO BET karena gagal floor (score/edge/G1)
   TETAP NO BET polos — hanya kasu semua-diveto yang berubah.
3. **Gerbang `!best`**: `best_match._passes_best_gate()` — hanya kandidat
   `pick_specific_confidence.label >= MEDIUM` + `pick_status == VALID`.
   Flag `models.decision.best_gate_enabled` (default ON). Kosong lolos →
   error payload eksplisit "{n} match dianalisa, tidak ada yang lolos".
   Shortlist kini membawa `confidence_tier` + `pick_status`, ditampilkan
   di ranking card. Tests mekanika ranking mematikan gate via cfg.

Test baru: identity lock (test_prediction_log.py), HIGH RISK renderer
(test_phase5_presentation.py), gate !best (test_best_commands.py).

## Selesai 2026-08-24 (plan v3 — perbaikan output BEST PICK)
Laporan: `reports/bestpick_evaluasi_elche-barca_2026-08-24.md` (F1–F14 + buku
67 pick ter-settle ROI −7.9%). Eksekusi paket inti A–D, suite **1465 passed**:
1. **F1 Elo-anchor λ** (`models.apply_elo_anchor`): λ final Totals/BTTS
   ditarik ke share Elo saat gap rating ≥150 (penuh ≥400). Akar masalah
   Elche v Barcelona: feature λ bilang Elche > Barca vs gap 715 → BTTS Yes
   terpilih, FT 0-5. Audit: `model_probs.elo_anchor_t`.
2. **F2 kalibrasi total** (`models.calibrate_total_to_market`): λ final
   ditarik ke fair O/U market (devig, setengah gap) — menutup split struktural
   1X2-vs-Totals; kasus Goztepe λ_total 4.11→3.48.
3. **F4-lite**: komponen `market` kini MENGHARGAI KESESUAIAN (maks di 0pp,
   0 di tepi band G2) via `market_component_reward_agreement` (default ON) —
   rekomendasi #2 postmortem 2026-08-22 yang tertunda.
4. **F14 kandidat 1X2** (`enable_1x2_signals`, default ON): Home/Draw/Away Win
   jadi kandidat BEST PICK penuh (implied margin-free 3-outcome); settlement
   1X2 ditambahkan. "Away Win" kini mungkin keluar sebagai BEST PICK.
5. **F11 renderer**: `_display_best_pick` TIDAK lagi menampilkan rank#1
   HIGH-RISK saat semua kandidat diveto → NO BET polos + alasan gate; snapshot
   tidak lagi menyimpan decision-layer pick saat SE sudah memutuskan (kebocoran
   Goztepe). Kebijakan lama 2026-08-23 di item 2 atas = DIGANTI.

Replay 4 fixture (input dari log): Elche BTTS Yes→**Over 2.5 WIN**;
Atalanta BTTS No→**Home Win WIN**; Goztepe bocoran Over→NO BET (hindari loss);
CFR Cluj tetap varian (label VERY HIGH→MEDIUM lebih jujur).
Config baru: `models.poisson.elo_anchor`, `models.poisson.market_total_calibration`,
`models.signal_engine.{enable_1x2_signals,market_component_reward_agreement}`.
Test baru: test_plan_v3_output_fixes.py (13 tes). Catatan: `respect_model_decision`
TETAP false (keputusan operator 2026-08-24).

## Selesai 2026-08-24 (blind replay 29 match + settle + P0 entity)
Laporan: `reports/bestpick_v3_blind_replay_2026-08-24.md`. Buku blind v3
(14 stake) **+32.2%** vs OLD (13 stake) **−14.2%**, FT dari LiveScore feed resmi.
1. **F8/F3 entity FIX**: `resolve_team_alias` dapat *significant-token
   containment guard* (`_NAME_STOPWORDS`) — nama kanon yang token signifikannya
   terkandung penuh di query MENANG atas alias generik. Repro yang dibetulkan:
   "Barcelona"→FC Barcelona (dulu Espanyol!), "Club Atletico de Madrid"→Atlético
   Madrid (dulu Real Madrid CF!). Alias eksplisit BARCELONA + CLUB ATLETICO DE
   MADRID ditambah. Efek samping disengaja: varian prefix ("FC Arsenal") kini
   menyatu ke identitas kanonik — pemecah identitas match_id mati. Self-audit
   323/323 kanon resolve ke dirinya.
2. NO BET set kemarin: 2 loss terhindar vs 3 winner ke-skip (PSV/Le Havre/
   Torino) — biaya disiplin NYATA, pantau mingguan. G4 ceiling 3.6 mem-blok
   kartu penuh Man City (λ 3.79) — kandidat league/market-aware ceiling,
   WAJIB replay dulu sebelum diubah.

## Selesai 2026-08-24 (identity firewall — anti "analisa tim salah")
Tujuan: match yang dianalisa DIJAMIN pasangan yang diminta (A vs B), bukan
A vs C. Suite **1484 passed** (1465 lama + 19 baru, 0 regresi):
1. **Modul baru `identity_gate.py`**: `canonical_side()` (resolusi kanonik
   STRICT: alias league-scoped → exact abbr-key ke semua klub terdaftar;
   nama tak terdecide = None, TIDAK PERNAH menebak), `check_pair_identity()`
   (bandingkan pasangan query vs fixture-terdeteksi vs resolved-teams;
   refuse hanya saat DUA sisi yakin beda klub; side-swap terkonfirmasi =
   warn), `preflight_history_lock()`, `refusal_payload()` (shape error
   standar — format.py/bot render verbatim).
2. **3 gate pre-flight di `find_specific_match`** (config
   `identity_firewall.{enabled,refuse_divergence,refuse_history_lock}`,
   default ON): G-A `preflight_fixture` = query vs detected fixture SEBELUM
   render browser (hemat 20–40s di kasus F3); G-B `post_resolve` = query vs
   resolved teams sebelum fase mahal lain; G-C `history_lock` =
   identity_lock_check dipanggil ~2 detik setelah _mid_canon jadi — resolver
   flip kini MENGHENTIKAN analisa, bukan menahan write setelah ~250s
   (end-of-run hold TETAP ada sebagai lapis kedua). Semua gate fail-open.
3. **Provenance**: payload sukses membawa `identity.entities` + checks —
   audit bisa lihat klub mana yang dianalisa tanpa re-derive.
4. **L0 roster**: `roster_builder.py` (`python -m agents.football.roster_builder`)
   merge teams.json × entity_registry → `cache/football/rosters.json`
   (323 klub kanonik; ID flashscore/football_data/livescore/thesportsdb
   menempel otomatis dari observasi live). Zero network; refresh per musim.
5. Regression tests (test_identity_gate.py): F3 Atlético→Real Madrid REFUSE,
   P0 Barcelona→Espanyol REFUSE, Forest-Leeds→Man Utd flip LOCKED, ejaan
   provider sama klub PASS, swap WARN, dyn-liga/log rusak FAIL-OPEN.

## Selesai 2026-08-24 (nowgoal mirror P1 + trend primer P2)
Probe live match Fulham-Chelsea (oddscomp/3003858): endpoint `t=20` Trends
hidup, wajib header Referer (tanpa itu `{"code":1002}`). Bet365 saja = 630
titik harga 1X2 ber-timestamp; Pinnacle/Betfair hanya serve kolom op.
Suite **1489 passed**:
1. **P1 mirror**: `DEFAULT_BASE_URL` → `https://live10.nowgoal26.com/`
   (lama nowgoal.net = 404 hari ini); config mirrors diurutkan alive-dulu
   (.net/6/7 dipertahankan di ekor — mirror sering bangkit). Probe CLI baru:
   `python -m agents.football.nowgoal --probe-mirrors` — verdict ALIVE hanya
   kalau path `/ajax/soccerajax` jawab JSON (homepage 200 HTML tetap DEAD);
   injectable `_fetch` untuk test. Fungsi: `probe_mirrors()`.
2. **P2 trend primer**: blok trend di analyse.py tidak lagi digate
   `len(_snaps) < 2` — fetch SELALU saat budget sehat, cache per-match
   (`ng_trend_{match_id}` TTL 1800s), merge dedup dengan series poll.
   `trend_to_snapshots` kini membawa atribusi `bookmaker`+`bookmaker_cid`
   per baris — bahan mentah deteksi sharp-money (steam sync / sharp-vs-soft)
   yang belum diimplement (butuh validasi replay dulu).

## Selesai 2026-08-24 (re-prioritas odds: NowGoal PRIMARY, OddsPapi validator)
Dasar: probe live + bedah konsumen payload (BTTS hanya ada di shape
OddsPapi/TheOddsAPI; nowgoal = euro/ou/ah saja). Suite **1494 passed**:
1. **Pembalikan prioritas di `find_specific_match`**: oddspapi DIDEMOSI
   (payload disimpan utk cross-check), nowgoal SELALU di-fetch dan jadi
   primary; re-promosi oddspapi hanya kalau nowgoal kosong; TheOddsAPI tetap
   terakhir. Flag provenance (`oddspapi_used`/`nowgoal_used`, labels sources)
   kini berarti "API dipanggil & memberi payload" — keduanya true saat
   validator jalan.
2. **`_merge_missing_btts()`** (pure, test_odds_priority.py): saat primary
   NowGoal, harga BTTS Yes/No diisi dari totals sumber sekunder — tanpa ini
   G7 require_price memveto SEMUA pick BTTS. Hanya label MISSING yang diisi;
   primary tetap single-writer.
3. **Quota OddsPapi hemat di poll**: `auto_odds_poll.sources` berurutan dan
   berhenti di sumber pertama sah (runner sudah begitu) → oddspapi di poll
   hanya fallback saat nowgoal mati; catatan arsitektur ada di
   `_note_sources` config. Panggilan oddspapi per analisa tetap 2 call
   (validator + BTTS) — disengaja.

## Pelajaran penting (biar gak ulang)
- **Jangan spawn banyak Chrome paralel/bertumpuk** — pernah bikin device user freeze.
  Maksimal 2 (dual-lane), pastikan close() jalan, cek zombie: `Get-Process chrome`
  filter CommandLine `--headless`
- Runner yang kena deadline (`os._exit`) ATAU dibunuh timeout dari luar =
  Chrome bocor. Selalu audit proses setelah run gagal
- Benchmark A/B wajib: bot production STOP dulu, kondisi cache disetarakan
  (hapus cache spesifik match), catat TTL yang expired sebagai handicap explisit
- `Start-Process -ArgumentList` motong spasi: quote manual '"Go Ahead Eagles"'
- User mengubah tampilan card tanpa mencatat — selalu konfirmasi field mana yang
  jadi sumber tampilan sebelum membandingkan dengan log

## Command cepat
```
.venv\Scripts\python.exe -m pytest tests\test_analyse_autodetect.py tests\test_prediction_log.py tests\test_elo.py tests\test_nowgoal.py tests\test_flashscore.py -q   # smoke suite
.venv\Scripts\python.exe -m agents.football.runner analyse --league eredivisie --home "X" --away "Y"   # analisa langsung
HERMES_LOG_LEVEL=INFO  # lihat "analyse phases:" per-fase di stderr
```
