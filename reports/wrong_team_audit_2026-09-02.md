# Audit: data tim yang salah pada analisa (2026-09-02)

## Gejala

Snapshot VPS 31 Agu–1 Sep (semua fixture di-resolve lewat LiveScore) menyimpan form yang bukan milik tim yang dianalisa:

| Match (1 Sep) | Form tersimpan | Hasil nyata tim (oldest→newest, sebelum kickoff) |
|---|---|---|
| Birmingham v **Southampton** | L-D-L-L-D | W 2-0, L 1-2, W 3-1, L 1-4, W 5-1 |
| **Stoke City** v Norwich | L-L-L-W-W | W 2-0, L 1-2, L 1-3, D 1-1, L 1-4 |
| **Portsmouth** v Derby | L-L-L-L-L | L 1-3, L 1-3, W 3-1, L 1-2 |
| Lincoln v Blackburn | L-L-L-W-W / D-W-W-L-L | W, L, L, W, W / W, D, W, L, L |

Form → gf/ga → attack/defence → λ Poisson → probabilitas → BEST PICK, jadi pick untuk match ini dihitung pada tim lain.

## Akar masalah

1. **Pencocokan nama tim berbasis substring** di setiap provider (`analyse._teams_match`, `nowgoal._same_team`, `oddspapi._same_team`, `tie_state._same_team`, `soccerdata_wrapper` `str.contains`). Pada cache feed LiveScore lokal: `teams_match("Southampton", "South Carolina United FC")` = True (185 dari 205 baris "Southampton" adalah klub lain), `"Stoke City"` ↔ `"Basingstoke"`, `"Portsmouth"` ↔ `"Port City FC"`, `"Birmingham City"` ↔ `"Birmingham City U18/U21"`, `"Parma"` ↔ `"Parma U20"`, `"Wolves"` ↔ `"Wollongong Wolves"`. Tidak ada pemeriksaan penanda U18/U21/Women/II/B, tidak ada pemeriksaan negara.
2. **Form LiveScore diambil BY NAME padahal ID terverifikasi sudah ada.** `multi_source._fetch_team_form_uncached` membangun ulang form dari halaman hasil harian dengan pencocokan nama (`_livescore_form`), sementara endpoint per-event `/form-e/{eid}` (di-key dengan ID tim) sudah di-parse (`livescore.parse_form`) tetapi tidak dipakai jalur form.
3. **Orientasi home/away yang fail-open**: `nowgoal._parse_analysis` menganggap baris sebagai "home" bila nama tim tidak cocok dengan KEDUA sisi (gf/ga tertukar); `soccerdata_wrapper` memutuskan sisi dari substring yang sama.
4. **Liga yang salah tidak dipropagasi**: saat kompetisi fixture berbeda dari liga yang diketik, hanya `standings_key` yang dikoreksi; `league_key` lama tetap dipakai untuk filter form G5, alias kanonik, dan pencarian tim (bukti: `entity_registry.json` mencatat Atalanta/Bologna di bawah `t:efl-championship:*`).
5. **Elo saat settle memakai nama tampilan** (`runner._update_elo_from` → `EloModel.update` → `resolve("Lille")` = None → membuat key baru "Lille" 1500, terpisah dari "Lille OSC"), dan tanpa K/home-advantage dari config.
6. Cache: kunci form by-name diberi label provider "flashscore"/"livescore" dengan ID provider lain; halaman feed yang tumpang tindih menghitung match yang sama dua kali.
7. Alias `teams.json` EFL Championship hanya 12 entri untuk 24 klub → canonical id terpotong (`t:efl-championship:lincoln` vs `:lincoln-city`), firewall identitas fail-open untuk klub-klub itu.

## Perbaikan

| # | Perubahan | File |
|---|---|---|
| 1 | Modul `team_identity.py`: satu matcher token-level (`names_match`, `match_side`, `same_fixture`, `has_marker`, `country_matches`): tanpa substring; token noise (FC/FK/RCD/UD…) diabaikan; penanda U-/Women/II/B/Jong harus simetris; qualifier City/United/Town harus sama bila keduanya ada; maksimal 1 token identitas ekstra (`strict=True` → 0). Semua matcher provider mendelegasi ke sini. | `team_identity.py`, `analyse.py`, `nowgoal.py`, `oddspapi.py`, `tie_state.py`, `datasources.py`, `soccerdata_wrapper.py`, `flashscore.py` |
| 2 | Form LiveScore **berdasarkan ID**: `livescore.team_form_by_id(payload, team_id)` memilih blok T1/T2 yang `ID`-nya sama dengan tim yang di-resolve; dipanggil di `_fetch_team_form_uncached` sebelum jalur by-name mana pun (`source = "livescore_event"`). | `livescore.py`, `multi_source.py` |
| 3 | Jalur by-name LiveScore: filter negara liga vs negara baris, identitas strict bila kompetisi bukan liga yang dianalisa, `match_side` (nama yang cocok dua sisi ditolak), dedupe per event id. | `multi_source.py` |
| 4 | NowGoal: baris yang tidak menamai tim tabel (atau menamai dua sisi) dibuang, bukan dianggap home. soccerdata: sisi dari `match_side`/`same_fixture`. | `nowgoal.py`, `soccerdata_wrapper.py` |
| 5 | `league_key` dan `meta["_league_key"]` mengikuti kompetisi fixture saat mismatch terdeteksi. | `analyse.py` |
| 6 | Standings: home & away yang jatuh ke baris yang sama dibuang. Tie state: pertemuan dari kompetisi berbeda bukan leg pertama. H2H window: atribusi token-level, tidak menimpa tally provider dengan 0. OddsPapi: fixture harus cocok pada orientasi yang sama & tidak ambigu, dicari pada tanggal kickoff, cache di-key fixture id. | `analyse.py`, `tie_state.py`, `livescore.py`, `oddspapi.py` |
| 7 | Settle Elo: key kanonik dulu (`resolve_first((canonical, display))`) dengan K/home-advantage config. | `runner.py` |
| 8 | Flashscore: containment squashed butuh overlap ≥ 60% & ≥ 4 huruf; baris ber-penanda ditolak; suggest memilih exact lalu slug terpendek (bukan terpanjang); form by-name di-cache dengan ID flashscore hasil resolve dan diverifikasi namanya. | `flashscore.py`, `multi_source.py` |
| 9 | `fetch_h2h` fallback LiveScore: `NameError` (`home`/`away` tidak terdefinisi) diperbaiki. | `multi_source.py` |
| 10 | Data: 86 alias EFL Championship ditambah (canonical sama dengan liga lain bila klub sudah ada); 6 entri registry beracun dihapus (Hull→Man City, Leeds→Man Utd, Barcelona→Espanyol, Atalanta/Bologna/West Ham di bawah EFL Championship); backup `entity_registry.json.bak_pre_identity_20260902`. | `teams.json`, `cache/football/entity_registry.json` |

## Verifikasi

- `tests/test_wrong_team_identity_2026_09_02.py` (18 test): 23 pasangan insiden ditolak, 23 pasangan sah diterima oleh SEMUA matcher provider; form per ID tidak tergantung urutan T1/T2 dan menolak ID yang bukan sisi event; jalur by-name mengabaikan baris negara lain / U21 / Women; baris NowGoal tanpa nama tim dibuang; H2H window tidak menolkan rekor; tie state beda kompetisi; Elo settle tidak membuat key "Lille"; OddsPapi menolak orientasi terbalik; halaman feed ganda dihitung sekali.
- Replay offline pada cache feed LiveScore nyata (2 Sep) dengan matcher baru: Southampton → 3-1, 5-1, 1-1; Stoke → 1-3, 1-4, 1-0; Portsmouth → 3-1, 1-2, 0-2; Parma → 0-1, 0-2, 0-2; Inter → 1-0 Cagliari; Wolves → 3-1, 4-1, 2-4 (semua benar-benar milik klub tersebut).
- Full suite: 1545 lulus; 3 gagal yang sama seperti sebelum perubahan (`test_bugfix_b1_b10` alias Almere, `test_dynamic_league` cache identity-guard, `test_football` jumlah liga 36). Dua test lama disesuaikan: fixture LiveScore Eredivisie kini menyertakan `country` (rule baru), placeholder "Derby" (kini klub dikenal) diganti nama fiktif.

## Sisa risiko yang kemudian ditutup (revisi 2)

| Risiko | Penutup | Test |
|---|---|---|
| Nama satu token ambigu ("Inter" vs "FC Inter Turku", "Wolves" vs "Wollongong Wolves") | Jalur by-name kini **menolak, bukan menebak**: `_livescore_form` mengembalikan None bila baris yang cocok berasal dari ≥2 klub berbeda (`team_identity.distinct_clubs`); `_pick_sphinx_team` / `_pick_suggest_team` mengembalikan None bila hit parsial berasal dari klub berbeda tanpa hit exact. Rantai provider lanjut ke sumber lain (atau form kosong yang jujur). | I12 |
| Registry entitas write-only | Dibaca saat fetch: `MultiSourceStatsFetcher._fallback_identity_ok` memverifikasi setiap hasil provider by-name (football-data / thesportsdb) terhadap klub yang diminta — canonical teams.json harus sama, `(provider, id)` yang sudah terdaftar untuk klub lain ditolak, tanpa bukti alias nama harus cocok token-level. | I11 |
| Kontaminasi konteks antar match (lineup/venue/stats) | Semua fetch konteks di-key oleh URL flashscore fixture; URL kini diverifikasi memuat slug kedua tim (`flashscore_url_matches`), bila menamai pasangan lain seluruh fetch konteks dilewati. Payload event-context yang menamai sisinya diverifikasi dengan `same_fixture` (dibuang bila pasangan lain, ditukar bila terbalik). Catatan: klaim agen bahwa dump debug Lincoln memuat venue Stoke tidak terbukti (dump debug tidak punya match_info sama sekali); guard tetap dipasang sebagai penutup umum. | I10 |

## Lapisan merge multi-source (revisi 3)

`datasources.py` kini membawa identitas per field, bukan hanya per provider:

- `FieldSample.entity` = pasangan tim yang benar-benar di-fetch oleh sumber (nama, id provider, canonical id). Adapter LiveScore mengisinya dari event yang di-resolve (`source_id`, `home_id`, `away_id`); adapter Flashscore dan field primer dari `analyse._primary_fields` mengisinya dari pasangan yang di-resolve.
- `sample_identity()` memutuskan `verified` / `reversed` / `unknown` / `reject` SEBELUM nilai dibandingkan: dari `entity` (canonical id menang atas ejaan), atau dari isi nilai (nama home/away pada `match`, nama pertemuan pada `h2h`, `team_name`/`name` per sisi pada form/lineup).
- `merge_field`: sampel `reject` tidak pernah menang dan tidak pernah dihitung sebagai "setuju" (dicatat di `identity_rejected` dengan alasan); sampel `reversed` ditukar sisinya. `source_metadata` kini memuat `identity` dan `identity_rejected` per field.
- `LiveScoreDataSource._orient` tidak lagi menukar sisi bila pencocokan terurut gagal; pasangan yang cocok dua arah atau tidak sama sekali ditolak.
- Test: `tests/test_multisource_identity_2026_09_02.py` (6 test, M1–M6).

## Sisa yang benar-benar tersisa

- Log lokal 26 Agu–1 Sep tetap berisi snapshot dengan form yang salah; evaluasi minggu ini (`eval_2026-09-02.md`) dihitung atas pick yang sebagian dibangun dari data itu. Angka bersih baru terlihat setelah deploy dan satu putaran match baru.
