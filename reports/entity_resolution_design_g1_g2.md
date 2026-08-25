# Desain G1 + G2 (+G5) — Entity Resolution Antar-Source

Tanggal: 2026-08-17
Berdasarkan audit entity resolution (8 poin) — lihat ringkasan audit sebelumnya.

---

## Masalah inti (dari audit)

| Gap | Pernyataan | Bukti konkret |
|---|---|---|
| G1 | Tidak ada canonical team_id + peta `(provider, provider_id) → canonical_id` | `teams.json` memetakan alias → canonical **NAMA**; ID yang tersimpan adalah ID provider lokal (`flashscore` string slug, `football_data` int, `thesportsdb` `idTeam`) dan tidak pernah dipetakan lintas provider |
| G2 | Match ID berbasis nama → duplikat tak terdeteksi | Hari ini: `LaLiga||Espanyol||Levante||2026-08-16` dan `LaLiga||RCD Espanyol de Barcelona||Levante UD||2026-08-16` = match yang SAMA, di-settle 2×. Akar: `resolve_team_alias("Espanyol","LaLiga")` → None karena "Espanyol"/"Levante" tidak ada di teams.json LaLiga |
| G5 | Join form/H2H by nama tanpa verifikasi konteks; coverage teams.json 16 liga | `_livescore_form` scan date-feed hanya pakai `teams_match(nama)` — tidak ada filter kompetisi/kickoff |

---

## Desain G1 — Canonical Team Identity + Registry

### Prinsip
- **Canonical team_id deterministik**: `t:{league_key_lower}:{slug(canonical_name)}`
  - `canonical_name` dari `_canonical_team_name` (prediction_log) → teams.json dulu, fallback suffix-strip.
  - Deterministik → tidak perlu menyimpan ID; bisa di-recompute kapan saja.
  - Sluggable → bisa dibaca manusia (debug), tidak seperti hash.
- **EntityRegistry**: file `cache/football/entity_registry.json` memetakan
  `{provider: {provider_id: {canonical_id, canonical_name, league_key}}}`.
  - `register(provider, provider_id, league_key, name)` → resolve canonical name → canonical_id → simpan.
  - `lookup(provider, provider_id)` → entri atau None.
  - `resolve(provider, provider_id)` → canonical_id.
  - `conflicts()` → `(provider, provider_id)` yang terdaftar dengan canonical_id BERBEDA = sinyal salah-klub/rename → dipakai sebagai guard.
  - Idempoten: register nama sama untuk id sama tidak menulis ulang.

### Wiring
- `multi_source.search_team` dan `search_teams_pair` (flashscore path): setelah resolve, `registry.register(provider, id, league_key, name)` dan tambah `canonical_id` di output team dict.
- Registry hanya **menambah** informasi; tidak pernah menghapus jalur yang ada → aman.

### Kenapa ini menyelesaikan G1
- Ada canonical ID internal yang stabil dan provider-agnostik.
- Mapping `(provider, provider_id) → canonical_id` terakumulasi dari run nyata.
- Conflict detection = lapisan verifikasi yang sebelumnya tidak ada.

---

## Desain G2 — Match Identity Verification

### Prinsip
1. **Data fix (akar duplikat)**: lengkapi teams.json dengan alias yang hilang
   (bukti: `Espanyol`/`Levante` di LaLiga). Setelah fix,
   `_canonical_team_name("Espanyol","LaLiga") == _canonical_team_name("RCD Espanyol de Barcelona","LaLiga")`
   → `make_match_id` menyatu → duplikat hilang dari akar, termasuk untuk snapshot LAMA
   (key dedupe memakai nama kanonik, jadi lama & baru collide).
2. **Entities di snapshot**: `append_snapshot(..., entities={home:{canonical_id, provider, provider_id, name}, away:{...}})`
   → row `entities` tersimpan. Ini memberi bukti ID per snapshot untuk:
   - audit (siapa resolve tim ini, dari provider mana, dengan ID apa);
   - verifikasi settle (lihat bawah);
   - basis evaluasi "data benar-benar dari canonical entity" (poin 8 audit).
3. **Verifikasi settle via ID**: `fetch_finished_livescore_results` teruskan `home_id`/`away_id`/`source_id`
   (sekarang dibuang). `settle_auto`:
   - bila result punya ID dan snapshot punya entities → cocokkan via registry:
     `registry.resolve("livescore", home_id) == entities.home.canonical_id` → verified;
   - nama cocok tapi canonical_id BERTENTANGAN → **jangan settle** (conflict guard, prinsip "never guess the wrong club");
   - tidak ada ID di salah satu sisi → fallback nama seperti sekarang (backward compatible).

### Kenapa ini menyelesaikan G2
- Duplikat match_id dihancurkan dari akar (data fix) — bukan patch di konsumen.
- Verifikasi ID menutup celah "nama terlihat sama tapi klub beda" dan "klub sama tapi nama beda".

---

## Desain G5 — Join Form/H2H Terverifikasi + Coverage

### Prinsip
1. **`_livescore_form` filter kompetisi**: ketika `league_key` dikenal, match di date-feed
   harus lolos `competition_league_key(fx.competition) == league_key` (atau containment
   display). Match cup/liga lain dengan nama tim sama tidak masuk form window.
   (Mengikuti pola `_competition_matches` yang sudah ada di `source_match.py`.)
2. **Coverage nama**: registry (G1) menjembatani tim di luar 16 liga —
   canonical_id deterministik dari nama; tim yang sama dari provider berbeda
   menyatu bila nama resolve ke canonical yang sama.
3. **Backfill data aman**: tim yang TERBUKTI dipakai pipeline (settled list 16 Agu)
   dan liganya jelas ditambahkan ke teams.json → form/standings join lebih baik,
   duplikat match_id berkurang.

### Kenapa ini menyelesaikan G5
- Form window tidak lagi bisa diisi match tim lain dari kompetisi berbeda.
- Tim di luar 16 liga tetap mendapat canonical identity yang konsisten.

---

## Urutan implementasi

1. **G1**: `entity_registry.py` (canonical_team_id + EntityRegistry) → wiring `multi_source`.
2. **G2**: backfill teams.json → `append_snapshot(entities=...)` → `analyse` build entities →
   `fetch_finished_livescore_results` teruskan ID → `settle_auto` verifikasi ID.
3. **G5**: `_livescore_form` filter kompetisi → backfill tim settled.
4. Test baru (registry, dedupe match_id Espanyol/Levante, settle conflict) + full suite.

## Metrik keberhasilan
- `make_match_id("LaLiga","Espanyol","Levante",...) == make_match_id("LaLiga","RCD Espanyol de Barcelona","Levante UD",...)`.
- Snapshot baru berisi `entities` dengan canonical_id yang konsisten.
- `settle_auto` menolak result yang canonical_id-nya bertentangan dengan snapshot.
- Full test suite tetap hijau (1235 → ~1245).
