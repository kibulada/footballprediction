# Desain: Dynamic League Discovery

Tanggal: 2026-08-17
Status: DESAIN (belum implementasi)

---

## 1. Masalah

Audit menemukan: bot **tidak bisa menganalisis liga yang belum terdaftar**, meski
engine resolusi (flashscore homepage fallback, livescore date-feed) secara teknis
mampu menemukan fixture-nya. Penyebab: tiga lapis whitelist statis + gate di
setiap entry point:

| Gate | Lokasi | Efek |
|---|---|---|
| `resolve_league_scored(league_query)` → error "liga tidak dikenal" | analyse.py:1395, source_match.py:463, best_match.py:241, compare.py:38 | Semua entry point hard-require league terdaftar |
| `LEAGUE_PATHS` (flashscore) 23 liga | flashscore.py:45 | Liga lain → fallback homepage (sudah jalan, tapi tak pernah dipakai utk analisa penuh) |
| `_key_from_meta` hardcoded 16 liga | multi_source.py:179 | League key tidak terdeteksi dari display utk liga lain |
| `odds_api_key` + `calibration_{slug}.json` | leagues.json, calibration.py:380 | Odds The-Odds-API & kalibrasi per-liga hanya ada utk liga terdaftar |

Detect path (tanpa keyword liga) sudah mengambil kompetisi dari fixture
(`detect_match.py:271`), tapi jika `competition_league_key(comp)` → None maka
bot menampilkan **info-only** (bot.py:1823): *"Kompetisi belum terdaftar untuk
analisa penuh (tidak ada odds/form model)"*.

---

## 2. Tujuan

1. `analisa match <home> vs <away>` (tanpa keyword liga) → full analysis untuk
   **liga apa pun** yang fixture-nya bisa ditemukan di flashscore/livescore.
2. `analisa match <liga> <home> vs <away>` dengan liga tak dikenal → resolve
   fixture dulu, ambil kompetisi dari fixture, buat league key dinamis.
3. Sinkronisasi lintas-provider tetap berbasis canonical (G1) — termasuk untuk
   liga dinamis.
4. Tanpa merusak jalur liga terdaftar (regresi = 0, full suite hijau).

---

## 3. Desain Inti: Dynamic League Key + Fixture-First Resolution

### D1. League key dinamis (deterministik, bisa di-recompute)

`leagues.json` tetap tabel kanonik utk liga terdaftar. Tambah fungsi di
`league_resolver.py`:

```python
def dynamic_league_key(competition: str) -> str:
    """Deterministik key utk kompetisi yang TIDAK terdaftar.

    format: "dyn:" + slug(competition)  (e.g. "dyn:copa-del-rey")
    - Tidak bertabrakan dengan key terdaftar (prefix "dyn:").
    - Bisa di-recompute kapan saja; tidak perlu persist.
    - competition_league_key(comp) mencoba terdaftar DULU; kalau None -> dynamic.
    """
```

Meta dinamis:
```python
def dynamic_league_meta(competition: str, *, country: str | None = None) -> dict:
    return {
        "display": competition,
        "country": country or "",
        "aliases": [competition],
        "dynamic": True,          # penanda: tanpa odds_api_key, tanpa kalibrasi
        "football_data_code": None,
        "odds_api_key": None,     # tidak ada The Odds API key
    }
```

### D2. Fixture-first: ubah urutan resolve di entry point

**Sekarang** (analyse.py:1395): `resolve_league_scored(query)` → error kalau None.

**Baru** — tambah helper `find_specific_match_fixture_first` (atau refactor
`find_specific_match` menerima `league_key`/`meta` yang sudah di-resolve, bukan
query):

```
1. Coba resolve_league_scored(query)  # jalur terdaftar (existing, unchanged)
2. Kalau None → resolve FIXTURE dulu:
   a. flashscore.resolve_match(None, home, away)   # None -> homepage fallback
      (sudah didukung: scrape_league_matches(None) = homepage, flashscore.py:1435)
      ATAU livescore date-feed scan (source_match._search_livescore butuh key ->
         buat varian _search_livescore_any(home, away) tanpa league filter)
   b. competition = fixture["competition"]  # DARI FIXTURE, bukan whitelist
   c. league_key = competition_league_key(competition) or dynamic_league_key(competition)
   d. meta = load_leagues().get(key) or dynamic_league_meta(competition, country=...)
3. Lanjut pipeline dengan league_key + meta tersebut (identik utk semua entry)
```

**Entry point yang di-refactor** (semua sekarang menerima `league_key`/`meta`
yang sudah di-resolve — query resolution dipindah ke satu helper):

| Entry | Perubahan |
|---|---|
| `find_specific_match` (analyse.py:1357) | terima `league_key` + `meta` (atau query → resolve via helper D2) |
| `find_source_match` (source_match.py:442) | sama |
| `best_match` / `compare` | sama (opsional: tetap butuh league utk batch; fokus utama analisa single match) |
| `runner.py --league required=True` | jadi opsional; tanpa `--league` → detect fixture-first |

### D3. Odds tanpa The Odds API (jalur dinamis)

`odds_key = meta.get("odds_api_key")` → None utk liga dinamis. Pipeline sudah
punya fallback berurutan (analyse.py:1450-1530):

```
oddspapi (find_fixture by name) → nowgoal (match_odds by name) → The Odds API
```

**Perubahan**: The Odds API branch (`if match_odds_payload is None and odds_key:`)
sudah otomatis di-skip saat `odds_key is None`. JADI: untuk liga dinamis, odds
datang dari **oddspapi/nowgoal by-name** — yang sudah jalan dan tidak butuh
league key. **Tidak ada perubahan kode di sini** — cukup memastikan branch
tersebut tidak error saat `odds_key=None` (verifikasi: `and odds_key` sudah ada).

### D4. `_key_from_meta` dinamis

`multi_source._key_from_meta` (multi_source.py:179) hardcoded 16 liga — dipakai
untuk `_league_key` saat meta tidak membawanya. Perubahan:

```python
def _key_from_meta(league_meta):
    # 1. _league_key eksplisit (jalur terdaftar, existing)
    # 2. display yang cocok dgn key terdaftar (existing, tapi dari leagues.json
    #    bukan hardcoded list -> load_leagues() sekali)
    # 3. dynamic: dynamic_league_key(display) kalau display bukan key terdaftar
```

Ini membuat league key dinamis mengalir ke seluruh `_fetch_team_form_uncached`,
cache key (`team_form_{provider}_{league_key}_{id}_{limit}`), dan `search_team`.

### D5. Standings / xG / kalibrasi: skip-without-gate (bukan hard error)

| Komponen | Perilaku utk liga dinamis |
|---|---|
| Standings (`fetch_league_standings(league_key)`) | `LEAGUE_PATHS` tidak punya path → None → field standings kosong, **pipeline lanjut** (sudah perilaku saat ini utk liga tanpa path) |
| xG (understat/FBref) | `supports_league` / `LEAGUE_MAP` tidak punya → None → skip (sudah) |
| Kalibrasi (`league_calibrator`) | `calibration_{slug}.json` tidak ada → `uncalibrated_league=True`, confidence dibatasi, saran NO BET kecuali edge lolos gate (**SUDAH DIIMPLEMENTASI**, analyse.py:1185-1197) |
| CLV gate | segment liga dinamis belum punya settled bets → gate menolak (sudah, `min_bets`) |

**Poin kunci**: pipeline SUDAH dirancang utk "data pendukung tidak ada → lanjut
dengan label jujur". Yang kurang hanya **membuka jalur masuk** (D2) — komponen
lainnya sudah degradasi dengan aman.

### D6. Canonical identity utk liga dinamis (G1 extension)

`canonical_team_id(league_key, name)` dengan league_key = `dyn:copa-del-rey`
menghasilkan `t:dyn-copa-del-rey:{slug}` — unik per kompetisi, tidak lagi
`t:unknown`. Registry (G1) otomatis bekerja karena hanya butuh key konsisten.

Perubahan kecil di `entity_registry.py`: tidak ada — `dynamic_league_key` cukup
menggantikan `None`/`"unknown"` yang sekarang dipakai.

---

## 4. Alur baru (contoh nyata)

**`analisa match barcelona vs real madrid` di Copa del Rey (belum terdaftar):**

```
1. resolve_league_scored("barcelona...") → None (bukan liga)
2. resolve_match(None, "barcelona", "real madrid") → homepage → ketemu,
   competition="Copa del Rey"
3. league_key = competition_league_key("Copa del Rey") → None →
   dynamic_league_key("Copa del Rey") = "dyn:copa-del-rey"
4. meta = dynamic_league_meta("Copa del Rey")
5. Pipeline:
   - odds: oddspapi/nowgoal by-name ✅
   - form/H2H: flashscore slug+id (dari resolve_match) ✅
   - standings/xG: None → skip ✅
   - kalibrasi: uncalibrated_league=True → confidence dibatasi, NO BET kecuali
     edge kuat ✅
6. Output: analisa penuh + label jujur "liga tanpa kalibrasi per-league"
```

---

## 5. File yang berubah

| File | Perubahan |
|---|---|
| `league_resolver.py` | + `dynamic_league_key`, + `dynamic_league_meta`, `competition_league_key` fallback dinamis (opsional) |
| `analyse.py` | `find_specific_match` terima `league_key`+`meta` (resolve dipindah ke helper); + `resolve_or_detect_league` |
| `source_match.py` | `find_source_match` pakai helper yang sama; `_search_livescore_any` (tanpa filter league) utk deteksi |
| `best_match.py` / `compare.py` | pakai helper yang sama (resolve dulu, error kalau benar-benar tak ketemu) |
| `multi_source.py` | `_key_from_meta` dinamis (load_leagues + dynamic fallback) |
| `runner.py` | `--league` jadi opsional utk `analyse`/`livescore`/`flashscore` |
| `bot.py` | branch info-only (bot.py:1823) → full analysis utk liga dinamis |
| `entity_registry.py` | tidak ada (key dinamis sudah cukup) |

---

## 6. Test & metrik keberhasilan

**Test baru:**
1. `dynamic_league_key("Copa del Rey") == "dyn:copa-del-rey"` deterministik, tidak collide dengan key terdaftar.
2. `find_specific_match` dengan `league_query` tak dikenal → resolve fixture (mock) → pipeline jalan dengan `dyn:` key.
3. `_key_from_meta({"display": "Copa del Rey"})` → `dyn:copa-del-rey` (bukan None).
4. Odds branch: `odds_key=None` → skip The Odds API, tidak raise (sudah ada `and odds_key`; tambah test eksplisit).
5. `canonical_team_id("dyn:copa-del-rey", "Barcelona")` → `t:dyn-copa-del-rey:barcelona`, dan registry resolve konsisten.
6. Regression: `competition_league_key` utk 31 liga terdaftar TIDAK berubah (20 kasus existing tetap hijau).

**Metrik:**
- Full suite tetap hijau (1254 → ~1265).
- `analisa match <unknown-league-fixture>` menghasilkan analisa penuh + label
  `uncalibrated_league=True`, bukan info-only.
- Liga terdaftar: output identik (0 regresi).

---

## 7. Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| False-positive fixture (nama mirip, kompetisi salah) | Pakai guard existing: `_side_role` ambigu → None, `competition_league_key` demonym guard, verifikasi canonical settle (G2). Untuk deteksi bebas-league, kompetisi WAJIB ada di fixture — tanpa kompetisi → info-only (tidak force). |
| Liga dinamis tanpa odds (oddspapi/nowgoal kosong) | Pipeline sudah menangani `match_odds_payload=None` → output NO BET dengan alasan "odds tidak tersedia". |
| `_key_from_meta` hardcoded 16 → dinamis bisa mengubah cache key liga terdaftar | Load dari leagues.json (bukan list hardcoded) — hasil identik utk 16 yang ada, jadi tidak ada invalidasi cache. |
| Kompetisi fixture = cup antar 2 tim yang juga main di liga | Competition dari fixture disinkronkan ke `league_mismatch` check (sudah ada, analyse.py:1489) — user query "la liga" utk match Copa tetap diflag. |
| Best/compare batch tanpa league | Tetap butuh league (batch semantics); hanya single-match yang dapat fixture-first. |

---

## 8. Urutan implementasi

1. `league_resolver`: `dynamic_league_key` + `dynamic_league_meta` + test.
2. `multi_source._key_from_meta` dinamis + test.
3. `analyse.find_specific_match`: refactor terima `league_key`/`meta` + helper `resolve_or_detect_league` + test (mock fixture).
4. `runner.py` + `bot.py`: `--league` opsional, branch info-only → full analysis.
5. `source_match._search_livescore_any` utk deteksi bebas-league (flashscore-first sudah didukung).
6. Full suite + verifikasi live 1 match liga tak terdaftar.

## 9. Batas desain (jujur)

- Liga dinamis TIDAK otomatis terdaftar permanen — key `dyn:` dibuat per-run,
  deterministik, tapi tidak menulis leagues.json. Registrasi permanen tetap
  manual (keputusan sadar: data provider/key odds/kalibrasi tetap perlu
  dikonfigurasi).
- Prediksi liga dinamis berlabel `uncalibrated` sampai kalibrasi per-league
  terkumpul (min_samples 400) — gate kehati-hatian tetap berlaku (ini desain,
  bukan bug).
