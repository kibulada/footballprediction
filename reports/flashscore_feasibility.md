# Flashscore sebagai Fallback Sofascore — Laporan Kelayakan (diuji live)

> Tanggal tes: 2026-08-12. Semua temuan dari uji langsung di jaringan ini
> (requests + seleniumbase UC browser). **Tidak ada kode bot yang diubah.**

## Ringkasan

**Flashscore BISA dijadikan provider fallback** pengganti/lawan Sofascore yang
sering 403. Data yang bot butuhkan (form, hasil, H2H, statistik/xG, fixtures)
**terbukti tersedia dan bisa di-extract**. Jalur paling andal = **browser render
via seleniumbase UC** (infrastruktur yang SUDAH ada di `sofascore_browser.py`),
bukan API ringan.

## Hasil tes live

### 1. Akses dasar — LULUS ✅
- `requests.get("https://www.flashscore.com")` → **200 OK**, 732KB, **TIDAK kena
  Cloudflare challenge** (kontras: sofascore 403).
- Halaman liga/matches juga 200 OK (SPA, konten di-render via JS).

### 2. Data yang dibutuhkan bot — SEMUA TERSEDIA ✅
Diuji via seleniumbase UC (headless):

| Kebutuhan bot | Sumber halaman | Hasil tes |
|---|---|---|
| **Form 5 (W/D/L + skor)** | `/team/{slug}/{id}/results/` | ✅ Baris: `08.08. 19:00 \| Valerenga \| Bodo/Glimt \| 1 \| 2 \| W` — skor + W/D/L utuh |
| **Upcoming fixture** | `/team/.../fixtures/` | ✅ Baris: `22.08. 21:00 \| Fauske Sprint \| Bodo/Glimt \| - \| -` |
| **Statistik match (xG!)** | `/match/.../summary/` | ✅ Kategori: `Expected goals (xG), Ball possession, Total shots, Big chances, Touches in opposition box` |
| **H2H (last matches)** | `/match/.../h2h/` | ✅ `H2H: LAST MATCHES: ... Valerenga \| Bodo/Glimt \| 1 \| 2 \| W` |
| **Info match (venue, wasit)** | match page | ✅ `[data-testid='wcl-summaryMatchInformation']` ada |
| **Match list per liga** | `/football/europe/champions-league/` | ✅ 22 match, link `a[href*='/match/']` lengkap dengan slug+hash |

### 3. Jalur API ringan (tanpa browser) — SEBAGIAN ❌
- `global.flashscore.ninja/2/x/feed/...` + header **`X-Fsign: SW9D1eZo`** →
  **200 OK** tanpa browser (mc_7 = match list per liga).
- TAPI data match live (`g_1_{id}`) di-encode **ter-hash** (butuh decoder JS
  internal Flashscore yang kompleks); endpoint `lh_` (H2H) & `1x2_` (odds) → 404.
- **Kesimpulan**: jalur API ringan tidak worth-it untuk data live; butuh browser.

### 4. Antarmuka / repo referensi
- Repo `gustavofariaa/FlashscoreScraping` = **Node.js + Playwright** CLI batch
  (scrape satu musim penuh → JSON/CSV). Lisensi **Unlicense** (bebas).
  Tidak cocok integrasi langsung (stack beda), tapi **selector DOM-nya valid**
  (terverifikasi sama dengan hasil tes kita): `.event__match--twoLine`,
  `[data-testid='wcl-statistics']`, dll.
- **TIDAK perlu Node/Playwright** — seleniumbase UC yang sudah terinstall cukup.

## Rekomendasi arsitektur (jika mau lanjut)

Provider baru `agents/football/flashscore.py`, mengikuti pola `sofascore.py` +
`sofascore_browser.py` yang sudah ada:

```
multi_source chain: sofascore → flashscore → football-data → thesportsdb
```

1. `FlashscoreBrowserClient` — reuse pola UC browser (`Driver(uc=True, headless2=True)`),
   session tunggal + throttle ~1.5s, `close()` bersih (dipakai `_cleanup_browser_zombies`).
2. `fetch_team_form(team)` → buka `/team/{slug}/{id}/results/`, parse
   `.event__match` rows → sequence W/D/L + gf/ga avg.
3. `fetch_upcoming_fixture(home, away)` → cari match di halaman liga
   (`/football/{region}/{league}/`) atau `/fixtures/` tim, cocokkan nama tim.
4. `fetch_match_stats(event_url)` → buka match page, parse
   `[data-testid='wcl-statistics-category']` + value → xG/possession/shots.
5. `fetch_h2h(event_url)` → buka `/h2h/`, parse LAST MATCHES.
6. Resolusi slug: `/match/football/{home-slug}-{hash}/{away-slug}-{hash}/` — slug
   team bisa dibangun dari nama (huruf kecil, strip non-alphanumeric) seperti
   `_slugify` di sofascore_browser, dengan fallback search di halaman liga.

### Risiko
- Flashscore = SPA: butuh browser render (8-15s per halaman, throttle 1.5s).
  Satu `analyse` ≈ 3-4 halaman (liga + 2 form + stats) ≈ **30-60s** — masih di
  bawah deadline runner 85s, mirip beban sofascore browser fallback.
- Nama tim perlu mapping/alias (sama seperti provider lain) — pakai
  `_teams_match` di analyse.py yang sudah toleran.
- Halaman match butuh `{event_id}` — resolusi via match link (pola
  `resolve_team_from_match_link` sofascore sudah ada sebagai template).

## Verdict
| Kriteria | Hasil |
|---|---|
| Bisa diakses dari jaringan ini? | ✅ 200 OK, tanpa Cloudflare block |
| Data form/H2H/xG/fixtures tersedia? | ✅ Terverifikasi live |
| Tanpa browser (API ringan)? | ❌ (data ter-hash; butuh browser) |
| Bisa pakai infrastruktur yang ada? | ✅ seleniumbase UC + `_slugify` + `_teams_match` |
| Effort | Menengah (~150-250 baris provider baru + tests) |
