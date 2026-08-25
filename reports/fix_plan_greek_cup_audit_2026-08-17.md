# Plan Perbaikan — Greek Cup Signal Validity Audit

**Tanggal**: 2026-08-17
**Match audit**: Anagennisi Karditsas vs Aris Thessaloniki (Greek Cup Preliminary Round)
**Skor audit**: data structurally valid tapi 12 issue ditemukan di registry, source reconciliation, dan gating logic.

---

## Ringkasan Prioritas

| ID | Priority | Issue | File utama | Effort |
|---|---|---|---|---|
| P0-1 | **P0** | Entity registry basketball merge | `cache/football/entity_registry.json` | Low |
| P0-2 | **P0** | Match status reconciliation (finished vs live) | `agents/football/source_match.py`, `analyse.py` | Medium |
| P1-1 | **P1** | Opening snapshot pin + display fallback | `agents/football/runner.py`, `format.py` | Low |
| P1-2 | **P1** | Source disagreement LOW → NO BET gate | `agents/football/signal_engine.py`, `analyse.py` | Low |
| P1-3 | **P1** | Score floor when stat/movement unavailable | `agents/football/signal_engine.py` | Medium |
| P2-1 | **P2** | H2H window filter (≤ 3 tahun) | `agents/football/flashscore_h2h.py` / `livescore.py` | Low |
| P2-2 | **P2** | Form sequence primary source rule | `agents/football/datasources.py` | Low |
| P2-3 | **P2** | 1X2 NO BET → AH pick info di expanded view only | `discord_signal_card_accordion.py` | Low |
| ~~P3-1~~ | ~~P3~~ | ~~Line movement tampil eksplisit di MARKET block~~ | ~~`format.py`~~ | ~~DROPPED~~ |
| P3-2 | **P3** | NowGoal context filter (exclude Club Friendly) | `nowgoal.py` | Medium |
| P3-3 | **P3** | Outlier liquidity gate (min 2 bookmaker) | `scorer.py` | Low |
| P3-4 | **P3** | Coverage floor → downgrade confidence | `signal_engine.py` | Low |

Total: 11 issue (P3-1 dropped per user). P0+P1 = 5 issue (wajib sebelum lanjut). P2+P3 = 6 issue (improvement).

---

## P0-1 — Entity registry basketball merge (CRITICAL)

### Problem
`cache/football/entity_registry.json` punya entry `thesportsdb.146998 → "ASK Karditsas BC"` (klub **basket**) di `league_key: "dyn:greece-cup-preliminary-round"`. Klub football yang main adalah **Anagennisi Karditsas** (Livescore ID 12308, Flashscore ID `27FAPnX6`). Lookup by thesportsdb ID akan match ke basketball club, importing data wrong.

### Solusi
1. **Hapus** entry `thesportsdb.146998` dari registry (atau split ke `dyn:greece-basket-league` kalau basketball punya league sendiri).
2. **Tambah** entry baru:
   - `flashscore.27FAPnX6 → {canonical_id: "t:dyn-greece-cup-preliminary-round:anagennisi-karditsas", canonical_name: "Anagennisi Karditsas", sport: "football"}`
   - `livescore.12308 → same`
3. **Tambah** `sport` field di registry entry shape (optional, backward-compat: default `"unknown"`).
4. **Backfill**: panggil `registry().register(...)` utk setiap source yang match di `analyse.py` pipeline sebelum lookup. Sekarang call sudah ada tapi mungkin skip kalau source ga match by ID.

### File yg diubah
- `cache/football/entity_registry.json` (manual edit)
- `agents/football/entity_registry.py` (shape extension)
- `agents/football/analyse.py` (ensure register called pre-lookup)

### Verifikasi
- Run `analyse --home "Karditsa" --away "Aris"` → cek `entity_registry.conflicts()` empty
- Cek `canonical_id` resolved ke `t:dyn-greece-cup-preliminary-round:anagennisi-karditsas` (bukan `...:ask-karditsas-bc`)

### Rollback
Restore `entity_registry.json` dari backup (`.bak` exists di cache).

---

## P0-2 — Match status reconciliation

### Problem
Bot pakai `flashscore.status` sbg primary → `"finished"`. Livescore bilang `"live"` score 0-0. Bot gated prediction → no pick, tp signal engine sempat jalan sblm gate, jd user liat signal. Inkonsisten.

### Solusi
Tambah `match_status_reconciliation()` di `agents/football/source_match.py`:

```python
def reconcile_status(
    flashscore_status: str | None,
    livescore_status: str | None,
    kickoff: datetime,
    score: dict | None,
    now: datetime,
) -> str:
    """Reconcile match status dari multiple sources.

    Logic:
    1. Kalau livescore == "live" AND score != finished AND now within
       [kickoff - 15m, kickoff + 4h] → return "live"
    2. Kalau flashscore == "finished" AND livescore None → return "finished"
    3. Kalau flashscore == "finished" AND livescore == "live" → prefer
       livescore (real-time lebih akurat), return "live"
    4. Default ke primary source
    """
```

Tambah field `reconciled_status` di `data_sources.match.value`.
Gunakan `reconciled_status` utk gate `prediction` di `analyse.py`, bukan `flashscore.status` langsung.

### File yg diubah
- `agents/football/source_match.py` (new fn)
- `agents/football/analyse.py` (ganti gate)

### Verifikasi
- Unit test: 4 cases di atas
- Live run `analyse --home "Karditsa" --away "Aris"` → cek `prediction` field present (bukan None)

### Rollback
Comment-out call ke `reconcile_status()`, fallback ke `flashscore.status`.

---

## P1-1 — Opening snapshot pin + display fallback

### Problem
Card display tulis `Movement: n/a (no opening prices)` padahal odds payload ada opening (Ladbrokes 1.60→1.36, dll). `opening_snapshot` belum di-pin karena first-time ingestion di `odds-poll` background task.

### Solusi
1. **Pin saat first ingestion**: di `agents/football/odds_fetcher.py` atau runner analyse, kalau `opening_snapshot` belum ada di cache tapi odds payload ada `opening_price`, **write ke cache** sebelum return.
2. **Display fallback**: di `format.py:_market_block()` dan `signal_engine.build_market_block()`, kalau `opening_snapshot` None tapi payload punya `opening`, pakai payload `opening` dgn flag `non_canonical: true`.
3. **Acknowledge**: tampilkan di card `⚠️ opening from current snapshot (not pinned)`.

### File yg diubah
- `agents/football/odds_fetcher.py` (pin logic)
- `agents/football/signal_engine.py` (`build_market_block` fallback)
- `agents/football/format.py` (`_market_block` fallback + display flag)

### Verifikasi
- Live run → `Market.Movement` block punya `Over 2.5 Opening: 1.60 Latest: 1.36` (bukan "n/a")
- Signal engine `movement` component populated (bukan UNAVAILABLE)

### Rollback
Revert fallback, keep pin logic only.

---

## P1-2 — Source disagreement LOW → NO BET gate

### Problem
`data_sources.{match,form,h2h}.confidence == "LOW"` (semua). Signal tetap diproses karena availability ok. User liat MEDIUM, realita LOW.

### Solusi
Tambah gate di `signal_engine.rank_and_pick()`:

```python
def _evidence_gate(data_sources: dict) -> bool:
    """Return True kalau signal bisa diproses.
    
    Kalau 3+ critical fields (match, form, h2h) semua confidence LOW → False.
    """
    critical = ["match", "form", "h2h"]
    lows = sum(
        1 for f in critical
        if (data_sources.get(f) or {}).get("confidence") == "LOW"
    )
    return lows < 3  # max 2 low tolerated
```

Panggil di `analyse.py` sblm `run_signal_engine()`. Kalau gate return False → skip signal engine, return NO BET dgn reason `"source confidence too low (3+ critical fields LOW)"`.

### File yg diubah
- `agents/football/signal_engine.py` (new fn `_evidence_gate`)
- `agents/football/analyse.py` (gate call)

### Verifikasi
- Run `analyse --home "Karditsa" --away "Aris"` → card top: `⚪ NO BET — source confidence LOW`
- Existing HIGH-quality matches (LaLiga, EPL) tetap jalan normal

### Rollback
Set `_evidence_gate` selalu return True.

---

## P1-3 — Score floor ketika statistical/movement unavailable

### Problem
Score 90/100 mostly dari value_signal (capped 40) + form_edge (30). Statistical (Group B) dan movement (Group D) UNAVAILABLE → di-exclude dr score weighting. Score tinggi **bukan karena keyakinan**, tapi karena edge value + form gap.

### Solusi
Tambah floor di `signal_engine.score_signals()`:

```python
def _apply_evidence_floor(score: float, components: dict, weights: dict) -> float:
    """Cap score kalau key evidence unavailable.
    
    Kalau statistical ATAU movement UNAVAILABLE → cap di 0.65 (MEDIUM upper).
    Kalau keduanya unavailable → cap di 0.52 (MEDIUM floor).
    """
    has_stat = "statistical" in components
    has_mv = components.get("movement", 0.5) != 0.5  # neutral = unavailable
    if not has_stat and not has_mv:
        return min(score, 0.52)
    if not has_stat or not has_mv:
        return min(score, 0.65)
    return score
```

Panggil sblm `confidence_label()`. Update label di card jadi `MEDIUM` (atau `LOW` kalau cap parah).

### File yg diubah
- `agents/football/signal_engine.py` (`score_signals` + new fn)
- `config/football.json` (tambah `models.signal_engine.evidence_floor` thresholds)

### Verifikasi
- Run audit match → score capped dari 0.90 → 0.52 (atau 0.65)
- Card confidence: `MEDIUM` (bukan VERY HIGH)

### Rollback
Disable call, score original.

---

## P2-1 — H2H window filter (≤ 3 tahun)

### Problem
H2H Karditsa vs Aris: 2018, 2017, 2016. 7-9 tahun lalu. Skuad, manager, divisi semua sudah berubah.

### Solusi
Di `agents/football/flashscore_h2h.py` dan `livescore.py`, tambah filter:

```python
H2H_WINDOW_YEARS = 3

def filter_h2h_recent(meetings: list, now: datetime) -> list:
    cutoff = now - timedelta(days=365 * H2H_WINDOW_YEARS)
    return [m for m in meetings if _parse_dt(m.get("kickoff")) >= cutoff]
```

Tambah metadata `h2h_window: "3y"`, `h2h_total_meetings: N`, `h2h_in_window: M`. Kalau `M == 0` → flag `h2h_relevance: stale` di output.

### File yg diubah
- `agents/football/flashscore_h2h.py` (filter)
- `agents/football/livescore.py` (filter)
- `agents/football/datasources.py` (metadata field)

### Verifikasi
- Run audit → `h2h_in_window: 0`, flag `stale: true` visible
- `statistical` component utk AH pakai data recent_goals (tidak bergantung H2H)

### Rollback
Set `H2H_WINDOW_YEARS = 99`.

---

## P2-2 — Form sequence primary source rule

### Problem
Flashscore: `L-D-L-L-L`. Livescore: `L-L-L-D-L`. Sequence beda despite goals match.

### Solusi
Di `agents/football/datasources.py:merge_form()`, tambah priority rule:

```python
FORM_SOURCE_PRIORITY = ["livescore", "flashscore"]  # live matches pakai livescore

def pick_primary_form(sources: dict) -> tuple[str, dict]:
    """Pilih form sequence by priority; return (source_used, value)."""
    for src in FORM_SOURCE_PRIORITY:
        v = sources.get(src)
        if v and v.get("sequence"):
            return src, v
    return "none", {}
```

Untuk `scheduled`/`live` matches → livescore primary.
Untuk `finished` matches → flashscore primary (lebih stabil post-match).
Tambah metadata `form_primary_source: "livescore" | "flashscore"`.

### File yg diubah
- `agents/football/datasources.py` (priority fn)

### Verifikasi
- Run audit → form konsisten
- score tidak swing antara runs

### Rollback
Revert ke first-wins atau median.

---

## P2-3 — 1X2 NO BET → AH pick info di expanded view

### Problem
`model 1X2: NO BET` di footer, tapi TOP SIGNAL = AH. User ignore note, bet di pick yang internal layer reject. Summary embed (collapse) terlalu noisy kalau warning dimunculkan — user mostly baca header + score, klik `🔽 Lihat Hasil` untuk detail.

### Solusi
Di `discord_signal_card_accordion.py:_best_pick_block()` (dipakai `build_expanded_embed`):

```python
def _best_pick_block(se: dict) -> list[str]:
    bp = se.get("best_pick")
    lines = []
    if bp:
        lines.append(f"🔥 {bp['selection'].upper()}")
        if se.get("model_decision_type") == "NO BET":
            lines.append("⚠️ **MODEL 1X2 REJECTS THIS PICK**")
            lines.append("Engine uses AH layer; 1X2 model says no value.")
        # ... existing
```

WARNING **tidak muncul** di `build_summary_embed` (collapsed). Hanya muncul saat user klik `🔽 Lihat Hasil` → expanded view. Footer di summary tetap punya disclaimer generic, bukan internal model conflict.

### File yg diubah
- `discord_signal_card_accordion.py` (warning position — expanded only)

### Verifikasi
- Summary embed (collapsed): no warning line, no model-decision info
- Expanded embed (setelah klik): warning visible di best_pick block
- Footer unchanged di kedua state

### Rollback
Restore warning di kedua state.

---

## P3-1 — DROPPED

Item di-drop per user decision. Line movement tidak perlu ditampilkan di MARKET block.

---

## P3-2 — NowGoal context filter

### Problem
Last30/last50 aggregates include `Club Friendlies`. Aris vs Napoli 6-2 di pramusim mempengaruhi distribusi goal timing.

### Solusi
Di `agents/football/nowgoal.py` aggregation, exclude competitions tertentu dari recent form aggregation:

```python
EXCLUDED_COMPETITIONS = {"Club Friendlies", "Club Friendly", "Pre-Season"}

def filter_recent_matches(matches: list, exclude: set = EXCLUDED_COMPETITIONS) -> list:
    return [m for m in matches if m.get("competition") not in exclude]
```

Tambah flag `nowgoal_context.excluded_competitions: ["Club Friendlies"]` di metadata.

### File
- `agents/football/nowgoal.py`

---

## P3-3 — Outlier liquidity gate

### Problem
Outlier 18Bet home 33.0 = 91.3% value, tp sample tipis. 1 bookmaker dari 12.

### Solusi
Di `agents/football/scorer.py:find_outlier()`, tambah minimum bookmaker count:

```python
def find_outlier(bookmaker_odds, consensus, threshold_pct, min_bm=3):
    """Outlier hanya dihitung kalau >= min_bm bookmakers diverging dari consensus."""
    ...
```

Atau flag `outlier_liquidity: "thin"` kalau cuma 1 bookmaker.

### File
- `agents/football/scorer.py`

---

## P3-4 — Coverage floor → downgrade confidence

### Problem
Coverage 4/8 fields (50%). Lineup, injuries, standings, recent_matches all unavailable. Confidence tetap MEDIUM/HIGH.

### Solusi
Di `signal_engine.rank_and_pick()`, tambah coverage check:

```python
def _coverage_floor(completeness: float, confidence: str) -> str:
    """Downgrade confidence kalau coverage tipis."""
    if completeness < 0.40 and confidence in ("VERY HIGH", "HIGH"):
        return "MEDIUM"
    if completeness < 0.25:
        return "LOW"
    return confidence
```

Threshold di `config/football.json:models.signal_engine.coverage_floor`.

### File
- `agents/football/signal_engine.py`
- `config/football.json`

---

## Urutan Eksekusi

### Phase 1 — Critical (P0)
1. P0-1: Entity registry basketball merge → 30 min
2. P0-2: Match status reconciliation → 2 jam (butuh unit test)

### Phase 2 — Important (P1)
3. P1-1: Opening snapshot pin + display fallback → 1 jam
4. P1-2: Source disagreement gate → 1 jam
5. P1-3: Score floor logic → 2 jam (butuh backtest regression)

### Phase 3 — Improvement (P2)
6. P2-1: H2H window filter → 30 min
7. P2-2: Form primary source rule → 30 min
8. P2-3: 1X2 NO BET info di expanded view → 30 min (revised: less work, expanded-only)

### Phase 4 — Polish (P3)
9. ~~P3-1: Line movement display → DROPPED~~
10. P3-2: NowGoal filter → 30 min
11. P3-3: Outlier liquidity gate → 30 min
12. P3-4: Coverage floor → 30 min

**Total estimate**: ~10 jam (~1.5 hari kerja).

---

## Test Strategy

### Unit tests per phase
- P0-1: registry conflict detection
- P0-2: 4 reconciliation cases
- P1-1: fallback when snapshot pinned/missing
- P1-2: gate thresholds (0, 1, 2, 3 LOW fields)
- P1-3: floor applied correctly
- P2-1: H2H filter (within/outside window)
- P2-2: primary source picked correctly
- P2-3: warning rendered top vs footer

### Regression checks
- Run `validate-multileague` utk LaLiga, EPL, Serie A setelah setiap phase
- Compare signal outputs sebelum/sesudah patch
- Pastikan HIGH-quality leagues (LaLiga) tidak turun confidence

### Integration test
- Run live `analyse --home "Karditsa" --away "Aris"` setelah Phase 2 selesai
- Verify: NO BET card, "source confidence LOW" reason visible
- Run `analyse --home "Espanyol" --away "Real Madrid"` (LaLiga, HIGH quality)
- Verify: HIGH confidence preserved, signal tetap ada

---

## Risk Assessment

| Phase | Risk | Mitigation |
|---|---|---|
| P0-1 | Hapus registry entry → break lookup kalau ada match yg depend | Backup `.bak` exists; test lookup sebelum delete |
| P0-2 | Reconciliation logic wrong → predict match finished | Failsafe ke primary source kalau rule ambiguous |
| P1-1 | Pin snapshot salah → movement jadi misleading | Validate `opening_price` exists + valid odds (>1.0) |
| P1-2 | Gate terlalu agresif → NO BET utk match valid | Start conservative (3+ LOW), expand kalau false positive |
| P1-3 | Floor terlalu strict → false MEDIUM utk match HIGH | Backtest 50 matches sebelum activate |

---

## Done Criteria

Phase 1 (P0): registry clean + match status reconciled → ready for Phase 2.
Phase 2 (P1): signal gate aktif + score floor working → audit match kembali, card NO BET.
Phase 3 (P2): H2H stale + form consistency + UI warning → polish quality.
Phase 4 (P3): all cosmetic + edge cases → ship.

End state: lower-tier cup matches spt Karditsa vs Aris → card NO BET dgn explicit reason. Top-tier matches (LaLiga, EPL) → unchanged signal quality.