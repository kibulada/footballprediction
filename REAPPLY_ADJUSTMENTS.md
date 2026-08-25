# REAPPLY ADJUSTMENTS — Panduan Mengaktifkan Kembali Fitur yang Di-Revert

> **Tanggal**: 2026-08-19
> **Tujuan**: Panduan untuk mengaktifkan kembali 3 fitur yang di-revert agar bot kembali ke perilaku "post-adjustment" (sesudah jam 01:13 dini hari).
> **Kondisi saat ini**: Bot dalam kondisi "kemarin malam" — tanpa P1-3, tanpa F2, tanpa statistical component.

---

## Ringkasan Fitur yang Di-Revert

| ID | Fitur | Fungsi | File Utama | Effort |
|---|---|---|---|---|
| **R1** | P1-3 Evidence Floor | Cap skor saat data tipis (0.52/0.65) | `signal_engine.py` | Low |
| **R2** | F2 Elo Prior Veto | NO BET saat model berbasis elo prior + form tipis | `signal_engine.py` | Low |
| **R3** | Statistical Component | Masukkan frekuensi gol ke scoring (weight 0.25) | `signal_engine.py` | Low |

**Catatan**: Semua fitur di-share lokasi yang sama (`signal_engine.py`), jadi bisa diaktifkan sekaligus atau satu per satu.

---

## R1 — P1-3 Evidence Floor

### Apa yang dilakukan
Membatasi skor maksimum saat komponen `statistical` dan/atau `movement` tidak tersedia. Mencegah skor tinggi palsu untuk match dengan data tipis.

| Kondisi | Cap Skor |
|---|---|
| Statistical DAN movement keduanya UNAVAILABLE | **0.52** (MEDIUM floor) |
| Salah satu UNAVAILABLE | **0.65** (MEDIUM upper) |
| Keduanya tersedia | tanpa cap |

### Cara Mengaktifkan

#### Langkah 1: Tambah konstanta di `signal_engine.py`

Cari blok ini (sekitar line 718, sebelum `COVERAGE_FLOOR_DEFAULTS`):

```python
# P3-4: coverage floor. Confidence is a statement about the EVIDENCE; when
```

**Tambahkan SEBELUMNYA:**

```python
# P1-3: evidence floor thresholds. When key evidence groups (statistical,
# movement) are unavailable the score is built on a thinner basis and the
# headline number must NOT look like a strong conviction pick. Caps are
# applied to the 0..1 score -- see ``_apply_evidence_floor``.
EVIDENCE_FLOOR_DEFAULTS: dict[str, float] = {
    "score_cap_both_unavailable": 0.52,   # neither stat nor movement -> MEDIUM floor
    "score_cap_one_unavailable":  0.65,   # one missing -> MEDIUM upper
}

```

#### Langkah 2: Tambah fungsi `_apply_evidence_floor` di `signal_engine.py`

Cari fungsi `_late_component`:

```python
def _late_component(mv: dict[str, Any]) -> float:
```

**Tambahkan fungsi ini SEBELUM `_late_component`:**

```python
def _apply_evidence_floor(
    score: float,
    components: dict[str, float],
    cfg: dict[str, Any] | None = None,
) -> float:
    """Cap the headline score when key evidence is unavailable.

    ``statistical`` absent (no empirical form frequencies) AND movement
    UNAVAILABLE (no opening price series) -> cap to MEDIUM floor; one
    missing -> cap to MEDIUM upper. Caps are opt-in via config keys
    ``score_cap_both_unavailable`` / ``score_cap_one_unavailable``; the
    defaults live in ``EVIDENCE_FLOOR_DEFAULTS``. Never lowers the score
    (a legitimately high score with all evidence stays high).
    """
    cfg = cfg or {}
    has_stat = "statistical" in components
    has_mv = _movement_available(components.get("_movement_block"))
    missing = (not has_stat) + (not has_mv)
    if missing == 2:
        cap = float(cfg.get("score_cap_both_unavailable", EVIDENCE_FLOOR_DEFAULTS["score_cap_both_unavailable"]))
    elif missing == 1:
        cap = float(cfg.get("score_cap_one_unavailable", EVIDENCE_FLOOR_DEFAULTS["score_cap_one_unavailable"]))
    else:
        return score
    return min(score, cap)


```

#### Langkah 3: Edit fungsi `score_signals()` di `signal_engine.py`

**3a. Tambah param `evidence_floor_cfg`:**

Cari:
```python
def score_signals(
    signals: list[Signal],
    *,
    weights: dict[str, float],
    min_edge_pp: float,
    conflict_pp: float,
    completeness: float,
    context: dict[str, Any] | None,
) -> None:
```

Ganti menjadi:
```python
def score_signals(
    signals: list[Signal],
    *,
    weights: dict[str, float],
    min_edge_pp: float,
    conflict_pp: float,
    completeness: float,
    context: dict[str, Any] | None,
    evidence_floor_cfg: dict[str, Any] | None = None,
) -> None:
```

**3b. Tambah `_movement_block` + hapus komentar statistical:**

Cari:
```python
        # Group B: statistical DISABLED — reverted to pre-adjustment behavior.
        # Data still flows from NowGoal but is NOT used in scoring.
        # Uncomment to re-enable: comps["statistical"] = s.components["statistical"]
        active = sum(weights.get(k, 0.0) for k in comps)
        total = sum(weights.get(k, 0.0) * comps[k] for k in comps)
        score = (total / active) if active > 0 else 0.0
        s.components = comps
        s.score = round(score, 3)
```

Ganti menjadi:
```python
        # Group B: statistical (may be absent).
        if "statistical" in s.components:
            comps["statistical"] = s.components["statistical"]
        # P1-3 internal key: keep the raw movement block accessible to
        # _apply_evidence_floor without polluting the displayed components.
        comps["_movement_block"] = s.movement or {}

        # Internal underscore-prefixed keys (e.g. ``_movement_block``, a dict
        # carrying the raw movement block for ``_apply_evidence_floor``) are
        # metadata, NOT score components -- they must never enter the weighted
        # sum (a dict value would raise on ``weight * value``).
        active = sum(weights.get(k, 0.0) for k in comps if not k.startswith("_"))
        total = sum(weights.get(k, 0.0) * comps[k] for k in comps if not k.startswith("_"))
        score = (total / active) if active > 0 else 0.0
        score = _apply_evidence_floor(score, comps, evidence_floor_cfg)
        # Strip the internal key before exposing components to callers.
        comps.pop("_movement_block", None)
        s.components = comps
        s.score = round(score, 3)
```

#### Langkah 4: Tambah forwarding di `run_signal_engine()`

Cari:
```python
    score_signals(
        signals,
        weights=weights,
        min_edge_pp=min_edge_pp,
        conflict_pp=conflict_pp,
        completeness=completeness,
        context=context,
    )
```

Ganti menjadi:
```python
    score_signals(
        signals,
        weights=weights,
        min_edge_pp=min_edge_pp,
        conflict_pp=conflict_pp,
        completeness=completeness,
        context=context,
        evidence_floor_cfg=(cfg or {}).get("evidence_floor") if isinstance(cfg, dict) else None,
    )
```

#### Langkah 5: Tambah config di `config/football.json`

Cari blok `"coverage_floor"`:

```json
   "coverage_floor": {
```

**Tambahkan SEBELUMNYA:**

```json
   "evidence_floor": {
    "score_cap_both_unavailable": 0.52,
    "score_cap_one_unavailable": 0.65
   },
```

### Verifikasi R1
- Run `analyse --home "Cardiff City" --away "Wrexham AFC"` → skor semua signal capped di 0.52 (kalau data tipis)
- Cek `signal_engine_ranking[].score` ≤ 0.52 untuk match tanpa statistical + movement

---

## R2 — F2 Elo Prior Veto

### Apa yang dilakukan
Memveto pick menjadi NO BET saat model berbasis elo prior (tim belum terseed) dengan form tipis dan tanpa H2H. Mencegah pick palsu dari model yang tidak cukup data.

### Cara Mengaktifkan

#### Langkah 1: Tambah param `evidence_floor` di `rank_and_pick()`

Cari:
```python
def rank_and_pick(
    signals: list[Signal],
    *,
    best_pick_margin: float,
    no_bet_score: float,
    min_confluence: int,
    conflict_pp: float,
    min_data_quality: float,
    completeness: float,
    confidence_thresholds: dict[str, float] | None = None,
    odds_disagreement: bool = False,
    model_decision_type: str | None = None,
) -> dict[str, Any]:
```

Ganti menjadi:
```python
def rank_and_pick(
    signals: list[Signal],
    *,
    best_pick_margin: float,
    no_bet_score: float,
    min_confluence: int,
    conflict_pp: float,
    min_data_quality: float,
    completeness: float,
    confidence_thresholds: dict[str, float] | None = None,
    odds_disagreement: bool = False,
    evidence_floor: dict[str, Any] | None = None,
    model_decision_type: str | None = None,
) -> dict[str, Any]:
```

#### Langkah 2: Tambah veto + cap logic di `rank_and_pick()`

Cari:
```python
        elif completeness < min_data_quality:
            reasons.append(f"data quality {completeness:.2f} < {min_data_quality:.2f}")
        else:
            decision = "BEST PICK"
            pick = best
            if model_decision_type in NON_ACTIONABLE_DECISIONS:
```

Ganti menjadi:
```python
        elif completeness < min_data_quality:
            reasons.append(f"data quality {completeness:.2f} < {min_data_quality:.2f}")
        elif evidence_floor and evidence_floor.get("veto"):
            # F2 veto: prior-Elo λ + thin/no form + no H2H is NOT a bettable
            # signal. NO BET with the explicit reason (never a silent pick).
            reasons.append(f"{evidence_floor.get('note', 'evidence tipis')} — {best.selection}")
        else:
            decision = "BEST PICK"
            pick = best
            if evidence_floor and not evidence_floor.get("veto"):
                # F2 cap: same thin prior-based evidence, but H2H exists --
                # never present HIGH on a prior alone.
                best.confidence = "LOW"
                best.evidence_notes = [evidence_floor.get("note", "evidence tipis")]
                reasons.append(evidence_floor["note"])
            if model_decision_type in NON_ACTIONABLE_DECISIONS:
```

#### Langkah 3: Tambah F2 logic di `run_signal_engine()`

Cari:
```python
    result = rank_and_pick(
        signals,
        best_pick_margin=best_pick_margin,
        no_bet_score=no_bet_score,
        min_confluence=min_confluence,
        conflict_pp=conflict_pp,
        min_data_quality=min_data_quality,
        completeness=eff_completeness,
        confidence_thresholds=cfg,
        odds_disagreement=disagreement,
        model_decision_type=model_decision_type,
    )
```

**Tambahkan SEBELUMNYA:**

```python
    # F2 (evidence floor): a model whose λ comes ONLY from a prior Elo rating
    # (teams never seeded -> 1500 default) plus a form window too thin for the
    # statistical component to mean anything (< MIN_EVIDENCE_FORM_MATCHES) is
    # NOT enough evidence to advertise a BEST PICK. When H2H is also absent
    # the pick is VETOED to NO BET (the ADO-Den-Haag-class incident: HOME -0
    # at 62/100 built on a 1500 prior + 1-match form, lost 0-2); when H2H
    # exists the confidence is capped LOW and the reason is surfaced on the
    # card. ``model_probs`` already carries elo_seeded / lambda_source from
    # the prediction engine -- this is the first consumer that actually uses
    # them instead of treating the prior as measured strength.
    evidence_floor: dict[str, Any] | None = None
    _mp = model_probs or {}
    if _mp.get("lambda_source") == "elo" and _mp.get("elo_seeded") is False:
        _hg = _recent_goals((stats or {}).get("home_recent_goals"))
        _ag = _recent_goals((stats or {}).get("away_recent_goals"))
        _thin = len(_hg) < MIN_EVIDENCE_FORM_MATCHES or len(_ag) < MIN_EVIDENCE_FORM_MATCHES
        if _thin:
            evidence_floor = {
                "veto": not has_h2h,
                "note": (
                    "model berbasis prior Elo (tim belum terseed) dengan form tipis "
                    f"(< {MIN_EVIDENCE_FORM_MATCHES} match) dan tanpa dukungan H2H"
                    if not has_h2h
                    else "model berbasis prior Elo (tim belum terseed) dengan form tipis "
                    f"(< {MIN_EVIDENCE_FORM_MATCHES} match) — confidence dibatasi"
                ),
            }
```

**Ganti forwarding:**
```python
    result = rank_and_pick(
        signals,
        best_pick_margin=best_pick_margin,
        no_bet_score=no_bet_score,
        min_confluence=min_confluence,
        conflict_pp=conflict_pp,
        min_data_quality=min_data_quality,
        completeness=eff_completeness,
        confidence_thresholds=cfg,
        odds_disagreement=disagreement,
        evidence_floor=evidence_floor,
        model_decision_type=model_decision_type,
    )
```

### Verifikasi R2
- Run `analyse --home "Fenerbahce" --away "Lyon"` → cek output: NO BET (jika elo prior + form tipis + no H2H)
- Run `analyse --home "Espanyol" --away "Real Madrid"` (LaLiga, ELO ada) → cek output: tetap BEST PICK

---

## R3 — Statistical Component

### Apa yang dilakukan
Menghitung frekuensi gol dari 10 match terakhir tiap tim dan memasukkannya ke scoring (weight 0.25). Membuat skor lebih realistis berdasarkan data aktual.

### Cara Mengaktifkan

#### Langkah 1: Aktifkan kembali di `score_signals()`

Cari (setelah R1 diaktifkan):
```python
        # Group B: statistical DISABLED — reverted to pre-adjustment behavior.
        # Data still flows from NowGoal but is NOT used in scoring.
        # Uncomment to re-enable: comps["statistical"] = s.components["statistical"]
```

Ganti menjadi:
```python
        # Group B: statistical (may be absent).
        if "statistical" in s.components:
            comps["statistical"] = s.components["statistical"]
```

**Catatan**: Baris ini sudah ada di R1 (langkah 3b). Jadi kalau R1 sudah diaktifkan, R3 sudah aktif juga.

### Verifikasi R3
- Run `analyse --home "Cardiff City" → Away" --away "Wrexham AFC"` → cek komponen scoring ada `statistical` (jika data NowGoal tersedia)
- Bandingkan skor: Over 2.5 seharusnya lebih tinggi dari BTTS (karena frekuensi gol mendukung)

---

## Urutan Aktivasi yang Disarankan

| Prioritas | Fitur | Kapan Aktifkan |
|---|---|---|
| 1 | **R3** (Statistical) | Aktifkan dulu — data lebih lengkap |
| 2 | **R1** (P1-3) | Aktifkan setelah R3 — cap berfungsi optimal |
| 3 | **R2** (F2) | Aktifkan terakhir — veto hanya untuk elo prior |

**Atau aktifkan sekaligus** dengan mengikuti langkah R1 → R2 → R3 secara berurutan.

---

## Rollback (Kembali ke Kondisi "Kemarin Malam")

Untuk mengembalikan ke kondisi saat ini (tanpa P1-3, F2, statistical):

1. **R3**: Ganti kembali komentar `"# Group B: statistical DISABLED..."`
2. **R1**: Hapus `EVIDENCE_FLOOR_DEFAULTS`, `_apply_evidence_floor()`, `_movement_block`, `evidence_floor_cfg` param, forwarding, config block
3. **R2**: Hapus `evidence_floor` param di `rank_and_pick()`, veto+cap logic, F2 block di `run_signal_engine()`

Atau ikuti panduan di `REVERT_P1-3.md` (untuk R1) dan revert manual untuk R2 + R3.

---

## Test Strategy

### Unit tests
- R1: `_apply_evidence_floor` caps correctly (both/one/none unavailable)
- R2: `rank_and_pick` vetoes when elo prior + thin form + no H2H
- R3: `score_signals` includes statistical when data present

### Regression checks
- Run `validate-multileague` untuk LaLiga, EPL, Serie A setelah setiap aktivasi
- Bandingkan signal outputs sebelum/sesudah aktivasi
- Pastikan HIGH-quality leagues tidak turun confidence

### Integration test
- Run `analyse --home "Cardiff City" → Away" --away "Wrexham AFC"` → bandingkan dengan log run 1
- Run `analyse --home "Fenerbahce" → Away" --away "Lyon"` → cek NO BET (R2 aktif)

---

## Risk Assessment

| Fitur | Risk | Mitigasi |
|---|---|---|
| R1 | Cap terlalu agresif → false MEDIUM | Start dengan threshold 0.52/0.65, adjust kalau perlu |
| R2 | Veto terlalu banyak → banyak NO BET | Hanya aktif untuk elo prior + form < 3 match + no H2H |
| R3 | Statistical noisy (sample kecil) | Weight 0.25 sudah cukup rendah; data 10 match wajar |
