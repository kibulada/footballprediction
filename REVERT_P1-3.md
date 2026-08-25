# REVERT P1-3 — Evidence Floor (Score Cap saat Evidence Tidak Tersedia)

> **Tanggal**: 2026-08-18
> **Tujuan**: Panduan membalik perubahan P1-3 ("Score floor ketika statistical/movement unavailable") di signal engine.
> **Latar belakang**: Pada audit match **Cardiff vs Wrexham** (EFL Championship, hasil 1-1), P1-3 membuat BEST PICK berubah dari **BTTS Yes (menang)** di run sebelum adjust menjadi **Over 2.5 (kalah)** di run setelah adjust. P1-3 men-cap skor semua sinyal di 0.52 saat komponen `statistical` + `movement` belum tersedia, sehingga pick di run pertama ditentukan tie-break model_prob — bukan karena sinyal kuat.

---

## 1. Apa yang dilakukan P1-3

P1-3 menambahkan **evidence floor**: jika komponen evidence kunci tidak tersedia, skor mentah di-cap agar tidak terlihat seperti pick yang kuat padahal basis evidennya tipis.

| Kondisi | Cap skor (0..1) |
|---|---|
| `statistical` DAN `movement` keduanya UNAVAILABLE | **0.52** (MEDIUM floor) |
| Salah satu UNAVAILABLE | **0.65** (MEDIUM upper) |
| Keduanya tersedia | tanpa cap |

Catatan: cap **hanya menurunkan** skor, tidak pernah menaikkan. Komponen `statistical` (Group B, weight 0.25) adalah fitur terpisah — lihat bagian 4 untuk opsi revert penuh.

---

## 2. File & lokasi yang terlibat

| File | Lokasi | Isi |
|---|---|---|
| `agents/football/signal_engine.py` | :720 | `EVIDENCE_FLOOR_DEFAULTS` (konstanta cap) |
| `agents/football/signal_engine.py` | :759 | `_apply_evidence_floor()` (fungsi cap) |
| `agents/football/signal_engine.py` | :1082–1087 | `comps["_movement_block"]` (key internal untuk floor) |
| `agents/football/signal_engine.py` | :1088 | `score = _apply_evidence_floor(...)` (pemanggilan) |
| `agents/football/signal_engine.py` | :1093 | `comps.pop("_movement_block", None)` |
| `config/football.json` | :292–296 | blok `signal_engine.evidence_floor` |

---

## 3. Opsi A — Revert minimal (matikan cap, tanpa hapus kode)

Paling aman & cepat. Cap dinonaktifkan, kode dibiarkan untuk rollback mudah.

### 3a. Nonaktifkan pemanggilan di `signal_engine.py:1088`

**Sebelum:**
```python
        score = (total / active) if active > 0 else 0.0
        score = _apply_evidence_floor(score, comps, evidence_floor_cfg)
        # Strip the internal key before exposing components to callers.
        comps.pop("_movement_block", None)
```

**Sesudah:**
```python
        score = (total / active) if active > 0 else 0.0
        # P1-3 REVERTED: evidence floor dinonaktifkan — skor mentah dipakai apa adanya.
        # score = _apply_evidence_floor(score, comps, evidence_floor_cfg)
        # Strip the internal key before exposing components to callers.
        comps.pop("_movement_block", None)
```

### 3b. (Opsional) Set cap ke 1.0 di config — `config/football.json:292`

Cara alternatif tanpa sentuh kode: set cap agar tidak pernah mengikat.

**Sebelum:**
```json
   "evidence_floor": {
    "score_cap_both_unavailable": 0.52,
    "score_cap_one_unavailable": 0.65
   },
```

**Sesudah:**
```json
   "evidence_floor": {
    "score_cap_both_unavailable": 1.0,
    "score_cap_one_unavailable": 1.0
   },
```

> ⚠️ Jangan lakukan 3a DAN 3b sekaligus — cukup salah satu. Kalau keduanya, tidak masalah secara hasil (cap 1.0 = tidak mengikat), tapi lebih bersih pakai satu cara.

### Verifikasi Opsi A
- Jalankan `python -m agents.football.analyse --home "Cardiff City" --away "Wrexham AFC"` (atau command analyse yang biasa dipakai).
- Cek output: skor sinyal TIDAK lagi mentok di 0.52 saat data tipis.
- Cek di `cache/football/predictions.jsonl` event `snapshot` → `signal_engine_ranking[].score` menunjukkan skor mentah (mis. Over 2.5 > 0.52 saat data lengkap).

---

## 4. Opsi B — Revert penuh (hapus kode P1-3 + komponen `statistical`)

> ⚠️ Ini mengembalikan perilaku **pra-adjustment penuh** (sebelum P1-3 DAN sebelum komponen statistical ikut dihitung). Gunakan hanya jika ingin kembali ke perilaku lama secara menyeluruh. Perlu backup file dulu.

### 4a. Hapus konstanta — `signal_engine.py:720`
Hapus blok:
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

### 4b. Hapus fungsi `_apply_evidence_floor` — `signal_engine.py:759`
Hapus seluruh fungsi (dari `def _apply_evidence_floor(` sampai sebelum `def _late_component(`).

### 4c. Hapus key internal `_movement_block` — `signal_engine.py:1082–1087` & `:1093`
- Hapus `comps["_movement_block"] = s.movement or {}` beserta komentar "P1-3 internal key...".
- Hapus `comps.pop("_movement_block", None)`.
- Hapus `score = _apply_evidence_floor(score, comps, evidence_floor_cfg)`.

### 4d. Hapus komponen `statistical` dari scoring — `signal_engine.py` (blok Group B di `score_signals`)
Hapus:
```python
        # Group B: statistical (may be absent).
        if "statistical" in s.components:
            comps["statistical"] = s.components["statistical"]
```
Dan (opsional, agar `statistical` tidak dihitung sama sekali) hapus blok `# ---- statistical support per signal` di `build_signals` (sekitar :1008–1022).

### 4e. Hapus blok config — `config/football.json:292`
```json
   "evidence_floor": {
    "score_cap_both_unavailable": 0.52,
    "score_cap_one_unavailable": 0.65
   },
```

> Catatan: dengan menghapus komponen `statistical`, skor kembali dihitung dari `model + market + movement + late_movement + data_quality` saja (normalisasi weight otomatis menyesuaikan).

### Verifikasi Opsi B
- Pastikan `grep -n "evidence_floor\|_movement_block\|statistical" agents/football/signal_engine.py` tidak menemukan sisa referensi yang error.
- Jalankan analyse untuk match yang sama → bandingkan skor dengan log run 1 (BTTS Yes 0.52, semua cap) — diharapkan skor mentah muncul (mis. Over 2.5 ~0.72 di run dengan data tipis).
- Jalankan regression: `validate-multileague` untuk LaLiga/EPL/Serie A → pastikan tidak ada error import / KeyError.

---

## 5. Rollback (balikin lagi kalau revert-nya mau dibatalkan)

- **Opsi A**: buka komentar `score = _apply_evidence_floor(...)` atau kembalikan cap ke 0.52/0.65 di config.
- **Opsi B**: restore `signal_engine.py` & `config/football.json` dari backup (salinan sebelum edit).

---

## 6. Risiko & catatan

| Risiko | Mitigasi |
|---|---|
| Tanpa evidence floor, match dengan data tipis bisa menampilkan skor tinggi (false confidence) | Inilah trade-off yang dipilih — skor mentah apa adanya |
| Opsi B menghapus komponen statistical → fitur lain (backtest, validation) yang membaca `components["statistical"]` perlu dicek | Cek `grep -rn "statistical" agents/football/` sebelum hapus |
| Config lama (dengan blok `evidence_floor`) tidak error kalau kode sudah dihapus — blok config diabaikan | Tidak masalah, tapi bersihkan biar konsisten |

**Keputusan akhir**: Opsi A (minimal) direkomendasikan — mudah di-revert lagi dan tidak menghapus fitur. Opsi B hanya jika memang ingin kembali ke perilaku lama secara penuh.
