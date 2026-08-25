# 🐛 Bug Report: 5 Match UCL — Aug 19, 2026

## Ringkasan

| Match | Result | Bot Pick | Outcome | Root Cause |
|---|---|---|---|---|
| Celtic vs LASK | 3-0 | Over 2.5 | ✅ Win | Legacy format (score=None) |
| Nijmegen vs Bodo/Glimt | 1-3 | Under 3.5 | ❌ Loss | Best pick override dari stability |
| Slovan vs Celje | 1-1 | Home -0.5 | ❌ Loss | Model salah prediksi winner |
| Hapoel vs Sabah | 2-1 | Under 2.5 | ❌ Loss | Lambda vs Model contradiction |
| Atletico vs Malaga | 2-0 | N/A | ❌ Loss | Match ID tidak match |

---

## 🐛 BUG #1: Lambda vs Model 1X2 Contradiction (CRITICAL)

**Match:** Hapoel Beer Sheva vs Sabah

**Data:**
```
Elo: Hapoel 1844 > Sabah 1756 (Home lebih kuat)
Model 1X2: Home 55.9% (Home menang)
Lambda: Home 0.728 < Away 1.526 (Away cetak lebih banyak gol!)
```

**Masalah:**
- Poisson lambdas bilang **Away cetak lebih banyak** (1.526 vs 0.728)
- Tapi Model 1X2 bilang **Home menang** (55.9%)
- Best pick jadi **Under 2.5** (total lambda = 2.254 < 2.5)
- Padahal match berakhir **2-1 (3 goals, Over)**

**Root Cause:**
Lambda dihitung dari **form features** (recent goals), bukan dari elo:
```python
# models.py line 232-233
lh = self.base_home * math.sqrt(ha * ad)  # attack/defense dari form
la = self.base_away * math.sqrt(aa * hd)
```

Form Hapoel: L-W-W-W-L → mungkin cetak sedikit gol recently
Form Sabah: W-W-L-W-W → mungkin cetak banyak gol recently

Jadi lambda Home rendah (0.728) karena form Hapoel jelek, tapi elo tetap tinggi (1844).

**Dampak:** Model punya 2 prediksi yang kontradiktif:
1. 1X2: Home menang (dari elo/combined)
2. Poisson: Away cetak lebih banyak (dari form)

---

## 🐛 BUG #2: Best Pick Score = None (Legacy Format)

**Match:** Celtic vs LASK, Nijmegen vs Bodo/Glimt

**Data:**
```
Best Pick: Over 2.5 (score=None, conf=None)
Signal Engine: Over 2.5 (score=65.5, conf=MEDIUM)
```

**Masalah:**
- `best_pick.score` = None (bukan 0, tapi None)
- `best_pick.confidence` = None
- Tapi `signal_engine_ranking[0].score` = 65.5

**Root Cause:**
Dua format `best_pick` berbeda:
1. **Legacy** (dari `decision.py`): `rank`, `market`, `selection`, `model_prob`, `edge`, `ev`, `grade`
2. **Baru** (dari `signal_engine.py`): `market`, `selection`, `score`, `confidence`, `components`, `movement`

Ketika fallback ke `decision.py`, format legacy dipakai tanpa `score`/`confidence`.

**Dampak:**
- Card显示 `Score: 0/100` atau `Score: N/A`
- User tidak tahu confidence sebenarnya

---

## 🐛 BUG #3: Best Pick Override oleh Stability Layer

**Match:** Nijmegen vs Bodo/Glimt

**Data:**
```
Snapshot 16:36: BP=Over 2.5 (score=0.677)
Snapshot 17:25: BP=Over 2.5 (score=0.677)
Snapshot 18:46: BP=Over 2.5 (score=None)
Snapshot 19:01: BP=Under 3.5 (score=None) ← OVERRIDE!
```

**Masalah:**
- Signal engine selalu pilih Over 2.5 (score 47.8-67.7)
- Tapi best pick diubah ke Under 3.5 di snapshot terakhir
- Under 3.5 = LOSS (1-3 = 4 goals > 3.5)

**Root Cause:**
`apply_pick_stability()` atau `decision.py` override signal engine's pick.

**Dampak:**
- Pick yang sebenarnya menang (Over 2.5) diubah jadi kalah (Under 3.5)

---

## 🐛 BUG #4: Atletico Madrid Missing Snapshot

**Match:** Atl. Madrid vs Malaga

**Data:**
```
Snapshot count: 0 (tidak ditemukan)
```

**Masalah:**
- Tidak ada snapshot untuk Atletico vs Malaga di Aug 19
- Padahal user bilang bot analisa match ini

**Root Cause:**
Match ID format berbeda:
- Prediksi pakai: `LaLiga||Atlético Madrid||CD Málaga||2026-08-19`
- Tapi script analisa cari: `LaLiga||Atl. Madrid||Malaga||2026-08-19`

**Dampak:**
- Tidak bisa evaluate performa bot untuk match ini

---

## 🐛 BUG #5: Slovan vs Celje — Model Salah Prediksi Winner

**Match:** Slovan Bratislava vs Celje

**Data:**
```
Model 1X2: Home 69.3% (yakin Home menang)
Lambda: Home 1.558 > Away 0.726 (Home cetak lebih banyak)
Actual: 1-1 (Draw)
Best Pick: Home -0.5 → LOSS
```

**Masalah:**
- Model sangat yakin Home menang (69.3%)
- Lambda juga support Home (1.558 > 0.726)
- Tapi match berakhir Draw

**Root Cause:**
Ini bukan bug, tapi **model limitation**. UCL qualification matches lebih unpredictable.

**Dampak:**
- Home -0.5 pada draw = half loss

---

## 📊 Rekomendasi Perbaikan

### P0 (Critical)
1. **Fix Lambda vs Model Contradiction** — Gunakan elo lambda sebagai primary, form sebagai adjustment
2. **Fix Legacy best_pick format** — Selalu sertakan `score` dan `confidence`

### P1 (High)
3. **Investigate stability layer override** — Kenapa best pick diubah dari Over 2.5 ke Under 3.5?
4. **Fix Atletico match ID** — Pastikan format konsisten

### P2 (Medium)
5. **Tambah confidence threshold** — Hanya tampilkan picks dengan confidence HIGH+
6. **Backtest 100 match** — Verify win rate setelah fixes
