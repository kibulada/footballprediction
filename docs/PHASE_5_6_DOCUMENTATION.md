# Phase 5-6 Documentation — Hermes Football Bot

**Tanggal**: 2026-08-20
**Model Version**: v3 (Final)
**Status**: Production

---

## Table of Contents

1. [Ringkasan Perubahan](#ringkasan-perubahan)
2. [Signal Engine Weights](#signal-engine-weights)
3. [Evidence Floor (P1-3)](#evidence-floor-p1-3)
4. [Statistical Component](#statistical-component)
5. [Market Intelligence](#market-intelligence)
6. [CLV Tracking](#clv-tracking)
7. [Completeness Adjustment (Option A)](#completeness-adjustment-option-a)
8. [Uncalibrated League Handling](#uncalibrated-league-handling)
9. [Timing Optimization](#timing-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Ringkasan Perubahan

### Phase 5.1-5.3: Core Signal Engine

| Komponen | Perubahan | dampak |
|---|---|---|
| **Evidence Floor (P1-3)** | Score cap 0.52 ketika statistical + movement unavailable | Mencegah score tinggi tanpa data |
| **Statistical** | Diaktifkan (weight 10%) | Form/H2H frequency masuk scoring |
| **Weights** | Model 35%, Market 20%, MI 15%, Movement 15%, Statistical 10%, DQ 5% | Rebalance prioritas |
| **AH Consensus** | Group by line sebelum median | Fix mixed data bug |
| **Evidence Floor** | Skip jika market support ada | Tidak cap score yang punya edge |

### Phase 5.4: Uncalibrated League

| Komponen | Perubahan | Dampak |
|---|---|---|
| **Score Cap** | Max 50/100 untuk uncalibrated | Liga non-top tidak misleading |
| **Safety Preference** | Under > BTTS > Over | Pilihan lebih aman |
| **Warning** | "LIGA TIDAK TERKALIBRASI" | User aware risiko |

### Phase 6: Market Intelligence

| Komponen | Perubahan | Dampak |
|---|---|---|
| **Steam Detector** | Deteksi >3% dalam 15 menit | Sharp money terdeteksi |
| **RLM** | Reverse Line Move detection | Public vs sharp money |
| **Multi-book** | 3+ bookmakers agree | Validasi pergerakan |
| **CLV Tracker** | Closing Line Value logging | Buktikan model menang vs market |
| **Market Intel Poll** | Background odds trend fetching | Data real-time |

### Phase 6.1: Completeness Adjustment

| Komponen | Perubahan | Dampak |
|---|---|---|
| **Option A** | Exclude disabled fields dari completeness | Field disabled tidak penalize |

---

## Signal Engine Weights

### Distribusi Bobot (v3 Final)

```
┌─────────────────────┬──────┬─────────────────────────────────────────────┐
│ Komponen            │ Bobot │ Fungsi                                      │
├─────────────────────┼──────┼─────────────────────────────────────────────┤
│ Model               │ 35%  │ Prediksi dari Elo+Poisson+calibration      │
│ Market (edge)       │ 20%  │ Edge model vs market odds                  │
│ Market Intelligence │ 15%  │ Steam/RLM/multi-book agreement             │
│ Movement            │ 15%  │ Opening → current price/line drift         │
│ Statistical         │ 10%  │ Form sequence, H2H frequency               │
│ Data Quality        │  5%  │ Input completeness                         │
│ Team Context        │  0%  │ Lineups/injuries (disabled)                │
│ Late Movement       │  0%  │ Penalty only (not scored)                  │
├─────────────────────┼──────┼─────────────────────────────────────────────┤
│ TOTAL               │ 100% │                                             │
└─────────────────────┴──────┴─────────────────────────────────────────────┘
```

### Confidence Thresholds

```
┌──────────────────┬─────────────┬─────────────────────────────────────────────┐
│ Label            │ Score Range │ Deskripsi                                  │
├──────────────────┼─────────────┼─────────────────────────────────────────────┤
│ VERY HIGH        │ ≥ 0.78      │ Sangat kuat, banyak konfirmasi             │
│ HIGH             │ ≥ 0.65      │ Kuat, beberapa konfirmasi                  │
│ MEDIUM           │ ≥ 0.52      │ Sedang, data cukup tapi kurang yakin       │
│ LOW              │ ≥ 0.40      │ Lemah, data terbatas                       │
│ NO BET           │ < 0.45      │ Tidak cukup bukti untuk bet                │
└──────────────────┴─────────────┴─────────────────────────────────────────────┘
```

### Contoh Perhitungan

```
Match: Atletico Madrid vs Malaga (La Liga)
Pick: Over 2.5 @ 1.53

Component Scores:
  model:              0.90 × 0.35 = 0.315  (model yakin 80%)
  market:             1.00 × 0.20 = 0.200  (edge +19.6pp = max)
  market_intelligence: 0.80 × 0.15 = 0.120  (steam terdeteksi)
  movement:           0.82 × 0.15 = 0.123  (1.3% toward)
  statistical:        0.80 × 0.10 = 0.080  (recent goals ada)
  data_quality:       0.75 × 0.05 = 0.038  (75% complete)

  Total:              0.876 = 88/100
  Confidence:         HIGH (≥ 0.65 threshold)
```

---

## Evidence Floor (P1-3)

### Apa itu Evidence Floor?

Evidence floor adalah safety mechanism yang **cap score** ketika key evidence unavailable.

### Logic

```python
def _apply_evidence_floor(score, components, cfg):
    # Market data IS evidence - don't floor when market exists
    if components.get("market", 0.0) > 0.0:
        return score  # NO floor when market supports
    
    has_stat = "statistical" in components
    has_mv = _movement_available(components.get("_movement_block"))
    missing = (not has_stat) + (not has_mv)
    
    if missing == 2:  # BOTH unavailable
        cap = 0.52    # MEDIUM floor
    elif missing == 1:  # ONE unavailable
        cap = 0.65    # MEDIUM upper
    else:
        return score  # Both available, no cap
    
    return min(score, cap)
```

### Thresholds (config: `models.signal_engine.evidence_floor`)

```json
{
  "evidence_floor": {
    "both_unavailable_cap": 0.52,
    "one_unavailable_cap": 0.65
  }
}
```

### Dampak

| Kondisi | Score Cap | Confidence Cap |
|---|---|---|
| Statistical + Movement available | No cap | No cap |
| Market support ada | No cap | No cap |
| Statistical unavailable saja | 0.65 | MEDIUM |
| Movement unavailable saja | 0.65 | MEDIUM |
| Keduanya unavailable | 0.52 | MEDIUM |

---

## Statistical Component

### Apa itu Statistical Component?

Komponen yang menghitung **empirical form/H2H frequencies** — seberapa sering tim menang/draw/kalah berdasarkan data historis.

### Data yang Digunakan

1. **Form Sequence**: Last 5-10 match results (W/D/L)
2. **H2H Frequency**: Head-to-head win/draw/loss ratios
3. **Recent Goals**: Average goals scored/conceded

### Bobot

- **10%** dari total score
- **0.5 (neutral)** jika data tidak tersedia
- **0.0 - 1.0** berdasarkan kualitas data

### Kapan Diaktifkan

```python
# Di signal_engine.py:score_signals()
if "statistical" in s.components:
    comps["statistical"] = round(s.components["statistical"], 3)
```

Statistical hanya diaktifkan jika:
1. Data form sequence tersedia (≥3 match)
2. Data H2H tersedia
3. Component value ≠ 0.5 (bukan neutral)

### Dampak pada Cardiff vs Wrexham

**Sebelum (statistical OFF)**:
```
BTTS YES: Score 52/100 (model only)
Over 2.5: Score 52/100 (model only)
→ BEST PICK: BTTS YES
```

**Sesudah (statistical ON)**:
```
Over 2.5: Score 58/100 (model + statistical + movement)
BTTS YES: Score 52/100 (model only)
→ BEST PICK: Over 2.5
```

**Catatan**: Match berakhir 1-1 (BTTS menang). Ini menunjukkan statistical component belum tentu improve prediction, tapi memberikan data lebih kaya untuk scoring.

---

## Market Intelligence

### Komponen Baru (Phase 6)

Market Intelligence adalah komponen baru yang mendeteksi:

1. **Steam Move**: Pergerakan odds > 3% dalam < 15 menit (tanda sharp money)
2. **Reverse Line Move (RLM)**: Odds berlawanan dengan public betting
3. **Multi-bookmaker Agreement**: 3+ bookmakers sepakat arah pergerakan

### Bobot

- **15%** dari total score
- **0.5 (neutral)** jika tidak ada data
- **0.0 - 1.0** berdasarkan kombinasi sinyal

### Steam Detector

```python
# agents/football/steam_detector.py

def detect_steam(trend_data, threshold_pct=3.0, window_minutes=15):
    """Deteksi steam move: >3% dalam <15 menit"""
    # Compare current odds vs odds N minutes ago
    # If change > threshold → STEAM detected

def detect_rlm(public_pct, odds_direction):
    """Deteksi Reverse Line Move"""
    # Public 75% di Home tapi Home odds NAIK → RLM
    # = Sharp money di Away

def multi_book_agreement(bookmaker_odds, threshold=3):
    """Validasi 3+ bookmakers sepakat"""
    # Count bookmakers dengan arah yang sama
    # Jika ≥ 3 → agreement
```

### Data Source

- **NowGoal**: `fetch_odds_trend()` — timestamped odds history
- **OddsPapi**: Current odds per bookmaker
- **The Odds API**: Historical odds (paid tier)

### Contoh Output

```
📊 Market Intel:
• Steam on home: +4.2% in 10min
• 4/5 books agree on home
• ✅ Market intel agrees with model
```

---

## CLV Tracking

### Apa itu CLV?

**Closing Line Value** = apakah model menang melawan market closing odds.

```
CLV = (closing_odds / entry_odds - 1) × 100

Contoh:
- Entry: Over 2.5 @ 1.90 (implied 52.6%)
- Closing: Over 2.5 @ 1.75 (implied 57.1%)
- CLV = (1.75 / 1.90 - 1) = -7.9%
- → KALAH melawan market (market bergerak MENENTANG kamu)

Contoh 2:
- Entry: Home -0.25 @ 1.85 (implied 54.1%)
- Closing: Home -0.25 @ 1.75 (implied 57.1%)
- CLV = (1.75 / 1.85 - 1) = -5.4%
- → KALAH melawan market
```

### CLV Metrics

| Metric | Target | Keterangan |
|---|---|---|
| **Average CLV** | > 0% | Positif = model menang vs market |
| **Positive CLV Rate** | > 55% | Lebih dari setengah pick menang vs market |
| **CLV per League** | Track | Identifikasi liga mana model kuat/lemah |
| **CLV per Market** | Track | Identifikasi AH/OU/BTTS mana profitable |

### Discord Commands

```
!football clv              # CLV dashboard
!football steam            # Steam alerts 24h
!football steam 48         # Steam alerts 48h
```

### CLV Dashboard Output

```
📊 CLV Dashboard — 15 picks analyzed
   Average CLV: +2.35%
   Positive CLV rate: 66.7%
   Positive: 10 | Negative: 5

By League:
  🟢 EPL: CLV +3.2% (8 picks, 75% positive)
  🔴 Serie A: CLV -1.1% (4 picks, 50% positive)
```

### CLV Gate (Auto-downgrade)

```python
def clv_gate(clv_history, min_positive_rate=0.50):
    """Auto-downgrade confidence jika CLV negatif konsisten"""
    if len(clv_history) < 5:
        return None  # Belum cukup data
    
    positive_rate = sum(1 for c in clv_history if c > 0) / len(clv_history)
    
    if positive_rate < min_positive_rate:
        return "CLV negative — model konsisten kalah vs market"
    return None
```

---

## Completeness Adjustment (Option A)

### Masalah

Sebelum Option A:
- `team_context` disabled (weight = 0%)
- Tapi `lineups/injuries` missing → completeness turun
- Coverage floor: completeness rendah → confidence di-downgrade ke MEDIUM
- **Inkonsisten**: field disabled tapi masih penalize

### Solusi

Option A: **Exclude disabled fields dari completeness calculation**.

```python
_CALIBRATION_WEIGHTS = {
    "model": 0.45,     # odds (0.25) + xG (0.20)
    "statistical": 0.55, # form (0.40) + H2H (0.15)
    "market": 0.0,     # derived dari odds
    "market_intelligence": 0.0,  # derived dari odds
    "movement": 0.0,   # derived dari odds snapshots
    "data_quality": 0.0, # meta-component
    "team_context": 0.0, # lineups/injuries (separate source)
}

def _adjust_completeness_for_weights(completeness, weights):
    total_active = sum(calib[k] for k,w in weights.items() if w > 0)
    ratio = total_active / sum(all_calib.values())
    return min(1.0, completeness / ratio)
```

### Dampak

| Skenario | Raw | Adjusted | Efek |
|---|---|---|---|
| Semua aktif (current) | 0.65 | **0.65** | Tidak berubah |
| statistical OFF | 0.65 | **1.00** | Naik! form+H2H tidak penalize |
| team_context ON (5%) | 0.65 | **0.65** | Tidak berubah |

### Coverage Floor Behavior

```python
def _coverage_floor(completeness, confidence, cfg):
    downgrade = cfg.get("downgrade_below", 0.40)
    low = cfg.get("low_below", 0.25)
    
    if completeness < low:
        return "LOW"
    if completeness < downgrade and confidence in ("VERY HIGH", "HIGH"):
        return "MEDIUM"
    return confidence
```

---

## Uncalibrated League Handling

### Score Cap

```python
UNCALIBRATED_SCORE_MAX = 0.50  # 50/100 cap

if not league_calibrated:
    score = min(score, UNCALIBRATED_SCORE_MAX)
```

### Safety Preference

```python
if not league_calibrated:
    if market == "Asian Handicap":
        score *= 1.05  # +5% boost (draw cover)
    elif market == "Total":
        if sel.startswith("under"):
            score = max(score, 0.48)  # floor 48/100
        else:
            # Over: edge-dependent penalty
            if edge < 5pp: score *= 0.90
            elif edge < 10pp: score *= 0.95
    elif market == "BTTS":
        score *= 0.95  # -5% (binary, no draw cover)
```

### Warning Display

```
⚠️ LIGA INI TIDAK TERKALIBRASI — model belum terbukti akurat untuk liga ini.
Rekomendasi: SKIP pick ini atau gunakan dengan potensi risiko tinggi.
```

### League Calibration Data

Source: `cache/football/calibration/`

| Liga | Status | Source |
|---|---|---|
| EPL | ✅ Calibrated | football-data.org |
| La Liga | ✅ Calibrated | football-data.org |
| Serie A | ✅ Calibrated | football-data.org |
| Bundesliga | ✅ Calibrated | football-data.org |
| Ligue 1 | ✅ Calibrated | football-data.org |
| Copa Libertadores | ❌ Uncalibrated | — |
| MLS | ❌ Uncalibrated | — |
| ASEAN Championship | ❌ Uncalibrated | — |

---

## Timing Optimization

### Odds Poll Schedule

```json
{
  "auto_odds_poll": {
    "schedule": [
      { "until_hours": 1, "interval_minutes": 5 },     // Last 1 hour
      { "until_hours": 24, "interval_minutes": 30 }     // 24h-1h before
    ]
  }
}
```

### Lineup Availability

- **Flashscore lineups**: Hanya di-fetch jika match ≤ 72 jam
- **Lineup trigger**: Check dalam 2 jam window sebelum kickoff
- **Confirmed lineup**: Biasanya 1 jam sebelum kickoff

### Rekomendasi Timing

| Timing | Kualitas | Risiko | Verdict |
|---|---|---|---|
| **-4 jam** | ⚠️ Lineups belum | Best pick bisa berubah | ❌ Terlalu awal |
| **-2 jam** | ✅ Predicted lineups | Best pick mungkin berubah | ⚠️ Bisa tapi belum final |
| **-1 jam** | ✅✅ Confirmed lineups | Market mulai volatile | ✅ **OPTIMAL** |
| **-30 menit** | ✅✅✅ Data paling lengkap | Budget analysis risk | ✅ **OPTIMAL** |
| **-15 menit** | ✅ Data final | Budget analysis pasti kehabisan | ⚠️ Terlalu mepet |

### Sweet Spot: -1 jam sampai -30 menit

**Kenapa:**
1. **Lineups**: Confirmed (1 jam sebelum kickoff,99% liga)
2. **Odds Snapshots**: Sudah12+ snapshots
3. **Movement**: Sudah terdeteksi dengan baik
4. **Steam Detection**: Cukup data untuk deteksi
5. **Budget**: Masih sisa untuk Flashscore browser render
6. **Closing Line**: Belum final tapi sudah representatif

---

## Troubleshooting

### Best Pick Berubah-ubah

**Penyebab:**
1. Odds berubah → edge berubah
2. Lineups keluar → model probabilitas berubah
3. Sharp money masuk → market bergerak signifikan
4. Statistical component → data fresh masuk

**Solusi:**
- Minta analisa **2 kali**: -2 jam (initial) + -30 menit (final)
- Atau tunggu **-1 jam** untuk data paling stabil

### Score 52/100 (Semua Signal Sama)

**Penyebab:**
- Liga tidak terkalibrasi (score cap 50/100)
- Evidence floor aktif (statistical + movement unavailable)
- Data quality rendah

**Solusi:**
- Cek apakah liga terkalibrasi
- Cek apakah ada market support (edge)
- Cek data completeness

### Confidence MEDIUM padahal Score 88/100

**Penyebab:**
- Coverage floor downgrade (completeness < 0.40)
- Odds disagreement caps confidence
- Late movement penalty

**Solusi:**
- Cek completeness (data_quality component)
- Cek odds_quality status
- Cek late_direction strength

### Movement: n/a (no opening prices)

**Penyebab:**
- Opening snapshot belum ter-pin
- Pertama kali analisa match ini
- Background odds poll belum jalan

**Solusi:**
- Tunggu 5-10 menit untuk odds poll berikutnya
- Atau analisa lagi nanti (opening akan ter-pin)

---

## File Reference

| File | Fungsi |
|---|---|
| `agents/football/signal_engine.py` | Core signal engine, weights, scoring |
| `agents/football/steam_detector.py` | Steam move + RLM detection |
| `agents/football/clv_tracker.py` | CLV tracking + dashboard |
| `agents/football/market_intel_poll.py` | Background odds trend fetching |
| `agents/football/format.py` | Discord card rendering |
| `agents/football/calibration.py` | League calibration + completeness |
| `agents/football/analyse.py` | Main analysis pipeline |
| `config/football.json` | Configuration (weights, thresholds) |

---

## Version History

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-08-17 | Initial (P1-3 evidence floor) |
| v2 | 2026-08-18 | Statistical ON, weight rebalance, market intelligence |
| v3 | 2026-08-20 | Completeness adjustment (Option A), CLV tracking |

---

**Generated with Codebuff 🤖**
**Co-Authored-By: Codebuff <noreply@codebuff.com>**
