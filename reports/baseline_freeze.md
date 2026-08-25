# Baseline Freeze — EPL 2022-2026 (Walk-Forward)

Tanggal freeze: 2026-08-12
Command reproduce:
```
python -m agents.football.validate --fixtures cache/football/epl_fixtures_2022_2026.json \
    --out reports/baseline_freeze_epl.json
```
Dataset: EPL 2022-2023 .. 2025-2026 (1.520 match), satu pass kronologis
(kickoff order), state Elo/form menyebrang musim. Anti-leakage: hanya fitur
pre-match. Calibration: fit in-sample 4.560 pasangan, a=0.0082, b=1.0129,
raw ECE 0.011 → calibrated ECE 0.0104.

## Aggregat (1.520 match)

| Model | Log Loss | Brier | ECE | Hit% | ROI (flat, edge≥2%) | bets |
|---|---|---|---|---|---|---|
| baseline (base-rate) | 1.0971 | 0.6495 | 0.0196 | 44.4% | −7.5% | 805 |
| elo | 0.9914 | 0.5921 | 0.0205 | 52.3% | −2.7% | 882 |
| poisson | 1.0341 | 0.6213 | 0.0157 | 49.2% | −0.5% | 765 |
| dc (Dixon-Coles) | 1.0363 | 0.6223 | 0.0174 | 48.9% | −0.1% | 740 |
| **ensemble (elo+poisson)** | **0.9886** | **0.5901** | **0.0110** | **53.1%** | **−1.9%** | 623 |
| market (benchmark) | 0.9652 | 0.5739 | 0.0084 | 54.5% | — | 0 |

## Konsistensi per musim (log-loss)

- Setiap model mengalahkan baseline di 4/4 musim (elo, poisson, dc, ensemble).
- **Tidak ada model yang mengalahkan MARKET di musim mana pun** (0/4) —
  market tetap benchmark yang belum terkalahkan.
- Poisson/DC skip match tanpa fitur form pre-match (10 → 1 match/musim).

## Fakta kunci untuk regression reference

1. Ensemble adalah model probabilitas terkuat (LL 0.9886, hit 53.1%),
   tetapi ROI flat-stake −1.9% → edge ≥2% belum profit (semua model negatif).
2. Dixon-Coles ≈ Poisson (LL 1.0363 vs 1.0341) — belum terbukti menambah.
3. Market adalah benchmark yang tidak boleh diklaim dikalahkan.
4. Semua metrik di atas adalah reference; perubahan apa pun harus diukur
   terhadap angka ini (out-of-sample, bukan in-sample).
