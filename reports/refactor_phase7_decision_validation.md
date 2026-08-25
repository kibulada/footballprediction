# Phase 7 — Decision-layer validation: the score does not rank bets

Method: strict chronological walk-forward over EPL 2022–26 (n = 1,520, the only
league with historical odds), scoring every match's top candidate with the SAME
7-component weighted score the live engine uses (`score_candidate` + `DEFAULT_WEIGHTS`),
then correlating that score with realized flat-stake ROI.

## Result

| metric | value |
|---|---|
| n scored | 1,520 |
| Spearman(decision score, realized ROI) | **0.0015** |
| mean score | 0.5654 |
| mean ROI (flat-stake on top-scored candidate) | **−7.6%** |

ROI by score tercile:

| score tercile | n | realized ROI |
|---|---|---|
| low | 506 | −3.4% |
| mid | 506 | −12.8% |
| high | 508 | −6.7% |

## Interpretation

- The Spearman correlation is effectively **zero**: a higher Decision Score does
  not predict higher realized ROI. The 7-component weighting has no demonstrated
  ranking power.
- Tercile ROI is **non-monotonic and negative everywhere** — the "high score"
  bucket does not beat the "low score" bucket, and no bucket is profitable.
- This is not a surprise given Phase 1: the model itself loses to the market
  (0/4 seasons), so no linear combination of its outputs can manufacture an edge.

## Scope limits (honest, not papered over)

- 1X2 only (historical caches carry a single 1X2 price, no totals/BTTS odds).
- ROI only — CLV is not measurable historically (no opening-vs-closing pair).
- EPL only (other leagues' caches have no odds).

## Action taken

The weights are a hand-set prior with zero validated contribution, so they are
marked in `config/football.json` as `weights_validated: false` with the null
result recorded. Betting decisions now ride the hard gates (edge/EV/sample/CI,
Phases 3 & 4) and the Kelly staking layer (Phase 6), not the 7-component score.
Re-running this validation on accumulating settled live data (Phase 8) is the
path to either re-weight or delete the score layer.
