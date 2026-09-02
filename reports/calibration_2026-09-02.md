# Calibration report (settled matches: 143 with 1X2 model+market)

## 1X2 (Brier / log-loss, lower is better; favourite hit-rate)

| probability | n | brier | logloss | fav-hit |
|---|---|---|---|---|
| model raw (pre-anchor) | 143 | 0.5254 | 0.9181 | 62% |
| model final (as logged) | 143 | 0.5254 | 0.9181 | 62% |
| market devig | 143 | 0.4834 | 0.8385 | 64% |
| 0.75 raw + 0.25 market | 143 | 0.5056 | 0.8754 | 64% |
| 0.50 raw + 0.50 market | 143 | 0.4920 | 0.8522 | 64% |
| 0.35 raw + 0.65 market | 143 | 0.4868 | 0.8439 | 63% |
| 0.25 raw + 0.75 market | 143 | 0.4846 | 0.8404 | 63% |
| both seeded: model raw | 64 | 0.5167 | 0.8857 | 62% |
| both seeded: market | 64 | 0.5171 | 0.8860 | 61% |
| one/none seeded: model raw | 79 | 0.5326 | 0.9443 | 62% |
| one/none seeded: market | 79 | 0.4561 | 0.8001 | 67% |

## 1X2 favourite hit-rate by model-vs-market gap (raw model)

| gap | n | hit |
|---|---|---|
| model >= market+3pp | 65 | 55% |
| within 3pp | 42 | 64% |
| model <= market-3pp | 36 | 72% |

## 1X2 favourite reliability (raw model bucket)

| bucket | n | hit | mean market |
|---|---|---|---|
| 0.3-0.4 | 8 | 38% | 38% |
| 0.4-0.5 | 36 | 42% | 45% |
| 0.5-0.6 | 27 | 70% | 53% |
| 0.6-0.7 | 28 | 71% | 61% |
| 0.7-0.8 | 32 | 78% | 62% |
| 0.8-0.9 | 12 | 58% | 58% |

## Over 2.5 (n=144)

| probability | brier | logloss |
|---|---|---|
| model raw | 0.2335 | 0.6590 |
| model final | 0.2335 | 0.6590 |
| market | 0.2122 | 0.6143 |
| 0.75 raw + 0.25 market | 0.2264 | 0.6443 |
| 0.50 raw + 0.50 market | 0.2205 | 0.6321 |
| 0.35 raw + 0.65 market | 0.2175 | 0.6258 |
| 0.25 raw + 0.75 market | 0.2158 | 0.6221 |

reliability Over 2.5 (raw model bucket): n / actual / mean market

| bucket | n | actual | market |
|---|---|---|---|
| 0.2-0.3 | 3 | 67% | 45% |
| 0.3-0.4 | 13 | 69% | 46% |
| 0.4-0.5 | 32 | 53% | 50% |
| 0.5-0.6 | 47 | 72% | 59% |
| 0.6-0.7 | 39 | 72% | 61% |
| 0.7-0.8 | 9 | 78% | 69% |
| 0.8-0.9 | 1 | 100% | 55% |

## BTTS Yes (n=129)

| probability | brier | logloss |
|---|---|---|
| model raw | 0.2520 | 0.6983 |
| model final | 0.2520 | 0.6983 |
| market | 0.2293 | 0.6512 |
| 0.75 raw + 0.25 market | 0.2432 | 0.6794 |
| 0.50 raw + 0.50 market | 0.2365 | 0.6657 |
| 0.35 raw + 0.65 market | 0.2335 | 0.6597 |
| 0.25 raw + 0.75 market | 0.2319 | 0.6565 |

reliability BTTS Yes (raw model bucket): n / actual / mean market

| bucket | n | actual | market |
|---|---|---|---|
| 0.2-0.3 | 17 | 35% | 50% |
| 0.3-0.4 | 14 | 64% | 51% |
| 0.4-0.5 | 25 | 76% | 52% |
| 0.5-0.6 | 41 | 54% | 54% |
| 0.6-0.7 | 27 | 67% | 59% |
| 0.7-0.8 | 5 | 40% | 62% |

## Weight fit (grid, components_1x2 rows: 0)

too few rows with persisted components (0 < 30) -- keep current weights

