# Phase 1 — Diagnostic: where the model does/doesn't carry information

Date: 2026-08-15
Method: strict chronological walk-forward replay of the 4 leagues that have
historical results caches (`cache/football/multileague_fixtures.json`,
5,792 matches), using the existing `validate.py` harness. Market benchmark =
margin-free implied probabilities from historical 1X2 odds.

## Availability of validation data (the first finding)

| Segment | Historical data present? |
|---|---|
| 1X2, EPL | YES — 1,520 matches **with real 1X2 odds** (market benchmark + ROI) |
| 1X2, LaLiga / Serie A / Bundesliga | Results only, **0 rows carry odds** → no market benchmark, no ROI |
| Over/Under 2.5 / 3.5, BTTS | NO historical validation at all (harness is 1X2-only; no totals odds in data) |
| Decision tier (STRONG/GOOD/LEAN/WATCH) | NO — live `predictions.jsonl` has 131 snapshots but only **1 settle** |

So the only segment where the model's edge can actually be measured against
the market is **EPL 1X2**. Every other league/market/tier is currently an
unmeasured claim.

## Segment table (aggregate over all seasons)

| League | n | model logloss (ensemble) | market logloss | delta (model − market) | flat-stake ROI |
|---|---|---|---|---|---|
| EPL | 1,520 | 0.9886 | 0.9652 | **+0.0234 (model worse)** | −1.9% |
| LaLiga | 1,520 | 0.9975 | n/a (no odds) | — | n/a |
| Serie A | 1,521 | 1.0033 | n/a | — | n/a |
| Bundesliga | 1,231 | 1.0036 | n/a | — | n/a |

Entropy floors: 1X2 = ln(3) ≈ 1.0986. The model sits at 0.99–1.00, i.e. it
extracts ~0.10 nats above a flat guess — but on the ONLY league where the
market is measurable it is 0.023 nats **worse** than the market.

EPL flat-stake ROI (best 1X2 pick, margin-free edge ≥ 2%): ensemble −1.9%,
elo −2.7%, poisson −0.5%, dc −0.1%. All negative over 1,520 bets.

## Consistency (the decisive number)

For EPL, `beats_market_seasons = 0/4` for **every** model (baseline, elo,
poisson, dc, ensemble) in every season 2022–2026. The model has never once
beaten the market on logloss in any season where the comparison exists.

## What this determines for the rest of the refactor

1. **Phase 2 (edge benchmark)** is confirmed necessary: the only measurable
   comparison (EPL) shows the model losing to the market, so any positive
   "edge" reported elsewhere is either (a) vs a soft line or (b) an
   unmeasured league — both illusory.
2. **Phase 5 (per-league calibration)** is confirmed: the EPL-fitted
   calibrator is applied to leagues where no market/edge validation exists.
3. **Phases 3/4 (CLV gate + sample-size)** are confirmed: the one league
   with ROI data is negative, and no tier-level settled data exists yet, so
   the HIGH/GOOD tiers currently certify nothing.
4. **Market/O-U/BTTS/tier diagnostics cannot be produced from existing data.**
   They must be built forward (Phase 8 paper-trading), not back-filled.
