# Phase 4 — Sample-size / CI tightening: before/after

The flat `min_bucket_n = 30` gate is replaced by a two-part rule, config-driven
(`models.decision.min_bucket_n = 200`, `models.decision.min_bucket_ci_halfwidth =
0.03`):

1. `n_bucket >= 200` (raised from 30), AND
2. the bucket's 95% binomial CI half-width on its realized rate `<= 3pp`.

A bucket that fails either is `INSUFFICIENT_SAMPLE` → no actionable tier.

## Before/after on the current `calibration_buckets.json`

| bucket | n | predicted | actual | gap | CI half | old (n≥30) | new (n≥200 ∧ CI≤3pp) |
|---|---|---|---|---|---|---|---|
| [0.00,0.50) | 14425 | 0.2775 | 0.2790 | −0.0015 | 0.007 | ✅ | ✅ |
| [0.50,0.55) | 898 | 0.5236 | 0.5312 | −0.0076 | 0.033 | ✅ | ❌ |
| [0.55,0.60) | 665 | 0.5741 | 0.5263 | **+0.0478** | 0.038 | ✅ | ❌ |
| [0.60,0.65) | 572 | 0.6250 | 0.6171 | +0.0079 | 0.040 | ✅ | ❌ |
| [0.65,0.70) | 411 | 0.6738 | 0.6886 | −0.0148 | 0.045 | ✅ | ❌ |
| [0.70,0.75) | 252 | 0.7249 | 0.7302 | −0.0053 | 0.055 | ✅ | ❌ |
| [0.75,0.80) | 121 | 0.7704 | 0.7851 | −0.0147 | 0.073 | ✅ | ❌ |
| [0.80,0.90) | 32 | 0.8188 | 0.8125 | +0.0063 | 0.135 | ✅ | ❌ |
| [0.90,1.00) | 0 | — | — | — | — | ❌ | ❌ |

**Result: 8/9 buckets passed the old gate; 1/9 pass the new gate.**

The only surviving bucket is the underdog sweep [0.00,0.50) — which
`best_prob_only` never credits with market value anyway. Every playable
bucket (0.50–0.90) fails on CI precision: their ±3.3–13.5pp confidence bands
are wider than the 3–20pp edge band the bot considers bettable.

Notable: the [0.55,0.60) bucket is the worst-calibrated (predicted 0.574 vs
actual 0.526, a +4.8pp overconfidence) sitting exactly in the favourite range.
Under the old n≥30 gate this bucket was treated as sufficient.

## Consequence

The engine now demotes the bettable range to `INSUFFICIENT_SAMPLE` until the
bucket table accumulates enough settled data to shrink the CI below 3pp. This
is the intended, honest outcome: with the current data the bot has no
statistically certifiable HIGH/GOOD/STRONG tier — which is exactly what the
walk-forward (Phase 1: model loses to market in 4/4 EPL seasons) already said.
