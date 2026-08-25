# Phase 8 — Paper-trade harness (go-live gate)

Deliverable is the harness + the go-live rule, not (yet) the 1,000 settled bets
— that accumulates via the existing auto-settle loop.

## What was built

1. **Strict go-live gate** (`clv_gate.gate_segment`, `require_roi_positive=True`):
   a segment (league × market × decision tier) graduates to actionable only when
   it has `>= min_bets` settled bets **AND** realized price CLV > 0 **AND**
   realized flat-stake ROI > 0. Positive ROI with negative CLV is variance, not
   skill — blocked. This is wired into every live decision via
   `run_decision_engine`, so an ungraduated segment can never emit STRONG/GOOD/
   LEAN/WATCH; it is demoted to NO BET (or MARKET PRIOR on thin data) with the
   reason logged and shown in Discord.

2. **`runner paper-trade` mode**: emits the per-segment graduation table
   (n, ROI, price CLV, graduates, reason) so the operator can see exactly which
   segments are still in paper mode and why.

## Current state (honest)

```
python -m agents.football.runner paper-trade
-> n_segments: 0, n_graduated: 0
```

The live log has only 1 settled match, so **zero segments qualify** — which is
the correct, safe state. The engine now refuses to recommend any bet until a
segment accumulates ≥200 settled bets with positive ROI **and** positive CLV.

## How it graduates over time

The auto-settle loop (`bot.py`) already appends settlements + Elo updates every
`auto_settle.interval_hours`. As settlements accumulate, `segment_clv_stats`
(backed by `_settled_records`) recomputes per-segment ROI + price CLV, and the
gate re-evaluates on every decision. A segment does NOT "graduate permanently" —
it is re-checked against the full (growing) sample each time, so an early lucky
streak cannot lock in go-live status, and a later negative-CLV drift demotes it
again.

## Remaining requirement

1,000 paper-traded settled bets, segmented by league × market × tier, with
positive ROI **and** positive CLV before real-money staking. This is a time-gated
requirement (a season+ of live data), not a code gap. The harness enforces it
automatically; nothing to manually flip.
