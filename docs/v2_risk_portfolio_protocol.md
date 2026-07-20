# V2 B1/B2 static protocol

This module converts frozen V1 Top10% intents into auditable strategy layers. It
does not fit predictors, change scores, optimize against 2026 results, or
implement B3 fusion.

## B1: one shared market-risk schedule

`RiskBudgetPolicy` is calibrated no later than 2025-12-31 from the independent
CSI300 `SharedMarketState`. The resulting `RiskBudgetSchedule` contains exactly
100%, 60%, 30%, or 0% equity for every signal date.

The policy's `market_state_hash` anchors the frozen calibration snapshot. The
schedule's `market_state_hash` anchors the scoring table, which may append
genuinely prospective rows after the policy is frozen. They are intentionally
not required to be equal: requiring equality would mutate the frozen policy
whenever a new live row arrives. The named feature interface and CSI300-only
state contract remain fixed, and missing features fail closed.

The schedule has no model, universe, score-coverage, or selected-stock input.
Every model and all four universes must reference the same schedule hash and
dated values. `apply_shared_risk_budget` accepts only an always-full B0 frame:

- a positive budget scales the frozen B0 securities proportionally and cannot
  silently return all cash;
- a zero budget removes all equity targets and sets cash to exactly 100%;
- a missing shared-state date, feature, or hash fails closed.

The runtime policy config remains `CALIBRATION_REQUIRED_BEFORE_RUN` until D0 and
the V1 selection boundary are frozen. Null thresholds are deliberately not
runnable defaults.

## B2: constraints before the common execution engine

`constrain_target_frame` reads the B1 target frame, PIT industry, signal-time
ADV and reference NAV. It never reads predictions or changes the selected-stock
ordering. The deterministic sequence is:

1. single-stock cap;
2. industry cap;
3. ADV participation cap on weight change;
4. one-way turnover cap;
5. residual capital to cash.

The output remains a `TargetFrame`, then `build_b2_constrained_ledger` calls the
existing `build_b0_ledger`. Therefore B0 and B2 share the trusted trading
calendar, T+1 next-open price, buy/sell eligibility, costs, carried positions,
and ledger accounting.

When transaction caps conflict with a reduced desired allocation, the prior
weight can be carried temporarily. The dated fallback is explicitly
`CARRY_DUE_TO_TRANSACTION_CAP`; it is not presented as satisfying the new
exposure cap. A 0% risk-budget target bypasses soft ADV/turnover limits and
requests full liquidation. Only exchange ineligibility such as suspension or a
price limit may leave an executed position, and that difference is exposed as
a rejected fill.

`TradingRestriction` adds the detailed reason (`LIMIT_UP`, `LIMIT_DOWN`,
`SUSPENDED`, or `ST_NOT_ELIGIBLE`) only when the authoritative D0 side-specific
eligibility already rejects the order. It cannot make an eligible side
ineligible by itself; conflicting or unanchored detailed restrictions fail
closed even when no order happens to be emitted on that side.

## Required evidence

Every B2 result contains:

- input and output target hashes;
- risk-schedule and constraint-policy hashes;
- per-security input, previous and constrained weights, industry, ADV delta cap
  and applied reasons;
- dated fallback status;
- target versus executed equity weight and closing cash weight;
- executed fills, rejects and detailed reject reasons;
- gross traded value, total cost and turnover;
- the canonical `PortfolioLedger` hash.

Limits in `configs/portfolio/v2/constraints-v1.json` remain null until frozen by
the research lead with dated sources. Synthetic tests instantiate explicit
values only to prove branches and determinism.

## B3 gate only

Plan011 static scope does not implement fusion. `evaluate_fusion_gate` returns
`ELIGIBLE` only when at least two models passed V1 and at least one pair is below
both validation-frozen error-correlation and holding-overlap thresholds. It
returns explicit `NOT_RUN` otherwise. No test or viewed-2026 observation may set
these thresholds or fusion weights.

## Server verification

After source sync to `/data/yilangliu/a_share_research/seven_model_research`:

```text
.venv/bin/python -m pytest -q tests/portfolio/test_v2_risk_budget.py tests/portfolio/test_v2_constraints.py tests/fusion/test_b3_gate_contract.py
.venv/bin/ruff check src/a_share_research/portfolio tests/portfolio tests/fusion
.venv/bin/python -m compileall -q src/a_share_research/portfolio tests/portfolio tests/fusion
```

These are synthetic contract checks. They do not constitute a data, training,
backtest, or performance result.
