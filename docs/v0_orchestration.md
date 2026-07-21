# V0/A0 orchestration contract

V0 is pre-registered as exactly 28 logical cells: seven named models across
CSI300, STAR50, tech32 and tech90.  The registry is Core-only (`A0`), predicts
the future five trading days at weekly frequency, and enters at the next
tradable open (`T+1_OPEN`).  Training is fixed to 2019--2024 and model selection
is fixed to 2025 validation.  The already viewed 2026-01-01 through 2026-07-17
interval is report-only and cannot select configurations.

The source blueprint is
`configs/experiments/v0/registry-v1.json`.  Expanding it hashes every declared
model config, adapter source/declaration and the upstream registry.  Its
`FrozenV0Registry.stable_hash()` is the pre-registration receipt.  It is not a
permission to execute.

## Two-stage evidence gate

1. `build_frozen_v0_registry` creates all 28 cells. Ridge, LightGBM,
   iTransformer and FACT are `RUNNABLE_PENDING_D0`. TimePro, TimeXer and S4M are
   `BLOCKED_LICENSE` and have no CPU/GPU lane.
2. `bind_runtime_evidence` accepts only exact server-produced SHA-256 values for
   the D0 manifest, asset registry, execution calendar, feature schema, shared
   CSI300 market state, environment/integrity receipts and license-gate
   receipts. A blocked-license cell stays blocked even if a caller supplies
   other evidence.

STAR50 inherits its D0 gate. Tech32 and tech90 may run only as exploratory
pools; later result contracts retain that classification. No execution unit is
created for a blocked cell.

## Isolation and accounting

- Ridge and LightGBM use the CPU lane with seed `20260719`.
- iTransformer uses GPU0 and FACT uses GPU1, each with seeds `20260719`,
  `20260720` and `20260721`.
- Each attempt has a unique run ID, output directory and checkpoint directory.
- `V0StatusTable` contains every expected cell/seed attempt. Terminal failures
  require a typed state and reason code; deleting or silently omitting a row
  fails validation.

This module only creates registries and execution descriptions. It does not
load data, import upstream models, train, infer, evaluate or write result files.
All such operations remain server-only under
`/data/yilangliu/a_share_research`.
