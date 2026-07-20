# Ridge / LightGBM sealed runner runbook

## Scope and invariants

This runner executes only the CPU tabular cells. V0 accepts A0; V1 accepts A1,
A2 or A3 and treats A0 as the already frozen V0 reference. Every job is fixed
to weekly future-5-day excess-return prediction with T+1 open entry,
2019-01-01 through 2024-12-31 fitting, and 2025-01-01 through 2025-12-31
validation/selection. The loader call is explicitly capped at 2025-12-31;
legacy-viewed 2026 rows fail closed.

The job manifest binds exact D0, environment, code, model-config and layout
files by absolute path and SHA-256. It also binds the D0 content/table hashes,
the causal append-only validation asset registry, shared CSI300 market state,
cell config, estimator package pin and (for formal universes) the formal D0
feature-eligibility receipt hash. Tech32 and tech100 remain
`EXPLORATORY_ONLY` regardless of model output.

A successful run publishes one output directory atomically. It contains:

- `predictions.json`: complete 2025 panel, including uncovered rows;
- `diagnostics.json`: fold/config/preprocessing/coverage diagnostics;
- `preprocessing_state.json`: training-only fitted preprocessing receipt;
- `run_manifest.json`: common receipt-bound `RunManifest`;
- `run_receipt.json`: exact inputs, hashes, protocol and coverage counts.

An unsuccessful attempt never publishes the final output directory. It raises a
typed `INVALID_DATA`, `INVALID_PROTOCOL`, `ADAPTER_FAIL`, `TRAIN_FAIL` or
`EVAL_FAIL` and, when the job path itself is valid, writes a sibling immutable
`<run_id>.failure.json`. A remaining `<run_id>.tmp` is evidence to inspect; do
not delete it automatically or reuse the same run id.

## Server-only preflight

Run only under the approved server root:

```bash
cd /data/yilangliu/a_share_research/seven_model_research
test "$PWD" = /data/yilangliu/a_share_research/seven_model_research
df -h /data/yilangliu/a_share_research
.venv/bin/python -m pip install -e '.[tabular,dev]'
```

Do not hand-build job JSON. The generator computes each `EvidenceFile.sha256`
on the server and rejects any input/output outside the approved research root.
It obtains `asset_registry_hash` by asking the canonical loader for the
complete 2025 panel and verifying that its permanent identity is append-only;
it never hashes a model's scoreable subset or a 2026 full-universe list. Do not
place the Tushare credential or raw data in a job manifest.

Formal CSI300 and STAR50 cells require an existing, D0-anchored
`FormalFeatureManifest` for every active information set. The receipt must:

- reference the final `D0Manifest.content_hash`;
- contain exactly the Core/F/F-missing/S inputs active for that A0-A3 gate;
- mark every admitted feature formally eligible;
- exist as a server file whose exact SHA-256 is recorded in the generation
  receipt.

The current `D0Manifest` schema does **not** itself contain those formal
receipts. Therefore each affected CSI300/STAR50 cell is recorded as a typed
`FORMAL_RECEIPT_MISSING` block until the D0 pipeline has materialized and
independently reviewed its receipt. Never substitute a gate hash,
feature-schema hash, arbitrary 64-character string, or a receipt derived after
seeing model results. A blocked formal cell does not prevent the other
universes from being generated. Tech32/tech100 do not receive formal receipts
and remain `EXPLORATORY_ONLY`.

Likewise, `BLOCKED`/`INVALID_DATA` D0 gates or rejected canonical table hashes
produce explicit `D0_GATE_BLOCKED` records for that universe only. For
example, a blocked STAR50 gate yields six runnable V0 jobs plus two STAR50
blocks, or eighteen runnable V1 jobs plus six STAR50 blocks. The generation
receipt always accounts for all 8 V0 or 24 V1 planned cells as either runnable
or blocked; nothing is silently skipped.

## Generate the fixed job matrices

Generate the V0 8-cell plan (A0 only) after final D0 exists:

```bash
.venv/bin/python scripts/generate_tabular_job_manifests.py \
  --phase V0 \
  --d0-manifest /data/yilangliu/a_share_research/data/manifests/d0-v1.json \
  --canonical-root /data/yilangliu/a_share_research/data/canonical/d0-v1 \
  --environment-receipt ridge=/data/yilangliu/a_share_research/receipts/upstream_v2/ridge_baseline_env_20260719.json \
  --environment-receipt lightgbm=/data/yilangliu/a_share_research/receipts/upstream_v2/lightgbm_baseline_env_20260719.json \
  --code-receipt /data/yilangliu/a_share_research/receipts/source/source-manifest-v1.json \
  --model-config ridge=/data/yilangliu/a_share_research/seven_model_research/configs/models/ridge-v1.json \
  --model-config lightgbm=/data/yilangliu/a_share_research/seven_model_research/configs/models/lightgbm-v1.json \
  --layout-config /data/yilangliu/a_share_research/seven_model_research/configs/features/tabular-layout-v1.json \
  --formal-feature-receipt CSI300:A0=/data/yilangliu/a_share_research/receipts/d0/formal-csi300-a0.json \
  --formal-feature-receipt STAR50:A0=/data/yilangliu/a_share_research/receipts/d0/formal-star50-a0.json \
  --run-root /data/yilangliu/a_share_research/runs \
  --job-root /data/yilangliu/a_share_research/runtime/jobs \
  --queue-root /data/yilangliu/a_share_research/runtime/queues
```

For V1, use `--phase V1` and supply CSI300/STAR50 receipts for A1, A2 and
A3. V1 plans exactly 24 cells and never retrains A0; V0 plans exactly 8 cells.
Only runnable cells enter queues, in deterministic chunks of at most 16. Every
queue is CPU serial. `generation_receipt.json` lists planned, runnable and
blocked counts plus every typed blocked-cell record. Generation refuses to
overwrite an existing job or queue directory. Inspect a `.tmp` directory
rather than deleting it blindly.

The normalized `cell_config_hash` binds phase, model, universe, formal versus
exploratory scope, A0-A3 gate, frozen 2019-2024/2025 protocol, seed, upstream
pin, D0 content, 2025 registry, formal receipt hash and the exact D0,
environment, source-code, model-config and layout file evidence. Any byte
change produces a different cell hash.

## Execute exactly one cell

```bash
.venv/bin/python scripts/run_tabular_cells.py \
  --job-spec /data/yilangliu/a_share_research/runtime/jobs/<run_id>.json
```

The JSON must be the canonical `TabularJobSpec.to_dict()` representation,
including `_schema: tabular_job_spec` and `_version: 1.0`. Do not hand-edit a
job after its cell hash is frozen.

## Execute a bounded CPU queue

```bash
.venv/bin/python scripts/run_tabular_cells.py \
  --queue-manifest /data/yilangliu/a_share_research/runtime/queues/<queue_id>.json
```

The queue is canonical `TabularQueueManifest` JSON. It contains 1-16 explicit,
unique jobs with disjoint output directories. Execution is serial, preserves
manifest order, performs no job discovery and has no hidden retry. A failed
cell stops the queue; inspect its typed failure receipt and create a new run id
only after the underlying evidence or code is deliberately changed.

## Verification before any comparison

```bash
.venv/bin/python -m pytest -q tests/models/tabular tests/experiments/test_tabular_runner.py
.venv/bin/python -m pytest -q tests/experiments/test_generate_tabular_jobs.py
.venv/bin/ruff check \
  src/a_share_research/models/tabular \
  src/a_share_research/experiments/tabular_runner.py \
  src/a_share_research/experiments/tabular_job_generator.py \
  tests/models/tabular tests/experiments/test_tabular_runner.py \
  tests/experiments/test_generate_tabular_jobs.py \
  scripts/run_tabular_cells.py scripts/generate_tabular_job_manifests.py
.venv/bin/python -m compileall -q \
  src/a_share_research/models/tabular \
  src/a_share_research/experiments/tabular_runner.py \
  src/a_share_research/experiments/tabular_job_generator.py \
  tests/models/tabular tests/experiments/test_tabular_runner.py \
  tests/experiments/test_generate_tabular_jobs.py \
  scripts/run_tabular_cells.py scripts/generate_tabular_job_manifests.py
```

Then verify that each prediction key and coverage state exactly matches the D0
validation panel, every hash in `run_receipt.json` resolves to the admitted
input or output, and no signal date exceeds 2025-12-31. Portfolio construction
and V2 evaluation are downstream and must not alter these model scores.
