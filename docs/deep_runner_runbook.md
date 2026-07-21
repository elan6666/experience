# iTransformer / FACT D0 Cell Runner

This entry point is server-only. It runs one receipt-bound V0/V1 cell, never
constructs a 2026/test loader during fitting, and never edits an author checkout.

## Frozen research contract

- Inputs: canonical weekly D0 snapshots, 96 weekly lookback steps.
- Target: one future-5-trading-day excess log return, entered at T+1 open.
- Fit: 2019-01-01 through 2024-12-31.
- Selection/validation: 2025-01-01 through 2025-12-31 only.
- Seeds: `20260719`, `20260720`, `20260721`.
- iTransformer: physical GPU 0 only.
- FACT: physical GPU 1 only, unmodified `core=0.5`.
- A0-A3 keep the same channel width, shared projector, author architecture,
  training parameters, and asset-token capacity. Only information values change.
- CSI300/STAR50 use fold-causal append-only identity. A 2025 identity unknown at
  the 2024 retrain cutoff remains in the complete PredictionFrame as
  `MODEL_UNSUPPORTED`.
- tech32/tech90 retain their disclosed retrospective 2026 selection and can
  only emit `EXPLORATORY_ONLY` manifests.

## Job manifest

`DeepJobSpec` is a canonical JSON contract. Every evidence field contains an
absolute server path and its exact SHA-256. Required evidence includes D0,
environment, upstream GPU integrity smoke, project-code receipt, model adapter
config, shared deep config, and (for CSI300/STAR50) the formal feature receipt.

`author_arguments` contains author architecture options only. The runner owns
and freezes `seq_len=96`, `pred_len=1`, asset widths, learning rate, epochs,
patience, and batch size. A production job should use the reviewed cell config;
do not invent parameters interactively. Recommended first bounded formal run is
30 epochs with patience 5, followed by the other two frozen seeds.

Output and checkpoint paths must end with:

```text
<model>/<universe-lower>/<A0|A1|A2|A3>/<seed>
```

## Server preflight

From `/data/yilangliu/a_share_research/seven_model_research`:

```bash
pwd
df -h /data/yilangliu
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
git -C /absolute/author/checkout status --porcelain --untracked-files=all
git -C /absolute/author/checkout rev-parse HEAD
```

The checkout must be detached/pinned and clean. The runner sets
`sys.dont_write_bytecode=True`, verifies clean state again after inference, and
fails rather than repairing upstream code.

## Launch the two GPU lanes

Use the model-specific compatible environment and expose exactly one physical
GPU to each process:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  .venv-itransformer/bin/python scripts/run_deep_cells.py \
  --job-spec /absolute/jobs/itransformer-csi300-a0-seed-20260719.json

CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv-fact/bin/python scripts/run_deep_cells.py \
  --job-spec /absolute/jobs/fact-csi300-a0-seed-20260719.json
```

The two lanes may run concurrently. Runs within one GPU lane should be queued
serially. Never reuse an output or checkpoint directory.

## Atomic artifacts

On success the runner atomically publishes:

- `predictions.json`: complete common PredictionFrame, including uncovered rows;
- `run_manifest.json`: validation-only RunManifest;
- `normalizer.json`: train-fold-only channel statistics;
- `provenance.json`: exact D0/table/config/code/environment/upstream hashes,
  architecture hash, parameter count, GPU, fit summary, and fidelity deviations;
- checkpoint directory containing `best.pt` selected by validation MSE.

Failures produce a typed `*.failure.json` and never masquerade as a negative
strategy result. Temporary directories and partial checkpoints must be audited
before manual removal and retry.

## Required verification after sync

Run only on the approved server:

```bash
.venv/bin/python -m pytest -q \
  tests/adapters tests/experiments/test_deep_runner.py
.venv/bin/ruff check \
  src/a_share_research/experiments/deep_runner.py \
  scripts/run_deep_cells.py tests/experiments/test_deep_runner.py
.venv/bin/python -m compileall -q src tests scripts/run_deep_cells.py
```

Then run one bounded synthetic/upstream GPU smoke before any real D0 cell. A
passing contract test is not a claimed research result; only a successful,
receipt-bound D0 run may publish predictions.
