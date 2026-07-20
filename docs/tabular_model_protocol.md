# Ridge and LightGBM causal adapter protocol

## Boundary

Both adapters consume the same frozen `Core + F + F-missing + S` layout. A0-A3
never change the column count or order. Disabled groups become constants before
any statistics are fitted. Row-level membership, observation and history masks
remain outside the model matrix and always control prediction coverage.

Each F value has a separate missing indicator. When F is disabled, both the F
value and its missing indicator are zero-gated; otherwise A0/A2 could infer
fundamental availability without receiving fundamental values.

## Causal preprocessing

The preprocessor rejects fit rows after its declared cutoff. Winsor bounds,
medians, neutralization coefficients, means and scales are learned from that
training fold only and serialized in a content-addressed state. Industry and
size neutralization is active only when F is enabled. Prediction rows are never
used to update this state.

## Package semantics

- Ridge calls `sklearn.linear_model.Ridge` directly.
- LightGBM calls `lightgbm.LGBMRegressor` directly. The Python default has no
  implicit early stopping. The checked-in research config explicitly requests
  a validation-only early-stopping callback; this is a project selection rule,
  not a claimed LightGBM default.
- Neither adapter changes the regression loss based on portfolio results.

Both adapters export a common `PredictionFrame`. The caller supplies data,
calendar, registry, market-state, config and code hashes in a draft
`RunManifest`; `complete_run_manifest` only binds the prediction hash and an
explicit result state after matching model, information set, layout, config,
training-data hash and seed.

## Server verification

Run only on `/data/yilangliu/a_share_research/seven_model_research`:

```bash
.venv/bin/python -m pip install -e '.[tabular,dev]'
.venv/bin/python -m pytest -q tests/models/tabular
.venv/bin/ruff check src/a_share_research/models/tabular tests/models/tabular
.venv/bin/python -m compileall -q src/a_share_research/models/tabular tests/models/tabular
```

The first D0 smoke must use a bounded TRAIN/VALIDATION slice for each universe,
write diagnostics and manifests to server-only artifacts, and avoid
`LEGACY_VIEWED` and `FUTURE_UNSEEN` labels. A formal smoke command is deferred
until Plan003 publishes its D0 manifest and loader; no legacy loader is allowed
as a substitute.
