# Deep A-share adapter contract

This layer is deliberately outside every author checkout. It creates a causal
stock identity, packs the canonical PIT panel, invokes an injected pinned author
model, and exports complete model-independent predictions. It does not contain a
copy or rewrite of an author's network.

## Identity and coverage

Each walk-forward retrain creates a `CausalAssetMaster` from membership evidence
known by that retrain cutoff. Existing slots are immutable. A later master may
append newly known stocks, but neither a loader nor a trading date may sort or
compact the slots. A member absent from the model's frozen master receives
`MODEL_UNSUPPORTED` until a causal retrain; it is not silently mapped into another
stock's channel.

## A0-A3 packing

All four information gates use the same ordered channel catalog:

`Core | F | F-missing | S`

Inactive channels are zeroed only after train-fold preprocessing. They are not
deleted, so input shape, author parameter count and training configuration remain
constant across A0-A3. A0 activates Core, A1 Core/F/F-missing, A2 Core/S and A3
all channels. Per-factor missing markers remain independent. Member, observed,
label, execution, loss and evaluation masks remain unchanged sidecar evidence and
are never treated as interchangeable values.

The adapter applies one shared trainable linear map from the frozen information
channels to one scalar per stock and date. The author therefore still receives
exactly one variate token per stock; adding F, missing-mask or S information does
not multiply the stock-token axis. The projector has `C + 1` parameters and is
shared across every stock, so its size does not depend on universe membership or
the A0-A3 gate. This keeps the upstream network untouched, but it remains an
A-share input adaptation rather than a verbatim paper-data reproduction. The
research label must therefore be **official backbone + protocol-safe runner +
A-share input adaptation**.

## Runtime fidelity

iTransformer and FACT wrappers receive an already instantiated model from the
pinned, read-only server checkout. The external projector consumes the observed
mask, then calls the native four-argument forecasting forward interface; no
upstream layer is imported into or copied into this package. The protocol-safe
runtime retains MSE, Adam, the upstream learning-rate schedule and validation-MSE
checkpoint selection while exposing no test loader to fitting. MSE is evaluated
only on targets that are both observed and label-available. FACT must keep the
documented mixed `core=0.5`; this layer does not repair its known frequency-only
branch.

Both pinned backbones implement `use_norm` with an in-place `x_enc /= stdev`.
That is safe for their original leaf data tensor, but invalidates autograd when
`x_enc` is the output of the trainable shared projector. The external runtime
therefore computes the same detached mean, population variance
(`unbiased=False`), `+1e-5`, square root, division and output denormalization
out-of-place. It temporarily sets the author's `use_norm` flag to false only for
the native four-argument call and restores the exact original value in `finally`.
Tensor and `(prediction, metadata...)` outputs are both preserved. This is a
mathematically equivalent A-share runtime adaptation, not an upstream source fix.

Prediction export requires every expected signal date, including a final partial
batch, and emits one row for every evaluation-master stock. Missing rows are an
adapter failure, not lower coverage. Legitimate non-coverage is expressed only by
the typed states `NOT_MEMBER`, `NOT_OBSERVED`, `INSUFFICIENT_HISTORY` or
`MODEL_UNSUPPORTED`. The primary five-day score is the sum of the five forecast
daily log-return target channels; it is not just the last day's return.

TimePro has no implementation here. Its adapter is a fail-closed
`BLOCKED_LICENSE` record and cannot clone, import, emulate or run upstream code.
