# V1 information-ablation registry protocol

V1 is an intent registry, not a training implementation. It consumes the
sealed 28-cell V0/A0 registry and expands it to a complete 112-cell table.
The 28 A0 rows are immutable `REFERENCE_V0` rows, so V1 creates only 84 new
A1/A2/A3 training intents.

## Frozen matrix

- Models: Ridge, LightGBM, iTransformer, FACT, TimePro, TimeXer and S4M.
- Universes: CSI300, STAR50, tech32 and tech100; each remains a separate panel.
- Gates: A0=`Core`, A1=`Core+F+F-missing`, A2=`Core+S`,
  A3=`Core+F+F-missing+S`.
- tech32 and tech100 remain `EXPLORATORY_ONLY` and cannot produce a formal
  winner. This label does not override a stronger `BLOCKED_LICENSE` reason.
- TimePro, TimeXer and S4M remain `BLOCKED_LICENSE` for all four gates and all
  four universes. A1-A3 rows for them record the block; they do not emulate or
  execute upstream code.

## Information-only invariant

Within one model/universe family, A0-A3 reuse the exact same typed
`TrainingSignature`: model-capacity hash, model-hyperparameter hash, 2019-2024
training window, 2025 validation window, weekly five-day open-to-open excess
return label, optimizer, maximum epochs, early stopping and seeds. The only
allowed change is the information gate. Every cell also carries the same
independently built CSI300 `market_state_hash`.

## Frozen comparisons

The attempted comparison family is exactly `A1-A0`, `A2-A0`, `A3-A0`,
`A3-A1` and `A3-A2`. Comparisons are paired on identical 2025 validation dates
and support. The already viewed 2026 interval is report-only and is unavailable
to registry construction, model selection or comparison-family changes.

The source module deliberately has no launcher, trainer or evaluator. The
server runner may act only on a sealed registry after Plan009 publishes the
real V0 references and Plan010 verification confirms the 28/84/112 counts.
