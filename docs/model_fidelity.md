# Model provenance and fidelity boundary

Audit date: 2026-07-19. This document freezes what can be called an official
reproduction before any A-share adapter is trained. The machine-readable source
of truth is [`configs/upstreams.lock.yaml`](../configs/upstreams.lock.yaml).

## Audit verdict

| Model | Official primary source | Pinned official implementation | License | Current gate |
|---|---|---|---|---|
| Ridge | [scikit-learn 1.9.0](https://pypi.org/project/scikit-learn/1.9.0/) | `sklearn.linear_model.Ridge` | BSD-3-Clause | READY baseline |
| LightGBM | [LightGBM 4.7.0](https://pypi.org/project/lightgbm/4.7.0/) | `lightgbm.LGBMRegressor` | MIT | READY baseline |
| iTransformer | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html) | [`thuml/iTransformer@c2426e6`](https://github.com/thuml/iTransformer/commit/c2426e68ca13f74aaec08045c5c724d8ad328124) | MIT | schema-v2 compatibility smoke PASS |
| FACT | [ICLR 2026](https://openreview.net/forum?id=j3gNYqrHtl) | [`wanghq21/FACT@aa82572`](https://github.com/wanghq21/FACT/commit/aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4) | MIT file with attribution ambiguity | schema-v2 compatibility smoke PASS_WITH_WARNING |
| TimePro | [ICML 2025](https://proceedings.mlr.press/v267/ma25p.html) | [`xwmaxwma/TimePro@70a20e5`](https://github.com/xwmaxwma/TimePro/commit/70a20e5a257b30eb026ee4316293cf4feeb92a1f) | no repository license | **BLOCKED_LICENSE** |
| TimeXer | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) | [`thuml/TimeXer@7601190`](https://github.com/thuml/TimeXer/commit/76011909357972bd55a27adba2e1be994d81b327) | no repository license | **BLOCKED_LICENSE** |
| S4M | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html) | [`WINTERWEEL/S4M@a718823`](https://github.com/WINTERWEEL/S4M/commit/a718823addd3606e763dfc261174e0135b2535f4) | no repository license | **BLOCKED_LICENSE** |

The three license blocks are not claims that the papers or repositories are
untrustworthy. They mean that public readability alone does not grant permission
to copy, modify, patch, or redistribute copyrighted source. The Apache-2.0 file
inside S4M's vendored S4 directory licenses S4, not the surrounding S4M code.
The blocked checkouts must not be created on the server until permission or an
explicit upstream license is recorded.

FACT has an MIT text at repository root, but its copyright line names THUML and
the year 2021 rather than the FACT authors/repository publication. Internal smoke
may proceed as `PASS_WITH_WARNING`, preserving the exact license file and setting
`license_review_required=true`; redistribution or publication waits for attribution
review. This is not treated as a silent clean MIT determination.

## What “official fidelity” means

For each deep model, the frozen author network, loss, optimizer, scheduler,
normalization, validation criterion, checkpoint choice, and forward inference are
the reference semantics. Our code may construct PIT A-share tensors, fixed stock
identities, observation masks and exports outside the author tree. It may not edit
author layers in place and still call the run an unmodified reproduction.

The common evaluator does not change training. It converts a frozen model's output
to `(signal_date, ts_code, score)` and computes RankIC, returns and execution metrics
using the same labels and masks across models.

## Native training semantics

- **iTransformer** embeds each variate's lookback as a token, applies an
  encoder-only Transformer across variates and projects to a multivariate point
  forecast. The author loop uses MSE, Adam, learning-rate adjustment and
  validation-MSE early stopping.
- **FACT** places an ordered set of variables into a two-dimensional layout and
  applies depth-wise, multi-dilation convolutions in time and frequency domains.
  It uses MSE, Adam and validation-MSE early stopping. Stock ordering is therefore
  a model input, not cosmetic metadata.
- **TimePro** uses Mamba-style selective scan plus deformable convolution to create
  variable- and time-aware hyper-states. Its author environment builds custom CUDA
  extensions. The native loop uses MSE, Adam and validation-MSE early stopping.
- **TimeXer** separates endogenous patches from exogenous variate tokens and uses a
  global endogenous token for cross-attention. In the author's exogenous `MS` path,
  the last column is the target and earlier columns are exogenous inputs. It uses
  MSE, Adam and validation-MSE early stopping.
- **S4M** jointly consumes values and observation masks through a prototype memory
  and missing-aware S4 stream. The official loop uses masked MSE, SGD with momentum
  and weight decay, validation early stopping, and saves memory-bank state with the
  checkpoint. Replacing this with Adam or impute-then-forecast is not S4M fidelity.

Ridge and LightGBM are project baselines, not paper reproductions. Their exact
package versions, preprocessing, hyperparameters and validation choices must still
be versioned and hashed. LightGBM has no implicit project early stopping: boosting
rounds come from versioned configuration. Early stopping is active only when an
explicit validation callback is configured, after which inference freezes the
recorded `best_iteration_`.

## Required protocol adaptation: hide the test set

iTransformer, FACT, TimePro and TimeXer author trainers instantiate the test loader
and print test loss after every training epoch. Their checkpoint decision is based
on validation loss, but repeatedly exposing test performance violates this project's
unseen-test protocol. Formal runs must therefore use an **external training runner**
that preserves the author model, MSE, Adam, scheduler, validation early stopping and
inference while omitting test-loader construction until the checkpoint is frozen.

This is labelled `official backbone + protocol-safe runner`, not “verbatim author
script”. S4M's observed training loop does not evaluate test each epoch, but its
hard-coded paths and mandatory tracking calls still require an external launch
wrapper. The wrapper may redirect outputs and disable network tracking; it may not
change masked loss, SGD, memory warm-up or memory serialization.

## Known upstream issues that must not be silently fixed

1. FACT's pinned `models/FACT.py` repeats `alpha == 1.0` in the branch intended for
   frequency-only execution. `core=0` can therefore enter an inconsistent path.
   Smoke the documented default mixed setting (`core=0.5`). A correction requires a
   separate patch file, before/after smoke and semantic review.
2. TimePro depends on `selective_scan_cuda_oflex_rh` and `DCNv4`; its CUDA 11.7-era
   environment may not build or execute on RTX 5090. Failure is `UPSTREAM_FAIL` or
   `BLOCKED`, not permission to replace kernels.
3. S4M commits Python cache files, uses hard-coded output directories and documents
   a weather path not present at that location. These are reproducibility risks.
4. TimeXer requires `seq_len % patch_len == 0`; the A-share adapter must validate
   this before launch.
5. iTransformer and FACT normalize with an in-place division. A trainable external
   feature projector makes its output a non-leaf autograd tensor, so the unchanged
   in-place statement fails during backward. The reviewed external wrapper must
   reproduce the same detached-mean, population-variance, epsilon and denormalization
   equations out-of-place, disable `use_norm` only during the native call, and restore
   it even when the author forward raises.

## A-share adaptation boundary

| Model | Allowed external adapter | Fidelity-breaking change |
|---|---|---|
| iTransformer | fixed stock slots, PIT tensor, input/coverage masks outside model, output inverse map, mathematically equivalent out-of-place native normalization | putting new factor channels into a rewritten embedding without labelling architecture adaptation |
| FACT | stable predeclared stock ordering, PIT tensor, external masks/export, mathematically equivalent out-of-place native normalization | reordering stocks by realized/future correlation or silently repairing convolution branches |
| TimePro | fixed stock slots and PIT tensor, server-only compatibility patch if reviewed | replacing selective scan/DCNv4 or changing MSE/Adam to make it run |
| TimeXer | target return as endogenous input; PIT factor/market-state series through author exogenous channel | concatenating future-known state, or rewriting cross-attention to accept features |
| S4M | author observation mask for truly unobserved values; external member/tradability/loss/evaluation masks | treating non-membership as ordinary missingness or changing SGD/masked-MSE/memory semantics |

Member, observed, feature-missing, buy, sell, loss and evaluation masks are distinct.
Only the observation mask is a native S4M input. No model may learn from a buy/sell
eligibility flag containing information unavailable at signal time.

## Server smoke order

1. Archive or mark stale every pre-schema-v2 receipt; it cannot be migrated into a
   passing receipt because the missing evidence was never measured.
2. Verify each allowed remote commit with `git ls-remote`, clone read-only into
   `/data/yilangliu/a_share_research/upstream/<model>@<short-sha>`, detach the exact
   commit, record `git status --porcelain --untracked-files=all`, hash worktree bytes
   outside `.git`/runtime caches, and verify the checkout root and all contents are
   read-only.
3. Create one isolated environment per deep model. `exact` preserves author pins;
   `compat-smoke` uses CUDA 12.8 only for RTX 5090 import/shape feasibility and is
   never evidence of exact reproduction. Each environment emits a rebuildable
   resolved lock and SHA-256.
4. Run the smallest official example when its dataset is available; otherwise run a
   minimal forward/backward synthetic shape using the unmodified author model.
5. Pass `--physical-gpu`; record `CUDA_VISIBLE_DEVICES`, Torch's logical current
   device, and `nvidia-smi` physical index/UUID/PCI evidence. GPU0/GPU1 results are
   invalid without this mapping.
6. Every success and failure writes a schema-v2 receipt with stage and hashed stderr.
7. Do not run TimePro, TimeXer or S4M until their license gate is resolved.

No server smoke has been run as part of this Mac-only source audit.

## Server verification receipt (2026-07-19)

The source audit was subsequently executed on the approved server. A prior
pre-schema-v2 registry query observed all five pins at repository HEAD, but that
receipt is stale by design. Five per-model schema-v2 registry refreshes on
2026-07-19 all timed out against GitHub; each is retained as
`UPSTREAM_FAIL/query_remote_heads` with its failed model and error type, not
upgraded to success. The exact
commits already present on the server were therefore re-audited locally from their
detached worktrees: commit, tracked and untracked status, complete content hash and
read-only permissions were all checked. This proves the fixed worktrees, but does
not substitute for the failed current remote-HEAD refresh.

Exact upstream dependency resolution was attempted first and failed on the
server's Python 3.12 because the historical pandas/Torch stacks do not provide a
compatible environment. Those schema-v2 failures are retained as environment
evidence and are not attributed to either model.

Two isolated CUDA 12.8 compatibility environments were then used only for a
minimal unmodified-author-model forward/backward smoke:

| Model | Commit | GPU | Torch | Shape | Result |
|---|---|---|---|---|---|
| iTransformer | `c2426e68ca13` | RTX 5090 / GPU0 | `2.7.1+cu128` | `2 x 2 x 9` | PASS |
| FACT | `aa825721d1a0` | RTX 5090 / GPU1 | `2.7.1+cu128` | `2 x 2 x 9` | PASS_WITH_WARNING |

The clean Git status, commit tree hash and full worktree-content hash were
identical before and after each smoke. Each receipt also binds the requested
physical GPU to its `nvidia-smi` index, UUID and PCI address and to the running
process UUID. The first iTransformer compatibility attempt used the PyPI CUDA
12.6 wheel and failed because it lacked `sm_120`; the retained successful receipt
uses the official PyTorch CUDA 12.8 wheel. A smoke-script defect that incorrectly
requested input gradients was corrected; author model code was never changed.

Ridge `scikit-learn==1.9.0` and LightGBM `4.7.0` passed isolated synthetic
fit/predict smoke. All current schema-v2 runtime receipts and resolved environment
locks remain server-only under
`/data/yilangliu/a_share_research/receipts/upstream_v2/` and
`/data/yilangliu/a_share_research/locks/upstream_v2/`.

TimePro, TimeXer and S4M remain `BLOCKED_LICENSE`. Their pinned commit identifiers
come from the stale source audit, while current server remote reachability is not
verified because all five schema-v2 refreshes timed out. No checkout, environment
build or execution was performed for the three blocked models.
