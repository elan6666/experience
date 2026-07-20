# Isolated upstream environments

Author repositories require incompatible Python/Torch stacks. Each environment is
created on the approved server only and must be locked after successful resolution.
Do not install these requirements on the Mac or merge them into the project venv.

- `provenance/requirements.txt` supports registry/checkout auditing only.
- `exact` uses the author's declared dependency versions. A failed exact environment
  is recorded as `ENV_FAIL`; it is not silently upgraded.
- `compat-smoke` uses a separately labelled CUDA 12.8/PyTorch environment only to
  test imports and tensor shapes on RTX 5090. It is not valid for reproduction training.
- iTransformer and FACT environment resolution starts only after detached checkouts
  and the FACT attribution-review warning are recorded.
- TimePro, TimeXer and S4M environment work is blocked by missing repository licenses.
- Bootstrap writes a rebuildable resolved lock, its SHA-256 and a schema-v2 receipt
  under server-only untracked paths. Generated locks and receipts are never committed.
