# D0 server runbook

All commands below run from
`/data/yilangliu/a_share_research/seven_model_research`; generated files remain
under the parent research root and are not copied to Git.

1. Run the read-only inventory and bounded bootstrap plan:

   ```bash
   .venv/bin/python -m pip install --requirement envs/data/requirements.txt
   .venv/bin/python scripts/materialize_d0.py audit \
     --research-root /data/yilangliu/a_share_research
   .venv/bin/python scripts/materialize_d0.py plan \
     --research-root /data/yilangliu/a_share_research
   ```

2. Fetch only bootstrap gaps with the approved proxy client. Inspect the plan
   first; no credential is accepted by the command:

   ```bash
   .venv/bin/python scripts/install_approved_proxy_client.py \
     --research-root /data/yilangliu/a_share_research \
     --receipt /data/yilangliu/a_share_research/receipts/d0/approved_proxy_client_install.json
   .venv/bin/python scripts/materialize_d0.py fetch \
     --research-root /data/yilangliu/a_share_research
   ```

3. After the audited calendar and security master JSONL are available, generate
   the full index history first, then the full bounded security request manifest.
   Each security market endpoint is split by
   permanent code; index data is split by year and suspensions by month. A
   response at the configured provider row limit fails as possible truncation.
   The full manifest also contains exactly two Shenwan L1
   `index_member_all` requests per permanent stock code: `is_new=Y` (at least
   one row) and `is_new=N` (zero rows allowed). Both calls are bounded by
   `ts_code`, have a fixed six-field projection, and reject 1000 rows as
   possible truncation. No unbounded industry call is permitted.

   ```bash
   .venv/bin/python scripts/generate_index_history_request_manifest.py \
     --out /data/yilangliu/a_share_research/data/requests/d0-index-history-v1.json
   .venv/bin/python scripts/materialize_d0.py fetch \
     --research-root /data/yilangliu/a_share_research \
     --request-manifest /data/yilangliu/a_share_research/data/requests/d0-index-history-v1.json
   .venv/bin/python scripts/stage_d0_bootstrap.py \
     --research-root /data/yilangliu/a_share_research \
     --index-manifest /data/yilangliu/a_share_research/data/requests/d0-index-history-v1.json \
     --tech100-workbook /data/yilangliu/a_share_research/data/raw/A股核心技术潜力Top100_2026-07-17.xlsx \
     --tech32-manifest /data/yilangliu/a_share_research/data/processed/tech32_open_to_open_v2/panel_manifest.json \
     --out-root /data/yilangliu/a_share_research/data/staged \
     --receipt /data/yilangliu/a_share_research/receipts/d0/bootstrap_staging_20260719.json
   .venv/bin/python scripts/generate_d0_request_manifest.py \
     --trade-calendar-jsonl /data/yilangliu/a_share_research/data/staged/trade_calendar.jsonl \
     --security-master-jsonl /data/yilangliu/a_share_research/data/staged/security_master.jsonl \
     --universe-codes-json /data/yilangliu/a_share_research/data/staged/four_universe_union_codes.json \
     --out /data/yilangliu/a_share_research/data/requests/d0-v1-pit-industry.json
   .venv/bin/python scripts/materialize_d0.py plan \
     --research-root /data/yilangliu/a_share_research \
     --request-manifest /data/yilangliu/a_share_research/data/requests/d0-v1-pit-industry.json
   .venv/bin/python scripts/materialize_d0.py fetch \
     --research-root /data/yilangliu/a_share_research \
     --request-manifest /data/yilangliu/a_share_research/data/requests/d0-v1-pit-industry.json
   ```

4. Build canonical tables from the exact request manifest, then compile the D0 gate.
   The compile step requires one membership/features/labels JSONL set and a
   coverage receipt for each universe. Do not pass
   `--star50-history-complete` unless the official-history audit is complete.

   ```bash
   .venv/bin/python scripts/build_canonical_d0.py \
     --research-root /data/yilangliu/a_share_research \
     --request-manifest /data/yilangliu/a_share_research/data/requests/d0-v1-pit-industry.json \
     --staged-root /data/yilangliu/a_share_research/data/staged \
     --out /data/yilangliu/a_share_research/data/canonical/d0-v1 \
     --cutoff 2026-07-17
   .venv/bin/python scripts/compile_d0_manifest.py \
     --canonical-root /data/yilangliu/a_share_research/data/canonical/d0-v1 \
     --raw-manifest-root /data/yilangliu/a_share_research/data/raw/d0_v1 \
     --materialization-receipt /data/yilangliu/a_share_research/data/canonical/d0-v1/materialization_receipt.json \
     --feature-schema configs/features/d0-v1.json \
     --security-master /data/yilangliu/a_share_research/data/staged/security_master.jsonl \
     --trading-calendar /data/yilangliu/a_share_research/data/staged/trade_calendar.jsonl \
     --market-state /data/yilangliu/a_share_research/data/canonical/d0-v1/shared_market_state.jsonl \
     --out /data/yilangliu/a_share_research/data/manifests/d0-v1.json
   ```

   Existing immutable partitions keep the same request hashes and are reused;
   only missing industry hashes are fetched. The canonical builder starts an
   industry interval at the first trading day strictly after `in_date`, ends it
   at `out_date` when present, and never projects a current classification into
   earlier dates. `shared_market_state.receipt.json` records daily CSI300 PIT
   industry coverage. Industry dispersion remains missing on dates below the
   configured coverage threshold.

5. Generate formal feature-eligibility sidecars only after the final D0
   manifest exists. The generator re-hashes the sealed canonical tables,
   validates every PIT feature row against its schema and independent missing
   mask, and checks every non-missing S value against the one shared CSI300
   market-state table. Causal missing values are allowed; they are not treated
   as ineligible merely because they are missing, but their factor-specific
   missing mask must exist and agree with the feature row.

   ```bash
   .venv/bin/python scripts/generate_formal_feature_receipts.py \
     --d0-manifest /data/yilangliu/a_share_research/data/manifests/d0-v1.json \
     --canonical-root /data/yilangliu/a_share_research/data/canonical/d0-v1 \
     --feature-schema /data/yilangliu/a_share_research/seven_model_research/configs/features/d0-v1.json \
     --out-dir /data/yilangliu/a_share_research/receipts/d0
   ```

   Only CSI300/STAR50 gates in `PASS` or `PASS_WITH_WARNING` receive A0--A3
   `FormalFeatureManifest` files. A blocked STAR50 gate produces **no** STAR50
   manifest; its `NOT_GENERATED` decision and exact D0 status remain explicit
   in `formal-feature-generation-audit.json`. Tech32/tech100 are never emitted
   by this command. All outputs are atomic and pre-existing receipt or `.tmp`
   paths cause a fail-closed rejection rather than replacement.

   The exact input names are Core for A0; Core + F + one `__missing` input for
   every F factor for A1; Core + S for A2; and all four groups for A3. The
   audit binds the final `D0Manifest.content_hash`, D0 file SHA, feature schema,
   canonical table SHAs, shared-state content hash, present/missing counts and
   each generated sidecar SHA. These receipts are data eligibility evidence,
   not model results.

6. Run server-only verification:

   ```bash
   .venv/bin/python -m pytest -q tests/data tests/universes tests/features tests/protocol
   .venv/bin/ruff check src/a_share_research/data src/a_share_research/universes src/a_share_research/features tests
   .venv/bin/python -m compileall -q src tests scripts
   ```

7. Generate the compact quality report only after the manifest passes:

   ```bash
   .venv/bin/python scripts/report_d0_quality.py \
     --manifest /data/yilangliu/a_share_research/data/manifests/d0-v1.json \
     --canonical-root /data/yilangliu/a_share_research/data/canonical/d0-v1 \
     --out /data/yilangliu/a_share_research/reports/d0-v1-quality.json
   ```

The copied-back compact audit may contain statuses, counts and hashes only. Raw
rows, predictions, logs and credentials remain server-side.
