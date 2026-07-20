# Legacy migration allowlist

Default policy: **nothing is approved for migration**.

Legacy directories are read-only references. A component may enter this source
tree only after an active plan records all of the following:

1. exact legacy source path and destination;
2. reason it is preferable to a clean implementation or upstream source;
3. provenance and applicable license;
4. tests that establish current protocol behavior;
5. PIT/leakage and secret review;
6. approving reviewer and date.

Bulk copies of legacy source, data, results, configs or reports are prohibited.

| Source path | Destination | Plan | Tests | Approval | Status |
|---|---|---|---|---|---|
| _none_ | — | — | — | — | not approved |

## Plan 003 read-only audit (no migration approved)

These files were located during the D0 server-data/layout audit. No source was
copied or imported. Their hashes record exactly what was inspected; any future
migration still requires tests, PIT/security review and explicit approval.

| Source path | SHA-256 | Audit reason | Destination | Status |
|---|---|---|---|---|
| `reproductions/github_a_share_handoff/src/a_share_research/data/history.py` | `f82bb1dc861a997549c72e3736a39e7c0f260cba6df67d6010cf69063781a549` | locate prior `index_weight` layout only | none | reviewed, not migrated |
| `reproductions/github_a_share_handoff/src/a_share_research/data/source.py` | `34887c6f02e7efc55c353d2d69c6b32d4a5b2c5a1cc020cf87c030330973dd8e` | identify prior endpoint/date-field inventory only | none | reviewed, not migrated |
| `reproductions/github_a_share_handoff/src/a_share_research/data/historical_panel.py` | `06fa04c3121edede2b5531c623826d62ced67be942442f53761363350904ffd3` | locate existing historical CSI300 server roots only | none | reviewed, not migrated |
| `research_program_v2/scripts/build_csi300_daily_snapshot_panel.py` | `5222fd9c0a749a06e3cf60f8dee4f1e0213759dce565f6780460494b52a8b536` | locate legacy CSI300 snapshot naming only | none | reviewed, not migrated |
| `research_program_v2/scripts/build_csi300_s4m_masked_panel.py` | `a25db4dada6b8b48003cb9186ddd5b1f028595984fe641affb64bc08de1926b7` | confirm legacy model-specific panel is unsuitable for canonical D0 | none | reviewed, not migrated |
