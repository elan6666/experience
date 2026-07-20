# D0 PIT data protocol

This cycle builds one canonical causal panel for CSI300, STAR50, tech32 and
tech100 before model-specific tensors are created.

- CSI300 and STAR50 use dated official membership snapshots. Missing STAR50
  history blocks formal ranking; latest constituents are never backfilled.
- tech32 and tech100 preserve their 2026-07-17 selection provenance and remain
  `EXPLORATORY_ONLY`, even when their data pass all technical gates.
- D0 retains raw values and independent missing flags. Imputation,
  winsorization, scaling and industry/size neutralization are fitted only in a
  later training fold.
- Date-only financial announcements and daily-basic valuation rows become
  usable no earlier than the next trading day. The original source trade or
  announcement date remains in every feature row.
- Labels use unadjusted next-open prices: `log(open[t+1+h]/open[t+1])` for
  `h in {1,5,20}`. Adjustment factors are recorded only as corporate-action
  evidence; no future-adjusted price enters an executable label.
- One CSI300 market-state table (trend, volatility, turnover, breadth and
  liquidity) is built once. Industry dispersion remains an explicit missing
  feature until historical effective-date industry evidence exists; the
  current `stock_basic.industry` value is never projected into history. Every
  run must reference the exact shared-state hash.
- Execution eligibility is a separate T+1 receipt with member, observed,
  buyable and sellable evidence. It does not alter feature or label masks.

Before provider calls, `scripts/materialize_d0.py audit` checks the existing
server datasets. Only missing or unverifiable logical datasets may enter the
incremental request plan. The provider is constructed solely through the
server-owned `scripts/tushare_proxy_client.py:get_pro`; its tutorial-compatible
proxy uses plain HTTP, which remains disclosed in every manifest.

Canonical construction reads only
`raw/d0_v1/<endpoint>/<partition_key>/<request_hash>` paths named by the frozen
request manifest. Unreferenced legacy partitions and every quarantine subtree
are ignored. Per-code market tables and per-universe annual signal shards carry
input/output receipts, so an interrupted server build resumes without accepting
stale output. Flat weekly features, compact 1/5/20-day labels and masks are
stream-joined from those shards; compact daily market files remain available to
deep sequence adapters.

Downstream code must use
`a_share_research.data.loaders:CanonicalDatasetLoader`. Its streaming API emits
validated canonical feature/label/mask rows, `iter_tabular_samples()` yields the
existing `TabularSample` contract, `load_panel_window()` pads append-only masks
to a caller-supplied causal fold master and returns the existing `PanelWindow`
contract, and `iter_daily_market()` exposes compact unadjusted sequences. The
loader never fits an imputer, scaler, neutralizer or model-specific transform.
