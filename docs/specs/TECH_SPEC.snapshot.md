# Technical Specification — Server-Only A-Share Research

## Architecture

所有计算位于服务器 `/data/yilangliu/a_share_research`：

`provider raw data → immutable snapshots → canonical PIT panel → universe/feature/mask datasets → official-code adapters → prediction store → common evaluator → risk/portfolio engine → compact reports`

Mac只保存源码、Byte OS规格和复制回的小型文档；不得在Mac运行下载、特征物化、训练、推理、回测或结果生成。

## Integrations

- Tushare兼容代理：只通过服务器安全凭据和项目代理客户端访问；令牌不得进入命令、日志或Git。
- 可选交叉校验源只用于数据一致性抽查，不替代正式PIT来源。
- 上游论文仓库固定commit；运行兼容补丁与模型改动分开保存。
- GPU0/GPU1可并行不同模型或股票池，但不得共享可变输出目录。

## Canonical data objects

- `universe_membership(asof_date, ts_code, universe, effective_from, effective_to, source)`
- `security_master(ts_code, list_date, delist_date, board, industry, identity_version)`
- `market_daily(trade_date, ts_code, ohlcv, amount, turnover, adj, limits, suspension, st_state)`
- `pit_features(asof_date, ts_code, feature_name, value, source_date, announce_time, missing_flag)`
- `market_state(asof_date, feature_name, value)`：固定由沪深300及全体可交易成分生成。
- `labels(signal_date, ts_code, horizon, open_to_open_return)`：信号日收盘后，按T+1开盘成交定义。
- `eligibility(signal_date, ts_code, universe, member, observed, tradable, complete)`
- `manifest(run_id, data_hash, split, upstream_commit, config_hash, seed, status)`
- `predictions(run_id, signal_date, ts_code, score, coverage_state)`
- `portfolio(run_id, trade_date, ts_code, target_weight, executed_weight, cost, reject_reason)`

## Splits and labels

- 训练：2019-01-01至2024-12-31；科创50正式协议从指数有效历史开始。
- 验证：2025-01-01至2025-12-31。
- 研究比较集：2026-01-01至2026-07-17，禁止继续调参。
- 最终未见集：协议冻结后新到数据；进入前不得读取标签或组合结果。
- 原生预测目标：作者代码保留其原生目标/loss；共同评估把预测转换成当日横截面分数。
- 主经济标签：`log(open[t+6] / open[t+1]) - benchmark_return[t+1:t+6]`。沪深300/科创50使用相应指数，科技32/100使用当日合格成分等权；绝对收益另行报告。日/月稳健性使用对应1/20交易日版本。
- split边界实施purge/embargo，防止未来标签重叠。

## Model boundary

- Ridge、LightGBM：共同表格PIT输入与训练期预处理。
- iTransformer、FACT、TimePro：保留上游预测网络/训练逻辑，通过外部动态主表、张量和掩码适配；多因子输入若必须改变嵌入接口，明确标记为“官方骨干+A股输入适配”，不得称为原样复现。
- TimeXer：Core作为内生序列；F/S通过作者外生变量通道进入。
- S4M：保留作者原生观察掩码；成员、上市和可交易状态在外部适配层生成。
- 所有模型输出转换为统一的 `(signal_date, ts_code, score)`；转换不修改模型训练目标。
- A0/A1/A2/A3对同一模型保持参数量和训练配置不变，仅切换信息组；否则变化必须作为独立架构实验而非信息消融。

动态股票池采用walk-forward冻结：每次训练/重训只使用当时已知的股票身份；新成员在模型能够因果重训前标记为不可评分。正式主表同时报告共同支持集结果和各模型原生覆盖率。

## Experiment structure

- V0：7模型×4股票池×Core，共28个运行单元。
- V1：完整表为7模型×4股票池×A0/A1/A2/A3；A0直接引用V0产物，新增84个运行单元，不重复训练。
- Seeds：深度模型至少3个；树/线性记录确定性或重复采样口径。
- Epoch：沿用作者默认或提高最大上限，同时使用验证早停；测试集不参与。
- V2：读取冻结预测，不重新训练以适配组合表现；B0/B1/B2/B3按顺序执行。

## Portfolio and execution

- 主持仓比例约10%；敏感性Top3/5/10/30预注册后运行。
- 统一市场风险预算只能使用当时已知的沪深300状态，输出100/60/30/0。
- 受约束组合包含单股/行业/流动性/换手/成本/容量边界；具体费率配置版本化并记录来源日期。
- 执行遵守T+1、逐日价格限制、停牌和未成交延续；不得假设按不可得收盘价成交。

## Implementation risks

- 科创50历史成分快照不完整会使正式比较阻断。
- 科技32/100是2026选择名单，存在不可消除的条件选择偏差。
- 固定宽度时序模型面对新成员可能覆盖不足；不允许按日期重排股票槽位。
- TimePro运行时兼容补丁可能影响复现边界，必须单独审计。
- 112主配置再交叉三频率、Top-K和风险档位会导致多重检验与算力爆炸。
- 2026比较集已污染为开发反馈，不能再次充当最终测试。

## Testing strategy

1. Schema/PIT：公告时间、成员有效区间、无未来数据、逐因子缺失标记。
2. Identity/mask：代码槽位不漂移、上市前/退市后/非成员正确遮罩。
3. Split/label：purge/embargo、T+1、horizon、无重叠泄漏。
4. Upstream smoke：每个官方仓库固定commit的最小训练/推理。
5. Adapter parity：无适配时对齐作者示例；有适配时张量形状和逆映射可验证。
6. Determinism：固定seed重放、manifest和预测哈希。
7. Backtest invariants：权重、现金、成本、不可交易延续、换手和净值恒等式。
8. Baseline regression：等权/动量/现金/指数结果可独立复算。
9. Result audit：异常收益、全零仓位、覆盖骤降和模型间结果差异逐项归因。
10. Sanity checks：小批量过拟合、随机标签、未来数据扰动不改变过去预测、日期/符号错位对照。

## Result state taxonomy

- `PASS` / `PASS_WITH_WARNING`：协议有效，可进入证据评级。
- `EXPLORATORY_ONLY`：数据有效但存在已知条件选择或历史覆盖限制。
- `BLOCKED` / `INVALID_DATA` / `INVALID_PROTOCOL`：外部阻断、PIT错误或比较协议不公平。
- `ADAPTER_FAIL` / `TRAIN_FAIL` / `EVAL_FAIL`：分别表示适配、训练或评价链路失败。
- `VALID_NEGATIVE`：链路完全正确但模型无预测/经济价值；不得误记为代码失败。

证据等级从E0（无样本外能力）到E3（在新未见数据、成本和约束后仍成立）；只有PASS类结果可评级和排名。

## Security and Git

- 凭据只保存在服务器安全路径，权限0600；永不打印或提交。
- Git只跟踪源码、轻量配置、补丁和文档；数据、结果、日志、checkpoint、权重和预测数组全部排除。
- FUTURE.md中的新仓库发布任务仍为parked，不在当前执行范围。
