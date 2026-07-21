# iTransformer / FACT V0-V1 作业生成与 GPU 队列

## 边界

本流程只能在服务器
`/data/yilangliu/a_share_research/seven_model_research` 执行。生成器不训练模型；
它仅核验 D0 门禁、正式特征收据与所有证据文件的 SHA-256，然后生成
`DeepJobSpec` 和每块 GPU 的串行队列。

- V0：`A0 × 4 股票池 × 2 模型 × 3 seeds = 24` 个计划单元。
- V1：`A1/A2/A3 × 4 股票池 × 2 模型 × 3 seeds = 72` 个计划单元。
- V1 不重训 A0，A0 使用 V0 的锁定结果作为消融参照。
- iTransformer 固定物理 GPU0，FACT 固定物理 GPU1。同一 GPU 内串行，
  两个 GPU 队列可以并行。
- 沪深300和科创50只有在 D0 通过且对应 A0-A3 正式收据存在时才能
  入队；科技32/科技100始终标为 `EXPLORATORY_ONLY`。
- 选择窗口固定为 2025-01-01 至 2025-12-31，生成收据明确记录
  `legacy_2026_selection_allowed=false`。

## 生成前必须存在的证据

1. 最终 D0 manifest 和被其 SHA-256 封印的 canonical tables。
2. iTransformer/FACT 各自的环境收据、上游完整性收据和只读作者 checkout。
3. 当前源码 manifest、模型 adapter config 与 `deep-common.json`。
4. 正式股票池对应信息集的 `FormalFeatureManifest`。一个因子缺失、一个
   独立缺失掩码缺失，或收据不是最终 D0 的内容哈希，都会将该 gate 的
   6 个单元记为 `BLOCKED`。

## 生成命令

先根据服务器实际收据路径设置下列变量，不要使用未核验的占位路径。
命令中每个 `MODEL=PATH` 都必须恰好出现一次。

```bash
cd /data/yilangliu/a_share_research/seven_model_research

.venv/bin/python scripts/generate_deep_job_manifests.py \
  --phase V0 \
  --d0-manifest /data/yilangliu/a_share_research/data/manifests/d0-v1.json \
  --canonical-root /data/yilangliu/a_share_research/data/canonical/d0-v1 \
  --upstream-root itransformer=/data/yilangliu/a_share_research/upstreams/itransformer \
  --upstream-root fact=/data/yilangliu/a_share_research/upstreams/fact \
  --environment-receipt itransformer=/ABSOLUTE/ITRANSFORMER_ENV_RECEIPT.json \
  --environment-receipt fact=/ABSOLUTE/FACT_ENV_RECEIPT.json \
  --integrity-receipt itransformer=/ABSOLUTE/ITRANSFORMER_INTEGRITY.json \
  --integrity-receipt fact=/ABSOLUTE/FACT_INTEGRITY.json \
  --code-receipt /data/yilangliu/a_share_research/receipts/source/source-manifest-v1.json \
  --adapter-config itransformer=/data/yilangliu/a_share_research/seven_model_research/configs/adapters/itransformer.json \
  --adapter-config fact=/data/yilangliu/a_share_research/seven_model_research/configs/adapters/fact.json \
  --common-config /data/yilangliu/a_share_research/seven_model_research/configs/adapters/deep-common.json \
  --formal-feature-receipt CSI300:A0=/ABSOLUTE/CSI300_A0_FORMAL.json \
  --formal-feature-receipt STAR50:A0=/ABSOLUTE/STAR50_A0_FORMAL.json \
  --run-root /data/yilangliu/a_share_research/runs \
  --checkpoint-root /data/yilangliu/a_share_research/checkpoints \
  --job-root /data/yilangliu/a_share_research/jobs/deep \
  --queue-root /data/yilangliu/a_share_research/queues/deep
```

生成 V1 时把 `--phase` 改为 `V1`，并为沪深300、科创50分别传入
A1、A2、A3 收据。不传 A0，因为 V1 不重训 A0。

## 队列执行

生成后先查看 `jobs/deep/<phase>/generation_receipt.json`，确认计划单元数、
`BLOCKED` 原因和每个 GPU 的可运行数。不得为了凑齐矩阵而运行被阻塞单元。

队列 JSON 中的 `jobs` 已是确定性 FIFO 顺序。调度器必须：

1. GPU0 队列的每个作业都使用 `CUDA_VISIBLE_DEVICES=0`。
2. GPU1 队列的每个作业都使用 `CUDA_VISIBLE_DEVICES=1`。
3. 每个 GPU 上一次只执行一个 `scripts/run_deep_cells.py --job-spec ...`。
4. 可在两个独立 shell 中同时启动 GPU0 和 GPU1 队列，但不得在单块 GPU
   内并发。
5. 任一作业返回非零状态时停止该 GPU 队列，保留类型化失败记录后复核；
   不要直接跳过。

## 不可更改的忠实性约束

- 作者 commit、架构、MSE loss、Adam optimizer、作者调度器和最佳验证
  MSE checkpoint 规则不变。
- 共享的逐股线性投影、成员/缺失掩码和 A 股输出适配器都位于作者
  源码之外，且在 cell hash 和运行收据中标注。
- 生成目录不可覆盖。需要重跑时，必须先根据现有收据做出明确的人工
  审计决定，不得由生成器静默删除。

## V1 实际生成命令（2026-07-20 已验证，六模型/4 deep）

V1 不重训 A0，仅生成 A1/A2/A3 消融 cell。源码变更后需先重建 source manifest
（此处为 v7）。CSI300 有 A1-A3 正式特征回执；TECH32/TECH90 为 EXPLORATORY_ONLY
不需正式回执即可入队；STAR50 被 D0 gate 阻断。

结果：planned=144，blocked=36（全为 STAR50 D0_GATE_BLOCKED），runnable=108
（CSI300/TECH32/TECH90 × A1/A2/A3 × 4 模型 × 3 seed），4 GPU 队列
（GPU0: itransformer+timexer，GPU1: fact+timepro）。

```bash
cd /data/yilangliu/a_share_research/seven_model_research
R=/data/yilangliu/a_share_research; SR=$R/seven_model_research
# 1) 源码变更后重建 source manifest
.venv/bin/python scripts/build_source_manifest.py --root $SR \
  --output $R/receipts/source/source-manifest-v7.json
# 2) 生成 V1 deep 清单（4 模型）
.venv/bin/python scripts/generate_deep_job_manifests.py --phase V1 \
  --d0-manifest $R/data/manifests/d0-v1.json \
  --canonical-root $R/data/canonical/d0-v1 \
  --upstream-root itransformer=$R/upstream/itransformer@c2426e68ca13 \
  --upstream-root fact=$R/upstream/fact@aa825721d1a0 \
  --upstream-root timepro=$R/upstream/timepro@70a20e5a257b \
  --upstream-root timexer=$R/upstream/timexer@760119093579 \
  --environment-receipt itransformer=$R/receipts/upstream_v2/itransformer_compat_env_20260719.json \
  --environment-receipt fact=$R/receipts/upstream_v2/fact_compat_env_20260719.json \
  --environment-receipt timepro=$R/receipts/upstream_v2/timepro_compat_env_20260720.json \
  --environment-receipt timexer=$R/receipts/upstream_v2/timexer_compat_env_20260720.json \
  --integrity-receipt itransformer=$R/receipts/upstream_v2/itransformer_checkout_20260719.json \
  --integrity-receipt fact=$R/receipts/upstream_v2/fact_checkout_20260719.json \
  --integrity-receipt timepro=$R/receipts/upstream_v2/timepro-checkout.json \
  --integrity-receipt timexer=$R/receipts/upstream_v2/timexer_checkout_20260720.json \
  --code-receipt $R/receipts/source/source-manifest-v7.json \
  --adapter-config itransformer=$SR/configs/adapters/itransformer.json \
  --adapter-config fact=$SR/configs/adapters/fact.json \
  --adapter-config timepro=$SR/configs/adapters/timepro.json \
  --adapter-config timexer=$SR/configs/adapters/timexer.json \
  --common-config $SR/configs/adapters/deep-common.json \
  --formal-feature-receipt CSI300:A1=$R/receipts/d0/formal-csi300-a1.json \
  --formal-feature-receipt CSI300:A2=$R/receipts/d0/formal-csi300-a2.json \
  --formal-feature-receipt CSI300:A3=$R/receipts/d0/formal-csi300-a3.json \
  --run-root $R/runs --checkpoint-root $R/checkpoints \
  --job-root $R/jobs/deep/v1 --queue-root $R/queues/deep/v1
```

待办：tabular V1（ridge/lightgbm × A1/A2/A3）尚未生成。`generate_tabular_job_manifests.py`
参数已确认（model-config=configs/models/{ridge,lightgbm}-v1.json、layout-config=
configs/features/tabular-layout-v1.json、environment-receipt=upstream_v2/*_baseline_env_*.json）。

**性能 bug（已定位）**：tabular 生成器在 `for universe` 循环内对每个 universe 调
`validation_registry_hash` -> `loader.iter_tabular_samples(start=2025,end=2025)`，
该 loader 全量解析 canonical JSONL（CSI300 features 3.6G）后过滤 2025 验证面板。
4 个 universe 串行全量解析 = 6-7 分钟（deep 仅 55 秒，因 deep 只算一次/复用 asset
registry hash）。修复方向：让 `iter_tabular_samples` 支持日期范围早退/索引，或缓存
per-universe 解析结果。生成器非卡死，纯 CPU JSONL 解析，耐心等可完成；或修复后重跑。

V1 训练待服务器维护后启动（108 deep cell，注意 iTransformer 收敛问题仍存在）。
