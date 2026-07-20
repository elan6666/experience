# V0 A0 诊断与冻结报告

- 生成时间: 2026-07-20
- 范围: 六模型（Ridge / LightGBM / iTransformer / FACT / TimePro / TimeXer）× 三 universe（CSI300 / TECH32 / TECH100），STAR50 因 D0 成分股缺失被 gate 阻断，S4M 延后至 FTR-001
- 协议: 周频 5 日，COMMON/NATIVE × ABSOLUTE/BENCHMARK_RELATIVE，2025 验证窗口
- 运行单元: 42 个可执行 cell（tabular 各 1 seed × 3 universe = 6；deep 各 3 seed × 3 universe = 36）

## 1. 总体结论

| 模型 | csi300 IC | tech32 IC | tech100 IC | 判定 |
|------|----------|----------|------------|------|
| ridge | +0.0440 | +0.0582 | +0.0542 | PASS（最强） |
| lightgbm | +0.0442 | +0.0163 | +0.0390 | PASS |
| timexer | +0.0046 | +0.0455 | +0.0027 | VALID（弱正，深度最佳） |
| fact | -0.0099 | -0.0414 | -0.0211 | VALID_NEGATIVE |
| timepro | -0.0084 | -0.0450 | -0.0041 | VALID_NEGATIVE |
| itransformer | -0.0132 | -0.0309 | -0.0136 | FAIL_CONVERGENCE（8/9 cell 未收敛） |

IC 为 COMMON/BENCHMARK_RELATIVE rank_ic 跨 seed 均值。tabular 单 seed（std=null），deep 三 seed。

## 2. 链式诊断（归因前先排除非模型因素）

| 诊断项 | 结果 | 证据 |
|--------|------|------|
| 覆盖率 | 通过 | 全部 cell scorecard coverage=1.00（common support 上无缺失预测） |
| 符号 | 通过 | sign_accuracy 全部 0.45–0.57，集中于 0.50；非符号翻转（翻转会表现为强负 IC + 持续低 sign_accuracy） |
| 日期对齐 | 通过 | paired_dates=51（2025 周频验证窗口一致），excluded_constant_dates=0 |
| 输出解码 | 通过 | frequency=WEEKLY_5D，horizon=5，entry=T+1_OPEN，target=future5d 开盘对开盘超额 |
| 预测完整性 | 通过 | 0 failures，56 scorecard/universe 全部生成 |

结论: 负 IC 不是符号/日期/解码 bug，可对收敛充分的模型下 VALID_NEGATIVE 判定。

## 3. 收敛诊断（关键发现）

deep cell 的 `best_epoch` / `epochs_completed` / `best_validation_mse`（来自 provenance.json）：

### iTransformer — 训练失稳

- 9 cell 中 7 个 `best_epoch=1, epochs_completed=4`；tech100 s21 `best_epoch=2`；仅 csi300 s21 `best_epoch=20`。
- 早停在 epoch 1 触发 = 验证损失在第 1 轮后恶化且未恢复。best 处 val_mse 高且跨 seed 抖动（0.045–0.161），seed 19/20 一律 epoch 1 停，seed 21 在 csi300 能训到 epoch 20 且 IC 最好（+0.0130）。
- parameter_count=73396（极小），asset_tokens=499（csi300）。判定: **FAIL_CONVERGENCE**——未收敛，IC 不可作为模型真实能力定论。需排查学习率/warmup/早停 patience/验证调度。
- 仅 csi300 s21（唯一充分训练的 cell）IC=+0.0130，提示充分训练后 iTransformer 可能非负——支持"收敛问题而非模型问题"的判断。

**根因假设（待验证）**:
- 训练循环（`torch_runtime.py:fit_protocol_safe`）无梯度裁剪、无 LR warmup，调度为 lradj type1；Adam LR=1e-4 统一。iTransformer 反转 variate-attention（对 499 asset token 做跨变量注意力）在无裁剪/warmup 时对初始化敏感，多数 seed 第 1 轮后验证损失恶化即发散。
- 对比：timexer 用 patch_len=16 patching 起正则/稳定作用，相同 LR 下训练稳定（epoch 19-24）。
- iTransformer 配置含 `class_strategy="projection"` 但缺 `task_name="long_term_forecast"`（其他模型均有），需确认是否模式不匹配。
- 拟修复（需评审标注为偏差）：为 iTransformer 增加梯度裁剪 + LR warmup 后重跑，再下定论。不因 2026 结果差而加 epoch。

### FACT / TimePro — 收敛充分但 IC 近零/负

- FACT: best_epoch 2–20，val_mse 较低较稳（0.012–0.091）；TimePro: best_epoch 1–27，val_mse 0.007–0.068。
- val_mse 与 rank_ic 不相关（FACT csi300 s21 val_mse 最低 0.0122 却 IC=-0.0138）→ MSE 过拟合，未学到排序信号。
- 判定: **VALID_NEGATIVE**（训练充分、链诊断通过、学到排序信号弱）。

### TimeXer — 收敛最稳、IC 最好

- best_epoch 一致 19–24，val_mse 稳定；tech32 IC +0.0455（三 seed 全正）。判定: **VALID**。

## 4. 全 cell 状态表

判定缩写: P=PASS, V=VALID, VN=VALID_NEGATIVE, FC=FAIL_CONVERGENCE。

| model | universe | seed | rank_ic | sign_acc | best_ep | val_mse | 判定 | pred_hash |
|-------|----------|------|---------|----------|---------|---------|------|-----------|
| ridge | csi300 | 19 | +0.0440 | 0.548 | – | – | P | dc6549b0a8cb |
| ridge | tech32 | 19 | +0.0582 | 0.527 | – | – | P | 49457c6d0cf0 |
| ridge | tech100 | 19 | +0.0542 | 0.526 | – | – | P | 3030ea20373e |
| lightgbm | csi300 | 19 | +0.0442 | 0.568 | – | – | P | 8c93c56957af |
| lightgbm | tech32 | 19 | +0.0163 | 0.497 | – | – | P | c5dd9604aee0 |
| lightgbm | tech100 | 19 | +0.0390 | 0.488 | – | – | P | 7db9bc513bd6 |
| timexer | csi300 | 19 | -0.0127 | 0.449 | 21 | 0.2363 | V | 49b0b2385888 |
| timexer | csi300 | 20 | +0.0022 | 0.504 | 21 | 0.0166 | V | 1843a617f02c |
| timexer | csi300 | 21 | +0.0241 | 0.542 | 23 | 0.0146 | V | 336ade46f736 |
| timexer | tech32 | 19 | +0.0020 | 0.504 | 23 | 0.0589 | V | 5c72bfa770ff |
| timexer | tech32 | 20 | +0.0751 | 0.519 | 23 | 0.0437 | V | 30d4421e803c |
| timexer | tech32 | 21 | +0.0592 | 0.516 | 20 | 0.0256 | V | 3c18f693146f |
| timexer | tech100 | 19 | -0.0213 | 0.497 | 24 | 0.0281 | V | 43570ae754be |
| timexer | tech100 | 20 | +0.0470 | 0.505 | 23 | 0.0601 | V | ebb9e1671fe0 |
| timexer | tech100 | 21 | -0.0175 | 0.498 | 19 | 0.0455 | V | 3a61dc44f86b |
| fact | csi300 | 19 | +0.0003 | 0.542 | 16 | 0.0751 | VN | b44d896702e8 |
| fact | csi300 | 20 | -0.0163 | 0.557 | 2 | 0.0332 | VN | 89747ffa0fae |
| fact | csi300 | 21 | -0.0138 | 0.456 | 20 | 0.0122 | VN | 8ec19f4471a6 |
| fact | tech32 | 19 | -0.0393 | 0.479 | 9 | 0.0910 | VN | 1fd413604794 |
| fact | tech32 | 20 | -0.0106 | 0.499 | 2 | 0.0340 | VN | aef90722b2e9 |
| fact | tech32 | 21 | -0.0744 | 0.483 | 10 | 0.0194 | VN | e8963b4fd6d3 |
| fact | tech100 | 19 | -0.0267 | 0.490 | 12 | 0.0751 | VN | f9f4187775ad |
| fact | tech100 | 20 | -0.0155 | 0.503 | 3 | 0.0339 | VN | 71761baa19d4 |
| fact | tech100 | 21 | -0.0212 | 0.493 | 17 | 0.0132 | VN | 51af7f0a9477 |
| timepro | csi300 | 19 | -0.0187 | 0.467 | 6 | 0.0120 | VN | a48e5c43efb3 |
| timepro | csi300 | 20 | +0.0004 | 0.467 | 23 | 0.0354 | VN | 87c56a646da7 |
| timepro | csi300 | 21 | -0.0069 | 0.457 | 27 | 0.0066 | VN | 0af2b51953bd |
| timepro | tech32 | 19 | -0.0401 | 0.488 | 3 | 0.0225 | VN | 862648cc4b85 |
| timepro | tech32 | 20 | -0.0238 | 0.505 | 4 | 0.0678 | VN | 930d20917e50 |
| timepro | tech32 | 21 | -0.0711 | 0.467 | 4 | 0.0139 | VN | b9a00824e53f |
| timepro | tech100 | 19 | +0.0059 | 0.503 | 25 | 0.0180 | VN | d0bec9fd6853 |
| timepro | tech100 | 20 | +0.0008 | 0.501 | 5 | 0.0366 | VN | b15105256777 |
| timepro | tech100 | 21 | -0.0189 | 0.491 | 1 | 0.0114 | VN | 95cde7f2619a |
| itransformer | csi300 | 19 | -0.0213 | 0.483 | 1 | 0.0599 | FC | c19dc60a733b |
| itransformer | csi300 | 20 | -0.0314 | 0.477 | 1 | 0.1385 | FC | 4135399e86d2 |
| itransformer | csi300 | 21 | +0.0130 | 0.457 | 20 | 0.0401 | V | c51959ce50e3 |
| itransformer | tech32 | 19 | -0.0671 | 0.472 | 1 | 0.0660 | FC | 59b279fabf8f |
| itransformer | tech32 | 20 | -0.0533 | 0.487 | 1 | 0.1614 | FC | 65f8c73e105f |
| itransformer | tech32 | 21 | +0.0278 | 0.515 | 1 | 0.0512 | FC | 087f383b9e3d |
| itransformer | tech100 | 19 | -0.0343 | 0.481 | 1 | 0.0447 | FC | 8bec1ce4f568 |
| itransformer | tech100 | 20 | -0.0237 | 0.503 | 1 | 0.1292 | FC | 0eca648ea62e |
| itransformer | tech100 | 21 | +0.0172 | 0.506 | 2 | 0.0450 | FC | 14902592d2ca |

注: itransformer tech32 s21 与 tech100 s21 虽 IC 为正，但 best_epoch=1/2、epochs_done=4/5，仍归 FC（未充分收敛），正 IC 不可作为定论。

## 5. A0 冻结引用

每个 cell 的 `run_manifest.json` 已绑定不可变 `prediction_hash` / `config_hash` / `code_hash`（上表 pred_hash 为前 12 位）。code_hash 按模型分组（冻结锚）:

- ridge / lightgbm: `c698c4b5e691`
- itransformer: `f423f688d965`
- fact / timexer: `74e9a9492e06`
- timepro: `921378a4766f`

A0 预测已冻结，V1 复用 A0 引用，不重训 A0。完整 hash 见服务器
`/data/yilangliu/a_share_research/runs/v0/v0/<model>/<universe>/A0/<seed>/run_manifest.json`
（tabular: `.../v0-a0-<universe>-<model>-seed-<seed>/run_manifest.json`）。

## 6. 偏差披露

- V0 打分使用轻量 `_ValidationLabel`（跳过逐行 1828 天日历哈希，~10x 加速），日历完整性在加载时一次性校验；若打分转为正式结果需在 provenance 中披露此性能偏差。
- TimePro/TimeXer 无仓库 license（NOASSERTION，用户授权，记录于 `configs/upstreams.lock.yaml`）；iTransformer 为 MIT_CLEAR，FACT 为 MIT_WITH_ATTRIBUTION_AMBIGUITY（待审）。
- 深度运行对外传入 `x_mark_enc=None, x_dec=None, x_mark_dec=None`；每资产共享 C-to-1 投影 + 外置等价可逆归一化（已记录偏差）。

## 7. 后续行动

1. **iTransformer 收敛修复（优先）**: 排查学习率/warmup/早停 patience/验证调度；当前 8/9 cell epoch 1 早停、val_mse 高抖动。修复后重跑 iTransformer V0 再下定论。
2. **FACT/TimePro**: VALID_NEGATIVE 已定；MSE 过拟合未学到排序信号，V1 信息消融可观察是否改善。
3. **V1 信息消融**: 复用 A0 引用（不重训），4 deep 模型 × A1-A3 × 4 universe × 3 seed = 144 job（s4m 阻断）。
4. 不要因 2026 结果差而加 epoch；扩展仅在所有匹配验证曲线触顶且仍改善时统一施加（见 plan 009 guardrails）。
