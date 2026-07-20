# Seven-Model A-Share Research

本目录是本期七模型、四股票池 A 股研究的唯一活动源码边界。研究目标是以
可审计的 PIT 数据协议，比较 Ridge、LightGBM、iTransformer、FACT、
TimePro、TimeXer 与 S4M，并在冻结预测之后评估风险预算和组合约束。

## 当前阶段

基础契约和 D0 的服务器端源码已经建立，但真实 D0 数据尚须通过服务器测试、
存量数据审计和增量物化门槛。尚未开始正式模型训练、回测或结果生成。历史目录
只能作为明确计划批准后的迁移来源，不能自动成为本期证据。

## 计算边界

- Mac：仅保存规格、源码和复制回的小型报告，不运行 Python、测试、数据处理或研究任务。
- 服务器：所有环境安装、测试、数据、训练、推理、回测和报告生成都必须位于
  `/data/yilangliu/a_share_research/seven_model_research`。
- 凭据：Tushare token 只存在服务器安全路径，源码不得读取、打印或复制它；未来数据客户端
  只能通过项目批准的代理入口构造。

## 目录

```text
src/a_share_research/  最小 Python 包
configs/               轻量、可提交配置
tests/foundation/      基础边界测试（仅服务器运行）
docs/specs/            本期规格快照
docs/policies/         迁移、安全和产物政策
scripts/               源码清单/哈希等非研究脚本
patches/               上游兼容补丁（当前为空）
```

D0 先运行只读 inventory audit，再生成有截断边界的请求清单。完整因果协议见
`docs/d0_protocol.md`；tech32/tech100 永久保留 `EXPLORATORY_ONLY` 标签。

## 服务器验证（同步后执行）

```bash
.venv/bin/python -m pytest -q tests/foundation
.venv/bin/ruff check src tests
.venv/bin/python -m compileall -q src tests
```

这些命令不得在 Mac 执行。Git 发布仍处于停放状态。
