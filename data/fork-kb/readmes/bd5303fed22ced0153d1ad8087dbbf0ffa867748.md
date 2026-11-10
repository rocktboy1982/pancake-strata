# Local Inference Benchmarks

[English](README.en.md) · [测试报告](reports/README.md) · [硬件档案](hardware/README.md) · [测试方法](methodology/README.md)

我的本地 AI 推理硬件测试记录。持续比较不同硬件、模型、量化和推理引擎在实际输入长度与并发条件下的表现。R9700 是第一套已发布配置，后续硬件沿用相同目录结构。

## 已发布测试

| 日期 | 硬件 | 模型与引擎 | 测试范围 | 报告 |
|---|---|---|---|---|
| 2026-11-08 | 4 × R9700 32 GB / EPYC 7452 | Qwen3.8-Flash-Next；Radiance 1.15.1 vs Strata 0.1.58 UD-Q4_K_XL | 短输入至 240K；1 / 8 / 16 客户端，覆盖范围见报告 | [中文](reports/2026-11-08-r9700-radiance-vs-strata/README.zh-CN.md) · [English](reports/2026-11-08-r9700-radiance-vs-strata/README.md) |

这次短输入单路生成速度为 **217.0 vs 139.3 tok/s**，16 客户端总吞吐为 **1094.1 vs 329.7 tok/s**；16K 冷输入有效 prefill 为 **7786.6 vs 2776.7 tok/s**。两套量化不同，Strata 使用 8 个活动槽，16 客户端结果包含排队。速度结果不代表质量等同，也不代表所有输入长度都测了 16 路。

## 目录

```text
hardware/       每套已测试硬件的配置档案
reports/        按日期和实验主题归档的报告
  <实验编号>/
    README.md           英文报告
    README.zh-CN.md      中文报告
    configuration.json  本次测试的实际配置快照
    data/               脱敏后的请求、批次、汇总数据
    assets/             本次报告图表
methodology/    指标定义与对比要求
templates/      新硬件、新实验的填写模板
```

新增硬件时复制 [hardware.json](templates/hardware.json)，填写实测配置并分配稳定的硬件编号；新增实验时使用 [report.md](templates/report.md)，把硬件编号、版本、权重、上下文、并发和缓存状态一起记录。历史报告保留当次配置快照，避免硬件升级后改变旧结果的含义。尚未测试的硬件不填入成绩表。

原 Strata 源码与完整历史保留在 [归档分支](https://github.com/Dylan997S/benchmarks-strata/tree/archive/strata-source-20261108)；本仓库首页用于独立测试记录。此前发布的报告分支仍可访问。本仓库不代表任何引擎的官方成绩。
