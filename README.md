# NAV

本目录是 `3d_wm_vln` 项目中的 navigation/world-model 工作区，包含当前 V1 Register + action-centered world model / VLN 方案的代码、脚本、配置、文档和论文草稿。设计、数据、训练、评测和运行知识库统一放在 `doc/`；原始日志、checkpoint 与生成结果分别保存在 `log/` 和 `result/`。

## 当前主线一页读完

当前正式主线是 **V1 shared WanBlock WorldNav**：

```text
Wan2.1-T2V-1.3B official init
  + T_latent=4 short future generation
  + Register long-history memory
  + action-centered interface
  + single shared WanBlock token stream
  + Stage One / Stage Two / Stage Three staged training
```

最短阅读路径：

| 顺序 | 读什么 | 文档 |
| ---: | --- | --- |
| 1 | 当前做到哪、哪些进程在跑、下一步是什么 | `doc/00_overview/project_status.md` |
| 2 | 哪些规则不能被普通实验悄悄改掉 | `doc/00_overview/project_rules_and_documentation_standard.md` |
| 3 | 为什么从 V0 改到 V1、每次关键选择是什么 | `doc/00_overview/decision_log.md` |
| 4 | 最终正式结构、训练与 acceptance 硬标准 | `doc/00_overview/final_formal_training_standard.md` |
| 5 | 当前模型结构、输入输出、Register、action interface、历史结构尝试 | `doc/01_model/model_evolution_and_current_architecture.md` |
| 6 | 数据集、T4 latent、window、pose/action 标注、VLN 渲染状态 | `doc/02_data/data_preparation_schema_and_status.md` |
| 7 | Stage One/Two/Three 训练计划、loss、课程、实验记录 | `doc/03_training/training_plan_and_experiment_log.md` |
| 8 | VBench/iWorldBench/R2R/RxR、外部复现、SOTA 调研 | `doc/04_evaluation/evaluation_reproduction_and_benchmarks.md` |
| 9 | 设计 insight、开放问题、未收敛思路 | `doc/05_insight/insight_log.md` |
| 10 | 数据、权重、日志、结果绝对路径 | `doc/06_operations/resource_inventory.md` |

## 当前文档体系

| 主题 | 唯一入口 | 包含内容 |
| --- | --- | --- |
| 项目状态 | `doc/00_overview/project_status.md` | 当前 run、进度、下一步、风险 |
| 项目规则 | `doc/00_overview/project_rules_and_documentation_standard.md` | 架构硬规则、完整 smoke 规则、概念性结果标注、文档规范 |
| 最终正式标准 | `doc/00_overview/final_formal_training_standard.md` | 最终结构、token/mask/action、混合数据、history、loss、ebs 与 acceptance 口径 |
| 决策日志 | `doc/00_overview/decision_log.md` | DEC 记录、方案变更原因、被替代方案 |
| 模型与接口 | `doc/01_model/model_evolution_and_current_architecture.md` | V0/V1 结构尝试、当前 shared WanBlock、Register/action/text/video token layout、InfiniteWorld / VGGT-Ω / DreamZero / GigaWorld / BridgeVLA++ 解析 |
| 数据与 Schema | `doc/02_data/data_preparation_schema_and_status.md` | 数据下载状态、T4 latent 规则、Action/Geometry schema、pose/action 标注、VLN 渲染、Kinetics VGGT 标注 |
| 训练 | `doc/03_training/training_plan_and_experiment_log.md` | 三阶段训练数据与 loss、Stage1/2/3 课程、RE10K pose-only formal、video-only 对照、V0 历史实验 |
| 评测与复现 | `doc/04_evaluation/evaluation_reproduction_and_benchmarks.md` | 完整 pipeline smoke、VBench/iWorldBench、R2R/RxR、外部 WAM/VLA 复现和 SOTA 调研 |
| Insight | `doc/05_insight/insight_log.md` | 尚未变成正式决策的设计分析与问题寄存器 |
| 资源总账 | `doc/06_operations/resource_inventory.md` | sharedata、权重、环境、日志、结果、tensorboard 端口与运行资源 |

Agent 执行规则入口在根目录 `AGENT.md`，与 `doc/00_overview/project_rules_and_documentation_standard.md`
共同约束代码、文档、实验和写作工作。

## 目录职责

```text
doc/00_overview/   项目状态、规则、决策日志
doc/01_model/      模型结构、接口、历史结构尝试、外部 backbone 解析
doc/02_data/       数据准备、schema、标注、VLN 渲染状态
doc/03_training/   训练计划、课程、loss、实验记录
doc/04_evaluation/ 评测协议、外部复现、benchmark / SOTA 调研
doc/05_insight/    设计 insight 与开放问题
doc/06_operations/ 路径、环境、权重、日志和资源索引
doc/_templates/    新文档和新实验记录模板
```

## 文档注册表

| ID | 状态 | 类型 | 路径 |
| --- | --- | --- | --- |
| NAV-OVR-002 | Live | 决策日志 | `doc/00_overview/decision_log.md` |
| NAV-OVR-003 | Live / Executive Summary | 项目状态 | `doc/00_overview/project_status.md` |
| NAV-OVR-010 | Active / Consolidated | 规则与文档规范 | `doc/00_overview/project_rules_and_documentation_standard.md` |
| NAV-OVR-020 | Live / Binding Standard | 最终正式结构与训练标准 | `doc/00_overview/final_formal_training_standard.md` |
| NAV-MDL-001 | Live / Source of Truth | 模型结构与接口总览 | `doc/01_model/model_evolution_and_current_architecture.md` |
| NAV-DAT-010 | Live / Source of Truth | 数据准备、Schema 与状态总览 | `doc/02_data/data_preparation_schema_and_status.md` |
| NAV-TRN-010 | Live / Source of Truth | 训练计划与实验记录总览 | `doc/03_training/training_plan_and_experiment_log.md` |
| NAV-TRN-012 | Completed / Weekly Snapshot | 2026-08-19 至 2026-08-26 周进展 | `doc/03_training/weekly_progress_report_20260826.md` |
| NAV-EVL-010 | Live / Reference | 评测、复现与 Benchmark 总览 | `doc/04_evaluation/evaluation_reproduction_and_benchmarks.md` |
| NAV-INS-001 | Active / 受规则约束 | Insight 记录 | `doc/05_insight/insight_log.md` |
| NAV-OPS-001 | Live | 资源总账 | `doc/06_operations/resource_inventory.md` |

## 旧文档合并关系

- 旧 `doc/01_design/` 与 `doc/02_architecture/` 已合并到 `doc/01_model/model_evolution_and_current_architecture.md`。
- 旧 `doc/03_data/` 已合并到 `doc/02_data/data_preparation_schema_and_status.md`。
- 旧 `doc/04_training/` 已合并到 `doc/03_training/training_plan_and_experiment_log.md`。
- 旧 `doc/05_evaluation/` 与 `doc/07_research/` 已合并到 `doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`。
- 旧 旧 08_insight/insight_log.md 已迁移到 `doc/05_insight/insight_log.md`。
- 旧 documentation standard 与 project invariants 两个规则文档已合并到 `doc/00_overview/project_rules_and_documentation_standard.md`。

## 其他 NAV 顶层目录

```text
essay/   论文写作目录；内容必须来自 doc/、code、log/result 已收敛内容。
config/  机器可读实验配置。
scripts/ 可执行入口与评测工具。
src/     NAV 源代码。
data/    个人目录下的训练 latent/cache 入口；真实大文件保存在此处但被 Git 忽略。
log/     原始训练日志、TensorBoard 与 checkpoint。
result/  生成视频、评测输入输出和汇总指标。
```

## V1 源代码组装入口

正式 V1 代码按“模块定义—组装入口—训练/推理脚本”分层：

```text
src/nav/v1/model/
  world_model.py   shared Wan/Register/Video/Pose 正式模型定义
  policy.py        Stage3 policy head 与完整 wrapper
  builder.py       build_model()；统一恢复 checkpoint 并组装最终模型

src/nav/v1/data/
  stage2.py        RE10K/SpatialVID/DL3DV mixed-window 数据实现
  r2r.py           R2R window 与自然分布 loader
  sampler.py       可替换采样策略，如四类严格均衡 sampler
  loader.py        build_data_loader()；统一组装数据流水线
  config.py        checkpoint/config 的统一反序列化

scripts/
  只负责参数、optimizer、训练循环、推理循环和产物记录；正式入口通过
  build_model() / build_data_loader() 获取模型和数据，不再复制构图代码。
```

新增实验优先注册新的 model assembler 或 data loader/sampler，再由脚本选择名称
完成组合。`nav.v1.stage2_final`、`nav.v1.stage3_r2r` 和
`nav.v1.stage3_single_action` 仅保留旧脚本 import 兼容，不再承载 canonical 实现。
本次迁移的逐 tensor、完整 checkpoint 和 Wan 前向等价验证记录位于
`result/v1_modular_refactor/assembly_equivalence_20260827/report.md`。

## 更新原则

1. 新内容优先进入对应主题总文档，不再新增碎片文件。
2. 若一个主题会长期独立维护且正文过大，才新增子文档，并在本 README 注册。
3. 方案变化必须同步 `doc/00_overview/decision_log.md`。
4. 路径、权重、日志、端口必须同步 `doc/06_operations/resource_inventory.md`。
5. 概念性结果必须标注 `【已验证→...】` 或 `【未验证】`。
