# NAV 文档中心

本目录是 NAV 的设计、数据、训练、评测和运行知识库。原始日志、checkpoint 与
生成结果不进入文档目录，分别保存在 `NAV/log/` 和 `NAV/result/`。

## 当前主线一页读完

现在的正式主线是 **V1 Stage One 完整版**：

```text
Wan2.1-T2V-1.3B official init
  + T_latent=4 short future generation
  + Register long-history memory
  + action-centered interface
  + Stage One/Two/Three staged training
```

最短阅读路径：

| 顺序 | 读什么 | 文档 |
| ---: | --- | --- |
| 1 | 当前做到哪、哪些进程在跑、下一步是什么 | `00_overview/project_status.md` |
| 2 | 哪些规则不能被普通实验悄悄改掉 | `00_overview/project_invariants.md` |
| 3 | 为什么从 V0 改到 V1、每次关键选择是什么 | `00_overview/decision_log.md` |
| 4 | V1 模型输入/输出、Register、action interface 怎么组成 | `01_design/v1_action_centered_io_interface.md` |
| 5 | V1 还有哪些结构参数待定，当前默认选择是什么 | `01_design/v1_open_design_questions.md` |
| 6 | Stage One/Two/Three 各用哪些数据、变量和 loss | `04_training/v1_three_stage_training_data_plan.md` |
| 7 | 当前 Stage One `T_latent=4` 如何对齐 InfiniteWorld history | `04_training/v1_stage_one_t4_iw_aligned.md` |
| 8 | 新版完整模型三阶段训练/推理链路是否跑通 | `05_evaluation/v1_full_pipeline_smoke.md` |
| 9 | 数据、权重、日志、结果绝对路径 | `06_operations/resource_inventory.md` |

## 文档分层

### V1 当前主线

| 问题 | 唯一入口 |
| --- | --- |
| 当前 V1 有哪些不可悄悄违反的硬规则 | `00_overview/project_invariants.md` |
| 当前模型到底改了什么 | `01_design/v1_action_centered_io_interface.md` |
| 当前训练如何从官方 Wan2.1 初始化 | `00_overview/decision_log.md` 的 DEC-030 |
| 当前 Stage One generation target 为什么是 `T_latent=4` | `04_training/v1_stage_one_t4_iw_aligned.md` |
| 三阶段各用哪些数据、变量、loss、配比 | `04_training/v1_three_stage_training_data_plan.md` |
| V1 action、geometry、sample schema | `03_data/v1_action_geometry_schema.md` |
| 原始视频怎样变成 T4 latent/window | `03_data/training_data_construction.md` |
| Sparse Pose 怎样变成逐帧 Action | `03_data/action_annotation_specification.md` |
| 当前数据下载、latent、训练运行状态 | `03_data/dataset_preparation_status.md` |
| VLN/R2R/RxR/ScaleVLN/LHPR 准备状态 | `03_data/vln_dataset_preparation_status.md` |
| 当前完整模型 smoke 和结构审计 | `05_evaluation/v1_full_pipeline_smoke.md` |

### V0 历史与 baseline

| 问题 | 入口 |
| --- | --- |
| 旧版 InfiniteWorld/Register A/B 方案 | `01_design/v0_stage_one_register_world_model.md` |
| 旧版多 chunk teacher-forcing 语义 | `04_training/v0_streaming_training_sample_semantics.md` |
| 旧 RE10K / DL3DV / SpatialVID 训练记录 | `04_training/v0_register_re10k_experiments.md`、`04_training/v0_stage_one_dl3dv_experiments.md` |
| InfiniteWorld 代码结构、HPMC 和论文一致性 | `02_architecture/v0_infinite_world_implementation.md` |
| VBench 复现协议与旧 A/B/InfiniteWorld 对比 | `05_evaluation/v0_vbench_infinite_world_protocol.md` |

### 外部工作、复现和调研

| 问题 | 入口 |
| --- | --- |
| DreamZero / Wan2.2 5B backbone latency 和结构 | `02_architecture/v1_dreamzero_wan22_5b_backbone.md` |
| GigaWorld policy DiT 输入维度和 co-training 思路 | `02_architecture/v1_gigaworld_policy_dit_inputs.md` |
| BridgeVLA++ memory 复现 | `02_architecture/v1_bridgevla_plus_memory_architecture.md` |
| Video/action co-train WAM 调研 | `07_research/v1_wam_representation_only_cotrain.md` |
| R2R/RxR 最新 SOTA 和 Qwen-RobotNav 对比 | `07_research/r2r_rxr_benchmark_sota.md` |

## 版本命名约定

- `v1_*`：当前主线。默认指官方 Wan2.1-1.3B init、`T_latent=4`、Register
  long-history memory、action-centered interface、Stage One/Two/Three 新三阶段。
- `v0_*`：历史与对照。默认指 InfiniteWorld checkpoint init、81-frame dense
  chunk、HPMC 对照、早期 A/B Register 结构和 VBench baseline。
- 无 `v0/v1` 前缀：通用规范、路径资源、运行状态、benchmark 背景，或历史兼容入口。

## 文档注册表

| ID | 状态 | 类型 | 路径 |
| --- | --- | --- | --- |
| NAV-OVR-001 | Active | 文档规范 | `00_overview/documentation_standard.md` |
| NAV-OVR-002 | Live | 决策日志 | `00_overview/decision_log.md` |
| NAV-OVR-003 | Live / Executive Summary | 项目状态 | `00_overview/project_status.md` |
| NAV-OVR-004 | Active / Guardrail | 规则声明 | `00_overview/project_invariants.md` |
| NAV-DES-001 | Active / Action-centered Principle Added | 设计规范 | `01_design/v0_stage_one_register_world_model.md` |
| NAV-DES-002 | Proposed / Confirmed Semantics / Action-centered Mask | 设计规范 | `01_design/v0_stage_two_3d_supervision.md` |
| NAV-DES-003 | Proposed / + Working Hypothesis DEC-022 / DEC-023 | 设计规范 | `01_design/v0_stage_three_navigation.md` |
| NAV-DES-004 | Proposed / Interface Draft | 设计规范 | `01_design/v1_action_centered_io_interface.md` |
| NAV-DES-005 | Live / Open Questions | 设计待定项登记表 | `01_design/v1_open_design_questions.md` |
| NAV-ARC-001 | Verified | 架构解析 | `02_architecture/v0_infinite_world_implementation.md` |
| NAV-ARC-002 | Verified | 架构解析 | `02_architecture/vggt_omega_register_architecture.md` |
| NAV-ARC-003 | Verified / Synthetic Core Probe | 架构解析 / Latency Probe | `02_architecture/v1_dreamzero_wan22_5b_backbone.md` |
| NAV-ARC-004 | Verified / Smoke Reproduced | 架构解析 / 复现准备 | `02_architecture/v1_bridgevla_plus_memory_architecture.md` |
| NAV-ARC-005 | Verified from Local Code / Smoke Reproduced | 架构解析 / DiT输入维度 | `02_architecture/v1_gigaworld_policy_dit_inputs.md` |
| NAV-DAT-001 | Reference | 数据调研 | `03_data/long_video_dataset_survey.md` |
| NAV-DAT-002 | Active | 数据规范 | `03_data/dataset_preparation_specification.md` |
| NAV-DAT-003 | Live | 数据状态 | `03_data/dataset_preparation_status.md` |
| NAV-DAT-004 | Running | 标注协议 | `03_data/kinetics_vggt_annotation.md` |
| NAV-DAT-005 | Active / Source of Truth | 数据构建规范 | `03_data/training_data_construction.md` |
| NAV-DAT-006 | Active / Source of Truth | Action标注规范 | `03_data/action_annotation_specification.md` |
| NAV-DAT-007 | Live | VLN数据状态 | `03_data/vln_dataset_preparation_status.md` |
| NAV-DAT-008 | Proposed / V1 Draft | 数据与实现 Schema | `03_data/v1_action_geometry_schema.md` |
| NAV-TRN-001 | Active | 训练规范 | `04_training/v0_multidataset_curriculum.md` |
| NAV-TRN-002 | Historical Baseline | 实验记录 | `04_training/v0_register_re10k_experiments.md` |
| NAV-TRN-003 | Active / Source of Truth | 训练样本语义 | `04_training/v0_streaming_training_sample_semantics.md` |
| NAV-TRN-004 | Proposed / Confirmed Semantics | 训练规范 | `04_training/v0_stage_two_3d_supervision_training.md` |
| NAV-TRN-005 | Running A / B ebs16 1000 steps | 训练实验规范 | `04_training/v0_stage_one_dl3dv_experiments.md` |
| NAV-TRN-006 | Accepted Draft / V1 Final Training Skeleton | 训练数据与损失设计 | `04_training/v1_three_stage_training_data_plan.md` |
| NAV-TRN-007 | Implemented / Running Formal Stage One | 训练规范 | `04_training/v1_stage_one_t4_iw_aligned.md` |
| NAV-EVL-001 | Verified Baseline | 评测协议 | `05_evaluation/v0_vbench_infinite_world_protocol.md` |
| NAV-EVL-002 | StreamVLN 100/500 + DualVLN 100 Done | 评测复现协议 | `05_evaluation/r2r_ce_sota_reproduction.md` |
| NAV-EVL-003 | Smoke Reproduced | 外部基线复现 | `05_evaluation/v1_bridgevla_plus_memorybench_reproduction.md` |
| NAV-EVL-004 | Smoke Reproduced | 外部 WAM 基线复现 | `05_evaluation/v1_fastwam_gigaworld_reproduction.md` |
| NAV-EVL-005 | Smoke Verified | 完整模型链路验证 | `05_evaluation/v1_full_pipeline_smoke.md` |
| NAV-OPS-001 | Live | 资源总账 | `06_operations/resource_inventory.md` |
| NAV-RES-001 | Reference | 外部调研 | `07_research/world_model_training_efficiency.md` |
| NAV-RES-002 | Verified Snapshot / External Results | Benchmark调研 | `07_research/r2r_rxr_benchmark_sota.md` |
| NAV-RES-003 | Active / Reproduction Targets Selected | WAM调研 / 复现候选 | `07_research/v1_wam_representation_only_cotrain.md` |
| NAV-INS-001 | Active / 受规则约束 | Insight 记录 | `08_insight/insight_log.md` |

## 目录职责

```text
00_overview/     文档规范、术语、决策日志
01_design/       NAV 当前采用的模型设计与阶段边界
02_architecture/ 外部论文、本地参考仓库和实现解析
03_data/         数据调研、规范、状态和标注协议
04_training/     训练配置、课程和实验记录
05_evaluation/   固定评测协议、基线和指标记录
06_operations/   路径、环境、权重与运行维护
07_research/     尚未成为项目配置的外部调研
08_insight/      设计 insight 与改进分析（受问题压缩-回答闭环规则约束）
_templates/      新文档和新实验记录模板
```

## 其他 NAV 顶层目录

```text
essay/   论文写作目录：核心 .tex 主文件、章节子文件、图表源码、草稿和投稿
         版本归档。不存放原始日志、checkpoint、生成视频或评测 JSON；论文
         中引用的数字与图表应回溯到 doc/ 与 result/ 中的稳定记录。essay
         内容必须来自 doc/ 与核心代码/实验收敛后的有效项目内容，未收敛的
         设想写在 doc/08_insight/，不进 essay。
config/  机器可读实验配置
scripts/ 可执行入口与评测工具
src/     NAV 源代码
log/     原始训练日志、TensorBoard 与 checkpoint
result/  生成视频、评测输入输出和汇总指标
```

## 状态含义

- `Active`：当前有效的设计或规范。
- `Live`：随外部运行状态持续更新，是当前事实来源。
- `Running`：对应任务正在执行。
- `Verified`：已与代码、论文或实际运行核对。
- `Reference`：提供背景，不直接定义当前实验。
- `Historical Baseline`：历史结果，只用于复现和比较。
- `Superseded`：已被新方案替代，保留追溯信息。

## 信息更新责任

一次数据或训练改动应按以下顺序同步：

```text
方法语义变化
├── 数据构建 → NAV-DAT-005
├── Action / geometry schema → NAV-DAT-008
├── Pose/Action 标注数学 → NAV-DAT-006
├── V1 模型接口 → NAV-DES-004
├── V1 待定项 → NAV-DES-005
├── V1 三阶段数据、loss、配比 → NAV-TRN-006
└── V1 Stage One T4/IW 对齐训练 → NAV-TRN-007
        ↓
机器配置 → config/
        ↓
执行入口 → scripts/
        ↓
实时数字 → NAV-DAT-003
        ↓
采用原因 → NAV-OVR-002
        ↓
绝对路径 → NAV-OPS-001
```

历史实验记录不得承担当前规范职责。当前数字只在 Live 文档维护，其他文档引用
其路径，避免复制后过期。

V0 文档只在复现旧实验、解释历史曲线或建立 baseline 时更新；V1 当前主线的
新选择必须落到 V1 设计/训练文档与 `decision_log.md`。
