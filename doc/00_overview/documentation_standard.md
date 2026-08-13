# NAV 文档系统规范

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-001` |
| 类型 | 文档规范（Documentation Standard） |
| 状态 | Active |
| 更新时间 | 2026-08-11 |
| 职责 | 规定目录、命名、状态、元信息和记录生命周期 |

## 命名规则

- 目录：`NN_topic`，编号表达阅读顺序而非创建时间。
- 文件：`lower_snake_case.md`，名称表达稳定职责，不使用日期作为主文件名。
- 文档 ID：`NAV-<TYPE>-NNN`；类型包括
  `OVR/DES/ARC/DAT/TRN/EVL/OPS/RES/INS`。
- 单篇文档只允许一个 `#` 一级标题。

## 内容骨架

正式文档依次包含：

1. 标题；
2. 元信息表；
3. 当前结论、摘要或执行入口；
4. 方法、配置或证据；
5. 限制与未决问题；
6. 相关脚本、配置、结果和文档。

动态状态文档应把最新快照放在最前，历史记录按时间倒序追加。实验记录必须明确
区分计划、运行中、成功、失败和废弃。

## 信息生命周期

```text
Research Note
    ↓ 被采用
Design / Specification
    ↓ 实现与运行
Experiment / Evaluation Record
    ↓ 形成稳定结论
Decision Log + Active Design
```

被替代文档保留原内容，将状态改为 `Superseded`，并链接替代决策和新文档。

### Insight 分析生命周期

`08_insight/insight_log.md` 遵循"问题压缩-回答"闭环：

```text
提出问题 → 写入问题寄存器（Open）
    ↓ 实验证明 / 决策采纳 / 显式废弃
问题寄存器标记 Answered / Superseded
    ↓ 所有 Open 清空
允许开启下一轮新分析
```

新分析的第一步必须引用问题寄存器确认无 `Open` 项；否则只能对已有 insight
做细化与实验设计，不得提出新的设计分析。

### Essay 内容来源生命周期

`essay/` 必须来自已收敛的有效项目内容，不得先于验证凭空撰写：

```text
doc/（设计/规范/决策） + src/（核心代码） + log/与result/（实验运行）
    ↓ 三者收敛（设计已定、代码已实现、实验已跑出并记录）
允许写入 essay/
```

- 未收敛的设想写入 `doc/08_insight/`，不进 essay。
- essay 与 doc/代码/实验结果不一致时，以 doc 与实验为准，先修正 essay。
- essay 中引用的数字与图表应标注 `doc/` 或 `result/` 来源路径。

## 文件与产物边界

- `doc/`：可读、可复现的知识摘要。
- `essay/`：论文写作目录，存放核心 `.tex` 主文件、章节子文件、图表源码、
  草稿和投稿版本归档；不存放原始日志、checkpoint、生成视频或评测 JSON。
- `config/`：机器可读实验配置。
- `scripts/`：可执行入口。
- `log/`：原始日志、TensorBoard 和 checkpoint。
- `result/`：生成视频、评测输入输出和汇总指标。
- `/sharedata/NAV/derived/`：可重新生成的数据派生物。

## 唯一事实来源分工

| 信息 | 文档 |
| --- | --- |
| 当前状态总览 | `00_overview/project_status.md` |
| 当前 V1 不变量 / 硬规则 | `00_overview/project_invariants.md` |
| V1 模型总体接口 | `01_design/v1_action_centered_io_interface.md` |
| V1 待定结构与默认选择 | `01_design/v1_open_design_questions.md` |
| 数据 tensor 与窗口语义 | `03_data/training_data_construction.md` |
| V1 action / geometry / sample schema | `03_data/v1_action_geometry_schema.md` |
| Pose/Action 数学定义 | `03_data/action_annotation_specification.md` |
| 实时运行和数量 | `03_data/dataset_preparation_status.md` |
| Register在线训练语义历史说明 | `04_training/v0_streaming_training_sample_semantics.md` |
| V1 三阶段数据、loss 与配比 | `04_training/v1_three_stage_training_data_plan.md` |
| V1 Stage One T4/IW 对齐训练 | `04_training/v1_stage_one_t4_iw_aligned.md` |
| 路径 | `06_operations/resource_inventory.md` |
| 决策原因 | `00_overview/decision_log.md` |

规范文档保存稳定语义，不复制频繁变化的完成数量；Live 文档保存当前数字，不重新
定义算法。若代码与文档不一致，先停止正式实验，核对后同步修改规范、配置和脚本。

## V0/V1 分层

- `v1_*`：当前主线。默认围绕官方 Wan2.1-1.3B init、`T_latent=4` micro
  target、Register long-history memory、action-centered interface 和三阶段训练。
- `v0_*`：历史方案或 baseline。默认围绕 InfiniteWorld checkpoint init、
  81-frame dense chunk、HPMC 对照、早期 A/B Register 结构和 VBench 复现。
- 通用文档不加版本前缀，但必须在“职责”或“当前结论”里说明它服务当前主线、
  历史追溯，还是路径/资源状态。
