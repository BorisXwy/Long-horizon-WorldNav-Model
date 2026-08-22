# NAV 项目规则与文档规范

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-010` |
| 类型 | 规则与文档规范 |
| 状态 | Active / Consolidated |
| 更新时间 | 2026-08-15 |
| 职责 | 集中维护项目硬规则、文档写作规则、验证口径和 agent 工作边界。 |

## 当前入口结论

本文件是规则类内容的合并入口。正式 V1 结构、完整链路 smoke、概念性结果标注、essay 来源链、文档格式等规则都以这里和 `AGENT.md` 为准；`AGENT.md` 面向执行者，本文件面向文档知识库。

## 合并主题

- NAV 项目硬规则与不变量
- NAV 文档系统规范

## 维护规则

- 后续相关主题优先更新本文档，不再新增细碎同主题文档。
- 历史 run、旧结构和 superseded 方案保留在对应章节中，用于追溯和对照。
- 若代码/日志/文档三线不一致，以代码与真实记录为准先修正文档。

## 合并正文


## NAV 项目硬规则与不变量


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-004` |
| 类型 | 规则声明（Project Invariants） |
| 状态 | Active / Guardrail |
| 更新时间 | 2026-08-14 |
| 职责 | 汇总当前 V1 主线中不应被普通实验随意改变的模型、数据、训练、评测和文档硬约束 |

### 当前结论

本文不是实验记录，也不是完整设计推导；它只保存“改代码、开训练、写论文时不能
悄悄违反”的硬规则。若确实要改变其中任何规则，必须先在
`decision_log.md` 新增决策，并同步修改对应设计、训练、数据和脚本文档。

### 版本边界

1. `v1_*` 是当前主线：official Wan2.1-1.3B initialization、`T_latent=4`
   micro future、Register long-history memory、action-centered interface、
   Stage One/Two/Three staged training。
2. `v0_*` 是历史与 baseline：InfiniteWorld checkpoint initialization、
   81-frame dense chunk、HPMC、早期 A/B Register 和 VBench 复现。
3. V1 正式训练不得在没有说明的情况下加载旧 RE10K A/B、旧 DL3DV A/B 或
   InfiniteWorld finetuned checkpoint。
4. InfiniteWorld checkpoint 只用于 V0 baseline、旧实验复现或显式 ablation；
   V1 正式 Stage One 默认从官方 Wan2.1 权重初始化。

### 模型不变量

1. V1 shared backbone 是 **single shared WanBlock token stream**：Register、
   generation branch 和 policy/action branch 都进入同一套 Wan2.1-style DiT
   blocks。正式主线不采用 GigaWorld-style dual-stream / MoT-style backbone，
   不复制一套 action expert transformer，也不给 policy 另建独立视觉 backbone。
   GigaWorld/DreamZero 只保留为 `A_noise -> action flow/diffusion decode` 的
   接口启发。
2. HPMC history compression 在 V1 主线中被移除或旁路；long history 由
   Register 递归承载。
3. InfiniteWorld 的 `20-channel mask` 在 V1 正式结构中删除；DiT patch stem
   回到 Wan 原生 `16 latent channels -> hidden dim`。prefix / future / action
   的身份由 token_type_embedding、position/timestep embedding 和 attention mask
   显式表达，不再用 channel concat 或独立 stream 表示。
4. latest local memory 保留；它表示最近局部视觉状态，不承担无限历史压缩职责。
5. Register 不保存 episode-specific state 到 checkpoint；checkpoint 保存的是
   RegisterCell、DiT、action interface / heads 等参数。
6. 新 episode 的初始状态固定为 `R_null`，不是 learnable episode memory；
   `R_null` 只提供固定 register slots / type-position template。第一个 history
   chunk 也通过同一个 RegisterCell 写入：
   `R_0 = RegisterCell(R_null, concat([visual_tokens(C_0), A_hist_0]))`；后续递归为
   `R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`。
7. 训练和推理中不得把 generated future 作为导航 history 回灌；导航 history
   只来自真实 observation。

### Action 与 Pose 不变量

1. 三类 action 语义必须分开：
   - `A_hist`：历史已执行 action / pseudo motion，参与 RegisterCell 更新，
     但必须以独立 action tokens / cross-attention context 的形式交互；
     禁止 action bias / latent bias 注入；
   - `A_cur`：当前 video generation condition，用于约束 future visual consequence；
   - `A_noise -> A_out`：policy/action output 路径，采用 Giga/DreamZero-style
     action flow / diffusion decode；Stage Three 才用 VLN action label 监督。
2. Stage One 不把 video pseudo action 训练成 policy；`A_out` 没有 action
   supervision。`A_cur` 必须作为 DiT condition/context 进入 video generation
   branch，并且只作用于 noisy future visual tokens，不作用于 history prefix 或
   `A_noise`。`A_noise` 作为 action token 主变量进入同一个 shared WanBlock
   token stream。
   禁止再使用“Register/local -> 旁路小 action head”冒充
   policy/video shared backbone。
3. Register 由统一的 `RegisterCell` 递归更新，不再区分 Extractor / Updater：
   `R_{-1}=R_null`，
   `R_i=RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`。
   `A_hist` 只能通过 token concat、cross-attention 或 gated token adapter 参与
   Register update；不得把 action
   embedding 加到 latent、Register、patch tokens 或 DiT hidden 上作为 additive
   bias，也不得使用 InfiniteWorld-style action bias / latent bias 方案。Pose、
   odometry 和 pseudo motion 只能用于构造 `A_hist` 标签/embedding 或训练监督，
   不作为额外 Register 输入通道。
4. `A_noise` 属于 shared WanBlock token stream 中的 noised action tokens；
   future visual tokens 可以 attend 到它，action tokens 不得 attend 到 future
   visual tokens。
5. timestep 采用 GigaWorld-Policy-style per-token 语义：
   - clean prefix / Register / local visual tokens：`t=0`；
   - future noisy visual tokens：`visual_t`，由 RFlowScheduler 采样并用于
     Stage One video flow loss；
   - `A_noise` action tokens：独立 `action_t`，默认
     `action_flow_shift=5.0`；
   - Stage One 中 `A_out` 不计算 action loss，`action_t` 只用于让 action token
     格式与 Stage Three policy/action denoise 路径对齐。
6. 默认 action horizon 固定为 `H_action = H_nav = 10`。闭环导航推理可以只执行
   denoise 后 action chunk 的第一个 action，但训练和接口统一保留 10-step
   action output。
7. Pose 不是模型推理输入。Pose 只能用于 pseudo action、3D supervision、
   数据审计和尺度校准。
8. Stage Three policy 输入不得依赖 GT pose；真实世界没有 GT pose 是默认假设。
9. 视频 pseudo action 与 VLN discrete action 的尺度和语义必须显式映射，不得
   把二者当作天然同分布标签。

### 数据与缓存不变量

1. V1 Stage One 当前正式缓存是 T4 micro latent，不是 V0 的 81-frame dense
   latent，也不是早期 sparse pack / toy route。
2. 每个 T4 micro chunk 固定：
   - RGB frames：13；
   - stride：12；
   - VAE latent time：`T_latent=4`。
3. 每个 micro chunk 必须独立调用 Wan VAE encoder。不得先整段视频 VAE 编码
   再切 latent。
4. T4 micro latent 可复用于不同 history window。样本构造时从已缓存 episode
   latent 中抽取窗口，不重复编码。
5. 对齐 InfiniteWorld 发生在 history/context span，而不是 target 长度：

   | IW-equivalent history | T4 micro history steps |
   | --- | ---: |
   | IW-1 | 7 |
   | IW-4 | 27 |
   | IW-8 | 54 |
   | IW-16 | 107 |

6. SpatialVID 数量天然占优，正式训练不得无说明地按自然文件顺序或自然样本量
   直接吞；必须记录 sampler 或数据比例。
7. 原始下载目录保持只读，并继续放在 `/sharedata/datasets/` 或对应公共
   assets 目录。自 2026-08-22 起，训练用 latent / tensor cache 的真实落盘
   位置改为个人项目目录 `NAV/data/train/<dataset>/<latent_run>/`，用于避免
   大规模 latent 挤满公共 `/sharedata`。`/sharedata/NAV/derived/` 只保留历史
   遗留、预算 manifest、轻量索引和必要的临时 render 中间态；新启动的大规模
   latent 编码任务必须默认写入 `NAV/data/train/`。

### 三阶段训练不变量

1. Stage One 是 video generation memory pretraining：
   `L_stage1 = L_visual_flow`，训练 Register long-history update 和 short
   future generation。
2. Stage Two 不是 cotrain/replay 命名；它是在 Stage One 同一视频生成前向中
   加 3D supervision/probe：
   `L_stage2 = L_stage1 + lambda_3d * L_3D`。
3. Stage Three 是 VLN policy/action supervision，并需要 replay Stage One/Two
   loss 防止 shared backbone/Register 退化：
   `L_stage3 = L_nav + lambda_video * L_visual_flow_replay + lambda_3d * L_3D_replay`。
4. Stage Three 推理是 policy-only：generation/video/3D heads 不执行。
5. 训练启动时扫描到的数据列表通常是固定的；后台新落盘 latent 不自动进入已启动
   训练，除非脚本显式支持动态 refresh，并在日志中记录。

### 评测与结论不变量

1. 不能仅凭 training loss 宣称 long-history memory 有效。
2. 声称 Register 长程有效前，至少需要：
   - normal Register vs zero Register；
   - true history vs shuffled history；
   - with local memory vs without local memory；
   - IW-1/4/8/16 分组指标；
   - teacher-forced 与 autoreg rollout 对比。
3. 声称优于 InfiniteWorld 前，必须在同一数据、同一 history span、同一生成/评测
   协议下比较。
4. 文档和 essay 中所有概念性结论必须标注 `【已验证→路径】` 或 `【未验证】`。

### 文档和运行不变量

1. 当前事实先看 `project_status.md`；路径先看 `resource_inventory.md`；原因先看
   `decision_log.md`。
2. 代码、记录、文档三线不一致时，以代码和真实 log/result 为准，先修文档，再
   继续正式实验。
3. 新训练必须记录：
   - checkpoint initialization；
   - loaded/missing/mismatched key audit；
   - data root 和 manifest；
   - history construction；
   - loss 组成；
   - batch、accumulation、optimizer、dtype；
   - save frequency；
   - TensorBoard/log/checkpoint 路径。
4. 结果类内容放在 `NAV/result/<eval_type>/<run_name>/...`；训练原始日志和
   checkpoint 放在 `NAV/log/`；不要把产物塞进 `doc/`。


## NAV 文档系统规范


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-001` |
| 类型 | 文档规范（Documentation Standard） |
| 状态 | Active |
| 更新时间 | 2026-08-11 |
| 职责 | 规定目录、命名、状态、元信息和记录生命周期 |

### 命名规则

- 目录：`NN_topic`，编号表达阅读顺序而非创建时间。
- 文件：`lower_snake_case.md`，名称表达稳定职责，不使用日期作为主文件名。
- 文档 ID：`NAV-<TYPE>-NNN`；类型包括
  `OVR/DES/ARC/DAT/TRN/EVL/OPS/RES/INS`。
- 单篇文档只允许一个 `#` 一级标题。

### 内容骨架

正式文档依次包含：

1. 标题；
2. 元信息表；
3. 当前结论、摘要或执行入口；
4. 方法、配置或证据；
5. 限制与未决问题；
6. 相关脚本、配置、结果和文档。

动态状态文档应把最新快照放在最前，历史记录按时间倒序追加。实验记录必须明确
区分计划、运行中、成功、失败和废弃。

### 信息生命周期

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

#### Insight 分析生命周期

`05_insight/insight_log.md` 遵循"问题压缩-回答"闭环：

```text
提出问题 → 写入问题寄存器（Open）
    ↓ 实验证明 / 决策采纳 / 显式废弃
问题寄存器标记 Answered / Superseded
    ↓ 所有 Open 清空
允许开启下一轮新分析
```

新分析的第一步必须引用问题寄存器确认无 `Open` 项；否则只能对已有 insight
做细化与实验设计，不得提出新的设计分析。

#### Essay 内容来源生命周期

`essay/` 必须来自已收敛的有效项目内容，不得先于验证凭空撰写：

```text
doc/（设计/规范/决策） + src/（核心代码） + log/与result/（实验运行）
    ↓ 三者收敛（设计已定、代码已实现、实验已跑出并记录）
允许写入 essay/
```

- 未收敛的设想写入 `doc/05_insight/`，不进 essay。
- essay 与 doc/代码/实验结果不一致时，以 doc 与实验为准，先修正 essay。
- essay 中引用的数字与图表应标注 `doc/` 或 `result/` 来源路径。

### 文件与产物边界

- `doc/`：可读、可复现的知识摘要。
- `essay/`：论文写作目录，存放核心 `.tex` 主文件、章节子文件、图表源码、
  草稿和投稿版本归档；不存放原始日志、checkpoint、生成视频或评测 JSON。
- `config/`：机器可读实验配置。
- `scripts/`：可执行入口。
- `log/`：原始日志、TensorBoard 和 checkpoint。
- `result/`：生成视频、评测输入输出和汇总指标。
- `/sharedata/NAV/derived/`：可重新生成的数据派生物。

### 唯一事实来源分工

| 信息 | 文档 |
| --- | --- |
| 当前状态总览 | `00_overview/project_status.md` |
| 当前 V1 不变量 / 硬规则 | `00_overview/project_rules_and_documentation_standard.md` |
| V1 模型总体接口 | `01_model/model_evolution_and_current_architecture.md` |
| V1 待定结构与默认选择 | `01_model/model_evolution_and_current_architecture.md` |
| 数据 tensor 与窗口语义 | `02_data/data_preparation_schema_and_status.md` |
| V1 action / geometry / sample schema | `02_data/data_preparation_schema_and_status.md` |
| Pose/Action 数学定义 | `02_data/data_preparation_schema_and_status.md` |
| 实时运行和数量 | `02_data/data_preparation_schema_and_status.md` |
| Register在线训练语义历史说明 | `03_training/training_plan_and_experiment_log.md` |
| V1 三阶段数据、loss 与配比 | `03_training/training_plan_and_experiment_log.md` |
| V1 Stage One T4/IW 对齐训练 | `03_training/training_plan_and_experiment_log.md` |
| 路径 | `06_operations/resource_inventory.md` |
| 决策原因 | `00_overview/decision_log.md` |

规范文档保存稳定语义，不复制频繁变化的完成数量；Live 文档保存当前数字，不重新
定义算法。若代码与文档不一致，先停止正式实验，核对后同步修改规范、配置和脚本。

### V0/V1 分层

- `v1_*`：当前主线。默认围绕官方 Wan2.1-1.3B init、`T_latent=4` micro
  target、Register long-history memory、action-centered interface 和三阶段训练。
- `v0_*`：历史方案或 baseline。默认围绕 InfiniteWorld checkpoint init、
  81-frame dense chunk、HPMC 对照、早期 A/B Register 结构和 VBench 复现。
- 通用文档不加版本前缀，但必须在“职责”或“当前结论”里说明它服务当前主线、
  历史追溯，还是路径/资源状态。
