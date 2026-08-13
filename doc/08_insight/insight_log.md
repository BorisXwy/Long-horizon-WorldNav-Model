# NAV 设计 Insight 与改进分析

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-INS-001` |
| 类型 | Insight 记录（Insight Log） |
| 状态 | Active / 受规则约束 |
| 更新时间 | 2026-08-08（Asia/Shanghai） |
| 职责 | 记录对 NAV idea 与网络设计的分析、改进 insight，以及"问题压缩-回答"闭环规则 |

## 结论与规则

本文件是 NAV 的设计 insight 与改进分析记录。为保证分析不发散、不堆积
悬空问题，本文件遵循以下规则：

1. **每次分析必须先压缩历史问题。** 任何一轮新分析开始前，必须把此前
   所有轮次提出的问题压缩进"问题寄存器"表，并标注回答状态
   （`Answered` / `Open` / `Superseded`）。
2. **所有历史问题被回答前不得开新分析。** 只有当问题寄存器中所有
   `Open` 问题都变为 `Answered`（含实验证明、决策采纳或显式废弃）后，
   才允许开启下一轮新分析。新分析的第一步就是引用问题寄存器确认无
   `Open` 项。
3. **Insight 主体必须详细。** 每条 insight 必须包含：动机、对应已调研
   工作、具体改法（不动 Wan2.1 backbone）、预期收益、可验证实验指针、
   风险。空泛的"可以考虑"不算 insight。
4. **insight 不等于决策。** insight 是候选改进；只有被采纳并写入
   `01_design/` 或 `04_training/` 并在 `decision_log.md` 记录后，才成为
   正式方案。本文件保持 `Active` 但不承担当前规范职责。
5. **问题被回答后压缩归档。** 已 `Answered` 的问题保留一行摘要与指针
   （指向 experiment.md 或 decision_log.md），不重复展开。

## 问题寄存器

下表压缩本文件历史上提出的所有问题及其回答状态。新分析前先核对此表。

| 轮次 | 问题 | 状态 | 回答指针 |
| --- | --- | --- | --- |
| R1 | Register 是否真编码视觉+物理理解而非表象压缩？ | Open（仅设计） | `essay/experiment/experiment.md` Round 1 设计（E1.1/E1.2/E1.3）；**无对应代码、无 result 产物**，几何 probe/动作条件预测均未落地 |
| R2 | 3D 监督是否必要，纯预测记忆是否已够？ | Open（仅设计） | `essay/experiment/experiment.md` Round 2 设计（E2.1/E2.2/E2.3）；**Stage Two 未实现**，无 3D 监督消融/revisit 产物 |
| R3 | teacher forcing 与推理的 exposure bias，Register 是否迁移？ | Open（部分 proxy） | `essay/experiment/experiment.md` Round 3 设计（E3.1–E3.4）；仅 3-chunk autoreg/gthist 短程 proxy（记录 7/8），长程在线迁移/scheduled sampling 未测 |
| R4 | 固定预算 Register 超长 horizon 是否退化 / 容量 scaling？ | Open（部分 proxy） | `essay/experiment/experiment.md` Round 4 设计（E4.1–E4.3）；仅 2/3-chunk VBench，NAV 16+ chunk autoreg 未跑，预算 ablation 未做 |
| R5 | 选哪一层的特征给导航？（开放问题） | Open（仅设计） | `essay/experiment/experiment.md` Round 5 设计（E5.1）；**Stage Three 未实现**，逐层导航消融未做；LoGoPlanner 作可行性先验 |
| R6 | 3D-aware Register 到底怎么用进导航策略？ | Open（仅设计） | `essay/relatedworks/related_works_survey.md` LoGoPlanner 条目；Stage Three 接 Navigation Head 只读不写为设计，**未实现** |
| R7 | Register 长程容量/漂移是否有结构性解法？ | Open | 见下文 Insight 1+2（几何知情 read-before-write + sink Register），待采纳或实验验证 |
| R8 | 3D 监督表示与流式因果性是否匹配？ | Open | 见下文 Insight 3（尺度不变表示 + 流式 3D teacher），待采纳或实验验证 |
| R9 | action 表示是否与几何监督同空间？ | Open | 见下文 Insight 5（action raymap 化），待采纳或实验验证 |
| R10 | exposure bias 是否有训练侧结构性解法？ | Open | 见下文 Insight 6（self-generated history 替换），待采纳或实验验证 |
| R11 | 首 chunk Extractor 是否引入参考帧偏置？ | Open | 见下文 Insight 3+4（无参考帧 probe + 流式 teacher），待采纳或实验验证 |
| R12 | agentic 导航的指令切换如何在 Register 上落地？ | Open | 见下文 Insight 7（Register-recache），待采纳或实验验证 |
| R13 | 长程下"递归更新式 Register"是否优于"HPMC 启发式窗口子采样"？ | Open | 短程（≤3 chunk, T_in≤80）两者容量等价、比不出差别；须长程训练（history 课程扩到 16+ chunk）+ 长程 autoreg eval（16+ chunk, NAV vs HPMC 同条件）才能证明。当前 updater 仅训过 ≤3 次更新，长程下 OOD，须先长程训练。见 `doc/04_training/v0_stage_one_dl3dv_experiments.md` 验证边界段 |
| R14 | pose-free Register 覆盖真机+纯视觉生成、RELIC 受限于有 pose 场景——这一定位是否成立？ | Open | 设计层成立（RELIC 绝对 pose 注入 Q/K，无 pose 不可用；NAV Register pose-free）。但"NAV 在无 pose 长程下不劣于 RELIC"需 revisit 对比实证。见 `essay/relatedworks/related_works.md` RELIC 段 |
| R15 | text 通道：Stage One 空 text 训练，Stage Three R2R 指令如何接入？ | Open | Stage One 用空 text，UMT5 cross-attention 未训练；R2R 需真实指令。候选：Stage One/二引入 caption 联合训练 / Stage Three 微调引入 / 冻结 backbone 用空 text。需决策+实验 |
| R16 | train-test action gap（训练 pose-derived 离散 vs 推理/demo 手录 0001.json）如何处理？ | Open | 训练 action 来自 pose 离散化（10 类、每 episode 自归一化、噪声大），推理用干净手录序列。需统一 action 来源或评估其对 action 跟随结论的污染 |
| R17 | 生成+Register→NavPolicy：Register 更新与推理频率如何对齐视频 chunk / Habitat step？ | Open（工作设想） | 2026-08-08：训练按接近 1 video chunk 的**空间尺度**更新 Register（观测常 ~4 frame）；推理快→可单步，慢→与 Register 同步。见 NAV-DES-003「暂定可行工作设想」、DEC-022。非完整 Insight 主体；待 packing/实验后升格或废弃 |

> 当前 `Open` 项：R1–R17（R1–R6 由 `Answered` 改为 `Open（仅设计/部分
> proxy）`，因按"实验证明才 Answered"规则，其仅 experiment.md 设计、无
> 代码/产物落地）。按规则，这些 insight/问题被采纳/实验验证/显式废弃
> 之前，不得开启下一轮全新的设计分析（对已有 insight 的细化与实验设计
> 不算"新分析"）。R13–R16 为 2026-08-07 讨论压缩登记；R17 为
> 2026-08-08 Stage Three 控制频率工作设想登记，尚未展开 Insight 主体。

## Insight 主体

以下 insight 均不动 Wan2.1 backbone，主要利用流式 3D 结构设计。每条按
动机 / 对应已调研工作 / 具体改法 / 预期收益 / 可验证实验指针 / 风险组织。

按所解决问题的性质分为五类。类别仅作归档索引，不改变各 Insight 内容与
编号；一条 Insight 跨多个 R 问题时，归入其主要类别，并在动机中标注关联 R。

| 类别 | 对应 R 问题 | 含 Insight |
| --- | --- | --- |
| A. 长程记忆结构与漂移 | R7 | Insight 1, 2 |
| B. 3D 几何监督与 readout 表示 | R8, R11, R5 | Insight 3, 4 |
| C. Action 与物理表示 | R9 | Insight 5 |
| D. 训练范式与 exposure bias | R3, R10 | Insight 6 |
| E. 导航接口与 agentic 扩展 | R12 | Insight 7 |

### 类别 A：长程记忆结构与漂移（对应 R7）

针对固定预算 Register 在超长 horizon 下的容量与漂移问题，从"更新机制"
与"全局锚点"两个角度改进 Register 内部结构，不碰 backbone。

#### Insight 1：Register 从"均匀递归更新"改为"几何知情 read-before-write"

- **动机**：当前 Updater 对所有 Register slot 做均匀递归更新，无空间
  结构，长 horizon 下关键信息被均匀覆盖（R7）。
- **对应已调研工作**：DeepVerse 的 geometry-aware memory read-and-write
  （按空间重叠/结构相似度检索相关历史再写入）；LingBot-Map 的
  anchor context + pose-reference window + trajectory memory 三段式
  空间记忆。
- **具体改法**：每个 Register slot 绑定一个轻量 state（pose / raymap /
  timestamp，外部参数，不进 DiT 权重）；Updater 先按"新 chunk 与各 slot
  的空间重叠/结构相似度"检索相关 slot，再只更新相关 slot，非相关 slot
  保留。这把 Register 从平坦 recurrent state 变成结构化空间记忆。
- **预期收益**：长 horizon 不再均匀覆盖，关键空间信息按需保留，结构性
  缓解 R7 容量问题，而非靠加大预算。
- **可验证实验指针**：在 E4.1 horizon scaling 上对比"均匀更新 vs
  read-before-write"；在 E4.3 信息保留检验上对比早期线索保留率。
- **风险**：检索机制引入额外计算与一个可学习检索头；需保证检索本身
  不成为新的 exposure bias 源。

#### Insight 2：引入 "sink Register" 作长程全局锚点

- **动机**：固定预算 Register 在数百 chunk 下仍会漂移（R7）。
- **对应已调研工作**：Rolling Forcing 的 attention sink（初始帧 KV 作全局
  锚点 + 动态 RoPE）；LongLive 的 frame sink（frame-level attention sink）。
- **具体改法**：在固定预算内划出少数 sink slot，由 Extractor 从首 chunk
  建立后**只读不更新**，Updater 只更新其余 slot。sink slot 作为全局上下文
  锚点。
- **预期收益**：成本几乎为零（Register 内部划分，不碰 backbone），给长程
  漂移一个稳定参考，结构性缓解 R7 退化。
- **可验证实验指针**：E4.1 horizon scaling 加"sink on/off"对照；长 episode
  的漂移度量（位姿累积误差）。
- **风险**：若首 chunk 不具代表性，sink 会固化偏置（与 R11 相关，需与
  Insight 3/4 的无参考帧设计协同）。

### 类别 B：3D 几何监督与 readout 表示（对应 R8、R11、R5）

针对 3D 监督的尺度/因果对齐问题与 readout 的选层/参考帧偏置问题，从
"监督表示与 teacher"和"readout 结构"两个角度改进，均不碰 backbone。

#### Insight 3：3D 监督改用尺度不变表示 + 流式 3D 前端作 teacher

- **动机**：度量 depth/pose 与 Wan VAE 压缩尺度不一致（已知跨数据集尺度
  归一化风险），且离线 VGGT 标签是非因果的，不反映"第 t chunk 到达时的
  流式状态"（R8、R11）。
- **对应已调研工作**：AETHER 的尺度不变归一化 disparity + 尺度不变 raymap
  对齐 DiT 时空框架；StreamVGGT 的因果流式 4D 重建；π³ 的无参考帧
  置换等变几何；GeometryForcing 对齐 VGGT 特征（NAV 用流式前端是其 streaming
  演进）。
- **具体改法**：(a) Stage Two 监督表示从度量 depth/pose 改为尺度不变
  disparity + raymap；(b) 用冻结的流式 3D 前端（StreamVGGT 或 π³）逐 chunk
  产几何伪标签，替代离线 VGGT，使监督信号因果且与流式状态对齐。
- **预期收益**：解决尺度不一致风险；监督与流式状态对齐提升 probe 质量；
  π³ 无参考帧顺带缓解 R11 首 chunk 偏置。
- **可验证实验指针**：E2.1 加"度量 vs 尺度不变"表示对照；probe 误差在
  流式 vs 离线 teacher 下的对比；首 chunk 不具代表性场景的 probe 稳定性。
- **风险**：流式 3D 前端本身的质量决定伪标签上限；需冻结前端避免引入
  额外训练自由度。

#### Insight 4：probe/readout 做成无参考帧 + 多层可学习融合

- **动机**：Round 5 选层是开放问题；单层手动选可能次优，且 probe 依赖
  固定首帧参考会传导首 chunk 偏置（R11）。
- **对应已调研工作**：π³ 的无参考帧置换等变 readout；多层特征融合的
  通用 readout 设计。
- **具体改法**：(a) readout 不依赖固定首帧参考（π³ 思想）；(b) 用一个轻量
  readout head 对多个 DiT 层的 Register-after-DiT 做注意力融合（backbone
  冻结，只学 head），让导航自己学组合，同时仍报逐层消融回答 R5。
- **预期收益**：把"选哪层"从手动选变为可学习，且不碰 backbone；无参考帧
  readout 缓解 R11。
- **可验证实验指针**：E5.1 逐层消融 + "多层融合"选项的导航 SR/SPL 对比；
  首 chunk 偏置场景的 readout 稳定性。
- **风险**：可学习融合头增加 Stage Three 参数，需防过拟合；可学习融合与
  "逐层报告"需同时保留以维持可解释性。

### 类别 C：Action 与物理表示（对应 R9）

针对 action 与几何监督不在同一表示空间、物理关系难以对齐的问题，把
action 几何化，使物理理解与几何理解在同一空间可读。

#### Insight 5：action 几何化为 raymap，与几何监督同空间

- **动机**：当前 Action 是 camera-motion pseudo-label，与几何监督在不同
  表示空间，物理关系（动作如何改变环境）难以与几何对齐（R9）。
- **对应已调研工作**：AETHER 把相机轨迹编码为尺度不变 raymap 作为几何
  知情动作空间，支持条件预测与规划。
- **具体改法**：把 action 编码为尺度不变 raymap 注入 Updater（外部条件，
  不进 DiT 权重），使 action 与几何监督共享同一表示空间。
- **预期收益**：强化 R1 的动作条件预测能力；让 Register 的物理理解
  （R1）与几何理解（R2）在同一空间可读，互相促进。
- **可验证实验指针**：E1.2 动作条件预测加"pseudo-label vs raymap"对照；
  导航 SR/SPL 在两种 action 表示下的对比。
- **风险**：raymap 化需重新对齐 Action 标注（NAV-DAT-006），可能影响
  已有数据流水线；需保证与现有 Action manifest 的可逆映射。

### 类别 D：训练范式与 exposure bias（对应 R3、R10）

针对训练用干净历史、推理用在线更新 Register 的 exposure bias，从训练侧
结构化缓解，推理仍只用真实观测。

#### Insight 6：Stage One 训练加 self-generated history 替换

- **动机**：训练用干净历史、推理用在线更新 Register，存在 exposure bias
  （R3 已验证可控，但训练侧可结构化进一步缓解）。
- **对应已调研工作**：Self-Forcing / Self-Forcing++ 的 self-generated
  rollout + backward noise initialization；Rolling Forcing 的非重叠窗口
  蒸馏。
- **具体改法**：Stage One 以一定概率把 Updater 输入的干净历史 latent
  替换为自生成 + 重新加噪的历史，让 Register 训练时就见过自身更新的
  退化分布。推理仍只用真实观测，回灌仅用于训练鲁棒性。
- **预期收益**：结构化对抗 exposure bias，提升 Register 在线迁移稳定性
  （E3.1 的训练侧补充）。
- **可验证实验指针**：E3.1 加"teacher forcing vs scheduled sampling"对照；
  E3.2 定位 scheduled sampling 是否必需。
- **风险**：自生成历史质量依赖当前生成能力，早期训练可能噪声过大；
  需设计概率退火。

### 类别 E：导航接口与 agentic 扩展（对应 R12）

针对 Stage Three agentic 导航中途切换子目标/指令时 Register 需支持任务
切换而不丢空间记忆的问题，给 Register 加 recache 机制。

#### Insight 7：Register-recache 支持 agentic 导航的指令切换

- **动机**：Stage Three 若接 Qwen-RobotNav 式两层 planner，中途会切换
  子目标/指令，Register 需支持任务切换而不丢空间记忆（R12）。
- **对应已调研工作**：LongLive 的 KV-recache（新 prompt 到来时刷新缓存）；
  Qwen-RobotNav 的可控观测协议与两层 planner。
- **具体改法**：给 Register 加 recache 机制——指令变化时刷新"任务相关"
  slot、保留"空间记忆"slot，对应 LongLive 的 KV-recache 思想落到 Register。
- **预期收益**：成本极小，让 NAV 支持 agentic 长程导航而不重训；把"可控
  观测协议"思想落到 Register 结构。
- **可验证实验指针**：构造中途切换子目标的 agentic 导航任务，对比
  recache on/off 的 SR/SPL 与空间记忆保留。
- **风险**：需明确"任务相关 vs 空间记忆"slot 的划分规则；过早引入增加
  Stage Three 复杂度，建议作为后期扩展。

## 优先级建议

按"杠杆/成本"排序（高→低）：

1. Insight 1 + 2（几何知情 read-before-write + sink Register）——最高
   杠杆，结构性解决长程容量与漂移，几乎零成本。
2. Insight 3（尺度不变表示 + 流式 3D teacher）——解决已知尺度风险与流式
   因果对齐，Stage Two 落地质量关键。
3. Insight 5（action raymap 化）——统一物理与几何表示空间，强化 R1。
4. Insight 6（self-generated history 训练）——结构化对抗 exposure bias。
5. Insight 4（多层可学习 readout + 无参考帧 probe）——把 R5 从手动选变为
   可学习，同时缓解 R11。
6. Insight 7（Register-recache）——面向 agentic 导航的可扩展接口，后期。

## 限制与未决问题

- R7–R12 为 `Open`，按规则需在采纳/实验验证/显式废弃后才能开新分析。
- Insight 1/2 改的是 Register 内部结构，需在 `01_design/` 增设 Register
  state 与更新机制的设计文档后才算正式方案。
- Insight 3 涉及与 NAV-DAT-006 Action 标注、NAV-DAT-005 数据构建的联动，
  采纳前需评估对现有流水线的影响。

## 相关文档

- 实验设计：`NAV/essay/experiment/experiment.md`
- 引言与核心问题：`NAV/essay/intro/intro.md`
- 相关工作：`NAV/essay/relatedworks/related_works.md`、`related_works_survey.md`
- 当前设计：`doc/01_design/v0_stage_one_register_world_model.md`、
  `v0_stage_two_3d_supervision.md`、`v0_stage_three_navigation.md`
- 决策记录：`doc/00_overview/decision_log.md`
- 调研参考：`doc/07_research/`、`doc/02_architecture/`
