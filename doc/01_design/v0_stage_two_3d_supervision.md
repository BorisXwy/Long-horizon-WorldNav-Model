# V0 Stage Two：视频生成模型中的 3D 监督设计

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DES-002` |
| 类型 | 设计规范（Design Specification） |
| 状态 | Proposed / Confirmed Semantics / Not Implemented |
| 更新时间 | 2026-08-09 |
| 职责 | 定义在完整视频生成前向中加入 3D Probe 与监督的网络设计 |

## 结论

Stage Two 完全处于 Stage One 的视频生成范式中。输入、Register 更新、
Local Memory、Action/Text condition、noisy target、DiT 前向和 Diffusion/RFlow
生成损失均保留；唯一新增内容是：

1. 对不同 DiT 层执行 3D diagnostic probe，确定最适合承载 3D 表示的层；
2. 在选定层上增加 3D Teacher/GT 监督信号；
3. 让 Register 先参与 DiT 视频生成前向，再从选定层取出其 contextualized
   representation 接受 3D 监督。

Stage Two 不训练 Navigation Policy，也不把 hidden/Register 接入导航任务；
这属于 Stage Three。

Stage Two 的 3D 监督遵循 action-centered 因果原则：未来视觉（Future Visual）
和未来 3D（Future 3D）只作为动作、Instruction/Goal 与真实观测条件下的
后果监督（Consequence Supervision），不能作为 action/navigation token 的输入
条件。换言之，Stage Two 可以让 future visual/3D tokens 读取 action/register/
current observation 来学习“执行该动作后世界如何变化”，但不得让 action tokens
读取 future visual/3D tokens 来预测动作。

## 阶段边界

Stage One 解决：

- Chunk-by-chunk 视频生成；
- 固定预算 history memory；
- Local Memory 保持短期连续性；
- Register 替代 Infinite-World HPMC history；
- Diffusion/RFlow 生成目标；
- 第一个 History Chunk 由 Extractor 产生 Register；
- 后续 History Chunk 由 Updater 递归更新 Register。

Stage Two 新增：

- 在原视频生成 forward 中缓存候选 DiT block hidden；
- Probe 不同层中经过 DiT contextualization 的 Register；
- 用 Frozen VGGT/VGGT-Ω 或真实 Pose/Depth/Pseudo-geometry 提供 3D 信号；
- 将 3D loss 与原 Diffusion/RFlow loss 联合训练；
- 得到 Stage Three 可读取的 `Register-after-DiT @ selected layer`。

Stage Two 当前不改变：

- Wan2.1-1.3B DiT 主干；
- Wan VAE；
- Infinite-World ActionEncoder；
- 最新一段 Local Memory；
- A/B 的 Register 注入位置；
- Stage One 已训练 checkpoint 的继承关系。

Stage Two 明确不包含：

- Navigation Policy；
- 导航 action head 或 VLN objective；
- 将生成视频作为导航 history；
- 为导航另建一套独立 perception backbone。

## 网络总体结构

```text
真实 History RGB ─► Wan VAE ─► Extractor/Updater ─► Register R_i
                                                    │
Noisy Target + Local + Action/Text + Register ──────┤
                                                    ▼
                                         Wan DiT 视频生成前向
                                                    │
                    ┌───────────────────────────────┼───────────┐
                    ▼                               ▼           ▼
             Diffusion/RFlow loss       候选层 Register hidden  生成输出
                                             │
                                             ▼
                                      3D Probe / 3D Head
                                             ▲
真实 RGB / Pose / Depth ─► Frozen Teacher ───┘
```

## 因果 Mask 与信息流

Stage Two 推荐采用与 GigaWorld-Policy 类似的 action-centered 信息流：

```text
Action / Nav Query tokens can attend to:
  - 真实 History Register
  - 当前 Observation / Local Memory
  - Instruction / Goal / Text
  - Robot/Nav state（若有）

Action / Nav Query tokens must NOT attend to:
  - Future Visual tokens
  - Future 3D tokens
  - Target future GT hidden

Future Visual / Future 3D tokens can attend to:
  - 真实 History Register
  - 当前 Observation / Local Memory
  - Instruction / Goal / Text
  - Action / Nav Query tokens
```

该 mask 的目的不是削弱 video/world modeling，而是明确因果方向：

\[
(\mathrm{history},\mathrm{obs},\mathrm{text})\rightarrow
\mathrm{action}\rightarrow
(\mathrm{future\ visual},\mathrm{future\ 3D})
\]

因此 future branch 是训练时 dense supervision 和 physical consistency regularizer；
它不构成 Stage Three policy 的必要输入。

## 真实 History 的在线状态

第一个真实观测 Chunk 不再依赖具有场景内容的可学习初始 Register。采用：

\[
H_0=E_{\mathrm{DiT}}(Z_0),\qquad
R_1=\operatorname{Extractor}(H_0)
\]

后续每个真实 Chunk：

\[
H_i=E_{\mathrm{DiT}}(Z_i),\qquad
R_{i+1}=\operatorname{Updater}(R_i,H_i)
\]

其中：

- `Extractor` 从第一个真实 Chunk 建立场景状态；
- `Updater` 负责将后续真实 Chunk 写入固定预算 Register；
- checkpoint 保存 Extractor、Updater 和 DiT 的能力参数；
- checkpoint 不应保存某个 episode 的 Register 状态；
- 新 episode 的 \(R\) 始终由第一段真实观测重新提取。

如果实现需要零张量或常量 query 触发 Extractor，该 seed 必须无场景信息且不可
训练；它不是待记忆的 `initial_registers`。

## VAE Latent 与 3D 可用性

Wan VAE 将 RGB 时空块压缩为规则 latent 网格。以81帧 Chunk 为例：

\[
[B,3,81,H_{\mathrm{rgb}},W_{\mathrm{rgb}}]
\rightarrow
[B,16,21,H_{\mathrm{rgb}}/8,W_{\mathrm{rgb}}/8]
\]

DiT 使用 `(1,2,2)` patch embedding 后仍保留三维 token 地址：

\[
H^l\in
\mathbb{R}^{B\times21\times H_{\mathrm{rgb}}/16
\times W_{\mathrm{rgb}}/16\times C}
\]

Attention 会混合 token 内容，但不会消除其时间、空间索引。因而 VAE/DiT
hidden 仍可学习相机运动、粗深度、跨帧对应、场景布局和导航可达空间。

VAE 压缩会损失高频边界与小物体细节，因此 Teacher target 不能直接按原始 RGB
分辨率逐点硬对齐；应在时间和空间上聚合到 student grid，或采用 feature
distillation。

## Probe 对象：经过 DiT 的 Register

Stage Two 的主要候选表示不是 Updater 刚输出、尚未参与生成前向的原始
\(R_i\)，而是它注入 DiT 后在第 \(l\) 层形成的表示：

\[
\widetilde R_i^l
=
\operatorname{SelectRegister}
\left(
\operatorname{DiT}^{1:l}
(X_{\mathrm{target},t},L_i,R_i,A_i,\mathrm{text})
\right)
\]

它已经在视频生成上下文中与 noisy target、Local Memory、Action/Text 发生
交互，更符合“视频 DiT 内部形成 3D 表示”的研究假设。

对 A，需要从 latent token sequence 中严格切出 Register prefix 对应位置；对
B，需要从 condition/context 流中读取经过相应 DiT block 更新后的 Register。
如果现有 B cross-attention 只读取静态 condition、没有产生逐层更新后的
condition hidden，则需增加可读回的 Register state 或双向更新接口，不能把
未经 DiT 更新的 condition embedding 称为 `Register-after-DiT`。

## 辅助分析：Spatial Hidden 与 Register

### Spatial Hidden：局部与稠密几何

从若干 DiT block 读取真实 history 对应的 hidden state，例如浅、中、深层各一
层。恢复为三维 token grid 后，通过轻量 projector 或 DPT-style head 监督：

- VGGT patch geometry feature；
- Depth / Disparity；
- Pointmap；
- Correspondence；
- Local relative pose / motion；
- Confidence。

Spatial Hidden 用于辅助验证 3D 信息是否存在于生成 token，并帮助判断
Register probe 的信息来源；Stage Three 的默认导航接口仍以选定层的
`Register-after-DiT` 为主。

### Register：Stage Two 的主监督与 Stage Three 接口

Register 监督：

- 当前 Chunk 相对历史坐标系的 Pose；
- 累计 Camera Trajectory；
- Global Scene Layout / Occupancy；
- 回访区域的一致性；
- Topological State；
- Navigation State。

Register 是固定预算 memory bottleneck，不要求单独恢复全分辨率 depth。

### A/B 的职责差异

| 方案 | Register结构 | 更适合承载 | 限制 |
| --- | --- | --- | --- |
| A | `[B,16,4,H,W]` latent-like spatial state | coarse depth、layout、spatial occupancy、history prefix | 必须严格区分 Register/Local/Target token |
| B | 16个256维 global tokens | trajectory、topology、scene/navigation state | 不应独自承担 dense geometry |

B 的 Register 是否真正被 DiT 更新必须由代码和梯度检查确认；不能只读取送入
cross-attention 前的静态 condition token。

## VGGT Teacher 的读取方式

VGGT/VGGT-Ω 同样从已经充分混合的 Transformer hidden 中读取 3D：

- Camera Head 从 Camera/Register Tokens 经过额外 Self-Attention 预测 Pose；
- Dense Head 从多层 Patch Tokens 恢复空间网格并融合预测 Depth；
- 关键不是 hidden channel 可解释，而是 token 类型与位置仍可寻址。

NAV 对应采用：

```text
History DiT multi-layer spatial tokens → Dense 3D projector/head
Updated Register                       → Pose/map/navigation head
```

VGGT-Ω 代码解析见
`../02_architecture/vggt_omega_register_architecture.md`。

## 时间与空间对齐

81个 RGB frame 对应21个 VAE latent time steps。Teacher 不能简单地将第 \(k\)
个 latent token 等同于某一张单独 RGB frame。应：

1. 对同一真实 history RGB 运行 Frozen Teacher；
2. 将 Teacher frame/patch feature 分到21个时间 bin；
3. 在每个 bin 内按有效帧和置信度聚合；
4. 将空间网格下采样或投影到 Student token grid；
5. 对齐后计算 feature/depth/pointmap loss。

Pose/trajectory 可以保留逐帧标签，但输出 head 应明确预测21-step latent
时间点、81-frame 原帧，或 Chunk 级相对变换，禁止混用。

## 与 Stage Three 的接口

Stage Two 最终必须冻结以下接口定义：

```text
selected_block_ids
Register token slicing rule（A/B分别定义）
Register-after-DiT tensor shape
normalization / projector
对应3D probe指标
```

Stage Three 复用 Stage Two 选出的 DiT 层及 Register 读取规则，将该表示输入
导航任务。Stage Two 本身不优化导航 loss。

## 实施顺序

1. 完成 Stage One 的 Extractor-first、Updater-later 改造；
2. 冻结 Stage One backbone，跨 DiT blocks 做 Register diagnostic probe；
3. 确定最有几何信息的层、A/B token slicing 和 readout；
4. 在完整视频生成 forward 上加入选定层的 3D supervision；
5. 联合微调 DiT、Extractor、Updater 与 3D head，同时保留 diffusion loss；
6. 将选定的 `Register-after-DiT` 接口交给 Stage Three。

实现前的训练细则以
`../04_training/v0_stage_two_3d_supervision_training.md` 为唯一事实来源。
