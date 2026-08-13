# V0 NAV 项目设计：InfiniteWorld Register 世界模型

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DES-001` |
| 类型 | 设计规范（Design Specification） |
| 状态 | Active / Implemented Prototype / Action-centered Principle Added |
| 更新时间 | 2026-08-09 |
| 职责 | 定义 Stage One 的目标、边界和 Register Memory 总体方案 |

本文档定义 NAV 当前模型方向。A/B Register 原型、RE10K 1000-step 全参数训练、
VBench 对照和 SpatialVID Short 数据流水线已经实现；当前工作重点是从既有
RE10K checkpoint 继续2–3 chunk teacher-forced history 训练。

## Stage One：基于 Register Token 的流式世界模型

### 全阶段共享骨干原则（Shared Backbone）

NAV 的三阶段不应被实现为三套彼此独立的感知网络。Stage One 的视频生成
DiT/Register 结构、Stage Two 的 3D 监督和 Stage Three 的导航策略，应共享同一
套核心时空表征骨干（Shared Spatiotemporal Backbone）：

```text
Stage One:  VideoGen / World Modeling 训练 Register 与 DiT 表征
Stage Two:  在同一视频生成前向中加入 3D supervision，选择 3D-aware layer
Stage Three:  读取该 shared backbone 的 Register-after-DiT / compact tokens 训练 Policy
```

因此 Stage One 不是只为生成视频服务的孤立模块，而是后续 3D 和导航能力的
表征预训练阶段。Stage Three 可以使用更小的 policy-only input interface，但
不应绕开 Stage One/Two 学到的 backbone 另建一套 perception backbone。

### Action-centered 因果原则

NAV 采用 action-centered / navigation-centered 因果方向：

```text
真实历史 Observation + 当前 Observation + Instruction/Goal
        │
        ▼
Action / Navigation Decision
        │
        ▼
Future Visual / Future 3D Consequence
```

未来视觉或未来 3D 是动作和目标条件下的结果（Consequence），不是动作预测的
输入条件（Policy Condition）。没有 action 和 text/goal，一个当前状态通常对应
无数可能未来；因此 future visual tokens 不应被允许反向影响 action tokens。

训练中可以保留 future visual / future 3D 预测作为 auxiliary supervision，
用于约束 action/register/backbone 表征与物理后果一致；推理导航时则可以删除
大规模 future tokens，只保留真实观测、Instruction、Register/Memory 与
action/query tokens。

### 阶段目标

Stage One 的目标是将现有视频世界模型改造为流式生成模型（Streaming
Generation Model），使模型能够持续接收动作或导航条件，并以分块
（Chunk-by-chunk）的方式连续生成视频。

在长时间流式生成过程中，模型不能无限保留和直接拼接全部历史视频，也不采用
持续增长的键值缓存（Key-Value Cache，KV Cache）保存过去信息。已经生成的
历史内容将被压缩为一组具有固定或受控规模的寄存器令牌（Register Tokens），
其设计思路参考 VGGT-Omega。

### Infinite-World 对 Wan2.1 的改造

Infinite-World 保留 Wan2.1-T2V-1.3B 的 DiT 生成骨干、UMT5 文本编码器和
Wan VAE，在此基础上加入动作编码器（Action Encoder），将平移与视角动作嵌入
加到待生成帧特征中；同时加入分层无位姿记忆压缩器（Hierarchical Pose-free
Memory Compressor，HPMC），把历史 VAE Latents 经局部/全局两级时序压缩为
固定预算，再将“压缩历史 + 最近一帧 + 当前噪声 Latents”连同区分条件与目标的
掩码一起送入原 DiT。论文与代码均表明，核心改造是为 Wan 增加动作条件和
定长历史记忆，而非替换其视频生成主干。

### 核心思路

每次流式生成由三类信息共同决定：

1. 当前可见上下文（Current Visual Context）：最近生成的视频帧或视频潜变量，
   用于保持局部时间连续性。
2. 历史寄存器令牌（History Register Tokens）：对更早历史内容的压缩表示，
   用于保存场景结构、已观察物体、空间关系和长期状态。
3. 当前控制条件（Current Control Condition）：当前分块对应的动作、相机控制、
   导航指令或其他条件。

概念上的生成流程为：

```text
初始图像 / 初始视频
        │
        ▼
当前视觉上下文 + 动作条件 + Register Tokens
        │
        ▼
生成下一个视频 Chunk
        │
        ├──► 保留最近帧，作为下一轮当前视觉上下文
        │
        └──► 更新 Register Tokens，写入新的历史信息
                        │
                        └──► 进入下一轮流式生成
```

随着生成过程不断推进，原始历史帧可以退出当前上下文窗口，但其中对后续生成
有用的信息应被写入 Register Tokens。模型由此在近程视频上下文长度受限的情况
下，仍能访问长期历史。

### 历史信息的表示方式

本阶段明确不采用以下两种方式作为长期历史的主要表示：

- 不直接拼接全部历史帧或历史潜变量。该方式的序列长度和计算量会随生成时长
  持续增长。
- 不将完整历史保存为不断增长的 KV Cache。KV Cache 更接近对历史计算结果的
  逐项缓存，存储规模仍会随序列长度增加，而且不一定形成适合世界模型长期
  记忆的结构化摘要。

长期历史改为 Register Tokens：

- Register Tokens 是模型内部可读写的记忆令牌；
- 令牌数量应保持固定，或仅在严格预算内变化；
- 每生成一个新 Chunk，模型根据当前内容更新这些令牌；
- 后续 Chunk 通过注意力机制读取 Register Tokens，而不再读取全部原始历史；
- Register Tokens 应尽量保留跨视角稳定的场景和状态信息，而不是简单复刻
  最近若干帧。

这里的“类似 VGGT-Omega”主要指使用持续更新的 Register Token 作为跨时间
信息载体。NAV 是否直接复用其具体网络模块、令牌更新规则或训练损失，尚未
确定。

需要特别说明：VGGT-Ω 原始实现不是在线流式记忆。它为每帧设置 16 个
Registers，在部分层中限制跨帧信息只能通过 Camera/Register Tokens 交换，但
仍一次性双向处理完整输入序列，且每次前向都重新初始化 Registers。NAV 借鉴的
是其“可读写信息瓶颈”思想，后续必须另行设计跨 Chunk 持久化、因果更新和固定
总预算机制。详细代码解析见 `NAV/doc/02_architecture/vggt_omega_register_architecture.md`。

### 预期能力

完成 Stage One 后，模型应具备以下基本能力：

- 以固定长度 Chunk 持续生成视频；
- 在有限的当前上下文窗口下运行；
- 使用 Register Tokens 继承早期生成历史；
- 控制长期推理的显存和计算量，不随总生成长度线性无限增长；
- 在视角离开后重新访问旧区域时，尽可能恢复一致的场景内容；
- 为后续导航条件、几何监督和闭环记忆实验提供流式骨干网络。

### 初步模块划分

后续实现可以按以下逻辑模块拆分：

| 模块 | 主要职责 |
| --- | --- |
| 流式生成器（Streaming Generator） | 根据当前上下文、控制条件和记忆生成下一个 Chunk |
| 当前上下文缓冲区（Context Buffer） | 保存最近的视频帧或潜变量 |
| 历史编码器（History Encoder） | 从新生成内容中提取需要写入长期记忆的信息 |
| Register Token 记忆 | 保存固定预算的长期历史表示 |
| 记忆更新器（Memory Updater） | 决定 Register Tokens 的保留、融合、覆盖与更新 |
| 记忆读取接口（Memory Reader） | 将 Register Tokens 注入视频生成骨干网络 |

上述模块仅用于明确职责边界，不代表最终代码类名或唯一实现方案。

### 当前已实现的 A/B

Infinite-World 的 HPMC history 已被移除，但保留最新 local memory 和原有
ActionEncoder、DiT、文本条件及 RFlow loss。

| 方案 | Register | History读取 | 注入位置 |
| --- | --- | --- | --- |
| A | `[B,16,4,H,W]` latent-like state | 每个空间位置以4个Register读取history时间token | `[Register; local; noisy target]` |
| B | 16个256维token | Register query读取池化history token | 与UMT5 text拼接为DiT condition |

两者都在每个 chunk 后递归更新固定预算 Register，不保存无限历史。详细张量和
训练样本语义见 `../04_training/v0_streaming_training_sample_semantics.md`。

### Register 初始化与更新语义

Stage One 的目标结构确定为“先提取、后更新”：

\[
R_1=\operatorname{Extractor}(H_0),\qquad
R_{i+1}=\operatorname{Updater}(R_i,H_i)
\]

即新 episode 的场景状态必须来自第一段 History Chunk，而不是来自 checkpoint
中保存的场景内容：

- 第一个 Chunk：`Extractor(H0) → R1`；
- 后续 Chunk：`Updater(Ri, Hi) → Ri+1`；
- checkpoint 保存 Extractor/Updater 的能力参数；
- checkpoint 不保存 episode-specific Register；
- 不使用携带场景先验的 learnable initial Register。

当前已实现 A/B 原型与既有 checkpoint 仍从 learnable initial Register 开始，
因此需要在 Stage One 代码中完成此项改造后，才进入 Stage Two。该状态差异不得
隐藏或描述为已经实现。

### 后续需要确定的问题

具体改造开始前，需要进一步确定：

- 选择哪个视频或世界模型作为 Stage One 的初始骨干网络（Backbone）；
- Register Tokens 位于像素空间、VAE 潜空间还是 DiT 特征空间；
- Register Tokens 的数量、维度和初始化方式；
- 记忆是每层独立、跨层共享，还是仅注入部分 Transformer 层；
- 采用交叉注意力（Cross-Attention）、令牌拼接或其他方式读取记忆；
- 每个 Chunk 后如何更新记忆，以及是否需要门控、淘汰或分层机制；
- 如何避免 Register Tokens 随时间发生信息覆盖、漂移或退化；
- 训练时采用多长的展开步数（Unroll Steps）；
- 是否冻结原始生成骨干网络，分阶段训练记忆模块；
- 如何构造包含离开、回访和闭环的训练数据；
- 如何分别评价短期画质、动作响应、长期一致性和回访恢复能力。

### 当前阶段边界

Short 阶段固定使用 Infinite-World/Wan2.1-1.3B、81帧 chunk、teacher forcing
和 last-target loss。当前不在训练中执行完整历史 diffusion rollout，也不把
Kinetics 的 episode-level VGGT估计伪装成逐帧 Action。

进入 Medium/Long 前必须补充：

1. scene-level train/validation split；
2. dataset-balanced sampler；
3. 多 target horizon 或 truncated BPTT 的成本实验；
4. scheduled sampling / rollout history 的 exposure-bias 实验；
5. 无记忆、HPMC、A、B 的公平长程对照；
6. Action方向、尺度与置信度的跨数据集审计。
