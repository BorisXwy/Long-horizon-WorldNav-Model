# NAV 科研 Idea 设计：空间可读的流式 World Action Memory

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-022` |
| 类型 | 研究调研、核心问题与科研 Idea 设计 |
| 状态 | Live / Research Concept |
| 更新时间 | 2026-10-22 |
| 职责 | 从相关工作和已有尝试中抽象科研问题，定义 NAV 要研究的核心机制与可证伪命题；不记录具体训练排期、资源调度或运行状态。 |

## 1. 研究调研情况

### 1.1 长程视频 World Model：解决了视觉连续性，尚未解决 embodied memory

[Infinite-World](https://arxiv.org/abs/2602.02393) 的核心贡献是 Hierarchical Pose-free Memory Compressor（HPMC）：把不断增长的历史 latent 递归压缩成固定预算表示，再用于 1000-frame 级别的交互式视频生成。它说明长程一致性可以通过模型内的递归 memory 获得，而不必依赖显式、完美的 pose。与此同时，模型仍然围绕 video generation 组织：历史的价值由生成质量和 loop closure 体现，memory 是否能被独立任务读取、是否具有可解释空间结构，并不是它的主要问题。

[LingBot-World](https://arxiv.org/abs/2601.20540) 以及后续 [LingBot-World 2.0](https://arxiv.org/abs/2607.07534) 把 video generation backbone 推向更长时域和更低延迟，强调 causal interaction、长时间上下文和实时生成。它们给出的方向是“用视频模型承载世界状态”，但其主要输出仍是未来视频。对 NAV 来说，最重要的启发是 chunk 化、因果 rollout 和固定计算预算；需要进一步回答的是：这个世界状态能否直接成为导航 policy 的 observation memory。

由这些工作可以抽象出一个空缺：现有 long-horizon video world model 有 memory，但 memory 的任务语义主要由生成损失决定；它是否包含可供导航使用的相机位姿、场景布局和可行动性，缺少独立约束。

### 1.2 Geometry-aware Transformer：说明 Register 可以承载场景级信息

[VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html) 使用 image tokens、camera token 和 register tokens，通过 frame-wise/global alternating attention，从多帧直接预测 camera、depth、point map 和 track。它的关键意义不是某个具体 head，而是证明了一个大型视觉 Transformer 的中间 token 可以直接成为多任务 3D 表征。

[VGGT-Ω](https://vggt-omega.github.io/) 进一步使用 registers 聚合 scene information，并通过 register attention 限制跨帧信息交换。官方说明中，register 让模型在更低显存下使用更多监督和无标注视频；同时，learned representation 可以帮助下游 VLA 与语言对齐。这个结果直接启发 NAV：历史压缩槽位不应只被当作生成条件，也应当被设计成可以被 geometry probe 读取的 scene state。

但 VGGT/ VGGT-Ω 与 NAV 的任务边界不同。它们从一段同时可见的多视图输入恢复几何，而 NAV 需要在在线流中逐步更新状态；它们的 register 是 scene reconstruction 的中间表示，而 NAV 的 register 还必须记录 action-conditioned temporal state。因此，直接把 VGGT register 接到 Wan 上不能自动得到导航 memory，必须研究 recurrent update、动作条件和当前观测边界。

### 1.3 World Action Model：证明 video backbone 可以服务 policy，但 inference interface 需要重新设计

[DreamZero](https://arxiv.org/abs/2602.15922) 将预训练 video diffusion backbone 改造成 World Action Model，同时预测未来视频和连续动作。它的核心论点是视频是 dense physical supervision，动作预测可以借助未来世界状态来学习，并通过系统优化实现实时闭环控制。

[GigaWorld-Policy](https://arxiv.org/abs/2603.17240) 更明确地把 action-centered WAM 写成两个耦合目标：当前 observation 和动作预测 future video，policy 侧可以在 inference 时不执行 future video generation。[GigaWorld-Policy-0.5](https://arxiv.org/abs/2607.13960) 又进一步使用 Mixture-of-Transformers，减少 action-only inference 的 active computation，并报告了较低的 action latency。

这些工作共同给出一个重要原则：future visual prediction 是训练信号，不应成为 policy 推理的必要输入；action token 需要一条不会读取 future visual token 的因果路径。它们也暴露出一个设计张力：视频生成的主 latent 空间很大，而 policy 输出只需要少量 action token。若没有明确的 token boundary 和 mask，视频质量、动作预测和推理速度会互相牵制。

### 1.4 Streaming VLN 与 memory-augmented VLA：证明在线接口必须区分 history、current observation 和 instruction

[StreamVLN](https://streamvln.github.io/) 用 slow-fast context modeling 处理连续视频：fast stream 使用 sliding-window KV cache，slow stream 用 token pruning 更新长期上下文；其数据同时包含 R2R、R2R-EnvDrop、RxR、ScaleVLN 和 DAgger。它证明了在线 VLN 的核心不是一次性读完整视频，而是持续维护有限上下文并用闭环反馈修正动作。

[BridgeVLA++](https://arxiv.org/abs/2608.05042) 把 spatio-temporal memory 加入 VLA，强调历史观测和空间上下文对 memory-dependent manipulation 的作用。[StreamingVLM](https://research.nvidia.com/labs/eai/publication/streamingvlm/) 则从通用视频流角度讨论了长序列下的延迟与内存增长。

这些工作共同说明，导航接口至少要区分三件事：已经压缩的历史状态、当前高分辨率 observation、以及 instruction/goal。若把三者全部拼成同一种 token，模型可以训练，但无法判断它到底从哪里获得长期空间信息，也很难在 policy-only inference 时缩减计算。

### 1.5 Geometry-enhanced video generation：3D supervision 应进入生成表征本身

[FANTASYWORLD](https://openreview.net/pdf/b5f99909effe3a0756c724559123fbc9b7e5e448.pdf) 在冻结 video foundation model 的基础上加入可训练 geometric branch，联合建模 video latent 和 implicit 3D field，并用 camera trajectory 约束视角一致性。它的启发是，空间监督可以和视频生成在同一 forward 中共同训练，而不必另建完全独立的 3D 网络。

对 NAV 来说，关键区别在监督位置：我们要监督的是 current observation/history 形成的 hidden/Register，而不是把生成出来的 future video 当作几何真值。这样才能确保 Stage3 policy 使用的是真实 observation memory，而不是模型自己的预测结果。

### 1.6 调研后的共同空缺

把上述工作放在一起，可以得到一个清晰的研究空缺：

```text
long-horizon video world model 解决“如何保持未来视频连续”
geometry transformer 解决“如何从视觉 token 读出空间结构”
world-action model 解决“如何用视频训练动作并在推理时只输出动作”
streaming VLN 解决“如何在在线输入下控制上下文预算”

尚未被统一解决的问题：
一个固定预算、可递归更新、带空间监督的 memory，能否同时作为
video world model 的历史条件和 navigation policy 的长期 observation state？
```

## 2. 现有问题

### 2.1 研究问题：我们当前证明了什么

已有工程结果提供了几个可靠的事实，但每个事实都有边界：

| 已有事实 | 可以支持的判断 | 不能支持的判断 |
| --- | --- | --- |
| InfiniteWorld 官方权重、VBench 和长程协议已复现【已验证→`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`】 | HPMC/视频 world model 的生成基线可重跑 | NAV Register 已经优于 HPMC |
| 旧 IWAligned/legacy20 生成支路可以下降，短程 pose proxy 也能下降【已验证→`doc/01_model/model_evolution_and_current_architecture.md`】 | Wan 生成支路和附加 pose path 可以在同一工程链路中运行 | 完整 V1 token/mask/Register/3D 组合已经收敛 |
| GigaNav Wan2.1-1.3B policy-only/cotrain action loss 可以下降【已验证→`doc/01_model/giga_nav_ablation_wan21_13b.md`】 | Wan backbone 和小 action interface 具备动作拟合能力 | GigaNav 已具备长历史 Register 或空间理解 |
| StreamVLN 官方 R2R 子集闭环复现已跑通【已验证→`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`】 | VLN 闭环评测入口、环境和 action 执行协议可复现 | NAV 的 action 预测已达到同等效果 |
| T4 video/R2R latent cache 已构建【已验证→`doc/02_data/data_preparation_schema_and_status.md`】 | 原始视频、pose、simulator action 可以统一到 chunk 数据接口 | 所有 full manifest 已全部编码成 latent |

因此当前核心问题不是“有没有一个可运行的模型”，而是尚未把三条证据链闭合：

```text
Register 是否真的保留长历史？
Register/hidden 是否真的包含可读 3D？
同一表征是否真的帮助 policy，而不是只靠 action label 的分布拟合？
```

### 2.2 表征问题：Register 到底应该记什么

如果 Register 只追求下一 chunk 的 video loss，它可能只记住对生成最有利的纹理或运动统计，未必记住可导航的场景结构；如果强行把 pose 作为输入或 condition，模型可能直接绕过视觉表征，得到看似很低的 pose loss。因此需要区分三种信息：

- **历史视觉状态**：过去已经观察到的场景、物体和布局。
- **历史动作状态**：过去执行了什么，视觉变化由什么动作造成。
- **当前局部证据**：当前 observation 中仍然可直接读出的高分辨率空间信息。

当前 V1 的设计把历史视觉和 `A_hist` 写入 Register，把 `Z_obs` 保留为单独 current input；这个边界符合 VGGT-Ω 的 scene register 启发，也符合 StreamVLN 的 slow/fast context 区分，但 Register 是否具有真正的 spatial persistence 仍需通过 history shuffle、zero Register、pose probe 和长程 revisit 实验验证【未验证】。

### 2.3 训练问题：视频 loss 下降不等于 world representation 学会了

过去多次实验中，visual loss 可以下降，而 pose loss 或 policy loss 震荡；或者 pose loss 下降，但生成质量几乎不变。这说明单独看总 loss 无法回答表征是否有效，至少存在四个混淆因素：

1. probe 可能从不合适的 layer 读取 hidden，导致 3D 信号没有进入有效表征。
2. pose target 的 frame/chunk 对齐、坐标系和 mask 可能有误。
3. video dataset 的 pose-derived action 和 VLN expert action 不是同一种动作分布，混合时不能把它们当作同一个 policy label。
4. Stage3 的 majority action、STOP padding 或窗口采样可能让 CE 下降，但没有学到状态到动作的关系。

所以真正的验收单位必须是“固定协议下的对照组”：visual-only vs visual+3D、local-only vs Register、teacher-forcing vs autoregressive、open-loop vs closed-loop，以及 action 模型相对 majority baseline 的提升。

### 2.4 计算问题：video generation 和 policy 的 token 预算不一致

视频生成的 noisy future latent 具有空间和时间维度，policy action 只需要少量离散或连续动作 token；如果让两者无条件双向注意，action 会读取 future visual target，训练指标会虚高；如果完全拆成两个网络，又失去 video dynamics 对 action 的 dense supervision。

因此真正要研究的是共享 backbone 的信息边界，而不是简单把两个 head 拼在最后。`A_noise`、`Z_future` 和 `A_cur` 必须通过 mask 分隔：policy 读取 history/current/text，video branch 读取 history/current/action condition，future visual 不能反向改变 action prediction。这个 mask 约束已有代码实现，但其对最终性能的必要性仍未通过系统 leakage ablation 验证【未验证】。

### 2.5 数据问题：可用数据覆盖了不同任务，但没有天然统一的 action 语义

RE10K、DL3DV、SpatialVID 和 Argoverse2 提供视频、相机 pose 或轨迹；R2R/RxR 提供 simulator RGB、instruction 和 expert action。视频数据适合训练 action-conditioned video generation，VLN 数据适合训练 policy label；二者都可以参与 Register history，但 action 的统计分布、坐标系和噪声完全不同。

当前数据 cache 和 manifest 已能统一读取，但正式研究仍需解决：如何控制 SpatialVID 的数量优势、如何对齐不同数据集的 action scale、如何保证 R2R STOP padding 不制造多数类假信号、如何让无 text 视频使用 empty embedding 而不把空文本变成 shortcut。数据规则已经定义，跨数据集训练效果还没有闭合验证【未验证】。

### 2.6 最需要回答的三个可证伪问题

1. **Memory test**：固定 Register token 预算下，随着 history 从 IW1 增加到 IW16，模型是否比 latest-local、zero Register 和 shuffled history 更好？
2. **Spatial test**：加入 current-chunk pose supervision 后，从 Register/current hidden 读取的 pose 是否稳定改善，同时 visual loss 不退化？
3. **Embodied test**：关闭 future video generation 后，policy 是否仍能利用 Register 和当前 observation 在 R2R/RxR closed-loop 中改善 SR/SPL？

如果这三个问题不能分别得到正面结果，论文只能声称“共享 Wan backbone 的工程组合”，不能声称形成了新的 spatial streaming world model。

## 3. 我们要研究的核心 Idea

### 3.1 一句话定义

我们要研究的是：**让一个 video diffusion backbone 学习一个由视觉历史和已执行动作递归更新的固定预算 Register，并用 3D supervision 约束这个 Register/current hidden 成为可读的 spatial state；训练时用 future video consequence 提供 dense dynamics signal，推理时只保留 Register、current observation、instruction 和 action readout 完成闭环导航。**

它不是把 video generator 直接当 policy，也不是给 policy 外接一个独立地图模块，而是研究一个共享 hidden 是否能同时满足三种用途：

```text
历史观测 + 历史动作  -> Register state
Register + current obs + A_cur -> future video consequence
Register + current obs + text -> action prediction
current hidden/Register -> pose / spatial probe
```

### 3.2 核心机制一：Register 是在线状态，而不是可学习的场景常量

每个 episode 从固定、不可训练的 `R_null` 开始。对第 i 个历史 chunk，RegisterCell 接收 visual tokens 和 `A_hist`，输出下一个 Register：


\[
R_i = \operatorname{RegisterCell}_{\theta}(R_{i-1},\; \operatorname{visual}(C_i),\; A^{hist}_i).
\]

当前 observation `C_{t-1}` 保留为 `Z_obs`，不在写入后再次作为 policy 的原始 history token；预测目标为 `C_t`。这样可以把“长期场景状态”和“当前局部证据”分开：Register 负责持续性，Z_obs 负责细节。

这个机制的科研问题是 Register 是否学习到跨时间可复用的状态，而不是简单的压缩缓存。证明它需要 history shuffle、Register reset、不同 history span 和 revisit/pose probe，而不能只看下一 chunk 的 MSE【未验证】。

### 3.3 核心机制二：用 3D supervision 让 world memory 具有空间可读性

Stage2 从 current-clean hidden/Register 读取 frame-level pose，后续在数据可靠时扩展 depth/point。监督目标不作为输入，生成 future 也不作为 3D teacher。目标函数是：

\[
\mathcal{L}_{stage2}=\mathcal{L}_{visual}+\lambda_{3D}\mathcal{L}_{pose/depth/point}.
\]

这里的关键不是增加一个 pose head，而是研究哪一层 hidden 最能同时承载动态预测和空间信息。我们会把 layer probe 当作研究实验：若中间层有最佳 pose 可读性，说明 world model 的空间状态并不等于最后的 video denoising output；若 Register pooled hidden 优于 current hidden，说明历史压缩槽位确实保留了场景级信息。

### 3.4 核心机制三：用 action-centered mask 连接 video consequence 和 policy

共享 WanBlock 中保留三种动作变量：

```text
A_hist  历史已执行动作：写入 Register
A_cur   当前动作条件：用于预测 action-conditioned future video
A_noise policy noisy action：输出 A_out / combo logits
```

`Z_future` 可以读取 Register、Z_obs、text 和 A_cur；`A_noise` 可以读取 Register、Z_obs、text 和自身 action tokens，但不能读取 future visual noise、clean A_cur 或 target action。这样 video branch 可以给 action branch 提供训练期的 dynamics regularization，policy branch 却不依赖推理时不存在的 future video。

这套设计的可检验预测是：

- 去掉 future video loss，action 在分布简单的数据上仍可拟合，但在未见场景或长程闭环上泛化下降【未验证】。
- 允许 action 读取 future visual，会让训练 CE 变好，但 open-loop/closed-loop gap 变大【未验证】。
- 正确隔离后，policy-only inference 的计算量接近 action token 路径，而不是完整视频生成路径【未验证】。

### 3.5 核心机制四：三阶段不是三个独立模型，而是同一表征的三个约束

```text
Stage1  predictive memory
  让 Register 能利用历史预测短期 future visual consequence

Stage2  spatial readability
  让 current hidden/Register 能恢复 pose 等 3D 状态

Stage3  embodied action
  让同一表征在 instruction + current obs 下输出 action
  并用 visual/3D replay 防止 policy fine-tuning 破坏前两种能力
```

三阶段的逻辑关系是“预测性 → 空间可读性 → 可行动性”。如果 Stage1 没有稳定的 Register memory，Stage2 的 pose loss 只能学习 shortcut；如果 Stage2 没有空间约束，Stage3 可能只学动作频率；如果 Stage3 没有 replay，导航 fine-tuning 可能破坏 world model。三者必须分别有对照和联合结果，不能只报告一个总 loss【未验证】。

### 3.6 核心新意与已有工作的边界

我们的新意不应表述为“首次把 video model 用于 navigation”，因为 DreamZero、GigaWorld-Policy、LingBot-VA 等已展示了 WAM 路线；也不应表述为“首次使用 register 或 3D supervision”，因为 VGGT-Ω、FANTASYWORLD 和 memory-augmented VLA 已有相关机制。

更准确的研究命题是：**在 Wan-style video world model 内部，把 Register 设计成可在线更新且可被 3D probe 约束的 embodied history state，并通过 action-centered causal interface 让同一 shared backbone 在训练时预测视频后果、在推理时直接输出导航动作。** 这个组合是否比“视频 HPMC + 独立 policy”或“纯 streaming VLN memory”更能保持空间一致性和长程导航能力，就是本文需要实验证明的地方【未验证】。

### 3.7 当前科研判断

现阶段最值得投入的不是继续扩大模型或盲目增加训练步数，而是先把三个可证伪问题做成同一套协议：

```text
Register 是否真的记住了历史？
hidden/Register 是否真的可读出空间？
这个空间状态是否真的改善了闭环 action？
```

一旦这三个问题分别通过，视频质量、3D probe 和 VLN policy 才能组成一个完整科研故事；如果其中一项失败，也能明确知道失败发生在 memory、geometry 还是 action interface，而不是继续用总 loss 猜测模型发生了什么。

### 3.8 参考工作

- Infinite-World: <https://arxiv.org/abs/2602.02393>
- LingBot-World: <https://arxiv.org/abs/2601.20540>
- LingBot-World 2.0: <https://arxiv.org/abs/2607.07534>
- VGGT: <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html>
- VGGT-Ω: <https://vggt-omega.github.io/>
- DreamZero: <https://arxiv.org/abs/2602.15922>
- GigaWorld-Policy: <https://arxiv.org/abs/2603.17240>
- GigaWorld-Policy-0.5: <https://arxiv.org/abs/2607.13960>
- StreamVLN: <https://streamvln.github.io/>
- BridgeVLA++: <https://arxiv.org/abs/2608.05042>
- FANTASYWORLD: <https://openreview.net/pdf/b5f99909effe3a0756c724559123fbc9b7e5e448.pdf>
