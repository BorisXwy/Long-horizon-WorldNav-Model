# Related Work

NAV 关注的问题是：spatial-aware streaming world model 中持续更新的 memory 能否作为具身智能体的 observation history。围绕这一问题，相关工作可以分为四条线索：视觉语言导航中的历史建模、长流式视频世界模型、几何可读的空间世界表示，以及 world-action models / representation-only control。

> 内部写作标注：本文对 NAV 自身能力结论保留 `【已验证】/【未验证】` 标注；他人工作描述作为背景，不加验证标注。

## VLN 中的历史建模

视觉语言导航长期依赖历史观测。早期方法通常将历史全景特征、轨迹 token 或 recurrent state 输入策略；近期方法进一步引入更长上下文、更复杂的 token 管理和显式记忆机制。例如，StreamVLN 使用 slow-fast context 组织短程响应与长程记忆，NavFoM 通过预算感知采样在固定 token 预算下选择历史，Qwen-RobotNav 将观测历史的数量、时间衰减和相机权重作为可控变量，Robostral Navigate 使用 prefix caching 和 tree attention mask 管理 episode 序列并避免动作泄漏。另一条相关路线把几何感知前端接入导航，例如 LoGoPlanner 使用 metric-aware geometry tokens 驱动端到端导航策略。

这些工作表明，长程 history 对导航性能至关重要。然而，它们的记忆通常是为导航任务定制的视觉表象压缩：历史被保留为帧、view tokens、cache 或显式几何前端，而不是一个由环境预测任务训练得到的 streaming world representation。因此，它们主要解决“策略如何访问更多历史”，但较少回答“历史表示是否编码了可预测、可几何读出的环境状态”。NAV 的目标正是在这一点上推进：我们不把 memory 作为 policy 的附属模块，而是将 world model 训练出的空间化 Register 作为 embodied observation history。

## 长流式视频世界模型

视频世界模型通过预测未来观测学习环境状态。长流式视频生成工作进一步研究如何在有限上下文下维持长 horizon 的一致性。InfiniteWorld 使用 pose-free hierarchical memory 将交互式世界模型扩展到 1000-frame 级别；Self-Forcing、Causal-Forcing、Rolling Forcing、LongLive 等通过 autoregressive diffusion、rolling cache、attention sink 或 distillation 缓解 train-test gap 和长程误差积累；RELIC、WorldMem 等工作进一步探索压缩 memory、外部 memory bank 或 state-aware retrieval。

这些方法证明了 streaming world model 可以在长时序下维护有用历史，但其 memory 主要服务于视频生成质量和 temporal consistency。对 NAV 而言，关键区别在于 memory 的用途：我们并不只追求生成更长视频，而是将流式 world model 的内部状态转化为导航策略可读取的 observation history。换句话说，NAV 借鉴长流式视频生成的 memory 机制，但将其目标从 generation continuity 推向 embodied decision making。

## 几何可读的空间世界表示

另一条重要线索是 3D / spatial-aware world models。GeometryForcing、FantasyWorld、AETHER、Voyager、DeepVerse 等工作将 depth、pose、pointmap、4D reconstruction 或 geometry-aware retrieval 注入视频生成，使生成模型不只拟合像素动态，也学习空间结构。VGGT-Ω 代表了更强的 feed-forward spatial reconstruction 方向：通过 scalable architecture、register attention 和多任务几何损失，证明 large-scale visual backbone 的中间表示可以服务于 depth、camera、point 和 matching 等空间任务。GenieDrive 等 driving world model 也将 4D occupancy 或显式几何引入生成。

这些工作为 NAV 提供了重要启发：world model 的 hidden state 可以通过几何监督变得 spatial-readable。然而，它们多数面向生成、重建或驾驶仿真，而不是将该空间表示作为在线 VLN memory。NAV 将几何监督放入 streaming memory 的形成过程：Stage Two 不另建独立 3D backbone，而是在同一 world-model forward 中选择 Register-after-DiT 层并施加 depth/pose/pointmap supervision，使 memory 同时服务于 future prediction 和 navigation readout【未验证：Stage Two 尚未实现】。

## World-Action Models 与 representation-only control

WAM 路线把世界模型从预测器推进到决策器。DreamZero、V-JEPA 2、DreamGen、EnerVerse-AC、WorldPlanner、mimic-video 等工作使用 action-conditioned prediction、latent planning、synthetic rollout 或 action decoder 连接 world modeling 和 robot control。近期 Fast-WAM、GigaWorld-Policy、ImageWAM 等进一步强调 representation-only 或 action-centered 设计：video modeling 在训练时塑造 shared representation，推理时可以关闭 dense video generation，直接进行 action prediction。

这一路线与 NAV 的 action-centered interface 最接近。它们说明 future modeling 不必在推理时显式生成视频，仍可以通过训练塑造动作表示。但多数 WAM 面向短程操作或机器人控制，缺乏长程空间导航所需的 persistent memory；同时，它们通常没有显式 depth/pose/pointmap supervision，难以保证 learned representation 具有导航所需的几何可读性。NAV 继承 representation-only WAM 的思想：future visual/3D 是训练时 consequence supervision，推理时只执行 policy path；但 NAV 将这一思想扩展到 quasi-infinite spatial memory，并把该 memory 定义为 embodied observation history。

## 与现有工作的区别

上述工作分别解决了 NAV 所需能力的一部分：VLN history 方法处理长观测输入，streaming world models 维护长程生成记忆，spatial world models 使表示具备几何可读性，WAM 将 future modeling 连接到动作预测。NAV 的区别在于将这些能力统一到同一个在线 Register memory 中：

- 相对 VLN history 方法，NAV 的 memory 不是任务定制的历史表象压缩，而是由 future prediction 和 geometry supervision 共同训练的 streaming world representation。
- 相对长流式视频世界模型，NAV 不仅让 memory 服务于 generation continuity，还将其作为 navigation policy 的 observation history。
- 相对 3D-aware world models，NAV 不止验证 hidden state 可读出几何，还将 spatial-readable memory 下游接入具身决策【未验证】。
- 相对 WAM，NAV 保留 action-centered / policy-only inference 的效率，但把短程 action representation 扩展为长程、空间化、递归更新的 navigation memory。

一句话，NAV 试图证明：**quasi-infinite memory in a spatial-aware streaming world model can serve as embodied observation history for navigation**【未验证：完整 Stage Two/Three 与导航评测尚未完成】。
