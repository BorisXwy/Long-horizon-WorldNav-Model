# 引言

具身导航要求智能体在连续观察中理解环境、记住历史，并根据语言指令做出动作。与静态视觉理解不同，导航中的关键证据往往分散在长轨迹中：智能体可能需要在数十甚至数百步之后利用早期看到的房间布局、门洞方向、目标线索或已访问区域。因而，导航策略不仅需要当前视觉识别能力，还需要一种可在线更新、可长期保留、可被动作策略读取的 observation history。

现有视觉语言导航方法通常把这一问题处理为多帧历史编码：将若干历史图像、全景视角或视觉 token 输入策略模型，再通过 attention、cache、sampling 或 recurrent state 压缩历史。这样的设计在短程任务中有效，但面临两个根本限制。第一，显式历史的计算和存储成本随轨迹长度增长；第二，压缩得到的历史表示多是任务驱动的视觉表象，缺乏可验证的空间结构和动作后果建模。换言之，现有 memory 往往回答“如何保留更多观测”，但没有充分回答“保留下来的表示是否构成一个可用于导航的世界表示”。

视频世界模型提供了另一种可能。通过预测未来观测，world model 学到的内部状态必须编码场景布局、视角变化、运动规律和动作后果。近期长流式视频生成和交互式世界模型已经展示出用固定预算 memory、rolling cache 或层级记忆维持长 horizon 生成的一系列能力；3D-aware video/world models 进一步表明，生成模型的内部特征可以被 depth、pose、pointmap 等几何监督塑造成 spatial-readable representation；WAM 和 representation-only control 工作则显示，future modeling 的价值可以主要体现在训练时塑造动作表示，而不必在推理时显式生成 dense future video。

这些进展仍留下一个面向具身导航的缺口：长流式世界模型的 memory 通常服务于生成连续性，而不是导航策略；3D-aware world models 面向生成或重建，尚未把几何可读表示作为在线 navigation memory；WAM 多集中于短程操作或 action-only policy，缺乏面向长程空间导航的递归历史接口。因此，一个自然问题是：

**一个 spatial-aware streaming world model 中近似无限长的 memory / world representation，能否作为具身智能体的 observation history？**

本文提出 NAV，围绕这一问题构建导航框架。NAV 维护一个固定预算的 Spatial Register Memory，将真实观测流和已执行动作递归写入同一个空间化世界状态。该状态具有 quasi-infinite memory 接口：episode 可以持续增长，但策略每一步读取的显式记忆大小保持固定。与普通视频 cache 不同，NAV 的 Register 被设计为 embodied observation history：它通过未来视觉后果预测学习可预测的环境状态，通过三维几何监督获得 spatial readability，并在导航阶段作为 policy 的输入。

NAV 的方法有三个核心设计。首先，我们将 observation history 重新表述为 streaming world representation。模型不直接向策略暴露全部历史帧，而是维护一个随真实观测在线更新的 \(M_t\)。其次，我们采用空间保持的 Register update，使历史在固定预算下被写入 \([B,16,4,H,W]\) 形式的 spatial memory，而不是一维 token cache。第三，我们使用 action-centered interface 区分历史已执行动作、视频后果预测动作和 policy query action：future visual/3D 只作为训练时 consequence supervision，推理时导航策略只读取 memory、当前观测和 instruction。

本文的预期贡献如下：

1. 我们提出将 spatial-aware streaming world model 的 quasi-infinite memory 作为 embodied observation history，用统一的世界表示连接长期记忆、空间理解和导航决策。
2. 我们设计 Spatial Register Memory，在固定显式预算下递归吸收真实观测历史，并保持与视频 latent 对齐的空间结构。
3. 我们提出分阶段训练目标：Stage One 用视觉后果预测训练 predictive memory，Stage Two 用 3D supervision 使 memory spatial-readable，Stage Three 用 VLN action supervision 将该 memory 接入 policy。
4. 我们预留一组实验，用于验证 long-history memory、geometry readability、action-centered interface 和 navigation performance 的必要性与有效性；截至当前，Stage Two/Three 及多数概念性结论仍为【未验证】，需要后续实验产物支撑。
