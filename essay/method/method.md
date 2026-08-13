# 方法

我们提出 NAV，一个将空间感知流式世界模型（spatial-aware streaming world model）用作具身导航记忆的框架。NAV 的核心假设是：导航智能体所需的长程观测历史不必以原始帧序列、增长型 token cache 或任务专用 recurrent state 保存；相反，它可以被一个持续更新的世界模型状态表示。该状态在固定预算内递归吸收真实观测和已执行动作，并通过未来后果预测、三维几何监督和导航动作监督被塑造成可用于决策的世界表示。换言之，NAV 将 world model 的内部记忆从视频生成的辅助缓存，转化为 embodied agent 可读取的 observation history。

> 内部写作标注：按照 `doc/AGENT.MD`，本文草稿中尚未由代码和实验产物支撑的概念性能力保留 `【未验证】` 标注。Stage One 的 T4/IW-aligned Register world-model 训练链路已实现并运行；Stage Two 的 3D supervision 与 Stage Three 的 VLN policy 仍是已接受设计但尚未实现。

## 3.1 Embodied Observation History as Streaming World Representation

视觉语言导航可以被看作一个部分可观测决策问题。给定语言指令 \(q\)、当前观测 \(o_t\)、历史观测 \(o_{<t}\) 和已执行动作 \(a_{<t}\)，智能体需要预测当前动作：

\[
a_t \sim \pi(a_t \mid q, o_{\leq t}, a_{<t}).
\]

困难在于，显式历史 \(o_{\leq t}\) 会随 episode 长度线性增长。长程导航又恰恰依赖早期观测中的空间线索，例如已访问区域、房间连接关系、目标物体提示和返回路径。因此，一个可扩展的导航系统需要在不保留全部历史帧的前提下，为策略提供足够的环境状态。

NAV 将这一问题重新表述为流式世界表示学习。模型维护一个固定预算记忆 \(M_t\)，用它替代显式历史：

\[
a_t \sim \pi_\theta(a_t \mid q, o_t, M_t),
\]

其中 \(M_t\) 由真实观测流递归更新：

\[
M_t = \operatorname{Update}_\theta(M_{t-1}, C_t, A^{hist}_t).
\]

\(C_t\) 表示由当前真实观测构成的短视频 latent chunk，\(A^{hist}_t\) 表示历史中已经执行过的 motion/action。我们将 \(M_t\) 称为 embodied observation history：它不是原始帧缓存，也不是某个 episode 的可学习初始状态，而是一个由世界模型训练得到、随观测在线演化的 compact world representation。

这一定义带来两个因果约束。第一，\(M_t\) 只由真实 observation 和已执行动作更新；模型生成的 future 不回灌到历史记忆。第二，当前待预测 action 不作为 policy condition；policy 只能读取 instruction、当前观测和 \(M_t\)。因此，future visual 或 future 3D 只能作为训练时的 consequence supervision，而不能成为导航决策的输入。

## 3.2 Quasi-Infinite Spatial Register Memory

NAV 使用固定预算的 Spatial Register Memory 表示长程历史。与增长型 cache 不同，Register 的显式大小与 episode 长度无关；与滑动窗口不同，它不是简单保留最近若干帧，而是通过可学习 Updater 将历史观测递归写入同一个记忆状态。我们称这种接口为 quasi-infinite memory：模型可以持续接收任意长度的 observation stream，而 policy 每一步读取的记忆大小保持不变。

当前 V1 主线采用 `latent_prefix` Register。记忆状态具有 latent-like 空间布局：

\[
M_t \in \mathbb{R}^{B \times 16 \times 4 \times H \times W}.
\]

它与 Wan VAE latent 在通道数和空间分辨率上对齐，但语义上不是一段普通视频 latent，而是一个可被 DiT backbone 读取、可由 Updater 持续写入的世界状态。新 episode 开始时，Extractor 从第一个真实历史 chunk 建立初始记忆：

\[
M_0 = E_\theta(C_0, A^{hist}_0).
\]

之后，每个新观测 chunk 递归更新同一固定形状的 Register：

\[
M_i = U_\theta(M_{i-1}, C_i, A^{hist}_i), \quad i=1,\dots,t.
\]

Spatial Register Memory 的关键是保留二维空间组织。对于每个空间位置 \((h,w)\)，Updater 以该位置的 Register 时间槽作为 query，以当前 chunk 在同一空间位置的时间序列作为 key/value，执行 cross-attention 和残差 FFN 更新：

\[
M_i[:, :, :, h, w]
= U_\theta^{h,w}
\big(M_{i-1}[:, :, :, h, w], C_i[:, :, :, h, w]\big).
\]

这种位置保持的更新使历史信息被写入一个 spatial-aware memory，而不是被压成无结构的一维 token 序列。长期历史因此被表示为一个空间化世界状态，供后续 DiT 表征和导航策略读取。需要强调的是，quasi-infinite 描述的是固定预算的在线接口，而不是无损记忆假设；其长程有效性仍需通过 Register ablation、history shuffle、不同 history span 和 autoregressive rollout 验证【未验证：当前 Stage One 已覆盖 IW-1/4/8/16 等价历史窗口训练，但长程生成和导航有效性评测尚未完成】。

## 3.3 Spatial-Aware World-Model Backbone

NAV 以 Wan2.1-style DiT 作为共享 backbone。我们移除 InfiniteWorld 风格的 HPMC history compression，保留 latest local memory 表示最近局部观测，并将 Spatial Register Memory 作为长期历史接口。这样，模型同时拥有短程局部状态和长程压缩世界状态：\(Z_{local}\) 保留最近视觉细节，\(M_t\) 保留跨 chunk 历史。

每个视频样本被组织为 T4 micro chunks。一个 micro chunk 包含 13 个 RGB frames，相邻 chunk 的 stride 为 12，并由 Wan VAE 独立编码为 \([16,4,H,W]\) latent。NAV 不通过增加 future target 长度来获得长程建模能力，而是通过增加 Register update 次数扩展 history span。为对齐 InfiniteWorld 的历史范围，IW-1/4/8/16 chunks 分别对应 7/27/54/107 个 T4 micro history updates。

给定历史 chunks \(C_{0:t}\)，模型先递归得到 \(M_t\)。随后，DiT 以加噪 future latent、Register memory、latest local latent、当前 action condition 和文本条件为输入，预测 future visual consequence：

\[
\hat{v}_\theta =
f_\theta(Z^\sigma_{future}, \sigma, M_t, Z_{local}, A_{cur}, q).
\]

在 `latent_prefix` 变体中，\(M_t\) 作为 spatial latent prefix 注入 DiT 的 visual/main stream，使 Register、local observation 和 noisy future latent 在同一 backbone 中交互。`dit_condition` 变体将 Register 投影到 condition token 侧，作为 ablation 保留。主线选择 `latent_prefix`，因为它把历史作为时空状态而非静态文本条件处理。

该 backbone 的作用不是单纯生成更长视频，而是学习一个可被读取的世界表示。视觉预测要求 \(M_t\) 保留与未来观测相关的环境状态；后续几何监督进一步鼓励 DiT hidden/Register-after-DiT 表示编码 depth、pose 和 correspondence 等空间信息；导航监督再将这些表示接入 policy。

## 3.4 Action-Centered Navigation Interface

NAV 采用 action-centered 接口，以避免训练和推理之间的因果语义错位。我们区分三类 action token：

\[
A^{hist}, \quad A^{cur}, \quad A^{query}.
\]

\(A^{hist}\) 是历史中已经执行过的 action/motion，用于调制历史 chunk 并参与 Register update；\(A^{cur}\) 是视频后果预测中的当前 action condition，用于学习“在该动作下未来会看到什么”；\(A^{query}\) 是 policy 侧的待生成变量，用于输出导航动作。三者不能合并：如果把 \(A^{cur}\) 直接作为 policy condition，会把目标动作泄漏给动作预测；如果不提供 \(A^{cur}\) 给 generation branch，视觉后果预测又无法建模 action-dependent dynamics。

Stage One 中，action query/output branch 已经以固定格式保留，并从 Register 和 local latent 中读取上下文；但 action loss 默认关闭，因此视频 pseudo action 不会被训练成导航 policy 标签【已验证→`src/nav/stage_one_action_interface.py` 与 `scripts/train_v1_wan_stage1_t4_history.py`】。真正的导航动作监督只在 Stage Three 使用 VLN action labels 进行。

这一接口对应如下因果方向：

\[
\text{history/current observation} + \text{instruction}
\rightarrow \text{action}
\rightarrow \text{future consequence}.
\]

训练时，future visual 和 future 3D 是由 action 诱导的 consequence supervision；推理时，policy 只读取 \(M_t\)、当前观测和 instruction，直接输出 action。由此，NAV 在训练中利用世界模型学习密集环境结构，在推理中保持 policy-only 的高效决策路径。

## 3.5 Training Objectives and Staged Optimization

上述结构由三类目标统一训练。我们将 loss 和分阶段训练集中在本节描述，因为它们共同定义 \(M_t\) 如何从流式视频记忆转化为导航可用的世界表示。

### Stage One: predictive memory pretraining

Stage One 训练 Register 和 DiT backbone 进行 short-future consequence prediction。给定真实 future latent \(Z_{future}\)，RFlow / diffusion scheduler 采样噪声水平 \(\sigma\)，得到加噪 latent \(Z^\sigma_{future}\)，模型预测 velocity/noise target \(v^\*\)。视觉损失为

\[
\mathcal{L}_{visual}
=
\mathbb{E}_{Z_{future},\sigma,\epsilon}
\left[
\left\|
f_\theta(Z^\sigma_{future}, \sigma, M_t, Z_{local}, A_{cur}, q)
- v^\*
\right\|_2^2
\right].
\]

该损失只施加在 future target latent 上；Register prefix、latest local latent 和 condition token 不作为重建目标，而是通过 future prediction 误差获得梯度。当前正式 Stage One 使用

\[
\mathcal{L}_{stage1}=\mathcal{L}_{visual},
\]

并设置 action format loss 权重为 0。此阶段使用视频数据构造 T4 micro latent windows，从真实历史 chunks 递归更新 Spatial Register Memory，并预测下一个 \([16,4,H,W]\) future latent。当前实现从官方 `Wan2.1-T2V-1.3B` safetensors 初始化，移除 HPMC，采用 `latent_prefix` Register，并混合 IW-1/4/8/16 等价历史窗口【已验证→`doc/00_overview/project_invariants.md`、`doc/04_training/v1_stage_one_t4_iw_aligned.md`、`src/nav/infinite_adapter.py`、`src/nav/spatial_register_memory.py`、`scripts/train_v1_wan_stage1_t4_history.py`；当前运行记录见 `log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/train.log`，截至本次检查记录到 step 680，未见 complete 事件】。

### Stage Two: spatial-readable world representation

Stage Two 在 Stage One 的同一个 generation forward 上加入 3D supervision，而不是另建独立 3D backbone。我们从多个 DiT 层和 Register-after-DiT 表示中读取候选特征 \(F_t^l\)，通过轻量 probe 预测 depth、relative pose 和 point/correspondence，并选择最适合作为导航输入的层 \(l^\*\)。几何损失为

\[
\mathcal{L}_{3D}
=
\lambda_d \mathcal{L}_{depth}
+ \lambda_p \mathcal{L}_{pose}
+ \lambda_m \mathcal{L}_{point}
+ \lambda_c \mathcal{L}_{conf}.
\]

其中 depth loss 可采用 masked L1 或 scale-invariant log depth；pose loss 包含 rotation geodesic 和 translation direction 误差；point/correspondence loss 只在 pseudo label 置信度足够时启用。Stage Two 的总目标为

\[
\mathcal{L}_{stage2}
=
\mathcal{L}_{visual}
+ \lambda_{3D}\mathcal{L}_{3D}.
\]

3D target、pose 和 intrinsics 只作为监督信号，不作为 policy condition 或真实推理输入。该阶段的目标是让 streaming world-model memory 不只服务于视觉预测，也能被读出稳定的空间结构【未验证：Stage Two 的 3D probe、geometry heads 和选层实验尚未实现】。

### Stage Three: memory-conditioned embodied navigation

Stage Three 将选定层表示 \(F_t^{nav}=F_t^{l^\*}\) 输入 navigation head，预测未来 \(H\) 步动作：

\[
\pi_\theta(a_{t:t+H-1} \mid F_t^{nav}, o_t, q).
\]

默认 \(H=4\)。训练时监督 action chunk；闭环推理时只执行第一个 action，然后接收新观测并更新 Register。导航主损失为

\[
\mathcal{L}_{nav}
=
\operatorname{CE}
\left(
\operatorname{NavHead}_\theta(F_t^{nav}, o_t, q),
a^\*_{t:t+H-1}
\right).
\]

为了避免 imitation learning 破坏 Stage One 学到的预测式记忆和 Stage Two 学到的空间表征，Stage Three 训练中加入 video/3D replay：

\[
\mathcal{L}_{stage3}
=
\lambda_{nav}\mathcal{L}_{nav}
+ \lambda_{stop}\mathcal{L}_{stop}
+ \lambda_v\mathcal{L}_{visual}^{replay}
+ \lambda_{3D}\mathcal{L}_{3D}^{replay}.
\]

VLN policy 样本包含 instruction、当前观测、真实历史 observation/action prefix 和未来 action chunk；不构建 future visual 作为 policy 输入。video/3D replay 只在训练中维持 shared world representation，推理时不执行 generation branch 或 3D heads【未验证：Stage Three navigation head、VLN loss、replay cotrain 和 policy-only 推理裁剪尚未实现】。

具体数据规模、dataset split、sampling ratio、训练步数和实现超参属于实验设置。Method 中只保留会改变模型语义的数据组织：T4 micro chunk 定义视觉后果预测的时间粒度，IW-equivalent span 定义 long-history Register update 范围，geometry target 定义空间可读性监督，VLN action chunk 定义 policy 输出语义。

## 3.6 在线导航推理

在线推理时，NAV 维护一个随 episode 更新的 Register state。初始观测 chunk 经过 Extractor 得到 \(M_0\)。每次智能体执行动作并获得新观测后，Updater 将真实观测和已执行动作写入 \(M_t\)。policy 读取 \(M_t\)、当前观测 \(o_t\) 和 instruction \(q\)，输出动作：

\[
a_t = \pi_\theta(M_t, o_t, q).
\]

视频生成 branch 可用于训练时的 dense consequence supervision 或离线想象分析，但默认不参与导航推理；生成 future 也不会写回 Register。因此，NAV 的推理接口可以概括为

\[
\text{real observation history}
\rightarrow
\text{quasi-infinite spatial world memory}
\rightarrow
\text{embodied action}.
\]

这正是本文的核心：把 spatial-aware streaming world model 中持续更新的世界表示，转化为导航智能体可长期使用的 embodied observation history。
