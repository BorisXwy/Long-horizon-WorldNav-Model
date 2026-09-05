# R2R-CE 训练型导航模型 Action Head 与预测形式调研

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-012` |
| 类型 | Benchmark / Architecture Survey |
| 状态 | Active / Literature-audited Snapshot |
| 更新时间 | 2026-09-06 |
| 职责 | 统一记录 R2R-CE 最新训练型模型的排名口径、Action Head、预测变量、监督形式与闭环执行方式 |

## 结论先行

截至 2026-09-06，按 **R2R-CE Val-Unseen、SR 优先、SPL 破同分、每项工作只取一个最佳标准配置** 的文献报告值排序，instruction-only 主口径的前十为：

1. Robostral Navigate：77.4 SR / 74.2 SPL；
2. Qwen-RobotNav-8B：72.1 / 66.6；
3. OmniNav：69.5 / 66.1；
4. AgentVLN-3B：67.2 / 64.7；
5. ABot-N0：66.4 / 63.9；
6. SPAN-Nav：66.3 / 59.3；
7. NavForesee：66.2 / 59.7；
8. TAMP-Nav：66.2 / 58.8；
9. Dual-Anchoring：65.6 / 62.1；
10. AwareVLN：65.4 / 55.1。

最重要的结构结论不是“哪一种 head 一统天下”，而是：

- 前十中没有一个方法把“`hidden -> Linear -> 4-way primitive logits`”作为唯一的完整导航接口。
- 五项工作预测有明确空间意义的连续中间变量：Qwen-RobotNav、OmniNav、ABot-N0、SPAN-Nav、NavForesee；其中三项是 MLP regression，两项是 Flow Matching。
- 另外五项保留 VLM 原生 token generation：Robostral、AgentVLN、TAMP-Nav、Dual-Anchoring、AwareVLN；它们预测 pixel waypoint、skill call 或 textualized action sequence，再由 controller/parser 执行。
- `action chunk` 不是问题本身。领先方法普遍使用 `H=4/5/8/30`，但大都采用 receding-horizon：每次只执行短前缀或跟随一个 waypoint，获得新 observation 后重规划。
- `STOP` 总是显式建模：离散 token/class、arrival flag，或 waypoint 之外的终止输出；没有把 STOP 仅当普通连续坐标隐式学习。
- R2R-CE 的高 SR 不能只归因于 Action Head。额外 panoramic/stereo/depth、target prior、外部 SLAM/controller、数据规模和 online RL 都会显著改变难度。

对 NAV 最直接的含义是：当前四类离散 head 可以作为必要的 **reactive baseline**，但不应被当作唯一候选。下一轮最有信息量的受控对照应是同一 Register/backbone 上比较：

1. `H=1` primitive CE；
2. `H=4` textual/discrete action chunk；
3. `H=5/8` egocentric waypoint head + 固定 low-level controller。

只有这样才能区分“Register/Backbone 没学到导航状态”和“输出空间过于离散、缺少几何 inductive bias”这两个原因。该建议是基于外部架构的综合判断，尚未在 NAV 上完成对照，记为【未验证】。

## 排名与可比性口径

### 为什么主表使用 R2R-CE，而不是经典 R2R

经典 R2R 在 Matterport3D navigation graph 上运行，动作是“选择当前节点的某个可导航相邻 viewpoint，或 STOP”。候选数量随状态变化。R2R-CE 则在 Habitat 连续环境中执行，最终落到 forward/turn/stop 或连续 waypoint/controller，更接近 NAV 当前数据与闭环执行。

因此：

- 经典 R2R 的 head 常是 `candidate score=[B,K_t+1]`，不是固定四分类；
- R2R-CE 的 model-level output 可能仍是 waypoint、pixel 或 action token，之后再转成 Habitat primitive；
- 两者的 SR/SPL 不能混排。

### 本快照的具体规则

- split：只排 **Val-Unseen**，不混入 Test、Val-Seen、RxR-CE 或物理机器人结果；
- 主指标：按 SR 降序；SR 相同按 SPL 降序；
- 方法去重：Qwen-RobotNav-4B/8B 只保留8B panoramic 最佳行；
- 包含：通过 imitation learning、SFT、DAgger、RFT/GRPO 等训练得到的导航模型；
- 排除：Human、zero-shot agent、不同的 VLN-PE 物理控制协议、明确使用精确/模糊 target-location prior 的方法；
- 数字性质：均为原论文或官方项目的 **self-reported Val-Unseen**，不是本项目统一环境重跑；
- 开放状态不参与排名，但单独记录。

EvalAI 动态页面目前无法稳定导出完整历史榜单，因此这里准确称为“可审计的最新文献榜”，不声称是 EvalAI 官方 Test leaderboard 的逐行镜像。

### StereoNav 的 81.1% 为什么不进入 instruction-only 主榜

StereoNav 报告 R2R-CE Val-Unseen 81.1 SR / 68.3 SPL，但其输入明确包含：

- stereo RGB；
- navigation instruction；
- **target-location prior**，即 fuzzy 或 precise 目标位置提示；
- 当前 pose 用于把目标先验持续渲染到视图中。

这不是标准的“只给 instruction、从起点寻找终点”设定。其 Action Head 是原生 LM head，自回归产生未来4步 textualized primitive action，动作空间为 `{FORWARD 0.25m, LEFT 15°, RIGHT 15°, STOP}`，并同时用 stereo depth head 做辅助监督。它适合作为“有目标位置先验时 action token head 能做到什么”的 upper-bound，不适合放进 instruction-only 排名。

来源：[StereoNav paper](https://arxiv.org/abs/2605.13328)、[official code](https://github.com/Yunheng-Wang/StereoNav)。

## 前十总表

| 排名 | 工作 | Backbone / 参数 | Observation 与额外系统 | 模型级输出 | Action Head / Loss | 闭环执行 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Robostral Navigate | in-house VLM 8B + diffusion policy 121M | monocular RGB history；无 depth/LiDAR | 可见时 `(u,v,Δx,Δy,Δθ)`；不可见时 `(Δx,Δy,Δθ)`；另有 STOP；低层为 `30×(dx,dy,dθ)` | VLM autoregressive output + 121M diffusion trajectory policy；SFT 后 CISPO | VLM 0.5Hz 选粗 waypoint；diffusion policy 10Hz 产1秒 trajectory；controller 100Hz |
| 2 | Qwen-RobotNav-8B | Qwen3-VL 8B | panoramic/multi-camera RGB history；task-adaptive token budget | `8×(x,y,θ)` | 4-layer MLP，hidden=512，GELU，24-d output；trajectory MSE + V-L next-token loss | 一次并行产8 waypoint，短前缀执行后重规划 |
| 3 | OmniNav | Qwen2.5-VL-3B | 最多20帧、panoramic history；fast-slow system | `5×(x,y,sinθ,cosθ,arrive)` | DiT Flow-Matching policy；CFM velocity loss；5-step Euler inference | fast policy 最多5Hz；waypoint 交给 speed/local controller |
| 4 | AgentVLN-3B | Qwen2.5-VL-3B | RGB-D、pose、增量 occupancy/topological map、skill library | structured skill call / pixel-aligned waypoint；盲区 fallback 为 `{Forward,Left,Right}` | 无独立固定维度数值 head；VLM 原生语言/skill decoding，instruction tuning | perception skill 更新图；planning skill 执行可变 `τ` 步；必要时 fine-grained correction |
| 5 | ABot-N0 | Qwen3-4B Brain + Action Expert | panoramic navigation history；层级 controller | `5×(x,y,θ)` local-BEV waypoints | Conditional Flow Matching head；Phase2 `next-token CE + CFM` | waypoint 由 CE-Nav/neural controller 转成 `vx,vy,vyaw` |
| 6 | SPAN-Nav | Qwen3-VL | multi-view RGB video；单个 learned spatial token；occupancy supervision | `M×(x,y,θ)` egocentric local trajectory；论文未披露固定 M | action hidden token -> multi-layer MLP；trajectory MSE | trajectory 交给 embodiment-specific local planner/controller |
| 7 | NavForesee | Qwen2.5-VL-3B | panoramic RGB、history pose embedding；short/long dream queries | `5×(x,y,sinθ,cosθ,arrival)` | 1 action query -> 2-layer Transformer -> 2-layer MLP/ReLU；action MSE | 到达标志控制 STOP；其余按 waypoint 连续执行并重规划 |
| 8 | TAMP-Nav | Qwen2.5-VL-7B | 四方向 RGB；depth 投影与固定 SLAM controller | `choice:view, pixel:[u,v]` 或 STOP | 原生 LM structured output；SFT + two-level GRPO | 每次指向一个 pixel waypoint，投影为3D local point，再由 SLAM执行 |
| 9 | Dual-Anchoring | StreamVLN/LLaVA-Video | monocular RGB stream；progress text + landmark-memory auxiliary | textualized low-level action sequence | 原生 LM next-token navigation loss `L_nav`；附加 `L_prog` 与 landmark feature MSE | parser 转为若干 25cm/15° primitive，执行后重新读取视频流 |
| 10 | AwareVLN | unified VLM | monocular RGB stream，均匀采样8帧 | 先输出 `[REASON]`/`[ACT]`；ACT 后生成动作文本并解析为1个或多个 primitive | special-token logits + native text-generation head；reason/action SFT | 稀疏 reasoning；ACT 模式执行 action sequence，再追加新帧 |

注：参数量、sensor 与输出格式来自各论文/官方项目；没有披露的维度不做推测。`M` 未知不等于单步。

## 逐项结构解析

### 1. Robostral Navigate：image-space pointer + 低层 diffusion policy

Robostral 不是一个普通 Action Head，而是两级系统。

第一级 8B VLM 读取 instruction 与 RGB history。若目标 waypoint 当前可见，输出：

\[
a_{vis}=(u,v,\Delta x,\Delta y,\Delta\theta),
\]

其中 `(u,v)` 是当前图像像素，后3项作为共同训练的 metric fallback。若目标不可见，则只输出：

\[
a_{invis}=(\Delta x,\Delta y,\Delta\theta).
\]

它还单独预测 STOP。第二级 121M diffusion policy 读取 coarse waypoint、机器人高度/半径、VLM 请求时的 context frame 和当前 frame，输出未来1秒的30个 `(dx,dy,dθ)`。因此论文中的“VLM action”与最终 motor command 中间还有学习式局部策略。

训练上，它把整条 episode 打包为 `I|O0|a0|...|On|an`，用 tree attention mask 让每个 action branch 只能看到对应 observation prefix，不能偷看之前的 ground-truth action。这样一次 forward 覆盖所有决策点，并把序列计算从朴素的 `O(T²)` 降到 `O(T)` token processing。SFT 后再用 CISPO 做 online RL。

证据：[paper](https://arxiv.org/abs/2607.20785)、[official introduction](https://mistral.ai/news/robostral-navigate/)。当前未见完整训练代码/权重发布。

### 2. Qwen-RobotNav：final hidden -> 4-layer MLP -> 8 waypoints

这是前十中最清楚、也最容易在 NAV 上做受控复现的 regression head：

```text
Qwen3-VL final trajectory hidden E_A [B,d]
  -> Linear(d,512) + GELU
  -> hidden layers
  -> Linear(512,24)
  -> reshape [B,8,3] = (x,y,theta) × 8
```

坐标按数据集99th percentile scale 归一化到 `[-1,1]`；训练用 waypoint MSE：

\[
L_{traj}=\lVert\hat W-W^*\rVert_2^2.
\]

总损失再加入 navigation-related V-L next-token loss，防止全量 trajectory regression 把 VLM 训成只看局部图像的 reactive mapper。它不需要 flow/noise，也不按8个 waypoint 逐 token 解码：一次 backbone forward 后并行得到24个连续量。

证据：[technical report](https://arxiv.org/abs/2606.18112)、[official repository](https://github.com/QwenLM/Qwen-RobotNav)。官方仓库当前只发布资料，不发布模型权重。

### 3. OmniNav：VLM condition + 小型 DiT Flow Matching

OmniNav 的 VLM 将 instruction、image history 与可选 coordinate token 融合为 `O_VLM`。Action Head 不是读一个 hidden 做线性回归，而是：

```text
noised waypoint tokens [B,5,5]
  -> DiT self-attention（waypoint temporal/spatial interaction）
  -> cross-attention(K,V = O_VLM)
  -> velocity [B,5,5]
```

每个 waypoint 是 `(x,y,sinθ,cosθ,arrival)`。训练随机采样 flow time `τ`，预测从 noise 到 GT trajectory 的 velocity；推理从 Gaussian noise 出发做5次 Euler integration。与大视频 diffusion 不同，noise token 只有25个标量，所以5步仍可达到5Hz左右。

证据：[ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad6363efb7af02f7db13d087c7e649bd-Abstract-Conference.html)、[official code](https://github.com/amap-cvlab/OmniNav)。

### 4. AgentVLN：Action Head 被替换成 skill scheduling interface

AgentVLN 的核心不是“小 head”，而是把 action space 从 primitive 改写成 perception/planning skill library。VLM 输出 structured call `c_k`；perception skill 可读取 RGB-D/pose 并更新 occupancy/topological map，planning skill 接受 pixel-aligned waypoint 并闭环执行 `τ>0` 个 low-level action。

只有在路径投影不可见、遮挡或偏航时，它才直接生成 `{Forward, Left, Right}` 作为细粒度修正。目标附近再通过视觉定位/技能终止。因此其67.2%不能解释为“3B VLM 的四分类 head 达到67.2%”：分数包含显式几何映射、全局图、技能库和局部控制器。

证据：[paper](https://arxiv.org/abs/2603.17670)、[official repository](https://github.com/Allenxinn/AgentVLN)。论文没有给出一个可与 `Linear(d,4)` 对照的独立 Action Head 参数表。

### 5. ABot-N0：Brain-conditioned Flow Matching Action Expert

ABot-N0 的 Qwen3-4B Cognitive Brain 提供 navigation context，Action Expert 预测 local BEV 中5个 `(x,y,θ)`。其选择 Flow Matching 的理由是相同状态可能存在绕左/绕右等多模态可行轨迹；MSE 会将两个 mode 平均成可能碰撞的中间轨迹。

Phase 2 同时优化：

\[
L=\lambda_{txt}L_{NTP}+\lambda_{flow}L_{CFM}.
\]

Phase 3 冻结 Brain，仅用 SAFE-GRPO 微调 Action Expert。模拟/真机执行还依赖 waypoint controller，最终输出速度 `v_x,v_y,v_{yaw}`。

证据：[technical report](https://arxiv.org/abs/2602.11598)、[project repository](https://github.com/amap-cvlab/ABot-Navigation)。

### 6. SPAN-Nav：空间 token 先显式成形，再进入 trajectory MLP

SPAN-Nav 在 Qwen3-VL 内压出一个 spatial token，并用 occupancy reconstruction、latent consistency 约束它。Action path 为：

\[
E_t^{act}=VLM(E^L,E_t^V,Proj_{O2H}(z_t^{out})),\qquad
T_t=AH(E_t^{act}),
\]

其中 `AH` 是 multi-layer MLP，`T_t={M×(x,y,θ)}`，Action Loss 是 trajectory MSE。训练 Stage I 用 GT occupancy token teacher-forcing；Stage II 改为 self-predicted spatial token，缩小训推差异。

这项工作对 NAV 的启发不是“加 pose loss 就会更好”，而是 3D supervision 必须进入 policy 真正消费的中间变量；否则几何 head 收敛不代表 action hidden 使用了几何信息。

证据：[paper](https://arxiv.org/abs/2603.09163)。截至检索日未确认完整官方代码仓库。

### 7. NavForesee：action query 读取 history + 两级 future query

NavForesee 把 history pose embedding、instruction、历史 RGB、short-horizon dream query 和 long-horizon dream query 放入 Qwen2.5-VL-3B。单个 learnable action query 可读所有这些信息，之后经过2-layer Transformer 和2-layer MLP/ReLU 输出5个 `(x,y,sinθ,cosθ,arrival)`。

Action Loss `L_a` 使用 MSE；同时有 depth SiLog loss 与 DINOv2/SAM semantic-feature MSE：

\[
L=\alpha L_d+\beta L_c+L_a.
\]

它是“从 predictive representation 中读取 action”的直接案例，但它使用显式 history pose embedding，不能作为无 pose Register 自然获得3D能力的证据。

证据：[paper](https://arxiv.org/abs/2512.01550)。截至检索日未见官方训练代码。

### 8. TAMP-Nav：原生 VLM 输出 view + pixel waypoint

TAMP-Nav 每步接收四个方向视图，原生 LM 输出类似 `choice: front, pixel: [u,v]` 的结构化文本，或 STOP。Depth 将 pixel 投影为局部3D点，固定 non-learned SLAM controller 完成低层执行。

其 SFT cold start 使用 Qwen2.5-VL-7B 与90k MultiNav-CoT trajectories；随后 two-level GRPO 同时使用 step-level local reward 与 episode-level global reward。这里没有 `Linear(d,4)`，pixel 坐标利用了 VLM 原有 visual grounding 能力。

证据：[paper](https://arxiv.org/abs/2608.17512)、[official repository](https://github.com/ZJU-OmniAI/Embodied-Navigator)。

### 9. Dual-Anchoring：StreamVLN 动作文本头不变，训练期加双锚定

Dual-Anchoring 继承 StreamVLN/LLaVA-Video 的自回归动作文本接口，输出类似“move forward 25cm, turn left 30°...”的序列，解析为若干 primitive。它没有把 action decoder 改成额外 MLP，而是在训练期增加：

- progress description next-token objective；
- 从历史 hidden 恢复最近 landmark 的 SAM feature MSE；
- DAgger corrective trajectory。

因此它说明：即使部署时 Action Head 完全不增加计算，训练期的状态锚定也可能提升原来的动作 token decoder。但论文报告的是整体训练策略效果，不能单独归因为 landmark head。

证据：[paper](https://arxiv.org/abs/2604.17473)。论文写明计划发布代码，但截至检索日未确认完整 release。

### 10. AwareVLN：先分类 REASON/ACT，再用同一个 LM head 生成动作

AwareVLN 每步先比较 special-token logits：

```text
d[REASON] > d[ACT] -> 生成 scene/progress/plan reasoning
otherwise          -> 生成 textual movement -> parser -> primitive sequence
```

primitive ontology 是 `{FORWARD, TURN-LEFT, TURN-RIGHT, STOP}`，但预测并不是一个简单 `4-way logits`：模型生成诸如“move forward 75cm”的文本，parser 展开成一个或多个25cm/15°动作。推理始终闭环追加新帧；reasoning 只在关键节点触发。

证据：[paper](https://arxiv.org/abs/2605.22816)、[official code](https://github.com/GWxuan/AwareVLN)。

## 榜外但重要的结构对照：ETP-R1

ETP-R1 的65.36 SR / 55.82 SPL 略低于前十截止线，但它提供最干净的 graph-policy 对照。

它先用冻结的 waypoint predictor 从 panoramic RGB-D 生成局部候选点，再维护全局 topological graph。Policy Head 对每个 candidate node（加一个 STOP node）独立给 scalar：

```text
node feature g_i + text-attended feature
  -> Linear(2d,2d) + ReLU + LayerNorm + Dropout
  -> Linear(2d,1)
  -> logits [B,K_t+1]
  -> CrossEntropy(candidate_or_STOP)
```

被访问/无效节点 mask 为 `-inf`，argmax/sample 得到高层节点；deterministic controller 将节点转成 Habitat primitive。训练为 joint pretraining（SAP+MLM）-> online DAgger SFT -> GRPO RFT。

证据：[paper](https://arxiv.org/abs/2512.20940)、[official code](https://github.com/Cepillar/ETP-R1)。官方代码中的 `NextActionPrediction` 和 `F.cross_entropy(global_logits, teacher_actions)` 与论文描述一致。

## Action Head 形式的横向归纳

| 类型 | 输出空间 | 代表工作 | 优点 | 主要代价/风险 |
| --- | --- | --- | --- | --- |
| primitive classification / textual primitive | 4个低层动作或其文本序列 | AwareVLN、Dual-Anchoring、StereoNav | 与 Habitat label 直接一致；CE/NTP 易实现 | 长串 forward 占比高；几何距离表达弱；易学 action prior |
| continuous waypoint regression | `H×(x,y,θ)` | Qwen-RobotNav、SPAN-Nav、NavForesee | 一次 forward 并行；几何含义明确；head 很小 | 单峰 MSE 可能平均多模态路径；依赖坐标归一化/controller |
| generative waypoint head | noised `H×D_a` -> velocity | OmniNav、ABot-N0 | 能表达多模态路径；并行 action chunk | 多次小 head integration；训练与采样实现更复杂 |
| visual pointing | `(view,u,v)` + STOP | Robostral、TAMP-Nav | 对齐 VLM visual grounding；弱化 embodiment/尺度依赖 | 不可见目标需 fallback；常依赖 depth/SLAM/controller |
| dynamic candidate scoring | `K_t` candidate + STOP | ETP-R1；经典 R2R 主流 | 大幅缩短 decision horizon；规划含义强 | 依赖 waypoint proposal/topological graph；不是 end-to-end primitive control |
| skill call | function id + arguments | AgentVLN | 将复杂控制分给专用技能；长程决策次数少 | 系统组件多，head 分数不能独立归因于 backbone |

## 对 NAV Stage Three 的可执行结论

### 不能从榜单推出的结论

- 不能说“Flow Matching 一定优于 CE”：最佳方法 Robostral 是混合系统，Qwen-RobotNav 使用简单 MSE，而 AwareVLN 使用文本动作。
- 不能说“head 越复杂越好”：Qwen 的4-layer MLP 已达到72.1%，核心能力主要在 backbone、数据和 observation protocol。
- 不能用 StereoNav 81.1% 证明四动作 token head 优于 waypoint head，因为它额外读取 target-location prior。
- 不能把 AgentVLN 67.2% 视作纯3B end-to-end action model：它使用 depth/pose/map/skill library。

### 对当前训练困难最相关的解释

当前 NAV 的 direct discrete policy 若长期停在 class-prior 附近，可能同时受三件事影响：

1. 输出空间只有四类，`FORWARD` 在视觉相近状态下对应不同剩余距离，head 得不到连续几何目标；
2. action target 与 current observation / instruction progress 未严格对齐时，增加 history 只会增加条件熵；
3. Register 虽保存历史，不代表最后的 action slot 能稳定读取“相对目标方向、可行空间、终止状态”。

外部前十共同采取的补救手段分别是：显式 waypoint/pixel bottleneck、progress reasoning、3D/occupancy auxiliary、DAgger/on-policy state、或大规模 trajectory co-training。它们共同反驳了“只换一个更大的线性 head 就足够”的假设。

### 建议的最小三组受控实验

以下均固定同一 `Register + Z_obs + instruction + shared WanBlock`、同一训练 window、同一数据与 EBS，只改 output interface：

| 实验 | 输出与 Loss | 执行 | 目的 |
| --- | --- | --- | --- |
| A：Reactive baseline | `[B,1,4]`，class-balanced CE | argmax单步，下一 observation 重规划 | 判断 backbone 是否至少能学当前 primitive |
| B：Action-token chunk | `[B,4,4]`，逐 slot CE；只执行第1步 | receding-horizon | 对齐 StreamVLN/Aware/Stereo 类接口，同时保留固定 token 位置 |
| C：Waypoint head | `[B,5,5]` 或 `[B,8,3]`，MSE；arrival 单独 BCE | controller 跟随首个有效 waypoint，再重规划 | 检验连续几何 bottleneck 是否比四类 action 更易从 Register 读出 |

如果 C 明显收敛而 A/B 不收敛，问题主要是 primitive label 的多解性与时间对齐；如果三者都不收敛，才更有理由归因于 Register/backbone 或 dataloader。此实验判据尚未在 NAV 完成，记为【未验证】。

## 一手来源与开放状态

| 工作 | 论文 | 官方实现状态（检索日） |
| --- | --- | --- |
| Robostral Navigate | [arXiv:2607.20785](https://arxiv.org/abs/2607.20785) | 未见完整训练代码/权重 |
| Qwen-RobotNav | [arXiv:2606.18112](https://arxiv.org/abs/2606.18112) | [官方资料仓库](https://github.com/QwenLM/Qwen-RobotNav)，明确暂无权重发布计划 |
| OmniNav | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad6363efb7af02f7db13d087c7e649bd-Abstract-Conference.html) | [完整官方仓库](https://github.com/amap-cvlab/OmniNav) |
| AgentVLN | [arXiv:2603.17670](https://arxiv.org/abs/2603.17670) | [官方仓库](https://github.com/Allenxinn/AgentVLN)，release 完整度需逐项检查 |
| ABot-N0 | [arXiv:2602.11598](https://arxiv.org/abs/2602.11598) | [项目仓库](https://github.com/amap-cvlab/ABot-Navigation)，完整训练资产需逐项检查 |
| SPAN-Nav | [arXiv:2603.09163](https://arxiv.org/abs/2603.09163) | 未确认完整官方仓库 |
| NavForesee | [arXiv:2512.01550](https://arxiv.org/abs/2512.01550) | 未见官方训练代码 |
| TAMP-Nav | [arXiv:2608.17512](https://arxiv.org/abs/2608.17512) | [官方仓库](https://github.com/ZJU-OmniAI/Embodied-Navigator) |
| Dual-Anchoring | [arXiv:2604.17473](https://arxiv.org/abs/2604.17473) | 论文承诺 release；检索日未确认完整仓库 |
| AwareVLN | [arXiv:2605.22816](https://arxiv.org/abs/2605.22816) | [官方仓库](https://github.com/GWxuan/AwareVLN) |
| ETP-R1 | [arXiv:2512.20940](https://arxiv.org/abs/2512.20940) | [完整官方仓库](https://github.com/Cepillar/ETP-R1) |
| StereoNav（非主榜） | [arXiv:2605.13328](https://arxiv.org/abs/2605.13328) | [完整官方仓库](https://github.com/Yunheng-Wang/StereoNav) |

VLN-CE 官方任务、动作与 baseline 入口：[official VLN-CE repository](https://github.com/jacobkrantz/VLN-CE)。

## 投稿前复核项

该领域在 2026 年更新很快。正式写入论文 comparison table 前需要重新确认：

1. arXiv 是否有新版数字或正式 conference 版本；
2. Val-Unseen 是否使用 monocular、panoramic、RGB-D、stereo 或 target prior；
3. method row 是单模型、ensemble、RL 后 checkpoint，还是最佳 validation run；
4. 是否公开 checkpoint，能否由 NAV 统一 Habitat build 复评；
5. Test leaderboard 与 Val-Unseen literature table 必须分开。
