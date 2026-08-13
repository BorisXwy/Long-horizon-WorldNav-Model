# V1 Representation-only WAM 与 Video-Action Co-training 调研

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-RES-003` |
| 类型 | 外部调研 / 复现候选选择 |
| 状态 | Active / Reproduction Targets Selected |
| 更新时间 | 2026-08-11 |
| 职责 | 调研“训练时 video/world supervision、推理时 action-only 或 compact future representation”的 WAM 路线，并确定第一批复现目标 |

## 当前结论

近期 WAM（World-Action Model）论文正在从早期的 **imagine-then-act** 转向
**representation-only / action-centered**：视频生成或未来视觉建模主要作为训练时
监督，用来塑造物理/时空表征；推理时不再显式生成 dense future video，而是直接从
compact observation / latent / action-query tokens 解码动作。这条路线与 NAV 的矛盾
完全同构：Stage One 需要 dense video generation 训练 world representation，Stage Three
需要 small input policy 做实时闭环。

第一批建议复现两个目标：

1. **Fast-WAM**：最直接回答“是否需要 test-time future imagination”。训练时保留
   video co-training，推理时跳过未来视频生成，直接 action generation。官方代码可用，
   LIBERO/RoboTwin 入口清晰。
2. **GigaWorld-Policy / GigaWorld-Policy-0.5**：action-centered WAM。训练时 future
   visual dynamics 监督 action representation，推理时 action-only decoding；其
   Mixture-of-Transformers / expert routing 对 NAV 的“video branch 与 policy branch
   共 backbone 但低延迟解耦”最有结构启发。

UVA 作为第三路线的思想先驱必须写入 related work，但复现优先级排在第二梯队：
它的“joint video-action latent + decoupled decoding”最干净，但主要是 2025 RSS 工作，
和我们当前需要的 Wan/DreamZero 时代 WAM 复现链路距离更远。

## 论文脉络

### 1. 早期 imagine-then-act：显式视频计划驱动动作

代表包括 UniPi、Video Policy / Video Generators are Robot Policies、Video Prediction
Policy 等。核心范式是：

```text
observation + instruction
    -> video generation / future visual plan
    -> inverse dynamics or action decoder
    -> robot action
```

优点是 future video 解释性强，视频数据可作为通用训练信号；缺点是推理时必须
执行 dense future generation，延迟高，且 action accuracy 受 future video 质量牵制。
这一路线证明了“视频模型可提供 policy representation”，但不是 NAV Stage Three
可直接采用的实时路径。

### 2. Joint WAM：video token 与 action token 在同一 DiT 中联合 denoise

代表包括 DreamZero、Unified World Model（UWM）等。核心范式是：

```text
[noisy video tokens | noisy action tokens | state tokens]
        + text / image / timestep conditions
            -> shared DiT / multimodal diffusion transformer
            -> video prediction head + action prediction head
```

DreamZero 的实现已经在本项目中复现和记录：它把 action/state register 拼入 Wan
DiT 主序列，DiT 同时输出 `video_noise_pred` 与 `action_noise_pred`。UWM 则进一步
把 video/action 作为 multimodal diffusion variables，并允许 separate diffusion timesteps，
从而统一 policy、forward dynamics、inverse dynamics 与 video prediction。

这一类模型共享最彻底，但推理若保留 dense video tokens 与多步 denoising，仍然慢。
因此后续工作开始把“video modeling 的训练收益”与“test-time 显式生成”拆开。

### 3. Representation-only / action-centered WAM：训练有视频，推理少生成或不生成

这一族是 NAV 当前最该借鉴的主线。

#### UVA（Unified Video Action Model, RSS 2025）

- 论文：https://arxiv.org/abs/2503.00200
- 项目页：https://unified-video-action-model.github.io/

UVA 的核心是 **joint video-action latent representation** 与 **decoupled
video-action decoding**。它通过统一 latent 表示桥接视频与动作，但输出端使用两个轻量
diffusion heads 解耦 video/action decoding；推理时可跳过 video generation 直接做 action
inference。项目页也明确指出：video generation 作为训练监督可提升 policy，而不降低
policy inference speed。

对 NAV 的启发：我们可以让 VideoGen Mode 与 Policy Mode 共享 Register / selected DiT
hidden interface，但使用不同 input adapter 和 output head。

#### Fast-WAM（2026）

- 论文：https://arxiv.org/abs/2603.16666
- 项目页：https://yuantianyuan01.github.io/FastWAM/
- 代码：https://github.com/yuantianyuan01/FastWAM

Fast-WAM 直接提出核心问题：WAM 的收益来自 test-time future imagination，还是来自
training-time video modeling？它保留 video co-training，但在推理时去掉显式 future
prediction，直接从 latent world representation 生成动作。官方项目页报告：

- backbone：Wan2.2-5B video DiT + 1B action expert；
- 训练：joint action prediction + video modeling；
- 推理：只保留 current observation clean latent tokens，video backbone 单次处理后
  直接 action generation；
- 延迟：单 RTX 5090D V2 32GB 上约 190ms；
- 关键消融：去掉 video co-training 比去掉 test-time imagination 造成更大性能下降。

对 NAV 的启发：Stage Three 不应每步生成完整未来视频；video generation 更适合作为
Stage One/Two 的 representation-shaping supervision，然后 policy 读取 compact Register
和小型 action-query tokens。

#### GigaWorld-Policy / GigaWorld-Policy-0.5（2026）

- 论文：https://arxiv.org/abs/2603.17240
- 项目页：https://gigaai-research.github.io/GigaWorld-Policy/
- 代码：https://github.com/open-gigaai/giga-world-policy

GigaWorld-Policy 指出 Joint WAM 的两个瓶颈：future visual dynamics 与 action joint
reasoning 推理昂贵；visual/motion 表示纠缠使 action accuracy 依赖 future video quality。
它转向 **action-centered WAM**：

```text
current observation -> predict future action sequence
predicted action + same observation -> generate future video as training supervision
inference -> action-only decoding, optional video generation closed
```

公开 repo 的 GigaWorld-Policy-0.5 进一步采用 mixed AC-WM + WAM pretraining，并引入
Mixture-of-Transformers，把 visual dynamics modeling 与 action generation 分给专门
experts，action-only inference 时减少 active computation；README 报告 RTX 4090 本地
约 85ms latency。

对 NAV 的启发：Policy Mode 应以 action/query 为中心；VideoGen branch 可以作为训练时
约束与可选分析输出，而不是每次导航决策的必经路径。

#### ImageWAM（2026）

- 论文：https://arxiv.org/abs/2606.19531
- 项目页：https://zhangwenyao1.github.io/ImageWAM/
- 代码：https://github.com/yuyangalin/ImageWAM

ImageWAM 进一步质疑 video generation 本身是否必要。它指出 video-based WAM 有三类
开销/风险：dense multi-frame future tokens 昂贵；full video prediction 会消耗容量于
action-irrelevant appearance details；长程 imagination 错误会误导 action。它改用 image
editing foundation model，建模 source-grounded target-frame transformation，使表示更关注
action-relevant visual changes。

对 NAV 的启发：即便我们保留 video generation 训练，policy 端也应主动压缩/筛选
action-relevant tokens，避免把容量浪费在导航无关的像素细节上。

#### Faster-WAM / Faster-WAM-family（2026）

- Faster-WAM efficient future conditioning：https://arxiv.org/abs/2608.04404
- Faster-WAM deep action module critique：https://arxiv.org/abs/2608.02365

这组最新工作针对 Fast-WAM 后的 trade-off：完全移除 test-time future representation
虽然快，但可能损失 OOD generalization；保留 joint future-action interaction 又慢。它们
提出 sparse future conditioning、SparseMoT、Interval KV-Fusion 等机制，在少数层/少数
阶段做 video-action interaction，避免每层深融合。

对 NAV 的启发：Stage Three 可考虑“稀疏读取 selected 3D/Video hidden layers”，而不是
每层都让 policy/action tokens 与 dense video tokens 深度交互。

## 复现目标选择

### 目标一：Fast-WAM

| 项 | 内容 |
| --- | --- |
| 选择理由 | 最直接验证“video co-training 训练有用、test-time future generation 可去掉” |
| 代码成熟度 | 官方 repo，README 包含 environment、model preparation、dataset、inference、training |
| 复现 benchmark | 优先 LIBERO smoke；条件允许再 RoboTwin |
| 关键复现项 | released checkpoint inference latency；video co-training vs w/o video co-training 消融若权重可得 |
| 与 NAV 对应 | 对应 NAV Stage Three 的 compact policy mode；验证不生成未来视频也能保留 WAM 收益 |

建议本地路径：

```text
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/FastWAM
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_fastwam
/sharedata/FastWAM
NAV/result/fastwam/
NAV/log/fastwam/
```

第一阶段只做：

```text
clone -> env -> download released checkpoint / minimal benchmark assets
-> single-task inference smoke -> latency logging
```

### 目标二：GigaWorld-Policy

| 项 | 内容 |
| --- | --- |
| 选择理由 | action-centered WAM，与 NAV 的“policy 小输入 + video branch 训练约束”最接近 |
| 代码成熟度 | 官方 repo 已公开，安装入口清晰；包含 GigaWorld-Policy-0.5 描述 |
| 复现 benchmark | 优先官方最小推理或 RoboTwin smoke；再看是否能复现 85ms latency |
| 关键复现项 | action-only inference latency；Mixture-of-Transformers / expert routing 是否真的绕开 video branch |
| 与 NAV 对应 | 对应 NAV 的 mode-specific adapter + shared hidden interface 设计 |

建议本地路径：

```text
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/giga-world-policy
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_gigaworld_policy
/sharedata/GigaWorld-Policy
NAV/result/gigaworld_policy/
NAV/log/gigaworld_policy/
```

第一阶段只做：

```text
clone -> env -> inspect model config / checkpoint availability
-> minimal inference or model-forward smoke
-> confirm action-only path vs optional video path
```

## 第二梯队

| 工作 | 暂不作为第一批复现的原因 | 后续用途 |
| --- | --- | --- |
| UVA | 思想非常干净，但 2025 RSS、与当前 Wan/DreamZero 系路线距离更远 | 写 related work；若 Fast-WAM/GigaWorld 复现受阻，转为候选 |
| ImageWAM | 更像“用 image editing 替代 video generation”的反命题，不是严格 video-gen co-training | 作为 action-relevant visual transformation 对照 |
| Faster-WAM | 2026-08 很新，代码/权重状态需进一步确认 | 后续用于设计 sparse layer fusion |

## 对 NAV Backbone/Input 设计的归纳

近期工作给出的共同答案是：

```text
不要让 policy 和 video generation 强行共用同一种 dense video input。
应该共用 latent/world representation、memory interface 与 selected backbone layers，
但使用 mode-specific input adapters 和 output heads。
```

对应 NAV：

```text
VideoGen Mode:
  dense noisy future latent
  + local memory
  + Register
  + text/action/timestep condition
  -> shared DiT / video branch
  -> video flow/noise head

Policy Mode:
  compact current obs tokens
  + Register
  + action/query tokens
  + instruction/goal condition
  -> selected shared DiT layers or policy adapter
  -> action / waypoint / subgoal head
```

训练时保留 video generation 与 3D supervision；推理时优先 policy-only，不显式生成
dense future video。Register 是二者的共同 memory interface，selected DiT hidden layer
是二者共享的 geometry/world representation interface。

## 2026 主会 / Workshop 算法趋势补充（2026-08-11）

本节只记录算法趋势，不替代 `NAV/essay/relatedworks/related_works_survey.md`
中的论文写作表述。写论文时必须继续区分 main conference、workshop、arXiv
technical report 和第三方列表待核条目。

### 趋势一：WAM 的争论焦点从“能否生成未来”转向“未来建模怎样服务动作”

2026 的 WAM / robot world-model 论文里，最接近 NAV 的共同问题是：

```text
future visual dynamics 是训练信号，还是推理时必经中间变量？
```

GigaWorld-Policy、Fast-WAM、mimic-video、World Action Models are Zero-shot
Policies 和 RSS 2026 的 Interactive World Simulator 给出的答案并不完全相同：

- GigaWorld / Fast-WAM：训练时保留 future visual dynamics，推理时 action-only；
- mimic-video：冻结视频骨干，在潜空间规划，再用轻量 IDM / action decoder 转动作；
- Interactive World Simulator：把 action-conditioned video world model 作为数据引擎
  与评测代理，而非直接把 video branch 嵌入 policy；
- World Action Models are Zero-shot Policies：强调 video/world-action 表征本身可
  直接产生可执行动作，但当前证据多在机器人操作域。

对 NAV 的直接约束：Stage Three 不应把 dense future video generation 作为导航闭环
必经路径；future video / 3D consequence 更适合作为 Stage One/Two 的
representation-shaping supervision。

### 趋势二：长程 interactive video world model 竞争点集中在 memory、cache 与几何一致性

ICLR/ICML/CVPR 2026 相关工作（Astra、Vid2World、LIVE、WorldPlay、GenieDrive、
VGGT-Ω、RELIC 等）把竞争点集中在：

- 自回归 / autoregressive denoising 如何减小 train-test gap；
- memory / KV cache / compressed history 如何覆盖 long horizon；
- 几何或 4D occupancy 如何约束视频世界模型；
- 实时交互时如何兼顾 responsiveness 与 coherence。

对 NAV 的直接约束：NAV 的“长程”不应只用 target video horizon 变长来证明，而应
明确拆成：

```text
short consequence prediction:
  T_latent=4 / 13 RGB frames

long memory span:
  Register rollout over IW-equivalent 1/4/8/16 chunks
```

这使 NAV 的核心假设变成“predict short, remember long”，与纯长视频生成路线区分。

### 趋势三：VLN / navigation 正从端到端 VLA 走向显式中间结构

RSS 2026 的 OpenFrontier 和 SuperMap 代表另一条与 NAV 接近但不同的路线：

- OpenFrontier：把导航表述为 sparse subgoal identification and reaching，用
  visual-language grounded frontiers 作 semantic anchors，强调 training-free 和系统简单；
- SuperMap：用可查询 4D scene graph 作为 VLN / embodied AI 的统一空间记忆；
- AdaNav、DualVLN、FantasyVLN 等 2026 OpenReview/CVPR 线索则强调 MLLM reasoning、
  uncertainty-adaptive reasoning 或 fast-slow dual system。

对 NAV 的直接约束：related work 不能只说“现有 VLN 缺少 memory”。更准确的说法是：

```text
现有 VLN 正在引入显式 frontier、map、scene graph、uncertainty reasoning 和
fast-slow systems；NAV 的不同在于把 memory 来源从导航任务定制结构，换成由
action-conditioned world modeling 与 geometry supervision 训练出的 online Register。
```

### 趋势四：会议论文写法更强调单一科学命题，tech report 更强调系统路线图

VGGT-Ω（CVPR 2026 Oral）是最值得 NAV 学的会议论文写法：它围绕
“feed-forward reconstruction 如何 scale”组织 Method，把 architecture、loss、
dynamic representation 和 self-supervised training 都放在同一命题下。

GigaWorld-Policy、GigaWorld-Policy-0.5、Qwen-RobotWorld、Infinite-World 等更像
tech report / model report：可以展开讲数据、pretraining pipeline、系统速度与 gallery。
NAV 若按会议论文写，应避免 tech-report 式堆模块，而应围绕一个核心命题组织：

```text
How can a streaming world-model memory become a navigation memory?
```

## 相关资源

- DreamZero 本地复现文档：`NAV/doc/02_architecture/v1_dreamzero_wan22_5b_backbone.md`
- BridgeVLA++ memory 复现：`NAV/doc/02_architecture/v1_bridgevla_plus_memory_architecture.md`
- NAV Stage One：`NAV/doc/01_design/v0_stage_one_register_world_model.md`
- NAV Stage Two：`NAV/doc/01_design/v0_stage_two_3d_supervision.md`
- NAV Stage Three：`NAV/doc/01_design/v0_stage_three_navigation.md`
