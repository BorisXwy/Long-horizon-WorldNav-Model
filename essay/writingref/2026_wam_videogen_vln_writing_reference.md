# 2026 WAM / VideoGen / VLN 写法参考

本文记录截至 2026-08-11 对 ICML 2026、CVPR 2026、ICLR 2026、RSS 2026
相关 WAM、video generation/world model、VLN/navigation 论文的写作观察。
本文件只服务论文写作结构，不承担算法事实唯一来源；算法类调研仍以
`NAV/doc/07_research/` 与 `NAV/essay/relatedworks/related_works_survey.md`
为准。

## 结论

如果 NAV 目标是会议论文写法，应优先学习 **VGGT-Ω** 这种“单一科学命题
驱动 Method”的组织方式，而不是直接学习 GigaWorld / Infinite-World / Qwen-RobotWorld
这类 tech report 的大系统铺陈。

NAV 的 Method 主线建议压成一个问题：

```text
How can a streaming world-model memory become a navigation memory?
```

也就是：

```text
长期导航需要 memory
  -> memory 不能只是帧缓存
  -> 用 action-conditioned consequence prediction 学动态
  -> 用 geometry supervision 使 memory spatial-readable
  -> policy 推理时 action-only 读取 memory
```

## 1. CVPR 2026：VGGT-Ω 的会议论文写法

状态：CVPR 2026 Oral / Best Paper Finalist。

VGGT-Ω 的 Method 组织非常紧，实际结构是：

```text
3 Method
  3.1 A New Scalable Architecture
      Feature Extraction and Tokenization
      Register Attention
      Decoding
        Depth
        Camera

  3.2 Training Losses
      Camera loss
      Depth loss
      Point loss
      Matching loss

  3.3 Dynamic Reconstruction
  3.4 Self-supervised Training
```

它不是按“DINO / Register / Head / Data”机械列模块，而是围绕
“feed-forward spatial reconstruction 如何 scale”推进：

| Intro 中的问题 | Method 对应小节 |
| --- | --- |
| global attention 和 high-resolution dense heads 太贵 | scalable architecture、register attention、lightweight decoding |
| 减少 prediction heads 后不能损失几何能力 | multi-task losses：camera / depth / point / matching |
| 静态重建不够，动态视频数据更多但更难 | dynamic reconstruction 中解释输出表示选择 |
| 标注数据不足以继续 scale | self-supervised teacher-student training |

对 NAV 的启发：

- Method 小节名应对应 intro 问题，而不是对应代码模块；
- Register 不能只作为“模块名”出现，必须说明它解决了哪个瓶颈；
- loss 小节要解释为什么这些监督会塑造目标表示；
- 如果写 conference paper，Data / Experiments 可以独立成节，不要把所有工程流水线塞进 Method。

## 2. GigaWorld-Policy：action-centered tech-report / arXiv 写法

状态：GigaWorld-Policy 原版为 arXiv preprint；GigaWorld-Policy-0.5 明确是
technical report。

GigaWorld-Policy 的 Method 小节很少：

```text
3 Method
  3.1 Problem Statement and Approach Overview
  3.2 The Architecture of GigaWorld-Policy
  3.3 GigaWorld-Policy: Training
```

3.2 内部用段落讲：

```text
Input Tokens
Shared Transformer Blocks
Causal Self-Attention for Video and Action Modeling
```

它的写法主线是：

```text
joint WAM 推理慢、action/video 表示纠缠
  -> action-centered formulation
  -> causal mask 防止 future-video 泄漏给 action
  -> training: action loss + video loss
  -> inference: future video optional, action-only decoding
```

对 NAV 的启发：

- 可借鉴 “Problem Statement and Approach Overview” 这种第一小节；
- action-centered 必须先定义因果关系，再讲 token packing；
- 不要把 GigaWorld 的大数据/pretraining/report-style 组织照搬成 NAV 的会议论文主体；
- NAV 的 action-only 不是短程 robot action-only，而是 conditioned on online long-horizon Register。

## 3. ICLR 2026：interactive world model 写法

代表信号：Astra、Vid2World、World-In-World，以及 ICLR World Models workshop
中的 World Action Models are Zero-shot Policies 等。

这一类写法常见结构：

```text
Problem: pretrained video diffusion lacks interactivity / controllability / long-horizon stability
Method:
  convert video model to action-conditioned world model
  introduce autoregressive or causal denoising
  add action encoding / modality adapter
  train with action-conditioned video prediction
Experiments:
  controllability, long-horizon consistency, responsiveness, video metrics
```

写作特点：

- 常把 “interactive world model” 作为总问题；
- Method 重心在 action-conditioned generation 和 autoregressive denoising；
- evaluation 强调 video quality、action following、long horizon consistency；
- 如果是 workshop，论证可以更概念化，但不能作为主会同等级证据。

对 NAV 的启发：

- ICLR 这条线适合支撑 NAV Stage One 的世界模型背景；
- 但 NAV 不应把 contribution 写成“又一个 interactive video world model”；
- NAV 应强调 memory 被 policy 读取，而不是只支撑 generation。

## 4. ICML 2026：长程 memory / cache / interactive video 的写法

ICML 2026 相关资料中，LIVE、WorldPlay、Quant VideoGen、Addressable Memory
等条目常被列入 long-horizon interactive video / memory / cache 路线；部分条目
需要继续核对 main conference、workshop 或 arXiv 状态。

这一类写法常见主线是：

```text
long-horizon video generation fails due to memory / error accumulation / compute
  -> introduce memory/cache/compression or geometric consistency mechanism
  -> train short, infer long / real-time interaction / horizon extension
  -> evaluate long video consistency and speed
```

对 NAV 的启发：

- NAV 可以吸收“memory/cost/horizon”问题设置；
- 但 NAV 的 horizon 主张应拆开：short future target + long Register span；
- 不能用 ICML/interactive-video 的 long-horizon video metrics 直接替代 navigation memory 证据。

## 5. RSS 2026：robotics paper 的系统与实机验证写法

RSS 2026 accepted papers 中，`World Models & Memory` session 与 NAV 很相关。
代表：

- Interactive World Simulator for Robot Policy Training and Evaluation；
- Self-Improving Robot Policy with Compositional World Model；
- Causal World Modeling for Robot Control；
- mimic-video；
- OpenFrontier；
- SuperMap。

RSS 写法常见主线：

```text
robotics deployment bottleneck
  -> compact formulation / system framework
  -> real-world data or real robot evaluation
  -> policy performance and correlation to real-world behavior
```

Interactive World Simulator 的摘要式组织尤其典型：

```text
action-conditioned video world models are promising but slow / unstable
  -> build an interactive world simulator
  -> use generated interaction data for policies
  -> evaluate both inside the world model and in the real world
  -> show correlation between world-model and real-world policy performance
```

OpenFrontier 的组织则是：

```text
end-to-end VLN/VLA requires training or fine-tuning
  -> formulate navigation as sparse subgoal identification and reaching
  -> use visual-language grounded frontiers as semantic anchors
  -> training-free, simple system, real robot deployment
```

对 NAV 的启发：

- RSS 写法重视 real-world / closed-loop / deployment evidence；
- 如果 NAV 论文投 robotics venue，Method 可以保留系统框架感；
- 如果投 CVPR/ICML/ICLR 主会，Method 应更收敛到核心 representation / memory claim。

## 对 NAV Method 的写法建议

当前最稳的 conference-style 结构不是“Stage One / Stage Two / Stage Three”平铺，
而是：

```text
3 Method
  3.1 Navigation Memory from Streaming World Models
      定义问题：world-model memory 如何成为 navigation memory。

  3.2 Online Predictive Register Memory
      解决长历史：Extractor / Updater / fixed budget / short consequence, long memory。

  3.3 Geometry-Readable Memory for Spatial Understanding
      解决“记忆不是表象缓存”：Register-after-DiT readout / 3D supervision。

  3.4 Action-Centered Policy Interface
      解决推理效率与因果性：A_hist / A_cur / A_query，future 不泄漏给 action。

  3.5 Training and Inference
      Stage One/Two/Three 作为 optimization schedule，而非 Method 主线。
```

如果需要更像 VGGT-Ω，可进一步压缩成：

```text
3 Method
  3.1 Online Register Architecture
  3.2 Predictive and Geometric Training Objectives
  3.3 Action-Only Navigation Inference
```

但后一版可能会牺牲可读性。建议第一版用于初稿，camera-ready 再压缩。

## 相关来源

- VGGT-Ω：CVPR 2026 Oral / Best Paper Finalist，arXiv: https://arxiv.org/abs/2605.15195
- GigaWorld-Policy：arXiv: https://arxiv.org/abs/2603.17240
- GigaWorld-Policy-0.5：technical report: https://arxiv.org/abs/2607.13960
- ICLR 2026 conference / OpenReview: https://iclr.cc/Conferences/2026
- RSS 2026 accepted papers: https://roboticsconference.org/program/papers/
- Interactive World Simulator: https://roboticsconference.org/program/papers/18/
- OpenFrontier: https://roboticsconference.org/program/papers/67/
