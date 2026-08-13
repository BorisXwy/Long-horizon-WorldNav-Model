# Related Works 调研汇总

本文件用于汇总 NAV 相关工作的调研，按四个研究方向组织。每篇论文按统一
模板记录：链接、官方 bibtex、问题、创新点、核心方法、训练/推理测试数据、
重要实验、对 NAV 的关系。新增论文追加到对应方向下，不删除已有条目；如需
废弃，标注 `Superseded` 并说明替代者。

## 研究方向

NAV 的相关工作分为以下四个方向：

1. **长流式视频生成模型**：chunk-by-chunk 滚动生成、固定预算长期记忆、
   超长视频世界模型。对应 NAV Stage One 的流式骨干与 Register Memory。
2. **3D / spatial-aware 世界模型**：在视频生成或世界模型中引入几何监督、
   depth/pose/pointmap、3D-aware 表示。对应 NAV Stage Two 的多层 probe 与
   3D 监督。
3. **VLN / VLA 的多帧历史问题**：视觉语言导航/动作中如何处理长观测历史、
   记忆压缩、exposure bias、history representation。对应 NAV Stage Three 的
   导航 history 接口。
4. **WAM 风格的导航模型**：world-model / world-action-model 风格的导航与
   控制策略，含 rollout、imagination、action-conditioned generation。对应
   NAV 整体"世界模型驱动导航"的定位与对比基线。

## 记录模板

每篇论文使用以下字段（无内容的写 `N/A` 或留空待补，不要编造）：

```markdown
### <序号>. <论文简称>（<年份，会议/期刊>）

- 链接：
  - 论文：<arXiv / 官方 URL>
  - 代码：<GitHub URL 或 N/A>
  - 项目页：<URL 或 N/A>
- 官方 bibtex：
  ```bibtex
  @inproceedings{...,
    ...
  }
  ```
- 问题：该论文要解决什么问题。
- 创新点：相对已有工作的核心新意。
- 核心方法：模型结构、训练目标、关键机制。
- 训练 / 推理测试数据：数据集、规模、划分；推理 benchmark 与指标。
- 重要实验：支撑结论的关键表格 / 图，及主要数字。
- 对 NAV 的关系：可借鉴 / 需对比 / 已超越 / 仅背景。
```

## 2026 会议信号快照（2026-08-11）

本节只记录与 NAV 写 related work 最接近的 2026 主会 / workshop / tech report
信号，避免把不同发表状态混为同一证据等级。完整论文条目仍按下方四个方向
维护；若某项只有项目页、workshop 或第三方列表佐证，在后续正式引用前必须
再核对论文 PDF / proceedings。

| venue / 状态 | 代表工作 | 与 NAV 的关系 | 写 related work 时的用法 |
| --- | --- | --- | --- |
| CVPR 2026 main | **VGGT-Ω**（CVPR 2026 Oral / Best Paper Finalist） | Register / scene tokens、register attention、多任务几何监督；证明 reconstruction 可作为 spatial understanding proxy | 放在 3D / spatial-aware 表征方向；强调 NAV 的区别是把几何可读 Register 改造成在线导航记忆 |
| CVPR 2026 main | **GenieDrive**：4D occupancy guided driving world model | driving world model 中显式 4D occupancy / geometry guidance 与 video generation 结合 | 仅作 geometry-guided video world model 背景；不是 VLN 直接对比 |
| CVPR 2026 main / program | **FantasyVLN**：VLN 中的 multimodal CoT / reasoning 路线 | 强调 VLN 正在引入 MLLM reasoning 与生成式中间过程 | 放在 VLN / VLA 多帧历史方向；与 NAV 的区别是 NAV 主打 world-model memory 而非语言推理链 |
| ICLR 2026 main | **Astra**：general interactive world model with autoregressive denoising | action-conditioned interactive world model；强调通用交互与自回归 denoising | 放在长流式视频生成方向；与 NAV 对比 memory 是否可被 policy 读取 |
| ICLR 2026 main | **Vid2World**：crafting video diffusion models to interactive world models | 将预训练 video diffusion 改造成 action-conditioned interactive world model | 放在长流式 / interactive video world model 方向；作为 NAV Stage One 的同类背景 |
| ICLR 2026 workshop | **World Action Models are Zero-shot Policies** | 强调 WAM 可作为零样本策略，但仍偏机器人操作和 video-to-action | 放在 WAM 风格导航/控制方向；证据等级标 workshop |
| ICML 2026 / OpenReview 或 workshop 待核 | **LIVE / WorldPlay / Quant VideoGen / Addressable Memory** 等长程 interactive video memory 工作 | 长程、实时、memory/KV/cache、几何一致性是 2026 视频世界模型主线 | 只作为“趋势快照”或待核条目；正式引用前需确认是否 main conference |
| RSS 2026 main | **Interactive World Simulator for Robot Policy Training and Evaluation** | action-conditioned video world model 支持 10min+、15FPS、政策训练与评测代理 | 放在 WAM/robot world simulator 方向；对 NAV 的 eval / data engine 设计有启发 |
| RSS 2026 main | **mimic-video** | 视频潜空间规划 + action decoder；video model 作为 compact predictive representation | 放在 representation-only WAM 方向；与 NAV policy-only 推理最接近 |
| RSS 2026 main | **OpenFrontier** | training-free visual-language grounded frontiers，导航被表述为 sparse subgoal identification | 放在 VLN / navigation 方向；与 NAV 对比：显式 frontier anchor vs learned world-model memory |
| RSS 2026 main | **SuperMap**（4D scene graph spatial memory） | 可查询 4D scene graph 作为 VLN / embodied AI 统一空间记忆 | 放在空间记忆方向；与 NAV 对比：显式 scene graph memory vs learned Register memory |

从 related work 写法上，当前最接近 NAV 的对比群不是单篇论文，而是三条线：

1. **长程交互视频世界模型**：Infinite-World、Astra、Vid2World、LIVE/WorldPlay、
   RELIC、LongLive、Rolling Forcing 等，回答“如何长时间稳定预测/生成”；
2. **几何可读空间表征**：VGGT-Ω、GenieDrive、Geometry/FantasyWorld 系列，
   回答“世界模型表示是否具有空间/几何结构”；
3. **WAM / policy-only 控制**：GigaWorld-Policy、Fast-WAM、mimic-video、
   World Action Models are Zero-shot Policies、Interactive World Simulator，
   回答“future modeling 如何转化为动作或策略，而非推理时必须生成视频”。

NAV 的 related work 主句应围绕这三条线的缺口组织：现有工作分别解决了
长程生成、几何理解或 action-only policy，但尚未把**在线长期记忆、几何监督和
VLN policy 读取接口**统一到同一个 Register memory 中。

## 论文清单

### 方向一：长流式视频生成模型

### 1. Infinite-World（2026，ICML 2026）

- 链接：
  - 论文：https://arxiv.org/abs/2602.02393
  - 代码：N/A（暂未公开）
  - 项目页：N/A
- 官方 bibtex：
  ```bibtex
  @article{wu2026infiniteworld,
    title={Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory},
    author={Wu, Ruiqi and He, Xuanhua and Cheng, Meng and Yang, Tianyu and Zhang, Yong and Kang, Zhuoliang and Cai, Xunliang and Wei, Xiaoming and Guo, Chunle and Li, Chongyi and Cheng, Ming-Ming},
    journal={arXiv preprint arXiv:2602.02393},
    year={2026}
  }
  ```
- 问题：交互式世界模型在长时序（数百至上千帧）生成中难以维持内容一致性、动态合理性与实时性，受限于有限的上下文窗口和误差累积。
- 创新点：提出 Pose-Free Hierarchical Memory（无姿态的分层记忆机制），将交互世界模型的生成时序扩展到 1000 帧级别，无需显式姿态输入即可维持长程一致性。
- 核心方法：分层记忆结构管理历史帧与状态信息；pose-free 设计避免对精确相机姿态的依赖；结合自回归视频扩散框架进行 chunk-by-chunk 流式生成。
- 训练 / 推理测试数据：N/A（论文页面未详细披露数据集与规模）。
- 重要实验：在 1000 帧时序 horizon 上验证了生成质量与一致性，相比已有交互世界模型在长时序稳定性上有显著提升。
- 对 NAV 的关系：可借鉴——其 pose-free 分层记忆与长 horizon 流式生成思路与 NAV 高度契合，是核心对比与借鉴对象。NAV 即从其原始 Infinite-World 初始化。

### 2. LingBot-World（2026，arXiv tech report）

- 链接：
  - 论文：https://arxiv.org/abs/2601.20540
  - 代码：https://github.com/Robbyant/lingbot-world
  - 项目页：https://technology.robbyant.com/lingbot-world
- 官方 bibtex：
  ```bibtex
  @article{lingbotworld2026,
    title={Advancing Open-source World Models},
    author={Robbyant Team and others},
    journal={arXiv preprint arXiv:2601.20540},
    year={2026}
  }
  ```
- 问题：开源世界模型在保真度、动态程度、长时序记忆与实时交互性上与闭源系统存在显著差距。
- 创新点：全面开源的世界模拟器框架，兼顾高保真、强动态、分钟级长时序记忆与亚秒级实时交互（16 fps 下延迟 <1s）。
- 核心方法：28B MoE 扩散 Transformer（继承 Wan2.2，双专家各约 14B）；动作通过相机嵌入与键盘适配器注入；后训练用因果注意力适配 + 分布匹配蒸馏把双向扩散转为自回归系统。
- 训练 / 推理测试数据：第一/第三人称网络视频、带控制的游戏录制、Unreal Engine 合成渲染；VBench 对比。
- 重要实验：唯一同时满足通用域+长程+高动态+720p+实时+开源；VBench 成像与美学质量均最高；Fast 版单 GPU 480p 下 16fps、延迟 <1s。
- 对 NAV 的关系：可借鉴 / 需对比——开源长时序交互世界模型的代表性工作，数据引擎、MoE 与实时交互设计可供参考。

### 3. Causal-Forcing（2026，arXiv preprint）

- 链接：
  - 论文：https://arxiv.org/abs/2602.02214
  - 代码：N/A
  - 项目页：https://thu-ml.github.io/CausalForcing.github.io
- 官方 bibtex：
  ```bibtex
  @article{zhu2026causal,
    title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
    author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
    journal={arXiv preprint arXiv:2602.02214},
    year={2026}
  }
  ```
- 问题：自回归视频扩散蒸馏中存在训练-测试不一致（exposure bias）与质量-速度难以兼顾的问题。
- 创新点：提出"正确"的自回归扩散蒸馏框架，兼顾高质量与实时交互，显著优于 Self-Forcing；支持 chunk-wise 与 frame-wise 模型，后者原生统一 T2V 与 I2V。
- 核心方法：因果注意力自回归结构支持 KV cache 加速；蒸馏目标对齐训练与推理的生成分布；面向实时交互的少步采样设计。
- 训练 / 推理测试数据：N/A（论文页面未详细披露）。
- 重要实验：在实时交互视频生成质量与延迟上优于已有自回归扩散蒸馏方法。
- 对 NAV 的关系：可借鉴——因果蒸馏与 KV cache 实时推理设计是 NAV 长流式实时生成的关键技术路径之一。

### 4. Self-Forcing（2025，arXiv preprint）

- 链接：
  - 论文：https://arxiv.org/abs/2506.08009
  - 代码：https://github.com/huangxun/Self-Forcing
  - 项目页：https://self-forcing.github.io
- 官方 bibtex：
  ```bibtex
  @article{huang2025selfforcing,
    title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
    author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
    journal={arXiv preprint arXiv:2506.08009},
    year={2025}
  }
  ```
- 问题：自回归视频扩散中 teacher forcing 训练与自回归推理之间存在 train-test gap（exposure bias），导致长视频生成时误差累积、质量退化。
- 创新点：让学生模型基于自身生成的历史进行下一帧生成，弥合训练-测试差距；单张 RTX 4090 即可实时流式生成。
- 核心方法：自回归扩散框架；训练中以 self-generated rollouts 作为条件，结合分布匹配蒸馏从短 horizon teacher 蒸馏到自回归 student；rolling KV cache 维持历史上下文。
- 训练 / 推理测试数据：基于短片段 teacher（如 5 秒）蒸馏；推理可外推到更长时序。
- 重要实验：相比 teacher forcing 基线，在长视频生成的时序一致性与质量上显著提升，减少误差累积。
- 对 NAV 的关系：可借鉴 / 需对比——Self-Forcing 是自回归视频扩散蒸馏的基础范式，NAV 的训练策略可直接借鉴并与之对比。

### 5. Wan2.2（2025，arXiv tech report）— 骨干网络背景

- 链接：
  - 论文：https://arxiv.org/abs/2503.20314
  - 代码：https://github.com/Wan-Video/Wan2.1
  - 项目页：N/A
- 官方 bibtex：
  ```bibtex
  @article{wan2025,
    title={Wan: Open and Advanced Large-Scale Video Generative Models},
    author={Team Wan and others},
    journal={arXiv preprint arXiv:2503.20314},
    year={2025}
  }
  ```
- 问题：开源视频基础模型在性能、全面性与消费级效率上难以同时兼顾，缺乏高质量、可扩展的视频生成基座。
- 创新点：基于 DiT 范式，提出新 VAE、可扩展预训练策略、大规模数据筛选；1.3B 模型仅需 8.19 GB VRAM，14B 模型在多项基准上超越开源与商用方案。
- 核心方法：DiT 架构；新型 3D 因果 VAE；多阶段可扩展预训练；支持文生视频、图生视频、视频编辑、个性化生成等多任务；首个支持中英视觉文字生成的模型。
- 训练 / 推理测试数据：数十亿级图像与视频数据预训练；在多个内部与外部 benchmark 上评测。
- 重要实验：14B 模型在多项 benchmark 上显著领先开源模型并超越商用方案；1.3B 模型在消费级 GPU 上高效运行。
- 对 NAV 的关系：仅背景——作为常用骨干网络（多个引用链新工作以 Wan2.1-1.3B 为基座进行长视频扩展），为 NAV 提供基础生成能力背景。

### 6. Self-Forcing++（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2510.02283
  - 代码：https://github.com/justincui03/Self-Forcing-Plus-Plus
  - 项目页：https://self-forcing-plus-plus.github.io/
- 官方 bibtex：
  ```bibtex
  @article{cui2025self,
    title={Self-Forcing++: Towards Minute-Scale High-Quality Video Generation},
    author={Cui, Justin and Wu, Jie and Li, Ming and Yang, Tao and Li, Xiaojie and Wang, Rui and Bai, Andrew and Ban, Yuanhao and Hsieh, Cho-Jui},
    journal={arXiv preprint arXiv:2510.02283},
    year={2025}
  }
  ```
- 问题：自回归视频扩散在超出 teacher 训练 horizon 后出现严重质量退化与误差累积，且长视频 teacher 数据难以获取。
- 创新点：无需长视频 teacher 监督或重训，仅用短 horizon teacher 知识引导学生；通过自生成长 rollout + 重新加噪 + 分布匹配蒸馏，让学生学会从退化状态恢复；引入 rolling KV cache 与 windowed sampling，指出 VBench 长视频评测偏差并提出 Visual Stability 指标。
- 核心方法：基于 Wan2.1-T2V-1.3B；backward noise initialization 重新加噪退化 rollout；windowed distribution-matching distillation 对齐 teacher；rolling KV cache 传递历史上下文（无 sink frames）；trunk size=3 分块生成。
- 训练 / 推理测试数据：短片段 teacher（5 秒）蒸馏；推理扩展到 100 秒（20× baseline），最大 4 分 15 秒（1023 latent frames，50× baseline）。
- 重要实验：生成时长 20×–50× 于 baseline，保真度与时序一致性显著优于已有方法；揭示 VBench 偏好过曝/退化帧的偏差并提出修正指标。
- 对 NAV 的关系：可借鉴 / 需对比——rolling KV cache、windowed 蒸馏与无 sink frame 设计是 NAV 长流式生成的直接可借鉴方案。

### 7. Rolling Forcing（2025，ICLR 2026）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2509.25161
  - 代码：https://github.com/TencentARC/RollingForcing
  - 项目页：https://kunhao-liu.github.io/Rolling_Forcing_Webpage/
- 官方 bibtex：
  ```bibtex
  @article{liu2025rolling,
    title={Rolling Forcing: Autoregressive Long Video Diffusion in Real Time},
    author={Liu, Kunhao and Hu, Wenbo and Xu, Jiale and Shan, Ying and Lu, Shijian},
    journal={arXiv preprint arXiv:2509.25161},
    year={2025}
  }
  ```
- 问题：流式视频生成在长 horizon 下存在严重误差累积，质量快速退化。
- 创新点：(1) Joint Denoising——rolling window 内同时去噪多帧且采用渐进递增噪声水平，放松相邻帧严格因果性，抑制误差增长；(2) Attention Sink——初始帧 KV 状态作为全局上下文锚点并动态调整 RoPE，增强长程全局一致性；(3) 非重叠窗口的少步蒸馏，缓解 exposure bias。
- 核心方法：自回归视频扩散 + rolling window 联合去噪 + attention sink 全局锚点 + 非重叠窗口蒸馏训练。
- 训练 / 推理测试数据：单 GPU 上实时 16 FPS 流式生成多分钟视频；与 Self Forcing 等对比。
- 重要实验：单 GPU 实时（16 FPS）多分钟流式生成，误差累积显著低于已有方法，质量与一致性优于可比开源模型。
- 对 NAV 的关系：可借鉴 / 需对比——rolling window 联合去噪与 attention sink 是 NAV 长流式生成的关键机制。

### 8. LongLive（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2509.22622
  - 代码：https://github.com/NVlabs/LongLive
  - 项目页：https://nvlabs.github.io/LongLive
- 官方 bibtex：
  ```bibtex
  @article{yang2025longlive,
    title={LongLive: Real-time Interactive Long Video Generation},
    author={Yang, Shuai and Huang, Wei and Chu, Ruihang and Xiao, Yicheng and Zhao, Yuyang and Wang, Xianbang and Li, Muyang and Xie, Enze and Chen, Yingcong and Lu, Yao and Han, Song and Chen, Yukang},
    journal={arXiv preprint arXiv:2509.22622},
    year={2025}
  }
  ```
- 问题：长视频生成在效率与质量上难以兼顾；因果 AR 模型在长视频训练中因记忆挑战质量退化；流式 prompt 输入的交互能力进一步增加一致性难度。
- 创新点：(1) KV-recache——新 prompt 到来时刷新缓存状态，实现平滑 prompt 切换；(2) Streaming Long Tuning——支持长视频训练并使训练-推理对齐（train-long–test-long）；(3) Short Window Attention + Frame-level Attention Sink，在保持长程一致性的同时加速生成。
- 核心方法：因果、帧级自回归设计；KV-recache 刷新缓存；streaming long tuning 长视频微调；短窗口注意力 + frame sink。基于 1.3B 短片段模型微调到分钟级生成，仅需 32 GPU-days；支持 INT8 量化推理。
- 训练 / 推理测试数据：VBench 评测短/长视频；单 H100 上 20.7 FPS（FP8 量化 24.8 FPS）；支持最长 240 秒视频。
- 重要实验：单 H100 上 20.7 FPS 实时生成，VBench 短/长视频表现均强；240s 长视频生成；INT8 量化仅轻微质量损失。
- 对 NAV 的关系：可借鉴 / 需对比——KV-recache、streaming long tuning 与 frame sink 是 NAV 流式长视频生成 + 交互 prompt 切换的直接可借鉴机制。

### 9. RELIC（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2512.04040
  - 代码：N/A
  - 项目页：https://relic-worldmodel.github.io/
- 官方 bibtex：
  ```bibtex
  @online{RelicWorldModel2025,
    author = {Hong, Yicong and Mei, Yiqun and Ge, Chongjian and Xu, Yiran and Zhou, Yang and Bi, Sai and Hold-Geoffroy, Yannick and Roberts, Mike and Fisher, Matthew and Shechtman, Eli and Sunkavalli, Kalyan and Liu, Feng and Li, Zhengqi and Tan, Hao},
    title = {RELIC: Interactive Video World Models with Long-Horizon Memory},
    url = {https://relic-worldmodel.github.io/},
    year = {2025}
  }
  ```
- 问题：真正可交互的世界模型需同时满足实时长程流式、一致的空间记忆与精确的用户控制，但多数方法只解决其中一项。
- 创新点：统一框架同时解决三难题——基于自回归视频扩散蒸馏，用高度压缩的历史 latent token（编码相对动作 + 绝对相机姿态）作为 KV cache 中的长程记忆，支持隐式 3D 一致性检索；提出 memory-efficient self-forcing 范式，实现长时序 teacher 与 student self-rollout 的全上下文蒸馏。
- 核心方法：14B 参数模型；压缩的 camera-aware 记忆 token 存于 KV cache；微调双向 teacher 生成超出原 5 秒训练 horizon 的序列，再转为因果 student；memory-efficient self-forcing 全上下文蒸馏。
- 训练 / 推理测试数据：在精选 Unreal Engine 渲染数据集上训练；单图 + 文本输入，实时 16 FPS 生成。
- 重要实验：实时 16 FPS 生成，在动作跟随准确性、长程流式稳定性、空间记忆检索鲁棒性上优于已有工作。
- 对 NAV 的关系：可借鉴 / 需对比——压缩 camera-aware KV cache 记忆与 memory-efficient self-forcing 蒸馏与 NAV 长流式 + 空间记忆目标高度一致，是核心对比与借鉴对象。RELIC 与 HPMC、NAV Register 同属"固定预算压缩记忆"族，但机制不同：RELIC 用滚动窗口 + 空间下采样（~4×）填预算、且记忆依赖绝对+相对 pose 注入 Q/K 做视角对齐检索；NAV 用递归更新填预算、pose-free（几何来自 Stage Two 监督，可单目预测）。关键区别：RELIC 在无可靠 pose（无里程计真机、纯视觉生成）下失效，NAV 不依赖 pose 故同时覆盖三类场景；RELIC 记忆只服务生成、不接下游决策，NAV Register 是导航接口。截至 2026-08 无公开代码/权重（仅项目页 https://relic-worldmodel.github.io/ 与网站源码仓），论文级参考，待官方发布后接入。

### 10. WorldMem（2025，NeurIPS 2025）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2504.12369
  - 代码：https://github.com/xizaoqu/worldmem
  - 项目页：https://xizaoqu.github.io/worldmem
- 官方 bibtex：
  ```bibtex
  @inproceedings{xiaoworldmem,
    title={WorldMem: Long-term Consistent World Simulation with Memory},
    author={Xiao, Zeqi and LAN, Yushi and Zhou, Yifan and Ouyang, Wenqi and Yang, Shuai and Zeng, Yanhong and Pan, Xingang},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
  }
  ```
- 问题：世界模拟中有限的时序上下文窗口导致难以维持长程一致性，尤其是 3D 空间一致性。
- 创新点：提出持续更新的外部 memory bank（memory frames + states（pose、timestamp）），通过 state-aware memory attention 检索相关历史帧，精确重建已观察场景；引入 timestamp 使框架不仅建模静态世界，还捕捉其动态演化。
- 核心方法：基于 Conditional Diffusion Transformer (CDiT) + Diffusion Forcing 训练支持自回归生成；memory frames 作为"clear" latents 与 noisy latents 共同参与去噪；state-aware cross-attention（当前帧为 query，memory 帧为 key/value）；Plücker embeddings 表示密集姿态。
- 训练 / 推理测试数据：定制 Minecraft benchmark 与 RealEstate10K；指标为 PSNR、LPIPS、FID。
- 重要实验：在超出上下文窗口的场景中准确重建已观察场景（Table 1，PSNR/LPIPS）；在动态环境中准确跟踪与跟随事件变化，验证感知与交互能力。
- 对 NAV 的关系：可借鉴——memory bank + state-aware memory attention 的显式记忆检索机制为 NAV 的长程空间一致性提供重要借鉴，与 Infinite-World 的分层记忆、RELIC 的压缩记忆形成不同路线对比。


### 1. GeometryForcing（2025，ICLR 2026）

- 链接：
  - 论文：https://arxiv.org/abs/2507.07982
  - 代码：https://github.com/MSR-RC/GeometryForcing
  - 项目页：https://geometryforcing.github.io/
- 官方 bibtex：
  ```bibtex
  @article{wu2025geometryforcing,
    title={Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling},
    author={Wu, Haoyu and Wu, Diankun and He, Tianyu and Guo, Junliang and Ye, Yang and Duan, Yueqi and Bian, Jiang},
    journal={arXiv preprint arXiv:2507.07982},
    year={2025}
  }
  ```
- 问题：视频扩散模型缺乏几何一致性，生成视频难以重建有意义的 3D 几何，导致时序与几何不一致。
- 创新点：提出 GF 范式，通过对齐 VGGT 的几何特征增强视频扩散模型，使学习到的特征能用于精确 3D 重建；相比 DFoT 生成更时序与几何一致的视频。
- 核心方法：将 VGGT 几何特征作为监督信号注入视频扩散训练，对齐特征空间；视频扩散骨干 + 几何对齐损失。
- 训练 / 推理测试数据：发布 reprojection error 与 revisit error 评测代码；具体数据集见论文。
- 重要实验：相比 DFoT 等基线，生成视频在时序与几何一致性上显著提升；GF-learned 特征可重建准确 3D 几何，而 baseline 特征无法重建。
- 对 NAV 的关系：可借鉴 / 需对比——其"视频扩散 + VGGT 几何监督"范式与 NAV Stage Two 的 3D 监督高度相关，是核心借鉴与对比对象。

### 2. FantasyWorld（2025，arXiv preprint）

- 链接：
  - 论文：https://arxiv.org/abs/2509.21657
  - 代码：N/A
  - 项目页：N/A
- 官方 bibtex：
  ```bibtex
  @article{fantasyworld2025,
    title={FantasyWorld: Distilling 3D Scene World Models for Actionable Video Generation},
    author={FantasyWorld Team},
    journal={arXiv preprint arXiv:2509.21657},
    year={2025}
  }
  ```
- 问题：视频生成模型缺乏 3D 空间感知，难以生成可导航、可交互的可控世界。
- 创新点：将 3D 场景世界模型蒸馏进视频生成，实现 actionable video generation；提供两个版本：严格复现论文配置的 480P 版与性能增强的 Wan2.2-Fun 控制版。
- 核心方法：以 3D 场景表示（首帧点云先验）作为条件蒸馏进 Wan 视频生成管线；Stage 1 几何骨干 + 视频生成联合。
- 训练 / 推理测试数据：FantasyWorld-Wan2.1-I2V-14B-480P（复现）与 FantasyWorld-Wan2.2-Fun-A14B-Control-Camera（性能）。
- 重要实验：在 3D 一致性与可控性上优于纯视频生成基线；首帧点云先验在视角超出先验范围时易退化。
- 对 NAV 的关系：可借鉴 / 需对比——其"3D 场景蒸馏进视频生成"是 NAV Stage Two 的直接对比路线；首帧点云先验的视野局限正是 NAV 多层 probe 要解决的问题。

### 3. VGGT-Ω（2026，arXiv preprint）

- 链接：
  - 论文：https://arxiv.org/abs/2605.15195
  - 代码：https://github.com/facebookresearch/vggt-omega
  - 项目页：http://vggt-omega.github.io/
- 官方 bibtex：
  ```bibtex
  @article{vggtomega2026,
    title={VGGT-Omega},
    author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Sch{\"o}nberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
    journal={arXiv preprint arXiv:2605.15195},
    year={2026}
  }
  ```
- 问题：前馈式 3D 重建需要更高质量、文本对齐与可扩展的几何基础模型。
- 创新点：VGGT 的升级版，提供 1B 参数 512 分辨率与 256 分辨率文本对齐两版；Register Token 机制处理长序列。
- 核心方法：前馈 transformer 架构，输入图像序列输出 depth/pose/pointmap；Register Token 压缩长序列信息。
- 训练 / 推理测试数据：HuggingFace 提供权重（需申请访问）；在线 demo 开放。
- 重要实验：在前馈 3D 重建质量与文本对齐上提升；Register Token 支持更长序列。
- 对 NAV 的关系：可借鉴——Register Token 机制是 NAV Register Memory 的直接灵感来源；作为 3D 监督的几何前端可借鉴。

### 4. StreamVGGT（2025，arXiv preprint）

- 链接：
  - 论文：https://arxiv.org/abs/2507.11539
  - 代码：https://github.com/lch01/StreamVGGT
  - 项目页：https://wzzheng.net/StreamVGGT
- 官方 bibtex：
  ```bibtex
  @article{zhuo2025streamvggt,
    title={Streaming 4D Visual Geometry Transformer},
    author={Zhuo, Dong and Zheng, Wenzhao and Guo, Jiahe and Wu, Yuqi and Zhou, Jie and Lu, Jiwen},
    journal={arXiv preprint arXiv:2507.11539},
    year={2025}
  }
  ```
- 问题：离线 3D 重建模型需对每张新图重新处理整段序列并重建整个场景，无法支持实时在线增量重建。
- 创新点：因果 transformer 架构，支持实时流式 4D 视觉几何感知，兼容 LLM 目标注意力机制（如 FlashAttention），兼顾快速推理与高质量 4D 重建。
- 核心方法：temporal causal attention + cached memory token 支持高效增量 on-the-fly 重建；从 VGGT 蒸馏得到流式版本。
- 训练 / 推理测试数据：从 VGGT 蒸馏；提供微调代码；HuggingFace demo 与 checkpoint。
- 重要实验：实时在线增量重建质量接近离线 VGGT，推理速度显著提升；支持长序列流式输入。
- 对 NAV 的关系：可借鉴——其因果流式 4D 重建 + cached memory token 是 NAV 流式几何感知的直接借鉴对象，与 Register Memory 思路呼应。

### 5. LingBot-Map（2026，arXiv preprint）

- 链接：
  - 论文：https://arxiv.org/abs/2604.14141
  - 代码：https://github.com/robbyant/lingbot-map
  - 项目页：https://technology.robbyant.com/lingbot-map
- 官方 bibtex：
  ```bibtex
  @article{lingbotmap2026,
    title={LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction},
    author={Robbyant Team},
    journal={arXiv preprint arXiv:2604.14141},
    year={2026}
  }
  ```
- 问题：流式 3D 重建需在单一框架内统一坐标 grounding、密集几何线索与长程漂移校正，并保持高效率。
- 创新点：Geometric Context Transformer，通过 anchor context、pose-reference window、trajectory memory 在单一流式框架内统一坐标 grounding、密集几何与长程漂移校正；feed-forward 架构 + paged KV cache attention，在 518×378 分辨率上对超 10000 帧长序列稳定 ~20 FPS 推理。
- 核心方法：anchor context（坐标 grounding）+ pose-reference window（密集几何）+ trajectory memory（长程漂移校正）；paged KV cache attention。
- 训练 / 推理测试数据：在多个 benchmark 上对比流式与迭代优化方法；指标为重建精度与 FPS。
- 重要实验：在多样 benchmark 上优于现有流式与迭代优化方法；超 10000 帧长序列 ~20 FPS 稳定推理。
- 对 NAV 的关系：可借鉴——其"anchor + pose window + trajectory memory"统一流式几何框架与 paged KV cache 是 NAV Stage Two 多层 probe 与流式 3D 监督的直接借鉴对象。

### 6. AETHER（2025，ICCV 2025 / RIWM Outstanding Paper）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2503.18945
  - 代码：https://github.com/InternRobotics/Aether
  - 项目页：https://aether-world.github.io/
- 官方 bibtex：
  ```bibtex
  @article{aether2025,
    title={Aether: Geometric-Aware Unified World Modeling},
    author={Aether Team and Zhu, Haoyi and Wang, Yifan and Zhou, Jianjun and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Shen, Chunhua and Pang, Jiangmiao and He, Tong},
    journal={arXiv preprint arXiv:2503.18945},
    year={2025}
  }
  ```
- 问题：几何重建与生成建模的整合仍是实现空间推理 AI 的关键挑战；现有世界模型多只预测视觉观测，忽略几何结构，且 4D 数据稀缺。
- 创新点：首次在统一框架内同时优化 4D 动态重建、动作条件视频预测、目标条件视觉规划三大能力，通过任务交错特征学习实现知识共享；完全在合成 4D 数据上训练即可零样本迁移到真实世界。
- 核心方法：基于 CogVideoX-5b-I2V 后训练；深度视频转为尺度不变归一化 disparity；相机轨迹编码为尺度不变 raymap 序列对齐 DiT 时空框架；鲁棒自动相机标注管线（动态掩码→视频切片→粗定位→CoTracker3+bundle adjustment 精修）；随机组合输入输出模态实现多任务联合优化。
- 训练 / 推理测试数据：合成 RGB-D 视频（DA-V、TheMatrix）；零样本评估于 MovieGen/Veo2 生成视频的 4D 重建、真实图像的动作跟随与视觉规划。
- 重要实验：真实世界零样本重建可比甚至优于领域专用模型；动作跟随与重建均实现零样本泛化；相机轨迹作为几何知情动作空间支持条件预测与规划。
- 对 NAV 的关系：可借鉴——其"重建+预测+规划"统一范式与"合成数据后训练"思路是 NAV 世界模型的直接参照；是 GeometryForcing/FantasyWorld 的核心对比 baseline。

### 7. Voyager（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2506.04225
  - 代码：https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
  - 项目页：https://voyager-world.github.io
- 官方 bibtex：
  ```bibtex
  @article{huang2025voyager,
    title={Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation},
    author={Huang, Tianyu and Zheng, Wangguandong and Wang, Tengfei and Liu, Yuhao and Wang, Zhenwei and Wu, Junta and Jiang, Jie and Li, Hui and Lau, Rynson WH and Zuo, Wangmeng and Guo, Chunchao},
    journal={arXiv preprint arXiv:2506.04225},
    year={2025}
  }
  ```
- 问题：从单图生成长程、3D 一致、可探索的 3D 场景仍困难；现有方法缺乏显式 3D 结构 grounding，长程空间不一致、视觉幻觉，且需事后 3D 重建耗时且引入伪影。
- 创新点：首个联合生成对齐 RGB 与深度序列的视频扩散模型，以相机轨迹为条件；引入可扩展 world cache（点云累积+point culling）与自回归平滑采样实现长程世界探索；提供可扩展数据引擎自动估计位姿与度量深度。
- 核心方法：world-consistent video diffusion 统一架构联合生成 RGB-D，以已有世界观测为条件保证全局一致；world cache 将生成帧反投影回 3D 并投影到目标视角给出 partial RGB-D 引导；point culling 实时剔除冗余点；自回归平滑采样扩展视频长度。
- 训练 / 推理测试数据：10 万+视频片段（真实采集 + Unreal Engine 渲染），自动估计相机位姿与度量深度。8 GPU 并行推理 6.69× 加速。
- 重要实验：联合深度建模使几何更一致，支持直接 3D 重建与无限世界扩展；视觉质量与几何精度明显优于现有方法。
- 对 NAV 的关系：可借鉴——其"RGB-D 联合生成 + world cache + 自回归扩展"是 NAV 可探索世界生成的直接参考；是 FantasyWorld 的核心对比 baseline。

### 8. DeepVerse（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2506.01103
  - 代码：https://github.com/SOTAMak1r/DeepVerse
  - 项目页：https://sotamak1r.github.io/deepverse/
- 官方 bibtex：
  ```bibtex
  @article{chen2025deepverse,
    title={DeepVerse: 4D Autoregressive Video Generation as a World Model},
    author={Chen, Junyi and Zhu, Haoyi and He, Xianglong and Wang, Yifan and Zhou, Jianjun and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Fu, Zhoujie and Pang, Jiangmiao and others},
    journal={arXiv preprint arXiv:2506.01103},
    year={2025}
  }
  ```
- 问题：现有交互式世界模型主要预测视觉观测，忽略几何结构与空间连贯等隐状态，导致误差快速累积、时间不一致（漂移与遗忘）。
- 创新点：首个自回归 4D 世界模型，在大规模合成数据（带精确空间标签）上训练，显式将前序几何预测（深度+相机位姿）纳入当前步的条件，从根本上缓解单模视觉范式的尺度模糊与漂移。
- 核心方法：自回归先验基于大规模真实视频学习动态模式，同时利用合成数据提供深度/位姿监督；每步预测不仅依赖前序 RGB 帧，还依赖前序几何估计；提出 geometry-aware memory read-and-write 机制，按空间重叠/结构相似度检索高相关历史作为条件输入，对抗遗忘。
- 训练 / 推理测试数据：大规模合成数据（精确深度+相机位姿标签）+ 大规模真实视频自回归先验。
- 重要实验：显式几何约束显著降低漂移、增强时间一致性，提升预测精度、视觉真实感与场景合理性；geometry-aware memory retrieval 有效保持长程空间一致。
- 对 NAV 的关系：可借鉴——其"4D 自回归 + 几何知情记忆检索"是 NAV 长程世界模型的重要参考；与 AETHER 同一团队，是其 4D 扩散世界模型的自回归演进。

### 9. π³（2025，ICLR 2026）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2507.13347
  - 代码：https://github.com/yyfz233/Pi3
  - 项目页：N/A
- 官方 bibtex：
  ```bibtex
  @misc{wang2025pi3,
    title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning},
    author={Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
    year={2025},
    eprint={2507.13347},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```
- 问题：现有前馈重建方法（含 VGGT）依赖固定参考视角，这一归纳偏置在参考帧不佳时导致不稳定与失败，限制鲁棒性与可扩展性。
- 创新点：首次系统识别并挑战固定参考视角偏置；提出完全置换等变架构，预测仿射不变相机位姿与尺度不变局部 point map，无需任何参考帧，对输入顺序天然鲁棒、高度可扩展。
- 核心方法：交替 view-wise 与全局自注意力（类似 VGGT），去除帧索引位置编码等顺序依赖组件；每帧预测相对自身相机坐标系的 affine-invariant pose 与 scale-invariant local pointmap；无全局坐标系。
- 训练 / 推理测试数据：支持单图、视频序列、无序图像集合（静态/动态场景）。评估于 Sintel 等多 benchmark，对比 VGGT、DUSt3R、MoGe。
- 重要实验：Sintel 上相机 ATE 从 VGGT 0.167 降至 0.074，尺度对齐视频深度 Abs Rel 0.299→0.233；推理 57.4 FPS（vs DUSt3R 1.25、VGGT 43.2）；随模型增大性能持续提升、收敛显著加快。
- 对 NAV 的关系：可借鉴——作为 VGGT 系列的置换等变后续，可作为 NAV 无参考帧偏置的几何感知前端；其 scaling law 与鲁棒性对 NAV 大规模空间感知有参考价值。

### 10. WorldExplorer（2025，SIGGRAPH Asia 2025）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2506.01799
  - 代码：N/A
  - 项目页：https://the-world-explorer.github.io
- 官方 bibtex：
  ```bibtex
  @inproceedings{schneider2025worldexplorer,
    author={Schneider, Manuel-Andreas and H{\"o}llein, Lukas and Nie{\ss}ner, Matthias},
    title={{WorldExplorer}: Towards Generating Fully Navigable {3D} Scenes},
    booktitle={Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
    year={2025},
    publisher={Association for Computing Machinery}
  }
  ```
- 问题：从文本生成 3D 世界时，现有方法可探索程度有限——超出中心或全景视角即出现拉伸、噪声伪影，无法支持真正无限制的探索。
- 创新点：首个支持高质量、全视角可探索的文本生成 3D 场景方法；以自回归视频轨迹生成迭代扩展场景，配合 scene memory 与 collision detection，首次实现真实且无限制的探索。
- 核心方法：先用多图构建 360° 全景（深度估计 + T2I inpainting）作为场景支架；再以视频扩散模型沿预定义短轨迹迭代生成多段视频探索场景纵深；scene memory 从已有视图选最相关先验作为条件；collision detection 动态调整轨迹长度避免穿墙；最后用 3DGS 优化融合所有视图，VGGT 估计点云初始化。
- 训练 / 推理测试数据：基于现有相机引导视频扩散模型，无需额外训练；评估为定性场景探索对比与 3DGS 重建质量。
- 重要实验：相比先前方法，在大相机运动下场景保持稳定、无拉伸/失真，首次实现真实无限制探索。
- 对 NAV 的关系：可借鉴——其"全景初始化 + 迭代视频轨迹扩展 + scene memory + 碰撞检测 + 3DGS 融合"是 NAV 可探索场景生成的工程范式参考；偏场景生成而非空间感知建模，仅背景。


### 1. NavDP（2025，ICRA 2026）

- 链接：
  - 论文：https://arxiv.org/abs/2505.08712
  - 代码：https://github.com/OpenRobotLab/NavDP
  - 项目页：https://wzcai99.github.io/navigation-diffusion-policy.github.io/
- 官方 bibtex：
  ```bibtex
  @misc{navdp2025,
    title={NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
    author={Cai, Wenzhe and Peng, Jiaqi and Yang, Yuqiang and Zhang, Yujian and Wei, Meng and Wang, Hanqing and Chen, Yilun and Wang, Tai and Pang, Jiangmiao},
    year={2025},
    booktitle={arXiv}
  }
  ```
- 问题：端到端视觉导航在动态开放世界中难以同时实现跨本体泛化与零样本 sim-to-real 迁移；模块级方法存在系统延迟与复合误差，学习方法受限于真实数据稀缺、缺乏负样本/交互反馈。
- 创新点：首次将扩散策略与 RL 式 critic 价值函数统一进单一 transformer，完全用仿真数据训练即可零样本迁移到多类机器人；利用仿真特权信息（全局最优规划器 + ESDF）分别监督 actor 与 critic，实现反事实推理与安全行为区分。
- 核心方法：多模态编码器融合 N=8 帧 RGB 历史 + 单帧深度，经轻量 transformer decoder 压缩为 N×16 token；统一 Policy Transformer 同时做扩散轨迹生成（DDPM 去噪，预测 24 个 waypoint）与轨迹评估（critic 打分），两任务共享权重、仅 query 与 attention mask 不同；critic 标签由 ESDF 增量与碰撞阈值构造；推理先批量生成候选轨迹再用 critic 选最优。
- 训练 / 推理测试数据：自建数据引擎 2500 轨迹/GPU/天，3000+ 场景，最终 200K+ 轨迹、1M+ 米、40M 图像；32×A100 训练。推理 benchmark：IsaacSim PointGoal（20 场景 2000 episode）/ NoGoal（10 场景 1000 episode），真实世界 Turtlebot4、Unitree Go2/G1、Galaxea R1。指标：SR、SPL；Time、Area。
- 重要实验：Table III PointGoal 仿真 SR 67.2/SPL 62.6，较 ViPlanner +6.3/+4.0；真实平均 SR 76.7，较 ViPlanner +23.4。Table IV NoGoal 仿真 Time 106.2/Area 274.1，约 NoMaD 的 2.9×/3.1×。Table V 消融：去深度 SR 降 10.3，去 RGB 降 5.1，去多帧 RGB 降 2.8，去 critic 降 7.8。Figure 6 跨本体消融：缺跨本体数据时 Galaxea R1 SR 从 90% 跌到 20%。
- 对 NAV 的关系：可借鉴（多帧 RGB 历史编码、critic 选择、仿真数据引擎）+ 需对比（PointGoal/NoGoal 真实跨本体基线）；其作者团队后续 StreamVLN/InternVLA-N1 已在 VLN-CE 上超越，故在 VLN 多帧历史方向上属"已被后续工作扩展"的背景与基线。

### 2. LDA-1B（2026，RSS 2026）

- 链接：
  - 论文：https://arxiv.org/abs/2602.12215
  - 代码：https://github.com/jiangranlv/LDA-1B
  - 项目页：https://pku-epic.github.io/LDA/
- 官方 bibtex：
  ```bibtex
  @article{lyu2026lda,
    title={LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion},
    author={Lyu, Jiangran and Liu, Kai and Zhang, Xuheng and Liao, Haoran and Feng, Yusen and Zhu, Wenxuan and Shen, Tingrui and Chen, Jiayi and Zhang, Jiazhao and Dong, Yifei and others},
    journal={arXiv preprint arXiv:2602.12215},
    year={2026}
  }
  ```
- 问题：现有机器人基础模型以行为克隆为主，只能用高质量专家数据，丢弃了异构具身数据中可迁移的动力学知识；Unified World Model 虽能联合建模动力学/策略/视频生成，但存在数据使用粗放、数据集碎片化、像素空间预测耦合冗余外观等限制，难以扩展到基础模型级。
- 创新点：提出"通用具身数据摄取"——按数据质量分配角色（无动作人类视频→视觉预测，低质量轨迹→动力学，高质量→策略+动力学）；构建并标准化 EI-30k（30k+ 小时统一格式数据集）；在结构化 DINO 潜空间而非像素/VAE 空间做预测，避免冗余外观建模；用多模态扩散 transformer（MM-DiT）处理异步视觉/动作流，实现 1B 参数稳定训练。
- 核心方法：基于 UWM 的四目标联合训练（policy / forward dynamics / inverse dynamics / visual forecasting），用 4 个可学习 task embedding + 2 个 register token 在同一扩散模型内灵活切换；MM-DiT 对动作与视觉 token 做共享自注意力、模态专属 QKV/FFN，语言经 cross-attention 注入，AdaLN 注入 timestep/任务条件；动作统一为 hand-centric 末端执行器增量；视觉 3Hz、动作 10Hz 异步双流；流匹配目标。MM-DiT 条件于过去 2 步历史观测+动作以捕获时序动态。
- 训练 / 推理测试数据：EI-30k = 真实机器人 8.03k h + 仿真机器人 8.6k h + 带动作人类 7.2k h + 无动作人类 10k h；48×H800，400k iter，4608 GPU 小时。仿真 benchmark：RoboCasa-GR1（24 任务）。真实：Galbot G1、Unitree G1。
- 重要实验：Table II RoboCasa-GR1 平均 SR：LDA-1B 55.4 vs GR00T-EI10k 51.3 vs GR00T-N1.6 47.6 vs UWM 14.2；VAE→DINO 表示带来 20.0→55.4 的巨幅提升。Fig 6/7 真实：Clean Rubbish LDA 35% vs 基线 0%；Pull Nail 80% vs π0.5 0%。Table IV 混合质量微调：加 30% 低质量轨迹，π0.5 掉 20%，LDA 反升 10%。Fig 10 scaling：动作预测误差随数据/参数单调下降至 6.6，UWM 则饱和。
- 对 NAV 的关系：可借鉴——其"结构化 DINO 潜空间做动力学预测、多任务 co-training、混合质量数据利用、用短历史条件捕获时序"思路对 VLN/VLA 多帧历史表示有直接启发；但本体为操作而非导航，需对比/迁移到导航 benchmark。

### 3. Qwen-RobotNav（2026，arXiv tech report）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2606.18112
  - 代码：https://github.com/QwenLM/Qwen-RobotNav
  - 项目页：https://www.alibabacloud.com/blog/qwen-robotnav
- 官方 bibtex：
  ```bibtex
  @misc{qwenrobotnav2026,
    title={Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System},
    author={Qwen Team},
    year={2026},
    eprint={2606.18112},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
  }
  ```
- 问题：多任务导航基础模型需在同一权重下统一 VLN/PointNav/ObjNav/Tracking 等，但不同任务对"看什么、记什么"的需求根本不同；现有统一模型把单一记忆假设写死在架构里，难以按任务/场景动态调整历史编码。
- 创新点：把多任务导航重新定义为"观测上下文建模"问题而非架构设计问题；提出参数化接口——任务模式 + 可控观测参数（视觉 token 预算、时间衰减、每相机权重），训练时对所有参数随机化，使模型从不固定在某一配置，推理时上层 planner 可动态切换而无需重训；相机身份/时序/本体全用自然语言标签与 prompt 前缀传达，对 Qwen3-VL 主干零结构改动。
- 核心方法：继承 Qwen3-VL 主干 + 4 层 MLP action head，输出 8 个 waypoint (x,y,θ)；观测协议把"记多少/怎么记"作为外部可控自由度；两层级系统：上层 planner（Qwen3.7-Plus）分解长程目标为子目标并切换任务模式/上下文策略，Qwen-RobotNav 作反应式 waypoint 预测器；与视觉-语言数据 co-training 防止退化为"反应式动作序列映射"。
- 训练 / 推理测试数据：15.6M 样本训练，2B/8B 两档并展示有利 scaling。benchmark：VLN-CE RxR、EVT-Bench（跟踪）、NAVSIM（自动驾驶 PDMS）、HM-EQA / EXPRESS-Bench（具身问答）。
- 重要实验：VLN-CE RxR 76.5% SR；EVT-Bench 90.0% tracking rate；NAVSIM 91.4 PDMS；agentic 系统在 HM-EQA 较前最优 +10.8%、EXPRESS-Bench +15.4%，且导航步数减少 77%。R2R-CE val unseen：4B 单相机 SR 66.9/SPL 60.5，8B（带深度）72.1/66.6。
- 对 NAV 的关系：可借鉴（把历史/上下文建模作为一阶可控接口、训练时参数随机化避免过拟合单一记忆策略）+ 需对比（VLN-CE/RxR-CE 当前强基线）。直接针对"多帧历史问题"——其观测协议本质就是可配置的多帧历史编码机制。

### 4. Robostral Navigate（2026，Mistral AI tech report）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2607.20785
  - 代码：N/A
  - 项目页：https://mistral.ai/news/robostral-navigate/
- 官方 bibtex：
  ```bibtex
  @misc{robostral2026,
    title={Robostral Navigate},
    author={Majumdar, Arjun and Sooriyarachchi, Avinash and Tibi, Benjamin and Bamford, Chris and Chane-Sane, Elliot and Lample, Guillaume and Chandu, Khyathi Raghavi and Ho Fuh, Ludovic and Poir{\'e}e, Mathieu and Millner, Rosalie and Mishra, Srijan and Cachet, Th{\'e}o and Chabal, Thomas},
    year={2026},
    eprint={2607.20785},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
  }
  ```
- 问题：导航系统大规模部署需最小化传感器假设、跨本体泛化、训练高效；当前最优系统依赖深度/LiDAR/多相机/预建图，限制了可部署硬件范围并增加成本与标定负担。
- 创新点：仅用单目 RGB 流，通过"pointing"（在当前相机视图里指向下一目标像素坐标 + 到达朝向）而非度量坐标输出 waypoint，使策略天然对相机内参与场景尺度鲁棒、跨轮式/腿式/空中本体免标定；提出 prefix-caching + tree attention mask 训练配方，把整条 episode 打包成单条训练序列，token 数降 22×、训练从月级降到天级；tree mask 阻止模型在训练时条件于历史 ground-truth 动作，强制视觉接地；再用在线 RL（CISPO）提升探索与恢复。
- 核心方法：8B VLM（从空间 grounding 模型初始化）取指令 + 单目 RGB 历史帧，0.5Hz 预测 waypoint（可见时 a_vis=(u,v,Δx,Δy,Δθ)，不可见时 a_invis=(Δx,Δy,Δθ)）+ STOP；下游 121M diffusion transformer 10Hz 生成 30 步动作 chunk，本体运动跟踪控制器转 100Hz 力矩；训练时机器人高度/半径/相机位姿全随机化以跨本体。
- 训练 / 推理测试数据：仿真生成 2.4M 轨迹、350k 场景；SFT 后用 CISPO 在 35k 困难子集上在线 RL。benchmark：R2R-CE、RxR-CE val unseen。指标：NE、OS、SR、SPL。
- 重要实验：Table 1 R2R-CE val unseen SR 77.4/SPL 74.2（单相机新 SOTA，超最佳单相机 Qwen-RobotNav-4B +10.5，超最佳深度/多相机 Qwen-RobotNav-8B +5.3）；RxR-CE SR 75.1/SPL 68.7。RL 阶段在 seen +3.47%、unseen +4.03%。
- 对 NAV 的关系：可借鉴（prefix-caching/tree mask 解决多帧历史训练的 exposure bias 与算力、pointing 跨本体表示、在线 RL 修复合成误差）+ 需对比（R2R-CE/RxR-CE 当前 SOTA）。其 tree attention mask 正是针对"多帧历史训练时模型偷看历史动作"这一 exposure bias 的直接解法。

### 5. StreamVLN（2025，ICRA 2026）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2507.05240
  - 代码：https://github.com/InternRobotics/StreamVLN
  - 项目页：https://streamvln.github.io/
- 官方 bibtex：
  ```bibtex
  @article{wei2025streamvln,
    title={StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
    author={Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and others},
    journal={arXiv preprint arXiv:2507.05240},
    year={2025}
  }
  ```
- 问题：基于 Video-LLM 的 VLN 在细粒度视觉理解、长程上下文建模与计算效率之间存在权衡；把整段视频塞进上下文成本高，截断又丢失长程历史。
- 创新点：提出慢-快（SlowFast）混合上下文建模，把"短程响应"与"长程记忆"解耦——快流用滑动窗口 KV cache 做多轮对话式动作生成，慢流用 3D-aware token 剪枝压缩历史视觉状态，使短 clip（16 帧）训练的模型能稳定处理长视频流且上下文长度/推理成本有界。
- 核心方法：基于 LLaVA-Video（Qwen2-7B + 视觉编码器）扩展为交错 vision-language-action 模型，自回归多轮对话输出动作符号（↑←→STOP）；快流：8 轮滑动窗口 + KV cache 复用，丢弃过期 action/prompt token；慢流：将 RGB-D 投影到 3D voxel，每 voxel 只保留最近 token，做空间/时间冗余剪枝。
- 训练 / 推理测试数据：导航数据 450K clip（R2R、R2R-EnvDrop、RxR、ScaleVLN）+ 240K DAgger 增强；通用多模态 248K 视频 VQA + 230K 交错图文。benchmark：R2R-CE、RxR-CE val unseen。指标：SR、SPL。
- 重要实验：R2R-CE val unseen SR 56.9 / SPL 51.9；RxR-CE val unseen SR 52.9 / SPL 46.0（RGB-only SOTA，被 Qwen-RobotNav/Robostral Navigate 后续超越）。效率：KV prefill 时间节省约 99%、token 减约 20%，实时低延迟。
- 对 NAV 的关系：可借鉴（慢-快上下文解耦、3D voxel token 剪枝、KV cache 复用）+ 需对比（VLN-CE RGB-only 基线）。是 NavDP 同团队后续在 VLN-CE 多帧历史方向上的直接演进。

### 6. NavFoM（2025，ICLR 2026）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2509.12129
  - 代码：N/A（接受后开源）
  - 项目页：https://pku-epic.github.io/NavFoM-Web/
- 官方 bibtex：
  ```bibtex
  @article{zhang2025embodied,
    title={Embodied navigation foundation model},
    author={Zhang, Jiazhao and Li, Anqi and Qi, Yunpeng and Li, Minghan and Liu, Jiahang and Wang, Shaoan and Liu, Haoran and Zhou, Gengze and Wu, Yuze and Li, Xingxing and others},
    journal={arXiv preprint arXiv:2509.12129},
    year={2025}
  }
  ```
- 问题：现有大 VLM 在具身导航上泛化仍局限于窄任务与特定本体架构，难以跨本体（四足/无人机/轮式/车）与跨任务（VLN/ObjNav/跟踪/自驾）统一。
- 创新点：首个跨本体×跨任务导航基础模型，8M 导航样本联合训练；引入 Temporal-Viewpoint Indicator（TVI）token 编码相机视角与任务时序上下文，统一处理不同相机配置与时间跨度；Budget-Aware Temporal Sampling（BATS）按指数遗忘曲线在有限 token 预算内优先近期观测，动态管理多帧历史成本。
- 核心方法：基于 Qwen2-7B + DINOv2 + SigLIP，扩展为双分支——导航分支输出 8 个归一化 waypoint (x,y,z,θ)，QA 分支自回归生成语言；TVI token 组织跨时刻/视角视觉 token 送入 LLM；3 层 MLP planning head 解码隐藏状态；轨迹 MSE 损失 + QA 交叉熵，L=β·L_nav + L_QA（β=10）。
- 训练 / 推理测试数据：12.7M 样本 = 8.02M 导航（VLN-CE R2R/RxR、OpenUAV、HM3D ObjectNav、EVT-Bench、nuScenes、Sekai Web-Video）+ 4.76M 开放世界 QA。7 个公开 benchmark + 真实世界实验，无需任务特定微调。
- 重要实验：在 7 个 benchmark 上达到 SOTA 或高度竞争性能（Robostral Navigate 表中 NavFoM R2R-CE val unseen SR 61.7/SPL 55.3，RxR-CE SR 64.4/SPL 56.2，使用深度/多相机）。真实世界实验验证泛化。
- 对 NAV 的关系：可借鉴（TVI token 统一多视角多帧历史、BATS 预算感知时间采样解决多帧历史成本）+ 需对比（跨本体跨任务导航基线）。其 BATS 的"指数遗忘 + 预算采样"正是多帧历史长度/成本权衡的一种范式，与 Qwen-RobotNav 的可控观测协议思路呼应。

### 7. LoGoPlanner（2025，arXiv preprint）— NavDP 同组后续

- 链接：
  - 论文：https://arxiv.org/abs/2512.19629
  - 代码：https://github.com/InternRobotics/NavDP/tree/master/baselines/logoplanner
  - 项目页：https://steinate.github.io/logoplanner.github.io/
- 官方 bibtex：
  ```bibtex
  @article{peng2025logoplanner,
    title={LoGoPlanner: Localization Grounded Navigation Policy with Metric-aware Visual Geometry},
    author={Peng, Jiaqi and Cai, Wenzhe and Yang, Yuqiang and Wang, Tai and Shen, Yuan and Pang, Jiangmiao},
    journal={arXiv preprint arXiv:2512.19629},
    year={2025}
  }
  ```
- 问题：现有端到端导航依赖独立定位模块、需精确传感器外参标定来做自状态估计，限制了跨本体与跨环境泛化。
- 创新点：提出 localization-grounded 端到端导航框架：(1) 微调长时序视觉几何骨干（Pi3）使预测 ground 到绝对度量尺度，实现隐式状态估计/定位；(2) 从历史观测重建周围场景几何，提供稠密细粒度环境感知以可靠避障；(3) 策略以上述辅助任务 bootstrap 出的 implicit geometry 为条件，减少误差传播、提升鲁棒性。
- 核心方法：以 Pi3 视觉几何骨干为基础微调得到 metric-aware 几何表示；从历史观测重建场景几何作为稠密环境感知；策略条件于 implicit geometry 而非显式定位输出，避免定位-规划级联的误差累积。PointNav 任务，支持仿真与 LeKiwi 真机部署。
- 训练 / 推理测试数据：复用 NavDP 环境与 InternData-N1 数据；仿真 IsaacSim PointNav（scenes_home、cluttered_hard 等），真机 LeKiwi（RGBD，无外部里程计，靠 implicit localization 到达目标）。
- 重要实验：在 NavDP benchmark 上作为强基线对比；真机无外部里程计即可到达目标坐标并停止，验证 implicit localization 的跨本体可行性。
- 对 NAV 的关系：高度可借鉴 / 需对比——它给出了"如何把视觉几何表示真正用进导航策略"的具体范式（metric-aware 微调 + 历史几何重建 + 策略条件于 implicit geometry），正是 NAV Stage Three"如何使用 3D-aware Register-after-DiT 做导航"的直接参照与对比基线；其"策略条件于 implicit geometry 而非显式定位"与 NAV"策略只读 Register 不参与生成"在解耦思路上同源。


### 1. DreamZero（2026，arXiv preprint / ICLR 2026 World Models Workshop）

- 链接：
  - 论文：https://arxiv.org/abs/2602.15922
  - 代码：https://github.com/dreamzero0/dreamzero
  - 项目页：https://dreamzero0.github.io
- 官方 bibtex：
  ```bibtex
  @misc{ye2026worldactionmodelszeroshot,
    title={World Action Models are Zero-shot Policies},
    author={Ye, Seonghyeon and Ge, Yunhao and Zheng, Kaiyuan and Gao, Shenyuan and Yu, Sihyun and Kurian, George and Indupuru, Suneel and Tan, You Liang and Zhu, Chuning and Xiang, Jiannan and Malik, Ayaan and Lee, Kyungmin and Liang, William and Ranawaka, Nadun and Gu, Jiasheng and Xu, Yinzhen and Wang, Guanzhi and Hu, Fengyuan and Narayan, Avnish and Bjorck, Johan and Wang, Jing and Kim, Gwanghyun and Niu, Dantong and Zheng, Ruijie and Xie, Yuqi and Wu, Jimmy and Wang, Qi and Julian, Ryan and Xu, Danfei and Du, Yilun and Chebotar, Yevgen and Reed, Scott and Kautz, Jan and Zhu, Yuke and Fan, Linxi "Jim" and Jang, Joel},
    year={2026},
    eprint={2602.15922},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
  }
  ```
- 问题：VLA 模型虽有语义泛化能力，但难以泛化到训练分布外的新动作/技能与新物理环境；且依赖大量重复示教数据。
- 创新点：首次系统验证"WAM（世界-动作模型）即零样本策略"——以预训练视频扩散骨干联合预测视频与动作，把动作学习从稠密状态-动作模仿转化为逆动力学对齐；相对此前 WAM（mimic-video、GE-Act、Cosmos-Policy 等）在数据多样性/规模、自回归架构、跨本体迁移上全面突破。
- 核心方法：14B 自回归视频扩散 Transformer（基于 Wan2.1-I2V-14B），chunk-wise teacher-forcing 流匹配目标，视频与动作共享去噪时间步、单模型端到端联合去噪；推理用 KV-cache 并以真实观测替换预测帧消除误差累积；DreamZero-Flash 解耦视频/动作去噪调度 + 系统级并行/缓存 + 量化/CUDA kernel 调优。
- 训练 / 推理测试数据：约 500 小时异构真实机器人数据（含 AgiBot G1、YAM 等）；推理 benchmark 为真实世界 RoboArena 与仿真 PolaRiS、Genie Sim 3.0（100 任务）；指标为任务进度与成功率。
- 重要实验：相对 SOTA VLA 在环境+任务泛化上平均任务进度提升 >2×；任务特定后训练后仍比 VLA 高 10%；仅 10–20 分钟人类/他机视频即可在 AgiBot G1 上带来 >42% 相对提升；30 分钟 play 数据即可将 G1 预训练模型迁移到全新 YAM 本体并保留零样本泛化；38× 推理加速、7Hz 闭环控制。
- 对 NAV 的关系：可借鉴——其"视频即稠密世界表征、动作逆动力学对齐、自回归+KV-cache 实时控制、跨本体视频迁移"思路可直接迁移到导航策略；需对比其 RoboArena/Genie Sim 3.0 评测协议。

### 2. LingBot-World（2026，arXiv preprint）— action-conditioned 视角

- 链接：
  - 论文：https://arxiv.org/abs/2601.20540
  - 代码：https://github.com/Robbyant/lingbot-world
  - 项目页：https://technology.robbyant.com/lingbot-world
- 官方 bibtex：
  ```bibtex
  @article{lingbotworld2026b,
    title={Advancing Open-source World Models},
    author={Robbyant Team and others},
    journal={arXiv preprint arXiv:2601.20540},
    year={2026}
  }
  ```
- 问题：从视频生成走向"可交互世界模拟器"面临高质量交互数据稀缺、分钟级长程一致性难、扩散采样计算代价过高无法实时三大瓶颈；且最强方案多为闭源。
- 创新点：首个完全开源的高能力交互式世界模拟器，兼具通用域、分钟级长程、高动态度与实时性；提出分层语义数据引擎 + 三阶段进化训练（预训练→MoE 中训练→因果蒸馏后训练）。
- 核心方法：28B MoE 扩散 Transformer（继承 Wan2.2，双专家各约 14B，高噪专家建模全局结构、低噪专家精修细节，每步仅激活一专家）；动作通过相机嵌入与自适应键盘适配器注入，视觉骨干冻结仅微调动作适配器；后训练用因果注意力适配 + 分布匹配蒸馏(DMD)+对抗优化把双向扩散转为自回归系统。
- 训练 / 推理测试数据：混合数据源——第一/第三人称网络视频、带 W/A/S/D 控制输入与相机参数的游戏录制、Unreal Engine 合成渲染（带真值相机内外参）；分层 caption；评测用 VBench 在 100 条 >30s 生成视频上对比 Yume-1.5、HY-World 1.5。
- 重要实验：Table 1 相对 Matrix-Game 2.0/Yume-1.5/HY-World 1.5/Mirage 2/Genie 3，唯一同时满足通用域+长程+高动态+720p+实时+开源；Table 2 VBench 成像与美学质量均最高；LingBot-World-Fast 单 GPU 480p 下 16fps、延迟 <1s。
- 对 NAV 的关系：可借鉴——其以键盘/相机控制驱动的可导航交互世界、长程空间记忆与实时性正是 WAM 风格导航仿真器的关键能力；可作为导航策略的训练/评测环境与数据引擎。

### 3. DreamGen（2025，CoRL 2025）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2505.12705
  - 代码：N/A（项目页提供模型与流程）
  - 项目页：https://research.nvidia.com/labs/gear/dreamgen/
- 官方 bibtex：
  ```bibtex
  @InProceedings{pmlr-v305-jang25a,
    title={DreamGen: Unlocking Generalization in Robot Learning through Video World Models},
    author={Jang, Joel and Ye, Seonghyeon and Lin, Zongyu and Xiang, Jiannan and Bjorck, Johan and Fang, Yu and Hu, Fengyuan and Huang, Spencer and Kundalia, Kaushil and Lin, Yen-Chen and Magne, Lo\"{i}c and Mandlekar, Ajay and Narayan, Avnish and Tan, You Liang and Wang, Guanzhi and Wang, Jing and Wang, Qi and Xu, Yinzhen and Zeng, Xiaohui and Zheng, Kaiyuan and Zheng, Ruijie and Liu, Ming-Yu and Zettlemoyer, Luke and Fox, Dieter and Kautz, Jan and Reed, Scott and Zhu, Yuke and Fan, Linxi},
    booktitle={Proceedings of The 9th Conference on Robot Learning},
    pages={5170--5194},
    year={2025},
    volume={305},
    series={Proceedings of Machine Learning Research},
    publisher={PMLR}
  }
  ```
- 问题：机器人策略依赖昂贵的人工遥操作数据，且每新增任务/环境都需重新采集；仿真合成数据存在 sim2real gap。
- 创新点：把视频世界模型当作"合成数据生成器"（而非实时规划器），提出 4 阶段 neural trajectories 流程，首次实现零样本行为泛化与零样本环境泛化；并给出 DreamGen Bench 将视频模型质量与下游策略成功关联。
- 核心方法：(1) LoRA 微调 image-to-video 扩散模型（默认 WAN2.1）到目标本体；(2) 用初始帧+语言指令生成大量合成机器人视频（含新行为/新环境）；(3) 用 latent action model 或逆动力学模型(IDM)恢复伪动作；(4) 在 video-action 对（neural trajectories）上训练 visuomotor 策略。
- 训练 / 推理测试数据：仿真用 RoboCasa（合成数据放大至 333×）；真实世界在 Fourier GR1、Franka Emika、SO-100 三种本体 9 个任务上验证，每任务仅 10–13 条真实轨迹；评测指标为成功率，并引入 DreamGen Bench（指令跟随+物理跟随）评估 8 个视频模型。
- 重要实验：仅用单一 pick&place 环境，GR1 即可执行 22 个新行为；GR00T N1 单独训 pick&place 在新行为/环境近乎 0%，DreamGen 在已见环境新行为达 43.2%、完全未见环境 28.5%；合成数据量与策略性能呈 log-linear 提升；GR1 任务 37%→46.4%、Franka 23%→37%、SO-100 21%→45.5%。
- 对 NAV 的关系：可借鉴——其"视频世界模型作为导航策略的数据引擎、用初始帧+指令生成新环境 rollout"的范式可直接用于导航场景扩增与泛化训练；DreamGen Bench 的指令/物理跟随指标可迁移到导航世界模型评测。

### 4. V-JEPA 2（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2506.09985
  - 代码：https://github.com/facebookresearch/vjepa2
  - 项目页：https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
- 官方 bibtex：
  ```bibtex
  @article{assran2025vjepa2,
    title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
    author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and others},
    journal={arXiv preprint arXiv:2506.09985},
    year={2025}
  }
  ```
- 问题：如何主要靠观察（互联网视频）而非大量交互来学习理解、预测并规划物理世界的世界模型；现有视频生成式世界模型偏重视觉质量、规划能力受限且计算昂贵。
- 创新点：以 JEPA（联合嵌入预测架构）在表征空间而非像素空间预测，忽略不可预测细节；两阶段——无动作自监督预训练 + 少量交互后训练出动作条件世界模型 V-JEPA 2-AC，实现零样本机器人规划。
- 核心方法：编码器最大 1B 参数，在 >100 万小时互联网视频+100 万图像上做 mask-denoising 特征预测预训练；V-JEPA 2-AC 为 300M 因果 transformer，block-causal 注意力自回归预测下一帧表征（条件于动作与历史状态）；推理用图像目标 + MPC 规划，对候选动作"想象"后果并按与目标接近度评分。
- 训练 / 推理测试数据：预训练用 1M+ 小时网络视频；动作条件后训练仅用 DROID 数据集 <62 小时无标签机器人视频；零样本部署到两个实验室的 Franka 机械臂做 pick-and-place。指标含 Something-Something v2(77.3 top-1)、Epic-Kitchens-100(39.7 R@5)、PerceptionTest(84.0)、TempCompass(76.9) 及机器人成功率。
- 重要实验：长程 pick-and-place 用视觉子目标序列达 65%–80% 成功率（新物体+新环境，未采集该环境任何数据、无任务训练或奖励）；证明自监督视频预训练 + 极少交互即可得到可规划的世界模型。
- 对 NAV 的关系：可借鉴/需对比——其"表征空间世界模型 + MPC 想象规划 + 极少交互数据"对导航的 goal-conditioned 规划高度相关，且像素无关的 latent 预测更适合实时导航；是 DreamZero 像素级 WAM 的重要对照路线。

### 5. mimic-video（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2512.15692
  - 代码：https://github.com/mimic-video/mimic-video
  - 项目页：https://mimic-video.github.io/
- 官方 bibtex：
  ```bibtex
  @misc{pai2025mimicvideo,
    title={mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs},
    author={Pai, Jonas and Achenbach, Liam and Montesinos, Victoriano and Forrai, Benedek and Mees, Oier and Nava, Elvis},
    year={2025},
    eprint={2512.15692},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
  }
  ```
- 问题：VLA 的 VLM 骨干在静态图文上预训练，缺乏物理因果与时序动力学，必须靠稀缺遥操作数据从零学物理，数据效率瓶颈严重。
- 创新点：提出 Video-Action Model(VAM) 范式——直接在预训练视频模型的潜空间规划，不生成完整视频，而是用中间表征条件化 flow-matching 动作解码器(IDM)，把多模态长程规划卸载给视频骨干、把控制简化为单模态非因果逆动力学。
- 核心方法：冻结预训练视频扩散骨干（Cosmos-Predict2-2B），给定初始观测+指令先在潜空间合成视觉计划，提取中间视频表征条件化一个从零训练的 flow matching 动作解码器恢复低层动作；解耦视频建模与控制。
- 训练 / 推理测试数据：跨单臂到双臂灵巧任务评测（仿真与真实世界）；与 VLA 架构对比样本效率与收敛速度；指标为成功率、样本效率、收敛速度。
- 重要实验：相对传统 VLA 架构样本效率提升 10×、收敛速度提升 2×，在仿真与真实机器人操作上达 SOTA；证明视频潜空间规划优于像素级生成式 WAM 的控制效率。
- 对 NAV 的关系：可借鉴——"冻结视频骨干 + 潜空间视觉计划 + 轻量动作解码器"为导航提供了高效实时控制范式，避免每步全视频生成的开销；是 DreamZero 端到端联合去噪路线的有力对照。

### 6. EnerVerse-AC / EVAC（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2505.09723
  - 代码：https://annaj2178.github.io/EnerverseAC.github.io/
  - 项目页：https://annaj2178.github.io/EnerverseAC.github.io/
- 官方 bibtex：
  ```bibtex
  @article{jiang2025enerverseac,
    title={EnerVerse-AC: Envisioning Embodied Environments with Action Condition},
    author={Jiang, Yuxin and Chen, Shengcong and Huang, Siyuan and Chen, Liliang and Zhou, Pengfei and Liao, Yue and He, Xindong and Liu, Chiming and Li, Hongsheng and Yao, Maoqing and Ren, Guanghui},
    journal={arXiv preprint arXiv:2505.09723},
    year={2025}
  }
  ```
- 问题：机器人模仿学习评测需真实机器人或大型 3D 仿真，成本高难规模化；现有世界模型多从语言指令生成视频+预测动作，而非真正按 agent 动作模拟环境动态，无法做可控测试。
- 创新点：提出 action-conditional 世界模型 EVAC，按 agent 预测动作生成未来视觉观测，兼作数据引擎与策略评测器；引入多级动作条件机制 + ray map 编码做多视图动态生成，并加入失败轨迹数据提升泛化。
- 核心方法：在 EnerVerse 基础上加 action-conditioning：末端执行器投影动作图与图像 latent 拼接（空间级），delta 动作编码经 cross-attention 融入（时间级）；空间 cross-attention + ray direction map 编码处理多视图与相机运动；记忆机制支持长程视频序列；在 AgiBot-World 数据上额外挖掘失败轨迹训练。
- 训练 / 推理测试数据：主要源自 AgiBot-World 数据集（>210 任务、100 万轨迹），并补充真实遥操作与推理中的失败案例；作为数据引擎时对人类轨迹做动作分段+空间增广生成新视频序列，作为评测器时生成动作条件视频供人或 Video-MLLM 审阅；指标为生成真实度与策略评测保真度。
- 重要实验：实验验证 EVAC 作为数据引擎与评测器的有效性，生成真实且动作可控的多视图视频，减少对真实机器人/复杂仿真的依赖，评测保真度与真实世界表现相关。
- 对 NAV 的关系：可借鉴——其"按动作模拟环境动态的多视图世界模型 + 失败轨迹覆盖"思路可用于导航场景的 rollout 仿真与策略评测；多视图与相机运动编码对具身导航视角切换尤其相关。

### 7. WorldPlanner（2025，arXiv preprint）[引用链新工作]

- 链接：
  - 论文：https://arxiv.org/abs/2511.03077
  - 代码：N/A
  - 项目页：N/A
- 官方 bibtex：
  ```bibtex
  @article{khorrambakht2025worldplanner,
    title={WorldPlanner: Monte Carlo Tree Search and MPC with Action-Conditioned Visual World Models},
    author={Khorrambakht, R. and Ortiz-Haro, Joaquim and Amigo, Joseph and Mostafa, Omar and Dugas, Daniel and Meier, Franziska and Righetti, Ludovic},
    journal={arXiv preprint arXiv:2511.03077},
    year={2025}
  }
  ```
- 问题：行为克隆(BC)策略难迁移到新任务，且采集目标导向示教需频繁复位、成本高；现有基础世界模型规划计算昂贵（每控制步可达数分钟），长程规划受限。
- 创新点：以少量非结构化 play 数据训练动作条件视觉世界模型 + 扩散动作采样器 + 奖励模型，用 MCTS + 零阶 MPC 在世界模型内做长程规划；用扩散动作采样器离散化搜索空间以约束在世界模型训练分布内、缓解其幻觉。
- 核心方法：小型自回归扩散视觉世界模型（DIAMOND 风格，从 Atari 推广到真实机器人），从 play 数据从零训练（数小时、单 GPU 数天）；MCTS 在动作/观测分布内搜索，零阶 MPC 执行规划轨迹；可选图像条件奖励模型。
- 训练 / 推理测试数据：仅数小时非结构化 play 数据（无需目标导向示教与环境复位）；在 3 个真实世界机器人任务（含刚性与可变形物体）上验证；指标为成功率，与 BC 基线对比。
- 重要实验：实验支持"规划显著优于 BC 基线"的假设，在标准操作测试环境上规划带来显著提升；动作采样器有效抑制世界模型规划时的幻觉/分布外 rollout。
- 对 NAV 的关系：可借鉴——其"play 数据 + 世界模型内 MCTS/MPC 想象规划"是导航中长程规划与避障的可控范式；小型从零世界模型路线与 DreamZero/V-JEPA 2 大模型路线形成互补，适合作为导航的局部动力学规划器对照。
