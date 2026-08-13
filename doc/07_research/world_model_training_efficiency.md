# 世界模型训练算力与速度调研

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-RES-001` |
| 类型 | 外部调研（Research Note） |
| 状态 | Reference |
| 更新时间 | 2026-07-28 |
| 职责 | 汇总公开世界模型训练规模、速度和可比性限制 |

更新时间：2026-07-24。

本文只记录论文、官方仓库或官方项目页能够核实的数据。论文没有公开的项目不根据
模型规模臆测训练时长；数据视频时长也不等同于训练 wall-clock time。

## 1. Infinite-World

来源：

- 论文：https://arxiv.org/abs/2602.02393
- 代码：https://github.com/MeiGen-AI/Infinite-World

公开训练设置：

- Backbone：Wan2.1-T2V-1.3B。
- 两阶段训练均为480P，AdamW，恒定 learning rate `1e-5`。
- Pre-training：30小时以上互联网第一视角与探索视频；直接压缩 history，最多4个
  temporal chunks。
- RDD fine-tuning：约30分钟 revisit-dense 实拍视频；历史上下文从1张图扩展至
  16 chunks，并启用 hierarchical compression。
- HPMC：3D-ResNet，时间压缩率4；5个重叠窗口，每窗64帧；最多320帧历史被压成
  固定 `Tmax=20`。
- 硬件：所有实验使用16张 NVIDIA H800。

没有公开：

- 两阶段的 steps、global batch size、每步耗时、总训练小时数或GPU-hours。
- 官方仓库没有发布完整训练入口与 optimizer loop，主要是推理实现。因此无法从
  公开信息可靠推回训练速度。

## 2. FantasyWorld

来源：

- 论文：https://arxiv.org/abs/2509.21657
- 代码：https://github.com/Fantasy-AMAP/fantasy-world

公开训练设置：

- Backbone：冻结的 Wan2.1-I2V-14B；训练约18万视频 clips。
- 数据包括 RealEstate10K、ACID、DL3DV、WildRGB、ScanNet、TartanAir。
- AdamW，learning rate `1e-5`。
- Stage 1（latent bridging）：只训练 geometry branch，20,000 steps，
  global batch 64；64张 H20，36小时。
- Stage 2（unified co-optimization）：81帧，`592×336` 或 `336×592`；
  只训练双向 cross-attention 与 camera-control adapter，核心 video/geometry
  backbones 冻结；10,000 steps，global batch 112；112张 H20，144小时。

由官方数字直接换算：

| 阶段 | 秒/optimizer step | Aggregate clips/s | GPU-hours |
| --- | ---: | ---: | ---: |
| Stage 1 | 6.48 | 9.88 | 2,304 H20·h |
| Stage 2 | 51.84 | 2.16 | 16,128 H20·h |
| 合计 | - | - | 18,432 H20·h |

这里的step速度是整个64/112卡集群完成一个global step的速度，不能与单卡step
直接比较。Stage 2虽然只训练adapter，仍需要让14B视频分支和geometry分支完成
81帧高分辨率前向与反向到adapter，因此每步很慢。

## 3. LingBot-World

来源：

- 论文：https://arxiv.org/abs/2601.20540
- 代码：https://github.com/Robbyant/lingbot-world

公开训练方法：

- 从 Wan2.2-I2V 初始化；两个约14B的high-noise/low-noise experts，总参数约28B，
  单个timestep只激活一个expert。
- Middle-training 采用5秒到60秒的 progressive curriculum，同时训练I2V与
  video continuation。
- Action stage冻结主DiT，只训练 action projection 与 AdaLN adapter。
- 使用 activation checkpointing、FSDP2 和 Ulysses context parallel。
- Post-training 先进行 block-causal/diffusion-forcing adaptation，再进行
  self-rollout、truncated gradient、DMD和adversarial distillation。

没有公开：

- 训练数据总量、各阶段steps、batch size、GPU型号/数量、wall-clock time和
  GPU-hours。
- 官方仓库仅发布推理代码，没有训练脚本。

论文公开的 `16 FPS @ 480P` 是 LingBot-World-Fast 的推理吞吐。原文写的是
“one GPU node”，官方命令使用 `torchrun --nproc_per_node=8`、Ulysses size 8，
因此不能将该数字写成“单张GPU 16 FPS”，更不能当作训练速度。

## 4. 公开程度更好的参照

### SANA-WM

官方项目：https://nvlabs.github.io/Sana/WM/

- 2.6B，约21.3万公开视频 clips。
- 主训练使用64张 H100、15天，即约23,040 H100·h。
- VAE适配另需约3.5天×64 H100，即约5,376 H100·h。
- 5秒阶段为每GPU batch 1；分钟级阶段使用CP=2，相当于每GPU 0.5 clip，
  global batch 32。
- 公开训练脚本、FSDP2/CP配置和部分蒸馏训练配置，是更适合做工程速度基准的项目。

### Endless World

论文：https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Endless_World_Real-Time_3D-Aware_Long_Video_Generation_CVPR_2026_paper.pdf

- Wan2.1-1.3B，832×480，使用4张H100训练。
- 论文没有给出总训练时长、steps或batch，因此仍不能计算训练吞吐。
- 单张H100推理约17 FPS；这是蒸馏后的推理速度，不是训练速度。

## 5. 对NAV实验的含义

现有工作支持以下判断：

1. 480P视频DiT全参训练本来就是高计算量任务。Infinite-World使用16张H800，
   并非单张48GB卡完成训练。
2. 工业项目依赖FSDP2与context parallel。LingBot-World明确把模型参数/optimizer
   state与长序列分别切到多卡；我们的当前NAV是单卡承载一个完整实验。
3. FantasyWorld选择冻结14B核心backbone，只训练geometry/adapter；即便如此，
   Stage 2在112张H20上仍需51.84秒/global step。
4. 日志中的step必须连同global batch与并行规模比较。`2秒/step`本身不能说明
   单样本或单位token更快。
5. 当前NAV的首要优化方向应是减小时空token、启用多卡context parallel/FSDP，
   其次才是常规算子优化。仅比较1.4B与3B参数量会严重误判视频模型训练成本。
