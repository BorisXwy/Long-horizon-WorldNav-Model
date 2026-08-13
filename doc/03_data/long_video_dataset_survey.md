# 长视频训练数据调研

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-001` |
| 类型 | 数据调研（Dataset Survey） |
| 状态 | Reference |
| 更新时间 | 2026-07-28 |
| 职责 | 比较候选长视频数据及其对流式训练的适用性 |

更新时间：2026-07-26

## 当前 NAV 数据的实际情况

本节描述的是 2026-07 旧 V0/early-V1 训练代码。相关旧训练脚本已在
2026-08-14 从当前 NAV 代码主线删除，仅作为历史问题来源保留。

当前训练确实只使用两个 chunk，而且不是运行时从长 episode 中抽取两个
chunk，而是在缓存阶段就固定成两个：

- `scripts/prepare_re10k_manifest.py` 将每个样本构造成 162 帧窗口，步长为
  81 帧；
- `scripts/cache_re10k_latents.py` 只编码 `[0:81]` 和 `[81:162]`；
- `scripts/train_infinite_register.py` 固定令第一个 chunk 为 `history`、第二个
  chunk 为 `target`。

所以每个训练样本只发生一次 Register 更新和一次 next-chunk prediction。
模型没有在训练中经历 3 个以上 chunk，也没有学习 Register 连续递归更新后的
误差累积。

本机 `/sharedata/RealEstate10K/manifests/frames_train.jsonl` 的统计为：

| 项目 | 数量 |
| --- | ---: |
| 已落盘 episode | 180 |
| 总帧记录 | 26,126 |
| episode 帧数（最小/中位/最大） | 41 / 123 / 279 |
| 可形成至少 2×81 帧的 episode | 57 |
| 可形成至少 4×81 帧的 episode | 0 |

因此现有本地 RE10K 子集不能支持原生 4/8/16-chunk 训练。

## Infinite-World 的长数据构造

论文中的训练不是只用两个 chunk：

- 预训练（Pre-training）混合不同上下文长度，历史上限为 4 个 temporal
  chunks；
- 微调（Fine-tuning）把历史长度从单张图像扩展到 16 个 chunks；
- 每个 chunk 为 81 个 RGB 帧时，16 chunks 大约是 1296 帧，即 30 FPS 下
  约 43.2 秒；
- 预训练使用超过 30 小时的互联网第一视角和探索视频；
- 最后的 revisit-dense fine-tuning 使用约 30 分钟自行拍摄的长视频，
  强调频繁 loop closure；论文称约 100 条这种序列即可形成较稳定的记忆行为。

其 HPMC（Hierarchical Memory-Preserving Compression）支持将最多 320 个
latent temporal frames 压到固定 20 帧。代码中的长历史路径先从完整历史取
5 个长度为 64 的重叠窗口，各压成 16 帧并拼成 80 帧，再压到 20 帧。

官方论文：
[Infinite-World](https://arxiv.org/html/2602.02393v1)。

截至本文更新时间，官方仓库发布了代码和权重，但没有公开论文所用的
30+ 小时互联网集合、30 分钟 revisit-dense 数据或其训练 manifest：
[Infinite-World GitHub](https://github.com/MeiGen-AI/Infinite-World)。

## LingBot-World 的长数据构造

LingBot-World 使用渐进式时长课程（Progressive Duration Curriculum）：

1. 从 5 秒视频开始；
2. 逐步延长到 60 秒；
3. 随时长同步调整 flow shift；
4. 同时训练 image-to-video 和 video-continuation；
5. 一分钟序列使用 FSDP2 和 Ulysses Context Parallel 处理显存压力。

其数据引擎混合：

- 第一视角真实视频，如 Ego4D、EPIC-KITCHENS；
- 第三视角真实视频；
- 带 RGB、WASD 和 camera control 的游戏数据；
- Unreal Engine 合成场景和轨迹。

处理流程包含分辨率/时长过滤、Koala 与 TransNetV2 分段、VLM
质量和运动过滤、相机位姿伪标注，以及 narrative/static/dense-temporal
三级 caption。论文明确说明训练来源与处理方法，但官方代码仓库没有发布组装后
的长视频训练语料、游戏数据或 UE 轨迹数据。

官方论文：
[LingBot-World](https://arxiv.org/html/2601.20540v1)。

## 可获得的公开长视频数据

| 数据集 | 规模和长时特性 | 对 NAV 的价值 | 限制 |
| --- | --- | --- | --- |
| [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) | 10,510 个场景；手机视频通常不少于 60 秒，航拍不少于 45 秒；提供相机参数 | 最适合先做 4/8/16-chunk 和 loop closure；室内外覆盖与 RE10K 接近 | 需接受数据条款；480P+pose 全量约 730 GB，建议先下子集 |
| [Sekai](https://huggingface.co/papers/2506.15675) | 超过 5,000 小时 walking、drone FPV/UAV，覆盖 100+ 国家、750 城市；含相机轨迹和 caption | 最接近 Infinite-World 的大规模探索视频预训练分布 | YouTube 来源的实际可下载率和许可证需在批量下载前审计 |
| [SpatialVID](https://github.com/NJU-3DV/SpatialVID) | 由 21,000 小时原始视频筛成约 2.7M clips、7,089 小时；含 pose、depth、motion instruction、caption | 适合扩大空间运动和文本条件多样性 | 很多条目是切分后的 clip，应先检查连续时长；许可证为 CC BY-NC-SA 4.0 |
| [Ego4D](https://ego4d-data.org/docs/start-here/) | 3,670+ 小时第一视角日常活动视频 | 可作为 LingBot 类第一视角通用先验 | 完整数据很大且需申请许可；并非所有视频都有连续相机位姿或导航动作 |
| [RealEstate10K](https://google.github.io/realestate10k/download.html) | YouTube 房产漫游视频，相机内外参和时间戳公开 | 和当前 pipeline 最兼容，可补抓比本地子集更长的 episode | 原始 YouTube 链接存在失效；当前本机子集最长仅 279 帧 |
| [ACID](https://cove.thecvf.com/datasets/609) | 航拍自然场景，含逐帧 3D camera parameters | 可补充长距离飞行和大范围视差 | 场景域偏航拍自然环境，且原视频链接可能失效 |

## 推荐实施顺序

### 第一阶段：先把训练目标改成长递归

缓存格式改为一个 episode 保存可变数量的独立 chunk，训练时按课程采样
`L ∈ {2, 4, 8, 16}`。Register 依次读取前面的 chunk，并在每次转移或最后一次
转移计算 next-chunk diffusion loss。显存不足时采用 truncated BPTT，例如
每 2–4 个 chunk detach 一次 Register。

这一步不能只对当前 162 帧 cache 做重复采样，否则模型看到的是重复内容，
无法学到长时间累积与 loop closure。

### 第二阶段：用 DL3DV-10K 子集验证

优先下载 50–100 个具有明显折返、环视或回到原位置的 60 秒序列。按 81
帧/chunk，单条视频能提供约 16–22 个连续 chunk，足以覆盖
Infinite-World 的 16-chunk 微调设置。先在 2→4→8→16 的课程上验证
Register 是否能在多次递归更新后保持场景信息。

### 第三阶段：扩大覆盖面

- Sekai 用于大规模第一视角/探索预训练；
- SpatialVID 用于带 pose、depth、caption 的空间运动训练；
- Ego4D 只作为第一视角通用视频先验，不作为首个导航记忆数据集。

### 第四阶段：构造小型 revisit-dense 集合

从 DL3DV/Sekai 筛选或自行拍摄约 100 条频繁回访的长序列，形成
Infinite-World RDD 的公开可复现替代版本。相比单纯增加随机视频小时数，
这部分对测试长期记忆和 loop closure 更关键。
