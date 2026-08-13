# V0 Infinite-World 项目结构、推理流程与论文一致性检查

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-ARC-001` |
| 类型 | 架构解析（Architecture Note） |
| 状态 | Verified |
| 更新时间 | 2026-07-28 |
| 职责 | 记录 Infinite-World 论文、代码和本地推理链路的对应关系 |

## 1. 项目定位

Infinite-World 以 Wan2.1-T2V-1.3B 为视频生成骨干，在 Wan DiT 上增加动作条件
（Action Conditioning）和分层无位姿记忆压缩器（Hierarchical Pose-free
Memory Compressor，HPMC），用于分块生成可交互的长视频。当前公开仓库以
推理代码为主，没有提供论文中的训练、动作标注、RDD 数据构建和完整评测数据。

## 2. 目录结构

```text
Infinite-World/
├── README.md                         项目说明、安装和权重说明
├── infer_local.sh                    本地单卡/多卡推理入口
├── requirements.txt                  Python 依赖
├── configs/
│   └── infworld_config.yaml          模型、权重、调度器和视频长度配置
├── prompts/
│   └── demo.yaml                     官方示例的文本、初始图和动作文件
├── assets/
│   ├── framework.png                 论文方法示意图
│   ├── title.png                     项目标题图
│   └── example_case/
│       ├── 0001.jpg / 0001.json      校园示例及逐帧动作
│       └── 0002.jpg / 0002.json      幻想城市示例及逐帧动作
├── scripts/
│   └── infworld_inference.py         完整推理和滚动生成主程序
└── infworld/
    ├── models/
    │   ├── dit_model.py              Wan DiT、HPMC、Action Encoder
    │   ├── scheduler.py              Rectified Flow 训练损失与采样器
    │   ├── umt5.py / t5.py           UMT5 文本编码器
    │   └── checkpoint.py             梯度检查点工具
    ├── vae/
    │   └── vae.py                    Wan VAE 编码、解码及封装
    ├── configs/
    │   └── bucket_config.py          分辨率和宽高比桶
    ├── context_parallel/
    │   └── context_parallel_util.py  上下文并行工具
    ├── utils/
    │   ├── data_utils.py             视频读取与保存
    │   ├── dataset_utils.py          图像/视频类型判断
    │   ├── prepare_dataloader.py     按字符串动态加载 Python 对象
    │   └── registry.py               注册工具
    └── clip/                         Wan 图像编码相关组件；当前 T2V 推理未使用
```

`outputs/` 是本地生成结果目录，不属于上游代码主体。

## 3. 核心配置

`configs/infworld_config.yaml` 定义：

| 配置 | 当前值 | 作用 |
| --- | ---: | --- |
| DiT | WanModel，约 1.3B | 视频扩散/流匹配生成骨干 |
| `dim` | 1536 | Transformer 隐藏维度 |
| `num_layers` | 30 | DiT 层数 |
| `num_heads` | 12 | 注意力头数 |
| `in_channels` | 20 | 16 个 VAE 通道加 4 个条件掩码通道 |
| Text Encoder | UMT5-XXL | 文本条件编码 |
| VAE | Wan2.1 VAE | RGB 视频与 16 通道 Latents 互换 |
| Sampling Steps | 30 | Rectified Flow 推理步数 |
| Shift | 7 | 该分辨率的时间步变换参数 |
| CFG | 5.0 | 文本无分类器引导强度 |
| Chunk | 81 RGB 帧 | 每轮生成的视频长度 |
| 精度 | bfloat16 | DiT 推理精度 |

当前权重确实包含 HPMC 和动作编码器参数：总参数约 14.47 亿，其中 DiT Blocks
约 13.93 亿、HPMC `latent_encoder` 约 2527 万、`action_encoder` 约 237 万。
加载日志为 `Missing: 0, Unexpected: 0`，说明推理结构与公开权重完全匹配。

## 4. 官方示例输入

`prompts/demo.yaml` 中每个任务包含：

```text
[文本提示词, 初始图像路径, 动作 JSON 路径]
```

动作 JSON 按 RGB 帧记录两条离散控制流：

- `move`：前进、后退、左右平移、组合移动、无动作或不确定；
- `view`：上下左右转动、组合转动、无动作或不确定。

两类动作各有 10 个离散编号，其中编号 9 表示不确定（Uncertain）。校园示例
包含 1396 个动作，可支持长于默认 13 chunks 的生成。

## 5. 推理调用链

### 5.1 启动与模型加载

```text
infer_local.sh
    ↓
scripts/infworld_inference.py
    ├─ 读取 infworld_config.yaml
    ├─ 加载 Wan VAE
    ├─ 加载 UMT5
    ├─ 创建 RFlowScheduler
    ├─ 创建修改后的 WanModel
    └─ 加载 infinite_world_model.ckpt
```

单卡模式直接运行 Python；多卡模式使用 `torchrun`。当前脚本将
`context_parallel_size` 固定为 1，因此多卡主要按不同提示词做数据并行，并未
启用论文训练所用的上下文并行。

### 5.2 输入预处理

初始图像经过以下处理：

1. 按宽高比选择最接近的分辨率桶；
2. 等比例缩放并中心裁剪；
3. 归一化到 `[-1, 1]`；
4. 使用 Wan VAE 编码为视频 Latents；
5. UMT5 将文本提示词和负面提示词编码为文本条件；
6. JSON 动作字符串映射为 `move/view` 整数序列。

### 5.3 单个 Chunk 的生成

每轮首先把当前完整 `video_buffer` 重新编码为历史 VAE Latents，HPMC 再将
这些历史 Latents 压缩。模型同时保留压缩前的最后一个 Latent 作为局部记忆：

```text
完整已生成视频
    ↓ Wan VAE Encode
历史 Latents
    ├─ HPMC → 最多 20 个压缩历史 Latents
    └─ 最后一个 Latent → Local Memory
```

随后在时间维拼接：

```text
[压缩历史, 最近一帧, 当前待去噪 Latents]
```

并添加二值条件掩码，区分已知历史和待预测部分。Action Encoder 先分别嵌入
`move` 与 `view`，再通过两层步长为 2 的一维卷积压缩到 VAE 时间频率，投影至
1536 维后，加到当前生成段的特征上。

Rectified Flow 调度器从随机噪声开始执行 30 步更新；条件分支与负面提示词分支
进行 Classifier-free Guidance（CFG）。生成的 21 个 Latents 经 VAE 解码为
81 帧。

### 5.4 跨 Chunk 滚动

每个新 Chunk 的第一帧与上一个 Chunk 的最后一帧重合，因此保存时丢弃新
Chunk 的第一帧：

```python
video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
```

所以总 RGB 帧数为：

\[
1+80\times N_{\mathrm{chunks}}
\]

例如 1、2、13、16 chunks 分别产生 81、161、1041、1281 帧。动作序列也以
80 帧为步长滑动，使边界动作与重合帧对齐。

## 6. HPMC 的代码实现

HPMC 在 `TemporalLatentEncoder` 中实现，主体是：

```text
Conv3D
 → 2×ResBlock3D
 → 时间下采样 ×2
 → 2×ResBlock3D
 → 时间下采样 ×2
 → ResBlock + Wan-style Attention + ResBlock
 → Conv3D
```

时间压缩率为 4，空间分辨率保持不变。

短历史（不超过 80 个 VAE 时间 Latents）直接压缩一次：

```text
L → L/4
```

长历史使用两级压缩：

```text
完整历史
 → 动态步长选取 5 个重叠窗口，每个窗口长度 64
 → 每个窗口 64→16
 → 拼接为 80
 → 再压缩为 20
```

这使进入 DiT 的长期记忆最多保持为 20 个时间 Latents，但仍保留每个 Latent
的空间网格。

## 7. 与论文的一致性检查

| 论文设计 | 本地推理实现 | 结论 |
| --- | --- | --- |
| 基于 Wanx-2.1-1.3B | 30 层、1536 维 WanModel，并复用 Wan VAE/UMT5 | 一致 |
| 平移和旋转动作解耦 | 独立 `move/view` Embedding，随后联合编码 | 一致 |
| 动作三态标注：No-op、Discrete、Uncertain | 两套词表均包含 `no-op`、离散动作和 `uncertain=9` | 推理接口一致；标注流水线未公开 |
| HPMC 时间压缩率 \(k=4\) | 两次 TemporalDownsample | 一致 |
| \(N=5\) 个重叠窗口 | `TARGET_N_CHUNKS=5` | 一致 |
| 窗口 \(W=64\) | `W_IN=64` | 一致 |
| 固定记忆预算 \(T_{\max}=20\) | `MAX_T_OUT=20` | 一致 |
| 动态滑动步长覆盖完整历史 | \(S=\lfloor(L-W)/(N-1)\rfloor\) | 一致 |
| 压缩历史、最后一帧和噪声目标拼接 | 代码按该顺序在时间维拼接 | 一致 |
| 二值掩码区分条件和预测目标 | 16 通道 Latents 加 4 通道掩码 | 一致 |
| HPMC 与 DiT 联合训练 | 权重同时包含 `latent_encoder` 和 DiT 参数 | 权重支持；训练代码未公开，无法独立复核 |
| 预训练历史最多 4 chunks，微调 1–16 chunks | 推理支持任意 `NUM_CHUNKS` | 接口兼容；训练采样代码未公开 |
| 论文评测每条轨迹 16 chunks | 脚本默认值为 13 chunks | 默认配置不一致，但环境变量可设为 16 |
| 固定计算开销的长时记忆 | DiT 接收的压缩记忆长度固定 | DiT 主体基本一致；端到端推理并非严格常数开销 |

## 8. 需要特别注意的差异

### 8.1 当前推理并非严格在线流式

代码每生成一个 Chunk，都会保留完整 RGB `video_buffer`，并重新用 VAE 编码
全部历史：

```text
完整 RGB 历史 → 每轮重新 VAE Encode → HPMC
```

因此固定的是“送入 DiT 的记忆长度”，不是整个推理系统的历史存储量和计算量。
随着视频增长，CPU/RAM 中的 RGB Buffer 和 VAE 重编码成本仍会增加。论文关于
常数计算开销的说法对 DiT 上下文成立，但对当前公开的端到端推理脚本并不完全
成立。

### 8.2 HPMC 不是在线更新的状态

当前 HPMC 每轮从完整历史重新计算压缩表示，没有把上一轮的 20 个压缩 Latents
作为状态直接递推。这与 NAV Stage One 计划的 Register Token 在线记忆不同。

### 8.3 “320 frames”实际是 VAE 时间位置

论文称分层设计覆盖最多 320 帧；代码中的 `5×64=320` 实际作用在 VAE Latent
时间轴。Wan VAE 约进行 4 倍时间压缩，因此它对应约 1280 个 RGB 视频帧，与
论文展示的 1000+ 帧能力相符。论文在这一处使用“frames”而非“latent frames”，
表述不够严格。

### 8.4 公开仓库不足以完整复现实验

仓库没有提供：

- 30 小时预训练数据及预处理；
- 基于 VGGT 相机位姿的 Uncertainty-aware Action Labeling 代码；
- 30 分钟 Revisit-Dense Dataset（RDD）；
- 模型训练入口和损失配置；
- 论文评测使用的 100 张初始图和 10 条 16-chunk 动作轨迹；
- 用户研究与 ELO 评测实现。

因此可以核对并复现公开权重的推理算法，但不能仅凭该仓库完整验证论文训练过程
和表格结果。

## 9. 官方示例复跑结果

2026-07-24 使用 `demo.yaml` 的第一个校园示例重新执行了标准 30-step、
1-chunk 推理：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World
CUDA_VISIBLE_DEVICES=0 \
INFWORLD_MAX_PROMPTS=1 \
INFWORLD_NUM_CHUNKS=1 \
INFWORLD_SAMPLING_STEPS=30 \
INFWORLD_OUTPUT_DIR="$PWD/outputs/example-recheck-step30-20260724" \
bash infer_local.sh 1
```

输出视频：

```text
/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World/outputs/example-recheck-step30-20260724/0000_A_serene_campus_walkway_lined_.mp4
```

检查结果：

| 项目 | 结果 |
| --- | --- |
| 权重加载 | `Missing: 0, Unexpected: 0` |
| 视频编码 | H.264，YUV 4:2:0 |
| 分辨率 | 896×448 |
| 帧率 | 30 FPS |
| 帧数 | 81 |
| 时长 | 2.7 秒 |
| 文件大小 | 19,395,356 字节 |
| 完整解码检查 | 通过，FFmpeg 未报告错误 |
| SHA-256 | `6d283fb83c3310dc85fda70582fd0f4609bf57fe5b40723795802e9280be41ac` |

该文件与 2026-07-23 使用相同 seed、配置和输入生成的校园视频 SHA-256 完全
一致，说明当前单卡推理在该机器上具有确定性和可复现性。
