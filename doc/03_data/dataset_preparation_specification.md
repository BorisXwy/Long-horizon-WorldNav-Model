# 长视频数据集准备任务书

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-002` |
| 类型 | 数据规范（Dataset Specification） |
| 状态 | Active |
| 更新时间 | 2026-07-28 |
| 职责 | 定义目录、manifest、质量检查和验收标准 |

## 一、任务背景

我们正在训练一个基于 Infinite-World 的流式视频世界模型（Streaming Video
World Model）。模型每次生成一个 81 帧的 video chunk，并使用 Register
保存更早的历史信息。

当前 RealEstate10K 本地数据只够构造两个连续 chunk：

```text
chunk 0（81 帧）→ chunk 1（81 帧）
```

后续训练需要连续的 4、8、16 个 chunks，即至少需要：

| chunks 数量 | 30 FPS 下原始帧数 | 近似时长 |
| ---: | ---: | ---: |
| 4 | 324 | 10.8 秒 |
| 8 | 648 | 21.6 秒 |
| 16 | 1296 | 43.2 秒 |

本任务需要准备以下五个数据集：

1. DL3DV-10K
2. Sekai
3. SpatialVID
4. Ego4D
5. RealEstate10K

目标不是简单地“把文件下载下来”，而是得到可供 NAV 项目直接筛选和训练的
长视频、元数据、相机信息和统一 manifest。

## 二、总体交付要求

### 2.1 公共目录

所有公共数据放在：

```text
/sharedata/datasets/<dataset_name>/
```

建议目录名固定为：

```text
/sharedata/datasets/DL3DV-10K/
/sharedata/datasets/Sekai/
/sharedata/datasets/SpatialVID/
/sharedata/datasets/Ego4D/
/sharedata/datasets/RealEstate10K/
```

不要把大型视频、压缩包、模型权重或缓存放在个人 home、NAV 代码目录或 Git
仓库中。

### 2.2 每个数据集的标准目录

每个数据集尽量整理为：

```text
<dataset_name>/
├── raw/                 # 官方原始文件或下载结果
├── videos/              # 可直接读取的视频；允许软链接到 raw
├── annotations/         # 官方 pose、caption、action 等标注
├── manifests/
│   ├── all.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── long_4chunk.jsonl
│   ├── long_8chunk.jsonl
│   ├── long_16chunk.jsonl
│   ├── revisit_candidates.jsonl
│   └── dataset_report.json
├── scripts/             # 下载、解包、扫描和 manifest 生成脚本
├── logs/                # 下载失败列表、扫描日志
├── LICENSE.txt          # 官方许可证或条款副本/链接
└── README_CN.md         # 中文使用说明
```

如果官方数据布局不适合移动，不要复制出第二份；保留官方布局，并在
`videos/` 或 manifest 中记录真实绝对路径。

### 2.3 统一 manifest 格式

`all.jsonl` 每行对应一个连续视频或 episode。无法获得的字段写 `null`，
不要编造。

```json
{
  "dataset": "DL3DV-10K",
  "episode_id": "unique_id",
  "video_path": "/sharedata/datasets/DL3DV-10K/...",
  "annotation_path": "/sharedata/datasets/DL3DV-10K/...",
  "source_url_or_id": null,
  "split": "train",
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "num_frames": 1800,
  "duration_sec": 60.0,
  "has_camera_pose": true,
  "pose_format": "说明坐标系和矩阵定义",
  "has_caption": false,
  "has_action": false,
  "valid_4chunk": true,
  "valid_8chunk": true,
  "valid_16chunk": true,
  "revisit_score": null,
  "license": "官方许可证名称",
  "status": "ok"
}
```

要求：

- `video_path` 必须使用绝对路径；
- 路径必须真实存在，并能被 OpenCV 或 ffmpeg 解码；
- `num_frames`、`fps`、分辨率和时长必须由实际文件扫描获得；
- 有 pose 时必须说明 pose 与视频帧、timestamp 的对齐方式；
- `episode_id` 在同一数据集中必须唯一；
- train/val 按 episode 划分，不能把同一长视频的不同窗口分到两个 split；
- 原始帧率不必统一重编码，manifest 中记录真实帧率即可。

### 2.4 长序列清单

按真实时长和帧数产生：

- `long_4chunk.jsonl`：可连续采样至少 324 帧；
- `long_8chunk.jsonl`：可连续采样至少 648 帧；
- `long_16chunk.jsonl`：可连续采样至少 1296 帧。

若视频不是 30 FPS，应按时间计算可覆盖长度，之后允许按 30 FPS 采样。不得通过
循环播放、复制帧或重复短 clip 的方式凑够长度。

### 2.5 视频有效性检查

对所有下载的视频执行自动检查：

1. 文件存在且大小大于 0；
2. ffprobe 能读取容器和视频流；
3. 能解码首帧、中间帧和末帧；
4. 实际帧数/时长不是 0；
5. 不是全黑视频；
6. 不是长时间完全静止；
7. 音频不是本项目的必要条件；
8. 记录损坏、失效链接和缺失标注，不能静默跳过。

检查结果写入 manifest 的 `status`，失败项写入：

```text
logs/failed_downloads.jsonl
logs/invalid_videos.jsonl
logs/missing_annotations.jsonl
```

不要求计算视频文件哈希，也不要为了校验重复复制大文件。

## 三、各数据集具体要求

## 3.1 DL3DV-10K

官方入口：

- <https://github.com/DL3DV-10K/Dataset>

优先级：最高。

第一阶段先准备 100 个完整长场景，不要一开始下载全量。选择标准：

- 单条手机视频优先不少于 60 秒；
- 航拍视频优先不少于 45 秒；
- 室内、室外、街道、建筑等场景尽量均衡；
- 优先包含折返、转身、绕行、环视或重新经过旧区域的序列；
- 必须保留相机参数、时间戳及官方 scene ID；
- 明确 pose 坐标系、矩阵方向和尺度定义；
- 统计可形成 4/8/16 chunks 的 episode 数量。

完成 100 条的小规模验收后，再决定是否扩大下载。

## 3.2 Sekai

官方入口：

- <https://huggingface.co/papers/2506.15675>

优先准备 walking、first-person、drone FPV 和 UAV 中的长序列。要求：

- 先确认 Hugging Face 数据结构和实际视频获取方式；
- 记录 YouTube/原始来源 ID 与失效链接；
- 保留 camera trajectory、location、scene、weather、caption 等已有标注；
- 区分 walking、FPV、UAV 等采集类型；
- 优先选择至少 45 秒且相机运动明显的序列；
- 第一阶段目标为至少 100 条有效长视频；
- 下载前在 `README_CN.md` 中写清楚许可证与YouTube衍生数据的使用限制。

如果官方只提供 URL、ID 或元数据，必须同时交付一个可恢复执行的下载脚本和失败
清单，不能把 URL 列表当成“已完成数据集”。

## 3.3 SpatialVID

官方入口：

- <https://github.com/NJU-3DV/SpatialVID>

SpatialVID 规模很大，第一阶段不要全量下载。要求：

- 先下载 metadata，统计 clip duration 分布；
- 统计满足 10.8、21.6、43.2 秒的 clip 数；
- 第一阶段选择至少 100 条满足 16 chunks 的视频；若不足，报告真实数量；
- 保留 camera pose、depth、motion instruction、mask 和 caption 的可用路径；
- 验证 pose/depth 与视频帧是否一一对应；
- README 中明确 CC BY-NC-SA 4.0 的非商业和 ShareAlike 限制；
- 不要把相邻但不连续的独立 clips 拼成一个 episode。

如果大多数 clip 只有几秒，应及时报告，不要继续盲目下载视频主体。

## 3.4 Ego4D

官方入口：

- <https://ego4d-data.org/docs/start-here/>

Ego4D 需要账号、许可或凭据时，不得共享个人密码、token、cookie 或 access key。
需要负责人登录或接受条款时，整理清楚操作步骤后再联系我们。

第一阶段要求：

- 先完成官方许可和下载工具配置；
- 不下载全量数据；
- 从长视频中选取约 100 个连续片段，每段至少 45–60 秒；
- 场景尽量覆盖行走、室内活动、室外活动和明显视角运动；
- 保留原视频 ID、participant ID 和时间范围；
- 同一原视频的多个窗口必须放在同一 split；
- 标明哪些序列没有 pose、caption 或 action；
- 不把它当作带导航动作的监督数据，主要作为第一视角视频先验。

## 3.5 RealEstate10K

官方入口：

- <https://google.github.io/realestate10k/download.html>

现有本地资源位于：

```text
/sharedata/RealEstate10K
```

不要覆盖、删除或重新移动现有数据。新增下载应放入统一的新目录，或者在确认后与
现有目录增量合并。

当前已知本地训练子集只有 180 个 episode，最长 279 帧，没有能形成 4 chunks
的 episode。本项要求：

- 从官方 metadata 中先统计理论上可覆盖 324/648/1296 帧的序列；
- 优先补抓能形成 16 chunks 的长序列；
- 保留 timestamp、intrinsics 和 camera extrinsics；
- 对 YouTube 失效、地区限制、视频删除分别记录；
- 第一阶段争取得到至少 100 条有效长 episode；
- 不要将现有 162 帧窗口视为独立原始 episode；
- 不要将同一 YouTube 视频的重叠窗口跨 train/val 划分。

若实际可下载的 16-chunk 序列明显不足，应提交下载成功率和长度分布，不需要人为
复制或插帧。

## 四、Revisit / Loop Closure 候选筛选

除一般长序列外，需要为长期记忆训练筛选约 100 条
`revisit_candidates`。优先满足：

- 摄像机重新回到之前出现过的位置；
- 转身后再次看到旧区域；
- 绕建筑、房间或物体一圈；
- 先远离再返回；
- 同一区域至少间隔 2 个 chunks 后再次出现。

有 camera pose 的数据，可先按相机中心距离和朝向自动提候选：

```text
时间间隔 >= 162 帧
位置距离较小
视角存在重叠
```

再为候选序列抽取 8–12 张均匀采样的 contact sheet 进行人工快速确认。
`revisit_candidates.jsonl` 增加：

```json
{
  "revisit_score": 0.82,
  "revisit_pairs": [[120, 900]],
  "review_status": "auto_candidate"
}
```

自动分数不是 ground truth，必须标注是自动筛选还是人工确认。

## 五、脚本要求

每个数据集至少提供以下可重复执行的脚本：

```text
scripts/download.sh
scripts/build_manifest.py
scripts/check_videos.py
scripts/report.py
```

要求：

- 下载支持断点续传；
- 重复运行不能重复下载已有文件；
- 数据根目录通过命令行参数指定，默认指向 `/sharedata/datasets/...`；
- 不在代码中写个人 token；
- 不执行危险的递归删除；
- 对失败样本给出明确日志；
- `report.py` 能独立重新生成 `dataset_report.json`；
- Python 依赖写入 `requirements.txt` 或 README；
- 所有说明使用中文，数据集名、字段名和技术名词保留英文。

建议命令形式：

```bash
bash scripts/download.sh --root /sharedata/datasets/DL3DV-10K --subset first100
python scripts/build_manifest.py --root /sharedata/datasets/DL3DV-10K
python scripts/check_videos.py --manifest manifests/all.jsonl
python scripts/report.py --manifest manifests/all.jsonl
```

## 六、数据报告要求

每个 `dataset_report.json` 至少包含：

```json
{
  "dataset": "DL3DV-10K",
  "download_date": "YYYY-MM-DD",
  "num_episodes": 100,
  "num_valid_videos": 98,
  "num_invalid_videos": 2,
  "total_duration_hours": 1.8,
  "duration_sec_min": 45.0,
  "duration_sec_median": 61.2,
  "duration_sec_max": 132.0,
  "num_4chunk": 98,
  "num_8chunk": 98,
  "num_16chunk": 91,
  "num_with_pose": 98,
  "num_with_caption": 0,
  "num_revisit_candidates": 27,
  "disk_usage_bytes": 123456789,
  "license": "...",
  "notes": "..."
}
```

同时在 `README_CN.md` 中用表格给出同样的核心统计，便于人工阅读。

## 七、分阶段验收

### 阶段 A：只分析 metadata

每个数据集先交付：

- 官方入口和许可证说明；
- metadata 的字段解释；
- 视频数量和时长分布；
- 预计满足 4/8/16 chunks 的数量；
- 预计下载空间；
- 下载是否需要账号、申请或人工接受条款。

阶段 A 完成并确认后，再进行大规模下载，避免下载大量无法用于长序列训练的短
clips。

### 阶段 B：每个数据集 5 条冒烟测试

每个数据集先准备 5 条可播放长视频，验证：

- 下载脚本可重复运行；
- manifest 字段正确；
- 视频可解码；
- pose/caption 能与视频对应；
- NAV 可以从每条视频连续读取 16 个 chunks。

### 阶段 C：第一批正式子集

建议目标：

| 数据集 | 第一批目标 |
| --- | ---: |
| DL3DV-10K | 100 条长 episode |
| Sekai | 100 条长 episode |
| SpatialVID | 100 条 16-chunk episode；不足则如实报告 |
| Ego4D | 100 个来自长原视频的连续片段 |
| RealEstate10K | 尽量获得 100 条 16-chunk episode |

### 阶段 D：最终交接

最终交付：

1. 数据绝对路径；
2. 下载和预处理脚本；
3. 所有 manifest；
4. `dataset_report.json`；
5. 中文 `README_CN.md`；
6. 下载失败和无效视频清单；
7. 许可证/访问限制；
8. 100 条左右的 revisit 候选总表；
9. 五个数据集的统一对比表。

没有满足目标数量不等于任务失败，但必须如实报告原因、成功率和可替代方案。

## 八、重要注意事项

- 不要删除或覆盖 `/sharedata` 中已有数据；
- 不要下载全量数据后才检查长度和许可证；
- 不要循环短视频、复制帧或拼接不连续 clips 来制造长样本；
- 不要擅自统一重编码所有视频，原视频优先保留；
- 不要在 Git 中提交视频、压缩包或大型标注；
- 不要将账号凭据写入脚本、日志或 README；
- 不要假设 metadata 存在就代表原视频可下载；
- 遇到需要签署数据条款、提供组织信息或大量占用公共存储时，先暂停并确认；
- 所有过程应可恢复、可重复、可统计。
