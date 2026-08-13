# 长视频数据准备状态

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-003` |
| 类型 | 动态状态（Operational Status） |
| 状态 | Live |
| 更新时间 | 2026-08-03 |
| 职责 | 汇总下载、标注和预处理的当前状态；历史过程保留在后半部分 |

更新时间：2026-08-07 18:40（Asia/Shanghai）

## 规模与 latent（与 VLN 表同口径）

统一：1 chunk = 81 帧 → latent `[16,21,56,112]` **float32 = 8.43 MB/chunk**（与现有 `.pt` 文件大小一致）。

| 数据集 | 落盘条数 | 训练资格条数 | 平均帧（中位） | 平均 chunk | 3.b 每条现有大小 | 4. 每条 latent（个数 × 8.43 MB） | latent 已完成 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SpatialVID | 30,000 视频（89 GB） | 23,837（≥2 chunks） | 499（405） | 6.16 | 视频 ≈**3.0 MB**/条 | **52.0 MB**（约 6.2 个） | 17,697 / 23,837 |
| DL3DV-10K | 146 scene / 141 资格 | 141 | 4337（3969） | 53.5 | 视频 ≈**630 MB** + images_8 ≈**76 MB** | **451 MB**（约 53.5 个） | 141 / 141 |
| RE10K | 帧在 `/sharedata/RealEstate10K/processed_frames` | 269 | 162（162） | 2.0 | 帧包 ≈**180 MB**/ep | **16.9 MB**（2 个） | 269 / 269 |
| Argoverse2 | 磁盘 train≈39 log；manifest 14 | 14 | 243（243） | 3.0 | 图像 ≈**75 MB**/ep | **25.3 MB**（3 个） | 8 / 14 |
| Sekai | pose/metadata ≈89 GB；**训练视频 0** | 标注 walking-hq 18,208（文件名 1800 帧→22 chunks） | 1800* | 22* | 视频未落盘 | 预期 **185 MB**/条* | 0 |
| Kinetics-400 | ≈298,824（430 GB） | 不进本轮 NAV | — | — | ≈**1.4–1.6 MB**/视频 | 未编码 | — |

\*Sekai 长度为官方文件名起止帧，视频下载后需 ffprobe 复核。`full_episodes_v1` latent 目录实测约 **666 GB+**（按文件合计 SpatialVID 已编码部分约 920 GB 量级，随续编码增长）。

## 当前执行快照

| 数据集 | 当前占用 | 后台状态 | 当前可用内容 |
| --- | ---: | --- | --- |
| Kinetics-400 | 430 GB | 下载完成；VGGT 标注运行中，但不进入本轮 NAV 训练 | 298,824 个视频 |
| RE10K | 38 GB | `nav_re10k` 等待私有 YouTube cookie；269个 full-episode latent 已完成 | 269个 full episode |
| DL3DV | 94 GB | 下载完成；141个 full-episode latent 已完成 | 141个可靠 Pose/Image episode |
| SpatialVID | 89 GB | 30,000条下载完成；full-episode latent 7-shard 并行编码中（14,776/23,837） | 23,837条满足至少2 chunks，action 已完成 |
| Argoverse 2 | 17 GB | rclone 下载继续；full-episode latent 7-shard 并行编码中（5/14） | 14个可靠 Pose/Image episode |
| SEKAI | 89 GB | 等待私有 YouTube cookie | Pose 已有，训练视频尚未落盘 |
| Ego4D | 50 KB | 等待 AWS credentials | 尚无有效视频 |

下载目录保持原始状态；NAV 派生 manifest 和 latent 只写入
`/sharedata/NAV/derived/`。当前两条并行准备链为：

- `nav_full_episode_prep`：7-shard 并行编码 `full_episodes_v1` latent，
  GPU 0 与 GPU 1 各跑一个 shard，其余显存供 Stage One 1.0 训练使用；
- Stage One 1.0 A/B 训练占用两卡主要显存（详见
  `../04_training/v0_stage_one_dl3dv_experiments.md`）。

`full_episodes_v1` manifest 共 24,261 条：SpatialVID 23,837、RE10K 269、
DL3DV 141、Argoverse2 14。chunk 分布以 2–11 chunks 为主，长尾覆盖到 119
chunks（DL3DV 等长 episode）。

DL3DV 必须读取与 `transforms.json` 一一对应的 `images_8/frame_*.png`。旧版
统一缓存曾将 pose 序号误当成原 MP4 帧号，因此新结果隔离写入
`/sharedata/NAV/derived/latents_pose_aligned/` 与
`/sharedata/NAV/derived/latents/full_episodes_v1/`，不覆盖旧文件。

当前实时快照：

| 派生数据 | 完成数 | 目标数 | 状态 |
| --- | ---: | ---: | --- |
| SpatialVID action | 23,837 | 23,837 | 完成 |
| full_episodes_v1 DL3DV latent | 141 | 141 | 完成 |
| full_episodes_v1 RE10K latent | 269 | 269 | 完成 |
| full_episodes_v1 SpatialVID latent | 14,776 | 23,837 | GPU 0/1 七路并行（shard 2、6 运行中） |
| full_episodes_v1 Argoverse2 latent | 5 | 14 | 等待 SpatialVID 后自动执行 |

SpatialVID 的30,000条中有6,163条不足162帧，按Short资格排除，不属于解码或
Pose损坏。Action审计发现1条Pose/index长度不一致，已按可配对最短长度处理并
记录异常。

2026-08-01 将数据准备切到 `full_episodes_v1` 口径：每条 episode 一次性缓存
全部完整 chunks，训练 loader 按 history 长度课程随机切片，不再固定 Short
2–3 chunk 窗口。7-shard 分片规则为全局 episode 序号 `index mod 7`，互不
重复；已有 `.pt` 自动跳过，输出使用临时文件原子替换。

本节是最新事实来源。下方内容保留早期准备过程和阶段性决策，用于追溯。

## 已完成的公共配置

- 统一数据根目录：`/sharedata/datasets/`
- 独立环境：`virtual_env/.venv_data_prep`
- 环境变量：`NAV/config/datasets.env`
- 通用 manifest 扫描器：`NAV/scripts/datasets/build_video_manifest.py`
- 首/中/尾帧有效性检查：`NAV/scripts/datasets/check_video_samples.py`
- Hugging Face 仓库清单审计：`NAV/scripts/datasets/audit_hf_dataset.py`
- Sekai metadata 长度统计：`NAV/scripts/datasets/analyze_sekai_metadata.py`

每个数据集已建立 `raw/videos/annotations/manifests/scripts/logs`，并提供
`README_CN.md` 和可恢复的 `scripts/download.sh`。

## 阶段 A 结果

| 数据集 | 官方数据规模 | 当前本地状态 | 下一步 |
| --- | ---: | --- | --- |
| DL3DV-10K | 480P+pose 810.8 GB；原视频 6.90 TB | 5 个 480P+pose scene 和对应完整视频已下载 | 5条均通过检查并满足16 chunks |
| Sekai | HF 公开仓库 102.0 GB | 5 份 CSV metadata 已下载；18,208 条 HQ pose 已解包并关联 manifest | YouTube 视频下载受登录/反机器人限制 |
| SpatialVID | 7.67 TB，545 groups | 1.0 GB 完整 metadata 已下载并统计 | 最长仅15.04秒，不适合8/16-chunk，暂停视频下载 |
| Ego4D | full-scale 约 7 TB | Ego4D CLI 已安装并验证 | 用户申请 License，并配置 14 天有效的 AWS 凭据 |
| RealEstate10K | 当前旧目录 38 GB | 软链接复用旧数据；没有复制 | 现有 180 episodes 最长 279 帧，需根据官方 ID 增量补抓 |

## Sekai metadata 关键统计

| split | 条目数 | 文件名标注长度 | 可覆盖 16 chunks |
| --- | ---: | ---: | ---: |
| real-walking | 299,173 | 1800 帧 | 299,173 |
| real-walking-hq | 18,208 | 1800 帧 | 18,208 |
| game-walking | 1,618 | 1800 帧 | 1,618 |
| real-drone | 23,912 | 300 帧 | 0 |
| game-drone | 932 | 300 帧 | 0 |

以上长度来自官方文件名的 start/end frame，必须在视频实际下载后再用 ffprobe
复核。walking 适合 16-chunk 训练；drone 的发布 clip 不能直接用于 4-chunk
训练。

## 需要用户完成的授权

### Hugging Face

服务器当前显示 `Not logged in`。需要在浏览器中分别接受：

- <https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P>
- <https://huggingface.co/datasets/SpatialVID/SpatialVID>

接受后在服务器执行：

```bash
hf auth login
```

Token 只输入 Hugging Face CLI，不要发送到聊天、写入脚本或日志。

2026-07-26 已使用账号完成 CLI 登录，并成功访问三个仓库。DL3DV 五个完整视频
均为 1920×1080、约 30 FPS、63.8–68.6 秒，全部通过 ffprobe 与首/中/尾帧
解码检查，并满足 16 chunks。

SpatialVID 的完整 metadata 共 2,715,740 条、总计 7,088.7 小时。实际单 clip
最长只有约 15.04 秒：1,257,389 条满足 4 chunks，满足 8 或 16 chunks的数量
均为 0。因此遵循阶段 A 规范，没有继续盲目下载约 14 GB/group 的视频主体。

## 约 0.5 TB 后台下载任务

2026-07-26 按总量约 0.5 TB、每个数据集约 0.1 TB 的要求建立五个独立脚本，
通过 tmux 后台运行：

| 数据集 | 预算 | 脚本 | tmux session | 当前行为 |
| --- | ---: | --- | --- | --- |
| DL3DV-10K | 100 GB | `NAV/scripts/datasets/download_budgeted/dl3dv_100gb.sh` | `nav_dl3dv` | 下载141个候选scene的原视频和480P+pose，达到预算即停 |
| SpatialVID | 约92 GB | `NAV/scripts/datasets/download_budgeted/spatialvid_100gb.sh` | `nav_spatialvid` | 下载group 0001–0006的视频和annotation，边下载边解包并删除压缩包 |
| Sekai | 100 GB | `NAV/scripts/datasets/download_budgeted/sekai_100gb.sh` | `nav_sekai` | 先下载约90 GB game-walking公开分卷；之后等待cookie补充real-walking |
| Ego4D | 约100 GB | `NAV/scripts/datasets/download_budgeted/ego4d_100gb.sh` | `nav_ego4d` | 等待AWS凭据，之后按约14小时UID子集下载 |
| RealEstate10K | 100 GB | `NAV/scripts/datasets/download_budgeted/re10k_100gb.sh` | `nav_re10k` | 等待YouTube cookie，随后逐视频下载并按实际占用停止 |

五个脚本均自带环境配置、断点续传、日志和预算检查。HF连接失败时每60秒自动
重试。状态检查命令：

```bash
bash /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/datasets/check_budget_downloads.sh
```

日志统一位于：

```text
/sharedata/datasets/<dataset>/logs/budget_download.log
```

需要查看后台终端时使用：

```bash
tmux attach -t nav_dl3dv
tmux attach -t nav_spatialvid
tmux attach -t nav_sekai
tmux attach -t nav_ego4d
tmux attach -t nav_re10k
```

Sekai 和 RealEstate10K 等待的私有 cookie 路径为
`/sharedata/private/youtube_cookies.txt`；该目录权限为700，cookie不得写入
Git、文档或下载日志。Ego4D 等待当前用户的 `~/.aws/credentials`。

### HF 大文件下载优化

2026-07-26 将 DL3DV、SpatialVID 和 Sekai 从 `huggingface_hub` 的单连接
重试切换为：

1. 使用用户私有 HF token 获取 gated 文件的临时签名 URL；
2. 使用 `aria2c` 以 8 个连接分片和断点续传；
3. 签名过期或连接中断后自动刷新 URL；
4. token 不出现在命令行、日志或公共目录。

公共实现为
`NAV/scripts/datasets/download_budgeted/hf_aria2_download.sh`。切换时保留并
迁移了约 147 MB DL3DV、1.69 GB SpatialVID 和 1.68 GB Sekai 的已有断点。
单文件实测速率约 0.6–0.9 MiB/s；主要收益是避免旧 HTTP client 关闭后持续
失败和每次等待 60 秒。SpatialVID `group_0001` annotation 已续传完成。

### Ego4D

按照 <https://ego4d-data.org/docs/start-here/> 接受 License。审批后将临时 AWS
凭据配置到本机 AWS profile。不要把 Access Key 发到聊天或写入仓库。

## 已知下载限制

Sekai-Real 的官方流程仍需从 YouTube 下载原视频。本机直接下载测试收到
`Sign in to confirm you're not a bot`，因此不能在没有合法用户 cookie 的情况下
自动完成视频冒烟样本。不要将浏览器 cookie 提交到 Git 或公共日志。

Sekai 的 `sekai-real-walking-hq.zip` 已确认不是视频包，而是相机轨迹包：
共 18,208 个 NPZ；每个 NPZ 包含 `intrinsic [3,3]` 和
`extrinsic [1800,4,4]`。统一 metadata manifest 位于
`/sharedata/datasets/Sekai/manifests/sekai_real_walking_hq_metadata.jsonl`。
