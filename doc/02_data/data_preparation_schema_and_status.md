# NAV 数据准备、Action/Geometry Schema 与状态

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-010` |
| 类型 | 数据准备、Schema 与状态总览 |
| 状态 | Live / Source of Truth |
| 更新时间 | 2026-08-23 |
| 职责 | 集中维护数据集调研、下载/预处理状态、T4 latent/window 构建、pose/action 标注、VLN 渲染和资源配比。 |

## 当前入口结论

本文件是数据侧唯一主入口。原始下载目录不被训练预处理污染；V1 使用先分块再
VAE latent 化的 T4 micro latent，可复用于不同 history window；Stage3 的 VLN
数据以 simulator 渲染产物进入统一样本组织。自 2026-08-22 起，训练 latent 的
真实落盘位置改为个人项目目录 `NAV/data/train/<dataset>/<latent_run>/`，
公共 `/sharedata` 只保留原始数据、assets、轻量 manifest 和必要临时 render。

2026-08-20 更新：Stage3 VLN 开始准备一个与当前 V1 输入/更新模式一致的
`T_latent=4` chunk latent 子集，组合目标约 **500 GiB latent**。已有 R2R
rendered RGB 可贡献约 `59,056` 个 T4 micro chunks / `44.16 GiB` latent；新增
RxR train budget manifest 选出 `74,584` episodes / `7,800,027` frames /
`609,687` T4 micro chunks / `455.85 GiB` latent。两者组合约 `500.0 GiB`
latent。注意：按已有 480×640 PNG 抽样均值 `260,461 bytes/frame` 估计，新增
RxR raw rendered PNG 中间态约 **1.89 TiB**，而 `/sharedata` 当前剩余约
`1.8T`；因此后续必须优先改成“render 成功并完成 latent 编码后清理 PNG”
或“流式 render→encode→delete frame”的模式，否则 raw RGB 中间态可能先于
latent 目标耗尽磁盘。

2026-08-21 更新：由于 raw rendered PNG 中间态增长过快，已切换为
`render → stream encode T4 latent → 校验 .pt 可读 → 删除 PNG` 的在线清理模式。
当前只删除 `*.png`，保留 `render_meta.json`，并在 episode frame 目录写入
`latent_cleanup.json` 作为可审计标记。R2R existing encoder 暂停，优先把 GPU1
剩余 VAE 显存用于 RxR stream encoder，避免正在增长的 RxR raw PNG 顶满磁盘。

2026-08-22 更新：为避免公共 `/sharedata` 被 latent 挤满，当前 VLN stoppad
stream encoder 已改为写入个人项目目录：

```text
NAV/data/train/rxr_ce/t4_micro_latents_stoppad_20260822_1423/
```

已生成的旧 sharedata stoppad latent 已移动到该目录，encoder 使用同一 run
继续续写；render 仍使用 `/sharedata/NAV/derived/v1/vln/rendered_obs/` 作为临时
PNG 中间态，完成 latent 校验后删除 PNG。

2026-08-22 16:05 更新：按“优先做 R2R train”的调度，已暂停 RxR 500GiB
stoppad 后台 renderer/encoder，保留已生成的 RxR latent 不删除；新建
`r2r_ce:standard:train` 全量 STOP-padding budget，并启动 R2R train 优先准备。
本次 R2R train manifest 覆盖 `10,819 episodes / 1,063,870 frames /
87,345 T4 micro chunks`，预计 latent `65.31 GiB`。latent 写入个人项目目录
`NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/`；instruction
embedding watcher 已挂起，等待 R2R latent encoder 全部退出后自动缓存到
`NAV/data/train/r2r_ce/text_embeddings_stoppad_20260822_1605/`。

2026-08-23 更新：R2R train 三块训练产物已完成：

```text
rendered obs:
  /sharedata/NAV/derived/v1/vln/rendered_obs/
    stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/

T4 micro latent:
  NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/

instruction/text embedding:
  NAV/data/train/r2r_ce/text_embeddings_stoppad_20260822_1605/
```

完成规模为 `10,819 episodes / 1,063,870 rendered frames / 87,345 T4 micro
chunks / 65.35 GiB latent / 10,819 instruction embeddings`。GPU0 当前已空闲；
保留的 `nav_v1_stage3_vln_render_r2r_train_guard` 只是 CPU 磁盘守护窗口，不代表
仍在占卡渲染或编码。

同日对 R2R action sidecar 做训练口径审计：`episode_action_path` 全部存在，
`n_actions` 与 `gt_actions` 长度一致，且每个 episode 恰好有一个 terminal
`STOP`。但当前 action JSON 仍是原始轨迹长度，只在末尾包含单个 `STOP`，没有把
`STOP` 按吸收态重复补齐到 97+ rendered frames。视觉/latent 已做 terminal
observation padding，因此 **Stage3 训练前必须统一 action STOP-padding 口径**：
要么重写/派生 padded action sidecar，要么在 Stage3 dataloader 中按
`render_num_frames` 对 `STOP` 后动作在线补齐；否则当前代码用
`(start_micro + history_micro) * 12` 取 10-step action horizon 时，大量窗口会
变成 `action_loss_mask=0`。

本轮 R2R 后台任务记录（截至 2026-08-23 已完成；仅保留路径追溯）：

| tmux | GPU | 任务 | 输出 / 日志 |
| --- | ---: | --- | --- |
| `nav_v1_stage3_vln_render_r2r_train_gpu0` | 0 | 已完成：R2R train 全量渲染；terminal STOP 后复制 terminal observation 到可构造 policy window | `/sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/`；`log/v1_data_prep/stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605.log` |
| `nav_v1_stage3_vln_encode_r2r_train_gpu0_s0..s5` | 0 | 已完成：6 路 hash-sharded stream encoder 编码 V1 `T_latent=4` micro latent，并在校验后删除 PNG | `NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/`；`log/v1_data_prep/stage3_vln_t4_stream_r2r_train_stoppad_navdata_gpu0_shard*_of6_20260822_1605.log` |
| `nav_v1_stage3_vln_text_r2r_train_after_latent` | 0 | 已完成：R2R instruction UMT5 embedding，共 10,819 个 `.pt` | `NAV/data/train/r2r_ce/text_embeddings_stoppad_20260822_1605/`；实际完成日志 `log/v1_data_prep/stage3_vln_text_r2r_train_stoppad_direct_20260823_1810.log` |
| `nav_v1_stage3_vln_render_r2r_train_guard` | CPU | 可保留/可关闭：磁盘守护窗口；当前不占 GPU | `log/v1_data_prep/vln_render_disk_guard_r2r_train_stoppad_20260822_1605.log` |

已暂停但保留结果的 RxR 任务：

```text
render: NAV/data/train/rxr_ce/t4_micro_latents_stoppad_20260822_1423/
raw rendered tmp: /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_rxr_budget500_stoppad_gpu0_20260822_1423/
```

新增脚本：

```text
scripts/datasets/build_v1_stage3_vln_budget_manifest.py
scripts/datasets/build_v1_stage3_vln_t4_micro_manifest.py
scripts/datasets/encode_v1_stage3_vln_t4_stream.py
scripts/watch_vln_render_disk_guard.sh
```

预算 manifest：

```text
/sharedata/NAV/derived/v1/vln/raw_policy_budgeted/
  stage3_vln_budget_t4_500g_with_existing_r2r_20260820_1259/
    manifests/stage3_vln_episodes.jsonl.gz
    manifests/budget_summary.json
```

R2R T4 micro manifest：

```text
/sharedata/NAV/derived/v1/vln/manifests/
  stage3_vln_t4_micro_r2r_existing_20260820_1830/
    stage3_vln_t4_micro_episodes.jsonl
    summary.json
```

2026-08-17 更新：已新增 motion/action 审计脚本
`scripts/analyze_crabwalk_motion.py`，报告位于
`result/data_audit/crabwalk_motion_20260817.json`。该审计用于确认各数据集是否
真实存在 lateral / strafe / “蟹行”运动，避免盲目把 `move/view` 定义成所有
数据集共享的导航动作。

## 合并主题

- 长视频数据准备状态
- V1 Action 与 Geometry 统一组织 Schema
- NAV 训练数据构建规范
- NAV 相机 Pose 与 Action 标注规范
- VLN 闭环数据集准备状态
- 长视频数据集准备任务书
- 长视频训练数据调研
- Kinetics-400 相机视角变化抽样

## 维护规则

- 后续相关主题优先更新本文档，不再新增细碎同主题文档。
- 历史 run、旧结构和 superseded 方案保留在对应章节中，用于追溯和对照。
- 若代码/日志/文档三线不一致，以代码与真实记录为准先修正文档。

## 合并正文


## 长视频数据准备状态


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-003` |
| 类型 | 动态状态（Operational Status） |
| 状态 | Live |
| 更新时间 | 2026-08-03 |
| 职责 | 汇总下载、标注和预处理的当前状态；历史过程保留在后半部分 |

更新时间：2026-08-07 18:40（Asia/Shanghai）

### 规模与 latent（与 VLN 表同口径）

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

### 当前执行快照

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
  `../03_training/training_plan_and_experiment_log.md`）。

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

### Crab-walk / lateral motion 审计（2026-08-17）

定义：

```text
pure_lateral / 纯蟹行候选:
  move in {3: go left, 4: go right}

strict_crab_walk / 严格蟹行:
  pure_lateral 且 view == 0
  即侧向平移且没有同步转头

lateral_component / 有侧向分量:
  move in {3,4,5,6,7,8}
  包括 diagonal motion
```

审计对象：

- 当前训练用 `stage1_t4_micro_episodes_spatial20.jsonl`
- 全量落盘 `stage1_t4_micro_episodes.jsonl`
- Stage3 R2R/VLN `stage3_vln_policy_chunks.jsonl.gz`

当前训练 manifest 统计：

| 数据集 | episodes | transitions | pure lateral | strict crab | lateral component | diagonal lateral | forward/back | yaw view |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Argoverse2 | 14 | 3,388 | 1.89% | 0.53% | 3.42% | 1.53% | 76.59% | 45.63% |
| DL3DV | 141 | 611,328 | 45.22% | 2.11% | 83.74% | 38.53% | 10.67% | 80.19% |
| RE10K | 269 | 43,309 | 8.04% | 0.12% | 24.30% | 16.27% | 57.83% | 65.82% |
| SpatialVID | 4,756 | 2,345,783 | 2.35% | 0.15% | 9.73% | 7.38% | 89.83% | 73.83% |

全量 manifest 与当前训练 manifest 的比例基本一致：

| 数据集 | episodes | transitions | pure lateral | strict crab | lateral component |
| --- | ---: | ---: | ---: | ---: | ---: |
| Argoverse2 | 14 | 3,388 | 1.89% | 0.53% | 3.42% |
| DL3DV | 141 | 611,328 | 45.22% | 2.11% | 83.74% |
| RE10K | 269 | 43,309 | 8.04% | 0.12% | 24.30% |
| SpatialVID | 23,837 | 11,876,845 | 2.38% | 0.13% | 9.76% |

R2R/VLN action chunk 统计：

```text
rows  = 782,861
steps = 7,828,610
primitive_names:
  MOVE_FORWARD = 4,845,312
  TURN_LEFT    = 1,153,639
  TURN_RIGHT   = 1,090,679
  STOP         = 134,360
  NOOP         = 604,620
lateral_step_ratio = 0.0000%
max_abs_lateral_delta = 0.0
max_abs_forward_delta = 0.25
```

### R2R train action 审计（2026-08-23，Stage3 当前口径）

统计口径：读取 R2R train rendered manifest
`stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/episodes/rendered_episodes.jsonl.gz`，
并按其中的 `episode_action_path` 读取 `gt_actions`；latent micro 数来自
`NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/manifests/`。
Stage3 当前 dataloader 的 action label 起点为
`(start_micro + history_micro) * 12`，`H_action=10`。

R2R 原始 action 分布：

| 指标 | 数值 |
| --- | ---: |
| episodes | 10,819 |
| scenes | 61 |
| missing action sidecar | 0 |
| `n_actions` mismatch | 0 |
| rendered frames mean / median | 98.33 / 97 |
| T4 micro chunks mean / median | 8.07 / 8 |
| action length mean / median | 58.35 / 56 |
| instruction words mean / median | 26.65 / 25 |
| terminal `STOP` episodes | 10,819 / 10,819 |
| multi-STOP episodes | 0 |
| non-zero after first STOP | 0 |
| first STOP index mean / median | 57.35 / 55 |
| rendered frames - action length mean / median | 39.99 / 41 |

原始 `gt_actions` 类别分布：

| 原始 action | count | ratio |
| --- | ---: | ---: |
| `STOP` | 10,819 | 1.71% |
| `MOVE_FORWARD` | 404,912 | 64.15% |
| `TURN_LEFT` | 111,177 | 17.61% |
| `TURN_RIGHT` | 104,336 | 16.53% |

当前正式 combo 映射：

| VLN action | combo_id | 含义 |
| --- | ---: | --- |
| `STOP` | 120 | `trans_id=10, rot_id=0` |
| `MOVE_FORWARD` | 12 | `trans_id=1, rot_id=0` |
| `TURN_LEFT` | 3 | `trans_id=0, rot_id=3` |
| `TURN_RIGHT` | 4 | `trans_id=0, rot_id=4` |

按当前 Stage3 window 构造，R2R train 只能支持 IW1 history：

| IW-equivalent history | history_micro | windows | 有至少 1 个有效 action label | 全 mask windows |
| --- | ---: | ---: | ---: | ---: |
| IW1 | 7 | 11,612 | 1,751 | 9,861 |
| IW4 | 27 | 0 | 0 | 0 |
| IW8 | 54 | 0 | 0 | 0 |
| IW16 | 107 | 0 | 0 | 0 |

解释：R2R stoppad 后每条大多只有 8 个 T4 micro chunks，因此 `history_micro=7`
后只剩 1 个 current obs window；IW4/8/16 的 micro history 长度超过 R2R
episode latent 长度，不能构造。

在“不补齐 STOP，只用当前 action file”的实际代码口径下，IW1 window 的
10-step label 有效数分布为 mean `1.10`、median `0`、p90 `6`、max `10`；
有效 label 内部分布为：

| action | count | ratio |
| --- | ---: | ---: |
| `STOP` | 898 | 7.03% |
| `MOVE_FORWARD` | 8,542 | 66.83% |
| `TURN_LEFT` | 1,724 | 13.49% |
| `TURN_RIGHT` | 1,617 | 12.65% |

如果按项目规则把 terminal `STOP` 视为吸收态，并把 `STOP` 补齐到
`render_num_frames`，则所有 IW1 windows 都有完整 10-step label；但 label 会
高度偏向 STOP：

| action | count | ratio |
| --- | ---: | ---: |
| `STOP` | 104,237 | 89.77% |
| `MOVE_FORWARD` | 8,542 | 7.36% |
| `TURN_LEFT` | 1,724 | 1.48% |
| `TURN_RIGHT` | 1,617 | 1.39% |

训练含义：

1. R2R train 已可作为 Stage3 policy 数据源，但默认只用于短 history / IW1；
   长 history policy 需要 RxR、LHPR 或其它更长导航数据。
2. Stage3 正式训练前必须修正 action STOP-padding；否则大部分 R2R windows
   对 CE 没有监督，训练日志里的 `vln_action_valid` 会偏低。
3. 修正 STOP-padding 后需要采样或 loss reweighting，否则 R2R 的 action label
   会被 terminal STOP 主导。推荐至少记录 non-terminal / terminal window 比例，
   并在 Stage3 sampler 中提高含 `MOVE_FORWARD/TURN_LEFT/TURN_RIGHT` 的窗口比例。
4. R2R 没有 lateral / strafe primitive，仍不支持把 crab-walk 作为默认 policy
   输出类别。

解释：

- DL3DV 的 lateral component 很高，但 top move/view pair 主要是
  `go left + turn right` / `go right + turn left`，更像环绕拍摄或
  camera orbit，不是 VLN agent 可直接执行的 strafe primitive。
- RE10K 有中等比例侧向分量，但 strict crab 极少。
- SpatialVID / Argoverse2 基本以 forward/back 与 yaw view 为主。
- R2R/VLN 完全没有 lateral delta，不支持把 strafe 作为默认 policy output。

训练含义：

```text
视频生成 Stage1/Stage2:
  原始 pose-derived trans/rot 会组合成单一 combo action：
    combo_id = trans_id * 12 + rot_id
  训练样本只暴露一种 action；在视频数据中它用作 a_condition / A_hist / A_cur。
  lateral combo 需要按数据集语义解释，不能一概称作可执行导航动作。

导航 Stage3:
  同一个 combo action space 用作 a_label / A_out supervision。
  R2R/Matterport 映射到 forward/turn/stop/noop 对应的 combo；
  默认不输出 crab-walk/strafe。
```

当前正式 action 字段：

| 数据来源 | action 用途 | 主字段 | shape | 说明 |
| --- | --- | --- | --- | --- |
| 视频生成数据 | condition | `a_hist_combo` | `[S,H]` | 历史 action，参与 Register update |
| 视频生成数据 | condition | `a_cur_combo` | `[H]` | 当前目标 chunk action，控制 future video consequence |
| VLN/R2R 数据 | history input | `a_hist_combo` | `[S,H]` | 过去执行过的导航动作，参与 Register update |
| VLN/R2R 数据 | policy label | `action_combo` | `[H]` | policy 监督标签；模型输出 `combo_logits=[B,H,144]` |

兼容字段：

```text
a_hist_move / a_hist_view / a_cur_move / a_cur_view
  仅用于从 pose-derived trans/rot 审计、以及给 InfiniteWorld native
  action_encoder(move, view) 做 shim。

a_hist_primitives / a_cur_primitives / action_primitives
  保留给旧脚本、旧 checkpoint 和诊断日志；不是当前正式 action schema。
```

### 已完成的公共配置

- 统一数据根目录：`/sharedata/datasets/`
- 独立环境：`virtual_env/.venv_data_prep`
- 环境变量：`NAV/config/datasets.env`
- 通用 manifest 扫描器：`NAV/scripts/datasets/build_video_manifest.py`
- 首/中/尾帧有效性检查：`NAV/scripts/datasets/check_video_samples.py`
- Hugging Face 仓库清单审计：`NAV/scripts/datasets/audit_hf_dataset.py`
- Sekai metadata 长度统计：`NAV/scripts/datasets/analyze_sekai_metadata.py`

每个数据集已建立 `raw/videos/annotations/manifests/scripts/logs`，并提供
`README_CN.md` 和可恢复的 `scripts/download.sh`。

### 阶段 A 结果

| 数据集 | 官方数据规模 | 当前本地状态 | 下一步 |
| --- | ---: | --- | --- |
| DL3DV-10K | 480P+pose 810.8 GB；原视频 6.90 TB | 5 个 480P+pose scene 和对应完整视频已下载 | 5条均通过检查并满足16 chunks |
| Sekai | HF 公开仓库 102.0 GB | 5 份 CSV metadata 已下载；18,208 条 HQ pose 已解包并关联 manifest | YouTube 视频下载受登录/反机器人限制 |
| SpatialVID | 7.67 TB，545 groups | 1.0 GB 完整 metadata 已下载并统计 | 最长仅15.04秒，不适合8/16-chunk，暂停视频下载 |
| Ego4D | full-scale 约 7 TB | Ego4D CLI 已安装并验证 | 用户申请 License，并配置 14 天有效的 AWS 凭据 |
| RealEstate10K | 当前旧目录 38 GB | 软链接复用旧数据；没有复制 | 现有 180 episodes 最长 279 帧，需根据官方 ID 增量补抓 |

### Sekai metadata 关键统计

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

### 需要用户完成的授权

#### Hugging Face

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

### 约 0.5 TB 后台下载任务

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

#### HF 大文件下载优化

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

#### Ego4D

按照 <https://ego4d-data.org/docs/start-here/> 接受 License。审批后将临时 AWS
凭据配置到本机 AWS profile。不要把 Access Key 发到聊天或写入仓库。

### 已知下载限制

Sekai-Real 的官方流程仍需从 YouTube 下载原视频。本机直接下载测试收到
`Sign in to confirm you're not a bot`，因此不能在没有合法用户 cookie 的情况下
自动完成视频冒烟样本。不要将浏览器 cookie 提交到 Git 或公共日志。

Sekai 的 `sekai-real-walking-hq.zip` 已确认不是视频包，而是相机轨迹包：
共 18,208 个 NPZ；每个 NPZ 包含 `intrinsic [3,3]` 和
`extrinsic [1800,4,4]`。统一 metadata manifest 位于
`/sharedata/datasets/Sekai/manifests/sekai_real_walking_hq_metadata.jsonl`。


## V1 Action 与 Geometry 统一组织 Schema


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| ID | NAV-DAT-008 |
| 类型 | 数据与实现 Schema |
| 状态 | Proposed / V1 Draft |
| 更新时间 | 2026-08-10 |
| 职责 | 定义 V1 中 video pseudo action、VLN action、previous executed action、3D geometry target 和模型 forward 所需样本的统一组织形式 |

### 核心结论

V1 最终组织为四个对象：

```text
ActionStep / ActionChunk:
  统一 video pseudo action、VLN GT action、previous executed action 的表示。

VaePack:
  统一 [obs, future_1..future_4] sparse pack 的 Wan VAE latent。

GeometryTarget:
  统一 pose / depth / point map / correspondence 等 3D supervision。

V1Sample:
  dataloader 输出给模型 forward 的完整样本。
```

关键语义：

```text
current/target action:
  是 shared WanBlock 内 action tokens 要生成的变量，不是 policy condition。

previous executed action:
  是历史中已经发生的事件，可用于 Register update。

pose:
  是训练 teacher / pseudo action / 3D supervision 来源，不是推理输入。

future visual / future 3D:
  是 generation branch / 3D probe 的监督目标，不是 policy condition。
```

### ActionStep

每个低层动作统一为：

```python
ActionStep = {
    "source": "video_pseudo | vln_gt | robot_executed | pseudo_motion",
    "primitive_id": int,
    "primitive_name": str,
    "delta_ego": [dx, dy, dz, dyaw, dpitch, droll],
    "trans_magnitude": float,
    "rot_magnitude": float,
    "trans_bucket": int,
    "rot_bucket": int,
    "valid": bool,
    "is_stop": bool,
    "confidence": float,
    "scale_type": "metric | simulator | normalized | pseudo | unknown"
}
```

#### primitive_id

第一版 primitive 统一到一个大表：

| ID | 名称 | 来源 |
| ---: | --- | --- |
| 0 | `NOOP` | video / padding |
| 1 | `STOP` | VLN |
| 2 | `MOVE_FORWARD` | VLN / video |
| 3 | `MOVE_BACKWARD` | video |
| 4 | `STRAFE_LEFT` | video / robot optional |
| 5 | `STRAFE_RIGHT` | video / robot optional |
| 6 | `TURN_LEFT` | VLN / video |
| 7 | `TURN_RIGHT` | VLN / video |
| 8 | `LOOK_UP` | VLN optional / video |
| 9 | `LOOK_DOWN` | VLN optional / video |
| 10 | `COMPOSITE` | video diagonal / mixed motion |
| 11 | `UNCERTAIN` | low confidence pseudo action |

VLN 数据通常只用：

```text
STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / optional LOOK_UP / LOOK_DOWN
```

视频数据可以使用完整 primitive 表，并额外监督 continuous delta。

#### delta_ego

`delta_ego` 是相机/agent 局部坐标下的连续运动：

```text
dx, dy, dz:
  egocentric translation

dyaw, dpitch, droll:
  egocentric rotation, radians
```

有 metric scale：

```text
meter / radian
```

无 metric scale：

```text
translation 用 episode median motion 归一化
rotation 仍用 radian 或归一化 rotation
```

VLN 离散动作可映射为近似 delta：

```text
MOVE_FORWARD ≈ [0, 0, +0.25m, 0, 0, 0]
TURN_LEFT    ≈ [0, 0, 0, +30°, 0, 0]
TURN_RIGHT   ≈ [0, 0, 0, -30°, 0, 0]
STOP         ≈ zero delta + is_stop
```

该 delta 可作为 auxiliary regression target 或尺度对齐审计，不要求 Stage Three
推理时输入。

### ActionChunk

模型不直接处理裸 `ActionStep`，而处理一个 horizon chunk：

```python
ActionChunk = {
    "steps": List[ActionStep],   # length = H_action or H_nav
    "horizon": int,
    "valid_mask": BoolTensor[H],
    "source": str,
    "summary": {
        "delta_total": [dx, dy, dz, dyaw, dpitch, droll],
        "dominant_primitive_id": int,
        "confidence": float
    }
}
```

第一版默认：

```text
H_nav = 4
H_video ∈ {4,16,32,48}
```

模型内部建议：

```text
step action tokens:
  用于 policy/action loss。

summary action token:
  可选，用于 generation branch 读取整体 motion intent。
```

若第一版想最小化实现，可以先不显式存 summary token，在 `ActionStem` 内由 step
tokens pooling 得到 summary hidden。

### VaePack

V1 generation branch 使用 sparse future pack：

```python
VaePack = {
    "sample_id": str,
    "dataset": str,
    "episode_id": str,
    "frame_indices": [t, f1, f2, f3, f4],
    "z_obs": Tensor[16, 1, H_lat, W_lat],
    "z_future": Tensor[16, 1, H_lat, W_lat],
    "rgb_paths": List[str],
    "confidence": float
}
```

默认 896×448 时：

```text
H_lat × W_lat = 56 × 112
Z_obs         = [16,1,56,112]
Z_future      = [16,1,56,112]
```

`z_future T=1` 是 one future latent time cell，不是一帧 RGB。

### GeometryTarget

3D supervision 单独组织，不进入 policy condition：

```python
GeometryTarget = {
    "teacher_source": "gt_pose | sparse_pose_interp | vggt | simulator | none",
    "intrinsics": Tensor[N, 3, 3],
    "poses_c2w": Tensor[N, 4, 4],
    "relative_poses": Tensor[N-1, 4, 4],
    "depth": Optional[Tensor[N, H_g, W_g]],
    "point_map": Optional[Tensor[N, H_g, W_g, 3]],
    "correspondence": Optional[Any],
    "confidence": Optional[Tensor],
    "scale_type": "metric | simulator | normalized | pseudo | unknown",
    "valid_mask": Tensor
}
```

来源优先级：

```text
1. GT pose / simulator pose
2. sparse pose interpolation
3. VGGT pseudo camera/depth/point map
4. optical flow / correspondence pseudo label
```

注意：

Stage Two 中，`GeometryTarget` 默认绑定到 current/local chunk，即预测
`C_t` 时的 `C_{t-1}`。Register history 只包含 `C_0...C_{t-2}`，不包含
`C_{t-1}`，这样 current chunk 的空间 token/grid hidden 可以作为 dense 3D
probe 的主要读出窗口。

参考 VGGT-Ω，3D probe 不应只读取一个全局 pooled register。推荐最小结构是：

```text
current hidden grid -> depth / point map / confidence
register + pooled current hidden -> relative pose / camera motion
```

候选监督：

```text
depth_current:
  [B, T_cur, H_g, W_g]

point_map_current:
  [B, T_cur, H_g, W_g, 3]

relative_pose_current:
  [B, T_cur-1, 4, 4] 或 compact 6/7/9D pose encoding

valid_mask / confidence:
  [B, T_cur, H_g, W_g]
```

其中 dense depth/point 来自 simulator/GT depth 或 VGGT/VGGT-Ω pseudo label；
relative pose 来自 dataset/simulator pose 或 sparse pose interpolation。全部只用于
训练监督，不作为模型推理输入。

注意：

```text
GeometryTarget 只作为 loss target。
不得作为 Stage Three policy input。
```

### V1Sample

Dataloader 最终输出：

```python
V1Sample = {
    "mode": "stage1_video | stage2_3d | stage3_vln",
    "dataset": str,
    "sample_id": str,

    "vae_pack": Optional[VaePack],
    "z_obs": Tensor,
    "z_future": Optional[Tensor],
    "z_future_noise": Optional[Tensor],
    "visual_timestep": Optional[Tensor],

    "register_history": {
        "history_obs": Optional[List[Any]],
        "previous_action_chunks": Optional[List[ActionChunk]],
        "history_mask": Tensor
    },

    "target_action_chunk": ActionChunk,
    "action_timestep": Optional[Tensor],

    "instruction": Optional[str],
    "text_tokens": Optional[Tensor],

    "geometry_target": Optional[GeometryTarget],

    "loss_mask": {
        "visual": bool,
        "action_ce": bool,
        "action_delta": bool,
        "geometry": bool,
        "nav": bool,
        "progress": bool
    }
}
```

Stage Three policy-only 时：

```text
z_future = None
z_future_noise = None
visual loss mask = false
generation branch 不执行或只作为 optional auxiliary
```

### 模型 forward 接口

推荐实现：

```python
outputs = model.forward_v1(
    z_obs=z_obs,
    register_state=R_t,
    action_query_or_noisy=action_tokens,
    text_tokens=text_tokens,
    z_future_noisy=z_future_noisy,          # Stage One/Two only
    previous_action=previous_action_tokens, # optional, history only
    timesteps={
        "visual": visual_t,
        "action": action_t,
    },
    masks=masks,
    mode=mode,
)
```

明确禁止：

```python
model.forward_v1(..., target_action_clean_as_condition=...)
model.forward_v1(..., gt_pose_as_policy_condition=...)
model.forward_v1(..., future_visual_as_policy_condition=...)
```

### 模型内部模块

当前推荐代码结构：

```text
RegisterCell:
  R_{-1}=R_null fixed template
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))

VisualStem:
  R_t, Z_obs, Z_future_noisy -> shared WanBlock visual tokens

ActionStem:
  A_hist / A_cur / A_noise -> shared WanBlock action tokens

TextConditioner:
  instruction/text -> condition tokens

Single Shared WanBlock Backbone:
  visual tokens + action tokens + text/condition tokens
  with explicit attention mask

GenerationHead:
  future tokens -> predicted noise/velocity(Z_future)

ActionFlowDecoder:
  shared WanBlock action-token hidden -> action velocity / primitive logits

GeometryProbe:
  selected hidden/Register -> depth/pose/point/correspondence prediction
```

当前实现策略：

```text
v1:
  unified RegisterCell
  single shared WanBlock token stream
  explicit policy-safe attention relation
  no action/latent additive bias
```

### Loss 接口

#### Stage One

```text
L_stage1 =
  L_visual_flow(Z_future)

Stage One 保留 A_noise / A_out 格式，但不计算 action supervision。
```

#### Stage Two

```text
L_stage2 =
  L_stage1
+ λ_3d L_geometry
```

#### Stage Three

```text
L_stage3 =
  L_action_flow
+ λ_ce CE(action primitive)
+ λ_replay_video L_visual_flow_replay
+ λ_replay_3d L_3D_replay
```

### 落盘组织

```text
/sharedata/NAV/derived/v1/
  manifests/
    source_<dataset>.jsonl
    stage1_video_packs.jsonl
    stage2_3d_packs.jsonl
    stage3_vln_policy.jsonl

  actions/
    <dataset>/<episode_id>.jsonl

  vae_packs/
    <dataset>/<sample_id>.pt

  geometry/
    <dataset>/<sample_id>.pt

  vln/
    rendered_obs/
    policy_chunks/

  audits/
    action_distribution_<dataset>.json
    displacement_alignment.json
    geometry_confidence_<dataset>.json
```

### 当前实现状态（2026-08-10）

已建立 V1 数据准备代码：

```text
NAV/src/nav/v1/schema.py
NAV/src/nav/v1/models/{masks,stems,heads}.py
NAV/src/nav/v1/models/full_model.py
NAV/scripts/datasets/build_v1_video_pack_manifest.py
NAV/scripts/datasets/encode_v1_vae_packs.py
NAV/scripts/smoke_v1_full_pipeline.py
NAV/config/v1_data_prep.yaml
```

当前已生成：

```text
/sharedata/NAV/derived/v1/manifests/stage1_video_packs.jsonl
```

统计：

| 数据集 | V1 pack samples |
| --- | ---: |
| RE10K | 2,152 |
| DL3DV | 1,128 |
| SpatialVID | 190,696 |
| Argoverse2 | 112 |
| Total | 194,088 |

该自然分布被 SpatialVID 主导；正式训练必须使用 weighted sampling，不能按 JSONL
自然顺序直接训练。

VAE smoke 已验证 1 条 RE10K pack：

```text
input RGB pack:
  [obs_t, future_1, future_2, future_3, future_4]

output:
  z_obs    [16,1,56,112]
  z_future [16,1,56,112]
```

产物：

```text
/sharedata/NAV/derived/v1/vae_packs_debug_smoke/re10k/
```

HDF5 shard smoke 也已验证：

```text
/sharedata/NAV/derived/v1/vae_packs_hdf5_smoke/
```

读取结果：

```text
len = 1
z_obs    = [16,1,56,112]
z_future = [16,1,56,112]
action_primitive = [0,3,3,3]
```

完整模型 full-pipeline smoke：

```text
bash NAV/scripts/run_v1_full_pipeline_smoke.sh cpu
```

结果：

```text
report:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json

stage checks:
  Stage One   = video generation loss + backward/update
  Stage Two   = Stage One + 3D probe loss + backward/update
  Stage Three = action flow/CE + video/3D rehearsal + backward/update

inference:
  videogen_z_future       = [2,16,2,8,8]
  policy_action_chunk     = [2,10,6]
  policy_primitive_logits = [2,10,12]
```

### 代码组织建议

```text
NAV/src/nav/v1/schema.py
NAV/src/nav/v1/data/action_schema.py
NAV/src/nav/v1/data/video_pack_dataset.py
NAV/src/nav/v1/data/vln_policy_dataset.py
NAV/src/nav/v1/models/register.py
NAV/src/nav/v1/models/stems.py
NAV/src/nav/v1/models/dit_wrapper.py
NAV/src/nav/v1/models/heads.py
NAV/src/nav/v1/losses.py

NAV/scripts/datasets/build_v1_action_manifest.py
NAV/scripts/datasets/build_v1_video_pack_manifest.py
NAV/scripts/datasets/encode_v1_vae_packs.py
NAV/scripts/datasets/build_v1_geometry_targets.py
NAV/scripts/datasets/build_v1_vln_policy_manifest.py
```


## NAV 训练数据构建规范


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-005` |
| 类型 | 数据规范（Data Construction Specification） |
| 状态 | Active / Source of Truth |
| 更新时间 | 2026-08-10 |
| 职责 | 定义从落盘视频、Pose、Action 到多 Chunk VAE latent 的唯一数据语义 |

### 当前结论

NAV 的训练数据不是预先计算好的 Register，而是连续视频 chunk、逐帧 action 和
可复用 VAE latent。Register 是随模型参数变化的网络内部状态，只能在训练或推理
forward 中在线递归计算。

标准数据链如下：

```text
只读原始视频/图像 + 相机 Pose
          │
          ├──► 稠密 Pose → 相邻帧 SE(3) → move/view action
          │
          └──► 连续 RGB 帧 → Wan VAE → chunk latent
                                      │
                                      ▼
训练时加载连续 latent/action → 在线更新 Register → 预测下一 chunk
```

原始下载目录只读。所有可重新生成的派生物写入 `/sharedata/NAV/derived/`。

### Chunk 与时间语义

每个视频 chunk 固定包含 81 个连续 RGB 帧，空间分辨率统一为
`448×896`。Wan2.1 VAE 将其编码为：

```text
RGB:    [B, 3, 81, 448, 896]
latent: [B, 16, 21, 56, 112]
```

因此本文档中的“3 chunks”始终指243个 RGB 帧和63个 VAE 时间位置，而不是
3帧或3个 latent token。不同 episode 的余数帧不能跨视频拼接；有效 chunk 数为：

\[
N_{\mathrm{chunk}}=\left\lfloor N_{\mathrm{covered\ frames}}/81\right\rfloor
\]

每个 chunk 必须独立调用一次 VAE encoder，并在 chunk 边界重置 temporal
context。一个 episode 的全部完整 chunks 一次性缓存；训练阶段改变 history
长度时只做 latent 切片，不重新编码 RGB。

末尾不足81帧的余数直接丢弃，不补帧、不与其他 episode 拼接。canonical v1
产物为：

```text
/sharedata/NAV/derived/
├── manifests/full_episodes_v1.jsonl
├── actions/full_episodes_v1/<dataset>/<sample_id>.json
└── latents/full_episodes_v1/<dataset>/<sample_id>.pt
```

### 离线与在线边界

#### 可以离线缓存

- 连续 RGB frame indices 或 image paths；
- 每个 chunk 的干净 VAE latent；
- 与目标 RGB 帧一一对应的 `move/view`；
- caption/text embedding；
- 数据集名、episode ID、窗口起点和原始 chunk 数。

#### 不能离线缓存

- Register state；
- 由当前 Register updater 产生的 history summary；
- 随 optimizer step 改变的 DiT hidden states 或 KV Cache；
- 将旧 checkpoint 计算出的 Register 当作新 checkpoint 的训练输入。

Register 满足：

\[
R_{i+1}=U_\theta(R_i,Z_i)
\]

其中参数 \(\theta\) 每次 optimizer step 都会变化，所以缓存 \(R_i\) 会产生过期
状态，并切断训练所需的反向传播图。

### 训练窗口选择

离线阶段不再为 Short/Medium/Long 选择窗口。训练 loader 从完整 episode cache
中按固定 seed 随机选择连续窗口，窗口最后一个 chunk 是 diffusion target，
此前 chunks 是 teacher-forced history。由此同一份 cache 可支持2–3、4–7、
8+ chunks以及之后的超长 history curriculum。

全量准备顺序固定为 `DL3DV → RE10K → SpatialVID → Argoverse2`。DL3DV
作为长程工程验证的基础数据优先完成；其141条完整 episode 全部原子落盘后写入：

```text
/sharedata/NAV/derived/stage_ready/full_episodes_v1_dl3dv.ready
```

该 marker 出现后即可使用 DL3DV 启动长 history 测试，其他数据集继续在后台编码。

### 各数据集对齐规则

| 数据集 | RGB来源 | Pose来源 | 当前可靠语义 |
| --- | --- | --- | --- |
| RE10K | `processed_frames/frames` | 原始逐帧相机外参 | 基本逐帧，部分短序列均匀重采样 |
| SpatialVID | 官方 MP4 | `poses.npy + indexes.txt` | sparse pose 插值覆盖首末标注点之间 |
| DL3DV | 原始 `video.mp4` 连续帧 | `transforms.json` 稀疏 SfM Pose | 稀疏 Pose 首尾均匀映射到完整 clip，旋转 Slerp、平移线性插值 |
| Argoverse 2 | `ring_front_center/*.jpg` | ego trajectory + camera extrinsic | 每个相机时间戳匹配最近 ego pose |
| Kinetics-400 | MP4 | VGGT估计 | 当前只有episode级统计，不进入逐帧Action训练 |
| Sekai | 视频尚未完整落盘 | 官方相机轨迹 | 等视频与轨迹文件级配对后再进入 |
| Ego4D | 尚未落盘 | 无当前可用逐帧导航Pose | 不进入当前训练 |

DL3DV 特别禁止把 `frame_00001...` 直接解释为原 MP4 的第1帧、第2帧。旧统一
缓存存在该风险，新结果隔离写入 `latents_pose_aligned/`。

### 派生数据格式

#### Action JSON/manifest

必要字段：

```text
sample_id, dataset, episode_id/video_path
frame_indices 或 frame_paths
move, view
num_frames, num_chunks, chunk_frames
pose_label_calibration
```

SpatialVID Short 还记录：

```text
source_num_frames, source_num_chunks
window_start_chunk, cached_chunks
annotation_audit
```

#### Latent PT

标准 payload：

```python
{
    "sample_id": str,
    "dataset": str,
    "chunks": Tensor[B, N, 16, 21, 56, 112],
    "move": LongTensor[N * 81],   # 或从外部 action sidecar 加载
    "view": LongTensor[N * 81],
    "num_chunks": int,
    "source_num_chunks": int,
}
```

SpatialVID 新缓存为实现 action/latent 并行，允许 PT 不内嵌 `move/view`，训练
时通过 `--action-cache` 按 `sample_id` 加载 JSON。Register 不属于 payload。

### 当前派生目录

```text
/sharedata/NAV/derived/
├── actions/spatialvid_short/
├── manifests/
│   ├── episodes.jsonl
│   └── remaining_pose_short.jsonl
├── latents/spatialvid/
└── latents_pose_aligned/
    ├── re10k/
    ├── dl3dv/
    └── argoverse2/
```

`latents/spatialvid/` 包含历史缓存和当前增量缓存；训练 loader 优先使用 PT 内嵌
action，没有时读取外部 JSON。`latents_pose_aligned/` 是修正帧/Pose 对齐后的
新目录，不得用旧 DL3DV cache 覆盖。

### V1 T4 micro chunk 语义

2026-08-10 起，V1 Stage One 主线不再把 generation target 固定为81-frame
dense chunk，也不沿用早期 sparse pack 的 `Z_future T=1`。新版默认：

```text
micro_frames = 13 RGB frames
micro_stride = 12 RGB frames
Wan VAE latent_t = 4
target latent = [B,16,4,56,112]
```

相邻 micro chunk 共享1帧边界，对齐 InfiniteWorld rollout 中每个 chunk 只新增
80帧的语义。由此：

```text
IW-1  history span -> 7   个 T4 micro history steps
IW-4  history span -> 27  个 T4 micro history steps
IW-8  history span -> 54  个 T4 micro history steps
IW-16 history span -> 107 个 T4 micro history steps
```

T4 派生物独立写入，不覆盖旧缓存：

```text
/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes.jsonl
/sharedata/NAV/derived/v1/t4_micro_latents/
```

每个 T4 micro chunk 必须从对应13帧 RGB 独立调用 Wan VAE encoder。不得先对
81帧或整段视频编码后再切出4个 latent cells 作为训练 target；这样会引入 VAE
temporal context 的训推边界不一致。

### 完整性与可复现要求

进入训练前至少验证：

1. `sample_id` 唯一；
2. chunk tensor 可加载、有限且非全零；
3. `chunks.shape[1] == cached_chunks`；
4. `len(move) == len(view) == cached_chunks × 81`，或外部 sidecar 能提供；
5. RGB/Pose 时间顺序严格一致；
6. 每条样本来自单一 episode；
7. 随机窗口由固定 seed 或稳定哈希决定；
8. 原始下载目录没有被预处理修改。

### 当前限制与后续必须完成

- 尚未建立正式 scene-level train/validation/test split，正式指标前必须补齐；
- SpatialVID 数量占绝对多数，训练必须进行 dataset-aware sampling；
- Action 是相机轨迹伪标签，不代表真实键盘或机器人控制；
- 跨数据集的尺度归一化当前按 episode 中位运动计算，语义一致但绝对速度不可比；
- Short latent 不是完整长视频缓存，后续不能直接据此宣称完成 Long curriculum；
- 训练前应冻结数据版本清单、样本数、哈希规则和异常排除列表。

相关标注定义见 `data_preparation_schema_and_status.md`，在线训练语义见
`../03_training/training_plan_and_experiment_log.md`。


## NAV 相机 Pose 与 Action 标注规范


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-006` |
| 类型 | 标注规范（Annotation Specification） |
| 状态 | Active / Source of Truth |
| 更新时间 | 2026-07-29 |
| 职责 | 定义稀疏 Pose 对齐、插值、相邻帧运动和 Infinite-World Action 离散化 |

### 标注目标

NAV 将相机轨迹转为与每个 RGB 帧一一对应的两条离散控制序列：

```text
move[t] ∈ {0,...,9}
view[t] ∈ {0,...,9}
```

它们是由 camera pose 推导的 pseudo-action，不是数据集原生用户输入。用途是让
Infinite-World 的原 ActionEncoder 在无真实键盘动作的数据上获得近似相机控制。

### Pose 规范化

所有输入最终统一为每帧3×4 world-to-camera：

\[
P_t=[R_t\mid t_t]
\]

若数据提供 camera-to-world 4×4 矩阵，先求逆再截取前3行。相机中心为：

\[
c_t=-R_t^\top t_t
\]

相邻帧在前一相机坐标系中的平移为：

\[
\Delta p_t=R_t(c_{t+1}-c_t)
\]

相对旋转为：

\[
\Delta R_t=R_{t+1}R_t^\top
\]

坐标约定为相机 \(x\) 向右、\(-z\) 向前。RE10K 的非严格旋转矩阵先通过 SVD
投影到最近的 \(SO(3)\)。

### 稀疏 Pose 变稠密

对于两个带标注帧 \(i,j\)：

- Translation：按原视频帧号线性插值；
- Rotation：使用 quaternion `Slerp`；
- 超出首末标注范围的帧不应作为可靠覆盖；当前窗口选择限制在可用范围；
- 重复 timestamp 先稳定排序并去重；
- `poses.npy` 与 `indexes.txt` 长度不一致时按最短长度配对，并记录
  `annotation_audit.length_mismatch=true`。

插值只表达“相邻标注间相机平滑运动”的近似，不能恢复中间的快速抖动、急转或
Pose tracking failure。未来应把标注间隔、置信度和旋转突变写为 sample weight。

### 连续运动到离散 Action

在每个 episode 内计算：

```text
translation magnitude = norm([Δx, Δz])
rotation magnitude    = hypot(yaw, pitch)
```

使用所有非零相邻帧变化的中位数作为 episode 内尺度：

```text
translation_median
rotation_median_rad
```

归一化后：

- ratio `< 0.25`：`no-op`；
- ratio `> 3.0`：`uncertain`；
- 其余按主方向或对角方向离散；
- 对角判定阈值为 `0.414≈tan(22.5°)`。

类别定义：

| ID | Move | View |
| ---: | --- | --- |
| 0 | no-op | no-op |
| 1 | go forward | turn up |
| 2 | go back | turn down |
| 3 | go left | turn left |
| 4 | go right | turn right |
| 5 | forward + left | up + left |
| 6 | forward + right | up + right |
| 7 | back + left | down + left |
| 8 | back + right | down + right |
| 9 | uncertain | uncertain |

第一帧没有前驱，默认标为0。随机窗口必须先在完整 episode 上计算相邻运动，再
切片；否则窗口首帧会被错误重置为 `no-op`。

### 数据集特例

#### SpatialVID

`indexes.txt` 第二列是原视频帧号，末项对应原视频最后一帧。Pose 以
`[tx,ty,tz,qx,qy,qz,qw]` 存储。全 episode 插值、标注和 calibration 完成后
再切确定性随机窗口。

#### DL3DV

Pose 来自 `transforms.json`，每条与 `images_8/frame_*.png` 一一对应。当前不
假设它与原 MP4 帧号存在直接对应关系，也不对原 MP4 全帧生成伪 action。

#### Argoverse 2

将 `city_SE3_egovehicle` 与 `egovehicle_SE3_sensor` 组合为相机轨迹。每个
`ring_front_center` 图像 timestamp 匹配最近的高频 ego pose。

#### RE10K

使用原逐帧外参；部分不足162帧的序列均匀重采样，因此可能出现重复 RGB 帧和
零位移 action。相关样本必须保留 `temporally_resampled` 标记。

### 已知分布风险

早期 Short cache 审计显示：

- SpatialVID 占样本绝大多数；
- `move=2` 在旧 SpatialVID 标签中占比异常高；
- DL3DV 的动作类别分布与 RE10K/SpatialVID 明显不同；
- 跨数据集坐标、Pose方向或轨迹质量错误会表现为类别单边倒。

因此每次冻结新数据版本都必须输出：

1. 各数据集 `move/view` 直方图；
2. no-op 与 uncertain 比例；
3. translation/rotation calibration 分布；
4. 插值间隔和长度异常数量；
5. 随机可视化若干轨迹及对应视频；
6. 跨数据集方向 sanity check。

只验证 tensor 可加载不足以证明 Action 语义正确。

### 当前脚本

| 职责 | 脚本 |
| --- | --- |
| 通用 Pose/Action manifest | `NAV/scripts/prepare_multidataset_manifest.py` |
| RE10K窗口与动作 | `NAV/scripts/prepare_re10k_manifest.py` |
| V1 T4 micro episode manifest | `NAV/scripts/datasets/build_v1_t4_micro_manifest.py` |
| V1 full-pipeline 模型链路验证 | `NAV/scripts/smoke_v1_full_pipeline.py` |

数据构建总规范见 `data_preparation_schema_and_status.md`。


## VLN 闭环数据集准备状态


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-007` |
| 类型 | 动态状态（Operational Status） |
| 状态 | Live |
| 更新时间 | 2026-08-07（Asia/Shanghai） |
| 职责 | 汇总 R2R-CE、RxR-CE、LHPR-VLN、ScaleVLN 在 `/sharedata/datasets/` 的落盘、核实与阻塞项 |

更新时间：2026-08-07 18:40（Asia/Shanghai）

### 规模与存储估算（2026-08-07 实测）

统一口径：

- **帧**：导航为 GT `actions` 长度（逐步观测）；LHPR long-horizon 取各 task 最长 trial 的 `action` 数；LHPR step_task 为 `end-start+1`。
- **可构造 chunk**：`floor(帧/81)`（不足 81 记 0）；1 chunk = 81 RGB → Wan VAE latent `[16,21,56,112]` **float32 = 8.43 MB**。
- **导航渲染后大小**：按 smoke（640×480）外推，PNG ≈135 KB/帧，raw RGB ≈0.92 MB/帧；本地尚无大规模 RGB。
- **关键事实**：多数 CE 逐步轨迹 **<81 帧**，平均完整 chunk ≪ 1；若 Stage Three 要 ≥2 chunks（≥162 帧），需更长轨迹、插值/更高频观测，或改 chunk 语义。

| 数据集 | 条数 | 平均帧（中位） | 平均完整 chunk | ≥1 / ≥2 chunk 条数 | 3.a 渲染后每条均值 PNG / raw | 4. 每条 latent 预期（个数 × 8.43 MB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R-CE standard | 16,844 | 58.3（56）* | 0.12 | 1,537 / 9 | 7.9 / 53.7 MB | 0.12 → **0.97 MB** |
| R2R-CE EnvDrop | 146,304 | 67.1（64） | 0.23 | 32,406 / 1,625 | 9.1 / 61.8 MB | 0.23 → **2.0 MB** |
| RxR-CE guide | 87,609 | 93.5（85）* | 0.66 | 41,507 / 8,720 | 12.6 / 86.1 MB | 0.66 → **5.6 MB** |
| RxR-CE follower | 76,170 | 109.1（94） | 0.86 | 44,821 / 13,968 | 14.8 / 100.6 MB | 0.86 → **7.2 MB** |
| ScaleVLN-CE 150k | 155,098 | 48.5（48） | ≈0.001 | 154 / 54 | 6.6 / 44.7 MB | ≈0 → **~0** |
| LHPR long-horizon | 1,052 | 84.4（80.5） | 0.53 | 526 / 29 | 11.4 / 77.8 MB | 0.53 → **4.5 MB** |
| LHPR step_task | 2,290 | 57.3（51） | 0.24 | 541 / 17 | 7.8 / 52.8 MB | 0.24 → **2.1 MB** |

\*R2R / RxR guide 长度统计来自有 GT 的 train+val；条数含 test。LHPR `batch_*.zip` 仍在续传（`batch_1` incomplete ≈9.8 GB，HF 频繁断连）。

粗总量（仅 PNG 渲染、核心闭环不含 EnvDrop/follower）：约 **16.8k×7.9 + 87.6k×12.6 + 155k×6.6 + 1.0k×11.4 ≈ 2.3 TB**；严格完整-chunk latent 则因短轨迹而远小于该量级。

ScaleVLN 离散增强 `R2R_scalevln_ft_aug_enc.json` 另有约 **2.89M** 条（图节点路径，非 Habitat 低层帧），未并入上表。

长视频线（SpatialVID / DL3DV / RE10K / AV2 / Sekai / Kinetics）见
`data_preparation_schema_and_status.md` 同期「规模与 latent」小节。

### 当前结论

四个闭环 VLN 数据集已按 NAV 标准目录落盘到 `/sharedata/datasets/`。
**R2R-CE、RxR-CE、ScaleVLN（含 CE 150k 子集）均已核实可用**；LHPR-VLN
任务标注已就绪，离线 batch 观测包仍在后台续传。

| 数据集 | 公共目录 | 当前占用 | 状态 | 当前可用内容 |
| --- | --- | ---: | --- | --- |
| R2R / R2R-CE | `/sharedata/datasets/R2R` | 505 MB | verified_local | `R2R_VLNCE_v1-3` + preprocessed；train 10819 / val_unseen 1839 |
| RxR / RxR-CE | `/sharedata/datasets/RxR` | 699 MB | verified_ce_ready；原始 GCS 阻塞 | `RxR_VLNCE_v0` guide：train 60300 / val_unseen 11006；StreamVLN RxR sidecar 已落盘 |
| LHPR-VLN | `/sharedata/datasets/LHPR-VLN` | 续传中 | 任务就绪；batch 续传 | `task`/`step_task`/`episode_task` 已解压；batch_1..8 后台下载 |
| ScaleVLN | `/sharedata/datasets/ScaleVLN` | ~50+ GB | verified_local | 预处理标注 + features 35GB + rvr 8.5GB + CE 150k 子集 |
| MP3D Habitat（共用场景） | `/sharedata/datasets/mp3d` | 34 GB | 已有 | R2R-CE / RxR-CE 共用 |

统计口径：占用为 `du -sh` 目录总量；episode 数来自各 split 的
`*guide*.json.gz` 内 `episodes` 字段计数（脚本
`NAV/scripts/datasets/verify_vln_four.py`）。

### 唯一路径与入口

| 用途 | 路径 |
| --- | --- |
| 公共根目录 | `/sharedata/datasets/` |
| 环境变量 | `NAV/config/datasets.env`（`R2R_ROOT`/`RXR_ROOT`/`LHPR_VLN_ROOT`/`SCALEVLN_ROOT`/`MP3D_HABITAT_ROOT`） |
| 总下载入口 | `NAV/scripts/datasets/download_vln_four.sh` |
| 核实脚本 | `NAV/scripts/datasets/verify_vln_four.py` |
| 单轨迹渲染 smoketest | `NAV/scripts/datasets/run_smoke_render_r2r.sh` |
| 渲染产物 | `/sharedata/NAV/derived/vln_render_smoke/` |
| 汇总日志 | `/sharedata/datasets/_vln_prep_logs/` |
| 各库 inventory | `<dataset>/manifests/inventory.json` |
| 各库说明 | `<dataset>/README_CN.md` |

### 分数据集状态

#### 1. R2R / R2R-CE

- 标准包与预处理包此前已落盘（2026-07-23），SHA-256 与
  `NAV-OPS-001` 一致。
- 2026-08-07 补齐 `raw/annotations/manifests/scripts/logs` 与
  `README_CN.md`。
- 已验证 split：train 10819、val_seen 778、val_unseen 1839、test 3408；
  preprocessed envdrop 146304。
- 场景：`/sharedata/datasets/mp3d`（与 RxR-CE 共用）。

#### 2. RxR / RxR-CE

- **RxR-CE** `RxR_VLNCE_v0.zip`（347 MB）已下载并解压到
  `raw/rxr_ce/RxR_VLNCE_v0/`。
- Guide episode 核实：

| Split | Episodes | Scenes |
| --- | ---: | ---: |
| train | 60,300 | 59 |
| val_seen | 6,746 | 57 |
| val_unseen | 11,006 | 11 |
| test_challenge | 9,557 | 17 |

- 原始 `gs://rxr-data` Guide/Follower jsonl：本机 HTTP 返回 HTML 拦截页，
  无法直接 wget；dense pose traces 需 `gsutil`（当前未安装），未下载。
- 闭环训练以 RxR-CE episode 内 pose/action 为准，不依赖原始 pose_traces 包。
- StreamVLN `RxR/annotations.json` sidecar（41 MB）已于 2026-08-07 14:00
  落盘：`annotations/streamvln_rxr_annotations`。

#### 3. LHPR-VLN

- HF：`Starry123/LHPR-VLN`（约 91.4 GB）。
- 已完成：`task.zip`、`step_task.zip`、`episode_task.zip` 下载并解压；
  `annotations/{task,step_task,episode_task}` 软链可用。
- 进行中：`batch_1.zip`…`batch_8.zip` 经 clash 代理续传
  （`_vln_prep_logs/lhpr_proxy.log`）。
- 在线闭环需要 HM3D 场景；batch 包为离线观测回放，非在线评测强制依赖。

#### 4. ScaleVLN

- HF：`OpenGVLab/ScaleVLN` + `cywan/StreamVLN-Trajectory-Data`。
- 已完成（2026-08-07 14:00 核实为 `verified_local`）：
  - `r2r_preprocess_data.zip`（522 MB）已解压；
  - `features.zip`（约 35 GB）；
  - `rvr_data.zip`（约 8.5 GB）；
  - CE 子集：`annotations/StreamVLN_CE/scalevln_subset_150k.json.gz`
    （22.96 MB）与 `annotations.json`（153 MB）。

### 后台任务

| 任务 | 日志 | 说明 |
| --- | --- | --- |
| LHPR batch 续传 | `/sharedata/datasets/_vln_prep_logs/lhpr_proxy.log` | `scripts/datasets/_run_lhpr_proxy.sh` |
| StreamVLN CE 下载记录 | `/sharedata/datasets/_vln_prep_logs/streamvln_ce_*.log` | 已完成 |
| 总控历史 | `/sharedata/datasets/_vln_prep_logs/master_*.log` | `download_vln_four.sh` |

续传完成后应再跑：

```bash
python3 NAV/scripts/datasets/verify_vln_four.py
```

### Habitat 单轨迹渲染 smoketest（2026-08-07）

入口：`NAV/scripts/datasets/run_smoke_render_r2r.sh`（解释器默认
`~/.conda/envs/unigoal`，habitat_sim 0.2.3）。

在 RTX 6000 Ada、640×480 RGB、VLN-CE 默认步进（0.25 m / 30°）下，渲染
R2R-CE train `episode_id=8033`（21 GT actions → 22 帧）：

| 口径 | 耗时 |
| --- | ---: |
| 场景加载（首次打开 `.glb`） | **26.17 s** |
| 逐步渲染合计（不含写盘） | 0.17 s（约 **130 FPS**） |
| 单帧均值 / 中位 | 7.7 ms / 1.4 ms |
| 含 PNG IO 的循环墙钟 | 2.15 s |
| 端到端（加载+渲染+写盘） | **28.3 s** |

粗外推（仅渲染、不计重复加载）：约 **0.008 s/帧**；若每集 50 步，
1000 集约 **6–7 分钟**纯渲染。**瓶颈是按 episode 重复加载场景**——按
scan 缓存复用后，大规模准备才接近该纯渲染速率。产物示例：
`/sharedata/NAV/derived/vln_render_smoke/20260807_175400_ep8033_gt_actions/`。

### V1 Stage3 R2R-CE RGB 渲染（2026-08-12）

按 NAV 派生目录规则，R2R-CE standard 的 train / val_seen / val_unseen 已启动
GPU0 后台渲染。该任务使用已有 raw policy skeleton，不修改原始标注或 MP3D
资产。

```text
tmux:
  nav_v1_stage3_vln_render_r2r

script:
  NAV/scripts/datasets/render_v1_stage3_vln_obs.py

check:
  bash NAV/scripts/check_v1_stage3_vln_render.sh \
    /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122

log:
  NAV/log/v1_data_prep/stage3_vln_render_r2r_standard_gpu0_20260812_100122.log

output:
  /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122/
```

渲染范围：

```text
R2R-CE standard train      10819 episodes
R2R-CE standard val_seen     778 episodes
R2R-CE standard val_unseen  1839 episodes
total                     13436 episodes
scenes                       72 MP3D scenes
resolution               480x640 RGB
format                   PNG
```

路径结构：

```text
rendered_obs/<run>/
  frames/r2r_ce/standard/<split>/ep<episode_id>/<frame_index>.png
  frames/r2r_ce/standard/<split>/ep<episode_id>/render_meta.json
  episodes/rendered_episodes.jsonl.gz
  manifests/render_summary.json
```

语义：

```text
00000.png = reset 后当前 observation
00001.png = 执行第 1 个 GT action 后 observation
...
policy chunk obs_index=t 对应 frame t
```

当前首次检查：

```text
completed = 100 / 13436 episodes
frames = 5256
errors = 0
size = 898 MB
GPU0 memory ≈ 956 MiB
```

注意：全 raw policy skeleton 含 167,658 episodes / 16.39M actions，按 PNG 体积会
超过当前可用 `/sharedata` 空间。因此本轮先渲染 R2R-CE standard 核心集；RxR-CE
guide/follower 需要在确认空间预算后分批继续。

### V1 Stage3 R2R-CE H=10 policy manifest（2026-08-17）

为对齐当前正式 action horizon 规则 `H_nav=10`，已重建 R2R-CE standard 的
Stage3 raw policy manifest，并与 2026-08-12 已完成的 rendered RGB frames 对齐。

Raw H=10 policy：

```text
/sharedata/NAV/derived/v1/vln/raw_policy_h10/20260817_034014/
```

Rendered H=10 policy：

```text
/sharedata/NAV/derived/v1/vln/rendered_policy_h10_r2r/20260817_034242/
```

关键脚本：

```bash
python NAV/scripts/datasets/build_v1_stage3_vln_raw.py \
  --out-root /sharedata/NAV/derived/v1/vln/raw_policy_h10 \
  --include r2r_ce:standard:train,r2r_ce:standard:val_seen,r2r_ce:standard:val_unseen \
  --action-horizon 10 \
  --history-steps 10 \
  --compress

python NAV/scripts/datasets/build_v1_stage3_rendered_policy_manifest.py \
  --raw-root /sharedata/NAV/derived/v1/vln/raw_policy_h10/latest \
  --render-root /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122 \
  --out-root /sharedata/NAV/derived/v1/vln/rendered_policy_h10_r2r \
  --check-files
```

统计：

| item | value |
| --- | ---: |
| episodes | 13,436 |
| policy chunks | 782,861 |
| actions | 782,861 |
| locations | 514,789 |
| rendered episodes joined | 13,436 |
| skipped unrendered episode | 0 |
| skipped missing frame | 0 |
| raw H=10 manifest size | 136 MB |
| rendered H=10 manifest size | 70 MB |

按 split：

| source | chunks |
| --- | ---: |
| `r2r_ce:standard:train` | 631,244 |
| `r2r_ce:standard:val_seen` | 46,158 |
| `r2r_ce:standard:val_unseen` | 105,459 |

抽样验证：

```text
sample_id = r2r_ce__standard__train__ep1__t00000
horizon = 10
steps = 10
obs_frame_path exists = true
history_frame_paths exists = true
```

该 manifest 解决了旧 raw policy `action_horizon=4` 与当前正式 `H_nav=10`
不一致的问题，并把 policy row 的 `obs_frame_path/history_frame_paths` 指向真实
rendered RGB PNG。下一步若用于 Stage3 训练，还需要把 rendered RGB observation
编码为 Wan/VAE latent 或构建在线 VAE cache；不得在 policy 输入中使用 GT pose。

### 需要用户完成的授权

1. ~~Hugging Face `cywan/StreamVLN-Trajectory-Data`~~：2026-08-07 已批准并下载完成。
2. 若需要原始 RxR dense pose traces：安装 `gsutil` 后执行
   `gsutil -m cp -R gs://rxr-data <dir>`（约 161 GB）；**闭环主路径不依赖此项**。
3. LHPR 在线闭环另需 HM3D 场景许可与下载（与任务标注分离）。

### 限制与未决

1. 原始 RxR GCS jsonl 在本机被 HTML 拦截，inventory 中
   `original_annotations` 为空；以 RxR-CE 为 Source of Truth，
   StreamVLN RxR sidecar 作文本补充。
2. ScaleVLN 官方包面向离散 DUET；连续环境使用
   `scalevln_subset_150k.json.gz`（已就绪）。
3. LHPR batch 包体量大、HF 易超时，必须走代理并允许长时间续传。
4. 本状态文档只覆盖 episode/标注落盘，不覆盖 NAV latent/action 派生缓存；
   派生仍写入 `/sharedata/NAV/derived/`（见 `NAV-DAT-003`）。

### 相关资源

- 下载脚本：`NAV/scripts/datasets/download_vln_four.sh`、
  `/sharedata/datasets/{R2R,RxR,LHPR-VLN,ScaleVLN}/scripts/`
- 资源总账：`../06_operations/resource_inventory.md`
- 决策：`../00_overview/decision_log.md` DEC-020
- 长视频数据状态（并行线）：`data_preparation_schema_and_status.md`


## 长视频数据集准备任务书


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-002` |
| 类型 | 数据规范（Dataset Specification） |
| 状态 | Active |
| 更新时间 | 2026-07-28 |
| 职责 | 定义目录、manifest、质量检查和验收标准 |

### 一、任务背景

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

### 二、总体交付要求

#### 2.1 公共目录

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

#### 2.2 每个数据集的标准目录

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

#### 2.3 统一 manifest 格式

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

#### 2.4 长序列清单

按真实时长和帧数产生：

- `long_4chunk.jsonl`：可连续采样至少 324 帧；
- `long_8chunk.jsonl`：可连续采样至少 648 帧；
- `long_16chunk.jsonl`：可连续采样至少 1296 帧。

若视频不是 30 FPS，应按时间计算可覆盖长度，之后允许按 30 FPS 采样。不得通过
循环播放、复制帧或重复短 clip 的方式凑够长度。

#### 2.5 视频有效性检查

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

### 三、各数据集具体要求

### 3.1 DL3DV-10K

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

### 3.2 Sekai

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

### 3.3 SpatialVID

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

### 3.4 Ego4D

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

### 3.5 RealEstate10K

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

### 四、Revisit / Loop Closure 候选筛选

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

### 五、脚本要求

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

### 六、数据报告要求

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

### 七、分阶段验收

#### 阶段 A：只分析 metadata

每个数据集先交付：

- 官方入口和许可证说明；
- metadata 的字段解释；
- 视频数量和时长分布；
- 预计满足 4/8/16 chunks 的数量；
- 预计下载空间；
- 下载是否需要账号、申请或人工接受条款。

阶段 A 完成并确认后，再进行大规模下载，避免下载大量无法用于长序列训练的短
clips。

#### 阶段 B：每个数据集 5 条冒烟测试

每个数据集先准备 5 条可播放长视频，验证：

- 下载脚本可重复运行；
- manifest 字段正确；
- 视频可解码；
- pose/caption 能与视频对应；
- NAV 可以从每条视频连续读取 16 个 chunks。

#### 阶段 C：第一批正式子集

建议目标：

| 数据集 | 第一批目标 |
| --- | ---: |
| DL3DV-10K | 100 条长 episode |
| Sekai | 100 条长 episode |
| SpatialVID | 100 条 16-chunk episode；不足则如实报告 |
| Ego4D | 100 个来自长原视频的连续片段 |
| RealEstate10K | 尽量获得 100 条 16-chunk episode |

#### 阶段 D：最终交接

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

### 八、重要注意事项

- 不要删除或覆盖 `/sharedata` 中已有数据；
- 不要下载全量数据后才检查长度和许可证；
- 不要循环短视频、复制帧或拼接不连续 clips 来制造长样本；
- 不要擅自统一重编码所有视频，原视频优先保留；
- 不要在 Git 中提交视频、压缩包或大型标注；
- 不要将账号凭据写入脚本、日志或 README；
- 不要假设 metadata 存在就代表原视频可下载；
- 遇到需要签署数据条款、提供组织信息或大量占用公共存储时，先暂停并确认；
- 所有过程应可恢复、可重复、可统计。


## 长视频训练数据调研


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-001` |
| 类型 | 数据调研（Dataset Survey） |
| 状态 | Reference |
| 更新时间 | 2026-07-28 |
| 职责 | 比较候选长视频数据及其对流式训练的适用性 |

更新时间：2026-07-26

### 当前 NAV 数据的实际情况

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

### Infinite-World 的长数据构造

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

### LingBot-World 的长数据构造

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

### 可获得的公开长视频数据

| 数据集 | 规模和长时特性 | 对 NAV 的价值 | 限制 |
| --- | --- | --- | --- |
| [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) | 10,510 个场景；手机视频通常不少于 60 秒，航拍不少于 45 秒；提供相机参数 | 最适合先做 4/8/16-chunk 和 loop closure；室内外覆盖与 RE10K 接近 | 需接受数据条款；480P+pose 全量约 730 GB，建议先下子集 |
| [Sekai](https://huggingface.co/papers/2506.15675) | 超过 5,000 小时 walking、drone FPV/UAV，覆盖 100+ 国家、750 城市；含相机轨迹和 caption | 最接近 Infinite-World 的大规模探索视频预训练分布 | YouTube 来源的实际可下载率和许可证需在批量下载前审计 |
| [SpatialVID](https://github.com/NJU-3DV/SpatialVID) | 由 21,000 小时原始视频筛成约 2.7M clips、7,089 小时；含 pose、depth、motion instruction、caption | 适合扩大空间运动和文本条件多样性 | 很多条目是切分后的 clip，应先检查连续时长；许可证为 CC BY-NC-SA 4.0 |
| [Ego4D](https://ego4d-data.org/docs/start-here/) | 3,670+ 小时第一视角日常活动视频 | 可作为 LingBot 类第一视角通用先验 | 完整数据很大且需申请许可；并非所有视频都有连续相机位姿或导航动作 |
| [RealEstate10K](https://google.github.io/realestate10k/download.html) | YouTube 房产漫游视频，相机内外参和时间戳公开 | 和当前 pipeline 最兼容，可补抓比本地子集更长的 episode | 原始 YouTube 链接存在失效；当前本机子集最长仅 279 帧 |
| [ACID](https://cove.thecvf.com/datasets/609) | 航拍自然场景，含逐帧 3D camera parameters | 可补充长距离飞行和大范围视差 | 场景域偏航拍自然环境，且原视频链接可能失效 |

### 推荐实施顺序

#### 第一阶段：先把训练目标改成长递归

缓存格式改为一个 episode 保存可变数量的独立 chunk，训练时按课程采样
`L ∈ {2, 4, 8, 16}`。Register 依次读取前面的 chunk，并在每次转移或最后一次
转移计算 next-chunk diffusion loss。显存不足时采用 truncated BPTT，例如
每 2–4 个 chunk detach 一次 Register。

这一步不能只对当前 162 帧 cache 做重复采样，否则模型看到的是重复内容，
无法学到长时间累积与 loop closure。

#### 第二阶段：用 DL3DV-10K 子集验证

优先下载 50–100 个具有明显折返、环视或回到原位置的 60 秒序列。按 81
帧/chunk，单条视频能提供约 16–22 个连续 chunk，足以覆盖
Infinite-World 的 16-chunk 微调设置。先在 2→4→8→16 的课程上验证
Register 是否能在多次递归更新后保持场景信息。

#### 第三阶段：扩大覆盖面

- Sekai 用于大规模第一视角/探索预训练；
- SpatialVID 用于带 pose、depth、caption 的空间运动训练；
- Ego4D 只作为第一视角通用视频先验，不作为首个导航记忆数据集。

#### 第四阶段：构造小型 revisit-dense 集合

从 DL3DV/Sekai 筛选或自行拍摄约 100 条频繁回访的长序列，形成
Infinite-World RDD 的公开可复现替代版本。相比单纯增加随机视频小时数，
这部分对测试长期记忆和 loop closure 更关键。


## Kinetics-400 相机视角变化抽样


> 合并自旧主题；原独立文件已并入本文，不再作为独立事实源维护。


| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-004` |
| 类型 | 标注协议与记录（Annotation Protocol） |
| 状态 | Running |
| 更新时间 | 2026-07-28 |
| 职责 | 定义 VGGT 相机运动标注、RE10K 标定和恢复方式 |

### 设置

- 数据根目录：`/sharedata/datasets/kinetics400`
- 抽样数量：100 条视频
- 抽样方式：对 train、validation、test 三个划分分层随机抽样
- 每条视频：均匀抽取 8 帧
- 模型：VGGT-1B，本地权重
- 输出目录：`NAV/result/kinetics_vggt_camera_100`

尺度无关指标包括：

- 累计相机旋转角（Cumulative Camera Rotation）；
- VGGT 相机路径长度除以中位预测场景深度
  （Camera Path / Median Scene Depth）。

本次启发式分级为：

- 低变化（Low）：累计旋转小于 2°，且位移/深度小于 0.02；
- 中等变化（Moderate）：累计旋转小于 8°，且位移/深度小于 0.10；
- 明显变化（Strong）：其余样本。

### 结果

| 等级 | 数量 | 比例 |
| --- | ---: | ---: |
| 低变化 | 10 | 10% |
| 中等变化 | 15 | 15% |
| 明显变化 | 75 | 75% |

- VGGT 推理成功：100/100；
- 累计旋转角中位数：24.55°；
- 相机路径/场景深度中位数：0.238。

### 解释与限制

结果说明 Kinetics-400 中存在大量跨 chunk 的明显视角变化，适合用于训练
Register/history memory 的长程变化建模。但 Kinetics 包含大量快速人体运动、
剪辑和非刚体物体；VGGT 可能把动态主体造成的对应关系变化解释为相机运动。
因此这些指标适合作为预筛选信号，不应直接视为相机位姿真值。正式构造训练集时，
还应结合 VGGT confidence、镜头切换检测和光流一致性过滤。
### Kinetics-400 全量 VGGT 相机运动标注

#### 标注目标

对 `/sharedata/datasets/kinetics400` 的 train、val、test 全部 episode 均匀抽取
8 帧，用 VGGT-1B 预测相机外参和稠密深度，并为每条视频保存：

- 首尾旋转角、累计旋转角；
- 相机轨迹长度、深度归一化平移量；
- `low / moderate / strong` 相机运动标签；
- VGGT 稠密深度置信度的 p10、p25、p50、p75；
- `low / reference / high` 置信度等级及 `motion_label_reliable` 标记。

结果位于 `NAV/result/kinetics_vggt_camera_all/metrics/`，运行日志位于
`NAV/result/kinetics_vggt_camera_all/logs/`。JSONL 每完成一个 episode 就立即
落盘，可中断后使用同一命令续跑。

#### RE10K 置信度标定

标定输入是 `/sharedata/RealEstate10K/vggt_runs` 中已有的 30 个场景。VGGT 的
confidence 由 `1 + exp(logit)` 得到，是**稠密深度置信度**，不是相机位姿正确
概率。其绝对值和这批 RE10K 的 Sim(3) 对齐相机中心误差没有显著相关性，因此
仅用它对运动标签做质量门控。

门槛保存在 `NAV/config/vggt_re10k_confidence.json`：

- `reference`：episode 的 p25 ≥ 1.9536 且 p50 ≥ 5.9850，即不低于 RE10K
  场景分布的下四分位；
- `high`：p25 ≥ 6.7623 且 p50 ≥ 12.7652，即达到 RE10K 场景中位水平；
- 其余为 `low`，仍保存运动数值和标签，但令 `motion_label_reliable=false`。

VGGT 官方可视化同样使用 confidence percentile 过滤低置信点。Kinetics 含大量
人物快速运动、剪辑和弱几何场景，校准 smoke test 的 3/3 条均低于 RE10K 门槛；
这不是程序失败，而是明确标出其相对 RE10K 的域外低置信状态。

#### 运行与恢复

单 GPU 全量运行：

```bash
bash NAV/scripts/run_kinetics_vggt_all.sh 0 1 cuda:0
```

脚本按视频路径稳定哈希分片。将来若有多张空闲 GPU，可令第二个参数为总分片数，
分别启动不同的 shard index。每个 shard 都读取已有 JSONL 并跳过已完成视频。

## Stage3 R2R rendered observation latent cache

### Full R2R obs latent cache 启动记录（2026-08-17）

目的：把已经渲染好的 R2R-CE RGB observations 编码为 Wan VAE latent，供
Stage3 policy 使用，避免训练时在线读 PNG + VAE 编码。

输入：

```text
/sharedata/NAV/derived/v1/vln/rendered_obs/
  stage3_vln_render_r2r_standard_gpu0_20260812_100122/
    episodes/rendered_episodes.jsonl.gz
```

输出：

```text
/sharedata/NAV/derived/v1/vln/obs_latents_r2r_full/
  r2r_standard_20260817_full/
    episodes/r2r_ce/standard/<split>/ep*.pt
    manifests/encoded_episodes_shard-000-of-002.jsonl
    manifests/encoded_episodes_shard-001-of-002.jsonl
```

每个 episode `.pt` 保存：

```text
obs_latents: [N,16,1,56,112] fp16
frame_paths: 原 RGB frame 路径
metadata: rendered episode row
```

后台任务：

```text
tmux session:
  nav_vln_obs_latents_full

GPU0:
  scripts/datasets/encode_v1_stage3_vln_obs_latents.py
    --device cuda:0
    --output-root /sharedata/NAV/derived/v1/vln/obs_latents_r2r_full
    --run-name r2r_standard_20260817_full
    --num-shards 2 --shard-index 0

GPU1:
  scripts/datasets/encode_v1_stage3_vln_obs_latents.py
    --device cuda:1
    --output-root /sharedata/NAV/derived/v1/vln/obs_latents_r2r_full
    --run-name r2r_standard_20260817_full
    --num-shards 2 --shard-index 1
```

日志：

```text
NAV/log/v1_data_prep/obs_latents_r2r_full_shard0_20260817.log
NAV/log/v1_data_prep/obs_latents_r2r_full_shard1_20260817.log
```

启动确认：

```text
每 shard 6718 episodes；总计约 13436 episodes。
2026-08-17 04:35 左右已落盘 141 episodes。
2026-08-17 06:14 手动恢复 encoder 后，manifest 计数：
  shard-000 = 627
  shard-001 = 710
  total     = 1337 / 13436 episodes
  disk      ≈ 14G
2026-08-17 07:17 Stage2 2k cotrain 占用 cuda:1，shard-001 暂停；
  shard-000 = 1314
  shard-001 = 773
  total     = 2087 / 13436 episodes
  disk      ≈ 22G
2026-08-17 07:18：
  shard-000 = 1328
  shard-001 = 773
  total     = 2101 / 13436 episodes
  disk      ≈ 22G
2026-08-17 07:36：
  shard-000 = 1385
  shard-001 = 773
  total     = 2158 / 13436 episodes
  disk      ≈ 22G
  note      = cuda:0 shard-000 running; cuda:1 shard-001 remains SIGSTOP while
              Stage2 RE10K-full 2k cotrain occupies GPU1. A resume watcher will
              SIGCONT shard-001 after the training PID exits.
2026-08-17 07:39：
  shard-000 = 1436
  shard-001 = 773
  total     = 2209 / 13436 episodes
  disk      ≈ 23G
2026-08-17 07:41：
  shard-000 = 1464
  shard-001 = 773
  total     = 2237 / 13436 episodes
  disk      ≈ 23G
2026-08-17 07:44：
  shard-000 = 1498
  shard-001 = 773
  total     = 2271 / 13436 episodes
  disk      ≈ 24G
2026-08-17 07:45：
  shard-000 = 1522
  shard-001 = 773
  total     = 2295 / 13436 episodes
  disk      ≈ 24G
2026-08-17 07:47：
  shard-000 = 1545
  shard-001 = 773
  total     = 2318 / 13436 episodes
  disk      ≈ 24G
2026-08-17 07:49：
  shard-000 = 1575
  shard-001 = 773
  total     = 2348 / 13436 episodes
  disk      ≈ 25G
2026-08-17 07:53：
  shard-000 = 1638
  shard-001 = 773
  total     = 2411 / 13436 episodes
  disk      ≈ 25G
2026-08-17 07:55：
  shard-000 = 1672
  shard-001 = 773
  total     = 2445 / 13436 episodes
  disk      ≈ 25G
2026-08-17 07:57：
  shard-000 = 1698
  shard-001 = 773
  total     = 2471 / 13436 episodes
  disk      ≈ 26G
2026-08-17 07:58：
  shard-000 = 1714
  shard-001 = 773
  total     = 2487 / 13436 episodes
  disk      ≈ 26G
2026-08-17 08:00：
  shard-000 = 1732
  shard-001 = 773
  total     = 2505 / 13436 episodes
  disk      ≈ 26G
2026-08-17 08:03：
  shard-000 = 1764
  shard-001 = 773
  total     = 2537 / 13436 episodes
  disk      ≈ 27G
2026-08-17 08:04：
  shard-000 = 1769
  shard-001 = 773
  total     = 2542 / 13436 episodes
  disk      ≈ 27G
2026-08-17 08:18：
  shard-000 = 1895
  shard-001 = 773
  total     = 2668 / 13436 episodes
  disk      ≈ 29G
  note      = cuda:0 shard-000 resumed after step500 decoded eval; cuda:1
              shard-001 remains SIGSTOP while Stage2 RE10K-full 2k cotrain
              continues on GPU1. Step500 eval temporarily paused shard-000 and
              then restored it successfully.
2026-08-17 08:22：
  shard-000 = 1954
  shard-001 = 773
  total     = 2727 / 13436 episodes
  note      = cuda:0 shard-000 running; cuda:1 shard-001 still SIGSTOP for
              Stage2 RE10K-full 2k cotrain.
2026-08-17 08:24：
  shard-000 = 1977
  shard-001 = 773
  total     = 2750 / 13436 episodes
  disk      ≈ 29G
  note      = cuda:0 shard-000 running; cuda:1 shard-001 still SIGSTOP.
2026-08-17 08:26：
  shard-000 = 2020
  shard-001 = 773
  total     = 2793 / 13436 episodes
  disk      ≈ 30G
  note      = cuda:0 shard-000 running; cuda:1 shard-001 still SIGSTOP.
2026-08-17 08:27：
  shard-000 = 2042
  shard-001 = 773
  total     = 2815 / 13436 episodes
  disk      ≈ 30G
  note      = cuda:0 shard-000 running; cuda:1 shard-001 still SIGSTOP.
```

Stage3 训练脚本 `scripts/train_v1_stage3_iw_aligned_r2r_policy.py` 已支持
`--obs-latent-manifest` 指向单个 manifest 文件，或直接指向上述 full cache run
目录；若传目录，脚本自动读取 `manifests/encoded_episodes*.jsonl(.gz)`。
