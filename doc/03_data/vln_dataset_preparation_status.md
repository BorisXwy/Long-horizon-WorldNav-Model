# VLN 闭环数据集准备状态

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-007` |
| 类型 | 动态状态（Operational Status） |
| 状态 | Live |
| 更新时间 | 2026-08-07（Asia/Shanghai） |
| 职责 | 汇总 R2R-CE、RxR-CE、LHPR-VLN、ScaleVLN 在 `/sharedata/datasets/` 的落盘、核实与阻塞项 |

更新时间：2026-08-07 18:40（Asia/Shanghai）

## 规模与存储估算（2026-08-07 实测）

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
`dataset_preparation_status.md` 同期「规模与 latent」小节。

## 当前结论

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

## 唯一路径与入口

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

## 分数据集状态

### 1. R2R / R2R-CE

- 标准包与预处理包此前已落盘（2026-07-23），SHA-256 与
  `NAV-OPS-001` 一致。
- 2026-08-07 补齐 `raw/annotations/manifests/scripts/logs` 与
  `README_CN.md`。
- 已验证 split：train 10819、val_seen 778、val_unseen 1839、test 3408；
  preprocessed envdrop 146304。
- 场景：`/sharedata/datasets/mp3d`（与 RxR-CE 共用）。

### 2. RxR / RxR-CE

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

### 3. LHPR-VLN

- HF：`Starry123/LHPR-VLN`（约 91.4 GB）。
- 已完成：`task.zip`、`step_task.zip`、`episode_task.zip` 下载并解压；
  `annotations/{task,step_task,episode_task}` 软链可用。
- 进行中：`batch_1.zip`…`batch_8.zip` 经 clash 代理续传
  （`_vln_prep_logs/lhpr_proxy.log`）。
- 在线闭环需要 HM3D 场景；batch 包为离线观测回放，非在线评测强制依赖。

### 4. ScaleVLN

- HF：`OpenGVLab/ScaleVLN` + `cywan/StreamVLN-Trajectory-Data`。
- 已完成（2026-08-07 14:00 核实为 `verified_local`）：
  - `r2r_preprocess_data.zip`（522 MB）已解压；
  - `features.zip`（约 35 GB）；
  - `rvr_data.zip`（约 8.5 GB）；
  - CE 子集：`annotations/StreamVLN_CE/scalevln_subset_150k.json.gz`
    （22.96 MB）与 `annotations.json`（153 MB）。

## 后台任务

| 任务 | 日志 | 说明 |
| --- | --- | --- |
| LHPR batch 续传 | `/sharedata/datasets/_vln_prep_logs/lhpr_proxy.log` | `scripts/datasets/_run_lhpr_proxy.sh` |
| StreamVLN CE 下载记录 | `/sharedata/datasets/_vln_prep_logs/streamvln_ce_*.log` | 已完成 |
| 总控历史 | `/sharedata/datasets/_vln_prep_logs/master_*.log` | `download_vln_four.sh` |

续传完成后应再跑：

```bash
python3 NAV/scripts/datasets/verify_vln_four.py
```

## Habitat 单轨迹渲染 smoketest（2026-08-07）

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

## V1 Stage3 R2R-CE RGB 渲染（2026-08-12）

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

## 需要用户完成的授权

1. ~~Hugging Face `cywan/StreamVLN-Trajectory-Data`~~：2026-08-07 已批准并下载完成。
2. 若需要原始 RxR dense pose traces：安装 `gsutil` 后执行
   `gsutil -m cp -R gs://rxr-data <dir>`（约 161 GB）；**闭环主路径不依赖此项**。
3. LHPR 在线闭环另需 HM3D 场景许可与下载（与任务标注分离）。

## 限制与未决

1. 原始 RxR GCS jsonl 在本机被 HTML 拦截，inventory 中
   `original_annotations` 为空；以 RxR-CE 为 Source of Truth，
   StreamVLN RxR sidecar 作文本补充。
2. ScaleVLN 官方包面向离散 DUET；连续环境使用
   `scalevln_subset_150k.json.gz`（已就绪）。
3. LHPR batch 包体量大、HF 易超时，必须走代理并允许长时间续传。
4. 本状态文档只覆盖 episode/标注落盘，不覆盖 NAV latent/action 派生缓存；
   派生仍写入 `/sharedata/NAV/derived/`（见 `NAV-DAT-003`）。

## 相关资源

- 下载脚本：`NAV/scripts/datasets/download_vln_four.sh`、
  `/sharedata/datasets/{R2R,RxR,LHPR-VLN,ScaleVLN}/scripts/`
- 资源总账：`../06_operations/resource_inventory.md`
- 决策：`../00_overview/decision_log.md` DEC-020
- 长视频数据状态（并行线）：`dataset_preparation_status.md`
