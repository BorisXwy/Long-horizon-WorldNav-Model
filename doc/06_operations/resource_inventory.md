# NAV 项目资源索引

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OPS-001` |
| 类型 | 资源总账（Resource Inventory） |
| 状态 | Live / Source of Truth |
| 更新时间 | 2026-08-29（新增流式 3D / navigation 官方复现资源） |
| 职责 | 维护数据、权重、环境、仓库、日志和结果的唯一标准路径 |

本文档记录 NAV 新实验可直接使用的数据集、预训练权重和本地参考仓库。
路径均已在当前机器上确认存在。新增资源时请同步补充用途、格式和验证状态。
自 2026-08-22 起，训练 latent / tensor cache 的真实落盘根目录为
`NAV/data/train/<dataset>/<latent_run>/`；公共 `/sharedata` 只保留原始数据、
assets、轻量 manifest 和必要临时 render 中间态。

当前 V1 技术设计入口是 `NAV/doc/01_model/model_evolution_and_current_architecture.md`。
V0 InfiniteWorld/Register A/B 旧方案保留在
`NAV/doc/01_model/model_evolution_and_current_architecture.md`，只作为历史和 baseline。

Infinite-World 的目录结构、完整推理调用链、HPMC 代码实现及其与论文的逐项
一致性检查记录在 `NAV/doc/01_model/model_evolution_and_current_architecture.md`。

## V1 action-centered 数据资源

V1 不复用 V0 的 81-frame dense latent 作为主缓存。当前正式 Stage One 主缓存为
`T_latent=4` micro chunk latent；早期 `[obs, future_1, future_2, future_3,
future_4]` sparse pack 只保留为历史速度记录。

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| V1 数据配置 | `NAV/config/v1_data_prep.yaml` | future stride、action horizon、VAE pack 输出格式 |
| V1 pack manifest | `/sharedata/NAV/derived/v1/manifests/stage1_video_packs.jsonl` | 194,088 条 5-frame sparse pack；早期 T=1 route |
| V1 T4 micro manifest | `/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes.jsonl` | 24,261 条 episode；`micro_frames=13`、`micro_stride=12`、`latent_t=4` |
| V1 T4 micro manifest spatial20 | `/sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl` | Stage One 默认候选；SpatialVID 用 deterministic md5(sample_id) 采样 20%，避免短视频数据压倒 DL3DV |
| V1 T4 micro latent smoke | `/sharedata/NAV/derived/v1/t4_micro_latents_smoke/` | 已验证 1 条 RE10K，`micro_latents=[1,8,16,4,56,112]` |
| V1 T4 micro latent 目标 | `/sharedata/NAV/derived/v1/t4_micro_latents/` | 全量 T4 micro latent 目标；当前不是正式训练默认 |
| V1 T4 micro latent spatial20 | `/sharedata/NAV/derived/v1/t4_micro_latents_spatial20/` | 当前正式 Stage One 训练默认数据根；SpatialVID deterministic 20% |
| V1 VAE debug smoke | `/sharedata/NAV/derived/v1/vae_packs_debug_smoke/` | 已验证 1 条 RE10K，`z_obs/z_future=[16,1,56,112]` |
| V1 HDF5 smoke | `/sharedata/NAV/derived/v1/vae_packs_hdf5_smoke/` | 已验证 1 条 RE10K，index+shard 可被 `V1Hdf5Dataset` 读取 |
| V1 HDF5 shard 目标 | `/sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7/` | LeRobot-like index + shard 格式；14路 full-GPU 编码 |
| V1 Stage3 VLN raw policy | `/sharedata/NAV/derived/v1/vln/raw_policy/` | CPU-only 构建 episode/action/policy chunk manifest |
| V1 Stage3 VLN rendered obs | `/sharedata/NAV/derived/v1/vln/rendered_obs/` | Habitat-Sim RGB 渲染输出；当前 R2R-CE standard train/val_seen/val_unseen 正在 GPU0 后台准备 |
| V1 Stage3 RxR 500GiB stoppad budget | `/sharedata/NAV/derived/v1/vln/raw_policy_budgeted/stage3_vln_budget_t4_500g_stoppad_20260822_1423/` | RxR guide/follower train budget；terminal STOP 按吸收态 padding；manifest 与预算仍在 sharedata；2026-08-22 16:05 起已暂停后台任务，保留已生成 latent |
| V1 Stage3 RxR T4 latent | `NAV/data/train/rxr_ce/t4_micro_latents_stoppad_20260822_1423/` | RxR stoppad 已生成部分；真实 `.pt` 写在个人 NAV/data/train，后续可从该目录恢复续写 |
| V1 Stage3 R2R train stoppad budget | `/sharedata/NAV/derived/v1/vln/raw_policy_budgeted/stage3_vln_budget_r2r_train_stoppad_20260822_1604/` | R2R train 全量；`10,819 episodes / 1,063,870 frames / 87,345 T4 micro chunks / 65.31 GiB latent` |
| V1 Stage3 R2R train rendered obs | `/sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_train_stoppad_gpu0_20260822_1605/` | Habitat-Sim 临时 PNG；渲染后由 stream encoder 校验 latent 并删除 PNG |
| V1 Stage3 R2R train T4 latent | `NAV/data/train/r2r_ce/t4_micro_latents_stoppad_20260822_1605/` | 当前优先准备的正式 V1 `T_latent=4` VLN latent |
| V1 Stage3 R2R train instruction embedding | `NAV/data/train/r2r_ce/text_embeddings_stoppad_20260822_1605/` | R2R latent 完成后自动缓存 Wan UMT5 instruction embedding；由后台 watcher 触发 |
| V1 full-pipeline smoke | `NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json` | 当前完整模型链路验证：Stage1/2/3 train + videogen/policy inference |
| V1 Stage2 RE10K pose-only formal run | `NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/` | 正在运行；RE10K-only，完整 T4 latent `[16,4,56,112]`，Wan-size 30 layers，Register 128，pose-only supervision |
| V1 Stage2 cotrain 原始组 | `NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_sharedwan_wanfull_2k_20260814_223648/` | 当前运行；`pose_head=mlp`、`lambda_pose=0.1`、完整 shared WanBlock |
| V1 Stage2 video-only 对照 | `NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_videoonly_sharedwan_wanfull_2k_20260815_025913/` | 已停止；`lambda_pose=0.0`，用于确认原 cotrain 的 `loss_visual` 与 video-only 几乎重合 |
| V1 Stage2 强 pose 对照 | `NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_linear_lam1_sharedwan_wanfull_2k_20260815_143827/` | 当前运行；`pose_head=linear`、`lambda_pose=1.0`，检查 3D loss 是否能更明显 reshape shared backbone |
| V1 Stage2 cotrain TensorBoard | `6017` | `cotrain_mlp_lam0p1` vs `cotrain_linear_lam1` |
| V1 Stage2 RE10K pose-only formal 文档 | `NAV/doc/03_training/training_plan_and_experiment_log.md` | 当前正式 Stage Two 验证训练的配置、preflight、loss、日志入口 |
| V1 Stage2 RE10K pose-video diagnostic | `NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/` | 非验收证据；缩小了 latent spatial resolution 与 backbone depth，只保留为调试追溯 |
| V1 Stage2 RE10K pose-video diagnostic eval | `NAV/result/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/eval_metrics.json` | 非验收证据；不得用于证明 Stage Two 方案有效 |
| V1 Stage2 RE10K diagnostic 文档 | `NAV/doc/03_training/training_plan_and_experiment_log.md` | 记录该 diagnostic 为什么无效，以及后续 pose-only 正式训练约束 |
| V1 Stage1 historical runs | `NAV/log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/` | 旧 action-interface run；仅保留为历史，不再代表当前代码 |
| V1 schema 文档 | `NAV/doc/02_data/data_preparation_schema_and_status.md` | ActionChunk、VaePack、GeometryTarget、V1Sample |

### V1 T4 micro latent spatial20 准备

2026-08-10 起，Stage One 的 T4 micro latent 默认不再全量处理 SpatialVID，
而是使用 deterministic 20% 子集。原因是全量 manifest 中 SpatialVID 占
`micro_chunks` 的 94.71%，会明显压倒 DL3DV；20% 子集后，自然帧量占比变为：

```text
manifest:
  /sharedata/NAV/derived/v1/manifests/stage1_t4_micro_episodes_spatial20.jsonl

summary:
  episodes=5,180
  covered_frames=2,978,792
  micro_chunks=247,801

natural micro_chunks ratio:
  SpatialVID 193,156 / 247,801 = 77.95%
  DL3DV       50,868 / 247,801 = 20.53%
  RE10K        3,497 / 247,801 =  1.41%
  Argoverse2     280 / 247,801 =  0.11%

valid long-history episodes:
  IW-1  = 5,180
  IW-4  = 3,242
  IW-8  = 1,666
  IW-16 =   140   # 基本由 DL3DV 提供
```

建议训练时不要完全按自然占比采样，而使用 dataset-aware sampler：

```text
Short / Medium history:
  SpatialVID 55–65%
  DL3DV      25–35%
  RE10K       5–10%
  Argoverse2  1–3%

Long history, especially IW-16 对齐:
  DL3DV only 或 DL3DV-heavy
```

当前后台编码配置：

```text
tmux session:
  nav_v1_t4_latent_spatial20_gpu0_0 ... nav_v1_t4_latent_spatial20_gpu0_5

logs:
  NAV/log/v1_data_prep/t4_micro/spatial20_gpu0_continue_20260811-002311/

outputs:
  /sharedata/NAV/derived/v1/t4_micro_latents_spatial20/

parallelism:
  GPU0:6 workers
  GPU1 reserved for current Stage One formal training
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

当前完整模型链路验证入口：

```bash
bash NAV/scripts/run_v1_full_pipeline_smoke.sh cpu
```

当前正式 run：

```text
tmux:
  nav_v1_stageone_final_train

run:
  NAV/log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/

TensorBoard:
  nav_v1_stageone_final_tensorboard
  0.0.0.0:6012

checkpoint init:
  /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
```

注意：训练启动时会扫描一次可用 latent 文件列表；后台新落盘的 SpatialVID latent
不会自动进入已启动训练，需要下一轮训练重新扫描数据根。

### V1 VAE pack 后台编码

2026-08-10 已校准为 14 路 tmux worker（GPU0:7、GPU1:7）。15 路
（GPU0:7、GPU1:8）可把 GPU1 顶满但会出现 CUDA OOM，因此当前默认采用
7x7 作为“吃满但稳定”的上限。数据准备类 GPU 工作之后按“尽量吃满显存但
不允许持续 OOM”的规则执行。

```text
tmux session:
  nav_v1_vae_worker0 ... nav_v1_vae_worker13

logs:
  NAV/log/v1_data_prep/encode_v1_vae_packs_worker0.log
  NAV/log/v1_data_prep/encode_v1_vae_packs_worker1.log

outputs:
  /sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7/workers/worker-000-of-014/
  ...
  /sharedata/NAV/derived/v1/vae_packs_hdf5_fullgpu14_7x7/workers/worker-013-of-014/
```

最新状态窗口（2026-08-10 01:41 CST）：14 个 worker 全部 running，error=0；
GPU0 约 44.3GB / 49.1GB、GPU1 约 40.6GB / 49.1GB，双卡 util=100%。
该窗口吞吐约 7.8 packs/s；194,088 条 pack 的正式准备 ETA 粗略为 7–10 小时，
后续若进入视频解码更重的数据段可能波动。

检查命令：

```bash
bash NAV/scripts/check_v1_vae_pack_encode.sh
```

启动命令：

```bash
bash NAV/scripts/run_v1_vae_pack_encode_2gpu.sh
```

### V1 Stage3 VLN raw policy 后台准备

2026-08-10 在不占用 GPU 的前提下启动 Stage3 raw policy 数据准备。当前
Habitat-Sim 环境即使设置 `CUDA_VISIBLE_DEVICES=` 与 `--gpu -1`，仍会尝试
创建 EGL CUDA device 0，并报：

```text
WindowlessEglApplication::tryCreateContext(): unable to find EGL device for CUDA device 0
WindowlessContext: Unable to create windowless context
```

因此当前只准备 CPU-only 的监督结构，不渲染 RGB 帧：

```text
tmux session:
  nav_v1_stage3_vln_raw

logs:
  NAV/log/v1_data_prep/build_v1_stage3_vln_raw.log

outputs:
  /sharedata/NAV/derived/v1/vln/raw_policy/<timestamp>/
    manifests/stage3_vln_episodes.jsonl.gz
    manifests/stage3_vln_policy_chunks.jsonl.gz
    actions/
    rendered_obs/   # path placeholder, render_status=pending
```

当前第一轮包含 R2R-CE standard train/val_seen/val_unseen 与 RxR-CE
guide/follower train/val_seen/val_unseen；EnvDrop 和 ScaleVLN 待核心 raw
完成后再追加。检查命令：

```bash
bash NAV/scripts/check_v1_stage3_vln_raw.sh
```

### V1 Stage3 VLN RGB render 后台准备

2026-08-12 启动 R2R-CE standard train / val_seen / val_unseen 的 Habitat RGB
渲染。输出写入 `/sharedata/NAV/derived/v1/vln/rendered_obs/`，不回写 raw policy
skeleton。

2026-08-20 扩展启动 500GiB latent 预算子集：已有 R2R rendered RGB 约
`59,056` 个 T4 micro chunks / `44.16 GiB` latent；新增 RxR guide/follower
train round-robin 选择 `74,584` episodes / `7,800,027` frames / `609,687`
T4 micro chunks / `455.85 GiB` latent。组合目标约 `500.0 GiB` latent。
当前 `/sharedata` 剩余约 `1.8T`，而新增 RxR raw PNG 中间态按抽样估计约
`1.89 TiB`，存在磁盘风险；后续应改为编码确认后清理 RGB 或流式
render→encode。

2026-08-21 已切换为在线清理：`encode_v1_stage3_vln_t4_stream.py` 监听
RxR render manifest，完成一个 episode 就编码为 `T_latent=4` micro latent，
校验 `.pt` 可读后删除该 episode 的 `*.png`，保留 `render_meta.json` 与
`latent_cleanup.json`。同时启动 disk guard：当 `/sharedata` free `<400GiB`
时自动停止 RxR render，保留 stream encoder 继续回收空间。R2R existing encoder
已暂停，待 RxR raw PNG 风险解除后恢复。

```text
tmux:
  nav_v1_stage3_vln_render_rxr_budget500_gpu1
  nav_v1_stage3_vln_t4_stream_rxr_gpu1
  nav_v1_stage3_vln_disk_guard

budget:
  /sharedata/NAV/derived/v1/vln/raw_policy_budgeted/
    stage3_vln_budget_t4_500g_with_existing_r2r_20260820_1259/

render:
  /sharedata/NAV/derived/v1/vln/rendered_obs/
    stage3_vln_render_rxr_budget500_gpu1_20260820_1830/

latent:
  /sharedata/NAV/derived/v1/vln/t4_micro_latents_500g/
    r2r_existing_20260820_1830/
    rxr_budget500_stream_20260821_0000/

logs:
  NAV/log/v1_data_prep/stage3_vln_render_rxr_budget500_gpu1_20260820_1830.log
  NAV/log/v1_data_prep/stage3_vln_t4_encode_r2r_gpu1_20260820_1830.log
  NAV/log/v1_data_prep/stage3_vln_t4_stream_rxr_gpu1_20260821_0000.log
  NAV/log/v1_data_prep/vln_render_disk_guard_20260821_0000.log

scripts:
  NAV/scripts/datasets/build_v1_stage3_vln_budget_manifest.py
  NAV/scripts/datasets/build_v1_stage3_vln_t4_micro_manifest.py
  NAV/scripts/datasets/render_v1_stage3_vln_obs.py
  NAV/scripts/datasets/encode_v1_t4_micro_latents.py
  NAV/scripts/datasets/encode_v1_stage3_vln_t4_stream.py
  NAV/scripts/watch_vln_render_disk_guard.sh
```

```text
tmux:
  nav_v1_stage3_vln_render_r2r

run:
  /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122/

log:
  NAV/log/v1_data_prep/stage3_vln_render_r2r_standard_gpu0_20260812_100122.log

check:
  bash NAV/scripts/check_v1_stage3_vln_render.sh \
    /sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122

script:
  NAV/scripts/datasets/render_v1_stage3_vln_obs.py
```

范围：

```text
r2r_ce:standard:train      10819 episodes
r2r_ce:standard:val_seen     778 episodes
r2r_ce:standard:val_unseen  1839 episodes
```

文件结构：

```text
frames/r2r_ce/standard/<split>/ep<episode_id>/<frame_index>.png
episodes/rendered_episodes.jsonl.gz
manifests/render_summary.json
```

### V1 完整模型链路验证（当前主线，2026-08-14）

当前主线以 `V1FullWorldNavModel` 为唯一有效代码入口，覆盖统一
`RegisterCell`、`A_hist/A_cur/A_noise`、dual-stream backbone、generation head、
action flow decoder 和 geometry probe。

```text
script:
  NAV/scripts/run_v1_full_pipeline_smoke.sh

report:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json

verified:
  Stage One   video generation train step
  Stage Two   video generation + 3D train step
  Stage Three action flow/CE + video/3D rehearsal train step
  Videogen inference
  Policy inference
  structural audit: no Extractor/Updater, no action/latent additive bias
```

### V1 Stage One 历史 Wan 训练（已从当前代码入口删除）

当前正式 Stage One 从官方 Wan2.1-T2V-1.3B 初始化，不再从 InfiniteWorld checkpoint
初始化。训练使用 T4 micro latent、Register latent prefix、Stage One action
interface 和 IW-1/4/8/16 等价长历史窗口。

```text
run:
  NAV/log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/

tmux:
  nav_v1_stageone_final_train

tensorboard:
  nav_v1_stageone_final_tensorboard
  0.0.0.0:6012

post-train sanity eval:
  watcher tmux:
    nav_v1_stageone_final_eval_waiter
  script:
    旧 generation sanity 脚本（2026-08-14 已删除，历史见 git）
  output:
    NAV/result/stageone_t4_generation_sanity/
      v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/
  log:
    NAV/log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650.eval_waiter.log

data_root:
  /sharedata/NAV/derived/v1/t4_micro_latents_spatial20

checkpoint:
  /sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors

checkpoint audit:
  source_type=wan2.1_official_safetensors
  loaded_keys=825
  mismatched_keys=0
  patch_embedding.weight:
    copy official 16 latent channels
    zero initialize 4 new mask channels

train:
  variant=latent_prefix
  train_scope=full
  trainable_parameters=1,422,300,004
  physical micro_batch_size=1
  gradient_accumulation_steps=16
  effective_batch_size=16
  steps=1000
  save_every=200
  history_iw_chunks=1,4,8,16
  history_micro_steps=7,27,54,107
  target_latent_t=4
  lambda_action_format=0
```

该批旧入口包含旧 `A_query` / action-bias / InfiniteWorld adapter 逻辑，2026-08-14
已从当前代码删除。若需要解释历史曲线，只读 log/doc；若必须复现，需要从 git
历史恢复清理前版本，不得把旧入口混入 V1 主线。

### V1 Stage1 Wan2.1 / InfiniteWorld sparse-pack 训练（历史记录）

2026-08-10 启动真实 backbone 版 V1 Stage1。初始化权重为
`/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt`；这是与
InfiniteWorld action/HPMC 改造兼容的 Wan2.1-1.3B backbone checkpoint。
训练时移除 HPMC，使用 V1 sparse pack：

```text
z_obs [B,16,1,56,112]       -> local_memory + Register extractor
z_future [B,16,1,56,112]    -> RFlow target
V1 primitive action horizon=4 -> 映射到 InfiniteWorld move/view action condition
```

本轮尚未接入额外 action prediction head；action 作为 video consequence
condition 注入 Wan action encoder，主 loss 为 InfiniteWorld RFlow diffusion
loss。

```text
run:
  NAV/log/v1-wan-stage1-full-mb2-ebs16-10000-20260810-132424/

environment:
  virtual_env/.venv_infinite_world
  flash_attn=2.7.4.post1
  2026-08-10 补装 h5py 与 pyarrow，用于读取 V1 HDF5/Parquet

checkpoint:
  /sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt
  load audit: missing_keys=[], unexpected_keys=[]

train:
  train_scope=full
  trainable_parameters=1,421,479,488
  physical micro_batch_size=2
  gradient_accumulation_steps=8
  effective_batch_size=16
  steps=10000
  AdamW lr=1e-5, weight_decay=0.01
  dtype=bf16
  save_every=1000

memory calibration:
  micro=1, accum=1: reserved≈29.47GB
  micro=2, accum=1: reserved≈45.40GB
  micro=3, accum=1: backward OOM
  formal micro=2, accum=8: step1 reserved≈45.40GB, realtime GPU used≈48.2GB

early signal:
  step 1 loss=0.2121, grad_norm=2.2969, sec/optimizer_step≈40.8
```

该 run 已被 2026-08-11 的官方 Wan2.1 init + T4 micro Stage One 历史 run
替代，只作为 sparse-pack / 历史速度记录保留。由于 effective batch 16
需要 8 次 micro forward/backward，单卡 10000 step
预计为 4–5 天量级；若需要加速，下一步应实现/验证双卡 DDP（需额外检查
DDP bucket 显存）。

VGGT-Ω 的完整网络、Register Token、交替注意力、预测头及其对 NAV Stage One
的适配边界记录在 `NAV/doc/01_model/model_evolution_and_current_architecture.md`。

当前两-chunk 数据限制、Infinite-World 与 LingBot-World 的长视频构造方式、
公开长视频数据集和推荐的数据课程记录在
`NAV/doc/02_data/data_preparation_schema_and_status.md`。

交给协作者执行的五个长视频数据集准备标准、统一 manifest、目录规范和分阶段
验收要求记录在 `NAV/doc/02_data/data_preparation_schema_and_status.md`。
实际下载、授权阻塞、metadata 统计和公共目录配置进度记录在
`NAV/doc/02_data/data_preparation_schema_and_status.md`。

## 项目目录结构

| 资源 | 路径与说明 |
| --- | --- |
| NAV 项目根目录 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV` |
| 论文写作目录 | `NAV/essay/`：核心 `.tex` 主文件、章节子文件、图表源码、草稿和投稿版本归档；不存放原始日志、checkpoint 或评测 JSON |
| 既有 SAO 实验 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/SAO` |
| 共享虚拟环境 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env` |
| 迁移后的个人资源 | `/sharedata/navigation_assets_xiewenyuan_legacy`；原路径为 `/mnt/pool1/sharehome/xiewenyuan/sharedata`，目录内容保持不变并添加 `legacy` 后缀 |

## 可用数据集

### RealEstate10K

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| 数据集根目录 | `/sharedata/RealEstate10K` | 当前本地布局约 35 GB |
| 已处理 RGB 帧 | `/sharedata/RealEstate10K/processed_frames/frames` | 按场景 ID 分组 |
| 原始视频 | `/sharedata/RealEstate10K/raw_videos` | 原始视频文件 |
| 数据清单（Manifest） | `/sharedata/RealEstate10K/manifests` | 训练集和验证集的场景及帧元数据 |
| 训练帧清单 | `/sharedata/RealEstate10K/manifests/frames_train.jsonl` | 帧级训练记录 |
| 验证帧清单 | `/sharedata/RealEstate10K/manifests/frames_val.jsonl` | 帧级验证记录 |
| 训练场景清单 | `/sharedata/RealEstate10K/manifests/scenes_train.json` | 训练场景列表 |
| 验证场景清单 | `/sharedata/RealEstate10K/manifests/scenes_val.json` | 验证场景列表 |
| 数据集报告 | `/sharedata/RealEstate10K/manifests/dataset_report.json` | 本地预处理与审计报告 |
| 原始相机标注 | `/sharedata/RealEstate10K/src_annotations` | 相机与序列标注 |
| VGGT 重建结果 | `/sharedata/RealEstate10K/vggt_runs` | 已有的逐场景几何重建结果 |
| NAV 全量训练清单 | `/sharedata/RealEstate10K/nav_register/manifests/train_all.jsonl` | 269个162帧窗口；包含由相机位姿生成的 `move/view` 伪标签 |
| NAV Wan VAE latent cache | `/sharedata/RealEstate10K/nav_register/latents` | 269个两-chunk cache；每个 chunk 为 `[1,16,21,56,112]` |

NAV 的 RE10K 数据转换、伪动作规则、两种 Register 注入方案、端到端训练结果和
运行状态见 `NAV/doc/03_training/training_plan_and_experiment_log.md`。

### R2R / R2R-CE

VLN-CE 官方的 `R2R_VLNCE_v1-3` 任务数据已于 2026-07-23 下载并验证。
现有 Matterport3D Habitat 资源包含全部 90 个场景，每个场景均具有一个
`.glb` 文件和一个 `.navmesh` 文件。

| 资源 | 路径 | 已验证内容 |
| --- | --- | --- |
| R2R 本地根目录 | `/sharedata/datasets/R2R` | 标准版及预处理版 v1-3 |
| 标准任务数据 | `/sharedata/datasets/R2R/R2R_VLNCE_v1-3` | `train`、`val_seen`、`val_unseen`、`test` |
| 预处理任务数据 | `/sharedata/datasets/R2R/R2R_VLNCE_v1-3_preprocessed` | 标准划分、真值动作（GT Actions）、嵌入及 EnvDrop |
| 标准版压缩包 | `/sharedata/datasets/R2R/R2R_VLNCE_v1-3.zip` | ZIP 完整性已验证 |
| 预处理版压缩包 | `/sharedata/datasets/R2R/R2R_VLNCE_v1-3_preprocessed.zip` | ZIP 完整性已验证 |
| Matterport3D Habitat 场景 | `/sharedata/datasets/mp3d/v1/tasks/mp3d` | 90 个 `.glb`、90 个 `.navmesh`，以及 house 和语义文件 |
| 原始 MP3D Habitat 压缩包 | `/sharedata/datasets/mp3d/v1/tasks/mp3d_habitat.zip` | 已有本地压缩包 |

已验证的任务数量：

| 数据划分（Split） | 任务数（Episodes） | 场景数（Scenes） |
| --- | ---: | ---: |
| train | 10,819 | 61 |
| val_seen | 778 | 53 |
| val_unseen | 1,839 | 11 |
| test | 3,408 | 18 |
| envdrop (preprocessed only) | 146,304 | 60 |

该任务数据源自 Matterport3D，使用时仍需遵守 Matterport3D 使用条款
（Terms of Use）和 VLN-CE 数据集许可证。

### RxR / RxR-CE

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| RxR 本地根目录 | `/sharedata/datasets/RxR` | 标准目录 + README_CN |
| RxR-CE 任务数据 | `/sharedata/datasets/RxR/raw/rxr_ce/RxR_VLNCE_v0` | train/val_seen/val_unseen/test_challenge |
| RxR-CE 压缩包 | `/sharedata/datasets/RxR/raw/rxr_ce/RxR_VLNCE_v0.zip` | 347 MB |
| 核实清单 | `/sharedata/datasets/RxR/manifests/inventory.json` | Guide episode 计数 |

已核实 Guide episode：train 60300、val_seen 6746、val_unseen 11006、
test_challenge 9557。原始 `gs://rxr-data` jsonl/pose traces 本机未落盘
（见 `NAV-DAT-007`）。场景与 R2R-CE 共用 `/sharedata/datasets/mp3d`。

### LHPR-VLN

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| 根目录 | `/sharedata/datasets/LHPR-VLN` | HF `Starry123/LHPR-VLN` |
| 任务标注 | `/sharedata/datasets/LHPR-VLN/annotations/{task,step_task,episode_task}` | 已解压 |
| 原始包 | `/sharedata/datasets/LHPR-VLN/raw/huggingface` | task/step/episode zip + batch 续传 |
| 下载日志 | `/sharedata/datasets/_vln_prep_logs/lhpr_proxy.log` | batch_1..8 续传 |

### ScaleVLN

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| 根目录 | `/sharedata/datasets/ScaleVLN` | HF `OpenGVLab/ScaleVLN` |
| 预处理标注 | `/sharedata/datasets/ScaleVLN/annotations/r2r_preprocess_data` | annotations/connectivity |
| 视觉特征包 | `/sharedata/datasets/ScaleVLN/raw/OpenGVLab/features.zip` | 约 35 GB，按需解压 |
| CE 子集 | `/sharedata/datasets/ScaleVLN/annotations/StreamVLN_CE` | `scalevln_subset_150k.json.gz` + annotations；已核实 |
| REVERIE 变体包 | `/sharedata/datasets/ScaleVLN/raw/OpenGVLab/rvr_data.zip` | 约 8.5 GB |
| RxR StreamVLN sidecar | `/sharedata/datasets/RxR/annotations/streamvln_rxr_annotations` | `annotations.json` 41 MB |

四套 VLN 的实时整备状态见 `NAV/doc/02_data/data_preparation_schema_and_status.md`。

### R2R-CE SOTA 复现（≥ StreamVLN，DEC-021 / NAV-EVL-002）

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| StreamVLN 代码 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/StreamVLN` | clone @`e48f6ff` |
| InternNav 代码 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/InternNav` | DualVLN 评测入口 @`7a5c624` |
| StreamVLN venv | `.../virtual_env/.venv_streamvln` | Python 3.10 venv |
| InternNav venv | `.../virtual_env/.venv_internnav` | Python 3.11 venv |
| 复现配置 | `NAV/config/r2r_ce_sota_reproduction.env` | 数据/权重/结果路径 |
| 复现脚本 | `NAV/scripts/eval_baselines/` | link / setup / download / run |
| 权重缓存 | `/sharedata/NAV/baselines/checkpoints/` | HF 下载落盘 |
| 评测产物 | `NAV/result/vln_ce/` | 按 method/run_id |
| 评测日志 | `NAV/log/vln_ce/` | setup 与 eval 日志 |
| 协议文档 | `NAV/doc/04_evaluation/evaluation_reproduction_and_benchmarks.md` | NAV-EVL-002 |

### 长视频数据准备区

五个长视频数据集统一配置在 `/sharedata/datasets/`：

| 数据集 | 公共目录 | 当前状态 |
| --- | --- | --- |
| DL3DV-10K | `/sharedata/datasets/DL3DV-10K` | 94GB；146个下载scene，141个Pose/Image配对episode |
| Sekai | `/sharedata/datasets/Sekai` | 89GB Pose/metadata；训练视频等待YouTube cookie |
| SpatialVID | `/sharedata/datasets/SpatialVID` | 89GB；30,000视频与标注，23,837条满足至少2 chunks |
| Ego4D | `/sharedata/datasets/Ego4D` | Ego4D CLI 环境已配置；等待 License 审批和 AWS 临时凭据 |
| RealEstate10K（长视频增量区） | `/sharedata/datasets/RealEstate10K` | 软链接复用现有 `/sharedata/RealEstate10K`，不复制、不覆盖 |
| Argoverse 2 Sensor（约100 GB子集） | `/sharedata/datasets/Argoverse2-Sensor-100GB` | 17GB、14个完整train logs；后台下载继续 |

统一环境配置为 `NAV/config/datasets.env`，独立 Python 环境为
`virtual_env/.venv_data_prep`。每个数据集的 `README_CN.md`、下载入口、
manifest 和日志分别保存在其公共目录中。

Argoverse 2 的选择清单位于
`/sharedata/datasets/Argoverse2-Sensor-100GB/manifests/selection_100gb.tsv`，
下载日志位于
`/sharedata/datasets/Argoverse2-Sensor-100GB/logs/download.log`。下载脚本为
`NAV/scripts/datasets/download_av2_100gb.sh`，使用官方公开 S3，无需账号。

Kinetics-400 的 VGGT 相机视角变化统计位于
`NAV/result/kinetics_vggt_camera_all`，中文说明见
`NAV/doc/02_data/data_preparation_schema_and_status.md`。

## NAV 当前派生训练数据

| 资源 | 路径 | 说明 |
| --- | --- | --- |
| SpatialVID Short action | `/sharedata/NAV/derived/actions/spatialvid_short` | 23,837个有效episode逐帧move/view |
| SpatialVID Short latent | `/sharedata/NAV/derived/latents/spatialvid` | GPU 1增量编码；历史payload可内嵌action，新payload使用外部JSON |
| 其他Pose数据manifest | `/sharedata/NAV/derived/manifests/remaining_pose_short.jsonl` | RE10K 269、DL3DV 141、Argoverse 2 14 |
| Pose-aligned latent | `/sharedata/NAV/derived/latents_pose_aligned` | 修正对齐后的RE10K/DL3DV/Argoverse 2 Short cache |
| Spatial单worker历史日志 | `NAV/log/data_prep/spatialvid-short-20260728` | 已被7-shard并行流水线替代 |
| Spatial并行流水线日志 | `NAV/log/data_prep/spatialvid-short-parallel-20260729` | 7个latent shard、验证与后续A/B训练 |
| 其他Pose流水线日志 | `NAV/log/data_prep/remaining-pose-20260729` | manifest、latent与验证 |

数据格式和路径的职责边界见 `NAV/doc/02_data/data_preparation_schema_and_status.md`。

已下载压缩包的 SHA-256：

```text
e2a81331524a6ca9a987d014a1affcb3b7174240cfae3647532bbaf68ef18404  R2R_VLNCE_v1-3.zip
3171f5bed90c81d7eb4db9554955b6229ec1fe96d808c9340582ba33eafdb4de  R2R_VLNCE_v1-3_preprocessed.zip
```

## 可用模型权重

| 模型或资源 | 路径 | 约占空间 | 预期用途 |
| --- | --- | ---: | --- |
| Wan2.2 TI2V 5B | `/sharedata/Wan2.2-TI2V-5B` | 32 GB | 既有 SAO 视频生成骨干网络（Backbone）和基线 |
| Wan2.2 VAE | `/sharedata/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 已计入上项 | 视频潜变量编解码 |
| StreamVGGT | `/sharedata/streamvggt` | 9.4 GB | 流式几何骨干网络和记忆机制实验 |
| LingBot-World Base Cam | `/sharedata/lingbot-world-base-cam` | 226 GB | 相机与动作条件世界模型参考 |
| LingBot-World Base Cam NF4 | `/sharedata/lingbot_world_base_cam_nf4` | 29 GB | 低显存 LingBot 推理实验 |
| LingBot-Map | `/sharedata/lingbot-map` | 14 GB | 流式几何与地图表征 |
| LingBot-Map 默认权重 | `/sharedata/lingbot-map/lingbot-map.pt` | 已计入上项 | 默认重建权重（Checkpoint） |
| LingBot-Map 第一阶段 | `/sharedata/lingbot-map/lingbot-map-stage1.pt` | 已计入上项 | 第一阶段权重 |
| LingBot-Map 长上下文版 | `/sharedata/lingbot-map/lingbot-map-long.pt` | 已计入上项 | 长上下文权重 |
| VGGT | `/sharedata/RealEstate10K/vggt_weights/model.pt` | 已计入数据集根目录 | 几何特征基线 |
| VGGT 配置 | `/sharedata/RealEstate10K/vggt_weights/config.json` | 已计入数据集根目录 | VGGT 模型配置 |
| VGGT-Omega 缓存 | `/sharedata/RealEstate10K/vggt_omega_weights` | 少量本地缓存 | 使用前需确认实际权重文件 |
| Wan2.1 T2V 1.3B | `/sharedata/Wan2.1-T2V-1.3B` | 约 16.4 GiB | 官方 `Wan-AI/Wan2.1-T2V-1.3B` 快照；480p 文生视频基线 |
| Infinite-World | `/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt` | 约 3.3 GiB | ICML 2026 官方 DiT；复用 Wan2.1 目录中内容相同的 VAE 和 UMT5 文件 |
| DreamZero 5B 所需 Wan2.1 CLIP | `/sharedata/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | 单文件 | DreamZero Wan2.2-5B 路径的 CLIP image encoder；未下载完整 Wan2.1 14B |

## 世界模型与视频评测环境

| 资源 | 代码或配置路径 | 虚拟环境 | 共享数据与缓存 |
| --- | --- | --- | --- |
| Infinite-World (`1e8fc49`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_infinite_world` | `/sharedata/Infinite-World` |
| VBench (`45e79ec`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/VBench` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_vbench` | `/sharedata/VBench` |
| DreamZero (`ab790c1`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/dreamzero` | 当前复用 `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_wan2_2` 并补装 DreamZero inference 依赖 | `/sharedata/Wan2.2-TI2V-5B`、`/sharedata/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |
| DreamDojo (`02f119b`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/DreamDojo` | 未配置；仅作为 NVIDIA DreamDojo 对照仓库 | 待按需下载 `/sharedata` 权重 |
| BridgeVLA++ memory (`8855333`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/BridgeVLA` | Conda: `/mnt/pool1/sharehome/xiewenyuan/.conda/envs/bridgevla_plus_gembench` | `/sharedata/BridgeVLA/data/bridgevla_data`、`/sharedata/BridgeVLA/data/bridgevla_ckpt` |
| Fast-WAM (`45d8e14`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/FastWAM` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_fastwam`（已安装） | `/sharedata/FastWAM`；`libero_uncond_2cam224.pt` 与 ActionDiT backbone 已可加载；smoke 结果 `NAV/result/fastwam/smoke/20260809_031029/summary.json`；脚本 `NAV/scripts/reproduce_wam/run_fastwam_smoke.sh` |
| GigaWorld-Policy-0.5 (`f3d5a88`) | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/giga-world-policy` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_gigaworld_policy`（已按 dependency conflict 修复安装） | `/sharedata/GigaWorld-Policy`；`Giga-World-Policy-0.5` transformer checkpoint 已落盘；transformer-only smoke 结果 `NAV/result/gigaworld_policy/smoke_transformer/20260809_025612/summary.json`；脚本 `NAV/scripts/reproduce_wam/run_gigaworld_transformer_smoke.sh` |

### BridgeVLA++ memoryBench

| 资源 | 路径 | 状态 |
| --- | --- | --- |
| 代码仓库 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/BridgeVLA` | 官方 `main` 分支，commit `8855333` |
| Conda 环境 | `/mnt/pool1/sharehome/xiewenyuan/.conda/envs/bridgevla_plus_gembench` | 已通过官方 import self-check；包含 PyRep/RLBench/CoppeliaSim/PyTorch3D/point-renderer |
| 数据根目录 | `/sharedata/BridgeVLA/data/bridgevla_data` | memoryBench zips 已下载；`test/put_block_back` 已展开用于 smoke |
| 权重根目录 | `/sharedata/BridgeVLA/data/bridgevla_ckpt` | BridgeVLA++ memoryBench、PaliGemma、CLIP RN50、memoryBench cache 已下载 |
| 下载脚本：权重 | `NAV/scripts/download_bridgevla_plus_memorybench_ckpt.sh` | 激活环境并下载 `memorybench paligemma clip memorybench_cache` |
| 下载脚本：数据 | `NAV/scripts/download_bridgevla_plus_memorybench_data.sh` | 官方 HF dataset 下载 + 解压入口；必要时可用 `wget -c` 续缺失 zip |
| 架构文档 | `NAV/doc/01_model/model_evolution_and_current_architecture.md` | NAV-ARC-004 |
| 复现记录 | `NAV/doc/04_evaluation/evaluation_reproduction_and_benchmarks.md` | NAV-EVL-003 |
| smoke 结果 | `NAV/result/bridgevla_plus_memorybench/smoke_25step/model_160/seed608/result.jsonl` | `put_block_back` 4 variations，4/4 success |

Wan2.1-T2V-1.3B 于 2026-07-23 完成下载验证：

- DiT: `/sharedata/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors`
  (`5,676,070,424` bytes, 825 safetensors keys);
- UMT5: `/sharedata/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth`
  (`11,361,920,418` bytes);
- VAE: `/sharedata/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`
  (`507,609,880` bytes).

UMT5 和 VAE 通过硬链接（Hard Link）复用同一公共文件系统中内容相同的文件，
避免重复占用空间；Wan 1.3B DiT 下载自 Hugging Face 官方仓库。

Infinite-World 在 `Infinite-World/configs/infworld_config.yaml` 中使用共享权重
的绝对路径，本地启动脚本会自动选择上述虚拟环境。短流程冒烟测试
（Smoke Test）命令如下：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World
INFWORLD_MAX_PROMPTS=1 INFWORLD_NUM_CHUNKS=1 INFWORLD_SAMPLING_STEPS=2 bash infer_local.sh 1
```

该命令已于 2026-07-23 在 RTX 6000 Ada 上验证。权重加载结果为
`Missing: 0, Unexpected: 0`，成功生成一段 896×448、81 帧的 H.264 视频，
输出至：

`/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/Infinite-World/outputs/infworld-ckpt0-step2-cfg5.0/0000_A_serene_campus_walkway_lined_.mp4`

VBench 共享缓存变量定义在 `VBench/config/sharedata.env`。共享评测元数据和
提示词集合位于 `/sharedata/VBench/datasets`；下载的 Hugging Face 与 Torch
权重均重定向至 `/sharedata/VBench/cache`。自定义视频评测命令如下：

```bash
cd /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/VBench
scripts/run_custom_eval.sh /path/to/video_or_directory temporal_flickering
```

2026-07-23 的 GPU 冒烟测试结果：

- 输入：`/sharedata/VBench/smoke_videos/testsrc.mp4`；
- 指标：时序闪烁（Temporal Flickering，`temporal_flickering`）；
- 得分：`0.9465423396989411`；
- 结果：`/sharedata/VBench/results/results_2026-07-23-20:56:14_eval_results.json`。

已在本地扩展 VBench 环境检查，使其接受服务器的 PyTorch CUDA 12.4 构建。
依赖 Detectron2 的指标仍受 VBench 上游 CUDA 12.1 限制；上述已验证指标不依赖
Detectron2。

此外，Infinite-World 已在两个官方演示条件上以 30 个采样步运行，并使用
VBench 六项技术指标子集完成评测。正式结果文件为
`NAV/result/vbench/infiniteworld_prior_runs/metrics/infinite_world_30step_subset/results_2026-07-23-22:15:19_eval_results.json`；
完整评测方案、命令、分数和可比性限制记录在
`NAV/doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`。

同一文档还记录了对原论文 VBench 协议的核对，以及一个 1 场景、2 chunks、
161 帧的小规模复现。论文同款四项平均分为 `0.842552`，原始结果位于
`NAV/result/vbench/infiniteworld_single_2chunks/metrics/results_2026-07-23-22:54:10_eval_results.json`。

2026-07-25 已用完全相同协议评测 NAV A/B 的 step 1000 全参权重。迁移后的完整
视频、指标和日志分别位于 `NAV/result/vbench/nav_a_single_2chunks` 和
`NAV/result/vbench/nav_b_single_2chunks`。InfiniteWorld 对照位于
`NAV/result/vbench/infiniteworld_single_2chunks`。完整分数、差值和可比性说明
仍记录在 `NAV/doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`。

## 本地参考仓库

以下仓库均与 `NAV` 同级，作为参考或依赖使用，不应复制到 `NAV` 内部。

| 仓库 | 路径 | 用途 |
| --- | --- | --- |
| 既有 SAO 实现 | `../SAO` | 既有尝试、配置、日志和权重 |
| Causal-Forcing | `../Causal-Forcing` | 因果式或流式视频生成 |
| Self-Forcing | `../Self-Forcing` | 自回归滚动生成训练 |
| GeometryForcing | `../GeometryForcing` | 视频特征的几何监督 |
| Wan2.2 | `../Wan2.2` | 视频扩散骨干网络 |
| FantasyWorld | `../fantasy-world` | 视频与几何统一建模 |
| VGGT-Omega | `../vggt-omega` | 相机令牌与寄存器令牌设计 |
| StreamVGGT | `../StreamVGGT` | 流式几何模型 |
| LingBot-World | `../lingbot-world` | 交互式世界模型 |
| LingBot-Map | `../lingbot-map` | 流式三维重建 |
| NavDP | `../NavDP` | 导航策略与基线 |
| LDA-1B | `../LDA-1B` | 潜空间动力学与动作建模 |
| DreamZero | `../dreamzero` | 基于世界模型的机器人策略参考 |
| Infinite-World | `../Infinite-World` | ICML 2026 长时程动作条件世界模型 |
| VBench | `../VBench` | 官方视频生成质量评测基准与提示词集合 |
| RELIC | （待跟踪，无开源） | 2025-12 arXiv 2512.04040，Adobe+Google，14B 长程交互世界模型；截至 2026-08 无公开代码/权重，仅有项目页 https://relic-worldmodel.github.io/ 与网站源码仓 `Relic-WorldModel/relic-worldmodel.github.io`（HTML）。论文级参考，待官方发布后 clone 到 `../relic` 并配 `virtual_env/.venv_relic` |

### 2026-08-29 流式 3D 与 navigation 官方复现资源

| 项目 | 仓库 revision | 虚拟环境 | 权重 | 已验证结果 |
| --- | --- | --- | --- | --- |
| LingBot-Map | `../lingbot-map` @ `1740f18ead3cca8e0e07dd0ec6b7030d22f0894c` | `../virtual_env/.venv_lingbot_map` | `/sharedata/lingbot-map/lingbot-map.pt` | `NAV/result/streaming_3d_reproduction/lingbot_map/official_university_24f_sdpa_20260829/` |
| ABot-Recon | `../ABot-Recon` @ `cd6a99f889aee7a165f59b414d4b351caa1c2956` | `../virtual_env/.venv_abot_recon` | `/sharedata/ABot-Recon/checkpoints/abot_recon.safetensors` | `NAV/result/streaming_3d_reproduction/abot_recon/official_university_24f_sdpa_20260829/` |
| LoGoPlanner | `../NavDP` @ `878740a2011856d0e3782dd6ccd880fd2eccd70f`；Pi3 @ `b412c3bd236dfd7686f1e4b48004d5087f2fa093` | `../virtual_env/.venv_navdp` | `/sharedata/LoGoPlanner/modelscope/logoplanner_policy.ckpt` | `NAV/result/streaming_3d_reproduction/logoplanner/official_nyuv2_rgbd_12f_torch251_20260829/` |

三者均已在 GPU 1 运行官方完整权重前向。统一启动入口为
`NAV/scripts/reproduction/run_{lingbot_map,abot_recon,logoplanner}_gpu1.sh`，日志在
`NAV/log/reproduction/streaming_3d_20260829/`；模型机制、精确 token 形状、训练监督、
sample 构造与复现限制记录在
`NAV/doc/04_evaluation/streaming_3d_reconstruction_and_navigation_reproduction.md`。

## 路径约定

- 在 `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV` 下执行 NAV 命令。
- `/sharedata` 下的数据集和预训练权重统一使用绝对路径。
- 本地参考仓库使用 `../Wan2.2` 这类同级相对路径。
- 新实验配置存放在 `config/`。
- 源代码存放在 `src/`，启动和评测工具存放在 `scripts/`，实验输出存放在
  `log/`。
- 不要在 NAV 源代码目录中存放大型下载权重或数据集。

### 实验结果目录原则

从 2026-07-26 起，所有推理、评测和可视化等结果类产物统一保存在：

```text
NAV/result/<测试大类>/<详细实验名>/
```

每个详细实验目录使用以下标准子目录：

```text
videos/   # 生成视频
metrics/  # 指标 JSON、统计汇总和报告
logs/     # 推理及评测日志
```

例如 VBench 三版统计实验使用：

```text
NAV/result/vbench/nav_a_stats10/
NAV/result/vbench/nav_b_stats10/
NAV/result/vbench/infiniteworld_stats10/
NAV/result/vbench/threeway_stats10/
```

目录层级的含义固定为：

1. `result/`：所有结果类内容的统一根目录；
2. `vbench/`、`inference/`、`visualization/` 等：测试大类；
3. `nav_a_stats10/` 等：可独立复现的详细实验；
4. `videos/`、`metrics/`、`logs/`：按产物类型划分。

数据集、预训练权重和公共评测模型仍放在 `/sharedata`；源代码、配置和启动脚本
仍分别放在 `src/`、`config/` 和 `scripts/`，不得混入 `result/`。历史 JSON
内部保存的旧绝对视频路径仅作为运行记录保留，实际文件位置以本节为准。

### Kinetics-400 VGGT 全量标注

- 原视频：`/sharedata/datasets/kinetics400/{train,val,test}_256/`
- VGGT 仓库：`/sharedata/RealEstate10K/vggt`
- VGGT 权重：`/sharedata/RealEstate10K/vggt_weights/model.pt`
- RE10K 置信度标定：`NAV/config/vggt_re10k_confidence.json`
- 全量标注结果：`NAV/result/kinetics_vggt_camera_all/metrics/`
- 后台日志：`NAV/result/kinetics_vggt_camera_all/logs/`
- 启动脚本：`NAV/scripts/run_kinetics_vggt_all.sh`
- 详细定义：`NAV/doc/02_data/data_preparation_schema_and_status.md`

### 多数据集训练派生数据

- 统一 episode manifest：`/sharedata/NAV/derived/manifests/episodes.jsonl`
- Manifest 汇总：`/sharedata/NAV/derived/manifests/episodes.summary.json`
- Wan VAE latent：`/sharedata/NAV/derived/latents/<dataset>/`
- 增量预处理入口：`NAV/scripts/run_incremental_data_prep.sh`
- 旧 Curriculum 配置与 A/B 训练入口：已从当前代码删除，仅作为历史记录保留
- 详细说明：`NAV/doc/03_training/training_plan_and_experiment_log.md`

上述目录均为 NAV 派生产物。原始下载目录保持只读；新增数据下载完成后重新运行增量
预处理入口即可扫描并缓存新增 episode。

## 资源验证清单

资源用于实验前应记录：

- [ ] 准确的权重（Checkpoint）或数据集版本；
- [ ] 许可证和允许的使用范围；
- [ ] 输入与输出格式；
- [ ] 预处理要求；
- [ ] 可用时记录校验和（Checksum）或不可变代码版本（Revision）；
- [ ] 已成功执行的本地加载命令；
- [ ] 推理与训练所需 GPU 显存。
