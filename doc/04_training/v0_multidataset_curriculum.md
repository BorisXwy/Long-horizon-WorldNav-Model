# V0 多数据集增量预处理与分阶段训练

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-001` |
| 类型 | 训练规范（Training Specification） |
| 状态 | Historical Baseline / 旧入口已从当前代码删除 |
| 更新时间 | 2026-07-29 |
| 职责 | 定义增量 latent、shuffle 和 Short/Medium/Long 课程训练 |

训练样本中 Register 的在线递归、teacher forcing、last-target 与推理差异，以
`v0_streaming_training_sample_semantics.md` 为唯一事实来源；数据 tensor 和
Action 的定义分别以 `../03_data/training_data_construction.md` 与
`../03_data/action_annotation_specification.md` 为准。

本文件保留 V0 多数据集课程训练历史。文中的旧配置和训练脚本已从当前 NAV
代码主线删除，不再作为 V1 训练入口。

## 数据隔离原则

DL3DV、SpatialVID、Argoverse 2 和 RE10K 的下载目录只读使用。所有派生产物统一写入：

```text
/sharedata/NAV/derived/
├── manifests/episodes.jsonl
└── latents/<dataset>/<sample_id>.pt
```

`prepare_multidataset_manifest.py` 每次执行都会重新扫描当前完整 episode，并使用
临时文件原子替换 manifest。因此下载中新增加的数据会在下次执行时自动出现；未完成
episode 不进入 manifest。脚本不会移动、重命名或修改下载文件。

`cache_multidataset_latents.py` 以 `dataset/sample_id` 和 sidecar 中的
`cached_chunks` 判断完成度。Short 阶段最多写入 3 chunks；Medium 只补第
4–7 chunks；Long 只补第 8 chunk 以后的内容。更新使用临时文件原子替换，已经
编码的 chunks 不重复计算。RE10K 已有完整 latent 使用硬链接复用。

以下是2026-07-27旧统一流水线的历史 manifest，共4,327条；不代表当前
SpatialVID Short 的数据总量：

| 数据集 | Episode |
| --- | ---: |
| RE10K | 269 |
| DL3DV | 50 |
| SpatialVID | 3,997 |
| Argoverse 2 | 11 |

Kinetics 当前只有 episode 级 VGGT 运动统计，没有完整逐帧外参，因此暂不伪造
`move/view`，也不进入本轮有 Pose 的训练 manifest。

## 统一格式

每条 manifest 保存视频或图像源、所选帧、离散 `move/view`、帧数和 chunk 数。
每个 chunk 固定 81 帧，对应 Wan VAE 的 21 个 latent 时间位置。

- DL3DV：将 `transform_matrix` 的 camera-to-world 转为 world-to-camera；
- SpatialVID：使用官方 `poses.npy` 的 world-to-camera，并用 Slerp 将 60 个稀疏
  Pose 插值到原视频帧；
- Argoverse 2：将 `city_SE3_egovehicle` 与前向相机外参组合，使用
  `ring_front_center`；
- RE10K：复用既有 GT Pose 生成的 `move/view`。

## 增量预处理命令

```bash
bash NAV/scripts/run_incremental_data_prep.sh cuda:1 0 1
```

多 GPU 时第二、三个参数分别是 shard index 和 shard 总数。新增下载完成后重复同一
命令即可，只会编码新增样本。

## Shuffle 与长上下文窗口

训练脚本不再按文件排序固定循环。它使用固定 `shuffle_seed`：

1. 每个 epoch 洗牌全部合格缓存；
2. 每个 batch 在允许范围内随机选择连续 chunk 数；
3. 每条 episode 随机选择窗口起点；
4. 窗口最后一个 chunk 是 diffusion target；
5. 之前所有 chunk 按时间顺序递归更新 Register；
6. local memory 始终取目标前最新 chunk 的最后一个 latent frame。

因此 6-chunk episode 在短阶段也可以贡献随机 2–3 chunk 片段，而不是只能进入
中等阶段。

## A/B 三阶段 Curriculum

### DL3DV history-extension 测试

此前基于 `images_8` 稀疏 SfM 帧得到的3–5 chunk口径已废止。canonical v1
改用完整原视频固定分块缓存；DL3DV 当前146条原视频平均约52.97 chunks。
只有 `full_episodes_v1` latent 与 dense Action 完成后，才启动6-chunk以内的
A/B history-extension 测试。

### 当前优先阶段：SpatialVID Short

在进入原三阶段多数据集课程前，先执行 SpatialVID 的 2–3 chunk
short-history 训练。A/B 不再从原始 InfiniteWorld 初始化，而是分别继承已经
完成的 RE10K 1000-step 全参数 checkpoint：

```text
A: NAV/log/re10k-all-a-full-ebs4-from-infinite/full-final.pt
B: NAV/log/re10k-all-b-full-ebs4-from-infinite/full-final.pt
```

每条落盘 SpatialVID episode 以 episode ID 的稳定哈希选择一个连续窗口：

```text
2-chunk: [teacher-forced C0] → target C1
3-chunk: [teacher-forced C0, C1] → target C2
```

离线并行准备两类相互独立的产物：

```text
/sharedata/NAV/derived/latents/spatialvid/<sample_id>.pt
/sharedata/NAV/derived/actions/spatialvid_short/<sample_id>.json
```

VAE latent 与逐帧 `move/view` 可以并行计算；Register 不落盘。训练 forward
从 checkpoint 中的 learnable initial state 开始，依次用干净 history latent
在线更新 Register，local memory 取最新 history chunk 的最后一个 latent
frame，只对窗口末尾 target 做一次 RFlow diffusion loss。chunk 之间不
detach，因此 target loss 会穿过全部 Register updates 反传。

以上描述仅对应当前尚未迁移的历史训练代码。新的 Stage One 结构必须改为首个
History Chunk 经 Extractor 产生 Register，后续 Chunk 才经 Updater；迁移后
课程数据与 last-target 口径不变。

本阶段采用 `random-prefix next-chunk teacher forcing` 的 last-target 版本，
而不是在训练中完整 diffusion rollout 历史。配置唯一事实来源为
`config/train_spatialvid_short.yaml`，持久后台入口为：

```bash
tmux new-session -d -s nav_spatialvid_parallel \
  "bash NAV/scripts/run_spatialvid_short_parallel_pipeline.sh"
```

流水线以7个稳定shard并行准备 latent（GPU 0三个、GPU 1四个），配对校验通过
后释放VAE worker，并在GPU 1上依次执行A、B。两者均为全参数训练、effective
batch size 4、最多20,000 steps，并允许loss平台后提前结束。

下表保留原三阶段 curriculum 目标。其旧 Stage 1“从原始 InfiniteWorld 初始化”
已被当前 SpatialVID Short 决策替代：当前 A/B 必须分别从各自 RE10K 1000-step
checkpoint 继续；两者仍不互相加载权重。Medium/Long 的初始化策略需在 Short
结果确认后冻结为新配置。

| Curriculum Phase | 连续片段 | 数据资格 | 最大步数 | 转换条件 |
| --- | --- | --- | ---: | --- |
| Short | 2–3 chunks | 至少 2 chunks | 20,000 | loss 进入平台 |
| Medium | 4–7 chunks | 至少 4 chunks | 15,000 | loss 进入平台 |
| Long | ≥8 chunks | 至少 8 chunks | 10,000 | loss 进入平台或到上限 |

“几乎收敛”定义为：至少训练规定最小步数后，以 200 step 平均 loss 为窗口；连续
3 个窗口的相对改善不足 1% 时进入下一阶段。Medium/Long 加载上一阶段的完整
backbone 和 Register，优化器在阶段边界重新初始化。

以下原课程数量为 2026-07-27 的历史基线；在 SpatialVID Short 完成后重新统计：

- Short 可用：4,327；
- Medium 可用：3,101；
- Long 可用：1,409。

单独调试某一完整课程时可使用：

```bash
bash NAV/scripts/run_curriculum_train.sh latent_prefix 0 experiment_a
bash NAV/scripts/run_curriculum_train.sh dit_condition 1 experiment_b
```

正式实验的唯一推荐入口是分阶段后台流水线：

```bash
tmux new-session -d -s nav_staged_pipeline \
  "bash NAV/scripts/run_staged_background_pipeline.sh"
```

执行依赖为：

```text
Short latent ready
├── GPU 0：A / Short
└── GPU 1：扩展 Medium latent
             ↓
          扩展 Long latent
             ↓
          B / Short → Medium → Long
```

A 在各阶段等待对应 ready marker，B 在完整 latent 后使用 GPU 1。任何 latent
缺失或编码错误都会触发增量重扫，不会用不完整数据静默训练。状态与日志统一位于
`NAV/log/full_pipeline/`。
