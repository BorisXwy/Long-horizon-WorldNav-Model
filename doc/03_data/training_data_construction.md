# NAV 训练数据构建规范

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-005` |
| 类型 | 数据规范（Data Construction Specification） |
| 状态 | Active / Source of Truth |
| 更新时间 | 2026-08-10 |
| 职责 | 定义从落盘视频、Pose、Action 到多 Chunk VAE latent 的唯一数据语义 |

## 当前结论

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

## Chunk 与时间语义

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

## 离线与在线边界

### 可以离线缓存

- 连续 RGB frame indices 或 image paths；
- 每个 chunk 的干净 VAE latent；
- 与目标 RGB 帧一一对应的 `move/view`；
- caption/text embedding；
- 数据集名、episode ID、窗口起点和原始 chunk 数。

### 不能离线缓存

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

## 训练窗口选择

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

## 各数据集对齐规则

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

## 派生数据格式

### Action JSON/manifest

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

### Latent PT

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

## 当前派生目录

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

## V1 T4 micro chunk 语义

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

## 完整性与可复现要求

进入训练前至少验证：

1. `sample_id` 唯一；
2. chunk tensor 可加载、有限且非全零；
3. `chunks.shape[1] == cached_chunks`；
4. `len(move) == len(view) == cached_chunks × 81`，或外部 sidecar 能提供；
5. RGB/Pose 时间顺序严格一致；
6. 每条样本来自单一 episode；
7. 随机窗口由固定 seed 或稳定哈希决定；
8. 原始下载目录没有被预处理修改。

## 当前限制与后续必须完成

- 尚未建立正式 scene-level train/validation/test split，正式指标前必须补齐；
- SpatialVID 数量占绝对多数，训练必须进行 dataset-aware sampling；
- Action 是相机轨迹伪标签，不代表真实键盘或机器人控制；
- 跨数据集的尺度归一化当前按 episode 中位运动计算，语义一致但绝对速度不可比；
- Short latent 不是完整长视频缓存，后续不能直接据此宣称完成 Long curriculum；
- 训练前应冻结数据版本清单、样本数、哈希规则和异常排除列表。

相关标注定义见 `action_annotation_specification.md`，在线训练语义见
`../04_training/v0_streaming_training_sample_semantics.md`。
