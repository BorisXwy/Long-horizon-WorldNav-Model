# V1 三阶段训练数据、组织与 Loss 设计

| 字段 | 内容 |
| --- | --- |
| ID | NAV-TRN-006 |
| 类型 | 训练数据与损失设计 |
| 状态 | Accepted Draft / V1 Final Training Skeleton |
| 更新时间 | 2026-08-14 |
| 职责 | 定义 V1 GigaWorld-like + Register NAV 的 Stage One/Two/Three 训练数据来源、样本组织、缓存格式、loss 设计和数据配比 |

## 核心结论

V1 不再围绕 81-frame dense video chunk 构造训练，而围绕 Register memory、
action-compatible interface 和 sparse future consequence 构造：

```text
history/current obs/instruction
  -> Register + fixed action-compatible tokens
  -> generation branch 预测 sparse future consequence
  -> Stage Three policy branch 预测 action/control
```

三阶段数据职责如下：

| 阶段 | 主目标 | 主要数据 | 推理保留 |
| --- | --- | --- | --- |
| Stage One | 让 shared DiT/Register 学会长历史视频记忆和 future consequence；action 接口固定存在，但 action output 不训练成 policy | DL3DV、SpatialVID、RE10K、Argoverse2 | generation；action interface present but not policy |
| Stage Two | 在同一视频生成前向中加入 3D supervision，找可用于导航的 Register/hidden layer | DL3DV、RE10K、SpatialVID、Argoverse2，配合 VGGT/pose pseudo label | generation + 3D probe；action interface fixed |
| Stage Three | 用 VLN 数据训练 policy/action output，同时 replay Stage One/Two 数据与 loss 防止 shared backbone/Register 退化 | R2R-CE、RxR-CE、ScaleVLN、LHPR-VLN + Stage One/Two replay pool | 训练 cotrain；推理 policy-only，generation branch 删除 |

V1 需要新建独立派生目录，不覆盖 V0：

```text
/sharedata/NAV/derived/v1/
  manifests/
  frame_packs/
  vae_packs/
  actions/
  geometry/
  vln/
  audits/
```

旧 `full_episodes_v1` 81-frame latent 属于 V0 dense chunk 口径，不作为 V1 主缓存。
早期 V1 sparse pack `[obs, future_1, future_2, future_3, future_4]` 已完成
smoke，但主线训练改为 `T_latent=4` micro chunk；具体规则见
`v1_stage_one_t4_iw_aligned.md`。

一个必须遵守的约束是：**pose 不是模型推理输入**。Pose 只在训练数据准备和
监督中使用：

```text
pose 可用于:
  pseudo action / displacement bucket 构造
  3D supervision target
  数据审计和尺度校准

pose 不可用于:
  Stage Three policy condition
  真实世界推理必需输入
  无 pose benchmark 的必需输入
```

V1 的 history/Register 因此不是 `pose-conditioned memory`，也不是把当前 GT action
塞进 policy condition。它应是：

```text
implicit Register storage
previous-action/motion-aware update via A_hist tokens when available
pose-supervised when available
```

真实世界中没有 GT pose 是正常情况；但 agent/robot 通常知道自己执行过的
action command，可能还可选获得 odometry/IMU。V1 默认允许历史已执行 action/motion
以 `A_hist` tokens 参与 Register update，但禁止 action bias / latent bias 注入。
当前要预测的 target action 不应在 Stage One 被强行训成 policy；它在视频生成数据中
最多作为 generation condition 或格式占位。真正的 policy/action output 在
Stage Three 用 VLN 数据训练。

## 统一样本结构

Action、GeometryTarget、VaePack 和 V1Sample 的统一字段定义见
`../03_data/v1_action_geometry_schema.md`。本文件只描述三阶段如何使用这些
schema 训练。

## History / Current / Target 时间边界

V1 正式样本必须采用下面的流式边界，避免 latest current chunk 同时进入
Register 和 Local：

```text
给定连续 micro chunks:
  C_0, C_1, ..., C_{t-2}, C_{t-1}, C_t

用于预测 C_t 时:
  Register history = C_0 ... C_{t-2}
  Current / Local  = C_{t-1}
  Target future    = C_t
```

也就是：

```text
R_{t-2} = RegisterCell(... RegisterCell(R_null, C_0, A_hist0) ..., C_{t-2}, A_hist_{t-2})
z_obs   = latent(C_{t-1})
z_future_target = latent(C_t)
```

禁止把 `C_{t-1}` 既写进 Register 又作为 `z_obs/current` 输入。Register 承担
更早历史压缩，Current/Local 承担最近观测细节。

这个边界也决定 Stage Two 的 3D supervision：主要监督窗口放在
`C_{t-1}` current chunk 上，而不是放在被 Register 压缩后的老历史上。理由是
current chunk 保留了最完整的空间 token/grid hidden，最接近 VGGT-Ω 中每帧
Patch Tokens + Camera/Register Tokens 被 camera/depth heads 读取的结构。

## 按数据集确定的训练变量表

以下表格是数据构建和 dataloader 的准绳。`EMPTY` 表示该变量以空 embedding /
unknown embedding / zero mask 的确定形式进入，不再称为“可选”；`不参与` 表示该
数据集不进入该阶段或该变量不构建。

### Stage One / Stage Two：视频数据变量

| 数据集 | Stage One | Stage Two | `C_hist` 历史视频 chunk | `A_hist` 历史 action/motion | `C_local` 当前/local chunk | `A_cur` 视频生成当前 action condition | `A_noise` action 主变量 | `Z_future` 视频 target | `Text` | `G_3D` 三维监督 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DL3DV | 使用 | 使用 | T4 micro latent windows | 由 GT pose 差分生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 pose 差分 pseudo motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | GT pose/intrinsic 生成 relative pose/depth/point target |
| SpatialVID 20% | 使用 | 使用 | T4 micro latent windows | 使用落盘 action/motion 生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 action/motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | 使用已有稀疏 pose 或 VGGT pseudo geometry，带 confidence mask |
| RE10K | 使用 | 使用 | T4 micro latent windows | 由相机 pose 差分生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 pose 差分 pseudo motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | camera pose/intrinsic 生成 relative pose/depth/point target |
| Argoverse2 | 使用 | 使用 | T4 micro latent windows | 由 ego pose / vehicle motion 生成 `A_hist` tokens；参与 Register update，no bias | 最后一个 history chunk 的 local latent | target window 的 ego-motion condition | FORMAT_ONLY，loss=0 | next T4 micro latent | EMPTY | metric ego pose / camera calibration 生成 3D target |
| Kinetics | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 |

Stage One 的固定 loss 规则：

```text
L_stage1 = L_visual_flow
action output branch:
  A_cur 必须进入 DiT condition/context；
  A_noise tokens 必须进入 action stream / shared-coupled DiT backbone；
  A_out 从 action stream hidden 经 Giga/DreamZero-style action flow decoder 读出；
  action loss = 0；
  不把视频 pseudo action 当 policy supervision。
```

Stage Two 在 Stage One forward 基础上增加：

```text
L_stage2 = L_visual_flow + λ_3d L_3D
```

### Stage Two current-chunk 3D 监督

参考 VGGT-Ω 的设计，Stage Two 不只监督一个全局 register 向量，而是在 shared
backbone 的若干候选层读取：

```text
current visual hidden tokens     # 对应 C_{t-1} 的 patch/latent-grid tokens
register-after-backbone tokens   # 历史上下文汇聚后的 scene/memory tokens
```

推荐第一版 probe：

```text
current hidden grid
  -> DepthHead / PointHead
  -> depth / point_map / confidence

register + pooled current hidden
  -> Camera/PoseHead
  -> relative pose / camera motion
```

其中 dense depth/point supervision 只作用在 current chunk 的可见视角上；
relative pose/camera supervision 可使用 current chunk 内帧间 pose，或
`C_{t-1} -> C_t` 的短期相机运动。Pose/depth/point 仍然只作为训练监督，不作为
推理输入。

Stage Two dataloader 因此需要为每个样本额外构造：

```text
geometry_window = C_{t-1}
geometry_target:
  intrinsics_current
  poses_current
  relative_poses_current
  depth_current or point_map_current
  confidence / valid_mask
  teacher_source
```

可用 teacher 来源按优先级：

```text
1. simulator / GT pose + depth
2. dataset camera pose + sparse/interpolated depth/point target
3. VGGT/VGGT-Ω pseudo depth / point map / confidence
```

Loss 第一版：

```text
L_3D =
  λ_depth * masked_scale_shift_depth_loss(D_pred, D_gt)
+ λ_point * masked_point_l1_or_l2(P_pred, P_gt)
+ λ_pose  * relative_pose_loss(T_pred, T_gt)
+ λ_conf  * confidence_calibration_loss(optional)
```

VGGT-Ω 给我们的直接启发是：保留 dense current tokens 做 3D readout，同时用
Register/scene tokens 提供跨视角上下文；不要只用一个 compressed Register 去
恢复全部空间细节。

### Stage Three：VLN 数据变量

| 数据集 | Stage Three | `C_hist` 历史观测 | `A_hist` 历史 action | `C_local` 当前观测 | `Instruction/Text` | `A_cur` 视频生成当前 action condition | `A_noise` action 输入 | `A_out` 监督 | `Z_future` 视频 target | `G_3D` 三维监督 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2R-CE | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | R2R instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| RxR-CE | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | RxR instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| ScaleVLN | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | ScaleVLN instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| LHPR-VLN | 使用 | 历史 RGB/全景观测进入 Register update | 已执行 simulator actions 作为 `A_hist` tokens 参与 Register update，no bias | 当前 RGB/全景观测 | LHPR instruction | 不参与 | noised action chunk `H_nav=10` | action flow 监督；可附加离散 CE | 不参与 | 不参与 |
| MP3D scene assets | 场景资源 | 渲染来源 | 不单独提供 | 渲染来源 | 不单独提供 | 不参与 | 不参与 | 不参与 | 不参与 | 不参与 |

Stage Three 的固定 policy batch 规则：

```text
VLN policy batch:
  L_nav = L_action_flow(A_noise -> A_out, gt_actions)
  可附加 discrete CE / STOP balance auxiliary，但不作为唯一主结构
  不构建 Z_future，不把 future visual/3D 作为 policy condition。

Replay batch:
  从 Stage One/Two 数据池采样 video/3D windows；
  继续计算 L_visual_flow / L_3D；
  用于保持 shared backbone/Register 的 video memory 与空间表征。
```

Stage Three 总 loss：

```text
L_stage3 =
  L_action_flow
+ optional λ_ce L_discrete_ce_aux / λ_stop L_stop_balance
+ λ_replay_video L_visual_flow_replay
+ λ_replay_3d   L_3D_replay
```

推理时：

```text
只执行 policy/action path；
generation/video/3D heads 不执行。
```

### Video/action 样本

每个 video 样本统一为：

```text
episode_id
dataset_name
current_frame_index t
history_indices h_0...h_m
future_sparse_indices f_1...f_4
pose/intrinsic for selected frames, if available
pseudo_action a_{t:t+H_action-1}, H_action=10
text/caption or empty prompt
scale metadata
confidence metadata
```

进入模型：

```text
Register:
  R_{-1} = fixed R_null
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  no action/latent/additive bias

visual/main stream:
  [Register tokens]
  [Z_obs clean]
  [Z_future noisy]

action stream:
  [A_noise]  # action chunk 的待生成/去噪变量，不是 policy condition

condition:
  text / timestep / modality / dataset / action-scale
```

其中 V1 generation branch 当前主线固定：

```text
RGB micro chunk:
  13 frames, stride 12 with adjacent chunks

VAE latent:
  history C_i and target C_target

shape:
  C_i      = [B,16,4,H_lat,W_lat]
  C_target = [B,16,4,H_lat,W_lat]
```

`T_latent=4` 对应13帧短期 future consequence。长程能力通过 Register 覆盖的
history/context span 与 InfiniteWorld 对齐，而不是通过增大 noisy target T。

### VLN policy 样本

每个 VLN 样本统一为：

```text
dataset_name
episode_id
scan/scene_id
instruction
step index s
current observation obs_s
history observation/action prefix
target low-level actions a_s...a_{s+H_nav-1}, H_nav=10
future observations: 不构建，第一版 Stage Three 不启用 generation auxiliary
pose/depth/goal distance: 不作为输入；第一版不构建 3D/progress auxiliary
```

Stage Three 推理时：

```text
generation branch:
  disabled, T_future = 0

policy input:
  obs_s + instruction + Register + A_noise tokens

policy output:
  action velocity/noise -> denoise/decode -> action chunk [B,H_nav]
```

默认：

```text
H_nav = 10
执行策略 = first-action execution
```

也就是模型预测 10 步，但闭环只先执行第 1 步，再用新 observation 更新 Register。

## V1 数据预处理与组织

### Step 1：Source manifest

所有数据先生成 source manifest：

```text
/sharedata/NAV/derived/v1/manifests/source_<dataset>.jsonl
```

每行包含：

```json
{
  "dataset": "...",
  "episode_id": "...",
  "frame_paths": ["..."],
  "poses": "... or inline summary",
  "intrinsics": "...",
  "fps": 30.0,
  "has_metric_scale": true,
  "has_instruction": false,
  "split": "train"
}
```

### Step 2：Action / displacement 标注

视频数据不按固定帧数切，而按空间位移 bucket 构造 future：

```text
small:
  约 1 个 Habitat forward，≈0.25m 或 episode-normalized 等价位移

medium:
  约 2-4 个 Habitat forward，≈0.5-1.0m 或等价位移

rotation:
  约 30° / 60° / 90°
```

有 metric pose 的数据：

```text
使用 meter/radian
```

无可靠 metric scale 的数据：

```text
使用 episode median motion 归一化
```

无 pose 数据：

```text
优先尝试 VGGT / optical flow / inverse dynamics 生成 pseudo motion；
若 confidence 不足，则不构造 motion-aligned history 样本。
```

action 标注输出：

```text
/sharedata/NAV/derived/v1/actions/<dataset>_actions.jsonl
```

### Step 3：Sparse frame pack manifest

为 Stage One/Two 建立 sample manifest：

```text
/sharedata/NAV/derived/v1/manifests/stage1_video_packs.jsonl
/sharedata/NAV/derived/v1/manifests/stage2_3d_packs.jsonl
```

每个 sample 明确记录：

```text
obs frame
4 个 future sparse frames
history rollout frames
action horizon
displacement bucket
dataset weight
confidence
```

### Step 4：VAE pack cache

V1 不以 81-frame chunk 为主缓存，而缓存 5-frame sparse pack：

```text
/sharedata/NAV/derived/v1/vae_packs/<dataset>/<sample_id>.pt
```

建议字段：

```python
{
  "z_obs": Tensor[16,1,H_lat,W_lat],
  "z_future": Tensor[16,1,H_lat,W_lat],
  "pack_frame_indices": [t, f1, f2, f3, f4],
  "actions": Tensor[10, action_dim_or_labels],
  "poses": ...,
  "intrinsics": ...,
  "scale": ...,
  "confidence": ...
}
```

训练初期可 on-the-fly VAE encode 做 smoke；正式训练应缓存 pack latent，避免每步
重复跑 VAE。

### 无 pose / 无 action 数据的降级路径

无 pose 或无 action 的视频不能作为主 Register history 训练样本，因为模型无法知道
观测变化由什么 motion 造成。它们可以按置信度分三类使用：

| 数据状态 | 使用方式 | 是否参与 Register update 主损失 |
| --- | --- | --- |
| 有真实 action/odometry 或可靠 pose | 主训练样本，监督 action + generation + 3D | 是 |
| 无 GT pose，但 VGGT/flow/inverse dynamics 产生可靠 pseudo motion | 低权重样本，带 confidence mask | 可少量参与 |
| 无 pose、无 action、无可靠 pseudo motion | 只做弱 generation/caption/domain regularization | 否 |

对于 generation benchmark 如果没有 action/pose：

```text
action token = null / no-op / text-implied motion
Register update = 不作为空间历史能力评估
指标用途 = 只看视觉质量或通用生成能力，不证明导航式 memory
```

这类 benchmark 可以跑，但不能用来证明 V1 的 action-stream-conditioned generation
或 spatial memory。

### Step 5：3D pseudo label cache

Stage Two 需要：

```text
/sharedata/NAV/derived/v1/geometry/<dataset>/<sample_id>.pt
```

可包含：

```text
depth / point map / relative pose / camera trajectory / confidence mask
```

来源优先级：

```text
1. 数据集真实 pose/intrinsic
2. 可由 pose 多视角推导的 relative geometry
3. VGGT pseudo depth/point/camera，带 confidence mask
```

真实 pose、VGGT camera、depth 或 point map 只作为 teacher signal。Stage Two forward
接口不能把 GT pose 当作 condition 输入给 policy/action tokens。

## Stage One：Video generation memory pretraining

### 目标

Stage One 训练 shared DiT/Register 的视频记忆与 consequence generation 能力：

```text
obs/history/Register/dataset-defined current action condition/text
  -> future visual consequence
```

Stage One 不是 policy 训练，也不把视频 pseudo action 当作有导航语义的 action
supervision。视频数据通常没有 instruction，很多数据也没有真实 embodied action；
因此 Stage One 的 action 只承担两个弱角色：

```text
1. history action / motion token:
  按上表确定：有 pose/action 的数据生成 previous motion/action；
  以独立 `A_hist` tokens 参与 RegisterCell 更新；
  禁止 action bias / latent bias 注入。

2. current action condition for video generation:
  按上表确定：有 pose/action 的数据生成 current action condition；没有则用 EMPTY。

3. noised action latent / action output:
   Stage One 只保留 token slot 和 forward 格式，不作为主要训练目标；
   默认不计算 policy/action loss。
```

真正有 policy 意义的 action output 在 Stage Three 用 VLN/R2R/RxR action label 训练。

### 使用数据

推荐第一版数据：

| 数据 | 用途 | 备注 |
| --- | --- | --- |
| DL3DV | 高质量 pose、长 episode、主几何视频数据 | 权重大于其原始条数占比 |
| SpatialVID | 大规模短中视频、已有 action 标注 | 需要控制占比，避免分布压倒其他数据 |
| RE10K | 室内/房产相机运动、pose 可用 | 规模小但与 VGGT/3D 表征相关 |
| Argoverse2 | 自驾视角、metric ego motion | 当前条数少，作为 domain diversity |

Kinetics 暂不进入 Stage One 主训练；除非后续 VGGT/action 置信度通过审计。

### 样本组织

每个样本：

```text
history:
  若干过去 video chunks
  dataset-defined A_hist tokens
  从 fixed R_null 开始，用同一个 RegisterCell teacher-forcing 在线 roll Register

current:
  local/current video chunk -> local memory

future:
  下一个 T4 micro chunk -> Z_future target

action:
  dataset-defined current action condition for generation
  A_noise token placeholder, no policy loss by default
```

当前 T4-IW 对齐主线使用 window-level temporal samples：

```text
T4 micro chunk:
  latent T=4, RGB 13 frames, stride 12 frames

history_iw_chunks:
  IW-1  -> 7   T4 micro history updates
  IW-4  -> 27  T4 micro history updates
  IW-8  -> 54  T4 micro history updates
  IW-16 -> 107 T4 micro history updates

target:
  next T4 micro chunk
```

训练样本单位不是 episode，而是 temporal window：

```text
(episode_latent_path, history_iw, start_micro)
```

这使长视频数据可展开为大量 windows；Stage One 的“数据量”按 window-level 统计，
不按 episode 条数统计。

### Loss

Stage One loss：

```text
L_stage1 =
  λ_v      L_visual_flow
+ λ_fmt    L_action_format
+ λ_reg    L_register_consistency
+ λ_aux    L_dataset_balance
```

建议第一版：

```text
L_visual_flow:
  只监督 Z_future 的 velocity/noise，不监督 clean Z_obs。

L_action_format:
  当前主实验固定 loss 权重为 0。该项只用于检查 action token slot / mask / head shape 是否能跑通；
  不把视频 pseudo action 当作真正 policy supervision。

L_register_consistency:
  第一版固定为 0；若后续打开需新建 ablation，不混入当前主实验。
```

推荐初始权重：

```text
λ_v = 1.0
λ_fmt = 0.0 initially
λ_reg = 0.0
```

只有在需要做接口 smoke 时，才把 `λ_fmt` 临时设为极小值，例如 `0.01`；
正式 Stage One 不以 action loss 作为优化目标。

### 数据配比

不要按 episode 条数自然采样。Stage One 应按 window-level 和 history bucket
统计采样权重；SpatialVID episode 多但短，DL3DV episode 少但长，window-level
会自然改善比例。

```text
Short history / IW-1, IW-4:
  SpatialVID 可以占较高比例，但需要 cap。

Long history / IW-8, IW-16:
  DL3DV 应主导，因为它提供长视频和高质量 pose。
```

第一版以实际 window count 为基础，再做 dataset-aware cap；不再预设固定
`35/40/15/10` 条数比例。

## Stage Two：3D-supervised generation training

### 目标

Stage Two 不改变 Stage One 的基本前向，只是在同一个 video generation forward
上额外增加 3D supervision 和 probe：

```text
same V1 generation-style forward
  + Stage One visual flow loss
  + selected hidden/Register layer
  + 3D heads/probes
```

核心不是另训一个 3D backbone，而是在不破坏 Stage One generation/Register memory
的前提下，确认 DiT/Register 的哪一层有可读 3D 表征。

### 使用数据

Stage Two 优先使用几何可信数据：

| 数据 | 用途 | 建议权重 |
| --- | --- | ---: |
| DL3DV | 主 3D 数据，pose/intrinsic 质量最好 | 50% |
| RE10K | 与 VGGT/房产相机运动相关，适合相机几何 | 20% |
| SpatialVID | 大规模补充分布，但稀疏 pose/pseudo 置信度需加权 | 20% |
| Argoverse2 | metric ego-motion 与户外/自驾几何 | 10% |

Kinetics 只在 VGGT confidence 分布通过 RE10K 对照后，作为弱 3D pseudo label
扩展；第一版不建议混入 Stage Two 主训练。

### 3D label 组织

每个 Stage Two sample 附加：

```text
camera intrinsics
relative pose between selected frames
depth or point map pseudo label
confidence mask
scale type: metric / normalized / unknown
```

3D 监督不作为 condition，只作为 loss target。

### Probe 层

第一轮 probe：

```text
25% layer
50% layer
75% layer
final layer
Register-after-DiT
```

选层依据：

```text
3D probe loss
generation loss 是否退化
Stage Three frozen-probe policy warmup 效果
```

### Loss

Stage Two loss：

```text
L_stage2 =
  L_stage1
+ λ_3d L_3d
```

其中：

```text
L_3d =
  λ_depth L_depth
+ λ_pose  L_relative_pose
+ λ_point L_point_or_correspondence
+ λ_conf  L_confidence_weighting
```

建议第一版：

```text
λ_v = 1.0
λ_fmt = 0.0
λ_3d = 0.2 initially

L_depth:
  scale-invariant log depth / masked L1

L_relative_pose:
  translation direction + rotation geodesic

L_point_or_correspondence:
  只在 pseudo label 置信度足够时启用
```

如果 3D loss 压坏 generation loss，先降低 `λ_3d` 到 `0.05`，不要马上删掉 3D
branch。

## Stage Three：VLN policy training

### 目标

Stage Three 把 Stage Two 选出的 shared backbone/Register 表征用于导航：

```text
obs/history/Register/instruction
  -> action/nav tokens
  -> action velocity/noise
  -> denoise/decode
  -> control
```

推理时删除 generation branch：

```text
T_future = 0
```

训练时不应完全丢弃 Stage One/Two 的数据和 loss。Stage Three 是 policy 主训练，
但需要 replay/cotrain video generation 与 3D supervision，避免 shared
backbone/Register 在 VLN imitation 中遗忘长历史视频记忆和空间表征。推理仍然
policy-only，generation/3D heads 不执行。

### 使用数据

Policy 主数据：

| 数据 | 用途 | 备注 |
| --- | --- | --- |
| R2R-CE | 标准 VLN-CE 主任务 | train/val_seen/val_unseen 结构清楚 |
| RxR-CE | 多语言/更长路径/更大规模 | guide/follower 可分阶段引入 |
| ScaleVLN | 大规模增强，适合 policy imitation | 注意与 R2R 分布关系 |
| LHPR-VLN | 长程/任务型导航补充 | batch/场景准备完成后逐步引入 |
| MP3D | R2R/RxR 共用场景 | 作为渲染/闭环环境，不是独立 instruction 数据 |

Replay/cotrain 数据：

| 数据 | replay 用途 | 备注 |
| --- | --- | --- |
| Stage One window pool | `L_visual_flow_replay`，保持 video consequence generation 与 Register memory | DL3DV、SpatialVID 20%、RE10K、Argoverse2 |
| Stage Two 3D pool | `L_3D_replay`，保持 spatial hidden/Register 表征 | DL3DV、RE10K 主导，SpatialVID/Argoverse2 低权重 |

### VLN 样本组织

每个 sample：

```text
instruction
current observation obs_s
history prefix obs/actions up to s
target action chunk a_s...a_{s+H_nav-1}, H_nav=10
future obs: 不构建
pose/progress/distance-to-goal: 不作为 policy 输入
```

默认：

```text
H_nav = 10
loss 监督 10 个 action
closed-loop evaluation 先执行第 1 个 action
```

短轨迹不足 10 步：

```text
用 STOP padding
mask 掉 episode 结束后的无效 step
STOP loss 单独加权
```

### Loss

Stage Three 主 loss：

```text
L_stage3 =
  λ_nav L_action_flow(A_noise -> A_out, gt_actions)
+ λ_stop L_stop_balance
+ λ_ce   L_discrete_ce_aux
+ λ_replay_video L_visual_flow_replay
+ λ_replay_3d   L_3D_replay
```

第一版建议从保守 replay 开始，具体配比需要实验：

```text
λ_nav = 1.0
λ_stop = 0.2
λ_ce = 0.1
λ_replay_video = 0.05
λ_replay_3d = 0.05

batch/task mix initial:
  VLN policy batch       70%
  Stage One video replay 20%
  Stage Two 3D replay    10%
```

待实验的配比：

| 配置 | VLN | Video replay | 3D replay | 用途 |
| --- | ---: | ---: | ---: | --- |
| S3-R0 | 100% | 0% | 0% | policy-only baseline，检查遗忘/退化 |
| S3-R1 | 80% | 15% | 5% | 轻 replay |
| S3-R2 | 70% | 20% | 10% | 默认候选 |
| S3-R3 | 60% | 25% | 15% | 强 replay，检查是否拖慢 policy 收敛 |

Replay batch 的关键约束：

```text
video/3D replay 只更新 shared backbone/Register/generation/3D heads；
policy batch 更新 shared backbone/Register/action head；
policy tokens 不能读取 replay future visual/3D；
replay loss 不把视频 pseudo action 当 policy supervision。
```

关键 mask：

```text
action/nav tokens:
  can read Register / obs / instruction / state
  cannot read future visual / future 3D
  cannot receive clean current/target action as condition

policy Register:
  cannot read future visual / future 3D

generation branch:
  can read Register / obs / action / instruction
```

### 数据配比

Stage Three 第一版推荐：

```text
R2R-CE     25%
RxR-CE     35%
ScaleVLN   30%
LHPR-VLN   10%
```

如果 LHPR batch 或 HM3D 仍未准备完全：

```text
R2R-CE     30%
RxR-CE     40%
ScaleVLN   30%
LHPR-VLN    0%
```

EnvDrop / follower / augmentation 不应一开始全量压入。建议作为 20% 以内的
augmentation pool：

```text
标准 CE episode: 80%
augmentation:    20%
```

## 三阶段训练顺序

建议执行顺序：

### Stage One-A：V1 smoke

```text
数据:
  DL3DV only

目标:
  跑通 [Register][local][Z_future] packing
  跑通 T_future=1 VAE encode/decode
  跑通 action interface placeholder，但不训练 policy/action loss

步数:
  1k
```

### Stage One-B：多数据 video generation memory pretrain

```text
数据:
  DL3DV / SpatialVID / RE10K / Argoverse2

配比:
  按 window-level + history bucket 统计后设 dataset cap

目标:
  稳定 generation branch + Register long-history update
  保留 action condition / action output token 格式，但不把视频 pseudo action 训成 policy
```

### Stage Two-A：3D probe scan

```text
数据:
  DL3DV / RE10K 主导

目标:
  冻结或半冻结 backbone，扫 probe 层
```

### Stage Two-B：3D joint finetune

```text
数据:
  DL3DV / RE10K / SpatialVID / Argoverse2

目标:
  在 Stage One video generation loss 上直接加 3D loss；
  action interface 继续保持格式兼容。
```

### Stage Three-A：policy-only imitation warmup

```text
数据:
  R2R-CE / RxR-CE / ScaleVLN

目标:
  generation branch 不执行；
  正式训练 action/nav output，让 action flow decoder 与离散 action decode 先收敛。

用途:
  作为 S3-R0 baseline，衡量 policy-only fine-tune 的遗忘/退化程度。
```

### Stage Three-B：policy + Stage One/Two replay cotrain

```text
数据:
  VLN: R2R-CE / RxR-CE / ScaleVLN / LHPR-VLN
  Stage One replay: DL3DV / SpatialVID 20% / RE10K / Argoverse2 video windows
  Stage Two replay: DL3DV / RE10K / SpatialVID / Argoverse2 3D windows

默认配比:
  VLN policy batch       70%
  Stage One video replay 20%
  Stage Two 3D replay    10%

目标:
  在提升 VLN SR/SPL 的同时，保持 Register long-history video memory 和 3D spatial
  representation，避免 Stage Three 破坏 Stage One/Two 学到的 backbone 能力。
```

## 需要优先实现的文件

```text
NAV/config/v1_stage_one_video_action.yaml
NAV/config/v1_stage_two_3d.yaml
NAV/config/v1_stage_three_vln.yaml

NAV/scripts/datasets/build_v1_video_pack_manifest.py
NAV/scripts/datasets/encode_v1_vae_packs.py
NAV/scripts/datasets/build_v1_vln_policy_manifest.py

NAV/src/nav/v1/data/video_action_dataset.py
NAV/src/nav/v1/data/vln_policy_dataset.py
NAV/src/nav/v1/losses.py
```

## 当前风险

1. V1 sparse pack latent 与 V0 81-frame latent 语义不同，不能混用缓存。
2. 如果 Register/action mask 没写对，Stage Two/Three 会产生 future leakage。
3. SpatialVID 量太大，必须 weighted sampling，否则分布会压倒 DL3DV/RE10K。
4. 视频 pseudo action 与 VLN discrete action 的尺度必须通过 displacement bucket
   对齐，不能只按 frame index 对齐。
5. Stage Three 的 generation auxiliary 只能训练时使用，不能成为 policy condition。
6. 不得把 GT pose 设计成推理输入；无 pose 数据只能走弱监督或 pseudo motion
   低置信路径。
