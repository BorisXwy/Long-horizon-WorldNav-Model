# V1 Action 与 Geometry 统一组织 Schema

| 字段 | 内容 |
| --- | --- |
| ID | NAV-DAT-008 |
| 类型 | 数据与实现 Schema |
| 状态 | Proposed / V1 Draft |
| 更新时间 | 2026-08-10 |
| 职责 | 定义 V1 中 video pseudo action、VLN action、previous executed action、3D geometry target 和模型 forward 所需样本的统一组织形式 |

## 核心结论

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
  是 action stream 要生成的变量，不是 policy condition。

previous executed action:
  是历史中已经发生的事件，可用于 Register update。

pose:
  是训练 teacher / pseudo action / 3D supervision 来源，不是推理输入。

future visual / future 3D:
  是 generation branch / 3D probe 的监督目标，不是 policy condition。
```

## ActionStep

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

### primitive_id

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

### delta_ego

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

## ActionChunk

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

## VaePack

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

## GeometryTarget

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

## V1Sample

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

## 模型 forward 接口

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

## 模型内部模块

当前推荐代码结构：

```text
RegisterCell:
  R_{-1}=R_null fixed template
  R_i = RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))

VisualStem:
  R_t, Z_obs, Z_future_noisy -> visual/main tokens

ActionStem:
  A_hist / A_cur / A_noise -> independent action tokens

TextConditioner:
  instruction/text -> condition tokens

DualStream Shared Backbone:
  visual stream + action stream + condition
  with explicit attention mask

GenerationHead:
  future tokens -> predicted noise/velocity(Z_future)

ActionFlowDecoder:
  action stream tokens -> action velocity / primitive logits

GeometryProbe:
  selected hidden/Register -> depth/pose/point/correspondence prediction
```

当前实现策略：

```text
v1:
  unified RegisterCell
  dual-stream / MoT-style visual stream + action expert stream
  explicit policy-safe attention relation
  no action/latent additive bias
```

## Loss 接口

### Stage One

```text
L_stage1 =
  L_visual_flow(Z_future)

Stage One 保留 A_noise / A_out 格式，但不计算 action supervision。
```

### Stage Two

```text
L_stage2 =
  L_stage1
+ λ_3d L_geometry
```

### Stage Three

```text
L_stage3 =
  L_action_flow
+ λ_ce CE(action primitive)
+ λ_replay_video L_visual_flow_replay
+ λ_replay_3d L_3D_replay
```

## 落盘组织

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

## 当前实现状态（2026-08-10）

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

## 代码组织建议

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
