# 流式 3D 重建与 3D 导航官方复现

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-011` |
| 类型 | 官方复现与结构分析 |
| 状态 | Completed / Reference |
| 更新时间 | 2026-08-29 |
| 职责 | 记录 LingBot-Map、ABot-Recon 与 LoGoPlanner 的完整权重复现、核心表征、训练监督、流式性质与样本构造。 |

## 复现目的与验收边界

本轮在 GPU 1 上运行三套官方完整模型与官方发布权重，目的不是用缩小网络做
smoke test，而是回答三个与 NAV Register 设计直接相关的问题：

1. LingBot-Map 与 ABot-Recon 的“流式重建”究竟保存什么状态；
2. 它们把什么 hidden representation 变成 camera/depth/point-map，训练时如何施加
   reconstruction supervision；
3. LoGoPlanner 如何把 3D backbone 的 hidden feature 接到 navigation policy，它是否
   真正递归流式，以及一个训练样本需要包含什么。

本轮仅缩短输入序列长度或选用本地真实 RGB-D 样本，没有缩减层数、宽度、head、
采样步数或替换官方前向。LoGoPlanner 当前开源仓库不包含官方训练 dataloader，
因此“精确训练样本字段”只能由论文监督目标与官方 inference API 共同反推，本文会
明确标记，不把推断写成已公开事实。

## 复现总览

| 模型 | 是否真正逐帧流式 | 在线历史状态 | 官方权重完整前向 | GPU 1 实测 |
| --- | --- | --- | --- | --- |
| LingBot-Map | 是 | 分页 KV cache：anchor dense tokens、local dense window、每个旧帧 6 个 trajectory tokens | 成功，24 帧真实图像 | 两次 6.44--7.85 s / 24 帧，3.06--3.73 FPS，峰值 5.64 GiB；本地使用 SDPA fallback |
| ABot-Recon | 是 | 固定 12 帧局部 KV；无持久 learned global memory | 成功，24 帧真实图像 | 完整进程 67.16 s（含加载、CPU 预处理与落盘），峰值由官方 demo 路径未单独记录 |
| LoGoPlanner | agent API 在线，但 3D backbone 非递归流式 | Python 保存全部 RGB-D；每步从全历史均匀重采样 12 帧，另取最近 8 帧，然后从头重算 | 成功，12 帧真实 RGB-D、16 candidates、10 DDPM steps | 两次 policy forward 0.99--1.16 s，峰值 5.00 GiB |

注意：LingBot-Map 论文约 20 FPS 的结果使用 FlashInfer paged KV kernel、
518×378、1000 帧及 64 帧 local window；本机为了先保证依赖稳定，复现采用官方
SDPA fallback、518×294 crop，并且 24 帧统计包含最初 8 帧 scale initialization，
不可直接把 3.06 FPS 当作论文速度复现失败。

## LingBot-Map：稀疏化长历史，但不是固定大小 Register

### 流式计算与核心表征

每个输入帧先经 DINOv2 ViT-L/14 得到 dense image patch tokens，再附加：

```text
1 camera token + 4 register tokens + 1 anchor/scale token = 6 special tokens/frame
```

随后通过 24 组交替的 Frame Attention 与 Geometric Context Attention（GCA）。
GCA 把历史拆成三类：

```text
当前帧 query
  ├─ attend anchor context：初始化帧的完整 dense tokens，固定坐标与尺度
  ├─ attend local pose-reference window：最近 k 帧的完整 dense tokens，提供视差和匹配
  └─ attend trajectory memory：更旧帧只保留每帧 6 个 special-token K/V
```

因此它不是 NAV 设想的单个固定预算递归 Register。它保留的是“每帧 6 个 compact
trajectory tokens”，历史增长从约 `M+6 tokens/frame` 降为 `6 tokens/frame`；anchor
与 local window 的 dense token 数固定，但 trajectory memory 仍随时间线性增长。
官方 FlashInfer 实现用 paged KV cache 追加新 token、回收越出窗口的 dense patch
pages，避免每帧重算全部历史和连续 KV 内存搬移。

### 如何从 representation 得到 reconstruction

核心 3D 能力仍在 transformer hidden，而不是后处理 point cloud：

- camera head 从每帧 camera token 读取 absolute camera pose；
- DPT-style depth head 从 dense image tokens 读取 per-pixel depth 与 confidence；
- inference 再用 depth、intrinsics、extrinsics 做确定性 unprojection，得到 world points。

也就是说，4 个 register tokens 主要参与跨帧几何上下文压缩，camera token 是姿态读出
槽位，dense patch hidden 是深度/稠密几何读出槽位。只读取 register 做全部 3D 任务
并不是该模型的做法。

### reconstruction task 如何训练

总损失为：

```text
L = lambda_depth * L_depth
  + lambda_abs_pose * L_abs_pose
  + lambda_rel_pose * L_rel_pose
```

- `L_depth`：带预测 uncertainty/confidence 的 depth regression，加 depth gradient
  consistency；
- `L_abs_pose`：camera-to-world absolute pose regression；
- `L_rel_pose`：local window 内所有 frame pair 的 rotation geodesic loss 与 translation
  L1，用因果可见的局部帧直接约束跨帧几何一致性。

训练先做 offline base：2--24 views、global cross-frame attention、160k iterations；
再从 base Q/K/V 初始化 streaming GCA，训练视图数从 24 线性增长到 320，local window
在 16--64 随机采样，再训练 160k iterations。长视频阶段使用 foldback sampler：随机
起点与 stride，触边后反向并更换 stride，从而生成长、连续但帧率多样的序列。
上述训练方案来自论文；当前官方仓库只发布模型、推理和 benchmark 入口，没有
`train.py`、optimizer config 或 dataset loader，因此本轮没有伪造一套“官方训练复现”。

### 本地结果

```text
result/streaming_3d_reproduction/lingbot_map/
  official_university_24f_sdpa_20260829/
    metrics.json
    predictions.pt
```

`predictions.pt` 中 24 帧的 `pose_enc [24,9]`、`depth [24,294,518,1]`、
`depth_conf`、`extrinsic [24,3,4]`、`intrinsic [24,3,3]` 全部 finite。模型参数量
为 1,157,943,540；推理结束时 SDPA KV 中有 24 个 frame blocks，约 1.72 GiB。

## ABot-Recon：固定局部记忆 + 显式位姿链式合成

### 流式计算与核心表征

ABot-Recon 的设计比 LingBot-Map 更严格地 bounded：当前帧只访问前 11 帧与自己，
local context 固定为 12，旧 K/V 直接丢弃；released checkpoint 默认没有 reference
frames、summary tokens 或持久 learned global memory。

每帧经 DINOv2 ViT-L/14 与 Pi3 风格 alternating intra-frame/inter-frame decoder：

```text
504×280 RGB
  -> 720 patch tokens + 5 Pi3 special/register/pose tokens
  -> fixed 12-frame causal local attention
  ├─ dense patch hidden -> current-camera local point map P_i + confidence
  └─ special/camera hidden -> adjacent relative transform T_(i-1 <- i)
```

它真正学习的核心表征是“当前相机坐标系 local point map + 相邻帧 relative pose”，而
不是网络内部的全局地图。全局 trajectory 与 world point cloud 在模型外通过连续
矩阵乘法合成：`T_(0<-i)=T_(0<-i-1) @ T_(i-1<-i)`，再把 local points 变换到 world。
可选 loop closure 也是 inference 后端，不是流式网络的持久神经记忆。

rotation refiner 会把 compact camera/motion token 与 dense visual token 交互，并用
causal temporal convolution 在局部窗口内细化相邻旋转；这解释了为何固定局部窗口
仍可通过较稳定的相对位姿链工作。

### reconstruction task 如何训练

论文技术报告给出的主要监督包括：

```text
L = lambda_pose * L_pose
  + lambda_smooth * L_smooth
  + lambda_pts * L_points
  + lambda_normal * L_normal
  + lambda_conf * L_confidence
```

- 对窗口内可组成的 frame pairs，把相邻 pose 连乘得到更长 gap 的 relative pose，
  用 composition-aware translation/rotation loss 监督，且长 gap 权重更高；
- local point map、surface normal 和 confidence 直接从 dense patch path 监督；
- residual smoothness 约束局部相对运动变化。

训练从 Pi3 初始化：Stage I 使用 32-frame clips、固定 window=12，学习相邻 pose 与
local geometry；Stage II 使用 128-frame clips 但网络窗口仍为 12，重点让短边 pose
在长 composition chains 上保持稳定；最后仅训练 confidence。多视图集合先由 pose
graph 排成 pseudo-sequence，原生视频则按时间间隔采样有序 clips。

截至 2026-08-29，官方仓库明确标注 training code/recipes 尚未发布，因此本轮能
完整复现 inference/evaluation output，但不能在不自行猜测实现的前提下运行官方
reconstruction training。

### 本地结果

```text
result/streaming_3d_reproduction/abot_recon/
  official_university_24f_sdpa_20260829/
    camera_poses.npy       # [24,4,4]
    relative_poses.npy     # [23,4,4]
    local_points.pt        # [24,280,504,3]
    confidence.pt          # [24,280,504]
    reconstruction.ply
    trajectory_bev.png
```

所有输出均 finite；模型参数量 1,000,934,176。`reconstruction.ply` 和
`trajectory_bev.png` 使用官方 exporter 从 local points 与 composed poses 生成，
不是额外训练的第三方重建器。

## LoGoPlanner：3D hidden 作为 policy condition，而非显式地图输入

### 3D feature 到 navigation policy 的完整数据流

官方权重对应的精确 inference token 流如下：

```text
全历史 RGB-D Python queue
  ├─ uniform sample 12 context frames
  │    RGB: DINO/Pi3 encoder -> 264 patch tokens/frame, dim 1024
  │    depth: DepthAnything-V2-S -> 264 tokens/frame, dim 384
  │    concat + linear fusion -> Pi3 alternating decoder hidden [B*T,269,2048]
  │       ├─ camera decoder -> pose-specific hidden, dim 512
  │       │    -> 16 state queries/frame -> compress -> Q_S [B,12,384]
  │       └─ point/world-point decoder -> geometry-specific hidden, dim 1024
  │            -> compress -> Q_G [B,12,384]
  │
  └─ latest 8 RGB frames + latest depth
       -> two DepthAnything-V2-S encoders + query compressor
       -> Q_RGBD [B,8,384]

point goal [B,3] -> state/goal encoder -> Q_goal [B,1,384]

policy condition = [time(1), goal repeated(3), Q_RGBD(8), Q_S(12), Q_G(12)]
                 = [B,36,384]

Gaussian action noise [B,24,3]
  -> linear [B,24,384]
  -> causal TransformerDecoder cross-attends 36 condition tokens
  -> epsilon [B,24,3]
  -> 10 DDPM denoising steps, 16 candidates
  -> critic ranking -> top trajectory
```

论文强调不把预测 point cloud、camera extrinsics 或 chassis pose 数值显式送入 planner。
policy 读取的是被 3D auxiliary tasks 塑形后的 `Q_S/Q_G` hidden features，因此避免
上游 3D 数值误差直接级联到规划。点云与位姿 head 的作用是给共享 hidden 强约束；
真正的导航信息接口是 task-specific query 压缩后的 token。

### 它是否流式

要区分两个层次：

- **系统/agent 层**：是在线闭环，每来一帧就加入 queue 并重新规划；
- **3D backbone 层**：不是 LingBot-Map/ABot-Recon 意义上的 streaming。官方
  `policy_agent.py` 保存完整历史，每步用 `get_indices(0,current,12)` 在整个历史
  区间均匀取 12 帧，再从头运行一次 Pi3；没有 recurrent state、KV cache 或增量
  register update。最近 8 帧由另一路固定窗口 RGB-D encoder 编码。

所以它的单步输入计算量有上界，但原始历史 storage 随 episode 增长，并且每步会
重复编码抽到的历史帧。这正是 NAV Register 可以替换/改进的接口：保留 LoGoPlanner
“3D hidden -> query -> policy”的做法，同时把重算的 12 帧 history 换成递归更新状态。

### 训练样本如何构造

论文公开的事实：训练数据由 simulator 生成，超过 200k trajectories、约 10M RGB-D
images；机器人为 differential-drive cylinder，camera height 在 0.25--1.25 m 随机，
pitch 在 0--30° 随机；随机 start/goal 后用 A*、greedy refinement、cubic spline
得到 collision-free path。Stage I 用 batch size 12 训练 24 h，微调 geometry decoder
和 task heads，depth 提供 metric scale prior，监督 scene point cloud 与 camera
extrinsics；Stage II 冻结 backbone decoder，用 batch size 32 训练 diffusion policy
和 task heads 3 天。

官方没有发布训练 dataset/sampler/optimizer 入口。由论文监督目标和 inference API
能确定一个 policy sample 至少需要以下字段，但这部分属于代码级反推：

| 字段 | shape/语义 | 用途 |
| --- | --- | --- |
| `context_rgbd` | 12 个覆盖历史区间的 RGB-D views | Pi3 metric geometry、state/scene tokens |
| `memory_rgbd` | 最近 8 个 RGB views + latest depth | local obstacle / appearance memory |
| `start_goal` | `[3]`，目标在起点坐标系的位置/朝向 | goal condition 与 sub-goal supervision |
| `future_action` | `[24,3]`，每步 `(dx,dy,dtheta)` | diffusion epsilon/noise-prediction target |
| camera/chassis poses | per context frame | camera extrinsic、odometry/state auxiliary losses |
| local/world points | metric per-pixel point maps | geometry auxiliary losses与 scale grounding |
| trajectory value/feasibility | candidate-level | critic/ranking supervision，具体构造未公开 |

因此，LoGoPlanner 不是语言指令 VLN sample；它是 point-goal + historical RGB-D +
future local trajectory 的监督样本。把它迁移到 NAV/R2R 时，还需要把 instruction token、
离散 action label 与 STOP semantics 接到 policy condition/head，不能直接照搬其
`24×3` continuous diffusion target。

### 本地结果与限制

```text
result/streaming_3d_reproduction/logoplanner/
  official_nyuv2_rgbd_12f_torch251_20260829/
    metrics.json
    policy_outputs.npz
```

官方 checkpoint 严格兼容加载，`missing=[]`、`unexpected=[]`；总参数
1,164,614,397。输出包含 `all_trajectories [1,16,24,3]`、`critic [1,16]`、
top/bottom trajectories 与 `sub_pointgoal [1,3]`，全部 finite。

本次输入取本地 NYUDv2 的 12 对真实 RGB-D，仅用于验证官方完整网络与 checkpoint
链路；这些文件不保证来自同一连续 trajectory，因此不能把预测轨迹质量当作论文
benchmark 复现。正式导航指标仍需官方 InternScenes/NavDP simulator 与 episode。

## 三者对 NAV 的直接结论

1. **流式 reconstruction 不只有一种 memory**：LingBot-Map 是“dense local + 稀疏
   per-frame global KV”，ABot-Recon 是“固定 local window + 显式 pose composition”。
   前者保留 learned long-range context，后者用更强的局部相对几何把全局状态移到网络外。
2. **3D readout 不能只盯 Register**：两个重建模型都从 dense visual hidden 读取
   depth/point map，从 special/camera hidden 读取 pose。Register 更适合作为跨帧上下文，
   不是唯一几何载体。
3. **训练长历史不必让 reconstruction target 也变长**：LingBot-Map 的 local relative
   pose loss、ABot-Recon 的 composition-aware pose chain，都说明可以监督局部预测，
   再用更长 sequence 验证/约束累积稳定性。
4. **LoGoPlanner 最值得复用的是 feature-level bridge**：3D auxiliary heads 塑形共享
   hidden，policy 使用 query-compressed geometry tokens，而不是显式 point cloud；这与
   NAV Stage Two probe -> Stage Three policy 的路线一致。
5. **NAV 的潜在差异点明确**：LoGoPlanner 目前每步重算 12 帧；若 NAV Register 能在
   不重算历史的情况下提供同等级的 state/scene query，并保持生成支路，则可形成可测的
   streaming efficiency 与 long-history contribution。

## 代码、环境与日志入口

```text
仓库：
  ../lingbot-map
  ../ABot-Recon
  ../NavDP/baselines/logoplanner

环境：
  ../virtual_env/.venv_lingbot_map
  ../virtual_env/.venv_abot_recon
  ../virtual_env/.venv_navdp

启动脚本：
  NAV/scripts/reproduction/run_lingbot_map_gpu1.sh
  NAV/scripts/reproduction/run_abot_recon_gpu1.sh
  NAV/scripts/reproduction/run_logoplanner_gpu1.sh

日志：
  NAV/log/reproduction/streaming_3d_20260829/
```

NavDP 官方 requirements 固定的 PyTorch 2.2.2 与其当前精确 Pi3 submodule 不兼容：
Pi3 需要 `torch.nn.attention.SDPBackend`，其自身 requirements 指向 PyTorch 2.5.1。
本地 `.venv_navdp` 因此使用 `torch==2.5.1+cu124`、`torchvision==0.20.1`、
`numpy==1.26.4`、`plyfile==1.0.3`，`pip check` 已通过。此变更只修复官方仓库内部
dependency drift，没有修改模型结构或 checkpoint。

官方资料：

- LingBot-Map paper：<https://arxiv.org/abs/2604.14141>
- LingBot-Map code：<https://github.com/Robbyant/lingbot-map>
- ABot-Recon project：<https://amap-cvlab.github.io/ABot-Recon-html/>
- ABot-Recon code / technical report：<https://github.com/amap-cvlab/ABot-Recon>
- LoGoPlanner paper：<https://arxiv.org/abs/2512.19629>
- LoGoPlanner code：<https://github.com/InternRobotics/NavDP/tree/master/baselines/logoplanner>
