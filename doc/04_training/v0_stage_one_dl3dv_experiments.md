# V0 Stage One 1.0：DL3DV 从头训练实验

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-005` |
| 类型 | 训练实验规范（Training Experiment Specification） |
| 状态 | A stopped @541 (ckpt@500) / B done @1000 / 验证边界已界定 |
| 更新时间 | 2026-08-07 |
| 职责 | 固定 Stage One 1.0 的初始化、数据、采样、显存实测和运行入口 |

## 实验定义

Stage One 1.0 分别训练 A/B（DEC-019，2026-08-03 更新）：

- Backbone 从原始 Infinite-World checkpoint 初始化；
- 不加载既有 RE10K A/B checkpoint，也不加载旧 A/B-10000 checkpoint；
- HPMC history 移除，Local Memory、Action/Text和RFlow保持不变；
- 新的 Register Extractor/Updater 随机初始化；
- Wan DiT、Extractor、Updater 全参数训练；
- 数据只使用 DL3DV full-episode latent 与 dense Action；
- 文本使用 empty UMT5 embedding；
- 训练 **1000 effective optimizer steps**，effective batch size 16；
- 每 **100 effective steps** 保存一次完整 checkpoint；
- 不使用 plateau early stop。

DEC-018 的 10000-step 计划已被 DEC-019 取代；旧 `stage-one-v1-dl3dv-{a,b}-10000`
run 作为历史保留，不用于正式续训。

## A 停止与续训记录（2026-08-04）

- 2026-08-04 18:59，A（`latent_prefix`，cuda:0）按用户指示优雅停止
  （SIGTERM，进程 4023980 已干净退出）。
- 停止时 A 已训练到 **step 541**，最后保存的 checkpoint 为
  `full-step-000500.pt`；step 501–541 的进度已写入 `train.log` 但未落盘
  到 checkpoint。
- B（`dit_condition`，cuda:1）未受影响，继续运行。
- **后续 A 从 step 500 续训**：使用 `full-step-000500.pt` 作为
  `resume_checkpoint`，run 名沿用
  `stage-one-v1-from-scratch-a-ebs16-1000`，其余参数不变（micro=1、accum=16、
  effective=16、total 1000 effective steps、save-every 100、DL3DV full-episode
  latent、empty UMT5）。续训命令示例：

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  latent_prefix 0 1 1000 stage-one-v1-from-scratch-a-ebs16-1000 100
```

并在对应配置中设置 `resume_checkpoint=
log/stage-one-v1-from-scratch-a-ebs16-1000/full-step-000500.pt`，
`register_checkpoint` 同源（若脚本区分）。续训前确认 GPU0 显存可用且
`full_episodes_v1` latent prep 是否仍在 GPU0 占用显存。

## Register 语义

首个 History Chunk：

\[
R_1=\operatorname{Extractor}(Z_0)
\]

后续 History Chunk：

\[
R_{i+1}=\operatorname{Updater}(R_i,Z_i)
\]

A 的 Extractor 将首 Chunk 时间池化为4个 spatial planes，再通过独立
Cross-Attention/FFN 提取 `[B,16,4,H,W]` Register。

B 的 Extractor 将首 Chunk 池化为 `4×2×2=16` 个观测 token，再通过独立
Cross-Attention/FFN 提取 `[B,16,256]` Register。

两者均不包含 `initial_registers` 参数；Extractor 与 Updater 不共享参数。

## 训练样本

141条 DL3DV full episode 全部至少包含4个 chunks，均可参与训练。

每个 optimizer step 从以下 history 长度中按打乱循环等比例选择：

```text
1 history: C0       → C1
2 history: C0,C1    → C2
3 history: C0,C1,C2 → C3
```

- episode 顺序每个 epoch shuffle；
- 每条 episode 内随机选择连续窗口起点；
- 只对最后一个 target 计算 Diffusion/RFlow loss；
- Register 在 forward 内在线 Extract/Update，不落盘；
- physical batch 内使用相同 history length，避免不规则张量 padding。

## 显存实测

设备为单张约47.40 GiB GPU，bf16、全参数、gradient checkpointing。

DEC-019（ebs16）实测：

| Variant | Micro batch | Accumulation | Effective batch | NAV 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| A latent_prefix | 1 | 16 | 16 | 33.24 GiB |
| B dit_condition | 2 | 8 | 16 | 28.71 GiB |

DEC-016/017（旧 ebs4/bs1 probe）实测保留供追溯：

| Variant | Physical batch | 结果 | 显存 |
| --- | ---: | --- | ---: |
| A | 4 | OOM during backward | 46.82 GiB已占用，仍需772 MiB |
| A | 1 | Pass | 29.21 GiB allocated peak |
| B | 1 | Pass | 16.48 GiB allocated peak |

DEC-019 调度采用：

- GPU 0：A，micro=1、accum=16、effective=16，1000 effective steps；
- GPU 1：B，micro=2、accum=8、effective=16，1000 effective steps；
- 两卡其余显存继续并行准备 `full_episodes_v1` latent（7-shard 幂等续传）。

2026-07-30启动时GPU 1被其他账号的 `vlm_server.py` 和 `depth_server.py` 占用
约41 GiB，不能擅自终止。该情况在 DEC-019 重启时已重新评估；当前 B 的
ebs16 配置在 GPU1 上稳定运行。

## 配置与入口

## 项目内 `from scratch` 的统一定义

本项目后续实验所称的 `from scratch` 并非随机初始化整个视频生成模型，而是：

- 仅加载原始 Infinite-World checkpoint 中的 Wan/DiT 预训练权重；
- 加载后按方案 A 或 B 改造网络结构；
- Extractor、Updater/Register 等新增模块随机初始化；
- 不加载任何旧 RE10K、旧 A/B 或其他 NAV 训练 checkpoint；
- `resume_checkpoint=null` 且 `register_checkpoint=null`。

因此，`from scratch A/B` 表示两种结构均从同一个原始 Infinite-World 基线独立
开始训练，而不是 A/B 互相继承，也不是续训此前实验。

配置：

```text
NAV/config/train_stage_one_v1_dl3dv.yaml
```

统一入口：

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  <latent_prefix|dit_condition> <gpu> <physical_batch> <steps> <run_name> <save_every>
```

DEC-019 正式单任务示例（effective batch 16、1000 effective steps、每 100
effective steps 存 checkpoint）：

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  latent_prefix 0 1 1000 stage-one-v1-from-scratch-a-ebs16-1000 100
```

```bash
bash NAV/scripts/run_stage_one_v1_dl3dv.sh \
  dit_condition 1 2 1000 stage-one-v1-from-scratch-b-ebs16-1000 100
```

注意：`physical_batch` 与 `gradient_accumulation_steps` 由配置中的
`effective_batch_size=16` 反推（A: micro=1 → accum=16；B: micro=2 → accum=8）。

入口自动激活项目 Python 环境，并启用
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

数据余量调度入口：

```text
NAV/scripts/run_latent_shard_queue.sh
```

当前 tmux：

```text
nav_stage1_v1_a
nav_stage1_v1_b
nav_full_episode_prep
nav_stage1_v1_tensorboard
```

正式 A/B ebs16 的独立 TensorBoard 运行在远程端口 6011，不包含旧 1000-step、
10000-step、smoke 或其他实验。

## Stage One 验证边界与后续待补实验（2026-08-07）

本节界定 Stage One 1.0 已验证什么、未验证什么，并给出后续待补实验清单。
对应 `doc/05_evaluation/v0_vbench_infinite_world_protocol.md` 记录 4/7/8 与
`doc/08_insight/insight_log.md` R13–R16。

### 已验证（Stage One 初步验证成立）

| 项 | 证据 | 强度 |
| --- | --- | --- |
| 训练能简单收敛 | A 训到 step 541（ckpt@500）、B 训到 step 1000，loss 下降无 NaN | 强 |
| Register 在**训练集**上表现正常 | gthist 用 DL3DV **训练 episode** 的 3 GT 历史 chunk 预测第 4 chunk，A@500/B@1000 单步保真 L1=0.0874/0.0917、SSIM=0.54，优于 InfiniteWorld（L1=0.1152）（记录 8） | 中（仅 1 episode、单 seed、空 text） |
| Register 是固定预算、比 HPMC 小 | A=4 帧 latent（~8MB）/ B=16 token（~128KB）vs HPMC 20 帧（~40MB），token 比 21:5:1 | 强（结构事实，非性能） |
| 短程 autoreg 技术质量 | 3-chunk autoreg VBench：A@500 MS/AQ/IQ 正常、DD=true（记录 7） | 中（仅 1 demo 场景） |

### 未验证（Stage One 尚未证明）

1. **泛化 + action 跟随**：gthist 用训练 episode + GT pose-derived action；autoreg 用
   demo 单场景 + 手录 0001.json。无 held-out 场景、无跨分布 action 跟随证据。
2. **长程有效显存缩减**：Register 尺寸更小已证，但"长程下保持性能"未证——NAV A/B
   的 16+ chunk autoreg 从未跑过（仅 InfiniteWorld 跑过 16-chunk）。
3. **autoreg 等价于 teacher-forced**：gthist 单步保真好，但 autoreg 3-chunk 里
   B 的 DD 塌缩为 false（记录 7）——autoreg ≠ teacher-forced，且无 NAV 长 autoreg 证据。
4. **Register 优于 HPMC**：3-chunk 下 HPMC 的 `T_in=63≤80` 不触发二次压缩，
   与"全量历史"等价，Register 与 HPMC 容量上等价、比不出设计差异（R13）。
   现有 3-chunk 对比对"Register 优于 HPMC"零证据。
5. **text 通道可用**：Stage One 用空 text，UMT5 cross-attention 未训练；
   推理却给真实 prompt。R2R 指令接入路径未定（R15）。
6. **train-test action 一致**：训练 pose-derived 离散 vs 推理手录，分布不一致（R16）。

### 后续待补实验（按性价比排）

1. **多场景多 seed autoreg 3-chunk**（复用 stats20 框架扩到 NAV A/B）——证伪/确认
   B 的 DD 塌缩是否系统性、A 是否稳定。
2. **NAV A/B 16-chunk autoreg**（用已改好的 cached-latent 脚本）——证明长程下
   固定预算 Register 不崩，是"有效显存缩减"的核心证据。
3. **长程训练**（history 课程扩到 16+ chunk）——当前 updater 仅训过 ≤3 次更新，
   长程下 OOD；要让 16-chunk autoreg 对 NAV 公平，必须先长程训练 updater。
   这是 R13"递归更新 vs 启发式窗口"主张的唯一证据来源。
4. **held-out 场景 + 训练分布 action 的 gthist**——证明泛化 + action 跟随
   （DL3DV 未训练 episode，action 仍用 pose-derived，隔离 text）。
5. **text 通道决策实验**——决定 Stage One/二是否回头引入 caption 联合训练
   （R15），影响 Stage Three 能否讲 R2R 故事。
6. **action 来源统一**——评估 train-test action gap 对结论的污染（R16）。

### 论文叙事风险

若长程对比（实验 2+3）做不出或 NAV 不优于 HPMC，"固定预算递归更新记忆优于
启发式窗口"这一强主张站不住，需退回到弱主张"Register 是一个可导航的固定
预算记忆"。当前 Stage One 定位为"初步验证"：仅证明 Register 机制能训得动、
训练分布内能工作、结构上比 HPMC 省——仅此而已。
