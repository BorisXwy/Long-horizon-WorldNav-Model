# NAV 项目当前进展与下一阶段

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-003` |
| 类型 | 项目状态（Project Status） |
| 状态 | Live / Executive Summary |
| 更新时间 | 2026-09-28（Asia/Shanghai） |
| 职责 | 汇总模型、数据、训练、评测、运行任务、风险和下一里程碑 |

## 当前快照（2026-09-28）

当前主线已经从“持续启动旧版 Stage2 训练”转为 **V1 数据/模型规范冻结后的可复现实验与 GigaNav 消融**。下方按 2026-08-14 记录的 Stage1/Stage2 运行状态均为历史，不代表当前仍在训练。

### 当前实现与训练

- V1 完整模型、Stage1/2/3 数据接口、Register、shared WanBlock、video/policy 分支和 probe 代码已在 `src/` 中模块化；训练配置通过 YAML 选择 video-only、policy-only 或 cotrain。
- GigaNav 是独立导航消融，不替代 V1 Register/3D 主线：使用官方 Wan2.1-T2V-1.3B 初始化，H=8 离散 policy，支持 policy-only 与 AC-WM/WAM cotrain。
- 最新 GigaNav cotrain 已完成 5,000 optimizer steps，日志与 checkpoint 位于 `log/giga_nav_wan21_h8_cotrain_adamw_gpu1_from_wan_20260907/`；policy-only 历史 run 位于 `log/giga_nav_wan21_h8_ebs32_20260903_003600/` 及其续训目录。当前没有 NAV 模型训练进程，TensorBoard 历史转发仍可从 `6044/6045` 查看。
- 最新 cotrain 使用完整 1.3B Wan backbone、AdamW、物理 batch 1、梯度累积 32、EBS=32；最后记录的 video flow loss 约 0.20，policy CE 需按 WAM 有效样本归一化解读，尚不能作为最终导航质量结论。

### 当前数据与流水线

- 视频侧当前可核验的主 manifest 是 `24,261 episodes / 155,051 micro chunks`；正式候选集是 SpatialVID deterministic 20% 加上全部 RE10K、DL3DV、Argoverse2，共 `5,180 episodes / 247,801 chunks`，已编码 T4 latent 约 `185.33 GiB`。
- VLN 侧 R2R train 已完成 simulator 渲染、STOP 吸收态 padding、T4 latent 编码和 UMT5 instruction embedding：`10,819 episodes / 87,345 chunks / 65.35 GiB latent`；RxR 只保留已完成的部分 budget latent。
- 统一规则是“原始数据不改写 → manifest → pose/action sidecar → T_latent=4 chunk → VAE latent → 可选 text embedding → 训练时在线构造 history/Register window”。Register 不预存，history 长度在 sampler 中复用同一 episode latent 抽取。
- RE10K/DL3DV/SpatialVID/Argoverse2 的 action 来自相机 pose；R2R/RxR 的 action 来自 simulator expert path。视频数据 action 进入 `a_condition`，VLN action 进入 `a_label`，统一离散为平移/旋转组合；完整血缘和目录见 `doc/02_data/data_preparation_schema_and_status.md`。

### 当前风险与下一步

1. 不把 full manifest 当成 full latent：当前已确认的是 SpatialVID 20% cache 和完整 R2R train cache。
2. RE10K、Sekai、Ego4D 后台窗口正在等待 cookie/AWS credentials，恢复后仍需重新核对下载量和 latent 完成度。
3. 下一步应在固定的数据 cache 上继续做完整 V1/GigaNav 对照、生成质量和开环 policy 评测，再决定是否恢复大规模数据准备。

## 历史快照（2026-08-14）

**2026-08-14 Stage Two pose-only 正式验证训练已启动**：当前正在运行一轮
RE10K-only Stage Two 训练。除数据集限制为 RE10K 外，模型结构、latent 几何和
loss 形式均按 V1 最终标准执行：

```text
tmux:
  nav_stage2_re10k_poseonly_formal

run:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440/

doc:
  NAV/doc/03_training/training_plan_and_experiment_log.md
```

启动前已通过 full-shape preflight：`[16,4,56,112]` T4 latent、30-layer
Wan-size visual stream、128 Register tokens、pose-only `L_visual + λ_pose L_pose`。
GPU preflight 在 batch=1、grad_accum=1 下通过，峰值约 39.73 GiB；正式训练
使用 micro_batch=1、grad_accum=16、effective_batch=16，step1 用时 75.43s，
峰值约 39.93 GiB。

**2026-08-14 Stage Two diagnostic 已降级**：此前用 RE10K T4 micro latent 与
`Wan2.1-T2V-1.3B` compatible initialization 跑过一次“视频生成 +
current-hidden pose supervision”diagnostic，但该 run 缩小了 latent spatial
resolution 且缩小了 backbone depth，违反当前“不得小型化验收”的项目规则。
它不作为 Stage Two 完整训练、正式 smoke、验收结果或方案有效性证据。记录仅作
调试追溯：

```text
run:
  NAV/log/v1_stage2_re10k_pose_video/v1_stage2_re10k_pose_video_1000step_20260814_022646/

doc:
  NAV/doc/03_training/training_plan_and_experiment_log.md
```

该 run 的数值不得再用于证明 pose supervision、视频生成质量或 Stage Two 方案
有效。后续 Stage Two pose-only 训练必须使用完整模型结构、完整 latent/video
几何和真实 RE10K pose target；只允许缩小数据量、步数和 batch/grad-accum。

**2026-08-14 代码状态更新**：V1 正式完整模型链路已经落到
`NAV/src/nav/v1/models/full_model.py`，并通过 full-pipeline smoke：

```text
NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json
```

验证内容包括 Stage One video generation loss、Stage Two 3D additional loss、
Stage Three policy/action loss + video/3D rehearsal loss、参数梯度/更新审计，
以及 videogen / policy 两种推理路径。旧 scaffold、旧 `A_query`、旧 action-bias
和旧 InfiniteWorld adapter 可执行入口已从当前代码主线删除；历史结论只保留在
V0 历史章节中。

**2026-08-14 设计覆盖说明**：V1 正式结构保留
`A_noise -> action flow decoder -> A_out`、`H_action=H_nav=10` 和明确 causal
attention 关系，但 DEC-041 已覆盖 DEC-034 中的 dual-stream / MoT-style backbone：
主结构必须是 **single shared WanBlock token stream**。此前
`A_query`、direct CE logits，以及当前代码中复制 action expert 的 dual-stream
路径都只能视为历史 smoke/diagnostic 或待清理实现，不再代表最终正式结构。
随后 DEC-037 将 Register 侧统一为
`RegisterCell`：每个 episode 从 fixed `R_null` 开始，
`R_i=RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`；`A_hist`
通过独立 action tokens / cross-attention context 交互，禁止 action bias /
latent bias / additive bias。

**2026-08-14 代码状态更新**：`src/nav/v1/models/full_model.py` 已从
`V1DualStreamBackbone` 切到 `V1SharedWanBackbone`。当前静态参数统计：
total ≈ `1.217B`，backbone ≈ `1.109B`，不再是 dual-stream 版本的 ≈`2.89B`。
语法检查通过，但正式 Stage One/Two/Three 训练和 full-pipeline smoke 仍需基于
shared WanBlock 版本重新启动，不能沿用 DEC-041 前的 smoke/run 作为验收。

V1 Stage One 已进入 text-conditioned fullmix 正式训练流水线：从官方
`Wan2.1-T2V-1.3B` `diffusion_pytorch_model.safetensors` 初始化 shared DiT
backbone，使用 `latent_prefix` Register、完整 Stage One action interface、
T_latent=4 micro chunk、IW-1/4/8/16 等价长历史窗口混训，目标 **5000
optimizer steps**，effective batch size 16，每 1000 steps 保存。GPU1 先缓存
SpatialVID UMT5 text condition，完成后自动启动训练；GPU0 继续 SpatialVID 20%
T4 latent 数据准备。

**Stage One 初步验证边界（2026-08-07 界定）**：已验证——训练能简单收敛、
Register 在训练集上单步保真正常（gthist 记录 8：A@500/B@1000 L1=0.0874/
0.0917 优于 InfiniteWorld）、Register 固定预算且比 HPMC 小。**未验证**——
泛化与 action 跟随、长程有效显存缩减（NAV 16+ chunk autoreg 未跑）、autoreg
等价于 teacher-forced（B 在 3-chunk autoreg DD 塌缩）、Register 优于 HPMC
（3-chunk 下 HPMC 不触发二次压缩、与全量历史等价，比不出差别）、text 通道
可用（Stage One 空 text）、train-test action 一致。核心待补：长程训练 +
长程 autoreg 对比（R13）。详见 `../03_training/training_plan_and_experiment_log.md` 验证
边界段与 `../05_insight/insight_log.md` R13–R16。

三阶段边界已确定：Stage One 是 video generation memory pretraining；
Stage Two 在同一视频生成前向中增加 3D supervision/probe；
Stage Three 进行 VLN action/nav supervision，并 replay Stage One/Two loss
避免 shared backbone/Register 退化。

## 已完成

### 模型

- 以 Infinite-World/Wan2.1-T2V-1.3B 为 backbone；
- 移除 HPMC history，保留 latest local memory；
- A：固定 `[B,16,4,H,W]` latent-like Register prefix；
- B：16个Register token投影后与UMT5 text condition拼接；
- Register 在 chunk 之间递归更新，不使用增长型 KV Cache；
- A/B 都支持 full-parameter training。

### RE10K 基线

- A/B 均从原始 Infinite-World checkpoint 初始化并完成1000 steps全参数训练；
- 当前 SpatialVID 续训分别加载各自 `full-final.pt`，A/B 不交叉继承；
- TensorBoard、checkpoint和训练日志已隔离保存。

### 训练语义

- 每个 chunk 为81个RGB帧、21个Wan VAE时间位置；
- 当前采用2–3 chunk random-prefix next-chunk teacher forcing；
- history 使用干净真实 latent；
- Register 在 forward 内在线构造，不能提前缓存；
- local memory 为最新 history chunk 的最后一个 latent frame；
- 只对窗口最后一个 target 计算 RFlow diffusion loss；
- chunk之间不detach，loss可穿过全部Register updates反传。

### Stage One 1.0 from-scratch 训练（DEC-019，当前）

- 目标改为 **1000 effective optimizer steps、effective batch size 16**；
- A：`latent_prefix`，micro_batch=1、gradient_accumulation=16、effective=16，
  NAV 进程峰值显存约 33.24 GiB；
- B：`dit_condition`，micro_batch=2、gradient_accumulation=8、effective=16，
  NAV 进程峰值显存约 28.71 GiB；
- A/B 均 from scratch：仅加载原始 Infinite-World checkpoint 的 Wan/DiT
  预训练权重；旧历史 A/B 模块中的 Extractor/Updater 随机初始化，不加载旧 RE10K 或旧 A/B-10000
  checkpoint（`resume_checkpoint=null`、`register_checkpoint=null`）；
- 数据只用 DL3DV 141 条 full episode latent 与 dense Action，文本用 empty
  UMT5；
- 每 100 effective steps 保存一次完整 checkpoint，不使用 plateau early stop；
- run 名：`stage-one-v1-from-scratch-a-ebs16-1000`、
  `stage-one-v1-from-scratch-b-ebs16-1000`。

旧 DEC-018 的 10000-step 计划已被 DEC-019 取代；旧 `stage-one-v1-dl3dv-{a,b}-10000`
run 作为历史保留，不用于正式续训。

### Stage One–Three 已确认边界（尚未实现）

- Stage One：该旧描述已被 DEC-037 覆盖；新版从 fixed `R_null` 开始，首步和后续
  都由同一个 `RegisterCell` 递归更新 Register；
- Stage Two：完整保留noisy target、Local、Action/Text、DiT和Diffusion/RFlow
  视频生成训练；
- Stage Two：先冻结backbone probe多个DiT层的Register-after-DiT，再选择最适合
  3D监督的层并联合训练；
- Stage Two不训练Navigation Policy；
- Stage Three：读取选定层的Register-after-DiT并接入Navigation Head/Policy；
- checkpoint只保存DiT、RegisterCell和readout能力参数，不保存某个
  episode的Register；
- 导航History只来自真实Observation，generated future不回灌。

### 数据与标注

- SpatialVID 30,000条已完成资格扫描和Action标注；
- 23,837条满足至少2 chunks，6,163条因不足162帧排除；
- RE10K 269、DL3DV 141、Argoverse 2 14条已生成pose-aligned Action manifest；
- DL3DV 已修正为使用与Pose一一对应的官方 `images_8`，不再误用原MP4帧号；
- Action与VAE latent已解耦，可并行准备；
- 原始下载目录保持只读。

### 验证

- SpatialVID 独立 action/latent 窗口元数据一致；
- A/B 都通过“外部action sidecar + online Register + RE10K checkpoint”的真实
  单步full-parameter forward/backward；
- Smoke A loss `0.05786`、Register grad norm `0.3926`、NAV峰值29.23GiB；
- Smoke B loss `0.05701`、Register grad norm `0.3711`、NAV峰值16.52GiB；
- 上述单点只验证链路，不作为A/B质量比较。

## 正在运行

| tmux | GPU | 任务 | 当前快照 |
| --- | ---: | --- | --- |
| `nav_v1_stageone_text_full5000` | 1 | V1 Stage One text-conditioned fullmix 5000 steps | 已完成 SpatialVID text cache 并进入训练；step 1 loss=1.3859，peak_reserved=32.78 GiB；run `v1-stageone-text-fullmix-wan21official-lp-mb1-ebs16-5000-20260812-020056` |
| `nav_stage2_re10k_poseonly_formal` | 0 | V1 Stage Two RE10K-only pose supervision formal verification | 正在跑 1000 optimizer steps；full T4 latent `[16,4,56,112]`、Wan-size 30 layers、Register 128、effective batch 16；run `v1_stage2_re10k_poseonly_formal_wanfull_re10k_20260814_092440` |
| `nav_v1_stageone_text_full5000_tensorboard` | CPU | TensorBoard for text-conditioned fullmix run | 监听 `0.0.0.0:6013` |
| `nav_v1_stage3_vln_render_r2r` | 0 | R2R-CE standard train/val_seen/val_unseen RGB 渲染 | 目标 13,436 episodes / 72 scenes；输出 `/sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122/` |
| `nav_v1_stageone_final_train` | 1 | V1 Stage One 1000-step 历史 run | 已完成；run `v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650` |
| `nav_v1_stageone_final_tensorboard` | CPU | TensorBoard for current Stage One run | 监听 `0.0.0.0:6012` |
| `nav_v1_t4_latent_spatial20_gpu0_0`–`_5` | 0 | SpatialVID 20% T4 micro latent 6-shard 数据准备 | 继续后台运行，原始下载目录不改动 |
| 已完成 | 0/1 | T4 micro latent: DL3DV / RE10K | 已进入当前训练窗口池 |

实时数字以后以 `../02_data/data_preparation_schema_and_status.md` 为准。

## 自动后续链

```text
Stage One 1.0 A/B ebs16 1000 steps 完成
        +
full_episodes_v1 latent 全部就绪（SpatialVID/Argoverse2 补齐）
        ↓
按 DEC-019 后续决策进入 Medium/Long 或 Stage Two
```

正式配置为 `NAV/config/train_stage_one_v1_dl3dv.yaml`（steps 已调整为 1000
effective、effective batch 16）。

## 关键风险

1. SpatialVID 样本量占绝对多数，未来多数据训练不能直接统一shuffle；
2. 尚无正式 scene-level split，当前流水线适合训练准备，不足以产生无泄漏指标；
3. Action 是camera-motion pseudo-label，不等于真实控制输入；
4. 跨数据集尺度按episode中位变化归一化，绝对速度语义不一致；
5. teacher forcing 与推理生成history之间存在exposure bias；
6. 旧DL3DV cache可能存在Pose/MP4帧误配，只能使用新的
   `latents_pose_aligned/dl3dv`；
7. Short缓存最多3 chunks，不能直接用于宣称Medium/Long已经准备完成；
8. Kinetics当前VGGT结果是episode级估计，不可伪装成逐帧可靠Action。

## 下一阶段门槛

### 当前 V1 Stage One 正式训练后的顺序

当前 `v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000`
结束后，不直接进入下一轮混合训练，先做 generation quality check：

1. 使用 final 或 step1000 checkpoint 做小规模生成质检；
2. 至少覆盖 IW-1 / IW-4 / IW-8 / IW-16 history 条件；
3. 检查视频是否可解码、是否黑屏、是否有像素变化、是否与 history/local
   observation 连贯；
4. 与官方 Wan2.1 init / InfiniteWorld baseline / 旧 V0 A/B 中可比设置做同协议
   对照时，必须明确 target horizon 是 `T_latent=4`，不能直接与81-frame target
   loss 数值混比；
5. 通过基本生成质检后，再进入 mixed target dataset training：重新扫描
   `/sharedata/NAV/derived/v1/t4_micro_latents_spatial20/`，纳入新落盘的
   SpatialVID 20%，并使用 dataset-aware sampler。

该顺序对应当前 V1 主线；下方部分旧门槛保留为 V0/历史参考，后续需要按
`project_rules_and_documentation_standard.md` 进一步重写。

SpatialVID Short 正式训练前：

- action/latent配对校验为零缺失；
- 固定train/validation scene split；
- 输出Action类别和calibration分布；
- 抽样可视化Pose轨迹、动作方向和视频；
- 冻结数据版本及异常排除清单。

进入 Medium/Long 前：

- 实现dataset-balanced sampler；
- 选择last-target、多target horizon或truncated BPTT；
- 决定是否加入scheduled sampling；
- 使用DL3DV等长episode进行≥8 chunks实验；
- 建立InfiniteWorld HPMC、A、B和无长期记忆基线。

进入 Stage Two 前：

- 完成 Stage One unified RegisterCell 代码改造；
- 明确A/B的Register-after-DiT读取规则；
- 完成冻结backbone的多层3D diagnostic probe；
- 固定81 RGB frames到21 latent time bins的Teacher对齐规则；
- 验证3D loss到DiT、RegisterCell的梯度路径；
- 保证Diffusion/RFlow生成loss始终参与Stage Two训练。

进入 Stage Three 前：

- 冻结Stage Two选定的DiT层号、Register slicing、shape和normalization；
- 完成A/B的Register-before/after-DiT对照；
- 输出3D-aware视频生成checkpoint；
- 通过真实Observation-only history与generated-future禁用测试。

## 文档入口

当前 V1 主线：

- 模型接口：`../01_model/model_evolution_and_current_architecture.md`
- 待定项与默认选择：`../01_model/model_evolution_and_current_architecture.md`
- 三阶段数据、变量、loss：`../03_training/training_plan_and_experiment_log.md`
- Stage One T4/IW 对齐训练：`../03_training/training_plan_and_experiment_log.md`
- 数据 tensor 与窗口：`../02_data/data_preparation_schema_and_status.md`
- Action / geometry schema：`../02_data/data_preparation_schema_and_status.md`

运行与资源：

- 实时数据状态：`../02_data/data_preparation_schema_and_status.md`
- VLN 四套闭环数据整备：`../02_data/data_preparation_schema_and_status.md`
- 资源路径：`../06_operations/resource_inventory.md`
- 决策记录：`decision_log.md`

历史与 baseline：

- V0 Register 在线训练语义：`../03_training/training_plan_and_experiment_log.md`
- V0 Stage Two 网络：`../01_model/model_evolution_and_current_architecture.md`
- V0 Stage Two 训练：`../03_training/training_plan_and_experiment_log.md`
- V0 Stage Three 导航：`../01_model/model_evolution_and_current_architecture.md`
- V0 多数据集课程：`../03_training/training_plan_and_experiment_log.md`
- R2R-CE ≥StreamVLN SOTA 复现：`../04_evaluation/evaluation_reproduction_and_benchmarks.md`
