# NAV 项目当前进展与下一阶段

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-OVR-003` |
| 类型 | 项目状态（Project Status） |
| 状态 | Live / Executive Summary |
| 更新时间 | 2026-08-14（Asia/Shanghai） |
| 职责 | 汇总模型、数据、训练、评测、运行任务、风险和下一里程碑 |

## 一句话状态

**2026-08-14 设计覆盖说明**：V1 正式结构已按 DEC-034 更新为
`A_noise -> action flow decoder -> A_out`、`H_action=H_nav=10`、
dual-stream / MoT-style backbone，以及明确 causal attention 关系。此前
`A_query` / single hidden-width / direct CE logits 相关训练与测速只能视为历史
smoke 或 diagnostic，不再代表最终正式结构。随后 DEC-037 将 Register 侧统一为
`RegisterCell`：每个 episode 从 fixed `R_null` 开始，
`R_i=RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))`；`A_hist`
通过独立 action tokens / cross-attention context 交互，禁止 action bias /
latent bias / additive bias。当前代码仍需按 DEC-034/037 重构后再启动新的正式训练。

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
长程 autoreg 对比（R13）。详见 `../04_training/v0_stage_one_dl3dv_experiments.md` 验证
边界段与 `../08_insight/insight_log.md` R13–R16。

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
| `nav_v1_stageone_text_full5000_tensorboard` | CPU | TensorBoard for text-conditioned fullmix run | 监听 `0.0.0.0:6013` |
| `nav_v1_stage3_vln_render_r2r` | 0 | R2R-CE standard train/val_seen/val_unseen RGB 渲染 | 目标 13,436 episodes / 72 scenes；输出 `/sharedata/NAV/derived/v1/vln/rendered_obs/stage3_vln_render_r2r_standard_gpu0_20260812_100122/` |
| `nav_v1_stageone_final_train` | 1 | V1 Stage One 1000-step 历史 run | 已完成；run `v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650` |
| `nav_v1_stageone_final_tensorboard` | CPU | TensorBoard for current Stage One run | 监听 `0.0.0.0:6012` |
| `nav_v1_t4_latent_spatial20_gpu0_0`–`_5` | 0 | SpatialVID 20% T4 micro latent 6-shard 数据准备 | 继续后台运行，原始下载目录不改动 |
| 已完成 | 0/1 | T4 micro latent: DL3DV / RE10K | 已进入当前训练窗口池 |

实时数字以后以 `../03_data/dataset_preparation_status.md` 为准。

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
`project_invariants.md` 进一步重写。

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

- 模型接口：`../01_design/v1_action_centered_io_interface.md`
- 待定项与默认选择：`../01_design/v1_open_design_questions.md`
- 三阶段数据、变量、loss：`../04_training/v1_three_stage_training_data_plan.md`
- Stage One T4/IW 对齐训练：`../04_training/v1_stage_one_t4_iw_aligned.md`
- 数据 tensor 与窗口：`../03_data/training_data_construction.md`
- Action / geometry schema：`../03_data/v1_action_geometry_schema.md`

运行与资源：

- 实时数据状态：`../03_data/dataset_preparation_status.md`
- VLN 四套闭环数据整备：`../03_data/vln_dataset_preparation_status.md`
- 资源路径：`../06_operations/resource_inventory.md`
- 决策记录：`decision_log.md`

历史与 baseline：

- V0 Register 在线训练语义：`../04_training/v0_streaming_training_sample_semantics.md`
- V0 Stage Two 网络：`../01_design/v0_stage_two_3d_supervision.md`
- V0 Stage Two 训练：`../04_training/v0_stage_two_3d_supervision_training.md`
- V0 Stage Three 导航：`../01_design/v0_stage_three_navigation.md`
- V0 多数据集课程：`../04_training/v0_multidataset_curriculum.md`
- R2R-CE ≥StreamVLN SOTA 复现：`../05_evaluation/r2r_ce_sota_reproduction.md`
