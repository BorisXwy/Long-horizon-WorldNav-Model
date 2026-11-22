# 3D 流式导航世界模型的初步验证与探索

- 文档 ID：`NAV-THS-001`
- 类型：学位论文章节草稿
- 状态：`Draft / Evidence-backed`
- 更新时间：2026-11-22
- 职责：总结 3D 流式导航世界模型的研究动机、方法设计、实验设置、初步结果与结论边界

## 本章摘要

本章研究一个介于视频世界模型（video world model）与视觉语言导航（Vision-and-Language Navigation, VLN）之间的问题：能否以视频生成模型为统一 backbone，将持续到来的视觉观测和已执行动作压缩为固定大小、可在线更新的 Register 状态，并依次通过未来视频预测、3D 几何监督和导航策略监督，使该状态同时具备预测性、空间可读性与可行动性。

围绕这一问题，本研究完成了从 InfiniteWorld/HPMC 到 Register memory、从 81-frame 大视频块到 `T_latent=4` 微块、从单纯视频生成到 3D pose co-training、再到 action-centered policy interface 的多轮验证。初步实验得到四项较为明确的结论。第一，在旧版 20 对配对 VBench 子集上，将固定大小 Register 作为 latent prefix 的方案 A 相对 InfiniteWorld 的四项均值提高 `0.005159`，且 Aesthetic Quality 提高 `0.029293`，说明显式递归状态至少能够替代部分 HPMC 历史压缩功能，但 Imaging Quality 同时下降 `0.008618`，尚不能据此断言全面优于 HPMC。【已验证→`result/vbench/20260727_threeway_vbench_2chunk_autoreg_stats20/metrics/report.md`】第二，基于 Wan2.1-T2V-1.3B 的 `T_latent=4` 模型能够在同一训练中覆盖 IW1/IW4/IW8/IW16 等价历史，1000 optimizer steps 后训练 loss 从 `1.5173` 降至 `0.0535`，并在四种历史长度下均解码出非黑屏、非静态的 13-frame 视频；但样本仅四条，且 latent L1 随历史增长而升高，因此它验证的是链路可训练性而非长程记忆收益。【已验证→`log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/train.log`；`result/stageone_t4_generation_sanity/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650_retry_20260812/summary.json`】第三，RE10K pose supervision 可与视频生成联合优化：300-step 配对实验中 co-training 与 video-only 的视觉 loss 基本重合，同时 pose eval loss 降至 `0.003712`；完整训练后 pose rotation error 达到 `6.31°`，30-step 视频去噪达到 `23.16 dB` PSNR。【已验证→`result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json`；`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】第四，导航策略尚未形成可靠闭环能力：正式 V1 Stage3 在自然 R2R 分布上明显低于多数类基线；GigaNav 消融虽在 `val_unseen` open-loop 达到 `57.96%` token accuracy，却在 20 条 train episode 的闭环测试中得到 `0% SR / 0% SPL`，且 `94.91%` 动作为 MOVE_FORWARD。【已验证→`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`；`result/giga_nav/step5000_r2r_val_unseen_open_loop.json`；`result/giga_nav/step5000_r2r_train_rolling13_full20_20260905/summary.json`】

因此，本章的阶段性贡献不是宣称已经得到完整可用的导航世界模型，而是建立了一个可复现的研究框架，并验证了其中“固定预算记忆可训练”“3D 监督可与生成共存”“world-model 能力可通过 replay 保留”三项基础条件；同时以负结果明确指出，下一阶段的关键不在继续叠加任务 head，而在于验证 Register 的真实空间记忆、抑制动作分布捷径，并使 instruction-conditioned open-loop 学习转化为稳定的 closed-loop control。

## 1 研究背景与问题定义

### 1.1 从视频生成到流式导航的结构矛盾

视频扩散模型擅长在高维时空 token 上学习外观、运动与条件之间的联合分布，但常规视频生成通常一次处理固定长度的视频块，计算量随历史帧数增长。导航智能体则需要长期运行：每一步只接收少量新观测，却必须持续记忆已经经过的房间、转弯、遮挡和指令进度。如果直接把全部历史帧拼接到 DiT 中，token 数量和显存会随时间增长；如果只保留最近观测，又会丢失长程空间状态。

另一方面，视频生成与导航策略的输入输出并不对称。视频分支需要以未来 noisy video latent 为主输入，并进行多步 denoising；policy 分支只需要读取历史状态、当前观测与 instruction，直接输出少量离散动作。如果没有严格定义 token 可见性，policy 可能在训练时读取推理阶段不存在的 future visual，或者视频噪声与 action noise 相互污染。即便 open-loop action accuracy 较高，这类 leakage 或动作频率捷径也可能在闭环中迅速累积成失败。

### 1.2 本研究的核心问题

本研究将问题概括为：

> 能否在一个 Wan-style video diffusion backbone 内，将历史视觉和已执行动作递归压缩为固定预算 Register，并用 3D supervision 约束其空间可读性；训练时借助 future video prediction 提供稠密动态信号，推理时关闭视频生成，仅使用 Register、当前观测、instruction 与 policy readout 完成闭环导航？

由此得到三个递进的研究问题：

1. **预测性（predictive memory）**：固定大小 Register 能否吸收越来越长的历史，并维持未来视频生成能力？
2. **空间可读性（spatial readability）**：Register/current hidden 是否能够恢复 current observation 中的相机 pose，并在加入 3D loss 后不破坏生成能力？
3. **可行动性（actionability）**：相同 backbone 的 policy token 能否在不读取 future video 的条件下，将 instruction 与视觉状态转化为可靠的闭环动作？

目前第一项已获得可训练性和短程生成证据，但缺少严格的 zero/shuffled Register 长程对照；第二项已获得 pose probe 收敛和视觉不退化证据，但尚未证明这些几何特征真正服务于导航；第三项尚未通过闭环验收。【已验证→`result/stageone_t4_generation_sanity/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650_retry_20260812/summary.json`；`result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json`；`result/giga_nav/step5000_r2r_train_rolling13_full20_20260905/summary.json`】

## 2 方法设计

### 2.1 固定预算的流式 Register

每个 episode 从固定且不可训练的空状态 `R_null` 开始。对于已经发生的第 (i) 个观测块 (C_i) 及其已执行动作 (A_i^{hist})，Register updater 递归计算

\[
R_i=\operatorname{RegisterCell}_{\theta}
\left(R_{i-1},\operatorname{VisualToken}(C_i),A_i^{hist}\right).
\]

Register 的 token 数量在整个 episode 内保持不变，因此历史长度增加时，backbone 的显式主输入不会随之线性增长。该设计并不把 `R_null` 当作可学习的场景先验；需要被 checkpoint 保存的是 extractor/updater 的压缩能力，而不是某个训练场景的初始 Register 内容。

为避免 current observation 被过早压缩，时间边界定义为

\[
C_{0:t-2}\rightarrow R_t,\qquad C_{t-1}\rightarrow Z_{obs},\qquad C_t\rightarrow Z_{future}^{target}.
\]

即更早的历史进入 Register，最近一个真实 observation chunk 作为局部细节输入 `Z_obs`，下一 chunk 作为视频预测目标。导航推理中也只使用真实观测更新 Register，不把模型生成结果写回历史。

### 2.2 Action-centered 统一接口

模型区分三种语义不同的动作：

- `A_hist`：过去已经执行的动作，与历史视觉块对齐，只用于 Register update；
- `A_cur`：控制未来视频后果的当前动作，作为 video generation condition；
- `A_noise -> A_out`：policy 查询槽位与输出，Stage3 中由 action head 解码为导航动作。

视频数据只提供 camera motion/pose-derived action，因此用于 `A_hist` 与 `A_cur`；VLN 数据提供 expert action，因此用于 `A_hist` 与 `A_out` 监督。两者共享“平移—旋转组合动作”的数据接口，但不能把视频 camera motion 与 R2R 的离散机器人动作视作完全相同的物理 ontology。

### 2.3 Shared WanBlock 与因果可见性

正式主线使用单一共享 WanBlock，而不是为视频和动作各建一套 Transformer。主输入、condition 和输出关系如下：

```text
history visual + A_hist
          |
          v
  fixed-budget Register R_t -------------------------------+
                                                            |
Z_obs tokens --------------------+                          |
Z_future noise tokens -----------+--> shared WanBlock ------+--> video velocity
A_noise / policy slots ----------+                          +--> action hidden -> logits
                                  ^                          +--> current/Register hidden -> pose
                                  |
                     text tokens + A_cur condition
```

其中 video future token 可以读取 `Register + Z_obs + text + A_cur`；policy token 可以读取 `Register + Z_obs + text + self action slots`，但不得读取 future visual noise、clean future target 或 `A_cur`。视频噪声 token 与 action query/noise token 之间保持分支隔离，避免训练阶段的 future information 进入 policy。当前 mask 约束已有完整代码路径，但其必要性仍缺少系统的 leakage ablation。【未验证：尚未完成不同 mask 的同配置闭环对照】

### 2.4 三阶段训练

三阶段并非三个独立模型，而是在同一接口上逐步增加监督：

**Stage1：预测性记忆。** 输入历史视觉、`A_hist`、`Z_obs`、text/empty text 与 `A_cur`，对 `Z_future` 计算 Rectified Flow / diffusion velocity loss：

\[
\mathcal L_{visual}=\|v_{\theta}(z_t,t,\mathcal C)-v^*\|_2^2.
\]

`A_noise/A_out` 的 token、mask 和 forward path 保留，但不计算 policy loss。训练混合 IW1/IW4/IW8/IW16 等价历史，Register update 与 `A_cur` 均混入一定比例的全零条件，以覆盖无动作或未知动作场景。

**Stage2：空间增强。** 数据与 Stage1 相同，在 current clean hidden 或 Register hidden 上增加 frame-level pose probe：

\[
\mathcal L_{stage2}=\mathcal L_{visual}+\lambda_{pose}\mathcal L_{pose}.
\]

pose 监督来自 `Z_obs` 对应的真实帧，而不是未来生成结果。当前验证主要使用相对 translation、rotation 与 FOV；depth/point supervision 尚未形成正式结果。【未验证：depth/point head 尚无可引用的正式训练产物】

**Stage3：导航后训练。** 使用 R2R/RxR 的 observation、instruction 和 expert action，对 action slot 的 hidden 施加离散交叉熵，同时 replay Stage1/2 数据避免遗忘：

\[
\mathcal L_{stage3}=\lambda_a\mathcal L_{action}
+\lambda_v\mathcal L_{visual}^{replay}
+\lambda_p\mathcal L_{pose}^{replay}.
\]

当前 replay 实验使用 `0.25 L_visual + 0.05 L_pose`，在小规模配对评测中保持了生成和 pose 能力。【已验证→`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】

## 3 数据组织与实现口径

### 3.1 统一微块

原始探索使用 InfiniteWorld 风格 81-frame chunk，难以与导航的短步交互对齐。新版将所有视频先按固定窗口切分，再独立进行 Wan VAE 编码：每个微块包含 13 个 RGB frames，stride 为 12，经 VAE 得到近似形状 `[16, 4, 56, 112]` 的 latent，即 `T_latent=4`。同一份离线微块可在训练时组合为不同长度 window，无需为短历史、长历史和导航任务重复编码。

为与 InfiniteWorld 的历史跨度对齐，IW1/IW4/IW8/IW16 分别映射为 7/27/54/107 次微块 Register update。这里的 IW 数量只表示历史时长等价关系，不表示模型重新生成 81 帧的大块。

### 3.2 视频与导航数据

当前可核验的数据规模如下。

| 数据侧 | 数据组成 | 已完成规模 | 训练用途 |
| --- | --- | ---: | --- |
| 视频全量 manifest | SpatialVID、RE10K、DL3DV、Argoverse2 | 24,261 episodes | 视频/action/pose 数据总账 |
| V1 正式候选 | SpatialVID deterministic 20% + 其余全量 | 5,180 episodes，247,801 T4 chunks，185.33 GiB latent | Stage1/2 visual mixing |
| R2R train | Habitat rendered RGB + expert action + instruction | 10,819 episodes，87,345 chunks，65.35 GiB latent | Stage3 policy |
| R2R text | UMT5 instruction embedding | 10,819 embeddings，169.19 GiB | Stage3 text condition |
| RxR 部分缓存 | rendered/encoded subset | 532 episodes，6,508 chunks，4.87 GiB | 后续多语言扩展 |

以上数据规模来自落盘 manifest 与缓存统计。【已验证→`doc/02_data/data_preparation_schema_and_status.md`】视频样本有 text 时直接编码，没有 text 时使用约定的 empty text embedding；R2R terminal `STOP` 按吸收态处理，STOP 后复制最后一帧并填充 STOP action，但训练 window 不应重复采样纯 padding 尾段。

### 3.3 Backbone 与训练配置

主要实验使用 Wan2.1-T2V-1.3B official checkpoint。Stage1 正式 run 载入 825 个官方 key，采用 full-parameter training，总可训练参数 `1,422,300,004`；micro batch size 为 1，gradient accumulation 为 16，effective batch size 为 16，峰值 reserved memory 为 `39.12 GiB`。【已验证→`log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/train.log`】模型沿用 Wan 的 30 层、hidden width 1536、12 heads 主干，并增加 Register updater、action interface、pose probe 与相应 token type/mask。

## 4 初步实验与结果

### 4.1 实验一：Register 能否替代 HPMC 历史压缩

早期 V0 在 InfiniteWorld 权重上构造两种 Register 注入方式：方案 A 将 Register 变换为 latent-prefix 风格 token，仅替换 HPMC history memory；方案 B 将 Register 与文本条件共同进入 DiT condition。三版模型在相同的 20 个 prompt×seed、2-chunk autoregressive、30 denoising steps、CFG 5.0 协议下进行 VBench 四项配对评测，并使用 10,000 次 bootstrap 计算 95% 置信区间。

| 模型 | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Imaging Quality | 四项平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| InfiniteWorld | 0.981515 | 1.000000 | 0.613715 | 0.729293 | 0.831131 |
| A：latent prefix | 0.981475 | 1.000000 | 0.643008 | 0.720675 | **0.836290** |
| B：condition | 0.981780 | 0.950000 | 0.628197 | 0.714298 | 0.818569 |

相对 InfiniteWorld，A 的四项平均差为 `+0.005159 [0.002738, 0.007967]`，Aesthetic Quality 差为 `+0.029293 [0.018802, 0.041482]`，但 Imaging Quality 为 `-0.008618 [-0.015926, -0.002075]`。B 的四项平均差为 `-0.012562`，区间跨零。【已验证→`result/vbench/20260727_threeway_vbench_2chunk_autoreg_stats20/metrics/report.md`】

在单 episode/seed 的 teacher-forced“3 个 GT history chunk 预测第 4 个”诊断中，A/B/InfiniteWorld 的 latent L1 分别为 `0.0874/0.0917/0.1152`，SSIM 为 `0.5412/0.5449/0.4733`，LPIPS 为 `0.3150/0.3290/0.3726`；但该结果样本数为 1，只能作为链路诊断，不能作为统计结论。【已验证→`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`】进一步的 3-chunk autoregressive 诊断中，A@500 的 VBench 平均约为 `0.8447`，而 B 出现 Dynamic Degree failure、平均约 `0.59`，提示 condition-only Register 更容易在递归生成中积累误差；同样由于单场景，该观察仍需扩大样本验证。【已验证→`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`】

该实验支持“Register 可以进入 video DiT 并承担历史状态”的弱结论，但没有证明 A 在长程和多数据分布上稳定优于 HPMC。新版因此保留 latent/main-stream Register 路线，同时将最终证明标准改为 IW1–IW16 的 zero/shuffled/local-only 对照。【未验证：正式 V1 长程对照尚未完成】

### 4.2 实验二：短生成块与长历史是否可同时训练

V1 Stage1 使用 `T_latent=4` target，并随机混合 IW1/IW4/IW8/IW16 历史。训练集包含 1,064 个满足当前筛选条件的 episode，共构造 214,363 个 temporal windows，其中四种历史分别为 73,788、58,075、46,706 和 35,794 个。1000 optimizer steps 中，loss 从 step 1 的 `1.5173` 降至 step 1000 的 `0.0535`，说明短 future 与多长度 Register rollout 可在完整 Wan backbone 上共同优化。【已验证→`log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/train.log`】

随后分别从 IW1/IW4/IW8/IW16 采样一条 window，执行 30-step denoising。四条结果均可解码，且 `black_like=false`、`static_like=false`。其 latent L1 依次为 `0.1655/0.1814/0.2002/0.2346`，生成视频相邻帧平均绝对差依次为 `0.0469/0.0426/0.0542/0.0642`。【已验证→`result/stageone_t4_generation_sanity/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650_retry_20260812/summary.json`】

该结果证明完整链路可以处理 107 次 Register update 后的 IW16 等价历史，但不能证明模型真正利用了早期历史：四条样本不是同 episode 的受控比较，latent L1 反而随历史跨度增大。后续必须固定 current/future，仅改变有效 history、打乱 history 或清空 Register，才能区分“模型能承受长 rollout”和“模型从长历史获益”。【未验证】

### 4.3 实验三：3D pose supervision 是否破坏视频生成

为验证 3D supervision 与生成目标的兼容性，先在完全配对的数据顺序和随机种子下比较 visual-only 与 visual+pose co-training。300-step gate 中，co-training 最终 visual eval loss 为 `0.411553`，video-only 为 `0.413210`；61 个公共记录点的 `co-training - video-only` 平均 visual loss 差仅 `+0.000410`，同时 pose eval loss 降至 `0.003712`。代价是峰值显存由 `27.94 GiB` 增至 `29.26 GiB`，记录窗口中的单步时间由 `7.57 s` 增至 `11.02 s`。【已验证→`result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json`】

完整 RE10K 训练进一步给出如下趋势：

| checkpoint | visual velocity MSE | pose MSE | translation MAE | rotation error | RGB PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 500 | 0.15448 | 0.003217 | 0.03474 | 7.36° | 21.64 dB |
| step 2000 | 0.10826 | 0.002566 | 0.03346 | 5.65° | 22.11 dB |

【已验证→`result/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_re10kfull_2k_step500_eval/eval_metrics.json`；`result/v1_stage2_iw_aligned_re10k_pose_video/iw_aligned_re10kfull_2k_step2000_eval/eval_metrics.json`】

在后续 Stage2 step4400 上，64 个 mixed samples 的 velocity MSE 为 `0.08693`、latent x0 cosine 为 `0.96178`；RE10K 64 samples/256 valid frames 的 pose MSE 为 `0.002746`、translation MAE 为 `0.03557`、rotation error 为 `6.31°`；4 个样本的 30-step full denoise 达到 `23.16 dB` PSNR 和 `0.04175` RGB L1。【已验证→`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】

这些结果说明 pose readout 可以在不显著牺牲 visual objective 的情况下被优化，但仍属于“几何可读性 proxy”。它没有证明 Register 形成了全局地图，也没有证明 pose 信息被 policy 使用；此外，step3400 到 step4400 的 4-sample PSNR 只提高 `0.0393 dB`，不足以宣称显著提升。

### 4.4 实验四：Stage3 replay 能否避免世界模型能力遗忘

Stage3 step800 与其来源 Stage2 step3400 在同一批样本和随机种子上配对回放。结果如下：

| 指标 | Stage2 source | Stage3 step800 | 相对变化 |
| --- | ---: | ---: | ---: |
| mixed velocity MSE | 0.090470 | 0.090515 | +0.05% |
| mixed latent x0 MSE | 0.029816 | 0.029822 | +0.02% |
| RE10K pose MSE | 0.002720 | 0.002366 | -13.00% |
| translation MAE | 0.04250 | 0.03595 | -15.41% |
| rotation error | 6.59° | 6.29° | -4.52% |
| full-denoise PSNR | 20.1499 dB | 20.1366 dB | -0.0133 dB |

在该小规模配对协议下，`0.25 L_visual + 0.05 L_pose` replay 将生成差异控制在数值波动量级，并保持或略微改善 pose 指标，因此未观察到 catastrophic forgetting。【已验证→`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】不过这只是 step800 的短训练与 4-sample full-denoise 检查，不能替代长周期、多 seed 的 replay ratio 消融。

### 4.5 实验五：导航动作为什么仍未形成可靠策略

正式 V1 Stage3 step800 在自然 R2R train 128 samples 上得到 CE `1.3613`、accuracy `15.63%`、macro recall `26.80%`，显著低于 MOVE_FORWARD 多数类基线 `75.78%`；在四类严格均衡的 128 samples 上，accuracy/macro recall 均为 `39.84%`，高于 25% 随机基线，但逐类 recall 为 STOP `81.25%`、MOVE_FORWARD `6.25%`、TURN_LEFT `34.38%`、TURN_RIGHT `37.50%`。打乱 instruction 只改变 `3.91%` 的 top-1，平均 probability total variation 为 `0.0308`。【已验证→`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】

这说明模型在均衡分布上学到部分区分信号，却在自然分布中形成新的类别偏置，且 instruction 的判别贡献很弱。后续 H=4 与 single-action 实验也出现了预测单一动作的坍缩，例如 H4 step200 的首动作 accuracy 等于多数类基线、macro recall 为 `0.25`，不能视为有效导航学习。【已验证→`result/v1_stage3_action_chunk_compare/paired256_old1400_vs_h4step200_20260828/summary.json`】

为了区分“V1 结构问题”与“Wan backbone 根本学不会动作”，本研究实现了隔离的 GigaNav ablation：保留 Wan2.1-T2V-1.3B backbone，采用 GigaWorld 风格的紧凑 observation/action interface，action horizon 为 8，effective batch size 为 32。step5000 在 R2R `val_unseen` 256 episodes、2048 tokens 上得到 `57.96%` open-loop token accuracy，其中 MOVE_FORWARD recall `82.50%`，其余三类 recall 仅 `14.73%/11.87%/13.28%`。【已验证→`result/giga_nav/step5000_r2r_val_unseen_open_loop.json`】这证明 Wan backbone 可以拟合 R2R 动作分布，但仍大量依赖 MOVE_FORWARD 频率。

更关键的是，在训推时间窗口对齐后的 R2R train 20-episode closed-loop 中，GigaNav 得到 `0% SR`、`0% SPL`、`5% oracle success`、平均 navigation error `11.53 m`，20 条 episode 全部运行到 500 steps，动作中 `94.91%` 为 MOVE_FORWARD。【已验证→`result/giga_nav/step5000_r2r_train_rolling13_full20_20260905/summary.json`】同一 Habitat 链路下，复现的 StreamVLN 在 R2R `val_unseen` 前 20 条 episode 上达到 `55% SR / 45.03% SPL / 70% oracle success / 4.21 m NE`，说明 simulator 与闭环评测协议本身能够区分有效策略。【已验证→`result/vln_ce/streamvln/smoke20_closedloop_r2r20_20260902_211519/metrics/result.json`】

因此，当前失败不能仅用“训练步数不足”解释。更可能的原因包括：离散 expert action 的类别和时序分布不平衡；短窗口 action imitation 没有显式建模 instruction progress 与 candidate geometry；视频 camera motion 与 VLN robot action 的语义间隙；以及 Register 的空间状态尚未通过导航相关任务验证。上述原因中只有类别坍缩和 instruction 弱敏感已被结果直接观察，其余机制解释仍需受控实验。【未验证】

## 5 阶段性讨论

### 5.1 已经建立的正向证据

1. **固定预算历史状态可进入视频 DiT。** V0 的 latent-prefix Register 在 20 对配对 VBench 上保持 Dynamic Degree，并获得更高的综合均值；V1 可在 107 次历史 update 后完成视频解码。【已验证→`result/vbench/20260727_threeway_vbench_2chunk_autoreg_stats20/metrics/report.md`；`result/stageone_t4_generation_sanity/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650_retry_20260812/summary.json`】
2. **短 future 与长 history 可以解耦。** `T_latent=4` 只控制每次预测/更新的局部时间分辨率，Register rollout 控制历史跨度，二者可以在 IW1/IW4/IW8/IW16 混合课程中联合训练。【已验证→`log/v1-stageone-final-wan21official-actioniface-window-longhist-mb1-ebs16-1000-20260811-023650/train.log`】
3. **3D 辅助目标与视频生成并非天然冲突。** 配对 gate 中 visual curves 几乎重合，pose loss 同时下降；完整 checkpoint 也能执行 full denoise 和 pose readout。【已验证→`result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json`；`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】
4. **能力保持可以显式度量。** Stage3 replay 在配对评测中维持了 visual/pose 指标，说明导航后训练不必必然遗忘世界模型能力。【已验证→`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】

### 5.2 尚未成立的关键假设

1. **尚未证明 Register 记住了对任务有用的长历史。** 当前结果只证明长 rollout 能运行，未完成 local-only、zero Register、shuffled history 和 revisit 对照。【未验证】
2. **尚未证明 pose probe 等价于 3D scene representation。** 低 pose loss 可能来自 current local hidden，而不一定来自历史 Register；需要 layer/register source probe 和 no-3D ablation。【未验证】
3. **尚未证明共享 backbone 带来导航增益。** V1 policy 与 GigaNav 都没有通过闭环验收；open-loop accuracy 不能替代 SR/SPL。【已验证→`result/giga_nav/step5000_r2r_train_rolling13_full20_20260905/summary.json`】
4. **尚未证明当前 action schema 足以统一视频与 VLN。** pose-derived camera motion 与 simulator discrete action 只在接口形式上统一，物理含义和频率分布仍不同。【未验证】

### 5.3 从负结果得到的设计修正

导航失败使研究重点从“再增加一个 policy head”转向“证明共享状态确实被 policy 使用”。后续实验应遵循三个原则。其一，所有 action 结果必须同时报告自然分布 accuracy、macro recall、逐类 recall、instruction shuffle/empty sensitivity 和闭环 SR/SPL，避免多数类 accuracy 伪进展。其二，3D supervision 应优先监督 current-clean hidden 与 Register 的可读性，并通过 source/layer probe 判断空间信息出现在哪一层，而不是只看总 pose loss。其三，视频生成仍是主要的 dense dynamics supervision，但 policy inference 必须严格关闭 future visual branch，并在相同 checkpoint 上测量其准确率、延迟和闭环行为。

## 6 后续实验计划

### 6.1 长程记忆验收

在同一 current/future 样本上比较 `full Register / local-only / zero Register / shuffled history / truncated history`，历史长度覆盖 IW1/IW4/IW8/IW16。主要指标包括 velocity MSE、full-denoise PSNR/LPIPS、跨 chunk identity/geometry consistency，以及 history 增长时的边际收益。只有 full Register 随有效历史增加稳定优于对照，才能声称获得长程视频记忆。【未验证】

### 6.2 3D 表征验收

冻结相同 Stage1 checkpoint，在不同 Wan layers、current tokens 与 Register tokens 上训练轻量 probe，比较 translation、rotation、FOV，后续增加 depth/point。随后进行 visual-only 与 visual+3D 的同 seed、同数据、同预算对照，并测试历史打乱后 pose 的变化。目标不是获得最低训练 pose loss，而是证明 Register 提供 local observation 之外的空间信息。【未验证】

### 6.3 导航验收

使用完整 R2R train 与 held-out `val_seen/val_unseen`，以 StreamVLN 复现结果作为环境 sanity baseline。训练时保持 Stage1/2 replay，动作侧比较 single-action、短 action chunk、class-balanced loss 与 trajectory-aware sampler；评测必须包含多数类基线、instruction sensitivity、open-loop 和至少 100-episode closed-loop SR/SPL。若共享 Wan policy 仍不能超过反应式小模型，则应将 Register 作为 scene-state adapter 接入成熟 VLN policy，而不是继续假设 video DiT 的最后层天然适合控制。【未验证】

### 6.4 论文级对照与消融

最终章节需要补齐以下矩阵：

| 研究因素 | 必要对照 | 目标问题 |
| --- | --- | --- |
| History memory | HPMC、Register、local-only、zero/shuffle | 固定预算 Register 是否真的利用历史 |
| Register 位置 | main latent prefix、condition | 历史状态应进入主流还是条件流 |
| 3D loss | visual-only、visual+pose、visual+pose+depth | 空间监督是否改善可读性与导航 |
| Policy branch | V1 shared、GigaNav、成熟 VLN baseline | shared video backbone 的收益与代价 |
| Replay ratio | 0、低、中、高 | Stage3 能力保持与策略学习的平衡 |
| Evaluation | teacher-forced、autoregressive、open-loop、closed-loop | 训练 proxy 是否能转化为实际能力 |

## 7 本章小结

本章完成了 3D 流式导航世界模型从概念到完整训练链路的初步验证。研究以固定预算 Register 解决长历史 token 增长，以 `T_latent=4` 微块对齐视频与导航时间尺度，以单一 WanBlock 承载 video、pose 与 policy token，并通过三阶段训练依次追求预测性、空间可读性和可行动性。

实验已经表明：Register 可以接入视频生成并在部分指标上接近或超过 HPMC；短 future 与 IW16 等价长历史可以共同训练；pose supervision 能与 visual objective 共存；Stage3 replay 能在短期内保持生成和 pose 能力。【已验证→`result/vbench/20260727_threeway_vbench_2chunk_autoreg_stats20/metrics/report.md`；`result/stage2_gate/iw_aligned_gate300_20260817_052230/report_final.json`；`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`】与此同时，现有证据尚不能证明 Register 具备真正的长程 3D scene memory，正式 V1 policy 和 GigaNav 也尚未形成有效闭环导航。【已验证→`result/giga_nav/step5000_r2r_train_rolling13_full20_20260905/summary.json`】

因此，下一阶段最重要的研究任务不是扩大模型或继续堆叠分支，而是用严格对照回答三个问题：模型是否实际使用长历史，3D 信息是否存在于可供 policy 读取的状态中，以及这种状态是否能在 held-out closed-loop VLN 中带来稳定收益。只有这三项同时成立，才能把当前“可训练的多任务视频 backbone”推进为真正的 3D 流式导航世界模型。

## 证据索引与写作说明

本章数字以 2026-11-22 前已经落盘的代码、日志和结果为准。主要事实来源为：

- 研究设计：`doc/00_overview/research_idea_design.md`
- 最终接口与训练标准：`doc/00_overview/final_formal_training_standard.md`
- 模型演进：`doc/01_model/model_evolution_and_current_architecture.md`
- 数据统计：`doc/02_data/data_preparation_schema_and_status.md`
- 训练记录：`doc/03_training/training_plan_and_experiment_log.md`
- 评测与复现：`doc/04_evaluation/evaluation_reproduction_and_benchmarks.md`
- Stage2/Stage3 最新对照：`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`

正文中的“已验证”只表示对应产物真实存在且支持该局部结论，不等价于在完整 benchmark、多随机种子或统计显著性层面完成最终验收；标为“未验证”的内容不得在学位论文定稿中改写为既成事实。
