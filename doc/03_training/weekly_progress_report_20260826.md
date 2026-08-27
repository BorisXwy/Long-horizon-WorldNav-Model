# NAV 周进展：从 Stage2 表征验证到 R2R 单步策略训练

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-012` |
| 类型 | 周进展汇报 |
| 状态 | Completed / Weekly Snapshot |
| 更新时间 | 2026-08-27（Asia/Shanghai） |
| 统计周期 | 2026-08-19 至 2026-08-26 |
| 职责 | 归纳本周 Stage2、R2R 数据与 Stage3 policy 主线进展，不替代训练和评测事实总账 |

## 本周结论

本周工作从“完整的 shared Wan/DiT 三分支结构已经跑通”推进到了两个更具体的
验证阶段：一是确认 Stage2 能在保持视频生成能力的同时，从中间 hidden state
读取相机位姿；二是完成 R2R 数据闭环并开始训练真正依赖 observation、Register
和 instruction 的离散导航策略。

最重要的新增认识是：此前 `H=10、144-class` Stage3 的 loss 下降主要受动作
分布捷径影响，并不能证明模型学会了导航。当前已将 policy 目标收敛为单步四分类，
并使用严格均衡采样，使训练 loss 必须依靠条件信息才能低于随机基线。这一改动让
任务明显变难，但也让后续指标更可信。

## Stage2：生成与 3D 表征验证

本周完成了 Wan DiT 第 `4/8/12/16/20/24/29` 层的 frozen probe sweep，并使用
VGGT-style dense camera-query head 从 current observation hidden 中读取逐帧
pose。综合 probe MSE 和轨迹可视化，当前正式 readout 采用第 16 层；结果说明
中间层包含可读的相机运动信息，但现有 pose 精度仍属于初步可用，而不是高精度
3D reconstruction。【已验证→`result/v1_stage2_pose_visualization/step2000_attached_vs_probe_l16_20260821_trajectory_check/pose_visualization_summary.json`】

Stage2 主线随后持续续训，并在 step3400 进行了同协议评测。64 个 RE10K pose
窗口上的 translation MAE 为 `0.04699`、rotation error 为 `7.74°`；4 个混合数据
窗口的 30-step full denoise 平均 PSNR 为 `21.32 dB`，生成视频均可正常播放。
这证明当前模型能够同时维持 video generation 与 pose supervision，但几何精度
和完整生成质量都仍有提升空间。【已验证→`result/v1_stage2_stage3_comparison/paired_step3400_vs_step400_seed20260827/report.md`】

## R2R 数据闭环

R2R train 的 `10,819` 个 episode 已完成 observation 渲染、`T_latent=4` latent
编码和 instruction embedding 缓存。训练 latent 与 text cache 分别存入
`NAV/data/train/r2r_ce/`，原始数据和渲染中间产物保留在公共 sharedata，避免继续
挤占公共派生数据目录。

本周同时修正了 Stage3 样本语义：policy target 必须是 `Z_obs` 之后的未来动作；
terminal `STOP` 按吸收态补齐，但每个 episode 最多采一个 terminal window，避免
padding 被重复放大成 STOP-heavy 数据集。`H=10` 口径可构造 `69,648` 个窗口；
切换到当前 `H=1` 单步控制后，可用自然 action anchor 为 `88,602` 条。当前训练中
instruction cache 命中正常，无 empty-text fallback。

## Stage3：从分布捷径转向真实条件学习

第一版 Stage3 使用 `H=10`、144 个 translation-rotation combo 类别，并从 Wan
最后一层 action-position hidden 读取离散 logits。其开环评测 accuracy 为
`58.75%`，但低于多数类基线 `60.08%`；左右转 recall 均为 `0`，打乱 instruction
只造成约 `1%` 的预测变化。因此该版本主要学到了 `MOVE_FORWARD` 先验，没有形成
有效的 instruction-conditioned policy。【已验证→`result/v1_stage2_stage3_comparison/paired_step3400_vs_step400_seed20260827/report.md`】

基于这一结果，当前 Stage3 改为更符合闭环导航的单步四分类：

```text
Wan final action-position hidden [B,1,1536]
  -> FP32 LayerNorm
  -> Linear(1536,512) + GELU
  -> Linear(512,4)
  -> STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
```

每个 effective batch 固定四类各 4 条，随机均衡基线为
`CE=ln(4)=1.3863`、accuracy/macro recall=`25%`。截至本周快照，最近
step251–287 的训练均值为 CE `1.259`、accuracy/macro recall `40.0%`；但
STOP recall 为 `78.4%`，MOVE recall 仅 `11.5%`。这表明模型已经获得少量条件
信号，同时存在明显的 STOP 过预测；该结论仅是训练内观察，尚未通过 held-out
R2R open-loop 或闭环导航验证。【未验证：当前 run 尚未到计划评测节点】

Stage3 仍保留 `0.25 * L_visual + 0.05 * L_pose` replay。已有配对评测显示，早期
Stage3 checkpoint 基本保住 pose 能力，但生成 PSNR 相比当时的 Stage2 最佳版本
下降约 `1.23 dB`；由于两者并非从同一 checkpoint 同步分叉，该结果只能说明存在
退化风险，不能作为严格的 catastrophic forgetting 归因。【已验证→`result/v1_stage2_stage3_comparison/paired_step3400_vs_step400_seed20260827/report.md`】

## 当前状态与下周计划

截至 2026-08-26 下午，两张 GPU 分别运行：

- Stage2：从 step3400 以 `lr=2e-6` 续训，当前约 step3962，目标 step4400；
  TensorBoard 端口 `6037`。
- Stage3：从 Stage2 step3400 初始化，进行 2000-step 单动作均衡训练，当前约
  step287；TensorBoard 端口 `6039`。

下周优先完成三件事：

1. 对 Stage2 新 checkpoint 复用相同的 full-denoise 与 RE10K pose 协议，确认
   继续训练是否真实改善生成和几何，而不是只降低 train loss。
2. 在 Stage3 step400/800 节点进行严格均衡的 held-out open-loop evaluation，
   同时报告 macro recall、四类 recall、instruction/observation ablation，禁止只看 CE。
3. 若 STOP 高召回、MOVE 低召回持续存在，优先检查 terminal cue、window 对齐和
   条件利用情况，再决定是否调整 loss 或模型结构。

## 2026-08-27 checkpoint 评测补充

Stage2 已训练至 step4400。最新 checkpoint 在 4 个混合样本的 30-step full
denoise 上得到平均 PSNR `23.16 dB`，在 64 个 RE10K pose 窗口上得到 pose MSE
`0.002746`、translation MAE `0.03557`、rotation error `6.31°`。与 step3400
进行同样本、同噪声对照后，full-denoise PSNR 仅提升 `0.039 dB`，因此当前结论是
继续训练保持稳定并略有改善，而不是显著提升。

Stage3 最新已保存的 step800 在四类严格均衡 R2R teacher-forced open-loop
评测上得到 accuracy/macro recall `39.84%`，高于 `25%` 随机基线；但在自然
R2R 分布上 accuracy 仅 `15.63%`，远低于 `75.78%` 的多数类基线。均衡集的
MOVE_FORWARD recall 只有 `6.25%`，且打乱 instruction 只改变 `3.91%` 的
top-1 预测，说明 policy 尚不可用，主要问题已从“是否超过随机”收敛为动作类别
偏置和条件利用不足。

将 Stage3 step800 与其来源 Stage2 step3400 做配对 replay 后，30-step 生成 PSNR
只变化 `-0.013 dB`，visual/latent one-step 指标也基本不变；pose MSE 反而下降
约 `13%`。因此本轮没有观察到生成或 3D catastrophic forgetting。完整协议、
混淆矩阵、视频路径与结论边界见
`result/v1_stage2_stage3_comparison/latest_step4400_step800_20260827/report.md`。
