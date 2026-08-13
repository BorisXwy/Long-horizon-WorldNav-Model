# V0 Stage Two：视频生成前向中的 3D Probe 与监督规范

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-004` |
| 类型 | 训练规范（Training Specification） |
| 状态 | Proposed / Confirmed Semantics / Not Implemented |
| 更新时间 | 2026-07-30 |
| 职责 | 定义完整视频生成前向中的 3D Probe、选层、监督位置与联合损失 |

## 结论

Stage Two 完全保持视频生成训练方式，不把模型改造成独立 3D encoder，也不引入
Navigation Policy。训练仍然包含：

```text
真实 History Chunks
    → Extractor-first / Updater-later Register
    → Register + Local Memory + Action/Text + Noisy Target
    → Wan DiT
    → Diffusion/RFlow loss
```

在这条完整生成前向上额外执行：

```text
多个候选 DiT block 的 Register-after-DiT
    → Frozen diagnostic probe
    → 选择3D最可读的层 l*
    → 3D Teacher/GT supervision
```

Stage Two 的交付物是经过 3D 监督的视频生成模型，以及可供 Stage Three 导航
读取的 `Register-after-DiT @ selected layer` 接口。

## 前置条件：Stage One 必须先改造

Stage Two 不再使用 learnable initial Register。一个 episode 的在线状态为：

\[
R_1=\operatorname{Extractor}(Z_0)
\]

\[
R_{i+1}=\operatorname{Updater}(R_i,Z_i),\quad i\ge1
\]

即：

- 第一个 History Chunk 负责提取 Register；
- 后续 History Chunk 才更新 Register；
- checkpoint 保存 Extractor/Updater 的能力参数；
- checkpoint 不保存 episode-specific Register。

当前旧 A/B checkpoint 仍包含 learnable initial Register，只能作为权重初始化
来源；进入 Stage Two 前必须完成结构迁移与 smoke test。

## 多 Chunk 视频生成样本

从同一 episode 采样：

```text
History: C0, C1, ..., Ck
Generation Target: Ck+1
3D Teacher/GT: C0, C1, ..., Ck 的真实历史几何标签
```

3D GT 只来自已经观察到的 History。未来生成 Target 即使训练数据中存在真实帧，
也不作为主 3D Teacher 对象；它仍只承担原视频生成的 Diffusion/RFlow target。
这样既保持完整视频生成训练，又确保被监督的是用于压缩历史的 Register。

历史使用真实干净 latent：

\[
Z_i=\operatorname{VAE}(C_i)
\]

按 Stage One 规则建立和更新 Register：

\[
R_1=E(Z_0),\qquad
R_{i+1}=U(R_i,Z_i)
\]

目标仍按原 RFlow/Diffusion 方式加噪：

\[
X_{k+1,t}=\alpha_t Z_{k+1}+\sigma_t\epsilon
\]

完整生成预测为：

\[
\hat v_{k+1}
=D_\theta(X_{k+1,t},t,R_{k+1},L_k,A_{k+1},\mathrm{text})
\]

其中 \(L_k\) 是 Infinite-World latest local memory。不能为了做 3D 监督绕过
noisy target、Local、Action/Text 或 DiT 生成 blocks。

## 主要 Probe 对象

Register 在注入 DiT 前只是 history memory。Stage Two 需要定位的是它参与视频
生成计算后形成的 contextualized representation：

\[
\widetilde R_{k+1}^{\,l}
=
\operatorname{SelectRegister}
\left[
D_\theta^{1:l}
(X_{k+1,t},L_k,R_{k+1},A_{k+1},\mathrm{text})
\right]
\]

对多个候选层 \(l\) 读取 \(\widetilde R^l\)，训练轻量 probe，判断哪一层最适合
3D 监督和 Stage Three 导航。

### A 的读取

A 的 Register 是 latent prefix：

```text
[Register prefix; Local Memory; Noisy Target]
```

每个候选 block 后按固定 token range 切出 Register prefix hidden。必须同时保存
grid size 和 prefix offset，禁止把 Local/Target token 混入 Register probe。

### B 的读取

B 的 Register 与 UMT5 text 拼接为 condition：

```text
[Text condition; Register condition]
```

必须确认选定 block 后能读取“经过 DiT 更新”的 Register。如果 B 的
cross-attention 仅把 condition 当作静态 K/V，而不回写 condition hidden，则
当前实现不存在严格意义的 `Register-after-DiT`。此时必须增加可回写的
Register stream、双向 cross-attention 或 block-level Register updater，再做
与 A 同口径的 probe。

不能把进入 DiT 前的静态 condition embedding 误称为 Stage Two 导航表示。

## Probe 实验

### 冻结范围

初始 probe 阶段冻结：

- Wan VAE；
- Wan DiT；
- Extractor；
- Updater；
- Action/Text encoders。

只训练小型 Linear/MLP/Transformer readout，避免 probe 本身重写 backbone。

### 候选层

从浅、中、深层均匀选择若干 DiT blocks。实际层号必须读取当前 Wan1.3B 代码后
写入 config，不在文档中凭经验硬编码。

每个候选层至少评估：

- Relative Camera Pose；
- Camera Trajectory；
- Coarse Layout / Occupancy；
- VGGT scene/register feature alignment；
- 跨 Chunk geometry retention；
- 3D probe 指标随 history 长度的变化。

可额外 probe target/history spatial tokens 的 Depth、Pointmap 或 patch feature，
用来判断 Register 的3D信息是否来自 DiT 的时空建模，但 Stage Three 的默认接口
仍是 `Register-after-DiT`。

### 选层标准

\[
l^\*=\arg\max_l
\operatorname{Score}
(\text{3D accuracy},\text{history retention},\text{cost},\text{stability})
\]

选择不能只看单个 depth 或 pose 数字，还要考虑：

- 多 Chunk 增长时是否稳定；
- A/B 是否使用同一公平 protocol；
- 导出该层是否增加额外 DiT 计算；
- 表示尺寸是否适合固定预算导航输入；
- 不同 dataset/scene split 上是否泛化。

## 3D Teacher 与 GT

Frozen VGGT/VGGT-Ω 读取相同样本的干净 RGB，提供：

- Camera Pose / Relative Motion；
- Scene/Register Feature；
- Patch Geometry Feature；
- Depth、Confidence；
- 可用时的 Pointmap / Matching。

已有真实 Pose/Depth 时优先使用 GT。Pseudo-label 必须保留 confidence、
visibility 和 validity mask。

Teacher 只提供 loss target，不作为 DiT condition，避免先验泄漏。

## 时间与空间对齐

每个81-frame Chunk 对应21个 Wan VAE latent time steps。逐点 Teacher target
需要：

1. 将81帧 Teacher 输出划分为21个 temporal bins；
2. 在 bin 内按 confidence 聚合；
3. 将空间网格 resize/project 到 DiT token grid；
4. 保存 source frame indices、bin、grid、mask 和 teacher version。

禁止用未经验证的单帧 `frame_id=4*latent_id` 直接硬对齐。Pose/trajectory head
必须明确预测21-step latent pose、81-frame pose，或 Chunk-level transform，
三者不能混用。

Register-level scene/trajectory supervision不要求与每个 patch逐点对应，但必须
明确参考坐标系和时间覆盖范围。

## 正式联合 3D 监督

Probe 选出 \(l^\*\) 后，在完整视频生成 forward 中加入：

\[
\mathcal L_{\mathrm{reg3D}}
=
\lambda_f\mathcal L_{\mathrm{teacher}}
(P(\widetilde R^{l^\*}),G)
+\lambda_{\mathrm{pose}}\mathcal L_{\mathrm{trajectory}}
+\lambda_{\mathrm{map}}\mathcal L_{\mathrm{layout}}
+\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{history}}
\]

可选 spatial auxiliary loss：

\[
\mathcal L_{\mathrm{spatial}}
=
\lambda_d\mathcal L_{\mathrm{depth}}
+\lambda_p\mathcal L_{\mathrm{point}}
+\lambda_{\mathrm{patch}}\mathcal L_{\mathrm{patch\ feature}}
\]

视频生成主体 loss：

\[
\mathcal L_{\mathrm{wm}}=\mathcal L_{\mathrm{RFlow/Diffusion}}
\]

总损失：

\[
\mathcal L
=
\lambda_{\mathrm{wm}}\mathcal L_{\mathrm{wm}}
+\lambda_{\mathrm{reg3D}}\mathcal L_{\mathrm{reg3D}}
+\lambda_{\mathrm{spatial}}\mathcal L_{\mathrm{spatial}}
\]

\(\lambda_{\mathrm{wm}}\) 必须大于0，保证 Stage Two 仍然是视频生成训练。各 loss
权重通过 gradient norm 和独立 validation 指标校准，不能只按 loss 数值大小
相加。

## 梯度路径

必须验证：

\[
\mathcal L_{\mathrm{reg3D}}
\rightarrow \widetilde R^{l^\*}
\rightarrow \mathrm{DiT}^{1:l^\*}
\rightarrow R
\rightarrow \mathrm{Extractor/Updater}
\]

以及：

\[
\mathcal L_{\mathrm{wm}}
\rightarrow \mathrm{DiT}
\rightarrow \mathrm{Register\ injection}
\]

日志至少记录：

- selected block Register gradient norm；
- Extractor gradient norm；
- Updater/Cross-Attention Q/K/V gradient norm；
- 3D projector/head gradient norm；
- Diffusion、Register 3D、Spatial auxiliary 各 loss；
- Teacher valid/confident ratio；
- 各候选层 probe 指标；
- 生成质量与3D指标的变化。

如使用 truncated BPTT，必须记录 detach 的 Chunk 边界。

## 分阶段执行

### Phase 0：Stage One 结构迁移

- 实现 Extractor-first / Updater-later；
- 从旧 A/B checkpoint 迁移可复用权重；
- 验证 episode reset 无场景状态泄漏；
- 验证2、3及更多 Chunk 的梯度路径。

### Phase 1：Frozen layer probe

- 完整运行视频生成 forward；
- 缓存多个 block 的 Register-after-DiT；
- 冻结 backbone，只训练 probe；
- 在独立 scene split 上选择 \(l^\*\)。

### Phase 2：3D head warm-up

- 固定 \(l^\*\)；
- 训练3D projectors/readout；
- 可先冻结大部分 DiT；
- 始终保留 Diffusion/RFlow loss。

### Phase 3：Full-parameter 3D-aware generation

- 解冻 DiT、Extractor 和 Updater；
- 延续 Short/Medium/Long history curriculum；
- 联合优化 diffusion 与 3D losses；
- 监控生成质量—3D能力 Pareto trade-off。

到此 Stage Two 结束。Navigation Head/Policy 的训练属于 Stage Three。

## 必须包含的对照

| 对照 | 目的 |
| --- | --- |
| Stage One，无3D loss | 判断3D监督收益 |
| 各候选 DiT layer frozen probe | 选择 \(l^\*\) |
| Register-before-DiT vs Register-after-DiT | 验证DiT contextualization价值 |
| Register probe vs spatial-token probe | 确认3D信息分布 |
| 只用 Register 3D loss vs 加 spatial auxiliary | 判断辅助稠密监督价值 |
| A vs B | 比较 latent-prefix 与 condition Register |
| GT Pose vs VGGT pseudo-label | 判断Teacher噪声影响 |

## Stage Two 验收标准

1. Stage One 已完成 Extractor-first / Updater-later 改造；
2. A/B 均有明确、可复现的 Register-after-DiT 读取规则；
3. Frozen probe 找到显著优于随机/注入前基线的3D层；
4. 选定层在不同 history length 和 scene split 上稳定；
5. 3D loss 梯度能到达 DiT、Extractor 与 Updater；
6. Diffusion/RFlow loss 始终参与训练；
7. 3D联合训练后的生成质量退化处于预设范围；
8. 输出 selected layer、tensor shape、normalization、projector与checkpoint。

总体网络边界见
`../01_design/v0_stage_two_3d_supervision.md`；Stage Three 导航接口见
`../01_design/v0_stage_three_navigation.md`。
