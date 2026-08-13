# V0 Stage Three：基于 3D-aware DiT Register 的导航设计

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DES-003` |
| 类型 | 设计规范（Design Specification） |
| 状态 | Proposed / Confirmed Boundary / Not Implemented |
| 更新时间 | 2026-08-09 |
| 职责 | 定义 Stage Three 如何复用 Stage Two 选定的 DiT Register 表示执行导航 |

## 结论

Stage Three 才引入导航任务。导航特征不是 Updater 原始输出，也不是单独绕开
生成骨干的 VAE latent，而是：

> Register 注入视频 DiT 后，在 Stage Two probe 确认具有最佳 3D 可读性的层上，
> 取出的 `Register-after-DiT` representation。

Stage Three 必须保留 Stage One/Two 的 shared backbone 关系：导航策略读取的
不是新建 perception backbone 的输出，而是视频生成 DiT/Register 在 Stage Two
3D 监督后形成的 compact representation。Stage Three 可以使用 policy-only
推理接口来减少 token 数和延迟，但该接口应是 shared backbone 的裁剪路径
（pruned/action-only path），而不是另起炉灶。

同时，Stage Three 采用 action-centered navigation 原则：

```text
Policy input:
 真实 History Register + 当前 Observation + Instruction/Goal + Nav State

Policy output:
 下一步动作 / action chunk / waypoint / discrete navigation action

Training-only auxiliary:
 future visual / future 3D / rollout consequence
```

Future visual / future 3D 只作为训练时的 consequence supervision 或 consistency
regularizer，不作为 policy condition。推理时默认删除 future branch，只保留
真实观测派生的 Register/Observation tokens 和 action/query tokens。

### 暂定可行工作设想：控制频率与 Register 更新（2026-08-08）【未验证】

在「生成 + Register → NavPolicy」改造上，当前相对可行、先记录备查的设想
（**非**已冻结接口；决策条目 `DEC-022`，insight 寄存器 `R17`）：

1. **训练侧（适配 NavPolicy）**：Register 更新频率尽量贴近视频预训练的
   chunk 尺度——一次更新对应的**空间变化尺度**接近 1 个 video chunk
   （室内约 1–4 次 Habitat `forward` 量级），但观测侧往往只有**个位数
   RGB frame**（例如 ~4 步）。即：用短开环观测窗触发一次
   Extractor/Updater，而不是假定凑满 81 帧；81 帧 VAE packing 若仍沿用，
   需另定 pad/插值协议（见 NAV-INS-001 / 视频–Habitat 尺度对照讨论）。
2. **推理侧**：
   - 若前向够快 → 允许**单步**（每 Habitat step 决策一次），Register
     可按同频或降频更新；
   - 若偏慢 → **决策频率与 Register 更新同步**（一次推理 / 一次短
     chunk 执行 / 一次 Register 更新），形成半闭环。
3. **明确不混为一谈**：控制 chunk（个位数 step）≠ Wan 固定 81 帧
   latent chunk；前者是策略更新节奏，后者是骨干张量形状约束。

此设想与 DEC-014（History 仅真实 Observation）兼容；具体 packing、
head 结构与训练损失尚未冻结。

因此核心接口为：

\[
F_t^{nav}
=
\operatorname{ReadRegister}
\left(
\operatorname{DiT}^{l^\*}
(X_t,L_t,R_t,A_t,\mathrm{text})
\right)
\]

其中 \(l^\*\) 由 Stage Two probe 和 3D 验证指标确定，不预先拍脑袋固定。

## 与前两阶段的关系

```text
Stage One
视频生成 + Extractor-first + Updater-later Register
        ↓
Stage Two
保持完整视频生成前向 + 多层3D Probe + 选定层3D监督
        ↓
Stage Three
读取选定层 Register-after-DiT + Navigation Head/Policy
```

Stage Three 继承：

- Stage One 的视频生成网络与在线 Register 更新；
- Stage Two 的选定 DiT layer；
- A/B 对应的 Register token slicing；
- Stage Two 的 3D projector/normalization（若 probe 证明需要）；
- 3D-aware checkpoint。

### Policy-only 小输入接口

Stage Three 不能直接复用完整视频生成张量作为策略输入，例如高分辨率
`[B,C,T,H,W]` VAE latent grid 或完整 future video tokens。完整 latent grid
属于 Stage One/Two 的 VideoGen/3D 训练路径；policy-only 推理必须控制 token
预算，优先使用：

- fixed-size Register tokens；
- 当前真实 Observation 的少量 visual prefix tokens；
- Instruction/Goal text tokens；
- action/query tokens；
- 可选 Nav state / pose / map summary tokens。

这一路径应借鉴 Fast-WAM / GigaWorld-Policy 的裁剪思想：训练时 future branch
提供 dense supervision，推理时 action/navigation branch 只读取 current/past
prefix，不生成也不读取 future visual tokens。

## 导航输入

默认导航策略读取：

\[
\pi(a_t\mid F_t^{nav},\mathrm{instruction},\mathrm{optional\ local})
\]

其中：

- \(F_t^{nav}\)：选定层的 `Register-after-DiT`；
- `instruction`：语言导航指令；
- `optional local`：当前局部视觉特征，仅作为后续消融，不改变 Register 主接口。

是否额外读取 spatial target hidden，应作为 Stage Three 消融，而不是 Stage Two
预先规定的主方案。

推荐 attention mask：

```text
Nav/action tokens can attend to:
  Register / Memory
  Current Observation prefix
  Instruction / Goal
  Nav state

Nav/action tokens must NOT attend to:
  Future Visual tokens
  Future 3D tokens
  Generated future hidden

Future auxiliary tokens can attend to:
  Register / Memory
  Current Observation prefix
  Instruction / Goal
  Nav/action tokens
```

## History 来源

导航训练和推理中的 History 只来自真实 Observation：

```text
第一个真实 Observation Chunk → Extractor → R1
后续真实 Observation Chunk   → Updater   → Rt
Rt 注入 DiT并经过选定层       → Ft_nav
```

模型生成的视频可以作为规划想象的可选分支，但不能写回真实 History Register，
也不能替代 observation-derived \(R_t\)。

## 训练范围

Stage Three 逐步比较：

1. 冻结 Stage Two backbone，只训练 Navigation Head；
2. 冻结大部分 DiT，微调 selected layer、Register Updater 和 Navigation Head；
3. 低学习率端到端微调，并保留 Stage Two 3D loss 防止几何能力遗忘；
4. 可选加入 diffusion/world-model auxiliary loss 防止生成能力退化。

导航目标、数据格式、具体 policy 架构和 benchmark 尚未冻结，后续单独形成
`NAV-TRN-005` 训练规范。

## 必须验证

- 读取的确实是经过 DiT block 更新后的 Register，而非注入前 embedding；
- Stage Two 的 probe 最优层在导航 validation 上具有可迁移性；
- 只用真实 Observation History 可以完成完整导航前向；
- episode reset 后 Register 不携带上一场景状态；
- 不同 history 长度下表示维度固定；
- A/B 使用各自正确的 token slicing；
- 导航梯度是否回传 DiT 必须由实验配置显式记录。
