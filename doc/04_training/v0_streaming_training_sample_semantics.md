# V0 Register 流式训练样本与在线状态语义

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-003` |
| 类型 | 训练语义规范（Training Semantics Specification） |
| 状态 | Historical Semantics / V0 only |
| 更新时间 | 2026-07-29 |
| 职责 | 定义单轮、多轮、teacher forcing、Register递归和训练/推理差异 |

## 当前采用的训练样本

本节描述 V0 Register 训练语义；当前 V1 主线以 `RegisterCell` 和
full-pipeline smoke 文档为准。

当前 SpatialVID Short 使用 random-prefix next-chunk teacher forcing。一个
样本是同一 episode 中连续的2或3个 chunk：

```text
2 chunks: [history C0]     → target C1
3 chunks: [history C0,C1]  → target C2
```

离线数据只提供干净 VAE latent \(Z_i\) 和目标 action \(A_i\)。Register 在
forward 内从 checkpoint 中的 learnable initial state 开始在线递归。

以上是现有 A/B 原型的实现事实，不是最终 Stage One 设计。下一次结构改造必须
替换为：

```python
register = extractor(chunks[0])
for history in chunks[1:-1]:
    register = updater(register, history)
```

即第一个 History Chunk 负责提取初始场景 Register，后续 History Chunk 才调用
Updater；不存在可学习的场景初始 Register。

## 两 Chunk 的完整前向

\[
R_1=\operatorname{Extractor}_\theta(Z_0)
\]

local memory 为 history 最后一个 latent frame：

\[
L_0=Z_0[:,:,-1:]
\]

只对目标 \(Z_1\) 采样 diffusion timestep 和噪声：

\[
X_{1,t}=\alpha_t Z_1+\sigma_t\epsilon
\]

模型预测：

\[
\hat v_1=D_\theta(X_{1,t},t,R_1,L_0,A_1,\mathrm{text})
\]

当前原型的 `detach_history=false` 使 loss 可反传至 Register 模块；改造后该
梯度应反传至 Extractor。VAE latent 是常量，不训练 VAE。

## 三 Chunk 与更多历史

三 chunk 样本：

\[
R_1=E(Z_0),\quad R_2=U(R_1,Z_1)
\]

\[
\hat v_2=D(X_{2,t},t,R_2,L_1,A_2,\mathrm{text})
\]

训练不先执行完整 diffusion sampling 来生成 \(C_1\)。历史直接使用真实干净
latent，这就是 teacher forcing。每个样本只有一次昂贵 DiT forward/backward，
历史长度只增加较轻的 Register updates。

当前代码对任意 \(N\)-chunk窗口采用：

```python
register = extractor(chunks[0])
for history in chunks[1:-1]:
    register = updater(register, history)
local = chunks[-2][:, :, -1:]
loss = diffusion_loss(chunks[-1], register, local, target_action)
```

## A/B 注入差异

### A / latent prefix

Register 为固定：

```text
[B,16,4,H,W]
```

每个空间位置的4个 Register planes 作为 query，读取 history chunk 经时间池化
得到的8个 token。DiT latent 侧为：

```text
[4 Register planes; 1 local plane; noisy target]
```

### B / DiT condition

history chunk 池化为空间/时间观测 token，16个 `[256]` Register token 作为
query 更新。随后投影为4096维并与 UMT5 文本拼接：

```text
latent side:    [1 local plane; noisy target]
condition side: [text tokens; Register tokens]
```

两者都保留 Infinite-World local memory，只替换 HPMC history。

## 训练与推理的差异

训练：

```text
真实 C0 → Register
真实 C1 → Register
...
Register + 最新真实 local frame → 预测下一真实 chunk 的 diffusion target
```

推理：

```text
生成 C0 → Register
生成 C1 → Register
...
Register + 最新生成 local frame → diffusion sampling 下一 chunk
```

训练时历史真实、推理时历史由模型生成，存在 autoregressive exposure bias。
当前先用 teacher forcing 保证稳定和效率；后续再评估：

- 少比例 scheduled sampling；
- 离线 rollout latent；
- 单步 predicted-\(x_0\) 更新；
- 对生成 history stop-gradient。

不得在没有成本评估时让训练为每个 history chunk 执行完整几十步 diffusion
sampling。

## Last-target 与 Dense-target

当前采用 last-target：

```text
C0,C1,C2,C3,C4 → C5
```

只对 \(C_5\) 计算 diffusion loss，但梯度可经过全部 Register updates。优点是
一次样本只运行一次 DiT。

未来可选 dense-target：

```text
C0 → C1
C0,C1 → C2
...
C0,...,C4 → C5
```

它仍可使用真实 history，不需要生成历史，但一个6-chunk序列需要5次 DiT
forward，成本约为 last-target 的5倍。Medium/Long 阶段更合理的折中是每个窗口
抽样短、中、长1–3个 target horizon。

## 当前 Short 训练

初始化：

```text
A: re10k-all-a-full-ebs4-from-infinite/full-final.pt
B: re10k-all-b-full-ebs4-from-infinite/full-final.pt
```

不是重新从原始 InfiniteWorld 开始，也不在 A/B 之间互相加载。

训练参数：

| 配置 | 值 |
| --- | --- |
| Dataset | SpatialVID |
| History | 1–2 chunks |
| Target | 最后1 chunk |
| Train scope | Full parameter |
| Optimizer | AdamW |
| LR | `1e-5` |
| Precision | bf16 |
| Micro batch | 1 |
| Gradient accumulation | 4 |
| Effective batch | 4 |
| Max steps | 20,000 |
| Plateau start | 2,000 steps |

配置见 `NAV/config/train_spatialvid_short.yaml`。

## Medium/Long 的推荐样本策略

- Medium 4–7 chunks：在线递归读取全部 history，随机监督1–2个 horizon；
- Long ≥8 chunks：优先使用 DL3DV 等长 episode，每条监督短、中、长 horizon；
- 长链可使用 truncated BPTT，但 detach 边界必须记录；
- 数据集先等权或显式加权，再在数据集内部 shuffle，避免 SpatialVID 约95%的
  帧数占比淹没其他域；
- Validation 必须按 scene/episode 隔离，不能让同一视频不同窗口跨 split。

## 必须记录的实验字段

每次训练日志和 checkpoint 至少包含：

```text
source checkpoint
dataset version / manifest
window policy
min/max chunks
target policy
teacher forcing / scheduled sampling ratio
detach policy
variant A/B
batch size / accumulation
optimizer / LR
steps / plateau rule
```

数据侧定义见 `../03_data/training_data_construction.md` 和
`../03_data/action_annotation_specification.md`。
