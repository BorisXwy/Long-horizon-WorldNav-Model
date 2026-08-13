# V0 Register 流式世界模型：RE10K 数据与训练记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-TRN-002` |
| 类型 | 实验记录（Experiment Record） |
| 状态 | Historical Baseline |
| 更新时间 | 2026-07-28 |
| 职责 | 保存 RE10K 上 A/B Register 方案的配置、结果和修订历史 |

## 1. 数据选择

Infinite-World 论文使用两类未公开数据：

- 超过 30 小时的网页第一人称与探索视频，用于预训练；
- 作者用 iPhone 17 Pro 自采的 30 分钟 Revisit-Dense Dataset（RDD），用于长程
  回访记忆微调。

截至 2026-07-24，官方仓库和 Hugging Face 页面均未提供这两个训练集或训练
脚本，因此无法按文件级别复现原数据。按照本项目约定，NAV 改用公共目录已有的
RealEstate10K（RE10K）：

```text
/sharedata/RealEstate10K
```

RE10K 当前本地数据包含 197 个有效帧场景、28,219 张匹配图像和逐帧 3×4
相机外参。NAV 派生了 89 个长度为 162 帧的两 chunk 训练窗口，覆盖 57 个
场景：

```text
/sharedata/RealEstate10K/nav_register/manifests/train.jsonl
```

RE10K 没有键盘动作。`prepare_re10k_manifest.py` 先用 SVD 将旋转矩阵投影到
最近的 SO(3)，再从相邻相机中心与相对旋转计算平移、yaw 和 pitch。它按场景
中位运动尺度归一化，映射到 Infinite-World 的 10 类 `move` 和 10 类 `view`；
极小变化标为 `no-op`，异常大变化标为 `uncertain`。这是伪标签
（pseudo-label），不是 RE10K 原生标注。

## 2. 网络改造

HPMC 的 `latent_encoder.*` 共 68 个 checkpoint keys 被移除。其余
Infinite-World 权重保持不变；审计结果为 905/905 keys 完全匹配、
`missing=0`、`unexpected=0`。

每个已生成或 teacher-forcing 的 chunk latent 先自适应池化成 4×4×4 个
观测 token，再由 16 个持久 Register token 作为 query，通过两层
cross-attention 读取 chunk 信息：

```text
R(k+1) = CrossAttention(query=R(k), key/value=Chunk(k))
```

两个方案共用同一更新器：

- A / `latent_prefix`：维护固定 `[B,16,4,H,W]` 的 latent-like Register。
  `T=4` 只表示4个 Register planes，它不是 VAE 视频 latent。每个空间位置
  的 Register 通过 cross-attention 读取 chunk 在同一空间位置的时间序列，
  随后沿时间轴放在一个 local-memory plane 前；
- B / `dit_condition`：将 Register 投影成 4096 维条件 token，追加到 UMT5
  context，使每个 DiT block 通过原有 cross-attention 读取长期状态。

当前训练冻结 Wan2.1/Infinite-World 1.3B backbone，只更新 Register 更新器和
注入投影。这不是全参数微调；目的是先稳定新记忆通路，并控制显存和灾难性遗忘。

## 3. 数据缓存

Wan2.1 VAE 已将 10 个真实 RE10K 样本编码为两段 latent，每段形状为
`[1,16,21,56,112]`，对应 81×448×896 RGB 帧：

```text
/sharedata/RealEstate10K/nav_register/latents
```

扩展缓存：

```bash
cd /mnt/pool1/sharehome/xiewenyuan
PYTHONPATH=academic/3d_wm_vln/Infinite-World \
CUDA_VISIBLE_DEVICES=0 \
academic/3d_wm_vln/virtual_env/.venv_infinite_world/bin/python \
academic/3d_wm_vln/NAV/scripts/cache_re10k_latents.py --max-samples 89
```

## 4. 已验证结果

两个方案都完成了真实 RE10K 上的端到端单步：

| 方案 | 目标函数 | Step 1 loss | Register grad norm | NAV 进程峰值显存 |
| --- | --- | ---: | ---: | ---: |
| A latent prefix | Infinite-World RFlow diffusion loss | 0.101113 | 0.171875 | 28.748 GiB |
| B DiT condition | Infinite-World RFlow diffusion loss | 0.389421 | 15.6875 | 13.866 GiB |

loss 不可直接横向判断方案优劣：两次随机 timestep 相同，但注入位置、可训练投影
和有效条件长度不同。这里只用于确认完整 forward/backward、梯度和显存。

正式实验各 1000 steps，10 个 cache 轮换，每 100 steps 保存一次：

| 方案 | GPU | tmux 会话 | 日志目录 |
| --- | ---: | --- | --- |
| A | 0 | `nav_reg_a` | `NAV/log/re10k-latent-prefix-formal-v2` |
| B | 1 | `nav_reg_b` | `NAV/log/re10k-dit-condition-formal-v2` |

训练采用论文相同的 448×896（480P bucket）、AdamW、常数学习率 `1e-5`、
bf16 和 RFlow 配置；当前 batch size 为 1，冻结 backbone，这两项不同于论文
的 16×H800 全参数训练。

查看状态：

```bash
tmux attach -t nav_reg_a
tmux attach -t nav_reg_b
tail -f NAV/log/re10k-latent-prefix-formal-v2/train.log
tail -f NAV/log/re10k-dit-condition-formal-v2/train.log
```

TensorBoard：

```bash
tensorboard --logdir \
  /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log \
  --port 6007
```

## 5. 当前限制

- RE10K 多为房产漫游，域分布比 Infinite-World 的开放域数据窄；
- 动作来自相机位姿伪标签，尺度不确定性和运动模糊会产生噪声；
- 目前只缓存 10/89 个窗口，正式比较前应缓存全部窗口并建立 scene-level
  train/validation split；
- 本历史实验的 Register 从 learnable initial state 开始，在两个 chunk 间
  递归；该结构已被新的 Stage One“首Chunk Extractor、后续Updater”设计取代，
  但旧checkpoint仍按原结构保存。更长的4/8/16 chunk curriculum需要同一场景
  的连续窗口串接；
- 空文本条件用于避免额外加载 11 GB UMT5。后续可对 RE10K 生成 caption cache，
  再恢复文本条件。

## 6. Register 阶段完成结果与全参阶段

两个 Register-only 实验均完成 1000 steps。由于 diffusion timestep 随机，
单点 loss 不适合判断趋势，以下比较前后各 100 steps：

| 方案 | 前 100 mean / median | 后 100 mean / median | mean 变化 |
| --- | --- | --- | ---: |
| A latent prefix | 0.13738 / 0.10039 | 0.15463 / 0.10405 | +12.55% |
| B DiT condition | 0.22996 / 0.18715 | 0.17155 / 0.11778 | -25.40% |

B 有明确改善；A 的中位数基本不变，mean 受高 timestep 的 loss 尖峰影响反而
上升。两者 Register gradient 均下降但未消失。

训练入口已支持 `--train-scope full`，并将上游旧式 reentrant gradient
checkpoint 改为 NAV 内部的 non-reentrant 调用，解决 DiT block 中
`hist/hist_cross` cache 在全参反传时重复访问计算图的问题。

B 全参单步已通过：

- trainable parameters：约 1.42B；
- loss：0.087897；
- gradient norm：1.796875；
- NAV 进程峰值显存：19.668 GiB。

A 全参反传实测需要约 39.49 GiB NAV 显存；GPU 0 同时有约 7.72 GiB 的其他
用户进程，最终因不足 194 MiB OOM。因此当前正式全参阶段先运行 B，A 等待独占
GPU 或引入 FSDP / activation offload，不把降低分辨率的结果与 480P 主实验混用。

## 7. 2026-07-24 结构修订

早期 A 使用 `16×256` token 再广播为空间常数 latent plane，且兼容层会重复
追加 local frame。该结构已经废弃，旧 A checkpoint 不再与新 A 兼容。

当前统一规则：

- local memory 与 Infinite-World 相同，取 history 最后一个 VAE latent
  time step，形状 `[B,16,1,H,W]`，且只拼接一次；
- A 的 history 完全由固定 `[B,16,4,H,W]` Register 代替，DiT latent
  输入顺序为 `[4 register planes; 1 local plane; noisy target]`；
- B 的 latent 侧只有 `[1 local plane; noisy target]`，Register 仅作为
  DiT cross-attention condition；
- Infinite-World 原始推理未指定 `memory_is_precomputed` 时仍走原 HPMC
  和 latest-frame 路径，不受 NAV 接口影响。

新结构单步端到端验证：

| 方案 | Condition latent budget | loss | Register grad norm | 峰值显存 |
| --- | --- | ---: | ---: | ---: |
| A | 4 Register + 1 local | 0.097775 | 0.104004 | 18.533 GiB |
| B | 1 local；Register 在 condition | 0.431135 | 15.375 | 10.243 GiB |

### B condition 的最终定义

B 不保留 HPMC history。Register 与文本 condition 拼接：

```text
latent side:    [1 latest local plane; noisy target]
condition side: [text tokens; Register tokens]
```

RE10K 本地 manifest 不含 caption、title 或 description，因此 text 使用 UMT5
对空字符串 `""` 的真实编码。Register 从 `[B,16,256]` 投影为
`[B,16,4096]`，追加到512个文本位置之后。拼接结果共同经过 Infinite-World
原 `text_embedding: 4096→1536→1536`，供30层 DiT cross-attention 使用。

## 8. 新 A 全参数正式训练

确认 A 相对 Infinite-World 的生成主干只替换 history memory：

```text
Infinite: [HPMC history; latest local; noisy target]
NAV-A:    [T=4 Register; latest local; noisy target]
```

ActionEncoder、20-channel patch input、condition mask、30层 DiT、文本
cross-attention、velocity head 和 RFlow loss 均沿用 Infinite-World。训练时仅将
上游 reentrant checkpoint 换为数值等价的 non-reentrant checkpoint，以支持
全参反向传播。

正式训练从 `/sharedata/Infinite-World/checkpoints/infinite_world_model.ckpt`
开始，加载审计 `missing=0, unexpected=0`；不加载任何旧 NAV-A Register
checkpoint，新 `[B,16,4,H,W]` Register 随机初始化。

```text
tmux session: nav_full_a
run: re10k-new-a-full-from-infinite-formal
trainable parameters: 1,421,435,008
steps: 1000
save interval: 100
```

已完成的前三步 loss 为 `0.097775、0.074608、0.081861`，NAV 进程峰值显存
为 `32.867 GiB`。

## 9. 新 B 全参数正式训练

B 从同一份原始 Infinite-World checkpoint 开始，不加载旧 Register 权重：

```text
latent:    [1 latest local memory; noisy target]
condition: [UMT5 empty-text embedding; 16 Register tokens]
history:   none
```

RE10K 不含 caption，因此空文本由 UMT5-XXL 对 `""` 实际编码并缓存于
`/sharedata/RealEstate10K/nav_register/text/empty_umt5.pt`，形状为
`[1,1,512,4096]`，有效文本 token 数为1。Register condition 为
`[1,1,16,4096]`，两者沿 token 维拼接。

```text
tmux session: nav_full_b
run: re10k-new-b-full-from-infinite-formal
trainable parameters: 1,424,037,200
steps: 1000
save interval: 100
```

全参 step 1 已完成：loss `0.208787`，全参数 gradient norm `18.625`，NAV
进程峰值显存 `16.480 GiB`。

## 10. 公平对照重启：Effective Batch Size 4

RE10K 当前实际缓存训练量：

| 范围 | 窗口 | 场景 | 窗口时长求和 | 去重覆盖时长 |
| --- | ---: | ---: | ---: | ---: |
| 已缓存 | 10 | 7 | 53.714 秒 / 0.0149 小时 | 45.939 秒 / 0.0128 小时 |
| 全部 manifest | 89 | 57 | 479.262 秒 / 0.1331 小时 | 395.439 秒 / 0.1098 小时 |

物理 batch 4 在 B 的全参 backward 中 OOM。修复上游 batch>1 时 action
embedding 被错误广播成 `B²` 的问题后，物理 batch 2 验证通过。因此两组统一
使用有效 batch 4：

| 方案 | Micro batch | Accumulation | Effective batch | Step 1 loss | NAV 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 4 | 4 | 0.073745 | 29.209 GiB |
| B | 2 | 2 | 4 | 0.221912 | 23.375 GiB |

两者均重新从原始 Infinite-World checkpoint 启动，`missing=0`、
`unexpected=0`，不加载旧 Register 权重，并使用相同的 UMT5 empty-text
embedding。计划均为1000个 optimizer steps，每步处理4个样本。

## 11. 使用完整 RE10K train split

训练数据已扩展为本机公共目录中全部180个 train episodes，共生成269个
两-chunk训练窗口：

| 项目 | 数量 |
| --- | ---: |
| Train episodes / scenes | 180 |
| 训练窗口 | 269 |
| 原始长度足够、直接切窗 | 146 |
| 短 episode、全时间跨度均匀重采样 | 123 |
| Latent cache | 269 |

长 episode 采用162帧窗口、stride 81，并补充最后一个窗口覆盖尾部。短于162帧
的 episode 不再丢弃，而是在其完整时间跨度上均匀采样到162帧；重复位置的
action label 为 no-op。全部 cache 已逐文件检查，manifest 与 cache ID
一一对应，`missing=0`、`extra=0`、坏文件为0；tensor shape 均为
`[1,2,16,21,56,112]`。

正式运行名为：

```text
A: re10k-all-a-full-ebs4-from-infinite
B: re10k-all-b-full-ebs4-from-infinite
```

两组均从原始 Infinite-World checkpoint 初始化，全参数训练1000个 optimizer
steps，不加载旧 Register 权重。由于共享 GPU 上存在其他任务，两组最终均使用
`micro batch=1, gradient accumulation=4, effective batch=4`，并开启
PyTorch expandable-segments allocator 以降低共享显存碎片。
