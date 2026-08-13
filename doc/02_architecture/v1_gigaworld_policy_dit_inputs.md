# V1 GigaWorld-Policy DiT 输入维度解析

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-ARC-005` |
| 类型 | 架构解析 / DiT 输入维度 |
| 状态 | Verified from Local Code / Smoke Reproduced |
| 更新时间 | 2026-08-09 |
| 职责 | 明确 GigaWorld-Policy-0.5 的 DiT 主输入、condition 输入、token 化规则和 action-only 推理路径 |

## 结论

GigaWorld-Policy-0.5 的 DiT 需要严格区分两类 visual latent：

```text
clean ref / obs latent:
  由当前真实 observation 编码得到，通常 T 很小，推理时作为 prefix/context。

noisy future latent:
  diffusion / flow matching 的待去噪 target，训练 video branch 时才进入；
  action-only 推理时为空，不作为 policy 输入。
```

它将 Wan2.2 的视觉 DiT 改造成 Mixture-of-Transformers（MoT）：

```text
Visual Stream: ref / future visual latent tokens, width = 3072
Action Stream: state / action tokens,              width = 1024
```

训练时同时输出 `visual_pred` 与 `action_pred`，计算 `visual_loss` 和
`action_loss`；推理时使用 `action_only=True`，只用 clean ref visual/state
prefix 构建 KV cache，然后每个 denoising step 只跑 action tokens。此时
`noisy_latents` 是空的 future visual slice。

因此它快的核心不是模型小，而是 policy 推理路径完全移除了 noisy future
visual target，只保留很小的 clean observation prefix：

```text
官方 open-loop 常用 ref image: 320×384 RGB
VAE latent: [B,48,1,48,40]
patch_size=[1,2,2]
ref visual tokens = 1×24×20 = 480
action tokens = 48
state tokens = 1
```

NAV smoke 为了快速验证使用了更小的 synthetic ref latent：

```text
ref_latents = [1,48,1,24,20]
ref visual tokens = 1×12×10 = 120
action = [1,48,16]
state = [1,1,16]
```

两者的 transformer core 完全一致，但 token 数不同；不要把 smoke 的 120 个
ref tokens 误认为官方 open-loop 默认视觉 token 数。

## Transformer 配置

本地复现 checkpoint：

```text
/sharedata/GigaWorld-Policy/Giga-World-Policy-0.5/config.json
```

核心配置：

| 项 | 值 | 含义 |
| --- | ---: | --- |
| `in_channels` / `out_channels` | 48 | Wan VAE latent channel |
| `patch_size` | `[1,2,2]` | 时间不降采样，空间 2×2 patch |
| `num_layers` | 30 | MoT block 数 |
| `num_attention_heads` | 24 | attention heads |
| `attention_head_dim` | 128 | head dim |
| visual hidden width | 3072 | `24×128` |
| `ffn_dim` | 14336 | visual expert FFN |
| `text_dim` | 4096 | UMT5 text embedding width |
| `action_expert_dim` | 1024 | action/state stream width |
| `action_ffn_dim` | 4096 | action expert FFN |
| `in_action_channels` / `out_action_channels` | 16 | checkpoint core 的 action/state channel |
| `num_embodiments` | 2 | embodiment-specific action/state encoder |

官方 open-loop 脚本中存在数据侧 14/32 维 robot state/action 的 padding 与切片逻辑；
本文档的 16 维指 checkpoint core 配置与本地 transformer smoke 验证的维度。

## 从原始输入到 DiT 主输入

### 原始视觉输入

Open-loop policy 从三路相机读取当前观测：

```text
cam_high:        [3,H,W]
cam_left_wrist:  [3,H,W]
cam_right_wrist: [3,H,W]
```

`get_ref_image_3views(..., dst_size=(320,384), layout="tshape")` 将三路相机拼成
一张 PIL image：

```text
ref_image RGB: [3,384,320]
```

随后 `WAPipeline.prepare_latents` 将 image 变为：

```text
image: [B,3,384,320]
image.unsqueeze(2): [B,3,1,384,320]
```

Wan VAE 编码：

```text
latent_condition = VAE(image)
                 ≈ [B,48,1,48,40]
```

这里空间压缩比例是 8，因此 `384/8=48`，`320/8=40`。GigaWorld-Policy-0.5
使用 `expand_timesteps=True`，action-only 推理只取：

```text
ref_latents_for_action = condition[:, :, :1]
                       = [B,48,1,48,40]
```

这一项是 clean observation latent。它不是 diffusion noise，也不是未来视频
target。它在 action-only 推理中作为视觉 prefix，被缓存为 KV。

### Future noisy latent 的来源

`prepare_latents` 还会按照 `num_frames` 创建视频生成用的 noisy future latent：

```text
num_latent_frames = (num_frames - 1) // 4 + 1
latents = randn([B,48,num_latent_frames,Hl,Wl])
```

在官方 open-loop 脚本中 action-only policy 调用使用：

```text
num_frames = 5
num_latent_frames = (5-1)//4 + 1 = 2
latents: [B,48,2,48,40]
```

但 action-only fast path 不把这段 noisy video latent 送进 transformer。代码中：

```text
ref_latents_for_action = condition[:, :, :1]     # [B,48,1,48,40]
empty_noisy_latents_for_action = condition[:, :, 1:1]  # [B,48,0,48,40]
```

因此 action-only 推理的 visual stream 只有 clean ref/obs latent，没有 noisy
future latent：

```text
visual_input = ref_latents
             = [B,48,1,48,40]
```

训练 video branch 时才会使用 noisy future latent；policy 推理路径将其删除。

### 原始 action/state 输入

Open-loop 脚本从数据取 robot state：

```text
state_raw: [14]
```

然后归一化、padding：

```text
norm_state: [1,32]   # 脚本数据侧 padding
```

checkpoint core 的 transformer 配置是 `in_action_channels=16`。本地 smoke 直接按
checkpoint core 使用：

```text
state:        [B,1,16]
action noise: [B,48,16]
```

其中 `48` 是 action horizon / action chunk。Action 在 diffusion/flow matching
意义下也是 noisy variable：

```text
action_noise  ~ N(0,I)
noisy_action  = action_noise * sigma + action_gt * (1-sigma)
action_target = action_noise - action_gt
```

## DiT 主输入：两个 stream

GigaWorld 的 `CasualWorldActionTransformer_MoT.forward` 接收：

```python
transformer(
    ref_latents=ref_latents,
    noisy_latents=noisy_latents,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    action=noisy_action,
    state=state,
    embodiment_id=embodiment_id,
)
```

这里真正进入 MoT blocks 的主输入不是一个单独 tensor，而是两个 stream。

### Visual Stream

代码路径：

```text
encode_visual_tokens(ref_latents, future_latents)
```

若训练时有 future/noisy visual latent：

```text
visual_input = concat(ref_latents, future_latents, dim=2)
```

shape：

```text
ref_latents:    [B,48,1,Hl,Wl]      # clean obs/history prefix
future_latents: [B,48,Tf,Hl,Wl]     # noisy future target
visual_input:   [B,48,1+Tf,Hl,Wl]
```

经过 3D patch embedding：

```text
patch_embedding = Conv3d(
    in_channels=48,
    out_channels=3072,
    kernel_size=[1,2,2],
    stride=[1,2,2],
)
```

得到：

```text
visual_hidden_states:
  [B, (1+Tf) * (Hl/2) * (Wl/2), 3072]
```

官方 open-loop action-only：

```text
ref_latents: [B,48,1,48,40]
Tf = 0
visual tokens = 1*24*20 = 480
visual_hidden_states = [B,480,3072]
```

也就是说，虽然 `prepare_latents` 可以构造 video noisy latent，但在 action-only
路径里：

```text
noisy_latents = [B,48,0,48,40]
future visual tokens = 0
```

本地 smoke：

```text
ref_latents: [1,48,1,24,20]
visual tokens = 1*12*10 = 120
visual_hidden_states = [1,120,3072]
```

训练时若 `num_frames=5`，Wan VAE latent time 约为：

```text
num_latent_frames = (5-1)//4 + 1 = 2
```

则完整 video-training visual stream 包含：

```text
1 个 clean/ref latent frame + 1 个 future/noisy latent frame
visual tokens = 2*24*20 = 960   # 以 384×320 为例
```

如果训练使用更长 video target，则增长的是 `future_latents` 的 `Tf`：

```text
future noisy tokens = Tf * (Hl/2) * (Wl/2)
clean obs/ref tokens = 1 * (Hl/2) * (Wl/2)
```

这和 NAV 之前的问题正好对应：大 T 的主要成本应只属于训练时的 future
consequence branch，而不应进入 Stage Three policy-only 推理。

### Action Stream

代码路径：

```text
encode_action_tokens(state, action)
```

state/action 先分别经过 embodiment-specific MLP：

```text
state:  [B,Ns,Da] -> [B,Ns,1024]
action: [B,Na,Da] -> [B,Na,1024]
```

其中 checkpoint core 中 `Da=16`，smoke 中：

```text
state:  [1,1,16]  -> [1,1,1024]
action: [1,48,16] -> [1,48,1024]
```

拼接后：

```text
action_hidden_states = [B, Ns+Na, 1024]
```

smoke / official action horizon 对应：

```text
Ns = 1
Na = 48
action_hidden_states = [B,49,1024]
```

注意：state token 和 action tokens 同属 action stream，只是 attention mask 中
state 被当作 prefix。

## Condition 输入

### Text / Instruction condition

`encoder_hidden_states` 来自 UMT5 / fixed T5 embedding：

```text
prompt_embeds: [B,L_text,4096]
```

官方 finetune config 中 `max_prompt_len=64`，因此常见为：

```text
prompt_embeds: [B,64,4096]
```

进入 MoT 后，text condition 分别投影给两个 stream：

```text
condition_embedder        : [B,L,4096] -> [B,L,3072]  # visual stream cross-attn
action_condition_embedder : [B,L,4096] -> [B,L,1024]  # action stream cross-attn
```

每个 MoT block 中：

```text
visual_hidden_states cross-attend encoder_visual_hidden_states
action_hidden_states cross-attend encoder_action_hidden_states
```

### Timestep condition

GigaWorld 使用 Flow Matching / diffusion timestep。训练时分别采样：

```text
visual_timestep: [B]
action_timestep: [B]
```

然后构造一个 per-token timestep 序列。语义顺序是：

```text
[state, ref, action, future]
```

shape：

```text
timestep: [B, Ns + Nref + Na + Nfuture]
```

`_split_timestep` 再按 stream 取出：

```text
action_timestep_seq = timestep[:, state_idx + action_idx]
visual_timestep_seq = timestep[:, ref_idx + future_idx]
```

并通过两个 time embedder：

```text
action_timestep_proj: [B,Ns+Na,6,1024]
visual_timestep_proj: [B,Nref+Nfuture,6,3072]
```

在 action-only 推理中，future visual token 不进入 forward，但 timestep template
仍按 ref/action 的位置构造；模型内部会裁剪到：

```text
Ns + Nref + Na
```

### Image condition

GigaWorld-Policy-0.5 checkpoint 配置：

```text
image_dim = null
added_kv_proj_dim = null
```

因此 0.5 的 MoT core 不额外使用 CLIP image embedding 作为 cross-attention
condition。当前 ref image 通过 VAE latent 进入 visual stream，而不是作为
`encoder_hidden_states_image` 进入 condition stream。

### Embodiment condition

`embodiment_id` 不作为 token 输入，而是选择 embodiment-specific linear
weights，用于：

```text
state_encoder
action_encoder
action_decoder
```

## MoT Block 内部计算

每层 `WanMoTTransformerBlock` 有两个 expert：

```text
visual_expert: width 3072, FFN 14336
action_expert: width 1024, FFN 4096
```

每层执行：

```text
1. visual/action 各自算 Q/K/V
2. Q/K/V 在 token 维拼接，做一次 joint self-attention
3. 输出再切回 action stream 与 visual stream
4. action stream 对 text 做 cross-attention + FFN
5. visual stream 对 text 做 cross-attention + FFN
```

伪代码：

```text
q_action,k_action,v_action = ActionExpert(action_hidden)
q_visual,k_visual,v_visual = VisualExpert(visual_hidden)

Q = concat(q_action, q_visual)
K = concat(k_action, k_visual)
V = concat(v_action, v_visual)

attn_out = Attention(Q,K,V, mot_attention_mask)

action_hidden = action_hidden + action_proj(attn_out_action)
visual_hidden = visual_hidden + visual_proj(attn_out_visual)

action_hidden = CrossAttnTextAndFFN(action_hidden, text_action)
visual_hidden = CrossAttnTextAndFFN(visual_hidden, text_visual)
```

## Causal Mask

虽然代码内部 stream 顺序是：

```text
action_stream = [state, action]
visual_stream = [ref, future]
```

但 mask 语义等价于：

```text
[state, ref, action, future]
```

允许关系：

```text
state  can attend: state + ref
ref    can attend: state + ref
action can attend: state + ref + action
future can attend: state + ref + action + future
```

因此：

```text
action 不读取 future visual
future visual 可以读取 action
```

这正是 action-centered causal interface：

```text
history/current obs/text -> action -> future visual consequence
```

## Action-only 推理路径

`scripts/inference_openloop.py` 中 action-only fast path：

```python
current_model(
    ref_latents=ref_latents_for_action,
    noisy_latents=empty_noisy_latents_for_action,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    action=action,
    state=state,
    action_only=True,
)
```

其中两个 visual latent 的实际含义是：

```text
ref_latents_for_action:
  clean current observation latent
  [B,48,1,48,40]   # official open-loop default

empty_noisy_latents_for_action:
  empty future visual target
  [B,48,0,48,40]
```

进入 `_forward_inference_action_only_cached`：

1. 第一次 forward：
   - encode `state` 和 `ref_latents`；
   - 对每层运行 `forward_prefix_cache`；
   - 缓存 prefix 的 K/V；
   - prefix 主要包含 `state + ref visual`。
2. 后续每个 denoising step：
   - 只 encode 当前 noisy action；
   - `forward_action_stack_with_prefix_cache` 中只用 action query；
   - K/V = cached prefix K/V + 当前 action K/V；
   - 输出 action denoise prediction。

官方 open-loop 典型 token 量：

```text
prefix:
  state tokens = 1
  ref visual tokens = 480

per-step:
  action tokens = 48
```

本地 smoke token 量：

```text
prefix:
  state tokens = 1
  ref visual tokens = 120

per-step:
  action tokens = 48
```

这就是 GigaWorld 推理快的核心：每步 action denoise 不再跑完整 future visual
grid，更不 decode video。

## 对 NAV 的直接约束

NAV 的 Stage Three policy-only interface 应学习这个变量分解，而不是把完整
`[B,C,T,H,W]` video latent grid 作为策略主输入：

```text
推荐 policy 主输入:
  Register / Memory compact tokens
  Current Observation visual prefix tokens
  Instruction text tokens
  Action/Nav query tokens

训练时 auxiliary:
  Future Visual / Future 3D consequence tokens

推理时删除:
  Future Visual / Future 3D tokens
```

否则即使 backbone 参数不大，也会因 token 数过大导致训练和推理速度不可接受。
