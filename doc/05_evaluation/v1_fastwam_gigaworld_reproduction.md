# V1 Fast-WAM 与 GigaWorld-Policy 复现记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-004` |
| 类型 | 外部 WAM 基线复现 |
| 状态 | Smoke Reproduced |
| 更新时间 | 2026-08-09 |
| 职责 | 记录 Fast-WAM 与 GigaWorld-Policy-0.5 的本地仓库、环境、权重、最小 smoke、latency 复现和阻塞 |

## 当前结论

本轮复现选择 `Fast-WAM` 与 `GigaWorld-Policy-0.5`，目标不是先跑完整机器人
leaderboard，而是先复现二者对 Wan/WAM backbone 的使用方式：

- Fast-WAM：验证 release checkpoint 的 `infer_action` 路径，即先用当前图像
  经过 video expert 得到 first-frame latent/video cache，再用 action expert 做多步
  action denoising；推理时不生成 dense future video。
- GigaWorld-Policy-0.5：优先验证 `CasualWorldActionTransformer_MoT` 的
  `action_only=True` / prefix-cache 路径；完整 open-loop server/client 需要额外
  Wan Diffusers base model 与 LeRobot v3 数据。

截至 2026-08-09 03:11，两个官方仓库已 clone，虚拟环境已配置，权重已落盘，
NAV 侧最小 smoke 已跑通。GigaWorld 官方 `requirements.txt` 存在
`diffusers==0.36.0` 与 `lerobot==0.4.4` 的 dependency conflict，本地采用
“安装除 `lerobot` 外的官方依赖，再 `--no-deps` 安装 `lerobot==0.4.4`”的方式保持
`diffusers==0.36.0`，不修改官方代码。

## Smoke 结果

| 项 | Fast-WAM | GigaWorld-Policy-0.5 |
| --- | ---: | ---: |
| 运行时间 | 2026-08-09 03:10 | 2026-08-09 02:56 |
| GPU | RTX 6000 Ada | RTX 6000 Ada |
| checkpoint | `libero_uncond_2cam224.pt` | `Giga-World-Policy-0.5` |
| denoising steps | 2 | 2 |
| load seconds | 63.19 | 151.45 |
| infer seconds | 4.35 | 4.11 |
| seconds / denoise step | 2.18 | 2.05 |
| peak allocated GiB | 14.44 | 11.34 |
| 输出 shape | action `[32,7]` | action `[1,48,16]` |
| 结果 JSON | `NAV/result/fastwam/smoke/20260809_031029/summary.json` | `NAV/result/gigaworld_policy/smoke_transformer/20260809_025612/summary.json` |

GigaWorld 的 `infer_seconds=4.11s` 包含首次 forward 的 PyTorch Inductor warmup /
compile 开销。两步分别为 `4.07s` 与 `0.036s`，因此后续正式 latency 统计需要区分
首步 warmup 和 steady-state。

## 路径

| 项 | Fast-WAM | GigaWorld-Policy |
| --- | --- | --- |
| 代码仓库 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/FastWAM` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/giga-world-policy` |
| commit | `45d8e14` | `f3d5a88` |
| 虚拟环境 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_fastwam` | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/virtual_env/.venv_gigaworld_policy` |
| 公共权重/数据 | `/sharedata/FastWAM` | `/sharedata/GigaWorld-Policy` |
| log | `NAV/log/fastwam` | `NAV/log/gigaworld_policy` |
| result | `NAV/result/fastwam` | `NAV/result/gigaworld_policy` |

## 执行入口

Fast-WAM transformer/action-only smoke：

```bash
GPU_ID=0 STEPS=2 bash /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/reproduce_wam/run_fastwam_smoke.sh
```

GigaWorld-Policy-0.5 transformer-only smoke：

```bash
GPU_ID=0 STEPS=2 bash /mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/scripts/reproduce_wam/run_gigaworld_transformer_smoke.sh
```

两个脚本都只封装 NAV 侧路径和最小随机输入，不修改官方 repo。输出为：

```text
NAV/result/fastwam/smoke/<run_id>/summary.json
NAV/result/gigaworld_policy/smoke_transformer/<run_id>/summary.json
```

## 关键日志

| 任务 | 日志或结果 |
| --- | --- |
| Fast-WAM 安装 | `NAV/log/fastwam/install_20260809_023904.log` |
| Fast-WAM ActionDiT 预处理与 smoke | 前台复现 run id `20260809_031029`；结果见 `NAV/result/fastwam/smoke/20260809_031029/summary.json` |
| GigaWorld 安装修复 | `NAV/log/gigaworld_policy/install_repair_20260809_024618.log` |
| GigaWorld transformer smoke | `NAV/result/gigaworld_policy/smoke_transformer/20260809_025612/summary.json` |

## 复现口径

### Fast-WAM

使用官方 `libero_uncond_2cam224.pt` 和 `libero_uncond_2cam224_dataset_stats.json`。
若 `/sharedata/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
不存在，脚本会先按官方 `scripts/preprocess_action_dit_backbone.py` 从 Wan2.2 预处理
ActionDiT backbone。本机需要显式关闭 `redirect_common_files`，否则 DiffSynth 会尝试
访问不可用的 `DiffSynth-Studio/Wan-Series-Converted-Safetensors`；NAV 脚本会临时生成
local-only config，只使用 `/sharedata/FastWAM/models` 下的本地 Wan 权重。

最小 smoke 使用：

- random image：`[1,3,224,448]`，范围 `[-1,1]`；
- random/zero context embedding：`[1,128,4096]`，绕开 T5 text encoder；
- proprio：按 `libero_2cam` 配置使用 8 维 zero proprio；
- action horizon：默认 32；
- denoising steps：默认 2，用于快速验证链路，正式 latency 可改为论文/官方默认 steps。

该路径验证的是 Fast-WAM 的 `infer_action`：

```text
image -> Wan VAE first-frame latent
      -> video expert pre_dit
      -> MoT prefill_video_cache
      -> action expert 多步 denoise
      -> action sequence
```

### GigaWorld-Policy-0.5

GigaWorld 完整 open-loop server/client 需要：

- transformer checkpoint：`/sharedata/GigaWorld-Policy/Giga-World-Policy-0.5`；
- Wan Diffusers base：`/sharedata/GigaWorld-Policy/base_models/Wan2.2-TI2V-5B-Diffusers`；
- norm stats JSON；
- LeRobot v3 packed dataset root 或等效 dummy/open-loop episode。

当前优先 smoke 是 transformer-only，直接加载官方 MoT transformer checkpoint，构造：

- `ref_latents=[1,48,1,24,20]`；
- `action=[1,48,16]`；
- `state=[1,1,16]`；
- `encoder_hidden_states=[1,64,4096]`；
- `action_only=True`。

该路径验证的是 GigaWorld 的 cached action-only forward：

```text
ref_latents + state -> prefix cache
action noise + timestep -> forward_action_stack_with_prefix_cache
                         -> action denoise prediction
```

## 已知问题

1. GigaWorld 官方依赖 pin 互相冲突：`diffusers==0.36.0` 与 `lerobot==0.4.4`
   的 metadata 要求不一致。本地修复策略只影响安装解析，不改模型代码。
2. GigaWorld 完整 open-loop 需要 Wan Diffusers layout；本机已有
   `/sharedata/Wan2.2-TI2V-5B` 是原生 Wan layout，不能直接作为 `BASE_MODEL`。
3. Fast-WAM 完整 LIBERO/RoboTwin benchmark 仍需官方仿真环境与 benchmark assets；
   当前先跑不依赖仿真的 model smoke。
4. 早先一次 Fast-WAM `libero_uncond_2cam224.pt` 由 aria2 产生了尾部全 0 的坏文件，
   虽然文件大小等于 metadata，但 `torch.load` 报
   `failed finding central directory`。已保留为
   `/sharedata/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt.bad_20260809_0305`，
   并用 `hf_hub_download('yuanty/fastwam', 'libero_uncond_2cam224.pt')` 重下。

## 下一步验收

1. 若需要正式 latency，分别记录 warmup 后的 steady-state 单步时间、官方默认 steps
   和完整 action chunk 时间。
2. 若 GigaWorld transformer-only smoke 通过，再补完整 open-loop server/client
   dummy episode 或小型真实 LeRobot episode。
3. 若需要 robot benchmark 分数，再配置 LIBERO/RoboTwin 或 GigaWorld 官方
   LeRobot v3 评测数据；当前 smoke 只验证 backbone/action-only 前向链路。
