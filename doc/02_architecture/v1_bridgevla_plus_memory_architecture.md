# V1 BridgeVLA++ Memory 架构复现记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-ARC-004` |
| 类型 | 架构解析 / 复现准备 |
| 状态 | Verified / Smoke Reproduced |
| 更新时间 | 2026-08-08 |
| 职责 | 记录 BridgeVLA++ memory-enhanced 版本的代码、环境、权重和 memory 机制 |

## 当前结论

BridgeVLA++ memory-enhanced 版本已经按官方 `main` 分支完成 clone、环境配置、权重下载和
memoryBench smoke 复现。代码位于 3d_wm_vln 同级仓库，数据与权重位于公共
`/sharedata/BridgeVLA`，不放入 NAV 自身目录。

- 官方仓库：`https://github.com/BridgeVLA/BridgeVLA`
- 官方项目页：`https://bridgevla-plus.github.io/`
- 论文：`https://arxiv.org/abs/2608.05042`
- Hugging Face 数据与权重发布页：`https://huggingface.co/datasets/LPY/BridgeVLA`
- ModelScope checkpoint 镜像：`https://modelscope.cn/models/susetiankong/bridgevla_plus`

## 本地仓库与环境

| 项 | 路径 / 值 |
| --- | --- |
| 代码仓库 | `/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/BridgeVLA` |
| 分支 | `main` |
| commit | `8855333` |
| Conda 环境 | `bridgevla_plus_gembench` |
| 环境路径 | `/mnt/pool1/sharehome/xiewenyuan/.conda/envs/bridgevla_plus_gembench` |
| 数据根目录 | `/sharedata/BridgeVLA/data/bridgevla_data` |
| 权重根目录 | `/sharedata/BridgeVLA/data/bridgevla_ckpt` |
| 仓库内 data 链接 | `BridgeVLA/data -> /sharedata/BridgeVLA/data` |

激活环境：

```bash
source /sharedata/anaconda3/etc/profile.d/conda.sh
conda activate bridgevla_plus_gembench
```

## 已配置资源

| 资源 | 路径 | 状态 |
| --- | --- | --- |
| BridgeVLA++ memoryBench checkpoint | `/sharedata/BridgeVLA/data/bridgevla_ckpt/bridgevla_plus/memorybench/model_160.pth` | 已下载，约 8.3GB |
| memoryBench exp config | `/sharedata/BridgeVLA/data/bridgevla_ckpt/bridgevla_plus/memorybench/exp_cfg.yaml` | 已下载 |
| memoryBench MVT config | `/sharedata/BridgeVLA/data/bridgevla_ckpt/bridgevla_plus/memorybench/mvt_cfg.yaml` | 已下载 |
| PaliGemma 3B | `/sharedata/BridgeVLA/data/bridgevla_ckpt/paligemma-3b-pt-224` | 已下载 |
| CLIP RN50 | `/sharedata/BridgeVLA/data/bridgevla_ckpt/clip/RN50.pt` | 已下载并通过脚本校验 |
| memoryBench data zips | `/sharedata/BridgeVLA/data/bridgevla_data/memorybench/data/{train,test}` | train/test 三个 task zip 均已下载 |
| memoryBench keyframe cache | `/sharedata/BridgeVLA/data/bridgevla_data/memorybench/data/train/_keyframe_cache/size128_v3` | 已下载 |

当前落盘体积口径：

| 路径 | 体积 |
| --- | ---: |
| `/sharedata/BridgeVLA/data/bridgevla_ckpt` | 19G |
| `/sharedata/BridgeVLA/data/bridgevla_data/memorybench` | 30G |

说明：为了先完成复现 smoke，已完整展开 `test/put_block_back`；`train/put_block_back`
在官方解压脚本执行中途保留了部分已展开 episode，后续如需训练或完整数据审计，应重新运行
官方解压脚本补齐全部 train/test task。zip 原包均已在公共目录保留。

## Memory 机制

memoryBench 对 BridgeVLA++ 的 memory 机制是核心验证对象。官方配置里包含两类 memory：

1. **Temporal Memory**：面向长程任务阶段的时间记忆。memoryBench README 与代码语义显示，
   它使用 frame-0 anchor 加最近两个已执行 keyframes，即 `memory.select: keyframe_gt`
   和 `k_temporal: 2`。这相当于把历史关键阶段压成少量 memory inputs，而不是把所有历史帧
   或所有 step 全部塞入策略。
2. **Spatial Memory**：面向局部空间锚定的记忆。官方 server/client 均要求
   `SPATIAL_MEMORY=true` 与 checkpoint 配置一致，用于 stage-2 的 spatial anchor。

memoryBench 复现时必须让 server 与 client 的 memory switch 一致：

```bash
TEMPORAL_MEMORY=true
SPATIAL_MEMORY=true
```

server 会读取 checkpoint 附带的 `mvt_cfg.yaml` 并检查 memory 开关；不匹配会直接中止。

## 对 NAV 的启发

BridgeVLA++ 的 memory 不是视频世界模型式的 latent history autoregression，而是面向
robot manipulation 的少量关键帧 / 空间锚点 memory。它对 NAV 的参考价值主要在于：

- memory 作为策略输入的一等条件，而不是简单追加完整历史；
- temporal memory 可以只保留 anchor + recent keyframes，显著限制 token / image 输入数量；
- spatial memory 与 policy stage 绑定，而不是离线保存 episode-specific hidden state。

这与 NAV register 思想兼容：NAV 的 Register 可以承担「可递归更新的历史摘要」，而
BridgeVLA++ 提醒我们在导航后训练阶段应明确区分 temporal anchor、recent observation
和 spatial anchor 三类信息。
