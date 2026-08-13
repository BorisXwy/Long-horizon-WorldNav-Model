# V1 完整模型链路 Smoke 记录

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-EVL-005` |
| 类型 | 完整模型链路验证（Full-pipeline Smoke） |
| 状态 | Smoke Verified |
| 更新时间 | 2026-08-14 |
| 职责 | 记录新版 V1 完整模型在三阶段训练、参数更新、结构约束和两种推理模式上的最小可运行验证 |

## 当前结论

当前有效 smoke 入口是：

```bash
bash NAV/scripts/run_v1_full_pipeline_smoke.sh cpu
```

或指定 CUDA：

```bash
CUDA_VISIBLE_DEVICES=0 bash NAV/scripts/run_v1_full_pipeline_smoke.sh cuda:0
```

该 smoke 不再使用旧 scaffold、小旁路 head、旧 A/B Register 或旧
InfiniteWorld adapter。参数可以随机初始化，但必须走当前完整结构：

```text
R_null fixed template
  -> RegisterCell(R_{i-1}, concat([visual_tokens(C_i), A_hist_i]))
  -> shared dual-stream backbone
      visual stream: Register / obs / future noisy video
      action stream: A_noise
  -> video generation head
  -> action flow decoder
  -> 3D geometry probe
```

## 覆盖范围

| 检查项 | 要求 |
| --- | --- |
| Stage One | `L_visual_flow` 正常 forward/backward/update；`A_cur`、`A_noise` 保留格式但不计算 action supervision |
| Stage Two | 同一视频生成前向上增加 `L_3D`，确认 `geometry_probe` 和 shared backbone 有梯度更新 |
| Stage Three | `L_action_flow + CE_aux` 为主，同时加入 video/3D rehearsal loss，避免 backbone/Register 退化 |
| Videogen inference | 输出 `z_future` 与 `future_velocity`，shape 与 noisy future latent 对齐 |
| Policy inference | 不输入 future noisy video，输出 `action_chunk`、`primitive_logits`、`primitive_ids` |
| 结构审计 | 必须使用 `RegisterCell`；不得出现 RegisterExtractor/RegisterUpdater；不得出现 action/latent additive bias |

## 2026-08-14 运行记录

CPU 完整链路：

```text
report:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014744/report.json

device:
  cpu

inference:
  videogen_z_future       = [2, 16, 2, 8, 8]
  videogen_future_velocity= [2, 16, 2, 8, 8]
  policy_action_chunk     = [2, 10, 6]
  policy_primitive_logits = [2, 10, 12]
  policy_primitive_ids    = [2, 10]
```

清理后的 CUDA smoke 也已通过：

```text
report:
  NAV/log/full_pipeline_smoke/v1_full_pipeline_smoke_20260814_014807/report.json

device:
  cuda:0
```

后续每次改动模型结构、attention 规则、loss 组成、action/token 接口或训练入口后，
必须重新跑该 full-pipeline smoke，并把 report 路径登记到本文或对应实验记录中。

## 旧入口清理

2026-08-14 已从当前代码主线移除旧 scaffold、旧 `A_query`、旧 action-bias、
旧 InfiniteWorld adapter 和旧 A/B Register 训练入口。V0 论文/实验结果文档仍可
作为历史解释，但不再对应可执行主线脚本。若需要复现 V0，请从 git 历史恢复到
清理前 commit，而不是把旧脚本混回 V1 主线。
