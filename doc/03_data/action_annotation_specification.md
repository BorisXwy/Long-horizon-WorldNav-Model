# NAV 相机 Pose 与 Action 标注规范

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-DAT-006` |
| 类型 | 标注规范（Annotation Specification） |
| 状态 | Active / Source of Truth |
| 更新时间 | 2026-07-29 |
| 职责 | 定义稀疏 Pose 对齐、插值、相邻帧运动和 Infinite-World Action 离散化 |

## 标注目标

NAV 将相机轨迹转为与每个 RGB 帧一一对应的两条离散控制序列：

```text
move[t] ∈ {0,...,9}
view[t] ∈ {0,...,9}
```

它们是由 camera pose 推导的 pseudo-action，不是数据集原生用户输入。用途是让
Infinite-World 的原 ActionEncoder 在无真实键盘动作的数据上获得近似相机控制。

## Pose 规范化

所有输入最终统一为每帧3×4 world-to-camera：

\[
P_t=[R_t\mid t_t]
\]

若数据提供 camera-to-world 4×4 矩阵，先求逆再截取前3行。相机中心为：

\[
c_t=-R_t^\top t_t
\]

相邻帧在前一相机坐标系中的平移为：

\[
\Delta p_t=R_t(c_{t+1}-c_t)
\]

相对旋转为：

\[
\Delta R_t=R_{t+1}R_t^\top
\]

坐标约定为相机 \(x\) 向右、\(-z\) 向前。RE10K 的非严格旋转矩阵先通过 SVD
投影到最近的 \(SO(3)\)。

## 稀疏 Pose 变稠密

对于两个带标注帧 \(i,j\)：

- Translation：按原视频帧号线性插值；
- Rotation：使用 quaternion `Slerp`；
- 超出首末标注范围的帧不应作为可靠覆盖；当前窗口选择限制在可用范围；
- 重复 timestamp 先稳定排序并去重；
- `poses.npy` 与 `indexes.txt` 长度不一致时按最短长度配对，并记录
  `annotation_audit.length_mismatch=true`。

插值只表达“相邻标注间相机平滑运动”的近似，不能恢复中间的快速抖动、急转或
Pose tracking failure。未来应把标注间隔、置信度和旋转突变写为 sample weight。

## 连续运动到离散 Action

在每个 episode 内计算：

```text
translation magnitude = norm([Δx, Δz])
rotation magnitude    = hypot(yaw, pitch)
```

使用所有非零相邻帧变化的中位数作为 episode 内尺度：

```text
translation_median
rotation_median_rad
```

归一化后：

- ratio `< 0.25`：`no-op`；
- ratio `> 3.0`：`uncertain`；
- 其余按主方向或对角方向离散；
- 对角判定阈值为 `0.414≈tan(22.5°)`。

类别定义：

| ID | Move | View |
| ---: | --- | --- |
| 0 | no-op | no-op |
| 1 | go forward | turn up |
| 2 | go back | turn down |
| 3 | go left | turn left |
| 4 | go right | turn right |
| 5 | forward + left | up + left |
| 6 | forward + right | up + right |
| 7 | back + left | down + left |
| 8 | back + right | down + right |
| 9 | uncertain | uncertain |

第一帧没有前驱，默认标为0。随机窗口必须先在完整 episode 上计算相邻运动，再
切片；否则窗口首帧会被错误重置为 `no-op`。

## 数据集特例

### SpatialVID

`indexes.txt` 第二列是原视频帧号，末项对应原视频最后一帧。Pose 以
`[tx,ty,tz,qx,qy,qz,qw]` 存储。全 episode 插值、标注和 calibration 完成后
再切确定性随机窗口。

### DL3DV

Pose 来自 `transforms.json`，每条与 `images_8/frame_*.png` 一一对应。当前不
假设它与原 MP4 帧号存在直接对应关系，也不对原 MP4 全帧生成伪 action。

### Argoverse 2

将 `city_SE3_egovehicle` 与 `egovehicle_SE3_sensor` 组合为相机轨迹。每个
`ring_front_center` 图像 timestamp 匹配最近的高频 ego pose。

### RE10K

使用原逐帧外参；部分不足162帧的序列均匀重采样，因此可能出现重复 RGB 帧和
零位移 action。相关样本必须保留 `temporally_resampled` 标记。

## 已知分布风险

早期 Short cache 审计显示：

- SpatialVID 占样本绝大多数；
- `move=2` 在旧 SpatialVID 标签中占比异常高；
- DL3DV 的动作类别分布与 RE10K/SpatialVID 明显不同；
- 跨数据集坐标、Pose方向或轨迹质量错误会表现为类别单边倒。

因此每次冻结新数据版本都必须输出：

1. 各数据集 `move/view` 直方图；
2. no-op 与 uncertain 比例；
3. translation/rotation calibration 分布；
4. 插值间隔和长度异常数量；
5. 随机可视化若干轨迹及对应视频；
6. 跨数据集方向 sanity check。

只验证 tensor 可加载不足以证明 Action 语义正确。

## 当前脚本

| 职责 | 脚本 |
| --- | --- |
| 通用 Pose/Action manifest | `NAV/scripts/prepare_multidataset_manifest.py` |
| RE10K窗口与动作 | `NAV/scripts/prepare_re10k_manifest.py` |
| V1 T4 micro episode manifest | `NAV/scripts/datasets/build_v1_t4_micro_manifest.py` |
| V1 full-pipeline 模型链路验证 | `NAV/scripts/smoke_v1_full_pipeline.py` |

数据构建总规范见 `training_data_construction.md`。
