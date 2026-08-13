# VGGT-Ω 网络结构与 Register Token 代码解析

| 字段 | 内容 |
| --- | --- |
| 文档 ID | `NAV-ARC-002` |
| 类型 | 架构解析（Architecture Note） |
| 状态 | Verified |
| 更新时间 | 2026-07-28 |
| 职责 | 解释 VGGT-Ω Register Token，并界定可迁移到 NAV 的机制 |

## 1. 模型定位

VGGT-Ω 是前馈式三维重建模型（Feed-forward 3D Reconstruction Model）。输入
一组无序或视频帧，单次前向传播输出每帧相机参数、深度图和深度置信度；可选
模型还会输出与语言对齐的序列特征。

它使用 Register Tokens（论文也称 Scene Tokens）汇聚多帧场景信息，并在部分
层中以 Register Attention 替代昂贵的全局注意力。原论文称该改造使训练显存
约为原 VGGT 的 30%，同时保留重建性能。

当前本地代码版本为 `39a0cb8`。

## 2. 项目结构

```text
vggt-omega/
├── README.md
├── demo_gradio.py                  交互式重建演示
├── examples/                       示例视频
├── visual_util.py                  相机和点云可视化
├── requirements*.txt
└── vggt_omega/
    ├── models/
    │   ├── vggt_omega.py           顶层模型与三个预测头的调度
    │   ├── aggregator.py           DINOv3、交替注意力和 Register Attention
    │   ├── heads/
    │   │   ├── camera_head.py      9 维相机参数预测
    │   │   ├── dense_head.py       深度与置信度预测
    │   │   └── text_alignment_head.py
    │   └── layers/
    │       ├── vision_transformer.py  DINOv3 ViT
    │       ├── block.py               Transformer Block
    │       ├── attention.py           Self-Attention、Q/K Norm、RoPE
    │       ├── patch_embed.py
    │       └── rope_position_encoding.py
    └── utils/
        ├── load_fn.py               图像加载与预处理
        ├── pose_enc.py              9 维相机编码与矩阵转换
        └── geometry.py              深度反投影等几何工具
```

## 3. 整体网络

1B 模型的主要配置：

| 参数 | 数值 |
| --- | ---: |
| Patch Size | 16×16 |
| Token Dimension | 1024 |
| Attention Heads | 16 |
| DINOv3 Blocks | 24 |
| Aggregator Alternating Blocks | 24 |
| Registers per Frame | 16 |
| Camera Tokens per Frame | 1 |
| Register Attention Layers | 2、6、9、14、20（从 0 开始） |
| Cached Feature Layers | 4、11、17、23 |

总体数据流为：

```text
输入图像 [B, F, 3, H, W]
        │
        ▼
DINOv3 ViT：逐帧提取 Patch Tokens
        │
        ▼
每帧拼接 [1 Camera Token, 16 Register Tokens, P Patch Tokens]
        │
        ▼
24 × {
    Frame Attention
    Global Attention 或 Register Attention
}
        │
        ├────────────► Camera Head ──► 9D Camera
        ├────────────► Dense Head  ──► Depth + Confidence
        └────────────► 可选 Text Alignment Head
```

其中 \(F\) 是输入帧数，\(P=(H/16)(W/16)\) 是每帧 Patch Token 数量。

## 4. DINOv3 图像编码器

`Aggregator.patch_embed` 的名字容易引起误解：它不是单独的一层 Patch
Projection，而是一套完整的 24 层 DINOv3 Vision Transformer。

每帧独立执行：

```text
RGB Image
 → Conv2D Patch Projection
 → DINO CLS + 4 Storage Tokens + Patch Tokens
 → 24 DINOv3 Blocks
 → LayerNorm
 → 仅取归一化后的 Patch Tokens
```

DINO 内部的 CLS Token 和 4 个 Storage Tokens 不传入后续 Aggregator。后续
使用的是 VGGT-Ω 自己定义的 Camera Token 和 Register Tokens。

DINOv3 与 Aggregator 都使用二维 RoPE（Rotary Position Embedding）编码 Patch
在图像中的空间位置；Camera/Register 前缀不应用 RoPE。

## 5. Register Token

### 5.1 初始化

代码定义：

```python
self.camera_token = nn.Parameter(torch.empty(1, 2, 1, 1024))
self.register_token = nn.Parameter(torch.empty(1, 2, 16, 1024))
```

第二个维度为 2，并不是两个时间状态，而是两套可学习初值：

- 第 0 套只用于参考帧（序列中的第一帧）；
- 第 1 套由其余所有帧共享。

展开后，每帧都有独立的 1 个 Camera Token 和 16 个 Register Tokens：

```text
Frame 0: [Camera_ref, 16 Registers_ref, P Patches]
Frame 1: [Camera_other, 16 Registers_other, P Patches]
Frame 2: [Camera_other, 16 Registers_other, P Patches]
...
```

非参考帧的初始值相同，但经过图像内容和注意力更新后会成为不同表示。这种设计
只标识“参考帧/其他帧”，没有帧序号嵌入，因此除参考帧外保持帧置换等变性
（Permutation Equivariance），可以处理可变数量和不同顺序的输入帧。

### 5.2 Register 保存什么

Register 没有单独的场景标签或显式重建监督。它们通过相机、深度、点图和匹配
等整体训练目标，逐层学习汇聚对多视图几何有用的信息。

论文用两项实验证明 Register 不只是计算辅助 Token：

- 将冻结的 Scene Tokens 添加到 VLA 模型输入后，LIBERO 平均成功率提升；
- 只让语言读出头读取 Camera/Register Tokens，而不读取 Patch Tokens，也能
  学到与场景描述对齐的序列表示。

因此它们既包含几何信息，也包含一定的场景布局、物体和语义信息。

## 6. 交替注意力

每个 Aggregator 层都包含两个独立 Transformer Block：

```text
Frame Block → Inter-frame Block
```

### 6.1 Frame Attention

对每一帧独立执行完整 Self-Attention：

```text
[Camera_i, Registers_i, Patches_i]
                 │
                 ▼
       帧内完整 Self-Attention
```

这一步完成两个方向的信息传递：

- Patch Tokens 将当前帧的图像和几何信息写入 Camera/Register Tokens；
- Camera/Register Tokens 将已聚合的信息重新广播到当前帧 Patch Tokens。

其复杂度约为：

\[
O\left(F(P+17)^2\right)
\]

### 6.2 Global Attention

普通跨帧层将所有帧的所有 Token 展平后做 Self-Attention：

```text
[Frame 0 所有 Tokens, ..., Frame F-1 所有 Tokens]
                         │
                         ▼
                  Global Attention
```

任意帧的 Patch 可以直接关注其他帧的 Patch，但复杂度约为：

\[
O\left((F(P+17))^2\right)
\]

这是原 VGGT 的主要显存和计算瓶颈。

### 6.3 Register Attention

在指定层中，代码只取所有帧的 Camera/Register Tokens：

```text
[Camera_0, Registers_0, ..., Camera_F-1, Registers_F-1]
                              │
                              ▼
                    Register Attention
```

Patch Tokens 在该跨帧步骤中保持不变。其复杂度约为：

\[
O\left((17F)^2\right)
\]

随后进入下一层 Frame Attention，更新后的 Register 再把跨帧信息传回各帧
Patch Tokens。因此信息路径是：

```text
Frame A Patches
 → Frame A Registers
 → 跨帧 Register Attention
 → Frame B Registers
 → Frame B Patches
```

Register Attention 是跨帧通信瓶颈，而不是删掉 Patch Tokens，也不是把全部
历史压缩成一个固定的全局状态。

### 6.4 两类跨帧注意力如何组合

24 层中，本地代码在索引 `2、6、9、14、20` 使用 Register Attention，其余
19 层仍使用 Global Attention：

```text
Layer 0  Global
Layer 1  Global
Layer 2  Register
Layer 3  Global
...
Layer 20 Register
Layer 23 Global
```

所以 VGGT-Ω 是混合设计。论文实验指出：

- 用 Register Attention 替换约四分之一 Global Attention，可以明显降低显存
  和 FLOPs，且基本不损失精度；
- 如果 24 层全部改成 Register Attention，FLOPs 可降到原来的约 6%，但精度会
  下降到原 VGGT 水平。

## 7. 多层特征缓存

Aggregator 在层 `4、11、17、23` 保存特征。每个保存点不是单一 1024 维输出，
而是拼接：

```text
当前层 Frame Attention 后特征（1024）
             +
当前层 Inter-frame Attention 后特征（1024）
             ↓
          2048 维
```

这同时保留了局部帧内表示和跨帧聚合表示。最后一层特征供 Camera Head 与
Text Alignment Head 使用；四个尺度共同供 Dense Head 使用。

## 8. 预测头

### 8.1 Camera Head

Camera Head 只读取每帧的 Camera Token 和 16 个 Registers，不读取 Patch：

```text
所有帧 Camera/Register Tokens
 → 4 层 Self-Attention
 → 取每帧 Camera Token
 → MLP
 → 9D Camera Encoding
```

9 维相机编码包括：

- Translation：3 维；
- Quaternion Rotation：4 维；
- Vertical/Horizontal Field of View：2 维。

旋转四元数在转换为相机矩阵时归一化，两个 FOV 通过正值激活约束。

### 8.2 Dense Head

Dense Head 只读取四个缓存层中的 Patch Tokens：

1. 将四层 2048 维特征分别投影到不同通道数；
2. 调整到四种空间尺度；
3. 使用 DPT 风格的低分辨率卷积融合模块逐级融合；
4. 在 1/4 分辨率通过轻量预测层输出；
5. 使用 Pixel Shuffle 恢复到原始分辨率；
6. 分别输出 Depth 和 Depth Confidence。

深度激活为：

\[
D=\exp(D_{\mathrm{logit}})
\]

置信度激活为：

\[
C=1+\exp(C_{\mathrm{logit}})
\]

推理时 Dense Head 可以按 8 帧分块解码，以降低预测头的峰值显存；Aggregator
仍一次处理所有输入帧。

### 8.3 Text Alignment Head

可选文本对齐头增加一个可学习 Language Token，将其与所有帧 Camera/Register
Tokens 拼接，经 4 层 Self-Attention 后读取 Language Token，并投影、归一化为
序列级语言对齐特征。它完全不读取 Patch Tokens。

## 9. 训练目标与公开代码边界

论文训练使用：

\[
\mathcal{L}=
\lambda_{\mathrm{cam}}\mathcal{L}_{\mathrm{cam}}+
\lambda_{\mathrm{depth}}\mathcal{L}_{\mathrm{depth}}+
\lambda_{\mathrm{point}}\mathcal{L}_{\mathrm{point}}+
\lambda_{\mathrm{match}}\mathcal{L}_{\mathrm{match}}
\]

推理网络只保留相机头和单个深度头。Point Map 与 Matching 仍在训练中提供辅助
监督，但不需要保留对应的高开销推理头。

当前公开仓库是最小推理实现，没有训练入口、数据管道及这些辅助损失代码，因此
可以完整分析前向网络，不能从仓库独立复现论文训练。

## 10. 对 NAV Stage One 的意义

VGGT-Ω Register 的可借鉴点：

- 使用少量可学习 Token 汇聚高分辨率 Patch 信息；
- 让跨帧通信经过固定数量的语义/几何瓶颈；
- 交替执行“局部写入/读取”和“跨帧汇聚”；
- Register 可以被下游任务直接读取，而不只是注意力中的临时变量；
- 为参考帧与后续帧使用不同初始化，有利于建立坐标锚点。

不能直接照搬的部分：

- VGGT-Ω 是双向、非因果、整段前馈模型，可以看到所有输入帧；
- Register 数量是“每帧 16 个”，总 Register 数仍随帧数线性增长；
- 每次 `forward` 都从可学习初值重新初始化，没有跨调用持久状态；
- 19/24 个跨帧层仍执行所有 Patch 的 Global Attention；
- 它解决的是多视图重建的训练效率问题，不是无限视频的在线记忆问题。

NAV 若要实现真正的流式 Register Memory，需要额外设计：

```text
上一 Chunk 的固定全局 Registers
            +
当前 Chunk 的 Patch/Latent Tokens
            ↓
因果或受限注意力
            ↓
更新后的固定全局 Registers + 当前 Chunk 输出
```

也就是说，应借鉴 VGGT-Ω 的“Register 作为信息瓶颈和可读写场景表示”，但将
其每帧、双向、整段处理机制改造成跨 Chunk 持久化、因果且固定总预算的状态。

## 11. 代码与论文表述差异

- 论文正文称约 25% 的 Global Attention 被 Register Attention 替换；当前
  1B 代码实际为 5/24 层，即约 20.8%。
- 论文公式将 Register Attention 表述为只有 Scene Registers 参与；当前代码
  实际让 Camera Token 和 Registers 一起参加跨帧注意力。
- 论文称高分辨率部分使用 MLP 加 Pixel Shuffle；代码中的最终 1×1 Conv 与
  逐像素 Linear 等价，同时仍保留低分辨率 DPT 卷积融合，这与论文后文说明
  一致。

这些差异不改变总体机制，但在复现或移植 Register Attention 时应以实际代码为
准。
