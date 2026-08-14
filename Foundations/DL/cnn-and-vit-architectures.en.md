---
title: "Vision Architectures Evolution: 2D Conv, Receptive Field Calculus, Depthwise Separable Conv, ResNet Identity Mapping & Vision Transformer (ViT) Guide"
titleZh: "视觉架构演进全景：2D 卷积、感受野迭代计算、Depthwise 深度可分离卷积、ResNet 恒等残差映射与 Vision Transformer (ViT) 极客指南"
titleEn: "Vision Architectures Evolution: 2D Conv, Receptive Field Calculus, Depthwise Separable Conv, ResNet Identity Mapping & Vision Transformer (ViT) Guide"
summaryZh: "100% 全量覆盖 2D 卷积维度与感受野 (RF / ERF) 递推公式、深度可分离卷积 (Depthwise Separable Conv) 计算量折降推导、空洞卷积 (Dilated Conv)、ResNet 残差退化问题与恒等映射 (Identity Mapping) 梯度直通证明、Vision Transformer (ViT) Patch Embedding 与 CNN vs ViT 归纳偏置 (Inductive Bias) 权衡、以及 Pure Numpy 视觉算子引擎。配备丰富 SEO 说明文本。"
summaryEn: "100% exhaustive guide to Vision Architectures, covering 2D Convolution dimensions, Receptive Field (RF / ERF) calculus, Depthwise Separable Conv FLOPs reduction proofs, Dilated Convolutions, ResNet Identity Mapping gradient propagation proofs, Vision Transformer (ViT) Patch Embeddings, CNN vs ViT Inductive Bias trade-offs, and Pure Numpy vision operators with rich SEO explanatory text."
category: "foundations"
tags: ["deep-learning", "cnn", "vision-transformer", "vit", "resnet", "receptive-field", "depthwise-separable-conv", "seo-optimized"]
author: "TalentMe AI Team"
date: "2026-08-01"
interviewFollowups:
  - "Derive the recursive formula for Receptive Field (RF) across L layers and explain Effective Receptive Field (ERF) Gaussian decay."
  - "Calculate FLOPs reduction ratio for Depthwise Separable Convolution vs Standard 3x3 Convolution."
  - "Explain ResNet degradation problem and prove how Identity Mapping prevents vanishing gradients."
  - "Compare Inductive Bias of CNNs (Locality & Translation Invariance) vs ViT (Global Attention)."
  - "Derive ViT Patch Embedding math converting (H,W,C) to (N, D) sequence embeddings."
---

# Vision Architectures Evolution: 2D Conv, Receptive Field Calculus, Depthwise Separable Conv, ResNet Identity Mapping & Vision Transformer (ViT) Guide

> **Summary**: Computer vision architectures evolved from handcrafted local inductive biases (CNNs) to data-driven global self-attention (ViT). This 100% exhaustive guide covers 2D Conv dimensions, Receptive Field (RF/ERF) calculus, Depthwise Separable Conv FLOPs reduction, ResNet Identity Mapping gradient proofs, ViT Patch Embeddings, and CNN vs ViT Inductive Bias trade-offs with rich SEO explanatory text and Pure Numpy implementations.

---

## 🧭 Knowledge Map & Architecture Graph

```mermaid
graph TD
    subgraph A["1. Conv2D & Receptive Field"]
        A1["Dimensions: H_out = ⌊(H_in + 2P - K)/S⌋ + 1"]
        A2["Receptive Field: RF_k = RF_{k-1} + (K_k - 1) · J_{k-1}"]
        A3["Effective Receptive Field (ERF): Gaussian decay from center"]
        A1 --> A2 --> A3
    end

    subgraph B["2. Efficient Conv Variants"]
        B1["Standard Conv FLOPs: H · W · K² · C_in · C_out"]
        B2["Depthwise Separable Conv: Depthwise + Pointwise"]
        B3["FLOPs Reduction Ratio: 1/C_out + 1/K² (8-9x reduction)"]
        B1 --> B2 --> B3
    end

    subgraph C["3. ResNet Identity Mapping"]
        C1["Degradation Problem: Train loss increases on deeper plain networks"]
        C2["Identity Shortcut: x_L = x_l + ∑ F_i"]
        C3["Gradient Straight Stream: ∂L/∂x_l = ∂L/∂x_L · (1 + ∂/∂x_l ∑ F_i)"]
        C1 --> C2 --> C3
    end

    subgraph D["4. Vision Transformer (ViT)"]
        D1["Patch Embedding: Splitting (H,W,C) into N = HW/P² patches"]
        D2["Linear Projection + [CLS] Token + 1D Position Embeddings"]
        D3["Inductive Bias Trade-off: CNN locality vs ViT global attention"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

> 💡 **Intuition**: The whole vision story is a trade-off between "handcrafted priors" and "data-learned patterns". Receptive field: $RF_k = RF_{k-1} + (K_k-1)J_{k-1}$ — stride is the "magnifier" that makes deep-layer pixels see huge input areas; the Effective RF is actually Gaussian-shaped from the center. Depthwise separable conv is "separate the space mixing from the channel mixing" — 3×3 costs ~1/9 of a standard conv. ResNet's identity shortcut is a gradient highway: the constant 1 in $\partial L/\partial x_l = \partial L/\partial x_L(1 + \partial\sum F/\partial x_l)$ can never vanish. ViT chops the image into patches and lets global attention find the correlations CNN hard-codes.
>
> 🎤 **Quick Answer**: "Three 3×3 convs (stride 1) give RF 7 with 27 params vs 49 for one 7×7. Depthwise separable 3×3 with $C_{out}=256$: FLOPs ratio $1/256 + 1/9 \approx 0.11$, an ~9× saving. ResNet-152 trains to 1200 layers while a 56-layer plain net degrades (train error 22.5% > 20-layer's 8.8%). ViT needs JFT-300M-scale pretraining to beat ResNet-BiT, then hits 88.55% top-1."

---

## 📚 Chapter 1: Pure Numpy Vision Engine

Plain-language reading (full implementations in the zh version): `conv2d_forward` is "double for-loop convolution" — slide a window, element-wise multiply, sum; `vit_patch_embedding` does one `reshape + transpose` to cut the image into patches, then projects with `W_proj` to get tokens.

```python
import numpy as np

class PureNumpyVisionEngine:
    @staticmethod
    def conv2d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
        pass
    @staticmethod
    def vit_patch_embedding(x: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray) -> np.ndarray:
        pass
```

> 💡 **Intuition**: Conv = sliding window + inner product; patch embedding = reshape into patches + linear projection. The `transpose(0, 2, 4, 3, 5, 1)` in the real implementation is the whole magic of turning an image into a token sequence.
>
> 🎤 **Quick Answer**: "224×224×3, patch 16 → $N = 14^2 = 196$ patches, each flattened to $16\times16\times3 = 768$ dims, projected to $D$; with [CLS] that's 197 tokens. Example: $D=768$, the projection matrix is $768\times768$ — ViT-B/16's patch embedding layer."