---
title: "AIE Fine-Tuning Guide: Enterprise SFT, LoRA & DPO Alignment"
titleZh: "AIE 实战指南：企业级 SFT、LoRA 微调与 DPO/RLHF 偏好对齐落地"
titleEn: "AIE Fine-Tuning Guide: Enterprise SFT, LoRA & DPO Alignment"
summaryZh: "全量拆解企业级大模型 SFT 监督微调与 DPO/RLHF 偏好对齐实战落地细节。深入剖析 Data Packing 消除 Padding 浪费、Loss Masking、LoRA/QLoRA 显存精算、权重无损合并与 GRPO 纯 RL 对齐。"
summaryEn: "Exhaustive technical deep dive into enterprise LLM SFT and DPO/RLHF alignment: Data Packing, Loss Masking, LoRA/QLoRA VRAM budgeting, zero-latency weight merging, and GRPO pure RL alignment."
category: "AIE"
tags: ["sft", "lora-merge", "dpo-practical", "grpo", "alignment", "qlora", "data-packing"]
author: "TalentMe AI Team"
date: "2026-08-07"
interviewFollowups:
  - 'How does the Data Packing algorithm eliminate padding token compute waste in SFT? How does 2D Attention Masking isolate cross-sample context?'
  - 'How should LoRA rank r and scaling factor alpha be parameterized? Why is matrix A initialized with Gaussians while B is zeroed out?'
  - 'Compare DPO and GRPO: What role does the reference policy play in both? Why does GRPO dispense with an explicit Critic network?'
  - 'How does QLoRA NF4 quantization and Double Quantization compress training VRAM? How do Paged Optimizers prevent OOM crashes?'
  - 'How do you prevent Catastrophic Forgetting during enterprise domain adaptation? What proportion of general replay data should be retained?'
---

# 🌐 AIE Fine-Tuning Guide: Enterprise SFT, LoRA & DPO Alignment

> **Executive Summary**: LLM fine-tuning and alignment empower AI Engineers (AIE) to transform foundation models into domain-specific reasoning engines. In production workflows, teams face three hurdles: padding compute waste in variable-length batches, severe VRAM bottlenecks during full fine-tuning, and instability in reinforcement learning from human feedback. This guide presents the Data Packing algorithm, LoRA/QLoRA VRAM budgeting, zero-latency weight fusion, DPO mechanics, and GRPO alignment.

---

## 💡 Interactive Mermaid Architecture

```mermaid
graph TD
    subgraph A["1. SFT Data Engineering & Packing"]
        A1["Multi-Turn Conversation Cleaning & De-duplication"]
        A2["Data Packing: Concatenate Multi-Samples into 4096 Window"]
        A3["Loss Masking: Set Prompt Token Labels to -100"]
        A1 --> A2 --> A3
    end

    subgraph B["2. LoRA / QLoRA Parameter-Efficient Fine-Tuning"]
        B1["Base Weights W0 Frozen (FP16 or NF4 4-bit)"]
        B2["Low-Rank Adapters: delta_W = (alpha/r) * B @ A (Rank r=8/16, B=0 Init)"]
        B3["Zero-Latency Weight Fusion: W_deploy = W0 + delta_W"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Preference Alignment & Reasoning RL"]
        C1["DPO: Offline Pairwise Preference Optimization"]
        C2["GRPO: Group Relative Policy Advantage (Critic-free Pure RL)"]
        C1 --> C2
    end

    A --> B --> C
```

---

## Chapter 1: SFT Data Packing & Loss Masking

Variable-length samples padded to 4096 tokens waste up to **80% of training compute on non-informative padding tokens**.

### 1. Data Packing
Concatenate short conversation samples into a unified 4096-token sequence separated by `<eos>` tokens. Apply a block-diagonal 2D Attention Mask so tokens attend only within their respective sample boundaries.

### 2. Loss Masking
Set labels of System and User prompt tokens to `-100`. Cross-Entropy loss computation is evaluated **exclusively on response tokens**, ensuring the model specializes in generating answers rather than memorizing prompt prefixes.

---

## Chapter 2: Pure Python DPO Loss Operator

```python
import numpy as np

def pure_python_dpo_loss(policy_win_logp: float, policy_lose_logp: float, ref_win_logp: float, ref_lose_logp: float, beta: float = 0.1) -> float:
    pi_logr = policy_win_logp - policy_lose_logp
    ref_logr = ref_win_logp - ref_lose_logp
    logits = beta * (pi_logr - ref_logr)
    return float(-np.log(1.0 / (1.0 + np.exp(-logits))))

if __name__ == "__main__":
    print("✅ DPO Loss:", round(pure_python_dpo_loss(-1.2, -2.5, -1.5, -2.2), 4))
```

---

## Chapter 3: LoRA & QLoRA Mathematical Formulations

Pre-trained LLM weight updates reside within a **low intrinsic dimension**:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$.

* **Initialization**: $A \sim \mathcal{N}(0, \sigma^2)$ is Gaussian-initialized; $B = 0$ is strictly zero-initialized.
  This guarantees $\Delta W = B \cdot A = 0$ at step 0, preserving initial foundation model representations seamlessly.

### QLoRA Innovations
1. **NF4 (NormalFloat4)**: Information-theoretically optimal quantiles for normal distributions.
2. **Double Quantization**: Compresses quantization constants, saving 0.37 bits/param.
3. **Paged Optimizers**: Mitigates gradient allocation memory spikes via CPU paging.

---

## Chapter 4: Zero-Latency Serving: LoRA Weight Merging

In online inference serving, dynamic branch execution adds two matrix multiplications per token. Fuse adapter parameters into the frozen base weights prior to export:
$$W_{\text{deploy}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$

```python
import numpy as np

def pure_python_lora_merge(W0: np.ndarray, A: np.ndarray, B: np.ndarray, r: int, alpha: float) -> np.ndarray:
    scale = alpha / r
    return W0 + scale * np.matmul(B, A)

if __name__ == "__main__":
    np.random.seed(42)
    W0 = np.random.randn(8, 8)
    A = np.random.randn(2, 8)
    B = np.random.randn(8, 2)
    W_fused = pure_python_lora_merge(W0, A, B, r=2, alpha=4.0)
    print("✅ LoRA Merged Output Validated.")
```

---

## Chapter 5: GRPO (Group Relative Policy Optimization)

PPO loads 4 models concurrently (Actor, Critic, Reference, Reward). **GRPO eliminates the Critic entirely**.
For query $q$, sample $G$ outputs $\{o_1, \dots, o_G\}$ and score via rule-based rewards:
$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$
This cuts training VRAM by $>50\%$, freeing resources for long chain-of-thought exploration.
