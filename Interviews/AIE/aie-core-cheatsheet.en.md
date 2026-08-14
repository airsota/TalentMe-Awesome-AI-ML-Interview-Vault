---
title: "AIE Core Cheatsheet: SFT, LoRA, RAG & Agent Interview Map"
titleZh: "AIE 大模型系统工程师核心地图：SFT/LoRA/RAG/Agent 面试必考"
titleEn: "AIE Core Cheatsheet: SFT, LoRA, RAG & Agent Interview Map"
summaryZh: "全量拆解 AIE (AI / LLM Systems Engineer) 岗位核心知识地图与面试高频考点。深入剖析大模型 SFT 数据 Packing、LoRA/QLoRA 高效微调、DPO/GRPO 偏好对齐、Advanced RAG 检索调优、投机采样 Speculative Decoding 与 vLLM 推理吞吐瓶颈。"
summaryEn: "Exhaustive technical deep dive into AIE (AI / LLM Systems Engineer) core knowledge map: SFT Data Packing, LoRA/QLoRA, DPO/GRPO, Advanced RAG, Speculative Decoding, and high-throughput vLLM serving."
category: "AIE"
tags: ["aie", "llm-engineer", "sft", "lora", "rag-interview", "speculative-decoding", "vllm"]
author: "TalentMe AI Team"
date: "2026-08-07"
interviewFollowups:
  - 'How do AIE roles differ fundamentally from traditional MLE positions across core competencies and system scopes?'
  - 'How do you systematically troubleshoot and resolve loss spikes or catastrophic forgetting during LLM SFT fine-tuning?'
  - 'Deconstruct the matrix algebra and memory advantages of LoRA and QLoRA: Why is A Gaussian-initialized while B is zeroed?'
  - 'How to engineer enterprise Advanced RAG with Sentence-Window retrieval, HyDE query expansion, and HNSW vector index tuning?'
  - 'What is Speculative Decoding? Why does it break the memory bandwidth bottleneck (Memory-Bound) in autoregressive generation?'
---

# 🌐 AIE Core Cheatsheet: SFT, LoRA, RAG & Agent Interview Map

> **Executive Summary**: The AI / LLM Systems Engineer (AIE) role spans model fine-tuning, retrieval engineering, agentic orchestration, and high-throughput serving systems. This cheatsheet deconstructs six essential modules: AIE vs. MLE competency matrices, SFT Data Packing with Loss Masking, LoRA/QLoRA VRAM budgeting, Advanced RAG architectures, Speculative Decoding, and production vLLM clusters.

---

## 💡 Interactive Mermaid Architecture

```mermaid
graph TD
    subgraph A["1. Data & Fine-Tuning Pipeline"]
        A1["SFT Data Packing: Deduplication, Quality Scoring, Loss Masking (labels=-100)"]
        A2["PEFT Fine-Tuning: LoRA (Rank 8/16), QLoRA (NF4 4-bit, Double Quant)"]
        A3["Preference Alignment: DPO (Implicit Reward) / GRPO (Group Advantage)"]
        A1 --> A2 --> A3
    end

    subgraph B["2. RAG & Agent Infrastructure"]
        B1["Advanced RAG: Sentence-Window, Parent-Document, HyDE Query Expansion"]
        B2["Hybrid Search: BM25 + Dense Vector RRF Fusion (k=60) -> Cross-Encoder Re-rank"]
        B3["Agent Orchestration: ReAct Loop, LangGraph State Machine, Sandboxed Execution"]
        B1 --> B2 --> B3
    end

    subgraph C["3. High-Throughput Serving"]
        C1["vLLM Engine: PagedAttention (Virtual Memory Paging) + Continuous Batching"]
        C2["Speculative Decoding: Draft Model Acceleration (Breaking Memory-Bound)"]
        C3["Ray Cluster: Multi-GPU Tensor Parallelism (TP=4/8), HTTP SSE Streaming"]
        C1 --> C2 --> C3
    end

    A --> B --> C
```

---

## Chapter 1: AIE vs. MLE Competency Matrix

| Dimension | Traditional MLE (Machine Learning Engineer) | AIE (LLM Systems Engineer) |
| :--- | :--- | :--- |
| **Model Core** | GBDT (XGBoost/LightGBM), DSSM Two-Tower, CNN | LLM (Decoder-Only Transformer), MoE |
| **Feature/Data** | Tabular Feature Stores, One-Hot, Entity Embeddings | Prompt Engineering, SFT Data Packing, Synthetic Data |
| **Fine-Tuning** | Hyperparameter Grid Search, Cross-Validation | LoRA / QLoRA, DPO / GRPO Alignment |
| **Serving** | C++ LibTorch / ONNX / Redis | vLLM, TensorRT-LLM, Ray Serve Clusters |

---

## Chapter 2: Pure Python Prompt Formatting Operator

```python
def pure_python_format_system_prompt(user_query: str, retrieved_docs: list) -> str:
    context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
    return (
        f"System: You are an expert AI assistant. Answer the user query strictly based on the context below.\n"
        f"Context:\n{context}\n\n"
        f"User Query: {user_query}\n"
        f"Answer:"
    )

if __name__ == "__main__":
    docs = ["Paris is the capital of France.", "Population is 2.1 Million."]
    print("✅ Formatted Prompt:\n", pure_python_format_system_prompt("What is the capital of France?", docs))
```

---

## Chapter 3: Advanced RAG Architecture

* **Sentence-Window Retrieval**: Index individual sentences for semantic precision; expand with $\pm 3$ surrounding sentences upon retrieval before feeding to the LLM context.
* **Parent-Document Indexing**: Index smaller chunks (200 tokens) in the vector store; return their full parent document chunks (2000 tokens) to the LLM.
* **Hypothetical Document Embeddings (HyDE)**: Prompt the LLM to hallucinate a tentative answer; vectorize this answer to search real documents, bridging query-document distributional mismatch.

---

## Chapter 4: Speculative Decoding & Overcoming the Memory Wall

Autoregressive token generation is strictly **Memory-Bound**: generating 1 token requires streaming all model weights from HBM to SRAM.

```mermaid
graph LR
    Draft["1. Small Draft Model (1.5B) rapidly generates K candidate tokens"] --> Target["2. Large Target Model (70B) verifies all K tokens in a single parallel forward pass"]
    Target --> Accept{"3. Probability ratio acceptance test"}
    Accept -->|Accepts m tokens| Out["4. Yields m+1 tokens in 1 step (2-3x speedup)"]
```

Acceptance probability: $\alpha = \min\left( 1, \frac{p(x)}{q(x)} \right)$. This mathematically guarantees exact target distribution equivalence with **zero loss in generation fidelity**.

```python
import numpy as np

def pure_python_speculative_verification(p_target: list[float], q_draft: list[float]) -> int:
    accepted = 0
    for p, q in zip(p_target, q_draft):
        ratio = p / q if q > 0 else 0.0
        if ratio >= 1.0 or np.random.rand() < ratio:
            accepted += 1
        else:
            break
    return accepted

if __name__ == "__main__":
    np.random.seed(42)
    print("✅ Speculative Accepted Count:", pure_python_speculative_verification([0.8, 0.7, 0.6, 0.1], [0.75, 0.65, 0.55, 0.4]))
```
