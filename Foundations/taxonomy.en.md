---
title: "TalentMe AI/ML/LLM Full Knowledge Taxonomy & Architecture Graph"
titleZh: "TalentMe AI/ML/LLM 全景技术拓扑图与知识树"
titleEn: "TalentMe AI/ML/LLM Full Knowledge Taxonomy & Architecture Graph"
summaryZh: "全量整合 AI 全领域 (ML, DL, LLM, Multimodal, Agentic, System Design, Math) 7 大方向的细粒度知识拓扑网络。包含交互式 Mermaid 结构图、算法演进脉络、数理公式闭环与大模型工程全景。"
summaryEn: "Comprehensive knowledge taxonomy and architecture graph covering 7 key AI domains: ML, DL, LLM, Multimodal, Agentic, System Design, and Math with interactive Mermaid flowcharts."
category: "foundations"
tags: ["taxonomy", "knowledge-graph", "machine-learning", "deep-learning", "llm", "multimodal", "agentic", "system-design", "ai-math"]
author: "TalentMe AI Team"
date: "2026-08-02"
---

# 🌐 TalentMe AI/ML/LLM Full Knowledge Taxonomy & Architecture Graph

> **Overview & Vision**: Modern Artificial Intelligence and Large Language Models have evolved into a massive, interdisciplinary, and mathematically rigorous engineering science. From fundamental linear algebra and chi-square distributions to classical ML (Polynomial Regression, SVM), deep learning activations and self-attention, to 100B+ LLMs, multimodal generation, embodied agents, and GPU distributed parallel systems. This document serves as the master taxonomy guide for the TalentMe Tech Vault, mapping out 7 core domains with interactive flowcharts and mathematical formulas.

---

## 📊 1. Machine Learning Taxonomy

### 1.1 ML Architecture Graph

```mermaid
graph TD
    subgraph A["1. Supervised Learning - Linear & Probabilistic"]
        A1["Linear Regression: OLS Normal Equation, Ridge (L2), Lasso (L1), VIF"]
        A2["Logistic Regression: Log-Odds Ratio, Sigmoid, MLE Cross-Entropy"]
        A3["SVM: Geometric Margin, Soft Margin C, KKT Conditions, RBF Kernel"]
        A4["Probabilistic Models: Naive Bayes (Laplace), HMM (Viterbi), CRF"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B["2. Decision Trees & Ensembles"]
        B1["Decision Trees: CART (Gini), ID3/C4.5 (Gain Ratio), Pruning"]
        B2["Bagging: Random Forest, Out-of-Bag (OOB) Evaluation"]
        B3["Boosting: GBDT (Negative Gradient), XGBoost (2nd Taylor), LightGBM"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Unsupervised Learning & Dimensionality Reduction"]
        C1["Clustering: K-Means (K-Means++ Init), DBSCAN, GMM (EM)"]
        C2["Distance & Neighbors: KNN (KD-Tree / Ball-Tree), Euclidean/Cosine"]
        C3["Dimensionality Reduction: PCA (SVD), t-SNE, UMAP"]
        C1 --> C2 --> C3
    end

    subgraph D["4. Metrics & Preprocessing"]
        D1["Metrics: Precision, Recall, F1/F-beta, ROC-AUC, PR-AUC"]
        D2["Sampling: SMOTE Oversampling, Focal Loss, Platt Scaling"]
        D3["Cross Validation: K-Fold, Stratified K-Fold, TimeSeriesSplit"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 🧠 2. Deep Learning Taxonomy

### 2.1 DL Architecture Graph

```mermaid
graph TD
    subgraph A["1. DL Foundations"]
        A1["Autograd & Backprop: Chain Rule & Computational Graphs"]
        A2["Activations: Sigmoid → Tanh → ReLU → GELU → SwiGLU"]
        A3["Losses: MSE, BCE, Cross-Entropy, Focal Loss, InfoNCE, Triplet"]
        A1 --> A2 --> A3
    end

    subgraph B["2. Normalization & Regularization"]
        B1["Normalization: BatchNorm, LayerNorm, GroupNorm"]
        B2["LLM Normalization: RMSNorm (Root Mean Square), Pre-LN vs Post-LN"]
        B3["Regularization: Inverted Dropout, L1/L2 Weight Decay, Max-Norm"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Core Architectures"]
        C1["Vision: Conv2D, Receptive Field, Depthwise Separable, ResNet, ViT"]
        C2["Sequence: RNN BPTT, LSTM Gating & Additive Path, GRU, xLSTM, Mamba"]
        C3["Graph: Adjacency/Laplacian, MPNN, GCN (Renormalization), GAT"]
        C4["Generative: GAN Minimax, JS Divergence Flaw, WGAN, WGAN-GP"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph D["4. Optimizers & Debugging"]
        D1["Optimizers: SGD → Momentum → RMSprop → Adam → AdamW (Decoupled)"]
        D2["Initialization: Xavier/Glorot (Tanh), Kaiming/He (ReLU) Variance Proofs"]
        D3["Debugging: 4-Step Framework, Overfit Single Batch, Grad-CAM"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## ⚡ 3. Large Language Models (LLM Taxonomy)

### 3.1 LLM Architecture Graph

```mermaid
graph TD
    subgraph A["1. Transformer Core Architecture"]
        A1["Attention: Scaled Dot-Product, Multi-Head Attention (MHA)"]
        A2["KV-Cache Optimization: Multi-Query (MQA), Grouped-Query (GQA)"]
        A3["Positional Encoding: Absolute 1D, Relative, RoPE (Rotary)"]
        A4["Long Context: FlashAttention 1/2/3 (Tiling & Recomputation), BigBird"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B["2. Tokenizer & PEFT"]
        B1["Tokenizers: BPE, WordPiece, Unigram, SentencePiece"]
        B2["Sampling: Temperature, Top-k, Top-p (Nucleus), Min-P Sampling"]
        B3["PEFT: LoRA (W = W₀ + B·A), QLoRA (NF4 4-bit), Prefix Tuning, Adapter"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Alignment & Reasoning"]
        C1["RLHF 3-Stage: SFT → Reward Model (RM) → PPO (Clipped Loss + GAE)"]
        C2["Direct Preference: DPO, ORPO, SimPO, IPO"]
        C3["Reasoning: DeepSeek-R1 (Pure RL), GRPO, Long CoT Distillation"]
        C1 --> C2 --> C3
    end

    subgraph D["4. MoE, Compression & Factuality"]
        D1["MoE: Top-k Routing, Auxiliary Loss, DeepSeek MLA & MTP"]
        D2["Quantization: INT8/INT4, GPTQ (Hessian), AWQ, SmoothQuant"]
        D3["SOTA Evolution: LLaMA 1/2/3, Qwen 2.5/3, DeepSeek-V3, Claude 4, GPT-4o"]
        D4["Hallucination & Factuality: FActScore, RAGAS, RoPE (PI/NTK/YaRN) Context Extension"]
        D1 --> D2 --> D3 --> D4
    end

    A --> B --> C --> D
```

---

## 👁️ 4. Multimodal AI & Generative Models Taxonomy

### 4.1 Multimodal Architecture Graph

```mermaid
graph TD
    subgraph A["1. Multimodal Alignment"]
        A1["CLIP: Dual-Tower (Vision ViT + Text Encoder)"]
        A2["Contrastive Loss: InfoNCE Loss, Temperature Scaling"]
        A3["Zero-Shot Transfer: Prompt Templates & Open-Vocabulary"]
        A1 --> A2 --> A3
    end

    subgraph B["2. Vision-Language Models (VLM)"]
        B1["VLM Pipeline: Visual Encoder + Projector + LLM Backbone"]
        B2["Architectures: LLaVA, Visual ChatGPT, Qwen-VL, DeepSeek-Janus Pro"]
        B3["Unification: Auto-Regressive Text-Image Generation"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Diffusion & Generative Models"]
        C1["DDPM: Forward Markov Noise & Reverse Denoising U-Net / DiT"]
        C2["Latent Diffusion (LDM): VAE Latent Compression + Stable Diffusion"]
        C3["Native Generation: GPT-4o Native Image, Veo3 / Sora Video Gen"]
        C1 --> C2 --> C3
    end

    subgraph D["4. Audio & World Models"]
        D1["Audio: Log-Mel Spectrogram, Whisper Encoder-Decoder"]
        D2["World Models: Yann LeCun JEPA (I-JEPA / V-JEPA) Non-Generative Prediction"]
        D3["Embodied AI: Vision-Language-Action (VLA) Robotics"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 🤖 5. Agentic Systems & RAG Engineering Taxonomy

### 5.1 Agentic Architecture Graph

```mermaid
graph TD
    subgraph A["1. RAG Systems"]
        A1["Chunking: Fixed-size, Sentence-level, Parent-Document Chunking"]
        A2["Hybrid Search: BM25 Sparse + Dense Vector Hybrid Search"]
        A3["Advanced RAG: RRF (Reciprocal Rank Fusion), HyDE, Cross-Encoder Re-ranking"]
        A1 --> A2 --> A3
    end

    subgraph B["2. Vector DB & Indexing"]
        B1["Embeddings: OpenAI Text-Embedding-3, Gemini Embeddings, BGE-M3"]
        B2["ANN Search: IVF Inverted File, PQ Product Quantization, HNSW Graphs"]
        B3["Vector DBs: Milvus, Pinecone, Qdrant, PGVector, Scalar Filtering"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Agent Patterns"]
        C1["Patterns: ReAct (Reason+Act), Self-Reflection, Plan-and-Execute"]
        C2["Graph & Control: LangGraph / AutoGPT State Machines, Human-in-the-loop"]
        C3["Autonomous Agents: Computer Control, AutoResearch, Claude Code"]
        C1 --> C2 --> C3
    end

    subgraph D["4. Tool Use & Evaluation"]
        D1["Tool Use: Toolformer API generation, Function Calling JSON Schema, Sandboxes"]
        D2["LLM-as-a-Judge: Single-Answer, Pairwise Evaluation, Bias Mitigation"]
        D3["Benchmarks: MMLU, GSM8K, HumanEval, SWE-bench, LiveCodeBench"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 🏗️ 6. AI System Architecture & Parallelism Taxonomy

### 6.1 System Design Architecture Graph

```mermaid
graph TD
    subgraph A["1. GPU Hardware Architecture"]
        A1["Hardware Structure: Streaming Multiprocessor (SM), Shared Memory, Cache, HBM"]
        A2["Tensor Cores: Mixed-Precision MMA / WGMMA Matrix Multiplication"]
        A3["Roofline Model: Arithmetic Intensity FLOPs/Byte (Memory vs Compute bound)"]
        A1 --> A2 --> A3
    end

    subgraph B["2. LLM Inference & Memory Acceleration"]
        B1["KV Cache Formula: Bytes = 2 × 2 × n_layers × n_heads × d_head × seq_len × batch_size"]
        B2["PagedAttention: vLLM Logical-Physical Block Mapping"]
        B3["Acceleration: Speculative Decoding, Chunked Prefill, Prefix Caching"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Distributed Training Parallelism"]
        C1["Data Parallel: DDP (Distributed Data Parallel) & AllReduce"]
        C2["Tensor Parallel (TP): Megatron-LM ColumnParallel & RowParallel"]
        C3["Pipeline Parallel (PP): 1F1B Scheduling & Bubble Reduction"]
        C4["DeepSpeed ZeRO: ZeRO-1 (Optimizer), ZeRO-2 (Gradient), ZeRO-3 (Parameter)"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph D["4. High-Concurrency Deployment & MLOps"]
        D1["Streaming Protocols: HTTP SSE (Server-Sent Events) vs WebSocket vs gRPC"]
        D2["Semantic Cache & Runtimes: Semantic Cache, TensorRT-LLM, ONNX Runtime"]
        D3["MLOps: Data Drift Monitoring, A/B Testing, CUPED Variance Reduction"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 📐 7. AI Math Foundations & Generalization Taxonomy

### 7.1 Math Architecture Graph

```mermaid
graph TD
    subgraph A["1. Probability & Information Theory"]
        A1["Bayes Inference: Prior P(θ), Likelihood P(X|θ), Posterior P(θ|X)"]
        A2["Information Theory: Shannon Information I(x), Entropy H(X)"]
        A3["Relative Entropy: KL Divergence D_KL(P || Q) Proof, Cross-Entropy H(P, Q)"]
        A1 --> A2 --> A3
    end

    subgraph B["2. Optimization & Generalization"]
        B1["Inductive Bias: CNN Locality/Translation Invariance vs Transformer Weak Bias"]
        B2["Double Descent: Interpolation Boundary & Over-parameterized Generalization"]
        B3["Learning Paradigms: Supervised, Self-Supervised (Contrastive/MAE), RL"]
        B1 --> B2 --> B3
    end

    A --> B
```