---
title: "TalentMe AI/ML/LLM/Infra 全景技术拓扑图与知识树 (Full AI Knowledge Taxonomy Graph)"
titleZh: "TalentMe AI/ML/LLM/Infra 全景技术拓扑图与知识树"
titleEn: "TalentMe AI/ML/LLM/Infra Full Knowledge Taxonomy & Architecture Graph"
summaryZh: "全量整合 AI 全领域 (ML, DL, LLM, RL, Multimodal, AI Infra, AI Engineering, System Design, Math) 9 大方向的细粒度知识拓扑网络。包含交互式 Mermaid 结构图、算法演进脉络、数理公式闭环与大模型工程全景。"
summaryEn: "Comprehensive knowledge taxonomy and architecture graph covering 9 key AI domains: ML, DL, LLM, RL, Multimodal, AI Infra, AI Engineering, System Design, and Math with interactive Mermaid flowcharts."
category: "foundations"
tags: ["taxonomy", "knowledge-graph", "machine-learning", "deep-learning", "llm", "multimodal", "ai-infra", "ai-engineering", "system-design", "ai-math"]
author: "TalentMe AI Team"
date: "2026-08-06"
---

# 🌐 TalentMe AI 全景技术拓扑图与知识树 (Full Taxonomy)

> **导读与全景视野**：现代人工智能已经形成由数理基础、深度模型、算力基础设施（AI Infra）、应用工程（AI Engineering）与工业级系统案例组成的庞大体系。本文档作为 TalentMe 前端 Tech Vault 的总纲性技术拓扑图，将 9 大核心子领域的知识图谱全量展开。

---

## 📊 1. 机器学习算法与数理基础 (ML Taxonomy)

```mermaid
graph TD
    subgraph A["1. 监督学习 - 线性与概率模型 (Linear & Probabilistic)"]
        A1["线性回归: OLS 正规方程, Ridge (L2), Lasso (L1), VIF 多重共线性"]
        A2["逻辑回归: Log-Odds 几率比, Sigmoid, MLE 交叉熵损失"]
        A3["支持向量机 (SVM): 几何间隔, 软间隔 C, KKT 条件, RBF 高斯核"]
        A4["概率图模型: 朴素贝叶斯 (拉普拉斯平滑), HMM (维特比解码), CRF"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B["2. 树模型与集成学习范式 (Trees & Ensembles)"]
        B1["决策树: CART (Gini 指数), ID3/C4.5 (信息增益比), 剪枝策略"]
        B2["Bagging: 随机森林 (Random Forest), Out-of-Bag (OOB) 评估"]
        B3["Boosting: GBDT (负梯度拟合), XGBoost (二阶泰勒), LightGBM (GOSS/EFB)"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 无监督学习与降维 (Unsupervised & Distance)"]
        C1["聚类: K-Means (K-Means++ 初始化), DBSCAN (密度聚类), GMM (EM 算法)"]
        C2["距离与近邻: KNN (KD-Tree / Ball-Tree 空间索引), 欧氏/余弦/曼哈顿距离"]
        C3["降维: PCA (主成分分析 / SVD 奇异值分解), t-SNE, UMAP"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 评估指标与预处理 (Metrics & Preprocessing)"]
        D1["分类指标: Precision, Recall, F1/F-beta, ROC-AUC, PR-AUC"]
        D2["数据采样与校准: SMOTE 过采样, Focal Loss, 概率校准 (Platt Scaling)"]
        D3["交叉验证: K-Fold, Stratified K-Fold, TimeSeriesSplit"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 🧠 2. 深度学习基础与网络架构 (DL Taxonomy)

```mermaid
graph TD
    subgraph A["1. 深度学习数理底座 (DL Foundations)"]
        A1["计算图与反向传播: Autograd 自动微分, 链式法则, 梯度流追踪"]
        A2["激活函数演进: Sigmoid → Tanh → ReLU → LeakyReLU → GELU → SwiGLU"]
        A3["损失函数族: MSE, BCE, Cross-Entropy, Focal Loss, InfoNCE, Triplet Loss"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 归一化与正则化工程 (Norm & Regularization)"]
        B1["归一化维度: BatchNorm (Batch 维), LayerNorm (Sequence 维), GroupNorm"]
        B2["大模型归一化: RMSNorm (均方根归一化简化), Pre-LN vs Post-LN"]
        B3["正则化: Dropout (Inverted Dropout), L1/L2 Weight Decay, Max-Norm 约束"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 核心网络架构变体 (Architectures)"]
        C1["视觉 (CNN & ViT): Conv2D, 感受野递推, Depthwise Separable Conv, ResNet 残差, ViT"]
        C2["序列 (RNN & SSM): RNN BPTT, LSTM 4 门与加性路径, GRU, xLSTM (mLSTM), Mamba (S6)"]
        C3["图网络 (GNN): 邻接/拉普拉斯矩阵, MPNN 消息传递, GCN (重归一化), GAT (多头图注意力)"]
        C4["生成模型 (GAN): Minimax 零和博弈, JS 散度缺陷, WGAN (Wasserstein 距离), WGAN-GP"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph D["4. 优化器与调试工程 (Optimizers & Debugging)"]
        D1["优化器演进: SGD → Momentum → RMSprop → Adam → AdamW (解耦权重衰减)"]
        D2["权重初始化: Xavier/Glorot (Tanh), Kaiming/He (ReLU) 方差守恒证明"]
        D3["模型调试: 4 步框架, Overfit Single Batch 验证, 数值梯度检查, Grad-CAM 可解释性"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## ⚡ 3. 大语言模型体系 (LLM Taxonomy)

```mermaid
graph TD
    subgraph A["1. Transformer 核心架构 (Core Architecture)"]
        A1["Attention 机制: Scaled Dot-Product, Multi-Head Attention (MHA)"]
        A2["KV-Cache 显存优化: Multi-Query (MQA), Grouped-Query (GQA)"]
        A3["位置编码演进: 绝对 1D 位置编码, 相对位置编码, RoPE 旋转位置编码"]
        A4["长上下文扩展: FlashAttention 1/2/3 (Tiling & 重算), BigBird, Longformer"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B["2. Tokenizer 与 Efficient Fine-Tuning (Tokenizer & PEFT)"]
        B1["分词算法: BPE (Byte-Pair Encoding), WordPiece, Unigram, SentencePiece"]
        B2["采样策略: Temperature 缩放, Top-k, Top-p (Nucleus), Min-P Sampling"]
        B3["高效微调 PEFT: LoRA (W = W₀ + B·A), QLoRA (NF4 4-bit), Prefix Tuning, Adapter"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 对齐与推理进化 (Alignment & Reasoning)"]
        C1["RLHF 3 阶段: SFT 监督微调 → Reward Model 奖励模型 → PPO (Clipped Loss + GAE)"]
        C2["偏好直接对齐: DPO (Direct Preference Optimization), ORPO, SimPO, IPO"]
        C3["推理能力强化: DeepSeek-R1 (Pure RL 纯强化学习), GRPO, 长 CoT 思维链蒸馏"]
        C1 --> C2 --> C3
    end

    subgraph D["4. MoE 架构、量化与真实性 (MoE & Factuality)"]
        D1["MoE 混合专家: Top-k Routing, 辅助 Loss 负载均衡, DeepSeek MLA & MTP"]
        D2["模型量化: INT8/INT4 均匀量化, GPTQ (Hessian 反矩阵), AWQ, SmoothQuant"]
        D3["SOTA 模型演进: LLaMA 1/2/3, Qwen 2.5/3, DeepSeek-V3, Claude 4, GPT-4o"]
        D4["幻觉诊断与上下文: FActScore 原子事实拆解, RAGAS 评估, RoPE 位置插值 (PI/NTK/YaRN)"]
        D1 --> D2 --> D3 --> D4
    end

    A --> B --> C --> D
```

---

## ⚡ 4. AI 基础设施与算力工程 (AI Infra Taxonomy)

```mermaid
graph TD
    subgraph A["1. GPU 硬件架构与算术强度 (GPU & Hardware)"]
        A1["SM 结构: Streaming Multiprocessor, Shared Memory, Tensor Cores"]
        A2["Roofline Model: 算术强度 I = FLOPs / Byte (Memory-Bound vs Compute-Bound)"]
        A3["Kernel 算子融合: RMSNorm + SwiGLU / FlashAttention 1/2/3 SRAM Tiling"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 显存优化与推理加速 (Memory & Acceleration)"]
        B1["KV Cache 精确推导: Bytes = 2 × 2 × n_layers × n_heads × d_head × seq_len × batch_size"]
        B2["vLLM PagedAttention: 逻辑-物理 Block 映射与 Copy-on-Write"]
        B3["Speculative Decoding: Draft Model 小模型采样与拒绝采样概率无损证明"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 4D 分布式并行与拓扑 (4D Parallelism)"]
        C1["数据与张量并行: DDP AllReduce, Megatron-LM 列/行并行"]
        C2["流水线并行 (PP): 1F1B 调度算法与 Bubble 气泡占比缩减"]
        C3["DeepSpeed ZeRO: ZeRO-1/2/3 显存切分与 4D (TP+PP+DP+EP) 拓扑"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 算子编程与集群调度 (Operators & Cluster)"]
        D1["CUDA & Triton 编程: Thread Block 层次、Bank Conflict 消除、Triton JIT"]
        D2["集群调度与 Ray: Ray Core (Tasks/Actors), Slurm, MIG 显存硬隔离, RoCE v2"]
        D1 --> D2
    end

    A --> B --> C --> D
```

---

## 🤖 5. AI 应用工程与 Agent 架构 (AI Engineering Taxonomy)

```mermaid
graph TD
    subgraph A["1. 检索增强生成 (Advanced RAG)"]
        A1["Chunking 策略: Fixed-size, Sentence-level, Parent-Document Chunking"]
        A2["混合检索: BM25 稀疏关键字 + Dense Vector 向量混合检索 (RRF 融合)"]
        A3["Advanced RAG: HyDE 假设性文档嵌入, Cross-Encoder 精准重排序"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 向量数据库与检索 (Vector DB & ANN)"]
        B1["Embeddings: Text-Embedding-3, BGE-M3, Gemini Embeddings"]
        B2["ANN 近似最近邻: IVF 倒排网格, PQ 乘积量化, HNSW 多层小世界图"]
        B3["生产级 Vector DB: Milvus, Pinecone, Qdrant, PGVector 标量过滤"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Agent 模式与工具调用 (Agent Patterns & Tools)"]
        C1["Agent 模式: ReAct 循环, Reflection 自自我反思, Plan-and-Execute"]
        C2["图控制与多 Agent: LangGraph / AutoGPT 状态图工程"]
        C3["Tool Use: Toolformer 自动 API 提示生成, Function Calling JSON Schema, 沙箱"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 评估与安全护栏 (Eval & Guardrails)"]
        D1["LLM-as-a-Judge: Single-Answer, Pairwise 评估, 偏差消除"]
        D2["Guardrails: System Prompts, Structured Outputs (Outlines), Llama Guard 越狱防护"]
        D1 --> D2
    end

    A --> B --> C --> D
```

---

## 🏗️ 6. 工业级系统设计案例 (Industry System Design Taxonomy)

```mermaid
graph TD
    subgraph A["1. 推荐与搜索广告系统 (Recommendation & Search Ads)"]
        A1["推荐系统 3 阶段: 召回 (Retrieval) -> 粗排 -> 精排 (Heavy Ranking) -> 重排 (Diversity)"]
        A2["双塔架构: DSSM User/Item 塔向量化召回, Feature Store 离在线一致性"]
        A3["搜索广告系统: Query 意图理解, 分布式倒排索引, RTB 实时竞价与 pCTR 预估"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 生产级 RAG/Agent & 多模态架构 (Production AI Apps)"]
        B1["大模型 RAG 系统架构: 多租户隔离, HTTP SSE 流式打字机推送, 语义缓存"]
        B2["多模态生成架构: Stable Diffusion / Sora 推理部署, 模型切片装载与 GPU 动态扩缩容"]
        B1 --> B2
    end

    subgraph C["3. 实时风控与业界经典 Case (Risk Control & Cases)"]
        C1["实时风控系统: Flink/Kafka 流批一体, 低延迟规则引擎与图风控"]
        C2["业界 Case Study: Pinterest 推荐系统, Netflix 视频流传输, Uber 实时调度"]
        C1 --> C2
    end

    A --> B --> C
```

---

## 📐 7. AI 数理基础与泛化理论 (Math Taxonomy)

```mermaid
graph TD
    subgraph A["1. 概率统计与信息论 (Probability & Information)"]
        A1["贝叶斯推断: 先验 P(θ), 似然 P(X|θ), 后验 P(θ|X) = P(X|θ)P(θ) / P(X)"]
        A2["信息论基础: 香农信息量 I(x), 自信息, 信息熵 H(X) = -∑ p(x) log p(x)"]
        A3["相对熵与交叉熵: KL 散度 D_KL(P || Q) 非对称性证明, Cross-Entropy H(P, Q)"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 凸优化与学习泛化理论 (Optimization & Generalization)"]
        B1["归纳偏置 (Inductive Bias): CNN 局部性/平移不变性 vs Transformer 弱偏置"]
        B2["Double Descent (双重下降): 插值界限 (Interpolation Boundary), 过参数化泛化"]
        B3["学习范式: 监督学习, 自监督学习 (Contrastive / Masked Autoencoder), 强化学习"]
        B1 --> B2 --> B3
    end

    A --> B
```