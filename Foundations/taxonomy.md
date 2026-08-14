---
title: "TalentMe AI/ML/LLM 全景技术拓扑图与知识树 (Full AI Knowledge Taxonomy Graph)"
titleZh: "TalentMe AI/ML/LLM 全景技术拓扑图与知识树"
titleEn: "TalentMe AI/ML/LLM Full Knowledge Taxonomy & Architecture Graph"
summaryZh: "全量整合 AI 全领域 (ML, DL, LLM, Multimodal, Agentic, System Design, Math) 7 大方向的细粒度知识拓扑网络。包含交互式 Mermaid 结构图、算法演进脉络、数理公式闭环与大模型工程全景。"
summaryEn: "Comprehensive knowledge taxonomy and architecture graph covering 7 key AI domains: ML, DL, LLM, Multimodal, Agentic, System Design, and Math with interactive Mermaid flowcharts."
category: "foundations"
tags: ["taxonomy", "knowledge-graph", "machine-learning", "deep-learning", "llm", "multimodal", "agentic", "system-design", "ai-math"]
author: "TalentMe AI Team"
date: "2026-08-02"
---

# 🌐 TalentMe AI/ML/LLM 全景技术拓扑图与知识树 (Full Taxonomy)

> **导读与全景视野**：现代人工智能与大模型技术已经演进为一个极其庞大、跨学科且数理严密的工程科学体系。从最底层的线性代数与概率卡方分布，到经典机器学习的多项式回归与 SVM，从深度学习的卷积与自注意力机制，到千亿大语言模型 (LLM)、多模态生成、具身智能 Agent 以及 GPU 分布式并行系统。本文档作为 TalentMe 前端 Tech Vault 的总纲性技术拓扑图，将 7 大核心子领域的知识图谱全量展开，为算法工程师、系统架构师与求职者提供全方位的知识盲区导航与数理索引。

---

## 📊 1. 机器学习算法与数理基础 (Machine Learning Taxonomy)

### 1.1 ML 全景结构 Mermaid 图

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

## 🧠 2. 深度学习基础与网络架构 (Deep Learning Taxonomy)

### 2.1 DL 全景结构 Mermaid 图

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

## ⚡ 3. 大语言模型体系 (Large Language Models Taxonomy)

### 3.1 LLM 全景结构 Mermaid 图

```mermaid
graph TD
    subgraph A["1. Transformer 基础架构 (Transformer Core)"]
        A1["注意力机制: Scaled Dot-Product, Multi-Head Attention (MHA)"]
        A2["显存优化注意力: Multi-Query Attention (MQA), Grouped-Query Attention (GQA)"]
        A3["位置编码: Absolute 1D, Relative Position, RoPE (旋转位置编码)"]
        A4["长文本注意力: FlashAttention 1/2/3 (Tiling 算子融合与重计算), BigBird"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph B["2. 分词与微调技术 (Tokenizer & PEFT)"]
        B1["分词算法: BPE (Byte-Pair Encoding), WordPiece, Unigram, SentencePiece"]
        B2["生成采样: Temperature 缩放, Top-k, Top-p (Nucleus), Min-P 采样, Repetition Penalty"]
        B3["高效微调 (PEFT): LoRA (W = W₀ + B·A), QLoRA (NF4 量化), Prefix Tuning, Adapter"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 偏好对齐与推理慢思考 (Alignment & Reasoning)"]
        C1["RLHF 3 阶段: SFT → 奖励模型 (RM) → PPO 策略梯度 (Clipped Loss + GAE)"]
        C2["直接偏好优化: DPO (隐式奖励代换), ORPO, SimPO, IPO"]
        C3["推理慢思考 (Reasoning): DeepSeek-R1 (纯 RL 自进化), GRPO (Group Relative), 长 CoT 蒸馏"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 前沿架构、量化与真实性 (MoE, Compression & Factuality)"]
        D1["混合专家 (MoE): Top-k 门控路由, Auxiliary Loss, DeepSeek MLA (低秩 KV) & MTP"]
        D2["量化压缩: INT8/INT4, GPTQ (Hessian 逆矩阵), AWQ (显著通道保护), SmoothQuant"]
        D3["SOTA 架构演进: LLaMA 1/2/3, Qwen 2.5/3, DeepSeek-V3, Claude 4, GPT-4o"]
        D4["幻觉与真实性: Intrinsic/Extrinsic, FActScore, RAGAS 评估, RoPE (PI/NTK/YaRN) 扩展"]
        D1 --> D2 --> D3 --> D4
    end

    A --> B --> C --> D
```

---

## 🎮 4. 强化学习与序列决策 (Reinforcement Learning Taxonomy)

### 4.1 RL 全景结构 Mermaid 图

```mermaid
graph TD
    subgraph A["1. 强化学习数理底座 (RL Foundations)"]
        A1["MDP 体系: 状态 S, 动作 A, 转移 P(s'|s,a), 标量奖励 R, 折扣因子 gamma"]
        A2["Bellman 方程: 期望方程 V^pi(s), Q^pi(s,a) 与最优方程 V*(s), Q*(s,a)"]
        A3["策略与价值迭代: Dynamic Programming, 策略评估 -> 策略改进"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 经典与深度 RL (Value & Policy Based)"]
        B1["Value-Based: Q-Learning, SARSA, DQN, Double DQN, Dueling DQN, Rainbow"]
        B2["Policy Gradient: REINFORCE, 策略梯度定理, Baseline 方差削减"]
        B3["Actor-Critic 范式: A2C/A3C, GAE  generalized advantage, TRPO, PPO (Clipped Loss)"]
        B4["连续控制与 Model-Based: DDPG, SAC (Soft Actor-Critic 熵最大化), TD3, MBPO, World Models"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph C["3. 老虎机与在线决策 (Bandits & Online Decision)"]
        C1["MAB 基础: Exploration vs Exploitation 权衡, epsilon-Greedy, UCB, Thompson Sampling"]
        C2["Contextual Bandits: LinUCB, LinRel, 在线新闻/搜索/推荐系统个性化"]
        C3["高维决策: Slate Bandits (列表决策), Combinatorial Bandits (组合决策)"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 智能体 RL 与推理搜索 (Agentic RL & Reasoning)"]
        D1["Agentic RL: 轨迹优化, 动作空间拓展, Hindsight 引导自蒸馏"]
        D2["MCTS 树搜索: Monte Carlo Tree Search, AlphaGo / OpenAI o1 / DeepSeek-R1 搜索演进"]
        D3["过程监督与可验证奖励: PRM (过程监督) vs ORM (结果监督), RLVR (可验证奖励) 训练"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 👁️ 5. 多模态大模型与生成式 AI (Multimodal Taxonomy)

### 4.1 Multimodal 全景结构 Mermaid 图

```mermaid
graph TD
    subgraph A["1. 多模态表征与跨模态对齐 (Multimodal Alignment)"]
        A1["CLIP 架构: Dual-Tower (Vision ViT + Text Encoder) 双塔结构"]
        A2["对比学习损失: InfoNCE Loss, 超参数 Temperature 缩放, Cross-Modal Similarity"]
        A3["Zero-Shot 迁移: Prompt Template 构造与开放词表分类"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 视觉语言大模型 (VLM & Vision-Language)"]
        B1["VLM 经典三段式: Visual Encoder + Linear/Cross-Attention Projector + LLM Backbone"]
        B2["主流代表架构: LLaVA (两阶段预训练+SFT), Visual ChatGPT, Qwen-VL, DeepSeek-Janus"]
        B3["解耦与统一范式: 语言-视觉统一自回归生成 vs 投影表征接入"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 扩散模型与图像/视频生成 (Diffusion & Generative)"]
        C1["DDPM 数理推导: 正向加噪 Markov 链, 反向去噪 U-Net / DiT 预测器"]
        C2["Latent Diffusion (LDM): VAE 潜空间压缩 + Condition 交叉注意力 (Stable Diffusion)"]
        C3["Native 图像与视频生成: GPT-4o Native Image, Veo3 / Sora 视频生成, Hypernetworks"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 语音、世界模型与具身智能 (Audio & World Models)"]
        D1["语音处理: 梅尔声谱图 (Log-Mel Spectrogram), Whisper 弱监督 Encoder-Decoder"]
        D2["世界模型 (JEPA): Yann LeCun 非生成式联合嵌入预测架构 (I-JEPA / V-JEPA)"]
        D3["具身智能 (Embodied AI): VLA (Vision-Language-Action) 机器人操纵与决策"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 🤖 5. 智能体系统与 RAG 工程 (Agentic & RAG Taxonomy)

### 5.1 Agentic 全景结构 Mermaid 图

```mermaid
graph TD
    subgraph A["1. 检索增强生成 pipeline (RAG Systems)"]
        A1["Chunking 策略: Fixed-size, Sentence-level, Parent-Document Chunking"]
        A2["混合检索: BM25 稀疏关键字 + Dense Vector 稠密向量混合检索"]
        A3["Advanced RAG: RRF (Reciprocal Rank Fusion) 倒数秩融合, HyDE 假设性文档嵌入, Cross-Encoder Re-ranking"]
        A1 --> A2 --> A3
    end

    subgraph B["2. 向量数据库与检索索引 (Vector DB & ANN)"]
        B1["向量 Embeddings: OpenAI Text-Embedding-3, Gemini Embeddings, BGE-M3"]
        B2["ANN 近似最近邻检索: IVF 倒排索引, PQ 乘积量化, HNSW (多层可扩展小世界图)"]
        B3["生产级 Vector DB: Milvus, Pinecone, Qdrant, PGVector, 标量过滤 (Scalar Filtering)"]
        B1 --> B2 --> B3
    end

    subgraph C["3. Agent 设计模式与工作流 (Agentic Patterns)"]
        C1["经典模式: ReAct (Reasoning + Acting) 循环, Reflection 自自我反思, Plan-and-Execute"]
        C2["图控制与多 Agent: LangGraph / AutoGPT 状态图图工程, Human-in-the-loop"]
        C3["前沿 Agent: Computer Control 电脑控制, AutoResearch 自主科学研究, Claude Code"]
        C1 --> C2 --> C3
    end

    subgraph D["4. 工具调用与 LLM 评估 (Tool Use & Evaluation)"]
        D1["Tool Use: Toolformer 自动 API 提示生成, Function Calling JSON Schema, 沙箱隔离"]
        D2["LLM-as-a-Judge: Single-Answer, Pairwise 对比评估, Position/Verbosity 偏差消除"]
        D3["基准测试: MMLU, GSM8K, HumanEval, SWE-bench, LiveCodeBench"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 🏗️ 6. AI 系统架构与分布式并行 (System Design Taxonomy)

### 6.1 System Design 全景结构 Mermaid 图

```mermaid
graph TD
    subgraph A["1. GPU 硬件架构与算术强度 (GPU & Hardware)"]
        A1["硬件物理结构: Streaming Multiprocessor (SM), Shared Memory, L1/L2 Cache, HBM 显存"]
        A2["Tensor Cores: 混合精度矩阵乘法 MMA / WGMMA 指令集 (FP16/BF16/FP8)"]
        A3["性能分析模型: Roofline Model (算术强度 FLOPs/Byte, Memory-bound vs Compute-bound)"]
        A1 --> A2 --> A3
    end

    subgraph B["2. LLM 推理显存与加速 (Inference & KV Cache)"]
        B1["KV Cache 显存公式: Bytes = 2 × 2 × n_layers × n_heads × d_head × seq_len × batch_size"]
        B2["PagedAttention: vLLM 逻辑-物理块映射 (消除显存碎片化), Block Manager"]
        B3["推理加速技术: Speculative Decoding (草稿小模型验证), Chunked Prefill, Prefix Caching"]
        B1 --> B2 --> B3
    end

    subgraph C["3. 分布式并行训练 (Distributed Parallelism)"]
        C1["数据并行: DDP (Distributed Data Parallel), AllReduce 梯度同步"]
        C2["张量并行 (TP): Megatron-LM 列并行 (ColumnParallel) 与行并行 (RowParallel)"]
        C3["流水线并行 (PP): 1F1B (One Forward One Backward) 调度算法, Bubble 时间缩减"]
        C4["DeepSpeed ZeRO: ZeRO-1 (Optimizer State), ZeRO-2 (Gradient), ZeRO-3 (Parameter)"]
        C1 --> C2 --> C3 --> C4
    end

    subgraph D["4. 高并发部署与 MLOps (Deployment & MLOps)"]
        D1["流式服务协议: HTTP SSE (Server-Sent Events) 打字机推送 vs WebSocket vs gRPC"]
        D2["语义缓存与 Runtime: Semantic Cache (Exact + Vector Match), TensorRT-LLM, ONNX Runtime"]
        D3["MLOps & 监控: Data Drift 漂移监控, A/B 测试, CUPED 方差降低, Tensorboard"]
        D1 --> D2 --> D3
    end

    A --> B --> C --> D
```

---

## 📐 7. AI 数理基础与泛化理论 (Math & Theory Taxonomy)

### 7.1 Math 全景结构 Mermaid 图

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

---

## 🚀 总结与全景图谱应用路线

本知识拓扑图（Taxonomy）全面覆盖了 AI 与大模型领域的全量底层概念与工程实践。无论是进行系统的基础知识复习、定位特定模态模型的计算边界，还是准备高端算法面试，都可以顺着以上 7 大 Mermaid 拓扑主干快速检索对应的深度技术指南！
