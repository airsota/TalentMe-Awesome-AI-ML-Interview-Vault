# 🚀 AIE Role Track · AI 系统工程师 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **25 cards · 125 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. 100k Concurrency Code Agent Architecture · 10 万并发 Code Agent 系统设计

> The 100k Concurrency Autonomous Code Agent & Isolated Sandbox System Architecture represents the quintessential AIE systems design benchmark, modeling end-to-end autonomous engineering agents (Devin/Cursor workspaces);

**Interview questions**

1. Diagram the end-to-end Code Agent architecture: Issue Parsing $\to$ File Localization $\to$ Patch Synthesis $\to$ Sandbox Execution $\to$ Unit Test Self-Healing $\to$ PR Creation?
2. Compare Docker vs Firecracker microVMs vs WASM across startup latency (<100ms), hypervisor isolation, and memory footprint?
3. How to use Tree-Sitter AST parsing and Symbol Tables to compress large 1M-line codebases into high-signal architectural skeletons?
4. How does Linux cgroups and seccomp system-call filtering prevent denial-of-service and unauthorized socket connections in sandboxes?
5. Design a pre-warmed container pool with Copy-on-Write snapshots supporting 100k concurrent code runs without storage exhaustion?

<details>
<summary>中文题目</summary>

1. 详细画出用户提出需求 $\to$ 代码定位 $\to$ 代码合成 $\to$ microVM 执行 $\to$ 单元测试自愈 $\to$ PR 生成的端到端系统架构图？
2. 代码沙箱选型对比：Docker 容器 vs Firecracker microVMs vs WebAssembly (WASM) 在冷启动延迟、安全隔离等级与内存开销上的权衡？
3. 超大代码库（100 万行代码）上下文检索方案：如何通过 Tree-Sitter 提取 AST 骨架、Symbol Table 符号索引与 ctags 实现无幻觉文件定位？
4. 死循环与恶意代码防御：如何在沙箱中通过 cgroups 限制 CPU/内存用量，并强制注入 10 秒硬性 Timeout 截断？
5. 10 万并发下的沙箱池预热与回收架构：如何通过 Warm-Pool 预热池、写时复制 (CoW) 磁盘镜像与快速 GC 保证高吞吐？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-code-agent-sandbox-system-design?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-code-agent-sandbox-system-design)

---

## 2. 10M Docs Enterprise RAG Architecture · 千万级文档企业 RAG 系统设计

> The Enterprise RAG System Architecture for 10 Million Documents represents the definitive enterprise GenAI system design benchmark, scaling heterogeneous ingestion, sub-second hybrid retrieval, and strict enterprise security over tens of millions of documents;

**Interview questions**

1. Diagram the complete 10M-document enterprise RAG architecture across Offline Ingestion, Hybrid Storage, Online Query Gateway, and LLM Inference?
2. Perform exact capacity planning: calculate RAM/SSD storage and HNSW index memory for 50M chunks ($1024$-dim vectors) with horizontal sharding?
3. Design a Change Data Capture (CDC) pipeline syncing document updates into vector and BM25 indices within seconds?
4. How to implement inline citation attribution (`[Doc 1, Page 3]`) combined with Natural Language Inference (NLI) to verify answer grounding?
5. How to proactively summarize architectural trade-offs (vector search speed vs GraphRAG global sensemaking costs) to secure high evaluation scores?

<details>
<summary>中文题目</summary>

1. 详细画出包含离线摄入流 (Kafka/Ray)、混合存储引擎 (Milvus+ES)、在线查询网关与 LLM 生成沙箱的千万级 RAG 架构图？
2. 容量与硬件资源估算：1000 万篇文档（平均每篇 3000 字，切分为 5000 万个 Chunk 向量，维度 1024）的 HNSW 显存/内存需求与集群分片设计？
3. 文档生命周期与增量更新 (CDC: Change Data Capture): 当底层企业文件发生修改或删除时，如何在秒级内完成向量库与全文索引的增量同步？
4. 引用归因 (Citation & Fact Verification): 如何在生成答案中自动化标注引用来源角标 `[Doc 1, Page 3]` 并通过 NLI 蕴含模型进行事实一致性校验？
5. 在 45 分钟系统设计面试结尾，如何主动反思当前 RAG 架构的 Trade-offs（如向量召回精度 vs 延迟、建图成本 vs GraphRAG 全局能力）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-enterprise-rag-10m-docs-system-design?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-enterprise-rag-10m-docs-system-design)

---

## 3. AIE vs MLE Competency Model · AIE vs MLE 能力模型与演进

> The AIE (AI Systems & Compound AI Engineer) vs MLE (Machine Learning Engineer) Competency Model defines the industry paradigm shift in technical specialization;

**Interview questions**

1. Explain why Berkeley's Compound AI Systems outperform monolithic LLMs in task reliability, cost, and controllability?
2. Detail the defense-in-depth engineering stack against LLM hallucinations: Grounded Prompting $\to$ RAG Grounding $\to$ Guardrails $\to$ Self-Correction?
3. How should an AIE structure the decision tree choosing between Prompting, RAG, LoRA SFT, and Continual Pre-training?
4. Explain the 4 pillars of production AIE reliability: deterministic guarantees, context budgets, tool sandboxing, and idempotent retries?
5. Describe the end-to-end LLMOps lifecycle from synthetic data curation, fine-tuning, automated evals to online telemetry?

<details>
<summary>中文题目</summary>

1. 深入剖析 Compound AI Systems (复合 AI 系统: 伯克利提出) 为什么在复杂任务上的表现与可靠性远超单一超大模型 (Monolithic LLM)？
2. AIE 面试中如何系统阐述处理大模型“幻觉 (Hallucination)”的工程防御纵深（Prompt 约束 $\to$ RAG 检索校验 $\to$ Guardrails 过滤 $\to$ Self-Correction）？
3. 当业务需要垂直领域定制时，AIE 如何在 Prompting、RAG、SFT (LoRA) 与持续预训练之间进行科学的技术选型决策树判断？
4. AIE 生产系统的四大可靠性护栏：确定性执行保证、上下文窗口预算、工具沙箱隔离与多层幂等重试？
5. LLMOps 全生命周期体系：从数据清洗、合成数据微调、离线评估 (Eval)、灰度金丝雀发布到在线 Tracing 监控闭环？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-vs-mle-competency-model?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-vs-mle-competency-model)

---

## 4. Catastrophic Forgetting & Rehearsal · 微调灾难性遗忘防范与重放

> Mitigating Catastrophic Forgetting via Experience Rehearsal & Parameter Freezing ensures that foundation models do not degrade in broad general intelligence, coding logic, or multi-lingual proficiency while fine-tuning on narrow domain datasets;

**Interview questions**

1. Explain why Full Fine-Tuning is significantly more susceptible to catastrophic forgetting compared to LoRA parameter isolation?
2. Analyze the Pareto trade-off of general-to-domain data mixing ratios (e.g. 10% general vs 90% domain) in preserving benchmark scores?
3. Why does Spherical Linear Interpolation (SLERP) outperform naive linear averaging when merging fine-tuned checkpoints?
4. How does KL-penalty loss regularization mathematically restrict policy drift away from the base model distribution during SFT?
5. Design an automated regression evaluation gate running MMLU, GSM8k, and HumanEval benchmarks before deploying fine-tuned weights?

<details>
<summary>中文题目</summary>

1. 深入剖析为什么全参数微调 (Full Fine-Tuning) 比 LoRA 微调更容易发生严重的灾难性遗忘？
2. 经验重放 (Experience Rehearsal) 的最佳配比策略：通用数据比例（如 10% 通用问答 + 90% 领域数据）对领域任务指标与通用 MMLU 指标的帕累托前沿权衡？
3. SLERP (球面线性插值) 相比简单线性权重加权平均在融合大模型权重时的几何优势？
4. KL 散度正则化约束 (KL-Penalty: 在损失函数中增加 $\mathcal{L} + \lambda D_{KL}(\pi_\theta \| \pi_{\text{base}})$) 在防止策略过度偏移中的数学机理？
5. 如何建立微调后的自动化防退化评估流水线（自动化运行 MMLU, GSM8k, HumanEval 等通用基准）作为上线发布前置门禁？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-catastrophic-forgetting-rehearsal?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-catastrophic-forgetting-rehearsal)

---

## 5. Context Budgeting & Memory Compaction · 上下文预算分配与记忆 Compaction

> Context Window Token Budgeting & Progressive Memory Compaction treats the finite LLM context window as scarce RAM managed through strict partition budgeting and automatic memory compaction; the budget allocation matrix partitions total tokens (e.g.

**Interview questions**

1. Diagram the 4-tier context window token budget partitioning and dynamic reallocation thresholds?
2. Explain how recursive summarization preserves structured task states, active constraints, and resolved sub-goals without data loss?
3. Explain the 'Lost in the Middle' attention bias and why placing critical system constraints at the beginning and end maximizes retrieval accuracy?
4. How should tool execution wrappers safely truncate massive 50k-token CLI logs or HTML payloads before passing them to the LLM?
5. How to structure conversation history sequentially to preserve static prompt prefixes and maximize cloud KV Cache reuse?

<details>
<summary>中文题目</summary>

1. 详细画出上下文窗口四大分区（System Prompt、Working Memory、Output Buffer、Safety Headroom）的 Token 预算配比与动态流动图？
2. 递归语义摘要 (Recursive Summarization) 算法：如何在压缩历史对话的同时严格保留“已完成子任务状态”、“未解决的缺陷”与“关键实体变量”？
3. 大模型在超长上下文下的“迷失在中间 (Lost in the Middle)”现象：为什么将关键 System 指令与最相关的 RAG 证据放置在上下文首尾能显著提升召回率？
4. 大工具响应裁剪策略：当 Bash 或 Web 搜索工具返回 50,000 字原始 HTML/日志时，如何通过正则表达式与 Head/Tail 采样将其安全截断至 1,000 tokens？
5. Prompt Caching 友好型上下文排布：如何保持 System Prompt 与历史前缀的绝对静态不变以最大化 KV Cache 命中率？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-context-budget-compaction?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-context-budget-compaction)

---

## 6. Cross-Encoder Re-rank & ColBERT · Cross-Encoder 重排与 ColBERT

> Cross-Encoder Re-ranking & ColBERT Late-Interaction Deep Relevance Optimization refines initial Top 50-100 hybrid retrieval candidates down to the Top 3-5 pristine evidence chunks before LLM prompt injection;

**Interview questions**

1. Diagram and contrast the computation architectures of Bi-Encoder, Cross-Encoder, and ColBERT Late-Interaction?
2. Derive ColBERT's MaxSim equation and explain how it performs token-level soft alignment without full sequence cross-attention?
3. How to optimize Cross-Encoder inference latency to sub-15ms for 50 candidates using TensorRT operator fusion and INT8 quantization?
4. Why does dynamic re-ranking score thresholding prevent hallucinations by rejecting ungrounded queries before reaching the LLM?
5. Explain how multi-lingual re-rankers project cross-lingual tokens into a shared semantic alignment space?

<details>
<summary>中文题目</summary>

1. 详细画出 Bi-Encoder 双塔结构、Cross-Encoder 深度全交互结构与 ColBERT 晚期交互结构的计算架构对比图？
2. 推导 ColBERT 的 MaxSim 相似度打分公式 $S(q, d) = \sum_{i \in q} \max_{j \in d} (E_q^{(i)} \cdot E_d^{(j)})$ 并解释其 Token 级软对齐物理意义？
3. Cross-Encoder 生产部署优化：如何通过 ONNX Runtime、TensorRT 算子融合与 INT8 动态量化将 50 个 Chunk 的重排耗时压缩至 15ms 以内？
4. 重排分数阈值过滤 (Score Threshold Filtering): 为什么设置动态截断阈值（如 Score $<0.35$ 直接剔除）能从根本上消除“无证据仍强行胡说”的幻觉？
5. 多语言重排模型 (如 BGE-M3 Reranker, Cohere Rerank v3) 在跨语言检索（英语提问，检索中文/日文文档）中的对齐机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-cross-encoder-rerank-optimization?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-cross-encoder-rerank-optimization)

---

## 7. DPO vs GRPO Preference Alignment · DPO vs GRPO 偏好对齐算法

> Modern Preference Alignment & Reasoning RL algorithms—Direct Preference Optimization (DPO) vs Group Relative Policy Optimization (GRPO)—govern safety alignment and advanced logical reasoning in LLMs;

**Interview questions**

1. Derive the mathematical proof of DPO showing how the reward function is algebraically substituted with the log-policy ratio under Bradley-Terry?
2. Explain the role of hyperparameter $\beta$ in DPO in penalizing KL divergence drift away from the frozen reference policy?
3. Why does GRPO eliminate the Critic model, and how does it compute cohort-standardized advantages across $G$ sampled outputs?
4. How does rule-based reward verification (unit tests, exact math matching, format tags) induce long-chain reasoning emergence in DeepSeek-R1?
5. Explain the likelihood displacement pathology in DPO where both win/lose probabilities drop, and how KTO/IPO addresses it?

<details>
<summary>中文题目</summary>

1. 详细推导 DPO 如何从 Bradley-Terry 模型 $P(y_1 \succ y_2|x) = \sigma(r(x, y_1) - r(x, y_2))$ 严格推导出不含显式奖励函数 $r$ 的参数化损失公式？
2. DPO 中超参数 $\beta$ (通常设为 0.1~0.5) 的物理意义：如何控制当前策略与参考策略 $\pi_{\text{ref}}$ 之间的 KL 散度漂移惩罚？
3. GRPO (Group Relative Policy Optimization) 为什么不需要 Critic 模型？详细画出其基于 Group 采样与优势标准化的算法流程图？
4. DeepSeek-R1 式纯强化学习训练：如何通过基于规则的奖励 (Rule-based Rewards: 编译器通过率、数学答案正确性、XML 格式标签) 驱动模型涌现长思考链？
5. DPO 训练中常见的问题：负样本似然同时下降 (Likelihood Displacement) 与数据偏好分布不平衡时的调优排障？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-dpo-grpo-preference-alignment?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-dpo-grpo-preference-alignment)

---

## 8. Eval Benchmarks: RAGAS & SWE-bench · 评估体系 RAGAS 与 SWE-bench

> The LLM Evaluation Benchmarks framework (RAGAS & SWE-bench) establishes rigorous quantitative scorecards determining production readiness for enterprise RAG and autonomous software engineering agents;

**Interview questions**

1. Explain the calculation formulation for RAGAS metrics and how LLM-as-a-Judge automates objective scoring?
2. Detail the 3 intrinsic biases of LLM-as-a-Judge (Position, Verbosity, Self-Enhancement) and how position swapping mitigates them?
3. Explain the containerized benchmark architecture of SWE-bench isolating multi-repo dependencies and unit test verification?
4. How to integrate DeepEval/Promptfoo into GitHub Actions CI/CD to block PR merges upon evaluation metric regressions?
5. How to leverage Evol-Instruct to automatically synthesize multi-hop Q&A test datasets from raw enterprise knowledge bases?

<details>
<summary>中文题目</summary>

1. 详细剖析 RAGAS 四大指标（Faithfulness、Answer Relevance、Context Precision、Context Recall）的数学计算逻辑与如何利用 LLM-as-a-Judge 自动化打分？
2. LLM-as-a-Judge 的三大固有偏置（位置偏置 Position Bias、长度偏置 Verbosity Bias、自偏好偏置 Self-Enhancement Bias）与如何通过交换候选顺序消除？
3. SWE-bench (以及 SWE-bench Lite) 的测试环境搭建与 Docker 隔离执行沙箱设计？
4. 单元测试与回归测试在 CI/CD 中如何与大模型 Eval 结合（如使用 DeepEval / Promptfoo 实现 PR 触发自动化拦截）？
5. 合成评估数据集生成 (Synthetic Testset Generation): 如何基于企业知识库通过演化策略 (Evol-Instruct) 自动生成上千条覆盖多跳推理的黄金 Q&A 测试对？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-eval-benchmarks-swe-bench?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-eval-benchmarks-swe-bench)

---

## 9. Fine-Tuning VRAM Calculator & ZeRO · 微调显存开销精确计算与 ZeRO

> The Fine-Tuning VRAM Breakdown Calculator & DeepSpeed ZeRO Scaling Matrix provides the exact mathematical formulas required to budget GPU cluster memory and prevent runtime CUDA Out-Of-Memory crashes;

**Interview questions**

1. Derive the exact $16\Phi$ bytes static memory formula for full-parameter AdamW mixed-precision training breakdown?
2. Diagram ZeRO-1, ZeRO-2, and ZeRO-3 parameter/gradient/optimizer state partitioning across GPUs and compare their communication overheads?
3. Why does long-context sequence length blow up activation memory quadratically, and how does Activation Checkpointing trade 20% compute for 70% memory savings?
4. Derive the memory footprint of LoRA fine-tuning showing why frozen base weights eliminate 99% of optimizer and gradient memory?
5. How does FlashAttention-2 reduce attention activation memory from $O(N^2)$ to $O(N)$ via SRAM tiling and online softmax?

<details>
<summary>中文题目</summary>

1. 现场推导全参数 AdamW 混合精度训练为什么静态显存占用正好为 $16\Phi$ 字节（逐项列出权重、梯度与优化器状态的字节数）？
2. 详细画出 DeepSpeed ZeRO-1、ZeRO-2 与 ZeRO-3 在多 GPU 节点间的状态切分示意图与各阶段的通信量开销？
3. 激活显存 (Activation Memory): 为什么长上下文（如 32k/128k）会导致激活显存呈二次方暴涨？激活重计算 (Activation Checkpointing) 如何以 20% 计算时间换取 70% 显存节省？
4. LoRA 微调显存计算：为什么 LoRA 微调 70B 模型仅需约 150GB 显存（只有 trainable adapter 参数才需要梯度与优化器状态）？
5. FlashAttention-2 是如何从硬件显存带宽与在线 Softmax 层面将自注意力激活显存从 $O(N^2)$ 降低至 $O(N)$ 的？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-finetune-vram-calculator?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-finetune-vram-calculator)

---

## 10. GraphRAG Entity Extraction & Community · GraphRAG 实体抽取与社区摘要

> Microsoft GraphRAG (Knowledge Graph Entity-Relationship Extraction & Hierarchical Community Summarization) overcomes the fatal limitation of standard Vector RAG on holistic 'Global Sensemaking' queries (e.g.

**Interview questions**

1. Diagram the end-to-end GraphRAG pipeline from entity extraction, Leiden clustering, community summarization to Map-Reduce global search?
2. Why does standard Vector RAG fail completely on dataset-wide global synthesis queries where GraphRAG excels?
3. Explain why the Leiden community detection algorithm fixes Louvain's disconnected cluster pathologies in knowledge graph partitioning?
4. How to optimize GraphRAG indexing costs using smaller extraction models, prompt compression, and prompt caching?
5. How to combine Graph Traversal and Vector Retrieval into a hybrid search expansion pipeline in production?

<details>
<summary>中文题目</summary>

1. 详细画出 GraphRAG 从“实体抽取 $\to$ Leiden 图聚类 $\to$ 社区摘要预生成 $\to$ Map-Reduce 全局问答”的端到端架构时序图？
2. 为什么在全库宏观提问（如“总结本书的主要人物冲突与核心主题”）场景下，传统 Vector RAG 的召回率几乎为 0 而 GraphRAG 能够完美回答？
3. Leiden 社区发现算法相比 Louvain 算法在消除不良连通社区与保证拓扑收敛性上的数学优势？
4. GraphRAG 建索引的高昂 Token 成本控制策略：如何通过小模型批处理抽取实体、Prompt 压缩与 Prompt Caching 将建图成本降低 80%？
5. 图检索 (Graph Search) 与向量检索 (Vector Search) 的在线混合融合：如何利用图拓扑遍历扩展向量召回的实体一度邻居？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-graph-rag-knowledge-graph-fusion?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-graph-rag-knowledge-graph-fusion)

---

## 11. Hybrid Search (BM25 + Dense) & RRF · BM25+向量混合检索与 RRF 融合

> Hybrid Search (Sparse BM25/SPLADE + Dense Vector) combined with Reciprocal Rank Fusion (RRF) is the industry standard retrieval architecture balancing exact keyword precision with deep semantic generalization;

**Interview questions**

1. Derive the RRF formula and explain why rank-based reciprocal aggregation is more robust than linear score weighting against uncalibrated scores?
2. How does SPLADE predict vocabulary-level sparse weights to achieve automated term expansion over static BM25?
3. How do modern vector databases (Qdrant/Milvus) orchestrate single-query hybrid search across sparse and dense indices natively?
4. Design a Query Intent Classifier that dynamically modulates the ratio between sparse and dense retrieval channels?
5. How to orchestrate concurrent async hybrid queries with strict 15ms latency budgets to satisfy end-to-end SLAs?

<details>
<summary>中文题目</summary>

1. 详细推导 RRF (Reciprocal Rank Fusion) 的打分公式，并解释为什么基于排名的归一化比基于原始 Score 线性加权（如 $\alpha S_{\text{bm25}} + (1-\alpha) S_{\text{dense}}$）更鲁棒？
2. 稀疏神经检索 (Learned Sparse Retrieval / SPLADE: 词表级别稀疏权重预测) 相比传统 BM25 在词汇扩展 (Term Expansion) 上的优势？
3. 在生产环境中，如何利用 Qdrant / Milvus 原生混合检索能力在一个引擎内部同时维护 Sparse 与 Dense 索引？
4. Query 意图自适应权重分流：如何识别用户 Query 属于“专有名词查找型”还是“概念推理型”，并动态调整 BM25 与 Dense 的召回比例？
5. 大规模并发下的混合检索延迟预算：如何通过并发 `asyncio.gather` 并行请求与设定 15ms 超时截断确保检索 SLA？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-hybrid-search-rrf-dense-sparse?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-hybrid-search-rrf-dense-sparse)

---

## 12. LangSmith-Style Tracing & Self-Healing · Agent Tracing 可观测性与自恢复

> Production Agent Step-Level Tracing & Distributed Observability (LangSmith/OpenInference standards) provides the foundational telemetry required to profile latency bottlenecks, audit per-step token expenditures, isolate hallucination root causes, and orchestra…

**Interview questions**

1. Diagram the hierarchical distributed tracing tree topology encompassing Root Trace, LLM Spans, Tool Spans, and Retriever Spans?
2. Explain how OpenInference semantic conventions standardize LLM telemetry attributes across different tracing backends?
3. How to design non-blocking asynchronous telemetry ingest with dynamic tail-sampling (100% on errors, 10% on success)?
4. Design a PII sanitization filter scrubbing credit cards, passwords, and user identities before persisting traces?
5. How to pipeline production user thumbs-down traces into automated negative examples for offline DPO preference dataset curation?

<details>
<summary>中文题目</summary>

1. 详细画出包含 Root Trace、LLM Span、Tool Span 与 Retriever Span 的树状调用追踪拓扑结构？
2. OpenTelemetry / OpenInference 语义约定 (Semantic Conventions) 在标准化大模型 Tracing 字段（如 `gen_ai.prompt`, `gen_ai.completion`, `gen_ai.usage.cost`）中的规范？
3. 高并发场景下的 Tracing 性能开销控制：如何通过异步无阻塞消息队列（Kafka / RabbitMQ）与动态采样率（如 100% 采样错误请求、10% 采样正常请求）降低系统负担？
4. 用户隐私与数据脱敏 (Data Masking / PII Sanitization): 如何在 Tracing 写入存储前通过正则表达式与 Named Entity Recognition (NER) 自动清洗身份证、手机号与密码？
5. 基于 Tracing 数据的“失败自动标注与合成数据集转化”闭环：如何将生产环境中标记为 Dislike 的失败 Trace 自动导出为 DPO 偏好对齐数据集？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-langsmith-style-tracing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-langsmith-style-tracing)

---

## 13. Loop Hard Termination & Stagnation Guards · Agent 循环硬终止与死循环检测

> The Agent Loop Hard Termination & Stagnation Guards framework establishes non-negotiable safety invariants preventing autonomous agent loops from running amok, exhausting API budgets, or causing service denial;

**Interview questions**

1. Write the complete Python async agent execution loop enforcing Max Iterations, wall-clock timeout, and token budget bounds?
2. Design a stagnation detector tracking cryptographic hashes of tool calls and prompt states to catch cognitive deadlocks?
3. How should an agent synthesize a graceful partial progress debrief and hand-off summary when an execution ceiling is reached?
4. How does LangGraph implement durable pause-and-resume breakpoints requiring explicit human approval before privileged actions?
5. How to route perpetually failing agent tasks into Dead Letter Queues for offline debugging and post-mortem review?

<details>
<summary>中文题目</summary>

1. 现场写出使用 Python `asyncio` 与状态机实现包含最大迭代轮数、墙钟超时与 Token 预算监控的 Agent 执行循环？
2. 停滞检测 (Stagnation Detection) 的哈希签名匹配算法：如何检测 Agent 连续以相同参数调用同一工具或输出高度相似的 Thought？
3. 当 Agent 触碰最大迭代次数硬终止后，如何向用户输出清晰的“半成品状态汇报 (Partial Progress Report)”与优雅降级方案？
4. 人机协同介入 (Human-in-the-Loop Interrupts): 如何在 LangGraph 中定义高危操作（如转账、删库、发邮件）的断点暂停与人工审批恢复？
5. 分布式 Agent 执行中的死信队列 (Dead Letter Queue, DLQ) 与任务重试幂等性设计？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-agent-loop-termination-guards?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-agent-loop-termination-guards)

---

## 14. LoRA/QLoRA VRAM & Weight Merge · LoRA/QLoRA 显存与权重合并

> LoRA/QLoRA Parameter-Efficient Fine-Tuning (PEFT), VRAM Budgeting & Adapter Lifecycle Management empowers engineers to fine-tune 7B-70B models on single commodity GPUs (e.g.

**Interview questions**

1. Derive LoRA's forward pass $h = W_0 x + \frac{\alpha}{r} B A x$ and explain why $B$ is initialized to exact zero while $A$ is Gaussian?
2. Explain the relationship between Rank $r$ and scaling factor $\alpha$ (why $\alpha = 2r$ is common practice) in learning stability?
3. Compare applying LoRA solely on attention projections vs all linear FFN layers in model accuracy and parameter efficiency?
4. Why does QLoRA's NF4 (NormalFloat4) preserve information entropy better than uniform INT4 on normally distributed weights?
5. How does `merge_and_unload()` fuse LoRA weights into base tensors to achieve zero additional serving latency in vLLM?

<details>
<summary>中文题目</summary>

1. 推导 LoRA 的前向更新公式 $h = W_0 x + \frac{\alpha}{r} B A x$，并解释为什么矩阵 $A$ 采用高斯初始化而矩阵 $B$ 初始化为全 0？
2. 超参数 Rank $r$（通常选 8, 16, 64）与缩放因子 $\alpha$ (Alpha，通常设为 $2r$) 的物理意义与调参权衡？
3. Target Modules 选择策略：仅微调注意力层 ($W_q, W_k, W_v$) vs 微调全线性层 ($W_q, W_k, W_v, W_o, \text{gate}, \text{up}, \text{down}$) 的泛化收益对比？
4. QLoRA 中 NF4 (NormalFloat4) 数据类型相比标准 INT4 为什么能从信息论角度更完美适配正态分布的神经网络权重？
5. 在生产环境部署时，如何使用 HuggingFace PEFT 的 `merge_and_unload()` 将 LoRA 权重直接合并写入基础模型以消除在线推理延迟？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-lora-qlora-adapter-lifecycle?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-lora-qlora-adapter-lifecycle)

---

## 15. Multi-Tenant RAG ACL & Security · 多租户 RAG 权限隔离与安全沙箱

> The Multi-Tenant Enterprise RAG Access Control (RBAC/ACL) & Security Isolation Architecture enforces strict data confidentiality boundaries across enterprise organizations, mathematically preventing unauthorized cross-tenant data leakage;

**Interview questions**

1. Contrast Physical Collection isolation vs Partition Keys vs Payload filtering across infrastructure costs, multi-tenant scalability, and compliance?
2. Explain why post-filtering triggers recall collapse when ACL masks 99% of documents, and how HNSW pre-filtering resolves it?
3. How to synchronize hierarchical LDAP/Active Directory permission groups into vector metadata for real-time ACL evaluation?
4. Design a cascade deletion pipeline satisfying GDPR Right to be Forgotten across vector indices, keyword stores, and semantic caches?
5. How to design context boundary isolation preventing untrusted low-privileged retrieved text from executing indirect injection exploits?

<details>
<summary>中文题目</summary>

1. 详细对比物理分集合隔离 (Per-Tenant Collection)、分区隔离 (Partition Key) 与 Payload 标量过滤三种模式在成本、扩展性与安全合规上的权衡？
2. 向量数据库“前置过滤 (Pre-filtering)”与“后置过滤 (Post-filtering)”的性能陷阱：为什么后置过滤在用户权限极小时会导致召回结果数量不足甚至为空？
3. 用户权限继承与动态组过滤 (Dynamic Group ACLs): 如何将企业 LDAP/Active Directory 权限树结构高效同步至向量库元数据中？
4. 文档删除与数据遗忘权 (GDPR Right to be Forgotten): 当企业租户注销或删除单个敏感文件时，如何在分布式向量库中秒级彻底清除所有 Chunk 向量？
5. 间接提示注入跨权限攻击防御：如何防止低权限攻击者通过诱导 LLM 生成系统指令来套取高权限上下文数据？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-multi-tenant-vector-acl-isolation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-multi-tenant-vector-acl-isolation)

---

## 16. Parent-Child & Sentence-Window Chunking · Parent-Child 与句窗分块策略

> Hierarchical Parent-Child & Sentence-Window Chunking resolves the fundamental trade-off in RAG systems between retrieval precision and generative context completeness;

**Interview questions**

1. Diagram the end-to-end Parent-Child dataflow from hierarchical chunking, child embedding, vector lookup to parent payload hydration?
2. Analyze the trade-off of sentence-window expansion radius $K$ on context coherence versus introducing irrelevant token noise?
3. Explain Semantic Chunking algorithms that detect semantic boundaries via cosine distance drops between consecutive sentence embeddings?
4. How do advanced document parsers preserve intact Markdown table structures and code blocks across chunk boundaries?
5. How does injecting document hierarchy breadcrumbs into child chunk metadata boost vector retrieval accuracy?

<details>
<summary>中文题目</summary>

1. 详细画出 Parent-Child 架构中“文档切分 $\to$ 子块 Embedding $\to$ 检索匹配 $\to$ 父块替换回填 Prompt”的完整数据流图？
2. 句窗检索 (Sentence-Window Retrieval) 中的窗口大小 $K$（如 $K=3$ 展开前后各 3 句）对检索噪声比与上下文完整性的影响分析？
3. 语义分块 (Semantic Chunking: 基于前后句子间的 Embedding 余弦相似度突变点作为分块边界) 的自适应切分算法？
4. 多模态文档解析：PDF 表格、复杂图表与代码块在切块时如何通过 Markdown 结构化保留 (如利用 Unstructured / MinerU) 防止表格断裂？
5. 元数据注入 (Metadata Enrichment): 在子块中注入全局标题路径（如 `Breadcrumb: 文档名 > 章节 > 小节`）对提升 Embedding 辨识度的收益？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-chunking-hierarchical-parent-child?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-chunking-hierarchical-parent-child)

---

## 17. Prompt Chains & Jailbreak Defense · 高级 Prompt 链与防越狱注入

> Advanced Prompt Chains, Chain-of-Thought & Jailbreak/Prompt-Injection Defense establishes deterministic reasoning pathways while hardening generative interfaces against adversarial bypasses;

**Interview questions**

1. Contrast Direct vs Indirect Prompt Injection attack vectors and detail defenses against poisoned retrieved documents in RAG?
2. Explain how XML boundary delimitation guides LLM attention to separate privileged system prompts from untrusted payload data?
3. How does Self-Consistency sampling across diverse CoT reasoning paths boost accuracy in multi-step logical tasks?
4. Design a dual-layer asynchronous safety pipeline leveraging Llama-Guard for input intent gating and output toxicity audits?
5. What specific prompt engineering techniques and canary tokens prevent attackers from exfiltrating proprietary system instructions?

<details>
<summary>中文题目</summary>

1. 直接提示注入 (Direct Injection) 与间接提示注入 (Indirect Injection) 的攻击路径差异与在 RAG 场景下的实操防御方案？
2. XML 结构化标签隔离技术（如 `<system>`, `<context>`, `<user_query>`）如何帮助 LLM 的注意力机制区分特权系统指令与非受信任用户数据？
3. 思维链 (CoT) 与自洽性采样 (Self-Consistency: 采样 $N$ 条推理路径并对最终答案投票) 在提升复杂多步逻辑推理准确率中的原理？
4. Guardrails (如 NeMo Guardrails, Llama-Guard) 在输入前置拦截与输出后置审查中的双向异步流水线设计？
5. 如何防范 Prompt 逆向工程与系统提示词泄漏攻击 (Prompt Extraction Attack)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-prompt-engineering-jailbreak-defense?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-prompt-engineering-jailbreak-defense)

---

## 18. ReAct, Plan-Execute & Reflexion · ReAct / Plan-Execute / Reflexion

> The Agent Orchestration Patterns Taxonomy structures production-grade autonomous LLM systems across varying task horizons, latency budgets, and reliability constraints;

**Interview questions**

1. Contrast ReAct vs Plan-and-Execute across execution latency, token consumption, and multi-step goal tracking reliability?
2. Explain how Reflexion's Actor, Evaluator, and Self-Reflection components achieve gradient-free self-improvement across iterations?
3. How does LangGraph's StateGraph model nodes, conditional edges, and human-in-the-loop breakpoints for deterministic agent workflows?
4. How to resolve multi-agent infinite chatter loops and consensus polarization using deterministic Judge arbitration and token bounds?
5. How to build an adaptive router dynamically dispatching between low-latency ReAct and deep Plan-and-Execute workflows?

<details>
<summary>中文题目</summary>

1. 详细对比 ReAct 与 Plan-and-Execute 模式在执行延迟 (Latency)、Token 消耗与长任务全局目标保持能力上的优劣势？
2. Reflexion 模式的三大核心组件（Actor 策略模型、Evaluator 评估器、Self-Reflection 语义记忆）如何实现跨轮次无梯度自主进化？
3. LangGraph 状态图 (StateGraph) 架构：如何通过 State Schema、Nodes 节点与 Conditional Edges 条件边构建支持循环与人工介入 (Human-in-the-loop) 的生产级 Agent？
4. 多 Agent 协作中的“群体盲目 (Groupthink)”与通信死循环如何通过设定投票裁判 (Judge LLM) 与发言轮数上限解决？
5. 在生产环境中，如何根据任务性质（即时交互 vs 离线批处理）动态选择 ReAct 快速执行与 Plan-and-Execute 深度规划？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-agent-orchestration-patterns?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-agent-orchestration-patterns)

---

## 19. SFT Data Packing & Loss Masking · SFT Data Packing 与 Loss Masking

> SFT Data Packing & Loss Masking are foundational data engineering mechanisms in Supervised Fine-Tuning maximizing training throughput while strictly focusing parameter updates on target completions;

**Interview questions**

1. Why must prompt tokens be masked with label `-100` during SFT, and what severe degradation occurs if loss is computed on prompts?
2. How does FlashAttention's variable-length API (`cu_seqlens`) prevent cross-document attention contamination in packed sequences?
3. How to construct token masks in multi-turn dialogues so that loss is computed solely on the final (or all) Assistant turns while conditioning on previous history?
4. Quantify the 3-5x speedup and memory efficiency gains of Data Packing over conventional batched padding?
5. How do Jinja2 Chat Templates standardize special token delimiters across Llama-3, Qwen-2.5, and Mistral in production SFT data pipelines?

<details>
<summary>中文题目</summary>

1. 为什么在 SFT 训练时必须对 Prompt 部分做 Loss Masking（将 label 置为 `-100`）？如果不做会导致模型产生什么严重的负面退化？
2. Data Packing 中如何使用 FlashAttention 的 `flash_attn_varlen_func` 与 `cu_seqlens` (累积序列长度数组) 彻底杜绝样本间的跨文档注意力穿越？
3. 多轮对话场景下的掩码构建：如何仅对 Assistant 角色的回合开启损失，同时将历史多轮 User / Assistant 文本作为上文仅参与前向？
4. Data Packing 相比传统 Padding 训练在 GPU 显存占用与训练吞吐量 (Tokens/sec) 上的定量加速效果（为什么能提速 3~5 倍）？
5. Chat Template (如 Jinja2 模板格式) 在统一不同开源基础模型（Llama-3, Qwen-2.5, Mistral）特殊 Token（`<|im_start|>`, `<|im_end|>`) 中的标准工程实践？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-sft-data-packing-loss-masking?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-sft-data-packing-loss-masking)

---

## 20. Speculative Decoding with Draft Models · 投机采样 Draft Model 推理加速

> Speculative Decoding & Draft-Model Inference Acceleration delivers mathematically lossless 2-3x throughput acceleration for autoregressive LLM decoding by exploiting compute-bound parallel verification against memory-bound single-token generation bottlenecks;

**Interview questions**

1. Derive the mathematical proof showing why Speculative Decoding's rejection sampling criterion produces zero distribution divergence from $P_{\text{target}}(x)$?
2. Derive the theoretical speedup multiplier equation as a function of draft length $K$, acceptance rate $\alpha$, and relative draft-to-target cost?
3. Contrast Draft-Free speculative methods: Medusa's multi-head classification heads vs Eagle's hidden-state contextual autoregression?
4. Explain Prompt Lookup Decoding which extracts candidate token n-grams directly from the input prompt without needing draft models?
5. How does vLLM orchestrate target and draft model memory co-location and parallel tensor scheduling during multi-tenant serving?

<details>
<summary>中文题目</summary>

1. 数学推导为什么投机采样的拒绝采样算法能严格证明其最终采样的概率分布与目标大模型 $P_{\text{target}}(x)$ 100% 绝对一致（无损数学证明）？
2. 草稿步长 $K$ (Lookahead Step, 通常取 3~5) 与草稿接受率 $\alpha$ (Acceptance Rate) 对最终实际加速比的理论模型推导？
3. 无草稿模型的投机加速范式：Medusa (多头预测并行生成) 与 Eagle (引入特征层自回归加速) 的原理对比？
4. 基于 N-Gram / Prompt 匹配的提示词投机采样 (Prompt Lookup Decoding): 为什么在代码重构与长文档翻译任务中无需额外训练任何模型即可提速 2 倍？
5. 在多租户高并发场景下，草稿模型与目标模型的 GPU 显存共存与并发调度策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-speculative-decoding-serving?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-speculative-decoding-serving)

---

## 21. SSE Streaming Gateway & Backpressure · SSE 流式输出网关与背压流控

> The Server-Sent Events (SSE) Streaming Gateway Architecture & Adaptive Backpressure Flow Control handles millions of concurrent long-lived streaming connections while insulating GPU inference clusters from slow-client bottlenecks;

**Interview questions**

1. Diagram the end-to-end SSE streaming dataflow and Abort Signal propagation across Client, Envoy Gateway, FastAPI, and vLLM?
2. Why must `proxy_buffering off;` and `X-Accel-Buffering: no` be configured in Nginx to prevent buffering delay spikes during SSE streaming?
3. Compare WebSocket vs SSE vs gRPC bidirectional streaming across firewall traversal, auto-reconnection, and architectural simplicity?
4. How to use FastAPI `request.is_disconnected()` asyncio generators to cancel active vLLM generation tasks upon socket drop?
5. How do frontend streaming UI parsers handle partial unclosed Markdown code blocks gracefully without layout flickering?

<details>
<summary>中文题目</summary>

1. 详细画出用户客户端、Nginx / Envoy 反向代理网关、FastAPI 中间层与 vLLM 推理引擎之间的 SSE 流式长连接与取消信号 (Abort Signal) 流转图？
2. Nginx / Load Balancer 缓冲配置避坑：为什么必须显式配置 `proxy_buffering off;` 与 `X-Accel-Buffering: no` 才能保证 Token 实时流式吐出？
3. WebSocket vs Server-Sent Events (SSE) vs gRPC 双向流在现代大模型对话系统架构中的全方位对比选型？
4. 客户端意外断开 (Client Disconnect) 的心跳探测与 GPU 任务主动取消：如何在 FastAPI 中利用 `request.is_disconnected()` 触发异步取消？
5. 流式输出中的 Markdown 语法闭合与代码高亮：前端如何平滑渲染正在打字机输出中尚未闭合的代码块三反引号？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-sse-streaming-gateway-backpressure?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-sse-streaming-gateway-backpressure)

---

## 22. Structured Output & Constrained Decoding · 结构化输出与约束解码

> Guaranteed Structured Outputs & Grammar-Constrained Decoding forces non-deterministic LLM token generation to strictly conform to predefined JSON Schemas and Pydantic models;

**Interview questions**

1. Explain how Outlines compiles JSON Schemas into DFAs to perform runtime token logit masking during autoregressive sampling?
2. Why does constrained decoding incur near-zero latency overhead thanks to offline DFA indexing?
3. How does OpenAI Structured Outputs with `strict=True` enforce schema adherence by banning optional fields without defaults?
4. How to use Pydantic recursive models to have LLMs emit fully valid, strongly-typed multi-step Task DAG plans in a single forward pass?
5. How do partial streaming JSON parsers incrementally construct UI data trees from incomplete token streams?

<details>
<summary>中文题目</summary>

1. 详细剖析 Outlines / SGLang 基于有限状态自动机 (DFA) 在 Logits 层面做 Token Masking 的底层执行原理？
2. 为什么语法约束解码 (Constrained Decoding) 不会显著降低 LLM 的推理吞吐量（解析 DFA 状态转移矩阵的离线预编译机制）？
3. Pydantic 结合 OpenAI Structured Outputs (严格模式 `strict=True`) 对缺失字段、额外字段与类型校验的处理机制？
4. 在端到端 Agent 系统中，如何利用结构化输出直接生成具备嵌套多步骤执行计划的 Task DAG？
5. 流式结构化输出解析 (Streaming Partial JSON Parser): 客户端如何在 Token 持续打字机流式到达时实时解析不完整的半成品 JSON 树？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-structured-output-json-schema?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-structured-output-json-schema)

---

## 23. Token Costs, TTFT/TPOT & ROI · Token 成本与 TTFT/TPOT 优化

> LLM Token Economics, TTFT/TPOT Latency Breakdown & ROI Optimization defines the quantitative systems and financial engineering required to scale GenAI applications economically;

**Interview questions**

1. Contrast Compute-Bound Prefill vs Memory-Bound Decode phases and analyze their direct impacts on TTFT and TPOT?
2. Design a dual-tier semantic cache combining exact hash matching with vector cosine thresholding (>0.95) to serve instantaneous cached responses?
3. Explain the underlying KV cache reuse mechanism of Prompt Caching and its financial cost advantages on long system prompts?
4. How does FrugalGPT dynamic model routing dynamically triage queries between 7B and 70B models to maximize accuracy per dollar?
5. How to build real-time LLMOps cost dashboards with per-tenant token quotas and automated circuit breakers?

<details>
<summary>中文题目</summary>

1. 详细对比 Prefill 阶段 (Compute-Bound 计算密集型) 与 Decode 阶段 (Memory-Bound 显存带宽受限) 的硬件瓶颈差异及对 TTFT/TPOT 的直接影响？
2. 语义缓存 (Semantic Cache / GPTCache) 的架构设计：如何通过向量相似度阈值（如 Cosine 相似度 $>0.95$）与精确 Hash 匹配实现微秒级响应？
3. Prompt Caching (如 Anthropic / Gemini 前缀缓存机制) 的底层 KV Cache 内存复用原理与计费优惠策略？
4. 智能模型路由器 (Model Router / FrugalGPT): 如何基于分类器或 LLM 意图识别动态将 Query 分流给 7B 小模型或 70B/400B 巨型模型？
5. 如何建立端到端 Token 财务监控大盘：按用户 ID、功能模块、模型版本统计每日消耗，并设定硬性预算熔断机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-cost-latency-roi-tradeoffs?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-cost-latency-roi-tradeoffs)

---

## 24. Tool Validation, Idempotency & Fallbacks · 工具调用校验、幂等与超时降级

> The Tool Calling Reliability Engineering Framework establishes 4 mandatory layers of protection converting stochastic LLM tool requests into hardened enterprise-grade API executions;

**Interview questions**

1. Diagram the end-to-end tool execution lifecycle incorporating Pydantic validation interceptors and LLM self-correction feedback loops?
2. Design an idempotency token generation algorithm and Redis distributed lock to prevent duplicate tool execution across retries?
3. Why is Exponential Backoff with Full Jitter strictly required to prevent Thundering Herd retries against rate-limited APIs?
4. Explain role-based permission tiers and execution sandboxes separating read-only queries from destructive system mutations?
5. Design graceful degradation and mock fallback policies ensuring the main agent loop continues when peripheral tools fail?

<details>
<summary>中文题目</summary>

1. 详细画出带有 Pydantic 前置拦截校验与“自我修复反馈回路 (Self-Correction Loop)”的工具调用生命周期时序图？
2. 幂等键 (Idempotency Key) 的生成规则与基于 Redis 的分布式排重锁设计（防止写操作并发重入）？
3. 为什么在分布式重试中必须引入“全抖动随机退避 (Full Jitter Backoff: $\text{sleep} = \text{random}(0, 2^k \times \text{base})$)”以避免羊群效应 (Thundering Herd)？
4. 只读工具 (Read-Only) vs 破坏性写工具 (Destructive Actions) 的权限分级与安全沙箱隔离设计？
5. 当外部天气/搜索工具发生 500 崩溃或超时断流时，如何通过预设默认值 (Fallback) 与优雅降级保障 Agent 主体流程不崩溃？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-tool-calling-resilience-idempotency?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-tool-calling-resilience-idempotency)

---

## 25. vLLM PagedAttention & Continuous Batching · vLLM PagedAttention 与连续批处理

> vLLM High-Throughput Inference Engine (PagedAttention Virtual Memory & Continuous Iteration-Level Batching) provides the standard open-source serving architecture achieving 96%+ GPU memory utilization and 3-5x serving throughput gains over naive runtimes;

**Interview questions**

1. Diagram the logical-to-physical KV cache block allocation and address translation managed by vLLM's Page Table?
2. Explain how iteration-level scheduling in Continuous Batching eliminates token generation bubbles inherent in static batching?
3. Explain how PagedAttention's Copy-on-Write (CoW) enables zero-copy prompt KV cache sharing across parallel branches in Beam Search?
4. How does Chunked Prefill interleave long prompt chunks with decode tokens to eliminate TTFT latency spikes?
5. Analyze NCCL communication overheads of Tensor Parallelism vs Pipeline Parallelism when serving 70B models on multi-GPU nodes in vLLM?

<details>
<summary>中文题目</summary>

1. 详细画出 PagedAttention 中 Logical KV Blocks、Page Table 与 Physical GPU Memory Blocks 之间的虚拟地址映射流转图？
2. 连续批处理 (Continuous Batching) 相比传统静态批处理 (Static Batching) 为什么能将 GPU 计算单元 (Compute Cores) 的闲置气泡消除至接近 0？
3. 写时复制 (Copy-on-Write, CoW) 机制在束搜索 (Beam Search) 与多分支 Agent 并行生成（多采样 Sample 共享同一 Prompt）中的显存共享原理？
4. Chunked Prefill (分块预填充): 如何通过将长 Prompt 切分并与 Decode 请求共同打包进同一个 Batch，彻底消除长 Prompt 带来的服务卡顿与抖动？
5. 张量并行 (Tensor Parallelism, TP) 与流水线并行 (Pipeline Parallelism, PP) 在 vLLM 多卡分布式推理中的配置与通信开销分析？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AIE/card/aie-vllm-pagedattention-continuous-batching?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=aie-vllm-pagedattention-continuous-batching)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
