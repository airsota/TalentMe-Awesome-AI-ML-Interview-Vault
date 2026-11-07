# 🤖 AI Engineering & Agents · AI 工程与智能体 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **65 cards · 325 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. ADC Asymmetric Distance Computation · ADC 非对称距离计算

> Asymmetric Distance Computation (ADC) is a lookup-table acceleration algorithm in Product Quantization (PQ) that evaluates Euclidean distance or dot product between an unquantized query vector $q \in \mathbb{R}^D$ and a quantized database vector…

**Interview questions**

1. Derive the ADC distance formula $d(q, \hat{x})^2 = \sum_{m=1}^M \|q_m - c_{m, \text{code}_m(x)}\|^2$ and its computational complexity?
2. Why is LUT pre-computation complexity $O(K \cdot D)$ negligible for single-query latency?
3. Quantify the differences between ADC and SDC in CPU L1/L2 cache hit rate and reconstruction accuracy?
4. How can CPU SIMD instructions (`_mm256_shuffle_epi8`) parallelize 16-byte LUT batch lookups?
5. How should the LUT be organized in GPU shared memory to prevent bank conflicts during batched ADC scans?

<details>
<summary>中文题目</summary>

1. 推导 ADC 距离计算公式 $d(q, \hat{x})^2 = \sum_{m=1}^M \|q_m - c_{m, \text{code}_m(x)}\|^2$ 及其计算复杂度？
2. 为什么预构建 LUT (查找表) 的时间复杂度为 $O(M \cdot K \cdot d^*) = O(K \cdot D)$，对单次 Query 延迟几乎可以忽略？
3. 比较 ADC 与 SDC (对称距离计算) 在内存缓存命中率与重构精度上的量化差异？
4. 如何利用 CPU SIMD (AVX2 / AVX-512 `_mm256_shuffle_epi8` / `_mm512_permutexvar_epi8`) 指令实现并行 16 字节 LUT 批量查表？
5. 在 GPU 平台上执行 Batched ADC 扫描时，LUT 表如何布局以最大化 Warp 共享内存 (Shared Memory) 的无冲突访问？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/adc-distance-computation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=adc-distance-computation)

---

## 2. Agent Cycle & Timeout Prevention · Agent 死循环熔断与超时控制

> Agent Cycle Detection, Timeout & Circuit Breaker is an enterprise reliability and cost-control framework that prevents autonomous agents from entering infinite looping states, repeating identical failing tool invocations, or exhausting API budgets;

**Interview questions**

1. Detail how LangGraph's `recursion_limit` parameter tracks execution steps and raises `GraphRecursionError` exceptions?
2. Explain the algorithmic principles of State Fingerprinting and sliding window hashing to detect repeating tool loops?
3. How to enforce precise end-to-end wall-clock timeouts in asynchronous workflows via `asyncio.wait_for()`?
4. Upon triggering a circuit breaker, how to extract partial progress from Checkpointers and synthesize a graceful degradation summary?
5. Design a Token Budget Circuit Breaker that forcefully halts execution once cumulative token consumption exceeds a hard ceiling?

<details>
<summary>中文题目</summary>

1. 详细剖析 LangGraph 中 `recursion_limit` 参数在图执行中如何通过计数器抛出 `GraphRecursionError` 异常？
2. 基于状态指纹 (State Fingerprinting / MD5 Hashing of Tool Calls) 实现的动态死循环检测算法原理与滑动窗口去重？
3. 在异步并发图执行中，如何利用 Python `asyncio.wait_for(workflow.ainvoke(), timeout=60.0)` 实现精确的端到端硬超时熔断？
4. 当触发熔断时，如何从 Checkpointer 中提取已完成的中间成果并生成“部分完成 + 失败根因说明”的优雅降级回答？
5. Token 预算熔断器 (Token Budget Circuit Breaker: 累计消耗超过 50,000 tokens 即强行刹车) 的设计与实现？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/agent-cycle-timeout-prevention?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=agent-cycle-timeout-prevention)

---

## 3. Agent Memory Architecture · Agent 长短期记忆工程

> Agent Memory Architecture is the multi-tiered persistence subsystem that endows stateless LLMs with cross-session personalization, long-term fact retention, and evolving skill accumulation;

**Interview questions**

1. Detail the workflow in Mem0 for dynamically extracting facts, updating existing memories, and invalidating contradictory stale facts?
2. Explain short-term working memory management combining sliding message windows with asynchronous incremental LLM summarization?
3. Derive the tripartite retrieval scoring formula for Episodic Memory: Semantic Similarity + Time Decay Factor + Importance Weight?
4. How does Procedural Memory persist successfully authored Python tool scripts into a dynamic skill library (e.g., Voyager paradigm)?
5. How to implement deterministic hard-deletion across vector/graph memory systems to comply with GDPR 'Right to be Forgotten' mandates?

<details>
<summary>中文题目</summary>

1. 详细剖析 Mem0 框架中基于 LLM 动态提取用户偏好、更新已有事实 (Memory Update) 与删除过期冲突事实 (Memory Invalidation) 的工作流？
2. 短期工作记忆的滑动窗口 (Sliding Window) 与基于 LLM 异步压缩的增量摘要生成 (Incremental Conversation Summary) 架构？
3. 情境记忆 (Episodic Memory) 的三维混合检索机制：向量语义相似度 + 时间衰减系数 (Time Decay) + 重要性权重 (Importance Score)？
4. 程序记忆 (Procedural Memory) 如何通过将成功执行的 Python 函数代码持久化到工具库中实现 Agent 的自我技能扩展 (Skill Accumulation)？
5. 在严格遵循 GDPR / CCPA 隐私合规体系下，如何实现用户遗忘权 (Right to be Forgotten) 在向量记忆库中的精准硬删除？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/agent-memory-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=agent-memory-architecture)

---

## 4. BGE-Reranker Series · BGE-Reranker 模型系列

> BGE-Reranker (BAAI's open-source re-ranker series) is the premier open-source Cross-Encoder model family for production RAG pipelines;

**Interview questions**

1. Detail the core upgrades in BGE-Reranker-v2-m3 over v1.5 regarding multilingual support and 8K context?
2. Compare scoring mechanisms between Causal LM-based re-rankers (Qwen/Gemma) vs BERT/RoBERTa discriminative re-rankers?
3. How to accelerate BGE-Reranker inference via ONNX Runtime / TensorRT-LLM with FP16/INT8 quantization?
4. Why are raw BGE-Reranker logits unnormalized, and how to calibrate dynamic score thresholding?
5. How to domain-adapt and fine-tune BGE-Reranker on proprietary contrastive datasets to boost NDCG?

<details>
<summary>中文题目</summary>

1. 详细剖析 BGE-Reranker-v2-m3 相比第一代 v1.5 在多语言支持 (100+ languages) 与长上下文 (8K context) 上的核心升级？
2. 基于 Causal LM (如 Qwen / Gemma) 的生成式重排序模型与传统基于 BERT/RoBERTa 的判别式重排序模型的打分原理差异？
3. 在 GPU 显存受限或高 QPS 服务场景下，使用 ONNX Runtime / TensorRT-LLM / vLLM 对 BGE-Reranker 进行 FP16/INT8 量化加速方案？
4. BGE-Reranker 输出的分数 (Logits) 为何未经 Sigmoid 归一化？如何根据 Logit 分数设定动态截断阈值 (Threshold Filtering)？
5. 在垂直领域知识库（如医疗/金融）中，如何基于私有对比数据微调 BGE-Reranker 以进一步提升 NDCG？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/bge-reranker?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bge-reranker)

---

## 5. BM25 Sparse Retrieval · BM25 词频拟合稀疏检索

> BM25 (Best Matching 25) is a classic probabilistic information retrieval (IR) sparse term-matching algorithm; its relevance score formula is…

**Interview questions**

1. Derive the mathematical interpretation of parameters $k_1$ (TF saturation) and $b$ (length penalty) in the BM25 formula?
2. Why must $k_1$ and $b$ be tuned specifically when indexing ultra-short titles or extremely long legal text documents?
3. How does Chinese tokenization quality (Jieba, IK) fundamentally impact BM25 inverted index accuracy and recall?
4. Compare classical BM25 vs neural learned sparse retrieval (SPLADE, BGE-M3 Sparse) in lexical expansion capabilities?
5. How does the Block-Max WAND algorithm accelerate Top-K BM25 query execution via aggressive inverted list pruning?

<details>
<summary>中文题目</summary>

1. 推导 BM25 得分公式中参数 $k_1$（词频饱和曲线）与参数 $b$（文档长度惩罚）的物理含义与数学极限制约？
2. 为什么在短文本或者极长文档检索中，需要针对性微调 $k_1$ 和 $b$ 的取值？
3. 中文分词（如 Jieba, IK-Analyzer, Pku-Segmenter）的分词质量对 BM25 倒排索引建立与检索命中率的影响？
4. 对比 BM25 与神经稀疏检索 (Neural Sparse Retrieval, 如 SPLADE / BGE-M3 Sparse) 在词汇扩展与泛化能力上的演进？
5. 在大规模并发场景下，如何基于 WAND (Weak AND) 算法实现 BM25 Top-K 检索的高速剪枝跳跃？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/bm25-sparse-search?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bm25-sparse-search)

---

## 6. Chain-of-Thought Prompting · CoT 思维链与分步推导

> Chain-of-Thought (CoT) Prompting is a paradigm that elicits intermediate reasoning steps from an LLM before producing the final answer;

**Interview questions**

1. From an autoregressive Transformer attention standpoint, why does generating intermediate CoT tokens increase test-time compute capacity?
2. How does Self-Consistency improve accuracy by sampling diverse reasoning chains and taking the majority vote on final outputs?
3. How does Process Supervision (PRMs) validate intermediate CoT steps against unfaithful rationalization errors?
4. What are the advantages of Least-to-Most Prompting over standard CoT when addressing 5+ step complex dependencies?
5. How to apply XML tag delimiters (`<thought>` / `<answer>`) to structure and isolate intermediate reasoning from user-facing responses?

<details>
<summary>中文题目</summary>

1. 从自回归 Transformer 注意力机制的角度，解释为什么生成中间 CoT Token 能够等价于为难题增加了计算量 (Test-Time Compute)？
2. Self-Consistency (自洽性采样 / 多数投票 CoT) 是如何通过采样多条不同推理链并对最终答案取众数提升准确率的？
3. 当 CoT 中间步骤出现“看似合理实则逻辑矛盾 (Faithfulness & Hallucination in CoT)”时，如何进行单步中间状态验证 (Process Supervision)？
4. Least-to-Most Prompting 与分层规划 (Hierarchical Planning) 在处理 5 步以上极长逻辑链问题时的优势？
5. 在生产 API 响应中，如何通过 XML 标签（如 `<thought>...</thought><answer>...</answer>`）将思考过程与最终用户答案结构化隔离？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/chain-of-thought-prompting?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=chain-of-thought-prompting)

---

## 7. Citation Grounding & Faithfulness · RAG 事实引用核查与忠实度

> Citation Grounding & Faithfulness Verification is a post-processing and evaluation pipeline that enforces fine-grained factual attribution, reference citation linking, and hallucination pruning on RAG-generated text;

**Interview questions**

1. Describe the algorithm using NLI models (DeBERTa-v3-mnli) to decompose responses into atomic claims and compute Faithfulness scores?
2. How does a post-verification validator audit citation precision and recall to eliminate hallucinated reference brackets?
3. Design a collaborative architecture combining prompt-level citation elicitation with post-generation NLI validation?
4. What is the automated rewrite-and-prune self-healing protocol when a claim triggers an NLI contradiction or neutrality flag?
5. Derive the mathematical formulas and computational complexity of Faithfulness and Groundedness metrics in Ragas?

<details>
<summary>中文题目</summary>

1. 详细描述基于 NLI 模型（如 DeBERTa-v3-large-mnli）将生成回答解构为 Atomic Claims 并计算 Faithfulness 得分的完整算法？
2. 针对大模型“幻觉伪造角标引用 (Hallucinated Citations)”的现象，后置验证器如何执行引用准确度 (Citation Precision & Recall) 自动化审计？
3. Prompt 级别引导引用（如要求模型生成形如 `根据 [Doc-A, p.12] ...`）与后置独立 NLI 验证模型的协同架构设计？
4. 当检测到某关键结论存在矛盾 (`Contradiction`) 或中立无支撑 (`Neutral`) 时，系统的自愈重写 (Rewrite & Prune) 机制？
5. Ragas 框架中的 Faithfulness 指标与 RAGAS Groundedness 指标的数学公式推导与计算开销？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/citation-grounding-verification?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=citation-grounding-verification)

---

## 8. Code Interpreter Runtime · Code Interpreter 代码解释器

> Code Interpreter Runtime is a stateful execution sandbox enabling LLMs to dynamically author and run arbitrary executable code (predominantly Python / Bash);

**Interview questions**

1. Detail the architecture of using Jupyter Kernel Gateways (IPython) to maintain stateful variable sessions across multi-turn runs?
2. How to capture visual artifacts (Matplotlib plots) as Base64 images or artifact URIs for multi-modal Vision LLM validation?
3. How to configure hardware-enforced timeouts and Linux cgroup memory quotas against infinite loops and memory bomb scripts?
4. Weigh pre-warmed fat images vs secure dynamic `pip install` when LLM scripts require uninstalled third-party packages?
5. How to parse Python syntax tracebacks and format them into an automated error-recovery prompt for self-correction?

<details>
<summary>中文题目</summary>

1. 详细剖析基于 Jupyter Kernel Gateway (IPython Kernel) 维持多轮代码执行中变量状态 (Stateful Session) 的底层架构？
2. 当代码执行生成静态图表（如 Matplotlib 图表）时，如何捕获并转换为 Base64 图片或文件 URL 供 Vision LLM 多模态核验？
3. 针对恶意无限死循环代码（如 `while True: pass`）或内存炸弹，如何配置硬件级超时截断 (Timeout Kill) 与 Cgroup 内存限制？
4. 大模型动态导入缺失的第三方 Python 依赖包时，预装镜像 (Pre-warmed Image) 与安全动态 `pip install` 的架构权衡？
5. 如何解析 Python 语法错误 Traceback 并自动将其格式化为纠偏 Prompt 供模型进行下一步自我修复？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/code-interpreter-runtime?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=code-interpreter-runtime)

---

## 9. Cohen's Kappa Agreement · Cohen's Kappa 标注一致性统计量

> Cohen's Kappa ($\kappa$) is a statistical metric that measures inter-rater agreement between two evaluators (e.g., an LLM Judge vs Human Expert, or two competing judge models) across qualitative categories, strictly adjusting for chance agreement;

**Interview questions**

1. Derive formulas for observed agreement $P_o$ and chance agreement $P_e$ from a $2 \times 2$ confusion matrix?
2. Explain the Landis & Koch benchmark scale and standard industry thresholds (typically $\kappa > 0.7$) for deploying LLM judges in production?
3. How does Quadratic Weighted Kappa penalize distance severity across ordinal multi-tier ratings (e.g., 1-5 scale)?
4. Explain the Kappa Paradox (high raw agreement coinciding with low Kappa) under extreme class prevalence skews?
5. How does Fleiss' Kappa generalize Cohen's Kappa to measure inter-rater reliability across 3+ simultaneous raters?

<details>
<summary>中文题目</summary>

1. 推导混淆矩阵下实际观测一致率 $P_o = \frac{a + d}{N}$ 与期望偶然一致率 $P_e = \frac{(a+b)(a+c) + (c+d)(b+d)}{N^2}$ 的数学公式？
2. Landis & Koch 标准解释表（$\kappa < 0.2$ 极差，0.41-0.60 中等，0.61-0.80 显著，0.81-1.00 近乎完美）在工业界上线 LLM 裁判的门限？
3. 加权卡帕系数 (Weighted Kappa: Quadratic Weighted Kappa) 在处理有序多分类（如 1-5 分连续打分）时的距离惩罚机制？
4. 什么是 Kappa 悖论 (Kappa Paradox: 高准确率但低 Kappa 值)？样本类别极度不均衡时如何分析该现象？
5. 当评估者数量超过 2 位（如 5 位人类专家 + 1 个大模型）时，如何从 Cohen's Kappa 扩展到 Fleiss' Kappa 综合一致性？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/cohens-kappa-agreement?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cohens-kappa-agreement)

---

## 10. Cohere Rerank Industrial API · Cohere Rerank 工业重排

> Cohere Rerank (flagship enterprise commercial API, latest Rerank-v3 / Rerank-v3.5) is the industry benchmark hosted re-ranking model for enterprise cloud-native RAG systems;

**Interview questions**

1. How does Cohere Rerank-v3 natively parse JSON structured documents and apply custom field weightings?
2. Compare cloud Cohere Rerank API vs on-prem BGE-Reranker across network RTT, unit token costs, and data privacy compliance?
3. How to apply chunked batching and cascade filtering when re-ranking 500+ candidate passages via Cohere API?
4. Analyze the statistical distribution of Cohere `relevance_score` and heuristics for dynamic Top-N thresholding?
5. Evaluate Cohere Multilingual Rerank capabilities on mixed low-resource languages and programming code queries?

<details>
<summary>中文题目</summary>

1. Cohere Rerank-v3 如何通过原生支持 JSON 结构体与特定字段权重配置提升半结构化数据重排效果？
2. 分析云端调用 Cohere Rerank API 相比本地自建 BGE-Reranker 在网络往返延迟 (RTT)、成本计费与数据隐私合规上的权衡？
3. 当输入候选列表多达 500+ 个切块时，如何通过分页批量调用 (Chunked Batching) 与两阶段级联过滤控制 API 耗时与费用？
4. Cohere Rerank 返回的相关性分数 `relevance_score` 的物理分布特性以及如何设定动态 Top-N 截断？
5. 在混合多模态检索中，Cohere Multilingual Rerank 对小语种与代码混合查询的表现优势？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/cohere-rerank?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cohere-rerank)

---

## 11. ColBERT Late Interaction · ColBERT 迟交互多向量重排

> ColBERT (Contextualized Late Interaction over BERT) is a multi-vector retrieval and re-ranking architecture that fuses the speed of Bi-Encoders with the precision of Cross-Encoders;

**Interview questions**

1. Derive the ColBERT MaxSim operator and explain how it captures both soft lexical matches and deep contextual semantics?
2. How does ColBERTv2 apply residual compression and centroid quantization to reduce multi-vector storage by 6x-10x?
3. Compare ColBERT (Late Interaction), Bi-Encoder, and Cross-Encoder across latency, VRAM footprint, and NDCG?
4. How does the PLAID search engine achieve millisecond-level retrieval via 3-stage pruning (centroid, BoW, exact MaxSim)?
5. How does ColPali extend ColBERT late interaction to Vision LLMs (PaliGemma) for native multi-vector screenshot retrieval?

<details>
<summary>中文题目</summary>

1. 推导 ColBERT MaxSim 算子公式，并详细剖析其为何既能捕捉软关键词匹配又能捕捉深度上下文语义？
2. ColBERTv2 如何通过残差压缩 (Residual Compression) 与聚类量化将多向量索引的存储占用缩小 6x~10x？
3. 对比 ColBERT (迟交互)、Bi-Encoder (独立双塔) 与 Cross-Encoder (全交互) 在延时、显存占用与 NDCG 上的三维权衡？
4. PLAID 引擎如何通过三阶段剪枝算法（质心剪枝、词袋剪枝、精确 MaxSim）实现万级候选的毫秒级检索？
5. ColPali 如何将 ColBERT 迟交互架构扩展至 Vision LLM (PaliGemma) 实现基于 PDF 页面截图的原生图像多向量检索？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/colbert-late-interaction?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=colbert-late-interaction)

---

## 12. Computer Use GUI Agent · Computer Use 桌面 GUI 控制

> Computer Use & GUI Agent (Anthropic Claude 3.5 Sonnet Computer Use, OS-World) is a multimodal agent paradigm that directly operates computer desktops by inspecting screen captures, grounding UI visual elements, and emitting native mouse clicks, drags, keystrok…

**Interview questions**

1. Detail the principles of visual grounding and pixel coordinate normalization over 1080p screenshots in VLMs?
2. How does the Set-of-Mark (SoM) visual prompting overlay numbered bounding boxes to boost clicking precision and reduce tokens?
3. What security boundary sandboxes (financial action blocks, bash privilege limits, egress proxies) are mandatory for Computer Use agents?
4. Design visual verification and dynamic polling strategies when handling slow web page loads or unexpected modal popups?
5. Analyze evaluation benchmarks (OS-World, WebArena) measuring GUI Agent completion rates across long-horizon OS workflows?

<details>
<summary>中文题目</summary>

1. 详细剖析 VLM 在全高清屏幕 (1920x1080) 截图上进行像素坐标归一化对齐与高精度视觉定位 (Visual Grounding) 的原理？
2. Set-of-Mark (SoM) 视觉标记技术如何通过在屏幕元素上预先叠加数字编号方框大幅提升模型点击准确率并降低 Token 消耗？
3. 为了保证系统安全，Computer Use 沙箱必须隔离的敏感权限（防恶意点击转账、防命令终端注入、严格网络代理过滤）有哪些？
4. 面对页面加载缓慢 (Network Lag) 或动态弹窗打断时，Agent 的等待重试与动态状态校验策略？
5. OS-World 与 WebArena 等权威基准评测体系对 GUI Agent 在端到端长链条操作系统任务上的评估指标？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/computer-use-gui-agent?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=computer-use-gui-agent)

---

## 13. Contextual Compression · 上下文压缩与提取式重排

> Contextual Compression & Extractive Summarization is a post-retrieval distillation pipeline that strips redundant tokens from retrieved passages prior to generation;

**Interview questions**

1. Detail the LLMLingua compression algorithm using small language model conditional perplexity to evaluate token information density?
2. How does LongLLMLingua mitigate 'Lost in the Middle' via query-aware dynamic budget allocation across retrieved passages?
3. Compare Sentence-level Extractive Compression vs Token-level Pruning in maintaining semantic coherence?
4. How robust are modern LLMs when reasoning over compressed, grammatically fragmented text excerpts?
5. Weigh the latency cost of running compression inference vs net latency savings on downstream LLM prefill and generation?

<details>
<summary>中文题目</summary>

1. 详细剖析 LLMLingua 基于小模型困惑度 (Perplexity / Conditional Perplexity) 衡量 Token 关键信息量的压缩算法机制？
2. LongLLMLingua 如何通过对长文档切块进行 Query-aware 动态预算分配（Dynamic Budget Allocation）解决“迷失在中间”问题？
3. 句子级抽取式压缩 (Sentence-level Extractor) vs Token 级硬截断压缩 (Token-level Pruning) 在语义连贯性上的对比？
4. 压缩后文本可能出现语法破碎或代词断裂，现代 LLM 对不完整语法的鲁棒性容忍度与准确率评估？
5. 在端到端 RAG 延迟链条中，引入 LLMLingua 压缩所需的推理耗时 vs 减少下游生成 LLM TTFT/TPOT 的净收益权衡？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/contextual-compression?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=contextual-compression)

---

## 14. Corrective RAG (CRAG) · Corrective RAG (CRAG) 纠偏检索

> Corrective Retrieval-Augmented Generation (CRAG) is a robust RAG architecture that deploys a lightweight Retrieval Evaluator to score the confidence of internal database retrieval;

**Interview questions**

1. Describe the confidence classification logic and threshold rules in CRAG's Retrieval Evaluator?
2. How does CRAG's Knowledge Refinement algorithm decompose, filter, and recompose fine-grained passage contents?
3. When falling back to Web Search (Tavily/Bing), how to clean, filter, and rerank live web search snippets on the fly?
4. Compare CRAG's modular 3-branch pipeline vs Self-RAG's internal token generation in production maintainability?
5. In air-gapped corporate networks without internet access, how should CRAG gracefully degrade when internal retrieval fails?

<details>
<summary>中文题目</summary>

1. 详细描述 CRAG 检索评估器 (Retrieval Evaluator, 如微调的 T5/DeBERTa) 的评分置信度分级阈值判别逻辑？
2. CRAG 中的知识精炼 (Knowledge Refinement: Decompose -> Filter -> Recompose) 算法是如何剔除切块内部细粒度噪音的？
3. 当进入 'Incorrect' 分支触发 Web Search (如 Tavily / Bing API) 时，如何对网络搜索结果进行即时清洗与重排？
4. 对比 CRAG (明确三分支评估与 Web 兜底) 与 Self-RAG (自省 Token 解码) 在工程落地实现与系统解耦上的优劣？
5. 在内部企业局域网（严格禁止外网访问）环境中，CRAG 的 'Incorrect' 外部搜索兜底分支应如何优雅降级（如转人工或明确告知无权限）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/corrective-rag-crag?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=corrective-rag-crag)

---

## 15. Cross-Encoder Architecture · Cross-Encoder 交叉重排架构

> Cross-Encoder is a second-stage re-ranking architecture that concatenates query $q$ and candidate document $d$ into a single combined sequence (e.g., [CLS] query [SEP] doc [SEP]) fed into a deep Transformer encoder;

**Interview questions**

1. Contrast computational complexity between Bi-Encoder and Cross-Encoder: why can Cross-Encoder not build offline ANN indices?
2. Why must production RAG pipelines enforce a two-stage 'Bi-Encoder coarse recall + Cross-Encoder fine rerank' architecture?
3. How to batch and optimize Cross-Encoder quadratic attention overhead when candidate chunks span up to 1024 tokens?
4. Evaluate out-of-domain generalization performance of fine-tuned Cross-Encoder models (e.g., ms-marco-MiniLM)?
5. How does knowledge distillation transfer Cross-Encoder teacher ranking capabilities into compact Bi-Encoder student models?

<details>
<summary>中文题目</summary>

1. 对比 Bi-Encoder 与 Cross-Encoder 在计算复杂度上的根本差异：为什么 Cross-Encoder 无法预先离线建立 ANN 索引？
2. 为什么在工业级 RAG 架构中必须采用“Bi-Encoder 粗排召回 + Cross-Encoder 精排重排”的两阶段管道？
3. Cross-Encoder 的序列长度二次方复杂度 ($O((L_q + L_d)^2)$) 在候选切块长达 1024 Token 时如何进行批处理 (Batching) 延迟优化？
4. 基于微调的 Cross-Encoder 模型（如 ms-marco-MiniLM-L-6-v2）在领域迁移（Domain Adaptation）中的泛化表现？
5. 如何利用知识蒸馏 (Knowledge Distillation) 将庞大的 Cross-Encoder 教师模型的精排能力蒸馏给小双塔学生模型？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/cross-encoder-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cross-encoder-architecture)

---

## 16. Dense Bi-Encoder Retrieval · Dense 双塔向量嵌入检索

> Dense Bi-Encoder Retrieval is a first-stage recall paradigm that projects query $q$ and document $d$ independently through Transformer encoders into a unified continuous semantic embedding space $E(q)$ and $E(d)$, scoring similarity via dot product or cosine d…

**Interview questions**

1. Why does the Bi-Encoder architecture enable offline document vector pre-computation and sub-millisecond ANN search?
2. Explain the role of Contrastive Learning (InfoNCE Loss) and Hard Negative Mining in training high-performance embedding models?
3. How does BGE-M3 jointly train a unified representation spanning Dense, Multi-Vector, and Sparse Lexical modalities?
4. How does Matryoshka Representation Learning (MRL) allow embedding truncation from 1536 to 256/512 dimensions without retraining?
5. Compare Mean Pooling vs CLS token representation fidelity on long texts approaching maximum context limits?

<details>
<summary>中文题目</summary>

1. 详细剖析 Bi-Encoder 架构为何能够支持离线预计算文档向量并利用 ANN 索引实现毫秒级快速召回？
2. 对比对比学习 (Contrastive Learning, 如 InfoNCE Loss) 与难负例挖掘 (Hard Negative Mining) 在训练优质 Embedding 模型中的核心地位？
3. BGE-M3 模型如何实现三位一体 (Dense + Multi-Vector ColBERT + Sparse Lexical) 统一表征的联合训练？
4. Matryoshka Representation Learning (MRL 套娃嵌入学习) 是如何支持向量在不重新训练下从 1536 维自由截断至 256/512 维的？
5. 当遇到超出 Embedding 最大窗口限制（如 512 或 8192 Tokens）的长文本时，平均池化 (Mean Pooling) vs CLS Token 的表征质量差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/dense-bi-encoder-retrieval?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dense-bi-encoder-retrieval)

---

## 17. Document Layout Parsing & OCR · 文档版面分析与表格提取

> Document Layout Analysis & Parsing is the visual-semantic preprocessing pipeline in RAG that decodes unstructured complex documents (PDFs, scanned images, Word, presentations);

**Interview questions**

1. Compare rule-based extractors (PyMuPDF/pdfminer) vs deep vision layout parsers (MinerU/Marker/Unstructured) in recall and GPU cost?
2. How to parse multi-page merged-cell financial tables into Markdown/HTML while retaining header metadata across splits?
3. Explain the algorithmic principles behind Reading Order Reconstruction in multi-column academic paper PDFs?
4. What are the advantages of end-to-end VLM PDF parsing (e.g., ColPali, Qwen2-VL) over traditional OCR + Chunking pipelines?
5. What is the fallback error-handling strategy for hybrid vector PDF text and scanned image formula LaTeX extraction?

<details>
<summary>中文题目</summary>

1. 对比基于规则的文本提取 (PyMuPDF/pdfminer) 与基于深度学习的视觉版面解析 (MinerU/Marker/Unstructured) 的召回与计算开销？
2. 对于复杂跨页、合并单元格财报表格，如何将其转换为 Markdown / HTML 并在表头注入元数据以防检索信息丢失？
3. 多栏排版（Multi-column / 双栏学术论文）中的阅读顺序重构 (Reading Order Reconstruction) 算法原理？
4. Vision-Language Model (如 ColPali, GPT-4o, Qwen2-VL) 直接端到端解析 PDF 页面相比传统 OCR + Chunking 的技术优势？
5. PDF 中扫描件公式识别（LaTeX 提取）与矢量字体提取混合管道的容错兜底策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/document-layout-parsing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=document-layout-parsing)

---

## 18. Elo Rating System · Elo 竞技场天梯评分体系

> The Elo Rating System (popularized in AI evaluation by LMSYS Chatbot Arena) is an adaptive ranking framework that computes global capability scores for competing LLMs via aggregated crowdsourced blind pairwise battles;

**Interview questions**

1. Derive the Elo expected win-rate and rating update formulas, and analyze the score surge when an underdog upsets a top model?
2. Compare Bradley-Terry Maximum Likelihood Estimation (MLE) vs online incremental Elo updates in rating convergence?
3. How does non-parametric Bootstrapping compute robust 95% Confidence Intervals across model Elo ratings?
4. How to structure category-specific Elo sub-arenas (e.g., Coding Arena, Hard Prompts, Multi-turn)?
5. How does the Glicko-2 algorithm utilize Rating Deviation (RD) to gracefully handle cold-start variance in new models?

<details>
<summary>中文题目</summary>

1. 推导 Elo 积分期望胜率公式与更新公式，并分析当弱模型爆冷战胜极强模型时的积分跃迁幅度？
2. Bradley-Terry 最大似然估计 (MLE) 与在线 Elo 算法在全量历史对战数据下收敛到全局稳定积分的差异？
3. 引导式自助抽样法 (Bootstrapping with 1000 resamples) 如何为每个模型的 Elo 积分计算 95% 置信区间 (95% CI)？
4. 针对特定垂直领域（如 Coding Arena, Hard Prompts, Multi-turn Arena），如何构建分领域垂直 Elo 天梯榜单？
5. 如何利用 Glicko-2 算法引入评分波动度 (Rating Deviation, RD) 与评分波动率以解决新接入冷启动模型的积分震荡？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/elo-rating-system?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=elo-rating-system)

---

## 19. Few-Shot In-Context Learning · Few-Shot 提示工程与样本挑选

> Few-Shot In-Context Learning (ICL) is a prompt engineering paradigm that guides a pre-trained LLM without updating model weights by supplying 3-5 curated input-output demonstrations (Exemplars) in context;

**Interview questions**

1. Explain the phenomena and causes of 'Ordering Sensitivity' and 'Majority Label Bias' in Few-Shot In-Context Learning?
2. How to implement Dynamic K-NN Exemplar Selection to pull the most semantically relevant demonstrations per query?
3. How does pairing Negative Exemplars (common mistakes + corrections) with Positive Exemplars improve error resilience?
4. Analyze the diminishing returns of scaling demonstration counts from 1 to 20+ against token cost and inference latency?
5. How to leverage OpenAI/Anthropic Prompt Caching on static Few-Shot exemplars to slash costs and TTFT by up to 90%?

<details>
<summary>中文题目</summary>

1. 详细剖析少样本提示中“样本顺序敏感性 (Ordering Sensitivity)”与“多数类别偏置 (Majority Label Bias)”的现象及成因？
2. 基于动态向量检索 (Dynamic K-NN Exemplar Selection) 从样本库中挑选与当前用户 Query 最相似的示例的工程实现？
3. 负向样本 (Negative Exemplars: 展示常见错误及纠偏说明) 与正向样本混合搭配对模型抗错能力的提升？
4. 随着 Few-Shot 样本数量从 1 增加到 20+ 时，准确率边际收益与 Prompt Token 成本及延迟增长的量化权衡？
5. 在生产 API 调用中，如何利用 OpenAI / Anthropic 的 Prompt Caching 特性将固定的 Few-Shot 示例缓存以降低 90% 费用与延迟？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/few-shot-in-context-learning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=few-shot-in-context-learning)

---

## 20. Function Calling Protocol · Function Calling 协议与参数注入

> Function Calling Protocol (Tool Use API) is the standardized industrial interface enabling LLMs to interact deterministically with external software environments;

**Interview questions**

1. Detail the control logic of `tool_choice` modes (`'auto'`, `'none'`, `'required'`, or forcing a specific function)?
2. How does the model inference runtime serialize JSON Schema tool definitions into system prompt / hidden context representations?
3. Analyze how the precision and clarity of function `description` texts dictate LLM Tool Selection Accuracy under tool ambiguity?
4. How to use Semantic Tool Search / Tool Routers to dynamically inject only the top-5 relevant tools when managing 50+ tools?
5. Compare Anthropic Claude's native XML tool use syntax vs OpenAI's JSON Schema function calling schema?

<details>
<summary>中文题目</summary>

1. 详细剖析 API 请求中 `tool_choice` 参数（`'auto'`, `'none'`, `'required'`, 或强制指定特定函数 `{'type': 'function', 'function': {'name': '...'}}`）的控制逻辑？
2. 大模型内部是如何将 `tools` 的 JSON Schema 定义序列化并注入系统提示词/隐藏注意力上下文中的？
3. 函数的 `description` 描述文本的清晰度对大模型“工具选择准确率 (Tool Selection Accuracy)”的决定性影响？
4. 当提供 50+ 个海量候选工具时，如何利用向量检索 (Semantic Tool Search / Router) 在预推理阶段动态精简注入 Top-5 工具以节省 Token？
5. Anthropic Claude Tool Use 原生协议（`<invoke>`, `<parameters>`）与 OpenAI Function Calling 协议在规范上的设计异同？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/function-calling-protocol?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=function-calling-protocol)

---

## 21. GraphRAG Knowledge Graph Search · GraphRAG 知识图谱增强检索

> GraphRAG (Microsoft's Knowledge Graph Augmented RAG architecture) is a next-generation retrieval paradigm fusing LLM information extraction with Hierarchical Community Detection;

**Interview questions**

1. Detail the dual query modes in GraphRAG: Local Search (entity neighborhood traversal) vs Global Search (community summary Map-Reduce)?
2. Explain the role and advantages of the Leiden community detection algorithm in GraphRAG hierarchical graph partitioning?
3. Analyze the massive LLM token ingestion costs of GraphRAG and challenges surrounding incremental graph updates?
4. How do native graph databases (Neo4j / Memgraph) store GraphRAG entities and optimize Cypher graph traversal queries?
5. Compare GraphRAG vs vector RAG benchmarks in domain-specific tasks (e.g., drug interactions, financial anti-money laundering)?

<details>
<summary>中文题目</summary>

1. 详细剖析 GraphRAG 的两种检索模式：Local Search (实体邻域图遍历) 与 Global Search (社区摘要 Map-Reduce 聚合) 的工作流程？
2. Leiden 社区发现算法在 GraphRAG 多层级图划分 (Hierarchical Graph Partitioning) 中的原理与优势？
3. GraphRAG 在全库建索引（知识图谱抽取 + 社区摘要生成）阶段消耗的天量 LLM Token 成本分析与增量更新难题？
4. Neo4j / Memgraph 图数据库在存储 GraphRAG 实体边与执行 Cypher 图查询时的工业级架构选型？
5. 对比 GraphRAG 与传统纯向量检索在特定垂直领域（如医疗药物相互作用、金融反洗钱关联图）的召回指标提升？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/graphrag-knowledge-graph?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=graphrag-knowledge-graph)

---

## 22. HNSW Heuristic Search & Routing · HNSW 启发式邻居选择与路由

> Heuristic Neighbor Selection and Greedy Routing constitute the algorithmic backbone of HNSW's connectivity and precision;

**Interview questions**

1. Detail the anti-isolation mechanism of `keepPrunedConnections` in HNSW heuristic neighbor selection?
2. Analyze how $M$ (16-64) and $efConstruction$ (100-400) impact index construction time and memory footprint non-linearly?
3. Describe the typical empirical curve of Recall@10 vs QPS throughput as $efSearch$ scales from 16 to 256?
4. Why is dynamic node deletion in HNSW notoriously hard? Explain soft tombstones and graph repair reconnection?
5. How does fine-grained node-level read/write locking ensure thread safety during concurrent vector ingestion?

<details>
<summary>中文题目</summary>

1. 详细剖析 HNSW 启发式邻居选择算法中 `keepPrunedConnections` (保留被剪枝连接) 参数的防孤立节点机制？
2. 分析参数 $M$ (如 16~64) 与 $efConstruction$ (如 100~400) 对建索引耗时与内存图大小的非线性增长影响？
3. 检索阶段调节 $efSearch$ 从 16 增加到 256 时，Recall@10 与 QPS 吞吐量的典型量化曲线规律？
4. HNSW 节点动态删除 (Node Deletion) 为何是业界难题？软删除墓碑标记 (Tombstone) 与图修复重连机制？
5. 并发插入 (Concurrent Ingestion) 时，如何通过细粒度读写锁 (Fine-grained Node Locking) 保证图结构线程安全？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/hnsw-heuristic-search?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hnsw-heuristic-search)

---

## 23. HNSW Multi-Layer Skip Graph · HNSW 跳表多层图结构

> Hierarchical Navigable Small World (HNSW) is an ANN graph index that combines the probabilistic multi-layer concept of Skip Lists with Navigable Small World (NSW) graph topology;

**Interview questions**

1. Derive the probability formula for assigning a node to layer $l$, and explain the normalization factor $m_L = 1/\ln(M)$?
2. Detail the two-stage greedy routing workflow from top Entry Point down to bottom Layer 0?
3. Why does HNSW possess both a high clustering coefficient and extremely short average path lengths?
4. Analyze the breakdown of HNSW memory consumption between neighbor pointer arrays and raw vector payloads?
5. Compare the architectural differences between HNSW and DiskANN (Vamana graph) in RAM footprint vs SSD random IO?

<details>
<summary>中文题目</summary>

1. 推导 HNSW 中节点分配到第 $l$ 层的概率公式，并解释归一化因子 $m_L = 1/\ln(M)$ 的数学作用？
2. 详细描述从全局入口点 (Enter Point) 顶层向下贪心路由到底层 (Layer 0) 的两阶段搜索算法流程？
3. 为什么 HNSW 能够兼具高聚类系数 (High Clustering Coefficient) 与极短平均路径长度 (Short Path Length)？
4. HNSW 索引内存开销中，邻居指针数组与原始向量存储的占比分析？
5. 对比 HNSW 与 DiskANN (Vamana 图索引) 在单机内存占用与 SSD 随机 IO 吞吐上的根本架构差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/hnsw-graph-structure?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hnsw-graph-structure)

---

## 24. HyDE Hypothetical Embeddings · HyDE 假设性文档嵌入

> Hypothetical Document Embeddings (HyDE) is a Zero-Shot retrieval enhancement technique that bridges semantic distribution gaps between short queries and long documents;

**Interview questions**

1. Why does HyDE successfully retrieve correct real documents even when the synthesized hypothetical document contains factual hallucinations?
2. Under what specific scenarios does HyDE completely fail (e.g., proprietary acronyms, private code repos, cold-start entities)?
3. How can small language models (1B-3B SLMs) mitigate the TTFT latency penalty introduced by HyDE's preliminary generation step?
4. How does Multi-HyDE (sampling 3-5 hypothetical docs and averaging embeddings) stabilize retrieval variance over single-doc HyDE?
5. What is the engineering heuristic for weighted fusion ($v_{\text{final}} = \alpha v_q + (1-\alpha) v_{\text{hypo}}$) of query and hypothetical vectors?

<details>
<summary>中文题目</summary>

1. 详细剖析 HyDE 为什么在假设文档包含事实性幻觉（Factually Incorrect）时依然能成功召回正确的真实文档？
2. HyDE 在哪些典型场景下会彻底失效（如特定生僻专属名词、私有库专有代码、冷启动实体）？
3. 引入 HyDE 后，首字响应时间 (TTFT) 增加了额外一次完整 LLM 生成的延迟开销，如何通过小模型（如 1B-3B SLM）进行延迟优化？
4. 多假设文档采样 (Multi-HyDE: 生成 3-5 篇假想文档并对 Embedding 取均值) 相比单假设文档的召回方差稳定性提升？
5. HyDE 生成的假想文档向量与原始 Query 向量进行加权融合 ($v_{\text{final}} = \alpha v_q + (1-\alpha) v_{\text{hypo}}$) 的工程调参策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/hyde-hypothetical-embeddings?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hyde-hypothetical-embeddings)

---

## 25. Industrial Benchmarks Suite · 工业标准评测套件 (MMLU/HumanEval)

> Standard Industrial Evaluation Benchmarks Suite is the canonical collection of open-source benchmark datasets that comprehensively quantify foundational and aligned LLM cognitive capabilities across domains;

**Interview questions**

1. Derive the unbiased pass@k formula $\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$ for HumanEval evaluation?
2. Explain the standard 5-shot MMLU evaluation protocol and score improvements unlocked by Chain-of-Thought reasoning?
3. How does IFEval evaluate 25+ verifiable instruction-following constraints using deterministic Python checkers rather than LLM judges?
4. Analyze root causes of Benchmark Contamination, N-gram de-contamination detection algorithms, and dynamic benchmarks?
5. Design automated continuous evaluation CI pipelines using lm-evaluation-harness, LightEval, or DeepEval?

<details>
<summary>中文题目</summary>

1. 详细剖析 HumanEval 的 pass@k 计算公式：为什么必须使用无偏估计公式 $\text{pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$ 进行蒙特卡洛采样？
2. MMLU (Massive Multitask Language Understanding) 的 5-shot 评估标准与 Chain-of-Thought (CoT) 解题模式下的得分跃迁？
3. IFEval (Instruction Following Evaluation) 如何通过确定性 Python 规则（不依赖 LLM 裁判）评估 25+ 种可验证指令遵循能力？
4. 评测集污染 (Benchmark Contamination) 的成因、N-gram 重合度检测算法与对抗性动态基准 (Dynamic Benchmarks) 的兴起？
5. 使用轻量评测框架（如 lm-evaluation-harness / LightEval / DeepEval）搭建企业内部自动化持续评测流水线的架构？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/industrial-benchmarks-suite?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=industrial-benchmarks-suite)

---

## 26. Instruction Hierarchy & System Prompt · 指令分层与 System Prompt 架构

> Instruction Hierarchy is a formal security architecture and prompt design framework that establishes strict privilege tiers across conflicting inputs;

**Interview questions**

1. How do OpenAI and Anthropic train models during RLHF/SFT to enforce role priority boundaries across `system`, `user`, and `tool` roles?
2. What is the standard modular architecture for writing production System Prompts (persona, workflow, format, guardrails)?
3. How does strict XML tagging (`<context>...</context>`) physically demarcate retrieved passages as non-executable read-only payload?
4. What is the arbitration and refusal protocol when user instructions directly contradict hard system prompt constraints?
5. How to prevent System Prompt attention decay and forgetting across massive 100K+ token conversational contexts?

<details>
<summary>中文题目</summary>

1. 详细剖析 OpenAI / Anthropic 在 RLHF 与指令微调中如何训练模型识别不同角色层级 (`system` vs `user` vs `tool`) 的对抗训练机制？
2. 生产 System Prompt 的模块化架构（角色设定、业务流程、输出规范、边界约束与防御注入）标准编写范式？
3. 使用明确的分隔符（如 `<context>` ... `</context>` 或 XML 标签）如何物理标记外部检索文档为不可执行的只读数据？
4. 当用户指令与系统提示词发生直接冲突（如用户要求绕过敏感限制）时，模型的优先级仲裁与安全拒答策略？
5. 如何防止模型在长上下文（100K+ tokens）后半段发生“系统提示词注意力衰减与遗忘 (Prompt Decay)”？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/instruction-hierarchy-system?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=instruction-hierarchy-system)

---

## 27. Instructor Pydantic Validation · Instructor 运行时重试自愈

> Instructor is a client-side structured extraction and runtime self-healing validation framework for Python; built on Pydantic validators (@field_validator), it evaluates deep domain-specific business rules (e.g., date ranges, numeric bounds, inter-field depend…

**Interview questions**

1. Detail the self-healing retry loop mechanism in Instructor combining `max_retries` with `ValidationError` prompt injection?
2. What extraction modes does Instructor support (Tool Calling, JSON Mode, Markdown JSON) and their model compatibility matrix?
3. How to combine Pydantic `@field_validator` with live database calls to validate field existence and trigger model retries?
4. How does streaming partial JSON extraction (`create_partial`) parse and yield incomplete Pydantic models in real-time?
5. Compare server-side CFG masking vs client-side Instructor retries across latency, token costs, and rule expressiveness?

<details>
<summary>中文题目</summary>

1. 详细剖析 Instructor 如何通过 `max_retries` 机制与 `ValidationError` 回填构建闭环自愈重试流？
2. Instructor 支持的底层提取模式（Tool Calling, JSON Mode, Markdown JSON, Structured Outputs）及其适用模型矩阵？
3. 使用 Pydantic `@field_validator` 结合外部 API 校验（如查询数据库验证用户 ID 是否存在）并在不合规时引导模型重试？
4. 流式结构化抽取 (Streaming Partial JSON / `client.chat.completions.create_partial`) 在前端实时渲染中的实现原理？
5. 对比纯服务端 CFG 状态机掩码与客户端 Instructor 重试方案在延迟开销、API 调用成本与规则表达力上的差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/instructor-pydantic-validation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=instructor-pydantic-validation)

---

## 28. IVF Inverted File Indexing · IVF 倒排网格与残差量化

> Inverted File Index (IVF) is a space-partitioning (Voronoi Partitioning) coarse-clustering ANN retrieval framework; during indexing, K-Means partitions the vector space into $N_{\text{list}}$ Voronoi cells, each managing an Inverted List of vectors within that…

**Interview questions**

1. Analyze the trade-offs of $N_{\text{list}}$ and $N_{\text{probe}}$ on search latency, QPS, and recall rate?
2. What is the Boundary Effect, and why must $N_{\text{probe}}$ be increased when target vectors lie near Voronoi cell boundaries?
3. Why does quantizing residual vectors $r = x - c$ yield lower Mean Squared Error (MSE) than quantizing raw vectors $x$?
4. In production continuous data ingestion, how to handle centroid data drift and execute graceful re-clustering?
5. Compare IVF-Flat, IVF-SQ8, and IVF-PQ benchmarks across recall, memory footprint, and build time?

<details>
<summary>中文题目</summary>

1. 分析 IVF 中参数 $N_{\text{list}}$ (聚类中心数) 与 $N_{\text{probe}}$ (探测桶数) 对检索延迟、QPS 与 Recall 的制约关系？
2. 什么是边界效应 (Boundary Effect)？为什么当目标向量恰好落在 Voronoi 胞元交界处时必须增大 $N_{\text{probe}}$？
3. 为什么对残差 $r = x - c$ 进行 PQ 量化相比对原始向量 $x$ 直接量化具有更小的均方误差 (MSE)？
4. 在数据持续动态插入 (Dynamic Ingestion) 的生产场景下，IVF 质心发生数据漂移 (Data Drift) 时应如何优雅重新训练 (Re-clustering)？
5. 对比 IVF-Flat、IVF-SQ8 与 IVF-PQ 在检索召回率、内存占用与构建速度上的完整对比基准？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/ivf-voronoi-indexing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ivf-voronoi-indexing)

---

## 29. Judge Position Bias Mitigation · Position Bias 位置偏置消除

> Position Bias Mitigation is the algorithmic debiasing framework in LLM-as-a-Judge evaluations that neutralizes the systematic cognitive bias where judge models inherently favor specific presentation positions (frequently exhibiting a 60%-70% win-rate bias towa…

**Interview questions**

1. Derive the truth table for declaring Model A win, Model B win, or Tie across the 4 combinations of the Bidirectional Swap Test?
2. Explain the underlying cause of Position Bias stemming from autoregressive causal attention masking and early-token attention anchoring?
3. How does Random Permutation Averaging mitigate position bias when ranking lists of 3+ candidates simultaneously?
4. Empirically evaluate the efficacy of prompt debiasing instructions ('Ignore presentation order') on reducing position bias?
5. How to estimate a global position bias coefficient on a sub-sample to calibrate large batches without doubling evaluation costs?

<details>
<summary>中文题目</summary>

1. 详细推导双向调换测试 (Bidirectional Swap) 中判定 Model A 获胜、Model B 获胜或判定为平局 (Tie) 的四种布尔组合真值表？
2. 从自回归 Transformer 因果注意力掩码 (Causal Attention Mask) 的底层机制分析位置偏置（前向注意力偏好）产生的本质原因？
3. 多候选列表 (Listwise > 3 items) 评估中的位置偏置缓解：多次随机全排列 (Random Permutation Averaging) 的统计学收敛性？
4. 在评估 Prompt 中显式加入“请注意忽略回答的先后呈现顺序，仅关注内容本质”等指令对削弱位置偏置的实际效果评估？
5. 位置双向对调测试将 API 评测费用与耗时翻倍 (2x Tokens)，如何在小批量样本上计算位置偏置系数并在大批量上进行统计校准？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/judge-position-bias?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=judge-position-bias)

---

## 30. Krippendorff's Alpha · Krippendorff's Alpha 多标注者信度

> Krippendorff's Alpha ($\alpha$) is a generalized statistical reliability coefficient that measures agreement among any arbitrary number of raters, flexibly accommodating missing observations across any metric scale (Nominal, Ordinal, Interval, Ratio);

**Interview questions**

1. Derive the disagreement formula of Krippendorff's Alpha under an Interval Metric scale ($d(v, c) = (v - c)^2$)?
2. How does Krippendorff's Alpha statistically handle missing observations across sparse evaluation matrices?
3. Analyze standard methodological criteria ($\alpha \ge 0.800$ for definitive conclusions, $\alpha \ge 0.667$ for tentative conclusions) in AI evaluation?
4. Compare statistical robustness between Krippendorff's Alpha and Fleiss' Kappa under sample volatility?
5. How to apply Krippendorff's Alpha to detect and filter out adversarial spammers in RLHF human annotation pools?

<details>
<summary>中文题目</summary>

1. 详细推导 Krippendorff's Alpha 在定距尺度 (Interval Metric: $d(v, c) = (v - c)^2$) 下的分歧度计算公式？
2. Krippendorff's Alpha 如何优雅处理稀疏标注矩阵（即并非每个样本都被所有评审员打分）中的缺失值？
3. 学术界与工业界通用的信度判定准则（$\alpha \ge 0.800$ 结论高度可信，$\alpha \ge 0.667$ 可得出初步暂定结论）的严谨性分析？
4. 对比 Krippendorff's Alpha 与 Fleiss' Kappa 在处理极端样本量与多标注者方差波动时的稳健性差异？
5. 在大模型微调 RLHF 数据集质检中，如何基于 Alpha 系数自动剔除打分方差过大的低质量标注工（Spammer Detection）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/krippendorffs-alpha?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=krippendorffs-alpha)

---

## 31. LangGraph Conditional Edges · LangGraph 条件边与动态路由

> Conditional Edges & Dynamic Routing is the core decision-branching mechanism in LangGraph state graphs that enables non-deterministic dynamic state transitions based on runtime state inspection;

**Interview questions**

1. Detail the implementation logic of standard LangGraph `should_continue(state)` routing functions (checking `tool_calls` presence to route to `tools` vs `END`)?
2. Explain the architectural decoupling of mapping router return values to concrete node names via `path_map` dictionaries?
3. How to use lightweight SLM classifiers or embedding similarity within conditional edge routers to reduce latency?
4. What is the fallback routing configuration when a router function encounters unhandled exceptions or emits invalid keys?
5. How does dynamic fan-out routing trigger parallel downstream nodes and synchronize merged state reducers?

<details>
<summary>中文题目</summary>

1. 详细分析典型 LangGraph 循环路由函数 `should_continue(state)` 的实现逻辑（判断 `tool_calls` 是否为空以决定流转至 `tools` 还是 `END`）？
2. 基于 `path_map: dict` 字典进行路由返回值到真实节点名称的映射解耦设计？
3. 如何利用结构化小分类模型（SLM Classifier）或嵌入相似度在条件边中构建确定性路由以减少大模型推理耗时？
4. 当路由函数由于意外异常或返回非法分支标识时，图执行引擎的默认兜底分支 (Fallback Route) 配置？
5. 在条件边中动态分发多个并行分支节点 (Dynamic Fan-out / Parallel Map-Reduce) 的状态合并与同步机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/conditional-edges-routing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=conditional-edges-routing)

---

## 32. LangGraph State Machine · LangGraph 状态图与持久化

> LangGraph is an enterprise-grade graph state-machine orchestration framework engineered specifically for complex, cyclic, multi-agent LLM systems;

**Interview questions**

1. Detail the 3 pillars of LangGraph: Typed State schemas (with Reducers), Nodes (executors), and Edges (transitions)?
2. How do State Reducers (e.g., `Annotated[list, operator.add]`) enable incremental list appending instead of destructive overwrites?
3. How does Checkpointer persistence (PostgresSaver) achieve fault-tolerant workflow resumption from exact failure nodes?
4. Explain the mechanics of `interrupt_before`/`interrupt_after` and Time Travel state modification in Human-in-the-Loop workflows?
5. Compare LangGraph vs AutoGen vs CrewAI across graph control granularity, deterministic constraint enforcement, and maintainability?

<details>
<summary>中文题目</summary>

1. 详细剖析 LangGraph 中 `StateGraph` 的核心三要素：Typed State (状态模式与 Reducer)、Nodes (执行算子) 与 Edges (边流转)？
2. 状态 Reducer 机制（如 `Annotated[list, operator.add]`）在实现消息列表增量追加而非全量覆盖中的数学作用？
3. Checkpointer 检查点持久化机制（MemorySaver / PostgresSaver）如何实现工作流在任意节点崩溃后的断点续跑 (Fault-tolerant Resume)？
4. Human-in-the-Loop (人工介入审批) 中的 `interrupt_before` / `interrupt_after` 与状态手动编辑重放 (Time Travel) 实现原理？
5. 对比 LangGraph 与 AutoGen、CrewAI 在图控制粒度、确定性流程约束与生产可维护性上的架构差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/langgraph-state-machine?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=langgraph-state-machine)

---

## 33. Llama Guard 3 Safety Moderation · Llama Guard 3 安全审查模型

> Llama Guard 3 (Meta's open-source safety moderation model fine-tuned on Llama-3.1-8B/1B) is a specialized discriminator for real-time conversational compliance auditing;

**Interview questions**

1. Detail Llama Guard 3's Harm Taxonomy risk categories (S1 through S13 safety standards)?
2. How to customize safety policy definitions in Llama Guard's prompt to enforce enterprise-specific compliance rules?
3. What are the structural prompt differences between Input Moderation and Output Moderation in Llama Guard?
4. How to apply sliding window chunking to perform real-time safety moderation over streaming output tokens?
5. Compare Llama Guard 3 1B vs 8B variants in classification F1-score and end-to-end inference latency?

<details>
<summary>中文题目</summary>

1. 详细梳理 Llama Guard 3 的 Harm Taxonomy 核心分类体系（S1 至 S13 类安全危害标准）？
2. 如何通过修改系统提示词中的危害策略定义 (Safety Policy Customization)，使 Llama Guard 支持企业自定义私有违规分类？
3. 输入审核 (Input Moderation) 与输出审核 (Output Moderation) 的 Prompt 结构与角色标记差异？
4. 在流式响应 (Streaming Output) 场景下，如何结合滑动窗口对实时 Token 流进行分块安全审核以避免延迟阻断？
5. Llama Guard 3 1B 轻量化边缘版 vs 8B 版在分类准确率 (F1-score) 与端到端推理延迟上的实测基准？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/llama-guard-moderation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=llama-guard-moderation)

---

## 34. MicroVM & Container Sandbox · MicroVM 与容器沙箱隔离

> MicroVM & Container Sandboxing is the underlying virtualization infrastructure enabling safe, isolated execution of untrusted LLM-generated code for Code Interpreters and Autonomous Agents;

**Interview questions**

1. Compare Docker containers, Google gVisor, and AWS Firecracker MicroVMs across startup latency, memory overhead, and isolation tier?
2. How does Firecracker's stripped device model enable 5ms cold-starts and thousands of concurrent MicroVMs per node?
3. How does gVisor's Sentry intercept Linux syscalls in user-space, and what is its performance penalty on high-IO tasks?
4. Design sandbox network egress policies (`network: none` vs domain-allowlist proxies) to prevent malicious data exfiltration?
5. How do AI sandbox platforms (E2B, Modal) implement filesystem snapshotting and instant microVM state forking?

<details>
<summary>中文题目</summary>

1. 详细对比 Docker 容器、Google gVisor 与 AWS Firecracker MicroVM 在启动延迟、显存/内存开销与安全隔离等级上的三维对比？
2. Firecracker 如何通过极简精简设备模型 (Minimal Device Model) 实现 5 毫秒级的瞬时 MicroVM 启动与千级实例单机并发？
3. gVisor 的 Sentry 架构如何在用户态截获系统调用 (Syscall Interception)，并分析其对高频文件 IO 密集型任务的性能衰减？
4. 沙箱网络隔离策略（完全禁用外部公网访问 `network: none`，或仅放行白名单域名网关）在防数据外排 (Data Exfiltration) 中的设计？
5. E2B (开源沙箱平台) 与 Modal 如何为 AI Agent 提供状态快照 (Snapshot & Fork) 与瞬时环境克隆？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/firecracker-gvisor-sandbox?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=firecracker-gvisor-sandbox)

---

## 35. Multi-Agent Role-Playing · Multi-Agent 角色扮演与群聊协同

> Multi-Agent Role-Playing & Group Chat Collaboration is a systemic framework that decomposes complex objectives across specialized expert Agent personas (e.g., Product Manager, Architect, Software Engineer, QA Tester) coordinated via structured group chat topol…

**Interview questions**

1. Detail the broadcast routing and dynamic next-speaker selection mechanism in AutoGen's `GroupChatManager`?
2. How does MetaGPT enforce SOP artifacts (PRD, Architecture, API specs) across multi-agent communications to prevent unconstrained chat drift?
3. How does Multi-Agent Debate leverage adversarial cross-examination across 2-3 agents to eliminate factual hallucinations?
4. How to resolve quadratic token cost explosions in group chats by adopting Shared Blackboard architectures for selective message broadcasting?
5. Design an Arbitrator intervention protocol to break multi-agent deadlocks where agents endlessly pass uncompleted tasks in circles?

<details>
<summary>中文题目</summary>

1. 详细剖析 AutoGen 中 `GroupChatManager` 基于大模型动态选择下一位发言者 (Next Speaker Selection) 的广播路由机制？
2. MetaGPT 如何通过引入标准软件工程产物（SOP 标准作业程序: PRD -> 架构图 -> API 设计 -> 代码）约束多 Agent 通信以防闲聊跑题？
3. 多智能体辩论 (Multi-Agent Debate) 机制如何通过让 2-3 个 Agent 相互质疑与事实交叉核验减少整体事实性幻觉？
4. 群聊消息广播中上下文 Token 呈二次方级激增的成本危机：如何通过共享黑板架构 (Shared Blackboard) 进行消息隔离？
5. 多智能体协作中出现“互踢皮球死锁 (Deadlock / Passing the buck)”时的仲裁器 (Arbitrator) 介入与强制决策机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/multi-agent-role-playing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multi-agent-role-playing)

---

## 36. Multi-Turn Tool Conversation · 多轮工具交互状态链管理

> Multi-Turn Tool Conversation State Management is the protocol that preserves conversational sequence integrity and causal consistency across complex multi-step tool-assisted LLM interactions;

**Interview questions**

1. Detail the standard message role flow: `user -> assistant(tool_calls) -> tool(tool_call_id) -> assistant(final)` and field requirements?
2. How to apply sliding window pruning and summarization to truncate massive historical tool outputs exceeding context budgets?
3. How to perform schema projection and data extraction before injecting massive raw API responses back into the model context?
4. How to persist pending `tool_call_id` states and asynchronous webhook callbacks across distributed Redis/SQL architectures?
5. How to achieve idempotent workflow resumption during network drops across multi-turn tool calling pipelines?

<details>
<summary>中文题目</summary>

1. 详细梳理标准 OpenAI 消息链结构：`user -> assistant(tool_calls) -> tool(tool_call_id) -> assistant(final)` 的角色字段流转规则？
2. 当多轮工具交互产生的上下文累积达到数十万 Token 时，如何对历史工具调用结果执行滑动摘要与修剪 (Pruning)？
3. 如果工具返回了大量冗余 JSON 数据（如 5000 行全量 API 报文），如何在回填给模型前进行字段投影 (Field Projection) 与提纯？
4. 分布式或异步任务中，如何利用 Redis / 数据库持久化保存尚未完成的 `tool_call_id` 状态与回调钩子？
5. 多轮工具调用中发生网络断开时，如何基于状态恢复机制实现幂等重试 (Idempotent Resume)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/multi-turn-tool-conversation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multi-turn-tool-conversation)

---

## 37. NeMo Guardrails Programmable Rails · NeMo Guardrails 可编程护栏

> NeMo Guardrails (NVIDIA's open-source programmable safety framework for LLMs) is a runtime conversational middleware using the declarative language Colang to govern dialog state flows and safety policies;

**Interview questions**

1. Detail the core components of Colang 2.0 (flows, user/bot intents, actions) and its underlying state-machine runtime?
2. How does NeMo Guardrails leverage embedding models for real-time canonical form intent mapping and boundary enforcement?
3. How do Dialog Rails detect off-topic drifting during multi-turn procedures and redirect the conversation back to the active workflow?
4. Analyze runtime latency overhead introduced by NeMo Guardrails and how to configure asynchronous parallel evaluation rails?
5. Compare NeMo Guardrails vs Llama Guard / Guardrails AI in policy expressiveness and architectural complexity?

<details>
<summary>中文题目</summary>

1. 详细剖析 Colang 2.0 声明式语法的核心元素（flows、user intent、bot intent、actions）及其状态机运行机制？
2. NeMo Guardrails 如何结合自研嵌入模型对用户输入进行实时意图聚类 (Canonical Form Mapping) 与边界阻断？
3. 对话流护栏 (Dialog Rails) 如何防止用户在业务办理过程中进行话题漂移 (Topic Drifting) 并温和拉回主流程？
4. NeMo Guardrails 运行时拦截对系统平均响应延迟 (Latency overhead) 的影响及如何配置异步并行护栏？
5. 对比 NeMo Guardrails 与 Llama Guard / Guardrails AI 在策略表达力与架构复杂度上的选型？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/nemo-guardrails?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=nemo-guardrails)

---

## 38. OpenAI Strict Structured Outputs · OpenAI Strict JSON Schema 输出

> Structured Outputs (Strict JSON Schema) is a deterministic constrained decoding engine provided by OpenAI and leading API platforms guaranteeing that LLM outputs conform 100% strictly to a developer-supplied JSON Schema;

**Interview questions**

1. Detail the underlying mechanism of `strict: true` JSON schema mode and its fundamental difference from legacy `json_object` mode?
2. What are the strict constraints on JSON Schemas in Strict Mode (all fields required, nullable types via `anyOf`)?
3. Analyze schema pre-compilation cold-start latency on initial requests and backend grammar caching optimization?
4. How to handle safety refusals gracefully by modeling an explicit `refusal` field within the target JSON Schema?
5. Compare native server-side Structured Outputs vs client-side retry libraries (Instructor/LangChain) in reliability and cost?

<details>
<summary>中文题目</summary>

1. 详细剖析 `response_format: { type: 'json_schema', strict: true }` 的底层实现原理与传统 `json_object` 模式的根本区别？
2. Strict Mode 下对 JSON Schema 的严格限制（如所有字段必须 `required`、不支持 `optional` 字段但允许 `anyOf: [{type: 'string'}, {type: 'null'}]`）？
3. Schema 预编译 (Schema Pre-processing) 对首个请求可能产生的轻微预热延迟 (Cold-start compile time) 以及缓存优化机制？
4. 当模型在受限解码过程中遇到无法表达的内容时，如何通过在 Schema 中显式定义 `refusal` 字段优雅处理拒答？
5. 对比 OpenAI 原生 Structured Outputs 与客户端自愈重试库 (如 Instructor / LangChain OutputParser) 的性能与成本？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/openai-structured-outputs?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=openai-structured-outputs)

---

## 39. Outlines FSM & CFG Masking · Outlines 状态机掩码硬解码

> Outlines / Guided Generation is a constrained decoding engine for open-source LLM inference frameworks (vLLM, SGLang, HuggingFace) that compiles Regular Expressions into Deterministic Finite Automata (DFAs) or leverages Context-Free Grammars (CFG) for real-tim…

**Interview questions**

1. Derive the compilation pipeline converting regular expressions to DFAs and mapping token vocabulary state transitions?
2. Why does multi-token subword tokenization not break continuous state tracking within the DFA transition graph?
3. How do vLLM and SGLang integrate Outlines / XGrammar operators to achieve batched GPU-accelerated grammar masking?
4. Analyze compilation overhead and memory optimization when compiling deeply nested recursive schemas into Pushdown Automata?
5. How to extract calibrated probabilities when constraining model outputs strictly to binary choices ('yes'/'no')?

<details>
<summary>中文题目</summary>

1. 推导正则表达式编译为确定性有限状态自动机 (DFA) 并映射至 Tokenizer 词表状态转移索引的完整流程？
2. 为什么跨多 Token 词汇（如浮点数切分为多个子词）不会破坏 DFA 状态机的连续状态跟踪？
3. vLLM 与 SGLang 如何通过深度集成 Outlines / XGrammar 算子实现批量请求并发下的高效 GPU 共享掩码？
4. 复杂的嵌套递归 JSON Schema 在编译为推导状态机 (Pushdown Automaton) 时的计算开销与内存优化？
5. 当对输出施加强约束（如限制只能输出 'yes' 或 'no'）时，模型在受限分布下的校准概率 (Calibrated Probability) 如何提取？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/outlines-cfg-masking?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=outlines-cfg-masking)

---

## 40. Pairwise A/B Comparison · Pairwise A/B 对比评估范式

> Pairwise Evaluation (Head-to-Head Comparison) is an assessment paradigm where anonymized responses from two candidate models (Model A vs Model B) for the same prompt are presented side-by-side to a judge LLM (or human annotator) to declare a winner (Model A, M…

**Interview questions**

1. Derive the Bradley-Terry formula $P(A > B) = \frac{e^{r_A}}{e^{r_A} + e^{r_B}}$ converting pairwise win rates into global latent capability scores?
2. Why must presentation order be systematically swapped (running both (A, B) and (B, A)) to neutralize Position Bias?
3. How should tie decisions (both models excellent or both failing) be parameterized and calculated in net win rates?
4. How to resolve the $O(N^2)$ pairwise combination explosion when evaluating $N$ models via Swiss-system tournament matching?
5. How does the MT-Bench benchmark implement Pairwise evaluation to assess multi-turn conversational competence?

<details>
<summary>中文题目</summary>

1. 推导 Bradley-Terry 概率模型公式 $P(A > B) = \frac{e^{r_A}}{e^{r_A} + e^{r_B}}$ 如何将成对 A/B 比较胜率转换为全局能力潜变量得分？
2. 为什么在 Pairwise 评估中，必须强制交换候选回答的呈现顺序（运行两轮：(A, B) 与 (B, A)）以消除位置偏置？
3. 当遇到平局 (Tie: 两个模型表现均优秀或均糟糕) 时，平局判定的置信度门限与胜率统计处理方式？
4. Pairwise 评估在比较 $N$ 个候选模型时面临的 $O(N^2)$ 组合爆炸问题：如何通过瑞士轮 (Swiss-system) 或随机匹配减少评测轮次？
5. MT-Bench 基准测试中如何使用 Pairwise 对比评估多轮对话交互能力 (Multi-turn Conversation Capability)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/pairwise-evaluation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pairwise-evaluation)

---

## 41. Parallel Function Calling · Parallel 并行工具调用

> Parallel Function Calling is an inference capability where the LLM emits multiple independent tool call requests within a single autoregressive step (returning multiple items in the tool_calls array, each with a unique id, e.g., querying weather for Beijing, S…

**Interview questions**

1. How does the model ensure parameter independence across multiple `tool_calls` items emitted within a single completion?
2. Design concurrent scheduling and exception handling in host engines via `asyncio.gather()` over parallel tool calls?
3. When one parallel tool call fails or times out, how to assemble partial successes with the specific error payload without crashing the batch?
4. How to explicitly disable parallel calling via `parallel_tool_calls: false` when operations demand strict sequential dependency ordering?
5. How to demux and reconstruct concurrent streaming JSON tool argument chunks using the stream's `index` field in real-time?

<details>
<summary>中文题目</summary>

1. 详细剖析大模型在单次生成中输出多个 `tool_calls` 数组项时，如何确保各个调用之间的参数完全解耦？
2. 宿主异步执行引擎（如 Python `asyncio.gather(*[execute_tool(call) for call in tool_calls])`）的并发调度与异常捕获策略？
3. 当并行调用的某一个工具超时或抛出 500 异常时，如何将其余成功结果与单项错误信息完整拼装返回而不阻塞流程？
4. 如何通过在 API 请求中设置 `parallel_tool_calls: false` 显式禁用并行调用以强制模型执行严格的串行依赖排期？
5. 在流式输出 (Streaming Tool Calls) 中，如何基于 `index` 字段实时还原多个并发工具调用的分块 JSON 参数？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/parallel-function-calling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=parallel-function-calling)

---

## 42. Parent-Child Chunking · Parent-Child 父子切块策略

> Parent-Child Chunking (Hierarchical Document Chunking) is a two-tier (or multi-tier) tree-structured RAG chunking paradigm;

**Interview questions**

1. Compare Parent-Child Chunking vs Sentence-Window Chunking in context structure and multi-level inheritance?
2. When multiple child chunks hit the same parent document, how to deduplicate and score-weight the returned parent?
3. How to leverage native Markdown heading trees (H1->H2->H3) to automatically build multi-tier Parent-Child tree node hierarchies?
4. Explain the architecture of storing parent documents in a DocStore (Mongo/Redis) synchronized with child vector indexes?
5. How to ensure child chunk boundaries do not cut through table rows or fenced code blocks within parent chunks?

<details>
<summary>中文题目</summary>

1. 详细比较 Parent-Child Chunking 与 Sentence-Window Chunking 在上下文组织形态与多级继承上的差异？
2. 当多个子切块 (Child Chunks) 命中同一父文档时，检索召回层如何执行父文档去重 (Deduplication) 与权重加权？
3. 如何利用文档原生的 Markdown 目录树 (H1->H2->H3) 自动构建多层级 Parent-Child 树状节点索引？
4. 父文档作为独立 DocStore 存储（如 MongoDB / Postgres / Key-Value Store）与向量库（存储子向量）的读写同步架构？
5. 如果父文档包含代码块或表格，子切块分割时如何保证不截断表格内部行？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/parent-child-chunking?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=parent-child-chunking)

---

## 43. Plan-and-Execute Pattern · Plan-and-Execute 任务规划解构

> Plan-and-Execute is a decoupled two-stage agentic architecture that separates macro task decomposition from micro-execution;

**Interview questions**

1. Detail the state and data flow between Planner (plan synthesis), Executor (sub-task runs), and Replanner (dynamic updates)?
2. How to model sub-task input-output dependencies using DAG topological sorting to enable concurrent execution of independent branches?
3. What are the threshold conditions for the Replanner to trigger full plan invalidation and re-synthesis upon unexpected step outcomes?
4. Why does Plan-and-Execute allow cost optimization by pairing a frontier Planner (GPT-4o) with lightweight Executors (GPT-4o-mini)?
5. Compare Plan-and-Execute vs ReAct across upfront planning latency, dynamic fault resilience, and total token overhead?

<details>
<summary>中文题目</summary>

1. 详细剖析 Planner（计划生成）、Executor（子任务执行）与 Replanner（动态重规划）三大组件的状态数据流？
2. 如何使用拓扑排序 (Topological Sorting) 与 DAG (有向无环图) 表达子任务之间的输入输出依赖关系并实现无依赖任务并发执行？
3. 当某个子任务执行结果彻底推翻原定后续方案时，Replanner 触发全局计划推倒重建 (Full Re-planning) 的阈值判定？
4. 为什么在 Plan-and-Execute 架构中，Planner 可以使用更聪明的大模型（如 GPT-4o），而 Executor 可以降级使用小模型（如 GPT-4o-mini）以大幅降本？
5. 对比 Plan-and-Execute 与 ReAct 在步骤规划延迟 (Planning overhead)、对意外错误的自适应鲁棒性与 Token 消耗上的差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/plan-and-execute-pattern?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=plan-and-execute-pattern)

---

## 44. Pointwise Evaluation Paradigm · Pointwise 单输出打分范式

> Pointwise Evaluation is the foundational LLM-as-a-Judge assessment paradigm that evaluates a single model response in isolation;

**Interview questions**

1. Design a standardized 1-5 scoring rubric detailing criteria and deduction rules for each numerical grade?
2. Why does forcing the judge model to 'write CoT reasoning first before emitting numerical scores' dramatically improve scoring consistency?
3. Analyze score clustering around central values (e.g., 8/10) in Pointwise judging and methods for score distribution calibration?
4. Compare Reference-based vs Reference-free judging in factual accuracy evaluation and benchmark preparation cost?
5. How does an ensemble panel of diverse judge models (GPT-4o, Claude 3.5, Gemini 1.5) taking median scores filter judge noise?

<details>
<summary>中文题目</summary>

1. 详细设计一份标准化的 1-5 分 Rubric 评分量规（定义每个分值段的具体合格条件与扣分红线）？
2. 为什么在打分 Prompt 中强制要求裁判模型“先写评判理由 (CoT Explanation)，最后再输出数字分数”能大幅提高打分一致性？
3. 分析 Pointwise 评估中常见的“分数膨胀与中间聚集 (Score Clustering / Central Tendency Bias)”现象及其校准方案？
4. Reference-based (有参考金标准答案) vs Reference-free (无参考答案纯裁判) 在事实准确性评估中的表现与成本差异？
5. 使用多个独立裁判模型（如 GPT-4o + Claude 3.5 + Gemini 1.5）组成裁判陪审团 (Judge Ensemble) 取中位数的降噪策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/pointwise-evaluation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pointwise-evaluation)

---

## 45. Product Quantization (PQ) · PQ 乘积量化与码本聚类

> Product Quantization (PQ) is a lossy vector compression algorithm based on orthogonal subspace decomposition; it divides a $D$-dimensional space into $M$ orthogonal lower-dimensional subspaces ($d^* = D/M$), runs K-Means clustering independently in each subspa…

**Interview questions**

1. Derive how PQ parameters ($M$ subspaces, $K$ centroid counts) dictate compression ratio and reconstruction error?
2. Why does applying OPQ (Optimized Product Quantization) orthogonal rotation before PQ significantly reduce quantization error?
3. For a 1024-dim FP32 vector with $M=32, K=256$, what is the compressed size in bytes and theoretical compression ratio?
4. Compare Symmetric Distance Computation (SDC) vs Asymmetric Distance Computation (ADC) in PQ precision and compute cost?
5. How does Residual Vector Quantization (RVQ) perform multi-stage cascaded quantization to improve recall over standard PQ?

<details>
<summary>中文题目</summary>

1. 详细推导 PQ 乘积量化的参数选择（$M$ 子空间数、$K$ 码本聚类中心数）对压缩比与重建误差的影响？
2. 为什么在 PQ 训练前对向量进行 OPQ (Optimized Product Quantization) 正交旋转可以大幅减少量化误差？
3. 1024 维 FP32 向量在 $M=32, K=256$ 设置下，单条向量被压缩为多少字节？理论内存压缩比是多少？
4. 比较对称距离计算 (SDC) 与非对称距离计算 (ADC) 在 PQ 检索中的距离精度与计算开销差异？
5. 残差量化 (Residual Vector Quantization, RVQ) 是如何在 PQ 基础上进行多级级联量化以提升召回率的？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/product-quantization-pq?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=product-quantization-pq)

---

## 46. Prompt Injection Defense · 提示词注入攻击与隔离防御

> Prompt Injection is a severe security vulnerability where crafted inputs hijack LLM control flow or override safety alignments, encompassing Direct Prompt Injection (Jailbreaking via user prompts) and Indirect Prompt Injection (embedding malicious instructions…

**Interview questions**

1. Contrast the attack vectors and blast radiuses between Direct Prompt Injection and Indirect Prompt Injection?
2. Explain the security design of the Dual-LLM architecture (Quarantine LLM parses untrusted text, Privileged LLM executes actions)?
3. Explain how adversarial suffix attacks (GCG gradient attacks) break safety alignment and how input perplexity filtering detects them?
4. How to use dynamic cryptographic Nonces to tag input boundaries and prevent attackers from forging XML closing tags?
5. How does Human-in-the-Loop (HITL) mandatory approval intercept high-risk tool operations triggered by malicious injections?

<details>
<summary>中文题目</summary>

1. 详细对比直接提示注入 (Direct Injection) 与间接提示注入 (Indirect Injection) 的攻击路径与危害面差异？
2. Dual-LLM 架构（Quarantine LLM 处理外部不可信数据并提取事实，Privileged LLM 仅接收纯结构化数据执行决策）的安全设计？
3. 对抗性前缀攻击 (Adversarial Suffix / GCG 梯度搜索攻击: 如追加乱码字符破坏安全对齐) 的成因与输入困惑度检测防御？
4. 如何使用随机 Nonce / 签名令牌 (Cryptographic Nonce Token) 封装输入边界以防攻击者伪造 XML 闭合标签？
5. 在 Agent 工具调用阶段，如何通过要求用户在终端显式审批确认 (Human-in-the-Loop Approval) 阻断高危写操作？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/prompt-injection-defense?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=prompt-injection-defense)

---

## 47. Query Rewriting & Decomposition · 查询重写与多跳子查询解构

> Query Rewriting & Sub-Query Decomposition is a core Pre-Retrieval RAG optimization technique that transforms raw user prompts into structured retrieval intents;

**Interview questions**

1. Describe the prompt pattern for performing conversational coreference resolution and synthesizing standalone queries from chat history?
2. Analyze latency vs accuracy trade-offs between parallel sub-query dispatch and sequential DAG step-by-step query execution?
3. How does Multi-Query Retrieval generate 3-5 diverse synonym query variations to maximize overall retrieval recall?
4. How does Step-Back Prompting generate high-level abstracted concept queries to augment RAG background retrieval?
5. How to constrain query rewriting LLM prompts to prevent hallucinated over-generalization drifting from the user's core intent?

<details>
<summary>中文题目</summary>

1. 详细描述多轮对话中基于聊天历史 (Chat History) 进行指代消解与完整独立问题 (Standalone Query) 生成的 Prompt 范式？
2. 子查询解构 (Sub-Query Decomposition) 中的并行分支 (Parallel Dispatch) 与 DAG 依赖串行执行（Step-by-step Execution）在延迟与准确率上的权衡？
3. Multi-Query Retrieval (多角度 Query 生成) 如何通过同时生成 3-5 个不同视角的同义 Prompt 提升检索召回率？
4. Step-Back Prompting (后退一步抽象查询) 如何通过生成高阶通用概念查询（如从具体物理题抽象为基础物理定律）辅助 RAG 检索？
5. 如何防止 LLM 在查询重写阶段发生幻觉过度泛化，导致偏离用户最初真实搜索意图？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/query-rewriting-decomposition?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=query-rewriting-decomposition)

---

## 48. RAG Triad Evaluation (Ragas) · RAG 三元组评估 (Ragas/TruLens)

> RAG Triad Evaluation (established by TruLens and Ragas frameworks) is the industry-standard modular evaluation framework that decomposes RAG into Query $Q$, Retrieved Context $C$, and Generated Answer $A$, evaluating 3 orthogonal unsupervised quality metrics:…

**Interview questions**

1. Derive the mathematical formula and prompt engineering workflow for Context Relevance in Ragas?
2. How does Faithfulness compute $S = \frac{|V|}{|T|}$ via claim extraction and NLI entailment against context passages?
3. How does Answer Relevance score ungrounded outputs by generating synthetic reverse questions $q_i$ and embedding cosine similarity?
4. How to optimize LLM API cost and evaluation latency when running Ragas over massive 10K+ token retrieved contexts?
5. How to correlate automated RAG Triad scores with real user telemetry (thumbs up/down) in production dashboards?

<details>
<summary>中文题目</summary>

1. 详细推导 Ragas 中 Context Relevance（从上下文提取与 Query 相关句子的比例）的计算公式与 Prompt 实现？
2. Faithfulness（忠实度指标）如何通过“声明拆解 (Claim Extraction) + 事实蕴含 (NLI Entailment)”计算分数 $S = \frac{|V|}{|T|}$？
3. Answer Relevance 如何通过反向推导（由模型回答 $A$ 反向生成若干个潜在问题 $q_i$ 并计算与原问题 $Q$ 的余弦相似度）进行无参考打分？
4. 针对超长召回上下文 (Context > 10K tokens)，如何降低 Ragas 计算三元组指标所需的 LLM API 评测成本与延迟？
5. 在生产监控看板中，如何将 RAG 三元组指标与用户真实点击/点赞点踩 (Feedback Signals) 进行相关性校验？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/rag-triad-evaluation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rag-triad-evaluation)

---

## 49. RankGPT LLM Listwise Reranker · RankGPT 大模型列表重排

> RankGPT is a generation-based re-ranking paradigm that leverages frontier LLMs (GPT-4o, Claude 3.5, Qwen-72B) and specialized Listwise prompts to globally compare a candidate document set and directly output sorted rank permutations (e.g., [3] > [1] > [5]);

**Interview questions**

1. Detail the sliding window bubble-sort workflow (window_size=20, step=10) and complexity in RankGPT?
2. Compare Pointwise, Pairwise, and Listwise paradigms in LLM re-ranking stability and token consumption?
3. What are the mitigation strategies for LLM Position Bias (favoring items at prompt start or end) in Listwise re-ranking?
4. Weigh distilled local open-source models (RankLLaMA) vs commercial frontier API calls in Listwise re-ranking?
5. How to implement robust regex JSON parsing and fallback retries when LLM outputs malformed permutation formats?

<details>
<summary>中文题目</summary>

1. 详细剖析 RankGPT 的滑动窗口 (Sliding Window: 如 window_size=20, step=10) 冒泡排序算法流程与复杂度？
2. Pointwise (逐个打分)、Pairwise (两两对比) 与 Listwise (整列表排序) 在 LLM 重排中的稳定性与 Token 消耗对比？
3. 大模型在执行 Listwise 排序时表现出的位置偏置 (Position Bias: 倾向给开头或结尾文档打高分) 缓解策略？
4. 采用蒸馏小模型（如 RankLLaMA / RankVicuna）本地部署 Listwise 重排相比直接调用云端 GPT-4 API 的性价比考量？
5. 如何解析并容错 LLM 生成非标准排序格式（如缺少编号、生成幻觉编号）时的 JSON 正则解析重试机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/llm-rankgpt-reranker?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=llm-rankgpt-reranker)

---

## 50. ReAct Thought-Action-Observation · ReAct 思考-动作-观察模式

> ReAct (Synergizing Reasoning and Acting in Language Models) is a foundational agentic architecture interleaving explicit symbolic reasoning traces with dynamic environment action execution;

**Interview questions**

1. Detail the structural format of ReAct prompts (Few-shot exemplars, formatting instructions, and stop sequence triggers)?
2. Why does generating explicit Thoughts within ReAct drastically reduce parameter errors and provide traceable interpretability?
3. How does a ReAct agent adjust its search trajectory via Thought when an Observation yields a 404 or irrelevant output?
4. How to mitigate ReAct step explosion and linear context bloat via `max_iterations` ceilings and conversational summarization?
5. Compare native Function Calling protocols vs early text-regex ReAct prompt parsing in execution reliability?

<details>
<summary>中文题目</summary>

1. 详细剖析 ReAct 提示词结构（Few-shot ReAct Exemplars、Format Instructions 与 Stop Sequences 截断标记）？
2. 为什么在 ReAct 循环中显式生成 Thought 能够显著降低工具调用参数错误率并提高长链条任务的推理轨迹可解释性？
3. 当 Observation 返回了包含 404 报错或完全无关的搜索结果时，ReAct Agent 是如何通过 Thought 进行策略调整与改道重试的？
4. ReAct 循环的步数爆炸 (Step Explosion) 与 Token 线性膨胀问题：如何设置 `max_iterations` 截断与历史状态摘要？
5. 原生 Function Calling (如 OpenAI tools) 相比早期基于文本正则匹配的 ReAct（解析 `Action: ...`）在解析鲁棒性上的演进？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/react-agent-pattern?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=react-agent-pattern)

---

## 51. Recursive Character Splitter · 递归字符切块 (Recursive Splitter)

> Recursive Character Text Splitter is a hierarchical text chunking algorithm that recursively evaluates a prioritized delimiter list (e.g., ["\n\n", "\n", ".

**Interview questions**

1. Analyze the trade-offs of `chunk_size` and `chunk_overlap` on embedding semantic density vs duplicate recall overhead?
2. How to customize delimiter lists for bilingual Chinese-English text to correctly prioritize Chinese punctuation and Markdown headings?
3. Why must chunk length be evaluated using the target embedding tokenizer rather than Python's raw `len()` character count?
4. How to tune `chunk_overlap` to prevent context bloat and LLM attention dilution from redundant duplicated segments?
5. How to apply AST-based code splitters for Python/JS to recursively segment code at class and function boundaries?

<details>
<summary>中文题目</summary>

1. 分析参数 `chunk_size` (如 256/512/1024) 与 `chunk_overlap` (如 10%~20%) 对向量嵌入语义密度与重复召回率的影响？
2. 针对中英文混排文档，如何自定义分隔符列表以正确优先识别中文句末标点（如 `。！？`）与 Markdown 标题标记（如 `### `）？
3. 为什么在计算切块长度时必须使用对应 Embedding 模型的 Tokenizer (如 cl100k_base / bge tokenizer) 而不是纯 Python `len()` 字符数？
4. 重叠部分 (`chunk_overlap`) 过大时在 RAG 上下文中引起的 Context 膨胀与 LLM 注意力稀释问题如何调优？
5. 代码文件（Python/JS）切分时，如何改用基于 AST (抽象语法树，如 LangChain CodeSplitter) 的类与函数级递归切分？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/recursive-character-splitter?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=recursive-character-splitter)

---

## 52. Reflexion Self-Correction · Reflexion 自我反思与反思记忆

> Reflexion is an advanced agentic architecture that reinforces task performance through verbal self-reflection rather than traditional reinforcement learning weight updates;

**Interview questions**

1. Detail the tripartite architecture of Reflexion: Actor, Evaluator, and Self-Reflection modules and their execution flow?
2. Explain the FIFO sliding window eviction and vector-based semantic retrieval across Reflexion memory buffers?
3. Why does unit-test grounded Reflexion boost HumanEval/LeetCode pass@1 from ~60% to over 90% across 3 trials?
4. Design prompt constraints to prevent models from authoring vague, ungrounded reflections that fail to guide self-correction?
5. Compare Reflexion vs traditional Reinforcement Learning (PPO/DPO) in sample efficiency, inference cost, and cold-start feasibility?

<details>
<summary>中文题目</summary>

1. 详细梳理 Reflexion 系统的三大核心模块（Actor 动作执行者、Evaluator 奖励评估器、Self-Reflection 反思生成器）的协作流？
2. Reflexion 反思记忆缓冲池 (Memory Buffer) 的 FIFO 滑动窗口淘汰与向量检索检索机制？
3. 为什么在代码生成基准 (HumanEval / LeetCode) 上，结合单元测试反馈的 Reflexion 能够将 pass@1 从 60% 提升至 90%+？
4. 防止模型生成“空洞敷衍的无效反思 (Vague / Hallucinated Reflections)”的 Prompt 架构设计与强制证据引用约束？
5. Reflexion 与传统强化学习 (RL / PPO / DPO) 在样本效率、推理成本与冷启动难度上的综合对比？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/reflexion-self-correction?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=reflexion-self-correction)

---

## 53. RRF Reciprocal Rank Fusion · RRF 倒数秩融合算法

> Reciprocal Rank Fusion (RRF) is an unsupervised, calibration-free multi-retriever rank aggregation algorithm; when merging heterogeneous search branches (e.g., BM25 scores $[0, 30+]$ and Dense cosine similarities $[0, 1]$), RRF discards incommensurable raw sco…

**Interview questions**

1. Derive the RRF scoring formula and explain how constant $k=60$ smoothens aggressive score gaps between top-ranked ranks?
2. Compare RRF scores between a document ranking #1 on a single branch vs ranking #5 across both branches?
3. How does Weighted RRF apply branch weights $w_m$ when domain requirements favor lexical or dense semantics?
4. Weigh RRF against supervised Learning to Rank (LTR / RankNet) regarding cold-start feasibility and annotation costs?
5. Analyze native RRF implementations and execution parameters within Elasticsearch and Qdrant Hybrid Search APIs?

<details>
<summary>中文题目</summary>

1. 推导 RRF 打分公式，并详细解释常数 $k=60$ 是如何平滑排名前列项的激进分数差异的？
2. 为什么在单一检索分支排名第一的文档（Rank 1），与在两路检索中均排名靠前（如两个 Rank 5）的文档，在 RRF 中的得分比较？
3. 加权 RRF (Weighted RRF: $\sum_m \frac{w_m}{k + r_m(d)}$) 在特定领域对 BM25 或 Dense 给予偏好倾斜时的应用？
4. 对比 RRF 与有监督排序学习 (Learning to Rank, LTR / RankNet) 在冷启动与标注成本上的优缺点？
5. Elasticsearch 与 Qdrant 官方 Hybrid Search API 底层对 RRF 的原生实现与执行参数调优？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/rrf-rank-fusion?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rrf-rank-fusion)

---

## 54. Scalar Quantization (SQ8/SQ4) · SQ8/SQ4 标量量化

> Scalar Quantization (SQ) is a vector compression technique that independently maps each component of a high-dimensional vector to discrete fixed-point integers;

**Interview questions**

1. Derive the SQ8 linear quantization and de-quantization formulas and source of precision loss?
2. What is the Recall difference between Global SQ and Per-Dimension SQ under heavy-tailed distribution?
3. How do modern CPU AVX-512 VNNI instructions accelerate int8 dot product operations?
4. Compare SQ8 vs PQ in terms of memory compression ratio, index construction time, and reconstruction error?
5. Why does SQ8 recall rate improve after applying outlier clipping / percentile thresholding?

<details>
<summary>中文题目</summary>

1. 推导 SQ8 的线性量化与反量化 (De-quantization) 公式及精度损失来源？
2. 全局均匀量化 (Global SQ) 与分维度量化 (Per-Dimension SQ) 在长尾分布下的 Recall 差异？
3. 现代 CPU 的 AVX-512 VNNI 指令如何加速 int8 向量的点积运算？
4. 对比 SQ8 与 PQ (乘积量化) 在内存压缩比、索引构建时间与重构误差上的优劣？
5. 为什么在向量经过异常值截断 (Outlier Clipping / Min-Max Percentile) 后，SQ8 的召回率会有所上升？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/scalar-quantization-sq8?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=scalar-quantization-sq8)

---

## 55. Self-RAG Reflection · Self-RAG 自省式反思检索

> Self-Reflective Retrieval-Augmented Generation (Self-RAG) is an adaptive framework that trains an LLM to autonomously decide when to retrieve, critique retrieval quality, and verify generation faithfulness;

**Interview questions**

1. Detail the generation conditions and evaluation dimensions of the 4 Self-RAG reflection tokens?
2. Explain the training pipeline of Critic and Generator models including synthetic reflection token annotation?
3. How does segment-level beam search leverage reflection token logit probabilities during inference generation?
4. Compare Self-RAG (internal model reflection) vs Agentic RAG (external state-machine loops) in latency and flexibility?
5. How does `[IsSUP]` enforce adherence to authoritative retrieved passages when they contradict internal parametric knowledge?

<details>
<summary>中文题目</summary>

1. 详细剖析 Self-RAG 中 4 类反思 Token (`[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]`) 的生成时机与各自的评判维度？
2. Self-RAG 训练管道中批评模型 (Critic Model) 与生成器模型 (Generator Model) 的数据标注生成与微调流程？
3. 在推理生成阶段，如何结合反思 Token 的 Logits 概率加权执行段落级束搜索 (Segment-level Beam Search)？
4. 对比 Self-RAG (模型内生反思) 与 LangGraph/Agentic RAG (外置循环状态机反思) 在延时开销与灵活性上的差异？
5. 当检索返回的文档与模型内生预训练记忆产生事实冲突时，`[IsSUP]` 如何强制引导模型基于权威文档生成？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/self-rag-reflection?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=self-rag-reflection)

---

## 56. Semantic Chunking · Semantic 语义断崖自适应切分

> Semantic Chunking is an adaptive, content-aware document segmentation algorithm driven strictly by semantic coherence rather than arbitrary character or token limits;

**Interview questions**

1. Describe the algorithmic workflow of mapping sentence embedding distance curves and identifying cut points via percentile thresholds?
2. How to apply a sliding buffer of 2-3 sentences to smooth distance curves against single-sentence noise spikes?
3. Analyze the extra embedding API calls and latency costs generated by Semantic Chunking during document ingestion?
4. How to enforce a `max_chunk_size` fallback ceiling to prevent extremely long single-topic sections from overflowing LLM contexts?
5. Weigh the empirical Recall improvements of Semantic Chunking vs ingestion cost overhead compared to Recursive Splitters?

<details>
<summary>中文题目</summary>

1. 详细描述计算相邻句子语义距离曲线并基于百分位数阈值 (Percentile Threshold / Standard Deviation) 确定断点的算法流程？
2. 为了防止单个长句子本身波动引起误切，如何采用滑动缓冲区 (Sliding Buffer of 2-3 sentences) 进行平滑距离计算？
3. 分析 Semantic Chunking 在文档入库处理时所产生的额外 Embedding API 调用次数与耗时成本？
4. 当遇到极端超长单主题段落时，如何结合 `max_chunk_size` 兜底机制避免单个切块超出 LLM 上下文限制？
5. 与基于规则的递归切分 (Recursive Splitter) 相比，在生产基准测试中 Semantic Chunking 带来的检索 Recall 提升幅度与性价比权衡？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/semantic-chunking?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=semantic-chunking)

---

## 57. Sentence-Window Chunking · Sentence-Window 句窗口切分

> Sentence-Window Retrieval (Small-to-Big Sentence Window) is an advanced RAG chunking architecture that cleanly decouples the 'Retrieval Unit' from the 'Generation Context Unit';

**Interview questions**

1. How does Sentence-Window resolve the fundamental contradiction between embedding information dilution and generation context deficiency?
2. When retrieval hits multiple adjacent sentences in the same paragraph, how to execute window merging to prevent duplicate context injection?
3. Analyze the trade-offs of `window_size` parameter (preceding/succeeding $k$ sentences) on LLM token budget vs answer completeness?
4. Quantify the vector count inflation and storage overhead of Sentence-Window compared to traditional overlapping paragraph chunks?
5. How to decouple architecture by offloading expanded window texts to Redis/SQL while vector DB only indexes single sentences?

<details>
<summary>中文题目</summary>

1. 详细剖析 Sentence-Window 如何从数学上解决向量嵌入“信息稀释 (Information Dilution)”与“上下文缺失”的根本矛盾？
2. 如果检索命中了同一段落中相邻的 3 个句子，如何执行窗口合并 (Window Merging / Deduplication) 以防止上下文重复注入？
3. 窗口大小参数 `window_size` (前 $k$ 句与后 $k$ 句) 对 LLM 上下文窗口容量消耗与答案完整度的权衡？
4. Sentence-Window 相比传统带有 overlap 的段落分块在向量索引数量（条数膨胀）与存储开销上的成本量化？
5. 在生产部署时，如何通过将窗口文本存储在 Redis / 关系数据库中以减轻向量数据库只负责存单句的架构负担？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/sentence-window-chunking?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=sentence-window-chunking)

---

## 58. Supervisor-Worker Architecture · Supervisor-Worker 主管智能体分发

> Supervisor-Worker (Hierarchical Agent Architecture) is a centralized dispatch pattern where a central orchestrator (Supervisor / Router Agent) coordinates a fleet of specialized vertical worker agents (e.g., Research Agent, Coder Agent, Plotting Agent) via hie…

**Interview questions**

1. Detail the architecture in LangGraph using Function Calling to construct a Supervisor router that emits `next_agent` state keys?
2. How does the Supervisor apply context scoping to transmit only necessary localized slices to Workers, preventing context pollution?
3. Design structured worker response formats (structured summaries vs raw logs) to bound the Supervisor's decision token overhead?
4. How to scale hierarchical multi-tier architectures (Director -> Lead -> Worker) for massive enterprise-grade engineering?
5. How does the Supervisor implement failover rescheduling or alternative worker dispatch when a worker encounters unhandled errors?

<details>
<summary>中文题目</summary>

1. 详细剖析 LangGraph 中基于 Function Calling 构建 Supervisor 路由分发器（返回 `next_agent` 字段）的架构实现？
2. 上下文隔离 (Context Scoping): Supervisor 在向特定 Worker 分派任务时，如何仅传递必要局部切片以防止 Worker 上下文污染？
3. Worker 返回给 Supervisor 的数据格式规范（结构化摘要 vs 完整执行日志）对 Supervisor 决策 Token 消耗的控制？
4. 多级层级结构 (Hierarchical Supervisor of Supervisors: 总监 -> 组长 -> 员工) 在超大规模复杂工程中的扩展模式？
5. 当某个 Worker 节点抛出不可恢复异常时，Supervisor 如何执行重新调度 (Rescheduling) 或切换备用 Worker？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/supervisor-worker-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=supervisor-worker-architecture)

---

## 59. SWE-bench Agent Evaluation · SWE-bench 代码与 Agent 评估

> SWE-bench (and SWE-bench Verified) is the gold-standard benchmark evaluating autonomous AI Coding Agents (Claude 3.5 Sonnet, Devin, Cursor Agent) on resolving real-world GitHub issues across complex multi-file codebases;

**Interview questions**

1. Detail the SWE-bench execution pipeline: Issue prompt -> Docker sandbox exploration -> Patch extraction -> Running unit test suites?
2. Why does SWE-bench Verified provide significantly higher evaluation fidelity than raw SWE-bench by filtering ambiguous issues?
3. How do agentic search strategies (ripgrep, ctags, AST indexing) dictate issue localization and ultimate solve rates?
4. Explain the dual verification assertions in SWE-bench: FAIL_TO_PASS (fixing the bug) and PASS_TO_PASS (preventing regression)?
5. How does transitioning to a Multi-Agent architecture (Localization -> Fix -> Test Generation) substantially boost SWE-bench resolution rates?

<details>
<summary>中文题目</summary>

1. 详细梳理 SWE-bench 评测流水线：Issue 描述输入 -> Agent Docker 沙箱探索 -> Git Patch 提取 -> 运行 PASS_TO_PASS 与 FAIL_TO_PASS 测试？
2. SWE-bench Lite 与 SWE-bench Verified（经过人类专家审查剔除有歧义或测试不完整的实例）在评测置信度上的提升？
3. Agent 在面对数十万行大型仓库时的代码检索与定位策略（Agentic Search / Ripgrep / AST 语义索引）对最终得分的决定性影响？
4. SWE-bench 中 FAIL_TO_PASS（原本失败的测试必须通过）与 PASS_TO_PASS（原本通过的测试不得被破坏）的双向断言验证机制？
5. 从单 Agent 盲目编辑到引入 Multi-Agent（定位 Agent + 修复 Agent + 单元测试验证 Agent）在提升 SWE-bench 解决率上的架构演进？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/swe-bench-agent-eval?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=swe-bench-agent-eval)

---

## 60. Tool Error Self-Healing Loop · 工具报错自愈重试循环

> Tool Error Self-Healing & Recovery Loop is the core fault-tolerance mechanism that keeps Autonomous Agents and tool pipelines executing resiliently through runtime failures;

**Interview questions**

1. Design error feedback formatting: how to prune massive Python tracebacks down to concise critical lines to save context tokens?
2. How does Exponential Backoff with Jitter resolve transient network drops and third-party API 429 rate limit exceptions?
3. How to detect repeating error patterns via state hashing when an LLM is trapped in an infinite error loop and trigger a breaker?
4. Design routing logic distinguishing Recoverable errors (syntax/schema fixes) from Fatal errors (403 forbidden, insufficient funds)?
5. How to instrument OpenTelemetry tracing and LangSmith monitoring to track tool self-healing retry counts and recovery success rates?

<details>
<summary>中文题目</summary>

1. 详细描述错误回填消息格式设计：如何将长达数千字的 Python Traceback 进行智能提纯（保留核心异常行与变量名）以节省 Token？
2. 指数退避算法 (Exponential Backoff with Jitter) 在处理临时网络抖动与第三方 API 限流 (429 Rate Limit) 中的应用？
3. 当模型陷入反复尝试相同错误参数的“死循环自愈陷阱 (Self-Healing Loop Trap)”时，如何进行状态指纹哈希对比并触发熔断降级？
4. 区分可恢复错误 (Recoverable: 如拼写错误、格式缺失) 与不可恢复致命错误 (Fatal: 如权限缺失 403、余额不足) 的分流路由策略？
5. 在生产监控体系中，如何对工具调用自愈重试次数与成功率建立分布式链路追踪 (OpenTelemetry Tracing / LangSmith)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/tool-error-healing-loop?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=tool-error-healing-loop)

---

## 61. Toolformer Self-Supervised Tool Use · Toolformer 自监督工具学习

> Toolformer (Meta's self-supervised tool-learning framework) is a training methodology that enables language models to autonomously discover where, when, and how to invoke external APIs without manual annotations;

**Interview questions**

1. Derive the mathematical interpretation of each term in Toolformer's loss filtering formula $L_{\text{call}} < \min(L_{\text{no\_call}}, L_{\text{same}}) - \tau$?
2. Why does evaluating $L_{\text{same}}$ with blank/neutral placeholders isolate true information gain contributed by tool outputs?
3. How does Toolformer execute API interceptions at inference time using specialized separator tokens (e.g., `->`)?
4. What are the limitations of Toolformer when handling multi-step nested API calls compared to modern ReAct agent paradigms?
5. How did subsequent models (Gorilla LLM, ToolLlama) extend Toolformer to thousands of complex real-world RESTful APIs?

<details>
<summary>中文题目</summary>

1. 推导 Toolformer 的 Loss 过滤公式 $L_{\text{call}} < \min(L_{\text{no\_call}}, L_{\text{same}}) - \tau$ 中各项损失的数学含义？
2. 为什么在计算 $L_{\text{same}}$ 时将 API 返回结果用空文本或通用占位符替代，以验证工具结果的真实信息增益？
3. Toolformer 在推理生成阶段如何通过自定义结束符（如 `->` 触发外部执行器拦截）实现模型与执行环境的解耦通信？
4. 当 API 调用的参数需要复杂多步依赖或嵌套调用时，Toolformer 架构的局限性与现代 Agent 框架的演进？
5. Gorilla LLM 与 ToolLlama 等后续工作如何将 Toolformer 扩展至对真实世界数千个复杂 RESTful API 的零样本泛化调用？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/toolformer-self-supervised?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=toolformer-self-supervised)

---

## 62. Tree of Thoughts (ToT) · Tree of Thoughts 思想树搜索

> Tree of Thoughts (ToT) is a non-linear search-based reasoning framework that generalizes Chain-of-Thought prompting by integrating classical tree search algorithms (BFS, DFS, A*) with LLM state generation and heuristic valuation;

**Interview questions**

1. Detail the 4 core ToT operations: Thought Generation, State Evaluation (Value vs Vote prompting), Search Scheduling, and Pruning?
2. Compare BFS vs DFS in ToT on Game of 24 puzzle benchmarks regarding solution quality and token budget?
3. Design a self-consistent value estimation prompt instructing LLMs to output 'sure / likely / impossible' states with numeric confidence?
4. How does Monte Carlo Tree Search (MCTS) extend ToT for autonomous test-time compute scaling in LLM reasoning?
5. What are practical engineering pruning heuristics (dynamic beam width limits) to control exponential token costs in ToT?

<details>
<summary>中文题目</summary>

1. 详细剖析 ToT 的 4 大核心操作算子：思想生成、状态评估 (Value Prompting vs Vote Prompting)、搜索调度与剪枝？
2. 对比 BFS (广度优先，适合浅层多分支探索) 与 DFS (深度优先，适合快速探底与回溯) 在求解 24 点问题时的表现与 Token 开销？
3. 如何设计自洽的启发式价值评估 Prompt（如要求模型给出 'sure / likely / impossible' 评价及置信度分数）？
4. 蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS) 结合 ToT (如 AlphaGo 范式) 在现代大模型自主推理中的演进与应用？
5. ToT 生成天量候选分支导致的 API 费用与高延迟问题，在工业落地中的剪枝策略 (Beam Width 动态约束)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/tree-of-thoughts-agent?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=tree-of-thoughts-agent)

---

## 63. Vector DB Filtering & Selection · 向量数据库标量过滤与选型

> Scalar Metadata Filtering is a co-processing query mechanism in vector databases that evaluates structural metadata predicates (e.g., tenant_id = 101 AND timestamp > 2026-01-01) alongside high-dimensional vector similarity search;

**Interview questions**

1. Why does HNSW post-filtering suffer total recall collapse when scalar filter selectivity is extremely restrictive (< 1%)?
2. How does Qdrant's payload inverted indexing achieve single-stage iterative filtering during HNSW traversal?
3. How does Milvus's Bitset mechanism perform zero-copy masked filtering during batched vector distance computations on QueryNodes?
4. Weigh the trade-offs between dedicated vector databases (Milvus / Qdrant) vs extended systems (Pgvector / ES) in write throughput and ops cost?
5. In multi-tenancy architecture, compare partition-key physical isolation vs global indexing with scalar filtering?

<details>
<summary>中文题目</summary>

1. 为什么当标量过滤条件极度严苛（选择率 Selectivity < 1%）时，HNSW 后过滤 (Post-filtering) 会发生召回率崩溃？
2. Qdrant 如何通过为每个 Payload 标量字段建立倒排索引，在 HNSW 遍历过程中实现 Single-Stage 实时过滤？
3. 分析 Milvus 的 Bitset 机制在 QueryNode 节点执行批量向量距离计算时如何实现零内存拷贝掩码过滤？
4. 对比专有分布式向量数据库 (Milvus / Qdrant) 与传统扩展 (Pgvector / Elasticsearch) 在高并发写入与运维成本上的选型权衡？
5. 在多租户 (Multi-tenancy) 架构中，分区隔离 (Partition-key Isolation) 与全局索引加标量过滤的性能与扩展性差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/vector-db-metadata-filtering?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vector-db-metadata-filtering)

---

## 64. Vector Distance Metrics & L2 Normalization · 向量距离度量与 L2 归一化

> Vector distance metrics are mathematical functions measuring geometric similarity in high-dimensional embedding spaces, primarily Cosine Similarity $\cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$, Euclidean Distance (L2) $d(u, v) = \|u - v\|_2$, and Dot Product…

**Interview questions**

1. Derive the mathematical equivalence formula between L2 distance and Cosine Similarity under L2 normalization?
2. Why is Dot Product execution on GPU/SIMD much faster than raw unnormalized Cosine Similarity?
3. In unnormalized vectors, why does dot product retrieval suffer severe bias towards text length and vector magnitude?
4. How to choose between Cosine Similarity and Manhattan Distance (L1) for high-dimensional sparse vectors (e.g., BM25/SPLADE)?
5. When modifying Metric Type (IP vs COSINE vs L2) in vector DB collection indexes, is full index rebuild required?

<details>
<summary>中文题目</summary>

1. 推导经过 L2 归一化后，L2 距离与余弦相似度之间的数学等价公式？
2. 为什么在 GPU/SIMD 上执行点积 (Dot Product) 效率远高于未经归一化的余弦距离？
3. 非归一化向量下，为什么点积检索容易受到文本长度和向量模长的严重偏置影响？
4. 余弦相似度与曼哈顿距离 (L1) 在高维稀疏向量（如 BM25/SPLADE）中的选型差异？
5. 主流向量数据库在创建集合索引时，修改 Metric Type（IP vs COSINE vs L2）是否需要全量重建图索引？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/vector-distance-metrics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vector-distance-metrics)

---

## 65. Verbosity & Self-Enhancement Bias · Verbosity 与 Self-Enhancement 偏置

> Verbosity & Self-Enhancement Bias Mitigation addresses two pervasive cognitive distortions in LLM judges: 1) Verbosity Bias (the tendency for judge LLMs to favor lengthy, overly verbose, formatting-heavy responses over concise accurate answers by a 30%+ margin…

**Interview questions**

1. Derive the logistic regression calibration model with length covariates to mathematically normalize Verbosity Bias?
2. Design prompt strategies that penalize verbosity by explicitly prioritizing factual density over length?
3. Quantify the empirical severity of Self-Enhancement bias when models judge outputs from their own architectural lineage?
4. Explain the application of Cross-Judge Evaluation Matrices where competing models cross-evaluate peers to reach equilibrium?
5. How does style normalization and verbal quirk removal prevent judge models from recognizing their own stylistic fingerprints?

<details>
<summary>中文题目</summary>

1. 详细推导基于文本长度的胜率回归校准模型（Logistic Regression with Length Feature）消除长文偏置的数学流程？
2. 设计能够有效压制冗长偏置的 Prompt 策略（如显式声明“简练性与信息密度优先，任何多余修饰废话一律扣分”）？
3. Self-Enhancement 偏置在不同模型家族（GPT 系列、Claude 系列、Llama 系列）作为裁判时的量化偏向统计规律？
4. 交叉评审矩阵 (Cross-Judge Evaluation Matrix: $N$ 个模型互相裁判并求纳什均衡) 在多模型评测中的应用？
5. 盲测掩码 (De-identification & Style Masking: 统一 Markdown 结构与去除模型口癖) 如何防止裁判识别出自身风格？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/AI_Engineering/card/judge-verbosity-self-bias?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=judge-verbosity-self-bias)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
