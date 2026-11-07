# 🏗️ Industrial System Design · 工业级系统设计 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **43 cards · 215 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. 10ms Financial Risk Decision Pipeline · 10ms 金融级风控决策全链路

> The 10ms Financial-Grade Real-Time Risk Pipeline is a distributed, mission-critical decision architecture executing comprehensive fraud evaluation and emitting 3-way final decisions (Accept, Reject, Challenge) for financial transactions under a strict 10ms Lat…

**Interview questions**

1. Diagram the 10ms real-time risk decision pipeline and allocate millisecond latency budgets across all internal stages?
2. How to orchestrate the synergy between deterministic Rule Engines (hard vetoes) and ML scoring models (soft risk probabilities)?
3. Design a 3-tier graceful degradation policy when the real-time feature store spikes past 5ms timeout thresholds?
4. How to calibrate dynamic step-up authentication thresholds (OTP/biometrics) balancing fraud prevention against user friction?
5. How to maintain strong consistency for dynamic rule reloads and streaming user frequency counters in active-active multi-region deployments?

<details>
<summary>中文题目</summary>

1. 详细画出 10ms 实时风控决策链路图，并分解网关、规则引擎、特征存储、模型推理与决策引擎的毫秒级耗时预算？
2. 规则引擎（硬拦截）与机器学习模型（软评分）在线上如何协同编排（规则前置过滤 $\to$ 模型打分 $\to$ 规则后置约束）？
3. 当实时特征库 (Redis/Aerospike) 发生网络抖动导致查询耗时超过 5ms 熔断阈值时，系统的三级优雅降级策略？
4. 三路决策（放行、拦截、挑战）中，动态二次验证挑战 (Step-up Authentication: 如人脸识别/OTP) 的触发阈值设计与用户摩擦成本权衡？
5. 风控系统的双机房双活 (Active-Active Multi-Region) 部署中，如何保证规则引擎热加载与用户频次特征跨机房强一致性？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/risk-decision-10ms-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=risk-decision-10ms-pipeline)

---

## 2. 5-Step System Design Interview Method · 系统设计 5 步面试答题方法论

> The 5-Step ML & System Design Interview Framework is the industry-standard methodology for deconstructing ambiguous, open-ended architecture problems (e.g., 'Design YouTube Recommendation' or 'Design a Distributed LLM Code Assistant') into rigorous, highly str…

**Interview questions**

1. How to structure proactive clarifying questions covering Functional goals and Non-Functional constraints in the opening 5 minutes?
2. How to systematically translate high-level business goals (Retention, GMV) into concrete ML optimization objectives (Multi-Task CTR/CVR)?
3. How to clearly illustrate dual-track Offline Training and Online Serving dataflows on architectural diagrams?
4. How to systematically handle 10x traffic surge and disaster-recovery failure mode follow-ups from senior interviewers?
5. How to proactively discuss technical trade-offs and future evolutionary roadmaps to secure a 'Strong Hire' rating?

<details>
<summary>中文题目</summary>

1. 面试前 5 分钟如何主动向面试官提问以澄清功能需求 (Functional) 与非功能约束 (Non-Functional: 延迟 SLA、QPS、可用性 99.99%)？
2. 如何从业务目标（如“提升留存与 GMV”）无缝转换为机器学习优化目标（如“多任务预测 CTR 与 CVR 并做 eCPM 排序”）？
3. 在白板或架构图上如何规范清晰地组织“离线训练流 (Offline)”与“在线推理流 (Online)”的双轨数据流拓扑？
4. 面对面试官突然抛出的“如果系统 QPS 暴涨 10 倍”或“数据中心断电”的极限压力追问时的应对与扩容套路？
5. 在回答最后阶段如何主动提出系统当前的架构缺陷 (Trade-offs) 与未来演进方向（如从双塔升级为全注意力/GraphRAG）以拿到 Strong Hire？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/five-step-system-design-framework?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=five-step-system-design-framework)

---

## 3. Ad Targeting & pCTR Prediction · 广告定向体系与 pCTR/pCVR 预估

> The Ad Targeting & pCTR/pCVR Prediction System forms the core revenue backbone of Computational Advertising; it encompasses: 1) The Targeting Engine, which filters active ad campaigns within milliseconds based on advertiser-defined constraints (DMP audience ta…

**Interview questions**

1. Why does recommendation only require relative ordering (high AUC), whereas computational advertising demands absolute calibrated probabilities?
2. Compare Isotonic Regression vs Platt Scaling in non-parametric calibration and monotonicity guarantees for ad CTR predictions?
3. How does the Delayed Feedback Model (DFM) mathematically correct for conversions occurring hours/days after the initial click?
4. Derive the FTRL (Follow The Regularized Leader) streaming algorithm for real-time online ad weight updates with $L_1$ sparsity?
5. How to enforce strict boundaries between Request-Time features and Post-Impression features to eliminate data leakage in ad logs?

<details>
<summary>中文题目</summary>

1. 为什么在推荐系统中只需保证排序相对序（AUC 高即可），而在广告竞价系统中必须做严格的“概率绝对值校准 (Probability Calibration)”？
2. 详细对比保序回归 (Isotonic Regression) 与 Platt Scaling 在广告 CTR 概率校准中的单调性保障与非参数拟合特性？
3. 广告点击反馈延迟 (Delayed Feedback Problem: 用户点击后可能数天后才发生下单转化) 如何通过延迟转化概率模型 (DFM) 修正？
4. 高维稀疏特征（如 User ID $\times$ Ad ID）中的 FTRL (Follow The Regularized Leader) 在线流式更新优化算法与 $L_1$ 稀疏性推导？
5. 广告特征穿越防护：如何严格区分“请求时特征 (Request-Time)”与“展示后特征 (Post-Impression)”，杜绝不可用未来特征注入？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/ad-targeting-pctr-system?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ad-targeting-pctr-system)

---

## 4. Async Task Queue & Decoupled State · 异步任务队列与状态解耦架构

> The Asynchronous Task Queue & Decoupled State Architecture (Celery, Temporal, BullMQ + Redis/Kafka) is the indispensable distributed backbone for long-running multimodal generative workloads (Image Diffusion 2-5s, Video Generation 30-180s);

**Interview questions**

1. Diagram the end-to-end async sequence flow across Client, Gateway, Redis/Kafka Queue, GPU Workers, and S3 Storage?
2. Design a multi-tier Priority Queue granting paid VIP subscribers pre-emptive GPU worker scheduling?
3. How do worker heartbeat leases and Dead Letter Queues (DLQ) safely re-enqueue tasks orphaned by worker hardware failures?
4. Explain low-overhead intermediate Latent preview and step progress broadcasting from GPU workers to WebSockets via Redis Pub/Sub?
5. How to broadcast cancellation signals to abort long-running GPU denoising loops immediately upon user disconnects?

<details>
<summary>中文题目</summary>

1. 详细画出客户端、API 网关、Redis/Kafka 任务队列、GPU Worker 集群与 S3 对象存储之间的异步任务流转时序图？
2. 优先级队列 (Priority Queue) 设计：如何为付费订阅会员 (VIP Tier) 提供排队插队与专属 GPU 算力池保障？
3. 任务超时与 Worker 崩溃自愈：如何通过心跳租约 (Heartbeat Lease / Dead Letter Queue) 自动重新分发死掉的 GPU 任务？
4. 实时进度回传机制：GPU Worker 在执行 30 步去噪过程中，如何低开销地通过 Redis Pub/Sub 与 WebSocket 向前端推送进度百分比与 Latent 临时预览图？
5. 用户主动取消长任务 (Task Cancellation)：当用户关闭窗口或点击取消时，如何通过广播信号及时通知 GPU 停止昂贵的后续步数计算？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/async-task-queue-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=async-task-queue-architecture)

---

## 5. Auction Mechanisms: GSP vs VCG · 拍卖博弈机制 GSP vs VCG

> Auction Mechanisms establish the game-theoretic rules determining ad slot allocations and settlement clearing prices among competing advertisers;

**Interview questions**

1. Derive the step-by-step VCG clearing price formula: $P_i = \sum_{j \neq i} v_j(S_{-i}^*) - \sum_{j \neq i} v_j(S^*)$ representing externalities?
2. Compare GSP vs VCG in multi-slot auctions: why does GSP fail strict Incentive Compatibility (truth-telling) across multiple slots?
3. Why do major platforms adopt GSP in production despite VCG's theoretical dominance (explaining explainability and platform revenue stability)?
4. Analyze Reserve Price design in GSP: Soft Floor vs Hard Floor tuning to maximize platform yield without depressing fill rate?
5. Analyze the modern resurgence of First-Price Auctions in header bidding and machine learning Bid Shading algorithms?

<details>
<summary>中文题目</summary>

1. 详细推导 VCG 拍卖定价公式：胜出者 $i$ 应支付的扣费 $P_i = \sum_{j \neq i} v_j(S_{-i}^*) - \sum_{j \neq i} v_j(S^*)$ 的计算全过程？
2. 对比 GSP 与 VCG 在多广告位场景下的核心差异：为什么在多位置场景下 GSP 并不是严格真实报价 (Not Truthful) 的？
3. 为什么在实际工业界中，尽管 VCG 具有完美的真实报价理论性质，绝大多数平台（如 Google, 阿里）仍然选择 GSP 机制？
4. 底价机制 (Reserve Price / Floor Price) 在 GSP 拍卖中的设计：软底价 (Soft Floor) 与硬底价 (Hard Floor) 对平台收入的博弈调优？
5. 现代程序化交易中第一价格拍卖 (First-Price Auction Header Bidding) 的回潮趋势与广告主出价修剪 (Bid Shading) 算法？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/gsp-vcg-auction-mechanisms?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gsp-vcg-auction-mechanisms)

---

## 6. Capacity Estimation & Latency Budgeting · 容量估算与延迟预算推演

> Capacity Estimation & Latency Budgeting is the first-principles quantitative derivation of throughput, storage footprint, network bandwidth, hardware compute costs, and latency SLAs during System Design interviews and engineering capacity planning;

**Interview questions**

1. Derive average QPS, peak QPS (3x), and egress bandwidth for a 100M DAU service with 100 feed requests per user per day?
2. Calculate the exact RAM footprint for 100 million 1024-dim FP32 vector embeddings indexed with HNSW (including 1.5x graph overhead)?
3. Calculate how many H100 GPUs are required to serve 1,000 concurrent streaming LLM sessions at 50 tokens/sec throughput?
4. Apply foundational latency numbers every programmer should know (L1 Cache 0.5ns, RAM 100ns, SSD 100µs, Datacenter RTT 0.5ms) during design derivations?
5. Design a tiered storage cost optimization architecture: 7-day hot data on NVMe SSD vs cold archive on S3 with 3x replication factor?

<details>
<summary>中文题目</summary>

1. 熟练推导“1 亿 DAU、每人每天浏览 100 次、每次请求生成 10 个视频推荐”场景下的平均 QPS、峰值 QPS 与网络带宽需求？
2. 对于包含 1 亿个物品、每个物品 1024 维 FP32 向量（单向量 4KB）的场景，推导 HNSW 内存索引总占用空间（含图拓扑结构开销 $\sim 1.5\times$）？
3. 大模型 Token 吞吐与 GPU 算力推演：在峰值 1000 并发流式请求下，生成速率 50 tokens/s 时需要多少张 H100 显卡？
4. 经典硬件延迟数字常识记忆（L1 缓存 0.5ns、L2 缓存 7ns、RAM 内存访问 100ns、SSD 读取 100µs、同机房网络 RTT 0.5ms、跨洋网络 150ms）在面试中的应用？
5. 存储系统的 3 副本机制 (3-way Replication) 与冷热数据分层（7 天热数据存 NVMe SSD，历史数据转存 S3 冷存储）的成本精算？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/capacity-estimation-latency-budget?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=capacity-estimation-latency-budget)

---

## 7. Context Budgeting & Token Management · 上下文预算与动态 Token 分配

> Context Budgeting & Dynamic Token Allocation is an engineering resource management framework that strategically partitions and dynamically compresses Prompt components under fixed LLM context limits (8k/32k/128k tokens) and quadratic attention compute costs;

**Interview questions**

1. Explain the attention mechanism causes behind the 'Lost in the Middle' effect and how positioning vital chunks at context extremes mitigates it?
2. Design high-speed Token counting and boundary-aware truncation preserving complete sentence semantics rather than raw character splits?
3. Trace the evolution of conversation history compression: sliding window buffers $\to$ asynchronous summary compaction $\to$ episodic vector retrieval?
4. How does strict deterministic prefix ordering of static system prompts and shared chunks maximize vLLM / SGLang Prefix KV-Cache hit rates?
5. Define the cascading priority eviction ladder when user prompts surge: drop lowest-scored RAG chunks $\to$ compress history $\to$ prune tool schemas?

<details>
<summary>中文题目</summary>

1. 详细剖析“Lost in the Middle (迷失在中间)”现象的注意力机制机理，以及在组装 RAG 上下文时如何通过“首尾两端放置最重要 Chunk”进行防御？
2. 基于 Tiktoken / SentencePiece 库的高性能 Token 计数器与动态截断算法（按句子边界优雅截断而非粗暴按字符截断）？
3. 多轮对话历史的动态分级压缩：从滑动窗口 (Sliding Window) $\to$ 异步 LLM 摘要化 (Summarization Memory) $\to$ 向量历史检索 (Episodic Memory) 的演进？
4. KV Cache 命中率 (Prefix Caching / Prompt Caching) 与上下文排列顺序的关系：为什么将固定 System Prompt 与公共 Chunk 严格前置能大幅提升吞吐？
5. 当用户 Prompt 输入长度不可预测地激增时，系统的优先级动态修剪策略（先剪低分 RAG Chunk $\to$ 再压缩历史 $\to$ 最后精简工具定义）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/context-budgeting-token-management?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=context-budgeting-token-management)

---

## 8. DCN-v2 Deep & Cross Network · DCN-v2 深度特征交叉网络

> DCN-v2 (Deep & Cross Network V2, proposed by Google in 2021) is a high-performance ranking network that explicitly and efficiently models arbitrary bounded-degree polynomial feature interactions;

**Interview questions**

1. Derive how the Cross Network recursion $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$ explicitly increments interaction degrees layer-by-layer?
2. How does DCN-v2's Low-Rank factorization ($W = U V^T$) slash computational complexity from $O(d^2)$ to $O(2dr)$?
3. Compare Parallel topology vs Stacked topology in DCN-v2 in terms of gradient propagation and feature synergy?
4. How does the MoE mechanism in DCN-v2 assign inputs across specialized expert sub-networks to capture multimodal interactions?
5. Benchmark floating point operations (FLOPs) and inference throughput (QPS) of DCN-v2 Cross Networks against traditional deep MLPs?

<details>
<summary>中文题目</summary>

1. 详细推导 Cross Network 的层级递推公式 $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$ 如何逐层增加特征多项式交互阶数？
2. DCN-v2 的低秩矩阵分解 (Low-Rank Architecture: $W = U V^T$) 如何将矩阵乘法复杂度从 $O(d^2)$ 降低到 $O(2dr)$？
3. 对比 DCN-v2 的两种典型拓扑组合模式：并联结构 (Parallel: Cross 与 Deep 分支并联) 与串联结构 (Stacked: Cross 堆叠在 Deep 之前) 的优缺点？
4. DCN-v2 中的 Mixture-of-Experts (MoE) 机制如何将输入空间划分给不同专家子空间以捕捉多峰分布特征交叉？
5. 在生产环境部署时，Cross Network 的浮点运算量 (FLOPs) 与传统深层 5 层 MLP 的吞吐量 (QPS) 实测对比？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/dcn-v2-feature-cross?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dcn-v2-feature-cross)

---

## 9. DeepFM Ranking Model · DeepFM 因子分解机精排模型

> DeepFM (Deep Factorization Machine, proposed by Huawei Noah's Ark Lab and CAS in 2017) is a landmark CTR ranking model that seamlessly integrates low-order explicit feature interactions with high-order deep non-linear representations;

**Interview questions**

1. Derive the mathematical proof reducing the FM 2nd-order interaction from $O(d^2)$ down to $O(kd)$ linear complexity?
2. Contrast DeepFM vs Wide & Deep vs FNN regarding embedding weight sharing and pre-training dependency?
3. How should continuous numerical features and high-cardinality categorical features be preprocessed and embedded differently in DeepFM?
4. Analyze DeepFM's log-loss optimization and GPU memory throughput optimization during high-concurrency online inference?
5. Trace the evolution from 2nd-order DeepFM to arbitrary-order explicit cross models like DCN and xDeepFM (Compressed Interaction Network)?

<details>
<summary>中文题目</summary>

1. 推导 FM 二阶特征交叉项如何从 $O(d^2)$ 的双重循环化简为 $O(kd)$ 线性时间复杂度：$\frac{1}{2} \sum_{f=1}^k [(\sum_{i=1}^d v_{i,f} x_i)^2 - \sum_{i=1}^d v_{i,f}^2 x_i^2]$？
2. 对比 DeepFM、Wide & Deep 与 FNN 在底层 Embedding 权重共享机制与预训练依赖上的根本差异？
3. 为什么在 DeepFM 中，连续数值特征 (Numerical Features) 与高维类别特征 (Categorical Features) 需要采用不同的处理与 Embedding 投影？
4. DeepFM 模型的二分类交叉熵损失函数 (Log-loss) 与在线 Serving 阶段的高并发 GPU 显存吞吐优化？
5. 从 DeepFM 到 DCN (Deep & Cross Network) 与 xDeepFM (CIN 向量级交叉) 的显式高阶交叉演进脉络？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/deepfm-ranking-model?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=deepfm-ranking-model)

---

## 10. Device Fingerprinting & Anti-Fraud · 设备指纹与对抗性黑产攻防

> Device Fingerprinting & Adversarial Anti-Fraud Engineering synthesizes multi-dimensional low-level hardware and runtime characteristics into an immutable, globally unique digital identifier (Device ID) resisting spoofing;

**Interview questions**

1. Explain why identical Canvas and WebGL rendering scripts yield deterministic pixel-level hash divergences across distinct GPU hardware?
2. How to detect dynamic runtime instrumentation frameworks (Frida, Xposed) via `/proc/self/maps` scanning and memory code checksums?
3. How to detect emulators and cloud device farms using QEMU virtual drivers, sensor omissions, and synthetic battery states?
4. How to design soft-matching Device ID algorithms using Locality-Sensitive Hashing (LSH) to tolerate partial OS upgrades without identity drift?
5. Design end-to-end anti-tamper communication protocols using TLS Certificate Pinning, white-box cryptography, and dynamic nonces?

<details>
<summary>中文题目</summary>

1. 详细剖析 Canvas 指纹与 WebGL 渲染指纹的底层数学原理：为什么相同代码在不同 GPU 显卡驱动与抗锯齿硬件下的像素哈希值必然产生微小确定性差异？
2. 在移动端 Android/iOS 上，如何检测 Frida/Xposed 等动态 Hook 框架在内存中的特征（如 `/proc/self/maps` 扫描、代码段 Inline Hook 完整性校验）？
3. 模拟器与真机识别技术：如何通过 QEMU 特有驱动文件、电池传感器缺失、CPU 温度不可变等特征识别云手机与自动化设备农场？
4. 设备指纹 ID 的稳定性与抗漂移设计：当用户升级操作系统或更新个别组件时，如何通过局部敏感哈希 (LSH) 保证 Device ID 连续不丢失？
5. 客户端 SDK 与服务端通信的端到端防抓包与重放攻击防御（动态混淆加壳、动态白盒密钥、双向 TLS Pinning 与 Request Token 签名）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/device-fingerprinting-anti-fraud?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=device-fingerprinting-anti-fraud)

---

## 11. Diffusion Pipeline Slicing & Placement · 扩散模型切片与多 GPU 拓扑放置

> Diffusion Model Pipeline Slicing & Multi-GPU Topology Placement distributes large-scale generative models (SDXL, Flux.1, SD3, DiT video generators) across heterogeneous GPU clusters by decomposing distinct functional sub-networks onto tailored hardware: 1) Tex…

**Interview questions**

1. Diagram the multi-GPU placement topology and inter-GPU communication across Text Encoder, DiT Denoising Loop, and VAE Decoder?
2. Analyze Sequential CPU Offloading mechanisms (swapping sub-models between host RAM and VRAM) and its latency trade-offs?
3. Explain how Tiled VAE decoding partitions large latent spaces into overlapping patches with boundary blending to prevent VRAM spikes?
4. How to apply Tensor Parallelism (TP=2/4) and Sequence Parallelism to scale large DiT backbones across NVLink-interconnected GPUs?
5. Design a dynamic multi-tenant LoRA serving architecture hot-swapping customized adapter weights onto a shared base DiT model in VRAM?

<details>
<summary>中文题目</summary>

1. 详细画出 Text Encoder (单次前向) $\to$ DiT/UNet (20-50 步去噪循环) $\to$ VAE Decoder (单次高分重建) 的跨 GPU 流水线放置与数据通信拓扑？
2. 在单卡显存受限场景下，Sequential CPU Offloading (按执行时序逐层将子模型在 CPU RAM 与 GPU VRAM 间换入换出) 的设计与延迟代价？
3. 超大尺寸图像解码时的 Tiled VAE (分块分片解码与边缘平滑羽化融合) 如何将 VAE 显存从 32GB 压缩至 2GB？
4. 针对百亿参数 DiT 模型的张量并行 (Tensor Parallelism: TP=2/TP=4) 与序列并行 (Sequence Parallelism) 在去噪步数上的通信开销优化？
5. 模型权重共享与多租户 LoRA 动态外挂：如何在固定的基础 DiT 模型上并发加载与热切换数百个不同画风的微调 LoRA 权重？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/diffusion-model-pipeline-slicing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=diffusion-model-pipeline-slicing)

---

## 12. Diffusion Step Distillation & Acceleration · 扩散推理步数蒸馏与算子加速

> Diffusion Step Distillation & Full-Stack Operator Acceleration constitutes the end-to-end optimization framework compressing slow iterative denoising (25-50 DDIM steps taking 5-10s) down to sub-second real-time inference (1-4 steps taking $<200\text{ms}$);

**Interview questions**

1. Compare DDIM (25 steps ODE), Latent Consistency Models (4 steps), and Adversarial Distillation (1 step Turbo) in generation quality and fidelity?
2. Explain how CFG requires two forward passes per step and how to fuse unconditional and conditional inputs into a single batched pass?
3. How do TensorRT static engines and CUDA Graph captures eliminate PyTorch CPU runtime overhead during 4-step fast generation loops?
4. Explain dynamic scaling factor calibration in FP8 E4M3 quantization to prevent underflow/overflow in DiT attention layers?
5. Design multi-profile TensorRT engines supporting dynamic aspect ratios and resolutions without triggering runtime recompilation?

<details>
<summary>中文题目</summary>

1. 详细对比 DDIM 采样器 (25 步 ODE 求解)、LCM 潜一致性模型 (4 步) 与 SDXL-Turbo 对抗蒸馏 (1 步) 的数学原理与生成质量权衡？
2. Classifier-Free Guidance (CFG: $\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + s (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$) 在每步去噪中的计算开销，以及如何通过批处理合并 (Batched Forward) 将两次前向合二为一？
3. 基于 TensorRT 的静态图编译 (Static Engine Building) 与 CUDA Graph 捕获技术如何彻底消除 PyTorch Python 运行时的 Kernel 启动开销？
4. FP8 (E4M3 vs E5M2) 低精度格式在扩散模型 DiT 矩阵乘法中的动态缩放因子 (Dynamic Scaling Factor) 与防精度溢出设计？
5. 动态分辨率生成下的动态 Shape (Dynamic Shapes) 与 TensorRT Profile 优化：如何预先构建多档分辨率引擎以避免实时重新编译引擎？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/diffusion-inference-acceleration?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=diffusion-inference-acceleration)

---

## 13. Distributed Inverted Index & Skip-List · 分布式倒排索引与跳表求交

> A Distributed Inverted Index is the foundational data structure and distributed partitioning architecture supporting sub-10ms keyword search across billions of documents;

**Interview questions**

1. Compare Document Partitioning vs Term Partitioning in distributed search across network fan-out and query throughput?
2. Derive the time complexity of Skip-List-accelerated Posting List intersection and optimal skip interval sizing?
3. How do WAND and Block-Max WAND algorithms use upper-bound term scores to prune posting traversals for accelerated Top-K search?
4. Explain integer posting list compression algorithms (VByte, FoR, SIMD-BP128) in reducing RAM footprint and accelerating decompression?
5. Explain the LSM-Tree-style real-time ingestion architecture (Memory Buffers + Immutable Segments + Async Merge) in search engines?

<details>
<summary>中文题目</summary>

1. 详细对比按文档分片 (Document Partitioning: 各分片存部分文档全量词) 与按词项分片 (Term Partitioning: 各分片存部分词全量文档) 在检索吞吐与网络开销上的权衡？
2. 基于跳表 (Skip List / Pointers) 实现两个升序 Posting List 快速求交的算法复杂度推导与跳步步长选取？
3. WAND (Weak AND) 与 Block-Max WAND 算法如何利用各词项的上界最大打分 (Max Score) 在不遍历完整倒排链的情况下实现精准 Top-K 剪枝？
4. 倒排链整数压缩编码算法（如 VByte 变长字节编码、Frame-of-Reference (FoR)、SIMD-BP128）在减少内存与提升 CPU 解压速度中的应用？
5. 实时增量更新 (Real-time Ingestion: 内存 In-Memory Index + LSM-Tree 结构 + 异步段合并 Segment Merge) 的读写一致性架构？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/distributed-inverted-index?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distributed-inverted-index)

---

## 14. Distributed Tracing & Observability · 全链路分布式追踪与可观测性

> Distributed Tracing & LLM Observability (OpenTelemetry, LangSmith, Arize Phoenix, OpenLLMetry) provides the comprehensive telemetry and telemetry telemetry recording full lifecycles across production LLM and Agent pipelines;

**Interview questions**

1. Define the calculation formulas and hardware bottlenecks of TTFT (Time To First Token) vs TPOT (Time Per Output Token)?
2. Design OpenTelemetry-compliant semantic conventions for LLM tracing across prompt tokens, completion tokens, and tool invocations?
3. How does nested Span tree topology record Agent Thought-Action-Observation loops and trigger watchdog circuit breaks on infinite tool loops?
4. Design an inline PII masking interceptor scrubbing credit cards, passwords, and PII from prompts before persisting spans to ClickHouse/Jaeger?
5. How to integrate asynchronous RAGAS span evaluators computing live Faithfulness and Answer Relevance metrics without blocking user responses?

<details>
<summary>中文题目</summary>

1. 详细定义大模型服务两大黄金延迟指标：首字延迟 (TTFT: Time To First Token) 与每输出 Token 延迟 (TPOT: Time Per Output Token) 的计算公式与瓶颈分析？
2. 基于 OpenTelemetry 规范设计自定义 LLM Semantic Conventions（如 `gen_ai.system`, `gen_ai.usage.prompt_tokens`, `gen_ai.request.model`）？
3. 多步 Agent 循环中如何通过 Parent-Child Span 树状拓扑图记录 Thought $\to$ Action $\to$ Observation 的完整轨迹并检测死循环超时？
4. 敏感数据脱敏与隐私保护 (PII Masking)：在追踪日志写入持久化存储（如 ClickHouse / Jaeger）之前，如何通过正则表达式/NER 自动遮蔽身份证、密码与银行卡？
5. 在线生成质量监控：如何将 RAGAS 评估指标（如忠实度 Faithfulness、答案相关度 Answer Relevance）作为异步 Span Evaluator 嵌入追踪流？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/distributed-tracing-observability?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distributed-tracing-observability)

---

## 15. DSSM Two-Tower Retrieval · DSSM 双塔向量化召回

> DSSM Two-Tower Retrieval is an asymmetric neural architecture that decouples user features and item features into two independent sub-networks;

**Interview questions**

1. Detail the compute efficiency of In-batch Negative Sampling in Two-Tower training and methods for Popularity Debias correction?
2. Why does the Two-Tower architecture strictly prohibit early low-level cross-features between user and item inputs?
3. How to update User Tower embeddings in real-time streaming pipelines to capture instant user interest shifts?
4. Explain the role and sensitivity of temperature hyperparameter $\tau$ in Two-Tower Softmax cross-entropy loss?
5. Analyze the optimal sampling mixture of Easy Negatives vs Hard Negatives to prevent embedding collapse and boost discrimination?

<details>
<summary>中文题目</summary>

1. 详细剖析 In-batch Negative Sampling (批内负采样) 在双塔模型训练中的计算优势与极端热门物品偏置消除 (Popularity Debias)？
2. 双塔模型为什么在结构上严禁 User 特征与 Item 特征在底层发生任何显式交叉（Cross Features）？
3. 当用户产生实时动态点击行为时，如何通过 Streaming Realtime User Tower 快速刷新用户表征以实现秒级兴趣捕捉？
4. 训练时使用的 Softmax 交叉熵损失函数中，温度超参数 $\tau$ (Temperature: $P = \frac{e^{u^T v / \tau}}{\sum e^{u^T v_j / \tau}}$) 的调参影响？
5. 分析“简易负例 (Easy Negatives: 随机负采)”与“难负例 (Hard Negatives: 曝光未点击/相似但不感兴趣)”在训练双塔判别能力中的配比？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/dssm-two-tower-retrieval?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dssm-two-tower-retrieval)

---

## 16. ESMM Entire Space Multi-Task Model · ESMM 全空间多任务模型

> ESMM (Entire Space Multi-Task Model, proposed by Alibaba Alimama at SIGIR 2018) is a multi-task learning architecture specifically engineered to resolve the twin industry dilemmas in Conversion Rate (CVR) estimation: Sample Selection Bias (SSB) and Data Sparsi…

**Interview questions**

1. Derive the joint loss function of ESMM over full impression space combining CTR and CTCVR targets?
2. Why does online serving extract the raw output of the CVR Tower directly as the predicted $p\text{CVR}$ without dividing by CTR?
3. How does embedding weight sharing between CTR and CVR towers leverage abundant CTR impressions to alleviate CVR data sparsity?
4. How does ESM2 extend the ESMM framework across multi-stage conversion DAG funnels (Impression -> Click -> Cart -> Purchase)?
5. Why does optimizing CVR via explicit division cause numerical instability, and how does multiplicative CTCVR loss avoid gradient explosions?

<details>
<summary>中文题目</summary>

1. 推导 ESMM 的联合损失函数公式 $\mathcal{L}(\theta_{ctr}, \theta_{cvr}) = \sum_{i=1}^N l(y_i, f(x_i; \theta_{ctr})) + \sum_{i=1}^N l(y_i \& z_i, f(x_i; \theta_{ctr}) \times f(x_i; \theta_{cvr}))$？
2. 为什么在在线推理 (Serving) 阶段，系统只需要直接提取 CVR Tower 的输出值作为预估转化率 $p\text{CVR}$？
3. ESMM 结构中 CTR 塔与 CVR 塔底层共享 Embedding 查找表是如何极大缓解 CVR 塔样本稀疏问题的？
4. 针对多步转化漏斗（如 曝光 -> 点击 -> 加购 -> 购买），如何将 ESMM 进一步拓展为 ESM2 (有向无环图全空间多步转化网络)？
5. 分析为什么直接在训练中让 CVR 塔以除法形式优化可能会导致数值不稳定性（如 $p\text{CTR} \to 0$ 时梯度爆炸）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/esmm-sample-selection-bias?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=esmm-sample-selection-bias)

---

## 17. Feature Store Online/Offline Consistency · Feature Store 离在线特征一致性

> A Feature Store (Feast, Hopsworks, Tecton) is an enterprise MLOps data infrastructure that standardizes the authoring, computation, storage, and serving of machine learning features;

**Interview questions**

1. Detail the algorithmic principles of Point-in-Time Join and how it mathematically prevents feature leakage from future timestamps?
2. Design a streaming real-time feature computation pipeline using Kafka, Flink, and Redis for 5-minute sliding window aggregations?
3. What are the default fallback values and graceful degradation policies when the online Redis feature store encounters latency spikes?
4. Explain versioning and metadata registration mechanisms reconciling daily batch features with second-level streaming features?
5. How to deploy production feature drift and data quality monitoring (missingness rate, PSI, distribution shifts) to catch pipeline breakages?

<details>
<summary>中文题目</summary>

1. 详细剖析 Point-in-Time Join (时点对齐连接) 的算法原理与如何从数学和时间戳上彻底防止数据穿越 (Data Leakage)？
2. 基于 Kafka + Flink + Redis 构建实时流式特征计算管道（如滑动窗口 5 分钟用户点击数、实时实时类目偏好）的架构？
3. 当在线 Redis 发生局部节点故障或网络超时（>5ms）时，Feature Store 的特征默认值兜底与降级熔断策略？
4. 离线批特征 (Batch Features: 每日更新) 与在线流特征 (Streaming Features: 秒级更新) 的版本管理与元数据注册机制？
5. 如何在生产中部署特征监控告警系统（监控特征缺失率、漂移度 PSI、方差异常）以及时发现数据源断流？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/feature-store-consistency?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=feature-store-consistency)

---

## 18. Flink Real-Time Feature Stream · Flink 实时特征计算与滑动窗口

> The Flink Real-Time Streaming Feature Platform serves as the mission-critical data heartbeat of real-time fraud detection systems;

**Interview questions**

1. Explain the distinctions among Event Time, Processing Time, and Ingestion Time, and how Watermark generation handles out-of-order logs?
2. How to avoid state memory explosion in Flink sliding windows (1-hour window, 1-sec slide) using incremental AggregateFunctions?
3. Explain the end-to-end Exactly-Once processing semantics implemented via Chandy-Lamport distributed checkpointing and two-phase commit sinks?
4. How does HyperLogLog enable ultra-low-memory real-time cardinality estimation (distinct cards per device) inside Flink streaming states?
5. Design fast recovery procedures for multi-hundred GB state backends using RocksDB incremental snapshots and Flink Savepoints?

<details>
<summary>中文题目</summary>

1. 详细剖析 Event Time、Processing Time 与 Ingestion Time 在乱序网络事件下的区别，以及 Watermark 水位线容忍延迟的设计？
2. Flink 滑动窗口 (Sliding Window: 窗口 1 小时，滑动步长 1 秒) 的状态爆炸问题，如何通过增量聚合函数 (AggregateFunction) 降维优化？
3. Exactly-Once 精确一次处理语义在 Flink 算子状态与下游 Redis/Kafka 写入中的两阶段提交 (2PC / Chandy-Lamport Checkpointing) 实现？
4. 实时风控中的“实体关联特征（如同手机号关联设备数、同设备关联银行卡数）”如何通过 HyperLogLog 算法实现基数极速估算？
5. 当 Flink 任务发生故障重启时，如何通过 Savepoint 与 RocksDB 状态后端在数秒内快速恢复百 GB 级别的窗口时序状态？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/flink-realtime-feature-stream?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=flink-realtime-feature-stream)

---

## 19. Graph Risk & Collusion Detection · 图风控与黑灰产团伙识别

> Graph-based Risk Control & Collusion Syndicate Detection models transaction ecosystems as Heterogeneous Information Networks (HIN) to unmask organized syndicate fraud, money laundering, and coordinated bot armies;

**Interview questions**

1. Diagram characteristic money laundering graph motifs: Fan-in aggregation, Fan-out dispersion, and Cyclic transfer paths?
2. Explain how the Louvain algorithm optimizes modularity $Q$ to discover densely connected fraud syndicates hierarchically?
3. How to perform distributed GNN training over billion-scale heterogeneous risk graphs using Graph Partitioning and PyG/DGL?
4. How do CARE-GNN and GAT mechanisms adaptively prune adversarial camouflage edges injected by fraudsters to trick GNN aggregators?
5. Design an in-memory graph cache architecture enabling sub-10ms 2-hop neighborhood traversals during live transaction gatekeeping?

<details>
<summary>中文题目</summary>

1. 详细画出金融洗钱中的“漏斗汇聚型 (Fan-in)”、“分散分流型 (Fan-out)”与“循环洗钱环 (Cyclic Motifs)”的典型图拓扑结构？
2. Louvain 社区发现算法的模块度 (Modularity: $Q = \frac{1}{2m} \sum_{i,j} [A_{ij} - \frac{k_i k_j}{2m}] \delta(c_i, c_j)$) 优化原理与多尺度分层聚类？
3. 面对包含数亿节点与百亿边的大规模异构图，如何通过图分块 (Graph Partitioning) 与分布式图引擎 (GraphScope / DGL / PyG) 实现分布式图训练？
4. CARE-GNN 与 GAT 如何在邻居节点聚合时抵抗黑产故意添加的虚假伪装边（Camouflage Attack / Adversarial Edges）？
5. 在线实时图特征抽取（如通过 Neo4j / Nebula Graph 在 10ms 内完成 2 度关联跳步查询与度中心性度量）的工程架构与缓存？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/graph-risk-collusion-detection?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=graph-risk-collusion-detection)

---

## 20. Imbalanced & Cost-Sensitive Learning · 极端不平衡样本与代价敏感学习

> Imbalanced Classification & Cost-Sensitive Machine Learning addresses the extreme sample skew (fraud positive prevalence $<0.01\% \sim 0.1\%$, ratios exceeding $1:10,000$) and highly asymmetric misclassification costs in financial risk modeling;

**Interview questions**

1. Derive Focal Loss $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$ and explain how the focusing parameter $\gamma$ downweights easy examples?
2. Compare EasyEnsemble vs BalanceCascade in training multiple bagging learners over partitioned negative sample majorities?
3. Why is Accuracy prohibited in fraud evaluation, and how do KS statistic, PR-AUC, and Top-1% Capture Rate serve as authoritative metrics?
4. Derive the optimal Bayesian decision threshold $p^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$ from cost matrix parameters $C_{\text{FP}}$ and $C_{\text{FN}}$?
5. How do unsupervised anomaly detection models (Isolation Forest, AutoEncoders) capture novel zero-day fraud without labeled positives?

<details>
<summary>中文题目</summary>

1. 推导 Focal Loss 的数学形式 $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$，并解释调节参数 $\gamma$ 如何动态降低易分样本权重？
2. 详细对比 EasyEnsemble 与 BalanceCascade 在利用海量负样本构建 Bagging 子模型中的负样本抽样策略？
3. 风控模型评估中为什么严禁使用 Accuracy？如何使用 KS 值 (Kolmogorov-Smirnov)、PR-AUC 与 Top-1% 捕获率作为核心验收标准？
4. 构建风控代价敏感决策矩阵：根据预期坏账损失 $C_{\text{FN}}$ 与误杀用户投诉成本 $C_{\text{FP}}$ 推导最优二分类决策阈值 $p^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$？
5. 无监督与半监督异常检测模型（如 Isolation Forest 孤立森林、One-Class SVM、AutoEncoder 重构误差）在应对全新零日欺诈模式中的应用？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/imbalanced-cost-sensitive-learning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=imbalanced-cost-sensitive-learning)

---

## 21. Learning to Rank & LambdaMART · 排序学习 LTR 与 LambdaMART

> Learning to Rank (LTR) is the systematic methodology applying machine learning to optimize ranking in Information Retrieval;

**Interview questions**

1. Derive the LambdaRank virtual gradient formula $\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta \text{NDCG}_{ij}|$ and explain its intuitive physics interpretation?
2. Compare Pointwise, Pairwise, and Listwise LTR loss functions in position sensitivity and computational complexity?
3. Why does LambdaMART (LightGBM) regularly outperform deep neural nets on dense tabular feature sets?
4. How to compile thousands of GBDT trees into native C++ instructions (e.g. Treelite) for sub-millisecond serving?
5. Trace the modern evolution from tree-based LambdaMART to Transformer-based Neural Listwise Rankers (MonoT5, RankGPT)?

<details>
<summary>中文题目</summary>

1. 推导 LambdaRank 中的虚拟受力梯度公式 $\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta \text{NDCG}_{ij}|$ 的物理与数学含义？
2. 对比 Pointwise、Pairwise 与 Listwise 三种 LTR 损失函数在位置偏置敏感性与复杂度上的优缺点？
3. 为什么在拥有数千个复杂工程统计特征（点击率、价格差、历史购买频次）的表格数据上，LambdaMART (LightGBM) 表现往往超越普通未调优的深度学习？
4. 基于 LightGBM / XGBoost 的 GBDT 快速推理优化：如何将数千棵树编译为 C++ 原生汇编或 if-else 决策流以降低 Serving 延迟？
5. 从 LambdaMART 到神经排序模型 (Neural Rankers: 如 Transformer-based RankGPT / MonoT5) 的技术演进与混合集成？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/learning-to-rank-lambdamart?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=learning-to-rank-lambdamart)

---

## 22. Memory Rule Engine & Bloom Filters · 内存规则引擎与布隆过滤器

> The In-Memory Rule Engine & Bloom Filter System provides the ultra-fast sub-millisecond first line of defense in financial risk architectures;

**Interview questions**

1. Derive the Bloom filter false positive rate $P \approx (1 - e^{-kn/m})^k$ and the optimal number of hash functions $k = \frac{m}{n} \ln 2$?
2. Why does a Bloom filter mathematically guarantee zero false negatives while tolerating a bounded false positive probability?
3. How do Counting Bloom Filters and Cuckoo Filters support dynamic deletion and resizable scaling for expiring blocklists?
4. How does compiling rule DSLs into native bytecode/machine code eliminate runtime interpretation overhead in high-throughput engines?
5. Explain how adaptive rule prioritization based on historical hit rates and short-circuit boolean logic slashes average evaluation time?

<details>
<summary>中文题目</summary>

1. 推导标准布隆过滤器的假阳性率 (False Positive Rate) 公式 $P \approx (1 - e^{-kn/m})^k$ 及最优哈希函数个数 $k = \frac{m}{n} \ln 2$？
2. 布隆过滤器为什么天然支持“判断元素绝对不在集合中 (Zero False Negative)”，但不保证“绝对在集合中 (False Positive)”？
3. 计数布隆过滤器 (Counting Bloom Filter) 与 Cuckoo Filter (布谷鸟过滤器) 如何实现黑名单元素的动态删除与扩容？
4. 内存规则引擎的 AST 表达式编译（如基于 Bytecode 编译为 JVM 字节码或 C++ 机器码）在压榨 CPU 吞吐中的实践？
5. 多规则并发执行时的短路求值 (Short-Circuit Evaluation) 与基于历史命中率自适应重排规则优先级的优化？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/memory-rule-engine-bloom?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=memory-rule-engine-bloom)

---

## 23. MMoE Multi-Gate Mixture-of-Experts · MMoE 多门控专家混合模型

> MMoE (Multi-gate Mixture-of-Experts, proposed by Google at KDD 2018) is a multi-task learning architecture designed to resolve negative transfer across weakly correlated or conflicting objectives in recommendation ranking;

**Interview questions**

1. Derive MMoE's multi-gate formulation $y_k = h^k(\sum_{i=1}^E g_i^k(x) f_i(x))$ and contrast with standard single-gate MoE and Shared-Bottom?
2. Why does MMoE's gating network automatically orthogonalize expert attention when tasks are negatively correlated (e.g. clickbait)?
3. How do dynamic loss balancing algorithms (GradNorm / Uncertainty Weighting) dynamically tune task weights $w_k$ in multi-task loss?
4. Analyze how the number of experts $E$ impacts parameter bloat and GPU serving latency in production?
5. What are the structural limitations of MMoE (all experts are shared, lacking task-exclusive isolation), leading to Tencent's PLE?

<details>
<summary>中文题目</summary>

1. 推导 MMoE 的多门控专家数学表达式 $y_k = h^k(\sum_{i=1}^E g_i^k(x) f_i(x))$，并对比其与单门控 MoE 及 Shared-Bottom 的差异？
2. 为什么当两个任务完全不相关或呈负相关（如“点击率高但停留时长极短的标题党”）时，MMoE 的门控网络能自动将专家注意力正交化？
3. 多任务损失函数加权 $\mathcal{L} = \sum_{k=1}^K w_k \mathcal{L}_k$ 中，动态损失平衡算法（如 GradNorm / UW 不确定性加权）如何动态调优各任务权重 $w_k$？
4. MMoE 专家池数量 $E$（如 $E=4\sim 8$）的选择对模型参数量膨胀与 GPU 算力吞吐的量化制约关系？
5. MMoE 架构为何仍然存在“所有专家均为完全共享，缺乏任务独占专家”的局限性？（由此引出腾讯的 PLE 架构）

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/mmoe-multitask-learning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mmoe-multitask-learning)

---

## 24. Multi-Tenant Partitioning & RBAC · 企业级多租户隔离与 RBAC 权限

> Multi-Tenant Partitioning & Role-Based Access Control (RBAC) enforces strict physical/logical isolation and granular access authorization across B2B SaaS customers and sensitive enterprise departments (HR, Legal, Finance);

**Interview questions**

1. Compare Metadata Filtering, Collection Namespaces, and Physical Cluster Isolation across cost, isolation security, and query latency?
2. Why does secure RAG mandate Pre-Filtering (ACL masking during HNSW graph traversal) rather than Post-Filtering after candidate generation?
3. Design IAM token parsing at the gateway that flattens LDAP/Active Directory organizational hierarchies into chunk-level ACL filters?
4. How to handle immediate permission revocation upon employee departures without re-embedding millions of vector chunks?
5. Design tamper-proof audit logging and AES-256 encryption at rest/transit to pass SOC2 and GDPR multi-tenant enterprise audits?

<details>
<summary>中文题目</summary>

1. 对比三种向量库多租户架构：标量元数据过滤 (Metadata Filter)、独立 Collection/Namespace、物理独占集群在成本、隔离性与 QPS 上的权衡？
2. 为什么在向量检索中必须执行“前置过滤 (Pre-Filtering: 在 HNSW 遍历图时施加 ACL 掩码)”，而不是“后置过滤 (Post-Filtering)”？
3. 基于 JWT Token 的 IAM 权限在网关层的无状态解析与将组织架构树 (LDAP / Active Directory) 动态扁平化为 Chunk ACL 数组的设计？
4. 当企业员工发生离职或部门调岗时，如何高效处理该用户在海量向量文档上的权限撤销与即时生效（无需全量重新 Embedding）？
5. 合规审计与防数据投毒：多租户环境下如何通过数据加密 (AES-256 at Rest & TLS in Transit) 与不可篡改审计日志防范越权探测？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/multi-tenant-partitioning-rbac?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multi-tenant-partitioning-rbac)

---

## 25. Netflix Page Generation & RecSys · Netflix 流媒体推荐与整页生成

> The Netflix Page Generation & Two-Dimensional Personalization Architecture elevates recommendation from flat 1D item rankings into dynamic 2D row-and-column layout synthesis;

**Interview questions**

1. Explain the classic Offline-Nearline-Online three-tier dataflow architecture powering Netflix page personalization?
2. How to enforce dynamic cross-row deduplication across dozens of horizontal carousels while preserving row relevance integrity?
3. Model Personalized Artwork Selection using Contextual Multi-Armed Bandits to maximize thumbnail click-through rates?
4. Why does Netflix optimize for Session Watch Time and Retention rather than raw click-through rate in its recommendation loss functions?
5. Design responsive page generation caching and prefetching strategies tailored to distinct clients (Smart TV vs Mobile App)?

<details>
<summary>中文题目</summary>

1. 详细剖析“离线 (Offline: 批处理深度模型) - 近线 (Nearline: 实时事件流与会话状态) - 在线 (Online: 规则组装与毫秒渲染)”三层数据流架构？
2. 整页生成中如何解决“跨行重复 (Cross-Row Deduplication)”与“信息茧房 (Filter Bubble)”：如何避免同一部热门剧出现在多个不同推荐行中？
3. 基于 Contextual Multi-Armed Bandit (上下文多臂老虎机) 算法进行个性化海报封面分流与动态收益评估的数学建模？
4. 流媒体核心业务指标的选取：为什么 Netflix 采用“会话观看时长 (Session Watch Time) 与次月留存率”而非单纯的“点击率 (CTR)”作为核心 KPI？
5. 不同客户端（4K 智能电视大屏遥控器浏览 vs 移动端小屏快速滑动）在行高、展示项数与预加载缓存策略上的系统自适应设计？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/netflix-recommendation-page-generation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=netflix-recommendation-page-generation)

---

## 26. Offline Ingestion & Vector Snapshot · 离线文档解析与向量快照流水线

> The Offline Document Ingestion & Vector Snapshot Pipeline is the enterprise-grade data engineering foundation of production LLM RAG systems processing millions of heterogeneous unstructured files (PDF, DOCX, PPT, HTML, Scanned Images);

**Interview questions**

1. Explain best practices for layout-aware PDF parsing across multi-column texts, nested figures, and table-to-Markdown conversions?
2. Design a distributed Ray/Spark batch embedding pipeline leveraging dynamic batching to maximize GPU Tensor Core throughput?
3. Why does the Parent-Child chunking strategy match small child chunks for vector precision while returning large parent chunks for LLM context?
4. How to design Blue-Green immutable vector index snapshots enabling instant atomic zero-downtime pointer swaps and rollbacks?
5. How to handle incremental updates and deletions via document content hashing and soft-delete metadata tombstones in vector stores?

<details>
<summary>中文题目</summary>

1. 复杂版式 PDF（跨页表格、双栏排版、嵌套图表）的结构化解析与表格 HTML/Markdown 转换最佳实践？
2. 基于 Ray / Spark 的分布式 Embedding 计算流水线：如何通过动态批处理 (Dynamic Batching) 与 GPU 流水线最大化 Tensor Core 利用率？
3. Parent-Child (父子分块) 索引策略：为什么在检索时用小粒度子 Chunk 算向量相似度，而在上下文喂给 LLM 时换用大粒度父 Chunk？
4. 向量索引的双版本快照 (Blue-Green Snapshot) 构建与元数据版本管理：如何实现百万级文档索引的秒级原子切换与回滚？
5. 文档增量更新与失效处理：当源文档被修改或删除时，如何通过文档 Hash 指纹与元数据过滤实现增量修剪？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/offline-document-ingestion-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=offline-document-ingestion-pipeline)

---

## 27. Online A/B Testing & Interleaving · 推荐在线 A/B 实验与 Interleaving

> Online A/B Testing & Interleaving constitute the gold-standard causal decision frameworks evaluating if algorithmic revisions deliver authentic business impact before 100% rollout;

**Interview questions**

1. Detail the Team Draft Interleaving algorithm and unbiased credit assignment for user clicks?
2. Why does Interleaving achieve identical statistical power with 1% sample size and 1/10 the duration of standard A/B tests?
3. Explain the architecture of multi-layer orthogonal experiment frameworks (Google's layered hashing) for running concurrent tests?
4. How to detect Sample Ratio Mismatch (SRM) via Chi-Square tests and mitigate network spillover in social networks?
5. How to isolate the Novelty Effect and measure the Slow Cooker Effect over multi-week experimentation windows?

<details>
<summary>中文题目</summary>

1. 详细描述 Team Draft Interleaving (球队选秀交替算法) 的交替合并规则与点击归属 (Credit Assignment) 的无偏计算？
2. 为什么 Interleaving 能够在仅需标准 A/B 测试 1% 的样本量和 1/10 的时间下达到相同的统计功效 (Statistical Power)？
3. 多层正交实验分层体系 (Overlapping Layered Experiment Architecture / Google 分层实验) 的参数哈希设计？
4. 如何检测与防范 A/B 实验中的样本比例不平衡 (Sample Ratio Mismatch, SRM) 与网络溢出效应 (Network Spillover Effects)？
5. 新颖性效应 (Novelty Effect: 用户因 UI 新鲜产生短期偏好) 与慢炖效应 (Slow Cooker Effect: 长期留存需数周显现) 的应对策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/recsys-ab-testing-interleaving?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=recsys-ab-testing-interleaving)

---

## 28. Online Parallel Retrieval Gateway · 在线多路并行检索网关

> The Online Parallel Retrieval Gateway serves as the high-throughput coordination nexus for live RAG requests under a strict $<50\text{ms}$ SLA;

**Interview questions**

1. Design a high-throughput async fan-out and fan-in retrieval gateway in Go/Rust with strict deadline contexts?
2. Explain why RRF mathematically achieves unbiased candidate fusion across incompatible score spaces (BM25 vs Cosine Similarity)?
3. How to optimize Cross-Encoder GPU memory and latency budgets using TensorRT / ONNX Runtime quantization for sub-20ms reranking?
4. Explain sub-query decomposition strategies that split comparative user questions into parallel search branches before merging?
5. Design circuit breaking and fallback policies that return partial BM25 results when vector database nodes time out past 30ms?

<details>
<summary>中文题目</summary>

1. 基于 Go (Goroutines) / Rust (Tokio) 设计高性能异步检索网关的并发扇出 (Fan-out) 与聚合 (Fan-in) 模式？
2. 倒数排名融合算法 (RRF: $\text{RRF}(d) = \sum_{m=1}^M \frac{1}{60 + r_m(d)}$) 在融合 BM25 得分与 Dense Cosine 得分时的数学无偏性？
3. Cross-Encoder 重排序模型（如 BGE-Reranker-Large）的 GPU 显存占用与推理耗时优化（量化 TensorRT-LLM / ONNX Runtime）？
4. 子查询并行拆解 (Sub-Query Decomposition: 将“对比 iPhone 15 与小米 14 的续航”拆为两个并行 Query) 的聚合与去重设计？
5. 当某个底层向量库节点发生网络超时（如超过 30ms 阈值）时，网关的自动短路与基于 BM25 结果的降级保底机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/online-parallel-retrieval-gateway?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=online-parallel-retrieval-gateway)

---

## 29. PID Budget Pacing & Ad Blending · PID 预算平滑与混排 Blending

> PID Budget Pacing & Ad-Organic Blending are control-theoretic architectures governing spend velocity and user experience harmony in digital advertising;

**Interview questions**

1. Detail the PID budget pacing state-error equation $e(t) = \text{Target}(t) - \text{Spent}(t)$ and the control update law?
2. Compare Probabilistic Throttling Pacing vs Bid Modification Pacing in advertiser ROI and auction win rates?
3. How to quantify and price the negative externality/churn cost of ad insertion on organic feed engagement in Blending?
4. Design an exploration allocation strategy using Multi-Armed Bandits (UCB / Thompson Sampling) to accelerate new ad cold-starts?
5. How to design Anti-Windup mechanisms and adaptive gains in PID pacing controllers to prevent runaway overshoots during traffic surges?

<details>
<summary>中文题目</summary>

1. 详细描述 PID 预算平滑控制器的状态误差方程 $e(t) = \text{BudgetTarget}(t) - \text{BudgetSpent}(t)$ 与控制律计算公式？
2. 基于概率丢弃 (Probabilistic Throttling) 的 Pacing 方案与基于动态出价修改 (Bid Modification / Shading) 的 Pacing 方案的优缺点对比？
3. 广告与自然结果混排 (Ad-Organic Blending) 中，如何量化“广告插入对自然内容点击率的负外部性冲击 (Negative User Experience Penalty)”？
4. 新广告冷启动策略：如何利用 Multi-Armed Bandit (UCB / Thompson Sampling) 算法分配探索流量以快速积累统计置信度？
5. 突发流量峰值（如突发热点新闻或大促零点）下的 PID 控制器抗积分饱和 (Anti-Windup) 与自适应增益保护设计？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/pid-budget-pacing-blending?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pid-budget-pacing-blending)

---

## 30. Pinterest Visual Search System · Pinterest 亿级视觉搜索与推荐

> The Pinterest Visual Search & Multi-Modal Recommendation System is a global industry benchmark supporting billions of visual pins and hundreds of millions of users across an immense bipartite interaction graph;

**Interview questions**

1. Explain the PinSage algorithm: random walk localized importance sampling and map-reduce distributed graph convolution scaling?
2. Diagram the end-to-end visual search pipeline from client crop detection to CNN embedding, ANN recall, and deduplication?
3. How does perceptual hashing (pHash/SimHash) and Hamming distance clustering filter visual duplicates in recommendation feeds?
4. Explain the Leaf-Root cluster routing architecture in the Manas vector search engine for load-balancing billions of embeddings?
5. How to mathematically blend visual similarity scores with user collaborative filtering signals to avoid overly rigid visual clustering?

<details>
<summary>中文题目</summary>

1. 详细剖析 PinSage 算法基于重要性采样随机游走 (Random Walk Importance Sampling) 与局部图卷积算子的数学原理与可扩展性？
2. Pinterest 视觉搜索 (Visual Lens) 的全链路架构：从客户端局部裁剪 (Crop) $\to$ 目标检测 $\to$ CNN 视觉嵌入 $\to$ ANN 检索 $\to$ 混合去重？
3. 紧凑图像哈希 (Compact Perceptual Hashing / SimHash) 在百万级图片推荐中进行近重复 (Near-Duplicate) 图片去重的算法与汉明距离阈值？
4. Manas 分布式向量搜索引擎的分片路由策略 (Leaf/Root Architecture) 与长尾查询的高并发负载均衡？
5. 如何权衡视觉外观相似度 (Visual Similarity) 与商品协同过滤偏好 (Collaborative Signal) 以防止推荐过于死板同质化？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/pinterest-visual-search-system?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pinterest-visual-search-system)

---

## 31. PLE Progressive Layered Extraction · PLE 渐进式分层抽取架构

> PLE (Progressive Layered Extraction, proposed by Tencent PCG at RecSys 2020 Best Paper) is a breakthrough multi-task recommendation architecture that eliminates Negative Transfer and Seesaw Phenomena;

**Interview questions**

1. Diagram the routing flow of task-specific vs shared experts across single-layer CGC and multi-layer progressive cascades in PLE?
2. Compare Shared-Bottom, Cross-Stitch, MMoE, and PLE in robustness against negatively correlated tasks (high click, zero completion)?
3. Why does cascading extraction (shallow layers extracting generic representations, deep layers isolating specific tasks) align with deep representation learning?
4. What are the parameter budget controls and distributed training VRAM optimizations when scaling PLE to 5-8 simultaneous targets?
5. How to combine multiple predicted probabilities ($p\text{CTR}, p\text{CVR}, p\text{Finish}$) into a unified ranking score in final serving?

<details>
<summary>中文题目</summary>

1. 详细画出 PLE 单层 CGC (Customized Gate Control) 模块与多层 Progressive 级联结构中的专家路由信息流图？
2. 对比 Shared-Bottom、Cross-Stitch、MMoE 与 PLE 在处理负相关任务（如高点击但低完播）时的抗干扰能力？
3. 为什么在多层 PLE 级联中，浅层抽取通用底层表征、深层抽取高阶任务专属特征符合深度表征学习的层级规律？
4. PLE 模型在包含 5-8 个目标的大规模工业级推荐系统中的参数量控制与训练显存优化策略？
5. 在最终融合打分阶段，如何将 PLE 输出的多个预估概率值（$p\text{CTR}, p\text{CVR}, p\text{Finish}$）组合为最终排序得分 $S = w_1 p\text{CTR} + w_2 p\text{CVR} + w_3 p\text{Finish}$？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/ple-multitask-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ple-multitask-architecture)

---

## 32. Pre-Ranking Lightweight Architecture · 粗排轻量模型与向量相似度剪枝

> The Pre-Ranking (Match-Filtering) Architecture is the critical intermediate layer bridging multi-channel retrieval and heavy ranking;

**Interview questions**

1. Compare Two-Tower Pre-ranking vs Lightweight Interactive Pre-ranking (COLD) across expressive power and compute latency?
2. When pre-ranker shares embedding tables with the heavy ranker, how to prevent pre-ranker gradients from corrupting embedding precision?
3. How does Knowledge Distillation train the lightweight pre-ranker using the heavy ranker's soft prediction scores as teacher targets?
4. Why does pre-ranking evaluation prioritize Top-K Hit Rate / Overlap with heavy ranking outputs alongside standalone AUC?
5. How do C++ serving engines leverage AVX-512 SIMD and INT8 quantization to maximize pre-ranking throughput over 1000 items in 5ms?

<details>
<summary>中文题目</summary>

1. 详细对比双塔粗排 (Two-Tower Pre-ranking) 与轻量交互粗排 (COLD / Light DNN) 在特征表达力与算力延迟上的权衡？
2. 粗排与精排共享 Embedding 表 (Shared Embedding) 时，如何防止粗排梯度回传破坏精排高精度的 Embedding 表征？
3. 知识蒸馏在粗排模型训练中的应用：如何以精排模型的 Soft-Target 预估分为教师信号训练轻量粗排网络？
4. 粗排的评估指标：为什么粗排的核心指标不仅看单独的 AUC，更要看与精排 Top-K 结果的命中重合率 (Top-K Overlap / Hit Rate)？
5. 在千级候选打分时，如何在 C++ 推理引擎中利用 SIMD 与模型量化 (INT8/FP16) 压榨极致的算力吞吐？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/pre-ranking-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pre-ranking-architecture)

---

## 33. Query Understanding Pipeline · Query 理解、分词与意图分类

> The Query Understanding Pipeline (QU) is the front-end NLP intelligence hub of search engines and e-commerce portals that structurally decodes raw search intents within $<5\text{ms}$ before physical indexing: 1) Normalization (punctuation stripping, case foldi…

**Interview questions**

1. Detail Term Weighting algorithms calculating importance weights between core entity nouns and modifier adjectives?
2. Design low-latency C++ inference acceleration for Transformer sequence-labeling NER models identifying e-commerce entities?
3. Explain the two-stage Query Spelling Correction architecture: Trie/phonetic candidate recall followed by n-gram/LM re-scoring?
4. How does synonym expansion prevent over-generalization via context-aware disambiguation (e.g. 'Apple' as fruit vs tech giant)?
5. How does Category Prediction enforce hard category filtering or soft boosting in e-commerce search to prevent irrelevant results?

<details>
<summary>中文题目</summary>

1. 详细剖析 Term Weighting (词权重计算) 算法：如何通过结巴/深度模型计算核心实体词与辅助修饰词的权重差异并在检索中加权？
2. 基于 Transformer 序列标注的电商多目标 NER (实体识别: 品牌/品类/属性/型号) 在低延迟 C++ 推理引擎中的加速方案？
3. Query 纠错系统中的候选召回（基于 Trie 字典树、编辑距离、拼音索引）与基于统计语言模型打分排序的两阶段架构？
4. 同义词扩展 (Synonym Expansion) 如何防止过度泛化（如“苹果”在不同上下文中被误扩展为水果或手机）的上下文消歧机制？
5. 电商搜索中的类目预测 (Category Prediction) 如何通过强行注入类目硬过滤或软提权 (Category Boosting) 避免搜索结果风马牛不相及？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/query-understanding-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=query-understanding-pipeline)

---

## 34. Queue-Depth-Driven GPU Autoscaling · 队列深度驱动的 GPU 弹性扩缩容

> Queue-Depth-Driven GPU Autoscaling (Kubernetes KEDA, Karpenter, Prometheus Custom Metrics) governs dynamic compute elasticity across heavy generative workloads, harmonizing strict queue wait SLAs with aggressive multi-million-dollar cloud GPU cost optimization…

**Interview questions**

1. Derive the mathematical formula for target replica counts based on queue backlog depth, worker capacity, and target latency?
2. How to accelerate multi-gigabyte GPU container cold starts from minutes to seconds using Stargz lazy-pulling and pre-warmed daemon nodes?
3. Design a hybrid Spot and On-Demand GPU fleet architecture that exploits 70% Spot discounts while hedging against node reclamation?
4. How does Kubernetes Pod Disruption Budget and preStop hook drain active long-running diffusion tasks before terminating GPU instances?
5. Explain the design of stabilization cooldown windows preventing metric flapping and thrashing during transient load dips?

<details>
<summary>中文题目</summary>

1. 推导基于队列积压深度与目标排队延迟的目标副本数计算公式 $R = \lceil \frac{\text{QueueLength}}{\text{CapacityPerWorker} \times \text{TargetLatency}} \rceil$？
2. GPU 容器冷启动耗时过长（拉取 20GB 大模型镜像与加载权重需 3~5 分钟）的秒级加速方案（Stargz 懒加载、镜像缓存节点池、DaemonSet 预热）？
3. 混合竞价实例池管理：如何将 Spot 抢占式 GPU（价格仅为按需的 20%~30%）与 On-Demand 保底按需实例混合编排以兼顾极致成本与高可用？
4. 优雅缩容与任务排空 (Graceful Scale-Down & Draining)：当扩缩容控制器决定缩减 GPU 节点时，如何防止正在生成视频的任务被强行 SIGKILL 杀死？
5. 缩容防抖动 (Scale-Down Stabilization Window)：如何通过冷却时间窗口（如维持 5-10 分钟平稳期）防止流量短暂波动导致 GPU Pod 频繁反复创建与销毁？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/queue-depth-gpu-autoscaling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=queue-depth-gpu-autoscaling)

---

## 35. Real-Time Bidding eCPM Ranking · 实时竞价 eCPM 排序与计费公式

> Real-Time Bidding (RTB) eCPM Ranking is the unifying mathematical standard across digital advertising exchanges that standardizes heterogeneous advertiser bidding models into expected revenue per thousand impressions (eCPM): 1) CPM Bidding:…

**Interview questions**

1. Derive the oCPM dynamic bidding formula and prove why smart bidding adjusts effective CPC to $\text{bid}_{\text{CPC}} = p\text{CVR} \times \text{bid}_{\text{CPA}}$?
2. How to design composite ad ranking functions incorporating quality scores $Q$ alongside pure eCPM to protect user experience?
3. Break down the RTB latency budget across SSP request dispatch, DSP internal retrieval, pCTR inference, and auction clearance within 50ms?
4. Explain the business rules and mathematical risk controls behind platform CPA guarantee and compensation mechanisms during advertiser cold-start?
5. Derive the GSP clearing price formula for the $i$-th ad: $P_i = \frac{\text{eCPM}_{i+1}}{1000 \times p\text{CTR}_i} + 0.01$?

<details>
<summary>中文题目</summary>

1. 推导从 CPA 目标出价到 oCPM 实时出价公式：为什么智能出价的底层本质是动态将出价折算为 $\text{bid}_{\text{CPC}} = p\text{CVR} \times \text{bid}_{\text{CPA}}$？
2. 当广告系统引入广告质量度评分 $Q$（如用户体验分、落地页质量）时，综合排序得分 $\text{RankScore} = f(\text{eCPM}, Q)$ 的设计原则？
3. 广告系统在面临展示机会时的竞价链路耗时分解（DSP 竞价请求 $\to$ 内部检索 $\to$ pCTR 计算 $\to$ eCPM 排序 $\to$ 竞价响应 $<50\text{ms}$）？
4. 在 oCPC / oCPM 模式下，平台提供的“转化成本保障 (Cost Protection / 赔付机制)”在冷启动阶段的触发条件与数学风控？
5. 分析“广义第二价格 (GSP)”下的扣费公式与 eCPM 排序的关系：第 $i$ 名胜出者的真实单次点击扣费 $P_i = \frac{\text{eCPM}_{i+1}}{1000 \times p\text{CTR}_i} + 0.01$ 的推导？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/ecpm-ranking-formula?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ecpm-ranking-formula)

---

## 36. RecSys Metrics: GAUC & NDCG · 推荐离线指标 GAUC 与 NDCG

> RecSys Offline Evaluation Metrics quantify ranking quality, personalization accuracy, and relevance fidelity before pushing model candidates online;

**Interview questions**

1. Derive mathematically why Global AUC suffers cross-user bias and demonstrate cases where Global AUC rises while GAUC drops?
2. Derive NDCG@K, DCG@K, and IDCG@K formulas, explaining the psychological grounding of logarithmic position discount $\log_2(i+1)$?
3. How should users with only positive (all-click) or only negative (zero-click) labels be filtered/handled when aggregating GAUC?
4. Compare MRR (Mean Reciprocal Rank) vs MAP (Mean Average Precision) across single-positive vs multi-positive recommendation tasks?
5. Why is a +0.005 GAUC improvement considered a massive production milestone directly driving millions in revenue?

<details>
<summary>中文题目</summary>

1. 详细推导为什么 Global AUC 会被用户活跃度差异所混淆，并通过反例说明 Global AUC 升高但 GAUC 反而下降的现象？
2. 推导 NDCG@K、DCG@K 与理想排序 IDCG@K 的完整计算公式，并解释对数折损因子 $\log_2(i+1)$ 的位置心理学依据？
3. 当用户在测试集中只有正样本（全点击）或只有负样本（全未点）时，计算 GAUC 时如何处理该用户的 $\text{AUC}_u$？
4. MRR (Mean Reciprocal Rank) 与 MAP (Mean Average Precision) 在单正例与多正例推荐评测中的适用场景？
5. 为什么离线 GAUC 提升 0.5% (5 个千分点) 在工业界被视为极其显著的重大突破，能直接带来可观的在线商业收入增长？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/recsys-offline-metrics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=recsys-offline-metrics)

---

## 37. RecSys Multi-Stage Funnel & 50ms SLA · 推荐多阶段漏斗与 50ms SLA

> The Multi-Stage Recommendation Funnel is a classic industrial cascading architecture that refines global item candidates ($10^7 \sim 10^8$) down to final user recommendations within a strict 50ms Latency SLA;

**Interview questions**

1. Diagram the 4-stage funnel architecture and explain why candidate sizes exhibit exponential decay: $10^8 \to 10^4 \to 10^3 \to 10^2 \to 10^1$?
2. Break down the latency budget across Network RTT, Feature Store fetching, and Model forward passes under a 50ms SLA?
3. Why does the retrieval layer optimize for Recall@K while heavy ranking optimizes for fidelity metrics like GAUC and NDCG?
4. How to implement graceful load shedding (e.g., pruning pre-ranking output from 1000 to 300) during peak traffic spikes to preserve SLA?
5. Analyze the 'retrieval ceiling dictates ranking quality' dependency and how to monitor multi-channel cross-coverage ratios?

<details>
<summary>中文题目</summary>

1. 详细画出 4 阶段推荐漏斗架构图，并解释为何各阶段的候选数量呈现 $10^8 \to 10^4 \to 10^3 \to 10^2 \to 10^1$ 的指数级收敛？
2. 在 50ms 端到端延迟 SLA 下，网络传输 (Network RTT)、特征拉取 (Feature Fetch)、模型前向推理 (Model Forward) 的耗时分解预算？
3. 为什么在召回层追求的是高召回率 (High Recall@K)，而在精排层追求的是高保真排序指标 (NDCG / GAUC)？
4. 当线上偶发突发流量（如双十一大促高峰）导致精排超时时，系统如何执行动态候选集截断（如 $1000 \to 300$）的降级熔断？
5. 推荐系统中“上游召回天花板决定下游精排上限”的漏斗木桶效应与多路召回交叉覆盖率监控？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/recsys-funnel-sla?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=recsys-funnel-sla)

---

## 38. Reranking Diversity & DPP Process · 重排多样性与行列式点过程 DPP

> Determinantal Point Processes (DPP) is a probabilistic diversity-ranking algorithm applied during the final Re-Ranking stage that rigorously balances individual item Quality against global list Diversity;

**Interview questions**

1. Derive the DPP kernel matrix factorization and the greedy MAP inference via Cholesky updates with $O(k^2 N)$ complexity?
2. Compare DPP vs Maximal Marginal Relevance (MMR) in diversity modeling and global relevance trade-offs?
3. Design a hybrid re-ranking pipeline combining hard sliding-window rules (min gap of 3 per category) with soft DPP diversity?
4. How to quantify recommendation diversity using Intra-List Distance (ILD), Catalog Coverage, Gini Index, and Entropy?
5. How to optimize C++ vectorized linear algebra routines for real-time DPP inference within a 5ms SLA?

<details>
<summary>中文题目</summary>

1. 推导 DPP 核矩阵 $L = \text{diag}(q) S \text{diag}(q)$ 与贪心快速选点算法 (Greedy MAP Inference / Cholesky 分解) 的 $O(k^2 N)$ 复杂度？
2. 对比 DPP 与最大边界相关算法 (Maximal Marginal Relevance, MMR) 在多样性建模能力与全局相关性权衡上的差异？
3. 业务规则硬打散（如滑动窗口约束“同一类目商品不得连续出现且间隔至少 3 个” / Sliding Window Rule）与软打散算法的协同？
4. 如何量化评估推荐列表的多样性（如 ILD 列表内距离、Coverage 覆盖率、Gini 坚尼系数、熵）？
5. 在极短的 5ms 重排延迟预算内，如何在 C++ 层面优化 DPP 的矩阵行列式运算以实现实时在线 Serving？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/reranking-diversity-dpp?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=reranking-diversity-dpp)

---

## 39. Risk Shadow Mode & Strategy Canary · 风控影子模式与策略灰度

> Risk Shadow Mode & Canary Policy Deployment provide the mission-critical zero-risk validation framework for releasing high-stakes fraud detection models and security rules into financial production environments;

**Interview questions**

1. Design a zero-overhead traffic mirroring architecture using Envoy/Kafka for asynchronous shadow pipeline evaluation?
2. Explain how Diff Matrix analytics quantify False Positive risk and incremental capture yield during shadow validation?
3. Define the automatic rollback guardrails (payment success rate, customer support ticket spikes, P99 latency spikes) during canary rollouts?
4. How to constrain financial loss limits and risk exposure caps when running live A/B tests on fraud detection policies?
5. Explain dynamic policy hot-reloading without container restarts and request-level policy version tagging for auditability?

<details>
<summary>中文题目</summary>

1. 详细设计基于异步消息队列 (Kafka / Envoy Traffic Shadowing) 的全量流量影子复制与旁路比对架构？
2. 影子模式中的“决策差异比对矩阵 (Diff Matrix: 旧放行但新拦截、旧拦截但新放行)”的统计指标设计与误杀率估算方法？
3. 金丝雀灰度放量过程中的核心护栏指标 (Guardrail Metrics: 支付转化率、客服投诉率、系统 P99 延迟) 与自动熔断回滚阈值？
4. 风控规则“AB 实验与对照分流”中的道德伦理与资金损失控制（为什么实验组放行欺诈会产生真实坏账损失，如何设限？）？
5. 多版本策略引擎的动态配置热加载 (Hot Reloading) 与基于版本号 (Policy Versioning) 的请求级染色与全链路回溯？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/risk-shadow-mode-canary?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=risk-shadow-mode-canary)

---

## 40. Semantic Cache & Request Deduplication · 语义缓存与请求去重

> The Semantic Cache & Request Deduplication Gateway (GPTCache, Redis Semantic Caching) is an intelligent acceleration layer positioned in front of LLM inference and RAG retrieval pipelines;

**Interview questions**

1. Compare Exact String Match vs Semantic Vector Match across cache hit rates, lookup latencies, and semantic hallucination risks?
2. How to rigorously calibrate the similarity threshold $\tau$ to prevent false cache collisions on semantically opposite prompts ('open account' vs 'close account')?
3. Design a Singleflight concurrency barrier preventing Cache Stampedes when thousands of users ask identical breaking queries simultaneously?
4. How to design cascading cache invalidation via metadata reverse indexes when underlying knowledge base documents are updated?
5. Explain cryptographic and metadata-scoped partitioning ensuring Tenant A's private responses never leak to Tenant B's semantic cache hits?

<details>
<summary>中文题目</summary>

1. 详细对比精确匹配缓存 (Exact String Match) 与语义向量缓存 (Semantic Vector Match) 在命中率、延迟与假阳性 (False Positive) 风险上的权衡？
2. 语义相似度阈值 $\tau$（如 $\tau=0.92$ vs $\tau=0.98$）的严谨标定方法：如何避免不同含义但句式相似的问题被错误命中（如“如何开户”与“如何销户”）？
3. 在高并发突发流量下，如何基于 Singleflight (单飞模式) 或 Redis 互斥锁防止缓存击穿 (Cache Stampede / Thundering Herd)？
4. 语义缓存的主动失效与更新策略 (Cache Eviction / Invalidation)：当底层知识库文档更新时，如何通过反向元数据索引级联清除失效缓存项？
5. 多租户语义缓存的安全隔离：如何保证租户 A 的私有问答结果绝对不会被租户 B 的相似 Query 命中并泄露？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/semantic-cache-gateway?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=semantic-cache-gateway)

---

## 41. Semantic Neural Search & Hybrid · 神经语义搜索与双塔/ColBERT 混合

> Semantic Neural Search & Hybrid Retrieval is a next-generation retrieval architecture unifying traditional Sparse Lexical Matching (BM25 Inverted Index) with Dense Deep Semantic Representations and Multi-Vector Late-Interaction (ColBERT);

**Interview questions**

1. Compare Dense Bi-Encoder, Sparse BM25, and Multi-Vector ColBERT across index RAM footprint, query throughput (QPS), and retrieval precision?
2. Why does Reciprocal Rank Fusion (RRF) exhibit superior robustness compared to naive linear score normalizations across divergent score distributions?
3. How to resolve 'Semantic Drift' (e.g. searching for iPhone cases but retrieving the phone itself) via hard entity masking and hybrid filters?
4. How does the SPLADE sparse neural model output dynamically weighted pseudo-terms directly indexable by standard Lucene inverted indices?
5. Design an asynchronous parallel fan-out gateway in Rust/Go with dynamic timeouts, circuit-breaking, and graceful degradation?

<details>
<summary>中文题目</summary>

1. 对比 Dense Bi-Encoder、Sparse BM25 与 Multi-Vector ColBERT 在索引空间占用、检索吞吐 (QPS) 与表达精度上的三维雷达图？
2. 倒数排名融合算法 (Reciprocal Rank Fusion, RRF: $\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$) 为什么比简单的分数线性加权更具鲁棒性？
3. 在电商搜索中，如何解决稠密向量检索带来的“语义漂移（如搜索 iPhone 15 手机壳却召回了 iPhone 15 手机本身）”？
4. SPLADE (Sparse Lexical and Expansion Model) 稀疏神经表征模型如何直接产出兼容标准倒排索引的动态词项权重与扩展？
5. 高并发场景下混合检索网关 (Hybrid Gateway) 的并行异步调用 (Async Fan-out)、超时自动熔断与降级兜底设计？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/semantic-neural-search?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=semantic-neural-search)

---

## 42. SSE Streaming & Backpressure Control · SSE 长连接流式传输与背压控制

> Server-Sent Events (SSE) Streaming & Reactive Backpressure Architecture is the standard high-performance protocol transmitting token-by-token LLM auto-regressive completions to client browsers over persistent unidirectional streams;

**Interview questions**

1. Compare SSE vs WebSocket across connection overhead, bidirectional capabilities, Nginx reverse proxy buffering, and reconnection primitives?
2. Explain why configuring `proxy_buffering off;` in Nginx reverse proxies is strictly mandatory to prevent token buffering in SSE streams?
3. How do async Python generators (AsyncGenerator) in vLLM apply backpressure to pause GPU token scheduling when clients lag?
4. Design Layer-7 load balancing (Envoy / HAProxy) for long-lived SSE connections to prevent connection skew and hot-spotting?
5. How to propagate client disconnect signals via HTTP cancellation tokens to immediately abort active GPU KV-Cache generation?

<details>
<summary>中文题目</summary>

1. 详细对比 SSE (Server-Sent Events) 与 WebSocket 在连接开销、双向通信、代理网关缓冲 (Nginx Buffering) 与断线重连机制上的差异？
2. Nginx 代理层配置：为什么必须显式配置 `proxy_buffering off;` 和 `X-Accel-Buffering: no;` 才能防止网关缓冲大模型流式 Token？
3. 大模型推理服务（如 vLLM / TensorRT-LLM）中的 Token 生成器与异步协程背压：如何通过 AsyncGenerator 队列防止消费者过慢导致的显存/内存泄漏？
4. 长连接连接池管理与负载均衡：如何避免数万个长连接导致特定后端网关节点连接数倾斜 (Connection Skew)？
5. 客户端异常断线检测与推理中断 (Client Disconnect Abort)：当用户主动点击“停止生成”或关闭网页时，如何跨服务通知 GPU 即刻终止该 Request 的 KV Cache 计算？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/sse-streaming-backpressure?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=sse-streaming-backpressure)

---

## 43. YouTube DNN Candidate Generation · YouTube DNN 召回架构

> YouTube DNN Candidate Generation (from Google's landmark 2016 paper) formalizes recommendation recall as an extreme multi-class classification problem over millions of video classes;

**Interview questions**

1. Explain why the 'Example Age' feature eliminates time bias and enables the model to favor newly uploaded fresh content?
2. Why does YouTube DNN enforce asymmetric chronological prediction (predicting next watch from past) rather than symmetric CBOW windows?
3. How does Sampled Softmax approximate full denominator partition functions to enable extreme classification over millions of items?
4. Why does the learned weight matrix of the Softmax layer mathematically equal item embeddings for online ANN retrieval?
5. Trace the evolution of watch history aggregation from simple average pooling to target-attention (DIN/SIM) architectures?

<details>
<summary>中文题目</summary>

1. 详细剖析“Example Age (样本年龄)”特征的设计原理：为什么将样本产生时间距今的时长作为特征输入能完美消除时间偏置 (Time Bias)？
2. 为什么在构建训练样本时采用“非对称预测 (Asymmetric Prediction: 仅用历史预测未来)”而不是类似 Word2Vec 的左右双向预测？
3. Sampled Softmax (如每次采样几千个负例) 如何将 Softmax 分母庞大的全量计算复杂度降为高效的局部负采样？
4. 在 Serving 阶段，Softmax 输出层的权重矩阵为什么能直接无缝等价为物品库的 Item 向量并导入 ANN 索引？
5. 用户历史观看序列的聚合方式：从简单的平均池化 (Average Pooling) 到 DIN / SIM (自注意力加权激活) 的演进动机？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/System_Design/card/youtube-dnn-retrieval?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=youtube-dnn-retrieval)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
