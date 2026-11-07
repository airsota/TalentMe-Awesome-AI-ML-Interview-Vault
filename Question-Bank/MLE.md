# 💻 MLE Role Track · 机器学习工程师 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **25 cards · 125 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. 5 CV Splitting Strategies & Leak Defense · 5 种交叉验证划分与防泄漏隔离

> The 5 Cross-Validation Strategies & Anti-Leakage Isolation framework defines the gold-standard offline validation methodologies guaranteeing zero data leakage;

**Interview questions**

1. Contrast K-Fold, Stratified K-Fold, GroupKFold, and TimeSeriesSplit across data distributions and deployment contexts?
2. Why does standard random K-Fold on temporal data cause severe Lookahead Leakage by predicting the past using future context?
3. Explain how GroupKFold prevents model memorization in medical imaging by isolating all images of a single patient to one fold?
4. How does wrapping transformers and estimators inside Scikit-Learn `Pipeline` objects programmatically prevent data leakage during CV?
5. Explain the architecture of Nested Cross-Validation (Outer 5-fold evaluation + Inner 3-fold hyperparameter search) to prevent hyperparameter optimism bias?

<details>
<summary>中文题目</summary>

1. 详细对比 K-Fold、Stratified K-Fold、GroupKFold 与 TimeSeriesSplit 的切分机制与适用业务场景？
2. 为什么在时间序列数据上如果错误地使用了标准随机 K-Fold，会导致极度严重的“未来信息穿越 (Lookahead Leakage)”？
3. GroupKFold 在医疗诊断（同患者多张 X 光片）与自动驾驶（同路段多帧连续图像）中防止“记忆重合”的机理解析？
4. Scikit-Learn `Pipeline` 对象的架构价值：如何通过 `Pipeline([('scaler', StandardScaler()), ('model', Ridge())])` 自动化杜绝 CV 泄漏？
5. 嵌套交叉验证 (Nested Cross-Validation: 外层 5 折评估泛化，内层 3 折调超参) 在彻底杜绝超参数过拟合中的原理？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-cross-validation-leakage-defense?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-cross-validation-leakage-defense)

---

## 2. 5-Step ML System Design Framework · MLE 系统设计 5 步面试方法论

> The 5-Step ML System Design Interview Framework provides the standardized 45-minute structural roadmap for deconstructing ambiguous business problems (e.g.

**Interview questions**

1. How to structure proactive clarifying questions separating Functional business goals from Non-Functional latency SLAs in the opening 5 minutes?
2. How to rigorously map high-level business goals (Retention, GMV) into concrete ML training objectives and loss functions?
3. How to cleanly organize the dual-track Offline Training and Online Inference pipelines on whiteboard architecture diagrams?
4. How to systematically answer 10x traffic spike follow-ups and zero-history user cold-start challenges?
5. How to proactively articulate architectural trade-offs (compute cost vs ranking accuracy) to secure a 'Strong Hire' recommendation?

<details>
<summary>中文题目</summary>

1. 面试前 5 分钟如何主动提出精准澄清问题：如何区分功能需求 (Functional) 与非功能需求 (Non-Functional: P99 延迟 <50ms、高可用 99.99%)？
2. 如何将抽象业务指标（如用户留存、GMV、满意度）转化为明确的机器学习监督目标（如二分类 pCTR、回归播放时长、多目标 MMoE）？
3. 离线管道 (Offline Batch Pipeline: Spark / Hive) 与在线推理流 (Online Serving: Redis / C++ Engine) 在白板架构图上的标准分层规范？
4. 如何自如应对面试官的压力追问（如“如果 QPS 瞬间翻 10 倍”或“新加入无行为的新用户冷启动该怎么推荐”）？
5. 在系统设计结尾如何主动反思当前架构的 Trade-offs（如简单双塔带来的精度损失 vs 复杂精排带来的算力成本）以拿到顶级评价？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-five-step-system-framework?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-five-step-system-framework)

---

## 3. Bias-Variance Tradeoff & Overfitting · 偏差-方差权衡与过拟合诊断

> The Bias-Variance Tradeoff mathematically decomposes generalization error into three orthogonal components: $\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Variance}(\hat{f}(x)) + \sigma^2$ (where $\sigma^2$ is irreducible noise);

**Interview questions**

1. Derive the mathematical proof of the Bias-Variance-Noise decomposition of Mean Squared Error?
2. Diagram Learning Curves (error vs training size) and identify asymptotic signatures separating high bias from high variance?
3. Contrast L1 (Laplacian prior / sparsity) vs L2 (Gaussian prior / weight decay) through Bayesian and geometric contours?
4. Explain the modern deep learning 'Double Descent' phenomenon where test error decreases again past the interpolation threshold?
5. Explain Early Stopping patience calibration and Exponential Moving Average (EMA) weight smoothing during training?

<details>
<summary>中文题目</summary>

1. 详细推导均方误差的偏差-方差分解公式：$\mathbb{E}[(y - \hat{f})^2] = (\mathbb{E}[\hat{f}] - f)^2 + \mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2] + \sigma^2$？
2. 画出训练集规模与误差之间的“学习曲线 (Learning Curves)”：如何根据曲线渐近线形态区分高偏差与高方差？
3. L1 正则 (Lasso: 产生稀疏解) 与 L2 正则 (Ridge: 权重衰减平滑) 在贝叶斯先验视角 (Laplace vs Gaussian) 下的几何与数学推导？
4. 现代深度学习中的“双重下降现象 (Double Descent: 当模型参数量超过插值阈值后，测试误差再次下降)”的机理解释？
5. 早停策略 (Early Stopping) 的耐心轮数 (Patience) 设置与 Checkpoint 权重平滑 (EMA: Exponential Moving Average)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-bias-variance-tradeoff?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-bias-variance-tradeoff)

---

## 4. Data Quality Threats & Clean Gates · 数据质量四大威胁与清洗管线

> The Data Quality Threats & Automated Cleaning Pipeline acts as the foremost defensive infrastructure preventing 'Garbage In, Garbage Out' contamination in production ML pipelines;

**Interview questions**

1. Explain how Confident Learning estimates joint distribution $P(\tilde{y}, y^*)$ to systematically flag label noise in training sets?
2. Explain MinHash and Locality-Sensitive Hashing (LSH) for high-speed Jaccard similarity deduplication over massive datasets?
3. Compare IQR fencing vs Winsorization in capping extreme numerical outliers without discarding row observations?
4. How to filter synthetic bot clicks from user logs using temporal click variance and request entropy?
5. How to integrate Great Expectations into production data CI/CD pipelines to enforce schema assertions before training jobs start?

<details>
<summary>中文题目</summary>

1. 置信学习 (Confident Learning: Cleanlab 核心算法) 如何通过预测概率矩阵与计数矩阵估计联合概率分布 $P(\tilde{y}, y^*)$ 并自动识别噪声标签？
2. 基于 MinHash 与局部敏感哈希 (LSH) 的海量文本/样本去重算法原理与 Jaccard 相似度快速近似？
3. 异常值处理策略：四分位距 (IQR / Tukey's Fences: $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$) 与缩尾处理 (Winsorization) 的优劣对比？
4. 爬虫与异常机器人流量识别：如何结合 User-Agent 熵、点击间隔方差与 IP 请求频次过滤虚假正负样本？
5. 自动化数据校验工具（如 Great Expectations, Deequ）在 CI/CD 管道中定义数据断言 (Expectation Suites) 的工程实践？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-data-quality-cleaning-gates?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-data-quality-cleaning-gates)

---

## 5. Distributed DDP Bottlenecks & Tuning · 分布式训练通信瓶颈与 DDP 调优

> Distributed Data Parallel (DDP) Communication Bottleneck Debugging & Optimization diagnoses hardware under-utilization, communication stalls, and linear scaling degradation across multi-node GPU clusters;

**Interview questions**

1. Diagram the timing execution of DDP Gradient Bucketing overlapping backward operator computation with NCCL AllReduce communication?
2. Explain why setting `find_unused_parameters=False` eliminates expensive full computation graph traversals in DDP?
3. Derive why Ring-AllReduce total data transferred per GPU is $2 \frac{N-1}{N} S$, proving communication volume is independent of node count $N$?
4. How to tune `num_workers`, `pin_memory=True`, and non-blocking CUDA memory copies to eliminate DataLoader host bottlenecks?
5. What are the systematic diagnostic steps using PyTorch Profiler to isolate communication vs computation GPU kernels?

<details>
<summary>中文题目</summary>

1. 详细画出 PyTorch DDP 内部梯度分桶 (Gradient Bucketing) 与“反向计算与 AllReduce 跨卡通信重叠”的时序执行图？
2. 为什么在没有参数被闲置时，将 `find_unused_parameters=False` 设置为 False 能显著提升 DDP 执行速度（消除全图反向遍历开销）？
3. Ring-AllReduce 算法通信量推导：为什么在 $N$ 张 GPU 传输大小为 $S$ 的梯度张量时，总通信量严格恒定为 $2 \frac{N-1}{N} S$，几乎与卡数 $N$ 无关？
4. 数据加载瓶颈排查：如何通过 `num_workers > 0`、`pin_memory=True` 与异步预加载 (Prefetching) 彻底消除 GPU 等待 CPU 数据的问题？
5. 使用 PyTorch Profiler (结合 Chrome Tracing / TensorBoard) 定位 `ncclKernel_AllReduce` 通信等待时间占比过长的标准调优步骤？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-distributed-communication-bottlenecks?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-distributed-communication-bottlenecks)

---

## 6. End-to-End RecSys Pipeline & SLA · 亿级推荐系统端到端链路与 SLA

> The End-to-End Large-Scale Recommendation Pipeline & SLA Allocation framework structures modern recommendation platforms serving hundreds of millions of DAUs across 10M+ items within a strict 50ms latency budget;

**Interview questions**

1. Diagram the 4-stage recommendation funnel and break down candidate sizes ($10^7 \to 10^4 \to 10^3 \to 10^2 \to 10^1$) and latency budgets?
2. Explain why online DSSM inference only executes the User Tower while the Item Tower is fully pre-computed offline into ANN vector indices?
3. How do multi-task models (MMoE/PLE) jointly optimize CTR, CVR, and dwell time into a unified ranking score in production?
4. How does Determinantal Point Processes (DPP) mathematically balance candidate relevance quality against category diversity in re-ranking?
5. Design dynamic candidate pruning and circuit-breaking fallback strategies when heavy rankers face traffic spikes?

<details>
<summary>中文题目</summary>

1. 详细画出 4 阶段推荐漏斗架构图，并分解各阶段候选规模（$10^7 \to 10^4 \to 10^3 \to 10^2 \to 10^1$）与延迟预算？
2. DSSM 双塔召回模型为什么在线推理时只执行 User 塔前向，而 Item 塔完全采用离线预计算 + 向量索引 (HNSW/IVF)？
3. 多目标精排模型 (MMoE / PLE) 如何同时兼顾点击率 (CTR)、转化率 (CVR) 与播放时长 (Watch Time) 并加权融合成最终排序分？
4. 重排阶段的行列式点过程 (DPP) 算法如何平衡推荐列表的相关度质量与商品类目多样性？
5. 当精排层遭遇流量洪峰导致 P99 延迟超标时，推荐网关的动态候选集截断与服务熔断降级策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-recsys-end-to-end-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-recsys-end-to-end-pipeline)

---

## 7. Ensemble Stacking & Blending · 模型集成 Stacking 与 Blending

> Ensemble Stacking & Blending are meta-learning frameworks combining heterogeneous base models (LightGBM, XGBoost, CatBoost, Deep MLPs, Logistic Regression) with orthogonal inductive biases to minimize variance and push performance boundaries;

**Interview questions**

1. Diagram the 5-fold Stacking OOF meta-feature generation and test set prediction averaging to demonstrate zero target leakage?
2. Compare Stacking vs Blending in sample efficiency, overfitting risks, and implementation complexity?
3. Why do production Stacking architectures mandate simple linear models (Ridge/Logistic Regression) as Level-2 Meta-Learners?
4. Explain why ensemble diversity across heterogeneous model families (GBDT + MLP + Factorization Machines) yields higher lift than homogeneous variants?
5. How to distill a slow 10-model Stacking ensemble into a single compact, high-throughput student network for low-latency production serving?

<details>
<summary>中文题目</summary>

1. 详细画出 5 折交叉验证 Stacking 中 OOF (Out-of-Fold) 元特征生成与测试集 $K$ 个基模型预测均值融合的无泄漏数据流图？
2. 对比 Stacking 与 Blending 在数据利用率、防过拟合能力与工程实现复杂度上的根本权衡？
3. 为什么在 Stacking 的 Level-2 元模型 (Meta-Learner) 中通常推荐使用简单的线性模型 (Logistic Regression / Ridge) 而非复杂的深层网络？
4. 基模型多样性 (Diversity) 假说：为什么融合 3 个异构模型 (如 GBDT + ResNet + KNN) 的效果通常远胜于融合 3 个仅调了参数的同构 GBDT？
5. 工业在线部署时的延迟代价与模型蒸馏 (Distillation)：如何将复杂的 10 个 Stacking 模型集成蒸馏回单个紧凑的高吞吐深度网络？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-ensemble-stacking-blending?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-ensemble-stacking-blending)

---

## 8. Feature Selection, Permutation & SHAP · 特征选择与置换重要性 / SHAP

> Feature Selection Taxonomy, Permutation Importance & SHAP Values provide the systematic framework to compress high-dimensional feature spaces, purge collinear noise, and attribute individual feature contributions via game theory;

**Interview questions**

1. Derive Shapley value axiomatic properties (Efficiency, Symmetry, Dummy, Additivity) and explain its marginal contribution formulation?
2. Explain why GBDT Split Count importance is biased towards continuous features with many unique values, and how TreeSHAP corrects this?
3. Explain how multi-collinearity distorts Permutation Importance and how Hierarchical Clustering Ward linkage groups redundant features?
4. How does TreeSHAP optimize exponential Shapley calculation down to polynomial time $O(T L D^2)$ using decision tree path structures?
5. Explain the algorithmic workflow of RFECV (Recursive Feature Elimination with Cross-Validation) in selecting optimal feature subsets?

<details>
<summary>中文题目</summary>

1. 推导合作博弈论 Shapley 值的经典公式 $\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$ 的公理化性质（效率性、对称性、虚拟性、加性）？
2. 为什么在 GBDT 树模型中，直接使用默认的 `feature_importances_` (Split Count) 会对高基数浮点数产生严重的虚假偏好？
3. 置换特征重要性 (Permutation Importance) 在遇到高度共线性特征（Collinear Features: 如两列特征相关度达 0.99）时的失真陷阱与聚类排查？
4. TreeSHAP 算法如何利用决策树的局部路由结构将 Shapley 值的指数级时间复杂度 $O(2^{|F|})$ 压缩至多项式时间 $O(T L D^2)$？
5. 递归特征消除 (RFE: Recursive Feature Elimination) 配合交叉验证 (RFECV) 在寻找全局最优特征数量 $K^*$ 时的算法流转？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-feature-selection-shap?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-feature-selection-shap)

---

## 9. Feature Store Consistency & No-Leak · Feature Store 一致性与穿越规避

> The Feature Store Online/Offline Consistency & Point-in-Time Anti-Leakage Architecture standardizes feature engineering across training and inference;

**Interview questions**

1. Explain how Point-in-Time (AS-OF) Joins match event observation timestamps against feature changelog snapshots in Spark/Flink?
2. Design a real-time streaming feature pipeline using Kafka, Flink, and Redis for 5-minute sliding window user clicks?
3. Design default imputation and fallback policies when the online Redis feature store encounters latency spikes or missing user keys?
4. How to detect Train-Serving Skew by logging inference feature snapshots to S3 and comparing them with offline training features?
5. Explain feature schema versioning and deprecation lifecycles enabling zero-downtime hot upgrades in production feature stores?

<details>
<summary>中文题目</summary>

1. 详细剖析 Point-in-Time Join (AS-OF Join) 的算法实现：如何在 Spark/Flink 中通过事件时间戳与特征变更快照实现无泄漏拼接？
2. 基于 Kafka + Flink + Redis 构建秒级实时流式特征计算管道（如滑动 5 分钟窗口用户点击数与实时类目偏好）的架构？
3. 在线特征缺失与默认值兜底：当 Redis 发生网络超时 (>5ms) 或新用户无实时特征时，Feature Store 的优雅降级与默认值设计？
4. 如何检测与排查“训练-服务偏斜 (Train-Serving Skew)”：如何通过请求日志采样与离线特征比对自动化告警？
5. 特征版本管理 (Feature Versioning) 与生命周期管理：当旧特征废弃或新特征上线时，如何在不停机的前提下实现特征热升级？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-feature-store-point-in-time?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-feature-store-point-in-time)

---

## 10. Gradient Clipping & NaN Loss Debug · 梯度爆炸排查与 NaN 损失定位

> Gradient Clipping & NaN Loss Debugging provides the systematic troubleshooting framework to detect, isolate, and remediate training instabilities, exploding gradients, and numerical divergence in deep neural networks and LLMs;

**Interview questions**

1. Derive Global Norm Gradient Clipping and explain why it preserves gradient direction vectors unlike element-wise value clipping?
2. Explain the state-machine rules of Dynamic Loss Scaling in FP16 mixed-precision training?
3. Why does BF16's 8-bit exponent match FP32's dynamic range, eliminating the need for Loss Scalers required by FP16's 5-bit exponent?
4. List the 5 most common mathematical operations that trigger NaNs in PyTorch computation graphs and their specific fixes?
5. How to monitor the Grad-Norm-to-Weight-Norm ratio as a vital diagnostic metric for optimizer learning rate health?

<details>
<summary>中文题目</summary>

1. 推导全局范数梯度裁剪 (Gradient Clipping by Global Norm) 的数学公式，为什么按全局范数缩放比按逐参数裁剪 (Clip by Value) 优越？
2. 动态损失缩放 (Dynamic Loss Scaling) 的状态机原理：连续 2000 步无溢出则 $S \leftarrow 2S$，一旦检测到 Inf/NaN 则跳过梯度更新并 $S \leftarrow S/2$？
3. BF16 (Bfloat16) 相比 FP16 为什么从硬件根源上彻底消除了动态损失缩放 (Loss Scaler) 的需求？
4. 常见产生 `NaN` 的五大底层数学陷阱（对 0 取对数 $\ln(0)$、除以 0、负数开方、Softmax 未平移大数上溢、LayerNorm 方差下溢 $\frac{1}{\sqrt{\sigma^2 + \epsilon}}$）的排查？
5. 在大规模分布式集群中，如何利用 TensorBoard / WandB 实时监控梯度范数 (Grad Norm) 与权重范数 (Weight Norm) 的比值（学习率健康度指标）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-gradient-clipping-nan-loss?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-gradient-clipping-nan-loss)

---

## 11. Live Coding: Logistic Regression SGD · 手写逻辑回归梯度下降与 L2

> Live Coding Logistic Regression with L2 Regularization & Mini-Batch SGD in Pure Numpy assesses an ML engineer's foundational mathematical derivations, Sigmoid activations, cross-entropy gradient calculus, and matrix broadcasting;

**Interview questions**

1. Write the complete Pure Numpy `LogisticRegression` class with `fit(X, y)`, `predict_proba(X)`, and vectorized L2 regularization?
2. Derive the log-likelihood cross-entropy loss and its parameter gradient $\nabla_w = \frac{1}{N} X^T (\hat{y} - y)$ from Maximum Likelihood principles?
3. Prove that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and explain its application in backpropagation chain rule simplifications?
4. Compare Newton-Raphson (IRLS) second-order quadratic convergence against first-order SGD in compute complexity ($O(d^3)$ Hessian inverse)?
5. Why is Feature Scaling strictly required for Logistic Regression, and how do elongated elliptical contours slow down gradient descent?

<details>
<summary>中文题目</summary>

1. 现场手写包含 `fit(X, y)`、`predict_proba(X)` 与 L2 正则化的完整的 `LogisticRegression` Python Class？
2. 从极大似然估计 (MLE) 完整推导逻辑回归对数损失函数及其权重梯度 $\nabla_w \mathcal{L} = \frac{1}{N} X^T (\hat{y} - y)$ 的全过程？
3. Sigmoid 导数公式 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ 的证明及其在反向传播链式法则中的应用？
4. 牛顿法 (Newton-Raphson / IRLS: 迭代加权最小二乘法) 相比一阶梯度下降在收敛速度（二阶超线性收敛）与计算复杂度 ($O(d^3)$ 求逆) 上的权衡？
5. 为什么逻辑回归必须配合特征标准化/归一化 (Feature Scaling)？如果不做标准化，椭圆损失等高线如何严重拖慢梯度下降收敛？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-coding-logistic-regression-sgd?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-coding-logistic-regression-sgd)

---

## 12. Live Coding: Multi-Head Self-Attention · 手撕 Multi-Head Self-Attention

> Live Coding Multi-Head Self-Attention (in zero-dependency Pure Numpy or PyTorch) is the premier benchmark live-coding problem in MLE senior interviews;

**Interview questions**

1. Write the complete Pure Numpy implementation of Scaled Dot-Product Attention, annotating tensor shapes at every step?
2. Mathematically prove why scaling by $\sqrt{d_k}$ is mandatory by showing that the variance of $q^T k$ equals $d_k$ under i.i.d. standard normal inputs?
3. Explain how causal masking constructs upper-triangular $-\infty$ masks and why it maps to exact zeros after Softmax?
4. Compare `torch.einsum` vs `torch.matmul` in implementing multi-head attention matrix multiplications?
5. Explain how FlashAttention's tiling and Online Softmax algorithms eliminate the $O(N^2)$ attention matrix HBM roundtrips?

<details>
<summary>中文题目</summary>

1. 现场写出包含 Batch 维度的 Scaled Dot-Product Attention 核心 Pure Numpy 代码，并标注每一步的 Tensor Shape？
2. 数学推导：为什么必须除以缩放因子 $\sqrt{d_k}$（证明当 $q_i, k_i \sim \mathcal{N}(0, 1)$ 独立同分布时，内积点积的方差为 $d_k$）？
3. 手写因果自回归掩码 (Causal Mask: `np.triu(np.ones(...), k=1)`) 的生成与为什么掩码值要填充为 `-1e9` 或 `-np.inf`？
4. 对比 `torch.einsum('bhqd,bhkd->bhqk', q, k)` 与标准 `torch.matmul(q, k.transpose(-2, -1))` 的可读性与底层算子执行？
5. FlashAttention 核心优化原理手撕阐述：如何通过 Tiling 分块计算与 Online Softmax 消除中间注意力矩阵 $O(N^2)$ 的 HBM 读写？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-coding-self-attention?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-coding-self-attention)

---

## 13. Live Coding: NMS & IoU Matrix · 手写目标检测 NMS 与 IoU 矩阵

> Live Coding Non-Maximum Suppression (NMS) & Intersection over Union (IoU) represents the quintessential computer vision coding problem testing bounding-box coordinate geometry and greedy filtering algorithms;

**Interview questions**

1. Write the complete Pure Numpy implementation of vectorized IoU and greedy NMS from scratch?
2. Explain why pixel width calculations add 1 ($w = x_2 - x_1 + 1$) for discrete raster grids vs continuous float coordinates?
3. Explain how Soft-NMS uses continuous Gaussian score decay to retain heavily occluded neighboring objects that hard NMS erroneously deletes?
4. Explain the coordinate-offset trick enabling multi-class batched NMS execution in a single vector operation?
5. How does DETR's bipartite Hungarian matching loss eliminate the need for heuristic NMS post-processing entirely?

<details>
<summary>中文题目</summary>

1. 现场手写包含向量化广播计算的 `compute_iou(boxes1, boxes2)` 与标准 `nms(boxes, scores, iou_threshold)` Pure Numpy 代码？
2. 边界框坐标边界处理：为什么在计算像素宽度与高度时需要加 1（即 $w = x_2 - x_1 + 1$）？在连续浮点坐标与离散像素坐标下的差异？
3. Soft-NMS (软 NMS) 算法的核心改进：为什么采用高斯连续衰减打分 $s_i = s_i e^{-\frac{\text{iou}^2}{\sigma}}$ 替代粗暴直接删除框？
4. Batched NMS (多类别 NMS): 如何通过坐标偏移技巧（如给每个类别 ID 赋予巨大的坐标偏移 $x' = x + \text{cls\_id} \times 4096$）在单次 NMS 调用中实现跨类别独立抑制？
5. 从启发式 NMS 到端到端无 NMS 架构演进：DETR (基于双边匹配匈牙利算法 Hungarian Loss) 如何从根本上消除后处理 NMS？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-coding-nms-iou?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-coding-nms-iou)

---

## 14. Live Coding: Numerically Safe Softmax · 手写 Softmax 防溢出与 Cross-Entropy

> Numerically Stable Softmax & Cross-Entropy is a classic coding problem testing an engineer's mastery over IEEE 754 floating-point limits and numerical stability tricks;

**Interview questions**

1. Write the complete Pure Numpy implementation of numerically stable Softmax supporting arbitrary batch axes?
2. Mathematically prove the translation invariance of the Softmax operator for any scalar constant $c$?
3. Why does evaluating $\log(\text{softmax}(z))$ separately cause underflow, and how does the Log-Sum-Exp (LSE) trick resolve it?
4. Derive the asymptotic output distribution when temperature $\tau \to 0$ (argmax one-hot) and $\tau \to \infty$ (uniform distribution)?
5. Why does PyTorch's `CrossEntropyLoss` fuse LogSoftmax and NLLLoss inside a single fused CUDA kernel for numerical stability and speed?

<details>
<summary>中文题目</summary>

1. 现场写出包含 2D 矩阵按指定轴 (axis=-1) 广播的数值稳定 Safe Softmax Pure Numpy 代码？
2. 数学证明 Softmax 的平移不变性：$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} e^{-c}}{\sum_j e^{z_j} e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$？
3. 为什么在计算交叉熵损失时，直接算 $\log(\text{softmax}(z))$ 依然会发生下溢，而必须使用 Log-Sum-Exp (LSE) 技巧：$\log(\sum e^{z_j}) = c + \log(\sum e^{z_j - c})$？
4. Softmax 带温度系数 $\tau$ 的形式 $\frac{e^{z_i / \tau}}{\sum e^{z_j / \tau}}$：当 $\tau \to 0$ 与 $\tau \to \infty$ 时的极限分布与输出熵推导？
5. PyTorch 内部 `torch.nn.CrossEntropyLoss` 为什么将 LogSoftmax 与 NLLLoss 融合在一个 C++ CUDA Kernel 中执行？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-coding-softmax-stability?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-coding-softmax-stability)

---

## 15. Live Coding: Vectorized K-Means · 手写 K-Means 向量化聚类迭代

> Live Coding Vectorized K-Means (Pure Numpy) evaluates an ML engineer's mastery over unsupervised Expectation-Maximization (EM) iterations and vectorized NumPy broadcasting;

**Interview questions**

1. Write the complete zero-loop vectorized distance matrix computation in Pure Numpy using Euclidean expansions?
2. Explain K-Means++ probabilistic initialization $P(x) \propto D(x)^2$ and how it mathematically prevents poor local minima?
3. How should K-Means handle empty clusters (clusters with zero assigned points) gracefully during M-step updates?
4. Prove that K-Means iterations monotonically decrease the Within-Cluster Sum of Squares (WCSS) objective function?
5. Explain how Mini-Batch K-Means samples stochastic batches to scale clustering across millions of records without OOM?

<details>
<summary>中文题目</summary>

1. 现场写出利用 $\|x - c\|^2 = \|x\|^2 + \|c\|^2 - 2 x c^T$ 矩阵展开实现完全无循环向量化计算距离矩阵的 Pure Numpy 代码？
2. K-Means++ 初始化算法的数学逻辑：为什么按照样本到最近已有质心距离的平方概率 $P(x) = \frac{D(x)^2}{\sum D(x')^2}$ 采样能极大改善局部最优？
3. 当某个簇在迭代过程中发生“空簇 (Empty Cluster: 没有样本被分配给该质心)”时，代码中的优雅容错与重新随机初始化处理？
4. K-Means 目标函数（畸变程度 Inertia / 组内平方和 WCSS）的单调下降与局部收敛性数学证明？
5. Mini-Batch K-Means 算法在亿级海量数据下的流式梯度更新近似与显存节约原理？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-coding-kmeans-clustering?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-coding-kmeans-clustering)

---

## 16. Loss Function Taxonomy & Gradients · 常见损失函数选型与梯度特性

> The Loss Function Taxonomy & Gradient Dynamics framework defines the formal mathematical objectives guiding gradient optimization across regression, classification, ranking, and metric learning;

**Interview questions**

1. Derive the gradient of Softmax Cross-Entropy loss with respect to logits $z_i$: $\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$?
2. Compare Huber Loss vs Smooth L1 Loss in formulation, outlier robustness, and first-derivative continuity?
3. Explain how Focal Loss's modulating factor $(1 - p_t)^\gamma$ scales down gradients from well-classified easy samples?
4. Analyze the role of temperature hyperparameter $\tau$ in InfoNCE contrastive loss in balancing hard negative mining against collapse?
5. Explain how Label Smoothing regularizes cross-entropy by softening one-hot targets and preventing logit overconfidence?

<details>
<summary>中文题目</summary>

1. 推导交叉熵损失函数对 Softmax 输入 Logits $z_i$ 的导数：$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$，并解释其简洁的“预测概率减去真实标签”物理意义？
2. 对比 Huber Loss 与 Smooth L1 Loss 在目标检测与回归任务中的数学公式与一阶导数连续性？
3. Focal Loss 中调节因子 $(1 - p_t)^\gamma$ 如何动态抑制已分类良好样本的梯度（当 $p_t=0.99$ 时梯度衰减 1000 倍）？
4. InfoNCE 损失函数中温度系数 $\tau$ (Temperature) 的作用：如何平衡难负例区分度与特征崩塌 (Representation Collapse)？
5. 标签平滑 (Label Smoothing Cross-Entropy: $y_{\text{smooth}} = (1 - \epsilon) y + \frac{\epsilon}{K}$) 在防止模型过度自信与过拟合中的数学机理？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-loss-function-taxonomy?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-loss-function-taxonomy)

---

## 17. LR Warmup & Cosine Annealing Tuning · 学习率 Warmup 与退火调优排障

> Learning Rate Warmup & Cosine Annealing Dynamic Scheduling governs the temporal trajectory of optimization step sizes in deep neural networks and LLMs to ensure early stability and late-stage convergence into flat minima;

**Interview questions**

1. Derive the Cosine Annealing schedule equation and explain why smooth half-cosine decay favors flat minima over abrupt step decays?
2. Why is Linear Warmup mandatory with AdamW (explaining uncalibrated second-moment variance $v_t$ in early iterations)?
3. Explain the theoretical justification and breakdown limits of the Linear Scaling Rule (scaling learning rate proportionally with batch size)?
4. Explain Leslie Smith's 1Cycle policy and how inverse momentum coupling achieves Super-Convergence in fewer epochs?
5. How should learning rates be rewarmed and decayed during Continual Pre-training on new domain corpora?

<details>
<summary>中文题目</summary>

1. 推导 Cosine Annealing 学习率更新公式，并解释为什么余弦衰减在训练中后期的平滑过渡比阶梯衰减 (Step Decay) 更有利于寻找平坦极小值 (Flat Minima)？
2. 为什么在大模型使用 AdamW 优化器时，Linear Warmup 是绝对不可或缺的（解释二阶矩 $v_t$ 刚开始时的方差未校准风险）？
3. 线性缩放法则 (Linear Scaling Rule: 当 Batch Size 扩大 $k$ 倍时，学习率应扩大 $k$ 倍，即 $\eta = k \cdot \eta_0$) 的理论依据与失效边界？
4. 1Cycle 学习率调度策略 (One-Cycle Policy: 学习率上升的同时动量下降) 在快速收敛 (Super-Convergence) 中的应用机制？
5. 持续预训练 (Continual Pre-training) 中的学习率重热 (Learning Rate Rewarming) 与学习率下限 $\eta_{\min}$（通常取 $\eta_{\max}$ 的 $10\%$）的设计？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-learning-rate-schedule-warmup?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-learning-rate-schedule-warmup)

---

## 18. Missing Value Imputation & MICE · 缺失值填充策略对比与 MICE

> Missing Value Imputation Strategies & MICE (Multiple Imputation by Chained Equations) establish the mathematical decision hierarchy treating missing data based on Rubin's three foundational mechanisms: 1) MCAR (Missing Completely at Random);

**Interview questions**

1. Contrast MCAR, MAR, and MNAR missing mechanisms, explaining why Missingness Indicators are mandatory under MNAR?
2. Detail the step-by-step round-robin algorithm of MICE and how it handles mixed continuous and categorical feature types?
3. Explain how gradient boosting trees (XGBoost/LightGBM) compute optimal default split directions for missing values at each node?
4. Why does time-series imputation prohibit Backward Fill and mandate Forward Fill (LOCF) to eliminate future lookahead leakage?
5. How to bundle trained imputation transformers into serialization pipelines ensuring online features inherit train-set parameters?

<details>
<summary>中文题目</summary>

1. 详细对比 MCAR、MAR 与 MNAR 三种缺失机制，为什么在 MNAR 场景下添加“缺失指示变量 (Missing Indicator Feature)”是关键提分技巧？
2. 多重链式方程插补 (MICE / Iterative Imputer) 的多轮迭代收敛算法步骤与如何处理多变量混合类型（连续 vs 类别）？
3. LightGBM / XGBoost 在树分裂时如何自动寻找缺失值的最优默认分支 (Default Split Direction)？
4. 时序数据缺失值填充：为什么严禁使用全局均值或后向填充 (Backward Fill)，而必须使用前向填充 (Forward Fill / LOCF) 或时序插值以防止穿越？
5. 在生产环境部署时，训练集拟合的插补器对象 (Imputer Fitted on Train) 如何无缝保存为 Pipeline 并在在线 Serving 时直接复用？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-missing-value-imputation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-missing-value-imputation)

---

## 19. Model Compression, Pruning & Quant · 模型压缩、剪枝与量化基础

> Model Compression, Pruning & Low-Bit Quantization optimize neural networks for deployment onto edge hardware or high-throughput cloud inference clusters by slashing memory footprints and memory-bandwidth bottlenecks;

**Interview questions**

1. Derive the mathematical scale factor $S$ and zero-point $Z$ formulas in uniform asymmetric affine quantization?
2. Explain why QAT utilizes FakeQuantize nodes in the forward pass and Straight-Through Estimators (STE) in the backward pass to bypass non-differentiable rounding?
3. Why does structured channel/head pruning achieve direct hardware speedups while unstructured sparse matrices require specialized sparse Tensor Cores?
4. Explain how high distillation temperatures $T$ soften output logit entropy to reveal dark knowledge across non-target classes?
5. Explain how SmoothQuant mathematically migrates activation quantization difficulty onto weights and how AWQ protects salient weight channels?

<details>
<summary>中文题目</summary>

1. 推导仿射量化 (Affine/Asymmetric Quantization) 中缩放因子 $S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$ 与零点偏移 $Z = \text{round}\left(\frac{-x_{\min}}{S}\right) + q_{\min}$ 的数学公式？
2. 详细对比训练后量化 (PTQ) 与量化感知训练 (QAT): 为什么 QAT 在前向传播时使用伪量化 (FakeQuantize) 并在反向传播时使用直通估计器 (STE: Straight-Through Estimator)？
3. 结构化剪枝 (Structured Pruning: 剪除整个卷积核/注意力头) 与非结构化剪枝 (Unstructured Pruning: 随机零散权重稀疏) 在硬件加速器上的真实执行效率差异？
4. 知识蒸馏中温度系数 $T$ (Temperature) 的作用：为什么高温 $T$ 能放大负标签之间的隐式相似度信息 (Dark Knowledge)？
5. 大模型前沿权重激活量化算法：AWQ (基于激活感知的通道加权保护) 与 SmoothQuant (将激活值量化难度数学等价迁移到权重) 的核心原理？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-quantization-pruning-basics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-quantization-pruning-basics)

---

## 20. Offline-Online Eval Gap & A/B Testing · 离线线上指标鸿沟与 A/B 验证

> The Offline-Online Metric Gap Root-Cause & A/B Validation Architecture addresses the perennial industry dilemma where offline AUC improvements fail to translate into online business gains (revenue, GMV, retention);

**Interview questions**

1. List the systematic 5-point root-cause diagnostic checklist when offline AUC gains fail to yield online business improvements?
2. Explain why GAUC (Group AUC) must replace Global AUC to eliminate cross-user activity disparities in recommendation evaluation?
3. Explain how Inverse Propensity Scoring (IPS) unbiases offline evaluation against historical logging policy selection bias?
4. Why does Team Draft Interleaving achieve statistical significance with 1% of the sample size and 1/10 the duration of standard A/B tests?
5. How to design multi-week A/B experimentation windows to isolate transient Novelty Effects and capture true long-term retention?

<details>
<summary>中文题目</summary>

1. 系统化列出导致“离线 AUC 涨但线上业务指标跌”的五大根因排查清单与针对性解决方案？
2. 为什么在推荐系统中必须使用 GAUC (Group AUC: 按用户分组加权 AUC) 替代全局 AUC 以消除跨用户活跃度偏置？
3. 反事实离线评估 (Counterfactual Offline Evaluation) 中的逆倾向得分加权 (Inverse Propensity Scoring, IPS: $\hat{R}_{\text{IPS}} = \frac{1}{N} \sum \frac{Y_i P_{\text{new}}(A_i|X_i)}{P_{\text{old}}(A_i|X_i)}$) 原理？
4. Interleaving (交替实验) 相比标准 A/B 测试在快速筛选淘汰弱模型中的样本量与时间效率优势？
5. 新颖性效应 (Novelty Effect) 与慢炖效应 (Slow Cooker Effect) 在长期 A/B 实验中的观测周期设定与指标平稳性判断？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-offline-online-eval-gap?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-offline-online-eval-gap)

---

## 21. OOF Target Encoding & Anti-Leakage · OOF 目标编码与防目标穿越机制

> Out-of-Fold (OOF) Target Encoding is the gold-standard dense representation method for high-cardinality categorical features (ZIP codes, merchant IDs, device models with $10^4+$ categories) engineered with strict anti-leakage invariants;

**Interview questions**

1. Write the complete Pure Numpy/Pandas implementation of 5-fold OOF Target Encoding with Bayesian smoothing?
2. Mathematically explain why naive target encoding causes target leakage by contaminating feature values with the row's own label?
3. Explain how CatBoost's Ordered Target Encoding calculates cumulative online historical statistics to eliminate leakage naturally?
4. How does adding Gaussian noise/jittering to target-encoded features regularize trees against memorizing rare categories?
5. How should target-encoded transformers handle unseen categorical levels during online inference using global prior baselines?

<details>
<summary>中文题目</summary>

1. 现场手写包含 5 折交叉验证 OOF 切分、贝叶斯平滑权重 $\lambda(n) = \frac{1}{1 + e^{-(n - k) / f}}$ 与测试集映射的 Pure Numpy / Pandas 代码？
2. 详细剖析为什么朴素目标编码 (Naive Target Encoding) 会造成“目标泄漏 (Target Leakage)”并在训练集上产生虚高的假完美性能？
3. CatBoost 原生目标编码 (Ordered Target Encoding) 如何基于时间序列/随机排列在线累计均值实现完全零泄露计算？
4. 向目标编码中添加高斯随机噪声 (Additive Gaussian Noise / Gaussian Jittering: $\hat{x} + \epsilon$) 进一步防止过拟合的调参策略？
5. 冷启动与未见类别 (Unseen Categories in Test): 当测试集出现训练集从未见过的全新商户 ID 时，如何通过全局先验均值 $\bar{y}_{\text{global}}$ 进行兜底？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-oof-target-encoding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-oof-target-encoding)

---

## 22. Optimizer Convergence & Momentum · 优化器收敛性与动量选型准则

> Optimizer Convergence Dynamics & Momentum Selection Rules establishes the algorithmic decision matrix choosing between first-order momentum and adaptive learning rate optimizers based on model topology (CNNs, Transformers, Sparse Embeddings);

**Interview questions**

1. Derive the mathematical difference between original Adam and AdamW in weight decay updates, explaining why L2 regularization fails in Adam?
2. Derive the initial step bias correction formulas $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ in Adam/AdamW?
3. Why does SGD with Momentum often generalize better to flat minima in Computer Vision compared to adaptive optimizers?
4. Explain memory savings of Adafactor (low-rank factorization) and the sign-gradient operation of the Lion optimizer?
5. How to tune $\beta_1, \beta_2,$ and $\epsilon$ to stabilize training against loss spikes and gradient oscillations in mixed precision?

<details>
<summary>中文题目</summary>

1. 详细对比经典 Adam 与 AdamW 在权重衰减 (Weight Decay) 更新公式上的数学差异，解释为什么 Adam 的 L2 正则失效？
2. 一阶动量 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ 与二阶动量 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ 在训练初期的偏差校正 (Bias Correction) 推导？
3. 为什么在卷积神经网络 (ResNet/ConvNeXt) 上 SGD+Momentum 的最终测试集泛化性能通常优于自适应类优化器？
4. Lion (EvoLved Sign Momentum) 优化器与 Adafactor (行/列低秩矩阵分解削减显存) 在大模型显存节约中的机制？
5. 当损失震荡或出现梯度尖峰时，优化器超参数（$\beta_1=0.9, \beta_2=0.95/0.98, \epsilon=10^{-8}\sim 10^{-6}$）的避坑调优策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-optimizer-convergence-rules?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-optimizer-convergence-rules)

---

## 23. PR-AUC vs ROC & Threshold Tuning · 不平衡 PR-AUC 与截断点标定

> PR-AUC Superiority in Imbalance & Optimal Threshold Calibration formalizes classification evaluation and operating point selection under severe positive rarity (e.g.

**Interview questions**

1. Mathematically prove why ROC curves are invariant to class prevalence shifts while PR curves reflect true positive ratio changes?
2. Contrast random baseline performance: ROC-AUC is always 0.5, whereas PR-AUC baseline equals the true positive class prevalence $P/(P+N)$?
3. Compare threshold selection criteria: Max F1 vs Youden's Index vs Fixed-Specificity constraint (e.g. FPR $\le 0.001$)?
4. How to use Isotonic Regression to calibrate raw model scores into true empirical probabilities for downstream dollar-cost decision making?
5. Why does linear trapezoidal integration overestimate PR-AUC, mandating interpolated Average Precision (AP) sums?

<details>
<summary>中文题目</summary>

1. 数学证明为什么当负样本数量 $N \to \infty$ 时，ROC 曲线保持不变，而 PR 曲线会随正负样本比例变化而剧烈移动？
2. 基准随机分类器 (Baseline Random Classifier) 在 ROC 空间（$\text{AUC}=0.5$）与 PR 空间（$\text{PR-AUC} = P/(P+N) = \text{Prior}$）的基线差异？
3. 最优分类阈值选型策略：最大化 F1 阈值、Youden's J 统计量（$\max [\text{TPR} - \text{FPR}]$）与固定特异度（如保证 Specificity $\ge 99.9\%$ 时选最高 Recall）的对比？
4. 在生产中如何通过保序回归 (Isotonic Regression) 概率校准将模型输出的相对排序 Logits 转换为真实的无偏后验概率 $P(y=1|x)$？
5. 为什么在计算 PR-AUC 时使用梯形法则 (Trapezoidal Rule) 会高估面积，而必须使用非插值的平均精度 (Average Precision / AP) 积分？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-pr-auc-imbalance-thresholding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-pr-auc-imbalance-thresholding)

---

## 24. Production PSI Drift Monitoring · 生产环境 PSI 特征漂移监控

> The Population Stability Index (PSI) & Feature Drift Monitoring System is the industry standard statistical metric in MLOps monitoring production distribution shifts, silent data pipeline failures, and model degradation without requiring immediate ground-truth…

**Interview questions**

1. Derive the mathematical equivalence between PSI and symmetric Kullback-Leibler (KL) divergence: $D_{KL}(T \| B) + D_{KL}(B \| T)$?
2. Write the complete Pure Numpy implementation computing PSI using equal-frequency quantile binning?
3. How does Laplace smoothing handle zero-frequency bins ($T_i = 0$) to prevent division-by-zero or $\ln(0)$ crashes?
4. Contrast Covariate Shift ($P(x)$ changes, $P(y|x)$ constant) vs Concept Drift ($P(y|x)$ changes), detailing re-weighting vs re-training remedies?
5. Compare PSI against Kolmogorov-Smirnov (KS) test, Chi-Square, and Wasserstein Distance across continuous vs discrete feature drift?

<details>
<summary>中文题目</summary>

1. 详细推导 PSI 计算公式 $\text{PSI} = \sum_{i=1}^k (T_i - B_i) \ln\left(\frac{T_i}{B_i}\right)$ 与相对熵 (KL 散度: $D_{KL}(T \| B) + D_{KL}(B \| T)$) 的对称化数学关系？
2. 现场写出使用 10 等频分箱 (Quantile Binning) 计算单特征与预测分 PSI 的 Pure Numpy / Python 代码？
3. 当某个分箱在线上样本中出现 0 占比（$T_i = 0$ 导致 $\ln(0)$ 未定义）时的平滑处理方法（如 Laplace 平滑加微小常数 $\epsilon=10^{-4}$）？
4. 概念漂移 (Concept Drift: $P(y|x)$ 改变) 与协变量偏移 (Covariate Shift: $P(x)$ 改变但 $P(y|x)$ 不变) 的本质区别与应对方案？
5. 除了 PSI 之外，KS 检验 (Kolmogorov-Smirnov)、卡方检验 (Chi-Square) 与 Wasserstein 距离 (Earth Mover's Distance) 在漂移检测中的适用场景？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-psi-feature-drift-monitoring?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-psi-feature-drift-monitoring)

---

## 25. Real-Time Risk & Delayed Feedback · 实时风控反欺诈与延迟反馈修正

> The Real-Time Risk Anti-Fraud Architecture & Delayed Feedback Correction framework defends financial platforms against adversarial syndicates, account takeover, and promo abuse under extreme sample skew and label feedback latency;

**Interview questions**

1. Derive the Maximum Likelihood expectation and EM formulation of the Delayed Feedback Model (DFM)?
2. Explain the orchestration sequence between in-memory deterministic rule engines and ML scoring models within a 10ms SLA?
3. How to combine Focal Loss and Cost-Sensitive Matrices to calibrate optimal decision thresholds under $<0.1\%$ fraud prevalence?
4. How does Graph Risk leverage heterogeneous entity graphs (Device-IP-Card) and Louvain community detection to bust fraud syndicates?
5. Explain how Shadow Mode asynchronously mirrors live traffic to evaluate false positive risks before canary ramp-up?

<details>
<summary>中文题目</summary>

1. 详细推导延迟反馈模型 (Delayed Feedback Model, DFM) 的最大似然期望对数似然函数与 EM 优化求解过程？
2. 实时风控决策全链路中规则引擎（硬规则拦截）与机器学习模型（软评分打分）在线上的执行时序与优先级决策？
3. 在欺诈样本比例低于 0.1% 的极端不平衡场景下，如何结合 Focal Loss 与代价敏感矩阵 (Cost-Sensitive Matrix) 设定拦截阈值？
4. 图风控 (Graph Risk) 在识别团伙作案中的应用：如何利用设备-IP-银行卡异构关联图与 Louvain 社区发现算法识别洗钱网络？
5. 影子模式 (Shadow Mode) 在高危风控模型上线中的应用：如何在零业务中断风险的前提下验证新模型的误杀率与拦截率？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/MLE/card/mle-risk-anti-fraud-architecture?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mle-risk-anti-fraud-architecture)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
