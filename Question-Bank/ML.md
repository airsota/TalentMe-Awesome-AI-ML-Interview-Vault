# 📊 Classic Machine Learning · 经典机器学习 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **44 cards · 220 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. AdaBoost Derivation · AdaBoost 算法手推

> AdaBoost fits the additive model $f(x) = \sum_{m=1}^{M} \alpha_m G_m(x)$ by forward stagewise additive modeling under exponential loss $L_{\exp}(f) = \sum_{i=1}^{n} e^{-y_i f(x_i)}$.

**Interview questions**

1. Whiteboard the AdaBoost weight: start from exponential loss, split samples into correct/misclassified, differentiate w.r.t. $\alpha_m$ to get $\alpha_m = \frac{1}{2} \ln \frac{1 - e_m}{e_m}$
2. Derive the multiplicative update $w_i \leftarrow w_i e^{-\alpha_m y_i G_m(x_i)}$: multiply by $e^{-\alpha_m}$ (correct) vs $e^{\alpha_m}$ (wrong); why must weights be renormalized? What is the ratio of the two factors?
3. Why train the weak learner on weighted data? Why does the objective reduce to the weighted error rate for fixed $\alpha_m$?
4. Why exponential loss? Its surrogate relationship with 0-1 loss and log loss (LogitBoost), and the tradeoffs
5. Meaning of the training error bound $\prod_m 2\sqrt{e_m(1 - e_m)}$; the relationship between AdaBoost and additive models / GBDT

<details>
<summary>中文题目</summary>

1. 白板推导 AdaBoost 权重: 从指数损失出发, 固定 $G_m$ 把样本拆成“分对/分错”两组, 对 $\alpha_m$ 求导得 $\alpha_m = \frac{1}{2} \ln \frac{1 - e_m}{e_m}$
2. 推导样本权重乘性更新 $w_i \leftarrow w_i e^{-\alpha_m y_i G_m(x_i)}$: 分对乘 $e^{-\alpha_m}$、分错乘 $e^{\alpha_m}$, 为什么必须归一化? 两个乘子之比是多少?
3. 为什么每轮用“带权数据”训练弱分类器? 固定 $\alpha_m$ 后目标为什么退化为加权错误率?
4. 为什么用指数损失? 它与 0-1 损失、对数损失 (LogitBoost) 的代理关系与各自优劣?
5. 训练误差上界 $\prod_m 2\sqrt{e_m(1 - e_m)}$ 的含义; AdaBoost 与加法模型/GBDT 的关系

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/adaboost-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=adaboost-derivation)

---

## 2. Bagging & Random Forest · Bagging 与随机森林

> Bagging (Bootstrap Aggregating) is a parallel ensemble: for $k$ base learners, draw $n$ samples with replacement from the training set (bootstrap) and train each learner on its own bag; predict by voting (classification) or averaging (regression).

**Interview questions**

1. Bootstrap derivation: probability a sample is never drawn is $(1 - \frac{1}{n})^n \to e^{-1} \approx 0.368$; how is OOB error computed and why does it need no separate validation set?
2. Variance decomposition: $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{k} \sigma^2$ — why does bagging reduce variance but not bias? Why is variance not zero as $k \to \infty$?
3. RF double randomness: bootstrap samples + random feature subsets ($m = \sqrt{d}$ classification / $d/3$ regression); why is feature randomness the key ingredient?
4. Why must bagging base learners be strong (deep, unpruned)? What happens if they are shallow?
5. OOB error vs K-Fold CV error: pros and cons; how does RF use OOB to compute feature importance?

<details>
<summary>中文题目</summary>

1. Bootstrap 推导: 单个样本 $n$ 次有放回抽样从未被抽中的概率 $(1 - \frac{1}{n})^n \to e^{-1} \approx 0.368$; OOB 误差如何计算? 为什么 OOB 误差无需单独验证集?
2. 方差分解: $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{k} \sigma^2$ — 为什么 Bagging 降方差不降偏差? 为什么 $k \to \infty$ 时方差不为 0?
3. 随机森林的双重随机性: 样本自助采样 + 分裂特征随机子集 ($m = \sqrt{d}$ 分类 / $d/3$ 回归); 为什么特征随机是关键而非可有可无?
4. 为什么 Bagging 的基学习器要强 (深树、不剪枝)? 换成浅树会怎样?
5. OOB 误差 vs K-Fold 交叉验证误差: 优劣对比; 随机森林如何利用 OOB 计算特征重要性?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/bagging-forest?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bagging-forest)

---

## 3. Baum-Welch (HMM EM) · HMM 参数学习 Baum-Welch

> The learning problem estimates HMM parameters $\lambda = (A, B, \pi)$ from observations alone with no state labels; Baum-Welch is exactly EM applied to HMMs.

**Interview questions**

1. Write the Baum-Welch reestimation formula $\hat a_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}$ and explain the meaning of $\gamma_t(i)$ and $\xi_t(i,j)$.
2. Why is Baum-Welch a special case of EM? What do its E-step and M-step correspond to?
3. How do forward-backward computations yield $\gamma_t(i)$ and $\xi_t(i,j)$ (combining $\alpha_t(i)$, $a_{ij}$, $b_j(o_{t+1})$, $\beta_{t+1}(j)$)?
4. Does Baum-Welch always converge to the global optimum? Why not, and how do you handle local optima?
5. What numerical problem arises in forward-backward over long sequences, and how is it fixed?

<details>
<summary>中文题目</summary>

1. 写出 Baum-Welch 的重估公式 $\hat a_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}$,并解释 $\gamma_t(i)$、$\xi_t(i,j)$ 的含义?
2. Baum-Welch 为什么是 EM 算法的特例?E 步和 M 步分别对应什么?
3. 前向-后向算法如何得到 $\gamma_t(i)$ 和 $\xi_t(i,j)$?(组合 $\alpha_t(i)$、$a_{ij}$、$b_j(o_{t+1})$、$\beta_{t+1}(j)$)
4. Baum-Welch 一定能收敛到全局最优吗?为什么?如何处理局部最优?
5. 长观测序列下前向-后向计算会有什么数值问题?如何解决?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/baum-welch?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=baum-welch)

---

## 4. Class Imbalance · 类别不平衡

> Class imbalance means highly skewed class ratios (e.g. 1:1000). Problem and impact: the model favors the majority class and the decision boundary is dragged toward the minority;

**Interview questions**

1. Why does PR beat ROC under class imbalance? How does the huge true-negative denominator dilute FPR and thus ROC?
2. Positive:negative = 1:1000, 99.9% accuracy, but business needs high recall: what are your steps (metric → method → validation)?
3. How does class weight (cost-sensitive learning) work? Is it mathematically equivalent to oversampling?
4. How do you move the decision threshold? How does it relate to resampling and weighted loss?
5. Under imbalance, which reflects business better, F1 or AUC? When is AUC still fine?

<details>
<summary>中文题目</summary>

1. 为什么类别不平衡时 PR 曲线优于 ROC 曲线? FPR 分母中的海量真负样本如何稀释 ROC?
2. 正负比 1:1000、准确率 99.9% 但业务要求高召回: 你的处理步骤(指标→方法→验证)是什么?
3. class weight(代价敏感学习)的原理? 它与过采样在数学上等价吗?
4. 阈值移动(threshold moving)怎么操作? 它与重采样、加权损失是什么关系?
5. 不平衡下 F1 与 AUC 哪个更能反映业务? 什么场景下用 AUC 依然合适?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/data-imbalance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=data-imbalance)

---

## 5. Confusion Matrix · 混淆矩阵

> the fundamental counting table for binary classification, splitting samples into 4 cells by “true label × predicted label”: TP (predicted positive, actually positive), FP (predicted positive, actually negative — Type I error), FN (predicted negative, actually…

**Interview questions**

1. Hand computation: given $TP=40, FP=10, FN=20, TN=30$, compute accuracy, precision, recall and F1.
2. Why is a model that always predicts negative, at 99% accuracy for a 1% prevalence disease, useless? Which cell gets sacrificed?
3. Difference between Type I (FP) and Type II (FN) errors; which cell is false-blocking vs missing spam?
4. How macro vs micro averaging are computed and differ; why choose macro when classes are severely imbalanced (e.g. 1000:1)?
5. How to represent and read a multiclass confusion matrix (diagonal, row-normalized, weighted average)?

<details>
<summary>中文题目</summary>

1. 手算: 给定 $TP=40, FP=10, FN=20, TN=30$, 求准确率、Precision、Recall、F1?
2. 为什么 1% 患病率的筛查模型“全预测为阴性”时准确率 99% 却不可用? 哪个格子被牺牲了?
3. Type I (FP) 与 Type II (FN) 错误的区别; 垃圾邮件被误拦与漏放分别对应哪个格子?
4. 宏平均与微平均如何计算、有何区别; 类别极不平衡 (如 1000:1) 时为何选宏平均?
5. 多分类混淆矩阵如何表示与解读 (对角线 / 每行归一 / 加权平均)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/confusion-matrix?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=confusion-matrix)

---

## 6. Data Drift & PSI · 数据漂移与 PSI

> PSI (Population Stability Index) is the most common metric for monitoring input-distribution drift, quantifying the binned divergence between the baseline (e.g.

**Interview questions**

1. Write $PSI = \sum_i (A_i - B_i)\ln\frac{A_i}{B_i}$ and define $A_i$/$B_i$; why is PSI = 0 when $A_i = B_i$? How does it relate to KL divergence?
2. PSI thresholds <0.1 / 0.1–0.25 / >0.25 and the action for each; how to handle zero bins ($B_i = 0$ or $A_i = 0$)?
3. Covariate drift vs concept drift: definitions, one real example each, and how each affects the model differently?
4. Why monitor PSI instead of online AUC alone? What are PSI's limitations (e.g. sensitivity to binning)?
5. The action pipeline when PSI ≥ 0.25 (attribute → retrain → rollback); how to distinguish real drift from data leakage or definition changes (feature/口径 change)?

<details>
<summary>中文题目</summary>

1. 写出 PSI 公式 $PSI = \sum_i (A_i - B_i)\ln\frac{A_i}{B_i}$ 并解释 $A_i$/$B_i$ 的含义;为什么 $A_i = B_i$ 时 PSI = 0?它与 KL 散度是什么关系?
2. PSI 经验阈值: <0.1 / 0.1–0.25 / >0.25 各代表什么处置动作?分箱时遇到 $B_i = 0$ 或 $A_i = 0$ 怎么处理?
3. Covariate Drift 与 Concept Drift 的区别?各举一个实际例子,两者对模型的影响有何不同?
4. 为什么监控模型退化要用 PSI 而不是只看线上 AUC?PSI 的局限性有哪些(如对分箱数敏感)?
5. 检测到 PSI ≥ 0.25 后的处置流程(归因→重训→回滚);如何区分真实漂移与数据泄漏/口径变化?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/psi-data-drift?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=psi-data-drift)

---

## 7. Data Leakage · 数据泄漏

> Data leakage means the training process uses information unavailable at validation/test (or production) time; three types: ① target leakage — features contain the target directly or indirectly (using “already defaulted” to predict default, refund amount to pre…

**Interview questions**

1. Give a concrete example of each leakage type (target/temporal/pipeline); why do all three inflate CV scores?
2. Why must scaling happen inside CV (split first, then fit the scaler)? What goes wrong with fit-on-all-then-split?
3. How do target encoding and group-mean statistics leak? Which leakage does the OOF procedure prevent?
4. How do you detect leakage (near-zero training error, CV far below training, abnormal importances, AUC near 1)?
5. What forms can future information take in time-series? How do lag features and rolling statistics leak when misused?

<details>
<summary>中文题目</summary>

1. 三类泄漏(目标/时序/管线)各举一个具体例子, 为什么它们都会让 CV 分数虚高?
2. 标准化为什么必须在 CV 内部做(先 split 再 fit scaler)? 先 fit 全量再 split 会发生什么?
3. 目标编码/类别均值统计如何泄漏? OOF 编码的具体步骤是防哪种泄漏?
4. 如何检测泄漏(训练误差极低、CV 远低于训练、特征重要性异常、AUC 接近 1)?
5. 时序数据中“未来信息”有哪些表现形式? 滞后特征/滚动统计做错了怎么泄漏?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/data-leakage?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=data-leakage)

---

## 8. DBSCAN & Hierarchical Clustering · DBSCAN 与层次聚类

> DBSCAN is a density-based clustering algorithm that needs no preset cluster count $K$ — only two parameters, ε and minPts.

**Interview questions**

1. Define core, border, and noise points in DBSCAN; how do larger/smaller ε and minPts change the clustering?
2. Distinguish density-reachable from density-connected; why is reachability asymmetric while connectedness is symmetric?
3. Core differences between DBSCAN and K-Means; which data shapes suit each; why does DBSCAN fail on uneven density?
4. Single, complete, and Ward linkage criteria in hierarchical clustering; what is the chaining effect?
5. Time complexity of DBSCAN and index acceleration; why is ε hard to choose in high dimensions?

<details>
<summary>中文题目</summary>

1. 给出 DBSCAN 中核心点、边界点、噪声点的判定条件? ε 和 minPts 增大/减小如何改变聚类结果?
2. 区分密度直达、密度可达、密度相连; 为什么密度可达不具备对称性而密度相连具备?
3. DBSCAN 与 K-Means 的核心区别? 各自适用什么数据形态? DBSCAN 对密度不均数据为何失效?
4. 层次聚类的单链接、全链接、Ward 准则分别用什么距离? 链式效应指什么?
5. DBSCAN 的时间复杂度与索引加速? 高维数据下为什么 ε 难选?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/dbscan-hierarchical?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dbscan-hierarchical)

---

## 9. Decision Tree Splitting · 决策树 CART 分裂

> A decision tree recursively partitions the input space with (feature, threshold) splits; the splitting criterion measures split quality.

**Interview questions**

1. Whiteboard Gini impurity $G = 1 - \sum_k p_k^2$: why does it reduce to $2p(1-p)$ for binary classes? Pros and cons vs entropy?
2. Write information gain and explain its bias toward many-valued features; how does the gain ratio fix it with the intrinsic value $\text{IV}(A)$?
3. What splitting criteria do ID3 / C4.5 / CART use? Why must CART be a binary tree?
4. How are continuous features split? Why only midpoints of adjacent sorted values? What is the per-feature complexity?
5. What are the splitting criterion and leaf output of a CART regression tree? Why is the leaf output the mean $\bar{y}$?

<details>
<summary>中文题目</summary>

1. 白板推导 Gini 不纯度 $G = 1 - \sum_k p_k^2$: 二分类为什么退化为 $2p(1-p)$? 与熵 $H = -\sum_k p_k \log p_k$ 相比有何优缺点?
2. 写出信息增益 $\text{Gain}(D, A) = H(D) - \sum_v \frac{|D_v|}{|D|} H(D_v)$ 并解释为什么它偏好取值多的特征; 增益比如何用固有值 $\text{IV}(A)$ 修正?
3. ID3 / C4.5 / CART 的分裂准则分别是什么? 为什么 CART 一定是二叉树?
4. 连续特征如何切分? 为什么只用相邻样本中点 $\frac{x^{(i)} + x^{(i+1)}}{2}$ 作候选切分点? 单特征的时间复杂度是多少?
5. CART 回归树的分裂准则与叶子输出是什么? 为什么叶子输出是均值 $\bar{y}$?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/decision-trees?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=decision-trees)

---

## 10. Distance Metrics & Indexing · 距离度量与 KD-Tree

> Distance metrics define similarity in feature space and directly drive the results of KNN, clustering, and retrieval. Common metrics: Euclidean distance $d_2(x, y) = \sqrt{\sum_i (x_i - y_i)^2}$ (L2, most natural when all dimensions share scale);

**Interview questions**

1. Write the Euclidean, Manhattan, and cosine similarity formulas; how does Minkowski distance unify them?
2. Why is cosine similarity $\cos \theta = \frac{x \cdot y}{\Vert x \Vert \Vert y \Vert}$ invariant to vector length? Why is it standard for text retrieval?
3. Euclidean vs cosine vs Manhattan — when to use each? Why is L1 more stable than L2 in high dimensions?
4. How does a KD-Tree build and query? Why does it degrade toward a linear scan in high dimensions?
5. Why is feature scaling essential for distances? How do different scales distort them? What is the standardization formula?

<details>
<summary>中文题目</summary>

1. 写出欧氏、曼哈顿、余弦相似度的公式? 闵可夫斯基距离如何统一三者?
2. 为什么余弦相似度 $\cos \theta = \frac{x \cdot y}{\Vert x \Vert \Vert y \Vert}$ 对向量长度不敏感? 文本检索为什么常用?
3. 何时用欧氏、何时用余弦、何时用曼哈顿? 为什么高维下 L1 比 L2 更稳定?
4. KD-Tree 如何构建与查询? 为什么高维下退化为接近线性扫描?
5. 特征缩放为什么对距离度量必要? 不同量纲特征如何扭曲距离? 标准化的公式是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/distance-metrics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distance-metrics)

---

## 11. EM Algorithm · EM 算法

> EM (expectation-maximization) is a general framework for maximum likelihood estimation in models with latent variables, alternating between an E-step and an M-step until convergence.

**Interview questions**

1. Write the E-step $Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}}[\log p(X, Z \mid \theta)]$ and M-step; why is it called expectation-maximization?
2. Use Jensen's inequality to prove EM never decreases $\log p(X \mid \theta)$; when is the lower bound tight?
3. Does EM reach a global or local optimum? Why is it initialization-sensitive? Common strategies such as K-Means to initialize GMM?
4. When is EM applicable — what are latent variables? How does EM handle missing data? vs direct gradient ascent?
5. How does E-step cost scale with the number of latent states? Which EM specializations are GMM and HMM (Baum-Welch)?

<details>
<summary>中文题目</summary>

1. 写出 EM 的 E 步 $Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}}[\log p(X, Z \mid \theta)]$ 与 M 步定义? 为什么叫“期望-最大化”?
2. 用 Jensen 不等式证明 EM 每次迭代使 $\log p(X \mid \theta)$ 单调不降; 下界何时取等?
3. EM 收敛到全局最优还是局部最优? 为什么对初始值敏感? 常用初始化策略有哪些 (K-Means 初始化 GMM)?
4. EM 的适用条件: 隐变量具体指什么? 如何用 EM 处理缺失数据? 与直接梯度上升最大化的区别?
5. E 步的计算复杂度如何随隐变量取值个数增长? GMM 与 HMM (Baum-Welch) 分别是 EM 的哪个特例?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/em-algorithm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=em-algorithm)

---

## 12. Ensemble Comparison · Bagging vs Boosting vs Stacking

> The three schools of ensemble learning. Bagging (random forests): base learners train independently and in parallel, each on its own bootstrap sample, voting/averaging at inference — base learners should be strong (deep, unpruned trees);

**Interview questions**

1. Comparison table: Bagging vs Boosting vs Stacking across base-learner strength (strong/weak/heterogeneous), parallelism (parallel/sequential/two-stage), data usage (bootstrap/weighted/OOF), bias vs variance reduction, and representative algorithms
2. Bias-variance decomposition and the average variance $\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{k}\sigma^2$: why does bagging reduce variance but boosting reduce bias?
3. Why does bagging need strong base learners (deep unpruned trees) while boosting needs weak ones (shallow)? What breaks if reversed?
4. Essential difference of stacking: heterogeneous models + meta-learner; why must meta-features come from OOF folds (leakage prevention)?
5. Scenario selection: which ensemble for high-variance vs high-bias underfitting? When does stacking pay off most?

<details>
<summary>中文题目</summary>

1. 对比表: Bagging vs Boosting vs Stacking 在 基学习器强度 (强/弱/异质)、并行性 (并行/串行/两阶段)、数据使用 (Bootstrap/加权/OOF)、降偏差还是降方差、代表算法 各维度
2. 偏差-方差分解: $\text{Err} = \text{bias}^2 + \text{variance} + \sigma_\epsilon^2$ 与平均方差 $\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{k}\sigma^2$ — 为什么 Bagging 降方差而 Boosting 降偏差?
3. 为什么 Bagging 要强基学习器 (深树不剪枝)、Boosting 要弱基学习器 (浅树)? 反过来会发生什么?
4. Stacking 与 Bagging/Boosting 的本质区别: 异质模型 + 元学习器; 为什么元特征必须用 OOF 折叠生成 (防泄漏)?
5. 场景选择题: 高方差欠拟合 vs 高偏差欠拟合分别选哪种集成? 什么时候 Stacking 收益最大?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/ensemble-comparison?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ensemble-comparison)

---

## 13. GBDT Negative Gradient · GBDT 负梯度拟合

> GBDT (Gradient Boosting Decision Tree) views boosting as gradient descent in function space: with the additive model $F_M(x) = \sum_{m=1}^{M} \rho_m h_m(x)$, each round updates along the negative gradient of the loss w.r.t.

**Interview questions**

1. Whiteboard the negative-gradient pseudo-residual $\tilde{y}_i = -\left[\frac{\partial L}{\partial F}\right]_{F = F_{m-1}}$: why does it equal $y - F_{m-1}$ under squared loss?
2. Derive leaf outputs: mean under squared loss, median under absolute loss; why a Newton step for binary log loss?
3. Relationship to AdaBoost: why is AdaBoost a special case of gradient boosting under exponential loss? How do base learners and weighting differ?
4. Role of shrinkage $\nu$: why do small learning rates with more trees usually work better? Effect on the bias-variance tradeoff?
5. Which loss functions can GBDT adapt? Compare its use cases with random forests (bias vs variance reduction)?

<details>
<summary>中文题目</summary>

1. 白板推导负梯度伪残差 $\tilde{y}_i = -\left[\frac{\partial L}{\partial F}\right]_{F = F_{m-1}}$: 平方损失下为什么就是 $y - F_{m-1}$?
2. 叶子输出推导: 平方损失叶子输出为均值、绝对值损失为中位数; 二分类 log-loss 下叶子值为什么是牛顿步?
3. GBDT 与 AdaBoost 的关系: 为什么指数损失下 AdaBoost 是梯度提升的特例? 两者的基学习器/加权方式有何不同?
4. 收缩 $\nu$ (学习率) 的作用: 为什么小学习率 + 多棵树通常更好? 对偏差-方差权衡的影响?
5. GBDT 能适配哪些损失函数? 与随机森林的适用场景对比 (降偏差 vs 降方差)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/boosting-gbdt?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=boosting-gbdt)

---

## 14. GMM Soft Assignment · GMM 软分配

> A Gaussian mixture model (GMM) models the data distribution as a weighted sum of $K$ Gaussian components, $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$, with mixing coefficients satisfying $\sum_k \pi_k = 1, \ \pi_k \ge 0$.Parameters are es…

**Interview questions**

1. Write the GMM density $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ and the responsibility $\gamma_{ik}$; why does $\sum_k \gamma_{ik} = 1$?
2. Why is K-Means the EM special case of GMM as $\Sigma_k \to 0$? How does the responsibility collapse to hard 0/1 assignment?
3. Derive the GMM M-step updates $\pi_k = \tfrac{N_k}{N}$ and $\mu_k = \tfrac{1}{N_k} \sum_i \gamma_{ik} x_i$; why are these weighted means?
4. When do GMM and K-Means clusters differ significantly? What extra value does soft assignment bring (confidence, anomaly detection)?
5. Parameter count and overfitting risk of GMM? Why can covariances degenerate to singular solutions, and what constraints help?

<details>
<summary>中文题目</summary>

1. 写出 GMM 的概率密度 $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ 与责任度 $\gamma_{ik}$ 公式? 为什么 $\gamma_{ik}$ 满足 $\sum_k \gamma_{ik} = 1$?
2. 为什么说 K-Means 是 GMM 在 $\Sigma_k \to 0$ 时的 EM 特例? 责任度如何退化为硬 0/1 指派?
3. 推导 GMM 的 M 步更新 $\pi_k = \tfrac{N_k}{N}$, $\mu_k = \tfrac{1}{N_k} \sum_i \gamma_{ik} x_i$? 为什么是加权均值?
4. GMM 与 K-Means 的聚类结果何时明显不同? 软分配的额外价值 (置信度、异常检测) 是什么?
5. GMM 的参数数量与过拟合风险? 协方差退化为奇异解的原因与常见约束?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/gmm-em?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gmm-em)

---

## 15. HMM Forward Algorithm (Evaluation) · HMM 前向算法 (评估)

> The evaluation problem asks for the observation probability $P(O \mid \lambda)$ given HMM parameters $\lambda = (A, B, \pi)$ and an observation sequence $O = o_1 o_2 \dots o_T$.

**Interview questions**

1. Write the forward recursion $\alpha_t(j) = [\sum_i \alpha_{t-1}(i)a_{ij}]b_j(o_t)$ with its initialization $\alpha_1(j)$ and termination $P(O \mid \lambda)$?
2. Why is naive computation of $P(O \mid \lambda)$ exponential in $O(N^T)$, and how does the forward algorithm reduce it to $O(N^2T)$?
3. How do the forward algorithm and Viterbi differ (sum vs max, and what each outputs)?
4. What is the probabilistic meaning of $\alpha_t(j)$, and how does it combine with the backward variable $\beta_t(j)$ to give $P(q_t = j \mid O, \lambda)$?
5. Compute $P(O \mid \lambda)$ by hand for a concrete HMM (e.g. $N=2$, $T=3$), showing each forward step?

<details>
<summary>中文题目</summary>

1. 写出前向算法递推 $\alpha_t(j) = [\sum_i \alpha_{t-1}(i)a_{ij}]b_j(o_t)$,并说明初值 $\alpha_1(j)$ 与终止条件 $P(O \mid \lambda)$?
2. 为什么直接计算 $P(O \mid \lambda)$ 是指数级 $O(N^T)$?前向算法如何把复杂度降到 $O(N^2T)$?
3. 前向算法与维特比算法有什么区别(求和 vs 取 max、各自输出什么)?
4. 前向变量 $\alpha_t(j)$ 的概率含义是什么?它如何与后向变量 $\beta_t(j)$ 组合得到 $P(q_t = j \mid O, \lambda)$?
5. 给定具体 HMM(如 $N=2$、$T=3$),手算 $P(O \mid \lambda)$ 的完整数值过程?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/hmm-evaluation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hmm-evaluation)

---

## 16. Huber Loss & Focal Loss · Huber 与 Focal Loss

> Huber loss (smooth L1): for residual $r = y - f(x)$, quadratic for small errors and linear for large ones, piecewise defined as…

**Interview questions**

1. Write the piecewise Huber loss $\mathcal{L}_\delta(r)$; what does δ control? Why is it continuous and differentiable at $|r|=\delta$?
2. Huber vs MSE vs MAE: gradient on outliers, differentiability at zero, and convergence speed?
3. Write $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log p_t$; what do γ and α each do?
4. Why does Focal loss help with class imbalance? Is it a replacement or complement to resampling and class weights?
5. How are Smooth L1 and Huber related? Why does detection use Smooth L1 for box regression instead of MSE?

<details>
<summary>中文题目</summary>

1. 写出 Huber loss 分段公式 $\mathcal{L}_\delta(r)$; δ 的作用是什么? 为什么说它在 $|r|=\delta$ 处连续可导?
2. Huber 与 MSE/MAE 对比: 对离群点梯度、原点可导性、收敛速度各有什么特点?
3. 写出 Focal loss 公式 $FL(p_t) = -\alpha_t(1-p_t)^\gamma \log p_t$; γ 与 α 各自的作用?
4. 为什么 Focal loss 能缓解类别不平衡? 它与重采样、class weight 是替代还是互补关系?
5. Smooth L1 与 Huber 的关系? 目标检测为什么框回归用 Smooth L1 而不是 MSE?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/loss-huber-focal?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=loss-huber-focal)

---

## 17. K-Fold Cross-Validation · K-Fold 交叉验证

> K-Fold cross-validation splits the dataset into K equal folds, trains on K-1 and validates on the remaining one in rotation; the error estimate is the mean over folds $\hat{E} = \frac{1}{K}\sum_{k=1}^{K} E_k$ where $E_k$ is the validation error of fold k.

**Interview questions**

1. Write the K-Fold error estimate $\hat{E} = \frac{1}{K}\sum_{k=1}^{K} E_k$ and explain why K=5 or 10 beats LOOCV (bias-variance angle)?
2. Stratified vs plain K-Fold? Why does ignoring stratification hurt classification? When must you use Group K-Fold?
3. Why does larger K reduce bias but increase variance? How does training-set overlap between folds affect variance?
4. CV and leakage: why must scaling, imputation and feature selection be done inside each fold?
5. What does a training error far below the CV error tell you? How does CV error relate to test error?

<details>
<summary>中文题目</summary>

1. 写出 K-Fold 误差估计公式 $\hat{E} = \frac{1}{K}\sum_{k=1}^{K} E_k$ 并说明: 为什么 K=5/10 比 LOOCV 更常用(偏差-方差角度)?
2. 分层 K-Fold 与普通 K-Fold 的区别? 分类数据为什么不分层会出问题? 什么场景必须用 Group K-Fold?
3. 为什么 K 越大偏差越低、方差越高? 各折训练集重叠度如何影响方差?
4. 交叉验证与数据泄漏的关系: 为什么标准化、缺失值填充、特征选择都必须在每一折内部完成?
5. 训练误差远低于交叉验证误差能说明什么? 交叉验证误差与测试误差的关系是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/cross-validation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cross-validation)

---

## 18. K-Means Algorithm · K-Means 算法

> K-Means partitions $n$ samples into $K$ mutually exclusive hard clusters by coordinate descent on the within-cluster sum of squares (WCSS) objective $J = \sum_{k=1}^{K} \sum_{x \in C_k} \Vert x - \mu_k \Vert^2$.Each iteration alternates two steps: ① assignment…

**Interview questions**

1. Write the K-Means objective $J = \sum_{k} \sum_{x \in C_k} \Vert x - \mu_k \Vert^2$; why do the assignment and update steps never increase $J$, guaranteeing convergence?
2. Derive the centroid update $\mu_k = \tfrac{1}{|C_k|} \sum_{x \in C_k} x$ as the minimizer of the subproblem; is K-Means a convex problem?
3. K-Means vs GMM (hard vs soft assignment)? How is K-Means fundamentally different from KNN?
4. Why does K-Means only reach a local optimum? How does initialization affect the result? Mitigations: K-Means++, multiple random restarts, K-Medoids?
5. What is the time complexity of K-Means? How to pick K (elbow method, silhouette)? Why is it outlier-sensitive and why does it fail on non-convex clusters?

<details>
<summary>中文题目</summary>

1. 手推 K-Means 目标函数 $J = \sum_{k} \sum_{x \in C_k} \Vert x - \mu_k \Vert^2$? 为什么分配步与更新步能保证 $J$ 单调不增、算法必收敛?
2. 推导质心更新公式 $\mu_k = \tfrac{1}{|C_k|} \sum_{x \in C_k} x$: 固定分配后对 $\mu_k$ 求导置零得最优解; K-Means 是凸优化吗?
3. K-Means 与 GMM 的区别 (硬分配 vs 软分配)? K-Means 与 KNN 有什么本质不同?
4. K-Means 为什么只能收敛到局部最优? 初始化如何影响结果? 缓解手段有哪些 (K-Means++、多次随机重启、K-Medoids)?
5. K-Means 的时间复杂度是多少? 如何选 K (肘部法、轮廓系数)? 为什么对异常值敏感、对非凸簇失效?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/kmeans?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kmeans)

---

## 19. K-Means++ & K Selection · K-Means++ 与 K 选择

> K-Means++ is an improved initialization for K-Means: the first centroid is chosen uniformly at random, then each subsequent centroid is sampled with probability proportional to the squared distance to the nearest already-chosen centroid, where…

**Interview questions**

1. Write the K-Means++ sampling probability $P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$ and explain $D(x)$; why squared distance rather than plain distance?
2. How much does K-Means++ improve over random initialization in expectation? What does the $\mathbb{E}[J] \le 8(\ln K + 2) J_{\mathrm{OPT}}$ bound mean?
3. How do the elbow method and silhouette score choose K? Range and interpretation of $s = \frac{b - a}{\max(a, b)}$?
4. Why is K-Means outlier-sensitive? How does K-Medoids mitigate it? Complexity comparison with K-Means?
5. What extra time does K-Means++ cost? How is the first centroid chosen? vs multiple random restarts?

<details>
<summary>中文题目</summary>

1. 写出 K-Means++ 采样概率 $P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$ 并解释 $D(x)$ 的含义? 为什么用距离平方而不是距离本身?
2. K-Means++ 相比随机初始化在期望上改善多少? 近似比 $\mathbb{E}[J] \le 8(\ln K + 2) J_{\mathrm{OPT}}$ 说明什么?
3. 肘部法和轮廓系数如何选择 K? 轮廓系数 $s = \frac{b - a}{\max(a, b)}$ 的取值范围与解释?
4. K-Means 为什么对异常值敏感? K-Medoids 如何缓解? 与 K-Means 的复杂度对比?
5. K-Means++ 的额外时间开销是多少? 第一个质心如何选取? 与多次随机重启哪个更常用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/kmeans-init?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kmeans-init)

---

## 20. Kernel Trick & RBF · 核技巧与 RBF

> SVM's dual optimization and prediction only depend on sample inner products $x_i^Tx_j$. The kernel trick computes inner products in a high- (even infinite-) dimensional feature space directly via $K(x, z) = \langle\phi(x), \phi(z)\rangle$ without ever construc…

**Interview questions**

1. Derive RBF as an infinite-dimensional map: expand $e^{-\gamma\Vert x-z\Vert^2} = e^{-\gamma\Vert x\Vert^2} e^{-\gamma\Vert z\Vert^2} \sum_k \frac{(2\gamma x \cdot z)^k}{k!}$ to expose the features.
2. Mercer's theorem: what makes a valid kernel (symmetric PSD)? Which functions are invalid? Prove sums/products of kernels stay kernels.
3. Derive the explicit feature dimension of the polynomial kernel $K = (x^Tz + c)^d$ (e.g. $\binom{n+d}{d}$) and how the kernel avoids building it.
4. RBF tuning: bias-variance behavior as $\gamma$ (or $\sigma$) grows/shrinks, interplay with $C$; which direction to move each under under/overfitting?
5. Which algorithms are kernelizable (need only inner products/norms) and which are not? Linear vs RBF when $d$ is large or $n$ is small?

<details>
<summary>中文题目</summary>

1. 推导 RBF 核是无穷维特征映射: 用泰勒展开 $e^{-\gamma\Vert x-z\Vert^2} = e^{-\gamma\Vert x\Vert^2} e^{-\gamma\Vert z\Vert^2} \sum_k \frac{(2\gamma x \cdot z)^k}{k!}$ 展开特征。
2. Mercer 定理: 合法核需要什么条件 (对称半正定), 哪些函数不是合法核? 证明核的线性组合/乘积仍是核。
3. 推导多项式核 $K = (x^Tz + c)^d$ 对应的显式特征映射维度 (如 $\binom{n+d}{d}$), 说明核如何避免构造它。
4. RBF 调参: $\gamma$ (或 $\sigma$) 变大/变小的偏差-方差行为, 与 C 的配合, 欠/过拟合时各自往哪个方向调?
5. 哪些算法可以核化 (只需内积/范数), 哪些不行? 线性核 vs RBF 在特征维度 $d$ 很大或 $n$ 很小时怎么选?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/kernel-trick?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kernel-trick)

---

## 21. KNN & K Selection · KNN 与 K 选择

> K-nearest neighbors (KNN) is a non-parametric lazy-learning algorithm: training only stores the data (zero training cost);

**Interview questions**

1. What is the KNN classification/regression procedure? Why is it called lazy learning? What actually happens at train time?
2. What errors come from too-small vs too-large K? How to choose K? How does distance weighting $w_i = 1/d(x, x_i)$ help?
3. What is the curse of dimensionality? Why do neighbors degenerate in high dimensions (relative distances converge)? Mitigations?
4. How does KNN differ from K-Means? Naive prediction complexity $O(nd)$ vs KD-Tree-accelerated inference?
5. Why is KNN sensitive to feature scaling? What preprocessing is required? Outputs for classification vs regression?

<details>
<summary>中文题目</summary>

1. KNN 的分类与回归流程是什么? 为什么叫惰性学习? 训练阶段到底做了什么?
2. K 太小与太大分别造成什么误差? 如何选 K? 距离加权投票 $w_i = 1/d(x, x_i)$ 如何改进?
3. 什么是维度灾难? 高维下为什么近邻退化 (相对距离趋同)? 有哪些缓解手段?
4. KNN 与 K-Means 的区别是什么? 朴素预测复杂度 $O(nd)$ 与 KD-Tree 加速后的复杂度?
5. 为什么 KNN 对特征缩放敏感? 需要哪种预处理? 分类与回归的输出分别是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/knn?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=knn)

---

## 22. Learning to Rank · 排序学习 LTR

> Learning to Rank formulates "score and order the candidate set for a query" as supervised learning in three paradigms: Pointwise regresses/classifies each document independently (e.g.

**Interview questions**

1. Compare Pointwise / Pairwise / Listwise: objective function, pros/cons and representative algorithms for each?
2. Write the LambdaMART gradient $\lambda_{ij} = -\frac{\sigma \Delta NDCG}{1 + e^{\sigma(s_i - s_j)}}$; explain the roles of $\Delta NDCG$ and $\sigma$, and why the negative sign?
3. Why does LambdaMART turn the non-differentiable NDCG into a differentiable regression target? What exactly does each GBDT tree fit?
4. How is LambdaRank's λ derived from RankNet's pairwise loss gradient?
5. Why use NDCG/MRR/GAUC instead of AUC for ranking? What is the essential difference from classification metrics?

<details>
<summary>中文题目</summary>

1. 对比 Pointwise / Pairwise / Listwise 三类范式: 各自的目标函数、优缺点与代表算法?
2. 写出 LambdaMART 的梯度 $\lambda_{ij} = -\frac{\sigma \Delta NDCG}{1 + e^{\sigma(s_i - s_j)}}$,解释 $\Delta NDCG$ 与 $\sigma$ 的作用,以及负号的含义?
3. 为什么说 LambdaMART 把不可导的 NDCG 变成了可导的回归目标?GBDT 每棵树拟合的到底是什么?
4. RankNet 的 pairwise 损失与 LambdaRank 的 λ 是什么关系(λ 如何从 RankNet 梯度推导出来)?
5. 排序任务为什么用 NDCG/MRR/GAUC 而不是 AUC?位置敏感性与分类指标的本质区别是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/learning-to-rank?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=learning-to-rank)

---

## 23. LightGBM GOSS & EFB · LightGBM GOSS/EFB

> LightGBM accelerates training on huge, high-dimensional data with three mechanisms. (1) Histogram-based splitting: each continuous feature is discretized into $k$ bins and splits are only searched on bin boundaries — split complexity drops from $O(n \cdot d)$…

**Interview questions**

1. GOSS mechanism: keep all large-gradient samples + randomly sample small-gradient ones; why are sampled small-gradient weights amplified by $\frac{1-a}{b}$? (unbiased estimation)
2. Why do large-gradient samples carry more information? Relationship between $g_i$ and GBDT pseudo-residuals
3. EFB: what makes features mutually exclusive? How is the conflict ratio measured? How much does bundling reduce complexity?
4. Leaf-wise vs level-wise growth: why is leaf-wise more efficient yet more prone to overfitting? Mitigation (max_depth)?
5. Histogram vs pre-sorted: memory/accuracy tradeoff from $O(n \cdot d)$ to $O(k \cdot d)$; overall LightGBM vs XGBoost comparison

<details>
<summary>中文题目</summary>

1. GOSS 原理与增益估计: 大梯度样本全保留 + 小梯度样本随机采样, 为什么小梯度样本权重要乘 $\frac{1-a}{b}$ 放大? (无偏估计)
2. 为什么大梯度样本携带更多信息? 梯度 $g_i$ 与 GBDT 伪残差的关系
3. EFB 互斥特征捆绑: 什么是互斥特征? 冲突度如何度量? 特征数从 $d$ 降到 bundles 后复杂度下降多少?
4. Leaf-wise 与 Level-wise 生长策略对比: 为什么 Leaf-wise 更高效但更容易过拟合? 如何缓解 (max_depth)?
5. 直方图算法 vs 预排序: 复杂度从 $O(n \cdot d)$ 到 $O(k \cdot d)$ 的内存与精度权衡; LightGBM vs XGBoost 总体对比

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/lightgbm-goss-efb?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=lightgbm-goss-efb)

---

## 24. Linear Regression OLS · 线性回归 OLS

> Linear regression (OLS) assumes $y = Xw + \varepsilon$ and minimizes the residual sum of squares $L(w) = \Vert y - Xw\Vert_2^2$.

**Interview questions**

1. Whiteboard the normal equation: expand $L(w) = \Vert y - Xw\Vert^2$, take the gradient, set it to 0, and arrive at $w = (X^TX)^{-1}X^Ty$; what condition makes $X^TX$ invertible?
2. What are the 5 OLS assumptions? What happens to unbiasedness, efficiency and consistency when each is violated?
3. What if $X^TX$ is singular or ill-conditioned? Compare pseudo-inverse, ridge $w = (X^TX + \lambda I)^{-1}X^Ty$ and dropping collinear features.
4. State the Gauss-Markov theorem: under the 5 assumptions OLS is BLUE — why is normality not among them?
5. Derive the normal equation's complexity (building $X^TX$ is $O(nd^2)$, inversion $O(d^3)$) and explain why high-dimensional sparse settings prefer gradient descent.

<details>
<summary>中文题目</summary>

1. 白板推导正规方程: 展开 $L(w) = \Vert y - Xw\Vert^2$ 求梯度、令为 0, 写出 $w = (X^TX)^{-1}X^Ty$; $X^TX$ 可逆需要什么条件?
2. 线性回归 5 大假设分别是什么? 每一条被违反时估计量的无偏性/有效性/一致性会怎样?
3. $X^TX$ 奇异或病态怎么办? 比较伪逆、岭回归 $w = (X^TX + \lambda I)^{-1}X^Ty$ 与剔除共线特征三种方案。
4. 陈述 Gauss-Markov 定理: 5 大假设下 OLS 是 BLUE (最优线性无偏估计), 为什么正态性假设不在其中?
5. 推导正规方程的时间/空间复杂度 (构造 $X^TX$ 为 $O(nd^2)$, 求逆为 $O(d^3)$), 说明高维稀疏场景为什么选梯度下降。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/linear-regression?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=linear-regression)

---

## 25. Linear-Chain CRF · 线性链条件随机场

> A linear-chain CRF is a discriminative probabilistic model for sequence labeling that models the conditional probability $P(y \mid x)$ directly rather than the joint $P(x, y)$.

**Interview questions**

1. Write the linear-chain CRF potential $P(y \mid x) = \frac{1}{Z(x)}\exp(\sum_t \sum_k \lambda_k f_k(y_{t-1},y_t,x,t))$ and the role of $Z(x)$?
2. How do CRFs and HMMs differ (discriminative vs generative, overlapping features, normalization)?
3. What is the label-bias problem? Why do MEMM/HMM suffer from it and how does CRF fix it?
4. What is the CRF training objective, why is it convex, and which algorithm performs inference?
5. How is the CRF partition function $Z(x)$ computed, and how does its complexity relate to the HMM forward algorithm?

<details>
<summary>中文题目</summary>

1. 写出线性链 CRF 的势函数 $P(y \mid x) = \frac{1}{Z(x)}\exp(\sum_t \sum_k \lambda_k f_k(y_{t-1},y_t,x,t))$,并说明 $Z(x)$ 的作用?
2. CRF 与 HMM 的区别?(判别式 vs 生成式、特征是否可重叠、归一化方式)
3. 什么是标注偏置 (label bias) 问题?MEMM/HMM 为什么会有、CRF 如何解决?
4. CRF 训练的目标函数是什么?为什么是凸的?推断用的是什么算法?
5. CRF 的配分函数 $Z(x)$ 如何计算?复杂度与 HMM 前向算法有何关系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/crf?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=crf)

---

## 26. Logistic Regression · 逻辑回归 Log-Odds

> logistic regression models the log-odds as a linear function $\ln\frac{p}{1-p} = w^Tx + b$, which inverts to the Sigmoid $p = \frac{1}{1 + e^{-(w^Tx + b)}} \in [0,1]$.

**Interview questions**

1. Whiteboard: invert $\ln\frac{p}{1-p} = w^Tx + b$ into $p = \frac{1}{1+e^{-w^Tx}}$, then derive $\frac{\partial L}{\partial z} = p - y$ via $p' = p(1-p)$ and the chain rule.
2. Why does LR use cross-entropy instead of squared loss? Explain the gradient saturation of squared loss as $p \to y$.
3. Show MLE ≡ cross-entropy: derive the negative log-likelihood $L = -\sum_i [y_i \ln p_i + (1-y_i)\ln(1-p_i)]$.
4. Compare LR vs SVM: hinge vs log loss, probabilistic output, support-vector dependence, and outlier robustness.
5. Why can't LR do multiclass directly? Contrast OvR vs Softmax extensions, and how class imbalance shifts the decision threshold.

<details>
<summary>中文题目</summary>

1. 白板推导: 从 $\ln\frac{p}{1-p} = w^Tx + b$ 反推 $p = \frac{1}{1+e^{-w^Tx}}$, 再用 $p' = p(1-p)$ 与链式法则推导 $\frac{\partial L}{\partial z} = p - y$。
2. 为什么 LR 用交叉熵而不是平方损失? 从梯度角度说明平方损失在 $p \to y$ 时的饱和问题。
3. 写出 MLE 与交叉熵的等价性: 负对数似然 $L = -\sum_i [y_i \ln p_i + (1-y_i)\ln(1-p_i)]$ 的推导。
4. LR 与 SVM 对比: hinge loss vs 对数损失的差异、是否输出概率、是否依赖支持向量、对异常值的鲁棒性。
5. 逻辑回归为什么不能直接做多分类? 对比 OvR/Softmax 两种扩展, 以及类别不平衡对决策阈值的影响。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/logistic-regression?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=logistic-regression)

---

## 27. MRR & GAUC · MRR 与 GAUC

> two industry-grade ranking/recommendation metrics. Mean reciprocal rank $MRR = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{rank_q}$, where $rank_q$ is the position of the first relevant item in the result list of query $q$, contributing 0 when nothing relevant exist…

**Interview questions**

1. Hand-compute MRR: first relevant items at ranks 1, 3 and 5 for 3 queries; what does a query with no relevant result contribute?
2. MRR vs NDCG: why does MRR only look at the first relevant position, and how does that connect to “search users only look at the first screen”?
3. Why use GAUC instead of global AUC in recommendation? How does the long-tailed user distribution mislead global AUC?
4. Hand-compute GAUC: user A with 80 samples at AUC 0.7 and user B with 20 samples at AUC 0.6; show why it differs from the simple average.
5. Industrial metric selection: which metrics for recall vs fine-ranking stages, and why does the DIN paper use GAUC?

<details>
<summary>中文题目</summary>

1. 手算 MRR: 3 个 query 的首个相关位置为 1、3、5, 求 MRR; 无相关结果的 query 贡献多少?
2. MRR 与 NDCG 的区别: 为什么 MRR 只看首个相关位置, 与“搜索用户只看第一屏”有什么关系?
3. 为什么推荐场景用 GAUC 而不是全局 AUC? 用户分布头重脚轻如何误导全局 AUC?
4. 手算 GAUC: 用户 A 80 条样本 AUC 0.7, 用户 B 20 条样本 AUC 0.6, 求 GAUC 并说明与简单平均的差异?
5. 工业排序评估如何选型: 召回阶段、精排阶段各用什么指标? 为什么 DIN 论文用 GAUC?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/mrr-gauc?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mrr-gauc)

---

## 28. Multiclass Strategies: OvR/OvO/Softmax · 多分类 OvR/OvO/Softmax

> three main strategies extend binary classifiers to $K$ classes. OvR (one-vs-rest): train $K$ 'class-vs-others' classifiers and predict by the highest score, $O(K \cdot d)$ parameters;

**Interview questions**

1. OvR vs OvO: number of models ($K$ vs $\frac{K(K-1)}{2}$), training/inference cost, tie-breaking; how many models for $K=100$?
2. Role of Softmax temperature $\tau$: derive the limits as $\tau \to 0$ and $\tau \to \infty$; why use $\tau > 1$ in distillation?
3. Whiteboard the Softmax + cross-entropy gradient $\frac{\partial L}{\partial z_i} = p_i - y_i$ and show it generalizes binary LR.
4. Why aren't OvR scores probabilities? How to calibrate (Platt scaling / Isotonic) and how does calibration differ from AUC?
5. The mutual-exclusion assumption: what tasks fit Softmax, why do multi-label tasks use per-class Sigmoid? When to pick OvR vs OvO vs Softmax?

<details>
<summary>中文题目</summary>

1. OvR vs OvO 对比: 模型数量 ($K$ vs $\frac{K(K-1)}{2}$)、训练/推理成本、投票平局问题, $K=100$ 时各需要多少个模型?
2. Softmax 温度 $\tau$ 的作用: 推导 $\tau \to 0$ 与 $\tau \to \infty$ 的极限, 蒸馏中为什么用 $\tau > 1$?
3. 白板推导 Softmax + 交叉熵的梯度 $\frac{\partial L}{\partial z_i} = p_i - y_i$, 并说明它是二分类 LR 的推广。
4. OvR 的得分为什么不是概率? 如何校准 (Platt scaling / Isotonic), 校准与 AUC 有何区别?
5. 类别互斥假设: Softmax 适合什么任务, 多标签任务为什么该用多个 Sigmoid? OvR/OvO/Softmax 何时选哪个?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/multiclass-strategies?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=multiclass-strategies)

---

## 29. Naive Bayes · 朴素贝叶斯

> Naive Bayes is a generative classifier built on Bayes' theorem; its core assumption is that features are conditionally independent given the class $C$.

**Interview questions**

1. Write the NB decision rule $P(C \mid x) \propto P(C)\prod_i P(x_i \mid C)$; what is the “naive” assumption and why can the denominator $P(x)$ be dropped?
2. Why is Laplace smoothing needed? Write $\hat P = \frac{N_{y,x_i}+\alpha}{N_y+\alpha|V|}$ and explain how zero probabilities are avoided.
3. What feature types are Multinomial, Gaussian, and Bernoulli NB each suited for?
4. Is NB generative or discriminative? How does it compare with logistic regression under limited data and strongly correlated features?
5. What happens when the conditional-independence assumption is violated, and why does NB still often deliver usable decisions in practice?

<details>
<summary>中文题目</summary>

1. 写出朴素贝叶斯分类公式 $P(C \mid x) \propto P(C)\prod_i P(x_i \mid C)$,并说明“朴素”假设是什么、分母 $P(x)$ 为什么可省略?
2. 为什么需要拉普拉斯平滑?写出平滑公式 $\hat P = \frac{N_{y,x_i}+\alpha}{N_y+\alpha|V|}$,零概率如何被避免?
3. 多项式、高斯、伯努利三种朴素贝叶斯变体分别适用于什么类型的特征?
4. 朴素贝叶斯是生成式还是判别式?与逻辑回归在小数据、特征相关性强时表现有何不同?
5. 条件独立假设被违反时会发生什么?为什么现实任务中它仍常给出可用的分类结果?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/naive-bayes?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=naive-bayes)

---

## 30. NDCG · NDCG

> a ranking quality metric measuring how well highly relevant documents are placed at the top. Discounted cumulative gain $DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$, where $rel_i$ is the graded relevance at position $i$ (typically 0-4) and…

**Interview questions**

1. Hand computation: relevance [3, 2, 1, 0] ranked as [2, 1, 3, 0]; compute DCG@4, IDCG@4 and NDCG@4.
2. Why $2^{rel_i}-1$ instead of $rel_i$? What is the intuition behind the position discount $\log_2(i+1)$?
3. Why divide DCG by IDCG? What problems does the unnormalized DCG have?
4. How NDCG differs from binary metrics: graded 0-4 relevance vs relevant/irrelevant; how does the truncation k affect results?
5. Relation between NDCG and MRR: what does NDCG@1 correspond to under binary relevance?

<details>
<summary>中文题目</summary>

1. 手算: 相关度 [3, 2, 1, 0] 被排成 [2, 1, 3, 0], 求 DCG@4、IDCG@4 与 NDCG@4?
2. 为什么用 $2^{rel_i}-1$ 而不是直接 $rel_i$? 位置折损 $\log_2(i+1)$ 的直觉是什么?
3. NDCG 为什么要除以 IDCG? 不归一化的 DCG 有哪些问题?
4. NDCG 与二值指标的差异: 0-4 分级相关 vs 相关/不相关; 截断 k 对结果有何影响?
5. NDCG 与 MRR 的关系: 二值相关时 NDCG@1 与首个相关位置的对应是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/ndcg?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ndcg)

---

## 31. PCA & SVD · PCA 与 SVD

> PCA (principal component analysis) is unsupervised dimensionality reduction: it finds the orthogonal directions of maximum variance and projects high-dimensional samples onto the top $k$ principal components.

**Interview questions**

1. Write PCA's max-variance problem $\max_v v^T \Sigma v \; \text{s.t.} \; \Vert v \Vert = 1$ and use Lagrange multipliers to prove the solution is the top eigenvector.
2. Derive PCA = SVD: why are the right singular vectors of the centered data the principal directions? Relationship between singular values and variance?
3. Why is maximizing variance equivalent to minimizing reconstruction error? What is the reconstruction error expression?
4. How to choose the number of components? Common thresholds for the cumulative explained variance ratio $\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$?
5. Why must features be standardized before PCA? How does PCA differ from linear/ridge regression and from kernel PCA / autoencoders?

<details>
<summary>中文题目</summary>

1. 写出 PCA 的方差最大化问题 $\max_v v^T \Sigma v \; \text{s.t.} \; \Vert v \Vert = 1$; 用拉格朗日乘子法证明解是最大特征向量?
2. 推导 PCA = SVD: 中心化数据矩阵的右奇异向量为什么就是主成分? 奇异值与方差的关系?
3. 为什么方差最大化等价于最小化重构误差? 重构误差的表达式是什么?
4. 如何选择主成分个数? 累计解释方差比 $\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$ 的常用阈值是多少?
5. PCA 前为什么必须标准化特征? PCA 与线性回归 (岭回归)、与核 PCA/自编码器的区别?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/pca?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pca)

---

## 32. Precision, Recall & F1 · Precision/Recall/F1

> three core classification metrics derived from the confusion matrix. Precision $P = \frac{TP}{TP + FP}$ (of the samples predicted positive, how many are truly positive — “quality over quantity”);

**Interview questions**

1. Hand computation: a model has $P = 0.9, R = 0.5$; compute F1 and explain why it is below the arithmetic mean.
2. Why does F1 use the harmonic mean instead of the arithmetic mean? What is F1 when $P = 1, R = 0.1$?
3. Meaning of β in $F_\beta$: what do $F_2$ and $F_{0.5}$ emphasize, and which fits medical screening vs ad placement?
4. When to choose the PR curve over ROC: why is PR more sensitive to class imbalance, and what does the baseline $P = \rho$ mean?
5. How to derive the PR curve from model scores, and how to pick the operating threshold (F1 maximization / cost minimization)?

<details>
<summary>中文题目</summary>

1. 手算: 某模型 $P = 0.9, R = 0.5$, 求 F1 并解释为什么低于算术平均?
2. 为什么 F1 用调和平均而不用算术平均? $P = 1, R = 0.1$ 时 F1 是多少?
3. Fβ 中 β 的含义与选择: $F_2$ 与 $F_{0.5}$ 分别侧重什么? 医疗筛查与广告投放各选哪个?
4. PR 曲线与 ROC 曲线如何选择: 类别不平衡时为什么 PR 更敏感? PR 基线 P = ρ 是什么含义?
5. 如何从模型概率分数画出 PR 曲线? 如何选最优阈值 (F1 最大化 / 代价最小化)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/precision-recall-f1?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=precision-recall-f1)

---

## 33. Probability Calibration · 概率校准

> Probability calibration turns model scores into true interpretable probabilities. Platt scaling fits a logistic regression on the raw score: $P(y=1|x) = \frac{1}{1+\exp(Af(x)+B)}$, with parameters $A$ and $B$ estimated by maximum likelihood on a separate valid…

**Interview questions**

1. Write Platt's formula $P(y=1|x) = \frac{1}{1+\exp(Af(x)+B)}$ and explain how A and B are fit. Is it the same thing as logistic regression?
2. Compare Platt vs Isotonic: assumptions, data requirements, overfitting risk, and which models each suits?
3. Write the Brier score $BS = \frac{1}{N}\sum (p_i - y_i)^2$; give examples of “good discrimination, bad calibration” and the reverse?
4. Why does calibration leave AUC unchanged? How does a rank-preserving monotone transform preserve AUC?
5. How to read a reliability diagram? Why is logistic regression naturally calibrated while SVM/GBDT need calibration?

<details>
<summary>中文题目</summary>

1. 写出 Platt 公式 $P(y=1|x) = \frac{1}{1+\exp(Af(x)+B)}$ 并解释 A、B 如何拟合? 它与逻辑回归本质相同吗?
2. Platt 与 Isotonic 的对比: 各自假设、数据量需求、过拟合风险、适用模型类型?
3. 写出 Brier 分数公式 $BS = \frac{1}{N}\sum (p_i - y_i)^2$; 举例“区分好但校准差”和“校准好但区分差”的模型?
4. 为什么校准不影响 AUC? 保序单调变换如何保持 AUC 不变?
5. 可靠性图怎么看? 为什么逻辑回归输出天然校准、而 SVM/GBDT 需要校准?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/probability-calibration?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=probability-calibration)

---

## 34. Resampling Methods: SMOTE/ADASYN/Tomek · 重采样方法 SMOTE/ADASYN/Tomek

> Resampling rebalances class distributions with data-level methods. SMOTE (Synthetic Minority Over-sampling Technique): for a minority sample $x_i$, randomly pick a k-nearest neighbor $x_{z_i}$ and interpolate along the segment to create…

**Interview questions**

1. Write the SMOTE formula $x_{new} = x_i + \lambda(x_{z_i} - x_i)$; why is $\lambda \in [0,1]$ and why does it reduce duplication overfitting?
2. SMOTE's limits: why is it unsafe for high-dimensional/text/sparse features? Is it sensitive to label noise?
3. ADASYN vs SMOTE? Where is the adaptivity (what drives the number of samples generated)?
4. What is a Tomek Link? Oversampling or cleaning? Why combine it with SMOTE?
5. Should SMOTE run before or after the train/validation split? What leakage occurs if it runs before?

<details>
<summary>中文题目</summary>

1. 写出 SMOTE 插值公式 $x_{new} = x_i + \lambda(x_{z_i} - x_i)$, 解释为什么 $\lambda \in [0,1]$、为什么能缓解复制过拟合?
2. SMOTE 的局限: 为什么对高维/文本/稀疏特征不安全? 对标签噪声敏感吗?
3. ADASYN 与 SMOTE 的区别? “自适应”体现在哪里(生成数量按什么决定)?
4. Tomek Links 是什么? 它属于过采样还是清洗? 为什么常与 SMOTE 组合使用?
5. SMOTE 应该在划分训练/验证集之前还是之后做? 放在划分前会带来什么泄漏?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/resampling-methods?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=resampling-methods)

---

## 35. ROC-AUC · ROC-AUC

> sweeping every classification threshold, plot the true positive rate $TPR = \frac{TP}{TP+FN}$ (= recall) on the y-axis against the false positive rate $FPR = \frac{FP}{FP+TN}$ (= 1 - specificity) on the x-axis; AUC is the area under this curve.

**Interview questions**

1. Hand-compute AUC: 4 samples with scores [0.9, 0.7, 0.5, 0.3] and labels [+, -, +, -], counting positive–negative pairs.
2. What is the probabilistic interpretation of AUC? Why is random guessing AUC ≈ 0.5 and a perfect ranking AUC = 1?
3. How to draw the ROC curve from model scores (threshold sweep updating TPR/FPR each step)? How is AUC computed by the trapezoid rule?
4. Why prefer the PR curve over ROC under extreme class imbalance, and why is AUC insensitive to the class ratio?
5. How to pick an operating threshold for a model with AUC = 0.75? Why doesn't AUC directly give a TPR/FPR working point?

<details>
<summary>中文题目</summary>

1. 手算 AUC: 4 个样本分数 [0.9, 0.7, 0.5, 0.3]、标签 [+, -, +, -], 用正负对计数求 AUC?
2. AUC 的概率解释是什么? 为什么随机猜测 AUC ≈ 0.5、完美排序 AUC = 1?
3. 如何从模型分数画出 ROC 曲线 (阈值扫描, 每步更新 TPR/FPR)? AUC 如何用梯形法计算?
4. 类别极不平衡时为什么用 PR 曲线而不用 ROC? 为什么 AUC 对类别比例不敏感?
5. AUC = 0.75 的模型如何选实际业务阈值? AUC 为什么不能直接给出 TPR/FPR 工作点?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/roc-auc?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=roc-auc)

---

## 36. SHAP & LIME · 可解释性 SHAP 与 LIME

> SHAP and LIME are the two most common model explainability methods. SHAP performs strictly additive feature attribution based on the game-theoretic Shapley value: $\phi_i = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[v(S\cup\{i\}) - v(S)]$…

**Interview questions**

1. Write the Shapley formula $\phi_i = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[v(S\cup\{i\}) - v(S)]$, explain the weight $\frac{|S|!(|N|-|S|-1)!}{|N|!}$, and why it is the unique attribution satisfying efficiency/symmetry/dummy/additivity?
2. SHAP's additivity: why does $\sum_i \phi_i = f(x) - E[f]$? How does it fundamentally differ from linear-model coefficients (LIME weights)?
3. Why can TreeSHAP compute exact values for trees? Decode $O(TLD^2)$ and explain why no sampling is needed.
4. Write LIME's surrogate objective $\min_g \mathcal{L}(f, g, \pi_x)$: what are $g$, $\pi_x$, $\mathcal{L}$? Why are LIME results unstable?
5. SHAP vs LIME: global vs local explanations, when to use which; applicability on tabular/image/text tasks?

<details>
<summary>中文题目</summary>

1. 写出 Shapley 值公式 $\phi_i = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[v(S\cup\{i\}) - v(S)]$,解释权重 $\frac{|S|!(|N|-|S|-1)!}{|N|!}$ 的含义,并说明它是唯一满足效率/对称/虚拟/可加性的归因?
2. SHAP 的加性属性: 为什么 $\sum_i \phi_i = f(x) - E[f]$?这与线性模型系数(LIME 的权重)有何本质区别?
3. TreeSHAP 为什么能对树模型精确计算?复杂度 $O(TLD^2)$ 各符号的含义,为什么不需要采样?
4. 写出 LIME 的局部替代目标 $\min_g \mathcal{L}(f, g, \pi_x)$: $g$、$\pi_x$、$\mathcal{L}$ 分别是什么?为什么 LIME 的结果不稳定?
5. SHAP vs LIME 对比: 全局 vs 局部解释分别怎么选?在表格/图像/文本任务上的适用性?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/shap-lime?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=shap-lime)

---

## 37. Soft Margin & Hinge · 软间隔 C 与 Hinge

> when data are not linearly separable, the hard margin (requiring $y_i(w^Tx_i + b) \ge 1$) is infeasible. The soft margin introduces slack variables $\xi_i \ge 0$ allowing violations at a linear penalty:…

**Interview questions**

1. Whiteboard the soft-margin objective: $\min \frac{1}{2}\Vert w\Vert^2 + C\sum_i \xi_i$ s.t. $y_i(w^Tx_i+b) \ge 1 - \xi_i, \xi_i \ge 0$, then eliminate $\xi$ to reach the hinge form.
2. Properties of hinge $\max(0, 1-yf(x))$: why it is a convex upper bound on $0\text{-}1$ loss, why gradient vanishes for $yf(x) \ge 1$, and how the non-differentiable point is handled.
3. Bias-variance tradeoff of $C$: behavior as $C \to \infty$ vs $C \to 0$, and how to choose $C$ via cross-validation.
4. Derive the box constraint $0 \le \alpha_i \le C$ in the soft-margin dual (eliminating the slack multipliers) and the meaning of the two support-vector types.
5. Hinge vs log loss: why is the SVM solution sparse but not LR's? How do they differ on probability output, outliers and overlapping distributions?

<details>
<summary>中文题目</summary>

1. 白板推导软间隔目标: 写出 $\min \frac{1}{2}\Vert w\Vert^2 + C\sum_i \xi_i$ s.t. $y_i(w^Tx_i+b) \ge 1 - \xi_i, \xi_i \ge 0$, 并消去 $\xi$ 化为 hinge 形式。
2. hinge loss $\max(0, 1-yf(x))$ 的性质: 为什么是 $0\text{-}1$ 损失的凸上界, 为什么 $yf(x) \ge 1$ 时梯度为 0, 不可微点如何处理?
3. $C$ 的偏差-方差权衡: $C \to \infty$ 与 $C \to 0$ 分别对应什么行为, 如何用交叉验证选择 $C$?
4. 推导软间隔对偶中盒式约束 $0 \le \alpha_i \le C$ 的由来 (对 $\xi_i$ 的拉格朗日乘子消元), 以及两类支持向量的含义。
5. hinge vs log loss 对比: 为什么 SVM 的解稀疏而 LR 不是? 两者对概率输出、异常值、重叠分布数据各有什么偏好?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/soft-margin?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=soft-margin)

---

## 38. SVM Margin & Duality · SVM 最大间隔与对偶

> SVM finds the hyperplane with maximum geometric margin. In the separable case the constraint $y_i(w^Tx_i + b) \ge 1$ (normalized functional margin) yields geometric margin $\frac{2}{\Vert w\Vert}$, so maximizing it is equivalent to minimizing…

**Interview questions**

1. Whiteboard the SVM dual: write $L = \frac{1}{2}\Vert w\Vert^2 - \sum_i \alpha_i[y_i(w^Tx_i + b) - 1]$, take $w, b$ derivatives, substitute back, and state the full dual objective with constraints.
2. Geometric meaning of KKT complementary slackness $\alpha_i[y_i(w^Tx_i+b) - 1] = 0$: why $\alpha_i = 0$ for non-support vectors and how support vectors are identified?
3. Geometric vs functional margin: derive why the margin is $\frac{2}{\Vert w\Vert}$ and why we normalize the functional margin.
4. Derive $f(x) = \sum_{i \in SV} \alpha_i y_i x_i^Tx + b$: why is $w$ a linear combination of support vectors, and what are the train/infer complexities?
5. Why can the dual be kernelized? Which terms in the dual contain only inner products, and where does the primal lack inner products?

<details>
<summary>中文题目</summary>

1. 白板推导 SVM 对偶: 写出 $L = \frac{1}{2}\Vert w\Vert^2 - \sum_i \alpha_i[y_i(w^Tx_i + b) - 1]$, 求 $w, b$ 偏导代回, 写出完整对偶目标与约束。
2. KKT 互补松弛 $\alpha_i[y_i(w^Tx_i+b) - 1] = 0$ 的几何含义: 为什么非支持向量的 $\alpha_i = 0$, 支持向量如何被识别?
3. 几何间隔与函数间隔的区别: 推导间隔为什么是 $\frac{2}{\Vert w\Vert}$, 为什么要把函数间隔归一化?
4. 推导决策面为什么是 $f(x) = \sum_{i \in SV} \alpha_i y_i x_i^Tx + b$: $w$ 为什么是支持向量的线性组合, 训练/推理复杂度各是多少?
5. 为什么对偶问题能被核技巧利用? 对偶目标中哪些项只含内积, 原问题哪个地方没有内积?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/svm-margin?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=svm-margin)

---

## 39. Target Encoding · 目标编码 Target Encoding

> Target encoding encodes high-cardinality categorical features (user_id, city, item ID) with the mean target per category.

**Interview questions**

1. Write the smoothed TE formula $TE = \lambda \cdot \text{mean}(y_i) + (1-\lambda) \cdot \text{global}$ with $\lambda(n_i) = \frac{n_i}{n_i+m}$; intuitively, what do m and $n_i$ do?
2. Why does full-data group-mean encoding leak? Trace the leakage path and the mechanism of inflated CV scores?
3. Concrete steps of OOF (out-of-fold) encoding? How does it differ from CatBoost's Ordered Target Statistics?
4. Why target encoding instead of one-hot for high-cardinality features? Compare dimensionality, sparsity and information content?
5. How to choose the smoothing m? What happens to small groups without shrinkage (high encoding variance, overfitting)?

<details>
<summary>中文题目</summary>

1. 写出平滑目标编码公式 $TE = \lambda \cdot \text{mean}(y_i) + (1-\lambda) \cdot \text{global}$, $\lambda(n_i) = \frac{n_i}{n_i+m}$; 直观解释 m 与 $n_i$ 的作用?
2. 为什么直接用全量组均值编码会泄漏? 描述具体泄漏路径与 CV 虚高的机理?
3. OOF (out-of-fold) 编码的具体步骤? 与 CatBoost Ordered Target Statistics 的区别?
4. 为什么高基数类别特征不用 one-hot 而用 target encoding? 维度、稀疏性、信息量三方面对比?
5. m 平滑参数怎么选? 小样本组不收缩会发生什么(编码方差大、过拟合)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/target-encoding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=target-encoding)

---

## 40. Time-Series Cross-Validation · 时序交叉验证

> Time-series data cannot use random K-Fold; walk-forward (rolling-window) validation must be used instead: the training set contains only past time steps and the validation set is the immediately following future window — step 1 trains on $[0, t]$ and validates…

**Interview questions**

1. Why does random K-Fold fail on time series? Trace the concrete leakage path (e.g. predicting t+1 with samples from t+1 onward)?
2. Describe the walk-forward rolling-window procedure: training span, validation span and step size for step k?
3. How to normalize, impute and target-encode in time-series modeling without leakage (history-only statistics)?
4. What is nested CV? What does each loop do, and which bias does it fix?
5. How does walk-forward differ from plain K-Fold in bias-variance and cost? When is the data loss acceptable?

<details>
<summary>中文题目</summary>

1. 为什么随机 K-Fold 在时序数据上失效? 具体泄漏路径是什么(举例: 预测 t+1 用了 t+1 及之后的样本)?
2. 画/描述 walk-forward 滚动窗口流程: 第 k 步训练区间、验证区间、滚动步长如何选择?
3. 归一化、缺失值填充、目标编码在时序建模中怎么处理才不泄漏(必须只用历史窗口统计)?
4. 嵌套 CV(nested CV)是什么? 外层内层各做什么, 解决什么偏差?
5. walk-forward 与常规 K-Fold 在偏差-方差与计算成本上的差异? 什么情况下损失可接受?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/timeseries-cv?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=timeseries-cv)

---

## 41. Tree Pruning · 决策树剪枝

> Pruning is structural regularization for decision trees, trading off training fit against model complexity. Cost-complexity pruning minimizes $R_\alpha(T) = R(T) + \alpha |T|$, where $R(T)$ is the empirical risk — for classification the weighted misclassificat…

**Interview questions**

1. Whiteboard cost-complexity $R_\alpha(T) = R(T) + \alpha |T|$: meaning of each symbol; how does the optimal subtree shrink as $\alpha$ grows?
2. Derive the pruning criterion: why prune node $t$ to a leaf iff $\frac{R(t) - R(T_t)}{|T_t| - 1} \leq \alpha$?
3. Pre-pruning vs post-pruning: time cost, over/underfitting risk, and generalization comparison
4. CART weakest-link pruning: how is the nested subtree sequence generated? How is $\alpha^*$ chosen by cross-validation?
5. How pruning affects bias-variance; relationship with XGBoost's $\gamma$, max_depth, and min_samples_leaf

<details>
<summary>中文题目</summary>

1. 白板推导代价复杂度 $R_\alpha(T) = R(T) + \alpha |T|$: 各符号含义; $\alpha$ 增大时最优子树如何变化?
2. 临界剪枝条件推导: 为什么内部节点 $t$ 当且仅当 $\frac{R(t) - R(T_t)}{|T_t| - 1} \leq \alpha$ 时剪成叶子?
3. 预剪枝 vs 后剪枝: 时间开销、过拟合/欠拟合风险、泛化能力对比
4. CART 最弱连接剪枝流程: 如何生成嵌套子树序列 $T_0 \supset T_1 \supset \cdots$? 如何用交叉验证选 $\alpha^*$?
5. 剪枝对偏差-方差的影响; 与 XGBoost 的 $\gamma$、max_depth、min_samples_leaf 的关系

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/tree-pruning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=tree-pruning)

---

## 42. VIF Multicollinearity · VIF 多重共线性

> multicollinearity means features are nearly linearly dependent, making $X^TX$ close to singular. VIF (Variance Inflation Factor) diagnoses it quantitatively: regress $x_j$ on the other $p-1$ features to obtain $R_j^2$, then $VIF_j = \frac{1}{1 - R_j^2}$.

**Interview questions**

1. Derive VIF: why $VIF_j = \frac{1}{1-R_j^2}$? Where does $R_j^2$ come from, and what $R_j^2$ does the threshold 10 correspond to?
2. Derive $Var(\hat{w}_j) = \frac{\sigma^2}{(1-R_j^2)\sum_i (x_{ij}-\bar{x}_j)^2}$ and explain how collinearity inflates variance.
3. Why do predictions stay good under multicollinearity? What exactly does it hurt (coefficients, standard errors, t-tests, feature importance)?
4. Compare remedies for collinearity: dropping features vs PCA vs ridge regression — cost and suitability of each.
5. Relationship between VIF, pairwise feature correlation and the condition number $\kappa(X) = \frac{\lambda_{max}}{\lambda_{min}}$; why are tree models immune?

<details>
<summary>中文题目</summary>

1. 推导 VIF: 为什么 $VIF_j = \frac{1}{1-R_j^2}$? $R_j^2$ 从哪来, 阈值 10 对应 $R_j^2$ 等于多少?
2. 推导 OLS 系数方差公式 $Var(\hat{w}_j) = \frac{\sigma^2}{(1-R_j^2)\sum_i (x_{ij}-\bar{x}_j)^2}$, 解释共线性如何膨胀方差。
3. 多重共线性下模型预测仍然很好, 为什么? 它主要伤害什么 (系数解释/标准误/t 检验/特征重要性)?
4. 处理共线性的方案对比: 剔除特征 vs PCA 降维 vs 岭回归, 各自代价与适用场景。
5. VIF、特征两两相关性和条件数 $\kappa(X) = \frac{\lambda_{max}}{\lambda_{min}}$ 三个指标的区别与联系, 树模型为什么不怕共线性?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/vif-multicollinearity?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vif-multicollinearity)

---

## 43. Viterbi Decoding · HMM 维特比解码

> The decoding problem asks for the most probable hidden state sequence $q^* = \arg\max_q P(q \mid O, \lambda)$ given $\lambda = (A, B, \pi)$ and observations $O$.

**Interview questions**

1. Write the Viterbi recursion $\delta_t(j) = \max_i[\delta_{t-1}(i)a_{ij}]b_j(o_t)$ and the backtracking procedure; what is the complexity?
2. How do Viterbi and the forward algorithm differ (sum vs max, their outputs, and why Viterbi needs the backpointer table $\psi$)?
3. Why does Viterbi guarantee a global optimum, unlike greedy per-position choice or beam search?
4. Work through a concrete example (e.g. $N=2$, $T=3$) by hand, deriving the best path and explaining backtracking.
5. How is Viterbi used in CRF/MEMM inference (the same decoding DP with different score functions)?

<details>
<summary>中文题目</summary>

1. 写出维特比递推 $\delta_t(j) = \max_i[\delta_{t-1}(i)a_{ij}]b_j(o_t)$ 及回溯过程,复杂度是多少?
2. 维特比与前向算法的区别?(求和 vs 取 max、各自输出什么、为什么需要前驱表 $\psi$)
3. 为什么维特比能保证全局最优?与贪心逐点选择、束搜索有何不同?
4. 给定具体例子(如 $N=2$、$T=3$),手推最优状态路径并说明回溯如何工作?
5. 维特比与 CRF/MEMM 推理有何联系(推断阶段共用同一 DP,只是势函数不同)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/viterbi?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=viterbi)

---

## 44. XGBoost 2nd-Order · XGBoost 二阶手推

> XGBoost is regularized second-order gradient boosting. Objective: $\mathcal{L} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i\right) + \sum_{k=1}^{K} \Omega(f_k)$ with tree regularization $\Omega(f) = \gamma T + \frac{1}{2} \lambda \Vert w \Vert^2$ ($T$ leaves, $w$ le…

**Interview questions**

1. Whiteboard XGBoost: from $\sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$, Taylor-expand to second order to get $\sum_i [g_i w_{q(x_i)} + \frac{1}{2} h_i w^2] + \lambda \Vert w \Vert^2$, then derive the optimal leaf weight $w_j^* = -\frac{G_j}{H_j + \lambda}$
2. Whiteboard the split gain $\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$: when do we not split?
3. Why second-order derivatives? Convergence advantage over first-order GBDT; what are $g_i$, $h_i$ under squared loss and cross-entropy?
4. Roles of $\gamma$ and $\lambda$: how do they affect leaf count and leaf weights? Relationship with tree pruning?
5. Engineering: exact greedy vs approximate quantile histograms, sparsity-aware splits (default direction for missing), column subsampling; differences vs LightGBM

<details>
<summary>中文题目</summary>

1. 白板手推 XGBoost: 从目标 $\sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$ 出发, 二阶泰勒展开得到 $\sum_i [g_i w_{q(x_i)} + \frac{1}{2} h_i w^2] + \lambda \Vert w \Vert^2$, 再推导最优叶子权重 $w_j^* = -\frac{G_j}{H_j + \lambda}$
2. 白板推导分裂结构增益 $\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$: 什么时候不分裂?
3. 为什么用二阶导? 与一阶 GBDT 相比的收敛优势; $g_i$, $h_i$ 在平方损失与交叉熵下分别是多少?
4. $\gamma$ 与 $\lambda$ 的作用: 复杂度惩罚如何影响叶子数与叶子权重? 与决策树剪枝的关系?
5. 工程机制: 精确贪心 vs 近似分位数直方图、稀疏感知分裂 (缺失值默认方向)、列采样; 与 LightGBM 的差异

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/ML/card/xgboost-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=xgboost-derivation)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
