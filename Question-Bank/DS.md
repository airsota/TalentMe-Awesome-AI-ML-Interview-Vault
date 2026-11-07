# 📈 DS Role Track · 数据科学家 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **25 cards · 125 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. A/A Testing & Hash Bucketing Uniformity · A/A 测试防膨胀与分流哈希均匀性

> A/A Testing (Placebo Experimentation) & Orthogonal Hash Bucketing provides baseline system certification ensuring experimental traffic splits are uniformly unbiased with non-inflated Type-I errors;

**Interview questions**

1. Explain how Kolmogorov-Smirnov (KS) tests verify that $p$-values generated across 1000 A/A tests follow uniform distribution $U(0, 1)$?
2. Explain Google's multi-layer orthogonal experimentation framework assigning unique layer salts to prevent inter-layer interference?
3. How to correct for pre-existing baseline imbalances using Difference-in-Differences or CUPED covariate adjustments?
4. Explain Re-randomization protocols where initial allocations are checked against balance thresholds before launch?
5. How to execute offline synthetic A/A simulations on historical telemetry to validate new hash bucketing configurations?

<details>
<summary>中文题目</summary>

1. 如何使用 Kolmogorov-Smirnov (KS) 检验验证 1000 次 A/A 实验生成的 $p$ 值序列是否严格服从 $U(0, 1)$ 均匀分布？
2. 多层正交实验分流架构 (Google Overlapping Experiment Infrastructure): 如何通过为每一层赋予独立 Salt (`hash(user_id + layer_salt)`) 实现跨层完全独立？
3. 桶间先天不平衡 (Pre-existing Baseline Differences): 如果 A/B 实验开始前发现两组在过去 30 天均值已有 0.5% 的差异，应如何利用差分法 (Difference-in-Differences) 或 CUPED 校正？
4. 重随机化 (Re-randomization): 在样本量较小的高风险实验中，如何通过在实验前检查协变量平衡性并拒绝不平衡分流方案？
5. 离线回放 A/A 测试 (Simulated / Offline A/A): 如何利用历史日志数据无需耗费真实流量即可在 1 分钟内完成上万次分流算法的 Monte Carlo 压力测试？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-aa-testing-bucketing-balance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-aa-testing-bucketing-balance)

---

## 2. Bayesian MMM & Media Saturation Curves · 贝叶斯营销组合模型 MMM 与饱和曲线

> Modern Bayesian Marketing Mix Modeling (MMM, powering Meta's LightweightMMM, Google Meridian, and Robyn) establishes aggregated top-down marketing attribution and budget optimization in privacy-first, cookie-less environments;

**Interview questions**

1. Derive Geometric and Weibull Adstock transformations modeling advertising carryover memory and lagged peak responses?
2. Derive the mathematical properties of the Hill Saturation Function and explain how parameters $K$ (half-saturation) and $S$ (slope) govern diminishing returns?
3. How to calibrate Bayesian MMM models by injecting empirical Geo-Lift causal experiment results as informative priors to resolve collinearity?
4. How does MMM decompose top-line sales into unprompted organic baseline, macroeconomic seasonality, and marketing-driven incremental volume?
5. How does the MMM Budget Optimizer utilize non-linear programming (SLSQP) to allocate media budgets along optimal marginal return frontiers?

<details>
<summary>中文题目</summary>

1. 写出几何衰减 Adstock 公式与 Weibull Adstock 公式的数学形式，并解释它们如何刻画广告记忆的持久性与延迟峰值？
2. 详细推导 Hill 饱和函数 $\text{Hill}(x; K, S) = \frac{x^S}{K^S + x^S}$ 的数学导数，并解释半饱和常数 $K$ 与形状参数 $S$ 如何决定渠道投资回报递减临界点？
3. 贝叶斯先验校准 (Bayesian Prior Calibration with Geo-Lift Experiments): 如何将区域提升实验 (Geo-Lift Test) 得出的真实因果 ROI 作为先验注入 MMM 模型以消除多重共线性？
4. 基础自然销量 (Baseline / Intercept Decomposition): 如何在 MMM 模型中剥离宏观经济趋势、节假日促销与价格变动，以分离出不依赖广告的自然有机销量？
5. 预算优化器 (Budget Optimizer): 如何在拟合好的 MMM 响应曲面上使用凸优化/非线性规划 (SLSQP) 计算各渠道在总预算 $B$ 下的最优分配点？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-marketing-mix-modeling-bayesian-mmm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-marketing-mix-modeling-bayesian-mmm)

---

## 3. Bootstrap Resampling & Empirical CI · Bootstrap 非参数重采样与置信区间

> Non-parametric Bootstrap Resampling & Empirical Confidence Intervals (Efron) provides a distribution-free resampling algorithm to estimate standard errors and confidence intervals for non-linear, non-normal summary statistics (e.g.

**Interview questions**

1. Write vectorized NumPy code executing $B=10,000$ Bootstrap resampling iterations to construct $95\%$ empirical confidence intervals for ratio metrics?
2. Contrast Percentile Bootstrap, Studentized Bootstrap, and BCa Bootstrap in adjusting for asymmetry and transformation bias?
3. Why is Cluster Bootstrap (sampling at the User ID level) mandatory when handling session-level repeated measurements to avoid correlation leakage?
4. Contrast Parametric Bootstrap (sampling from fitted parametric distributions) vs Non-parametric Bootstrap in sample efficiency and assumption risks?
5. How to construct a two-sample permutation null distribution via pooled resampling under $H_0$ to compute exact empirical $p$-values?

<details>
<summary>中文题目</summary>

1. 写出使用 Pure Python/NumPy 实现非参数 Bootstrap 计算比率指标（如 $\text{CTR} = \frac{\sum \text{Clicks}}{\sum \text{Impressions}}$）$95\%$ 置信区间的向量化代码逻辑？
2. 对比百分位数法 (Percentile Bootstrap)、标准误差正态近似法与偏差校正加速法 (BCa Bootstrap) 在处理偏态分布统计量时的理论精度？
3. 双层聚类数据 (Cluster-correlated Data / 用户与会话级多事件): 为什么必须使用 Block/Cluster Bootstrap（以用户 User ID 为单位整块重采样）而不是逐条日志独立重采样？
4. 参数化 Bootstrap (Parametric Bootstrap) vs 非参数化 Bootstrap (Non-parametric Bootstrap) 的核心差异与适用场景？
5. Bootstrap 在构建假设检验两样本差异 ($H_0: \theta_T = \theta_C$) 时的重居中零假设置换检验 (Permutation / Resampling under $H_0$) 步骤？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-bootstrap-resampling-confidence-intervals?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-bootstrap-resampling-confidence-intervals)

---

## 4. Causal Forest GRF & CATE Estimation · 因果森林 Causal Forest 与 CATE 估计

> Causal Forests & Generalized Random Forests (GRF, Athey & Wager, Stanford) establishes rigorous non-parametric estimation of Conditional Average Treatment Effects (CATE: $\tau(x) = \mathbb{E}[Y(1) - Y(0) | X=x]$) with asymptotic Gaussian normality;

**Interview questions**

1. Contrast standard Random Forest splitting criteria (MSE minimization) vs Causal Tree splitting criteria (treatment effect variance maximization)?
2. Explain why Honest Estimation (splitting structure selection from leaf evaluation) eliminates adaptive overfitting and restores asymptotic normality?
3. Derive how GRF transforms trees into adaptive kernel weighting functions $\alpha_i(x)$ solving local Generalized Method of Moments (GMM) equations?
4. How does Causal Forest compute variable importance metrics ranking which covariates drive the strongest treatment heterogeneity?
5. Explain Double Machine Learning (DML) residualization orthogonalizing outcomes and treatments to accelerate Causal Forest convergence?

<details>
<summary>中文题目</summary>

1. 详细对比标准随机森林 (Random Forest) 的分裂准则（均方误差 MSE 最小化）与因果树 (Causal Tree) 的分裂准则（因果效应方差最大化）？
2. 诚实树 (Honest Trees) 理论：为什么将样本拆分为“分裂集”与“估计集”能够消除自适应搜索导致的乐观过拟合偏置并恢复中心极限定理？
3. 广义随机森林 (Generalized Random Forests, GRF): 如何通过局部矩估计 (Local GMM) 与自适应核加权 $\alpha_i(x)$ 求解 CATE 优化问题？
4. 因果特征重要性 (Causal Feature Importance): 如何评估哪些用户特征（如年龄、历史活跃天数、设备价格）对干预效果的异质性驱动贡献最大？
5. Double Machine Learning (DML / EconML / Chernozhukov): DML 正交残差化与因果森林的结合机制（Robins-Frisch-Waugh 降维定理）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-causal-forest-grf-cate-estimation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-causal-forest-grf-cate-estimation)

---

## 5. Cohort Retention & LTV Lifetime Value · 用户留存队列分析与 LTV 终身价值

> Cohort Retention Analysis, Survival Analysis (Kaplan-Meier & Cox Proportional Hazards) & Probabilistic LTV Modeling (BG/NBD & Gamma-Gamma, Fader & Hardie) provides the mathematical foundations for evaluating product product-market fit, sustainable monetization…

**Interview questions**

1. Derive the BG/NBD generative probability process modeling latent transaction rate $\lambda$ (Gamma) and dropout probability $p$ (Beta)?
2. Why is an asymptotically flattening retention curve the defining mathematical hallmark of Product-Market Fit (PMF)?
3. Explain why healthy unit economics dictate $\text{LTV} / \text{CAC} \ge 3$ with CAC Payback periods $\le 12$ months?
4. Explain how multi-task deep neural architectures process early 7-day sequence embeddings to forecast 365-day cumulative LTV?
5. Explain how Right Censoring creates survival estimation bias and how Kaplan-Meier product-limit estimators correct for censored user tenures?

<details>
<summary>中文题目</summary>

1. 详细推导 BG/NBD (Beta-Geometric / Negative Binomial Distribution) 模型中活跃状态下的购买发生率与丢弃发生率的概率生成过程？
2. 留存曲线的渐近稳态 (Asymptotic Retention Plateau): 为什么一条最终“趋于水平”的留存曲线是验证产品具备 Product-Market Fit (PMF) 的黄金充要指标？
3. LTV / CAC 比率的黄金法则：为什么在 SaaS 与互联网消费金融中，通常要求 $\text{LTV} / \text{CAC} \ge 3$ 且 CAC 回本周期 (Payback Period) $\le 12$ 个月？
4. 基于深度学习的动态 LTV 预测 (Deep LTV via Transformers / Multi-task Learning): 如何将用户前 7 天的多行为序列特征编码用于预测 365 天长期消费？
5. 右删失问题 (Right Censoring in Survival Analysis): 为什么直接对未流失用户的历史观察时长取平均会产生严重向下低估偏置？Kaplan-Meier 如何解决删失问题？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-user-retention-cohort-ltv-modeling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-user-retention-cohort-ltv-modeling)

---

## 6. CUPED Variance Reduction Derivation · CUPED 方差缩减严格数学推导

> Controlled-experiment Using Pre-Experiment Data (CUPED, Deng et al. 2013, Microsoft) establishes optimal variance reduction for online experimentation, slashing required sample sizes and experiment durations by up to 50% without altering sample traffic;

**Interview questions**

1. Derive the mathematical proof of optimal $\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$ and variance reduction factor $(1 - \rho^2)$ via variance minimization calculus?
2. Why must covariate $X$ be strictly pre-treatment? Explain the disaster of post-treatment conditioning absorbing the true causal treatment effect?
3. Derive the Multivariate CUPED extension using multi-feature OLS regression projections to absorb orthogonal variance dimensions?
4. How should CUPED handle cold-start new users lacking pre-experiment history without violating unbiasedness (zero-imputation vs subgroup mean)?
5. Explain how modern ML-assisted variance reduction replaces linear $\theta X$ with non-linear GBDT predictions $\hat{Y} = f(X_{\text{pre}})$?

<details>
<summary>中文题目</summary>

1. 现场完整书写 CUPED 目标函数方差对 $\theta$ 的求导过程，严格推导出最优解 $\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$ 与方差缩减比例 $(1 - \rho^2)$？
2. 协变量 $X$ 的强外生性要求：为什么协变量 $X$ 必须严格在实验上线前采集？如果采集了实验期间的特征作为 $X$ 会导致什么严重的因果推断灾难（内生性吸收因果效应）？
3. 多协变量多元 CUPED 扩展：如何使用 OLS 线性回归 $\tilde{Y} = Y - X \beta$ 同时纳入多个历史特征（历史点击、历史留存、用户设备类型）？
4. 新用户冷启动无历史数据时的处理：针对实验期间注册的全新用户（无实验前 7 天数据），工程上应如何给其赋默认值（如赋 0 或总体均值 $\bar{X}$）而不破坏估计无偏性？
5. 非线性 CUPED 与 ML 辅助方差缩减 (Machine Learning-Assisted Variance Reduction / XGBoost 预测)：如何利用 GBDT 预测 $\hat{Y} = f(X_{\text{pre}})$ 构建残差？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-cuped-variance-reduction-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-cuped-variance-reduction-derivation)

---

## 7. DiD Parallel Trends & Event Study · 双重差分 DID 与平行趋势事件研究

> Difference-in-Differences (DiD) & Dynamic Event Study Methods estimate causal treatment effects for non-randomized policy interventions, city-level rollouts, or macro-marketing campaigns;

**Interview questions**

1. Derive how the OLS interaction coefficient $\beta$ in TWFE regression models algebraically equals the 4-cell difference-in-differences?
2. How does Event Study regression evaluate dynamic multi-period pre-trends and post-treatment lag effects?
3. Explain the Goodman-Bacon decomposition showing why classical TWFE produces negative weighting errors under staggered rollout timings?
4. Explain how Callaway & Sant'Anna (CS-DiD) estimator fixes staggered treatment heterogeneity using not-yet-treated or never-treated controls?
5. How to remediate non-parallel baseline trends using PSM-DiD or Synthetic Controls?

<details>
<summary>中文题目</summary>

1. 写出双向固定效应 (Two-Way Fixed Effects, TWFE) 回归方程，并严格推导为什么交互项系数 $\beta$ 恰好等于四个样本均值的双重差分？
2. 事件研究法 (Event Study Design): 如何通过引入滞后与前置指示变量 $\sum_{k=-K}^M \beta_k (\text{Treat}_i \times \mathbb{I}_{t = t_0 + k})$ 检验事前平行趋势并观察事后动态衰减？
3. 交错双重差分 (Staggered DiD / Goodman-Bacon 分解定理): 为什么当不同城市在不同时间点分批次逐步上线时，标准 TWFE 回归会产生严重的“负权重偏差 (Negative Weights)”？
4. 现代多期交错 DiD 修正估计器：Callaway & Sant'Anna (CS-DiD) 或 Sun & Abraham 估计器如何通过清洁对照组 (Clean Control Groups) 消除异质性偏差？
5. 当平行趋势假设不满足时（如处理组自然增长快于对照组），如何结合倾向得分匹配 (PSM-DiD) 或合成控制法 (Synthetic Control) 进行稳健性补救？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-did-parallel-trends-event-study?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-did-parallel-trends-event-study)

---

## 8. Hypothesis Testing, Type I/II & Power · 假设检验、两类错误与功效权衡

> Statistical Hypothesis Testing, Type I/II Error Trade-offs & Power Analysis formalizes the decision-theoretic framework for online A/B experimentation;

**Interview questions**

1. Draw the dual probability density curves of $H_0$ and $H_1$ highlighting exact geometric areas corresponding to $\alpha/2$, $\beta$, and Power ($1-\beta$)?
2. Why must product A/B tests enforce Two-Tailed tests even when positive metric lifts are anticipated (protecting against severe regressions)?
3. Contrast Statistical Significance ($p < 0.05$) vs Practical Business Significance (Cohen's $d$) under big-data sample regimes?
4. Why does Welch's $t$-test serve as the robust default over Student's $t$-test under unequal group sizes and heteroscedastic variances?
5. Prove the asymptotic equivalence between the 2-sample proportion $Z$-test and Pearson's Chi-Square Test for binomial ratio metrics?

<details>
<summary>中文题目</summary>

1. 现场画出 $H_0$ 零假设分布曲线与 $H_1$ 备择假设分布曲线，并用阴影准确标出 $\alpha/2$、$\beta$ 与 $\text{Power} = 1 - \beta$ 的几何区域？
2. 单尾检验 (One-Tailed Test) vs 双尾检验 (Two-Tailed Test): 为什么在大多数产品 A/B 测试中即使预期是正向提升也必须坚持使用双尾检验？
3. 效应量 (Effect Size / Cohen's $d = \frac{\mu_T - \mu_C}{\sigma_{\text{pooled}}}$) 与统计显著性 ($p < 0.05$) 的本质区别：为什么在大样本下极微小的无业务意义波动也会 $p$ 值显著？
4. 独立双样本 $t$ 检验 (Student's $t$) 与 Welch's $t$ 检验：当实验组与对照组样本量不相等或方差齐性不满足时，为什么应默认选择 Welch's $t$ 检验？
5. 比率类指标 (如点击率 CTR / 转化率 CVR) 的卡方检验 ($\chi^2$ Test) 与双样本比例 $Z$ 检验的渐近等价性证明？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-hypothesis-testing-power-analysis?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-hypothesis-testing-power-analysis)

---

## 9. Instrumental Variables & 2SLS Estimation · 工具变量法 IV 与两阶段回归 2SLS

> Instrumental Variables (IV, Angrist & Imbens Nobel Prize framework) & Two-Stage Least Squares (2SLS) uniquely identifies causal treatment effects in the presence of unobserved confounding, omitted variable bias, or bidirectional reverse causality;

**Interview questions**

1. Derive the 2SLS estimator equations and prove the algebraic equality of the Wald Estimator $\hat{\beta}_{\text{IV}} = \frac{\text{Cov}(Y, Z)}{\text{Cov}(D, Z)}$?
2. Explain why IV identifies LATE strictly for Compliers and does not identify effects for Always-takers or Never-takers under monotonicity?
3. Why does first-stage $F$-statistic $< 10$ signal weak instrument pathology causing severe finite-sample bias and unbounded asymptotic variance?
4. Explain how notification annoyance directly violates the Exclusion Restriction when using app push prompts as instruments for feature adoption?
5. Contrast Intention-to-Treat (ITT) measuring macro-business ROI vs Treatment-on-Treated (TOT/LATE) measuring pure algorithmic efficacy?

<details>
<summary>中文题目</summary>

1. 写出 2SLS 两阶段线性回归方程体系，并证明 Wald 估计量 $\hat{\beta}_{\text{IV}} = \frac{\text{Cov}(Y, Z)}{\text{Cov}(D, Z)} = \frac{\bar{Y}_1 - \bar{Y}_0}{\bar{D}_1 - \bar{D}_0}$ 的代数推导？
2. 局部平均因果效应 (LATE: Local Average Treatment Effect): 在存在 Compliers（遵从者）、Always-takers 与 Never-takers 时，为什么 IV 只能度量 Compliers 群体的因果增益？
3. 弱工具变量检验 (Weak Instrument Problem / Stock-Yogo Test): 为什么当一阶段回归 $F$ 统计量 $< 10$ 时，IV 估计量的方差和偏置会发生灾难性放大？
4. 排他性约束 (Exclusion Restriction) 违背案例：如果在 A/B 测试中以“弹窗通知 (Push Notification)”作为是否使用新功能的工具变量，为什么弹窗本身的打扰效应会直接破坏排他性？
5. 意向性治疗分析 (ITT: Intention-to-Treat) vs 治疗接受者效应 (Treatment-on-Treated, TOT): 为什么产品侧看 ITT（整体业务 ROI），算法侧看 TOT（算法纯净增益）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-instrumental-variables-2sls-lall?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-instrumental-variables-2sls-lall)

---

## 10. Meta-Learners: S- vs T- vs X-Learner · 元学习器 S/T/X-Learner 算法对比

> Meta-Learners for CATE Estimation (S-Learner, T-Learner, X-Learner & R-Learner, Künzel et al. & Nie et al.) formulate causal inference algorithms leveraging any standard regression model (LightGBM, XGBoost, Deep Nets) as base estimators;

**Interview questions**

1. Write the complete 4-step X-Learner algorithm detailing counterfactual residual imputation and propensity-weighted aggregation?
2. Why does X-Learner vastly outperform T-Learner under severe treatment-control sample imbalance (e.g. 1:100 ratio)?
3. Explain why GBDT greedy split selections and L1 regularization induce severe shrinkage bias on treatment effect predictions in S-Learner?
4. Derive R-Learner's objective via Robinson's transformation showing how residualizing outcomes and treatments yields Neyman Orthogonality?
5. How to deploy production-grade X-Learner and R-Learner pipelines using Microsoft EconML and Uber CausalML packages?

<details>
<summary>中文题目</summary>

1. 写出 X-Learner 的完整四步执行算法：1) 拟合 $\mu_0, \mu_1$；2) 交叉插补伪效应 $D_1, D_0$；3) 拟合残差模型 $\tau_1, \tau_0$；4) 用倾向得分 $e(X)$ 加权组合？
2. 为什么在处理组样本量极少（如 $N_T = 1,000$ vs $N_C = 100,000$）的极端不平衡场景下，X-Learner 能够大幅战胜 T-Learner？
3. S-Learner 的“正则化偏置 (Regularization Bias)”：为什么使用 L1 正则化（Lasso）或 GBDT 贪心树分裂时，处理变量 $T$ 很容易被完全忽略？
4. R-Learner 的 Robinson 变换推导：如何将 Neyman 正交性 (Neyman Orthogonality) 与残差化表达为加权平方损失 $\min_\tau \frac{1}{n} \sum (\tilde{Y}_i - \tau(X_i) \tilde{T}_i)^2$？
5. 因果包选型实战：如何使用微软 EconML 与 Uber CausalML 工业级因果推断库快速落地 X-Learner 与 R-Learner？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-meta-learners-s-t-x-learner?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-meta-learners-s-t-x-learner)

---

## 11. Metric Anomaly Root-Cause Drilldown · 指标异动根因归因与多维下钻拆解

> Metric Anomaly Root-Cause Diagnostics & Multi-Dimensional Drilldown provides systematic incident response methodologies when critical top-line KPIs fluctuate unexpectedly (e.g.

**Interview questions**

1. Deliver a structured incident investigation answering: 'App DAU dropped 5% this morning, walk through your complete root-cause diagnostic framework from macro to micro?'
2. Explain how Logarithmic Mean Divisia Index (LMDI) decomposes multiplicative metric shifts without residual terms?
3. Explain how user demographic mix shifts trigger Simpson's Paradox (every sub-channel conversion increases while total conversion drops)?
4. How do decision tree slicing and information gain heuristics isolate the highest-impact multi-dimensional anomaly clusters?
5. How to confirm that an isolated candidate dimension is the true causal driver rather than a correlated bystander via rollout telemetry?

<details>
<summary>中文题目</summary>

1. 现场模拟经典 DS 面试大题：“如果今天早上发现 App 大盘 DAU 突然下降了 5%，请给出你从宏观到微观的完整排查思路与下钻框架？”
2. 对数平均迪氏指数 (LMDI: Logarithmic Mean Divisia Index) 与拉氏因子分解：如何精确将乘法指标的变动百分比无残差分配给各个子因子？
3. 辛普森悖论 (Simpson's Paradox / Mix Shift 结构性变动): 为什么会出现“所有渠道的转化率都在上升，但大盘总转化率却在下降”的现象？如何使用标准化均值法拆解？
4. 多维下钻中的波浪式信息熵搜索 (Decision Tree-based Dimensional Slicing / Ripple Analysis): 如何在大规模稀疏维度中快速找出贡献度最大的最小属性集？
5. 因果归因 vs 统计相关：在定位出指标异动与某个发布版本强相关后，如何通过灰度切流与回归分析排除虚假伴生效应？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-metric-anomaly-root-cause-drilldown?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-metric-anomaly-root-cause-drilldown)

---

## 12. Multi-Touch Attribution: Markov & Shapley · 多触点归因 MTA：马尔可夫与 Shapley

> Multi-Touch Attribution (MTA) Frameworks (Markov Chain Removal Effects & Cooperative Game-Theoretic Shapley Values) evaluate the marginal fractional contribution of diverse marketing touchpoints (Search, Social, Display, Email, Influencer) along the customer c…

**Interview questions**

1. Derive the absorbing Markov Chain transition matrix formulation and step-by-step calculation of Removal Effects?
2. Derive the Shapley Value cooperative game formula and define its 4 governing axioms (Efficiency, Symmetry, Dummy, Additivity)?
3. How to resolve the exponential $O(2^N)$ computational complexity of exact Shapley values via Monte Carlo coalition sampling?
4. Explain why privacy regulations (Apple ATT, third-party cookie deprecation) break user-level MTA tracking and drive adoption of aggregated MMM?
5. How to calibrate algorithmic MTA models against ground-truth causal Geo-Lift experiments?

<details>
<summary>中文题目</summary>

1. 写出马尔可夫链归因中吸收态转移概率矩阵的构造方法，并详细推导移除效应 (Removal Effect) 的数学计算步骤？
2. 写出沙普利值 (Shapley Value: $\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} (v(S \cup \{i\}) - v(S))$) 的合作博弈公式，并说明四大公理？
3. 为什么在实际落地中，当触点渠道数量 $N > 15$ 时，Shapley 值的 $O(2^N)$ 计算复杂度会发生爆炸，如何通过 Monte Carlo 随机采样近似求解？
4. 隐私政策变革冲击（Apple ATT 弹窗与三方 Cookie 废除）对用户级路径跟踪的破坏：为什么行业正加速从 MTA 转向隐私安全宏观的 MMM (市场营销组合模型)？
5. 增量因果归因 (Incrementality-based Attribution): 如何通过增量实验 (Geo-Lift Experiment) 矫正 MTA 算法模型得出的归因系数？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-multitouch-attribution-shapley-markov?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-multitouch-attribution-shapley-markov)

---

## 13. Multiple Comparisons: FWER vs FDR-BH · 多重比较校正：Bonferroni vs FDR

> The Multiple Comparisons in Experimentation (FWER vs FDR-BH Corrections) framework formalizes decision protocols when evaluating multi-arm variants (A/B/n testing), multi-metric dashboards, or multidimensional subgroup drilldowns;

**Interview questions**

1. Contrast Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg (BH) across ranking algorithms, statistical conservatism, and Type-II error rates?
2. Write the complete algorithm for Benjamini-Hochberg step-up procedure determining the cutoff index $k^*$ such that $p_{(k^*)} \le \frac{k^*}{K} q$?
3. Map FWER vs FDR to business scenarios: mission-critical GMV safety vs marketing tag discovery vs multi-armed bandits?
4. Explain how post-hoc subgroup cherry-picking triggers both multiple testing bias and Simpson's Paradox?
5. Explain how Benjamini-Yekutieli (BY) generalizes FDR control under arbitrary positive dependence and correlation across metrics?

<details>
<summary>中文题目</summary>

1. 详细对比 Bonferroni、Holm-Bonferroni 与 Benjamini-Hochberg (BH) 在假设前提、排序步骤、检验保守性与第二类错误 (Type-II Error) 上的全方位差异？
2. 写出 Benjamini-Hochberg (BH) 算法的完整执行步骤：如何对 $K$ 个 $p$ 值升序排列、计算自适应阈值线并确定拒绝域截断点 $k^*$？
3. 业务场景选型：在以下三种场景中分别应选择 FWER 还是 FDR？A) 核心交易链路大盘 GMV 与故障率检测；B) 运营团队从 200 个用户标签中寻找高潜力转化人群；C) 推荐系统多臂老虎机 (MAB) 探索？
4. 多维下钻分析中的“辛普森悖论 (Simpson's Paradox)”与多重检验陷阱：为什么在细分人群中反复挑选显著群组会导致严重的样本选择偏差？
5. Benjamini-Yekutieli (BY) 校正在检验统计量存在正相关/任意复杂相关依赖性下的理论扩展？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-multiple-testing-fwer-fdr-bh?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-multiple-testing-fwer-fdr-bh)

---

## 14. Network Interference & Switchback Design · 网络溢出干扰与 Switchback 实验

> Network Interference, Spillover Effects & Switchback / Cluster Experimentation resolves SUTVA (Stable Unit Treatment Value Assumption) violations in two-sided marketplaces (Uber, DoorDash, Meituan) and social networks;

**Interview questions**

1. Explain the dual requirements of SUTVA (No Interference and No Hidden Variations) and why two-sided driver supply pools violate it?
2. Why must Switchback experiments enforce a 5-10 minute washout buffer between treatment flips to drain carryover state?
3. How to correct for temporal autocorrelation in Switchback time-block clusters using Cluster-Robust Standard Errors (CRSE)?
4. How to deploy Synthetic Controls when geographic randomization units are constrained to single-digit major metropolitan clusters?
5. Explain Two-Sided Randomization designs estimating both Direct and Spillover network effects simultaneously?

<details>
<summary>中文题目</summary>

1. 详细解释 SUTVA 稳定单元处理值假设的两大核心内涵（无干扰性 No Interference 与无变体版本），并分析为什么共享司机池直接违背该假设？
2. Switchback 实验设计中的“冲刷期 (Washout Period / Burn-in Buffer)”的作用：为什么在切换策略的前 5~10 分钟必须丢弃订单数据？
3. Switchback 实验样本量与方差估计中的“自相关问题 (Autocorrelation in Time-Series Clusters)”：为什么必须采用聚类稳健标准误 (Cluster-Robust Standard Errors) 或两阶段回归？
4. 合成对照在地理集群实验中的应用：当可供随机化的独立城市数量很少（如仅有 10 个核心一线城市）时，如何设计 Synthetic Control 准实验？
5. 直接效应 (Direct Effect) 与溢出效应 (Spillover / Indirect Effect) 的双随机化实验 (Two-sided Randomization): 如何同时估计直接效果与网络外溢程度？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-network-interference-switchback-cluster?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-network-interference-switchback-cluster)

---

## 15. North Star Metric & Guardrail Framework · 北极星指标设计与 Guardrail 护栏

> The North Star Metric (NSM) Strategic Architecture & Guardrail Metric Defense System connects empirical algorithms directly to sustainable enterprise business growth;

**Interview questions**

1. Derive the authentic North Star Metric and 3 critical Guardrail Metrics for Uber, TikTok, or Airbnb?
2. Construct an actionable Metric Hierarchy Tree decomposing top-line GMV into operational sub-driver levers?
3. Why must product teams align on actionable leading input metrics rather than lagging output financials?
4. How to apply Goodhart's Law to preemptively design defense mechanisms against metric gaming and artificial manipulation?
5. Explain Ron Kohavi's Overall Evaluation Criterion (OEC) combining conflicting metrics into a normalized decision objective function?

<details>
<summary>中文题目</summary>

1. 以具体业务（如 Uber 出行、抖音短视频、Airbnb 民宿）为例，详细推导其北极星指标是什么，并列出 3 个决定性的反向护栏指标 (Guardrail Metrics)？
2. 北极星指标分解树 (Metric Tree / DuPont Breakdown): 如何将大盘 GMV 拆解为 $\text{GMV} = \text{DAU} \times \text{转化率} \times \text{客单价} \times \text{人均单数}$ 的多层驱动因果树？
3. 输入指标 (Input/Leading Metrics) vs 输出指标 (Output/Lagging Metrics): 为什么产品团队日常考核必须看 Leading 指标而非直接看 Lagging 利润？
4. 古德哈特定律 (Goodhart's Law: 当一个指标变成目标时，它就不再是一个好指标): 在指标体系中如何设计反作弊与反刷量机制？
5. 综合整体评估准则 (OEC: Overall Evaluation Criterion / Kohavi 提出): 如何通过线性加权或非线性效用函数将多个冲突指标合成为单个决策分数？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-north-star-metric-guardrails?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-north-star-metric-guardrails)

---

## 16. Novelty, Primacy & Carryover Effects · 新奇效应、首因效应与长效隔离

> Novelty Effects, Primacy / Change Aversion & Long-term Carryover Effects formalize the behavioral and temporal dynamics separating transient experimental noise from sustainable steady-state business equilibria;

**Interview questions**

1. Explain how plotting longitudinal learning curves across New vs Existing user cohorts isolates novelty curiosity from genuine utility?
2. Why are Universal Long-term Holdouts essential for measuring cumulative compounding gains of dozens of marginal feature releases?
3. Explain how Reverse Experiments (shutting off a mature feature for 5% of users) verify durable causal retention impact?
4. Why must promotional experiments monitor post-intervention periods for 14-30 days to account for intertemporal demand shifting?
5. Explain Susan Athey's Surrogate Index methodology predicting 6-month user LTV from 2-week short-term behavioral proxies?

<details>
<summary>中文题目</summary>

1. 如何通过绘制“新用户 (New Users) vs 老用户 (Existing Users)”的时序指标学习曲线 (Cohort Learning Curves) 精确分离新奇效应与真实功能增量？
2. 长期保留对照组 (Universal Long-term Holdout Group: 长期保留 1%~5% 流量永远不发布新特性): 为什么它对评估多个微小改动的累积复合长期因果效应至关重要？
3. 反转实验 (Reverse Experiment / Turn-off Test): 在一个全量发布 3 个月的功能上，将 5% 用户切回旧版本，如何通过“关掉功能看跌幅”验证真实留存收益？
4. 促销补贴的蚕食效应 (Cannibalization) 与需求前置 (Intertemporal Substitution): 为什么必须在实验结束后继续追踪 14~30 天的净增量 GMV？
5. 替代指标 (Surrogate Metrics / Athey et al.): 如何构建短期可观测的代理指标在短短 2 周内高保真预测 6 个月后的长期用户 LTV 留存？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-novelty-primacy-carryover-effects?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-novelty-primacy-carryover-effects)

---

## 17. P-hacking, Peeking & mSPRT Sequential · P-hacking 偷窥与序贯检验 mSPRT

> The Continuous Monitoring / Peeking Problem & Mixture Sequential Probability Ratio Test (mSPRT) establishes rigorous martingales-based defenses against false-positive rate inflation during continuous experimental monitoring;

**Interview questions**

1. Derive why Brownian Motion first hitting times cause cumulative Type-I error rates to asymptotically approach 100% under continuous unadjusted peeking?
2. Derive the mathematical proof of mSPRT's error bounds using non-negative martingales and Doob's Maximal Inequality?
3. Explain how Lan-DeMets alpha-spending functions allocate Type-I error budgets conservatively across $K$ interim looks?
4. Why do Always-Valid Confidence Sequences guarantee $\ge 1-\alpha$ coverage probability uniformly across all stopping times?
5. When to deploy mSPRT for rapid disaster mitigation vs standard fixed-horizon testing for retention metrics with long carryover?

<details>
<summary>中文题目</summary>

1. 数学推导为什么布朗运动 (Brownian Motion) 的首达时间 (First Hitting Time) 会导致连续偷窥下的累积第一类错误随时间单调递增逼近 100%？
2. 详细推导 mSPRT (Mixture Sequential Probability Ratio Test) 中利用非负鞅 (Non-negative Martingale) 与 Doob 最大值不等式控制总体错误率 $\le \alpha$ 的数学证明？
3. Alpha-Spending 函数 (Lan-DeMets $\alpha$-spending / O'Brien-Fleming): 如何在预先设定的 $K$ 个检查节点上动态分配 $\alpha$ 预算？
4. 恒有效置信区间 (Always-Valid Confidence Intervals): 为什么其区间宽度随时间样本量增加逐渐收敛，且在任意停时 (Stopping Time $\tau$) 下覆盖真实效应的概率恒 $\ge 1-\alpha$？
5. 工业权衡：在业务实际中，什么时候应选用 mSPRT 序贯检验（如严重指标下跌快速止损），什么时候应坚持固定周期双周实验（如留存类长效指标）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-p-hacking-peeking-msprt?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-p-hacking-peeking-msprt)

---

## 18. Precision Marketing & Uplift ROI Budget · 精准营销触达定价与 ROI 预算分配

> Constrained Uplift Budget Allocation & Dynamic Pricing Optimization bridges individual CATE predictions to operational profit maximization under strict financial constraints;

**Interview questions**

1. Formulate coupon allocation as a fractional knapsack problem and prove greedy sorting by marginal efficiency $\frac{\text{CATE}_i}{\text{Cost}_i}$ is mathematically optimal?
2. Explain how binary search over Lagrangian dual multiplier $\lambda$ derives the exact cutoff threshold enforcing global budget constraint $B$?
3. How to resolve multi-treatment coupon tier selection by comparing incremental marginal efficiency slopes between adjacent cost tiers?
4. How to integrate contact frequency fatigue constraints and future LTV churn penalties directly into the objective function?
5. How does a PID control loop dynamically adjust shadow price $\lambda_t$ throughout the day to achieve uniform budget pacing across 24 hours?

<details>
<summary>中文题目</summary>

1. 将精准发券问题形式化为分数背包问题 (Fractional Knapsack Problem)，并证明按边际增益效率 $\frac{\text{CATE}_i}{\text{Cost}_i}$ 贪心排序是线性松弛下的最优解？
2. 拉格朗日对偶乘子 (Lagrangian Multiplier $\lambda$ / 影子价格 Shadow Price): 如何通过二分搜索或梯度下降快速求解满足全局预算约束 $\sum \text{Cost}_i \le B$ 的唯一截断阈值 $\lambda^*$？
3. 多 Treatment 决策矩阵：当同一个用户可以选择“不发券”、“发 5 元券”或“发 10 元券”时，如何根据边际递增增益 (Incremental Marginal CATE) 求解最优动作？
4. 疲劳度约束与长期流失惩罚：如何将用户频次控制 (Frequency Capping: 每周最多触达 2 次) 与短信骚扰导致的未来 LTV 惩罚项融入优化目标？
5. 在线动态预算平滑 (Online Budget Pacing): 在全天 24 小时流量持续到达时，如何使用 PID 控制器动态调整影子价格 $\lambda_t$ 防止预算在早高峰瞬间耗尽？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-precision-marketing-budget-allocation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-precision-marketing-budget-allocation)

---

## 19. PSM Matching & Doubly Robust Estimator · 倾向得分 PSM 与双重稳健估计器

> Propensity Score Matching (PSM), Inverse Probability Weighting (IPW) & Doubly Robust Estimators (AIPW, Robins et al.) eliminate confounding and selection bias from high-dimensional observational data;

**Interview questions**

1. Prove the Rosenbaum-Rubin dimension reduction theorem establishing conditional independence given 1D propensity score $e(X)$?
2. Derive Inverse Probability Weighting (IPW) estimator and explain extreme variance explosions when propensity scores approach 0 or 1?
3. Derive the AIPW estimator formula and prove why misspecification in either the propensity score or outcome model preserves unbiasedness?
4. How to audit propensity score overlap histograms and enforce common support trimming to prevent dangerous extrapolation?
5. Why is Standardized Mean Difference (SMD $< 0.1$) preferred over $p$-values to evaluate post-matching covariate balance?

<details>
<summary>中文题目</summary>

1. 详细证明 Rosenbaum-Rubin 降维定理：为什么在 $Y(0), Y(1) \perp T \mid X$ 成立时，必有 $Y(0), Y(1) \perp T \mid e(X)$？
2. 写出逆概率加权 (IPW: $\hat{\tau}_{\text{IPW}} = \frac{1}{n} \sum \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)} \right]$) 公式，并分析为什么当 $e(X) \to 0$ 或 $1$ 时方差会发生剧烈爆炸？
3. 写出双重稳健估计器 (Augmented IPW / AIPW) 的完整代数形式，并证明其“双重保险 (Double Robustness)”的数学无偏性机制？
4. 共同支撑域 (Common Support / Overlap): 如何通过直方图检查倾向得分分布重叠，并剔除离群极端样本以防外推风险？
5. 协变量平衡性检验 (Covariate Balance Check): 为什么匹配完成后必须计算标准化均值差 (SMD: Standardized Mean Difference $< 0.1$) 而不是看 $t$ 检验 $p$ 值？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-propensity-score-matching-doubly-robust?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-propensity-score-matching-doubly-robust)

---

## 20. Qini Curve, AUUC & Uplift Evaluation · Qini 曲线、AUUC 评估与增益标定

> Uplift Model Offline Evaluation Metrics (Qini Curve, Qini Score, AUUC - Area Under Uplift Curve & Cumulative Gain Charts) solve the fundamental evaluation challenge that individual counterfactual treatment effects are unobservable, rendering classical ROC-AUC/…

**Interview questions**

1. Derive the Qini curve formula and explain why the scaling factor $\frac{N_t(k)}{N_c(k)}$ is required to adjust for treatment-control sample imbalance?
2. Contrast Qini Curve vs AUUC in coordinate normalization and cumulative percentage definitions?
3. How to construct the random baseline and the theoretical Oracle upper bound on Qini visualization plots?
4. Why must a calibrated Uplift model exhibit strict monotonic degradation across sorted decile bins?
5. How to perform Uplift Calibration checks comparing predicted CATE against empirical bucketed treatment-control rate margins?

<details>
<summary>中文题目</summary>

1. 写出计算累积 Qini 曲线的数学公式 $Q(k) = Y_{t}(k) - \frac{N_t(k)}{N_c(k)} Y_{c}(k)$，并说明为什么需要乘以样本量缩放因子 $\frac{N_t(k)}{N_c(k)}$？
2. 详细对比 Qini 曲线与 AUUC (Cumulative Uplift Curve) 在纵坐标定义与归一化方式上的数学差异？
3. 随机基线 (Random Baseline) 与理想最优基线 (Oracle Upper Bound): 如何在 Qini 曲线图中绘制随机分流的对角线与理论完美的最高收益天花板？
4. Decile Uplift 分组增益柱状图 (Lift by Decile Chart): 为什么一个优秀 Uplift 模型的 Decile 柱状图必须呈现严格的“从左到右单调递减”分布？
5. 校准性评估 (Uplift Calibration): 如何检验模型预测的增益值 $\hat{\tau}(X) = 0.05$ 与实际 A/B 测试观测到的分组真实增益 $(p_T - p_C)$ 是否精准吻合？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-qini-curve-auuc-uplift-metrics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-qini-curve-auuc-uplift-metrics)

---

## 21. Rubin Potential Outcomes & DAG d-Separation · Rubin 潜在结果与因果 DAG d-分离

> The Dual Foundations of Causal Inference (Donald Rubin's Potential Outcomes Framework & Judea Pearl's Causal DAGs / $d$-Separation) provides mathematical identification of causal treatment effects from purely observational data when randomized A/B tests are im…

**Interview questions**

1. Derive Pearl's Backdoor and Frontdoor criteria with graph-theoretic $d$-separation blocking conditions?
2. Diagram Collider Bias (Berkson's Fallacy) and illustrate how conditioning on a collider creates spurious negative correlations?
3. Explain the 3 foundational assumptions of Potential Outcomes: Ignorability, Positivity/Overlap, and SUTVA in practice?
4. Why does conditioning on post-treatment mediators in regression models attenuate and bias the total causal treatment effect?
5. Contrast Pearl's causal intervention distribution $P(Y | do(X=x))$ with the observational conditional distribution $P(Y | X=x)$ using do-calculus?

<details>
<summary>中文题目</summary>

1. 深入推导 Pearl 后门准则 (Backdoor Criterion) 与前门准则 (Frontdoor Criterion) 的数学公式与图拓扑阻断条件？
2. 详细画出对撞偏置 (Collider Bias / Berkson's Fallacy) 的经典图拓扑，并用具体业务案例（如‘为什么才华横溢和高颜值在好莱坞明星中呈现负相关’）解释其产生机理？
3. 潜在结果框架的三大核心假定：无混杂性 (Unconfoundedness / Ignorability)、重叠性/正权性 (Overlap / Positivity) 与 SUTVA 的现实业务意义？
4. Bad Controls（坏控制变量）识别：为什么在回归方程中控制了处于因果链中介位置的变量 (Mediator) 会导致真实因果效应被严重低估？
5. Do-Calculus ($do(X=x)$ 算子): 为什么 $P(Y | do(X=x))$（主动干预分布）与 $P(Y | X=x)$（被动条件观测分布）存在本质代数差异？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-rubin-potential-outcomes-dag-d-separation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-rubin-potential-outcomes-dag-d-separation)

---

## 22. Sample Size Derivation & MDE Budget · 样本量公式严格推导与 MDE 估算

> The A/B Test Minimum Sample Size Formula & MDE Derivation establishes the exact sample requirements per variant under significance $\alpha$ and power $1 - \beta$;

**Interview questions**

1. Derive the step-by-step mathematical proof of sample size $n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$ from normal distribution quantile critical boundaries?
2. Explain why halving the target effect size $\delta$ quadruples ($4\times$) required sample size due to the inverse-square law?
3. Derive effective sample size $n_{\text{eff}} = \frac{2 n_T n_C}{n_T + n_C}$ under skewed 90/10 traffic splits and quantify statistical power loss?
4. Why must online experiments run for integer weekly cycles (7 or 14 days) regardless of reaching sample size early to absorb weekly seasonality?
5. How to resolve sample size explosions on heavy-tailed revenue metrics via Winsorization, log transforms, or CUPED variance reduction?

<details>
<summary>中文题目</summary>

1. 从正态分布分位数临界值定义出发，现场手写完整推导出样本量公式 $n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$ 的每一步几何代数变换？
2. 当想要检测的提升幅度 $\delta$ 缩小一半（例如从 $2\%$ 缩小到 $1\%$）时，为什么所需样本量不是翻倍，而是剧增为原来的 4 倍（平方反比律）？
3. 不平衡分流 (Unequal Traffic Allocation: 如 90% 对照 vs 10% 实验): 如何使用调和平均数公式计算等效有效样本量 $n_{\text{eff}} = \frac{2 n_T n_C}{n_T + n_C}$？
4. 实验运行周期计算：为什么 A/B 测试通常必须跑满至少 1 到 2 个完整周（7~14 天），以消除“周内效应 (Day-of-Week Seasonality)”？
5. 长尾高方差指标（如用户消费金额 Revenue / GMV）的样本量爆炸问题：如何通过对数变换、极端值截断 (Winsorization) 或 CUPED 压缩方差？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-sample-size-formula-mde-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-sample-size-formula-mde-derivation)

---

## 23. SRM Detection & Chi-Square Diagnostics · 样本比例失衡 SRM 卡方检测排查

> Sample Ratio Mismatch (SRM) Detection & Root-Cause Diagnostics serves as the primary health-check guardrail in online experimentation platforms;

**Interview questions**

1. Derive the simplified Chi-Square statistic $\chi^2 = \frac{(N_T - N_C)^2}{N_T + N_C}$ for symmetric 50/50 split SRM verification?
2. Detail the 4 root-cause categories of SRM: Tracking Loss, Crash Dropouts, Latency Timeouts, and Bot Skew with diagnostic steps?
3. Explain how latency-induced dropouts create Survivorship Bias, converting a disastrous UX regression into a phantom metric lift?
4. Why must user randomization logs trigger precisely at feature exposure rather than generic app launch to prevent dilution and SRM?
5. Why does industry strictly mandate discarding SRM-compromised experiments rather than attempting post-hoc statistical weighting corrections?

<details>
<summary>中文题目</summary>

1. 写出计算 50/50 分流下 SRM 卡方统计量 $\chi^2 = \frac{(N_T - N/2)^2}{N/2} + \frac{(N_C - N/2)^2}{N/2} = \frac{(N_T - N_C)^2}{N_T + N_C}$ 的简化公式？
2. SRM 排查四大根因分类：1) 埋点丢失 (Tracking Loss)；2) 客户端崩溃过滤 (Crash Bias)；3) 延迟超时 (Performance Dropout)；4) 爬虫机器人过滤 (Bot Skew)，如何逐一排查？
3. 为什么当实验组网页加载变慢导致 SRM 时，简单的均值比较会产生“幸存者偏差 (Survivorship Bias)”从而得出完全相反的错误结论？
4. 触发事件后分流 (Trigger-based Analysis): 为什么必须在进入实验触发点 (Trigger Point) 时才进行样本分配记录，而不是在全量 App 启动时记录？
5. 当排查发现 SRM 确实存在时，是否可以通过“倾向得分加权 (Propensity Weighting)”挽救数据？为什么工业界金标准坚持直接报废并重新修复上线？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-srm-sample-ratio-mismatch-detection?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-srm-sample-ratio-mismatch-detection)

---

## 24. Synthetic Control Method & Placebo Checks · 合成控制法 SCM 与安慰剂空间检验

> The Synthetic Control Method (SCM, Abadie et al.) & Placebo Permutation Inference constructs data-driven synthetic counterfactuals for causal policy evaluation when an intervention impacts strictly a single aggregate unit (e.g.

**Interview questions**

1. Formulate the constrained convex quadratic optimization problem minimizing pre-intervention predictor distance subject to $w_j \ge 0, \sum w_j = 1$?
2. Explain how in-space placebo iterations across all donor units construct empirical Post/Pre-RMSPE ratios to compute exact permutation $p$-values?
3. Why does re-running SCM on an artificial pre-intervention placebo date evaluate model specification validity?
4. Explain Synthetic Difference-in-Differences (SDID) combining both unit weights and time weights to handle fixed level offsets?
5. How does economic spillover between donor units and treated units contaminate the synthetic counterfactual, and how to filter donor pools?

<details>
<summary>中文题目</summary>

1. 写出合成控制法求解权重向量 $W^*$ 的受约束二次规划 (Constrained Quadratic Programming) 凸优化数学公式？
2. 空间安慰剂检验 (In-Space Placebo Test) 与均方预测误差比值 (Post/Pre-RMSPE Ratio): 如何利用置换经验分布计算类似于 $p$-value 的统计显著性？
3. 时间安慰剂检验 (In-Time Placebo Test): 为什么将干预发生时间点人为虚构提前到干预发生前 2 年并重新运行 SCM 能检验模型是否过拟合？
4. 矩阵补全因果推断与合成双重差分 (Synthetic DiD / Arkhangelsky et al.): 如何结合单位权重与时间权重克服传统 SCM 对水平平移不变量的敏感？
5. 供体池污染 (Donor Pool Contamination): 如果供体池中的某些城市与处理城市存在经济外溢或人口流动交互，会导致什么偏差？如何清洗供体池？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-synthetic-control-method-placebo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-synthetic-control-method-placebo)

---

## 25. Uplift Modeling & 4-Quadrant Profiling · Uplift 增益建模与用户四象限细分

> Uplift Modeling (Heterogeneous Treatment Effect Modeling) & 4-Quadrant User Segmentation transforms precision marketing and algorithmic discounting from predicting baseline likelihoods to predicting individual causal increments (ITE / CATE:…

**Interview questions**

1. Diagram the 4-Quadrant Uplift Matrix ($Y(0)$ vs $Y(1)$) and define optimal budget allocation rules across Persuadables, Sure Things, Lost Causes, and Sleeping Dogs?
2. Contrast Response Models (predicting $P(Y=1|X)$) vs Uplift Models (predicting $\mathbb{E}[Y(1)-Y(0)|X]$) in business objective functions?
3. Why is a clean randomized A/B trial dataset required as the ground truth training set for supervised Uplift modeling?
4. Derive the optimal profit-maximizing treatment policy threshold $\tau(X) > \frac{C}{V}$ balancing coupon cost $C$ against gross margin $V$?
5. How to formulate Multi-Treatment Uplift models to optimize continuous discount elasticities across multiple denomination tiers?

<details>
<summary>中文题目</summary>

1. 画出 Uplift 用户四象限矩阵图（横轴：不干预行为 $Y(0)$，纵轴：干预行为 $Y(1)$），标出四类人群并说明营销预算的最优分配策略？
2. 响应模型 (Response Model / 预测点击率 CTR) vs 增益模型 (Uplift Model / 预测因果增益 $\tau(X)$) 在业务目标函数上的本质区别？
3. 为什么在训练 Uplift 模型前，必须通过严格的随机化 A/B 实验（随机分流 Treatment 与 Control）收集训练样本？
4. 净增量收益公式：如果一张优惠券成本为 $C$，客单价利润为 $V$，推导最优发券决策阈值 $\tau(X) > \frac{C}{V}$？
5. 多价格阶梯与多 Treatment Uplift 扩展：如何对不同面额优惠券（如 5元、10元、20元）进行多分支连续增益弹性建模？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DS/card/ds-uplift-modeling-four-quadrants?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ds-uplift-modeling-four-quadrants)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
