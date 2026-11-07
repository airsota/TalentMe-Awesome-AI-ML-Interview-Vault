# 📐 Math & Optimization · 数理基础与优化 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **32 cards · 160 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. Adam/AdamW Math · Adam/AdamW 偏差修正推导

> Adam = exponential moving averages of the first moment $m_t$ and second moment $v_t$, plus bias correction. With gradient $g_t$: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$.

**Interview questions**

1. Write Adam's moment recursions, expand $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$ and explain how historical weights decay.
2. Prove the bias: sum the geometric series to show $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$ and explain why the bias is worst at $t=1$ ($1-\beta_1 = 0.1$).
3. Why divide by $1-\beta_1^t$? Prove $\mathbb{E}[\hat m_t] = \mathbb{E}[g]$ after correction; why must the second moment $\hat v_t$ be corrected and square-rooted?
4. Compare Adam+L2 vs AdamW: how does dividing $\lambda w_t$ by $\sqrt{\hat v_t}$ distort the effective regularization, and what does decoupling fix?
5. Write the full update $w_{t+1} = w_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$; why does $\epsilon$ belong in the denominator rather than the numerator?

<details>
<summary>中文题目</summary>

1. 写出 Adam 的矩估计递归式,并把 $m_t$ 展开为 $m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$,说明历史权重如何衰减。
2. 证明偏差: 用等比数列求和证 $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$,并说明为什么 $t=1$ 时偏差最严重($1-\beta_1 = 0.1$)。
3. 为什么偏差修正要除以 $1-\beta_1^t$?证明修正后 $\mathbb{E}[\hat m_t] = \mathbb{E}[g]$;二阶矩 $\hat v_t$ 为何也要修正且要开方。
4. 对比 Adam+L2 与 AdamW: $\lambda w_t$ 被 $\sqrt{\hat v_t}$ 缩放会怎样扭曲有效正则强度?解耦 weight decay 的意义是什么?
5. 写出完整更新 $w_{t+1} = w_t - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$;为什么 $\epsilon$ 放在分母而不是分子?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/adamw-math?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=adamw-math)

---

## 2. Bayesian Inference · 贝叶斯推断

> Bayesian inference treats the parameter $\theta$ as a random variable and combines the prior $p(\theta)$ with the likelihood $p(x|\theta)$ into a posterior distribution:…

**Interview questions**

1. Write the Bayesian posterior formula and explain each term? How does the posterior change with data size?
2. Differences among MLE, MAP and Bayesian inference? What does MAP estimate from the posterior?
3. What is a conjugate prior? Give the Beta-Binomial and Normal-Normal examples.
4. Bayesian advantages over MLE with small samples? How does the prior act as regularization?
5. Approximation methods when the posterior is intractable (MCMC, variational inference, Laplace approximation)?

<details>
<summary>中文题目</summary>

1. 写出贝叶斯后验公式并解释每一项?后验如何随数据量变化?
2. MLE、MAP 与贝叶斯推断的区别?MAP 是后验的什么估计?
3. 什么是共轭先验?举 Beta-二项与正态-正态两个例子。
4. 小样本下贝叶斯相对 MLE 的优势?先验如何起到正则化的作用?
5. 后验积分无解析解时有哪些近似方法(MCMC、变分推断、拉普拉斯近似)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/bayesian-inference?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bayesian-inference)

---

## 3. Bias-Variance Decomposition · 偏差方差分解

> The bias-variance decomposition splits expected squared test error into three terms: $\text{EGE} = \text{Bias}^2 + \text{Var} + \sigma^2$.

**Interview questions**

1. Whiteboard-prove $\mathbb{E}[(y-\hat f)^2] = \text{Bias}^2 + \text{Var} + \sigma^2$: show the expansion and both cross-term cancellations ($\mathbb{E}[\epsilon] = 0$ and $\mathbb{E}_D[\hat f] = \bar f$).
2. Why is the cross term $2\mathbb{E}[\epsilon(f^* - \hat f)]$ zero? Which independence/expectation assumptions does it rely on? Why is $\sigma^2$ irreducible?
3. As model complexity grows, how do Bias and Var change? Which term dominates in underfitting vs overfitting?
4. Explain via the decomposition: why does bagging mainly reduce variance, and boosting mainly reduce bias?
5. Why does the decomposition fail for classification with 0-1 loss? What alternative (e.g. Brier score) can be used?

<details>
<summary>中文题目</summary>

1. 白板证明 $\mathbb{E}[(y-\hat f)^2] = \text{Bias}^2 + \text{Var} + \sigma^2$:写出展开、两次交叉项消去($\mathbb{E}[\epsilon] = 0$ 与 $\mathbb{E}_D[\hat f] = \bar f$)的每一步。
2. 交叉项 $2\mathbb{E}[\epsilon(f^* - \hat f)]$ 为什么为 0?依赖哪些独立性/期望假设?噪声 $\sigma^2$ 为什么不可约?
3. 模型复杂度增大时 Bias 与 Var 各自如何变化?欠拟合与过拟合分别由哪一项主导?
4. 用分解式解释: bagging 为什么主要降方差?boosting 为什么主要降偏差?
5. 该分解对分类(0-1 损失)为什么不成立?可以用什么替代(Brier score)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/bias-variance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bias-variance)

---

## 4. Bootstrap · Bootstrap

> Bootstrap estimates the sampling distribution of a statistic $\hat\theta = g(x_1, \dots, x_n)$ by resampling with replacement: independently $B$ times, draw $n$ observations from the original sample and compute $\hat\theta^{*b}$ ($b = 1, \dots, B$);

**Interview questions**

1. Full procedure and formula for estimating $\mathrm{Var}(\hat\theta)$ by bootstrap? Why is it "using the empirical distribution to approximate the true distribution"?
2. How is the percentile interval constructed? How does it differ from the normal approximation $\hat\theta \pm z_{\alpha/2} \widehat{\mathrm{SE}}$?
3. Bootstrap vs CLT: when are they equivalent, and when is bootstrap superior (no analytic SE, small samples)?
4. Parametric vs non-parametric bootstrap: what are the differences?
5. Assumptions and failure modes of bootstrap (independence, information ceiling of resamples, time series)?

<details>
<summary>中文题目</summary>

1. bootstrap 重采样估计 $\mathrm{Var}(\hat\theta)$ 的完整步骤与公式?为什么说它是“用经验分布近似真实分布”?
2. 百分位置信区间如何构造?与正态近似区间 $\hat\theta \pm z_{\alpha/2} \widehat{\mathrm{SE}}$ 有何区别?
3. bootstrap 与 CLT 的关系:何时两者等价?bootstrap 何时更优?
4. 参数化 bootstrap 与非参数 bootstrap 的区别?
5. bootstrap 的假设与失效场景(独立性、重采样集的信息上限、时间序列)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/bootstrap?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bootstrap)

---

## 5. Causal Inference (Rubin) · 因果推断与 Rubin 框架

> Causal inference under the Rubin potential-outcomes framework defines effects counterfactually: the potential outcomes of an individual under treatment $T=1$ and control $T=0$ are $Y(1)$ and $Y(0)$, and the average treatment effect is…

**Interview questions**

1. Define ATE under potential outcomes. Why can't observed data estimate it directly?
2. Confounders vs colliders? Why does conditioning on a collider introduce bias?
3. Role of the propensity score and the full four-step PSM pipeline?
4. Give a correlation ≠ causation example and state the ignorability assumption.
5. PSM vs direct regression adjustment: pros and cons? How to run balance checks?

<details>
<summary>中文题目</summary>

1. 潜在结果框架下 ATE $= \mathbb{E}[Y(1) - Y(0)]$ 的定义?为什么观测数据无法直接估计?
2. 混淆变量与碰撞变量的区别?控制碰撞变量为什么反而引入偏差?
3. 倾向得分 $e(x) = P(T=1 \mid x)$ 的作用与 PSM 的完整四步流程?
4. 举一个相关性 ≠ 因果性的例子, 并说明可忽略性 (ignorability) 假设的含义?
5. PSM 相比直接对协变量回归调整的优缺点?如何做平衡性检验?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/causal-inference?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=causal-inference)

---

## 6. Central Limit Theorem · 中心极限定理 CLT

> The Central Limit Theorem (CLT) states that for i.i.d. samples $X_1,\dots,X_n$ with finite mean $\mu$ and variance $\sigma^2 < \infty$, the standardized sample mean converges in distribution to a standard normal:…

**Interview questions**

1. State the CLT. What conditions are required (i.i.d., finite variance)?
2. Why does the standardized mean converge at rate $1/\sqrt{n}$? Explain via fluctuation cancellation.
3. What is the standard error of $\bar{X}$? How many more samples halve the error?
4. How does the CLT differ from and relate to Chebyshev's inequality and the LLN?
5. Why does the CLT fail for infinite variance (e.g. Cauchy)? What happens to the sample mean?

<details>
<summary>中文题目</summary>

1. 中心极限定理的内容是什么?成立需要哪些条件(i.i.d.、方差有限)?
2. 为什么标准化均值的收敛速度是 $1/\sqrt{n}$?用波动抵消直觉解释。
3. 样本均值 $\bar{X}$ 的标准误是多少?要让误差减半需要多少倍样本?
4. CLT 与切比雪夫不等式、大数定律的区别与联系是什么?
5. 方差不存在时(如 Cauchy 分布)CLT 为何失效?样本均值收敛到哪里?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/clt?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=clt)

---

## 7. Common Distributions · 常用分布族

> Common distribution families with parameter-mean-variance table: ① Normal $\mathcal{N}(\mu,\sigma^2)$: pdf $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, mean $\mu$, variance $\sigma^2$;② Bernoulli $\text{Bern}(p)$: $P(X=1) = p$, mean…

**Interview questions**

1. Write the parameters, mean and variance of Normal, Bernoulli, Binomial, Poisson and Exponential?
2. Relations among Binomial, Poisson and Normal (limits and approximations)?
3. What is the memoryless property of the Exponential? How does it model waiting times?
4. Why Bernoulli (cross-entropy) for classification and Normal (MSE) for regression — from the likelihood perspective?
5. Poisson has mean = variance; what do you do with over-dispersed count data?

<details>
<summary>中文题目</summary>

1. 写出正态、伯努利、二项、泊松、指数分布的参数、期望与方差表?
2. 二项、泊松、正态三者之间有什么关系(极限与逼近)?
3. 指数分布的无记忆性是什么?如何用它建模等待时间?
4. 为什么二分类用伯努利(交叉熵)损失、回归用正态(MSE)损失?从似然角度解释。
5. 泊松分布均值等于方差;遇到过离散 (over-dispersion) 的计数数据该怎么办?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/distributions?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distributions)

---

## 8. Confidence Intervals · 置信区间

> A confidence interval is a random interval constructed from the sample; for a normal mean the classic form is $CI = \bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$, with $z_{\alpha/2}$ the standard-normal quantile and $\sigma/\sqrt{n}$ the standard error — w…

**Interview questions**

1. The 95% CI $\bar{x} \pm z_{\alpha/2}\sigma/\sqrt{n}$ for a normal mean with known variance: meaning of each component?
2. Strict coverage interpretation: why can't we say the true value lies in the interval with 95% probability?
3. How does the width change when confidence goes 95%→99%, or when $n$ doubles?
4. Why the t-distribution when $\sigma$ is unknown? Difference from the normal approximation?
5. What is the duality between confidence intervals and two-sided hypothesis tests?

<details>
<summary>中文题目</summary>

1. 正态总体均值、方差已知时 95% CI 公式 $\bar{x} \pm z_{\alpha/2}\sigma/\sqrt{n}$ 各部分的含义?
2. 覆盖率 (coverage) 的严格解释: 为什么不能说“真值有 95% 概率在区间内”?
3. 置信水平从 95% 提到 99%, 或样本量翻倍, 区间宽度分别如何变化?
4. 未知 $\sigma$ 时为什么用 $t$ 分布?与正态近似的区别?
5. 置信区间与双侧假设检验的对偶关系是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/confidence-interval?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=confidence-interval)

---

## 9. Convex Optimization & KKT · 凸优化与 KKT

> Convex optimization studies problems whose objective and feasible set are both convex ($f(\theta x_1 + (1-\theta)x_2) \le \theta f(x_1) + (1-\theta)f(x_2)$ for all $\theta \in [0,1]$, so any local optimum is global).

**Interview questions**

1. Write the Lagrangian $L = f + \sum_i\lambda_i h_i + \sum_j\mu_j g_j$ and all four KKT conditions (stationarity / primal / dual / complementary slackness) with intuition for each.
2. Intuition of complementary slackness $\mu_j g_j = 0$: why must $\mu_j = 0$ when the constraint is inactive? Give an example where the constraint is active on the boundary.
3. Solve $\min \frac{1}{2}(x-2)^2$ s.t. $x \le 1$ via KKT: write all equations, discuss the $\mu = 0$ branch, and solve.
4. When are KKT conditions necessary and sufficient? What is the relation between Slater's condition and strong duality $f^* = d^*$?
5. In hard-margin SVM duality, which constraints are active? How do support vectors connect to complementary slackness $\mu_j g_j = 0$?

<details>
<summary>中文题目</summary>

1. 写出拉格朗日函数 $L = f + \sum_i\lambda_i h_i + \sum_j\mu_j g_j$ 与四条 KKT 条件(平稳性/原始可行/对偶可行/互补松弛),并解释每条的含义。
2. 互补松弛 $\mu_j g_j = 0$ 的几何含义: 为什么约束不起作用时乘子必须为 0?举一个约束在边界上起作用的例子。
3. 用 KKT 求解 $\min \frac{1}{2}(x-2)^2$ s.t. $x \le 1$:写出全部 KKT 方程、讨论 $\mu = 0$ 分支并求解。
4. KKT 在什么条件下是充要条件?Slater 条件与强对偶 $f^* = d^*$ 的关系是什么?
5. 硬间隔 SVM 的对偶推导中,哪些约束是“起作用”的?支持向量与互补松弛 $\mu_j g_j = 0$ 如何联系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/convex-kkt?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=convex-kkt)

---

## 10. Double Descent · Double Descent

> Double descent (Belkin et al. 2019) describes test error evolving with model capacity/parameter count in two descending segments.

**Interview questions**

1. Sketch the double-descent curve: mark the underparametrized regime, interpolation threshold and overparametrized regime; contrast with the classical U-shape.
2. Why is error worst near the interpolation threshold ($p \approx n$)? What is the model doing to training noise there?
3. Why does generalization improve in the overparametrized regime? What is the link between min-norm interpolation and implicit regularization?
4. Why don't large models (params ≫ samples) degrade along the U-shape? How does this connect to scaling laws?
5. Does double descent violate the no-free-lunch theorem? How does it reconcile with the bias-variance decomposition?

<details>
<summary>中文题目</summary>

1. 画出 double descent 曲线:标注欠参数化区、插值阈值、过参数化区三段,并与经典 U 形曲线对比。
2. 为什么插值阈值($p \approx n$)附近误差最大?此时模型对训练噪声做了什么?
3. 过参数化区为什么泛化反而变好?最小范数插值解与隐式正则化的关系是什么?
4. 为什么大模型(参数量 ≫ 样本数)没有按 U 形曲线变差?与 scaling law 的联系是什么?
5. double descent 违反“无免费午餐”定理吗?它与偏差-方差分解如何调和?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/double-descent?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=double-descent)

---

## 11. Entropy, KL & Cross-Entropy · 熵 / KL 散度 / 交叉熵

> Entropy $H(X) = -\sum_x p(x)\log p(x)$ measures uncertainty (expected bits of an optimal code); KL divergence measures the discrepancy between two distributions, $D_{KL}(P \Vert Q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$, with equality iff $P = Q$.

**Interview questions**

1. Prove $D_{KL}(P \Vert Q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$ (via Jensen) and state when equality holds?
2. Why is KL divergence asymmetric? Give a numeric example.
3. Relation among entropy, cross-entropy and KL? Why is cross-entropy loss = entropy + KL?
4. Does KL satisfy symmetry or the triangle inequality? What symmetrized alternative is used?
5. Where does KL appear in classification, distillation and RLHF? Geometric intuition of minimizing KL?

<details>
<summary>中文题目</summary>

1. 证明 KL 散度非负:$D_{KL}(P \Vert Q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$,并说明等号何时成立?
2. KL 散度为什么不对称?给出一个数值例子。
3. 熵、交叉熵与 KL 三者的关系?为什么交叉熵损失 = 熵 + KL?
4. KL 满足对称性与三角不等式吗?需要对称距离时用什么?
5. 分类、知识蒸馏、RLHF 中 KL 分别出现在哪里?最小化 KL 的几何直觉是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/entropy-kl?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=entropy-kl)

---

## 12. Gradient Descent · 梯度下降

> Gradient descent minimizes a loss by iterating down the negative gradient: $w_{t+1} = w_t - \eta\nabla L(w_t)$. For an $L$-smooth ($\Vert\nabla L(w) - \nabla L(w')\Vert \le L\Vert w - w'\Vert$) and $\mu$-strongly convex loss with $\eta \le 2/L$ it converges, a…

**Interview questions**

1. Write the GD update $w_{t+1} = w_t - \eta\nabla L(w_t)$ and derive the step-size bound $\eta \le 2/L$ for an $L$-smooth convex loss?
2. How does the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ set the convergence rate $(1 - 1/\kappa)^t$? Why do ill-conditioned quadratics zig-zag?
3. Prove SGD's gradient is an unbiased estimate of the full gradient, $\mathbb{E}[g_t] = \nabla L(w_t)$; by what factor does mini-batching cut the variance?
4. Why does SGD hover around a noise ball — where does $\frac{\eta\sigma^2}{2\mu}$ come from, and how does it depend on the learning rate and batch size?
5. What happens with too-large or too-small learning rates? Why does momentum $v_t = \beta v_{t-1} + (1-\beta)\nabla L$ damp zig-zag oscillation?

<details>
<summary>中文题目</summary>

1. 写出梯度下降更新公式 $w_{t+1} = w_t - \eta\nabla L(w_t)$,并推导对 $L$-光滑凸函数学习率上限 $\eta \le 2/L$?
2. 条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 如何决定收敛率 $(1 - 1/\kappa)^t$?为什么病态二次型呈锯齿轨迹?
3. 证明 SGD 梯度是全局梯度的无偏估计: $\mathbb{E}[g_t] = \nabla L(w_t)$;mini-batch 把方差降到多少倍?
4. 为什么 SGD 收敛在“噪声球”附近?误差下界 $\frac{\eta\sigma^2}{2\mu}$ 从哪来,与学习率和 batch size 什么关系?
5. 学习率过大或过小分别会发生什么?动量项 $v_t = \beta v_{t-1} + (1-\beta)\nabla L$ 为什么能抑制锯齿震荡?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/gradient-descent?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gradient-descent)

---

## 13. High-Dim Geometry & Concentration · 高维几何与测度集中

> Concentration of measure quantifies how fast random quantities cluster around their mean, while high-dimensional geometry reveals counter-intuitive facts about $\mathbb{R}^d$.

**Interview questions**

1. State Markov, Chebyshev and Hoeffding, with the conditions each requires?
2. Derive Chebyshev from Markov? Why is Chebyshev tighter?
3. Why does Hoeffding decay exponentially? Which one is used in ERM/PAC generalization bounds?
4. What is the thin shell phenomenon? Why does the volume of a high-dimensional unit ball concentrate near the shell?
5. Why are high-dimensional random unit vectors nearly orthogonal? Dot-product magnitude? Link to attention scaling?

<details>
<summary>中文题目</summary>

1. 写出 Markov、Chebyshev、Hoeffding 三个不等式并说明各自需要的条件?
2. 如何由 Markov 不等式推出 Chebyshev 不等式?为什么 Chebyshev 更紧?
3. Hoeffding 为什么能给出指数衰减?证明泛化界(ERM/PAC)时用哪一个?
4. Thin Shell 现象是什么?为什么高维单位球的体积集中在球壳附近?
5. 为什么高维随机单位向量几乎正交?点积的量级是多少?与注意力缩放的联系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/high-dim-geometry?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=high-dim-geometry)

---

## 14. Hypothesis Testing · 假设检验

> Hypothesis testing makes a binary decision between the null $H_0$ and the alternative $H_1$: reject $H_0$ when the test statistic falls in the rejection region.

**Interview questions**

1. Define Type I error $\alpha$, Type II error $\beta$ and power $1-\beta$. Why does lowering $\alpha$ raise $\beta$?
2. Precise definition of a p-value? Why is it not the probability that $H_0$ is true?
3. How does the family-wise error accumulate over $k$ tests? Principle and cost of Bonferroni $\alpha'=\alpha/k$?
4. FWER vs FDR? The concrete steps of the Benjamini–Hochberg procedure?
5. How do $\alpha$, effect size and sample size affect power? How to do power analysis / sample-size planning?

<details>
<summary>中文题目</summary>

1. 第一类错误 $\alpha$、第二类错误 $\beta$ 与检验功效 $1-\beta$ 的定义?降低 $\alpha$ 为什么增大 $\beta$?
2. p 值的精确定义是什么?为什么不能说 p 值是 $H_0$ 为真的概率?
3. 做 $k$ 次检验时总体假阳性率如何累积?Bonferroni 校正 $\alpha' = \alpha/k$ 的原理与代价?
4. FWER 与 FDR 的区别?Benjamini–Hochberg 流程的具体步骤?
5. 显著性水平、效应量、样本量如何影响功效?如何做功效分析与样本量规划?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/hypothesis-testing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hypothesis-testing)

---

## 15. Importance Sampling · 重要性采样

> Importance Sampling estimates $\mathbb{E}_p[f]$ without sampling from $p$ directly: when $p$ is hard to sample, draw from an easy $q$ and reweight, $$\mathbb{E}_p[f] = \int f(x) p(x) \, dx = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_q[f(x) w(x)], \qu…

**Interview questions**

1. Derive $\mathbb{E}_p[f] = \mathbb{E}_q[f w]$ with weight $w = p/q$; why is the sample-mean estimator unbiased?
2. What is the variance formula $\mathrm{Var}_q[f w]$? Why does a mismatched $q$ blow up the variance?
3. What is the optimal proposal? Why does it drive variance to 0, and why can't it be used in practice?
4. How is the effective sample size $ESS = 1/\sum \hat{w}_i^2$ defined and what does it measure?
5. Importance sampling vs rejection sampling: estimating expectations vs generating samples; weighted average vs 0/1 acceptance?

<details>
<summary>中文题目</summary>

1. 推导 $\mathbb{E}_p[f] = \mathbb{E}_q[f w]$ 与权重 $w = p/q$,为什么样本均值估计无偏?
2. 重要性采样的方差公式 $\mathrm{Var}_q[f w]$ 是什么?为何 $q$ 与 $p$ 失配时方差爆炸?
3. 最优提议分布是什么?为什么它把方差降到 0?实际中为何不能直接用?
4. 有效样本量 $ESS = 1/\sum \hat{w}_i^2$ 如何定义?它衡量什么?
5. 重要性采样与拒绝采样的区别?(算期望 vs 生成样本;权重加权 vs 0/1 接受)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/importance-sampling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=importance-sampling)

---

## 16. Inductive Bias · 归纳偏置

> Inductive bias is the set of built-in prior assumptions in a model/algorithm that makes it prefer certain hypotheses over others given finite data, enabling generalization to unseen data.

**Interview questions**

1. What is inductive bias? How do CNN translation equivariance and weight sharing buy sample efficiency (vs a same-size MLP)?
2. Why must the Transformer add positional encoding? Formalize the permutation equivariance of attention (permuting inputs only permutes output positions).
3. Why are weight decay / early stopping / data augmentation / dropout also inductive biases? What prior does each inject?
4. Pros and risks of strong bias: when is a bias wrong? How does this relate to the bias-variance tradeoff?
5. How do modern LLMs trade strong architectural bias for weak bias plus massive data? What are the costs and benefits?

<details>
<summary>中文题目</summary>

1. 什么是归纳偏置?举例说明 CNN 的平移等变性与权值共享如何带来样本效率(与同规模 MLP 对比)。
2. 为什么 Transformer 必须加位置编码?写出注意力对输入置换的等变性(置换输入只置换输出位置)。
3. 为什么权重衰减 / early stopping / 数据增强 / dropout 也都算归纳偏置?各注入什么先验?
4. 强偏置的优点与风险: 什么时候偏置是“错的”?它与偏差-方差权衡如何关联?
5. 现代 LLM 如何用“弱架构偏置 + 大数据”路线替代经典模型的强偏置?代价与收益是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/inductive-bias?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=inductive-bias)

---

## 17. Inverse Transform Sampling · 逆变换采样

> Inverse Transform Sampling is the most basic way to draw a random variable from an arbitrary 1-D distribution $F$: if $U \sim \text{Uniform}(0, 1)$, then $X = F^{-1}(U)$ has distribution $F$.

**Interview questions**

1. Prove inverse transform sampling: derive $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$ and justify each step?
2. What are the applicability conditions of inverse transform sampling? Why can't it be applied to the normal directly?
3. Draw $X \sim \text{Exp}(\lambda)$ via inverse transform: derive the explicit $F^{-1}$ and write the algorithm?
4. Inverse transform sampling vs Box-Muller: differences and when to use each?
5. How does inverse transform sampling work for discrete distributions (Bernoulli/Poisson)? Complexity?

<details>
<summary>中文题目</summary>

1. 证明逆变换采样正确性:推演 $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$ 并说明每步依据?
2. 逆变换采样的适用条件是什么?正态分布为什么不能直接用它?
3. 用逆变换采样生成指数分布 $X \sim \text{Exp}(\lambda)$:求 $F^{-1}$ 的具体形式并写出算法?
4. 逆变换采样与 Box-Muller 方法的区别?各自适用什么场景?
5. 对离散分布(如伯努利/泊松),逆变换采样如何实现?复杂度是多少?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/inverse-transform?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=inverse-transform)

---

## 18. Jacobian & Hessian · Jacobian 与 Hessian

> For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the $m \times n$ matrix $J$ with entries $J_{ij} = \partial f_i / \partial x_j$ — each row is the gradient row-vector of component $f_i$.

**Interview questions**

1. Dimension check: why is the Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$ $m \times n$ and not $n \times m$? Write the Jacobian of $f(x,y) = (x^2 + y, xy)$ by hand.
2. Derive: gradient $(A + A^T)x$ and Hessian $A + A^T$ of the quadratic form $f(x) = x^TAx$; why does it simplify to $2Ax$ when $A$ is symmetric?
3. By hand: compute the Jacobian/Hessian of $f(x_1, x_2) = x_1^2 + 3x_1x_2 + 2x_2^2$ and test convexity via the principal-minor criterion ($a > 0$ and $ac - b^2 > 0$) or eigenvalues.
4. Prove: a twice-differentiable $f$ is convex iff its Hessian is everywhere positive semidefinite; why does this matter for global optimality (convex ⟹ local = global)?
5. Chain-rule dimension check: for the scalar loss $L = \Vert Xw - y \Vert^2$, use the dimension rule to confirm the gradient is $2X^T(Xw - y)$ and not any other transpose arrangement.

<details>
<summary>中文题目</summary>

1. 维度检查: 为什么 $f: \mathbb{R}^n \to \mathbb{R}^m$ 的 Jacobian 是 $m \times n$ 而非 $n \times m$? 手写 $f(x,y) = (x^2 + y, xy)$ 的 Jacobian?
2. 推导: 二次型 $f(x) = x^TAx$ 的梯度 $(A + A^T)x$ 与 Hessian $A + A^T$; $A$ 对称时为何简化为 $2Ax$?
3. 手算: 对 $f(x_1, x_2) = x_1^2 + 3x_1x_2 + 2x_2^2$ 求 Jacobian/Hessian, 用主元判据 ($a > 0$ 且 $ac - b^2 > 0$) 或特征值判断凸性?
4. 证明: 二阶可微 $f$ 凸当且仅当 Hessian 处处半正定; 为什么这对全局最优性至关重要 (凸 ⟹ 局部 = 全局)?
5. 链式法则维度核对: 标量损失 $L = \Vert Xw - y \Vert^2$ 对 $w$ 的梯度, 用维度法则确认答案是 $2X^T(Xw - y)$ 而非其他转置组合?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/jacobian-hessian?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=jacobian-hessian)

---

## 19. Markov Chain & Detailed Balance · 马尔可夫链与细致平衡

> Detailed Balance is the key sufficient condition for a Markov chain to have stationary distribution $\pi$: if the transition kernel $P(i \to j)$ satisfies the pairwise reversibility $$\pi(i) P(i \to j) = \pi(j) P(j \to i), \quad \forall\, i, j,$$ then $\pi$ is…

**Interview questions**

1. From detailed balance $\pi(i) P(i \to j) = \pi(j) P(j \to i)$, prove $\sum_i \pi(i) P(i \to j) = \pi(j)$ — i.e. $\pi$ is stationary — justifying each step?
2. Is detailed balance sufficient or necessary for stationarity? Give an example of a stationary but non-reversible chain?
3. How does the vector form $\pi^T P = \pi^T$ follow from detailed balance?
4. Why do MCMC constructions use detailed balance instead of solving $\pi^T P = \pi^T$ directly?
5. Which regularity conditions (irreducibility, aperiodicity, positive recurrence) are additionally needed for convergence to stationarity?

<details>
<summary>中文题目</summary>

1. 从细致平衡 $\pi(i) P(i \to j) = \pi(j) P(j \to i)$ 证明 $\sum_i \pi(i) P(i \to j) = \pi(j)$,即 $\pi$ 是平稳分布,写出每步依据?
2. 细致平衡与平稳分布的关系:充分还是必要?能否举例说明存在平稳但不可逆的链?
3. 平稳分布向量形式 $\pi^T P = \pi^T$ 如何从细致平衡推出?
4. 为什么构造 MCMC 时用细致平衡而不是直接解线性方程组 $\pi^T P = \pi^T$?
5. 链收敛到平稳分布还需要哪些正则条件(不可约、非周期、正常返)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/detailed-balance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=detailed-balance)

---

## 20. Matrix Calculus & Softmax Gradient · 矩阵求导与 Softmax 梯度

> Matrix calculus and the softmax gradient are whiteboard fundamentals of derivation interviews. First, quadratic forms: expand $x^TAx = \sum_{i,j} x_i A_{ij} x_j$ and differentiate w.r.t.

**Interview questions**

1. Whiteboard: expand $x^TAx = \sum_{i,j}x_iA_{ij}x_j$ to derive $\frac{\partial x^TAx}{\partial x} = (A+A^T)x$; why does it reduce to $2Ax$ when $A$ is symmetric?
2. Whiteboard the softmax+CE joint gradient $\frac{\partial L}{\partial z_i} = p_i - y_i$: derive $\frac{\partial p_j}{\partial z_i}$ for $i=j$ vs $i\neq j$ and simplify with $\sum_j y_j = 1$.
3. Derive the least-squares gradient $\frac{\partial}{\partial x}\Vert Ax - b\Vert_2^2 = 2A^T(Ax-b)$ and apply the dimension-check rule to each factor.
4. Why does $\log p_j$ in CE cancel the softmax denominator to leave $p-y$? Is the gradient still clean under squared loss $\frac{1}{2}(p_i - y_i)^2$?
5. Dimension-check the chain rule: given $\frac{\partial L}{\partial z} \in \mathbb{R}^C$ and the shape of $\frac{\partial z}{\partial W}$, how do they multiply to yield $\frac{\partial L}{\partial W}$?

<details>
<summary>中文题目</summary>

1. 白板推导 $\frac{\partial x^TAx}{\partial x} = (A+A^T)x$:展开 $x^TAx = \sum_{i,j}x_iA_{ij}x_j$ 逐步求偏导,$A$ 对称时为什么等于 $2Ax$?
2. 白板推导 Softmax+CE 联合梯度 $\frac{\partial L}{\partial z_i} = p_i - y_i$:分 $i=j$ / $i\neq j$ 两种情形求 $\frac{\partial p_j}{\partial z_i}$,并利用 $\sum_j y_j = 1$ 化简。
3. 推导最小二乘梯度 $\frac{\partial}{\partial x}\Vert Ax - b\Vert_2^2 = 2A^T(Ax-b)$,并解释维度检查法则($2A^T(Ax-b)$ 各因子形状)。
4. 为什么交叉熵里的 $\log p_j$ 恰好“抵消”softmax 的分母使梯度变成 $p-y$?换成平方损失 $\frac{1}{2}(p_i - y_i)^2$ 梯度还这么简洁吗?
5. 链式法则维度核对: 给定 $\frac{\partial L}{\partial z} \in \mathbb{R}^C$ 与 $\frac{\partial z}{\partial W}$ 的形状,如何相乘得到 $\frac{\partial L}{\partial W}$?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/matrix-calculus-deep?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=matrix-calculus-deep)

---

## 21. Matrix Decomposition Applications · 矩阵分解应用

> Three flagship applications. (1) Low-rank approximation (Eckart-Young theorem): truncating $A = U\Sigma V^T$ to the top $k$ singular values gives $A_k = U_k\Sigma_k V_k^T$, the best rank-$\le k$ approximation of $A$, with Frobenius error…

**Interview questions**

1. State the Eckart-Young theorem: why is $A_k = U_k\Sigma_k V_k^T$ the optimal rank-$k$ approximation? How is $\sqrt{\sum_{i>k}\sigma_i^2}$ derived?
2. Derive PCA = SVD: why are principal directions right singular vectors of $X$ rather than eigenvectors? Why is the variance of component $i$ equal to $\sigma_i^2/(n-1)$?
3. By hand: for a 2×2 matrix, compute the best rank-1 approximation $A_1 = \sigma_1 u_1 v_1^T$ and its Frobenius-norm error.
4. Why does $A^+ = V\Sigma^+U^T$ deliver both least-squares and minimum-norm solutions? Numerical advantages over the normal equations $(X^TX)^{-1}X^T$?
5. Storage accounting: how many parameters does a rank-$k$ approximation need (vs $mn$)? Why is LoRA a low-rank approximation and by how much does it cut trainable parameters?

<details>
<summary>中文题目</summary>

1. 陈述 Eckart-Young 定理: 为什么保留最大 $k$ 个奇异值的 $A_k = U_k\Sigma_k V_k^T$ 是秩 $k$ 最优近似? 误差 $\sqrt{\sum_{i>k}\sigma_i^2}$ 如何推导?
2. 推导 PCA = SVD: 为什么主成分方向是 $X$ 的右奇异向量而非特征向量? 各主成分方差为何是 $\sigma_i^2/(n-1)$?
3. 手算: 给定 2×2 矩阵, 求秩 1 最优近似 $A_1 = \sigma_1 u_1 v_1^T$ 并计算 F-范数误差?
4. 伪逆 $A^+ = V\Sigma^+U^T$ 为何同时给出最小二乘与最小范数解? 与正规方程 $(X^TX)^{-1}X^T$ 相比的数值优势?
5. 手算参数对比: 秩 $k$ 近似需要多少存储 (对 $mn$)? LoRA 为什么是低秩近似, 可训练参数降多少?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/matrix-applications?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=matrix-applications)

---

## 22. MCMC & Detailed Balance · MCMC 与细致平衡

> Markov Chain Monte Carlo (MCMC) builds a Markov chain whose stationary distribution is the target $\pi$ and samples along the chain.

**Interview questions**

1. Derive the MH acceptance rate $\alpha = \min(1, \pi(j) q(i \mid j)/(\pi(i) q(j \mid i)))$ and verify the chain satisfies detailed balance (stationary at $\pi$)?
2. Relationship between Gibbs and MH: why is the Gibbs acceptance rate always 1? Prove it?
3. What is burn-in and why is it needed? How do you diagnose convergence (Gelman-Rubin $\hat R$, trace plots, acceptance rate)?
4. What does the acceptance rate reduce to under a symmetric proposal $q(j \mid i) = q(i \mid j)$ (the Metropolis algorithm)?
5. Can MCMC handle unnormalized $\pi$? Why does the acceptance rate only need $\pi$ up to a constant?

<details>
<summary>中文题目</summary>

1. 推导 MH 接受率 $\alpha = \min(1, \pi(j) q(i \mid j)/(\pi(i) q(j \mid i)))$,并验证该链满足细致平衡(即平稳分布为 $\pi$)?
2. Gibbs 采样与 MH 的关系:为什么 Gibbs 每步接受率恒为 1?证明之?
3. 什么是 burn-in?为什么需要?如何诊断收敛(Gelman-Rubin $\hat R$、迹图、接受率)?
4. 对称提议 $q(j \mid i) = q(i \mid j)$ 时接受率退化成什么(Metropolis 算法)?
5. 未归一化的 $\pi$ 能用 MCMC 吗?接受率中的 $\pi$ 为什么只需差常数倍?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/mcmc?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mcmc)

---

## 23. Mutual Information · 互信息

> Mutual information quantifies how much one random variable tells us about another: $I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = \sum_x \sum_y p(x,y)\log\frac{p(x,y)}{p(x)p(y)} \ge 0$.

**Interview questions**

1. Define mutual information and give its three equivalent forms. Why is $I(X;Y) \ge 0$?
2. How does MI differ from the correlation coefficient? Why does zero correlation not imply independence?
3. What is the necessary and sufficient condition for $I(X;Y) = 0$? Relation with conditional entropy $H(X|Y)$?
4. Meaning of the chain rule $I(X,Y;Z) = I(X;Z) + I(Y;Z|X)$? How is it used in feature selection?
5. Given a joint table $p(x,y)$, compute MI by hand? How to estimate MI for continuous variables?

<details>
<summary>中文题目</summary>

1. 写出互信息的定义及三种等价形式?为什么 $I(X;Y) \ge 0$?
2. 互信息与相关系数有什么区别?为什么相关系数为 0 不代表独立?
3. $I(X;Y) = 0$ 的充要条件是什么?与条件熵 $H(X|Y)$ 的关系?
4. 链式法则 $I(X,Y;Z) = I(X;Z) + I(Y;Z|X)$ 的含义?在特征选择中如何应用?
5. 给定联合分布 $p(x,y)$,如何手算互信息?连续变量如何估计互信息?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/mutual-information?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mutual-information)

---

## 24. Newton's Method · 牛顿法 vs 梯度下降

> Newton's method uses second-order information: take the second-order Taylor expansion at $w$, $f(w + \Delta w) \approx f(w) + \nabla f^T\Delta w + \frac{1}{2}\Delta w^T H\Delta w$ (H = Hessian), differentiate w.r.t.

**Interview questions**

1. Derive the Newton step $\Delta w = -H^{-1}\nabla f$ from the second-order Taylor expansion (show the full stationarity derivation).
2. Why does Newton converge in one step on quadratics and quadratically in general? Explain the rate $\Vert w_{k+1}-w^*\Vert \le C\Vert w_k-w^*\Vert^2$.
3. Ill-conditioned example $f = x^2 + 100y^2$: estimate steps for GD ($\kappa=100$) vs Newton and explain why Newton lands in one step.
4. What is Newton's cost ($O(D^3)$ per step, $O(D^2)$ memory)? How do L-BFGS and Gauss-Newton approximate the Hessian?
5. For non-convex functions the Hessian can be indefinite — why does the Newton direction become unreliable? How do damped Newton and $H + \lambda I$ (Levenberg-Marquardt) fix it?

<details>
<summary>中文题目</summary>

1. 从二阶泰勒展开推导牛顿步 $\Delta w = -H^{-1}\nabla f$(写出对 $\Delta w$ 求导置零的完整步骤)。
2. 为什么牛顿法对二次函数一步收敛、对一般函数二次收敛?给出收敛阶定义 $\Vert w_{k+1}-w^*\Vert \le C\Vert w_k-w^*\Vert^2$ 的含义。
3. 病态例题 $f = x^2 + 100y^2$:分别估计 GD($\kappa=100$)与牛顿法各需多少步收敛,并解释牛顿法一步到位的机理。
4. 牛顿法的计算代价($O(D^3)$/步、$O(D^2)$ 内存)是多少?L-BFGS 与 Gauss-Newton 分别如何近似 Hessian?
5. 非凸函数 Hessian 可能非正定,牛顿方向为何不再可靠?阻尼牛顿 / $H + \lambda I$(Levenberg-Marquardt)如何修复?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/newton-method?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=newton-method)

---

## 25. Point Estimation · 点估计 MLE/矩估计

> Point estimation maps samples to a single parameter value: Maximum Likelihood Estimation (MLE) picks the parameter that maximizes the probability of the observed data, $\hat{\theta} = \arg\max_{\theta} \prod_{i=1}^{n} p(x_i \mid \theta)$;

**Interview questions**

1. Write the MLE definition and derive the MLE of a normal mean.
2. Why take the log-likelihood for MLE? Does it change the optimum?
3. Meaning of the Cramér–Rao bound and the definition of Fisher information?
4. MLE vs method of moments: differences, pros and cons?
5. Asymptotic properties of MLE (consistency, asymptotic normality, efficiency)?

<details>
<summary>中文题目</summary>

1. 写出 MLE 的定义式, 并推导正态分布均值的 MLE。
2. 为什么 MLE 通常取对数似然再求导?是否改变最优解?
3. Cramér–Rao 下界 $\text{Var}(\hat{\theta}) \ge 1/I(\theta)$ 的含义与 Fisher 信息量的定义?
4. MLE 与矩估计 (MoM) 的区别及各自的优缺点?
5. MLE 的大样本性质(一致性、渐近正态、渐近有效)是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/estimation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=estimation)

---

## 26. Probability Axioms & Bayes · 概率公理与贝叶斯定理

> The probability axioms are three conditions for measuring uncertainty — non-negativity $P(A) \ge 0$, normalization $P(\Omega) = 1$, and countable additivity $P(\bigcup_{i} A_i) = \sum_i P(A_i)$ for disjoint $A_i$.

**Interview questions**

1. State the three probability axioms and prove $P(A^c) = 1 - P(A)$ and $P(A \cup B) = P(A) + P(B) - P(A \cap B)$?
2. Derive Bayes' theorem and the law of total probability? What is the base rate fallacy?
3. Prevalence 1%, sensitivity and specificity 99% — what is the posterior given a positive test?
4. Which terms are the prior, likelihood, evidence and posterior, and how do they jointly shape inference?
5. Conditional independence vs independence? Why does $P(A|B) = P(A)$ mean $A,B$ are independent?

<details>
<summary>中文题目</summary>

1. 写出概率的三条公理,并证明 $P(A^c) = 1 - P(A)$ 与 $P(A \cup B) = P(A) + P(B) - P(A \cap B)$?
2. 推导贝叶斯公式与全概率公式?什么叫基准率谬误?
3. 患病率 1%、检测灵敏度与特异度均 99%,检测阳性时后验概率是多少?
4. 先验、似然、证据、后验分别对应公式中的哪一项?它们如何共同决定推断结果?
5. 条件独立与独立的关系?为什么 $P(A|B) = P(A)$ 时称 $A,B$ 独立?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/probability-axioms?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=probability-axioms)

---

## 27. Projection & Least Squares · 投影与最小二乘

> The least-squares problem $\min_w \Vert Xw - y \Vert^2$ is, geometrically, an orthogonal projection: when $y \notin C(X)$ no exact solution exists, and the optimum requires $Xw$ to equal the orthogonal projection $\hat{y} = Py$ of $y$ onto the column space…

**Interview questions**

1. Derivation: from the residual orthogonality $X^T(y - Xw) = 0$, derive the normal equations $w = (X^TX)^{-1}X^Ty$ and sketch the projection geometry.
2. Prove $P = X(X^TX)^{-1}X^T$ is an orthogonal projection ($P^2 = P$, $P^T = P$); onto which subspace does $I - P$ project?
3. By hand: fit a line $y = w_0 + w_1x$ to 3 data points by constructing $X$ and $y$ and solving the normal equations.
4. When is $X^TX$ singular? How does the pseudoinverse $X^+ = V\Sigma^+U^T$ yield the minimum-norm solution, and how does it differ from $(X^TX)^{-1}X^T$?
5. Why are the normal equations numerically unstable ($\kappa(X^TX) = \kappa(X)^2$)? How do pseudoinverse, regularization, or QR mitigate collinearity?

<details>
<summary>中文题目</summary>

1. 推导: 从残差正交条件 $X^T(y - Xw) = 0$ 手推正规方程 $w = (X^TX)^{-1}X^Ty$, 画出投影几何示意图?
2. 证明 $P = X(X^TX)^{-1}X^T$ 是正交投影矩阵 ($P^2 = P$, $P^T = P$), 并说明 $I - P$ 投影到哪个子空间?
3. 手算: 给定 3 个数据点拟合直线 $y = w_0 + w_1x$, 构造 $X$ 与 $y$, 用正规方程解出 $w$?
4. $X^TX$ 何时不可逆? 秩亏时伪逆 $X^+ = V\Sigma^+U^T$ 如何给出最小范数解, 与 $(X^TX)^{-1}X^T$ 有何区别?
5. 为什么正规方程数值不稳 ($\kappa(X^TX) = \kappa(X)^2$)? 共线特征时如何缓解 (伪逆/正则化/QR)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/projection-least-squares?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=projection-least-squares)

---

## 28. Regularization Geometry · L1/L2 正则化几何

> Regularization is equivalent to adding a penalty or constraining the solution: $\min_w L(w)$ s.t. $\Vert w\Vert_p \le t$.

**Interview questions**

1. Draw the L1 diamond and L2 ball; explain why contour tangency at diamond vertices yields sparse solutions while ball tangencies are generically off-axis.
2. Derive the ridge closed form $w = (X^TX+\lambda I)^{-1}X^Ty$ and use SVD to analyze the per-direction shrinkage $\sigma_i^2/(\sigma_i^2+\lambda)$.
3. Bayesian view: write the log-posterior under Gaussian/Laplace priors and show L2 = Gaussian-prior MAP, L1 = Laplace-prior MAP.
4. Why can L1 land exactly on non-differentiable corners? Write and explain the lasso subgradient condition $0 \in X^T(Xw-y) + \lambda\partial\Vert w\Vert_1$.
5. How does the regularization strength $\lambda$ trade off bias vs variance? Why is regularization described as trading bias for variance?

<details>
<summary>中文题目</summary>

1. 画出 L1(菱形)与 L2(球)的约束域,解释为什么损失椭圆与菱形相切于顶点时得到稀疏解,而球切点一般不在坐标轴上。
2. 推导岭回归闭式解 $w = (X^TX+\lambda I)^{-1}X^Ty$,并用 SVD 说明各方向收缩因子 $\sigma_i^2/(\sigma_i^2+\lambda)$ 的行为。
3. 贝叶斯视角: 写出高斯/拉普拉斯先验下的对数后验,证明 L2 = 高斯先验 MAP、L1 = 拉普拉斯先验 MAP。
4. 为什么 L1 能恰好落在不可导的拐角处?写出 lasso 的次梯度条件 $0 \in X^T(Xw-y) + \lambda\partial\Vert w\Vert_1$ 并解释。
5. 正则化强度 $\lambda$ 如何联动偏差-方差?为什么说正则化是“以偏差换方差”的手段?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/regularization-geometry?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=regularization-geometry)

---

## 29. Rejection Sampling · 拒绝采样

> Rejection Sampling generates samples that are exactly distributed as the target $p(x)$ when direct sampling is hard: pick an easy-to-sample proposal $q(x)$ and a constant $M$ with $p(x) \le M q(x)$ for all $x$ (envelope condition).

**Interview questions**

1. Derive the acceptance probability $P(\text{accept}) = 1/M$; why is it independent of the normalization constant of $p$?
2. Full algorithm of rejection sampling? How does the expected number of trials relate to $M$?
3. Why does rejection sampling fail in high dimensions (volume concentration, exponentially growing envelope)?
4. How to choose a good proposal $q$ and constant $M$? What does a loose envelope cost?
5. Can rejection sampling handle unnormalized $p$? How is the constant factor absorbed?

<details>
<summary>中文题目</summary>

1. 推导拒绝采样的接受概率 $P(\text{accept}) = 1/M$,说明它为何与 $p$ 的归一化常数无关?
2. 拒绝采样的完整算法流程是什么?期望重试次数与 $M$ 的关系?
3. 为什么拒绝采样在高维会失效?(体积集中、包络常数指数增长的直觉)
4. 如何选择好的提议分布 $q$ 与常数 $M$?包络过松会付出什么代价?
5. 拒绝采样能处理未归一化的 $p$ 吗?如何修改使 $p$ 可差一个常数倍?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/rejection-sampling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rejection-sampling)

---

## 30. SVD & Eigen-Decomposition · SVD 与特征值分解

> For any $A \in \mathbb{R}^{m \times n}$, the singular value decomposition is $A = U\Sigma V^T$, where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal and $\Sigma$ is diagonal with singular values…

**Interview questions**

1. Compute the SVD of a 2×2 matrix by hand: eigenvalues/eigenvectors of $A^TA$ give $\sigma_i$ and $V$, then $u_i = Av_i/\sigma_i$ gives $U$; verify $A = U\Sigma V^T$.
2. Derive $\sigma_i^2 = \lambda_i(A^TA)$; why does SVD exist for every matrix while EVD needs a diagonalizable square matrix?
3. Geometry: why does the unit circle map to an ellipse with semi-axes $\sigma_i$? How do $V^T$, $\Sigma$, $U$ correspond to rotate, scale, rotate?
4. How does the condition number $\kappa = \sigma_1/\sigma_n$ affect numerical stability? How much can ill-conditioned systems (e.g. Hilbert matrices) amplify errors?
5. How do SVD and EVD relate for symmetric matrices? How can SVD directly read off bases of the four fundamental subspaces (first $r$ columns of $V$ = row space, etc.)?

<details>
<summary>中文题目</summary>

1. 手算 2×2 矩阵的 SVD: 求 $A^TA$ 的特征值/特征向量得到 $\sigma_i$ 与 $V$, 再由 $u_i = Av_i/\sigma_i$ 求 $U$, 验证 $A = U\Sigma V^T$?
2. 推导 $\sigma_i^2 = \lambda_i(A^TA)$; 为什么 SVD 对任意矩阵存在而 EVD 只对可对角化方阵适用?
3. 几何: 为什么单位圆在 $A$ 作用下变成椭圆, 半轴长为 $\sigma_i$? 旋转+缩放如何对应 $V^T$、$\Sigma$、$U$ 三步?
4. 条件数 $\kappa = \sigma_1/\sigma_n$ 如何影响数值稳定性? 病态矩阵 (如希尔伯特矩阵) 解误差被放大多少?
5. 对称矩阵的 SVD 与 EVD 有何关系? 如何用 SVD 直接读出四个基本子空间的基 ($V$ 前 $r$ 列 = 行空间等)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/svd-evd?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=svd-evd)

---

## 31. Variational Inference & ELBO · 变分推断与 ELBO

> Variational inference approximates an intractable posterior $p(z|x)$ with a tractable distribution $q(z|x)$ via the evidence lower bound (ELBO).

**Interview questions**

1. Fully derive $\log p(x) = \text{ELBO} + D_{KL}(q \Vert p)$?
2. How does Jensen's inequality prove that ELBO is a lower bound on $\log p(x)$?
3. Two equivalent ELBO forms (reconstruction + KL penalty)? Why does maximizing ELBO minimize the KL gap?
4. Relation between variational inference and EM? What do the E and M steps do?
5. What is the mean-field assumption? Why does VI tend to underestimate posterior variance?

<details>
<summary>中文题目</summary>

1. 完整推导 $\log p(x) = \text{ELBO} + D_{KL}(q \Vert p)$?
2. 如何用 Jensen 不等式证明 ELBO 是 $\log p(x)$ 的下界?
3. ELBO 的两种等价形式(重构项 + KL 惩罚)如何互相转换?优化 ELBO 为何等价于最小化与后验的 KL?
4. 变分推断与 EM 算法的关系?E 步与 M 步分别做什么?
5. 均值场近似的假设是什么?为什么 VI 倾向于低估后验方差?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/elbo-vi?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=elbo-vi)

---

## 32. Vector Spaces & Fundamental Subspaces · 向量空间与四大子空间

> A vector space is a set closed under addition and scalar multiplication; a subspace is a nonempty subset containing zero and closed under linear combinations.

**Interview questions**

1. By hand: for a concrete matrix (e.g. 2×3), write bases and dimensions of all four fundamental subspaces and verify rank-nullity.
2. State the dimension relations $\dim C(A) + \dim N(A) = n$ and $\dim C(A) + \dim N(A^T) = m$; which ambient space does each term live in?
3. Prove $N(A^T) = C(A)^\perp$ and show every $y \in \mathbb{R}^m$ splits uniquely into a $C(A)$ part plus an $N(A^T)$ part.
4. Derive why a unique solution to $Ax = b$ is equivalent to $N(A) = \{0\}$; demonstrate no-solution, unique-solution, and infinite-solution cases with 2×2 examples.
5. Why is $\text{rank}(A) = \text{rank}(A^T)$? Derive it fully from $N(A) = C(A^T)^\perp$ and the rank-nullity theorem.

<details>
<summary>中文题目</summary>

1. 手算: 给定具体矩阵 (如 2×3 矩阵), 写出四个基本子空间的基与维数, 并验证秩-零化度公式?
2. 默写维度关系 $\dim C(A) + \dim N(A) = n$ 与 $\dim C(A) + \dim N(A^T) = m$, 解释 m、n 各来自哪个空间?
3. 证明 $N(A^T) = C(A)^\perp$, 并说明为何任意 $y \in \mathbb{R}^m$ 可唯一分解为 $C(A)$ 分量与 $N(A^T)$ 分量?
4. 推导: $Ax = b$ 解唯一等价于 $N(A) = \{0\}$, 用 2×2 实例演示无解、唯一解、无穷多解三种情形?
5. 为什么 $\text{rank}(A) = \text{rank}(A^T)$? 用 $N(A) = C(A^T)^\perp$ 与秩-零化度定理完整推导?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/Math/card/vector-spaces?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vector-spaces)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
