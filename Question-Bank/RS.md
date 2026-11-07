# 🎓 RS Role Track · 算法科学家 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **25 cards · 125 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. 3-5 Year Research Vision & Narrative · 3-5 年 Research Vision 规划叙事

> The 3-to-5 Year Research Vision & Scientific Narrative framework structures the overarching strategic roadmap and technical leadership narrative demanded in Senior/Principal Research Scientist hiring loops;

**Interview questions**

1. Deliver a structured 5-minute presentation answering 'What is your research vision for the next 3-5 years?' balancing grand ambition with concrete milestones?
2. How to articulate the intellectual continuity linking your past foundational papers to your upcoming 5-year research agenda?
3. How do you structure a balanced research portfolio combining moonshot foundational bets with de-risked intermediate milestones?
4. How does your research roadmap establish defensible competitive differentiation in the era of multi-billion dollar industrial compute clusters?
5. How to integrate cross-disciplinary mathematical concepts (statistical mechanics, game theory) to unlock novel AI architectures?

<details>
<summary>中文题目</summary>

1. 现场模拟回答经典 RS 面试压轴大题：“What is your research vision for the next 3-5 years?”，组织一次逻辑严密、兼具宏大愿景与扎实里程碑的 5 分钟答辩？
2. 如何清晰阐述你过去的博士/博士后研究主线与你未来 3~5 年规划之间的内在学术连续性 (Intellectual Continuity)？
3. 在规划长期课题时，如何权衡“高风险、高回报的底层创新 (High-Risk High-Reward)”与“保证稳定产出的阶段性里程碑 (De-risked Deliverables)”？
4. 算力与数据壁垒破局：在面对大机构万卡集群算力垄断时，你的 3~5 年规划如何通过“算法效率”、“小模型推理涌现”或“新数学表征”形成独特竞争壁垒？
5. 跨学科融合创新：如何将统计物理学（自旋玻璃模型）、控制理论（微分博弈）或认知科学（双系统理论）有机融入你的 AI 研究路线图中？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-3-to-5-year-research-roadmap?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-3-to-5-year-research-roadmap)

---

## 2. 4-Step SOTA Paper Breakdown Framework · 论文拆解 4 步法：假设/痛点/消融

> The 4-Step SOTA Paper Breakdown Framework structures a compelling, executive-level technical narrative dissecting landmark AI research papers (Transformer, LoRA, DPO, DeepSeek-R1) in 10-15 minute research rounds;

**Interview questions**

1. Deliver a structured 5-minute technical briefing on a landmark paper (e.g. DeepSeek-R1 or LoRA) using the 4-step breakdown framework?
2. How to identify hidden theoretical assumptions in published papers and articulate the boundary regimes where the algorithm fails?
3. How to explain complex variational equations using intuitive physical metaphors during whiteboard presentation rounds?
4. How to critically audit a paper's experimental integrity regarding baseline tuning, missing ablations, and benchmark leakage?
5. How to connect insights from external papers to synthesize novel hypotheses for your own primary research domain?

<details>
<summary>中文题目</summary>

1. 以一篇代表性论文（如 DeepSeek-R1 或 LoRA）为例，现场使用“假设 $\to$ 瓶颈 $\to$ 算法 $\to$ 消融” 4 步法组织一次 5 分钟的高密度口头汇报？
2. 如何敏锐识别论文中的“隐性前提假设 (Hidden Assumptions)”（如数据独立同分布、高信噪比、单模态分布等），并指出当假设不成立时的失效场景？
3. 在复述论文核心数学公式时，如何用通俗而精炼的物理直觉解释复杂的公式（如将 DPO 解释为隐式奖励对数几率比）？
4. 如何对论文的实验设计进行批判性审视（如基线是否调优充分、是否缺少关键消融、是否存在数据污染）？
5. 如何将他人论文中的思想迁移并应用到你自己的当前研究课题中（展现 Research Taste 与方法论迁移能力）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-four-step-paper-breakdown-method?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-four-step-paper-breakdown-method)

---

## 3. Ablation Study Design & Gradient Blocking · 消融实验组合设计与梯度阻断

> Rigorous Ablation Matrix Design & Gradient Blocking Causality Analysis provides the scientific proof that every individual novel component in a proposed model genuinely contributes orthogonal performance gains;

**Interview questions**

1. Design the standard academic ablation table for 3 novel modules ($A, B, C$) demonstrating component combinations, VRAM, throughput, and test scores?
2. Explain how gradient blocking (`.detach()`) on auxiliary heads isolates representational benefits from gradient interference?
3. How to construct hyperparameter sensitivity curves showing model robustness across wide parameter ranges ($\lambda \in [0.01, 10]$)?
4. How should an RS transparently report and analyze negative ablation findings to provide valuable diagnostic insights for the research community?
5. How to perform Compute-Equalized Ablations where baseline models are trained with expanded compute budgets to match your proposed model?

<details>
<summary>中文题目</summary>

1. 如果你的新模型包含 3 个新模块（$A, B, C$），请设计出标准的消融实验表格，展示组件开关组合、显存开销、训练速度与最终测试分数的汇报格式？
2. 梯度阻断 (Gradient Blocking via `.detach()`): 如何通过阻断辅助任务 (Auxiliary Loss) 的梯度回传，验证表征学习与多任务梯度的因果贡献？
3. 超参数敏感度分析 (Sensitivity Analysis): 如何绘制关键超参数（如 Loss 权重 $\lambda$、温度系数 $\tau$、LoRA Rank $r$）在宽广区间内的稳定性曲线？
4. 消融实验中的“负结果 (Negative Results)”呈现：当某个设计的子模块未达预期时，如何在论文中诚实反思并提炼出对社区有价值的深刻洞见？
5. 计算成本平权消融：当某个新结构比 Baseline 慢 20% 时，如何将 Baseline 训练更长 Epoch 直至消耗相同 GPU Hours 进行平权对比？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-ablation-study-gradient-blocking?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-ablation-study-gradient-blocking)

---

## 4. Bonferroni Correction & Selection Bias · 多重假设检验校正与选择偏差

> The Multiple Testing Correction (Bonferroni, Benjamini-Hochberg FDR & Selection Bias Mitigation) framework defends statistical evaluations against the catastrophic inflation of false-positive discoveries (P-hacking);

**Interview questions**

1. Derive the mathematical proof showing why testing $K=100$ trials yields a $>99.4\%$ probability of generating a spurious false-positive result?
2. Contrast Bonferroni (controlling FWER) vs Benjamini-Hochberg (controlling FDR) in statistical conservatism and Type-II error power?
3. Why does repeated model selection on validation sets induce overfitting to the test distribution without clean holdout splits?
4. How does non-parametric Bootstrap Resampling construct empirical 95% confidence intervals on model performance deltas?
5. Explain Data Snooping and HARKing pitfalls and how experiment pre-registration enforces scientific integrity?

<details>
<summary>中文题目</summary>

1. 数学推导为什么在不进行多重校正时，测试 $K=100$ 次实验后产生至少一个假阳性显著结果的概率高达 $1 - (0.95)^{100} \approx 99.4\%$？
2. 详细对比 Bonferroni 强控制族误差率 (FWER) 与 Benjamini-Hochberg 控制假发现率 (FDR) 在检验保守性与统计功效 (Power) 上的权衡？
3. 多次调优后的测试集污染 (Overfitting the Test Set): 为什么在验证集上调完超参后，必须在完全未见过的独立测试集上进行一次性前向评估？
4. Bootstrap 重采样法 (Bootstrap Resampling: 采样 10,000 次计算 95% 置信区间 CI) 在验证模型指标差异中的非参数应用？
5. 数据嗅探 (Data Snooping) 与事后假设构建 (HARKing: Hypothesizing After Results are Known) 在科研道德中的危害与预注册实验 (Pre-registration) 防护？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-bonferroni-multiple-testing-correction?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-bonferroni-multiple-testing-correction)

---

## 5. Chinchilla Compute-Optimal Scaling Laws · Chinchilla 算力最优定律与比率

> The Chinchilla Compute-Optimal Scaling Laws (Hoffmann et al., DeepMind) formalizes the exact mathematical Pareto frontier allocating compute budgets between model parameters $N$ and dataset tokens $D$;

**Interview questions**

1. Derive the optimal scaling exponents $a = \frac{\beta}{\alpha+\beta}$ and $b = \frac{\alpha}{\alpha+\beta}$ using Lagrange Multipliers under compute constraint $C = 6ND$?
2. Explain the mathematical derivation showing why training compute is precisely $6N$ FLOPs per token (2N forward + 4N backward)?
3. Why does industry embrace Inference-Aware Over-training (e.g. 8B models trained on 15T tokens) to optimize lifetime serving cost?
4. Analyze the mathematical discontinuity between smooth cross-entropy scaling and sudden emergent metric jumps on non-linear accuracy thresholds?
5. How does high-quality synthetic data generation and reasoning search bypass the impending internet human text data wall?

<details>
<summary>中文题目</summary>

1. 利用拉格朗日乘子法，在约束条件 $C = 6ND$ 下对损失函数 $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ 完整推导最优解 $N^* \propto C^a$ 与 $D^* \propto C^b$？
2. 为什么算力估算公式中每个 Token 的前向 + 反向浮点运算量 (FLOPs) 精确等于 $6N$（前向 $2N$，反向 $4N$）？
3. 推断感知缩放 (Inference-Aware Scaling): 为什么在工业落地中，宁愿打破 Chinchilla 最优比进行“超饱和训练 (Over-training: 如 7B 模型训练 15T Tokens)”，以极小化后续万亿次在线推理的显存与延迟？
4. 下游下游任务指标（如准确率、BLEU）与预训练交叉熵损失 (Cross-Entropy Loss) 之间的非线性断裂（Emergent Abilities / 涌现能力）的机理解析？
5. 数据天花板 (Data Wall) 危机：当互联网高质量公开发表人类文本（约 10~20 万亿 Tokens）即将耗尽时，合成数据 (Synthetic Data) 与自对齐如何维持 Scaling Law 延续？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-chinchilla-compute-optimal-scaling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-chinchilla-compute-optimal-scaling)

---

## 6. Controlled Variables & Multi-Seed Variance · 单变量控制、多种子与方差控制

> The Rigorous Experiment Design Framework (Single Controlled Variables, Baseline Parity & Multi-Seed Statistical Significance) sets the gold standard separating professional AI Research Scientists from naive heuristic tuners;

**Interview questions**

1. Why are single-seed comparisons fundamentally unreliable in RL and reasoning tasks due to high trajectory exploration variance?
2. Derive the test statistics for paired $t$-tests and Wilcoxon signed-rank tests to establish $p < 0.05$ algorithmic superiority?
3. How should an RS construct equal-FLOP and equal-parameter baselines when proposing a new architectural block?
4. How to enforce deterministic multi-library seed synchronization across PyTorch, NumPy, Python standard random, and CUDA backend drivers?
5. Explain the academic reporting distinction between Sample Standard Deviation (Std) and Standard Error of the Mean (SEM) in experimental tables?

<details>
<summary>中文题目</summary>

1. 为什么在强化学习与长思维链推理实验中，单种子 (Single Seed) 比较极度不可靠（解释奖励方差与探索随机性的发散效应）？
2. 配对双尾 $t$-检验 (Paired Two-Tailed $t$-Test) 与 Wilcoxon 符号秩检验在验证新算法显著性时的数学原假设与检验统计量计算？
3. 公平比较原则 (Fair Baseline Comparison): 如果你的新模型增加了 10% 的参数量，你应该如何设计同等参数量 (Equal Parameter Count) 或同等计算量 (Equal FLOPs) 的基线对照组？
4. 随机数发生器 (RNG) 的跨库一致性控制：如何在 PyTorch、NumPy、Python `random` 与 CUDA 中实现真正的全局种子同步？
5. 在论文主表格中呈现 $\text{Mean} \pm \text{Std}$ 的学术规范：如何清晰区分运行种子间的样本标准差 (Std) 与均值标准误 (SEM)？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-controlled-variable-seed-variance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-controlled-variable-seed-variance)

---

## 7. DeepSeek-R1 Pure RL & Long CoT Emergence · DeepSeek-R1 纯 RL 自我纠错与长思维

> DeepSeek-R1 Pure RL Reasoning & Self-Correction Emergence establishes the paradigm breakthrough showing that complex long-chain reasoning and self-reflection can spontaneously emerge from zero SFT cold-start data via pure rule-based reinforcement learning (GRP…

**Interview questions**

1. Explain why DeepSeek-R1-Zero emerges long-chain reasoning purely from rule-based verifiers without human-curated SFT cold-starts?
2. How does GRPO's cohort-normalized advantage estimation eliminate Critic models and resolve VRAM bottlenecks over 32k thinking tokens?
3. Explain the Language Mixing pathology in R1-Zero and detail how DeepSeek-R1's multi-stage SFT and RL pipeline resolved it?
4. Why does distilling 671B R1 reasoning traces into 1.5B/7B dense models surpass running pure RL directly on small models?
5. How to extend the R1 rule-based RL paradigm into open-ended scientific exploration where deterministic compilers do not exist?

<details>
<summary>中文题目</summary>

1. 深入剖析 DeepSeek-R1-Zero 为什么完全不需要人工标注的 SFT 数据，仅凭规则奖励 (Rule-based Verifiers) 就能自发涌现复杂的长思维链？
2. GRPO (群组相对策略优化) 在 R1 训练中如何通过群体采样优势消除 PPO 的 Critic 模型，并解决数万 Token 思考链下的显存爆炸？
3. R1-Zero 出现的“语言混杂 (Language Mixing)”与可读性退化问题：正式版 DeepSeek-R1 是如何通过微量冷启动 SFT + 多阶段 RL 彻底解决的？
4. 推理蒸馏 (Reasoning Distillation): 为什么直接将 671B R1 的长思维链蒸馏给 1.5B/7B 小模型，比让小模型自己在纯 RL 中探索的效果更优？
5. 如何将 R1 的规则奖励强化学习范式迁移拓展至非数学/代码的开放性领域（如基于大模型作为验证器的开放式科学探索）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-deepseek-r1-pure-rl-emergence?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-deepseek-r1-pure-rl-emergence)

---

## 8. Diffusion SDE Stochastic Calculus Proof · 扩散模型 SDE 连续随机微分推导

> Score-Based Continuous Stochastic Differential Equations (Score SDE, pioneered by Yang Song) unifies discrete DDPMs and score-matching generative models into a continuous-time stochastic calculus framework;

**Interview questions**

1. Derive the appearance of the score function $\nabla_x \ln p_t(x)$ in reverse-time SDEs via Fokker-Planck and Anderson's theorem?
2. Derive the Probability Flow ODE equation and explain why it shares the exact same marginal probability trajectories as the stochastic SDE?
3. Why does Denoising Score Matching (DSM) bypass intractable marginal score calculations by training on known conditional Gaussian kernels?
4. Derive Classifier-Free Guidance (CFG) as a linear extrapolation between conditional and unconditional score estimates?
5. How does DPM-Solver leverage exponential integrators over the semi-linear Probability Flow ODE to achieve photorealistic sampling in under 10 steps?

<details>
<summary>中文题目</summary>

1. 从伊藤引理 (Itô's Lemma) 与福克-普朗克方程 (Fokker-Planck Equation) 推导反向时间 SDE 中得分函数 $\nabla_x \ln p_t(x)$ 的出现机理？
2. 详细写出概率流常微分方程 (Probability Flow ODE: $\mathrm{d}x = [f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \ln p_t(x)] \mathrm{d}t$) 的形式，并解释为什么它与 SDE 共享相同的边缘概率轨迹？
3. 去噪得分匹配 (Denoising Score Matching, DSM: $\mathbb{E}[\|s_\theta(x_t, t) - \nabla_{x_t} \ln p(x_t | x_0)\|^2]$) 为什么能将不可计算的真实全量分布对数梯度转化为已知的条件高斯对数梯度？
4. 无分类器引导 (Classifier-Free Guidance, CFG): $\tilde{s}_\theta(x_t, c) = s_\theta(x_t, \emptyset) + s \cdot (s_\theta(x_t, c) - s_\theta(x_t, \emptyset))$ 的得分组合推导与温度缩放机理？
5. DPM-Solver 算法如何利用概率流 ODE 的半线性结构 (Semi-linear Structure) 与指数积分器将生成步数压缩至 10 步以内？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-diffusion-score-sde-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-diffusion-score-sde-derivation)

---

## 9. DPO Optimal Policy & Implicit Reward Proof · DPO 闭式最优策略与隐式奖励推导

> The Direct Preference Optimization (DPO) Closed-Form Proof mathematically derives the exact algebraic equivalence between parameterized policies and implicit ground-truth reward functions;

**Interview questions**

1. Derive the step-by-step calculus of variations proof yielding the optimal policy $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) e^{r(x, y)/\beta}$ under KL regularized objectives?
2. Explain mathematically why the intractable partition function $\beta \ln Z(x)$ cancels out when computing pairwise reward differences?
3. Derive DPO's parameter gradient expression and analyze how the implicit reward error dynamically weights gradient steps?
4. Compare IPO and KTO theoretical extensions in regularizing against DPO's overconfidence under noisy preference distributions?
5. Explain DPO from a Fenchel Convex Duality perspective where policy probabilities serve as dual variables to the reward objective?

<details>
<summary>中文题目</summary>

1. 现场手写推导从 KL 约束目标 $\max_\pi \mathbb{E}[r(x, y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$ 求解变分极大值得到 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) e^{r(x, y)/\beta}$ 的每一步代数变换？
2. 为什么在代入 Bradley-Terry 概率模型时，未知的配分函数 (Partition Function) $\beta \ln Z(x)$ 会被完全对消？
3. 写出 DPO 损失函数的参数梯度表达式：$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) [\nabla_\theta \ln \pi_\theta(y_w|x) - \nabla_\theta \ln \pi_\theta(y_l|x)]$ 并解释其自适应加权机制？
4. IPO (Identity Preference Optimization) 与 KTO (Kahneman-Tversky Optimization) 针对 DPO 在非确定性偏好下过拟合问题的理论修正？
5. 从凸对偶性 (Convex Duality) 视角解释 DPO：为什么 DPO 本质上是带强凸惩罚的 Fenchel 对偶形式？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-dpo-closed-form-proof?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-dpo-closed-form-proof)

---

## 10. Embodied VLA Vision-Language-Action Models · 具身智能 VLA 视觉语言动作联合建模

> Embodied Vision-Language-Action (VLA) Foundation Models (RT-2, OpenVLA, Octo) transform multimodal foundation models from passive image-text understanding into closed-loop physical perception, spatial reasoning, and continuous robotic motor actuation;

**Interview questions**

1. Explain how RT-2 discretizes 7-DoF continuous robotic action spaces into 256 token bins integrated into standard language vocabularies?
2. Why does Action Chunking (predicting future $K$-step trajectories) combined with Temporal Ensembling prevent high-frequency robotic jerking?
3. Explain how Diffusion Policies model multi-modal action distributions to resolve ambiguous obstacle-avoidance demonstrations?
4. How to bridge the Sim-to-Real gap in VLA models using Domain Randomization and closed-loop tactile/proprioceptive feedback?
5. How do video world models act as synthetic simulators generating infinite synthetic manipulation trajectories to bootstrap VLA policies?

<details>
<summary>中文题目</summary>

1. 详细剖析 RT-2 如何将 7 自由度机械臂动作空间离散化为 256 个分箱并作为特殊 Token 融入大模型输出词表的数学机制？
2. 动作分块与时序平滑 (Action Chunking & Temporal Ensembling / ACT): 为什么一次性预测未来 $K$ 步动作轨迹并做滑动平均加权比单步逐帧自回归更平稳？
3. 扩散策略 (Diffusion Policy for Robotic Manipulation): 如何利用条件扩散模型对多模态动作分布（解决多峰行为歧义，如绕过障碍物的左右双向选择）进行建模？
4. Sim-to-Real 跨越真实世界差距：如何通过域随机化 (Domain Randomization) 与触觉/本体感知反馈闭环提升端到端操控成功率？
5. 具身智能世界模型与策略学习的融合：如何利用 Sora 式视频预测模型作为内部物理模拟器生成合成遥操作数据供 VLA 训练？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-embodied-vla-foundation-models?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-embodied-vla-foundation-models)

---

## 11. Flow Matching & Optimal Transport · Flow Matching 连续流匹配与最优传输

> Continuous Flow Matching (Lipman et al.) & Optimal Transport Flow Matching (OT-FM, powering Stable Diffusion 3 and Flux.1) revolutionizes continuous generative modeling by replacing curved stochastic diffusion SDE paths with mathematically minimal straight-lin…

**Interview questions**

1. Derive the Conditional Flow Matching (CFM) loss showing why optimal transport linear paths yield constant velocity targets $\dot{\psi}_t = x_1 - x_0$?
2. Explain why Monge-Kantorovich Optimal Transport couplings minimize total kinetic transportation energy across trajectories?
3. Explain the Multimodal Diffusion Transformer (MMDiT) architecture predicting Flow Matching vector fields across joint vision-text streams?
4. Explain how Rectified Flow's Re-Flow procedure straightens trajectories further to enable 1-step or 2-step ultra-fast distillation?
5. Derive exact log-likelihood computation in Flow Matching using the continuous instantaneous change of variables formula and Hutchinson's trace estimator?

<details>
<summary>中文题目</summary>

1. 详细推导条件流匹配 (Conditional Flow Matching, CFM) 与最优传输插值 $\psi_t(x_0, x_1) = (1 - t)x_0 + t x_1$ 下的目标速度场回归损失？
2. 为什么最优传输 (Optimal Transport, OT) 匹配相比随机独立配对能最小化粒子轨迹的总动态运输能量 (Kinetic Energy $\int \|v_t\|^2 \mathrm{d}t$)？
3. Diffusion Transformers (DiT / MMDiT): 如何在 Flow Matching 中将视觉 Latent Token 与文本 Token 共同输入双流/单流 Transformer 预测速度场？
4. 从 Flow Matching 到修正流 (Rectified Flow / InstaFlow): 为什么通过 Re-Flow (重新对齐采样轨迹) 可以进一步将直线性提升至 1 步极速生成？
5. Flow Matching 与连续正则化流 (Continuous Normalizing Flows, CNF) 在精确似然估计 (Exact Likelihood via Hutchinson's Trace Estimator) 中的理论等价性？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-flow-matching-optimal-transport?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-flow-matching-optimal-transport)

---

## 12. Fundamental Novelty vs Incremental Tweaks · 算法本质突破 vs 工程增量鉴别

> The Fundamental Algorithmic Novelty vs Incremental Engineering framework establishes objective criteria distinguishing true paradigm-shifting breakthroughs from cosmetic engineering packaging;

**Interview questions**

1. Analyze DeepSeek-V3's Multi-Head Latent Attention (MLA) and Auxiliary-Loss-Free load balancing as dual theoretical-engineering breakthroughs?
2. How to craft a rigorous rebuttal when reviewers claim your method is 'just an engineering combination', demonstrating emergent theoretical synergy?
3. How to elevate the theoretical depth of a paper by proving your proposed operator generalizes established classical operators as special cases?
4. How to design stress tests (extreme noise, OOD domain shifts, context limits) demonstrating fundamental robustness over brittle heuristics?
5. Explain why complex manual sparse attention heuristics were largely abandoned in favor of dense attention scaled via FlashAttention hardware kernels?

<details>
<summary>中文题目</summary>

1. 以 DeepSeek-V3 的 MLA (Multi-Head Latent Attention) 与 MoE 无辅助损失负载均衡为例，详细剖析为什么它属于兼具理论优雅与工程落地的顶尖创新？
2. 当审稿人质疑你的论文“只是多个已有技术的工程组合 (Just an engineering combination)”时，如何从涌现效应 (Emergent Synthesis) 与互补理论交互角度进行强力 Rebuttal 反驳？
3. 如何通过数学上的形式化泛化证明（如将新算子证明为经典算子的广义推广推广形式）提升论文的理论深度？
4. 负向消融 (Stress Testing): 如何通过在极端恶劣条件（极端噪声、跨领域分布偏移、超长序列）下测试新算法，证明其本质鲁棒性？
5. 从历史视角回顾：为什么曾经风靡一时的诸多复杂注意力变体（如各种手动稀疏模式）大多被淘汰，而标准的 Full Attention 配合 FlashAttention 最终胜出？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-novelty-vs-incremental-engineering?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-novelty-vs-incremental-engineering)

---

## 13. Grid vs Random vs Bayesian Search · 网格/随机/贝叶斯超参搜索对比

> The Hyperparameter Optimization framework (Grid vs Random vs Bayesian Optimization) establishes the mathematical foundations to maximize model performance under strict compute budgets;

**Interview questions**

1. Derive the mathematical proof showing why $N=60$ random trials achieves $>95\%$ probability of catching the top 5% true optimum region?
2. Derive the Expected Improvement (EI) acquisition function formula under Gaussian Process posterior distributions in Bayesian optimization?
3. Why must scale hyperparameters (learning rates $\eta$, weight decay $\lambda$) be sampled log-uniformly rather than linearly?
4. Explain how Successive Halving and Hyperband (ASHA) dynamically allocate compute by aggressively pruning unpromising configurations early?
5. Why is reporting total search spaces and tuning budgets mandatory to avoid over-tuning optimism bias in academic literature?

<details>
<summary>中文题目</summary>

1. 数学推导为什么在随机搜索中只需 $N=60$ 次采样即可有 $P = 1 - (1 - 0.05)^{60} \approx 0.9536$（超 95% 概率）命中前 5% 顶尖超参数区域？
2. 高斯过程回归 (Gaussian Process Regression) 与采集函数 (Expected Improvement, EI: $\mathbb{E}[\max(0, f(x) - f(x^+))]$) 在贝叶斯优化中的计算步骤？
3. 对数均匀采样 (Log-Uniform Sampling): 为什么对于学习率 $\eta \in [10^{-5}, 10^{-1}]$ 与权重衰减 $\lambda$，必须在对数空间而不是线性空间采样？
4. 连续减半与 Hyperband 算法 (Successive Halving & Hyperband / ASHA): 如何通过“早期早停多轮淘汰弱配置”将搜索效率提升 10 倍？
5. 在论文中公开超参数搜索空间 (Search Space) 与调优轮数 (Tuning Budget) 对保障科学诚信与消除过度调优偏置 (Over-tuning Bias) 的必要性？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-hyperparameter-search-bayesian-opt?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-hyperparameter-search-bayesian-opt)

---

## 14. Inference-Time Compute & Search Scaling · Inference-time Compute 缩放定律

> Inference-Time Compute Scaling Laws (underpinning OpenAI o1 and DeepSeek-R1) formalizes the second paradigm of scaling: trading test-time compute FLOPs for sustained logarithmic/power-law accuracy gains on complex reasoning benchmarks;

**Interview questions**

1. Compare Best-of-$N$, Beam Search, and MCTS in scaling curves of test-time compute versus Pass@1 accuracy gains?
2. Why do Process Reward Models (PRM) outperform Outcome Reward Models (ORM) in guiding intermediate tree search branches?
3. How does pure reinforcement learning with rule-based verifiers induce the emergence of long-chain self-correction and backtracking?
4. Derive the compute trade-off showing how test-time reasoning compute allows a small 1.5B model to surpass a zero-shot 70B monolithic model on math?
5. How to prevent test-time search from exploiting adversarial vulnerabilities in learned Reward Models when $N$ scales into thousands?

<details>
<summary>中文题目</summary>

1. 详细对比 Best-of-$N$ 拒绝采样、Beam Search 与 MCTS 在测试期算力消耗 (FLOPs) 与最终求解成功率 (Pass@1) 上的缩放曲线？
2. 过程奖励模型 (Process Reward Model, PRM) 相比结果奖励模型 (Outcome Reward Model, ORM) 在树搜索中提供单步细粒度打分 (Step-level Credit Assignment) 的数学优势？
3. 自回归长思维链 (Thinking Tokens) 中的自我纠错与回溯 (Self-Correction & Backtracking) 是如何通过纯强化学习在无人工标注监督下自发涌现的？
4. 推理期算力与预训练算力的等价置换模型：一个经过大量推理思考的小模型（如 1.5B R1-Distill）在数学能力上为什么能超越单向前向的 70B/400B 巨型模型？
5. 测试期搜索遭遇的“奖励黑客 (Reward Hacking)”与“过拟合验证器 (Overfitting the Verifier)”的防御策略？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-inference-time-compute-scaling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-inference-time-compute-scaling)

---

## 15. Overparameterization & Neural Tangent Kernel · 过参数化与神经正切核 NTK 理论

> The Neural Tangent Kernel (NTK) Theory (Jacot, Gabriel & Hongler) connects highly non-convex deep neural network gradient dynamics in the infinite-width limit ($m \to \infty$) directly to analytically tractable linear Kernel Regression;

**Interview questions**

1. Derive the continuous-time output dynamics $\frac{\mathrm{d}f_t(X)}{\mathrm{d}t} = -\Theta_t(X, X) (f_t(X) - y)$ under gradient flow?
2. Explain why the NTK matrix remains static throughout gradient descent optimization in the infinite-width lazy training regime?
3. Contrast Lazy Training (NTK regime) against Feature Learning (Rich/Mean-Field regime) and explain why finite-width nets benefit from representation learning?
4. Explain Spectral Bias in the NTK framework showing why neural networks prioritize learning low-frequency functional harmonics before high-frequency noise?
5. How does NTK-Aware RoPE exploit neural tangent kernel spectral properties to scale positional base frequencies for long contexts?

<details>
<summary>中文题目</summary>

1. 定义经验神经正切核 (Empirical NTK) $\Theta_t(x, x') = \nabla_\theta f_t(x)^T \nabla_\theta f_t(x')$，并推导在连续时间梯度流 $\frac{\mathrm{d}\theta_t}{\mathrm{d}t} = -\nabla_\theta \mathcal{L}$ 下模型输出的线性微分演化方程？
2. 为什么在宽度 $m \to \infty$ 极限下，NTK 矩阵在整个训练轨迹中保持冻结恒定（$\Theta_t \to \Theta_\infty$）？
3. 懒惰训练机制 (Lazy Training) vs 特征学习机制 (Feature Learning / Mean-Field Regime) 的本质分歧：为什么真实有限宽神经网络依赖特征重塑而非静态 NTK？
4. 双重下降现象 (Double Descent) 在 NTK 视角下的谱偏差 (Spectral Bias) 与频域傅里叶分析（低频分量优先收敛）？
5. 从 NTK 到现代大模型位置编码扩展：NTK-Aware RoPE 如何利用神经正切核的频域衰减缩放基频以实现长文本外推？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-overparameterization-neural-tangent-kernel?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-overparameterization-neural-tangent-kernel)

---

## 16. Peer Review Critique & Rebuttal Defense · 顶会 Reviewer 审稿视角与压力答辩

> The Peer Review Critique & Author Rebuttal Defense Framework trains AI Research Scientists to adopt an Area Chair / Senior Reviewer mindset to preemptively isolate fatal weaknesses in manuscripts and execute persuasive rebuttals under intense scrutiny;

**Interview questions**

1. How to craft a statistically rigorous rebuttal against 'marginal gains within noise level' using paired $p$-values and Cohen's $d$ effect sizes?
2. Act as a senior conference reviewer and isolate 3 critical theoretical or empirical vulnerabilities in a published SOTA paper?
3. How to prioritize compute resources during high-pressure 7-day rebuttal periods to execute the single most decisive ablation experiment requested by reviewers?
4. How to articulate theoretical limitations and boundary regimes transparently to demonstrate scholarly rigor and earn reviewer trust?
5. Explain the meta-review decision dynamics of Area Chairs when reconciling split reviewer scores based on rebuttal quality?

<details>
<summary>中文题目</summary>

1. 当审稿人给出经典恶评：“The performance gain is marginal and likely within noise level (提升微弱，可能是噪声)”时，如何通过多种子显著性检验 $p$-value 与 Effect Size (Cohen's $d$) 进行降维打击式反驳？
2. 以一篇发表的顶会论文为例，现场扮演严格的 Reviewer，挑出其在理论假设、实验设计或消融分析上的 3 个核心致命弱点？
3. 在有限的 Rebuttal 周期（通常 7 天内），如何高效调度 GPU 算力跑出最具杀伤力的“决定性消融实验 (Decisive Ablations)”？
4. 如何向审稿人清晰解释复杂理论定理的物理直觉与边界限制，展示学者谦逊与科学求真精神？
5. Area Chair (AC) 视角决策逻辑：在面对一个 Borderline (分歧较大: 一个 Accept、两个 Reject) 的稿件时，AC 是如何依据 Rebuttal 补充实验的质量做出最终录用裁决的？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-paper-peer-review-critique?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-paper-peer-review-critique)

---

## 17. PPO Clipped Surrogate Lower Bound Proof · PPO 剪切代理目标函数下界证明

> The PPO Clipped Surrogate Objective & Pessimistic Lower Bound Proof formalizes how Proximal Policy Optimization transforms TRPO's second-order trust-region constraints into an efficient first-order gradient objective;

**Interview questions**

1. Diagram and mathematically define the piecewise objective curves of $L^{\text{CLIP}}$ under positive ($\hat{A}_t > 0$) vs negative ($\hat{A}_t < 0$) advantages?
2. Derive GAE's formulation and analyze how hyperparameter $\lambda$ exponentially balances TD bias against Monte Carlo variance?
3. Explain why the $\min$ operator forms a pessimistic lower bound that ignores policy changes that would overly optimize unclipped surrogate errors?
4. Derive the Fisher Information Matrix from second-order Taylor expansions of KL divergence in TRPO's natural gradient framework?
5. Why is entropy regularization essential in PPO to maintain exploration and prevent premature policy entropy collapse?

<details>
<summary>中文题目</summary>

1. 推导并画出当优势函数 $\hat{A}_t > 0$ 与 $\hat{A}_t < 0$ 时，$L^{\text{CLIP}}(\theta)$ 随概率比 $r_t(\theta)$ 变化的分段函数曲线图？
2. 广义优势估计 (Generalized Advantage Estimation, GAE): 详细推导 $\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$ 中参数 $\lambda$ 如何在偏差与方差之间进行指数权衡？
3. 为什么在损失函数中取 $\min(r_t \hat{A}_t, \text{clipped} \cdot \hat{A}_t)$ 能构成对原始代理目标的“悲观下界 (Pessimistic Bound)”？
4. TRPO 的自然策略梯度 (Natural Policy Gradient) 与 KL 约束泰勒二阶展开：如何推导 Fisher 信息矩阵 $F = \mathbb{E}[\nabla \ln \pi \nabla \ln \pi^T]$？
5. PPO 为什么需要熵正则化项 (Entropy Bonus $S[\pi_\theta](s_t)$)？如何通过调节熵权重防止策略过早塌缩至确定性局部极小值？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-ppo-clipped-objective-proof?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-ppo-clipped-objective-proof)

---

## 18. Process Reward Models PRM & Monte Carlo · 过程奖励模型 PRM 与蒙特卡洛搜索

> Process Supervision Reward Models (PRMs) & Step-Level Monte Carlo Credit Assignment revolutionize mathematical reasoning by replacing sparse final trajectory scores with dense step-by-step verification;

**Interview questions**

1. Contrast Outcome Supervision (ORM) vs Process Supervision (PRM) in eliminating lucky guesses and hallucinated intermediate deductions?
2. Explain how Math-Shepherd constructs automated step-level PRM datasets using Monte Carlo tree rollouts without human annotation?
3. Why does the product of step-wise PRM probabilities outperform terminal scores in Best-of-$N$ trajectory selection?
4. Why do large-scale search algorithms ($N > 1000$) exploit reward hacking vulnerabilities in learned PRMs and how to regularize them?
5. How to integrate step-level PRM dense rewards into PPO/GRPO policy gradients to eliminate temporal variance in credit assignment?

<details>
<summary>中文题目</summary>

1. 详细对比结果监督 (Outcome Supervision, ORM) 与过程监督 (Process Supervision, PRM) 在抑制逻辑幻觉与防止“侥幸做对”上的数学机理？
2. Math-Shepherd 算法如何通过从中间步骤发起 $K$ 次 Monte Carlo Rollouts 自动化构建海量 PRM 训练数据集？
3. 为什么在 Best-of-$N$ 搜索中，采用“每步 PRM 概率连乘积 $\prod_{t=1}^T P(\text{step}_t = \text{correct})$”对候选链打分显著优于仅使用链末尾分？
4. PRM 遭遇的对抗性攻击 (Adversarial Exploitation): 为什么当搜索宽度 $N$ 扩大至数千时，生成模型会学会生成欺骗 PRM 的看似专业假步骤？
5. 在强化学习策略梯度中，如何利用 PRM 提供的单步稠密奖励替代传统 GAE 中的稀疏优势计算？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-process-reward-models-prm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-process-reward-models-prm)

---

## 19. Reproducibility Checklist & Determinism · 可复现性检查清单与确定性计算

> The Reproducibility Checklist, Environment Freezing & Deterministic Computation Protocol provides the engineering rigor ensuring deep learning experiments reproduce identical numbers bit-for-bit across different GPU clusters;

**Interview questions**

1. Write the complete PyTorch deterministic setup configuring CuDNN flags and `CUBLAS_WORKSPACE_CONFIG` to ban non-deterministic CUDA kernels?
2. Explain why omitting `worker_init_fn` in multi-process PyTorch DataLoaders causes worker threads to sample identical mini-batch sequences?
3. Explain why floating-point non-associativity in parallel reduction operations causes tiny numerical divergence across GPU architectures?
4. Detail the 4-step canonical workflow for reproducing published papers from baseline alignment to ablation verification?
5. Explain the key audit criteria required by NeurIPS Reproducibility Checklists for empirical machine learning submissions?

<details>
<summary>中文题目</summary>

1. PyTorch 开启确定性模式的标准代码模板：如何正确配置 `cudnn.benchmark = False`, `use_deterministic_algorithms(True)` 与设置环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`？
2. 分布式训练中多进程 DataLoader 的种子陷阱：为什么如果没有在 `worker_init_fn` 中为每个 Worker 独立分配种子，会导致多进程加载重复的数据序列？
3. 浮点数加法结合律失效（$(a + b) + c \neq a + (b + c)$）与 GPU 并行原子加法 (Atomic Add) 在不同 GPU 架构上导致数值微小差异的根因？
4. 论文复现 4 步标准流程：从阅读论文、基线数字对齐、模块替换到超参数敏感度测试的系统化复现方法？
5. NeurIPS / ICML 官方 Reproducibility Checklist 核心审查条目（数据划分、算子细节、算力配置、随机种子数）对学术严谨性的规范？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-reproducibility-checklist-env-freeze?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-reproducibility-checklist-env-freeze)

---

## 20. Research Taste & Strategic Topic Selection · Research Taste：学术价值与选题

> Research Taste & Strategic Topic Selection represents the highest-order intellectual instinct of an elite AI Research Scientist: formulating high-leverage foundational questions that define new paradigms rather than publishing incremental papers on crowded, tr…

**Interview questions**

1. Explain the core thesis of Rich Sutton's 'The Bitter Lesson' and why general search/learning with compute scaling consistently beats human inductive biases?
2. How do you evaluate and defend high-risk foundational exploration against safe incremental benchmark-tuning during research project prioritization?
3. Explain how Pasteur's Quadrant frames industrial AI research: pursuing fundamental theoretical understanding motivated by real-world application frontiers?
4. How to identify saturation plateaus on existing model paradigms and execute timely strategic pivots toward emerging frontiers?
5. How can compute-constrained academic labs execute asymmetric research strategies to achieve breakthrough discoveries?

<details>
<summary>中文题目</summary>

1. 深度剖析 Rich Sutton 的《苦涩的教训 (The Bitter Lesson)》核心思想：为什么在过去 70 年 AI 历史上，依赖通用算力扩展的方法总是最终战胜人类手工设计的先验知识？
2. 在面对两个候选课题时：A（微调现有模块在某个特定子榜单上刷分 1%）vs B（探索长推理链纯强化学习涌现机理），你如何系统阐述选题评估逻辑？
3. 巴斯德象限 (Pasteur's Quadrant: 寻求基础科学理解的同时致力于实际应用) 在指导工业界 AI 实验室研究选题中的应用？
4. 如何识别即将遭遇“报酬递减 (Diminishing Returns)”的技术路线，并在关键时刻果断自我革新转向全新技术范式？
5. 当外部算力资源（GPU 数量）受到限制时，小团队如何利用差异化研究策略（如架构机理分析、高效算法、合成数据对齐）做出世界级突破？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-research-taste-topic-selection?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-research-taste-topic-selection)

---

## 21. RoPE Complex Inner Product Derivation · RoPE 旋转位置编码复数内积证明

> The Rotary Position Embedding (RoPE) Complex Inner Product Proof establishes the mathematical necessity of orthogonal block-diagonal rotation matrices for encoding relative position semantics into Transformers;

**Interview questions**

1. Derive the functional equation proof establishing $f_q(x_m, m) = (W_q x_m) e^{i m \theta}$ in 2D complex space?
2. Write the formal block-diagonal matrix $R_{\Theta, m}^d$ and prove norm preservation and long-range semantic decay properties?
3. Explain NTK-aware scaling which alters base frequency $b' = b \cdot s^{d/(d-2)}$ to interpolate high frequencies while extrapolating low frequencies?
4. How does YaRN utilize scale-dependent frequency masking and attention temperature factors to stabilize long-context perplexity?
5. Compare RoPE against ALiBi and NoPE across computational efficiency, sequence extrapolation, and empirical LLM benchmarks?

<details>
<summary>中文题目</summary>

1. 从泛函方程 $\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n)$ 完整推导二维复数平面下 $f_q(x_m, m) = (W_q x_m) e^{i m \theta}$ 的证明过程？
2. 写出 $d$ 维空间中分块对角旋转矩阵 $R_{\Theta, m}^d$ 的具体矩阵形式，并证明正交矩阵的保范性（$\|R_m x\| = \|x\|$）与距离衰减性质？
3. NTK-Aware RoPE 缩放 (NTK-aware Scaling) 如何通过修改基频底数 $b' = b \cdot s^{d/(d-2)}$ 实现无需微调直接将上下文长度外推 4~8 倍？
4. YaRN (Yet another RoPE extensioN) 如何通过“频段温度修正 (Attention Entropy Temperature Scaling)”解决长文本注意力熵扩散问题？
5. RoPE 与 ALiBi (Attention with Linear Biases)、NoPE (No Position Embedding in DeepSeek-V3) 的理论与实证优劣对比？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-rope-rotation-matrix-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-rope-rotation-matrix-derivation)

---

## 22. Sample Complexity & Rademacher Bounds · 样本复杂度与 Rademacher 泛化界

> Statistical Learning Theory: Sample Complexity, VC-Dimension & Empirical Rademacher Complexity establishes formal mathematical upper bounds on true out-of-sample generalization risk within PAC (Probably Approximately Correct) learning theory;

**Interview questions**

1. Derive the uniform Rademacher generalization error bound using Hoeffding's and McDiarmid's concentration inequalities?
2. Contrast VC-Dimension (combinatorial, distribution-free) against Rademacher Complexity (data-dependent on empirical distributions)?
3. Why do parameter-count-based generalization bounds collapse on overparameterized networks, while Spectral Margin Bounds remain tight?
4. Derive the exact Rademacher complexity bound for linear hypothesis classes with bounded $L_2$ Euclidean norms: $\hat{\mathcal{R}}_S \le \frac{B \cdot \max \|x\|_2}{\sqrt{m}}$?
5. Prove mathematically why gradient descent on overparameterized linear models implicitly converges to the minimum $L_2$-norm interpolating solution?

<details>
<summary>中文题目</summary>

1. 从霍夫丁不等式 (Hoeffding's Inequality) 与 McDiarmid 不等式完整推导 Rademacher 复杂度均匀泛化误差上界定理？
2. 对比 VC 维 (VC-Dimension: 纯组合几何性质、分布无关) 与 Rademacher 复杂度 (数据依赖、反映底层数据流形分布) 的核心数学差异？
3. 为什么在现代深度神经网络中，参数量 $P \gg N$（过参数化百万倍）时，传统基于参数量的复杂度界会崩溃（变为无意义的大数），而谱范数界 (Spectral Bounds) 依然有效？
4. 对偶范数与向量集中不等式 (Vector Concentration Inequalities): 线性分类器 $\mathcal{H} = \{x \mapsto w^T x: \|w\|_2 \le B\}$ 的 Rademacher 复杂度推导？
5. 隐式正则化理论 (Implicit Regularization): 梯度下降在过参数化线性回归中为什么能自动收敛至最小 $L_2$ 范数解（无显式正则项）？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-sample-complexity-generalization-bounds?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-sample-complexity-generalization-bounds)

---

## 23. Self-Attention Lipschitz & Spectral Bound · 自注意力 Lipschitz 连续与谱范数

> The Lipschitz Continuity & Spectral Norm Bounds of Self-Attention mathematically bounds output perturbation sensitivity with respect to input variations in deep Transformer blocks;

**Interview questions**

1. Derive the Jacobian matrix of Softmax $J(z) = \text{diag}(p) - p p^T$ and prove its operator spectral norm is bounded by 1?
2. Why does omitting $\sqrt{d}$ scale down Softmax Jacobian gradients to near-zero as dimension $d$ grows, inducing vanishing gradients?
3. Explain how multi-layer pure self-attention without residual connections causes exponential Rank Collapse towards rank-1 token uniformity?
4. Contrast Pre-LN vs Post-LN through Lipschitz gradient norm backpropagation stability at initialization?
5. Explain how Spectral Normalization uses Power Iteration to enforce $\sigma_{\max}(W) \le 1$ and regularize Lipschitz bounds in deep networks?

<details>
<summary>中文题目</summary>

1. 推导 Softmax 函数关于输入向量的雅可比矩阵 $J_{\text{softmax}}(z) = \text{diag}(p) - p p^T$，并证明其谱范数上界有界于 1？
2. 为什么在自注意力机制中，如果不除以 $\sqrt{d}$，Softmax 的输入方差随维度增大，会导致其雅可比矩阵的导数区域迅速饱和至接近 0（梯度消失）？
3. 注意力机制中的“秩崩溃 (Rank Collapse / Token Uniformity)”现象：为什么没有残差连接 (Residual) 时，多层纯自注意力前向会使所有 Token 向量指数级收敛为一个相同向量？
4. Pre-LN vs Post-LN 的 Lipschitz 梯度反向传播动力学差异：为什么 Pre-LN 能够在初始化阶段提供恒定的梯度范数使得无需 Warmup 即可稳定训练？
5. 谱范数归一化 (Spectral Normalization / 幂迭代法 Power Iteration) 在约束注意力权重矩阵最大奇异值 $\sigma_{\max}(W) \le 1$ 中的实现机制？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-transformer-attention-lipschitz-bound?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-transformer-attention-lipschitz-bound)

---

## 24. Sora Video World Models & Spacetime Patch · Sora 视频生成世界模型与时空 Patch

> Sora-Style Video World Models & 3D Spacetime Latent Patch Architectures transition generative models from static 2D images into interactive physical World Simulators understanding geometry, temporal continuity, and physical dynamics;

**Interview questions**

1. Diagram the tensor dimension transformations from raw $(T, H, W, C)$ video through 3D VAE compression into 3D spacetime tokens fed to DiT?
2. Why does the Transformer-based DiT architecture exhibit superior compute-scaling efficiency over convolutional 2D+1D UNets?
3. How does Sora handle variable-resolution, arbitrary-aspect-ratio videos natively without center cropping via flexible patch packing?
4. Explain how large-scale video prediction naturally induces 3D parallax geometry and object permanence without explicit 3D inductive biases?
5. Analyze the current theoretical failure modes of video world models in physical causality inversion and multi-body rigid collision dynamics?

<details>
<summary>中文题目</summary>

1. 详细画出 3D Spacetime VAE 将原始视频流压缩为 Latent 并切分为 3D Spacetime Patches 输入 DiT 的完整张量维度演变图？
2. 为什么在视频生成中，基于 Transformer 的 DiT 架构相比传统的 2D+1D UNet 架构具有更强、更线性的 Scaling Law 算力扩展特性？
3. 可变分辨率与可变时长原生支持：Sora 如何通过 NaViT 式动态序列打包 (Dynamic Sequence Packing) 在单 Batch 内混合训练不同宽高比与时长的视频？
4. 物理世界常识与几何遮挡一致性 (Object Permanence): 视频世界模型是如何通过大量视频预测自发学习到三维空间视角变换与透视法则的？
5. 当前视频世界模型的技术理论局限性：为什么仍然会发生“因果逻辑倒错（如玻璃杯打碎后自发复原）”与“流体/复杂多刚体碰撞计算错误”？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-world-models-video-generation-sora?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-world-models-video-generation-sora)

---

## 25. Transformer Expressive Power & Turing Limit · Transformer 表达能力与图灵完备界

> The Expressive Power, Circuit Complexity & Turing Completeness of Transformers ($TC^0$ & RASP formalisms) establishes the mathematical boundaries of what self-attention can and cannot compute;

**Interview questions**

1. Prove why fixed-depth Transformers are contained in $TC^0$ and cannot compute $N$-bit Parity in a single forward pass without CoT?
2. Prove that autoregressive Transformers with variable-length Chain-of-Thought generation are Turing Complete?
3. Contrast Hard vs Soft Attention in formal language theory across Regular Languages (DFA) and Context-Free Grammars (PDA)?
4. Explain the theoretical equivalence between Self-Attention and fully connected Graph Neural Network Message Passing with dynamic adjacency kernels?
5. Compare State Space Models (Mamba) vs Transformers in theoretical associative recall capacity limits?

<details>
<summary>中文题目</summary>

1. 从电路复杂性理论 (Circuit Complexity) 证明为什么固定深度的 Transformer 属于 $TC^0$ 类，因而无法在一步前向内求解 $N$ 位的奇偶校验 (Parity Problem)？
2. 数学证明为什么当允许 Transformer 生成长度为 $O(N)$ 或 $O(N^2)$ 的思维链 Token 时，它能够模拟任意图灵机状态转移函数？
3. 硬注意力 (Hard Attention) 与软注意力 (Soft Attention) 在状态机识别（正则语言 DFA vs 上下文无关文法 CFL）表达能力上的分界？
4. 自注意力机制与图神经网络 (GNN / Message Passing) 的理论等价性：全连接自注意力如何被视作带动态边权重的完全图消息传递？
5. 状态空间模型 (SSM / Mamba) 与线性注意力在表达复杂状态追踪（如 Associative Recall 关联记忆）上的理论容量界限与 Transformer 的对比？

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RS/card/rs-expressive-power-turing-completeness?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rs-expressive-power-turing-completeness)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
