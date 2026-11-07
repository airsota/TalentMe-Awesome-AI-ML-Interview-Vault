# 🎮 RL & Post-Training Alignment · 强化学习与后训练对齐 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **38 cards · 190 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. Actor-Critic · Actor-Critic 框架

> Actor-Critic combines an Actor (policy network $\pi_\theta(a|s)$ generating actions) and a Critic (value network $V_w(s)$ evaluating states): the Actor updates along the policy-gradient direction, and the Critic supplies a value baseline and advantage estimate…

**Interview questions**

1. Define the advantage A(s,a)=Q(s,a)−V(s) and justify using it over Q or the raw reward.
2. Whiteboard: derive the Actor's TD-error update from the policy gradient theorem and prove the V(s) baseline is unbiased.
3. How is the Critic loss derived? How do TD(λ)/GAE trade off bias and variance?
4. How does Actor-Critic improve on REINFORCE? Why lower variance and online training?
5. How do A2C/A3C, PPO, SAC, GRPO reuse the Actor-Critic backbone, and what do they change?

<details>
<summary>中文题目</summary>

1. 写出优势函数 A(s,a)=Q(s,a)−V(s),说明为何用优势而非 Q 或奖励本身更新策略?
2. 白板手推: 从策略梯度定理推导 Actor 的 TD 误差更新方向,并证明价值基线 V(s) 无偏?
3. Critic 的损失 L=E[(r+γV(s')−V(s))²] 如何得到?TD(λ)/GAE 如何折中偏差与方差?
4. Actor-Critic 相比 REINFORCE 改进在哪?方差为何更低、可在线训练?
5. A2C/A3C、PPO、SAC、GRPO 如何复用 Actor-Critic 骨架?各做了哪些改动?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/actor-critic?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=actor-critic)

---

## 2. Behavioral Cloning · 行为克隆 BC

> Behavioral cloning (BC) is the most direct imitation method: treat expert trajectories as labeled data and run supervised learning on $(s,a)$ pairs, minimizing $\min_\pi \mathbb{E}_{(s,a) \sim \pi_E}[-\log \pi(a|s)]$ — maximum-likelihood fitting of the expert…

**Interview questions**

1. Why is BC's compounding error $\mathcal{O}(T^2)$? Give the derivation intuition: how does per-step error $\epsilon$ accumulate quadratically along the trajectory?
2. What is causal confusion? Give a driving/robotics example and its mitigations.
3. BC vs DAgger? How does DAgger fix covariate shift?
4. Why does BC look fine in training but fail at deployment? How does the i.i.d. assumption break?
5. BC vs GAIL/IRL: which imitation approach fits which scenario?

<details>
<summary>中文题目</summary>

1. BC 的复合误差为什么是 $\mathcal{O}(T^2)$?给出推导直觉:每步独立误差 $\epsilon$ 如何沿轨迹平方累积?
2. 因果混淆 (Causal Confusion) 是什么?举一个自动驾驶/机器人例子,并说明如何缓解。
3. BC 与 DAgger 的区别?DAgger 如何解决协变量偏移 (covariate shift)?
4. 为什么 BC 训练时表现良好、部署时错误连环发生?i.i.d. 假设如何失效?
5. BC 与 GAIL/逆向 RL 的对比:分别适合什么模仿学习场景?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/behavioral-cloning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=behavioral-cloning)

---

## 3. Conservative Q-Learning · CQL 保守 Q 学习

> Conservative Q-Learning (CQL) is a SOTA offline RL baseline that adds a conservative regularizer to the standard Bellman update, systematically suppressing Q-estimates of OOD actions:…

**Interview questions**

1. Why does CQL yield a Q lower bound? Write the CQL objective and give the derivation intuition for $\hat{Q}^\pi \le Q^\pi$.
2. What do the log-sum-exp term and the data-expectation term do? Why soft-max instead of max?
3. CQL vs policy-constraint methods (BCQ / TD3+BC): conservative value vs constrained policy?
4. How does CQL's coefficient $\alpha$ trade off conservatism vs optimality? Consequences of too large or too small?
5. How does CQL's Q lower bound serve offline-to-online fine-tuning?

<details>
<summary>中文题目</summary>

1. 为什么 CQL 能给出 Q 下界?写出 CQL 目标,并给出 $\hat{Q}^\pi \le Q^\pi$ 的推导直觉。
2. CQL 目标中 log-sum-exp 项与数据期望项各起什么作用?为什么用 soft-max 而非 max?
3. CQL 与策略约束法 (BCQ / TD3+BC) 的路线差异?『保守价值』 vs 『约束策略』?
4. CQL 的系数 $\alpha$ 如何权衡保守性与最优性?过大/过小的后果?
5. CQL 学到的 Q 下界如何服务于 offline-to-online 微调?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/cql?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cql)

---

## 4. Contextual Bandit · 上下文老虎机

> A Contextual Bandit extends MAB with an observable context feature $x_t \in \mathbb{R}^d$ per round; the policy $\pi(x)$ picks an action given features, maximizing the conditional expected reward $\mathbb{E}[r \mid x, a]$.

**Interview questions**

1. Write LinUCB's ridge solution and confidence term; explain $(X_a^T X_a + \lambda I)^{-1}$ and the role of $\lambda I$?
2. Compute the cost of estimating $\hat{\theta}_a$ in $d$ dimensions; why does the inverse covariance shrink as samples accumulate?
3. Fundamental differences among contextual bandits, full RL, and supervised (offline) ML?
4. How do you construct context features for recsys, and how do contextual bandits mitigate cold-start?
5. LinUCB vs online logistic regression/GBDT: what if linearity fails (kernel / neural contextual bandits)?

<details>
<summary>中文题目</summary>

1. 写出 LinUCB 的岭回归解与置信项, 解释 $(X_a^T X_a + \lambda I)^{-1}$ 的含义与 $\lambda I$ 的作用?
2. 数值/推导题: $d$ 维特征下估计 $\hat{\theta}_a$ 的复杂度?为什么逆协方差矩阵随样本增加而收缩?
3. contextual bandit 与完整 RL、监督学习 (offline ML) 的本质区别?
4. 在推荐系统中如何构造上下文特征?如何缓解冷启动?
5. LinUCB vs 在线逻辑回归/GBDT: 线性假设不成立时怎么办 (kernel/neural contextual bandit)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/contextual-bandit?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=contextual-bandit)

---

## 5. CoT & Reasoning RL · 思维链与推理强化

> CoT (Chain-of-Thought) decomposes hard problems into explicit intermediate reasoning steps at inference time; RL-fortified reasoning (RLVR+GRPO) instead bakes long-chain reasoning into the parameters at training time — CoT is the decoding-time behavior those m…

**Interview questions**

1. Relation between CoT and RL reasoning training (RLVR+GRPO): test-time vs training-time methods?
2. Why do RL reasoning models produce explicit reasoning chains? CoT's role in the policy?
3. CoT emergence vs model size? Why do small models gain little?
4. How does self-consistency combine with CoT/RL to boost accuracy?
5. CoT limitations: when is it ineffective or harmful? How does test-time compute help?

<details>
<summary>中文题目</summary>

1. CoT 与 RL 强化推理 (RLVR+GRPO) 的关系:推理时方法与训练时方法的区别?
2. 为什么 RL 推理模型需要显式推理链?CoT 在策略中的角色?
3. CoT 的涌现性与参数量关系?小模型为何收益有限?
4. Self-consistency 如何与 CoT/RL 结合提升推理准确率?
5. CoT 的局限:何时无效甚至有害?推理时扩展如何补足?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/cot-reasoning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cot-reasoning)

---

## 6. Direct Preference Optimization · DPO 直接偏好优化

> DPO (Direct Preference Optimization) reparameterizes the KL-regularized RLHF objective into a closed-form loss over policy probability ratios, removing the reward model and the online RL loop.

**Interview questions**

1. Write the DPO loss and explain every symbol (y_w/y_l/π_ref/β)?
2. Derivation: how does Z(x) cancel? Give the closed-form solution and the 3-step chain?
3. What does β mean? Effects of large vs small β?
4. Why does DPO need no reward model or RL? Its relation to the RLHF objective?
5. DPO vs GRPO selection criteria? DPO's main limitations (data quality, length bias, role of the reference model)?

<details>
<summary>中文题目</summary>

1. 写出 DPO 损失函数,解释各符号 (y_w/y_l/π_ref/β) 的含义?
2. 推导:配分函数 Z(x) 是如何消掉的?列出闭式解与 3 步推导链?
3. β 的含义?β 大与小分别有什么影响?
4. DPO 为什么不需要奖励模型和强化学习?它与 RLHF 目标的关系?
5. DPO vs GRPO 的选择标准?DPO 的主要局限 (数据质量、长度偏置、参考模型作用)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/dpo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dpo)

---

## 7. DQN Trio · DQN 三件套

> DQN (Deep Q-Network) approximates the optimal action-value function with a network $Q_\theta(s,a)$, trained by least-squares regression toward a frozen TD target:…

**Interview questions**

1. Write DQN's loss and explain the target network's role and update scheme.
2. Whiteboard: prove max overestimation E[max X] ≥ max E[X] via Jensen and derive the decoupled Double-DQN target.
3. What does experience replay solve? Limits of uniform sampling; how does prioritized replay improve it?
4. The DQN trio: which problem does each of replay, target network, Double-DQN solve? Further improvements?
5. Why must the TD target be frozen? Consequences of updating the target network too often or too slowly?

<details>
<summary>中文题目</summary>

1. 写出 DQN 损失 L=E[(r+γ max Q_{θ⁻}(s',a') − Q_θ(s,a))²],说明目标网络的作用与更新方式?
2. 白板手推: 用 Jensen 不等式证明 max 算子带来高估 E[max X] ≥ max E[X],并推导 Double DQN 的解耦目标?
3. 经验回放解决什么问题?均匀采样有何局限,优先级回放如何改进?
4. DQN 三件套: 回放、目标网络、Double-DQN 各解决哪个问题?还有哪些改进?
5. 为什么 TD 目标必须固定?目标网络更新太频繁或太慢各有什么后果?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/dqn?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dqn)

---

## 8. Dreamer RSSM (Latent Imagination) · Dreamer RSSM 潜想象

> Dreamer learns a world model from pixel observations with an RSSM (Recurrent State-Space Model) and trains the policy in latent imagination.

**Interview questions**

1. What do RSSM's deterministic and stochastic states capture? Write and explain h_t = f(h_{t-1}, z_{t-1}, a_{t-1}).
2. How does Dreamer train its policy 'in dreams'? The full latent imagined rollout procedure?
3. Why is the stochastic latent z_t necessary for imagined rollouts? What happens with purely deterministic or purely stochastic models?
4. Dreamer vs MuZero latent modeling: similarities and differences?
5. How much more sample-efficient is Dreamer than model-free SOTA on pixel control, and why?

<details>
<summary>中文题目</summary>

1. RSSM 的确定性状态与随机状态分别建模什么?写出更新式 h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) 并解释。
2. Dreamer 如何"做梦"训练策略?潜想象 rollout 的完整流程?
3. 为什么随机潜变量 z_t 对想象 rollout 是必要的?纯确定性/纯随机模型会怎样?
4. Dreamer 与 MuZero 的潜空间建模有何异同?
5. Dreamer 在像素控制上的样本效率比无模型 SOTA 高多少?为什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/dreamer?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dreamer)

---

## 9. Dyna Architecture (Dyna-Q) · Dyna 架构 (Dyna-Q)

> The Dyna architecture (Dyna-Q) hybridizes model learning with model-free learning: after each real interaction it updates Q on real experience, then treats the learned world model as a 'free environment' to generate $n$ imagined rollouts and applies the same Q…

**Interview questions**

1. What is the core idea of Dyna-Q? Write the key pseudocode steps.
2. How are states/actions sampled for imagined rollouts? Problems if the imagined distribution mismatches reality?
3. How does n (imagined steps per interaction) affect convergence speed and compute?
4. Dyna vs Priority Sweeping: what differs?
5. What is Dyna's deep-RL counterpart (e.g. how MBPO mixes model rollouts)?

<details>
<summary>中文题目</summary>

1. Dyna-Q 的核心思想是什么?写出伪代码的关键步骤。
2. 想象 rollout 中状态/动作如何采样?分布与真实不一致会有什么问题?
3. n(每次交互的想象步数)如何影响收敛速度与计算开销?
4. Dyna 与 Priority Sweeping 有什么区别?
5. Dyna 在深度 RL 中的对应物是什么?(如 MBPO 如何混合模型 rollout)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/dyna?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dyna)

---

## 10. Embodied AI & Robotics · 具身智能与机器人

> Embodied AI couples RL/planning with visual world models for robot control: V-JEPA predicts visual features in latent space (a visual world model), while Diffusion Policy directly predicts multimodal continuous action distributions — Gaussian policies cannot e…

**Interview questions**

1. How do RL and visual world models split roles in embodied AI? V-JEPA's role?
2. Why can't Gaussian policies model multimodal actions? How does Diffusion Policy fix it?
3. Sample-efficiency bottleneck in robot RL? How do world models/demonstrations help?
4. Commonalities between embodied AI and LLM reasoning RL (prediction/planning/world models)?
5. V-JEPA latent-space prediction vs explicit pixel prediction: pros and cons?

<details>
<summary>中文题目</summary>

1. 具身智能中 RL 与视觉世界模型如何分工?V-JEPA 的作用?
2. 为什么高斯策略无法建模多峰动作?Diffusion Policy 如何解决?
3. 机器人 RL 的样本效率瓶颈?世界模型/演示数据如何缓解?
4. 具身智能与 LLM 推理强化的共性 (预测/规划/世界模型)?
5. V-JEPA 潜空间预测与显式像素预测的对比?各自优劣?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/robotics-embodied?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=robotics-embodied)

---

## 11. Exploration-Exploitation · 探索-利用权衡

> The exploration-exploitation dilemma is the trade-off an agent faces between trying unknown actions to gather information and exploiting the currently best action for reward — the two cannot be maximized simultaneously, and optimal exploration should be alloca…

**Interview questions**

1. Write the ε-greedy policy formula and explain why the greedy action gets 1-ε+ε/|A|.
2. Write the UCB1 selection formula and explain each term. Why does √(2ln t/n_a) shrink with n_a?
3. Compare ε-greedy, UCB, and Thompson sampling: exploration logic, regret bounds, use cases.
4. What is regret? Why does UCB achieve O(ln T) regret on MABs?
5. Why should optimal exploration be allocated by information gain? Other exploration tricks in RL (optimistic init, count-based bonuses)?

<details>
<summary>中文题目</summary>

1. 写出 ε-greedy 的策略公式并解释:为什么贪心动作概率是 1-ε+ε/|A|?
2. 写出 UCB1 的选臂公式并逐项解释?置信项 √(2ln t / n_a) 为什么随 n_a 收缩?
3. ε-greedy、UCB、Thompson 采样三种策略的对比:探索逻辑、后悔界、适用场景?
4. 什么是后悔 (regret)?MAB 上 UCB 为什么能达到 O(ln T) 的后悔?
5. 为什么最优探索要按信息增益分配?RL 训练中还有哪些探索技巧 (乐观初始化、计数奖励)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/exploration-exploitation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=exploration-exploitation)

---

## 12. GAIL & Inverse RL · GAIL 与逆向 RL

> Inverse RL (IRL) recovers a reward function from expert trajectories; GAIL (Generative Adversarial Imitation Learning) merges adversarial training with max-entropy IRL to learn a policy directly:…

**Interview questions**

1. Write the GAIL objective and explain each term; which divergence between the expert and policy distributions does it minimize?
2. Why does GAIL mitigate compounding error compared to BC? What signal does the discriminator provide?
3. What is the max-entropy IRL assumption $p(\tau) \propto \exp(R(\tau))$? Which IRL ill-posedness (e.g., constant-reward trivial solution) does it fix?
4. Common issues with GAIL's discriminator (implicit reward)? Role of the entropy bonus $-\lambda \mathcal{H}(\pi)$?
5. IRL/GAIL and RLHF reward modeling both infer preferences from behavior — what are the key differences?

<details>
<summary>中文题目</summary>

1. 写出 GAIL 的目标函数,解释每一项的含义;它最小化的是专家与策略轨迹分布之间的什么散度?
2. GAIL 相比 BC 为什么能缓解复合误差?判别器提供了什么样的信号?
3. 最大熵 IRL 的假设 $p(\tau) \propto \exp(R(\tau))$ 是什么?它解决了 IRL 的什么病态问题 (如恒定奖励平凡解)?
4. GAIL 训练中判别器 (即隐式奖励) 的常见问题?熵正则 $-\lambda \mathcal{H}(\pi)$ 的作用?
5. IRL/GAIL 与 RLHF 的奖励建模同为『从行为反推偏好』,两者区别在哪里?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/gail-inverse-rl?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gail-inverse-rl)

---

## 13. Group Relative Policy Optimization · GRPO 组相对策略优化

> GRPO (Group Relative Policy Optimization) is the online policy-optimization algorithm used by DeepSeekMath/R1 for RLVR-style reasoning RL.

**Interview questions**

1. Write GRPO's group-relative advantage formula and explain why group normalization removes the critic?
2. Write GRPO's full objective (ratio, clip, KL term) and explain every symbol?
3. Why is GRPO more memory-efficient than PPO? Compare memory footprints (critic/GAE)?
4. Role of the KL penalty? Why the non-normalized KL estimator? How does group normalization resist reward-scale drift?
5. GRPO vs PPO vs DPO: use cases and selection criteria?

<details>
<summary>中文题目</summary>

1. 写出 GRPO 组内 Advantage 公式,并解释为什么组内归一化能免去 Critic?
2. 写出 GRPO 完整目标函数 (概率比、clip、KL 项),说明各符号含义?
3. GRPO 相比 PPO 为什么省显存?对比两者的显存构成 (价值网络/GAE)?
4. KL 惩罚项的作用?为什么用非归一化 KL 估计?组内相对化如何抗奖励尺度漂移?
5. GRPO、PPO、DPO 三者的适用场景与选择标准?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/grpo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=grpo)

---

## 14. Implicit Q-Learning · IQL expectile 回归

> Implicit Q-Learning (IQL) is a SOTA offline RL baseline built on expectile regression $\mathcal{L}_2^\tau(u) = |\tau - \mathbb{1}(u<0)| u^2$: for residual $u = r + \gamma Q(s',a') - Q(s,a)$, positive residuals get weight $\tau$ and negative ones $1-\tau$;

**Interview questions**

1. How does IQL's expectile loss $\mathcal{L}_2^\tau(u) = |\tau - \mathbb{1}(u<0)| u^2$ work? Which quantile does $\tau \to 1$ target?
2. Why does IQL never evaluate OOD actions? How does that fundamentally differ from CQL's explicit OOD penalization?
3. How do value extraction and policy extraction combine in IQL? What is the actor's objective?
4. How does the expectile implicitly approximate $\max_{a'} Q(s',a')$? The bias-variance trade-off?
5. How do IQL and CQL differ in offline-to-online settings, and why?

<details>
<summary>中文题目</summary>

1. IQL 的 expectile 回归损失 $\mathcal{L}_2^\tau(u) = |\tau - \mathbb{1}(u<0)| u^2$ 如何工作?$\tau \to 1$ 时回归到哪个分位?
2. IQL 为什么不需要评估 OOD 动作?它与 CQL 的『显式压低 OOD Q』有何根本区别?
3. IQL 中价值提取与策略提取如何配合?actor 的目标函数是什么?
4. expectile 如何隐式近似 $\max_{a'} Q(s',a')$?它的 bias-variance 权衡是什么?
5. IQL 与 CQL 在 offline-to-online 场景中的表现差异及原因?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/iql?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=iql)

---

## 15. Long-Term Retention RL · 长效留存延迟奖励

> Long-term retention RL models D1/D7 retention and LTV as delayed rewards: the true return of today's action (e.g., a recommended item) is observed days or weeks later, so a discount factor $\gamma$ is introduced to model cross-period value, with long-term obje…

**Interview questions**

1. Why do delayed rewards make credit assignment hard, and what role does the discount $\gamma$ play?
2. Formula: $J = \mathbb{E}[\sum_t \gamma^t r_t]$ — what business objectives do $\gamma \to 0$ and $\gamma \to 1$ correspond to?
3. Why does optimizing CTR alone hurt retention, and how do you design rewards for both short- and long-term?
4. Attributing delayed feedback: TD bootstrapping vs MC, and how surrogate rewards speed up learning?
5. How do you plug retention as delayed feedback into bandits (delayed-reward bandits) or RL?

<details>
<summary>中文题目</summary>

1. 为什么延迟奖励会造成 credit assignment 困难?折扣因子 $\gamma$ 的作用是什么?
2. 公式题: $J = \mathbb{E}[\sum_t \gamma^t r_t]$, 解释 $\gamma \to 0$ 与 $\gamma \to 1$ 分别对应什么业务目标?
3. 为什么只看 CTR 会伤害留存?如何设计兼顾短期与长期的奖励?
4. 延迟反馈的归因方法: TD 自举 vs MC, 代理奖励 (surrogate) 如何加速学习?
5. 如何把次留作为延迟反馈接入 MAB/在线学习 (延迟奖励 bandit) 或 RL?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/long-term-retention?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=long-term-retention)

---

## 16. MC vs TD Learning · 蒙特卡洛 vs 时序差分

> Monte Carlo (MC) updates from the full episode return, defined as discounted cumulative reward $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$, where $T$ is the terminal time step, $r_{t+k+1}$ the reward $k+1$ steps after $t$, and $\gamma^k$ the discount;

**Interview questions**

1. Write $G_t$ and the TD target in full and explain each term. How do the two updates differ?
2. Why is MC unbiased with high variance, and TD biased with low variance? Where do the bias and variance come from?
3. What does TD's "bootstrapping" mean? Its key differences from MC?
4. How does TD(λ) unify MC and TD? What do λ = 0 and λ = 1 reduce to?
5. Why does TD dominate in practice? How does the TD error appear in DQN/Q-learning updates?

<details>
<summary>中文题目</summary>

1. 写出 MC 回报 Gt 与 TD 目标的完整公式并逐项解释?两者的更新规则有何不同?
2. 为什么说 MC 无偏高方差、TD 有偏低方差?偏差和方差各来自哪里?
3. TD 的"自举" (bootstrap) 指什么?与 MC 的主要区别?
4. TD(λ) 如何统一 MC 与 TD?λ 取 0 和 1 各退化成什么?
5. 为什么 TD 是工业界主流?DQN/Q-learning 中 TD 误差如何参与更新?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/monte-carlo-td?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=monte-carlo-td)

---

## 17. MCTS Planning · MCTS 规划

> MCTS (Monte Carlo Tree Search) plans sequential decisions as tree search, looping over four steps: selection (pick nodes by UCT $UCT = \bar{v} + c\sqrt{\frac{\ln N}{n}}$, balancing exploration vs exploitation, with $N$ the parent's visit count and $n$ the chil…

**Interview questions**

1. What do MCTS's four steps do? Write the UCT formula and explain each symbol?
2. How does MCTS combine with policy-value networks (AlphaGo/AlphaZero self-play)?
3. How are nodes/actions/values defined for MCTS in LLM reasoning? Why PRM values?
4. MCTS vs plain sampling (best-of-N): compute complexity vs gains?
5. How does MuZero run MCTS in latent space? Difference from explicit dynamics models?

<details>
<summary>中文题目</summary>

1. MCTS 四步 (选择/扩展/模拟/回溯) 分别做什么?写出 UCT 公式及符号含义?
2. MCTS 与策略-价值网络如何结合 (AlphaGo/AlphaZero 自我对弈范式)?
3. LLM 推理中 MCTS 的节点、动作、价值分别如何定义?为什么用 PRM 做价值?
4. MCTS 与简单多次采样 (best-of-N) 的计算复杂度与收益对比?
5. MuZero 如何在潜空间内做 MCTS?与显式动态模型的区别?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/mcts?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mcts)

---

## 18. MDP & Bellman Equations · MDP 与 Bellman 方程

> An MDP (Markov Decision Process) models sequential decisions as a 5-tuple $(S, A, P, R, \gamma)$: $S$ states, $A$ actions, $P(s'|s,a)$ transition probabilities, $R(s,a)$ immediate rewards, and $\gamma \in [0,1)$ discount factor.

**Interview questions**

1. Write the MDP 5-tuple and explain each element. What role does the discount γ play?
2. Derive and write out the Bellman optimality equations for V* and Q*, explaining every term.
3. How are V* and Q* related? Why does the Bellman equation embody "optimal substructure"?
4. Difference between the Bellman expectation and optimality equations? How is Vπ defined for a given policy π?
5. Why does γ < 1 guarantee a bounded, convergent value function? What happens at γ = 1 or γ = 0?

<details>
<summary>中文题目</summary>

1. 写出 MDP 五元组并逐项解释?折扣因子 γ 的作用是什么?
2. 手推并完整写出 Bellman 最优方程的 V* 与 Q* 形式,逐项解释每个符号的含义?
3. V* 与 Q* 的关系?为什么说 Bellman 方程体现了"最优子结构"?
4. Bellman 期望方程与最优方程的区别?给定策略 π 的 Vπ 如何定义?
5. 为什么 γ < 1 能保证值函数有界且迭代收敛?γ = 1 或 γ = 0 时各有什么问题?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/mdp-bellman?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mdp-bellman)

---

## 19. Model Predictive Control (MPC) · MPC 规划 (Model Predictive Control)

> MPC (Model Predictive Control) uses a known or learned environment model to plan over a finite horizon $H$ at every decision step, maximizing cumulative return: $$J = \sum_{k=0}^{H-1} \gamma^k \, r(s_k, a_k)$$with $s_{k+1} \sim P_\phi(\cdot | s_k, a_k)$ rolled…

**Interview questions**

1. Write MPC's planning objective and explain the receding horizon and why it matters.
2. How do Random Shooting and CEM search action sequences? CEM's iterative procedure?
3. MPC vs a learned policy (e.g. π from PPO/SAC): pros and cons?
4. How do model error and finite horizon H affect MPC? What if H is too small?
5. Typical MPC usage in robotics/autonomous driving (trajectory optimization + closed-loop replanning)?

<details>
<summary>中文题目</summary>

1. 写出 MPC 的规划目标,并解释滚动时域 (receding horizon) 的含义与作用。
2. Random Shooting 与 CEM 分别如何搜索动作序列?CEM 的迭代流程?
3. MPC 相比学习策略函数(如 PPO/SAC 学出的 π)有哪些优缺点?
4. 模型误差与有限时域 H 如何影响 MPC?H 取太小会怎样?
5. MPC 在机器人/自动驾驶中的典型用法?(轨迹优化 + 闭环重规划)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/mpc?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mpc)

---

## 20. Model-Based vs Model-Free · 基于模型 vs 无模型 (MB vs MF)

> The two RL paradigms contrasted: | Aspect | Model-Based | Model-Free | | --- | --- | --- | | Sample efficiency | High (imagined rollouts reuse data) | Low (needs vast real interaction) | | Model bias | Yes (a wrong model misleads the policy) | None (learns fro…

**Interview questions**

1. Core differences between model-based and model-free: sample efficiency, model bias, compute?
2. Derive the compounding error O(H²ε): give the intuition and the summation.
3. When to choose model-based vs model-free (interaction cost, goal variability, data scale)?
4. How to hybridize the two to mitigate each other's flaws (Dyna, MBPO, offline RL)?
5. How to quantify/detect model bias? How to plan robustly under model uncertainty?

<details>
<summary>中文题目</summary>

1. 基于模型与无模型在样本效率、模型偏差、计算开销上的核心差异?
2. 推导复合误差 O(H²ε):给出直觉与求和过程。
3. 什么时候选基于模型,什么时候选无模型?(交互成本、目标可变性、数据规模)
4. 如何混合两者缓解各自缺点?(Dyna、MBPO、离线 RL 的具体做法)
5. 模型偏差如何量化/检测?规划时如何对模型不确定性保持鲁棒?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/model-vs-model-free?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=model-vs-model-free)

---

## 21. Multi-Agent RL (CTDE) · 多智能体 CTDE/MAPPO

> MARL (Multi-Agent Reinforcement Learning) trains multiple agents simultaneously. From each agent's perspective the environment is non-stationary — other agents' policies keep evolving — so naively applying single-agent algorithms is unstable.

**Interview questions**

1. Explain CTDE: what do centralized training and decentralized execution mean, and why are they needed?
2. How does MAPPO's shared centralized Critic differ from single-agent PPO; how is the advantage computed?
3. How does MADDPG's centralized critic Q_i(s,a₁..a_N) mitigate non-stationarity, and how does it differ from MAPPO?
4. Whiteboard: write MAPPO's Actor gradient and advantage formula, and relate them to PPO / the policy gradient theorem.
5. The three MARL challenges — non-stationarity, credit assignment, exponential action space — and mitigations for each.

<details>
<summary>中文题目</summary>

1. 解释 CTDE:集中训练与分布执行分别指什么?为什么要这样做?
2. MAPPO 的共享集中式 Critic 与单智能体 PPO 有何不同?优势函数如何计算?
3. MADDPG 的集中式 Critic Q_i(s,a₁..a_N) 如何缓解环境非平稳?与 MAPPO 有何异同?
4. 白板手推: 写出 MAPPO 的 Actor 梯度与优势计算公式,并说明其与 PPO/策略梯度定理的继承关系?
5. MARL 三大难点:非平稳、credit assignment、指数动作空间,各有哪些缓解手段?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/marl?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=marl)

---

## 22. Multi-Armed Bandit (MAB) · 多臂老虎机 MAB

> The Multi-Armed Bandit (MAB) is a single-state, no-state-transition online decision problem: at each round pick an action $a_t$ among $K$ arms and observe a stochastic reward $r_t$ (e.g., Bernoulli click $r_t \sim \text{Bernoulli}(\mu_{a_t})$).

**Interview questions**

1. Write the regret definition and explain each term: why use the best arm's mean $\mu^*$ instead of realized rewards?
2. Numerical: 3 arms with $\mu=(0.8, 0.5, 0.3)$; over 100 rounds arm1 is chosen 60 times, arms 2/3 20 each — compute the regret?
3. How does MAB differ from full RL (MDP)? How does it degenerate from an MDP?
4. Core ideas and regret orders of ε-greedy, UCB, and Thompson sampling?
5. Lower bound: why is any algorithm's worst-case regret at least $\Omega(\sqrt{KT})$?

<details>
<summary>中文题目</summary>

1. 写出 regret 定义并解释每一项:为何用最优臂期望 $\mu^*$ 而非实际收益?
2. 数值题:3 臂 $\mu=(0.8, 0.5, 0.3)$, 玩 100 轮, 臂1 被选 60 次、臂2/3 各 20 次, 计算 regret?
3. MAB 与完整 RL (MDP) 的差别?MAB 如何从 MDP 退化而来?
4. ε-greedy、UCB、Thompson 采样各自的核心思想与 regret 量级?
5. regret 下界:为什么任何算法的 worst-case regret 至少是 $\Omega(\sqrt{KT})$?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/mab?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mab)

---

## 23. MuZero Latent Model · MuZero 潜空间模型

> MuZero extends AlphaZero's tree search to environments with unknown rules by learning dynamics and reward in latent space with three neural functions: ① representation: $s^0_t = h_\theta(s_{t-1}, a_{t-1})$ — compresses (history of) observations into a latent s…

**Interview questions**

1. MuZero's three functions (representation/dynamics/prediction): inputs, outputs, and roles?
2. How does MuZero plan without an explicit real-reward model? How is latent reward prediction trained?
3. MuZero's MCTS vs AlphaZero's MCTS: what differs?
4. How are MuZero's policy/value training targets constructed from search?
5. In what sense does MuZero generalize beyond AlphaZero to the real world?

<details>
<summary>中文题目</summary>

1. MuZero 三个函数(representation/dynamics/prediction)的输入输出与各自作用?
2. MuZero 如何不显式建模真实奖励也能规划?潜空间奖励预测如何训练?
3. MuZero 的 MCTS 与 AlphaZero 的 MCTS 有什么区别?
4. MuZero 的策略/价值训练目标如何由搜索构造?
5. 相比 AlphaZero,MuZero 在何种意义上能推广到真实世界?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/muzero?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=muzero)

---

## 24. Offline Distribution Shift · Offline 分布偏移

> Distribution shift is the core difficulty of offline RL: the value function and policy learn only from a fixed dataset $\mathcal{D}$, yet during training the policy keeps querying Q on out-of-distribution (OOD) state-action pairs the data never covered;

**Interview questions**

1. What is the mechanism of extrapolation error? Why do Q-estimates for OOD actions inflate systematically?
2. How does Bellman bootstrapping amplify extrapolation error? Write the error-propagation chain in $Q(s,a)=r+\gamma \max_{a'} \hat{Q}(s',a')$ and explain the max operator's role.
3. How does offline distribution shift differ from, and relate to, online DQN overestimation?
4. Why must an offline policy stay in-distribution? Enumerate three mitigation families (policy constraint, conservative Q, data generation) with one algorithm each.
5. Given a poorly covered fixed dataset, how would you diagnose extrapolation error from Q-values, and what training metric warns of policy collapse?

<details>
<summary>中文题目</summary>

1. 外推误差 (Extrapolation Error) 的产生机制是什么?为什么 Q 对 OOD 动作的估计会系统性虚高?
2. Bellman 自举如何放大外推误差?写出 $Q(s,a)=r+\gamma \max_{a'} \hat{Q}(s',a')$ 的误差传播链,解释 max 算子的放大作用。
3. Offline 分布偏移与在线 DQN 的 overestimation 问题有何区别与联系?
4. 为什么说离线 RL 的策略必须『待在数据分布内』?列出三类缓解思路并各举一例算法 (策略约束 / 保守 Q / 数据生成)。
5. 给定覆盖度差的固定数据集,如何从 Q 值诊断外推误差?哪个训练指标能预警策略崩溃?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/distribution-shift?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distribution-shift)

---

## 25. Offline-to-Online Fine-Tuning · 离线到在线微调

> Offline-to-online RL first trains a conservative policy on fixed offline data, then keeps optimizing through online interaction;

**Interview questions**

1. Why must conservatism be annealed in offline-to-online? What happens if you go online with a large penalty coefficient?
2. What is the role of replay mixing? How does the new/old data sampling ratio affect stability?
3. CQL or IQL for offline-to-online? Why is IQL often more stable?
4. How does RLHF relate to offline-to-online? How does the reference-policy KL constraint act as 'conservatism'?
5. How do you schedule the anneal from conservative to fully online? When can old-data replay be dropped?

<details>
<summary>中文题目</summary>

1. 为什么 offline-to-online 需要保守惩罚退火?带着过大的保守系数直接上线会怎样?
2. 回放混合 (Replay Mixing) 的作用?新/旧数据采样比例如何影响训练稳定性?
3. CQL 与 IQL 谁更适合 offline-to-online?为什么 IQL 常表现更稳?
4. RLHF 与 offline-to-online 的联系?参考策略的 KL 约束如何充当『保守惩罚』?
5. 如何设计从保守到纯在线的退火判据?何时可以停止回放旧数据?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/offline-to-online?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=offline-to-online)

---

## 26. Policy Gradient Theorem · 策略梯度 REINFORCE

> The policy gradient theorem gives a model-free gradient of $J(\theta)=\mathbb{E}_{\tau}[R(\tau)]$; the core is the log-derivative trick and the expectation form $\nabla_\theta J = \mathbb{E}_{\tau}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) G_t]$.

**Interview questions**

1. Whiteboard: derive the policy gradient theorem from J(θ)=E_τ[R(τ)]; why do the dynamics terms cancel?
2. What is the log-derivative trick, and why does it yield an unbiased gradient estimator?
3. Prove E[∇log π(a|s)·b(s)] = 0 and explain why a baseline reduces variance without biasing the gradient.
4. Why does reward-to-go G_t have lower variance than the full return? How does REINFORCE relate to MC methods?
5. Root causes of high variance and step-size sensitivity; which improvements (TRPO/PPO/GAE) emerged?

<details>
<summary>中文题目</summary>

1. 白板手推: 从 J(θ)=E_τ[R(τ)] 出发推导策略梯度定理,说明动力学项为何消去?
2. Log-Derivative Trick 的原理?为什么由此得到的梯度估计是无偏的?
3. 证明 E[∇log π(a|s)·b(s)] = 0,并解释基线为什么能降方差而不改变梯度期望?
4. reward-to-go G_t 与整段回报相比为什么方差更低?REINFORCE 与 MC 的关系?
5. 策略梯度高方差与步长敏感的根源?演化出哪些改进(TRPO/PPO/GAE)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/policy-gradient?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=policy-gradient)

---

## 27. PPO Clipped Objective · PPO 截断目标

> PPO (Proximal Policy Optimization) approximates TRPO's KL trust region with a clipped probability ratio, needing only first-order optimization.

**Interview questions**

1. Write PPO's clipped objective; what is the default ε, and why the min?
2. Whiteboard: with A=+1,r=2 and A=−1,r=0.5 show numerically how min+clip bounds the update.
3. What does the ratio r_t(θ) mean, and what role does importance sampling play?
4. Why do positive and negative advantages clip in opposite directions; when is the clip active?
5. How does PPO relate to TRPO, and why is PPO the default for RLHF/RLVR?

<details>
<summary>中文题目</summary>

1. 写出 PPO 截断目标 min(r·A, clip(r,1−ε,1+ε)·A),ε 默认取多少?为什么要 min?
2. 白板手推: 用数值例(A=+1,r=2 与 A=−1,r=0.5)演示 min+clip 如何限制更新幅度?
3. 概率比 r_t(θ) 的意义?重要性采样在其中扮演什么角色?
4. Advantage 为正和为负时截断为何方向相反?clip 何时生效、何时不生效?
5. PPO 与 TRPO 的关系?为什么 RLHF/RLVR 默认用 PPO?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/ppo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ppo)

---

## 28. Process Reward Model · 过程奖励模型 PRM

> A PRM (Process Reward Model) scores each intermediate reasoning step, refining credit assignment from 'was the answer right' to 'which step went wrong': outcome reward models (ORMs) give one overall 0/1 and cannot tell the policy which step to fix, while step-…

**Interview questions**

1. PRM vs ORM: differences? How does a PRM refine credit assignment?
2. How are PRM training labels obtained (human step labels / Monte-Carlo MC-EST)?
3. What role does a PRM play in MCTS and best-of-N reranking?
4. Main PRM challenges (label noise, error accumulation, generator/verifier coverage mismatch)?
5. How do RLVR outcome rewards and PRM process rewards complement each other? Trade-offs?

<details>
<summary>中文题目</summary>

1. PRM 与 ORM (结果奖励模型) 的区别?PRM 细化 credit assignment 的原理?
2. PRM 的训练标签如何获得 (人工逐步标注 / 蒙特卡洛 MC-EST 估计)?
3. PRM 在 MCTS 与 best-of-N 重排中各起什么作用?
4. PRM 的主要挑战 (标签噪声、误差累积、生成器与验证器的覆盖偏差)?
5. RLVR 结果奖励与 PRM 过程奖励如何配合?各自的取舍?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/prm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=prm)

---

## 29. Q-Learning (Off-Policy) · Q-Learning 离策略

> Q-Learning is a temporal-difference off-policy control algorithm that learns the optimal action-value function $Q^*$ directly, with update $Q(s,a) \leftarrow Q(s,a) + \alpha\left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$, where $\alpha$ is the learning…

**Interview questions**

1. Write the Q-learning update in full and explain α, γ, the max term, and the TD error.
2. Why is Q-Learning off-policy? What are its behavior and target policies?
3. Convergence conditions for tabular Q-learning? What are the Robbins-Monro step-size conditions?
4. The overestimation problem caused by the max operator? How does Double Q-Learning fix it?
5. Q-Learning vs SARSA? How do they behave differently in the Cliff Walking environment?

<details>
<summary>中文题目</summary>

1. 手推并完整写出 Q-Learning 更新公式,逐项解释 α、γ、max 项与 TD 误差?
2. 为什么 Q-Learning 是离策略算法?行为策略与目标策略分别是什么?
3. 表格型 Q-Learning 的收敛条件?Robbins-Monro 步长条件是什么?
4. max 算子带来的过估计 (overestimation) 问题?Double Q-Learning 如何解决?
5. Q-Learning 与 SARSA 的区别?悬崖行走 (Cliff Walking) 场景下两者的行为差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/q-learning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=q-learning)

---

## 30. Reward Design & Hacking · 奖励设计与 Hacking

> Reward design comes in three forms: sparse rewards ($\pm 1$ only at termination — unbiased but hard to explore), dense rewards (signal every step — faster learning but potentially misleading), and potential-based reward shaping — augmenting the reward with $$F…

**Interview questions**

1. Pros and cons of sparse vs dense rewards? How to choose per task?
2. Write the shaping formula F = γΦ(s') − Φ(s) and explain why it preserves the optimal policy (Ng's theorem)?
3. What is reward hacking? Give an LLM-scenario example (RLHF/RLVR)?
4. How to mitigate reward hacking? Why are verifiable rewards immune?
5. Best practices for reward design in LLM reasoning RL (verifiable rewards vs reward models vs PRMs)?

<details>
<summary>中文题目</summary>

1. 稀疏奖励与密集奖励各自的优缺点?如何根据任务选择?
2. 写出势函数 shaping 公式 F = γΦ(s') − Φ(s),并说明为什么它不改变最优策略 (Ng 定理)?
3. 什么是 reward hacking?举一个 LLM 场景 (RLHF/RLVR) 的例子?
4. 如何缓解 reward hacking?可验证奖励为什么天然免疫?
5. LLM 推理 RL 中奖励设计的最佳实践 (可验证奖励 vs 奖励模型 vs PRM)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/reward-design?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=reward-design)

---

## 31. RL with Verifiable Rewards · RLVR 可验证奖励

> RLVR (Reinforcement Learning from Verifiable Rewards) uses rule/program-verifiable signals as rewards, replacing human or AI reward models: exact-answer matching for math (parsed to canonical form);

**Interview questions**

1. Types of verifiable rewards in RLVR? Why is RLVR considered unhackable?
2. How do RLVR and GRPO combine (G samples, group-relative advantage, no critic)?
3. How is the pass@k reward defined for code? How does it differ from answer matching?
4. Which tasks suit RLVR? What verifiability is required, and what about open-ended tasks?
5. RLVR vs reward-model RLHF: compare cost, bias, and robustness?

<details>
<summary>中文题目</summary>

1. RLVR 的可验证奖励有哪几类?为什么说它是不可被黑客攻击 (unhackable) 的?
2. RLVR 如何与 GRPO 组合 (采样 G 个回答、组内 Advantage、免 Critic)?
3. 代码任务中 pass@k 奖励如何定义?与数学答案匹配有何不同?
4. 哪些任务适合 RLVR?对任务的验证性有什么要求?开放式任务怎么办?
5. RLVR 与奖励模型 RLHF 的对比:成本、偏差、鲁棒性三方面?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/rlvr?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rlvr)

---

## 32. SAC Max-Entropy · SAC 最大熵

> SAC (Soft Actor-Critic) is a maximum-entropy off-policy Actor-Critic whose objective augments cumulative reward with a policy entropy regularizer:…

**Interview questions**

1. Write SAC's objective and explain the meaning of maximum entropy.
2. Whiteboard: derive the soft Bellman equation and the closed-form improvement π∝exp(Q/α) from the entropy-regularized objective.
3. What does twin-Q min solve, and how does it differ from Double DQN?
4. Design of the auto-temperature loss J(α); does α increase or decrease when entropy is below the target?
5. Why is SAC the default for continuous control? Benefits of off-policy + stochastic policy + entropy?

<details>
<summary>中文题目</summary>

1. 写出 SAC 的目标 J=E[Σ(r_t + αH(π(·|s_t)))],解释最大熵的意义?
2. 白板手推: 从熵正则目标推导软 Bellman 方程与策略改进的闭式解 π∝exp(Q/α)?
3. 双 Q 取 min 解决什么问题?与 Double DQN 有何异同?
4. 温度 α 自动调节的损失 J(α) 如何设计?熵低于目标时 α 增大还是减小?
5. SAC 为什么是连续控制的默认选择?离策略+随机策略+熵正则的组合优势?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/sac?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=sac)

---

## 33. Slate / Combinatorial Bandit · Slate/组合老虎机

> A Slate Bandit (combinatorial bandit) extends actions from single items to a whole page-level list: each round pick $K$ items out of $N$ candidates to form a slate $S$ with $|S| = K$, i.e.

**Interview questions**

1. Combinatorics: how many slates for $N=10, K=3$? Why can't we estimate every super-arm individually?
2. Top-K combinatorial exploration: how do item-level UCB/TS scores with top-K selection guarantee exploration?
3. Difference between item-level and slate-level reward modeling, and when to use each?
4. How does position bias distort estimates, and how do you correct for it?
5. Modeling dependencies within a slate: diversity, mutual exclusion, and click co-occurrence constraints?

<details>
<summary>中文题目</summary>

1. 组合复杂度: $N=10, K=3$ 时有多少种 slate?为什么不能逐一估计每个超臂?
2. Top-K 组合探索的公式: 如何在 item 级做 UCB/TS 并取前 K, 探索如何保证?
3. item 级奖励与 slate 级奖励建模的差别?各自适用场景?
4. 位置偏差 (position bias) 如何影响估计?如何修正?
5. slate 内 item 间的依赖如何建模: 多样性/互斥/协同点击的约束?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/slate-bandit?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=slate-bandit)

---

## 34. Thompson Sampling · Thompson 采样

> Thompson Sampling (TS, probability matching) is a Bayesian exploration algorithm: maintain a posterior over each arm's mean reward;

**Interview questions**

1. Derive the Beta posterior: prior $\text{Beta}(\alpha, \beta)$ + Bernoulli observations → $\text{Beta}(\alpha + w, \beta + l)$?
2. Numerical: prior $\text{Beta}(1,1)$; arm1 has 8 wins/2 losses, arm2 3 wins/7 losses — write both posteriors and argue which arm the next sample favors.
3. TS vs UCB regret: why does TS often perform better in practice?
4. How does TS degrade under non-stationary rewards, and what fixes exist?
5. The probability-matching property: play probability equals posterior probability of optimality — what does it imply for exploration?

<details>
<summary>中文题目</summary>

1. 推导 Beta 后验:先验 $\text{Beta}(\alpha, \beta)$ + Bernoulli 观察 → 后验 $\text{Beta}(\alpha + w, \beta + l)$?
2. 数值题:先验 $\text{Beta}(1,1)$, 臂1 赢 8 输 2、臂2 赢 3 输 7, 写出两个后验并说明下一次采样更可能选哪个臂?
3. TS 与 UCB 的 regret 对比?为什么工业界 TS 实际表现常更好?
4. 非平稳环境 (奖励漂移) 下 TS 如何退化?有哪些修正?
5. probability matching 性质:选臂概率 = 该臂为最优的后验概率, 这对探索意味着什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/thompson-sampling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=thompson-sampling)

---

## 35. TRPO → PPO Lineage · TRPO 与 PPO 脉络

> TRPO (Trust Region Policy Optimization) constrains the single-step policy update via a KL trust region to guarantee monotonic improvement: $\max_\theta\;

**Interview questions**

1. Write TRPO's constrained problem: the surrogate objective and the KL constraint E[D_KL]≤δ.
2. Whiteboard: linearize the objective, expand KL to second order, derive the natural gradient Δθ=αF⁻¹g, and explain why the Fisher matrix approximates the Hessian.
3. How does TRPO guarantee monotonic improvement? Roles of line search and the coefficient C=4εγ/(1-γ)².
4. The TRPO → PPO lineage: how does ratio clipping approximate the KL trust region?
5. Where is TRPO's computational bottleneck (Hessian/conjugate gradient), and how is it accelerated in practice?

<details>
<summary>中文题目</summary>

1. 写出 TRPO 的约束优化问题:替代目标与 KL 约束 E[D_KL]≤δ 分别是什么?
2. 白板手推: 目标线性化 + KL 二阶展开,推导自然梯度方向 Δθ=αF⁻¹g,说明 Fisher 矩阵为何是海森近似?
3. TRPO 如何保证单调改进?线性搜索与系数 C=4εγ/(1-γ)² 的作用?
4. TRPO 与 PPO 的脉络:概率比 clip 如何近似 KL 信任域?
5. TRPO 的计算瓶颈(海森/共轭梯度)在哪?实践中如何加速?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/trpo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=trpo)

---

## 36. UCB1 (Upper Confidence Bound) · UCB1 乐观估计

> UCB1 is a frequentist optimistic exploration algorithm: each round picks the arm with the largest confidence upper bound $a_t = \arg\max_a \left[ \hat{\mu}_a + \sqrt{\frac{2\ln t}{n_a}} \right]$, where $\hat{\mu}_a$ is the empirical mean of arm $a$, $n_a$ its…

**Interview questions**

1. Meaning of each term in UCB1? Why is $\sqrt{2\ln t / n_a}$ the optimal exploration bonus (Hoeffding)?
2. Numerical: at $t=100$, 3 arms with $(n_a, \hat{\mu}_a)$: arm1 (60, 0.6), arm2 (30, 0.5), arm3 (10, 0.4) — compute each UCB value and pick the arm.
3. Sketch the $O(\ln T)$ regret proof: why is each bad arm's play count bounded by $O(\ln T)$?
4. How does UCB's exploration fundamentally differ from Thompson sampling and ε-greedy?
5. Limitations and extensions of UCB1: non-stationary environments, reward assumptions, UCB1-tuned / sliding-window UCB?

<details>
<summary>中文题目</summary>

1. UCB1 公式中每项的含义?为何 $\sqrt{2\ln t / n_a}$ 是最优探索量 (Hoeffding)?
2. 数值题: $t=100$, 3 臂统计 $(n_a, \hat{\mu}_a)$: 臂1 (60, 0.6)、臂2 (30, 0.5)、臂3 (10, 0.4), 手算 UCB 值并指出选哪个臂?
3. 推导 UCB regret $O(\ln T)$ 的思路:为什么坏臂被选的次数有 $O(\ln T)$ 上界?
4. UCB 与 Thompson 采样、ε-greedy 的探索方式本质区别?
5. UCB1 的局限与扩展:非平稳环境、奖励分布假设、UCB1-tuned / sliding-window UCB?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/ucb?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ucb)

---

## 37. Value vs Policy Iteration · 价值迭代 vs 策略迭代

> Value Iteration (VI) iterates the Bellman optimality equation directly with synchronous sweeps over all states: $V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)\left[ r(s,a) + \gamma V_k(s') \right]$, where $V_k$ is the estimate at round $k$, $\max_a$ performs an impl…

**Interview questions**

1. Write the value iteration update and explain each term. Why does VI converge to V*?
2. What are the two steps of policy iteration? Why does the improvement theorem guarantee monotonic gains?
3. Compare VI vs PI in complexity and convergence speed. Why does PI need fewer rounds but cost more per round?
4. Is an intermediate Vk necessarily the value function of some policy? How is the final policy extracted?
5. Given a small grid-world MDP, manually run 1-2 sweeps of VI/PI to demonstrate the updates.

<details>
<summary>中文题目</summary>

1. 手推并写出价值迭代的更新公式,逐项解释符号?为什么 VI 能收敛到 V*?
2. 策略迭代的两步是什么?策略改进定理为什么保证单调提升?
3. VI 与 PI 的复杂度与收敛速度对比?为什么 PI 轮数少但每轮贵?
4. VI 迭代中的 Vk 是否一定是某个策略的值函数?最终策略如何取?
5. 给定一个小格点 MDP,手动跑 1-2 轮 VI/PI 演示更新过程?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/value-iteration-policy?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=value-iteration-policy)

---

## 38. World Model Learning · 世界模型学习 (World Model)

> A world model parameterized by $\phi$ learns the environment's transition and reward distribution $P_\phi(s', r | s, a)$: given state $s$ and action $a$, predict next state $s'$ and immediate reward $r$.

**Interview questions**

1. What is the world model's training objective? Write the MLE objective and explain each term.
2. Why does a world model improve sample efficiency? Imagined rollouts vs real interaction?
3. How does model error hurt the downstream policy? What is compounding error O(H²ε)?
4. Explicit (pixel/state-level) vs implicit (latent) world models — differences? One example each?
5. Beyond classic RL, what new world-model applications exist for LLMs/agents?

<details>
<summary>中文题目</summary>

1. 世界模型学习的目标函数是什么?写出 MLE 训练目标及各项含义。
2. 世界模型为什么能提升样本效率?想象 rollout 与真实交互的差异?
3. 模型误差如何影响下游策略?什么是复合误差 O(H²ε)?
4. 显式(像素/状态级)与隐式(潜空间)世界模型的区别?各举一例。
5. 除了经典 RL,世界模型在 LLM/Agent 场景有哪些新应用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/RL/card/world-model?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=world-model)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
