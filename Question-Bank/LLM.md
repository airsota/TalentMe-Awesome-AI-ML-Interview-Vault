# ⚡ Large Language Models · 大语言模型 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **48 cards · 240 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. Agent & Tool Calling · Agent 与工具调用

> agents give LLMs external capabilities through tool calling, built on three mechanisms: (1) Function-calling schema — declaring each API as name + description + parameters (JSON Schema);

**Interview questions**

1. Walk the function-calling pipeline (schema → output → validation → execution → feedback) and the failure handling at each step.
2. How does grammar-constrained decoding work, and how does JSON mode differ from normal decoding in guaranteeing parseability?
3. How does the ReAct loop terminate — max steps and loop detection?
4. How do agents self-correct — retry vs switch tools on timeout, bad args, or empty results?
5. With many tools, how do you prevent tool-selection hallucination — tool descriptions, routing, and result verification?

<details>
<summary>中文题目</summary>

1. Function Calling 的完整链路:Schema 声明 → 模型输出 → 参数校验 → 执行 → 回填,每一步的失败处理怎么做?
2. 语法约束解码的原理是什么?JSON mode 与普通解码在保证可解析性上的区别?
3. ReAct 的 Thought-Action-Observation 循环如何实现终止?最大步数与循环检测怎么设?
4. Agent 如何自我纠错?工具调用失败(超时、参数错、结果空)时应该重试还是换工具?
5. 多工具场景如何防止工具选择幻觉:工具描述怎么写、如何做工具路由与结果验证?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/agent-tool-calling?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=agent-tool-calling)

---

## 2. Alignment Tax & Preference Data · 对齐税与偏好数据

> Alignment Tax is the phenomenon where alignment training (RLHF/DPO) improves helpfulness/harmlessness but can degrade other capabilities (math, code, long-tail knowledge, output diversity, creativity), quantifiable as relative regression on a benchmark…

**Interview questions**

1. What is alignment tax? Typical manifestations and which capabilities are most affected?
2. How do you measure alignment tax, and how to mitigate it (KL coefficient, data mix, multi-objective)?
3. How does preference-data quality affect alignment, and what makes a good preference pair?
4. How is labeler agreement evaluated, and what does low agreement indicate?
5. Compare RLAIF (AI feedback) vs human preference data: pros and cons?

<details>
<summary>中文题目</summary>

1. 什么是对齐税?典型表现有哪些?举例哪些能力容易受损?
2. 如何度量对齐税?如何在训练中缓解(如 KL 系数、数据配比、多目标)?
3. 偏好数据的质量如何影响对齐效果?高质量偏好对的构造要点?
4. 标注者一致性 (Agreement) 如何评估?低一致性说明什么?
5. RLAIF (AI 反馈) 与人工偏好数据的优缺点对比?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/alignment-tax?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=alignment-tax)

---

## 3. Attention Variants MHA/MQA/GQA · 注意力变体 MHA/MQA/GQA

> MHA (Multi-Head Attention) gives every head its own K/V projections; MQA (Multi-Query Attention) shares a single K/V pair across all heads;

**Interview questions**

1. What does GQA save vs MHA? Meaning of each term in the KV-cache formula $2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$?
2. How to choose the number of groups G in GQA? The quality vs memory trade-off?
3. Difference between MQA and GQA? Pros and cons of each?
4. Why does sharing K/V barely affect training but matter a lot for inference (KV cache exists only at inference)?
5. Configs of mainstream models: KV-head counts for LLaMA-2/3, Mistral, Qwen?

<details>
<summary>中文题目</summary>

1. GQA 相比 MHA 省了什么?KV cache 显存公式 $2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$ 每项含义?
2. GQA 组数 G 如何选择?质量与显存的权衡?
3. MQA 与 GQA 的区别?各自优缺点?
4. 为什么共享 KV 对训练影响小、对推理影响大?(KV cache 只在推理出现)
5. 主流模型分别用什么配置?LLaMA-2/3、Mistral、Qwen 的 KV 头数?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/attention-variants?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=attention-variants)

---

## 4. Auto Metrics vs LLM-as-Judge · 自动指标 vs LLM-Judge

> three classic automatic metrics and their limits: (1) BLEU — n-gram precision with a brevity penalty, $\text{BLEU} = BP \cdot \exp\big(\sum_n w_n \log p_n\big)$, built for MT, blind to synonyms and legitimate paraphrases;

**Interview questions**

1. What are BLEU's three components (precision, n-grams, brevity penalty)? When does it give misleadingly high scores?
2. Perplexity definition and limits: why doesn't low PPL imply high generation quality?
3. In Cohen's Kappa, what are p_o and p_e, and why is it stricter than raw agreement?
4. How do you design experiments to remove the big three judge biases — position, self-preference, verbosity?
5. When are BLEU/ROUGE appropriate vs LLM-judge vs human eval, and how do you trade off cost?

<details>
<summary>中文题目</summary>

1. BLEU 的三个组成部分是什么(精确率、n-gram、长度惩罚 BP)?它在什么情况下给出误导性高分?
2. 困惑度的定义与局限:为什么低困惑度不代表生成质量高?
3. Cohen's Kappa 公式中 p_o 和 p_e 分别是什么?为什么比简单一致率更严格?
4. LLM-as-judge 的三大偏差(位置、自偏好、冗长)怎么设计实验消除?
5. 什么场景适合 BLEU/ROUGE,什么场景必须用 LLM-judge 或人工评测?成本权衡如何?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/eval-metrics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=eval-metrics)

---

## 5. Benchmarks MMLU/GSM8K · 评测基准 MMLU/GSM8K

> four representative benchmarks anchor different capability axes: (1) MMLU — 57-subject multiple choice, 5-shot CoT, world knowledge and reasoning breadth;

**Interview questions**

1. What capability does each of MMLU/GSM8K/HumanEval/IFEval measure, and why 5-shot rather than zero-shot?
2. Derive pass@k: with c passing out of n samples, why does the binomial formula give the correct estimate?
3. What are the signs and causes of benchmark saturation, and how do you respond (harder variants, dynamic evals, leakage control)?
4. How do you detect and avoid data contamination — dedup, public timelines, held-out sets?
5. GSM8K gains hugely from CoT yet trivial formatting changes help too — what does that imply, and how do you control prompt sensitivity in eval?

<details>
<summary>中文题目</summary>

1. MMLU、GSM8K、HumanEval、IFEval 各自测什么能力?为什么用 5-shot 而非 zero-shot?
2. pass@k 公式如何推导?n 个采样中 c 个通过时 pass@k 为什么用组合数表示?
3. 基准饱和的表现与成因是什么?有哪些应对(更难变体、动态评测、防泄漏)?
4. 数据污染如何检测与规避?去重、公开时间线、held-out 集分别怎么做?
5. GSM8K 上 CoT 收益大但简单换行也有效——说明什么?评测时如何控制提示敏感性?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/benchmarks?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=benchmarks)

---

## 6. Chain-of-Thought · 思维链 CoT

> CoT (Chain-of-Thought) prompting: include intermediate reasoning steps in examples so the model decomposes complex problems into stepwise reasoning instead of jumping to the answer.

**Interview questions**

1. Few-shot vs zero-shot CoT? Why does "Let's think step by step" work?
2. How does CoT emergence relate to model size, and why do small models benefit little?
3. How does self-consistency work, and why does majority voting improve accuracy?
4. Limitations of CoT? When is it ineffective or even harmful?
5. How does CoT relate to test-time compute scaling?

<details>
<summary>中文题目</summary>

1. Few-shot CoT 与 Zero-shot CoT 各是什么?为什么 “Let's think step by step” 有效?
2. CoT 的涌现性与参数量有什么关系?为什么小模型收益不大?
3. Self-consistency 的原理?为什么多数投票能提高准确率?
4. CoT 的局限有哪些?什么时候 CoT 无效甚至有害?
5. CoT 与推理时扩展 (test-time compute) 的关系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/cot?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cot)

---

## 7. Classic NLP Tasks · 经典 NLP 任务

> pre- vs post-LLM paradigms for four classic tasks. (1) Text classification — before: TF-IDF/BOW + LR/GBDT feature engineering, or BERT's [CLS] into $\text{softmax}(W^\top h_{\text{[CLS]}})$;

**Interview questions**

1. LLM vs BERT for classification: accuracy, cost, latency, and cold-start trade-offs?
2. Sequence-labeling NER (BIO, BiLSTM-CRF) vs LLM extraction: pros and cons of each?
3. Why was NMT replacing SMT a milestone, and what role did BLEU play?
4. How do you design Map-Reduce summarization for very long documents, and how do you mitigate information loss at merge?
5. LLM-to-small-model distillation pipeline: how do you build training data and measure distillation loss?

<details>
<summary>中文题目</summary>

1. 分类任务 LLM 与 BERT 范式对比:准确率、成本、延迟、冷启动分别什么表现?
2. NER 的序列标注路线(BIO、BiLSTM-CRF)与 LLM 抽取路线的优劣分别是什么?
3. 为什么说 NMT 取代 SMT 是深度学习的里程碑?BLEU 在其中扮演什么角色?
4. 超长文本摘要的 Map-Reduce 方案怎么设计?合并阶段的信息丢失如何缓解?
5. LLM 蒸馏到小模型的落地路径:如何构造标注数据、如何评估蒸馏损失?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/classic-tasks?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=classic-tasks)

---

## 8. Context Engineering · 上下文工程

> context engineering treats the finite context window as a scarce resource to manage systematically — three pillars: (1) system prompts — fixed role, behavior rules, and format constraints stable across turns, versioned over time (e.g.

**Interview questions**

1. What are the principles of system-prompt design, and how do you layer it against user input and retrieval to avoid override?
2. Typical memory tiers and their lifetimes; when and how does context compaction trigger?
3. For a 200K window, how do you budget instruction/history/retrieval/output, and why must output be reserved?
4. Under per-token pricing, when do you resend long history vs use summary/prompt caching?
5. Context engineering vs Lost-in-the-Middle: where in the window should critical info go to maximize usage?

<details>
<summary>中文题目</summary>

1. 系统提示设计有哪些原则?如何与用户输入、检索内容分层避免互相覆盖?
2. 记忆分层的常见层级与各层寿命是什么?上下文压缩 (compaction) 何时触发、怎么触发?
3. 给一个 200K 窗口设计 token 预算:指令/历史/检索/输出各分多少?为什么必须预留输出?
4. 上下文成本模型:输入/输出 token 计费下,长历史重发 vs 摘要缓存(如 prompt caching)怎么选?
5. 上下文工程与 Lost-in-the-Middle 的关系:关键信息放窗口哪个位置最能保证被利用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/context-engineering?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=context-engineering)

---

## 9. Decoding Strategies · 解码策略

> Decoding strategies decide how to pick the next token from the vocabulary distribution during generation. Greedy takes the argmax;

**Interview questions**

1. Role of temperature; what do T=0 and T→∞ degenerate to?
2. top-p vs top-k? Why is top-p more adaptive?
3. Why temperature before top-p? What if the order is swapped?
4. Drawbacks of greedy and beam? Effects of beam size?
5. Sampling vs deterministic decoding: suitable tasks and the trade-off?

<details>
<summary>中文题目</summary>

1. temperature 的作用与极限:$T=0$、$T \to \infty$ 分别等价于什么解码?
2. top-p 与 top-k 的区别?为什么说 top-p 更自适应?
3. 为什么标准顺序是先 temperature 再 top-p?顺序颠倒会怎样?
4. greedy 与 beam search 的缺点?beam size 太大/太小的影响?
5. 采样与确定性解码分别适合什么任务?如何权衡?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/decoding-strategies?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=decoding-strategies)

---

## 10. Dense vs MoE · Dense vs MoE 对比

> Dense models activate all parameters per token; MoE activates only a subset of expert parameters per token (sparse activation).

**Interview questions**

1. Capability difference between MoE and Dense at the same active-param count, and why?
2. Why is serving MoE more expensive?
3. What is MoE's all-to-all communication cost?
4. When should you pick Dense over MoE?
5. How is MoE compressed via quantization/distillation?

<details>
<summary>中文题目</summary>

1. 相同激活参数量下 MoE 与 Dense 的能力差异及原因?
2. 为什么 MoE 的服务成本更高?
3. MoE 的 all-to-all 通信开销指什么?
4. 什么场景该选 Dense 而非 MoE?
5. MoE 如何与量化/蒸馏结合做压缩?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/dense-vs-moe?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dense-vs-moe)

---

## 11. Direct Preference Optimization · DPO 直接偏好优化

> DPO (Direct Preference Optimization) is an alignment method that removes the need for a reward model and RL. Key insight: RLHF's reward model implicitly defines an optimal policy;

**Interview questions**

1. What's the core difference between DPO and RLHF? Why does DPO need no reward model or RL?
2. Walk through the DPO loss derivation: how does the partition function Z(x) cancel out?
3. What does the β hyperparameter mean in DPO? Effects of large vs small β?
4. How do DPO and GRPO differ, and when would you use each?
5. DPO limitations: why is it sensitive to preference-data quality, and what role does the reference model play?

<details>
<summary>中文题目</summary>

1. DPO 和 RLHF 的核心区别是什么?DPO 为什么不需要奖励模型和强化学习?
2. DPO 的损失函数推导:配分函数 Z(x) 是如何被消掉的?
3. DPO 的 β 超参数是什么含义?β 大和小分别有什么影响?
4. DPO 和 GRPO 的区别是什么?分别适用什么场景?
5. DPO 的局限:为什么有人说 DPO 对偏好数据质量敏感?参考模型在其中起什么作用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/dpo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dpo)

---

## 12. FFN & SwiGLU · 前馈层与激活

> the Transformer feed-forward layer $FFN(x) = W_2 \cdot GELU(W_1 x)$ holds roughly 2/3 of model parameters (excluding embeddings);

**Interview questions**

1. Why does FFN hold ~2/3 of parameters? Design of intermediate width $4d$ vs $\frac{8}{3}d$?
2. Essential difference between SwiGLU and a plain GELU MLP (input-dependent gating)?
3. Properties of SiLU/Swish: why smoother and saturation-free vs ReLU?
4. GLU family: activations used by ReGLU, GEGLU and SwiGLU?
5. Why is FFN activation memory large in training, and how to mitigate (caching, recomputation)?

<details>
<summary>中文题目</summary>

1. FFN 为什么占参数约 2/3?中间维度 $4d$ 与 $\frac{8}{3}d$ 的设计?
2. SwiGLU 与普通 GELU MLP 的本质区别?(输入依赖的门控机制)
3. SiLU/Swish 的性质:为什么比 ReLU 更平滑、无饱和?
4. GLU 家族对比:ReGLU、GEGLU、SwiGLU 分别用什么激活?
5. 训练时 FFN 激活显存为什么大?如何缓解?(激活缓存、重算)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/feedforward-swiglu?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=feedforward-swiglu)

---

## 13. FlashAttention · FlashAttention

> FlashAttention is an IO-aware exact attention implementation: it tiles $Q K^T$ into SRAM, uses online softmax with running max $m_i$ and normalization factor $l_i$ (recursively rescaled when merging blocks: $m_{new} = max(m, m_{local})$), never writes the…

**Interview questions**

1. Why is attention memory-bound? The bandwidth/capacity gap between HBM and SRAM?
2. How does online softmax merge block statistics (recursive formulas for running max m and normalizer l)?
3. Why recompute in the backward pass instead of storing? Memory from $O(n^2)$ to what?
4. What did FlashAttention-2 improve over v1 (dropping intermediate matrices, fewer non-matmul ops, ~2x)?
5. Why is FlashAttention exact? Essential difference from sparse/low-rank attention?

<details>
<summary>中文题目</summary>

1. 为什么说 attention 是 memory-bound?HBM 与 SRAM 的带宽/容量差距?
2. online softmax 如何合并分块统计量?(运行最大值 m 与归一化因子 l 的递归公式)
3. 反向传播为什么重算而不存储?显存从 $O(n^2)$ 降到多少?
4. FlashAttention-2 相比 v1 改进了什么?(省略中间矩阵、减少非矩阵运算、约 2×)?
5. FlashAttention 为什么是精确的?与稀疏/低秩注意力有什么本质区别?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/flashattention?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=flashattention)

---

## 14. Group Relative Policy Optimization · GRPO 相对策略优化

> GRPO (Group Relative Policy Optimization) is the RL alignment method used by DeepSeek-R1. Core idea: sample $G$ responses per prompt and compute advantage from group-relative scores — subtract the group mean, divide by the group std — eliminating the value/cri…

**Interview questions**

1. What's GRPO's key improvement over PPO, and why can it drop the critic?
2. How is GRPO's advantage computed? The group-relative normalization formula?
3. How does R1 combine GRPO with verifiable rewards (RLVR) for reasoning?
4. How does group size G affect training, and how is the KL penalty designed?
5. When would you choose GRPO vs DPO?

<details>
<summary>中文题目</summary>

1. GRPO 相比 PPO 的核心改进是什么?为什么能去掉 Critic 网络?
2. GRPO 的 Advantage 如何计算?组内相对化公式是什么?
3. GRPO 在 DeepSeek-R1 中如何使用可验证奖励 (RLVR) 做推理强化?
4. GRPO 组大小 G 对训练的影响?KL 惩罚项如何设计?
5. GRPO 和 DPO 的适用场景有什么区别?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/grpo?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=grpo)

---

## 15. Hallucination Mitigation · 幻觉缓解

> hallucination mitigation is a toolbox of techniques for cutting fabricated content, on three main lines: (1) RAG — grounding generation on retrieved documents, the conditional becomes $P(y \mid x, z)$ with $z$ the retrieved evidence, plus requiring citations;

**Interview questions**

1. Why does RAG reduce hallucination, and how do you defend against irrelevant retrieval making it worse?
2. Implementing honest refusal: confidence thresholds, answerability classifiers, and abstain training data — how do you design each?
3. Why does self-consistency reduce reasoning hallucination, and what is the majority-vote formula?
4. What post-hoc verification methods exist, and how do you automatically check factual errors?
5. Evaluating mitigation: how do you measure the refusal-vs-helpfulness trade-off and tune false refusals?

<details>
<summary>中文题目</summary>

1. RAG 为什么能缓解幻觉?如果检索返回的是无关文档,幻觉反而会加重,如何防御?
2. 诚实拒绝的工程实现:置信度阈值、可应答性分类器、"不知道"训练数据分别怎么设计?
3. 自一致性 (self-consistency) 为什么能降低推理型幻觉?多数投票公式是什么?
4. 生成后校验 (post-hoc verification) 有哪些手段?如何自动检查事实错误?
5. 幻觉缓解的评估:如何度量"拒绝率 vs 有用性"的权衡?误拒绝怎么调?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/hallucination-mitigation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hallucination-mitigation)

---

## 16. Hallucination Types & Causes · 幻觉类型与成因

> Hallucination is when a model produces content that contradicts facts, its context, or logic. Three types: factual (invented facts — wrong dates, people, numbers), logical (broken reasoning chains — contradictions, wrong steps), and contextual (conflicts with…

**Interview questions**

1. Name the three hallucination types and give one concrete example of each.
2. Explain hallucination from the training objective: why doesn't the MLE objective max Σ log P(y_t|y_<t,x) guarantee factual correctness?
3. Which type does cutoff-induced hallucination belong to, and how do you mitigate it via interface design?
4. How do sampling parameters affect hallucination rate? Does greedy decoding fully eliminate it?
5. Three hallucination evaluation lenses — fact-based, context-based, logic-based: what tests does each map to?

<details>
<summary>中文题目</summary>

1. 幻觉分哪三类?分别举一个具体例子。
2. 从训练目标角度解释幻觉根源:为什么 MLE 目标 max Σ log P(y_t|y_<t,x) 不保证事实正确?
3. 知识截止 (knowledge cutoff) 导致的幻觉属于哪一类?如何用接口设计缓解?
4. 解码参数 (temperature/采样) 如何影响幻觉率?贪婪解码是否完全消除幻觉?
5. 评估幻觉的三个视角:基于事实、基于上下文、基于逻辑,分别对应什么测试?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/hallucination-types?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=hallucination-types)

---

## 17. Knowledge Distillation · 知识蒸馏

> Knowledge distillation (KD) has a student model mimic the teacher's soft output distribution $p_T = \text{softmax}(z/T)$, where temperature $T$ softens probabilities to expose inter-class dark knowledge;

**Interview questions**

1. Why learn from soft targets instead of hard labels?
2. Role of temperature T; effects of T too large or too small?
3. How do distillation and quantization relate — why are they orthogonal?
4. Common KD variants (logits, features, self-distillation)?
5. What happens when the student-teacher capacity gap is large, and how to mitigate?

<details>
<summary>中文题目</summary>

1. 蒸馏为什么学软标签而不是硬标签?
2. 温度 T 的作用?T 过大/过小有什么影响?
3. 蒸馏与量化是什么关系?为什么说两者正交?
4. 蒸馏的常见变体有哪些 (logits/特征/自蒸馏)?
5. 学生与教师容量差距过大时会出现什么问题?如何缓解?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/distillation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=distillation)

---

## 18. KV Cache Management · KV Cache 管理

> at autoregressive decode time every new token must attend to all history, so past K/V are cached to avoid recomputation: cache size $= 2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$ (2 bytes per fp16 element × K and V × layers × heads × head dim × seq len…

**Interview questions**

1. Meaning of each term in $2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$? A concrete worked example?
2. Why does the KV cache exist only at inference, not training (parallel full-sequence training vs sequential generation)?
3. How PagedAttention works? How do block size and page tables reduce fragmentation?
4. What are StreamingLLM's attention sinks, and why are the first tokens special?
5. How much do GQA, MLA and INT4 quantization each shrink the KV cache (687 → 86GB etc.)?

<details>
<summary>中文题目</summary>

1. KV cache 显存公式 $2 \cdot 2 \cdot L \cdot H \cdot d \cdot s \cdot b$ 每项含义?给一组具体数字?
2. 为什么 KV cache 只在推理出现、训练不需要?(训练并行算整序列,推理串行生成)
3. PagedAttention 的原理?块大小、页表如何减少碎片?
4. StreamingLLM 的 attention sink 是什么?为什么前几个 token 特殊?
5. GQA、MLA、INT4 量化分别把 KV cache 降到多少?(687 → 86GB 等)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/kv-cache-management?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kv-cache-management)

---

## 19. Long-Context Eval Needle/Lost-in-Middle · 长上下文评测 Needle/Lost-in-Middle

> two evaluation methods for long-context ability. (1) Needle-in-a-Haystack: insert one memorable fact (the needle) at a random position inside a long irrelevant text (the haystack), ask for it, and measure precise recall…

**Interview questions**

1. How do you design a needle-in-a-haystack test? What are the heatmap axes and what does an ideal model look like?
2. What is the U-shaped Lost-in-the-Middle dip, and why is middle content most often ignored?
3. Training-side mitigations for Lost-in-the-Middle: positional encoding, document order, attention bias — list them.
4. Inference-side organization: how to order retrieved chunks, where to put key info, and what section markers do.
5. Does a perfect needle score mean long-context is fully usable? What are its blind spots?

<details>
<summary>中文题目</summary>

1. Needle-in-a-Haystack 实验怎么设计?结果热力图横轴纵轴分别是什么,理想模型是什么形状?
2. Lost-in-the-Middle 的 U 形凹陷是什么?为什么中间信息最容易被忽略?
3. 缓解 Lost-in-the-Middle 的训练侧方法有哪些(位置编码、文档顺序、注意力偏置)?
4. 推理侧如何组织长上下文:检索结果排序、关键信息放哪里、分段标记的作用?
5. Needle 测试满分是否代表长上下文能力完全可用?它的盲区是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/needle-lost-middle?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=needle-lost-middle)

---

## 20. Long-Context Extension · 长上下文扩展

> Long-context extension scales the effective context from 4K/8K to 128K/1M, via two main routes. Route one: positional-encoding surgery on RoPE — PI (Positional Interpolation) scales position m by $1/s$ for linear interpolation;

**Interview questions**

1. Why does naive RoPE extrapolation fail?
2. PI vs NTK-aware scaling: what differs fundamentally?
3. YaRN's mechanism and the role of the temperature t?
4. The two routes of context extension and their trade-offs?
5. Compute/memory cost of long context and KV-management techniques?

<details>
<summary>中文题目</summary>

1. RoPE 直接外推为什么会失效?
2. PI 与 NTK 感知缩放的本质区别?
3. YaRN 的原理与温度系数 t 的作用?
4. 上下文扩展的两条路线与各自取舍?
5. 长上下文的计算/显存代价与 KV 管理手段?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/context-extension?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=context-extension)

---

## 21. LoRA / QLoRA Low-Rank Adaptation · LoRA/QLoRA 低秩适配

> LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method: freeze the pretrained weights $W$ and train only a low-rank delta $\Delta W = BA$ with $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, rank $r \ll d$.

**Interview questions**

1. Why does LoRA work? What justifies the low-rank assumption?
2. How do you pick the rank r? Effects of r too large or small?
3. How are LoRA weights merged at inference? Deploying multiple LoRAs?
4. QLoRA's mechanism: 4-bit quant, paged optimizer, double quantization — what does each solve?
5. LoRA vs full fine-tuning: quality gaps and when full FT is required?

<details>
<summary>中文题目</summary>

1. LoRA 为什么有效?低秩假设的依据是什么?
2. LoRA 的秩 r 如何选择?r 太大/太小的影响?
3. LoRA 在推理时如何合并权重?多个 LoRA 如何部署?
4. QLoRA 的原理:4-bit 量化 + 分页优化器 + 双重量化分别解决什么?
5. LoRA 和全量微调 (Full FT) 的效果差异?什么时候必须全量?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/lora?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=lora)

---

## 22. Low-Bit INT8/INT4 · 低比特 INT8/INT4

> Low-bit quantization drops weights/activations to 8-bit or 4-bit integers. Storage goes from 2 bytes (FP16) to 1 byte (INT8) to 0.5 byte (INT4) per weight — INT4 weights cut weight memory by 75% (FP16 70B needs 140GB, INT4 only ~35GB).

**Interview questions**

1. What are the memory/bandwidth savings of INT8/INT4 vs FP16?
2. Why does INT4 save 75% memory — walk through the math?
3. What is group-wise quantization and why does it help at low bits?
4. How do you choose the calibration set, and why is it accuracy-critical?
5. What are common accuracy-loss sources at low bits and how to mitigate them?

<details>
<summary>中文题目</summary>

1. INT8/INT4 相比 FP16 的显存与带宽收益各是多少?
2. 为什么说 INT4 省 75% 显存?计算过程是什么?
3. 分组量化 (group-wise) 是什么?为什么低比特下更有效?
4. 校准集如何选择?为什么它是精度关键?
5. 低比特下常见的精度损失来源有哪些?如何缓解?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/low-bit?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=low-bit)

---

## 23. MCTS Reasoning Search · MCTS 推理搜索

> MCTS (Monte Carlo Tree Search) organizes generation as tree search: nodes are reasoning states (partial reasoning produced so far), edges are next actions.

**Interview questions**

1. What do MCTS's four steps (selection/expansion/simulation/backpropagation) each do?
2. Write the UCT formula, explain each symbol, and how it balances exploration vs exploitation?
3. How is MCTS combined with an LLM? How are nodes, actions, and values defined?
4. The role of PRMs in MCTS? Why are step-level rewards better than outcome rewards?
5. MCTS vs plain sampling (best-of-N): compute complexity vs gains?

<details>
<summary>中文题目</summary>

1. MCTS 四步(选择/扩展/模拟/回溯)分别做什么?
2. 写出 UCT 公式并解释各符号?如何平衡探索与利用?
3. MCTS 如何与 LLM 结合?节点、动作、价值分别如何定义?
4. 过程奖励模型 (PRM) 在 MCTS 中的作用?为什么逐步奖励优于结果奖励?
5. MCTS vs 简单多次采样 (best-of-N):计算复杂度与收益的对比?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/mcts-search?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mcts-search)

---

## 24. Model Families & Evolution · 模型家族与演进

> Mainstream open-source families differ in architecture details and training recipes: LLaMA — BPE tokenizer with a 32K vocab (128K in LLaMA-3), SwiGLU + RMSNorm + RoPE, GQA from 70B up, trained on 15.6T tokens;

**Interview questions**

1. LLaMA-3 vs Qwen: tokenizer and positional-encoding differences?
2. What is DeepSeek-V3's MoE + MLA configuration?
3. How does vocab size affect Chinese token efficiency and inference cost?
4. What does the token-scaling trend (1.4T → 15.6T) imply?
5. How does each family support long context (YARN / positional surgery)?

<details>
<summary>中文题目</summary>

1. LLaMA-3 与 Qwen 在分词、位置编码上的差异?
2. DeepSeek-V3 的 MoE + MLA 配置是什么?
3. 词表大小如何影响中文 token 效率与推理成本?
4. 训练 token 规模从 1.4T 涨到 15.6T 说明什么趋势?
5. 各家族如何支持长上下文(YARN / 位置编码改造)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/model-families?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=model-families)

---

## 25. MoE Load Balancing · 负载均衡损失

> without constraints, the gate makes tokens pile onto a few experts, causing load imbalance and routing collapse (a few experts absorb nearly all signal).

**Interview questions**

1. What are f_i, P_i, α, and N in the load-balancing loss?
2. Why does imbalance occur — what is expert/routing collapse?
3. What is the capacity factor — effects of it being too large or small?
4. How are overflow tokens handled?
5. Trade-offs of the aux-loss weight α being too large or small?

<details>
<summary>中文题目</summary>

1. 负载均衡损失公式中 f_i、P_i、α、N 各是什么?
2. 为什么会出现负载不均?专家崩溃 (routing collapse) 是什么?
3. 容量因子 (capacity factor) 是什么?过大/过小的影响?
4. 超限 token 如何处理?
5. 均衡损失权重 α 太大/太小的权衡?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/moe-load-balance?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=moe-load-balance)

---

## 26. MoE Routing Top-k · MoE 路由 Top-k

> MoE (Mixture-of-Experts) places $N$ experts in FFN layers; a gating network $g(x) = \text{softmax}(W_g x)$ scores every token and selects the top-$k$ experts, outputting the weighted sum $y = \sum_{i \in \text{top-}k} g_i(x) E_i(x)$;

**Interview questions**

1. How does MoE routing work — what does top-k selection mean?
2. Why 671B total but 37B active — what do active params mean?
3. What is the gating network's input and how is it trained?
4. Trade-offs between top-1 and top-2 routing?
5. Why does MoE cut both training and inference cost?

<details>
<summary>中文题目</summary>

1. MoE 路由如何工作?top-k 选择意味着什么?
2. 为什么 671B 总参只有 37B 激活?激活参数的含义?
3. 门控网络的输入与训练方式是什么?
4. top-1 与 top-2 路由的权衡?
5. MoE 为什么能同时降低训练和推理成本?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/moe-routing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=moe-routing)

---

## 27. Multi-head Latent Attention · MLA 低秩 KV 压缩

> MLA (Multi-head Latent Attention) is the low-rank KV compression introduced by DeepSeek-V2: the layer's K/V are first compressed into a low-dimensional latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c = 512$ in DeepSeek-V2), and per-head K/V are reconstruct…

**Interview questions**

1. How does MLA compress KV? Dimensionality of $c_t^{KV}$ and the reconstruction process?
2. Why does MLA need decoupled RoPE? What role does $k_t^R$ play?
3. How much memory does MLA save vs GQA? Concrete numbers (41.5GB → 5.1GB)?
4. How does MLA cut activation memory in training (recomputing K/V at prefill)?
5. Essential difference between MLA and MQA/GQA (sharing vs low-rank projection)?

<details>
<summary>中文题目</summary>

1. MLA 如何压缩 KV?$c_t^{KV}$ 的维度与还原过程?
2. MLA 为什么需要解耦 RoPE?$k_t^R$ 的作用是什么?
3. MLA 相比 GQA 显存省多少?给出具体数字(41.5GB → 5.1GB 的对比)
4. 训练时 MLA 如何减少激活显存?(Prefill 阶段重算 K/V)
5. MLA 与 MQA/GQA 的本质区别?(共享 vs 低秩投影)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/mla-attention?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mla-attention)

---

## 28. Padding & Packing · 填充与打包 Padding/Packing

> Padding pads sequences of unequal length in a batch to the same length for GPU matrix alignment — usually dynamically to the batch's longest sequence (or a multiple of 8/16);

**Interview questions**

1. Dynamic vs fixed-length padding? Why pad to multiples of 8/16?
2. How does sequence packing remove pad waste? Masking sample seams?
3. Problems when pads enter attention? Role of the attention mask?
4. Loss computation under packing? Why a loss mask is needed?
5. How padding/packing relate to FlashAttention and PagedAttention?

<details>
<summary>中文题目</summary>

1. 动态 padding 与固定长度 padding 各节省什么?为什么 pad 到 8/16 的倍数?
2. Sequence packing 如何消除 pad token 浪费?跨样本边界如何 mask?
3. pad token 参与注意力计算会带来什么问题?attention mask 的作用?
4. Packing 时 loss 计算要注意什么?为什么需要 loss mask?
5. Padding/Packing 与 FlashAttention、PagedAttention 的关系?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/padding-packing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=padding-packing)

---

## 29. Positional Encoding RoPE/ALiBi · 位置编码 RoPE/ALiBi

> RoPE (Rotary Position Embedding) rotates the $2j$-th and $2j+1$-th dims of Q/K by angle $m \theta_j$ with $\theta_j = 10000^{-2j/d}$, so the inner product of any positions $m$, $n$ depends only on their relative distance $m - n$;

**Interview questions**

1. RoPE rotation formula? Why does the inner product depend only on the relative offset $m - n$?
2. RoPE wavelength: period of the lowest-frequency dim? When does base=10000 cause periodic ambiguity (~62K tokens; LLaMA-3 uses 500000)?
3. RoPE vs absolute and learnable position embeddings?
4. How ALiBi works, and its pros and cons?
5. Why can't RoPE extrapolate directly to longer sequences (leading to PI/NTK/YaRN)?

<details>
<summary>中文题目</summary>

1. RoPE 的旋转公式?为什么内积只依赖相对位置 $m - n$?
2. RoPE 的波长公式:最低频维度周期约多少?base=10000 的周期混淆出现在多长序列?(约 6 万 token,LLaMA-3 改用 500000)
3. RoPE 与绝对位置编码、可学习位置编码的对比?
4. ALiBi 的原理与优缺点?
5. RoPE 为什么不能直接外推到长序列?(引出 PI/NTK/YaRN)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/positional-encoding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=positional-encoding)

---

## 30. PPO Clipped Objective · PPO 截断损失

> PPO (Proximal Policy Optimization) constrains policy updates via a clipped probability ratio. With the ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$, the clipped objective is…

**Interview questions**

1. Write the PPO clipped objective and explain the clip's role. Why ε = 0.2?
2. How does clipping behave for positive vs negative advantages? What happens at r=2?
3. What does the ratio r_t(θ) mean, and what role does importance sampling play?
4. How does PPO relate to TRPO, and why is PPO preferred in practice?
5. How is the reward constructed in RLHF-PPO? Roles of GAE and the KL penalty?

<details>
<summary>中文题目</summary>

1. 写出 PPO 截断目标函数并解释 clip 的作用?为什么 ε 取 0.2?
2. Advantage 为正和为负时,截断分别如何生效?r=2 时会发生什么?
3. 概率比 r_t(θ) 的含义?重要性采样在其中扮演什么角色?
4. PPO 与 TRPO 的关系?为什么实践中 PPO 更常用?
5. RLHF 中 PPO 的奖励如何构造?GAE 和 KL 惩罚的作用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/ppo-clip?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ppo-clip)

---

## 31. Pre-LN & RMSNorm · 归一化 Pre-LN/RMSNorm

> LayerNorm standardizes the last dimension: $LN(x) = \frac{x - mean}{\sqrt{var}} \cdot \gamma + \beta$; Pre-LN places normalization before the residual sublayer ($x + SubLayer(LN(x))$) while Post-LN applies it after the residual addition;

**Interview questions**

1. Structural difference between Pre-LN and Post-LN? Why is Pre-LN more stable for deep models?
2. RMSNorm formula? Difference from LayerNorm, and why dropping the mean works?
3. Downsides of Pre-LN (residual drift; late-training accuracy vs Post-LN, warmup switching)?
4. Why does the position of normalization relative to the residual matter? Why must the $\gamma$ scale be kept?
5. What problem does DeepNorm solve (stable training of 1000-layer Post-LN models)?

<details>
<summary>中文题目</summary>

1. Pre-LN 与 Post-LN 的结构区别?为什么 Pre-LN 深层训练更稳定?
2. RMSNorm 的公式?与 LayerNorm 的差异及为什么去掉均值可行?
3. Pre-LN 的缺点?(残差信号漂移,后期精度可能不如 Post-LN,可预热后切换)
4. 归一化在残差内外的位置为什么重要?$\gamma$ 缩放参数为什么必须保留?
5. DeepNorm 解决什么问题?(Post-LN 千层模型的稳定训练)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/preln-rmsnorm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=preln-rmsnorm)

---

## 32. Pre-Training Data Engineering · 预训练数据工程

> Pre-training data engineering is the pipeline that sets the model's capability ceiling: cleaning → dedup → quality filtering → domain mix → annealing.

**Interview questions**

1. How does MinHash+LSH work? Why is approximate dedup sufficient?
2. How does PPL filtering work? Is low PPL always high quality?
3. How to design the domain mix (Web:Code:Math)? Which capabilities does it shape?
4. What is the annealing phase? Why inject high-quality/synthetic data?
5. Synthetic data (Evol-Instruct etc.) and the contamination risk?

<details>
<summary>中文题目</summary>

1. MinHash 与 LSH 的原理? 为什么近似去重够用?
2. PPL 质量过滤怎么做? 低 PPL 一定代表高质量吗?
3. 域配比(Web:Code:Math)如何设计? 影响哪些能力?
4. 退火期是什么? 为什么要注入高质量/合成数据?
5. 合成数据(Evol-Instruct 等)与数据污染的风险?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/pretrain-data-engineering?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pretrain-data-engineering)

---

## 33. Pre-Training Objectives & Optimizer Details · 训练目标与优化器细节

> The mainstream pretraining objective is Next-Token Prediction (autoregressive cross-entropy $L = -\sum \log P(x_t | x_{<t})$);

**Interview questions**

1. Next-Token vs FIM objectives? How are FIM samples constructed?
2. Why β₂=0.95 and wd=0.1 in AdamW? Difference from standard Adam?
3. Role & implementation of gradient clipping (global vs per-parameter norm)?
4. Causes of loss spikes and how the rollback strategy works?
5. How to design the LR schedule (warmup + cosine decay)?

<details>
<summary>中文题目</summary>

1. Next-Token 与 FIM 目标的区别? FIM 样本如何构造?
2. AdamW 为什么 β₂=0.95、wd=0.1? 与标准 Adam 的差异?
3. 梯度裁剪的作用与实现(global norm vs per-parameter)?
4. Loss Spike 的成因与回退策略如何执行?
5. 学习率调度(warmup + cosine)如何设计?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/pretrain-training-details?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=pretrain-training-details)

---

## 34. PRM & RLVR · PRM 与 RLVR

> PRM (Process Reward Model) scores each intermediate reasoning step, refining credit assignment; RLVR (RL from Verifiable Rewards) replaces human/AI reward models with rule/program-verifiable rewards — e.g.

**Interview questions**

1. PRM vs ORM (outcome reward model): differences, PRM's advantages, and its main challenges?
2. Why does RLVR eliminate reward hacking? Examples of verifiable rewards?
3. How does DeepSeek-R1 combine RLVR and GRPO for reasoning training?
4. How are PRMs used at test time (MCTS, best-of-N reranking)?
5. Limitations of PRM/RLVR? What verifiability do the tasks require?

<details>
<summary>中文题目</summary>

1. PRM 与 ORM (结果奖励模型) 的区别?PRM 的优点与主要挑战?
2. RLVR 为什么能消除奖励黑客?举几个可验证奖励的例子?
3. DeepSeek-R1 如何结合 RLVR 与 GRPO 训练推理能力?
4. PRM 在推理时搜索 (MCTS / Best-of-N 重排) 中如何使用?
5. PRM/RLVR 的局限?对任务的验证性有什么要求?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/prm-rlvr?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=prm-rlvr)

---

## 35. Quantization PTQ/QAT · 量化 PTQ/QAT

> Quantization maps FP16 weights $W$ to low-bit integers $x_q = \text{round}(x/s) + z$ with scale $s = (x_{\max}-x_{\min})/(2^b-1)$ and zero-point $z$ aligned to the b-bit range.

**Interview questions**

1. Differences between PTQ and QAT, and when to use each?
2. Role of the calibration set in PTQ; how its size/distribution affect accuracy?
3. What is the straight-through estimator (STE) in QAT and why is it needed?
4. Why does PTQ dominate for LLMs, and when is QAT required?
5. How is quantization error measured, and what are common error sources?

<details>
<summary>中文题目</summary>

1. PTQ 与 QAT 的区别?各自的适用场景?
2. PTQ 中校准集的作用?校准集规模/分布如何影响精度?
3. QAT 中直通估计器 (STE) 的原理?为什么需要它?
4. 为什么 LLM 领域以 PTQ 为主?什么情况必须用 QAT?
5. 量化误差如何衡量?常见的误差来源有哪些?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/ptq-qat?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ptq-qat)

---

## 36. Reference-Free Alignment SimPO/ORPO/KTO · 无参考对齐 SimPO/ORPO/KTO

> reference-free alignment methods drop the reference policy and/or the separate SFT stage: ① SimPO (Simple Preference Optimization) removes the reference model entirely, using the sequence-level average log-probability…

**Interview questions**

1. How does SimPO remove the reference model? Its implicit reward and advantages?
2. ORPO's loss composition? Why does it do SFT + alignment in one step?
3. KTO vs DPO: core differences, and what behavioral-economics theory underlies KTO?
4. Compare SimPO/ORPO/KTO with DPO: memory, data requirements, training stability?
5. Common risks of reference-free methods? Issues from losing the KL anchor?

<details>
<summary>中文题目</summary>

1. SimPO 如何去掉参考模型?其隐式奖励的定义和优点?
2. ORPO 的损失构成?为什么能一步完成 SFT 与对齐?
3. KTO 与 DPO 的核心差异?KTO 损失基于什么行为经济学理论?
4. SimPO/ORPO/KTO 与 DPO 在显存、数据要求、训练稳定性上的对比?
5. 无参考对齐的共同风险?失去 KL 锚点可能带来什么问题?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/simpo-orpo-kto?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=simpo-orpo-kto)

---

## 37. RLHF 3-Stage Pipeline · RLHF 三阶段

> RLHF (Reinforcement Learning from Human Feedback) is the 3-stage pipeline for aligning LLMs to human preferences: ① SFT instruction tuning;

**Interview questions**

1. Walk through the 3 stages of RLHF: inputs, outputs, and objectives of each stage?
2. How is the reward model trained? The Bradley-Terry loss and training data format?
3. Why is the KL penalty βlog(πθ/πref) needed in the PPO stage? What happens without it?
4. Why does the RM model preference rather than absolute scores? What is reward hacking?
5. Compare RLHF vs DPO: pipeline, cost, and stability differences?

<details>
<summary>中文题目</summary>

1. RLHF 三阶段的完整流程是什么?每一阶段的输入、输出与训练目标?
2. 奖励模型如何训练?Bradley-Terry 模型的损失函数与训练数据格式?
3. PPO 阶段为什么需要 KL 惩罚项 βlog(πθ/πref)?不惩罚会怎样?
4. 奖励模型为什么建模偏好而非绝对分数?奖励黑客 (Reward Hacking) 是什么?
5. RLHF 与 DPO 的对比:流程、成本、稳定性各有什么差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/rlhf-pipeline?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rlhf-pipeline)

---

## 38. RoPE Extrapolation PI/NTK/YaRN · RoPE 外推方案 PI/NTK/YaRN

> three ways to extend RoPE from the training length to longer contexts at low cost: PI (Position Interpolation) linearly rescales position $m$ to $m / s$ ($s$ = scaling factor);

**Interview questions**

1. Why does direct RoPE extrapolation fail (periodic ambiguity, frequencies beyond training coverage)?
2. How PI works and its main flaw (uniform scaling loses high-frequency resolution)?
3. Difference between NTK-aware and NTK-by-parts? The wavelength criterion?
4. What are YaRN's two components? Role of temperature $t \approx \sqrt{s}$?
5. How to evaluate extrapolation methods (long-doc perplexity + needle precision recall)?

<details>
<summary>中文题目</summary>

1. 为什么 RoPE 直接外推会失效?(周期混淆、频率超出训练覆盖)
2. PI 的原理与主要缺陷?(均匀缩放损失高频分辨率)
3. NTK-aware 与 NTK-by-parts 的区别?波长判据是什么?
4. YaRN 由哪两部分组成?温度 $t \approx \sqrt{s}$ 的作用?
5. 如何评估外推方案的效果?(长文档困惑度 + Needle 精确召回)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/rope-extrapolation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rope-extrapolation)

---

## 39. Scaled Dot-Product Attention · 缩放点积注意力

> Scaled Dot-Product Attention is the core attention operator of the Transformer: $Attention(Q,K,V) = softmax(\frac{Q K^T}{\sqrt{d_k}}) V$.

**Interview questions**

1. Why divide by $\sqrt{d_k}$? What breaks without it (variance analysis, softmax saturation)?
2. Time and space complexity of scaled dot-product attention? Why is $O(n^2)$ the long-context bottleneck?
3. Shapes of the Q/K/V projections? How are heads split and concatenated?
4. Difference and implementation of padding mask vs causal mask?
5. Single vs multi-head: why h heads rather than one wide $d$-dim head?

<details>
<summary>中文题目</summary>

1. 为什么注意力要除以 $\sqrt{d_k}$?不除以会怎样(方差分析与 softmax 饱和)?
2. 缩放点积注意力的时间与空间复杂度是多少?为什么 $O(n^2)$ 是长上下文瓶颈?
3. Q、K、V 的投影矩阵形状是什么?多头如何拆分与拼接?
4. padding mask 与 causal mask 的区别与实现方式?
5. 单头 vs 多头:为什么拆成 h 个头而不是一个 $d$ 维大头?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/attention-scaled-dot?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=attention-scaled-dot)

---

## 40. Scaling Laws · 缩放定律

> Scaling laws state that the cross-entropy loss follows power laws in parameters N, data D and compute C: $L(N) \propto N^{-\alpha}$, $L(D) \propto D^{-\beta}$, $L(C) \propto C^{-\gamma}$, with α≈0.076 and β≈0.095.

**Interview questions**

1. Power-law forms? Kaplan vs Chinchilla differences?
2. Why does data matter more than parameters at fixed compute?
3. Chinchilla's content and how the 20:1 data-to-param ratio was derived?
4. Do scaling laws extrapolate forever? Saturation and emergence?
5. Using scaling laws to choose N/D and estimate training cost?

<details>
<summary>中文题目</summary>

1. 幂律公式的形式? Kaplan 与 Chinchilla 两种观点的差异?
2. 为什么同等算力下数据比参数更重要?
3. Chinchilla 法则的内容与 20:1 数据参数比如何得出?
4. 缩放定律能否无限外推? 饱和与涌现现象?
5. 如何用缩放定律选择 N/D 并估算训练成本?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/scaling-laws?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=scaling-laws)

---

## 41. SmoothQuant/AWQ & FP8 · 激活异常值处理与 FP8

> Three advanced quantization schemes: SmoothQuant migrates activation outliers into weights via a mathematically equivalent transform — rewriting as $Y = (X \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} W)$ keeps the output unchanged but smooths the activati…

**Interview questions**

1. Why does SmoothQuant work — how are outliers migrated to weights?
2. How does AWQ find sensitive channels, and why protect the top 1%?
3. E4M3 vs E5M2 — differences and where each is used?
4. What are W8A8 and W4A16, and what do they buy?
5. Why are activations harder to quantize than weights?

<details>
<summary>中文题目</summary>

1. SmoothQuant 为什么有效?异常值如何迁到权重上?
2. AWQ 如何识别敏感通道?为什么保护前 1% 通道?
3. FP8 的 E4M3 与 E5M2 有何区别?各自用于何处?
4. W8A8 与 W4A16 分别是什么?收益如何?
5. 为什么激活比权重更难量化?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/smoothquant-awq-fp8?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=smoothquant-awq-fp8)

---

## 42. Soft Prompts / Adapters · 软提示 P-Tuning/Adapter

> Soft prompts and adapters are PEFT routes that freeze the backbone and train only a few extra parameters: 1) Prefix-Tuning prepends learnable key/value vectors before attention in every layer (~0.1%–2% of params);

**Interview questions**

1. Prefix-Tuning vs P-Tuning: what differs?
2. Why is the Adapter bottleneck effective (residual + nonlinearity)?
3. Soft prompts vs LoRA: comparison and how to choose?
4. Why freeze the backbone and train only extra params? Benefits & risks?
5. Multi-task serving: how do multiple prompts/adapters coexist?

<details>
<summary>中文题目</summary>

1. Prefix-Tuning 与 P-Tuning 的区别?
2. Adapter 的瓶颈结构为什么有效(残差 + 非线性)?
3. Soft Prompt 与 LoRA 的对比与选型?
4. 为什么冻结主干只训附加参数? 收益与风险?
5. 多任务部署: 多组 soft prompt/adapter 如何共存?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/soft-prompts?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=soft-prompts)

---

## 43. Speculative Decoding (cross-module) · 推测解码(跨模块)

> Speculative Decoding is a lossless speedup for autoregressive generation: a lightweight draft model predicts $\gamma$ tokens, the target LLM verifies them in parallel — one forward pass computes logits at $\gamma+1$ positions — accepting the matching prefix an…

**Interview questions**

1. Speedup formula: how do α and γ affect the gain?
2. Why is speculative decoding lossless? How rejection sampling preserves distribution?
3. Choosing the draft model? Trade-off between α and draft cost?
4. EAGLE/Medusa vs standard draft models? Why tree-structured heads win?
5. Which scenarios gain the most — and which gain little?

<details>
<summary>中文题目</summary>

1. 加速比公式 $E \approx \frac{1-\alpha^{\gamma}}{1-\alpha}$:$\alpha$、$\gamma$ 如何影响加速?
2. 为什么推测解码是无损的?接受-拒绝采样如何保证分布不变?
3. 草稿模型如何选择?接受率 α 与草稿开销之间的权衡?
4. EAGLE/Medusa 与标准草稿模型的区别?树状多 Head 为什么更优?
5. 什么场景收益最大、什么场景收益有限?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/speculative-decoding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=speculative-decoding)

---

## 44. SSM & Hybrid Architectures · SSM/线性注意力混合

> SSM (State Space Model) models sequences with a linear recurrence: $h_t = A h_{t-1} + B x_t$, $y_t = C h_t$, with fixed state size independent of sequence length;

**Interview questions**

1. SSM recurrence? How is the state updated, and what does fixed state size imply?
2. What is Mamba's selective mechanism, and why must B, C, $\Delta_t$ depend on the input?
3. Why do hybrids like Jamba keep attention layers? What role does each component play?
4. How does RWKV/linear attention's recurrence correspond to standard attention?
5. Why are SSMs fast to train and cheap at inference (parallel scan + constant state)?

<details>
<summary>中文题目</summary>

1. SSM 的递推公式是什么?状态如何更新?状态大小固定意味着什么?
2. Mamba 的选择性机制是什么?为什么 B、C、$\Delta_t$ 要输入依赖?
3. Jamba 这类混合架构为什么还要保留注意力层?各自承担什么职责?
4. RWKV/线性注意力的递推形式与标准注意力如何对应?
5. SSM 训练快、推理省的原理是什么?(并行扫描 + 常数状态)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/ssm-hybrid?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ssm-hybrid)

---

## 45. Supervised Fine-Tuning (SFT) · 指令微调 SFT

> SFT (Supervised Fine-Tuning) fine-tunes the pretrained model on high-quality instruction–response pairs to turn a text-continuation engine into an instruction follower.

**Interview questions**

1. SFT vs pretraining: data, objective, epochs?
2. Why does quality beat quantity in SFT data?
3. Can SFT overfit? How to choose epochs and LR?
4. What is the relationship between SFT and RLHF/DPO?
5. How are instruction datasets built and filtered (human/LLM-generated)?

<details>
<summary>中文题目</summary>

1. SFT 与预训练的区别(数据、目标、epoch)?
2. 为什么 SFT 数据质量 > 数量?
3. SFT 会过拟合吗? epoch 与学习率怎么选?
4. SFT 与 RLHF/DPO 的关系?
5. 指令数据如何构造与过滤(人工/LLM 生成)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/sft-basics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=sft-basics)

---

## 46. Test-Time Compute Scaling · 推理时扩展

> Test-time compute scaling spends more compute at generation time to buy accuracy, orthogonal to training-time scaling laws.

**Interview questions**

1. Main forms of test-time compute scaling, and how it differs from training-time scaling laws?
2. What is a thinking budget, and how does o1 control it via reasoning effort?
3. Best-of-N vs majority voting: differences and the signals each requires?
4. The diminishing-returns curve: where does it flatten, and which tasks benefit most?
5. How is test-time compute distilled back into small models (self-play/distillation)?

<details>
<summary>中文题目</summary>

1. 推理时扩展有哪些主要形式?与训练时扩展 (scaling laws) 的区别?
2. 思考预算 (thinking budget) 是什么?o1 如何用 reasoning effort 控制?
3. Best-of-N 与 majority voting 的区别?各需要什么信号?
4. 推理时扩展的收益曲线:边际收益递减体现在哪?什么任务收益最大?
5. 如何把推理时扩展蒸馏回小模型(自博弈/蒸馏路线)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/test-time-compute?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=test-time-compute)

---

## 47. Text Embeddings · 文本嵌入 Embeddings

> Text embeddings map tokens or whole sentences into dense, low-dimensional real vectors so semantically similar texts lie close in vector space.

**Interview questions**

1. Token-level vs sentence-level embeddings? When CLS vs mean pooling?
2. Why cosine similarity over dot product or L2 distance? Its range?
3. Embedding's role in RAG: offline indexing, online ANN search, reranking?
4. Estimating params/memory of the vocab embedding table?
5. Static (Word2Vec) vs contextual representations (Transformer outputs)?

<details>
<summary>中文题目</summary>

1. token 级与句子级嵌入的区别?CLS 与 mean pooling 各自适用什么?
2. 为什么用余弦相似度而不是内积或 L2 距离?其取值范围是什么?
3. Embedding 在 RAG 中的完整流程:离线建索引、在线 ANN 检索、重排序?
4. 词表嵌入表 $E \in \mathbb{R}^{V \times d}$ 的参数量与显存如何估算?
5. 静态词向量 (Word2Vec) 与上下文相关表示 (Transformer 层输出) 的差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/embeddings?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=embeddings)

---

## 48. Tokenization: BPE / WordPiece · 分词 BPE/WordPiece

> BPE (Byte Pair Encoding) is a subword tokenization algorithm: start from byte/character-level units, count co-occurrence of adjacent symbol pairs, merge the most frequent byte pair into a new vocabulary symbol each iteration until the target vocab size $V$ is…

**Interview questions**

1. BPE training: starting unit, what each iteration does, stopping condition?
2. Why do LLM vocabs sit at 30K-150K? Trade-offs of larger/smaller vocabs?
3. How does byte-level BPE guarantee no OOV for arbitrary Unicode text?
4. Selection criteria of BPE vs WordPiece vs Unigram?
5. Tokenizer impact: average tokens per English word? Multilingual/code behavior?

<details>
<summary>中文题目</summary>

1. BPE 的训练过程:从什么单位开始,每次迭代做什么,何时停止?
2. 为什么 LLM 词表普遍在 3-15 万?词表过大/过小的权衡?
3. 字节级 BPE 如何保证任意 Unicode 文本没有未登录词 (OOV)?
4. BPE 与 WordPiece、Unigram 的选择准则分别是什么?
5. tokenizer 对模型的影响:平均一个英文单词几个 token?多语言/代码表现?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/LLM/card/bpe-tokenization?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=bpe-tokenization)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
