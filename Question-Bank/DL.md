# 🧠 Deep Learning Foundations · 深度学习基础 — Interview Question Bank

> Part of the [TalentMe Interview Question Bank](../QUESTION-BANK.md) · **38 cards · 190 questions**
> [🃏 Practice all 1,898 flashcards](https://talentme.airsota.com/tools/flashcards?utm_source=github&utm_medium=question-bank&utm_campaign=vault) · [🧭 Open the interactive mindmap](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL?utm_source=github&utm_medium=question-bank&utm_campaign=vault)

English questions are shown inline; expand **中文题目** for the Chinese version. Each card links to the full answer, derivation, and code.

---

## 1. Activation Functions · 激活函数演进

> The evolution of activation functions is the main thread of nonlinearity in deep learning. Sigmoid squashes to $(0,1)$: $\sigma(x)=\frac{1}{1+e^{-x}}$ with derivative $\sigma'(x)=\sigma(x)(1-\sigma(x))=\sigma(1-\sigma)$, peaking at only $0.25$ and saturating a…

**Interview questions**

1. Derive $\sigma'(x)=\sigma(x)(1-\sigma(x))$ on the whiteboard; why do deep nets with Sigmoid inevitably suffer vanishing gradients (peak derivative 0.25)?
2. What does ReLU fix vs Sigmoid? Causes of Dead ReLU and its fixes (LeakyReLU, bias initialization)?
3. Write the GELU formula $\Phi(x)x+(1-\Phi(x))0.5x$ with its properties; why do BERT/GPT use GELU over ReLU?
4. Gating mechanism of SwiGLU $x\sigma(xW_g)\otimes(xW_u)$; why do LLaMA/Mistral FFNs use SwiGLU?
5. Compare derivative ranges and saturation/death issues of Sigmoid/Tanh/ReLU/GELU; why are zero-centered outputs preferred?

<details>
<summary>中文题目</summary>

1. 白板推导 Sigmoid 导数 $\sigma'(x)=\sigma(x)(1-\sigma(x))$,并解释为什么深层网络用 Sigmoid 必然梯度消失(导数峰值仅 0.25)?
2. ReLU 相比 Sigmoid 解决了什么?Dead ReLU 的成因与缓解手段(LeakyReLU、偏置初始化)各是什么?
3. 写出 GELU 公式 $\Phi(x)x+(1-\Phi(x))0.5x$ 并说明性质;为什么 BERT/GPT 用 GELU 而不用 ReLU?
4. SwiGLU 公式 $x\sigma(xW_g)\otimes(xW_u)$ 的门控机制?为什么 LLaMA/Mistral 的 FFN 用 SwiGLU?
5. 对比 Sigmoid/Tanh/ReLU/GELU 的导数范围与饱和/死亡问题;为什么激活输出最好零中心?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/activation-functions?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=activation-functions)

---

## 2. Adam & AdamW · Adam/AdamW

> Adam (adaptive moment estimation) maintains exponential moving averages of the first moment $m_t$ and the second moment (uncentered variance) $v_t$ of the gradient, with bias correction: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$,…

**Interview questions**

1. Write Adam's four formulas (moment recursions, bias correction, parameter update) and quote the defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.
2. Why is bias correction needed? Sum the geometric series to prove $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$, explain why the bias is worst at $t=1$ ($1-\beta_1 = 0.1$), and why dividing by $1-\beta_1^t$ gives $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$.
3. Why update by $\hat{m}_t/\sqrt{\hat{v}_t}$? Relation to RMSProp/AdaGrad; why must $\epsilon$ sit in the denominator (numerical stability + step bound $\eta/\sqrt{\epsilon}$)?
4. Adam vs SGD+momentum: use cases and generalization differences (why SGD often wins on CV while large models standardize on Adam); why Adam converges fast early but can be unstable later.
5. Write the AdamW update and compare with Adam+L2: how does dividing the L2 term by $\sqrt{\hat{v}_t}$ distort the effective regularization? Why does decoupling give a constant per-step decay $1 - \eta\lambda$?

<details>
<summary>中文题目</summary>

1. 默写 Adam 四个公式(矩估计、偏差修正、参数更新),并指出默认 $\beta_1 = 0.9$、$\beta_2 = 0.999$、$\epsilon = 10^{-8}$、$\eta = 10^{-3}$。
2. 为什么需要偏差修正?用等比数列求和证明 $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$,说明 $t=1$ 时偏差最严重($1-\beta_1 = 0.1$),为什么除以 $1-\beta_1^t$ 后 $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$。
3. 为什么更新用 $\hat{m}_t/\sqrt{\hat{v}_t}$?与 RMSProp/AdaGrad 的关系;为什么 $\epsilon$ 必须放在分母(数值稳定 + 步长上界 $\eta/\sqrt{\epsilon}$)。
4. Adam vs SGD+动量: 各自的适用场景与泛化差异(为什么 CV 上 SGD 常更好、而大模型标配 Adam);Adam 早期收敛快但后期可能不稳的原因。
5. 写出 AdamW 更新式并对比 Adam+L2: L2 项被 $\sqrt{\hat{v}_t}$ 缩放如何扭曲有效正则强度?解耦 weight decay 后每步衰减率为何恒为 $1 - \eta\lambda$?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/adam-optimizer?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=adam-optimizer)

---

## 3. Autograd Compute Graph · Autograd 动态图

> During the forward pass the framework builds a computation graph (DAG) on the fly: every tensor that requires grad records the operation that produced it as grad_fn, and the graph is rebuilt per iteration following the current control flow (dynamic graph).

**Interview questions**

1. How does backprop determine the computation order? Why reverse topological order?
2. Dynamic vs static computation graphs: differences, pros and cons?
3. Role of grad_fn? How is the chain rule applied layer by layer along the DAG?
4. Why are in-place operations forbidden or error-prone on tensors that require grad?
5. Difference between .detach() and requires_grad=False? What is a leaf tensor?

<details>
<summary>中文题目</summary>

1. 反向传播如何确定梯度计算顺序? 为什么需要拓扑逆序?
2. 动态图与静态图的区别与各自优缺点?
3. grad_fn 的作用? 链式法则如何沿 DAG 逐层回传?
4. 为什么 in-place 操作在需要梯度时会被禁止或报错?
5. .detach() 与 requires_grad=False 的区别? 什么是 leaf 张量?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/autograd-engine?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=autograd-engine)

---

## 4. Batch Normalization · BatchNorm 批归一化

> Batch Normalization normalizes activations along the batch axis per feature channel to combat internal covariate shift. Given input $x$, compute the batch mean and variance $\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$,…

**Interview questions**

1. Write the full BatchNorm computation: how $\mu_B$ and $\sigma_B^2$ are computed, the normalization $\hat{x} = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$, and the role of the affine parameters $\gamma,\beta$.
2. Train vs inference behavior: why must inference use the running mean/variance (EMA) instead of batch statistics, and when is the EMA updated?
3. Why does BN fail with small batches? What happens at batch=1, and what problems arise from train/inference statistic mismatch?
4. Why do Transformers/LLMs use LayerNorm instead of BatchNorm? What issues does BN face with variable-length sequences and padding?
5. Why does BN speed up convergence (ICS / loss-surface smoothing)? What is its regularizing side effect, and what order is used when combining it with Dropout?

<details>
<summary>中文题目</summary>

1. 写出 BatchNorm 的完整计算流程: 如何算 $\mu_B$、$\sigma_B^2$, 归一化公式 $\hat{x} = \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$ 与仿射变换 $y = \gamma\hat{x}+\beta$ 中 $\gamma,\beta$ 的作用?
2. 训练与推理时 BN 的行为差异: 训练用 batch 统计量, 推理为什么必须用滑动平均 (running mean/var)? EMA 在哪个阶段更新?
3. 小 batch 下 BN 为什么失效? batch=1 时会发生什么? 训练/推理统计量不一致会造成什么问题?
4. 为什么 Transformer/LLM 中 BN 很少用而改用 LayerNorm? BN 在变长序列、padding 上的问题是什么?
5. BN 为何能加速收敛 (内部协变量偏移/loss 平滑)? 它有什么正则化副作用, 与 Dropout 并用的顺序如何?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/batchnorm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=batchnorm)

---

## 5. Classic CNN Architectures · 经典 CNN 演进

> The CNN evolution line: LeNet (1998, 5-layer conv+pool+FC, handwritten digits) → VGG (2014, all-3×3 small-kernel stacking, 16/19 layers) → ResNet (2015, residual shortcuts, up to 152 layers) → DenseNet (2017, dense layer-wise concatenation, feature reuse).

**Interview questions**

1. Why stack two 3×3 convs instead of one 5×5? Compare receptive field, parameters and computation.
2. Outline LeNet→VGG→ResNet→DenseNet; what problem does each generation solve over its predecessor?
3. Why does halving resolution while doubling channels keep FLOPs roughly constant?
4. Compare VGG vs ResNet: depth, parameters, training difficulty, accuracy.
5. Why does depth matter for CNN accuracy? What is the bottleneck of naive deepening (degradation)?

<details>
<summary>中文题目</summary>

1. 为什么用两个 3×3 堆叠替代一个 5×5 卷积? 感受野、参数量、计算量分别对比。
2. 简述 LeNet→VGG→ResNet→DenseNet 的演进, 每一代解决了上一代的什么问题?
3. CNN 设计中“分辨率减半、通道翻倍”为什么能让计算量基本不变?
4. VGG 与 ResNet 对比: 深度、参数量、训练难度、精度表现有什么差异?
5. 为什么深度对 CNN 精度重要? 单纯增加深度的瓶颈是什么(退化问题)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/cnn-architectures?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=cnn-architectures)

---

## 6. Convolution Basics · 卷积与感受野

> Convolution is the fundamental operator of CNNs, with two core properties — parameter sharing and local connectivity. For input $H \times W \times C_{in}$ and kernel $k_h \times k_w \times C_{in} \times C_{out}$, the output size is…

**Interview questions**

1. Hand-compute the output size: 224×224 input, 5×5 kernel, stride 2, padding 1.
2. Count parameters: 3×3 conv, 64 input channels, 128 output channels, including bias.
3. What do parameter sharing and local connectivity each solve? Why is conv parameter count independent of input resolution?
4. A 5×5 conv vs two stacked 3×3 convs have the same receptive field; compare their parameter counts and the saving.
5. What does a 1×1 conv do? How does dilated convolution enlarge the receptive field without downsampling?

<details>
<summary>中文题目</summary>

1. 手算输出尺寸: 输入 224×224, 5×5 卷积核, stride=2, padding=1, 输出多大?
2. 手算参数量: 3×3 卷积, 输入 64 通道输出 128 通道(含 bias)有多少参数?
3. 参数共享和局部连接分别解决了什么问题? 为什么卷积参数量与输入分辨率无关?
4. 一个 5×5 卷积和一个 3×3+3×3 堆叠感受野相同, 参数量分别是多少? 省了多少?
5. 1×1 卷积的作用是什么? 空洞卷积(dilated conv)如何在不降分辨率的前提下扩大感受野?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/convolution-basics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=convolution-basics)

---

## 7. Data Augmentation · 数据增强

> Data augmentation expands the training set with label-preserving transformations. Geometric: horizontal flip, random crop, rotation, scaling, color jitter (exploiting translation/flip invariance priors).

**Interview questions**

1. Write the Mixup formula. Why does Mixup act as regularization?
2. Why do random crops/flips work? Which image priors do they exploit?
3. How does augmentation relate to dropout and weight decay as regularizers?
4. How does augmentation construct positive/negative pairs in self-supervised learning (SimCLR)?
5. What are the side effects of over-aggressive augmentation, and how to avoid them?

<details>
<summary>中文题目</summary>

1. 写出 Mixup 的公式? 为什么 Mixup 等价于正则化?
2. 随机裁剪/翻转为什么有效? 利用了图像的什么先验?
3. 数据增强与 dropout/weight decay 在正则化上的关系?
4. 增强在自监督学习中如何构造正负样本 (SimCLR)?
5. 过度增强有什么副作用? 如何避免?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/data-augmentation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=data-augmentation)

---

## 8. DDP & Ring-AllReduce · DDP 与 Ring-AllReduce

> In DDP (distributed data parallelism), N ranks each hold the gradients $g_i$ of their data shard, and training must sum (or average) them into the global gradient — exactly the All-Reduce operation.

**Interview questions**

1. Derive Ring-AllReduce's volume: $S/N$ per step, $2(N-1)$ steps total → $V = 2\frac{N-1}{N}S$; show $V \to 2S$ as $N \to \infty$, i.e. the volume is independent of N.
2. Describe the two phases: how chunks circulate (index $(i-k) \bmod N$), why each phase takes exactly N−1 steps, and how every rank ends with the global sum of all chunks.
3. What is the per-rank volume of naive All-Reduce (each rank sends full $S$ to N−1 others) — $O(NS)$? Why is Ring bandwidth-optimal ($2S$ being near the lower bound of reading the full gradient)?
4. If bandwidth doesn't grow with N, why does latency still grow ($2(N-1)$ serial hops)? In which scenarios does NCCL switch to tree/hierarchical algorithms (small messages, slow networks, across nodes)?
5. How does DDP hide communication latency: gradient bucketing (~25MB) + launching each bucket's All-Reduce asynchronously once ready, overlapping with the remaining backprop; why is global gradient clipping done after All-Reduce and before the step?

<details>
<summary>中文题目</summary>

1. 推导 Ring-AllReduce 通信量: 每步每卡 $S/N$、两阶段共 $2(N-1)$ 步 → $V = 2\frac{N-1}{N}S$;证明当 $N \to \infty$ 时 $V \to 2S$,即通信量与 N 无关。
2. 描述 Scatter-Reduce 与 All-Gather 两阶段: chunk 如何绕环流转(下标 $(i-k) \bmod N$)、每阶段为何恰好 N−1 步、每个 rank 最终如何得到全部 chunk 的全局和。
3. 朴素 All-Reduce(每卡向其余 N−1 卡发全量 $S$)的每卡通信量是多少($O(NS)$)?为什么说 Ring 是带宽最优($2S$ 是下界,全量梯度至少被读一遍)?
4. 带宽不随 N 增长,延迟为什么仍随 N 增长(2(N−1) 次串行 hop)?什么场景下 NCCL 会改用 tree/分层算法(小消息、慢网络、跨节点)?
5. DDP 如何隐藏通信延迟: bucket 分桶(约 25MB)+ 反向中每个 bucket 就绪即异步 All-Reduce 与后续反向重叠;全局梯度裁剪的时机为什么是 All-Reduce 之后、step 之前?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/ddp-ring-allreduce?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ddp-ring-allreduce)

---

## 9. Depthwise Separable Conv · 深度可分离卷积

> Depthwise separable convolution splits a standard conv into two steps: Depthwise — each channel is convolved independently with one $k \times k$ kernel (no cross-channel mixing); Pointwise — a $1 \times 1$ conv mixes channels linearly.

**Interview questions**

1. Derive the depthwise-separable vs standard-conv compute ratio $\frac{1}{C_{out}} + \frac{1}{k^2}$; where does each term come from?
2. Hand-compute: 3×3 depthwise separable conv with C_out=256 — what fraction of standard-conv compute is left, and what is the saving?
3. Why does MobileNet need the 1×1 pointwise conv? What breaks if you keep only depthwise?
4. What is MobileNetV2's inverted residual (linear bottleneck)? Why does ReLU destroy low-dimensional features?
5. How do depthwise separable conv and grouped conv (e.g. ResNeXt's 32 groups) differ in compute and expressiveness?

<details>
<summary>中文题目</summary>

1. 推导深度可分离卷积相对标准卷积的算力比 $\frac{1}{C_{out}} + \frac{1}{k^2}$, 说明两项各自来源。
2. 手算: 3×3 深度可分离卷积, C_out=256, 计算量约为标准卷积的多少? 省了多少倍?
3. 为什么 MobileNet 需要 1×1 pointwise 卷积? 只用 depthwise 会怎样?
4. MobileNetV2 的反向残差(线性瓶颈)是什么? 为什么 ReLU 会破坏低维特征?
5. 深度可分离卷积与分组卷积(如 ResNeXt 的 32 组)在算力和表达力上有什么区别?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/depthwise-separable?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=depthwise-separable)

---

## 10. Dropout · Dropout

> Dropout randomly zeroes neurons with probability $p$ during training to prevent co-adaptation. The standard implementation is Inverted Dropout: sample a Bernoulli mask $m \sim \text{Bernoulli}(1-p)$ and scale $y = \frac{1}{1-p} m \odot x$ so that…

**Interview questions**

1. Write the inverted-dropout training formula $y = \frac{1}{1-p} m \odot x$ ($m \sim \text{Bernoulli}(1-p)$) and explain why inference needs no scaling to preserve the expectation.
2. What is the mathematical relationship between naive dropout (scaling at test) and inverted dropout (scaling at train)? Why does every framework implement the inverted version?
3. Why does dropout regularize: its link to ensemble learning and co-adaptation — and which layers is it usually applied to?
4. What happens if dropout is not disabled at inference (model.eval())? What problems arise from train/inference inconsistency?
5. What order should be used when combining dropout with BatchNorm, and why is dropout usually applied before BN?

<details>
<summary>中文题目</summary>

1. 写出 Inverted Dropout 训练公式 $y = \frac{1}{1-p} m \odot x$ ($m \sim \text{Bernoulli}(1-p)$) 并解释为什么推理时不缩放能保持期望一致?
2. 朴素 Dropout (推理时缩放) 与 Inverted Dropout (训练时缩放) 的数学关系? 为什么工程上都用 Inverted 版本?
3. Dropout 为什么能正则化: 与集成学习、共适应 (co-adaptation) 的关系? 一般在哪些层用?
4. 模型推理 (model.eval()) 时忘记关闭 Dropout 会怎样? 训练/推理行为不一致会导致什么问题?
5. Dropout 与 BatchNorm 并用的顺序? 为什么一般先 Dropout 再 BN?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/dropout?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=dropout)

---

## 11. GAN Adversarial Training · GAN 对抗训练

> GANs pit a generator $G$ against a discriminator $D$ in a zero-sum min-max game: $\min_G\max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$.

**Interview questions**

1. Write GAN's min-max objective and derive the optimal discriminator $D^*(x)$?
2. Why does substituting $D^*$ reduce GAN to minimizing $JSD(p_{data} \Vert p_g)$?
3. Why is GAN training unstable? Mechanisms behind vanishing gradients and mode collapse?
4. Why do GANs suffer mode collapse while VAEs do not? How do their sample qualities differ?
5. Common techniques to stabilize GAN training (label smoothing / TTUR / spectral norm)?

<details>
<summary>中文题目</summary>

1. 写出 GAN 的 min-max 目标函数, 并推导最优判别器 $D^*(x)$?
2. 把 $D^*$ 代入后为什么 GAN 等价于最小化 $JSD(p_{data} \Vert p_g)$?
3. GAN 训练为什么不稳定? 梯度消失与模式坍缩的机理?
4. 为什么 GAN 容易模式坍缩而 VAE 不会? 两者生成质量有何差异?
5. 缓解 GAN 训练不稳定的常用技巧 (label smoothing / TTUR / spectral norm)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/gan?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gan)

---

## 12. Gradient Checking · 梯度检查

> Approximate the derivative numerically and compare it with the analytical gradient: central difference $f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$, error $O(\epsilon^2)$ (one-sided differencing is only $O(\epsilon)$), with $\epsilon$ typica…

**Interview questions**

1. Write the central-difference gradient formula. Why is the error $O(\epsilon^2)$?
2. How is relative error computed? Why do shallow and deep networks need different thresholds?
3. Why disable dropout/BN randomness and use tiny samples during gradient checking?
4. When is gradient checking appropriate? Why can't it run constantly?
5. What goes wrong if $\epsilon$ is too large or too small?

<details>
<summary>中文题目</summary>

1. 写出中心差分数值梯度公式, 为什么误差是 $O(\epsilon^2)$?
2. 相对误差如何计算? 浅网络与深网络的阈值为何不同?
3. 梯度检查时为什么要关闭 dropout/BN 随机性并只用小样本?
4. 梯度检查的最佳时机? 为什么不能高频执行?
5. $\epsilon$ 过大或过小分别会有什么问题?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/gradient-check?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gradient-check)

---

## 13. Gradient Clipping · 梯度裁剪

> Gradient clipping bounds the gradient norm before the optimizer step to prevent exploding gradients. Two approaches: ① per-parameter (value) clipping $g_i \leftarrow \text{clip}(g_i, -c, c)$, i.e.

**Interview questions**

1. Write the global-norm clipping formula: compute $G = \Vert g \Vert_2$, and when $G > c$ apply $g \leftarrow g \cdot \frac{c}{G}$; why does uniform scaling preserve the direction and bound the step norm at $c$?
2. Per-parameter clipping $g_i \leftarrow g_i \cdot \min(1, c/|g_i|)$ vs global-norm clipping: drawbacks of each (direction distortion, uncontrollable global norm), and when to use which.
3. When does clipping happen in DDP (after gradient All-Reduce, before the step)? Why clip only after synchronization? Why clip the FP32 gradients in mixed precision?
4. Full loss-spike (NaN/Inf) handling: why must steps with NaN gradients be skipped, halving the AMP loss scale ($S \leftarrow S/2$) with step replay, and the ordering of checkpoint rollback and LR reduction.
5. Effects of $c$ being too small / too large? How can monitoring tell you that clipping is too frequent (the ratio $c/G$ persistently far below 1) and the LR is too large?

<details>
<summary>中文题目</summary>

1. 写出全局范数裁剪公式: 先算 $G = \Vert g \Vert_2$,当 $G > c$ 时 $g \leftarrow g \cdot \frac{c}{G}$;为什么统一缩放能保持方向并把步长范数界在 $c$?
2. 逐参数裁剪 $g_i \leftarrow g_i \cdot \min(1, c/|g_i|)$ vs 全局范数裁剪: 各自的缺点(方向扭曲、全局范数不可控),什么场景用哪个?
3. 在 DDP 中裁剪的时机是什么(梯度 All-Reduce 之后、step 之前)?为什么必须先完成梯度同步再裁剪?混合精度下为什么在 FP32 梯度上裁剪?
4. Loss Spike(NaN/Inf)的完整处理流程: 为什么必须跳过含 NaN 梯度的 step、AMP 下 loss scale 减半($S \leftarrow S/2$)并重放、checkpoint 回滚与降 lr 的次序。
5. 裁剪阈值 $c$ 太小/太大分别有什么影响?如何从监控数据判断“裁剪过于频繁”说明学习率过大(缩放比 $c/G$ 长期远小于 1)?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/gradient-clipping-amp?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gradient-clipping-amp)

---

## 14. Graph Attention Network · GAT 注意力

> GAT (Graph Attention Network, Veličković et al. 2018) assigns learnable importance weights to neighbors via attention. After a linear transform $Wh_i$, it scores the concatenated pair with a shared attention vector:…

**Interview questions**

1. Write the GAT attention formula $\alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(a^T[Wh_i \Vert Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\mathrm{LeakyReLU}(a^T[Wh_i \Vert Wh_k]))}$ and explain the roles of $\Vert$ concatenation, LeakyReLU, and softmax.
2. Core difference GAT vs GCN: GCN's coefficients are fixed by topology (degree-based); how does GAT distinguish same-degree neighbors (structurally symmetric but semantically different pairs)?
3. Multi-head handling: why do hidden layers concatenate heads while the last layer averages them? How does multi-head differ from single-head in training stability and capacity?
4. Why is softmax applied only over neighbors $\mathcal{N}(i)$ instead of the whole graph? Compare with Transformer's global softmax in terms of complexity and locality.
5. Similarities and differences between GAT and Transformer self-attention: where are query/key/value, how is graph structure injected via masking, and when can one be converted into the other?

<details>
<summary>中文题目</summary>

1. 默写 GAT 注意力系数公式 $\alpha_{ij} = \frac{\exp(\mathrm{LeakyReLU}(a^T[Wh_i \Vert Wh_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\mathrm{LeakyReLU}(a^T[Wh_i \Vert Wh_k]))}$ 并解释 $\Vert$ 拼接、LeakyReLU 与 softmax 各自的作用。
2. GAT vs GCN 的核心区别: GCN 的系数是结构固定的(由度决定), GAT 为什么能区分同度邻居(结构不可分但语义不同的节点对)?
3. 多头注意力的两种处理: 中间层为什么拼接、输出层为什么平均? 多头与单头在训练稳定性与表达能力上的差异。
4. 为什么注意力系数只对邻居 $\mathcal{N}(i)$ 做 softmax 归一化而不是对全图? 与 Transformer 全局 softmax 的对比(复杂度与局部性)。
5. GAT 与 Transformer 自注意力的异同: 查询/键/值在哪里、图结构如何通过 mask 融入、二者何时可以互相转换?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/gat?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gat)

---

## 15. Graph Conv Network · GCN 谱域

> GCN (Graph Convolutional Network, Kipf & Welling 2017) propagates as $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$, where $\tilde{A} = A + I$ adds self-loops, $\tilde{D}$ is its degree matrix, $W^{(l)}$ is learnable, and…

**Interview questions**

1. Write the GCN propagation $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$ and explain the roles of $\tilde{A} = A + I$, the symmetric normalization, and $W^{(l)}$.
2. Spectral derivation: why is the full graph convolution infeasible ($O(n^2)$ eigen-decomposition, dense $U$)? How does the first-order Chebyshev approximation $g_\theta \approx \theta_0 I + \theta_1 \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ reduce step by step to the GCN formula?
3. Why symmetric normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ instead of the random-walk normalization $\tilde{D}^{-1}\tilde{A}$? Differences in numerical stability and aggregation semantics (weighted average vs normalized sum).
4. Mechanism of oversmoothing: why do node representations converge as layers deepen (repeated low-pass filtering, dominant spectral components)? What are the mitigations?
5. Transductive vs inductive learning: why is classic GCN transductive? How does it generalize to unseen nodes/graphs at inference?

<details>
<summary>中文题目</summary>

1. 默写 GCN 传播公式 $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$, 并解释 $\tilde{A} = A + I$、$\tilde{D}^{-1/2}(\cdot)\tilde{D}^{-1/2}$、$W^{(l)}$ 各自的作用。
2. 谱域推导: 完整图卷积为何不可行($O(n^2)$ 特征分解、稠密 $U$), 一阶切比雪夫近似 $g_\theta \approx \theta_0 I + \theta_1 \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 如何一步步化简到 GCN 公式?
3. 为什么用对称归一化 $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 而不是随机游走归一化 $\tilde{D}^{-1}\tilde{A}$? 两者在数值稳定性与聚合语义(加权平均 vs 归一化求和)上的区别。
4. 过平滑(oversmoothing)的机制: 为什么层数加深后节点表示趋同(反复低通滤波、谱主特征向量主导)? 有哪些缓解手段?
5. 直推式(transductive)与归纳式(inductive)学习: 经典 GCN 为什么是直推式? 训练时未见的新节点/新图如何泛化?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/gcn?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gcn)

---

## 16. Graph Representation · 图表示与邻接矩阵

> A graph $G = (V, E)$ is encoded as matrices so neural networks can process it. The adjacency matrix $A \in \mathbb{R}^{n \times n}$ has $A_{ij} = 1$ if nodes $i, j$ share an edge (arbitrary weights for weighted graphs);

**Interview questions**

1. Define the adjacency matrix $A$ and degree matrix $D$ ($A_{ij}$, $d_i = \sum_j A_{ij}$); why is adding self-loops $\tilde{A} = A + I$ necessary for GNNs?
2. Derive the element form of symmetric normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, i.e. $\frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i \tilde{d}_j}}$, and explain why it prevents high-degree nodes from dominating aggregation.
3. Properties of the graph Laplacian $L = D - A$ and normalized Laplacian $\mathcal{L} = I - D^{-1/2}AD^{-1/2}$: the quadratic form $x^T L x = \sum_{(i,j)}(x_i - x_j)^2$ and the eigenvalue range $[0, 2]$.
4. Name one example for each of the three graph task levels (node/edge/graph), and what are the prediction targets and typical loss forms?
5. How to extend the basic matrix representation: weighted graphs, directed graphs (non-symmetric $A$), heterogeneous graphs (multiple edge types), and sparse storage for billion-node graphs?

<details>
<summary>中文题目</summary>

1. 写出邻接矩阵 $A$ 与度矩阵 $D$ 的定义($A_{ij}$, $d_i = \sum_j A_{ij}$), 自环 $\tilde{A} = A + I$ 为什么是 GNN 的必要操作?
2. 推导对称归一化 $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 的元素形式 $\frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i \tilde{d}_j}}$, 并解释它为什么能避免度大节点主导聚合结果。
3. 图拉普拉斯 $L = D - A$ 与归一化拉普拉斯 $\mathcal{L} = I - D^{-1/2}AD^{-1/2}$ 的性质: 二次型 $x^T L x = \sum_{(i,j)}(x_i - x_j)^2$ 与特征值范围 $[0, 2]$ 的含义。
4. 图任务的三大层级(节点级/边级/图级)各举一例, 各自的预测目标与常用 loss 形式是什么?
5. 如何扩展基本矩阵表示: 加权图、有向图(非对称 $A$)、异质图(多类型边)以及超大图的邻接矩阵稀疏存储?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/graph-basics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=graph-basics)

---

## 17. GRU · GRU

> GRU (Gated Recurrent Unit) is a simplified LSTM variant with only two gates. The update gate $z_t = \sigma(W_z[h_{t-1}, x_t] + b_z)$ decides how much history to keep vs how much new information to write;

**Interview questions**

1. Write out GRU's update gate $z_t$, reset gate $r_t$, candidate $\tilde{h}_t$ and output $h_t = (1 - z_t)h_{t-1} + z_t\tilde{h}_t$; explain the role of each gate.
2. Structural differences: why does GRU have ~25% fewer parameters (3 vs 4 weight sets)? How does the absence of a cell state affect memory granularity?
3. What happens when $z_t \approx 0$ ($h_t \approx h_{t-1}$ additive shortcut)? How does this correspond to LSTM's forget gate $f_t \approx 1$?
4. Role of the reset gate: when $r_t \approx 0$ the candidate $\tilde{h}_t$ depends only on the current input — why does this help capture short-term dependencies and discard irrelevant history?
5. When to pick GRU over LSTM (data size, parameter budget, inference latency)? Why is the accuracy gap between them usually small on typical tasks?

<details>
<summary>中文题目</summary>

1. 默写 GRU 更新门 $z_t$、重置门 $r_t$、候选 $\tilde{h}_t$ 与输出 $h_t = (1 - z_t)h_{t-1} + z_t\tilde{h}_t$ 的公式, 并解释每个门的职责。
2. GRU vs LSTM 的结构差异: 为什么 GRU 参数量约少 25%(3 组 vs 4 组权重)? 没有细胞状态对记忆粒度有什么影响?
3. 更新门 $z_t \approx 0$ 时 GRU 发生了什么($h_t \approx h_{t-1}$ 的加性直通)? 这与 LSTM 遗忘门 $f_t \approx 1$ 的机制有何对应?
4. 重置门的作用: $r_t \approx 0$ 时候选 $\tilde{h}_t$ 只依赖当前输入, 这为什么有利于捕捉短期依赖和丢弃无用历史?
5. 何时选 GRU 而非 LSTM: 数据量、参数量预算、推理延迟等角度; 两者在常见任务上的精度差距为何通常很小?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/gru?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=gru)

---

## 18. Kaiming Init Derivation · Kaiming 初始化方差推导

> The Kaiming (He) initialization derivation constructs a variance-preserving weight distribution for ReLU networks, concluding $\text{Var}(w) = 2/d_{\text{in}}$.

**Interview questions**

1. Full derivation: from $\text{Var}(y) = d\,E[w^2]\,E[x^2]$, through ReLU zeroing, derive $\text{Var}(w) = 2/d_{\text{in}}$ (write every step, including the zero-mean and independence assumptions).
2. Derive the factor 2: use the integral $E[(x')^2] = \tfrac{1}{2}\text{Var}(z)$ to show how ReLU zeroing halves the activation energy; which assumptions does it rely on (zero mean, symmetric distribution, $w$ independent of $x$)?
3. Why are the forward and backward variance-preserving conditions $2/d_{\text{in}}$ and $2/d_{\text{out}}$ respectively? How does ReLU's ½ gradient pass-through enter the backward derivation? Which does the framework default to (fan_in)?
4. Kaiming vs Xavier: how does $\text{Var}(w) = 2/d$ reduce to $1/d$ without ReLU? Why do tanh networks use Xavier while ReLU networks must use Kaiming (what goes wrong with $1/d$)?
5. PReLU generalization: how does $\text{Var}(w) = \frac{2}{(1+a^2)d_{\text{in}}}$ follow from $E[(x')^2] = \tfrac{1+a^2}{2}\text{Var}(z)$; if the assumptions break (asymmetric or nonzero-mean inputs), which step of the derivation fails?

<details>
<summary>中文题目</summary>

1. 完整推导链: 由 $\text{Var}(y) = d\,E[w^2]\,E[x^2]$ 出发,经过 ReLU 置零推导出 $\text{Var}(w) = 2/d_{\text{in}}$(要求写出每一步,含零均值与独立性假设)。
2. 推导 Kaiming 因子 2: 用积分 $E[(x')^2] = \tfrac{1}{2}\text{Var}(z)$ 说明 ReLU 置零负半轴如何把激活均方砍半;依赖哪些假设(零均值、对称分布、$w$ 与 $x$ 独立)?
3. 前向与反向的方差保持条件为什么分别是 $2/d_{\text{in}}$ 与 $2/d_{\text{out}}$?反向时 ReLU 导数的“梯度通过率 ½”如何进入推导?框架默认用哪个(fan_in)?
4. Kaiming 与 Xavier 的关系: 无 ReLU 时 $\text{Var}(w) = 2/d$ 如何退化为 $1/d$?为什么 tanh 网络用 Xavier 而 ReLU 网络必须用 Kaiming(用 $1/d$ 会怎样)?
5. PReLU 推广: 推导 $\text{Var}(w) = \frac{2}{(1+a^2)d_{\text{in}}}$ 如何来自 $E[(x')^2] = \tfrac{1+a^2}{2}\text{Var}(z)$;如果假设不成立(如输入非对称、非零均值)推导哪一步失效?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/kaiming-init-derivation?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=kaiming-init-derivation)

---

## 19. Label Smoothing · Label Smoothing

> label smoothing mixes the one-hot label with a uniform distribution: $y' = y(1-\epsilon) + \frac{\epsilon}{K}$, where $K$ is the number of classes and $\epsilon$ is typically 0.1;

**Interview questions**

1. Write the label-smoothing formula $y' = y(1-\epsilon) + \frac{\epsilon}{K}$, explain the roles of $\epsilon$ and $K$, and state the usual value of $\epsilon$.
2. Why does label smoothing prevent overconfidence? How does it affect logit magnitudes and model calibration?
3. Why does label smoothing hurt knowledge distillation? What information in the teacher's output does it destroy?
4. What regularization term is label smoothing equivalent to adding to cross-entropy? Write the corresponding loss form.
5. In which scenarios should label smoothing be avoided? How does it interact with temperature scaling?

<details>
<summary>中文题目</summary>

1. 写出 Label Smoothing 公式 $y' = y(1-\epsilon) + \frac{\epsilon}{K}$ 并解释 $\epsilon$ 与 $K$ 的作用? $\epsilon$ 通常取多少?
2. Label Smoothing 为什么能防止过自信 (overconfidence)? 对 logits 大小与模型校准 (calibration) 的影响?
3. 为什么 Label Smoothing 会损害知识蒸馏? teacher 输出中的什么信息被平滑抹掉了?
4. Label Smoothing 等价于在交叉熵上添加什么样的正则项? 写出对应的损失函数形式?
5. 什么场景不应该用 Label Smoothing? 它与温度缩放 (temperature) 的配合使用?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/label-smoothing?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=label-smoothing)

---

## 20. LayerNorm & RMSNorm · LayerNorm 与 RMSNorm

> LayerNorm normalizes a single sample across its feature dimension, independent of the batch: for $x \in \mathbb{R}^d$, compute $\mu = \frac{1}{d}\sum_{i=1}^{d}x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$, giving…

**Interview questions**

1. Write the LayerNorm formula $y = \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$, state which axis it normalizes over, and why it is independent of batch size.
2. How does RMSNorm differ from LayerNorm, why does removing the mean matter little, and what does it save in LLMs (compute / reported speedup)?
3. The role of $\gamma$ and $\beta$ in LN: what happens if they are removed, and why are they needed to restore representational power?
4. Pre-Norm vs Post-Norm: why do modern Transformers use Pre-LN (training stability and the residual gradient path)?
5. Why must RNNs/variable-length sequences use LayerNorm rather than BatchNorm? Compare their batch dependency and padding handling.

<details>
<summary>中文题目</summary>

1. 写出 LayerNorm 公式 $y = \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ 并说明它沿哪个维度归一化、为什么与 batch size 无关?
2. RMSNorm 与 LN 的区别: 去掉均值中心化为什么影响很小? 在 LLM 中主要省了什么 (计算量/提速多少)?
3. LN 中 $\gamma, \beta$ 的作用: 去掉可学习参数会发生什么? 为什么需要它们恢复表达能力?
4. Pre-Norm 与 Post-Norm 的区别, 为什么现代 Transformer 用 Pre-LN (训练稳定性、残差梯度路径)?
5. 为什么 RNN/变长序列场景必须用 LN 而非 BN? 对比两者在 batch 依赖、padding 处理上的差异?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/layernorm-rmsnorm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=layernorm-rmsnorm)

---

## 21. Learning Rate Scheduling · 学习率调度

> LR scheduling changes the learning rate $\eta_t$ over training according to a plan. Four mainstream strategies: ① Warmup (linear): $\eta_t = \eta_{\max}\cdot\frac{t}{T_{\text{warm}}}$ for $t = 1 \ldots T_{\text{warm}}$, ramping linearly from near zero to the p…

**Interview questions**

1. Write the linear warmup $\eta_t = \eta_{\max}\cdot\frac{t}{T_{\text{warm}}}$ and cosine annealing $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\frac{t}{T}\pi\right)\right)$; what is $\eta_t$ at $t=0$ and $t=T$?
2. Why is warmup needed at the start? Derive why $\eta/\sqrt{\hat{v}_t}$ can be huge early in Adam (only a few gradients have accumulated into $\hat{v}_t$) and why large batches make it riskier.
3. Cosine annealing vs step decay: why is a smooth decline better (avoiding loss spikes from cliffs)? What does a tiny final LR do (converging to flat minima, better generalization)?
4. Linear scaling rule (Goyal): why does doubling the batch size roughly double the LR? How does gradient noise scale with batch size? Why must warmup accompany it?
5. Typical large-model config: warmup fraction (1%–5%), peak LR, cosine endpoint; how do the schedule and the optimizer (Adam's $\beta_2$, $\epsilon$) interact?

<details>
<summary>中文题目</summary>

1. 写出线性 warmup 公式 $\eta_t = \eta_{\max}\cdot\frac{t}{T_{\text{warm}}}$ 与余弦退火公式 $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\left(\frac{t}{T}\pi\right)\right)$,说明 t=0 与 t=T 时 $\eta_t$ 分别等于什么。
2. 为什么训练初期需要 warmup?推导 Adam 早期 $\eta/\sqrt{\hat{v}_t}$ 为什么可能巨大($\hat{v}_t$ 只积累了几个梯度),以及大 batch 下风险为何更大。
3. 余弦退火 vs 阶梯衰减: 为什么平滑下降更好(避免悬崖式掉速的 loss 突跳)?末端小学习率的作用(收敛到平坦极小、泛化更好)。
4. 线性缩放规则(Goyal): batch size 翻倍时学习率为什么约翻倍?梯度噪声如何随 batch 变化?为什么必须配 warmup?
5. 典型大模型配置: warmup 步数占比(1%~5%)、峰值学习率、余弦末端最小值;调度与优化器(Adam 的 $\beta_2$、$\epsilon$)如何协同设定?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/lr-scheduler?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=lr-scheduler)

---

## 22. Loss Functions · 损失函数族

> The loss family dictates the geometry of the objective and convergence speed. Cross-entropy is standard for classification: $L=-\sum_{i}y_i\log p_i$, reducing to BCE $-[y\log\hat y+(1-y)\log(1-\hat y)]$ for binary cases;

**Interview questions**

1. Derive the softmax+CE gradient $p_i-y_i$; contrast with softmax+MSE $(p_i-y_i)p_i(1-p_i)$ — why does CE converge faster?
2. Write Focal Loss $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$; how do easy-sample weights change at $\gamma=2$ and why does it fix imbalance?
3. Write the InfoNCE formula and explain temperature $\tau$; why does a too-small $\tau$ over-strengthen negative-sample gradients?
4. Relation between CE and BCE? Why BCE for multi-label classification instead of multi-class CE?
5. Why is CE not used for regression? Compare MSE / MAE / Huber gradients and outlier robustness?

<details>
<summary>中文题目</summary>

1. 白板推导 Softmax+CE 梯度 $p_i-y_i$,并对比 Softmax+MSE 的 $(p_i-y_i)p_i(1-p_i)$: 为什么 CE 收敛更快?
2. 写出 Focal Loss $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$;$\gamma=2$ 时易分样本($p_t\to1$)权重如何变化?为什么能解决类别不平衡?
3. 写出 InfoNCE 公式并解释温度 $\tau$ 的作用;为什么 $\tau$ 太小会让负样本梯度过强?
4. CE 与 BCE 的关系?多标签分类为什么用 BCE 而非多分类 CE?
5. 回归任务为什么不用 CE?MSE / MAE / Huber 的梯度与对离群点鲁棒性对比?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/loss-functions-dl?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=loss-functions-dl)

---

## 23. LSTM Gating · LSTM 门控

> LSTM (Long Short-Term Memory) solves the RNN long-term dependency problem with “three gates and one state”. Forget gate $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$, input gate $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$, candidate memory…

**Interview questions**

1. Write out the LSTM gate equations (forget/input/output gates, candidate $\tilde{c}_t$, cell state $c_t$, hidden state $h_t$) and explain the role of each gate.
2. Why does LSTM mitigate vanishing gradients? Derive the additive path $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ and explain why gradients flow through long distances when $f_t \approx 1$.
3. Why is the forget-gate bias initialized to a large positive value ($b_f \approx 1$)? What happens if it saturates to 0 early (cell cleared, long memory lost)?
4. Compare LSTM vs GRU: GRU merges forget+input into an update gate and drops the separate cell state, cutting weights from 4 sets to 3; what is the capacity-vs-efficiency trade-off?
5. What does xLSTM (2024) improve: what problems do sLSTM's exponential gating & memory mixing and mLSTM's matrix memory & covariance update solve?

<details>
<summary>中文题目</summary>

1. 默写 LSTM 三门一状态公式(遗忘/输入/输出门、候选 $\tilde{c}_t$、细胞状态 $c_t$、隐藏态 $h_t$), 并说明每个门各自的职责。
2. 为什么 LSTM 能缓解梯度消失? 推导 $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ 的加性路径, 说明遗忘门 $f_t \approx 1$ 时梯度为何可以直通长距离。
3. 遗忘门偏置为什么要初始化为较大正值($b_f \approx 1$)? 遗忘门过早饱和到 0 会发生什么(细胞状态被清空、长记忆丢失)?
4. LSTM 与 GRU 的结构与参数对比: GRU 把遗忘+输入门合并为更新门、去掉独立细胞状态, 参数量从 4 组权重降到 3 组; 表达能力与训练效率的权衡是什么?
5. xLSTM(2024)改进了什么: sLSTM 的指数门控/内存混合与 mLSTM 的矩阵记忆/协方差更新各自解决什么问题?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/lstm?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=lstm)

---

## 24. Mixed Precision FP16/BF16/FP8 · 混合精度 FP16/BF16/FP8

> Mixed precision uses low-precision formats (FP16/BF16/FP8) for the forward and backward passes while keeping FP32 master weights and optimizer states, trading precision for speed and memory.

**Interview questions**

1. Write FP16's bit layout (1 sign + 5 exponent + 10 mantissa) and max value $65504 = 2^{15} \times (2 - 2^{-10})$; explain why training gradients often underflow (below $2^{-24} \approx 6\times10^{-8}$).
2. Loss scaling: why multiply the loss by $S$ before backprop and divide back on FP32 master weights before updating? Why does the dynamic policy (halve $S$ on overflow and skip the step, double after a clean streak) work?
3. Why doesn't BF16 (1+8+7) need loss scaling? Its exponent width equals FP32's, giving a range of about $\pm3.4\times10^{38}$; what is the cost (7 mantissa bits) and when is the precision insufficient?
4. Why are FP8 E4M3 and E5M2 used for the forward pass and gradients respectively? Compare their range/precision trade-offs (max ~448 vs ~57344, 3 vs 2 mantissa bits).
5. Standard mixed-precision pipeline: FP32 master weights, low-precision forward/backward, loss scaling and unscaling, FP32 updates; how does it compose with gradient clipping, EMA and DDP?

<details>
<summary>中文题目</summary>

1. 写出 FP16 位分布(1 符号 + 5 指数 + 10 尾数)与最大可表示数 $65504 = 2^{15} \times (2 - 2^{-10})$;解释训练中梯度为什么容易下溢(小于 $2^{-24} \approx 6\times10^{-8}$)。
2. Loss Scaling 原理: 为什么反向前把 loss 乘 $S$、更新前在 FP32 主权重上除回?动态策略(溢出时 $S$ 减半并跳过该步、连续无溢出则翻倍)为什么有效?
3. BF16(1+8+7)为什么不需要 loss scaling?指数位与 FP32 相同 → 范围约 $\pm3.4\times10^{38}$;代价(尾数 7 位)是什么,什么场景精度不够?
4. FP8 E4M3 与 E5M2 为什么分别用于前向与梯度?比较两者的范围/精度取舍(最大约 448 vs 57344,尾数 3 vs 2 位)。
5. 混合精度标准流程: FP32 master weights、低精度前向/反向、loss scaling 与 unscale、FP32 更新;与梯度裁剪、EMA、DDP 如何配合?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/mixed-precision-fp8?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=mixed-precision-fp8)

---

## 25. Norm Methods Comparison · BN vs LN vs GN 对比

> the three normalization families differ only in which axis their statistics are computed over (input $x \in \mathbb{R}^{N \times C \times H \times W}$): • BN: statistics over (N, H, W) — across batch and spatial dims, per channel:…

**Interview questions**

1. For $x \in \mathbb{R}^{N \times C \times H \times W}$, compare the normalization axes of BN/LN/GN: what is the statistic range of each (batch/sample/channel)?
2. Why is BN unsuitable for Transformers/variable-length sequences while LN is the default? Compare their batch dependency.
3. What problem does GroupNorm solve? Why does BN fail on small-batch detection/segmentation (1–2 images/GPU) while GN works?
4. What are BN's small-batch weakness and LN's channel-independence assumption? In which settings is LN worse than BN?
5. Why does normalization speed up training? Contrast the internal-covariate-shift view with the loss-landscape-smoothing view.

<details>
<summary>中文题目</summary>

1. 对 $x \in \mathbb{R}^{N \times C \times H \times W}$, 画出 BN/LN/GN 的归一化维度对比: 各自统计量的计算范围 (batch/样本/通道) 是什么?
2. 为什么 BN 在 Transformer/变长序列中不适用而 LN 成为标配? 两者的 batch 依赖差异?
3. GN 解决什么问题? 在小 batch (每卡 1~2 张) 的检测/分割任务中为什么 BN 失效而 GN 可行?
4. BN 的小 batch 缺陷与 LN 的通道独立性假设各是什么? 什么场景下 LN 不如 BN?
5. 归一化为什么能加速训练? 对比内部协变量偏移 (ICS) 与 loss 曲面平滑两种解释视角?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/normalization-comparison?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=normalization-comparison)

---

## 26. Receptive Field & FLOPs · 感受野与 FLOPs 计算

> The receptive field (RF) is the region of the original input visible to one output pixel, with recurrence $RF_l = RF_{l-1} + (k_l - 1)\prod_{i=1}^{l-1} s_i$, where $k_l$ is the kernel size of layer $l$ and $s_i$ the stride of layer $i$ — the stride product sho…

**Interview questions**

1. Hand-compute receptive fields: three stacked 3×3 stride-1 layers? With the first layer at stride 2? Show the recurrence.
2. Hand-compute FLOPs: 3×3 conv, 64→128 channels on 224×224 input. Why the factor of 2?
3. Derive the receptive-field recurrence $RF_l = RF_{l-1} + (k_l-1)\prod_{i=1}^{l-1} s_i$; what does the stride product mean?
4. Why do two stacked 3×3 convs match a 5×5 receptive field with fewer params and FLOPs? Give numbers.
5. Which factors does standard-conv FLOPs scale with, and how do you quickly estimate a whole network's FLOPs?

<details>
<summary>中文题目</summary>

1. 手算感受野: 连续 3 层 3×3(stride=1)感受野多大? 若第一层 stride=2 呢? 写出递推过程。
2. 手算 FLOPs: 224×224×64 输入上 3×3 卷积输出 128 通道, FLOPs 是多少? 为什么乘 2?
3. 推导感受野递推公式 $RF_l = RF_{l-1} + (k_l-1)\prod_{i=1}^{l-1} s_i$, 解释 stride 连乘项的物理含义。
4. 为什么堆叠两个 3×3 等于一个 5×5 的感受野, 但参数与计算量更少? 请给出数字。
5. 标准卷积的 FLOPs 与哪些因子成正比? 如何快速估算一个网络的整体 FLOPs?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/receptive-field-flops?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=receptive-field-flops)

---

## 27. ResNet Skip Connections · ResNet 残差连接

> A residual block computes $y = F(x, \{W_i\}) + x$ — the layer learns the residual $F(x) = \mathcal{H}(x) - x$ instead of the direct map $\mathcal{H}(x)$.

**Interview questions**

1. Derive the residual-block gradient $\frac{\partial y}{\partial x} = 1 + \frac{\partial F}{\partial x}$ and explain why the identity shortcut mitigates vanishing gradients.
2. What is the degradation problem? Why does a 56-layer net have higher training error than a 20-layer one without overfitting?
3. Why must the shortcut be the identity $x$ rather than a learned transform such as 1×1 conv?
4. When F changes channels/resolution, how do you match the shortcut dimensions, and what does each option cost?
5. How does ResNet make 152-layer (even 1000-layer) training possible? Answer from both optimization and gradient perspectives.

<details>
<summary>中文题目</summary>

1. 推导残差块梯度 $\frac{\partial y}{\partial x} = 1 + \frac{\partial F}{\partial x}$, 解释恒等捷径为什么能缓解梯度消失。
2. 退化问题(degradation)是什么? 为什么 56 层比 20 层训练误差更高却并非过拟合?
3. 为什么捷径必须是恒等映射 $x$ 而不是可学习的变换(如 1×1 卷积)?
4. 残差块输出维度不匹配时(F 改变通道数/分辨率), 有哪些处理方式? 各有什么代价?
5. ResNet 如何让训练 152 层甚至上千层网络成为可能? 从优化与梯度两个角度回答。

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/resnet-skip?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=resnet-skip)

---

## 28. RNN & BPTT · RNN 与 BPTT

> An RNN (recurrent neural network) is the fundamental model for sequential data, defined by weight sharing across time steps.

**Interview questions**

1. Write the vanilla RNN forward pass $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$; why are weights shared across time steps and why is the parameter count independent of $T$?
2. How does BPTT differ from standard backprop? How does the error travel through $\frac{\partial h_t}{\partial h_{t-1}}$ chain products, and why is the gradient w.r.t. $W_{hh}$ a sum of contributions from all time steps?
3. Derive vanishing/exploding gradients from $\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W_{hh}^T \mathrm{diag}(\tanh')$; how do the spectral radius $\rho(W_{hh})$ and the activation derivative determine whether long-range dependencies are learnable ($0.9^{50} \approx 0.005$)?
4. List the mitigations: why does gradient clipping (e.g. clip 1.0) fix exploding but not vanishing gradients? What roles do orthogonal initialization and gating play?
5. Why can't RNNs be trained in parallel ($h_t$ depends on $h_{t-1}$)? What does per-step complexity $O(d^2)$ mean? How does this contrast with Transformer ($O(L^2)$, parallel) and Mamba ($O(L)$, parallel)?

<details>
<summary>中文题目</summary>

1. 写出标准 RNN 前向 $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$, 解释为什么跨时间步共享权重、参数量为何与序列长度 $T$ 无关?
2. BPTT 与标准反向传播的区别: 误差如何沿 $\frac{\partial h_t}{\partial h_{t-1}}$ 连乘回到早期时刻, 为什么对 $W_{hh}$ 的梯度是所有时间步贡献之和?
3. 推导梯度消失/爆炸: 展开 $\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W_{hh}^T \mathrm{diag}(\tanh')$, 说明谱半径 $\rho(W_{hh})$ 与激活函数导数如何决定长期依赖能否学到 ($0.9^{50} \approx 0.005$)。
4. 缓解梯度问题的手段有哪些? 为什么梯度裁剪(clip 1.0)对爆炸有效、对消失无效? 正交初始化与门控分别起什么作用?
5. RNN 为什么无法并行训练($h_t$ 依赖 $h_{t-1}$)? 单步复杂度 $O(d^2)$ 指什么? 与 Transformer($O(L^2)$ 可并行)和 Mamba($O(L)$ 可并行)的对比如何?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/rnn-basics?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=rnn-basics)

---

## 29. SGD & Momentum · SGD 与动量

> Plain SGD updates along the negative instantaneous gradient: $\theta_{t+1} = \theta_t - \eta g_t$ with $g_t = \nabla_\theta L(\theta_t)$;

**Interview questions**

1. Write the momentum update $v_t = \gamma v_{t-1} + \eta g_t$, $\theta_{t+1} = \theta_t - v_t$; expand $v_t = \eta\sum_{k=0}^{t}\gamma^k g_{t-k}$ and explain how historical gradients decay by $\gamma^k$.
2. Why does momentum suppress oscillation and speed up convergence? Explain via the geometric weighted sum: consistent gradients accumulate, high-frequency reversals cancel, and why this helps on contours with large condition number $\kappa$.
3. Write the NAG update $v_t = \gamma v_{t-1} + \eta\nabla_\theta L(\theta_t - \gamma v_{t-1})$; how does the look-ahead step differ from vanilla momentum, and why does the convex rate improve from $O(1/t)$ to $O(1/t^2)$?
4. What does the default $\gamma = 0.9$ mean? What problems arise from too large / too small $\gamma$ (overshoot vs no memory)? Why is the effective step under constant gradient $\eta/(1-\gamma)$ (~10x amplification)?
5. How does momentum interact with LR scheduling? Why does momentum usually require LR decay (amplified effective step, end-of-training oscillation)? What is the single-learning-rate limitation of momentum vs Adam?

<details>
<summary>中文题目</summary>

1. 写出动量更新公式 $v_t = \gamma v_{t-1} + \eta g_t$、$\theta_{t+1} = \theta_t - v_t$,展开 $v_t = \eta\sum_{k=0}^{t}\gamma^k g_{t-k}$,说明历史梯度的权重如何按 $\gamma^k$ 衰减。
2. 为什么动量能抑制震荡、加速收敛?从几何加权和的角度解释: 方向一致的梯度累加、高频翻转的梯度抵消,在条件数 $\kappa$ 大的椭圆等高线上为什么收敛更快。
3. 写出 NAG 更新 $v_t = \gamma v_{t-1} + \eta\nabla_\theta L(\theta_t - \gamma v_{t-1})$,解释“先外推一步再算梯度”与普通动量的区别,以及凸问题上 $O(1/t) \to O(1/t^2)$ 的收敛速率提升。
4. 默认 $\gamma = 0.9$ 意味着什么?$\gamma$ 太大/太小分别导致什么问题(过冲回弹 vs 无记忆)?动量对梯度常数的有效步长为什么是 $\eta/(1-\gamma)$(≈10 倍放大)?
5. 动量与学习率调度如何配合?为什么加动量后通常要配学习率衰减(有效步长被放大、末期易震荡)?与 Adam 相比动量只有单一学习率的局限是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/sgd-momentum?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=sgd-momentum)

---

## 30. SSM & Mamba · SSM 与 Mamba

> State-space models (SSMs, e.g. S4/Mamba) frame sequence modeling as a linear time-invariant system: continuous form $\dot{h}(t) = Ah(t) + Bx(t)$, $y(t) = Ch(t)$;

**Interview questions**

1. Write the discretized SSM recurrence $h_t = \bar{A}h_{t-1} + \bar{B}x_t$ and derive the continuous-to-discrete step ($\bar{A} = e^{A\Delta}$, ZOH); what is the role of $\Delta$?
2. Training parallelism: why is a time-invariant SSM equivalent to a global convolution $y = \bar{K} * x$? After adding selectivity, how does the parallel scan (associative scan) restore parallel training?
3. What is the selective mechanism: why must $\Delta_t, B_t, C_t$ depend on the input $x_t$? How does it compare with LSTM gating (content-adaptive vs fixed parameters)?
4. Complexity comparison: self-attention is $O(L^2)$ in time and memory vs Mamba's $O(L)$; what does this mean concretely for $L = 100k$ long context?
5. Why are hybrid Transformer + Mamba architectures popular (Mamba blocks every few layers)? What are the potential weaknesses of pure Mamba?

<details>
<summary>中文题目</summary>

1. 写出 SSM 离散化递推 $h_t = \bar{A}h_{t-1} + \bar{B}x_t$ 并推导连续→离散($\bar{A} = e^{A\Delta}$ 的 ZOH 推导), $\Delta$ 起什么作用?
2. 训练并行化: 时不变 SSM 为什么等价于全局卷积 $y = \bar{K} * x$? 引入选择性后如何用并行扫描(associative scan)恢复并行训练?
3. 选择性机制是什么: $\Delta_t, B_t, C_t$ 为什么必须依赖输入 $x_t$? 它与 LSTM 门控的异同(内容感知 vs 固定参数)?
4. 复杂度对比: 自注意力 $O(L^2)$ 的时间与显存 vs Mamba 的 $O(L)$; 在 $L = 100k$ 长上下文下分别意味着什么?
5. 为什么流行的是 Transformer + Mamba 混合架构(每几层插入 Mamba 块)? 纯 Mamba 的潜在弱点是什么?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/ssm-mamba?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=ssm-mamba)

---

## 31. Training Debugging · 训练调试流程

> A systematic pipeline for fixing broken training. Step 1 — overfit a single batch: train on 1 (or a few) samples; loss should quickly drop to ~0, otherwise there is an implementation bug — inspect the data pipeline (label misalignment/normalization/shuffling),…

**Interview questions**

1. Steps and rationale of the single-batch overfit test? Which bugs does it isolate first?
2. Order of checks when training loss does not drop (data pipeline→forward→backward→optimizer)?
3. Typical loss-curve shapes for underfitting, overfitting, and wrong learning rates?
4. Common causes of NaN loss and how to debug them?
5. How do you fix high-train/low-val versus low-train/high-val situations?

<details>
<summary>中文题目</summary>

1. 单 batch 过拟合调试的步骤与原理? 它优先排查哪类 bug?
2. 训练 loss 不降时的排查顺序 (数据管线→前向→反向→优化器)?
3. 欠拟合、过拟合、学习率过小/过大各自的 loss 曲线形态?
4. loss 变 NaN 的常见原因与排查方法?
5. train 高 val 低与 train 低 val 高分别如何修复?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/training-debugging?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=training-debugging)

---

## 32. Transfer Learning · 迁移学习与微调

> Transfer learning initializes a target task with weights pretrained on a large dataset (e.g. ImageNet). The freezing strategy follows feature hierarchy: shallow layers capture generic low-level features (edges/textures/colors) while deep layers encode task-spe…

**Interview questions**

1. When to use feature extraction vs fine-tuning? What are the deciding factors?
2. Why are shallow features generic while deep features are task-specific?
3. Causes of catastrophic forgetting and how to mitigate it?
4. Why is fine-tuning's learning rate usually far smaller than from-scratch training?
5. What to do when the pretraining distribution differs significantly from the target?

<details>
<summary>中文题目</summary>

1. 什么情况下用特征提取、什么情况下用微调? 判断依据?
2. 为什么浅层特征通用、深层特征任务相关?
3. 灾难性遗忘的成因与缓解方法?
4. 微调的学习率为什么通常远小于从头训练?
5. 预训练数据与目标分布差异大时怎么办?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/transfer-learning?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=transfer-learning)

---

## 33. VAE & Reparameterization · VAE 与重参数化

> A VAE is a variational generative model trained on the evidence lower bound: $\log p(x) \ge \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \Vert p(z))$ — a reconstruction term plus a KL regularizer toward the prior.

**Interview questions**

1. Derive the VAE ELBO. Why is it a lower bound on $\log p(x)$?
2. What problem does the reparameterization trick solve? Why can't we sample straight from $q(z|x)$?
3. Core difference between VAE and AE? Why can't an AE generate new samples?
4. Why are VAE samples blurry? The mechanism of mode averaging?
5. How are the KL and reconstruction terms balanced? What are $\beta$-VAE and posterior collapse?

<details>
<summary>中文题目</summary>

1. 推导 VAE 的 ELBO? 为什么它是 $\log p(x)$ 的下界?
2. 重参数化技巧解决什么问题? 为什么不直接从 $q(z|x)$ 采样?
3. VAE 与 AE 的核心区别? 为什么 AE 不能生成新样本?
4. VAE 生成图像为什么模糊? 模式平均的机理?
5. KL 项与重构项如何平衡? 什么是 $\beta$-VAE 与 posterior collapse?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/vae?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vae)

---

## 34. Vanishing/Exploding Gradients · 梯度消失/爆炸

> Vanishing/exploding gradients are the core obstacle to training deep nets and RNNs, rooted in the chain product of Jacobians along depth: $\frac{\partial L}{\partial h_0}=\prod_{t=1}^{T}\frac{\partial h_t}{\partial h_{t-1}}$, whose norm is bounded by…

**Interview questions**

1. Explain vanishing/exploding gradients via $\prod_{t}\Vert J_t\Vert\le\gamma^T$ with the numeric example $0.9^{50}\approx0.005$?
2. Why do RNNs suffer vanishing gradients more than CNNs? How does the LSTM additive cell-state path help?
3. Which variances do Xavier and Kaiming preserve? Why does Kaiming introduce a factor of 2 for ReLU?
4. Why do residual connections $y=F(x)+x$ mitigate vanishing gradients? What does the identity-shortcut gradient highway mean?
5. Besides init and residuals, what other fixes exist (normalization / ReLU family / gradient clipping / LSTM gating)? Mechanisms?

<details>
<summary>中文题目</summary>

1. 用连乘机制 $\prod_{t}\Vert J_t\Vert\le\gamma^T$ 解释梯度消失/爆炸,并给出数字例子 $0.9^{50}\approx0.005$ 说明什么?
2. 为什么 RNN 比 CNN 更易梯度消失?LSTM 细胞状态的加性路径如何缓解?
3. Xavier 与 Kaiming 初始化分别保持什么方差?为什么 ReLU 下 Kaiming 要乘以 2?
4. 残差连接 $y=F(x)+x$ 为什么能缓解梯度消失?“恒等捷径梯度直通”的具体含义?
5. 除初始化与残差外还有哪些缓解手段(归一化/ReLU 族/梯度裁剪/LSTM 门控)?各自机制?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/vanishing-exploding?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vanishing-exploding)

---

## 35. ViT & Inductive Bias · ViT 与归纳偏置

> ViT cuts an image into $p \times p$ patches, flattens and linearly projects them into tokens (prepending a [CLS] token for classification plus position embeddings), then feeds them into a standard Transformer encoder;

**Interview questions**

1. Why does ViT need far more training data than CNNs? Answer from the inductive-bias perspective.
2. Hand-compute patchify: 224×224 input, patch 16 — how many tokens (including [CLS])?
3. Pros and cons of ViT vs CNN; at what data scale does ViT overtake?
4. What do position embeddings and the [CLS] token do in ViT? What happens without position encoding?
5. How do you make ViT work with limited data (distillation, augmentation, smaller patches, local attention/Swin)?

<details>
<summary>中文题目</summary>

1. 为什么 ViT 需要远多于 CNN 的训练数据? 从归纳偏置的角度回答。
2. 手算 patchify: 224×224 输入、patch=16 时得到多少个 token?(含 [CLS])
3. ViT 与 CNN 各有什么优缺点? 在什么数据规模下 ViT 开始反超?
4. ViT 中位置编码和 [CLS] token 的作用是什么? 去掉位置编码会怎样?
5. 数据有限时如何让 ViT 可用?(蒸馏、数据增强、减小 patch、局部注意力/Swin)

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/vit?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=vit)

---

## 36. Weight Decay & AdamW · 权重衰减与 AdamW

> L2 regularization adds $\frac{\lambda}{2}\Vert\theta\Vert_2^2$ to the loss, turning the update into $\theta \leftarrow \theta - \eta(\nabla L + \lambda\theta)$;

**Interview questions**

1. Derive the equivalence of L2 regularization and weight decay under plain SGD: write both updates $\theta \leftarrow \theta - \eta(\nabla L+\lambda\theta)$ and $\theta \leftarrow (1-\eta\lambda)\theta - \eta\nabla L$ and state the equivalence condition.
2. Why are L2 and weight decay inequivalent under Adam? What happens when $\lambda\theta$ is divided by $\sqrt{\hat{v}}+\epsilon$?
3. Write the decoupled AdamW update $\theta \leftarrow \theta - \eta\lambda\theta - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$ and explain why it generalizes better than Adam + L2.
4. Hyperparameter scale: what is typical for AdamW weight decay (0.01–0.1)? How does it relate to L2's common 5e-4?
5. Does weight decay drive weights all the way to zero? Why does it mostly constrain the norm rather than zero out parameters?

<details>
<summary>中文题目</summary>

1. 推导 L2 正则化与 weight decay 在朴素 SGD 下的等价性: 写出 $\theta \leftarrow \theta - \eta(\nabla L + \lambda\theta)$ 与 $\theta \leftarrow (1-\eta\lambda)\theta - \eta\nabla L$ 并说明等价条件?
2. 为什么 Adam 下 L2 与 weight decay 不等价? $\lambda\theta$ 项被 $\sqrt{\hat{v}}+\epsilon$ 除会造成什么后果?
3. 写出 AdamW 的解耦更新公式 $\theta \leftarrow \theta - \eta\lambda\theta - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$ 并解释为什么能提升泛化 (相对 Adam+L2)?
4. weight decay 的超参量级: AdamW 下典型取多少 (0.01~0.1)? 与 L2 常用的 5e-4 是什么关系?
5. weight decay 会把权重衰减到 0 吗? 为什么它主要约束范数而非清零参数?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/weight-decay?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=weight-decay)

---

## 37. Weight Initialization · 权重初始化

> Weight initialization chooses a distribution for the weights before training, aiming to keep the variance of forward activations and backward gradients preserved across layers (neither shrinking nor exploding), avoiding vanishing/exploding gradients.

**Interview questions**

1. Why can't all weights be initialized to 0 (or the same value)? Explain the symmetry problem: identical initialization → identical gradients → symmetry never breaks → equivalent to a single neuron.
2. Derive Xavier's forward condition: from $\text{Var}(y) = d\,\text{Var}(w)\,\text{Var}(x)$ and variance preservation get $\text{Var}(w) = 1/d_{\text{in}}$; why does the final formula average as $2/(d_{\text{in}}+d_{\text{out}})$ (compromise between $1/d_{\text{in}}$ forward and $1/d_{\text{out}}$ backward)?
3. Write Xavier and Kaiming formulas and their activation targets: tanh/sigmoid → Xavier ($2/(d_{\text{in}}+d_{\text{out}})$), ReLU → Kaiming ($2/d_{\text{in}}$); why (ReLU zeroes half → factor 2)?
4. Where does the uniform bound $\sqrt{6/(d_{\text{in}}+d_{\text{out}})}$ come from? Solve $a^2/3 = 2/(d_{\text{in}}+d_{\text{out}})$ for $U[-a,a]$.
5. Initialization vs vanishing/exploding gradients: how does the initial variance determine signal propagation (magnitudes of forward activations and backward gradients); why are initialization and BN complementary?

<details>
<summary>中文题目</summary>

1. 为什么不能把所有权重初始化为 0(或相同值)?写出对称性问题: 相同初始化 → 相同梯度 → 对称性永不破缺 → 等价于一个神经元。
2. 推导 Xavier 前向条件: 由 $\text{Var}(y) = d\,\text{Var}(w)\,\text{Var}(x)$ 令方差保持得 $\text{Var}(w) = 1/d_{\text{in}}$;为什么最终取 $2/(d_{\text{in}}+d_{\text{out}})$(前向 $1/d_{\text{in}}$ 与反向 $1/d_{\text{out}}$ 的折中)?
3. 写出 Xavier 与 Kaiming 的公式及适用激活: tanh/sigmoid 用 Xavier($2/(d_{\text{in}}+d_{\text{out}})$),ReLU 用 Kaiming($2/d_{\text{in}}$);原因是什么(ReLU 置零一半 → 因子 2)?
4. 均匀分布边界 $\sqrt{6/(d_{\text{in}}+d_{\text{out}})}$ 怎么来的?由 $U[-a,a]$ 方差 $a^2/3$ 等于目标方差 $2/(d_{\text{in}}+d_{\text{out}})$ 反解。
5. 初始化与梯度消失/爆炸: 深度网络中初始化方差如何决定信号传递(前向激活与反向梯度的量级);为什么说初始化与 BN 是互补手段?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/weight-init?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=weight-init)

---

## 38. WGAN & Wasserstein · WGAN 与 Wasserstein

> WGAN replaces the JS divergence with the Wasserstein distance (earth-mover, via the Kantorovich-Rubinstein duality): $W(P_r, P_g) = \sup_{\Vert f \Vert_L \le 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{z \sim p_z}[f(G(z))]$, requiring the critic $f$ to be 1…

**Interview questions**

1. State the definition of Wasserstein distance and its Kantorovich-Rubinstein dual form?
2. Why do JSD/KL gradients vanish for non-overlapping supports while Wasserstein does not?
3. How is the 1-Lipschitz constraint enforced? Trade-offs of weight clipping vs gradient penalty?
4. Why is the WGAN-GP penalty minimized when $\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2 = 1$?
5. Why does WGAN's critic omit sigmoid, and what does its loss value mean?

<details>
<summary>中文题目</summary>

1. 写出 Wasserstein 距离的定义与 Kantorovich-Rubinstein 对偶形式?
2. 为什么 JSD/KL 在分布不重叠时梯度消失, 而 Wasserstein 距离不会?
3. 1-Lipschitz 约束如何实现? 权重裁剪与梯度惩罚各有什么问题?
4. WGAN-GP 梯度惩罚项为什么在 $\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2 = 1$ 时最优?
5. 为什么 WGAN 的 critic 输出不加 sigmoid? WGAN 的 loss 数值有什么含义?

</details>

[Read the full answer, derivation & code →](https://talentme.airsota.com/resources/tech/roadmap/mindmap/DL/card/wgan?utm_source=github&utm_medium=question-bank&utm_campaign=vault&utm_content=wgan)

---
← Back to [the full question bank index](../QUESTION-BANK.md)
